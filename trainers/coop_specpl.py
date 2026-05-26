import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.vae import VAEAttributeTeacher  # returns low/high spectral bands for SpecPL

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="path/to/model")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'CoOp',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    # for name, param in model.state_dict().items():
    #     print(name, param.shape)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts



class GranuleModulator(nn.Module):
    """Training-only: cond -> FiLM(image_features)."""

    def __init__(self, dim=512):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2, bias=True),
            nn.SiLU(),
            nn.Linear(dim * 2, dim * 2, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=1e-3)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=1e-3)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, cond, img_feat):
        cond = cond.to(torch.float32)
        img_feat = img_feat.to(torch.float32)

        h = self.ln(cond)
        s, b = self.mlp(h).chunk(2, dim=-1)
        s = torch.tanh(s)
        out = (1.0 + s) * img_feat + b
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return out


class SharedIndivFusion(nn.Module):
    """Training-only: fuse(shared, indiv) -> cond.

    anchor on shared (shared provides transferable semantics; indiv only contributes residual granules).
    """

    def __init__(self, dim=512):
        super().__init__()
        self.ln_s = nn.LayerNorm(dim)
        self.ln_a = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.ln_out = nn.LayerNorm(dim)

        nn.init.normal_(self.mlp[0].weight, std=1e-3)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=1e-3)
        nn.init.zeros_(self.mlp[2].bias)

    @staticmethod
    def _l2n(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    def forward(self, shared, indiv):
        shared = shared.to(torch.float32)
        indiv = indiv.to(torch.float32)

        s = self.ln_s(shared)
        a = self.ln_a(indiv)
        h = torch.cat([s, a], dim=-1)
        delta = self.mlp(h)

        # Anchor the conditioning vector on transferable shared semantics.
        cond = self.ln_out(s + delta)
        cond = self._l2n(cond)
        return cond


class GlobalAttrBankRefiner(nn.Module):
    """Attribute-bank refiner used by CoOp + SpecPL.

    Inference:
      text -> retrieve attr_tokens -> refine text (no VAE).
    Training:
      teacher_low -> label-free semantic constraint
      teacher_high -> granule modulation (factual/counterfactual)
    """

    def __init__(self, cfg, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.text_dim = clip_model.ln_final.weight.shape[0]  # 512

        ccfg = cfg.TRAINER.COOP
        self.bank_size = int(getattr(ccfg, "ATTR_BANK_SIZE", 64))
        self.tau = float(getattr(ccfg, "ATTR_BANK_TAU", 0.07))

        # reuse ATTR_LAMBDA as semantic align weight
        self.lmbd_sem = float(getattr(ccfg, "ATTR_LAMBDA", 0.1))
        self.lmbd_g_f = float(getattr(ccfg, "ATTR_LAMBDA_G_F", 0.1))
        self.lmbd_g_cf = float(getattr(ccfg, "ATTR_LAMBDA_G_CF", 0.1))

        self.shared_source = str(getattr(ccfg, "SHARED_SOURCE", "raw_text"))

        # Frozen attribute tokens act as a compact retrieval codebook.
        tokens = torch.randn(self.bank_size, self.text_dim, dtype=torch.float32)
        tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        self.register_buffer("attr_tokens", tokens, persistent=True)
        self.register_buffer("attr_token_counts", torch.zeros(self.bank_size, dtype=torch.long), persistent=True)

        self.agg = nn.Sequential(
            nn.Linear(self.text_dim * 2, self.text_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim, bias=True),
        )
        self.ln = nn.LayerNorm(self.text_dim)

        # C1: dual-band teacher (VAEAttributeTeacher)
        pretrained_id = getattr(ccfg, "VAE_PRETRAINED_ID", "REPA-E/e2e-qwenimage-vae")
        highpass_k = int(getattr(ccfg, "VAE_HIGHPASS_K", 7))
        lowpass_k = getattr(ccfg, "VAE_LOWPASS_K", None)
        use_var = bool(getattr(ccfg, "VAE_USE_VAR", True))

        # teacher __init__ does NOT accept return_bands; forward does.
        self.teacher = VAEAttributeTeacher(
            pretrained_id=pretrained_id,
            highpass_k=highpass_k,
            out_dim=self.text_dim,
            use_var=use_var,
            lowpass_k=lowpass_k,
        )
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.fusion = SharedIndivFusion(dim=self.text_dim)
        self.granule_modulator = GranuleModulator(dim=self.text_dim)

    @staticmethod
    def _l2n(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    @torch.no_grad()
    def teacher_bands(self, image_clip_normed):
        # forward(image, return_bands=True) -> (low, high)
        low, high = self.teacher(image_clip_normed, return_bands=True)
        return self._l2n(low), self._l2n(high)

    @torch.no_grad()
    def init_attr_tokens(self, data_loader, device, max_batches=32, momentum=0.05):
        """Initialize frozen attr_tokens from teacher_low on a few base-train batches.
        Run once before training (recommended before DataParallel wrapping).
        """
        self.eval()
        self.to(device)

        filled = int((self.attr_token_counts > 0).sum().item())
        if filled >= self.bank_size:
            return

        for bidx, batch in enumerate(data_loader):
            if bidx >= int(max_batches):
                break

            if isinstance(batch, dict) and "img" in batch:
                imgs = batch["img"]
            elif isinstance(batch, (list, tuple)) and len(batch) > 0:
                imgs = batch[0]
            else:
                continue

            imgs = imgs.to(device)
            low, _ = self.teacher_bands(imgs)  # (B,512)

            for i in range(low.size(0)):
                f = low[i]

                if filled < self.bank_size:
                    self.attr_tokens[filled].copy_(f)
                    self.attr_token_counts[filled] += 1
                    filled += 1
                    continue

                sims = torch.matmul(self.attr_tokens, f)  # cosine since normalized
                idx = int(torch.argmax(sims).item())
                updated = (1.0 - float(momentum)) * self.attr_tokens[idx] + float(momentum) * f
                self.attr_tokens[idx].copy_(self._l2n(updated))
                self.attr_token_counts[idx] += 1

        self.attr_tokens.copy_(self._l2n(self.attr_tokens))

    def refine_text(self, text_features, return_stats=False):
        """Text refinement via retrieval over frozen attr_tokens."""
        out_dtype = text_features.dtype

        q = self._l2n(text_features.to(torch.float32))
        bank = self._l2n(self.attr_tokens)

        logits = (q @ bank.t()) / max(self.tau, 1e-6)
        w = logits.softmax(dim=-1)  # (C,M)
        r = w @ bank

        fused = torch.cat([q, r], dim=-1)
        delta = self.agg(fused)
        y = self.ln(q + delta)
        y = self._l2n(y).to(out_dtype)

        if not return_stats:
            return y

        with torch.no_grad():
            ent = (-w * (w + 1e-12).log()).sum(dim=-1).mean()
            top1w = w.max(dim=-1).values.mean()
            filled_ratio = float((self.attr_token_counts > 0).float().mean().item())
            cnt_min = float(self.attr_token_counts.min().item())
            cnt_mean = float(self.attr_token_counts.float().mean().item())

        stats = {
            "retr_entropy": float(ent.item()),
            "retr_top1w": float(top1w.item()),
            "token_filled_ratio": filled_ratio,
            "token_count_min": cnt_min,
            "token_count_mean": cnt_mean,
        }
        return y, stats

    def build_shared(self, text_features, refined_text, image_feat, labels):
        """Shared factor (float32 normalized)."""
        src = self.shared_source.lower()

        if src == "raw_text":
            s = text_features.to(torch.float32)[labels]
            return self._l2n(s)

        if src == "refined_text":
            s = refined_text.to(torch.float32)[labels]
            return self._l2n(s)

        if src == "image":
            s = image_feat.to(torch.float32)
            return self._l2n(s)

        s = text_features.to(torch.float32)[labels]
        return self._l2n(s)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.attr_refiner = GlobalAttrBankRefiner(cfg, clip_model)

    def forward(self, image, label=None, return_log_dict=False):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)

        if self.training:
            refined_text, ref_stats = self.attr_refiner.refine_text(text_features, return_stats=True)
        else:
            refined_text = self.attr_refiner.refine_text(text_features, return_stats=False)

        image_n = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        text_n = refined_text / (refined_text.norm(dim=-1, keepdim=True) + 1e-6)
        logits = logit_scale * image_n @ text_n.t()

        if not self.training:
            return logits

        # ===== losses =====
        loss_cls = F.cross_entropy(logits, label)

        # C1: teacher low/high
        t_low, t_high = self.attr_refiner.teacher_bands(image)  # float32 normalized

        # Label-free semantic alignment between teacher_low and expected text semantics.
        with torch.no_grad():
            p = logits.detach().to(torch.float32).softmax(dim=-1)  # (B,C)

        txt_sem = text_features.to(torch.float32)
        txt_sem = txt_sem / (txt_sem.norm(dim=-1, keepdim=True) + 1e-6)  # (C,512)
        t_exp = p @ txt_sem
        t_exp = t_exp / (t_exp.norm(dim=-1, keepdim=True) + 1e-6)

        cos_sem = (t_exp * t_low).sum(dim=-1)
        loss_sem = (1.0 - cos_sem).mean()

        shared = self.attr_refiner.build_shared(
            text_features=text_features,
            refined_text=refined_text,
            image_feat=image_n,
            labels=label,
        )

        # Fuse shared semantics with high-frequency details for granule guidance.
        cond_f = self.attr_refiner.fusion(shared, t_high)

        img_f32 = image_n.to(torch.float32)
        txt_f32 = text_n.to(torch.float32)
        ls = logit_scale.to(torch.float32)

        # factual granule
        img_g_f = self.attr_refiner.granule_modulator(cond_f, img_f32)
        logits_g_f = ls * (img_g_f @ txt_f32.t())
        loss_g_f = F.cross_entropy(logits_g_f, label)

        # counterfactual granule
        perm = torch.randperm(t_high.size(0), device=t_high.device)
        t_high_cf = t_high[perm]
        label_cf = label[perm]
        cond_cf = self.attr_refiner.fusion(shared, t_high_cf)

        img_g_cf = self.attr_refiner.granule_modulator(cond_cf, img_f32)
        logits_g_cf = ls * (img_g_cf @ txt_f32.t())
        loss_g_cf = F.cross_entropy(logits_g_cf, label_cf)

        loss = (
            loss_cls
            + self.attr_refiner.lmbd_sem * loss_sem
            + self.attr_refiner.lmbd_g_f * loss_g_f
            + self.attr_refiner.lmbd_g_cf * loss_g_cf
        )

        if not return_log_dict:
            return loss

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean()
            cos_s_h = (shared * t_high).sum(dim=-1).mean()
            cos_c_h = (cond_f * t_high).sum(dim=-1).mean()

        log_dict = {
            "loss": float(loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_sem_align": float(loss_sem.item()),
            "loss_g_f": float(loss_g_f.item()),
            "loss_g_cf": float(loss_g_cf.item()),
            "cos_sem": float(cos_sem.mean().item()),
            "cos_shared_high": float(cos_s_h.item()),
            "cos_cond_high": float(cos_c_h.item()),
            "acc": float(acc.item()),
            "shared_source": 0.0,
        }
        log_dict.update(ref_stats)
        return loss, log_dict

@TRAINER_REGISTRY.register()
class CoOpSpecPL(TrainerX):
    """Context Optimization (CoOp) + SpecPL.

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    _SLIM_FORMAT = "specpl_slim_v1"
    _LOAD_DROP_EXACT = (
        "prompt_learner.token_prefix",
        "prompt_learner.token_suffix",
        "token_prefix",
        "token_suffix",
    )
    _SLIM_KEEP_PREFIXES = (
        "prompt_learner.",
        "attr_refiner.agg.",
        "attr_refiner.ln.",
        "attr_refiner.fusion.",
        "attr_refiner.granule_modulator.",
    )
    _SLIM_KEEP_EXACT = (
        "attr_refiner.attr_tokens",
        "attr_refiner.attr_token_counts",
    )
    _SLIM_KEEP_SUBSTRINGS = ("VPT",)

    @classmethod
    def _is_slim_key(cls, key):
        if key in cls._LOAD_DROP_EXACT:
            return False
        if key in cls._SLIM_KEEP_EXACT:
            return True
        for prefix in cls._SLIM_KEEP_PREFIXES:
            if key.startswith(prefix):
                return True
        for sub in cls._SLIM_KEEP_SUBSTRINGS:
            if sub in key:
                return True
        return False

    @classmethod
    def _drop_load_only_keys(cls, state_dict):
        return {k: v for k, v in state_dict.items() if k not in cls._LOAD_DROP_EXACT}

    @classmethod
    def _is_slim_checkpoint(cls, state_dict, ckpt_format):
        if ckpt_format == cls._SLIM_FORMAT:
            return True
        return all(cls._is_slim_key(k) for k in state_dict.keys())

    @staticmethod
    def _resolve_checkpoint_path(directory, registered_name, model_file):
        candidates = [
            osp.join(directory, registered_name, model_file),
            osp.join(directory, "CoOp_RouteC", model_file),
            osp.join(directory, "prompt_learner", model_file),
        ]
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if osp.exists(path):
                return path
        return candidates[0]

    @staticmethod
    def _resolve_resume_dir(directory, registered_name):
        candidates = [
            osp.join(directory, registered_name),
            osp.join(directory, "CoOp_RouteC"),
            osp.join(directory, "prompt_learner"),
        ]
        seen = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            if osp.isdir(path):
                return path
        return candidates[0]

    @staticmethod
    def _find_latest_checkpoint(model_dir):
        pointer = osp.join(model_dir, "checkpoint")
        if osp.exists(pointer):
            with open(pointer, "r") as f:
                model_name = f.readline().strip()
            if model_name:
                ckpt_path = osp.join(model_dir, model_name)
                if osp.exists(ckpt_path):
                    return ckpt_path

        if not osp.isdir(model_dir):
            return None

        model_files = [f for f in os.listdir(model_dir) if f.startswith("model.pth.tar-")]
        if model_files:
            def parse_epoch(filename):
                try:
                    return int(filename.rsplit("-", 1)[-1])
                except ValueError:
                    return -1

            latest = max(model_files, key=parse_epoch)
            return osp.join(model_dir, latest)

        best_path = osp.join(model_dir, "model-best.pth.tar")
        if osp.exists(best_path):
            return best_path

        return None

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        trainable_keywords = [
            "prompt_learner",
            "attr_refiner.agg",
            "attr_refiner.ln",
            "attr_refiner.fusion",
            "attr_refiner.granule_modulator",
        ]

        for name, param in self.model.named_parameters():
            if any(k in name for k in trainable_keywords):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # safety: keep teacher frozen
        for name, param in self.model.named_parameters():
            if "attr_refiner.teacher" in name:
                param.requires_grad_(False)

        enabled = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # C2: optional one-shot init for frozen attr_tokens
        init_batches = int(getattr(cfg.TRAINER.COOP, "ATTR_INIT_BATCHES", 0))
        if init_batches > 0:
            loader = getattr(self, "train_loader_x", None)
            if loader is None:
                loader = getattr(self, "train_loader", None)
            if loader is not None:
                print(f"Initializing attr_tokens with VAE teacher_low (batches={init_batches})")
                self.model.attr_refiner.init_attr_tokens(loader, device=self.device, max_batches=init_batches)
            else:
                print("Warning: train loader not found; skip attr_tokens init")

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CoOp_RouteC", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                out = model(image, label, return_log_dict=True)
            loss, log_dict = out
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            out = model(image, label, return_log_dict=True)
            loss, log_dict = out
            optim.zero_grad()
            loss.backward()
            optim.step()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return log_dict

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        # Mirror `stat/slim_checkpoints.py` exactly: keep only state_dict (slim),
        # format, epoch, val_result. The optimizer / scheduler from Dassl carry
        # references to the entire frozen model (Dassl's ConstantWarmupScheduler
        # pickles its `successor.optimizer`, whose `param_groups['params']` are
        # the full `model.parameters()`) so persisting them blows the checkpoint
        # back up to ~800 MB. Resume-from-mid-epoch is intentionally not
        # supported by the slim format.
        names = self.get_model_names()

        for name in names:
            full_sd = self._models[name].state_dict()
            slim_sd = {
                k: v for k, v in full_sd.items()
                if self._is_slim_key(k)
            }

            save_checkpoint(
                {
                    "state_dict": slim_sd,
                    "epoch": epoch + 1,
                    "val_result": val_result,
                    "format": self._SLIM_FORMAT,
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        model_dirs = {}

        for name in names:
            model_dir = self._resolve_resume_dir(directory, name)
            if not osp.exists(model_dir):
                print("No checkpoint found, train from scratch")
                return 0
            model_dirs[name] = model_dir

        print(f"Found checkpoint at {directory} (will resume training)")

        start_epoch = 0
        for name in names:
            ckpt_path = self._find_latest_checkpoint(model_dirs[name])
            if ckpt_path is None:
                print(f'No checkpoint file found under "{model_dirs[name]}", train from scratch')
                return 0

            checkpoint = load_checkpoint(ckpt_path)
            state_dict = self._drop_load_only_keys(checkpoint["state_dict"])
            ckpt_format = checkpoint.get("format", None)

            if "prompt_learner.ctx" not in state_dict and "ctx" in state_dict:
                state_dict = {("prompt_learner." + k): v for k, v in state_dict.items()}

            is_slim = self._is_slim_checkpoint(state_dict, ckpt_format)
            if not is_slim:
                state_dict = {k: v for k, v in state_dict.items() if self._is_slim_key(k)}

            epoch_ckpt = int(checkpoint.get("epoch", 0) or 0)
            print(
                f'Resuming {name} from "{ckpt_path}" '
                f"(format={'slim' if is_slim else 'full'}, epoch={epoch_ckpt}, kept_keys={len(state_dict)})"
            )

            self._models[name].load_state_dict(state_dict, strict=False)

            # Slim checkpoints intentionally do not persist optimizer/scheduler
            # (see save_model). Old "fat" checkpoints from output_specpl/ still
            # have them, so honour them when present for backward compatibility.
            if self._optims[name] is not None and checkpoint.get("optimizer") is not None:
                self._optims[name].load_state_dict(checkpoint["optimizer"])
            if self._scheds[name] is not None and checkpoint.get("scheduler") is not None:
                self._scheds[name].load_state_dict(checkpoint["scheduler"])

            start_epoch = max(start_epoch, epoch_ckpt)

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = self._resolve_checkpoint_path(directory, name, model_file)
            if not osp.exists(model_path):
                tried = [
                    osp.join(directory, name, model_file),
                    osp.join(directory, "CoOp_RouteC", model_file),
                    osp.join(directory, "prompt_learner", model_file),
                ]
                msg = "Model not found. Tried:\n  " + "\n  ".join(tried)
                raise FileNotFoundError(msg)

            checkpoint = load_checkpoint(model_path)
            state_dict = self._drop_load_only_keys(checkpoint["state_dict"])
            epoch_ = checkpoint.get("epoch", None)
            ckpt_format = checkpoint.get("format", None)

            # If this is a prompt_learner-only checkpoint, prefix keys
            if "prompt_learner.ctx" not in state_dict and "ctx" in state_dict:
                state_dict = {("prompt_learner." + k): v for k, v in state_dict.items()}

            is_slim = self._is_slim_checkpoint(state_dict, ckpt_format)
            if not is_slim:
                state_dict = {k: v for k, v in state_dict.items() if self._is_slim_key(k)}

            msg_epoch = "" if epoch_ is None else f", epoch={epoch_}"
            print(
                f'Loading weights to {name} from "{model_path}" '
                f"(format={'slim' if is_slim else 'full'}{msg_epoch}, kept_keys={len(state_dict)})"
            )

            self._models[name].load_state_dict(state_dict, strict=False)
