import os.path as osp
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.vae import VAEAttributeTeacher  

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="path/to/model")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "MaPLe",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "maple_length": cfg.TRAINER.MAPLE.N_CTX,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]  # 512
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH

        assert cfg_imsize == clip_imsize

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print("MaPLe design: Multi-modal Prompt Learning")
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj = nn.Linear(ctx_dim, 768)
        self.proj.half()

        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text = nn.ParameterList(
            [nn.Parameter(torch.empty(n_ctx, 512)) for _ in range(self.compound_prompts_depth - 1)]
        )
        for p in self.compound_prompts_text:
            nn.init.normal_(p, std=0.02)

        single_layer = nn.Linear(ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts

    def construct_prompts(self, ctx, prefix, suffix):
        return torch.cat([prefix, ctx, suffix], dim=1)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = self.construct_prompts(ctx, self.token_prefix, self.token_suffix)

        visual_deep_prompts = []
        for layer, p in zip(self.compound_prompt_projections, self.compound_prompts_text):
            visual_deep_prompts.append(layer(p))

        return prompts, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts


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
    """Training-only fusion anchored on transferable shared semantics."""

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

        # Keep the conditioning vector anchored on shared semantics.
        cond = self.ln_out(s + delta)
        cond = self._l2n(cond)
        return cond


class GlobalAttrBankRefiner(nn.Module):
    """Attribute-bank refiner used by MaPLe + SpecPL.

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

        mcfg = cfg.TRAINER.MAPLE
        self.bank_size = int(getattr(mcfg, "ATTR_BANK_SIZE", 64))
        self.tau = float(getattr(mcfg, "ATTR_BANK_TAU", 0.07))

        # reuse ATTR_LAMBDA as semantic align weight
        self.lmbd_sem = float(getattr(mcfg, "ATTR_LAMBDA", 0.1))
        self.lmbd_g_f = float(getattr(mcfg, "ATTR_LAMBDA_G_F", 0.1))
        self.lmbd_g_cf = float(getattr(mcfg, "ATTR_LAMBDA_G_CF", 0.1))

        self.shared_source = str(getattr(mcfg, "SHARED_SOURCE", "raw_text"))

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

        # Frozen VAE teacher provides low/high spectral supervision.
        pretrained_id = getattr(mcfg, "VAE_PRETRAINED_ID", "REPA-E/e2e-qwenimage-vae")
        highpass_k = int(getattr(mcfg, "VAE_HIGHPASS_K", 7))
        lowpass_k = getattr(mcfg, "VAE_LOWPASS_K", None)
        use_var = bool(getattr(mcfg, "VAE_USE_VAR", True))

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
        # your teacher: forward(image) -> high, forward(image, return_bands=True) -> (low, high)
        low, high = self.teacher(image_clip_normed, return_bands=True)
        return self._l2n(low), self._l2n(high)

    @torch.no_grad()
    def init_attr_tokens(self, data_loader, device, max_batches=32, momentum=0.05):
        """Initialize frozen attr_tokens from teacher_low on a few base-train batches.
        This runs once before training (recommended before DataParallel wrapping).
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
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.attr_refiner = GlobalAttrBankRefiner(cfg, clip_model)

    def forward(self, image, label=None, return_log_dict=False):
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_text, deep_vision = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_text)

        if self.training:
            refined_text, ref_stats = self.attr_refiner.refine_text(text_features, return_stats=True)
        else:
            refined_text = self.attr_refiner.refine_text(text_features, return_stats=False)

        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_vision)

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
class MaPLeSpecPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC in ["fp32", "amp"]:
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
                if "VPT" in name:
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
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        # C2: optional one-shot init for frozen attr_tokens
        init_batches = int(getattr(cfg.TRAINER.MAPLE, "ATTR_INIT_BATCHES", 0))
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
        self.register_model("MaPLe_RouteC", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

        if torch.cuda.device_count() > 1:
            print(f"Multiple GPUs detected (n_gpus={torch.cuda.device_count()}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MAPLE.PREC

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

    # Keys that should be discarded regardless of checkpoint format
    # because they are class-name-dependent and rebuilt at construction time.
    _LOAD_DROP_EXACT = (
        "prompt_learner.token_prefix",
        "prompt_learner.token_suffix",
        "token_prefix",
        "token_suffix",
    )

    # Slim checkpoints (produced by `stat/slim_checkpoints.py`) only contain the
    # learnable + behavior-relevant tensors. The remaining model parameters
    # (image_encoder.*, text_encoder.*, attr_refiner.teacher.*, logit_scale)
    # come from the public CLIP / VAE checkpoints already loaded in
    # `build_model`, so they must NOT be re-loaded from a full checkpoint
    # either when a slim format is detected.
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
        if key in cls._SLIM_KEEP_EXACT:
            return True
        for prefix in cls._SLIM_KEEP_PREFIXES:
            if key.startswith(prefix):
                return True
        for sub in cls._SLIM_KEEP_SUBSTRINGS:
            if sub in key:
                return True
        return False

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at \"{model_path}\"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            ckpt_format = checkpoint.get("format", None)
            epoch_ckpt = checkpoint.get("epoch", None)

            for k in self._LOAD_DROP_EXACT:
                if k in state_dict:
                    del state_dict[k]

            is_slim = ckpt_format == "specpl_slim_v1" or all(
                self._is_slim_key(k) for k in state_dict.keys()
            )

            if not is_slim:
                # Old/full checkpoint: drop frozen CLIP / VAE / logit_scale so we
                # don't overwrite the freshly-loaded pretrained weights with the
                # (identical, but redundant) snapshot. This also makes the load
                # behavior match the slim path.
                state_dict = {k: v for k, v in state_dict.items() if self._is_slim_key(k)}

            msg_format = "slim" if is_slim else "full"
            msg_epoch = "" if epoch_ckpt is None else f", epoch={epoch_ckpt}"
            print(
                f'Loading weights to {name} from "{model_path}" '
                f'(format={msg_format}{msg_epoch}, kept_keys={len(state_dict)})'
            )

            missing, unexpected = self._models[name].load_state_dict(state_dict, strict=False)
            if unexpected:
                print(f"  warning: {len(unexpected)} unexpected key(s), e.g. {unexpected[:3]}")
