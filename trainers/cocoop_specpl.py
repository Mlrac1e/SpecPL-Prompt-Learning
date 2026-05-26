import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.vae import VAEAttributeTeacher  # returns low/high spectral bands for SpecPL

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "trainer": "CoCoOp",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
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
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(
            OrderedDict(
                [
                    ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
                    ("relu", nn.ReLU(inplace=True)),
                    ("linear2", nn.Linear(vis_dim // 16, ctx_dim)),
                ]
            )
        )

        if cfg.TRAINER.COCOOP.PREC == "fp16":
            self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)

        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)  # (B, n_cls, n_tkn, ctx_dim)
        return prompts


# =========================
# SpecPL modules: attribute refinement and granularity supervision.
# =========================

class GranuleModulator(nn.Module):
    """Train-only: cond -> FiLM(image_features). Inference unused."""
    def __init__(self, dim: int = 512):
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

    def forward(self, cond: torch.Tensor, img_feat: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, dim: int = 512):
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
    def _l2n(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    def forward(self, shared: torch.Tensor, indiv: torch.Tensor) -> torch.Tensor:
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
    """
    Attribute-bank refiner used by CoCoOp + SpecPL.

    The frozen codebook refines text features at inference. During training,
    low-frequency teacher features provide semantic supervision while
    high-frequency features drive granularity-aware modulation.
    """
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.text_dim = clip_model.ln_final.weight.shape[0]  # 512

        mcfg = cfg.TRAINER.COCOOP

        # token/codebook size (still use old names for backward compat)
        self.bank_size = int(getattr(mcfg, "ATTR_BANK_SIZE", 64))
        self.tau = float(getattr(mcfg, "ATTR_BANK_TAU", 0.07))

        # weights
        self.lmbd_sem = float(getattr(mcfg, "ATTR_LAMBDA_SEM", getattr(mcfg, "ATTR_LAMBDA", 0.1)))
        self.lmbd_g_f = float(getattr(mcfg, "ATTR_LAMBDA_G_F", 0.1))
        self.lmbd_g_cf = float(getattr(mcfg, "ATTR_LAMBDA_G_CF", 0.1))

        # shared source: raw_text | refined_text | image (note: raw/refined path uses label)
        self.shared_source = str(getattr(mcfg, "SHARED_SOURCE", "raw_text"))

        # C2: frozen attr tokens (buffers)
        tokens = torch.randn(self.bank_size, self.text_dim, dtype=torch.float32)
        tokens = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-6)
        self.register_buffer("attr_tokens", tokens, persistent=True)
        self.register_buffer("token_counts", torch.zeros(self.bank_size, dtype=torch.float32), persistent=True)
        self.register_buffer("token_steps", torch.zeros(1, dtype=torch.long), persistent=True)

        # online init/update (optional)
        self.token_init_steps = int(getattr(mcfg, "ATTR_TOKEN_INIT_STEPS", 200))
        self.token_momentum = float(getattr(mcfg, "ATTR_TOKEN_MOMENTUM", 0.1))
        self.token_soft_assign = bool(getattr(mcfg, "ATTR_TOKEN_SOFT_ASSIGN", True))

        # text refinement head (learnable)
        self.agg = nn.Sequential(
            nn.Linear(self.text_dim * 2, self.text_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.text_dim, self.text_dim, bias=True),
        )
        self.ln = nn.LayerNorm(self.text_dim)

        # C1: Prism dual-band teacher
        pretrained_id = getattr(mcfg, "VAE_PRETRAINED_ID", "REPA-E/e2e-qwenimage-vae")
        highpass_k = int(getattr(mcfg, "VAE_HIGHPASS_K", 7))
        lowpass_k = getattr(mcfg, "VAE_LOWPASS_K", None)
        self.teacher = VAEAttributeTeacher(
            pretrained_id=pretrained_id,
            highpass_k=highpass_k,
            lowpass_k=lowpass_k,
            out_dim=self.text_dim,
        )
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.fusion = SharedIndivFusion(dim=self.text_dim)
        self.granule_modulator = GranuleModulator(dim=self.text_dim)

    @staticmethod
    def _l2n(x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + 1e-6)

    # ----- C1 teacher -----

    def teacher_bands(self, image_clip_normed: torch.Tensor):
        """Return (low, high) normalized, float32."""
        with torch.no_grad():
            t_low, t_high = self.teacher(image_clip_normed, return_bands=True)
        return self._l2n(t_low.to(torch.float32)), self._l2n(t_high.to(torch.float32))

    # ----- C2 token update -----

    @torch.no_grad()
    def maybe_update_tokens(self, t_low: torch.Tensor):
        """
        Optional online initialization/update of frozen tokens (buffers):
        - uses teacher low-band attributes (label-free)
        - runs only for first token_init_steps steps
        """
        if self.token_init_steps <= 0:
            return
    
        step = int(self.token_steps.item())
        if step >= self.token_init_steps:
            return
    
        q = self._l2n(t_low.to(torch.float32))  # (B,D)
        bank = self._l2n(self.attr_tokens.to(torch.float32))  # (M,D)
    
        logits = (q @ bank.t()) / max(self.tau, 1e-6)  # (B,M)
    
        if self.token_soft_assign:
            w = logits.softmax(dim=-1)  # (B,M)
            denom = w.sum(dim=0) + 1e-6  # (M,)
            new = (w.t() @ q) / denom.unsqueeze(1)  # (M,D)
            has = (denom > 1e-3).to(torch.float32).unsqueeze(1)  # (M,1)
        else:
            idx = logits.argmax(dim=-1)  # (B,)
            new = torch.zeros_like(bank)
            cnt = torch.zeros(self.bank_size, device=q.device, dtype=torch.float32)
            new.index_add_(0, idx, q)
            cnt.index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
            denom = cnt + 1e-6
            new = new / denom.unsqueeze(1)
            has = (cnt > 0).to(torch.float32).unsqueeze(1)
    
        m = float(self.token_momentum)
        updated = bank * (1.0 - m) + new * m
        updated = bank * (1.0 - has) + updated * has  # keep old if no assignments
        updated = self._l2n(updated)
    
        # write back to buffers (same device as buffers)
        self.attr_tokens.copy_(updated)
    
        # IMPORTANT: keep denom on same device as token_counts
        denom_f = denom.detach().to(dtype=self.token_counts.dtype, device=self.token_counts.device)
        self.token_counts.add_(denom_f)
    
        self.token_steps.add_(1)


    # ----- retrieval/refine using frozen tokens -----

    def retrieve_attr(self, query: torch.Tensor, return_stats: bool = False):
        """
        query: (B,512) or (C,512)
        return:
          r: (B,512) float32 normalized
        """
        q = self._l2n(query.to(torch.float32))
        bank = self._l2n(self.attr_tokens.to(torch.float32))

        logits = (q @ bank.t()) / max(self.tau, 1e-6)
        w = logits.softmax(dim=-1)
        r = w @ bank
        r = self._l2n(r)

        if not return_stats:
            return r

        with torch.no_grad():
            ent = (-w * (w + 1e-12).log()).sum(dim=-1).mean()
            top1w = w.max(dim=-1).values.mean()
        stats = {"inst_retr_entropy": float(ent.item()), "inst_retr_top1w": float(top1w.item())}
        return r, stats

    def refine_text(self, text_features: torch.Tensor, return_stats: bool = False):
        """
        text_features: (C,512) fp16/fp32
        return: refined_text (C,512) same dtype as input
        """
        out_dtype = text_features.dtype

        q = self._l2n(text_features.to(torch.float32))
        bank = self._l2n(self.attr_tokens.to(torch.float32))

        logits = (q @ bank.t()) / max(self.tau, 1e-6)
        w = logits.softmax(dim=-1)  # (C,M)
        r = w @ bank                # (C,512)

        fused = torch.cat([q, r], dim=-1)
        delta = self.agg(fused)
        y = self.ln(q + delta)
        y = self._l2n(y).to(out_dtype)

        if not return_stats:
            return y

        with torch.no_grad():
            ent = (-w * (w + 1e-12).log()).sum(dim=-1).mean()
            top1w = w.max(dim=-1).values.mean()
            token_norm_mean = bank.norm(dim=-1).mean()
            filled_ratio = (self.token_counts > 0).float().mean()
            steps = float(self.token_steps.item())
        stats = {
            "retr_entropy": float(ent.item()),
            "retr_top1w": float(top1w.item()),
            "token_norm_mean": float(token_norm_mean.item()),
            "token_filled_ratio": float(filled_ratio.item()),
            "token_steps": float(steps),
        }
        return y, stats

    # Label-free semantic constraint from the low-frequency band.

    def loss_semantic(self, logits: torch.Tensor, text_features: torch.Tensor, t_low: torch.Tensor):
        """
        Label-free semantic constraint:
          p = softmax(logits.detach())
          t_exp = sum_c p_c * normalize(text_features_c)
          L = 1 - cos(t_exp, t_low)
        """
        p = logits.detach().to(torch.float32).softmax(dim=-1)  # (B,C)

        tf = text_features.to(torch.float32)
        if tf.dim() == 2:
            tf = tf.unsqueeze(0).expand(p.size(0), -1, -1)  # (B,C,D)
        tf = self._l2n(tf)

        t_exp = torch.einsum("bc,bcd->bd", p, tf)
        t_exp = self._l2n(t_exp)

        t_low = self._l2n(t_low.to(torch.float32))
        cos = (t_exp * t_low).sum(dim=-1)
        loss = 1.0 - cos
        return loss.mean(), cos.mean()

    # ----- shared factor builder (unchanged, may use label depending on source) -----

    def build_shared(self, text_features: torch.Tensor, refined_text: torch.Tensor,
                     image_feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        shared source: raw_text | refined_text | image
        text_features/refined_text can be (C,512) or (B,C,512)
        """
        src = self.shared_source.lower()

        if src == "image":
            s = image_feat.to(torch.float32)
            return self._l2n(s)

        if src == "raw_text":
            tf = text_features.to(torch.float32)
            if tf.dim() == 2:
                s = tf[labels]
            else:
                b = labels.size(0)
                s = tf[torch.arange(b, device=labels.device), labels]
            return self._l2n(s)

        if src == "refined_text":
            rt = refined_text.to(torch.float32)
            if rt.dim() == 2:
                s = rt[labels]
            else:
                b = labels.size(0)
                s = rt[torch.arange(b, device=labels.device), labels]
            return self._l2n(s)

        tf = text_features.to(torch.float32)
        if tf.dim() == 2:
            s = tf[labels]
        else:
            b = labels.size(0)
            s = tf[torch.arange(b, device=labels.device), labels]
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

    def forward(self, image, label=None, return_log_dict: bool = False):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        # CLIP image features (fp16/fp32 per cfg)
        image_features = self.image_encoder(image.type(self.dtype))
        image_n = image_features / image_features.norm(dim=-1, keepdim=True)

        # C1 teacher bands (train-only) + optional C2 token update
        if self.training:
            t_low, t_high = self.attr_refiner.teacher_bands(image)  # float32 normalized
            # self.attr_refiner.maybe_update_tokens(t_low)

        # instance-conditioned prompts
        prompts = self.prompt_learner(image_n)  # (B,C,T,D)

        refined_list = []
        text_list = []
        stats_list = []

        for pts_i in prompts:  # pts_i: (C,T,D)
            tf_i = self.text_encoder(pts_i, tokenized_prompts)  # (C,512)
            if self.training:
                rt_i, st_i = self.attr_refiner.refine_text(tf_i, return_stats=True)
                stats_list.append(st_i)
            else:
                rt_i = self.attr_refiner.refine_text(tf_i, return_stats=False)

            refined_list.append(rt_i)
            text_list.append(tf_i)

        refined_text = torch.stack(refined_list, dim=0)  # (B,C,512)
        text_features = torch.stack(text_list, dim=0)    # (B,C,512)

        text_n = refined_text / refined_text.norm(dim=-1, keepdim=True)
        logits = logit_scale * torch.einsum("bd,bcd->bc", image_n, text_n)  # (B,C)

        # if not self.training:
        #     return logits
        if not self.training:
            alpha = 0.5  # Blend raw and refined text features at evaluation.
            mix = (1 - alpha) * text_features + alpha * refined_text
            mix = mix.to(torch.float32)
            mix = mix / (mix.norm(dim=-1, keepdim=True) + 1e-6)
        
            logits = logit_scale.to(torch.float32) * torch.einsum("bd,bcd->bc", image_n.to(torch.float32), mix)
            return logits

        # ===== losses =====
        loss_cls = F.cross_entropy(logits, label)

        # Label-free semantic constraint from the low-frequency band.
        loss_sem, cos_sem = self.attr_refiner.loss_semantic(logits, text_features, t_low)

        # shared factor (may use label depending on shared_source)
        shared = self.attr_refiner.build_shared(
            text_features=text_features,
            refined_text=refined_text,
            image_feat=image_n,
            labels=label,
        )  # (B,512) float32 normalized

        # Fuse shared semantics with high-frequency details for granule guidance.
        cond_f = self.attr_refiner.fusion(shared, t_high)  # (B,512) float32

        # granule losses (float32)
        img_f32 = image_n.to(torch.float32)
        txt_f32 = text_n.to(torch.float32)
        ls = logit_scale.to(torch.float32)

        # factual granule
        img_g_f = self.attr_refiner.granule_modulator(cond_f, img_f32)
        logits_g_f = ls * torch.einsum("bd,bcd->bc", img_g_f, txt_f32)
        loss_g_f = F.cross_entropy(logits_g_f, label)

        # counterfactual granule (permute high-band + labels)
        perm = torch.randperm(t_high.size(0), device=t_high.device)
        t_high_cf = t_high[perm]
        label_cf = label[perm]
        cond_cf = self.attr_refiner.fusion(shared, t_high_cf)

        img_g_cf = self.attr_refiner.granule_modulator(cond_cf, img_f32)
        logits_g_cf = ls * torch.einsum("bd,bcd->bc", img_g_cf, txt_f32)
        loss_g_cf = F.cross_entropy(logits_g_cf, label_cf)

        loss = (
            loss_cls
            + self.attr_refiner.lmbd_sem * loss_sem
            + self.attr_refiner.lmbd_g_f * loss_g_f
            + self.attr_refiner.lmbd_g_cf * loss_g_cf
        )

        if not return_log_dict:
            return loss

        # aggregate retrieval stats across batch (mean)
        ref_stats = {}
        if len(stats_list) > 0:
            keys = stats_list[0].keys()
            for k in keys:
                ref_stats[k] = float(sum(d[k] for d in stats_list) / len(stats_list))

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean()

            cos_s_h = (shared * t_high).sum(dim=-1).mean()
            cos_c_h = (cond_f * t_high).sum(dim=-1).mean()

        log_dict = {
            "loss": float(loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_sem": float(loss_sem.item()),
            "loss_g_f": float(loss_g_f.item()),
            "loss_g_cf": float(loss_g_cf.item()),
            "cos_sem": float(cos_sem.item()),
            "cos_shared_high": float(cos_s_h.item()),
            "cos_cond_high": float(cos_c_h.item()),
            "acc": float(acc.item()),
            "shared_source": 0.0,  # placeholder
        }
        log_dict.update(ref_stats)
        return loss, log_dict


@TRAINER_REGISTRY.register()
class CoCoOpSpecPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        # trainable keywords (C2 removes attr_bank)
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

        # teacher always frozen
        for name, param in self.model.named_parameters():
            if "attr_refiner.teacher" in name:
                param.requires_grad_(False)

        enabled = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CoCoOp_RouteC", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
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

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            # ignore fixed token buffers
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(f'Loading weights to {name} from "{model_path}"')
            self._models[name].load_state_dict(state_dict, strict=False)
