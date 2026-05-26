import os
import os.path as osp
import sys
import time
import resource
import torch

try:
    import psutil
except ImportError:
    psutil = None
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.modules.loss import _Loss

from tqdm import tqdm
import copy

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, save_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.vae import VAEAttributeTeacher  # returns low/high spectral bands for SpecPL

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


def load_clip_to_cpu(cfg, model_name="CLIP"):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root="path/to/model")

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "model": model_name,
        "rep_tokens_layers": cfg.TRAINER.MMRL.REP_LAYERS,
        "n_rep_tokens": cfg.TRAINER.MMRL.N_REP_TOKENS,
    }
    model = clip.build_model_MMRL(state_dict or model.state_dict(), design_details)
    return model


class TextEncoder_MMRL(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_rep_tokens_text):
        n_rep_tokens = compound_rep_tokens_text[0].shape[0]
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        eot_index = tokenized_prompts.argmax(dim=-1)
        combined = [x, compound_rep_tokens_text, 0, eot_index]
        outputs = self.transformer(combined)

        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), eot_index + n_rep_tokens] @ self.text_projection
        return x


class TextEncoder_CLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        outputs = self.transformer(x)
        x = outputs
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def _get_text_base_features_zero_shot(cfg, classnames, clip_model, text_encoder):
    device = next(text_encoder.parameters()).device
    dataset = cfg.DATASET.NAME
    template = CUSTOM_TEMPLATES[dataset]

    text_encoder = text_encoder.cuda()

    with torch.no_grad():
        tokenized_prompts = []
        for text in tqdm(classnames, desc="Extracting text features"):
            tokens = clip.tokenize(template.format(text.replace("_", " ")))
            tokens = tokens.to(device)
            tokenized_prompts.append(tokens)
        tokenized_prompts = torch.cat(tokenized_prompts)

        embeddings = clip_model.token_embedding(tokenized_prompts).type(clip_model.dtype)
        outputs = text_encoder(embeddings.cuda(), tokenized_prompts.cuda())
        text_embeddings = outputs

    text_encoder = text_encoder.to(device)
    return text_embeddings


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MultiModalRepresentationLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        n_rep_tokens = cfg.TRAINER.MMRL.N_REP_TOKENS
        self.dtype = clip_model.dtype

        text_dim = clip_model.ln_final.weight.shape[0]
        visual_dim = clip_model.visual.ln_post.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        rep_dim = cfg.TRAINER.MMRL.REP_DIM

        self.rep_layers_length = len(cfg.TRAINER.MMRL.REP_LAYERS)
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        dataset = cfg.DATASET.NAME
        template = CUSTOM_TEMPLATES[dataset]

        tokenized_prompts = []
        for text in classnames:
            tokens = clip.tokenize(template.format(text.replace("_", " ")))
            tokenized_prompts.append(tokens)
        self.tokenized_prompts = torch.cat(tokenized_prompts)

        with torch.no_grad():
            self.prompt_embeddings = clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

        self.compound_rep_tokens = nn.Parameter(torch.empty(n_rep_tokens, rep_dim))
        nn.init.normal_(self.compound_rep_tokens, std=0.02)

        single_layer_r2v = nn.Linear(rep_dim, visual_dim)
        single_layer_r2t = nn.Linear(rep_dim, text_dim)

        self.compound_rep_tokens_r2vproj = _get_clones(single_layer_r2v, self.rep_layers_length)
        self.compound_rep_tokens_r2tproj = _get_clones(single_layer_r2t, self.rep_layers_length)

    def forward(self):
        compound_rep_tokens_visual = []
        compound_rep_tokens_text = []

        for index in range(self.rep_layers_length):
            rep_tokens = self.compound_rep_tokens
            rep_mapped_to_text = self.compound_rep_tokens_r2tproj[index](rep_tokens)
            rep_mapped_to_visual = self.compound_rep_tokens_r2vproj[index](rep_tokens)
            compound_rep_tokens_text.append(rep_mapped_to_text.type(self.dtype))
            compound_rep_tokens_visual.append(rep_mapped_to_visual.type(self.dtype))

        return compound_rep_tokens_text, compound_rep_tokens_visual


# -------------------------------
# SpecPL attribute refinement and granularity modules.
# -------------------------------

class GranuleModulator(nn.Module):
    """Training-only: cond -> FiLM(image_feature)."""

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
    """Fuse transferable shared semantics with residual high-frequency details."""

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

        # Anchor the conditioning vector on shared semantics.
        cond = self.ln_out(s + delta)
        return self._l2n(cond)


class GlobalAttrTokenRefiner(nn.Module):
    """Attribute-token refiner used by MMRL + SpecPL.

    Frozen attribute tokens refine text features at inference. During training,
    the VAE teacher provides low-frequency semantic supervision and
    high-frequency granularity guidance.
    """

    def __init__(self, cfg, clip_model):
        super().__init__()
        self.dtype = clip_model.dtype
        self.text_dim = clip_model.ln_final.weight.shape[0]

        mcfg = cfg.TRAINER.MMRL
        self.bank_size = int(getattr(mcfg, "ATTR_BANK_SIZE", 64))
        self.tau = float(getattr(mcfg, "ATTR_BANK_TAU", 0.07))

        # Reuse the same loss-weight names as the other SpecPL trainers.
        self.lmbd_sem = float(getattr(mcfg, "ATTR_LAMBDA", 0.1))
        self.lmbd_g_f = float(getattr(mcfg, "ATTR_LAMBDA_G_F", 0.1))
        self.lmbd_g_cf = float(getattr(mcfg, "ATTR_LAMBDA_G_CF", 0.1))

        self.shared_source = str(getattr(mcfg, "SHARED_SOURCE", "raw_text"))

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

        # Frozen VAE teacher used as the spatial-spectral proxy.
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
        low, high = self.teacher(image_clip_normed, return_bands=True)
        return self._l2n(low), self._l2n(high)

    @torch.no_grad()
    def init_attr_tokens(self, data_loader, device, max_batches=32, momentum=0.05):
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
            low, _ = self.teacher_bands(imgs)

            for i in range(low.size(0)):
                f = low[i]
                if filled < self.bank_size:
                    self.attr_tokens[filled].copy_(f)
                    self.attr_token_counts[filled] += 1
                    filled += 1
                    continue

                sims = torch.matmul(self.attr_tokens, f)
                idx = int(torch.argmax(sims).item())
                updated = (1.0 - float(momentum)) * self.attr_tokens[idx] + float(momentum) * f
                self.attr_tokens[idx].copy_(self._l2n(updated))
                self.attr_token_counts[idx] += 1

        self.attr_tokens.copy_(self._l2n(self.attr_tokens))

    def refine_text(self, text_features, return_stats=False):
        out_dtype = text_features.dtype

        q = self._l2n(text_features.to(torch.float32))
        bank = self._l2n(self.attr_tokens)

        logits = (q @ bank.t()) / max(self.tau, 1e-6)
        w = logits.softmax(dim=-1)
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

        stats = {
            "retr_entropy": float(ent.item()),
            "retr_top1w": float(top1w.item()),
            "token_filled_ratio": float(filled_ratio),
        }
        return y, stats

    def build_shared(self, text_features, refined_text, image_feat, labels):
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

        s = refined_text.to(torch.float32)[labels]
        return self._l2n(s)

class CustomCLIP(nn.Module):
    """
    MMRL + SpecPL keeps separate raw and refined text branches.

    The raw branch preserves transferable semantics, while the refined branch
    carries SpecPL's attribute-bank signal for the representation/fusion path.
    """

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.alpha = cfg.TRAINER.MMRL.ALPHA
        self.classnames = classnames

        self.representation_learner = MultiModalRepresentationLearner(cfg, classnames, clip_model).type(
            clip_model.dtype
        )
        self.tokenized_prompts = self.representation_learner.tokenized_prompts
        self.register_buffer("prompt_embeddings", self.representation_learner.prompt_embeddings)

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder_MMRL(clip_model)
        self.dtype = clip_model.dtype

        # SpecPL refiner, VAE teacher, and granule modules.
        self.attr_refiner = GlobalAttrTokenRefiner(cfg, clip_model)

        # caches for eval
        self.text_features_for_inference = None              # raw
        self.refined_text_features_for_inference = None      # refined
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def reset_cache(self):
        self.text_features_for_inference = None
        self.refined_text_features_for_inference = None
        self.compound_rep_tokens_text_for_inference = None
        self.compound_rep_tokens_visual_for_inference = None

    def forward(self, image, return_aux=False, teacher_bands=None):
        # -------- text: raw & refined (cache in eval) --------
        if self.representation_learner.training:
            compound_rep_tokens_text, compound_rep_tokens_visual = self.representation_learner()
            text_features_raw = self.text_encoder(
                self.prompt_embeddings, self.tokenized_prompts, compound_rep_tokens_text
            )
            refined_text, ref_stats = self.attr_refiner.refine_text(text_features_raw, return_stats=True)
        else:
            if self.text_features_for_inference is None or self.refined_text_features_for_inference is None:
                self.compound_rep_tokens_text_for_inference, self.compound_rep_tokens_visual_for_inference = (
                    self.representation_learner()
                )
                self.text_features_for_inference = self.text_encoder(
                    self.prompt_embeddings,
                    self.tokenized_prompts,
                    self.compound_rep_tokens_text_for_inference,
                )
                self.refined_text_features_for_inference = self.attr_refiner.refine_text(
                    self.text_features_for_inference, return_stats=False
                )

            compound_rep_tokens_text = self.compound_rep_tokens_text_for_inference
            compound_rep_tokens_visual = self.compound_rep_tokens_visual_for_inference
            text_features_raw = self.text_features_for_inference
            refined_text = self.refined_text_features_for_inference
            ref_stats = None

        # -------- image: global & rep --------
        image_features, image_features_rep = self.image_encoder([image.type(self.dtype), compound_rep_tokens_visual])

        image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)
        image_features_rep = image_features_rep / (image_features_rep.norm(dim=-1, keepdim=True) + 1e-6)

        # -------- normalize two text branches --------
        text_raw_n = text_features_raw / (text_features_raw.norm(dim=-1, keepdim=True) + 1e-6)
        text_ref_n = refined_text / (refined_text.norm(dim=-1, keepdim=True) + 1e-6)

        # Raw text serves the semantic branch; refined text serves the rep branch.
        logits = 100.0 * (image_features @ text_raw_n.t())
        logits_rep = 100.0 * (image_features_rep @ text_ref_n.t())
        logits_fusion = self.alpha * logits + (1.0 - self.alpha) * logits_rep

        # Return raw text features for the original MMRL regularizer.
        if (not self.training) or (not return_aux):
            return logits, logits_rep, logits_fusion, image_features, text_raw_n

        # Training-only auxiliary tensors for semantic and granule losses.
        if teacher_bands is None:
            t_low, t_high = self.attr_refiner.teacher_bands(image)  # float32 normalized
        else:
            t_low, t_high = teacher_bands

        aux = {
            "image_features_rep": image_features_rep,         # normalized
            "text_features_raw": text_features_raw,           # (C,512) raw (unnorm)
            "text_raw_n": text_raw_n,                          # normalized raw
            "refined_text_raw": refined_text,                 # (C,512) refined (before final normalize)
            "text_n": text_ref_n,                             # normalized refined (keep old key for trainer)
            "t_low": t_low,
            "t_high": t_high,
            "ref_stats": ref_stats if ref_stats is not None else {},
        }
        return logits, logits_rep, logits_fusion, image_features, text_raw_n, aux

class MMRL_Loss(_Loss):
    def __init__(self, reg_weight=1.0, alpha=0.7):
        super(MMRL_Loss, self).__init__()
        self.reg_weight = reg_weight
        self.alpha = alpha

    def forward(
        self,
        logits,
        logits_rep,
        image_features,
        text_features,
        image_features_clip,
        text_features_clip,
        label,
    ):
        xe_loss1 = F.cross_entropy(logits, label)
        xe_loss2 = F.cross_entropy(logits_rep, label)

        cossim_reg_img = 1.0 - torch.mean(F.cosine_similarity(image_features, image_features_clip, dim=1))
        cossim_reg_text = 1.0 - torch.mean(F.cosine_similarity(text_features, text_features_clip, dim=1))

        return self.alpha * xe_loss1 + (1.0 - self.alpha) * xe_loss2 + self.reg_weight * cossim_reg_img + self.reg_weight * cossim_reg_text


@TRAINER_REGISTRY.register()
class MMRLSpecPL(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.MMRL.PREC in ["fp16", "fp32", "amp"]

    _SLIM_FORMAT = "specpl_slim_v1"
    _LOAD_DROP_EXACT = (
        "prompt_embeddings",
    )
    _SLIM_KEEP_PREFIXES = (
        "representation_learner.",
        "attr_refiner.agg.",
        "attr_refiner.ln.",
        "attr_refiner.fusion.",
        "attr_refiner.granule_modulator.",
    )
    _SLIM_KEEP_EXACT = (
        "attr_refiner.attr_tokens",
        "attr_refiner.attr_token_counts",
        "image_encoder.proj_rep",
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
            osp.join(directory, "MultiModalRepresentationLearner", model_file),
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
            osp.join(directory, "MultiModalRepresentationLearner"),
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

    def _get_attr_refiner(self):
        model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        return model.attr_refiner

    @staticmethod
    def _teacher_bands_for_loss(refiner, image):
        # Use no_grad (not inference_mode): teacher outputs are targets/constants
        # but must be regular tensors so loss_sem / granule losses can backward
        # through fusion and the rest of the trainable graph.
        with torch.no_grad():
            low, high = refiner.teacher_bands(image)
        return low, high

    @staticmethod
    def _format_elapsed_mmss(elapsed_sec):
        minutes = int(elapsed_sec // 60)
        seconds = float(elapsed_sec) - minutes * 60
        return f"{minutes:02d}m {seconds:05.2f}s"

    @staticmethod
    def _format_bytes(size_bytes):
        size_mb = float(size_bytes) / float(1024 ** 2)
        size_gb = float(size_bytes) / float(1024 ** 3)
        return f"{int(size_bytes)} bytes ({size_mb:.2f} MB, {size_gb:.4f} GB)"

    @staticmethod
    def _format_flops(flops):
        if flops is None or flops <= 0:
            return "N/A"
        for scale, unit in ((1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs"), (1e3, "KFLOPs")):
            if flops >= scale:
                return f"{flops / scale:.4f} {unit}"
        return f"{float(flops):.0f} FLOPs"

    def _maybe_cuda_sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _reset_cuda_peak_memory_stats(self):
        if not torch.cuda.is_available():
            return
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_idx)

    def _get_cpu_memory_stats(self):
        rss_bytes = None
        if psutil is not None:
            rss_bytes = int(psutil.Process(os.getpid()).memory_info().rss)
        peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform.startswith("darwin"):
            peak_rss_bytes = int(peak_rss_raw)
        else:
            peak_rss_bytes = int(peak_rss_raw) * 1024
        return {
            "rss_bytes": rss_bytes,
            "peak_rss_bytes": peak_rss_bytes,
        }

    def _get_cuda_memory_stats(self):
        if not torch.cuda.is_available():
            return None

        stats = {
            "devices": int(torch.cuda.device_count()),
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
        for device_idx in range(stats["devices"]):
            stats["allocated_bytes"] += int(torch.cuda.memory_allocated(device_idx))
            stats["reserved_bytes"] += int(torch.cuda.memory_reserved(device_idx))
            stats["peak_allocated_bytes"] += int(torch.cuda.max_memory_allocated(device_idx))
            stats["peak_reserved_bytes"] += int(torch.cuda.max_memory_reserved(device_idx))
        return stats

    def _print_runtime_resources(self, prefix):
        cpu_stats = self._get_cpu_memory_stats()
        cpu_rss_str = self._format_bytes(cpu_stats["rss_bytes"]) if cpu_stats["rss_bytes"] is not None else "N/A"
        cpu_peak_str = self._format_bytes(cpu_stats["peak_rss_bytes"])
        gpu_stats = self._get_cuda_memory_stats()

        if gpu_stats is None:
            print(
                f"{prefix} resources: cpu_rss={cpu_rss_str}, "
                f"cpu_peak_rss={cpu_peak_str}, gpu=CPU-only"
            )
            return

        print(
            f"{prefix} resources: cpu_rss={cpu_rss_str}, cpu_peak_rss={cpu_peak_str}, "
            f"gpu_alloc={self._format_bytes(gpu_stats['allocated_bytes'])}, "
            f"gpu_reserved={self._format_bytes(gpu_stats['reserved_bytes'])}, "
            f"gpu_peak_alloc={self._format_bytes(gpu_stats['peak_allocated_bytes'])}, "
            f"gpu_peak_reserved={self._format_bytes(gpu_stats['peak_reserved_bytes'])}, "
            f"gpus={gpu_stats['devices']}"
        )

    def _init_runtime_timers(self):
        self.training_time_start = None
        self.training_seconds = 0.0
        self.training_gpu_hours = 0.0
        self.inference_seconds = 0.0
        self.inference_gpu_hours = 0.0
        self.inference_samples = 0
        self.inference_flops = 0.0
        self.inference_latency_sec = 0.0

    def _gpu_hours_from_seconds(self, elapsed_sec):
        if isinstance(self.device, torch.device):
            device_type = self.device.type
        else:
            device_type = str(self.device).split(":")[0]
        if device_type == "cuda" and torch.cuda.is_available():
            active_gpus = max(1, torch.cuda.device_count())
        else:
            active_gpus = 0
        return float(elapsed_sec) * float(active_gpus) / 3600.0

    def _profile_single_inference(self, sample_input):
        if sample_input is None or sample_input.numel() == 0:
            return None

        sample = sample_input[:1]
        repeats = int(getattr(self.cfg.TRAINER.MMRL, "INFER_PROFILE_REPEATS", 10))
        repeats = max(repeats, 1)
        model = self.model

        self._maybe_cuda_sync()
        with torch.no_grad():
            _ = model(sample)
        self._maybe_cuda_sync()

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(sample)
        self._maybe_cuda_sync()
        latency_sec = (time.perf_counter() - start) / float(repeats)

        flops = None
        profile_error = None
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                profile_memory=False,
                record_shapes=False,
            ) as prof:
                with torch.no_grad():
                    _ = model(sample)
                    self._maybe_cuda_sync()

            total_flops = 0.0
            for evt in prof.key_averages():
                evt_flops = getattr(evt, "flops", 0) or 0
                total_flops += float(evt_flops)
            if total_flops > 0:
                flops = total_flops
        except Exception as exc:
            profile_error = f"{type(exc).__name__}: {exc}"

        return {
            "latency_sec": latency_sec,
            "flops": flops,
            "error": profile_error,
        }

    def before_train(self):
        super().before_train()
        self._reset_cuda_peak_memory_stats()
        self._maybe_cuda_sync()
        self.training_time_start = time.perf_counter()
        print("MMRL training timer started")

    def after_train(self):
        self._maybe_cuda_sync()
        if self.training_time_start is not None:
            elapsed_sec = time.perf_counter() - self.training_time_start
            self.training_seconds = float(elapsed_sec)
            self.training_gpu_hours = self._gpu_hours_from_seconds(elapsed_sec)
            elapsed_mmss = self._format_elapsed_mmss(elapsed_sec)
            print(
                f"MMRL training finished in {elapsed_sec:.2f}s "
                f"({elapsed_mmss}), GPU Hours={self.training_gpu_hours:.4f}"
            )
            self._print_runtime_resources("MMRL training")
        super().after_train()

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.num_classes = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg, "MMRL")
        clip_model_zero_shot = load_clip_to_cpu(cfg)

        if cfg.TRAINER.MMRL.PREC in ["fp32", "amp"]:
            clip_model.float()
            clip_model_zero_shot.float()

        self.dtype = clip_model.dtype

        with torch.no_grad():
            self.text_encoder_clip = TextEncoder_CLIP(clip_model_zero_shot)
            text_features_clip = _get_text_base_features_zero_shot(cfg, classnames, clip_model_zero_shot, self.text_encoder_clip)
            self.text_features_clip = text_features_clip / (text_features_clip.norm(dim=-1, keepdim=True) + 1e-6)
        self.image_encoder_clip = clip_model_zero_shot.visual

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = [
            "representation_learner",
            "image_encoder.proj_rep",
            # SpecPL trainables.
            "attr_refiner.agg",
            "attr_refiner.ln",
            "attr_refiner.fusion",
            "attr_refiner.granule_modulator",
        ]

        for name, param in self.model.named_parameters():
            update = any(k in name for k in names_to_update)
            # keep teacher frozen
            if "attr_refiner.teacher" in name:
                update = False
            param.requires_grad_(update)

        enabled = {n for n, p in self.model.named_parameters() if p.requires_grad}
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.image_encoder_clip.to(self.device)
        self.text_features_clip = self.text_features_clip.to(self.device)
        self._init_runtime_timers()

        # Optional one-shot initialization for frozen attribute tokens.
        loader = getattr(self, "train_loader_x", None)
        if loader is None:
            loader = getattr(self, "train_loader", None)

        init_batches = int(getattr(cfg.TRAINER.MMRL, "ATTR_INIT_BATCHES", 10))
        if init_batches > 0:
            if loader is not None:
                print(f"Initializing attr_tokens with VAE teacher_low (batches={init_batches})")
                self.model.attr_refiner.init_attr_tokens(loader, device=self.device, max_batches=init_batches)
            else:
                print("Warning: train loader not found; skip attr_tokens init")

        reg_weight = cfg.TRAINER.MMRL.REG_WEIGHT
        alpha = cfg.TRAINER.MMRL.ALPHA
        self.criterion = MMRL_Loss(reg_weight=reg_weight, alpha=alpha)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("MultiModalRepresentationLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.MMRL.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.image_encoder_clip = nn.DataParallel(self.image_encoder_clip)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MMRL.PREC

        # helper to access attr_refiner weights under DP
        refiner = model.module.attr_refiner if isinstance(model, nn.DataParallel) else model.attr_refiner

        if prec == "amp":
            with autocast():
                with torch.no_grad():
                    image_features_clip = self.image_encoder_clip(image.type(self.dtype))
                    image_features_clip = image_features_clip / (image_features_clip.norm(dim=-1, keepdim=True) + 1e-6)

                teacher_low, teacher_high = self._teacher_bands_for_loss(refiner, image)
                logits, logits_rep, logits_fusion, image_features, text_features, aux = model(
                    image,
                    return_aux=True,
                    teacher_bands=(teacher_low, teacher_high),
                )
                # multi-gpu compatibility
                text_features = text_features[0 : self.num_classes]

                base_loss = self.criterion(
                    logits,
                    logits_rep,
                    image_features,
                    text_features,
                    image_features_clip,
                    self.text_features_clip,
                    label,
                )

                # Label-free semantic alignment between teacher_low and expected text semantics.
                t_low = aux["t_low"]
                text_raw = aux["text_features_raw"].to(torch.float32)
                text_sem = text_raw / (text_raw.norm(dim=-1, keepdim=True) + 1e-6)

                with torch.no_grad():
                    p = logits.detach().to(torch.float32).softmax(dim=-1)

                t_exp = p @ text_sem
                t_exp = t_exp / (t_exp.norm(dim=-1, keepdim=True) + 1e-6)
                cos_sem = (t_exp * t_low).sum(dim=-1)
                loss_sem = (1.0 - cos_sem).mean()

                # High-frequency band drives factual/counterfactual granule losses.
                t_high = aux["t_high"]
                img_global = image_features.to(torch.float32)  # normalized
                refined_text_raw = aux["refined_text_raw"]             # (C,512) dtype
                refined_text_n = aux["text_n"].to(torch.float32)

                shared = refiner.build_shared(
                    text_features=text_raw,
                    refined_text=refined_text_raw,
                    image_feat=img_global,
                    labels=label,
                )
                cond_f = refiner.fusion(shared, t_high)
                img_g_f = refiner.granule_modulator(cond_f, img_global)
                logits_g_f = 100.0 * (img_g_f @ refined_text_n.t())
                loss_g_f = F.cross_entropy(logits_g_f, label)

                perm = torch.randperm(t_high.size(0), device=t_high.device)
                t_high_cf = t_high[perm]
                label_cf = label[perm]
                cond_cf = refiner.fusion(shared, t_high_cf)
                img_g_cf = refiner.granule_modulator(cond_cf, img_global)
                logits_g_cf = 100.0 * (img_g_cf @ refined_text_n.t())
                loss_g_cf = F.cross_entropy(logits_g_cf, label_cf)

                loss = base_loss + refiner.lmbd_sem * loss_sem + refiner.lmbd_g_f * loss_g_f + refiner.lmbd_g_cf * loss_g_cf

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        else:
            with torch.no_grad():
                image_features_clip = self.image_encoder_clip(image.type(self.dtype))
                image_features_clip = image_features_clip / (image_features_clip.norm(dim=-1, keepdim=True) + 1e-6)

            teacher_low, teacher_high = self._teacher_bands_for_loss(refiner, image)
            logits, logits_rep, logits_fusion, image_features, text_features, aux = model(
                image,
                return_aux=True,
                teacher_bands=(teacher_low, teacher_high),
            )
            text_features = text_features[0 : self.num_classes]

            base_loss = self.criterion(
                logits,
                logits_rep,
                image_features,
                text_features,
                image_features_clip,
                self.text_features_clip,
                label,
            )

            # Label-free semantic alignment.
            t_low = aux["t_low"]
            text_raw = aux["text_features_raw"].to(torch.float32)
            text_sem = text_raw / (text_raw.norm(dim=-1, keepdim=True) + 1e-6)

            with torch.no_grad():
                p = logits.detach().to(torch.float32).softmax(dim=-1)

            t_exp = p @ text_sem
            t_exp = t_exp / (t_exp.norm(dim=-1, keepdim=True) + 1e-6)
            cos_sem = (t_exp * t_low).sum(dim=-1)
            loss_sem = (1.0 - cos_sem).mean()

            # granule losses on global branch
            t_high = aux["t_high"]
            img_global = image_features.to(torch.float32)
            refined_text_raw = aux["refined_text_raw"]
            refined_text_n = aux["text_n"].to(torch.float32)

            shared = refiner.build_shared(
                text_features=text_raw,
                refined_text=refined_text_raw,
                image_feat=img_global,
                labels=label,
            )
            cond_f = refiner.fusion(shared, t_high)
            img_g_f = refiner.granule_modulator(cond_f, img_global)
            logits_g_f = 100.0 * (img_g_f @ refined_text_n.t())
            loss_g_f = F.cross_entropy(logits_g_f, label)

            perm = torch.randperm(t_high.size(0), device=t_high.device)
            t_high_cf = t_high[perm]
            label_cf = label[perm]
            cond_cf = refiner.fusion(shared, t_high_cf)
            img_g_cf = refiner.granule_modulator(cond_cf, img_global)
            logits_g_cf = 100.0 * (img_g_cf @ refined_text_n.t())
            loss_g_cf = F.cross_entropy(logits_g_cf, label_cf)

            loss = base_loss + refiner.lmbd_sem * loss_sem + refiner.lmbd_g_f * loss_g_f + refiner.lmbd_g_cf * loss_g_cf

            optim.zero_grad()
            loss.backward()
            optim.step()

        output = logits_fusion
        loss_summary = {
            "loss": float(loss.item()),
            "loss_base": float(base_loss.item()),
            "loss_sem_align": float(loss_sem.item()),
            "loss_g_f": float(loss_g_f.item()),
            "loss_g_cf": float(loss_g_cf.item()),
            "cos_sem": float(cos_sem.mean().item()),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # add retrieval stats if present
        ref_stats = aux.get("ref_stats", {})
        if isinstance(ref_stats, dict):
            loss_summary.update(ref_stats)
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    def parse_batch_test(self, batch):
        input = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        # ensure refined-text cache is fresh for the current weights
        m = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        if hasattr(m, "reset_cache"):
            m.reset_cache()

        sub_cls = self.cfg.DATASET.SUBSAMPLE_CLASSES
        dataset = self.cfg.DATASET.NAME
        task = self.cfg.TASK

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        self._reset_cuda_peak_memory_stats()
        self._maybe_cuda_sync()
        inference_start = time.perf_counter()
        inference_num_samples = 0
        profile_sample = None

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            inference_num_samples += int(label.size(0))
            if profile_sample is None and input.size(0) > 0:
                profile_sample = input[:1]

            logits, _, logits_fusion, _, _ = self.model(input)

            if task == "B2N":
                output = logits_fusion if sub_cls == "base" else logits
            elif task == "FS":
                output = logits_fusion
            elif task == "CD":
                output = logits_fusion if dataset == "ImageNet" else logits
            else:
                raise ValueError("The TASK must be either B2N, CD, or FS.")

            self.evaluator.process(output, label)

        self._maybe_cuda_sync()
        inference_elapsed_sec = time.perf_counter() - inference_start
        self.inference_seconds = float(inference_elapsed_sec)
        self.inference_gpu_hours = self._gpu_hours_from_seconds(inference_elapsed_sec)
        self.inference_samples = int(inference_num_samples)
        avg_infer_sec = inference_elapsed_sec / float(max(inference_num_samples, 1))
        print(
            f"MMRL inference total ({split}): samples={inference_num_samples}, "
            f"time={inference_elapsed_sec:.2f}s ({self._format_elapsed_mmss(inference_elapsed_sec)}), "
            f"avg_per_image={avg_infer_sec * 1000.0:.2f} ms ({avg_infer_sec:.6f}s), "
            f"GPU Hours={self.inference_gpu_hours:.4f}"
        )
        self._print_runtime_resources(f"MMRL inference ({split})")

        profile_stats = self._profile_single_inference(profile_sample)
        if profile_stats is not None:
            self.inference_latency_sec = float(profile_stats["latency_sec"])
            self.inference_flops = float(profile_stats["flops"] or 0.0)
            print(
                f"MMRL inference profile ({split}, warm-text, bs=1): "
                f"latency={self.inference_latency_sec * 1000.0:.2f} ms ({self.inference_latency_sec:.6f}s), "
                f"FLOPs={int(self.inference_flops) if self.inference_flops > 0 else 'N/A'} "
                f"({self._format_flops(profile_stats['flops'])})"
            )
            if profile_stats["error"] is not None:
                print(f"MMRL inference FLOPs profiling warning: {profile_stats['error']}")

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def save_model(self, epoch, directory, is_best=False, val_result=None, model_name=""):
        # Mirror `stat/slim_checkpoints.py`: keep only learnable tensors; drop optimizer
        # / scheduler (they reference the full frozen model and inflate checkpoints).
        names = self.get_model_names()

        for name in names:
            full_sd = self._models[name].state_dict()
            slim_sd = {k: v for k, v in full_sd.items() if self._is_slim_key(k)}

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

            is_slim = self._is_slim_checkpoint(state_dict, ckpt_format)
            if not is_slim:
                state_dict = {k: v for k, v in state_dict.items() if self._is_slim_key(k)}

            epoch_ckpt = int(checkpoint.get("epoch", 0) or 0)
            print(
                f'Resuming {name} from "{ckpt_path}" '
                f"(format={'slim' if is_slim else 'full'}, epoch={epoch_ckpt}, kept_keys={len(state_dict)})"
            )

            self._models[name].load_state_dict(state_dict, strict=False)

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
        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = self._resolve_checkpoint_path(directory, name, model_file)
            if not osp.exists(model_path):
                tried = [
                    osp.join(directory, name, model_file),
                    osp.join(directory, "MultiModalRepresentationLearner", model_file),
                ]
                msg = "Model not found. Tried:\n  " + "\n  ".join(tried)
                raise FileNotFoundError(msg)

            checkpoint = load_checkpoint(model_path)
            state_dict = self._drop_load_only_keys(checkpoint["state_dict"])
            epoch_ckpt = checkpoint.get("epoch", None)
            ckpt_format = checkpoint.get("format", None)

            is_slim = self._is_slim_checkpoint(state_dict, ckpt_format)
            if not is_slim:
                state_dict = {k: v for k, v in state_dict.items() if self._is_slim_key(k)}

            msg_epoch = "" if epoch_ckpt is None else f", epoch={epoch_ckpt}"
            print(
                f'Loading weights to {name} from "{model_path}" '
                f"(format={'slim' if is_slim else 'full'}{msg_epoch}, kept_keys={len(state_dict)})"
            )

            self._models[name].load_state_dict(state_dict, strict=False)
