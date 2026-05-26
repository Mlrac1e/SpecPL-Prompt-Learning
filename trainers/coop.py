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

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from tqdm import tqdm, trange
_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

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


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    @staticmethod
    def _format_elapsed_mmss(elapsed_sec):
        minutes = int(elapsed_sec // 60)
        seconds = float(elapsed_sec) - minutes * 60
        return f"{minutes:02d}m {seconds:05.2f}s"

    @staticmethod
    def _format_cache_size(size_bytes):
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
        cpu_rss_str = self._format_cache_size(cpu_stats["rss_bytes"]) if cpu_stats["rss_bytes"] is not None else "N/A"
        cpu_peak_str = self._format_cache_size(cpu_stats["peak_rss_bytes"])
        gpu_stats = self._get_cuda_memory_stats()

        if gpu_stats is None:
            print(
                f"{prefix} resources: cpu_rss={cpu_rss_str}, "
                f"cpu_peak_rss={cpu_peak_str}, gpu=CPU-only"
            )
            return

        print(
            f"{prefix} resources: cpu_rss={cpu_rss_str}, cpu_peak_rss={cpu_peak_str}, "
            f"gpu_alloc={self._format_cache_size(gpu_stats['allocated_bytes'])}, "
            f"gpu_reserved={self._format_cache_size(gpu_stats['reserved_bytes'])}, "
            f"gpu_peak_alloc={self._format_cache_size(gpu_stats['peak_allocated_bytes'])}, "
            f"gpu_peak_reserved={self._format_cache_size(gpu_stats['peak_reserved_bytes'])}, "
            f"gpus={gpu_stats['devices']}"
        )

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
        repeats = int(getattr(self.cfg.TRAINER.COOP, "INFER_PROFILE_REPEATS", 10))
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
        print("CoOp_v2 training timer started")

    def after_train(self):
        self._maybe_cuda_sync()
        if getattr(self, "training_time_start", None) is not None:
            elapsed_sec = time.perf_counter() - self.training_time_start
            self.training_seconds = float(elapsed_sec)
            self.training_gpu_hours = self._gpu_hours_from_seconds(elapsed_sec)
            elapsed_mmss = self._format_elapsed_mmss(elapsed_sec)
            print(
                f"CoOp_v2 training finished in {elapsed_sec:.2f}s "
                f"({elapsed_mmss}), GPU Hours={self.training_gpu_hours:.4f}"
            )
        self._print_runtime_resources("CoOp_v2 training")
        super().after_train()

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
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            print(param.dtype)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.training_time_start = None
        self.training_seconds = 0.0
        self.training_gpu_hours = 0.0
        self.inference_seconds = 0.0
        self.inference_gpu_hours = 0.0
        self.inference_samples = 0
        self.inference_flops = 0.0
        self.inference_latency_sec = 0.0

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

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

            output = self.model(input)
            self.evaluator.process(output, label)

        self._maybe_cuda_sync()
        inference_elapsed_sec = time.perf_counter() - inference_start
        self.inference_seconds = float(inference_elapsed_sec)
        self.inference_gpu_hours = self._gpu_hours_from_seconds(inference_elapsed_sec)
        self.inference_samples = int(inference_num_samples)
        avg_infer_sec = inference_elapsed_sec / float(max(inference_num_samples, 1))
        print(
            f"CoOp_v2 inference total ({split}): samples={inference_num_samples}, "
            f"time={inference_elapsed_sec:.2f}s ({self._format_elapsed_mmss(inference_elapsed_sec)}), "
            f"avg_per_image={avg_infer_sec * 1000.0:.2f} ms ({avg_infer_sec:.6f}s), "
            f"GPU Hours={self.inference_gpu_hours:.4f}"
        )
        self._print_runtime_resources(f"CoOp_v2 inference ({split})")

        profile_stats = self._profile_single_inference(profile_sample)
        if profile_stats is not None:
            self.inference_latency_sec = float(profile_stats["latency_sec"])
            self.inference_flops = float(profile_stats["flops"] or 0.0)
            print(
                f"CoOp_v2 inference profile ({split}, warm-cache, bs=1): "
                f"latency={self.inference_latency_sec * 1000.0:.2f} ms ({self.inference_latency_sec:.6f}s), "
                f"FLOPs={int(self.inference_flops) if self.inference_flops > 0 else 'N/A'} "
                f"({self._format_flops(profile_stats['flops'])})"
            )
            if profile_stats["error"] is not None:
                print(f"CoOp_v2 inference FLOPs profiling warning: {profile_stats['error']}")

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

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
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
