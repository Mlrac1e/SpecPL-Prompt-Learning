import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEAttributeTeacher(nn.Module):
    """Frozen VAE teacher with spectral disentanglement.

    - Input: CLIP-normalized image tensor (B, 3, H, W)
    - Output: float32 attribute vectors in CLIP text space (B, out_dim)

    Backward compatibility:
      - forward(image) returns high-band only
      - forward(image, return_bands=True) returns (low, high)
    """

    teacher_name = "vae_disentangled"

    def __init__(
        self,
        pretrained_id="REPA-E/e2e-qwenimage-vae",
        highpass_k=7,
        out_dim=512,
        use_var=True,
        lowpass_k=None,
    ):
        super().__init__()

        from diffusers import AutoencoderKLQwenImage

        self.vae = AutoencoderKLQwenImage.from_pretrained(pretrained_id).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.base_z_dim = 16
        self.use_var = bool(use_var)
        self.z_dim = self.base_z_dim * (2 if self.use_var else 1)

        self.highpass_k = int(highpass_k)
        self.lowpass_k = int(lowpass_k) if lowpass_k is not None else int(highpass_k)
        self.out_dim = int(out_dim)

        self.proj_low = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        self.proj_high = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

        for proj in (self.proj_low, self.proj_high):
            nn.init.normal_(proj[1].weight, std=1e-3)
            nn.init.zeros_(proj[1].bias)
            nn.init.normal_(proj[3].weight, std=1e-3)
            nn.init.zeros_(proj[3].bias)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean, persistent=False)
        self.register_buffer("clip_std", std, persistent=False)

    def _prepare_vae_input(self, image_clip_normed: torch.Tensor) -> torch.Tensor:
        img = image_clip_normed.to(torch.float32)
        img = img * self.clip_std + self.clip_mean
        img = img.clamp(0.0, 1.0)
        img = img * 2.0 - 1.0
        return img.unsqueeze(2)

    def _encode_posterior(self, image_clip_normed: torch.Tensor):
        img = self._prepare_vae_input(image_clip_normed)
        posterior = self.vae.encode(img).latent_dist
        mu = posterior.mean

        if self.use_var:
            if hasattr(posterior, "std") and posterior.std is not None:
                sigma = posterior.std
            else:
                sigma = torch.exp(0.5 * posterior.logvar)
        else:
            sigma = None
        return mu, sigma

    def _lowpass(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if k > 1:
            pad = k // 2
            return F.avg_pool3d(z, kernel_size=(1, k, k), stride=1, padding=(0, pad, pad))
        return z

    def _highpass(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if k > 1:
            low = self._lowpass(z, k)
            return z - low
        return z

    @staticmethod
    def _stats_from_latent(z: torch.Tensor) -> torch.Tensor:
        return z.abs().mean(dim=[2, 3, 4])

    @torch.no_grad()
    def forward(self, image_clip_normed: torch.Tensor, return_bands: bool = False):
        mu, sigma = self._encode_posterior(image_clip_normed)

        low_mu = self._lowpass(mu, self.lowpass_k)
        feats_low = [self._stats_from_latent(low_mu)]
        if sigma is not None:
            low_sigma = self._lowpass(sigma, self.lowpass_k)
            feats_low.append(self._stats_from_latent(low_sigma))
        stats_low = torch.cat(feats_low, dim=1)
        out_low = self.proj_low(stats_low)

        high_mu = self._highpass(mu, self.highpass_k)
        feats_high = [self._stats_from_latent(high_mu)]
        if sigma is not None:
            high_sigma = self._highpass(sigma, self.highpass_k)
            feats_high.append(self._stats_from_latent(high_sigma))
        stats_high = torch.cat(feats_high, dim=1)
        out_high = self.proj_high(stats_high)

        if return_bands:
            return out_low, out_high
        return out_high


class VAEHolisticAttributeTeacher(VAEAttributeTeacher):
    """Frozen VAE teacher without low/high disentanglement.

    It uses the same VAE latent space as the full method but extracts a single
    holistic attribute vector. For compatibility with the existing training code,
    return_bands=True yields (holistic, holistic).
    """

    teacher_name = "vae_holistic"

    def __init__(
        self,
        pretrained_id="REPA-E/e2e-qwenimage-vae",
        out_dim=512,
        use_var=True,
        **kwargs,
    ):
        super().__init__(
            pretrained_id=pretrained_id,
            highpass_k=1,
            out_dim=out_dim,
            use_var=use_var,
            lowpass_k=1,
        )
        self.proj_holistic = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        nn.init.normal_(self.proj_holistic[1].weight, std=1e-3)
        nn.init.zeros_(self.proj_holistic[1].bias)
        nn.init.normal_(self.proj_holistic[3].weight, std=1e-3)
        nn.init.zeros_(self.proj_holistic[3].bias)

    @torch.no_grad()
    def forward(self, image_clip_normed: torch.Tensor, return_bands: bool = False):
        mu, sigma = self._encode_posterior(image_clip_normed)
        feats = [self._stats_from_latent(mu)]
        if sigma is not None:
            feats.append(self._stats_from_latent(sigma))
        stats = torch.cat(feats, dim=1)
        out = self.proj_holistic(stats)

        if return_bands:
            return out, out
        return out


class CLIPViTAttributeTeacher(nn.Module):
    """Frozen CLIP-ViT patch-space teacher with simple low/high decomposition."""

    teacher_name = "clip_disentangled"

    def __init__(
        self,
        clip_visual,
        highpass_k=7,
        out_dim=512,
        use_var=True,
        lowpass_k=None,
    ):
        super().__init__()

        if not hasattr(clip_visual, "conv1"):
            raise ValueError(
                "CLIPViTAttributeTeacher expects a ViT-style CLIP visual encoder with conv1 patch embedding."
            )

        patch_weight = clip_visual.conv1.weight.detach().to(torch.float32).clone()
        self.register_buffer("patch_weight", patch_weight, persistent=False)

        if getattr(clip_visual.conv1, "bias", None) is not None:
            patch_bias = clip_visual.conv1.bias.detach().to(torch.float32).clone()
            self.register_buffer("patch_bias", patch_bias, persistent=False)
        else:
            self.patch_bias = None

        stride = clip_visual.conv1.stride
        padding = clip_visual.conv1.padding
        self.patch_stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.patch_padding = padding if isinstance(padding, tuple) else (padding, padding)

        self.base_z_dim = int(patch_weight.shape[0])
        self.use_var = bool(use_var)
        self.z_dim = self.base_z_dim * (2 if self.use_var else 1)

        self.highpass_k = int(highpass_k)
        self.lowpass_k = int(lowpass_k) if lowpass_k is not None else int(highpass_k)
        self.out_dim = int(out_dim)

        self.proj_low = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )
        self.proj_high = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, out_dim, bias=True),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        )

        for proj in (self.proj_low, self.proj_high):
            nn.init.normal_(proj[1].weight, std=1e-3)
            nn.init.zeros_(proj[1].bias)
            nn.init.normal_(proj[3].weight, std=1e-3)
            nn.init.zeros_(proj[3].bias)

    def _patch_embed(self, image_clip_normed: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            image_clip_normed.to(torch.float32),
            self.patch_weight,
            self.patch_bias,
            stride=self.patch_stride,
            padding=self.patch_padding,
        )

    def _lowpass(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if k > 1:
            pad = k // 2
            return F.avg_pool2d(z, kernel_size=k, stride=1, padding=pad)
        return z

    def _highpass(self, z: torch.Tensor, k: int) -> torch.Tensor:
        if k > 1:
            return z - self._lowpass(z, k)
        return z

    def _stats_from_map(self, z: torch.Tensor) -> torch.Tensor:
        mean_abs = z.abs().mean(dim=[2, 3])
        if not self.use_var:
            return mean_abs
        std = z.flatten(2).std(dim=-1, unbiased=False)
        return torch.cat([mean_abs, std], dim=1)

    @torch.no_grad()
    def forward(self, image_clip_normed: torch.Tensor, return_bands: bool = False):
        z = self._patch_embed(image_clip_normed)
        low_map = self._lowpass(z, self.lowpass_k)
        high_map = self._highpass(z, self.highpass_k)

        out_low = self.proj_low(self._stats_from_map(low_map))
        out_high = self.proj_high(self._stats_from_map(high_map))

        if return_bands:
            return out_low, out_high
        return out_high


def build_attr_teacher(cfg, clip_model, out_dim):
    ccfg = cfg.TRAINER.COOP
    mode = str(getattr(ccfg, "ATTR_TEACHER_MODE", "vae_disentangled")).lower()

    pretrained_id = getattr(ccfg, "VAE_PRETRAINED_ID", "REPA-E/e2e-qwenimage-vae")
    vae_highpass_k = int(getattr(ccfg, "VAE_HIGHPASS_K", 7))
    vae_lowpass_k = getattr(ccfg, "VAE_LOWPASS_K", None)
    vae_use_var = bool(getattr(ccfg, "VAE_USE_VAR", True))

    clip_highpass_k = int(getattr(ccfg, "CLIP_HIGHPASS_K", vae_highpass_k))
    clip_lowpass_k = getattr(ccfg, "CLIP_LOWPASS_K", vae_lowpass_k)
    clip_use_var = bool(getattr(ccfg, "CLIP_USE_VAR", vae_use_var))

    if mode in {"vae_disentangled", "vae_dis", "vae+disentangled"}:
        return VAEAttributeTeacher(
            pretrained_id=pretrained_id,
            highpass_k=vae_highpass_k,
            out_dim=out_dim,
            use_var=vae_use_var,
            lowpass_k=vae_lowpass_k,
        )

    if mode in {"vae_holistic", "vae_no_disentanglement", "vae_holistic_no_disentanglement", "vae+holistic"}:
        return VAEHolisticAttributeTeacher(
            pretrained_id=pretrained_id,
            out_dim=out_dim,
            use_var=vae_use_var,
        )

    if mode in {"clip_disentangled", "clip_simple", "clip_patch_disentangled", "clip"}:
        return CLIPViTAttributeTeacher(
            clip_visual=clip_model.visual,
            highpass_k=clip_highpass_k,
            out_dim=out_dim,
            use_var=clip_use_var,
            lowpass_k=clip_lowpass_k,
        )

    raise ValueError(
        f"Unknown ATTR_TEACHER_MODE={mode}. Expected one of: "
        "vae_disentangled, vae_holistic, clip_disentangled"
    )
