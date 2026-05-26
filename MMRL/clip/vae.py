import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage


class VAEAttributeTeacher(nn.Module):
    """Freeze pretrained VAE; train-time teacher producing (low, high) attribute vectors.

    - Input: CLIP-normalized image tensor (B,3,H,W)
    - Output: float32 attribute vectors in CLIP text space (B,out_dim)

    Backward compatibility:
      - forward(image) returns **high-band** only
      - forward(image, return_bands=True) returns (low, high)
    """

    def __init__(
        self,
        pretrained_id="REPA-E/e2e-qwenimage-vae",
        highpass_k=7,
        out_dim=512,
        use_var=True,
        lowpass_k=None,
    ):
        super().__init__()

        self.vae = AutoencoderKLQwenImage.from_pretrained(pretrained_id).eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        self.base_z_dim = 16
        self.use_var = bool(use_var)
        self.z_dim = self.base_z_dim * (2 if self.use_var else 1)

        self.highpass_k = int(highpass_k)
        self.lowpass_k = int(lowpass_k) if lowpass_k is not None else int(highpass_k)
        self.out_dim = int(out_dim)

        # Separate heads: low/high statistics have different distributions.
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
        """Convert (B,16,1,h,w) -> (B,16) via mean(|.|) over spatial dims."""
        return z.abs().mean(dim=[2, 3, 4])

    @torch.no_grad()
    def forward(self, image_clip_normed: torch.Tensor, return_bands: bool = False):
        img = image_clip_normed.to(torch.float32)
        # de-normalize from CLIP space -> [0,1]
        img = img * self.clip_std + self.clip_mean
        img = img.clamp(0.0, 1.0)
        # map to [-1,1] for VAE
        img = img * 2.0 - 1.0
        img = img.unsqueeze(2)  # (B,3,1,H,W)

        posterior = self.vae.encode(img).latent_dist
        mu = posterior.mean  # (B,16,1,h,w)

        if self.use_var:
            if hasattr(posterior, "std") and posterior.std is not None:
                sigma = posterior.std
            else:
                sigma = torch.exp(0.5 * posterior.logvar)
        else:
            sigma = None

        # Low band
        low_mu = self._lowpass(mu, self.lowpass_k)
        feats_low = [self._stats_from_latent(low_mu)]
        if sigma is not None:
            low_sigma = self._lowpass(sigma, self.lowpass_k)
            feats_low.append(self._stats_from_latent(low_sigma))
        stats_low = torch.cat(feats_low, dim=1)  # (B,z_dim)
        out_low = self.proj_low(stats_low)       # (B,out_dim)

        # High band
        high_mu = self._highpass(mu, self.highpass_k)
        feats_high = [self._stats_from_latent(high_mu)]
        if sigma is not None:
            high_sigma = self._highpass(sigma, self.highpass_k)
            feats_high.append(self._stats_from_latent(high_sigma))
        stats_high = torch.cat(feats_high, dim=1)  # (B,z_dim)
        out_high = self.proj_high(stats_high)      # (B,out_dim)

        if return_bands:
            return out_low, out_high
        return out_high
