from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


VJEPA_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/vjepa2-main")
WAN_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main_official")


def _pad_to_multiple(video: torch.Tensor, multiple: int = 16) -> torch.Tensor:
    height, width = video.shape[-2:]
    target_height = ((height + multiple - 1) // multiple) * multiple
    target_width = ((width + multiple - 1) // multiple) * multiple
    pad_height = target_height - height
    pad_width = target_width - width
    leading_shape = video.shape[:-3]
    channels = video.shape[-3]
    flattened = video.reshape(-1, channels, height, width)
    padded = F.pad(
        flattened,
        (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2),
        mode="replicate",
    )
    return padded.view(*leading_shape, channels, target_height, target_width)


class FrozenVJEPA2Extractor(nn.Module):
    feature_dim = 1408
    temporal_stride = 2
    spatial_stride = 16

    def __init__(self, checkpoint: Path, device: torch.device, num_frames: int = 10) -> None:
        super().__init__()
        sys.path.insert(0, str(VJEPA_ROOT))
        from src.models.vision_transformer import vit_giant_xformers

        model = vit_giant_xformers(
            img_size=(224, 384),
            patch_size=16,
            num_frames=num_frames,
            tubelet_size=2,
            use_sdpa=True,
            use_rope=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            handle_nonsquare_inputs=True,
        )
        checkpoint_state = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        target_state = checkpoint_state.get("target_encoder", checkpoint_state)
        cleaned_state = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in target_state.items()
        }
        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"V-JEPA checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.model = model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)
        self.device = device
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, video_btchw: torch.Tensor) -> torch.Tensor:
        video = _pad_to_multiple(video_btchw.to(self.device, non_blocking=True))
        video = video.permute(0, 2, 1, 3, 4)
        video = (video - self.image_mean) / self.image_std
        video = video.to(torch.bfloat16)
        tokens = self.model(video)
        batch, _, frames, height, width = video.shape
        latent_time = frames // self.temporal_stride
        latent_height = height // self.spatial_stride
        latent_width = width // self.spatial_stride
        expected_tokens = latent_time * latent_height * latent_width
        if tokens.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Unexpected V-JEPA tokens {tuple(tokens.shape)}; expected N={expected_tokens}"
            )
        return tokens.view(
            batch, latent_time, latent_height, latent_width, self.feature_dim
        ).float()


class FrozenWanVAEExtractor(nn.Module):
    feature_dim = 48
    temporal_stride = 4
    spatial_stride = 16

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        super().__init__()
        sys.path.insert(0, str(WAN_ROOT))
        from wan.modules.vae2_2 import Wan2_2_VAE

        helper = Wan2_2_VAE(
            vae_pth=str(checkpoint),
            dtype=torch.bfloat16,
            device=str(device),
        )
        self.model = helper.model.eval().requires_grad_(False)
        self.register_buffer("latent_mean", helper.scale[0], persistent=False)
        self.register_buffer("latent_inv_std", helper.scale[1], persistent=False)
        self.device = device

    @torch.no_grad()
    def forward(self, video_btchw: torch.Tensor) -> torch.Tensor:
        frames = video_btchw.shape[1]
        if (frames - 1) % self.temporal_stride != 0:
            raise ValueError(
                f"Wan VAE requires 4n+1 frames, got {frames}; use 9, 13, 17, ..."
            )
        video = _pad_to_multiple(video_btchw.to(self.device, non_blocking=True))
        video = video.permute(0, 2, 1, 3, 4).mul(2.0).sub(1.0)
        scale = (self.latent_mean, self.latent_inv_std)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents = self.model.encode(video, scale)
        return latents.permute(0, 2, 3, 4, 1).float()

    @torch.no_grad()
    def decode(self, latents_bthwd: torch.Tensor) -> torch.Tensor:
        latents = latents_bthwd.permute(0, 4, 1, 2, 3).to(self.device)
        scale = (self.latent_mean, self.latent_inv_std)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            video = self.model.decode(latents, scale)
        return video.add(1.0).div(2.0).clamp(0.0, 1.0).permute(0, 2, 1, 3, 4)


def build_frozen_extractor(
    space: str,
    checkpoint: Path,
    device: torch.device,
    num_frames: int,
) -> nn.Module:
    if space == "vjepa":
        return FrozenVJEPA2Extractor(checkpoint, device, num_frames=num_frames)
    if space == "vae":
        return FrozenWanVAEExtractor(checkpoint, device)
    raise ValueError(f"Unsupported feature space: {space}")
