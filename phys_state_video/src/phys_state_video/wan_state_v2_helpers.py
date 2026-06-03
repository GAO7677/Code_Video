from __future__ import annotations

from dataclasses import dataclass

from .predictor_wan_state_v2 import resample_temporal_features
from .utils import require_torch

torch = require_torch()
F = torch.nn.functional


def compute_latent_step_count(frame_steps: int, temporal_stride: int) -> int:
    if frame_steps <= 0:
        raise ValueError(f"frame_steps must be positive, got {frame_steps}")
    if temporal_stride <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    return 1 + max(frame_steps - 1, 0) // temporal_stride


def compute_future_latent_steps(context_steps: int, future_steps: int, temporal_stride: int) -> int:
    total = compute_latent_step_count(context_steps + future_steps, temporal_stride)
    context = compute_latent_step_count(context_steps, temporal_stride)
    future = total - context
    if future <= 0:
        raise ValueError(
            f"future latent steps must be positive, got context_steps={context_steps}, future_steps={future_steps}, "
            f"temporal_stride={temporal_stride}, total_latents={total}, context_latents={context}"
        )
    return future


def resample_camera_to_latent_steps(camera: torch.Tensor, target_steps: int) -> torch.Tensor:
    return resample_temporal_features(camera, target_steps)


@dataclass(slots=True)
class MockLatentExtractor:
    latent_channels: int = 16
    latent_height: int = 8
    latent_width: int = 8
    temporal_stride: int = 4
    device: str = "cpu"

    def encode_context_frames_raw(self, context_frames: torch.Tensor) -> torch.Tensor:
        if context_frames.ndim != 5:
            raise ValueError(
                f"expected context frames with shape [B, K, 3, H, W], got {tuple(context_frames.shape)}"
            )
        batch, context_steps = context_frames.shape[:2]
        latent_steps = compute_latent_step_count(context_steps, self.temporal_stride)
        frames = context_frames.to(self.device).float()
        flattened = frames.permute(0, 2, 3, 4, 1).contiguous().view(
            batch,
            frames.shape[2] * frames.shape[3] * frames.shape[4],
            context_steps,
        )
        resized_time = F.interpolate(
            flattened,
            size=latent_steps,
            mode="linear",
            align_corners=False,
        )
        resized_time = resized_time.view(batch, 3, frames.shape[3], frames.shape[4], latent_steps).permute(0, 4, 1, 2, 3)
        spatial = F.interpolate(
            resized_time.reshape(batch * latent_steps, 3, frames.shape[3], frames.shape[4]),
            size=(self.latent_height, self.latent_width),
            mode="bilinear",
            align_corners=False,
        ).view(batch, latent_steps, 3, self.latent_height, self.latent_width)
        if self.latent_channels <= 3:
            return spatial[:, :, : self.latent_channels]
        repeats = (self.latent_channels + 2) // 3
        expanded = spatial.repeat(1, 1, repeats, 1, 1)
        return expanded[:, :, : self.latent_channels].contiguous()
