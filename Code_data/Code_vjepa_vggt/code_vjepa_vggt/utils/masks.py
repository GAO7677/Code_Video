from __future__ import annotations

import torch


def latent_frame_mask(
    num_video_frames: int,
    num_context_frames: int,
    vae_stride_t: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    lat_t = (num_video_frames - 1) // vae_stride_t + 1
    context_lat_t = max(1, (num_context_frames - 1) // vae_stride_t + 1)
    context_mask = torch.zeros(lat_t, dtype=torch.float32, device=device)
    context_mask[:context_lat_t] = 1.0
    future_mask = 1.0 - context_mask
    return context_mask, future_mask


def broadcast_latent_mask(mask_t: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    # latents: [C, T, H, W]
    return mask_t.view(1, -1, 1, 1).to(dtype=latents.dtype, device=latents.device)


def expand_context_latents_to_full(
    context_latents: torch.Tensor,
    full_latents: torch.Tensor,
) -> torch.Tensor:
    # context_latents/full_latents: [C, T, H, W]
    out = torch.zeros_like(full_latents)
    copy_t = min(context_latents.shape[1], full_latents.shape[1])
    out[:, :copy_t] = context_latents[:, :copy_t]
    return out
