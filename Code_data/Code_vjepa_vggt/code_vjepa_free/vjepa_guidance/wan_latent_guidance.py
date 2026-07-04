from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class WanVJEPAConfig:
    guidance_steps: int = 6
    min_step_percent: float = 0.2
    max_step_percent: float = 0.8
    latent_step_size: float = 0.02
    preview_downsample_factor: int = 2
    preview_frame_stride: int = 1
    window_size: int = 16
    context_frames: int = 8
    stride: int = 4
    reduction: str = "mean"
    gradient_normalization: str = "rms"
    max_grad_norm: float | None = 10.0
    # Trust-region style guards for training-free guidance. These are off by
    # default so existing configs remain unchanged.
    max_correction_ratio: float | None = None
    stay_close_max_video_l1: float | None = None
    artifact_guard_mode: str = "none"
    # "surprise": legacy self-consistency windowing over the whole generation.
    # "context_anchored": align generated future to V-JEPA's prediction from the
    #   real conditioning frames (uses clean_prefix as fixed context).
    guidance_mode: str = "surprise"


def pick_guidance_step_indices(
    total_steps: int,
    *,
    count: int = 6,
    min_step_percent: float = 0.2,
    max_step_percent: float = 0.8,
) -> list[int]:
    if total_steps <= 0:
        return []
    if count <= 0:
        return []

    start = int(round((total_steps - 1) * min_step_percent))
    end = int(round((total_steps - 1) * max_step_percent))
    start = max(0, min(start, total_steps - 1))
    end = max(start, min(end, total_steps - 1))
    if count == 1:
        return [int(round((start + end) * 0.5))]

    grid = torch.linspace(start, end, steps=count)
    return sorted({int(round(v.item())) for v in grid})


def scheduler_sigma_for_timestep(scheduler, timestep: torch.Tensor | int) -> torch.Tensor:
    if getattr(scheduler, "step_index", None) is not None:
        return scheduler.sigmas[scheduler.step_index]
    if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(scheduler.timesteps.device)
    index = scheduler.index_for_timestep(timestep, scheduler.timesteps)
    return scheduler.sigmas[index]


def predict_x0_from_flow_model_output(
    *,
    scheduler,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
) -> torch.Tensor:
    sigma_t = scheduler_sigma_for_timestep(scheduler, timestep)
    sigma_t = sigma_t.to(device=latent_xt.device, dtype=latent_xt.dtype)
    while sigma_t.ndim < latent_xt.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    return latent_xt - sigma_t * model_output


def _as_wan_latent_list(latent: torch.Tensor) -> list[torch.Tensor]:
    if latent.ndim != 4:
        raise ValueError(f"Expected Wan latent [C,T,H,W], got {tuple(latent.shape)}")
    return [latent]


def _decode_with_wan_vae(vae, latent: torch.Tensor) -> torch.Tensor:
    decoded = vae.decode(_as_wan_latent_list(latent))
    if not isinstance(decoded, (list, tuple)) or not decoded:
        raise RuntimeError("Wan VAE returned an unexpected decode result")
    video = decoded[0]
    if video.ndim != 4:
        raise RuntimeError(f"Expected decoded video [C,T,H,W], got {tuple(video.shape)}")
    return video


def decode_preview_video(
    *,
    vae,
    x0_latent: torch.Tensor,
    preview_downsample_factor: int = 2,
    preview_frame_stride: int = 1,
) -> torch.Tensor:
    preview_latent = x0_latent
    if preview_frame_stride > 1:
        preview_latent = preview_latent[:, ::preview_frame_stride].contiguous()
    if preview_downsample_factor > 1:
        preview_latent = F.interpolate(
            preview_latent.unsqueeze(0),
            scale_factor=(1.0, 1.0 / preview_downsample_factor, 1.0 / preview_downsample_factor),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

    preview_video = _decode_with_wan_vae(vae, preview_latent)
    return preview_video.unsqueeze(0)


def normalize_gradient(gradient: torch.Tensor, mode: str = "rms", eps: float = 1e-6) -> torch.Tensor:
    if mode == "none":
        return gradient
    if mode == "rms":
        scale = gradient.pow(2).mean().sqrt().clamp_min(eps)
        return gradient / scale
    if mode == "l2":
        scale = gradient.norm().clamp_min(eps)
        return gradient / scale
    raise ValueError(f"Unsupported gradient normalization mode: {mode}")


def apply_vjepa_latent_guidance_with_decoder(
    *,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
    preview_decoder: Callable[..., torch.Tensor],
    energy_fn: Callable[..., torch.Tensor],
    config: WanVJEPAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_for_grad = latent_xt.detach().float().requires_grad_(True)
    model_output = model_output.detach().float()

    x0_pred = predict_x0_from_flow_model_output(
        scheduler=scheduler,
        latent_xt=latent_for_grad,
        model_output=model_output,
        timestep=timestep,
    )

    preview_video = preview_decoder(
        x0_pred,
        preview_downsample_factor=config.preview_downsample_factor,
        preview_frame_stride=config.preview_frame_stride,
    )

    energy = energy_fn(
        preview_video,
        window_size=config.window_size,
        context_frames=config.context_frames,
        stride=config.stride,
        reduction=config.reduction,
    )

    gradient = torch.autograd.grad(energy, latent_for_grad, retain_graph=False, create_graph=False)[0]
    if config.max_grad_norm is not None:
        gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        grad_norm = float(gradient.norm().item())
        if grad_norm > config.max_grad_norm:
            gradient = gradient * (config.max_grad_norm / max(grad_norm, 1e-6))

    gradient = normalize_gradient(gradient, mode=config.gradient_normalization)
    corrected = latent_xt.detach().float() - config.latent_step_size * gradient
    corrected = corrected.to(dtype=latent_xt.dtype)

    stats = {
        "energy": float(energy.detach().item()),
        "grad_rms": float(gradient.detach().pow(2).mean().sqrt().item()),
        "latent_rms": float(latent_xt.detach().float().pow(2).mean().sqrt().item()),
        "preview_frames": float(preview_video.shape[2]),
        "preview_height": float(preview_video.shape[3]),
        "preview_width": float(preview_video.shape[4]),
    }
    return corrected, stats


def apply_vjepa_latent_guidance(
    *,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
    vae,
    energy_fn: Callable[..., torch.Tensor],
    config: WanVJEPAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    return apply_vjepa_latent_guidance_with_decoder(
        latent_xt=latent_xt,
        model_output=model_output,
        timestep=timestep,
        scheduler=scheduler,
        preview_decoder=lambda x0_pred, preview_downsample_factor, preview_frame_stride: decode_preview_video(
            vae=vae,
            x0_latent=x0_pred,
            preview_downsample_factor=preview_downsample_factor,
            preview_frame_stride=preview_frame_stride,
        ),
        energy_fn=energy_fn,
        config=config,
    )
