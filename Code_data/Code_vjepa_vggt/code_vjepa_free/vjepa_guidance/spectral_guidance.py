from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate_mask_thw(mask_bthw: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask_bthw
    if mask_bthw.ndim == 3:
        mask_bthw = mask_bthw.unsqueeze(0)
    if mask_bthw.ndim != 4:
        raise ValueError(f"Expected mask [B,T,H,W] or [T,H,W], got {tuple(mask_bthw.shape)}")
    bsz, frames, height, width = mask_bthw.shape
    kernel = 2 * int(radius) + 1
    flat = mask_bthw.reshape(bsz * frames, 1, height, width).float()
    dilated = F.max_pool2d(flat, kernel_size=kernel, stride=1, padding=int(radius))
    return dilated.reshape(bsz, frames, height, width).clamp_(0.0, 1.0)


def _rgb_to_luma(video_btchw: torch.Tensor) -> torch.Tensor:
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(video_btchw.shape)}")
    if int(video_btchw.shape[1]) != 3:
        raise ValueError(f"Expected RGB video, got shape {tuple(video_btchw.shape)}")
    red = video_btchw[:, 0]
    green = video_btchw[:, 1]
    blue = video_btchw[:, 2]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def compute_temporal_lowpass_residual_map(
    video_btchw: torch.Tensor,
    *,
    future_start_idx: int,
    lowpass_ratio: float = 0.18,
    normalize_percentile: float = 95.0,
) -> torch.Tensor:
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected video [B,C,T,H,W], got {tuple(video_btchw.shape)}")
    if future_start_idx <= 0 or future_start_idx >= int(video_btchw.shape[2]):
        raise ValueError(
            f"future_start_idx must be in [1, T-1], got {future_start_idx} for T={int(video_btchw.shape[2])}"
        )
    if float(lowpass_ratio) <= 0:
        raise ValueError(f"lowpass_ratio must be > 0, got {lowpass_ratio}")

    luma = _rgb_to_luma(video_btchw.float())
    previous = torch.cat([luma[:, :1], luma[:, :-1]], dim=1)
    residual = (luma - previous).abs()
    future_residual = residual[:, future_start_idx:]
    bsz, future_frames, height, width = future_residual.shape
    if future_frames <= 0:
        raise ValueError("Future clip is empty after future_start_idx split")

    fy = torch.fft.fftfreq(height, d=1.0, device=future_residual.device).view(1, 1, height, 1)
    fx = torch.fft.fftfreq(width, d=1.0, device=future_residual.device).view(1, 1, 1, width)
    radial = torch.sqrt(fx.square() + fy.square())
    cutoff = float(lowpass_ratio) * float(radial.max().item())
    lowpass = (radial <= cutoff).to(dtype=future_residual.dtype)

    fft = torch.fft.fft2(future_residual, dim=(-2, -1))
    lowpass_only = torch.fft.ifft2(fft * lowpass, dim=(-2, -1)).real.abs()
    smoothed = F.avg_pool2d(
        lowpass_only.reshape(bsz * future_frames, 1, height, width),
        kernel_size=5,
        stride=1,
        padding=2,
    ).reshape(bsz, future_frames, height, width)

    percentile = min(100.0, max(50.0, float(normalize_percentile))) / 100.0
    flat = smoothed.reshape(bsz, -1)
    scale = torch.quantile(flat, q=percentile, dim=1, keepdim=True).clamp_min_(1.0e-6)
    normalized = (smoothed / scale.view(bsz, 1, 1, 1)).clamp_(0.0, 1.0)
    return normalized
