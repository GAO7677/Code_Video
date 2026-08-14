"""Noise-gated cross-frame point correspondence losses for Wan Q/K maps."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def points_to_token_coordinates(
    points_xy: torch.Tensor,
    *,
    pixel_hw: tuple[int, int],
    token_hw: tuple[int, int],
) -> torch.Tensor:
    """Map pixel-space points to continuous token-center coordinates."""
    if points_xy.shape[-1] != 2:
        raise ValueError(f"expected [...,2] points, got {points_xy.shape}")
    pixel_h, pixel_w = map(int, pixel_hw)
    token_h, token_w = map(int, token_hw)
    if min(pixel_h, pixel_w, token_h, token_w) <= 1:
        raise ValueError(f"invalid pixel/token geometry: {pixel_hw}/{token_hw}")
    points = points_xy.float()
    x = points[..., 0] * float(token_w - 1) / float(pixel_w - 1)
    y = points[..., 1] * float(token_h - 1) / float(pixel_h - 1)
    return torch.stack((x.clamp(0, token_w - 1), y.clamp(0, token_h - 1)), dim=-1)


def token_grid_coordinates(
    token_hw: tuple[int, int],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    height, width = map(int, token_hw)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid token grid: {token_hw}")
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def gaussian_soft_targets(
    centers_tn2: torch.Tensor,
    *,
    token_hw: tuple[int, int],
    sigma_tokens: float,
) -> torch.Tensor:
    """Return continuous Gaussian token labels as ``[T,N,H*W]``."""
    if centers_tn2.ndim != 3 or centers_tn2.shape[-1] != 2:
        raise ValueError(f"expected [T,N,2] centers, got {centers_tn2.shape}")
    if not math.isfinite(float(sigma_tokens)) or float(sigma_tokens) <= 0.0:
        raise ValueError("sigma_tokens must be positive and finite")
    centers = centers_tn2.float()
    grid = token_grid_coordinates(
        token_hw,
        device=centers.device,
        dtype=centers.dtype,
    )
    squared_distance = (centers[..., None, :] - grid).square().sum(dim=-1)
    target = torch.exp(-0.5 * squared_distance / float(sigma_tokens) ** 2)
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def noise_reliability_gate(
    sigma: torch.Tensor | float,
    *,
    gamma: float = 1.0,
    cutoff: float = 0.75,
) -> torch.Tensor:
    """Return an SNR-derived gate, with zero supervision above ``cutoff``."""
    if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("gamma must be positive and finite")
    if not math.isfinite(float(cutoff)) or not 0.0 < float(cutoff) <= 1.0:
        raise ValueError("cutoff must lie in (0,1]")
    value = torch.as_tensor(sigma, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool(
        (value > 1).any()
    ):
        raise ValueError("sigma must be finite and lie in [0,1]")
    snr = (1.0 - value).square() / value.square().clamp_min(1.0e-8)
    soft_gate = snr / (snr + float(gamma))
    return torch.where(value < float(cutoff), soft_gate, torch.zeros_like(soft_gate))


def cross_frame_point_terms(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    *,
    point_coordinates_tn2: torch.Tensor,
    point_visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    source_frame: int,
    sigma_tokens: float,
    coordinate_huber_beta: float,
    future_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute per-head soft-CE and coordinate terms from one layer's Q/K.

    Q/K use ``[B,T*H*W,H_selected,D]``. The source point is quantized only for
    selecting its Query token; every target remains a continuous Gaussian label.
    """
    if q_bshd.ndim != 4 or q_bshd.shape != k_bshd.shape:
        raise ValueError(f"expected matching [B,S,H,D] Q/K, got {q_bshd.shape}/{k_bshd.shape}")
    if point_coordinates_tn2.ndim != 3 or point_coordinates_tn2.shape[-1] != 2:
        raise ValueError(
            f"expected [T,N,2] point coordinates, got {point_coordinates_tn2.shape}"
        )
    if point_visibility_tn.shape != point_coordinates_tn2.shape[:2]:
        raise ValueError(
            "point visibility shape mismatch: "
            f"{point_visibility_tn.shape}/{point_coordinates_tn2.shape[:2]}"
        )
    if not math.isfinite(float(coordinate_huber_beta)) or float(
        coordinate_huber_beta
    ) <= 0.0:
        raise ValueError("coordinate_huber_beta must be positive and finite")

    token_h, token_w = map(int, token_hw)
    frame_tokens = token_h * token_w
    time_count, point_count = point_coordinates_tn2.shape[:2]
    if q_bshd.shape[1] != time_count * frame_tokens:
        raise ValueError(
            f"Q/K token count {q_bshd.shape[1]} does not match "
            f"T={time_count}, token_hw={token_hw}"
        )
    if not 0 <= int(source_frame) < time_count:
        raise ValueError(f"source_frame={source_frame} outside T={time_count}")

    q = q_bshd.reshape(
        q_bshd.shape[0], time_count, frame_tokens, q_bshd.shape[2], q_bshd.shape[3]
    )
    k = k_bshd.reshape(
        k_bshd.shape[0], time_count, frame_tokens, k_bshd.shape[2], k_bshd.shape[3]
    )
    centers = point_coordinates_tn2.to(device=q.device, dtype=torch.float32)
    visibility = point_visibility_tn.to(device=q.device, dtype=torch.bool)
    source_xy = centers[int(source_frame)].round().long()
    source_rows = (
        source_xy[:, 1].clamp(0, token_h - 1) * token_w
        + source_xy[:, 0].clamp(0, token_w - 1)
    )
    query = q[:, int(source_frame), source_rows].float()
    logits = torch.einsum("bnhd,btkhd->bthnk", query, k.float())
    logits = logits / math.sqrt(float(q_bshd.shape[-1]))
    log_probability = logits.log_softmax(dim=-1)
    probability = log_probability.exp()
    targets = gaussian_soft_targets(
        centers,
        token_hw=token_hw,
        sigma_tokens=float(sigma_tokens),
    )
    ce_contribution = -targets[None, :, None] * log_probability
    ce = ce_contribution.sum(dim=-1)

    grid = token_grid_coordinates(
        token_hw,
        device=q.device,
        dtype=probability.dtype,
    )
    predicted_coordinates = torch.einsum("bthnk,kc->bthnc", probability, grid)
    coordinate_target = centers[None, :, None].expand_as(predicted_coordinates)
    coordinate_huber = F.smooth_l1_loss(
        predicted_coordinates,
        coordinate_target,
        beta=float(coordinate_huber_beta),
        reduction="none",
    ).mean(dim=-1)

    valid = visibility & visibility[int(source_frame)][None, :]
    frame_ids = torch.arange(time_count, device=q.device)[:, None]
    if future_only:
        valid = valid & (frame_ids > int(source_frame))
    else:
        valid = valid & (frame_ids != int(source_frame))
    if not bool(valid.any()):
        raise RuntimeError("no valid cross-frame point pairs")
    return {
        "attention": probability,
        "target": targets,
        "ce_contribution": ce_contribution,
        "ce": ce,
        "predicted_coordinates": predicted_coordinates,
        "coordinate_huber": coordinate_huber,
        "valid": valid,
        "grid_coordinates": grid,
        "source_rows": source_rows,
    }


def coordinate_loss_sensitivity(
    attention_tns: torch.Tensor,
    predicted_tn2: torch.Tensor,
    target_tn2: torch.Tensor,
    *,
    token_hw: tuple[int, int],
    beta: float,
) -> torch.Tensor:
    """Approximate ``|d L_coord / d logit_s|`` for visualization."""
    if attention_tns.ndim != 3:
        raise ValueError(f"expected [T,N,S] attention, got {attention_tns.shape}")
    if predicted_tn2.shape != target_tn2.shape or predicted_tn2.shape[-1] != 2:
        raise ValueError(
            f"predicted/target coordinate mismatch: {predicted_tn2.shape}/{target_tn2.shape}"
        )
    if float(beta) <= 0.0:
        raise ValueError("beta must be positive")
    attention = attention_tns.float()
    predicted = predicted_tn2.to(attention)
    target = target_tn2.to(attention)
    difference = predicted - target
    derivative = torch.where(
        difference.abs() < float(beta),
        difference / float(beta),
        difference.sign(),
    ) / 2.0
    grid = token_grid_coordinates(
        token_hw,
        device=attention.device,
        dtype=attention.dtype,
    )
    centered_grid = grid[None, None] - predicted[..., None, :]
    logit_derivative = attention * torch.einsum(
        "tnsc,tnc->tns", centered_grid, derivative
    )
    return logit_derivative.abs()
