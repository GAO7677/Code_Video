"""Conditional cross-frame point correspondence terms for Wan Q/K maps."""

from __future__ import annotations

import math

import torch


def points_to_token_coordinates(
    points_xy: torch.Tensor,
    *,
    pixel_hw: tuple[int, int],
    token_hw: tuple[int, int],
) -> torch.Tensor:
    """Map pixel centers to continuous token-cell center coordinates."""
    if points_xy.shape[-1] != 2:
        raise ValueError(f"expected [...,2] points, got {points_xy.shape}")
    pixel_h, pixel_w = map(int, pixel_hw)
    token_h, token_w = map(int, token_hw)
    if min(pixel_h, pixel_w, token_h, token_w) <= 1:
        raise ValueError(f"invalid pixel/token geometry: {pixel_hw}/{token_hw}")
    points = points_xy.float()
    x = (points[..., 0] + 0.5) * float(token_w) / float(pixel_w) - 0.5
    y = (points[..., 1] + 0.5) * float(token_h) / float(pixel_h) - 0.5
    return torch.stack((x.clamp(0, token_w - 1), y.clamp(0, token_h - 1)), dim=-1)


def bilinear_sample_token_features(
    features_bshd: torch.Tensor,
    coordinates_n2: torch.Tensor,
    *,
    token_hw: tuple[int, int],
) -> torch.Tensor:
    """Sample ``[B,H*W,H,D]`` token features at continuous xy coordinates."""
    if features_bshd.ndim != 4:
        raise ValueError(f"expected [B,H*W,H,D] features, got {features_bshd.shape}")
    if coordinates_n2.ndim != 2 or coordinates_n2.shape[-1] != 2:
        raise ValueError(f"expected [N,2] coordinates, got {coordinates_n2.shape}")
    token_h, token_w = map(int, token_hw)
    if features_bshd.shape[1] != token_h * token_w:
        raise ValueError(
            f"feature token count {features_bshd.shape[1]} does not match {token_hw}"
        )

    coordinates = coordinates_n2.to(device=features_bshd.device, dtype=torch.float32)
    x = coordinates[:, 0].clamp(0, token_w - 1)
    y = coordinates[:, 1].clamp(0, token_h - 1)
    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=token_w - 1)
    y1 = (y0 + 1).clamp(max=token_h - 1)
    dx = x - x0
    dy = y - y0

    def gather(rows: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
        return features_bshd[:, rows * token_w + columns]

    top = gather(y0, x0) * (1.0 - dx)[None, :, None, None]
    top = top + gather(y0, x1) * dx[None, :, None, None]
    bottom = gather(y1, x0) * (1.0 - dx)[None, :, None, None]
    bottom = bottom + gather(y1, x1) * dx[None, :, None, None]
    return top * (1.0 - dy)[None, :, None, None] + bottom * dy[None, :, None, None]


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
) -> torch.Tensor:
    """Return a smooth SNR-derived reliability weight in ``[0,1]``."""
    if not math.isfinite(float(gamma)) or float(gamma) <= 0.0:
        raise ValueError("gamma must be positive and finite")
    value = torch.as_tensor(sigma, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()) or bool(
        (value > 1).any()
    ):
        raise ValueError("sigma must be finite and lie in [0,1]")
    snr = (1.0 - value).square() / value.square().clamp_min(1.0e-8)
    return snr / (snr + float(gamma))


def conditional_correspondence_objective(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    scheduler_sigma: float,
    gate_gamma: float,
    lambda_corr: float,
) -> dict[str, torch.Tensor]:
    """Compute the weighted soft-label CE for an aggregate spatial map."""
    if probability.shape != target.shape or probability.ndim < 2:
        raise ValueError(
            f"probability/target shape mismatch: {probability.shape}/{target.shape}"
        )
    if valid.shape != probability.shape[:-1]:
        raise ValueError(f"valid mask shape mismatch: {valid.shape}/{probability.shape[:-1]}")
    if not math.isfinite(float(lambda_corr)) or float(lambda_corr) <= 0.0:
        raise ValueError("lambda_corr must be positive and finite")
    valid_mask = valid.to(device=probability.device, dtype=torch.bool)
    if not bool(valid_mask.any()):
        raise RuntimeError("no valid correspondence targets")

    ce_contribution = -target.to(probability) * probability.clamp_min(1.0e-12).log()
    ce = ce_contribution.sum(dim=-1)
    raw_soft_ce = ce.masked_select(valid_mask).mean()
    gate = noise_reliability_gate(
        float(scheduler_sigma),
        gamma=float(gate_gamma),
    ).to(probability)
    gated_soft_ce = gate * raw_soft_ce
    return {
        "ce_contribution": ce_contribution,
        "ce": ce,
        "raw_soft_ce": raw_soft_ce,
        "noise_gate": gate,
        "gated_soft_ce": gated_soft_ce,
        "loss": float(lambda_corr) * gated_soft_ce,
    }


def cross_frame_point_terms(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    *,
    point_coordinates_tn2: torch.Tensor,
    point_visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    source_frame: int,
    sigma_tokens: float,
    future_only: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute per-head conditional spatial probabilities from one layer's Q/K.

    Q/K use ``[B,T*H*W,H_selected,D]``. This first implementation intentionally
    supports one video per forward because its point tracks have no batch axis.
    """
    if q_bshd.ndim != 4 or q_bshd.shape != k_bshd.shape:
        raise ValueError(f"expected matching [B,S,H,D] Q/K, got {q_bshd.shape}/{k_bshd.shape}")
    if q_bshd.shape[0] != 1:
        raise ValueError("point tracks are unbatched; expected Q/K batch size 1")
    if point_coordinates_tn2.ndim != 3 or point_coordinates_tn2.shape[-1] != 2:
        raise ValueError(
            f"expected [T,N,2] point coordinates, got {point_coordinates_tn2.shape}"
        )
    if point_visibility_tn.shape != point_coordinates_tn2.shape[:2]:
        raise ValueError(
            "point visibility shape mismatch: "
            f"{point_visibility_tn.shape}/{point_coordinates_tn2.shape[:2]}"
        )
    token_h, token_w = map(int, token_hw)
    frame_tokens = token_h * token_w
    time_count = point_coordinates_tn2.shape[0]
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
    query = bilinear_sample_token_features(
        q[:, int(source_frame)],
        centers[int(source_frame)],
        token_hw=token_hw,
    ).float()
    logits = torch.einsum("bnhd,btkhd->bthnk", query, k.float())
    logits = logits / math.sqrt(float(q_bshd.shape[-1]))
    probability = logits.softmax(dim=-1)
    targets = gaussian_soft_targets(
        centers,
        token_hw=token_hw,
        sigma_tokens=float(sigma_tokens),
    )

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
        "valid": valid,
        "source_query": query,
    }
