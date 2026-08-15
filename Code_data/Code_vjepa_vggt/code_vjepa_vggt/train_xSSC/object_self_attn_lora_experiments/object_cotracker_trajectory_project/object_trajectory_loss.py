"""Object-focused point-trajectory losses for frozen video trackers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_tracks(
    tracks: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Normalize ``[..., (x, y)]`` pixel coordinates to ``[0, 1]``."""
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be [B,T,N,2], got {tuple(tracks.shape)}")
    if int(height) <= 1 or int(width) <= 1:
        raise ValueError(f"invalid track resolution: height={height}, width={width}")
    scale = tracks.new_tensor((float(width - 1), float(height - 1)))
    return tracks.float() / scale


def relative_tracks(tracks: torch.Tensor, *, anchor_frame: int) -> torch.Tensor:
    if tracks.ndim != 4 or tracks.shape[-1] != 2:
        raise ValueError(f"tracks must be [B,T,N,2], got {tuple(tracks.shape)}")
    if not 0 <= int(anchor_frame) < int(tracks.shape[1]):
        raise ValueError(
            f"anchor_frame={anchor_frame} outside T={int(tracks.shape[1])}"
        )
    return tracks - tracks[:, int(anchor_frame) : int(anchor_frame) + 1]


def object_trajectory_loss(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_visibility: torch.Tensor,
    *,
    height: int,
    width: int,
    anchor_frame: int,
    future_start_frame: int,
    huber_delta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare object-point displacement tracks using a fixed GT-validity gate.

    Prediction visibility is deliberately absent from this API. Allowing it to
    remove loss terms would let a trainable video generator evade supervision by
    producing frames that the frozen tracker marks invisible.
    """
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(
            f"pred/GT track shape mismatch: {pred_tracks.shape}/{gt_tracks.shape}"
        )
    expected_visibility = pred_tracks.shape[:-1]
    if gt_visibility.shape != expected_visibility:
        raise ValueError(
            "GT visibility shape mismatch: "
            f"expected {expected_visibility}, got {gt_visibility.shape}"
        )
    if not 0 <= int(future_start_frame) < int(pred_tracks.shape[1]):
        raise ValueError(
            f"future_start_frame={future_start_frame} outside T={pred_tracks.shape[1]}"
        )
    if float(huber_delta) <= 0.0:
        raise ValueError("huber_delta must be positive")

    gt_tracks = gt_tracks.detach().clone()
    gt_visibility = gt_visibility.detach().clone()
    pred = relative_tracks(
        normalize_tracks(pred_tracks, height=height, width=width),
        anchor_frame=anchor_frame,
    )
    gt = relative_tracks(
        normalize_tracks(gt_tracks, height=height, width=width),
        anchor_frame=anchor_frame,
    )
    difference = pred[:, int(future_start_frame) :] - gt[:, int(future_start_frame) :]
    valid = gt_visibility[:, int(future_start_frame) :].bool()
    if not bool(valid.any()):
        raise ValueError("GT tracker produced no visible future object points")

    per_coordinate = F.smooth_l1_loss(
        pred[:, int(future_start_frame) :],
        gt[:, int(future_start_frame) :],
        beta=float(huber_delta),
        reduction="none",
    )
    per_point_loss = per_coordinate.mean(dim=-1)
    per_point_raw_huber = per_point_loss * float(huber_delta)
    loss = per_point_loss[valid].mean()
    raw_huber = per_point_raw_huber[valid].mean()

    distance = torch.linalg.vector_norm(difference, dim=-1)
    squared_distance = difference.square().sum(dim=-1)
    gt_motion = torch.linalg.vector_norm(gt[:, int(future_start_frame) :], dim=-1)
    diagnostics = {
        "normalized_ade": distance[valid].mean(),
        "normalized_rmse": squared_distance[valid].mean().sqrt(),
        "normalized_gt_motion": gt_motion[valid].mean(),
        "raw_huber": raw_huber,
        "valid_fraction": valid.float().mean(),
        "valid_count": valid.sum(),
        "per_frame_loss": _masked_frame_mean(per_point_loss, valid),
        "per_frame_raw_huber": _masked_frame_mean(per_point_raw_huber, valid),
        "per_frame_ade": _masked_frame_mean(distance, valid),
        "per_point_distance": distance,
        "valid_future": valid,
    }
    return loss, diagnostics


def visibility_aware_trajectory_loss(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_visibility_probability: torch.Tensor,
    gt_confidence_probability: torch.Tensor,
    pred_visibility_probability: torch.Tensor,
    *,
    height: int,
    width: int,
    anchor_frame: int,
    future_start_frame: int,
    huber_delta: float,
    visibility_threshold: float,
    visibility_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weight coordinate targets by frozen GT reliability, never predicted visibility."""
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(
            f"pred/GT track shape mismatch: {pred_tracks.shape}/{gt_tracks.shape}"
        )
    expected_scores = pred_tracks.shape[:-1]
    for name, value in (
        ("gt_visibility_probability", gt_visibility_probability),
        ("gt_confidence_probability", gt_confidence_probability),
        ("pred_visibility_probability", pred_visibility_probability),
    ):
        if value.shape != expected_scores:
            raise ValueError(
                f"{name} shape mismatch: expected {expected_scores}, got {value.shape}"
            )
    if not 0.0 < float(visibility_threshold) < 1.0:
        raise ValueError("visibility_threshold must be in (0, 1)")
    if float(visibility_loss_weight) < 0.0:
        raise ValueError("visibility_loss_weight must be non-negative")

    gt_tracks = gt_tracks.detach().clone()
    gt_visibility_probability = gt_visibility_probability.detach().float().clone()
    gt_confidence_probability = gt_confidence_probability.detach().float().clone()
    pred = relative_tracks(
        normalize_tracks(pred_tracks, height=height, width=width),
        anchor_frame=anchor_frame,
    )
    gt = relative_tracks(
        normalize_tracks(gt_tracks, height=height, width=width),
        anchor_frame=anchor_frame,
    )
    start = int(future_start_frame)
    difference = pred[:, start:] - gt[:, start:]
    per_coordinate = F.smooth_l1_loss(
        pred[:, start:],
        gt[:, start:],
        beta=float(huber_delta),
        reduction="none",
    )
    per_point_coordinate = per_coordinate.mean(dim=-1)
    gt_visible = gt_visibility_probability[:, start:] > float(visibility_threshold)
    weights = (
        gt_visible.float()
        * gt_confidence_probability[:, start:].clamp(0.0, 1.0)
    )
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("GT tracker produced no reliable visible future object points")
    coordinate_loss = (per_point_coordinate * weights).sum() / weight_sum
    pred_visibility = pred_visibility_probability[:, start:].clamp(1e-6, 1.0 - 1e-6)
    per_point_visibility = -torch.log(pred_visibility)
    visibility_loss = (per_point_visibility * weights).sum() / weight_sum
    total = coordinate_loss + float(visibility_loss_weight) * visibility_loss

    distance = torch.linalg.vector_norm(difference, dim=-1)
    squared_distance = difference.square().sum(dim=-1)
    gt_motion = torch.linalg.vector_norm(gt[:, start:], dim=-1)
    diagnostics = {
        "coordinate_loss": coordinate_loss,
        "visibility_loss": visibility_loss,
        "total_loss": total,
        "raw_huber": coordinate_loss * float(huber_delta),
        "normalized_ade": (distance * weights).sum() / weight_sum,
        "normalized_rmse": ((squared_distance * weights).sum() / weight_sum).sqrt(),
        "normalized_gt_motion": (gt_motion * weights).sum() / weight_sum,
        "valid_fraction": gt_visible.float().mean(),
        "valid_count": gt_visible.sum(),
        "effective_weight_sum": weight_sum,
        "effective_weight_fraction": weights.mean(),
        "mean_gt_visibility_probability": gt_visibility_probability[:, start:].mean(),
        "mean_gt_confidence_probability": gt_confidence_probability[:, start:].mean(),
        "mean_pred_visibility_probability": pred_visibility_probability[:, start:].mean(),
        "per_frame_coordinate_loss": _weighted_frame_mean(
            per_point_coordinate, weights
        ),
        "per_frame_visibility_loss": _weighted_frame_mean(
            per_point_visibility, weights
        ),
        "per_frame_total_loss": _weighted_frame_mean(
            per_point_coordinate
            + float(visibility_loss_weight) * per_point_visibility,
            weights,
        ),
        "per_frame_ade": _weighted_frame_mean(distance, weights),
        "weights": weights,
        "valid_future": gt_visible,
    }
    return total, diagnostics


def _masked_frame_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    numerator = (values * valid).sum(dim=-1)
    denominator = valid.sum(dim=-1).clamp_min(1)
    result = numerator / denominator
    return result.masked_fill(valid.sum(dim=-1) == 0, float("nan"))


def _weighted_frame_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    numerator = (values * weights).sum(dim=-1)
    denominator = weights.sum(dim=-1)
    result = numerator / denominator.clamp_min(1e-12)
    return result.masked_fill(denominator <= 0.0, float("nan"))
