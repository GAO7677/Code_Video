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

    per_coordinate = F.huber_loss(
        pred[:, int(future_start_frame) :],
        gt[:, int(future_start_frame) :],
        delta=float(huber_delta),
        reduction="none",
    )
    per_point_huber = per_coordinate.mean(dim=-1)
    loss = per_point_huber[valid].mean()

    distance = torch.linalg.vector_norm(difference, dim=-1)
    squared_distance = difference.square().sum(dim=-1)
    gt_motion = torch.linalg.vector_norm(gt[:, int(future_start_frame) :], dim=-1)
    diagnostics = {
        "normalized_ade": distance[valid].mean(),
        "normalized_rmse": squared_distance[valid].mean().sqrt(),
        "normalized_gt_motion": gt_motion[valid].mean(),
        "valid_fraction": valid.float().mean(),
        "valid_count": valid.sum(),
        "per_frame_ade": _masked_frame_mean(distance, valid),
        "per_point_distance": distance,
        "valid_future": valid,
    }
    return loss, diagnostics


def _masked_frame_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    numerator = (values * valid).sum(dim=-1)
    denominator = valid.sum(dim=-1).clamp_min(1)
    result = numerator / denominator
    return result.masked_fill(valid.sum(dim=-1) == 0, float("nan"))
