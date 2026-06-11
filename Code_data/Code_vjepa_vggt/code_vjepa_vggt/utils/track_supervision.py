from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrackBoxAlignment:
    matched_gt_indices: torch.Tensor
    matched_gt_centers: torch.Tensor
    matched_gt_valid: torch.Tensor
    pair_cost: torch.Tensor


def box_centers_and_validity(
    boxes: torch.Tensor,
    image_hw: tuple[int, int],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    # boxes: [B,T,K,4], normalized xyxy in [0,1]
    height, width = image_hw
    x0 = boxes[..., 0]
    y0 = boxes[..., 1]
    x1 = boxes[..., 2]
    y1 = boxes[..., 3]
    valid = (x1 - x0 > eps) & (y1 - y0 > eps)
    cx = 0.5 * (x0 + x1) * width
    cy = 0.5 * (y0 + y1) * height
    centers = torch.stack([cx, cy], dim=-1)
    return centers, valid.float()


def align_tracks_to_boxes(
    tracks: torch.Tensor,
    gt_boxes: torch.Tensor,
    image_hw: tuple[int, int],
) -> TrackBoxAlignment:
    # tracks: [B,T,Kp,2], gt_boxes: [B,T,Kg,4]
    gt_centers, gt_valid = box_centers_and_validity(gt_boxes, image_hw=image_hw)
    batch, frames, pred_objects, _ = tracks.shape
    gt_objects = gt_boxes.shape[2]

    diff = (tracks.unsqueeze(3) - gt_centers.unsqueeze(2)).abs().sum(dim=-1)
    valid = gt_valid.unsqueeze(2)
    valid_counts = valid.sum(dim=1)
    pair_cost = (diff * valid).sum(dim=1) / valid_counts.clamp_min(1.0)
    pair_cost = torch.where(valid_counts > 0, pair_cost, torch.full_like(pair_cost, 1.0e6))

    matched_gt_indices = torch.zeros(batch, pred_objects, dtype=torch.long, device=tracks.device)
    matched_gt_centers = torch.zeros(batch, frames, pred_objects, 2, dtype=tracks.dtype, device=tracks.device)
    matched_gt_valid = torch.zeros(batch, frames, pred_objects, dtype=tracks.dtype, device=tracks.device)

    for b in range(batch):
        used_gt: set[int] = set()
        for pred_idx in range(pred_objects):
            best_gt = None
            best_cost = None
            for gt_idx in range(gt_objects):
                if gt_idx in used_gt:
                    continue
                cost = pair_cost[b, pred_idx, gt_idx]
                if best_cost is None or float(cost.item()) < best_cost:
                    best_cost = float(cost.item())
                    best_gt = gt_idx
            if best_gt is None:
                best_gt = 0
            used_gt.add(best_gt)
            matched_gt_indices[b, pred_idx] = best_gt
            matched_gt_centers[b, :, pred_idx] = gt_centers[b, :, best_gt]
            matched_gt_valid[b, :, pred_idx] = gt_valid[b, :, best_gt]

    return TrackBoxAlignment(
        matched_gt_indices=matched_gt_indices,
        matched_gt_centers=matched_gt_centers,
        matched_gt_valid=matched_gt_valid,
        pair_cost=pair_cost,
    )


def track_box_l1_loss(
    tracks: torch.Tensor,
    matched_gt_centers: torch.Tensor,
    matched_gt_valid: torch.Tensor,
) -> torch.Tensor:
    weights = matched_gt_valid.unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    return ((tracks - matched_gt_centers).abs() * weights).sum() / denom
