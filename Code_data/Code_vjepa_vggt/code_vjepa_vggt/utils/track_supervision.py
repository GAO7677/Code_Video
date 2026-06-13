from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


@dataclass
class TrackBoxAlignment:
    matched_gt_indices: torch.Tensor
    matched_gt_centers: torch.Tensor
    matched_gt_valid: torch.Tensor
    pair_cost: torch.Tensor


def _greedy_linear_assignment(cost_matrix: torch.Tensor) -> tuple[list[int], list[int]]:
    rows = list(range(int(cost_matrix.shape[0])))
    cols = list(range(int(cost_matrix.shape[1])))
    out_rows: list[int] = []
    out_cols: list[int] = []
    remaining = cost_matrix.clone()
    while rows and cols:
        flat_idx = int(torch.argmin(remaining).item())
        row_local = flat_idx // remaining.shape[1]
        col_local = flat_idx % remaining.shape[1]
        out_rows.append(rows.pop(row_local))
        out_cols.append(cols.pop(col_local))
        if not rows or not cols:
            break
        row_mask = torch.ones(remaining.shape[0], dtype=torch.bool, device=remaining.device)
        col_mask = torch.ones(remaining.shape[1], dtype=torch.bool, device=remaining.device)
        row_mask[row_local] = False
        col_mask[col_local] = False
        remaining = remaining[row_mask][:, col_mask]
    return out_rows, out_cols


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
    tracks = torch.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
    gt_boxes = torch.nan_to_num(gt_boxes, nan=0.0, posinf=0.0, neginf=0.0)
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
        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(pair_cost[b].detach().cpu().numpy())
            assigned = {int(r): int(c) for r, c in zip(row_ind.tolist(), col_ind.tolist())}
        else:
            row_ind, col_ind = _greedy_linear_assignment(pair_cost[b])
            assigned = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
        default_gt = int(pair_cost[b].mean(dim=0).argmin().item()) if gt_objects > 0 else 0
        for pred_idx in range(pred_objects):
            best_gt = assigned.get(pred_idx, default_gt)
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
    tracks = torch.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
    matched_gt_centers = torch.nan_to_num(matched_gt_centers, nan=0.0, posinf=0.0, neginf=0.0)
    matched_gt_valid = torch.nan_to_num(matched_gt_valid, nan=0.0, posinf=0.0, neginf=0.0)
    weights = matched_gt_valid.unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0)
    return ((tracks - matched_gt_centers).abs() * weights).sum() / denom


def track_box_iou_loss(
    tracks: torch.Tensor,
    gt_boxes: torch.Tensor,
    matched_gt_indices: torch.Tensor,
    image_hw: tuple[int, int],
    radius_px: float = 12.0,
) -> torch.Tensor:
    tracks = torch.nan_to_num(tracks, nan=0.0, posinf=0.0, neginf=0.0)
    gt_boxes = torch.nan_to_num(gt_boxes, nan=0.0, posinf=0.0, neginf=0.0)
    batch, frames, pred_objects, _ = tracks.shape
    height, width = image_hw
    pred_boxes = tracks.new_zeros(batch, frames, pred_objects, 4)
    pred_boxes[..., 0] = (tracks[..., 0] - radius_px) / max(float(width), 1.0)
    pred_boxes[..., 1] = (tracks[..., 1] - radius_px) / max(float(height), 1.0)
    pred_boxes[..., 2] = (tracks[..., 0] + radius_px) / max(float(width), 1.0)
    pred_boxes[..., 3] = (tracks[..., 1] + radius_px) / max(float(height), 1.0)
    pred_boxes = pred_boxes.clamp(0.0, 1.0)

    aligned_gt = gt_boxes.new_zeros(batch, frames, pred_objects, 4)
    valid = gt_boxes.new_zeros(batch, frames, pred_objects)
    for b in range(batch):
        for pred_idx in range(pred_objects):
            gt_idx = int(matched_gt_indices[b, pred_idx].item())
            aligned_gt[b, :, pred_idx] = gt_boxes[b, :, gt_idx]
            box = gt_boxes[b, :, gt_idx]
            valid[b, :, pred_idx] = ((box[:, 2] - box[:, 0]) > 1e-6) & ((box[:, 3] - box[:, 1]) > 1e-6)

    inter_x0 = torch.maximum(pred_boxes[..., 0], aligned_gt[..., 0])
    inter_y0 = torch.maximum(pred_boxes[..., 1], aligned_gt[..., 1])
    inter_x1 = torch.minimum(pred_boxes[..., 2], aligned_gt[..., 2])
    inter_y1 = torch.minimum(pred_boxes[..., 3], aligned_gt[..., 3])
    inter_w = (inter_x1 - inter_x0).clamp_min(0.0)
    inter_h = (inter_y1 - inter_y0).clamp_min(0.0)
    inter = inter_w * inter_h
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp_min(0.0) * (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp_min(0.0)
    gt_area = (aligned_gt[..., 2] - aligned_gt[..., 0]).clamp_min(0.0) * (aligned_gt[..., 3] - aligned_gt[..., 1]).clamp_min(0.0)
    union = (pred_area + gt_area - inter).clamp_min(1e-6)
    iou = inter / union
    valid = valid.to(dtype=iou.dtype)
    denom = valid.sum().clamp_min(1.0)
    return ((1.0 - iou) * valid).sum() / denom
