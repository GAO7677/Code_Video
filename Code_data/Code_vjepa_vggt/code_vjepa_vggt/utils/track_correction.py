from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class TrackCorrectionStats:
    corrected_points: int
    total_points: int
    snap_dist_mean_px: float
    snap_dist_max_px: float


@dataclass
class _MaskSupport:
    mask_u8: np.ndarray
    safe_mask_u8: np.ndarray
    support_coords_xy: np.ndarray


def _build_mask_support(mask_hw: np.ndarray, *, avoid_edges: bool = True) -> _MaskSupport | None:
    mask_u8 = (mask_hw > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return None

    support_mask = mask_u8
    if avoid_edges:
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        if max_dist > 1e-6:
            safe_thresh = max(1.5, 0.25 * max_dist)
            safe_mask = (dist >= safe_thresh).astype(np.uint8)
            if int(safe_mask.sum()) > 0:
                support_mask = safe_mask

    ys, xs = np.where(support_mask > 0)
    if xs.size == 0 or ys.size == 0:
        ys, xs = np.where(mask_u8 > 0)
    support_coords_xy = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
    return _MaskSupport(
        mask_u8=mask_u8,
        safe_mask_u8=support_mask.astype(np.uint8),
        support_coords_xy=support_coords_xy.astype(np.float32),
    )


def _snap_point_to_support(
    point_xy: np.ndarray,
    support: _MaskSupport,
    *,
    avoid_edges: bool = True,
) -> tuple[np.ndarray, bool, float]:
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    height, width = support.mask_u8.shape[:2]
    inside = 0 <= x < width and 0 <= y < height and bool(support.mask_u8[y, x] > 0)
    if inside:
        if not avoid_edges:
            return point_xy.astype(np.float32), False, 0.0
        safe = bool(support.safe_mask_u8[y, x] > 0)
        if safe:
            return point_xy.astype(np.float32), False, 0.0

    if support.support_coords_xy.shape[0] == 0:
        return point_xy.astype(np.float32), False, 0.0

    diff = support.support_coords_xy - point_xy[None, :].astype(np.float32)
    dist2 = (diff[:, 0] ** 2) + (diff[:, 1] ** 2)
    idx = int(np.argmin(dist2))
    snapped = support.support_coords_xy[idx].astype(np.float32)
    return snapped, True, float(np.sqrt(float(dist2[idx])))


def project_tracks_to_object_masks(
    tracks_tk2: torch.Tensor | np.ndarray,
    object_masks_thw: list[np.ndarray] | np.ndarray,
    query_owner: list[int],
    *,
    avoid_edges: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, TrackCorrectionStats]:
    if isinstance(tracks_tk2, torch.Tensor):
        tracks_np = tracks_tk2.detach().cpu().numpy().astype(np.float32)
        out_device = tracks_tk2.device
        out_dtype = tracks_tk2.dtype
    else:
        tracks_np = np.asarray(tracks_tk2, dtype=np.float32)
        out_device = None
        out_dtype = None

    if isinstance(object_masks_thw, np.ndarray):
        masks_list = [object_masks_thw[obj_idx] for obj_idx in range(object_masks_thw.shape[0])]
    else:
        masks_list = list(object_masks_thw)

    corrected = tracks_np.copy()
    mask_hit = np.zeros(tracks_np.shape[:2], dtype=np.float32)
    snap_distances: list[float] = []
    total_points = 0
    corrected_points = 0

    support_cache: dict[tuple[int, int], _MaskSupport | None] = {}
    for obj_idx, mask_thw in enumerate(masks_list):
        query_indices = [q for q, owner in enumerate(query_owner) if int(owner) == int(obj_idx)]
        if not query_indices:
            continue
        for t in range(mask_thw.shape[0]):
            cache_key = (obj_idx, t)
            if cache_key not in support_cache:
                support_cache[cache_key] = _build_mask_support(mask_thw[t], avoid_edges=avoid_edges)
            support = support_cache[cache_key]
            if support is None:
                continue
            for q in query_indices:
                total_points += 1
                snapped, did_snap, snap_dist = _snap_point_to_support(
                    corrected[t, q],
                    support,
                    avoid_edges=avoid_edges,
                )
                if did_snap:
                    corrected_points += 1
                    corrected[t, q] = snapped
                    snap_distances.append(snap_dist)
                mask_hit[t, q] = 1.0

    if out_device is not None:
        corrected_t = torch.from_numpy(corrected).to(device=out_device, dtype=out_dtype)
        mask_hit_t = torch.from_numpy(mask_hit).to(device=out_device, dtype=out_dtype)
    else:
        corrected_t = torch.from_numpy(corrected)
        mask_hit_t = torch.from_numpy(mask_hit)

    stats = TrackCorrectionStats(
        corrected_points=corrected_points,
        total_points=total_points,
        snap_dist_mean_px=float(np.mean(snap_distances)) if snap_distances else 0.0,
        snap_dist_max_px=float(np.max(snap_distances)) if snap_distances else 0.0,
    )
    return corrected_t, mask_hit_t, stats
