from __future__ import annotations

import math

import numpy as np


def sample_points_from_box(box_xyxy_px: np.ndarray, num_points: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0 or num_points <= 0:
        return np.zeros((max(num_points, 0), 2), dtype=np.float32)

    cols = max(1, int(math.ceil(math.sqrt(float(num_points)))))
    rows = max(1, int(math.ceil(float(num_points) / float(cols))))
    xs = np.linspace(x0 + 0.2 * (x1 - x0), x1 - 0.2 * (x1 - x0), cols, dtype=np.float32)
    ys = np.linspace(y0 + 0.2 * (y1 - y0), y1 - 0.2 * (y1 - y0), rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    if grid.shape[0] >= num_points:
        return grid[:num_points].astype(np.float32)
    repeat = np.repeat(grid[-1:], num_points - grid.shape[0], axis=0)
    return np.concatenate([grid, repeat], axis=0).astype(np.float32)


def sample_points_from_mask(mask_hw: np.ndarray, num_points: int) -> np.ndarray:
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0 or ys.size == 0 or num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    coords = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
    if coords.shape[0] >= num_points:
        index = np.linspace(0, coords.shape[0] - 1, num_points, dtype=np.int64)
        return coords[index].astype(np.float32)
    repeat = np.repeat(coords[-1:], num_points - coords.shape[0], axis=0)
    return np.concatenate([coords, repeat], axis=0).astype(np.float32)


def build_vggt_query_prior(
    sam_masks_thw: np.ndarray,
    sam_boxes_t4: np.ndarray,
    *,
    num_queries: int,
) -> tuple[np.ndarray, str]:
    frame0_mask = sam_masks_thw[0]
    query_points = sample_points_from_mask(frame0_mask, num_queries)
    if query_points.shape[0] == num_queries and np.any(frame0_mask > 0):
        return query_points.astype(np.float32), "sam_mask_frame0"
    box0 = sam_boxes_t4[0]
    return sample_points_from_box(box0, num_queries), "sam_box_frame0"
