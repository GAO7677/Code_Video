from __future__ import annotations

import math

import cv2
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


def _sample_points_from_coords(coords_xy: np.ndarray, num_points: int) -> np.ndarray:
    if coords_xy.shape[0] == 0 or num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if coords_xy.shape[0] <= num_points:
        repeat = np.repeat(coords_xy[-1:], num_points - coords_xy.shape[0], axis=0)
        return np.concatenate([coords_xy, repeat], axis=0).astype(np.float32)

    cols = max(1, int(math.ceil(math.sqrt(float(num_points)))))
    rows = max(1, int(math.ceil(float(num_points) / float(cols))))
    x0 = float(coords_xy[:, 0].min())
    x1 = float(coords_xy[:, 0].max())
    y0 = float(coords_xy[:, 1].min())
    y1 = float(coords_xy[:, 1].max())
    xs = np.linspace(x0, x1, cols, dtype=np.float32)
    ys = np.linspace(y0, y1, rows, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)[:num_points]

    selected: list[np.ndarray] = []
    used = np.zeros((coords_xy.shape[0],), dtype=bool)
    for target in grid:
        diff = coords_xy - target[None, :]
        dist = (diff[:, 0] ** 2) + (diff[:, 1] ** 2)
        dist[used] = np.inf
        idx = int(np.argmin(dist))
        if not np.isfinite(dist[idx]):
            idx = int(np.argmin((coords_xy - target[None, :])[:, 0] ** 2 + (coords_xy - target[None, :])[:, 1] ** 2))
        used[idx] = True
        selected.append(coords_xy[idx])
    return np.stack(selected, axis=0).astype(np.float32)


def sample_points_from_mask(mask_hw: np.ndarray, num_points: int, *, avoid_edges: bool = True) -> np.ndarray:
    mask_u8 = (mask_hw > 0).astype(np.uint8)
    ys, xs = np.where(mask_u8 > 0)
    if xs.size == 0 or ys.size == 0 or num_points <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    coords_xy = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=-1)
    if not avoid_edges:
        return _sample_points_from_coords(coords_xy, num_points)

    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    if max_dist <= 1e-6:
        return _sample_points_from_coords(coords_xy, num_points)

    safe_thresh = max(1.5, 0.25 * max_dist)
    safe_mask = dist >= safe_thresh
    safe_ys, safe_xs = np.where(safe_mask)
    if safe_xs.size == 0 or safe_ys.size == 0:
        return _sample_points_from_coords(coords_xy, num_points)
    safe_coords_xy = np.stack([safe_xs.astype(np.float32), safe_ys.astype(np.float32)], axis=-1)
    return _sample_points_from_coords(safe_coords_xy, num_points)


def _extract_mask_components(mask_hw: np.ndarray) -> list[dict[str, np.ndarray | float]]:
    mask_u8 = (mask_hw > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    total_fg_area = max(int(mask_u8.sum()), 1)
    min_area = max(12, int(round(total_fg_area * 0.02)))
    components: list[dict[str, np.ndarray | float]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_mask = (labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        safe_thresh = max(1.5, 0.25 * max_dist)
        safe_mask = dist >= safe_thresh
        safe_area = float(safe_mask.sum())
        confidence = safe_area / max(float(area), 1.0)
        score = max(float(area) * max(confidence, 1.0e-3), 1.0)
        x, y, w, h = cv2.boundingRect(comp_mask)
        box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
        components.append(
            {
                "mask": comp_mask,
                "box": box,
                "area": float(area),
                "confidence": float(confidence),
                "score": float(score),
            }
        )
    return sorted(components, key=lambda item: float(item["score"]), reverse=True)


def _allocate_queries_per_component(components: list[dict[str, np.ndarray | float]], num_queries: int) -> list[int]:
    if num_queries <= 0 or not components:
        return []
    if len(components) > num_queries:
        components[:] = components[:num_queries]

    scores = np.asarray([float(item["score"]) for item in components], dtype=np.float32)
    scores = np.maximum(scores, 1.0e-6)
    alloc = np.zeros((len(components),), dtype=np.int64)
    rank = np.argsort(-scores)
    for idx in rank[: min(len(components), num_queries)]:
        alloc[idx] = 1
    remaining = int(num_queries - alloc.sum())
    if remaining <= 0:
        return alloc.tolist()

    fractional = (scores / scores.sum()) * float(remaining)
    base = np.floor(fractional).astype(np.int64)
    alloc += base
    remaining = int(num_queries - alloc.sum())
    if remaining > 0:
        residual = fractional - base.astype(np.float32)
        for idx in np.argsort(-residual)[:remaining]:
            alloc[idx] += 1
    return alloc.tolist()


def build_vggt_query_prior(
    sam_masks_thw: np.ndarray,
    sam_boxes_t4: np.ndarray,
    *,
    num_queries: int,
) -> tuple[np.ndarray, str]:
    frame0_mask = sam_masks_thw[0]
    components = _extract_mask_components(frame0_mask)
    if components:
        alloc = _allocate_queries_per_component(components, num_queries)
        sampled = []
        for component, comp_queries in zip(components, alloc):
            if comp_queries <= 0:
                continue
            comp_points = sample_points_from_mask(component["mask"], comp_queries, avoid_edges=True)
            if comp_points.shape[0] > 0:
                sampled.append(comp_points.astype(np.float32))
        if sampled:
            query_points = np.concatenate(sampled, axis=0)
            if query_points.shape[0] < num_queries:
                top_up = sample_points_from_mask(frame0_mask, num_queries - query_points.shape[0], avoid_edges=True)
                if top_up.shape[0] > 0:
                    query_points = np.concatenate([query_points, top_up.astype(np.float32)], axis=0)
            query_points = query_points[:num_queries]
            if query_points.shape[0] == num_queries:
                return query_points.astype(np.float32), f"sam_mask_frame0_components{len(components)}"

    query_points = sample_points_from_mask(frame0_mask, num_queries, avoid_edges=True)
    if query_points.shape[0] == num_queries and np.any(frame0_mask > 0):
        return query_points.astype(np.float32), "sam_mask_frame0"
    box0 = sam_boxes_t4[0]
    return sample_points_from_box(box0, num_queries), "sam_box_frame0"
