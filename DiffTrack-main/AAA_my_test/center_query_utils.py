#!/usr/bin/env python3
"""Select one existing query nearest each SAM2 region center."""

from __future__ import annotations

from pathlib import Path

import numpy as np


DEFAULT_CACHE = Path(
    "/data/gaoya/agent-data/cache/physiciq_selected_sam2_regions"
)


def select_center_queries(
    cache_root: Path, case_key: str, regions: list[dict]
) -> dict[str, dict]:
    cache_path = cache_root / case_key / "regions.npz"
    cache = np.load(cache_path)
    query_points = cache["query_points"]
    masks = cache["masks_rhw"].astype(bool)
    height, width = masks.shape[1:]
    selected = {}

    for region in regions:
        name = region["region_name"]
        start = int(region["point_start"])
        end = int(region["point_end"])
        candidates = query_points[start:end]
        slot = region.get("region_slot")
        if slot is None:
            target = np.array([width / 2.0, height / 2.0], dtype=np.float32)
            mask_slot = len(masks) - 1
        else:
            mask_slot = int(slot)
            ys, xs = np.nonzero(masks[mask_slot])
            if not len(xs):
                raise RuntimeError(f"empty mask for {case_key}/{name}")
            target = np.array([xs.mean(), ys.mean()], dtype=np.float32)

        local_index = int(np.linalg.norm(candidates - target, axis=1).argmin())
        global_index = start + local_index
        point = query_points[global_index]
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        if not masks[mask_slot, y, x]:
            raise RuntimeError(f"selected center query is outside {case_key}/{name}")
        selected[name] = {
            "global_index": global_index,
            "local_index": local_index,
            "target_xy": target.tolist(),
            "query_xy_cache": point.tolist(),
            "distance_to_target_px": float(np.linalg.norm(point - target)),
        }
    return selected
