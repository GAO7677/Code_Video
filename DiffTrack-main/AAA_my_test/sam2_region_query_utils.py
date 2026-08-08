#!/usr/bin/env python3
"""Shared SAM2 region cache and query-sampling utilities."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CACHE_ROOT = Path("/data/gaoya/agent-data/cache/toydataset_sam2_regions")
QUERY_CONTEXT_FRAME = 4
TOKEN_STRIDE = 32


@dataclass(frozen=True)
class QueryRegion:
    region_name: str
    region_type: str
    region_phrase: str | None
    region_slot: int | None
    point_start: int
    point_end: int
    mask_area: int
    source_mask_frame: int
    used_frame_fallback: bool


@dataclass
class RegionQueryCache:
    case_key: str
    query_points: np.ndarray
    masks_rhw: np.ndarray
    regions: list[QueryRegion]
    context_frame_rgb: np.ndarray
    metadata: dict[str, Any]


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def farthest_point_sample(mask: np.ndarray, count: int) -> np.ndarray:
    """Deterministically spread query points over the valid mask interior."""
    yx = np.argwhere(np.asarray(mask, dtype=bool))
    if len(yx) < count:
        raise ValueError(f"mask has {len(yx)} valid pixels, cannot sample {count}")
    center = yx.mean(axis=0)
    selected = [int(np.argmin(np.square(yx - center).sum(axis=1)))]
    min_distance = np.square(yx - yx[selected[0]]).sum(axis=1).astype(np.float64)
    for _ in range(1, count):
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.square(yx - yx[next_index]).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
    return yx[np.asarray(selected)][:, ::-1].astype(np.float32)


def _nearest_nonempty_mask(masks_thw: np.ndarray, frame_index: int) -> tuple[np.ndarray, int]:
    masks = np.asarray(masks_thw) > 0
    if masks.ndim != 3:
        raise ValueError(f"expected SAM2 masks [T,H,W], got {masks.shape}")
    frame_index = min(max(int(frame_index), 0), len(masks) - 1)
    if masks[frame_index].any():
        return masks[frame_index], frame_index
    candidates = [index for index, mask in enumerate(masks) if mask.any()]
    if not candidates:
        raise ValueError("SAM2 track has no non-empty mask")
    selected = min(candidates, key=lambda index: (abs(index - frame_index), index))
    return masks[selected], int(selected)


def _ordered_tracks(grounding_sample: Any, object_phrases: list[str]) -> list[Any]:
    """Restore dataset phrase order after provider score-based deduplication."""
    remaining = list(grounding_sample.object_tracks)
    ordered: list[Any] = []
    for phrase in object_phrases:
        normalized = phrase.strip().lower()
        match_index = next(
            (
                index
                for index, track in enumerate(remaining)
                if str(getattr(track, "source_phrase", "") or "").strip().lower() == normalized
            ),
            None,
        )
        if match_index is None:
            match_index = next(
                (
                    index
                    for index, track in enumerate(remaining)
                    if str(getattr(track, "phrase", "") or "").strip().lower() == normalized
                ),
                None,
            )
        if match_index is None:
            continue
        ordered.append(remaining.pop(match_index))
    ordered.extend(remaining)
    return ordered


def build_regions_from_grounding(
    *,
    case_key: str,
    grounding_sample: Any,
    object_phrases: list[str],
    context_frame_rgb: np.ndarray,
    query_frame_index: int = QUERY_CONTEXT_FRAME,
    points_per_region: int = 8,
    object_erode_px: int = 11,
    background_erode_px: int = 31,
) -> RegionQueryCache:
    if points_per_region <= 0:
        raise ValueError("points_per_region must be positive")
    frame = np.asarray(context_frame_rgb, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected RGB frame [H,W,3], got {frame.shape}")
    height, width = frame.shape[:2]
    tracks = _ordered_tracks(grounding_sample, object_phrases)
    if len(tracks) != len(object_phrases):
        detected = [
            str(getattr(track, "source_phrase", None) or getattr(track, "phrase", ""))
            for track in tracks
        ]
        raise RuntimeError(
            f"{case_key}: expected {len(object_phrases)} objects but grounding produced "
            f"{len(tracks)} tracks: {detected}"
        )

    raw_masks: list[np.ndarray] = []
    source_frames: list[int] = []
    for track in tracks:
        mask, source_frame = _nearest_nonempty_mask(track.masks_thw, query_frame_index)
        if mask.shape != (height, width):
            raise ValueError(
                f"{case_key}: SAM2 mask {mask.shape} does not match frame {(height, width)}"
            )
        raw_masks.append(mask)
        source_frames.append(source_frame)

    # Resolve rare overlap deterministically so each pixel belongs to at most one object.
    assigned = np.zeros((height, width), dtype=bool)
    object_masks: list[np.ndarray] = []
    for mask in raw_masks:
        exclusive = np.asarray(mask, dtype=bool) & ~assigned
        assigned |= np.asarray(mask, dtype=bool)
        eroded = erode_mask(exclusive, object_erode_px)
        if eroded.sum() < points_per_region:
            eroded = exclusive
        if eroded.sum() < points_per_region:
            raise RuntimeError(
                f"{case_key}: object mask has only {int(eroded.sum())} pixels after overlap removal"
            )
        object_masks.append(eroded)

    # Background excludes every object, including object_C in three-object scenes.
    object_union = np.logical_or.reduce(raw_masks) if raw_masks else np.zeros((height, width), bool)
    background = erode_mask(~object_union, background_erode_px)
    border = max(int(background_erode_px), TOKEN_STRIDE)
    background[:border] = False
    background[-border:] = False
    background[:, :border] = False
    background[:, -border:] = False
    if background.sum() < points_per_region:
        raise RuntimeError(f"{case_key}: background has too few valid pixels")

    query_parts: list[np.ndarray] = []
    regions: list[QueryRegion] = []
    masks: list[np.ndarray] = []
    offset = 0
    for index, (phrase, mask, source_frame) in enumerate(
        zip(object_phrases, object_masks, source_frames)
    ):
        points = farthest_point_sample(mask, points_per_region)
        region_name = f"object_{chr(ord('A') + index)}"
        query_parts.append(points)
        masks.append(mask)
        regions.append(
            QueryRegion(
                region_name=region_name,
                region_type="object",
                region_phrase=phrase,
                region_slot=index,
                point_start=offset,
                point_end=offset + len(points),
                mask_area=int(mask.sum()),
                source_mask_frame=int(source_frame),
                used_frame_fallback=int(source_frame) != int(query_frame_index),
            )
        )
        offset += len(points)

    background_points = farthest_point_sample(background, points_per_region)
    query_parts.append(background_points)
    masks.append(background)
    regions.append(
        QueryRegion(
            region_name="background",
            region_type="background",
            region_phrase=None,
            region_slot=None,
            point_start=offset,
            point_end=offset + len(background_points),
            mask_area=int(background.sum()),
            source_mask_frame=int(query_frame_index),
            used_frame_fallback=False,
        )
    )
    query_points = np.concatenate(query_parts, axis=0).astype(np.float32)
    masks_rhw = np.stack(masks).astype(np.uint8)
    for region, mask in zip(regions, masks_rhw):
        points = query_points[region.point_start : region.point_end]
        xy = np.rint(points).astype(np.int64)
        if not np.all(mask[xy[:, 1], xy[:, 0]] > 0):
            raise AssertionError(f"{case_key}/{region.region_name}: sampled point outside mask")

    return RegionQueryCache(
        case_key=case_key,
        query_points=query_points,
        masks_rhw=masks_rhw,
        regions=regions,
        context_frame_rgb=frame,
        metadata={
            "case_key": case_key,
            "query_context_frame": int(query_frame_index),
            "points_per_region": int(points_per_region),
            "object_erode_px": int(object_erode_px),
            "background_erode_px": int(background_erode_px),
            "object_count": len(object_phrases),
            "region_count": len(regions),
            "grounding_debug": dict(getattr(grounding_sample, "debug", {}) or {}),
        },
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def save_region_cache(
    cache_dir: Path,
    cache: RegionQueryCache,
    *,
    save_visualizations: bool = True,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / "regions.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            query_points=cache.query_points,
            masks_rhw=cache.masks_rhw,
            context_frame_rgb=cache.context_frame_rgb,
        )
    os.replace(temporary, cache_dir / "regions.npz")
    payload = {
        **cache.metadata,
        "regions": [asdict(region) for region in cache.regions],
    }
    _atomic_write_json(cache_dir / "regions.json", payload)
    if save_visualizations:
        save_region_query_visualizations(cache_dir, cache)
    _atomic_write_json(cache_dir / "complete.json", {"case_key": cache.case_key})


def load_region_cache(cache_root: Path, case_key: str) -> RegionQueryCache:
    cache_dir = Path(cache_root) / case_key
    payload = json.loads((cache_dir / "regions.json").read_text(encoding="utf-8"))
    with np.load(cache_dir / "regions.npz") as arrays:
        query_points = arrays["query_points"].astype(np.float32)
        masks_rhw = arrays["masks_rhw"].astype(np.uint8)
        context_frame_rgb = arrays["context_frame_rgb"].astype(np.uint8)
    regions = [QueryRegion(**item) for item in payload["regions"]]
    if len(regions) != len(masks_rhw):
        raise ValueError(f"{case_key}: region metadata/mask count mismatch")
    if regions[-1].region_name != "background":
        raise ValueError(f"{case_key}: final cached region must be background")
    expected_points = regions[-1].point_end
    if expected_points != len(query_points):
        raise ValueError(f"{case_key}: region slices cover {expected_points}/{len(query_points)} points")
    return RegionQueryCache(
        case_key=case_key,
        query_points=query_points,
        masks_rhw=masks_rhw,
        regions=regions,
        context_frame_rgb=context_frame_rgb,
        metadata={key: value for key, value in payload.items() if key != "regions"},
    )


def _region_colors(count: int) -> list[tuple[int, int, int]]:
    palette = [
        (232, 73, 45),
        (26, 145, 182),
        (242, 174, 39),
        (60, 159, 85),
        (179, 80, 166),
    ]
    return [palette[index % len(palette)] for index in range(count)]


def save_region_query_visualizations(output_dir: Path, cache: RegionQueryCache) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = _region_colors(len(cache.regions))
    combined = cache.context_frame_rgb.copy()
    files: list[str] = []
    for index, (region, mask) in enumerate(zip(cache.regions, cache.masks_rhw)):
        color = np.asarray(colors[index], dtype=np.uint8)
        mask_bool = mask > 0
        combined[mask_bool] = (0.58 * combined[mask_bool] + 0.42 * color).astype(np.uint8)
        region_image = cache.context_frame_rgb.copy()
        region_image[mask_bool] = (0.52 * region_image[mask_bool] + 0.48 * color).astype(np.uint8)
        points = cache.query_points[region.point_start : region.point_end]
        for point_index, (x, y) in enumerate(points):
            center = (int(round(float(x))), int(round(float(y))))
            cv2.circle(region_image, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(region_image, center, 6, tuple(int(v) for v in color), 2, cv2.LINE_AA)
            cv2.putText(
                region_image,
                str(point_index),
                (center[0] + 7, center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(combined, center, 5, tuple(int(v) for v in color), -1, cv2.LINE_AA)
        label = region.region_name
        if region.region_phrase:
            label += f": {region.region_phrase}"
        cv2.putText(region_image, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(region_image, label, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        region_dir = output_dir / "regions" / region.region_name
        region_dir.mkdir(parents=True, exist_ok=True)
        relative = f"regions/{region.region_name}/mask_points.png"
        cv2.imwrite(str(output_dir / relative), cv2.cvtColor(region_image, cv2.COLOR_RGB2BGR))
        files.append(relative)
    cv2.putText(
        combined,
        f"SAM2 masks and queries | context frame {cache.metadata['query_context_frame']}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        combined,
        f"SAM2 masks and queries | context frame {cache.metadata['query_context_frame']}",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
    )
    cv2.imwrite(str(output_dir / "sam2_regions_points.png"), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    return ["sam2_regions_points.png", *files]


def region_metadata(region: QueryRegion) -> dict[str, Any]:
    return {
        "region_name": region.region_name,
        "region_type": region.region_type,
        "region_phrase": region.region_phrase,
        "region_slot": region.region_slot,
        "point_start": region.point_start,
        "point_end": region.point_end,
    }
