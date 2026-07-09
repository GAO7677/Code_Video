from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

import cv2
import numpy as np
import torch

from code_vjepa_vggt.utils.object_priors import _allocate_queries_per_component
from code_vjepa_vggt.utils.object_priors import _extract_mask_components
from code_vjepa_vggt.utils.object_priors import sample_points_from_box
from code_vjepa_vggt.utils.object_priors import sample_points_from_mask


@dataclass(slots=True)
class GTMaskRepairConfig:
    enabled: bool = False
    oversample_factor: int = 4
    min_visible_ratio: float = 0.60
    min_in_mask_ratio: float = 0.60
    color_tolerance: int = 18


def resolve_raw_sample_dir_from_video_path(video_path: str | Path | None) -> Path | None:
    if video_path is None:
        return None
    candidate = Path(video_path).expanduser().resolve()
    if not candidate.is_file():
        return None
    if candidate.name != "rgba.mp4":
        return None
    sample_dir = candidate.parent
    for required_name in ("rgba.mp4", "segmentation.mp4", "metadata.json"):
        if not (sample_dir / required_name).is_file():
            return None
    return sample_dir


def _read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return np.stack(frames, axis=0)


def _resize_mask(mask_hw: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(image_hw[0]), int(image_hw[1])
    resized = cv2.resize(mask_hw.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8)


def _decode_instance_masks(
    segmentation_rgb_thwc: np.ndarray,
    metadata: dict[str, Any],
    *,
    color_tolerance: int,
) -> tuple[np.ndarray, list[str]]:
    object_data = metadata.get("object_data", {})
    object_types = [str(item) for item in object_data.get("type", [])]
    color_map = metadata.get("segmentation_color_map", {})
    object_colors: list[np.ndarray] = []
    seg_ids = object_data.get("segmentation_id", [])
    for object_idx in range(len(object_types)):
        color = color_map.get(str(object_idx + 1))
        if color is None and object_idx < len(seg_ids):
            color = color_map.get(str(int(seg_ids[object_idx])))
        if color is None:
            raise KeyError(f"missing segmentation color for object_idx={object_idx}")
        object_colors.append(np.asarray(color, dtype=np.int16))

    num_frames, height, width, _ = segmentation_rgb_thwc.shape
    num_objects = len(object_colors)
    masks = np.zeros((num_frames, num_objects, height, width), dtype=np.uint8)
    frame_i16 = segmentation_rgb_thwc.astype(np.int16)
    for object_idx, color in enumerate(object_colors):
        diff = np.abs(frame_i16 - color[None, None, None, :])
        masks[:, object_idx] = (diff.max(axis=-1) <= int(color_tolerance)).astype(np.uint8)
    return masks, object_types


def _build_subset_gt_masks(
    *,
    sample: dict[str, Any],
    sample_dir: Path,
    image_hw: tuple[int, int],
    color_tolerance: int,
) -> tuple[np.ndarray, list[str]] | None:
    metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}
    sampled_frame_indices = metadata.get("sampled_frame_indices")
    context_frame_indices = sample.get("context_frame_indices")
    if not isinstance(sampled_frame_indices, list) or context_frame_indices is None:
        return None

    local_ctx_indices = [int(v) for v in torch.as_tensor(context_frame_indices).detach().cpu().tolist()]
    raw_ctx_indices = [
        int(sampled_frame_indices[idx])
        for idx in local_ctx_indices
        if 0 <= int(idx) < len(sampled_frame_indices)
    ]
    if not raw_ctx_indices:
        return None

    raw_metadata = json_load(sample_dir / "metadata.json")
    segmentation_rgb = _read_video_rgb(sample_dir / "segmentation.mp4")
    masks_tnhw_raw, object_types = _decode_instance_masks(
        segmentation_rgb,
        raw_metadata,
        color_tolerance=int(color_tolerance),
    )

    subset_masks: list[np.ndarray] = []
    for raw_idx in raw_ctx_indices:
        if not (0 <= int(raw_idx) < int(masks_tnhw_raw.shape[0])):
            return None
        frame_masks = masks_tnhw_raw[int(raw_idx)]
        resized = np.stack([_resize_mask(mask_hw, image_hw) for mask_hw in frame_masks], axis=0)
        subset_masks.append(resized)
    return np.stack(subset_masks, axis=0), object_types


def json_load(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    inter = float(np.logical_and(mask_a > 0, mask_b > 0).sum())
    union = float(np.logical_or(mask_a > 0, mask_b > 0).sum())
    if union <= 1.0e-6:
        return 0.0
    return inter / union


def _match_tracks_to_gt(
    *,
    track_masks_tnhw: list[np.ndarray],
    gt_masks_tnhw: np.ndarray,
) -> list[int | None]:
    matched: list[int | None] = []
    used_gt_ids: set[int] = set()
    gt_object_count = int(gt_masks_tnhw.shape[1])
    for track_masks in track_masks_tnhw:
        best_gt = None
        best_score = 0.0
        for gt_idx in range(gt_object_count):
            if gt_idx in used_gt_ids:
                continue
            ious: list[float] = []
            for frame_idx in range(min(int(track_masks.shape[0]), int(gt_masks_tnhw.shape[0]))):
                track_mask = track_masks[frame_idx]
                gt_mask = gt_masks_tnhw[frame_idx, gt_idx]
                if int(track_mask.sum()) <= 0 and int(gt_mask.sum()) <= 0:
                    continue
                ious.append(_mask_iou(track_mask, gt_mask))
            score = float(np.mean(ious)) if ious else 0.0
            if score > best_score:
                best_score = score
                best_gt = gt_idx
        if best_gt is not None and best_score >= 0.10:
            used_gt_ids.add(int(best_gt))
            matched.append(int(best_gt))
        else:
            matched.append(None)
    return matched


def _choose_anchor_frame(mask_thw: np.ndarray, preferred_frame: int) -> int:
    if 0 <= int(preferred_frame) < int(mask_thw.shape[0]) and int(mask_thw[int(preferred_frame)].sum()) > 0:
        return int(preferred_frame)
    areas = mask_thw.reshape(mask_thw.shape[0], -1).sum(axis=1)
    if int((areas > 0).sum()) == 0:
        return 0
    return int(np.argmax(areas))


def _sample_candidate_queries(
    mask_thw: np.ndarray,
    *,
    anchor_frame: int,
    num_candidates: int,
) -> np.ndarray:
    anchor_mask = mask_thw[int(anchor_frame)]
    if int(anchor_mask.sum()) <= 0:
        ys, xs = np.where(mask_thw.reshape(-1, *mask_thw.shape[1:]).sum(axis=0) > 0)
        if xs.size == 0 or ys.size == 0:
            return np.zeros((int(num_candidates), 2), dtype=np.float32)
        box = np.asarray([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)], dtype=np.float32)
        return sample_points_from_box(box, int(num_candidates)).astype(np.float32)

    components = _extract_mask_components(anchor_mask)
    if components:
        alloc = _allocate_queries_per_component(components, int(num_candidates))
        sampled: list[np.ndarray] = []
        for component, count in zip(components, alloc):
            if int(count) <= 0:
                continue
            points = sample_points_from_mask(component["mask"], int(count), avoid_edges=True)
            if points.shape[0] > 0:
                sampled.append(points.astype(np.float32))
        if sampled:
            merged = np.concatenate(sampled, axis=0)
            if merged.shape[0] >= int(num_candidates):
                return merged[: int(num_candidates)].astype(np.float32)
            top_up = sample_points_from_mask(anchor_mask, int(num_candidates - merged.shape[0]), avoid_edges=True)
            if top_up.shape[0] > 0:
                merged = np.concatenate([merged, top_up.astype(np.float32)], axis=0)
            if merged.shape[0] >= int(num_candidates):
                return merged[: int(num_candidates)].astype(np.float32)
    return sample_points_from_mask(anchor_mask, int(num_candidates), avoid_edges=True).astype(np.float32)


def _point_inside_mask(mask_hw: np.ndarray, xy: np.ndarray) -> bool:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if y < 0 or y >= int(mask_hw.shape[0]) or x < 0 or x >= int(mask_hw.shape[1]):
        return False
    return bool(mask_hw[y, x] > 0)


def _mask_margin(distance_hw: np.ndarray, xy: np.ndarray) -> float:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    if y < 0 or y >= int(distance_hw.shape[0]) or x < 0 or x >= int(distance_hw.shape[1]):
        return 0.0
    return float(distance_hw[y, x])


def _score_candidates_for_object(
    *,
    object_gt_masks_thw: np.ndarray,
    query_points_xy: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    all_gt_masks_tnhw: np.ndarray,
) -> list[tuple[int, float]]:
    object_visible_frames = int((object_gt_masks_thw.reshape(object_gt_masks_thw.shape[0], -1).sum(axis=1) > 0).sum())
    visible_areas = object_gt_masks_thw.reshape(object_gt_masks_thw.shape[0], -1).sum(axis=1)
    visible_areas = visible_areas[visible_areas > 0]
    object_scale = float(math.sqrt(float(np.median(visible_areas)))) if visible_areas.size > 0 else 1.0
    object_scale = max(object_scale, 1.0)
    distance_maps = [
        cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5) if int(mask.sum()) > 0 else np.zeros_like(mask, dtype=np.float32)
        for mask in object_gt_masks_thw
    ]
    scored: list[tuple[int, float]] = []
    for point_idx in range(int(query_points_xy.shape[0])):
        visible_frames = 0
        same_mask_frames = 0
        other_object_frames = 0
        background_frames = 0
        margins: list[float] = []
        for frame_idx in range(int(tracks_tk2.shape[0])):
            if int(object_gt_masks_thw[frame_idx].sum()) <= 0:
                continue
            if float(visibility_tk[frame_idx, point_idx]) <= 0.5:
                continue
            visible_frames += 1
            xy = tracks_tk2[frame_idx, point_idx]
            if _point_inside_mask(object_gt_masks_thw[frame_idx], xy):
                same_mask_frames += 1
                margins.append(_mask_margin(distance_maps[frame_idx], xy))
                continue
            hit_other = False
            for other_idx in range(int(all_gt_masks_tnhw.shape[1])):
                if _point_inside_mask(all_gt_masks_tnhw[frame_idx, other_idx], xy):
                    hit_other = True
                    break
            if hit_other:
                other_object_frames += 1
            else:
                background_frames += 1
        visible_ratio = float(visible_frames) / max(float(object_visible_frames), 1.0)
        in_mask_ratio = float(same_mask_frames) / max(float(object_visible_frames), 1.0)
        retained_given_visible = float(same_mask_frames) / max(float(visible_frames), 1.0)
        mean_mask_margin = float(np.mean(margins)) if margins else 0.0
        norm_margin = min(mean_mask_margin / object_scale, 1.0)
        score_value = (
            4.0 * in_mask_ratio
            + 2.5 * retained_given_visible
            + 1.0 * visible_ratio
            + 0.25 * norm_margin
            - 0.75 * (float(other_object_frames) / max(float(object_visible_frames), 1.0))
            - 0.35 * (float(background_frames) / max(float(object_visible_frames), 1.0))
        )
        scored.append((int(point_idx), float(score_value)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def repair_grouped_queries_with_gt_masks(
    *,
    sample: dict[str, Any],
    image_hw: tuple[int, int],
    frames_bthwc_01: torch.Tensor,
    grouped_queries_px: np.ndarray,
    object_valid_mask: np.ndarray,
    object_tracks: list[Any],
    prompt_frame_idx: int,
    points_per_object: int,
    run_cotracker: Callable[..., Any],
    config: GTMaskRepairConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bool(config.enabled):
        return grouped_queries_px, {"applied": False, "reason": "disabled"}

    sample_dir = resolve_raw_sample_dir_from_video_path(sample.get("video_path"))
    if sample_dir is None:
        return grouped_queries_px, {"applied": False, "reason": "no_raw_sample_dir"}

    gt_payload = _build_subset_gt_masks(
        sample=sample,
        sample_dir=sample_dir,
        image_hw=image_hw,
        color_tolerance=int(config.color_tolerance),
    )
    if gt_payload is None:
        return grouped_queries_px, {"applied": False, "reason": "gt_mask_subset_unavailable"}
    gt_masks_tnhw, gt_object_types = gt_payload

    valid_track_masks: list[np.ndarray] = []
    valid_object_ids: list[int] = []
    for object_idx in range(min(int(object_valid_mask.shape[0]), len(object_tracks))):
        if float(object_valid_mask[object_idx]) <= 0.5:
            continue
        valid_track_masks.append(np.asarray(object_tracks[object_idx].masks_thw, dtype=np.uint8))
        valid_object_ids.append(int(object_idx))
    if not valid_track_masks:
        return grouped_queries_px, {"applied": False, "reason": "no_valid_tracks"}

    matched_gt_ids = _match_tracks_to_gt(track_masks_tnhw=valid_track_masks, gt_masks_tnhw=gt_masks_tnhw)

    candidate_queries: list[np.ndarray] = []
    candidate_frame_ids: list[float] = []
    candidate_owner: list[int] = []
    owner_slices: dict[int, tuple[int, int]] = {}
    oversample_count = int(points_per_object) * max(int(config.oversample_factor), 1)
    for local_idx, object_idx in enumerate(valid_object_ids):
        gt_idx = matched_gt_ids[local_idx]
        if gt_idx is None:
            continue
        gt_masks_thw = gt_masks_tnhw[:, int(gt_idx)]
        anchor_frame = _choose_anchor_frame(gt_masks_thw, int(prompt_frame_idx))
        candidates = _sample_candidate_queries(gt_masks_thw, anchor_frame=int(anchor_frame), num_candidates=int(oversample_count))
        if candidates.shape[0] <= 0:
            continue
        start = len(candidate_owner)
        end = start + int(candidates.shape[0])
        candidate_queries.append(candidates.astype(np.float32))
        candidate_frame_ids.extend([float(anchor_frame)] * int(candidates.shape[0]))
        candidate_owner.extend([int(object_idx)] * int(candidates.shape[0]))
        owner_slices[int(object_idx)] = (start, end)
    if not candidate_queries:
        return grouped_queries_px, {"applied": False, "reason": "no_gt_matched_candidates"}

    flat_queries = np.concatenate(candidate_queries, axis=0).astype(np.float32)
    query_points_prior = torch.from_numpy(flat_queries).unsqueeze(0).to(device=frames_bthwc_01.device, dtype=frames_bthwc_01.dtype)
    query_frame_ids = torch.tensor(candidate_frame_ids, dtype=frames_bthwc_01.dtype, device=frames_bthwc_01.device).view(1, -1, 1)
    cot_out = run_cotracker(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_frame_ids=query_frame_ids,
        query_image_hw=image_hw,
    )
    tracks_tk2 = cot_out.tracks[0].detach().float().cpu().numpy()
    visibility_tk = cot_out.visibility[0].detach().float().cpu().numpy()

    repaired = np.asarray(grouped_queries_px, dtype=np.float32).copy()
    repair_items: list[dict[str, Any]] = []
    gt_by_object = {object_idx: matched_gt_ids[valid_object_ids.index(object_idx)] for object_idx in valid_object_ids if object_idx in valid_object_ids}
    for local_idx, object_idx in enumerate(valid_object_ids):
        if int(object_idx) not in owner_slices:
            continue
        gt_idx = matched_gt_ids[local_idx]
        if gt_idx is None:
            continue
        start, end = owner_slices[int(object_idx)]
        scored = _score_candidates_for_object(
            object_gt_masks_thw=gt_masks_tnhw[:, int(gt_idx)],
            query_points_xy=flat_queries[start:end],
            tracks_tk2=tracks_tk2[:, start:end],
            visibility_tk=visibility_tk[:, start:end],
            all_gt_masks_tnhw=gt_masks_tnhw,
        )
        selected_local = [idx for idx, _ in scored[: int(points_per_object)]]
        if len(selected_local) < int(points_per_object):
            fallback = list(range(min(int(points_per_object), int(end - start))))
            selected_local = (selected_local + [idx for idx in fallback if idx not in selected_local])[: int(points_per_object)]
        repaired[int(object_idx)] = flat_queries[start:end][selected_local]
        repair_items.append(
            {
                "object_idx": int(object_idx),
                "gt_object_idx": int(gt_idx),
                "gt_object_type": gt_object_types[int(gt_idx)] if int(gt_idx) < len(gt_object_types) else "",
                "selected_local_indices": [int(v) for v in selected_local],
                "top_scores": [float(score) for _, score in scored[: int(points_per_object)]],
            }
        )

    return repaired, {
        "applied": True,
        "sample_dir": str(sample_dir),
        "valid_object_ids": valid_object_ids,
        "matched_gt_ids": matched_gt_ids,
        "repair_items": repair_items,
    }
