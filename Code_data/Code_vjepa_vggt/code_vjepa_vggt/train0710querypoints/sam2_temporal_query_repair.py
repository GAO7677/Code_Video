"""No-GT query repair using propagated SAM2 masks and CoTracker trajectories."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import cv2
import numpy as np
import torch

from code_vjepa_vggt.utils.object_priors import _extract_mask_components, sample_points_from_mask


@dataclass(frozen=True)
class TemporalQueryRepairConfig:
    oversample_factor: int = 4
    min_visible_ratio: float = 0.60
    min_in_mask_ratio: float = 0.60


@dataclass(frozen=True)
class PointQuality:
    query_x: float
    query_y: float
    visible_ratio: float
    in_mask_ratio: float
    retained_given_visible: float
    mean_mask_margin_px: float
    score: float


def _point_inside(mask: np.ndarray, xy: np.ndarray) -> bool:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x] > 0)


def _main_component(mask: np.ndarray, prompt_box: np.ndarray) -> np.ndarray | None:
    components = _extract_mask_components(mask)
    if not components:
        return None
    center = np.asarray(
        [0.5 * float(prompt_box[0] + prompt_box[2]), 0.5 * float(prompt_box[1] + prompt_box[3])],
        dtype=np.float32,
    )
    for component in components:
        component_mask = np.asarray(component["mask"], dtype=np.uint8)
        if _point_inside(component_mask, center):
            return component_mask
    return np.asarray(components[0]["mask"], dtype=np.uint8)


def _point_qualities(
    *,
    masks_thw: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
) -> list[PointQuality]:
    target_visible = np.asarray([(mask > 0).any() for mask in masks_thw], dtype=bool)
    distance_maps = [
        cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        if bool(mask.any())
        else np.zeros_like(mask, dtype=np.float32)
        for mask in masks_thw
    ]
    visible_areas = [mask.sum() for mask in masks_thw if mask.any()]
    scale = max(float(np.sqrt(np.median(visible_areas or [1]))), 1.0)
    qualities: list[PointQuality] = []
    for point_idx in range(tracks_tk2.shape[1]):
        visible = visibility_tk[:, point_idx] >= 0.5
        same_mask = 0
        margins: list[float] = []
        for frame_idx in range(tracks_tk2.shape[0]):
            if not target_visible[frame_idx] or not visible[frame_idx]:
                continue
            xy = tracks_tk2[frame_idx, point_idx]
            if _point_inside(masks_thw[frame_idx], xy):
                same_mask += 1
                x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
                margins.append(float(distance_maps[frame_idx][y, x]))
        visible_ratio = float(visible.mean())
        in_mask_ratio = float(same_mask) / max(float(target_visible.sum()), 1.0)
        retained = float(same_mask) / max(float((visible & target_visible).sum()), 1.0)
        margin = float(np.mean(margins)) if margins else 0.0
        qualities.append(
            PointQuality(
                query_x=float(tracks_tk2[0, point_idx, 0]),
                query_y=float(tracks_tk2[0, point_idx, 1]),
                visible_ratio=visible_ratio,
                in_mask_ratio=in_mask_ratio,
                retained_given_visible=retained,
                mean_mask_margin_px=margin,
                score=4.0 * in_mask_ratio + visible_ratio + min(margin / scale, 1.0),
            )
        )
    return qualities


def _select_diverse(
    qualities: list[PointQuality],
    *,
    count: int,
    min_visible_ratio: float,
    min_in_mask_ratio: float,
) -> list[int]:
    eligible = [
        idx
        for idx, item in enumerate(qualities)
        if item.visible_ratio >= min_visible_ratio and item.in_mask_ratio >= min_in_mask_ratio
    ]
    if len(eligible) < count:
        return []
    points = np.asarray([[item.query_x, item.query_y] for item in qualities], dtype=np.float32)
    selected = [max(eligible, key=lambda idx: qualities[idx].score)]
    diagonal = max(float(np.linalg.norm(np.ptp(points[eligible], axis=0))), 1.0)
    while len(selected) < count:
        remaining = [idx for idx in eligible if idx not in selected]
        selected.append(
            max(
                remaining,
                key=lambda idx: qualities[idx].score
                + 0.75
                * min(float(np.linalg.norm(points[idx] - points[chosen])) / diagonal for chosen in selected),
            )
        )
    return selected


def repair_grouped_queries_with_sam2_tracks(
    *,
    image_hw: tuple[int, int],
    frames_bthwc_01: torch.Tensor,
    grouped_queries_px: np.ndarray,
    object_valid_mask: np.ndarray,
    object_tracks: list[Any],
    prompt_frame_idx: int,
    points_per_object: int,
    run_cotracker: Callable[..., Any],
    config: TemporalQueryRepairConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Repair per-slot queries without raw GT masks or box fallback."""
    repaired = np.asarray(grouped_queries_px, dtype=np.float32).copy()
    repaired_valid = np.asarray(object_valid_mask, dtype=np.float32).copy()
    candidate_parts: list[np.ndarray] = []
    candidate_frame_ids: list[float] = []
    slot_slices: dict[int, tuple[int, int]] = {}
    slot_reports: dict[int, dict[str, Any]] = {}
    candidate_count = int(points_per_object) * max(int(config.oversample_factor), 1)

    for slot_idx in range(min(len(repaired_valid), len(object_tracks))):
        if repaired_valid[slot_idx] <= 0.5:
            continue
        track = object_tracks[slot_idx]
        masks_thw = np.asarray(track.masks_thw, dtype=np.uint8)
        anchor = min(max(int(prompt_frame_idx), 0), int(masks_thw.shape[0]) - 1)
        component = _main_component(masks_thw[anchor], np.asarray(track.box_prompt_xyxy))
        if component is None:
            repaired_valid[slot_idx] = 0.0
            slot_reports[slot_idx] = {"slot": slot_idx, "phrase": str(track.phrase), "valid": False, "reason": "empty_anchor_mask"}
            continue
        candidates = sample_points_from_mask(component, candidate_count, avoid_edges=True).astype(np.float32)
        if candidates.shape[0] < int(points_per_object):
            repaired_valid[slot_idx] = 0.0
            slot_reports[slot_idx] = {"slot": slot_idx, "phrase": str(track.phrase), "valid": False, "reason": "insufficient_anchor_candidates"}
            continue
        start = sum(part.shape[0] for part in candidate_parts)
        candidate_parts.append(candidates)
        candidate_frame_ids.extend([float(anchor)] * int(candidates.shape[0]))
        slot_slices[slot_idx] = (start, start + int(candidates.shape[0]))

    if not candidate_parts:
        return repaired, repaired_valid, {"applied": True, "valid_slot_count": 0, "slot_reports": list(slot_reports.values())}

    flat_candidates = np.concatenate(candidate_parts, axis=0)
    candidate_queries = torch.from_numpy(flat_candidates).unsqueeze(0).to(
        device=frames_bthwc_01.device, dtype=frames_bthwc_01.dtype
    )
    candidate_frame_tensor = torch.tensor(candidate_frame_ids, device=frames_bthwc_01.device, dtype=frames_bthwc_01.dtype).view(1, -1, 1)
    candidate_out = run_cotracker(
        frames_bthwc_01,
        query_points_prior=candidate_queries,
        query_frame_ids=candidate_frame_tensor,
        query_image_hw=image_hw,
    )
    tracks_tk2 = candidate_out.tracks[0].detach().float().cpu().numpy()
    visibility_tk = candidate_out.visibility[0].detach().float().cpu().numpy()

    for slot_idx, (start, end) in slot_slices.items():
        track = object_tracks[slot_idx]
        qualities = _point_qualities(
            masks_thw=np.asarray(track.masks_thw, dtype=np.uint8),
            tracks_tk2=tracks_tk2[:, start:end],
            visibility_tk=visibility_tk[:, start:end],
        )
        selected = _select_diverse(
            qualities,
            count=int(points_per_object),
            min_visible_ratio=float(config.min_visible_ratio),
            min_in_mask_ratio=float(config.min_in_mask_ratio),
        )
        report: dict[str, Any] = {
            "slot": int(slot_idx),
            "phrase": str(track.phrase),
            "valid": bool(selected),
            "candidate_count": int(end - start),
            "selected_ids": [int(item) for item in selected],
            "candidate": [asdict(item) for item in qualities],
        }
        if not selected:
            repaired_valid[slot_idx] = 0.0
            report["reason"] = "insufficient_temporal_candidates"
        else:
            repaired[slot_idx] = flat_candidates[start:end][selected]
        slot_reports[slot_idx] = report

    return repaired, repaired_valid, {
        "applied": True,
        "config": asdict(config),
        "valid_slot_count": int((repaired_valid > 0.5).sum()),
        "slot_reports": [slot_reports[idx] for idx in sorted(slot_reports)],
    }
