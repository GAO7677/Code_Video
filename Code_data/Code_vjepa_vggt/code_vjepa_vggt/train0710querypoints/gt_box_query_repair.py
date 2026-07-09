from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable

import numpy as np
import torch

from code_vjepa_vggt.train0710querypoints.gt_mask_query_repair import CandidatePointScore
from code_vjepa_vggt.train0710querypoints.gt_mask_query_repair import _select_candidate_indices
from code_vjepa_vggt.utils.object_priors import sample_points_from_box


@dataclass(slots=True)
class GTBoxRepairConfig:
    enabled: bool = False
    oversample_factor: int = 4
    min_visible_ratio: float = 0.60
    min_in_box_ratio: float = 0.60


def _boxes_norm_to_px(boxes_tn4: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(image_hw[0]), int(image_hw[1])
    boxes = np.asarray(boxes_tn4, dtype=np.float32).copy()
    boxes[..., 0] *= float(width)
    boxes[..., 2] *= float(width)
    boxes[..., 1] *= float(height)
    boxes[..., 3] *= float(height)
    return boxes


def _valid_box(box_xyxy: np.ndarray) -> bool:
    return bool(float(box_xyxy[2] - box_xyxy[0]) > 1.0e-6 and float(box_xyxy[3] - box_xyxy[1]) > 1.0e-6)


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a.tolist()]
    bx0, by0, bx1, by1 = [float(v) for v in box_b.tolist()]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = max(area_a + area_b - inter, 1.0e-6)
    return float(inter / union)


def _point_inside_box(box_xyxy: np.ndarray, xy: np.ndarray) -> bool:
    x = float(xy[0])
    y = float(xy[1])
    return bool(x >= float(box_xyxy[0]) and x <= float(box_xyxy[2]) and y >= float(box_xyxy[1]) and y <= float(box_xyxy[3]))


def _choose_anchor_frame_from_boxes(boxes_t4: np.ndarray, preferred_frame: int) -> int:
    if 0 <= int(preferred_frame) < int(boxes_t4.shape[0]) and _valid_box(boxes_t4[int(preferred_frame)]):
        return int(preferred_frame)
    areas = np.maximum(boxes_t4[:, 2] - boxes_t4[:, 0], 0.0) * np.maximum(boxes_t4[:, 3] - boxes_t4[:, 1], 0.0)
    if int((areas > 0).sum()) == 0:
        return 0
    return int(np.argmax(areas))


def _sample_candidate_queries_from_box(
    boxes_t4: np.ndarray,
    *,
    anchor_frame: int,
    num_candidates: int,
) -> np.ndarray:
    anchor_box = boxes_t4[int(anchor_frame)].astype(np.float32)
    if not _valid_box(anchor_box):
        valid_boxes = [box for box in boxes_t4 if _valid_box(box)]
        if valid_boxes:
            anchor_box = np.asarray(valid_boxes[0], dtype=np.float32)
        else:
            return np.zeros((int(num_candidates), 2), dtype=np.float32)
    return sample_points_from_box(anchor_box, int(num_candidates)).astype(np.float32)


def _match_tracks_to_gt_boxes(
    *,
    track_boxes_tn4: list[np.ndarray],
    gt_boxes_tn4: np.ndarray,
) -> list[int | None]:
    matched: list[int | None] = []
    used_gt_ids: set[int] = set()
    gt_object_count = int(gt_boxes_tn4.shape[1])
    for track_boxes in track_boxes_tn4:
        best_gt = None
        best_score = 0.0
        for gt_idx in range(gt_object_count):
            if gt_idx in used_gt_ids:
                continue
            ious: list[float] = []
            for frame_idx in range(min(int(track_boxes.shape[0]), int(gt_boxes_tn4.shape[0]))):
                pred_box = track_boxes[frame_idx]
                gt_box = gt_boxes_tn4[frame_idx, gt_idx]
                if not _valid_box(pred_box) or not _valid_box(gt_box):
                    continue
                ious.append(_box_iou(pred_box, gt_box))
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


def _score_candidates_for_object_boxes(
    *,
    object_gt_boxes_t4: np.ndarray,
    query_points_xy: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
    all_gt_boxes_tn4: np.ndarray,
) -> list[CandidatePointScore]:
    object_visible_frames = int(sum(1 for box in object_gt_boxes_t4 if _valid_box(box)))
    scored: list[CandidatePointScore] = []
    for point_idx in range(int(query_points_xy.shape[0])):
        visible_frames = 0
        same_box_frames = 0
        other_object_frames = 0
        background_frames = 0
        for frame_idx in range(int(tracks_tk2.shape[0])):
            if not _valid_box(object_gt_boxes_t4[frame_idx]):
                continue
            if float(visibility_tk[frame_idx, point_idx]) <= 0.5:
                continue
            visible_frames += 1
            xy = tracks_tk2[frame_idx, point_idx]
            if _point_inside_box(object_gt_boxes_t4[frame_idx], xy):
                same_box_frames += 1
                continue
            hit_other = False
            for other_idx in range(int(all_gt_boxes_tn4.shape[1])):
                if _point_inside_box(all_gt_boxes_tn4[frame_idx, other_idx], xy):
                    hit_other = True
                    break
            if hit_other:
                other_object_frames += 1
            else:
                background_frames += 1
        visible_ratio = float(visible_frames) / max(float(object_visible_frames), 1.0)
        in_box_ratio = float(same_box_frames) / max(float(object_visible_frames), 1.0)
        retained_given_visible = float(same_box_frames) / max(float(visible_frames), 1.0)
        score_value = (
            4.0 * in_box_ratio
            + 2.5 * retained_given_visible
            + 1.0 * visible_ratio
            - 0.75 * (float(other_object_frames) / max(float(object_visible_frames), 1.0))
            - 0.35 * (float(background_frames) / max(float(object_visible_frames), 1.0))
        )
        scored.append(
            CandidatePointScore(
                point_idx=int(point_idx),
                score=float(score_value),
                visible_ratio=float(visible_ratio),
                in_mask_ratio=float(in_box_ratio),
                retained_given_visible=float(retained_given_visible),
                other_object_ratio=float(other_object_frames) / max(float(object_visible_frames), 1.0),
                background_ratio=float(background_frames) / max(float(object_visible_frames), 1.0),
            )
        )
    scored.sort(key=lambda item: (item.score, item.in_mask_ratio, item.retained_given_visible), reverse=True)
    return scored


def repair_grouped_queries_with_gt_boxes(
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
    config: GTBoxRepairConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not bool(config.enabled):
        return grouped_queries_px, {"applied": False, "reason": "disabled"}
    if "context_boxes" not in sample:
        return grouped_queries_px, {"applied": False, "reason": "missing_context_boxes"}

    gt_boxes_norm = torch.as_tensor(sample["context_boxes"]).detach().float().cpu().numpy()
    if gt_boxes_norm.ndim != 3 or gt_boxes_norm.shape[-1] != 4:
        return grouped_queries_px, {"applied": False, "reason": "invalid_context_boxes_shape"}
    gt_boxes_tn4 = _boxes_norm_to_px(gt_boxes_norm, image_hw=image_hw)

    valid_track_boxes: list[np.ndarray] = []
    valid_object_ids: list[int] = []
    for object_idx in range(min(int(object_valid_mask.shape[0]), len(object_tracks))):
        if float(object_valid_mask[object_idx]) <= 0.5:
            continue
        valid_track_boxes.append(np.asarray(object_tracks[object_idx].boxes_t4, dtype=np.float32))
        valid_object_ids.append(int(object_idx))
    if not valid_track_boxes:
        return grouped_queries_px, {"applied": False, "reason": "no_valid_tracks"}

    matched_gt_ids = _match_tracks_to_gt_boxes(track_boxes_tn4=valid_track_boxes, gt_boxes_tn4=gt_boxes_tn4)

    candidate_queries: list[np.ndarray] = []
    candidate_frame_ids: list[float] = []
    owner_slices: dict[int, tuple[int, int]] = {}
    oversample_count = int(points_per_object) * max(int(config.oversample_factor), 1)
    for local_idx, object_idx in enumerate(valid_object_ids):
        gt_idx = matched_gt_ids[local_idx]
        if gt_idx is None:
            continue
        gt_boxes_t4 = gt_boxes_tn4[:, int(gt_idx)]
        anchor_frame = _choose_anchor_frame_from_boxes(gt_boxes_t4, int(prompt_frame_idx))
        candidates = _sample_candidate_queries_from_box(
            gt_boxes_t4,
            anchor_frame=int(anchor_frame),
            num_candidates=int(oversample_count),
        )
        if candidates.shape[0] <= 0:
            continue
        start = sum(int(item.shape[0]) for item in candidate_queries)
        end = start + int(candidates.shape[0])
        candidate_queries.append(candidates.astype(np.float32))
        candidate_frame_ids.extend([float(anchor_frame)] * int(candidates.shape[0]))
        owner_slices[int(object_idx)] = (start, end)
    if not candidate_queries:
        return grouped_queries_px, {"applied": False, "reason": "no_gt_matched_candidates", "matched_gt_ids": matched_gt_ids}

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
    for local_idx, object_idx in enumerate(valid_object_ids):
        if int(object_idx) not in owner_slices:
            continue
        gt_idx = matched_gt_ids[local_idx]
        if gt_idx is None:
            continue
        start, end = owner_slices[int(object_idx)]
        scored = _score_candidates_for_object_boxes(
            object_gt_boxes_t4=gt_boxes_tn4[:, int(gt_idx)],
            query_points_xy=flat_queries[start:end],
            tracks_tk2=tracks_tk2[:, start:end],
            visibility_tk=visibility_tk[:, start:end],
            all_gt_boxes_tn4=gt_boxes_tn4,
        )
        selected_local = _select_candidate_indices(
            scored,
            points_per_object=int(points_per_object),
            min_visible_ratio=float(config.min_visible_ratio),
            min_in_mask_ratio=float(config.min_in_box_ratio),
        )
        if len(selected_local) < int(points_per_object):
            fallback = list(range(min(int(points_per_object), int(end - start))))
            selected_local = (selected_local + [idx for idx in fallback if idx not in selected_local])[: int(points_per_object)]
        repaired[int(object_idx)] = flat_queries[start:end][selected_local]
        repair_items.append(
            {
                "object_idx": int(object_idx),
                "gt_object_idx": int(gt_idx),
                "selected_local_indices": [int(v) for v in selected_local],
                "top_scores": [float(item.score) for item in scored[: int(points_per_object)]],
                "top_visible_ratios": [float(item.visible_ratio) for item in scored[: int(points_per_object)]],
                "top_in_box_ratios": [float(item.in_mask_ratio) for item in scored[: int(points_per_object)]],
            }
        )

    return repaired, {
        "applied": True,
        "valid_object_ids": valid_object_ids,
        "matched_gt_ids": matched_gt_ids,
        "repair_items": repair_items,
    }
