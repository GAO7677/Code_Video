from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt.adapters.sam2_motion import (
    GroundingDINOTextDetector,
    SAM2MotionTracker,
    build_motion_prompt_box,
    build_motion_prompt_boxes,
)
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data.ball_block_dataset import BallBlockVideoDataset
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior
from code_vjepa_vggt.utils.track_correction import project_tracks_to_object_masks
from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes


OBJECT_COLORS = [
    (214, 40, 40),
    (247, 127, 0),
    (252, 191, 73),
    (42, 157, 143),
    (39, 125, 161),
    (106, 76, 147),
]
QUERY_COLORS = [
    (0, 180, 216),
    (0, 119, 182),
    (131, 56, 236),
    (58, 134, 255),
    (255, 0, 110),
    (251, 86, 7),
    (46, 196, 182),
    (138, 201, 38),
]
GT_COLOR = (255, 255, 255)


@dataclass
class ObjectTrack:
    box_prompt_xyxy: np.ndarray
    masks_thw: np.ndarray
    boxes_t4: np.ndarray
    score: float
    phrase: str


def box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a.tolist()]
    bx0, by0, bx1, by1 = [float(v) for v in box_b.tolist()]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = max(area_a + area_b - inter, 1.0e-6)
    return float(inter / union)


def box_containment_ratio_xyxy(container_box: np.ndarray, inner_box: np.ndarray) -> float:
    cx0, cy0, cx1, cy1 = [float(v) for v in container_box.tolist()]
    ix0, iy0, ix1, iy1 = [float(v) for v in inner_box.tolist()]
    inter_x0 = max(cx0, ix0)
    inter_y0 = max(cy0, iy0)
    inter_x1 = min(cx1, ix1)
    inter_y1 = min(cy1, iy1)
    inter = max(0.0, inter_x1 - inter_x0) * max(0.0, inter_y1 - inter_y0)
    inner_area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inner_area <= 1.0e-6:
        return 0.0
    return float(inter / inner_area)


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def write_mp4(path: Path, frames_thwc_uint8: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames_thwc_uint8.shape[1]), int(frames_thwc_uint8.shape[2])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames_thwc_uint8:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def colorize_scalar_video(video_thw: np.ndarray) -> np.ndarray:
    values = np.asarray(video_thw, dtype=np.float32)
    valid = np.isfinite(values)
    if not np.any(valid):
        zeros = np.zeros(values.shape + (3,), dtype=np.uint8)
        return zeros
    lo = float(np.nanpercentile(values[valid], 5.0))
    hi = float(np.nanpercentile(values[valid], 95.0))
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((values - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    frames = []
    for frame_hw in norm:
        heat_bgr = cv2.applyColorMap((frame_hw * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        frames.append(cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def colorize_world_points_video(world_points_thwc: np.ndarray) -> np.ndarray:
    pts = np.asarray(world_points_thwc, dtype=np.float32)
    rgb = np.zeros_like(pts, dtype=np.float32)
    for channel in range(min(3, pts.shape[-1])):
        values = pts[..., channel]
        valid = np.isfinite(values)
        if not np.any(valid):
            continue
        lo = float(np.nanpercentile(values[valid], 5.0))
        hi = float(np.nanpercentile(values[valid], 95.0))
        if hi <= lo:
            hi = lo + 1.0
        rgb[..., channel] = np.clip((values - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    return (rgb * 255.0).astype(np.uint8)


def draw_box_rgb(image: np.ndarray, box_xyxy_px: np.ndarray, color_rgb: tuple[int, int, int], label: str) -> None:
    x0, y0, x1, y1 = [int(round(v)) for v in box_xyxy_px.tolist()]
    if x1 <= x0 or y1 <= y0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.rectangle(image, (x0, y0), (x1, y1), color_bgr, 2)
    cv2.putText(image, label, (x0 + 2, max(y0 + 14, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)


def draw_point_rgb(image: np.ndarray, point_xy: np.ndarray, color_rgb: tuple[int, int, int], label: str, radius: int = 5) -> None:
    x, y = [int(round(v)) for v in point_xy.tolist()]
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    cv2.circle(image, (x, y), radius, color_bgr, 2)
    if label:
        cv2.putText(image, label, (x + 6, max(y - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)


def build_multi_object_prompt(caption: str) -> str:
    caption_lower = str(caption).lower()
    ordered = ["sphere", "ball", "block", "box", "cube", "cylinder", "capsule"]
    found = []
    for token in ordered:
        if token in caption_lower and token not in found:
            found.append(token)
    if not found:
        return caption
    return " . ".join(found) + " ."


def build_open_vocab_prompt(
    caption: str,
    *,
    text_prompt: str,
    extra_prompt_terms: str,
    include_caption_terms: bool,
) -> str:
    parts: list[str] = []
    if include_caption_terms:
        caption_prompt = build_multi_object_prompt(caption).strip()
        if caption_prompt:
            parts.extend([item.strip() for item in caption_prompt.split(".") if item.strip()])
    if extra_prompt_terms.strip():
        raw = extra_prompt_terms.replace(",", ".")
        parts.extend([item.strip() for item in raw.split(".") if item.strip()])
    if text_prompt.strip():
        raw = text_prompt.replace(",", ".")
        parts.extend([item.strip() for item in raw.split(".") if item.strip()])

    deduped: list[str] = []
    seen: set[str] = set()
    for item in parts:
        norm = item.lower()
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(item)
    return " . ".join(deduped) + (" ." if deduped else "")


def score_mask(mask_hw: np.ndarray) -> tuple[float, float]:
    mask_u8 = (mask_hw > 0).astype(np.uint8)
    area = float(mask_u8.sum())
    if area <= 0:
        return 0.0, 0.0
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    safe_thresh = max(1.5, 0.25 * max_dist)
    safe_area = float((dist >= safe_thresh).sum())
    confidence = safe_area / max(area, 1.0)
    return area, confidence


def score_track(track: ObjectTrack) -> float:
    if track.masks_thw.size <= 0:
        return max(float(track.score), 1.0e-6)
    areas = []
    confs = []
    for mask_hw in track.masks_thw:
        area, conf = score_mask(mask_hw)
        if area <= 0:
            continue
        areas.append(area)
        confs.append(conf)
    if not areas:
        return max(float(track.score), 1.0e-6)
    mean_area = float(np.mean(np.asarray(areas, dtype=np.float32)))
    mean_conf = float(np.mean(np.asarray(confs, dtype=np.float32)))
    return max(float(track.score), 1.0e-3) * max(mean_area, 1.0) * max(mean_conf, 1.0e-3)


def track_similarity_iou(track_a: ObjectTrack, track_b: ObjectTrack) -> float:
    prompt_iou = box_iou_xyxy(track_a.box_prompt_xyxy.astype(np.float32), track_b.box_prompt_xyxy.astype(np.float32))
    if track_a.boxes_t4.shape[0] <= 0 or track_b.boxes_t4.shape[0] <= 0:
        return float(prompt_iou)
    length = min(int(track_a.boxes_t4.shape[0]), int(track_b.boxes_t4.shape[0]))
    temporal_ious = [
        box_iou_xyxy(track_a.boxes_t4[t].astype(np.float32), track_b.boxes_t4[t].astype(np.float32))
        for t in range(length)
    ]
    if not temporal_ious:
        return float(prompt_iou)
    return float(max(prompt_iou, float(np.mean(np.asarray(temporal_ious, dtype=np.float32)))))


def dedupe_object_tracks(tracks: list[ObjectTrack], iou_threshold: float) -> list[ObjectTrack]:
    if len(tracks) <= 1:
        return tracks
    ranked = sorted(tracks, key=score_track, reverse=True)
    kept: list[ObjectTrack] = []
    for candidate in ranked:
        if any(track_similarity_iou(candidate, existing) >= float(iou_threshold) for existing in kept):
            continue
        kept.append(candidate)
    return kept


def suppress_container_tracks(
    tracks: list[ObjectTrack],
    *,
    containment_ratio_threshold: float,
    min_contained_tracks: int,
    min_area_ratio: float,
    small_track_iou_threshold: float,
) -> list[ObjectTrack]:
    if len(tracks) <= 1:
        return tracks

    keep_mask = [True] * len(tracks)
    mean_boxes = [np.mean(track.boxes_t4.astype(np.float32), axis=0) for track in tracks]
    mean_areas = [
        max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        for box in mean_boxes
    ]

    for big_idx, big_track in enumerate(tracks):
        if not keep_mask[big_idx]:
            continue
        big_box = mean_boxes[big_idx]
        big_area = mean_areas[big_idx]
        if big_area <= 1.0e-6:
            continue

        contained_indices: list[int] = []
        for small_idx, small_track in enumerate(tracks):
            if small_idx == big_idx or not keep_mask[small_idx]:
                continue
            small_box = mean_boxes[small_idx]
            small_area = mean_areas[small_idx]
            if small_area <= 1.0e-6:
                continue
            if big_area < (small_area * float(min_area_ratio)):
                continue
            containment = box_containment_ratio_xyxy(big_box.astype(np.float32), small_box.astype(np.float32))
            if containment >= float(containment_ratio_threshold):
                contained_indices.append(small_idx)

        unique_small_indices: list[int] = []
        for idx in contained_indices:
            if any(
                box_iou_xyxy(mean_boxes[idx].astype(np.float32), mean_boxes[kept_idx].astype(np.float32))
                >= float(small_track_iou_threshold)
                for kept_idx in unique_small_indices
            ):
                continue
            unique_small_indices.append(idx)

        if len(unique_small_indices) < int(min_contained_tracks):
            continue

        keep_mask[big_idx] = False

    return [track for track, keep in zip(tracks, keep_mask) if keep]


def allocate_queries(tracks: list[ObjectTrack], num_queries: int) -> list[int]:
    if not tracks or num_queries <= 0:
        return []
    scores = []
    for item in tracks:
        area, conf = score_mask(item.masks_thw[0])
        scores.append(max(area * max(conf, 1.0e-3) * max(item.score, 1.0e-3), 1.0e-6))
    scores_arr = np.asarray(scores, dtype=np.float32)
    alloc = np.zeros((len(tracks),), dtype=np.int64)
    order = np.argsort(-scores_arr)
    for idx in order[: min(len(tracks), num_queries)]:
        alloc[idx] = 1
    remaining = int(num_queries - alloc.sum())
    if remaining > 0:
        fractional = (scores_arr / scores_arr.sum()) * float(remaining)
        base = np.floor(fractional).astype(np.int64)
        alloc += base
        remaining = int(num_queries - alloc.sum())
        if remaining > 0:
            residual = fractional - base.astype(np.float32)
            for idx in np.argsort(-residual)[:remaining]:
                alloc[idx] += 1
    return alloc.tolist()


def allocate_queries_with_minimum(
    tracks: list[ObjectTrack],
    num_queries: int,
    min_queries_per_object: int,
) -> list[int]:
    if not tracks or num_queries <= 0:
        return []
    track_count = len(tracks)
    if track_count == 0:
        return []

    # If the requested minimum cannot be satisfied, spread queries as evenly as possible.
    if (track_count * min_queries_per_object) > num_queries:
        alloc = np.zeros((track_count,), dtype=np.int64)
        base = num_queries // track_count
        rem = num_queries % track_count
        alloc[:] = base
        if rem > 0:
            scores = []
            for item in tracks:
                area, conf = score_mask(item.masks_thw[0])
                scores.append(max(area * max(conf, 1.0e-3) * max(item.score, 1.0e-3), 1.0e-6))
            order = np.argsort(-np.asarray(scores, dtype=np.float32))
            for idx in order[:rem]:
                alloc[idx] += 1
        return alloc.tolist()

    alloc = np.full((track_count,), int(min_queries_per_object), dtype=np.int64)
    remaining = int(num_queries - alloc.sum())
    if remaining <= 0:
        return alloc.tolist()

    extra = np.asarray(allocate_queries(tracks, remaining), dtype=np.int64)
    alloc += extra
    return alloc.tolist()


def detect_and_track_objects(
    frames_tchw_01: np.ndarray,
    caption: str,
    *,
    sam2_device: str,
    gdino_device: str,
    max_objects: int,
    prompt_frame_mode: str,
    proposal_source: str,
    motion_score_ratio: float,
    text_prompt: str,
    extra_prompt_terms: str,
    include_caption_terms: bool,
    gdino_box_threshold: float,
    gdino_text_threshold: float,
    track_dedupe_iou_threshold: float,
    container_suppress_ratio_threshold: float,
    container_suppress_min_contained: int,
    container_suppress_min_area_ratio: float,
    container_suppress_small_iou_threshold: float,
) -> tuple[list[ObjectTrack], int]:
    def make_box_fallback_track(box_xyxy: np.ndarray, score: float, phrase: str) -> ObjectTrack | None:
        height, width = int(frames_tchw_01.shape[-2]), int(frames_tchw_01.shape[-1])
        fallback_box = np.asarray(box_xyxy, dtype=np.float32)
        x0, y0, x1, y1 = [int(round(v)) for v in fallback_box.tolist()]
        x0 = max(0, min(x0, width - 1))
        x1 = max(0, min(x1, width))
        y0 = max(0, min(y0, height - 1))
        y1 = max(0, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            return None
        masks = np.zeros((frames_tchw_01.shape[0], height, width), dtype=np.uint8)
        masks[:, y0:y1, x0:x1] = 1
        boxes = np.repeat(fallback_box[None, :], repeats=frames_tchw_01.shape[0], axis=0).astype(np.float32)
        return ObjectTrack(
            box_prompt_xyxy=fallback_box,
            masks_thw=masks,
            boxes_t4=boxes,
            score=float(score),
            phrase=str(phrase),
        )

    if prompt_frame_mode == "first":
        prompt_frame_idx = 0
    else:
        prompt_frame_idx = max(frames_tchw_01.shape[0] - 1, 0)

    proposal_source = str(proposal_source)
    motion_multi = build_motion_prompt_boxes(frames_tchw_01, max_boxes=max_objects)
    if proposal_source in {"motion_only", "motion_then_gdino"} and motion_multi.boxes_xyxy.shape[0] > 0:
        prompt_frame_idx = int(motion_multi.prompt_frame_idx)

    candidate_boxes: list[np.ndarray] = []
    candidate_scores: list[float] = []
    candidate_phrases: list[str] = []

    def add_candidate(box_xyxy: np.ndarray, score: float, phrase: str) -> None:
        box = np.asarray(box_xyxy, dtype=np.float32)
        if not np.all(np.isfinite(box)):
            return
        if float(box[2] - box[0]) <= 1.0 or float(box[3] - box[1]) <= 1.0:
            return
        for idx, existing in enumerate(candidate_boxes):
            if box_iou_xyxy(box, existing) >= 0.75:
                if float(score) > float(candidate_scores[idx]):
                    candidate_boxes[idx] = box
                    candidate_scores[idx] = float(score)
                    candidate_phrases[idx] = str(phrase)
                return
        if len(candidate_boxes) >= max_objects:
            return
        candidate_boxes.append(box)
        candidate_scores.append(float(score))
        candidate_phrases.append(str(phrase))

    def add_motion_candidates() -> None:
        motion_boxes = motion_multi.boxes_xyxy[:max_objects]
        motion_scores = motion_multi.scores[:max_objects]
        if motion_boxes.shape[0] == 0:
            return
        top_score = float(motion_scores.max()) if motion_scores.size > 0 else 0.0
        min_keep_score = max(1.0e-6, top_score * max(float(motion_score_ratio), 0.0))
        for idx, (box_xyxy, score) in enumerate(zip(motion_boxes, motion_scores)):
            if float(score) < min_keep_score:
                continue
            add_candidate(box_xyxy, float(score), f"motion_component_{idx}")

    def add_gdino_candidates() -> None:
        prompt_text = build_open_vocab_prompt(
            caption,
            text_prompt=text_prompt,
            extra_prompt_terms=extra_prompt_terms,
            include_caption_terms=include_caption_terms,
        )
        detector = GroundingDINOTextDetector(
            device=gdino_device,
            max_boxes=max_objects,
            box_threshold=float(gdino_box_threshold),
            text_threshold=float(gdino_text_threshold),
        )
        try:
            detection = detector.detect(frames_tchw_01[prompt_frame_idx], prompt_text, guidance_box_xyxy=None)
        except Exception as exc:
            print(f"[warn] GroundingDINO detect failed: {exc}")
            return
        for box_xyxy, score, phrase in zip(
            detection.boxes_xyxy[:max_objects],
            detection.scores[:max_objects],
            detection.phrases[:max_objects],
        ):
            add_candidate(box_xyxy, float(score), str(phrase))

    if proposal_source == "gdino_only":
        add_gdino_candidates()
    elif proposal_source == "motion_only":
        add_motion_candidates()
    elif proposal_source == "motion_then_gdino":
        add_motion_candidates()
        if len(candidate_boxes) < max_objects:
            add_gdino_candidates()
    elif proposal_source == "gdino_then_motion":
        add_gdino_candidates()
        if len(candidate_boxes) < max_objects:
            add_motion_candidates()
    else:
        raise ValueError(f"unsupported proposal_source={proposal_source}")

    if candidate_boxes:
        track_boxes = np.stack(candidate_boxes, axis=0).astype(np.float32)[:max_objects]
        track_scores = np.asarray(candidate_scores, dtype=np.float32)[:max_objects]
        track_phrases = candidate_phrases[:max_objects]
    else:
        track_boxes = np.zeros((0, 4), dtype=np.float32)
        track_scores = np.zeros((0,), dtype=np.float32)
        track_phrases = []

    if track_boxes.shape[0] == 0 and proposal_source != "gdino_only":
        motion_box = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=motion_multi.prompt_frame_idx)
        track_boxes = motion_box[None, :]
        track_scores = np.asarray([1.0], dtype=np.float32)
        track_phrases = ["motion_proxy"]
        prompt_frame_idx = int(motion_multi.prompt_frame_idx)

    tracker = SAM2MotionTracker(device=sam2_device, enable_text_prompt=False)
    outputs: list[ObjectTrack] = []
    for box_xyxy, score, phrase in zip(track_boxes, track_scores, track_phrases):
        try:
            sam_out = tracker.track(
                frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                prompt_box_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                caption="",
            )
        except Exception as exc:
            print(f"[warn] SAM2 track failed for phrase={phrase!r}, box={np.asarray(box_xyxy).tolist()}: {exc}")
            fallback_track = make_box_fallback_track(np.asarray(box_xyxy, dtype=np.float32), float(score), f"{phrase}_box_fallback")
            if fallback_track is not None:
                outputs.append(fallback_track)
            continue
        if int(sam_out.masks_thw[0].sum()) <= 0:
            fallback_track = make_box_fallback_track(np.asarray(box_xyxy, dtype=np.float32), float(score), f"{phrase}_empty_mask_fallback")
            if fallback_track is not None:
                outputs.append(fallback_track)
            continue
        outputs.append(
            ObjectTrack(
                box_prompt_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                masks_thw=sam_out.masks_thw.astype(np.uint8),
                boxes_t4=sam_out.boxes_t4.astype(np.float32),
                score=float(score),
                phrase=str(phrase),
            )
        )
    if not outputs and track_boxes.shape[0] > 0:
        fallback_track = make_box_fallback_track(
            np.asarray(track_boxes[0], dtype=np.float32),
            float(track_scores[0]) if len(track_scores) > 0 else 1.0,
            str(track_phrases[0]) if len(track_phrases) > 0 else "pseudo_box_track",
        )
        if fallback_track is not None:
            outputs.append(fallback_track)
    deduped_outputs = dedupe_object_tracks(outputs, iou_threshold=float(track_dedupe_iou_threshold))
    if len(deduped_outputs) < len(outputs):
        print(f"[info] deduped object tracks: {len(outputs)} -> {len(deduped_outputs)}")
    suppressed_outputs = suppress_container_tracks(
        deduped_outputs,
        containment_ratio_threshold=float(container_suppress_ratio_threshold),
        min_contained_tracks=int(container_suppress_min_contained),
        min_area_ratio=float(container_suppress_min_area_ratio),
        small_track_iou_threshold=float(container_suppress_small_iou_threshold),
    )
    if len(suppressed_outputs) < len(deduped_outputs):
        print(f"[info] suppressed container tracks: {len(deduped_outputs)} -> {len(suppressed_outputs)}")
    return suppressed_outputs, prompt_frame_idx


def build_query_prior_from_tracks(tracks: list[ObjectTrack], num_queries: int) -> tuple[np.ndarray, list[int], list[int], str]:
    if not tracks:
        return np.zeros((0, 2), dtype=np.float32), [], [], "none"
    alloc = allocate_queries(tracks, num_queries)
    query_sets = []
    owners: list[int] = []
    for obj_idx, (track, nq) in enumerate(zip(tracks, alloc)):
        if nq <= 0:
            continue
        pts, _ = build_vggt_query_prior(track.masks_thw, track.boxes_t4, num_queries=nq)
        if pts.shape[0] == 0:
            continue
        query_sets.append(pts.astype(np.float32))
        owners.extend([obj_idx] * int(pts.shape[0]))
    if not query_sets:
        pts, src = build_vggt_query_prior(tracks[0].masks_thw, tracks[0].boxes_t4, num_queries=num_queries)
        return pts.astype(np.float32), [0] * int(pts.shape[0]), [num_queries], src
    all_points = np.concatenate(query_sets, axis=0)[:num_queries]
    owners = owners[: int(all_points.shape[0])]
    return all_points.astype(np.float32), owners, alloc, f"multi_sam_objects{len(tracks)}"


def _fallback_points_from_box(box_xyxy: np.ndarray, num_queries: int) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box_xyxy.tolist()]
    if x1 <= x0 or y1 <= y0 or num_queries <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    if num_queries == 1:
        return np.asarray([[cx, cy]], dtype=np.float32)
    grid_side = int(np.ceil(np.sqrt(num_queries)))
    xs = np.linspace(x0 + 0.25 * (x1 - x0), x1 - 0.25 * (x1 - x0), grid_side)
    ys = np.linspace(y0 + 0.25 * (y1 - y0), y1 - 0.25 * (y1 - y0), grid_side)
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    center = np.asarray([[cx, cy]], dtype=np.float32)
    pts = np.concatenate([center, pts.astype(np.float32)], axis=0)
    return pts[:num_queries].astype(np.float32)


def build_query_prior_from_tracks_with_minimum(
    tracks: list[ObjectTrack],
    num_queries: int,
    min_queries_per_object: int,
) -> tuple[np.ndarray, list[int], list[int], str]:
    if not tracks:
        return np.zeros((0, 2), dtype=np.float32), [], [], "none"
    alloc = allocate_queries_with_minimum(tracks, num_queries, min_queries_per_object)
    query_sets = []
    owners: list[int] = []
    for obj_idx, (track, nq) in enumerate(zip(tracks, alloc)):
        if nq <= 0:
            continue
        pts, _ = build_vggt_query_prior(track.masks_thw, track.boxes_t4, num_queries=nq)
        if pts.shape[0] == 0:
            pts = _fallback_points_from_box(track.box_prompt_xyxy.astype(np.float32), nq)
            if pts.shape[0] == 0:
                continue
        query_sets.append(pts.astype(np.float32))
        owners.extend([obj_idx] * int(pts.shape[0]))
    if not query_sets:
        pts, src = build_vggt_query_prior(tracks[0].masks_thw, tracks[0].boxes_t4, num_queries=num_queries)
        if pts.shape[0] == 0:
            pts = _fallback_points_from_box(tracks[0].box_prompt_xyxy.astype(np.float32), num_queries)
            src = "fallback_prompt_box_grid"
        return pts.astype(np.float32), [0] * int(pts.shape[0]), alloc, src
    all_points = np.concatenate(query_sets, axis=0)[:num_queries]
    owners = owners[: int(all_points.shape[0])]
    return all_points.astype(np.float32), owners, alloc, f"multi_sam_objects{len(tracks)}_min{min_queries_per_object}"


def render_overlay_video(
    *,
    context_video: torch.Tensor,
    object_tracks: list[ObjectTrack],
    prompt_frame_idx: int,
    query_points_px_k2: np.ndarray,
    query_owner: list[int],
    tracks_native_tk2: torch.Tensor,
    tracks_corrected_tk2: torch.Tensor,
    correction_mask_tk: torch.Tensor,
    visibility_tk: torch.Tensor,
    gt_boxes: torch.Tensor | None = None,
    box_only_overlay: bool = False,
) -> np.ndarray:
    frames = []
    image_hw = (context_video.shape[-2], context_video.shape[-1])
    for t in range(context_video.shape[1]):
        frame = tensor_frame_to_uint8_hwc(context_video[:, t]).copy()
        if gt_boxes is not None:
            for obj_idx in range(gt_boxes.shape[1]):
                gt_box = gt_boxes[t, obj_idx]
                if not bool((gt_box[2] - gt_box[0] > 1e-6).item() and (gt_box[3] - gt_box[1] > 1e-6).item()):
                    continue
                scale = torch.tensor([image_hw[1], image_hw[0], image_hw[1], image_hw[0]], dtype=gt_box.dtype)
                draw_box_rgb(frame, (gt_box.cpu() * scale).numpy().astype(np.float32), GT_COLOR, f"gt{obj_idx}")

        for obj_idx, obj_track in enumerate(object_tracks):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            draw_box_rgb(frame, obj_track.boxes_t4[t].astype(np.float32), color, f"sam{obj_idx}")
            if t == prompt_frame_idx:
                draw_box_rgb(frame, obj_track.box_prompt_xyxy.astype(np.float32), color, f"prompt{obj_idx}")

        if not box_only_overlay:
            if t == 0:
                for q_idx, point in enumerate(query_points_px_k2):
                    owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
                    color = OBJECT_COLORS[owner % len(OBJECT_COLORS)] if owner >= 0 else (17, 17, 17)
                    draw_point_rgb(frame, point.astype(np.float32), color, f"q{q_idx}@o{owner}", radius=6)

            for q_idx in range(tracks_native_tk2.shape[1]):
                owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
                color = QUERY_COLORS[q_idx % len(QUERY_COLORS)] if owner < 0 else OBJECT_COLORS[owner % len(OBJECT_COLORS)]
                point = tracks_native_tk2[t, q_idx].cpu().numpy().astype(np.float32)
                label = f"q{q_idx}"
                if float(visibility_tk[t, q_idx].item()) < 0.5:
                    label += "(inv)"
                draw_point_rgb(frame, point, color, label)
                if float(correction_mask_tk[t, q_idx].item()) > 0.5:
                    draw_point_rgb(frame, tracks_corrected_tk2[t, q_idx].cpu().numpy().astype(np.float32), (255, 255, 0), f"fix{q_idx}", radius=4)
        frames.append(frame)
    return np.stack(frames, axis=0)


def evaluate_case(
    sample: dict,
    *,
    case_group: str,
    vggt_adapter: VGGTTrackAdapter,
    device: torch.device,
    sam2_device: str,
    gdino_device: str,
    output_dir: Path,
    min_queries_per_object: int,
    prompt_frame_mode: str,
    save_depth_video: bool,
    save_world_points_video: bool,
    max_objects: int,
    proposal_source: str,
    motion_score_ratio: float,
    text_prompt: str,
    extra_prompt_terms: str,
    include_caption_terms: bool,
    gdino_box_threshold: float,
    gdino_text_threshold: float,
    box_only_overlay: bool,
    track_dedupe_iou_threshold: float,
    container_suppress_ratio_threshold: float,
    container_suppress_min_contained: int,
    container_suppress_min_area_ratio: float,
    container_suppress_small_iou_threshold: float,
) -> dict:
    context_video = sample["context_video"]
    frames_tchw_01 = ((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy()
    object_tracks, prompt_frame_idx = detect_and_track_objects(
        frames_tchw_01,
        sample["caption"],
        sam2_device=sam2_device,
        gdino_device=gdino_device,
        max_objects=int(max_objects),
        prompt_frame_mode=prompt_frame_mode,
        proposal_source=str(proposal_source),
        motion_score_ratio=float(motion_score_ratio),
        text_prompt=str(text_prompt),
        extra_prompt_terms=str(extra_prompt_terms),
        include_caption_terms=bool(include_caption_terms),
        gdino_box_threshold=float(gdino_box_threshold),
        gdino_text_threshold=float(gdino_text_threshold),
        track_dedupe_iou_threshold=float(track_dedupe_iou_threshold),
        container_suppress_ratio_threshold=float(container_suppress_ratio_threshold),
        container_suppress_min_contained=int(container_suppress_min_contained),
        container_suppress_min_area_ratio=float(container_suppress_min_area_ratio),
        container_suppress_small_iou_threshold=float(container_suppress_small_iou_threshold),
    )
    query_points_px, query_owner, query_alloc, prior_source = build_query_prior_from_tracks_with_minimum(
        object_tracks,
        vggt_adapter.num_queries,
        min_queries_per_object,
    )
    query_points_prior = torch.from_numpy(query_points_px).unsqueeze(0).to(device=device, dtype=context_video.dtype)

    frames_bthwc = context_video.unsqueeze(0).permute(0, 2, 3, 4, 1).float()
    frames_bthwc = (frames_bthwc + 1.0) / 2.0
    with torch.no_grad():
        vggt_out = vggt_adapter(
            frames_bthwc.to(device),
            query_points_prior=query_points_prior,
            query_image_hw=(context_video.shape[-2], context_video.shape[-1]),
        )
    tracks = vggt_out.tracks
    track_image_hw = vggt_out.image_hw
    scale_x = float(context_video.shape[-1]) / float(track_image_hw[1])
    scale_y = float(context_video.shape[-2]) / float(track_image_hw[0])
    tracks_native = tracks.clone()
    tracks_native[..., 0] *= scale_x
    tracks_native[..., 1] *= scale_y
    tracks_corrected, correction_mask, correction_stats = project_tracks_to_object_masks(
        tracks_native[0].cpu(),
        [track.masks_thw for track in object_tracks],
        query_owner,
        avoid_edges=True,
    )
    tracks_corrected = tracks_corrected.unsqueeze(0).to(device=tracks_native.device, dtype=tracks_native.dtype)
    correction_mask = correction_mask.unsqueeze(0).to(device=tracks_native.device, dtype=tracks_native.dtype)

    gt_boxes = sample.get("context_boxes")
    metrics = None
    if gt_boxes is not None:
        alignment = align_tracks_to_boxes(
            tracks=tracks_native,
            gt_boxes=gt_boxes.unsqueeze(0).to(device),
            image_hw=(context_video.shape[-2], context_video.shape[-1]),
        )
        valid_mask = alignment.matched_gt_valid > 0.5
        l1 = (tracks_native - alignment.matched_gt_centers).abs().sum(dim=-1)
        metrics = {
            "mean_center_l1_px": float(l1[valid_mask].mean().item()) if valid_mask.any() else 0.0,
            "valid_track_points": int(valid_mask.sum().item()),
        }

    overlay = render_overlay_video(
        context_video=context_video,
        object_tracks=object_tracks,
        prompt_frame_idx=prompt_frame_idx,
        query_points_px_k2=query_points_px,
        query_owner=query_owner,
        tracks_native_tk2=tracks_native[0].cpu(),
        tracks_corrected_tk2=tracks_corrected[0].cpu(),
        correction_mask_tk=correction_mask[0].cpu(),
        visibility_tk=vggt_out.visibility[0].cpu(),
        gt_boxes=gt_boxes,
        box_only_overlay=bool(box_only_overlay),
    )
    raw_path = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__overlay.mp4"
    write_mp4(raw_path, overlay, fps=int(sample.get("_fps", 8)))

    depth_video_path = None
    world_points_video_path = None
    if save_depth_video and vggt_out.depth is not None:
        depth_frames = colorize_scalar_video(vggt_out.depth[0, ..., 0].detach().cpu().numpy())
        depth_video_path = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__vggt_depth.mp4"
        write_mp4(depth_video_path, depth_frames, fps=int(sample.get("_fps", 8)))
    if save_world_points_video and vggt_out.world_points is not None:
        world_frames = colorize_world_points_video(vggt_out.world_points[0].detach().cpu().numpy())
        world_points_video_path = output_dir / f"{case_group}__{Path(sample['video_path']).stem}__vggt_world_points.mp4"
        write_mp4(world_points_video_path, world_frames, fps=int(sample.get("_fps", 8)))

    return {
        "group": case_group,
        "caption": sample["caption"],
        "video_path": sample["video_path"],
        "object_count": len(object_tracks),
        "query_alloc": query_alloc,
        "query_owner": query_owner,
        "sam_prior_source": prior_source,
        "prompt_frame_idx": prompt_frame_idx,
        "min_queries_per_object": int(min_queries_per_object),
        "object_phrases": [item.phrase for item in object_tracks],
        "object_scores": [item.score for item in object_tracks],
        "overlay_video": str(raw_path.relative_to(output_dir.parent)),
        "depth_video": str(depth_video_path.relative_to(output_dir.parent)) if depth_video_path is not None else None,
        "world_points_video": str(world_points_video_path.relative_to(output_dir.parent)) if world_points_video_path is not None else None,
        "metrics": metrics,
        "correction_stats": {
            "corrected_points": correction_stats.corrected_points,
            "total_points": correction_stats.total_points,
            "snap_dist_mean_px": correction_stats.snap_dist_mean_px,
            "snap_dist_max_px": correction_stats.snap_dist_max_px,
        },
        "shapes": {
            "context_video": list(context_video.unsqueeze(0).shape),
            "query_points": list(query_points_prior.shape),
            "vggt_tracks": list(vggt_out.tracks.shape),
            "vggt_tracks_corrected": list(tracks_corrected.shape),
            "object_masks": [list(item.masks_thw.shape) for item in object_tracks],
        },
    }


def build_report(results: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks = []
    for idx, result in enumerate(results):
        metrics_line = ""
        if result["metrics"] is not None:
            metrics_line = f"<p><b>Metrics:</b> mean_center_l1_px={result['metrics']['mean_center_l1_px']:.2f}, valid_track_points={result['metrics']['valid_track_points']}</p>"
        blocks.append(
            f"""
  <section class="case">
    <h2>Case {idx} | {result['group']}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Video:</b> {result['video_path']}</p>
    <p><b>Detected SAM objects:</b> {result['object_count']} | <b>Prior source:</b> {result['sam_prior_source']}</p>
    <p><b>Object phrases:</b> {result['object_phrases']}</p>
    <p><b>Object scores:</b> {result['object_scores']}</p>
    <p><b>Query alloc per object:</b> {result['query_alloc']}</p>
    {metrics_line}
    <p><b>流程:</b> GroundingDINO 检测框 -> SAM2 单目标传播 -> 从 mask/box 采样 query points -> VGGT 输出 tracks + depth + world_points</p>
    <h3>Overlay</h3>
    <video controls preload="none" playsinline src="{result['overlay_video']}"></video>
    {"<h3>VGGT Depth</h3><video controls preload='none' playsinline src='%s'></video>" % result['depth_video'] if result['depth_video'] is not None else ""}
    {"<h3>VGGT World Points</h3><video controls preload='none' playsinline src='%s'></video>" % result['world_points_video'] if result['world_points_video'] is not None else ""}
    <pre>{json.dumps({'shapes': result['shapes'], 'query_owner': result['query_owner'], 'prompt_frame_idx': result['prompt_frame_idx']}, indent=2, ensure_ascii=False)}</pre>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>VGGT Multi-Object SAM Viewer</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>VGGT Multi-Object SAM Viewer</h1>
  <p>single-object case 来自 phys_state，带 GT；multi-object case 来自 ball_block，无 GT。每个物体先由 GroundingDINO 多框检测，再分别跑单目标 SAM2 传播，最后按物体面积和内部置信度分配 query 数喂给 VGGT。</p>
  {''.join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument(
        "--ball-block-root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block",
    )
    parser.add_argument("--num-single", type=int, default=2)
    parser.add_argument("--num-multi", type=int, default=3)
    parser.add_argument("--phys-start-index", type=int, default=0)
    parser.add_argument("--phys-split", default=None)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument(
        "--proposal-source",
        choices=["gdino_only", "motion_only", "motion_then_gdino", "gdino_then_motion"],
        default="gdino_then_motion",
    )
    parser.add_argument("--motion-score-ratio", type=float, default=0.15)
    parser.add_argument("--text-prompt", default="")
    parser.add_argument("--extra-prompt-terms", default="object . thing . item . tool . toy . rigid object .")
    parser.add_argument("--disable-caption-terms", action="store_true")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--track-dedupe-iou-threshold", type=float, default=0.75)
    parser.add_argument("--container-suppress-ratio-threshold", type=float, default=0.95)
    parser.add_argument("--container-suppress-min-contained", type=int, default=2)
    parser.add_argument("--container-suppress-min-area-ratio", type=float, default=1.5)
    parser.add_argument("--container-suppress-small-iou-threshold", type=float, default=0.7)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--min-queries-per-object", type=int, default=4)
    parser.add_argument("--prompt-frame-mode", choices=["first", "last"], default="first")
    parser.add_argument("--sam2-device", default="cpu")
    parser.add_argument("--gdino-device", default="cpu")
    parser.add_argument("--skip-depth-video", action="store_true")
    parser.add_argument("--skip-world-points-video", action="store_true")
    parser.add_argument("--box-only-overlay", action="store_true")
    parser.add_argument("--ball-num-frames", type=int, default=16)
    parser.add_argument("--ball-num-context-frames", type=int, default=16)
    parser.add_argument(
        "--output-dir",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/outputs/vggt_sam_multi_object_viewer",
    )
    parser.add_argument("--port", type=int, default=8783)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    phys_split = args.phys_split if args.phys_split is not None else data_cfg["split"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    phys_ds = PhysStateEpisodeDataset(
        root=data_cfg["root"],
        split=phys_split,
        resolution=tuple(data_cfg["resolution"]),
        num_context_frames=int(data_cfg["num_context_frames"]),
        context_fraction=float(data_cfg.get("context_fraction", 0.5)),
        random_context_frames=False,
        seed=int(cfg.get("experiment", {}).get("seed", 42)),
    )
    ball_ds = BallBlockVideoDataset(
        root=args.ball_block_root,
        num_frames=int(args.ball_num_frames),
        num_context_frames=int(args.ball_num_context_frames),
        resolution=tuple(data_cfg["resolution"]),
    )
    vggt_adapter = VGGTTrackAdapter(
        model_path=model_cfg.get("vggt_model_path"),
        num_queries=int(args.num_queries),
        device=str(device),
        input_hw=tuple(model_cfg["vggt_input_hw"]),
    ).to(device)

    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    results = []

    phys_start = max(int(args.phys_start_index), 0)
    phys_end = min(len(phys_ds), phys_start + max(int(args.num_single), 0))
    for idx in range(phys_start, phys_end):
        sample = phys_ds[idx]
        sample["_fps"] = int(data_cfg.get("fps", 8))
        results.append(
            evaluate_case(
                sample,
                case_group="single_phys_state",
                vggt_adapter=vggt_adapter,
                device=device,
                sam2_device=str(args.sam2_device),
                gdino_device=str(args.gdino_device),
                output_dir=assets_dir,
                min_queries_per_object=int(args.min_queries_per_object),
                prompt_frame_mode=str(args.prompt_frame_mode),
                save_depth_video=not bool(args.skip_depth_video),
                save_world_points_video=not bool(args.skip_world_points_video),
                max_objects=int(args.max_objects),
                proposal_source=str(args.proposal_source),
                motion_score_ratio=float(args.motion_score_ratio),
                text_prompt=str(args.text_prompt),
                extra_prompt_terms=str(args.extra_prompt_terms),
                include_caption_terms=not bool(args.disable_caption_terms),
                gdino_box_threshold=float(args.gdino_box_threshold),
                gdino_text_threshold=float(args.gdino_text_threshold),
                box_only_overlay=bool(args.box_only_overlay),
                track_dedupe_iou_threshold=float(args.track_dedupe_iou_threshold),
                container_suppress_ratio_threshold=float(args.container_suppress_ratio_threshold),
                container_suppress_min_contained=int(args.container_suppress_min_contained),
                container_suppress_min_area_ratio=float(args.container_suppress_min_area_ratio),
                container_suppress_small_iou_threshold=float(args.container_suppress_small_iou_threshold),
            )
        )

    for idx in range(min(args.num_multi, len(ball_ds))):
        sample = ball_ds[idx]
        sample["_fps"] = int(data_cfg.get("fps", 8))
        results.append(
            evaluate_case(
                sample,
                case_group="multi_ball_block",
                vggt_adapter=vggt_adapter,
                device=device,
                sam2_device=str(args.sam2_device),
                gdino_device=str(args.gdino_device),
                output_dir=assets_dir,
                min_queries_per_object=int(args.min_queries_per_object),
                prompt_frame_mode=str(args.prompt_frame_mode),
                save_depth_video=not bool(args.skip_depth_video),
                save_world_points_video=not bool(args.skip_world_points_video),
                max_objects=int(args.max_objects),
                proposal_source=str(args.proposal_source),
                motion_score_ratio=float(args.motion_score_ratio),
                text_prompt=str(args.text_prompt),
                extra_prompt_terms=str(args.extra_prompt_terms),
                include_caption_terms=not bool(args.disable_caption_terms),
                gdino_box_threshold=float(args.gdino_box_threshold),
                gdino_text_threshold=float(args.gdino_text_threshold),
                box_only_overlay=bool(args.box_only_overlay),
                track_dedupe_iou_threshold=float(args.track_dedupe_iou_threshold),
                container_suppress_ratio_threshold=float(args.container_suppress_ratio_threshold),
                container_suppress_min_contained=int(args.container_suppress_min_contained),
                container_suppress_min_area_ratio=float(args.container_suppress_min_area_ratio),
                container_suppress_small_iou_threshold=float(args.container_suppress_small_iou_threshold),
            )
        )

    html_path = build_report(results, output_dir)
    print(f"eval report: {html_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{args.port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
