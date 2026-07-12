from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from code_vjepa_vggt.adapters.sam2_motion import (
    GroundingDINOTextDetector,
    SAM2MotionTracker,
    build_motion_prompt_box,
    build_motion_prompt_boxes,
)
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior


@dataclass
class DetectedObjectTrack:
    box_prompt_xyxy: np.ndarray
    masks_thw: np.ndarray
    boxes_t4: np.ndarray
    score: float
    phrase: str


@dataclass
class ViewerGroundingSample:
    grouped_queries_px: np.ndarray
    object_valid_mask: np.ndarray
    context_boxes_norm: np.ndarray
    object_tracks: list[DetectedObjectTrack]
    prior_source: str
    prompt_mode: str
    prompt_frame_idx: int
    debug: dict[str, Any]


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
        caption_prompt = str(caption).strip()
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


def score_track(track: DetectedObjectTrack) -> float:
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
    return float(np.mean(areas) * max(np.mean(confs), 1.0e-3) * max(float(track.score), 1.0e-3))


def track_similarity_iou(track_a: DetectedObjectTrack, track_b: DetectedObjectTrack) -> float:
    box_a = np.mean(track_a.boxes_t4.astype(np.float32), axis=0)
    box_b = np.mean(track_b.boxes_t4.astype(np.float32), axis=0)
    return box_iou_xyxy(box_a, box_b)


def _phrase_terms(phrase: str) -> set[str]:
    normalized = str(phrase).lower().replace("_", " ").replace("-", " ")
    return {term for term in normalized.split() if term}


def track_is_nested_duplicate(
    track_a: DetectedObjectTrack,
    track_b: DetectedObjectTrack,
    *,
    containment_threshold: float = 0.85,
) -> bool:
    """Catch same-class nested boxes that IoU-only dedupe misses."""
    if not (_phrase_terms(track_a.phrase) & _phrase_terms(track_b.phrase)):
        return False
    box_a = np.mean(track_a.boxes_t4.astype(np.float32), axis=0)
    box_b = np.mean(track_b.boxes_t4.astype(np.float32), axis=0)
    containment = max(
        box_containment_ratio_xyxy(box_a, box_b),
        box_containment_ratio_xyxy(box_b, box_a),
    )
    if containment < float(containment_threshold):
        return False
    center_a = 0.5 * (box_a[:2] + box_a[2:])
    center_b = 0.5 * (box_b[:2] + box_b[2:])
    size_a = np.maximum(box_a[2:] - box_a[:2], 1.0)
    size_b = np.maximum(box_b[2:] - box_b[:2], 1.0)
    center_distance = float(np.linalg.norm(center_a - center_b))
    reference_diagonal = float(max(np.linalg.norm(size_a), np.linalg.norm(size_b), 1.0))
    return center_distance <= 0.25 * reference_diagonal


def dedupe_object_tracks(tracks: list[DetectedObjectTrack], iou_threshold: float) -> list[DetectedObjectTrack]:
    if len(tracks) <= 1:
        return tracks
    ranked = sorted(tracks, key=score_track, reverse=True)
    kept: list[DetectedObjectTrack] = []
    for candidate in ranked:
        if any(
            track_similarity_iou(candidate, existing) >= float(iou_threshold)
            or track_is_nested_duplicate(candidate, existing)
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def suppress_container_tracks(
    tracks: list[DetectedObjectTrack],
    *,
    containment_ratio_threshold: float,
    min_contained_tracks: int,
    min_area_ratio: float,
    small_track_iou_threshold: float,
) -> list[DetectedObjectTrack]:
    if len(tracks) <= 1:
        return tracks

    keep_mask = [True] * len(tracks)
    mean_boxes = [np.mean(track.boxes_t4.astype(np.float32), axis=0) for track in tracks]
    mean_areas = [
        max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        for box in mean_boxes
    ]

    for big_idx, _ in enumerate(tracks):
        if not keep_mask[big_idx]:
            continue
        big_box = mean_boxes[big_idx]
        big_area = mean_areas[big_idx]
        if big_area <= 1.0e-6:
            continue

        contained_indices: list[int] = []
        for small_idx, _ in enumerate(tracks):
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

        if len(unique_small_indices) >= int(min_contained_tracks):
            keep_mask[big_idx] = False

    return [track for track, keep in zip(tracks, keep_mask) if keep]


def fallback_points_from_box(box_xyxy: np.ndarray, num_queries: int) -> np.ndarray:
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


class ViewerGroundingBoxProvider:
    """Training-time detector that mirrors the viewer's GDINO + SAM2 box flow."""

    def __init__(
        self,
        *,
        device: str,
        segment_len: int,
        max_objects: int,
        points_per_object: int,
        proposal_source: str,
        motion_score_ratio: float,
        text_prompt: str,
        extra_prompt_terms: str,
        include_caption_terms: bool,
        gdino_box_threshold: float,
        gdino_text_threshold: float,
        prompt_frame_mode: str,
        track_dedupe_iou_threshold: float,
        container_suppress_ratio_threshold: float,
        container_suppress_min_contained: int,
        container_suppress_min_area_ratio: float,
        container_suppress_small_iou_threshold: float,
    ) -> None:
        self.device = str(device)
        self.segment_len = int(segment_len)
        self.max_objects = int(max_objects)
        self.points_per_object = int(points_per_object)
        self.proposal_source = str(proposal_source)
        self.motion_score_ratio = float(motion_score_ratio)
        self.text_prompt = str(text_prompt)
        self.extra_prompt_terms = str(extra_prompt_terms)
        self.include_caption_terms = bool(include_caption_terms)
        self.gdino_box_threshold = float(gdino_box_threshold)
        self.gdino_text_threshold = float(gdino_text_threshold)
        self.prompt_frame_mode = str(prompt_frame_mode)
        self.track_dedupe_iou_threshold = float(track_dedupe_iou_threshold)
        self.container_suppress_ratio_threshold = float(container_suppress_ratio_threshold)
        self.container_suppress_min_contained = int(container_suppress_min_contained)
        self.container_suppress_min_area_ratio = float(container_suppress_min_area_ratio)
        self.container_suppress_small_iou_threshold = float(container_suppress_small_iou_threshold)

        self.tracker = SAM2MotionTracker(
            device=self.device,
            segment_len=self.segment_len,
            enable_text_prompt=False,
        )
        self.detector = GroundingDINOTextDetector(
            device=self.device,
            max_boxes=self.max_objects,
            box_threshold=self.gdino_box_threshold,
            text_threshold=self.gdino_text_threshold,
        )

    def _make_box_fallback_track(self, box_xyxy: np.ndarray, score: float, phrase: str, height: int, width: int, frames: int) -> DetectedObjectTrack | None:
        fallback_box = np.asarray(box_xyxy, dtype=np.float32)
        x0, y0, x1, y1 = [int(round(v)) for v in fallback_box.tolist()]
        x0 = max(0, min(x0, width - 1))
        x1 = max(0, min(x1, width))
        y0 = max(0, min(y0, height - 1))
        y1 = max(0, min(y1, height))
        if x1 <= x0 or y1 <= y0:
            return None
        masks = np.zeros((frames, height, width), dtype=np.uint8)
        masks[:, y0:y1, x0:x1] = 1
        boxes = np.repeat(fallback_box[None, :], repeats=frames, axis=0).astype(np.float32)
        return DetectedObjectTrack(
            box_prompt_xyxy=fallback_box,
            masks_thw=masks,
            boxes_t4=boxes,
            score=float(score),
            phrase=str(phrase),
        )

    def _detect_and_track_objects(
        self,
        frames_tchw_01: np.ndarray,
        caption: str,
    ) -> tuple[list[DetectedObjectTrack], int, dict[str, Any]]:
        if self.prompt_frame_mode == "first":
            prompt_frame_idx = 0
        else:
            prompt_frame_idx = max(frames_tchw_01.shape[0] - 1, 0)

        proposal_source = str(self.proposal_source)
        motion_multi = build_motion_prompt_boxes(frames_tchw_01, max_boxes=self.max_objects)
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
            if len(candidate_boxes) >= self.max_objects:
                return
            candidate_boxes.append(box)
            candidate_scores.append(float(score))
            candidate_phrases.append(str(phrase))

        def add_motion_candidates() -> None:
            motion_boxes = motion_multi.boxes_xyxy[: self.max_objects]
            motion_scores = motion_multi.scores[: self.max_objects]
            if motion_boxes.shape[0] == 0:
                return
            top_score = float(motion_scores.max()) if motion_scores.size > 0 else 0.0
            min_keep_score = max(1.0e-6, top_score * max(float(self.motion_score_ratio), 0.0))
            for idx, (box_xyxy, score) in enumerate(zip(motion_boxes, motion_scores)):
                if float(score) < min_keep_score:
                    continue
                add_candidate(box_xyxy, float(score), f"motion_component_{idx}")

        prompt_text = build_open_vocab_prompt(
            caption,
            text_prompt=self.text_prompt,
            extra_prompt_terms=self.extra_prompt_terms,
            include_caption_terms=self.include_caption_terms,
        )

        def add_gdino_candidates() -> None:
            detection = self.detector.detect(frames_tchw_01[prompt_frame_idx], prompt_text, guidance_box_xyxy=None)
            for box_xyxy, score, phrase in zip(
                detection.boxes_xyxy[: self.max_objects],
                detection.scores[: self.max_objects],
                detection.phrases[: self.max_objects],
            ):
                add_candidate(box_xyxy, float(score), str(phrase))

        if proposal_source == "gdino_only":
            add_gdino_candidates()
        elif proposal_source == "motion_only":
            add_motion_candidates()
        elif proposal_source == "motion_then_gdino":
            add_motion_candidates()
            if len(candidate_boxes) < self.max_objects:
                add_gdino_candidates()
        elif proposal_source == "gdino_then_motion":
            add_gdino_candidates()
            if len(candidate_boxes) < self.max_objects:
                add_motion_candidates()
        else:
            raise ValueError(f"unsupported proposal_source={proposal_source}")

        if candidate_boxes:
            track_boxes = np.stack(candidate_boxes, axis=0).astype(np.float32)[: self.max_objects]
            track_scores = np.asarray(candidate_scores, dtype=np.float32)[: self.max_objects]
            track_phrases = candidate_phrases[: self.max_objects]
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

        outputs: list[DetectedObjectTrack] = []
        height, width = int(frames_tchw_01.shape[-2]), int(frames_tchw_01.shape[-1])
        frames = int(frames_tchw_01.shape[0])
        for box_xyxy, score, phrase in zip(track_boxes, track_scores, track_phrases):
            try:
                sam_out = self.tracker.track(
                    frames_tchw_01,
                    prompt_frame_idx=prompt_frame_idx,
                    prompt_box_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                    caption="",
                )
            except Exception:
                fallback_track = self._make_box_fallback_track(
                    np.asarray(box_xyxy, dtype=np.float32),
                    float(score),
                    f"{phrase}_box_fallback",
                    height,
                    width,
                    frames,
                )
                if fallback_track is not None:
                    outputs.append(fallback_track)
                continue
            if int(sam_out.masks_thw[0].sum()) <= 0:
                fallback_track = self._make_box_fallback_track(
                    np.asarray(box_xyxy, dtype=np.float32),
                    float(score),
                    f"{phrase}_empty_mask_fallback",
                    height,
                    width,
                    frames,
                )
                if fallback_track is not None:
                    outputs.append(fallback_track)
                continue
            outputs.append(
                DetectedObjectTrack(
                    box_prompt_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                    masks_thw=sam_out.masks_thw.astype(np.uint8),
                    boxes_t4=sam_out.boxes_t4.astype(np.float32),
                    score=float(score),
                    phrase=str(phrase),
                )
            )

        deduped_outputs = dedupe_object_tracks(outputs, iou_threshold=self.track_dedupe_iou_threshold)
        suppressed_outputs = suppress_container_tracks(
            deduped_outputs,
            containment_ratio_threshold=self.container_suppress_ratio_threshold,
            min_contained_tracks=self.container_suppress_min_contained,
            min_area_ratio=self.container_suppress_min_area_ratio,
            small_track_iou_threshold=self.container_suppress_small_iou_threshold,
        )
        return suppressed_outputs[: self.max_objects], prompt_frame_idx, {
            "proposal_source": proposal_source,
            "prompt_text": prompt_text,
            "prompt_frame_idx": int(prompt_frame_idx),
            "raw_candidate_count": int(len(candidate_boxes)),
            "tracked_candidate_count": int(len(outputs)),
            "deduped_track_count": int(len(deduped_outputs)),
            "dedupe_removed_count": int(len(outputs) - len(deduped_outputs)),
            "track_count": int(len(suppressed_outputs)),
            "container_suppressed_count": int(len(deduped_outputs) - len(suppressed_outputs)),
        }

    def _build_grouped_queries(self, tracks: list[DetectedObjectTrack]) -> tuple[np.ndarray, np.ndarray]:
        grouped_queries = np.zeros((self.max_objects, self.points_per_object, 2), dtype=np.float32)
        object_valid_mask = np.zeros((self.max_objects,), dtype=np.float32)
        for obj_idx, track in enumerate(tracks[: self.max_objects]):
            pts, _ = build_vggt_query_prior(track.masks_thw, track.boxes_t4, num_queries=self.points_per_object)
            if pts.shape[0] == 0:
                pts = fallback_points_from_box(track.box_prompt_xyxy.astype(np.float32), self.points_per_object)
            if pts.shape[0] == 0:
                continue
            if pts.shape[0] < self.points_per_object:
                extra = pts[-1:].repeat(self.points_per_object - pts.shape[0], axis=0)
                pts = np.concatenate([pts, extra], axis=0)
            grouped_queries[obj_idx] = pts[: self.points_per_object].astype(np.float32)
            object_valid_mask[obj_idx] = 1.0
        return grouped_queries, object_valid_mask

    def _build_context_boxes_norm(
        self,
        tracks: list[DetectedObjectTrack],
        *,
        num_frames: int,
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        height, width = int(image_hw[0]), int(image_hw[1])
        boxes = np.zeros((num_frames, self.max_objects, 4), dtype=np.float32)
        for obj_idx, track in enumerate(tracks[: self.max_objects]):
            box_t4 = track.boxes_t4.astype(np.float32).copy()
            box_t4[:, [0, 2]] /= max(float(width), 1.0)
            box_t4[:, [1, 3]] /= max(float(height), 1.0)
            boxes[: box_t4.shape[0], obj_idx] = np.clip(box_t4, 0.0, 1.0)
        return boxes

    def build_sample(
        self,
        *,
        frames_tchw_01: np.ndarray,
        caption: str,
        image_hw: tuple[int, int],
    ) -> ViewerGroundingSample:
        object_tracks, prompt_frame_idx, detect_debug = self._detect_and_track_objects(frames_tchw_01, caption)
        grouped_queries_px, object_valid_mask = self._build_grouped_queries(object_tracks)
        context_boxes_norm = self._build_context_boxes_norm(
            object_tracks,
            num_frames=int(frames_tchw_01.shape[0]),
            image_hw=image_hw,
        )
        prior_source = f"viewer_grounded_sam_objects{int(object_valid_mask.sum())}"
        prompt_mode = f"{self.prompt_frame_mode}_viewer_grounding"
        return ViewerGroundingSample(
            grouped_queries_px=grouped_queries_px,
            object_valid_mask=object_valid_mask,
            context_boxes_norm=context_boxes_norm,
            object_tracks=object_tracks,
            prior_source=prior_source,
            prompt_mode=prompt_mode,
            prompt_frame_idx=prompt_frame_idx,
            debug={
                **detect_debug,
                "object_count": int(object_valid_mask.sum()),
                "object_phrases": [track.phrase for track in object_tracks],
                "object_scores": [float(track.score) for track in object_tracks],
                "object_track_scores": [float(score_track(track)) for track in object_tracks],
                "object_prompt_boxes_xyxy": [
                    [float(value) for value in track.box_prompt_xyxy.tolist()]
                    for track in object_tracks
                ],
                "object_mean_track_boxes_xyxy": [
                    [
                        float(value)
                        for value in np.mean(track.boxes_t4.astype(np.float32), axis=0).tolist()
                    ]
                    for track in object_tracks
                ],
                "context_boxes_norm": list(context_boxes_norm.shape),
                "grouped_queries_px": list(grouped_queries_px.shape),
            },
        )
