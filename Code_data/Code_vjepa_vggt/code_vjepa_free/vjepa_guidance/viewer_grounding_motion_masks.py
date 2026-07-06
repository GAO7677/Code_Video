from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from code_vjepa_free.vjepa_guidance.motion_masks import MotionMaskResult, _dilate_mask
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
    ViewerGroundingSample,
)


@dataclass
class ViewerGroundingMaskDebug:
    prompt_frame_idx: int
    prompt_mode: str
    prior_source: str
    track_count: int
    object_valid_mask: np.ndarray
    grouped_queries_px: np.ndarray
    context_boxes_norm: np.ndarray
    debug: dict[str, Any]


def _normalize_heat(heat: np.ndarray) -> np.ndarray:
    heat = np.nan_to_num(heat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(heat.max(initial=0.0))
    if max_value <= 1.0e-6:
        return np.zeros_like(heat, dtype=np.float32)
    return np.clip(heat / max_value, 0.0, 1.0).astype(np.float32)


def _video_u8_to_tchw01(video_thwc_u8: np.ndarray) -> np.ndarray:
    return np.transpose(video_thwc_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))


def _filter_track_masks(track_masks: list[np.ndarray]) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for mask_thw in track_masks:
        if mask_thw.ndim != 3:
            continue
        if int(mask_thw.sum()) <= 0:
            continue
        kept.append((mask_thw > 0).astype(np.float32))
    return kept


def _union_tracks(track_masks: list[np.ndarray]) -> np.ndarray:
    union = np.zeros_like(track_masks[0], dtype=np.float32)
    for mask_thw in track_masks:
        union = np.maximum(union, mask_thw.astype(np.float32))
    return np.clip(union, 0.0, 1.0).astype(np.float32)


def _xor_motion_from_union(union_mask_thw: np.ndarray) -> np.ndarray:
    motion = np.zeros_like(union_mask_thw, dtype=np.float32)
    if union_mask_thw.shape[0] <= 1:
        return motion
    diff = np.abs(union_mask_thw[1:] - union_mask_thw[:-1]).astype(np.float32)
    motion[1:] = np.maximum(motion[1:], diff)
    motion[:-1] = np.maximum(motion[:-1], diff)
    return np.clip(motion, 0.0, 1.0).astype(np.float32)


def _trajectory_envelope(union_mask_thw: np.ndarray, *, dilate_px: int) -> np.ndarray:
    envelope_hw = (union_mask_thw.max(axis=0) > 0).astype(np.float32)
    if dilate_px > 0:
        envelope_hw = _dilate_mask(envelope_hw[None, ...], dilate_px=dilate_px)[0]
    return np.repeat(envelope_hw[None, ...], union_mask_thw.shape[0], axis=0).astype(np.float32)


def _empty_like(video_thwc_u8: np.ndarray, name: str) -> MotionMaskResult:
    frames, height, width = video_thwc_u8.shape[:3]
    zeros = np.zeros((frames, height, width), dtype=np.float32)
    return MotionMaskResult(
        name=name,
        heat=zeros,
        mask=zeros,
        coverage=0.0,
        threshold=0.0,
    )


def build_viewer_grounding_provider(
    *,
    device: str = "cuda:0",
    segment_len: int = 8,
    max_objects: int = 4,
    points_per_object: int = 8,
    proposal_source: str = "gdino_only",
    motion_score_ratio: float = 0.15,
    text_prompt: str = "box . cube . block . cylinder . capsule . sphere . ball .",
    extra_prompt_terms: str = "",
    include_caption_terms: bool = False,
    gdino_box_threshold: float = 0.20,
    gdino_text_threshold: float = 0.15,
    prompt_frame_mode: str = "first",
    track_dedupe_iou_threshold: float = 0.75,
    container_suppress_ratio_threshold: float = 0.95,
    container_suppress_min_contained: int = 2,
    container_suppress_min_area_ratio: float = 1.5,
    container_suppress_small_iou_threshold: float = 0.7,
) -> ViewerGroundingBoxProvider:
    return ViewerGroundingBoxProvider(
        device=str(device),
        segment_len=int(segment_len),
        max_objects=int(max_objects),
        points_per_object=int(points_per_object),
        proposal_source=str(proposal_source),
        motion_score_ratio=float(motion_score_ratio),
        text_prompt=str(text_prompt),
        extra_prompt_terms=str(extra_prompt_terms),
        include_caption_terms=bool(include_caption_terms),
        gdino_box_threshold=float(gdino_box_threshold),
        gdino_text_threshold=float(gdino_text_threshold),
        prompt_frame_mode=str(prompt_frame_mode),
        track_dedupe_iou_threshold=float(track_dedupe_iou_threshold),
        container_suppress_ratio_threshold=float(container_suppress_ratio_threshold),
        container_suppress_min_contained=int(container_suppress_min_contained),
        container_suppress_min_area_ratio=float(container_suppress_min_area_ratio),
        container_suppress_small_iou_threshold=float(container_suppress_small_iou_threshold),
    )


def compute_viewer_grounding_object_motion_masks(
    video_thwc_u8: np.ndarray,
    *,
    caption: str = "",
    provider: ViewerGroundingBoxProvider | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    motion_dilate_px: int = 10,
    support_dilate_px: int = 20,
) -> tuple[dict[str, MotionMaskResult], ViewerGroundingMaskDebug]:
    frames_tchw_01 = _video_u8_to_tchw01(video_thwc_u8)
    image_hw = (int(video_thwc_u8.shape[1]), int(video_thwc_u8.shape[2]))
    owned_provider = provider or build_viewer_grounding_provider(**(provider_kwargs or {}))
    grounding_sample: ViewerGroundingSample = owned_provider.build_sample(
        frames_tchw_01=frames_tchw_01,
        caption=str(caption),
        image_hw=image_hw,
    )

    track_masks = _filter_track_masks([track.masks_thw for track in grounding_sample.object_tracks])
    if not track_masks:
        empty = {
            "viewer_object_union": _empty_like(video_thwc_u8, "viewer_object_union"),
            "viewer_motion_xor": _empty_like(video_thwc_u8, "viewer_motion_xor"),
            "viewer_trajectory_envelope": _empty_like(video_thwc_u8, "viewer_trajectory_envelope"),
            "viewer_guidance_support": _empty_like(video_thwc_u8, "viewer_guidance_support"),
        }
        return empty, ViewerGroundingMaskDebug(
            prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
            prompt_mode=str(getattr(grounding_sample, "prompt_mode", "")),
            prior_source=str(getattr(grounding_sample, "prior_source", "")),
            track_count=0,
            object_valid_mask=np.asarray(grounding_sample.object_valid_mask, dtype=np.float32),
            grouped_queries_px=np.asarray(grounding_sample.grouped_queries_px, dtype=np.float32),
            context_boxes_norm=np.asarray(grounding_sample.context_boxes_norm, dtype=np.float32),
            debug=dict(getattr(grounding_sample, "debug", {})),
        )

    object_union = _union_tracks(track_masks)
    motion_xor = _xor_motion_from_union(object_union)
    motion_xor = _dilate_mask(motion_xor, dilate_px=motion_dilate_px)
    trajectory_envelope = _trajectory_envelope(object_union, dilate_px=support_dilate_px)
    guidance_support = np.clip(motion_xor * trajectory_envelope, 0.0, 1.0).astype(np.float32)
    guidance_support = _dilate_mask(guidance_support, dilate_px=max(1, motion_dilate_px // 2))

    results = {
        "viewer_object_union": MotionMaskResult(
            name="viewer_object_union",
            heat=_normalize_heat(object_union),
            mask=np.clip(object_union, 0.0, 1.0).astype(np.float32),
            coverage=float(object_union.mean()),
            threshold=0.5,
        ),
        "viewer_motion_xor": MotionMaskResult(
            name="viewer_motion_xor",
            heat=_normalize_heat(motion_xor),
            mask=np.clip(motion_xor, 0.0, 1.0).astype(np.float32),
            coverage=float(motion_xor.mean()),
            threshold=0.5,
        ),
        "viewer_trajectory_envelope": MotionMaskResult(
            name="viewer_trajectory_envelope",
            heat=_normalize_heat(trajectory_envelope),
            mask=np.clip(trajectory_envelope, 0.0, 1.0).astype(np.float32),
            coverage=float(trajectory_envelope.mean()),
            threshold=0.5,
        ),
        "viewer_guidance_support": MotionMaskResult(
            name="viewer_guidance_support",
            heat=_normalize_heat(guidance_support),
            mask=np.clip(guidance_support, 0.0, 1.0).astype(np.float32),
            coverage=float(guidance_support.mean()),
            threshold=0.5,
        ),
    }
    return results, ViewerGroundingMaskDebug(
        prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
        prompt_mode=str(getattr(grounding_sample, "prompt_mode", "")),
        prior_source=str(getattr(grounding_sample, "prior_source", "")),
        track_count=len(track_masks),
        object_valid_mask=np.asarray(grounding_sample.object_valid_mask, dtype=np.float32),
        grouped_queries_px=np.asarray(grounding_sample.grouped_queries_px, dtype=np.float32),
        context_boxes_norm=np.asarray(grounding_sample.context_boxes_norm, dtype=np.float32),
        debug=dict(getattr(grounding_sample, "debug", {})),
    )


def extract_viewer_grounding_motion_mask_thw(
    video_thwc_u8: np.ndarray,
    *,
    caption: str = "",
    method: str = "viewer_guidance_support",
    provider: ViewerGroundingBoxProvider | None = None,
    provider_kwargs: dict[str, Any] | None = None,
    motion_dilate_px: int = 10,
    support_dilate_px: int = 20,
) -> np.ndarray:
    """
    Standard project API:
      input:  video_thwc_u8  [T,H,W,3]
      output: motion_mask_thw [T,H,W] float32 in {0,1}
    """
    results, _ = compute_viewer_grounding_object_motion_masks(
        video_thwc_u8,
        caption=caption,
        provider=provider,
        provider_kwargs=provider_kwargs,
        motion_dilate_px=motion_dilate_px,
        support_dilate_px=support_dilate_px,
    )
    if method not in results:
        raise KeyError(f"unknown viewer grounding motion mask method: {method}")
    return results[method].mask.astype(np.float32)


def summarize_debug_payload(debug: ViewerGroundingMaskDebug) -> dict[str, Any]:
    return {
        "prompt_frame_idx": int(debug.prompt_frame_idx),
        "prompt_mode": str(debug.prompt_mode),
        "prior_source": str(debug.prior_source),
        "track_count": int(debug.track_count),
        "object_valid_mask": np.asarray(debug.object_valid_mask, dtype=np.float32).tolist(),
        "grouped_queries_px": np.asarray(debug.grouped_queries_px, dtype=np.float32).tolist(),
        "context_boxes_norm": np.asarray(debug.context_boxes_norm, dtype=np.float32).tolist(),
        "debug": dict(debug.debug),
    }
