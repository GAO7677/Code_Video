from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from code_vjepa_free.vjepa_guidance.motion_masks import MotionMaskResult, _dilate_mask
from code_vjepa_vggt.adapters.sam2_motion import SAM2MotionTracker, build_motion_prompt_boxes


@dataclass
class SAM2TrackDebug:
    prompt_frame_idx: int
    prompt_mode: str
    track_count: int
    boxes_xyxy: np.ndarray
    prompt_boxes_xyxy: np.ndarray
    prompt_texts: list[str]


def _normalize_heat(heat: np.ndarray) -> np.ndarray:
    heat = np.nan_to_num(heat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(heat.max(initial=0.0))
    if max_value <= 1.0e-6:
        return np.zeros_like(heat, dtype=np.float32)
    return np.clip(heat / max_value, 0.0, 1.0).astype(np.float32)


def _video_u8_to_tchw01(video_thwc_u8: np.ndarray) -> np.ndarray:
    return np.transpose(video_thwc_u8.astype(np.float32) / 255.0, (0, 3, 1, 2))


def _filter_track_masks(tracks_mask_thw: list[np.ndarray]) -> list[np.ndarray]:
    kept: list[np.ndarray] = []
    for mask_thw in tracks_mask_thw:
        if mask_thw.ndim != 3:
            continue
        if int(mask_thw.sum()) <= 0:
            continue
        kept.append((mask_thw > 0).astype(np.float32))
    return kept


def _union_tracks(track_masks: list[np.ndarray]) -> np.ndarray:
    if not track_masks:
        raise ValueError("track_masks must not be empty")
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
    repeated = np.repeat(envelope_hw[None, ...], union_mask_thw.shape[0], axis=0)
    return np.clip(repeated, 0.0, 1.0).astype(np.float32)


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


def compute_sam2_object_motion_masks(
    video_thwc_u8: np.ndarray,
    *,
    sam2_device: str = "cuda:0",
    segment_len: int = 8,
    max_objects: int = 4,
    top_frames: int = 3,
    motion_dilate_px: int = 10,
    support_dilate_px: int = 20,
    tracker_model_cfg: str | Path = "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml",
    tracker_checkpoint_path: str | Path = "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt",
) -> tuple[dict[str, MotionMaskResult], SAM2TrackDebug]:
    frames_tchw_01 = _video_u8_to_tchw01(video_thwc_u8)
    motion_prompt = build_motion_prompt_boxes(
        frames_tchw_01,
        max_boxes=max_objects,
        top_frames=top_frames,
    )
    if motion_prompt.boxes_xyxy.shape[0] == 0:
        empty = {
            "sam2_object_union": _empty_like(video_thwc_u8, "sam2_object_union"),
            "sam2_motion_xor": _empty_like(video_thwc_u8, "sam2_motion_xor"),
            "sam2_trajectory_envelope": _empty_like(video_thwc_u8, "sam2_trajectory_envelope"),
            "sam2_guidance_support": _empty_like(video_thwc_u8, "sam2_guidance_support"),
        }
        return empty, SAM2TrackDebug(
            prompt_frame_idx=int(motion_prompt.prompt_frame_idx),
            prompt_mode=str(motion_prompt.prompt_mode),
            track_count=0,
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            prompt_boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            prompt_texts=[],
        )

    tracker = SAM2MotionTracker(
        device=str(sam2_device),
        model_cfg=str(tracker_model_cfg),
        checkpoint_path=str(tracker_checkpoint_path),
        segment_len=int(segment_len),
        enable_text_prompt=False,
    )
    track_masks: list[np.ndarray] = []
    track_boxes: list[np.ndarray] = []
    prompt_texts: list[str] = []
    for prompt_box_xyxy in motion_prompt.boxes_xyxy[: max_objects]:
        try:
            sam_out = tracker.track(
                frames_tchw_01,
                prompt_frame_idx=int(motion_prompt.prompt_frame_idx),
                prompt_box_xyxy=np.asarray(prompt_box_xyxy, dtype=np.float32),
                caption="",
            )
        except Exception:
            continue
        if int(sam_out.masks_thw.sum()) <= 0:
            continue
        track_masks.append((sam_out.masks_thw > 0).astype(np.float32))
        track_boxes.append(sam_out.boxes_t4.astype(np.float32))
        prompt_texts.append(str(sam_out.prompt_mode))

    filtered_tracks = _filter_track_masks(track_masks)
    if not filtered_tracks:
        empty = {
            "sam2_object_union": _empty_like(video_thwc_u8, "sam2_object_union"),
            "sam2_motion_xor": _empty_like(video_thwc_u8, "sam2_motion_xor"),
            "sam2_trajectory_envelope": _empty_like(video_thwc_u8, "sam2_trajectory_envelope"),
            "sam2_guidance_support": _empty_like(video_thwc_u8, "sam2_guidance_support"),
        }
        return empty, SAM2TrackDebug(
            prompt_frame_idx=int(motion_prompt.prompt_frame_idx),
            prompt_mode=str(motion_prompt.prompt_mode),
            track_count=0,
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            prompt_boxes_xyxy=motion_prompt.boxes_xyxy.astype(np.float32),
            prompt_texts=prompt_texts,
        )

    object_union = _union_tracks(filtered_tracks)
    motion_xor = _xor_motion_from_union(object_union)
    motion_xor = _dilate_mask(motion_xor, dilate_px=motion_dilate_px)
    trajectory_envelope = _trajectory_envelope(object_union, dilate_px=support_dilate_px)
    guidance_support = np.clip(motion_xor * trajectory_envelope, 0.0, 1.0).astype(np.float32)
    guidance_support = _dilate_mask(guidance_support, dilate_px=max(1, motion_dilate_px // 2))

    results = {
        "sam2_object_union": MotionMaskResult(
            name="sam2_object_union",
            heat=_normalize_heat(object_union),
            mask=np.clip(object_union, 0.0, 1.0).astype(np.float32),
            coverage=float(object_union.mean()),
            threshold=0.5,
        ),
        "sam2_motion_xor": MotionMaskResult(
            name="sam2_motion_xor",
            heat=_normalize_heat(motion_xor),
            mask=np.clip(motion_xor, 0.0, 1.0).astype(np.float32),
            coverage=float(motion_xor.mean()),
            threshold=0.5,
        ),
        "sam2_trajectory_envelope": MotionMaskResult(
            name="sam2_trajectory_envelope",
            heat=_normalize_heat(trajectory_envelope),
            mask=np.clip(trajectory_envelope, 0.0, 1.0).astype(np.float32),
            coverage=float(trajectory_envelope.mean()),
            threshold=0.5,
        ),
        "sam2_guidance_support": MotionMaskResult(
            name="sam2_guidance_support",
            heat=_normalize_heat(guidance_support),
            mask=np.clip(guidance_support, 0.0, 1.0).astype(np.float32),
            coverage=float(guidance_support.mean()),
            threshold=0.5,
        ),
    }
    boxes_xyxy = np.stack(track_boxes, axis=0).astype(np.float32) if track_boxes else np.zeros((0, object_union.shape[0], 4), dtype=np.float32)
    return results, SAM2TrackDebug(
        prompt_frame_idx=int(motion_prompt.prompt_frame_idx),
        prompt_mode=str(motion_prompt.prompt_mode),
        track_count=len(filtered_tracks),
        boxes_xyxy=boxes_xyxy,
        prompt_boxes_xyxy=motion_prompt.boxes_xyxy.astype(np.float32),
        prompt_texts=prompt_texts,
    )


def summarize_debug_payload(debug: SAM2TrackDebug) -> dict[str, Any]:
    return {
        "prompt_frame_idx": int(debug.prompt_frame_idx),
        "prompt_mode": str(debug.prompt_mode),
        "track_count": int(debug.track_count),
        "prompt_boxes_xyxy": np.asarray(debug.prompt_boxes_xyxy, dtype=np.float32).tolist(),
        "track_boxes_xyxy": np.asarray(debug.boxes_xyxy, dtype=np.float32).tolist(),
        "prompt_texts": list(debug.prompt_texts),
    }
