from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from ..case_inputs import EvalCase, coerce_eval_case, first_path
from ..records import load_payload
from .common import emit_result, result_record


_SOURCE_VIDEO_KEYS = (
    "source_video",
    "source_video_path",
    "source",
    "reference_video",
    "reference_video_path",
    "gt_video",
    "gt_video_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-view approximate Physics-IQ evaluation."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=None,
        help="Case JSON containing output video metadata and optionally source_video.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Candidate video path.",
    )
    parser.add_argument(
        "--source-video",
        type=Path,
        default=None,
        help="Reference/source video path for single-view comparison.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )
    parser.add_argument(
        "--threshold-value",
        type=int,
        default=10,
        help="Threshold used to build binary motion masks.",
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="Resize reference frames by this factor before scoring, matching the official benchmark style.",
    )
    return parser.parse_args()


def _load_case_and_source(
    *,
    input_json: Path | None = None,
    video: Path | None = None,
    source_video: Path | None = None,
) -> tuple[EvalCase, Path]:
    payload: dict[str, Any] | None = None
    if input_json is not None:
        payload = load_payload(input_json)
        payload["_json_path"] = str(input_json)
        case = coerce_eval_case(payload)
    else:
        if video is None:
            raise ValueError("Either input_json or video must be provided")
        case = EvalCase(video_path=video)

    if source_video is not None:
        resolved_source = source_video
    else:
        if payload is None:
            raise ValueError("source_video is required when input_json is not provided")
        base_dir_value = payload.get("json_path") or payload.get("_json_path")
        base_dir = Path(base_dir_value).parent if isinstance(base_dir_value, str) and base_dir_value else None
        resolved_source = first_path(payload, _SOURCE_VIDEO_KEYS, base_dir=base_dir)
        if resolved_source is None:
            raise ValueError(
                "case payload does not contain a usable source video path; pass --source-video explicitly"
            )

    if case.metadata is not None:
        metadata = dict(case.metadata)
        metadata["source_video"] = str(resolved_source)
        case = replace(case, metadata=metadata)
    return case, resolved_source


def _ensure_video(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video has no readable frames: {path}")
    if fps <= 0:
        fps = 30.0
    return frames, fps


def _sample_frame_indices(num_frames: int, fps: float, timestamps: np.ndarray) -> np.ndarray:
    indices = np.clip(np.floor(timestamps * fps + 1e-6).astype(np.int64), 0, num_frames - 1)
    return indices


def _resize_and_normalize(frames: list[np.ndarray], target_size: tuple[int, int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in frames:
        resized = cv2.resize(frame, target_size)
        out.append(resized.astype(np.float32) / 255.0)
    return out


def _build_motion_masks(
    frames: list[np.ndarray],
    *,
    threshold_value: int,
) -> list[np.ndarray]:
    if not frames:
        return []
    gray0 = cv2.cvtColor((frames[0] * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray0 = cv2.GaussianBlur(gray0, (5, 5), 0)
    avg_frame = gray0.astype(np.float32)
    masks: list[np.ndarray] = [np.zeros_like(gray0, dtype=np.uint8)]
    kernel = np.ones((5, 5), np.uint8)
    for frame in frames[1:]:
        gray_frame = cv2.cvtColor((frame * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        cv2.accumulateWeighted(gray_frame, avg_frame, 0.3)
        avg_gray_frame = cv2.convertScaleAbs(avg_frame)
        frame_diff = cv2.absdiff(gray_frame, avg_gray_frame)
        _, binary_frame = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, kernel)
        binary_frame = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)
        masks.append((binary_frame > 127).astype(np.uint8))
    return masks


def _mse_per_frame(video1: list[np.ndarray], video2: list[np.ndarray]) -> list[float]:
    if len(video1) != len(video2):
        raise ValueError("Videos must have the same number of frames")
    values: list[float] = []
    for frame1, frame2 in zip(video1, video2):
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must have the same size")
        mse = float(np.mean((frame1.astype(np.float32) - frame2.astype(np.float32)) ** 2))
        values.append(mse)
    return values


def _spatiotemporal_iou_per_frame(mask1: list[np.ndarray], mask2: list[np.ndarray]) -> list[float]:
    values: list[float] = []
    for left, right in zip(mask1, mask2):
        intersection = float(np.logical_and(left, right).sum())
        union = float(np.logical_or(left, right).sum())
        values.append(1.0 if union == 0 else intersection / union)
    return values


def _spatial_binary_mask(mask_frames: list[np.ndarray]) -> np.ndarray:
    if not mask_frames:
        return np.zeros((1, 1), dtype=np.uint8)
    spatial_mask = np.max(mask_frames, axis=0)
    return (spatial_mask > 0).astype(np.uint8) * 255


def _weighted_spatial_mask(mask_frames: list[np.ndarray]) -> np.ndarray:
    if not mask_frames:
        return np.zeros((1, 1), dtype=np.float32)
    return np.sum(mask_frames, axis=0, dtype=np.float32) / float(len(mask_frames))


def _compute_weighted_spatial_iou(weighted_spatial_1: np.ndarray, weighted_spatial_2: np.ndarray) -> float:
    intersection = np.minimum(weighted_spatial_1, weighted_spatial_2)
    union = np.maximum(weighted_spatial_1, weighted_spatial_2)
    valid_pixels = union > 0
    if np.sum(valid_pixels) == 0:
        return 1.0
    return float(np.sum(intersection[valid_pixels]) / np.sum(union[valid_pixels]))


def score_case(
    case: EvalCase | Path | str | Mapping[str, Any],
    *,
    source_video_path: Path | str | None = None,
    threshold_value: int = 10,
    downsample_factor: int = 4,
) -> dict[str, Any]:
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")

    normalized = coerce_eval_case(case)
    metadata_source: Path | None = None
    if source_video_path is not None:
        metadata_source = Path(source_video_path)
    elif normalized.metadata is not None:
        base_dir_value = normalized.metadata.get("json_path") or normalized.metadata.get("_json_path")
        base_dir = Path(base_dir_value).parent if isinstance(base_dir_value, str) and base_dir_value else None
        metadata_source = first_path(normalized.metadata, _SOURCE_VIDEO_KEYS, base_dir=base_dir)
    if metadata_source is None:
        raise ValueError("source_video_path is required for approximate Physics-IQ scoring")

    output_video_path = normalized.video_path
    _ensure_video(output_video_path)
    _ensure_video(metadata_source)

    output_frames_raw, output_fps = _read_video(output_video_path)
    source_frames_raw, source_fps = _read_video(metadata_source)

    output_duration = len(output_frames_raw) / output_fps
    source_duration = len(source_frames_raw) / source_fps
    compare_duration = min(output_duration, source_duration)
    compare_fps = min(output_fps, source_fps)
    compare_count = max(1, int(np.floor(compare_duration * compare_fps + 1e-6)))
    timestamps = np.arange(compare_count, dtype=np.float32) / float(compare_fps)

    output_indices = _sample_frame_indices(len(output_frames_raw), output_fps, timestamps)
    source_indices = _sample_frame_indices(len(source_frames_raw), source_fps, timestamps)
    sampled_output_raw = [output_frames_raw[int(idx)] for idx in output_indices]
    sampled_source_raw = [source_frames_raw[int(idx)] for idx in source_indices]

    source_h, source_w = sampled_source_raw[0].shape[:2]
    target_w = max(1, source_w // downsample_factor)
    target_h = max(1, source_h // downsample_factor)
    target_size = (target_w, target_h)

    output_frames = _resize_and_normalize(sampled_output_raw, target_size)
    source_frames = _resize_and_normalize(sampled_source_raw, target_size)

    mse_per_frame = _mse_per_frame(source_frames, output_frames)
    source_masks = _build_motion_masks(source_frames, threshold_value=threshold_value)
    output_masks = _build_motion_masks(output_frames, threshold_value=threshold_value)

    spatiotemporal_iou_per_frame = _spatiotemporal_iou_per_frame(source_masks, output_masks)
    spatial_source = _spatial_binary_mask(source_masks)
    spatial_output = _spatial_binary_mask(output_masks)
    spatial_iou = _spatiotemporal_iou_per_frame([spatial_source], [spatial_output])[0]

    weighted_source = _weighted_spatial_mask(source_masks)
    weighted_output = _weighted_spatial_mask(output_masks)
    weighted_spatial_iou = _compute_weighted_spatial_iou(weighted_source, weighted_output)

    mse_mean = float(np.mean(mse_per_frame))
    spatiotemporal_iou_mean = float(np.mean(spatiotemporal_iou_per_frame))
    raw_score = ((spatiotemporal_iou_mean + spatial_iou + weighted_spatial_iou) / 3.0) - mse_mean
    physics_iq_score = round(max(min(raw_score * 100.0, 100.0), 0.0), 2)

    return {
        "score": physics_iq_score,
        "physics_iq_score": physics_iq_score,
        "official": False,
        "method": "physics_iq_single_view_approx",
        "reference_video": str(metadata_source),
        "mse_mean": round(mse_mean, 6),
        "spatiotemporal_iou_mean": round(spatiotemporal_iou_mean, 6),
        "spatial_iou": round(float(spatial_iou), 6),
        "weighted_spatial_iou": round(float(weighted_spatial_iou), 6),
        "raw_score": round(float(raw_score), 6),
        "num_frames_compared": int(compare_count),
        "compare_duration_sec": round(float(compare_count / compare_fps), 6),
        "compare_fps": round(float(compare_fps), 6),
        "output_fps": round(float(output_fps), 6),
        "source_fps": round(float(source_fps), 6),
        "output_duration_sec": round(float(output_duration), 6),
        "source_duration_sec": round(float(source_duration), 6),
        "target_size": [int(target_w), int(target_h)],
        "downsample_factor": int(downsample_factor),
        "threshold_value": int(threshold_value),
        "frame_alignment": "timestamp_resample_to_shorter_duration",
        "score_formula": "100 * clip(((spatiotemporal_iou_mean + spatial_iou + weighted_spatial_iou) / 3) - mse_mean, 0, 1)",
        "notes": (
            "Approximate single-view score. It drops the official multi-view and two-take physical variance terms."
        ),
    }


def main() -> None:
    args = parse_args()
    case, source_video = _load_case_and_source(
        input_json=args.input_json,
        video=args.video,
        source_video=args.source_video,
    )
    result = score_case(
        case,
        source_video_path=source_video,
        threshold_value=args.threshold_value,
        downsample_factor=args.downsample_factor,
    )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
