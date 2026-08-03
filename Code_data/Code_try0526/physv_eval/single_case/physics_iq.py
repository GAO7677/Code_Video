from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import av
import cv2
import numpy as np

from ..case_inputs import EvalCase, coerce_eval_case, first_path
from ..records import load_payload, stable_path_id
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
_CONTEXT_FRAME_KEYS = (
    "context_frames",
    "used_context_frames",
    "model_args.context_frames",
)
_CONTEXT_MODE_CHOICES = ("with_context", "without_context")
_OPENCV_NUM_THREADS = 1


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
        help="Resize frames by this factor after optional context clipping and GT-to-output spatial alignment.",
    )
    parser.add_argument(
        "--context-mode",
        choices=_CONTEXT_MODE_CHOICES,
        default="with_context",
        help=(
            "with_context compares from frame 0; without_context drops the first context_frames "
            "from both output and source before scoring."
        ),
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=None,
        help=(
            "Number of context frames to drop when --context-mode=without_context. "
            "If omitted, the scorer tries to infer it from case metadata."
        ),
    )
    parser.add_argument(
        "--aligned-video-dir",
        type=Path,
        default=None,
        help=(
            "Directory used to save the exact aligned/resized video pair that participates in scoring. "
            "Defaults to /data/gaoya/agent-data/outputs/physics_iq_single_case/<case-id>/"
        ),
    )
    return parser.parse_args()


def _get_nested(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first_int(payload: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _get_nested(payload, key) if "." in key else payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and np.isfinite(value):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text.isdigit():
                return int(text)
    return None


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


def _infer_context_frames_from_case(case: EvalCase) -> int | None:
    if case.metadata is not None:
        inferred = _first_int(case.metadata, _CONTEXT_FRAME_KEYS)
        if inferred is not None:
            return inferred
    context_path = case.context_video_path
    if context_path is None or not context_path.is_file():
        return None
    cap = cv2.VideoCapture(str(context_path))
    if not cap.isOpened():
        return None
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count > 0:
        return count
    return None


def _resolve_context_frames(case: EvalCase, explicit_context_frames: int | None, context_mode: str) -> int:
    if explicit_context_frames is not None:
        if explicit_context_frames < 0:
            raise ValueError(f"context_frames must be >= 0, got {explicit_context_frames}")
        return explicit_context_frames
    inferred = _infer_context_frames_from_case(case)
    if inferred is not None:
        return inferred
    if context_mode == "without_context":
        raise ValueError(
            "context_frames is required for context_mode=without_context when it cannot be inferred from case metadata"
        )
    return 0


def _sample_frame_indices(num_frames: int, fps: float, timestamps: np.ndarray) -> np.ndarray:
    indices = np.clip(np.floor(timestamps * fps + 1e-6).astype(np.int64), 0, num_frames - 1)
    return indices


def _resize_frames(frames: list[np.ndarray], target_size: tuple[int, int]) -> list[np.ndarray]:
    return [cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR) for frame in frames]


def _resize_and_normalize(frames: list[np.ndarray], target_size: tuple[int, int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for frame in frames:
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        out.append(resized.astype(np.float32) / 255.0)
    return out


def _clip_frames(frames: list[np.ndarray], start_frame: int, *, label: str) -> list[np.ndarray]:
    if start_frame <= 0:
        return list(frames)
    if start_frame >= len(frames):
        raise ValueError(
            f"{label} has {len(frames)} frames, cannot drop leading context_frames={start_frame}"
        )
    return list(frames[start_frame:])


def _truncate_frames(frames: list[np.ndarray], max_count: int) -> list[np.ndarray]:
    if max_count <= 0:
        raise ValueError(f"max_count must be positive, got {max_count}")
    if len(frames) <= max_count:
        return list(frames)
    return list(frames[:max_count])


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


def _default_aligned_video_dir(output_video_path: Path, source_video_path: Path) -> Path:
    case_id = f"{stable_path_id(output_video_path)}__ref__{stable_path_id(source_video_path)}"
    return Path("/data/gaoya/agent-data/outputs/physics_iq_single_case") / case_id


def _make_even_size(width: int, height: int) -> tuple[int, int]:
    even_width = width if width % 2 == 0 else width - 1
    even_height = height if height % 2 == 0 else height - 1
    return max(2, even_width), max(2, even_height)


def _write_video(frames: list[np.ndarray], path: Path, fps: float) -> None:
    if not frames:
        raise ValueError("Cannot write an empty video")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    rate = Fraction(str(fps)).limit_denominator(1000)
    stream = container.add_stream("libx264", rate=rate)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    try:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame_u8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_u8 = frame
            rgb_frame = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2RGB)
            video_frame = av.VideoFrame.from_ndarray(rgb_frame, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _to_u8_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame.copy()
    return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def _build_side_by_side_frames(
    output_frames: list[np.ndarray],
    source_frames: list[np.ndarray],
) -> list[np.ndarray]:
    if len(output_frames) != len(source_frames):
        raise ValueError("Need the same number of frames to build a side-by-side video")
    compare_frames: list[np.ndarray] = []
    for output_frame, source_frame in zip(output_frames, source_frames):
        left = _to_u8_bgr(output_frame)
        right = _to_u8_bgr(source_frame)
        if left.shape != right.shape:
            raise ValueError("Need the same frame shape to build a side-by-side video")
        merged = np.concatenate([left, right], axis=1)
        width = left.shape[1]
        cv2.putText(
            merged,
            "Prediction",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            merged,
            "Reference",
            (width + 20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        compare_frames.append(merged)
    return compare_frames


def score_case(
    case: EvalCase | Path | str | Mapping[str, Any],
    *,
    source_video_path: Path | str | None = None,
    threshold_value: int = 10,
    downsample_factor: int = 4,
    context_mode: str = "with_context",
    context_frames: int | None = None,
    aligned_video_dir: Path | str | None = None,
) -> dict[str, Any]:
    # Binary masks near the motion threshold can vary across OpenCV worker threads.
    cv2.setNumThreads(_OPENCV_NUM_THREADS)
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be >= 1")
    if context_mode not in _CONTEXT_MODE_CHOICES:
        raise ValueError(f"context_mode must be one of {_CONTEXT_MODE_CHOICES}, got {context_mode!r}")

    normalized = coerce_eval_case(case)
    resolved_context_frames = _resolve_context_frames(normalized, context_frames, context_mode)
    output_start_frame = resolved_context_frames if context_mode == "without_context" else 0
    source_start_frame = resolved_context_frames if context_mode == "without_context" else 0

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

    output_frames_all, output_fps = _read_video(output_video_path)
    source_frames_all, source_fps = _read_video(metadata_source)

    output_frames_raw = _clip_frames(output_frames_all, output_start_frame, label="output_video")
    source_frames_clipped = _clip_frames(source_frames_all, source_start_frame, label="source_video")
    source_frames_clipped = _truncate_frames(source_frames_clipped, len(output_frames_raw))

    output_duration = len(output_frames_raw) / output_fps
    source_duration = len(source_frames_clipped) / source_fps
    compare_duration = min(output_duration, source_duration)
    compare_fps = min(output_fps, source_fps)
    compare_count = max(1, int(np.floor(compare_duration * compare_fps + 1e-6)))
    timestamps = np.arange(compare_count, dtype=np.float32) / float(compare_fps)

    output_indices = _sample_frame_indices(len(output_frames_raw), output_fps, timestamps)
    source_indices = _sample_frame_indices(len(source_frames_clipped), source_fps, timestamps)
    sampled_output_raw = [output_frames_raw[int(idx)] for idx in output_indices]
    sampled_source_raw = [source_frames_clipped[int(idx)] for idx in source_indices]

    output_h, output_w = sampled_output_raw[0].shape[:2]
    output_size = (output_w, output_h)
    sampled_source_resized_raw = _resize_frames(sampled_source_raw, output_size)

    target_w, target_h = _make_even_size(
        max(1, output_w // downsample_factor),
        max(1, output_h // downsample_factor),
    )
    target_size = (target_w, target_h)

    output_frames = _resize_and_normalize(sampled_output_raw, target_size)
    source_frames = _resize_and_normalize(sampled_source_resized_raw, target_size)

    aligned_dir = (
        Path(aligned_video_dir)
        if aligned_video_dir is not None
        else _default_aligned_video_dir(output_video_path, metadata_source)
    )
    aligned_output_path = aligned_dir / "scored_output_video.mp4"
    aligned_source_path = aligned_dir / "scored_source_video.mp4"
    compare_side_by_side_path = aligned_dir / "compare_side_by_side.mp4"
    _write_video(output_frames, aligned_output_path, compare_fps)
    _write_video(source_frames, aligned_source_path, compare_fps)
    _write_video(_build_side_by_side_frames(output_frames, source_frames), compare_side_by_side_path, compare_fps)

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
        "context_mode": str(context_mode),
        "context_frames_used": int(resolved_context_frames),
        "output_start_frame": int(output_start_frame),
        "source_start_frame": int(source_start_frame),
        "output_frames_after_context_clip": int(len(output_frames_raw)),
        "source_frames_after_context_clip": int(len(source_frames_clipped)),
        "scored_output_video": str(aligned_output_path),
        "scored_source_video": str(aligned_source_path),
        "compare_side_by_side": str(compare_side_by_side_path),
        "video_codec": "h264",
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
        "output_spatial_size": [int(output_w), int(output_h)],
        "source_aligned_size": [int(output_w), int(output_h)],
        "target_size": [int(target_w), int(target_h)],
        "downsample_factor": int(downsample_factor),
        "threshold_value": int(threshold_value),
        "opencv_threads": int(_OPENCV_NUM_THREADS),
        "frame_alignment": "timestamp_resample_to_shorter_duration_after_optional_context_clip",
        "spatial_alignment": "resize_source_to_output_before_downsample",
        "score_formula": "100 * clip(((spatiotemporal_iou_mean + spatial_iou + weighted_spatial_iou) / 3) - mse_mean, 0, 1)",
        "notes": (
            "Approximate single-view score. It drops the official multi-view and two-take physical variance terms. "
            "When context_mode=without_context, the scorer drops the first context_frames from both output and source "
            "before temporal alignment."
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
        context_mode=args.context_mode,
        context_frames=args.context_frames,
        aligned_video_dir=args.aligned_video_dir,
    )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
