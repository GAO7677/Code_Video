from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..case_inputs import EvalCase, coerce_eval_case, first_path
from ..records import stable_path_id
from .common import emit_result, load_eval_case, result_record
from .physics_iq import (
    _CONTEXT_MODE_CHOICES,
    _SOURCE_VIDEO_KEYS,
    _build_side_by_side_frames,
    _clip_frames,
    _ensure_video,
    _read_video,
    _resize_frames,
    _resolve_context_frames,
    _sample_frame_indices,
    _write_video,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSINONE_ROOT = REPO_ROOT / "PhysInOne-main"
if str(PHYSINONE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSINONE_ROOT))

from pmf import compute_pmf
from pmf.core import _ensure_5d_b_t_c_h_w, align_pred_to_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case PhysInOne PMF evaluation.")
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing output video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument(
        "--source-video",
        type=Path,
        default=None,
        help="Reference/source video path for PMF comparison.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--context-mode",
        choices=_CONTEXT_MODE_CHOICES,
        default="with_context",
        help=(
            "with_context compares from frame 0; without_context drops the first context_frames "
            "from both output and source before PMF."
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
        "--device",
        default="cpu",
        help="Torch device used for PMF, for example cpu or cuda.",
    )
    parser.add_argument(
        "--aligned-video-dir",
        type=Path,
        default=None,
        help=(
            "Directory used to save the exact aligned/resized video pair that participates in PMF. "
            "Defaults to /data/gaoya/agent-data/outputs/physinone_pmf_single_case/<case-id>/"
        ),
    )
    return parser.parse_args()


def _default_aligned_video_dir(output_video_path: Path, source_video_path: Path) -> Path:
    case_id = f"{stable_path_id(output_video_path)}__ref__{stable_path_id(source_video_path)}"
    return Path("/data/gaoya/agent-data/outputs/physinone_pmf_single_case") / case_id


def _resolve_source_video(case: EvalCase, source_video_path: Path | str | None) -> Path:
    if source_video_path is not None:
        return Path(source_video_path)
    if case.metadata is None:
        raise ValueError("source_video_path is required for PMF scoring")
    base_dir_value = case.metadata.get("json_path") or case.metadata.get("_json_path")
    base_dir = Path(base_dir_value).parent if isinstance(base_dir_value, str) and base_dir_value else None
    resolved = first_path(case.metadata, _SOURCE_VIDEO_KEYS, base_dir=base_dir)
    if resolved is None:
        raise ValueError("source_video_path is required for PMF scoring")
    return resolved


def _bgr_frames_to_video_tensor(frames_bgr: list[np.ndarray]) -> torch.Tensor:
    frames_rgb = np.ascontiguousarray(np.stack(frames_bgr, axis=0)[:, :, :, ::-1])
    return torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().unsqueeze(0)


def _tensor_b_t_c_h_w_to_bgr_frames(video: torch.Tensor) -> list[np.ndarray]:
    frames_rgb = (
        video.squeeze(0)
        .permute(0, 2, 3, 1)
        .detach()
        .cpu()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    frames_bgr = np.ascontiguousarray(frames_rgb[:, :, :, ::-1])
    return [frame for frame in frames_bgr]


def score_case(
    case: EvalCase | Path | str | Mapping[str, Any],
    *,
    source_video_path: Path | str | None = None,
    context_mode: str = "with_context",
    context_frames: int | None = None,
    device: str = "cpu",
    aligned_video_dir: Path | str | None = None,
) -> dict[str, Any]:
    if context_mode not in _CONTEXT_MODE_CHOICES:
        raise ValueError(f"context_mode must be one of {_CONTEXT_MODE_CHOICES}, got {context_mode!r}")

    normalized = coerce_eval_case(case)
    metadata_source = _resolve_source_video(normalized, source_video_path)
    resolved_context_frames = _resolve_context_frames(normalized, context_frames, context_mode)
    output_start_frame = resolved_context_frames if context_mode == "without_context" else 0
    source_start_frame = resolved_context_frames if context_mode == "without_context" else 0

    output_video_path = normalized.video_path
    _ensure_video(output_video_path)
    _ensure_video(metadata_source)

    output_frames_all, output_fps = _read_video(output_video_path)
    source_frames_all, source_fps = _read_video(metadata_source)
    output_frames_clipped = _clip_frames(output_frames_all, output_start_frame, label="output_video")
    source_frames_clipped = _clip_frames(source_frames_all, source_start_frame, label="source_video")

    output_duration = len(output_frames_clipped) / output_fps
    source_duration = len(source_frames_clipped) / source_fps
    compare_duration = min(output_duration, source_duration)
    compare_fps = min(output_fps, source_fps)
    compare_count = max(1, int(np.floor(compare_duration * compare_fps + 1e-6)))
    timestamps = np.arange(compare_count, dtype=np.float32) / float(compare_fps)

    output_indices = _sample_frame_indices(len(output_frames_clipped), output_fps, timestamps)
    source_indices = _sample_frame_indices(len(source_frames_clipped), source_fps, timestamps)
    sampled_output_frames = [output_frames_clipped[int(idx)] for idx in output_indices]
    sampled_source_frames = [source_frames_clipped[int(idx)] for idx in source_indices]

    output_h, output_w = sampled_output_frames[0].shape[:2]
    output_size = (output_w, output_h)
    sampled_source_frames_resized = _resize_frames(sampled_source_frames, output_size)

    pred_tensor = _bgr_frames_to_video_tensor(sampled_output_frames)
    gt_tensor = _bgr_frames_to_video_tensor(sampled_source_frames_resized)

    gt_for_pmf = _ensure_5d_b_t_c_h_w(gt_tensor).to(device)
    pred_for_pmf = _ensure_5d_b_t_c_h_w(pred_tensor).to(device)
    pred_aligned = align_pred_to_gt(pred_for_pmf, gt_for_pmf)
    score_tensor = compute_pmf(gt_tensor, pred_tensor, device=device)
    pmf_score = float(score_tensor.squeeze().detach().cpu().item())

    pred_used_frames = _tensor_b_t_c_h_w_to_bgr_frames(pred_aligned)
    gt_used_frames = _tensor_b_t_c_h_w_to_bgr_frames(gt_for_pmf)

    aligned_dir = (
        Path(aligned_video_dir)
        if aligned_video_dir is not None
        else _default_aligned_video_dir(output_video_path, metadata_source)
    )
    pred_used_path = aligned_dir / "pred_used_for_pmf.mp4"
    gt_used_path = aligned_dir / "gt_used_for_pmf.mp4"
    compare_side_by_side_path = aligned_dir / "compare_side_by_side.mp4"
    _write_video(pred_used_frames, pred_used_path, compare_fps)
    _write_video(gt_used_frames, gt_used_path, compare_fps)
    _write_video(_build_side_by_side_frames(pred_used_frames, gt_used_frames), compare_side_by_side_path, compare_fps)

    used_shape = np.shape(pred_used_frames[0]) if pred_used_frames else (0, 0, 0)
    used_frame_count = len(pred_used_frames)
    return {
        "score": pmf_score,
        "pmf_score": pmf_score,
        "official": True,
        "method": "physinone_pmf_single_case",
        "reference_video": str(metadata_source),
        "context_mode": str(context_mode),
        "context_frames_used": int(resolved_context_frames),
        "output_start_frame": int(output_start_frame),
        "source_start_frame": int(source_start_frame),
        "output_frames_after_context_clip": int(len(output_frames_clipped)),
        "source_frames_after_context_clip": int(len(source_frames_clipped)),
        "num_frames_compared": int(compare_count),
        "compare_duration_sec": round(float(compare_duration), 6),
        "pred_used_for_pmf": str(pred_used_path),
        "gt_used_for_pmf": str(gt_used_path),
        "compare_side_by_side": str(compare_side_by_side_path),
        "video_codec": "h264",
        "metric_direction": "higher_is_better",
        "device": str(device),
        "output_fps": round(float(output_fps), 6),
        "source_fps": round(float(source_fps), 6),
        "compare_fps": round(float(compare_fps), 6),
        "output_duration_sec": round(float(output_duration), 6),
        "source_duration_sec": round(float(source_duration), 6),
        "output_spatial_size": [int(output_w), int(output_h)],
        "source_aligned_size": [int(output_w), int(output_h)],
        "used_shape": [int(used_frame_count), int(used_shape[0]), int(used_shape[1]), int(used_shape[2])],
        "frame_alignment": "timestamp_resample_to_common_duration_before_pmf",
        "spatial_alignment": "resize_source_to_output_before_pmf",
        "notes": (
            "PhysInOne PMF similarity score. Higher is better. The scorer optionally drops the first "
            "context_frames from both videos, samples both streams at shared timestamps over their common "
            "duration, resizes the source to the output spatial size, then applies the official PMF implementation."
        ),
    }


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video)
    result = score_case(
        case,
        source_video_path=args.source_video,
        context_mode=args.context_mode,
        context_frames=args.context_frames,
        device=args.device,
        aligned_video_dir=args.aligned_video_dir,
    )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
