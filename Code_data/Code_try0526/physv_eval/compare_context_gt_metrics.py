from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from physv_eval.single_case.physics_iq import score_case as score_physics_iq

REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSINONE_ROOT = REPO_ROOT / "PhysInOne-main"
if str(PHYSINONE_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSINONE_ROOT))

from pmf import compute_pmf
from pmf.core import _ensure_5d_b_t_c_h_w, align_pred_to_gt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare with/without-context GT variants, then compare physics_iq and "
            "PhysInOne PMF on the same spatially aligned inputs."
        )
    )
    parser.add_argument("--pred", type=Path, required=True, help="Prediction video path.")
    parser.add_argument("--gt-full", type=Path, required=True, help="Full GT video path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--context-frames", type=int, default=8, help="Context prefix length in frames.")
    parser.add_argument("--target-width", type=int, default=None, help="Optional override for GT resize width.")
    parser.add_argument("--target-height", type=int, default=None, help="Optional override for GT resize height.")
    parser.add_argument("--pmf-device", default="cpu", help="cpu or cuda for PhysInOne PMF.")
    parser.add_argument(
        "--physics-iq-downsample-factor",
        type=int,
        default=4,
        help="Downsample factor forwarded into physics_iq.py after GT variant export.",
    )
    parser.add_argument(
        "--physics-iq-threshold-value",
        type=int,
        default=10,
        help="Motion threshold forwarded into physics_iq.py.",
    )
    return parser.parse_args()


def ensure_video_exists(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")


def read_video_bgr(path: Path) -> tuple[np.ndarray, float]:
    ensure_video_exists(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0), fps


def get_ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_video_mp4v(path: Path, frames_bgr: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count, height, width, _ = frames_bgr.shape
    if frame_count <= 0:
        raise ValueError(f"Cannot write empty video: {path}")
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {path}")
    try:
        for frame in frames_bgr:
            writer.write(frame)
    finally:
        writer.release()


def transcode_to_h264_baseline(src_path: Path, dst_path: Path) -> None:
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src_path),
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level:v",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode {src_path} -> {dst_path}\n{proc.stderr}"
        )


def write_video_h264_baseline(path: Path, frames_bgr: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="context_metric_tmp_",
        suffix=".mp4",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        write_video_mp4v(temp_path, frames_bgr, fps)
        transcode_to_h264_baseline(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def resize_frames_bgr(frames_bgr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    resized = [
        cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        for frame in frames_bgr
    ]
    return np.stack(resized, axis=0)


def slice_frames(frames_bgr: np.ndarray, start: int, count: int) -> np.ndarray:
    end = min(len(frames_bgr), start + count)
    if start < 0 or start >= len(frames_bgr):
        raise ValueError(f"Invalid slice start {start} for {len(frames_bgr)}-frame video")
    sliced = frames_bgr[start:end]
    if len(sliced) <= 0:
        raise ValueError(f"Empty frame slice [{start}:{end}]")
    return sliced


def convert_bgr_to_rgb(frames_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frames_bgr[:, :, :, ::-1])


def tensor_from_rgb_frames(frames_rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames_rgb).permute(0, 3, 1, 2).float().unsqueeze(0)


def tensor_b_t_c_h_w_to_rgb_numpy(video: torch.Tensor) -> np.ndarray:
    return (
        video.squeeze(0)
        .permute(0, 2, 3, 1)
        .detach()
        .cpu()
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )


def save_side_by_side_video(pred_rgb: np.ndarray, gt_rgb: np.ndarray, path: Path, fps: float) -> None:
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError("Side-by-side export requires identical shapes.")
    _, height, width, channels = pred_rgb.shape
    canvas = np.zeros((pred_rgb.shape[0], height, width * 2, channels), dtype=np.uint8)
    canvas[:, :, :width, :] = pred_rgb
    canvas[:, :, width:, :] = gt_rgb
    for frame in canvas:
        cv2.putText(
            frame,
            "Prediction",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "GT",
            (width + 20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    write_video_h264_baseline(path, canvas[:, :, :, ::-1], fps)


def save_side_by_side_from_video_paths(
    *,
    pred_path: Path,
    gt_path: Path,
    out_path: Path,
) -> dict[str, object]:
    pred_bgr, pred_fps = read_video_bgr(pred_path)
    gt_bgr, gt_fps = read_video_bgr(gt_path)
    pred_rgb = convert_bgr_to_rgb(pred_bgr)
    gt_rgb = convert_bgr_to_rgb(gt_bgr)
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(
            f"Cannot build side-by-side from mismatched shapes: {pred_rgb.shape} vs {gt_rgb.shape}"
        )
    fps = min(float(pred_fps), float(gt_fps))
    save_side_by_side_video(pred_rgb, gt_rgb, out_path, fps)
    return {
        "compare_side_by_side": str(out_path),
        "compare_shape": list(pred_rgb.shape),
        "compare_fps": fps,
    }


def compute_pmf_bundle(
    *,
    pred_path: Path,
    gt_path: Path,
    out_dir: Path,
    device: str,
) -> dict[str, object]:
    pred_bgr, pred_fps = read_video_bgr(pred_path)
    gt_bgr, gt_fps = read_video_bgr(gt_path)
    pred_rgb = convert_bgr_to_rgb(pred_bgr)
    gt_rgb = convert_bgr_to_rgb(gt_bgr)

    pred_tensor = tensor_from_rgb_frames(pred_rgb)
    gt_tensor = tensor_from_rgb_frames(gt_rgb)
    gt_for_pmf = _ensure_5d_b_t_c_h_w(gt_tensor).to(device)
    pred_for_pmf = _ensure_5d_b_t_c_h_w(pred_tensor).to(device)
    pred_aligned = align_pred_to_gt(pred_for_pmf, gt_for_pmf)
    score_tensor = compute_pmf(gt_tensor, pred_tensor, device=device)
    score = float(score_tensor.squeeze().detach().cpu().item())

    pred_used_rgb = tensor_b_t_c_h_w_to_rgb_numpy(pred_aligned)
    gt_used_rgb = tensor_b_t_c_h_w_to_rgb_numpy(gt_for_pmf)

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_used_path = out_dir / "pred_used_for_pmf.mp4"
    gt_used_path = out_dir / "gt_used_for_pmf.mp4"
    compare_path = out_dir / "compare_side_by_side.mp4"
    write_video_h264_baseline(pred_used_path, pred_used_rgb[:, :, :, ::-1], gt_fps)
    write_video_h264_baseline(gt_used_path, gt_used_rgb[:, :, :, ::-1], gt_fps)
    save_side_by_side_video(pred_used_rgb, gt_used_rgb, compare_path, gt_fps)

    return {
        "score": score,
        "pred_original_shape": list(pred_rgb.shape),
        "gt_original_shape": list(gt_rgb.shape),
        "used_shape": list(gt_used_rgb.shape),
        "pred_fps": float(pred_fps),
        "gt_fps": float(gt_fps),
        "pred_used_for_pmf": str(pred_used_path),
        "gt_used_for_pmf": str(gt_used_path),
        "compare_side_by_side": str(compare_path),
        "metric_direction": "higher_is_better",
    }


def variant_record(
    *,
    name: str,
    variant_dir: Path,
    pred_variant_path: Path,
    gt_variant_path: Path,
    pred_start_frame: int,
    pred_frame_count: int,
    gt_start_frame: int,
    gt_frame_count: int,
    physics_iq_result: dict[str, object],
    pmf_result: dict[str, object],
) -> dict[str, object]:
    return {
        "variant": name,
        "variant_dir": str(variant_dir),
        "pred_variant_video": str(pred_variant_path),
        "gt_variant_video": str(gt_variant_path),
        "pred_start_frame": int(pred_start_frame),
        "pred_frame_count": int(pred_frame_count),
        "gt_start_frame": int(gt_start_frame),
        "gt_frame_count": int(gt_frame_count),
        "physics_iq": physics_iq_result,
        "physinone_pmf": pmf_result,
    }


def main() -> None:
    args = parse_args()
    pred_path = args.pred.expanduser().resolve()
    gt_full_path = args.gt_full.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_bgr, pred_fps = read_video_bgr(pred_path)
    gt_full_bgr, gt_full_fps = read_video_bgr(gt_full_path)

    pred_frame_count, pred_height, pred_width = pred_bgr.shape[:3]
    target_width = int(args.target_width or pred_width)
    target_height = int(args.target_height or pred_height)

    base_metadata = {
        "pred_path": str(pred_path),
        "gt_full_path": str(gt_full_path),
        "pred_shape": list(pred_bgr.shape),
        "gt_full_shape": list(gt_full_bgr.shape),
        "pred_fps": float(pred_fps),
        "gt_full_fps": float(gt_full_fps),
        "context_frames": int(args.context_frames),
        "target_size": [target_width, target_height],
        "physics_iq_downsample_factor": int(args.physics_iq_downsample_factor),
        "physics_iq_threshold_value": int(args.physics_iq_threshold_value),
        "pmf_device": str(args.pmf_device),
    }

    variants = [
        ("with_context", 0),
        ("without_context", int(args.context_frames)),
    ]
    records: list[dict[str, object]] = []

    for name, clip_start_frame in variants:
        variant_dir = out_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        pred_variant_frames = slice_frames(
            pred_bgr,
            clip_start_frame,
            pred_frame_count - clip_start_frame,
        )
        pred_variant_path = variant_dir / "pred_variant_h264b.mp4"
        write_video_h264_baseline(pred_variant_path, pred_variant_frames, pred_fps)

        gt_variant_frames = slice_frames(
            gt_full_bgr,
            clip_start_frame,
            len(pred_variant_frames),
        )
        gt_variant_frames = resize_frames_bgr(gt_variant_frames, target_width, target_height)
        gt_variant_path = variant_dir / "gt_variant_resized_h264b.mp4"
        write_video_h264_baseline(gt_variant_path, gt_variant_frames, gt_full_fps)

        physics_iq_dir = variant_dir / "physics_iq"
        physics_iq_result = score_physics_iq(
            {"video": str(pred_variant_path)},
            source_video_path=gt_variant_path,
            threshold_value=int(args.physics_iq_threshold_value),
            downsample_factor=int(args.physics_iq_downsample_factor),
            aligned_video_dir=physics_iq_dir,
        )
        physics_iq_compare = save_side_by_side_from_video_paths(
            pred_path=Path(str(physics_iq_result["scored_output_video"])),
            gt_path=Path(str(physics_iq_result["scored_source_video"])),
            out_path=physics_iq_dir / "compare_side_by_side.mp4",
        )
        physics_iq_result.update(physics_iq_compare)

        pmf_dir = variant_dir / "physinone_pmf"
        pmf_result = compute_pmf_bundle(
            pred_path=pred_variant_path,
            gt_path=gt_variant_path,
            out_dir=pmf_dir,
            device=str(args.pmf_device),
        )

        records.append(
            variant_record(
                name=name,
                variant_dir=variant_dir,
                pred_variant_path=pred_variant_path,
                gt_variant_path=gt_variant_path,
                pred_start_frame=clip_start_frame,
                pred_frame_count=int(len(pred_variant_frames)),
                gt_start_frame=clip_start_frame,
                gt_frame_count=int(len(gt_variant_frames)),
                physics_iq_result=physics_iq_result,
                pmf_result=pmf_result,
            )
        )

    summary = {
        "metadata": base_metadata,
        "variants": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "variant\tgt_start_frame\tgt_frame_count\tphysics_iq_score\tphysinone_pmf_score",
    ]
    for record in records:
        lines.append(
            "\t".join(
                [
                    str(record["variant"]),
                    str(record["pred_start_frame"]),
                    str(record["pred_frame_count"]),
                    str(record["gt_start_frame"]),
                    str(record["gt_frame_count"]),
                    str(record["physics_iq"]["score"]),
                    f"{float(record['physinone_pmf']['score']):.6f}",
                ]
            )
        )
    header = "variant\tpred_start_frame\tpred_frame_count\tgt_start_frame\tgt_frame_count\tphysics_iq_score\tphysinone_pmf_score"
    lines[0] = header
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
