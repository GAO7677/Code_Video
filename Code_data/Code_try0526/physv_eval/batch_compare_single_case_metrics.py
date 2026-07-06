from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from physv_eval.single_case.physics_iq import score_case as score_physics_iq
from physv_eval.single_case.pmf import score_case as score_pmf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-score one basename across a prediction root using single_case.physics_iq "
            "and single_case.pmf, then export annotated stitched compare videos."
        )
    )
    parser.add_argument("--pred-root", type=Path, required=True, help="Root directory containing prediction videos.")
    parser.add_argument("--basename", required=True, help="Filename to match under pred-root, for example sample.mp4.")
    parser.add_argument("--gt-full", type=Path, required=True, help="Ground-truth full video path.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for batch artifacts.")
    parser.add_argument(
        "--context-mode",
        choices=("with_context", "without_context"),
        default="without_context",
        help="Context handling mode forwarded into both metrics.",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="Context frame count forwarded into both metrics when context-mode=without_context.",
    )
    parser.add_argument(
        "--pmf-device",
        default="cpu",
        help="Torch device used for PMF scoring, for example cpu or cuda.",
    )
    parser.add_argument(
        "--physics-iq-downsample-factor",
        type=int,
        default=4,
        help="Downsample factor forwarded into single_case.physics_iq.",
    )
    parser.add_argument(
        "--physics-iq-threshold-value",
        type=int,
        default=10,
        help="Motion threshold forwarded into single_case.physics_iq.",
    )
    return parser.parse_args()


def read_video_bgr(path: Path) -> tuple[np.ndarray, float]:
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
    height, width = frames_bgr.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer for {path}")
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
    proc = __import__("subprocess").run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode {src_path} -> {dst_path}\n{proc.stderr}"
        )


def write_video_h264_baseline(path: Path, frames_bgr: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.stem + "_tmp_mp4v.mp4")
    try:
        write_video_mp4v(temp_path, frames_bgr, fps)
        transcode_to_h264_baseline(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] == width:
        return frame
    scale = float(width) / float(frame.shape[1])
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def put_multiline_text(frame: np.ndarray, lines: list[str], x: int, y: int) -> None:
    for idx, line in enumerate(lines):
        yy = y + idx * 32
        cv2.putText(
            frame,
            line,
            (x, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def build_annotated_compare_video(
    *,
    physics_compare_path: Path,
    pmf_compare_path: Path,
    out_path: Path,
    title: str,
    context_mode: str,
    physics_score: float,
    pmf_score: float,
) -> dict[str, object]:
    physics_frames, physics_fps = read_video_bgr(physics_compare_path)
    pmf_frames, pmf_fps = read_video_bgr(pmf_compare_path)
    compare_fps = min(float(physics_fps), float(pmf_fps))
    compare_count = min(len(physics_frames), len(pmf_frames))
    physics_frames = physics_frames[:compare_count]
    pmf_frames = pmf_frames[:compare_count]

    target_width = max(int(physics_frames.shape[2]), int(pmf_frames.shape[2]))
    physics_resized = np.stack([resize_to_width(frame, target_width) for frame in physics_frames], axis=0)
    pmf_resized = np.stack([resize_to_width(frame, target_width) for frame in pmf_frames], axis=0)

    header_height = 96
    gap = 16
    canvas_height = (
        header_height
        + physics_resized.shape[1]
        + gap
        + pmf_resized.shape[1]
    )
    canvas_width = target_width

    output_frames: list[np.ndarray] = []
    for physics_frame, pmf_frame in zip(physics_resized, pmf_resized):
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 18)
        put_multiline_text(
            canvas,
            [
                title,
                f"context_mode={context_mode} | physics_iq={physics_score:.2f} | pmf={pmf_score:.6f}",
            ],
            20,
            34,
        )

        physics_y0 = header_height
        physics_y1 = physics_y0 + physics_frame.shape[0]
        canvas[physics_y0:physics_y1, : physics_frame.shape[1], :] = physics_frame
        cv2.putText(
            canvas,
            "Physics-IQ compare_side_by_side",
            (20, physics_y0 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        pmf_y0 = physics_y1 + gap
        pmf_y1 = pmf_y0 + pmf_frame.shape[0]
        canvas[pmf_y0:pmf_y1, : pmf_frame.shape[1], :] = pmf_frame
        cv2.putText(
            canvas,
            "PhysInOne PMF compare_side_by_side",
            (20, pmf_y0 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        output_frames.append(canvas)

    output_np = np.stack(output_frames, axis=0)
    write_video_h264_baseline(out_path, output_np, compare_fps)
    return {
        "compare_side_by_side": str(out_path),
        "compare_fps": compare_fps,
        "compare_frame_count": int(compare_count),
        "compare_shape": [int(output_np.shape[0]), int(output_np.shape[1]), int(output_np.shape[2]), int(output_np.shape[3])],
    }


def main() -> None:
    args = parse_args()
    pred_root = args.pred_root.expanduser().resolve()
    gt_full = args.gt_full.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_paths = sorted(pred_root.rglob(args.basename))
    if not pred_paths:
        raise FileNotFoundError(f"No files named {args.basename!r} found under {pred_root}")

    records: list[dict[str, object]] = []
    for pred_path in pred_paths:
        rel_parent = pred_path.parent.relative_to(pred_root)
        case_name = str(rel_parent).replace("/", "__")
        case_dir = out_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        physics_result = score_physics_iq(
            {"video": str(pred_path)},
            source_video_path=gt_full,
            threshold_value=int(args.physics_iq_threshold_value),
            downsample_factor=int(args.physics_iq_downsample_factor),
            context_mode=str(args.context_mode),
            context_frames=int(args.context_frames),
            aligned_video_dir=case_dir / "physics_iq",
        )
        pmf_result = score_pmf(
            {"video": str(pred_path)},
            source_video_path=gt_full,
            context_mode=str(args.context_mode),
            context_frames=int(args.context_frames),
            device=str(args.pmf_device),
            aligned_video_dir=case_dir / "pmf",
        )

        annotated_compare = build_annotated_compare_video(
            physics_compare_path=Path(str(physics_result["compare_side_by_side"])),
            pmf_compare_path=Path(str(pmf_result["compare_side_by_side"])),
            out_path=case_dir / "compare_side_by_side.mp4",
            title=str(rel_parent),
            context_mode=str(args.context_mode),
            physics_score=float(physics_result["score"]),
            pmf_score=float(pmf_result["score"]),
        )

        record = {
            "case_name": case_name,
            "relative_parent": str(rel_parent),
            "prediction_video": str(pred_path),
            "gt_full": str(gt_full),
            "context_mode": str(args.context_mode),
            "context_frames": int(args.context_frames),
            "physics_iq": physics_result,
            "pmf": pmf_result,
            "annotated_compare": annotated_compare,
        }
        records.append(record)
        (case_dir / "result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "pred_root": str(pred_root),
        "basename": str(args.basename),
        "gt_full": str(gt_full),
        "context_mode": str(args.context_mode),
        "context_frames": int(args.context_frames),
        "count": len(records),
        "records": records,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "case_name\trelative_parent\tphysics_iq_score\tpmf_score\tannotated_compare"
    ]
    for record in records:
        lines.append(
            "\t".join(
                [
                    str(record["case_name"]),
                    str(record["relative_parent"]),
                    f"{float(record['physics_iq']['score']):.2f}",
                    f"{float(record['pmf']['score']):.6f}",
                    str(record["annotated_compare"]["compare_side_by_side"]),
                ]
            )
        )
    (out_dir / "summary.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
