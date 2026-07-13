#!/usr/bin/env python3
"""Build frame-aligned contact sheets, H.264 videos, and temporal diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-list", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Method specification LABEL=/path/to/output/root; may be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, default=Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"))
    parser.add_argument("--tile-width", type=int, default=448)
    parser.add_argument("--tile-height", type=int, default=256)
    parser.add_argument("--contact-frame-indices", default="0,7,8,16,24,32,40,48")
    return parser.parse_args()


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"method must use LABEL=PATH syntax: {value}")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError(f"method label is empty: {value}")
    if not path.is_dir():
        raise FileNotFoundError(f"method root not found: {path}")
    return label, path


def case_stems(path: Path) -> list[str]:
    stems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            stems.append(Path(value).stem)
    return stems


def resolve_case_video(root: Path, stem: str) -> Path:
    matches = sorted(root.rglob(f"{stem}.mp4"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {stem}.mp4 under {root}, found {len(matches)}: {matches}"
        )
    return matches[0]


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return frames, fps


def letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def labeled_tile(
    frame: np.ndarray,
    *,
    label: str,
    frame_index: int,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = letterbox(frame, width, height)
    cv2.rectangle(canvas, (0, 0), (width, 29), (0, 0, 0), thickness=-1)
    cv2.putText(
        canvas,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"frame {frame_index:02d}",
        (width - 92, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def frame_mae(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.abs(first.astype(np.float32) - second.astype(np.float32)).mean() / 255.0)


def temporal_metrics(frames: list[np.ndarray], context_frames: int = 8) -> dict[str, float | int]:
    future_start = min(context_frames, len(frames) - 1)
    boundary_mae = frame_mae(frames[future_start - 1], frames[future_start]) if future_start > 0 else 0.0
    future = frames[future_start:]
    motion = [frame_mae(a, b) for a, b in zip(future, future[1:])]
    anchor_drift = [frame_mae(frames[future_start - 1], frame) for frame in future] if future_start > 0 else []
    laplacian_variance = [
        float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        for frame in future
    ]
    return {
        "frame_count": len(frames),
        "prefix_future_boundary_mae": boundary_mae,
        "future_adjacent_motion_mean": float(np.mean(motion)) if motion else 0.0,
        "future_adjacent_motion_max": float(np.max(motion)) if motion else 0.0,
        "future_anchor_drift_mean": float(np.mean(anchor_drift)) if anchor_drift else 0.0,
        "future_anchor_drift_max": float(np.max(anchor_drift)) if anchor_drift else 0.0,
        "future_laplacian_variance_mean": float(np.mean(laplacian_variance)) if laplacian_variance else 0.0,
    }


def write_h264_comparison(
    *,
    output_path: Path,
    method_frames: list[tuple[str, list[np.ndarray]]],
    fps: float,
    width: int,
    height: int,
    ffmpeg: Path,
) -> None:
    frame_count = min(len(frames) for _, frames in method_frames)
    temporary_path = output_path.with_suffix(".temporary.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height * len(method_frames)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary_path}")
    for frame_index in range(frame_count):
        rows = [
            labeled_tile(
                frames[frame_index],
                label=label,
                frame_index=frame_index,
                width=width,
                height=height,
            )
            for label, frames in method_frames
        ]
        writer.write(np.concatenate(rows, axis=0))
    writer.release()
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    temporary_path.unlink()


def main() -> None:
    args = parse_args()
    methods = [parse_method(value) for value in args.method]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_indices = [int(value.strip()) for value in args.contact_frame_indices.split(",") if value.strip()]
    metric_rows: list[dict[str, object]] = []
    manifest: dict[str, object] = {"methods": [], "cases": []}

    for label, root in methods:
        manifest["methods"].append({"label": label, "root": str(root)})

    for stem in case_stems(args.case_list.expanduser().resolve()):
        loaded: list[tuple[str, Path, list[np.ndarray], float]] = []
        for label, root in methods:
            video_path = resolve_case_video(root, stem)
            frames, fps = read_video(video_path)
            loaded.append((label, video_path, frames, fps))
            metric_rows.append(
                {
                    "case": stem,
                    "method": label,
                    "video": str(video_path),
                    **temporal_metrics(frames),
                }
            )

        common_frame_count = min(len(frames) for _, _, frames, _ in loaded)
        frame_indices = [min(max(0, index), common_frame_count - 1) for index in requested_indices]
        contact_rows = []
        for label, _, frames, _ in loaded:
            tiles = [
                labeled_tile(
                    frames[index],
                    label=label,
                    frame_index=index,
                    width=int(args.tile_width),
                    height=int(args.tile_height),
                )
                for index in frame_indices
            ]
            contact_rows.append(np.concatenate(tiles, axis=1))
        contact_path = output_dir / f"{stem}_contact_sheet.jpg"
        if not cv2.imwrite(str(contact_path), np.concatenate(contact_rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 94]):
            raise RuntimeError(f"failed to write contact sheet: {contact_path}")

        comparison_path = output_dir / f"{stem}_aligned_comparison_h264.mp4"
        write_h264_comparison(
            output_path=comparison_path,
            method_frames=[(label, frames) for label, _, frames, _ in loaded],
            fps=loaded[0][3],
            width=int(args.tile_width),
            height=int(args.tile_height),
            ffmpeg=args.ffmpeg.expanduser().resolve(),
        )
        manifest["cases"].append(
            {
                "case": stem,
                "frame_count": common_frame_count,
                "contact_sheet": str(contact_path),
                "comparison_video": str(comparison_path),
            }
        )
        print(comparison_path)

    metrics_path = output_dir / "temporal_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    manifest["temporal_metrics"] = str(metrics_path)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
