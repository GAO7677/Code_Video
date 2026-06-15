#!/usr/bin/env python3
"""
Prepare a cleaned D benchmark view for context-aware evaluation.

What this script does:
- reads ABD_test/D/_meta/source_manifest.json
- analyzes motion in each source_video
- trims a 3s context clip around the most informative motion window
- regenerates the first frame PNG for the trimmed clip
- writes per-case meta JSON files compatible with batch_eval_lora.py
- writes an all-cases meta list and a high-motion meta list

This keeps the original benchmark data untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np


DEFAULT_SOURCE_MANIFEST = Path(
    "/data/gaoya/AAA_test_video/Output_try0526/ABD_test/D/_meta/source_manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/ABD_test/D_clean")
DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_MOTION_WINDOW_SECONDS = 1.5
DEFAULT_HIGH_MOTION_THRESHOLD = 0.20
DEFAULT_PRE_ROLL_SECONDS = 0.75
DEFAULT_KEEP_TOP_K = 16


@dataclass
class MotionInfo:
    fps: float
    frame_count: int
    duration: float
    diffs: np.ndarray
    smoothed_diffs: np.ndarray
    best_window_start: int
    best_window_score: float
    onset_frame: int | None
    onset_score: float
    threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a cleaned D benchmark view.")
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--motion-window-seconds", type=float, default=DEFAULT_MOTION_WINDOW_SECONDS)
    parser.add_argument("--high-motion-threshold", type=float, default=DEFAULT_HIGH_MOTION_THRESHOLD)
    parser.add_argument("--keep-top-k", type=int, default=DEFAULT_KEEP_TOP_K)
    parser.add_argument("--pre-roll-seconds", type=float, default=DEFAULT_PRE_ROLL_SECONDS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sanitize(text: str) -> str:
    safe = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("._") or "sample"


def read_manifest(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")
    return data


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_motion_info(video_path: Path, motion_window_seconds: float) -> MotionInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0.0

    diffs: list[float] = []
    prev_gray: np.ndarray | None = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if prev_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            diffs.append(diff)
        prev_gray = gray
    cap.release()

    if not diffs:
        return MotionInfo(
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            diffs=np.asarray([], dtype=np.float32),
            smoothed_diffs=np.asarray([], dtype=np.float32),
            best_window_start=0,
            best_window_score=0.0,
            onset_frame=None,
            onset_score=0.0,
            threshold=0.0,
        )

    diffs_arr = np.asarray(diffs, dtype=np.float32)
    if len(diffs_arr) >= 3:
        smoothed = np.convolve(diffs_arr, np.ones(3, dtype=np.float32) / 3.0, mode="valid")
    else:
        smoothed = diffs_arr.copy()

    window_len = max(1, min(int(round(motion_window_seconds * fps)) - 1, len(smoothed)))
    best_start = 0
    best_score = -1.0
    if len(smoothed) >= window_len:
        for start in range(0, len(smoothed) - window_len + 1):
            score = float(smoothed[start : start + window_len].mean())
            if score > best_score:
                best_score = score
                best_start = start
    else:
        best_score = float(smoothed.mean())
        best_start = 0

    median = float(np.median(diffs_arr))
    mad = float(np.median(np.abs(diffs_arr - median)))
    threshold = max(median + 3.0 * mad, 0.18)
    onset_frame: int | None = None
    onset_score = 0.0
    for idx, value in enumerate(smoothed, start=2):
        if float(value) >= threshold:
            onset_frame = idx
            onset_score = float(value)
            break

    return MotionInfo(
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        diffs=diffs_arr,
        smoothed_diffs=np.asarray(smoothed, dtype=np.float32),
        best_window_start=best_start,
        best_window_score=float(best_score),
        onset_frame=onset_frame,
        onset_score=float(onset_score),
        threshold=float(threshold),
    )


def trim_context_video(
    source_video: Path,
    output_video: Path,
    fps: float,
    trim_start_sec: float,
    window_seconds: float,
) -> None:
    ensure_dir(output_video.parent)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        f"{trim_start_sec:.6f}",
        "-t",
        f"{window_seconds:.6f}",
        "-i",
        str(source_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    subprocess.run(cmd, check=True)


def save_first_frame(video_path: Path, output_png: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open trimmed video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read first frame from {video_path}")
    ensure_dir(output_png.parent)
    cv2.imwrite(str(output_png), frame)


def build_case_meta(
    *,
    item: dict[str, Any],
    source_video: Path,
    source_context_video: Path,
    trimmed_context_video: Path,
    first_frame_png: Path,
    motion: MotionInfo,
    trim_start_sec: float,
    window_seconds: float,
    case_dir: Path,
) -> dict[str, Any]:
    sample_name = str(Path(item["context_video"]).parent.name)
    category = str(item.get("category") or "unknown")
    case_key = f"{sanitize(category)}_{sample_name}"
    payload: dict[str, Any] = {
        "sample_id": sample_name,
        "category": category,
        "caption": str(item.get("caption") or ""),
        "paths": {
            "context_video_path": str(trimmed_context_video),
            "first_frame_path": str(first_frame_png),
            "source_video_path": str(source_video),
            "original_context_video_path": str(source_context_video),
            "original_first_frame_path": str(Path(item["first_frame"])),
        },
        "processing": {
            "case_key": case_key,
            "motion_window_seconds": DEFAULT_MOTION_WINDOW_SECONDS,
            "motion_threshold": motion.threshold,
            "best_window_start_frame": motion.best_window_start,
            "best_window_score": motion.best_window_score,
            "motion_onset_frame": motion.onset_frame,
            "motion_onset_score": motion.onset_score,
            "trim_start_sec": trim_start_sec,
            "trim_end_sec": trim_start_sec + window_seconds,
            "window_seconds": window_seconds,
            "fps": motion.fps,
            "frame_count": motion.frame_count,
            "duration_sec": motion.duration,
            "high_motion": motion.best_window_score >= motion.threshold,
        },
        "source": {
            "category": category,
            "source_video": str(source_video),
            "first_frame": str(Path(item["first_frame"])),
            "context_video": str(source_context_video),
        },
    }
    payload["paths"]["meta_json_path"] = str(case_dir / "meta.json")
    return payload


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    cases_root = args.output_root / "cases"
    meta_root = args.output_root / "_meta"
    ensure_dir(cases_root)
    ensure_dir(meta_root)

    items = read_manifest(args.source_manifest)
    scored_items: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        source_video = Path(item["source_video"])
        source_context_video = Path(item["context_video"])
        motion = compute_motion_info(source_video, args.motion_window_seconds)
        scored_items.append(
            {
                "index": index,
                "item": item,
                "source_video": source_video,
                "source_context_video": source_context_video,
                "motion": motion,
            }
        )

    scored_items.sort(
        key=lambda entry: (
            float(entry["motion"].best_window_score),
            float(entry["motion"].onset_score),
        ),
        reverse=True,
    )

    selected_items = [
        entry
        for entry in scored_items
        if entry["motion"].best_window_score >= args.high_motion_threshold
    ]
    if args.keep_top_k is not None and args.keep_top_k >= 0:
        selected_items = selected_items[: args.keep_top_k]

    selected_items.sort(key=lambda entry: int(entry["index"]))

    output_manifest: list[dict[str, Any]] = []
    all_meta_paths: list[str] = []
    selected_meta_paths: list[str] = []

    for rank, entry in enumerate(selected_items):
        item = entry["item"]
        index = int(entry["index"])
        source_video = entry["source_video"]
        source_context_video = entry["source_context_video"]
        motion = entry["motion"]
        sample_name = str(source_context_video.parent.name)
        category = str(item.get("category") or "unknown")
        case_dir = cases_root / f"{index:03d}_{sanitize(category)}_{sample_name}"
        trimmed_context_video = case_dir / "context_video.mp4"
        first_frame_png = case_dir / "first_frame.png"
        meta_json = case_dir / "meta.json"

        if meta_json.exists() and not args.overwrite:
            case_payload = json.loads(meta_json.read_text(encoding="utf-8"))
        else:
            if motion.onset_frame is not None:
                onset_sec = motion.onset_frame / motion.fps
                trim_start_sec = max(0.0, onset_sec - args.pre_roll_seconds)
            else:
                trim_start_sec = 0.0

            max_trim_start = max(0.0, motion.duration - args.window_seconds)
            trim_start_sec = min(trim_start_sec, max_trim_start)

            trim_context_video(
                source_video=source_video,
                output_video=trimmed_context_video,
                fps=motion.fps,
                trim_start_sec=trim_start_sec,
                window_seconds=args.window_seconds,
            )
            save_first_frame(trimmed_context_video, first_frame_png)

            case_payload = build_case_meta(
                item=item,
                source_video=source_video,
                source_context_video=source_context_video,
                trimmed_context_video=trimmed_context_video,
                first_frame_png=first_frame_png,
                motion=motion,
                trim_start_sec=trim_start_sec,
                window_seconds=args.window_seconds,
                case_dir=case_dir,
            )
            meta_json.write_text(json.dumps(case_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        output_manifest.append(case_payload)
        all_meta_paths.append(str(meta_json))
        if motion.best_window_score >= args.high_motion_threshold:
            selected_meta_paths.append(str(meta_json))
        print(
            f"[{rank + 1:02d}/{len(selected_items)}] {sample_name} "
            f"best={motion.best_window_score:.3f} onset={motion.onset_frame}"
        )

    manifest_path = meta_root / "source_manifest.json"
    manifest_path.write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_meta_list = meta_root / "meta_list.txt"
    all_meta_list.write_text("\n".join(all_meta_paths) + "\n", encoding="utf-8")

    high_motion_list = meta_root / "high_motion_meta_list.txt"
    high_motion_list.write_text("\n".join(selected_meta_paths) + "\n", encoding="utf-8")

    summary = {
        "total_cases": len(output_manifest),
        "high_motion_cases": len(selected_meta_paths),
        "high_motion_threshold": args.high_motion_threshold,
        "keep_top_k": args.keep_top_k,
        "window_seconds": args.window_seconds,
        "motion_window_seconds": args.motion_window_seconds,
    }
    (meta_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"meta_list: {all_meta_list}")
    print(f"high_motion_meta_list: {high_motion_list}")


if __name__ == "__main__":
    main()
