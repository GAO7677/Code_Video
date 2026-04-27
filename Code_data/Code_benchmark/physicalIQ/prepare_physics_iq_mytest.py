#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATASET_ROOT / "mytest"
DEFAULT_DESCRIPTIONS_CSV = Path(
    "/home/gaoya/Code_Video/physics-IQ-benchmark-main/descriptions/descriptions.csv"
)
FPS = 30
CONTEXT_FRAMES = 90
FUTURE_FRAMES = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Physics-IQ take-1 samples into per-sample folders."
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--descriptions_csv", type=Path, default=DEFAULT_DESCRIPTIONS_CSV)
    return parser.parse_args()


def load_take1_rows(descriptions_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with descriptions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"]
            if "take-1" not in scenario:
                continue
            if "perspective-center" not in row["generated_video_name"]:
                continue
            rows.append(row)
    rows.sort(key=lambda item: item["generated_video_name"])
    return rows


def switch_frame_name(generated_video_name: str) -> str:
    stem = Path(generated_video_name).stem
    file_id, perspective, scenario = stem.split("_", 2)
    return f"{file_id}_switch-frames_anyFPS_{perspective}_{scenario}.jpg"


def filename_from_scenario(scenario: str, video_type: str, fps: int) -> str:
    file_id, perspective, take, scenario_name = Path(scenario).stem.split("_", 3)
    return f"{file_id}_{video_type}_{fps}FPS_{perspective}_{take}_{scenario_name}.mp4"


def copy_switch_frame_to_png(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img.convert("RGB").save(dst)


def load_video_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def write_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)


def write_frame(frame: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    full_root = args.dataset_root / "full-videos" / "take-1" / "30FPS"
    switch_root = args.dataset_root / "switch-frames"

    rows = load_take1_rows(args.descriptions_csv)
    aggregate: list[dict[str, object]] = []

    for idx, row in enumerate(rows, start=1):
        scenario = row["scenario"]
        generated_video_name = row["generated_video_name"]
        sample_id = Path(generated_video_name).stem
        sample_dir = args.output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        full_src = full_root / filename_from_scenario(scenario, "full-videos", 30)
        switch_src = switch_root / switch_frame_name(generated_video_name)

        context_dst = sample_dir / "context_video.mp4"
        future_gt_dst = sample_dir / "future_gt_video.mp4"
        first_frame_dst = sample_dir / "first_frame.png"
        meta_dst = sample_dir / "meta.json"

        for src in [full_src, switch_src]:
            if not src.exists():
                raise FileNotFoundError(f"Missing required source file: {src}")

        full_frames = load_video_frames(full_src)
        expected_total = CONTEXT_FRAMES + FUTURE_FRAMES
        if len(full_frames) < expected_total:
            raise ValueError(
                f"Full video {full_src} has only {len(full_frames)} frames, expected at least {expected_total}"
            )
        context_frames = full_frames[:CONTEXT_FRAMES]
        future_frames = full_frames[CONTEXT_FRAMES:CONTEXT_FRAMES + FUTURE_FRAMES]
        switch_frame = full_frames[CONTEXT_FRAMES - 1]

        write_video(context_frames, context_dst, fps=FPS)
        write_video(future_frames, future_gt_dst, fps=FPS)
        write_frame(switch_frame, first_frame_dst)

        perspective = sample_id.split("_", 2)[1]
        scenario_slug = sample_id.split("_", 2)[2]
        sample_json = {
            "sample_id": sample_id,
            "caption": row["description"],
            "category": row["category"],
            "scenario": row["scenario"],
            "generated_video_name": row["generated_video_name"],
            "take": "take-1",
            "fps": FPS,
            "perspective": perspective,
            "scenario_slug": scenario_slug,
            "context_frame_range": [0, CONTEXT_FRAMES - 1],
            "future_frame_range": [CONTEXT_FRAMES, CONTEXT_FRAMES + FUTURE_FRAMES - 1],
            "first_frame_index_in_full_video": CONTEXT_FRAMES - 1,
            "paths": {
                "sample_dir": str(sample_dir.resolve()),
                "full_video_path": str(full_src.resolve()),
                "context_video_path": str(context_dst.resolve()),
                "future_gt_video_path": str(future_gt_dst.resolve()),
                "first_frame_path": str(first_frame_dst.resolve()),
            },
            "source_paths": {
                "original_full_video_path": str(full_src.resolve()),
                "original_switch_frame_path": str(switch_src.resolve()),
            },
        }
        meta_dst.write_text(json.dumps(sample_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        aggregate.append(sample_json)
        print(f"[{idx}/{len(rows)}] {sample_id}")

    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in aggregate:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "output_root": str(args.output_root.resolve()),
        "sample_count": len(aggregate),
        "manifest_jsonl": str(manifest_path.resolve()),
        "descriptions_csv": str(args.descriptions_csv.resolve()),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
