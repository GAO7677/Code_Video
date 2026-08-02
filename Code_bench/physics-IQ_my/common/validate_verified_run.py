#!/usr/bin/env python3
"""Validate model outputs against the reusable Physics-IQ Verified contract."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
from pathlib import Path


OFFICIAL_REPO = Path(
    "/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main"
)
DEFAULT_DESCRIPTIONS = (
    OFFICIAL_REPO / "descriptions" / "best_practice" / "descriptions_base.csv"
)
DEFAULT_FFPROBE = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")
DURATION_SECONDS = 5.0
DURATION_TOLERANCE = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folders", nargs="+", type=Path)
    parser.add_argument("--descriptions-file", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def expected_names(descriptions_file: Path) -> list[str]:
    with descriptions_file.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if "_take-1_" in row["scenario"]
        ]
    rows.sort(key=lambda row: int(row["scenario"].split("_", 1)[0]))
    names = [row["generated_video_name"] for row in rows]
    if len(names) != 198 or len(set(names)) != 198:
        raise RuntimeError(
            f"official descriptions must define 198 unique take-1 outputs, got {len(names)}"
        )
    for index, name in enumerate(names, start=1):
        if not name.startswith(f"{index:04d}_"):
            raise RuntimeError(f"non-contiguous official generated name: {name}")
    return names


def probe_video(ffprobe: Path, path: Path) -> dict[str, object]:
    payload = json.loads(
        subprocess.check_output(
            [
                str(ffprobe),
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "name": path.name,
        "frames": int(stream["nb_read_frames"]),
        "fps": float(numerator) / float(denominator),
        "duration": float(payload["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def validate_folder(
    folder: Path,
    names: list[str],
    ffprobe: Path,
    workers: int,
) -> dict[str, object]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"run folder not found: {folder}")

    expected = set(names)
    actual = {path.name for path in folder.iterdir() if path.is_file() and path.suffix == ".mp4"}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    non_mp4 = sorted(
        path.name
        for path in folder.iterdir()
        if path.is_file() and path.suffix != ".mp4" and not path.name.startswith(".")
    )
    if missing or extra or non_mp4:
        raise RuntimeError(
            f"invalid run folder {folder}: missing={missing}, extra={extra}, non_mp4={non_mp4}"
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        metadata = list(
            executor.map(lambda name: probe_video(ffprobe, folder / name), names)
        )

    invalid_duration = [
        row
        for row in metadata
        if abs(float(row["duration"]) - DURATION_SECONDS) > DURATION_TOLERANCE
    ]
    if invalid_duration:
        raise RuntimeError(
            f"videos outside 5.000 +/- 0.001 seconds: {invalid_duration[:10]}"
        )

    fps_values = {round(float(row["fps"]), 6) for row in metadata}
    if len(fps_values) != 1:
        raise RuntimeError(f"run contains inconsistent FPS values: {sorted(fps_values)}")
    fps = fps_values.pop()
    if fps <= 0 or abs(fps - round(fps)) > 1e-6:
        raise RuntimeError(f"official pipeline requires a positive integer FPS, got {fps}")

    return {
        "folder": str(folder),
        "num_videos": len(metadata),
        "fps": fps,
        "duration_seconds": DURATION_SECONDS,
        "frame_counts": sorted({int(row["frames"]) for row in metadata}),
        "resolutions": sorted(
            {f"{row['width']}x{row['height']}" for row in metadata}
        ),
    }


def main() -> None:
    args = parse_args()
    if not args.ffprobe.is_file():
        raise FileNotFoundError(f"ffprobe not found: {args.ffprobe}")
    names = expected_names(args.descriptions_file.expanduser().resolve())
    summaries = [
        validate_folder(folder, names, args.ffprobe, args.workers)
        for folder in args.run_folders
    ]
    result = {
        "benchmark": "Physics-IQ Verified",
        "descriptions_file": str(args.descriptions_file.expanduser().resolve()),
        "runs": summaries,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
