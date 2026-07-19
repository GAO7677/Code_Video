#!/usr/bin/env python3
"""Validate xSSC batch inference outputs, decoded video pixels, and JSON numerics."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from decord import VideoReader, cpu


def _find_nonfinite(value, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"{path}={value}")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(_find_nonfinite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_find_nonfinite(child, f"{path}[{index}]"))
    return errors


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-json-list", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=49)
    parser.add_argument("--expected-height", type=int, default=512)
    parser.add_argument("--expected-width", type=int, default=896)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_paths = [
        Path(line.strip())
        for line in args.input_json_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    expected_stems = sorted({path.stem for path in expected_paths})
    step_dir = args.output_root / str(summary["step"])
    errors: list[str] = []

    if int(summary["num_total"]) != len(expected_paths):
        errors.append(
            f"summary total {summary['num_total']} != input entries {len(expected_paths)}"
        )
    if int(summary["num_failed"]) or int(summary["num_skipped"]):
        errors.append(
            f"failed={summary['num_failed']} skipped={summary['num_skipped']}"
        )
    if int(summary["num_success"]) != len(expected_paths):
        errors.append(
            f"success {summary['num_success']} != input entries {len(expected_paths)}"
        )

    videos: list[dict[str, object]] = []
    for stem in expected_stems:
        video_path = step_dir / f"{stem}.mp4"
        if not video_path.is_file() or video_path.stat().st_size <= 0:
            errors.append(f"missing or empty video: {video_path}")
            continue
        try:
            reader = VideoReader(str(video_path), ctx=cpu(0))
            frame_count = len(reader)
            frames = reader.get_batch(list(range(frame_count))).asnumpy()
            finite = bool(np.isfinite(frames).all())
            pixel_std = float(frames.astype(np.float32).std())
            pixel_min = int(frames.min())
            pixel_max = int(frames.max())
            shape = list(frames.shape)
            if frame_count != args.expected_frames:
                errors.append(f"{video_path}: frames={frame_count}")
            if shape[1:3] != [args.expected_height, args.expected_width]:
                errors.append(f"{video_path}: decoded shape={shape}")
            if not finite:
                errors.append(f"{video_path}: non-finite decoded pixels")
            if pixel_std <= 1.0 or pixel_max - pixel_min <= 4:
                errors.append(
                    f"{video_path}: suspicious pixels min={pixel_min} max={pixel_max} std={pixel_std}"
                )
            videos.append(
                {
                    "path": str(video_path),
                    "shape": shape,
                    "pixel_min": pixel_min,
                    "pixel_max": pixel_max,
                    "pixel_std": pixel_std,
                    "finite": finite,
                }
            )
        except Exception as exc:
            errors.append(f"{video_path}: decode failed: {exc}")

    json_files = sorted(args.output_root.rglob("*.json"))
    for json_path in json_files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{json_path}: invalid JSON: {exc}")
            continue
        nonfinite = _find_nonfinite(payload)
        if nonfinite:
            errors.append(f"{json_path}: non-finite values: {nonfinite[:5]}")

    report = {
        "status": "passed" if not errors else "failed",
        "summary": summary,
        "input_entries": len(expected_paths),
        "unique_cases": len(expected_stems),
        "validated_videos": len(videos),
        "validated_json_files": len(json_files),
        "videos": videos,
        "errors": errors,
    }
    report_path = args.report or (args.output_root / "health_report.json")
    _atomic_write(report_path, report)
    print(json.dumps({key: report[key] for key in ("status", "input_entries", "unique_cases", "validated_videos", "validated_json_files", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
