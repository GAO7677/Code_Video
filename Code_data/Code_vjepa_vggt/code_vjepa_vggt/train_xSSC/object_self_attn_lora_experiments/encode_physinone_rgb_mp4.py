#!/usr/bin/env python3
"""Create rgb.mp4 beside caption.txt for completed PhysInOne cases."""

from __future__ import annotations

import argparse
import json
import os
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2


def case_dirs(root: Path):
    # The downloader creates and atomically renames temporary case directories.
    # A recursive glob can therefore observe a directory just as it is removed.
    # Treat that race as a normal polling condition instead of terminating the
    # long-running encoder.
    try:
        paths = list(root.glob("**/.complete.json"))
    except FileNotFoundError:
        return
    yield from (path.parent for path in paths)


def encode_case(case_dir: Path) -> bool:
    trajectory_dirs = sorted(case_dir.glob("*_trajectory"))
    if not trajectory_dirs:
        return False
    trajectory_dir = trajectory_dirs[0]
    output_path = trajectory_dir / "rgb.mp4"
    if output_path.exists() and output_path.stat().st_size > 0:
        return False
    rgb_dirs = sorted(trajectory_dir.glob("CineCamera_*/rgb"))
    if len(rgb_dirs) != 1:
        raise RuntimeError(f"Expected one selected RGB directory, got {rgb_dirs}")
    rgb_dir = rgb_dirs[0]
    frames = sorted(rgb_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No RGB frames found in {rgb_dir}")
    camera_json = next(trajectory_dir.glob("blender_CineCamera_*.json"), None)
    fps = 30.0
    if camera_json is not None:
        payload = json.loads(camera_json.read_text(encoding="utf-8"))
        fps = float(payload.get("fps") or 30.0)

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read {frames[0]}")
    height, width = first.shape[:2]
    partial = output_path.with_name("rgb.mp4.partial")
    partial.unlink(missing_ok=True)
    container = av.open(str(partial), mode="w", format="mp4")
    try:
        stream = container.add_stream(
            "libx264",
            rate=Fraction(fps).limit_denominator(1000),
        )
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for frame_path in frames:
            bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot read {frame_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            video_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    except Exception:
        container.close()
        partial.unlink(missing_ok=True)
        raise
    else:
        container.close()
    os.replace(partial, output_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=1700)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        encoded = 0
        failures = 0
        for case_dir in case_dirs(args.root):
            try:
                if encode_case(case_dir):
                    encoded += 1
            except Exception as exc:
                failures += 1
                print(f"FAILED {case_dir}: {exc}", flush=True)
        try:
            mp4_count = sum(1 for _ in args.root.glob("**/rgb.mp4"))
            complete_count = sum(1 for _ in args.root.glob("**/.complete.json"))
        except FileNotFoundError:
            mp4_count = 0
            complete_count = 0
        print(
            f"MP4 progress: {mp4_count}/{args.expected_cases}; "
            f"complete_cases={complete_count}; new={encoded}; failures={failures}",
            flush=True,
        )
        if args.once or complete_count >= args.expected_cases:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
