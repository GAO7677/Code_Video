#!/usr/bin/env python3
"""Create short prefix clips and JSON cases for frame-budget comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[8, 16])
    return parser.parse_args()


def write_prefix(source: Path, target: Path, count: int) -> int:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot create video: {target}")
    written = 0
    try:
        while written < count:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        capture.release()
        writer.release()
    if written != count:
        raise RuntimeError(f"Requested {count} frames but wrote {written}: {source}")
    return written


def main() -> None:
    args = parse_args()
    source = args.video.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_list = args.output_dir / "prefix_cases.txt"
    lines: list[str] = []
    for count in sorted(set(args.frames)):
        if count <= 0:
            raise ValueError("Frame counts must be positive")
        variant = f"{args.case_id}_first{count}"
        video_path = args.output_dir / "videos" / f"rgb_cycles_first{count}.mp4"
        json_path = args.output_dir / "cases" / f"{variant}.json"
        written = write_prefix(source, video_path, count)
        payload = {
            "case_id": variant,
            "input_caption": f"prefix comparison / first {count} frames",
            "source_video": str(video_path),
            "prefix_frame_count": count,
            "prefix_source_case": args.case_id,
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines.append(str(json_path))
        print(f"created={variant} frames={written} video={video_path}", flush=True)
    case_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"case_list={case_list}", flush=True)


if __name__ == "__main__":
    main()
