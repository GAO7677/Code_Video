#!/usr/bin/env python3
"""Remove xSSC's clean V2V prefix and create official 5-second MP4 files."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
FFPROBE = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")
FPS = 24
RAW_FRAMES = 189
PREFIX_OUTPUT_FRAMES = 69
OUTPUT_FRAMES = 120
OUTPUT_SECONDS = 5.0


def probe(path: Path) -> tuple[int, float, float]:
    payload = json.loads(
        subprocess.check_output(
            [
                str(FFPROBE),
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,nb_read_frames:format=duration",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    stream = payload["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return (
        int(stream["nb_read_frames"]),
        float(numerator) / float(denominator),
        float(payload["format"]["duration"]),
    )


def output_is_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        frames, fps, duration = probe(path)
    except (KeyError, ValueError, subprocess.SubprocessError):
        return False
    return (
        frames == OUTPUT_FRAMES
        and abs(fps - FPS) < 1e-6
        and abs(duration - OUTPUT_SECONDS) < 0.001
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-folder", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_folder = args.raw_folder.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    json_paths = [
        Path(line.strip())
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_count = len(json_paths)
    if not 1 <= expected_count <= 198:
        raise RuntimeError(f"expected between 1 and 198 input JSON files, found {expected_count}")

    expected_raw = {f"{path.stem}.mp4" for path in json_paths}
    actual_raw = {path.name for path in raw_folder.glob("*.mp4")}
    if actual_raw != expected_raw:
        missing = sorted(expected_raw - actual_raw)
        extra = sorted(actual_raw - expected_raw)
        raise RuntimeError(f"raw MP4 set mismatch; missing={missing}, extra={extra}")

    for index, json_path in enumerate(json_paths, start=1):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        output_name = payload["generated_video_name"]
        if not output_name.startswith(f"{index:04d}_"):
            raise RuntimeError(f"invalid official output order: {output_name}")
        source = raw_folder / f"{json_path.stem}.mp4"
        raw_frames, raw_fps, _ = probe(source)
        if raw_frames != RAW_FRAMES or abs(raw_fps - FPS) >= 1e-6:
            raise RuntimeError(
                f"raw video must be {RAW_FRAMES} frames at {FPS} FPS: {source}; "
                f"got frames={raw_frames}, fps={raw_fps}"
            )

        target = output_folder / output_name
        if not args.force and output_is_valid(target):
            continue
        temporary = target.with_suffix(".tmp.mp4")
        command = [
            str(FFMPEG),
            "-v",
            "error",
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            (
                f"trim=start_frame={PREFIX_OUTPUT_FRAMES}:end_frame={RAW_FRAMES},"
                f"setpts=PTS-STARTPTS,fps={FPS}"
            ),
            "-frames:v",
            str(OUTPUT_FRAMES),
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
            str(temporary),
        ]
        subprocess.run(command, check=True)
        if not output_is_valid(temporary):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"official output validation failed: {target}")
        temporary.replace(target)
        print(f"[{index:03d}/{expected_count:03d}] {target.name}")

    outputs = sorted(output_folder.glob("*.mp4"))
    if len(outputs) != expected_count or not all(output_is_valid(path) for path in outputs):
        raise RuntimeError(
            f"final folder is not a complete validated {expected_count}-video run"
        )
    print(f"Official 5-second run folder: {output_folder}")


if __name__ == "__main__":
    main()
