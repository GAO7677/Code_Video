#!/usr/bin/env python3
"""Convert the VGGT gallery's referenced MP4 files to browser-compatible H.264."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio_ffmpeg

from code_vjepa_vggt.visualize_vggt_0717_manifest_cases import write_json


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vggt_0717_train10_context8_prefix49"
)


def h264_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_h264.mp4")


def convert_one(input_path: Path, output_path: Path, ffmpeg: str) -> Path:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        temporary = output_path.with_suffix(".tmp.mp4")
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-threads",
            "1",
            str(temporary),
        ]
        subprocess.run(command, check=True)
        temporary.replace(output_path)
    with output_path.open("rb") as handle:
        header = handle.read()
    if b"avc1" not in header:
        raise RuntimeError(f"H.264 avc1 marker missing from {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    jobs: list[tuple[Path, Path]] = []
    result_paths = sorted(root.glob("*/cases/*/result.json"))
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        for relative in result.get("videos", {}).values():
            if not relative:
                continue
            input_path = root / str(relative)
            jobs.append((input_path, h264_path(input_path)))
    unique_jobs = list(dict.fromkeys(jobs))
    print(f"converting {len(unique_jobs)} unique gallery videos with {args.workers} workers", flush=True)
    converted: dict[Path, Path] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(convert_one, input_path, output_path, ffmpeg): (input_path, output_path)
            for input_path, output_path in unique_jobs
        }
        for future in as_completed(futures):
            input_path, output_path = futures[future]
            future.result()
            converted[input_path] = output_path
            print(f"converted {len(converted)}/{len(unique_jobs)} {input_path.name}", flush=True)

    updated_cases = 0
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "ok":
            continue
        videos = dict(result.get("videos", {}))
        for key, relative in videos.items():
            if not relative:
                continue
            original = root / str(relative)
            converted_path = converted[original]
            videos[key] = str(converted_path.relative_to(root))
        result["videos"] = videos
        result["video_encoding"] = "H.264/AVC (avc1), yuv420p, browser-compatible"
        write_json(result_path, result)
        updated_cases += 1
    print(f"updated {updated_cases} result manifests", flush=True)


if __name__ == "__main__":
    main()
