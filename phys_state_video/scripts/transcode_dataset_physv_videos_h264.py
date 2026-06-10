#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcode Dataset_physV videos to browser-friendly H.264 mp4 files.")
    parser.add_argument(
        "--input-root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos",
        help="Source video root.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos_h264",
        help="Target video root.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="H.264 constant rate factor. Lower means higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 preset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing transcoded files.",
    )
    return parser.parse_args()


def transcode_file(ffmpeg_exe: str, src: Path, dst: Path, crf: int, preset: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite and dst.stat().st_size > 0:
        print(f"[skip] {dst}")
        return

    cmd = [
        ffmpeg_exe,
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[ok] {dst}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")

    files = sorted(input_root.rglob("*.mp4"))
    print(f"[info] input_root={input_root}")
    print(f"[info] output_root={output_root}")
    print(f"[info] files={len(files)}")
    for src in files:
        rel = src.relative_to(input_root)
        dst = output_root / rel
        transcode_file(ffmpeg_exe, src, dst, args.crf, args.preset, args.overwrite)
    print("[done]")


if __name__ == "__main__":
    main()
