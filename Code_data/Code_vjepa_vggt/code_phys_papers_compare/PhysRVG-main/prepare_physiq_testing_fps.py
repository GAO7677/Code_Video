from __future__ import annotations

import argparse
import math
import os
import subprocess
from pathlib import Path

import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare physics-IQ verified split-videos/testing/<int(fps)>FPS from "
            "the canonical 30FPS testing videos using ffmpeg."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, required=True)
    parser.add_argument("--source-fps-dirname", type=str, default="30FPS")
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    return parser.parse_args()


def _target_frame_count(target_fps: float, clip_seconds: float) -> int:
    return int(round(float(target_fps) * float(clip_seconds)))


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    source_dir = benchmark_root / "split-videos" / "testing" / args.source_fps_dirname
    target_dir = benchmark_root / "split-videos" / "testing" / f"{int(args.target_fps)}FPS"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if not source_dir.is_dir():
        raise FileNotFoundError(f"source testing dir not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    video_paths = sorted(source_dir.glob("*.mp4"))
    if not video_paths:
        raise RuntimeError(f"no mp4 files found under {source_dir}")

    frame_count = _target_frame_count(float(args.target_fps), float(args.clip_seconds))
    fps_expr = f"{frame_count}/{float(args.clip_seconds):.12g}"
    print(f"[prepare-fps] ffmpeg={ffmpeg_exe}")
    print(f"[prepare-fps] source_dir={source_dir}")
    print(f"[prepare-fps] target_dir={target_dir}")
    print(f"[prepare-fps] target_fps={args.target_fps}")
    print(f"[prepare-fps] clip_seconds={args.clip_seconds}")
    print(f"[prepare-fps] target_frame_count={frame_count}")
    print(f"[prepare-fps] files={len(video_paths)}")

    for index, video_path in enumerate(video_paths, start=1):
        output_path = target_dir / video_path.name.replace(args.source_fps_dirname, f"{int(args.target_fps)}FPS")
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"[prepare-fps] skip {index}/{len(video_paths)} {output_path.name}")
            continue

        tmp_path = output_path.with_suffix(".tmp.mp4")
        if tmp_path.exists():
            tmp_path.unlink()

        cmd = [
            ffmpeg_exe,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps_expr}",
            "-frames:v",
            str(frame_count),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tmp_path),
        ]
        subprocess.run(cmd, check=True)
        os.replace(tmp_path, output_path)
        print(f"[prepare-fps] done {index}/{len(video_paths)} {output_path.name}")

    print("[prepare-fps] complete")


if __name__ == "__main__":
    main()
