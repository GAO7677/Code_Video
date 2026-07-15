from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare exact 5-second Physics-IQ videos.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=5.0)
    return parser.parse_args()


def probe(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return frames, fps


def main() -> None:
    args = parse_args()
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    expected_frames = round(args.fps * args.duration)
    args.output_root.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for index, entry in enumerate(entries, 1):
        source = args.input_root / entry["generated_video_name"]
        target = args.output_root / entry["generated_video_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.is_file() and probe(target) == (expected_frames, float(args.fps)):
            print(f"[skip] {index}/{len(entries)} {target.name}")
            continue
        temporary = target.with_suffix(".tmp.mp4")
        subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source), "-vf", f"fps={args.fps}", "-frames:v",
                str(expected_frames), "-an", "-c:v", "libx264", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ],
            check=True,
        )
        temporary.replace(target)
        frames, fps = probe(target)
        if frames != expected_frames or abs(fps - args.fps) > 0.01:
            raise RuntimeError(f"invalid submission video: {target} frames={frames} fps={fps}")
        print(f"[done] {index}/{len(entries)} {target.name}")


if __name__ == "__main__":
    main()
