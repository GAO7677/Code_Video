from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Physics-IQ submission folder to exact 30 FPS / 150 frames / 5 seconds "
            "for official benchmark evaluation."
        )
    )
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--target-frames", type=int, default=150)
    parser.add_argument("--copy-json", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _probe_video(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, frames


def _transcode_one(
    *,
    ffmpeg_exe: str,
    input_path: Path,
    output_path: Path,
    target_fps: float,
    target_frames: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(input_path),
        "-vf",
        f"fps={target_fps:.12g}",
        "-frames:v",
        str(target_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_path, output_path)


def main() -> None:
    args = parse_args()
    input_folder = args.input_folder.expanduser().resolve()
    output_folder = args.output_folder.expanduser().resolve()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if not input_folder.is_dir():
        raise FileNotFoundError(f"input folder not found: {input_folder}")

    output_folder.mkdir(parents=True, exist_ok=True)
    mp4_paths = sorted(input_folder.glob("*.mp4"))
    if not mp4_paths:
        raise RuntimeError(f"no mp4 files found under {input_folder}")

    print(f"[prepare-submission] ffmpeg={ffmpeg_exe}")
    print(f"[prepare-submission] input_folder={input_folder}")
    print(f"[prepare-submission] output_folder={output_folder}")
    print(f"[prepare-submission] target_fps={args.target_fps}")
    print(f"[prepare-submission] target_frames={args.target_frames}")
    print(f"[prepare-submission] num_videos={len(mp4_paths)}")

    summary: dict[str, object] = {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "target_fps": float(args.target_fps),
        "target_frames": int(args.target_frames),
        "videos": [],
    }

    for index, input_path in enumerate(mp4_paths, start=1):
        output_path = output_folder / input_path.name
        if output_path.exists() and not args.overwrite:
            fps, frames = _probe_video(output_path)
            if abs(fps - float(args.target_fps)) < 1e-3 and frames == int(args.target_frames):
                print(f"[prepare-submission] skip {index}/{len(mp4_paths)} {output_path.name}")
                summary["videos"].append(
                    {
                        "name": input_path.name,
                        "status": "skipped_existing",
                        "output_fps": fps,
                        "output_frames": frames,
                    }
                )
                continue

        _transcode_one(
            ffmpeg_exe=ffmpeg_exe,
            input_path=input_path,
            output_path=output_path,
            target_fps=float(args.target_fps),
            target_frames=int(args.target_frames),
        )
        out_fps, out_frames = _probe_video(output_path)
        print(
            f"[prepare-submission] done {index}/{len(mp4_paths)} {output_path.name} "
            f"(fps={out_fps:.6f}, frames={out_frames})"
        )
        summary["videos"].append(
            {
                "name": input_path.name,
                "status": "converted",
                "output_fps": out_fps,
                "output_frames": out_frames,
            }
        )

    if args.copy_json:
        for json_path in sorted(input_folder.glob("*.json")):
            shutil.copy2(json_path, output_folder / json_path.name)

    summary_path = output_folder / "_prepare_submission_30fps_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"[prepare-submission] summary={summary_path}")


if __name__ == "__main__":
    main()
