#!/usr/bin/env python3
"""Batch Blender/Cycles previews for existing PhysV samples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_CACHE_ROOT = Path("/data/gaoya/agent-data/cache/physv_cycles_previews")
BLENDER_SCRIPT = Path(__file__).with_name("render_physv_cycles.py")
DEFAULT_BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_ids", nargs="+")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=Path(shutil.which("ffmpeg") or Path(sys.executable).with_name("ffmpeg")),
    )
    parser.add_argument(
        "--ffprobe",
        type=Path,
        default=None,
        help="Optional ffprobe binary when it is not beside --ffmpeg.",
    )
    parser.add_argument("--gpu", default="7", help="Physical GPU index exposed to Blender.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--keep-frames", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def video_info(path: Path, ffprobe: Path) -> dict:
    result = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def export_trajectory_json(source: Path, target: Path) -> int:
    arrays = np.load(source, allow_pickle=False)
    object_names = [str(value) for value in arrays["object_names"]]
    payload: dict[str, object] = {
        "object_names": object_names,
        "frame_times_s": arrays["frame_times_s"].tolist(),
    }
    for name in object_names:
        payload[f"{name}_positions"] = arrays[f"{name}_positions"].tolist()
        payload[f"{name}_rotations"] = arrays[f"{name}_rotations"].tolist()
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return len(payload["frame_times_s"])


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    cache_root = args.cache_root.resolve()
    ffprobe = args.ffprobe or args.ffmpeg.with_name("ffprobe")
    if not args.ffmpeg.is_file() or not ffprobe.is_file():
        raise FileNotFoundError(f"ffmpeg/ffprobe not found beside {args.ffmpeg}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for case_id in args.case_ids:
        sample_dir = dataset_root / "samples" / case_id
        if not (sample_dir / "metadata.json").is_file():
            raise FileNotFoundError(sample_dir / "metadata.json")
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        fps = int(metadata["simulation"]["fps"])
        work_dir = cache_root / case_id
        frames_dir = work_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        trajectory_json = work_dir / "trajectories.json"
        expected_frames = export_trajectory_json(sample_dir / "raw" / "trajectories.npz", trajectory_json)
        run_checked(
            [
                str(args.blender), "-b", "--python", str(BLENDER_SCRIPT), "--",
                "--sample-dir", str(sample_dir),
                "--trajectory-json", str(trajectory_json),
                "--output-dir", str(frames_dir),
                "--width", str(args.width),
                "--height", str(args.height),
                "--samples", str(args.samples),
                "--exposure", str(args.exposure),
                "--engine", args.engine,
                "--device", "CUDA",
                "--output-format", "PNG",
            ],
            env=env,
        )
        render_report = frames_dir / "render_metadata.json"
        rendered_frames = sorted(frames_dir.glob("frame_*.png"))
        if not render_report.is_file() or len(rendered_frames) != expected_frames:
            raise RuntimeError(
                f"Blender render incomplete for {case_id}: "
                f"metadata={render_report.is_file()}, frames={len(rendered_frames)}/{expected_frames}"
            )
        target = sample_dir / "videos" / "rgb_cycles.mp4"
        temporary = target.with_suffix(".tmp.mp4")
        run_checked(
            [
                str(args.ffmpeg), "-y", "-loglevel", "warning",
                "-framerate", str(fps), "-start_number", "1",
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ]
        )
        temporary.replace(target)
        render_metadata = json.loads(render_report.read_text(encoding="utf-8"))
        render_metadata["video"] = video_info(target, ffprobe)
        render_metadata["output_video"] = str(target)
        (sample_dir / "videos" / "rgb_cycles.json").write_text(
            json.dumps(render_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not args.keep_frames:
            shutil.rmtree(frames_dir)
        print(json.dumps({"case_id": case_id, "video": str(target), "probe": video_info(target, ffprobe)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
