#!/usr/bin/env python3
"""Render a texture-only CYCLES refinement without changing strict GT."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
EXPERIMENT_ROOT = STRICT_ROOT / "refine/R001_v2v_obstacle_v140_basketball_texture"
BLENDER_SCRIPT = PROJECT_ROOT / "scripts/render_physv_cycles.py"
DEFAULT_BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
DEFAULT_FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
CASE_ID = "v2v_obstacle_v140"
TEXTURE_SOURCE_URL = "https://lpc.opengameart.org/content/basket-ball-texture"
TEXTURE_FILE_URL = "https://lpc.opengameart.org/sites/default/files/balldimpled.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--gpu", default="0", help="Physical GPU exposed to Blender; GPU 4 is forbidden.")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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


def video_info(path: Path, ffmpeg: Path) -> dict:
    ffprobe = ffmpeg.with_name("ffprobe")
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


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def encode_video(frame_dir: Path, output: Path, fps: int, ffmpeg: Path) -> None:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"no rendered frames in {frame_dir}")
    temporary = output.with_suffix(".tmp.mp4")
    run_checked(
        [
            str(ffmpeg), "-y", "-loglevel", "warning", "-framerate", str(fps),
            "-i", str(frame_dir / "frame_%04d.png"),
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ]
    )
    temporary.replace(output)


def encode_prefix(input_video: Path, output: Path, frame_count: int, ffmpeg: Path) -> None:
    """Write an exact context prefix without changing the full render."""
    temporary = output.with_suffix(".tmp.mp4")
    run_checked(
        [
            str(ffmpeg), "-y", "-loglevel", "warning", "-i", str(input_video),
            "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "slow",
            "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temporary),
        ]
    )
    temporary.replace(output)


def main() -> None:
    args = parse_args()
    if str(args.gpu) == "4":
        raise ValueError("GPU 4 is forbidden by workspace policy")
    sample_dir = STRICT_ROOT / "samples" / CASE_ID
    texture = EXPERIMENT_ROOT / "assets/balldimpled.png"
    override_json = EXPERIMENT_ROOT / "assets/material_overrides.json"
    output_dir = EXPERIMENT_ROOT / "render" / args.mode
    frames_dir = output_dir / "_frames"
    output_video = output_dir / "rgb_cycles.mp4"
    trajectory_json = output_dir / "trajectories.json"
    render_report_path = frames_dir / "render_metadata.json"
    manifest_path = EXPERIMENT_ROOT / "experiment.json"
    if not sample_dir.is_dir():
        raise FileNotFoundError(sample_dir)
    if not texture.is_file():
        raise FileNotFoundError(texture)
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    if not args.ffmpeg.is_file() or not args.ffmpeg.with_name("ffprobe").is_file():
        raise FileNotFoundError(f"ffmpeg/ffprobe pair not found beside {args.ffmpeg}")
    if output_video.exists() and not args.force:
        raise FileExistsError(f"output exists; use --force to replace: {output_video}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    expected_frames = export_trajectory_json(sample_dir / "raw/trajectories.npz", trajectory_json)
    frame_limit = 3 if args.mode == "smoke" else 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_checked(
        [
            str(args.blender), "-b", "--python", str(BLENDER_SCRIPT), "--",
            "--sample-dir", str(sample_dir),
            "--trajectory-json", str(trajectory_json),
            "--output-dir", str(frames_dir),
            "--width", "896", "--height", "512",
            "--samples", str(args.samples), "--exposure", "0",
            "--frame-limit", str(frame_limit),
            "--engine", "CYCLES", "--device", "CUDA", "--output-format", "PNG",
            "--material-overrides-json", str(override_json),
            "--basketball-texture", str(texture),
        ],
        env=env,
    )
    rendered_frames = sorted(frames_dir.glob("frame_*.png"))
    expected_rendered = min(frame_limit, expected_frames) if frame_limit else expected_frames
    if not render_report_path.is_file() or len(rendered_frames) != expected_rendered:
        raise RuntimeError(
            f"incomplete render: metadata={render_report_path.is_file()} "
            f"frames={len(rendered_frames)}/{expected_rendered}"
        )

    metadata = json.loads(render_report_path.read_text(encoding="utf-8"))
    fps = int(json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))["simulation"]["fps"])
    encode_video(frames_dir, output_video, fps, args.ffmpeg)
    context_videos = {}
    if args.mode == "full":
        for context_frames in (8, 16):
            context_output = output_dir / f"context{context_frames}_cycles.mp4"
            encode_prefix(output_video, context_output, context_frames, args.ffmpeg)
            context_videos[f"context{context_frames}"] = {
                "path": str(context_output),
                "frames": context_frames,
                "video": video_info(context_output, args.ffmpeg),
            }
    metadata.update(
        {
            "refine_schema_version": "physv_cycles_refine_texture_v1",
            "experiment_id": EXPERIMENT_ROOT.name,
            "parent_case": CASE_ID,
            "change_scope": "RGB material only; geometry, physics, camera, trajectory and strict GT unchanged",
            "texture_source": {
                "name": "balldimpled.png",
                "page_url": TEXTURE_SOURCE_URL,
                "file_url": TEXTURE_FILE_URL,
                "license": "CC-BY 3.0",
                "attribution": "Downdate; collaborator Charlie",
                "local_file": str(texture),
            },
            "strict_protocol": {"width": 896, "height": 512, "fps": fps, "frame_count": expected_rendered},
            "output_video": str(output_video),
            "video": video_info(output_video, args.ffmpeg),
            "context_videos": context_videos,
            "truth_inheritance": str(EXPERIMENT_ROOT / "truth_inheritance/inheritance.json"),
        }
    )
    (output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.mode == "full":
        for frame in rendered_frames:
            frame.unlink()
        render_report_path.unlink()
        frames_dir.rmdir()
    manifest = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema_version": "physv_cycles_refine_experiment_v1",
            "experiment_id": EXPERIMENT_ROOT.name,
            "parent_case": CASE_ID,
            "change_scope": "RGB material only; geometry, physics, camera, trajectory and strict GT unchanged",
            "texture": {
                "path": str(texture),
                "page_url": TEXTURE_SOURCE_URL,
                "file_url": TEXTURE_FILE_URL,
                "license": "CC-BY 3.0",
                "sha256": "678e7487fcdd6fb016d09affe028053b2fd1f29046152c57ae0f80524c257587",
            },
            "truth_inheritance": str(EXPERIMENT_ROOT / "truth_inheritance/inheritance.json"),
        }
    )
    runs = manifest.setdefault("runs", {})
    runs[args.mode] = {
        "status": "completed",
        "video": str(output_video),
        "frames": expected_rendered,
        "samples": args.samples,
        "probe": metadata["video"],
        "context_videos": context_videos,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": args.mode, "video": str(output_video), "frames": expected_rendered, "probe": metadata["video"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
