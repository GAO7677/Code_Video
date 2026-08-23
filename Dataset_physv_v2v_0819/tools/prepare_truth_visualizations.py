#!/usr/bin/env python3
"""Publish per-sample depth and instance-ID truth videos for the overlay viewer."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import cv2


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay"
)
DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_FFMPEG = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg")
CONTEXT_FRAME_COUNT = 8

TRUTH_STREAMS = {
    "instance_ids": {
        "video": Path("videos/masks.mp4"),
        "npz": Path("raw/instance_ids.npz"),
        "label": "实例 ID 真值",
        "context_note": "前 8 帧的全物体逐像素实例 ID 可视化",
        "source_note": "完整视频的全物体逐像素实例 ID 可视化",
    },
    "depth": {
        "video": Path("videos/depth.mp4"),
        "npz": Path("raw/depth.npz"),
        "label": "深度真值",
        "context_note": "前 8 帧的深度真值伪彩可视化",
        "source_note": "完整视频的深度真值伪彩可视化",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="veryfast")
    return parser.parse_args()


def video_spec(path: Path) -> dict[str, int | float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    try:
        return {
            "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
            "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        }
    finally:
        capture.release()


def make_context_video(
    ffmpeg: Path,
    source: Path,
    target: Path,
    crf: int,
    preset: str,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.tmp{target.suffix}")
    if temporary.exists():
        temporary.unlink()
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-frames:v",
        str(CONTEXT_FRAME_COUNT),
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
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(target)


def relative_url(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def make_trajectory_view(case: dict) -> dict:
    return {
        "label": "红色轨迹 + 动态 GT mask",
        "camera": "Cycles",
        "context8": case["context8_overlay"],
        "source": case["source_overlay"],
        "width": case["width"],
        "height": case["height"],
        "fps": case["fps"],
        "context_note": "frame 0-7：红色轨迹与对应动态物体 GT mask",
        "source_note": "完整 Cycles 视频：红色轨迹逐帧累积，mask 为仿真动态物体真值",
        "raw_source": case.get("mask_source", ""),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    dataset_root = args.dataset_root.resolve()
    ffmpeg = args.ffmpeg.resolve()
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    if not cases:
        raise RuntimeError("overlay manifest has no cases")

    truth_dir = output_root / "truth_videos"
    for index, case in enumerate(cases, start=1):
        sample_id = str(case["sample_id"])
        sample_dir = dataset_root / "samples" / sample_id
        views = {"trajectory_mask": make_trajectory_view(case)}
        for key, definition in TRUTH_STREAMS.items():
            source = sample_dir / definition["video"]
            raw_source = sample_dir / definition["npz"]
            if not source.is_file() or not raw_source.is_file():
                raise FileNotFoundError(f"{sample_id}: missing {source} or {raw_source}")
            source_target = truth_dir / f"{sample_id}__{key}_source.mp4"
            context_target = truth_dir / f"{sample_id}__{key}_ctx8.mp4"
            source_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, source_target)
            make_context_video(ffmpeg, source_target, context_target, args.crf, args.preset)
            source_spec = video_spec(source_target)
            context_spec = video_spec(context_target)
            if context_spec["frame_count"] != CONTEXT_FRAME_COUNT:
                raise RuntimeError(
                    f"{sample_id}: {key} context has {context_spec['frame_count']} frames, "
                    f"expected {CONTEXT_FRAME_COUNT}"
                )
            views[key] = {
                "label": definition["label"],
                "camera": "原始仿真相机",
                "context8": relative_url(context_target, output_root),
                "source": relative_url(source_target, output_root),
                "width": source_spec["width"],
                "height": source_spec["height"],
                "fps": source_spec["fps"],
                "context_note": definition["context_note"],
                "source_note": definition["source_note"],
                "raw_source": str(raw_source),
            }
        case["visualizations"] = views
        print(f"[{index:02d}/{len(cases)}] {sample_id}", flush=True)

    manifest["schema_version"] = "physv_cycles_trajectory_overlay_v3"
    manifest["visualization_modes"] = {
        "trajectory_mask": "Cycles video with red trajectory and dynamic-object GT mask",
        "instance_ids": "Simulator-camera visualization of raw/instance_ids.npz",
        "depth": "Simulator-camera visualization of raw/depth.npz",
    }
    manifest["truth_visualization_note"] = (
        "Depth and instance-ID views use the original simulator camera at 1280x720. "
        "They are GT visualizations, not SAM/SAM2 predictions."
    )
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"WROTE {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
