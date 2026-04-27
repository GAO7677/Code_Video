#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


DEFAULT_PREPARED_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physInOne_AB_pure/prepared")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/dataset/vLAR-PhysInOne/mytest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export PhysInOne pure A/B inputs and GT into per-sample folders."
    )
    parser.add_argument("--prepared_root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_rows(prepared_root: Path) -> list[dict]:
    rows: list[dict] = []
    for manifest_name in ["group_A_pure_source_manifest.jsonl", "group_B_pure_source_manifest.jsonl"]:
        manifest_path = prepared_root / manifest_name
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda item: item["sample_id"])
    return rows


def list_context_frames(context_dir: Path) -> list[Path]:
    frame_paths = sorted(
        path
        for path in context_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    )
    if not frame_paths:
        raise FileNotFoundError(f"No context frames found in {context_dir}")
    return frame_paths


def write_context_video(frame_paths: list[Path], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame_path in frame_paths:
            with Image.open(frame_path) as img:
                writer.append_data(np.asarray(img.convert("RGB")))


def load_full_video_frames(source_zip: Path, camera_name: str) -> list[np.ndarray]:
    with zipfile.ZipFile(source_zip) as zf:
        members = zf.namelist()
        suffix = f"/{camera_name}/rgb/"
        frame_names = sorted(
            name for name in members if suffix in name and name.endswith(".jpg")
        )
        if not frame_names:
            raise FileNotFoundError(
                f"No RGB frames found for {camera_name} in {source_zip}"
            )

        frames: list[np.ndarray] = []
        for frame_name in frame_names:
            with zf.open(frame_name) as fp:
                with Image.open(fp) as img:
                    frames.append(np.asarray(img.convert("RGB")))
    return frames


def write_rgb_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.prepared_root)
    aggregate: list[dict] = []

    for idx, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        sample_dir = args.output_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        context_dir = Path(row["context_frames_dir"])
        frame_paths = list_context_frames(context_dir)

        first_frame_src = frame_paths[0]
        ti2v_input_src = Path(row["image_path"])
        gt_src = Path(row["gt_video_path"])

        first_frame_dst = sample_dir / "first_frame.png"
        context_video_dst = sample_dir / "context_video.mp4"
        gt_dst = sample_dir / "future_gt_video.mp4"
        full_video_dst = sample_dir / "full_video.mp4"
        json_dst = sample_dir / "meta.json"

        shutil.copy2(first_frame_src, first_frame_dst)
        shutil.copy2(gt_src, gt_dst)

        fps = int(row.get("fps", 30))
        write_context_video(frame_paths, context_video_dst, fps=fps)
        full_video_frames = load_full_video_frames(
            source_zip=Path(row["source_zip"]),
            camera_name=str(row["camera_name"]),
        )
        write_rgb_video(full_video_frames, full_video_dst, fps=fps)

        sample_json = {
            "sample_id": sample_id,
            "caption": row["prompt"],
            "group_id": row.get("group_id"),
            "group_name": row.get("group_name"),
            "split": row.get("split"),
            "camera_name": row.get("camera_name"),
            "physics_types": row.get("physics_types", []),
            "selection_mode": row.get("selection_mode"),
            "source_zip": row.get("source_zip"),
            "fps": fps,
            "context_frames": row.get("context_frames"),
            "future_frames": row.get("future_frames"),
            "paths": {
                "sample_dir": str(sample_dir.resolve()),
                "future_gt_video_path": str(gt_dst.resolve()),
                "full_video_path": str(full_video_dst.resolve()),
                "context_video_path": str(context_video_dst.resolve()),
                "first_frame_path": str(first_frame_dst.resolve()),
            },
            "source_paths": {
                "original_gt_video_path": str(gt_src.resolve()),
                "original_context_frames_dir": str(context_dir.resolve()),
                "original_ti2v_input_image_path": str(ti2v_input_src.resolve()),
            },
        }
        json_dst.write_text(json.dumps(sample_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        aggregate.append(sample_json)
        print(f"[{idx}/{len(rows)}] {sample_id}")

    manifest_path = args.output_root / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for item in aggregate:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "output_root": str(args.output_root.resolve()),
        "sample_count": len(aggregate),
        "manifest_jsonl": str(manifest_path.resolve()),
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
