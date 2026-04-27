#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image

import sys

TAXONOMY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data")
if str(TAXONOMY_ROOT) not in sys.path:
    sys.path.insert(0, str(TAXONOMY_ROOT))

from physinone_benchmark_taxonomy import GROUP_ID_TO_GROUP, classify_sample_name  # noqa: E402

from physinone_benchmark_common import build_prompt, write_json, write_jsonl  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/vLAR-PhysInOne")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physInOne_AB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PhysInOne A/B benchmark inputs.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--groups", nargs="*", default=["A", "B"])
    parser.add_argument(
        "--selection_mode",
        choices=["contains", "pure"],
        default="contains",
        help="contains: sample is selected if it includes the group. pure: sample belongs only to that group.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["SinglePhysics", "DoublePhysics", "TriplePhysics", "RootLevel"],
    )
    parser.add_argument("--camera_name", default="CineCamera_0")
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--future_frames", type=int, default=41)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_zip_stem(name: str) -> str:
    if name.endswith(".zip"):
        name = name[:-4]
    return name


def split_name_for_path(path: Path) -> str:
    return path.parent.name if path.parent.name in {"SinglePhysics", "DoublePhysics", "TriplePhysics"} else "RootLevel"


def should_keep(sample_groups: set[str], target_group: str, mode: str) -> bool:
    if mode == "contains":
        return target_group in sample_groups
    if mode == "pure":
        return sample_groups == {target_group}
    raise ValueError(f"Unsupported selection mode: {mode}")


def camera_sort_key(camera_name: str) -> Tuple[int, str]:
    suffix = camera_name.rsplit("_", 1)[-1]
    return (int(suffix), camera_name) if suffix.isdigit() else (10**9, camera_name)


def resolve_camera_name(zip_path: Path, preferred_camera_name: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
    cameras = sorted(
        {
            member.rsplit("/rgb/0000.jpg", 1)[0].rsplit("/", 1)[-1]
            for member in members
            if member.endswith("/rgb/0000.jpg") and "/CineCamera_" in member
        },
        key=camera_sort_key,
    )
    if not cameras:
        raise FileNotFoundError(f"Could not find any CineCamera rgb frames in {zip_path}")
    return preferred_camera_name if preferred_camera_name in cameras else cameras[0]


def read_zip_rgb_frames(zip_path: Path, camera_name: str, num_frames: int) -> Tuple[str, List[Image.Image]]:
    frames: List[Image.Image] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = set(zf.namelist())
        resolved_camera_name = resolve_camera_name(zip_path, camera_name)

        camera_prefix = None
        frame0_suffix = f"/{resolved_camera_name}/rgb/0000.jpg"
        for member in members:
            if member.endswith(frame0_suffix):
                camera_prefix = member[: -len(frame0_suffix)]
                break
        if camera_prefix is None:
            raise FileNotFoundError(f"Could not find {resolved_camera_name}/rgb/0000.jpg in {zip_path}")
        for idx in range(num_frames):
            member = f"{camera_prefix}/{resolved_camera_name}/rgb/{idx:04d}.jpg"
            if member not in members:
                raise FileNotFoundError(f"Missing frame {member} in {zip_path}")
            raw = zf.read(member)
            frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    return resolved_camera_name, frames


def save_rgb_frames(frames: List[Image.Image], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: List[str] = []
    for idx, frame in enumerate(frames):
        path = out_dir / f"frame_{idx:03d}.png"
        frame.save(path)
        frame_paths.append(str(path))
    return frame_paths


def save_video(frames: List[Image.Image], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    prepared_rows: Dict[str, List[Dict[str, Any]]] = {group_id: [] for group_id in args.groups}
    counts: Dict[str, int] = {group_id: 0 for group_id in args.groups}

    for zip_path in sorted(args.dataset_root.rglob("*.zip")):
        split = split_name_for_path(zip_path)
        if split not in set(args.splits):
            continue

        info = classify_sample_name(zip_path.name)
        sample_groups = {group["group_id"] for group in info["benchmark_groups"]}
        if not sample_groups:
            continue

        for group_id in args.groups:
            if not should_keep(sample_groups, group_id, args.selection_mode):
                continue

            group_meta = GROUP_ID_TO_GROUP[group_id]
            sample_stem = normalize_zip_stem(zip_path.name)
            actual_camera_name = resolve_camera_name(zip_path, args.camera_name)
            sample_id = f"{group_id}__{sample_stem}__{actual_camera_name}"
            sample_root = args.output_root / "prepared" / group_id / sample_id

            total_needed = args.context_frames + args.future_frames
            if args.overwrite or not (sample_root / "future_gt.mp4").exists():
                resolved_camera_name, frames = read_zip_rgb_frames(zip_path, actual_camera_name, total_needed)
                context = frames[: args.context_frames]
                future = frames[args.context_frames : total_needed]

                save_rgb_frames(context, sample_root / "context_frames")
                context[-1].save(sample_root / "input_image.png")
                save_video(future, sample_root / "future_gt.mp4", fps=args.fps)

            if not (sample_root / "future_gt.mp4").exists():
                raise FileNotFoundError(f"Missing prepared future_gt.mp4 after processing {zip_path}")

            prompt = build_prompt(group_meta["group_name"], info["physics_types"], actual_camera_name)
            row = {
                "sample_id": sample_id,
                "prompt": prompt,
                "image_path": str((sample_root / "input_image.png").resolve()),
                "context_frames_dir": str((sample_root / "context_frames").resolve()),
                "gt_video_path": str((sample_root / "future_gt.mp4").resolve()),
                "generated_start_frame": 0,
                "gt_start_frame": 0,
                "physics_types": info["physics_types"],
                "benchmark_groups": info["benchmark_groups"],
                "group_id": group_id,
                "group_name": group_meta["group_name"],
                "group_slug": group_meta["group_slug"],
                "selection_mode": args.selection_mode,
                "split": split,
                "camera_name": actual_camera_name,
                "source_zip": str(zip_path.resolve()),
                "context_frames": args.context_frames,
                "future_frames": args.future_frames,
                "fps": args.fps,
            }
            prepared_rows[group_id].append(row)
            counts[group_id] += 1

    summary = {
        "selection_mode": args.selection_mode,
        "camera_name": args.camera_name,
        "context_frames": args.context_frames,
        "future_frames": args.future_frames,
        "fps": args.fps,
        "counts": counts,
    }
    write_json(args.output_root / "prepared" / "prepare_summary.json", summary)
    for group_id, rows in prepared_rows.items():
        out_path = args.output_root / "prepared" / f"group_{group_id}_{args.selection_mode}_source_manifest.jsonl"
        write_jsonl(out_path, rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
