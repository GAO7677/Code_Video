#!/usr/bin/env python3
"""Prepare one deterministic 30-case PyBullet train validation set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import PyBullet0713NoGTBoxDataset


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5")
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/xssc_train_validation_30cases")
PYTHON_FFMPEG = "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"


def write_video(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = [
        PYTHON_FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
        "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]
    subprocess.run(cmd, input=frames.tobytes(), check=True)


def sample_key(index: int, metadata: dict) -> str:
    raw = str(metadata.get("sample_key") or f"row_{index:06d}")
    return raw.replace("/", "_").replace(" ", "_")[:180]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not args.force:
        print(manifest_path)
        return
    root.mkdir(parents=True, exist_ok=True)
    dataset = PyBullet0713NoGTBoxDataset(
        root=str(ROOT), split="train", resolution=(512, 896),
        num_frames=49, num_context_frames=8, sampling_strategy="prefix",
    )
    if len(dataset) < args.count:
        raise RuntimeError(f"PyBullet train has {len(dataset)} samples, need {args.count}")
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), args.count))
    records, input_paths = [], []
    for pos, index in enumerate(indices, start=1):
        sample = dataset[index]
        metadata = dict(sample.get("metadata", {}))
        case_id = f"case_{pos:02d}_{index:07d}_{sample_key(index, metadata)}"
        frames = sample["video"].detach().float().cpu().permute(1, 2, 3, 0).numpy()
        frames = np.clip((frames + 1.0) * 127.5, 0, 255).round().astype(np.uint8)
        media = root / "media" / case_id
        gt = media / "gt_49f.mp4"
        context = media / "context_08f.mp4"
        image = media / "first_frame.jpg"
        if args.force or not gt.is_file():
            write_video(gt, frames)
            write_video(context, frames[:8])
            Image.fromarray(frames[0]).save(image, quality=95)
        payload = {
            "source_video": str(gt), "input_video": str(context),
            "input_image": str(image), "input_caption": str(sample["caption"]),
            "training_dataset_source": "pybullet", "training_dataset_index": int(index),
            "original_video_path": str(sample["video_path"]),
        }
        input_path = root / "inputs" / f"{case_id}.json"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        input_paths.append(input_path)
        records.append({
            "case_id": case_id, "source": "pybullet", "source_index": int(index),
            "sample_key": str(metadata.get("sample_key", case_id)),
            "prompt": str(sample["caption"]), "input_json": str(input_path),
            "gt_video": str(gt), "context_video": str(context),
            "original_video_path": str(sample["video_path"]),
        })
    inputs = root / "inputs" / "cases.txt"
    inputs.write_text("".join(f"{p}\n" for p in input_paths), encoding="utf-8")
    payload = {
        "schema_version": 1, "seed": args.seed, "count": args.count,
        "dataset": "PyBullet0713NoGTBoxDataset", "split": "train",
        "sampling_strategy": "prefix", "resolution": [512, 896],
        "num_frames": 49, "context_frames": 8, "indices": indices,
        "cases": records,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
