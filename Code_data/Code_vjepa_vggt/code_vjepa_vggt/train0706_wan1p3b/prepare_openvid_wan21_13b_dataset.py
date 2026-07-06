#!/usr/bin/env python3
"""
Build a Wan2.1-1.3B-ready OpenVid parquet dataset root from newly downloaded shards.

This script:
1. Filters parquet rows with the same rules used by training.
2. Writes a filtered `train/` parquet directory.
3. Generates config JSON files that existing Wan2.1-1.3B training scripts can consume.
4. Runs a smoke read through the training dataset loader.

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/prepare_openvid_wan21_13b_dataset.py \
  --input-root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
  --output-root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code_vjepa_vggt.train0706_wan1p3b.dataset import WanTI2VDataset


DEFAULT_TEMPLATE_MIX_CONFIG = THIS_DIR / "dataset_mix_config.json"
DEFAULT_FILTER_SCRIPT = THIS_DIR / "filter_openvid_parquet.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a Wan2.1-1.3B OpenVid training dataset root from parquet shards."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Directory containing source OpenVid parquet shards.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dataset root to create. The filtered parquet files will be placed under <output-root>/train.",
    )
    parser.add_argument(
        "--template-mix-config",
        type=Path,
        default=DEFAULT_TEMPLATE_MIX_CONFIG,
        help="Template mixed dataset config used by the Wan2.1-1.3B OpenVid LoRA training recipe.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=24,
        help="Minimum frame count required by the target OpenVid LoRA recipe.",
    )
    parser.add_argument("--height", type=int, default=384, help="Smoke-check height.")
    parser.add_argument("--width", type=int, default=672, help="Smoke-check width.")
    parser.add_argument("--dataset-repeat", type=int, default=1, help="Repeat value to write into generated OpenVid config.")
    parser.add_argument("--max-files", type=int, default=None, help="Debug option passed to the filter script.")
    parser.add_argument("--max-rows-per-file", type=int, default=None, help="Debug option passed to the filter script.")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip loader smoke validation after export.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete the output root first if it already exists.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_filter(args: argparse.Namespace, train_root: Path, report_root: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(DEFAULT_FILTER_SCRIPT),
        "--input-root",
        str(args.input_root.resolve()),
        "--output-root",
        str(train_root.resolve()),
        "--report-root",
        str(report_root.resolve()),
        "--num-frames",
        str(int(args.num_frames)),
        "--keep-going",
    ]
    if args.max_files is not None:
        cmd.extend(["--max-files", str(int(args.max_files))])
    if args.max_rows_per_file is not None:
        cmd.extend(["--max-rows-per-file", str(int(args.max_rows_per_file))])

    print("[prepare] filter command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return json.loads((report_root / "summary.json").read_text(encoding="utf-8"))


def build_openvid_only_config(train_root: Path, repeat: int) -> list[dict[str, Any]]:
    return [
        {
            "type": "openvid",
            "path": str(train_root.resolve()),
            "repeat": int(repeat),
        }
    ]


def build_mixed_config(template_path: Path, train_root: Path, repeat: int) -> list[dict[str, Any]]:
    data = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Template mix config must be a non-empty list: {template_path}")

    replaced = False
    result = []
    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"Template mix config item must be a dict, got {type(item).__name__}")
        entry = dict(item)
        item_type = str(entry.get("type", "")).strip().lower()
        if not replaced and item_type == "openvid":
            entry["path"] = str(train_root.resolve())
            entry["repeat"] = int(repeat)
            replaced = True
        result.append(entry)

    if not replaced:
        raise ValueError(f"Template mix config does not contain an OpenVid entry: {template_path}")
    return result


def run_openvid_smoke(config_path: Path, height: int, width: int, num_frames: int) -> dict[str, Any]:
    dataset = WanTI2VDataset(
        dataset_base_path=str(config_path.resolve()),
        dataset_metadata_path="",
        dataset_repeat=1,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
    )
    sample = dataset.dataset[0]
    video = sample["video"]
    first_frame = video[0]
    return {
        "prompt_preview": str(sample.get("prompt", ""))[:200],
        "num_frames_loaded": int(len(video)),
        "first_frame_shape": list(first_frame.size[::-1]) + [3] if hasattr(first_frame, "size") else None,
        "dataset_stats": dataset.dataset_stats,
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    template_mix_config = args.template_mix_config.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not template_mix_config.is_file():
        raise FileNotFoundError(f"Template mix config not found: {template_mix_config}")

    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --force to replace it."
            )
        shutil.rmtree(output_root)

    train_root = output_root / "train"
    report_root = output_root / "reports"
    config_root = output_root / "configs"
    meta_root = output_root / "meta"
    output_root.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    filter_summary = run_filter(args=args, train_root=train_root, report_root=report_root)

    openvid_only_config = build_openvid_only_config(train_root=train_root, repeat=args.dataset_repeat)
    mixed_config = build_mixed_config(
        template_path=template_mix_config,
        train_root=train_root,
        repeat=args.dataset_repeat,
    )

    openvid_only_config_path = config_root / "openvid_only_config.json"
    mixed_config_path = config_root / "dataset_mix_config_wan21_13b.json"
    write_json(openvid_only_config_path, openvid_only_config)
    write_json(mixed_config_path, mixed_config)

    smoke_summary = None
    if not args.skip_smoke:
        smoke_summary = run_openvid_smoke(
            config_path=openvid_only_config_path,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
        )
        write_json(meta_root / "smoke_summary.json", smoke_summary)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "train_root": str(train_root),
        "report_root": str(report_root),
        "openvid_only_config": str(openvid_only_config_path),
        "mixed_config": str(mixed_config_path),
        "num_frames_required": int(args.num_frames),
        "filter_summary": filter_summary,
        "smoke_summary": smoke_summary,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "suggested_train_command": (
            "DATASET_CONFIG="
            + str(mixed_config_path)
            + " CUDA_VISIBLE_DEVICES=3,5,6,7 sh "
            + str(THIS_DIR / "run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh")
        ),
    }
    write_json(meta_root / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
