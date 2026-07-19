#!/usr/bin/env python3
"""Run the four GT VAE/coordinate variants used by the attention-transfer test."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_DATASET = Path("/data/gaoya/agent-data/datasets/physiciq_selected_qk")
DEFAULT_CACHE = Path("/data/gaoya/agent-data/cache/physiciq_selected_sam2_regions")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_gt_attention_transfer_l23_s39"
)
VARIANTS = (
    ("gt_framewise_stretch", "framewise_anchors", "cache"),
    ("gt_framewise_cover_crop", "framewise_anchors", "cover_crop"),
    ("gt_whole_stretch", "whole_video", "cache"),
    ("gt_whole_cover_crop", "whole_video", "cover_crop"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def case_keys(dataset_root: Path) -> list[str]:
    keys = []
    for path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys.append(str(payload["case_key"]))
    if not keys:
        raise RuntimeError(f"no cases found under {dataset_root}")
    return keys


def command(args: argparse.Namespace, variant: tuple[str, str, str], case_key: str) -> list[str]:
    name, vae_mode, coordinate_mode = variant
    result = [
        str(PYTHON),
        str(HERE / "analyze_wan_gt_toy_worker.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--analysis-region-cache-root", str(args.cache_root.resolve()),
        "--output-dir", str((args.output_root / name).resolve()),
        "--worker-id", "0",
        "--num-workers", "1",
        "--case-keys", case_key,
        "--video-field", "source_video",
        "--vae-encode-mode", vae_mode,
        "--query-coordinate-mode", coordinate_mode,
        "--sampling-steps", "40",
        "--analysis-layers", "23",
        "--analysis-step-indices", "39",
        "--analysis-visualize-layer", "23",
        "--analysis-visualize-step-index", "39",
        "--analysis-no-hidden",
        "--save-attention-probabilities",
        "--device", "cuda:0",
    ]
    if args.overwrite:
        result.append("--overwrite")
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = deque((variant, key) for variant in VARIANTS for key in case_keys(args.dataset_root))
    available = deque(str(gpu) for gpu in args.gpus)
    running: list[tuple[str, str, str, subprocess.Popen, object, Path]] = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(
        (
            "/home/gaoya/Code_Video/DiffTrack-main",
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
            "/home/gaoya/Code_Video/DiffSynth-Studio-main",
            "/home/gaoya/Code_Video/Code_data/Code_train/train_0419",
        )
    )
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    while tasks or running:
        while tasks and available:
            variant, key = tasks.popleft()
            gpu = available.popleft()
            name = variant[0]
            log_path = args.output_root / name / "logs" / f"{key}_gpu{gpu}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("w", encoding="utf-8")
            env = environment.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(
                command(args, variant, key),
                cwd=str(HERE.parent),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((gpu, name, key, process, handle, log_path))
            print(f"start {name}/{key} on GPU {gpu}, pid={process.pid}", flush=True)
        time.sleep(5)
        survivors = []
        for gpu, name, key, process, handle, log_path in running:
            code = process.poll()
            if code is None:
                survivors.append((gpu, name, key, process, handle, log_path))
                continue
            handle.close()
            available.append(gpu)
            if code != 0:
                print(f"FAILED {name}/{key} on GPU {gpu}: {log_path}", file=sys.stderr)
                for _, _, _, child, child_handle, _ in survivors:
                    child.terminate()
                    child_handle.close()
                raise SystemExit(code)
            print(f"complete {name}/{key} on GPU {gpu}", flush=True)
        running = survivors


if __name__ == "__main__":
    main()
