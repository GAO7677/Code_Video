#!/usr/bin/env python3
"""Generate 49-frame Stage1b, LoRA, and baseline Q/K + CoTracker results."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_DATASET = Path("/data/gaoya/agent-data/datasets/physiciq_selected_qk")
DEFAULT_CACHE = Path("/data/gaoya/agent-data/cache/physiciq_selected_sam2_regions")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk_49f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("stage1b", "lora", "baseline", "gt"),
        default=["stage1b", "lora", "baseline"],
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def command_for(model: str, args: argparse.Namespace, worker_id: int) -> list[str]:
    common = [
        "--dataset-root", str(args.dataset_root.resolve()),
        "--analysis-region-cache-root", str(args.cache_root.resolve()),
        "--output-dir", str((args.output_root / model).resolve()),
        "--worker-id", str(worker_id),
        "--num-workers", str(len(args.gpus)),
        "--num-frames", "49",
        "--context-frames", "8",
        "--sampling-steps", "40",
        "--analysis-layers", "23",
        "--analysis-step-indices", "39",
        "--analysis-visualize-layer", "23",
        "--analysis-visualize-step-index", "39",
        "--analysis-no-hidden",
    ]
    if args.overwrite:
        common.append("--overwrite")
    if model == "stage1b":
        return [
            str(PYTHON),
            str(HERE / "run_stage1b_kubric_analysis_worker.py"),
            "--checkpoint",
            "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
            "train_stage1b_kubric0708/checkpoints/step-004000",
            "--analysis-device", "cuda:0",
            *common,
        ]
    if model == "gt":
        return [
            str(PYTHON),
            str(HERE / "analyze_wan_gt_toy_worker.py"),
            "--video-field", "source_video",
            "--vae-encode-mode", "whole_video",
            "--query-coordinate-mode", "cache",
            "--allow-short-gt",
            "--device", "cuda:0",
            *common,
        ]
    command = [str(PYTHON), str(HERE / "run_lorav2v_toy_analysis_worker.py")]
    if model == "baseline":
        command.append("--base-model-only")
    command.extend(
        ["--device", "cuda:0", "--analysis-query-coordinate-mode", "cover_crop", *common]
    )
    return command


def run_model(model: str, args: argparse.Namespace) -> None:
    log_dir = args.output_root / model / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["PYTHONPATH"] = ":".join(
        (
            "/home/gaoya/Code_Video/DiffTrack-main",
            "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
            "/home/gaoya/Code_Video/DiffSynth-Studio-main",
            "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
            "/home/gaoya/Code_Video/Code_data/Code_train/train_0419",
        )
    )
    processes = []
    handles = []
    for worker_id, gpu in enumerate(args.gpus):
        env = environment.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = log_dir / f"worker_{worker_id:02d}_gpu{gpu}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command_for(model, args, worker_id),
            cwd=str(HERE.parent),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handles.append(handle)
        processes.append((worker_id, gpu, process, log_path))
        print(f"{model}: worker {worker_id} GPU {gpu}, pid={process.pid}", flush=True)
    try:
        while any(process.poll() is None for _, _, process, _ in processes):
            print(
                ", ".join(
                    f"w{worker_id}/gpu{gpu}="
                    f"{'running' if process.poll() is None else process.returncode}"
                    for worker_id, gpu, process, _ in processes
                ),
                flush=True,
            )
            time.sleep(20)
    finally:
        for handle in handles:
            handle.close()
    failures = [item for item in processes if item[2].returncode != 0]
    if failures:
        for worker_id, gpu, process, log_path in failures:
            print(
                f"FAILED {model} w{worker_id} GPU {gpu}: {process.returncode} {log_path}",
                file=sys.stderr,
            )
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        print(f"Starting {model}", flush=True)
        run_model(model, args)
        print(f"Completed {model}", flush=True)


if __name__ == "__main__":
    main()
