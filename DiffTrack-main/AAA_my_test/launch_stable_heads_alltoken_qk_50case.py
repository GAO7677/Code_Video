#!/usr/bin/env python3
"""Foreground, resumable 4-GPU launcher for stable-head all-token Q/K."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


WORKER = Path(__file__).with_name("capture_stable_heads_alltoken_qk_worker.py")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/three_model_stable_heads_alltoken_qk_50case"
)
PYTHON = "/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
STEPS = tuple(range(40))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "6", "7"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--models", nargs="+", choices=("gt", "lora", "baseline"),
        default=["gt", "lora", "baseline"],
    )
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument(
        "--combinations",
        default="20:9,20:17,26:7,18:11,24:6,19:0",
        help="Comma-separated block:head pairs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def environment(gpu: str, combinations: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ALLTOKEN_COMBINATIONS": combinations,
            "PYTHONPATH": ":".join(
                (
                    "/home/gaoya/Code_Video/DiffTrack-main",
                    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
                    "/home/gaoya/Code_Video/DiffSynth-Studio-main",
                    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419",
                )
            ),
        }
    )
    return env


def run_model(args: argparse.Namespace, model: str) -> None:
    layers = sorted(
        {int(item.split(":")[0]) for item in args.combinations.split(",")}
    )
    model_root = args.output_dir.resolve() / model
    log_dir = model_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for worker_id, gpu in enumerate(args.gpus):
        command = [
            PYTHON,
            str(WORKER),
            "--model-kind", model,
            "--worker-id", str(worker_id),
            "--num-workers", str(len(args.gpus)),
            "--output-dir", str(model_root),
            "--sampling-steps", "40",
            "--analysis-matching-mode", "q_to_k",
            "--analysis-layers", *map(str, layers),
            "--analysis-step-indices", *map(str, STEPS),
            "--analysis-no-hidden",
            "--analysis-no-video",
        ]
        if model != "gt":
            command.append("--analysis-no-cotracker")
        if args.case_keys:
            command.extend(("--case-keys", *args.case_keys))
        if args.overwrite:
            command.append("--overwrite")
        log_path = log_dir / f"worker_{worker_id:02d}_gpu{gpu}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd="/home/gaoya/Code_Video/DiffTrack-main",
            env=environment(str(gpu), args.combinations),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handles.append(handle)
        processes.append((worker_id, gpu, process, log_path))
        print(
            f"[{model}] worker {worker_id} GPU {gpu}: pid={process.pid} log={log_path}",
            flush=True,
        )
    try:
        while any(process.poll() is None for _, _, process, _ in processes):
            states = ", ".join(
                f"w{worker_id}/gpu{gpu}="
                f"{'running' if process.poll() is None else process.returncode}"
                for worker_id, gpu, process, _ in processes
            )
            print(f"[{model}] {states}", flush=True)
            time.sleep(30)
    except KeyboardInterrupt:
        for _, _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in handles:
            handle.close()
    failures = [item for item in processes if item[2].returncode != 0]
    if failures:
        for worker_id, gpu, process, log_path in failures:
            print(
                f"[{model}] failed w{worker_id}/gpu{gpu}: "
                f"code={process.returncode}, log={log_path}",
                file=sys.stderr,
            )
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    for model in args.models:
        run_model(args, model)
    print(f"all-token Q/K complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
    layers = sorted({int(item.split(":")[0]) for item in args.combinations.split(",")})
