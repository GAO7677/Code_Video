#!/usr/bin/env python3
"""Foreground multi-GPU launcher for the raw-physics LoRA analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


WORKER = Path(__file__).with_name("run_lorav2v_toy_analysis_worker.py")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_openvid_0613pybullet_lora_step000500_sam2_regions_steps40"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--headwise", action="store_true")
    parser.add_argument("--analysis-layer", type=int, default=5)
    parser.add_argument("--analysis-step", type=int, default=39)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for worker_id, gpu in enumerate(args.gpus):
        command = [
            "/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
            str(WORKER),
            "--worker-id", str(worker_id),
            "--num-workers", str(len(args.gpus)),
            "--output-dir", str(output_dir),
            "--sampling-steps", str(args.sampling_steps),
        ]
        if args.case_keys:
            command.extend(["--case-keys", *args.case_keys])
        if args.overwrite:
            command.append("--overwrite")
        if args.headwise:
            command.extend([
                "--analysis-matching-mode", "headwise",
                "--analysis-layers", str(args.analysis_layer),
                "--analysis-step-indices", str(args.analysis_step),
                "--analysis-no-hidden", "--analysis-no-video",
            ])
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        environment["PYTHONPATH"] = ":".join(
            [str(CODE_ROOT) for CODE_ROOT in (
                Path("/home/gaoya/Code_Video/DiffTrack-main"),
                Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt"),
                Path("/home/gaoya/Code_Video/DiffSynth-Studio-main"),
                Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419"),
            )]
        )
        log_path = log_dir / f"worker_{worker_id:02d}_gpu{gpu}.log"
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd="/home/gaoya/Code_Video/DiffTrack-main",
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        handles.append(handle)
        processes.append((worker_id, gpu, process, log_path))
        print(f"started worker {worker_id} on GPU {gpu}: pid={process.pid} log={log_path}", flush=True)
    try:
        while True:
            states = [process.poll() for _, _, process, _ in processes]
            if all(state is not None for state in states):
                break
            print(
                ", ".join(
                    f"w{worker_id}/gpu{gpu}={'running' if state is None else state}"
                    for (worker_id, gpu, _, _), state in zip(processes, states)
                ),
                flush=True,
            )
            time.sleep(20)
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
            print(f"failed worker {worker_id} gpu{gpu} code={process.returncode} log={log_path}", file=sys.stderr)
        raise SystemExit(1)
    print(f"all workers completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
