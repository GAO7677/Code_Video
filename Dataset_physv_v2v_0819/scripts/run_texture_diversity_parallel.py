#!/usr/bin/env python3
"""Launch the high-diversity demo jobs across explicitly selected GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-list", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--exposure", type=float, default=-0.15)
    args = parser.parse_args()

    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus:
        raise ValueError("gpu-list must contain at least one device")
    if any(item == "4" for item in gpus):
        raise ValueError("GPU 4 is forbidden by the workspace rules")
    if args.workers_per_gpu <= 0:
        raise ValueError("workers-per-gpu must be positive")

    output_root = args.output_root.resolve()
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    generator = Path(__file__).with_name("generate_texture_diversity_demo.py").resolve()
    worker_count = len(gpus) * args.workers_per_gpu
    processes: list[tuple[int, str, subprocess.Popen, object]] = []
    for worker_index in range(worker_count):
        gpu = gpus[worker_index % len(gpus)]
        log_path = output_root / "logs" / f"supervisor_worker_{worker_index:02d}_gpu{gpu}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        command = [
            sys.executable,
            str(generator),
            "--source-root",
            str(args.source_root.resolve()),
            "--output-root",
            str(output_root),
            "--seed",
            str(args.seed),
            "--samples",
            str(args.samples),
            "--exposure",
            str(args.exposure),
            "--gpu",
            gpu,
            "--worker-index",
            str(worker_index),
            "--worker-count",
            str(worker_count),
        ]
        print(f"launch worker={worker_index}/{worker_count} gpu={gpu} log={log_path}", flush=True)
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((worker_index, gpu, process, log_handle))

    failed = False
    for worker_index, gpu, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        print(f"worker_done worker={worker_index} gpu={gpu} returncode={return_code}", flush=True)
        failed = failed or return_code != 0

    finalize_command = [
        sys.executable,
        str(generator),
        "--source-root",
        str(args.source_root.resolve()),
        "--output-root",
        str(output_root),
        "--finalize",
    ]
    subprocess.run(finalize_command, check=True)
    if failed:
        raise SystemExit("one or more diversity workers returned a non-zero status")


if __name__ == "__main__":
    main()
