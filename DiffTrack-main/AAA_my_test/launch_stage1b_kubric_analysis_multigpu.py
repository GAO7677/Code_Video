#!/usr/bin/env python3
"""Launch four foreground ToyDataset analysis workers on selected GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


WORKER = Path(__file__).with_name("run_stage1b_kubric_analysis_worker.py")
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/stage1b_kubric_generation_analysis_step004000"
)
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_kubric0708/checkpoints/step-004000"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_count = len(args.gpus)
    processes = []
    log_handles = []
    for worker_id, gpu in enumerate(args.gpus):
        command = [
            "/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
            str(WORKER),
            "--checkpoint",
            str(args.checkpoint),
            "--output-dir",
            str(output_dir),
            "--worker-id",
            str(worker_id),
            "--num-workers",
            str(worker_count),
            "--sampling-steps",
            str(args.sampling_steps),
            "--num-frames",
            str(args.num_frames),
            "--context-frames",
            "8",
            "--analysis-device",
            "cuda:0",
        ]
        if args.case_keys:
            command.extend(["--case-keys", *args.case_keys])
        if args.overwrite:
            command.append("--overwrite")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        environment["PYTHONPATH"] = ":".join(
            [
                "/home/gaoya/Code_Video/DiffTrack-main",
                "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
                "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
            ]
        )
        log_path = log_dir / f"worker_{worker_id:02d}_gpu{gpu}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=environment,
            cwd="/home/gaoya/Code_Video/DiffTrack-main",
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, gpu, process, log_path))
        log_handles.append(log_handle)
        print(f"started worker {worker_id} on GPU {gpu}: pid={process.pid} log={log_path}", flush=True)

    try:
        while True:
            states = [process.poll() for _, _, process, _ in processes]
            if all(state is not None for state in states):
                break
            status = ", ".join(
                f"w{worker_id}/gpu{gpu}={'running' if state is None else state}"
                for (worker_id, gpu, _, _), state in zip(processes, states)
            )
            print(status, flush=True)
            time.sleep(20)
    except KeyboardInterrupt:
        for _, _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for handle in log_handles:
            handle.close()

    failures = [
        (worker_id, gpu, process.returncode, log_path)
        for worker_id, gpu, process, log_path in processes
        if process.returncode != 0
    ]
    if failures:
        for failure in failures:
            print(f"failed worker: {failure}", file=sys.stderr)
        raise SystemExit(1)
    print(f"all {worker_count} workers completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
