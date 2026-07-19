#!/usr/bin/env python3
"""Launch GroundingDINO + SAM2 region preprocessing across GPUs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4"])
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    log_dir = cache_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    handles = []
    for worker_id, gpu in enumerate(args.gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = ":".join(
            (
                "/home/gaoya/Code_Video/DiffTrack-main",
                "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt",
                "/home/gaoya/Grounded-SAM-2-main",
            )
        )
        log_path = log_dir / f"worker_{worker_id:02d}_gpu{gpu}.log"
        handle = log_path.open("w", encoding="utf-8")
        command = [
            str(PYTHON), str(HERE / "precompute_toydataset_sam2_regions.py"),
            "--dataset-root", str(args.dataset_root.resolve()),
            "--cache-root", str(cache_root),
            "--worker-id", str(worker_id),
            "--num-workers", str(len(args.gpus)),
            "--device", "cuda:0",
        ]
        process = subprocess.Popen(
            command, cwd=str(HERE.parent), env=env, stdout=handle, stderr=subprocess.STDOUT
        )
        handles.append(handle)
        processes.append((worker_id, gpu, process, log_path))
        print(f"grounded SAM2 worker {worker_id} GPU {gpu}, pid={process.pid}", flush=True)
    for worker_id, gpu, process, log_path in processes:
        return_code = process.wait()
        print(f"grounded SAM2 worker {worker_id} GPU {gpu} exit={return_code}: {log_path}", flush=True)
    for handle in handles:
        handle.close()
    completed = sum(1 for path in cache_root.glob("case_*/complete.json"))
    if completed == 0:
        raise SystemExit("no grounded SAM2 case completed")
    print(f"grounded SAM2 completed caches: {completed}", flush=True)


if __name__ == "__main__":
    main()
