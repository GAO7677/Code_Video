#!/usr/bin/env python3
"""Launch analyze_real_toy.py across multiple GPUs for the 0718 toy case set."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("/home/gaoya/Code_Video/DiffTrack-main"))
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset"),
    )
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/difftrack_0718toy_case50_sam2_regions/tracks"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/difftrack_0718toy_case50_sam2_regions/"
            "cogvideox_2b_steps_0_10_20_29_39"
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/gaoya/agent-data/weights/CogVideoX-2b-modelscope"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/cache/huggingface"),
    )
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--model", choices=["cogvideox_t2v_2b", "cogvideox_t2v_5b"], default="cogvideox_t2v_2b")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--inverse-steps", nargs="+", type=int, default=[0, 10, 20, 29, 39])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_sample_count(track_dir: Path) -> int:
    manifest_path = track_dir / "tracks_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return len(payload["samples"])


def main() -> int:
    args = parse_args()
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise ValueError("No GPUs provided")

    sample_count = load_sample_count(args.track_dir)
    chunk = math.ceil(sample_count / len(gpu_ids))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(args.cache_dir)
    env["PYTHONPATH"] = f"{args.repo_root / 'diffusers' / 'src'}:{args.repo_root}"

    procs: list[tuple[str, int, int, subprocess.Popen[str], Path]] = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        start = worker_id * chunk
        end = min(sample_count, start + chunk)
        if start >= end:
            continue
        log_path = log_dir / f"worker_gpu{gpu_id}_{start}_{end}.log"
        cmd = [
            str(args.python_bin),
            "AAA_my_test/analyze_real_toy.py",
            "--dataset-root",
            str(args.dataset_root),
            "--track-dir",
            str(args.track_dir),
            "--output-dir",
            str(args.output_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--model-path",
            str(args.model_path),
            "--model",
            args.model,
            "--device",
            f"cuda:{gpu_id}",
            "--num-inference-steps",
            str(args.num_inference_steps),
            "--inverse-steps",
            *[str(value) for value in args.inverse_steps],
            "--start",
            str(start),
            "--end",
            str(end),
            "--matching-accuracy",
            "--conf-attn-score",
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=args.repo_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((gpu_id, start, end, proc, log_path))
        print(f"Started gpu={gpu_id} range=[{start}, {end}) pid={proc.pid} log={log_path}", flush=True)

    failed = False
    for gpu_id, start, end, proc, log_path in procs:
        code = proc.wait()
        if code == 0:
            print(f"Completed gpu={gpu_id} range=[{start}, {end})", flush=True)
        else:
            print(
                f"Failed gpu={gpu_id} range=[{start}, {end}) exit={code} log={log_path}",
                file=sys.stderr,
                flush=True,
            )
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
