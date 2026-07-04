#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume a partially completed Wan2.1 probe feature extraction by building a pending-only manifest "
            "and sharding the remaining samples across the requested GPUs."
        )
    )
    parser.add_argument("--manifest_csv", type=Path, required=True)
    parser.add_argument("--extract_root", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument(
        "--pending_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/tmp/pending_extract"),
    )
    parser.add_argument(
        "--model_root",
        type=Path,
        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers"),
    )
    parser.add_argument("--python_bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--visible_gpus", default="1,6,7")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--capture_steps", default="10,25,40")
    parser.add_argument("--capture_layers", default="2,8,14,20,29")
    parser.add_argument("--capture_branches", default="cond", choices=["cond", "both"])
    parser.add_argument("--label", default="resume_pending")
    return parser.parse_args()


def parse_csv_list(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def count_rows(csv_path: Path) -> int:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run_subprocess(cmd: list[Path | str]) -> None:
    pretty_cmd = " ".join(str(item) for item in cmd)
    print("[run]", pretty_cmd, flush=True)
    subprocess.run([str(item) for item in cmd], check=True)


def main() -> int:
    args = parse_args()
    visible_gpus = parse_csv_list(args.visible_gpus)
    if not visible_gpus:
        raise ValueError("visible_gpus must not be empty")

    args.pending_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    pending_csv = args.pending_root / f"{args.manifest_csv.stem}_pending.csv"
    build_pending_cmd = [
        args.python_bin,
        SCRIPT_ROOT / "build_pending_extract_manifest.py",
        "--manifest_csv",
        args.manifest_csv,
        "--extract_root",
        args.extract_root,
        "--output_csv",
        pending_csv,
        "--split_count",
        str(len(visible_gpus)),
    ]
    run_subprocess(build_pending_cmd)

    pending_rows = count_rows(pending_csv)
    if pending_rows == 0:
        print(json.dumps({"status": "nothing_to_do", "pending_csv": str(pending_csv)}, ensure_ascii=False), flush=True)
        return 0

    shard_csvs = [
        args.pending_root / f"{pending_csv.stem}_shard{idx}{pending_csv.suffix}"
        for idx in range(len(visible_gpus))
    ]

    running: list[tuple[subprocess.Popen[str], object, Path]] = []
    for shard_idx, (gpu_idx, shard_csv) in enumerate(zip(visible_gpus, shard_csvs)):
        shard_rows = count_rows(shard_csv)
        if shard_rows == 0:
            print(
                json.dumps(
                    {
                        "status": "skip_empty_shard",
                        "gpu_idx": gpu_idx,
                        "shard_csv": str(shard_csv),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        log_path = args.results_root / "logs" / f"{args.label}_shard{shard_idx}_gpu{gpu_idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            args.python_bin,
            SCRIPT_ROOT.parent / "probe_wan21" / "extract_probe_features.py",
            "--model_root",
            args.model_root,
            "--manifest_csv",
            shard_csv,
            "--output_root",
            args.extract_root,
            "--device",
            "cuda:0",
            "--dtype",
            args.dtype,
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--guidance_scale",
            str(args.guidance_scale),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--num_frames",
            str(args.num_frames),
            "--capture_steps",
            args.capture_steps,
            "--capture_layers",
            args.capture_layers,
            "--capture_branches",
            args.capture_branches,
            "--no_image_cond",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print(
            json.dumps(
                {
                    "status": "launch",
                    "gpu_idx": gpu_idx,
                    "rows": shard_rows,
                    "shard_csv": str(shard_csv),
                    "log_path": str(log_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        handle = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            [str(item) for item in cmd],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running.append((proc, handle, log_path))

    failed_logs: list[str] = []
    for proc, handle, log_path in running:
        try:
            if proc.wait() != 0:
                failed_logs.append(str(log_path))
        finally:
            handle.close()

    if failed_logs:
        print(json.dumps({"status": "failed", "logs": failed_logs}, ensure_ascii=False), flush=True)
        return 1

    print(
        json.dumps(
            {
                "status": "ok",
                "pending_csv": str(pending_csv),
                "remaining_rows_at_launch": pending_rows,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
