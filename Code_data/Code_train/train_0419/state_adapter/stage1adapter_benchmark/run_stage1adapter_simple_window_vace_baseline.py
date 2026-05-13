#!/usr/bin/env python3
"""Run native VACE V2V baseline on stage1adapter simple-window benchmark metas."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/stage1adapter_simple_window")
DEFAULT_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage1adapter/Genesis_simple_window")
DEFAULT_VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")
META_BUILDER = SCRIPT_DIR / "build_stage1adapter_simple_window_benchmark_meta.py"
BATCH_EVAL_VACE = TRAIN0419_ROOT / "batch_eval_vace.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VACE V2V baseline on stage1adapter_simple_window.")
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--benchmark_root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--vace_root", type=Path, default=DEFAULT_VACE_ROOT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_build_meta", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print(" ".join(cmd))
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    meta_root = args.benchmark_root / "tools" / "meta"
    split_tag = f"stage1adapter_simple_window_{args.split}"
    meta_list_path = meta_root / split_tag / f"benchmark_meta_json_paths_{args.split}.txt"

    if not args.skip_build_meta:
        run_cmd(
            [
                sys.executable,
                str(META_BUILDER),
                "--summary_root",
                str(args.summary_root),
                "--split",
                args.split,
                "--output_root",
                str(meta_root),
                *(["--overwrite"] if args.overwrite else []),
            ]
        )

    output_root = args.benchmark_root / "output" / "VACE_1_3B_V2V" / "variable_ctx"
    runtime_root = args.benchmark_root / "tools" / "runtime" / "vace_v2v_variable_ctx"

    run_cmd(
        [
            sys.executable,
            str(BATCH_EVAL_VACE),
            "--vace_root",
            str(args.vace_root),
            "--meta_list_path",
            str(meta_list_path),
            "--output_root",
            str(output_root),
            "--runtime_root",
            str(runtime_root),
            "--model_name",
            "vace_v2v_variable_ctx",
            "--mode",
            "v2v_clipref",
            "--device",
            args.device,
            "--height",
            "544",
            "--width",
            "720",
            "--fps",
            str(args.fps),
            "--num_frames",
            "13",
            "--context_frames",
            "4",
            "--num_inference_steps",
            str(args.num_inference_steps),
            "--cfg_scale",
            str(args.cfg_scale),
            "--seed",
            str(args.seed),
            "--quality",
            str(args.quality),
            *(["--overwrite"] if args.overwrite else []),
        ]
    )


if __name__ == "__main__":
    main()
