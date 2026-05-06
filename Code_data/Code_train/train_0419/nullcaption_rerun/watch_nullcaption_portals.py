#!/usr/bin/env python3
"""Rebuild null-caption portals whenever new completed sidecars appear."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
NULL_OUTPUT_DIR = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/output/VACE_1_3B_V2V/context_08f")
WAN_PYTHON = Path("/data/gaoya/miniconda3/envs/wan/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser.parse_args()


def count_completed(output_dir: Path) -> int:
    return len(list(output_dir.glob("*.json")))


def run_builders() -> None:
    commands = [
        [
            str(WAN_PYTHON),
            str(TRAIN0419_ROOT / "build_stage0_sidecar_portal.py"),
            "--benchmark_root",
            "/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption",
            "--output_root",
            "/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/output",
            "--portal_subdir",
            "tools/visualization/output_sidecar_portal",
        ],
        [
            str(WAN_PYTHON),
            str(TRAIN0419_ROOT / "nullcaption_rerun" / "build_caption_vs_nullcaption_portal.py"),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    last_count = -1
    while True:
        current_count = count_completed(NULL_OUTPUT_DIR)
        if current_count != last_count:
            print(f"[watch] completed_cases={current_count}", flush=True)
            run_builders()
            last_count = current_count
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
