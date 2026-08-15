#!/usr/bin/env python3
"""Run the expanded 21-entry validation matrix on physical GPU 7."""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / "validation_30cases_config.json"
PYTHON = "/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ROOT = Path("/data/gaoya/agent-data/outputs/xssc_train_validation_30cases")
PROJECT_ROOT = "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt"
DIFFSYNTH_ROOT = "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"


def environment(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONPATH=f"{PROJECT_ROOT}:{DIFFSYNTH_ROOT}",
        CUDA_VISIBLE_DEVICES=str(gpu),
    )
    return env


def run(command: list[str], log_name: str, gpu: int) -> None:
    log_root = ROOT / "pipeline_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / log_name
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment(gpu),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=7)
    args = parser.parse_args()
    if args.gpu == 4:
        raise SystemExit("GPU4 prohibited")
    log_name = f"runner-gpu{args.gpu}-expanded.log"
    run(
        [
            PYTHON,
            "-u",
            str(PROJECT / "run_validation_30cases.py"),
            "--config",
            str(CONFIG),
            "--gpu",
            str(args.gpu),
        ],
        log_name,
        args.gpu,
    )
    run(
        [
            PYTHON,
            "-u",
            str(PROJECT / "run_validation_30_loss.py"),
            "--config",
            str(CONFIG),
            "--gpu",
            str(args.gpu),
        ],
        log_name,
        args.gpu,
    )
    run(
        [
            PYTHON,
            str(PROJECT / "build_validation_30cases_hub.py"),
            "--config",
            str(CONFIG),
        ],
        log_name,
        args.gpu,
    )


if __name__ == "__main__":
    main()
