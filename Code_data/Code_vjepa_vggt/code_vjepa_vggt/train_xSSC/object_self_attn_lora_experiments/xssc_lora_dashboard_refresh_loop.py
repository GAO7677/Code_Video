#!/usr/bin/env python3
"""Periodically rebuild the xSSC LoRA dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = Path(__file__).with_name("build_xssc_lora_checkpoint_dashboard.py")
    command = [
        "/home/gaoya/miniconda3/envs/wan-cu128/bin/python",
        str(builder),
        "--config",
        str(args.config.expanduser().resolve()),
    ]
    while True:
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"[dashboard-refresh] start {started}", flush=True)
        subprocess.run(command, check=True)
        finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"[dashboard-refresh] complete {finished}", flush=True)
        if args.once:
            return
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
