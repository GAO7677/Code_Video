#!/usr/bin/env python3
"""Wait for S-depth preflight, then repeatedly drain the resumable task queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
WORKER = SCRIPT_DIR / "run_head_role_dose_control_pilot_worker.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--worker-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    complete = root / "generation.complete"
    failed = root / "generation.failed"
    preflight_complete = Path(
        config["execution"]["preflight_complete_file"]
    ).expanduser().resolve()
    preflight_failed = preflight_complete.with_name("preflight.failed")
    poll_seconds = int(config["execution"]["poll_seconds"])

    while not preflight_complete.is_file():
        if preflight_failed.is_file():
            raise RuntimeError("S-depth preflight failed; formal worker aborted")
        print(
            f"[s-depth-worker] waiting for preflight: {preflight_complete}",
            flush=True,
        )
        time.sleep(poll_seconds)

    while not complete.is_file():
        if failed.is_file():
            raise RuntimeError("S-depth generation failed")
        subprocess.run(
            [
                str(PYTHON),
                str(WORKER),
                "--config",
                str(config_path),
                "--gpu",
                str(args.gpu),
                "--worker-id",
                args.worker_id,
            ],
            check=True,
        )
        if not complete.is_file() and not failed.is_file():
            time.sleep(min(poll_seconds, 10))


if __name__ == "__main__":
    main()
