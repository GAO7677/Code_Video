#!/usr/bin/env python3
"""Keep GT sidecars current while the independent metric workers run."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
SCRIPT = Path(__file__).with_name("gt_metrics_8844.py")


def run_is_active() -> bool:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "args"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    needle = "gt_metrics_8844.py run"
    return any(needle in line for line in output.splitlines())


def main() -> None:
    while True:
        subprocess.run(
            [str(PYTHON), str(SCRIPT), "summarize"],
            check=False,
        )
        if not run_is_active():
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
