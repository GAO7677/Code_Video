#!/usr/bin/env python3
"""Keep Scene-Enabled case and aggregate metric pages in sync.

The metric workers write result JSONs and completion markers independently of
the static 8844 pages.  Refresh the lightweight case/average pages on a short
poll interval so a normal browser refresh exposes every value already written,
without rerunning inference or benchmark metrics.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


WATCH_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch")
METHOD = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled"
REFRESH = Path(__file__).with_name("refresh_utonia_enabled_gallery_records.py")
REFRESH_METRIC_PAGES = Path(__file__).with_name(
    "refresh_incremental_metric_pages.py"
)
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)
METRICS_PER_STEP = 14
POLL_SECONDS = 60


def marker_count(dataset: str) -> int:
    if dataset == "test5":
        root = WATCH_ROOT / "state" / "metrics" / METHOD
    else:
        root = WATCH_ROOT / "state" / "physiciq" / "metrics" / METHOD
    return sum(
        len(list((root / f"step-{step:06d}").glob("*.json")))
        for step in STEPS
    )


def refresh() -> None:
    subprocess.run([str(PYTHON), str(REFRESH)], check=True)
    subprocess.run([str(PYTHON), str(REFRESH_METRIC_PAGES)], check=True)


def main() -> None:
    expected = len(STEPS) * METRICS_PER_STEP
    while True:
        refresh()
        test5_done = marker_count("test5")
        physiciq_done = marker_count("physiciq")
        print(
            f"scene-enabled page refresh: test5={test5_done}/{expected} "
            f"physiciq={physiciq_done}/{expected}",
            flush=True,
        )
        if test5_done >= expected and physiciq_done >= expected:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
