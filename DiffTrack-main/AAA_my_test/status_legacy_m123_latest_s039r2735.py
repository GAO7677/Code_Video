#!/usr/bin/env python3
"""Report exact completion for the ten-case latest S039 M1/M2/M3 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    build_tasks,
    task_root,
)


BATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
MANIFEST = BATCH_ROOT / "cases_other10_6seeds_latest.json"
OUTPUT_ROOT = BATCH_ROOT / "attention_matrix_ablations_temporal_tube_v1"
HEAD_SCOPES = ("top100", "bottom100", "all720")
RANKING_TAG = "s039r2735"
MASK_MODES = tuple(
    f"{operator}_{temporal}"
    for operator in ("self", "incoming", "outgoing")
    for temporal in ("only", "future", "same", "past")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="print READY EXPECTED ERRORS for shell coordinators",
    )
    return parser.parse_args()


def task_ready(root: Path) -> bool:
    return all(
        (root / name).is_file()
        for name in ("complete.json", "manifest.json", "generated.mp4")
    )


def counts() -> tuple[int, int, int]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected_modes = set(MASK_MODES)
    ready = 0
    expected = 0
    errors = 0
    for sample in manifest["samples"]:
        tasks = build_tasks(sample, HEAD_SCOPES, RANKING_TAG)
        for task in tasks:
            if str(task["mask_mode"]) not in selected_modes:
                continue
            root = task_root(task, OUTPUT_ROOT)
            expected += 1
            ready += int(task_ready(root))
            errors += int((root / "error.txt").is_file())
    return ready, expected, errors


def main() -> None:
    args = parse_args()
    ready, expected, errors = counts()
    if args.counts_only:
        print(ready, expected, errors)
        return
    percent = 100.0 * ready / expected if expected else 0.0
    print(
        f"latest S039 M1/M2/M3: {ready}/{expected} ({percent:.2f}%) "
        f"errors={errors}"
    )


if __name__ == "__main__":
    main()
