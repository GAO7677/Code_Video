#!/usr/bin/env python3
"""Report strict per-case progress for the ten-case six-seed ablation batch."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (
    build_tasks as build_fixed_tasks,
    task_root as fixed_task_root,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    build_tasks as build_tube_tasks,
    task_root as tube_task_root,
)


BATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
MANIFEST = BATCH_ROOT / "cases_other10_6seeds.json"
FIXED_ROOT = BATCH_ROOT / "attention_matrix_ablations_v2"
TUBE_ROOT = BATCH_ROOT / "attention_matrix_ablations_temporal_tube_v1"
TUBE_MODES = {
    "self_only",
    "incoming_only",
    "outgoing_only",
    "query_row",
    "key_value_column",
    "cross_boundary",
    "row_and_column",
    "literal_kv_zero",
}


def ready(root: Path) -> bool:
    return all((root / name).is_file() for name in ("complete.json", "manifest.json", "generated.mp4"))


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    fixed = [task for task in build_fixed_tasks(manifest) if int(task["top_n"]) == 100]
    tube = [
        task
        for sample in manifest["samples"]
        for task in build_tube_tasks(sample)
        if str(task["mask_mode"]) in TUBE_MODES
    ]
    progress: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "seeds": 0,
            "baselines_ready": 0,
            "fixed_ready": 0,
            "fixed_expected": 0,
            "tube_ready": 0,
            "tube_expected": 0,
            "errors": 0,
        }
    )
    for sample in manifest["samples"]:
        case = str(sample["case"])
        progress[case]["seeds"] += 1
        progress[case]["baselines_ready"] += Path(str(sample["baseline_video"])).is_file()
    for task in fixed:
        case = str(task["case"])
        root = fixed_task_root(task, FIXED_ROOT)
        progress[case]["fixed_expected"] += 1
        progress[case]["fixed_ready"] += ready(root)
        progress[case]["errors"] += (root / "error.txt").is_file()
    for task in tube:
        case = str(task["case"])
        root = tube_task_root(task, TUBE_ROOT)
        progress[case]["tube_expected"] += 1
        progress[case]["tube_ready"] += ready(root)
        progress[case]["errors"] += (root / "error.txt").is_file()

    print(
        f"{'case':92}  base   fixed       tube        errors"
    )
    totals = defaultdict(int)
    for case in sorted(progress):
        row = progress[case]
        print(
            f"{case:92}  {row['baselines_ready']:>2}/{row['seeds']:<2}  "
            f"{row['fixed_ready']:>4}/{row['fixed_expected']:<4}  "
            f"{row['tube_ready']:>4}/{row['tube_expected']:<4}  {row['errors']:>3}"
        )
        for key, value in row.items():
            totals[key] += value
    print(
        f"{'TOTAL':92}  {totals['baselines_ready']:>2}/{totals['seeds']:<2}  "
        f"{totals['fixed_ready']:>4}/{totals['fixed_expected']:<4}  "
        f"{totals['tube_ready']:>4}/{totals['tube_expected']:<4}  {totals['errors']:>3}"
    )


if __name__ == "__main__":
    main()
