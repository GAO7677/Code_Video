#!/usr/bin/env python3
"""Incrementally aggregate PhysicIQ67 legacy first-latent PCK@32 results."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import (
    CASES,
    OUTPUT_ROOT,
    all_tasks,
    read_seeds,
    run_dir,
)


AGGREGATE_DIR = OUTPUT_ROOT / "aggregate"


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def aggregate_once() -> int:
    tasks = all_tasks()
    correct = np.zeros((40, 30, 24), dtype=np.int64)
    comparisons = np.zeros_like(correct)
    error_sum = np.zeros((40, 30, 24), dtype=np.float64)
    completed = []
    per_case: dict[str, int] = {}
    for case, seed in tasks:
        output = run_dir(case.key, seed)
        if not (output / "complete.json").is_file() or not (output / "metrics.npz").is_file():
            continue
        with np.load(output / "metrics.npz") as arrays:
            correct += arrays["correct32"].astype(np.int64)
            comparisons += arrays["comparisons"].astype(np.int64)
            error_sum += arrays["error_sum"].astype(np.float64)
        completed.append({"case": case.key, "seed": int(seed)})
        per_case[case.key] = per_case.get(case.key, 0) + 1

    with np.errstate(divide="ignore", invalid="ignore"):
        pck = np.where(comparisons > 0, 100.0 * correct / comparisons, np.nan)
        mean_error = np.where(comparisons > 0, error_sum / comparisons, np.nan)
    ranking = [
        {
            "step": step,
            "block": block,
            "head": head,
            "pck32": None if not np.isfinite(pck[step, block, head]) else float(pck[step, block, head]),
            "mean_error_px": (
                None
                if not np.isfinite(mean_error[step, block, head])
                else float(mean_error[step, block, head])
            ),
            "comparisons": int(comparisons[step, block, head]),
        }
        for step in range(40)
        for block in range(30)
        for head in range(24)
    ]
    ranking.sort(
        key=lambda row: (
            row["pck32"] is None,
            -(row["pck32"] if row["pck32"] is not None else -1.0),
            row["mean_error_px"] if row["mean_error_px"] is not None else float("inf"),
        )
    )

    head_correct = correct.sum(axis=0)
    head_comparisons = comparisons.sum(axis=0)
    head_error = error_sum.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        head_pck = np.where(head_comparisons > 0, 100.0 * head_correct / head_comparisons, np.nan)
        head_mean_error = np.where(head_comparisons > 0, head_error / head_comparisons, np.nan)
    across_steps = [
        {
            "block": block,
            "head": head,
            "pck32": None if not np.isfinite(head_pck[block, head]) else float(head_pck[block, head]),
            "mean_error_px": (
                None
                if not np.isfinite(head_mean_error[block, head])
                else float(head_mean_error[block, head])
            ),
            "comparisons": int(head_comparisons[block, head]),
        }
        for block in range(30)
        for head in range(24)
    ]
    across_steps.sort(
        key=lambda row: (
            row["pck32"] is None,
            -(row["pck32"] if row["pck32"] is not None else -1.0),
            row["mean_error_px"] if row["mean_error_px"] is not None else float("inf"),
        )
    )

    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        AGGREGATE_DIR / "combined_counts.npz",
        correct32=correct,
        comparisons=comparisons,
        error_sum=error_sum,
        pck32=pck,
    )
    category_counts = Counter(case.category for case in CASES)
    protocol = {
        "case_count": len(CASES),
        "seed_count": len(read_seeds()),
        "expected_runs": len(tasks),
        "steps": 40,
        "blocks": 30,
        "heads": 24,
        "pck_threshold_px": 32,
        "query": "object regions only; 8 points per automatically grounded object",
        "query_pixel_frame": 0,
        "query_latent_frame": 0,
        "resolution": [704, 1280],
        "case_categories": dict(sorted(category_counts.items())),
    }
    atomic_json(
        AGGREGATE_DIR / "ranking.json",
        {
            "protocol": protocol,
            "ranking_unit": "step/block/head",
            "aggregation": "micro PCK@32 over visible object-query point/latent comparisons",
            "completed_runs": len(completed),
            "expected_runs": len(tasks),
            "global_step_block_head": ranking,
            "block_head_across_all_steps": across_steps,
        },
    )
    summary = {
        "protocol": protocol,
        "completed_runs": len(completed),
        "expected_runs": len(tasks),
        "per_case": per_case,
        "top_step_block_head": ranking[:10],
        "top_block_head_across_steps": across_steps[:10],
        "final": len(completed) == len(tasks),
    }
    atomic_json(AGGREGATE_DIR / "summary.json", summary)
    if summary["final"]:
        atomic_json(
            AGGREGATE_DIR / "final_top10.json",
            {"source": str(AGGREGATE_DIR / "ranking.json"), "entries": ranking[:10]},
        )
    print(f"aggregated {len(completed)}/{len(tasks)} runs", flush=True)
    return len(completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()
    expected = len(all_tasks())
    while True:
        completed = aggregate_once()
        if not args.watch or completed >= expected:
            return
        time.sleep(max(10, int(args.interval)))


if __name__ == "__main__":
    main()
