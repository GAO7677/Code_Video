#!/usr/bin/env python3
"""Aggregate fixed Wan layer/step metrics across cases with case-level bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from AAA_my_test.wan_motion_utils import OUTPUT_ROOT, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=OUTPUT_ROOT / "batch_base")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / "aggregate_base")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def family(case_key: str) -> str:
    case_id = int(case_key.split("_")[1])
    return "F1" if case_id <= 15 else "F2" if case_id <= 35 else "F3"


def case_macro_rows(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row["motion_class"] != "moving_object":
            continue
        grouped[(row["case_key"], int(row["layer"]), int(row["step_index"]))].append(row)
    result = []
    metrics = ["pck32", "mean_error_px", "mean_direction_cosine", "rigidity_error_px"]
    for (case_key, layer, step), values in grouped.items():
        record = {
            "case_key": case_key,
            "family": family(case_key),
            "layer": layer,
            "step_index": step,
            "timestep": values[0]["timestep"],
            "sigma": values[0]["sigma"],
        }
        for metric in metrics:
            available = [float(row[metric]) for row in values if row.get(metric) is not None]
            record[metric] = float(np.mean(available)) if available else None
        result.append(record)
    return result


def bootstrap_ci(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    metric_paths = sorted(args.input_dir.glob("worker_*/*/metrics.json"))
    if not metric_paths:
        metric_paths = sorted(args.input_dir.glob("*/metrics.json"))
    all_rows = []
    for path in metric_paths:
        all_rows.extend(json.loads(path.read_text())["rows"])
    if not all_rows:
        raise FileNotFoundError(f"No batch metrics found under {args.input_dir}")

    macro = case_macro_rows(all_rows)
    grouped = defaultdict(list)
    for row in macro:
        grouped[(row["layer"], row["step_index"])].append(row)
    rng = np.random.default_rng(args.seed)
    summary = []
    for (layer, step), rows in grouped.items():
        pck = np.asarray([row["pck32"] for row in rows if row["pck32"] is not None])
        error = np.asarray([row["mean_error_px"] for row in rows if row["mean_error_px"] is not None])
        direction = np.asarray(
            [row["mean_direction_cosine"] for row in rows if row["mean_direction_cosine"] is not None]
        )
        low, high = bootstrap_ci(pck, args.bootstrap_repeats, rng)
        record = {
            "layer": layer,
            "step_index": step,
            "timestep": rows[0]["timestep"],
            "sigma": rows[0]["sigma"],
            "case_count": len(rows),
            "macro_pck32": float(pck.mean()),
            "pck32_ci95_low": low,
            "pck32_ci95_high": high,
            "macro_mean_error_px": float(error.mean()),
            "macro_direction_cosine": float(direction.mean()) if direction.size else None,
        }
        for family_name in ("F1", "F2", "F3"):
            family_pck = [row["pck32"] for row in rows if row["family"] == family_name and row["pck32"] is not None]
            record[f"{family_name.lower()}_pck32"] = float(np.mean(family_pck)) if family_pck else None
        summary.append(record)
    summary.sort(key=lambda row: (row["macro_pck32"], -row["macro_mean_error_px"]), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    atomic_write_json(args.output_dir / "case_macro_metrics.json", macro)
    atomic_write_json(
        args.output_dir / "best_configs.json",
        {
            "completed_sample_count": len({row["case_key"] for row in all_rows}),
            "motion_case_count": len({row["case_key"] for row in macro}),
            "ranking_rule": "case-macro moving-object PCK@32, then lower mean error",
            "best": summary[:3],
        },
    )
    print(json.dumps(summary[:10], indent=2))


if __name__ == "__main__":
    main()
