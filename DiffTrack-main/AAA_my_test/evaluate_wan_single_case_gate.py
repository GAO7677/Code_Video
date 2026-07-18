#!/usr/bin/env python3
"""Evaluate the Wan single-case scan gate before batch execution."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from AAA_my_test.wan_motion_utils import OUTPUT_ROOT, atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=OUTPUT_ROOT / "single_case" / "case_019_wheel_hits_block_base",
    )
    parser.add_argument("--expected-layers", type=int, nargs="+", default=[0, 5, 11, 17, 23, 29])
    parser.add_argument("--expected-steps", type=int, nargs="+", default=[0, 12, 24, 36, 49])
    return parser.parse_args()


def weighted_mean(rows: list[dict], key: str) -> float:
    valid = [row for row in rows if row.get(key) is not None]
    denominator = sum(row["comparisons"] for row in valid)
    if not denominator:
        return float("nan")
    return sum(row[key] * row["comparisons"] for row in valid) / denominator


def main() -> None:
    args = parse_args()
    metrics_path = args.result_dir / "metrics.json"
    complete_path = args.result_dir / "complete.json"
    if not metrics_path.exists() or not complete_path.exists():
        raise FileNotFoundError("Single-case scan is incomplete: metrics.json or complete.json is missing")
    rows = json.loads(metrics_path.read_text())["rows"]
    expected = {(layer, step) for layer in args.expected_layers for step in args.expected_steps}
    actual = {(int(row["layer"]), int(row["step_index"])) for row in rows}
    missing = sorted(expected - actual)

    non_finite = []
    required_numeric = [
        "mean_error_px",
        "median_error_px",
        "pck32",
        "static_mean_error_px",
        "static_pck32",
        "predicted_mean_displacement_px",
        "target_mean_displacement_px",
    ]
    for row in rows:
        for key in required_numeric:
            if row.get(key) is None or not math.isfinite(float(row[key])):
                non_finite.append(
                    {"layer": row["layer"], "step": row["step_index"], "region": row["region_name"], "metric": key}
                )

    grouped = defaultdict(list)
    for row in rows:
        if row["motion_class"] == "moving_object":
            grouped[(int(row["layer"]), int(row["step_index"]))].append(row)

    ranking = []
    passing_candidates = []
    for layer, step in sorted(expected):
        motion_rows = grouped.get((layer, step), [])
        if not motion_rows:
            continue
        pck32 = weighted_mean(motion_rows, "pck32")
        static_pck32 = weighted_mean(motion_rows, "static_pck32")
        mean_error = weighted_mean(motion_rows, "mean_error_px")
        static_error = weighted_mean(motion_rows, "static_mean_error_px")
        predicted_displacement = weighted_mean(motion_rows, "predicted_mean_displacement_px")
        target_displacement = weighted_mean(motion_rows, "target_mean_displacement_px")
        pck_gain = pck32 - static_pck32
        error_reduction = (static_error - mean_error) / static_error if static_error > 0 else 0.0
        motion_ratio = predicted_displacement / target_displacement if target_displacement > 0 else 0.0
        candidate = {
            "layer": layer,
            "step_index": step,
            "timestep": motion_rows[0]["timestep"],
            "sigma": motion_rows[0]["sigma"],
            "moving_pck32": pck32,
            "static_pck32": static_pck32,
            "pck32_gain": pck_gain,
            "moving_mean_error_px": mean_error,
            "static_mean_error_px": static_error,
            "relative_error_reduction": error_reduction,
            "predicted_to_target_motion_ratio": motion_ratio,
        }
        candidate["passes_motion_gate"] = (
            5 <= layer <= 23
            and (pck_gain >= 5.0 or error_reduction >= 0.10)
            and motion_ratio >= 0.10
        )
        ranking.append(candidate)
        if candidate["passes_motion_gate"]:
            passing_candidates.append(candidate)

    ranking.sort(key=lambda row: (row["moving_pck32"], -row["moving_mean_error_px"]), reverse=True)
    report = {
        "passed": not missing and not non_finite and bool(passing_candidates),
        "result_dir": str(args.result_dir),
        "expected_combinations": len(expected),
        "observed_combinations": len(actual & expected),
        "missing_combinations": missing,
        "non_finite_metrics": non_finite,
        "passing_candidates": passing_candidates,
        "best_candidate": ranking[0] if ranking else None,
        "gate_definition": {
            "middle_layers": [5, 23],
            "minimum_pck32_gain_points": 5.0,
            "minimum_relative_error_reduction": 0.10,
            "minimum_predicted_to_target_motion_ratio": 0.10,
        },
    }
    atomic_write_json(args.result_dir / "gate_report.json", report)
    with (args.result_dir / "layer_step_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranking[0]) if ranking else ["layer", "step_index"])
        writer.writeheader()
        writer.writerows(ranking)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
