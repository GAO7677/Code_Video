#!/usr/bin/env python3
"""Validate and aggregate 50-case, all-step, all-block, per-head Q@K results."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case")
MODELS = {
    "gt": "GT teacher-forced",
    "lora": "LoRA step-000500",
    "baseline": "Wan2.2 baseline",
}
LAYERS = tuple(range(30))
STEPS = tuple(range(40))
HEADS = tuple(range(24))
EXPECTED_COMBINATIONS = len(LAYERS) * len(STEPS) * len(HEADS)
TRACK_KEY = re.compile(r"^qk_head(\d{2})_layer(\d{2})_step(\d{3})_predictions$")
METRICS = ("pck8", "pck16", "pck32", "mean_error_px")


def empty_accumulator() -> dict:
    return {
        "cases": 0,
        "comparisons": 0,
        "macro_pck8": 0.0,
        "macro_pck16": 0.0,
        "macro_pck32": 0.0,
        "macro_mean_error_px": 0.0,
        "pooled_pck8": 0.0,
        "pooled_pck16": 0.0,
        "pooled_pck32": 0.0,
        "pooled_mean_error_px": 0.0,
        "timestep": None,
        "sigma": None,
    }


def weighted_case_rows(rows: list[dict]) -> dict[tuple[int, int, int, str], dict]:
    grouped: dict[tuple[int, int, int, str], dict] = defaultdict(
        lambda: {"comparisons": 0, **{metric: 0.0 for metric in METRICS}, "timestep": None, "sigma": None}
    )
    for row in rows:
        method = str(row.get("method", ""))
        if not method.startswith("qk_head"):
            continue
        region_type = row.get("region_type")
        if region_type == "object":
            scope = "objects"
        elif region_type == "background":
            scope = "background"
        else:
            continue
        comparisons = int(row.get("comparisons", 0))
        if comparisons <= 0:
            continue
        head = int(method.removeprefix("qk_head"))
        key = (int(row["layer"]), int(row["step_index"]), head, scope)
        target = grouped[key]
        target["comparisons"] += comparisons
        for metric in METRICS:
            target[metric] += float(row[metric]) * comparisons
        target["timestep"] = float(row["timestep"])
        target["sigma"] = float(row["sigma"])
    for target in grouped.values():
        comparisons = target["comparisons"]
        for metric in METRICS:
            target[metric] /= comparisons
    return grouped


def aggregate() -> tuple[list[dict], dict]:
    totals: dict[tuple[str, int, int, int, str], dict] = defaultdict(empty_accumulator)
    validation = {
        "expected": {
            "models": 3,
            "cases_per_model": 50,
            "steps": 40,
            "blocks": 30,
            "heads": 24,
            "combinations_per_case": EXPECTED_COMBINATIONS,
            "total_case_step_block_head_combinations": 3 * 50 * EXPECTED_COMBINATIONS,
        },
        "models": {},
        "errors": [],
    }

    for model in MODELS:
        case_dirs = sorted(path for path in (ROOT / model / "cases").iterdir() if path.is_dir())
        model_validation = {"cases": len(case_dirs), "complete_cases": 0, "validated_cases": 0}
        validation["models"][model] = model_validation
        if len(case_dirs) != 50:
            validation["errors"].append(f"{model}: expected 50 case directories, found {len(case_dirs)}")

        for case_index, case_dir in enumerate(case_dirs, start=1):
            if (case_dir / "complete.json").is_file():
                model_validation["complete_cases"] += 1
            else:
                validation["errors"].append(f"{model}/{case_dir.name}: missing complete.json")

            with np.load(case_dir / "predicted_tracks.npz", allow_pickle=False) as tracks:
                combinations = set()
                for name in tracks.files:
                    match = TRACK_KEY.match(name)
                    if match:
                        head, layer, step = map(int, match.groups())
                        combinations.add((layer, step, head))
            expected = {(layer, step, head) for layer in LAYERS for step in STEPS for head in HEADS}
            missing = expected - combinations
            extra = combinations - expected
            if missing or extra:
                validation["errors"].append(
                    f"{model}/{case_dir.name}: tracks={len(combinations)}, missing={len(missing)}, extra={len(extra)}"
                )
            else:
                model_validation["validated_cases"] += 1

            rows = json.loads((case_dir / "metrics.json").read_text())
            case_metrics = weighted_case_rows(rows)
            metric_combinations = {(layer, step, head) for layer, step, head, _ in case_metrics}
            missing_metrics = expected - metric_combinations
            if missing_metrics:
                validation["errors"].append(
                    f"{model}/{case_dir.name}: missing {len(missing_metrics)} metric combinations"
                )

            for (layer, step, head, scope), case_row in case_metrics.items():
                target = totals[(model, layer, step, head, scope)]
                target["cases"] += 1
                comparisons = case_row["comparisons"]
                target["comparisons"] += comparisons
                for metric in METRICS:
                    target[f"macro_{metric}"] += case_row[metric]
                    target[f"pooled_{metric}"] += case_row[metric] * comparisons
                target["timestep"] = case_row["timestep"]
                target["sigma"] = case_row["sigma"]
            print(f"{model} [{case_index:02d}/{len(case_dirs):02d}] {case_dir.name}", flush=True)

    output = []
    for (model, layer, step, head, scope), values in sorted(totals.items()):
        cases = values["cases"]
        comparisons = values["comparisons"]
        row = {
            "model": model,
            "model_label": MODELS[model],
            "scope": scope,
            "layer": layer,
            "step": step,
            "head": head,
            "timestep": values["timestep"],
            "sigma": values["sigma"],
            "cases": cases,
            "comparisons": comparisons,
        }
        for metric in METRICS:
            row[f"macro_{metric}"] = values[f"macro_{metric}"] / cases if cases else math.nan
            row[f"pooled_{metric}"] = values[f"pooled_{metric}"] / comparisons if comparisons else math.nan
        output.append(row)
    validation["valid"] = not validation["errors"]
    return output, validation


def write_outputs(rows: list[dict], validation: dict) -> None:
    summary_path = ROOT / "block_step_head_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_per_grid = {}
    for row in rows:
        key = (row["model"], row["scope"], row["layer"], row["step"])
        previous = best_per_grid.get(key)
        if previous is None or (row["macro_pck32"], -row["macro_mean_error_px"]) > (
            previous["macro_pck32"], -previous["macro_mean_error_px"]
        ):
            best_per_grid[key] = row
    best_path = ROOT / "best_head_per_block_step.csv"
    best_rows = [best_per_grid[key] for key in sorted(best_per_grid)]
    with best_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best_rows[0]))
        writer.writeheader()
        writer.writerows(best_rows)

    top = {}
    for model in MODELS:
        candidates = [row for row in rows if row["model"] == model and row["scope"] == "objects"]
        candidates.sort(key=lambda row: (-row["macro_pck32"], row["macro_mean_error_px"]))
        top[model] = candidates[:20]
    (ROOT / "top_combinations.json").write_text(json.dumps(top, indent=2))
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2))

    lines = [
        "# 50-case all-step, all-block, per-head Q@K validation",
        "",
        "No averaging across heads. Every S000-S039 / L00-L29 / H00-H23 trajectory is preserved.",
        "",
        f"Validation: **{'PASS' if validation['valid'] else 'FAIL'}**",
        "",
        "| model | rank | step | block | head | valid object cases | macro PCK@32 | pooled PCK@32 | macro error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, candidates in top.items():
        for rank, row in enumerate(candidates[:10], start=1):
            lines.append(
                f"| {MODELS[model]} | {rank} | S{row['step']:03d} | L{row['layer']:02d} | "
                f"H{row['head']:02d} | {row['cases']} | {row['macro_pck32']:.2f}% | "
                f"{row['pooled_pck32']:.2f}% | {row['macro_mean_error_px']:.2f}px |"
            )
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    rows, validation = aggregate()
    write_outputs(rows, validation)
    print(f"rows={len(rows)} validation={'PASS' if validation['valid'] else 'FAIL'}")
    print(ROOT / "block_step_head_summary.csv")


if __name__ == "__main__":
    main()
