#!/usr/bin/env python3
"""Build equal-model-weight rankings across GT, LoRA, and Wan2.2 Baseline."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case")
SOURCE = ROOT / "block_step_head_summary.csv"
OUTPUT = ROOT / "three_model_combined_summary.csv"
REPORT = ROOT / "THREE_MODEL_COMBINED_RESULTS.md"
MODELS = ("gt", "lora", "baseline")
METRICS = (
    "macro_pck8", "macro_pck16", "macro_pck32", "macro_mean_error_px",
    "pooled_pck8", "pooled_pck16", "pooled_pck32", "pooled_mean_error_px",
)


def main() -> None:
    grouped: dict[tuple[str, int, int, int], dict[str, dict]] = defaultdict(dict)
    with SOURCE.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["scope"], int(row["step"]), int(row["layer"]), int(row["head"]))
            grouped[key][row["model"]] = row

    combined = []
    for (scope, step, block, head), model_rows in sorted(grouped.items()):
        missing = set(MODELS) - set(model_rows)
        if missing:
            raise RuntimeError(f"missing models for {scope}/S{step}/L{block}/H{head}: {sorted(missing)}")
        rows = [model_rows[model] for model in MODELS]
        pck32 = {model: float(model_rows[model]["macro_pck32"]) for model in MODELS}
        row = {
            "scope": scope,
            "step": step,
            "block": block,
            "head": head,
            "models": 3,
            "valid_cases": sum(int(item["cases"]) for item in rows),
            "total_cases": 150,
            "comparisons": sum(int(item["comparisons"]) for item in rows),
            "timestep": statistics.fmean(float(item["timestep"]) for item in rows),
            "sigma": statistics.fmean(float(item["sigma"]) for item in rows),
            **{metric: statistics.fmean(float(item[metric]) for item in rows) for metric in METRICS},
            "worst_model_macro_pck32": min(pck32.values()),
            "best_model_macro_pck32": max(pck32.values()),
            "std_model_macro_pck32": statistics.pstdev(pck32.values()),
            "gt_macro_pck32": pck32["gt"],
            "lora_macro_pck32": pck32["lora"],
            "baseline_macro_pck32": pck32["baseline"],
        }
        combined.append(row)

    for scope in ("objects", "background"):
        ranked = sorted(
            (row for row in combined if row["scope"] == scope),
            key=lambda row: (-row["macro_pck32"], row["macro_mean_error_px"], -row["worst_model_macro_pck32"]),
        )
        for rank, row in enumerate(ranked, start=1):
            row["rank_macro_pck32"] = rank

    fieldnames = [
        "scope", "rank_macro_pck32", "step", "block", "head", "models",
        "valid_cases", "total_cases", "comparisons", "timestep", "sigma", *METRICS,
        "worst_model_macro_pck32", "best_model_macro_pck32", "std_model_macro_pck32",
        "gt_macro_pck32", "lora_macro_pck32", "baseline_macro_pck32",
    ]
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(combined, key=lambda row: (row["scope"], row["rank_macro_pck32"])))

    objects = sorted(
        (row for row in combined if row["scope"] == "objects"),
        key=lambda row: row["rank_macro_pck32"],
    )
    lines = [
        "# Three-model combined all-step ranking",
        "",
        "GT, LoRA, and Wan2.2 Baseline receive equal model weight.",
        "",
        "| rank | step | block | head | mean PCK@32 | worst model PCK@32 | mean error | GT | LoRA | Baseline |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in objects[:30]:
        lines.append(
            f"| {row['rank_macro_pck32']} | S{row['step']:03d} | L{row['block']:02d} | H{row['head']:02d} | "
            f"{row['macro_pck32']:.2f}% | {row['worst_model_macro_pck32']:.2f}% | "
            f"{row['macro_mean_error_px']:.2f}px | {row['gt_macro_pck32']:.2f}% | "
            f"{row['lora_macro_pck32']:.2f}% | {row['baseline_macro_pck32']:.2f}% |"
        )
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"rows={len(combined)} object_combinations={len(objects)}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
