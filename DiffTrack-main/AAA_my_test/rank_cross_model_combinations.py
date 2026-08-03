#!/usr/bin/env python3
"""Rank Block x Head combinations that are robust across all three models."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_headwise_50case")
SOURCE = ROOT / "block_head_summary.csv"
MODELS = ("gt", "lora", "baseline")


def main() -> None:
    combinations = defaultdict(dict)
    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "object":
                continue
            combinations[(int(row["block"]), int(row["head"]))][row["model"]] = row

    results = []
    for (block, head), per_model in combinations.items():
        if any(model not in per_model for model in MODELS):
            continue
        item = {"block": block, "head": head}
        for model in MODELS:
            row = per_model[model]
            item[f"{model}_pck8"] = float(row["macro_pck8"])
            item[f"{model}_pck16"] = float(row["macro_pck16"])
            item[f"{model}_pck32"] = float(row["macro_pck32"])
            item[f"{model}_error"] = float(row["macro_mean_error_px"])
            item[f"{model}_valid_cases"] = int(row["valid_cases"])
        for threshold in (8, 16, 32):
            values = [item[f"{model}_pck{threshold}"] for model in MODELS]
            item[f"min_pck{threshold}"] = min(values)
            item[f"mean_pck{threshold}"] = sum(values) / len(values)
            item[f"spread_pck{threshold}"] = max(values) - min(values)
        errors = [item[f"{model}_error"] for model in MODELS]
        item["worst_error"] = max(errors)
        item["mean_error"] = sum(errors) / len(errors)
        results.append(item)

    results.sort(key=lambda row: (
        -row["min_pck32"],
        -row["mean_pck32"],
        row["worst_error"],
        row["block"],
        row["head"],
    ))
    for rank, row in enumerate(results, start=1):
        row["robust_rank"] = rank

    fields = [
        "robust_rank", "block", "head",
        "min_pck8", "mean_pck8", "spread_pck8",
        "min_pck16", "mean_pck16", "spread_pck16",
        "min_pck32", "mean_pck32", "spread_pck32",
        "worst_error", "mean_error",
        *[f"{model}_{metric}" for model in MODELS for metric in ("pck8", "pck16", "pck32", "error", "valid_cases")],
    ]
    with (ROOT / "cross_model_robust_rankings.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (ROOT / "cross_model_robust_rankings.json").write_text(json.dumps({
        "ranking_rule": "min macro Object PCK@32 desc, mean PCK@32 desc, worst error asc",
        "combinations": results,
    }, indent=2))

    lines = [
        "# Cross-model robust Block x Head ranking",
        "",
        "Primary score: minimum Macro Object PCK@32 across GT, LoRA, and Baseline.",
        "Tie-breakers: mean PCK@32 descending, worst mean error ascending.",
        "",
        "| rank | combination | robust min PCK@32 | 3-model mean | GT | LoRA | Baseline | worst error |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results[:30]:
        lines.append(
            f"| {row['robust_rank']} | B{row['block']:02d} x H{row['head']:02d} | "
            f"{row['min_pck32']:.2f}% | {row['mean_pck32']:.2f}% | "
            f"{row['gt_pck32']:.2f}% | {row['lora_pck32']:.2f}% | "
            f"{row['baseline_pck32']:.2f}% | {row['worst_error']:.2f}px |"
        )
    (ROOT / "CROSS_MODEL_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"ranked {len(results)} shared combinations")
    for row in results[:10]:
        print(
            f"#{row['robust_rank']:02d} B{row['block']:02d}/H{row['head']:02d} "
            f"min={row['min_pck32']:.2f} mean={row['mean_pck32']:.2f} "
            f"gt={row['gt_pck32']:.2f} lora={row['lora_pck32']:.2f} "
            f"baseline={row['baseline_pck32']:.2f} worst_error={row['worst_error']:.2f}"
        )


if __name__ == "__main__":
    main()
