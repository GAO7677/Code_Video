#!/usr/bin/env python3
"""Aggregate all-case metrics for every model/block/head combination."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_headwise_50case")
MODELS = {
    "gt": "GT teacher-forced",
    "lora": "LoRA step-000500",
    "baseline": "Wan2.2 Baseline",
}
METRICS = ("pck8", "pck16", "pck32", "mean_error_px")
HEAD_PATTERN = re.compile(r"qk_head(\d+)$")


def weighted(rows: list[dict], key: str) -> float:
    comparisons = sum(int(row["comparisons"]) for row in rows)
    return sum(float(row[key]) * int(row["comparisons"]) for row in rows) / comparisons


def aggregate_case(model: str, case: str, rows: list[dict]) -> list[dict]:
    groups: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for row in rows:
        match = HEAD_PATTERN.fullmatch(str(row.get("method", "")))
        scope = row.get("region_type")
        if not match or scope not in ("object", "background") or int(row.get("comparisons", 0)) <= 0:
            continue
        groups[(int(row["layer"]), int(match.group(1)), scope)].append(row)
    result = []
    for (block, head, scope), selected in groups.items():
        item = {
            "model": model,
            "case": case,
            "block": block,
            "head": head,
            "scope": scope,
            "comparisons": sum(int(row["comparisons"]) for row in selected),
        }
        item.update({key: weighted(selected, key) for key in METRICS})
        result.append(item)
    return result


def main() -> None:
    case_rows = []
    model_case_counts = {}
    for model in MODELS:
        paths = sorted((ROOT / model / "cases").glob("*/metrics.json"))
        model_case_counts[model] = len(paths)
        for path in paths:
            rows = json.loads(path.read_text())
            case_rows.extend(aggregate_case(model, path.parent.name, rows))

    summary_groups: dict[tuple[str, int, int, str], list[dict]] = defaultdict(list)
    for row in case_rows:
        summary_groups[(row["model"], row["block"], row["head"], row["scope"])].append(row)

    summary = []
    for (model, block, head, scope), rows in summary_groups.items():
        comparisons = sum(int(row["comparisons"]) for row in rows)
        item = {
            "model": model,
            "model_label": MODELS[model],
            "block": block,
            "head": head,
            "scope": scope,
            "total_cases": model_case_counts[model],
            "valid_cases": len(rows),
            "comparisons": comparisons,
        }
        for key in METRICS:
            item[f"macro_{key}"] = sum(float(row[key]) for row in rows) / len(rows)
            item[f"pooled_{key}"] = sum(float(row[key]) * int(row["comparisons"]) for row in rows) / comparisons
        summary.append(item)

    summary.sort(key=lambda row: (row["model"], row["scope"], -row["macro_pck32"], row["macro_mean_error_px"], row["block"], row["head"]))
    ranks = defaultdict(int)
    for row in summary:
        rank_key = (row["model"], row["scope"])
        ranks[rank_key] += 1
        row["rank_pck32"] = ranks[rank_key]

    csv_fields = [
        "model", "model_label", "scope", "rank_pck32", "block", "head",
        "total_cases", "valid_cases", "comparisons",
        *[f"macro_{key}" for key in METRICS],
        *[f"pooled_{key}" for key in METRICS],
    ]
    with (ROOT / "block_head_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary)

    payload = {"models": {}}
    for model, label in MODELS.items():
        payload["models"][model] = {
            "label": label,
            "total_cases": model_case_counts[model],
            "scopes": {
                scope: [row for row in summary if row["model"] == model and row["scope"] == scope]
                for scope in ("object", "background")
            },
        }
    (ROOT / "combination_rankings.json").write_text(json.dumps(payload, indent=2))

    top = {
        model: payload["models"][model]["scopes"]["object"][:10]
        for model in MODELS
    }
    (ROOT / "top_combinations.json").write_text(json.dumps(top, indent=2))

    lines = [
        "# 50-case all-block, all-head combination ranking",
        "",
        "Ranking: macro Object PCK@32 descending; ties use macro mean error ascending.",
        "Each case is weighted equally after comparison-weighted aggregation of its object regions.",
        "",
    ]
    for model, label in MODELS.items():
        lines.extend([
            f"## {label}", "",
            "| rank | block | head | valid/total cases | macro PCK@8 | macro PCK@16 | macro PCK@32 | macro error | pooled PCK@32 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in top[model]:
            lines.append(
                f"| {row['rank_pck32']} | B{row['block']:02d} | H{row['head']:02d} | "
                f"{row['valid_cases']}/{row['total_cases']} | {row['macro_pck8']:.2f}% | "
                f"{row['macro_pck16']:.2f}% | {row['macro_pck32']:.2f}% | "
                f"{row['macro_mean_error_px']:.2f}px | {row['pooled_pck32']:.2f}% |"
            )
        lines.append("")
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print(f"aggregated {len(case_rows)} case rows into {len(summary)} model/block/head/scope rows")
    print(ROOT / "block_head_summary.csv")


if __name__ == "__main__":
    main()
