#!/usr/bin/env python3
"""Aggregate all 720 block/head uniform-diagonal metrics and render curves."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_uniform_diagonal_5case"
)
SUMMARY = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "block_step_head_summary.csv"
)
MODELS = ("gt", "lora", "baseline")
METRICS = (
    "queryframe_diagonal_mass",
    "queryframe_diagonal_frame_entropy",
    "queryframe_joint",
    "queryframe_balanced_diagonal",
)


def main() -> None:
    pck = {}
    with SUMMARY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] == "objects" and int(row["step"]) == 39:
                pck[(row["model"], int(row["layer"]), int(row["head"]))] = float(
                    row["macro_pck32"]
                )
    raw = []
    for model in MODELS:
        for path in sorted((ROOT / model / "cases").glob("case_*/all_token_qk/uniform_diagonal_metrics.csv")):
            case = path.parents[1].name
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    block, head = int(row["block"]), int(row["head"])
                    raw.append(
                        {
                            "model": model,
                            "case": case,
                            "block": block,
                            "head": head,
                            "pck32_50case": pck[(model, block, head)],
                            **{metric: float(row[metric]) for metric in METRICS},
                        }
                    )
    grouped = defaultdict(list)
    for row in raw:
        grouped[(row["block"], row["head"])].append(row)
    rows = []
    for (block, head), values in grouped.items():
        rows.append(
            {
                "block": block,
                "head": head,
                "pck32": float(np.mean([row["pck32_50case"] for row in values])),
                **{metric: float(np.mean([row[metric] for row in values])) for metric in METRICS},
                **{
                    f"{model}_{metric}": float(
                        np.mean([row[metric] for row in values if row["model"] == model])
                    )
                    for model in MODELS
                    for metric in ("queryframe_joint", "queryframe_balanced_diagonal")
                },
            }
        )
    rows.sort(key=lambda row: row["queryframe_joint"], reverse=True)
    output_csv = ROOT / "all720_uniform_diagonal_summary.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pck_values = np.asarray([row["pck32"] for row in rows])
    joint = np.asarray([row["queryframe_joint"] for row in rows])
    balanced = np.asarray([row["queryframe_balanced_diagonal"] for row in rows])
    diagonal = np.asarray([row["queryframe_diagonal_mass"] for row in rows])
    entropy = np.asarray([row["queryframe_diagonal_frame_entropy"] for row in rows])
    pearson = pearsonr(pck_values, joint)
    spearman = spearmanr(pck_values, joint)

    figure, axes = plt.subplots(3, 1, figsize=(15, 14), constrained_layout=True)
    rank = np.arange(1, len(rows) + 1)
    axes[0].plot(rank, joint, label="joint", linewidth=2.0, color="#c55335")
    axes[0].plot(rank, diagonal, label="diagonal mass", linewidth=1.2, color="#227965")
    axes[0].plot(rank, entropy, label="diagonal-frame entropy", linewidth=1.2, color="#304f78")
    axes[0].plot(rank, balanced, label="balanced diagonal", linewidth=1.2, color="#9b792c")
    axes[0].set(xlabel="Combination rank by joint score", ylabel="Metric value", title="All 720 Block/Head combinations")
    axes[0].legend(ncol=4)

    pck_order = np.argsort(-pck_values)
    axes[1].plot(rank, pck_values[pck_order] / 100.0, label="PCK@32 / 100", color="#17211e", linewidth=2)
    axes[1].plot(rank, joint[pck_order], label="joint score", color="#c55335", linewidth=1.5)
    axes[1].plot(rank, balanced[pck_order], label="balanced diagonal", color="#9b792c", linewidth=1.2)
    axes[1].set(xlabel="Combination rank by PCK@32", ylabel="Value", title="Uniform-diagonal metrics along the PCK@32 ranking")
    axes[1].legend()

    colors = np.asarray([row["block"] for row in rows])
    scatter = axes[2].scatter(joint, pck_values, c=colors, cmap="viridis", s=24, alpha=.75)
    slope, intercept = np.polyfit(joint, pck_values, 1)
    xline = np.linspace(joint.min(), joint.max(), 100)
    axes[2].plot(xline, slope * xline + intercept, color="#c55335", linewidth=2)
    axes[2].set(
        xlabel="Uniform spatial-diagonal joint score",
        ylabel="Three-model mean Macro PCK@32",
        title=f"PCK@32 relationship: Pearson r={pearson.statistic:.3f}, Spearman rho={spearman.statistic:.3f}",
    )
    figure.colorbar(scatter, ax=axes[2], label="Block")
    curve = ROOT / "all720_uniform_diagonal_curves.png"
    figure.savefig(curve, dpi=180)
    plt.close(figure)

    report = {
        "combinations": len(rows),
        "case_model_rows": len(raw),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "top10_joint": rows[:10],
        "bottom10_joint": rows[-10:],
        "summary_csv": str(output_csv),
        "curve": str(curve),
    }
    (ROOT / "all720_uniform_diagonal_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
