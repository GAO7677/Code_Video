#!/usr/bin/env python3
"""Aggregate 720-head current/neighbor-frame diagonal metrics."""

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
    "/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_5case"
)
SUMMARY = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "block_step_head_summary.csv"
)
MODELS = ("gt", "lora", "baseline")
METRICS = (
    "neighbor3_diagonal_mass",
    "neighbor3_diagonal_uniformity",
    "neighbor3_joint",
    "neighbor3_balanced_diagonal",
    "neighbor3_self_fraction",
    "allblock_diagonal_purity",
    "allblock_min_diagonal_purity",
    "allblock_p10_diagonal_purity",
    "allblock_offdiagonal_fraction",
    "neighbor3_allblock_diagonal_score",
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
        pattern = "cases/case_*/all_token_qk/uniform_diagonal_metrics.csv"
        for path in sorted((ROOT / model).glob(pattern)):
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
        row = {
            "block": block,
            "head": head,
            "pck32": float(np.mean([value["pck32_50case"] for value in values])),
        }
        for metric in METRICS:
            metric_values = [value[metric] for value in values]
            row[metric] = float(np.mean(metric_values))
            row[f"{metric}_std"] = float(np.std(metric_values))
        for model in MODELS:
            model_values = [value for value in values if value["model"] == model]
            row[f"{model}_pck32"] = float(
                np.mean([value["pck32_50case"] for value in model_values])
            )
            for metric in (
                "neighbor3_diagonal_uniformity",
                "neighbor3_balanced_diagonal",
                "allblock_diagonal_purity",
                "allblock_min_diagonal_purity",
                "neighbor3_allblock_diagonal_score",
            ):
                row[f"{model}_{metric}"] = float(
                    np.mean([value[metric] for value in model_values])
                )
        rows.append(row)

    rows.sort(
        key=lambda row: row["neighbor3_allblock_diagonal_score"], reverse=True
    )
    for rank, row in enumerate(rows, 1):
        row["strict_rank"] = rank

    output_csv = ROOT / "all720_neighbor_diagonal_summary.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    balanced = np.asarray([row["neighbor3_balanced_diagonal"] for row in rows])
    strict = np.asarray([row["neighbor3_allblock_diagonal_score"] for row in rows])
    purity = np.asarray([row["allblock_diagonal_purity"] for row in rows])
    min_purity = np.asarray([row["allblock_min_diagonal_purity"] for row in rows])
    uniformity = np.asarray([row["neighbor3_diagonal_uniformity"] for row in rows])
    mass = np.asarray([row["neighbor3_diagonal_mass"] for row in rows])
    joint = np.asarray([row["neighbor3_joint"] for row in rows])
    pck_values = np.asarray([row["pck32"] for row in rows])
    rank = np.arange(1, len(rows) + 1)
    pearson = pearsonr(strict, pck_values)
    spearman = spearmanr(strict, pck_values)

    figure, axes = plt.subplots(3, 1, figsize=(15, 14), constrained_layout=True)
    axes[0].plot(rank, strict, label="strict combined", color="#bd4f32", lw=2.2)
    axes[0].plot(rank, balanced, label="neighbor balanced", color="#d7893f", lw=1.2)
    axes[0].plot(rank, purity, label="all-block diagonal purity", color="#176b61", lw=1.4)
    axes[0].plot(rank, min_purity, label="weakest-block purity", color="#6a4c93", lw=1.2)
    axes[0].plot(rank, mass, label="neighbor diagonal mass", color="#176b61", lw=1.3)
    axes[0].plot(rank, uniformity, label="three-frame uniformity", color="#294d77", lw=1.3)
    axes[0].plot(rank, joint, label="mass x uniformity", color="#9a7625", lw=1.3)
    axes[0].set(
        xlabel="Rank by strict combined score",
        ylabel="Metric value",
        title="720 Block-Head combinations: current and adjacent frame diagonals",
    )
    axes[0].legend(ncol=3)

    for model, color in zip(MODELS, ("#2b4c7e", "#bb5a37", "#26715f")):
        axes[1].plot(
            rank,
            [row[f"{model}_neighbor3_allblock_diagonal_score"] for row in rows],
            label=model,
            color=color,
            lw=1.4,
        )
    axes[1].set(
        xlabel="Shared 720-head rank",
        ylabel="Per-model strict combined score",
        title="Cross-model consistency along the shared ranking",
    )
    axes[1].legend()

    scatter = axes[2].scatter(
        strict,
        pck_values,
        c=[row["block"] for row in rows],
        cmap="viridis",
        s=25,
        alpha=0.75,
    )
    axes[2].set(
        xlabel="Strict neighbor × all-block diagonal score",
        ylabel="Three-model mean Macro PCK@32",
        title=(
            f"PCK relationship: Pearson r={pearson.statistic:.3f}, "
            f"Spearman rho={spearman.statistic:.3f}"
        ),
    )
    figure.colorbar(scatter, ax=axes[2], label="Block")
    curve = ROOT / "all720_neighbor_diagonal_curves.png"
    figure.savefig(curve, dpi=180)
    plt.close(figure)

    report = {
        "combinations": len(rows),
        "case_model_rows": len(raw),
        "ranking_metric": "neighbor3_allblock_diagonal_score",
        "ranking_definition": (
            "mean(3 * min(diag[t-1], diag[t], diag[t+1]) * "
            "min_over_all_7_target_frames(diagonal_mass / frame_block_mass))"
        ),
        "pearson_r_with_pck32": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho_with_pck32": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "top30": rows[:30],
        "bottom30": rows[-30:],
        "summary_csv": str(output_csv),
        "curve": str(curve),
    }
    (ROOT / "all720_neighbor_diagonal_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
