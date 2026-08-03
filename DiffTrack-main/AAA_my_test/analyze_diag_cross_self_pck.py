#!/usr/bin/env python3
"""Aggregate diagonal cross/self metrics and compare them with PCK@32."""

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
    "/data/gaoya/agent-data/outputs/three_model_all720_diag_cross_self_5case"
)
PCK_PATH = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_uniform_diagonal_5case/"
    "all720_uniform_diagonal_summary.csv"
)
MODELS = ("gt", "lora", "baseline")
FIELDS = (
    "queryframe_diagonal_cross_self_log_mean",
    "queryframe_diagonal_cross_self_log_median",
    "queryframe_diagonal_cross_self_log_std",
    "queryframe_diagonal_cross_dominant_fraction",
)


def correlations(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def binned_rows(
    x: np.ndarray, y: np.ndarray, edges: np.ndarray, labels: list[str]
) -> list[dict[str, float | int | str]]:
    output = []
    for index, label in enumerate(labels):
        if index == len(labels) - 1:
            mask = (x >= edges[index]) & (x <= edges[index + 1])
        else:
            mask = (x >= edges[index]) & (x < edges[index + 1])
        values = y[mask]
        if not len(values):
            continue
        output.append(
            {
                "interval": label,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(len(values)),
                "mean_pck32": float(values.mean()),
                "median_pck32": float(np.median(values)),
                "std_pck32": float(values.std()),
                "pck75_fraction": float((values >= 75).mean()),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    records: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    file_count = 0
    record_count = 0
    per_model_counts = defaultdict(int)
    for model in MODELS:
        paths = sorted(
            (ROOT / model / "cases").glob(
                "case_*/all_token_qk/uniform_diagonal_metrics.csv"
            )
        )
        for path in paths:
            file_count += 1
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 720:
                raise RuntimeError(f"expected 720 rows in {path}, found {len(rows)}")
            for row in rows:
                key = (int(row["block"]), int(row["head"]))
                for field in FIELDS:
                    value = float(row[field])
                    records[key][field].append(value)
                    records[key][f"{model}_{field}"].append(value)
                record_count += 1
                per_model_counts[model] += 1
    if file_count != 15 or len(records) != 720:
        raise RuntimeError(
            f"expected 15 files and 720 combinations, got {file_count}, {len(records)}"
        )

    with PCK_PATH.open(encoding="utf-8") as handle:
        pck = {
            (int(row["block"]), int(row["head"])): float(row["pck32"])
            for row in csv.DictReader(handle)
        }

    summary = []
    for block, head in sorted(records):
        values = records[(block, head)]
        row = {"block": block, "head": head, "pck32": pck[(block, head)]}
        for field in FIELDS:
            row[field] = float(np.mean(values[field]))
            for model in MODELS:
                row[f"{model}_{field}"] = float(
                    np.mean(values[f"{model}_{field}"])
                )
        summary.append(row)
    write_csv(ROOT / "diag_cross_self_summary.csv", summary)

    y = np.asarray([row["pck32"] for row in summary])
    median_key = "queryframe_diagonal_cross_self_log_median"
    fraction_key = "queryframe_diagonal_cross_dominant_fraction"
    x = np.asarray([row[median_key] for row in summary])
    fraction = np.asarray([row[fraction_key] for row in summary])
    abs_x = np.abs(x)

    report = {
        "files": file_count,
        "case_model_head_rows": record_count,
        "combinations": len(summary),
        "per_model_rows": dict(per_model_counts),
        "metric_definition": (
            "median over query-frame-1 tokens of log(((other-six-frame "
            "diagonal mass)/6 + eps)/(same-frame diagonal mass + eps))"
        ),
        "log_median_vs_pck32": correlations(x, y),
        "cross_dominant_fraction_vs_pck32": correlations(fraction, y),
        "absolute_log_median_vs_pck32": correlations(abs_x, y),
        "per_model_log_median_vs_combined_pck32": {},
    }
    for model in MODELS:
        model_x = np.asarray([row[f"{model}_{median_key}"] for row in summary])
        report["per_model_log_median_vs_combined_pck32"][model] = correlations(
            model_x, y
        )

    linear = np.polyfit(x, y, 1)
    quadratic = np.polyfit(x, y, 2)
    y_linear = np.polyval(linear, x)
    y_quadratic = np.polyval(quadratic, x)
    denominator = float(((y - y.mean()) ** 2).sum())
    report["linear_r2"] = float(1 - ((y - y_linear) ** 2).sum() / denominator)
    report["quadratic_r2"] = float(
        1 - ((y - y_quadratic) ** 2).sum() / denominator
    )
    report["quadratic_peak_log_ratio"] = float(-quadratic[1] / (2 * quadratic[0]))

    fixed_edges = np.asarray([-np.inf, -1, -0.5, 0, 0.5, 1, np.inf])
    fixed_labels = ["<-1", "[-1,-0.5)", "[-0.5,0)", "[0,0.5)", "[0.5,1)", ">=1"]
    fixed = binned_rows(x, y, fixed_edges, fixed_labels)
    write_csv(ROOT / "diag_cross_self_fixed_bins.csv", fixed)

    quantile_edges = np.quantile(x, np.linspace(0, 1, 11))
    quantile_labels = [f"Q{index + 1}" for index in range(10)]
    quantiles = binned_rows(x, y, quantile_edges, quantile_labels)
    write_csv(ROOT / "diag_cross_self_deciles.csv", quantiles)
    report["best_fixed_interval"] = max(fixed, key=lambda row: row["mean_pck32"])
    report["best_decile"] = max(quantiles, key=lambda row: row["mean_pck32"])
    (ROOT / "diag_cross_self_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    order = np.argsort(x)
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axis = axes[0, 0]
    scatter = axis.scatter(x, y, c=fraction, cmap="viridis", s=18, alpha=0.72)
    axis.plot(x[order], y_quadratic[order], color="#c74b32", lw=2)
    axis.axvline(0, color="#222", ls="--", lw=1)
    axis.set(xlabel="Diagonal cross/self log median", ylabel="PCK@32 (%)", title="720 combinations")
    figure.colorbar(scatter, ax=axis, label="Cross-dominant token fraction")

    axis = axes[0, 1]
    centers = np.arange(len(fixed))
    axis.bar(centers, [row["mean_pck32"] for row in fixed], color="#287a67")
    axis.errorbar(
        centers,
        [row["mean_pck32"] for row in fixed],
        yerr=[row["std_pck32"] for row in fixed],
        fmt="none",
        color="#17211e",
        capsize=3,
    )
    axis.set_xticks(centers, [row["interval"] for row in fixed])
    axis.set(xlabel="Fixed log-ratio interval", ylabel="Mean PCK@32 (%)", title="Fixed-bin PCK")
    for index, row in enumerate(fixed):
        axis.text(index, row["mean_pck32"] + 1, f'n={row["count"]}', ha="center", fontsize=8)

    axis = axes[1, 0]
    axis.scatter(fraction, y, s=18, alpha=0.65, color="#c65738")
    axis.set(
        xlabel="Cross-dominant query-token fraction",
        ylabel="PCK@32 (%)",
        title="Whether other-frame mean exceeds same-frame",
    )

    axis = axes[1, 1]
    centers = np.arange(len(quantiles))
    axis.plot(centers, [row["mean_pck32"] for row in quantiles], marker="o", color="#176654")
    axis.set_xticks(centers, [row["interval"] for row in quantiles])
    axis.set(xlabel="Log-ratio decile", ylabel="Mean PCK@32 (%)", title="Equal-count deciles")
    for index, row in enumerate(quantiles):
        axis.text(index, row["mean_pck32"] + 0.8, f'{row["mean_pck32"]:.1f}', ha="center", fontsize=8)
    figure.savefig(ROOT / "diag_cross_self_pck_relationship.png", dpi=180)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
