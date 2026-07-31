#!/usr/bin/env python3
"""Merge two dit_ablation_metric_stats.csv tables and plot per-method curves.

Reads the metric-stats CSVs produced by ``plot_dit_ablation_metrics.py`` (one
per run), classifies every row by its ``result_dir`` PATH (not by the possibly
wrong ``model``/``ablation`` columns), merges them, and draws one multi-panel
figure. Each metric is one panel; the x-axis is the ablated block; each method
(model + ablation kind) is one colored curve. Per-model baselines are drawn as
horizontal dashed reference lines.

This script only READS the existing CSV tables. It does not modify or re-run
any existing script.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_CSVS = (
    Path(
        "/data/gaoya/AAA_test_video/0623/test/v2v_wan/_metric_plots/"
        "leaf_folders/dit_ablation_metric_stats.csv"
    ),
    Path(
        "/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG/_metric_plots/"
        "rvg_leaf_folders/dit_ablation_metric_stats.csv"
    ),
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v_wan/_metric_plots/merged"
)

# Classify strictly from the result_dir path.
MODE_RE = re.compile(
    r"(whole_block|self_attn_zero|object_cross_attn|"
    r"text_cross_attn_zero|ffn_zero|lora_off)_block(\d+)"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MODEL_ORDER = {"wan_lora": 0, "xssc": 1, "physrvg": 2}
MODE_LABELS = {
    "whole_block": "Whole block bypass",
    "self_attn_zero": "Self-attn = 0",
    "object_cross_attn": "Object cross-attn = 0",
    "text_cross_attn_zero": "Text cross-attn = 0",
    "ffn_zero": "FFN = 0",
    "lora_off": "LoRA disabled",
}
MODE_ORDER = {
    mode: idx
    for idx, mode in enumerate(
        (
            "whole_block",
            "self_attn_zero",
            "object_cross_attn",
            "text_cross_attn_zero",
            "ffn_zero",
            "lora_off",
        )
    )
}
BASELINE_COLORS = {
    "wan_lora": "#555555",
    "xssc": "#8A8A8A",
    "physrvg": "#B0B0B0",
}
# Metric titles / display order, mirroring the source script for consistency.
METRIC_TITLES = {
    "physics_iq_with_context": "Physics-IQ with context",
    "physics_iq_without_context": "Physics-IQ without context",
    "physics_iq_verified_proxy": "Physics-IQ Verified proxy",
    "pmf_with_context": "PMF with context",
    "pmf_without_context": "PMF without context",
    "wmreward": "WMReward surprise",
    "vbench_subject_consistency": "VBench subject consistency",
    "vbench_background_consistency": "VBench background consistency",
    "vbench_temporal_flickering": "VBench temporal flickering",
    "vbench_motion_smoothness": "VBench motion smoothness",
    "vbench_dynamic_degree": "VBench dynamic degree",
    "vbench_aesthetic_quality": "VBench aesthetic quality",
    "vbench_imaging_quality": "VBench imaging quality",
    "videophy2_sa": "VideoPhy2 SA (generated only)",
    "videophy2_pc": "VideoPhy2 PC (generated only)",
    "videophy2_joint_rate": "VideoPhy2 joint rate (generated only)",
    "videophy2_pc_raw": "VideoPhy2 PC raw (full video)",
    "cosmos_reason1": "Cosmos-Reason1",
}
METRIC_ORDER = {key: idx for idx, key in enumerate(METRIC_TITLES)}
def classify_from_path(result_dir: str) -> tuple[str, str, int | None]:
    """Return (model, mode, block_id) inferred from the result_dir path only."""
    parts_lower = [p.lower() for p in Path(result_dir).parts]
    part_set = set(parts_lower)
    if "wan_lora" in part_set:
        model = "wan_lora"
    elif "xssc" in part_set:
        model = "xssc"
    elif "phyrvg" in part_set or "physrvg" in part_set:
        model = "physrvg"
    else:
        raise ValueError(f"Cannot infer model from path: {result_dir}")

    if "baseline" in part_set:
        return model, "baseline", None
    match = MODE_RE.search(result_dir)
    if match is None:
        raise ValueError(f"Cannot infer ablation from path: {result_dir}")
    return model, match.group(1), int(match.group(2))


def method_key(model: str, mode: str) -> tuple[str, str]:
    return (model, mode)


def method_label(model: str, mode: str) -> str:
    return f"{MODEL_LABELS[model]} - {MODE_LABELS[mode]}"


def method_sort_key(model: str, mode: str) -> tuple[int, int]:
    return (MODEL_ORDER.get(model, 99), MODE_ORDER.get(mode, 99))


# points[metric][(model, mode)][block_id] = mean ; baselines[metric][model] = mean
def read_csvs(
    csv_paths: list[Path], complete_only: bool
) -> tuple[dict, dict, set, set]:
    points: dict[str, dict[tuple[str, str], dict[int, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    baselines: dict[str, dict[str, float]] = defaultdict(dict)
    seen_methods: set[tuple[str, str]] = set()
    seen_blocks: set[int] = set()

    for csv_path in csv_paths:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if complete_only and row.get("complete_67", "").strip() != "True":
                    continue
                mean_raw = row.get("mean", "").strip()
                if not mean_raw:
                    continue
                mean = float(mean_raw)
                if not math.isfinite(mean):
                    continue
                metric = row["metric"]
                model, mode, block_id = classify_from_path(row["result_dir"])
                if mode == "baseline":
                    baselines[metric][model] = mean
                    continue
                points[metric][method_key(model, mode)][block_id] = mean
                seen_methods.add(method_key(model, mode))
                seen_blocks.add(block_id)
    return points, baselines, seen_methods, seen_blocks


def build_color_map(methods: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    cmap = plt.get_cmap("tab20")
    palette = [cmap(i) for i in range(cmap.N)]
    if len(methods) > len(palette):
        extra = plt.get_cmap("tab20b")
        palette += [extra(i) for i in range(extra.N)]
    return {method: palette[i % len(palette)] for i, method in enumerate(methods)}


def direction_from_csv(csv_paths: list[Path]) -> dict[str, str]:
    directions: dict[str, str] = {}
    for csv_path in csv_paths:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                directions.setdefault(row["metric"], row.get("direction", "higher"))
    return directions


def write_merged_csv(path: Path, csv_paths: list[Path]) -> None:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for csv_path in csv_paths:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and not fieldnames:
                fieldnames = list(reader.fieldnames)
                if "source_csv" not in fieldnames:
                    fieldnames.append("source_csv")
                if "path_model" not in fieldnames:
                    fieldnames += ["path_model", "path_mode", "path_block"]
            for row in reader:
                model, mode, block_id = classify_from_path(row["result_dir"])
                row["source_csv"] = str(csv_path)
                row["path_model"] = model
                row["path_mode"] = mode
                row["path_block"] = "" if block_id is None else str(block_id)
                rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def best_over_blocks(
    per_block: dict[int, float], direction: str
) -> tuple[int, float] | None:
    """Return (block_id, value) of the best ablation point for one metric."""
    finite = {b: v for b, v in per_block.items() if math.isfinite(v)}
    if not finite:
        return None
    picker = max if direction == "higher" else min
    block = picker(finite, key=finite.get)
    return block, finite[block]


def write_model_table(
    csv_path: Path,
    model: str,
    points: dict,
    baselines: dict,
    blocks: list[int],
    directions: dict[str, str],
) -> list[dict[str, str]]:
    """Wide table for one baseline/model: rows = metric, cols = each mode x block.

    Also records the per-metric baseline, the best ablation config, its value,
    and the delta vs baseline. Returns the best-summary rows for the caller.
    """
    modes = sorted(
        {mode for (m, mode) in {mm for by in points.values() for mm in by} if m == model},
        key=lambda mode: MODE_ORDER.get(mode, 99),
    )
    metrics = sorted(points.keys(), key=lambda m: METRIC_ORDER.get(m, 999))

    col_pairs = [(mode, block) for mode in modes for block in blocks]
    header = ["metric", "direction", "baseline"]
    header += [f"{mode}_block{block:02d}" for (mode, block) in col_pairs]
    header += ["best_config", "best_value", "delta_vs_baseline"]

    summary_rows: list[dict[str, str]] = []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for metric in metrics:
            direction = directions.get(metric, "higher")
            base = baselines.get(metric, {}).get(model)
            row = [metric, direction, _fmt(base)]

            # Flatten all (mode, block) values, find global best across modes+blocks.
            all_points: dict[tuple[str, int], float] = {}
            for mode in modes:
                per_block = points[metric].get((model, mode), {})
                for block in blocks:
                    value = per_block.get(block)
                    row.append(_fmt(value))
                    if value is not None and math.isfinite(value):
                        all_points[(mode, block)] = value

            if all_points:
                picker = max if direction == "higher" else min
                (best_mode, best_block) = picker(all_points, key=all_points.get)
                best_value = all_points[(best_mode, best_block)]
                best_config = f"{best_mode}_block{best_block:02d}"
                delta = (
                    "" if base is None else f"{best_value - base:+.4f}"
                )
            else:
                best_config, best_value, delta = "", None, ""
            row += [best_config, _fmt(best_value), delta]
            writer.writerow(row)

            summary_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "metric": metric,
                    "metric_title": METRIC_TITLES.get(metric, metric),
                    "direction": direction,
                    "baseline": _fmt(base),
                    "best_config": best_config,
                    "best_value": _fmt(best_value),
                    "delta_vs_baseline": delta,
                }
            )
    return summary_rows


def write_best_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "model", "model_label", "metric", "metric_title", "direction",
        "baseline", "best_config", "best_value", "delta_vs_baseline",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_summary_md(
    path: Path, rows_by_model: dict[str, list[dict[str, str]]]
) -> None:
    lines = ["# 各 baseline 下每个指标的最优 block 消融结果", ""]
    for model in sorted(rows_by_model, key=lambda m: MODEL_ORDER.get(m, 99)):
        lines.append(f"## {MODEL_LABELS[model]}")
        lines.append("")
        lines.append(
            "| 指标 | 方向 | Baseline | 最优消融配置 | 最优值 | Δ vs baseline |"
        )
        lines.append("| --- | :---: | ---: | --- | ---: | ---: |")
        for r in rows_by_model[model]:
            arrow = "↑" if r["direction"] == "higher" else "↓"
            lines.append(
                f"| {r['metric_title']} | {arrow} | {r['baseline']} | "
                f"{r['best_config'] or '—'} | **{r['best_value'] or '—'}** | "
                f"{r['delta_vs_baseline'] or '—'} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(
    output_png: Path,
    output_pdf: Path,
    points: dict,
    baselines: dict,
    methods: list[tuple[str, str]],
    blocks: list[int],
    directions: dict[str, str],
    color_map: dict[tuple[str, str], str],
    dpi: int,
    title: str = "Merged DiT Block Ablation Metrics (by method)",
) -> None:
    metrics = sorted(points.keys(), key=lambda m: METRIC_ORDER.get(m, 999))
    x_positions = np.arange(len(blocks))
    num_columns = 2
    num_rows = math.ceil(len(metrics) / num_columns)
    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(19, 4.5 * num_rows), constrained_layout=False
    )
    axes_flat = list(np.atleast_1d(axes).flat)

    for axis, metric in zip(axes_flat, metrics):
        for method in methods:
            per_block = points[metric].get(method, {})
            values = [per_block.get(block, np.nan) for block in blocks]
            if not np.isfinite(values).any():
                continue
            axis.plot(
                x_positions,
                values,
                color=color_map[method],
                linestyle="-",
                marker="o",
                markersize=5,
                linewidth=2,
                alpha=0.95,
                zorder=2,
            )
        for model, base in sorted(
            baselines.get(metric, {}).items(), key=lambda kv: MODEL_ORDER.get(kv[0], 99)
        ):
            if math.isfinite(base):
                axis.axhline(
                    base,
                    color=BASELINE_COLORS.get(model, "#202020"),
                    linestyle="--",
                    linewidth=1.6,
                    alpha=0.9,
                    zorder=1,
                )
        symbol = "↑" if directions.get(metric, "higher") == "higher" else "↓"
        axis.set_title(
            f"{METRIC_TITLES.get(metric, metric)} ({symbol})",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        axis.set_xticks(x_positions, [str(block) for block in blocks])
        axis.set_xlabel("Block", fontsize=11)
        axis.set_ylabel("Mean score", fontsize=11)
        axis.grid(axis="both", color="#D9DDDF", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)

    for axis in axes_flat[len(metrics):]:
        axis.axis("off")

    legend_handles = [
        Line2D([0], [0], color=color_map[method], linewidth=3, marker="o",
               label=method_label(*method))
        for method in methods
    ]
    legend_handles += [
        Line2D([0], [0], color=BASELINE_COLORS.get(model, "#202020"), linewidth=2,
               linestyle="--", label=f"{MODEL_LABELS[model]} baseline")
        for model in sorted(
            {m for base in baselines.values() for m in base},
            key=lambda m: MODEL_ORDER.get(m, 99),
        )
    ]
    fig.suptitle(
        title,
        fontsize=24,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=4,
        frameon=False,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.008,
        "Methods classified by result path. X-axis = ablated block. "
        "WMReward is lower-is-better; other metrics higher-is-better.",
        ha="center",
        fontsize=11,
        color="#4D5559",
    )
    top = 1.0 - min(0.14, 0.02 + 0.012 * math.ceil(len(legend_handles) / 4))
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.05, top=top, hspace=0.42, wspace=0.24
    )
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two ablation metric CSVs and plot per-method curves."
    )
    parser.add_argument(
        "--csv", type=Path, action="append", default=None,
        help="Metric-stats CSV path (repeatable). Defaults to the two known runs.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--complete-only", action="store_true",
        help="Only use rows with complete_67 == True (default: use all rows).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_paths = [p.expanduser().resolve() for p in (args.csv or list(DEFAULT_CSVS))]
    missing = [str(p) for p in csv_paths if not p.is_file()]
    if missing:
        raise SystemExit("Missing CSV(s):\n  " + "\n  ".join(missing))

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points, baselines, seen_methods, seen_blocks = read_csvs(
        csv_paths, args.complete_only
    )
    if not seen_methods:
        raise SystemExit("No ablation method points found in the given CSVs.")

    methods = sorted(seen_methods, key=lambda mm: method_sort_key(*mm))
    blocks = sorted(seen_blocks)
    directions = direction_from_csv(csv_paths)
    color_map = build_color_map(methods)

    merged_csv = output_dir / "merged_ablation_metric_stats.csv"
    write_merged_csv(merged_csv, csv_paths)

    # Per-baseline wide tables + best-per-metric summary.
    models_all = sorted(
        {model for model, _ in methods}, key=lambda m: MODEL_ORDER.get(m, 99)
    )
    summary_by_model: dict[str, list[dict[str, str]]] = {}
    model_table_paths: list[tuple[str, Path]] = []
    for model in models_all:
        table_path = output_dir / f"ablation_table_{model}.csv"
        summary_by_model[model] = write_model_table(
            table_path, model, points, baselines, blocks, directions
        )
        model_table_paths.append((model, table_path))
    best_csv = output_dir / "ablation_best_per_metric.csv"
    best_md = output_dir / "ablation_best_per_metric.md"
    write_best_summary_csv(
        best_csv, [r for model in models_all for r in summary_by_model[model]]
    )
    write_best_summary_md(best_md, summary_by_model)

    # Combined figure across all baselines/models.
    output_png = output_dir / "merged_ablation_by_method.png"
    output_pdf = output_dir / "merged_ablation_by_method.pdf"
    plot(
        output_png, output_pdf, points, baselines, methods, blocks,
        directions, color_map, args.dpi,
        title="Merged DiT Block Ablation Metrics (all baselines)",
    )

    # One figure per baseline (model), keeping colors consistent with combined.
    models_present = sorted(
        {model for model, _ in methods}, key=lambda m: MODEL_ORDER.get(m, 99)
    )
    per_model_outputs: list[tuple[str, Path]] = []
    for model in models_present:
        model_methods = [mm for mm in methods if mm[0] == model]
        # Restrict points/baselines to this model only.
        model_points = {
            metric: {mm: per for mm, per in by_method.items() if mm[0] == model}
            for metric, by_method in points.items()
        }
        model_points = {
            metric: by_method for metric, by_method in model_points.items() if by_method
        }
        model_baselines = {
            metric: {m: v for m, v in base.items() if m == model}
            for metric, base in baselines.items()
        }
        model_png = output_dir / f"ablation_{model}_by_method.png"
        model_pdf = output_dir / f"ablation_{model}_by_method.pdf"
        plot(
            model_png, model_pdf, model_points, model_baselines, model_methods,
            blocks, directions, color_map, args.dpi,
            title=f"{MODEL_LABELS[model]} DiT Block Ablation Metrics",
        )
        per_model_outputs.append((model, model_png))

    print(f"Input CSVs: {len(csv_paths)}")
    for p in csv_paths:
        print(f"  {p}")
    print(f"Methods ({len(methods)}):")
    for model, mode in methods:
        blks = sorted(
            {b for metric in points for b in points[metric].get((model, mode), {})}
        )
        print(f"  {method_label(model, mode)}  blocks={blks}")
    print(f"Blocks: {blocks}")
    print(f"Metrics: {len(points)}")
    print(f"Merged CSV: {merged_csv}")
    print("Per-baseline tables:")
    for model, table_path in model_table_paths:
        print(f"  [{MODEL_LABELS[model]}] {table_path}")
    print(f"Best-per-metric CSV: {best_csv}")
    print(f"Best-per-metric MD:  {best_md}")
    print(f"Combined PNG: {output_png}")
    print(f"Combined PDF: {output_pdf}")
    print("Per-baseline PNGs:")
    for model, png in per_model_outputs:
        print(f"  [{MODEL_LABELS[model]}] {png}")


if __name__ == "__main__":
    main()
