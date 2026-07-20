#!/usr/bin/env python3
"""Plot concise curves for the DiffTrack-compatible Q/K grid experiment."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


OUTPUTS = Path("/data/gaoya/agent-data/outputs")
DEFAULT_OUTPUT = OUTPUTS / "difftrack_qk_with_video_20260719_report_assets"
DATASETS = {
    "PhysicIQ67": OUTPUTS
    / "physiciq67_difftrack_qk_grid_with_video/grid_ranking/grid_summary.csv",
    "Test100": OUTPUTS
    / "test100_51_difftrack_qk_grid_with_video/grid_ranking/grid_summary.csv",
}
MODELS = ("gt", "stage1b", "lora", "baseline")
GENERATED_MODELS = ("stage1b", "lora", "baseline")
LAYERS = (0, 5, 11, 17, 23, 29)
STEPS = (0, 10, 20, 29, 39)
RADII = (4, 8, 16, 32)

LAYER_COLORS = {
    0: "#64748b",
    5: "#d97706",
    11: "#0f766e",
    17: "#c2410c",
    23: "#2563eb",
    29: "#1f2937",
}
MODEL_COLORS = {
    "gt": "#111827",
    "stage1b": "#c2410c",
    "lora": "#0f766e",
    "baseline": "#2563eb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "object_centers":
                continue
            parsed: dict[str, Any] = {
                "model": row["model"],
                "layer": int(row["layer"]),
                "step": int(row["step_index"]),
                "mean_error_px": float(row["mean_error_px"]),
            }
            for radius in RADII:
                parsed[f"pck{radius}"] = float(row[f"pck{radius}"])
            parsed["mean_pck"] = float(row["mean_pck"])
            rows.append(parsed)
    return rows


def generated_average(
    rows: list[dict[str, Any]], layer: int, step: int, metric: str
) -> float:
    selected = [
        row[metric]
        for row in rows
        if row["model"] in GENERATED_MODELS
        and row["layer"] == layer
        and row["step"] == step
    ]
    if len(selected) != len(GENERATED_MODELS):
        raise RuntimeError(f"Missing generated rows for L{layer}/S{step}/{metric}")
    return sum(selected) / len(selected)


def style_axis(axis: Any) -> None:
    axis.set_facecolor("#fffdf7")
    axis.grid(axis="y", color="#d6d3ca", linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8c887d")
    axis.tick_params(colors="#374151")


def save_figure(figure: Any, output: Path) -> None:
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor="#f2eee3")
    plt.close(figure)


def add_figure_heading(figure: Any, title: str, subtitle: str) -> None:
    figure.suptitle(
        title,
        x=0.08,
        y=0.975,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color="#18211d",
    )
    figure.text(0.08, 0.915, subtitle, fontsize=10, color="#5f665f")


def plot_layer_step_curves(
    data: dict[str, list[dict[str, Any]]], output: Path
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True)
    plot_data: dict[str, Any] = {}
    for axis, (dataset, rows) in zip(axes, data.items(), strict=True):
        dataset_data: dict[str, Any] = {}
        for layer in LAYERS:
            values = [generated_average(rows, layer, step, "pck32") for step in STEPS]
            dataset_data[str(layer)] = values
            emphasis = layer in (5, 11, 17, 29)
            axis.plot(
                STEPS,
                values,
                marker="o",
                markersize=5.5 if emphasis else 4.5,
                linewidth=2.2 if emphasis else 1.4,
                alpha=1.0 if emphasis else 0.72,
                color=LAYER_COLORS[layer],
                label=f"L{layer}",
            )
        for step_index, step in enumerate(STEPS):
            best_layer = max(
                LAYERS,
                key=lambda layer: dataset_data[str(layer)][step_index],
            )
            best_value = dataset_data[str(best_layer)][step_index]
            axis.scatter(
                [step], [best_value], s=105, facecolors="none", edgecolors="#111827", linewidths=1.6, zorder=5
            )
            axis.annotate(
                f"L{best_layer}",
                (step, best_value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#111827",
            )
        style_axis(axis)
        axis.set_title(dataset, loc="left", fontsize=15, fontweight="bold")
        axis.set_xlabel("Denoising step index")
        axis.set_xticks(STEPS)
        axis.set_ylim(30, 100)
        plot_data[dataset] = dataset_data
    axes[0].set_ylabel("Model-balanced object-center PCK@32 (%)")
    axes[1].legend(ncol=2, frameon=False, loc="lower right")
    add_figure_heading(
        figure,
        "Where coarse correspondence is most readable",
        "Each curve fixes one Transformer layer; black rings mark the best layer at each step.",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(figure, output)
    return plot_data


def top_mean_pck_combinations(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    candidates = []
    for layer in LAYERS:
        for step in STEPS:
            score = generated_average(rows, layer, step, "mean_pck")
            error = generated_average(rows, layer, step, "mean_error_px")
            candidates.append((score, -error, -layer, -step, layer, step))
    candidates.sort(reverse=True)
    return [(layer, step) for _, _, _, _, layer, step in candidates[:3]]


def plot_radius_profiles(
    data: dict[str, list[dict[str, Any]]], output: Path
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True)
    rank_colors = ("#0f766e", "#c2410c", "#2563eb")
    plot_data: dict[str, Any] = {}
    for axis, (dataset, rows) in zip(axes, data.items(), strict=True):
        dataset_data = {}
        for rank, ((layer, step), color) in enumerate(
            zip(top_mean_pck_combinations(rows), rank_colors, strict=True), start=1
        ):
            values = [generated_average(rows, layer, step, f"pck{radius}") for radius in RADII]
            dataset_data[f"top{rank}"] = {
                "layer": layer,
                "step": step,
                "values": values,
            }
            axis.plot(
                RADII,
                values,
                marker="o",
                markersize=6,
                linewidth=2.4,
                color=color,
                label=f"Top-{rank}: L{layer}/S{step}",
            )
        style_axis(axis)
        axis.set_title(dataset, loc="left", fontsize=15, fontweight="bold")
        axis.set_xlabel("PCK radius (pixels)")
        axis.set_xticks(RADII)
        axis.set_ylim(0, 100)
        axis.legend(frameon=False, loc="upper left")
        plot_data[dataset] = dataset_data
    axes[0].set_ylabel("Model-balanced object-center PCK (%)")
    add_figure_heading(
        figure,
        "Top mean-PCK configurations are coarse, not pixel-precise",
        "The same fixed layer/step is evaluated at all four radii; configurations are ranked by mean-PCK.",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    # Legends in both panels otherwise make tight-bbox collapse the heading gap.
    figure.subplots_adjust(top=0.72)
    save_figure(figure, output)
    return plot_data


def plot_model_envelopes(
    data: dict[str, list[dict[str, Any]]], output: Path
) -> dict[str, Any]:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharey=True)
    label_offsets = {
        "gt": (-7, 9),
        "stage1b": (7, -13),
        "lora": (-8, -13),
        "baseline": (8, 9),
    }
    plot_data: dict[str, Any] = {}
    for axis, (dataset, rows) in zip(axes, data.items(), strict=True):
        dataset_data = {}
        for model in MODELS:
            values = []
            best_layers = []
            for step in STEPS:
                candidates = [
                    row
                    for row in rows
                    if row["model"] == model and row["step"] == step
                ]
                best = max(
                    candidates,
                    key=lambda row: (row["pck32"], -row["mean_error_px"], -row["layer"]),
                )
                values.append(best["pck32"])
                best_layers.append(best["layer"])
            dataset_data[model] = {"values": values, "best_layers": best_layers}
            axis.plot(
                STEPS,
                values,
                marker="o",
                markersize=5.5,
                linewidth=2.2,
                color=MODEL_COLORS[model],
                label=model.upper() if model == "gt" else model,
            )
            for step, value, layer in zip(STEPS, values, best_layers, strict=True):
                axis.annotate(
                    f"L{layer}",
                    (step, value),
                    xytext=label_offsets[model],
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=MODEL_COLORS[model],
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "#fffdf7",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                )
        style_axis(axis)
        axis.set_title(dataset, loc="left", fontsize=15, fontweight="bold")
        axis.set_xlabel("Denoising step index")
        axis.set_xticks(STEPS)
        axis.set_ylim(40, 100)
        plot_data[dataset] = dataset_data
    axes[0].set_ylabel("Per-model best-over-layer PCK@32 (%)")
    axes[1].legend(ncol=2, frameon=False, loc="lower right")
    add_figure_heading(
        figure,
        "Best-layer envelope differs by model and dataset",
        "Each point selects the best of six layers at that step; labels show the selected layer.",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    save_figure(figure, output)
    return plot_data


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {name: load_rows(path) for name, path in DATASETS.items()}
    payload = {
        "layer_step_curves": plot_layer_step_curves(
            data, output / "pck32_by_layer_and_step.png"
        ),
        "radius_profiles": plot_radius_profiles(
            data, output / "top3_mean_pck_radius_profiles.png"
        ),
        "model_envelopes": plot_model_envelopes(
            data, output / "pck32_best_layer_model_envelopes.png"
        ),
    }
    (output / "curve_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote curves to {output}")


if __name__ == "__main__":
    main()
