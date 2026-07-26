#!/usr/bin/env python3
"""Stitch and quantify five representative Block-17 Head heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_self_attention_head_roles import _metrics


CATEGORIES = {
    "S": (3, "intraframe spatial"),
    "T": (8, "moving-ball trajectory"),
    "P": (5, "fixed-position alignment"),
    "C": (23, "history/context"),
    "G": (7, "global aggregation"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _bin_frames(bins: int, token_count: int, spatial_tokens: int) -> np.ndarray:
    centers = np.minimum(
        ((np.arange(bins) + 0.5) * token_count / bins).astype(np.int64),
        token_count - 1,
    )
    return centers // spatial_tokens


def _temporal_matrix(
    key_mass: np.ndarray,
    *,
    bin_frames: np.ndarray,
    query_counts: np.ndarray,
    temporal_tokens: int,
) -> np.ndarray:
    result = np.zeros((temporal_tokens, temporal_tokens), dtype=np.float64)
    for query_t in range(temporal_tokens):
        query_ids = np.flatnonzero(bin_frames == query_t)
        weights = query_counts[query_ids].astype(np.float64)
        rows = np.average(key_mass[query_ids], axis=0, weights=weights)
        for key_t in range(temporal_tokens):
            result[query_t, key_t] = rows[bin_frames == key_t].sum()
    result /= np.maximum(result.sum(axis=1, keepdims=True), 1.0e-30)
    return result


def _plot_full_matrices(
    samples: list[dict],
    output_path: Path,
    boundaries: list[float],
) -> tuple[float, float]:
    matrices = np.stack([sample["matrix"] for sample in samples])
    positive = matrices[matrices > 0]
    epsilon = float(positive.min()) * 0.5
    display = np.log10(np.maximum(matrices, epsilon))
    low, high = np.percentile(display[np.isfinite(display)], [1.0, 99.8])

    figure, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=150)
    image = None
    for axis, sample, values in zip(axes.flat, samples, display):
        image = axis.imshow(
            values,
            cmap="magma",
            interpolation="nearest",
            origin="upper",
            vmin=float(low),
            vmax=float(high),
            aspect="equal",
        )
        for boundary in boundaries:
            axis.axhline(boundary, color="white", linewidth=0.25, alpha=0.45)
            axis.axvline(boundary, color="white", linewidth=0.25, alpha=0.45)
        axis.set_title(
            f"step {sample['step']:02d} | {sample['category']} "
            f"| H{sample['head']:02d}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    for row, step in enumerate((5, 15, 25, 35)):
        axes[row, 0].set_ylabel(f"step {step}\nquery bins")
    for column, category in enumerate(CATEGORIES):
        axes[-1, column].set_xlabel(f"{category} | key bins")
    figure.suptitle(
        "Wan+LoRA Block 17 | all-query/all-key attention | shared log10 scale",
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.05, right=0.94, top=0.94, bottom=0.05, wspace=0.06, hspace=0.13
    )
    assert image is not None
    colorbar_axis = figure.add_axes((0.955, 0.12, 0.012, 0.74))
    figure.colorbar(image, cax=colorbar_axis, label="log10 key-bin attention mass")
    figure.savefig(output_path)
    plt.close(figure)
    return float(low), float(high)


def _plot_temporal_matrices(samples: list[dict], output_path: Path) -> None:
    values = np.stack([sample["temporal_matrix"] for sample in samples])
    high = float(np.percentile(values, 99.5))
    figure, axes = plt.subplots(4, 5, figsize=(15.5, 12.5), dpi=150)
    image = None
    for axis, sample, matrix in zip(axes.flat, samples, values):
        image = axis.imshow(
            matrix,
            cmap="magma",
            interpolation="nearest",
            origin="upper",
            vmin=0.0,
            vmax=high,
            aspect="equal",
        )
        axis.set_title(
            f"step {sample['step']:02d} | {sample['category']} "
            f"| H{sample['head']:02d}"
        )
        axis.set_xticks(range(0, 13, 2))
        axis.set_yticks(range(0, 13, 2))
    for row in range(4):
        axes[row, 0].set_ylabel("query latent t")
    for column in range(5):
        axes[-1, column].set_xlabel("key latent t")
    figure.suptitle(
        "Latent-time attention mass | 13x13 | shared linear scale",
        fontsize=15,
    )
    figure.subplots_adjust(
        left=0.06, right=0.93, top=0.94, bottom=0.06, wspace=0.18, hspace=0.22
    )
    assert image is not None
    colorbar_axis = figure.add_axes((0.95, 0.12, 0.012, 0.74))
    figure.colorbar(image, cax=colorbar_axis, label="attention mass")
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    temporal_tokens, grid_h, grid_w = (
        int(value) for value in summary["latent_grid"]
    )
    token_count = temporal_tokens * grid_h * grid_w
    spatial_tokens = grid_h * grid_w
    steps = [int(value) for value in summary["step_numbers_one_based"]]

    samples = []
    records = []
    boundaries: list[float] | None = None
    for step in steps:
        entry = next(
            item
            for item in summary["steps"]
            if int(item["step_number_one_based"]) == step
        )
        matrix_path = root / entry["directory"] / entry["matrix_npz"]
        metadata = entry["matrix_metadata"]
        query_counts = np.asarray(metadata["query_bin_counts"], dtype=np.float64)
        bins = int(metadata["output_bins"])
        bin_frames = _bin_frames(bins, token_count, spatial_tokens)
        boundaries = [
            frame * spatial_tokens * bins / token_count - 0.5
            for frame in range(1, temporal_tokens)
        ]
        with np.load(matrix_path) as arrays:
            all_attention = arrays["key_mass"].astype(np.float64)
        all_metrics = _metrics(
            all_attention,
            token_count=token_count,
            temporal_tokens=temporal_tokens,
        )
        for category, (head, description) in CATEGORIES.items():
            matrix = all_attention[head]
            metrics = all_metrics[head]
            top_count = max(1, int(round(0.01 * bins)))
            top_mass = float(
                np.sort(matrix, axis=1)[:, -top_count:].sum(1).mean()
            )
            peak_uniform_ratio = float(matrix.max(1).mean() * bins)
            record = {
                "step": step,
                "category": category,
                "head": head,
                "description": description,
                "normalized_entropy": float(metrics[0]),
                "same_frame_mass": float(metrics[1]),
                "first_frame_mass": float(metrics[2]),
                "mean_frame_distance": float(metrics[3]),
                "aligned_cross_time_mass": float(metrics[4]),
                "aligned_cross_time_enrichment": float(metrics[5]),
                "local_same_frame_mass": float(metrics[6]),
                "local_same_frame_enrichment": float(metrics[7]),
                "past_frame_mass": float(metrics[8]),
                "future_frame_mass": float(metrics[9]),
                "history_bias": float(metrics[8] - metrics[9]),
                "top_1pct_key_mass": top_mass,
                "mean_row_peak_vs_uniform": peak_uniform_ratio,
                "effective_key_bins": float(
                    math.exp(float(metrics[0]) * math.log(bins))
                ),
            }
            records.append(record)
            samples.append(
                {
                    "step": step,
                    "category": category,
                    "head": head,
                    "matrix": matrix,
                    "temporal_matrix": _temporal_matrix(
                        matrix,
                        bin_frames=bin_frames,
                        query_counts=query_counts,
                        temporal_tokens=temporal_tokens,
                    ),
                }
            )

    assert boundaries is not None
    full_path = output_dir / "block17_category_full_heatmaps_shared_scale.png"
    low, high = _plot_full_matrices(samples, full_path, boundaries)
    temporal_path = output_dir / "block17_category_temporal_13x13_heatmaps.png"
    _plot_temporal_matrices(samples, temporal_path)

    csv_path = output_dir / "block17_category_metrics_by_step.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    aggregate = []
    for category, (head, description) in CATEGORIES.items():
        rows = [row for row in records if row["category"] == category]
        item = {
            "category": category,
            "head": head,
            "description": description,
        }
        for key in records[0]:
            if key in {"step", "category", "head", "description"}:
                continue
            item[f"{key}_mean"] = float(np.mean([row[key] for row in rows]))
            item[f"{key}_step05"] = rows[0][key]
            item[f"{key}_step35"] = rows[-1][key]
        aggregate.append(item)
    aggregate_path = output_dir / "block17_category_metrics_aggregate.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    payload = {
        "source": str(root),
        "matrix_semantics": (
            "Exact all-query/all-key softmax, pooled from 5824 tokens to "
            "512 contiguous query/key bins; key_mass rows approximately sum to one."
        ),
        "categories": CATEGORIES,
        "steps": steps,
        "shared_log10_limits": [low, high],
        "records": records,
        "aggregate": aggregate,
        "artifacts": {
            "full_heatmaps": str(full_path),
            "temporal_heatmaps": str(temporal_path),
            "metrics_by_step": str(csv_path),
            "metrics_aggregate": str(aggregate_path),
        },
    }
    (output_dir / "block17_category_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["artifacts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
