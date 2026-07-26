#!/usr/bin/env python3
"""Classify Block-17 heads using only same-frame versus cross-frame mass."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _bin_frame_overlap(
    counts: np.ndarray, *, frames: int, spatial_tokens: int
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ends = starts + counts
    overlap = np.zeros((len(counts), frames), dtype=np.float64)
    for index, (start, end) in enumerate(zip(starts, ends)):
        for frame in range(frames):
            frame_start = frame * spatial_tokens
            frame_end = frame_start + spatial_tokens
            overlap[index, frame] = max(
                0, min(int(end), frame_end) - max(int(start), frame_start)
            )
    overlap /= counts[:, None]
    pure_frame = np.full(len(counts), -1, dtype=np.int64)
    pure = np.isclose(overlap.max(axis=1), 1.0)
    pure_frame[pure] = overlap[pure].argmax(axis=1)
    return overlap, pure_frame


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    cutoff = quantile * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, cutoff)])


def _optimal_two_cluster(values: np.ndarray) -> tuple[np.ndarray, float]:
    order = np.argsort(values)
    sorted_values = values[order]
    best_sse = float("inf")
    best_split = -1
    for split in range(1, len(values)):
        low = sorted_values[:split]
        high = sorted_values[split:]
        sse = float(
            ((low - low.mean()) ** 2).sum()
            + ((high - high.mean()) ** 2).sum()
        )
        if sse < best_sse:
            best_sse = sse
            best_split = split
    threshold = float(
        0.5
        * (
            sorted_values[best_split - 1]
            + sorted_values[best_split]
        )
    )
    labels = np.where(values >= threshold, "spatial", "temporal")
    return labels, threshold


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_query_maps(
    arrays: list[np.ndarray],
    *,
    steps: list[int],
    pure_frame: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(16, 10), dpi=150)
    image = None
    display = np.stack(arrays)
    high = float(np.percentile(display[:, :, pure_frame >= 0], 99.5))
    for axis, step, values in zip(axes, steps, display):
        shown = values.copy()
        shown[:, pure_frame < 0] = np.nan
        image = axis.imshow(
            shown,
            cmap="viridis",
            interpolation="nearest",
            aspect="auto",
            vmin=0.0,
            vmax=high,
        )
        axis.set_title(f"denoising step {step:02d}")
        axis.set_ylabel("Head")
        axis.set_yticks(range(0, 24, 3))
    axes[-1].set_xlabel(
        "pooled query bin in time-major order; white bins cross frame boundaries"
    )
    figure.suptitle(
        "Per-query-bin same-frame attention mass | Wan+LoRA Block 17",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.06, right=0.92, top=0.92, bottom=0.07, hspace=0.34
    )
    assert image is not None
    colorbar_axis = figure.add_axes((0.94, 0.14, 0.012, 0.70))
    figure.colorbar(image, cax=colorbar_axis, label="same-frame mass")
    figure.savefig(output_path)
    plt.close(figure)


def _plot_head_summary(
    records: list[dict], *, threshold: float, output_path: Path
) -> None:
    ordered = sorted(records, key=lambda row: row["same_frame_mass_mean"])
    labels = [f"H{int(row['head']):02d}" for row in ordered]
    same = np.asarray([row["same_frame_mass_mean"] for row in ordered])
    cross = 1.0 - same
    colors = [
        "#167d8d" if row["classification"] == "spatial" else "#d05a3a"
        for row in ordered
    ]
    figure, axes = plt.subplots(
        2, 1, figsize=(14, 8), dpi=150, height_ratios=(1.1, 1.0)
    )
    x = np.arange(len(ordered))
    axes[0].bar(x, same, color=colors)
    axes[0].axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"two-cluster threshold {threshold:.3f}",
    )
    axes[0].set_ylabel("mean same-frame mass")
    axes[0].set_xticks(x, labels)
    axes[0].legend(loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(x, same, color="#167d8d", label="same frame")
    axes[1].bar(
        x, cross, bottom=same, color="#d05a3a", label="different frames"
    )
    axes[1].set_ylabel("attention mass")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="upper left", ncols=2)
    figure.suptitle(
        "Block-17 spatial/temporal classification from same-frame mass",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    frames, grid_h, grid_w = (
        int(value) for value in summary["latent_grid"]
    )
    spatial_tokens = grid_h * grid_w
    steps = [int(value) for value in summary["step_numbers_one_based"]]

    step_rows: list[dict] = []
    query_maps: list[np.ndarray] = []
    pure_frame_reference: np.ndarray | None = None
    for step in steps:
        entry = next(
            item
            for item in summary["steps"]
            if int(item["step_number_one_based"]) == step
        )
        counts = np.asarray(
            entry["matrix_metadata"]["query_bin_counts"], dtype=np.float64
        )
        key_overlap, pure_frame = _bin_frame_overlap(
            counts.astype(np.int64),
            frames=frames,
            spatial_tokens=spatial_tokens,
        )
        if pure_frame_reference is None:
            pure_frame_reference = pure_frame
        elif not np.array_equal(pure_frame_reference, pure_frame):
            raise RuntimeError("query binning differs between denoising steps")
        with np.load(root / entry["directory"] / entry["matrix_npz"]) as arrays:
            attention = arrays["key_mass"].astype(np.float64)
        same_mass = np.full((attention.shape[0], attention.shape[1]), np.nan)
        valid_queries = np.flatnonzero(pure_frame >= 0)
        for query_bin in valid_queries:
            frame = pure_frame[query_bin]
            same_mass[:, query_bin] = (
                attention[:, query_bin, :] @ key_overlap[:, frame]
            )
        query_maps.append(same_mass)
        valid_weights = counts[valid_queries]
        for head in range(attention.shape[0]):
            values = same_mass[head, valid_queries]
            mean_same = float(np.average(values, weights=valid_weights))
            mean_cross = 1.0 - mean_same
            same_density = mean_same * frames
            cross_density = mean_cross * frames / (frames - 1)
            step_rows.append(
                {
                    "step": step,
                    "head": head,
                    "valid_query_bins": len(valid_queries),
                    "represented_query_tokens": int(valid_weights.sum()),
                    "same_frame_mass_mean": mean_same,
                    "different_frame_mass_mean": mean_cross,
                    "same_frame_mass_q25": _weighted_quantile(
                        values, valid_weights, 0.25
                    ),
                    "same_frame_mass_median": _weighted_quantile(
                        values, valid_weights, 0.50
                    ),
                    "same_frame_mass_q75": _weighted_quantile(
                        values, valid_weights, 0.75
                    ),
                    "same_frame_majority_query_fraction": float(
                        valid_weights[values > 0.5].sum()
                        / valid_weights.sum()
                    ),
                    "same_frame_density_enrichment": same_density,
                    "different_frame_density_enrichment": cross_density,
                    "same_vs_different_density_ratio": float(
                        same_density / max(cross_density, 1.0e-30)
                    ),
                }
            )

    head_records: list[dict] = []
    aggregate_same = np.zeros(24, dtype=np.float64)
    for head in range(24):
        rows = [row for row in step_rows if row["head"] == head]
        aggregate_same[head] = np.mean(
            [row["same_frame_mass_mean"] for row in rows]
        )
    labels, threshold = _optimal_two_cluster(aggregate_same)
    for head in range(24):
        rows = [row for row in step_rows if row["head"] == head]
        step_same = np.asarray(
            [row["same_frame_mass_mean"] for row in rows]
        )
        step_labels = np.where(step_same >= threshold, "spatial", "temporal")
        mean_same = float(step_same.mean())
        mean_cross = 1.0 - mean_same
        same_density = mean_same * frames
        cross_density = mean_cross * frames / (frames - 1)
        head_records.append(
            {
                "head": head,
                "classification": labels[head],
                "same_frame_mass_mean": mean_same,
                "different_frame_mass_mean": mean_cross,
                "same_frame_mass_step05": step_same[0],
                "same_frame_mass_step15": step_same[1],
                "same_frame_mass_step25": step_same[2],
                "same_frame_mass_step35": step_same[3],
                "same_frame_majority_query_fraction_mean": float(
                    np.mean(
                        [
                            row["same_frame_majority_query_fraction"]
                            for row in rows
                        ]
                    )
                ),
                "same_frame_density_enrichment": same_density,
                "different_frame_density_enrichment": cross_density,
                "same_vs_different_density_ratio": float(
                    same_density / max(cross_density, 1.0e-30)
                ),
                "step_consistency": float(np.mean(step_labels == labels[head])),
            }
        )

    step_path = output_dir / "per_step_query_bin_statistics.csv"
    heads_path = output_dir / "head_spatiotemporal_classification.csv"
    _write_csv(step_path, step_rows)
    _write_csv(heads_path, head_records)
    query_npz = output_dir / "same_frame_mass_per_query_bin.npz"
    assert pure_frame_reference is not None
    np.savez_compressed(
        query_npz,
        same_frame_mass=np.stack(query_maps).astype(np.float32),
        steps=np.asarray(steps, dtype=np.int64),
        query_frame=pure_frame_reference,
    )
    query_plot = output_dir / "same_frame_mass_per_query_bin.png"
    summary_plot = output_dir / "head_same_vs_different_frame_summary.png"
    _plot_query_maps(
        query_maps,
        steps=steps,
        pure_frame=pure_frame_reference,
        output_path=query_plot,
    )
    _plot_head_summary(
        head_records, threshold=threshold, output_path=summary_plot
    )

    spatial_heads = [
        int(row["head"])
        for row in head_records
        if row["classification"] == "spatial"
    ]
    temporal_heads = [
        int(row["head"])
        for row in head_records
        if row["classification"] == "temporal"
    ]
    valid_bins = int(np.sum(pure_frame_reference >= 0))
    excluded_bins = int(np.sum(pure_frame_reference < 0))
    report = f"""# Block17 spatial/temporal Head classification

Case: `{summary["case"]}`. Model: `{summary["model"]}`.

This classification uses only same-frame versus different-frame attention
mass for every retained pooled query bin. The original 5824 query tokens were
stored as 512 contiguous query bins; {valid_bins} bins lie wholly inside one
latent frame and {excluded_bins} frame-boundary bins are excluded.

The binary split is the minimum-SSE two-cluster partition of each Head's mean
same-frame mass over denoising steps 5/15/25/35. The resulting threshold is
`{threshold:.6f}`. This is a relative Block17 classification, not an absolute
claim that a Head sends more than 50% of its mass to the same frame.

| Class | Heads |
|---|---|
| Spatial | {", ".join(f"H{head:02d}" for head in spatial_heads)} |
| Temporal | {", ".join(f"H{head:02d}" for head in temporal_heads)} |

See `head_spatiotemporal_classification.csv` for raw same/different-frame
mass, density enrichment, and denoising-step consistency.
"""
    report_path = output_dir / "README.md"
    report_path.write_text(report, encoding="utf-8")
    payload = {
        "source": str(root),
        "classification_basis": (
            "two-cluster split of four-step mean same-frame attention mass"
        ),
        "threshold": threshold,
        "spatial_heads": spatial_heads,
        "temporal_heads": temporal_heads,
        "valid_query_bins": valid_bins,
        "excluded_boundary_query_bins": excluded_bins,
        "artifacts": {
            "report": str(report_path),
            "head_csv": str(heads_path),
            "per_step_csv": str(step_path),
            "per_query_npz": str(query_npz),
            "per_query_plot": str(query_plot),
            "summary_plot": str(summary_plot),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
