#!/usr/bin/env python3
"""Group all Block-17 attention heads by an existing role classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_block17_category_heatmaps import _bin_frames, _temporal_matrix
from analyze_self_attention_head_roles import _metrics


ROLES = {
    "S": "intraframe spatial",
    "T": "moving-ball trajectory",
    "P": "fixed-position alignment",
    "C": "history/context",
    "G": "global aggregation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--classification-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_classification(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        if int(row["block"]) != 17:
            continue
        head = int(row["head"])
        if head in result:
            raise ValueError(f"duplicate Block-17 Head {head} in {path}")
        result[head] = row
    if set(result) != set(range(24)):
        raise ValueError(
            f"expected Block-17 Heads 0..23, found {sorted(result)}"
        )
    return result


def _save_full_montage(
    *,
    role: str,
    heads: list[int],
    samples: dict[tuple[int, int], dict],
    steps: list[int],
    boundaries: list[float],
    limits: tuple[float, float],
    classifications: dict[int, dict[str, str]],
    output_path: Path,
) -> None:
    rows, columns = len(steps), len(heads)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(max(6.5, 3.25 * columns), 12.8),
        dpi=150,
        squeeze=False,
    )
    image = None
    for row_index, step in enumerate(steps):
        for column_index, head in enumerate(heads):
            axis = axes[row_index, column_index]
            sample = samples[(step, head)]
            image = axis.imshow(
                sample["log_matrix"],
                cmap="magma",
                interpolation="nearest",
                origin="upper",
                vmin=limits[0],
                vmax=limits[1],
                aspect="equal",
            )
            for boundary in boundaries:
                axis.axhline(
                    boundary, color="white", linewidth=0.22, alpha=0.40
                )
                axis.axvline(
                    boundary, color="white", linewidth=0.22, alpha=0.40
                )
            secondary = classifications[head]["aggregate_secondary_role"]
            suffix = (
                f" / {secondary}"
                if classifications[head]["classification"].endswith("混合")
                else ""
            )
            axis.set_title(f"step {step:02d} | H{head:02d}{suffix}", fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(f"step {step}\nquery bins")
    for column_index, head in enumerate(heads):
        axes[-1, column_index].set_xlabel(f"H{head:02d} | key bins")
    figure.suptitle(
        f"Wan+LoRA Block 17 | {role}: {ROLES[role]} | "
        "all-query/all-key | globally shared log10 scale",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.045,
        right=0.95,
        top=0.93,
        bottom=0.05,
        wspace=0.06,
        hspace=0.14,
    )
    assert image is not None
    colorbar_axis = figure.add_axes((0.965, 0.13, 0.012, 0.72))
    figure.colorbar(
        image, cax=colorbar_axis, label="log10 key-bin attention mass"
    )
    figure.savefig(output_path)
    plt.close(figure)


def _save_temporal_montage(
    *,
    role: str,
    heads: list[int],
    samples: dict[tuple[int, int], dict],
    steps: list[int],
    high: float,
    classifications: dict[int, dict[str, str]],
    output_path: Path,
) -> None:
    rows, columns = len(steps), len(heads)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(max(6.5, 3.0 * columns), 11.8),
        dpi=150,
        squeeze=False,
    )
    image = None
    for row_index, step in enumerate(steps):
        for column_index, head in enumerate(heads):
            axis = axes[row_index, column_index]
            image = axis.imshow(
                samples[(step, head)]["temporal_matrix"],
                cmap="magma",
                interpolation="nearest",
                origin="upper",
                vmin=0.0,
                vmax=high,
                aspect="equal",
            )
            secondary = classifications[head]["aggregate_secondary_role"]
            suffix = (
                f" / {secondary}"
                if classifications[head]["classification"].endswith("混合")
                else ""
            )
            axis.set_title(f"step {step:02d} | H{head:02d}{suffix}", fontsize=10)
            axis.set_xticks(range(0, 13, 2))
            axis.set_yticks(range(0, 13, 2))
        axes[row_index, 0].set_ylabel("query latent t")
    for column_index in range(columns):
        axes[-1, column_index].set_xlabel("key latent t")
    figure.suptitle(
        f"Wan+LoRA Block 17 | {role}: {ROLES[role]} | "
        "13x13 latent-time mass | globally shared linear scale",
        fontsize=14,
    )
    figure.subplots_adjust(
        left=0.05,
        right=0.94,
        top=0.92,
        bottom=0.06,
        wspace=0.22,
        hspace=0.24,
    )
    assert image is not None
    colorbar_axis = figure.add_axes((0.96, 0.13, 0.012, 0.72))
    figure.colorbar(image, cax=colorbar_axis, label="attention mass")
    figure.savefig(output_path)
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    classification_path = args.classification_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    classifications = _read_classification(classification_path)
    groups: dict[str, list[int]] = defaultdict(list)
    for head, row in classifications.items():
        groups[row["aggregate_primary_role"]].append(head)
    if set(groups) != set(ROLES):
        raise ValueError(f"expected roles {list(ROLES)}, found {sorted(groups)}")

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    temporal_tokens, grid_h, grid_w = (
        int(value) for value in summary["latent_grid"]
    )
    token_count = temporal_tokens * grid_h * grid_w
    spatial_tokens = grid_h * grid_w
    steps = [int(value) for value in summary["step_numbers_one_based"]]

    raw_samples: dict[tuple[int, int], dict] = {}
    records: list[dict] = []
    all_positive: list[np.ndarray] = []
    all_temporal: list[np.ndarray] = []
    boundaries: list[float] | None = None
    for step in steps:
        entry = next(
            item
            for item in summary["steps"]
            if int(item["step_number_one_based"]) == step
        )
        metadata = entry["matrix_metadata"]
        query_counts = np.asarray(metadata["query_bin_counts"], dtype=np.float64)
        bins = int(metadata["output_bins"])
        bin_frames = _bin_frames(bins, token_count, spatial_tokens)
        boundaries = [
            frame * spatial_tokens * bins / token_count - 0.5
            for frame in range(1, temporal_tokens)
        ]
        with np.load(root / entry["directory"] / entry["matrix_npz"]) as arrays:
            all_attention = arrays["key_mass"].astype(np.float64)
        all_metrics = _metrics(
            all_attention,
            token_count=token_count,
            temporal_tokens=temporal_tokens,
        )
        all_positive.append(all_attention[all_attention > 0])
        for head in range(24):
            matrix = all_attention[head]
            temporal_matrix = _temporal_matrix(
                matrix,
                bin_frames=bin_frames,
                query_counts=query_counts,
                temporal_tokens=temporal_tokens,
            )
            all_temporal.append(temporal_matrix)
            metrics = all_metrics[head]
            top_count = max(1, int(round(0.01 * bins)))
            role = classifications[head]["aggregate_primary_role"]
            records.append(
                {
                    "step": step,
                    "head": head,
                    "primary_role": role,
                    "secondary_role": classifications[head][
                        "aggregate_secondary_role"
                    ],
                    "classification": classifications[head]["classification"],
                    "role_stability": float(
                        classifications[head]["aggregate_role_stability"]
                    ),
                    "role_margin": float(classifications[head]["role_margin"]),
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
                    "top_1pct_key_mass": float(
                        np.sort(matrix, axis=1)[:, -top_count:].sum(1).mean()
                    ),
                    "mean_row_peak_vs_uniform": float(
                        matrix.max(1).mean() * bins
                    ),
                    "effective_key_bins": float(
                        math.exp(float(metrics[0]) * math.log(bins))
                    ),
                }
            )
            raw_samples[(step, head)] = {
                "matrix": matrix,
                "temporal_matrix": temporal_matrix,
            }

    assert boundaries is not None
    minimum_positive = min(float(values.min()) for values in all_positive)
    epsilon = minimum_positive * 0.5
    log_values = np.concatenate(
        [np.log10(np.maximum(values, epsilon)) for values in all_positive]
    )
    log_limits = tuple(
        float(value) for value in np.percentile(log_values, [1.0, 99.8])
    )
    temporal_high = float(np.percentile(np.stack(all_temporal), 99.5))
    for sample in raw_samples.values():
        sample["log_matrix"] = np.log10(
            np.maximum(sample["matrix"], epsilon)
        )

    artifacts: dict[str, dict[str, str]] = {}
    for role in ROLES:
        heads = sorted(groups[role])
        full_path = output_dir / f"category_{role}_all_heads_full.png"
        temporal_path = output_dir / f"category_{role}_all_heads_temporal.png"
        _save_full_montage(
            role=role,
            heads=heads,
            samples=raw_samples,
            steps=steps,
            boundaries=boundaries,
            limits=log_limits,
            classifications=classifications,
            output_path=full_path,
        )
        _save_temporal_montage(
            role=role,
            heads=heads,
            samples=raw_samples,
            steps=steps,
            high=temporal_high,
            classifications=classifications,
            output_path=temporal_path,
        )
        artifacts[role] = {
            "full": str(full_path),
            "temporal": str(temporal_path),
        }

    by_head: list[dict] = []
    metric_keys = [
        key
        for key in records[0]
        if key
        not in {
            "step",
            "head",
            "primary_role",
            "secondary_role",
            "classification",
            "role_stability",
            "role_margin",
        }
    ]
    for head in range(24):
        rows = [row for row in records if row["head"] == head]
        item = {
            "head": head,
            "primary_role": rows[0]["primary_role"],
            "secondary_role": rows[0]["secondary_role"],
            "classification": rows[0]["classification"],
            "role_stability": rows[0]["role_stability"],
            "role_margin": rows[0]["role_margin"],
        }
        for key in metric_keys:
            item[f"{key}_mean"] = float(np.mean([row[key] for row in rows]))
            item[f"{key}_step05"] = rows[0][key]
            item[f"{key}_step35"] = rows[-1][key]
        by_head.append(item)

    by_category: list[dict] = []
    for role in ROLES:
        rows = [row for row in records if row["primary_role"] == role]
        item = {
            "primary_role": role,
            "description": ROLES[role],
            "head_count": len(groups[role]),
            "heads": " ".join(f"H{head:02d}" for head in sorted(groups[role])),
        }
        for key in metric_keys:
            item[f"{key}_mean"] = float(np.mean([row[key] for row in rows]))
            for step in (5, 35):
                step_rows = [row for row in rows if row["step"] == step]
                item[f"{key}_step{step:02d}"] = float(
                    np.mean([row[key] for row in step_rows])
                )
        by_category.append(item)

    records_path = output_dir / "metrics_by_step_and_head.csv"
    heads_path = output_dir / "metrics_by_head_aggregate.csv"
    categories_path = output_dir / "metrics_by_category_aggregate.csv"
    _write_csv(records_path, records)
    _write_csv(heads_path, by_head)
    _write_csv(categories_path, by_category)

    readme_lines = [
        "# Block17 Wan+LoRA heatmaps grouped by Head role",
        "",
        "The role labels come from exact moving-ball query attention at output "
        "frame 8. The heatmaps in this directory instead show all-query/all-key "
        "attention pooled from 5824 tokens into 512 contiguous bins.",
        "",
        "Each Head is assigned once using its aggregate primary role. Mixed "
        "Heads retain their secondary role in panel titles.",
        "",
        "| Role | Heads |",
        "|---|---|",
    ]
    for role in ROLES:
        readme_lines.append(
            f"| {role} ({ROLES[role]}) | "
            f"{', '.join(f'H{head:02d}' for head in sorted(groups[role]))} |"
        )
    readme_lines.extend(
        [
            "",
            "All full matrices share one global log10 color scale. All 13x13 "
            "temporal matrices share one global linear color scale.",
        ]
    )
    (output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8"
    )
    payload = {
        "source": str(root),
        "classification": str(classification_path),
        "groups": {role: sorted(groups[role]) for role in ROLES},
        "shared_log10_limits": log_limits,
        "shared_temporal_high": temporal_high,
        "artifacts": artifacts,
        "metrics": {
            "by_step_and_head": str(records_path),
            "by_head": str(heads_path),
            "by_category": str(categories_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
