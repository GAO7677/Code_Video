#!/usr/bin/env python3
"""Analyze spatial/temporal Head consistency across cases, seeds, and models."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("wan_lora", "physrvg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


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
        0.5 * (sorted_values[best_split - 1] + sorted_values[best_split])
    )
    return values >= threshold, threshold


def _load_run(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    step_values = []
    for entry in summary["steps"]:
        npz_path = (
            summary_path.parent
            / entry["directory"]
            / entry["statistics_npz"]
        )
        with np.load(npz_path) as arrays:
            same = arrays["same_frame_mass"].astype(np.float64)
        if same.shape != (24, 5824):
            raise RuntimeError(f"unexpected shape {same.shape} in {npz_path}")
        step_values.append(same.mean(1))
    mean_same = np.stack(step_values).mean(0)
    spatial, threshold = _optimal_two_cluster(mean_same)
    metadata = summary["case_metadata"]
    return {
        "model": summary["model"],
        "case_key": summary["case"],
        "source_case": metadata["source_case"],
        "seed": int(metadata["seed"]),
        "groups": tuple(metadata["groups"]),
        "same_frame_mass": mean_same,
        "spatial": spatial,
        "threshold": threshold,
        "summary": str(summary_path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    return float(np.logical_and(first, second).sum() / max(int(union), 1))


def main() -> None:
    args = parse_args()
    statistics_root = args.statistics_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _load_run(path)
        for path in sorted(statistics_root.glob("*/*/summary.json"))
    ]
    counts = defaultdict(int)
    for run in runs:
        counts[run["model"]] += 1
    if not args.allow_incomplete:
        for model in MODELS:
            if counts[model] != 69:
                raise RuntimeError(
                    f"expected 69 completed {model} runs, found {counts[model]}"
                )

    run_rows: list[dict] = []
    head_rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for model in MODELS:
        model_runs = [run for run in runs if run["model"] == model]
        if not model_runs:
            continue
        seed42 = next(
            run
            for run in model_runs
            if run["source_case"] == "0613pybullet_sample_001460_w002"
            and run["seed"] == 42
        )
        for run in model_runs:
            run_rows.append(
                {
                    "model": model,
                    "case_key": run["case_key"],
                    "source_case": run["source_case"],
                    "seed": run["seed"],
                    "groups": " ".join(run["groups"]),
                    "threshold": run["threshold"],
                    "spatial_head_count": int(run["spatial"].sum()),
                    "spatial_heads": " ".join(
                        f"H{head:02d}"
                        for head in np.flatnonzero(run["spatial"])
                    ),
                    "agreement_with_reference": float(
                        np.mean(run["spatial"] == seed42["spatial"])
                    ),
                    "spatial_jaccard_with_reference": _jaccard(
                        run["spatial"], seed42["spatial"]
                    ),
                    "summary": run["summary"],
                }
            )

        group_stats = {}
        for group in ("test5", "seed_sweep"):
            group_runs = [run for run in model_runs if group in run["groups"]]
            frequencies = np.stack(
                [run["spatial"] for run in group_runs]
            ).mean(0)
            same_values = np.stack(
                [run["same_frame_mass"] for run in group_runs]
            )
            group_stats[group] = {
                "run_count": len(group_runs),
                "spatial_frequency": frequencies.tolist(),
                "consensus_spatial_heads_80pct": [
                    int(head) for head in np.flatnonzero(frequencies >= 0.8)
                ],
                "mean_head_agreement_with_reference": float(
                    np.mean(
                        [
                            np.mean(run["spatial"] == seed42["spatial"])
                            for run in group_runs
                        ]
                    )
                ),
                "mean_spatial_jaccard_with_reference": float(
                    np.mean(
                        [
                            _jaccard(run["spatial"], seed42["spatial"])
                            for run in group_runs
                        ]
                    )
                ),
            }
            for head in range(24):
                head_rows.append(
                    {
                        "model": model,
                        "group": group,
                        "head": head,
                        "run_count": len(group_runs),
                        "spatial_frequency": frequencies[head],
                        "same_frame_mass_mean": same_values[:, head].mean(),
                        "same_frame_mass_std": same_values[:, head].std(),
                        "same_frame_mass_min": same_values[:, head].min(),
                        "same_frame_mass_max": same_values[:, head].max(),
                    }
                )
        summaries[model] = {
            "completed_runs": len(model_runs),
            "reference_spatial_heads": [
                int(head) for head in np.flatnonzero(seed42["spatial"])
            ],
            "groups": group_stats,
        }

    _write_csv(output_dir / "run_classifications.csv", run_rows)
    _write_csv(output_dir / "head_generalization.csv", head_rows)

    figure, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=150)
    for row, model in enumerate(MODELS):
        for column, group in enumerate(("test5", "seed_sweep")):
            axis = axes[row, column]
            selected = [
                record
                for record in head_rows
                if record["model"] == model and record["group"] == group
            ]
            if not selected:
                axis.axis("off")
                continue
            frequency = np.asarray(
                [record["spatial_frequency"] for record in selected]
            )
            axis.bar(
                np.arange(24),
                frequency,
                color=np.where(frequency >= 0.8, "#167d8d", "#d05a3a"),
            )
            axis.axhline(0.8, color="black", linestyle="--", linewidth=1)
            axis.set_ylim(0, 1.02)
            axis.set_xticks(range(24), [f"H{i:02d}" for i in range(24)], rotation=60)
            axis.set_ylabel("spatial classification frequency")
            axis.set_title(f"{model} | {group}")
            axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Block17 spatial-Head generalization across cases and seeds",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.95))
    figure.savefig(output_dir / "spatial_head_frequency.png")
    plt.close(figure)

    payload = {
        "statistics_root": str(statistics_root),
        "run_counts": dict(counts),
        "classification": (
            "Each run independently uses the minimum-SSE two-cluster split "
            "of four-step mean exact same-frame mass over 24 Heads."
        ),
        "summaries": summaries,
    }
    (output_dir / "generalization_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
