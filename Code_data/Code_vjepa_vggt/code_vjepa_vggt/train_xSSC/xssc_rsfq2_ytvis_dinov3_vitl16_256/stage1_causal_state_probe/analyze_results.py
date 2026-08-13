#!/usr/bin/env python3
"""Case-balanced paired contrasts for the completed Stage-1 matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage1_causal_state_probe.io_utils import atomic_write_json  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--metric", default="velocity_nrmse")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--practical-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_runs(root):
    runs = []
    for summary_file in sorted(Path(root).rglob("summary.json")):
        summary = json.loads(summary_file.read_text())
        if summary.get("format") != "xssc_stage1_evaluation_v1":
            continue
        cases_file = summary_file.with_name("cases.csv")
        if not cases_file.is_file():
            raise FileNotFoundError(cases_file)
        with cases_file.open() as stream:
            rows = list(csv.DictReader(stream))
        runs.append({"summary": summary, "rows": rows, "path": str(summary_file)})
    if not runs:
        raise FileNotFoundError(f"No Stage-1 evaluation results below {root}")
    return runs


def values_by_video(run, metric, horizon):
    return {
        int(row["video_index"]): float(row[metric])
        for row in run["rows"]
        if int(row["horizon"]) == horizon and row.get(metric, "") != ""
    }


def average_seed_values(indexed, keys, metric, horizon):
    groups = [values_by_video(indexed[key], metric, horizon) for key in keys]
    common = set(groups[0])
    for group in groups[1:]:
        common.intersection_update(group)
    return {
        video: float(np.mean([group[video] for group in groups]))
        for video in sorted(common)
    }


def paired_contrast(left, right, samples, rng):
    common = sorted(set(left).intersection(right))
    if not common:
        raise ValueError("Paired contrast has no common videos")
    differences = np.asarray([left[index] - right[index] for index in common])
    boot = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(differences), size=(count, len(differences)))
        boot[start : start + count] = differences[indices].mean(axis=1)
    probability_nonpositive = (np.count_nonzero(boot <= 0) + 1) / (samples + 1)
    probability_nonnegative = (np.count_nonzero(boot >= 0) + 1) / (samples + 1)
    left_mean = float(np.mean([left[index] for index in common]))
    right_mean = float(np.mean([right[index] for index in common]))
    return {
        "videos": len(common),
        "left_mean": left_mean,
        "right_mean": right_mean,
        "difference": float(differences.mean()),
        "relative_improvement": (right_mean - left_mean) / max(abs(right_mean), 1e-12),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_two_sided_bootstrap": float(
            min(1.0, 2 * min(probability_nonpositive, probability_nonnegative))
        ),
    }


def holm_adjust(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_two_sided_bootstrap"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index]["p_two_sided_bootstrap"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def run_key(summary):
    return (
        summary["representation"],
        int(summary["history"]),
        summary["context"],
        int(summary["seed"]),
    )


def write_csv(rows, path):
    if not rows:
        return
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    runs = load_runs(args.results_root.resolve())
    indexed = {run_key(run["summary"]): run for run in runs}
    rng = np.random.default_rng(args.seed)
    histories = []
    contexts = []
    horizons = sorted(
        set(horizon for run in runs for horizon in run["summary"]["horizons"])
    )

    for representation in sorted({key[0] for key in indexed}):
        for context in ("individual", "set"):
            for seed in sorted({key[3] for key in indexed}):
                for horizon in horizons:
                    family = []
                    baseline_key = (representation, 1, context, seed)
                    if baseline_key not in indexed:
                        continue
                    baseline = values_by_video(indexed[baseline_key], args.metric, horizon)
                    for history in (2, 4):
                        candidate_key = (representation, history, context, seed)
                        if candidate_key not in indexed:
                            continue
                        candidate = values_by_video(
                            indexed[candidate_key], args.metric, horizon
                        )
                        row = {
                            "contrast_type": "history",
                            "representation": representation,
                            "context": context,
                            "seed": seed,
                            "horizon": horizon,
                            "contrast": f"H{history}-H1",
                            # Error metrics are lower-is-better: negative favors left.
                            **paired_contrast(
                                candidate, baseline, args.bootstrap_samples, rng
                            ),
                        }
                        row["supports_left_at_threshold"] = (
                            row["relative_improvement"] >= args.practical_threshold
                            and row["ci95_high"] < 0
                        )
                        row["improvement_below_margin"] = (
                            row["ci95_low"]
                            > -args.practical_threshold * abs(row["right_mean"])
                        )
                        family.append(row)
                    holm_adjust(family)
                    histories.extend(family)

    # Formal seed-mean contrasts average each video's metric across seeds before
    # videos receive equal weight in the paired bootstrap.
    all_seeds = sorted({key[3] for key in indexed})
    for representation in sorted({key[0] for key in indexed}):
        for context in ("individual", "set"):
            for horizon in horizons:
                baseline_keys = [
                    (representation, 1, context, seed)
                    for seed in all_seeds
                    if (representation, 1, context, seed) in indexed
                ]
                if not baseline_keys:
                    continue
                family = []
                baseline = average_seed_values(
                    indexed, baseline_keys, args.metric, horizon
                )
                for history in (2, 4):
                    candidate_keys = [
                        (representation, history, context, seed)
                        for seed in all_seeds
                        if (representation, history, context, seed) in indexed
                    ]
                    common_seeds = sorted(
                        set(key[3] for key in baseline_keys).intersection(
                            key[3] for key in candidate_keys
                        )
                    )
                    if not common_seeds:
                        continue
                    baseline = average_seed_values(
                        indexed,
                        [(representation, 1, context, seed) for seed in common_seeds],
                        args.metric,
                        horizon,
                    )
                    candidate = average_seed_values(
                        indexed,
                        [
                            (representation, history, context, seed)
                            for seed in common_seeds
                        ],
                        args.metric,
                        horizon,
                    )
                    row = {
                        "contrast_type": "history_seed_mean",
                        "representation": representation,
                        "context": context,
                        "seed": "mean",
                        "seed_count": len(common_seeds),
                        "horizon": horizon,
                        "contrast": f"H{history}-H1",
                        **paired_contrast(
                            candidate, baseline, args.bootstrap_samples, rng
                        ),
                    }
                    row["supports_left_at_threshold"] = (
                        row["relative_improvement"] >= args.practical_threshold
                        and row["ci95_high"] < 0
                    )
                    row["improvement_below_margin"] = (
                        row["ci95_low"]
                        > -args.practical_threshold * abs(row["right_mean"])
                    )
                    family.append(row)
                holm_adjust(family)
                histories.extend(family)

    for representation in sorted({key[0] for key in indexed}):
        for history in (1, 2, 4):
            for seed in sorted({key[3] for key in indexed}):
                individual_key = (representation, history, "individual", seed)
                set_key = (representation, history, "set", seed)
                if individual_key not in indexed or set_key not in indexed:
                    continue
                for horizon in horizons:
                    row = {
                        "contrast_type": "context",
                        "representation": representation,
                        "history": history,
                        "seed": seed,
                        "horizon": horizon,
                        "contrast": "set-individual",
                        **paired_contrast(
                            values_by_video(indexed[set_key], args.metric, horizon),
                            values_by_video(
                                indexed[individual_key], args.metric, horizon
                            ),
                            args.bootstrap_samples,
                            rng,
                        ),
                    }
                    row["supports_left_at_threshold"] = (
                        row["relative_improvement"] >= args.practical_threshold
                        and row["ci95_high"] < 0
                    )
                    contexts.append(row)

    for representation in sorted({key[0] for key in indexed}):
        for history in (1, 2, 4):
            for horizon in horizons:
                common_seeds = [
                    seed
                    for seed in all_seeds
                    if (representation, history, "individual", seed) in indexed
                    and (representation, history, "set", seed) in indexed
                ]
                if not common_seeds:
                    continue
                individual = average_seed_values(
                    indexed,
                    [
                        (representation, history, "individual", seed)
                        for seed in common_seeds
                    ],
                    args.metric,
                    horizon,
                )
                context = average_seed_values(
                    indexed,
                    [
                        (representation, history, "set", seed)
                        for seed in common_seeds
                    ],
                    args.metric,
                    horizon,
                )
                row = {
                    "contrast_type": "context_seed_mean",
                    "representation": representation,
                    "history": history,
                    "seed": "mean",
                    "seed_count": len(common_seeds),
                    "horizon": horizon,
                    "contrast": "set-individual",
                    **paired_contrast(
                        context, individual, args.bootstrap_samples, rng
                    ),
                }
                row["supports_left_at_threshold"] = (
                    row["relative_improvement"] >= args.practical_threshold
                    and row["ci95_high"] < 0
                )
                contexts.append(row)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "xssc_stage1_paired_analysis_v1",
        "metric": args.metric,
        "bootstrap_samples": args.bootstrap_samples,
        "practical_threshold": args.practical_threshold,
        "history_contrasts": histories,
        "context_contrasts": contexts,
    }
    atomic_write_json(payload, output_dir / "paired_contrasts.json")
    write_csv(histories, output_dir / "history_contrasts.csv")
    write_csv(contexts, output_dir / "context_contrasts.csv")
    print(json.dumps({"history_rows": len(histories), "context_rows": len(contexts)}, indent=2))


if __name__ == "__main__":
    main()
