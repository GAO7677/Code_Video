#!/usr/bin/env python3
"""Summarize STC benchmark metrics with same-model, same-seed baselines."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_bench"
)


@dataclass(frozen=True)
class Metric:
    name: str
    path: tuple[str, ...]
    direction: str


METRICS = (
    Metric("physics_iq_with_context", ("physics_iq_with_context", "score"), "higher"),
    Metric("physics_iq_without_context", ("physics_iq_without_context", "score"), "higher"),
    Metric("pmf_with_context", ("pmf_with_context", "score"), "higher"),
    Metric("pmf_without_context", ("pmf_without_context", "score"), "higher"),
    Metric("wmreward_surprise", ("wmreward", "surprise"), "lower"),
    Metric("vbench_subject_consistency", ("vbench_subject_consistency", "score"), "higher"),
    Metric("vbench_background_consistency", ("vbench_background_consistency", "score"), "higher"),
    Metric("vbench_temporal_flickering", ("vbench_temporal_flickering", "score"), "higher"),
    Metric("vbench_motion_smoothness", ("vbench_motion_smoothness", "score"), "higher"),
    Metric("vbench_dynamic_degree", ("vbench_dynamic_degree", "score"), "higher"),
    Metric("vbench_aesthetic_quality", ("vbench_aesthetic_quality", "score"), "higher"),
    Metric("vbench_imaging_quality", ("vbench_imaging_quality", "score"), "higher"),
    Metric("videophy2_sa", ("videophy2", "sa_score"), "higher"),
    Metric("videophy2_pc", ("videophy2", "pc_score"), "higher"),
    Metric("videophy2_joint_rate", ("videophy2", "joint_pass"), "higher"),
    Metric("videophy2_pc_raw", ("videophy2", "pc_raw_score"), "higher"),
    Metric("cosmos_reason1", ("cosmos_reason1", "score"), "higher"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    return parser.parse_args()


def nested_number(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return float("nan")
        value = value.get(key)
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def main() -> None:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    rows = []
    for path in sorted((batch_root / "cases").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("_stc_bench")
        if not isinstance(metadata, dict):
            continue
        denoise_range = metadata.get("denoise_step_range") or [np.nan, np.nan]
        row = {
            "entry_id": metadata["entry_id"],
            "model": metadata["model"],
            "seed": int(metadata["seed"]),
            "variant": metadata["variant"],
            "role": metadata["role"],
            "denoise_start": denoise_range[0],
            "denoise_end": denoise_range[1],
        }
        row.update({metric.name: nested_number(payload, metric.path) for metric in METRICS})
        rows.append(row)
    frame = pd.DataFrame(rows)
    results_root = batch_root / "analysis"
    results_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results_root / "per_video_metrics.csv", index=False)

    group_keys = ["model", "variant", "role", "denoise_start", "denoise_end"]
    aggregate_rows = []
    for key, group in frame.groupby(group_keys, dropna=False):
        record = dict(zip(group_keys, key))
        record["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            values = group[metric.name].dropna().to_numpy(float)
            record[f"{metric.name}_count"] = len(values)
            record[f"{metric.name}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric.name}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        aggregate_rows.append(record)
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(results_root / "aggregate_metrics.csv", index=False)

    baseline = frame[frame["variant"] == "baseline"].set_index(["model", "seed"])
    paired_rows = []
    for row in frame[frame["variant"] != "baseline"].itertuples():
        key = (row.model, row.seed)
        if key not in baseline.index:
            continue
        baseline_row = baseline.loc[key]
        record = {
            "entry_id": row.entry_id,
            "model": row.model,
            "seed": row.seed,
            "variant": row.variant,
            "role": row.role,
            "denoise_start": row.denoise_start,
            "denoise_end": row.denoise_end,
        }
        for metric in METRICS:
            value = float(getattr(row, metric.name))
            reference = float(baseline_row[metric.name])
            record[f"{metric.name}_value"] = value
            record[f"{metric.name}_baseline"] = reference
            raw_delta = value - reference
            record[f"{metric.name}_delta"] = raw_delta
            record[f"{metric.name}_improvement"] = (
                raw_delta if metric.direction == "higher" else -raw_delta
            )
        paired_rows.append(record)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(results_root / "paired_vs_baseline_per_seed.csv", index=False)

    summary_rows = []
    for key, group in paired.groupby(
        ["model", "variant", "role", "denoise_start", "denoise_end"],
        dropna=False,
    ):
        record = dict(
            zip(
                ["model", "variant", "role", "denoise_start", "denoise_end"],
                key,
            )
        )
        record["n_seeds"] = int(group["seed"].nunique())
        for metric in METRICS:
            values = group[f"{metric.name}_improvement"].dropna().to_numpy(float)
            record[f"{metric.name}_count"] = len(values)
            record[f"{metric.name}_improvement_mean"] = (
                float(values.mean()) if len(values) else np.nan
            )
            record[f"{metric.name}_improvement_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
            record[f"{metric.name}_improvement_rate"] = (
                float(np.mean(values > 0)) if len(values) else np.nan
            )
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_root / "paired_vs_baseline_summary.csv", index=False)

    coverage = {
        metric.name: int(frame[metric.name].notna().sum())
        for metric in METRICS
    }
    (results_root / "coverage.json").write_text(
        json.dumps(
            {
                "num_entries": len(frame),
                "metric_coverage": coverage,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[stc-bench-summary] entries={len(frame)} output={results_root}")


if __name__ == "__main__":
    main()
