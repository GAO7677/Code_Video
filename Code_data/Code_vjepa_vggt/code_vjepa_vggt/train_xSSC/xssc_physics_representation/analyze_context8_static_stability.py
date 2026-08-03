#!/usr/bin/env python3
"""Compare frame-0 and first-8-frame medoid references for decoder-static features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_REPORT = Path(
    "/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1/report"
)
METRICS = ("l1", "l2", "cosine")
ROLES = ("ball", "block")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_change_percent(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 100.0 * (right - left) / np.maximum(left, 1.0e-12)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.expanduser().resolve()
    output_dir = report_dir / "context8_static_stability_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(
        (report_dir / "context8_reference_metrics.json").read_text(encoding="utf-8")
    )
    records = metadata["records"]
    arrays = np.load(report_dir / "context8_reference_metrics.npz")
    models = list(dict.fromkeys(record["model"] for record in records))
    per_case = []
    for index, record in enumerate(records):
        if record["representation"] != "decoder_static":
            continue
        row: dict[str, Any] = {
            "curve_index": index,
            "model": record["model"],
            "case_id": record["case_id"],
            "family": record["family"],
            "role": record["role"],
            "quality_pass": bool(record["quality_pass"]),
            "context_recall": float(record["context_recall"]),
            "context_medoid_frame": int(record["context_medoid_frame"]),
        }
        for metric in METRICS:
            frame0 = arrays[f"frame0_{metric}"][index, metadata["comparison_start_frame"] :]
            context8 = arrays[f"context8_{metric}"][index, metadata["comparison_start_frame"] :]
            frame0_mean = float(np.mean(frame0))
            context8_mean = float(np.mean(context8))
            frame0_std = float(np.std(frame0))
            context8_std = float(np.std(context8))
            row.update(
                {
                    f"frame0_{metric}_mean": frame0_mean,
                    f"context8_{metric}_mean": context8_mean,
                    f"{metric}_mean_change_percent": float(
                        paired_change_percent(np.array(frame0_mean), np.array(context8_mean))
                    ),
                    f"frame0_{metric}_curve_std": frame0_std,
                    f"context8_{metric}_curve_std": context8_std,
                    f"{metric}_curve_std_change_percent": float(
                        paired_change_percent(np.array(frame0_std), np.array(context8_std))
                    ),
                }
            )
        per_case.append(row)

    aggregate = []
    for scope in ("physics", "all"):
        for model in models:
            for role in ROLES:
                selected = [
                    row for row in per_case
                    if row["model"] == model and row["role"] == role
                    and row["quality_pass"]
                    and (scope == "all" or row["family"] == "physics")
                ]
                if not selected:
                    continue
                row = {"scope": scope, "model": model, "role": role, "clean_case_count": len(selected)}
                for metric in METRICS:
                    frame0_mean = np.array([item[f"frame0_{metric}_mean"] for item in selected])
                    context8_mean = np.array([item[f"context8_{metric}_mean"] for item in selected])
                    frame0_std = np.array([item[f"frame0_{metric}_curve_std"] for item in selected])
                    context8_std = np.array([item[f"context8_{metric}_curve_std"] for item in selected])
                    row.update(
                        {
                            f"frame0_{metric}_mean_median": float(np.median(frame0_mean)),
                            f"context8_{metric}_mean_median": float(np.median(context8_mean)),
                            f"{metric}_paired_mean_change_percent_median": float(
                                np.median(paired_change_percent(frame0_mean, context8_mean))
                            ),
                            f"{metric}_lower_mean_count": int(np.sum(context8_mean < frame0_mean)),
                            f"frame0_{metric}_curve_std_median": float(np.median(frame0_std)),
                            f"context8_{metric}_curve_std_median": float(np.median(context8_std)),
                            f"{metric}_paired_curve_std_change_percent_median": float(
                                np.median(paired_change_percent(frame0_std, context8_std))
                            ),
                            f"{metric}_lower_curve_std_count": int(np.sum(context8_std < frame0_std)),
                        }
                    )
                aggregate.append(row)

    write_csv(output_dir / "per_case.csv", per_case)
    write_csv(output_dir / "aggregate.csv", aggregate)
    summary = {
        "definition": {
            "mean_distance": "mean D(static(t), reference) over frames 8..149; lower means the reference is closer to the future trajectory",
            "distance_curve_std": "temporal standard deviation of D(static(t), reference) over frames 8..149; lower means distance to that reference fluctuates less",
            "caveat": "Changing a fixed reference does not change adjacent-frame D(static(t), static(t-1)); these statistics measure reference representativeness, not intrinsic feature smoothness",
            "inference": "descriptive only; the eight physics configurations are not independent simulation seeds",
        },
        "per_case_count": len(per_case),
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[complete] per_case={len(per_case)} aggregate={len(aggregate)} output={output_dir}")


if __name__ == "__main__":
    main()
