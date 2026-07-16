#!/usr/bin/env python3
"""Compare normalized validation convergence and overfitting signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_records(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def first_step_at_or_below(records, threshold):
    for record in records:
        if record["normalized_val_loss"] <= threshold:
            return record["global_step"]
    return None


def summarize(mode, records):
    initial = records[0]["val_loss"]
    normalized = []
    for record in records:
        item = dict(record)
        item["mode"] = mode
        item["normalized_val_loss"] = record["val_loss"] / initial
        normalized.append(item)
    auc = float("nan")
    if len(normalized) > 1:
        area = 0.0
        span = normalized[-1]["global_step"] - normalized[0]["global_step"]
        for left, right in zip(normalized, normalized[1:]):
            width = right["global_step"] - left["global_step"]
            area += width * (left["normalized_val_loss"] + right["normalized_val_loss"]) / 2
        auc = area / span if span else float("nan")
    warning_steps = [r["global_step"] for r in normalized if r.get("overfit_warning")]
    summary = {
        "mode": mode,
        "num_validations": len(normalized),
        "initial_val_loss": initial,
        "best_val_loss": min(record["val_loss"] for record in normalized),
        "best_normalized_val_loss": min(r["normalized_val_loss"] for r in normalized),
        "normalized_val_auc": auc,
        "step_to_10pct_improvement": first_step_at_or_below(normalized, 0.90),
        "step_to_20pct_improvement": first_step_at_or_below(normalized, 0.80),
        "step_to_30pct_improvement": first_step_at_or_below(normalized, 0.70),
        "first_overfit_warning_step": warning_steps[0] if warning_steps else None,
    }
    return normalized, summary


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    summaries = []
    for mode in ("pybullet", "kubric", "mixed"):
        records = load_records(args.run_root / mode / "metrics/step_metrics.jsonl")
        if not records:
            continue
        normalized, summary = summarize(mode, records)
        all_records.extend(normalized)
        summaries.append(summary)
    write_csv(args.output_dir / "normalized_convergence.csv", all_records)
    write_csv(args.output_dir / "convergence_summary.csv", summaries)
    (args.output_dir / "convergence_summary.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
