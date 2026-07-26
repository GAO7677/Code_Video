#!/usr/bin/env python3
"""Classify heads by context-t1 object-query attention inside/outside its frame."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODELS = ("wan_lora", "xssc", "physrvg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--block", type=int, default=17)
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--query-time", type=int, default=1)
    parser.add_argument("--representatives-per-class", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    representatives = {}
    for model in MODELS:
        summary_path = (
            args.maps_root.expanduser().resolve()
            / f"block{args.block:02d}"
            / "matrices"
            / model
            / args.case
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entry = next(
            item
            for item in summary["steps"]
            if int(item["step_number_one_based"]) == args.step
        )
        with np.load(
            summary_path.parent / entry["directory"] / entry["maps_npz"]
        ) as arrays:
            attention = arrays["attention"].astype(np.float64)
            selected_heads = arrays["selected_heads"].astype(int)
        if selected_heads.tolist() != list(range(24)):
            raise ValueError(
                f"{model} must contain heads 0..23, got {selected_heads.tolist()}"
            )
        query_attention = attention[:, args.query_time]
        frame_mass = query_attention.sum(axis=(2, 3))
        total_mass = frame_mass.sum(axis=1)
        same_mass = frame_mass[:, args.query_time] / total_mass
        outside_mass = 1.0 - same_mass
        same_enrichment = same_mass * 13.0
        outside_enrichment = outside_mass * 13.0 / 12.0
        density_log2_ratio = np.log2(
            np.maximum(same_enrichment, 1.0e-12)
            / np.maximum(outside_enrichment, 1.0e-12)
        )
        past_mass = frame_mass[:, : args.query_time].sum(axis=1) / total_mass
        future_mass = frame_mass[:, args.query_time + 1 :].sum(axis=1) / total_mass
        labels = np.where(density_log2_ratio >= 0.0, "in_frame", "out_frame")
        inside_order = np.argsort(density_log2_ratio)[::-1]
        outside_order = np.argsort(density_log2_ratio)
        count = int(args.representatives_per_class)
        representatives[model] = {
            "most_in_frame": [int(head) for head in inside_order[:count]],
            "most_out_frame_leaning": [
                int(head) for head in outside_order[:count]
            ],
            "class_counts": {
                "in_frame": int((labels == "in_frame").sum()),
                "out_frame": int((labels == "out_frame").sum()),
            },
        }
        for head in range(24):
            rows.append(
                {
                    "model": model,
                    "block": args.block,
                    "denoise_step": args.step,
                    "query_time": args.query_time,
                    "head": head,
                    "class": str(labels[head]),
                    "same_frame_mass": float(same_mass[head]),
                    "outside_frame_mass": float(outside_mass[head]),
                    "same_frame_enrichment": float(same_enrichment[head]),
                    "outside_frame_enrichment": float(outside_enrichment[head]),
                    "same_vs_outside_density_log2": float(
                        density_log2_ratio[head]
                    ),
                    "past_frame_mass": float(past_mass[head]),
                    "future_frame_mass": float(future_mass[head]),
                }
            )
    csv_path = output / "t1_head_classification.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "case": args.case,
        "block": args.block,
        "denoise_step": args.step,
        "query_time": args.query_time,
        "definition": {
            "in_frame": "same-frame attention density >= outside-frame density",
            "out_frame": "same-frame attention density < outside-frame density",
            "uniform_same_frame_mass": 1.0 / 13.0,
            "score": "log2((same_mass/(1/13))/(outside_mass/(12/13)))",
        },
        "representatives": representatives,
        "heads": rows,
    }
    json_path = output / "t1_head_classification.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(representatives, ensure_ascii=False))


if __name__ == "__main__":
    main()
