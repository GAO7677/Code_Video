#!/usr/bin/env python3
"""Classify heads from completed moving-query feature captures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze_multiblock_ball_query_heads import ROLE_LABELS, _role_scores
from moving_query_attention import FEATURE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--block", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    representatives = {}
    for model in ("wan_lora", "xssc", "physrvg"):
        summary_path = (
            root
            / f"block{args.block:02d}"
            / "matrices"
            / model
            / args.case
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        samples = []
        for entry in summary["steps"]:
            with np.load(
                summary_path.parent / entry["directory"] / entry["features_npz"]
            ) as arrays:
                samples.append({name: arrays[name] for name in FEATURE_NAMES})
        features = {
            name: np.stack([sample[name] for sample in samples]).mean(0)
            for name in FEATURE_NAMES
        }
        scores = _role_scores(features)
        roles = list(ROLE_LABELS)
        matrix = np.stack([scores[role] for role in roles], axis=1)
        order = np.argsort(matrix, axis=1)
        representatives[model] = {}
        for role in roles:
            ranking = np.argsort(scores[role])[::-1]
            representatives[model][role] = [
                int(head) for head in ranking[:3]
            ]
        role_indices, assigned_heads = linear_sum_assignment(-matrix.T)
        representatives[model]["unique"] = {
            roles[int(role_index)]: int(head)
            for role_index, head in zip(role_indices, assigned_heads)
        }
        for head in range(matrix.shape[0]):
            primary = roles[int(order[head, -1])]
            secondary = roles[int(order[head, -2])]
            row = {
                "model": model,
                "block": args.block,
                "head": head,
                "role": primary,
                "role_label": ROLE_LABELS[primary],
                "secondary_role": secondary,
                "margin": float(
                    matrix[head, order[head, -1]]
                    - matrix[head, order[head, -2]]
                ),
            }
            row.update(
                {
                    f"{role.lower()}_score": float(scores[role][head])
                    for role in roles
                }
            )
            rows.append(row)
    with (output / "head_roles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "case": args.case,
        "block": args.block,
        "role_labels": ROLE_LABELS,
        "representatives": representatives,
        "heads": rows,
    }
    (output / "head_roles.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(representatives, ensure_ascii=False))


if __name__ == "__main__":
    main()
