#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from train_ridge_probe import (
    evaluate_combo,
    load_index_rows,
    safe_float,
    tensor_to_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a grid of ridge-probe experiments over different feature groups and frame reductions."
        )
    )
    parser.add_argument(
        "--index_csv",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/indices/probe_index.csv",
    )
    parser.add_argument(
        "--output_root",
        default="/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_results/grid_search",
    )
    parser.add_argument("--target_field", default="source_surprise_score")
    parser.add_argument("--group_field", default="basename")
    parser.add_argument("--max_splits", type=int, default=4)
    parser.add_argument("--alphas", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument(
        "--frame_reduces",
        default="mean,flatten",
        help="Comma-separated list of frame reduction modes to evaluate.",
    )
    parser.add_argument(
        "--feature_groups",
        default=(
            "h_post_global_mean;"
            "delta_h_global_mean;"
            "h_post_frame_mean;"
            "delta_h_frame_mean;"
            "h_post_token_l2_mean;"
            "delta_h_token_l2_mean;"
            "h_post_global_mean+delta_h_global_mean;"
            "h_post_frame_mean+delta_h_frame_mean;"
            "h_post_token_l2_mean+delta_h_token_l2_mean;"
            "h_post_global_mean+h_post_token_l2_mean;"
            "delta_h_global_mean+delta_h_token_l2_mean;"
            "h_post_global_mean+delta_h_global_mean+h_post_frame_mean+delta_h_frame_mean"
        ),
        help=(
            "Semicolon-separated feature groups. Inside each group, combine features with '+'."
        ),
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--allowed_steps",
        default="",
        help="Optional comma-separated capture step indices to keep.",
    )
    parser.add_argument(
        "--allowed_layers",
        default="",
        help="Optional comma-separated layer indices to keep.",
    )
    return parser.parse_args()


def parse_feature_groups(raw_value: str) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    for chunk in raw_value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        feature_keys = [item.strip() for item in chunk.split("+") if item.strip()]
        if not feature_keys:
            continue
        group_name = "+".join(feature_keys)
        groups.append((group_name, feature_keys))
    return groups


def parse_frame_reduces(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def extract_group_samples(
    rows: Iterable[dict[str, str]],
    *,
    feature_groups: list[tuple[str, list[str]]],
    target_field: str,
    group_field: str,
    frame_reduce: str,
    allowed_steps: set[int] | None = None,
    allowed_layers: set[int] | None = None,
) -> dict[tuple[int, int, str, str], list[dict[str, object]]]:
    combos: dict[tuple[int, int, str, str], list[dict[str, object]]] = defaultdict(list)

    for row in rows:
        target_value = safe_float(row.get(target_field))
        if target_value is None:
            continue

        feature_path = row["feature_path"]
        payload = torch.load(feature_path, map_location="cpu")
        features = payload["features"]
        meta = payload.get("meta", {})
        group_value = row.get(group_field) or meta.get(group_field) or row.get("sample_id")

        for step_key, step_payload in features.items():
            step_idx = int(step_key)
            if allowed_steps is not None and step_idx not in allowed_steps:
                continue
            cond_layers = step_payload.get("branches", {}).get("cond", {})
            for layer_key, layer_payload in cond_layers.items():
                layer_idx = int(layer_key)
                if allowed_layers is not None and layer_idx not in allowed_layers:
                    continue
                for feature_group_name, feature_keys in feature_groups:
                    vectors: list[np.ndarray] = []
                    missing_feature = False
                    for feature_key in feature_keys:
                        if feature_key not in layer_payload:
                            missing_feature = True
                            break
                        vectors.append(tensor_to_vector(layer_payload[feature_key], frame_reduce))
                    if missing_feature:
                        continue
                    combo_vector = np.concatenate(vectors, axis=0)
                    combos[(step_idx, layer_idx, frame_reduce, feature_group_name)].append(
                        {
                            "sample_id": row["sample_id"],
                            "group": group_value,
                            "target": target_value,
                            "vector": combo_vector,
                        }
                    )

    return combos


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    index_rows = load_index_rows(args.index_csv)
    feature_groups = parse_feature_groups(args.feature_groups)
    frame_reduces = parse_frame_reduces(args.frame_reduces)
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    allowed_steps = {int(item.strip()) for item in args.allowed_steps.split(",") if item.strip()} or None
    allowed_layers = {int(item.strip()) for item in args.allowed_layers.split(",") if item.strip()} or None

    result_rows: list[dict[str, object]] = []
    for frame_reduce in frame_reduces:
        combos = extract_group_samples(
            index_rows,
            feature_groups=feature_groups,
            target_field=args.target_field,
            group_field=args.group_field,
            frame_reduce=frame_reduce,
            allowed_steps=allowed_steps,
            allowed_layers=allowed_layers,
        )
        for (step_idx, layer_idx, combo_frame_reduce, feature_group_name), records in sorted(combos.items()):
            metrics = evaluate_combo(records, alphas=alphas, max_splits=args.max_splits)
            result_rows.append(
                {
                    "step_idx": step_idx,
                    "layer_idx": layer_idx,
                    "frame_reduce": combo_frame_reduce,
                    "feature_group": feature_group_name,
                    **metrics,
                }
            )

    results_csv = output_root / "probe_grid_metrics.csv"
    fieldnames = [
        "step_idx",
        "layer_idx",
        "frame_reduce",
        "feature_group",
        "status",
        "n_samples",
        "n_groups",
        "feature_dim",
        "pearson",
        "spearman",
        "r2",
        "mae",
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    ok_rows = [row for row in result_rows if row["status"] == "ok"]
    ok_rows.sort(
        key=lambda row: (
            -(row["pearson"] if row["pearson"] == row["pearson"] else float("-inf")),
            -(row["spearman"] if row["spearman"] == row["spearman"] else float("-inf")),
            row["mae"] if row["mae"] == row["mae"] else float("inf"),
        )
    )
    summary = {
        "index_csv": args.index_csv,
        "target_field": args.target_field,
        "group_field": args.group_field,
        "frame_reduces": frame_reduces,
        "feature_groups": [name for name, _ in feature_groups],
        "num_results": len(result_rows),
        "num_ok_results": len(ok_rows),
        "top_k": args.top_k,
        "top_results": ok_rows[: args.top_k],
    }
    summary_json = output_root / "probe_grid_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(results_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
