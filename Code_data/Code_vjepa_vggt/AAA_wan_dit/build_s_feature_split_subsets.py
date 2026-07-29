#!/usr/bin/env python3
"""Build block-matched, disjoint local/same-frame S-head subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ("wan_lora", "xssc", "physrvg")
LOCAL_ID = "S_local_k32_r00_exactblock"
SAME_ID = "S_same_k32_r00_exactblock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=32)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def block_histogram(targets: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(block): count
        for block, count in sorted(Counter(row["block"] for row in targets).items())
    }


def main() -> None:
    args = parse_args()
    feature_path = args.features.expanduser().resolve()
    output = args.output.expanduser().resolve()
    with feature_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_head: dict[tuple[int, int], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        key = (int(row["block"]), int(row["head"]))
        model = row["model"]
        if model not in MODELS or model in by_head[key]:
            raise RuntimeError(f"invalid feature row for {key}/{model}")
        by_head[key][model] = {
            "local_enrichment_raw_mean": float(row["local_enrichment_raw_mean"]),
            "local_enrichment_rank_mean": float(row["local_enrichment_rank_mean"]),
            "same_frame_mass_raw_mean": float(row["same_frame_mass_raw_mean"]),
            "same_frame_mass_rank_mean": float(row["same_frame_mass_rank_mean"]),
        }
    if len(by_head) != 159 or any(set(value) != set(MODELS) for value in by_head.values()):
        raise RuntimeError("expected 159 common S heads with all three models")

    partition = []
    for (block, head), model_values in sorted(by_head.items()):
        local_rank = float(
            np.mean(
                [model_values[model]["local_enrichment_rank_mean"] for model in MODELS]
            )
        )
        same_rank = float(
            np.mean(
                [model_values[model]["same_frame_mass_rank_mean"] for model in MODELS]
            )
        )
        delta = local_rank - same_rank
        partition.append(
            {
                "block": block,
                "head": head,
                "feature_class": "local_dominant" if delta > 0 else "same_frame_dominant",
                "cross_model_local_rank_mean": local_rank,
                "cross_model_same_frame_rank_mean": same_rank,
                "rank_advantage_local_minus_same": delta,
                "model_statistics": model_values,
            }
        )

    local_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    same_by_block: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in partition:
        target = (
            local_by_block
            if row["feature_class"] == "local_dominant"
            else same_by_block
        )
        target[row["block"]].append(row)
    for values in local_by_block.values():
        values.sort(
            key=lambda row: (
                -row["rank_advantage_local_minus_same"],
                row["head"],
            )
        )
    for values in same_by_block.values():
        values.sort(
            key=lambda row: (
                row["rank_advantage_local_minus_same"],
                row["head"],
            )
        )

    candidate_pairs = []
    for block in range(30):
        for local, same in zip(local_by_block[block], same_by_block[block]):
            candidate_pairs.append(
                {
                    "block": block,
                    "local": local,
                    "same": same,
                    "contrast": (
                        local["rank_advantage_local_minus_same"]
                        - same["rank_advantage_local_minus_same"]
                    ),
                }
            )
    candidate_pairs.sort(
        key=lambda pair: (
            -pair["contrast"],
            pair["block"],
            pair["local"]["head"],
            pair["same"]["head"],
        )
    )
    if len(candidate_pairs) < args.k:
        raise RuntimeError(
            f"only {len(candidate_pairs)} exact-block pairs are available for k={args.k}"
        )
    selected_pairs = candidate_pairs[: args.k]
    local_targets = sorted(
        [pair["local"] for pair in selected_pairs],
        key=lambda row: (row["block"], row["head"]),
    )
    same_targets = sorted(
        [pair["same"] for pair in selected_pairs],
        key=lambda row: (row["block"], row["head"]),
    )
    local_keys = {(row["block"], row["head"]) for row in local_targets}
    same_keys = {(row["block"], row["head"]) for row in same_targets}
    if len(local_keys) != args.k or len(same_keys) != args.k:
        raise RuntimeError("selected subsets do not contain k unique heads")
    if local_keys & same_keys:
        raise RuntimeError(f"selected subsets overlap: {sorted(local_keys & same_keys)}")
    local_histogram = block_histogram(local_targets)
    same_histogram = block_histogram(same_targets)
    if local_histogram != same_histogram:
        raise RuntimeError("selected subsets have different block histograms")

    matching = "exact_block_feature_contrast"
    payload = {
        "schema_version": 1,
        "experiment": "s_local_vs_same_frame_feature_split",
        "selection_policy": {
            "source": str(feature_path),
            "source_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            "base_population": "159 heads classified as S in all three models",
            "feature_statistic": (
                "cross-model mean of per-observation 720-head mean-rank"
            ),
            "partition_rule": (
                "local_dominant iff mean_rank(local_enrichment) > "
                "mean_rank(same_frame_mass); otherwise same_frame_dominant"
            ),
            "pairing_rule": (
                "pair opposite feature classes within the same block, rank pairs "
                "by local-minus-same contrast, retain the strongest k pairs"
            ),
            "k_per_ablation_subset": args.k,
            "strictly_disjoint": True,
            "identical_block_histogram": True,
        },
        "full_partition": {
            "local_dominant_count": len(
                [row for row in partition if row["feature_class"] == "local_dominant"]
            ),
            "same_frame_dominant_count": len(
                [
                    row
                    for row in partition
                    if row["feature_class"] == "same_frame_dominant"
                ]
            ),
            "heads": partition,
        },
        "selected_pair_count": len(selected_pairs),
        "available_exact_block_pair_count": len(candidate_pairs),
        "selected_pairs": [
            {
                "block": pair["block"],
                "local_head": pair["local"]["head"],
                "same_frame_head": pair["same"]["head"],
                "rank_contrast": pair["contrast"],
            }
            for pair in selected_pairs
        ],
        "subsets": {
            LOCAL_ID: {
                "role": "S",
                "feature_subtype": "local_enrichment",
                "k": args.k,
                "replicate": 0,
                "matching": matching,
                "block_histogram": local_histogram,
                "targets": local_targets,
            },
            SAME_ID: {
                "role": "S",
                "feature_subtype": "same_frame_mass",
                "k": args.k,
                "replicate": 0,
                "matching": matching,
                "block_histogram": same_histogram,
                "targets": same_targets,
            },
        },
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "full_partition": {
                    "local": payload["full_partition"]["local_dominant_count"],
                    "same": payload["full_partition"]["same_frame_dominant_count"],
                },
                "selected_per_group": args.k,
                "intersection": len(local_keys & same_keys),
                "block_histogram": local_histogram,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
