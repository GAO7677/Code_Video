#!/usr/bin/env python3
"""Freeze the exhaustive S-feature partition and its depth intersections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DOMINANCE_CLASSES = (
    ("local_dominant", "local_enrichment", "Local-enrichment dominant S"),
    ("same_frame_dominant", "same_frame_mass", "Same-frame-mass dominant S"),
)
DEPTH_STRATA = (
    ("early", 0, 10),
    ("middle", 10, 20),
    ("late", 20, 30),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def block_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(block): count
        for block, count in sorted(
            Counter(int(row["block"]) for row in rows).items()
        )
    }


def target_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "block": int(row["block"]),
            "head": int(row["head"]),
            "rank_advantage_local_minus_same": float(
                row["rank_advantage_local_minus_same"]
            ),
            "cross_model_local_rank_mean": float(
                row["cross_model_local_rank_mean"]
            ),
            "cross_model_same_frame_rank_mean": float(
                row["cross_model_same_frame_rank_mean"]
            ),
        }
        for row in sorted(rows, key=lambda item: (int(item["block"]), int(item["head"])))
    ]


def subset_record(
    rows: list[dict[str, Any]],
    *,
    dominance_class: str,
    feature_subtype: str,
    label: str,
    depth_stratum: str | None,
    block_start: int,
    block_end: int,
) -> dict[str, Any]:
    targets = target_rows(rows)
    return {
        "role": "S",
        "dominance_class": dominance_class,
        "feature_subtype": feature_subtype,
        "label": label,
        "depth_stratum": depth_stratum,
        "block_start_inclusive": block_start,
        "block_end_exclusive": block_end,
        "k": len(targets),
        "replicate": 0,
        "matching": "exhaustive_s_feature_dominance_partition",
        "block_histogram": block_histogram(targets),
        "targets": targets,
    }


def main() -> None:
    args = parse_args()
    source = args.source_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    partition = payload["full_partition"]
    rows = partition["heads"]

    keys = [(int(row["block"]), int(row["head"])) for row in rows]
    if len(keys) != 159 or len(set(keys)) != 159:
        raise RuntimeError(
            f"Expected 159 unique public-stable S heads, found {len(set(keys))}"
        )

    subsets: dict[str, dict[str, Any]] = {}
    class_keys: dict[str, set[tuple[int, int]]] = {}
    depth_keys: dict[str, set[tuple[int, int]]] = {}
    for dominance_class, feature_subtype, label in DOMINANCE_CLASSES:
        selected = [
            row for row in rows if row["feature_class"] == dominance_class
        ]
        subset_id = f"S_{dominance_class}_all"
        subsets[subset_id] = subset_record(
            selected,
            dominance_class=dominance_class,
            feature_subtype=feature_subtype,
            label=label,
            depth_stratum=None,
            block_start=0,
            block_end=30,
        )
        class_keys[dominance_class] = {
            (int(row["block"]), int(row["head"])) for row in selected
        }
        for depth, start, end in DEPTH_STRATA:
            depth_rows = [
                row for row in selected if start <= int(row["block"]) < end
            ]
            depth_subset_id = f"S_{dominance_class}_depth_{depth}"
            subsets[depth_subset_id] = subset_record(
                depth_rows,
                dominance_class=dominance_class,
                feature_subtype=feature_subtype,
                label=f"{label} / {depth}",
                depth_stratum=depth,
                block_start=start,
                block_end=end,
            )
            depth_keys[depth_subset_id] = {
                (int(row["block"]), int(row["head"])) for row in depth_rows
            }

    universe = set(keys)
    local = class_keys["local_dominant"]
    same = class_keys["same_frame_dominant"]
    if local & same or local | same != universe:
        raise RuntimeError("Dominance classes must be disjoint and exhaustive")
    for dominance_class, _, _ in DOMINANCE_CLASSES:
        parts = [
            depth_keys[f"S_{dominance_class}_depth_{depth}"]
            for depth, _, _ in DEPTH_STRATA
        ]
        if any(parts[i] & parts[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError(f"Depth subsets overlap for {dominance_class}")
        if set().union(*parts) != class_keys[dominance_class]:
            raise RuntimeError(f"Depth subsets are not exhaustive for {dominance_class}")

    result = {
        "schema_version": 1,
        "experiment": "exhaustive_s_feature_dominance_and_depth_ablation",
        "selection_policy": {
            "source": str(source),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "population": "159 heads classified as S in all three models",
            "feature_statistic": (
                "cross-model mean of per-observation 720-head normalized rank"
            ),
            "partition_rule": (
                "local_dominant iff local_enrichment_rank_mean > "
                "same_frame_mass_rank_mean; otherwise same_frame_dominant"
            ),
            "depth_strata": [
                {
                    "name": name,
                    "block_start_inclusive": start,
                    "block_end_exclusive": end,
                }
                for name, start, end in DEPTH_STRATA
            ],
            "strictly_disjoint": True,
            "exhaustive": True,
            "same_targets_for_all_models": True,
        },
        "validation": {
            "all_s_count": len(universe),
            "local_dominant_count": len(local),
            "same_frame_dominant_count": len(same),
            "class_intersection_count": len(local & same),
            "class_union_count": len(local | same),
        },
        "subsets": subsets,
    }
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "output": str(output),
                "subsets": {
                    subset_id: record["k"]
                    for subset_id, record in subsets.items()
                },
                "validation": result["validation"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
