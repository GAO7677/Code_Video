#!/usr/bin/env python3
"""Merge the frozen matched-S and dominance/depth subsets used by the 34-config run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MATCHED_IDS = (
    "S_local_k32_r00_exactblock",
    "S_same_k32_r00_exactblock",
    "S_local_same_union_k64_r00_exactblock",
)
DOMINANCE_IDS = (
    "S_local_dominant_all",
    "S_local_dominant_depth_early",
    "S_local_dominant_depth_middle",
    "S_local_dominant_depth_late",
    "S_same_frame_dominant_all",
    "S_same_frame_dominant_depth_early",
    "S_same_frame_dominant_depth_middle",
    "S_same_frame_dominant_depth_late",
)
EXPECTED_COUNTS = {
    "S_local_k32_r00_exactblock": 32,
    "S_same_k32_r00_exactblock": 32,
    "S_local_same_union_k64_r00_exactblock": 64,
    "S_local_dominant_all": 100,
    "S_local_dominant_depth_early": 34,
    "S_local_dominant_depth_middle": 25,
    "S_local_dominant_depth_late": 41,
    "S_same_frame_dominant_all": 59,
    "S_same_frame_dominant_depth_early": 24,
    "S_same_frame_dominant_depth_middle": 15,
    "S_same_frame_dominant_depth_late": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-manifest", type=Path, required=True)
    parser.add_argument("--dominance-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> tuple[Path, bytes, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported manifest schema: {resolved}")
    return resolved, raw, payload


def _targets(record: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(row["block"]), int(row["head"])) for row in record["targets"]}


def main() -> None:
    args = parse_args()
    matched_path, matched_raw, matched = _load(args.matched_manifest)
    dominance_path, dominance_raw, dominance = _load(args.dominance_manifest)
    selected: dict[str, Any] = {}
    for subset_id in MATCHED_IDS:
        selected[subset_id] = matched["subsets"][subset_id]
    for subset_id in DOMINANCE_IDS:
        selected[subset_id] = dominance["subsets"][subset_id]

    if tuple(selected) != MATCHED_IDS + DOMINANCE_IDS:
        raise AssertionError("Unexpected subset order")
    for subset_id, expected in EXPECTED_COUNTS.items():
        record = selected[subset_id]
        targets = _targets(record)
        if int(record["k"]) != expected or len(targets) != expected:
            raise ValueError(
                f"{subset_id}: expected {expected} unique targets, "
                f"found k={record.get('k')} unique={len(targets)}"
            )

    local_all = _targets(selected["S_local_dominant_all"])
    same_all = _targets(selected["S_same_frame_dominant_all"])
    if local_all & same_all:
        raise ValueError("Dominance classes must be disjoint")
    for prefix, all_targets in (
        ("S_local_dominant", local_all),
        ("S_same_frame_dominant", same_all),
    ):
        depth_union = set()
        for depth in ("early", "middle", "late"):
            depth_targets = _targets(selected[f"{prefix}_depth_{depth}"])
            if depth_union & depth_targets:
                raise ValueError(f"{prefix} depth subsets overlap")
            depth_union |= depth_targets
        if depth_union != all_targets:
            raise ValueError(f"{prefix} depth subsets do not partition all targets")

    payload = {
        "schema_version": 1,
        "experiment": "openvid_lora_step10000_strict_head34",
        "selection_policy": {
            "operation": (
                "reuse the exact frozen subsets from the prior three-model "
                "experiments without reselection"
            ),
            "matched_source": str(matched_path),
            "matched_source_sha256": hashlib.sha256(matched_raw).hexdigest(),
            "dominance_source": str(dominance_path),
            "dominance_source_sha256": hashlib.sha256(dominance_raw).hexdigest(),
            "denoise_ranges": [[0, 10], [10, 20], [0, 40]],
            "baseline_configs": 1,
            "ablation_configs": len(selected) * 3,
            "total_configs": 1 + len(selected) * 3,
        },
        "subsets": selected,
        "validation": {
            "subset_count": len(selected),
            "matched_subset_count": len(MATCHED_IDS),
            "dominance_subset_count": len(DOMINANCE_IDS),
            "dominance_classes_disjoint": True,
            "dominance_depth_partitions_exact": True,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "subsets": len(selected),
                "ablation_configs": len(selected) * 3,
                "total_configs_with_baseline": 1 + len(selected) * 3,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
