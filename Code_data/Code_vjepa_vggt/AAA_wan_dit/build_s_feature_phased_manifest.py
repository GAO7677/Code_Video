#!/usr/bin/env python3
"""Freeze Local-32, Same-frame-32, and Union-64 into one phased manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--union-manifest", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    split_path = args.split_manifest.expanduser().resolve()
    union_path = args.union_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    split_raw = split_path.read_bytes()
    union_raw = union_path.read_bytes()
    split = json.loads(split_raw)
    union = json.loads(union_raw)
    subsets = {**split["subsets"], **union["subsets"]}
    expected_subtypes = {
        "local_enrichment": 32,
        "same_frame_mass": 32,
        "local_same_union": 64,
    }
    observed = {
        record["feature_subtype"]: int(record["k"])
        for record in subsets.values()
    }
    if observed != expected_subtypes:
        raise RuntimeError(f"Unexpected phased subsets: {observed}")
    keys = {
        subtype: {
            (int(target["block"]), int(target["head"]))
            for target in record["targets"]
        }
        for subtype, expected_k in expected_subtypes.items()
        for record in subsets.values()
        if record["feature_subtype"] == subtype
    }
    if keys["local_enrichment"] & keys["same_frame_mass"]:
        raise RuntimeError("The two 32-head subsets overlap")
    if keys["local_enrichment"] | keys["same_frame_mass"] != keys["local_same_union"]:
        raise RuntimeError("The 64-head subset is not the exact union of the 32-head subsets")
    payload = {
        "schema_version": 1,
        "experiment": "s_feature_category_phased_ablation",
        "selection_policy": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": hashlib.sha256(split_raw).hexdigest(),
            "union_manifest": str(union_path),
            "union_manifest_sha256": hashlib.sha256(union_raw).hexdigest(),
            "operation": "freeze the existing three S-feature subsets without reselection",
            "denoise_ranges": [[0, 10], [10, 20]],
        },
        "subsets": subsets,
    }
    atomic_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "subsets": observed,
                "local_same_intersection": len(
                    keys["local_enrichment"] & keys["same_frame_mass"]
                ),
                "union_exact": (
                    keys["local_enrichment"] | keys["same_frame_mass"]
                    == keys["local_same_union"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
