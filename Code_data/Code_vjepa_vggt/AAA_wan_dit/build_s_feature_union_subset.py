#!/usr/bin/env python3
"""Build the frozen 64-head union of the two S-feature ablation subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


UNION_ID = "S_local_same_union_k64_r00_exactblock"


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


def main() -> None:
    args = parse_args()
    source = args.source_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    raw = source.read_bytes()
    payload = json.loads(raw)
    by_subtype = {
        record["feature_subtype"]: record
        for record in payload["subsets"].values()
    }
    required = {"local_enrichment", "same_frame_mass"}
    if set(by_subtype) != required:
        raise RuntimeError(
            f"Expected exactly {sorted(required)}, found {sorted(by_subtype)}"
        )

    local = by_subtype["local_enrichment"]
    same = by_subtype["same_frame_mass"]
    if int(local["k"]) != 32 or int(same["k"]) != 32:
        raise RuntimeError("The frozen source subsets must each contain 32 heads")
    local_keys = {
        (int(target["block"]), int(target["head"])) for target in local["targets"]
    }
    same_keys = {
        (int(target["block"]), int(target["head"])) for target in same["targets"]
    }
    if local_keys & same_keys:
        raise RuntimeError("The source subsets overlap")

    targets_by_key = {
        (int(target["block"]), int(target["head"])): target
        for target in [*local["targets"], *same["targets"]]
    }
    targets = [
        targets_by_key[key]
        for key in sorted(local_keys | same_keys)
    ]
    if len(targets) != 64:
        raise RuntimeError(f"Expected a 64-head union, found {len(targets)}")
    histogram = {
        str(block): count
        for block, count in sorted(
            Counter(int(target["block"]) for target in targets).items()
        )
    }
    union = {
        "schema_version": 1,
        "experiment": "s_local_and_same_frame_union",
        "selection_policy": {
            "source_manifest": str(source),
            "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "operation": "set union of the frozen local_enrichment and same_frame_mass subsets",
            "strictly_disjoint_sources": True,
            "source_k_each": 32,
            "union_k": 64,
        },
        "subsets": {
            UNION_ID: {
                "role": "S",
                "feature_subtype": "local_same_union",
                "k": 64,
                "replicate": 0,
                "matching": "exact_block_feature_union",
                "block_histogram": histogram,
                "source_subset_ids": [
                    key
                    for key, record in payload["subsets"].items()
                    if record["feature_subtype"] in required
                ],
                "targets": targets,
            }
        },
    }
    atomic_json(output, union)
    print(
        json.dumps(
            {
                "output": str(output),
                "subset_id": UNION_ID,
                "source_manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "local_count": len(local_keys),
                "same_frame_count": len(same_keys),
                "intersection": len(local_keys & same_keys),
                "union_count": len(targets),
                "block_histogram": histogram,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
