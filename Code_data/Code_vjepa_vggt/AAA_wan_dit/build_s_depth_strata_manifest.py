#!/usr/bin/env python3
"""Freeze all public S-heads into early, middle, and late depth strata."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from common22_public_head_targets import load_public_head_targets


DEFAULT_REPORT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/"
    "partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_head_role_depth_strata/"
    "configs/s_depth_strata.json"
)
STRATA = (
    ("early", 0, 10),
    ("middle", 10, 20),
    ("late", 20, 30),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = args.report.expanduser().resolve()
    output = args.output.expanduser().resolve()
    targets, source = load_public_head_targets(report)
    subsets = {}
    for name, start, end in STRATA:
        selected = sorted(
            (block, head)
            for block, head in targets["S"]
            if start <= block < end
        )
        subset_id = f"S_depth_{name}_B{start:02d}_{end - 1:02d}_all"
        subsets[subset_id] = {
            "role": "S",
            "k": len(selected),
            "replicate": 0,
            "matching": "all_role_heads_within_depth_stratum",
            "depth_stratum": name,
            "block_start_inclusive": start,
            "block_end_exclusive": end,
            "block_histogram": {
                str(block): count
                for block, count in sorted(
                    Counter(block for block, _ in selected).items()
                )
            },
            "targets": [
                {"block": block, "head": head}
                for block, head in selected
            ],
        }
    payload = {
        "schema_version": 1,
        "experiment": "wan_dit_s_head_depth_strata_total_effect",
        "selection": (
            "all cross-model public stable S-heads whose block lies in the "
            "specified half-open depth stratum"
        ),
        "source_report": source,
        "source_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        "depth_strata": [
            {
                "name": name,
                "block_start_inclusive": start,
                "block_end_exclusive": end,
            }
            for name, start, end in STRATA
        ],
        "subsets": subsets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "[s-depth-manifest] "
        + " ".join(
            f"{record['depth_stratum']}={record['k']}"
            for record in subsets.values()
        )
        + f" output={output}"
    )


if __name__ == "__main__":
    main()
