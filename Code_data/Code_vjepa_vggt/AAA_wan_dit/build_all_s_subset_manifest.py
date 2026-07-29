#!/usr/bin/env python3
"""Freeze the complete public-stable All-S target set as a matched subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common22_public_head_targets import targets_for_role


DEFAULT_REPORT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/"
    "partial_analysis/snapshot_20260728T0245Z/common22/aggregate_heads.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_all_s_missing/"
    "configs/all_s_subset.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets, source = targets_for_role(args.report, "S")
    if len(targets) != 159:
        raise RuntimeError(f"Expected 159 All-S targets, found {len(targets)}")
    subset_id = "S_all_public_stable_k159"
    payload = {
        "schema_version": 1,
        "experiment": "wan_dit_all_s_missing_stage_seed_completion",
        "selection": "all 159 cross-model public stable S heads",
        "source_report": source,
        "source_report_sha256": source["sha256"],
        "subsets": {
            subset_id: {
                "role": "S",
                "k": len(targets),
                "replicate": 0,
                "matching": "all_public_stable_s_heads",
                "block_histogram": dict(
                    sorted(Counter(block for block, _ in targets).items())
                ),
                "targets": [
                    {"block": block, "head": head} for block, head in targets
                ],
            }
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "subset_id": subset_id,
                "targets": len(targets),
                "source_sha256": source["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
