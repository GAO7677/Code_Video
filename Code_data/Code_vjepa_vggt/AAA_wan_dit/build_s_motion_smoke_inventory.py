#!/usr/bin/env python3
"""Select one complete case for the S-head Motion Impact smoke report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_SOURCE = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/inventory.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_smoke/inventory.json"
)
DEFAULT_CASE = "0613pybullet_sample_000301_w000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-id", default=DEFAULT_CASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.source.read_text(encoding="utf-8"))
    entries = []
    for entry in payload["entries"]:
        if entry["case_id"] != args.case_id:
            continue
        family = entry["family"]
        stage = entry.get("denoise_step_range")
        if family in ("gt", "baseline"):
            entries.append(entry)
        elif family == "s_feature" and stage == [0, 40]:
            entries.append(entry)
        elif family == "s_depth" and stage in ([0, 10], [10, 20]):
            entries.append(entry)
    counts = {
        family: sum(entry["family"] == family for entry in entries)
        for family in ("gt", "baseline", "s_feature", "s_depth")
    }
    expected = {"gt": 1, "baseline": 6, "s_feature": 9, "s_depth": 36}
    if counts != expected:
        raise RuntimeError(f"Incomplete smoke inventory: {counts} != {expected}")
    output = {
        **{key: value for key, value in payload.items() if key not in ("entries", "missing", "counts")},
        "schema_version": 3,
        "smoke": True,
        "case_id": args.case_id,
        "entries": entries,
        "missing": [],
        "counts": {"actual": counts, "expected": expected, "total": len(entries)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(f"[s-motion-smoke-inventory] case={args.case_id} entries={len(entries)}")
    print(args.output)


if __name__ == "__main__":
    main()
