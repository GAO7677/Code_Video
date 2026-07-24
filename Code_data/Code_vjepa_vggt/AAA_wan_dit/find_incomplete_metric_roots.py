#!/usr/bin/env python3
"""Print result roots that still have missing values for one metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_paths(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-list", type=Path, required=True)
    parser.add_argument("--input-allowlist", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    args = parser.parse_args()

    allowed_inputs = {str(path) for path in read_paths(args.input_allowlist)}
    for root in read_paths(args.baseline_list):
        num_cases = 0
        num_missing = 0
        for path in root.rglob("*.json"):
            if (
                path.name in {"summary.json", "batch_manifest.json", "eval_summary.json", "result.json"}
                or path.name.startswith("eval_summary_")
            ):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            input_json = payload.get("input_json") or payload.get("case_json")
            if not input_json:
                continue
            if str(Path(input_json).expanduser().resolve()) not in allowed_inputs:
                continue
            num_cases += 1
            if payload.get(args.metric) is None:
                num_missing += 1
        if num_cases and num_missing:
            print(f"{root}\t{num_missing}\t{num_cases}")


if __name__ == "__main__":
    main()

