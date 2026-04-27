#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect PhysInOne benchmark summaries into one CSV.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--summary_csv", type=Path, required=True)
    return parser.parse_args()


def flatten_record(model_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"model_name": model_dir.name}
    for prefix in ["short_metrics", "i2v_metrics", "continuation_metrics"]:
        for key, value in (payload.get(prefix) or {}).items():
            row[f"{prefix.replace('_metrics', '')}_{key}"] = value
    return row


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, Any]] = []
    for path in sorted(args.output_root.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(flatten_record(path.parent, payload))

    fieldnames = sorted({key for row in rows for key in row.keys()})
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.summary_csv)


if __name__ == "__main__":
    main()
