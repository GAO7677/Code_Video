#!/usr/bin/env python3
"""Merge a full-video result with prefix-frame comparison results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.full) + read_rows(args.prefix)
    if not rows or any(row.get("status") != "ok" for row in rows):
        raise ValueError("All comparison rows must have status=ok")
    base_case = rows[0]["case_id"]
    for number, row in enumerate(rows, start=1):
        row["case_number"] = number
        row["comparison_group"] = base_case
        if row["case_id"] == base_case:
            row["comparison_variant"] = "完整视频"
        elif "_first8" in row["case_id"]:
            row["comparison_variant"] = "前 8 帧"
        elif "_first16" in row["case_id"]:
            row["comparison_variant"] = "前 16 帧"
        else:
            row["comparison_variant"] = row["case_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged_rows={len(rows)} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
