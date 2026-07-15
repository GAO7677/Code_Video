#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_summary(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = {
        key: int(payload[key])
        for key in ("num_total", "num_success", "num_failed", "num_skipped")
    }
    if counts["num_total"] <= 0:
        raise ValueError("inference summary contains no cases")
    if counts["num_failed"] != 0:
        raise ValueError(f"inference failed for {counts['num_failed']} cases")
    if counts["num_success"] != counts["num_total"]:
        raise ValueError(
            "inference did not complete every case: "
            f"success={counts['num_success']} total={counts['num_total']} "
            f"skipped={counts['num_skipped']}"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    counts = validate_summary(args.summary)
    print(json.dumps({"status": "passed", **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
