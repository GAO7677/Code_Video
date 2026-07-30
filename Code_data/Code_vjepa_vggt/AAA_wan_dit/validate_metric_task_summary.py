#!/usr/bin/env python3
"""Reject metric tasks that exited successfully without scoring every case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-cases", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary.expanduser().resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    status = payload.get("metric_status")
    expected = args.expected_cases
    errors = []
    if not isinstance(status, dict):
        errors.append("metric_status is missing")
    else:
        checks = {
            "num_cases": expected,
            "num_success": expected,
            "num_failed": 0,
            "completed": expected,
        }
        for key, expected_value in checks.items():
            if status.get(key) != expected_value:
                errors.append(
                    f"{key}={status.get(key)!r}, expected {expected_value!r}"
                )
    if payload.get("errors"):
        errors.append(f"summary contains {len(payload['errors'])} errors")
    if errors:
        raise RuntimeError(
            f"Invalid metric task summary {summary_path}: " + "; ".join(errors)
        )
    print(
        f"[metric-summary-valid] {summary_path.name} "
        f"cases={expected} success={expected}"
    )


if __name__ == "__main__":
    main()
