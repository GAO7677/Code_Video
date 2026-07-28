#!/usr/bin/env python3
"""Return success only for a fully annotated phased-ablation task."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VARIANT_PATTERN = re.compile(
    r"^[STPCG](?:_(?:top|bottom)10)?_steps(?P<start>\d{2})_(?P<end>\d{2})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    return parser.parse_args()


def iter_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def main() -> int:
    args = parse_args()
    match = VARIANT_PATTERN.fullmatch(args.variant)
    if match is None:
        raise ValueError(f"Unsupported phased variant: {args.variant}")
    expected_range = [int(match.group("start")), int(match.group("end"))]
    task_root = (
        args.output_root
        / "generated"
        / args.model
        / f"seed-{args.seed:06d}"
        / f"role-{args.variant}"
    )
    videos = [
        path
        for path in task_root.rglob("*.mp4")
        if path.is_file() and path.stat().st_size > 0
    ]
    if not videos:
        return 1
    for path in task_root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for record in iter_dicts(payload):
            if (
                record.get("target_forward_call_count_ok") is True
                and record.get("active_denoise_step_range") == expected_range
                and record.get("observed_target_forward_calls")
                == record.get("expected_target_forward_calls")
            ):
                return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
