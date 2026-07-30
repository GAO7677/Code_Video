#!/usr/bin/env python3
"""Remap copied SSH118 result metadata to canonical local case JSONs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--input-json-allowlist", type=Path, required=True)
    parser.add_argument("--expected-sidecars", type=int, default=1440)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    allowlist = args.input_json_allowlist.expanduser().resolve()
    canonical = {
        path.stem: path
        for line in allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
        for path in [Path(line.strip()).expanduser().resolve()]
    }
    missing_inputs = [str(path) for path in canonical.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing canonical input JSONs: {missing_inputs}")

    sidecars = sorted(
        path
        for path in generation_root.rglob("*.json")
        if path.with_suffix(".mp4").is_file()
    )
    if len(sidecars) != args.expected_sidecars:
        raise RuntimeError(
            f"Expected {args.expected_sidecars} video sidecars, found {len(sidecars)}"
        )

    changed = 0
    already_canonical = 0
    unknown_cases: list[str] = []
    for sidecar in sidecars:
        case_json = canonical.get(sidecar.stem)
        if case_json is None:
            unknown_cases.append(str(sidecar))
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        previous = payload.get("input_json")
        canonical_text = str(case_json)
        if previous == canonical_text:
            already_canonical += 1
            continue
        if isinstance(previous, str) and previous:
            payload.setdefault("input_json_original_ssh118", previous)
        payload["input_json"] = canonical_text
        atomic_json(sidecar, payload)
        changed += 1

    if unknown_cases:
        raise RuntimeError(
            f"{len(unknown_cases)} sidecars do not match the canonical allowlist: "
            f"{unknown_cases[:5]}"
        )
    report = {
        "generation_root": str(generation_root),
        "input_json_allowlist": str(allowlist),
        "sidecars": len(sidecars),
        "changed": changed,
        "already_canonical": already_canonical,
        "canonical_cases": len(canonical),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
