#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IGNORED_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-roots", type=Path, required=True)
    parser.add_argument("--input-json-allowlist", type=Path, required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--required-field", default="score")
    parser.add_argument("--expected-cases", type=int, default=67)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    roots = [
        Path(line.strip()).expanduser().resolve()
        for line in args.result_roots.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    allowed = {
        Path(line.strip()).expanduser().resolve()
        for line in args.input_json_allowlist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    failures: list[dict[str, Any]] = []
    verified = 0
    for root in roots:
        matched = 0
        valid = 0
        if not root.is_dir():
            failures.append({"root": str(root), "error": "missing_result_root"})
            continue
        for path in sorted(root.glob("*.json")):
            if path.name in IGNORED_NAMES or path.name.startswith("eval_summary_"):
                continue
            payload = load_json(path)
            if payload is None:
                continue
            input_json = payload.get("input_json") or payload.get("case_json")
            if not isinstance(input_json, str):
                continue
            if Path(input_json).expanduser().resolve() not in allowed:
                continue
            matched += 1
            result = payload.get(args.metric)
            if not isinstance(result, dict):
                failures.append({"result_json": str(path), "error": "missing_metric"})
                continue
            value = result.get(args.required_field)
            if not isinstance(value, (int, float)):
                failures.append(
                    {
                        "result_json": str(path),
                        "error": "missing_required_numeric_field",
                        "field": args.required_field,
                    }
                )
                continue
            valid += 1
            verified += 1
        if matched != args.expected_cases or valid != args.expected_cases:
            failures.append(
                {
                    "root": str(root),
                    "error": "unexpected_case_count",
                    "expected": args.expected_cases,
                    "matched": matched,
                    "valid": valid,
                }
            )

    report = {
        "complete": not failures,
        "metric": args.metric,
        "num_roots": len(roots),
        "expected_records": len(roots) * args.expected_cases,
        "verified_records": verified,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("complete", "metric", "verified_records")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
