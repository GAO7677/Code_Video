#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_TASK = "generated_only_sa_pc_joint"
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
    parser.add_argument("--expected-cases", type=int, default=67)
    parser.add_argument("--expected-context-frames", type=int, default=8)
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
    root_reports: list[dict[str, Any]] = []
    for root in roots:
        matched = 0
        if not root.is_dir():
            failures.append({"root": str(root), "error": "missing_result_root"})
            continue
        for path in sorted(root.glob("*.json")):
            if path.name in IGNORED_NAMES or path.name.startswith("eval_summary_"):
                continue
            payload = load_json(path)
            if payload is None:
                continue
            input_json_value = payload.get("input_json") or payload.get("case_json")
            if not isinstance(input_json_value, str):
                continue
            if Path(input_json_value).expanduser().resolve() not in allowed:
                continue
            matched += 1
            result = payload.get("videophy2")
            if not isinstance(result, dict) or result.get("task") != EXPECTED_TASK:
                failures.append({"result_json": str(path), "error": "missing_generated_only_result"})
                continue

            required_numeric = (
                "sa_score",
                "pc_score",
                "joint_pass",
                "pc_raw_score",
                "input_frames",
                "generated_only_frames",
            )
            if any(not isinstance(result.get(key), (int, float)) for key in required_numeric):
                failures.append({"result_json": str(path), "error": "missing_numeric_fields"})
                continue
            expected_joint = int(result["sa_score"] >= 4 and result["pc_score"] >= 4)
            if result["joint_pass"] != expected_joint or result.get("score") != expected_joint:
                failures.append({"result_json": str(path), "error": "invalid_joint_value"})
                continue
            if result.get("context_frames_removed") != args.expected_context_frames:
                failures.append(
                    {
                        "result_json": str(path),
                        "error": "unexpected_context_frames",
                        "value": result.get("context_frames_removed"),
                    }
                )
                continue
            if result["generated_only_frames"] != result["input_frames"] - args.expected_context_frames:
                failures.append({"result_json": str(path), "error": "invalid_generated_only_frame_count"})
                continue
            verified += 1

        if matched != args.expected_cases:
            failures.append(
                {
                    "root": str(root),
                    "error": "unexpected_case_count",
                    "expected": args.expected_cases,
                    "actual": matched,
                }
            )
        root_reports.append({"root": str(root), "matched_cases": matched})

    report = {
        "complete": not failures,
        "num_roots": len(roots),
        "expected_cases_per_root": args.expected_cases,
        "expected_records": len(roots) * args.expected_cases,
        "verified_records": verified,
        "failures": failures,
        "roots": root_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("complete", "num_roots", "verified_records")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
