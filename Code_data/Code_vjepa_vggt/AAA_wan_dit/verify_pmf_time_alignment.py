#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = ("pmf_with_context", "pmf_without_context")
EXPECTED_ALIGNMENT = "timestamp_resample_to_common_duration_before_pmf"
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
    root_reports: list[dict[str, Any]] = []
    for root in roots:
        matched_cases = 0
        aligned_metrics = 0
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
            matched_cases += 1
            for metric in METRICS:
                result = payload.get(metric)
                if not isinstance(result, dict):
                    failures.append({"result_json": str(path), "metric": metric, "error": "missing_metric"})
                    continue
                if result.get("frame_alignment") != EXPECTED_ALIGNMENT:
                    failures.append(
                        {
                            "result_json": str(path),
                            "metric": metric,
                            "error": "unexpected_alignment",
                            "value": result.get("frame_alignment"),
                        }
                    )
                    continue
                compared = result.get("num_frames_compared")
                used_shape = result.get("used_shape")
                if (
                    not isinstance(compared, int)
                    or compared < 1
                    or not isinstance(used_shape, list)
                    or not used_shape
                    or used_shape[0] != compared
                ):
                    failures.append(
                        {
                            "result_json": str(path),
                            "metric": metric,
                            "error": "invalid_compared_frame_metadata",
                            "num_frames_compared": compared,
                            "used_shape": used_shape,
                        }
                    )
                    continue
                aligned_metrics += 1
        if matched_cases != args.expected_cases:
            failures.append(
                {
                    "root": str(root),
                    "error": "unexpected_case_count",
                    "expected": args.expected_cases,
                    "actual": matched_cases,
                }
            )
        root_reports.append(
            {
                "root": str(root),
                "matched_cases": matched_cases,
                "aligned_metrics": aligned_metrics,
            }
        )

    report = {
        "complete": not failures,
        "expected_alignment": EXPECTED_ALIGNMENT,
        "num_roots": len(roots),
        "expected_cases_per_root": args.expected_cases,
        "expected_metric_records": len(roots) * args.expected_cases * len(METRICS),
        "verified_metric_records": sum(item["aligned_metrics"] for item in root_reports),
        "failures": failures,
        "roots": root_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("complete", "num_roots", "verified_metric_records")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
