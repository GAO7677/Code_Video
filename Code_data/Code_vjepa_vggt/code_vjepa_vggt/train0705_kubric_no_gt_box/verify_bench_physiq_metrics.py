#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
    "wmreward",
    "videophy2",
    "cosmos_reason1",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
IGNORED_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}


def load_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("input_json") or payload.get("case_json"), str):
        return None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-json-allowlist", type=Path, default=None)
    args = parser.parse_args()

    allowed_input_jsons = None
    if args.input_json_allowlist is not None:
        allowed_input_jsons = {
            Path(line.strip()).expanduser().resolve()
            for line in args.input_json_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    roots = [
        Path(line.strip()).expanduser().resolve()
        for line in args.baseline_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing: list[dict[str, Any]] = []
    root_counts: list[dict[str, Any]] = []
    total_cases = 0
    for root in roots:
        cases = 0
        if not root.is_dir():
            missing.append({"root": str(root), "error": "missing result root"})
            continue
        for path in sorted(root.glob("*.json")):
            if path.name in IGNORED_NAMES or path.name.startswith("eval_summary_"):
                continue
            payload = load_payload(path)
            if payload is None:
                continue
            input_json = Path(payload.get("input_json") or payload.get("case_json")).expanduser().resolve()
            if allowed_input_jsons is not None and input_json not in allowed_input_jsons:
                continue
            cases += 1
            absent = [field for field in METRIC_FIELDS if payload.get(field) is None]
            if absent:
                missing.append({"result_json": str(path), "missing_metrics": absent})
        total_cases += cases
        root_counts.append({"root": str(root), "num_cases": cases})

    report = {
        "baseline_list": str(args.baseline_list.resolve()),
        "num_roots": len(roots),
        "num_cases": total_cases,
        "allowlist_size": None if allowed_input_jsons is None else len(allowed_input_jsons),
        "metric_fields": list(METRIC_FIELDS),
        "num_incomplete_cases": len(missing),
        "complete": not missing,
        "roots": root_counts,
        "incomplete": missing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("num_roots", "num_cases", "num_incomplete_cases", "complete")}, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
