#!/usr/bin/env python3
"""Aggregate strict bench.py sidecars for the P0 dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physicsiq-verified-strict-metrics"
)
VBA = (
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
METRICS = VBA + ("videophy2", "cosmos_reason1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def metric_value(payload: dict[str, Any], metric: str) -> float | None:
    if metric.startswith("vbench_"):
        bucket = payload.get(metric)
        return finite_number(bucket.get("score")) if isinstance(bucket, dict) else None
    if metric == "cosmos_reason1":
        bucket = payload.get(metric)
        return finite_number(bucket.get("score")) if isinstance(bucket, dict) else None
    bucket = payload.get("videophy2")
    if not isinstance(bucket, dict):
        return None
    # The reference table exposes the joint rate, SA, PC, PC raw, and pass
    # columns.  `score` and `joint_rate` are the same per-case pass indicator.
    if metric == "videophy2":
        return finite_number(bucket.get("score"))
    return None


def videophy_fields(payload: dict[str, Any]) -> dict[str, float | None]:
    bucket = payload.get("videophy2")
    if not isinstance(bucket, dict):
        return {
            "videophy2": None,
            "videophy2_sa": None,
            "videophy2_pc": None,
            "videophy2_joint_rate": None,
            "videophy2_pc_raw": None,
        }
    return {
        "videophy2": finite_number(bucket.get("score")),
        "videophy2_sa": finite_number(bucket.get("sa_score")),
        "videophy2_pc": finite_number(bucket.get("pc_score")),
        "videophy2_joint_rate": finite_number(bucket.get("joint_rate", bucket.get("joint_pass"))),
        "videophy2_pc_raw": finite_number(bucket.get("pc_raw_score")),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "count": len(values),
        "mean": round(fmean(values), 10) if values else None,
        "min": round(min(values), 10) if values else None,
        "max": round(max(values), 10) if values else None,
    }
    result["std"] = round(stdev(values), 10) if len(values) > 1 else 0.0 if values else None
    return result


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    methods: dict[str, Any] = {}
    required = int(manifest.get("expected_cases", 198))
    all_complete = True
    for method in manifest["methods"]:
        key = str(method["key"])
        result_root = Path(method["result_root"])
        records = sorted(result_root.glob("*.json"))
        per_case: dict[str, dict[str, Any]] = {}
        for record_path in records:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            case_key = record_path.stem
            row: dict[str, Any] = {
                "result_json": str(record_path),
                "input_json": payload.get("input_json"),
            }
            row.update({metric: metric_value(payload, metric) for metric in METRICS})
            row.update(videophy_fields(payload))
            per_case[case_key] = row
        aggregate: dict[str, Any] = {}
        for metric in METRICS + (
            "videophy2_sa",
            "videophy2_pc",
            "videophy2_joint_rate",
            "videophy2_pc_raw",
        ):
            values = [
                value
                for row in per_case.values()
                if (value := finite_number(row.get(metric))) is not None
            ]
            aggregate[metric] = summarize(values)
        coverage = {
            metric: int(aggregate[metric]["count"]) for metric in aggregate
        }
        complete = all(coverage[metric] == required for metric in METRICS)
        all_complete = all_complete and complete
        methods[key] = {
            "label": method["label"],
            "result_root": str(result_root),
            "num_records": len(per_case),
            "expected_cases": required,
            "complete": complete,
            "coverage": coverage,
            "aggregate": aggregate,
            "cases": per_case,
        }
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if all_complete else "partial",
        "expected_cases": required,
        "metric_names": list(METRICS),
        "protocol": {
            "runner": "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py",
            "videophy2_task": "generated_only_sa_pc_joint",
            "vbench_dimensions": list(VBA),
            "candidate_input": "120-frame Physics-IQ Verified submission video",
            "context_frames_removed_by_metric_adapter": 0,
        },
        "methods": methods,
    }
    if not all_complete and not args.allow_partial:
        # Still write the partial snapshot so the dashboard can show progress,
        # then return a nonzero status for automation.
        atomic_write(root / "strict_metrics.json", output)
        raise SystemExit("strict metric coverage is partial; use --allow-partial to accept it")
    atomic_write(root / "strict_metrics.json", output)
    print(json.dumps({"output": str(root / "strict_metrics.json"), "status": output["status"], "methods": {k: v["coverage"] for k, v in methods.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
