#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0713pybullet")

MOTION_METRIC_KEYS = (
    "method",
    "frame_count",
    "transition_count",
    "fps",
    "analysis_resolution",
    "top_flow_percent",
    "min_motion_px",
    "noise_mad_scale",
    "motion_degree_diag_pct_per_second",
    "motion_object_diag_pct_per_second",
    "motion_object_energy",
    "motion_vbench_top_diag_pct_per_second",
    "moving_area_ratio",
    "moving_area_ratio_p90",
    "motion_presence_ratio",
    "motion_object_p90_diag_pct_per_second",
    "motion_temporal_p90_px_per_frame",
    "motion_temporal_peak_px_per_frame",
    "relative_motion_level",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach computed motion metrics to every rigid video sample manifest/meta."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--include-direction-check",
        action="store_true",
        help="Also process dataset-root/direction_check when its manifest and motion metrics exist.",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=1.0,
        help="Threshold below which a sample is marked drop_lt_threshold.",
    )
    parser.add_argument(
        "--filter-metric",
        default="motion_vbench_top_diag_pct_per_second",
        help="Metric used for keep/drop filtering.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _case_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("case_id") or record.get("sample_key") or Path(str(record["output_root"])).name)


def _compact_motion_metrics(record: dict[str, Any], drop_threshold: float, filter_metric: str) -> dict[str, Any]:
    metrics = {key: record[key] for key in MOTION_METRIC_KEYS if key in record}
    filter_score = metrics.get(filter_metric)
    metrics["primary_metric"] = filter_metric
    metrics["drop_threshold"] = float(drop_threshold)
    metrics["filter_metric"] = filter_metric
    metrics["filter_value"] = filter_score
    metrics["filter_status"] = (
        "drop_lt_threshold"
        if filter_score is not None and float(filter_score) < float(drop_threshold)
        else "keep_ge_threshold"
    )
    metrics["source_video"] = record.get("video", "")
    metrics["original_resolution"] = record.get("original_resolution", [])
    return metrics


def _attach_to_json_file(path: Path, motion_payload: dict[str, Any], sidecar_path: Path) -> None:
    if not path.exists():
        return
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return
    payload["motion_metrics_path"] = str(sidecar_path)
    payload["motion_metrics"] = motion_payload
    _write_json(path, payload)


def _process_manifest_root(manifest_root: Path, drop_threshold: float, filter_metric: str) -> dict[str, Any]:
    manifest_path = manifest_root / "manifest.json"
    metrics_path = manifest_root / "motion_metrics.json"
    if not manifest_path.exists() or not metrics_path.exists():
        return {
            "manifest_root": str(manifest_root),
            "cases": 0,
            "missing_manifest_or_metrics": True,
        }

    manifest = _load_json(manifest_path)
    motion_payload = _load_json(metrics_path)
    if not isinstance(manifest, list):
        raise ValueError(f"manifest must be a list: {manifest_path}")
    metric_records = motion_payload.get("records", [])
    metrics_by_case = {str(record["case_id"]): record for record in metric_records}

    updated_manifest: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    drop_cases: list[str] = []
    keep_cases: list[str] = []

    for manifest_record in manifest:
        if not isinstance(manifest_record, dict):
            continue
        case_id = _case_id_from_record(manifest_record)
        metric_record = metrics_by_case.get(case_id)
        if not metric_record:
            missing_cases.append(case_id)
            updated_manifest.append(manifest_record)
            continue

        if filter_metric not in metric_record:
            raise KeyError(f"filter metric not found for {case_id}: {filter_metric}")
        compact = _compact_motion_metrics(metric_record, drop_threshold, filter_metric)
        if compact["filter_status"] == "drop_lt_threshold":
            drop_cases.append(case_id)
        else:
            keep_cases.append(case_id)

        meta_path = Path(str(manifest_record.get("meta", ""))) if manifest_record.get("meta") else None
        sidecar_path = (
            meta_path.parent / f"{case_id}_motion_metrics.json"
            if meta_path is not None
            else Path(str(manifest_record["output_root"])) / "meta" / f"{case_id}_motion_metrics.json"
        )
        sidecar_payload = {
            "case_id": case_id,
            "family_key": manifest_record.get("family_key", metric_record.get("family_key", "")),
            "video": manifest_record.get("video", metric_record.get("video", "")),
            "motion_metrics": compact,
        }
        _write_json(sidecar_path, sidecar_payload)

        manifest_record["motion_metrics_path"] = str(sidecar_path)
        manifest_record["motion_metrics"] = compact
        _attach_to_json_file(Path(str(manifest_record["output_root"])) / "case_manifest.json", compact, sidecar_path)
        if meta_path is not None:
            _attach_to_json_file(meta_path, compact, sidecar_path)
        updated_manifest.append(manifest_record)

    _write_json(manifest_path, updated_manifest)
    summary = {
        "manifest_root": str(manifest_root),
        "cases": len(updated_manifest),
        "metrics_attached": len(keep_cases) + len(drop_cases),
        "keep_ge_threshold": len(keep_cases),
        "drop_lt_threshold": len(drop_cases),
        "missing_metrics": missing_cases,
        "drop_cases": drop_cases,
        "drop_threshold": float(drop_threshold),
        "primary_metric": filter_metric,
        "filter_metric": filter_metric,
        "filter_rule": f"{filter_metric} >= {float(drop_threshold):.6g}",
    }
    _write_json(manifest_root / "motion_filter_summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    manifest_roots = [args.dataset_root]
    direction_root = args.dataset_root / "direction_check"
    if args.include_direction_check and direction_root.exists():
        manifest_roots.append(direction_root)
    summaries = [_process_manifest_root(path, args.drop_threshold, args.filter_metric) for path in manifest_roots]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
