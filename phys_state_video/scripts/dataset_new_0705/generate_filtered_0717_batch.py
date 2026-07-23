#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .backfill_motion_metrics import _compact_motion_metrics
from .compute_motion_degree import compute_video_motion
from .render_sim_0705 import render_generated_case
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    build_scenario_family_catalog,
    preview_diversity_report,
)


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5")
DEFAULT_FILTER_METRIC = "motion_vbench_top_diag_pct_per_second"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a filtered rigid PyBullet dataset, keeping samples until motion quality target is met."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-keep", type=int, default=5000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed-base", type=int, default=20260717)
    parser.add_argument("--family-pattern", default="balanced")
    parser.add_argument("--scene-style", default="indoor_realistic")
    parser.add_argument("--direction-mode", default="auto")
    parser.add_argument("--size-scale", type=float, default=1.0)
    parser.add_argument("--camera-distance-scale", type=float, default=DEFAULT_CAMERA_DISTANCE_SCALE)
    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--top-flow-percent", type=float, default=0.05)
    parser.add_argument("--min-motion-px", type=float, default=0.05)
    parser.add_argument("--noise-mad-scale", type=float, default=3.0)
    parser.add_argument("--filter-metric", default=DEFAULT_FILTER_METRIC)
    parser.add_argument("--drop-threshold", type=float, default=1.0)
    parser.add_argument("--max-attempts", type=int, default=7000)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--start-attempt", type=int, default=None)
    parser.add_argument(
        "--discard-rejected",
        action="store_true",
        help="Delete rejected case directories after recording their manifest entries.",
    )
    return parser.parse_args()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _replace_path_strings(payload: Any, old: str, new: str) -> Any:
    if isinstance(payload, str):
        return payload.replace(old, new)
    if isinstance(payload, list):
        return [_replace_path_strings(item, old, new) for item in payload]
    if isinstance(payload, dict):
        return {key: _replace_path_strings(value, old, new) for key, value in payload.items()}
    return payload


def _rewrite_json_paths(case_root: Path, old_root: Path, new_root: Path) -> None:
    old = str(old_root)
    new = str(new_root)
    for json_path in case_root.rglob("*.json"):
        payload = _read_json(json_path, {})
        _write_json(json_path, _replace_path_strings(payload, old, new))


def _family_targets(total: int, pattern: str) -> dict[str, int]:
    families = list(build_scenario_family_catalog().keys())
    if pattern != "balanced":
        raise ValueError(f"unsupported family pattern: {pattern}")
    base = total // len(families)
    remainder = total - base * len(families)
    return {family: base + (1 if index < remainder else 0) for index, family in enumerate(families)}


def _family_sort_key(family_key: str) -> tuple[int, str]:
    if family_key.startswith("F") and family_key[1:].isdigit():
        return (int(family_key[1:]), family_key)
    return (999, family_key)


def _next_family(accepted_counts: dict[str, int], targets: dict[str, int]) -> str | None:
    pending = [family for family, target in targets.items() if accepted_counts.get(family, 0) < target]
    if not pending:
        return None
    return min(
        pending,
        key=lambda family: (
            accepted_counts.get(family, 0) / max(targets[family], 1),
            _family_sort_key(family),
        ),
    )


def _detect_next_attempt(output_root: Path, attempts_manifest: list[dict[str, Any]]) -> int:
    max_attempt = -1
    for record in attempts_manifest:
        if isinstance(record, dict) and isinstance(record.get("attempt_index"), int):
            max_attempt = max(max_attempt, int(record["attempt_index"]))
    for root in (output_root / "cases", output_root / "rejected", output_root / "_staging"):
        for case_dir in root.glob("F*/0717_f*_attempt*"):
            suffix = case_dir.name.rsplit("attempt", 1)[-1]
            if suffix.isdigit():
                max_attempt = max(max_attempt, int(suffix))
    return max_attempt + 1


def _motion_payload(metric_record: dict[str, Any], drop_threshold: float, filter_metric: str) -> dict[str, Any]:
    if filter_metric not in metric_record:
        raise KeyError(f"filter metric not found: {filter_metric}")
    return _compact_motion_metrics(metric_record, drop_threshold, filter_metric)


def _motion_summary(records: list[dict[str, Any]], filter_metric: str) -> dict[str, Any]:
    if not records:
        return {"case_count": 0, "primary_metric": filter_metric, "records": []}
    values = np.asarray([float(record[filter_metric]) for record in records], dtype=np.float64)
    low_threshold, high_threshold = (float(value) for value in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]))
    return {
        "definition": {
            "primary_metric": filter_metric,
            "filter_rule": f"{filter_metric} >= 1",
            "motion_vbench_top_diag_pct_per_second": "top 5% residual dense optical-flow magnitude, normalized as % frame diagonal per second",
            "motion_object_diag_pct_per_second": "robust moving-pixel residual optical-flow motion, normalized as % frame diagonal per second",
        },
        "case_count": len(records),
        "primary_metric": filter_metric,
        "relative_level_thresholds": {"low_max": low_threshold, "high_min": high_threshold},
        "summary": {
            f"{filter_metric}_mean": float(values.mean()),
            f"{filter_metric}_median": float(np.median(values)),
            f"{filter_metric}_min": float(values.min()),
            f"{filter_metric}_max": float(values.max()),
        },
        "records": records,
    }


def _case_manifest_record(
    *,
    case_id: str,
    attempt_index: int,
    family_key: str,
    seed: int,
    case_root: Path,
    case_manifest: dict[str, Any],
    motion_metrics: dict[str, Any],
    motion_sidecar_path: Path,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "attempt_index": attempt_index,
        "family_key": family_key,
        "seed": seed,
        "output_root": str(case_root),
        "video": str(case_root / "videos" / f"{case_id}.mp4"),
        "meta": str(case_root / "meta" / f"{case_id}.json"),
        "object_phrases_path": str(case_root / "meta" / f"{case_id}_object_phrases.json"),
        "motion_metrics_path": str(motion_sidecar_path),
        "caption": case_manifest.get("caption", ""),
        "short_caption": case_manifest.get("short_caption", ""),
        "object_nouns": case_manifest.get("object_nouns", []),
        "object_phrases": case_manifest.get("object_phrases", []),
        "dynamic_object_phrases": case_manifest.get("dynamic_object_phrases", []),
        "static_object_phrases": case_manifest.get("static_object_phrases", []),
        "object_phrase_details": case_manifest.get("object_phrase_details", []),
        "motion_metrics": motion_metrics,
        "negative_prompt": case_manifest.get("negative_prompt", ""),
        "size_scale": case_manifest.get("size_scale", 1.0),
        "camera_distance_scale": case_manifest.get("camera_distance_scale", 1.0),
    }


def _attach_motion_files(
    *,
    case_root: Path,
    case_id: str,
    family_key: str,
    record: dict[str, Any],
    motion_metrics: dict[str, Any],
) -> Path:
    sidecar_path = case_root / "meta" / f"{case_id}_motion_metrics.json"
    sidecar_payload = {
        "case_id": case_id,
        "family_key": family_key,
        "video": record["video"],
        "motion_metrics": motion_metrics,
    }
    _write_json(sidecar_path, sidecar_payload)
    for json_path in (case_root / "case_manifest.json", case_root / "meta" / f"{case_id}.json"):
        payload = _read_json(json_path, {})
        if not isinstance(payload, dict):
            continue
        payload["case_id"] = case_id
        payload["motion_metrics_path"] = str(sidecar_path)
        payload["motion_metrics"] = motion_metrics
        _write_json(json_path, payload)
    return sidecar_path


def _write_state(
    *,
    output_root: Path,
    manifest: list[dict[str, Any]],
    rejected_manifest: list[dict[str, Any]],
    attempts_manifest: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    targets: dict[str, int],
    start_time: float,
    filter_metric: str,
    drop_threshold: float,
) -> None:
    accepted_counts = {family: 0 for family in targets}
    for record in manifest:
        accepted_counts[str(record["family_key"])] = accepted_counts.get(str(record["family_key"]), 0) + 1
    rejected_counts: dict[str, int] = {family: 0 for family in targets}
    for record in rejected_manifest:
        rejected_counts[str(record["family_key"])] = rejected_counts.get(str(record["family_key"]), 0) + 1
    motion_records = [
        {
            "case_id": record["case_id"],
            "family_key": record["family_key"],
            **record["motion_metrics"],
        }
        for record in manifest
    ]
    _write_json(output_root / "manifest.json", manifest)
    _write_json(output_root / "rejected_manifest.json", rejected_manifest)
    _write_json(output_root / "attempts_manifest.json", attempts_manifest)
    _write_json(output_root / "motion_metrics.json", _motion_summary(motion_records, filter_metric))
    _write_json(output_root / "reports" / "failure_report.json", failures)
    _write_json(output_root / "reports" / "diversity_report.json", preview_diversity_report())
    summary = {
        "output_root": str(output_root),
        "target_keep": sum(targets.values()),
        "accepted": len(manifest),
        "rejected": len(rejected_manifest),
        "failures": len(failures),
        "attempts": len(attempts_manifest),
        "accepted_counts": accepted_counts,
        "rejected_counts": rejected_counts,
        "targets": targets,
        "filter_metric": filter_metric,
        "drop_threshold": float(drop_threshold),
        "filter_rule": f"{filter_metric} >= {float(drop_threshold):.6g}",
        "elapsed_s": time.time() - start_time,
    }
    _write_json(output_root / "generation_state.json", summary)
    _write_json(
        output_root / "motion_filter_summary.json",
        {
            "manifest_root": str(output_root),
            "cases": len(manifest),
            "metrics_attached": len(manifest),
            "keep_ge_threshold": len(manifest),
            "drop_lt_threshold": len(rejected_manifest),
            "missing_metrics": [],
            "drop_cases": [record["case_id"] for record in rejected_manifest],
            "drop_threshold": float(drop_threshold),
            "primary_metric": filter_metric,
            "filter_metric": filter_metric,
            "filter_rule": f"{filter_metric} >= {float(drop_threshold):.6g}",
        },
    )


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output_root = args.output_root
    for path in [
        output_root / "cases",
        output_root / "rejected",
        output_root / "_staging",
        output_root / "reports",
        output_root / "logs",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    targets = _family_targets(args.target_keep, args.family_pattern)
    manifest: list[dict[str, Any]] = _read_json(output_root / "manifest.json", [])
    rejected_manifest: list[dict[str, Any]] = _read_json(output_root / "rejected_manifest.json", [])
    attempts_manifest: list[dict[str, Any]] = _read_json(output_root / "attempts_manifest.json", [])
    failures: list[dict[str, Any]] = _read_json(output_root / "reports" / "failure_report.json", [])

    accepted_counts = {family: 0 for family in targets}
    for record in manifest:
        accepted_counts[str(record["family_key"])] = accepted_counts.get(str(record["family_key"]), 0) + 1

    attempt_index = args.start_attempt if args.start_attempt is not None else _detect_next_attempt(output_root, attempts_manifest)
    checkpoint_counter = 0

    while len(manifest) < args.target_keep and attempt_index < args.max_attempts:
        family_key = _next_family(accepted_counts, targets)
        if family_key is None:
            break
        case_id = f"0717_{family_key.lower()}_attempt{attempt_index:06d}"
        seed = int(args.seed_base + attempt_index * 1009)
        staging_root = output_root / "_staging" / family_key / case_id
        if staging_root.exists():
            shutil.rmtree(staging_root)

        print(
            f"[attempt {attempt_index:06d}] start {case_id} family={family_key} "
            f"accepted={len(manifest)}/{args.target_keep}",
            flush=True,
        )
        attempt_record: dict[str, Any] = {
            "attempt_index": attempt_index,
            "case_id": case_id,
            "family_key": family_key,
            "seed": seed,
        }
        try:
            record = render_generated_case(
                family_key=family_key,
                sample_key=case_id,
                seed=seed,
                output_root=staging_root,
                width=args.width,
                height=args.height,
                scene_style=args.scene_style,
                direction_mode=args.direction_mode,
                size_scale=args.size_scale,
                camera_distance_scale=args.camera_distance_scale,
            )
            motion = compute_video_motion(
                Path(record["video"]),
                analysis_width=args.analysis_width,
                top_flow_percent=args.top_flow_percent,
                min_motion_px=args.min_motion_px,
                noise_mad_scale=args.noise_mad_scale,
            )
            metric_record = {
                "case_id": case_id,
                "family_key": family_key,
                "direction_mode": args.direction_mode,
                **motion,
            }
            motion_metrics = _motion_payload(metric_record, args.drop_threshold, args.filter_metric)
            keep = motion_metrics["filter_status"] == "keep_ge_threshold"
            destination_parent = output_root / ("cases" if keep else "rejected") / family_key
            destination_root = destination_parent / case_id
            if destination_root.exists():
                shutil.rmtree(destination_root)
            destination_parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging_root), str(destination_root))
            _rewrite_json_paths(destination_root, staging_root, destination_root)

            record = _replace_path_strings(record, str(staging_root), str(destination_root))
            motion_metrics = _replace_path_strings(motion_metrics, str(staging_root), str(destination_root))
            motion_metrics["source_video"] = str(destination_root / "videos" / f"{case_id}.mp4")
            sidecar_path = _attach_motion_files(
                case_root=destination_root,
                case_id=case_id,
                family_key=family_key,
                record=record,
                motion_metrics=motion_metrics,
            )
            case_manifest = _read_json(destination_root / "case_manifest.json", {})
            manifest_record = _case_manifest_record(
                case_id=case_id,
                attempt_index=attempt_index,
                family_key=family_key,
                seed=seed,
                case_root=destination_root,
                case_manifest=case_manifest,
                motion_metrics=motion_metrics,
                motion_sidecar_path=sidecar_path,
            )
            if keep:
                manifest.append(manifest_record)
                accepted_counts[family_key] = accepted_counts.get(family_key, 0) + 1
                outcome = "KEEP"
            else:
                manifest_record["rejection_reason"] = (
                    f"{args.filter_metric}={float(motion_metrics['filter_value']):.6g} < {args.drop_threshold:.6g}"
                )
                rejected_manifest.append(manifest_record)
                outcome = "DROP"
                if args.discard_rejected:
                    shutil.rmtree(destination_root)
                    manifest_record["output_root_deleted"] = True
            attempt_record.update(
                {
                    "status": outcome.lower(),
                    "filter_metric": args.filter_metric,
                    "filter_value": motion_metrics["filter_value"],
                    "drop_threshold": args.drop_threshold,
                    "output_root": str(destination_root),
                }
            )
            attempts_manifest.append(attempt_record)
            print(
                f"[attempt {attempt_index:06d}] {outcome} {case_id} "
                f"{args.filter_metric}={float(motion_metrics['filter_value']):.4f} "
                f"accepted={len(manifest)}/{args.target_keep}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - long-run guard
            failures.append({**attempt_record, "status": "failed", "error": repr(exc)})
            attempts_manifest.append({**attempt_record, "status": "failed", "error": repr(exc)})
            print(f"[attempt {attempt_index:06d}] FAIL {case_id}: {exc!r}", flush=True)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)
            attempt_index += 1
            checkpoint_counter += 1

        if checkpoint_counter >= args.checkpoint_every:
            checkpoint_counter = 0
            _write_state(
                output_root=output_root,
                manifest=manifest,
                rejected_manifest=rejected_manifest,
                attempts_manifest=attempts_manifest,
                failures=failures,
                targets=targets,
                start_time=start_time,
                filter_metric=args.filter_metric,
                drop_threshold=args.drop_threshold,
            )

    _write_state(
        output_root=output_root,
        manifest=manifest,
        rejected_manifest=rejected_manifest,
        attempts_manifest=attempts_manifest,
        failures=failures,
        targets=targets,
        start_time=start_time,
        filter_metric=args.filter_metric,
        drop_threshold=args.drop_threshold,
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "accepted": len(manifest),
                "target_keep": args.target_keep,
                "rejected": len(rejected_manifest),
                "failures": len(failures),
                "attempts": len(attempts_manifest),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
