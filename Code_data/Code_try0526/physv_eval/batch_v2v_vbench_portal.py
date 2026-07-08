from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .case_inputs import EvalCase
from .paths import AGENT_OUTPUT_ROOT
from .records import load_payload
from .vbench2_official import OfficialVBench2Runner
from .vbench_official import OfficialVBenchRunner


TARGET_JSON_NAME = "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json"
VBENCH_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
VBENCH2_PER_VIDEO_DIMENSIONS = [
    "Human_Anatomy",
    "Human_Identity",
    "Human_Clothes",
    "Multi-View_Consistency",
]
VBENCH2_DATASET_DIMENSIONS = ["Diversity"]


@dataclass
class CaseRecord:
    json_path: Path
    video_path: Path
    caption: str | None
    group: str
    method: str
    payload: dict[str, Any]

    @property
    def case_rel(self) -> str:
        return str(self.json_path.parent.relative_to(self.json_path.parents[2]))


@dataclass
class SkippedCase:
    json_path: Path
    reason: str
    detail: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-evaluate VBench/VBench2 metrics for v2v JSON cases and render a local portal summary.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/test/v2v"),
        help="Root directory containing target JSON files.",
    )
    parser.add_argument(
        "--json-name",
        default=TARGET_JSON_NAME,
        help="Leaf JSON filename to scan for.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=AGENT_OUTPUT_ROOT / "v2v_vbench_portal",
        help="Directory for summary outputs and portal assets.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick validation.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index. User requested gpu0.",
    )
    parser.add_argument(
        "--skip-vbench",
        action="store_true",
        help="Skip VBench execution and only aggregate existing fields.",
    )
    parser.add_argument(
        "--skip-vbench2",
        action="store_true",
        help="Skip VBench2 execution and only aggregate existing fields.",
    )
    parser.add_argument(
        "--disable-local-ckpt",
        action="store_true",
        help="Disable local checkpoint preference. By default this batch pipeline prefers local weights/repos under /data/gaoya/ckpt.",
    )
    parser.add_argument(
        "--read-frame",
        action="store_true",
        help="Forward read_frame flag when available.",
    )
    parser.add_argument(
        "--imaging-quality-preprocessing-mode",
        default="longer",
        choices=["shorter", "longer", "shorter_centercrop", "None"],
        help="VBench imaging-quality preprocessing mode.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only re-render the portal from an existing summary.json in out-dir.",
    )
    return parser.parse_args()


def scan_cases(root: Path, json_name: str, limit: int | None) -> tuple[list[CaseRecord], list[SkippedCase]]:
    json_paths = sorted(root.rglob(json_name))
    if limit is not None:
        json_paths = json_paths[:limit]

    records: list[CaseRecord] = []
    skipped: list[SkippedCase] = []
    for json_path in json_paths:
        payload = load_payload(json_path)
        output_video = payload.get("output_video")
        if not isinstance(output_video, str) or not output_video.strip():
            skipped.append(
                SkippedCase(
                    json_path=json_path.resolve(),
                    reason="missing_output_video",
                    detail="JSON does not contain an evaluable output_video field.",
                )
            )
            continue
        video_path = Path(output_video).resolve()
        if not video_path.is_file():
            skipped.append(
                SkippedCase(
                    json_path=json_path.resolve(),
                    reason="missing_video_file",
                    detail=str(video_path),
                )
            )
            continue
        caption = payload.get("input_caption")
        if caption is not None:
            caption = str(caption).strip() or None
        records.append(
            CaseRecord(
                json_path=json_path.resolve(),
                video_path=video_path,
                caption=caption,
                group=json_path.parents[1].name,
                method=str(payload.get("method") or json_path.parent.name),
                payload=payload,
            )
        )
    return records, skipped


def ensure_runtime_env(gpu: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    for key in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
        os.environ.pop(key, None)


def scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def nested_scalar(bucket: Any, *keys: str) -> float | None:
    current = bucket
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return scalar(current)


def extract_existing_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    wmreward = payload.get("wmreward")
    physics_iq_ctx = payload.get("physics_iq_with_context")
    physics_iq_nctx = payload.get("physics_iq_without_context")
    pmf_ctx = payload.get("pmf_with_context")
    pmf_nctx = payload.get("pmf_without_context")
    videophy2 = payload.get("videophy2")
    cosmos = payload.get("cosmos_reason1")
    return {
        "wmreward_similarity": nested_scalar(wmreward, "similarity"),
        "wmreward_surprise": nested_scalar(wmreward, "surprise"),
        "physics_iq_with_context": nested_scalar(physics_iq_ctx, "score"),
        "physics_iq_without_context": nested_scalar(physics_iq_nctx, "score"),
        "pmf_with_context": nested_scalar(pmf_ctx, "score"),
        "pmf_without_context": nested_scalar(pmf_nctx, "score"),
        "videophy2": nested_scalar(videophy2, "score"),
        "cosmos_reason1": nested_scalar(cosmos, "score"),
    }


def base_metric_result() -> dict[str, Any]:
    return {"status": "pending", "score": None}


def no_valid_samples_result(note: str | None = None) -> dict[str, Any]:
    result = {"status": "no_valid_samples", "score": None}
    if note:
        result["note"] = note
    return result


def should_preserve_prior_result(prior: Any, current: Any) -> bool:
    if not isinstance(prior, dict) or not isinstance(current, dict):
        return False
    prior_status = prior.get("status")
    current_status = current.get("status")
    return prior_status in {"ok", "error", "no_valid_samples"} and current_status in {"skipped", "pending"}


def run_vbench_dimensions(
    cases: list[CaseRecord],
    out_dir: Path,
    *,
    load_ckpt_from_local: bool,
    read_frame: bool,
    imaging_quality_preprocessing_mode: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    by_video: dict[str, dict[str, dict[str, Any]]] = {
        str(case.video_path): {dimension: base_metric_result() for dimension in VBENCH_DIMENSIONS}
        for case in cases
    }
    run_status: dict[str, Any] = {}
    runner = OfficialVBenchRunner(
        device="cuda",
        load_ckpt_from_local=load_ckpt_from_local,
        read_frame=read_frame,
        imaging_quality_preprocessing_mode=imaging_quality_preprocessing_mode,
    )
    eval_cases = [EvalCase(video_path=case.video_path, caption=case.caption) for case in cases]

    for dimension in VBENCH_DIMENSIONS:
        run_name = f"batch_{dimension}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = out_dir / "official_runs" / "vbench" / dimension
        try:
            result = runner.score_batch(
                eval_cases,
                dimension=dimension,
                output_path=output_path,
                run_name=run_name,
            )
            run_status[dimension] = {
                "status": "ok",
                "score": result.get("score"),
                "result_json": result.get("result_json"),
                "full_info_json": result.get("full_info_json"),
            }
            for item in result.get("raw_results", []):
                video_path = str(Path(str(item.get("video_path"))).resolve())
                if video_path not in by_video:
                    continue
                by_video[video_path][dimension] = {
                    "status": "ok",
                    "score": scalar(item.get("video_results")),
                }
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            run_status[dimension] = {"status": "error", "error": error_text}
            for video_path in by_video:
                by_video[video_path][dimension] = {"status": "error", "score": None, "error": error_text}
    return by_video, run_status


def run_vbench2_dimensions(
    cases: list[CaseRecord],
    out_dir: Path,
    *,
    load_ckpt_from_local: bool,
    read_frame: bool,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    by_video: dict[str, dict[str, dict[str, Any]]] = {
        str(case.video_path): {dimension: base_metric_result() for dimension in VBENCH2_PER_VIDEO_DIMENSIONS}
        for case in cases
    }
    run_status: dict[str, Any] = {}
    dataset_metrics: dict[str, Any] = {}
    runner = OfficialVBench2Runner(
        device="cuda",
        load_ckpt_from_local=load_ckpt_from_local,
        read_frame=read_frame,
    )
    eval_cases = [EvalCase(video_path=case.video_path, caption=case.caption) for case in cases]

    for dimension in VBENCH2_PER_VIDEO_DIMENSIONS:
        run_name = f"batch_{dimension}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_path = out_dir / "official_runs" / "vbench2" / dimension
        try:
            result = runner.score_batch(
                eval_cases,
                dimension=dimension,
                output_path=output_path,
                run_name=run_name,
            )
            run_status[dimension] = {
                "status": "ok",
                "score": result.get("score"),
                "result_json": result.get("result_json"),
                "full_info_json": result.get("full_info_json"),
            }
            for item in result.get("raw_results", []):
                video_path = str(Path(str(item.get("video_path"))).resolve())
                if video_path not in by_video:
                    continue
                by_video[video_path][dimension] = {
                    "status": "ok",
                    "score": scalar(item.get("video_results")),
                }
        except ZeroDivisionError:
            note = "Official VBench2 found zero valid samples for this dimension on the current dataset."
            run_status[dimension] = {"status": "no_valid_samples", "score": None, "note": note}
            for video_path in by_video:
                by_video[video_path][dimension] = no_valid_samples_result(note)
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            run_status[dimension] = {"status": "error", "error": error_text}
            for video_path in by_video:
                by_video[video_path][dimension] = {"status": "error", "score": None, "error": error_text}

    diversity_name = "vbench2_diversity"
    diversity_out = out_dir / "official_runs" / "vbench2" / "Diversity"
    try:
        result = runner.score_batch(
            eval_cases,
            dimension="Diversity",
            output_path=diversity_out,
            run_name=f"batch_Diversity_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        dataset_metrics[diversity_name] = {
            "status": "ok",
            "score": result.get("score"),
            "result_json": result.get("result_json"),
            "full_info_json": result.get("full_info_json"),
            "note": "Official VBench2 Diversity is dataset-level here because the current inputs are one shared prompt with many single outputs, not per-method 20-sample prompt groups.",
        }
        run_status["Diversity"] = {
            "status": "ok",
            "score": result.get("score"),
            "result_json": result.get("result_json"),
            "full_info_json": result.get("full_info_json"),
        }
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        dataset_metrics[diversity_name] = {
            "status": "error",
            "score": None,
            "error": error_text,
            "note": "Official VBench2 Diversity was attempted as a dataset-level metric over the shared caption group.",
        }
        run_status["Diversity"] = {"status": "error", "error": error_text}

    return by_video, run_status, dataset_metrics


def relpath(target: Path, start: Path) -> str:
    return os.path.relpath(target.resolve(), start.resolve()).replace("\\", "/")


def build_summary(
    cases: list[CaseRecord],
    out_dir: Path,
    *,
    skipped_cases: list[SkippedCase],
    vbench_by_video: dict[str, dict[str, dict[str, Any]]],
    vbench_status: dict[str, Any],
    vbench2_by_video: dict[str, dict[str, dict[str, Any]]],
    vbench2_status: dict[str, Any],
    dataset_metrics: dict[str, Any],
) -> dict[str, Any]:
    portal_dir = out_dir / "portal"
    records: list[dict[str, Any]] = []
    for case in cases:
        video_key = str(case.video_path)
        vbench = vbench_by_video.get(video_key, {dimension: base_metric_result() for dimension in VBENCH_DIMENSIONS})
        vbench2 = vbench2_by_video.get(
            video_key,
            {dimension: base_metric_result() for dimension in VBENCH2_PER_VIDEO_DIMENSIONS},
        )
        errors: list[str] = []
        for name, item in vbench.items():
            if item.get("status") == "error":
                errors.append(f"VBench {name}: {item.get('error')}")
        for name, item in vbench2.items():
            if item.get("status") == "error":
                errors.append(f"VBench2 {name}: {item.get('error')}")
        records.append(
            {
                "json_path": str(case.json_path),
                "video_path": str(case.video_path),
                "json_rel": relpath(case.json_path, portal_dir),
                "video_rel": relpath(case.video_path, portal_dir),
                "case_rel": str(case.json_path.relative_to(case.json_path.parents[2])).replace("\\", "/"),
                "group": case.group,
                "method": case.method,
                "caption": case.caption,
                "existing_metrics": extract_existing_metrics(case.payload),
                "vbench": vbench,
                "vbench2": vbench2,
                "errors": errors,
            }
        )

    return {
        "root": str(cases[0].json_path.parents[2]) if cases else "",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "skipped_case_count": len(skipped_cases),
        "skipped_cases": [
            {
                "json_path": str(item.json_path),
                "reason": item.reason,
                "detail": item.detail,
            }
            for item in skipped_cases
        ],
        "records": records,
        "dimension_runs": {
            "vbench": vbench_status,
            "vbench2": vbench2_status,
        },
        "dataset_metrics": dataset_metrics,
    }


def merge_with_existing_summary(summary: dict[str, Any], existing_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not existing_summary:
        return summary

    prior_records = {
        str(record.get("json_path")): record
        for record in existing_summary.get("records", [])
        if isinstance(record, dict) and record.get("json_path")
    }
    for record in summary.get("records", []):
        prior = prior_records.get(str(record.get("json_path")))
        if not prior:
            continue

        if isinstance(prior.get("vbench"), dict):
            merged_vbench = dict(prior["vbench"])
            for key, value in record.get("vbench", {}).items():
                if should_preserve_prior_result(merged_vbench.get(key), value):
                    continue
                merged_vbench[key] = value
            record["vbench"] = merged_vbench

        if isinstance(prior.get("vbench2"), dict):
            merged_vbench2 = dict(prior["vbench2"])
            for key, value in record.get("vbench2", {}).items():
                if should_preserve_prior_result(merged_vbench2.get(key), value):
                    continue
                merged_vbench2[key] = value
            record["vbench2"] = merged_vbench2

        prior_errors = prior.get("errors") if isinstance(prior.get("errors"), list) else []
        current_errors = record.get("errors") if isinstance(record.get("errors"), list) else []
        deduped_errors = list(dict.fromkeys([*prior_errors, *current_errors]))
        record["errors"] = deduped_errors

    prior_runs = existing_summary.get("dimension_runs")
    if isinstance(prior_runs, dict):
        merged_runs = {}
        for bucket_name in ("vbench", "vbench2"):
            merged_bucket = {}
            if isinstance(prior_runs.get(bucket_name), dict):
                merged_bucket.update(prior_runs[bucket_name])
            if isinstance(summary.get("dimension_runs", {}).get(bucket_name), dict):
                for key, value in summary["dimension_runs"][bucket_name].items():
                    if should_preserve_prior_result(merged_bucket.get(key), value):
                        continue
                    merged_bucket[key] = value
            merged_runs[bucket_name] = merged_bucket
        summary["dimension_runs"] = merged_runs

    prior_dataset_metrics = existing_summary.get("dataset_metrics")
    if isinstance(prior_dataset_metrics, dict):
        merged_dataset_metrics = dict(prior_dataset_metrics)
        merged_dataset_metrics.update(summary.get("dataset_metrics", {}))
        summary["dataset_metrics"] = merged_dataset_metrics

    return summary


def render_portal(summary_json: Path, out_dir: Path) -> None:
    from .render_v2v_vbench_portal import main as render_main
    import sys

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "render_v2v_vbench_portal.py",
            "--summary-json",
            str(summary_json),
            "--out-dir",
            str(out_dir / "portal"),
        ]
        render_main()
    finally:
        sys.argv = argv_backup


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "summary.json"
    existing_summary = None
    if summary_json.is_file():
        existing_summary = json.loads(summary_json.read_text(encoding="utf-8"))

    if args.render_only:
        if not summary_json.is_file():
            raise FileNotFoundError(f"Summary not found for render-only mode: {summary_json}")
        render_portal(summary_json, args.out_dir)
        return

    ensure_runtime_env(args.gpu)
    cases, skipped_cases = scan_cases(args.root, args.json_name, args.limit)
    vbench_by_video = {str(case.video_path): {} for case in cases}
    vbench_status: dict[str, Any] = {}
    vbench2_by_video = {str(case.video_path): {} for case in cases}
    vbench2_status: dict[str, Any] = {}
    dataset_metrics: dict[str, Any] = {}

    effective_local_ckpt = not args.disable_local_ckpt
    if not args.skip_vbench:
        vbench_by_video, vbench_status = run_vbench_dimensions(
            cases,
            args.out_dir,
            load_ckpt_from_local=effective_local_ckpt,
            read_frame=args.read_frame,
            imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
        )
    else:
        vbench_status = {dimension: {"status": "skipped"} for dimension in VBENCH_DIMENSIONS}
        vbench_by_video = {
            str(case.video_path): {dimension: {"status": "skipped", "score": None} for dimension in VBENCH_DIMENSIONS}
            for case in cases
        }

    if not args.skip_vbench2:
        vbench2_by_video, vbench2_status, dataset_metrics = run_vbench2_dimensions(
            cases,
            args.out_dir,
            load_ckpt_from_local=effective_local_ckpt,
            read_frame=args.read_frame,
        )
    else:
        vbench2_status = {
            dimension: {"status": "skipped"} for dimension in [*VBENCH2_PER_VIDEO_DIMENSIONS, *VBENCH2_DATASET_DIMENSIONS]
        }
        vbench2_by_video = {
            str(case.video_path): {
                dimension: {"status": "skipped", "score": None}
                for dimension in VBENCH2_PER_VIDEO_DIMENSIONS
            }
            for case in cases
        }

    summary = build_summary(
        cases,
        args.out_dir,
        skipped_cases=skipped_cases,
        vbench_by_video=vbench_by_video,
        vbench_status=vbench_status,
        vbench2_by_video=vbench2_by_video,
        vbench2_status=vbench2_status,
        dataset_metrics=dataset_metrics,
    )
    summary = merge_with_existing_summary(summary, existing_summary)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_portal(summary_json, args.out_dir)


if __name__ == "__main__":
    main()
