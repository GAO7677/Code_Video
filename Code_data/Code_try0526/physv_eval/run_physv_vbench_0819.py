#!/usr/bin/env python3
"""Run the single-case VBench dimensions for every PhysV dataset sample."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_vbench")
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-case-id", action="append", default=[], help="Re-evaluate these case IDs even when a prior score exists.")
    parser.add_argument("--dimensions", nargs="+", choices=DIMENSIONS, default=list(DIMENSIONS))
    parser.add_argument("--load-ckpt-from-local", action="store_true")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument(
        "--imaging-quality-preprocessing-mode",
        default="longer",
        choices=["shorter", "longer", "shorter_centercrop", "None"],
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_cases(dataset_root: Path, limit: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    sample_dirs = sorted(path for path in (dataset_root / "samples").iterdir() if path.is_dir())
    if limit is not None:
        sample_dirs = sample_dirs[:limit]
    for sample_dir in sample_dirs:
        metadata = read_json(sample_dir / "metadata.json")
        video_path = (sample_dir / "videos/rgb.mp4").resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        captions = metadata.get("captions", {})
        specific = captions.get("specific", {}) if isinstance(captions, dict) else {}
        cases.append(
            {
                "case_id": sample_dir.name,
                "video_path": str(video_path),
                "caption": specific.get("text") if isinstance(specific, dict) else None,
                "source_group": metadata.get("source_group", ""),
                "family_key": metadata.get("family_key", ""),
            }
        )
    return cases


def empty_metric() -> dict[str, Any]:
    return {"status": "pending", "score": None}


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    report_path = (args.report_json or (dataset_root / "reports/vbench_metrics.json")).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    # CUDA_VISIBLE_DEVICES is set before importing torch through the VBench runner.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from physv_eval.case_inputs import EvalCase
    from physv_eval.vbench_official import OfficialVBenchRunner

    cases = load_cases(dataset_root, args.limit)
    prior: dict[str, Any] = {}
    if report_path.is_file():
        prior = read_json(report_path)
    case_metrics: dict[str, dict[str, Any]] = {
        case["case_id"]: {
            "video": case["video_path"],
            "source_group": case["source_group"],
            "family_key": case["family_key"],
            "dimensions": {
                dimension: (prior.get("cases", {}).get(case["case_id"], {}).get("dimensions", {}).get(dimension, empty_metric()))
                for dimension in args.dimensions
            },
        }
        for case in cases
    }
    run_status: dict[str, Any] = {
        dimension: prior.get("runs", {}).get(dimension, {"status": "pending"})
        for dimension in args.dimensions
    }
    runner = OfficialVBenchRunner(
        device="cuda",
        output_root=args.output_root,
        load_ckpt_from_local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
    )
    eval_cases = [EvalCase(video_path=Path(case["video_path"]), caption=case["caption"]) for case in cases]
    path_to_case = {str(Path(case["video_path"]).resolve()): case["case_id"] for case in cases}

    for dimension in args.dimensions:
        pending_cases = [
            case for case in cases
            if case["case_id"] in set(args.force_case_id)
            or case_metrics[case["case_id"]]["dimensions"].get(dimension, {}).get("status") != "ok"
        ]
        pending_case_ids = {case["case_id"] for case in pending_cases}
        if not pending_cases:
            print(f"skip {dimension}: complete result already exists", flush=True)
            continue
        print(f"run {dimension}: {len(pending_cases)} pending cases on cuda:{args.gpu}", flush=True)
        try:
            result = runner.score_batch(
                [EvalCase(video_path=Path(case["video_path"]), caption=case["caption"]) for case in pending_cases],
                dimension=dimension,
                output_path=args.output_root / "official_runs" / dimension,
                run_name=f"physv_v2v_0819_{dimension}",
            )
            seen: set[str] = set()
            for item in result.get("raw_results", []):
                video_path = str(Path(str(item.get("video_path"))).resolve())
                case_id = path_to_case.get(video_path)
                if case_id is None:
                    continue
                score = item.get("video_results")
                case_metrics[case_id]["dimensions"][dimension] = {
                    "status": "ok",
                    "score": float(score) if score is not None else None,
                }
                seen.add(case_id)
            for case_id in pending_case_ids:
                if case_id not in seen:
                    case_metrics[case_id]["dimensions"][dimension] = {
                        "status": "error",
                        "score": None,
                        "error": "VBench returned no per-video result",
                    }
            run_status[dimension] = {
                "status": "ok",
                "dataset_score": result.get("score"),
                "result_json": result.get("result_json"),
                "full_info_json": result.get("full_info_json"),
                "output_path": result.get("output_path"),
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"error {dimension}: {error}", flush=True)
            run_status[dimension] = {"status": "error", "error": error}
            for case_id in pending_case_ids:
                case_metrics[case_id]["dimensions"][dimension] = {"status": "error", "score": None, "error": error}
        report = {
            "schema_version": "physv_vbench_metrics_v1",
            "dataset_root": str(dataset_root),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "device": f"cuda:{args.gpu}",
            "dimensions": list(args.dimensions),
            "runs": run_status,
            "cases": case_metrics,
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": "physv_vbench_metrics_v1",
        "dataset_root": str(dataset_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": f"cuda:{args.gpu}",
        "dimensions": list(args.dimensions),
        "runs": run_status,
        "cases": case_metrics,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={report_path}", flush=True)
    print(f"cases={len(case_metrics)}", flush=True)


if __name__ == "__main__":
    main()
