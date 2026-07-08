from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..vbench_official import OfficialVBenchRunner
from .common import emit_result, load_eval_case, result_record


TABLE1_METRIC_ALIASES = {
    "subj_cons": "subject_consistency",
    "subject_consistency": "subject_consistency",
    "back_cons": "background_consistency",
    "background_consistency": "background_consistency",
    "moti_smoo": "motion_smoothness",
    "motion_smoothness": "motion_smoothness",
    "dyna_degr": "dynamic_degree",
    "dynamic_degree": "dynamic_degree",
    "aest_qual": "aesthetic_quality",
    "aesthetic_quality": "aesthetic_quality",
    "image_qual": "imaging_quality",
    "imaging_quality": "imaging_quality",
    "table1_all": "table1_all",
}

TABLE1_METRIC_ORDER = [
    ("subj_cons", "subject_consistency"),
    ("back_cons", "background_consistency"),
    ("image_qual", "imaging_quality"),
    ("moti_smoo", "motion_smoothness"),
    ("dyna_degr", "dynamic_degree"),
    ("aest_qual", "aesthetic_quality"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case VBench evaluation.")
    parser.add_argument(
        "--dimension",
        required=True,
        help=(
            "VBench dimension, Table-1 alias, or table1_all. "
            "Examples: subject_consistency, image_qual, moti_smoo, table1_all"
        ),
    )
    parser.add_argument("--input-json", type=Path, default=None, help="Case JSON containing video metadata.")
    parser.add_argument("--video", type=Path, default=None, help="Video path for the single case.")
    parser.add_argument("--caption", default=None, help="Optional prompt override for custom_input mode.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--output-path", type=Path, default=None, help="Optional directory for official VBench outputs.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--full-json-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-ckpt-from-local", action="store_true")
    parser.add_argument("--read-frame", action="store_true")
    parser.add_argument(
        "--imaging-quality-preprocessing-mode",
        default="longer",
        choices=["shorter", "longer", "shorter_centercrop", "None"],
    )
    return parser.parse_args()


def resolve_requested_dimension(name: str) -> str:
    key = name.strip().lower()
    if key not in TABLE1_METRIC_ALIASES:
        supported = ", ".join(sorted(TABLE1_METRIC_ALIASES))
        raise ValueError(f"Unsupported VBench/Table-1 dimension {name!r}. Supported names: {supported}")
    return TABLE1_METRIC_ALIASES[key]


def score_case(
    case: Path | str | dict[str, Any],
    *,
    dimension: str,
    caption: str | None = None,
    output_path: Path | None = None,
    runner: OfficialVBenchRunner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialVBenchRunner()
    resolved_dimension = resolve_requested_dimension(dimension)
    if resolved_dimension == "table1_all":
        raise ValueError("Use score_case_many(..., dimensions=['table1_all']) or the CLI with --dimension table1_all.")
    return active_runner.score_case(
        case,
        dimension=resolved_dimension,
        caption=caption,
        output_path=output_path,
    )


def score_case_many(
    case: Path | str | dict[str, Any],
    *,
    dimensions: list[str] | tuple[str, ...],
    caption: str | None = None,
    output_path: Path | None = None,
    runner: OfficialVBenchRunner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialVBenchRunner()
    requested = [resolve_requested_dimension(name) for name in dimensions]
    if requested == ["table1_all"]:
        requested = [dimension for _, dimension in TABLE1_METRIC_ORDER]

    results: dict[str, Any] = {}
    for alias, canonical in TABLE1_METRIC_ORDER:
        if canonical not in requested:
            continue
        results[alias] = active_runner.score_case(
            case,
            dimension=canonical,
            caption=caption,
            output_path=output_path,
        )
    return {
        "requested_dimension": "table1_all" if dimensions == ["table1_all"] else list(dimensions),
        "table1_metrics": results,
    }


def main() -> None:
    args = parse_args()
    case = load_eval_case(input_json=args.input_json, video=args.video, caption=args.caption)
    runner = OfficialVBenchRunner(
        repo_root=args.repo_root,
        full_json_dir=args.full_json_dir,
        device=args.device,
        load_ckpt_from_local=args.load_ckpt_from_local,
        read_frame=args.read_frame,
        imaging_quality_preprocessing_mode=args.imaging_quality_preprocessing_mode,
    )
    resolved_dimension = resolve_requested_dimension(args.dimension)
    if resolved_dimension == "table1_all":
        result = score_case_many(
            case,
            dimensions=["table1_all"],
            caption=args.caption,
            output_path=args.output_path,
            runner=runner,
        )
    else:
        result = score_case(
            case,
            dimension=args.dimension,
            caption=args.caption,
            output_path=args.output_path,
            runner=runner,
        )
    emit_result(result_record(case, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
