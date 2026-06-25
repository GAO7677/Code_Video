from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
for path in [ROOT, TRY0526_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from physv_eval.single_case.ball_block import score_case as score_ball_block
from physv_eval.single_case.cosmos_reason1 import score_case as score_cosmos_reason1
from physv_eval.single_case.pdi import score_case as score_pdi
from physv_eval.single_case.phyground import score_case as score_phyground
from physv_eval.single_case.proxy import score_case as score_proxy
from physv_eval.single_case.videophy2 import score_case as score_videophy2
from physv_eval.single_case.wmreward import score_case as score_wmreward


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
DEFAULT_INPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons")


MetricFunc = Callable[[dict[str, Any], Path, Path], dict[str, Any] | None]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    field: str
    runner: MetricFunc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate generated video results under a directory and "
            "write metrics back into each result json."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help="Root directory containing generated result json/mp4 files.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory used to resolve input_json paths when they are relative.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help="Optional summary json path. Defaults to <result-root>/eval_summary.json.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos_reason1", "ball_block"],
        choices=["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos_reason1", "ball_block"],
        help="Metric names to evaluate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing metric fields in result json files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be evaluated, do not write files.",
    )
    parser.add_argument(
        "--videophy2-task",
        default="pc",
        choices=["sa", "pc", "rule"],
        help="Task used for VideoPhy-2 evaluation.",
    )
    parser.add_argument(
        "--videophy2-caption",
        default=None,
        help="Optional caption override for VideoPhy-2 when input json lacks a usable caption.",
    )
    parser.add_argument(
        "--phyground-general-only",
        action="store_true",
        help="Run only the general subset for PhyGround.",
    )
    parser.add_argument(
        "--pdi-caption",
        default="ball",
        help="Caption used by ball_block/PDI-style scoring when needed.",
    )
    return parser.parse_args()


def build_metric_specs(args: argparse.Namespace) -> list[MetricSpec]:
    def run_pdi_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path, gt_video_path
        caption = case.get("input_caption") or case.get("caption") or args.pdi_caption
        return score_pdi(case, text_query=caption)

    def run_wmreward_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path, gt_video_path
        return score_wmreward(case)

    def run_proxy_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path
        case = dict(case)
        case["context_video"] = str(gt_video_path)
        return score_proxy(case, context_video_path=gt_video_path)

    def run_videophy2_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path, gt_video_path
        payload = case
        caption = payload.get("input_caption") or payload.get("caption") or args.videophy2_caption
        rule = payload.get("rule") or payload.get("physical_law") or payload.get("law")
        return score_videophy2(case, task=args.videophy2_task, caption=caption, rule=rule)

    def run_phyground_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path, gt_video_path
        payload = case
        caption = payload.get("input_caption") or payload.get("caption")
        metrics = None
        laws = [] if args.phyground_general_only else None
        return score_phyground(case, caption=caption, metrics=metrics, laws=laws)

    def run_cosmos_reason1_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path, gt_video_path
        return score_cosmos_reason1(case)

    def run_ball_block_metric(case: dict[str, Any], result_json_path: Path, gt_video_path: Path) -> dict[str, Any] | None:
        del result_json_path, gt_video_path
        payload = case
        caption = payload.get("input_caption") or payload.get("caption") or args.pdi_caption
        return score_ball_block(case, caption=caption)

    runners: dict[str, MetricFunc] = {
        "pdi": run_pdi_metric,
        "wmreward": run_wmreward_metric,
        "proxy": run_proxy_metric,
        "videophy2": run_videophy2_metric,
        "phyground": run_phyground_metric,
        "cosmos_reason1": run_cosmos_reason1_metric,
        "ball_block": run_ball_block_metric,
    }

    return [MetricSpec(name=name, field=name, runner=runners[name]) for name in args.metrics]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def round_floats(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {key: round_floats(item, ndigits=ndigits) for key, item in value.items()}
    if isinstance(value, list):
        return [round_floats(item, ndigits=ndigits) for item in value]
    if isinstance(value, tuple):
        return [round_floats(item, ndigits=ndigits) for item in value]
    return value


def resolve_input_json_path(result_payload: dict[str, Any], result_json_path: Path, input_root: Path) -> Path:
    input_json = result_payload.get("input_json")
    if not isinstance(input_json, str) or not input_json.strip():
        raise ValueError(f"Missing input_json in {result_json_path}")
    candidate = Path(input_json).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        alt = (input_root / candidate).resolve()
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"Cannot resolve input_json for {result_json_path}: {input_json}")


def resolve_gt_video_path(input_json_path: Path, _result_payload: dict[str, Any], input_root: Path) -> Path:
    source_payload = load_json(input_json_path)
    source_video = source_payload.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError(f"Missing source_video in source json: {input_json_path}")
    candidate = Path(source_video).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        alt = (input_root / candidate).resolve()
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"Cannot resolve source_video from {input_json_path}: {source_video}")


def collect_result_jsons(result_root: Path) -> list[Path]:
    json_paths: list[Path] = []
    for path in sorted(result_root.rglob("*.json")):
        if path.name in {"summary.json", "batch_manifest.json", "eval_summary.json"}:
            continue
        json_paths.append(path)
    return json_paths


def metric_summary_record(metric_outputs: dict[str, dict[str, Any] | None]) -> dict[str, Any]:
    summary: dict[str, Any] = {"metrics": {}}
    for name, output in metric_outputs.items():
        summary["metrics"][name] = round_floats(output)
    return summary


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    summary_path = args.output_summary.expanduser().resolve() if args.output_summary is not None else result_root / "eval_summary.json"
    metric_specs = build_metric_specs(args)

    result_json_paths = collect_result_jsons(result_root)
    summary_entries: list[dict[str, Any]] = []
    num_scored = 0
    num_failed = 0

    for result_json_path in result_json_paths:
        try:
            result_payload = load_json(result_json_path)
            if not isinstance(result_payload.get("input_json"), str):
                continue
            input_json_path = resolve_input_json_path(result_payload, result_json_path, input_root)
            gt_video_path = resolve_gt_video_path(input_json_path, result_payload, input_root)
            candidate_video_path = result_json_path.with_suffix(".mp4")
            if not candidate_video_path.is_file():
                candidate_video = result_payload.get("output_video")
                if isinstance(candidate_video, str) and candidate_video.strip():
                    candidate_video_path = Path(candidate_video).expanduser().resolve()
            if not candidate_video_path.is_file():
                raise FileNotFoundError(f"Missing candidate video for {result_json_path}")

            case_payload = dict(result_payload)
            case_payload["video"] = str(candidate_video_path)
            case_payload["context_video"] = str(gt_video_path)
            metric_outputs: dict[str, dict[str, Any] | None] = {}
            for spec in metric_specs:
                if not args.overwrite and spec.field in result_payload:
                    metric_outputs[spec.name] = result_payload.get(spec.field)  # type: ignore[assignment]
                    continue
                metric_outputs[spec.name] = spec.runner(case_payload, result_json_path, gt_video_path)
                if metric_outputs[spec.name] is not None:
                    result_payload[spec.field] = round_floats(metric_outputs[spec.name])

            result_payload["gt_video"] = str(gt_video_path)
            result_payload["eval_metrics"] = metric_summary_record(metric_outputs)
            if not args.dry_run:
                write_json(result_json_path, result_payload)
            summary_entries.append(
                {
                    "result_json": str(result_json_path),
                    "candidate_video": str(candidate_video_path),
                    "gt_video": str(gt_video_path),
                    "input_json": str(input_json_path),
                    "metrics": metric_outputs,
                }
            )
            num_scored += 1
        except Exception as exc:
            num_failed += 1
            summary_entries.append(
                {
                    "result_json": str(result_json_path),
                    "error": str(exc),
                }
            )

    summary_payload = {
        "result_root": str(result_root),
        "input_root": str(input_root),
        "num_result_jsons": len(result_json_paths),
        "num_scored": num_scored,
        "num_failed": num_failed,
        "metrics": [spec.name for spec in metric_specs],
        "entries": summary_entries,
    }
    if not args.dry_run:
        write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
