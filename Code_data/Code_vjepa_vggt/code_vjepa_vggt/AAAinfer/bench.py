from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import shutil

ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
for path in [ROOT, TRY0526_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from physv_eval.proxy_runner import ProxyRunner
from physv_eval.official_pdi import OfficialPDIRunner
from physv_eval.wmreward_official import WMRewardRunner
from physv_eval.videophy2_auto import VideoPhy2Runner
from physv_eval.phyground_official import OfficialPhyGroundRunner
from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner
from physv_eval.single_case.ball_block import score_case as score_ball_block
from physv_eval.single_case.ball_block import TMP_DIR as BALL_BLOCK_TMP_DIR


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
DEFAULT_INPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons")


@dataclass
class CaseRecord:
    result_json_path: Path
    result_payload: dict[str, Any]
    input_json_path: Path
    gt_video_path: Path
    candidate_video_path: Path


MetricFunc = Callable[[CaseRecord], dict[str, Any] | None]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    field: str
    builder: Callable[[argparse.Namespace], MetricFunc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-evaluate generated videos with metric-first scheduling: "
            "load one metric model, score all jsons, and backfill immediately."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos_reason1", "ball_block"],
        choices=["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos_reason1", "ball_block"],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--videophy2-task", default="pc", choices=["sa", "pc", "rule"])
    parser.add_argument("--videophy2-caption", default=None)
    parser.add_argument("--phyground-general-only", action="store_true")
    parser.add_argument("--pdi-caption", default="ball")
    return parser.parse_args()


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


def resolve_gt_video_path(input_json_path: Path, input_root: Path) -> Path:
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
    return sorted(
        [
            path
            for path in result_root.rglob("*.json")
            if path.name not in {"summary.json", "batch_manifest.json", "eval_summary.json"}
        ]
    )


def resolve_candidate_video_path(result_json_path: Path, result_payload: dict[str, Any]) -> Path:
    candidate_video_path = result_json_path.with_suffix(".mp4")
    if candidate_video_path.is_file():
        return candidate_video_path.resolve()
    candidate_video = result_payload.get("output_video")
    if isinstance(candidate_video, str) and candidate_video.strip():
        path = Path(candidate_video).expanduser().resolve()
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing candidate video for {result_json_path}")


def derive_method_name(result_payload: dict[str, Any], fallback_video_path: Path | None = None) -> str | None:
    output_video = result_payload.get("output_video")
    if isinstance(output_video, str) and output_video.strip():
        output_video_path = Path(output_video).expanduser()
        if output_video_path.parent.name:
            return output_video_path.parent.name
    if fallback_video_path is not None and fallback_video_path.parent.name:
        return fallback_video_path.parent.name
    return None


def build_case_payload(record: CaseRecord) -> dict[str, Any]:
    payload = dict(record.result_payload)
    payload["video"] = str(record.candidate_video_path)
    payload["context_video"] = str(record.gt_video_path)
    return payload


def sanitize_metric_value(metric_name: str, value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    payload = round_floats(value)
    if metric_name == "pdi":
        for key in [
            "ra_math_pass",
            "ra_ground_rmse",
            "ra_scale_jump",
            "ra_reproj_err",
            "ra_overall_pass",
            "raw_report_path",
        ]:
            payload.pop(key, None)
        raw_report_path = value.get("raw_report_path")
        if isinstance(raw_report_path, str) and raw_report_path:
            report_path = Path(raw_report_path)
            report_dir = report_path.parent
            if report_path.exists():
                report_path.unlink(missing_ok=True)
            if report_dir.exists():
                shutil.rmtree(report_dir, ignore_errors=True)
    return payload


def cleanup_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("eval_metrics", None)
    payload.pop("gt_video", None)
    if isinstance(payload.get("pdi"), dict):
        payload["pdi"] = sanitize_metric_value("pdi", payload["pdi"])
    return payload


def apply_result_defaults(record: CaseRecord) -> None:
    method = derive_method_name(record.result_payload, fallback_video_path=record.candidate_video_path)
    if method is not None:
        record.result_payload["method"] = method
    cleanup_result_payload(record.result_payload)


def build_metric_specs(args: argparse.Namespace) -> list[MetricSpec]:
    def build_pdi(_: argparse.Namespace) -> MetricFunc:
        runner = OfficialPDIRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.pdi_caption
            return runner.run_case(case, text_query=caption)

        return run

    def build_wmreward(_: argparse.Namespace) -> MetricFunc:
        runner = WMRewardRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            return runner.score_case(build_case_payload(record))

        return run

    def build_proxy(_: argparse.Namespace) -> MetricFunc:
        runner = ProxyRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            return runner.score_case(case, context_video_path=record.gt_video_path)

        return run

    def build_videophy2(_: argparse.Namespace) -> MetricFunc:
        runner = VideoPhy2Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.videophy2_caption
            rule = case.get("rule") or case.get("physical_law") or case.get("law")
            return runner.score_case(case, task=args.videophy2_task, caption=caption, rule=rule)

        return run

    def build_phyground(_: argparse.Namespace) -> MetricFunc:
        runner = OfficialPhyGroundRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption")
            metrics = None
            laws = [] if args.phyground_general_only else None
            return runner.score_case(case, caption=caption, metrics=metrics, laws=laws)

        return run

    def build_cosmos_reason1(_: argparse.Namespace) -> MetricFunc:
        runner = OfficialCosmosReason1Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            return runner.score_case(build_case_payload(record))

        return run

    def build_ball_block(_: argparse.Namespace) -> MetricFunc:
        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.pdi_caption
            result = score_ball_block(case, caption=caption)
            shutil.rmtree(BALL_BLOCK_TMP_DIR / "pdi" / record.candidate_video_path.stem, ignore_errors=True)
            shutil.rmtree(BALL_BLOCK_TMP_DIR / "jepa" / record.candidate_video_path.stem, ignore_errors=True)
            return result

        return run

    builders: dict[str, Callable[[argparse.Namespace], MetricFunc]] = {
        "pdi": build_pdi,
        "wmreward": build_wmreward,
        "proxy": build_proxy,
        "videophy2": build_videophy2,
        "phyground": build_phyground,
        "cosmos_reason1": build_cosmos_reason1,
        "ball_block": build_ball_block,
    }
    return [MetricSpec(name=name, field=name, builder=builders[name]) for name in args.metrics]


def prepare_cases(result_root: Path, input_root: Path) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
    cases: list[CaseRecord] = []
    errors: list[dict[str, Any]] = []
    for result_json_path in collect_result_jsons(result_root):
        try:
            result_payload = load_json(result_json_path)
            if not isinstance(result_payload.get("input_json"), str):
                continue
            input_json_path = resolve_input_json_path(result_payload, result_json_path, input_root)
            gt_video_path = resolve_gt_video_path(input_json_path, input_root)
            candidate_video_path = resolve_candidate_video_path(result_json_path, result_payload)
            cases.append(
                CaseRecord(
                    result_json_path=result_json_path,
                    result_payload=result_payload,
                    input_json_path=input_json_path,
                    gt_video_path=gt_video_path,
                    candidate_video_path=candidate_video_path,
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "result_json": str(result_json_path),
                    "stage": "prepare",
                    "error": str(exc),
                }
            )
    return cases, errors


def write_summary(
    summary_path: Path,
    *,
    result_root: Path,
    input_root: Path,
    metric_specs: list[MetricSpec],
    cases: list[CaseRecord],
    metric_status: dict[str, dict[str, Any]],
    errors: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    summary_payload = {
        "result_root": str(result_root),
        "input_root": str(input_root),
        "num_result_jsons": len(cases),
        "metrics": [spec.name for spec in metric_specs],
        "metric_status": round_floats(metric_status),
        "errors": errors,
    }
    if not dry_run:
        write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


def preclean_cases(cases: list[CaseRecord], *, dry_run: bool) -> None:
    for record in cases:
        apply_result_defaults(record)
        if not dry_run:
            write_json(record.result_json_path, record.result_payload)


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    input_root = args.input_root.expanduser().resolve()
    summary_path = args.output_summary.expanduser().resolve() if args.output_summary is not None else result_root / "eval_summary.json"
    metric_specs = build_metric_specs(args)

    cases, errors = prepare_cases(result_root, input_root)
    preclean_cases(cases, dry_run=args.dry_run)
    metric_status: dict[str, dict[str, Any]] = {}
    if not args.dry_run:
        write_json(summary_path, {})

    for spec in metric_specs:
        print(f"[metric:start] {spec.name} cases={len(cases)}")
        runner = spec.builder(args)
        num_success = 0
        num_failed = 0
        for index, record in enumerate(cases, start=1):
            try:
                if not args.overwrite and spec.field in record.result_payload:
                    print(f"[metric:skip] {spec.name} {index}/{len(cases)} {record.result_json_path.name}")
                    num_success += 1
                    continue
                metric_value = sanitize_metric_value(spec.name, runner(record))
                record.result_payload[spec.field] = metric_value
                apply_result_defaults(record)
                if not args.dry_run:
                    write_json(record.result_json_path, record.result_payload)
                num_success += 1
                print(f"[metric:done] {spec.name} {index}/{len(cases)} {record.result_json_path.name}")
            except Exception as exc:
                num_failed += 1
                errors.append(
                    {
                        "metric": spec.name,
                        "result_json": str(record.result_json_path),
                        "error": str(exc),
                        "traceback": traceback.format_exc(limit=3),
                    }
                )
                print(f"[metric:error] {spec.name} {index}/{len(cases)} {record.result_json_path.name}: {exc}")
            metric_status[spec.name] = {
                "num_cases": len(cases),
                "num_success": num_success,
                "num_failed": num_failed,
                "completed": index,
            }
            write_summary(
                summary_path,
                result_root=result_root,
                input_root=input_root,
                metric_specs=metric_specs,
                cases=cases,
                metric_status=metric_status,
                errors=errors,
                dry_run=args.dry_run,
            )
        print(f"[metric:finish] {spec.name} success={num_success} failed={num_failed}")


if __name__ == "__main__":
    main()
