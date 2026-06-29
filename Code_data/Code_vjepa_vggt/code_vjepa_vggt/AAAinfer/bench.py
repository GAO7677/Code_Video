'''
# 评估pdi指标
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric pdi \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v

# 评估wmreward指标
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric wmreward \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v

# 评估单视角近似 Physics-IQ 指标
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py \
  --metric physics_iq \
  --result-root /data/gaoya/AAA_test_video/0623/test/v2v

  
# 一键启动所有指标的评估
CUDA_VISIBLE_DEVICES=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.sh /data/gaoya/AAA_test_video/0623/test/v2v


# 统计并可视化指标报告
/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/render_v2v_metric_report.py --result-root /data/gaoya/AAA_test_video/0623/test/v2v
pyport /data/gaoya/AAA_test_video/0623/test/report/v2v 8893


# 把test_5.txt中的json路径对应的所有方法输出视频复制到output-root中

/home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/export_v2v_case_videos.py \
    --txt-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --output-root /data/gaoya/agent-data/outputs/v2v_case_export_test5




'''
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
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

from physv_eval.paths import FLUX_PYTHON


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
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
            "Batch-evaluate one metric over all result jsons under result-root, "
            "loading one metric model per process and backfilling immediately."
        )
    )
    metric_choices = ["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos_reason1", "physics_iq"]
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--metric", required=True, choices=metric_choices)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--videophy2-task", default="pc", choices=["sa", "pc", "rule"])
    parser.add_argument("--videophy2-caption", default=None)
    parser.add_argument("--phyground-general-only", action="store_true")
    parser.add_argument("--pdi-caption", default="ball")
    parser.add_argument("--flux-python", type=Path, default=FLUX_PYTHON, help=argparse.SUPPRESS)
    parser.add_argument("--cosmos-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--flux-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--physics-iq-output-root",
        type=Path,
        default=Path("/tmp/gaoya/physics_iq_single_case/AAAinfer_bench"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


@contextmanager
def locked_result_json(path: Path):
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def resolve_input_json_path(result_payload: dict[str, Any], result_json_path: Path) -> Path:
    input_json = result_payload.get("input_json")
    if not isinstance(input_json, str) or not input_json.strip():
        raise ValueError(f"Missing input_json in {result_json_path}")
    candidate = Path(input_json).expanduser().resolve()
    if not candidate.is_absolute():
        raise ValueError(f"input_json must be an absolute path in {result_json_path}: {input_json}")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve input_json for {result_json_path}: {input_json}")


def resolve_gt_video_path(input_json_path: Path) -> Path:
    source_payload = load_json(input_json_path)
    source_video = source_payload.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError(f"Missing source_video in source json: {input_json_path}")
    candidate = Path(source_video).expanduser().resolve()
    if not candidate.is_absolute():
        raise ValueError(f"source_video must be an absolute path in {input_json_path}: {source_video}")
    if candidate.is_file():
        return candidate
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
    def normalize_ckpt_method_name(name: str) -> str:
        normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
        return normalized or name

    def derive_method_name_from_ckpt_path(ckpt_path: Path) -> str | None:
        candidate_path = ckpt_path.expanduser()
        if candidate_path.is_file() or candidate_path.suffix:
            step_dir = candidate_path.parent
            if not step_dir.name.startswith("step-"):
                return None
            checkpoint_parent = step_dir.parent
            step_name = step_dir.name
        else:
            step_name = candidate_path.name
            checkpoint_parent = candidate_path.parent
        if not step_name:
            return None
        if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
            method_root = normalize_ckpt_method_name(checkpoint_parent.parent.name)
            return f"{method_root}_{step_name}"
        if checkpoint_parent.name:
            method_root = normalize_ckpt_method_name(checkpoint_parent.name)
            return f"{method_root}_{step_name}"
        return None

    ckpt = result_payload.get("ckpt")
    if isinstance(ckpt, str) and ckpt.strip():
        derived_from_ckpt = derive_method_name_from_ckpt_path(Path(ckpt))
        if derived_from_ckpt is not None:
            return derived_from_ckpt

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
    payload["source_video"] = str(record.gt_video_path)
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


def metric_already_completed(payload: dict[str, Any], field: str) -> bool:
    if field not in payload:
        return False
    return payload.get(field) is not None


def apply_payload_defaults(payload: dict[str, Any], *, candidate_video_path: Path) -> dict[str, Any]:
    existing_method = payload.get("method")
    method = derive_method_name(payload, fallback_video_path=candidate_video_path)
    if method is not None:
        should_replace = not isinstance(existing_method, str) or not existing_method.strip()
        if isinstance(existing_method, str):
            stripped_method = existing_method.strip()
            if re.fullmatch(r"step-\d+", stripped_method):
                should_replace = True
        if should_replace:
            payload["method"] = method
    cleanup_result_payload(payload)
    return payload


def maybe_delegate_flux_metric(args: argparse.Namespace) -> bool:
    if args.metric not in {"phyground", "cosmos_reason1"}:
        return False
    if args.flux_worker:
        return False
    if args.metric == "cosmos_reason1" and args.cosmos_worker:
        return False

    flux_python = args.flux_python.expanduser().resolve()
    cmd = [
        str(flux_python),
        str(Path(__file__).resolve()),
        "--metric",
        args.metric,
        "--result-root",
        str(args.result_root.expanduser().resolve()),
        "--flux-worker",
    ]
    if args.metric == "cosmos_reason1":
        cmd.append("--cosmos-worker")
    if args.output_summary is not None:
        cmd.extend(["--output-summary", str(args.output_summary.expanduser().resolve())])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.metric == "phyground" and args.phyground_general_only:
        cmd.append("--phyground-general-only")

    env = os.environ.copy()
    pythonpath_entries = [str(ROOT), str(TRY0526_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTHONNOUSERSITE"] = "1"

    print(f"[{args.metric}:delegate] python={flux_python}")
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return True


def build_metric_spec(args: argparse.Namespace) -> MetricSpec:
    def build_pdi(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.official_pdi import OfficialPDIRunner
        from physv_eval.single_case.pdi import score_case as score_pdi_case

        runner = OfficialPDIRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.pdi_caption
            return score_pdi_case(case, text_query=caption, runner=runner)

        return run

    def build_wmreward(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.wmreward import score_case as score_wmreward_case
        from physv_eval.wmreward_official import WMRewardRunner

        runner = WMRewardRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            return score_wmreward_case(build_case_payload(record), runner=runner)

        return run

    def build_proxy(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.proxy_runner import ProxyRunner
        from physv_eval.single_case.proxy import score_case as score_proxy_case

        runner = ProxyRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            return score_proxy_case(case, context_video_path=record.gt_video_path, runner=runner)

        return run

    def build_videophy2(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.videophy2 import score_case as score_videophy2_case
        from physv_eval.videophy2_auto import VideoPhy2Runner

        runner = VideoPhy2Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption") or args.videophy2_caption
            rule = case.get("rule") or case.get("physical_law") or case.get("law")
            return score_videophy2_case(
                case,
                task=args.videophy2_task,
                caption=caption,
                rule=rule,
                runner=runner,
            )

        return run

    def build_phyground(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.phyground_official import OfficialPhyGroundRunner
        from physv_eval.single_case.phyground import score_case as score_phyground_case

        runner = OfficialPhyGroundRunner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or case.get("caption")
            metrics = None
            laws = [] if args.phyground_general_only else None
            return score_phyground_case(case, caption=caption, metrics=metrics, laws=laws, runner=runner)

        return run

    def build_cosmos_reason1(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner
        from physv_eval.single_case.cosmos_reason1 import score_case as score_cosmos_reason1_case

        runner = OfficialCosmosReason1Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            return score_cosmos_reason1_case(build_case_payload(record), runner=runner)

        return run

    def build_physics_iq(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.physics_iq import score_case as score_physics_iq_case

        physics_iq_output_root = args.physics_iq_output_root.expanduser().resolve()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            aligned_video_dir = (
                physics_iq_output_root
                / derive_method_name(record.result_payload, fallback_video_path=record.candidate_video_path)
                if derive_method_name(record.result_payload, fallback_video_path=record.candidate_video_path)
                else physics_iq_output_root / record.result_json_path.stem
            ) / record.input_json_path.stem
            return score_physics_iq_case(
                case,
                source_video_path=record.gt_video_path,
                aligned_video_dir=aligned_video_dir,
            )

        return run

    builders: dict[str, Callable[[argparse.Namespace], MetricFunc]] = {
        "pdi": build_pdi,
        "wmreward": build_wmreward,
        "proxy": build_proxy,
        "videophy2": build_videophy2,
        "phyground": build_phyground,
        "cosmos_reason1": build_cosmos_reason1,
        "physics_iq": build_physics_iq,
    }
    return MetricSpec(name=args.metric, field=args.metric, builder=builders[args.metric])


def prepare_cases(result_root: Path) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
    cases: list[CaseRecord] = []
    errors: list[dict[str, Any]] = []
    for result_json_path in collect_result_jsons(result_root):
        try:
            result_payload = load_json(result_json_path)
            if not isinstance(result_payload.get("input_json"), str):
                continue
            input_json_path = resolve_input_json_path(result_payload, result_json_path)
            gt_video_path = resolve_gt_video_path(input_json_path)
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
    metric_spec: MetricSpec,
    cases: list[CaseRecord],
    metric_status: dict[str, Any],
    errors: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    summary_payload = {
        "result_root": str(result_root),
        "num_result_jsons": len(cases),
        "metric": metric_spec.name,
        "metric_status": round_floats(metric_status),
        "errors": errors,
    }
    if not dry_run:
        write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))

def main() -> None:
    args = parse_args()
    if maybe_delegate_flux_metric(args):
        return
    result_root = args.result_root.expanduser().resolve()
    summary_path = (
        args.output_summary.expanduser().resolve()
        if args.output_summary is not None
        else result_root / f"eval_summary_{args.metric}.json"
    )
    metric_spec = build_metric_spec(args)

    cases, errors = prepare_cases(result_root)
    metric_status: dict[str, Any] = {}
    if not args.dry_run:
        write_json(summary_path, {})

    print(f"[metric:start] {metric_spec.name} cases={len(cases)}")
    runner = metric_spec.builder(args)
    num_success = 0
    num_failed = 0
    for index, record in enumerate(cases, start=1):
        try:
            with locked_result_json(record.result_json_path):
                current_payload = load_json(record.result_json_path)
                current_payload = apply_payload_defaults(
                    copy.deepcopy(current_payload),
                    candidate_video_path=record.candidate_video_path,
                )
                if not args.overwrite and metric_already_completed(current_payload, metric_spec.field):
                    if not args.dry_run:
                        write_json(record.result_json_path, current_payload)
                    print(f"[metric:skip] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}")
                    num_success += 1
                    metric_status = {
                        "num_cases": len(cases),
                        "num_success": num_success,
                        "num_failed": num_failed,
                        "completed": index,
                    }
                    write_summary(
                        summary_path,
                        result_root=result_root,
                        metric_spec=metric_spec,
                        cases=cases,
                        metric_status=metric_status,
                        errors=errors,
                        dry_run=args.dry_run,
                    )
                    continue

            metric_value = sanitize_metric_value(metric_spec.name, runner(record))

            with locked_result_json(record.result_json_path):
                latest_payload = load_json(record.result_json_path)
                latest_payload = apply_payload_defaults(
                    copy.deepcopy(latest_payload),
                    candidate_video_path=record.candidate_video_path,
                )
                if not args.overwrite and metric_already_completed(latest_payload, metric_spec.field):
                    if not args.dry_run:
                        write_json(record.result_json_path, latest_payload)
                    print(f"[metric:skip-race] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}")
                    num_success += 1
                else:
                    latest_payload[metric_spec.field] = metric_value
                    latest_payload = apply_payload_defaults(
                        latest_payload,
                        candidate_video_path=record.candidate_video_path,
                    )
                    if not args.dry_run:
                        write_json(record.result_json_path, latest_payload)
                    num_success += 1
                    print(f"[metric:done] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}")
        except Exception as exc:
            num_failed += 1
            errors.append(
                {
                    "metric": metric_spec.name,
                    "result_json": str(record.result_json_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )
            print(f"[metric:error] {metric_spec.name} {index}/{len(cases)} {record.result_json_path.name}: {exc}")
        metric_status = {
            "num_cases": len(cases),
            "num_success": num_success,
            "num_failed": num_failed,
            "completed": index,
        }
        write_summary(
            summary_path,
            result_root=result_root,
            metric_spec=metric_spec,
            cases=cases,
            metric_status=metric_status,
            errors=errors,
            dry_run=args.dry_run,
        )
    print(f"[metric:finish] {metric_spec.name} success={num_success} failed={num_failed}")


if __name__ == "__main__":
    main()
