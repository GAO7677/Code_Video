#!/usr/bin/env python3
"""
Usage:
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/bench_ti2v_t2v.py \
    --metric physics_iq_with_context \
    --result-root /data/gaoya/AAA_test_video/0623/test/ti2v \
    --limit 1

Notes:
  - This script is dedicated to train0705 t2v/ti2v result folders.
  - It does not infer pipe inputs from the source case json.
  - Only input fields that already exist in each result json are passed through.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gc
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

from physv_eval.paths import VPHY_PYTHON

ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
for path in (ROOT, TRY0526_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/ti2v")
DEFAULT_PHYSICS_IQ_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/train0705_ti2v_t2v_metrics/physics_iq_single_case"
)
DEFAULT_PMF_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/train0705_ti2v_t2v_metrics/pmf_single_case"
)
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}


@dataclass
class CaseRecord:
    result_json_path: Path
    result_payload: dict[str, Any]
    input_json_path: Path
    input_payload: dict[str, Any]
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
            "Batch-evaluate one metric over train0705 t2v/ti2v result jsons, "
            "using only the real input_* fields already present in each result json."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument(
        "--metric",
        required=True,
        choices=[
            "wmreward",
            "videophy2",
            "cosmos_reason1",
            "physics_iq",
            "physics_iq_with_context",
            "pmf_with_context",
        ],
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--videophy2-task", default="pc", choices=["sa", "pc", "rule"])
    parser.add_argument("--videophy2-caption", default=None)
    parser.add_argument("--wmreward-reset-interval", type=int, default=16)
    parser.add_argument("--physics-iq-output-root", type=Path, default=DEFAULT_PHYSICS_IQ_OUTPUT_ROOT)
    parser.add_argument("--pmf-output-root", type=Path, default=DEFAULT_PMF_OUTPUT_ROOT)
    parser.add_argument("--pmf-device", default="cpu")
    parser.add_argument("--physics-iq-threshold-value", type=int, default=10)
    parser.add_argument("--physics-iq-downsample-factor", type=int, default=4)
    parser.add_argument("--cosmos-python", type=Path, default=VPHY_PYTHON, help=argparse.SUPPRESS)
    parser.add_argument("--cosmos-worker", action="store_true", help=argparse.SUPPRESS)
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
        input_json = result_payload.get("case_json")
    if not isinstance(input_json, str) or not input_json.strip():
        raise ValueError(f"Missing input_json/case_json in {result_json_path}")
    candidate = Path(input_json).expanduser().resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Cannot resolve input_json for {result_json_path}: {input_json}")


def resolve_gt_video_path(input_json_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = load_json(input_json_path)
    source_video = payload.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError(f"Missing source_video in source json: {input_json_path}")
    candidate = Path(source_video).expanduser().resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Cannot resolve source_video from {input_json_path}: {source_video}")
    return candidate, payload


def resolve_candidate_video_path(result_json_path: Path, result_payload: dict[str, Any]) -> Path:
    candidate_video_path = result_json_path.with_suffix(".mp4")
    if candidate_video_path.is_file():
        return candidate_video_path.resolve()
    output_video = result_payload.get("output_video") or result_payload.get("video_path")
    if isinstance(output_video, str) and output_video.strip():
        candidate = Path(output_video).expanduser().resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing candidate video for {result_json_path}")


def collect_result_jsons(result_root: Path) -> list[Path]:
    result_jsons: list[Path] = []
    for dirpath, _, filenames in os.walk(result_root, followlinks=True):
        current_dir = Path(dirpath)
        for filename in filenames:
            if not filename.endswith(".json"):
                continue
            if filename in EXCLUDED_JSON_NAMES or filename.startswith("eval_summary_"):
                continue
            result_jsons.append((current_dir / filename).resolve())
    return sorted(result_jsons)


def derive_method_name(result_payload: dict[str, Any], fallback_video_path: Path | None = None) -> str | None:
    method = result_payload.get("method")
    if isinstance(method, str) and method.strip():
        return method.strip()

    ckpt = result_payload.get("ckpt")
    if isinstance(ckpt, str) and ckpt.strip():
        ckpt_path = Path(ckpt).expanduser()
        if ckpt_path.name.startswith("step-") and ckpt_path.parent.name == "checkpoints":
            root_name = ckpt_path.parent.parent.name or "model"
            return f"{root_name}_{ckpt_path.name}"

    model_preset = result_payload.get("model_preset")
    if isinstance(model_preset, str) and model_preset.strip():
        return model_preset.strip()

    if fallback_video_path is not None and fallback_video_path.parent.name:
        return fallback_video_path.parent.name
    return None


def cleanup_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload.pop("eval_metrics", None)
    payload.pop("gt_video", None)
    return payload


def apply_payload_defaults(payload: dict[str, Any], *, candidate_video_path: Path) -> dict[str, Any]:
    payload = cleanup_result_payload(payload)
    method = derive_method_name(payload, fallback_video_path=candidate_video_path)
    if method is not None and not (isinstance(payload.get("method"), str) and payload["method"].strip()):
        payload["method"] = method
    if not (isinstance(payload.get("output_video"), str) and payload["output_video"].strip()):
        payload["output_video"] = str(candidate_video_path)
    if not (isinstance(payload.get("video_path"), str) and payload["video_path"].strip()):
        payload["video_path"] = str(candidate_video_path)
    return payload


def metric_already_completed(payload: dict[str, Any], field: str) -> bool:
    return field in payload and payload.get(field) is not None


def resolve_caption(result_payload: dict[str, Any], input_payload: dict[str, Any]) -> str | None:
    for source in (result_payload, input_payload):
        for key in ("input_caption", "caption", "prompt", "input_prompt"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def resolve_rule(result_payload: dict[str, Any], input_payload: dict[str, Any]) -> str | None:
    for source in (result_payload, input_payload):
        for key in ("rule", "physical_law", "law"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def build_case_payload(record: CaseRecord) -> dict[str, Any]:
    result_payload = record.result_payload
    caption = resolve_caption(result_payload, record.input_payload)
    rule = resolve_rule(result_payload, record.input_payload)

    payload: dict[str, Any] = {
        "video": str(record.candidate_video_path),
        "video_path": str(record.candidate_video_path),
        "output_video": str(record.candidate_video_path),
        "source_video": str(record.gt_video_path),
        "input_json": str(record.input_json_path),
        "case_json": str(record.input_json_path),
        "mode": result_payload.get("mode"),
        "conditioning_mode": result_payload.get("conditioning_mode"),
        "context_frames": result_payload.get("context_frames"),
        "used_context_frames": result_payload.get("used_context_frames"),
        "method": derive_method_name(result_payload, fallback_video_path=record.candidate_video_path),
    }

    if caption is not None:
        payload["input_caption"] = caption
        payload["caption"] = caption
        payload["prompt"] = caption
    if rule is not None:
        payload["rule"] = rule
        payload["physical_law"] = rule
        payload["law"] = rule

    input_image = result_payload.get("input_image")
    if isinstance(input_image, str) and input_image.strip():
        payload["input_image"] = input_image.strip()

    input_video = result_payload.get("input_video")
    if isinstance(input_video, str) and input_video.strip():
        payload["input_video"] = input_video.strip()
        payload["context_video"] = input_video.strip()

    for key in (
        "height",
        "width",
        "fps",
        "seed",
        "step",
        "cfg_scale",
        "guidance",
        "num_frames_requested",
        "num_frames_generated",
        "num_inference_steps",
        "model_preset",
        "wan_root",
        "lora_path",
    ):
        if key in result_payload:
            payload[key] = result_payload[key]
    return payload


def sanitize_metric_value(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return round_floats(value)


def maybe_delegate_cosmos_metric(args: argparse.Namespace) -> bool:
    if args.metric != "cosmos_reason1":
        return False
    if args.cosmos_worker:
        return False

    cosmos_python = args.cosmos_python.expanduser().resolve()
    cmd = [
        str(cosmos_python),
        str(Path(__file__).resolve()),
        "--metric",
        args.metric,
        "--result-root",
        str(args.result_root.expanduser().resolve()),
        "--cosmos-worker",
    ]
    if args.output_summary is not None:
        cmd.extend(["--output-summary", str(args.output_summary.expanduser().resolve())])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.limit is not None:
        cmd.extend(["--limit", str(int(args.limit))])
    if int(args.num_shards) > 1:
        cmd.extend(["--num-shards", str(int(args.num_shards)), "--shard-index", str(int(args.shard_index))])

    env = os.environ.copy()
    pythonpath_entries = [str(ROOT), str(TRY0526_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    env["PYTHONNOUSERSITE"] = "1"

    print(f"[ti2v_t2v_metric:delegate] metric=cosmos_reason1 python={cosmos_python}")
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    return True


def build_metric_spec(args: argparse.Namespace) -> MetricSpec:
    def build_method_case_dir(base_root: Path, record: CaseRecord, metric_name: str | None = None) -> Path:
        method_name = derive_method_name(record.result_payload, fallback_video_path=record.candidate_video_path)
        method_dir = method_name if method_name else record.result_json_path.stem
        path = base_root
        if metric_name is not None:
            path = path / metric_name
        return path / method_dir / record.input_json_path.stem

    def build_wmreward(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.wmreward import score_case as score_wmreward_case
        from physv_eval.wmreward_official import WMRewardRunner

        reset_interval = max(1, int(args.wmreward_reset_interval))
        runner: WMRewardRunner | None = None
        cases_since_reset = 0

        def post_case_cleanup(active_runner: WMRewardRunner | None) -> None:
            if active_runner is None:
                return
            torch_module = getattr(active_runner, "_torch", None)
            if torch_module is not None and torch_module.cuda.is_available():
                try:
                    torch_module.cuda.empty_cache()
                except Exception:
                    pass
            gc.collect()

        def cleanup_runner(active_runner: WMRewardRunner | None) -> None:
            if active_runner is None:
                return
            models = getattr(active_runner, "_models", None)
            if isinstance(models, tuple):
                for model in models[:3]:
                    if hasattr(model, "cpu"):
                        try:
                            model.cpu()
                        except Exception:
                            pass
            if hasattr(active_runner, "_models"):
                active_runner._models = None
            post_case_cleanup(active_runner)

        def run(record: CaseRecord) -> dict[str, Any] | None:
            nonlocal runner, cases_since_reset
            if runner is None or cases_since_reset >= reset_interval:
                cleanup_runner(runner)
                runner = WMRewardRunner()
                cases_since_reset = 0
            try:
                return score_wmreward_case(build_case_payload(record), runner=runner)
            finally:
                cases_since_reset += 1
                post_case_cleanup(runner)

        return run

    def build_videophy2(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.videophy2 import score_case as score_videophy2_case
        from physv_eval.videophy2_auto import VideoPhy2Runner

        runner = VideoPhy2Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            caption = case.get("input_caption") or args.videophy2_caption
            rule = case.get("rule") or case.get("physical_law") or case.get("law")
            return score_videophy2_case(
                case,
                task=args.videophy2_task,
                caption=caption,
                rule=rule,
                runner=runner,
            )

        return run

    def build_cosmos_reason1(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.cosmos_reason1 import score_case as score_cosmos_reason1_case
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

        runner = OfficialCosmosReason1Runner()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            return score_cosmos_reason1_case(build_case_payload(record), runner=runner)

        return run

    def build_physics_iq(metric_name: str, context_mode: str | None) -> Callable[[argparse.Namespace], MetricFunc]:
        def factory(_: argparse.Namespace) -> MetricFunc:
            from physv_eval.single_case.physics_iq import score_case as score_physics_iq_case

            physics_iq_output_root = args.physics_iq_output_root.expanduser().resolve()

            def run(record: CaseRecord) -> dict[str, Any] | None:
                case = build_case_payload(record)
                aligned_video_dir = build_method_case_dir(physics_iq_output_root, record, metric_name)
                kwargs: dict[str, Any] = {
                    "source_video_path": record.gt_video_path,
                    "threshold_value": int(args.physics_iq_threshold_value),
                    "downsample_factor": int(args.physics_iq_downsample_factor),
                    "aligned_video_dir": aligned_video_dir,
                }
                if context_mode is not None:
                    kwargs["context_mode"] = context_mode
                return score_physics_iq_case(case, **kwargs)

            return run

        return factory

    def build_pmf_with_context(_: argparse.Namespace) -> MetricFunc:
        from physv_eval.single_case.pmf import score_case as score_pmf_case

        pmf_output_root = args.pmf_output_root.expanduser().resolve()

        def run(record: CaseRecord) -> dict[str, Any] | None:
            case = build_case_payload(record)
            aligned_video_dir = build_method_case_dir(pmf_output_root, record, "pmf_with_context")
            return score_pmf_case(
                case,
                source_video_path=record.gt_video_path,
                context_mode="with_context",
                device=str(args.pmf_device),
                aligned_video_dir=aligned_video_dir,
            )

        return run

    builders: dict[str, Callable[[argparse.Namespace], MetricFunc]] = {
        "wmreward": build_wmreward,
        "videophy2": build_videophy2,
        "cosmos_reason1": build_cosmos_reason1,
        "physics_iq": build_physics_iq("physics_iq", None),
        "physics_iq_with_context": build_physics_iq("physics_iq_with_context", "with_context"),
        "pmf_with_context": build_pmf_with_context,
    }
    return MetricSpec(name=args.metric, field=args.metric, builder=builders[args.metric])


def prepare_cases(result_root: Path, limit: int | None = None) -> tuple[list[CaseRecord], list[dict[str, Any]]]:
    cases: list[CaseRecord] = []
    errors: list[dict[str, Any]] = []

    for result_json_path in collect_result_jsons(result_root):
        try:
            result_payload = load_json(result_json_path)
            if not (
                isinstance(result_payload.get("input_json"), str)
                or isinstance(result_payload.get("case_json"), str)
            ):
                continue
            input_json_path = resolve_input_json_path(result_payload, result_json_path)
            gt_video_path, input_payload = resolve_gt_video_path(input_json_path)
            candidate_video_path = resolve_candidate_video_path(result_json_path, result_payload)
            cases.append(
                CaseRecord(
                    result_json_path=result_json_path,
                    result_payload=result_payload,
                    input_json_path=input_json_path,
                    input_payload=input_payload,
                    gt_video_path=gt_video_path,
                    candidate_video_path=candidate_video_path,
                )
            )
            if limit is not None and len(cases) >= max(0, int(limit)):
                break
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
    args: argparse.Namespace,
    result_root: Path,
    metric_spec: MetricSpec,
    metric_status: dict[str, Any],
    errors: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    summary_payload = {
        "result_root": str(result_root),
        "metric": metric_spec.name,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "limit": args.limit,
        "metric_status": round_floats(metric_status),
        "errors": errors,
    }
    if not dry_run:
        write_json(summary_path, summary_payload)
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if int(args.num_shards) <= 0:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard-index < num-shards, got "
            f"shard-index={args.shard_index}, num-shards={args.num_shards}"
        )
    if maybe_delegate_cosmos_metric(args):
        return

    result_root = args.result_root.expanduser().resolve()
    summary_path = (
        args.output_summary.expanduser().resolve()
        if args.output_summary is not None
        else result_root / f"eval_summary_{args.metric}.json"
    )
    metric_spec = build_metric_spec(args)

    cases, errors = prepare_cases(result_root, limit=args.limit)
    if int(args.num_shards) > 1:
        cases = [
            record
            for case_index, record in enumerate(cases)
            if case_index % int(args.num_shards) == int(args.shard_index)
        ]

    if not args.dry_run:
        write_json(summary_path, {})

    print(
        f"[ti2v_t2v_metric:start] metric={metric_spec.name} cases={len(cases)} "
        f"shard={int(args.shard_index) + 1}/{int(args.num_shards)} result_root={result_root}"
    )
    runner = metric_spec.builder(args)
    num_success = 0
    num_failed = 0
    metric_status: dict[str, Any] = {}

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
                    print(
                        f"[ti2v_t2v_metric:skip] metric={metric_spec.name} "
                        f"{index}/{len(cases)} json={record.result_json_path}"
                    )
                    num_success += 1
                    metric_status = {
                        "num_cases": len(cases),
                        "num_success": num_success,
                        "num_failed": num_failed,
                        "completed": index,
                    }
                    write_summary(
                        summary_path,
                        args=args,
                        result_root=result_root,
                        metric_spec=metric_spec,
                        metric_status=metric_status,
                        errors=errors,
                        dry_run=args.dry_run,
                    )
                    continue

            metric_value = sanitize_metric_value(runner(record))

            with locked_result_json(record.result_json_path):
                latest_payload = load_json(record.result_json_path)
                latest_payload = apply_payload_defaults(
                    copy.deepcopy(latest_payload),
                    candidate_video_path=record.candidate_video_path,
                )
                if not args.overwrite and metric_already_completed(latest_payload, metric_spec.field):
                    if not args.dry_run:
                        write_json(record.result_json_path, latest_payload)
                    print(
                        f"[ti2v_t2v_metric:skip-race] metric={metric_spec.name} "
                        f"{index}/{len(cases)} json={record.result_json_path}"
                    )
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
                    print(
                        f"[ti2v_t2v_metric:done] metric={metric_spec.name} "
                        f"{index}/{len(cases)} json={record.result_json_path}"
                    )
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
            print(
                f"[ti2v_t2v_metric:error] metric={metric_spec.name} "
                f"{index}/{len(cases)} json={record.result_json_path} error={exc}"
            )

        metric_status = {
            "num_cases": len(cases),
            "num_success": num_success,
            "num_failed": num_failed,
            "completed": index,
        }
        write_summary(
            summary_path,
            args=args,
            result_root=result_root,
            metric_spec=metric_spec,
            metric_status=metric_status,
            errors=errors,
            dry_run=args.dry_run,
        )

    print(
        f"[ti2v_t2v_metric:finish] metric={metric_spec.name} "
        f"success={num_success} failed={num_failed} result_root={result_root}"
    )


if __name__ == "__main__":
    main()
