from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


"""
Examples

Run the default metric set and backfill every `video.json` under the run root:
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/bench.py \
    --root /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff

Run only WMReward and PDI for the first case:
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/bench.py \
    --root /data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff \
    --metrics wmreward pdi \
    --limit 1
"""


PHYSV_PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
if str(PHYSV_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSV_PROJECT_ROOT))

from physv_eval.official_pdi import OfficialPDIRunner
from physv_eval.paths import FLUX_PYTHON, VPHY_PYTHON
from physv_eval.records import (
    get_cosmos_reason1,
    get_official_pdi,
    get_phyground,
    get_videophy2_auto,
    get_wmreward,
    load_payload,
    metric_value,
    save_payload,
    set_cosmos_reason1,
    set_official_pdi,
    set_phyground,
    set_videophy2_auto,
    set_wmreward,
)
from physv_eval.single_case.pdi import score_case as score_pdi_case
from physv_eval.single_case.wmreward import score_case as score_wmreward_case
from physv_eval.wmreward_official import WMRewardRunner


DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/test/precheck_v2_s42_ti2v_5b_fluxff")
DEFAULT_METRICS = [
    "pdi",
    "wmreward",
    "videophy2_pc",
    "videophy2_sa",
    "phyground",
    "cosmos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate vjepa_exp TI2V outputs and backfill metric results into each video.json."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=["pdi", "wmreward", "videophy2_pc", "videophy2_sa", "phyground", "cosmos"],
    )
    parser.add_argument("--case-id", action="append", default=None, help="Only evaluate selected case_id values.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--pdi-python", default=None)
    parser.add_argument("--wmreward-cuda-visible-devices", default=None)
    parser.add_argument("--videophy-python", type=Path, default=VPHY_PYTHON)
    parser.add_argument("--videophy-cuda-visible-devices", default=None)
    parser.add_argument("--flux-python", type=Path, default=FLUX_PYTHON)
    parser.add_argument("--flux-cuda-visible-devices", default=None)
    parser.add_argument("--phyground-general-only", action="store_true", default=True)
    return parser.parse_args()


def iter_case_jsons(root: Path, case_ids: set[str] | None, limit: int | None) -> list[Path]:
    rows: list[Path] = []
    for json_path in sorted(root.glob("*/video.json")):
        case_id = json_path.parent.name
        if case_ids is not None and case_id not in case_ids:
            continue
        rows.append(json_path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def resolve_prompt(payload: dict[str, Any]) -> str | None:
    for key in ("caption", "prompt", "input_video_prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def canonicalize_payload(payload: dict[str, Any], json_path: Path) -> tuple[dict[str, Any], bool]:
    changed = False
    video_path = payload.get("output_video")
    if not isinstance(video_path, str) or not video_path.strip():
        candidate = json_path.with_suffix(".mp4")
        if candidate.is_file():
            video_path = str(candidate)
            payload["output_video"] = video_path
            changed = True
    if isinstance(video_path, str) and video_path.strip():
        if payload.get("video") != video_path:
            payload["video"] = video_path
            changed = True
        if payload.get("video_path") != video_path:
            payload["video_path"] = video_path
            changed = True

    prompt = resolve_prompt(payload)
    if prompt is not None and payload.get("prompt") != prompt:
        payload["prompt"] = prompt
        changed = True
    if prompt is not None and payload.get("caption") != prompt:
        payload["caption"] = prompt
        changed = True
    if payload.get("_json_path") != str(json_path):
        payload["_json_path"] = str(json_path)
        changed = True
    return payload, changed


def build_eval_payload(payload: dict[str, Any], json_path: Path) -> dict[str, Any]:
    eval_payload = dict(payload)
    eval_payload["_json_path"] = str(json_path)
    eval_payload["json_path"] = str(json_path)
    video_path = eval_payload.get("output_video") or eval_payload.get("video") or eval_payload.get("video_path")
    if video_path:
        eval_payload["output_video"] = str(video_path)
        eval_payload["video"] = str(video_path)
        eval_payload["video_path"] = str(video_path)
    prompt = resolve_prompt(eval_payload)
    if prompt is not None:
        eval_payload["prompt"] = prompt
        eval_payload["caption"] = prompt
    return eval_payload


def should_run(metric_name: str, payload: dict[str, Any], refresh: bool) -> bool:
    if refresh:
        return True
    if metric_name == "pdi":
        return get_official_pdi(payload) is None or metric_value(payload, "official_pdi") is None
    if metric_name == "wmreward":
        return get_wmreward(payload) is None or metric_value(payload, "wmreward_jepa") is None
    if metric_name == "videophy2_pc":
        return get_videophy2_auto(payload) is None or metric_value(payload, "videophy2_auto_pc") is None
    if metric_name == "videophy2_sa":
        return get_videophy2_auto(payload) is None or metric_value(payload, "videophy2_auto_sa") is None
    if metric_name == "phyground":
        return get_phyground(payload) is None or metric_value(payload, "phyground_general_avg") is None
    if metric_name == "cosmos":
        return get_cosmos_reason1(payload) is None or metric_value(payload, "cosmos_reason1") is None
    raise KeyError(metric_name)


def run_subprocess_metric(
    *,
    module_name: str,
    python_bin: Path,
    json_path: Path,
    cuda_visible_devices: str | None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vjepa_bench_") as tmp_dir:
        output_json = Path(tmp_dir) / "result.json"
        cmd = [
            str(python_bin),
            "-m",
            module_name,
            "--input-json",
            str(json_path),
            "--output-json",
            str(output_json),
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        if cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
        subprocess.run(cmd, cwd=PHYSV_PROJECT_ROOT, env=env, check=True)
        return json.loads(output_json.read_text(encoding="utf-8"))


def make_summary(root: Path, metrics: list[str], cases: list[Path]) -> dict[str, Any]:
    return {
        "root": str(root),
        "metrics": list(metrics),
        "num_cases": len(cases),
        "updated": {name: 0 for name in metrics},
        "skipped": {name: 0 for name in metrics},
        "failed": {name: 0 for name in metrics},
        "errors": [],
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run root does not exist: {root}")

    case_ids = set(args.case_id) if args.case_id else None
    case_jsons = iter_case_jsons(root, case_ids, args.limit)
    if not case_jsons:
        raise FileNotFoundError(f"no video.json files found under {root}")

    pdi_runner = (
        OfficialPDIRunner(
            python_bin=args.pdi_python,
            cuda_visible_devices=args.cuda_visible_devices,
        )
        if "pdi" in args.metrics
        else None
    )
    wmreward_runner = (
        WMRewardRunner(
            cuda_visible_devices=args.wmreward_cuda_visible_devices or args.cuda_visible_devices,
        )
        if "wmreward" in args.metrics
        else None
    )
    summary = make_summary(root, args.metrics, case_jsons)

    for index, json_path in enumerate(case_jsons, start=1):
        payload = load_payload(json_path)
        payload, normalized_changed = canonicalize_payload(payload, json_path)
        eval_payload = build_eval_payload(payload, json_path)
        changed = bool(normalized_changed)

        print(f"[{index}/{len(case_jsons)}] {json_path.parent.name}", flush=True)
        for metric_name in args.metrics:
            if not should_run(metric_name, payload, args.refresh):
                summary["skipped"][metric_name] += 1
                print(f"  [skip] {metric_name}", flush=True)
                continue

            try:
                prompt = resolve_prompt(eval_payload)
                if metric_name == "pdi":
                    result = score_pdi_case(
                        eval_payload,
                        text_query=prompt,
                        refresh=args.refresh,
                        runner=pdi_runner,
                    )
                    set_official_pdi(payload, result)
                elif metric_name == "wmreward":
                    result = score_wmreward_case(eval_payload, runner=wmreward_runner)
                    set_wmreward(payload, result)
                elif metric_name == "videophy2_pc":
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.videophy2",
                        python_bin=args.videophy_python.expanduser().resolve(),
                        json_path=json_path,
                        cuda_visible_devices=args.videophy_cuda_visible_devices or args.cuda_visible_devices,
                        extra_args=["--task", "pc"],
                    )
                    set_videophy2_auto(payload, result)
                elif metric_name == "videophy2_sa":
                    if not prompt:
                        raise ValueError(f"missing prompt/caption for {json_path}")
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.videophy2",
                        python_bin=args.videophy_python.expanduser().resolve(),
                        json_path=json_path,
                        cuda_visible_devices=args.videophy_cuda_visible_devices or args.cuda_visible_devices,
                        extra_args=["--task", "sa", "--caption", prompt],
                    )
                    set_videophy2_auto(payload, result)
                elif metric_name == "phyground":
                    if not prompt:
                        raise ValueError(f"missing prompt/caption for {json_path}")
                    extra_args = ["--caption", prompt]
                    if args.phyground_general_only:
                        extra_args.append("--general-only")
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.phyground",
                        python_bin=args.flux_python.expanduser().resolve(),
                        json_path=json_path,
                        cuda_visible_devices=args.flux_cuda_visible_devices or args.cuda_visible_devices,
                        extra_args=extra_args,
                    )
                    set_phyground(payload, result)
                elif metric_name == "cosmos":
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.cosmos_reason1",
                        python_bin=args.flux_python.expanduser().resolve(),
                        json_path=json_path,
                        cuda_visible_devices=args.flux_cuda_visible_devices or args.cuda_visible_devices,
                    )
                    set_cosmos_reason1(payload, result)
                else:
                    raise KeyError(metric_name)

                changed = True
                summary["updated"][metric_name] += 1
                print(f"  [done] {metric_name}", flush=True)
            except Exception as exc:
                summary["failed"][metric_name] += 1
                summary["errors"].append(
                    {
                        "case_id": json_path.parent.name,
                        "json_path": str(json_path),
                        "metric": metric_name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"  [fail] {metric_name}: {exc}", flush=True)
                if args.fail_fast:
                    if changed:
                        save_payload(json_path, payload)
                    raise

        if changed:
            save_payload(json_path, payload)

    summary_json = (
        args.summary_json.expanduser().resolve()
        if args.summary_json is not None
        else root / "bench_summary.json"
    )
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[summary] {summary_json}", flush=True)


if __name__ == "__main__":
    main()
