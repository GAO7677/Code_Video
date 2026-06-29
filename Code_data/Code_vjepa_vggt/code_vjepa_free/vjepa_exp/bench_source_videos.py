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
Backfill source-video metrics into GT json files under ABD_test.

Behavior:
- If `metric_results` already exists and `output_video == source_video`, copy it to
  `source_metric_results` directly.
- Otherwise compute the missing metrics against `source_video` and write them to
  `source_metric_results`.

Example:
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/bench_source_videos.py \
    --root /data/gaoya/AAA_test_video/Output_try0526/ABD_test \
    --groups A B D \
    --metrics pdi wmreward videophy2_pc videophy2_sa phyground cosmos \
    --cuda-visible-devices 7
"""


PHYSV_PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
if str(PHYSV_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSV_PROJECT_ROOT))

from physv_eval.official_pdi import OfficialPDIRunner
from physv_eval.paths import FLUX_PYTHON, VPHY_PYTHON
from physv_eval.single_case.pdi import score_case as score_pdi_case
from physv_eval.single_case.wmreward import score_case as score_wmreward_case
from physv_eval.wmreward_official import WMRewardRunner


DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/ABD_test")
DEFAULT_METRICS = ["pdi", "wmreward", "videophy2_pc", "videophy2_sa", "phyground", "cosmos"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill source_video metric results into GT json files under ABD_test."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--groups", nargs="+", default=["A", "B", "D"])
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=["pdi", "wmreward", "videophy2_pc", "videophy2_sa", "phyground", "cosmos"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--pdi-python", default=None)
    parser.add_argument("--wmreward-cuda-visible-devices", default=None)
    parser.add_argument("--videophy-python", type=Path, default=VPHY_PYTHON)
    parser.add_argument("--videophy-cuda-visible-devices", default=None)
    parser.add_argument("--flux-python", type=Path, default=FLUX_PYTHON)
    parser.add_argument("--flux-cuda-visible-devices", default=None)
    parser.add_argument("--phyground-general-only", action="store_true", default=True)
    return parser.parse_args()


def iter_gt_jsons(root: Path, groups: list[str], limit: int | None) -> list[Path]:
    rows: list[Path] = []
    for group in groups:
        rows.extend(sorted((root / group / "GT").glob("*.json")))
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_prompt(payload: dict[str, Any]) -> str | None:
    for key in ("input_prompt", "prompt", "caption"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_source_eval_payload(payload: dict[str, Any], json_path: Path) -> dict[str, Any]:
    source_video = payload.get("source_video")
    if not isinstance(source_video, str) or not source_video.strip():
        raise ValueError(f"missing source_video in {json_path}")
    prompt = resolve_prompt(payload)
    eval_payload = {
        "_json_path": str(json_path),
        "json_path": str(json_path),
        "source_video": source_video,
        "output_video": source_video,
        "video": source_video,
        "video_path": source_video,
        "prompt": prompt,
        "caption": prompt,
        "input_prompt": prompt,
        "input_image": payload.get("input_image"),
        "case_key": payload.get("case_key", json_path.stem),
        "group": payload.get("group"),
        "benchmark": payload.get("benchmark"),
        "method_name": payload.get("method_name"),
    }
    return eval_payload


def should_run(metric_name: str, source_metric_results: dict[str, Any], refresh: bool) -> bool:
    if refresh:
        return True
    if metric_name == "pdi":
        return "official_pdi" not in source_metric_results
    if metric_name == "wmreward":
        return "wmreward_jepa" not in source_metric_results
    if metric_name == "videophy2_pc":
        videophy = source_metric_results.get("videophy2_auto", {})
        return not isinstance(videophy, dict) or videophy.get("pc_score") is None
    if metric_name == "videophy2_sa":
        videophy = source_metric_results.get("videophy2_auto", {})
        return not isinstance(videophy, dict) or videophy.get("sa_score") is None
    if metric_name == "phyground":
        return "phyground" not in source_metric_results
    if metric_name == "cosmos":
        return "cosmos_reason1" not in source_metric_results
    raise KeyError(metric_name)


def run_subprocess_metric(
    *,
    module_name: str,
    python_bin: Path,
    payload: dict[str, Any],
    cuda_visible_devices: str | None,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="abd_source_bench_") as tmp_dir:
        input_json = Path(tmp_dir) / "input.json"
        output_json = Path(tmp_dir) / "result.json"
        input_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cmd = [
            str(python_bin),
            "-m",
            module_name,
            "--input-json",
            str(input_json),
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


def source_metric_summary_from_results(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    pdi = results.get("official_pdi")
    if isinstance(pdi, dict) and "pdi_score" in pdi:
        summary["pdi_score"] = pdi.get("pdi_score")
        for key in ("grade", "scale_component", "traj_component", "epsilon_rigidity", "vp_component"):
            if key in pdi:
                summary[key] = pdi.get(key)
    wmreward = results.get("wmreward_jepa")
    if isinstance(wmreward, dict):
        if "similarity" in wmreward:
            summary["wmreward_similarity"] = wmreward.get("similarity")
        if "surprise" in wmreward:
            summary["wmreward_surprise"] = wmreward.get("surprise")
    videophy = results.get("videophy2_auto")
    if isinstance(videophy, dict):
        if "pc_score" in videophy:
            summary["videophy2_pc"] = videophy.get("pc_score")
        if "sa_score" in videophy:
            summary["videophy2_sa"] = videophy.get("sa_score")
    phyground = results.get("phyground")
    if isinstance(phyground, dict):
        for key in ("general_avg", "SA", "PTV", "persistence"):
            if key in phyground:
                summary[f"phyground_{key}"] = phyground.get(key)
    cosmos = results.get("cosmos_reason1")
    if isinstance(cosmos, dict) and "score" in cosmos:
        summary["cosmos_reason1"] = cosmos.get("score")
    return summary


def make_summary(root: Path, groups: list[str], metrics: list[str], cases: list[Path]) -> dict[str, Any]:
    return {
        "root": str(root),
        "groups": groups,
        "metrics": metrics,
        "num_cases": len(cases),
        "migrated_existing": 0,
        "updated": {name: 0 for name in metrics},
        "skipped": {name: 0 for name in metrics},
        "failed": {name: 0 for name in metrics},
        "errors": [],
    }


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    case_jsons = iter_gt_jsons(root, args.groups, args.limit)
    if not case_jsons:
        raise FileNotFoundError(f"no GT jsons found under {root}")

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

    summary = make_summary(root, args.groups, args.metrics, case_jsons)

    for index, json_path in enumerate(case_jsons, start=1):
        payload = load_json(json_path)
        source_results = dict(payload.get("source_metric_results") or {})
        changed = False
        print(f"[{index}/{len(case_jsons)}] {json_path}", flush=True)

        if not args.refresh and not source_results:
            existing_metric_results = payload.get("metric_results")
            if (
                isinstance(existing_metric_results, dict)
                and existing_metric_results
                and payload.get("output_video") == payload.get("source_video")
            ):
                payload["source_metric_results"] = existing_metric_results
                payload["source_metric_summary"] = source_metric_summary_from_results(existing_metric_results)
                save_json(json_path, payload)
                summary["migrated_existing"] += 1
                print("  [migrate] metric_results -> source_metric_results", flush=True)
                continue

        eval_payload = build_source_eval_payload(payload, json_path)
        prompt = resolve_prompt(payload)

        for metric_name in args.metrics:
            if not should_run(metric_name, source_results, args.refresh):
                summary["skipped"][metric_name] += 1
                print(f"  [skip] {metric_name}", flush=True)
                continue

            try:
                if metric_name == "pdi":
                    result = score_pdi_case(
                        eval_payload,
                        text_query=prompt,
                        refresh=args.refresh,
                        runner=pdi_runner,
                    )
                    source_results["official_pdi"] = result
                elif metric_name == "wmreward":
                    result = score_wmreward_case(eval_payload, runner=wmreward_runner)
                    source_results["wmreward_jepa"] = result
                elif metric_name == "videophy2_pc":
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.videophy2",
                        python_bin=args.videophy_python.expanduser().resolve(),
                        payload=eval_payload,
                        cuda_visible_devices=args.videophy_cuda_visible_devices or args.cuda_visible_devices,
                        extra_args=["--task", "pc"],
                    )
                    merged = dict(source_results.get("videophy2_auto") or {})
                    merged.update(result)
                    source_results["videophy2_auto"] = merged
                elif metric_name == "videophy2_sa":
                    if not prompt:
                        raise ValueError(f"missing prompt for {json_path}")
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.videophy2",
                        python_bin=args.videophy_python.expanduser().resolve(),
                        payload=eval_payload,
                        cuda_visible_devices=args.videophy_cuda_visible_devices or args.cuda_visible_devices,
                        extra_args=["--task", "sa", "--caption", prompt],
                    )
                    merged = dict(source_results.get("videophy2_auto") or {})
                    merged.update(result)
                    source_results["videophy2_auto"] = merged
                elif metric_name == "phyground":
                    if not prompt:
                        raise ValueError(f"missing prompt for {json_path}")
                    extra_args = ["--caption", prompt]
                    if args.phyground_general_only:
                        extra_args.append("--general-only")
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.phyground",
                        python_bin=args.flux_python.expanduser().resolve(),
                        payload=eval_payload,
                        cuda_visible_devices=args.flux_cuda_visible_devices or args.cuda_visible_devices,
                        extra_args=extra_args,
                    )
                    source_results["phyground"] = result
                elif metric_name == "cosmos":
                    result = run_subprocess_metric(
                        module_name="physv_eval.single_case.cosmos_reason1",
                        python_bin=args.flux_python.expanduser().resolve(),
                        payload=eval_payload,
                        cuda_visible_devices=args.flux_cuda_visible_devices or args.cuda_visible_devices,
                    )
                    source_results["cosmos_reason1"] = result
                else:
                    raise KeyError(metric_name)

                summary["updated"][metric_name] += 1
                changed = True
                print(f"  [done] {metric_name}", flush=True)
            except Exception as exc:
                summary["failed"][metric_name] += 1
                summary["errors"].append(
                    {
                        "json_path": str(json_path),
                        "metric": metric_name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"  [fail] {metric_name}: {exc}", flush=True)
                if args.fail_fast:
                    payload["source_metric_results"] = source_results
                    payload["source_metric_summary"] = source_metric_summary_from_results(source_results)
                    save_json(json_path, payload)
                    raise

        if changed:
            payload["source_metric_results"] = source_results
            payload["source_metric_summary"] = source_metric_summary_from_results(source_results)
            save_json(json_path, payload)

    summary_path = (
        args.summary_json.expanduser().resolve()
        if args.summary_json is not None
        else root / "_meta" / "abd_source_bench_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
