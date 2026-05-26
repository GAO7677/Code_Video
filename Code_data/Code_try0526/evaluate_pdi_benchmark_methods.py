#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from rerank_video.video_utils import write_json


PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")
BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench")
OUTPUT_ROOT = BENCHMARK_ROOT / "output"
RESULT_CSV = BENCHMARK_ROOT / "result" / "metrics.csv"
RUN_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_benchmark_methods_eval")
GT_RUN_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_gt15_official_eval")
FLORENCE_MODEL = Path("/data/gaoya/ckpt/microsoft-Florence-2-base")
METHODS = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]


def _extract(text: str, pattern: str, cast: type | None = None) -> Any:
    import re

    match = re.search(pattern, text, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    return cast(value) if cast else value


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")
    return {
        "pdi_score": _extract(text, r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": _extract(text, r"OVERALL GRADE:\s*(.+)"),
        "scale_component": _extract(text, r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_component": _extract(text, r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "epsilon_rigidity": _extract(text, r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "rigidity_strategy": _extract(text, r"Rigidity Strategy:\s*(.+)"),
        "vp_component": _extract(text, r"VP Component .*?:\s*([0-9.]+)", float),
        "ra_math_pass": _extract(text, r"RA Math Pass:\s*(True|False)"),
        "ra_mllm_success": _extract(text, r"RA MLLM Success:\s*(True|False)"),
        "ra_mllm_score": _extract(text, r"RA MLLM Score:\s*([0-9]+)", int),
        "ra_overall_pass": _extract(text, r"RA Overall Pass:\s*(True|False)"),
    }


def parse_bool(value: Any) -> bool | None:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def build_temp_config(cache_dir: Path) -> Path:
    with (PDI_ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["cache_dir"] = str(cache_dir)
    temp_config = cache_dir / "default_eval.yaml"
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    temp_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return temp_config


def evaluate_single(method: str, task: str, clip_name: str, json_path: Path) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    video_path = Path(payload["video_path"])
    output_dir = RUN_ROOT / method / task
    report_dir = output_dir / clip_name
    report_path = report_dir / f"{clip_name}_pdi_report.txt"
    cache_dir = PDI_ROOT / "output" / "benchmark_cache" / method / task / clip_name
    config_path = build_temp_config(cache_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PDI_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PDI_FLORENCE_MODEL_ID"] = str(FLORENCE_MODEL)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "5")

    cmd = [
        "python",
        "evaluation/main.py",
        "--input",
        str(video_path),
        "--config",
        str(config_path),
        "--output_dir",
        str(output_dir),
        "--text",
        str(payload["prompt"]),
    ]
    subprocess.run(cmd, cwd=PDI_ROOT, env=env, check=True)
    metrics = parse_report(report_path)
    payload["raw_report_path"] = str(report_path)
    payload["metrics"] = {
        **metrics,
        "grade_letter": str(metrics["grade"]).split()[0] if metrics.get("grade") else "",
        "ra_math_pass_bool": parse_bool(metrics.get("ra_math_pass")),
        "ra_mllm_success_bool": parse_bool(metrics.get("ra_mllm_success")),
        "ra_overall_pass_bool": parse_bool(metrics.get("ra_overall_pass")),
    }
    write_json(json_path, payload)
    return payload


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def collect_method_payloads(method: str) -> list[dict[str, Any]]:
    if method == "GT":
        rows: list[dict[str, Any]] = []
        for json_path in sorted((OUTPUT_ROOT / "GT").glob("*/*.json")):
            rows.append(json.loads(json_path.read_text(encoding="utf-8")))
        return rows

    rows = []
    for json_path in sorted((OUTPUT_ROOT / method).glob("*/*.json")):
        task = json_path.parent.name
        clip_name = json_path.stem
        rows.append(evaluate_single(method, task, clip_name, json_path))
    return rows


def summarize_method(method: str, rows: list[dict[str, Any]], gt_row: dict[str, Any] | None = None) -> dict[str, Any]:
    if method == "GT" and gt_row is not None:
        return gt_row

    pdi_scores = [float(row["metrics"]["pdi_score"]) for row in rows]
    scale_scores = [float(row["metrics"]["scale_component"]) for row in rows]
    traj_scores = [float(row["metrics"]["traj_component"]) for row in rows]
    rigid_scores = [float(row["metrics"]["epsilon_rigidity"]) for row in rows]
    vp_scores = [float(row["metrics"]["vp_component"]) for row in rows]
    grade_counts = Counter(str(row["metrics"]["grade_letter"]) for row in rows)
    ra_math_pass = sum(1 for row in rows if row["metrics"].get("ra_math_pass_bool") is True)
    ra_overall_pass = sum(1 for row in rows if row["metrics"].get("ra_overall_pass_bool") is True)
    return {
        "benchmark": "PDI-Bench",
        "method": method,
        "provider": method,
        "num_videos": len(rows),
        "mean_pdi_score": f"{mean(pdi_scores):.6f}",
        "mean_scale_component": f"{mean(scale_scores):.6f}",
        "mean_traj_component": f"{mean(traj_scores):.6f}",
        "mean_epsilon_rigidity": f"{mean(rigid_scores):.6f}",
        "mean_vp_component": f"{mean(vp_scores):.6f}",
        "grade_A_count": grade_counts.get("A", 0),
        "grade_B_count": grade_counts.get("B", 0),
        "grade_C_count": grade_counts.get("C", 0),
        "ra_math_pass_count": ra_math_pass,
        "ra_overall_pass_count": ra_overall_pass,
        "summary_report_path": str(RUN_ROOT / method),
    }


def load_existing_gt_row() -> dict[str, Any] | None:
    if not RESULT_CSV.is_file():
        return None
    with RESULT_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["method"] == "GT":
                return dict(row)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Subset of benchmark methods to evaluate and summarize.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    existing_gt_row = load_existing_gt_row()
    summaries: list[dict[str, Any]] = []
    for method in args.methods:
        rows = collect_method_payloads(method)
        summaries.append(summarize_method(method, rows, gt_row=existing_gt_row))

    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    write_json(
        BENCHMARK_ROOT / "manifest_methods_eval.json",
        {
            "methods": args.methods,
            "result_csv": RESULT_CSV,
            "run_root": RUN_ROOT,
        },
    )
    print(json.dumps({"result_csv": str(RESULT_CSV), "methods": args.methods}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
