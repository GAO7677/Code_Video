#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import yaml


DATASET_ROOT = Path("/data/gaoya/dataset/AnteaWu-PDI-Dataset")
PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI_full183_repro")
FLORENCE_MODEL = Path("/data/gaoya/ckpt/microsoft-Florence-2-base")

PROVIDER_TO_PAPER_NAME = {
    "GT": "Ground Truth (GT)",
    "seedance": "Seedance 2.0",
    "cogvideoX": "CogVideoX-3",
    "Flow": "Veo 3.1",
    "wan22": "Wan 2.2",
    "Sora": "Sora",
    "hunyuan": "HunyuanVideo",
}

PAPER_TABLE1 = {
    "Ground Truth (GT)": {
        "pdi_score": 0.1206,
        "ci95_low": 0.1018,
        "ci95_high": 0.1386,
        "scale": 0.0660,
        "traj": 0.1764,
        "rigid": 0.1182,
        "std": 0.0378,
        "outlier_ratio": 0.0,
        "mathpass_ratio": 86.7,
    },
    "Seedance 2.0": {
        "pdi_score": 0.2422,
        "ci95_low": 0.1954,
        "ci95_high": 0.2920,
        "scale": 0.2295,
        "traj": 0.2064,
        "rigid": 0.3392,
        "std": 0.1315,
        "outlier_ratio": 0.0,
        "mathpass_ratio": 89.3,
    },
    "CogVideoX-3": {
        "pdi_score": 0.2480,
        "ci95_low": 0.1656,
        "ci95_high": 0.4093,
        "scale": 0.3135,
        "traj": 0.2033,
        "rigid": 0.2065,
        "std": 0.3065,
        "outlier_ratio": 3.6,
        "mathpass_ratio": 85.7,
    },
    "Veo 3.1": {
        "pdi_score": 0.4521,
        "ci95_low": 0.2611,
        "ci95_high": 0.7247,
        "scale": 0.7507,
        "traj": 0.2271,
        "rigid": 0.3049,
        "std": 0.6980,
        "outlier_ratio": 7.1,
        "mathpass_ratio": 50.0,
    },
    "Wan 2.2": {
        "pdi_score": 0.5595,
        "ci95_low": 0.2572,
        "ci95_high": 1.0766,
        "scale": 0.9317,
        "traj": 0.2096,
        "rigid": 0.5150,
        "std": 1.2301,
        "outlier_ratio": 7.1,
        "mathpass_ratio": 67.9,
    },
    "Sora": {
        "pdi_score": 0.8255,
        "ci95_low": 0.2652,
        "ci95_high": 1.4847,
        "scale": 1.6753,
        "traj": 0.2711,
        "rigid": 0.2345,
        "std": 1.7312,
        "outlier_ratio": 14.3,
        "mathpass_ratio": 70.4,
    },
    "HunyuanVideo": {
        "pdi_score": 0.8825,
        "ci95_low": 0.3094,
        "ci95_high": 1.6018,
        "scale": 1.8469,
        "traj": 0.2515,
        "rigid": 0.2160,
        "std": 1.7730,
        "outlier_ratio": 14.3,
        "mathpass_ratio": 57.1,
    },
}


@dataclass(frozen=True)
class EvalRow:
    provider: str
    task: str
    clip_index: str
    prompt: str
    file_path: Path

    @property
    def unique_id(self) -> str:
        clip_part = self.clip_index if self.clip_index else "gt"
        safe_prompt = re.sub(r"[^0-9A-Za-z._-]+", "_", self.prompt.strip()).strip("_")
        return f"{self.provider}__{self.task}__{clip_part}__{safe_prompt}"

    @property
    def paper_name(self) -> str:
        return PROVIDER_TO_PAPER_NAME[self.provider]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce PDI-Bench Table 1 on the local AnteaWu dataset.")
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--pdi_root", type=Path, default=PDI_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--florence_model", type=Path, default=FLORENCE_MODEL)
    parser.add_argument("--providers", nargs="+", default=list(PROVIDER_TO_PAPER_NAME.keys()))
    parser.add_argument("--limit_per_provider", type=int, default=None)
    parser.add_argument("--cuda_visible_devices", default="3")
    parser.add_argument("--python_bin", default=os.environ.get("PYTHON", "") or shutil.which("python") or "python")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_metadata(dataset_root: Path, providers: set[str]) -> list[EvalRow]:
    rows: list[EvalRow] = []
    metadata_path = dataset_root / "metadata.csv"
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            provider = raw["provider"]
            if provider not in providers:
                continue
            rel_path = Path(raw["file_path"])
            disk_rel = Path(*["partial_occlusion" if part == "Partial_Occlusion" else part for part in rel_path.parts])
            file_path = dataset_root / disk_rel
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing dataset file: {file_path} (from metadata {rel_path})")
            rows.append(
                EvalRow(
                    provider=provider,
                    task=raw["task"],
                    clip_index=(raw["clip_index"] or "").strip(),
                    prompt=raw["prompt"].strip(),
                    file_path=file_path,
                )
            )
    return rows


def build_eval_config(base_config: Path, cache_dir: Path) -> Path:
    with base_config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["cache_dir"] = str(cache_dir)
    out_path = cache_dir / "default_eval.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")

    def extract(pattern: str, cast: type | None = None) -> Any:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            return None
        value = match.group(1).strip()
        return cast(value) if cast else value

    def as_bool(value: Any) -> bool | None:
        if value == "True":
            return True
        if value == "False":
            return False
        return None

    result = {
        "pdi_score": extract(r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": extract(r"OVERALL GRADE:\s*(.+)"),
        "scale_component": extract(r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_component": extract(r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "epsilon_rigidity": extract(r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "rigidity_strategy": extract(r"Rigidity Strategy:\s*(.+)"),
        "vp_component": extract(r"VP Component .*?:\s*([0-9.]+)", float),
        "ra_math_pass": as_bool(extract(r"RA Math Pass:\s*(True|False)")),
        "ra_ground_rmse": extract(r"RA Ground RMSE:\s*([0-9.eE+-]+)", float),
        "ra_scale_jump": extract(r"RA Scale Jump:\s*([0-9.eE+-]+)", float),
        "ra_reproj_err": extract(r"RA Reproj Err:\s*([0-9.eE+-]+)", float),
        "ra_overall_pass": as_bool(extract(r"RA Overall Pass:\s*(True|False)")),
    }
    return result


def bootstrap_ci(values: list[float], rng: random.Random, iters: int, use_median: bool) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    samples = []
    n = len(values)
    for _ in range(iters):
        picked = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(median(picked) if use_median else mean(picked))
    samples.sort()
    low_idx = max(0, int(0.025 * len(samples)) - 1)
    high_idx = min(len(samples) - 1, int(0.975 * len(samples)) - 1)
    return samples[low_idx], samples[high_idx]


def compute_outlier_ratio_iqr(values: list[float]) -> float | None:
    if len(values) < 4:
        return None
    sorted_vals = sorted(values)
    def percentile(p: float) -> float:
        idx = (len(sorted_vals) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_vals[lo]
        frac = idx - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac
    q1 = percentile(0.25)
    q3 = percentile(0.75)
    iqr = q3 - q1
    thresh = q3 + 1.5 * iqr
    count = sum(v > thresh for v in values)
    return 100.0 * count / len(values)


def summarize_provider(rows: list[dict[str, Any]], bootstrap_iters: int, seed: int) -> dict[str, Any]:
    pdi_values = [row["pdi_score"] for row in rows if row["pdi_score"] is not None]
    scale_values = [row["scale_component"] for row in rows if row["scale_component"] is not None]
    traj_values = [row["traj_component"] for row in rows if row["traj_component"] is not None]
    rigid_values = [row["epsilon_rigidity"] for row in rows if row["epsilon_rigidity"] is not None]
    vp_values = [row["vp_component"] for row in rows if row["vp_component"] is not None]
    math_passes = [row["ra_math_pass"] for row in rows if row["ra_math_pass"] is not None]
    overall_passes = [row["ra_overall_pass"] for row in rows if row["ra_overall_pass"] is not None]

    rng_mean = random.Random(seed)
    rng_median = random.Random(seed)
    mean_ci_low, mean_ci_high = bootstrap_ci(pdi_values, rng_mean, bootstrap_iters, use_median=False)
    median_ci_low, median_ci_high = bootstrap_ci(pdi_values, rng_median, bootstrap_iters, use_median=True)

    summary = {
        "num_sequences": len(rows),
        "valid_sequences": len(pdi_values),
        "mean_pdi_score": mean(pdi_values) if pdi_values else float("nan"),
        "median_pdi_score": median(pdi_values) if pdi_values else float("nan"),
        "ci95_mean_low": mean_ci_low,
        "ci95_mean_high": mean_ci_high,
        "ci95_median_low": median_ci_low,
        "ci95_median_high": median_ci_high,
        "mean_scale_component": mean(scale_values) if scale_values else float("nan"),
        "mean_traj_component": mean(traj_values) if traj_values else float("nan"),
        "mean_epsilon_rigidity": mean(rigid_values) if rigid_values else float("nan"),
        "mean_vp_component": mean(vp_values) if vp_values else float("nan"),
        "std_pdi_score": (sum((x - mean(pdi_values)) ** 2 for x in pdi_values) / len(pdi_values)) ** 0.5 if pdi_values else float("nan"),
        "outlier_ratio_iqr_guess": compute_outlier_ratio_iqr(pdi_values),
        "mathpass_ratio": 100.0 * sum(bool(x) for x in math_passes) / len(math_passes) if math_passes else float("nan"),
        "overall_pass_ratio": 100.0 * sum(bool(x) for x in overall_passes) / len(overall_passes) if overall_passes else float("nan"),
    }
    return summary


def run_one(
    item: EvalRow,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    provider_root = run_root / "outputs" / item.provider
    staged_dir = provider_root / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_video = staged_dir / f"{item.unique_id}.mp4"
    report_dir = provider_root / "reports" / item.unique_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{item.unique_id}_pdi_report.txt"
    cache_dir = provider_root / "cache" / item.unique_id
    config_path = build_eval_config(args.pdi_root / "configs" / "default.yaml", cache_dir)

    if staged_video.exists() or staged_video.is_symlink():
        staged_video.unlink()
    os.symlink(item.file_path.resolve(), staged_video)

    if not report_path.exists() or args.refresh:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(args.pdi_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PDI_FLORENCE_MODEL_ID"] = str(args.florence_model)
        env["PDI_EVAL_VIDEO_ID"] = item.unique_id
        if args.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
        cmd = [
            args.python_bin,
            "evaluation/main.py",
            "--input",
            str(staged_video),
            "--config",
            str(config_path),
            "--output_dir",
            str(provider_root / "reports"),
            "--text",
            item.prompt,
        ]
        subprocess.run(cmd, cwd=args.pdi_root, env=env, check=True)

    metrics = parse_report(report_path)
    return {
        "provider": item.provider,
        "paper_name": item.paper_name,
        "task": item.task,
        "clip_index": item.clip_index,
        "prompt": item.prompt,
        "source_video": str(item.file_path),
        "staged_video": str(staged_video),
        "report_dir": str(report_dir),
        "report_path": str(report_path),
        "unique_id": item.unique_id,
        **metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    providers = set(args.providers)
    run_root = args.output_root
    run_root.mkdir(parents=True, exist_ok=True)

    items = load_metadata(args.dataset_root, providers)
    if args.limit_per_provider is not None:
        kept: list[EvalRow] = []
        counts: dict[str, int] = {}
        for item in items:
            count = counts.get(item.provider, 0)
            if count >= args.limit_per_provider:
                continue
            kept.append(item)
            counts[item.provider] = count + 1
        items = kept

    per_video_rows: list[dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{len(items)}] {item.provider} :: {item.task} :: {item.prompt}", flush=True)
        per_video_rows.append(run_one(item, args, run_root))

    per_video_csv = run_root / "results" / "per_video_metrics.csv"
    write_csv(per_video_csv, per_video_rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_video_rows:
        grouped.setdefault(row["paper_name"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    for paper_name, rows in grouped.items():
        summary = summarize_provider(rows, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
        summary_row = {"paper_name": paper_name, **summary}
        summary_rows.append(summary_row)

        paper_ref = PAPER_TABLE1.get(paper_name, {})
        compare_rows.append(
            {
                "paper_name": paper_name,
                "paper_pdi_score": paper_ref.get("pdi_score"),
                "local_mean_pdi_score": summary["mean_pdi_score"],
                "delta_mean_pdi_score": (summary["mean_pdi_score"] - paper_ref["pdi_score"]) if "pdi_score" in paper_ref else None,
                "paper_ci95_low": paper_ref.get("ci95_low"),
                "paper_ci95_high": paper_ref.get("ci95_high"),
                "local_ci95_mean_low": summary["ci95_mean_low"],
                "local_ci95_mean_high": summary["ci95_mean_high"],
                "local_ci95_median_low": summary["ci95_median_low"],
                "local_ci95_median_high": summary["ci95_median_high"],
                "paper_scale": paper_ref.get("scale"),
                "local_mean_scale": summary["mean_scale_component"],
                "paper_traj": paper_ref.get("traj"),
                "local_mean_traj": summary["mean_traj_component"],
                "paper_rigid": paper_ref.get("rigid"),
                "local_mean_rigid": summary["mean_epsilon_rigidity"],
                "paper_std": paper_ref.get("std"),
                "local_std": summary["std_pdi_score"],
                "paper_outlier_ratio": paper_ref.get("outlier_ratio"),
                "local_outlier_ratio_iqr_guess": summary["outlier_ratio_iqr_guess"],
                "paper_mathpass_ratio": paper_ref.get("mathpass_ratio"),
                "local_mathpass_ratio": summary["mathpass_ratio"],
                "local_overall_pass_ratio": summary["overall_pass_ratio"],
                "num_sequences": summary["num_sequences"],
                "valid_sequences": summary["valid_sequences"],
            }
        )

    summary_rows.sort(key=lambda row: row["paper_name"])
    compare_rows.sort(key=lambda row: row["paper_name"])
    write_csv(run_root / "results" / "provider_summary.csv", summary_rows)
    write_csv(run_root / "results" / "table1_reproduction_compare.csv", compare_rows)

    notes = {
        "dataset_root": str(args.dataset_root),
        "pdi_root": str(args.pdi_root),
        "output_root": str(run_root),
        "providers": sorted(providers),
        "paper_vs_code_notes": [
            "论文 Table 1 第 10 页表头写的是 PDI Score(mean of residuals)，但附录 A.3 第 19 页写的是对有效序列报告 median + 95% bootstrap CI。",
            "当前官方 evaluation_*.py 脚本默认输出 mean/std/min/max，不直接等价于论文 Table 1。",
            "当前公开代码中的默认权重来自 configs/default.yaml: w_scale=0.4, w_trajectory=0.4, w_rigidity=0.2, w_vp=0.0。",
            "VP Component 在当前代码会计算并写入 report，但默认不计入最终 PDI。",
            "Outlier 列在公开论文与代码中未找到明确阈值定义；本复现结果仅提供一个 IQR-based guess，不能视为论文官方口径。",
            "为避免不同类别/模型下同名视频在 Mega-SAM 工作目录中互相覆盖，本复现通过 PDI_EVAL_VIDEO_ID 强制使用唯一 video_id。",
            "metadata.csv 使用 Partial_Occlusion，磁盘目录实际为 partial_occlusion；本脚本已做映射。",
        ],
    }
    (run_root / "results" / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "per_video_csv": str(per_video_csv),
        "provider_summary_csv": str(run_root / "results" / "provider_summary.csv"),
        "compare_csv": str(run_root / "results" / "table1_reproduction_compare.csv"),
        "notes_json": str(run_root / "results" / "notes.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
