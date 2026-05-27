#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import yaml

from reproduce_pdi_table1 import (
    DATASET_ROOT,
    FLORENCE_MODEL,
    OUTPUT_ROOT,
    PAPER_TABLE1,
    PDI_ROOT,
    PROVIDER_TO_PAPER_NAME,
    EvalRow,
    build_eval_config,
    load_metadata,
    parse_report,
    summarize_provider,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent PDI-Bench Table 1 reproduction runner.")
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--pdi_root", type=Path, default=PDI_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT.parent / "PDI_full183_repro_persistent")
    parser.add_argument("--florence_model", type=Path, default=FLORENCE_MODEL)
    parser.add_argument("--providers", nargs="+", default=list(PROVIDER_TO_PAPER_NAME.keys()))
    parser.add_argument("--limit_per_provider", type=int, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda_visible_devices", default="3")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def load_base_config(pdi_root: Path) -> dict[str, Any]:
    config_path = pdi_root / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key in ("sam_ckpt", "sam_cfg", "tracker_ckpt", "mega_sam_ckpt"):
        value = cfg.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            cfg[key] = str((pdi_root / value).resolve())
    return cfg


def stage_video(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src.resolve(), dst)


def write_report(report: dict[str, Any], input_video: Path, report_dir: Path, prompt: str) -> Path:
    report_path = report_dir / f"{input_video.stem}_pdi_report.txt"
    breakdown = report.get("breakdown", {})
    ra = report.get("reconstruction_audit")
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 50 + "\n")
        handle.write("        PDI-Eval Final Audit Report\n")
        handle.write("=" * 50 + "\n")
        handle.write(f"Video Source:  {input_video}\n")
        handle.write(f"Target: text='{prompt}'\n")
        handle.write("-" * 50 + "\n")
        handle.write(f"FINAL PDI SCORE: {report['pdi_score']:.4f}\n")
        handle.write(f"OVERALL GRADE:   {report['grade']}\n")
        handle.write("-" * 50 + "\n")
        handle.write("INDICATOR BREAKDOWN:\n")
        handle.write(f" - Scale Component (1/Z Law):      {breakdown.get('scale_component', 0):.4f}\n")
        handle.write(f" - Trajectory Component (H-X):     {breakdown.get('traj_component', 0):.4f}\n")
        handle.write(f" - Epsilon Rigidity: {breakdown.get('epsilon_rigidity', 0):.4f}\n")
        handle.write(f" - Rigidity Strategy:              {breakdown.get('rigidity_strategy', 'N/A')}\n")
        handle.write(f" - VP Component (View Consistency):{breakdown.get('vp_component', 0):.4f}\n")
        if ra is not None:
            math = ra.get("math", {})
            handle.write("-" * 50 + "\n")
            handle.write("RECONSTRUCTION AUDIT:\n")
            handle.write(f" - RA Math Pass:    {math.get('math_pass')}\n")
            handle.write(f" - RA Ground RMSE:  {math.get('ground_rmse')}\n")
            handle.write(f" - RA Scale Jump:   {math.get('scale_jump')}\n")
            handle.write(f" - RA Reproj Err:   {math.get('reprojection_residual')}\n")
            mllm = ra.get("mllm")
            if mllm is not None:
                handle.write(f" - RA MLLM Success: {mllm.get('reconstruction_success')}\n")
                handle.write(f" - RA MLLM Score:   {mllm.get('score')}\n")
                if mllm.get("reason"):
                    handle.write(f" - RA MLLM Reason:  {mllm.get('reason')}\n")
            handle.write(f" - RA Overall Pass: {ra.get('overall_pass')}\n")
        handle.write("-" * 50 + "\n")
        handle.write(f"Results generated at: {report_dir}\n")
        handle.write("=" * 50 + "\n")
    return report_path


def render_artifacts(pipeline: Any, report: dict[str, Any], input_video: Path, report_dir: Path) -> None:
    from pdi_eval.utils.visualizer import EvidenceVisualizer
    import cv2
    import numpy as np

    viz = EvidenceVisualizer(output_dir=str(report_dir))
    stem = input_video.stem
    viz.draw_error_curves(report["breakdown"]["scale_history"], report["breakdown"]["traj_history"], stem)
    viz.draw_volume_stability(report["breakdown"]["volume_history"], stem)

    if pipeline.last_masks is not None:
        cap = cv2.VideoCapture(str(input_video))
        raw_frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        raw_arr = np.array(raw_frames) if raw_frames else None
        viz.save_mask_sample(pipeline.last_masks, raw_arr, stem)


def maybe_limit_items(items: list[EvalRow], limit_per_provider: int | None) -> list[EvalRow]:
    if limit_per_provider is None:
        return items
    kept: list[EvalRow] = []
    counts: dict[str, int] = {}
    for item in items:
        count = counts.get(item.provider, 0)
        if count >= limit_per_provider:
            continue
        kept.append(item)
        counts[item.provider] = count + 1
    return kept


def run_one(
    item: EvalRow,
    pipeline: Any,
    args: argparse.Namespace,
    run_root: Path,
) -> dict[str, Any]:
    provider_root = run_root / "outputs" / item.provider
    staged_dir = provider_root / "staged"
    staged_video = staged_dir / f"{item.unique_id}.mp4"
    report_dir = provider_root / "reports" / item.unique_id
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{item.unique_id}_pdi_report.txt"
    cache_dir = provider_root / "cache" / item.unique_id
    build_eval_config(args.pdi_root / "configs" / "default.yaml", cache_dir)
    stage_video(item.file_path, staged_video)

    if report_path.exists() and not args.refresh:
        metrics = parse_report(report_path)
    else:
        pipeline.set_cache_dir(str(cache_dir))
        os.environ["PDI_EVAL_VIDEO_ID"] = item.unique_id
        report = pipeline.run(
            video_path=str(staged_video),
            text_query=item.prompt,
            render_output_dir=str(report_dir),
        )
        render_artifacts(pipeline, report, staged_video, report_dir)
        report_path = write_report(report, staged_video, report_dir, item.prompt)
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


def main() -> None:
    args = parse_args()
    providers = set(args.providers)
    run_root = args.output_root
    run_root.mkdir(parents=True, exist_ok=True)

    os.environ["PDI_FLORENCE_MODEL_ID"] = str(args.florence_model)
    os.environ["PDI_KEEP_FLORENCE_LOADED"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    import sys

    sys.path.insert(0, str(args.pdi_root / "src"))
    from pdi_eval.pipeline import PDIEvaluationPipeline

    items = maybe_limit_items(load_metadata(args.dataset_root, providers), args.limit_per_provider)
    base_config = load_base_config(args.pdi_root)
    base_config["cache_dir"] = str(run_root / "bootstrap_cache")
    pipeline = PDIEvaluationPipeline(config=base_config)

    per_video_rows: list[dict[str, Any]] = []
    total = len(items)
    for idx, item in enumerate(items, start=1):
        print(f"[{idx}/{total}] {item.provider} :: {item.task} :: {item.prompt}", flush=True)
        per_video_rows.append(run_one(item, pipeline, args, run_root))

    per_video_csv = run_root / "results" / "per_video_metrics.csv"
    write_csv(per_video_csv, per_video_rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in per_video_rows:
        grouped.setdefault(row["paper_name"], []).append(row)

    summary_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    for paper_name, rows in grouped.items():
        summary = summarize_provider(rows, bootstrap_iters=args.bootstrap_iters, seed=args.seed)
        summary_rows.append({"paper_name": paper_name, **summary})
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
        "runner_mode": "persistent_single_process",
        "paper_vs_code_notes": [
            "该版本复用同一个 PDIEvaluationPipeline，避免每个 case 重新加载 SAM2、CoTracker 和 Florence-2。",
            "Mega-SAM 仍然按视频调用外部脚本，但前端模型重载开销已去掉。",
            "论文 Table 1 与附录 A.3 的统计口径不一致，因此结果同时保留 mean-like 与 median-like 聚合解释。",
        ],
    }
    (run_root / "results" / "notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "per_video_csv": str(per_video_csv),
            "provider_summary_csv": str(run_root / "results" / "provider_summary.csv"),
            "compare_csv": str(run_root / "results" / "table1_reproduction_compare.csv"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
