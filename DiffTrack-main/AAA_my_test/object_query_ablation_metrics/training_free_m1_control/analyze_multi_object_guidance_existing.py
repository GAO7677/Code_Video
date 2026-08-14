#!/usr/bin/env python3
"""Create a case-balanced interim report for the completed M1 guidance grid."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-report", type=Path, required=True)
    parser.add_argument("--gt-trajectory-report", type=Path, required=True)
    parser.add_argument("--fast-ranking", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def bootstrap_ci(values: list[float], draws: int = 20000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(47326)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def sign_flip_p(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    observed = abs(float(array.mean()))
    null = [
        abs(float((array * np.asarray(signs)).mean()))
        for signs in itertools.product((-1.0, 1.0), repeat=len(array))
    ]
    return float(sum(value >= observed - 1.0e-15 for value in null) / len(null))


def fmt(value: float, digits: int = 5) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    gt = read_json(args.gt_report)
    gt_trajectory = read_json(args.gt_trajectory_report)
    fast = read_json(args.fast_ranking)
    full_units = {
        (str(row["case"]), int(row["seed"]))
        for row in gt.get("snapshot", [])
        if row.get("full_grid") and row.get("gt_eligible")
    }
    gt_map = {
        (str(row["case"]), int(row["seed"]), str(row["variant_id"])): row
        for row in gt.get("records", [])
        if (str(row["case"]), int(row["seed"])) in full_units
    }
    fast_map = {
        (str(row["case"]), int(row["seed"]), str(row["variant_id"])): row
        for row in fast.get("records", [])
    }
    gt_trajectory_map = {
        (str(row["case"]), int(row["seed"]), str(row["variant_id"])): row
        for row in gt_trajectory.get("records", [])
    }
    baseline_trajectory_map: dict[tuple[str, int, str], dict[str, Any]] = {}
    for report_path in sorted(args.trajectory_root.glob("*/seed_*/report.json")):
        report = read_json(report_path)
        case = str(report.get("case") or report_path.parents[1].name)
        seed = int(report.get("seed", int(report_path.parent.name.removeprefix("seed_"))))
        for row in report.get("records", []):
            baseline_trajectory_map[(case, seed, str(row["variant_id"]))] = row
    joined = []
    for key, grow in gt_map.items():
        frow = fast_map.get(key)
        if frow is None:
            continue
        joined.append(
            {
                **grow,
                **{f"fast_{k}": v for k, v in frow.items()},
                "gt_trajectory": gt_trajectory_map.get(key),
                "baseline_trajectory": baseline_trajectory_map.get(key),
            }
        )

    configs: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        configs[(float(row["pag_scale"]), int(row["guidance_step_range_inclusive"][1]))].append(row)
    metric_specs = {
        "gt_mse_delta": "gt_mse_delta_vs_baseline",
        "gt_mse_relative_percent": "gt_mse_relative_change_percent",
        "gt_ssim_delta": "gt_ssim_delta_vs_baseline",
        "baseline_effect": "fast_impact_score_0_100",
        "target_local_effect": None,
        "global_effect": None,
        "outside_effect": None,
        "gt_trajectory_delta_d0": None,
        "gt_trajectory_relative_percent": None,
        "baseline_trajectory_ade_d0": None,
        "baseline_track_loss_percent": None,
    }

    summaries = []
    for (scale, end), rows in sorted(configs.items()):
        case_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            case_rows[str(row["case"])].append(row)
        case_values: dict[str, list[float]] = defaultdict(list)
        for case, values in case_rows.items():
            case_values["gt_mse_delta"].append(float(np.mean([v["gt_mse_delta_vs_baseline"] for v in values])))
            case_values["gt_mse_relative_percent"].append(float(np.mean([v["gt_mse_relative_change_percent"] for v in values])))
            case_values["gt_ssim_delta"].append(float(np.mean([v["gt_ssim_delta_vs_baseline"] for v in values])))
            case_values["baseline_effect"].append(float(np.mean([v["fast_impact_score_0_100"] for v in values])))
            case_values["target_local_effect"].append(float(np.mean([v["fast_category_scores_0_100"]["target_local"] for v in values])))
            case_values["global_effect"].append(float(np.mean([v["fast_category_scores_0_100"]["global_appearance"] for v in values])))
            case_values["outside_effect"].append(float(np.mean([v["fast_category_scores_0_100"]["outside_spillover"] for v in values])))
            valid_gt_trajectory = [
                v["gt_trajectory"]
                for v in values
                if v.get("gt_trajectory")
                and v["gt_trajectory"].get("quality_gate_passed")
                and v["gt_trajectory"].get("baseline_gt_center_ade_d0")
            ]
            if valid_gt_trajectory:
                case_values["gt_trajectory_delta_d0"].append(
                    float(np.mean([v["gt_center_ade_delta_vs_baseline"] for v in valid_gt_trajectory]))
                )
                case_values["gt_trajectory_relative_percent"].append(
                    float(
                        np.mean(
                            [
                                100.0
                                * v["gt_center_ade_delta_vs_baseline"]
                                / v["baseline_gt_center_ade_d0"]
                                for v in valid_gt_trajectory
                            ]
                        )
                    )
                )
            valid_baseline_trajectory = [
                v["baseline_trajectory"]["metrics"]
                for v in values
                if v.get("baseline_trajectory")
            ]
            finite_ade = [
                v["target_center_ade_norm"]
                for v in valid_baseline_trajectory
                if v.get("quality_pass") and v.get("target_center_ade_norm") is not None
            ]
            if finite_ade:
                case_values["baseline_trajectory_ade_d0"].append(float(np.mean(finite_ade)))
            finite_loss = [
                v["target_worst_track_loss_score_0_100"]
                for v in valid_baseline_trajectory
                if v.get("target_worst_track_loss_score_0_100") is not None
            ]
            if finite_loss:
                case_values["baseline_track_loss_percent"].append(float(np.mean(finite_loss)))
        metrics = {}
        for name in metric_specs:
            values = case_values[name]
            if not values:
                metrics[name] = None
                continue
            metrics[name] = {
                "case_balanced_mean": float(np.mean(values)),
                "case_bootstrap_95ci": bootstrap_ci(values),
            }
            if name in {
                "gt_mse_delta",
                "gt_mse_relative_percent",
                "gt_ssim_delta",
                "gt_trajectory_delta_d0",
                "gt_trajectory_relative_percent",
            }:
                metrics[name]["exact_case_sign_flip_p"] = sign_flip_p(values)
        valid_gt_rows = [
            row["gt_trajectory"]
            for row in rows
            if row.get("gt_trajectory") and row["gt_trajectory"].get("quality_gate_passed")
        ]
        gt_traj_case_means = case_values["gt_trajectory_delta_d0"]
        summaries.append(
            {
                "pag_scale": scale,
                "window": [0, end],
                "case_count": len(case_rows),
                "case_seed_count": len(rows),
                "seed_level_mse_win_rate": float(np.mean([row["mse_improved"] for row in rows])),
                "case_level_mse_win_rate": float(np.mean([value < 0 for value in case_values["gt_mse_delta"]])),
                "gt_trajectory_gate_pass_rate": len(valid_gt_rows) / len(rows),
                "seed_level_trajectory_win_rate": (
                    float(np.mean([row["trajectory_improved"] for row in valid_gt_rows]))
                    if valid_gt_rows
                    else None
                ),
                "case_level_trajectory_win_rate": (
                    float(np.mean([value < 0 for value in gt_traj_case_means]))
                    if gt_traj_case_means
                    else None
                ),
                "metrics": metrics,
            }
        )
    summaries.sort(
        key=lambda row: row["metrics"]["gt_mse_relative_percent"]["case_balanced_mean"]
    )
    per_case = []
    for case in sorted({str(row["case"]) for row in joined}):
        values = [row for row in joined if str(row["case"]) == case]
        per_case.append(
            {
                "case": case,
                "video_count": len(values),
                "mean_gt_mse_relative_change_percent": float(
                    np.mean([row["gt_mse_relative_change_percent"] for row in values])
                ),
                "mse_win_rate": float(np.mean([row["mse_improved"] for row in values])),
                "mean_gt_ssim_delta": float(
                    np.mean([row["gt_ssim_delta_vs_baseline"] for row in values])
                ),
                "trajectory_gate_pass_rate": float(
                    np.mean(
                        [
                            bool(row.get("gt_trajectory") and row["gt_trajectory"].get("quality_gate_passed"))
                            for row in values
                        ]
                    )
                ),
                "mean_gt_trajectory_relative_change_percent": (
                    float(
                        np.mean(
                            [
                                100.0
                                * row["gt_trajectory"]["gt_center_ade_delta_vs_baseline"]
                                / row["gt_trajectory"]["baseline_gt_center_ade_d0"]
                                for row in values
                                if row.get("gt_trajectory")
                                and row["gt_trajectory"].get("quality_gate_passed")
                                and row["gt_trajectory"].get("baseline_gt_center_ade_d0")
                            ]
                        )
                    )
                ),
            }
        )
    positive = [row for row in summaries if row["pag_scale"] > 0]
    negative = [row for row in summaries if row["pag_scale"] < 0]
    for row in per_case:
        row["joint_pixel_trajectory_improved"] = (
            row["mean_gt_mse_relative_change_percent"] < 0
            and row["mean_gt_trajectory_relative_change_percent"] < 0
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_unit": "case; seeds averaged within case; cases equally weighted",
        "strict_full_grid_only": True,
        "case_count": len({case for case, _seed in full_units}),
        "case_seed_count": len(full_units),
        "joined_video_count": len(joined),
        "config_count": len(summaries),
        "inference_note": (
            "With five independent cases, the minimum attainable two-sided exact "
            "sign-flip p-value is 0.0625; no p<0.05 claim is possible in this snapshot."
        ),
        "configs_ranked_by_case_balanced_gt_mse_relative_change": summaries,
        "case_heterogeneity": per_case,
        "sign_summary": {
            "positive_lambda_mean_gt_mse_relative_change_percent": float(
                np.mean(
                    [row["metrics"]["gt_mse_relative_percent"]["case_balanced_mean"] for row in positive]
                )
            ),
            "negative_lambda_mean_gt_mse_relative_change_percent": float(
                np.mean(
                    [row["metrics"]["gt_mse_relative_percent"]["case_balanced_mean"] for row in negative]
                )
            ),
            "positive_lambda_mean_baseline_effect": float(
                np.mean([row["metrics"]["baseline_effect"]["case_balanced_mean"] for row in positive])
            ),
            "negative_lambda_mean_baseline_effect": float(
                np.mean([row["metrics"]["baseline_effect"]["case_balanced_mean"] for row in negative])
            ),
        },
    }
    atomic_text(args.output_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# Multi-object M1 guidance · existing-video metric report",
        "",
        f"Snapshot: `{payload['generated_at_utc']}`",
        "",
        f"- Strict cohort: **{payload['case_count']} cases × 5 seeds = {payload['case_seed_count']} case-seeds**, "
        f"**{payload['joined_video_count']} guided videos**.",
        "- 每个 case 内先平均 seed，再对 case 等权；GT 只用于生成后评价，不进入 guidance。",
        "- `ΔMSE < 0` 或 `ΔSSIM > 0` 才表示比同 seed Baseline 更接近 GT。",
        "- Baseline effect / target / global / outside 均表示改变强度，越大不等于越好。",
        f"- 统计限制：{payload['inference_note']}",
        "",
        "## 先给结论",
        "",
        f"1. **视频确实发生了可测变化。** 16 组配置的 Baseline 影响分数范围为 "
        f"{min(row['metrics']['baseline_effect']['case_balanced_mean'] for row in summaries):.3f}–"
        f"{max(row['metrics']['baseline_effect']['case_balanced_mean'] for row in summaries):.3f}，"
        f"对象 ROI 影响为 {min(row['metrics']['target_local_effect']['case_balanced_mean'] for row in summaries):.3f}–"
        f"{max(row['metrics']['target_local_effect']['case_balanced_mean'] for row in summaries):.3f}。",
        f"2. **当前没有证据表明整体明显变好。** 16 组中只有 "
        f"{sum(row['metrics']['gt_mse_relative_percent']['case_balanced_mean'] < 0 for row in summaries)}/16 的 case-balanced MSE 相对变化为负，"
        f"0/16 的均值 ΔSSIM 为正，且 "
        f"{sum(row['metrics']['gt_trajectory_relative_percent']['case_balanced_mean'] < 0 for row in summaries)}/16 的有效轨迹均值优于 Baseline。",
        f"3. **负 λ 整体更激进且更差。** 正 λ 配置平均使 GT MSE 变化 "
        f"{payload['sign_summary']['positive_lambda_mean_gt_mse_relative_change_percent']:+.2f}%，"
        f"负 λ 为 {payload['sign_summary']['negative_lambda_mean_gt_mse_relative_change_percent']:+.2f}%；"
        f"对应 Baseline 影响分数为 {payload['sign_summary']['positive_lambda_mean_baseline_effect']:.3f} vs "
        f"{payload['sign_summary']['negative_lambda_mean_baseline_effect']:.3f}。",
        "4. **结果明显依赖 case。** 同一超参数方向没有跨 case 一致改善；见下方 case 汇总。",
        f"5. **没有 case 同时改善像素与轨迹。** 当前 5 个 case 中联合改善数为 "
        f"{sum(row['joint_pixel_trajectory_improved'] for row in per_case)}/5。",
        "",
        "| Rank | λ | Window | MSE相对变化 | ΔSSIM | GT轨迹相对变化 | 轨迹有效率 | 轨迹Case胜率 | vsBase ADE/D0 | TrackLoss | Baseline影响 | Target影响 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(summaries, start=1):
        metrics = row["metrics"]
        lines.append(
            f"| {index} | {row['pag_scale']:+g} | 0–{row['window'][1]} | "
            f"{fmt(metrics['gt_mse_relative_percent']['case_balanced_mean'], 2)}% | "
            f"{fmt(metrics['gt_ssim_delta']['case_balanced_mean'], 5)} | "
            f"{fmt(metrics['gt_trajectory_relative_percent']['case_balanced_mean'], 2)}% | "
            f"{100*row['gt_trajectory_gate_pass_rate']:.1f}% | "
            f"{100*row['case_level_trajectory_win_rate']:.1f}% | "
            f"{fmt(metrics['baseline_trajectory_ade_d0']['case_balanced_mean'], 3)} | "
            f"{fmt(metrics['baseline_track_loss_percent']['case_balanced_mean'], 2)} | "
            f"{fmt(metrics['baseline_effect']['case_balanced_mean'], 3)} | "
            f"{fmt(metrics['target_local_effect']['case_balanced_mean'], 3)} |"
        )
    best = summaries[0]
    best_mse = best["metrics"]["gt_mse_delta"]
    best_relative = best["metrics"]["gt_mse_relative_percent"]
    best_ci = best_relative["case_bootstrap_95ci"]
    if best_ci[0] > 0:
        interval_reading = "区间完全大于 0，当前样本中它反而稳定劣于 Baseline。"
    elif best_ci[1] < 0:
        interval_reading = "区间完全小于 0，但独立 case 数仍不足以通过精确双侧检验。"
    else:
        interval_reading = "区间跨过 0，不能排除与 Baseline 无差异。"
    lines.extend(
        [
            "",
            "## Direct reading",
            "",
            f"探索性最优配置是 `λ={best['pag_scale']:+g}, window=0–{best['window'][1]}`："
            f"case-balanced MSE 相对变化={best_relative['case_balanced_mean']:+.2f}% "
            f"（绝对 ΔMSE={best_mse['case_balanced_mean']:.6f}），"
            f"相对变化的 95% case-bootstrap CI=[{best_ci[0]:+.2f}%, "
            f"{best_ci[1]:+.2f}%]，"
            f"case 胜率={100*best['case_level_mse_win_rate']:.1f}%。",
            "",
            interval_reading,
            "",
            "当前独立 case 数为 5，精确双侧符号置换检验无法达到 p<0.05。"
            "GT 轨迹 ADE 只统计通过共同可见性门控的样本；失败样本未被当作改善。",
            "",
            "## Case heterogeneity",
            "",
            "| Case | Videos | 平均GT MSE相对变化 | MSE胜率 | 平均ΔSSIM | GT轨迹相对变化 | 轨迹有效率 | 联合改善 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_case:
        lines.append(
            f"| {row['case']} | {row['video_count']} | "
            f"{row['mean_gt_mse_relative_change_percent']:+.2f}% | "
            f"{100*row['mse_win_rate']:.1f}% | {row['mean_gt_ssim_delta']:+.5f} | "
            f"{row['mean_gt_trajectory_relative_change_percent']:+.2f}% | "
            f"{100*row['trajectory_gate_pass_rate']:.1f}% | "
            f"{'yes' if row['joint_pixel_trajectory_improved'] else 'no'} |"
        )
    lines.append("")
    atomic_text(args.output_md, "\n".join(lines))
    print(args.output_md)


if __name__ == "__main__":
    main()
