#!/usr/bin/env python3
"""Analyze S-head subtype and depth ablations with case-clustered statistics."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stc_motion import (
    PLAUSIBILITY_COMPONENTS,
    PLAUSIBILITY_SCALES,
    impact_score,
    load_features,
    pair_metrics,
    safe_nanmean,
    track_state,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
)
DEFAULT_REPORT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/motion-n-analysis"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
SUBTYPE_LABELS = {
    "local_enrichment": "Local-enrichment S",
    "same_frame_mass": "Same-frame-mass S",
    "local_same_union": "Local + Same union",
}
DEPTH_LABELS = {
    "early": "Early B00-09",
    "middle": "Middle B10-19",
    "late": "Late B20-29",
}
DOMINANT_LABELS = {
    "S_local_dominant_all": "Local dominant | all",
    "S_local_dominant_depth_early": "Local dominant | B00-09",
    "S_local_dominant_depth_middle": "Local dominant | B10-19",
    "S_local_dominant_depth_late": "Local dominant | B20-29",
    "S_same_frame_dominant_all": "Same-frame dominant | all",
    "S_same_frame_dominant_depth_early": "Same-frame dominant | B00-09",
    "S_same_frame_dominant_depth_middle": "Same-frame dominant | B10-19",
    "S_same_frame_dominant_depth_late": "Same-frame dominant | B20-29",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUTPUT_ROOT / "inventory.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def calibrated_plausibility(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    scaled = []
    for component in PLAUSIBILITY_COMPONENTS:
        column = f"{component}_scaled"
        result[column] = np.log1p(
            np.maximum(result[component].astype(float), 0.0)
            / PLAUSIBILITY_SCALES[component]
        ).fillna(np.log1p(10.0))
        scaled.append(column)
    result["plausibility_distance_gt"] = result[scaled].mean(axis=1)
    baseline_distance = (
        result[result["family"] == "baseline"]
        .set_index(["case_id", "model", "seed"])["plausibility_distance_gt"]
        .to_dict()
    )
    result["gt_gain_vs_baseline"] = [
        baseline_distance.get((row.case_id, row.model, row.seed), np.nan)
        - row.plausibility_distance_gt
        for row in result.itertuples()
    ]
    return result


def bootstrap_case_means(
    case_values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(case_values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(frame: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    keys = [
        "family",
        "model",
        "variant",
        "subset_id",
        "subtype",
        "depth_stratum",
        "dominance_class",
        "head_count",
        "denoise_start",
        "denoise_end",
    ]
    rng = np.random.default_rng(42)
    rows = []
    for key, group in frame.groupby(keys, dropna=False):
        case_level = (
            group.groupby("case_id", as_index=False)
            .agg(
                impact_score=("impact_score", "mean"),
                gt_gain_vs_baseline=("gt_gain_vs_baseline", "mean"),
                tracking_failure=("tracking_failure", "mean"),
            )
        )
        impact = case_level["impact_score"].to_numpy(float)
        gain = case_level["gt_gain_vs_baseline"].to_numpy(float)
        finite_impact = impact[np.isfinite(impact)]
        finite_gain = gain[np.isfinite(gain)]
        impact_low, impact_high = bootstrap_case_means(
            impact, bootstrap_samples, rng
        )
        gain_low, gain_high = bootstrap_case_means(gain, bootstrap_samples, rng)
        head_count = int(key[7])
        rows.append(
            {
                **dict(zip(keys, key)),
                "n_cases": int(group["case_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "n_pairs": len(group),
                "tracking_failure_rate": float(group["tracking_failure"].mean()),
                "impact_mean": float(np.nanmean(impact)),
                "impact_std_across_cases": (
                    float(np.std(finite_impact, ddof=1))
                    if len(finite_impact) > 1
                    else 0.0
                ),
                "impact_ci_low": impact_low,
                "impact_ci_high": impact_high,
                "impact_per_head_approx": (
                    float(np.nanmean(impact)) / head_count if head_count else 0.0
                ),
                "gt_gain_mean": float(np.nanmean(gain)),
                "gt_gain_std_across_cases": (
                    float(np.std(finite_gain, ddof=1))
                    if len(finite_gain) > 1
                    else 0.0
                ),
                "gt_gain_ci_low": gain_low,
                "gt_gain_ci_high": gain_high,
            }
        )
    result = pd.DataFrame(rows)
    result["gt_gain_label"] = np.where(
        result["n_cases"] < 3,
        "insufficient_cases",
        np.where(
            result["gt_gain_ci_low"] > 0,
            "closer_to_gt",
            np.where(result["gt_gain_ci_high"] < 0, "farther_from_gt", "uncertain"),
        ),
    )
    return result


def build_interaction_table(aggregate_frame: pd.DataFrame) -> pd.DataFrame:
    feature = aggregate_frame[aggregate_frame["family"] == "s_feature"]
    rows = []
    for (model, start, end), group in feature.groupby(
        ["model", "denoise_start", "denoise_end"]
    ):
        values = group.set_index("subtype")
        if not all(name in values.index for name in SUBTYPE_LABELS):
            continue
        local = float(values.loc["local_enrichment", "impact_mean"])
        same = float(values.loc["same_frame_mass", "impact_mean"])
        union = float(values.loc["local_same_union", "impact_mean"])
        rows.append(
            {
                "model": model,
                "denoise_start": int(start),
                "denoise_end": int(end),
                "local_impact": local,
                "same_impact": same,
                "union_impact": union,
                "union_minus_mean_single": union - 0.5 * (local + same),
                "union_minus_max_single": union - max(local, same),
            }
        )
    return pd.DataFrame(rows)


def save_family_plot(
    aggregate_frame: pd.DataFrame,
    family: str,
    output: Path,
) -> None:
    data = aggregate_frame[aggregate_frame["family"] == family].copy()
    if family == "s_feature":
        label_column, labels = "subtype", SUBTYPE_LABELS
        family_title = "S feature subtype"
    elif family == "s_depth":
        label_column, labels = "depth_stratum", DEPTH_LABELS
        family_title = "S depth strata"
    else:
        label_column, labels = "subset_id", DOMINANT_LABELS
        family_title = "S feature dominance x depth"
    stages = sorted(
        {
            (int(row.denoise_start), int(row.denoise_end))
            for row in data.itertuples()
        }
    )
    metrics = (
        ("impact_mean", "Motion Impact", "viridis", False),
        ("impact_per_head_approx", "Impact / head (approx.)", "magma", False),
        ("gt_gain_mean", "GT gain", "RdYlGn", True),
    )
    figure, axes = plt.subplots(3, 3, figsize=(15, 12), squeeze=False)
    for model_index, model in enumerate(MODEL_LABELS):
        subset = data[data["model"] == model]
        for metric_index, (metric, title, cmap_name, diverging) in enumerate(metrics):
            matrix = np.full((len(labels), len(stages)), np.nan)
            for row_index, key in enumerate(labels):
                for stage_index, stage in enumerate(stages):
                    cells = subset[
                        (subset[label_column] == key)
                        & (subset["denoise_start"] == stage[0])
                        & (subset["denoise_end"] == stage[1])
                    ]
                    if len(cells):
                        matrix[row_index, stage_index] = float(cells.iloc[0][metric])
            axis = axes[model_index, metric_index]
            finite = matrix[np.isfinite(matrix)]
            kwargs: dict[str, Any] = {"cmap": cmap_name, "aspect": "auto"}
            if diverging and len(finite):
                bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
                kwargs.update(vmin=-bound, vmax=bound)
            image = axis.imshow(matrix, **kwargs)
            axis.set_xticks(
                range(len(stages)),
                [f"[{start},{end})" for start, end in stages],
                rotation=30,
                ha="right",
            )
            axis.set_yticks(range(len(labels)), list(labels.values()))
            axis.set_title(f"{MODEL_LABELS[model]} | {title}")
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    if np.isfinite(matrix[y, x]):
                        axis.text(
                            x,
                            y,
                            f"{matrix[y, x]:+.3f}" if diverging else f"{matrix[y, x]:.3f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                        )
            figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    figure.suptitle(
        family_title,
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def format_table(aggregate_frame: pd.DataFrame, family: str) -> str:
    data = aggregate_frame[aggregate_frame["family"] == family].sort_values(
        ["model", "impact_mean"], ascending=[True, False]
    )
    rows = []
    for row in data.itertuples():
        if family == "s_feature":
            category = SUBTYPE_LABELS.get(row.subtype, row.subtype)
        elif family == "s_depth":
            category = DEPTH_LABELS.get(row.depth_stratum, row.depth_stratum)
        else:
            category = DOMINANT_LABELS.get(row.subset_id, row.subset_id)
        gain_class = (
            "good"
            if row.gt_gain_label == "closer_to_gt"
            else "bad" if row.gt_gain_label == "farther_from_gt" else "uncertain"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
            f"<td>{html.escape(str(category))}</td>"
            f"<td>[{int(row.denoise_start)},{int(row.denoise_end)})</td>"
            f"<td>{int(row.head_count)}</td>"
            f"<td>{int(row.n_cases)} / {int(row.n_seeds)}</td>"
            f"<td>{row.impact_mean:.3f} [{row.impact_ci_low:.3f}, {row.impact_ci_high:.3f}]</td>"
            f"<td>{row.impact_per_head_approx:.5f}</td>"
            f"<td class='{gain_class}'>{row.gt_gain_mean:+.3f} "
            f"[{row.gt_gain_ci_low:+.3f}, {row.gt_gain_ci_high:+.3f}]</td>"
            f"<td>{100.0 * row.tracking_failure_rate:.1f}%</td>"
            "</tr>"
        )
    return "".join(rows)


def build_html(
    aggregate_frame: pd.DataFrame,
    per_video: pd.DataFrame,
    interactions: pd.DataFrame,
) -> str:
    feature_rows = format_table(aggregate_frame, "s_feature")
    depth_rows = format_table(aggregate_frame, "s_depth")
    dominant_rows = format_table(aggregate_frame, "s_dominant_depth")
    interaction_rows = "".join(
        "<tr>"
        f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
        f"<td>[{int(row.denoise_start)},{int(row.denoise_end)})</td>"
        f"<td>{row.local_impact:.3f}</td><td>{row.same_impact:.3f}</td>"
        f"<td>{row.union_impact:.3f}</td>"
        f"<td>{row.union_minus_mean_single:+.3f}</td>"
        f"<td>{row.union_minus_max_single:+.3f}</td>"
        "</tr>"
        for row in interactions.itertuples()
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S Head 运动影响分析</title><style>
:root{{--bg:#101416;--panel:#1a2024;--line:#39434a;--text:#edf2f4;--muted:#aab5bb;--good:#63c59d;--bad:#eb7d74;--warn:#dfb663}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:13px/1.45 Arial,"Noto Sans SC",sans-serif;letter-spacing:0}}
header,main{{max-width:1500px;margin:auto;padding:15px 18px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}h1{{font-size:23px}}h2{{font-size:18px;margin:25px 0 8px}}.muted{{color:var(--muted)}}.notes{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}}.note{{background:var(--panel);border:1px solid var(--line);padding:9px}}
img{{display:block;width:100%;background:#fff;border:1px solid var(--line);margin-top:8px}}.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{border:1px solid var(--line);padding:6px 8px;white-space:nowrap;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{background:#252d32;position:sticky;top:0}}.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}.uncertain{{color:var(--warn)}}a{{color:#70cdb8}}
@media(max-width:900px){{.notes{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>S Head 子类别 × 去噪阶段 × 深度：运动影响</h1>
<p class="muted">已完成 {len(per_video)} 个生成视频的配对分析；每个视频与同 case、同模型、同 seed 的 baseline 比较。</p></header><main>
<div class="notes">
<div class="note"><b>Motion Impact</b><br>RAFT 向量场、强运动曲线、物体轨迹和速度相对 baseline 的归一化差异。越高表示运动改变越大，不表示更合理。</div>
<div class="note"><b>GT gain</b><br>正值表示消融结果比 baseline 更接近同 case 的 49 帧 GT；负值表示更远。区间为按 case 聚类 bootstrap 的 95% CI。</div>
<div class="note"><b>数量控制</b><br>Local 与 Same 都是 32 heads，可直接比较。Impact/head 仅用于 64-head union 和不同深度 head 数的敏感性检查；由于网络非线性，它不是单 head 因果贡献。</div>
</div>
<h2>S 子类别</h2><img src="s_feature_motion_heatmaps.png" alt="S subtype motion heatmaps">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>类别</th><th>阶段</th><th>Heads</th><th>Cases / Seeds</th><th>Impact [95% CI]</th><th>Impact/head</th><th>GT gain [95% CI]</th><th>追踪失败</th></tr></thead><tbody>{feature_rows}</tbody></table></div>
<h2>S 深度组合</h2><img src="s_depth_motion_heatmaps.png" alt="S depth motion heatmaps">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>深度</th><th>阶段</th><th>Heads</th><th>Cases / Seeds</th><th>Impact [95% CI]</th><th>Impact/head</th><th>GT gain [95% CI]</th><th>追踪失败</th></tr></thead><tbody>{depth_rows}</tbody></table></div>
<h2>S 主导特征 × 深度</h2><img src="s_dominant_depth_motion_heatmaps.png" alt="S dominance and depth motion heatmaps">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>主导类别 / 深度</th><th>阶段</th><th>Heads</th><th>Cases / Seeds</th><th>Impact [95% CI]</th><th>Impact/head</th><th>GT gain [95% CI]</th><th>追踪失败</th></tr></thead><tbody>{dominant_rows}</tbody></table></div>
<h2>Local + Same 联合诊断</h2><p class="muted">若 union−max(single) 为正，64-head 联合消融比任一 32-head 单类改变更大；该差值仍受 head 数影响，不解释为线性交互因果量。</p>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>阶段</th><th>Local</th><th>Same</th><th>Union</th><th>Union−mean(single)</th><th>Union−max(single)</th></tr></thead><tbody>{interaction_rows}</tbody></table></div>
<p class="muted" style="margin-top:12px">原始结果：<a href="per_video_metrics.csv">per_video_metrics.csv</a> · <a href="aggregate_metrics.csv">aggregate_metrics.csv</a> · <a href="interaction_diagnostics.csv">interaction_diagnostics.csv</a> · <a href="ranking_by_model.csv">ranking_by_model.csv</a> · <a href="/s-head-ablation/">对应视频页</a></p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    entries = {entry["entry_id"]: entry for entry in inventory["entries"]}
    features = {
        entry_id: loaded
        for entry_id, entry in entries.items()
        if (loaded := load_features(args.output_root, entry)) is not None
    }
    if len(features) != len(entries):
        raise RuntimeError(
            f"Feature extraction incomplete: {len(features)}/{len(entries)}"
        )
    start_frame = args.context_frames - 1
    gt_by_case = {}
    baseline_by_key = {}
    for entry_id, (arrays, metadata) in features.items():
        entry = entries[entry_id]
        if entry["family"] == "gt":
            gt_by_case[entry["case_id"]] = (
                arrays,
                track_state(arrays, metadata, start_frame),
            )
        elif entry["family"] == "baseline":
            baseline_by_key[
                (entry["case_id"], entry["model"], int(entry["seed"]))
            ] = (
                arrays,
                track_state(arrays, metadata, start_frame),
            )

    rows = []
    for entry_id, (arrays, metadata) in features.items():
        entry = entries[entry_id]
        if entry["kind"] != "generated":
            continue
        key = (entry["case_id"], entry["model"], int(entry["seed"]))
        if key not in baseline_by_key or entry["case_id"] not in gt_by_case:
            raise RuntimeError(f"Missing paired baseline or GT for {entry_id}")
        baseline_arrays, baseline_state = baseline_by_key[key]
        gt_arrays, gt_state = gt_by_case[entry["case_id"]]
        state = track_state(arrays, metadata, start_frame)
        gt_metrics = pair_metrics(
            arrays, state, gt_arrays, gt_state, start_frame, "gt"
        )
        baseline_metrics = pair_metrics(
            arrays, state, baseline_arrays, baseline_state, start_frame, "baseline"
        )
        stage = entry["denoise_step_range"] or [np.nan, np.nan]
        rows.append(
            {
                "entry_id": entry_id,
                "family": entry["family"],
                "case_id": entry["case_id"],
                "model": entry["model"],
                "seed": int(entry["seed"]),
                "variant": entry["variant"],
                "subset_id": entry.get("subset_id"),
                "subtype": entry.get("subtype"),
                "depth_stratum": entry.get("depth_stratum"),
                "dominance_class": entry.get("dominance_class"),
                "head_count": int(entry["head_count"]),
                "denoise_start": stage[0],
                "denoise_end": stage[1],
                "impact_score": (
                    0.0
                    if entry["family"] == "baseline"
                    else impact_score(
                        baseline_metrics,
                        baseline_arrays,
                        baseline_state,
                        start_frame,
                    )
                ),
                "tracking_failure": state["tracking_failure"],
                "object_visibility": state["object_visibility"],
                "background_drift_mean": safe_nanmean(state["background_drift_curve"]),
                "object_displacement_mean": safe_nanmean(state["object_displacement_curve"]),
                "object_speed_mean": safe_nanmean(state["object_speed_curve"]),
                "object_acceleration_mean": safe_nanmean(state["object_acceleration_curve"]),
                **gt_metrics,
                **baseline_metrics,
            }
        )
    per_video = calibrated_plausibility(pd.DataFrame(rows))
    ablations = per_video[
        per_video["family"].isin(
            ("s_feature", "s_depth", "s_dominant_depth")
        )
    ]
    aggregate_frame = aggregate(ablations, args.bootstrap_samples)
    interactions = build_interaction_table(aggregate_frame)
    ranking = aggregate_frame.sort_values(
        ["model", "impact_mean"], ascending=[True, False]
    ).copy()
    ranking.insert(1, "impact_rank_within_model", ranking.groupby("model").cumcount() + 1)

    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(results_dir / "per_video_metrics.csv", index=False)
    aggregate_frame.to_csv(results_dir / "aggregate_metrics.csv", index=False)
    interactions.to_csv(results_dir / "interaction_diagnostics.csv", index=False)
    ranking.to_csv(results_dir / "ranking_by_model.csv", index=False)
    protocol = {
        "schema_version": 1,
        "context_frames": args.context_frames,
        "first_evaluated_flow_pair": [start_frame, start_frame + 1],
        "track_query_frame": start_frame,
        "region_method": "per-case SAM2 AMG at final context frame",
        "spatial_preprocess": "center crop to 7:4 then resize",
        "impact_reference": "same case, model, and seed baseline",
        "gt_reference": "49 frames at 30 FPS and 896x512",
        "bootstrap_unit": "case; seed values are averaged within each case",
        "bootstrap_samples": args.bootstrap_samples,
        "plausibility_components": list(PLAUSIBILITY_COMPONENTS),
        "plausibility_scales": PLAUSIBILITY_SCALES,
        "impact_per_head_warning": (
            "Approximate sensitivity normalization only; transformer ablations "
            "are nonlinear and head interactions are not additive."
        ),
    }
    atomic_write(
        results_dir / "protocol.json",
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    save_family_plot(
        aggregate_frame,
        "s_feature",
        args.report_dir / "s_feature_motion_heatmaps.png",
    )
    save_family_plot(
        aggregate_frame,
        "s_depth",
        args.report_dir / "s_depth_motion_heatmaps.png",
    )
    save_family_plot(
        aggregate_frame,
        "s_dominant_depth",
        args.report_dir / "s_dominant_depth_motion_heatmaps.png",
    )
    per_video.to_csv(args.report_dir / "per_video_metrics.csv", index=False)
    aggregate_frame.to_csv(args.report_dir / "aggregate_metrics.csv", index=False)
    interactions.to_csv(args.report_dir / "interaction_diagnostics.csv", index=False)
    ranking.to_csv(args.report_dir / "ranking_by_model.csv", index=False)
    atomic_write(
        args.report_dir / "index.html",
        build_html(aggregate_frame, per_video, interactions),
    )
    print(
        f"[analyze-s-motion] features={len(features)} per_video={len(per_video)} "
        f"aggregate={len(aggregate_frame)} report={args.report_dir / 'index.html'}"
    )


if __name__ == "__main__":
    main()
