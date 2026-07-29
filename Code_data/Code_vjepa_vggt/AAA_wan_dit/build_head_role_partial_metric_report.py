#!/usr/bin/env python3
"""Render live matched-triplet metric curves and tables for the pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_head_role_dose_control_case_gallery import build_records, source_cases
from summarize_head_role_dose_control import METRICS


DEFAULT_GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/"
    "head-role-dose-control-pilot"
)
BASELINE_FALLBACK = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_seed851_baseline_bench/analysis/"
    "per_video_metrics.csv"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MATCH_LABELS = {
    "approx_depth": "k=8 近似深度匹配",
    "exact_block": "k=5 完全同Block匹配",
}
PLOT_MATCH_LABELS = {
    "approx_depth": "k=8 approximate-depth match",
    "exact_block": "k=5 exact same-block match",
}
METRIC_LABELS = {
    "physics_iq_with_context": "Physics-IQ ctx",
    "physics_iq_without_context": "Physics-IQ noctx",
    "pmf_with_context": "PMF ctx",
    "pmf_without_context": "PMF noctx",
    "wmreward_surprise": "WMReward surprise",
    "vbench_subject_consistency": "VBench subject",
    "vbench_background_consistency": "VBench background",
    "vbench_temporal_flickering": "VBench flicker",
    "vbench_motion_smoothness": "VBench smoothness",
    "vbench_dynamic_degree": "VBench dynamic",
    "vbench_aesthetic_quality": "VBench aesthetic",
    "vbench_imaging_quality": "VBench imaging",
    "videophy2_sa": "VideoPhy2 SA",
    "videophy2_pc": "VideoPhy2 PC",
    "videophy2_joint_rate": "VideoPhy2 joint",
    "videophy2_pc_raw": "VideoPhy2 PC raw",
    "cosmos_reason1": "Cosmos-Reason1",
}
ROLE_COLORS = {"S": "#3c8dbc", "T": "#d97943", "C": "#4ca66b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def hydrate_baseline_metrics(records: list[dict[str, Any]]) -> int:
    """Fill missing seed-851 baseline values from its completed benchmark."""
    if not BASELINE_FALLBACK.is_file():
        return 0
    fallback = pd.read_csv(BASELINE_FALLBACK)
    fallback["case_id"] = fallback["entry_id"].str.split(
        "__baseline__", n=1
    ).str[-1]
    fallback = fallback.set_index(["model", "seed", "case_id"])
    filled = 0
    for record in records:
        if record["kind"] != "baseline":
            continue
        key = (
            str(record["model"]),
            int(record["seed"]),
            str(record["case_id"]),
        )
        if key not in fallback.index:
            continue
        source = fallback.loc[key]
        for metric in METRICS:
            if finite(record["metrics"].get(metric.name)) is not None:
                continue
            value = finite(source.get(metric.name))
            if value is not None:
                record["metrics"][metric.name] = value
                filled += 1
    return filled


def build_frames(
    records: list[dict[str, Any]],
    bootstrap_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in records:
        row = {
            key: record[key]
            for key in (
                "kind",
                "model",
                "seed",
                "case_id",
                "subset_id",
                "role",
                "k",
                "replicate",
                "matching",
                "start",
                "end",
                "video",
            )
        }
        row.update(record["metrics"])
        rows.append(row)
    per_video = pd.DataFrame(rows)
    baseline = {
        (row["model"], int(row["seed"]), row["case_id"]): row
        for row in records
        if row["kind"] == "baseline"
    }
    groups: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in records:
        if row["kind"] != "ablation":
            continue
        key = (
            row["model"],
            int(row["seed"]),
            row["case_id"],
            row["matching"],
            int(row["k"]),
            int(row["replicate"]),
            int(row["start"]),
            int(row["end"]),
        )
        groups.setdefault(key, {})[row["role"]] = row
    matched_rows = []
    for key, role_rows in groups.items():
        if set(role_rows) != {"S", "T", "C"}:
            continue
        model, seed, case_id, matching, k, replicate, start, end = key
        reference = baseline.get((model, seed, case_id))
        for metric in METRICS:
            reference_value = (
                finite(reference["metrics"].get(metric.name))
                if reference is not None
                else None
            )
            values = {
                role: finite(role_rows[role]["metrics"].get(metric.name))
                for role in ("S", "T", "C")
            }
            if any(value is None for value in values.values()):
                continue
            sign = 1.0 if metric.direction == "higher" else -1.0
            for role, value in values.items():
                delta = (
                    float(value) - reference_value
                    if reference_value is not None
                    else None
                )
                matched_rows.append(
                    {
                        "model": model,
                        "seed": seed,
                        "case_id": case_id,
                        "matching": matching,
                        "k": k,
                        "replicate": replicate,
                        "start": start,
                        "end": end,
                        "role": role,
                        "metric": metric.name,
                        "value": value,
                        "baseline": reference_value,
                        "delta": delta,
                        "harm": -sign * delta if delta is not None else None,
                    }
                )
    matched = pd.DataFrame(matched_rows)
    if matched.empty:
        return per_video, matched, pd.DataFrame()
    collapsed_keys = [
        "model",
        "seed",
        "case_id",
        "matching",
        "k",
        "start",
        "end",
        "role",
        "metric",
    ]
    collapsed = matched.groupby(collapsed_keys, as_index=False).agg(
        value=("value", "mean"),
        baseline=("baseline", "mean"),
        harm=("harm", "mean"),
        n_replicates=("replicate", "nunique"),
    )
    rng = np.random.default_rng(20260729)
    aggregate_rows = []
    group_keys = ["model", "matching", "k", "start", "end", "role", "metric"]
    for key, group in collapsed.groupby(group_keys, sort=True):
        case_scores = group.groupby("case_id")["value"].mean().to_numpy(float)
        score_mean = float(case_scores.mean())
        if len(case_scores) > 1:
            draws = rng.choice(
                case_scores,
                size=(bootstrap_samples, len(case_scores)),
                replace=True,
            ).mean(axis=1)
            score_low, score_high = np.quantile(draws, [0.025, 0.975])
        else:
            score_low = score_high = score_mean
        paired_baseline = group.dropna(subset=["baseline", "harm"])
        if paired_baseline.empty:
            paired_score_mean = baseline_mean = harm_mean = harm_low = harm_high = None
            baseline_cases = 0
        else:
            paired_score_mean = float(
                paired_baseline.groupby("case_id")["value"].mean().mean()
            )
            baseline_mean = float(
                paired_baseline.groupby("case_id")["baseline"].mean().mean()
            )
            case_harms = (
                paired_baseline.groupby("case_id")["harm"].mean().to_numpy(float)
            )
            harm_mean = float(case_harms.mean())
            if len(case_harms) > 1:
                harm_draws = rng.choice(
                    case_harms,
                    size=(bootstrap_samples, len(case_harms)),
                    replace=True,
                ).mean(axis=1)
                harm_low, harm_high = np.quantile(harm_draws, [0.025, 0.975])
            else:
                harm_low = harm_high = harm_mean
            baseline_cases = int(paired_baseline["case_id"].nunique())
        original = matched
        for column, value in zip(group_keys, key):
            original = original[original[column] == value]
        aggregate_rows.append(
            {
                **dict(zip(group_keys, key)),
                "score_mean": score_mean,
                "score_ci95_low": float(score_low),
                "score_ci95_high": float(score_high),
                "paired_score_mean": paired_score_mean,
                "baseline_mean": baseline_mean,
                "harm_mean": harm_mean,
                "harm_ci95_low": (
                    float(harm_low) if harm_low is not None else None
                ),
                "harm_ci95_high": (
                    float(harm_high) if harm_high is not None else None
                ),
                "n_cases": int(group["case_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "n_case_seed": len(group),
                "n_baseline_cases": baseline_cases,
                "n_matched_triplets": int(
                    original[
                        ["seed", "case_id", "replicate"]
                    ].drop_duplicates().shape[0]
                ),
                "replicate_min": int(group["n_replicates"].min()),
                "replicate_max": int(group["n_replicates"].max()),
            }
        )
    return per_video, matched, pd.DataFrame(aggregate_rows)


def plot_metric(
    aggregate: pd.DataFrame,
    metric_name: str,
    model: str,
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11, 3.6),
        squeeze=False,
        sharey=True,
    )
    metric_frame = (
        aggregate[
            (aggregate.metric == metric_name)
            & (aggregate.model == model)
        ]
        if not aggregate.empty
        else aggregate
    )
    for column_index, matching in enumerate(MATCH_LABELS):
        axis = axes[0, column_index]
        selected = (
            metric_frame[metric_frame.matching == matching]
            if not metric_frame.empty
            else pd.DataFrame()
        )
        stages = sorted(
            {
                (int(row.start), int(row.end))
                for row in selected.itertuples()
            }
        )
        if not stages:
            axis.text(
                0.5,
                0.5,
                "Pending: no complete S/T/C matched triplet",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#777",
            )
            axis.set_xticks([])
        else:
            x = np.arange(len(stages))
            for role in ("S", "T", "C"):
                role_frame = selected[selected.role == role].set_index(
                    ["start", "end"]
                )
                available = [stage for stage in stages if stage in role_frame.index]
                if not available:
                    continue
                means = np.asarray(
                    [
                        float(role_frame.loc[stage, "score_mean"])
                        for stage in available
                    ]
                )
                lows = np.asarray(
                    [
                        float(role_frame.loc[stage, "score_ci95_low"])
                        for stage in available
                    ]
                )
                highs = np.asarray(
                    [
                        float(role_frame.loc[stage, "score_ci95_high"])
                        for stage in available
                    ]
                )
                positions = np.asarray([stages.index(stage) for stage in available])
                axis.errorbar(
                    positions,
                    means,
                    yerr=np.vstack((means - lows, highs - means)),
                    marker="o",
                    capsize=3,
                    color=ROLE_COLORS[role],
                    label=role,
                )
            baseline_values = selected["baseline_mean"].dropna()
            if not baseline_values.empty:
                axis.axhline(
                    float(baseline_values.mean()),
                    color="#252a28",
                    linewidth=1.2,
                    linestyle="--",
                    label="Baseline",
                )
            axis.set_xticks(x, [f"{start}-{end}" for start, end in stages])
            axis.grid(axis="y", alpha=0.2)
        axis.set_title(PLOT_MATCH_LABELS[matching])
        if column_index == 0:
            axis.set_ylabel("raw metric score")
        axis.set_xlabel("denoise steps")
    handles: list[Any] = []
    labels: list[str] = []
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),
            ncol=len(labels),
            frameon=False,
        )
    direction = next(
        metric.direction for metric in METRICS if metric.name == metric_name
    )
    fig.suptitle(
        f"{MODEL_LABELS[model]} · {METRIC_LABELS[metric_name]} · "
        f"{direction} is better",
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    gallery = args.gallery_root.expanduser().resolve()
    report = gallery / "metrics"
    cases = source_cases(Path(config["input_list"]).expanduser().resolve())
    records = build_records(config, root, cases)
    baseline_fallback_values = hydrate_baseline_metrics(records)
    per_video, matched, aggregate = build_frames(records, args.bootstrap_samples)
    report.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(report / "partial_per_video_metrics.csv", index=False)
    matched.to_csv(report / "partial_matched_triplets.csv", index=False)
    aggregate.to_csv(report / "partial_aggregate.csv", index=False)
    plot_root = report / "plots"
    for model in MODEL_LABELS:
        for metric in METRICS:
            plot_metric(
                aggregate,
                metric.name,
                model,
                plot_root / model / f"{metric.name}.png",
            )
    coverage = []
    for metric in METRICS:
        available = int(per_video[metric.name].notna().sum())
        matched_count = (
            int(
                matched[matched.metric == metric.name][
                    ["model", "seed", "case_id", "matching", "replicate", "start", "end"]
                ]
                .drop_duplicates()
                .shape[0]
            )
            if not matched.empty
            else 0
        )
        role_counts = {
            role: int(
                per_video[
                    (per_video.kind == "ablation") & (per_video.role == role)
                ][metric.name].notna().sum()
            )
            for role in ("S", "T", "C")
        }
        baseline_count = int(
            per_video[per_video.kind == "baseline"][metric.name].notna().sum()
        )
        coverage.append(
            {
                "metric": metric.name,
                "label": METRIC_LABELS[metric.name],
                "direction": metric.direction,
                "available_videos": available,
                "total_videos": len(per_video),
                "available_by_role": role_counts,
                "available_baselines": baseline_count,
                "matched_stc_triplets": matched_count,
            }
        )
    model_coverage: dict[str, list[dict[str, Any]]] = {}
    for model in MODEL_LABELS:
        model_frame = per_video[per_video.model == model]
        model_matched = (
            matched[matched.model == model]
            if not matched.empty
            else matched
        )
        entries = []
        for metric in METRICS:
            role_counts = {
                role: int(
                    model_frame[
                        (model_frame.kind == "ablation")
                        & (model_frame.role == role)
                    ][metric.name].notna().sum()
                )
                for role in ("S", "T", "C")
            }
            matched_count = (
                int(
                    model_matched[model_matched.metric == metric.name][
                        [
                            "seed",
                            "case_id",
                            "matching",
                            "replicate",
                            "start",
                            "end",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                )
                if not model_matched.empty
                else 0
            )
            entries.append(
                {
                    "metric": metric.name,
                    "label": METRIC_LABELS[metric.name],
                    "direction": metric.direction,
                    "available_videos": int(model_frame[metric.name].notna().sum()),
                    "total_videos": len(model_frame),
                    "available_by_role": role_counts,
                    "available_baselines": int(
                        model_frame[model_frame.kind == "baseline"][
                            metric.name
                        ].notna().sum()
                    ),
                    "matched_stc_triplets": matched_count,
                }
            )
        model_coverage[model] = entries
    payload = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "generation_tasks_complete": len(
            {
                (
                    record["model"],
                    record["seed"],
                    record["subset_id"],
                    record["start"],
                    record["end"],
                )
                for record in records
                if record["kind"] == "ablation"
            }
        ),
        "generation_tasks_expected": 252,
        "baseline_fallback": str(BASELINE_FALLBACK),
        "baseline_fallback_values": baseline_fallback_values,
        "coverage": coverage,
        "model_coverage": model_coverage,
        "aggregate": json.loads(aggregate.to_json(orient="records")),
        "metric_labels": METRIC_LABELS,
        "model_labels": MODEL_LABELS,
        "matching_labels": MATCH_LABELS,
    }
    atomic_write(
        report / "data.json",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
    )
    atomic_write(report / "index.html", REPORT_HTML)
    print(
        f"[partial-metric-report] records={len(per_video)} "
        f"matched_rows={len(matched)} aggregate={len(aggregate)} output={report}"
    )


REPORT_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S/T/C Dose-Control · 全指标</title>
<style>
:root{--bg:#f4f5f2;--ink:#202423;--muted:#66706b;--line:#cbd1cd;--accent:#176f62;--panel:#fff;--warn:#9a6b1c}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 system-ui,sans-serif}
header,main{max-width:1600px;margin:auto;padding:16px 22px}header{border-bottom:1px solid var(--line)}
h1,h2,h3,p{margin:0}h1{font-size:23px}h2{font-size:19px}h3{font-size:16px}.sub,.note{color:var(--muted);margin-top:5px}
.links{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}.links a{color:var(--accent)}
.status{color:var(--accent);font-weight:700;margin-top:4px}.model-tabs{display:flex;gap:0;margin:3px 0 18px;border-bottom:1px solid var(--line)}
.model-tabs button{border:0;border-bottom:3px solid transparent;background:transparent;padding:10px 18px;color:var(--muted);font:inherit;font-weight:700;cursor:pointer}
.model-tabs button.active{border-color:var(--accent);color:var(--ink);background:#fff}.model-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:8px}
.model-counts{color:var(--muted);font-variant-numeric:tabular-nums}.table-wrap{overflow:auto;border:1px solid var(--line);background:#fff}
table{width:100%;border-collapse:collapse;font-size:10px;font-variant-numeric:tabular-nums}th,td{padding:5px 7px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}thead th{background:#e9ece9}.coverage{margin-bottom:22px}.coverage table{font-size:11px}
.metric-list{display:grid;gap:14px}.metric-card{background:var(--panel);border:1px solid var(--line);border-radius:4px;overflow:hidden}
.metric-head{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:10px 12px;border-bottom:1px solid var(--line)}
.metric-meta{display:flex;gap:7px;align-items:center;flex-wrap:wrap;color:var(--muted);font-size:11px}.badge{border:1px solid var(--line);padding:2px 6px;background:#f8f9f7}
.badge.ready{color:var(--accent);border-color:#9dbeb7}.badge.pending{color:var(--warn);border-color:#d8c39b}
.metric-body{display:grid;grid-template-columns:minmax(470px,1.05fr) minmax(560px,1fr);align-items:start}.plot{min-width:0;border-right:1px solid var(--line)}
.plot img{display:block;width:100%;height:auto}.metric-table{min-width:0;max-height:385px;overflow:auto}.pending{color:var(--warn)}.positive{color:#b44a42}.negative{color:#18825e}
.master-section{margin:0 0 24px}.master-section h2{margin-bottom:3px}.master-wrap{margin-top:8px;overflow:auto;max-height:72vh;border:1px solid var(--line);background:#fff}
.master-table{width:max-content;min-width:100%;font-size:10px}.master-table th,.master-table td{min-width:112px;padding:6px 7px;text-align:left;vertical-align:top}
.master-table th:first-child,.master-table td:first-child{position:sticky;left:0;z-index:2;min-width:165px;background:#fff;border-right:1px solid var(--line)}
.master-table thead th{position:sticky;top:0;z-index:3;background:#e9ece9}.master-table thead th:first-child{z-index:4;background:#e9ece9}
.metric-direction{display:block;color:var(--muted);font-size:9px;font-weight:400}.cell-score{font-weight:750}.cell-baseline{color:var(--muted);font-size:9px}
.cell-change{font-weight:700;font-size:10px}.cell-change.better{color:#18825e}.cell-change.worse{color:#b44a42}.cell-change.pending{color:var(--warn);font-weight:400}
@media(max-width:1050px){.metric-body{grid-template-columns:1fr}.plot{border-right:0;border-bottom:1px solid var(--line)}.model-head{align-items:start;flex-direction:column}}
</style></head><body><header><h1>S/T/C 等数量与深度匹配 · 全指标</h1>
<p class="status" id="status">读取中</p><p class="sub">同一页面整合全部17项指标，并按模型独立展示。曲线只使用同模型、同seed、同case、同阶段、同replicate下完整的S/T/C配对；纵轴为原始分数，虚线为可用Baseline，harm为正表示消融后变差。</p>
<nav class="links"><a href="/visualizations/">可视化总入口</a><a href="../cases/">逐Case视频</a><a href="partial_per_video_metrics.csv">逐视频CSV</a><a href="partial_matched_triplets.csv">匹配三元组CSV</a><a href="partial_aggregate.csv">聚合CSV</a></nav></header>
<main><nav class="model-tabs" id="model-tabs" aria-label="模型"></nav>
<section><div class="model-head"><div><h2 id="model-title"></h2><p class="note">覆盖率与以下所有曲线、表格均仅属于当前模型。</p></div><p class="model-counts" id="model-counts"></p></div>
<div class="table-wrap coverage" id="coverage"></div></section>
<section class="master-section"><h2>全指标相对 Baseline 总表</h2><p class="note">每个单元格依次显示与Baseline成对样本的当前均值、同一批样本的Baseline均值和方向归一化变化；“改善”为优于Baseline，“变差”为劣于Baseline。</p><div class="master-wrap" id="master-table"></div></section>
<section class="metric-list" id="metrics"></section></main>
<script>
let D,activeModel;const q=id=>document.getElementById(id);
function num(x){return x===null||x===undefined?"Pending":Number(x).toPrecision(4)}
function coverageFor(model){return D.model_coverage[model]}
function rowsFor(model,metric){const order={approx_depth:0,exact_block:1};return D.aggregate.filter(x=>x.model===model&&x.metric===metric).sort((a,b)=>order[a.matching]-order[b.matching]||a.start-b.start||a.role.localeCompare(b.role))}
function renderTabs(){q("model-tabs").innerHTML=Object.keys(D.model_labels).map(model=>`<button type="button" data-model="${model}" class="${model===activeModel?"active":""}">${D.model_labels[model]}</button>`).join("");q("model-tabs").querySelectorAll("button").forEach(button=>button.onclick=()=>{activeModel=button.dataset.model;location.hash=activeModel;render()})}
function renderCoverage(){const rows=coverageFor(activeModel);q("coverage").innerHTML=`<table><thead><tr><th>指标</th><th>方向</th><th>S已评分</th><th>T已评分</th><th>C已评分</th><th>Baseline</th><th>已有/总视频</th><th>完整S/T/C三元组</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${x.label}</td><td>${x.direction}</td><td>${x.available_by_role.S}</td><td>${x.available_by_role.T}</td><td>${x.available_by_role.C}</td><td class="${x.available_baselines?"":"pending"}">${x.available_baselines}</td><td>${x.available_videos}/${x.total_videos}</td><td class="${x.matched_stc_triplets?"":"pending"}">${x.matched_stc_triplets}</td></tr>`).join("")}</tbody></table>`}
function masterCell(row){if(!row)return`<span class="cell-change pending">Pending</span>`;const quality=row.harm_mean===null?null:-Number(row.harm_mean),change=quality===null?"Pending":Math.abs(quality)<1e-12?"持平 0":quality>0?`改善 +${num(quality)}`:`变差 ${num(quality)}`,klass=quality===null?"pending":quality>=0?"better":"worse",score=row.paired_score_mean===null?row.score_mean:row.paired_score_mean;return`<div class="cell-score">P ${num(score)}</div><div class="cell-baseline">B ${num(row.baseline_mean)}</div><div class="cell-change ${klass}">${change}</div>`}
function renderMaster(){const metrics=coverageFor(activeModel),rows=D.aggregate.filter(x=>x.model===activeModel),order={approx_depth:0,exact_block:1},roles={S:0,T:1,C:2},keys=[...new Map(rows.map(x=>[`${x.matching}|${x.start}|${x.end}|${x.role}`,x])).values()].sort((a,b)=>order[a.matching]-order[b.matching]||a.start-b.start||roles[a.role]-roles[b.role]),lookup=new Map(rows.map(x=>[`${x.matching}|${x.start}|${x.end}|${x.role}|${x.metric}`,x]));q("master-table").innerHTML=`<table class="master-table"><thead><tr><th>消融配置</th>${metrics.map(x=>`<th>${x.label}<span class="metric-direction">${x.direction} is better</span></th>`).join("")}</tr></thead><tbody>${keys.length?keys.map(key=>`<tr><td>${D.matching_labels[key.matching]} · ${key.start}-${key.end} · ${key.role}</td>${metrics.map(metric=>`<td>${masterCell(lookup.get(`${key.matching}|${key.start}|${key.end}|${key.role}|${metric.metric}`))}</td>`).join("")}</tr>`).join(""):`<tr><td>当前没有完整S/T/C配对结果</td><td colspan="${metrics.length}" class="pending">Pending</td></tr>`}</tbody></table>`}
function summaryTable(rows){return `<table><thead><tr><th>匹配</th><th>阶段</th><th>Role</th><th>分数均值</th><th>分数95% CI</th><th>Baseline</th><th>harm</th><th>harm 95% CI</th><th>Cases</th><th>Seeds</th><th>Triplets</th><th>Rep.</th></tr></thead><tbody>${rows.length?rows.map(x=>`<tr><td>${D.matching_labels[x.matching]}</td><td>${x.start}-${x.end}</td><td>${x.role}</td><td>${num(x.score_mean)}</td><td>[${num(x.score_ci95_low)}, ${num(x.score_ci95_high)}]</td><td>${num(x.baseline_mean)}</td><td class="${x.harm_mean===null?"pending":x.harm_mean>0?"positive":"negative"}">${num(x.harm_mean)}</td><td>${x.harm_mean===null?"Pending":`[${num(x.harm_ci95_low)}, ${num(x.harm_ci95_high)}]`}</td><td>${x.n_cases}</td><td>${x.n_seeds}</td><td>${x.n_matched_triplets}</td><td>${x.replicate_min}-${x.replicate_max}</td></tr>`).join(""):`<tr><td colspan="12" class="pending">当前没有完整S/T/C配对结果</td></tr>`}</tbody></table>`}
function renderMetrics(){const stamp=Date.now();q("metrics").innerHTML=coverageFor(activeModel).map(metric=>{const rows=rowsFor(activeModel,metric.metric),ready=metric.matched_stc_triplets>0;return `<article class="metric-card"><div class="metric-head"><h3>${metric.label}</h3><div class="metric-meta"><span class="badge">${metric.direction} is better</span><span class="badge ${ready?"ready":"pending"}">${ready?`${metric.matched_stc_triplets} triplets`:"Pending"}</span><span>S/T/C ${metric.available_by_role.S}/${metric.available_by_role.T}/${metric.available_by_role.C}</span></div></div><div class="metric-body"><div class="plot"><img loading="lazy" src="plots/${activeModel}/${metric.metric}.png?t=${stamp}" alt="${D.model_labels[activeModel]} ${metric.label} curve"></div><div class="metric-table">${summaryTable(rows)}</div></div></article>`}).join("")}
function render(){renderTabs();const coverage=coverageFor(activeModel),ready=coverage.filter(x=>x.matched_stc_triplets>0).length,triplets=coverage.reduce((sum,x)=>sum+x.matched_stc_triplets,0);q("model-title").textContent=D.model_labels[activeModel];q("model-counts").textContent=`已有严格结果 ${ready}/17 项 · 配对三元组计数 ${triplets}`;renderCoverage();renderMaster();renderMetrics()}
fetch("data.json").then(x=>x.json()).then(data=>{D=data;const requested=location.hash.slice(1);activeModel=Object.hasOwn(D.model_labels,requested)?requested:Object.keys(D.model_labels)[0];q("status").textContent=`生成 ${D.generation_tasks_complete}/${D.generation_tasks_expected}组 · Baseline回填 ${D.baseline_fallback_values}值 · 更新 ${D.updated_utc}`;render()}).catch(error=>q("status").textContent=`加载失败: ${error}`);
</script></body></html>"""


if __name__ == "__main__":
    main()
