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
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MATCH_LABELS = {
    "approx_depth": "k=8 近似深度匹配",
    "exact_block": "k=5 完全同Block匹配",
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
        if reference is None:
            continue
        for metric in METRICS:
            reference_value = finite(reference["metrics"].get(metric.name))
            values = {
                role: finite(role_rows[role]["metrics"].get(metric.name))
                for role in ("S", "T", "C")
            }
            if reference_value is None or any(value is None for value in values.values()):
                continue
            sign = 1.0 if metric.direction == "higher" else -1.0
            for role, value in values.items():
                delta = float(value) - reference_value
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
                        "harm": -sign * delta,
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
        harm=("harm", "mean"),
        value=("value", "mean"),
        baseline=("baseline", "mean"),
        n_replicates=("replicate", "nunique"),
    )
    rng = np.random.default_rng(20260729)
    aggregate_rows = []
    group_keys = ["model", "matching", "k", "start", "end", "role", "metric"]
    for key, group in collapsed.groupby(group_keys, sort=True):
        case_means = group.groupby("case_id")["harm"].mean().to_numpy(float)
        estimate = float(case_means.mean())
        if len(case_means) > 1:
            draws = rng.choice(
                case_means,
                size=(bootstrap_samples, len(case_means)),
                replace=True,
            ).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
        else:
            low = high = estimate
        original = matched
        for column, value in zip(group_keys, key):
            original = original[original[column] == value]
        aggregate_rows.append(
            {
                **dict(zip(group_keys, key)),
                "harm_mean": estimate,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "raw_value_mean": float(group["value"].mean()),
                "baseline_mean": float(group["baseline"].mean()),
                "n_cases": int(group["case_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "n_case_seed": len(group),
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
    output: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), squeeze=False)
    metric_frame = (
        aggregate[aggregate.metric == metric_name]
        if not aggregate.empty
        else aggregate
    )
    for row_index, model in enumerate(MODEL_LABELS):
        for column_index, matching in enumerate(MATCH_LABELS):
            axis = axes[row_index, column_index]
            selected = (
                metric_frame[
                    (metric_frame.model == model)
                    & (metric_frame.matching == matching)
                ]
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
                            float(role_frame.loc[stage, "harm_mean"])
                            for stage in available
                        ]
                    )
                    lows = np.asarray(
                        [
                            float(role_frame.loc[stage, "ci95_low"])
                            for stage in available
                        ]
                    )
                    highs = np.asarray(
                        [
                            float(role_frame.loc[stage, "ci95_high"])
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
                axis.axhline(0, color="#666", linewidth=0.8)
                axis.set_xticks(x, [f"{start}-{end}" for start, end in stages])
                axis.grid(axis="y", alpha=0.2)
            axis.set_title(f"{MODEL_LABELS[model]} · {MATCH_LABELS[matching]}")
            if column_index == 0:
                axis.set_ylabel("harm vs baseline")
            if row_index == 2:
                axis.set_xlabel("denoise steps")
    axes[0, 1].legend(frameon=False)
    fig.suptitle(
        f"{METRIC_LABELS[metric_name]} · positive harm means worse",
        fontweight="bold",
    )
    fig.tight_layout()
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
    per_video, matched, aggregate = build_frames(records, args.bootstrap_samples)
    report.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(report / "partial_per_video_metrics.csv", index=False)
    matched.to_csv(report / "partial_matched_triplets.csv", index=False)
    aggregate.to_csv(report / "partial_aggregate.csv", index=False)
    plot_root = report / "plots"
    for metric in METRICS:
        plot_metric(aggregate, metric.name, plot_root / f"{metric.name}.png")
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
        coverage.append(
            {
                "metric": metric.name,
                "label": METRIC_LABELS[metric.name],
                "direction": metric.direction,
                "available_videos": available,
                "total_videos": len(per_video),
                "matched_stc_triplets": matched_count,
            }
        )
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
        "coverage": coverage,
        "aggregate": aggregate.where(pd.notna(aggregate), None).to_dict(
            orient="records"
        ),
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
<title>S/T/C Dose-Control · 动态指标</title>
<style>
:root{--bg:#f4f5f2;--ink:#202423;--muted:#66706b;--line:#cbd1cd;--accent:#176f62;--panel:#fff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.45 system-ui,sans-serif}
header,main{max-width:1480px;margin:auto;padding:16px 22px}header{border-bottom:1px solid var(--line)}
h1,h2,p{margin:0}h1{font-size:23px}h2{margin:24px 0 7px;font-size:18px}.sub,.note{color:var(--muted);margin-top:5px}.links{display:flex;gap:12px;flex-wrap:wrap;margin-top:8px}.links a{color:var(--accent)}
.controls{display:flex;gap:10px;align-items:end;flex-wrap:wrap;margin-top:13px}label{display:grid;gap:3px;color:var(--muted);font-size:11px}select{padding:7px 9px;border:1px solid var(--line);background:#fff;min-width:190px}
.plot{margin-top:10px;background:var(--panel);border:1px solid var(--line)}.plot img{display:block;width:100%;height:auto}
.table-wrap{overflow:auto;border:1px solid var(--line);background:#fff}table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead th{position:sticky;top:0;background:#e9ece9}
.pending{color:#a17626}.positive{color:#b44a42}.negative{color:#18825e}.status{color:var(--accent);font-weight:700;margin-top:4px}
</style></head><body><header><h1>S/T/C 等数量与深度匹配 · 动态指标</h1>
<p class="status" id="status">读取中</p><p class="sub">仅使用S、T、C三类均完成的同模型、同seed、同case、同阶段、同replicate配对。harm为正表示消融后指标变差。</p>
<nav class="links"><a href="/visualizations/">可视化总入口</a><a href="../cases/">逐Case视频</a><a href="partial_per_video_metrics.csv">逐视频CSV</a><a href="partial_matched_triplets.csv">匹配三元组CSV</a><a href="partial_aggregate.csv">聚合CSV</a></nav></header>
<main><section><h2>指标曲线</h2><div class="controls"><label>指标<select id="metric"></select></label></div><div class="plot"><img id="plot" alt="metric curve"></div></section>
<section><h2>当前覆盖率</h2><p class="note">available videos是已有原始分数的视频数；matched S/T/C triplets才是进入比较曲线的完整配对数。</p><div class="table-wrap" id="coverage"></div></section>
<section><h2>配对聚合表</h2><div class="controls"><label>模型<select id="model"></select></label><label>匹配方式<select id="matching"></select></label></div><div class="table-wrap" id="summary"></div></section></main>
<script>
let D;const q=id=>document.getElementById(id);function options(id,values,label=x=>x){q(id).innerHTML=values.map(x=>`<option value="${x}">${label(x)}</option>`).join("")}
function num(x){return x===null||x===undefined?"Pending":Number(x).toPrecision(4)}
function renderPlot(){q("plot").src=`plots/${q("metric").value}.png?t=${Date.now()}`}
function renderCoverage(){q("coverage").innerHTML=`<table><thead><tr><th>指标</th><th>方向</th><th>已有视频</th><th>总视频</th><th>完整S/T/C三元组</th></tr></thead><tbody>${D.coverage.map(x=>`<tr><td>${x.label}</td><td>${x.direction}</td><td>${x.available_videos}</td><td>${x.total_videos}</td><td class="${x.matched_stc_triplets?"":"pending"}">${x.matched_stc_triplets}</td></tr>`).join("")}</tbody></table>`}
function renderSummary(){const model=q("model").value,matching=q("matching").value,metric=q("metric").value,rows=D.aggregate.filter(x=>x.model===model&&x.matching===matching&&x.metric===metric).sort((a,b)=>a.start-b.start||a.role.localeCompare(b.role));q("summary").innerHTML=`<table><thead><tr><th>阶段</th><th>Role</th><th>harm均值</th><th>95% CI</th><th>原始均值</th><th>Baseline均值</th><th>Cases</th><th>Seeds</th><th>Triplets</th><th>Replicate范围</th></tr></thead><tbody>${rows.length?rows.map(x=>`<tr><td>${x.start}-${x.end}</td><td>${x.role}</td><td class="${x.harm_mean>0?"positive":"negative"}">${num(x.harm_mean)}</td><td>[${num(x.ci95_low)}, ${num(x.ci95_high)}]</td><td>${num(x.raw_value_mean)}</td><td>${num(x.baseline_mean)}</td><td>${x.n_cases}</td><td>${x.n_seeds}</td><td>${x.n_matched_triplets}</td><td>${x.replicate_min}-${x.replicate_max}</td></tr>`).join(""):`<tr><td colspan="10" class="pending">当前没有完整配对结果</td></tr>`}</tbody></table>`}
fetch("data.json").then(x=>x.json()).then(data=>{D=data;q("status").textContent=`生成 ${D.generation_tasks_complete}/${D.generation_tasks_expected}组 · 更新 ${D.updated_utc}`;options("metric",D.coverage.map(x=>x.metric),x=>D.metric_labels[x]);options("model",Object.keys(D.model_labels),x=>D.model_labels[x]);options("matching",Object.keys(D.matching_labels),x=>D.matching_labels[x]);for(const id of["metric","model","matching"])q(id).onchange=()=>{renderPlot();renderSummary()};renderPlot();renderCoverage();renderSummary()}).catch(error=>q("status").textContent=`加载失败: ${error}`);
</script></body></html>"""


if __name__ == "__main__":
    main()
