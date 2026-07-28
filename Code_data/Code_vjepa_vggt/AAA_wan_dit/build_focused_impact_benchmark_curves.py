#!/usr/bin/env python3
"""Plot benchmark changes for the highest-Impact stage/role configurations."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_stc_bench_metrics import METRICS


DEFAULT_ABLATION_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_st_phased_seed851_bench"
)
DEFAULT_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_seed851_baseline_bench"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/seed851/benchmark-metrics/focused-impact-curves"
)
MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
FOCUS = {
    "wan_lora": (
        ("S", "role-S_steps00_10", "S-only [0,10)", 0.693),
        ("T", "role-T_steps00_10", "T-only [0,10)", 0.641),
        ("ST", "role-ST_steps00_05", "S+T [0,5)", 0.720),
    ),
    "xssc": (
        ("S", "role-S_steps00_05", "S-only [0,5)", 0.689),
        ("T", "role-T_steps00_05", "T-only [0,5)", 0.641),
        ("ST", "role-ST_steps00_05", "S+T [0,5)", 0.698),
    ),
    "physrvg": (
        ("S", "role-S_steps05_10", "S-only [5,10)", 0.573),
        ("T", "role-T_steps00_10", "T-only [0,10)", 0.527),
        ("ST", "role-ST_steps05_10", "S+T [5,10)", 0.609),
    ),
}
ROLE_COLORS = {"S": "#238b7b", "T": "#d38b18", "ST": "#3979b9"}
METRIC_TITLES = {
    "physics_iq_with_context": "Physics-IQ with context",
    "physics_iq_without_context": "Physics-IQ without context",
    "pmf_with_context": "PMF with context",
    "pmf_without_context": "PMF without context",
    "wmreward_surprise": "WMReward surprise",
    "vbench_subject_consistency": "VBench subject consistency",
    "vbench_background_consistency": "VBench background consistency",
    "vbench_temporal_flickering": "VBench temporal flickering",
    "vbench_motion_smoothness": "VBench motion smoothness",
    "vbench_dynamic_degree": "VBench dynamic degree",
    "vbench_aesthetic_quality": "VBench aesthetic quality",
    "vbench_imaging_quality": "VBench imaging quality",
    "videophy2_sa": "VideoPhy2 semantic adherence",
    "videophy2_pc": "VideoPhy2 physical commonsense",
    "videophy2_joint_rate": "VideoPhy2 joint pass rate",
    "videophy2_pc_raw": "VideoPhy2 physical commonsense raw",
    "cosmos_reason1": "Cosmos-Reason1",
}
FOCUSED_METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
    "wmreward_surprise",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "videophy2_pc",
    "videophy2_joint_rate",
    "cosmos_reason1",
)
TABLE_METRICS = (
    "physics_iq_with_context",
    "pmf_with_context",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "videophy2_pc",
    "cosmos_reason1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    return parser.parse_args()


def case_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            entry_id.split(f"__{variant}__", 1)[1]
            for entry_id, variant in zip(frame["entry_id"], frame["variant"])
        ],
        index=frame.index,
    )


def bootstrap_ci(
    values: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metric_direction(name: str) -> str:
    return next(metric.direction for metric in METRICS if metric.name == name)


def compute_statistics(
    ablation: pd.DataFrame,
    baseline: pd.DataFrame,
    samples: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records: list[dict[str, Any]] = []
    for model in MODELS:
        model_baseline = baseline[
            (baseline["model"] == model) & (baseline["variant"] == "baseline")
        ].set_index("case_id")
        for role, variant, label, impact in FOCUS[model]:
            selected = ablation[
                (ablation["model"] == model) & (ablation["variant"] == variant)
            ].set_index("case_id")
            shared = selected.index.intersection(model_baseline.index)
            for metric_name in FOCUSED_METRICS:
                baseline_values = model_baseline.loc[shared, metric_name]
                selected_values = selected.loc[shared, metric_name]
                valid_baseline = model_baseline[metric_name].dropna()
                expected = int(len(valid_baseline))
                mask = baseline_values.notna() & selected_values.notna()
                raw_delta = (
                    selected_values[mask].to_numpy(float)
                    - baseline_values[mask].to_numpy(float)
                )
                sign = -1.0 if metric_direction(metric_name) == "lower" else 1.0
                improvement = raw_delta * sign
                count = len(improvement)
                mean = float(improvement.mean()) if count else float("nan")
                standard_deviation = (
                    float(improvement.std(ddof=1)) if count > 1 else float("nan")
                )
                paired_effect_dz = (
                    mean / standard_deviation
                    if math.isfinite(standard_deviation)
                    and standard_deviation > 0
                    else float("nan")
                )
                low, high = (
                    bootstrap_ci(improvement, samples, rng)
                    if count
                    else (float("nan"), float("nan"))
                )
                records.append(
                    {
                        "model": model,
                        "model_label": MODEL_LABELS[model],
                        "role": role,
                        "variant": variant,
                        "focus_label": label,
                        "impact": impact,
                        "metric": metric_name,
                        "metric_title": METRIC_TITLES[metric_name],
                        "metric_direction": metric_direction(metric_name),
                        "paired_count": count,
                        "expected_count": expected,
                        "complete": bool(count == expected and expected > 0),
                        "raw_delta_mean": (
                            float(raw_delta.mean()) if count else float("nan")
                        ),
                        "improvement_mean": mean,
                        "paired_effect_dz": paired_effect_dz,
                        "improvement_ci_low": low,
                        "improvement_ci_high": high,
                        "improvement_rate": (
                            float((improvement > 0).mean())
                            if count
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(records)


def save_metric_plot(stats: pd.DataFrame, metric: str, path: Path) -> None:
    subset = stats[stats["metric"] == metric]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    all_complete = subset[subset["complete"]]["improvement_mean"].to_numpy(float)
    bound = max(
        abs(float(all_complete.min())) if len(all_complete) else 0.0,
        abs(float(all_complete.max())) if len(all_complete) else 0.0,
        1e-6,
    )
    margin = bound * 0.25
    for axis, model in zip(axes, MODELS):
        rows = subset[subset["model"] == model].set_index("role")
        axis.axhline(0, color="#222", linewidth=1.2, label="Baseline")
        for x, role in enumerate(("S", "T", "ST"), start=1):
            row = rows.loc[role]
            if bool(row["complete"]):
                mean = float(row["improvement_mean"])
                axis.errorbar(
                    x,
                    mean,
                    yerr=np.array(
                        [
                            [mean - float(row["improvement_ci_low"])],
                            [float(row["improvement_ci_high"]) - mean],
                        ]
                    ),
                    fmt="o",
                    color=ROLE_COLORS[role],
                    markersize=7,
                    capsize=4,
                    linewidth=1.5,
                )
                axis.plot((0, x), (0, mean), color=ROLE_COLORS[role], alpha=0.3)
                axis.annotate(
                    f"{mean:+.3g}",
                    (x, mean),
                    xytext=(0, 8 if mean >= 0 else -13),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            else:
                axis.text(
                    x,
                    -bound - margin * 0.35,
                    f"pending\n{int(row['paired_count'])}/{int(row['expected_count'])}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="#8b4b45",
                )
        labels = ["Baseline", *[item[2] for item in FOCUS[model]]]
        axis.set_xticks(range(4), labels, rotation=28, ha="right")
        axis.set_title(MODEL_LABELS[model])
        axis.set_ylim(-bound - margin, bound + margin)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Improvement vs matched baseline")
    fig.suptitle(
        f"{METRIC_TITLES[metric]}: focused high-Impact configurations\n"
        "Upward is better; mean paired change with 95% bootstrap CI"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def status_cell(row: pd.Series) -> str:
    count = int(row["paired_count"])
    expected = int(row["expected_count"])
    if not bool(row["complete"]):
        return f'<td class="pending">待补 {count}/{expected}</td>'
    mean = float(row["improvement_mean"])
    css = "good" if mean > 0 else "bad" if mean < 0 else ""
    return (
        f'<td class="{css}">{mean:+.4g}'
        f"<small>{count}/{expected}</small></td>"
    )


def summary_rows(stats: pd.DataFrame) -> str:
    rows = []
    for model in MODELS:
        for role, variant, label, impact in FOCUS[model]:
            selected = stats[
                (stats["model"] == model) & (stats["variant"] == variant)
            ].set_index("metric")
            cells = "".join(status_cell(selected.loc[metric]) for metric in TABLE_METRICS)
            rows.append(
                "<tr>"
                f"<th>{MODEL_LABELS[model]}</th><td>{html.escape(label)}</td>"
                f"<td>{impact:.3f}</td>{cells}</tr>"
            )
    return "".join(rows)


def notable_findings(stats: pd.DataFrame) -> list[str]:
    complete = stats[stats["complete"]].copy()
    complete["stable"] = (
        (complete["improvement_ci_low"] > 0)
        | (complete["improvement_ci_high"] < 0)
    )
    findings = []
    for model in MODELS:
        candidates = complete[
            (complete["model"] == model) & complete["stable"]
        ].copy()
        candidates["abs_dz"] = candidates["paired_effect_dz"].abs()
        for row in candidates.nlargest(4, "abs_dz").itertuples():
            direction = "提高" if row.improvement_mean > 0 else "降低"
            findings.append(
                f"{MODEL_LABELS[model]} 的 {row.focus_label}："
                f"{row.metric_title} 相对 baseline {direction} "
                f"{abs(row.improvement_mean):.4g}（配对效应 dz={row.paired_effect_dz:+.2f}）"
            )
    return findings


def build_html(stats: pd.DataFrame) -> str:
    metric_sections = "".join(
        "<section class='metric'>"
        f"<h2>{html.escape(METRIC_TITLES[metric])}</h2>"
        "<p>纵轴已统一为 improvement：向上表示优于同 case baseline。</p>"
        f"<img src='plots/{html.escape(metric)}.png' alt='{html.escape(metric)}'>"
        "</section>"
        for metric in FOCUSED_METRICS
    )
    findings = "".join(
        f"<li>{html.escape(item)}</li>" for item in notable_findings(stats)
    )
    headers = "".join(
        f"<th>{html.escape(METRIC_TITLES[metric])}</th>"
        for metric in TABLE_METRICS
    )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>高 Impact 方案指标变化</title>
<style>
:root{{--bg:#f4f5f2;--panel:#fff;--ink:#202423;--muted:#66706b;--line:#cfd4d0;--accent:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
header,main{{max-width:1800px;margin:auto;padding:16px 22px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}
h1{{font-size:23px}}h2{{font-size:17px}}.muted,.metric p{{color:var(--muted)}}.note{{background:#fff;border-left:3px solid var(--accent);padding:10px 12px;margin:14px 0}}
.table-wrap{{overflow-x:auto;background:#fff;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}
th,td{{padding:7px 9px;border-bottom:1px solid #e5e9e6;text-align:right}}th:first-child,td:nth-child(2){{text-align:left}}
td small{{display:block;color:var(--muted)}}.good{{color:#15724d}}.bad{{color:#a23c35}}.pending{{color:#8b4b45}}
.findings{{columns:2;background:#fff;padding:12px 28px;margin:14px 0}}.findings li{{break-inside:avoid;margin-bottom:5px}}
.metrics{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px 16px;margin-top:20px}}
.metric{{min-width:0;border-bottom:1px solid var(--line);padding-bottom:16px}}img{{display:block;width:100%;margin-top:8px;border:1px solid var(--line);background:#fff}}
@media(max-width:900px){{.metrics{{grid-template-columns:1fr}}.findings{{columns:1}}}}
</style></head><body><header><h1>高 Impact 方案：指标变化曲线</h1>
<p class="muted">Seed 851，test_5 的 20 个 case；按模型选择 S-only、T-only、S+T 各自 Impact 最大的阶段。</p></header>
<main><div class="note"><strong>读图规则：</strong>所有值均与同模型、同 case baseline 配对。纵轴向上统一表示指标改善；WMReward surprise 原本越低越好，已反向。只有覆盖达到 baseline 可评 case 数的点进入正式曲线，未完成项显示 pending。</div>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>关注方案</th><th>Impact</th>{headers}</tr></thead>
<tbody>{summary_rows(stats)}</tbody></table></div>
<ul class="findings">{findings}</ul>
<div class="metrics">{metric_sections}</div>
<p class="muted">下载：<a href="focused_metric_changes.csv">完整配对统计 CSV</a> · <a href="../">返回全部指标曲线</a></p>
</main></body></html>"""


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ablation = pd.read_csv(
        args.ablation_root.expanduser().resolve()
        / "analysis"
        / "per_video_metrics.csv"
    )
    baseline = pd.read_csv(
        args.baseline_root.expanduser().resolve()
        / "analysis"
        / "per_video_metrics.csv"
    )
    ablation["case_id"] = case_ids(ablation)
    baseline["case_id"] = case_ids(baseline)
    stats = compute_statistics(
        ablation,
        baseline,
        args.bootstrap_samples,
    )
    stats.to_csv(output_dir / "focused_metric_changes.csv", index=False)
    for metric in FOCUSED_METRICS:
        save_metric_plot(
            stats,
            metric,
            output_dir / "plots" / f"{metric}.png",
        )
    atomic_text(output_dir / "index.html", build_html(stats))
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
