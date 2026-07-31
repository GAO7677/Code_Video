#!/usr/bin/env python3
"""Build one evidence-oriented page for all completed S-head ablations."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery"
)
DEFAULT_OUTPUT = GALLERY_ROOT / "s-head-integrated-analysis"
MOTION_DIR = GALLERY_ROOT / "multiseed/motion-n-analysis/partial"
BENCH_DIR = GALLERY_ROOT / "multiseed/benchmark-metrics"
DOSE_DIR = GALLERY_ROOT / "head-role-dose-control-pilot/metrics"
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
    "openvid_lora_step10000": "Wan+OpenVid LoRA",
}
MODEL_COLORS = {
    "wan_lora": "#1f77b4",
    "xssc": "#d97706",
    "physrvg": "#188263",
    "openvid_lora_step10000": "#a33f72",
}
SUBTYPE_LABELS = {
    "local_enrichment": "Local-enrichment",
    "same_frame_mass": "Same-frame-mass",
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
BENCHMARK_METRICS = (
    ("physics_iq_with_context", "Physics-IQ ctx"),
    ("pmf_with_context", "PMF ctx"),
    ("vbench_motion_smoothness", "Motion smoothness"),
    ("vbench_dynamic_degree", "Dynamic degree"),
    ("videophy2_pc", "VideoPhy2 PC"),
    ("cosmos_reason1", "Cosmos Reason"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}"
    return pattern.format(number)


def stage_label(start: Any, end: Any) -> str:
    return f"[{int(float(start)):02d},{int(float(end)):02d})"


def evidence_class(low: float, high: float, positive_is_good: bool = True) -> str:
    if low > 0:
        return "good" if positive_is_good else "bad"
    if high < 0:
        return "bad" if positive_is_good else "good"
    return "uncertain"


def save_stage_plot(benchmark: pd.DataFrame, output: Path) -> None:
    data = benchmark[benchmark["role"] == "S"].copy()
    stages = sorted(
        {
            (int(row.denoise_start), int(row.denoise_end))
            for row in data.itertuples()
        },
        key=lambda pair: (pair[0], pair[1]),
    )
    labels = [stage_label(*stage) for stage in stages]
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), squeeze=False)
    for axis, (metric, title) in zip(axes.flat, BENCHMARK_METRICS):
        column = f"{metric}_improvement_mean"
        for model in MODEL_LABELS:
            subset = data[data["model"] == model].set_index(
                ["denoise_start", "denoise_end"]
            )
            values = [
                (
                    float(subset.loc[(float(start), float(end)), column])
                    if (float(start), float(end)) in subset.index
                    else np.nan
                )
                for start, end in stages
            ]
            axis.plot(
                range(len(stages)),
                values,
                marker="o",
                linewidth=2,
                markersize=5,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
        axis.axhline(0.0, color="#677078", linewidth=1)
        axis.set_title(f"{title}: paired delta vs baseline")
        axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=3, loc="best")
    figure.suptitle(
        "All S-head ablation by denoising stage",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def save_dose_plot(dose: pd.DataFrame, output: Path) -> None:
    data = dose[
        (dose["role"] == "S")
        & dose["metric"].isin(("physics_iq_with_context", "pmf_with_context"))
    ].copy()
    models = [model for model in MODEL_LABELS if model in set(data["model"])]
    figure, axes = plt.subplots(
        2,
        len(models),
        figsize=(5 * len(models), 8),
        squeeze=False,
    )
    designs = [
        ("exact_block", 5, "[exact] k=5", "o"),
        ("approx_depth", 8, "[depth] k=8", "s"),
    ]
    for column_index, model in enumerate(models):
        subset = data[data["model"] == model]
        for row_index, metric in enumerate(
            ("physics_iq_with_context", "pmf_with_context")
        ):
            axis = axes[row_index, column_index]
            metric_rows = subset[subset["metric"] == metric]
            for matching, k, label, marker in designs:
                points = metric_rows[
                    (metric_rows["matching"] == matching)
                    & (metric_rows["k"] == k)
                ].sort_values(["start", "end"])
                if points.empty:
                    continue
                x = np.arange(len(points))
                y = points["harm_mean"].to_numpy(float)
                low = y - points["harm_ci95_low"].to_numpy(float)
                high = points["harm_ci95_high"].to_numpy(float) - y
                axis.errorbar(
                    x,
                    y,
                    yerr=[low, high],
                    marker=marker,
                    capsize=3,
                    linewidth=1.5,
                    label=label,
                )
                axis.set_xticks(
                    x,
                    [
                        stage_label(start, end)
                        for start, end in zip(points["start"], points["end"])
                    ],
                )
            axis.axhline(0.0, color="#677078", linewidth=1)
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(
                f"{MODEL_LABELS[model]} | "
                f"{'Physics-IQ ctx' if row_index == 0 else 'PMF ctx'}"
            )
            axis.set_ylabel("harm = baseline - ablation")
            if row_index == 0 and column_index == 0:
                axis.legend(frameon=False)
    figure.suptitle(
        "S-head count control: positive harm means score degradation",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def feature_table(motion: pd.DataFrame) -> str:
    rows = []
    data = motion[motion["family"] == "s_feature"].sort_values(
        ["model", "denoise_start", "denoise_end", "subtype"]
    )
    for row in data.itertuples():
        gain_class = evidence_class(
            float(row.gt_gain_ci_low), float(row.gt_gain_ci_high)
        )
        rows.append(
            f"<tr data-model='{html.escape(row.model)}'>"
            f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
            f"<td>{html.escape(SUBTYPE_LABELS.get(row.subtype, row.subtype))}</td>"
            f"<td>{stage_label(row.denoise_start, row.denoise_end)}</td>"
            f"<td>{int(row.head_count)}</td>"
            f"<td>{int(row.n_cases)} / {int(row.n_seeds)}</td>"
            f"<td>{fmt(row.impact_mean)} "
            f"<span class='ci'>[{fmt(row.impact_ci_low)}, {fmt(row.impact_ci_high)}]</span></td>"
            f"<td>{fmt(row.impact_per_head_approx, 5)}</td>"
            f"<td class='{gain_class}'>{fmt(row.gt_gain_mean, signed=True)} "
            f"<span class='ci'>[{fmt(row.gt_gain_ci_low, signed=True)}, "
            f"{fmt(row.gt_gain_ci_high, signed=True)}]</span></td></tr>"
        )
    return "".join(rows)


def depth_table(motion: pd.DataFrame) -> str:
    rows = []
    data = motion[motion["family"] == "s_depth"].sort_values(
        ["model", "denoise_start", "denoise_end", "depth_stratum"]
    )
    for row in data.itertuples():
        gain_class = evidence_class(
            float(row.gt_gain_ci_low), float(row.gt_gain_ci_high)
        )
        rows.append(
            f"<tr data-model='{html.escape(row.model)}'>"
            f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
            f"<td>{html.escape(DEPTH_LABELS.get(row.depth_stratum, row.depth_stratum))}</td>"
            f"<td>{stage_label(row.denoise_start, row.denoise_end)}</td>"
            f"<td>{int(row.head_count)}</td>"
            f"<td>{int(row.n_cases)} / {int(row.n_seeds)}</td>"
            f"<td>{fmt(row.impact_mean)} "
            f"<span class='ci'>[{fmt(row.impact_ci_low)}, {fmt(row.impact_ci_high)}]</span></td>"
            f"<td>{fmt(row.impact_per_head_approx, 5)}</td>"
            f"<td class='{gain_class}'>{fmt(row.gt_gain_mean, signed=True)} "
            f"<span class='ci'>[{fmt(row.gt_gain_ci_low, signed=True)}, "
            f"{fmt(row.gt_gain_ci_high, signed=True)}]</span></td></tr>"
        )
    return "".join(rows)


def dominant_table(motion: pd.DataFrame) -> str:
    rows = []
    data = motion[motion["family"] == "s_dominant_depth"].sort_values(
        ["model", "denoise_start", "denoise_end", "subset_id"]
    )
    for row in data.itertuples():
        gain_class = evidence_class(
            float(row.gt_gain_ci_low), float(row.gt_gain_ci_high)
        )
        label = DOMINANT_LABELS.get(row.subset_id, row.subset_id)
        rows.append(
            f"<tr data-model='{html.escape(row.model)}'>"
            f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
            f"<td>{html.escape(str(label))}</td>"
            f"<td>{stage_label(row.denoise_start, row.denoise_end)}</td>"
            f"<td>{int(row.head_count)}</td>"
            f"<td>{int(row.n_cases)} / {int(row.n_seeds)}</td>"
            f"<td>{fmt(row.impact_mean)} "
            f"<span class='ci'>[{fmt(row.impact_ci_low)}, {fmt(row.impact_ci_high)}]</span></td>"
            f"<td>{fmt(row.impact_per_head_approx, 5)}</td>"
            f"<td class='{gain_class}'>{fmt(row.gt_gain_mean, signed=True)} "
            f"<span class='ci'>[{fmt(row.gt_gain_ci_low, signed=True)}, "
            f"{fmt(row.gt_gain_ci_high, signed=True)}]</span></td></tr>"
        )
    return "".join(rows)


def benchmark_table(benchmark: pd.DataFrame) -> str:
    rows = []
    data = benchmark[benchmark["role"] == "S"].sort_values(
        ["model", "denoise_start", "denoise_end"]
    )
    for row in data.itertuples():
        cells = []
        for metric, _ in BENCHMARK_METRICS:
            value = getattr(row, f"{metric}_improvement_mean")
            cells.append(
                f"<td class='{'good' if value > 0 else 'bad' if value < 0 else ''}'>"
                f"{fmt(value, signed=True)}</td>"
            )
        rows.append(
            f"<tr data-model='{html.escape(row.model)}'>"
            f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
            f"<td>{stage_label(row.denoise_start, row.denoise_end)}</td>"
            f"<td>{int(row.n_seeds)}</td>{''.join(cells)}</tr>"
        )
    return "".join(rows)


def dose_table(dose: pd.DataFrame) -> str:
    rows = []
    data = dose[
        (dose["role"] == "S")
        & dose["metric"].isin(("physics_iq_with_context", "pmf_with_context"))
    ].sort_values(["model", "metric", "matching", "start", "end"])
    labels = {
        "physics_iq_with_context": "Physics-IQ ctx",
        "pmf_with_context": "PMF ctx",
    }
    for row in data.itertuples():
        result_class = evidence_class(
            float(row.harm_ci95_low),
            float(row.harm_ci95_high),
            positive_is_good=False,
        )
        design = (
            "Exact block"
            if row.matching == "exact_block"
            else "Approx. depth"
        )
        rows.append(
            f"<tr data-model='{html.escape(row.model)}'>"
            f"<td>{html.escape(MODEL_LABELS[row.model])}</td>"
            f"<td>{html.escape(labels[row.metric])}</td>"
            f"<td>{design}</td><td>{int(row.k)}</td>"
            f"<td>{stage_label(row.start, row.end)}</td>"
            f"<td>{int(row.n_cases)} / {int(row.n_seeds)}</td>"
            f"<td class='{result_class}'>{fmt(row.harm_mean, signed=True)} "
            f"<span class='ci'>[{fmt(row.harm_ci95_low, signed=True)}, "
            f"{fmt(row.harm_ci95_high, signed=True)}]</span></td></tr>"
        )
    return "".join(rows)


def conclusion_payload(
    motion: pd.DataFrame,
    interactions: pd.DataFrame,
    dose: pd.DataFrame,
) -> list[dict[str, str]]:
    feature = motion[
        (motion["family"] == "s_feature")
        & (motion["denoise_start"] == 0)
        & (motion["denoise_end"] == 40)
    ]
    subtype_lines = []
    for model in MODEL_LABELS:
        values = feature[feature["model"] == model].set_index("subtype")
        if {"local_enrichment", "same_frame_mass"}.issubset(values.index):
            local = float(values.loc["local_enrichment", "impact_mean"])
            same = float(values.loc["same_frame_mass", "impact_mean"])
            subtype_lines.append(
                f"{MODEL_LABELS[model]} {same:.3f} vs {local:.3f}"
            )
    union_lines = []
    for row in interactions.itertuples():
        union_lines.append(
            f"{MODEL_LABELS[row.model]} "
            f"{stage_label(row.denoise_start, row.denoise_end)} "
            f"{row.union_minus_max_single:+.3f}"
        )
    depth = motion[
        (motion["family"] == "s_depth")
        & (motion["denoise_start"] == 10)
        & (motion["denoise_end"] == 20)
    ]
    depth_lines = []
    for model in MODEL_LABELS:
        values = depth[depth["model"] == model]
        if values.empty:
            continue
        total = values.loc[values["impact_mean"].idxmax()]
        per_head = values.loc[values["impact_per_head_approx"].idxmax()]
        depth_lines.append(
            f"{MODEL_LABELS[model]} 总量={DEPTH_LABELS[total.depth_stratum]}，"
            f"单位head={DEPTH_LABELS[per_head.depth_stratum]}"
        )
    reliable_dose = dose[
        (dose["role"] == "S")
        & (
            (dose["harm_ci95_low"] > 0)
            | (dose["harm_ci95_high"] < 0)
        )
    ]
    findings = [
        {
            "tag": "阶段效应",
            "title": "S 消融改变运动，但“改变更大”不等于“物理更好”",
            "body": (
                "全部 S 的阶段实验中，Physics-IQ 与 PMF 在多个后期阶段给出相反方向；"
                "因此不能用单一物理分数替代 Motion Impact，也不能把 Physics-IQ 上升直接解释为生成更合理。"
            ),
        },
        {
            "tag": "子类别",
            "title": "固定 32 heads 后，直接比较 Local 与 Same-frame 的全程运动影响",
            "body": "；".join(subtype_lines)
            + "。该结论是同 head 数的直接比较，目前仍主要来自 seed 851。",
        },
        {
            "tag": "联合消融",
            "title": "64-head union 相对单类的结果随模型变化，不能概括为协同",
            "body": "Union−max(single)："
            + "；".join(union_lines)
            + "。正值表示 union 改变更大，负值表示仍弱于最强的 32-head 单类；"
            "由于网络非线性且 head 数翻倍，这不是可加的交互因果量。",
        },
        {
            "tag": "深度",
            "title": "10–20 去噪阶段由 Early 决定总影响，Middle 的单位-head 敏感度最高",
            "body": "；".join(depth_lines)
            + "。这说明“哪一层总影响大”和“单个 head 哪一层更敏感”是两个不同问题。",
        },
        {
            "tag": "剂量控制",
            "title": "k=5 / k=8 的物理指标证据总体偏弱",
            "body": (
                f"当前数量控制表中，仅 {len(reliable_dose)}/{len(dose[dose['role'] == 'S'])} "
                "个已汇总单元的 95% CI 不跨 0；应优先报告区间与配对覆盖率，"
                "不宜仅凭均值给 S head 下强因果结论。"
            ),
        },
    ]
    openvid = "openvid_lora_step10000"
    openvid_feature = feature[feature["model"] == openvid].set_index("subtype")
    openvid_late = motion[
        (motion["model"] == openvid)
        & (motion["subset_id"] == "S_local_dominant_depth_late")
        & (motion["denoise_start"] == 0)
        & (motion["denoise_end"] == 10)
    ]
    if {
        "local_enrichment",
        "same_frame_mass",
        "local_same_union",
    }.issubset(openvid_feature.index) and not openvid_late.empty:
        local = openvid_feature.loc["local_enrichment"]
        same = openvid_feature.loc["same_frame_mass"]
        union = openvid_feature.loc["local_same_union"]
        late = openvid_late.iloc[0]
        findings.insert(
            1,
            {
                "tag": "OpenVid",
                "title": "OpenVid 对 S-head 消融更敏感，但改变方向取决于 head 类别",
                "body": (
                    f"全程固定 32-head：Local Impact={local.impact_mean:.3f}、"
                    f"Same-frame Impact={same.impact_mean:.3f}；"
                    f"Same-frame GT gain={same.gt_gain_mean:+.3f}，"
                    f"union GT gain={union.gt_gain_mean:+.3f}。"
                    f"相反，0–10 消融 Local-dominant late 的 "
                    f"Impact={late.impact_mean:.3f}、GT gain={late.gt_gain_mean:+.3f}，"
                    "说明“运动改变大”既可能更接近 GT，也可能破坏原有合理运动。"
                ),
            },
        )
    dominant_depth = motion[
        (motion["family"] == "s_dominant_depth")
        & motion["depth_stratum"].notna()
    ]
    same_middle_early = dominant_depth[
        (dominant_depth["dominance_class"] == "same_frame_dominant")
        & (dominant_depth["depth_stratum"] == "middle")
        & (dominant_depth["denoise_start"] == 0)
        & (dominant_depth["denoise_end"] == 10)
    ]
    local_late_early = dominant_depth[
        (dominant_depth["dominance_class"] == "local_dominant")
        & (dominant_depth["depth_stratum"] == "late")
        & (dominant_depth["denoise_start"] == 0)
        & (dominant_depth["denoise_end"] == 10)
    ]
    if not same_middle_early.empty and not local_late_early.empty:
        findings.insert(
            2,
            {
                "tag": "主导×深度",
                "title": "Middle 的单位-head 敏感度最高，Local-Late 的早期消融最可能损害合理运动",
                "body": (
                    "四模型平均：0–10 的 Same-frame-dominant Middle "
                    f"Impact/head={same_middle_early.impact_per_head_approx.mean():.5f}；"
                    "Local-dominant Late 的 "
                    f"GT gain={local_late_early.gt_gain_mean.mean():+.3f}。"
                    "前者说明少量 Middle heads 能高效改变轨迹，后者说明 Late Local heads "
                    "更可能在早期去噪中维持正确运动；总 Impact 与单位-head 敏感度不能混为一谈。"
                ),
            },
        )
    return findings


def build_html(
    motion: pd.DataFrame,
    interactions: pd.DataFrame,
    benchmark: pd.DataFrame,
    dose: pd.DataFrame,
    status: dict[str, Any],
) -> str:
    conclusions = conclusion_payload(motion, interactions, dose)
    conclusion_html = "".join(
        "<article class='finding'>"
        f"<span>{html.escape(item['tag'])}</span>"
        f"<h3>{html.escape(item['title'])}</h3>"
        f"<p>{html.escape(item['body'])}</p></article>"
        for item in conclusions
    )
    status_items = []
    for key, label in (
        ("s_feature", "S 子类"),
        ("s_feature_union", "S union"),
        ("s_feature_phased", "S 分阶段"),
        ("s_depth", "S 深度"),
        ("s_dominant_depth", "S 主导×深度"),
    ):
        item = status.get(key, {})
        counts = item.get("state_counts", {})
        status_items.append(
            f"<div><b>{label}</b><span>{counts.get('complete', 0)} complete · "
            f"{counts.get('running', 0)} running · {counts.get('failed', 0)} failed</span>"
            f"<small>{item.get('ready_videos', 0)} / {item.get('expected_videos', 0)} videos</small></div>"
        )
    benchmark_headers = "".join(
        f"<th>{html.escape(label)} Δ</th>" for _, label in BENCHMARK_METRICS
    )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S Head 消融统一分析</title>
<style>
:root{{--bg:#f3f5f2;--paper:#fff;--ink:#202523;--muted:#64706a;--line:#c8d0ca;--strong:#156c5c;--blue:#246b9e;--orange:#b45b13;--good:#16724e;--bad:#b23b31;--warn:#8a6817}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,"Noto Sans SC",sans-serif;letter-spacing:0}}
header{{background:#202a27;color:#f4f7f5;border-bottom:4px solid #d6a847}}.shell{{max-width:1500px;margin:auto;padding:20px 28px}}h1,h2,h3,p{{margin:0}}h1{{font-size:28px}}h2{{font-size:20px;margin-bottom:5px}}h3{{font-size:15px;margin:3px 0}}.lead{{max-width:980px;margin-top:6px;color:#ced8d3}}
.toplinks{{display:flex;flex-wrap:wrap;gap:12px;margin-top:13px}}a{{color:var(--blue)}}header a{{color:#aee6d6}}nav{{position:sticky;top:0;z-index:5;background:rgba(243,245,242,.96);border-bottom:1px solid var(--line)}}nav .shell{{display:flex;align-items:center;gap:6px;padding-top:8px;padding-bottom:8px;overflow:auto}}button,.navlink{{border:1px solid #aeb8b1;background:#fff;color:var(--ink);padding:6px 9px;font:inherit;text-decoration:none;white-space:nowrap}}button.active{{background:var(--strong);border-color:var(--strong);color:#fff}}
main{{max-width:1500px;margin:auto}}section{{padding:25px 28px;border-bottom:1px solid var(--line)}}.section-head{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:12px}}.muted,.ci{{color:var(--muted)}}.ci{{font-size:12px}}
.definitions{{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);background:var(--paper)}}.definitions div{{padding:10px 12px;border-right:1px solid var(--line)}}.definitions div:last-child{{border-right:0}}
.findings{{border-top:1px solid var(--line)}}.finding{{display:grid;grid-template-columns:105px minmax(260px,.7fr) minmax(420px,1.3fr);gap:18px;padding:12px 4px;border-bottom:1px solid var(--line);align-items:start}}.finding>span{{color:var(--strong);font-weight:750}}.finding p{{color:#46504b}}
.status-grid{{display:grid;grid-template-columns:repeat(4,1fr);background:var(--paper);border:1px solid var(--line)}}.status-grid div{{padding:9px 11px;border-right:1px solid var(--line)}}.status-grid div:last-child{{border-right:0}}.status-grid span,.status-grid small{{display:block;color:var(--muted)}}
.figure{{display:block;width:100%;max-height:1050px;object-fit:contain;background:#fff;border:1px solid var(--line)}}.table-wrap{{overflow:auto;max-height:590px;border:1px solid var(--line);background:#fff;margin-top:10px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:6px 8px;border-bottom:1px solid #dfe4e0;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#e7ebe8;z-index:1;font-size:12px}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f7faf8}}.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}.uncertain{{color:var(--warn)}}
.links-row{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}.links-row a{{background:#fff;border:1px solid var(--line);padding:6px 9px;text-decoration:none}}footer{{padding:20px 28px 35px;color:var(--muted)}}[hidden]{{display:none!important}}
@media(max-width:900px){{.shell,section{{padding-left:14px;padding-right:14px}}.definitions,.status-grid{{grid-template-columns:1fr}}.definitions div,.status-grid div{{border-right:0;border-bottom:1px solid var(--line)}}.finding{{grid-template-columns:1fr;gap:3px}}}}
</style></head><body>
<header><div class="shell"><h1>S Head 消融统一分析</h1>
<p class="lead">统一整理全部 S head、S 子类别、深度分层和 head 数量控制实验。所有变化均与同 case、同模型、同 seed baseline 配对；Motion Impact 表示改变大小，不代表质量方向。</p>
<div class="toplinks"><a href="/">返回 8946 首页</a><a href="/s-head-ablation/">视频逐例比较</a><a href="/common-stc-all-heads-qk-seed851/">S head Q@K</a><a href="/head-role-depth-distribution/">Head 深度分布</a></div></div></header>
<nav><div class="shell"><b>模型</b><button class="active" data-model-filter="all">全部</button><button data-model-filter="wan_lora">Wan+LoRA</button><button data-model-filter="xssc">Wan+xSSC</button><button data-model-filter="physrvg">PhysRVG</button><button data-model-filter="openvid_lora_step10000">OpenVid LoRA</button><a class="navlink" href="#conclusions">结论</a><a class="navlink" href="#all-s">全部 S</a><a class="navlink" href="#subtypes">子类别</a><a class="navlink" href="#depth">深度</a><a class="navlink" href="#dominant">主导×深度</a><a class="navlink" href="#dose">数量控制</a></div></nav>
<main>
<section><div class="definitions"><div><b>Motion Impact ↑</b><br><span class="muted">RAFT 流场、强运动曲线、物体轨迹与速度相对 baseline 的归一化改变。只衡量改变大小。</span></div><div><b>GT gain ↑</b><br><span class="muted">正值表示比 baseline 更接近 49 帧 GT；必须结合按 case bootstrap 的 95% CI。</span></div><div><b>Benchmark Δ ↑</b><br><span class="muted">消融分数减 baseline 分数。不同评测器可能冲突，不合成为单一“物理正确”结论。</span></div></div></section>
<section id="conclusions"><div class="section-head"><div><h2>当前结论</h2><p class="muted">先看结论，再沿页面向下检查证据与覆盖率。</p></div><span class="muted">更新 {updated}</span></div><div class="findings">{conclusion_html}</div></section>
<section><div class="section-head"><div><h2>实验覆盖</h2><p class="muted">正在生成或失败重试的配置不进入完成数据表；页面结论以 CSV 中已有配对结果为准。</p></div></div><div class="status-grid">{''.join(status_items)}</div></section>
<section id="all-s"><div class="section-head"><div><h2>1. 全部 S head × 去噪阶段</h2><p class="muted">第一批大规模实验；正数表示相对 baseline 的指标分数上升。注意不同阶段的 seed 覆盖数并不完全相同。</p></div><a href="/multiseed/benchmark-metrics/">打开原 503-case 页面</a></div>
<img class="figure" src="all_s_benchmark_stage.png" alt="全部S head分阶段指标变化">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>去噪阶段</th><th>Seeds</th>{benchmark_headers}</tr></thead><tbody>{benchmark_table(benchmark)}</tbody></table></div></section>
<section id="subtypes"><div class="section-head"><div><h2>2. S 子类别数量控制</h2><p class="muted">Local-enrichment 与 Same-frame-mass 各取 32 heads，可直接比较；union 为 64 heads，只用于联合敏感性检查。</p></div><a href="/multiseed/motion-n-analysis/partial/">打开原 Motion 页面</a></div>
<img class="figure" src="../multiseed/motion-n-analysis/partial/s_feature_motion_heatmaps.png" alt="S子类别Motion热力图">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>S 子类别</th><th>阶段</th><th>Heads</th><th>Cases / Seeds</th><th>Impact [95% CI]</th><th>Impact/head</th><th>GT gain [95% CI]</th></tr></thead><tbody>{feature_table(motion)}</tbody></table></div></section>
<section id="depth"><div class="section-head"><div><h2>3. S head 深度分层</h2><p class="muted">Early、Middle、Late 的 head 数分别不同；总 Impact 回答“整层组合影响”，Impact/head 仅作敏感度归一化，不能当作可加的单-head因果效应。</p></div><a href="/head-role-depth-distribution/">查看 head 分布</a></div>
<img class="figure" src="../multiseed/motion-n-analysis/partial/s_depth_motion_heatmaps.png" alt="S深度Motion热力图">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>深度</th><th>阶段</th><th>Heads</th><th>Cases / Seeds</th><th>Impact [95% CI]</th><th>Impact/head</th><th>GT gain [95% CI]</th></tr></thead><tbody>{depth_table(motion)}</tbody></table></div></section>
<section id="dominant"><div class="section-head"><div><h2>4. S 主导特征 × 深度</h2><p class="muted">Local-dominant 与 Same-frame-dominant 是互斥全集划分；all 与深度子集 head 数不同，需同时查看 Impact 和 Impact/head。</p></div><a href="/head-role-dose-control-pilot/cases/">查看逐 case 视频</a></div>
<img class="figure" src="../multiseed/motion-n-analysis/partial/s_dominant_depth_motion_heatmaps.png" alt="S主导特征和深度Motion热力图">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>主导类别 / 深度</th><th>阶段</th><th>Heads</th><th>Cases / Seeds</th><th>Impact [95% CI]</th><th>Impact/head</th><th>GT gain [95% CI]</th></tr></thead><tbody>{dominant_table(motion)}</tbody></table></div></section>
<section id="dose"><div class="section-head"><div><h2>5. Head 数量与匹配策略控制</h2><p class="muted">Exact-block k=5 与 approximate-depth k=8；harm = baseline − ablation，正值表示消融使指标下降。区间跨 0 时标为不确定。</p></div><a href="/head-role-dose-control-pilot/metrics/">打开完整 17 项指标</a></div>
<img class="figure" src="s_dose_control.png" alt="S head剂量控制">
<div class="table-wrap"><table><thead><tr><th>模型</th><th>指标</th><th>匹配</th><th>k</th><th>阶段</th><th>Cases / Seeds</th><th>Harm [95% CI]</th></tr></thead><tbody>{dose_table(dose)}</tbody></table></div></section>
<section><h2>证据入口</h2><div class="links-row"><a href="/s-head-ablation/">逐 case 视频</a><a href="/head-role-dose-control-pilot/cases/">数量与深度消融视频</a><a href="/common-stc-all-heads-qk-seed851/">全部 S head Q@K</a><a href="/multiseed/benchmark-metrics/paired_vs_baseline_summary.csv">Benchmark 配对 CSV</a><a href="/multiseed/motion-n-analysis/partial/aggregate_metrics.csv">Motion 汇总 CSV</a><a href="/head-role-dose-control-pilot/metrics/partial_aggregate.csv">剂量控制 CSV</a></div></section>
</main><footer>页面由 build_s_head_integrated_analysis.py 生成；只汇总现有结果，不重算视频或指标。</footer>
<script>
const buttons=[...document.querySelectorAll("[data-model-filter]")],rows=[...document.querySelectorAll("tbody tr[data-model]")];
buttons.forEach(button=>button.addEventListener("click",()=>{{buttons.forEach(item=>item.classList.remove("active"));button.classList.add("active");const model=button.dataset.modelFilter;rows.forEach(row=>row.hidden=model!=="all"&&row.dataset.model!==model);}}));
</script></body></html>"""


def main() -> None:
    args = parse_args()
    motion = read_csv(MOTION_DIR / "aggregate_metrics.csv")
    interactions = read_csv(MOTION_DIR / "interaction_diagnostics.csv")
    benchmark = read_csv(BENCH_DIR / "paired_vs_baseline_summary.csv")
    dose = read_csv(DOSE_DIR / "partial_aggregate.csv")
    status_path = GALLERY_ROOT / "multiseed/motion-n-analysis/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stage_plot(benchmark, args.output_dir / "all_s_benchmark_stage.png")
    save_dose_plot(dose, args.output_dir / "s_dose_control.png")
    atomic_write(
        args.output_dir / "index.html",
        build_html(motion, interactions, benchmark, dose, status),
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "motion": str(MOTION_DIR / "aggregate_metrics.csv"),
            "benchmark": str(BENCH_DIR / "paired_vs_baseline_summary.csv"),
            "dose": str(DOSE_DIR / "partial_aggregate.csv"),
            "status": str(status_path),
        },
        "rows": {
            "motion": len(motion),
            "benchmark_s": int((benchmark["role"] == "S").sum()),
            "dose_s": int((dose["role"] == "S").sum()),
        },
    }
    atomic_write(
        args.output_dir / "manifest.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"[s-head-analysis] output={args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
