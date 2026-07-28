#!/usr/bin/env python3
"""Render all absolute STC benchmark curves on one page."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from summarize_stc_bench_metrics import METRICS


DEFAULT_BATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_bench"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/benchmark-metrics"
)
MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLE_COLORS = {
    "S": "#238b7b",
    "T": "#d38b18",
    "ST": "#3979b9",
    "C": "#c75b68",
}
ROLE_ORDER = ("S", "T", "ST", "C")
COUNT_OFFSETS = {
    "S": (0, 7),
    "T": (0, -10),
    "ST": (8, 1),
    "C": (-8, 1),
}
STAGES = (
    (0, 5),
    (0, 10),
    (0, 15),
    (5, 10),
    (5, 15),
    (10, 20),
    (20, 30),
    (30, 40),
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--title", default="S / T / ST 分阶段消融指标")
    parser.add_argument("--companion-url")
    parser.add_argument("--companion-label")
    parser.add_argument("--secondary-companion-url")
    parser.add_argument("--secondary-companion-label")
    parser.add_argument("--baseline-batch-root", type=Path)
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def mean_ci(values: pd.Series) -> tuple[float, float, int]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return np.nan, np.nan, 0
    mean = float(array.mean())
    ci = (
        1.96 * float(array.std(ddof=1)) / math.sqrt(len(array))
        if len(array) > 1
        else 0.0
    )
    return mean, ci, len(array)


def add_case_ids(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["case_id"] = [
        entry_id.split(f"__{variant}__", 1)[1]
        for entry_id, variant in zip(frame["entry_id"], frame["variant"])
    ]
    return frame


def compute_paired_deltas(per_video: pd.DataFrame) -> pd.DataFrame:
    frame = add_case_ids(per_video)
    records = []
    role_rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    for model in MODEL_NAMES:
        baseline = frame[
            (frame["model"] == model) & (frame["variant"] == "baseline")
        ].set_index("case_id")
        methods = (
            frame[
                (frame["model"] == model)
                & (frame["variant"] != "baseline")
            ][["role", "denoise_start", "denoise_end"]]
            .drop_duplicates()
            .sort_values(
                ["denoise_start", "denoise_end", "role"],
                key=lambda column: (
                    column.map(role_rank)
                    if column.name == "role"
                    else column
                ),
            )
        )
        for method in methods.itertuples(index=False):
            selected = frame[
                (frame["model"] == model)
                & (frame["role"] == method.role)
                & (frame["denoise_start"] == method.denoise_start)
                & (frame["denoise_end"] == method.denoise_end)
            ].set_index("case_id")
            shared = selected.index.intersection(baseline.index)
            for metric in METRICS:
                baseline_values = baseline.loc[shared, metric.name]
                selected_values = selected.loc[shared, metric.name]
                mask = baseline_values.notna() & selected_values.notna()
                deltas = (
                    selected_values[mask].to_numpy(float)
                    - baseline_values[mask].to_numpy(float)
                )
                expected = int(baseline[metric.name].notna().sum())
                raw_delta = (
                    float(deltas.mean()) if len(deltas) else float("nan")
                )
                improvement = (
                    -raw_delta if metric.direction == "lower" else raw_delta
                )
                records.append(
                    {
                        "model": model,
                        "role": method.role,
                        "denoise_start": int(method.denoise_start),
                        "denoise_end": int(method.denoise_end),
                        "metric": metric.name,
                        "metric_direction": metric.direction,
                        "raw_delta_mean": raw_delta,
                        "improvement_mean": improvement,
                        "paired_count": int(len(deltas)),
                        "expected_count": expected,
                        "complete": bool(len(deltas) == expected and expected > 0),
                    }
                )
    return pd.DataFrame(records)


def format_delta(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:+.4g}"


def delta_status_cell(row: pd.Series) -> str:
    count = int(row["paired_count"])
    expected = int(row["expected_count"])
    if not bool(row["complete"]):
        return (
            f"<td class='pending'>待补"
            f"<small>{count}/{expected}</small></td>"
        )
    improvement = float(row["improvement_mean"])
    raw_delta = float(row["raw_delta_mean"])
    css_class = "good" if improvement > 0 else "bad" if improvement < 0 else ""
    status = "改善" if improvement > 0 else "下降" if improvement < 0 else "持平"
    return (
        f"<td class='{css_class}'>{html.escape(format_delta(raw_delta))}"
        f"<small>{status} · {count}/{expected}</small></td>"
    )


def build_delta_tables(paired_deltas: pd.DataFrame) -> str:
    metric_headers = "".join(
        f"<th>{html.escape(METRIC_TITLES[metric.name])}</th>"
        for metric in METRICS
    )
    role_labels = {"S": "S-only", "T": "T-only", "ST": "S+T", "C": "C-only"}
    tables = []
    role_rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    for model, model_label in MODEL_NAMES.items():
        model_rows = paired_deltas[paired_deltas["model"] == model]
        methods = (
            model_rows[["role", "denoise_start", "denoise_end"]]
            .drop_duplicates()
            .sort_values(
                ["denoise_start", "denoise_end", "role"],
                key=lambda column: (
                    column.map(role_rank)
                    if column.name == "role"
                    else column
                ),
            )
        )
        rows = []
        for method in methods.itertuples(index=False):
            selected = model_rows[
                (model_rows["role"] == method.role)
                & (model_rows["denoise_start"] == method.denoise_start)
                & (model_rows["denoise_end"] == method.denoise_end)
            ].set_index("metric")
            cells = "".join(
                delta_status_cell(selected.loc[metric.name])
                for metric in METRICS
            )
            method_label = (
                f"{role_labels.get(method.role, method.role)} "
                f"[{int(method.denoise_start)},{int(method.denoise_end)})"
            )
            rows.append(
                f"<tr><th>{html.escape(method_label)}</th>{cells}</tr>"
            )
        tables.append(
            f"<section class='delta-model'><h2>{html.escape(model_label)}</h2>"
            "<div class='delta-table-wrap'><table class='delta-table'>"
            f"<thead><tr><th>消融方法</th>{metric_headers}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div></section>"
        )
    return "".join(tables)


def stage_values(
    frame: pd.DataFrame,
    model: str,
    role: str,
    value_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means, cis, counts = [], [], []
    for start, end in STAGES:
        values = frame[
            (frame["model"] == model)
            & (frame["role"] == role)
            & (frame["denoise_start"] == start)
            & (frame["denoise_end"] == end)
        ][value_column]
        mean, ci, count = mean_ci(values)
        means.append(mean)
        cis.append(ci)
        counts.append(count)
    return np.asarray(means), np.asarray(cis), np.asarray(counts)


def save_metric_plot(
    per_video: pd.DataFrame,
    metric_name: str,
    metric_direction: str,
    path: Path,
) -> None:
    x = np.arange(len(STAGES))
    labels = [f"[{start},{end})" for start, end in STAGES]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True)
    handles = []
    for row_index, model in enumerate(MODEL_NAMES):
        absolute_axis = axes[row_index]
        baseline_values = per_video[
            (per_video["model"] == model)
            & (per_video["variant"] == "baseline")
        ][metric_name]
        baseline_mean, _, baseline_count = mean_ci(baseline_values)
        if math.isfinite(baseline_mean):
            absolute_axis.axhline(
                baseline_mean,
                color="#222222",
                linewidth=1.8,
                label=f"Baseline (n={baseline_count})",
            )
            absolute_axis.annotate(
                f"Baseline n{baseline_count}",
                (x[-1], baseline_mean),
                xytext=(-3, 5),
                textcoords="offset points",
                ha="right",
                fontsize=7,
                color="#222222",
            )
        for role in ROLE_ORDER:
            means, _, counts = stage_values(
                per_video,
                model,
                role,
                metric_name,
            )
            valid = np.isfinite(means)
            if valid.any():
                (handle,) = absolute_axis.plot(
                    x[valid],
                    means[valid],
                    marker="o",
                    linewidth=1.8,
                    color=ROLE_COLORS[role],
                    label=role,
                )
                for point_x, point_y, count in zip(
                    x[valid],
                    means[valid],
                    counts[valid],
                ):
                    absolute_axis.annotate(
                        f"n{int(count)}",
                        (point_x, point_y),
                        xytext=COUNT_OFFSETS[role],
                        textcoords="offset points",
                        ha="center",
                        fontsize=6,
                        color=ROLE_COLORS[role],
                    )
                if row_index == 0:
                    handles.append(handle)

        absolute_axis.set_title(
            f"{MODEL_NAMES[model]} | absolute ({metric_direction} is better)"
        )
        absolute_axis.set_ylabel(METRIC_TITLES[metric_name])
        absolute_axis.grid(alpha=0.2)
        absolute_axis.set_xticks(x, labels, rotation=35, ha="right")
        absolute_axis.tick_params(axis="x", labelbottom=True)
    if handles:
        fig.legend(
            handles,
            ROLE_ORDER,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.972),
            ncol=len(ROLE_ORDER),
            frameon=False,
        )
    fig.suptitle(
        f"{METRIC_TITLES[metric_name]}: absolute score",
        y=0.997,
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_html(
    coverage: dict[str, int],
    num_entries: int,
    paired_deltas: pd.DataFrame,
    report_title: str,
    companion_url: str | None,
    companion_label: str | None,
    secondary_companion_url: str | None,
    secondary_companion_label: str | None,
) -> str:
    sections = []
    for metric in METRICS:
        name = metric.name
        title = METRIC_TITLES[name]
        count = coverage.get(name, 0)
        sections.append(
            f"<section class='metric' id='{html.escape(name)}'>"
            f"<h2>{html.escape(title)}</h2>"
            f"<p class='coverage'>Coverage: {count}/{num_entries}; "
            f"{html.escape(metric.direction)} is better for the absolute score.</p>"
            f"<img src='plots/{html.escape(name)}.png' "
            f"alt='{html.escape(title)} curves'></section>"
        )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    auto_refresh = (
        "setTimeout(() => location.reload(), 60000);"
        if any(coverage.get(metric.name, 0) < num_entries for metric in METRICS)
        else ""
    )
    companion = ""
    if companion_url and companion_label:
        companion = (
            f"<a class='companion' href='{html.escape(companion_url)}'>"
            f"{html.escape(companion_label)}</a>"
        )
    if secondary_companion_url and secondary_companion_label:
        companion += (
            f"<a class='companion secondary' "
            f"href='{html.escape(secondary_companion_url)}'>"
            f"{html.escape(secondary_companion_label)}</a>"
        )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report_title)}</title>
<style>
:root{{--bg:#f4f5f2;--ink:#202423;--muted:#656d69;--line:#cdd2ce;--accent:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
header,main{{max-width:1800px;margin:auto;padding:16px 22px}}header{{border-bottom:1px solid var(--line)}}
h1,h2,p{{margin:0}}h1{{font-size:23px}}h2{{font-size:18px;margin:0 0 3px}}
.muted,.coverage{{color:var(--muted)}}.companion{{display:inline-block;margin-top:8px;color:var(--accent);font-weight:700}}
.companion.secondary{{margin-left:16px}}
.delta-summary{{margin-top:18px}}.delta-summary>p{{color:var(--muted);margin:3px 0 12px}}
.delta-model{{margin:18px 0 26px}}.delta-model h2{{margin-bottom:7px}}
.delta-table-wrap{{overflow:auto;border:1px solid var(--line);background:white}}
.delta-table{{border-collapse:collapse;min-width:2500px;width:100%;font-size:12px;white-space:nowrap}}
.delta-table th,.delta-table td{{padding:6px 8px;border-right:1px solid #e6e9e7;border-bottom:1px solid #e6e9e7;text-align:right}}
.delta-table thead th{{position:sticky;top:0;z-index:2;background:#edf1ee}}
.delta-table th:first-child{{position:sticky;left:0;z-index:3;background:#f8faf8;text-align:left}}
.delta-table thead th:first-child{{z-index:4;background:#e5ebe7}}
.delta-table small{{display:block;color:var(--muted);font-size:10px}}
.delta-table .good{{color:#14734d;background:#f1faf5}}.delta-table .bad{{color:#a13d35;background:#fff6f4}}
.delta-table .pending{{color:#8b4b45;background:#fff9ed}}
.metrics{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px 18px;margin-top:18px}}
.metric{{min-width:0;border-bottom:1px solid var(--line);padding-bottom:18px}}img{{display:block;width:100%;margin-top:10px;border:1px solid var(--line);background:white}}
.note{{margin-top:16px;padding-top:12px;border-top:1px solid var(--line);color:var(--muted)}}
@media(max-width:900px){{.metrics{{grid-template-columns:1fr}}}}
</style></head><body><header>
<h1>{html.escape(report_title)}</h1>
<p class="muted">更新：{updated}。本页连续展示全部 absolute 指标；每张图包含三个模型，若批次含 Baseline 则以横线表示，点旁 nX 是参与均值的有效样本数。</p>
{companion}
</header><main>
<section class="delta-summary"><h2>相对对应 Baseline 的完整配对变化</h2>
<p>单元格主值为消融分数减去同模型、同 case baseline 分数的均值（Δ）。绿色表示按该指标方向判断为改善，红色表示下降；小字为状态和有效配对数。WMReward surprise 越低越好，因此负 Δ 标为改善。</p>
{build_delta_tables(paired_deltas)}</section>
<div class="metrics">{''.join(sections)}</div>
<p class="note">曲线仅展示均值，不绘制方差或置信区间；指标差异需要结合有效样本数、运动轨迹分析和视频人工核验。</p>
</main><script>{auto_refresh}</script></body></html>"""


def main() -> None:
    args = parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    analysis_root = batch_root / "analysis"
    per_video = pd.read_csv(analysis_root / "per_video_metrics.csv")
    if args.baseline_batch_root:
        baseline_analysis = (
            args.baseline_batch_root.expanduser().resolve()
            / "analysis"
            / "per_video_metrics.csv"
        )
        if baseline_analysis.is_file():
            baseline = pd.read_csv(baseline_analysis)
            baseline = baseline[baseline["variant"] == "baseline"]
            per_video = pd.concat((per_video, baseline), ignore_index=True)
    coverage_payload = json.loads(
        (analysis_root / "coverage.json").read_text(encoding="utf-8")
    )
    coverage = coverage_payload["metric_coverage"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paired_deltas = compute_paired_deltas(per_video)
    paired_deltas.to_csv(output_dir / "paired_delta_table.csv", index=False)
    for metric in METRICS:
        save_metric_plot(
            per_video,
            metric.name,
            metric.direction,
            output_dir / "plots" / f"{metric.name}.png",
        )
    for filename in (
        "per_video_metrics.csv",
        "aggregate_metrics.csv",
        "paired_vs_baseline_per_seed.csv",
        "paired_vs_baseline_summary.csv",
        "coverage.json",
    ):
        source = analysis_root / filename
        target = output_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    atomic_text(
        output_dir / "index.html",
        build_html(
            coverage,
            int(coverage_payload["num_entries"]),
            paired_deltas,
            args.title,
            args.companion_url,
            args.companion_label,
            args.secondary_companion_url,
            args.secondary_companion_label,
        ),
    )
    print(f"[stc-bench-report] {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
