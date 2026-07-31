#!/usr/bin/env python3
"""Build one evidence-oriented page for all completed S-head ablations."""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_REPORT = (
    Path(__file__).resolve().parent / "S_HEAD_CROSS_MODEL_ANALYSIS.md"
)
MOTION_DIR = GALLERY_ROOT / "multiseed/motion-n-analysis/partial"
BENCH_DIR = GALLERY_ROOT / "multiseed/benchmark-metrics"
DOSE_DIR = GALLERY_ROOT / "head-role-dose-control-pilot/metrics"
HEAD_ROLE_MANIFEST = (
    GALLERY_ROOT / "head-role-dose-control-pilot/manifest.json"
)
CROSS_MODELS = (
    "wan_lora",
    "xssc",
    "openvid_lora_step10000",
)
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
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
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


def bootstrap_mean_ci(
    values: list[float],
    *,
    group_key: str,
    samples: int = 20_000,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.nan, np.nan, np.nan
    if array.size == 1:
        value = float(array[0])
        return value, value, value
    seed = int.from_bytes(
        hashlib.sha256(group_key.encode("utf-8")).digest()[:8],
        byteorder="little",
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(array.mean()), float(low), float(high)


def paired_head_benchmark(manifest_path: Path) -> pd.DataFrame:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload["records"]
    rows: list[dict[str, Any]] = []
    metrics = ("physics_iq_with_context", "pmf_with_context")
    for model in CROSS_MODELS:
        model_records = [
            record
            for record in records
            if record.get("model") == model and record.get("seed") == 851
        ]
        baselines = {
            record["case_id"]: record
            for record in model_records
            if record.get("kind") == "baseline"
        }
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in model_records:
            if record.get("kind") not in {
                "s_feature_split",
                "s_dominant_depth",
            }:
                continue
            key = (
                record["kind"],
                record["subset_id"],
                int(record["start"]),
                int(record["end"]),
            )
            groups.setdefault(key, []).append(record)
        for (kind, subset_id, start, end), group in groups.items():
            for metric in metrics:
                differences: list[float] = []
                for record in group:
                    baseline = baselines.get(record["case_id"])
                    if baseline is None:
                        continue
                    value = record.get("metrics", {}).get(metric)
                    base_value = baseline.get("metrics", {}).get(metric)
                    if not isinstance(value, (int, float)) or not isinstance(
                        base_value, (int, float)
                    ):
                        continue
                    if not np.isfinite(value) or not np.isfinite(base_value):
                        continue
                    differences.append(float(value - base_value))
                if not differences:
                    continue
                group_key = (
                    f"{model}|{kind}|{subset_id}|{start}|{end}|{metric}"
                )
                mean, low, high = bootstrap_mean_ci(
                    differences,
                    group_key=group_key,
                )
                rows.append(
                    {
                        "model": model,
                        "family": kind,
                        "subset_id": subset_id,
                        "denoise_start": start,
                        "denoise_end": end,
                        "metric": metric,
                        "n_cases": len(differences),
                        "delta_mean": mean,
                        "delta_ci_low": low,
                        "delta_ci_high": high,
                    }
                )
    return pd.DataFrame(rows)


def validate_cross_model_inputs(
    motion: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    expected_feature_counts = {
        "S_local_k32_r00_exactblock": 32,
        "S_same_k32_r00_exactblock": 32,
        "S_local_same_union_k64_r00_exactblock": 64,
    }
    expected_dominant_counts = {
        "S_local_dominant_all": 100,
        "S_local_dominant_depth_early": 34,
        "S_local_dominant_depth_middle": 25,
        "S_local_dominant_depth_late": 41,
        "S_same_frame_dominant_all": 59,
        "S_same_frame_dominant_depth_early": 24,
        "S_same_frame_dominant_depth_middle": 15,
        "S_same_frame_dominant_depth_late": 20,
    }
    expected = expected_feature_counts | expected_dominant_counts
    for model in CROSS_MODELS:
        for subset_id, head_count in expected.items():
            for start, end in ((0, 10), (10, 20), (0, 40)):
                row = select_motion(
                    motion,
                    model=model,
                    subset_id=subset_id,
                    start=start,
                    end=end,
                )
                if (
                    int(row.head_count) != head_count
                    or int(row.n_cases) != 20
                    or int(row.n_seeds) != 1
                ):
                    raise ValueError(
                        f"Coverage mismatch for {model}/{subset_id}/"
                        f"{start}-{end}: heads={row.head_count}, "
                        f"cases={row.n_cases}, seeds={row.n_seeds}"
                    )
    if len(paired) != 198 or set(paired["n_cases"]) != {20}:
        raise ValueError(
            "Expected 198 paired Physics-IQ/PMF summaries with 20 cases each"
        )


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


def select_motion(
    motion: pd.DataFrame,
    *,
    model: str,
    subset_id: str,
    start: int,
    end: int,
) -> pd.Series:
    rows = motion[
        (motion["model"] == model)
        & (motion["subset_id"] == subset_id)
        & (motion["denoise_start"] == start)
        & (motion["denoise_end"] == end)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one motion row for {model}/{subset_id}/{start}-{end}, "
            f"found {len(rows)}"
        )
    return rows.iloc[0]


def select_paired_metric(
    paired: pd.DataFrame,
    *,
    model: str,
    subset_id: str,
    start: int,
    end: int,
    metric: str,
) -> pd.Series:
    rows = paired[
        (paired["model"] == model)
        & (paired["subset_id"] == subset_id)
        & (paired["denoise_start"] == start)
        & (paired["denoise_end"] == end)
        & (paired["metric"] == metric)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one metric row for {model}/{subset_id}/{start}-{end}/"
            f"{metric}, found {len(rows)}"
        )
    return rows.iloc[0]


def conclusion_payload(
    motion: pd.DataFrame,
    interactions: pd.DataFrame,
    paired: pd.DataFrame,
) -> list[dict[str, str]]:
    pmf_lines = []
    early_lines = []
    late_lines = []
    feature_means = []
    for model in CROSS_MODELS:
        local_pmf = select_paired_metric(
            paired,
            model=model,
            subset_id="S_local_dominant_all",
            start=0,
            end=40,
            metric="pmf_with_context",
        )
        pmf_lines.append(
            f"{MODEL_LABELS[model]} {local_pmf.delta_mean:+.3f} "
            f"[{local_pmf.delta_ci_low:+.3f}, {local_pmf.delta_ci_high:+.3f}]"
        )
        local_early = select_motion(
            motion,
            model=model,
            subset_id="S_local_k32_r00_exactblock",
            start=0,
            end=10,
        )
        same_early = select_motion(
            motion,
            model=model,
            subset_id="S_same_k32_r00_exactblock",
            start=0,
            end=10,
        )
        early_lines.append(
            f"{MODEL_LABELS[model]} Same={same_early.impact_mean:.3f}, "
            f"Local={local_early.impact_mean:.3f}"
        )
        late_motion = select_motion(
            motion,
            model=model,
            subset_id="S_local_dominant_depth_late",
            start=0,
            end=10,
        )
        late_pmf = select_paired_metric(
            paired,
            model=model,
            subset_id="S_local_dominant_depth_late",
            start=0,
            end=10,
            metric="pmf_with_context",
        )
        late_lines.append(
            f"{MODEL_LABELS[model]} GT gain={late_motion.gt_gain_mean:+.3f}, "
            f"PMF Δ={late_pmf.delta_mean:+.3f}"
        )
        feature_rows = motion[
            (motion["model"] == model) & (motion["family"] == "s_feature")
        ]
        feature_means.append(
            f"{MODEL_LABELS[model]}={feature_rows.impact_mean.mean():.3f}"
        )
    union_lines = []
    for model in CROSS_MODELS:
        model_rows = interactions[interactions["model"] == model]
        values = []
        for start, end in ((0, 10), (10, 20), (0, 40)):
            rows = model_rows[
                (model_rows["denoise_start"] == start)
                & (model_rows["denoise_end"] == end)
            ]
            if len(rows) != 1:
                raise ValueError(
                    f"Expected one interaction row for {model}/{start}-{end}"
                )
            row = rows.iloc[0]
            values.append(
                f"{stage_label(start, end)} "
                f"{row.union_minus_max_single:+.3f}"
            )
        union_lines.append(f"{MODEL_LABELS[model]}: {', '.join(values)}")
    return [
        {
            "tag": "G3-D",
            "title": "全程消融 Local-dominant all 后，三模型 PMF 均下降",
            "body": "；".join(pmf_lines)
            + "。三组按 case bootstrap 95% CI 均低于 0。该结论只覆盖当前三模型、"
            "seed 851 和 20 个 case，不能外推到其他模型或数据分布。",
        },
        {
            "tag": "G3-R",
            "title": "固定为 32 heads 时，0–10 阶段的 Same-frame 平均 Impact 均高于 Local",
            "body": "；".join(early_lines)
            + "。这是三模型同方向均值，不是 Same−Local 差值的显著性检验。",
        },
        {
            "tag": "G3-R",
            "title": "Local-dominant Late × 0–10 在三模型中均出现负向 GT gain 和 PMF 均值",
            "body": "；".join(late_lines)
            + "。其中 xSSC 的 PMF 和部分 GT gain 区间跨 0，因此只能称为重复方向，"
            "不能称为三模型均已显著。",
        },
        {
            "tag": "模型内",
            "title": "固定子类别实验的平均运动敏感度存在模型差异",
            "body": "；".join(feature_means)
            + "。OpenVid 均值最高、xSSC 最低是当前配置上的描述性排序；训练权重、"
            "条件分支和 head 分类来源均有差异，不能据此归因于某个模块。",
        },
        {
            "tag": "模型内",
            "title": "Local+Same union 的结果不呈统一交互方向",
            "body": "Union−max(single)："
            + "；".join(union_lines)
            + "。该量同时改变 head 数，且网络响应非线性，只能描述联合消融结果，"
            "不能解释为严格的协同或抵消因果效应。",
        },
        {
            "tag": "I",
            "title": "“Same-frame 更敏感、Local 更支撑物理连续性”是机制假设，不是已证实功能标签",
            "body": (
                "这一解释来自 Motion Impact 与 PMF 的组合模式。要验证功能分工，"
                "仍需等 head 数、等输出能量、更多 seeds、held-out cases，以及"
                "单 head 或小 k 干预。"
            ),
        },
    ]


def cross_model_feature_table(motion: pd.DataFrame) -> str:
    rows = []
    subsets = (
        ("S_local_k32_r00_exactblock", "Local-32"),
        ("S_same_k32_r00_exactblock", "Same-frame-32"),
        ("S_local_same_union_k64_r00_exactblock", "Union-64"),
    )
    for model in CROSS_MODELS:
        for start, end in ((0, 10), (10, 20), (0, 40)):
            values = [
                select_motion(
                    motion,
                    model=model,
                    subset_id=subset_id,
                    start=start,
                    end=end,
                )
                for subset_id, _ in subsets
            ]
            impact_cells = "".join(
                f"<td>{fmt(row.impact_mean)}</td>" for row in values
            )
            gain_cells = "".join(
                f"<td>{fmt(row.gt_gain_mean, signed=True)}</td>"
                for row in values
            )
            rows.append(
                f"<tr data-model='{html.escape(model)}'>"
                f"<td>{html.escape(MODEL_LABELS[model])}</td>"
                f"<td>{stage_label(start, end)}</td>"
                f"{impact_cells}{gain_cells}</tr>"
            )
    return "".join(rows)


def cross_model_dominant_table(
    motion: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    rows = []
    for model in CROSS_MODELS:
        local = select_motion(
            motion,
            model=model,
            subset_id="S_local_dominant_all",
            start=0,
            end=40,
        )
        same = select_motion(
            motion,
            model=model,
            subset_id="S_same_frame_dominant_all",
            start=0,
            end=40,
        )
        local_pmf = select_paired_metric(
            paired,
            model=model,
            subset_id="S_local_dominant_all",
            start=0,
            end=40,
            metric="pmf_with_context",
        )
        same_pmf = select_paired_metric(
            paired,
            model=model,
            subset_id="S_same_frame_dominant_all",
            start=0,
            end=40,
            metric="pmf_with_context",
        )
        local_physics = select_paired_metric(
            paired,
            model=model,
            subset_id="S_local_dominant_all",
            start=0,
            end=40,
            metric="physics_iq_with_context",
        )
        same_physics = select_paired_metric(
            paired,
            model=model,
            subset_id="S_same_frame_dominant_all",
            start=0,
            end=40,
            metric="physics_iq_with_context",
        )
        rows.append(
            f"<tr data-model='{html.escape(model)}'>"
            f"<td>{html.escape(MODEL_LABELS[model])}</td>"
            f"<td>{fmt(local.impact_per_head_approx, 5)} / "
            f"{fmt(same.impact_per_head_approx, 5)}</td>"
            f"<td>{fmt(local.gt_gain_mean, signed=True)} / "
            f"{fmt(same.gt_gain_mean, signed=True)}</td>"
            f"<td class='bad'>{fmt(local_pmf.delta_mean, signed=True)} "
            f"<span class='ci'>[{fmt(local_pmf.delta_ci_low, signed=True)}, "
            f"{fmt(local_pmf.delta_ci_high, signed=True)}]</span></td>"
            f"<td>{fmt(same_pmf.delta_mean, signed=True)} "
            f"<span class='ci'>[{fmt(same_pmf.delta_ci_low, signed=True)}, "
            f"{fmt(same_pmf.delta_ci_high, signed=True)}]</span></td>"
            f"<td>{fmt(local_physics.delta_mean, signed=True)} / "
            f"{fmt(same_physics.delta_mean, signed=True)}</td></tr>"
        )
    return "".join(rows)


def cross_model_depth_table(
    motion: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    rows = []
    for model in CROSS_MODELS:
        depth_rows = motion[
            (motion["model"] == model)
            & (motion["family"] == "s_dominant_depth")
            & motion["depth_stratum"].notna()
            & (motion["denoise_start"] == 0)
            & (motion["denoise_end"] == 10)
        ]
        top = depth_rows.loc[depth_rows["impact_per_head_approx"].idxmax()]
        late = select_motion(
            motion,
            model=model,
            subset_id="S_local_dominant_depth_late",
            start=0,
            end=10,
        )
        late_pmf = select_paired_metric(
            paired,
            model=model,
            subset_id="S_local_dominant_depth_late",
            start=0,
            end=10,
            metric="pmf_with_context",
        )
        rows.append(
            f"<tr data-model='{html.escape(model)}'>"
            f"<td>{html.escape(MODEL_LABELS[model])}</td>"
            f"<td>{html.escape(DOMINANT_LABELS[top.subset_id])}</td>"
            f"<td>{fmt(top.impact_per_head_approx, 5)}</td>"
            f"<td>{fmt(late.gt_gain_mean, signed=True)} "
            f"<span class='ci'>[{fmt(late.gt_gain_ci_low, signed=True)}, "
            f"{fmt(late.gt_gain_ci_high, signed=True)}]</span></td>"
            f"<td>{fmt(late_pmf.delta_mean, signed=True)} "
            f"<span class='ci'>[{fmt(late_pmf.delta_ci_low, signed=True)}, "
            f"{fmt(late_pmf.delta_ci_high, signed=True)}]</span></td></tr>"
        )
    return "".join(rows)


def build_cross_model_markdown(
    motion: pd.DataFrame,
    interactions: pd.DataFrame,
    paired: pd.DataFrame,
) -> str:
    findings = conclusion_payload(motion, interactions, paired)
    lines = [
        "# Wan S-Head 跨模型消融分析",
        "",
        "## 1. 分析范围",
        "",
        "本文比较 Wan+LoRA、Wan+xSSC 与 Wan+OpenVid LoRA(step-10000)。"
        "公平对照部分固定为 seed 851、相同 20 个 source cases、相同 baseline "
        "配对、相同 head 集合和相同去噪区间。视频为 49 帧。",
        "",
        "OpenVid 使用此前冻结的公共 S-head 列表，没有针对 OpenVid 重新分类。"
        "因此本文能比较相同干预位置的响应，不能判断这些位置是否也是 OpenVid "
        "自身最稳定的 S heads。",
        "",
        "### 证据标签",
        "",
        "| 标签 | 含义 |",
        "|---|---|",
        "| G3-D | 三个受测模型均有直接配对指标支持；若涉及区间，三个模型的 95% CI 均满足所述方向。只表示当前三模型内复现，不代表外部模型普适性。 |",
        "| G3-R | 三个模型出现同方向均值或排序，但模型间差值本身未完成显著性检验，或至少一个模型的区间跨 0。 |",
        "| 模型内 | 当前模型/配置上的描述性结果，不声称跨模型成立。 |",
        "| I | 从现象推导的机制解释或后续假设，不是指标直接证明的事实。 |",
        "",
        "## 2. 核心结论",
        "",
        "| 证据 | 结论 | 支撑与边界 |",
        "|---|---|---|",
    ]
    for finding in findings:
        lines.append(
            f"| {finding['tag']} | {finding['title']} | {finding['body']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 等量 32-head 公平对照",
            "",
            "`Motion Impact` 只表示相对同 case baseline 的运动变化大小；"
            "`GT gain > 0` 表示轨迹指标更接近 GT，不等同于整体物理质量提高。",
            "",
            "| 模型 | 阶段 | Local Impact | Same Impact | Union Impact | Local GT gain | Same GT gain | Union GT gain |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    subsets = (
        "S_local_k32_r00_exactblock",
        "S_same_k32_r00_exactblock",
        "S_local_same_union_k64_r00_exactblock",
    )
    for model in CROSS_MODELS:
        for start, end in ((0, 10), (10, 20), (0, 40)):
            values = [
                select_motion(
                    motion,
                    model=model,
                    subset_id=subset_id,
                    start=start,
                    end=end,
                )
                for subset_id in subsets
            ]
            lines.append(
                f"| {MODEL_LABELS[model]} | {start}-{end} | "
                f"{values[0].impact_mean:.3f} | {values[1].impact_mean:.3f} | "
                f"{values[2].impact_mean:.3f} | "
                f"{values[0].gt_gain_mean:+.3f} | "
                f"{values[1].gt_gain_mean:+.3f} | "
                f"{values[2].gt_gain_mean:+.3f} |"
            )
    lines.extend(
        [
            "",
            "## 4. 主导类别全程消融",
            "",
            "Local-dominant all 为 100 heads，Same-frame-dominant all 为 59 heads。"
            "`Impact/head` 只是总 Impact 除以 head 数的近似归一化，不是可加的"
            "单-head 因果效应。",
            "",
            "| 模型 | Impact/head Local / Same | GT gain Local / Same | PMF Δ Local [95% CI] | PMF Δ Same [95% CI] | Physics-IQ Δ Local / Same |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in CROSS_MODELS:
        local = select_motion(
            motion,
            model=model,
            subset_id="S_local_dominant_all",
            start=0,
            end=40,
        )
        same = select_motion(
            motion,
            model=model,
            subset_id="S_same_frame_dominant_all",
            start=0,
            end=40,
        )
        metric_rows = {}
        for subset_id, prefix in (
            ("S_local_dominant_all", "local"),
            ("S_same_frame_dominant_all", "same"),
        ):
            for metric in (
                "pmf_with_context",
                "physics_iq_with_context",
            ):
                metric_rows[(prefix, metric)] = select_paired_metric(
                    paired,
                    model=model,
                    subset_id=subset_id,
                    start=0,
                    end=40,
                    metric=metric,
                )
        local_pmf = metric_rows[("local", "pmf_with_context")]
        same_pmf = metric_rows[("same", "pmf_with_context")]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{local.impact_per_head_approx:.5f} / "
            f"{same.impact_per_head_approx:.5f} | "
            f"{local.gt_gain_mean:+.3f} / {same.gt_gain_mean:+.3f} | "
            f"{local_pmf.delta_mean:+.3f} "
            f"[{local_pmf.delta_ci_low:+.3f}, {local_pmf.delta_ci_high:+.3f}] | "
            f"{same_pmf.delta_mean:+.3f} "
            f"[{same_pmf.delta_ci_low:+.3f}, {same_pmf.delta_ci_high:+.3f}] | "
            f"{metric_rows[('local', 'physics_iq_with_context')].delta_mean:+.3f} / "
            f"{metric_rows[('same', 'physics_iq_with_context')].delta_mean:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Physics-IQ 与 PMF 在多组消融中方向不同。这里保留原始变化，"
            "不把二者合成为“物理正确性”总分，也不把 Physics-IQ 上升单独解释为质量提高。",
            "",
            "## 5. 深度相关现象",
            "",
            "| 模型 | 0-10 单位 head Impact 最大子集 | Impact/head | Local-Late 0-10 GT gain [95% CI] | PMF Δ [95% CI] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model in CROSS_MODELS:
        depth_rows = motion[
            (motion["model"] == model)
            & (motion["family"] == "s_dominant_depth")
            & motion["depth_stratum"].notna()
            & (motion["denoise_start"] == 0)
            & (motion["denoise_end"] == 10)
        ]
        top = depth_rows.loc[depth_rows["impact_per_head_approx"].idxmax()]
        late = select_motion(
            motion,
            model=model,
            subset_id="S_local_dominant_depth_late",
            start=0,
            end=10,
        )
        late_pmf = select_paired_metric(
            paired,
            model=model,
            subset_id="S_local_dominant_depth_late",
            start=0,
            end=10,
            metric="pmf_with_context",
        )
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{DOMINANT_LABELS[top.subset_id].replace(' | ', ' / ')} | "
            f"{top.impact_per_head_approx:.5f} | "
            f"{late.gt_gain_mean:+.3f} "
            f"[{late.gt_gain_ci_low:+.3f}, {late.gt_gain_ci_high:+.3f}] | "
            f"{late_pmf.delta_mean:+.3f} "
            f"[{late_pmf.delta_ci_low:+.3f}, {late_pmf.delta_ci_high:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## 6. 已有分析的整合与修正",
            "",
            "- 保留：Motion Impact 与质量方向必须分开；Physics-IQ 与 PMF "
            "存在方向冲突；不同 head 数必须同时报告总 Impact 和近似 Impact/head。",
            "- 收紧：旧页面中“Middle 最重要”“Local-Late 负责正确运动”等表述，"
            "改为模型内最高均值或机制假设。现有指标只能说明干预响应，不能直接确定功能。",
            "- 收紧：OpenVid 的平均 Impact 较高只作为当前配置排序；由于 OpenVid "
            "没有独立重做 head 分类，不能据此断言其更依赖 S-head。",
            "- 不合并：k=5/k=8 剂量实验、全部 S 多 seed 实验和本报告的 seed851 "
            "32/59/100-head 实验回答的问题不同，页面继续保留原表，但不混成一个效应量。",
            "",
            "## 7. 局限与下一步验证",
            "",
            "1. 当前三模型公平比较只有一个 seed 和 20 个 case；case 类型也不是独立总体抽样。",
            "2. OpenVid 缺少本批 VideoPhy2/Cosmos 完整分数，跨模型结论主要依赖 Motion、GT、PMF 与 Physics-IQ。",
            "3. Union 同时增加 head 数，不能作为严格的交互项；应补等数量、等 block、等输出能量设计。",
            "4. Impact/head 是近似剂量归一化。网络存在非线性，不能由 all-head 结果反推单 head 效应。",
            "5. 若要声称机制或外部模型泛化，需要更多 seeds、held-out cases、OpenVid 独立分类和小 k 干预。",
            "",
            "## 8. 数据来源",
            "",
            f"- Motion 汇总：`{MOTION_DIR / 'aggregate_metrics.csv'}`",
            f"- 联合消融诊断：`{MOTION_DIR / 'interaction_diagnostics.csv'}`",
            f"- 配对 benchmark 原始记录：`{HEAD_ROLE_MANIFEST}`",
            f"- 全部 S 多 seed benchmark：`{BENCH_DIR / 'paired_vs_baseline_summary.csv'}`",
            f"- 数量控制结果：`{DOSE_DIR / 'partial_aggregate.csv'}`",
            "",
        ]
    )
    return "\n".join(lines)


def build_html(
    motion: pd.DataFrame,
    interactions: pd.DataFrame,
    benchmark: pd.DataFrame,
    dose: pd.DataFrame,
    paired: pd.DataFrame,
    status: dict[str, Any],
) -> str:
    conclusions = conclusion_payload(motion, interactions, paired)
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
.evidence-key{{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);background:var(--paper);margin-top:12px}}.evidence-key div{{padding:9px 11px;border-right:1px solid var(--line)}}.evidence-key div:last-child{{border-right:0}}.evidence-key b{{display:block;color:var(--strong)}}.findings{{border-top:1px solid var(--line)}}.finding{{display:grid;grid-template-columns:105px minmax(260px,.7fr) minmax(420px,1.3fr);gap:18px;padding:12px 4px;border-bottom:1px solid var(--line);align-items:start}}.finding>span{{color:var(--strong);font-weight:750}}.finding p{{color:#46504b}}
.status-grid{{display:grid;grid-template-columns:repeat(4,1fr);background:var(--paper);border:1px solid var(--line)}}.status-grid div{{padding:9px 11px;border-right:1px solid var(--line)}}.status-grid div:last-child{{border-right:0}}.status-grid span,.status-grid small{{display:block;color:var(--muted)}}
.figure{{display:block;width:100%;max-height:1050px;object-fit:contain;background:#fff;border:1px solid var(--line)}}.table-wrap{{overflow:auto;max-height:590px;border:1px solid var(--line);background:#fff;margin-top:10px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:6px 8px;border-bottom:1px solid #dfe4e0;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#e7ebe8;z-index:1;font-size:12px}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}tr:hover td{{background:#f7faf8}}.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}.uncertain{{color:var(--warn)}}
.links-row{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}.links-row a{{background:#fff;border:1px solid var(--line);padding:6px 9px;text-decoration:none}}footer{{padding:20px 28px 35px;color:var(--muted)}}[hidden]{{display:none!important}}
@media(max-width:900px){{.shell,section{{padding-left:14px;padding-right:14px}}.definitions,.status-grid,.evidence-key{{grid-template-columns:1fr}}.definitions div,.status-grid div,.evidence-key div{{border-right:0;border-bottom:1px solid var(--line)}}.finding{{grid-template-columns:1fr;gap:3px}}}}
</style></head><body>
<header><div class="shell"><h1>S Head 消融统一分析</h1>
<p class="lead">统一整理全部 S head、S 子类别、深度分层和 head 数量控制实验。所有变化均与同 case、同模型、同 seed baseline 配对；Motion Impact 表示改变大小，不代表质量方向。</p>
<div class="toplinks"><a href="/">返回 8946 首页</a><a href="analysis.md">完整 Markdown</a><a href="/s-head-ablation/">视频逐例比较</a><a href="/common-stc-all-heads-qk-seed851/">S head Q@K</a><a href="/head-role-depth-distribution/">Head 深度分布</a></div></div></header>
<nav><div class="shell"><b>模型</b><button class="active" data-model-filter="all">全部</button><button data-model-filter="wan_lora">Wan+LoRA</button><button data-model-filter="xssc">Wan+xSSC</button><button data-model-filter="physrvg">PhysRVG</button><button data-model-filter="openvid_lora_step10000">OpenVid LoRA</button><a class="navlink" href="#conclusions">结论</a><a class="navlink" href="#cross-model">三模型对照</a><a class="navlink" href="#all-s">全部 S</a><a class="navlink" href="#subtypes">子类别</a><a class="navlink" href="#depth">深度</a><a class="navlink" href="#dominant">主导×深度</a><a class="navlink" href="#dose">数量控制</a></div></nav>
<main>
<section><div class="definitions"><div><b>Motion Impact ↑</b><br><span class="muted">RAFT 流场、强运动曲线、物体轨迹与速度相对 baseline 的归一化改变。只衡量改变大小。</span></div><div><b>GT gain ↑</b><br><span class="muted">正值表示比 baseline 更接近 49 帧 GT；必须结合按 case bootstrap 的 95% CI。</span></div><div><b>Benchmark Δ ↑</b><br><span class="muted">消融分数减 baseline 分数。不同评测器可能冲突，不合成为单一“物理正确”结论。</span></div></div></section>
<section id="conclusions"><div class="section-head"><div><h2>审慎结论</h2><p class="muted">结论按证据等级展示；“三模型复现”仅指本页受测模型、seed 851 和 20 个 case，不代表外部模型普适性。</p></div><span class="muted">更新 {updated}</span></div>
<div class="evidence-key"><div><b>G3-D</b><span>三个模型都有直接配对指标和满足方向的区间。</span></div><div><b>G3-R</b><span>三个模型均值同方向，但差值未检验或至少一个区间跨 0。</span></div><div><b>模型内</b><span>当前模型上的描述性排序，不外推。</span></div><div><b>I</b><span>机制解释或待验证假设。</span></div></div>
<div class="findings">{conclusion_html}</div></section>
<section id="cross-model"><div class="section-head"><div><h2>Wan+LoRA / Wan+xSSC / Wan+OpenVid 对照</h2><p class="muted">固定 seed 851、20 个相同 case 和冻结的公共 S-head 列表。OpenVid 未独立重做 S-head 分类。</p></div><a href="analysis.md">查看精简报告</a></div>
<h3>等量 Local-32 / Same-frame-32 与 Union-64</h3>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>阶段</th><th>Local Impact</th><th>Same Impact</th><th>Union Impact</th><th>Local GT gain</th><th>Same GT gain</th><th>Union GT gain</th></tr></thead><tbody>{cross_model_feature_table(motion)}</tbody></table></div>
<h3 style="margin-top:18px">主导类别全程消融 [00,40)</h3><p class="muted">Local all=100 heads，Same all=59 heads。Impact/head 是近似归一化；Physics-IQ 与 PMF 方向冲突时不合并。</p>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>Impact/head Local / Same</th><th>GT gain Local / Same</th><th>PMF Δ Local [95% CI]</th><th>PMF Δ Same [95% CI]</th><th>Physics-IQ Δ Local / Same</th></tr></thead><tbody>{cross_model_dominant_table(motion, paired)}</tbody></table></div>
<h3 style="margin-top:18px">0–10 深度相关现象</h3><p class="muted">“单位 head 最大”只描述当前组合的 Impact/head，不等于单 head 因果重要性。</p>
<div class="table-wrap"><table><thead><tr><th>模型</th><th>单位 head Impact 最大子集</th><th>Impact/head</th><th>Local-Late GT gain [95% CI]</th><th>Local-Late PMF Δ [95% CI]</th></tr></thead><tbody>{cross_model_depth_table(motion, paired)}</tbody></table></div></section>
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
    paired = paired_head_benchmark(HEAD_ROLE_MANIFEST)
    validate_cross_model_inputs(motion, paired)
    status_path = GALLERY_ROOT / "multiseed/motion-n-analysis/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_stage_plot(benchmark, args.output_dir / "all_s_benchmark_stage.png")
    save_dose_plot(dose, args.output_dir / "s_dose_control.png")
    paired.to_csv(
        args.output_dir / "cross_model_paired_metrics.csv",
        index=False,
    )
    report = build_cross_model_markdown(motion, interactions, paired)
    atomic_write(args.report_path, report)
    atomic_write(args.output_dir / "analysis.md", report)
    atomic_write(
        args.output_dir / "index.html",
        build_html(motion, interactions, benchmark, dose, paired, status),
    )
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "motion": str(MOTION_DIR / "aggregate_metrics.csv"),
            "benchmark": str(BENCH_DIR / "paired_vs_baseline_summary.csv"),
            "dose": str(DOSE_DIR / "partial_aggregate.csv"),
            "head_role_manifest": str(HEAD_ROLE_MANIFEST),
            "status": str(status_path),
        },
        "rows": {
            "motion": len(motion),
            "benchmark_s": int((benchmark["role"] == "S").sum()),
            "dose_s": int((dose["role"] == "S").sum()),
            "cross_model_paired_metrics": len(paired),
        },
        "report": str(args.report_path),
    }
    atomic_write(
        args.output_dir / "manifest.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"[s-head-analysis] output={args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
