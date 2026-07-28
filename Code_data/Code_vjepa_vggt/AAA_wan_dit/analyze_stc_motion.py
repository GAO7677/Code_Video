#!/usr/bin/env python3
"""Rank phased ablations by motion impact and GT-relative plausibility."""

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
from scipy.signal import savgol_filter


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis"
)
DEFAULT_REPORT_DIR = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/motion-analysis"
)
MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
ROLE_COLORS = {"S": "#36a692", "T": "#e5a93f", "C": "#d46c78"}
PLAUSIBILITY_COMPONENTS = (
    "flow_vector_rmse_gt",
    "flow_top05_curve_rmse_gt",
    "object_trajectory_rmse_gt",
    "object_speed_curve_rmse_gt",
    "object_acceleration_curve_rmse_gt",
    "background_drift_abs_error_gt",
    "object_visibility_abs_error_gt",
)
PLAUSIBILITY_SCALES = {
    "flow_vector_rmse_gt": 0.005,
    "flow_top05_curve_rmse_gt": 0.010,
    "object_trajectory_rmse_gt": 0.050,
    "object_speed_curve_rmse_gt": 0.005,
    "object_acceleration_curve_rmse_gt": 0.002,
    "background_drift_abs_error_gt": 0.005,
    "object_visibility_abs_error_gt": 0.250,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_OUTPUT_ROOT / "inventory.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--minimum-seeds", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_features(
    output_root: Path,
    entry: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    feature_dir = output_root / "features" / entry["source"]["cache_key"]
    archive_path = feature_dir / "features.npz"
    metadata_path = feature_dir / "metadata.json"
    if not archive_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            return None
        with np.load(archive_path) as archive:
            arrays = {key: archive[key] for key in archive.files}
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return arrays, metadata


def interpolate_and_smooth(
    tracks: np.ndarray,
    visibility: np.ndarray,
    start_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    segment = tracks[start_frame:].astype(np.float64)
    visible = visibility[start_frame:].astype(bool)
    times = np.arange(len(segment))
    result = np.full_like(segment, np.nan)
    valid_points = visible.mean(axis=0) >= 0.25
    for point in np.flatnonzero(valid_points):
        indices = np.flatnonzero(visible[:, point])
        if len(indices) < 2:
            valid_points[point] = False
            continue
        for axis in range(2):
            result[:, point, axis] = np.interp(
                times,
                indices,
                segment[indices, point, axis],
            )
    if len(segment) >= 7:
        result[:, valid_points] = savgol_filter(
            result[:, valid_points],
            window_length=7,
            polyorder=2,
            axis=0,
            mode="interp",
        )
    return result, visible, valid_points


def finite_median(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmedian(values, axis=axis)


def track_state(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    start_frame: int,
) -> dict[str, Any]:
    tracks, visibility, valid = interpolate_and_smooth(
        arrays["tracks_norm"],
        arrays["track_visibility"],
        start_frame,
    )
    region_ids = arrays["track_region_ids"].astype(int)
    regions = metadata["settings"].get("track_regions", [])
    object_region_ids = {
        index
        for index, region in enumerate(regions)
        if region.get("region_type") == "object"
    }
    background_region_ids = {
        index
        for index, region in enumerate(regions)
        if region.get("region_type") == "background"
    }
    if not object_region_ids:
        object_region_ids = {int(value) for value in np.unique(region_ids) if value >= 0}
    object_mask = np.isin(region_ids, sorted(object_region_ids)) & valid
    background_mask = np.isin(region_ids, sorted(background_region_ids)) & valid
    if not background_mask.any():
        background_mask = (~object_mask) & valid
    if not object_mask.any() or not background_mask.any():
        raise ValueError("Object/background CoTracker regions are required")

    displacement = tracks - tracks[0:1]
    global_displacement = finite_median(displacement[:, background_mask], axis=1)
    corrected = displacement - global_displacement[:, None, :]
    velocity = np.diff(corrected, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)
    object_displacement = np.linalg.norm(corrected[:, object_mask], axis=-1)
    object_speed = np.linalg.norm(velocity[:, object_mask], axis=-1)
    object_acceleration = np.linalg.norm(acceleration[:, object_mask], axis=-1)
    object_jerk = np.linalg.norm(jerk[:, object_mask], axis=-1)
    background_displacement = np.linalg.norm(
        displacement[:, background_mask],
        axis=-1,
    )
    return {
        "tracks": tracks,
        "visibility": visibility,
        "valid_points": valid,
        "object_mask": object_mask,
        "background_mask": background_mask,
        "corrected_displacement": corrected,
        "object_displacement_curve": finite_median(object_displacement, axis=1),
        "object_speed_curve": finite_median(object_speed, axis=1),
        "object_acceleration_curve": finite_median(object_acceleration, axis=1),
        "object_jerk_curve": finite_median(object_jerk, axis=1),
        "background_drift_curve": finite_median(background_displacement, axis=1),
        "object_visibility": float(
            arrays["track_visibility"][start_frame:, np.isin(region_ids, sorted(object_region_ids))].mean()
        ),
    }


def rmse(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    mask = np.isfinite(first) & np.isfinite(second)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(np.square(first[mask] - second[mask]))))


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def pair_metrics(
    arrays: dict[str, np.ndarray],
    state: dict[str, Any],
    reference_arrays: dict[str, np.ndarray],
    reference_state: dict[str, Any],
    flow_start: int,
    suffix: str,
) -> dict[str, float]:
    flow = arrays["flow_norm"][flow_start:].astype(np.float32)
    reference_flow = reference_arrays["flow_norm"][flow_start:].astype(np.float32)
    shared_objects = (
        state["object_mask"]
        & reference_state["object_mask"]
        & state["valid_points"]
        & reference_state["valid_points"]
    )
    trajectory_error = rmse(
        state["corrected_displacement"][:, shared_objects],
        reference_state["corrected_displacement"][:, shared_objects],
    )
    return {
        f"flow_vector_rmse_{suffix}": rmse(flow, reference_flow),
        f"flow_top05_curve_rmse_{suffix}": rmse(
            arrays["flow_top05"][flow_start:],
            reference_arrays["flow_top05"][flow_start:],
        ),
        f"object_trajectory_rmse_{suffix}": trajectory_error,
        f"object_speed_curve_rmse_{suffix}": rmse(
            state["object_speed_curve"],
            reference_state["object_speed_curve"],
        ),
        f"object_acceleration_curve_rmse_{suffix}": rmse(
            state["object_acceleration_curve"],
            reference_state["object_acceleration_curve"],
        ),
        f"background_drift_abs_error_{suffix}": abs(
            float(np.nanmean(state["background_drift_curve"]))
            - float(np.nanmean(reference_state["background_drift_curve"]))
        ),
        f"object_visibility_abs_error_{suffix}": abs(
            state["object_visibility"] - reference_state["object_visibility"]
        ),
    }


def impact_score(
    metrics: dict[str, float],
    baseline_arrays: dict[str, np.ndarray],
    baseline_state: dict[str, Any],
    flow_start: int,
) -> float:
    ratios = [
        metrics["flow_vector_rmse_baseline"]
        / max(rms(baseline_arrays["flow_norm"][flow_start:].astype(np.float32)), 0.002),
        metrics["flow_top05_curve_rmse_baseline"]
        / max(rms(baseline_arrays["flow_top05"][flow_start:]), 0.002),
        metrics["object_trajectory_rmse_baseline"]
        / max(rms(baseline_state["corrected_displacement"][:, baseline_state["object_mask"]]), 0.01),
        metrics["object_speed_curve_rmse_baseline"]
        / max(rms(baseline_state["object_speed_curve"]), 0.002),
    ]
    finite = [value for value in ratios if math.isfinite(value)]
    return float(np.mean(np.log1p(finite))) if finite else float("nan")


def calibrated_plausibility_distance(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    normalized_columns = []
    for component in PLAUSIBILITY_COMPONENTS:
        values = result[component].astype(float)
        scale = PLAUSIBILITY_SCALES[component]
        column = f"{component}_scaled"
        result[column] = np.log1p(np.maximum(values, 0.0) / scale)
        normalized_columns.append(column)
    result["plausibility_distance_gt"] = result[normalized_columns].mean(axis=1)
    baseline = (
        result[result["variant"] == "baseline"]
        .set_index(["model", "seed"])["plausibility_distance_gt"]
        .to_dict()
    )
    result["plausibility_gain_vs_baseline"] = [
        baseline.get((row.model, row.seed), np.nan) - row.plausibility_distance_gt
        for row in result.itertuples()
    ]
    return result


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_rows(frame: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    records = []
    keys = ["model", "variant", "role", "denoise_start", "denoise_end"]
    for key, group in frame.groupby(keys, dropna=False):
        impact = group["impact_score"].to_numpy(float)
        gain = group["plausibility_gain_vs_baseline"].to_numpy(float)
        impact_low, impact_high = bootstrap_ci(impact, bootstrap_samples, rng)
        gain_low, gain_high = bootstrap_ci(gain, bootstrap_samples, rng)
        records.append(
            {
                **dict(zip(keys, key)),
                "n_seeds": int(group["seed"].nunique()),
                "impact_mean": float(np.nanmean(impact)),
                "impact_std": float(np.nanstd(impact, ddof=1)) if len(impact) > 1 else 0.0,
                "impact_ci_low": impact_low,
                "impact_ci_high": impact_high,
                "plausibility_gain_mean": float(np.nanmean(gain)),
                "plausibility_gain_std": (
                    float(np.nanstd(gain, ddof=1)) if len(gain) > 1 else 0.0
                ),
                "plausibility_gain_ci_low": gain_low,
                "plausibility_gain_ci_high": gain_high,
            }
        )
    aggregate = pd.DataFrame(records)
    labels = []
    for row in aggregate.itertuples():
        if row.variant == "baseline":
            labels.append("baseline")
        elif row.n_seeds < 3:
            labels.append("insufficient_seeds")
        elif row.plausibility_gain_ci_low > 0:
            labels.append("closer_to_gt")
        elif row.plausibility_gain_ci_high < 0:
            labels.append("farther_from_gt")
        else:
            labels.append("uncertain")
    aggregate["reasonableness_label"] = labels
    return aggregate


def save_scatter(aggregate: pd.DataFrame, path: Path, minimum_seeds: int) -> None:
    data = aggregate[
        (aggregate["variant"] != "baseline")
        & (aggregate["n_seeds"] >= minimum_seeds)
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, model in zip(axes, MODEL_NAMES):
        subset = data[data["model"] == model]
        for row in subset.itertuples():
            axis.errorbar(
                row.impact_mean,
                row.plausibility_gain_mean,
                xerr=[[row.impact_mean - row.impact_ci_low], [row.impact_ci_high - row.impact_mean]],
                yerr=[
                    [row.plausibility_gain_mean - row.plausibility_gain_ci_low],
                    [row.plausibility_gain_ci_high - row.plausibility_gain_mean],
                ],
                fmt="o",
                color=ROLE_COLORS.get(row.role, "#999999"),
                alpha=0.85,
                capsize=2,
            )
            axis.annotate(
                f"{row.role}[{int(row.denoise_start)},{int(row.denoise_end)})",
                (row.impact_mean, row.plausibility_gain_mean),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        axis.axhline(0, color="#777777", linewidth=1)
        axis.set_title(MODEL_NAMES[model])
        axis.set_xlabel("Impact vs same-seed baseline (larger = stronger)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("GT plausibility gain (positive = closer to GT)")
    fig.suptitle("Motion impact and GT-relative plausibility, mean with 95% bootstrap CI")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmaps(aggregate: pd.DataFrame, path: Path, minimum_seeds: int) -> None:
    data = aggregate[
        (aggregate["variant"] != "baseline")
        & (aggregate["n_seeds"] >= minimum_seeds)
    ].copy()
    data["stage"] = [
        f"[{int(start)},{int(end)})"
        for start, end in zip(data["denoise_start"], data["denoise_end"])
    ]
    stages = sorted(
        data["stage"].unique(),
        key=lambda value: tuple(int(part) for part in value.strip("[]()").split(",")),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    for row_index, model in enumerate(MODEL_NAMES):
        subset = data[data["model"] == model]
        for column_index, (metric, title) in enumerate(
            (
                ("impact_mean", "Impact (higher = larger change)"),
                ("plausibility_gain_mean", "GT plausibility gain (higher = better)"),
            )
        ):
            matrix = np.full((3, len(stages)), np.nan)
            for role_index, role in enumerate(("S", "T", "C")):
                for stage_index, stage in enumerate(stages):
                    cells = subset[(subset["role"] == role) & (subset["stage"] == stage)]
                    if len(cells):
                        matrix[role_index, stage_index] = float(cells.iloc[0][metric])
            axis = axes[row_index, column_index]
            finite = matrix[np.isfinite(matrix)]
            if metric == "plausibility_gain_mean" and len(finite):
                bound = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
                image = axis.imshow(matrix, cmap="RdYlGn", vmin=-bound, vmax=bound, aspect="auto")
            else:
                image = axis.imshow(matrix, cmap="viridis", aspect="auto")
            axis.set_xticks(range(len(stages)), stages, rotation=45, ha="right")
            axis.set_yticks(range(3), ("S", "T", "C"))
            axis.set_title(f"{MODEL_NAMES[model]} | {title}")
            for y in range(matrix.shape[0]):
                for x in range(matrix.shape[1]):
                    if np.isfinite(matrix[y, x]):
                        axis.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_temporal_curves(
    curves: pd.DataFrame,
    aggregate: pd.DataFrame,
    path: Path,
    minimum_seeds: int,
) -> None:
    metric_specs = (
        ("object_displacement", "Object displacement", "normalized image units"),
        ("flow_top05", "Top-5% RAFT motion", "normalized units / frame"),
        ("background_drift", "Background drift", "normalized image units"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex="col")
    gt = curves[curves["model"] == "gt"]
    for row_index, model in enumerate(MODEL_NAMES):
        model_curves = curves[curves["model"] == model]
        candidates = aggregate[
            (aggregate["model"] == model)
            & (aggregate["variant"] != "baseline")
            & (aggregate["n_seeds"] >= minimum_seeds)
        ].nlargest(3, "impact_mean")
        variants = ["baseline", *candidates["variant"].tolist()]
        for column_index, (metric, title, unit) in enumerate(metric_specs):
            axis = axes[row_index, column_index]
            if len(gt):
                gt_curve = gt.groupby("relative_frame")[metric].mean()
                axis.plot(
                    gt_curve.index,
                    gt_curve.values,
                    color="#111111",
                    linewidth=2.2,
                    linestyle="--",
                    label="GT",
                )
            for variant in variants:
                subset = model_curves[model_curves["variant"] == variant]
                if subset.empty:
                    continue
                grouped = subset.groupby("relative_frame")[metric]
                mean = grouped.mean()
                std = grouped.std().fillna(0.0)
                role = (
                    "baseline"
                    if variant == "baseline"
                    else str(subset.iloc[0]["role"])
                )
                color = "#3876b7" if variant == "baseline" else ROLE_COLORS.get(role, "#777777")
                label = (
                    "Baseline"
                    if variant == "baseline"
                    else f"{role}[{int(subset.iloc[0]['denoise_start'])},{int(subset.iloc[0]['denoise_end'])})"
                )
                axis.plot(mean.index, mean.values, color=color, linewidth=1.6, label=label)
                axis.fill_between(
                    mean.index,
                    mean.values - std.values,
                    mean.values + std.values,
                    color=color,
                    alpha=0.13,
                )
            axis.set_title(f"{MODEL_NAMES[model]} | {title}")
            axis.set_ylabel(unit)
            axis.grid(alpha=0.2)
            if row_index == 2:
                axis.set_xlabel("Frames after final context frame")
            axis.legend(fontsize=7, loc="upper right")
    fig.suptitle("Temporal motion curves, mean +/- cross-seed standard deviation")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_html(
    aggregate: pd.DataFrame,
    per_video: pd.DataFrame,
    inventory: dict[str, Any],
    minimum_seeds: int,
) -> str:
    ranked = aggregate[
        (aggregate["variant"] != "baseline")
        & (aggregate["n_seeds"] >= minimum_seeds)
    ].sort_values(["impact_mean"], ascending=False)
    rows = []
    for row in ranked.itertuples():
        label_class = {
            "closer_to_gt": "good",
            "farther_from_gt": "bad",
            "uncertain": "uncertain",
            "insufficient_seeds": "uncertain",
        }.get(row.reasonableness_label, "")
        rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_NAMES.get(row.model, row.model))}</td>"
            f"<td>{html.escape(str(row.role))}</td>"
            f"<td>[{int(row.denoise_start)},{int(row.denoise_end)})</td>"
            f"<td>{row.n_seeds}</td>"
            f"<td>{row.impact_mean:.3f} ± {row.impact_std:.3f}</td>"
            f"<td>{row.plausibility_gain_mean:+.3f} ± {row.plausibility_gain_std:.3f}</td>"
            f"<td class='{label_class}'>{html.escape(row.reasonableness_label)}</td>"
            "</tr>"
        )
    prompt = inventory["case"]["prompt"]
    completed = len(per_video)
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>运动影响与物理合理性</title>
<style>
:root{{--bg:#111416;--panel:#1b1f22;--line:#353b40;--text:#f2f4f5;--muted:#a9b0b6;--good:#62c49a;--bad:#ec7f76;--warn:#e4b65d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 system-ui,sans-serif}}
header,main{{max-width:1500px;margin:auto;padding:16px 20px}}header{{border-bottom:1px solid var(--line)}}h1,h2,p{{margin:0}}h1{{font-size:23px}}h2{{font-size:18px;margin:22px 0 8px}}.muted{{color:var(--muted)}}.protocol{{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:10px;margin-top:14px}}.note{{padding:11px;background:var(--panel);border:1px solid var(--line);border-radius:6px}}
.plots{{display:grid;grid-template-columns:1fr;gap:14px}}.plots img{{display:block;width:100%;background:#fff;border:1px solid var(--line)}}
.table-wrap{{overflow:auto}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:7px 9px;border:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}thead th{{position:sticky;top:0;background:#252a2e}}.good{{color:var(--good);font-weight:700}}.bad{{color:var(--bad);font-weight:700}}.uncertain{{color:var(--warn)}}a{{color:#72c8b7}}
@media(max-width:800px){{.protocol{{grid-template-columns:1fr}}}}
</style></head><body><header>
<h1>运动影响与物理合理性</h1>
<p class="muted">{html.escape(prompt)}</p>
<p class="muted">已分析 {completed} 个模型输出；汇总仅展示至少 {minimum_seeds} 个 seed 的配置。</p>
</header><main>
<div class="protocol">
<div class="note"><b>变化大小（Impact）</b><br>同模型、同 seed、同输入下，与 baseline 的 RAFT 向量场、运动强度曲线、物体轨迹和速度曲线差异。越大表示该消融改变运动越强，不代表质量更好。</div>
<div class="note"><b>合理性（GT plausibility gain）</b><br>比较消融和 baseline 各自到 GT 的固定尺度综合距离；正值表示消融让运动更接近 GT，负值表示更远。GT 在 context 后基本静止，因此额外漂移、弹飞、背景运动及追踪丢失都会被惩罚；所有原始分项也保留在 CSV 中。</div>
<div class="note"><b>时间与空间对齐</b><br>完整提取 49 帧；统计从 frame 7→8 开始。SAM2 物体/背景查询固定在最后一帧 context；GT 按 7:4 中心裁剪后缩放，所有轨迹使用归一化坐标。</div>
<div class="note"><b>结论边界</b><br>RAFT/CoTracker 是运动证据，不是物理定律判定器。遮挡、形变或物体消失会降低追踪可靠性，因此高影响且置信区间跨 0 的配置标为 uncertain，需要结合视频人工核验。</div>
</div>
<h2>二维结论图</h2><div class="plots"><img src="impact_plausibility_scatter.png"><img src="role_stage_heatmaps.png"><img src="temporal_motion_curves.png"></div>
<h2>配置汇总</h2><div class="table-wrap"><table><thead><tr><th>模型</th><th>Head 类</th><th>去噪区间</th><th>Seeds</th><th>Impact ↑</th><th>GT gain ↑</th><th>判断</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table></div>
<p class="muted" style="margin-top:10px">原始数据：<a href="per_video_metrics.csv">per_video_metrics.csv</a> · <a href="aggregate_metrics.csv">aggregate_metrics.csv</a> · <a href="temporal_curves.csv">temporal_curves.csv</a> · <a href="../stc-phased/">视频对照页</a></p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    feature_map: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
    for entry in inventory["entries"]:
        loaded = load_features(args.output_root, entry)
        if loaded is not None:
            feature_map[entry["entry_id"]] = loaded
    if "gt" not in feature_map:
        raise RuntimeError("GT features have not been extracted")
    start_frame = args.context_frames - 1
    flow_start = start_frame
    gt_arrays, gt_metadata = feature_map["gt"]
    gt_state = track_state(gt_arrays, gt_metadata, start_frame)

    entries_by_id = {entry["entry_id"]: entry for entry in inventory["entries"]}
    baseline_features: dict[tuple[str, int], tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]] = {}
    for entry_id, (arrays, metadata) in feature_map.items():
        entry = entries_by_id[entry_id]
        if entry["variant"] == "baseline":
            baseline_features[(entry["model"], int(entry["seed"]))] = (
                arrays,
                metadata,
                track_state(arrays, metadata, start_frame),
            )

    rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for relative_frame in range(len(gt_state["object_displacement_curve"])):
        curve_rows.append(
            {
                "entry_id": "gt",
                "model": "gt",
                "seed": np.nan,
                "variant": "gt",
                "role": "gt",
                "denoise_start": np.nan,
                "denoise_end": np.nan,
                "relative_frame": relative_frame,
                "absolute_frame": start_frame + relative_frame,
                "object_displacement": gt_state["object_displacement_curve"][relative_frame],
                "object_speed": (
                    gt_state["object_speed_curve"][relative_frame]
                    if relative_frame < len(gt_state["object_speed_curve"])
                    else np.nan
                ),
                "flow_top05": (
                    gt_arrays["flow_top05"][flow_start + relative_frame]
                    if flow_start + relative_frame < len(gt_arrays["flow_top05"])
                    else np.nan
                ),
                "background_drift": gt_state["background_drift_curve"][relative_frame],
            }
        )
    for entry_id, (arrays, metadata) in feature_map.items():
        entry = entries_by_id[entry_id]
        if entry["kind"] != "generated":
            continue
        baseline = baseline_features.get((entry["model"], int(entry["seed"])))
        if baseline is None:
            continue
        baseline_arrays, _, baseline_state = baseline
        state = track_state(arrays, metadata, start_frame)
        gt_metrics = pair_metrics(
            arrays,
            state,
            gt_arrays,
            gt_state,
            flow_start,
            "gt",
        )
        baseline_metrics = pair_metrics(
            arrays,
            state,
            baseline_arrays,
            baseline_state,
            flow_start,
            "baseline",
        )
        if entry["variant"] == "baseline":
            impact = 0.0
        else:
            impact = impact_score(
                baseline_metrics,
                baseline_arrays,
                baseline_state,
                flow_start,
            )
        denoise_range = entry["denoise_step_range"] or [np.nan, np.nan]
        rows.append(
            {
                "entry_id": entry_id,
                "model": entry["model"],
                "seed": int(entry["seed"]),
                "variant": entry["variant"],
                "role": entry["role"] or "baseline",
                "denoise_start": denoise_range[0],
                "denoise_end": denoise_range[1],
                "impact_score": impact,
                "object_visibility": state["object_visibility"],
                "background_drift_mean": float(np.nanmean(state["background_drift_curve"])),
                "object_displacement_mean": float(np.nanmean(state["object_displacement_curve"])),
                "object_speed_mean": float(np.nanmean(state["object_speed_curve"])),
                "object_acceleration_mean": float(np.nanmean(state["object_acceleration_curve"])),
                "object_jerk_mean": float(np.nanmean(state["object_jerk_curve"])),
                **gt_metrics,
                **baseline_metrics,
            }
        )
        for relative_frame in range(len(state["object_displacement_curve"])):
            curve_rows.append(
                {
                    "entry_id": entry_id,
                    "model": entry["model"],
                    "seed": int(entry["seed"]),
                    "variant": entry["variant"],
                    "role": entry["role"] or "baseline",
                    "denoise_start": denoise_range[0],
                    "denoise_end": denoise_range[1],
                    "relative_frame": relative_frame,
                    "absolute_frame": start_frame + relative_frame,
                    "object_displacement": state["object_displacement_curve"][relative_frame],
                    "object_speed": (
                        state["object_speed_curve"][relative_frame]
                        if relative_frame < len(state["object_speed_curve"])
                        else np.nan
                    ),
                    "flow_top05": (
                        arrays["flow_top05"][flow_start + relative_frame]
                        if flow_start + relative_frame < len(arrays["flow_top05"])
                        else np.nan
                    ),
                    "background_drift": state["background_drift_curve"][relative_frame],
                }
            )
    if not rows:
        raise RuntimeError("No generated entries have both features and a same-seed baseline")
    per_video = calibrated_plausibility_distance(pd.DataFrame(rows))
    aggregate = aggregate_rows(per_video, args.bootstrap_samples)
    curves = pd.DataFrame(curve_rows)

    results_dir = args.output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(results_dir / "per_video_metrics.csv", index=False)
    aggregate.to_csv(results_dir / "aggregate_metrics.csv", index=False)
    curves.to_csv(results_dir / "temporal_curves.csv", index=False)
    protocol = {
        "schema_version": 1,
        "context_frames": args.context_frames,
        "first_evaluated_flow_pair": [start_frame, start_frame + 1],
        "track_query_frame": start_frame,
        "spatial_alignment": "center crop to 7:4, then resize",
        "impact_reference": "same model, same seed baseline",
        "plausibility_reference": "GT/source 49-frame video",
        "plausibility_components": list(PLAUSIBILITY_COMPONENTS),
        "plausibility_component_scales": PLAUSIBILITY_SCALES,
        "plausibility_gain_sign": "positive means closer to GT than baseline",
        "bootstrap_samples": args.bootstrap_samples,
        "minimum_seeds_for_report": args.minimum_seeds,
        "limitations": [
            "RAFT and CoTracker estimates can be affected by appearance changes and occlusion.",
            "GT similarity is evidence of plausibility for this case, not a universal physics oracle.",
            "All temporal comparisons use normalized frame index across 49 frames.",
        ],
    }
    atomic_write_text(
        results_dir / "protocol.json",
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n",
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    save_scatter(
        aggregate,
        args.report_dir / "impact_plausibility_scatter.png",
        args.minimum_seeds,
    )
    save_heatmaps(
        aggregate,
        args.report_dir / "role_stage_heatmaps.png",
        args.minimum_seeds,
    )
    save_temporal_curves(
        curves,
        aggregate,
        args.report_dir / "temporal_motion_curves.png",
        args.minimum_seeds,
    )
    per_video.to_csv(args.report_dir / "per_video_metrics.csv", index=False)
    aggregate.to_csv(args.report_dir / "aggregate_metrics.csv", index=False)
    curves.to_csv(args.report_dir / "temporal_curves.csv", index=False)
    atomic_write_text(
        args.report_dir / "index.html",
        build_html(
            aggregate,
            per_video,
            inventory,
            args.minimum_seeds,
        ),
    )
    print(
        f"[analysis] feature_entries={len(feature_map)} per_video={len(per_video)} "
        f"aggregate={len(aggregate)}"
    )
    print(args.report_dir / "index.html")


if __name__ == "__main__":
    main()
