#!/usr/bin/env python3
"""Summarize and plot the 67-case Wan DiT ablation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan")
DEFAULT_ALLOWLIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "_metric_plots"
METHOD_PATTERN = re.compile(
    r"^(whole_block|self_attn_zero|object_cross_attn|"
    r"text_cross_attn_zero|ffn_zero|lora_off)_block(\d{2})$"
)
TRAINING_CHECKPOINT_PATTERN = re.compile(
    r"^xssc_lora_("
    r"full_sa|full_sa_resume|"
    r"s_head59|s_head59_resume|"
    r"t_head70|t_head70_resume|"
    r"t_head70_no_object|"
    r"t_head70_slot_dedup_merge|"
    r"slot_dedup_merge|"
    r"full_sa_no_object"
    r")_step-(\d+)_steps\d+_\d+x\d+_ctx\d+_\d+f(?:_.+)?$"
)
TRAINING_VARIANT_ALIASES = {
    "full_sa_resume": "full_sa",
    "s_head59_resume": "s_head59",
    "t_head70_resume": "t_head70",
}
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MODE_LABELS = {
    "baseline": "Baseline",
    "whole_block": "Whole block bypass",
    "self_attn_zero": "Self-attention output = 0",
    "object_cross_attn": "Object cross-attention output = 0",
    "text_cross_attn_zero": "Text cross-attention output = 0",
    "ffn_zero": "FFN output = 0",
    "lora_off": "LoRA disabled",
}
MODE_ORDER = {
    mode: index
    for index, mode in enumerate(
        (
            "baseline",
            "whole_block",
            "self_attn_zero",
            "object_cross_attn",
            "text_cross_attn_zero",
            "ffn_zero",
            "lora_off",
        )
    )
}
MODEL_STYLES = {
    "wan_lora": {"linestyle": "--", "marker": "o"},
    "xssc": {"linestyle": "-", "marker": "s"},
    "physrvg": {"linestyle": "-", "marker": "D"},
}
MODE_COLORS = {
    "whole_block": "#E41A1C",
    "self_attn_zero": "#2166D1",
    "object_cross_attn": "#009E73",
    "text_cross_attn_zero": "#7B2CBF",
    "ffn_zero": "#E67E22",
    "lora_off": "#5D6D7E",
}
TRAINING_VARIANT_LABELS = {
    "full_sa": "Full-SA + Object",
    "s_head59": "S-head59 + Object",
    "t_head70": "T-head70 + Object",
    "t_head70_no_object": "T-head70 + No-Object",
    "t_head70_slot_dedup_merge": "T-head70 + Object + Slot-Dedup",
    "slot_dedup_merge": "Full-SA + Object + Slot-Dedup",
    "full_sa_no_object": "Full-SA + No-Object",
}
TRAINING_VARIANT_COLORS = {
    "full_sa": "#D62728",
    "s_head59": "#2CA02C",
    "t_head70": "#9467BD",
    "t_head70_no_object": "#E377C2",
    "t_head70_slot_dedup_merge": "#17BECF",
    "slot_dedup_merge": "#1F77B4",
    "full_sa_no_object": "#FF7F0E",
}
TRAINING_VARIANT_MARKERS = {
    "full_sa": "o",
    "s_head59": "s",
    "t_head70": "^",
    "t_head70_no_object": "h",
    "t_head70_slot_dedup_merge": "v",
    "slot_dedup_merge": "D",
    "full_sa_no_object": "X",
}
TRAINING_VARIANT_LINESTYLES = {
    "full_sa": "-",
    "s_head59": "--",
    "t_head70": "-.",
    "t_head70_no_object": (0, (3, 2)),
    "t_head70_slot_dedup_merge": (0, (3, 1, 1, 1)),
    "slot_dedup_merge": ":",
    "full_sa_no_object": (0, (5, 1)),
}
TRAINING_VARIANT_ORDER = {
    variant: index
    for index, variant in enumerate(
        (
            "full_sa",
            "s_head59",
            "t_head70",
            "t_head70_no_object",
            "t_head70_slot_dedup_merge",
            "slot_dedup_merge",
            "full_sa_no_object",
        )
    )
}


def nested_score(*keys: str) -> Callable[[dict[str, Any]], float | None]:
    def extract(payload: dict[str, Any]) -> float | None:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    return extract


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    direction: str
    extract: Callable[[dict[str, Any]], float | None]


METRICS = (
    Metric(
        "physics_iq_with_context",
        "Physics-IQ with context",
        "higher",
        nested_score("physics_iq_with_context", "score"),
    ),
    Metric(
        "physics_iq_without_context",
        "Physics-IQ without context",
        "higher",
        nested_score("physics_iq_without_context", "score"),
    ),
    Metric(
        "physics_iq_verified_proxy",
        "Physics-IQ Verified proxy",
        "higher",
        nested_score("physics_iq_verified_proxy", "score"),
    ),
    Metric(
        "pmf_with_context",
        "PMF with context",
        "higher",
        nested_score("pmf_with_context", "score"),
    ),
    Metric(
        "pmf_without_context",
        "PMF without context",
        "higher",
        nested_score("pmf_without_context", "score"),
    ),
    Metric(
        "wmreward",
        "WMReward surprise",
        "lower",
        nested_score("wmreward", "surprise"),
    ),
    Metric(
        "vbench_subject_consistency",
        "VBench subject consistency",
        "higher",
        nested_score("vbench_subject_consistency", "score"),
    ),
    Metric(
        "vbench_background_consistency",
        "VBench background consistency",
        "higher",
        nested_score("vbench_background_consistency", "score"),
    ),
    Metric(
        "vbench_temporal_flickering",
        "VBench temporal flickering",
        "higher",
        nested_score("vbench_temporal_flickering", "score"),
    ),
    Metric(
        "vbench_motion_smoothness",
        "VBench motion smoothness",
        "higher",
        nested_score("vbench_motion_smoothness", "score"),
    ),
    Metric(
        "vbench_dynamic_degree",
        "VBench dynamic degree",
        "higher",
        nested_score("vbench_dynamic_degree", "score"),
    ),
    Metric(
        "vbench_aesthetic_quality",
        "VBench aesthetic quality",
        "higher",
        nested_score("vbench_aesthetic_quality", "score"),
    ),
    Metric(
        "vbench_imaging_quality",
        "VBench imaging quality",
        "higher",
        nested_score("vbench_imaging_quality", "score"),
    ),
    Metric(
        "videophy2_sa",
        "VideoPhy2 SA (generated only)",
        "higher",
        nested_score("videophy2", "sa_score"),
    ),
    Metric(
        "videophy2_pc",
        "VideoPhy2 PC (generated only)",
        "higher",
        nested_score("videophy2", "pc_score"),
    ),
    Metric(
        "videophy2_joint_rate",
        "VideoPhy2 joint rate (generated only)",
        "higher",
        nested_score("videophy2", "joint_pass"),
    ),
    Metric(
        "videophy2_pc_raw",
        "VideoPhy2 PC raw (full video)",
        "higher",
        nested_score("videophy2", "pc_raw_score"),
    ),
    Metric(
        "cosmos_reason1",
        "Cosmos-Reason1",
        "higher",
        nested_score("cosmos_reason1", "score"),
    ),
)


@dataclass(frozen=True)
class MetricStat:
    method: Method
    metric: Metric
    count: int
    mean: float | None
    complete: bool


@dataclass(frozen=True)
class Method:
    method_id: str
    model: str
    mode: str
    block_id: int | None
    result_dir: Path

    @property
    def sort_key(self) -> tuple[int, int, int]:
        model_order = {"wan_lora": 0, "xssc": 1, "physrvg": 2}.get(
            self.model, 99
        )
        block_order = -1 if self.block_id is None else self.block_id
        return model_order, block_order, MODE_ORDER[self.mode]


@dataclass(frozen=True)
class TrainingCheckpoint:
    variant: str
    step: int
    result_dir: Path

    @property
    def method_id(self) -> str:
        return f"xssc_lora/{self.variant}/step-{self.step:06d}"

    @property
    def sort_key(self) -> tuple[int, int]:
        return TRAINING_VARIANT_ORDER[self.variant], self.step


@dataclass(frozen=True)
class TrainingMetricStat:
    checkpoint: TrainingCheckpoint
    metric: Metric
    count: int
    mean: float | None
    complete: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 67-case metric means and create one multi-panel DiT ablation plot."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--input-txt",
        type=Path,
        default=None,
        help=(
            "Optional txt containing one result leaf directory per line. "
            "When set, only listed directories are plotted."
        ),
    )
    parser.add_argument("--input-json-allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-cases", type=int, default=67)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Write and plot only metric records complete for all expected cases.",
    )
    return parser.parse_args()


def discover_methods(root: Path) -> list[Method]:
    methods: list[Method] = []
    for model in ("wan_lora", "xssc"):
        model_root = root / model
        if not model_root.is_dir():
            continue
        for config_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
            if config_dir.name.startswith("_"):
                continue
            if config_dir.name == "baseline":
                mode = "baseline"
                block_id = None
            else:
                match = METHOD_PATTERN.fullmatch(config_dir.name)
                if match is None:
                    continue
                mode = match.group(1)
                block_id = int(match.group(2))
            result_dir = config_dir if model == "wan_lora" else config_dir / "results"
            if result_dir.is_dir():
                methods.append(
                    Method(
                        method_id=f"{model}/{config_dir.name}",
                        model=model,
                        mode=mode,
                        block_id=block_id,
                        result_dir=result_dir.resolve(),
                    )
                )
    return sorted(methods, key=lambda method: method.sort_key)


def infer_model(result_dir: Path) -> str:
    path_parts = {part.lower() for part in result_dir.parts}
    if "wan_lora" in path_parts:
        return "wan_lora"
    if "xssc" in path_parts:
        return "xssc"
    if "phyrvg" in path_parts or "physrvg" in path_parts:
        return "physrvg"
    raise ValueError(f"Cannot infer model from result directory: {result_dir}")


def infer_config(result_dir: Path) -> tuple[str, int | None, str]:
    for candidate in (result_dir, *result_dir.parents):
        if candidate.name == "baseline":
            return "baseline", None, candidate.name
        match = METHOD_PATTERN.fullmatch(candidate.name)
        if match is not None:
            return match.group(1), int(match.group(2)), candidate.name
    raise ValueError(f"Cannot infer ablation config from result directory: {result_dir}")


def read_result_dirs_from_txt(path: Path) -> list[Path]:
    result_dirs = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not result_dirs:
        raise ValueError(f"No result directories found in {path}")
    if len(result_dirs) != len(set(result_dirs)):
        raise ValueError(f"Duplicate result directories found in {path}")
    return result_dirs


def infer_training_checkpoint(result_dir: Path) -> TrainingCheckpoint | None:
    for candidate in (result_dir, *result_dir.parents):
        match = TRAINING_CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match is not None:
            variant = TRAINING_VARIANT_ALIASES.get(match.group(1), match.group(1))
            return TrainingCheckpoint(
                variant=variant,
                step=int(match.group(2)),
                result_dir=result_dir,
            )
    return None


def discover_methods_from_txt(path: Path) -> list[Method]:
    result_dirs = read_result_dirs_from_txt(path)

    methods: list[Method] = []
    method_ids: set[str] = set()
    for result_dir in result_dirs:
        if infer_training_checkpoint(result_dir) is not None:
            continue
        model = infer_model(result_dir)
        mode, block_id, config_name = infer_config(result_dir)
        method_id = f"{model}/{config_name}"
        if method_id in method_ids:
            raise ValueError(f"Duplicate method inferred from {path}: {method_id}")
        method_ids.add(method_id)
        methods.append(
            Method(
                method_id=method_id,
                model=model,
                mode=mode,
                block_id=block_id,
                result_dir=result_dir,
            )
        )
    return sorted(methods, key=lambda method: method.sort_key)


def discover_training_checkpoints_from_txt(path: Path) -> list[TrainingCheckpoint]:
    checkpoints = [
        checkpoint
        for result_dir in read_result_dirs_from_txt(path)
        if (checkpoint := infer_training_checkpoint(result_dir)) is not None
    ]
    method_ids = [checkpoint.method_id for checkpoint in checkpoints]
    if len(method_ids) != len(set(method_ids)):
        raise ValueError(f"Duplicate training checkpoint inferred from {path}")
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.sort_key)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_allowlist(path: Path) -> set[Path]:
    paths = {
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return paths


def resolve_input_json(payload: dict[str, Any]) -> Path | None:
    for key in ("input_json", "case_json"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
    return None


def load_allowed_payloads(
    result_dir: Path, allowed_input_jsons: set[Path]
) -> dict[Path, dict[str, Any]]:
    payloads: dict[Path, dict[str, Any]] = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.name in {
            "summary.json",
            "result.json",
            "batch_manifest.json",
            "eval_summary.json",
        } or path.name.startswith("eval_summary_"):
            continue
        payload = load_json(path)
        if payload is None:
            continue
        input_json = resolve_input_json(payload)
        if input_json is not None and input_json in allowed_input_jsons:
            payloads[input_json] = payload
    return payloads


def compute_stats(
    methods: list[Method],
    allowed_input_jsons: set[Path],
    expected_cases: int,
) -> list[MetricStat]:
    stats: list[MetricStat] = []
    for method in methods:
        payloads = load_allowed_payloads(method.result_dir, allowed_input_jsons)
        for metric in METRICS:
            values = [
                value
                for payload in payloads.values()
                if (value := metric.extract(payload)) is not None
            ]
            count = len(values)
            stats.append(
                MetricStat(
                    method=method,
                    metric=metric,
                    count=count,
                    mean=float(np.mean(values)) if values else None,
                    complete=count == expected_cases,
                )
            )
    return stats


def compute_training_stats(
    checkpoints: list[TrainingCheckpoint],
    allowed_input_jsons: set[Path],
    expected_cases: int,
) -> list[TrainingMetricStat]:
    stats: list[TrainingMetricStat] = []
    for checkpoint in checkpoints:
        payloads = load_allowed_payloads(checkpoint.result_dir, allowed_input_jsons)
        for metric in METRICS:
            values = [
                value
                for payload in payloads.values()
                if (value := metric.extract(payload)) is not None
            ]
            count = len(values)
            stats.append(
                TrainingMetricStat(
                    checkpoint=checkpoint,
                    metric=metric,
                    count=count,
                    mean=float(np.mean(values)) if values else None,
                    complete=count == expected_cases,
                )
            )
    return stats


def write_stats_csv(
    path: Path,
    stats: list[MetricStat],
    expected_cases: int,
    complete_only: bool = False,
) -> None:
    fieldnames = (
        "method_id",
        "model",
        "model_label",
        "ablation",
        "ablation_label",
        "layer",
        "metric",
        "direction",
        "score_count",
        "expected_count",
        "complete_67",
        "mean",
        "result_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            if complete_only and not stat.complete:
                continue
            writer.writerow(
                {
                    "method_id": stat.method.method_id,
                    "model": stat.method.model,
                    "model_label": MODEL_LABELS[stat.method.model],
                    "ablation": stat.method.mode,
                    "ablation_label": MODE_LABELS[stat.method.mode],
                    "layer": (
                        "" if stat.method.block_id is None else stat.method.block_id
                    ),
                    "metric": stat.metric.key,
                    "direction": stat.metric.direction,
                    "score_count": stat.count,
                    "expected_count": expected_cases,
                    "complete_67": stat.complete,
                    "mean": "" if stat.mean is None else f"{stat.mean:.8f}",
                    "result_dir": str(stat.method.result_dir),
                }
            )


def write_training_stats_csv(
    path: Path,
    stats: list[TrainingMetricStat],
    expected_cases: int,
    complete_only: bool = False,
) -> None:
    fieldnames = (
        "method_id",
        "variant",
        "variant_label",
        "training_step",
        "metric",
        "direction",
        "score_count",
        "expected_count",
        "complete_67",
        "mean",
        "result_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            if complete_only and not stat.complete:
                continue
            writer.writerow(
                {
                    "method_id": stat.checkpoint.method_id,
                    "variant": stat.checkpoint.variant,
                    "variant_label": TRAINING_VARIANT_LABELS[
                        stat.checkpoint.variant
                    ],
                    "training_step": stat.checkpoint.step,
                    "metric": stat.metric.key,
                    "direction": stat.metric.direction,
                    "score_count": stat.count,
                    "expected_count": expected_cases,
                    "complete_67": stat.complete,
                    "mean": "" if stat.mean is None else f"{stat.mean:.8f}",
                    "result_dir": str(stat.checkpoint.result_dir),
                }
            )


def stat_index(
    stats: list[MetricStat],
) -> dict[tuple[str, str, int | None, str], MetricStat]:
    return {
        (
            stat.method.model,
            stat.method.mode,
            stat.method.block_id,
            stat.metric.key,
        ): stat
        for stat in stats
    }


def complete_value(stat: MetricStat | None) -> float:
    if stat is None or not stat.complete or stat.mean is None:
        return np.nan
    return stat.mean


def plot_metrics(
    output_png: Path,
    output_pdf: Path,
    stats: list[MetricStat],
    expected_cases: int,
    dpi: int,
    model: str,
    metrics: tuple[Metric, ...] = METRICS,
) -> dict[str, dict[str, int]]:
    indexed = stat_index(stats)
    model_methods = [
        method for method in {stat.method for stat in stats} if method.model == model
    ]
    block_ids = sorted(
        {
            method.block_id
            for method in model_methods
            if method.block_id is not None
        }
    )
    plot_modes = sorted(
        {method.mode for method in model_methods if method.mode != "baseline"},
        key=MODE_ORDER.__getitem__,
    )
    available_points = {
        (method.mode, method.block_id)
        for method in model_methods
        if method.block_id is not None
    }
    x_positions = np.arange(len(block_ids) + 1)
    x_labels = ("Baseline",) + tuple(str(block_id) for block_id in block_ids)
    num_columns = 2
    num_rows = math.ceil(len(metrics) / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(19, 4.5 * num_rows),
        constrained_layout=False,
    )
    axes_flat = list(np.atleast_1d(axes).flat)
    completeness: dict[str, dict[str, int]] = {}

    for axis, metric in zip(axes_flat, metrics):
        plotted_ablation_points = 0
        total_ablation_points = 0
        baseline = indexed.get((model, "baseline", None, metric.key))
        baseline_value = complete_value(baseline)
        for mode in plot_modes:
            # Baseline is shown independently and must not connect to layer 0.
            values = [np.nan]
            for block_id in block_ids:
                stat = indexed.get((model, mode, block_id, metric.key))
                if (mode, block_id) in available_points:
                    total_ablation_points += 1
                if stat is not None and stat.complete:
                    plotted_ablation_points += 1
                values.append(complete_value(stat))
            if np.isfinite(values).any():
                style = MODEL_STYLES.get(
                    model, {"linestyle": "-", "marker": "o"}
                )
                axis.plot(
                    x_positions,
                    values,
                    color=MODE_COLORS[mode],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=6,
                    linewidth=2,
                    alpha=0.95,
                    zorder=2,
                )

        if np.isfinite(baseline_value):
            axis.axhline(
                baseline_value,
                color="#202020",
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=1,
            )

        completeness[metric.key] = {
            "complete_ablation_points": plotted_ablation_points,
            "expected_ablation_points": total_ablation_points,
        }
        direction_symbol = "\u2191" if metric.direction == "higher" else "\u2193"
        axis.set_title(
            f"{metric.title} ({direction_symbol})",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        axis.set_xticks(x_positions, x_labels)
        axis.set_xlabel("Layer", fontsize=11)
        axis.set_ylabel("Mean score", fontsize=11)
        axis.grid(axis="both", color="#D9DDDF", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
        if plotted_ablation_points == 0:
            axis.text(
                0.5,
                0.5,
                f"No ablation result has {expected_cases}/{expected_cases} scores",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6A7175",
                fontsize=11,
            )

    for axis in axes_flat[len(metrics) :]:
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODE_COLORS[mode],
            linewidth=3,
            label=MODE_LABELS[mode],
        )
        for mode in plot_modes
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#202020",
            linewidth=2,
            linestyle="--",
            label=f"{MODEL_LABELS[model]} baseline",
        )
    )
    fig.suptitle(
        f"{MODEL_LABELS[model]} DiT Block Ablation Metrics",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=4,
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.014,
        (
            f"Only points with {expected_cases}/{expected_cases} finite case scores are "
            "shown. WMReward uses surprise (lower is better); all other metrics are "
            "higher is better."
        ),
        ha="center",
        fontsize=11,
        color="#4D5559",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.91,
        hspace=0.38,
        wspace=0.24,
    )
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return completeness


def plot_training_metrics(
    output_png: Path,
    output_pdf: Path,
    stats: list[TrainingMetricStat],
    expected_cases: int,
    dpi: int,
    metrics: tuple[Metric, ...],
) -> dict[str, dict[str, int]]:
    indexed = {
        (stat.checkpoint.variant, stat.checkpoint.step, stat.metric.key): stat
        for stat in stats
    }
    checkpoints = {stat.checkpoint for stat in stats}
    variants = sorted(
        {checkpoint.variant for checkpoint in checkpoints},
        key=TRAINING_VARIANT_ORDER.__getitem__,
    )
    steps = sorted({checkpoint.step for checkpoint in checkpoints})
    num_columns = 2
    num_rows = math.ceil(len(metrics) / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(19, 4.5 * num_rows),
        constrained_layout=False,
    )
    axes_flat = list(np.atleast_1d(axes).flat)
    completeness: dict[str, dict[str, int]] = {}

    for axis, metric in zip(axes_flat, metrics):
        complete_points = 0
        expected_points = 0
        for variant in variants:
            variant_steps = sorted(
                checkpoint.step
                for checkpoint in checkpoints
                if checkpoint.variant == variant
            )
            values: list[float] = []
            plotted_steps: list[int] = []
            for step in variant_steps:
                expected_points += 1
                stat = indexed.get((variant, step, metric.key))
                if stat is not None and stat.complete and stat.mean is not None:
                    complete_points += 1
                    plotted_steps.append(step)
                    values.append(stat.mean)
            if values:
                axis.plot(
                    plotted_steps,
                    values,
                    color=TRAINING_VARIANT_COLORS[variant],
                    linestyle=TRAINING_VARIANT_LINESTYLES[variant],
                    marker=TRAINING_VARIANT_MARKERS[variant],
                    markersize=7,
                    linewidth=2,
                    label=TRAINING_VARIANT_LABELS[variant],
                )

        completeness[metric.key] = {
            "complete_points": complete_points,
            "expected_points": expected_points,
        }
        direction_symbol = "\u2191" if metric.direction == "higher" else "\u2193"
        axis.set_title(
            f"{metric.title} ({direction_symbol})",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        axis.set_xlabel("Training step", fontsize=11)
        axis.set_ylabel("Mean score", fontsize=11)
        axis.set_xticks(steps)
        axis.grid(axis="both", color="#D9DDDF", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
        if complete_points == 0:
            axis.text(
                0.5,
                0.5,
                f"No result has {expected_cases}/{expected_cases} scores",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6A7175",
                fontsize=11,
            )

    for axis in axes_flat[len(metrics) :]:
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRAINING_VARIANT_COLORS[variant],
            marker=TRAINING_VARIANT_MARKERS[variant],
            linestyle=TRAINING_VARIANT_LINESTYLES[variant],
            linewidth=2,
            label=TRAINING_VARIANT_LABELS[variant],
        )
        for variant in variants
    ]
    fig.suptitle(
        "Wan+xSSC LoRA Training Checkpoint Metrics",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.014,
        (
            f"Only points with {expected_cases}/{expected_cases} finite case scores are "
            "shown. WMReward uses surprise (lower is better); all other metrics are "
            "higher is better."
        ),
        ha="center",
        fontsize=11,
        color="#4D5559",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.91,
        hspace=0.38,
        wspace=0.24,
    )
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return completeness


def write_plot_index(
    path: Path,
    plots: dict[str, dict[str, str]],
    training_plot: dict[str, str] | None,
) -> None:
    sections: list[str] = []
    if training_plot is not None:
        training_png_path = Path(training_plot["png"])
        training_png = training_png_path.name
        training_png_version = training_png_path.stat().st_mtime_ns
        training_pdf = Path(training_plot["pdf"]).name
        sections.append(
            f"<section><h2>Wan+xSSC training checkpoints</h2>"
            f"<p><a href='{training_pdf}'>PDF</a></p>"
            f"<img src='{training_png}?v={training_png_version}' "
            f"alt='Wan+xSSC training checkpoint metrics'>"
            f"</section>"
        )
    for model, files in plots.items():
        png_path = Path(files["png"])
        png_name = png_path.name
        png_version = png_path.stat().st_mtime_ns
        pdf_name = Path(files["pdf"]).name
        sections.append(
            f"<section><h2>{MODEL_LABELS[model]} block ablations</h2>"
            f"<p><a href='{pdf_name}'>PDF</a></p>"
            f"<img src='{png_name}?v={png_version}' "
            f"alt='{MODEL_LABELS[model]} block ablation metrics'>"
            f"</section>"
        )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Wan metric plots</title><style>"
        "body{font-family:Arial,sans-serif;margin:24px;color:#202428;background:#f4f5f6}"
        "main{max-width:1800px;margin:auto}section{margin:0 0 28px;padding:16px;"
        "background:#fff;border:1px solid #d8dcdf;border-radius:6px}"
        "h1,h2{letter-spacing:0}img{display:block;width:100%;height:auto}"
        "a{color:#155ca2}</style></head><body><main>"
        "<h1>Wan metric plots</h1>"
        + "".join(sections)
        + "</main></body></html>\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    input_txt = (
        args.input_txt.expanduser().resolve() if args.input_txt is not None else None
    )
    allowlist_path = args.input_json_allowlist.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif input_txt is not None:
        output_dir = input_txt.parent / "_metric_plots" / input_txt.stem
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_input_jsons = read_allowlist(allowlist_path)
    if len(allowed_input_jsons) != args.expected_cases:
        raise ValueError(
            f"Allowlist has {len(allowed_input_jsons)} unique cases, "
            f"but --expected-cases={args.expected_cases}"
        )

    methods = (
        discover_methods_from_txt(input_txt)
        if input_txt is not None
        else discover_methods(result_root)
    )
    training_checkpoints = (
        discover_training_checkpoints_from_txt(input_txt)
        if input_txt is not None
        else []
    )
    if not methods and not training_checkpoints:
        source = input_txt if input_txt is not None else result_root
        raise RuntimeError(
            f"No ablation methods or training checkpoints found from {source}"
        )

    stats = compute_stats(methods, allowed_input_jsons, args.expected_cases)
    training_stats = compute_training_stats(
        training_checkpoints,
        allowed_input_jsons,
        args.expected_cases,
    )
    missing_result_dirs = [
        str(method.result_dir) for method in methods if not method.result_dir.is_dir()
    ]
    missing_training_result_dirs = [
        str(checkpoint.result_dir)
        for checkpoint in training_checkpoints
        if not checkpoint.result_dir.is_dir()
    ]
    csv_path = output_dir / "dit_ablation_metric_stats.csv"
    training_csv_path = output_dir / "xssc_lora_training_step_metric_stats.csv"
    manifest_path = output_dir / "dit_ablation_metric_plot_manifest.json"

    write_stats_csv(
        csv_path,
        stats,
        args.expected_cases,
        complete_only=args.complete_only,
    )
    write_training_stats_csv(
        training_csv_path,
        training_stats,
        args.expected_cases,
        complete_only=args.complete_only,
    )
    plotted_metrics = (
        tuple(
            metric
            for metric in METRICS
            if any(
                stat.metric.key == metric.key and stat.complete
                for stat in (*stats, *training_stats)
            )
        )
        if args.complete_only
        else METRICS
    )
    if not plotted_metrics:
        raise RuntimeError("No metric has a complete result point yet")
    model_ids = sorted(
        {method.model for method in methods},
        key=lambda model: {"wan_lora": 0, "xssc": 1, "physrvg": 2}.get(model, 99),
    )
    plots: dict[str, dict[str, str]] = {}
    metric_completeness: dict[str, dict[str, dict[str, int]]] = {}
    for model in model_ids:
        png_path = output_dir / f"dit_ablation_{model}_all_metrics.png"
        pdf_path = output_dir / f"dit_ablation_{model}_all_metrics.pdf"
        metric_completeness[model] = plot_metrics(
            png_path,
            pdf_path,
            stats,
            args.expected_cases,
            args.dpi,
            model,
            plotted_metrics,
        )
        plots[model] = {"png": str(png_path), "pdf": str(pdf_path)}

    training_plot: dict[str, str] | None = None
    training_metric_completeness: dict[str, dict[str, int]] = {}
    if training_checkpoints:
        training_png_path = output_dir / "xssc_lora_training_step_all_metrics.png"
        training_pdf_path = output_dir / "xssc_lora_training_step_all_metrics.pdf"
        training_metric_completeness = plot_training_metrics(
            training_png_path,
            training_pdf_path,
            training_stats,
            args.expected_cases,
            args.dpi,
            plotted_metrics,
        )
        training_plot = {
            "png": str(training_png_path),
            "pdf": str(training_pdf_path),
        }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root),
        "input_txt": None if input_txt is None else str(input_txt),
        "input_json_allowlist": str(allowlist_path),
        "expected_cases": args.expected_cases,
        "num_methods": len(methods),
        "num_training_checkpoints": len(training_checkpoints),
        "missing_result_dirs": missing_result_dirs,
        "missing_training_result_dirs": missing_training_result_dirs,
        "num_metrics": len(plotted_metrics),
        "complete_only": args.complete_only,
        "plotted_metrics": [metric.key for metric in plotted_metrics],
        "stats_csv": str(csv_path),
        "training_stats_csv": str(training_csv_path),
        "plots": plots,
        "training_plot": training_plot,
        "metric_completeness": metric_completeness,
        "training_metric_completeness": training_metric_completeness,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_plot_index(output_dir / "index.html", plots, training_plot)

    print(f"Methods: {len(methods)}")
    print(f"Training checkpoints: {len(training_checkpoints)}")
    print(f"Missing result directories: {len(missing_result_dirs)}")
    print(
        "Missing training result directories: "
        f"{len(missing_training_result_dirs)}"
    )
    print(f"Allowed cases: {len(allowed_input_jsons)}")
    for model in model_ids:
        completeness = metric_completeness[model]
        print(f"[{MODEL_LABELS[model]}]")
        for metric in plotted_metrics:
            progress = completeness[metric.key]
            print(
                f"{metric.key}: "
                f"{progress['complete_ablation_points']}/"
                f"{progress['expected_ablation_points']} complete ablation points"
            )
    print(f"Stats CSV: {csv_path}")
    if training_plot is not None:
        print(f"Training stats CSV: {training_csv_path}")
        print(f"Training PNG: {training_plot['png']}")
        print(f"Training PDF: {training_plot['pdf']}")
    for model in model_ids:
        print(f"{MODEL_LABELS[model]} PNG: {plots[model]['png']}")
        print(f"{MODEL_LABELS[model]} PDF: {plots[model]['pdf']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
