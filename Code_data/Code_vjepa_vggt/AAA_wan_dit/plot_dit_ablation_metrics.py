#!/usr/bin/env python3
"""Summarize and plot the 67-case Wan DiT ablation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from build_v2v_wan_case_gallery import MODEL_LABELS, MODE_LABELS, Method, discover_methods


DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan")
DEFAULT_ALLOWLIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "_metric_plots"
BLOCK_IDS = (0, 5, 11, 17, 19, 29)
PLOT_MODES = ("whole_block", "self_attn_zero", "object_cross_attn")
MODEL_STYLES = {
    "wan_lora": {"linestyle": "--", "marker": "o"},
    "xssc": {"linestyle": "-", "marker": "s"},
}
MODE_COLORS = {
    "whole_block": "#E41A1C",
    "self_attn_zero": "#2166D1",
    "object_cross_attn": "#009E73",
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
        "videophy2",
        "VideoPhy2",
        "higher",
        nested_score("videophy2", "score"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 67-case metric means and create one multi-panel DiT ablation plot."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--input-json-allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-cases", type=int, default=67)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


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


def write_stats_csv(path: Path, stats: list[MetricStat], expected_cases: int) -> None:
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
) -> dict[str, dict[str, int]]:
    indexed = stat_index(stats)
    x_positions = np.arange(len(BLOCK_IDS) + 1)
    x_labels = ("Baseline",) + tuple(str(block_id) for block_id in BLOCK_IDS)
    fig, axes = plt.subplots(7, 2, figsize=(19, 32), constrained_layout=False)
    axes_flat = list(axes.flat)
    completeness: dict[str, dict[str, int]] = {}

    for axis, metric in zip(axes_flat, METRICS):
        plotted_ablation_points = 0
        total_ablation_points = 0
        baseline = indexed.get((model, "baseline", None, metric.key))
        baseline_value = complete_value(baseline)
        for mode in PLOT_MODES:
            if model == "wan_lora" and mode == "object_cross_attn":
                continue
            # Baseline is shown independently and must not connect to layer 0.
            values = [np.nan]
            for block_id in BLOCK_IDS:
                stat = indexed.get((model, mode, block_id, metric.key))
                total_ablation_points += 1
                if stat is not None and stat.complete:
                    plotted_ablation_points += 1
                values.append(complete_value(stat))
            if np.isfinite(values).any():
                style = MODEL_STYLES[model]
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
            axis.scatter(
                [0],
                [baseline_value],
                marker="*",
                s=260,
                facecolor="#FFFFFF" if model == "wan_lora" else "#FFD166",
                edgecolor="#202020",
                linewidth=1.6,
                zorder=5,
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

    for axis in axes_flat[len(METRICS) :]:
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODE_COLORS["whole_block"],
            linewidth=3,
            label="Whole block bypass",
        ),
        Line2D(
            [0],
            [0],
            color=MODE_COLORS["self_attn_zero"],
            linewidth=3,
            label="Self-attention output = 0",
        ),
        Line2D(
            [0],
            [0],
            color=MODE_COLORS["object_cross_attn"],
            linewidth=3,
            label="Object cross-attention output = 0",
        ),
        Line2D(
            [0],
            [0],
            color="#202020",
            marker="*",
            markerfacecolor="#FFFFFF" if model == "wan_lora" else "#FFD166",
            markersize=14,
            linestyle="None",
            label=f"{MODEL_LABELS[model]} baseline",
        ),
    ]
    if model == "wan_lora":
        legend_handles = [
            handle
            for handle in legend_handles
            if handle.get_label() != "Object cross-attention output = 0"
        ]
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


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    allowlist_path = args.input_json_allowlist.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_input_jsons = read_allowlist(allowlist_path)
    if len(allowed_input_jsons) != args.expected_cases:
        raise ValueError(
            f"Allowlist has {len(allowed_input_jsons)} unique cases, "
            f"but --expected-cases={args.expected_cases}"
        )

    methods = discover_methods(result_root)
    if not methods:
        raise RuntimeError(f"No ablation methods found under {result_root}")

    stats = compute_stats(methods, allowed_input_jsons, args.expected_cases)
    csv_path = output_dir / "dit_ablation_metric_stats.csv"
    lora_png_path = output_dir / "dit_ablation_wan_lora_all_metrics.png"
    lora_pdf_path = output_dir / "dit_ablation_wan_lora_all_metrics.pdf"
    xssc_png_path = output_dir / "dit_ablation_xssc_all_metrics.png"
    xssc_pdf_path = output_dir / "dit_ablation_xssc_all_metrics.pdf"
    manifest_path = output_dir / "dit_ablation_metric_plot_manifest.json"

    write_stats_csv(csv_path, stats, args.expected_cases)
    lora_completeness = plot_metrics(
        lora_png_path,
        lora_pdf_path,
        stats,
        args.expected_cases,
        args.dpi,
        "wan_lora",
    )
    xssc_completeness = plot_metrics(
        xssc_png_path,
        xssc_pdf_path,
        stats,
        args.expected_cases,
        args.dpi,
        "xssc",
    )
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root),
        "input_json_allowlist": str(allowlist_path),
        "expected_cases": args.expected_cases,
        "num_methods": len(methods),
        "num_metrics": len(METRICS),
        "stats_csv": str(csv_path),
        "plots": {
            "wan_lora": {
                "png": str(lora_png_path),
                "pdf": str(lora_pdf_path),
            },
            "xssc": {
                "png": str(xssc_png_path),
                "pdf": str(xssc_pdf_path),
            },
        },
        "metric_completeness": {
            "wan_lora": lora_completeness,
            "xssc": xssc_completeness,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Methods: {len(methods)}")
    print(f"Allowed cases: {len(allowed_input_jsons)}")
    for model, completeness in (
        ("wan_lora", lora_completeness),
        ("xssc", xssc_completeness),
    ):
        print(f"[{MODEL_LABELS[model]}]")
        for metric in METRICS:
            progress = completeness[metric.key]
            print(
                f"{metric.key}: "
                f"{progress['complete_ablation_points']}/"
                f"{progress['expected_ablation_points']} complete ablation points"
            )
    print(f"Stats CSV: {csv_path}")
    print(f"Wan+LoRA PNG: {lora_png_path}")
    print(f"Wan+LoRA PDF: {lora_pdf_path}")
    print(f"Wan+xSSC PNG: {xssc_png_path}")
    print(f"Wan+xSSC PDF: {xssc_pdf_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
