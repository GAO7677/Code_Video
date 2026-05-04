#!/usr/bin/env python3
"""Compute per-sample metrics and cumulative curves for stage0 sample300 outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_validation_vbench as rv


MODEL_SPECS = [
    ("base-ti2v-5b", "output/wan2_2_5B_baseline_TI2V"),
    ("step-008000", "output/wan2.25B_lora_sample300_full49/step-008000"),
    ("step-010000", "output/wan2.25B_lora_sample300_full49/step-010000"),
    ("wan_pure_ti2v_5b", "output/Wan2_2_5B_pure_TI2V"),
    ("vace_ti2v_firstframe", "output/VACE_1_3B_TI2V"),
    ("vace_v2v_ctx01f", "output/VACE_1_3B_V2V/context_01f"),
    ("vace_v2v_ctx02f", "output/VACE_1_3B_V2V/context_02f"),
    ("vace_v2v_ctx04f", "output/VACE_1_3B_V2V/context_04f"),
    ("vace_v2v_ctx08f", "output/VACE_1_3B_V2V/context_08f"),
]

METRIC_SPECS = [
    ("future_psnr", "PSNR", False),
    ("future_ssim", "SSIM", False),
    ("future_lpips", "LPIPS", True),
    ("future_dino", "DINO", False),
]

PALETTE = [
    "#b5532d",
    "#1f6f8b",
    "#3f7d20",
    "#805b10",
    "#7b4697",
    "#c75f7a",
    "#28705e",
    "#4f4f9c",
    "#9c4129",
]

DISPLAY_NAMES = {
    "base-ti2v-5b": "Wan baseline",
    "step-008000": "step-008000",
    "step-010000": "step-010000",
    "wan_pure_ti2v_5b": "Wan pure TI2V",
    "vace_ti2v_firstframe": "VACE TI2V",
    "vace_v2v_ctx01f": "VACE ctx01",
    "vace_v2v_ctx02f": "VACE ctx02",
    "vace_v2v_ctx04f": "VACE ctx04",
    "vace_v2v_ctx08f": "VACE ctx08",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample300 cumulative metric curves for stage0 models.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/model_metrics_sample300_curves"),
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cumulative_mean(values: list[float]) -> list[float]:
    running = 0.0
    output: list[float] = []
    for index, value in enumerate(values, start=1):
        running += float(value)
        output.append(running / index)
    return output


def build_model_payload(
    *,
    benchmark_root: Path,
    runtime_root: Path,
    model_name: str,
    rel_model_dir: str,
    height: int,
    width: int,
    metric_suite: rv.ValidationMetricSuite,
) -> dict[str, Any]:
    generated_dir = benchmark_root / rel_model_dir
    entries = rv.load_entries_for_compare(model_name, generated_dir, runtime_root / model_name)
    metric_payload = rv.compute_future_gt_metrics(
        entries,
        height=height,
        width=width,
        metric_suite=metric_suite,
    )
    per_sample = metric_payload.get("per_sample", [])
    curve_payload: dict[str, Any] = {}
    for metric_name, _, _ in METRIC_SPECS:
        values = [float(item[metric_name]) for item in per_sample if metric_name in item]
        curve_payload[metric_name] = {
            "values": values,
            "cumulative_mean": cumulative_mean(values),
        }
    return {
        "model_name": model_name,
        "display_name": DISPLAY_NAMES.get(model_name, model_name),
        "generated_dir": str(generated_dir),
        "num_samples": len(per_sample),
        "aggregate": metric_payload.get("aggregate", {}),
        "per_sample": per_sample,
        "curves": curve_payload,
    }


def build_curve_png(curve_data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("#f4f1e8")

    model_names = list(curve_data["models"].keys())
    for axis, (metric_name, title, lower_better) in zip(axes, METRIC_SPECS):
        for index, model_name in enumerate(model_names):
            model_block = curve_data["models"][model_name]
            series = model_block.get("curves", {}).get(metric_name, {}).get("cumulative_mean", [])
            if not series:
                continue
            xs = np.arange(1, len(series) + 1)
            axis.plot(
                xs,
                series,
                color=PALETTE[index % len(PALETTE)],
                linewidth=2.0,
                label=model_block.get("display_name", model_name),
            )
        axis.set_title(f"{title} cumulative mean", fontsize=12)
        axis.set_xlabel("Sample index in benchmark list")
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.25)
        badge = "lower better" if lower_better else "higher better"
        axis.text(
            0.98,
            0.02,
            badge,
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#6e675d",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Stage0 sample300 cumulative metrics by model", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runtime_root = benchmark_root / "tools" / "runtime"
    output_root.mkdir(parents=True, exist_ok=True)

    metric_suite = rv.ValidationMetricSuite()
    models: dict[str, Any] = {}
    combined_rows: list[dict[str, Any]] = []

    for model_name, rel_model_dir in MODEL_SPECS:
        model_payload = build_model_payload(
            benchmark_root=benchmark_root,
            runtime_root=runtime_root,
            model_name=model_name,
            rel_model_dir=rel_model_dir,
            height=args.height,
            width=args.width,
            metric_suite=metric_suite,
        )
        models[model_name] = model_payload
        for sample_index, item in enumerate(model_payload.get("per_sample", []), start=1):
            row = {
                "model_name": model_name,
                "display_name": model_payload.get("display_name", model_name),
                "sample_index": sample_index,
                "dataset": item.get("dataset"),
                "sample_id": item.get("sample_id"),
            }
            for metric_name, _, _ in METRIC_SPECS:
                row[metric_name] = item.get(metric_name)
                cumulative_values = model_payload["curves"][metric_name]["cumulative_mean"]
                row[f"{metric_name}_cumulative_mean"] = cumulative_values[sample_index - 1]
            combined_rows.append(row)

    curve_data = {
        "benchmark_root": str(benchmark_root),
        "metric_names": [metric_name for metric_name, _, _ in METRIC_SPECS],
        "models": models,
    }
    write_json(output_root / "sample300_curve_data.json", curve_data)
    write_csv(output_root / "sample300_curve_data.csv", combined_rows)
    build_curve_png(curve_data, output_root / "sample300_cumulative_metrics.png")
    print(output_root / "sample300_curve_data.json")


if __name__ == "__main__":
    main()
