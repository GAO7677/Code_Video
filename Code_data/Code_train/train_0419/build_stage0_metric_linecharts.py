#!/usr/bin/env python3
"""Build aggregate metric line charts for stage0 benchmark models from latest summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_SPECS = [
    ("base-ti2v-5b", "Wan baseline", "wan2_2_5B_baseline_TI2V"),
    ("step-008000", "step-008000", "wan2.25B_lora_sample300_full49/step-008000"),
    ("step-010000", "step-010000", "wan2.25B_lora_sample300_full49/step-010000"),
    ("wan_pure_ti2v_5b", "Wan pure TI2V", "Wan2_2_5B_pure_TI2V"),
    ("vace_ti2v_firstframe", "VACE TI2V", "VACE_1_3B_TI2V"),
    ("vace_v2v_ctx01f", "VACE ctx01", "VACE_1_3B_V2V/context_01f"),
    ("vace_v2v_ctx02f", "VACE ctx02", "VACE_1_3B_V2V/context_02f"),
    ("vace_v2v_ctx04f", "VACE ctx04", "VACE_1_3B_V2V/context_04f"),
    ("vace_v2v_ctx08f", "VACE ctx08", "VACE_1_3B_V2V/context_08f"),
]

MODEL_OUTPUT_DIRS = {name: subdir for name, _, subdir in MODEL_SPECS}
DISPLAY_NAMES = {name: label for name, label, _ in MODEL_SPECS}
MODEL_ORDER = [name for name, _, _ in MODEL_SPECS]
DATASET_COLORS = {
    "kubric_tfds_movi-d": "#b5532d",
    "version_1_genesis_rigid_data_all_cases": "#2a6f8f",
    "physics-iq-benchmark": "#3b7c32",
    "vLAR-PhysInOne": "#7b4fa3",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": "#9a6230",
}

FUTURE_METRICS = [
    ("future_psnr", "PSNR", False),
    ("future_ssim", "SSIM", False),
    ("future_lpips", "LPIPS", True),
    ("future_dino", "DINO", False),
]

VBENCH_METRICS = [
    ("subject_consistency", "Subject Consistency", False),
    ("background_consistency", "Background Consistency", False),
    ("motion_smoothness", "Motion Smoothness", False),
    ("temporal_flickering", "Temporal Flickering", False),
    ("dynamic_degree", "Dynamic Degree", False),
    ("imaging_quality", "Imaging Quality", False),
    ("aesthetic_quality", "Aesthetic Quality", False),
    ("overall_consistency", "Overall Consistency", False),
    ("temporal_style", "Temporal Style", False),
]

DATASET_ORDER = [
    "kubric_tfds_movi-d",
    "version_1_genesis_rigid_data_all_cases",
    "physics-iq-benchmark",
    "vLAR-PhysInOne",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps",
]

DATASET_LABELS = {
    "kubric_tfds_movi-d": "MOVI-D",
    "version_1_genesis_rigid_data_all_cases": "GenesisRigid",
    "physics-iq-benchmark": "Physics-IQ",
    "vLAR-PhysInOne": "vLAR",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": "OpenVidHD",
}

TESTSET_SPECS = [
    ("benchmark_meta_json_paths_full_sample300.txt", "sample300_full"),
    ("benchmark_meta_json_paths_full_sample300_genesis56.txt", "sample300_genesis56"),
    ("common_case_meta_json_paths.txt", "common_case"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stage0 aggregate metric line charts.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/model_metric_linecharts_latest"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def normalize_dataset_label(raw_path: str, payload: dict[str, Any]) -> str:
    text = str(raw_path)
    dataset = str(payload.get("dataset") or "").strip()
    lowered = text.lower()
    if "kubric_tfds_movi-d" in text or dataset == "MOVI-D":
        return "kubric_tfds_movi-d"
    if "version_1_genesis_rigid_data_all_cases" in text or dataset == "GenesisRigid" or "genesis" in lowered:
        return "version_1_genesis_rigid_data_all_cases"
    if "physics-iq-benchmark" in text:
        return "physics-iq-benchmark"
    if "vLAR-PhysInOne" in text:
        return "vLAR-PhysInOne"
    if "mvp-lab-OpenVidHD-0.4M-720p-48fps" in text:
        return "mvp-lab-OpenVidHD-0.4M-720p-48fps"
    return dataset or "(unknown)"


def collect_testset_compositions(meta_root: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for filename, testset_name in TESTSET_SPECS:
        meta_path = meta_root / filename
        if not meta_path.is_file():
            continue
        counts: dict[str, int] = {}
        paths = [line.strip() for line in meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for raw_path in paths:
            try:
                payload = load_json(Path(raw_path))
            except Exception:
                payload = {}
            dataset = normalize_dataset_label(raw_path, payload)
            counts[dataset] = counts.get(dataset, 0) + 1
        items.append(
            {
                "testset_name": testset_name,
                "meta_list_path": str(meta_path),
                "num_cases": len(paths),
                "composition": counts,
            }
        )
    return items


def load_future_summaries(result_root: Path) -> dict[str, dict[str, Any]]:
    del result_root
    summaries: dict[str, dict[str, Any]] = {}
    return summaries


def load_vbench_summaries(result_root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for dirname in ["model_metrics_vbench_short_gpu3", "model_metrics_vbench_short_gpu5"]:
        summary_path = result_root / dirname / "metrics_by_model.json"
        if not summary_path.is_file():
            continue
        payload = load_json(summary_path)
        for model_name, block in payload.items():
            if isinstance(block, dict):
                summaries[model_name] = block
    return summaries


def load_sidecar_counts(output_root: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for model_name, _, subdir in MODEL_SPECS:
        total = 0
        future = 0
        vbench = 0
        model_dir = output_root / subdir
        if model_dir.is_dir():
            for json_path in sorted(model_dir.glob("*.json")):
                total += 1
                try:
                    payload = load_json(json_path)
                except Exception:
                    continue
                if "future_metrics" in payload:
                    future += 1
                if "vbench_metrics" in payload:
                    vbench += 1
        counts[model_name] = {
            "total_sidecars": total,
            "future_sidecars": future,
            "vbench_sidecars": vbench,
        }
    return counts


def load_future_summaries_from_sidecars(output_root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for model_name, _, subdir in MODEL_SPECS:
        model_dir = output_root / subdir
        per_sample: list[dict[str, Any]] = []
        per_dataset_samples: dict[str, list[dict[str, Any]]] = {}
        eval_height = None
        eval_width = None
        if model_dir.is_dir():
            for json_path in sorted(model_dir.glob("*.json")):
                try:
                    payload = load_json(json_path)
                except Exception:
                    continue
                future_metrics = payload.get("future_metrics")
                if not isinstance(future_metrics, dict):
                    continue
                resolution = future_metrics.get("evaluation_resolution")
                if isinstance(resolution, dict):
                    if eval_height is None:
                        eval_height = resolution.get("height")
                    if eval_width is None:
                        eval_width = resolution.get("width")
                per_sample.append(future_metrics)
                dataset_name = str(payload.get("dataset") or "unknown")
                per_dataset_samples.setdefault(dataset_name, []).append(future_metrics)

        aggregate: dict[str, float] = {}
        for metric_key, _, _ in FUTURE_METRICS:
            values = [
                float(item[metric_key])
                for item in per_sample
                if isinstance(item.get(metric_key), (int, float))
            ]
            if values:
                aggregate[metric_key] = sum(values) / len(values)
        pair_counts = [
            float(item["future_pair_count"])
            for item in per_sample
            if isinstance(item.get("future_pair_count"), (int, float))
        ]
        if pair_counts:
            aggregate["future_pair_count"] = sum(pair_counts) / len(pair_counts)

        per_dataset: dict[str, Any] = {}
        for dataset_name, dataset_items in per_dataset_samples.items():
            dataset_aggregate: dict[str, float] = {}
            for metric_key, _, _ in FUTURE_METRICS:
                values = [
                    float(item[metric_key])
                    for item in dataset_items
                    if isinstance(item.get(metric_key), (int, float))
                ]
                if values:
                    dataset_aggregate[metric_key] = sum(values) / len(values)
            dataset_pair_counts = [
                float(item["future_pair_count"])
                for item in dataset_items
                if isinstance(item.get("future_pair_count"), (int, float))
            ]
            if dataset_pair_counts:
                dataset_aggregate["future_pair_count"] = sum(dataset_pair_counts) / len(dataset_pair_counts)
            per_dataset[dataset_name] = {
                "num_per_sample_metrics": len(dataset_items),
                "aggregate": dataset_aggregate,
            }

        summaries[model_name] = {
            "model_name": model_name,
            "evaluation_height": eval_height,
            "evaluation_width": eval_width,
            "num_per_sample_metrics": len(per_sample),
            "aggregate": aggregate,
            "per_dataset": per_dataset,
        }
    return summaries


def build_combined_summary(
    *,
    benchmark_root: Path,
) -> dict[str, dict[str, Any]]:
    result_root = benchmark_root / "result"
    output_root = benchmark_root / "output"
    future_summaries = load_future_summaries_from_sidecars(output_root)
    vbench_summaries = load_vbench_summaries(result_root)
    sidecar_counts = load_sidecar_counts(output_root)

    combined: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_ORDER:
        future_block = future_summaries.get(model_name, {})
        vbench_block = vbench_summaries.get(model_name, {})
        aggregate_future = future_block.get("aggregate", {}) if isinstance(future_block, dict) else {}
        aggregate_vbench = vbench_block.get("vbench_short_metrics", {}) if isinstance(vbench_block, dict) else {}
        combined[model_name] = {
            "model_name": model_name,
            "display_name": DISPLAY_NAMES.get(model_name, model_name),
            "output_subdir": MODEL_OUTPUT_DIRS.get(model_name, ""),
            "future_metrics": aggregate_future,
            "vbench_metrics": aggregate_vbench,
            "future_num_samples": future_block.get("num_per_sample_metrics"),
            "vbench_num_samples": vbench_block.get("num_samples"),
            "future_eval_resolution": {
                "height": future_block.get("evaluation_height"),
                "width": future_block.get("evaluation_width"),
            },
            "future_per_dataset": future_block.get("per_dataset", {}),
            "vbench_completed_dimensions": vbench_block.get("completed_dimensions", []),
            "sidecar_counts": sidecar_counts.get(model_name, {}),
        }
    return combined


def plot_metric_series(
    *,
    model_summary: dict[str, dict[str, Any]],
    metric_specs: list[tuple[str, str, bool]],
    value_key: str,
    count_key: str,
    title_prefix: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(20, 14), dpi=180)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("#f4f1e8")
    palette = ["#b5532d", "#2a6f8f", "#3b7c32", "#9a6230", "#7b4fa3", "#ce5d7a", "#2a7a68", "#4a5db0", "#8e3d2a"]

    x_labels = [DISPLAY_NAMES.get(name, name) for name in MODEL_ORDER]
    x_positions = list(range(len(MODEL_ORDER)))

    for idx, (metric_key, metric_title, lower_better) in enumerate(metric_specs):
        axis = axes[idx]
        y_values = []
        for model_name in MODEL_ORDER:
            block = model_summary.get(model_name, {})
            metric_block = block.get(value_key, {})
            value = metric_block.get(metric_key) if isinstance(metric_block, dict) else None
            y_values.append(value)
        axis.plot(x_positions, y_values, marker="o", linewidth=2.2, color=palette[idx % len(palette)])
        for xpos, model_name, value in zip(x_positions, MODEL_ORDER, y_values):
            if value is None:
                axis.text(xpos, 0.02, "missing", fontsize=8, ha="center", va="bottom", transform=axis.get_xaxis_transform())
                continue
            count = model_summary.get(model_name, {}).get(count_key)
            if isinstance(count, (int, float)):
                axis.text(xpos, value, f"{value:.4f}\n(n={int(count)})", fontsize=8, ha="center", va="bottom")
            else:
                axis.text(xpos, value, f"{value:.4f}", fontsize=8, ha="center", va="bottom")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(x_labels, rotation=20, ha="right")
        axis.set_title(metric_title, fontsize=13)
        axis.grid(True, alpha=0.25)
        badge = "lower better" if lower_better else "higher better"
        axis.text(0.98, 0.02, badge, transform=axis.transAxes, ha="right", va="bottom", fontsize=9, color="#6e675d")

    for axis in axes[len(metric_specs) :]:
        axis.axis("off")

    fig.suptitle(title_prefix, fontsize=18, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_future_dataset_panel(
    *,
    model_summary: dict[str, dict[str, Any]],
    dataset_name: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("#f4f1e8")
    x_labels = [DISPLAY_NAMES.get(name, name) for name in MODEL_ORDER]
    x_positions = list(range(len(MODEL_ORDER)))
    dataset_label = DATASET_LABELS.get(dataset_name, dataset_name)
    color = DATASET_COLORS.get(dataset_name, "#b5532d")

    for axis, (metric_key, metric_title, lower_better) in zip(axes, FUTURE_METRICS):
        y_values = []
        for model_name in MODEL_ORDER:
            dataset_block = model_summary.get(model_name, {}).get("future_per_dataset", {}).get(dataset_name, {})
            aggregate = dataset_block.get("aggregate", {}) if isinstance(dataset_block, dict) else {}
            y_values.append(aggregate.get(metric_key))
        axis.plot(x_positions, y_values, marker="o", linewidth=2.2, color=color)
        for xpos, model_name, value in zip(x_positions, MODEL_ORDER, y_values):
            dataset_block = model_summary.get(model_name, {}).get("future_per_dataset", {}).get(dataset_name, {})
            count = dataset_block.get("num_per_sample_metrics") if isinstance(dataset_block, dict) else None
            if value is None:
                axis.text(xpos, 0.02, "missing", fontsize=8, ha="center", va="bottom", transform=axis.get_xaxis_transform())
                continue
            if isinstance(count, (int, float)):
                axis.text(xpos, value, f"{value:.4f}\n(n={int(count)})", fontsize=8, ha="center", va="bottom")
            else:
                axis.text(xpos, value, f"{value:.4f}", fontsize=8, ha="center", va="bottom")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(x_labels, rotation=20, ha="right")
        axis.set_title(metric_title, fontsize=13)
        axis.grid(True, alpha=0.25)
        badge = "lower better" if lower_better else "higher better"
        axis.text(0.98, 0.02, badge, transform=axis.transAxes, ha="right", va="bottom", fontsize=9, color="#6e675d")

    fig.suptitle(f"{dataset_label} Future Metrics by Model", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_summary_rows(model_summary: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        block = model_summary[model_name]
        future_metrics = block.get("future_metrics", {})
        vbench_metrics = block.get("vbench_metrics", {})
        sidecar_counts = block.get("sidecar_counts", {})
        row = {
            "model_name": model_name,
            "display_name": block.get("display_name"),
            "future_num_samples": block.get("future_num_samples"),
            "future_sidecars": sidecar_counts.get("future_sidecars"),
            "future_total_sidecars": sidecar_counts.get("total_sidecars"),
            "future_psnr": future_metrics.get("future_psnr"),
            "future_ssim": future_metrics.get("future_ssim"),
            "future_lpips": future_metrics.get("future_lpips"),
            "future_dino": future_metrics.get("future_dino"),
            "future_pair_count": future_metrics.get("future_pair_count"),
            "future_eval_height": block.get("future_eval_resolution", {}).get("height"),
            "future_eval_width": block.get("future_eval_resolution", {}).get("width"),
            "vbench_num_samples": block.get("vbench_num_samples"),
            "vbench_sidecars": sidecar_counts.get("vbench_sidecars"),
            "subject_consistency": vbench_metrics.get("subject_consistency"),
            "background_consistency": vbench_metrics.get("background_consistency"),
            "motion_smoothness": vbench_metrics.get("motion_smoothness"),
            "temporal_flickering": vbench_metrics.get("temporal_flickering"),
            "dynamic_degree": vbench_metrics.get("dynamic_degree"),
            "imaging_quality": vbench_metrics.get("imaging_quality"),
            "aesthetic_quality": vbench_metrics.get("aesthetic_quality"),
            "overall_consistency": vbench_metrics.get("overall_consistency"),
            "temporal_style": vbench_metrics.get("temporal_style"),
        }
        rows.append(row)
    return rows


def write_index(
    *,
    output_root: Path,
    model_summary: dict[str, dict[str, Any]],
    testset_compositions: list[dict[str, Any]],
) -> None:
    summary_rows = build_summary_rows(model_summary)
    model_rows_html = []
    for row in summary_rows:
        future_psnr = "" if row["future_psnr"] is None else f"{row['future_psnr']:.4f}"
        future_ssim = "" if row["future_ssim"] is None else f"{row['future_ssim']:.4f}"
        future_lpips = "" if row["future_lpips"] is None else f"{row['future_lpips']:.4f}"
        future_dino = "" if row["future_dino"] is None else f"{row['future_dino']:.4f}"
        subject_consistency = "" if row["subject_consistency"] is None else f"{row['subject_consistency']:.4f}"
        motion_smoothness = "" if row["motion_smoothness"] is None else f"{row['motion_smoothness']:.4f}"
        imaging_quality = "" if row["imaging_quality"] is None else f"{row['imaging_quality']:.4f}"
        model_rows_html.append(
            "<tr>"
            f"<td>{row['display_name']}</td>"
            f"<td>{row['future_num_samples']}/{row['future_total_sidecars']}</td>"
            f"<td>{future_psnr}</td>"
            f"<td>{future_ssim}</td>"
            f"<td>{future_lpips}</td>"
            f"<td>{future_dino}</td>"
            f"<td>{row['vbench_num_samples']}</td>"
            f"<td>{subject_consistency}</td>"
            f"<td>{motion_smoothness}</td>"
            f"<td>{imaging_quality}</td>"
            "</tr>"
        )

    composition_cards = []
    for item in testset_compositions:
        rows = []
        for dataset_name, count in sorted(item.get("composition", {}).items()):
            label = DATASET_LABELS.get(dataset_name, dataset_name)
            rows.append(f"<tr><td>{label}</td><td>{count}</td></tr>")
        composition_cards.append(
            "<section class='card'>"
            f"<h2>{item['testset_name']}</h2>"
            f"<p class='meta'>{item['num_cases']} cases<br>{item['meta_list_path']}</p>"
            "<table><thead><tr><th>dataset</th><th>count</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "</section>"
        )

    dataset_chart_cards = []
    for dataset_name in DATASET_ORDER:
        filename = f"future_{dataset_name.replace('/', '_')}.png"
        if not (output_root / filename).exists():
            continue
        dataset_chart_cards.append(
            "<section class='card'>"
            f"<h2>{DATASET_LABELS.get(dataset_name, dataset_name)}</h2>"
            "<p class='meta'>该测试数据集上，各模型 future 指标的单独折线图。横轴是模型，纵轴是指标值。</p>"
            f"<img src='{filename}' alt='{dataset_name} future metrics'>"
            "</section>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage0 Metric Line Charts</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background: #f4f1e8;
      color: #1e1b16;
    }}
    .shell {{
      width: min(1800px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 12px 0 24px;
    }}
    .hero {{
      padding: 12px 16px;
      background: #fffdf8;
      border: 1px solid #d8cfbf;
      border-radius: 12px;
      margin-bottom: 12px;
    }}
    .hero h1 {{ margin: 0 0 6px; font-size: 24px; }}
    .hero p {{ margin: 0 0 4px; color: #6e675d; font-size: 13px; line-height: 1.45; }}
    .grid {{
      display: grid;
      gap: 12px;
    }}
    .card {{
      padding: 10px 12px;
      background: #fffdf8;
      border: 1px solid #d8cfbf;
      border-radius: 12px;
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .card p.meta {{
      margin: 0 0 8px;
      color: #6e675d;
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
    }}
    .card img {{
      display: block;
      width: 100%;
      border-radius: 8px;
      background: #f8f4eb;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 5px 6px;
      border-bottom: 1px solid #d8cfbf;
      text-align: left;
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Stage0 Metric Line Charts</h1>
      <p>future_metrics 折线图基于同名 sidecar / per-sample 汇总的 300 样本平均值；图中每个点的 n 表示该模型当前实际完成 future 指标的样本数。</p>
      <p>vbench_metrics 折线图基于 GPU3/GPU5 汇总出的模型级 `vbench_short_metrics`，使用当前已有完整 300 样本结果。</p>
      <p>当前 9 个模型的 `future_metrics` 和 `vbench_metrics` 都已完成 300/300 回填；future 指标直接从最新 sidecar 聚合，避免旧 summary 文件滞后。</p>
    </section>
    <section class="grid">
      <section class="card">
        <h2>Future Metrics</h2>
        <img src="future_metrics.png" alt="future metrics">
      </section>
      <section class="card">
        <h2>VBench Short Metrics</h2>
        <img src="vbench_metrics.png" alt="vbench metrics">
      </section>
      {''.join(dataset_chart_cards)}
      <section class="card">
        <h2>Model Summary</h2>
        <table>
          <thead>
            <tr>
              <th>model</th>
              <th>future n</th>
              <th>PSNR</th>
              <th>SSIM</th>
              <th>LPIPS</th>
              <th>DINO</th>
              <th>vbench n</th>
              <th>subject</th>
              <th>motion</th>
              <th>imaging</th>
            </tr>
          </thead>
          <tbody>{''.join(model_rows_html)}</tbody>
        </table>
      </section>
      {''.join(composition_cards)}
    </section>
  </div>
</body>
</html>
"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ensure_dir(output_root)

    model_summary = build_combined_summary(benchmark_root=benchmark_root)
    testset_compositions = collect_testset_compositions(benchmark_root / "tools" / "meta")

    plot_metric_series(
        model_summary=model_summary,
        metric_specs=FUTURE_METRICS,
        value_key="future_metrics",
        count_key="future_num_samples",
        title_prefix="Stage0 Benchmark Future Metrics on 300 Samples",
        output_path=output_root / "future_metrics.png",
    )
    plot_metric_series(
        model_summary=model_summary,
        metric_specs=VBENCH_METRICS,
        value_key="vbench_metrics",
        count_key="vbench_num_samples",
        title_prefix="Stage0 Benchmark VBench Short Metrics on 300 Samples",
        output_path=output_root / "vbench_metrics.png",
    )
    for dataset_name in DATASET_ORDER:
        plot_future_dataset_panel(
            model_summary=model_summary,
            dataset_name=dataset_name,
            output_path=output_root / f"future_{dataset_name.replace('/', '_')}.png",
        )

    summary_rows = build_summary_rows(model_summary)
    write_json(output_root / "metrics_summary.json", model_summary)
    write_csv(
        output_root / "metrics_summary.csv",
        summary_rows,
        fieldnames=list(summary_rows[0].keys()) if summary_rows else [],
    )
    write_json(output_root / "testset_compositions.json", testset_compositions)
    write_index(output_root=output_root, model_summary=model_summary, testset_compositions=testset_compositions)
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
