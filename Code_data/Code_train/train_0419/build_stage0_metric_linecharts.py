#!/usr/bin/env python3
"""Build aggregate metric comparison line charts for stage0 benchmark models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WAN_ORDER = ["base-ti2v-5b", "step-008000", "step-010000"]
VACE_ORDER = [
    "vace_ti2v_firstframe",
    "vace_v2v_ctx01f",
    "vace_v2v_ctx02f",
    "vace_v2v_ctx04f",
    "vace_v2v_ctx08f",
]
ALL_ORDER = WAN_ORDER + ["wan_pure_ti2v_5b"] + VACE_ORDER

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

METRICS = [
    ("future_psnr", "PSNR", False),
    ("future_ssim", "SSIM", False),
    ("future_lpips", "LPIPS", True),
    ("future_dino", "DINO", False),
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
    parser = argparse.ArgumentParser(description="Build aggregate metric line charts for stage0 benchmark.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/result/model_metric_linecharts"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_dataset_label(raw_path: str, payload: dict[str, Any]) -> str:
    text = str(raw_path)
    dataset = str(payload.get("dataset") or "").strip()
    if "kubric_tfds_movi-d" in text or dataset == "MOVI-D":
        return "kubric_tfds_movi-d"
    if "version_1_genesis_rigid_data_all_cases" in text or dataset == "GenesisRigid" or "genesis" in text.lower():
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


def build_overall_chart(models: dict[str, dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("#f4f1e8")

    x_labels = [DISPLAY_NAMES.get(name, name) for name in ALL_ORDER if name in models]
    x_positions = list(range(len(x_labels)))

    for axis, (metric_key, title, lower_better) in zip(axes, METRICS):
        y_values = []
        valid_positions = []
        for idx, model_name in enumerate(ALL_ORDER):
            if model_name not in models:
                continue
            value = models[model_name].get("aggregate", {}).get(metric_key)
            y_values.append(value)
            valid_positions.append(idx if len(x_labels) == len(ALL_ORDER) else len(valid_positions))
        axis.plot(valid_positions, y_values, marker="o", linewidth=2.2, color="#b5532d")
        for xpos, value in zip(valid_positions, y_values):
            if value is not None:
                axis.text(xpos, value, f"{value:.3f}", fontsize=8, ha="center", va="bottom")
        axis.set_xticks(valid_positions)
        axis.set_xticklabels(x_labels, rotation=20, ha="right")
        axis.set_title(f"Overall {title}", fontsize=13)
        axis.grid(True, alpha=0.25)
        badge = "lower better" if lower_better else "higher better"
        axis.text(0.98, 0.02, badge, transform=axis.transAxes, ha="right", va="bottom", fontsize=9, color="#6e675d")

    fig.suptitle("Stage0 Benchmark Overall Metric Comparison", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_dataset_chart(models: dict[str, dict[str, Any]], output_path: Path, metric_key: str, title: str, lower_better: bool) -> None:
    fig, axis = plt.subplots(figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor("#f4f1e8")
    palette = ["#b5532d", "#2a6f8f", "#3b7c32", "#9a6230", "#7b4fa3", "#ce5d7a", "#2a7a68", "#4a5db0", "#8e3d2a"]

    categories = [dataset for dataset in DATASET_ORDER if any(dataset in models[m].get("per_dataset", {}) for m in models)]
    x_positions = list(range(len(categories)))

    for idx, model_name in enumerate(ALL_ORDER):
        if model_name not in models:
            continue
        values = []
        for dataset in categories:
            value = models[model_name].get("per_dataset", {}).get(dataset, {}).get("aggregate", {}).get(metric_key)
            values.append(value)
        axis.plot(
            x_positions,
            values,
            marker="o",
            linewidth=2.0,
            label=DISPLAY_NAMES.get(model_name, model_name),
            color=palette[idx % len(palette)],
        )

    axis.set_xticks(x_positions)
    axis.set_xticklabels([DATASET_LABELS.get(name, name) for name in categories], rotation=15, ha="right")
    axis.set_title(f"{title} by Dataset", fontsize=14)
    axis.grid(True, alpha=0.25)
    badge = "lower better" if lower_better else "higher better"
    axis.text(0.98, 0.02, badge, transform=axis.transAxes, ha="right", va="bottom", fontsize=9, color="#6e675d")
    axis.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_index(output_root: Path, testset_compositions: list[dict[str, Any]]) -> None:
    images = [("overall_metrics.png", "Overall metrics")]
    cards = []
    for filename, title in images:
        cards.append(
            "<section class='card'>"
            f"<h2>{title}</h2>"
            f"<img src='{filename}' alt='{title}'>"
            "</section>"
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
      width: min(1600px, calc(100vw - 24px));
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
    .hero p {{ margin: 0; color: #6e675d; font-size: 13px; line-height: 1.45; }}
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
      <p>基于现有评测汇总指标直接绘制。主图只保留 300 个样本整体平均指标的模型对比折线图；下方同时列出 stage0_V2V 下各测试集的组成。</p>
    </section>
    <section class="grid">
      {''.join(cards)}
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

    wan_json = benchmark_root / "result" / "model_metrics_wan_v2v_fair" / "metrics_by_model.json"
    vace_json = benchmark_root / "result" / "model_metrics_vace_family_native" / "metrics_by_model.json"
    models = {}
    if wan_json.is_file():
        models.update(load_json(wan_json))
    if vace_json.is_file():
        models.update(load_json(vace_json))
    testset_compositions = collect_testset_compositions(benchmark_root / "tools" / "meta")

    build_overall_chart(models, output_root / "overall_metrics.png")
    for metric_key, title, lower_better in METRICS:
        build_dataset_chart(models, output_root / f"dataset_{metric_key.split('_', 1)[1]}.png", metric_key, title, lower_better)
    write_index(output_root, testset_compositions)
    write_json(output_root / "testset_compositions.json", testset_compositions)
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
