#!/usr/bin/env python3
"""Build aggregate metric line charts for stage0 benchmark models from latest summaries."""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
from pathlib import Path
from typing import Any

import build_stage0_compact_selected_portal as compact_portal
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
    ("vace_v2v_ctx08f_nullcaption", "VACE ctx08_nullcaption", "VACE_1_3B_V2V_nullcaption/context_08f"),
    (
        "vace_v2v_fullctx_fullvideo_nullcaption",
        "VACE fullctx_fullvideo_nullcaption",
        "VACE_1_3B_V2V_nullcaption/context_fullctx_fullvideo",
    ),
]

MODEL_OUTPUT_DIRS = {name: subdir for name, _, subdir in MODEL_SPECS}
DISPLAY_NAMES = {name: label for name, label, _ in MODEL_SPECS}
MODEL_ORDER = [name for name, _, _ in MODEL_SPECS]

SAMPLE300_PRIMARY_MODEL_SUBDIR = "wan2_2_5B_baseline_TI2V"
SHOWCASE_CASES_PER_DATASET = 1

MODEL_SETUP_ROWS = [
    ("Wan baseline", "text + first frame + context video", "context-aware", "672x384"),
    ("step-008000", "text + first frame + context video", "context-aware", "672x384"),
    ("step-010000", "text + first frame + context video", "context-aware", "672x384"),
    ("Wan pure TI2V", "text + first frame image", "TI2V", "672x384"),
    ("VACE TI2V", "text + first frame image", "TI2V", "720x544"),
    ("VACE ctx01", "text + VACE video + mask, context=1 frame", "V2V", "720x544"),
    ("VACE ctx02", "text + VACE video + mask, context=2 frames", "V2V", "720x544"),
    ("VACE ctx04", "text + VACE video + mask, context=4 frames", "V2V", "720x544"),
    ("VACE ctx08", "text + VACE video + mask, context=8 frames", "V2V", "720x544"),
    ("VACE ctx08_nullcaption", "empty text + VACE video + mask, context=8 frames", "V2V", "720x544"),
    ("VACE fullctx_fullvideo_nullcaption", "empty text + full context video + full video", "V2V", "720x544"),
]

DATASET_LABELS = {
    "kubric_tfds_movi-d": "MOVI-D",
    "version_1_genesis_rigid_data_all_cases": "GenesisRigid",
    "physics-iq-benchmark": "Physics-IQ",
    "vLAR-PhysInOne": "vLAR",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": "OpenVidHD",
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


def load_future_summaries(result_root: Path) -> dict[str, dict[str, Any]]:
    del result_root
    summaries: dict[str, dict[str, Any]] = {}
    return summaries


def load_vbench_summaries(result_root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    candidate_paths = sorted(result_root.glob("model_metrics_vbench_short*/metrics_by_model.json"))
    for summary_path in candidate_paths:
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


def collect_sample300_composition(output_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    base_dir = output_root / SAMPLE300_PRIMARY_MODEL_SUBDIR
    if not base_dir.is_dir():
        return counts
    for json_path in sorted(base_dir.glob("*.json")):
        try:
            payload = load_json(json_path)
        except Exception:
            continue
        dataset_name = str(payload.get("dataset") or "unknown")
        counts[dataset_name] = counts.get(dataset_name, 0) + 1
    return counts


def select_showcase_cases(output_root: Path) -> list[tuple[str, str]]:
    base_entries = compact_portal.base_case_payload(output_root / SAMPLE300_PRIMARY_MODEL_SUBDIR)
    by_dataset: dict[str, list[str]] = {}
    for payload in base_entries:
        dataset = str(payload.get("dataset") or "unknown")
        sample_id = str(payload.get("sample_id") or "")
        if sample_id:
            by_dataset.setdefault(dataset, []).append(sample_id)
    selected: list[tuple[str, str]] = []
    for dataset in DATASET_LABELS:
        bucket = sorted(set(by_dataset.get(dataset, [])))
        for sample_id in bucket[:SHOWCASE_CASES_PER_DATASET]:
            selected.append((dataset, sample_id))
    return selected


def render_case_media(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    web_path = "/" + path.lstrip("/")
    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return f"<img loading='lazy' src='{html_lib.escape(web_path)}' alt='media'>"
    return (
        "<video controls preload='none' muted playsinline>"
        f"<source src='{html_lib.escape(web_path)}' type='video/mp4'>"
        "</video>"
    )


def render_case_input_assets(assets: list[dict[str, str]]) -> str:
    if not assets:
        return "<div class='missing'>Missing</div>"
    chunks = []
    for asset in assets:
        role = str(asset.get("role") or "input")
        path = asset.get("path")
        chunks.append(
            "<div class='mini-media'>"
            f"<div class='mini-head'>{html_lib.escape(role)}</div>"
            f"{render_case_media(path if isinstance(path, str) else None)}"
            "</div>"
        )
    return "".join(chunks)


def build_showcase_cases(benchmark_root: Path) -> list[dict[str, Any]]:
    output_root = benchmark_root / "output"
    portal_dir = (benchmark_root / "tools/visualization/compact_selected_portal").resolve()
    portal_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for dataset, sample_id in select_showcase_cases(output_root):
        sample_key = compact_portal.bel.sanitize_filename(f"{dataset}__{sample_id}")
        sample_asset_dir = portal_dir / "assets" / "samples" / sample_key
        sample_asset_dir.mkdir(parents=True, exist_ok=True)

        full_video_asset: str | None = None
        shared_caption = ""
        model_records: list[dict[str, Any]] = []
        for model_name, _, model_subdir in MODEL_SPECS:
            try:
                _, payload = compact_portal.find_payload(output_root, model_subdir, dataset, sample_id)
            except FileNotFoundError:
                model_records.append(
                    {
                        "display_name": DISPLAY_NAMES.get(model_name, model_name),
                        "status": "missing",
                        "seed": None,
                        "resolution": "-",
                        "fps": None,
                        "task": "",
                        "context_frames": None,
                        "input_assets": [],
                        "output_asset": None,
                    }
                )
                continue

            model_asset_dir = sample_asset_dir / compact_portal.sanitize_token(model_name)
            model_asset_dir.mkdir(parents=True, exist_ok=True)
            input_assets = compact_portal.materialize_input_assets(
                payload=payload,
                benchmark_root=benchmark_root,
                asset_dir=model_asset_dir,
            )
            paths = payload.get("paths", {})
            if not isinstance(paths, dict):
                paths = {}
            if full_video_asset is None:
                full_video_asset = compact_portal.link_reference_asset(
                    benchmark_root=benchmark_root,
                    asset_dir=sample_asset_dir,
                    raw_path=paths.get("full_video_path") if isinstance(paths.get("full_video_path"), str) else None,
                    link_name="gt_full_video.mp4",
                )

            model_inputs = payload.get("model_inputs", {})
            if isinstance(model_inputs, dict) and not shared_caption:
                shared_caption = str(model_inputs.get("input_text") or payload.get("caption") or "")
            elif not shared_caption:
                shared_caption = str(payload.get("caption") or "")

            generation = payload.get("generation_params", {})
            if not isinstance(generation, dict):
                generation = {}
            width = generation.get("width")
            height = generation.get("height")
            resolution = "-"
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                resolution = f"{width}x{height}"
            output_asset = None
            output_path = paths.get("output_video_path")
            if isinstance(output_path, str) and output_path and Path(output_path).exists():
                output_asset = compact_portal.relpath_from_root(benchmark_root, Path(output_path))

            model_records.append(
                {
                    "display_name": DISPLAY_NAMES.get(model_name, model_name),
                    "status": str(payload.get("status") or ""),
                    "seed": payload.get("seed"),
                    "resolution": resolution,
                    "fps": generation.get("fps"),
                    "task": str(generation.get("conditioning_mode") or generation.get("task") or ""),
                    "context_frames": generation.get("used_context_frames") or generation.get("context_frames"),
                    "input_assets": input_assets,
                    "output_asset": output_asset,
                }
            )

        cases.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "caption": shared_caption,
                "full_video_asset": full_video_asset,
                "models": model_records,
            }
        )
    return cases


def render_showcase_cases(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""

    dataset_cards: list[str] = []
    order = {name: idx for idx, name in enumerate(DATASET_LABELS)}
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_dataset.setdefault(str(case.get("dataset") or "unknown"), []).append(case)

    for dataset in sorted(by_dataset, key=lambda item: order.get(item, 999)):
        sample_cards: list[str] = []
        for case in by_dataset[dataset]:
            model_cols: list[str] = []
            for model in case.get("models", []):
                seed = model.get("seed")
                fps = model.get("fps")
                context_frames = model.get("context_frames")
                meta_parts = [
                    f"seed={seed}" if isinstance(seed, int) else "seed=-",
                    str(model.get("resolution") or "-"),
                    f"{fps} fps" if isinstance(fps, int) else "fps=-",
                ]
                if isinstance(context_frames, int) and context_frames > 0:
                    meta_parts.append(f"ctx={context_frames}")
                task = str(model.get("task") or "")
                model_cols.append(
                    "<div class='model-col'>"
                    "<div class='tile-head'>"
                    f"<span class='model-name'>{html_lib.escape(str(model.get('display_name') or 'model'))}</span>"
                    f"<span class='status'>{html_lib.escape(str(model.get('status') or ''))}</span>"
                    "</div>"
                    f"<p class='model-meta'>{html_lib.escape(' · '.join(meta_parts))}</p>"
                    f"<p class='model-task'>{html_lib.escape(task)}</p>"
                    "<div class='mini-head'>input_visual_conditions</div>"
                    f"<div class='inputs-grid'>{render_case_input_assets(model.get('input_assets', []))}</div>"
                    "<div class='output-box'>"
                    "<div class='mini-head'>output_video</div>"
                    f"{render_case_media(model.get('output_asset') if isinstance(model.get('output_asset'), str) else None)}"
                    "</div>"
                    "</div>"
                )

            sample_cards.append(
                "<article class='sample-card'>"
                "<div class='sample-top'>"
                f"<span class='dataset-tag'>{html_lib.escape(DATASET_LABELS.get(dataset, dataset))}</span>"
                f"<h3>{html_lib.escape(str(case.get('sample_id') or 'sample'))}</h3>"
                f"<p class='caption'><strong>input_text:</strong> {html_lib.escape(str(case.get('caption') or ''))}</p>"
                "</div>"
                "<div class='case-overview'>"
                "<div class='shared-col gt-col'>"
                "<div class='tile-head'><span class='model-name'>GT Full Video</span></div>"
                f"{render_case_media(case.get('full_video_asset') if isinstance(case.get('full_video_asset'), str) else None)}"
                "</div>"
                "</div>"
                f"<div class='models-grid'>{''.join(model_cols)}</div>"
                "</article>"
            )

        dataset_cards.append(
            "<section class='dataset-block'>"
            f"<div class='dataset-block-head'><h3>{html_lib.escape(DATASET_LABELS.get(dataset, dataset))}</h3>"
            "<p>固定展示 1 个代表样本。样本顶部给出共享文本条件，GT 单独展示；下方用统一网格对比各模型输出和对应视觉输入。</p></div>"
            f"{''.join(sample_cards)}"
            "</section>"
        )

    return (
        "<section class='card case-panel'>"
        "<h2>Selected Cases</h2>"
        "<p class='meta'>每个数据集展示 1 个代表样本。每个模型卡片都包含实际视觉输入、输出视频，以及 seed、分辨率、fps、context 帧数等基本信息。</p>"
        f"{''.join(dataset_cards)}"
        "</section>"
    )


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
    sample300_composition: dict[str, int],
    showcase_cases: list[dict[str, Any]],
) -> None:
    del testset_compositions
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

    future_metric_notes = [
        ("PSNR", "像素级重建误差，越高越好；对细小偏差敏感。"),
        ("SSIM", "结构和局部纹理一致性，越高越好；比 PSNR 更贴近结构观感。"),
        ("LPIPS", "感知相似度，越低越好；更关注人眼觉得像不像。"),
        ("DINO", "高层语义和目标状态一致性，越高越好；更适合看物体结果是否对。"),
    ]
    vbench_metric_notes = [
        ("Subject Consistency", "主体时序稳定性，越高越好；漂移和变形会降分。"),
        ("Background Consistency", "背景连贯性，越高越好；背景乱跳会降分。"),
        ("Motion Smoothness", "运动顺滑程度，越高越好；生硬断裂会降分。"),
        ("Temporal Flickering", "时序闪烁控制，越高越好；亮度和纹理跳变会降分。"),
        ("Dynamic Degree", "动态幅度大小；不是越高越好，过低太静，过高可能是假运动。"),
        ("Imaging Quality", "单帧画质，越高越好；反映清晰度、噪声和成像质量。"),
        ("Aesthetic Quality", "主观观感和美学质量，越高越好。"),
        ("Overall Consistency", "整段视频整体一致性，越高越好；可作通用时序辅助参考。"),
        ("Temporal Style", "时间维风格稳定性，越高越好；看颜色和质感是否前后一致。"),
    ]
    future_notes_html = "".join(
        "<li>"
        f"<strong>{name}</strong>: {desc}"
        "</li>"
        for name, desc in future_metric_notes
    )
    vbench_notes_html = "".join(
        "<li>"
        f"<strong>{name}</strong>: {desc}"
        "</li>"
        for name, desc in vbench_metric_notes
    )
    composition_rows = "".join(
        "<tr>"
        f"<td>{DATASET_LABELS.get(dataset_name, dataset_name)}</td>"
        f"<td>{count}</td>"
        "</tr>"
        for dataset_name, count in sorted(sample300_composition.items(), key=lambda item: (-item[1], item[0]))
    )
    setup_rows = "".join(
        "<tr>"
        f"<td>{model_name}</td>"
        f"<td>{inputs}</td>"
        f"<td>{task}</td>"
        f"<td>{resolution}</td>"
        "</tr>"
        for model_name, inputs, task, resolution in MODEL_SETUP_ROWS
    )
    showcase_cases_html = render_showcase_cases(showcase_cases)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage0 Metric Line Charts</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: rgba(255, 251, 244, 0.94);
      --panel-strong: #fffdf8;
      --line: #d8cfbf;
      --line-soft: #e7dece;
      --ink: #1f1b16;
      --muted: #6d655b;
      --accent: #b5532d;
      --accent-soft: #f3d7c9;
      --teal: #2a6f8f;
      --olive: #557a46;
      --ok-bg: #dcebdc;
      --ok-ink: #28563c;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181,83,45,0.10), transparent 24%),
        radial-gradient(circle at top right, rgba(42,111,143,0.08), transparent 20%),
        linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .shell {{
      width: min(1720px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }}
    .hero {{
      padding: 18px 20px;
      background: linear-gradient(180deg, rgba(255,253,248,0.98), rgba(255,248,238,0.95));
      border: 1px solid var(--line);
      border-radius: 18px;
      margin-bottom: 14px;
      box-shadow: 0 18px 40px rgba(73, 52, 33, 0.06);
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    .hero p {{
      margin: 0 0 5px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      max-width: 1320px;
    }}
    .top-layout {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .stack {{
      display: grid;
      gap: 14px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr);
      gap: 14px;
      margin-bottom: 14px;
    }}
    .grid {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      padding: 14px 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 14px 32px rgba(73, 52, 33, 0.05);
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 17px;
      letter-spacing: -0.01em;
    }}
    .card p.meta {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-word;
    }}
    .card img {{
      display: block;
      width: 100%;
      border-radius: 12px;
      background: #f8f4eb;
      border: 1px solid var(--line-soft);
    }}
    .chart-card img {{
      min-height: 560px;
      object-fit: contain;
    }}
    .chart-subnotes {{
      margin: 0 0 10px;
      padding: 10px 12px;
      border: 1px solid var(--line-soft);
      border-radius: 12px;
      background: #faf6ee;
      font-size: 12px;
      line-height: 1.5;
    }}
    .chart-subnotes ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .chart-subnotes li {{
      margin: 0 0 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 7px 8px;
      border-bottom: 1px solid var(--line-soft);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: #5a4e42;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .table-tight td:first-child, .table-tight th:first-child {{
      width: 26%;
    }}
    .summary-card {{
      overflow-x: auto;
    }}
    .setup-card {{
      display: grid;
      gap: 14px;
    }}
    .setup-grid {{
      display: grid;
      grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.3fr);
      gap: 14px;
    }}
    .mini-note {{
      margin: 0;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }}
    .dataset-block {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }}
    .dataset-block-head {{
      padding: 12px 14px;
      border: 1px solid var(--line-soft);
      border-radius: 14px;
      background: linear-gradient(180deg, #fffaf2, #faf4ea);
    }}
    .dataset-block-head h3 {{
      margin: 0 0 4px;
      font-size: 16px;
    }}
    .dataset-block-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
    }}
    .sample-card {{
      padding: 12px;
      border: 1px solid var(--line-soft);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,251,244,0.98), rgba(255,248,238,0.95));
    }}
    .sample-top {{
      display: grid;
      gap: 6px;
      margin-bottom: 12px;
    }}
    .sample-top h3 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.3;
      word-break: break-word;
    }}
    .dataset-tag {{
      justify-self: start;
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #5c311f;
      font-size: 11px;
      font-weight: 600;
    }}
    .caption {{
      margin: 0;
      color: #473e36;
      font-size: 12px;
      line-height: 1.5;
    }}
    .case-overview {{
      display: grid;
      grid-template-columns: minmax(0, 380px);
      gap: 10px;
      margin-bottom: 12px;
    }}
    .shared-col, .model-col {{
      padding: 8px;
      border: 1px solid var(--line-soft);
      border-radius: 12px;
      background: var(--panel-strong);
    }}
    .gt-col video {{
      min-height: 210px;
      max-height: 240px;
    }}
    .models-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }}
    .tile-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      margin-bottom: 8px;
      font-size: 11px;
      font-weight: 700;
    }}
    .model-name {{
      color: #5b2717;
      word-break: break-word;
    }}
    .status {{
      color: var(--ok-ink);
      background: var(--ok-bg);
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 10px;
      white-space: nowrap;
    }}
    .model-meta {{
      margin: 8px 0 2px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.45;
    }}
    .model-task {{
      margin: 0 0 8px;
      color: #4c675a;
      font-size: 11px;
      line-height: 1.35;
      font-weight: 600;
    }}
    .inputs-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
      margin-bottom: 10px;
    }}
    .mini-media, .output-box {{
      border: 1px solid var(--line-soft);
      border-radius: 10px;
      overflow: hidden;
      background: #fbf8f2;
    }}
    .mini-head {{
      padding: 6px 7px;
      border-bottom: 1px solid var(--line-soft);
      background: rgba(239, 231, 218, 0.7);
      color: #55493d;
      font-size: 10px;
      font-weight: 700;
      line-height: 1.2;
      word-break: break-word;
    }}
    video, img {{
      display: block;
      width: 100%;
      min-height: 126px;
      max-height: 170px;
      object-fit: contain;
      background: #0d0d0d;
    }}
    .output-box video {{
      min-height: 180px;
      max-height: 220px;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 120px;
      color: var(--muted);
      background: repeating-linear-gradient(
        45deg,
        rgba(216, 207, 191, 0.35),
        rgba(216, 207, 191, 0.35) 10px,
        rgba(255, 253, 248, 0.75) 10px,
        rgba(255, 253, 248, 0.75) 20px
      );
      font-size: 12px;
    }}
    .case-panel {{
      padding-top: 16px;
    }}
    @media (max-width: 1100px) {{
      .top-layout,
      .charts-grid,
      .setup-grid,
      .stats-grid,
      .models-grid {{
        grid-template-columns: 1fr;
      }}
      .case-overview {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Stage0 Metric Line Charts</h1>
      <p>future_metrics 折线图基于同名 sidecar / per-sample 汇总的 300 样本平均值；图中每个点的 n 表示该模型当前实际完成 future 指标的样本数。</p>
      <p>vbench_metrics 折线图基于 GPU3/GPU5 汇总出的模型级 `vbench_short_metrics`，使用当前已有完整 300 样本结果。</p>
      <p>future 指标直接从最新 sidecar 聚合，避免旧 summary 文件滞后；新增的 `VACE ctx08_nullcaption` 和 `VACE fullctx_fullvideo_nullcaption` 都按已并入的实际样本数统计，未完成指标的列会显示为空。</p>
      <p>指标简述：`PSNR/SSIM/DINO` 越高越好，分别偏像素重建、结构一致性、语义一致性；`LPIPS` 越低越好，更接近人眼感知差异；`VBench` 各项主要反映主体稳定、背景稳定、运动平滑、闪烁、画质和美学质量。</p>
    </section>
    <section class="top-layout">
      <div class="stack">
        <section class="card summary-card">
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
        <section class="card setup-card">
          <h2>Benchmark Setup</h2>
          <p class="meta">当前主结果对应 `sample300_full`。future 指标比较生成 future 段与 GT future；VBench 比较完整生成视频。不同模型输入条件和输出分辨率并不完全一致，下面单独列出。</p>
          <div class="setup-grid">
            <div>
              <table class="table-tight">
                <thead>
                  <tr>
                    <th>dataset</th>
                    <th>count</th>
                  </tr>
                </thead>
                <tbody>{composition_rows}</tbody>
              </table>
            </div>
            <div>
              <table>
                <thead>
                  <tr>
                    <th>model</th>
                    <th>inputs</th>
                    <th>task</th>
                    <th>resolution</th>
                  </tr>
                </thead>
                <tbody>{setup_rows}</tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </section>
    <section class="charts-grid">
      <section class="card chart-card">
        <h2>Future Metrics</h2>
        <p class="meta">比较 300 个样本上生成 future 段与 GT future 的对齐程度；`PSNR/SSIM/DINO` 越高越好，`LPIPS` 越低越好。</p>
        <div class="chart-subnotes">
          <ul>{future_notes_html}</ul>
        </div>
        <img src="future_metrics.png" alt="future metrics">
      </section>
      <section class="card chart-card">
        <h2>VBench Short Metrics</h2>
        <p class="meta">比较完整生成视频的通用质量与时序稳定性；大多数指标越高越好，`Dynamic Degree` 主要反映运动幅度，不是单调越高越好。</p>
        <div class="chart-subnotes">
          <ul>{vbench_notes_html}</ul>
        </div>
        <img src="vbench_metrics.png" alt="vbench metrics">
      </section>
    </section>
    <section class="grid">
      {showcase_cases_html}
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
    testset_compositions: list[dict[str, Any]] = []
    sample300_composition = collect_sample300_composition(benchmark_root / "output")
    showcase_cases = build_showcase_cases(benchmark_root)

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

    summary_rows = build_summary_rows(model_summary)
    write_json(output_root / "metrics_summary.json", model_summary)
    write_csv(
        output_root / "metrics_summary.csv",
        summary_rows,
        fieldnames=list(summary_rows[0].keys()) if summary_rows else [],
    )
    write_json(output_root / "testset_compositions.json", testset_compositions)
    write_index(
        output_root=output_root,
        model_summary=model_summary,
        testset_compositions=testset_compositions,
        sample300_composition=sample300_composition,
        showcase_cases=showcase_cases,
    )
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
