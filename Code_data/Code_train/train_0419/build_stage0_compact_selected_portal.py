#!/usr/bin/env python3
"""Build a compact selected-case portal for stage0 benchmark comparison."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image

import batch_eval_lora as bel
import batch_eval_vace as bev


MODEL_SPECS = [
    ("base-ti2v-5b", "wan2_2_5B_baseline_TI2V"),
    ("step-008000", "wan2.25B_lora_sample300_full49/step-008000"),
    ("step-010000", "wan2.25B_lora_sample300_full49/step-010000"),
    ("wan_pure_ti2v_5b", "Wan2_2_5B_pure_TI2V"),
    ("vace_ti2v_firstframe", "VACE_1_3B_TI2V"),
    ("vace_v2v_ctx01f", "VACE_1_3B_V2V/context_01f"),
    ("vace_v2v_ctx02f", "VACE_1_3B_V2V/context_02f"),
    ("vace_v2v_ctx04f", "VACE_1_3B_V2V/context_04f"),
    ("vace_v2v_ctx08f", "VACE_1_3B_V2V/context_08f"),
    ("vace_v2v_ctx08f_nullcaption", "VACE_1_3B_V2V_nullcaption/context_08f"),
]

DATASET_QUOTAS = {
    "kubric_tfds_movi-d": 2,
    "version_1_genesis_rigid_data_all_cases": 2,
    "physics-iq-benchmark": 2,
    "vLAR-PhysInOne": 2,
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": 2,
}

DATASET_LABELS = {
    "kubric_tfds_movi-d": "MOVI-D",
    "version_1_genesis_rigid_data_all_cases": "GenesisRigid",
    "physics-iq-benchmark": "Physics-IQ",
    "vLAR-PhysInOne": "vLAR",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": "OpenVidHD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact selected-case portal for stage0 benchmark.")
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V"),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/output"),
    )
    parser.add_argument(
        "--portal_subdir",
        type=Path,
        default=Path("tools/visualization/compact_selected_portal"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def relpath_from_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def sanitize_token(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._")
    return safe or "item"


def ensure_symlink(target: Path, link_path: Path) -> str | None:
    if not target.exists():
        return None
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path.name


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def save_video(path: Path, frames: list[Image.Image], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bel.save_video(frames, str(path), fps=fps, quality=5)


def base_case_payload(base_output_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for json_path in sorted(base_output_root.glob("*.json")):
        payload = read_json(json_path)
        entries.append(payload)
    return entries


def select_cases(base_entries: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    by_dataset: dict[str, list[tuple[str, str, str]]] = {}
    for payload in base_entries:
        dataset = str(payload.get("dataset") or "unknown")
        sample_id = str(payload.get("sample_id") or "")
        caption = str(payload.get("caption") or "")
        by_dataset.setdefault(dataset, []).append((dataset, sample_id, caption))
    selected: list[tuple[str, str, str]] = []
    for dataset, quota in DATASET_QUOTAS.items():
        bucket = sorted(by_dataset.get(dataset, []), key=lambda item: item[1])
        selected.extend(bucket[:quota])
    return selected


def find_payload(output_root: Path, model_subdir: str, dataset: str, sample_id: str) -> tuple[Path, dict[str, Any]]:
    stem = bel.sanitize_filename(f"{dataset}__{sample_id}")
    json_path = output_root / model_subdir / f"{stem}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing sidecar: {json_path}")
    return json_path, read_json(json_path)


def build_case_stub(payload: dict[str, Any]) -> dict[str, Any]:
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    dataset = str(payload.get("dataset") or "unknown")
    return {
        "dataset": dataset,
        "sample_id": str(payload.get("sample_id") or "unknown"),
        "caption": str(payload.get("caption") or ""),
        "context_path": str(paths.get("context_video_path") or ""),
        "source_paths": paths,
        "context_resize_mode": bel.resolve_context_resize_mode(dataset),
    }


def materialize_wan_inputs(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> list[dict[str, str]]:
    generation = payload.get("generation_params", {})
    paths = payload.get("paths", {})
    if not isinstance(generation, dict) or not isinstance(paths, dict):
        return []
    context_video_path = paths.get("context_video_path")
    if not isinstance(context_video_path, str) or not context_video_path:
        return []
    height = int(generation.get("height") or 0)
    width = int(generation.get("width") or 0)
    fps = int(generation.get("fps") or 8)
    used_context_frames = int(generation.get("used_context_frames") or generation.get("context_frames") or 0)
    conditioning_mode = str(generation.get("conditioning_mode") or "")
    resize_mode = bel.resolve_context_resize_mode(str(payload.get("dataset") or "unknown"))

    assets: list[dict[str, str]] = []
    first_frame_path = asset_dir / "input_image.png"
    if not first_frame_path.exists():
        first_frame = bel.load_input_image(
            first_frame_path=Path(paths["first_frame_path"]) if isinstance(paths.get("first_frame_path"), str) else None,
            context_path=Path(context_video_path),
            height=height,
            width=width,
            resize_mode=resize_mode,
        )
        save_image(first_frame_path, first_frame)
    assets.append({"role": "input_image", "path": relpath_from_root(benchmark_root, first_frame_path), "kind": "image"})

    if conditioning_mode == "context_aware" and used_context_frames > 0:
        context_asset = asset_dir / "input_context_video.mp4"
        if not context_asset.exists():
            frames = bel.load_context_frames(
                context_path=Path(context_video_path),
                context_frames=used_context_frames,
                height=height,
                width=width,
                resize_mode=resize_mode,
            )
            save_video(context_asset, frames, fps=fps)
        assets.append(
            {"role": "input_context_video", "path": relpath_from_root(benchmark_root, context_asset), "kind": "video"}
        )
    return assets


def materialize_vace_inputs(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> list[dict[str, str]]:
    generation = payload.get("generation_params", {})
    if not isinstance(generation, dict):
        return []
    height = int(generation.get("height") or 0)
    width = int(generation.get("width") or 0)
    fps = int(generation.get("fps") or 8)
    aligned_num_frames = int(generation.get("aligned_generation_num_frames") or generation.get("num_frames") or 49)
    context_frames = int(generation.get("used_context_frames") or generation.get("context_frames") or 0)
    mode = str(generation.get("conditioning_mode") or "")
    case = build_case_stub(payload)

    video_asset = asset_dir / "input_vace_video.mp4"
    mask_asset = asset_dir / "input_vace_video_mask.mp4"
    if not video_asset.exists() or not mask_asset.exists():
        video_input, video_mask, _ = bev.build_vace_inputs(
            case=case,
            mode=mode,
            context_frames=context_frames,
            height=height,
            width=width,
            aligned_num_frames=aligned_num_frames,
        )
        save_video(video_asset, video_input, fps=fps)
        save_video(mask_asset, video_mask, fps=fps)
    return [
        {"role": "input_vace_video", "path": relpath_from_root(benchmark_root, video_asset), "kind": "video"},
        {"role": "input_vace_video_mask", "path": relpath_from_root(benchmark_root, mask_asset), "kind": "video"},
    ]


def materialize_input_assets(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> list[dict[str, str]]:
    model_inputs = payload.get("model_inputs", {})
    if isinstance(model_inputs, dict):
        raw_items = model_inputs.get("actual_visual_conditions")
        if isinstance(raw_items, list):
            assets: list[dict[str, str]] = []
            all_ok = True
            for item in raw_items:
                if not isinstance(item, dict):
                    all_ok = False
                    break
                role = str(item.get("role") or "")
                raw_path = item.get("path")
                if not role or not isinstance(raw_path, str) or not raw_path:
                    all_ok = False
                    break
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = benchmark_root / raw_path
                if not candidate.exists():
                    all_ok = False
                    break
                kind = "image" if candidate.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} else "video"
                assets.append({"role": role, "path": relpath_from_root(benchmark_root, candidate), "kind": kind})
            if all_ok and assets:
                return assets

    pipeline_kwargs = model_inputs.get("pipeline_kwargs", []) if isinstance(model_inputs, dict) else []
    if isinstance(pipeline_kwargs, list) and "vace_video" in pipeline_kwargs:
        try:
            return materialize_vace_inputs(payload=payload, benchmark_root=benchmark_root, asset_dir=asset_dir)
        except FileNotFoundError:
            return []
    try:
        return materialize_wan_inputs(payload=payload, benchmark_root=benchmark_root, asset_dir=asset_dir)
    except FileNotFoundError:
        return []


def link_reference_asset(
    *,
    benchmark_root: Path,
    asset_dir: Path,
    raw_path: str | None,
    link_name: str,
) -> str | None:
    if not raw_path:
        return None
    target = Path(raw_path)
    if not target.exists():
        return None
    link_name = ensure_symlink(target, asset_dir / link_name)
    if not link_name:
        return None
    return relpath_from_root(benchmark_root, asset_dir / link_name)


def render_media(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    web_path = "/" + path.lstrip("/")
    lower = path.lower()
    if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return f"<img loading='lazy' src='{html.escape(web_path)}' alt='media'>"
    return (
        "<video controls preload='metadata' muted playsinline>"
        f"<source src='{html.escape(web_path)}' type='video/mp4'>"
        "</video>"
    )


def render_input_assets(assets: list[dict[str, str]]) -> str:
    chunks = []
    for asset in assets:
        chunks.append(
            "<div class='mini-media'>"
            f"<div class='mini-head'>{html.escape(asset['role'])}</div>"
            f"{render_media(asset['path'])}"
            "</div>"
        )
    return "".join(chunks) if chunks else "<div class='missing'>Missing</div>"


def format_metric_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "-"


def render_metric_pills(*, future_metrics: dict[str, Any], vbench_metrics: dict[str, Any]) -> str:
    pills = [
        ("f_psnr", format_metric_value(future_metrics.get("future_psnr"))),
        ("f_ssim", format_metric_value(future_metrics.get("future_ssim"))),
        ("f_lpips", format_metric_value(future_metrics.get("future_lpips"))),
        ("f_dino", format_metric_value(future_metrics.get("future_dino"))),
        ("vb_subj", format_metric_value(vbench_metrics.get("subject_consistency"))),
        ("vb_motion", format_metric_value(vbench_metrics.get("motion_smoothness"))),
        ("vb_img", format_metric_value(vbench_metrics.get("imaging_quality"))),
    ]
    return "".join(
        "<div class='metric-pill'>"
        f"<span class='metric-key'>{html.escape(key)}</span>"
        f"<span class='metric-val'>{html.escape(value)}</span>"
        "</div>"
        for key, value in pills
    )


def build_case_record(
    *,
    benchmark_root: Path,
    output_root: Path,
    portal_dir: Path,
    dataset: str,
    sample_id: str,
    caption: str,
) -> dict[str, Any]:
    sample_key = bel.sanitize_filename(f"{dataset}__{sample_id}")
    sample_asset_dir = portal_dir / "assets" / "samples" / sample_key
    sample_asset_dir.mkdir(parents=True, exist_ok=True)

    model_records: list[dict[str, Any]] = []
    full_video_asset: str | None = None
    for model_name, model_subdir in MODEL_SPECS:
        _, payload = find_payload(output_root, model_subdir, dataset, sample_id)
        model_asset_dir = sample_asset_dir / sanitize_token(model_name)
        model_asset_dir.mkdir(parents=True, exist_ok=True)
        input_assets = materialize_input_assets(
            payload=payload,
            benchmark_root=benchmark_root,
            asset_dir=model_asset_dir,
        )
        paths = payload.get("paths", {})
        if not isinstance(paths, dict):
            paths = {}
        if full_video_asset is None:
            full_video_asset = link_reference_asset(
                benchmark_root=benchmark_root,
                asset_dir=sample_asset_dir,
                raw_path=paths.get("full_video_path") if isinstance(paths.get("full_video_path"), str) else None,
                link_name="gt_full_video.mp4",
            )
        output_path = paths.get("output_video_path")
        output_asset = None
        if isinstance(output_path, str) and output_path and Path(output_path).exists():
            output_asset = relpath_from_root(benchmark_root, Path(output_path))
        model_records.append(
            {
                "model_name": model_name,
                "status": str(payload.get("status") or ""),
                "seed": payload.get("seed"),
                "input_assets": input_assets,
                "output_asset": output_asset,
                "future_metrics": payload.get("future_metrics", {}) if isinstance(payload.get("future_metrics"), dict) else {},
                "vbench_metrics": payload.get("vbench_metrics", {}) if isinstance(payload.get("vbench_metrics"), dict) else {},
            }
        )

    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "caption": caption,
        "full_video_asset": full_video_asset,
        "models": model_records,
    }


def build_html(
    cases: list[dict[str, Any]],
    metric_summary: dict[str, list[dict[str, str]]],
    curve_summary: dict[str, Any],
) -> str:
    del metric_summary
    del curve_summary
    dataset_options = "".join(
        f"<option value='{html.escape(dataset)}'>{html.escape(dataset)}</option>"
        for dataset in sorted({case['dataset'] for case in cases})
    )
    grouped_cards: list[str] = []
    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_dataset.setdefault(case["dataset"], []).append(case)

    for dataset in sorted(by_dataset.keys(), key=lambda item: list(DATASET_QUOTAS.keys()).index(item) if item in DATASET_QUOTAS else 999):
        cards = []
        for case in by_dataset[dataset]:
            model_cols = []
            for model in case["models"]:
                model_cols.append(
                    "<div class='model-col'>"
                    "<div class='tile-head'>"
                    f"<span class='model-name'>{html.escape(model['model_name'])}</span>"
                    f"<span class='status'>{html.escape(model['status'])}</span>"
                    "</div>"
                    "<div class='mini-head'>input_conditions</div>"
                    f"<div class='metric-pills'>{render_metric_pills(future_metrics=model['future_metrics'], vbench_metrics=model['vbench_metrics'])}</div>"
                    f"<div class='inputs-grid'>{render_input_assets(model['input_assets'])}</div>"
                    "<div class='output-box'>"
                    "<div class='mini-head'>output_video</div>"
                    f"{render_media(model['output_asset'])}"
                    "</div>"
                    "</div>"
                )
            cards.append(
                "<article class='sample-card' "
                f"data-dataset='{html.escape(case['dataset'].lower())}' "
                f"data-sample-id='{html.escape(case['sample_id'].lower())}' "
                f"data-caption='{html.escape(case['caption'].lower())}'>"
                "<div class='sample-top'>"
                f"<span class='dataset-tag'>{html.escape(DATASET_LABELS.get(case['dataset'], case['dataset']))}</span>"
                f"<h2>{html.escape(case['sample_id'])}</h2>"
                f"<p class='caption'>{html.escape(case['caption'])}</p>"
                "</div>"
                "<div class='sample-row'>"
                "<div class='shared-col gt-col'>"
                "<div class='tile-head'><span class='model-name'>gt_full_video</span></div>"
                f"{render_media(case['full_video_asset'])}"
                "</div>"
                f"<div class='models-strip'>{''.join(model_cols)}</div>"
                "</div>"
                "</article>"
            )
        grouped_cards.append(
            "<section class='dataset-block'>"
            f"<div class='dataset-block-head'><h2>{html.escape(DATASET_LABELS.get(dataset, dataset))}</h2><p>固定示例样本，用于比较不同模型在该测试数据集上的输入条件与输出结果。</p></div>"
            f"{''.join(cards)}"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage0 Compact Selected Portal</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
      --panel-2: #fff9ee;
      --ink: #1e1b16;
      --muted: #6e675d;
      --line: #d8cfbf;
      --accent: #b5532d;
      --accent-soft: #f3d7c9;
      --ok-soft: #d6ead9;
      --ok-ink: #28563c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181,83,45,0.10), transparent 26%),
        linear-gradient(180deg, #f9f6ef 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .shell {{
      width: min(1960px, calc(100vw - 16px));
      margin: 0 auto;
      padding: 10px 0 20px;
    }}
    .hero {{
      padding: 12px 16px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,253,248,0.92);
      margin-bottom: 10px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.05;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 4px 6px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 8px;
      margin-bottom: 10px;
    }}
    input, select {{
      width: 100%;
      padding: 9px 11px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-size: 13px;
    }}
    .case-list {{
      display: grid;
      gap: 8px;
    }}
    .dataset-block {{
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }}
    .dataset-block-head {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,253,248,0.95);
    }}
    .dataset-block-head h2 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .dataset-block-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .sample-card {{
      padding: 8px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,253,248,0.95);
    }}
    .sample-top {{
      display: grid;
      gap: 4px;
      margin-bottom: 8px;
    }}
    .sample-top h2 {{
      margin: 0;
      font-size: 14px;
      line-height: 1.2;
    }}
    .dataset-tag {{
      justify-self: start;
      padding: 3px 8px;
      border-radius: 999px;
      background: #efe7da;
      color: #4f4338;
      font-size: 11px;
      font-weight: 600;
    }}
    .caption {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }}
    .sample-row {{
      display: grid;
      grid-template-columns: 200px 1fr;
      gap: 8px;
      align-items: start;
    }}
    .models-strip {{
      display: grid;
      grid-auto-flow: column;
      grid-auto-columns: minmax(220px, 220px);
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 4px;
    }}
    .shared-col, .model-col {{
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-2);
    }}
    .tile-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 6px;
      margin-bottom: 6px;
      font-size: 11px;
      font-weight: 700;
    }}
    .model-name {{
      color: #5b2717;
      word-break: break-word;
    }}
    .status {{
      color: var(--ok-ink);
      background: var(--ok-soft);
      border-radius: 999px;
      padding: 2px 6px;
      font-size: 10px;
      white-space: nowrap;
    }}
    .inputs-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 6px;
    }}
    .metric-pills {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px;
      margin-bottom: 6px;
    }}
    .metric-pill {{
      display: flex;
      justify-content: space-between;
      gap: 6px;
      padding: 3px 5px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255, 253, 248, 0.9);
      font-size: 10px;
      line-height: 1.2;
    }}
    .metric-key {{
      color: #6e675d;
      font-weight: 700;
    }}
    .metric-val {{
      color: #1e1b16;
      font-variant-numeric: tabular-nums;
    }}
    .mini-media, .output-box {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fbf8f2;
    }}
    .mini-head {{
      padding: 5px 6px;
      border-bottom: 1px solid var(--line);
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
      min-height: 120px;
      max-height: 140px;
      object-fit: contain;
      background: #0d0d0d;
    }}
    .gt-col video {{
      min-height: 170px;
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
    @media (max-width: 1100px) {{
      .filters, .sample-row {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Stage0 Compact Selected Portal</h1>
      <p>按测试数据集分组展示固定 case。每个样本共用一份 GT full video，右侧横向条带展示 9 个模型各自的输入条件和输出视频。输入条件部分展示实际输入给模型的视觉条件，样本标题下方展示对应输入文本。</p>
    </section>
    <section class="filters">
      <input id="searchBox" type="search" placeholder="Search sample id or caption">
      <select id="datasetFilter">
        <option value="">All datasets</option>
        {dataset_options}
      </select>
    </section>
    <section id="caseList" class="case-list">
      {''.join(grouped_cards)}
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.sample-card'));
    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      const dataset = datasetFilter.value.toLowerCase();
      for (const card of cards) {{
        const haystack = `${{card.dataset.sampleId}} ${{card.dataset.caption}} ${{card.dataset.dataset}}`.toLowerCase();
        const matchSearch = !search || haystack.includes(search);
        const matchDataset = !dataset || card.dataset.dataset === dataset;
        card.style.display = matchSearch && matchDataset ? '' : 'none';
      }}
    }}
    searchBox.addEventListener('input', applyFilters);
    datasetFilter.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    benchmark_root = args.benchmark_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    portal_dir = (benchmark_root / args.portal_subdir).resolve()
    portal_dir.mkdir(parents=True, exist_ok=True)

    base_entries = base_case_payload(output_root / "wan2_2_5B_baseline_TI2V")
    selected = select_cases(base_entries)
    cases = [
        build_case_record(
            benchmark_root=benchmark_root,
            output_root=output_root,
            portal_dir=portal_dir,
            dataset=dataset,
            sample_id=sample_id,
            caption=caption,
        )
        for dataset, sample_id, caption in selected
    ]
    html_path = portal_dir / "index.html"
    write_text(html_path, build_html(cases, {}, {}))
    summary = {
        "num_cases": len(cases),
        "html_path": str(html_path),
        "portal_url_path": f"/{relpath_from_root(benchmark_root, html_path)}",
        "selected_case_keys": [f"{case['dataset']}::{case['sample_id']}" for case in cases],
    }
    write_json(portal_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
