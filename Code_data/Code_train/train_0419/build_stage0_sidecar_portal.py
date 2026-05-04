#!/usr/bin/env python3
"""Build a stage0 portal and materialize actual model-input visualization assets."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import batch_eval_lora as bel
import batch_eval_vace as bev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portal from stage0 sidecar json files.")
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
        default=Path("tools/visualization/output_sidecar_portal"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def web_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.replace(os.sep, "/").lstrip("/")
    return f"/{normalized}"


def sanitize_token(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text).strip("._")
    return safe or "item"


def ensure_clean_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()


def ensure_symlink(target: Path, link_path: Path) -> str | None:
    if not target.exists():
        return None
    ensure_clean_parent(link_path)
    link_path.symlink_to(target)
    return link_path.name


def save_image_asset(image: Image.Image, path: Path) -> None:
    ensure_clean_parent(path)
    image.convert("RGB").save(path)


def save_video_asset(frames: list[Image.Image], path: Path, fps: int) -> None:
    ensure_clean_parent(path)
    bel.save_video(frames, str(path), fps=fps, quality=5)


def media_html(path: str | None) -> str:
    if not path:
        return "<div class='missing'>Missing</div>"
    resolved = web_path(path)
    if not resolved:
        return "<div class='missing'>Missing</div>"
    lowered = path.lower()
    if lowered.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return f"<img loading='lazy' src='{html.escape(resolved)}' alt='input image'>"
    if lowered.endswith(".mp4"):
        return (
            f"<video controls preload='metadata' muted playsinline>"
            f"<source src='{html.escape(resolved)}' type='video/mp4'>"
            "</video>"
        )
    return f"<a href='{html.escape(resolved)}' target='_blank' rel='noreferrer'>{html.escape(Path(path).name)}</a>"


def text_html(text: str | None) -> str:
    if not text:
        return "<div class='missing'>Missing</div>"
    return f"<div class='text-slot-body'>{html.escape(text)}</div>"


def render_media_slot(*, title: str, body: str, extra_class: str = "") -> str:
    class_attr = "media-slot"
    if extra_class:
        class_attr += f" {extra_class}"
    return (
        f"<div class='{class_attr}'>"
        f"<div class='slot-head'>{html.escape(title)}</div>"
        f"{body}"
        "</div>"
    )


def render_input_group(items: list[dict[str, Any]]) -> str:
    if not items:
        return "<div class='media-grid single'><div class='missing'>Missing</div></div>"
    cards = []
    for item in items:
        role = str(item.get("role") or "input")
        kind = str(item.get("kind") or "media")
        if kind == "text":
            cards.append(render_media_slot(title=role, body=text_html(str(item.get("text") or "")), extra_class="text-slot"))
            continue
        cards.append(render_media_slot(title=role, body=media_html(item.get("path"))))
    grid_class = "media-grid multi" if len(items) > 1 else "media-grid single"
    return f"<div class='{grid_class}'>{''.join(cards)}</div>"


def render_output_slot(path: str | None) -> str:
    return render_media_slot(title="output_video", body=media_html(path), extra_class="output-slot")


def render_reference_slot(path: str | None) -> str:
    return render_media_slot(title="gt_full_video", body=media_html(path), extra_class="reference-slot")


def input_asset_signature(items: list[dict[str, Any]]) -> str:
    normalized: list[dict[str, str]] = []
    for item in items:
        normalized.append(
            {
                "role": str(item.get("role") or ""),
                "kind": str(item.get("kind") or ""),
                "path": str(item.get("path") or ""),
                "text": str(item.get("text") or ""),
            }
        )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True)


def build_input_group_signature(payload: dict[str, Any]) -> str:
    paths = payload.get("paths", {})
    generation_params = payload.get("generation_params", {})
    model_inputs = payload.get("model_inputs", {})
    if not isinstance(paths, dict):
        paths = {}
    if not isinstance(generation_params, dict):
        generation_params = {}
    if not isinstance(model_inputs, dict):
        model_inputs = {}
    signature_payload = {
        "caption": str(model_inputs.get("input_text") or payload.get("caption") or ""),
        "conditioning_mode": str(generation_params.get("conditioning_mode") or model_inputs.get("conditioning_mode") or ""),
        "pipeline_kwargs": list(model_inputs.get("pipeline_kwargs") or []),
        "height": int(generation_params.get("height") or 0),
        "width": int(generation_params.get("width") or 0),
        "fps": int(generation_params.get("fps") or 0),
        "used_context_frames": int(
            generation_params.get("used_context_frames")
            or generation_params.get("context_frames")
            or 0
        ),
        "context_video_path": str(paths.get("context_video_path") or ""),
        "first_frame_path": str(paths.get("first_frame_path") or ""),
        "actual_roles": [
            str(item.get("role") or "")
            for item in model_inputs.get("actual_visual_conditions", [])
            if isinstance(item, dict)
        ],
    }
    return json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)


def reset_assets_root(assets_root: Path) -> None:
    if assets_root.exists():
        shutil.rmtree(assets_root)
    assets_root.mkdir(parents=True, exist_ok=True)


def resolve_reference_asset(
    *,
    raw_path: str | None,
    benchmark_root: Path,
    asset_dir: Path,
    link_name: str,
) -> str | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.exists():
        return None
    if candidate.is_relative_to(benchmark_root):
        return relative_to_root(benchmark_root, candidate)
    linked_name = ensure_symlink(candidate, asset_dir / link_name)
    if not linked_name:
        return None
    return relative_to_root(benchmark_root, asset_dir / linked_name)


def build_fallback_input_assets(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> list[dict[str, Any]]:
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        return []

    raw_roles = paths.get("input_roles")
    raw_specs: list[dict[str, str]] = []
    if isinstance(raw_roles, list):
        for idx, raw_role in enumerate(raw_roles):
            if isinstance(raw_role, dict):
                role = str(raw_role.get("role") or f"input_{idx + 1}")
                path = raw_role.get("path")
                if isinstance(path, str) and path:
                    raw_specs.append({"role": role, "path": path})
            elif isinstance(raw_role, str):
                raw_input = paths.get("input_path")
                if isinstance(raw_input, list) and idx < len(raw_input) and isinstance(raw_input[idx], str):
                    raw_specs.append({"role": raw_role, "path": raw_input[idx]})
    if not raw_specs:
        raw_input = paths.get("input_path")
        if isinstance(raw_input, str) and raw_input:
            raw_specs.append({"role": "input_1", "path": raw_input})
        elif isinstance(raw_input, list):
            for idx, item in enumerate(raw_input, start=1):
                if isinstance(item, str) and item:
                    raw_specs.append({"role": f"input_{idx}", "path": item})

    assets: list[dict[str, Any]] = []
    for idx, spec in enumerate(raw_specs):
        raw_path = spec["path"]
        suffix = Path(raw_path).suffix or ".bin"
        resolved = resolve_reference_asset(
            raw_path=raw_path,
            benchmark_root=benchmark_root,
            asset_dir=asset_dir,
            link_name=f"fallback_input_{idx:02d}{suffix}",
        )
        if resolved:
            assets.append({"role": spec["role"], "path": resolved, "kind": "media"})
    return assets


def build_case_stub(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = str(payload.get("dataset") or "unknown")
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    return {
        "dataset": dataset,
        "sample_id": str(payload.get("sample_id") or "unknown"),
        "caption": str(payload.get("caption") or ""),
        "context_path": str(paths.get("context_video_path") or ""),
        "source_paths": paths,
        "context_resize_mode": bel.resolve_context_resize_mode(dataset),
    }


def materialize_wan_actual_inputs(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> list[dict[str, Any]]:
    generation_params = payload.get("generation_params", {})
    paths = payload.get("paths", {})
    if not isinstance(generation_params, dict) or not isinstance(paths, dict):
        return []

    context_video_path = paths.get("context_video_path")
    if not isinstance(context_video_path, str) or not context_video_path:
        return []

    height = int(generation_params.get("height") or 0)
    width = int(generation_params.get("width") or 0)
    fps = int(generation_params.get("fps") or 8)
    used_context_frames = int(
        generation_params.get("used_context_frames")
        or generation_params.get("context_frames")
        or 0
    )
    conditioning_mode = str(generation_params.get("conditioning_mode") or "")
    resize_mode = bel.resolve_context_resize_mode(str(payload.get("dataset") or "unknown"))

    first_frame = bel.load_input_image(
        first_frame_path=Path(paths["first_frame_path"]) if isinstance(paths.get("first_frame_path"), str) else None,
        context_path=Path(context_video_path),
        height=height,
        width=width,
        resize_mode=resize_mode,
    )

    assets: list[dict[str, Any]] = []
    input_image_path = asset_dir / "input_image.png"
    save_image_asset(first_frame, input_image_path)
    assets.append(
        {
            "role": "input_image",
            "path": relative_to_root(benchmark_root, input_image_path),
            "kind": "media",
        }
    )

    if conditioning_mode == "context_aware" and used_context_frames > 0:
        context_frames = bel.load_context_frames(
            context_path=Path(context_video_path),
            context_frames=used_context_frames,
            height=height,
            width=width,
            resize_mode=resize_mode,
        )
        context_video_asset = asset_dir / "input_context_video.mp4"
        save_video_asset(context_frames, context_video_asset, fps=fps)
        assets.append(
            {
                "role": "input_context_video",
                "path": relative_to_root(benchmark_root, context_video_asset),
                "kind": "media",
            }
        )
    return assets


def materialize_vace_actual_inputs(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> list[dict[str, Any]]:
    generation_params = payload.get("generation_params", {})
    if not isinstance(generation_params, dict):
        return []

    mode = str(generation_params.get("conditioning_mode") or "")
    height = int(generation_params.get("height") or 0)
    width = int(generation_params.get("width") or 0)
    fps = int(generation_params.get("fps") or 8)
    aligned_num_frames = int(
        generation_params.get("aligned_generation_num_frames")
        or generation_params.get("num_frames")
        or 0
    )
    context_frames = int(
        generation_params.get("used_context_frames")
        or generation_params.get("context_frames")
        or 0
    )

    case = build_case_stub(payload)
    video_input, video_mask, _ = bev.build_vace_inputs(
        case=case,
        mode=mode,
        context_frames=context_frames,
        height=height,
        width=width,
        aligned_num_frames=aligned_num_frames,
    )
    input_video_path = asset_dir / "input_vace_video.mp4"
    input_mask_path = asset_dir / "input_vace_video_mask.mp4"
    save_video_asset(video_input, input_video_path, fps=fps)
    save_video_asset(video_mask, input_mask_path, fps=fps)
    return [
        {
            "role": "input_vace_video",
            "path": relative_to_root(benchmark_root, input_video_path),
            "kind": "media",
        },
        {
            "role": "input_vace_video_mask",
            "path": relative_to_root(benchmark_root, input_mask_path),
            "kind": "media",
        },
    ]


def materialize_actual_inputs(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
    asset_dir: Path,
) -> tuple[list[dict[str, Any]], str | None]:
    model_inputs = payload.get("model_inputs", {})
    pipeline_kwargs = model_inputs.get("pipeline_kwargs", []) if isinstance(model_inputs, dict) else []
    try:
        if isinstance(pipeline_kwargs, list) and "vace_video" in pipeline_kwargs:
            return materialize_vace_actual_inputs(
                payload=payload,
                benchmark_root=benchmark_root,
                asset_dir=asset_dir,
            ), None
        return materialize_wan_actual_inputs(
            payload=payload,
            benchmark_root=benchmark_root,
            asset_dir=asset_dir,
        ), None
    except Exception as exc:
        return [], repr(exc)


def extract_existing_actual_input_assets(
    *,
    payload: dict[str, Any],
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    model_inputs = payload.get("model_inputs", {})
    if not isinstance(model_inputs, dict):
        return []
    raw_items = model_inputs.get("actual_visual_conditions")
    if not isinstance(raw_items, list):
        return []
    assets: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            return []
        role = str(item.get("role") or "")
        path = item.get("path")
        if not role or not isinstance(path, str) or not path:
            return []
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = benchmark_root / path
        if not candidate.exists():
            return []
        assets.append({"role": role, "path": path, "kind": "media"})
    return assets


def update_payload_with_actual_assets(
    *,
    payload: dict[str, Any],
    actual_input_assets: list[dict[str, Any]],
    build_error: str | None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(payload, ensure_ascii=False))
    paths = updated.setdefault("paths", {})
    model_inputs = updated.setdefault("model_inputs", {})
    if not isinstance(paths, dict):
        paths = {}
        updated["paths"] = paths
    if not isinstance(model_inputs, dict):
        model_inputs = {}
        updated["model_inputs"] = model_inputs

    asset_paths = [item["path"] for item in actual_input_assets if isinstance(item.get("path"), str)]
    if len(asset_paths) == 1:
        paths["input_path"] = asset_paths[0]
    elif asset_paths:
        paths["input_path"] = asset_paths
    paths["input_roles"] = [
        {"role": item["role"], "path": item["path"]}
        for item in actual_input_assets
        if isinstance(item.get("path"), str)
    ]
    model_inputs["actual_visual_conditions"] = [
        {"role": item["role"], "path": item["path"]}
        for item in actual_input_assets
        if isinstance(item.get("path"), str)
    ]
    model_inputs["input_text"] = str(updated.get("caption") or "")
    if build_error:
        model_inputs["visualization_build_error"] = build_error
    else:
        model_inputs.pop("visualization_build_error", None)
    return updated


def collect_entries(
    *,
    benchmark_root: Path,
    output_root: Path,
    portal_dir: Path,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    assets_root = portal_dir / "assets" / "records"

    for json_path in sorted(output_root.rglob("*.json")):
        payload = read_json(json_path)
        model_name = str(payload.get("model_name") or json_path.parent.name)
        dataset = str(payload.get("dataset") or "unknown")
        sample_id = str(payload.get("sample_id") or json_path.stem)
        record_tag = sanitize_token(f"{model_name}__{dataset}__{sample_id}")
        asset_dir = assets_root / record_tag
        asset_dir.mkdir(parents=True, exist_ok=True)

        actual_input_assets = extract_existing_actual_input_assets(
            payload=payload,
            benchmark_root=benchmark_root,
        )
        build_error = None
        if not actual_input_assets:
            actual_input_assets, build_error = materialize_actual_inputs(
                payload=payload,
                benchmark_root=benchmark_root,
                asset_dir=asset_dir,
            )
        if not actual_input_assets:
            actual_input_assets = build_fallback_input_assets(
                payload=payload,
                benchmark_root=benchmark_root,
                asset_dir=asset_dir,
            )

        updated_payload = update_payload_with_actual_assets(
            payload=payload,
            actual_input_assets=actual_input_assets,
            build_error=build_error,
        )
        if updated_payload != payload:
            write_json(json_path, updated_payload)
        payload = updated_payload
        paths = payload.get("paths", {})
        if not isinstance(paths, dict):
            paths = {}

        input_assets = list(actual_input_assets)
        input_assets.append(
            {
                "role": "input_text",
                "kind": "text",
                "text": str(payload.get("caption") or ""),
            }
        )

        output_asset = None
        raw_output_path = paths.get("output_video_path")
        if isinstance(raw_output_path, str) and raw_output_path:
            candidate = Path(raw_output_path)
            if candidate.exists():
                output_asset = relative_to_root(benchmark_root, candidate)
        if output_asset is None:
            sibling_mp4 = json_path.with_suffix(".mp4")
            if sibling_mp4.exists():
                output_asset = relative_to_root(benchmark_root, sibling_mp4)

        full_video_asset = resolve_reference_asset(
            raw_path=paths.get("full_video_path") if isinstance(paths.get("full_video_path"), str) else None,
            benchmark_root=benchmark_root,
            asset_dir=asset_dir,
            link_name="gt_full_video.mp4",
        )

        entries.append(
            {
                "model_name": model_name,
                "dataset": dataset,
                "sample_id": sample_id,
                "caption": str(payload.get("caption") or ""),
                "status": str(payload.get("status") or ""),
                "input_assets": input_assets,
                "output_asset": output_asset,
                "full_video_asset": full_video_asset,
                "json_relpath": relative_to_root(benchmark_root, json_path),
                "build_error": build_error,
                "input_group_signature": build_input_group_signature(payload),
            }
        )
    return entries


def group_entries_by_sample(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = f"{entry['dataset']}::{entry['sample_id']}"
        group = groups.setdefault(
            key,
            {
                "dataset": entry["dataset"],
                "sample_id": entry["sample_id"],
                "caption": entry["caption"],
                "models": [],
            },
        )
        if not group.get("caption") and entry.get("caption"):
            group["caption"] = entry["caption"]
        group["models"].append(entry)
    grouped = sorted(
        groups.values(),
        key=lambda item: (str(item["dataset"]).lower(), str(item["sample_id"]).lower()),
    )
    for group in grouped:
        group["models"] = sorted(group["models"], key=lambda item: str(item["model_name"]).lower())
        input_groups: dict[str, dict[str, Any]] = {}
        for entry in group["models"]:
            signature = str(entry.get("input_group_signature") or input_asset_signature(entry.get("input_assets", [])))
            input_group = input_groups.setdefault(
                signature,
                {
                    "signature": signature,
                    "input_assets": entry.get("input_assets", []),
                    "full_video_asset": entry.get("full_video_asset"),
                    "entries": [],
                },
            )
            if not input_group.get("full_video_asset") and entry.get("full_video_asset"):
                input_group["full_video_asset"] = entry.get("full_video_asset")
            input_group["entries"].append(entry)
        group["input_groups"] = sorted(
            input_groups.values(),
            key=lambda item: (
                min(str(entry["model_name"]).lower() for entry in item["entries"]),
                len(item["entries"]),
            ),
        )
        for input_group in group["input_groups"]:
            input_group["entries"] = sorted(
                input_group["entries"],
                key=lambda item: str(item["model_name"]).lower(),
            )
    return grouped


def render_output_card(entry: dict[str, Any]) -> str:
    model_name = html.escape(entry["model_name"])
    status = html.escape(entry["status"])
    json_relpath = html.escape(entry["json_relpath"])
    output_html = render_output_slot(entry.get("output_asset"))
    error_html = ""
    if entry.get("build_error"):
        error_html = f"<p class='build-error'>{html.escape(str(entry['build_error']))}</p>"
    return (
        "<section class='output-card' "
        f"data-model='{model_name.lower()}'>"
        "<div class='meta-row'>"
        f"<span class='badge model'>{model_name}</span>"
        f"<span class='badge status'>{status}</span>"
        "</div>"
        f"<p class='json-path compact'>{json_relpath}</p>"
        f"{error_html}"
        f"{output_html}"
        "</section>"
    )


def render_input_compare_group(group: dict[str, Any]) -> str:
    entries = group.get("entries", [])
    input_assets = group.get("input_assets", [])
    input_html = render_input_group(input_assets)
    gt_html = render_reference_slot(group.get("full_video_asset"))
    outputs_html = "".join(render_output_card(entry) for entry in entries)
    model_count = len(entries)
    return (
        "<section class='input-compare-group'>"
        "<div class='meta-row compare-head'>"
        f"<span class='badge compare'>{model_count} output(s) share this input</span>"
        "</div>"
        "<div class='compare-grid'>"
        f"<div class='shared-column'>{input_html}</div>"
        f"<div class='gt-column'>{gt_html}</div>"
        f"<div class='outputs-grid'>{outputs_html}</div>"
        "</div>"
        "</section>"
    )


def render_cards(groups: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for group in groups:
        dataset = html.escape(group["dataset"])
        sample_id = html.escape(group["sample_id"])
        caption = html.escape(group.get("caption", ""))
        models = group.get("models", [])
        model_names_lower = " ".join(str(item["model_name"]).lower() for item in models)
        compare_groups_html = "".join(render_input_compare_group(item) for item in group.get("input_groups", []))
        chunks.append(
            "<article class='sample-card' "
            f"data-models='{html.escape(model_names_lower)}' "
            f"data-dataset='{dataset.lower()}' "
            f"data-sample-id='{sample_id.lower()}' "
            f"data-caption='{caption.lower()}'>"
            "<div class='meta-row'>"
            f"<span class='badge dataset'>{dataset}</span>"
            f"<span class='badge sample-count'>{len(models)} model(s)</span>"
            "</div>"
            f"<h3>{sample_id}</h3>"
            "<div class='compare-stack'>"
            f"{compare_groups_html}"
            "</div>"
            "</article>"
        )
    return "".join(chunks)


def build_html(entries: list[dict[str, Any]]) -> str:
    groups = group_entries_by_sample(entries)
    dataset_options = "".join(
        f"<option value='{html.escape(dataset)}'>{html.escape(dataset)}</option>"
        for dataset in sorted({entry["dataset"] for entry in entries})
    )
    model_options = "".join(
        f"<option value='{html.escape(model)}'>{html.escape(model)}</option>"
        for model in sorted({entry["model_name"] for entry in entries})
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage0 Output Sidecars</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --panel: #fffdf8;
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
      width: min(1960px, calc(100vw - 20px));
      margin: 0 auto;
      padding: 14px 0 24px;
    }}
    .hero {{
      margin-bottom: 12px;
      padding: 14px 18px;
      background: rgba(255,253,248,0.90);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: 0 10px 26px rgba(33, 24, 16, 0.05);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.05;
    }}
    .hero code {{
      font-size: 12px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }}
    .filters {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 8px;
      margin: 10px 0 12px;
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
    .record-list {{
      display: grid;
      gap: 8px;
    }}
    .sample-card {{
      padding: 10px;
      background: rgba(255,253,248,0.95);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(33, 24, 16, 0.04);
    }}
    .compare-stack {{
      display: grid;
      gap: 8px;
    }}
    .input-compare-group {{
      padding: 8px;
      background: #fffaf2;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: minmax(240px, 300px) minmax(180px, 220px) minmax(720px, 1fr);
      gap: 8px;
      align-items: start;
    }}
    .shared-column, .gt-column {{
      display: grid;
      gap: 6px;
    }}
    .outputs-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
      gap: 8px;
      align-items: start;
    }}
    .output-card {{
      padding: 8px;
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 10px;
    }}
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 6px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #efe7da;
      color: #4f4338;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    .badge.model {{
      background: var(--accent-soft);
      color: #6e2a13;
    }}
    .badge.status {{
      background: var(--ok-soft);
      color: var(--ok-ink);
    }}
    .badge.compare {{
      background: #ede4d5;
      color: #5d4d3a;
    }}
    .sample-card h3 {{
      margin: 0 0 6px;
      font-size: 14px;
      line-height: 1.2;
    }}
    .json-path {{
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
    .json-path.compact {{
      margin: 0 0 6px;
      font-size: 10px;
    }}
    .build-error {{
      margin: 0 0 8px;
      color: #8b3f1f;
      font-size: 12px;
      word-break: break-word;
    }}
    .media-grid {{
      display: grid;
      gap: 6px;
    }}
    .media-grid.multi {{
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    }}
    .media-slot {{
      background: #fbf8f2;
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      min-height: 132px;
    }}
    .slot-head {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line);
      font-size: 11px;
      font-weight: 700;
      color: #55493d;
      background: rgba(239, 231, 218, 0.65);
    }}
    .output-slot .slot-head {{
      background: rgba(243, 215, 201, 0.7);
      color: #6e2a13;
    }}
    .reference-slot .slot-head {{
      background: rgba(214, 234, 217, 0.75);
      color: var(--ok-ink);
    }}
    .text-slot-body {{
      min-height: 132px;
      padding: 8px 10px;
      background: #fffdf9;
      color: var(--ink);
      font-size: 12px;
      line-height: 1.35;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    video, img {{
      display: block;
      width: 100%;
      height: 100%;
      min-height: 132px;
      object-fit: contain;
      background: #0d0d0d;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 132px;
      padding: 12px;
      color: var(--muted);
      background: repeating-linear-gradient(
        45deg,
        rgba(216, 207, 191, 0.35),
        rgba(216, 207, 191, 0.35) 10px,
        rgba(255, 253, 248, 0.75) 10px,
        rgba(255, 253, 248, 0.75) 20px
      );
    }}
    @media (max-width: 1320px) {{
      .compare-grid {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 1080px) {{
      .filters {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>Stage0 Output Sidecar Portal</h1>
      <p>Each sample is grouped by identical prepared inputs. Models that consume the same <code>input_*</code> and <code>input_text</code> now share one compact input block, so their outputs can be compared side by side.</p>
    </section>
    <section class="filters">
      <input id="searchBox" type="search" placeholder="Search sample id or caption">
      <select id="modelFilter">
        <option value="">All models</option>
        {model_options}
      </select>
      <select id="datasetFilter">
        <option value="">All datasets</option>
        {dataset_options}
      </select>
    </section>
    <section id="recordList" class="record-list">
      {render_cards(groups)}
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const modelFilter = document.getElementById('modelFilter');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.sample-card'));
    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      const model = modelFilter.value.toLowerCase();
      const dataset = datasetFilter.value.toLowerCase();
      for (const card of cards) {{
        const compareGroups = Array.from(card.querySelectorAll('.input-compare-group'));
        let visibleGroupCount = 0;
        for (const group of compareGroups) {{
          const outputCards = Array.from(group.querySelectorAll('.output-card'));
          let visibleOutputCount = 0;
          for (const outputCard of outputCards) {{
            const cardMatchesModel = !model || outputCard.dataset.model === model;
            outputCard.style.display = cardMatchesModel ? '' : 'none';
            if (cardMatchesModel) visibleOutputCount += 1;
          }}
          group.style.display = visibleOutputCount > 0 ? '' : 'none';
          if (visibleOutputCount > 0) visibleGroupCount += 1;
        }}
        const matchesModel = visibleGroupCount > 0;
        const matchesDataset = !dataset || card.dataset.dataset === dataset;
        const haystack = `${{card.dataset.sampleId}} ${{card.dataset.caption}} ${{card.dataset.models}} ${{card.dataset.dataset}}`.toLowerCase();
        const matchesSearch = !search || haystack.includes(search);
        card.style.display = matchesModel && matchesDataset && matchesSearch ? '' : 'none';
      }}
    }}
    searchBox.addEventListener('input', applyFilters);
    modelFilter.addEventListener('change', applyFilters);
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
    assets_root = portal_dir / "assets" / "records"
    assets_root.mkdir(parents=True, exist_ok=True)

    entries = collect_entries(
        benchmark_root=benchmark_root,
        output_root=output_root,
        portal_dir=portal_dir,
    )
    html_path = portal_dir / "index.html"
    html_path.write_text(build_html(entries), encoding="utf-8")

    summary = {
        "entry_count": len(entries),
        "sample_count": len(group_entries_by_sample(entries)),
        "html_path": str(html_path),
        "portal_url_path": f"/{relative_to_root(benchmark_root, html_path)}",
    }
    write_json(portal_dir / "build_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
