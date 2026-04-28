#!/usr/bin/env python3
"""Build a local HTML preview for Stage-1A / Stage-1B subset windows."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import socket
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STATE_NAMES = ["u", "v", "d", "w", "h", "du", "dv", "dd", "vis"]
STATE_LABELS = {
    "u": "center x",
    "v": "center y",
    "d": "depth",
    "w": "bbox width",
    "h": "bbox height",
    "du": "vx",
    "dv": "vy",
    "dd": "vz",
    "vis": "visible",
}
STATE_COLORS = [
    "#ff6b6b",
    "#4dabf7",
    "#51cf66",
    "#f59f00",
    "#845ef7",
    "#e64980",
]


def copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def annotate_frame(frame: Image.Image, label: str) -> Image.Image:
    image = frame.copy()
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((8, 8, 140, 42), radius=8, fill=(0, 0, 0, 190))
    draw.text((16, 16), label, fill=(255, 255, 255))
    return image


def draw_state_overlay(
    frames: Sequence[Image.Image],
    y_state_raw: np.ndarray,
    object_colors: Sequence[str],
) -> List[Image.Image]:
    overlay_frames: List[Image.Image] = []
    for t, frame in enumerate(frames):
        image = frame.copy()
        draw = ImageDraw.Draw(image)
        for obj_idx in range(y_state_raw.shape[1]):
            u, v, _d, w, h, _du, _dv, _dd, vis = y_state_raw[t, obj_idx]
            if vis < 0.5 or w <= 1 or h <= 1:
                continue
            color = object_colors[obj_idx % len(object_colors)]
            x1 = float(u - w / 2.0)
            y1 = float(v - h / 2.0)
            x2 = float(u + w / 2.0)
            y2 = float(v + h / 2.0)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), fill=color)
            draw.rounded_rectangle((x1, max(4.0, y1 - 28.0), x1 + 88.0, max(28.0, y1)), radius=6, fill=color)
            draw.text((x1 + 8.0, max(8.0, y1 - 24.0)), f"obj {obj_idx}", fill="white")
        overlay_frames.append(image)
    return overlay_frames


def make_strip(frames: Sequence[Image.Image], dst: Path, max_thumb_height: int = 200) -> None:
    if not frames:
        raise ValueError("Cannot create strip from empty frame list.")
    thumbs = []
    for frame in frames:
        scale = max_thumb_height / float(frame.height)
        width = max(1, int(round(frame.width * scale)))
        thumbs.append(frame.resize((width, max_thumb_height), Image.Resampling.BILINEAR))
    total_width = sum(img.width for img in thumbs) + 8 * (len(thumbs) - 1)
    canvas = Image.new("RGB", (total_width, max_thumb_height), color=(18, 18, 20))
    cursor = 0
    for img in thumbs:
        canvas.paste(img, (cursor, 0))
        cursor += img.width + 8
    canvas.save(dst)


def make_gif(frames: Sequence[Image.Image], dst: Path, max_side: int = 640, duration_ms: int = 350) -> None:
    if not frames:
        return
    processed = []
    for frame in frames:
        scale = min(max_side / float(frame.width), max_side / float(frame.height), 1.0)
        size = (max(1, int(round(frame.width * scale))), max(1, int(round(frame.height * scale))))
        processed.append(frame.resize(size, Image.Resampling.BILINEAR))
    processed[0].save(
        dst,
        save_all=True,
        append_images=processed[1:],
        duration=duration_ms,
        loop=0,
    )


def save_state_plots(
    y_state_raw: np.ndarray,
    y_state_norm: np.ndarray,
    objects: Sequence[dict],
    out_dir: Path,
) -> list[str]:
    out_names: list[str] = []
    timesteps = np.arange(y_state_raw.shape[0], dtype=np.int32)
    for obj_idx in range(y_state_raw.shape[1]):
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
        title_bits = [f"obj {obj_idx}"]
        if obj_idx < len(objects):
            name = str(objects[obj_idx].get("name") or objects[obj_idx].get("category") or "").strip()
            role = str(objects[obj_idx].get("role") or "").strip()
            if name:
                title_bits.append(name)
            if role:
                title_bits.append(role)
        fig.suptitle(" | ".join(title_bits), fontsize=16)
        for state_id, state_name in enumerate(STATE_NAMES):
            ax = axes[state_id // 3][state_id % 3]
            ax.plot(
                timesteps,
                y_state_raw[:, obj_idx, state_id],
                marker="o",
                linewidth=2.0,
                color="#1864ab",
                label="raw",
            )
            ax.plot(
                timesteps,
                y_state_norm[:, obj_idx, state_id],
                marker="s",
                linewidth=1.5,
                color="#e67700",
                alpha=0.8,
                label="norm",
            )
            ax.set_title(STATE_LABELS[state_name])
            ax.set_xlabel("future frame index")
            ax.grid(alpha=0.25)
            if state_name == "vis":
                ax.set_ylim(-0.05, 1.05)
            if state_id == 0:
                ax.legend()
        out_name = f"state_object_{obj_idx}.png"
        fig.savefig(out_dir / out_name, dpi=160)
        plt.close(fig)
        out_names.append(out_name)
    return out_names


def state_table_html(y_state_raw: np.ndarray, objects: Sequence[dict]) -> str:
    blocks: list[str] = []
    for obj_idx in range(y_state_raw.shape[1]):
        title = f"Object {obj_idx}"
        if obj_idx < len(objects):
            obj = objects[obj_idx]
            extra = " / ".join(
                part
                for part in [
                    str(obj.get("name") or "").strip(),
                    str(obj.get("category") or "").strip(),
                    str(obj.get("role") or "").strip(),
                ]
                if part
            )
            if extra:
                title = f"{title} ({extra})"
        headers = "".join(f"<th>{name}</th>" for name in ["t"] + STATE_NAMES)
        rows = []
        for t in range(y_state_raw.shape[0]):
            values = [f"{float(y_state_raw[t, obj_idx, state_id]):.3f}" for state_id in range(y_state_raw.shape[2])]
            row = "".join(f"<td>{value}</td>" for value in [str(t)] + values)
            rows.append(f"<tr>{row}</tr>")
        blocks.append(
            f"""
<section class="state-table-card">
  <h4>{html.escape(title)}</h4>
  <div class="table-wrap">
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</section>
"""
        )
    return "".join(blocks)


def objects_html(objects: Sequence[dict], num_objects: int) -> str:
    cards: list[str] = []
    for obj_idx in range(num_objects):
        obj = dict(objects[obj_idx]) if obj_idx < len(objects) else {}
        lines = [
            ("object_id", obj.get("object_id", "n/a")),
            ("source_object_id", obj.get("source_object_id", "n/a")),
            ("role", obj.get("role", "n/a")),
            ("category", obj.get("category", obj.get("name", "n/a"))),
            ("dataset_source", obj.get("dataset_source", obj.get("source_tag", "n/a"))),
        ]
        body = "".join(
            f"<div><strong>{html.escape(str(k))}</strong>: {html.escape(str(v))}</div>"
            for k, v in lines
        )
        cards.append(
            f"""
<div class="object-card">
  <div class="object-head" style="border-color:{STATE_COLORS[obj_idx % len(STATE_COLORS)]}">
    object {obj_idx}
  </div>
  {body}
</div>
"""
        )
    return "".join(cards)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage-1A/1B subset windows as a local HTML portal.")
    parser.add_argument(
        "--subset_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/stage1_subsets_v1"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/portal_hub/stage1_subset_preview"),
    )
    parser.add_argument("--num_windows_per_subset", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8140)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stage_configs() -> Dict[str, Dict[str, object]]:
    return {
        "stage1a_precontact_strict": {
            "title": "Stage-1A Precontact Strict",
            "logic": [
                "先用 `frame_phase`、`contact_graph` 和 `event_windows` 三路信息取最早接触帧。",
                "窗口必须在首个接触帧之前，并额外留出 `safety_margin=2` 帧缓冲。",
                "context 长度固定 8 帧，future 候选长度是 8 / 12 / 16，步长 4。",
                "context 每一帧都要求至少有一个可见物体，future 段要求主物体可见比例达到阈值。",
            ],
        },
        "stage1b_simple_dynamics": {
            "title": "Stage-1B Simple Dynamics",
            "logic": [
                "同样扫描 context=8 的窗口，但未来长度扩展到 8 / 12 / 16 / 24 / 41。",
                "不要求窗口一定在碰撞前，可以覆盖整段视频。",
                "future 段允许物体与环境接触，但不能与任意其他物体发生接触。",
                "物体-物体接触同时检查 `contact_graph` 和 `event_windows`，任一路命中都会被过滤。",
            ],
        },
    }


def count_bucket_from_path(path_text: str) -> str:
    for part in Path(path_text).parts:
        if part.startswith("count_"):
            return part
    return "unknown"


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_entries(accepted: Sequence[dict], limit: int, seed: int) -> List[dict]:
    items = list(accepted)
    if len(items) <= limit:
        return items
    random.Random(seed).shuffle(items)
    picked: List[dict] = []
    seen_samples = set()
    for item in items:
        sample_dir = str(item.get("sample_dir", ""))
        if sample_dir in seen_samples:
            continue
        picked.append(item)
        seen_samples.add(sample_dir)
        if len(picked) >= limit:
            return picked
    for item in items:
        if item in picked:
            continue
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def load_frames(frame_paths: Sequence[str], prefix: str) -> List[Image.Image]:
    return [annotate_frame(Image.open(path).convert("RGB"), f"{prefix} {idx}") for idx, path in enumerate(frame_paths)]


def build_sample_report(stage_name: str, item: dict, dst_dir: Path) -> dict:
    ensure_dir(dst_dir)
    window_dir = Path(item["out_dir"])
    meta = json.loads((window_dir / "pair_meta.json").read_text(encoding="utf-8"))
    payload = np.load(window_dir / "state_pair.npz")

    x_idx = np.asarray(payload["x_frame_indices"]).astype(np.int32)
    y_idx = np.asarray(payload["y_frame_indices"]).astype(np.int32)
    if "y_state_raw" in payload:
        y_state_raw = np.asarray(payload["y_state_raw"]).astype(np.float32)
    else:
        state_raw = np.asarray(payload["state_raw"]).astype(np.float32)
        y_state_raw = state_raw[int(y_idx[0]) : int(y_idx[-1]) + 1]
    if "y_state_norm" in payload:
        y_state_norm = np.asarray(payload["y_state_norm"]).astype(np.float32)
    elif "y_state" in payload:
        y_state_norm = np.asarray(payload["y_state"]).astype(np.float32)
    else:
        raise KeyError(f"No y_state_norm/y_state found in {window_dir / 'state_pair.npz'}")
    context_frames = load_frames(meta["x_frame_paths"], "context")
    future_frames = load_frames(meta["y_frame_paths"], "future")
    future_overlay_frames = draw_state_overlay(
        frames=[frame.copy() for frame in future_frames],
        y_state_raw=y_state_raw,
        object_colors=STATE_COLORS,
    )

    make_strip(context_frames, dst_dir / "context_strip.png")
    make_strip(future_frames, dst_dir / "future_strip.png")
    make_strip(future_overlay_frames, dst_dir / "future_overlay_strip.png")
    make_gif(context_frames, dst_dir / "context.gif")
    make_gif(future_frames, dst_dir / "future.gif")
    make_gif(future_overlay_frames, dst_dir / "future_overlay.gif")
    state_plot_names = save_state_plots(
        y_state_raw=y_state_raw,
        y_state_norm=y_state_norm,
        objects=meta.get("objects", []),
        out_dir=dst_dir,
    )

    source_video_path = Path(meta["source_sample_dir"]) / "videos" / "rgb.mp4"
    has_source_video = source_video_path.exists()
    if has_source_video:
        copy_or_symlink(source_video_path, dst_dir / "source_rgb.mp4")

    report_meta = {
        "stage_name": stage_name,
        "window_dir": str(window_dir),
        "source_sample_dir": meta.get("source_sample_dir", ""),
        "count_bucket": count_bucket_from_path(meta.get("source_sample_dir", "")),
        "start_index": int(meta.get("start_index", 0)),
        "context_len": int(meta["context_len"]),
        "future_len": int(meta["future_len"]),
        "num_objects": int(np.asarray(payload["object_ids"]).shape[0]),
        "main_object_index": int(meta.get("main_object_index", 0)),
        "future_main_visibility_ratio": float(meta.get("future_main_visibility_ratio", 0.0)),
        "future_main_visibility_threshold": float(meta.get("future_main_visibility_threshold", 0.0)),
        "first_contact_frame": meta.get("first_contact_frame"),
        "valid_end": meta.get("valid_end"),
        "subset_rule": str(meta.get("subset_rule", "")),
        "objects": meta.get("objects", []),
        "y_state_raw": y_state_raw.tolist(),
        "object_object_contact_filter": meta.get("object_object_contact_filter"),
        "window_interactions": meta.get("window_interactions"),
        "has_source_video": has_source_video,
        "state_plot_names": state_plot_names,
    }

    (dst_dir / "sample_info.json").write_text(
        json.dumps(report_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (dst_dir / "index.html").write_text(build_sample_html(report_meta), encoding="utf-8")
    return report_meta


def logic_html(lines: Sequence[str]) -> str:
    return "".join(f"<li>{html.escape(line)}</li>" for line in lines)


def object_contact_html(info: dict | None) -> str:
    if not info:
        return '<div class="muted">没有额外记录 object-object 过滤细节。</div>'
    rows = []
    for key in ("graph_hit", "event_hit", "any_hit"):
        rows.append(
            f'<div class="metric-card compact"><span class="metric-label">{html.escape(key)}</span><span class="metric-value">{html.escape(str(info.get(key)))}</span></div>'
        )
    return '<div class="metric-grid">' + "".join(rows) + "</div>"


def build_sample_html(report_meta: dict) -> str:
    plot_cards = "".join(
        f"""
<div class="plot-card">
  <img src="{html.escape(name)}" alt="{html.escape(name)}">
</div>
"""
        for name in report_meta["state_plot_names"]
    )
    first_contact = report_meta["first_contact_frame"]
    valid_end = report_meta["valid_end"]
    source_video_html = (
        """
        <div class="media-card" style="margin-top:16px;">
          <h3>Source RGB Video</h3>
          <video controls preload="metadata" src="source_rgb.mp4"></video>
        </div>
        """
        if report_meta["has_source_video"]
        else ""
    )
    contact_html = object_contact_html(report_meta.get("object_object_contact_filter"))
    interactions = report_meta.get("window_interactions") or {}
    future_window = interactions.get("future_window", {}) if isinstance(interactions, dict) else {}
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(report_meta['stage_name'])}</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf3;
      --panel2: #f7efe3;
      --ink: #1c1814;
      --muted: #6f665d;
      --line: #dccfbe;
      --accent: #0f766e;
      --accent2: #b45309;
      --shadow: rgba(40, 28, 16, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(180,83,9,0.10), transparent 24%),
        var(--bg);
    }}
    .page {{
      width: min(1480px, calc(100vw - 28px));
      margin: 18px auto 40px;
    }}
    .hero, .section, .plot-card, .state-table-card {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 18px 40px var(--shadow);
    }}
    .hero, .section {{ padding: 22px 24px; margin-bottom: 18px; }}
    .hero h1, .section h2, .media-card h3 {{ margin: 0; }}
    .hero h1 {{ font-size: 30px; line-height: 1.08; }}
    .hero p, .muted {{ color: var(--muted); }}
    .metric-grid, .grid-2, .object-grid, .plot-grid {{ display: grid; gap: 16px; }}
    .metric-grid {{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }}
    .metric-card {{
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.55);
      min-height: 88px;
    }}
    .metric-card.compact {{ min-height: 0; }}
    .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-value {{
      display: block;
      font-weight: 700;
      font-size: 22px;
      word-break: break-word;
    }}
    .metric-value.small {{ font-size: 14px; line-height: 1.5; font-weight: 500; }}
    .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); }}
    .media-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.48);
    }}
    .media-card img, .media-card video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #0f1115;
    }}
    .media-card a {{ color: var(--accent); }}
    .object-grid {{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }}
    .object-card {{
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.5);
    }}
    .object-head {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 4px 10px;
      border-left: 5px solid var(--accent);
      border-radius: 8px;
      font-weight: 700;
      background: rgba(255,255,255,0.6);
    }}
    .plot-grid {{ grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }}
    .plot-card {{ padding: 10px; }}
    .plot-card img {{ width: 100%; border-radius: 12px; display: block; background: white; }}
    .state-table-card {{ padding: 16px; margin-bottom: 16px; }}
    .state-table-card h4 {{ margin: 0 0 12px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      min-width: 760px;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: center; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    ul {{ margin: 0; padding-left: 18px; line-height: 1.7; color: #463d35; }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(report_meta['stage_name'])}</h1>
      <p>窗口目录: <code>{html.escape(report_meta['window_dir'])}</code></p>
      <p>样本目录: <code>{html.escape(report_meta['source_sample_dir'])}</code></p>
      <p>subset rule: {html.escape(report_meta['subset_rule'])}</p>
    </section>

    <section class="section">
      <h2>1. 窗口筛选摘要</h2>
      <div class="metric-grid">
        <div class="metric-card"><span class="metric-label">count bucket</span><span class="metric-value">{html.escape(report_meta['count_bucket'])}</span></div>
        <div class="metric-card"><span class="metric-label">start index</span><span class="metric-value">{report_meta['start_index']}</span></div>
        <div class="metric-card"><span class="metric-label">context len</span><span class="metric-value">{report_meta['context_len']}</span></div>
        <div class="metric-card"><span class="metric-label">future len</span><span class="metric-value">{report_meta['future_len']}</span></div>
        <div class="metric-card"><span class="metric-label">objects</span><span class="metric-value">{report_meta['num_objects']}</span></div>
        <div class="metric-card"><span class="metric-label">main object index</span><span class="metric-value">{report_meta['main_object_index']}</span></div>
        <div class="metric-card"><span class="metric-label">future main vis ratio</span><span class="metric-value">{report_meta['future_main_visibility_ratio']:.3f}</span></div>
        <div class="metric-card"><span class="metric-label">vis threshold</span><span class="metric-value">{report_meta['future_main_visibility_threshold']:.3f}</span></div>
        <div class="metric-card"><span class="metric-label">first contact frame</span><span class="metric-value">{html.escape(str(first_contact))}</span></div>
        <div class="metric-card"><span class="metric-label">valid end</span><span class="metric-value">{html.escape(str(valid_end))}</span></div>
        <div class="metric-card"><span class="metric-label">object count</span><span class="metric-value">{html.escape(str(interactions.get('object_count', 'n/a')))}</span></div>
        <div class="metric-card"><span class="metric-label">future collision episodes</span><span class="metric-value">{html.escape(str(future_window.get('collision_episode_count', 'n/a')))}</span></div>
        <div class="metric-card"><span class="metric-label">future collision type</span><span class="metric-value">{html.escape(str(future_window.get('collision_type_bucket', 'n/a')))}</span></div>
        <div class="metric-card"><span class="metric-label">future bucket</span><span class="metric-value small">{html.escape(str(interactions.get('future_bucket', 'n/a')))}</span></div>
      </div>
    </section>

    <section class="section">
      <h2>2. Context / Future / Overlay</h2>
      <div class="grid-2">
        <div class="media-card">
          <h3>Context 输入帧</h3>
          <img src="context_strip.png" alt="context strip">
          <p class="muted"><a href="context.gif">打开 context GIF</a></p>
        </div>
        <div class="media-card">
          <h3>Future GT 帧</h3>
          <img src="future_strip.png" alt="future strip">
          <p class="muted"><a href="future.gif">打开 future GIF</a></p>
        </div>
      </div>
      <div class="media-card" style="margin-top:16px;">
        <h3>Oracle State Overlay</h3>
        <img src="future_overlay_strip.png" alt="future overlay strip">
        <p class="muted">彩框按 object index 着色，框和中心点来自未来段的原始 `u,v,w,h,vis`；状态曲线里同时保留 raw 和 normalized 两套值。</p>
        <p class="muted"><a href="future_overlay.gif">打开 overlay GIF</a></p>
      </div>
      {source_video_html}
    </section>

    <section class="section">
      <h2>3. Stage1B Contact Filter</h2>
      {contact_html}
    </section>

    <section class="section">
      <h2>4. 物体静态信息</h2>
      <div class="object-grid">
        {objects_html(report_meta['objects'], report_meta['num_objects'])}
      </div>
    </section>

    <section class="section">
      <h2>5. 未来状态曲线</h2>
      <div class="plot-grid">
        {plot_cards}
      </div>
      <p class="muted">蓝线是原始像素/深度值，橙线是窗口里实际保存的归一化状态。</p>
    </section>

    <section class="section">
      <h2>6. 条件张量明细</h2>
      {state_table_html(np.asarray(report_meta['y_state_raw']), report_meta['objects'])}
    </section>
  </div>
</body>
</html>
"""


def build_index_html(portal_rel: str, stage_cards: Dict[str, List[dict]]) -> str:
    config = stage_configs()
    sections = []
    for stage_name, cards in stage_cards.items():
        logic = logic_html(config[stage_name]["logic"])
        cards_html = "".join(
            f"""
<article class="sample-card">
  <div class="sample-thumb">
    <img src="{html.escape(card['rel_dir'])}/future_overlay_strip.png" alt="{html.escape(card['title'])}">
  </div>
  <div class="sample-body">
    <h3>{html.escape(card['title'])}</h3>
    <p>{html.escape(card['summary'])}</p>
    <div class="badge-row">
      <span class="badge">{html.escape(card['count_bucket'])}</span>
      <span class="badge">ctx {card['context_len']}</span>
      <span class="badge">fut {card['future_len']}</span>
      <span class="badge">{card['num_objects']} objects</span>
    </div>
    <p><a class="button" href="{html.escape(card['rel_dir'])}/index.html">打开样本页</a></p>
  </div>
</article>
"""
            for card in cards
        )
        sections.append(
            f"""
<section class="section">
  <div class="section-head">
    <div>
      <h2>{html.escape(str(config[stage_name]['title']))}</h2>
      <p class="muted">输出目录: <code>{html.escape(portal_rel)}/{html.escape(stage_name)}</code></p>
    </div>
  </div>
  <div class="logic-box">
    <h3>筛选逻辑</h3>
    <ul>{logic}</ul>
  </div>
  <div class="sample-grid">
    {cards_html}
  </div>
</section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Stage1 Subset Preview</title>
  <style>
    :root {{
      --bg: #efe8dc;
      --panel: #fffaf2;
      --panel2: #f9f0e1;
      --ink: #1b1612;
      --muted: #6d6458;
      --line: #dbcbb5;
      --accent: #9a3412;
      --accent2: #065f46;
      --shadow: rgba(49, 32, 17, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(154,52,18,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(6,95,70,0.10), transparent 24%),
        var(--bg);
    }}
    .page {{
      max-width: 1420px;
      margin: 0 auto;
      padding: 28px 22px 60px;
    }}
    .hero, .section {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 42px var(--shadow);
    }}
    .hero {{
      padding: 28px 30px;
      margin-bottom: 22px;
    }}
    .hero h1, .section h2, .logic-box h3, .sample-body h3 {{ margin: 0; }}
    .hero h1 {{ font-size: 36px; line-height: 1.06; letter-spacing: -0.02em; }}
    .hero p, .muted {{ color: var(--muted); }}
    .section {{
      padding: 22px 24px;
      margin-bottom: 20px;
    }}
    .sample-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
    }}
    .sample-card {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.46);
    }}
    .sample-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 14px;
      background: #0c0e12;
    }}
    .sample-body p {{ line-height: 1.55; }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 14px;
    }}
    .badge {{
      background: rgba(255,255,255,0.78);
      color: #694d33;
      border: 1px solid #dcc7aa;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
    }}
    .button {{
      display: inline-block;
      text-decoration: none;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 14px;
    }}
    .logic-box {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.54);
      margin-bottom: 18px;
    }}
    ul {{ margin: 10px 0 0; padding-left: 18px; line-height: 1.7; color: #4b4339; }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Stage1A / Stage1B Subset Preview</h1>
      <p>这个页面把 `build_stage1_subsets.py` 产出的两个子集逻辑重新整理成可核对的样本浏览页。每条样本页都包含 context/future RGB、future overlay、每个物体的 9 维状态曲线，以及窗口级筛选元数据。</p>
      <p>当前 portal 相对目录: <code>{html.escape(portal_rel)}</code></p>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def start_server(output_dir: Path, host: str, port: int) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is already in use: {exc}") from exc

    log_path = output_dir / "server.log"
    command = [
        "python3",
        "-m",
        "http.server",
        str(port),
        "--bind",
        host,
        "--directory",
        str(output_dir.parent),
    ]
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return str(process.pid)


def main() -> None:
    args = parse_args()
    config = stage_configs()
    if not args.subset_root.exists():
        raise FileNotFoundError(f"subset_root does not exist: {args.subset_root}")

    ensure_dir(args.output_dir)
    stage_cards: Dict[str, List[dict]] = {}

    for stage_offset, stage_name in enumerate(config):
        manifest_path = args.subset_root / stage_name / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        manifest = load_manifest(manifest_path)
        selected = select_entries(
            manifest.get("accepted", []),
            limit=int(args.num_windows_per_subset),
            seed=int(args.seed) + stage_offset,
        )
        stage_dir = args.output_dir / stage_name
        ensure_dir(stage_dir)
        cards: List[dict] = []
        for sample_idx, item in enumerate(selected):
            sample_dir = stage_dir / f"sample_{sample_idx:02d}"
            report_meta = build_sample_report(stage_name, item, sample_dir)
            cards.append(
                {
                    "title": f"{config[stage_name]['title']} #{sample_idx + 1}",
                    "summary": (
                        f"{report_meta['count_bucket']} | start={report_meta['start_index']} | "
                        f"future={report_meta['future_len']} | vis={report_meta['future_main_visibility_ratio']:.3f}"
                    ),
                    "rel_dir": f"{stage_name}/sample_{sample_idx:02d}",
                    "count_bucket": report_meta["count_bucket"],
                    "context_len": report_meta["context_len"],
                    "future_len": report_meta["future_len"],
                    "num_objects": report_meta["num_objects"],
                }
            )
        stage_cards[stage_name] = cards

    (args.output_dir / "index.html").write_text(
        build_index_html(args.output_dir.name, stage_cards),
        encoding="utf-8",
    )

    if args.serve:
        pid = start_server(args.output_dir, args.host, int(args.port))
        host_for_user = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        print(f"server_pid={pid}")
        print(f"url=http://{host_for_user}:{int(args.port)}/{args.output_dir.name}/index.html")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
