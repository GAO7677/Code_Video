#!/usr/bin/env python3
"""Generate a local HTML report for one oracle-state adapter training sample."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ADAPTER_ROOT = SCRIPT_DIR.parent
TRAIN0419_ROOT = STATE_ADAPTER_ROOT.parent
if str(STATE_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(STATE_ADAPTER_ROOT))
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

from state_adapter_dataset import OracleStateWindowDataset


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


def format_shape(shape: Sequence[int]) -> str:
    return "[" + ", ".join(str(int(x)) for x in shape) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one oracle-state adapter training sample as a local HTML page."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
        ),
    )
    parser.add_argument("--height", type=int, default=736)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--index", type=int, default=-1, help="-1 means auto-pick an illustrative sample.")
    parser.add_argument("--prefer_future_len", type=int, default=13)
    parser.add_argument("--min_objects", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/tmp/oracle_state_adapter_vis"),
    )
    parser.add_argument("--title", type=str, default="Oracle State Adapter Sample")
    parser.add_argument("--compute_loss", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--wan_root",
        type=Path,
        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"),
    )
    parser.add_argument(
        "--preset_tv2v_root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_ctx49_736x1280_lora"),
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_or_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def load_meta_and_payload(window_dir: Path) -> tuple[dict, np.lib.npyio.NpzFile]:
    meta = json.loads((window_dir / "pair_meta.json").read_text(encoding="utf-8"))
    payload = np.load(window_dir / "state_pair.npz")
    return meta, payload


def select_sample_index(
    dataset: OracleStateWindowDataset,
    prefer_future_len: int,
    min_objects: int,
) -> int:
    best_index = 0
    best_score = None
    for idx, window_dir in enumerate(dataset.window_dirs):
        meta = json.loads((window_dir / "pair_meta.json").read_text(encoding="utf-8"))
        payload = np.load(window_dir / "state_pair.npz")
        future_len = int(meta.get("future_len", 0))
        num_objects = int(payload["object_ids"].shape[0])
        score = (
            0 if future_len == prefer_future_len else abs(future_len - prefer_future_len) + 100,
            0 if num_objects >= min_objects else min_objects - num_objects,
            -num_objects,
            idx,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_index = idx
    return best_index


def load_frame_sequence(frame_paths: Sequence[str]) -> List[Image.Image]:
    return [Image.open(path).convert("RGB") for path in frame_paths]


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
        image = annotate_frame(frame, f"future {t}")
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


def format_loss_html(loss_info: dict | None) -> str:
    if loss_info is None:
        return '<div class="muted">Loss not computed for this report.</div>'
    if "error" in loss_info:
        return f'<div class="error">Loss computation failed: {html.escape(loss_info["error"])}</div>'
    return f"""
<div class="metric-grid">
  <div class="metric-card"><span class="metric-label">single-sample loss</span><span class="metric-value">{loss_info['loss']:.6f}</span></div>
  <div class="metric-card"><span class="metric-label">seed</span><span class="metric-value">{loss_info['seed']}</span></div>
  <div class="metric-card"><span class="metric-label">device</span><span class="metric-value">{html.escape(str(loss_info['device']))}</span></div>
  <div class="metric-card"><span class="metric-label">preset TV2V ckpt</span><span class="metric-value small">{html.escape(str(loss_info['preset_lora_path']))}</span></div>
</div>
"""


def shape_metrics_html(shape_info: dict | None) -> str:
    if shape_info is None:
        return '<div class="muted">Shape flow not computed for this report.</div>'
    if "error" in shape_info:
        return f'<div class="error">Shape flow computation failed: {html.escape(shape_info["error"])}</div>'
    metrics = shape_info["metrics"]
    cards = [
        ("raw future frames", metrics["raw_future_frames"]),
        ("total latent frames", metrics["latent_total_frames"]),
        ("clean prefix latent frames", metrics["clean_prefix_latent_frames"]),
        ("future latent frames", metrics["future_latent_frames"]),
        ("spatial tokens / frame", metrics["spatial_tokens_per_frame"]),
        ("flattened seq len", metrics["sequence_len_after_flatten"]),
    ]
    return '<div class="metric-grid">' + "".join(
        f'<div class="metric-card"><span class="metric-label">{html.escape(str(label))}</span><span class="metric-value">{html.escape(str(value))}</span></div>'
        for label, value in cards
    ) + "</div>"


def shape_flow_html(shape_info: dict | None) -> str:
    if shape_info is None:
        return '<div class="muted">Shape flow not computed for this report.</div>'
    if "error" in shape_info:
        return f'<div class="error">Shape flow computation failed: {html.escape(shape_info["error"])}</div>'

    rows = []
    for row in shape_info["rows"]:
        rows.append(
            f"""
<tr>
  <td>{html.escape(str(row['stage']))}</td>
  <td><code>{html.escape(str(row['tensor']))}</code></td>
  <td><code>{html.escape(str(row['shape']))}</code></td>
  <td>{html.escape(str(row['explanation']))}</td>
</tr>
"""
        )
    return f"""
<div class="muted" style="margin-bottom:14px; line-height:1.7;">
  {html.escape(shape_info['summary'])}
</div>
<div class="table-wrap">
  <table class="shape-table">
    <thead>
      <tr>
        <th>stage</th>
        <th>tensor</th>
        <th>shape</th>
        <th>explanation</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
"""


def build_html(
    title: str,
    report_meta: dict,
    state_plot_names: Sequence[str],
    shape_info: dict | None,
    loss_info: dict | None,
) -> str:
    plot_cards = "".join(
        f"""
<div class="plot-card">
  <img src="{html.escape(name)}" alt="{html.escape(name)}">
</div>
"""
        for name in state_plot_names
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0e1116;
      --card: #151a21;
      --card2: #1c222c;
      --text: #eef2f7;
      --muted: #97a1af;
      --line: #2b3340;
      --accent: #3bc9db;
      --accent2: #ffd166;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(59,201,219,0.15), transparent 28%),
        radial-gradient(circle at top right, rgba(255,209,102,0.12), transparent 22%),
        var(--bg);
    }}
    .page {{
      width: min(1500px, calc(100vw - 32px));
      margin: 24px auto 40px;
    }}
    .hero, .section, .plot-card, .state-table-card {{
      background: linear-gradient(180deg, var(--card), var(--card2));
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.2);
    }}
    .hero {{
      padding: 24px 28px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .hero p {{
      color: var(--muted);
      line-height: 1.6;
      margin: 8px 0 0;
    }}
    .section {{
      padding: 20px 22px;
      margin-bottom: 18px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 20px;
    }}
    .muted {{ color: var(--muted); }}
    .error {{ color: #ff8787; }}
    .metric-grid, .grid-2, .object-grid, .plot-grid {{
      display: grid;
      gap: 16px;
    }}
    .metric-grid {{
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    }}
    .metric-card {{
      padding: 14px 16px;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--line);
      border-radius: 14px;
      min-height: 96px;
    }}
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
    .metric-value.small {{
      font-size: 14px;
      font-weight: 500;
      line-height: 1.5;
    }}
    .grid-2 {{
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
    }}
    .media-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.02);
    }}
    .media-card img {{
      width: 100%;
      border-radius: 10px;
      display: block;
      background: #0a0c10;
    }}
    .media-card a {{
      color: var(--accent);
    }}
    .object-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .object-card {{
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.03);
    }}
    .object-head {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 4px 10px;
      border-left: 5px solid var(--accent);
      background: rgba(255,255,255,0.02);
      border-radius: 8px;
      font-weight: 700;
    }}
    .plot-grid {{
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    }}
    .plot-card {{
      padding: 10px;
    }}
    .plot-card img {{
      width: 100%;
      border-radius: 12px;
      display: block;
      background: white;
    }}
    .state-table-card {{
      padding: 16px;
      margin-bottom: 16px;
    }}
    .state-table-card h4 {{
      margin: 0 0 12px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      border-collapse: collapse;
      min-width: 760px;
      width: 100%;
      font-size: 13px;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: center;
    }}
    .shape-table th:nth-child(4),
    .shape-table td:nth-child(4) {{
      text-align: left;
      min-width: 520px;
      white-space: normal;
      line-height: 1.6;
    }}
    .pipeline {{
      display: grid;
      gap: 10px;
    }}
    .pipe-step {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255,255,255,0.02);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p>{html.escape(report_meta['prompt'])}</p>
      <p class="muted">窗口目录: <code>{html.escape(report_meta['window_dir'])}</code></p>
    </section>

    <section class="section">
      <h2>1. 这条样本怎么进模型</h2>
      <div class="pipeline">
        <div class="pipe-step"><strong>输入视频</strong>: 原始窗口长度是 <code>C+K={report_meta['context_len']}+{report_meta['future_len']}</code>，训练时整段视频都会进 VAE / DiT。</div>
        <div class="pipe-step"><strong>Context</strong>: 前 <code>C={report_meta['context_len']}</code> 帧 RGB 作为干净 prefix，不加噪，作为 <code>context_video</code> 写回 latent 前缀。</div>
        <div class="pipe-step"><strong>Condition</strong>: 后 <code>K={report_meta['future_len']}</code> 帧每个物体的 9 维状态 <code>[u,v,d,w,h,du,dv,dd,vis]</code> 来自 <code>anchor_targets.npz + metadata.json</code>，先变成 object token，再按帧 pool，再做 temporal encoder，得到 future-aligned 条件 token。</div>
        <div class="pipe-step"><strong>调制位置</strong>: 条件 token 不替代 Wan latent，只通过 adapter 在每个 block 后对 future 帧 token 做 frame-aligned modulation；context 部分调制为零。</div>
        <div class="pipe-step"><strong>GT 与 Loss</strong>: GT 是 future RGB 对应的 future latent。训练目标仍是 Wan 的 flow-matching / denoising 目标，loss 只在 future 帧上计算，不在 context 帧上计算。</div>
      </div>
    </section>

    <section class="section">
      <h2>2. 样本摘要</h2>
      <div class="metric-grid">
        <div class="metric-card"><span class="metric-label">context len</span><span class="metric-value">{report_meta['context_len']}</span></div>
        <div class="metric-card"><span class="metric-label">future len</span><span class="metric-value">{report_meta['future_len']}</span></div>
        <div class="metric-card"><span class="metric-label">objects</span><span class="metric-value">{report_meta['num_objects']}</span></div>
        <div class="metric-card"><span class="metric-label">raw resolution</span><span class="metric-value">{report_meta['raw_resolution']}</span></div>
        <div class="metric-card"><span class="metric-label">train resize</span><span class="metric-value">{report_meta['train_resolution']}</span></div>
        <div class="metric-card"><span class="metric-label">source sample</span><span class="metric-value small">{html.escape(report_meta['source_sample_dir'])}</span></div>
      </div>
    </section>

    <section class="section">
      <h2>3. 输入 / 条件 / GT</h2>
      <div class="grid-2">
        <div class="media-card">
          <h3>Context 输入帧</h3>
          <img src="context_strip.png" alt="context strip">
          <p class="muted">训练里这部分会先编码成 clean prefix latents，保持干净不加噪。</p>
          <p><a href="context.gif">打开 context GIF</a></p>
        </div>
        <div class="media-card">
          <h3>Future GT 帧</h3>
          <img src="future_strip.png" alt="future strip">
          <p class="muted">这是监督目标对应的 future RGB，可视化时用原始帧；训练时会 resize 到 {html.escape(report_meta['train_resolution'])} 再进 VAE。</p>
          <p><a href="future.gif">打开 future GIF</a></p>
        </div>
      </div>
      <div class="media-card" style="margin-top:16px;">
        <h3>Oracle 条件画在 future GT 上</h3>
        <img src="future_overlay_strip.png" alt="future overlay strip">
        <p class="muted">彩框来自 oracle state 里的 <code>u,v,w,h,vis</code>。这部分正是 adapter 的条件输入之一，另外 <code>du,dv,dd</code> 也会送进状态 MLP。</p>
        <p><a href="future_overlay.gif">打开 overlay GIF</a></p>
      </div>
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
      <p class="muted">蓝线是原始像素/深度坐标，橙线是训练中实际喂给 adapter 的归一化版本。</p>
    </section>

    <section class="section">
      <h2>6. 条件张量明细</h2>
      {state_table_html(np.asarray(report_meta['y_state_raw']), report_meta['objects'])}
    </section>

    <section class="section">
      <h2>7. Condition Shape Flow</h2>
      <p class="muted">这一段专门回答“future state 条件经过每个模块后 shape 如何变化”。这里给的是当前实现的真实 shape，而不是抽象示意图。</p>
      {shape_metrics_html(shape_info)}
      <div style="margin-top:16px;">
        {shape_flow_html(shape_info)}
      </div>
    </section>

    <section class="section">
      <h2>8. 单样本 Loss</h2>
      {format_loss_html(loss_info)}
      <p class="muted">这里的 loss 是当前实现下一次真实前向的随机 flow-matching 损失。由于 timestep 和噪声是随机采样的，所以不同 seed 的数值会变化。</p>
    </section>
  </div>
</body>
</html>
"""


def compute_single_sample_loss(args: argparse.Namespace, index: int) -> dict:
    import torch

    from train import build_wan22_ti2v5b_model_paths, find_tokenizer_path
    from train_state_adapter import StateAwareWanTrainingModule, resolve_latest_checkpoint

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = OracleStateWindowDataset(
        dataset_root=str(args.dataset_root),
        height=args.height,
        width=args.width,
        dataset_repeat=1,
    )
    sample = dataset[index]
    preset_lora_path = resolve_latest_checkpoint(str(args.preset_tv2v_root))
    try:
        module = StateAwareWanTrainingModule(
            model_paths=build_wan22_ti2v5b_model_paths(str(args.wan_root)),
            tokenizer_path=find_tokenizer_path(str(args.wan_root)),
            trainable_models="animate_adapter",
            preset_lora_path=preset_lora_path,
            preset_lora_model="dit",
            device=args.device,
        )
        module.to(device=args.device)
        module.pipe.animate_adapter.to(device=args.device, dtype=module.pipe.torch_dtype)
        module.eval()
        loss = module.forward(sample).detach().float().item()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        return {
            "loss": float(loss),
            "seed": int(args.seed),
            "device": str(args.device),
            "preset_lora_path": str(preset_lora_path),
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "seed": int(args.seed),
            "device": str(args.device),
            "preset_lora_path": str(preset_lora_path),
        }
    finally:
        if "module" in locals():
            del module
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def compute_shape_flow(args: argparse.Namespace, index: int) -> dict:
    import torch

    from train import build_wan22_ti2v5b_model_paths, find_tokenizer_path
    from train_state_adapter import StateAwareWanTrainingModule, resolve_latest_checkpoint

    dataset = OracleStateWindowDataset(
        dataset_root=str(args.dataset_root),
        height=args.height,
        width=args.width,
        dataset_repeat=1,
    )
    sample = dataset[index]
    preset_lora_path = resolve_latest_checkpoint(str(args.preset_tv2v_root))
    try:
        module = StateAwareWanTrainingModule(
            model_paths=build_wan22_ti2v5b_model_paths(str(args.wan_root)),
            tokenizer_path=find_tokenizer_path(str(args.wan_root)),
            trainable_models="animate_adapter",
            preset_lora_path=preset_lora_path,
            preset_lora_model="dit",
            device=args.device,
        )
        module.to(device=args.device)
        module.pipe.animate_adapter.to(device=args.device, dtype=module.pipe.torch_dtype)
        module.eval()

        inputs = module.get_pipeline_inputs(sample)
        inputs = module.transfer_data_to_device(inputs, module.pipe.device, module.pipe.torch_dtype)
        for unit in module.pipe.units:
            inputs = module.pipe.unit_runner(unit, module.pipe, *inputs)

        shared = inputs[0]
        adapter = module.pipe.animate_adapter
        oracle_state = shared["oracle_state"]
        raw_oracle_state_shape = tuple(oracle_state.shape)

        if oracle_state.dim() == 3:
            oracle_state_b = oracle_state.unsqueeze(0)
        else:
            oracle_state_b = oracle_state

        batch, raw_frames, num_objects, state_dim = oracle_state_b.shape
        dynamic_tokens = adapter.state_mlp(oracle_state_b)
        valid_mask = oracle_state_b[..., -1] > 0.5
        frame_tokens_before_temporal = adapter.frame_pool(dynamic_tokens, valid_mask=valid_mask)
        frame_tokens_after_temporal = adapter.temporal_encoder(frame_tokens_before_temporal)

        latents = shared["latents"]
        input_latents = shared["input_latents"]
        clean_prefix_latents = shared["clean_prefix_latents"]
        clean_prefix_len = int(shared["num_clean_prefix_latents"])
        patched = module.pipe.dit.patchify(latents)
        f, h, w = map(int, patched.shape[2:])
        spatial_tokens_per_frame = int(h * w)
        sequence_len = int(f * h * w)
        future_latent_frames = int(f - clean_prefix_len)

        future_plan_tokens = adapter.encode_future_plan(
            oracle_state=oracle_state,
            target_frames=future_latent_frames,
        )
        modulation = adapter.modulation_heads[0](future_plan_tokens)
        gamma, beta = modulation.chunk(2, dim=-1)
        zeros = latents.new_zeros((batch, clean_prefix_len, gamma.shape[-1]))
        gamma_full = torch.cat([zeros, gamma], dim=1)
        beta_full = torch.cat([zeros, beta], dim=1)
        flattened_hidden = patched.reshape(patched.shape[0], patched.shape[1], sequence_len).transpose(1, 2).contiguous()
        frame_major_hidden = flattened_hidden.reshape(batch, f, spatial_tokens_per_frame, adapter.dit_dim)

        rows = [
            {
                "stage": "dataset",
                "tensor": "oracle_state",
                "shape": format_shape(raw_oracle_state_shape),
                "explanation": "从数据集读出的 future 9 维状态，按 [K, N, 9] 排列。这里 K=13, N=3。",
            },
            {
                "stage": "adapter input",
                "tensor": "oracle_state (batched)",
                "shape": format_shape(tuple(oracle_state_b.shape)),
                "explanation": "进入 encode_future_plan 后补 batch 维，变成 [B, K, N, 9]。",
            },
            {
                "stage": "state mlp",
                "tensor": "dynamic_tokens",
                "shape": format_shape(tuple(dynamic_tokens.shape)),
                "explanation": "9 维动态状态经过两层 MLP，得到每帧每物体一个动态 token，不再引入 object/source/category 等静态身份信息。",
            },
            {
                "stage": "frame pool",
                "tensor": "frame_tokens",
                "shape": format_shape(tuple(frame_tokens_before_temporal.shape)),
                "explanation": "同一帧内对 N 个物体的动态 token 做 attention pooling，压成每帧一个 token [B, K, 1024]。",
            },
            {
                "stage": "temporal encoder",
                "tensor": "frame_tokens_temporal",
                "shape": format_shape(tuple(frame_tokens_after_temporal.shape)),
                "explanation": "浅层 temporal transformer 在 raw future 时间轴上编码运动计划，shape 保持不变。",
            },
            {
                "stage": "latent encode",
                "tensor": "input_latents",
                "shape": format_shape(tuple(input_latents.shape)),
                "explanation": "整段 21 帧视频先过 Wan VAE，得到 [B, 48, F, H, W]。这条样本 raw 21 帧被压到 F=6 个 latent frame。",
            },
            {
                "stage": "context encode",
                "tensor": "clean_prefix_latents",
                "shape": format_shape(tuple(clean_prefix_latents.shape)),
                "explanation": "前 8 帧 context 单独过 VAE 得到 clean prefix latent。当前样本占 2 个 latent frame。",
            },
            {
                "stage": "time align",
                "tensor": "future_plan_tokens",
                "shape": format_shape(tuple(future_plan_tokens.shape)),
                "explanation": "把 raw future 时间轴上的 13 个 frame token 自适应池化到 Wan future latent 时间轴，得到 [B, F_fut, 1024]。当前 F_fut=4。",
            },
            {
                "stage": "patchify",
                "tensor": "patched_latents",
                "shape": format_shape(tuple(patched.shape)),
                "explanation": "Wan DiT 的 patchify 后张量是 [B, D_dit, F, h, w]。当前 D_dit=3072, grid=(6,23,40)。",
            },
            {
                "stage": "flatten",
                "tensor": "hidden_states",
                "shape": format_shape(tuple(flattened_hidden.shape)),
                "explanation": "把 patchify 结果展平为 DiT block 看到的 token 序列 [B, F*h*w, 3072]。当前序列长度是 5520。",
            },
            {
                "stage": "per-block head",
                "tensor": "modulation",
                "shape": format_shape(tuple(modulation.shape)),
                "explanation": "每个 block 各有一个 modulation head，把 [B, F_fut, 1024] 投影到 [B, F_fut, 2*3072]。",
            },
            {
                "stage": "split",
                "tensor": "gamma / beta",
                "shape": format_shape(tuple(gamma.shape)),
                "explanation": "沿最后一维切成 gamma 和 beta，两者都是 [B, F_fut, 3072]。",
            },
            {
                "stage": "prefix zero",
                "tensor": "gamma_full / beta_full",
                "shape": format_shape(tuple(gamma_full.shape)),
                "explanation": "在前面拼上 context 对应的全零 modulation，使 context 部分不受未来状态条件影响。当前变成 [B, 6, 3072]。",
            },
            {
                "stage": "frame-major reshape",
                "tensor": "hidden_states_frame_major",
                "shape": format_shape(tuple(frame_major_hidden.shape)),
                "explanation": "把 block hidden states 改写成 [B, F, spatial_tokens_per_frame, 3072]，便于做 frame-aligned broadcast modulation。",
            },
            {
                "stage": "broadcast mod",
                "tensor": "gamma.unsqueeze(2) / beta.unsqueeze(2)",
                "shape": format_shape((batch, f, 1, adapter.dit_dim)),
                "explanation": "扩一维后按同一帧内所有空间 token 共享同一个 gamma/beta，因此 future state 只调制时间位置，不替代空间内容。",
            },
        ]
        summary = (
            f"当前样本 raw future 有 {raw_frames} 帧，但 Wan VAE/patchify 后总 latent 时间长度只有 {f}，"
            f"其中前缀 context 占 {clean_prefix_len} 个 clean latent frame，所以真正需要 condition 的 future latent frame 只有 {future_latent_frames}。"
            f"因此 adapter 会把 [B,{raw_frames},1024] 的 raw-time frame token 压到 [B,{future_latent_frames},1024]，"
            f"再为每个 DiT block 生成 [B,{future_latent_frames},3072] 的 gamma/beta，并在前面补零成 [B,{f},3072]，只调制 future，不调制 context。"
        )
        return {
            "summary": summary,
            "metrics": {
                "raw_future_frames": int(raw_frames),
                "latent_total_frames": int(f),
                "clean_prefix_latent_frames": int(clean_prefix_len),
                "future_latent_frames": int(future_latent_frames),
                "spatial_tokens_per_frame": int(spatial_tokens_per_frame),
                "sequence_len_after_flatten": int(sequence_len),
            },
            "rows": rows,
            "device": str(args.device),
            "preset_lora_path": str(preset_lora_path),
        }
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "device": str(args.device),
            "preset_lora_path": str(preset_lora_path),
        }
    finally:
        if "module" in locals():
            del module
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    dataset = OracleStateWindowDataset(
        dataset_root=str(args.dataset_root),
        height=args.height,
        width=args.width,
        dataset_repeat=1,
    )
    index = args.index if args.index >= 0 else select_sample_index(dataset, args.prefer_future_len, args.min_objects)
    window_dir = dataset.window_dirs[index]
    meta, payload = load_meta_and_payload(window_dir)

    context_frames_raw = [annotate_frame(frame, f"context {t}") for t, frame in enumerate(load_frame_sequence(meta["x_frame_paths"]))]
    future_frames_raw = [annotate_frame(frame, f"future {t}") for t, frame in enumerate(load_frame_sequence(meta["y_frame_paths"]))]
    future_overlay_frames = draw_state_overlay(
        frames=load_frame_sequence(meta["y_frame_paths"]),
        y_state_raw=payload["y_state_raw"],
        object_colors=STATE_COLORS,
    )

    make_strip(context_frames_raw, args.output_dir / "context_strip.png")
    make_strip(future_frames_raw, args.output_dir / "future_strip.png")
    make_strip(future_overlay_frames, args.output_dir / "future_overlay_strip.png")
    make_gif(context_frames_raw, args.output_dir / "context.gif")
    make_gif(future_frames_raw, args.output_dir / "future.gif")
    make_gif(future_overlay_frames, args.output_dir / "future_overlay.gif")

    state_plot_names = save_state_plots(
        y_state_raw=payload["y_state_raw"],
        y_state_norm=payload["y_state_norm"],
        objects=meta.get("objects", []),
        out_dir=args.output_dir,
    )

    source_video_path = Path(meta["source_sample_dir"]) / "videos" / "rgb.mp4"
    if source_video_path.exists():
        copy_or_symlink(source_video_path, args.output_dir / "source_rgb.mp4")

    report_meta = {
        "prompt": meta.get("prompt", ""),
        "window_dir": str(window_dir),
        "source_sample_dir": meta.get("source_sample_dir", ""),
        "context_len": int(meta["context_len"]),
        "future_len": int(meta["future_len"]),
        "num_objects": int(payload["object_ids"].shape[0]),
        "objects": meta.get("objects", []),
        "raw_resolution": f"{meta.get('resolution', ['?', '?'])[0]} x {meta.get('resolution', ['?', '?'])[1]}",
        "train_resolution": f"{args.width} x {args.height}",
        "y_state_raw": payload["y_state_raw"].tolist(),
    }

    shape_info = compute_shape_flow(args, index)
    loss_info = compute_single_sample_loss(args, index) if args.compute_loss else None
    if loss_info is not None:
        (args.output_dir / "loss.json").write_text(
            json.dumps(loss_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (args.output_dir / "shape_flow.json").write_text(
        json.dumps(shape_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (args.output_dir / "sample_info.json").write_text(
        json.dumps(
            {
                "index": index,
                "window_dir": str(window_dir),
                "meta": report_meta,
                "shape_flow": shape_info,
                "loss": loss_info,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    html_text = build_html(
        title=args.title,
        report_meta=report_meta,
        state_plot_names=state_plot_names,
        shape_info=shape_info,
        loss_info=loss_info,
    )
    (args.output_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"index={index}")
    print(f"window_dir={window_dir}")
    print(f"output_dir={args.output_dir}")
    if loss_info is not None:
        print(json.dumps(loss_info, ensure_ascii=False))


if __name__ == "__main__":
    main()
