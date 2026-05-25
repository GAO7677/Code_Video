#!/usr/bin/env python3
# 用途：统计 hq_preview_0525 样本中每个物体在视频里的平均面积占比，并生成本地单页可视化。

from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


DATA_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/hq_preview_0525/train/rigid")
OUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/tmp/hq_preview_0525_area_ratio_portal")
MAX_FRAMES_FOR_GIF = 28


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_gif(video_path: Path, gif_path: Path, max_frames: int = MAX_FRAMES_FOR_GIF) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, frame_count // max_frames) if frame_count > max_frames else 1
    frames: list[Image.Image] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    if not frames:
        return None
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    return gif_path


def iter_sample_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for meta_path in sorted(root.rglob("meta.json")):
        sample_dir = meta_path.parent
        if (sample_dir / "physics" / "seg.npy").exists() and (sample_dir / "videos" / "rgb.mp4").exists():
            out.append(sample_dir)
    return out


def compute_object_area_rows(sample_dir: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    seg = np.load(sample_dir / "physics" / "seg.npy")
    if seg.ndim != 3:
        raise RuntimeError(f"Unexpected seg shape for {sample_dir}: {seg.shape}")
    total_pixels = float(seg.shape[1] * seg.shape[2])
    rows: list[dict[str, Any]] = []
    for obj in metadata.get("objects", []):
        if not isinstance(obj, dict):
            continue
        seg_id = int(obj.get("seg_id", -1))
        if seg_id < 0:
            continue
        mask = (seg == seg_id)
        frame_ratio = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float64) / total_pixels
        source_id = str(obj.get("source_object_id") or obj.get("object_id") or "unknown")
        rows.append(
            {
                "object_id": int(obj.get("object_id", -1)),
                "seg_id": seg_id,
                "source_object_id": source_id,
                "role": str(obj.get("role") or "unknown"),
                "motion_type": str(obj.get("motion_type") or obj.get("object_motion_type") or "unknown"),
                "avg_ratio": float(np.mean(frame_ratio)),
                "max_ratio": float(np.max(frame_ratio)),
                "visible_fraction": float(np.mean(frame_ratio > 0.0)),
            }
        )
    rows.sort(key=lambda x: (-x["avg_ratio"], x["object_id"]))
    return rows


def build_card(sample_dir: Path) -> str:
    meta = load_json(sample_dir / "meta.json")
    rows = compute_object_area_rows(sample_dir, meta)
    video_path = sample_dir / "videos" / "rgb.mp4"
    gif_name = f"{hashlib.md5(str(video_path).encode('utf-8')).hexdigest()[:12]}_{sample_dir.name}.gif"
    gif_path = OUT_ROOT / "_assets" / gif_name
    make_gif(video_path, gif_path)
    gif_src = os.path.relpath(gif_path, OUT_ROOT)
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{row['object_id']}</td>"
            f"<td>{row['seg_id']}</td>"
            f"<td>{html.escape(row['source_object_id'])}</td>"
            f"<td>{html.escape(row['role'])}</td>"
            f"<td>{html.escape(row['motion_type'])}</td>"
            f"<td>{row['avg_ratio'] * 100.0:.2f}%</td>"
            f"<td>{row['max_ratio'] * 100.0:.2f}%</td>"
            f"<td>{row['visible_fraction'] * 100.0:.1f}%</td>"
            "</tr>"
        )
    scene_comp = str(meta.get("scene_composition") or "unknown")
    count_bucket = str(meta.get("object_count_bucket") or "unknown")
    motion_category = str(meta.get("motion_category") or "unknown")
    caption = str(meta.get("caption") or "")
    return f"""
    <article class="card">
      <img class="media" src="{html.escape(gif_src)}" loading="lazy">
      <div class="body">
        <div class="name">{html.escape(sample_dir.name)}</div>
        <div class="meta">{html.escape(scene_comp)} | {html.escape(count_bucket)} | {html.escape(motion_category)}</div>
        <div class="caption"><strong>Caption:</strong> {html.escape(caption or '(empty)')}</div>
        <div class="caption"><strong>说明：</strong> 下表的平均面积占比是按全部视频帧求平均，单位是相对整张图像的百分比。</div>
        <table>
          <thead>
            <tr>
              <th>obj</th>
              <th>seg</th>
              <th>source</th>
              <th>role</th>
              <th>motion</th>
              <th>avg</th>
              <th>max</th>
              <th>visible</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
        <div class="path">{html.escape(str(sample_dir))}</div>
      </div>
    </article>
    """


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    sample_dirs = iter_sample_dirs(DATA_ROOT)
    cards = [build_card(sample_dir) for sample_dir in sample_dirs]
    html_text = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>hq_preview_0525 area ratio portal</title>
  <style>
    body {{ margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f5f6f8; color: #111; }}
    header {{ padding: 16px 18px; position: sticky; top: 0; background: rgba(245,246,248,.96); border-bottom: 1px solid #d8dde6; z-index: 10; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .sub {{ margin-top: 6px; color: #666; font-size: 13px; }}
    main {{ padding: 16px; display: grid; grid-template-columns: repeat(auto-fill, minmax(760px, 1fr)); gap: 14px; }}
    .card {{ background: #fff; border: 1px solid #d8dde6; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 14px rgba(0,0,0,.05); }}
    .media {{ width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #000; display: block; }}
    .body {{ padding: 12px; }}
    .name {{ font-weight: 700; font-size: 15px; word-break: break-all; }}
    .meta, .caption, .path {{ margin-top: 8px; font-size: 12px; color: #555; word-break: break-word; }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; color: #777; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #e7ebf1; padding: 6px 5px; font-size: 12px; }}
    th {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <header>
    <h1>hq_preview_0525 物体面积占比</h1>
    <div class="sub">当前页面统计每个样本里每个物体在整段视频中的平均面积占比，平均方式为对全部帧直接求均值。当前样本数：{len(sample_dirs)}</div>
  </header>
  <main>
    {''.join(cards) if cards else '<div>暂无可统计样本。</div>'}
  </main>
</body>
</html>"""
    (OUT_ROOT / "index.html").write_text(html_text, encoding="utf-8")
    (OUT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": str(DATA_ROOT),
                "out_root": str(OUT_ROOT),
                "sample_count": len(sample_dirs),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT_ROOT / "index.html")


if __name__ == "__main__":
    main()
