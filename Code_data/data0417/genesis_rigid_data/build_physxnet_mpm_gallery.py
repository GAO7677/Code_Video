#!/usr/bin/env python3
"""Build a local HTML gallery for PhysXNet MPM sample visualizations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PhysXNet MPM Gallery</title>
  <style>
    :root {{ --bg:#f6f1e9; --panel:rgba(255,251,246,.94); --ink:#221c16; --muted:#6b6256; --accent:#9f4a22; --border:rgba(34,28,22,.12); --shadow:0 18px 44px rgba(61,43,24,.12); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); font-family:Georgia,"Times New Roman",serif; background:radial-gradient(circle at top left, rgba(159,74,34,.12), transparent 24rem), linear-gradient(180deg,#fbf7f1 0%,#f5f0e8 50%,#ebdfcf 100%); }}
    main {{ width:min(1620px, calc(100vw - 28px)); margin:0 auto; padding:24px 0 48px; }}
    .hero,.card {{ border:1px solid var(--border); border-radius:26px; background:var(--panel); box-shadow:var(--shadow); }}
    .hero {{ padding:28px; }}
    .cards {{ display:grid; gap:22px; margin-top:24px; }}
    .card {{ padding:18px; }}
    h1 {{ margin:0; font-size:clamp(2rem,3vw,3.1rem); }}
    h2 {{ margin:0; font-size:1.35rem; }}
    .sub {{ margin-top:12px; color:var(--muted); line-height:1.65; }}
    .chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
    .chip {{ padding:6px 10px; border-radius:999px; border:1px solid rgba(159,74,34,.18); background:rgba(159,74,34,.08); color:var(--accent); font-size:.9rem; }}
    .path {{ margin-top:12px; padding:10px 12px; border-radius:12px; background:rgba(34,28,22,.05); font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; overflow-wrap:anywhere; }}
    .video-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; margin-top:16px; }}
    .pane {{ padding:12px; border:1px solid var(--border); border-radius:18px; background:rgba(255,255,255,.66); }}
    .pane h3 {{ margin:0 0 10px; font-size:1rem; }}
    video {{ width:100%; aspect-ratio:4/3; background:#101010; border-radius:14px; object-fit:contain; }}
    .img-grid {{ display:grid; grid-template-columns:1.3fr 1fr; gap:12px; margin-top:14px; align-items:start; }}
    .img-grid img {{ width:100%; border-radius:16px; border:1px solid var(--border); background:#fff; }}
    .links {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; }}
    .links a {{ text-decoration:none; color:var(--accent); border:1px solid rgba(159,74,34,.2); background:rgba(159,74,34,.07); border-radius:999px; padding:8px 12px; }}
    .links a:hover {{ background:rgba(159,74,34,.12); }}
    @media (max-width: 820px) {{
      main {{ width:min(100vw - 16px, 1620px); }}
      .hero,.card {{ padding:16px; border-radius:20px; }}
      .img-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>PhysXNet MPM Gallery</h1>
      <div class="sub">对象 `{object_id}`，MPM 仿真。当前页面汇总 RGB、Depth、Mask、白底 HSV 光流视频，拼接预览图，轨迹图和交互式 3D mesh 场景。</div>
      <div class="sub">当前已完成样本数：{count}。</div>
    </section>
    <section id="cards" class="cards"></section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const records = JSON.parse(document.getElementById('records').textContent || '[]');
    const cards = document.getElementById('cards');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    cards.innerHTML = records.map((item) => {{
      const chips = [item.motion_category, item.interaction_pattern].filter(Boolean).map(v => `<span class="chip">${{esc(v)}}</span>`).join('');
      const videos = [
        ['RGB', item.rgb_video],
        ['Depth', item.depth_video],
        ['Mask', item.mask_video],
        ['HSV Flow', item.flow_video],
      ].filter(([,href]) => href).map(([label, href]) => `
        <article class="pane">
          <h3>${{esc(label)}}</h3>
          <video controls preload="none" playsinline src="${{encodeURI(href)}}"></video>
        </article>`).join('');
      return `
        <article class="card">
          <h2>${{esc(item.scene_id)}}</h2>
          <div class="chips">${{chips}}</div>
          <div class="path">${{esc(item.rel_dir)}}</div>
          <div class="video-grid">${{videos}}</div>
          <div class="img-grid">
            <a href="${{encodeURI(item.preview_grid)}}" target="_blank" rel="noreferrer"><img src="${{encodeURI(item.preview_grid)}}" alt="preview grid"></a>
            <a href="${{encodeURI(item.trajectory_png)}}" target="_blank" rel="noreferrer"><img src="${{encodeURI(item.trajectory_png)}}" alt="trajectory overview"></a>
          </div>
          <div class="links">
            <a href="${{encodeURI(item.scene_3d_html)}}" target="_blank" rel="noreferrer">交互 3D 场景</a>
            <a href="${{encodeURI(item.preview_grid)}}" target="_blank" rel="noreferrer">拼接总览图</a>
            <a href="${{encodeURI(item.trajectory_png)}}" target="_blank" rel="noreferrer">轨迹图</a>
          </div>
        </article>`;
    }}).join('');
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_root", type=str, required=True)
    parser.add_argument("--output_html", type=str, default=None)
    return parser.parse_args()


def build_record(sample_dir: Path) -> dict | None:
    metadata_path = sample_dir / "metadata.json"
    vis_dir = sample_dir / "visualizations"
    required = [
        metadata_path,
        sample_dir / "videos" / "rgb.mp4",
        sample_dir / "videos" / "depth.mp4",
        vis_dir / "mask_vis.mp4",
        vis_dir / "flow_vis.mp4",
        vis_dir / "preview_grid.png",
        vis_dir / "trajectory_overview.png",
        vis_dir / "scene_3d.html",
    ]
    if not all(path.exists() for path in required):
        return None
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    rel_dir = sample_dir.name
    return {
        "scene_id": rel_dir,
        "motion_category": meta.get("motion_category", ""),
        "interaction_pattern": meta.get("interaction_pattern", ""),
        "rel_dir": rel_dir,
        "rgb_video": f"{rel_dir}/videos/rgb.mp4",
        "depth_video": f"{rel_dir}/videos/depth.mp4",
        "mask_video": f"{rel_dir}/visualizations/mask_vis.mp4",
        "flow_video": f"{rel_dir}/visualizations/flow_vis.mp4",
        "preview_grid": f"{rel_dir}/visualizations/preview_grid.png",
        "trajectory_png": f"{rel_dir}/visualizations/trajectory_overview.png",
        "scene_3d_html": f"{rel_dir}/visualizations/scene_3d.html",
    }


def main() -> None:
    args = parse_args()
    cases_root = Path(args.cases_root).resolve()
    records = []
    object_id = ""
    for sample_dir in sorted(path for path in cases_root.iterdir() if path.is_dir()):
        record = build_record(sample_dir)
        if record is None:
            continue
        records.append(record)
        if not object_id:
            object_id = sample_dir.name.split("__", 1)[0]

    output_html = Path(args.output_html).resolve() if args.output_html else cases_root / "physxnet_mpm_gallery.html"
    html = HTML_TEMPLATE.format(
        object_id=object_id or "unknown",
        count=len(records),
        records_json=json.dumps(records, ensure_ascii=False),
    )
    output_html.write_text(html, encoding="utf-8")
    print(f"[DONE] wrote {output_html}")
    print(f"[INFO] cases included: {len(records)}")


if __name__ == "__main__":
    main()
