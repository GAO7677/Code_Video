#!/usr/bin/env python3
"""Build a single-page gallery for a few state-validation cases."""

from __future__ import annotations

import html
import json
import os
import shutil
from pathlib import Path


ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_check/state_validation_window")
OUT_DIR = ROOT / "gallery_cases"


CASE_SUMMARIES = [
    ROOT / "test__genesis/cases/00_genesis_heldout_0142__10005__case003_static_highdrop/summary.json",
    ROOT / "test__genesis/cases/01_genesis_heldout_0141__10005__case002_static_right/summary.json",
    ROOT / "test__movi_d/cases/00_movi_d_test_0001__video_131/summary.json",
    ROOT / "test__movi_d/cases/01_movi_d_test_0002__video_166/summary.json",
    ROOT / "train__genesis/cases/00_10007__case000_static_center__cf_no_collision_neg__ratio11/summary.json",
    ROOT / "train__genesis/cases/02_10007__case001_static_left__cf_no_collision_neg__ratio11/summary.json",
    ROOT / "train__movi_d/cases/00_movi_d_train__video_1497/summary.json",
    ROOT / "train__movi_d/cases/01_movi_d_train__video_163/summary.json",
]


def fmt_num(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets_dir = OUT_DIR / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    manifest = []
    for summary_path in CASE_SUMMARIES:
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        case_dir = summary_path.parent
        src_video = case_dir / "overlay.mp4"
        video_name = f"{summary['sample_id']}.mp4"
        dst_video = assets_dir / video_name
        if src_video.exists():
            shutil.copy2(src_video, dst_video)
        rel_video = os.path.relpath(dst_video, OUT_DIR)
        rel_case = os.path.relpath(case_dir / "index.html", OUT_DIR)
        metrics = summary["metrics"]
        caption = str(summary.get("caption", "")).strip()
        detail_caption = str(summary.get("detail_caption", "")).strip()
        cards.append(
            f"""
<article class="card">
  <div class="meta-top">
    <div>
      <h2>{html.escape(summary['sample_id'])}</h2>
      <p class="dataset">{html.escape(summary['dataset'])}</p>
    </div>
    <a class="open-link" href="{html.escape(rel_case)}">详情页</a>
  </div>
  <video src="{html.escape(rel_video)}" controls preload="metadata"></video>
  <p class="caption"><strong>Caption:</strong> {html.escape(caption or 'n/a')}</p>
  <p class="detail"><strong>Detail:</strong> {html.escape(detail_caption or 'n/a')}</p>
  <div class="metric-grid">
    <div><span>center err</span><strong>{fmt_num(metrics['center_projection_error_px']['mean'])}</strong></div>
    <div><span>bbox IoU</span><strong>{fmt_num(metrics['bbox_iou']['mean'])}</strong></div>
    <div><span>depth rel</span><strong>{fmt_num(metrics['depth_consistency_rel']['mean'])}</strong></div>
    <div><span>vel diff</span><strong>{fmt_num(metrics['velocity_diff_error']['mean'])}</strong></div>
  </div>
  <p class="path">{html.escape(summary['sample_dir'])}</p>
</article>
"""
        )
        manifest.append(
            {
                "sample_id": summary["sample_id"],
                "dataset": summary["dataset"],
                "summary_json": str(summary_path),
                "overlay_mp4": str(case_dir / "overlay.mp4"),
                "case_page": str(case_dir / "index.html"),
            }
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>State Validation Gallery</title>
  <style>
    body {{
      margin: 0;
      padding: 20px;
      font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #f8fafc 0%, #e8eef5 42%, #f4efe5 100%);
      color: #14213d;
    }}
    .wrap {{ max-width: 1820px; margin: 0 auto; }}
    .hero {{
      background: rgba(255,255,255,0.92);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      padding: 20px 22px;
      margin-bottom: 18px;
      box-shadow: 0 10px 30px rgba(15,23,42,0.07);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(255,255,255,0.94);
      border: 1px solid rgba(15,23,42,0.08);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(15,23,42,0.06);
    }}
    .meta-top {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .meta-top h2 {{
      font-size: 18px;
      line-height: 1.25;
      margin: 0 0 4px;
    }}
    .dataset {{
      margin: 0;
      color: #475569;
      font-size: 13px;
    }}
    .open-link {{
      color: #0f766e;
      font-weight: 700;
      text-decoration: none;
      white-space: nowrap;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(15,23,42,0.08);
      background: #111827;
      margin-bottom: 10px;
    }}
    .caption, .detail {{
      margin: 8px 0;
      font-size: 14px;
      line-height: 1.45;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0 12px;
    }}
    .metric-grid div {{
      background: rgba(15,23,42,0.04);
      border-radius: 10px;
      padding: 8px 10px;
    }}
    .metric-grid span {{
      display: block;
      color: #64748b;
      font-size: 12px;
      margin-bottom: 3px;
    }}
    .metric-grid strong {{
      font-size: 14px;
    }}
    .path {{
      margin: 0;
      font-size: 12px;
      line-height: 1.4;
      color: #475569;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>State Validation Gallery</h1>
      <p>几个代表性 case 放在同一页面里，直接看视频上叠加的 state 标注。</p>
      <p>总入口：<a href="../index.html">state_validation_window/index.html</a></p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html_text, encoding="utf-8")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_DIR / "index.html")


if __name__ == "__main__":
    main()
