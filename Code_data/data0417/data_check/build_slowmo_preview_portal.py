"""Build a compact local portal for slow-motion Genesis preview samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample_root",
        type=Path,
        required=True,
        help="Root directory that contains generated sample folders with meta.json.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Portal output directory.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Slowmo Preview Portal",
        help="Page title.",
    )
    return parser.parse_args()


def scan_samples(sample_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for meta_path in sorted(sample_root.rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        video_rel = meta.get("outputs", {}).get("rgb_video", "videos/rgb.mp4")
        video_path = meta_path.parent / video_rel
        gif_path = meta_path.parent / "visualizations" / "rgb_preview.gif"
        media_path = gif_path if gif_path.exists() else video_path
        if not media_path.exists():
            continue
        playback = meta.get("video_playback", {}) or {}
        sim = meta.get("simulation", {}) or {}
        samples.append(
            {
                "scene_id": meta.get("scene_id", meta_path.parent.name),
                "case_name": meta.get("case_name", "unknown"),
                "count_bucket": meta.get("object_count_bucket", "unknown"),
                "composition": meta.get("scene_composition", "unknown"),
                "motion_category": meta.get("motion_category", "unknown"),
                "num_objects": meta.get("num_objects", "unknown"),
                "media_path": str(media_path),
                "media_type": "gif" if gif_path.exists() else "video",
                "meta_path": str(meta_path),
                "slowdown_factor": float(playback.get("slowdown_factor", sim.get("playback_slowdown_factor", 1.0))),
                "base_fps": float(playback.get("base_video_fps", sim.get("base_video_fps", sim.get("video_fps", 0.0)))),
                "effective_fps": float(playback.get("effective_video_fps", sim.get("video_fps", 0.0))),
                "duration_source": str(sim.get("duration_source", "")),
                "requested_duration_sec": float(sim.get("requested_duration_sec", 0.0) or 0.0),
                "physical_duration_sec": float(sim.get("physical_duration_sec", 0.0) or 0.0),
                "frame_dt": float(sim.get("frame_dt", 0.0) or 0.0),
                "frames": int(meta.get("frames", 0) or 0),
            }
        )
    return samples


def render_html(title: str, samples: list[dict[str, Any]]) -> str:
    cards = []
    for sample in samples:
        if sample["media_type"] == "gif":
            media_html = f'<img class="video" src="{sample["media_path"]}" loading="lazy" />'
        else:
            media_html = f'<video class="video" src="{sample["media_path"]}" controls muted loop preload="metadata"></video>'
        cards.append(
            f"""
            <article class="card">
              {media_html}
              <div class="body">
                <div class="scene">{sample['scene_id']}</div>
                <div class="meta"><span>{sample['case_name']}</span><span>{sample['count_bucket']}</span><span>{sample['num_objects']} objects</span></div>
                <div class="meta"><span>{sample['composition']}</span><span>{sample['motion_category']}</span></div>
                <div class="badge">slowdown x{sample['slowdown_factor']:.2f}</div>
                <div class="fps">base {sample['base_fps']:.2f} fps -> export {sample['effective_fps']:.2f} fps</div>
                <div class="fps">duration_source {sample['duration_source']} | requested {sample['requested_duration_sec']:.3f}s | physical {sample['physical_duration_sec']:.3f}s</div>
                <div class="fps">frame_dt {sample['frame_dt']:.3f}s | frames {sample['frames']}</div>
                <div class="path">{sample['meta_path']}</div>
              </div>
            </article>
            """
        )
    cards_html = "\n".join(cards) if cards else '<p class="empty">No preview samples found.</p>'
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f3ec;
      --panel: #fffdf8;
      --line: #d8d1c4;
      --text: #22201d;
      --muted: #6c655d;
      --accent: #c96f37;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background: linear-gradient(180deg, #f3efe7 0%, #f8f5ef 100%);
      color: var(--text);
    }}
    header {{
      padding: 18px 22px 8px 22px;
      position: sticky;
      top: 0;
      background: rgba(248, 245, 239, 0.92);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
      z-index: 10;
    }}
    h1 {{
      margin: 0;
      font-size: 22px;
      font-weight: 700;
    }}
    .sub {{
      margin-top: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    main {{
      padding: 18px 20px 28px 20px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(39, 30, 11, 0.06);
    }}
    .video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #000;
      display: block;
    }}
    .body {{
      padding: 12px 14px 14px 14px;
    }}
    .scene {{
      font-size: 15px;
      font-weight: 700;
      line-height: 1.35;
      word-break: break-word;
    }}
    .meta, .fps {{
      margin-top: 8px;
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }}
    .badge {{
      margin-top: 10px;
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #fde6d8;
      color: #9a4d1f;
      font-size: 12px;
      font-weight: 700;
    }}
    .path {{
      margin-top: 10px;
      font-size: 11px;
      color: #7d756c;
      word-break: break-all;
    }}
    .empty {{
      padding: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="sub">展示新导出的轻度慢放样本。慢放通过降低导出 FPS 实现，不改变物理帧序列。</div>
  </header>
  <main>
    {cards_html}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    samples = scan_samples(args.sample_root)
    html = render_html(args.title, samples)
    (args.output_root / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
