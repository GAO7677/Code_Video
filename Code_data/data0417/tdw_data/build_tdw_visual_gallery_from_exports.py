#!/usr/bin/env python3
"""Build a simple image/video gallery from existing TDW genesis-format exports."""

from __future__ import annotations

import json
from pathlib import Path


EXPORT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_genesis_format_exports")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_visual_gallery")
TITLE = "TDW Visual Gallery"


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def build_items() -> list[dict]:
    items = []
    for meta_path in sorted(EXPORT_ROOT.glob("train/rigid/*/*/*/meta.json")):
        sample_dir = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rgb_frame = sample_dir / "rgb" / "frame_000.png"
        rgb_video = sample_dir / "videos" / "rgb.mp4"
        depth_video = sample_dir / "visualizations" / "depth_vis.mp4"
        if not rgb_frame.exists() or not rgb_video.exists():
            continue
        items.append(
            {
                "sample_name": meta.get("scene_id", sample_dir.name),
                "case_name": meta.get("case_name", ""),
                "scene": meta.get("scene_composition", ""),
                "kind": meta.get("simulator_type", ""),
                "num_objects": meta.get("num_objects", 0),
                "frames": meta.get("frames", 0),
                "objects": [obj.get("source_object_id", obj.get("name", "")) for obj in meta.get("objects", [])],
                "rgb_frame": rel(rgb_frame, OUTPUT_ROOT.parent),
                "rgb_video": rel(rgb_video, OUTPUT_ROOT.parent),
                "depth_video": rel(depth_video, OUTPUT_ROOT.parent) if depth_video.exists() else None,
                "sample_dir": str(sample_dir),
            }
        )
    return items


def build_html(items: list[dict]) -> str:
    cards = []
    for item in items:
        depth_block = f'<video controls preload="metadata" src="../{item["depth_video"]}"></video>' if item["depth_video"] else ""
        cards.append(
            f"""<article class="card">
  <img src="../{item['rgb_frame']}" alt="{item['sample_name']} first frame">
  <div class="meta">
    <div class="pill">{item['scene']}</div>
    <div class="pill">{item['kind']}</div>
    <h3>{item['case_name']}</h3>
    <p>objects={item['num_objects']} | frames={item['frames']}</p>
    <p>model refs: {", ".join(item["objects"])}</p>
    <video controls preload="metadata" src="../{item['rgb_video']}"></video>
    {depth_block}
    <code>{item['sample_dir']}</code>
  </div>
</article>"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{TITLE}</title>
  <style>
    :root {{
      --bg: #efe6d6;
      --panel: rgba(255, 252, 246, 0.96);
      --ink: #171410;
      --muted: #69635b;
      --accent: #3f6a56;
      --border: rgba(52, 42, 29, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(196, 162, 112, 0.26), transparent 28%),
        radial-gradient(circle at right 12%, rgba(126, 153, 141, 0.22), transparent 22%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1700px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 40px rgba(45, 35, 22, 0.10);
    }}
    .hero {{ padding: 28px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(30px, 5vw, 54px); line-height: 0.95; }}
    h3 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0 0 8px; color: var(--muted); line-height: 1.65; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 18px; }}
    .card {{ overflow: hidden; }}
    .card img {{ width: 100%; display: block; aspect-ratio: 16/9; object-fit: cover; background: #ddd; }}
    .meta {{ padding: 16px 18px 18px; }}
    .pill {{
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(63, 106, 86, 0.12);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    video {{ width: 100%; display: block; margin-top: 10px; background: #000; aspect-ratio: 16/9; }}
    code {{
      display: block;
      margin-top: 12px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      word-break: break-all;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{TITLE}</h1>
      <p>基于现有 TDW 导出结果生成的图像/视频可视化页面。这里展示的是已经稳定生成出来的样本首帧和 RGB 视频，所以可以直接映射到本地端口浏览，不依赖重新渲染 TDW 资源缩略图。</p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    items = build_items()
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_ROOT / "index.html").write_text(build_html(items), encoding="utf-8")
    print(OUTPUT_ROOT / "index.html")


if __name__ == "__main__":
    main()
