#!/usr/bin/env python3
"""生成对比页 + 启动本地端口"""

from __future__ import annotations

import html
import json
import os
import signal
import subprocess
from pathlib import Path

OUTPUT_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VIDEO_DIR = OUTPUT_DIR / "videos"
DEFAULT_PORT = 18702


def scenario_meta(name: str) -> dict:
    """Pre-defined metadata for each scenario."""
    mapping = {
        "e03_mu05_m1":  {"group": "恢复系数", "e": 0.3, "mu": 0.5, "m": 1.0,  "desc": "塑性碰撞 — 球几乎不反弹"},
        "e05_mu05_m1":  {"group": "恢复系数", "e": 0.5, "mu": 0.5, "m": 1.0,  "desc": "中等弹性 — 部分动能损失"},
        "e07_mu05_m1":  {"group": "恢复系数", "e": 0.7, "mu": 0.5, "m": 1.0,  "desc": "高弹性 — 球明显反弹"},
        "e09_mu05_m1":  {"group": "恢复系数", "e": 0.9, "mu": 0.5, "m": 1.0,  "desc": "超高弹性 — 球快速弹飞"},
        "e07_mu01_m1":  {"group": "摩擦系数", "e": 0.7, "mu": 0.1, "m": 1.0, "desc": "低摩擦 — 碰撞打滑"},
        "e07_mu10_m1":  {"group": "摩擦系数", "e": 0.7, "mu": 1.0, "m": 1.0, "desc": "高摩擦 — 咬合带动旋转"},
        "e07_mu05_m01": {"group": "球质量",   "e": 0.7, "mu": 0.5, "m": 0.1, "desc": "轻球 (0.1kg) — 弹飞"},
        "e07_mu05_m5":  {"group": "球质量",   "e": 0.7, "mu": 0.5, "m": 5.0, "desc": "重球 (5.0kg) — 推动木块"},
    }
    return mapping.get(name, {})


def generate_html(port: int) -> Path:
    """Generate index.html with side-by-side video comparison."""
    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    cards = []
    for vp in videos:
        name = vp.stem
        meta = scenario_meta(name)
        group = meta.get("group", "其他")
        e = meta.get("e", "-")
        mu = meta.get("mu", "-")
        m = meta.get("m", "-")
        desc = meta.get("desc", "")
        cards.append({
            "name": name,
            "group": group,
            "e": e,
            "mu": mu,
            "m": m,
            "desc": desc,
            "filename": vp.name,
        })

    # Group by parameter category
    groups: dict[str, list] = {}
    for card in cards:
        groups.setdefault(card["group"], []).append(card)

    group_sections = []
    for group_name, items in groups.items():
        video_cards = []
        for item in items:
            video_cards.append(f"""
            <div class="video-card">
              <video controls autoplay loop muted playsinline>
                <source src="videos/{html.escape(item['filename'])}" type="video/mp4">
              </video>
              <div class="param-bar">
                <span class="param-badge">e = {item['e']}</span>
                <span class="param-badge">μ = {item['mu']}</span>
                <span class="param-badge">m = {item['m']} kg</span>
              </div>
              <div class="desc">{html.escape(item['desc'])}</div>
            </div>
            """)
        group_sections.append(f"""
        <section class="group">
          <h2>{html.escape(group_name)} 变化</h2>
          <div class="video-grid">
            {''.join(video_cards)}
          </div>
        </section>
        """)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>球撞击木块 — 物理参数对比</title>
  <style>
    :root {{
      --bg: #1a1815;
      --panel: #252320;
      --line: #3d3830;
      --text: #e8e4dd;
      --muted: #9d968a;
      --accent: #e08840;
      --blue: #6ba4d1;
      --green: #6db87d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
      background: var(--bg);
    }}
    .page {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 32px;
    }}
    .sub {{
      color: var(--muted);
      margin: 0 0 24px;
      font-size: 14px;
      line-height: 1.6;
    }}
    .sub strong {{ color: var(--accent); }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 28px;
    }}
    .legend span {{
      padding: 8px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      font-size: 13px;
    }}
    .group {{
      margin-bottom: 36px;
    }}
    h2 {{
      font-size: 20px;
      margin: 0 0 16px;
      padding-bottom: 10px;
      border-bottom: 1px solid var(--line);
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 18px;
    }}
    .video-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      overflow: hidden;
      transition: border-color 0.2s;
    }}
    .video-card:hover {{
      border-color: var(--accent);
    }}
    video {{
      width: 100%;
      display: block;
      background: #000;
    }}
    .param-bar {{
      display: flex;
      gap: 8px;
      padding: 12px 14px 4px;
    }}
    .param-badge {{
      padding: 4px 10px;
      border-radius: 6px;
      background: rgba(255,255,255,0.08);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.03em;
    }}
    .desc {{
      padding: 6px 14px 14px;
      font-size: 13px;
      color: var(--muted);
    }}
    .footnote {{
      margin-top: 32px;
      padding-top: 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>⚾ 球撞击木块 — 物理参数对比</h1>
    <p class="sub">
      初始条件：球初速 <strong>v₀ = (3.5, 0, 1.8) m/s</strong>（抛物线轨迹），重力 <strong>g = 9.81 m/s²</strong><br />
      木块质量固定 1.5kg，球质量可变；所有场景均受重力、摩擦力、空气阻尼<br />
      视频上标注了实时速度和速度方向箭头（橙色=球，蓝色=块）
    </p>
    <div class="legend">
      <span>e = 恢复系数（0=完全非弹性, 1=完全弹性）</span>
      <span>μ = 摩擦系数</span>
      <span>m = 球质量</span>
      <span>所有视频循环播放，可按空格暂停</span>
    </div>
    {''.join(group_sections)}
    <div class="footnote">
      Powered by PyBullet &middot; {len(videos)} scenarios &middot;
      <a href="http://127.0.0.1:{port}" style="color:var(--accent)">http://127.0.0.1:{port}</a>
    </div>
  </div>
</body>
</html>
"""
    path = OUTPUT_DIR / "index.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    html_path = generate_html(args.port)
    print(f"Report: {html_path}")

    if args.render_only:
        return

    print(f"\nServing at http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")

    # Use python http.server
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(args.port), "--directory", str(OUTPUT_DIR)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def handler(signum, frame):
        proc.terminate()
        proc.wait()
        print("\nStopped.")

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
