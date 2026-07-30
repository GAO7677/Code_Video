#!/usr/bin/env python3
"""Build the stable visualization entry page for xSSC LoRA experiments."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    gallery_manifest_path = run_root / "gallery" / "manifest.json"
    manifest = json.loads(gallery_manifest_path.read_text(encoding="utf-8"))
    methods = "".join(
        f"<li><strong>{escape(method['label'])}</strong>"
        f"<span>{escape(method['directory'])}</span></li>"
        for method in manifest["methods"]
    )
    num_cases = int(manifest["num_cases"])
    num_requested = int(manifest.get("num_requested", num_cases))
    status = "已完成" if num_cases == num_requested else "生成中"
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC LoRA 训练可视化总览</title>
  <style>
    :root {{
      --bg: #f3f5f6;
      --surface: #fff;
      --ink: #182226;
      --muted: #66747a;
      --line: #d6dde0;
      --accent: #006d77;
      --warm: #9b3a31;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Noto Sans SC", Arial, sans-serif;
    }}
    header {{
      padding: 22px 26px 18px;
      background: var(--surface);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 7px; font-size: 24px; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    .section-title {{ margin: 0 0 10px; font-size: 14px; color: var(--muted); }}
    .entry {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      padding: 18px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    h2 {{ margin: 0 0 7px; font-size: 18px; }}
    .meta {{ color: var(--muted); font-size: 14px; }}
    .status {{
      align-self: start;
      min-width: 116px;
      padding: 10px 12px;
      text-align: center;
      color: var(--accent);
      border-left: 3px solid var(--accent);
      background: #edf6f5;
      font-weight: 750;
    }}
    .status strong {{ display: block; margin-top: 3px; font-size: 18px; }}
    ul {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 7px 14px;
      margin: 14px 0 16px;
      padding: 0;
      list-style: none;
    }}
    li {{
      min-width: 0;
      padding-top: 7px;
      border-top: 1px solid var(--line);
      font-size: 13px;
    }}
    li strong {{ display: block; color: var(--warm); }}
    li span {{
      display: block;
      margin-top: 2px;
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    a {{
      display: inline-block;
      padding: 9px 13px;
      color: #fff;
      background: var(--accent);
      border-radius: 5px;
      text-decoration: none;
      font-weight: 750;
    }}
    @media (max-width: 720px) {{
      main {{ padding: 14px; }}
      .entry {{ grid-template-columns: 1fr; }}
      ul {{ grid-template-columns: 1fr; }}
      .status {{ justify-self: stretch; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC LoRA 训练可视化总览</h1>
    <p>Object cross-attention 与 self-attention LoRA 系列实验</p>
  </header>
  <main>
    <div class="section-title">推理对比</div>
    <section class="entry">
      <div>
        <h2>最新 checkpoint · test_5</h2>
        <div class="meta">8 context frames · 49 output frames · 512×896 · 8 denoising steps · seed 42</div>
        <ul>{methods}</ul>
        <a href="gallery/">进入案例对比</a>
      </div>
      <div class="status">{status}<strong>{num_cases}/{num_requested}</strong></div>
    </section>
  </main>
</body>
</html>
"""
    output_path = run_root / "index.html"
    output_path.write_text(page, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
