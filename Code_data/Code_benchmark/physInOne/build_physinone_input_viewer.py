#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML viewer for PhysInOne benchmark inputs.")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--viewer_dir", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def relpath(path: str | Path, start: Path) -> str:
    return Path(path).resolve().relative_to(start.resolve()).as_posix()


def render_sample(sample: dict[str, Any], output_root: Path) -> str:
    context_dir = Path(sample["context_frames_dir"])
    context_frames = sorted(
        p for p in context_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    context_imgs = "\n".join(
        f'<img src="/{html.escape(relpath(path, output_root))}" alt="{html.escape(path.name)}" loading="lazy">'
        for path in context_frames
    )
    prompt = html.escape(sample["prompt"])
    physics_types = ", ".join(sample.get("physics_types") or [])
    group_label = f'{sample.get("group_id", "")} - {sample.get("group_name", "")}'.strip(" -")
    return f"""
    <section class="card">
      <div class="meta">
        <div><span class="label">sample_id</span><code>{html.escape(sample["sample_id"])}</code></div>
        <div><span class="label">group</span>{html.escape(group_label)}</div>
        <div><span class="label">split</span>{html.escape(str(sample.get("split", "")))}</div>
        <div><span class="label">physics_types</span>{html.escape(physics_types)}</div>
      </div>
      <div class="prompt-block">
        <div class="label">text prompt</div>
        <pre>{prompt}</pre>
      </div>
      <div class="io-grid">
        <div class="panel">
          <div class="label">TI2V visual input</div>
          <img class="hero" src="/{html.escape(relpath(sample["image_path"], output_root))}" alt="input image" loading="lazy">
          <div class="caption">last frame of 8-frame context window</div>
        </div>
        <div class="panel">
          <div class="label">TV2V visual input</div>
          <div class="filmstrip">{context_imgs}</div>
          <div class="caption">8 context frames</div>
        </div>
      </div>
    </section>
    """


def build_html(samples: list[dict[str, Any]], output_root: Path) -> str:
    cards = "\n".join(render_sample(sample, output_root) for sample in samples)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhysInOne Pure A/B Inputs</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf9;
      --ink: #1f2328;
      --muted: #5f6b76;
      --line: #d7d0c5;
      --accent: #0b6bcb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(11,107,203,0.10), transparent 28%),
        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .summary {{
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 15px;
    }}
    .card {{
      background: rgba(255,255,255,0.82);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin: 0 0 20px;
      backdrop-filter: blur(8px);
      box-shadow: 0 12px 40px rgba(53, 43, 27, 0.08);
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 10px 16px;
      margin-bottom: 14px;
      font-size: 14px;
    }}
    .label {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 11px;
      font-weight: 700;
    }}
    pre {{
      margin: 0;
      padding: 14px;
      border-radius: 12px;
      background: #f7f8fa;
      border: 1px solid #e7eaf0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.5;
    }}
    .io-grid {{
      display: grid;
      grid-template-columns: minmax(260px, 420px) 1fr;
      gap: 18px;
      margin-top: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .hero {{
      width: 100%;
      display: block;
      border-radius: 10px;
      border: 1px solid #ddd5ca;
    }}
    .filmstrip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .filmstrip img {{
      width: 100%;
      display: block;
      border-radius: 8px;
      border: 1px solid #ddd5ca;
      background: #ece8df;
    }}
    .caption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
    }}
    @media (max-width: 960px) {{
      body {{ padding: 14px; }}
      .io-grid {{ grid-template-columns: 1fr; }}
      .filmstrip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PhysInOne Pure A/B Input Viewer</h1>
    <p class="summary">Samples: {len(samples)}. This page shows the exact text prompt, TI2V image input, and TV2V 8-frame context input used for the benchmark.</p>
    {cards}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir or (args.output_root / "viewer")
    viewer_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for manifest in args.manifest:
        samples.extend(load_jsonl(manifest))
    samples.sort(key=lambda item: (item.get("group_id", ""), item.get("sample_id", "")))

    payload = {"samples": samples}
    (viewer_dir / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (viewer_dir / "index.html").write_text(build_html(samples, args.output_root), encoding="utf-8")
    print(viewer_dir / "index.html")


if __name__ == "__main__":
    main()
