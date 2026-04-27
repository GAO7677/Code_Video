#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/data/gaoya/dataset/vLAR-PhysInOne/mytest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a single-page local viewer for PhysInOne mytest samples."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--viewer_dir", type=Path, default=None)
    return parser.parse_args()


def load_samples(root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for meta_path in sorted(root.glob("*/meta.json")):
        sample = json.loads(meta_path.read_text(encoding="utf-8"))
        samples.append(sample)
    samples.sort(key=lambda item: (item.get("group_id", ""), item.get("sample_id", "")))
    return samples


def relpath(path: str | Path, start: Path) -> str:
    return Path(path).resolve().relative_to(start.resolve()).as_posix()


def render_sample(sample: dict[str, Any], root: Path) -> str:
    paths = sample["paths"]
    physics_types = ", ".join(sample.get("physics_types") or [])
    group_label = f'{sample.get("group_id", "")} - {sample.get("group_name", "")}'.strip(" -")
    caption = html.escape(sample.get("caption", ""))
    sample_id = html.escape(sample["sample_id"])

    first_frame = "/" + html.escape(relpath(paths["first_frame_path"], root))
    context_video = "/" + html.escape(relpath(paths["context_video_path"], root))
    future_gt_video = "/" + html.escape(relpath(paths["future_gt_video_path"], root))
    full_video = "/" + html.escape(relpath(paths["full_video_path"], root))

    return f"""
    <section class="row">
      <div class="cell meta">
        <div class="sample-id">{sample_id}</div>
        <div class="meta-grid">
          <div><span class="label">group</span>{html.escape(group_label)}</div>
          <div><span class="label">split</span>{html.escape(str(sample.get("split", "")))}</div>
          <div><span class="label">camera</span>{html.escape(str(sample.get("camera_name", "")))}</div>
          <div><span class="label">physics</span>{html.escape(physics_types)}</div>
        </div>
        <div class="caption-block">
          <span class="label">caption</span>
          <p>{caption}</p>
        </div>
      </div>
      <div class="cell frame-cell">
        <span class="label">first frame</span>
        <img src="{first_frame}" alt="{sample_id} first frame" loading="lazy">
      </div>
      <div class="cell video-cell">
        <span class="label">context video</span>
        <video controls preload="metadata" src="{context_video}"></video>
      </div>
      <div class="cell video-cell">
        <span class="label">future gt</span>
        <video controls preload="metadata" src="{future_gt_video}"></video>
      </div>
      <div class="cell video-cell">
        <span class="label">full video</span>
        <video controls preload="metadata" src="{full_video}"></video>
      </div>
    </section>
    """


def build_html(samples: list[dict[str, Any]], root: Path) -> str:
    rows = "\n".join(render_sample(sample, root) for sample in samples)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhysInOne Mytest Viewer</title>
  <style>
    :root {{
      --bg: #f2efe8;
      --panel: #fffdfa;
      --ink: #1e252b;
      --muted: #5a6772;
      --line: #d8d1c5;
      --accent: #0c6a6d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, rgba(12,106,109,0.08), transparent 20%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
    }}
    .wrap {{
      width: min(1920px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 18px 0 32px;
    }}
    h1 {{
      margin: 0 0 8px;
      padding: 0 12px;
      font-size: 30px;
    }}
    .summary {{
      margin: 0 0 18px;
      padding: 0 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .row {{
      display: grid;
      grid-template-columns: minmax(280px, 360px) 220px repeat(3, minmax(240px, 1fr));
      gap: 12px;
      align-items: start;
      margin: 0 0 12px;
      padding: 12px;
      background: rgba(255, 253, 250, 0.9);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(40, 34, 23, 0.06);
    }}
    .cell {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      min-height: 100%;
    }}
    .meta {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .sample-id {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.45;
      word-break: break-word;
      color: #16313b;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
      font-size: 14px;
    }}
    .label {{
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 10px;
      font-weight: 700;
    }}
    .caption-block p {{
      margin: 0;
      white-space: pre-wrap;
      font-size: 14px;
      line-height: 1.5;
    }}
    img, video {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #ebe5da;
      border: 1px solid #ddd6cb;
    }}
    .frame-cell img {{
      aspect-ratio: 1 / 1;
      object-fit: cover;
    }}
    .video-cell video {{
      aspect-ratio: 1 / 1;
      object-fit: contain;
    }}
    @media (max-width: 1600px) {{
      .row {{
        grid-template-columns: minmax(260px, 340px) 200px repeat(3, minmax(220px, 1fr));
      }}
    }}
    @media (max-width: 1200px) {{
      .row {{
        grid-template-columns: 1fr 1fr;
      }}
    }}
    @media (max-width: 720px) {{
      .wrap {{
        width: min(100vw, calc(100vw - 12px));
      }}
      .row {{
        grid-template-columns: 1fr;
        padding: 10px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PhysInOne Mytest Viewer</h1>
    <p class="summary">Samples: {len(samples)}. Each row shows one sample with first frame, context video, future ground truth, and the full original video.</p>
    {rows}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    viewer_dir = args.viewer_dir or (args.root / "viewer")
    viewer_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.root)
    payload = {"samples": samples}
    (viewer_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (viewer_dir / "index.html").write_text(
        build_html(samples, args.root),
        encoding="utf-8",
    )
    print(viewer_dir / "index.html")


if __name__ == "__main__":
    main()
