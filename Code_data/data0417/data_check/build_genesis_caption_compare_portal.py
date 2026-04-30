#!/usr/bin/env python3
"""Build a local HTML portal to compare Genesis window/mytest captions against raw videos."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases")
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/genesis_caption_compare_portal")

GROUPS = [
    {
        "slug": "stage1adapter_train_genesis",
        "title": "Stage1adapter Train Genesis",
        "root": ROOT / "stage1adapter" / "train" / "genesis",
    },
    {
        "slug": "stage1adapter_test_genesis",
        "title": "Stage1adapter Test Genesis",
        "root": ROOT / "stage1adapter" / "test" / "genesis",
    },
    {
        "slug": "mytest_genesis",
        "title": "Mytest Genesis Heldout",
        "root": ROOT / "mytest",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_raw_meta(source_sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        candidate = source_sample_dir / name
        if candidate.exists():
            return candidate
    return None


def current_video_path(sample_dir: Path, meta: dict[str, Any]) -> str:
    paths = meta.get("paths", {}) if isinstance(meta.get("paths"), dict) else {}
    for key in ("full_video_path", "video_path", "rgb_video_path"):
        value = paths.get(key)
        if value and Path(value).exists():
            return str(value)
    direct = sample_dir / "full_video.mp4"
    if direct.exists():
        return str(direct)
    fallback = sample_dir / "videos" / "rgb.mp4"
    if fallback.exists():
        return str(fallback)
    return ""


def source_sample_dir(meta: dict[str, Any]) -> Path | None:
    source_paths = meta.get("source_paths", {}) if isinstance(meta.get("source_paths"), dict) else {}
    source_dir = source_paths.get("source_sample_dir")
    if source_dir and Path(source_dir).exists():
        return Path(source_dir)
    return None


def raw_video_path(source_dir: Path | None) -> str:
    if source_dir is None:
        return ""
    candidate = source_dir / "videos" / "rgb.mp4"
    return str(candidate) if candidate.exists() else ""


def build_record(meta_path: Path) -> dict[str, Any]:
    meta = load_json(meta_path)
    sample_dir = meta_path.parent
    source_dir = source_sample_dir(meta)
    raw_meta_path = find_raw_meta(source_dir) if source_dir is not None else None
    raw_meta = load_json(raw_meta_path) if raw_meta_path is not None else {}
    return {
        "sample_dir": str(sample_dir),
        "sample_name": sample_dir.name,
        "current_caption": str(meta.get("caption") or ""),
        "current_detail": str(meta.get("detail_caption") or ""),
        "current_video": current_video_path(sample_dir, meta),
        "raw_sample_dir": str(source_dir) if source_dir is not None else "",
        "raw_video": raw_video_path(source_dir),
        "raw_caption": str(raw_meta.get("caption") or ""),
        "raw_detail": str(raw_meta.get("detail_caption") or ""),
    }


def build_gif(src_video: str, dst_gif: Path) -> str:
    if not src_video:
        return ""
    src_path = Path(src_video)
    if not src_path.exists():
        return ""
    dst_gif.parent.mkdir(parents=True, exist_ok=True)
    reader = imageio.get_reader(str(src_path))
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()
    if not frames:
        return ""
    imageio.mimsave(str(dst_gif), frames, format="GIF", fps=8)
    return str(dst_gif)


def build_html(groups: list[dict[str, Any]]) -> str:
    nav_links = "\n".join(
        f"<a href='#{html.escape(group['slug'])}'>{html.escape(group['title'])}</a>" for group in groups
    )
    sections = []
    for group in groups:
        cards = []
        for item in group["items"]:
            current_detail = item["current_detail"] or "<missing>"
            raw_detail = item["raw_detail"] or "<missing>"
            current_caption = item["current_caption"] or "<missing>"
            raw_caption = item["raw_caption"] or "<missing>"
            current_video = (
                f"<img class='gif-preview' src='{html.escape(item['current_gif'])}' alt='current sample gif preview'>"
                if item["current_gif"]
                else "<div class='missing'>current gif missing</div>"
            )
            raw_video = (
                f"<video controls preload='metadata' playsinline src='{html.escape(item['raw_video'])}'></video>"
                if item["raw_video"]
                else "<div class='missing'>raw video missing</div>"
            )
            cards.append(
                f"""
                <article class="card">
                  <div class="card-head">
                    <h3>{html.escape(item['sample_name'])}</h3>
                    <p class="path">{html.escape(item['sample_dir'])}</p>
                    <p class="path raw-path">{html.escape(item['raw_sample_dir'])}</p>
                  </div>
                  <div class="videos">
                    <div class="video-block">
                      <div class="video-title">Current Sample GIF</div>
                      {current_video}
                    </div>
                    <div class="video-block">
                      <div class="video-title">Raw Source Video</div>
                      {raw_video}
                    </div>
                  </div>
                  <div class="captions">
                    <div class="caption-box">
                      <div class="caption-title">Current Caption</div>
                      <p>{html.escape(current_caption)}</p>
                    </div>
                    <div class="caption-box">
                      <div class="caption-title">Current Detail Caption</div>
                      <p>{html.escape(current_detail)}</p>
                    </div>
                    <div class="caption-box">
                      <div class="caption-title">Raw Caption</div>
                      <p>{html.escape(raw_caption)}</p>
                    </div>
                    <div class="caption-box">
                      <div class="caption-title">Raw Detail Caption</div>
                      <p>{html.escape(raw_detail)}</p>
                    </div>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section id="{html.escape(group['slug'])}" class="group">
              <div class="group-head">
                <h2>{html.escape(group['title'])}</h2>
                <p>showing {group['shown']} / {group['total']} samples</p>
              </div>
              <div class="cards">{''.join(cards)}</div>
            </section>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Genesis Caption Compare Portal</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf8;
      --ink: #1e1a17;
      --muted: #6d6258;
      --line: #dbcdbb;
      --accent: #985f2a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(152,95,42,0.10), transparent 28%),
        linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 16px 24px;
      background: rgba(255, 253, 248, 0.92);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ margin: 0; color: var(--muted); }}
    nav {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    nav a {{
      color: var(--accent);
      text-decoration: none;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff8f0;
    }}
    main {{ padding: 20px 24px 40px; }}
    .group {{ margin-bottom: 44px; }}
    .group-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 16px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(49, 33, 16, 0.06);
    }}
    .card h3 {{ margin: 0 0 6px; font-size: 18px; }}
    .path {{
      margin: 0 0 4px;
      color: var(--muted);
      word-break: break-all;
      font-size: 12px;
    }}
    .raw-path {{ color: #85664d; }}
    .videos {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .video-title, .caption-title {{
      font-size: 12px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 6px;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
      border: 1px solid #d6c7b4;
    }}
    .gif-preview {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid #d6c7b4;
      background: #000;
    }}
    .captions {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .caption-box {{
      background: #fff8ef;
      border: 1px solid #eadcc9;
      border-radius: 12px;
      padding: 12px;
    }}
    .caption-box p {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.45;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 160px;
      border: 1px dashed var(--line);
      border-radius: 12px;
      color: var(--muted);
      background: #faf5ec;
    }}
    @media (max-width: 1000px) {{
      .videos, .captions {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Genesis Caption Compare Portal</h1>
    <p class="sub">Compare current window/heldout captions against the original raw Genesis videos and raw captions.</p>
    <nav>{nav_links}</nav>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local portal for Genesis caption comparison")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit_per_group", type=int, default=24)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups_out = []
    manifest = []
    for group in GROUPS:
        meta_paths = sorted(group["root"].rglob("meta.json"))
        items = [build_record(path) for path in meta_paths[: max(0, args.limit_per_group)]]
        for idx, item in enumerate(items):
            gif_path = args.output_dir / "assets" / group["slug"] / f"{idx:03d}_{item['sample_name']}.gif"
            item["current_gif"] = build_gif(item["current_video"], gif_path)
        groups_out.append(
            {
                "slug": group["slug"],
                "title": group["title"],
                "total": len(meta_paths),
                "shown": len(items),
                "items": items,
            }
        )
        manifest.append(
            {
                "slug": group["slug"],
                "title": group["title"],
                "total": len(meta_paths),
                "shown": len(items),
            }
        )

    (args.output_dir / "index.html").write_text(build_html(groups_out), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
