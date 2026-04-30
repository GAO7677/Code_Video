#!/usr/bin/env python3
"""Build a compact GIF portal for organized raw/window path lists."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image, ImageOps


DEFAULT_LIST_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/organized_view_split_complexity_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/portals/view_split_complexity_preview_v1"
)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>View / Split / Complexity Preview</title>
  <style>
    :root {
      --bg:#f5efe6;
      --bg2:#ddd1bf;
      --panel:rgba(255,252,247,.95);
      --line:rgba(45,31,18,.12);
      --ink:#221912;
      --muted:#6f6358;
      --accent:#8f5d2c;
      --accent2:#2c6e77;
      --shadow:0 18px 42px rgba(40,29,18,.12);
    }
    * { box-sizing:border-box; }
    body {
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(143,93,44,.16), transparent 26rem),
        radial-gradient(circle at top right, rgba(44,110,119,.12), transparent 28rem),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }
    main {
      width:min(1800px, calc(100vw - 18px));
      margin:0 auto;
      padding:10px 0 32px;
    }
    .hero, .section {
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:20px;
      box-shadow:var(--shadow);
    }
    .hero {
      position:sticky;
      top:8px;
      z-index:5;
      padding:16px 18px;
      backdrop-filter:blur(8px);
    }
    h1 {
      margin:0;
      font-size:clamp(1.6rem, 2.4vw, 2.4rem);
      line-height:1.05;
      letter-spacing:-0.02em;
    }
    .sub {
      margin-top:8px;
      color:var(--muted);
      font-size:.95rem;
      line-height:1.5;
    }
    .stats, .toolbar, .badges {
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }
    .stats { margin-top:12px; }
    .pill, .badge {
      border-radius:999px;
      border:1px solid var(--line);
      padding:5px 10px;
      font-size:.82rem;
      background:#fff8ee;
      white-space:nowrap;
    }
    .toolbar {
      margin-top:12px;
      align-items:center;
    }
    .toolbar input, .toolbar select {
      min-width:180px;
      border:1px solid var(--line);
      border-radius:10px;
      padding:8px 10px;
      background:rgba(255,255,255,.9);
      color:var(--ink);
      font:inherit;
    }
    .sections {
      display:grid;
      gap:12px;
      margin-top:14px;
    }
    .section { padding:14px; }
    .section-head {
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:flex-start;
    }
    .section-head h2 {
      margin:0;
      font-size:1.1rem;
      line-height:1.08;
    }
    .meta {
      margin-top:4px;
      color:var(--muted);
      font-size:.84rem;
      line-height:1.42;
      overflow-wrap:anywhere;
    }
    .sample-grid {
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));
      gap:10px;
      margin-top:12px;
    }
    .sample-card {
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,255,255,.88);
      padding:10px;
    }
    .sample-title {
      display:flex;
      justify-content:space-between;
      gap:8px;
      align-items:flex-start;
    }
    .sample-title h3 {
      margin:0;
      font-size:1rem;
      line-height:1.16;
      overflow-wrap:anywhere;
    }
    .preview-grid {
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:10px;
      margin-top:10px;
    }
    .preview-grid.raw-only { grid-template-columns:1fr; }
    .preview {
      border:1px solid var(--line);
      border-radius:14px;
      padding:8px;
      background:rgba(248,243,236,.8);
    }
    .preview img {
      width:100%;
      display:block;
      border-radius:10px;
      border:1px solid var(--line);
      background:#ece3d6;
    }
    .caption {
      margin-top:6px;
      color:var(--muted);
      font-size:.78rem;
      line-height:1.4;
      overflow-wrap:anywhere;
    }
    .path {
      margin-top:8px;
      font-size:.77rem;
      line-height:1.42;
      color:var(--accent2);
      overflow-wrap:anywhere;
    }
    .hidden { display:none !important; }
    @media (max-width: 1100px) {
      .preview-grid { grid-template-columns:1fr; }
      .sample-grid { grid-template-columns:1fr; }
    }
    @media (max-width: 980px) {
      main { width:min(100vw - 12px, 1800px); }
      .hero { position:static; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Raw / Window Preview Portal</h1>
      <div class="sub">按 view / split / complexity 抽样展示。raw 样本展示完整 RGB 片段；window 样本展示 context / future GT / full 三段 GIF。</div>
      <div class="stats">
        <span class="pill">sections: __SECTION_COUNT__</span>
        <span class="pill">samples: __SAMPLE_COUNT__</span>
        <span class="pill">size: __FRAME_WIDTH__x__FRAME_HEIGHT__</span>
        <span class="pill">fps: __FPS__</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 raw / window / train / test / case">
        <select id="sectionFilter">
          <option value="">全部 section</option>
          __SECTION_OPTIONS__
        </select>
      </div>
    </section>
    <section class="sections" id="sections"></section>
  </main>
  <script id="records" type="application/json">__RECORDS_JSON__</script>
  <script>
    const records = JSON.parse(document.getElementById('records').textContent || '[]');
    const root = document.getElementById('sections');
    const searchBox = document.getElementById('searchBox');
    const sectionFilter = document.getElementById('sectionFilter');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    root.innerHTML = records.map((section) => {
      const cards = section.samples.map((item) => {
        const previews = item.previews.map((preview) => `
          <div class="preview">
            <img loading="lazy" src="${encodeURI(preview.gif_rel)}" alt="${esc(preview.label)}">
            <div class="caption">${esc(preview.label)}</div>
          </div>
        `).join('');
        const gridClass = item.previews.length === 1 ? 'preview-grid raw-only' : 'preview-grid';
        return `
          <article class="sample-card" data-section="${esc(section.slug)}" data-search="${esc(item.search_text)}">
            <div class="sample-title">
              <div>
                <h3>${esc(item.sample_id)}</h3>
                <div class="meta">${esc(item.dataset)} | ${esc(item.view)} | ${esc(item.split)} | ${esc(item.complexity)}</div>
              </div>
              <div class="badges">
                <span class="badge">${esc(item.dataset)}</span>
                <span class="badge">${esc(item.view)}</span>
              </div>
            </div>
            <div class="${gridClass}">${previews}</div>
            <div class="caption">${esc(item.caption)}</div>
            <div class="path">${esc(item.sample_dir)}</div>
          </article>
        `;
      }).join('');
      return `
        <article class="section" data-section="${esc(section.slug)}" data-search="${esc(section.search_text)}">
          <div class="section-head">
            <div>
              <h2>${esc(section.title)}</h2>
              <div class="meta">showing ${section.samples.length} sample(s)</div>
            </div>
            <div class="badges"><span class="badge">${section.samples.length} samples</span></div>
          </div>
          <div class="sample-grid">${cards}</div>
        </article>
      `;
    }).join('');
    const applyFilter = () => {
      const q = searchBox.value.trim().toLowerCase();
      const sectionValue = sectionFilter.value;
      for (const sectionNode of root.querySelectorAll('.section')) {
        const sectionOk = !sectionValue || sectionNode.dataset.section === sectionValue;
        const sectionSearch = sectionNode.dataset.search.toLowerCase();
        let anyCardVisible = false;
        for (const cardNode of sectionNode.querySelectorAll('.sample-card')) {
          const ok = (!q || cardNode.dataset.search.toLowerCase().includes(q) || sectionSearch.includes(q)) && sectionOk;
          cardNode.classList.toggle('hidden', !ok);
          if (ok) anyCardVisible = true;
        }
        const sectionVisible = sectionOk && (!q || sectionSearch.includes(q) || anyCardVisible);
        sectionNode.classList.toggle('hidden', !sectionVisible);
      }
    };
    searchBox.addEventListener('input', applyFilter);
    sectionFilter.addEventListener('change', applyFilter);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_root", type=Path, default=DEFAULT_LIST_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max_per_section", type=int, default=1)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8136)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_free_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"no free port found near {preferred}")


def start_server(root_dir: Path, host: str, port: int) -> tuple[int, str]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            host,
            "--directory",
            str(root_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    view_host = "127.0.0.1" if host == "0.0.0.0" else host
    return process.pid, f"http://{view_host}:{port}/index.html"


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def slugify(text: str) -> str:
    chars: list[str] = []
    for ch in str(text).lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "item"


def pick_meta_path(sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        path = sample_dir / name
        if path.exists():
            return path
    return None


def load_caption(sample_dir: Path) -> str:
    meta_path = pick_meta_path(sample_dir)
    if not meta_path:
        return ""
    meta = load_json(meta_path)
    return str(meta.get("caption") or meta.get("prompt") or "")


def detect_dataset(sample_dir: Path) -> str:
    parts = {part.lower() for part in sample_dir.parts}
    if "movi-d" in parts or "movi_d" in parts:
        return "movi-d"
    if "genesis" in parts:
        return "genesis"
    if "version_1_genesis_rigid_data_all_cases" in parts:
        return "genesis"
    return "unknown"


def video_frames_from_mp4(video_path: Path, frame_size: tuple[int, int], fps: float) -> list[Image.Image]:
    frames: list[Image.Image] = []
    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 12.0) or 12.0)
    stride = max(1, int(round(src_fps / max(fps, 0.1))))
    try:
        for idx, frame in enumerate(reader):
            if idx % stride != 0:
                continue
            image = Image.fromarray(frame).convert("RGB")
            thumb = ImageOps.contain(image, frame_size)
            canvas = Image.new("RGB", frame_size, (245, 239, 230))
            canvas.paste(thumb, ((frame_size[0] - thumb.width) // 2, (frame_size[1] - thumb.height) // 2))
            frames.append(canvas)
    finally:
        reader.close()
    return frames


def video_frames_from_png_dir(rgb_dir: Path, frame_size: tuple[int, int]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_path in sorted(rgb_dir.glob("frame_*.png")):
        image = Image.open(frame_path).convert("RGB")
        thumb = ImageOps.contain(image, frame_size)
        canvas = Image.new("RGB", frame_size, (245, 239, 230))
        canvas.paste(thumb, ((frame_size[0] - thumb.width) // 2, (frame_size[1] - thumb.height) // 2))
        frames.append(canvas)
    return frames


def write_gif(gif_path: Path, frames: list[Image.Image], fps: float) -> bool:
    if not frames:
        return False
    ensure_dir(gif_path.parent)
    duration_ms = max(1, int(round(1000.0 / max(float(fps), 0.1))))
    first, *rest = frames
    first.save(
        gif_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for frame in frames:
        frame.close()
    return True


def build_raw_preview(sample_dir: Path, output_dir: Path, frame_size: tuple[int, int], fps: float, overwrite: bool) -> list[dict[str, str]]:
    gif_path = output_dir / "raw_full.gif"
    if gif_path.exists() and not overwrite:
        return [{"label": "raw RGB", "gif_rel": gif_path.name}]

    mp4_path = sample_dir / "videos" / "rgb.mp4"
    if mp4_path.exists():
        frames = video_frames_from_mp4(mp4_path, frame_size=frame_size, fps=fps)
    else:
        frames = video_frames_from_png_dir(sample_dir / "rgb", frame_size=frame_size)
    if write_gif(gif_path, frames, fps=fps):
        return [{"label": "raw RGB", "gif_rel": gif_path.name}]
    return []


def build_window_previews(sample_dir: Path, output_dir: Path, frame_size: tuple[int, int], fps: float, overwrite: bool) -> list[dict[str, str]]:
    previews: list[dict[str, str]] = []
    sources = [
        ("context_video.mp4", "context.gif", "context"),
        ("future_gt_video.mp4", "future.gif", "future GT"),
        ("full_video.mp4", "full.gif", "full"),
    ]
    for video_name, gif_name, label in sources:
        gif_path = output_dir / gif_name
        if gif_path.exists() and not overwrite:
            previews.append({"label": label, "gif_rel": gif_path.name})
            continue
        video_path = sample_dir / video_name
        if not video_path.exists():
            continue
        frames = video_frames_from_mp4(video_path, frame_size=frame_size, fps=fps)
        if write_gif(gif_path, frames, fps=fps):
            previews.append({"label": label, "gif_rel": gif_path.name})
    return previews


def collect_sections(list_root: Path, max_per_section: int) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for view in ("raw", "window"):
        for split in ("train", "test"):
            split_dir = list_root / view / split
            if not split_dir.exists():
                continue
            for txt_path in sorted(split_dir.glob("*.txt")):
                if txt_path.stem == "_all_samples":
                    continue
                sample_dirs = [
                    Path(line.strip())
                    for line in txt_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if not sample_dirs:
                    continue
                sections.append(
                    {
                        "view": view,
                        "split": split,
                        "complexity": txt_path.stem,
                        "sample_dirs": sample_dirs[: max(1, int(max_per_section))],
                    }
                )
    return sections


def build_records(
    *,
    list_root: Path,
    output_root: Path,
    frame_size: tuple[int, int],
    fps: float,
    overwrite: bool,
    max_per_section: int,
) -> list[dict[str, Any]]:
    sections_data = collect_sections(list_root, max_per_section=max_per_section)
    records: list[dict[str, Any]] = []
    for section in sections_data:
        title = f"{section['view']} / {section['split']} / {section['complexity']}"
        slug = slugify(title)
        samples: list[dict[str, Any]] = []
        for sample_dir in section["sample_dirs"]:
            sample_slug = slugify(sample_dir.name)
            asset_dir = output_root / "assets" / slug / sample_slug
            ensure_dir(asset_dir)
            if section["view"] == "raw":
                previews = build_raw_preview(
                    sample_dir=sample_dir,
                    output_dir=asset_dir,
                    frame_size=frame_size,
                    fps=fps,
                    overwrite=overwrite,
                )
            else:
                previews = build_window_previews(
                    sample_dir=sample_dir,
                    output_dir=asset_dir,
                    frame_size=frame_size,
                    fps=fps,
                    overwrite=overwrite,
                )
            if not previews:
                continue
            samples.append(
                {
                    "sample_id": sample_dir.name,
                    "sample_dir": str(sample_dir),
                    "dataset": detect_dataset(sample_dir),
                    "view": section["view"],
                    "split": section["split"],
                    "complexity": section["complexity"],
                    "caption": load_caption(sample_dir),
                    "previews": previews,
                    "search_text": " ".join(
                        [
                            sample_dir.name,
                            str(sample_dir),
                            detect_dataset(sample_dir),
                            section["view"],
                            section["split"],
                            section["complexity"],
                            load_caption(sample_dir),
                        ]
                    ).lower(),
                }
            )
        if samples:
            records.append(
                {
                    "title": title,
                    "slug": slug,
                    "search_text": f"{title} {' '.join(sample['search_text'] for sample in samples)}",
                    "samples": samples,
                }
            )
    return records


def write_html(output_root: Path, records: list[dict[str, Any]], fps: float, frame_size: tuple[int, int]) -> None:
    section_options = "\n".join(
        f'<option value="{section["slug"]}">{section["title"]}</option>' for section in records
    )
    html_text = (
        HTML_TEMPLATE.replace("__SECTION_COUNT__", str(len(records)))
        .replace("__SAMPLE_COUNT__", str(sum(len(section["samples"]) for section in records)))
        .replace("__FRAME_WIDTH__", str(frame_size[0]))
        .replace("__FRAME_HEIGHT__", str(frame_size[1]))
        .replace("__FPS__", str(fps))
        .replace("__SECTION_OPTIONS__", section_options)
        .replace("__RECORDS_JSON__", json.dumps(records, ensure_ascii=False))
    )
    (output_root / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    list_root = args.list_root.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    records = build_records(
        list_root=list_root,
        output_root=output_root,
        frame_size=(int(args.width), int(args.height)),
        fps=float(args.fps),
        overwrite=bool(args.overwrite),
        max_per_section=int(args.max_per_section),
    )
    write_json(output_root / "records.json", records)
    write_html(output_root=output_root, records=records, fps=float(args.fps), frame_size=(int(args.width), int(args.height)))

    payload = {
        "output_root": str(output_root),
        "section_count": len(records),
        "sample_count": sum(len(section["samples"]) for section in records),
    }
    if args.serve:
        port = find_free_port(args.host, int(args.port))
        pid, url = start_server(output_root, args.host, port)
        payload["server_pid"] = pid
        payload["url"] = url
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
