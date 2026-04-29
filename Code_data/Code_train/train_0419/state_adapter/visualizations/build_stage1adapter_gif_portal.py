#!/usr/bin/env python3
"""Build a GIF preview portal for stage1adapter samples."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
from PIL import Image, ImageOps


STAGE1ADAPTER_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter"
)
DEFAULT_OUTPUT_DIR = STAGE1ADAPTER_ROOT / "gif_portal"
DEFAULT_MANIFEST_PATH = STAGE1ADAPTER_ROOT / "portal_manifest.json"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Stage1Adapter GIF Portal</title>
  <style>
    :root {{
      --bg:#f1ecdf;
      --bg2:#e3d6c2;
      --panel:rgba(255,252,247,.96);
      --line:rgba(34,24,17,.12);
      --ink:#201812;
      --muted:#6a5d52;
      --accent:#9a5a2b;
      --accent2:#2a6671;
      --shadow:0 18px 40px rgba(39,28,18,.12);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(154,90,43,.14), transparent 24rem),
        radial-gradient(circle at top right, rgba(42,102,113,.11), transparent 28rem),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }}
    main {{
      width:min(1840px, calc(100vw - 18px));
      margin:0 auto;
      padding:10px 0 36px;
    }}
    .hero, .section {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:20px;
      box-shadow:var(--shadow);
    }}
    .hero {{
      padding:16px 18px;
      position:sticky;
      top:8px;
      z-index:5;
      backdrop-filter:blur(8px);
    }}
    h1 {{
      margin:0;
      font-size:clamp(1.6rem, 2.4vw, 2.5rem);
      line-height:1.04;
      letter-spacing:-0.02em;
    }}
    .sub {{
      margin-top:8px;
      color:var(--muted);
      line-height:1.5;
      font-size:.96rem;
    }}
    .stats, .toolbar, .badges {{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }}
    .stats {{
      margin-top:12px;
    }}
    .pill, .badge {{
      border-radius:999px;
      border:1px solid var(--line);
      padding:5px 10px;
      font-size:.82rem;
      background:#fff8ee;
      white-space:nowrap;
    }}
    .toolbar {{
      margin-top:12px;
      align-items:center;
    }}
    .toolbar input, .toolbar select {{
      min-width:180px;
      border:1px solid var(--line);
      border-radius:10px;
      padding:8px 10px;
      background:rgba(255,255,255,.9);
      color:var(--ink);
      font:inherit;
    }}
    .toolbar button {{
      border:1px solid rgba(154,90,43,.18);
      border-radius:10px;
      background:rgba(154,90,43,.08);
      color:var(--accent);
      padding:8px 10px;
      font:inherit;
      cursor:pointer;
    }}
    .sections {{
      display:grid;
      gap:12px;
      margin-top:14px;
    }}
    .section {{
      padding:14px;
    }}
    .section-head {{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:flex-start;
    }}
    .section-head h2 {{
      margin:0;
      font-size:1.15rem;
      line-height:1.08;
    }}
    .meta {{
      margin-top:4px;
      color:var(--muted);
      font-size:.84rem;
      line-height:1.42;
      overflow-wrap:anywhere;
    }}
    .sample-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(420px, 1fr));
      gap:10px;
      margin-top:12px;
    }}
    .sample-card {{
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,255,255,.88);
      padding:10px;
    }}
    .sample-title {{
      display:flex;
      justify-content:space-between;
      gap:8px;
      align-items:flex-start;
    }}
    .sample-title h3 {{
      margin:0;
      font-size:1rem;
      line-height:1.16;
      overflow-wrap:anywhere;
    }}
    .preview-grid {{
      display:grid;
      grid-template-columns:repeat(3, minmax(0, 1fr));
      gap:10px;
      margin-top:10px;
    }}
    .preview {{
      border:1px solid var(--line);
      border-radius:14px;
      padding:8px;
      background:rgba(248,243,236,.8);
    }}
    .preview img {{
      width:100%;
      display:block;
      border-radius:10px;
      border:1px solid var(--line);
      background:#ece3d6;
    }}
    .caption {{
      margin-top:6px;
      color:var(--muted);
      font-size:.78rem;
      line-height:1.4;
      overflow-wrap:anywhere;
    }}
    .links {{
      margin-top:8px;
      font-size:.8rem;
      line-height:1.45;
    }}
    .links a {{
      color:var(--accent2);
      text-decoration:none;
      margin-right:10px;
    }}
    .hidden {{ display:none !important; }}
    @media (max-width: 1100px) {{
      .preview-grid {{ grid-template-columns:1fr; }}
      .sample-grid {{ grid-template-columns:1fr; }}
    }}
    @media (max-width: 980px) {{
      main {{ width:min(100vw - 12px, 1840px); }}
      .hero {{ position:static; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Stage1Adapter GIF Preview Portal</h1>
      <div class="sub">将现有 context / future GT / full 视频转成缩略 GIF 预览，用于本地快速浏览。训练数据本体未修改，只新增 `gif_portal/assets` 预览资产。</div>
      <div class="stats">
        <span class="pill">sections: {section_count}</span>
        <span class="pill">samples: {sample_count}</span>
        <span class="pill">size: {frame_width}x{frame_height}</span>
        <span class="pill">fps: {fps}</span>
        <span class="pill">generated: {generated_at}</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 split / dataset / sample">
        <select id="sectionFilter">
          <option value="">全部 section</option>
          {section_options}
        </select>
        <button id="reloadBtn" type="button">刷新页面</button>
      </div>
    </section>
    <section class="sections" id="sections"></section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const records = JSON.parse(document.getElementById('records').textContent || '[]');
    const root = document.getElementById('sections');
    const searchBox = document.getElementById('searchBox');
    const sectionFilter = document.getElementById('sectionFilter');
    const reloadBtn = document.getElementById('reloadBtn');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    root.innerHTML = records.map((section) => {{
      const cards = section.samples.map((item) => `
        <article class="sample-card" data-section="${{esc(section.slug)}}" data-search="${{esc(item.search_text)}}">
          <div class="sample-title">
            <div>
              <h3>${{esc(item.sample_id)}}</h3>
              <div class="meta">${{esc(item.dataset)}} | ${{esc(item.split)}} | ctx=${{item.context_frames}} | fut=${{item.future_frames}} | full=${{item.full_frames}}</div>
            </div>
            <div class="badges">
              <span class="badge">${{esc(item.dataset)}}</span>
              <span class="badge">${{esc(item.split)}}</span>
            </div>
          </div>
          <div class="preview-grid">
            <div class="preview">
              <img loading="lazy" src="${{encodeURI(item.context_gif_rel)}}" alt="context gif">
              <div class="caption">context GIF</div>
            </div>
            <div class="preview">
              <img loading="lazy" src="${{encodeURI(item.future_gif_rel)}}" alt="future gif">
              <div class="caption">future GT GIF</div>
            </div>
            <div class="preview">
              <img loading="lazy" src="${{encodeURI(item.full_gif_rel)}}" alt="full gif">
              <div class="caption">full GIF</div>
            </div>
          </div>
          <div class="links">
            <a href="${{encodeURI(item.sample_rel)}}" target="_blank" rel="noreferrer">sample dir</a>
            <a href="${{encodeURI(item.meta_rel)}}" target="_blank" rel="noreferrer">meta.json</a>
          </div>
          <div class="caption">${{esc(item.caption)}}</div>
        </article>
      `).join('');
      return `
        <article class="section" data-section="${{esc(section.slug)}}" data-search="${{esc(section.search_text)}}">
          <div class="section-head">
            <div>
              <h2>${{esc(section.title)}}</h2>
              <div class="meta">showing ${{section.samples.length}} samples</div>
            </div>
            <div class="badges"><span class="badge">${{section.samples.length}} samples</span></div>
          </div>
          <div class="sample-grid">${{cards}}</div>
        </article>
      `;
    }}).join('');
    const applyFilter = () => {{
      const q = searchBox.value.trim().toLowerCase();
      const sectionValue = sectionFilter.value;
      for (const sectionNode of root.querySelectorAll('.section')) {{
        const sectionOk = !sectionValue || sectionNode.dataset.section === sectionValue;
        const sectionSearch = sectionNode.dataset.search.toLowerCase();
        let anyCardVisible = false;
        for (const cardNode of sectionNode.querySelectorAll('.sample-card')) {{
          const ok = (!q || cardNode.dataset.search.toLowerCase().includes(q) || sectionSearch.includes(q)) && sectionOk;
          cardNode.classList.toggle('hidden', !ok);
          if (ok) anyCardVisible = true;
        }}
        const sectionVisible = sectionOk && (!q || sectionSearch.includes(q) || anyCardVisible);
        sectionNode.classList.toggle('hidden', !sectionVisible);
      }}
    }};
    searchBox.addEventListener('input', applyFilter);
    sectionFilter.addEventListener('change', applyFilter);
    reloadBtn.addEventListener('click', () => window.location.reload());
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage_root", type=Path, default=STAGE1ADAPTER_ROOT)
    parser.add_argument("--portal_manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8128)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_per_section", type=int, default=0, help="0 means keep all samples")
    parser.add_argument("--force_server", action="store_true", help="Start a new server even if 8117 is already serving stage_root.")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(text: str) -> str:
    chars = []
    for ch in str(text).lower():
        if ch.isalnum():
            chars.append(ch)
        else:
            chars.append("-")
    slug = "".join(chars)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "section"


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
    return process.pid, f"http://{view_host}:{port}/gif_portal/index.html"


def stage_root_server_is_alive() -> bool:
    try:
        result = subprocess.run(
            [
                "curl",
                "-I",
                "--max-time",
                "5",
                "http://127.0.0.1:8117/index.html",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def load_video_frames(video_path: Path, frame_size: tuple[int, int], fps: float) -> list[Image.Image]:
    if not video_path.exists():
        return []
    frames: list[Image.Image] = []
    reader = imageio.get_reader(str(video_path))
    meta = reader.get_meta_data()
    src_fps = float(meta.get("fps", 12.0) or 12.0)
    target_fps = max(float(fps), 0.1)
    stride = max(1, int(round(src_fps / target_fps)))
    try:
        for idx, frame in enumerate(reader):
            if idx % stride != 0:
                continue
            image = Image.fromarray(frame).convert("RGB")
            thumb = ImageOps.contain(image, frame_size)
            canvas = Image.new("RGB", frame_size, (245, 239, 230))
            x_offset = (frame_size[0] - thumb.width) // 2
            y_offset = (frame_size[1] - thumb.height) // 2
            canvas.paste(thumb, (x_offset, y_offset))
            frames.append(canvas)
    finally:
        reader.close()
    if not frames:
        reader = imageio.get_reader(str(video_path))
        try:
            for frame in reader:
                image = Image.fromarray(frame).convert("RGB")
                thumb = ImageOps.contain(image, frame_size)
                canvas = Image.new("RGB", frame_size, (245, 239, 230))
                x_offset = (frame_size[0] - thumb.width) // 2
                y_offset = (frame_size[1] - thumb.height) // 2
                canvas.paste(thumb, (x_offset, y_offset))
                frames.append(canvas)
                break
        finally:
            reader.close()
    return frames


def build_gif_from_video(
    *,
    video_path: Path,
    gif_path: Path,
    frame_size: tuple[int, int],
    fps: float,
    overwrite: bool,
) -> bool:
    if gif_path.exists() and not overwrite and gif_path.stat().st_mtime >= video_path.stat().st_mtime:
        return True
    frames = load_video_frames(video_path, frame_size=frame_size, fps=fps)
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


def rel_from_stage_root(stage_root: Path, path: Path) -> str:
    return path.relative_to(stage_root).as_posix()


def build_records(
    *,
    stage_root: Path,
    manifest_payload: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    frame_size: tuple[int, int],
    fps: float,
    overwrite: bool,
    max_per_section: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for section_title, items in manifest_payload.items():
        section_slug = slugify(section_title)
        section_samples = items if int(max_per_section) <= 0 else items[: int(max_per_section)]
        samples: list[dict[str, Any]] = []
        for item in section_samples:
            sample_id = str(item["sample_id"])
            asset_dir = output_dir / "assets" / section_slug / sample_id
            context_gif = asset_dir / "context.gif"
            future_gif = asset_dir / "future.gif"
            full_gif = asset_dir / "full.gif"
            ok_context = build_gif_from_video(
                video_path=Path(str(item["context_video_path"])),
                gif_path=context_gif,
                frame_size=frame_size,
                fps=fps,
                overwrite=overwrite,
            )
            ok_future = build_gif_from_video(
                video_path=Path(str(item["future_gt_video_path"])),
                gif_path=future_gif,
                frame_size=frame_size,
                fps=fps,
                overwrite=overwrite,
            )
            ok_full = build_gif_from_video(
                video_path=Path(str(item["full_video_path"])),
                gif_path=full_gif,
                frame_size=frame_size,
                fps=fps,
                overwrite=overwrite,
            )
            if not (ok_context and ok_future and ok_full):
                continue
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": str(item.get("dataset", "")),
                    "split": str(item.get("split", "")),
                    "caption": str(item.get("caption", "")),
                    "context_frames": int(item.get("context_frames", 0) or 0),
                    "future_frames": int(item.get("future_frames", 0) or 0),
                    "full_frames": int(item.get("full_frames", 0) or 0),
                    "sample_rel": "../" + str(item.get("rel_dir", "")),
                    "meta_rel": "../" + rel_from_stage_root(stage_root, Path(str(item["meta_json_path"]))),
                    "context_gif_rel": context_gif.relative_to(output_dir).as_posix(),
                    "future_gif_rel": future_gif.relative_to(output_dir).as_posix(),
                    "full_gif_rel": full_gif.relative_to(output_dir).as_posix(),
                    "search_text": " ".join(
                        [
                            section_title,
                            sample_id,
                            str(item.get("dataset", "")),
                            str(item.get("split", "")),
                            str(item.get("caption", "")),
                            str(item.get("rel_dir", "")),
                        ]
                    ),
                }
            )
        records.append(
            {
                "title": section_title,
                "slug": section_slug,
                "samples": samples,
                "search_text": " ".join(
                    [section_title] + [str(sample["search_text"]) for sample in samples]
                ),
            }
        )
    return records


def write_html(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    frame_size: tuple[int, int],
    fps: float,
) -> None:
    sample_count = sum(len(section["samples"]) for section in records)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    section_options = "\n          ".join(
        f'<option value="{section["slug"]}">{section["title"]}</option>' for section in records
    )
    html = HTML_TEMPLATE.format(
        section_count=len(records),
        sample_count=sample_count,
        frame_width=frame_size[0],
        frame_height=frame_size[1],
        fps=fps,
        generated_at=generated_at,
        section_options=section_options,
        records_json=json.dumps(records, ensure_ascii=False),
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage_root = args.stage_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    manifest_payload = dict(load_json(args.portal_manifest.resolve()))
    frame_size = (int(args.width), int(args.height))
    records = build_records(
        stage_root=stage_root,
        manifest_payload=manifest_payload,
        output_dir=output_dir,
        frame_size=frame_size,
        fps=float(args.fps),
        overwrite=bool(args.overwrite),
        max_per_section=int(args.max_per_section),
    )
    write_html(records=records, output_dir=output_dir, frame_size=frame_size, fps=float(args.fps))

    if stage_root_server_is_alive() and not bool(args.force_server):
        url = "http://127.0.0.1:8117/gif_portal/index.html"
        print(json.dumps({"output_dir": str(output_dir), "url": url, "pid": None}, ensure_ascii=False))
        return

    port = find_free_port(args.host, int(args.port))
    pid, url = start_server(stage_root, args.host, port)
    print(json.dumps({"output_dir": str(output_dir), "url": url, "pid": pid}, ensure_ascii=False))


if __name__ == "__main__":
    main()
