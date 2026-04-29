#!/usr/bin/env python3
"""Build a local GIF portal for grouped ABCDE raw/window samples."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_GROUPED_JSON = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/adapter_abcde_curated_group_paths.json"
)
DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/data_summary/adapter_abcde_curated_group_gif_portal"
)

GROUP_TITLES = {
    "A": "A",
    "B": "B",
    "C": "C",
    "D": "D",
    "E": "E",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ABCDE Curated Raw/Window Portal</title>
  <style>
    :root {{
      --bg:#efe7da;
      --bg2:#ddd0ba;
      --panel:rgba(255,252,247,.96);
      --ink:#211913;
      --muted:#6d6157;
      --line:rgba(33,25,19,.1);
      --accent:#915a33;
      --accent2:#34656f;
      --shadow:0 18px 40px rgba(47,31,19,.12);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(145,90,51,.13), transparent 25rem),
        radial-gradient(circle at top right, rgba(52,101,111,.12), transparent 28rem),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }}
    main {{
      width:min(1800px, calc(100vw - 18px));
      margin:0 auto;
      padding:10px 0 36px;
    }}
    .hero, .group {{
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
    .stats, .toolbar, .tags {{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }}
    .stats {{
      margin-top:12px;
    }}
    .pill, .tag {{
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
      border:1px solid rgba(145,90,51,.18);
      border-radius:10px;
      background:rgba(145,90,51,.08);
      color:var(--accent);
      padding:8px 10px;
      font:inherit;
      cursor:pointer;
    }}
    .groups {{
      display:grid;
      gap:12px;
      margin-top:14px;
    }}
    .group {{
      padding:14px;
      border-left:10px solid #8e724f;
    }}
    .group-head {{
      display:flex;
      justify-content:space-between;
      gap:10px;
      align-items:flex-start;
    }}
    .group-head h2 {{
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
    .case-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(360px, 1fr));
      gap:10px;
      margin-top:12px;
    }}
    .case-card {{
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,255,255,.88);
      padding:10px;
    }}
    .case-title {{
      display:flex;
      justify-content:space-between;
      gap:8px;
      align-items:flex-start;
    }}
    .case-title h3 {{
      margin:0;
      font-size:1rem;
      line-height:1.16;
    }}
    .preview-grid {{
      display:grid;
      grid-template-columns:repeat(2, minmax(0, 1fr));
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
    @media (max-width: 980px) {{
      main {{ width:min(100vw - 12px, 1800px); }}
      .hero {{ position:static; }}
      .case-grid {{ grid-template-columns:1fr; }}
      .preview-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>ABCDE Curated Raw / Window RGB Portal</h1>
      <div class="sub">每组展示前 {samples_per_group} 个样本。左侧是 raw 全长 RGB，右侧是 window 对应的 context+future RGB 拼接 GIF，全部自动循环播放。</div>
      <div class="stats">
        <span class="pill">groups: {group_count}</span>
        <span class="pill">samples shown: {sample_count}</span>
        <span class="pill">generated: {generated_at}</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 group / case / path">
        <select id="groupFilter">
          <option value="">全部分组</option>
          <option value="A">A</option>
          <option value="B">B</option>
          <option value="C">C</option>
          <option value="D">D</option>
          <option value="E">E</option>
        </select>
        <button id="reloadBtn" type="button">刷新页面</button>
      </div>
    </section>
    <section class="groups" id="groups"></section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const records = JSON.parse(document.getElementById('records').textContent || '[]');
    const root = document.getElementById('groups');
    const searchBox = document.getElementById('searchBox');
    const groupFilter = document.getElementById('groupFilter');
    const reloadBtn = document.getElementById('reloadBtn');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    root.innerHTML = records.map((group) => {{
      const casesHtml = group.samples.map((item) => `
        <article class="case-card" data-group="${{esc(group.group)}}" data-search="${{esc(item.search_text)}}">
          <div class="case-title">
            <div>
              <h3>${{esc(item.sample_name)}}</h3>
              <div class="meta">group=${{esc(group.group)}} | ctx=${{item.context_len}} | fut=${{item.future_len}} | start=${{item.start_index}}</div>
            </div>
            <div class="tag">${{esc(item.case_family)}}</div>
          </div>
          <div class="preview-grid">
            <div class="preview">
              <img loading="lazy" src="${{encodeURI(item.raw_gif_rel)}}" alt="raw gif">
              <div class="caption">raw RGB</div>
            </div>
            <div class="preview">
              <img loading="lazy" src="${{encodeURI(item.window_gif_rel)}}" alt="window gif">
              <div class="caption">window RGB: context then future</div>
            </div>
          </div>
          <div class="links">
            <a href="${{encodeURI(item.raw_dir_rel)}}" target="_blank" rel="noreferrer">raw dir</a>
            <a href="${{encodeURI(item.window_dir_rel)}}" target="_blank" rel="noreferrer">window dir</a>
            <a href="${{encodeURI(item.pair_meta_rel)}}" target="_blank" rel="noreferrer">pair_meta</a>
          </div>
          <div class="caption">${{esc(item.raw_path_short)}}<br>${{esc(item.window_path_short)}}</div>
        </article>
      `).join('');
      return `
        <article class="group" data-group="${{esc(group.group)}}" data-search="${{esc(group.search_text)}}">
          <div class="group-head">
            <div>
              <h2>${{esc(group.title)}}</h2>
              <div class="meta">showing ${{group.samples.length}} samples</div>
            </div>
            <div class="tags"><span class="tag">group ${{esc(group.group)}}</span></div>
          </div>
          <div class="case-grid">${{casesHtml}}</div>
        </article>
      `;
    }}).join('');
    const applyFilter = () => {{
      const q = searchBox.value.trim().toLowerCase();
      const groupValue = groupFilter.value;
      for (const groupNode of root.querySelectorAll('.group')) {{
        const groupOk = !groupValue || groupNode.dataset.group === groupValue;
        const groupSearch = groupNode.dataset.search.toLowerCase();
        let anyCaseVisible = false;
        for (const caseNode of groupNode.querySelectorAll('.case-card')) {{
          const ok = (!q || caseNode.dataset.search.toLowerCase().includes(q) || groupSearch.includes(q)) && groupOk;
          caseNode.classList.toggle('hidden', !ok);
          if (ok) anyCaseVisible = true;
        }}
        const groupVisible = groupOk && (!q || groupSearch.includes(q) || anyCaseVisible);
        groupNode.classList.toggle('hidden', !groupVisible);
      }}
    }};
    searchBox.addEventListener('input', applyFilter);
    groupFilter.addEventListener('change', applyFilter);
    reloadBtn.addEventListener('click', () => window.location.reload());
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped_json", type=Path, default=DEFAULT_GROUPED_JSON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--samples_per_group", type=int, default=10)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8126)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--label_height", type=int, default=24)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"refusing to replace real directory: {dst}")
        dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def frame_to_canvas(
    frame_path: Path,
    frame_size: tuple[int, int],
    label: str,
    label_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    with Image.open(frame_path) as image:
        thumb = ImageOps.contain(image.convert("RGB"), frame_size)
    canvas = Image.new("RGB", (frame_size[0], frame_size[1] + label_height), (245, 239, 230))
    x_offset = (frame_size[0] - thumb.width) // 2
    y_offset = (frame_size[1] - thumb.height) // 2
    canvas.paste(thumb, (x_offset, y_offset))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        [(0, frame_size[1]), (frame_size[0], frame_size[1] + label_height)],
        fill=(54, 46, 40),
    )
    draw.text((8, frame_size[1] + 6), label, fill=(248, 244, 236), font=font)
    return canvas


def build_gif(
    frame_paths: list[Path],
    dst: Path,
    frame_size: tuple[int, int],
    fps: float,
    labels: list[str],
    label_height: int,
) -> None:
    if not frame_paths:
        return
    ensure_dir(dst.parent)
    font = get_font()
    frames: list[Image.Image] = []
    for frame_path, label in zip(frame_paths, labels):
        frames.append(frame_to_canvas(frame_path, frame_size, label, label_height, font))
    duration_ms = max(1, int(round(1000.0 / max(fps, 0.1))))
    first, *rest = frames
    first.save(
        dst,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    for frame in frames:
        frame.close()


def shorten_path(path: Path) -> str:
    text = path.as_posix()
    prefix = DATASET_ROOT.as_posix().rstrip("/") + "/"
    return text.replace(prefix, "")


def find_free_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
        return port
    raise RuntimeError(f"no free port found near {preferred}")


def start_server(output_dir: Path, host: str, port: int) -> tuple[int, str]:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "http.server",
            str(port),
            "--bind",
            host,
            "--directory",
            str(output_dir),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    view_host = "127.0.0.1" if host == "0.0.0.0" else host
    return process.pid, f"http://{view_host}:{port}/index.html"


def collect_raw_frames(raw_dir: Path) -> list[Path]:
    rgb_dir = raw_dir / "rgb"
    if not rgb_dir.exists():
        return []
    return sorted(rgb_dir.glob("*.png"))


def load_pair_meta(window_dir: Path) -> dict[str, Any]:
    return dict(load_json(window_dir / "pair_meta.json"))


def build_sample_record(
    group_name: str,
    item: dict[str, str],
    sample_index: int,
    output_dir: Path,
    frame_size: tuple[int, int],
    fps: float,
    label_height: int,
) -> dict[str, Any]:
    raw_dir = Path(item["raw"]).resolve()
    window_dir = Path(item["window"]).resolve()
    pair_meta = load_pair_meta(window_dir)
    raw_frames = collect_raw_frames(raw_dir)
    context_frames = [Path(path) for path in pair_meta.get("x_frame_paths", [])]
    future_frames = [Path(path) for path in pair_meta.get("y_frame_paths", [])]
    window_frames = context_frames + future_frames
    sample_name = raw_dir.name
    asset_dir = output_dir / "assets" / group_name / f"{sample_index:02d}_{sample_name}"
    raw_gif = asset_dir / "raw.gif"
    window_gif = asset_dir / "window.gif"
    build_gif(
        raw_frames,
        raw_gif,
        frame_size=frame_size,
        fps=fps,
        labels=[f"RAW {i:02d}" for i in range(len(raw_frames))],
        label_height=label_height,
    )
    window_labels = (
        [f"CTX {i:02d}" for i in range(len(context_frames))]
        + [f"FUT {i:02d}" for i in range(len(future_frames))]
    )
    build_gif(
        window_frames,
        window_gif,
        frame_size=frame_size,
        fps=fps,
        labels=window_labels,
        label_height=label_height,
    )
    case_family = raw_dir.parent.name
    return {
        "sample_name": sample_name,
        "case_family": case_family,
        "context_len": int(pair_meta.get("context_len", len(context_frames))),
        "future_len": int(pair_meta.get("future_len", len(future_frames))),
        "start_index": int(pair_meta.get("start_index", 0)),
        "raw_gif_rel": raw_gif.relative_to(output_dir).as_posix(),
        "window_gif_rel": window_gif.relative_to(output_dir).as_posix(),
        "raw_dir_rel": "data_root/" + shorten_path(raw_dir),
        "window_dir_rel": "data_root/" + shorten_path(window_dir),
        "pair_meta_rel": "data_root/" + shorten_path(window_dir / "pair_meta.json"),
        "raw_path_short": shorten_path(raw_dir),
        "window_path_short": shorten_path(window_dir),
        "search_text": " ".join(
            [
                group_name,
                sample_name,
                case_family,
                raw_dir.as_posix(),
                window_dir.as_posix(),
            ]
        ),
    }


def build_records(
    grouped_payload: dict[str, list[dict[str, str]]],
    output_dir: Path,
    samples_per_group: int,
    frame_size: tuple[int, int],
    fps: float,
    label_height: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group_name in sorted(grouped_payload):
        items = grouped_payload[group_name][:samples_per_group]
        samples = [
            build_sample_record(
                group_name=group_name,
                item=dict(item),
                sample_index=index,
                output_dir=output_dir,
                frame_size=frame_size,
                fps=fps,
                label_height=label_height,
            )
            for index, item in enumerate(items)
        ]
        records.append(
            {
                "group": group_name,
                "title": GROUP_TITLES.get(group_name, group_name),
                "samples": samples,
                "search_text": " ".join(
                    [group_name] + [sample["search_text"] for sample in samples]
                ),
            }
        )
    return records


def write_html(
    records: list[dict[str, Any]],
    output_dir: Path,
    samples_per_group: int,
) -> None:
    sample_count = sum(len(group["samples"]) for group in records)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    html = HTML_TEMPLATE.format(
        samples_per_group=samples_per_group,
        group_count=len(records),
        sample_count=sample_count,
        generated_at=generated_at,
        records_json=json.dumps(records, ensure_ascii=False),
    )
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    safe_symlink(output_dir / "data_root", DATASET_ROOT)
    grouped_payload = dict(load_json(args.grouped_json.resolve()))
    frame_size = (args.width, args.height)
    records = build_records(
        grouped_payload=grouped_payload,
        output_dir=output_dir,
        samples_per_group=args.samples_per_group,
        frame_size=frame_size,
        fps=args.fps,
        label_height=args.label_height,
    )
    write_html(records, output_dir, args.samples_per_group)
    port = find_free_port(args.host, args.port)
    pid, url = start_server(output_dir, args.host, port)
    print(json.dumps({"output_dir": str(output_dir), "pid": pid, "url": url}, ensure_ascii=False))


if __name__ == "__main__":
    main()
