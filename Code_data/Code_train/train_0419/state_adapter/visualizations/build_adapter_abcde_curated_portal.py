#!/usr/bin/env python3
"""Build a local HTML portal for the curated ABCDE adapter dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps


RAW_MANIFEST = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/adapter_abcde_curated/raw_manifest_abcde_curated.json"
)
WINDOW_MANIFEST = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/adapter_abcde_curated/window_manifest_abcde_curated.json"
)
CASE_JSON = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.json"
)
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/adapter_abcde_curated")

GROUP_TITLES = {
    "A": "A: 单物体静置锚点",
    "B": "B: 单物体重力下坠",
    "C": "C: 单物体入场平移",
    "D": "D: 双物体无互撞独立运动",
    "E": "E: 更自由的单物体抛射",
}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Adapter ABCDE Curated Portal</title>
  <style>
    :root {{
      --bg:#f0eadf;
      --bg2:#ddd4c3;
      --panel:rgba(255,255,252,.94);
      --ink:#201912;
      --muted:#675b50;
      --line:rgba(32,25,18,.1);
      --accent:#8f4a2d;
      --accent2:#2a6464;
      --shadow:0 18px 40px rgba(56,40,28,.12);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(143,74,45,.13), transparent 24rem),
        radial-gradient(circle at top right, rgba(42,100,100,.12), transparent 28rem),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }}
    main {{
      width:min(1850px, calc(100vw - 20px));
      margin:0 auto;
      padding:12px 0 40px;
    }}
    .hero, .group {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:20px;
      box-shadow:var(--shadow);
    }}
    .hero {{
      padding:16px 18px 14px;
      position:sticky;
      top:8px;
      z-index:5;
      backdrop-filter:blur(8px);
    }}
    h1 {{
      margin:0;
      font-size:clamp(1.6rem, 2.4vw, 2.6rem);
      line-height:1.03;
      letter-spacing:-0.02em;
    }}
    .sub {{
      margin-top:8px;
      color:var(--muted);
      line-height:1.5;
      font-size:.96rem;
    }}
    .stats, .toolbar, .tags, .links {{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }}
    .stats {{
      margin-top:12px;
    }}
    .pill, .tag, .badge {{
      border-radius:999px;
      border:1px solid var(--line);
      padding:5px 10px;
      font-size:.82rem;
      background:#fff9ef;
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
      background:rgba(255,255,255,.86);
      color:var(--ink);
      font:inherit;
    }}
    .toolbar button {{
      border:1px solid rgba(143,74,45,.18);
      border-radius:10px;
      background:rgba(143,74,45,.08);
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
      border-left:10px solid #92714c;
    }}
    .group-head {{
      display:flex;
      justify-content:space-between;
      gap:12px;
      align-items:flex-start;
    }}
    .group-head h2 {{
      margin:0;
      font-size:1.16rem;
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
      background:rgba(255,255,255,.86);
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
      line-height:1.15;
    }}
    .desc {{
      margin-top:6px;
      color:var(--muted);
      font-size:.86rem;
      line-height:1.45;
    }}
    .split {{
      margin-top:10px;
      border:1px solid var(--line);
      border-radius:14px;
      padding:10px;
      background:rgba(251,248,241,.72);
    }}
    .split h4 {{
      margin:0 0 8px;
      font-size:.95rem;
    }}
    .preview-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
      gap:8px;
    }}
    .preview video, .preview img {{
      width:100%;
      display:block;
      border-radius:12px;
      border:1px solid var(--line);
      background:#efe8db;
    }}
    .caption {{
      margin-top:5px;
      color:var(--muted);
      font-size:.76rem;
      line-height:1.35;
      overflow-wrap:anywhere;
    }}
    .links a {{
      color:var(--accent2);
      text-decoration:none;
    }}
    .muted {{
      color:var(--muted);
      font-size:.84rem;
    }}
    .hidden {{ display:none !important; }}
    @media (max-width: 980px) {{
      main {{ width:min(100vw - 12px, 1850px); }}
      .hero {{ position:static; }}
      .group-head {{ display:block; }}
      .case-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Adapter ABCDE Curated</h1>
      <div class="sub">这页只展示你已经确认过口径的 ABCDE curated 数据。每个 case 同时给 raw 版本和 preprocess 后的 window 版本，方便直接核对训练输入是否和原始样本一致。</div>
      <div class="stats">
        <span class="pill">total raw: {total_raw}</span>
        <span class="pill">total windows: {total_windows}</span>
        <span class="pill">future lens: {future_lengths}</span>
        <span class="pill">generated: {generated_at}</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 group / case / motion">
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
      const casesHtml = group.cases.map((item) => {{
        const rawHtml = item.raw_previews.map((preview) => `
          <div class="preview">
            <video controls muted preload="metadata" src="${{encodeURI(preview.video_rel)}}"></video>
            <div class="caption">${{esc(preview.sample_name)}}<br>${{esc(preview.group_dir)}}</div>
          </div>
        `).join('');
        const windowHtml = item.window_previews.map((preview) => `
          <div class="preview">
            <div class="caption">context</div>
            <video controls muted preload="metadata" src="${{encodeURI(preview.context_video_rel)}}"></video>
            <div class="caption" style="margin-top:6px;">future</div>
            <video controls muted preload="metadata" src="${{encodeURI(preview.future_video_rel)}}"></video>
            <div class="caption">src=${{esc(preview.sample_name)}} | fut=${{preview.future_len}} | start=${{preview.start_index}}</div>
            <div class="links">
              <a href="${{encodeURI(preview.pair_meta_rel)}}" target="_blank" rel="noreferrer">pair_meta</a>
              <a href="${{encodeURI(preview.window_dir_rel)}}" target="_blank" rel="noreferrer">window dir</a>
              <a href="${{encodeURI(preview.source_video_rel)}}" target="_blank" rel="noreferrer">source video</a>
            </div>
          </div>
        `).join('');
        const tags = [
          `raw ${item.raw_count}`,
          `windows ${item.window_count}`,
          `case ${item.case_id}`,
        ].map((v) => `<span class="tag">${{esc(v)}}</span>`).join('');
        return `
          <article class="case-card" data-group="${{esc(group.group)}}" data-search="${{esc(item.search_text)}}">
            <div class="case-title">
              <div>
                <h3>${{esc(item.case_name)}}</h3>
                <div class="meta">${{esc(item.motion_cn)}}</div>
              </div>
              <div class="badge">${{esc(group.group)}}</div>
            </div>
            <div class="tags" style="margin-top:8px;">${{tags}}</div>
            <div class="split">
              <h4>Raw Version</h4>
              <div class="preview-grid">${{rawHtml || '<div class="muted">暂无 raw 预览。</div>'}}</div>
            </div>
            <div class="split">
              <h4>Window Version</h4>
              <div class="preview-grid">${{windowHtml || '<div class="muted">暂无 window 预览。</div>'}}</div>
            </div>
          </article>
        `;
      }}).join('');
      return `
        <article class="group" data-group="${{esc(group.group)}}" data-search="${{esc(group.search_text)}}">
          <div class="group-head">
            <div>
              <h2>${{esc(group.title)}}</h2>
              <div class="meta">raw=${{group.raw_count}} | windows=${{group.window_count}} | cases=${{group.cases.length}}</div>
            </div>
            <div class="tags">
              <span class="tag">group ${{esc(group.group)}}</span>
            </div>
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def build_strip(frame_paths: list[str], dst: Path, thumb_size: tuple[int, int] = (160, 120)) -> None:
    images: list[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), thumb_size)
            canvas = Image.new("RGB", thumb_size, (247, 243, 235))
            offset = ((thumb_size[0] - thumb.width) // 2, (thumb_size[1] - thumb.height) // 2)
            canvas.paste(thumb, offset)
            images.append(canvas)
    if not images:
        return
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    strip = Image.new("RGB", (width, height), (250, 246, 240))
    cursor = 0
    for image in images:
        strip.paste(image, (cursor, 0))
        cursor += image.width
    ensure_dir(dst.parent)
    strip.save(dst, quality=90)
    for image in images:
        image.close()
    strip.close()


def build_video_preview(
    frame_paths: list[str],
    dst: Path,
    frame_size: tuple[int, int] = (480, 360),
    fps: float = 12.0,
) -> None:
    if not frame_paths:
        return
    ensure_dir(dst.parent)
    with imageio.get_writer(
        str(dst),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
        quality=8,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ) as writer:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                thumb = ImageOps.contain(image.convert("RGB"), frame_size)
                canvas = Image.new("RGB", frame_size, (247, 243, 235))
                offset = ((frame_size[0] - thumb.width) // 2, (frame_size[1] - thumb.height) // 2)
                canvas.paste(thumb, offset)
                writer.append_data(np.asarray(canvas))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_manifest", type=Path, default=RAW_MANIFEST)
    parser.add_argument("--window_manifest", type=Path, default=WINDOW_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8112)
    parser.add_argument("--raw_previews_per_case", type=int, default=2)
    parser.add_argument("--window_previews_per_case", type=int, default=2)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def start_server(output_dir: Path, host: str, port: int) -> tuple[int, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
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
    return process.pid, f"http://{host}:{port}/index.html"


def make_window_preview(
    preview_index: int,
    record: dict[str, Any],
    output_dir: Path,
    raw_root: Path,
    window_root: Path,
) -> dict[str, Any]:
    window_dir = Path(record["window_dir"]).resolve()
    pair_meta_path = window_dir / "pair_meta.json"
    meta = load_json(pair_meta_path)
    case_name = str(record["case_name"])
    sample_name = Path(str(record["source_sample_dir"])).name
    asset_dir = output_dir / "assets" / case_name / f"window_{preview_index:02d}"
    context_video = asset_dir / "context.mp4"
    future_video = asset_dir / "future.mp4"
    build_video_preview(list(meta.get("x_frame_paths", [])), context_video)
    build_video_preview(list(meta.get("y_frame_paths", [])), future_video)
    source_sample_dir = Path(str(record["source_sample_dir"])).resolve()
    source_video_rel = "raw/" + source_sample_dir.relative_to(raw_root).as_posix() + "/videos/rgb.mp4"
    rel_window_dir = window_dir.relative_to(window_root)
    return {
        "sample_name": sample_name,
        "future_len": int(record["future_len"]),
        "start_index": int(record["start_index"]),
        "context_video_rel": str(context_video.relative_to(output_dir).as_posix()),
        "future_video_rel": str(future_video.relative_to(output_dir).as_posix()),
        "pair_meta_rel": "windows/" + str((rel_window_dir / "pair_meta.json").as_posix()),
        "window_dir_rel": "windows/" + str(rel_window_dir.as_posix()),
        "source_video_rel": source_video_rel,
    }


def main() -> None:
    args = parse_args()
    raw_payload = load_json(args.raw_manifest.resolve())
    window_payload = load_json(args.window_manifest.resolve())
    case_payload = load_json(CASE_JSON.resolve())
    case_defs = {int(key): dict(value) for key, value in dict(case_payload["explicit_cases"]).items()}
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    ensure_dir(output_dir / "assets")

    raw_root = Path(str(raw_payload["raw_root"])).resolve()
    window_root = Path(str(window_payload["window_root"])).resolve()
    safe_symlink(output_dir / "raw", raw_root)
    safe_symlink(output_dir / "windows", window_root)

    raw_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_payload["records"]:
        raw_by_case[str(record["case_name"])].append(dict(record))

    window_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in window_payload["records"]:
        window_by_case[str(record["case_name"])].append(dict(record))

    case_meta: dict[str, dict[str, Any]] = {}
    for records in raw_by_case.values():
        if not records:
            continue
        first = records[0]
        case_id = int(first["case_id"])
        case_meta[str(first["case_name"])] = {
            "case_id": case_id,
            "motion_cn": str(case_defs.get(case_id, {}).get("motion_cn", "")),
        }

    records_out: list[dict[str, Any]] = []
    for group in ("A", "B", "C", "D", "E"):
        case_names = sorted(
            [case_name for case_name, items in raw_by_case.items() if items and str(items[0]["group"]) == group],
            key=lambda name: (case_meta.get(name, {}).get("case_id", 0), name),
        )
        case_cards: list[dict[str, Any]] = []
        group_raw_count = 0
        group_window_count = 0
        search_tokens = [group, GROUP_TITLES.get(group, group)]
        for case_name in case_names:
            raw_records = raw_by_case.get(case_name, [])
            window_records = window_by_case.get(case_name, [])
            group_raw_count += len(raw_records)
            group_window_count += len(window_records)
            motion_cn = str(case_meta.get(case_name, {}).get("motion_cn", ""))

            raw_previews = []
            for record in raw_records[: int(args.raw_previews_per_case)]:
                sample_dir = Path(str(record["sample_dir"]))
                rel = sample_dir.relative_to(raw_root)
                raw_previews.append(
                    {
                        "sample_name": sample_dir.name,
                        "group_dir": "/".join(rel.parts[:-1]),
                        "video_rel": "raw/" + rel.as_posix() + "/videos/rgb.mp4",
                    }
                )

            window_previews = []
            for preview_index, record in enumerate(window_records[: int(args.window_previews_per_case)]):
                window_previews.append(
                    make_window_preview(preview_index, record, output_dir, raw_root=raw_root, window_root=window_root)
                )

            case_id = int(raw_records[0]["case_id"])
            item = {
                "case_id": case_id,
                "case_name": case_name,
                "motion_cn": motion_cn,
                "raw_count": len(raw_records),
                "window_count": len(window_records),
                "raw_previews": raw_previews,
                "window_previews": window_previews,
            }
            item["search_text"] = " ".join(
                [group, case_name, motion_cn, str(case_id), str(len(raw_records)), str(len(window_records))]
            ).lower()
            case_cards.append(item)
            search_tokens.extend([case_name, motion_cn, str(case_id)])

        records_out.append(
            {
                "group": group,
                "title": GROUP_TITLES.get(group, group),
                "raw_count": group_raw_count,
                "window_count": group_window_count,
                "cases": case_cards,
                "search_text": " ".join(search_tokens).lower(),
            }
        )

    html_text = HTML_TEMPLATE
    replacements = {
        "{total_raw}": str(int(raw_payload["total_raw_samples"])),
        "{total_windows}": str(int(window_payload["total_windows"])),
        "{future_lengths}": ",".join(sorted(window_payload.get("future_len_counts", {}).keys(), key=int)),
        "{generated_at}": os.popen("date -u '+%Y-%m-%d %H:%M UTC'").read().strip(),
        "{records_json}": json.dumps(records_out, ensure_ascii=False),
    }
    for needle, value in replacements.items():
        html_text = html_text.replace(needle, value)
    html_text = html_text.replace("{{", "{").replace("}}", "}")
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    manifest = {
        "raw_manifest": str(args.raw_manifest.resolve()),
        "window_manifest": str(args.window_manifest.resolve()),
        "records": records_out,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pid, url = start_server(output_dir, host=str(args.host), port=int(args.port))
    print(f"output_dir={output_dir}")
    print(f"pid={pid}")
    print(f"url={url}")


if __name__ == "__main__":
    main()
