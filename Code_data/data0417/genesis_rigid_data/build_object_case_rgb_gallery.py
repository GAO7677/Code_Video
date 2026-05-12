#!/usr/bin/env python3
# 用途：按主物体与 case 汇总 RGB 样本画廊。
"""Build a local HTML gallery grouped by main object id and case RGB GIFs."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/object_case_rgb_gallery_current")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Genesis Object Case RGB Gallery</title>
  <style>
    :root {{
      --bg:#efe7da;
      --bg2:#e1d5c2;
      --panel:rgba(255,251,245,.94);
      --panel-soft:rgba(255,255,255,.72);
      --ink:#201811;
      --muted:#6e6155;
      --line:rgba(32,24,17,.11);
      --accent:#9a4028;
      --accent-2:#2f6c73;
      --shadow:0 22px 48px rgba(58,38,21,.12);
      --chip:#fff7ef;
      --factual:#f5efe4;
      --same:#fce3cf;
      --none:#e0efde;
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(154,64,40,.16), transparent 24rem),
        radial-gradient(circle at top right, rgba(47,108,115,.16), transparent 26rem),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }}
    main {{
      width:min(1880px, calc(100vw - 20px));
      margin:0 auto;
      padding:12px 0 36px;
    }}
    .hero, .object {{
      border:1px solid var(--line);
      border-radius:20px;
      box-shadow:var(--shadow);
      background:var(--panel);
      backdrop-filter:blur(8px);
    }}
    .hero {{
      position:sticky;
      top:8px;
      z-index:5;
      padding:16px 18px 14px;
    }}
    h1 {{
      margin:0;
      font-size:clamp(1.6rem, 2.4vw, 2.7rem);
      line-height:1.02;
      letter-spacing:-0.02em;
    }}
    .sub {{
      margin-top:8px;
      color:var(--muted);
      font-size:.96rem;
      line-height:1.48;
    }}
    .stats, .toolbar, .object-index, .object-tags, .section-tags, .card-links {{
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
      background:var(--chip);
      padding:5px 10px;
      font-size:.82rem;
      white-space:nowrap;
    }}
    .toolbar {{
      margin-top:12px;
      align-items:center;
    }}
    .toolbar input, .toolbar select {{
      min-width:170px;
      border-radius:10px;
      border:1px solid var(--line);
      background:rgba(255,255,255,.86);
      padding:8px 10px;
      font:inherit;
      color:var(--ink);
    }}
    .toolbar button {{
      border-radius:10px;
      border:1px solid rgba(154,64,40,.18);
      background:rgba(154,64,40,.08);
      color:var(--accent);
      padding:8px 10px;
      font:inherit;
      cursor:pointer;
    }}
    .toolbar button:hover {{
      background:rgba(154,64,40,.14);
    }}
    .object-index {{
      margin-top:12px;
      max-height:160px;
      overflow:auto;
      padding-right:4px;
    }}
    .object-index a {{
      text-decoration:none;
      color:var(--accent);
      border:1px solid var(--line);
      border-radius:999px;
      background:rgba(255,255,255,.64);
      padding:6px 10px;
      font-size:.8rem;
    }}
    .objects {{
      display:grid;
      gap:12px;
      margin-top:14px;
    }}
    .object {{
      padding:14px;
    }}
    .object-head {{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:12px;
    }}
    .object-title h2 {{
      margin:0;
      font-size:1.18rem;
      line-height:1.1;
      letter-spacing:-0.01em;
    }}
    .meta {{
      margin-top:4px;
      color:var(--muted);
      font-size:.84rem;
      line-height:1.4;
      overflow-wrap:anywhere;
    }}
    .sections {{
      display:grid;
      gap:10px;
      margin-top:12px;
    }}
    .section {{
      border:1px solid var(--line);
      border-radius:16px;
      background:var(--panel-soft);
      padding:10px;
    }}
    .section-head {{
      display:flex;
      align-items:flex-start;
      justify-content:space-between;
      gap:10px;
      margin-bottom:8px;
    }}
    .section-head h3 {{
      margin:0;
      font-size:.98rem;
      line-height:1.15;
    }}
    .video-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
      gap:8px;
    }}
    .card {{
      border:1px solid var(--line);
      border-radius:14px;
      background:rgba(255,255,255,.88);
      padding:8px;
      min-width:0;
    }}
    .card.factual {{ background:linear-gradient(180deg, rgba(245,239,228,.96), rgba(255,255,255,.92)); }}
    .card.same_scene_negative {{ background:linear-gradient(180deg, rgba(252,227,207,.92), rgba(255,255,255,.92)); }}
    .card.no_collision_negative {{ background:linear-gradient(180deg, rgba(224,239,222,.94), rgba(255,255,255,.92)); }}
    .card-top {{
      display:flex;
      justify-content:space-between;
      align-items:flex-start;
      gap:6px;
      margin-bottom:6px;
    }}
    .card-name {{
      font-size:.83rem;
      line-height:1.2;
      font-weight:700;
      overflow-wrap:anywhere;
    }}
    .card-role {{
      flex:0 0 auto;
      color:var(--muted);
      font-size:.72rem;
      white-space:nowrap;
    }}
    .card img {{
      width:100%;
      aspect-ratio:4/3;
      display:block;
      border-radius:10px;
      background:#111;
      object-fit:contain;
    }}
    .card-links {{
      justify-content:space-between;
      margin-top:6px;
      font-size:.73rem;
    }}
    .card-links a {{
      text-decoration:none;
      color:var(--accent-2);
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }}
    .hidden {{ display:none !important; }}
    @media (max-width: 980px) {{
      main {{ width:min(100vw - 12px, 1880px); }}
      .hero {{ position:static; }}
      .object-head, .section-head {{ display:block; }}
      .object-tags, .section-tags {{ margin-top:8px; }}
      .video-grid {{ grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Genesis Object Case RGB Gallery</h1>
      <div class="sub">按主物体 <code>object_id</code> 聚合当前数据集里已经落盘的所有 case。每个 object 下再按实际目录组别展开，卡片默认显示 <code>rgb.gif</code>，同时保留原视频和 metadata 入口。</div>
      <div class="stats">
        <span class="pill">objects: {object_count}</span>
        <span class="pill">samples: {sample_count}</span>
        <span class="pill">updated: {updated_at}</span>
        <span class="pill">dataset: {dataset_root}</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 object_id / case / bucket / scene">
        <select id="bucketFilter">
          <option value="">全部 group</option>
        </select>
        <select id="roleFilter">
          <option value="">全部 role</option>
          <option value="factual">factual</option>
          <option value="same_scene_negative">same-scene negative</option>
          <option value="no_collision_negative">no-collision negative</option>
        </select>
        <button id="reloadBtn" type="button">刷新页面</button>
        <button id="eagerBtn" type="button">加载全部 gif</button>
      </div>
      <div class="object-index" id="objectIndex"></div>
    </section>
    <section class="objects" id="objects"></section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const records = JSON.parse(document.getElementById('records').textContent || '[]');
    const root = document.getElementById('objects');
    const objectIndex = document.getElementById('objectIndex');
    const searchBox = document.getElementById('searchBox');
    const bucketFilter = document.getElementById('bucketFilter');
    const roleFilter = document.getElementById('roleFilter');
    const reloadBtn = document.getElementById('reloadBtn');
    const eagerBtn = document.getElementById('eagerBtn');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    const roleLabel = (role) => {{
      if (role === 'same_scene_negative') return 'cf:same-scene';
      if (role === 'no_collision_negative') return 'cf:no-collision';
      return 'factual';
    }};
    const bucketValues = [...new Set(records.flatMap((obj) => obj.sections.map((sec) => sec.group_key)).filter(Boolean))].sort();
    bucketFilter.innerHTML += bucketValues.map((value) => `<option value="${{esc(value)}}">${{esc(value)}}</option>`).join('');
    objectIndex.innerHTML = records.map((obj) => `<a href="#${{encodeURIComponent(obj.anchor)}}">${{esc(obj.object_id)}} · ${{obj.sample_count}}</a>`).join('');
    root.innerHTML = records.map((obj) => {{
      const objectTags = [
        `groups ${{obj.group_count}}`,
        `samples ${{obj.sample_count}}`,
        `factual ${{obj.factual_count}}`,
        `same-scene ${{obj.same_scene_count}}`,
        `no-collision ${{obj.no_collision_count}}`,
      ].map((v) => `<span class="tag">${{esc(v)}}</span>`).join('');
      const sectionsHtml = obj.sections.map((sec) => {{
        const sectionTags = [
          `scene ${{sec.scene_composition}}`,
          `bucket ${{sec.count_bucket}}`,
          `items ${{sec.item_count}}`,
        ].map((v) => `<span class="tag">${{esc(v)}}</span>`).join('');
        const cardsHtml = sec.items.map((item) => `
          <article class="card ${{esc(item.role_kind)}}" data-role="${{esc(item.role_kind)}}" data-search="${{esc(item.search_text)}}">
            <div class="card-top">
              <div class="card-name">${{esc(item.case_label)}}</div>
              <div class="card-role">${{esc(roleLabel(item.role_kind))}}</div>
            </div>
            <img loading="lazy" src="${{encodeURI(item.gif_preview)}}" alt="${{esc(item.case_label)}} rgb gif">
            <div class="card-links">
              <a href="${{encodeURI(item.gif_preview)}}" target="_blank" rel="noreferrer">gif</a>
              <a href="${{encodeURI(item.rgb_video)}}" target="_blank" rel="noreferrer">video</a>
              <a href="${{encodeURI(item.metadata_path)}}" target="_blank" rel="noreferrer">meta</a>
            </div>
          </article>`).join('');
        return `
          <section class="section" data-group-key="${{esc(sec.group_key)}}" data-search="${{esc(sec.search_text)}}">
            <div class="section-head">
              <div>
                <h3>${{esc(sec.group_key)}}</h3>
                <div class="meta">${{esc(sec.rel_group_dir)}}</div>
              </div>
              <div class="section-tags">${{sectionTags}}</div>
            </div>
            <div class="video-grid">${{cardsHtml}}</div>
          </section>`;
      }}).join('');
      return `
        <article
          class="object"
          id="${{encodeURIComponent(obj.anchor)}}"
          data-search="${{esc(obj.search_text)}}"
        >
          <div class="object-head">
            <div class="object-title">
              <h2>object_id = ${{esc(obj.object_id)}}</h2>
              <div class="meta">当前 object 共有 ${{obj.group_count}} 个组别，${{obj.sample_count}} 个样本</div>
            </div>
            <div class="object-tags">${{objectTags}}</div>
          </div>
          <div class="sections">${{sectionsHtml}}</div>
        </article>`;
    }}).join('');

    const applyFilter = () => {{
      const q = searchBox.value.trim().toLowerCase();
      const groupValue = bucketFilter.value;
      const roleValue = roleFilter.value;
      for (const objectNode of root.querySelectorAll('.object')) {{
        let anySectionVisible = false;
        for (const sectionNode of objectNode.querySelectorAll('.section')) {{
          const sectionQueryOk = !q || sectionNode.dataset.search.toLowerCase().includes(q) || objectNode.dataset.search.toLowerCase().includes(q);
          const sectionBucketOk = !groupValue || sectionNode.dataset.groupKey === groupValue;
          let anyCardVisible = false;
          for (const cardNode of sectionNode.querySelectorAll('.card')) {{
            const cardQueryOk = !q || cardNode.dataset.search.toLowerCase().includes(q) || sectionNode.dataset.search.toLowerCase().includes(q);
            const cardRoleOk = !roleValue || cardNode.dataset.role === roleValue;
            const visible = sectionQueryOk && sectionBucketOk && cardQueryOk && cardRoleOk;
            cardNode.classList.toggle('hidden', !visible);
            if (visible) anyCardVisible = true;
          }}
          sectionNode.classList.toggle('hidden', !anyCardVisible);
          if (anyCardVisible) anySectionVisible = true;
        }}
        const objectQueryOk = !q || objectNode.dataset.search.toLowerCase().includes(q);
        objectNode.classList.toggle('hidden', !(anySectionVisible && objectQueryOk));
      }}
    }};

    searchBox.addEventListener('input', applyFilter);
    bucketFilter.addEventListener('change', applyFilter);
    roleFilter.addEventListener('change', applyFilter);
    reloadBtn.addEventListener('click', () => window.location.reload());
    eagerBtn.addEventListener('click', () => {{
      for (const img of document.querySelectorAll('img')) {{
        img.loading = 'eager';
      }}
    }});
  </script>
</body>
</html>
"""


@dataclass(frozen=True)
class SampleCard:
    case_label: str
    role_kind: str
    rgb_video: str
    gif_preview: str
    metadata_path: str
    sample_name: str
    search_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_gif_side", type=int, default=360)
    parser.add_argument("--gif_duration_ms", type=int, default=120)
    parser.add_argument("--workers", type=int, default=max(4, min(16, (os.cpu_count() or 8))))
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8102)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_symlink(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def role_kind_from_meta(meta: dict[str, Any], sample_name: str) -> str:
    counterfactual = dict(meta.get("counterfactual", {}) or {})
    kind = str(counterfactual.get("kind", "") or "")
    if kind in {"same_scene_negative", "no_collision_negative"}:
        return kind
    if "__cf_same_scene_neg" in sample_name:
        return "same_scene_negative"
    if "__cf_no_collision_neg" in sample_name:
        return "no_collision_negative"
    return "factual"


def case_label_from_meta(sample_dir: Path, meta: dict[str, Any]) -> str:
    case_name = str(meta.get("case_name") or "")
    if not case_name:
        if "__" in sample_dir.name:
            case_name = sample_dir.name.split("__", 1)[1]
        else:
            case_name = sample_dir.name
    counterfactual = dict(meta.get("counterfactual", {}) or {})
    if not counterfactual:
        return case_name
    parent_name = str(counterfactual.get("parent_case_name") or case_name)
    kind = str(counterfactual.get("kind") or "counterfactual")
    return f"{parent_name} [{kind}]"


def iter_sample_dirs(dataset_root: Path) -> Iterable[Path]:
    for meta_path in sorted(dataset_root.rglob("metadata.json")):
        sample_dir = meta_path.parent
        rgb_dir = sample_dir / "rgb"
        video_path = sample_dir / "videos" / "rgb.mp4"
        if rgb_dir.is_dir() and video_path.exists():
            yield sample_dir


def build_records(dataset_root: Path) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[SampleCard]]] = defaultdict(lambda: defaultdict(list))
    object_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"factual": 0, "same_scene_negative": 0, "no_collision_negative": 0})

    for sample_dir in iter_sample_dirs(dataset_root):
        metadata_path = sample_dir / "metadata.json"
        rel_dir = sample_dir.relative_to(dataset_root)
        rel_parts = rel_dir.parts
        if len(rel_parts) < 2:
            continue
        object_id = sample_dir.name.split("__", 1)[0]
        group_parts = rel_parts[:-1]
        rel_group_dir = "/".join(group_parts)
        scene_composition = rel_parts[0]
        count_bucket = next((part for part in rel_parts if part.startswith("count_")), "")
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        role_kind = role_kind_from_meta(meta, sample_dir.name)
        case_label = case_label_from_meta(sample_dir, meta)
        card = SampleCard(
            case_label=case_label,
            role_kind=role_kind,
            rgb_video=f"dataset/{rel_dir.as_posix()}/videos/rgb.mp4",
            gif_preview=f"gifs/{rel_dir.as_posix()}/videos/rgb.gif",
            metadata_path=f"dataset/{rel_dir.as_posix()}/metadata.json",
            sample_name=sample_dir.name,
            search_text=" ".join(
                [
                    object_id,
                    rel_group_dir,
                    scene_composition,
                    count_bucket,
                    sample_dir.name,
                    case_label,
                    role_kind,
                ]
            ).lower(),
        )
        grouped[object_id][rel_group_dir].append(card)
        object_stats[object_id][role_kind] += 1

    records: list[dict[str, Any]] = []
    for object_id in sorted(grouped.keys(), key=lambda text: (len(text), text)):
        sections: list[dict[str, Any]] = []
        for rel_group_dir in sorted(grouped[object_id].keys()):
            items = grouped[object_id][rel_group_dir]
            items_sorted = sorted(
                items,
                key=lambda item: (
                    0 if item.role_kind == "factual" else 1 if item.role_kind == "same_scene_negative" else 2,
                    item.case_label,
                    item.sample_name,
                ),
            )
            parts = rel_group_dir.split("/")
            scene_composition = parts[0] if parts else ""
            count_bucket = next((part for part in parts if part.startswith("count_")), "")
            group_key = rel_group_dir
            sections.append(
                {
                    "group_key": group_key,
                    "rel_group_dir": rel_group_dir,
                    "scene_composition": scene_composition,
                    "count_bucket": count_bucket,
                    "item_count": len(items_sorted),
                    "search_text": " ".join(
                        [object_id, rel_group_dir, scene_composition, count_bucket] + [item.search_text for item in items_sorted]
                    ).lower(),
                    "items": [item.__dict__ for item in items_sorted],
                }
            )
        stats = object_stats[object_id]
        sample_count = sum(section["item_count"] for section in sections)
        records.append(
            {
                "anchor": f"object-{object_id}",
                "object_id": object_id,
                "group_count": len(sections),
                "sample_count": sample_count,
                "factual_count": int(stats["factual"]),
                "same_scene_count": int(stats["same_scene_negative"]),
                "no_collision_count": int(stats["no_collision_negative"]),
                "search_text": " ".join(
                    [object_id] + [section["search_text"] for section in sections]
                ).lower(),
                "sections": sections,
            }
        )
    records.sort(key=lambda item: (-int(item["sample_count"]), item["object_id"]))
    return records


def make_rgb_gif(rgb_dir: Path, dst: Path, max_side: int, duration_ms: int) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return False
    frame_paths = sorted(rgb_dir.glob("*.png"))
    if not frame_paths:
        return False
    ensure_dir(dst.parent)
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as frame:
            image = frame.convert("RGB")
            scale = min(max_side / float(image.width), max_side / float(image.height), 1.0)
            size = (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            )
            frames.append(image.resize(size, Image.Resampling.BILINEAR))
    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
    )
    for image in frames:
        image.close()
    return True


def make_rgb_gif_task(task: tuple[str, str, int, int]) -> int:
    rgb_dir_str, dst_str, max_side, duration_ms = task
    created = make_rgb_gif(Path(rgb_dir_str), Path(dst_str), max_side=max_side, duration_ms=duration_ms)
    return 1 if created else 0


def collect_gif_tasks(dataset_root: Path, output_dir: Path, max_side: int, duration_ms: int) -> list[tuple[str, str, int, int]]:
    tasks: list[tuple[str, str, int, int]] = []
    for sample_dir in iter_sample_dirs(dataset_root):
        rel_dir = sample_dir.relative_to(dataset_root)
        dst = output_dir / "gifs" / rel_dir / "videos" / "rgb.gif"
        tasks.append((str(sample_dir / "rgb"), str(dst), int(max_side), int(duration_ms)))
    return tasks


def write_manifest(output_dir: Path, dataset_root: Path, records: list[dict[str, Any]], args: argparse.Namespace, gifs_created: int) -> None:
    manifest = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "object_count": len(records),
        "sample_count": sum(int(item["sample_count"]) for item in records),
        "gif_count_created": int(gifs_created),
        "gif_max_side": int(args.max_gif_side),
        "gif_duration_ms": int(args.gif_duration_ms),
        "workers": int(args.workers),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_html(output_dir: Path, dataset_root: Path, records: list[dict[str, Any]]) -> Path:
    output_path = output_dir / "index.html"
    html_text = HTML_TEMPLATE.format(
        object_count=len(records),
        sample_count=sum(int(item["sample_count"]) for item in records),
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        dataset_root=html.escape(dataset_root.as_posix()),
        records_json=json.dumps(records, ensure_ascii=False),
    )
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


class GalleryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, index_name: str, **kwargs: Any) -> None:
        self.index_name = index_name
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"", "/"}:
            self.path = f"/{self.index_name}"
        super().do_GET()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "gifs")
    reset_symlink(output_dir / "dataset", dataset_root)

    tasks = collect_gif_tasks(
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_side=int(args.max_gif_side),
        duration_ms=int(args.gif_duration_ms),
    )
    gifs_created = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
            for created in executor.map(make_rgb_gif_task, tasks, chunksize=8):
                gifs_created += int(created)

    records = build_records(dataset_root)
    output_path = write_html(output_dir, dataset_root, records)
    write_manifest(output_dir, dataset_root, records, args, gifs_created)

    print(f"[DONE] gallery={output_path}")
    print(f"[INFO] object_count={len(records)} sample_count={sum(int(item['sample_count']) for item in records)}")
    print(f"[INFO] gifs_created={gifs_created} total_tasks={len(tasks)}")

    if not args.serve:
        return
    handler = partial(GalleryHandler, directory=str(output_dir), index_name=output_path.name)
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"[INFO] browse=http://127.0.0.1:{int(args.port)}/{output_path.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopped server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
