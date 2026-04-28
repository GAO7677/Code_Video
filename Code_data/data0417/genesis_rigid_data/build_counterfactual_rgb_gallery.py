#!/usr/bin/env python3
"""Build a compact local HTML gallery for samples that already have counterfactual videos."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Counterfactual RGB Gallery</title>
  <style>
    :root {{
      --bg:#f3efe8;
      --panel:#fbf8f2;
      --panel-strong:#fffdf9;
      --ink:#211b16;
      --muted:#756a5c;
      --line:rgba(33,27,22,.10);
      --accent:#8a3e1b;
      --accent-soft:rgba(138,62,27,.10);
      --fact:#d7e6f5;
      --same:#fde0c8;
      --none:#e0f0d8;
      --shadow:0 16px 40px rgba(62,42,24,.10);
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(138,62,27,.10), transparent 28rem),
        linear-gradient(180deg, #faf7f1 0%, #f1ebe1 55%, #e8dfd1 100%);
    }}
    main {{
      width:min(1820px, calc(100vw - 20px));
      margin:0 auto;
      padding:14px 0 36px;
    }}
    .hero, .group {{
      background:rgba(251,248,242,.92);
      border:1px solid var(--line);
      border-radius:18px;
      box-shadow:var(--shadow);
    }}
    .hero {{
      padding:16px 18px 14px;
      position:sticky;
      top:8px;
      z-index:4;
      backdrop-filter:blur(8px);
    }}
    h1 {{
      margin:0;
      font-size:clamp(1.5rem, 2.2vw, 2.5rem);
      line-height:1.05;
    }}
    .sub {{
      margin-top:8px;
      color:var(--muted);
      font-size:.96rem;
      line-height:1.45;
    }}
    .stats {{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      margin-top:10px;
    }}
    .pill {{
      border-radius:999px;
      padding:5px 10px;
      border:1px solid var(--line);
      background:var(--panel-strong);
      font-size:.85rem;
      white-space:nowrap;
    }}
    .toolbar {{
      display:flex;
      gap:8px;
      flex-wrap:wrap;
      margin-top:10px;
      align-items:center;
    }}
    .toolbar input, .toolbar select {{
      border:1px solid var(--line);
      border-radius:10px;
      background:var(--panel-strong);
      padding:8px 10px;
      color:var(--ink);
      min-width:160px;
      font:inherit;
    }}
    .toolbar button {{
      border:1px solid rgba(138,62,27,.18);
      border-radius:10px;
      background:var(--accent-soft);
      color:var(--accent);
      padding:8px 10px;
      font:inherit;
      cursor:pointer;
    }}
    .toolbar button:hover {{
      background:rgba(138,62,27,.16);
    }}
    .index {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
      gap:8px;
      margin-top:12px;
    }}
    .index a {{
      text-decoration:none;
      color:var(--accent);
      border:1px solid var(--line);
      border-radius:10px;
      background:rgba(255,255,255,.62);
      padding:7px 10px;
      font-size:.84rem;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }}
    .groups {{
      display:grid;
      gap:10px;
      margin-top:12px;
    }}
    .group {{
      padding:12px;
    }}
    .group-head {{
      display:flex;
      gap:10px;
      align-items:flex-start;
      justify-content:space-between;
      margin-bottom:10px;
    }}
    .group-title {{
      min-width:0;
    }}
    .group-title h2 {{
      margin:0;
      font-size:1.05rem;
      line-height:1.15;
    }}
    .meta {{
      margin-top:4px;
      color:var(--muted);
      font-size:.84rem;
      line-height:1.35;
      overflow-wrap:anywhere;
    }}
    .tag-row {{
      display:flex;
      gap:6px;
      flex-wrap:wrap;
      justify-content:flex-end;
      max-width:44%;
    }}
    .tag {{
      border-radius:999px;
      padding:4px 8px;
      font-size:.78rem;
      border:1px solid var(--line);
      background:#fff;
      white-space:nowrap;
    }}
    .video-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
      gap:8px;
    }}
    .card {{
      border:1px solid var(--line);
      border-radius:12px;
      background:rgba(255,255,255,.84);
      padding:8px;
      min-width:0;
    }}
    .card.factual {{ background:linear-gradient(180deg, rgba(215,230,245,.60), rgba(255,255,255,.88)); }}
    .card.same_scene_negative {{ background:linear-gradient(180deg, rgba(253,224,200,.72), rgba(255,255,255,.88)); }}
    .card.no_collision_negative {{ background:linear-gradient(180deg, rgba(224,240,216,.78), rgba(255,255,255,.88)); }}
    .card-top {{
      display:flex;
      justify-content:space-between;
      gap:6px;
      align-items:flex-start;
      margin-bottom:6px;
    }}
    .card-name {{
      font-size:.84rem;
      line-height:1.2;
      font-weight:600;
      overflow-wrap:anywhere;
    }}
    .card-role {{
      font-size:.72rem;
      color:var(--muted);
      white-space:nowrap;
      flex:0 0 auto;
    }}
    .card img {{
      width:100%;
      aspect-ratio:4/3;
      display:block;
      background:#111;
      border-radius:10px;
      object-fit:contain;
    }}
    .card-links {{
      display:flex;
      justify-content:space-between;
      gap:8px;
      margin-top:5px;
      font-size:.73rem;
    }}
    .card-links a {{
      color:var(--accent);
      text-decoration:none;
      overflow:hidden;
      text-overflow:ellipsis;
      white-space:nowrap;
    }}
    .hidden {{ display:none !important; }}
    @media (max-width: 900px) {{
      main {{ width:min(100vw - 12px, 1820px); }}
      .hero {{ position:static; }}
      .group-head {{ display:block; }}
      .tag-row {{ margin-top:8px; max-width:none; justify-content:flex-start; }}
      .video-grid {{ grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Counterfactual RGB Gallery</h1>
      <div class="sub">只展示已经出现反事实 case 的样本组。每个分组按 <code>object_id + scene_composition + count_bucket</code> 聚合，并把该组当前已有的所有 RGB 视频紧凑排在一起，便于对照 factual / same-scene negative / no-collision negative。</div>
      <div class="stats">
        <span class="pill">groups: {group_count}</span>
        <span class="pill">videos: {video_count}</span>
        <span class="pill">updated: {updated_at}</span>
        <span class="pill">dataset: {dataset_root}</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 object_id / case / bucket">
        <select id="bucketFilter">
          <option value="">全部 bucket</option>
        </select>
        <button id="reloadBtn" type="button">刷新页面</button>
        <button id="expandBtn" type="button">全部展开播放控件</button>
      </div>
      <div class="index" id="groupIndex"></div>
    </section>
    <section class="groups" id="groups"></section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const groups = JSON.parse(document.getElementById('records').textContent || '[]');
    const groupRoot = document.getElementById('groups');
    const groupIndex = document.getElementById('groupIndex');
    const searchBox = document.getElementById('searchBox');
    const bucketFilter = document.getElementById('bucketFilter');
    const reloadBtn = document.getElementById('reloadBtn');
    const expandBtn = document.getElementById('expandBtn');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    const roleLabel = (item) => {{
      if (item.cf_kind === 'same_scene_negative') return 'cf:same-scene';
      if (item.cf_kind === 'no_collision_negative') return 'cf:no-collision';
      return 'factual';
    }};
    const cardClass = (item) => item.cf_kind || 'factual';
    const bucketValues = [...new Set(groups.map((g) => g.count_bucket).filter(Boolean))].sort();
    bucketFilter.innerHTML += bucketValues.map((v) => `<option value="${{esc(v)}}">${{esc(v)}}</option>`).join('');
    groupIndex.innerHTML = groups.map((g) => `<a href="#${{encodeURIComponent(g.anchor)}}">${{esc(g.object_id)}} · ${{esc(g.count_bucket)}} · ${{g.video_count}}</a>`).join('');
    groupRoot.innerHTML = groups.map((g) => {{
      const tags = [
        `factual ${{g.factual_count}}`,
        `same-scene ${{g.same_scene_count}}`,
        `no-collision ${{g.no_collision_count}}`,
      ].map((v) => `<span class="tag">${{esc(v)}}</span>`).join('');
      const cards = g.items.map((item) => `
        <article class="card ${{cardClass(item)}}" data-search="${{esc(item.search_text)}}">
          <div class="card-top">
            <div class="card-name">${{esc(item.case_label)}}</div>
            <div class="card-role">${{esc(roleLabel(item))}}</div>
          </div>
          <img loading="lazy" src="${{encodeURI(item.gif_preview || item.rgb_video)}}" alt="${{esc(item.case_label)}} gif preview">
          <div class="card-links">
            <a href="${{encodeURI(item.gif_preview || item.rgb_video)}}" target="_blank" rel="noreferrer">gif</a>
            <a href="${{encodeURI(item.rgb_video)}}" target="_blank" rel="noreferrer">video</a>
            <a href="${{encodeURI(item.metadata_path)}}" target="_blank" rel="noreferrer">meta</a>
          </div>
        </article>`).join('');
      return `
        <article
          class="group"
          id="${{encodeURIComponent(g.anchor)}}"
          data-search="${{esc(g.search_text)}}"
          data-bucket="${{esc(g.count_bucket)}}"
        >
          <div class="group-head">
            <div class="group-title">
              <h2>${{esc(g.object_id)}} · ${{esc(g.scene_composition)}} · ${{esc(g.count_bucket)}}</h2>
              <div class="meta">${{esc(g.rel_group_dir)}} · videos=${{g.video_count}}</div>
            </div>
            <div class="tag-row">${{tags}}</div>
          </div>
          <div class="video-grid">${{cards}}</div>
        </article>`;
    }}).join('');

    const applyFilter = () => {{
      const q = searchBox.value.trim().toLowerCase();
      const bucket = bucketFilter.value;
      for (const node of groupRoot.querySelectorAll('.group')) {{
        const okQuery = !q || node.dataset.search.toLowerCase().includes(q);
        const okBucket = !bucket || node.dataset.bucket === bucket;
        node.classList.toggle('hidden', !(okQuery && okBucket));
      }}
    }};

    searchBox.addEventListener('input', applyFilter);
    bucketFilter.addEventListener('change', applyFilter);
    reloadBtn.addEventListener('click', () => window.location.reload());
    expandBtn.addEventListener('click', () => {{
      for (const image of document.querySelectorAll('img')) {{
        image.loading = 'eager';
      }}
    }});
  </script>
</body>
</html>
"""


class GalleryHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, index_name: str, **kwargs: Any) -> None:
        self.index_name = index_name
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"", "/"}:
            self.path = f"/{self.index_name}"
        super().do_GET()


@dataclass
class SampleItem:
    case_label: str
    role_rank: int
    cf_kind: str
    rgb_video: str
    gif_preview: str
    metadata_path: str
    search_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=str, required=True, help="Root like .../train/rigid")
    parser.add_argument("--output_html", type=str, default=None, help="Default: <dataset_root>/counterfactual_rgb_gallery.html")
    parser.add_argument("--serve", action="store_true", help="Serve the generated gallery on a local HTTP port.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args()


def role_rank_and_kind(meta: dict[str, Any]) -> tuple[int, str, str]:
    counterfactual = dict(meta.get("counterfactual", {}) or {})
    kind = str(counterfactual.get("kind", "") or "")
    if kind == "same_scene_negative":
        return 1, kind, "cf_same_scene_neg"
    if kind == "no_collision_negative":
        return 2, kind, "cf_no_collision_neg"
    return 0, "", "factual"


def case_label_from_dir(sample_dir: Path, meta: dict[str, Any]) -> str:
    counterfactual = dict(meta.get("counterfactual", {}) or {})
    case_name = str(meta.get("case_name") or "")
    if not case_name:
        prefix = f"{sample_dir.name.split('__', 1)[0]}__"
        if sample_dir.name.startswith(prefix):
            case_name = sample_dir.name[len(prefix):]
        else:
            case_name = sample_dir.name
    if counterfactual:
        parent = str(counterfactual.get("parent_case_name") or "unknown_parent")
        kind = str(counterfactual.get("kind") or "counterfactual")
        return f"{parent} [{kind}]"
    return case_name


def build_groups(dataset_root: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    cf_groups: set[tuple[str, str, str]] = set()

    for metadata_path in sorted(dataset_root.rglob("metadata.json")):
        sample_dir = metadata_path.parent
        try:
            rel_parts = sample_dir.relative_to(dataset_root).parts
        except ValueError:
            continue
        if len(rel_parts) < 3:
            continue
        scene_composition, count_bucket, sample_name = rel_parts[0], rel_parts[1], rel_parts[2]
        object_id = sample_name.split("__", 1)[0]
        key = (object_id, scene_composition, count_bucket)
        grouped[key].append(sample_dir)
        if "__cf_same_scene_neg" in sample_name or "__cf_no_collision_neg" in sample_name:
            cf_groups.add(key)

    records: list[dict[str, Any]] = []
    for key in sorted(cf_groups):
        object_id, scene_composition, count_bucket = key
        items: list[SampleItem] = []
        factual_count = 0
        same_scene_count = 0
        no_collision_count = 0
        for sample_dir in sorted(set(grouped[key]), key=lambda p: p.name):
            metadata_path = sample_dir / "metadata.json"
            rgb_path = sample_dir / "videos" / "rgb.mp4"
            if not metadata_path.exists() or not rgb_path.exists():
                continue
            meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            role_rank, cf_kind, role_slug = role_rank_and_kind(meta)
            if role_rank == 0:
                factual_count += 1
            elif cf_kind == "same_scene_negative":
                same_scene_count += 1
            elif cf_kind == "no_collision_negative":
                no_collision_count += 1
            rel_dir = sample_dir.relative_to(dataset_root).as_posix()
            case_label = case_label_from_dir(sample_dir, meta)
            items.append(
                SampleItem(
                    case_label=case_label,
                    role_rank=role_rank,
                    cf_kind=cf_kind,
                    rgb_video=f"{rel_dir}/videos/rgb.mp4",
                    gif_preview=f"gifs/{rel_dir}/videos/rgb.gif",
                    metadata_path=f"{rel_dir}/metadata.json",
                    search_text=" ".join(
                        [
                            object_id,
                            scene_composition,
                            count_bucket,
                            sample_dir.name,
                            case_label,
                            role_slug,
                        ]
                    ).lower(),
                )
            )
        if not items:
            continue
        items.sort(key=lambda item: (item.role_rank, item.case_label, item.rgb_video))
        rel_group_dir = f"{scene_composition}/{count_bucket}"
        records.append(
            {
                "anchor": f"{object_id}-{scene_composition}-{count_bucket}",
                "object_id": object_id,
                "scene_composition": scene_composition,
                "count_bucket": count_bucket,
                "rel_group_dir": rel_group_dir,
                "video_count": len(items),
                "factual_count": factual_count,
                "same_scene_count": same_scene_count,
                "no_collision_count": no_collision_count,
                "search_text": " ".join(
                    [object_id, scene_composition, count_bucket] + [item.search_text for item in items]
                ).lower(),
                "items": [item.__dict__ for item in items],
            }
        )
    records.sort(key=lambda g: (-g["same_scene_count"] - g["no_collision_count"], g["object_id"], g["count_bucket"]))
    return records


def write_gallery(dataset_root: Path, output_html: Path, records: list[dict[str, Any]]) -> None:
    video_count = sum(int(group["video_count"]) for group in records)
    updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = HTML_TEMPLATE.format(
        group_count=len(records),
        video_count=video_count,
        updated_at=updated_at,
        dataset_root=dataset_root.as_posix(),
        records_json=json.dumps(records, ensure_ascii=False),
    )
    output_html.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    output_html = Path(args.output_html).resolve() if args.output_html else dataset_root / "counterfactual_rgb_gallery.html"
    records = build_groups(dataset_root)
    write_gallery(dataset_root, output_html, records)
    print(f"[DONE] wrote {output_html}")
    print(f"[INFO] groups={len(records)} videos={sum(int(group['video_count']) for group in records)}")
    if not args.serve:
        return
    handler = partial(GalleryHandler, directory=str(dataset_root), index_name=output_html.name)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[INFO] browse: http://127.0.0.1:{args.port}/{output_html.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopped server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
