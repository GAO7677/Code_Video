#!/usr/bin/env python3
"""Build a local portal for simple no-collision / env-only adapter windows."""

from __future__ import annotations

import argparse
import html
import json
import random
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ADAPTER_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(STATE_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(STATE_ADAPTER_ROOT))

from visualize_stage1_subsets import build_sample_report, ensure_dir


DEFAULT_DATASETS = [
    {
        "name": "genesis",
        "title": "Genesis",
        "root": "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain",
    },
    {
        "name": "movi_d",
        "title": "MOVI-D",
        "root": "/data/gaoya/dataset/kubric_tfds_movi-d/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain",
    },
]

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Simple Collision Filter Preview</title>
  <style>
    :root {{
      --bg: #ece5dc;
      --panel: #fffaf3;
      --panel2: #f7efe2;
      --ink: #1b1713;
      --muted: #6a6258;
      --line: #d8cbb9;
      --accent: #7c2d12;
      --accent2: #0f766e;
      --shadow: rgba(43, 30, 20, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(124,45,18,0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(15,118,110,0.10), transparent 24%),
        var(--bg);
    }}
    .page {{ max-width: 1560px; margin: 0 auto; padding: 28px 22px 60px; }}
    .hero, .section {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 42px var(--shadow);
    }}
    .hero {{ padding: 28px 30px; margin-bottom: 22px; }}
    .hero h1, .section h2, .sample-body h3 {{ margin: 0; }}
    .hero h1 {{ font-size: 36px; line-height: 1.06; letter-spacing: -0.02em; }}
    .hero p, .muted {{ color: var(--muted); }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 16px;
    }}
    .metric-card {{
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.5);
    }}
    .metric-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .metric-value {{
      display: block;
      font-weight: 700;
      font-size: 24px;
    }}
    .toolbar {{
      display:flex;
      flex-wrap:wrap;
      gap:10px;
      margin-top:16px;
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
      border:1px solid rgba(124,45,18,.18);
      border-radius:10px;
      background:rgba(124,45,18,.08);
      color:var(--accent);
      padding:8px 10px;
      font:inherit;
      cursor:pointer;
    }}
    .section {{ padding: 22px 24px; margin-bottom: 20px; }}
    .sample-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 18px;
    }}
    .sample-card {{
      display: grid;
      grid-template-columns: minmax(640px, 1.25fr) minmax(320px, 0.75fr);
      gap: 18px;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      background: rgba(255,255,255,0.46);
    }}
    .media-panel {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: rgba(255,255,255,0.58);
    }}
    .media-label {{
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .media-panel img {{
      width: 100%;
      display: block;
      border-radius: 14px;
      background: #0d0f13;
    }}
    .sample-body p {{ line-height: 1.55; }}
    .meta-line {{ margin: 8px 0; word-break: break-word; }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 12px 0 14px;
    }}
    .badge {{
      background: rgba(255,255,255,0.78);
      color: #694d33;
      border: 1px solid #dcc7aa;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
    }}
    .button {{
      display: inline-block;
      text-decoration: none;
      color: white;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 999px;
      padding: 10px 14px;
      font-size: 14px;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .hidden {{ display:none !important; }}
    @media (max-width: 1080px) {{
      .sample-card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Simple No-Collision / Env-Only Window Preview</h1>
      <p>按训练 window 级标签筛选，而不是 raw sample 级标签。当前筛选规则是 <code>future_collision_type_bucket in {{none, env_only}}</code> 且 <code>motion_complexity in {{static, simple}}</code>。</p>
      <div class="metric-grid">
        <div class="metric-card"><span class="metric-label">Genesis matched</span><span class="metric-value">{genesis_count}</span></div>
        <div class="metric-card"><span class="metric-label">MOVI-D matched</span><span class="metric-value">{movi_count}</span></div>
        <div class="metric-card"><span class="metric-label">Total matched</span><span class="metric-value">{total_count}</span></div>
        <div class="metric-card"><span class="metric-label">Generated</span><span class="metric-value" style="font-size:16px">{generated_at}</span></div>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 dataset / bucket / source">
        <select id="datasetFilter">
          <option value="">全部数据源</option>
          <option value="genesis">Genesis</option>
          <option value="movi_d">MOVI-D</option>
        </select>
        <select id="collisionFilter">
          <option value="">全部 future collision</option>
          <option value="none">none</option>
          <option value="env_only">env_only</option>
        </select>
        <select id="motionFilter">
          <option value="">全部 complexity</option>
          <option value="static">static</option>
          <option value="simple">simple</option>
        </select>
        <button id="reloadBtn" type="button">刷新页面</button>
      </div>
    </section>
    {sections_html}
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const datasetFilter = document.getElementById('datasetFilter');
    const collisionFilter = document.getElementById('collisionFilter');
    const motionFilter = document.getElementById('motionFilter');
    const reloadBtn = document.getElementById('reloadBtn');
    const applyFilter = () => {{
      const q = searchBox.value.trim().toLowerCase();
      const ds = datasetFilter.value;
      const col = collisionFilter.value;
      const mot = motionFilter.value;
      for (const section of document.querySelectorAll('.section')) {{
        let anyVisible = false;
        for (const card of section.querySelectorAll('.sample-card')) {{
          const okQ = !q || card.dataset.search.includes(q);
          const okDs = !ds || card.dataset.dataset === ds;
          const okCol = !col || card.dataset.collision === col;
          const okMot = !mot || card.dataset.motion === mot;
          const visible = okQ && okDs && okCol && okMot;
          card.classList.toggle('hidden', !visible);
          if (visible) anyVisible = true;
        }}
        section.classList.toggle('hidden', !anyVisible);
      }}
    }};
    searchBox.addEventListener('input', applyFilter);
    datasetFilter.addEventListener('change', applyFilter);
    collisionFilter.addEventListener('change', applyFilter);
    motionFilter.addEventListener('change', applyFilter);
    reloadBtn.addEventListener('click', () => window.location.reload());
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets_json",
        type=str,
        default=json.dumps(DEFAULT_DATASETS, ensure_ascii=False),
        help="JSON list of {name,title,root}.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/gaoya/portal_hub_sim/simple_collision_filter_preview"),
    )
    parser.add_argument("--num_windows_per_bucket", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8115)
    return parser.parse_args()


def load_records(dataset_name: str, dataset_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair_meta_path in sorted(dataset_root.rglob("pair_meta.json")):
        meta = json.loads(pair_meta_path.read_text(encoding="utf-8"))
        wi = meta.get("window_interactions") or {}
        fw = wi.get("future_window") or {}
        mc = meta.get("motion_complexity") or {}
        future_collision = str(fw.get("collision_type_bucket", ""))
        motion_label = str(mc.get("label", ""))
        if future_collision not in {"none", "env_only"}:
            continue
        if motion_label not in {"static", "simple"}:
            continue
        frame_paths = list(meta.get("x_frame_paths", [])) + list(meta.get("y_frame_paths", []))
        if not frame_paths or any(not Path(str(path)).exists() for path in frame_paths):
            continue
        source_sample_dir = str(meta.get("source_sample_dir", ""))
        records.append(
            {
                "dataset": dataset_name,
                "window_dir": str(pair_meta_path.parent),
                "out_dir": str(pair_meta_path.parent),
                "source_sample_dir": source_sample_dir,
                "future_collision_type_bucket": future_collision,
                "future_bucket": str(wi.get("future_bucket", "")),
                "motion_complexity": motion_label,
                "object_count": int(wi.get("object_count", 0)),
            }
        )
    return records


def select_records(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if len(records) <= limit:
        return list(records)
    items = list(records)
    random.Random(seed).shuffle(items)
    picked: list[dict[str, Any]] = []
    seen_sources = set()
    for item in items:
        source_dir = str(item.get("source_sample_dir", ""))
        if source_dir in seen_sources:
            continue
        picked.append(item)
        seen_sources.add(source_dir)
        if len(picked) >= limit:
            return picked
    for item in items:
        if item in picked:
            continue
        picked.append(item)
        if len(picked) >= limit:
            break
    return picked


def start_server(output_dir: Path, host: str, port: int) -> tuple[int, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    process = subprocess.Popen(
        [
            "python3",
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


def build_section_html(title: str, cards: list[dict[str, Any]]) -> str:
    cards_html = "".join(
        f"""
<article class="sample-card"
  data-dataset="{html.escape(card['dataset'])}"
  data-collision="{html.escape(card['future_collision_type_bucket'])}"
  data-motion="{html.escape(card['motion_complexity'])}"
  data-search="{html.escape(card['search_text'])}">
  <div class="media-panel">
    <div class="media-label">Full RGB GIF</div>
    <img loading="lazy" src="{html.escape(card['rel_dir'])}/source_rgb_full.gif" alt="{html.escape(card['title'])}">
  </div>
  <div class="sample-body">
    <h3>{html.escape(card['title'])}</h3>
    <p>{html.escape(card['summary'])}</p>
    <div class="badge-row">
      <span class="badge">{html.escape(card['dataset'])}</span>
      <span class="badge">{html.escape(card['future_collision_type_bucket'])}</span>
      <span class="badge">{html.escape(card['motion_complexity'])}</span>
      <span class="badge">{card['num_objects']} objects</span>
      <span class="badge">fut {card['future_len']}</span>
    </div>
    <p class="meta-line"><strong>source</strong>: <code>{html.escape(card['source_tag'])}</code></p>
    <p class="meta-line"><strong>window</strong>: start={card['start_index']}, context={card['context_len']}, future={card['future_len']}</p>
    <p><a class="button" href="{html.escape(card['rel_dir'])}/index.html">详情页</a></p>
  </div>
</article>
"""
        for card in cards
    )
    return f"""
<section class="section">
  <h2>{html.escape(title)}</h2>
  <div class="sample-grid">{cards_html}</div>
</section>
"""


def main() -> None:
    args = parse_args()
    datasets = json.loads(args.datasets_json)
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)

    all_records: list[dict[str, Any]] = []
    dataset_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    for ds in datasets:
        dataset_name = str(ds["name"])
        dataset_root = Path(str(ds["root"])).resolve()
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        records = load_records(dataset_name, dataset_root)
        all_records.extend(records)
        dataset_counts[dataset_name] += len(records)
        for item in records:
            bucket_key = f"{dataset_name}__{item['future_collision_type_bucket']}__{item['motion_complexity']}"
            bucket_counts[bucket_key] += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_records:
        key = f"{item['dataset']}__{item['future_collision_type_bucket']}__{item['motion_complexity']}"
        grouped[key].append(item)

    sections_html_parts: list[str] = []
    manifest_records: list[dict[str, Any]] = []
    for bucket_index, bucket_key in enumerate(sorted(grouped.keys())):
        selected = select_records(grouped[bucket_key], int(args.num_windows_per_bucket), int(args.seed) + bucket_index)
        cards: list[dict[str, Any]] = []
        for sample_idx, item in enumerate(selected):
            sample_dir = output_dir / bucket_key / f"sample_{sample_idx:02d}"
            report_meta = build_sample_report(bucket_key, item, sample_dir)
            card = {
                "title": f"{bucket_key} #{sample_idx + 1}",
                "summary": (
                    f"{report_meta['count_bucket']} | {item['future_collision_type_bucket']} | "
                    f"{item['motion_complexity']} | vis={report_meta['future_main_visibility_ratio']:.3f}"
                ),
                "rel_dir": f"{bucket_key}/sample_{sample_idx:02d}",
                "dataset": item["dataset"],
                "future_collision_type_bucket": item["future_collision_type_bucket"],
                "motion_complexity": item["motion_complexity"],
                "num_objects": report_meta["num_objects"],
                "future_len": report_meta["future_len"],
                "context_len": report_meta["context_len"],
                "start_index": report_meta["start_index"],
                "source_tag": report_meta["source_sample_dir"],
                "search_text": " ".join(
                    [
                        item["dataset"],
                        item["future_collision_type_bucket"],
                        item["motion_complexity"],
                        report_meta["source_sample_dir"],
                        report_meta["count_bucket"],
                    ]
                ).lower(),
            }
            cards.append(card)
            manifest_records.append(
                {
                    "bucket_key": bucket_key,
                    "sample_dir": str(sample_dir),
                    "card": card,
                }
            )
        sections_html_parts.append(build_section_html(bucket_key, cards))

    html_text = HTML_TEMPLATE.format(
        genesis_count=dataset_counts.get("genesis", 0),
        movi_count=dataset_counts.get("movi_d", 0),
        total_count=sum(dataset_counts.values()),
        generated_at=subprocess.check_output(["date", "-u", "+%Y-%m-%d %H:%M UTC"], text=True).strip(),
        sections_html="".join(sections_html_parts),
    )
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    manifest = {
        "datasets": datasets,
        "dataset_counts": dict(dataset_counts),
        "bucket_counts": dict(bucket_counts),
        "records": manifest_records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pid, url = start_server(output_dir, str(args.host), int(args.port))
    print(f"output_dir={output_dir}")
    print(f"pid={pid}")
    print(f"url={url}")


if __name__ == "__main__":
    main()
