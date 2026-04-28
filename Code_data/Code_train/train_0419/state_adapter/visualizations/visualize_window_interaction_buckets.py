#!/usr/bin/env python3
"""Build a local HTML preview grouped by object count and future collision buckets."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ADAPTER_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(STATE_ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(STATE_ADAPTER_ROOT))

from visualize_stage1_subsets import build_sample_report, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize oracle-state windows grouped by interaction buckets.")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/portal_hub/window_interaction_preview"),
    )
    parser.add_argument("--num_windows_per_bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument("--max_buckets", type=int, default=24)
    return parser.parse_args()


def frame_paths_exist(meta: dict) -> bool:
    for key in ("x_frame_paths", "y_frame_paths"):
        paths = meta.get(key, [])
        if not isinstance(paths, list) or not paths:
            return False
        for path in paths:
            if not Path(str(path)).exists():
                return False
    return True


def iter_window_records(dataset_root: Path, sample_filter: str) -> List[dict]:
    records: List[dict] = []
    for pair_meta_path in sorted(dataset_root.rglob("pair_meta.json")):
        meta = json.loads(pair_meta_path.read_text(encoding="utf-8"))
        if sample_filter:
            haystack = f"{pair_meta_path.parent} {meta.get('source_sample_dir', '')}"
            if sample_filter not in haystack:
                continue
        interactions = meta.get("window_interactions")
        if not isinstance(interactions, dict) or "future_bucket" not in interactions:
            continue
        if not frame_paths_exist(meta):
            continue
        records.append(
            {
                "window_dir": pair_meta_path.parent,
                "meta": meta,
                "future_bucket": str(interactions["future_bucket"]),
                "object_count": int(interactions.get("object_count", 0)),
                "collision_type_bucket": str(interactions.get("future_window", {}).get("collision_type_bucket", "")),
                "collision_episode_count": int(interactions.get("future_window", {}).get("collision_episode_count", 0)),
            }
        )
    return records


def bucket_sort_key(bucket_name: str, count: int) -> tuple:
    parts = bucket_name.split("__")
    obj_part = parts[0] if parts else "obj999"
    obj_num = int(obj_part.replace("obj", "")) if obj_part.startswith("obj") else 999
    return (obj_num, -int(count), bucket_name)


def select_records(records: List[dict], limit: int, seed: int) -> List[dict]:
    if len(records) <= limit:
        return list(records)
    items = list(records)
    random.Random(seed).shuffle(items)
    picked: List[dict] = []
    seen_sources = set()
    for item in items:
        source_dir = str(item["meta"].get("source_sample_dir", ""))
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


def build_index_html(bucket_cards: Dict[str, List[dict]], bucket_counts: Dict[str, int], portal_rel: str) -> str:
    sections = []
    for bucket_name, cards in bucket_cards.items():
        cards_html = "".join(
            f"""
<article class="sample-card">
  <div class="sample-thumb">
    <img src="{html.escape(card['rel_dir'])}/future_overlay_strip.png" alt="{html.escape(card['title'])}">
  </div>
  <div class="sample-body">
    <h3>{html.escape(card['title'])}</h3>
    <p>{html.escape(card['summary'])}</p>
    <div class="badge-row">
      <span class="badge">{html.escape(bucket_name)}</span>
      <span class="badge">{card['object_count']} objects</span>
      <span class="badge">{card['collision_episode_count']} future collisions</span>
      <span class="badge">{html.escape(card['collision_type_bucket'])}</span>
    </div>
    <p><a class="button" href="{html.escape(card['rel_dir'])}/index.html">打开样本页</a></p>
  </div>
</article>
"""
            for card in cards
        )
        sections.append(
            f"""
<section class="section">
  <h2>{html.escape(bucket_name)}</h2>
  <p class="muted">count = {bucket_counts.get(bucket_name, 0)}</p>
  <div class="sample-grid">{cards_html}</div>
</section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Window Interaction Preview</title>
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
    .page {{ max-width: 1460px; margin: 0 auto; padding: 28px 22px 60px; }}
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
    .section {{ padding: 22px 24px; margin-bottom: 20px; }}
    .sample-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      gap: 18px;
    }}
    .sample-card {{
      display: grid;
      grid-template-columns: 1.1fr 1fr;
      gap: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.46);
    }}
    .sample-thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      border-radius: 14px;
      background: #0d0f13;
    }}
    .sample-body p {{ line-height: 1.55; }}
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
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Window Interaction Preview</h1>
      <p>按 `object_count + future_collision_count_bucket + future_collision_type_bucket` 组合分桶。这里的碰撞统计只看 future 段，不把 context 里的接触混进来。</p>
      <p>当前 portal 相对目录: <code>{html.escape(portal_rel)}</code></p>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = iter_window_records(args.dataset_root, args.sample_filter)
    if not records:
        raise RuntimeError(f"No records with window_interactions found under {args.dataset_root}")

    bucket_to_records: Dict[str, List[dict]] = {}
    for record in records:
        bucket_to_records.setdefault(record["future_bucket"], []).append(record)
    bucket_counts = {bucket: len(items) for bucket, items in bucket_to_records.items()}
    sorted_buckets = sorted(bucket_counts, key=lambda name: bucket_sort_key(name, bucket_counts[name]))
    if int(args.max_buckets) > 0:
        sorted_buckets = sorted_buckets[: int(args.max_buckets)]

    ensure_dir(args.output_dir)
    bucket_cards: Dict[str, List[dict]] = {}
    for bucket_idx, bucket_name in enumerate(sorted_buckets):
        selected = select_records(
            bucket_to_records[bucket_name],
            limit=int(args.num_windows_per_bucket),
            seed=int(args.seed) + bucket_idx,
        )
        bucket_dir = args.output_dir / bucket_name
        ensure_dir(bucket_dir)
        cards: List[dict] = []
        for sample_idx, record in enumerate(selected):
            sample_dir = bucket_dir / f"sample_{sample_idx:02d}"
            report_meta = build_sample_report(
                stage_name=f"window_interactions/{bucket_name}",
                item={"out_dir": str(record["window_dir"])},
                dst_dir=sample_dir,
            )
            cards.append(
                {
                    "title": f"{bucket_name} #{sample_idx + 1}",
                    "summary": (
                        f"{report_meta['count_bucket']} | start={report_meta['start_index']} | "
                        f"future={report_meta['future_len']}"
                    ),
                    "rel_dir": f"{bucket_name}/sample_{sample_idx:02d}",
                    "object_count": int(record["object_count"]),
                    "collision_episode_count": int(record["collision_episode_count"]),
                    "collision_type_bucket": str(record["collision_type_bucket"]),
                }
            )
        bucket_cards[bucket_name] = cards

    (args.output_dir / "index.html").write_text(
        build_index_html(bucket_cards, bucket_counts, args.output_dir.name),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "sample_filter": args.sample_filter,
                "num_windows_per_bucket": int(args.num_windows_per_bucket),
                "max_buckets": int(args.max_buckets),
                "bucket_counts": {bucket: bucket_counts[bucket] for bucket in sorted_buckets},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
