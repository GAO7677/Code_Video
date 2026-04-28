#!/usr/bin/env python3
"""Build a compact GIF portal grouped by collision complexity."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from build_recent_sample_motion_bucket_portal import (
    _fmt_mtime,
    _group_counts,
    copy_or_symlink,
    ensure_dir,
    iter_recent_samples,
    make_rgb_gif,
)
from sample_bucket_labels import COLLISION_PROFILE_LABELS, COLLISION_PROFILE_ORDER


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/collision_complexity_bucket_preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact GIF preview page grouped by collision complexity.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modified_within_hours", type=float, default=0.0)
    parser.add_argument("--max_recent_samples", type=int, default=0)
    parser.add_argument("--samples_per_bucket", type=int, default=24)
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument("--max_gif_side", type=int, default=320)
    parser.add_argument("--gif_duration_ms", type=int, default=120)
    return parser.parse_args()


def _select_cards(records: List[Dict[str, Any]], samples_per_bucket: int) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["collision_profile_bucket"])].append(record)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    for collision_profile in COLLISION_PROFILE_ORDER:
        items = grouped.get(collision_profile, [])
        items.sort(
            key=lambda item: (
                -int(item["num_objects"]),
                -int(item["obj_obj_event_count"]),
                -int(item["obj_env_event_count"]),
                -float(item["motion_score"]),
                -float(item["mtime"]),
            )
        )
        selected[collision_profile] = items[: int(samples_per_bucket)] if int(samples_per_bucket) > 0 else items
    return selected


def build_html(
    records: List[Dict[str, Any]],
    selected_cards: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> str:
    collision_counts = _group_counts(records, "collision_profile_bucket")
    count_bucket_counts = _group_counts(records, "count_bucket")
    role_counts = _group_counts(records, "sample_role")

    summary_chips = "".join(
        f'<span class="chip"><strong>{html.escape(COLLISION_PROFILE_LABELS.get(label, label))}</strong> {collision_counts.get(label, 0)}</span>'
        for label in COLLISION_PROFILE_ORDER
    )
    filter_options_count = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)} ({count_bucket_counts[label]})</option>'
        for label in count_bucket_counts
        if label
    )
    filter_options_collision = "".join(
        f'<option value="{html.escape(label)}">{html.escape(COLLISION_PROFILE_LABELS.get(label, label))} ({collision_counts.get(label, 0)})</option>'
        for label in COLLISION_PROFILE_ORDER
    )
    filter_options_role = "".join(
        f'<option value="{html.escape(label)}">{html.escape(label)} ({role_counts[label]})</option>'
        for label in role_counts
        if label
    )

    sections: List[str] = []
    for collision_profile in COLLISION_PROFILE_ORDER:
        cards = selected_cards.get(collision_profile, [])
        cards_html: List[str] = []
        for idx, record in enumerate(cards):
            rel_dir = f"{collision_profile}/sample_{idx:02d}"
            cards_html.append(
                f"""
<article class="card"
  data-count="{html.escape(str(record['count_bucket']))}"
  data-collision="{html.escape(str(record['collision_profile_bucket']))}"
  data-role="{html.escape(str(record['sample_role']))}">
  <a class="thumb-link" href="{html.escape(rel_dir)}/rgb.mp4">
    <img loading="lazy" src="{html.escape(rel_dir)}/preview.gif" alt="{html.escape(record['sample_dir'].name)}">
  </a>
  <div class="card-body">
    <div class="title-row">
      <h3>{html.escape(record['sample_dir'].name)}</h3>
      <span class="score">{record['motion_score']:.2f}</span>
    </div>
    <div class="mini-row">{html.escape(record['count_bucket'])} | {record['num_objects']} obj | motion {html.escape(record['motion_label'])}</div>
    <div class="badge-row">
      <span class="badge collision">{html.escape(COLLISION_PROFILE_LABELS.get(str(record['collision_profile_bucket']), str(record['collision_profile_bucket'])))}</span>
      <span class="badge">{html.escape(record['collision_count_bucket'])}</span>
      <span class="badge">{html.escape(record['sample_role'])}</span>
    </div>
    <div class="mini-row">obj-obj {record['obj_obj_event_count']} | obj-env {record['obj_env_event_count']}</div>
    <div class="mini-row muted">{html.escape(record['scene_composition'])}</div>
    <div class="mini-row muted">{html.escape(_fmt_mtime(float(record['mtime'])))}</div>
    <div class="link-row">
      <a href="{html.escape(rel_dir)}/rgb.mp4">mp4</a>
      <a href="{html.escape(rel_dir)}/metadata.json">meta</a>
      <a href="{html.escape(rel_dir)}/summary.json">summary</a>
    </div>
  </div>
</article>
"""
            )

        sections.append(
            f"""
<section class="bucket">
  <div class="bucket-head">
    <h2>{html.escape(COLLISION_PROFILE_LABELS.get(collision_profile, collision_profile))}</h2>
    <span class="bucket-count">dataset count {collision_counts.get(collision_profile, 0)} | showing {len(cards)}</span>
  </div>
  <div class="grid">
    {''.join(cards_html) if cards_html else '<p class="empty">No samples in this collision bucket.</p>'}
  </div>
</section>
"""
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Collision Complexity Bucket Preview</title>
  <style>
    :root {{
      --bg: #ece7df;
      --panel: #fffaf3;
      --ink: #191816;
      --muted: #676056;
      --line: #d8ccb8;
      --shadow: rgba(39, 28, 15, 0.08);
      --accent: #8f3a21;
      --accent2: #1a6c67;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(143,58,33,0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(26,108,103,0.10), transparent 24%),
        var(--bg);
    }}
    .page {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 20px 16px 40px;
    }}
    .hero, .bucket {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,250,243,0.98), rgba(247,239,228,0.96));
      box-shadow: 0 16px 40px var(--shadow);
    }}
    .hero {{
      padding: 18px 20px;
      margin-bottom: 14px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: 30px;
      line-height: 1.08;
      margin-bottom: 8px;
    }}
    .hero p {{
      margin: 6px 0;
      color: var(--muted);
      line-height: 1.45;
    }}
    .chip-row, .control-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chip {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.72);
      font-size: 12px;
    }}
    .control {{
      display: grid;
      gap: 4px;
      min-width: 160px;
    }}
    .control label {{
      font-size: 12px;
      color: var(--muted);
    }}
    select {{
      height: 34px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 0 10px;
    }}
    .bucket {{
      margin-bottom: 12px;
      padding: 12px;
    }}
    .bucket-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }}
    .bucket-count {{
      color: var(--muted);
      font-size: 12px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 10px;
    }}
    .card {{
      display: grid;
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 8px;
      background: rgba(255,255,255,0.66);
    }}
    .thumb-link {{
      display: block;
      text-decoration: none;
    }}
    .thumb-link img {{
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      border-radius: 10px;
      background: #101216;
      display: block;
    }}
    .card-body {{
      display: grid;
      gap: 5px;
    }}
    .title-row {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: start;
    }}
    .title-row h3 {{
      font-size: 13px;
      line-height: 1.22;
      word-break: break-word;
    }}
    .score {{
      font-size: 12px;
      color: #fff;
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      border-radius: 999px;
      padding: 3px 8px;
    }}
    .mini-row {{
      font-size: 11px;
      line-height: 1.35;
      word-break: break-word;
    }}
    .muted {{
      color: var(--muted);
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      background: rgba(255,255,255,0.84);
      font-size: 11px;
    }}
    .badge.collision {{
      border-color: #e2b7a6;
      color: #8f3a21;
    }}
    .link-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      font-size: 11px;
    }}
    .link-row a {{
      color: var(--accent2);
      text-decoration: none;
    }}
    .link-row a:hover {{
      text-decoration: underline;
    }}
    .empty {{
      color: var(--muted);
      margin: 6px 0;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Collision Complexity Bucket Preview</h1>
      <p>主桶按碰撞复杂度分组：`no_collision / env_only / obj_obj_only_c1 / obj_obj_only_c2plus / mixed_c1 / mixed_c2plus`。页面覆盖全部碰撞桶类型，并保留 `count_bucket` 与 `sample_role` 过滤器。</p>
      <p>当前输出目录: <code>{html.escape(output_dir.name)}</code></p>
      <div class="chip-row">{summary_chips}</div>
      <div class="control-row">
        <div class="control">
          <label for="countFilter">count bucket</label>
          <select id="countFilter">
            <option value="">all</option>
            {filter_options_count}
          </select>
        </div>
        <div class="control">
          <label for="collisionFilter">collision</label>
          <select id="collisionFilter">
            <option value="">all</option>
            {filter_options_collision}
          </select>
        </div>
        <div class="control">
          <label for="roleFilter">sample role</label>
          <select id="roleFilter">
            <option value="">all</option>
            {filter_options_role}
          </select>
        </div>
      </div>
    </section>
    {''.join(sections)}
  </div>
  <script>
    const filters = {{
      count: document.getElementById('countFilter'),
      collision: document.getElementById('collisionFilter'),
      role: document.getElementById('roleFilter'),
    }};
    const cards = Array.from(document.querySelectorAll('.card'));
    function applyFilters() {{
      const count = filters.count.value;
      const collision = filters.collision.value;
      const role = filters.role.value;
      cards.forEach((card) => {{
        const ok = (!count || card.dataset.count === count)
          && (!collision || card.dataset.collision === collision)
          && (!role || card.dataset.role === role);
        card.style.display = ok ? '' : 'none';
      }});
    }}
    Object.values(filters).forEach((node) => node.addEventListener('change', applyFilters));
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    records = iter_recent_samples(args)
    if not records:
        raise RuntimeError(f"No samples found under {args.dataset_root}.")

    selected_cards = _select_cards(records, samples_per_bucket=int(args.samples_per_bucket))

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    ensure_dir(args.output_dir)

    summary_payload: Dict[str, Any] = {
        "dataset_root": str(args.dataset_root),
        "modified_within_hours": float(args.modified_within_hours),
        "max_recent_samples": int(args.max_recent_samples),
        "samples_per_bucket": int(args.samples_per_bucket),
        "sample_count": int(len(records)),
        "collision_profile_counts": _group_counts(records, "collision_profile_bucket"),
        "count_bucket_counts": _group_counts(records, "count_bucket"),
        "sample_role_counts": _group_counts(records, "sample_role"),
        "selected_samples": {},
    }

    for collision_profile in COLLISION_PROFILE_ORDER:
        bucket_dir = args.output_dir / collision_profile
        ensure_dir(bucket_dir)
        chosen = selected_cards.get(collision_profile, [])
        selected_meta: List[Dict[str, Any]] = []
        for idx, record in enumerate(chosen):
            dst_dir = bucket_dir / f"sample_{idx:02d}"
            ensure_dir(dst_dir)
            make_rgb_gif(
                rgb_dir=record["rgb_dir"],
                dst=dst_dir / "preview.gif",
                max_side=int(args.max_gif_side),
                duration_ms=int(args.gif_duration_ms),
            )
            copy_or_symlink(record["video_path"], dst_dir / "rgb.mp4")
            (dst_dir / "metadata.json").write_text(
                json.dumps(record["metadata"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            sample_summary = {
                "sample_dir": str(record["sample_dir"]),
                "count_bucket": str(record["count_bucket"]),
                "num_objects": int(record["num_objects"]),
                "scene_composition": str(record["scene_composition"]),
                "sample_role": str(record["sample_role"]),
                "motion_category": str(record["motion_category"]),
                "motion_label": str(record["motion_label"]),
                "motion_score": float(record["motion_score"]),
                "motion_metrics": dict(record["motion_metrics"]),
                "collision_profile_bucket": str(record["collision_profile_bucket"]),
                "collision_type_bucket": str(record["collision_type_bucket"]),
                "collision_count_bucket": str(record["collision_count_bucket"]),
                "obj_obj_event_count": int(record["obj_obj_event_count"]),
                "obj_env_event_count": int(record["obj_env_event_count"]),
                "mtime": float(record["mtime"]),
            }
            (dst_dir / "summary.json").write_text(
                json.dumps(sample_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            selected_meta.append(sample_summary)
        summary_payload["selected_samples"][collision_profile] = selected_meta

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(build_html(records, selected_cards, args.output_dir), encoding="utf-8")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
