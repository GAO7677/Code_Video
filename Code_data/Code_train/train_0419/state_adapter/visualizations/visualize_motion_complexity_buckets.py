#!/usr/bin/env python3
"""Build a local HTML preview grouped by motion-complexity buckets."""

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

from motion_complexity import MOTION_COMPLEXITY_LABELS, summarize_motion_complexity
from visualize_stage1_subsets import build_sample_report, ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize oracle-state windows grouped by motion complexity.")
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
        default=Path("/data/gaoya/AAA_test_video/portal_hub/motion_complexity_preview"),
    )
    parser.add_argument("--num_windows_per_bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--sample_filter",
        type=str,
        default="",
        help="Optional substring filter on source_sample_dir or window_dir.",
    )
    return parser.parse_args()


def iter_window_records(dataset_root: Path, sample_filter: str) -> List[dict]:
    records: List[dict] = []
    for pair_meta_path in sorted(dataset_root.rglob("pair_meta.json")):
        window_dir = pair_meta_path.parent
        meta = json.loads(pair_meta_path.read_text(encoding="utf-8"))
        if sample_filter:
            haystack = f"{window_dir} {meta.get('source_sample_dir', '')}"
            if sample_filter not in haystack:
                continue
        motion_complexity = meta.get("motion_complexity")
        if not isinstance(motion_complexity, dict) or "label" not in motion_complexity:
            continue
        records.append(
            {
                "window_dir": window_dir,
                "meta": meta,
                "label": str(motion_complexity["label"]),
                "score": float(motion_complexity.get("score", 0.0)),
            }
        )
    return records


def frame_paths_exist(meta: dict) -> bool:
    for key in ("x_frame_paths", "y_frame_paths"):
        paths = meta.get(key, [])
        if not isinstance(paths, list) or not paths:
            return False
        for path in paths:
            if not Path(str(path)).exists():
                return False
    return True


def select_records(records: List[dict], limit: int, seed: int) -> List[dict]:
    valid_records = [record for record in records if frame_paths_exist(record["meta"])]
    if len(valid_records) <= limit:
        return list(valid_records)
    items = list(valid_records)
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


def build_index_html(bucket_cards: Dict[str, List[dict]], bucket_summary: Dict[str, int], portal_rel: str) -> str:
    sections = []
    descriptions = {
        "static": "速度和加速度都很低，future 段大部分时间几乎不动。",
        "simple": "通常只有单物体在运动，轨迹和速度变化都比较简单。",
        "moderate": "运动更明显，可能有多物体参与，但整体仍然比较可预测。",
        "complex": "高速、明显变速、或者多物体同时运动，future 规划最复杂。",
    }
    for bucket in MOTION_COMPLEXITY_LABELS:
        cards = bucket_cards.get(bucket, [])
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
      <span class="badge">{html.escape(bucket)}</span>
      <span class="badge">score {card['score']:.3f}</span>
      <span class="badge">fut {card['future_len']}</span>
      <span class="badge">{card['num_objects']} objects</span>
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
  <div class="section-head">
    <div>
      <h2>{html.escape(bucket)}</h2>
      <p class="muted">{html.escape(descriptions[bucket])}</p>
      <p class="muted">count = {bucket_summary.get(bucket, 0)}</p>
    </div>
  </div>
  <div class="sample-grid">
    {cards_html if cards_html else '<p class="muted">No samples selected for this bucket.</p>'}
  </div>
</section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Motion Complexity Preview</title>
  <style>
    :root {{
      --bg: #ece6dc;
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
    .page {{
      max-width: 1460px;
      margin: 0 auto;
      padding: 28px 22px 60px;
    }}
    .hero, .section {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 42px var(--shadow);
    }}
    .hero {{
      padding: 28px 30px;
      margin-bottom: 22px;
    }}
    .hero h1, .section h2, .sample-body h3 {{ margin: 0; }}
    .hero h1 {{ font-size: 36px; line-height: 1.06; letter-spacing: -0.02em; }}
    .hero p, .muted {{ color: var(--muted); }}
    .section {{
      padding: 22px 24px;
      margin-bottom: 20px;
    }}
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
      <h1>Motion Complexity Preview</h1>
      <p>按 `static / simple / moderate / complex` 分桶展示 oracle-state 训练窗口。每条样本页包含 context/future RGB、future overlay、逐物体 9 维状态曲线，以及该窗口的复杂度分数。</p>
      <p>当前 portal 相对目录: <code>{html.escape(portal_rel)}</code></p>
      <div class="metric-grid">
        {''.join(f'<div class="metric-card"><span class="metric-label">{html.escape(bucket)}</span><span class="metric-value">{bucket_summary.get(bucket, 0)}</span></div>' for bucket in MOTION_COMPLEXITY_LABELS)}
      </div>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"dataset_root does not exist: {args.dataset_root}")

    records = iter_window_records(args.dataset_root, args.sample_filter)
    if not records:
        raise RuntimeError(f"No pair_meta.json records with motion_complexity found under {args.dataset_root}")

    bucket_summary = summarize_motion_complexity(record["label"] for record in records)
    by_bucket: Dict[str, List[dict]] = {bucket: [] for bucket in MOTION_COMPLEXITY_LABELS}
    for record in records:
        by_bucket.setdefault(record["label"], []).append(record)

    ensure_dir(args.output_dir)
    bucket_cards: Dict[str, List[dict]] = {}
    for bucket_idx, bucket in enumerate(MOTION_COMPLEXITY_LABELS):
        selected = select_records(
            by_bucket.get(bucket, []),
            limit=int(args.num_windows_per_bucket),
            seed=int(args.seed) + bucket_idx,
        )
        bucket_dir = args.output_dir / bucket
        ensure_dir(bucket_dir)
        cards: List[dict] = []
        for sample_idx, record in enumerate(selected):
            sample_dir = bucket_dir / f"sample_{sample_idx:02d}"
            report_meta = build_sample_report(
                stage_name=f"motion_complexity/{bucket}",
                item={"out_dir": str(record["window_dir"])},
                dst_dir=sample_dir,
            )
            report_meta["motion_complexity"] = bucket
            report_meta["motion_complexity_score"] = float(record["score"])
            sample_info_path = sample_dir / "sample_info.json"
            sample_info = json.loads(sample_info_path.read_text(encoding="utf-8"))
            sample_info["motion_complexity"] = bucket
            sample_info["motion_complexity_score"] = float(record["score"])
            sample_info_path.write_text(json.dumps(sample_info, ensure_ascii=False, indent=2), encoding="utf-8")
            cards.append(
                {
                    "title": f"{bucket} #{sample_idx + 1}",
                    "summary": (
                        f"{report_meta['count_bucket']} | start={report_meta['start_index']} | "
                        f"future={report_meta['future_len']} | score={record['score']:.3f}"
                    ),
                    "rel_dir": f"{bucket}/sample_{sample_idx:02d}",
                    "score": float(record["score"]),
                    "future_len": int(report_meta["future_len"]),
                    "num_objects": int(report_meta["num_objects"]),
                }
            )
        bucket_cards[bucket] = cards

    (args.output_dir / "index.html").write_text(
        build_index_html(bucket_cards, bucket_summary, args.output_dir.name),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root),
                "sample_filter": args.sample_filter,
                "num_windows_per_bucket": int(args.num_windows_per_bucket),
                "bucket_summary": bucket_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
