from __future__ import annotations

import argparse
import html
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_score_breakdown(item: dict[str, Any]) -> str:
    score_breakdown = item.get("score_breakdown", {})
    if not isinstance(score_breakdown, dict):
        return ""
    cells = []
    for key, value in sorted(score_breakdown.items()):
        cells.append(
            f"<div class='score-cell'><div class='score-key'>{html.escape(key)}</div>"
            f"<div class='score-value'>{html.escape(str(value))}</div></div>"
        )
    return "".join(cells)


def _render_reasons(item: dict[str, Any]) -> str:
    reasons = item.get("reject_reasons", [])
    if not isinstance(reasons, list) or not reasons:
        return "<span class='ok-pill'>keep</span>"
    return "".join(f"<span class='bad-pill'>{html.escape(str(r))}</span>" for r in reasons)


def _render_card(item: dict[str, Any], video_href: str) -> str:
    label = str(item.get("label", "unknown"))
    caption = str(item.get("caption", item.get("captions", "")) or "")
    caption_short = caption if len(caption) <= 260 else caption[:257] + "..."
    return f"""
<article class="card">
  <div class="card-top">
    <div class="tag">{html.escape(label)}</div>
    <div class="tag subtle">{html.escape(str(item.get("video_name", "")))}</div>
    <div class="tag score">score {html.escape(str(item.get("score_total", "")))}</div>
    <div class="tag tier">tier {html.escape(str(item.get("tier", "")))}</div>
  </div>
  <video controls preload="metadata" playsinline src="{html.escape(video_href)}"></video>
  <p class="caption">{html.escape(caption_short)}</p>
  <div class="reason-row">{_render_reasons(item)}</div>
  <div class="score-grid">{_render_score_breakdown(item)}</div>
  <details>
    <summary>Raw record</summary>
    <pre>{html.escape(json.dumps(item, ensure_ascii=False, indent=2))}</pre>
  </details>
</article>
"""


def _pick_examples(records: list[dict[str, Any]], per_label: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[str(item.get("label", ""))].append(item)
    rng = random.Random(seed)
    selected: dict[str, list[dict[str, Any]]] = {}
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        items = sorted(items, key=lambda x: float(x.get("score_total", 0.0)), reverse=True)
        selected[label] = items[: min(per_label, len(items))]
    return selected


def _render_sections(title: str, items_by_label: dict[str, list[dict[str, Any]]], video_dir_name: str) -> str:
    sections = [f"<h1>{html.escape(title)}</h1>"]
    for label in sorted(items_by_label):
        cards = []
        for item in items_by_label[label]:
            video_href = f"{video_dir_name}/{item['video_name']}"
            cards.append(_render_card(item, video_href))
        sections.append(
            f"""
<section class="label-section">
  <div class="label-header">
    <h2>{html.escape(label)}</h2>
    <div class="label-meta">{len(items_by_label[label])} sampled</div>
  </div>
  <div class="card-grid">{''.join(cards)}</div>
</section>
"""
        )
    return "".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review gallery from WISA cleaning manifests.")
    parser.add_argument(
        "--clean-output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/wisa_clean_train0705"),
    )
    parser.add_argument("--keep-per-label", type=int, default=4)
    parser.add_argument("--reject-per-label", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/wisa_clean_train0705_review"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_output_dir = args.clean_output_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_dir = output_dir / "keep_videos"
    reject_dir = output_dir / "reject_videos"
    keep_dir.mkdir(parents=True, exist_ok=True)
    reject_dir.mkdir(parents=True, exist_ok=True)

    keep_manifest = _read_json(clean_output_dir / "keep_manifest.json")
    reject_manifest = _read_json(clean_output_dir / "reject_manifest.json")
    if not isinstance(keep_manifest, list) or not isinstance(reject_manifest, list):
        raise RuntimeError("expected keep/reject manifests to be JSON lists")

    keep_items = _pick_examples(keep_manifest, args.keep_per_label, args.seed)
    reject_items = _pick_examples(reject_manifest, args.reject_per_label, args.seed + 1)

    for items_by_label, dst_dir in ((keep_items, keep_dir), (reject_items, reject_dir)):
        for items in items_by_label.values():
            for item in items:
                source = Path(str(item["local_video_path"]))
                target = dst_dir / source.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(source)

    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>WISA Cleaning Review</title>
  <style>
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background: #f4f1ea;
      color: #1f2933;
    }}
    .shell {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      font-size: 28px;
      margin: 28px 0 14px;
    }}
    .label-section {{
      margin-bottom: 28px;
    }}
    .label-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }}
    .card-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: #fffaf4;
      border: 1px solid #dccfbe;
      border-radius: 18px;
      padding: 14px;
    }}
    .card-top {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .tag {{
      background: rgba(171,59,47,0.12);
      color: #ab3b2f;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
    }}
    .tag.subtle {{
      background: rgba(40,102,110,0.10);
      color: #28666e;
    }}
    .tag.score {{
      background: rgba(0,0,0,0.06);
      color: #111827;
    }}
    .tag.tier {{
      background: rgba(244,162,97,0.16);
      color: #9a3412;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
    }}
    .caption {{
      line-height: 1.5;
      margin: 12px 0;
    }}
    .reason-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
    }}
    .ok-pill, .bad-pill {{
      display: inline-block;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .ok-pill {{
      background: rgba(22,163,74,0.14);
      color: #166534;
    }}
    .bad-pill {{
      background: rgba(185,28,28,0.12);
      color: #991b1b;
    }}
    .score-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 8px;
    }}
    .score-cell {{
      background: #fff;
      border: 1px solid #dccfbe;
      border-radius: 10px;
      padding: 8px;
    }}
    .score-key {{
      font-size: 11px;
      color: #6b7280;
      text-transform: uppercase;
    }}
    .score-value {{
      margin-top: 4px;
      font-weight: 700;
    }}
    details {{
      margin-top: 10px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff;
      border: 1px solid #dccfbe;
      border-radius: 10px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <div class="shell">
    {_render_sections("Kept Samples", keep_items, "keep_videos")}
    {_render_sections("Rejected Samples", reject_items, "reject_videos")}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(body, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index_html": str(output_dir / "index.html"),
                "keep_sampled_total": sum(len(v) for v in keep_items.values()),
                "reject_sampled_total": sum(len(v) for v in reject_items.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
