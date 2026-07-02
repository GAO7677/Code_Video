#!/usr/bin/env python3
import argparse
import csv
import html
import os
from collections import defaultdict
from pathlib import Path


CSS = """
:root {
  --bg: #f4efe7;
  --card: #fffaf2;
  --ink: #1f1a17;
  --muted: #6f665f;
  --line: #d8cfc5;
  --accent: #a33a2b;
  --accent-2: #1f6b5b;
  --shadow: 0 14px 30px rgba(55, 36, 19, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  background:
    radial-gradient(circle at top right, rgba(163,58,43,0.12), transparent 30%),
    linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
  color: var(--ink);
}
.wrap {
  width: min(1500px, calc(100vw - 48px));
  margin: 0 auto;
  padding: 36px 0 64px;
}
h1 {
  font-size: clamp(34px, 4vw, 58px);
  line-height: 0.95;
  margin: 0 0 10px;
  letter-spacing: -0.04em;
}
.sub {
  color: var(--muted);
  font-size: 18px;
  margin-bottom: 28px;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 30px;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 14px;
  background: rgba(255,255,255,0.55);
  font-size: 14px;
}
.group {
  background: color-mix(in srgb, var(--card) 88%, white);
  border: 1px solid var(--line);
  border-radius: 24px;
  box-shadow: var(--shadow);
  padding: 22px;
  margin-bottom: 22px;
}
.group-head {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: baseline;
  flex-wrap: wrap;
  border-bottom: 1px solid var(--line);
  padding-bottom: 14px;
  margin-bottom: 18px;
}
.group-title {
  font-size: 28px;
  margin: 0;
}
.group-stats {
  color: var(--muted);
  font-size: 15px;
}
.items {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 16px;
}
.item {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: white;
  overflow: hidden;
}
.item.good {
  border-top: 5px solid var(--accent-2);
}
.item.bad {
  border-top: 5px solid var(--accent);
}
.item-mid {
  border-top: 5px solid #b99341;
}
video {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #191919;
}
.item-body {
  padding: 14px;
}
.score {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.03em;
  margin-bottom: 8px;
}
.path {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  font-size: 12px;
  line-height: 1.45;
  color: var(--muted);
  word-break: break-all;
}
.rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 50%;
  background: var(--ink);
  color: white;
  font-size: 14px;
  margin-right: 10px;
  vertical-align: middle;
}
@media (max-width: 720px) {
  .wrap { width: min(100vw - 20px, 1500px); padding-top: 20px; }
  .group { padding: 16px; border-radius: 18px; }
  .group-title { font-size: 22px; }
}
"""


def build_html(rows, output_path, source_csv, top_k, video_url_prefix):
    groups = defaultdict(list)
    for row in rows:
        base = os.path.basename(row["video_path"])
        groups[base].append(row)

    ranked = []
    for base, items in groups.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda r: float(r["surprise_score"]))
        scores = [float(r["surprise_score"]) for r in items]
        ranked.append(
            {
                "basename": base,
                "items": items,
                "gap": max(scores) - min(scores),
                "min_score": min(scores),
                "max_score": max(scores),
            }
        )

    ranked.sort(key=lambda x: (x["gap"], len(x["items"]), x["basename"]), reverse=True)
    ranked = ranked[:top_k]

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>WMReward Gap Visualization</title>",
        f"<style>{CSS}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Same Basename, Biggest Surprise Gaps</h1>",
        "<div class='sub'>Grouped by identical video basename, sorted by max minus min surprise score.</div>",
        "<div class='meta'>",
        f"<div class='pill'>Source CSV: {html.escape(source_csv)}</div>",
        f"<div class='pill'>Groups shown: {len(ranked)}</div>",
        f"<div class='pill'>Rows in CSV: {len(rows)}</div>",
        "</div>",
    ]

    for idx, group in enumerate(ranked, start=1):
        parts.append("<section class='group'>")
        parts.append("<div class='group-head'>")
        parts.append(
            f"<h2 class='group-title'><span class='rank'>{idx}</span>{html.escape(group['basename'])}</h2>"
        )
        parts.append(
            "<div class='group-stats'>"
            f"gap = <strong>{group['gap']:.8f}</strong> | "
            f"min = {group['min_score']:.8f} | "
            f"max = {group['max_score']:.8f} | "
            f"count = {len(group['items'])}"
            "</div>"
        )
        parts.append("</div><div class='items'>")
        for item_idx, row in enumerate(group["items"]):
            cls = "item-mid"
            if item_idx == 0:
                cls = "good"
            elif item_idx == len(group["items"]) - 1:
                cls = "bad"
            rel_url = f"{video_url_prefix.rstrip('/')}/{row['relative_path']}"
            score = float(row["surprise_score"])
            rel = row["relative_path"]
            parts.append(f"<article class='item {cls}'>")
            parts.append(
                f"<video controls preload='metadata' src='{html.escape(rel_url)}'></video>"
            )
            parts.append("<div class='item-body'>")
            parts.append(f"<div class='score'>{score:.8f}</div>")
            parts.append(f"<div class='path'>{html.escape(rel)}</div>")
            parts.append("</div></article>")
        parts.append("</div></section>")

    parts.append("</div></body></html>")
    Path(output_path).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--video_url_prefix", default="dataset")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    build_html(rows, args.output_html, args.csv_path, args.top_k, args.video_url_prefix)
    print(args.output_html)


if __name__ == "__main__":
    main()
