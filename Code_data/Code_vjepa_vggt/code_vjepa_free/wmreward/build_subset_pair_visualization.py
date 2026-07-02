#!/usr/bin/env python3
import argparse
import csv
import html
import os
from collections import defaultdict
from pathlib import Path


CSS = """
:root {
  --bg: #f3f0ea;
  --card: #fffdf8;
  --ink: #1a1715;
  --muted: #726961;
  --line: #ddd4ca;
  --low: #196b57;
  --high: #a8382a;
  --shadow: 0 16px 34px rgba(44, 29, 16, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  background:
    radial-gradient(circle at 10% 10%, rgba(25,107,87,0.10), transparent 24%),
    radial-gradient(circle at 90% 0%, rgba(168,56,42,0.12), transparent 28%),
    linear-gradient(180deg, #f7f4ee, var(--bg));
}
.wrap { width: min(1500px, calc(100vw - 40px)); margin: 0 auto; padding: 32px 0 60px; }
h1 { font-size: clamp(34px, 4vw, 58px); line-height: .95; margin: 0 0 8px; letter-spacing: -0.04em; }
.sub { color: var(--muted); font-size: 18px; margin-bottom: 24px; }
.meta { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 26px; }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: 8px 14px; background: rgba(255,255,255,0.65); font-size: 14px; }
.pair { background: var(--card); border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); padding: 20px; margin-bottom: 22px; }
.head { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; flex-wrap: wrap; border-bottom: 1px solid var(--line); margin-bottom: 18px; padding-bottom: 12px; }
.title { font-size: 28px; margin: 0; }
.stats { color: var(--muted); font-size: 15px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.card { border: 1px solid var(--line); border-radius: 18px; background: white; overflow: hidden; }
.card.low { border-top: 6px solid var(--low); }
.card.high { border-top: 6px solid var(--high); }
video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #111; }
.body { padding: 14px; }
.role { text-transform: uppercase; letter-spacing: 0.08em; font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.score { font-size: 30px; font-weight: 700; letter-spacing: -0.03em; margin-bottom: 8px; }
.path { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; font-size: 12px; color: var(--muted); line-height: 1.45; word-break: break-all; }
.rank { display: inline-flex; width: 34px; height: 34px; align-items: center; justify-content: center; border-radius: 50%; background: var(--ink); color: #fff; font-size: 14px; margin-right: 10px; vertical-align: middle; }
@media (max-width: 860px) {
  .cols { grid-template-columns: 1fr; }
  .wrap { width: min(100vw - 20px, 1500px); }
  .title { font-size: 22px; }
}
"""


def build_html(rows, output_html, source_csv, video_url_prefix):
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
    ordered_pairs = sorted(by_pair.items(), key=lambda kv: int(kv[0].split("_")[-1]))

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>WMReward Subset16 Pairs</title>",
        f"<style>{CSS}</style>",
        "</head><body><div class='wrap'>",
        "<h1>Subset16 WMReward Pairs</h1>",
        "<div class='sub'>Each row shows the lowest-surprise and highest-surprise sample for the same basename.</div>",
        "<div class='meta'>",
        f"<div class='pill'>Source: {html.escape(source_csv)}</div>",
        f"<div class='pill'>Pairs: {len(ordered_pairs)}</div>",
        f"<div class='pill'>Videos: {len(rows)}</div>",
        "</div>",
    ]

    for idx, (_, items) in enumerate(ordered_pairs, start=1):
        items = sorted(items, key=lambda r: 0 if r["role"] == "low" else 1)
        gap = float(items[-1]["surprise_score"]) - float(items[0]["surprise_score"])
        basename = items[0]["basename"]
        parts.append("<section class='pair'>")
        parts.append("<div class='head'>")
        parts.append(
            f"<h2 class='title'><span class='rank'>{idx}</span>{html.escape(basename)}</h2>"
        )
        parts.append(
            f"<div class='stats'>gap = <strong>{gap:.8f}</strong> | "
            f"low = {float(items[0]['surprise_score']):.8f} | "
            f"high = {float(items[-1]['surprise_score']):.8f}</div>"
        )
        parts.append("</div><div class='cols'>")
        for row in items:
            role = row["role"]
            rel_url = f"{video_url_prefix.rstrip('/')}/{row['relative_path']}"
            parts.append(f"<article class='card {role}'>")
            parts.append(
                f"<video controls preload='metadata' src='{html.escape(rel_url)}'></video>"
            )
            parts.append("<div class='body'>")
            parts.append(f"<div class='role'>{html.escape(role)}</div>")
            parts.append(f"<div class='score'>{float(row['surprise_score']):.8f}</div>")
            parts.append(f"<div class='path'>{html.escape(row['relative_path'])}</div>")
            parts.append("</div></article>")
        parts.append("</div></section>")

    parts.append("</div></body></html>")
    Path(output_html).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--video_url_prefix", default="dataset")
    args = parser.parse_args()

    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    os.makedirs(os.path.dirname(args.output_html), exist_ok=True)
    build_html(rows, args.output_html, args.csv_path, args.video_url_prefix)
    print(args.output_html)


if __name__ == "__main__":
    main()
