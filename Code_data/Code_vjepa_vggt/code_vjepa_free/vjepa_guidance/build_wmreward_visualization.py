#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


CSS = """
:root {
  --bg: #f3efe8;
  --card: #fffdf8;
  --ink: #161412;
  --muted: #6e675f;
  --line: #dbd2c7;
  --green: #1d6e5f;
  --red: #a63d2f;
  --shadow: 0 16px 34px rgba(41, 28, 16, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Georgia, "Times New Roman", serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top right, rgba(29,110,95,0.10), transparent 24%),
    radial-gradient(circle at left top, rgba(166,61,47,0.08), transparent 20%),
    linear-gradient(180deg, #f7f4ed 0%, var(--bg) 100%);
}
.wrap { width: min(1550px, calc(100vw - 40px)); margin: 0 auto; padding: 32px 0 56px; }
h1 { font-size: clamp(34px, 4vw, 58px); line-height: 0.95; margin: 0 0 8px; letter-spacing: -0.04em; }
.sub { color: var(--muted); font-size: 18px; margin-bottom: 24px; }
.meta { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }
.pill { border: 1px solid var(--line); border-radius: 999px; padding: 8px 14px; background: rgba(255,255,255,0.65); font-size: 14px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.card.good { border-top: 6px solid var(--green); }
.card.bad { border-top: 6px solid var(--red); }
.card.err { border-top: 6px solid #8a7f76; }
video { display: block; width: 100%; aspect-ratio: 16 / 9; background: #111; }
.body { padding: 16px; }
.head { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; margin-bottom: 8px; }
.name { font-size: 19px; line-height: 1.15; margin: 0; word-break: break-word; }
.score { font-size: 28px; font-weight: 700; letter-spacing: -0.03em; }
.score-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.path { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; font-size: 12px; color: var(--muted); line-height: 1.45; word-break: break-all; margin-top: 8px; }
.stats { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }
.stat { border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; background: rgba(255,255,255,0.7); font-size: 12px; }
.error { color: var(--red); font-size: 13px; line-height: 1.4; margin-top: 10px; white-space: pre-wrap; word-break: break-word; }
@media (max-width: 720px) {
  .wrap { width: min(100vw - 20px, 1550px); padding-top: 20px; }
}
"""


def classify_row(row: dict[str, str]) -> str:
    if row["status"] != "ok":
        return "err"
    name = row["basename"].lower()
    if "guided" in name or "vjepa" in name:
        return "good"
    if "baseline" in name:
        return "bad"
    return "good"


def build_html(rows: list[dict[str, str]], output_html: Path, source_csv: Path) -> None:
    ok_scores = [float(row["surprise_score"]) for row in rows if row["status"] == "ok" and row["surprise_score"]]
    rows_sorted = sorted(
        rows,
        key=lambda row: (row["status"] != "ok", float(row["surprise_score"] or "inf"), row["basename"]),
    )
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>WMReward Video Scores</title>",
        f"<style>{CSS}</style>",
        "</head><body><div class='wrap'>",
        "<h1>WMReward Scoreboard</h1>",
        "<div class='sub'>Lower surprise means higher past-to-future predictability under the V-JEPA world model.</div>",
        "<div class='meta'>",
        f"<div class='pill'>CSV: {html.escape(str(source_csv))}</div>",
        f"<div class='pill'>Videos: {len(rows)}</div>",
        f"<div class='pill'>OK rows: {sum(row['status'] == 'ok' for row in rows)}</div>",
        (
            f"<div class='pill'>Surprise range: {min(ok_scores):.8f} - {max(ok_scores):.8f}</div>"
            if ok_scores
            else "<div class='pill'>No successful rows</div>"
        ),
        "</div>",
        "<div class='grid'>",
    ]

    for row in rows_sorted:
        rel_video = html.escape("../" + row["relative_path"])
        cls = classify_row(row)
        parts.append(f"<article class='card {cls}'>")
        if row["status"] == "ok":
            parts.append(f"<video controls preload='metadata' src='{rel_video}'></video>")
        else:
            parts.append("<div style='aspect-ratio:16/9;background:#1a1a1a;'></div>")
        parts.append("<div class='body'>")
        parts.append("<div class='head'>")
        parts.append(f"<h2 class='name'>{html.escape(row['basename'])}</h2>")
        if row["status"] == "ok":
            parts.append(
                "<div>"
                "<div class='score-label'>Surprise</div>"
                f"<div class='score'>{float(row['surprise_score']):.8f}</div>"
                "</div>"
            )
        else:
            parts.append("<div><div class='score-label'>Status</div><div class='score'>ERROR</div></div>")
        parts.append("</div>")
        parts.append(f"<div class='path'>{html.escape(row['relative_path'])}</div>")
        if row["status"] == "ok":
            parts.append("<div class='stats'>")
            parts.append(f"<div class='stat'>Similarity {float(row['similarity_score']):.8f}</div>")
            parts.append(f"<div class='stat'>Frames {row['sampled_frames']}</div>")
            parts.append(f"<div class='stat'>Windows {row['num_windows']}</div>")
            parts.append(
                f"<div class='stat'>w/c/s {row['window_size']}/{row['context_frames']}/{row['stride']}</div>"
            )
            parts.append("</div>")
        else:
            parts.append(f"<div class='error'>{html.escape(row['error'])}</div>")
        parts.append("</div></article>")

    parts.append("</div></div></body></html>")
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an HTML visualization for WMReward video scores.")
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--output_html", type=Path, required=True)
    args = parser.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    build_html(rows, args.output_html, args.csv_path)
    print(args.output_html)


if __name__ == "__main__":
    main()
