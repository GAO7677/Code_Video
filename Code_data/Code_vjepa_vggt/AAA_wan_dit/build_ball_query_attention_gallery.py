#!/usr/bin/env python3
"""Build a cross-model gallery for exact ball-query attention maps."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


MODELS = (
    ("wan_lora", "Wan+LoRA"),
    ("xssc", "Wan+xSSC"),
    ("physrvg", "PhysRVG"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries = []
    for model, label in MODELS:
        case_dir = args.root / model / args.case
        summary_path = case_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entries.append((model, label, case_dir, summary))

    args.output.mkdir(parents=True, exist_ok=True)
    sections = []
    for model, label, case_dir, summary in entries:
        relative = case_dir.relative_to(args.output.parent).as_posix()
        contacts = "".join(
            "<figure>"
            f"<a href='../{html.escape(relative + '/' + step['directory'] + '/' + step['contact_sheet'])}'>"
            f"<img loading='lazy' src='../{html.escape(relative + '/' + step['directory'] + '/' + step['contact_sheet'])}'></a>"
            f"<figcaption>Denoise step {int(step['step_number_one_based'])}</figcaption>"
            "</figure>"
            for step in summary["steps"]
        )
        preview = html.escape(relative + "/" + str(summary["query_preview"]))
        details = html.escape(relative + "/index.html")
        sections.append(
            f"<section class='{model}'>"
            f"<div class='model-head'><h2>{label}</h2>"
            f"<a class='details' href='../{details}'>Open 24-head details</a></div>"
            f"<img class='preview' src='../{preview}'>"
            f"<div class='contacts'>{contacts}</div>"
            "</section>"
        )

    manifest = {
        "case": args.case,
        "models": [model for model, _ in MODELS],
        "steps": [5, 15, 25, 35],
        "heads": list(range(24)),
        "query": {
            "video_frame": 8,
            "latent_time": 2,
            "coords": [[2, 6, 13], [2, 6, 14], [2, 7, 13], [2, 7, 14]],
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block {entries[0][3]['block_id']} ball-query self-attention</title>
<style>
:root{{--bg:#edf0ee;--surface:#fff;--text:#19211d;--line:#cad2cd;--green:#236d4b;--blue:#2b6198;--orange:#a65325}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:3;padding:13px 18px;color:#fff;background:#21332a;border-bottom:1px solid #456052}}
header h1{{margin:0;font-size:19px}} header p{{margin:4px 0 0;color:#c1d0c8}}
main{{max-width:2300px;margin:auto;padding:18px}} section{{margin-top:24px;padding-top:12px;border-top:3px solid var(--blue)}}
section.xssc{{border-color:var(--green)}} section.physrvg{{border-color:var(--orange)}}
.model-head{{display:flex;align-items:center;justify-content:space-between;gap:12px}} h2{{margin:0 0 10px;font-size:18px}}
.details{{padding:8px 11px;color:#fff;background:#2d5541;border-radius:4px;text-decoration:none}}
.preview{{display:block;width:min(100%,1200px);height:auto;border:1px solid var(--line);background:#fff}}
.contacts{{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:10px;margin-top:12px}}
figure{{margin:0;padding:7px;background:var(--surface);border:1px solid var(--line);border-radius:6px}}
figure img{{display:block;width:100%;height:auto}} figcaption{{padding-top:5px;color:#5f6c65}}
@media(max-width:850px){{.contacts{{grid-template-columns:1fr}} .model-head{{align-items:flex-start;flex-direction:column}}}}
</style></head><body>
<header><h1>Block {entries[0][3]['block_id']} ball-query self-attention</h1>
<p>{html.escape(args.case)} | video frame 8 -> latent t=2 | 4 ball patches | 24 heads</p></header>
<main>{''.join(sections)}</main></body></html>"""
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(f"gallery={args.output / 'index.html'}")


if __name__ == "__main__":
    main()
