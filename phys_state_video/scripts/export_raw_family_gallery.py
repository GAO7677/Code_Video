#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a raw-sample gallery grouped by family.")
    parser.add_argument("--data-root", required=True, help="Raw root containing split/F*/sample_xxxxxx.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-family", type=int, default=6)
    return parser.parse_args()


def clean_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def gather_samples(split_root: Path, per_family: int) -> dict[str, list[dict[str, object]]]:
    families: dict[str, list[dict[str, object]]] = defaultdict(list)
    for family_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        family_name = family_dir.name
        for sample_dir in sorted(path for path in family_dir.iterdir() if path.is_dir())[:per_family]:
            video_path = sample_dir / "video.mp4"
            meta_path = sample_dir / "meta.json"
            states_path = sample_dir / "states.npz"
            if not video_path.exists() or not meta_path.exists():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            families[family_name].append(
                {
                    "sample_id": sample_dir.name,
                    "family": family_name,
                    "video_path": video_path,
                    "meta_path": meta_path,
                    "states_path": states_path if states_path.exists() else None,
                    "template_key": clean_text(meta.get("template_key")) or "(missing)",
                    "title": clean_text(meta.get("title")) or sample_dir.name,
                    "description": clean_text(meta.get("description")) or "(empty)",
                    "objects": [obj.get("shape", "?") for obj in meta.get("objects", [])],
                }
            )
    return dict(sorted(families.items()))


def render_html(report: dict[str, object]) -> str:
    sections: list[str] = []
    for family_name, cases in report["families"].items():
        cards: list[str] = []
        for case in cases:
            objects = " / ".join(case["objects"]) if case["objects"] else "(unknown)"
            cards.append(
                f"""
                <article class="case-card">
                  <div class="card-head">
                    <div>
                      <div class="eyebrow">{html.escape(family_name)}</div>
                      <h3>{html.escape(case['sample_id'])}</h3>
                    </div>
                    <div class="chip">{html.escape(case['template_key'])}</div>
                  </div>
                  <p class="title">{html.escape(case['title'])}</p>
                  <p class="desc">{html.escape(case['description'])}</p>
                  <p class="meta">objects: {html.escape(objects)}</p>
                  <video controls preload="metadata" src="{html.escape(case['video_rel'])}"></video>
                  <div class="links">
                    <a href="{html.escape(case['video_rel'])}">video.mp4</a>
                    <a href="{html.escape(case['meta_rel'])}">meta.json</a>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="family-block">
              <div class="family-head">
                <h2>{html.escape(family_name)}</h2>
                <span class="chip">{len(cases)} cases</span>
              </div>
              <div class="card-grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    summary_cards = "".join(
        f'<div class="summary-card"><strong>{html.escape(name)}</strong><span>{len(cases)} cases</span></div>'
        for name, cases in report["families"].items()
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raw Family Gallery</title>
  <style>
    :root {{
      --bg0:#f3efe7;
      --bg1:#ddd0bb;
      --panel:rgba(255,252,247,0.96);
      --line:#d7c8b2;
      --ink:#1f1a16;
      --muted:#6c665c;
      --accent:#8d4a1f;
      --accent2:#0f615a;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"IBM Plex Sans","Source Han Sans SC","Noto Sans SC",sans-serif;
      background:
        radial-gradient(circle at top left, rgba(141,74,31,0.13), transparent 22%),
        radial-gradient(circle at top right, rgba(15,97,90,0.12), transparent 24%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }}
    .page {{ max-width:1680px; margin:0 auto; padding:26px; }}
    .hero,.family-block,.case-card {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:22px;
    }}
    .hero,.family-block {{ padding:20px; margin-bottom:18px; }}
    .eyebrow {{ color:var(--accent); letter-spacing:.08em; text-transform:uppercase; font-size:12px; margin-bottom:6px; }}
    .intro,.desc,.meta {{ color:var(--muted); line-height:1.7; }}
    .summary {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:14px; margin-top:16px; }}
    .summary-card {{ background:rgba(246,240,231,0.92); border:1px solid var(--line); border-radius:16px; padding:14px; }}
    .summary-card strong {{ display:block; font-size:24px; color:var(--accent2); }}
    .summary-card span {{ color:var(--muted); font-size:13px; }}
    .family-head,.card-head {{ display:flex; justify-content:space-between; gap:14px; align-items:center; }}
    .card-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-top:16px; }}
    .case-card {{ padding:16px; }}
    .chip {{ padding:8px 12px; border-radius:999px; border:1px solid var(--line); background:rgba(246,240,231,0.92); color:var(--muted); font-size:13px; }}
    .title {{ font-weight:700; margin:12px 0 8px; }}
    video {{ width:100%; display:block; border-radius:14px; background:#000; margin-top:12px; }}
    .links {{ display:flex; gap:14px; margin-top:10px; flex-wrap:wrap; }}
    .links a {{ color:var(--accent2); text-decoration:none; font-weight:600; }}
    @media (max-width: 1100px) {{
      .summary {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .card-grid {{ grid-template-columns:1fr; }}
    }}
    @media (max-width: 700px) {{
      .summary {{ grid-template-columns:1fr; }}
      .family-head,.card-head {{ flex-direction:column; align-items:flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">{html.escape(str(report['split']).upper())}</div>
      <h1>按 Family 抽样的 Raw Case 可视化</h1>
      <p class="intro">这页直接从 raw 仿真样本里按 `F1-F5` 抽 case，每个卡片展示原始 H.264 视频、模板键和场景描述。这样能保证五大类都能直接看到，不受 episode 预处理进度影响。</p>
      <div class="summary">{summary_cards}</div>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_root = data_root / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    families = gather_samples(split_root, args.per_family)
    report_families: dict[str, list[dict[str, object]]] = {}
    for family_name, cases in families.items():
        out_cases: list[dict[str, object]] = []
        for case in cases:
            out_cases.append(
                {
                    "sample_id": case["sample_id"],
                    "template_key": case["template_key"],
                    "title": case["title"],
                    "description": case["description"],
                    "objects": case["objects"],
                    "video_rel": os.path.relpath(case["video_path"], output_dir),
                    "meta_rel": os.path.relpath(case["meta_path"], output_dir),
                }
            )
        report_families[family_name] = out_cases

    report = {
        "split": args.split,
        "data_root": str(data_root),
        "per_family": args.per_family,
        "families": report_families,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"index": str(output_dir / "index.html"), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
