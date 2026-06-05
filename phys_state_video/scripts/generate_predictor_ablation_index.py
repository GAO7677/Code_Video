#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an index page for predictor ablation comparison reports.")
    parser.add_argument("--output-dir", required=True, help="Directory to write index.html.")
    parser.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="One or more report.json files from predictor overlay comparison exports.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = path
    return payload


def render_html(reports: list[dict]) -> str:
    cards = []
    for report in reports:
        rel_link = report["_path"].parent.relative_to(report["_path"].parent.parent)
        summaries = report.get("model_summaries", [])
        summary_rows = []
        for item in summaries:
            summary_rows.append(
                f"""
                <tr>
                  <td>{html.escape(str(item.get("label", "")))}</td>
                  <td>{float(item.get("center_error_mean") or 0.0):.4f}</td>
                  <td>{float(item.get("boundary_center_delta_error_mean") or 0.0):.4f}</td>
                  <td>{float(item.get("boundary_motion_delta_error_mean") or 0.0):.4f}</td>
                  <td>{float(item.get("boundary_log_scale_delta_error_mean") or 0.0):.4f}</td>
                </tr>
                """
            )
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(report.get("mode", "comparison"))}</div>
              <h2>{html.escape(report.get("label_a", "A"))} vs {html.escape(report.get("label_b", "B"))}</h2>
              <p class="meta">A: {html.escape(report.get("predictor_a_name", ""))}</p>
              <p class="meta">B: {html.escape(report.get("predictor_b_name", ""))}</p>
              <p class="meta">cases={int(report.get("case_count", 0))} | <a href="{html.escape(str(rel_link / 'index.html'))}">打开对比页</a></p>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Center</th>
                    <th>Boundary Center</th>
                    <th>Boundary Motion</th>
                    <th>Boundary Scale</th>
                  </tr>
                </thead>
                <tbody>
                  {''.join(summary_rows)}
                </tbody>
              </table>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>predictor ablation index</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 246, 0.96);
      --line: #dfd3c4;
      --ink: #1f1f1b;
      --muted: #6f675d;
      --accent: #0f5a52;
      --accent2: #b8642a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(184, 100, 42, 0.12), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 90, 82, 0.12), transparent 22%),
        linear-gradient(180deg, #f7f3ea 0%, #efe5d8 100%);
    }}
    .page {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 18px;
    }}
    .hero p {{
      color: var(--muted);
      line-height: 1.75;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      padding: 18px;
    }}
    .eyebrow {{
      color: var(--accent2);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .meta {{
      color: var(--muted);
      margin: 6px 0;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #eadfce;
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #f1e8db;
      color: #714724;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Predictor Continuity Ablation Index</h1>
      <p>这个总入口汇总了同一批 case 上的 predictor continuity 消融对比页。每个子页面都是同一条 GT future video 上的左右并排对比，重点关注首个 future step 的边界衔接是否更顺，以及 boundary delta 指标是否下降。</p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [load_report(Path(item).resolve()) for item in args.reports]
    reports.sort(key=lambda item: f"{item.get('label_a', '')}-{item.get('label_b', '')}")
    (output_dir / "index.html").write_text(render_html(reports), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
