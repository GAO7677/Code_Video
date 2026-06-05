#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


SCALE_ORDER = {
    "control": 0,
    "boundary0.1": 1,
    "boundary0.5": 2,
    "boundary1.0": 3,
}

SCALE_TITLES = {
    "control": "scale=0.0",
    "boundary0.1": "scale=0.1",
    "boundary0.5": "scale=0.5",
    "boundary1.0": "scale=1.0",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an aggregate index page for predictor ablation reports.")
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
    payload["_slug"] = path.parent.name
    payload["_scale_label"] = str(payload.get("label_b", "unknown"))
    return payload


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def rel_from_root(report: dict, rel_path: str) -> str:
    return html.escape(str(Path(report["_slug"]) / rel_path))


def case_nav(case_rows: list[dict]) -> str:
    items = []
    for row in case_rows:
        case_id = str(row["case_id"])
        items.append(
            f'<a class="case-chip" href="#{html.escape(case_id)}">{html.escape(case_id)}</a>'
        )
    return "".join(items)


def aggregate_cases(reports: list[dict]) -> list[dict]:
    report_by_label = {str(report["_scale_label"]): report for report in reports}
    ordered_labels = sorted(report_by_label.keys(), key=lambda item: SCALE_ORDER.get(item, 999))
    base_report = report_by_label[ordered_labels[0]]

    cases = []
    for base_case in base_report.get("cases", []):
        case_id = str(base_case["case_id"])
        scales = []
        for label in ordered_labels:
            report = report_by_label[label]
            match = next((item for item in report.get("cases", []) if str(item.get("case_id")) == case_id), None)
            if match is None:
                continue
            baseline_model = next((item for item in match.get("models", []) if item.get("label") == "baseline"), None)
            compare_model = next((item for item in match.get("models", []) if item.get("label") != "baseline"), None)
            scales.append(
                {
                    "label": label,
                    "title": SCALE_TITLES.get(label, label),
                    "slug": report["_slug"],
                    "subpage": f'{report["_slug"]}/index.html',
                    "state_compare_video": f'{report["_slug"]}/{match["state_compare_video"]}',
                    "condition_compare_video": f'{report["_slug"]}/{match["condition_compare_video"]}',
                    "baseline_metrics": baseline_model or {},
                    "compare_metrics": compare_model or {},
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "split": str(base_case.get("split", "")),
                "template_key": str(base_case.get("template_key", "")),
                "prompt": str(base_case.get("prompt", "")),
                "context_video": f'{base_report["_slug"]}/{base_case["context_video"]}',
                "gt_video": f'{base_report["_slug"]}/{base_case["gt_video"]}',
                "scales": scales,
            }
        )
    return cases


def render_summary_table(reports: list[dict]) -> str:
    rows = []
    ordered = sorted(reports, key=lambda item: SCALE_ORDER.get(str(item["_scale_label"]), 999))
    for report in ordered:
        compare_summary = next(
            (item for item in report.get("model_summaries", []) if item.get("label") != "baseline"),
            None,
        )
        if compare_summary is None:
            continue
        rows.append(
            f"""
            <tr>
              <td>{html.escape(SCALE_TITLES.get(str(report["_scale_label"]), str(report["_scale_label"])))}</td>
              <td>{fmt_metric(compare_summary.get("center_error_mean"))}</td>
              <td>{fmt_metric(compare_summary.get("boundary_center_delta_error_mean"))}</td>
              <td>{fmt_metric(compare_summary.get("boundary_motion_delta_error_mean"))}</td>
              <td>{fmt_metric(compare_summary.get("boundary_log_scale_delta_error_mean"))}</td>
              <td><a href="{html.escape(str(report["_slug"]) + '/index.html')}">子页</a></td>
            </tr>
            """
        )
    return "".join(rows)


def render_case_section(case_row: dict) -> str:
    baseline_metrics = {}
    if case_row["scales"]:
        baseline_metrics = case_row["scales"][0].get("baseline_metrics", {})
    baseline_pred = baseline_metrics.get("predictor_metrics", {})
    baseline_boundary = baseline_metrics.get("boundary_metrics", {})

    compare_cards = []
    for scale in case_row["scales"]:
        compare_pred = scale["compare_metrics"].get("predictor_metrics", {})
        compare_boundary = scale["compare_metrics"].get("boundary_metrics", {})
        compare_cards.append(
            f"""
            <article class="video-card compare-card">
              <div class="card-topline">
                <div>
                  <div class="eyebrow">Ablation</div>
                  <h3>{html.escape(scale["title"])}</h3>
                </div>
                <a class="subpage-link" href="{html.escape(scale["subpage"])}">子页</a>
              </div>
              <video controls preload="none" muted playsinline loop src="{html.escape(scale["state_compare_video"])}"></video>
              <div class="metric-strip">
                <span>Ctr {fmt_metric(compare_pred.get("center_error"))}</span>
                <span>BC {fmt_metric(compare_boundary.get("boundary_center_delta_error"))}</span>
                <span>BM {fmt_metric(compare_boundary.get("boundary_motion_delta_error"))}</span>
                <span>BS {fmt_metric(compare_boundary.get("boundary_log_scale_delta_error"))}</span>
              </div>
              <details>
                <summary>Condition 对比视频</summary>
                <video controls preload="none" muted playsinline loop src="{html.escape(scale["condition_compare_video"])}"></video>
              </details>
            </article>
            """
        )

    return f"""
    <section class="case-section" id="{html.escape(case_row["case_id"])}">
      <div class="case-header">
        <div>
          <div class="eyebrow">{html.escape(case_row["split"])} | {html.escape(case_row["template_key"])}</div>
          <h2>{html.escape(case_row["case_id"])}</h2>
        </div>
        <p class="prompt">{html.escape(case_row["prompt"])}</p>
      </div>

      <div class="baseline-panel">
        <div class="baseline-title">Baseline 指标</div>
        <div class="metric-strip">
          <span>Ctr {fmt_metric(baseline_pred.get("center_error"))}</span>
          <span>BC {fmt_metric(baseline_boundary.get("boundary_center_delta_error"))}</span>
          <span>BM {fmt_metric(baseline_boundary.get("boundary_motion_delta_error"))}</span>
          <span>BS {fmt_metric(baseline_boundary.get("boundary_log_scale_delta_error"))}</span>
        </div>
      </div>

      <div class="lead-grid">
        <article class="video-card">
          <div class="eyebrow">Context</div>
          <h3>context video</h3>
          <video controls preload="none" muted playsinline loop src="{html.escape(case_row["context_video"])}"></video>
        </article>
        <article class="video-card">
          <div class="eyebrow">Ground Truth</div>
          <h3>future gt</h3>
          <video controls preload="none" muted playsinline loop src="{html.escape(case_row["gt_video"])}"></video>
        </article>
      </div>

      <div class="compare-grid">
        {''.join(compare_cards)}
      </div>
    </section>
    """


def render_html(reports: list[dict]) -> str:
    reports = sorted(reports, key=lambda item: SCALE_ORDER.get(str(item["_scale_label"]), 999))
    case_rows = aggregate_cases(reports)
    summary_rows = render_summary_table(reports)
    case_sections = "".join(render_case_section(item) for item in case_rows)
    nav = case_nav(case_rows)

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>predictor continuity ablation aggregate</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 246, 0.96);
      --line: #dfd3c4;
      --ink: #1f1f1b;
      --muted: #6f675d;
      --accent: #0f5a52;
      --accent2: #b8642a;
      --shadow: 0 18px 40px rgba(55, 40, 22, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
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
      max-width: 1760px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .summary, .case-section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 16px;
    }}
    .hero p {{
      color: var(--muted);
      line-height: 1.7;
      margin: 10px 0 0;
    }}
    .case-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .case-chip {{
      text-decoration: none;
      color: var(--accent);
      border: 1px solid rgba(15, 90, 82, 0.18);
      background: rgba(15, 90, 82, 0.06);
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }}
    .summary {{
      padding: 18px 20px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent2);
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
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
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .case-section {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    .case-header {{
      display: grid;
      grid-template-columns: minmax(0, 380px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
      margin-bottom: 14px;
    }}
    .case-header h2 {{
      margin: 0;
      font-size: 28px;
    }}
    .prompt {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      background: rgba(255,255,255,0.55);
      border: 1px dashed #dbcdb9;
      border-radius: 14px;
      padding: 14px 16px;
    }}
    .baseline-panel {{
      margin-bottom: 16px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid #e6dac8;
      background: linear-gradient(180deg, rgba(255,255,255,0.75), rgba(248,241,231,0.95));
    }}
    .baseline-title {{
      font-size: 13px;
      color: #7a694f;
      margin-bottom: 8px;
      font-weight: 700;
    }}
    .metric-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .metric-strip span {{
      display: inline-flex;
      align-items: center;
      padding: 7px 10px;
      border-radius: 999px;
      background: #f1e8db;
      color: #5b4b37;
      font-size: 13px;
      font-weight: 700;
    }}
    .lead-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 16px;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .video-card {{
      background: rgba(255,255,255,0.62);
      border: 1px solid #e6dac8;
      border-radius: 16px;
      padding: 14px;
    }}
    .video-card h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    .card-topline {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 12px;
    }}
    .subpage-link {{
      white-space: nowrap;
      font-size: 13px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #161616;
      margin-bottom: 10px;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    @media (max-width: 1200px) {{
      .case-header, .lead-grid, .compare-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">Aggregate View</div>
      <h1>Predictor Continuity Ablation</h1>
      <p>这个页面把同一批 case 的四组 continuity scale 结果直接汇总到一页。每个 case 先给出 context video 和 GT future video，再把各个 scale 的 state overlay 对比视频并排放出来，方便直接看 future 边界是否更连续、同时有没有牺牲整体轨迹质量。</p>
      <div class="case-nav">{nav}</div>
    </section>

    <section class="summary">
      <div class="eyebrow">Global Summary</div>
      <table>
        <thead>
          <tr>
            <th>Scale</th>
            <th>Center</th>
            <th>Boundary Center</th>
            <th>Boundary Motion</th>
            <th>Boundary Scale</th>
            <th>Link</th>
          </tr>
        </thead>
        <tbody>
          {summary_rows}
        </tbody>
      </table>
    </section>

    {case_sections}
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [load_report(Path(item).resolve()) for item in args.reports]
    (output_dir / "index.html").write_text(render_html(reports), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
