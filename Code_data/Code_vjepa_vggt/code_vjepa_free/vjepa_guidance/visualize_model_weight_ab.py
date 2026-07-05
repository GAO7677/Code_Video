#!/usr/bin/env python3
from __future__ import annotations

"""
Build and optionally serve a local dashboard for model-weight A/B results.

Build only:
  python visualize_model_weight_ab.py \
    --scores-dir /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/scores \
    --output-html /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/ab_dashboard/index.html

Build + serve from /data/gaoya:
  python visualize_model_weight_ab.py \
    --scores-dir /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/scores \
    --output-html /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/ab_dashboard/index.html \
    --serve \
    --serve-root /data/gaoya \
    --port 8891
"""

import argparse
import html
import json
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any


SERVE_ROOT_DEFAULT = Path("/data/gaoya")

FAMILY_TITLES = {
    "train0705_step002500": "train0705 step-002500",
    "train0705_step005000": "train0705 step-005000",
    "wan22_official_ti2v5b": "Wan2.2 official TI2V-5B",
    "wan22_early_lora_step000500": "Wan2.2 early LoRA step-000500",
}

SUMMARY_ORDER = [
    "wan22_official_ti2v5b",
    "wan22_early_lora_step000500",
    "train0705_step002500",
    "train0705_step005000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a dashboard for model-weight A/B results.")
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--serve-root", type=Path, default=SERVE_ROOT_DEFAULT)
    parser.add_argument("--title", type=str, default="Model-Weight A/B Dashboard")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8891)
    return parser.parse_args()


def family_sort_key(family_id: str) -> tuple[int, str]:
    try:
        return (SUMMARY_ORDER.index(family_id), family_id)
    except ValueError:
        return (len(SUMMARY_ORDER), family_id)


def safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_web_path(file_path: str | None, serve_root: Path) -> str | None:
    if not file_path:
        return None
    path = Path(file_path).resolve()
    try:
        rel = path.relative_to(serve_root.resolve())
    except ValueError:
        return None
    return "/" + rel.as_posix()


def load_dashboard_payload(scores_dir: Path, serve_root: Path) -> dict[str, Any]:
    family_files = sorted(
        [
            path
            for path in scores_dir.glob("*_summary.json")
            if path.name != "combined_summary.json"
        ],
        key=lambda path: family_sort_key(path.stem.removesuffix("_summary")),
    )

    families: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}

    for summary_path in family_files:
        family_id = summary_path.stem.removesuffix("_summary")
        payload = safe_read_json(summary_path)
        summary_by_method = payload.get("summary_by_method", {})
        rows = payload.get("rows", [])

        family_info = {
            "family_id": family_id,
            "title": FAMILY_TITLES.get(family_id, family_id),
            "summary": summary_by_method,
            "method_dirs": payload.get("method_dirs", {}),
        }
        families.append(family_info)

        for row in rows:
            case_id = row["case_id"]
            method = row["method"]
            case_info = cases.setdefault(
                case_id,
                {
                    "case_id": case_id,
                    "prompt": row.get("prompt"),
                    "source_video": row.get("source_video"),
                    "input_json": row.get("input_json"),
                    "families": {},
                },
            )
            case_info["prompt"] = case_info["prompt"] or row.get("prompt")
            case_info["source_video"] = case_info["source_video"] or row.get("source_video")
            case_info["input_json"] = case_info["input_json"] or row.get("input_json")

            family_bucket = case_info["families"].setdefault(
                family_id,
                {
                    "family_id": family_id,
                    "title": FAMILY_TITLES.get(family_id, family_id),
                    "baseline": None,
                    "guided": None,
                },
            )
            family_bucket[method] = {
                "case_id": case_id,
                "method": method,
                "video_path": row.get("video_path"),
                "video_url": to_web_path(row.get("video_path"), serve_root),
                "sidecar_path": row.get("sidecar_path"),
                "physics_iq_score": row.get("physics_iq_score"),
                "videophy2_score": row.get("videophy2_score"),
                "cosmos_reason1_score": row.get("cosmos_reason1_score"),
                "surprise": row.get("surprise"),
                "similarity": row.get("similarity"),
            }

    case_list: list[dict[str, Any]] = []
    for case_id in sorted(cases):
        case = cases[case_id]
        family_entries: list[dict[str, Any]] = []
        for family_id in sorted(case["families"], key=family_sort_key):
            entry = case["families"][family_id]
            baseline = entry.get("baseline")
            guided = entry.get("guided")

            delta = {
                "surprise": None,
                "physics_iq": None,
                "videophy2": None,
                "cosmos_reason1": None,
                "similarity": None,
            }
            if baseline and guided:
                for key, base_key, guided_key in [
                    ("surprise", "surprise", "surprise"),
                    ("physics_iq", "physics_iq_score", "physics_iq_score"),
                    ("videophy2", "videophy2_score", "videophy2_score"),
                    ("cosmos_reason1", "cosmos_reason1_score", "cosmos_reason1_score"),
                    ("similarity", "similarity", "similarity"),
                ]:
                    base_val = baseline.get(base_key)
                    guided_val = guided.get(guided_key)
                    if base_val is not None and guided_val is not None:
                        delta[key] = float(guided_val) - float(base_val)

            family_entries.append(
                {
                    "family_id": family_id,
                    "title": entry["title"],
                    "baseline": baseline,
                    "guided": guided,
                    "delta": delta,
                }
            )

        case_list.append(
            {
                "case_id": case["case_id"],
                "prompt": case["prompt"],
                "source_video": case["source_video"],
                "source_video_url": to_web_path(case["source_video"], serve_root),
                "input_json": case["input_json"],
                "families": family_entries,
            }
        )

    return {
        "families": sorted(families, key=lambda item: family_sort_key(item["family_id"])),
        "cases": case_list,
    }


def render_html(payload: dict[str, Any], title: str, output_html: Path) -> None:
    data_json = json.dumps(payload, ensure_ascii=False)
    family_count = len(payload["families"])
    case_count = len(payload["cases"])
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f2efe7;
      --panel: #fffdf8;
      --ink: #161412;
      --muted: #6f685d;
      --line: #d9d0c3;
      --accent: #155a71;
      --good: #1b6e4f;
      --bad: #a3432f;
      --shadow: 0 18px 38px rgba(39, 25, 14, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(21,90,113,0.12), transparent 26%),
        radial-gradient(circle at left top, rgba(163,67,47,0.10), transparent 18%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
      font-family: Georgia, "Times New Roman", serif;
    }}
    .wrap {{
      width: min(1680px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(34px, 4vw, 58px);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .sub {{
      color: var(--muted);
      font-size: 17px;
      margin: 10px 0 20px;
    }}
    .meta, .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .pill, select {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.78);
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 14px;
      color: var(--ink);
    }}
    select {{
      min-width: 380px;
    }}
    .section {{
      margin-top: 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 22px;
    }}
    .section h2 {{
      margin: 0 0 14px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
      gap: 16px;
    }}
    .family-card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      background: #fffefb;
      padding: 16px;
    }}
    .family-card h3 {{
      margin: 0 0 10px;
      font-size: 19px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 12px;
    }}
    .kpi {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: rgba(245,240,232,0.55);
    }}
    .kpi .label {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .kpi .value {{
      font-size: 22px;
      letter-spacing: -0.03em;
      font-weight: 700;
    }}
    .delta.good {{ color: var(--good); }}
    .delta.bad {{ color: var(--bad); }}
    .delta.neutral {{ color: var(--ink); }}
    .case-head {{
      display: grid;
      grid-template-columns: 1.3fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    .case-meta {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(250,247,241,0.7);
    }}
    .case-meta .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 5px;
    }}
    .case-meta .body {{
      font-size: 15px;
      line-height: 1.45;
      word-break: break-word;
    }}
    .family-compare {{
      margin-top: 18px;
      display: grid;
      gap: 16px;
    }}
    .compare-card {{
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 14px;
      background: #fffefb;
    }}
    .compare-top {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .compare-top h3 {{
      margin: 0;
      font-size: 20px;
    }}
    .delta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
    }}
    .delta-pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 11px;
      font-size: 12px;
      background: rgba(255,255,255,0.78);
    }}
    .video-pair {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}
    .video-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: #faf7f1;
    }}
    .video-card header {{
      padding: 10px 12px;
      font-size: 15px;
      font-weight: 700;
      border-bottom: 1px solid var(--line);
    }}
    video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      background: #141414;
    }}
    .video-metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 12px 12px;
    }}
    .metric-pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 9px;
      font-size: 12px;
      background: rgba(255,255,255,0.82);
    }}
    .mono {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }}
    .family-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
      font-size: 14px;
    }}
    .family-table th, .family-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 8px;
      text-align: left;
      vertical-align: top;
    }}
    .family-table th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1100px) {{
      .case-head {{ grid-template-columns: 1fr; }}
      .video-pair {{ grid-template-columns: 1fr; }}
      select {{ min-width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{html.escape(title)}</h1>
    <div class="sub">All scored A/B families in one dashboard. Top section is family-level summary. Bottom section aligns the same test sample across different model families and shows baseline/guided videos with metrics.</div>
    <div class="meta">
      <div class="pill">Families: {family_count}</div>
      <div class="pill">Cases: {case_count}</div>
      <div class="pill">Serve root: /data/gaoya</div>
    </div>

    <section class="section">
      <h2>Family Summary</h2>
      <div id="summary-grid" class="summary-grid"></div>
    </section>

    <section class="section">
      <h2>Case Explorer</h2>
      <div class="controls">
        <select id="case-select"></select>
      </div>
      <div id="case-view"></div>
    </section>
  </div>

  <script>
    const DATA = {data_json};

    function fmt(value, digits = 4) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'NA';
      return Number(value).toFixed(digits);
    }}

    function clsForDelta(value, preferNegative=false) {{
      if (value === null || value === undefined || Number.isNaN(value)) return 'neutral';
      if (Math.abs(value) < 1e-12) return 'neutral';
      if (preferNegative) return value < 0 ? 'good' : 'bad';
      return value > 0 ? 'good' : 'bad';
    }}

    function renderSummary() {{
      const grid = document.getElementById('summary-grid');
      grid.innerHTML = '';
      DATA.families.forEach((family) => {{
        const baseline = family.summary.baseline || {{}};
        const guided = family.summary.guided || {{}};
        const card = document.createElement('div');
        card.className = 'family-card';
        card.innerHTML = `
          <h3>${{family.title}}</h3>
          <div class="muted mono">${{family.family_id}}</div>
          <div class="kpi-grid">
            <div class="kpi">
              <div class="label">Delta Surprise</div>
              <div class="value delta ${{clsForDelta(guided.mean_delta_surprise_vs_baseline, true)}}">${{fmt(guided.mean_delta_surprise_vs_baseline, 6)}}</div>
            </div>
            <div class="kpi">
              <div class="label">Delta Physics-IQ</div>
              <div class="value delta ${{clsForDelta(guided.mean_delta_physics_iq_vs_baseline, false)}}">${{fmt(guided.mean_delta_physics_iq_vs_baseline, 4)}}</div>
            </div>
            <div class="kpi">
              <div class="label">Delta VideoPhy2</div>
              <div class="value delta ${{clsForDelta(guided.mean_delta_videophy2_vs_baseline, false)}}">${{fmt(guided.mean_delta_videophy2_vs_baseline, 4)}}</div>
            </div>
            <div class="kpi">
              <div class="label">Delta Cosmos-R1</div>
              <div class="value delta ${{clsForDelta(guided.mean_delta_cosmos_reason1_vs_baseline, false)}}">${{fmt(guided.mean_delta_cosmos_reason1_vs_baseline, 4)}}</div>
            </div>
          </div>
          <table class="family-table">
            <thead>
              <tr>
                <th>Method</th>
                <th>Cases</th>
                <th>Surprise</th>
                <th>Physics-IQ</th>
                <th>VideoPhy2</th>
                <th>Cosmos-R1</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>baseline</td>
                <td>${{baseline.num_cases ?? 'NA'}}</td>
                <td>${{fmt(baseline.mean_surprise, 6)}}</td>
                <td>${{fmt(baseline.mean_physics_iq, 4)}}</td>
                <td>${{fmt(baseline.mean_videophy2, 4)}}</td>
                <td>${{fmt(baseline.mean_cosmos_reason1, 4)}}</td>
              </tr>
              <tr>
                <td>guided</td>
                <td>${{guided.num_cases ?? 'NA'}}</td>
                <td>${{fmt(guided.mean_surprise, 6)}}</td>
                <td>${{fmt(guided.mean_physics_iq, 4)}}</td>
                <td>${{fmt(guided.mean_videophy2, 4)}}</td>
                <td>${{fmt(guided.mean_cosmos_reason1, 4)}}</td>
              </tr>
            </tbody>
          </table>
        `;
        grid.appendChild(card);
      }});
    }}

    function metricPills(record) {{
      if (!record) return '<div class="muted">Missing</div>';
      return `
        <div class="metric-pill">surprise ${{fmt(record.surprise, 6)}}</div>
        <div class="metric-pill">similarity ${{fmt(record.similarity, 6)}}</div>
        <div class="metric-pill">physics-IQ ${{fmt(record.physics_iq_score, 2)}}</div>
        <div class="metric-pill">videophy2 ${{fmt(record.videophy2_score, 2)}}</div>
        <div class="metric-pill">cosmos-R1 ${{fmt(record.cosmos_reason1_score, 2)}}</div>
      `;
    }}

    function videoCard(title, record) {{
      if (!record) {{
        return `
          <div class="video-card">
            <header>${{title}}</header>
            <div style="aspect-ratio:16/9;background:#141414"></div>
            <div class="video-metrics"><div class="metric-pill">Missing</div></div>
          </div>
        `;
      }}
      const videoTag = record.video_url
        ? `<video controls preload="metadata" src="${{record.video_url}}"></video>`
        : `<div style="aspect-ratio:16/9;background:#141414"></div>`;
      const path = record.video_path ? `<div class="mono" style="padding:0 12px 12px">${{record.video_path}}</div>` : '';
      return `
        <div class="video-card">
          <header>${{title}}</header>
          ${{videoTag}}
          <div class="video-metrics">${{metricPills(record)}}</div>
          ${{path}}
        </div>
      `;
    }}

    function renderCase(caseId) {{
      const entry = DATA.cases.find((item) => item.case_id === caseId);
      const view = document.getElementById('case-view');
      if (!entry) {{
        view.innerHTML = '<div class="muted">Case not found.</div>';
        return;
      }}

      const source = entry.source_video_url
        ? `<a href="${{entry.source_video_url}}" target="_blank" rel="noopener">${{entry.source_video}}</a>`
        : (entry.source_video || 'NA');
      const inputJson = entry.input_json || 'NA';

      let compareHtml = '';
      entry.families.forEach((family) => {{
        compareHtml += `
          <article class="compare-card">
            <div class="compare-top">
              <h3>${{family.title}}</h3>
              <div class="delta-row">
                <div class="delta-pill delta ${{clsForDelta(family.delta.surprise, true)}}">Δsurprise ${{fmt(family.delta.surprise, 6)}}</div>
                <div class="delta-pill delta ${{clsForDelta(family.delta.physics_iq, false)}}">Δphysics-IQ ${{fmt(family.delta.physics_iq, 2)}}</div>
                <div class="delta-pill delta ${{clsForDelta(family.delta.videophy2, false)}}">Δvideophy2 ${{fmt(family.delta.videophy2, 2)}}</div>
                <div class="delta-pill delta ${{clsForDelta(family.delta.cosmos_reason1, false)}}">Δcosmos-R1 ${{fmt(family.delta.cosmos_reason1, 2)}}</div>
              </div>
            </div>
            <div class="video-pair">
              ${{videoCard('Baseline', family.baseline)}}
              ${{videoCard('Guided', family.guided)}}
            </div>
          </article>
        `;
      }});

      view.innerHTML = `
        <div class="case-head">
          <div class="case-meta">
            <div class="label">Case ID</div>
            <div class="body mono">${{entry.case_id}}</div>
          </div>
          <div class="case-meta">
            <div class="label">Input JSON</div>
            <div class="body mono">${{inputJson}}</div>
          </div>
          <div class="case-meta">
            <div class="label">Prompt</div>
            <div class="body">${{entry.prompt || 'NA'}}</div>
          </div>
          <div class="case-meta">
            <div class="label">Source Video</div>
            <div class="body mono">${{source}}</div>
          </div>
        </div>
        <div class="family-compare">${{compareHtml}}</div>
      `;
    }}

    function init() {{
      renderSummary();
      const select = document.getElementById('case-select');
      DATA.cases.forEach((entry) => {{
        const option = document.createElement('option');
        option.value = entry.case_id;
        const prompt = entry.prompt ? ` | ${{entry.prompt.slice(0, 80)}}` : '';
        option.textContent = `${{entry.case_id}}${{prompt}}`;
        select.appendChild(option);
      }});
      select.addEventListener('change', () => renderCase(select.value));
      if (DATA.cases.length > 0) {{
        select.value = DATA.cases[0].case_id;
        renderCase(select.value);
      }}
    }}

    init();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html_text, encoding="utf-8")


def serve_forever(serve_root: Path, port: int) -> None:
    handler = partial(SimpleHTTPRequestHandler, directory=str(serve_root))
    httpd = HTTPServer(("0.0.0.0", port), handler)
    print(f"Serving {serve_root} at http://localhost:{port}", flush=True)
    httpd.serve_forever()


def main() -> None:
    args = parse_args()
    payload = load_dashboard_payload(args.scores_dir, args.serve_root)
    render_html(payload, args.title, args.output_html)
    print(args.output_html, flush=True)
    if args.serve:
        serve_forever(args.serve_root, args.port)


if __name__ == "__main__":
    main()
