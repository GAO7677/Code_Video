#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from physv_eval.records import load_payload, metric_value


BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark")
OUTPUT_ROOT = BENCHMARK_ROOT / "output"
REPORT_DIR = BENCHMARK_ROOT / "report_progress"
REPORT_PATH = REPORT_DIR / "index.html"

METHODS = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
METHOD_LABELS = {
    "GT": "GT",
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
    "VACE_1p3B_ctx08": "VACE 1.3B ctx=8",
}
METRICS = [
    ("official_pdi", "PDI", True),
    ("wmreward_surprise", "WMReward", True),
    ("cosmos_reason1", "Cosmos", False),
    ("vjepa_temporal_relation_raw_error", "RelRaw", True),
    ("vjepa_delta_relation_raw_error", "DeltaRel", True),
    ("vjepa_delta_profile_error", "DeltaProf", True),
    ("videophy2_auto_pc", "VPhy-PC", False),
    ("videophy2_auto_sa", "VPhy-SA", False),
]


def rel_from_report(target: str | Path) -> str:
    return html.escape(os.path.relpath(Path(target).resolve(), REPORT_DIR.resolve()).replace("\\", "/"))


def video_path_for(payload: dict[str, Any]) -> Path | None:
    for key in ("video", "video_path"):
        value = payload.get(key)
        if value and Path(value).is_file():
            return Path(value)
    output_path = (payload.get("paths") or {}).get("output_video_path")
    if output_path and Path(output_path).is_file():
        return Path(output_path)
    return None


def first_frame_for(payload: dict[str, Any]) -> Path | None:
    for key in ("first_frame",):
        value = payload.get(key)
        if value and Path(value).is_file():
            return Path(value)
    first_frame_path = (payload.get("paths") or {}).get("first_frame_path")
    if first_frame_path and Path(first_frame_path).is_file():
        return Path(first_frame_path)
    return None


def fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def discover_cases() -> list[dict[str, Any]]:
    gt_root = OUTPUT_ROOT / "GT" / "physics-iq-benchmark"
    gt_paths = sorted(gt_root.glob("*.json"))
    cases: list[dict[str, Any]] = []
    for gt_path in gt_paths:
        gt_payload = load_payload(gt_path)
        case = {
            "clip_name": gt_payload["clip_name"],
            "sample_id": gt_payload.get("sample_id", gt_payload["clip_name"]),
            "prompt": gt_payload.get("prompt", ""),
            "category": gt_payload.get("category", ""),
            "scenario": gt_payload.get("scenario", ""),
            "first_frame": first_frame_for(gt_payload),
            "methods": {},
        }
        for method in METHODS:
            json_path = OUTPUT_ROOT / method / "physics-iq-benchmark" / f"{gt_payload['clip_name']}.json"
            if json_path.is_file():
                payload = load_payload(json_path)
                case["methods"][method] = payload
        cases.append(case)
    return cases


def build_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        payloads = [case["methods"][method] for case in cases if method in case["methods"]]
        by_method[method] = {
            "count": len(payloads),
            "metric_counts": {
                key: sum(1 for payload in payloads if metric_value(payload, key) is not None)
                for key, _, _ in METRICS
            },
        }
    full_compare_count = sum(1 for case in cases if all(method in case["methods"] for method in METHODS))
    return {
        "num_cases": len(cases),
        "full_compare_count": full_compare_count,
        "by_method": by_method,
    }


def render_summary_cards(summary: dict[str, Any]) -> str:
    cards = [
        f"""
        <section class="stat-card emphasis">
          <div class="stat-label">Cases</div>
          <div class="stat-value">{summary['num_cases']}</div>
          <div class="stat-sub">GT case list discovered from output root</div>
        </section>
        """,
        f"""
        <section class="stat-card emphasis">
          <div class="stat-label">4-way Comparable</div>
          <div class="stat-value">{summary['full_compare_count']}</div>
          <div class="stat-sub">cases that currently have all four methods</div>
        </section>
        """,
    ]
    for method in METHODS:
        item = summary["by_method"][method]
        metric_parts = " · ".join(
            f"{label} {item['metric_counts'][key]}"
            for key, label, _ in METRICS
        )
        cards.append(
            f"""
            <section class="stat-card">
              <div class="stat-label">{html.escape(METHOD_LABELS[method])}</div>
              <div class="stat-value">{item['count']} / {summary['num_cases']}</div>
              <div class="stat-sub">{metric_parts}</div>
            </section>
            """
        )
    return f"<div class='stat-grid'>{''.join(cards)}</div>"


def render_method_cell(method: str, payload: dict[str, Any] | None) -> str:
    if payload is None:
        return """
        <td class="method-cell missing">
          <div class="status missing">Missing</div>
        </td>
        """

    video_path = video_path_for(payload)
    video_html = (
        f'<video controls preload="metadata" src="{rel_from_report(video_path)}"></video>'
        if video_path is not None else '<div class="video-missing">No video</div>'
    )
    metric_chips = []
    for key, label, lower_is_better in METRICS:
        value = metric_value(payload, key)
        direction = "↓" if lower_is_better else "↑"
        css = "metric-chip present" if value is not None else "metric-chip absent"
        metric_chips.append(
            f"<span class='{css}'><strong>{html.escape(label)} {direction}</strong> {html.escape(fmt(value))}</span>"
        )
    conditioning_mode = html.escape(str(payload.get("conditioning_mode", "-")))
    return f"""
    <td class="method-cell">
      <div class="status present">Ready</div>
      <div class="condition-line"><strong>Cond</strong> {conditioning_mode}</div>
      {video_html}
      <div class="metric-chip-grid">{''.join(metric_chips)}</div>
    </td>
    """


def render_case_rows(cases: list[dict[str, Any]]) -> str:
    rows = []
    for case in cases:
        first_frame = (
            f'<img src="{rel_from_report(case["first_frame"])}" alt="first frame" />'
            if case["first_frame"] is not None else "<div class='thumb-missing'>No frame</div>"
        )
        missing_methods = [METHOD_LABELS[method] for method in METHODS if method not in case["methods"]]
        status_text = "Complete" if not missing_methods else f"Missing: {', '.join(missing_methods)}"
        method_cells = "".join(render_method_cell(method, case["methods"].get(method)) for method in METHODS)
        rows.append(
            f"""
            <tr>
              <td class="meta-cell">
                <div class="thumb-box">{first_frame}</div>
                <div class="clip-name">{html.escape(case['clip_name'])}</div>
                <div class="meta-line"><strong>Category</strong> {html.escape(case['category'] or '-')}</div>
                <div class="meta-line"><strong>Status</strong> {html.escape(status_text)}</div>
                <div class="meta-line"><strong>Scenario</strong> {html.escape(case['scenario'] or '-')}</div>
                <div class="prompt-box">{html.escape(case['prompt'])}</div>
              </td>
              {method_cells}
            </tr>
            """
        )
    return "".join(rows)


def render_html(cases: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    headers = "".join(f"<th>{html.escape(METHOD_LABELS[method])}</th>" for method in METHODS)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Physics-IQ Progress Report</title>
  <style>
    :root {{
      --bg: #f6f0e7;
      --panel: rgba(255, 252, 247, 0.96);
      --line: #d8cbbb;
      --text: #241b14;
      --muted: #74675d;
      --accent: #8d4d2e;
      --ok-bg: #e8f4e5;
      --ok-text: #2f5d29;
      --missing-bg: #f8e7e6;
      --missing-text: #8c413b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(180, 132, 92, 0.16), transparent 24%),
        radial-gradient(circle at right top, rgba(77, 112, 154, 0.11), transparent 20%),
        linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1880px;
      margin: 0 auto;
      padding: 24px 22px 40px;
    }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 18px 44px rgba(76, 59, 44, 0.08);
      padding: 22px 24px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      letter-spacing: 0.01em;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }}
    .stat-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin: 18px 0 22px;
    }}
    .stat-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 16px;
      box-shadow: 0 16px 36px rgba(76, 59, 44, 0.06);
    }}
    .stat-card.emphasis {{
      background: linear-gradient(135deg, rgba(246, 236, 222, 0.98), rgba(255, 252, 247, 0.96));
    }}
    .stat-label {{
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .stat-value {{
      font-size: 32px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .stat-sub {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 18px 44px rgba(76, 59, 44, 0.08);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1780px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(245, 236, 225, 0.98);
      padding: 14px 12px;
      text-align: left;
      font-size: 14px;
    }}
    td {{
      padding: 12px;
    }}
    .meta-cell {{
      width: 370px;
      min-width: 370px;
    }}
    .method-cell {{
      width: 360px;
      min-width: 360px;
    }}
    .method-cell.missing {{
      background: rgba(248, 231, 230, 0.38);
    }}
    .thumb-box img, .thumb-missing {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #f1ece6;
      margin-bottom: 10px;
    }}
    .thumb-missing {{
      padding: 24px 12px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
    }}
    .clip-name {{
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 8px;
      line-height: 1.35;
    }}
    .meta-line {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
      line-height: 1.5;
    }}
    .prompt-box {{
      margin-top: 10px;
      color: var(--text);
      font-size: 13px;
      line-height: 1.7;
      background: rgba(255,255,255,0.52);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .status {{
      display: inline-block;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .status.present {{
      background: var(--ok-bg);
      color: var(--ok-text);
    }}
    .status.missing {{
      background: var(--missing-bg);
      color: var(--missing-text);
    }}
    .condition-line {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
      margin-bottom: 10px;
    }}
    .video-missing {{
      border: 1px dashed var(--line);
      border-radius: 14px;
      padding: 24px 12px;
      text-align: center;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .metric-chip-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .metric-chip {{
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 6px 8px;
      border-radius: 12px;
      font-size: 12px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.65);
    }}
    .metric-chip.absent {{
      color: var(--muted);
      opacity: 0.72;
    }}
    .metric-chip.present strong {{
      color: var(--accent);
    }}
    @media (max-width: 1500px) {{
      .stat-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Physics-IQ Progress Report</h1>
      <p class="sub">
        这个页面直接读取 <code>{html.escape(str(OUTPUT_ROOT))}</code> 下当前已有的 JSON / MP4，反映的是实时进度，不依赖旧的 1-case 汇总文件。
        每个方法卡片显示已生成样本数，以及已有指标覆盖数。下方按 case 展示四个方法当前有没有视频、有没有指标，可以直接在页面里播放。
      </p>
    </section>
    {render_summary_cards(summary)}
    <section class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Case</th>
            {headers}
          </tr>
        </thead>
        <tbody>
          {render_case_rows(cases)}
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = discover_cases()
    summary = build_summary(cases)
    REPORT_PATH.write_text(render_html(cases, summary), encoding="utf-8")
    print(json.dumps({
        "report_path": str(REPORT_PATH),
        "num_cases": len(cases),
        "full_compare_count": summary["full_compare_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
