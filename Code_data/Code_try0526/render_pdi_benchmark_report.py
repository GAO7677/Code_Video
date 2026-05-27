#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench")
OUTPUT_ROOT = BENCHMARK_ROOT / "output"
RESULT_CSV = BENCHMARK_ROOT / "result" / "metrics.csv"
REPORT_DIR = BENCHMARK_ROOT / "report"
REPORT_PATH = REPORT_DIR / "index.html"

METHODS = ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
METHOD_LABELS = {
    "GT": "GT",
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
    "VACE_1p3B_ctx08": "VACE 1.3B ctx=8",
}
TASK_ORDER = [
    "Biological_Motion",
    "Curved_Motion",
    "Dynamic_Tracking",
    "Longitudinal_Convergence",
    "partial_occlusion",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_summary_rows() -> list[dict[str, str]]:
    with RESULT_CSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relpath(target: str | Path) -> str:
    return html.escape(os.path.relpath(Path(target).resolve(), REPORT_DIR.resolve()).replace("\\", "/"))


def href_from_report(target: str | Path) -> str:
    return html.escape(os.path.relpath(Path(target).resolve(), REPORT_DIR.resolve()).replace("\\", "/"))


def discover_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    gt_root = OUTPUT_ROOT / "GT"
    task_rank = {task: idx for idx, task in enumerate(TASK_ORDER)}
    for json_path in sorted(gt_root.glob("*/*.json"), key=lambda p: (task_rank.get(p.parent.name, 99), p.parent.name, p.stem)):
        payload = load_json(json_path)
        case = {
            "task": payload["task"],
            "clip_name": payload["clip_name"],
            "prompt": payload["prompt"],
            "first_frame": payload.get("first_frame"),
            "source_video_path": payload.get("source_video_path") or payload.get("source"),
            "methods": {},
        }
        for method in METHODS:
            method_json = OUTPUT_ROOT / method / payload["task"] / f"{payload['clip_name']}.json"
            if not method_json.is_file():
                continue
            case["methods"][method] = load_json(method_json)
        cases.append(case)
    return cases


def metric_value(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    metrics = payload.get("metrics")
    if not metrics:
        return None
    value = metrics.get(key)
    if value is None or value == "":
        return None
    return float(value)


def resolve_video_path(payload: dict[str, Any]) -> str:
    for key in ("video_path", "copied_video_path", "source"):
        value = payload.get(key)
        if value:
            return str(value)
    raise KeyError("No usable video path found in payload")


def metric_text(payload: dict[str, Any] | None, key: str) -> str:
    value = metric_value(payload, key)
    return "-" if value is None else f"{value:.4f}"


def bool_text(value: Any) -> str:
    if value is True or value == "True":
        return "True"
    if value is False or value == "False":
        return "False"
    return "-"


def best_methods_for_metric(case: dict[str, Any], key: str) -> set[str]:
    pairs: list[tuple[str, float]] = []
    for method in METHODS:
        payload = case["methods"].get(method)
        value = metric_value(payload, key)
        if value is not None:
            pairs.append((method, value))
    if not pairs:
        return set()
    best = min(value for _, value in pairs)
    return {method for method, value in pairs if abs(value - best) < 1e-12}


def summary_table(rows: list[dict[str, str]]) -> str:
    header = """
    <table class="summary-table">
      <thead>
        <tr>
          <th>方法</th>
          <th>PDI 均值 ↓</th>
          <th>Scale 均值 ↓</th>
          <th>Traj 均值 ↓</th>
          <th>Rigid 均值 ↓</th>
          <th>VP 均值 ↓</th>
          <th>A / B / C</th>
          <th>RA Overall Pass</th>
        </tr>
      </thead>
      <tbody>
    """
    body_rows = []
    for row in rows:
        label = METHOD_LABELS.get(row["method"], row["method"])
        body_rows.append(
            f"""
            <tr>
              <td>{html.escape(label)}</td>
              <td>{row['mean_pdi_score']}</td>
              <td>{row['mean_scale_component']}</td>
              <td>{row['mean_traj_component']}</td>
              <td>{row['mean_epsilon_rigidity']}</td>
              <td>{row['mean_vp_component']}</td>
              <td>{row['grade_A_count']} / {row['grade_B_count']} / {row['grade_C_count']}</td>
              <td>{row['ra_overall_pass_count']} / {row['num_videos']}</td>
            </tr>
            """
        )
    return header + "".join(body_rows) + """
      </tbody>
    </table>
    """


def render_video_card(method: str, payload: dict[str, Any]) -> str:
    video_rel = href_from_report(resolve_video_path(payload))
    report_rel = href_from_report(payload["raw_report_path"]) if payload.get("raw_report_path") else ""
    metrics = payload.get("metrics", {})
    conditioning = payload.get("conditioning_mode", "-")
    return f"""
    <article class="video-card">
      <div class="video-card-head">
        <h3>{html.escape(METHOD_LABELS.get(method, method))}</h3>
        <div class="video-meta">
          <span><strong>PDI ↓</strong> {metric_text(payload, 'pdi_score')}</span>
          <span><strong>Grade</strong> {html.escape(str(metrics.get('grade_letter', '-')))}</span>
          <span><strong>条件</strong> {html.escape(conditioning)}</span>
        </div>
      </div>
      <video controls preload="metadata" src="{video_rel}"></video>
      <div class="video-links">
        <a href="{video_rel}">视频</a>
        <a href="{report_rel}">官方报告</a>
      </div>
    </article>
    """


def render_metric_row(case: dict[str, Any], method: str) -> str:
    payload = case["methods"].get(method)
    if payload is None:
        return f"""
        <tr>
          <td>{html.escape(METHOD_LABELS.get(method, method))}</td>
          <td colspan="8">缺失</td>
        </tr>
        """
    metrics = payload.get("metrics", {})
    best_pdi = "best" if method in best_methods_for_metric(case, "pdi_score") else ""
    best_scale = "best" if method in best_methods_for_metric(case, "scale_component") else ""
    best_traj = "best" if method in best_methods_for_metric(case, "traj_component") else ""
    best_rigid = "best" if method in best_methods_for_metric(case, "epsilon_rigidity") else ""
    best_vp = "best" if method in best_methods_for_metric(case, "vp_component") else ""
    return f"""
    <tr>
      <td>{html.escape(METHOD_LABELS.get(method, method))}</td>
      <td class="{best_pdi}">{metric_text(payload, 'pdi_score')}</td>
      <td class="{best_scale}">{metric_text(payload, 'scale_component')}</td>
      <td class="{best_traj}">{metric_text(payload, 'traj_component')}</td>
      <td class="{best_rigid}">{metric_text(payload, 'epsilon_rigidity')}</td>
      <td class="{best_vp}">{metric_text(payload, 'vp_component')}</td>
      <td>{html.escape(str(metrics.get('grade_letter', '-')))}</td>
      <td>{bool_text(metrics.get('ra_math_pass_bool'))}</td>
      <td>{bool_text(metrics.get('ra_overall_pass_bool'))}</td>
    </tr>
    """


def render_case(case: dict[str, Any]) -> str:
    first_frame = ""
    if case.get("first_frame"):
        first_frame = f'<img src="{href_from_report(case["first_frame"])}" alt="first frame" />'

    cards = "".join(
        render_video_card(method, case["methods"][method])
        for method in METHODS
        if method in case["methods"]
    )
    metric_rows = "".join(render_metric_row(case, method) for method in METHODS)
    return f"""
    <section class="case-card" id="{html.escape(case['task'])}-{html.escape(case['clip_name'])}">
      <div class="case-head">
        <div>
          <div class="eyebrow">{html.escape(case['task'])}</div>
          <h2>{html.escape(case['clip_name'])}</h2>
          <p class="prompt"><strong>Prompt</strong>: {html.escape(case['prompt'])}</p>
        </div>
        <div class="input-frame">
          <div class="input-title">输入首帧 / 参考首帧</div>
          {first_frame}
        </div>
      </div>
      <div class="video-grid">
        {cards}
      </div>
      <div class="table-wrap">
        <table class="case-table">
          <thead>
            <tr>
              <th>方法</th>
              <th>PDI ↓</th>
              <th>Scale ↓</th>
              <th>Traj ↓</th>
              <th>Rigid ↓</th>
              <th>VP ↓</th>
              <th>Grade</th>
              <th>RA Math</th>
              <th>RA Overall</th>
            </tr>
          </thead>
          <tbody>
            {metric_rows}
          </tbody>
        </table>
      </div>
    </section>
    """


def render_html(summary_rows: list[dict[str, str]], cases: list[dict[str, Any]]) -> str:
    case_sections = "".join(render_case(case) for case in cases)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PDI-Bench Case Report</title>
  <style>
    :root {{
      --bg: #f4efe8;
      --card: rgba(255, 251, 245, 0.92);
      --line: #d7c7b4;
      --text: #201914;
      --muted: #6f6257;
      --accent: #8a4b2b;
      --best: #e7f4e2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(193,141,102,0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(91,123,155,0.14), transparent 24%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px 28px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      letter-spacing: 0.02em;
    }}
    .sub {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.7;
      font-size: 15px;
    }}
    .note-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .note-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 14px 34px rgba(72, 51, 34, 0.08);
      line-height: 1.75;
      font-size: 14px;
    }}
    .summary-table, .case-table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
    }}
    .summary-table th, .summary-table td, .case-table th, .case-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
      vertical-align: top;
    }}
    .summary-table th, .case-table th {{
      background: rgba(244, 235, 222, 0.95);
      font-weight: 700;
    }}
    .case-card {{
      margin-top: 22px;
      background: rgba(255, 252, 247, 0.72);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 18px 44px rgba(80, 59, 40, 0.08);
    }}
    .case-head {{
      display: grid;
      grid-template-columns: 1.5fr 0.9fr;
      gap: 18px;
      align-items: start;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .prompt {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }}
    .input-frame {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 12px;
    }}
    .input-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .input-frame img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #f0ece6;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .video-card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
    }}
    .video-card h3 {{
      margin: 0;
      font-size: 20px;
    }}
    .video-card-head {{
      margin-bottom: 10px;
    }}
    .video-meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }}
    .video-card video {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
      margin-bottom: 10px;
    }}
    .video-links {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .best {{
      background: var(--best);
      font-weight: 700;
    }}
    @media (max-width: 1100px) {{
      .note-grid, .case-head, .video-grid {{
        grid-template-columns: 1fr;
      }}
      .page {{
        padding: 18px;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>PDI-Bench 官方结果总览</h1>
    <p class="sub">
      本页读取 <code>PDI-Bench/output</code> 下每个视频对应的 JSON 和官方报告，按 case 展示 GT、Wan、VACE 的视频与指标。<br />
      <strong>官方 PDI 与各个子项都是误差，统一是 ↓ 越低越好</strong>。当前官方配置里 <code>w_vp=0.0</code>，所以报告里虽然显示了 <code>VP Component</code>，但它默认不参与最终总分。
    </p>
    <div class="note-grid">
      <div class="note-card">
        <strong>怎么读总表</strong><br />
        <code>mean_pdi_score</code> 是官方总误差均值，越低越好。<code>Scale</code>、<code>Traj</code>、<code>Rigid</code>、<code>VP</code> 也都是误差。<br />
        由于某些 case 会出现很大的 <code>scale_component</code> outlier，均值可能被极少数样本明显拉高。
      </div>
      <div class="note-card">
        <strong>怎么读 case</strong><br />
        每个 case 里把同一输入对应的 4 个方法放在一起比较。下面的表格按行给出同一个 case 的官方指标，绿色高亮表示该 case 内该指标最小，也就是该指标下最优。
      </div>
    </div>
    {summary_table(summary_rows)}
    {case_sections}
  </div>
</body>
</html>
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = load_summary_rows()
    cases = discover_cases()
    REPORT_PATH.write_text(render_html(summary_rows, cases), encoding="utf-8")
    print(json.dumps({"report_path": str(REPORT_PATH), "num_cases": len(cases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
