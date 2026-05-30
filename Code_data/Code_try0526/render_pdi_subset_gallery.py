#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench")
OUTPUT_ROOT = BENCHMARK_ROOT / "output"
REPORT_DIR = BENCHMARK_ROOT / "report_subset"
REPORT_PATH = REPORT_DIR / "index.html"
MANIFEST_PATH = REPORT_DIR / "selected_cases.json"

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


@dataclass(slots=True)
class CaseRecord:
    task: str
    clip_name: str
    prompt: str
    first_frame: str | None
    source_video_path: str | None
    methods: dict[str, dict[str, Any]]
    spread: float
    selected_reason: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def href_from_report(target: str | Path) -> str:
    return html.escape(
        os.path.relpath(Path(target).resolve(),
                        REPORT_DIR.resolve()).replace("\\", "/"))


def resolve_video_path(payload: dict[str, Any]) -> str:
    for key in ("copied_video_path", "video_path", "source"):
        value = payload.get(key)
        if value:
            return str(value)
    raise KeyError("No usable video path found in payload")


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


def metric_text(payload: dict[str, Any] | None, key: str) -> str:
    value = metric_value(payload, key)
    return "-" if value is None else f"{value:.4f}"


def bool_text(value: Any) -> str:
    if value is True or value == "True":
        return "True"
    if value is False or value == "False":
        return "False"
    return "-"


def discover_all_cases() -> list[CaseRecord]:
    cases: list[CaseRecord] = []
    gt_root = OUTPUT_ROOT / "GT"
    task_rank = {task: idx for idx, task in enumerate(TASK_ORDER)}
    for json_path in sorted(gt_root.glob("*/*.json"),
                            key=lambda p: (task_rank.get(p.parent.name, 99),
                                           p.parent.name, p.stem)):
        gt_payload = load_json(json_path)
        methods: dict[str, dict[str, Any]] = {}
        valid = True
        pdi_values: list[float] = []
        for method in METHODS:
            method_json = OUTPUT_ROOT / method / gt_payload["task"] / f"{gt_payload['clip_name']}.json"
            if not method_json.is_file():
                valid = False
                break
            payload = load_json(method_json)
            methods[method] = payload
            pdi = metric_value(payload, "pdi_score")
            if pdi is None or not math.isfinite(pdi):
                valid = False
                break
            pdi_values.append(pdi)
        if not valid:
            continue
        spread = max(pdi_values) - min(pdi_values)
        cases.append(
            CaseRecord(
                task=gt_payload["task"],
                clip_name=gt_payload["clip_name"],
                prompt=gt_payload["prompt"],
                first_frame=gt_payload.get("first_frame"),
                source_video_path=gt_payload.get("source_video_path")
                or gt_payload.get("source"),
                methods=methods,
                spread=spread,
                selected_reason="",
            ))
    return cases


def choose_representative_cases(cases: list[CaseRecord]) -> list[CaseRecord]:
    selected: list[CaseRecord] = []
    by_task: dict[str, list[CaseRecord]] = {task: [] for task in TASK_ORDER}
    for case in cases:
        by_task.setdefault(case.task, []).append(case)
    for task in TASK_ORDER:
        candidates = sorted(by_task.get(task, []),
                            key=lambda case: (-case.spread, case.clip_name))
        if not candidates:
            continue
        case = candidates[0]
        best_method = min(
            METHODS,
            key=lambda method: metric_value(case.methods[method], "pdi_score"))
        case.selected_reason = (
            f"Selected as the highest-disagreement case in `{task}`. "
            f"PDI spread across methods is {case.spread:.4f}; "
            f"lowest PDI comes from {METHOD_LABELS[best_method]}."
        )
        selected.append(case)
    return selected


def best_methods_for_metric(case: CaseRecord, key: str) -> set[str]:
    pairs: list[tuple[str, float]] = []
    for method in METHODS:
        value = metric_value(case.methods.get(method), key)
        if value is not None:
            pairs.append((method, value))
    if not pairs:
        return set()
    best = min(value for _, value in pairs)
    return {method for method, value in pairs if abs(value - best) < 1e-12}


def render_video_card(method: str, payload: dict[str, Any]) -> str:
    metrics = payload.get("metrics", {})
    video_rel = href_from_report(resolve_video_path(payload))
    report_rel = href_from_report(payload["raw_report_path"])
    grade = html.escape(str(metrics.get("grade", "-")))
    conditioning = html.escape(str(payload.get("conditioning_mode", "-")))
    return f"""
    <article class="video-card">
      <div class="video-card-head">
        <div>
          <div class="method-name">{html.escape(METHOD_LABELS.get(method, method))}</div>
          <div class="meta-line">
            <span><strong>PDI</strong> {metric_text(payload, 'pdi_score')}</span>
            <span><strong>Scale</strong> {metric_text(payload, 'scale_component')}</span>
            <span><strong>Traj</strong> {metric_text(payload, 'traj_component')}</span>
          </div>
          <div class="meta-line">
            <span><strong>Rigid</strong> {metric_text(payload, 'epsilon_rigidity')}</span>
            <span><strong>VP</strong> {metric_text(payload, 'vp_component')}</span>
            <span><strong>Grade</strong> {grade}</span>
          </div>
        </div>
        <div class="pill">{conditioning}</div>
      </div>
      <video controls preload="metadata" src="{video_rel}"></video>
      <div class="link-row">
        <a href="{video_rel}">Open video</a>
        <a href="{report_rel}">Open official report</a>
      </div>
    </article>
    """


def render_metric_row(case: CaseRecord, method: str) -> str:
    payload = case.methods[method]
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
      <td>{html.escape(str(metrics.get('grade', '-')))}</td>
      <td>{bool_text(metrics.get('ra_math_pass'))}</td>
      <td>{bool_text(metrics.get('ra_overall_pass'))}</td>
    </tr>
    """


def render_case(case: CaseRecord) -> str:
    cards = "".join(
        render_video_card(method, case.methods[method]) for method in METHODS)
    rows = "".join(render_metric_row(case, method) for method in METHODS)
    frame_html = ""
    if case.first_frame:
        frame_html = (
            f'<img src="{href_from_report(case.first_frame)}" alt="first frame" />')
    source_line = ""
    if case.source_video_path:
        source_line = (
            f'<div class="source-line"><strong>Source</strong>: '
            f'<code>{html.escape(case.source_video_path)}</code></div>')
    return f"""
    <section class="case-card" id="{html.escape(case.task)}-{html.escape(case.clip_name)}">
      <div class="case-top">
        <div class="case-copy">
          <div class="eyebrow">{html.escape(case.task)}</div>
          <h2>{html.escape(case.clip_name)}</h2>
          <p class="prompt"><strong>Prompt</strong>: {html.escape(case.prompt)}</p>
          <p class="reason">{html.escape(case.selected_reason)}</p>
          {source_line}
        </div>
        <div class="frame-panel">
          <div class="frame-title">Reference First Frame</div>
          {frame_html}
        </div>
      </div>
      <div class="video-grid">
        {cards}
      </div>
      <div class="table-wrap">
        <table class="metric-table">
          <thead>
            <tr>
              <th>Method</th>
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
            {rows}
          </tbody>
        </table>
      </div>
    </section>
    """


def render_navigation(cases: list[CaseRecord]) -> str:
    items = []
    for case in cases:
        anchor = f"{case.task}-{case.clip_name}"
        items.append(
            f'<a href="#{html.escape(anchor)}">{html.escape(case.task)} / {html.escape(case.clip_name)}</a>'
        )
    return "".join(items)


def render_html(cases: list[CaseRecord]) -> str:
    sections = "".join(render_case(case) for case in cases)
    nav = render_navigation(cases)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PDI-Bench Representative Cases</title>
  <style>
    :root {{
      --bg: #f6f1ea;
      --panel: rgba(255, 252, 246, 0.95);
      --line: #dacabb;
      --text: #231b14;
      --muted: #726457;
      --accent: #8e4d2d;
      --best: #e4f3dd;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--text);
      font-family: "Helvetica Neue", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      background:
        radial-gradient(circle at 0% 0%, rgba(180, 132, 92, 0.18), transparent 25%),
        radial-gradient(circle at 100% 0%, rgba(76, 112, 156, 0.12), transparent 22%),
        linear-gradient(180deg, #f9f5ef 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1720px;
      margin: 0 auto;
      padding: 24px 24px 48px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 1.7fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .hero-card, .nav-card, .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 42px rgba(82, 63, 47, 0.08);
    }}
    .hero-card {{
      padding: 22px 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 38px;
      letter-spacing: 0.01em;
    }}
    .sub {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 15px;
    }}
    .nav-card {{
      padding: 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-content: start;
    }}
    .nav-card a {{
      color: var(--accent);
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.55);
      font-size: 13px;
    }}
    .nav-card a:hover {{ background: rgba(255,255,255,0.9); }}
    .case-card {{
      padding: 20px;
      margin-bottom: 22px;
    }}
    .case-top {{
      display: grid;
      grid-template-columns: 1.5fr 0.8fr;
      gap: 18px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    .prompt, .reason, .source-line {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.75;
      font-size: 14px;
    }}
    .frame-panel {{
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.5);
    }}
    .frame-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .frame-panel img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid var(--line);
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }}
    .video-card {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.58);
    }}
    .video-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .method-name {{
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .meta-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .pill {{
      height: fit-content;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: #000;
      margin-bottom: 10px;
    }}
    .link-row {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .link-row a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: rgba(255,255,255,0.58);
    }}
    .metric-table th, .metric-table td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
    }}
    .metric-table th {{
      background: rgba(245, 236, 225, 0.98);
    }}
    .best {{
      background: var(--best);
      font-weight: 700;
    }}
    code {{
      font-family: "SFMono-Regular", "Consolas", monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    @media (max-width: 1200px) {{
      .hero, .case-top, .video-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <section class="hero-card">
        <h1>PDI-Bench Representative Case Gallery</h1>
        <p class="sub">
          This page picks one representative case per benchmark task, using the largest cross-method PDI spread within that task.
          Each case shows the same sample across <code>GT</code>, <code>Wan2.2-5B TI2V</code>, <code>VACE 1.3B TI2V</code>, and <code>VACE 1.3B ctx=8</code>,
          with playable videos, the reference first frame, and official PDI sub-metrics. All metrics are errors, so lower is better.
        </p>
      </section>
      <nav class="nav-card">
        {nav}
      </nav>
    </div>
    {sections}
  </div>
</body>
</html>
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cases = choose_representative_cases(discover_all_cases())
    REPORT_PATH.write_text(render_html(cases), encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(
            [{
                "task": case.task,
                "clip_name": case.clip_name,
                "spread": round(case.spread, 6),
                "reason": case.selected_reason,
            } for case in cases],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report_path": str(REPORT_PATH),
                "manifest_path": str(MANIFEST_PATH),
                "num_cases": len(cases),
            },
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
