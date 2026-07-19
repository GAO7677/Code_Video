#!/usr/bin/env python3
"""Build a static dashboard that annotates GT and generated videos with PhysV single-case metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/data/gaoya")
PHYSV_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")

if str(PHYSV_ROOT) not in sys.path:
    sys.path.insert(0, str(PHYSV_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory containing GT case JSON sidecars.")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Method spec in the form label=/path/to/method_dir. Repeat for multiple methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for report.json and index.html under /data/gaoya/agent-data/outputs.",
    )
    parser.add_argument("--title", default="PhysV Metric Dashboard")
    parser.add_argument("--subtitle", default="GT and generated videos with per-case metric annotations.")
    parser.add_argument(
        "--case-list-json",
        type=Path,
        default=None,
        help="Optional JSON with {'cases': [...]} or a plain list of case keys to include.",
    )
    parser.add_argument(
        "--compute-missing",
        action="store_true",
        help="If a sidecar is missing metrics, call physv_eval.single_case to fill them.",
    )
    parser.add_argument(
        "--refresh-metrics",
        action="store_true",
        help="Ignore sidecar metrics and recompute all requested metrics.",
    )
    parser.add_argument(
        "--serve-root",
        type=Path,
        default=DATA_ROOT,
        help="Filesystem root that will be exposed by the local HTTP server.",
    )
    parser.add_argument(
        "--browser-assets-root",
        type=Path,
        default=None,
        help="Optional root containing browser-ready mp4 assets. If omitted, try to infer it from --gt-dir.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def resolve_video_path(payload: dict[str, Any], json_path: Path) -> Path | None:
    candidates = [
        nested_get(payload, "output_video"),
        nested_get(payload, "video"),
        nested_get(payload, "video_path"),
        nested_get(payload, "paths", "output_video_path"),
        nested_get(payload, "paths", "video_path"),
        nested_get(payload, "source_video"),
        nested_get(payload, "source"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    sibling_mp4 = json_path.with_suffix(".mp4")
    return sibling_mp4 if sibling_mp4.exists() else None


def resolve_prompt(payload: dict[str, Any]) -> str:
    for key in [
        "input_prompt",
        "prompt",
        "caption",
        "input_caption",
        "text_prompt",
        "clip_name",
        "scenario",
        "description",
    ]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_case_key(payload: dict[str, Any], json_path: Path) -> str:
    for key in ["case_key", "clip_name", "name"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json_path.stem


def extract_metric_bundle(payload: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        "wmreward_surprise": first_value(
            nested_get(payload, "metric_results", "wmreward_jepa", "surprise"),
            nested_get(payload, "wmreward_jepa", "surprise"),
        ),
        "wmreward_similarity": first_value(
            nested_get(payload, "metric_results", "wmreward_jepa", "similarity"),
            nested_get(payload, "wmreward_jepa", "similarity"),
        ),
        "videophy2_sa": first_value(
            nested_get(payload, "metric_results", "videophy2_auto", "sa_score"),
            payload.get("videophy2_auto_sa"),
            payload.get("videophy2_score"),
        ),
        "videophy2_pc": first_value(
            nested_get(payload, "metric_results", "videophy2_auto", "pc_score"),
            payload.get("videophy2_auto_pc"),
            payload.get("videophy2_score"),
        ),
        "cosmos_reason1": first_value(
            nested_get(payload, "metric_results", "cosmos_reason1", "score"),
            payload.get("cosmos_reason1_score"),
            nested_get(payload, "cosmos_reason1", "score"),
        ),
    }


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def metric_missing(metrics: dict[str, Any]) -> bool:
    required = ["wmreward_surprise", "videophy2_sa", "videophy2_pc", "cosmos_reason1"]
    return any(metrics.get(name) is None for name in required)


class MetricComputer:
    def __init__(self) -> None:
        self._wmreward_runner = None
        self._videophy_runner = None
        self._cosmos_runner = None

    def compute(self, case_payload: dict[str, Any]) -> dict[str, Any]:
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner
        from physv_eval.single_case.cosmos_reason1 import score_case as cosmos_score_case
        from physv_eval.single_case.videophy2 import score_case as videophy2_score_case
        from physv_eval.single_case.wmreward import score_case as wmreward_score_case
        from physv_eval.videophy2_auto import VideoPhy2Runner
        from physv_eval.wmreward_official import WMRewardRunner

        if self._wmreward_runner is None:
            self._wmreward_runner = WMRewardRunner()
        if self._videophy_runner is None:
            self._videophy_runner = VideoPhy2Runner()
        if self._cosmos_runner is None:
            self._cosmos_runner = OfficialCosmosReason1Runner()

        wmreward = wmreward_score_case(case_payload, runner=self._wmreward_runner)
        videophy_pc = videophy2_score_case(case_payload, task="pc", runner=self._videophy_runner)
        videophy_sa = videophy2_score_case(
            case_payload,
            task="sa",
            caption=resolve_prompt(case_payload) or None,
            runner=self._videophy_runner,
        )
        cosmos = cosmos_score_case(case_payload, runner=self._cosmos_runner)
        return {
            "wmreward_surprise": wmreward.get("surprise"),
            "wmreward_similarity": wmreward.get("similarity"),
            "videophy2_sa": videophy_sa.get("score"),
            "videophy2_pc": videophy_pc.get("score"),
            "cosmos_reason1": cosmos.get("score"),
        }


@dataclass
class Entry:
    label: str
    json_path: Path
    video_path: Path | None
    metrics: dict[str, Any]
    source_video: Path | None
    prompt: str


def parse_method_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
    else:
        raw_path = spec
        label = Path(spec).name
    label = label.strip()
    method_dir = Path(raw_path).expanduser().resolve()
    if not label:
        raise ValueError(f"Invalid empty method label in spec: {spec}")
    if not method_dir.is_dir():
        raise NotADirectoryError(method_dir)
    return label, method_dir


def load_case_filter(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        values = payload.get("cases", [])
    elif isinstance(payload, list):
        values = payload
    else:
        raise TypeError(f"Unsupported case filter payload type: {type(payload)!r}")
    return {str(item) for item in values}


def to_url(path: Path | None, serve_root: Path) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    serve_root = serve_root.resolve()
    try:
        relative = resolved.relative_to(serve_root)
    except ValueError:
        return None
    return "/" + relative.as_posix()


def browser_asset_root(gt_dir: Path, override: Path | None) -> Path | None:
    if override is not None:
        return override.expanduser().resolve()
    inferred = gt_dir.resolve().parent.parent / "_viz_v2" / gt_dir.resolve().parent.name / "baseline" / "assets"
    return inferred if inferred.is_dir() else None


def browser_video_path(
    browser_root: Path | None,
    case_key: str,
    label: str,
) -> Path | None:
    if browser_root is None:
        return None
    candidate = browser_root / case_key / label / "output.browser.mp4"
    return candidate if candidate.exists() else None


def mean_or_none(values: list[float]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return statistics.mean(valid) if valid else None


def build_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for entry in entries:
        metric_rows = entry["metrics"]
        summary.append(
            {
                "label": entry["label"],
                "count": len(metric_rows),
                "wmreward_surprise": mean_or_none([row["wmreward_surprise"] for row in metric_rows]),
                "videophy2_sa": mean_or_none([row["videophy2_sa"] for row in metric_rows]),
                "videophy2_pc": mean_or_none([row["videophy2_pc"] for row in metric_rows]),
                "cosmos_reason1": mean_or_none([row["cosmos_reason1"] for row in metric_rows]),
            }
        )
    return summary


def render_html(report: dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --ink: #17201e;
      --muted: #67716d;
      --paper: #f6f1e7;
      --panel: rgba(255, 253, 247, 0.92);
      --line: #d7ccba;
      --red: #d15a36;
      --green: #12795f;
      --blue: #2e6da5;
      --gold: #c68a2e;
      --shadow: 0 18px 56px rgba(36, 41, 36, 0.11);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 0%, rgba(209,90,54,0.12), transparent 30rem),
        radial-gradient(circle at 92% 16%, rgba(18,121,95,0.12), transparent 30rem),
        linear-gradient(135deg, #f8f3e9 0%, #ece4d7 100%);
      font-family: "Trebuchet MS", "Noto Sans CJK SC", sans-serif;
      min-height: 100vh;
    }
    main { width: min(1600px, calc(100% - 34px)); margin: 0 auto; padding: 34px 0 70px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.55fr);
      gap: 30px;
      align-items: end;
      padding: 14px 0 28px;
    }
    .eyebrow {
      color: var(--red);
      text-transform: uppercase;
      letter-spacing: 0.17em;
      font: 800 12px/1.2 "Trebuchet MS", sans-serif;
    }
    h1, h2, h3 { margin: 0; font-family: Georgia, "Noto Serif CJK SC", serif; }
    h1 {
      margin-top: 10px;
      font-size: clamp(42px, 5vw, 82px);
      line-height: 0.92;
      letter-spacing: -0.055em;
    }
    .hero p {
      margin: 18px 0 0;
      max-width: 860px;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }
    .stamp {
      display: grid;
      gap: 10px;
      padding-left: 26px;
      border-left: 1px solid var(--line);
      font-size: 13px;
    }
    .stamp b { color: var(--red); }
    .summary {
      display: grid;
      grid-template-columns: 1.15fr repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 26px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px 22px 4px 4px;
      padding: 20px 22px;
      box-shadow: var(--shadow);
    }
    .verdict {
      background: #1d2725;
      color: #fffaf1;
    }
    .verdict p {
      color: #c7d1cb;
      line-height: 1.6;
      margin: 8px 0 0;
      font-size: 13px;
    }
    .metric-card span {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      display: block;
    }
    .metric-card strong {
      font: 600 34px/1 Georgia, serif;
      display: block;
      margin: 10px 0 4px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
      margin: 22px 0 14px;
    }
    .field label {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin: 0 0 6px;
    }
    select {
      width: 100%;
      border: 1px solid var(--ink);
      background: rgba(255,255,255,0.82);
      color: var(--ink);
      border-radius: 3px;
      padding: 11px 12px;
      font: 700 13px "Trebuchet MS", sans-serif;
    }
    .viewer {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 4px 28px 4px 4px;
      padding: 18px;
      box-shadow: var(--shadow);
    }
    .viewer-head {
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: start;
      margin-bottom: 14px;
    }
    .viewer h2 {
      font-size: clamp(24px, 3vw, 38px);
      letter-spacing: -0.03em;
    }
    .viewer .desc {
      margin: 7px 0 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.65;
      max-width: 980px;
    }
    .badge {
      white-space: nowrap;
      background: var(--green);
      color: white;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 11px;
      font-weight: 800;
    }
    .video-grid {
      display: grid;
      grid-template-columns: repeat(var(--columns), minmax(0, 1fr));
      gap: 14px;
    }
    .video-card {
      border: 1px solid var(--line);
      border-radius: 4px 18px 4px 4px;
      overflow: hidden;
      background: #fbf7ee;
    }
    .video-card header {
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 10px;
    }
    .video-card h3 {
      font-size: 21px;
      letter-spacing: -0.02em;
    }
    .sub {
      color: var(--muted);
      font-size: 12px;
    }
    video {
      width: 100%;
      aspect-ratio: 16 / 9;
      display: block;
      background: #141918;
    }
    .metrics {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 14px 14px;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: rgba(255,255,255,0.82);
      font-size: 12px;
    }
    .pill b { color: var(--ink); }
    .source {
      margin-top: 18px;
      padding: 18px 20px;
      border-left: 4px solid var(--gold);
      background: rgba(255,253,247,0.68);
      color: #4c5652;
      line-height: 1.65;
      font-size: 13px;
    }
    .mono {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      color: var(--muted);
      word-break: break-all;
    }
    @media (max-width: 1000px) {
      main { width: min(100% - 20px, 860px); padding-top: 20px; }
      .hero, .summary, .video-grid { grid-template-columns: 1fr; }
      .stamp { border-left: 0; border-top: 1px solid var(--line); padding: 18px 0 0; }
      .toolbar { grid-template-columns: 1fr; }
      .viewer-head { display: block; }
      .badge { display: inline-block; margin-top: 10px; }
    }
  </style>
</head>
<body>
<main>
  <header class="hero">
    <div>
      <div class="eyebrow">PhysV single-case metrics</div>
      <h1>视频里谁更像<br>物理合理世界？</h1>
      <p id="subtitle"></p>
    </div>
    <aside class="stamp" id="stamp"></aside>
  </header>

  <section class="summary" id="summary"></section>

  <section class="toolbar">
    <div class="field">
      <label>Case</label>
      <select id="case-select"></select>
    </div>
    <div class="field">
      <label>Prompt</label>
      <select id="source-select" disabled></select>
    </div>
  </section>

  <section class="viewer">
    <div class="viewer-head">
      <div>
        <h2 id="case-title"></h2>
        <p class="desc" id="case-desc"></p>
      </div>
      <span class="badge" id="case-badge"></span>
    </div>
    <div class="video-grid" id="video-grid"></div>
    <div class="source">
      <div><b>指标口径</b> WMReward 取 official <code>surprise</code>，越低越好；VideoPhy2 同时展示 <code>SA</code> 与 <code>PC</code>，越高越好；Cosmos Reason1 是 1-5 分 physical-plausibility 打分，越高越好。</div>
      <div class="mono" id="source-line"></div>
    </div>
  </section>
</main>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById('payload').textContent);
const caseSelect = document.getElementById('case-select');
const sourceSelect = document.getElementById('source-select');
const fmt = (x, digits=3) => x === null || x === undefined ? 'NA' : Number(x).toFixed(digits);
const fmtInt = x => x === null || x === undefined ? 'NA' : String(x);
document.getElementById('subtitle').textContent = DATA.subtitle;
document.getElementById('stamp').innerHTML = [
  `<span>GT <b>${DATA.gt_label}</b></span>`,
  `<span>Methods <b>${DATA.methods.map(x => x.label).join(' · ')}</b></span>`,
  `<span>Cases <b>${DATA.cases.length}</b></span>`,
  `<span>Metrics <b>WMReward / VideoPhy2 / Cosmos-R1</b></span>`
].join('');

function summaryCard(item, extraClass='') {
  return `
    <article class="card ${extraClass}">
      <div class="metric-card">
        <span>${item.label}</span>
        <strong>${fmt(item.wmreward_surprise, 4)}</strong>
        <small>mean surprise</small>
      </div>
      <div class="metrics" style="padding-left:0;padding-right:0;padding-bottom:0">
        <div class="pill">VideoPhy2-SA <b>${fmt(item.videophy2_sa, 2)}</b></div>
        <div class="pill">VideoPhy2-PC <b>${fmt(item.videophy2_pc, 2)}</b></div>
        <div class="pill">Cosmos-R1 <b>${fmt(item.cosmos_reason1, 2)}</b></div>
        <div class="pill">Cases <b>${item.count}</b></div>
      </div>
    </article>`;
}

document.getElementById('summary').innerHTML =
  `<article class="card verdict"><h2>单 case 指标直接挂到视频上</h2><p>这里不再把指标藏在 sidecar JSON 里，而是按 case 对齐 GT 和方法视频，直接展示每个视频自身的 WMReward、VideoPhy2 和 Cosmos-R1。</p></article>` +
  DATA.summary.map(item => summaryCard(item)).join('');

function renderCase() {
  const index = Number(caseSelect.value) || 0;
  const entry = DATA.cases[index];
  document.getElementById('case-title').textContent = entry.case_key;
  document.getElementById('case-desc').textContent = entry.prompt || 'No prompt found.';
  document.getElementById('case-badge').textContent = entry.category || 'uncategorized';
  sourceSelect.innerHTML = `<option>${entry.prompt || 'No prompt found.'}</option>`;
  document.getElementById('source-line').textContent = entry.source_video || 'No source video recorded.';
  document.getElementById('video-grid').style.setProperty('--columns', String(entry.entries.length));
  document.getElementById('video-grid').innerHTML = entry.entries.map(item => {
    const metrics = item.metrics || {};
    const videoTag = item.video_url
      ? `<video controls muted loop playsinline preload="metadata" src="${item.video_url}"></video>`
      : `<div style="display:grid;place-items:center;aspect-ratio:16/9;background:#1d2321;color:#c9d1cb">Missing video</div>`;
    return `
      <article class="video-card">
        <header>
          <div>
            <h3>${item.label}</h3>
            <div class="sub">${item.kind}</div>
          </div>
        </header>
        ${videoTag}
        <div class="metrics">
          <div class="pill">WMReward <b>${fmt(metrics.wmreward_surprise, 4)}</b></div>
          <div class="pill">VideoPhy2-SA <b>${fmtInt(metrics.videophy2_sa)}</b></div>
          <div class="pill">VideoPhy2-PC <b>${fmtInt(metrics.videophy2_pc)}</b></div>
          <div class="pill">Cosmos-R1 <b>${fmtInt(metrics.cosmos_reason1)}</b></div>
        </div>
      </article>`;
  }).join('');
}

caseSelect.innerHTML = DATA.cases.map((item, index) =>
  `<option value="${index}">${item.case_key}${item.category ? ' · ' + item.category : ''}</option>`
).join('');
caseSelect.addEventListener('change', renderCase);
renderCase();
</script>
</body>
</html>""".replace("__PAYLOAD__", payload).replace("__TITLE__", report["title"])


def main() -> None:
    args = parse_args()
    if not args.method:
        raise SystemExit("At least one --method is required.")
    if not args.gt_dir.is_dir():
        raise NotADirectoryError(args.gt_dir)
    if not str(args.output_dir.resolve()).startswith(str(DATA_ROOT.resolve())):
        raise ValueError(f"Output dir must live under {DATA_ROOT}")

    methods = [parse_method_spec(spec) for spec in args.method]
    case_filter = load_case_filter(args.case_list_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    browser_root = browser_asset_root(args.gt_dir, args.browser_assets_root)

    metric_computer = MetricComputer() if (args.compute_missing or args.refresh_metrics) else None
    gt_entries_for_summary: list[dict[str, Any]] = []
    method_entries_for_summary: dict[str, list[dict[str, Any]]] = {label: [] for label, _ in methods}
    cases_payload: list[dict[str, Any]] = []

    gt_json_paths = sorted(
        path for path in args.gt_dir.glob("*.json") if path.is_file() and not path.name.startswith(".")
    )
    for gt_json_path in gt_json_paths:
        gt_payload = load_json(gt_json_path)
        case_key = resolve_case_key(gt_payload, gt_json_path)
        if case_filter is not None and case_key not in case_filter:
            continue

        prompt = resolve_prompt(gt_payload)
        gt_video_path = resolve_video_path(gt_payload, gt_json_path)
        gt_browser_path = browser_video_path(browser_root, case_key, "GT")
        gt_metrics = extract_metric_bundle(gt_payload)
        if (args.refresh_metrics or metric_missing(gt_metrics)) and metric_computer is not None:
            gt_case_payload = {"video": str(gt_video_path)} if gt_video_path else {}
            if prompt:
                gt_case_payload["caption"] = prompt
            gt_metrics = metric_computer.compute(gt_case_payload)
        gt_source_video = None
        if gt_payload.get("source_video"):
            gt_source_video = Path(str(gt_payload["source_video"]))

        case_entries = [
            {
                "label": "GT",
                "kind": "ground truth",
                "video_url": to_url(gt_browser_path or gt_video_path, args.serve_root),
                "metrics": gt_metrics,
            }
        ]
        gt_entries_for_summary.append(gt_metrics)

        for label, method_dir in methods:
            method_json_path = method_dir / gt_json_path.name
            method_payload = load_json(method_json_path) if method_json_path.exists() else {}
            method_video_path = resolve_video_path(method_payload, method_json_path) if method_payload else None
            method_browser_path = browser_video_path(browser_root, case_key, label)
            method_metrics = extract_metric_bundle(method_payload) if method_payload else {
                "wmreward_surprise": None,
                "wmreward_similarity": None,
                "videophy2_sa": None,
                "videophy2_pc": None,
                "cosmos_reason1": None,
            }
            if (args.refresh_metrics or metric_missing(method_metrics)) and metric_computer is not None and method_video_path:
                case_payload = {"video": str(method_video_path)}
                if prompt:
                    case_payload["caption"] = prompt
                method_metrics = metric_computer.compute(case_payload)
            case_entries.append(
                {
                    "label": label,
                    "kind": "generated",
                    "video_url": to_url(method_browser_path or method_video_path, args.serve_root),
                    "metrics": method_metrics,
                }
            )
            method_entries_for_summary[label].append(method_metrics)

        cases_payload.append(
            {
                "case_key": case_key,
                "category": gt_payload.get("category"),
                "prompt": prompt,
                "source_video": str(gt_source_video) if gt_source_video else None,
                "entries": case_entries,
            }
        )

    summary_input = [{"label": "GT", "metrics": gt_entries_for_summary}]
    summary_input.extend({"label": label, "metrics": rows} for label, rows in method_entries_for_summary.items())
    report = {
        "title": args.title,
        "subtitle": args.subtitle,
        "gt_label": "GT",
        "methods": [{"label": label, "dir": str(path)} for label, path in methods],
        "cases": cases_payload,
        "summary": build_summary(summary_input),
        "serve_root": str(args.serve_root),
    }

    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.output_dir / "index.html").write_text(render_html(report))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "index_html": str(args.output_dir / "index.html"),
                "report_json": str(args.output_dir / "report.json"),
                "num_cases": len(cases_payload),
                "methods": [label for label, _ in methods],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
