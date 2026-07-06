#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any


CSS = """
:root {
  --bg: #f4efe8;
  --paper: rgba(255, 252, 246, 0.94);
  --ink: #181412;
  --muted: #6f655c;
  --line: #d9cfc3;
  --accent: #0d6b63;
  --good: #1f7a61;
  --bad: #a53e31;
  --shadow: 0 18px 40px rgba(42, 28, 17, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
  background:
    radial-gradient(circle at top right, rgba(13,107,99,0.12), transparent 24%),
    radial-gradient(circle at left top, rgba(164,69,43,0.08), transparent 18%),
    linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
}
.wrap {
  width: min(1580px, calc(100vw - 24px));
  margin: 0 auto;
  padding: 24px 0 56px;
}
h1 {
  margin: 0 0 10px;
  font-size: clamp(32px, 4vw, 58px);
  line-height: 0.96;
  letter-spacing: -0.04em;
}
.sub {
  color: var(--muted);
  font-size: 18px;
  line-height: 1.5;
  margin-bottom: 20px;
  max-width: 980px;
}
.meta, .stats, .links {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 14px;
  background: rgba(255,255,255,0.68);
  font-size: 13px;
}
.hero, .case {
  background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(255,250,243,0.94));
  border: 1px solid var(--line);
  border-radius: 26px;
  box-shadow: var(--shadow);
}
.hero {
  padding: 22px 22px 20px;
  margin-bottom: 22px;
}
.case {
  margin-bottom: 18px;
  overflow: hidden;
}
.case-head {
  padding: 18px 18px 12px;
  border-bottom: 1px solid var(--line);
}
.case-title {
  margin: 0 0 6px;
  font-size: 24px;
  line-height: 1.05;
}
.prompt {
  color: var(--muted);
  font-size: 15px;
  line-height: 1.45;
  margin: 0;
}
.case-body {
  padding: 16px 18px 20px;
}
.video-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(320px, 1fr));
  gap: 16px;
  margin-top: 14px;
}
.card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 20px;
  overflow: hidden;
}
.card.guided {
  border-color: rgba(31,122,97,0.28);
}
.card.baseline {
  border-color: rgba(165,62,49,0.20);
}
.card video {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #111;
}
.card-body {
  padding: 14px;
}
.card-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
  margin-bottom: 8px;
}
.label {
  margin: 0;
  font-size: 18px;
  font-family: "SFMono-Regular", Consolas, monospace;
}
.delta {
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.delta.good { color: var(--good); }
.delta.bad { color: var(--bad); }
.stats .pill {
  background: rgba(255,255,255,0.78);
  padding: 6px 10px;
  font-size: 12px;
}
.links {
  margin-top: 10px;
}
.links a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px solid rgba(13,107,99,0.25);
}
.links a:hover {
  border-bottom-color: rgba(13,107,99,0.7);
}
.path {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  word-break: break-all;
  font-family: "SFMono-Regular", Consolas, monospace;
}
@media (max-width: 900px) {
  .video-grid { grid-template-columns: 1fr; }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a baseline vs guided portal for test_5 trace runs.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--title", default="Wan2.2 Official test_5 Baseline vs Guided")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_from_html(output_html: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=output_html.parent.resolve()).replace(os.sep, "/")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_delta(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return f"+{value}" if value > 0 else str(value)
    if isinstance(value, float):
        return f"+{value:.{digits}f}" if value > 0 else f"{value:.{digits}f}"
    return str(value)


def build_case_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        bucket = by_case.setdefault(str(row["case_id"]), {})
        bucket[str(row["method"])] = row
    return by_case


def build_html(summary: dict[str, Any], trace_root: Path, output_html: Path, title: str, summary_json: Path) -> None:
    rows = list(summary.get("rows", []))
    by_case = build_case_map(rows)
    ranking = list(summary.get("ranking_by_mean_delta_surprise", []))
    guided_summary = next((row for row in ranking if row.get("method") == "guided"), {})
    baseline_summary = next((row for row in ranking if row.get("method") == "baseline"), {})

    cards: list[str] = []
    for case_id in sorted(by_case):
        baseline = by_case[case_id].get("baseline")
        guided = by_case[case_id].get("guided")
        if baseline is None or guided is None:
            continue
        prompt = guided.get("prompt") or baseline.get("prompt") or ""
        trace_case_dir = trace_root / case_id
        trace_index = trace_case_dir / "case.json"
        guided_delta = guided.get("delta_surprise_vs_baseline")
        guided_delta_cls = "good" if isinstance(guided_delta, (int, float)) and guided_delta < 0 else "bad"

        cards.append(
            f"""
            <section class="case">
              <div class="case-head">
                <h2 class="case-title">{html.escape(case_id)}</h2>
                <p class="prompt">{html.escape(str(prompt))}</p>
              </div>
              <div class="case-body">
                <div class="stats">
                  <div class="pill">guided Δsurprise {html.escape(fmt_delta(guided.get('delta_surprise_vs_baseline'), 6))}</div>
                  <div class="pill">guided Δphysics_iq {html.escape(fmt_delta(guided.get('delta_physics_iq_vs_baseline'), 4))}</div>
                  <div class="pill">guided Δvideophy2 {html.escape(fmt_delta(guided.get('delta_videophy2_vs_baseline'), 4))}</div>
                  <div class="pill">guided Δcosmos {html.escape(fmt_delta(guided.get('delta_cosmos_reason1_vs_baseline'), 4))}</div>
                </div>
                <div class="video-grid">
                  <article class="card baseline">
                    <video controls preload="metadata" src="{html.escape(rel_from_html(output_html, Path(baseline['video_path'])))}"></video>
                    <div class="card-body">
                      <div class="card-head">
                        <h3 class="label">baseline</h3>
                        <div class="delta">{html.escape(fmt_delta(0.0, 6))}</div>
                      </div>
                      <div class="stats">
                        <div class="pill">surprise {html.escape(fmt(baseline.get('surprise'), 6))}</div>
                        <div class="pill">physics_iq {html.escape(fmt(baseline.get('physics_iq_score'), 4))}</div>
                        <div class="pill">videophy2 {html.escape(fmt(baseline.get('videophy2_score'), 4))}</div>
                        <div class="pill">cosmos {html.escape(fmt(baseline.get('cosmos_reason1_score'), 4))}</div>
                      </div>
                      <div class="path">{html.escape(str(baseline['video_path']))}</div>
                    </div>
                  </article>
                  <article class="card guided">
                    <video controls preload="metadata" src="{html.escape(rel_from_html(output_html, Path(guided['video_path'])))}"></video>
                    <div class="card-body">
                      <div class="card-head">
                        <h3 class="label">guided</h3>
                        <div class="delta {guided_delta_cls}">{html.escape(fmt_delta(guided_delta, 6))}</div>
                      </div>
                      <div class="stats">
                        <div class="pill">surprise {html.escape(fmt(guided.get('surprise'), 6))}</div>
                        <div class="pill">physics_iq {html.escape(fmt(guided.get('physics_iq_score'), 4))}</div>
                        <div class="pill">videophy2 {html.escape(fmt(guided.get('videophy2_score'), 4))}</div>
                        <div class="pill">cosmos {html.escape(fmt(guided.get('cosmos_reason1_score'), 4))}</div>
                      </div>
                      <div class="links">
                        {"<a href='" + html.escape(rel_from_html(output_html, trace_case_dir / "case.json")) + "'>trace case.json</a>" if trace_index.exists() else "<span class='pill'>trace pending</span>"}
                        {"<a href='" + html.escape(rel_from_html(output_html, trace_case_dir)) + "/'>open trace folder</a>" if trace_case_dir.is_dir() else ""}
                      </div>
                      <div class="path">{html.escape(str(guided['video_path']))}</div>
                    </div>
                  </article>
                </div>
              </div>
            </section>
            """
        )

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <div class="sub">
        Baseline vs guided comparison for the full <code>test_5.txt</code> set. Each case shows both videos,
        physical metrics, and a direct pointer to the guided trace artifacts when available.
      </div>
      <div class="meta">
        <div class="pill">summary: {html.escape(str(summary_json))}</div>
        <div class="pill">trace root: {html.escape(str(trace_root))}</div>
        <div class="pill">guided mean Δsurprise: {html.escape(fmt_delta(guided_summary.get('mean_delta_surprise_vs_baseline'), 6))}</div>
        <div class="pill">guided mean Δphysics_iq: {html.escape(fmt_delta(guided_summary.get('mean_delta_physics_iq_vs_baseline'), 4))}</div>
        <div class="pill">baseline mean surprise: {html.escape(fmt(baseline_summary.get('mean_surprise'), 6))}</div>
      </div>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_json = args.summary_json.expanduser().resolve()
    summary = load_json(summary_json)
    trace_root = args.trace_root.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    build_html(summary, trace_root, output_html, args.title, summary_json)
    print(output_html)


if __name__ == "__main__":
    main()
