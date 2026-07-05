from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/report/same_input_metric_portal")
DEFAULT_METRICS_DOC = Path("/home/gaoya/Code_Video/Code_data/Code_try0526/AAAmd/eval_metrics.md")
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}


@dataclass(frozen=True)
class MetricColumn:
    key: str
    label: str
    direction: str
    precision: int
    path: tuple[str, ...]
    short_note: str


CORE_COLUMNS: tuple[MetricColumn, ...] = (
    MetricColumn("physics_iq_score", "Physics-IQ", "up", 2, ("physics_iq", "score"), "Approx single-view score"),
    MetricColumn("wmreward_surprise", "WMReward Surprise", "down", 4, ("wmreward", "surprise"), "Lower is better"),
    MetricColumn("wmreward_similarity", "WMReward Similarity", "up", 4, ("wmreward", "similarity"), "1 - surprise"),
    MetricColumn("videophy2_score", "VideoPhy-2", "up", 2, ("videophy2", "score"), "Judge score"),
    MetricColumn("phyground_general_avg", "PhyGround", "up", 3, ("phyground", "general_avg"), "Judge average"),
    MetricColumn("cosmos_reason1_score", "Cosmos-Reason1", "up", 2, ("cosmos_reason1", "score"), "Judge score"),
    MetricColumn("pdi_score", "PDI", "down", 4, ("pdi", "pdi_score"), "Official error score"),
    MetricColumn(
        "proxy_temporal_relation_raw_error",
        "Proxy Temporal Err",
        "down",
        4,
        ("proxy", "details", "temporal_relation_raw_error"),
        "Lower is better",
    ),
    MetricColumn(
        "proxy_delta_relation_raw_error",
        "Proxy Delta Err",
        "down",
        4,
        ("proxy", "details", "delta_relation_raw_error"),
        "Lower is better",
    ),
    MetricColumn(
        "proxy_delta_profile_error",
        "Proxy Profile Err",
        "down",
        4,
        ("proxy", "details", "delta_profile_error"),
        "Lower is better",
    ),
)

PHYSICS_IQ_COLUMNS: tuple[MetricColumn, ...] = (
    MetricColumn("physics_iq_mse_mean", "MSE Mean", "down", 6, ("physics_iq", "mse_mean"), "Pixel MSE after alignment"),
    MetricColumn(
        "physics_iq_spatiotemporal_iou_mean",
        "ST-IoU Mean",
        "up",
        6,
        ("physics_iq", "spatiotemporal_iou_mean"),
        "Per-frame motion-mask IoU",
    ),
    MetricColumn("physics_iq_spatial_iou", "Spatial IoU", "up", 6, ("physics_iq", "spatial_iou"), "Union over time"),
    MetricColumn(
        "physics_iq_weighted_spatial_iou",
        "Weighted Spatial IoU",
        "up",
        6,
        ("physics_iq", "weighted_spatial_iou"),
        "Time-weighted mask overlap",
    ),
    MetricColumn("physics_iq_raw_score", "Raw Score", "up", 6, ("physics_iq", "raw_score"), "Before x100 clip"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group V2V outputs by the same input video/json, then build a static portal "
            "to compare Physics-IQ and related metrics across methods."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metrics-doc", type=Path, default=DEFAULT_METRICS_DOC)
    parser.add_argument("--min-methods", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def format_value(value: float | None, precision: int) -> str:
    if value is None:
        return "NA"
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    slug = slug.strip("._-")
    return slug or "unknown"


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def resolve_abs(value: str | None) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).expanduser().resolve()


def rel_href(page_dir: Path, target: Path) -> str:
    return os.path.relpath(str(target), str(page_dir)).replace(os.sep, "/")


def parse_direction_table(path: Path) -> dict[str, dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: dict[str, dict[str, str]] = {}
    in_table = False
    for line in lines:
        if line.strip().startswith("| 指标 |"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.strip().startswith("|"):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 4:
            continue
        metric_name, primary_field, direction, description = parts
        if metric_name == "---":
            continue
        rows[metric_name] = {
            "field": primary_field,
            "direction": direction,
            "description": description,
        }
    return rows


def discover_result_jsons(result_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(result_root.rglob("*.json")):
        if path.name in EXCLUDED_JSON_NAMES:
            continue
        if path.name.startswith("eval_summary_"):
            continue
        payload = load_json(path)
        if payload is None or "input_json" not in payload:
            continue
        paths.append(path)
    return paths


def collect_metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    for column in CORE_COLUMNS + PHYSICS_IQ_COLUMNS:
        metrics[column.key] = to_float(nested_get(payload, column.path))
    return metrics


def derive_method_name(payload: dict[str, Any], result_json_path: Path) -> str:
    method = payload.get("method")
    if isinstance(method, str) and method.strip():
        return method.strip()
    return result_json_path.parent.name or result_json_path.stem


def load_case_meta(input_json_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    input_payload = load_json(input_json_path) or {}
    source_video = resolve_abs(
        input_payload.get("source_video")
        or payload.get("source_video")
    )
    input_video = resolve_abs(
        payload.get("input_video")
        or input_payload.get("input_video")
        or input_payload.get("context_video")
    )
    caption = (
        payload.get("input_caption")
        or input_payload.get("input_caption")
        or input_payload.get("prompt")
        or input_payload.get("caption")
        or input_payload.get("input_caption")
    )
    return {
        "input_json_path": input_json_path,
        "input_payload": input_payload,
        "source_video": source_video,
        "input_video": input_video,
        "caption": caption if isinstance(caption, str) else None,
    }


def compute_best_ids(rows: list[dict[str, Any]], columns: tuple[MetricColumn, ...]) -> dict[str, set[str]]:
    best: dict[str, set[str]] = {}
    for column in columns:
        numeric_pairs = [
            (row["row_id"], row["metrics"].get(column.key))
            for row in rows
            if row["metrics"].get(column.key) is not None
        ]
        if not numeric_pairs:
            continue
        values = [value for _, value in numeric_pairs if value is not None]
        target = min(values) if column.direction == "down" else max(values)
        best[column.key] = {
            row_id
            for row_id, value in numeric_pairs
            if value is not None and abs(value - target) <= 1e-12
        }
    return best


def build_case_page(
    *,
    case_name: str,
    page_dir: Path,
    group_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    directions: dict[str, dict[str, str]],
) -> str:
    best_core = compute_best_ids(rows, CORE_COLUMNS)
    best_piq = compute_best_ids(rows, PHYSICS_IQ_COLUMNS)
    input_video = group_meta.get("input_video")
    source_video = group_meta.get("source_video")
    caption = group_meta.get("caption")

    direction_cards: list[str] = []
    for column in CORE_COLUMNS:
        note = column.short_note
        if column.label == "WMReward Surprise":
            doc = directions.get("WMReward")
        elif column.label.startswith("Proxy"):
            doc = directions.get("Geometry Proxy / VJEPA Proxy")
        elif column.label == "VideoPhy-2":
            doc = directions.get("VideoPhy-2")
        elif column.label == "PhyGround":
            doc = directions.get("PhyGround")
        elif column.label == "Cosmos-Reason1":
            doc = directions.get("Cosmos-Reason1")
        elif column.label == "PDI":
            doc = directions.get("PDI-Bench Official")
        else:
            doc = directions.get("Physics-IQ Approx")
        body = note
        if doc is not None:
            body = f"{doc['direction']} | {doc['field']}"
        direction_cards.append(
            f"""
            <div class="chip">
              <div class="chip-title">{html_escape(column.label)}</div>
              <div class="chip-body">{html_escape(body)}</div>
            </div>
            """
        )

    summary_cards = [
        f'<div class="summary-card"><span>Methods</span><strong>{len(rows)}</strong></div>',
        f'<div class="summary-card"><span>Input JSON</span><strong>{html_escape(group_meta["input_json_path"].name)}</strong></div>',
    ]
    if input_video is not None:
        summary_cards.append(f'<div class="summary-card"><span>Input Video</span><strong>{html_escape(input_video.name)}</strong></div>')
    if source_video is not None:
        summary_cards.append(f'<div class="summary-card"><span>Reference Video</span><strong>{html_escape(source_video.name)}</strong></div>')

    reference_cards: list[str] = []
    if input_video is not None and input_video.is_file():
        reference_cards.append(
            f"""
            <article class="ref-card">
              <h3>Input Video</h3>
              <p>{html_escape(str(input_video))}</p>
              <video controls preload="metadata" playsinline src="{html_escape(rel_href(page_dir, input_video))}"></video>
            </article>
            """
        )
    if source_video is not None and source_video.is_file():
        reference_cards.append(
            f"""
            <article class="ref-card">
              <h3>Reference / Source Video</h3>
              <p>{html_escape(str(source_video))}</p>
              <video controls preload="metadata" playsinline src="{html_escape(rel_href(page_dir, source_video))}"></video>
            </article>
            """
        )

    core_header = ['<th class="sticky-col">Method</th>']
    core_header.extend(f"<th>{html_escape(column.label)}</th>" for column in CORE_COLUMNS)
    piq_header = ['<th class="sticky-col">Method</th>']
    piq_header.extend(f"<th>{html_escape(column.label)}</th>" for column in PHYSICS_IQ_COLUMNS)

    core_rows: list[str] = []
    piq_rows: list[str] = []
    gallery_cards: list[str] = []

    for row in rows:
        row_id = row["row_id"]
        label_html = (
            f'<div class="method-name">{html_escape(row["method_name"])}</div>'
            f'<div class="method-path">{html_escape(row["result_json_path"])}</div>'
        )
        core_cells = [f'<td class="sticky-col">{label_html}</td>']
        piq_cells = [f'<td class="sticky-col">{label_html}</td>']
        score_bits: list[str] = []
        piq_bits: list[str] = []
        badges: list[str] = []

        for column in CORE_COLUMNS:
            value = row["metrics"].get(column.key)
            is_best = row_id in best_core.get(column.key, set()) and value is not None
            klass = "metric-cell best" if is_best else "metric-cell"
            core_cells.append(f'<td class="{klass}">{html_escape(format_value(value, column.precision))}</td>')
            if value is not None:
                score_bits.append(
                    f'<div class="metric-line{" best" if is_best else ""}"><span>{html_escape(column.label)}</span><strong>{html_escape(format_value(value, column.precision))}</strong></div>'
                )
            if is_best:
                badges.append(f'<span class="badge">{html_escape(column.label)} best</span>')

        for column in PHYSICS_IQ_COLUMNS:
            value = row["metrics"].get(column.key)
            is_best = row_id in best_piq.get(column.key, set()) and value is not None
            klass = "metric-cell best" if is_best else "metric-cell"
            piq_cells.append(f'<td class="{klass}">{html_escape(format_value(value, column.precision))}</td>')
            if value is not None:
                piq_bits.append(
                    f'<div class="metric-line{" best" if is_best else ""}"><span>{html_escape(column.label)}</span><strong>{html_escape(format_value(value, column.precision))}</strong></div>'
                )

        core_rows.append(f"<tr>{''.join(core_cells)}</tr>")
        piq_rows.append(f"<tr>{''.join(piq_cells)}</tr>")

        gallery_cards.append(
            f"""
            <article class="video-card">
              <div class="video-head">
                <div>
                  <h3>{html_escape(row["method_name"])}</h3>
                  <p>{html_escape(row["result_json_path"])}</p>
                </div>
                <div class="badges">{''.join(badges) if badges else '<span class="badge muted">No best badge</span>'}</div>
              </div>
              <video controls preload="metadata" playsinline src="{html_escape(rel_href(page_dir, row['output_video_path']))}"></video>
              <div class="metric-grid">
                {''.join(score_bits) if score_bits else '<div class="metric-line"><span>No metrics</span><strong>NA</strong></div>'}
              </div>
              <div class="metric-grid piq-grid">
                {''.join(piq_bits) if piq_bits else '<div class="metric-line"><span>No Physics-IQ details</span><strong>NA</strong></div>'}
              </div>
            </article>
            """
        )

    caption_block = (
        f'<div class="caption-box"><span>Prompt / Caption</span><strong>{html_escape(caption)}</strong></div>'
        if caption
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(case_name)}</title>
  <style>
    :root {{
      --bg: #f6efe3;
      --paper: #fffdf8;
      --ink: #1f1a16;
      --muted: #6e655c;
      --line: #d9cdbd;
      --accent: #0f766e;
      --accent-soft: #d8f4ec;
      --best: #fff3b0;
      --best-border: #d79c00;
      --shadow: 0 18px 50px rgba(60, 43, 23, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.11), transparent 24rem),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 22rem),
        linear-gradient(180deg, #faf5ed 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }}
    .shell {{ max-width: 1760px; margin: 0 auto; padding: 26px 22px 38px; }}
    .hero, .panel {{
      background: rgba(255, 253, 248, 0.9);
      border: 1px solid rgba(217, 205, 189, 0.9);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 24px; }}
    h1 {{
      margin: 0;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(28px, 4vw, 46px);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .sub {{
      color: var(--muted);
      margin: 12px 0 0;
      line-height: 1.65;
      max-width: 1100px;
    }}
    .summary-grid, .ref-grid, .chip-grid, .gallery {{
      display: grid;
      gap: 16px;
    }}
    .summary-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-top: 18px;
    }}
    .summary-card, .chip, .caption-box {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .summary-card span, .caption-box span {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .summary-card strong, .caption-box strong {{ line-height: 1.5; }}
    .caption-box {{ margin-top: 18px; }}
    .chip-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin-top: 18px;
    }}
    .chip-title {{ font-weight: 700; margin-bottom: 6px; }}
    .chip-body {{ color: var(--muted); font-size: 13px; line-height: 1.45; }}
    .formula {{
      margin-top: 18px;
      background: #f3eee6;
      border: 1px dashed #c9bba7;
      border-radius: 18px;
      padding: 16px 18px;
    }}
    .formula code {{
      display: block;
      font-size: 14px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .panel {{ margin-top: 22px; overflow: hidden; }}
    .panel-head {{ padding: 18px 22px 12px; border-bottom: 1px solid rgba(217, 205, 189, 0.7); }}
    .panel-head h2 {{ margin: 0; font-size: 24px; font-family: "IBM Plex Serif", "Noto Serif SC", serif; }}
    .panel-head p {{ margin: 8px 0 0; color: var(--muted); }}
    .ref-grid {{
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      padding: 18px;
    }}
    .ref-card, .video-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      overflow: hidden;
      box-shadow: 0 10px 26px rgba(50, 36, 20, 0.07);
    }}
    .ref-card h3, .video-head h3 {{ margin: 0; font-size: 18px; }}
    .ref-card p, .video-head p {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-all;
    }}
    .ref-card h3, .ref-card p {{ padding: 18px 18px 0; }}
    .video-head {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      padding: 18px 18px 12px;
    }}
    video {{
      display: block;
      width: 100%;
      background: #15110e;
      aspect-ratio: 16 / 9;
    }}
    .gallery {{
      grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
      padding: 18px;
    }}
    .table-wrap {{ overflow: auto; }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1400px;
      font-size: 14px;
    }}
    th, td {{
      padding: 13px 12px;
      border-bottom: 1px solid rgba(217, 205, 189, 0.7);
      border-right: 1px solid rgba(217, 205, 189, 0.4);
      vertical-align: top;
      background: rgba(255, 253, 248, 0.76);
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: #eee2d0;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
      text-align: left;
    }}
    .sticky-col {{
      position: sticky;
      left: 0;
      z-index: 2;
      min-width: 360px;
      background: #faf4ea;
    }}
    th.sticky-col {{ z-index: 4; background: #e6d6bf; }}
    .method-name {{ font-weight: 700; margin-bottom: 6px; }}
    .method-path {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-all;
    }}
    .metric-cell {{ font-variant-numeric: tabular-nums; font-weight: 600; }}
    .best {{
      background: linear-gradient(180deg, #fff9d3 0%, var(--best) 100%);
      box-shadow: inset 0 0 0 2px var(--best-border);
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 10px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.muted {{
      color: #925b10;
      background: #fff0dd;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 16px 18px;
    }}
    .piq-grid {{
      border-top: 1px solid rgba(217, 205, 189, 0.65);
      background: #fbf7f0;
    }}
    .metric-line {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      border: 1px solid rgba(217, 205, 189, 0.75);
      background: #f8f1e8;
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 13px;
    }}
    .metric-line span {{ color: var(--muted); }}
    .metric-line strong {{ font-variant-numeric: tabular-nums; }}
    .back-link {{
      display: inline-flex;
      margin-top: 18px;
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    @media (max-width: 900px) {{
      .shell {{ padding: 16px 12px 28px; }}
      .hero {{ padding: 18px; }}
      .sticky-col {{ min-width: 280px; }}
      .gallery, .ref-grid, .metric-grid {{ grid-template-columns: 1fr; }}
      .video-head {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>{html_escape(case_name)}</h1>
      <p class="sub">This page compares different outputs generated from the same input video/json. Physics-IQ here is the project single-view approximate variant, not the official multi-view benchmark.</p>
      <div class="summary-grid">
        {''.join(summary_cards)}
      </div>
      {caption_block}
      <div class="chip-grid">
        {''.join(direction_cards)}
      </div>
      <div class="formula">
        <strong>Physics-IQ computation</strong>
        <code>score = 100 * clip(((spatiotemporal_iou_mean + spatial_iou + weighted_spatial_iou) / 3) - mse_mean, 0, 1)</code>
        <p class="sub">Implementation path: <code>physv_eval/single_case/physics_iq.py</code>. The scorer aligns the candidate and source videos by timestamp up to the shorter duration, downsamples to one-quarter source resolution, builds motion masks, then combines three IoU terms against pixel MSE.</p>
      </div>
      <a class="back-link" href="../../index.html">Back to portal index</a>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Reference Clips</h2>
        <p>The same inputs are shown once here so you can compare all generated outputs against the same conditioning clip and reference future video.</p>
      </div>
      <div class="ref-grid">
        {''.join(reference_cards) if reference_cards else '<div class="summary-card">No reference videos could be resolved for this case.</div>'}
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Core Metrics</h2>
        <p>Best values are highlighted per column, respecting each metric direction.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>{''.join(core_header)}</tr></thead>
          <tbody>{''.join(core_rows)}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Physics-IQ Breakdown</h2>
        <p>These are the exact components used inside the approximate Physics-IQ formula for this repo.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr>{''.join(piq_header)}</tr></thead>
          <tbody>{''.join(piq_rows)}</tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Output Gallery</h2>
        <p>Each generated video card carries both the cross-metric summary and the Physics-IQ subterms for fast visual cross-checking.</p>
      </div>
      <div class="gallery">
        {''.join(gallery_cards)}
      </div>
    </section>
  </main>
</body>
</html>
"""


def build_index_page(
    *,
    output_root: Path,
    case_rows: list[dict[str, Any]],
) -> str:
    table_rows: list[str] = []
    card_rows: list[str] = []
    for row in case_rows:
        caption = row.get("caption") or ""
        table_rows.append(
            f"""
            <tr>
              <td><a href="{html_escape(row['rel_page'])}">{html_escape(row['case_name'])}</a></td>
              <td>{row['num_methods']}</td>
              <td>{html_escape(format_value(row.get('best_physics_iq'), 2))}</td>
              <td>{html_escape(row.get('best_physics_iq_method') or 'NA')}</td>
              <td>{html_escape(format_value(row.get('best_wmreward_similarity'), 4))}</td>
              <td>{html_escape(row.get('best_wmreward_method') or 'NA')}</td>
              <td>{html_escape(row['input_json_name'])}</td>
            </tr>
            """
        )
        card_rows.append(
            f"""
            <article class="case-card">
              <h3><a href="{html_escape(row['rel_page'])}">{html_escape(row['case_name'])}</a></h3>
              <div class="meta">methods {row['num_methods']} | best Physics-IQ {html_escape(format_value(row.get('best_physics_iq'), 2))}</div>
              <p>{html_escape(caption[:240] + ('...' if len(caption) > 240 else ''))}</p>
              <div class="mini-grid">
                <div><span>Best Physics-IQ</span><strong>{html_escape(row.get('best_physics_iq_method') or 'NA')}</strong></div>
                <div><span>Best WMReward Sim</span><strong>{html_escape(row.get('best_wmreward_method') or 'NA')}</strong></div>
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Same Input Video Metric Portal</title>
  <style>
    :root {{
      --bg: #f5efe4;
      --paper: #fffdf8;
      --ink: #211b16;
      --muted: #6f665d;
      --line: #d8ccbc;
      --accent: #0f766e;
      --shadow: 0 18px 50px rgba(61, 44, 23, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.11), transparent 24rem),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 22rem),
        linear-gradient(180deg, #faf5ed 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }}
    .shell {{ max-width: 1640px; margin: 0 auto; padding: 28px 22px 38px; }}
    .hero, .panel {{
      background: rgba(255, 253, 248, 0.9);
      border: 1px solid rgba(216, 204, 188, 0.9);
      border-radius: 26px;
      box-shadow: var(--shadow);
    }}
    .hero {{ padding: 24px; }}
    h1 {{
      margin: 0;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .sub {{ color: var(--muted); margin: 12px 0 0; line-height: 1.65; max-width: 1100px; }}
    .formula {{
      margin-top: 18px;
      padding: 16px 18px;
      border-radius: 18px;
      background: #f3eee6;
      border: 1px dashed #c8b9a5;
    }}
    code {{
      display: block;
      margin-top: 8px;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    .panel {{ margin-top: 22px; overflow: hidden; }}
    .panel-head {{ padding: 18px 22px 12px; border-bottom: 1px solid rgba(216, 204, 188, 0.7); }}
    .panel-head h2 {{ margin: 0; font-size: 24px; font-family: "IBM Plex Serif", "Noto Serif SC", serif; }}
    .panel-head p {{ margin: 8px 0 0; color: var(--muted); }}
    .table-wrap {{ overflow: auto; }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1100px;
    }}
    th, td {{
      padding: 13px 12px;
      border-bottom: 1px solid rgba(216, 204, 188, 0.7);
      border-right: 1px solid rgba(216, 204, 188, 0.45);
      background: rgba(255, 253, 248, 0.76);
      text-align: left;
    }}
    th {{
      background: #eee2d0;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
    }}
    a {{ color: var(--accent); text-decoration: none; font-weight: 700; }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 18px;
      padding: 18px;
    }}
    .case-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 10px 26px rgba(50, 36, 20, 0.07);
    }}
    .case-card h3 {{ margin: 0; font-size: 20px; }}
    .case-card p {{ color: var(--muted); line-height: 1.6; margin: 12px 0; }}
    .meta {{ color: var(--muted); margin-top: 8px; font-size: 13px; }}
    .mini-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .mini-grid div {{
      background: #f8f1e8;
      border: 1px solid rgba(216, 204, 188, 0.8);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .mini-grid span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
    }}
    @media (max-width: 900px) {{
      .shell {{ padding: 16px 12px 28px; }}
      .hero {{ padding: 18px; }}
      .mini-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Same Input Video Metric Portal</h1>
      <p class="sub">This portal groups outputs by the same input video/json so we can compare different generations against identical conditioning. Each case page includes the generated videos, the approximate Physics-IQ score and its sub-terms, plus the other benchmark metrics already written into each result json.</p>
      <div class="formula">
        <strong>Physics-IQ in this repo</strong>
        <code>100 * clip(((spatiotemporal_iou_mean + spatial_iou + weighted_spatial_iou) / 3) - mse_mean, 0, 1)</code>
        <p class="sub">Implementation: <code>/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/single_case/physics_iq.py</code>. This is a single-view approximate metric, not the official multi-view Physics-IQ benchmark.</p>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Case Index</h2>
        <p>Sorted by the number of methods available for the same input case.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Case</th>
              <th>Methods</th>
              <th>Best Physics-IQ</th>
              <th>Best Physics-IQ Method</th>
              <th>Best WMReward Sim</th>
              <th>Best WMReward Method</th>
              <th>Input JSON</th>
            </tr>
          </thead>
          <tbody>
            {''.join(table_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Case Cards</h2>
        <p>Use these when you want a faster visual entry point than the full table.</p>
      </div>
      <div class="gallery">
        {''.join(card_rows)}
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    metrics_doc = args.metrics_doc.expanduser().resolve()
    directions = parse_direction_table(metrics_doc)

    grouped: dict[str, dict[str, Any]] = {}
    for result_json_path in discover_result_jsons(result_root):
        payload = load_json(result_json_path)
        if payload is None:
            continue
        input_json_path = resolve_abs(payload.get("input_json"))
        output_video_path = resolve_abs(payload.get("output_video")) or result_json_path.with_suffix(".mp4")
        if input_json_path is None or not input_json_path.is_file():
            continue
        if output_video_path is None or not output_video_path.is_file():
            continue
        key = str(input_json_path)
        if key not in grouped:
            grouped[key] = {
                "group_meta": load_case_meta(input_json_path, payload),
                "rows": [],
            }
        grouped[key]["rows"].append(
            {
                "row_id": slugify(str(result_json_path.relative_to(result_root))),
                "result_json_path": str(result_json_path),
                "output_video_path": output_video_path,
                "method_name": derive_method_name(payload, result_json_path),
                "metrics": collect_metrics(payload),
            }
        )

    case_items: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for bundle in grouped.values():
        rows = sorted(bundle["rows"], key=lambda item: (item["method_name"], item["result_json_path"]))
        if len(rows) < args.min_methods:
            continue
        case_items.append((bundle["group_meta"], rows))

    case_items.sort(key=lambda item: (-len(item[1]), item[0]["input_json_path"].name))
    if args.max_cases is not None:
        case_items = case_items[: args.max_cases]

    output_root.mkdir(parents=True, exist_ok=True)
    case_summaries: list[dict[str, Any]] = []

    for group_meta, rows in case_items:
        case_slug = slugify(group_meta["input_json_path"].stem)
        case_name = group_meta["input_json_path"].stem
        page_dir = output_root / "cases" / case_slug
        html_text = build_case_page(
            case_name=case_name,
            page_dir=page_dir,
            group_meta=group_meta,
            rows=rows,
            directions=directions,
        )
        write_text(page_dir / "index.html", html_text)

        best_piq_row = max(
            (row for row in rows if row["metrics"].get("physics_iq_score") is not None),
            key=lambda row: row["metrics"]["physics_iq_score"],
            default=None,
        )
        best_wm_row = max(
            (row for row in rows if row["metrics"].get("wmreward_similarity") is not None),
            key=lambda row: row["metrics"]["wmreward_similarity"],
            default=None,
        )
        case_summaries.append(
            {
                "case_name": case_name,
                "rel_page": rel_href(output_root, page_dir / "index.html"),
                "num_methods": len(rows),
                "input_json_name": group_meta["input_json_path"].name,
                "caption": group_meta.get("caption"),
                "best_physics_iq": None if best_piq_row is None else best_piq_row["metrics"].get("physics_iq_score"),
                "best_physics_iq_method": None if best_piq_row is None else best_piq_row["method_name"],
                "best_wmreward_similarity": None if best_wm_row is None else best_wm_row["metrics"].get("wmreward_similarity"),
                "best_wmreward_method": None if best_wm_row is None else best_wm_row["method_name"],
            }
        )

    index_html = build_index_page(output_root=output_root, case_rows=case_summaries)
    write_text(output_root / "index.html", index_html)
    write_text(output_root / "summary.json", json.dumps(case_summaries, ensure_ascii=False, indent=2) + "\n")
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
