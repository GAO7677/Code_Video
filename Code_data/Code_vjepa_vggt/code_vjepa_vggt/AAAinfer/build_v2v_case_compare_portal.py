from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MetricColumn:
    key: str
    label: str
    direction: str
    source_label: str
    field_path: tuple[str, ...]
    precision: int = 4
    note: str | None = None


METRIC_COLUMNS: tuple[MetricColumn, ...] = (
    MetricColumn(
        key="pdi_score",
        label="PDI",
        direction="down",
        source_label="PDI-Bench Official",
        field_path=("pdi", "pdi_score"),
        precision=4,
    ),
    MetricColumn(
        key="wmreward_surprise",
        label="WMReward Surprise",
        direction="down",
        source_label="WMReward",
        field_path=("wmreward", "surprise"),
        precision=4,
    ),
    MetricColumn(
        key="proxy_temporal_relation_raw_error",
        label="Proxy Temporal Err",
        direction="down",
        source_label="Geometry Proxy / VJEPA Proxy",
        field_path=("proxy", "details", "temporal_relation_raw_error"),
        precision=4,
    ),
    MetricColumn(
        key="proxy_delta_relation_raw_error",
        label="Proxy Delta Err",
        direction="down",
        source_label="Geometry Proxy / VJEPA Proxy",
        field_path=("proxy", "details", "delta_relation_raw_error"),
        precision=4,
    ),
    MetricColumn(
        key="proxy_delta_profile_error",
        label="Proxy Profile Err",
        direction="down",
        source_label="Geometry Proxy / VJEPA Proxy",
        field_path=("proxy", "details", "delta_profile_error"),
        precision=4,
    ),
    MetricColumn(
        key="videophy2_score",
        label="VideoPhy-2",
        direction="up",
        source_label="VideoPhy-2",
        field_path=("videophy2", "score"),
        precision=2,
    ),
    MetricColumn(
        key="phyground_general_avg",
        label="PhyGround Avg",
        direction="up",
        source_label="PhyGround",
        field_path=("phyground", "general_avg"),
        precision=3,
    ),
    MetricColumn(
        key="cosmos_reason1_score",
        label="Cosmos-Reason1",
        direction="up",
        source_label="Cosmos-Reason1",
        field_path=("cosmos_reason1", "score"),
        precision=2,
    ),
    MetricColumn(
        key="physics_iq_score",
        label="Physics-IQ Approx",
        direction="up",
        source_label="Physics-IQ Approx",
        field_path=("physics_iq", "score"),
        precision=2,
        note="Direction inferred from score semantics because it is not listed in eval_metrics.md.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local HTML portal comparing multiple V2V outputs for one case."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-doc", type=Path, required=True)
    parser.add_argument("--title", type=str, required=True)
    parser.add_argument("--video-rel-path", action="append", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def nested_get(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
        return None
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
        if math.isfinite(number):
            return number
    return None


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
        if not in_table:
            continue
        if not line.strip().startswith("|"):
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


def format_value(value: float | None, precision: int) -> str:
    if value is None:
        return "NA"
    text = f"{value:.{precision}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def slugify(rel_path: str) -> str:
    return (
        rel_path.replace("./", "")
        .replace("/", "__")
        .replace(".mp4", "")
        .replace(" ", "_")
        .replace(":", "_")
    )


def derive_label(rel_path: str) -> str:
    path = Path(rel_path)
    parts = path.parts
    if len(parts) >= 3:
        return f"{parts[-3]} / {parts[-2]}"
    if len(parts) >= 2:
        return f"{parts[-2]} / {parts[-1]}"
    return rel_path


def html_escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def build_html(
    *,
    title: str,
    doc_rows: dict[str, dict[str, str]],
    rows: list[dict[str, Any]],
    best_by_metric: dict[str, set[str]],
) -> str:
    directions = []
    for column in METRIC_COLUMNS:
        doc = doc_rows.get(column.source_label)
        if doc is None:
            source_text = column.note or "Direction defined by portal config."
        else:
            source_text = f"{doc['direction']}  主字段: {doc['field']}"
        directions.append(
            f"""
            <div class="metric-chip">
              <div class="metric-chip-name">{html_escape(column.label)}</div>
              <div class="metric-chip-body">{html_escape(source_text)}</div>
            </div>
            """
        )

    header_cells = ['<th class="sticky-col">Method</th>']
    for column in METRIC_COLUMNS:
        header_cells.append(f"<th>{html_escape(column.label)}</th>")
    header_html = "".join(header_cells)

    body_rows = []
    gallery_cards = []

    for row in rows:
        method_id = row["method_id"]
        table_cells = [f'<td class="sticky-col"><div class="method-name">{html_escape(row["label"])}</div><div class="method-path">{html_escape(row["rel_path"])}</div></td>']
        badges = []
        metric_lines = []

        for column in METRIC_COLUMNS:
            value = row["metrics"].get(column.key)
            is_best = method_id in best_by_metric.get(column.key, set()) and value is not None
            best_class = " best" if is_best else ""
            cell = f'<td class="metric-cell{best_class}">{html_escape(format_value(value, column.precision))}</td>'
            table_cells.append(cell)
            if value is not None:
                metric_lines.append(
                    f'<div class="metric-line{best_class}"><span>{html_escape(column.label)}</span><strong>{html_escape(format_value(value, column.precision))}</strong></div>'
                )
            if is_best:
                badges.append(f'<span class="badge">{html_escape(column.label)} best</span>')

        body_rows.append(f"<tr>{''.join(table_cells)}</tr>")

        metrics_present = row["metrics_present"]
        gallery_cards.append(
            f"""
            <article class="card" id="{html_escape(method_id)}">
              <div class="card-head">
                <div>
                  <h3>{html_escape(row["label"])}</h3>
                  <p>{html_escape(row["rel_path"])}</p>
                </div>
                <div class="badges">{''.join(badges) if badges else '<span class="badge muted">No best-score badge</span>'}</div>
              </div>
              <video controls preload="metadata" playsinline src="{html_escape(row['page_video_src'])}"></video>
              <div class="meta-grid">
                <div><span>Method</span><strong>{html_escape(row['method_name'])}</strong></div>
                <div><span>Metrics Present</span><strong>{html_escape(', '.join(metrics_present) if metrics_present else 'none')}</strong></div>
              </div>
              <div class="metric-list">
                {''.join(metric_lines) if metric_lines else '<div class="metric-line muted"><span>No metrics</span><strong>NA</strong></div>'}
              </div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_escape(title)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffdf8;
      --ink: #1f1a17;
      --muted: #6f665f;
      --line: #d7c9b6;
      --accent: #0f766e;
      --accent-soft: #d9f4ee;
      --warn: #b45309;
      --shadow: 0 18px 50px rgba(71, 52, 32, 0.12);
      --best: #fff4b3;
      --best-border: #e0a100;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 26rem),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.10), transparent 22rem),
        linear-gradient(180deg, #f8f3eb 0%, var(--bg) 100%);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
    }}
    .shell {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    .hero {{
      background: rgba(255, 253, 248, 0.85);
      border: 1px solid rgba(215, 201, 182, 0.85);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      backdrop-filter: blur(14px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(30px, 4vw, 52px);
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .sub {{
      max-width: 980px;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.65;
      margin: 0;
    }}
    .metric-direction-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }}
    .metric-chip {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      min-height: 92px;
    }}
    .metric-chip-name {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .metric-chip-body {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }}
    .panel {{
      margin-top: 24px;
      background: rgba(255, 253, 248, 0.9);
      border: 1px solid rgba(215, 201, 182, 0.85);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-head {{
      padding: 20px 24px 10px;
      border-bottom: 1px solid rgba(215, 201, 182, 0.7);
    }}
    .panel-head h2 {{
      margin: 0;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: 24px;
    }}
    .panel-head p {{
      margin: 8px 0 0;
      color: var(--muted);
    }}
    .table-wrap {{
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      min-width: 1180px;
      font-size: 14px;
    }}
    th, td {{
      padding: 14px 12px;
      border-bottom: 1px solid rgba(215, 201, 182, 0.6);
      border-right: 1px solid rgba(215, 201, 182, 0.4);
      vertical-align: top;
      background: rgba(255, 253, 248, 0.75);
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 3;
      background: #efe5d6;
      text-align: left;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .sticky-col {{
      position: sticky;
      left: 0;
      z-index: 2;
      background: #faf4ea;
      min-width: 360px;
    }}
    th.sticky-col {{
      z-index: 4;
      background: #e8dac6;
    }}
    .method-name {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .method-path {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-all;
    }}
    .metric-cell {{
      font-variant-numeric: tabular-nums;
      font-weight: 600;
    }}
    .best {{
      background: linear-gradient(180deg, #fff9d5 0%, var(--best) 100%);
      box-shadow: inset 0 0 0 2px var(--best-border);
    }}
    .gallery {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
      padding: 20px;
    }}
    .card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      overflow: hidden;
      box-shadow: 0 12px 30px rgba(46, 35, 24, 0.08);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 18px 12px;
      align-items: flex-start;
    }}
    .card-head h3 {{
      margin: 0;
      font-size: 18px;
    }}
    .card-head p {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
      word-break: break-all;
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
      white-space: nowrap;
    }}
    .badge.muted {{
      color: var(--warn);
      background: #fff1dd;
    }}
    video {{
      display: block;
      width: 100%;
      background: #181411;
      aspect-ratio: 16 / 9;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 16px 18px 0;
    }}
    .meta-grid div {{
      background: #f8f2e8;
      border: 1px solid rgba(215, 201, 182, 0.7);
      border-radius: 14px;
      padding: 10px 12px;
    }}
    .meta-grid span {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 5px;
    }}
    .meta-grid strong {{
      display: block;
      font-size: 13px;
      line-height: 1.45;
      word-break: break-word;
    }}
    .metric-list {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      padding: 16px 18px 18px;
    }}
    .metric-line {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      background: #f7efe4;
      border: 1px solid rgba(215, 201, 182, 0.7);
      border-radius: 12px;
      padding: 10px 12px;
      font-size: 13px;
    }}
    .metric-line span {{
      color: var(--muted);
    }}
    .metric-line strong {{
      font-variant-numeric: tabular-nums;
      font-size: 14px;
    }}
    .metric-line.best {{
      border-color: var(--best-border);
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .shell {{ padding: 18px 12px 28px; }}
      .hero {{ padding: 20px; border-radius: 22px; }}
      .sticky-col {{ min-width: 280px; }}
      .metric-list, .meta-grid {{ grid-template-columns: 1fr; }}
      .card-head {{ flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>{html_escape(title)}</h1>
      <p class="sub">This page compares one shared sample across multiple methods under <code>/data/gaoya/AAA_test_video/0623/test/v2v</code>. Highlighted cells are the best values among the displayed methods, using the metric direction quick-reference from <code>eval_metrics.md</code>. Physics-IQ Approx is shown as an extra metric with direction inferred from its score semantics.</p>
      <div class="metric-direction-grid">
        {''.join(directions)}
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Scoreboard</h2>
        <p>Rows preserve the order you supplied. Best values are highlighted per metric column when numeric values are available.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>{header_html}</tr>
          </thead>
          <tbody>
            {''.join(body_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <div class="panel-head">
        <h2>Video Gallery</h2>
        <p>Each card includes the local video preview, the method name from json, and the same metric values shown above.</p>
      </div>
      <div class="gallery">
        {''.join(gallery_cards)}
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metrics_doc = args.metrics_doc.expanduser().resolve()
    doc_rows = parse_direction_table(metrics_doc)

    rows: list[dict[str, Any]] = []

    for rel_path in args.video_rel_path:
        clean_rel = rel_path.strip()
        if not clean_rel:
            continue
        rel_no_prefix = clean_rel[2:] if clean_rel.startswith("./") else clean_rel
        video_path = (result_root / rel_no_prefix).resolve()
        json_path = video_path.with_suffix(".json")
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not json_path.is_file():
            raise FileNotFoundError(f"JSON not found: {json_path}")
        payload = load_json(json_path)
        metrics = {column.key: to_float(nested_get(payload, column.field_path)) for column in METRIC_COLUMNS}
        metrics_present = [name for name in ("wmreward", "physics_iq", "videophy2", "phyground", "cosmos_reason1", "pdi", "proxy") if name in payload]
        rows.append(
            {
                "method_id": slugify(rel_no_prefix),
                "label": derive_label(rel_no_prefix),
                "rel_path": rel_no_prefix,
                "page_video_src": f"../../v2v/{rel_no_prefix}",
                "json_path": str(json_path),
                "method_name": payload.get("method") or Path(rel_no_prefix).parent.name,
                "metrics": metrics,
                "metrics_present": metrics_present,
            }
        )

    best_by_metric: dict[str, set[str]] = {}
    for column in METRIC_COLUMNS:
        numeric_pairs = [
            (row["method_id"], row["metrics"][column.key])
            for row in rows
            if row["metrics"][column.key] is not None
        ]
        if not numeric_pairs:
            continue
        values = [value for _, value in numeric_pairs]
        target = min(values) if column.direction == "down" else max(values)
        best_ids = {
            method_id
            for method_id, value in numeric_pairs
            if value is not None and abs(value - target) <= 1e-12
        }
        best_by_metric[column.key] = best_ids

    output_dir.mkdir(parents=True, exist_ok=True)
    html_text = build_html(
        title=args.title,
        doc_rows=doc_rows,
        rows=rows,
        best_by_metric=best_by_metric,
    )
    index_path = output_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    print(index_path)


if __name__ == "__main__":
    main()
