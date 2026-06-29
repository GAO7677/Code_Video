from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/report/v2v")
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json"}


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    higher_is_better: bool


METRICS: tuple[MetricDef, ...] = (
    MetricDef("pdi_score", "PDI", False),
    MetricDef("wmreward_surprise", "WMReward Surprise", False),
    MetricDef("proxy_score", "Proxy Score", True),
    MetricDef("videophy2_score", "VideoPhy2-PC", True),
    MetricDef("phyground_general_avg", "PhyGround", True),
    MetricDef("cosmos_reason1_score", "Cosmos-Reason1", True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read v2v benchmark result jsons, summarize per-method metrics, "
            "and render a static HTML report plus line charts."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return None


def extract_metric_values(payload: dict[str, Any]) -> dict[str, float | None]:
    return {
        "pdi_score": to_float(nested_get(payload, "pdi", "pdi_score")),
        "wmreward_surprise": to_float(nested_get(payload, "wmreward", "surprise")),
        "proxy_score": to_float(nested_get(payload, "proxy", "score")),
        "videophy2_score": to_float(nested_get(payload, "videophy2", "score")),
        "phyground_general_avg": to_float(nested_get(payload, "phyground", "general_avg")),
        "cosmos_reason1_score": to_float(nested_get(payload, "cosmos_reason1", "score")),
    }


def discover_result_jsons(result_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(result_root.rglob("*.json")):
        if path.name in EXCLUDED_JSON_NAMES:
            continue
        if path.name.startswith("eval_summary_"):
            continue
        payload = load_json(path)
        if payload is None:
            continue
        if "input_json" not in payload:
            continue
        paths.append(path)
    return paths


def normalize_method(method: str | None, json_path: Path) -> str:
    if isinstance(method, str) and method.strip():
        return method.strip()
    return json_path.parent.name or "<unknown>"


def split_method_step(method: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"(?P<family>.+?)_step-?(?P<step>\d+)", method)
    if match is None:
        return method, None
    return match.group("family"), int(match.group("step"))


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def fmt_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def fmt_ratio(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0/0"
    return f"{numerator}/{denominator}"


def metric_direction_arrow(metric: MetricDef) -> str:
    return "↑" if metric.higher_is_better else "↓"


def metric_display_label(metric: MetricDef) -> str:
    return f"{metric.label} {metric_direction_arrow(metric)}"


def is_better_metric(candidate: float, incumbent: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return candidate > incumbent
    return candidate < incumbent


def compute_best_metric_values(method_rows: list[dict[str, Any]]) -> dict[str, float]:
    best_values: dict[str, float] = {}
    for metric in METRICS:
        best_value: float | None = None
        for row in method_rows:
            value = row.get(f"{metric.key}_mean")
            if not isinstance(value, (int, float)):
                continue
            value = float(value)
            if best_value is None or is_better_metric(value, best_value, metric.higher_is_better):
                best_value = value
        if best_value is not None:
            best_values[metric.key] = best_value
    return best_values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_methods(result_jsons: list[Path]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for json_path in result_jsons:
        payload = load_json(json_path)
        if payload is None:
            continue
        method = normalize_method(payload.get("method"), json_path)
        family, step = split_method_step(method)
        bucket = buckets.setdefault(
            method,
            {
                "method": method,
                "family": family,
                "step": step,
                "num_cases": 0,
                "json_paths": [],
            },
        )
        bucket["num_cases"] += 1
        bucket["json_paths"].append(str(json_path))
        for metric in METRICS:
            metric_values = bucket.setdefault(f"{metric.key}_values", [])
            value = extract_metric_values(payload)[metric.key]
            if value is not None:
                metric_values.append(value)

    rows: list[dict[str, Any]] = []
    for method in sorted(
        buckets,
        key=lambda item: (
            buckets[item]["family"],
            buckets[item]["step"] is None,
            buckets[item]["step"] if buckets[item]["step"] is not None else 10**12,
            item,
        ),
    ):
        bucket = buckets[method]
        row: dict[str, Any] = {
            "method": bucket["method"],
            "family": bucket["family"],
            "step": bucket["step"],
            "num_cases": bucket["num_cases"],
        }
        for metric in METRICS:
            values = list(bucket.get(f"{metric.key}_values", []))
            row[f"{metric.key}_count"] = len(values)
            row[f"{metric.key}_mean"] = mean_or_none(values)
        rows.append(row)
    return rows


def load_progress_rows(result_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_root.glob("eval_summary_*.json")):
        payload = load_json(path)
        if payload is None:
            continue
        metric_status = payload.get("metric_status")
        if not isinstance(metric_status, dict):
            continue
        rows.append(
            {
                "metric": payload.get("metric", path.stem.replace("eval_summary_", "")),
                "num_cases": payload.get("num_result_jsons"),
                "completed": metric_status.get("completed"),
                "num_success": metric_status.get("num_success"),
                "num_failed": metric_status.get("num_failed"),
                "errors_count": len(payload.get("errors", [])) if isinstance(payload.get("errors"), list) else None,
                "summary_path": str(path),
                "updated_at": path.stat().st_mtime,
            }
        )
    rows.sort(key=lambda row: str(row["metric"]))
    return rows


def infer_complete_case_count(method_rows: list[dict[str, Any]]) -> int:
    counts: list[int] = []
    for row in method_rows:
        num_cases = row.get("num_cases")
        if isinstance(num_cases, int) and num_cases > 0:
            counts.append(num_cases)
    return max(counts) if counts else 0


def render_line_charts(output_dir: Path, method_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_rows: list[dict[str, str]] = []
    complete_case_count = infer_complete_case_count(method_rows)

    for metric in METRICS:
        points = []
        for row in method_rows:
            mean_value = row.get(f"{metric.key}_mean")
            metric_count = row.get(f"{metric.key}_count")
            num_cases = row.get("num_cases")
            if mean_value is None:
                continue
            if not isinstance(metric_count, int) or not isinstance(num_cases, int):
                continue
            if num_cases != complete_case_count:
                continue
            if metric_count != complete_case_count:
                continue
            points.append((str(row["method"]), float(mean_value), num_cases))

        if not points:
            plt.close()
            continue

        plt.figure(figsize=(max(12, len(points) * 0.9), 6))
        xs = list(range(len(points)))
        ys = [item[1] for item in points]
        labels = [item[0] for item in points]
        plt.plot(xs, ys, marker="o", linewidth=2, color="#8c5a2b")
        for x_value, (_, y_value, num_cases) in zip(xs, points, strict=False):
            plt.annotate(
                f"n={num_cases}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
            )

        plt.title(f"{metric_display_label(metric)} by Method (complete runs only)")
        plt.xlabel("Method")
        plt.ylabel(metric.label)
        plt.xticks(xs, labels, rotation=35, ha="right")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        chart_path = charts_dir / f"{metric.key}.png"
        plt.savefig(chart_path, dpi=180)
        plt.close()
        chart_rows.append({"metric": metric_display_label(metric), "path": str(chart_path)})

    return chart_rows


def rel_path(from_dir: Path, target: Path) -> str:
    return html.escape(os.path.relpath(target.resolve(), from_dir.resolve()).replace("\\", "/"))


def render_html(
    result_root: Path,
    output_dir: Path,
    progress_rows: list[dict[str, Any]],
    method_rows: list[dict[str, Any]],
    chart_rows: list[dict[str, str]],
) -> str:
    best_metric_values = compute_best_metric_values(method_rows)
    progress_table_rows = []
    for row in progress_rows:
        completed = row.get("completed") or 0
        num_cases = row.get("num_cases") or 0
        progress_table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('metric', '-')))}</td>"
            f"<td>{completed}</td>"
            f"<td>{html.escape(fmt_ratio(int(completed), int(num_cases) if isinstance(num_cases, int) else 0))}</td>"
            f"<td>{html.escape(str(row.get('num_success', '-')))}</td>"
            f"<td>{html.escape(str(row.get('num_failed', '-')))}</td>"
            f"<td>{html.escape(str(row.get('errors_count', '-')))}</td>"
            "</tr>"
        )

    metric_headers = "".join(
        f"<th>{html.escape(metric_display_label(metric))} Mean</th>"
        f"<th>{html.escape(metric_display_label(metric))} Count</th>"
        for metric in METRICS
    )
    method_table_rows = []
    for row in method_rows:
        metric_cells = []
        for metric in METRICS:
            mean_value = row.get(f"{metric.key}_mean")
            mean_text = html.escape(fmt_metric(mean_value))
            best_value = best_metric_values.get(metric.key)
            is_best = (
                isinstance(mean_value, (int, float))
                and best_value is not None
                and math.isclose(float(mean_value), float(best_value), rel_tol=1e-12, abs_tol=1e-12)
            )
            if is_best:
                mean_text = f"<strong>{mean_text}</strong>"
            metric_cells.append(f"<td>{mean_text}</td>")
            metric_cells.append(f"<td>{html.escape(str(row.get(f'{metric.key}_count', 0)))}</td>")
        method_table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['method']))}</td>"
            f"<td>{html.escape('-' if row['step'] is None else str(row['step']))}</td>"
            f"<td>{html.escape(str(row['num_cases']))}</td>"
            f"{''.join(metric_cells)}"
            "</tr>"
        )

    chart_sections = []
    for row in chart_rows:
        chart_path = Path(row["path"])
        chart_sections.append(
            "<section class='chart-card'>"
            f"<h3>{html.escape(row['metric'])}</h3>"
            f"<img src='{rel_path(output_dir, chart_path)}' alt='{html.escape(row['metric'])}' />"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>V2V Metric Report</title>
  <style>
    :root {{
      --bg: #f4efe7;
      --panel: #fffaf2;
      --line: #d7cbbb;
      --text: #261d17;
      --muted: #6f655d;
      --accent: #8c5a2b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", "Noto Sans", sans-serif;
      background: linear-gradient(180deg, #faf7f1 0%, var(--bg) 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 10px 24px rgba(64, 46, 31, 0.08);
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f7efe4;
      z-index: 1;
    }}
    .table-wrap {{
      overflow: auto;
      max-height: 70vh;
    }}
    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 16px;
    }}
    .chart-card {{
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
    }}
    .chart-card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      border: 1px solid #ece2d5;
      background: white;
    }}
    .mono {{
      font-family: "SFMono-Regular", Consolas, monospace;
      color: var(--accent);
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>V2V Metric Report</h1>
      <p>Result root: <span class="mono">{html.escape(str(result_root))}</span></p>
      <p>Output dir: <span class="mono">{html.escape(str(output_dir))}</span></p>
      <p>Direction notes: WMReward uses the official <span class="mono">surprise ↓</span> convention; Proxy Score is the compatibility score <span class="mono">exp(-error) ↑</span>.</p>
    </section>

    <section class="panel">
      <h2>Metric Progress</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Metric</th>
              <th>Completed</th>
              <th>Progress</th>
              <th>Success</th>
              <th>Failed</th>
              <th>Error Rows</th>
            </tr>
          </thead>
          <tbody>
            {''.join(progress_table_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Method Summary</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Step</th>
              <th>Cases</th>
              {metric_headers}
            </tr>
          </thead>
          <tbody>
            {''.join(method_table_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Line Charts</h2>
      <div class="charts">
        {''.join(chart_sections) if chart_sections else '<p>No method charts available yet.</p>'}
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else DEFAULT_OUTPUT_ROOT.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    result_jsons = discover_result_jsons(result_root)
    method_rows = summarize_methods(result_jsons)
    progress_rows = load_progress_rows(result_root)
    chart_rows = render_line_charts(output_dir, method_rows)

    summary_json_path = output_dir / "method_summary.json"
    summary_csv_path = output_dir / "method_summary.csv"
    progress_csv_path = output_dir / "metric_progress.csv"
    html_path = output_dir / "index.html"

    summary_json_path.write_text(
        json.dumps(
            {
                "result_root": str(result_root),
                "num_result_jsons": len(result_jsons),
                "method_rows": method_rows,
                "progress_rows": progress_rows,
                "chart_rows": chart_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(summary_csv_path, method_rows)
    write_csv(progress_csv_path, progress_rows)
    html_path.write_text(
        render_html(result_root, output_dir, progress_rows, method_rows, chart_rows),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "result_root": str(result_root),
            "num_result_jsons": len(result_jsons),
            "output_dir": str(output_dir),
            "html_report": str(html_path),
            "method_summary_csv": str(summary_csv_path),
            "metric_progress_csv": str(progress_csv_path),
            "num_charts": len(chart_rows),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
