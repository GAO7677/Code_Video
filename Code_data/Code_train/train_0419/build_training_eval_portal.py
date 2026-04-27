#!/usr/bin/env python3
"""Build a local HTML portal for comparing training-time benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora"
)
PORTAL_SUBDIR = Path("visualization/training_eval_portal")
COMPARE_PORTAL_SUBDIR = Path("visualization/benchmark_compare_portal")
BENCHMARK_SUBDIR = Path("test/fixed24_generation")
BENCHMARK_RUNTIME_SUBDIR = Path("test/_benchmark_runtime/fixed24_generation")
VALIDATION_RUNTIME_SUBDIR = Path("test/_benchmark_runtime/validation100_vbench")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local HTML portal for training-time benchmark and validation outputs."
    )
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark_root", type=Path, default=None)
    parser.add_argument(
        "--compare_model_names",
        type=str,
        default="base-ti2v-5b,step-008000",
        help="Comma-separated model names under benchmark_root/generated_videos/.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def relative_to_root(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def web_path(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.replace(os.sep, "/").lstrip("/")
    return f"/{normalized}"


def ensure_symlink(target: Path, link_path: Path) -> str | None:
    if not target.exists():
        return None
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return link_path.name
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path.name


def gather_benchmark_samples(output_root: Path, portal_dir: Path) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    benchmark_root = output_root / BENCHMARK_SUBDIR
    runtime_root = output_root / BENCHMARK_RUNTIME_SUBDIR
    asset_root = portal_dir / "assets" / "samples"

    step_dirs = sorted(
        path for path in benchmark_root.glob("step-*") if path.is_dir()
    )
    steps = [path.name for path in step_dirs]

    per_sample: dict[str, dict[str, Any]] = {}
    benchmark_step_summaries: list[dict[str, Any]] = []

    for step_dir in step_dirs:
        step_name = step_dir.name
        runtime_summary_path = runtime_root / step_name / "summary.json"
        if runtime_summary_path.is_file():
            payload = read_json(runtime_summary_path)
            benchmark_step_summaries.append(
                {
                    "step": step_name,
                    "summary": payload.get("summary", {}),
                }
            )

        for json_path in sorted(step_dir.glob("*.json")):
            payload = read_json(json_path)
            sample_key = f"{payload.get('dataset', 'unknown')}::{payload.get('sample_id', json_path.stem)}"
            sample_entry = per_sample.setdefault(
                sample_key,
                {
                    "dataset": payload.get("dataset", "unknown"),
                    "sample_id": payload.get("sample_id", json_path.stem),
                    "scenario": payload.get("scenario"),
                    "caption": payload.get("caption", ""),
                    "benchmark_steps": {},
                    "paths": {},
                    "generation_params": {},
                },
            )
            sample_entry["generation_params"] = payload.get("generation_params", {})
            sample_entry["paths"] = payload.get("paths", {})
            video_path = Path(payload["paths"]["output_video_path"])
            sample_entry["benchmark_steps"][step_name] = relative_to_root(output_root, video_path)

    sample_cards: list[dict[str, Any]] = []
    for sample in sorted(
        per_sample.values(),
        key=lambda item: (str(item["dataset"]).lower(), str(item["sample_id"]).lower()),
    ):
        dataset_tag = str(sample["dataset"]).replace("/", "_")
        sample_tag = str(sample["sample_id"]).replace("/", "_")
        sample_asset_dir = asset_root / f"{dataset_tag}__{sample_tag}"
        source_paths = sample.get("paths", {})
        linked_assets = {}
        for source_key, asset_name in (
            ("context_video_path", "context_video.mp4"),
            ("future_gt_video_path", "future_gt_video.mp4"),
            ("full_video_path", "full_video.mp4"),
            ("first_frame_path", "first_frame.png"),
            ("meta_json_path", "meta.json"),
        ):
            raw_path = source_paths.get(source_key)
            if not raw_path:
                continue
            linked_name = ensure_symlink(Path(raw_path), sample_asset_dir / asset_name)
            if linked_name:
                linked_assets[source_key] = relative_to_root(
                    output_root,
                    sample_asset_dir / linked_name,
                )
        sample_cards.append(
            {
                "dataset": sample["dataset"],
                "sample_id": sample["sample_id"],
                "scenario": sample.get("scenario"),
                "caption": sample.get("caption", ""),
                "generation_params": sample.get("generation_params", {}),
                "benchmark_steps": sample["benchmark_steps"],
                "linked_assets": linked_assets,
            }
        )

    return steps, sample_cards, benchmark_step_summaries


def gather_validation_data(output_root: Path) -> list[dict[str, Any]]:
    validation_root = output_root / VALIDATION_RUNTIME_SUBDIR
    steps: list[dict[str, Any]] = []
    for step_dir in sorted(path for path in validation_root.glob("step-*") if path.is_dir()):
        summary_path = step_dir / "summary.json"
        curve_path = step_dir / "context_curve.csv"
        if not summary_path.is_file():
            continue
        payload = read_json(summary_path)
        curve_rows = read_csv_rows(curve_path) if curve_path.is_file() else []
        steps.append(
            {
                "step": step_dir.name,
                "summary": payload.get("summary", {}),
                "curve_rows": curve_rows,
            }
        )
    return steps


def metric_prefers_lower(metric_name: str) -> bool:
    lowered = metric_name.lower()
    return "lpips" in lowered or lowered.endswith("_mse")


def parse_step_number(step_name: str) -> int:
    digits = "".join(ch for ch in step_name if ch.isdigit())
    return int(digits) if digits else 0


def format_metric_value(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.4f}"
    return f"{value:.5f}"


def build_line_chart_svg(points: list[tuple[float, float]], metric_name: str) -> str:
    width = 360
    height = 180
    padding_left = 42
    padding_right = 16
    padding_top = 18
    padding_bottom = 28
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        delta = 1.0 if math.isclose(min_y, 0.0) else abs(min_y) * 0.1
        min_y -= delta
        max_y += delta

    def x_to_svg(value: float) -> float:
        ratio = (value - min_x) / (max_x - min_x)
        return padding_left + ratio * plot_width

    def y_to_svg(value: float) -> float:
        ratio = (value - min_y) / (max_y - min_y)
        return padding_top + (1.0 - ratio) * plot_height

    polyline_points = " ".join(
        f"{x_to_svg(x):.2f},{y_to_svg(y):.2f}" for x, y in points
    )
    circles = "".join(
        f"<circle cx='{x_to_svg(x):.2f}' cy='{y_to_svg(y):.2f}' r='3.5'></circle>"
        for x, y in points
    )
    x_labels = "".join(
        f"<text x='{x_to_svg(x):.2f}' y='{height - 8}' text-anchor='middle'>{int(x)}</text>"
        for x in xs
    )
    y_ticks = []
    for tick_id in range(4):
        value = min_y + (max_y - min_y) * tick_id / 3
        y = y_to_svg(value)
        y_ticks.append(
            f"<line x1='{padding_left}' y1='{y:.2f}' x2='{width - padding_right}' y2='{y:.2f}'></line>"
            f"<text x='{padding_left - 8}' y='{y + 4:.2f}' text-anchor='end'>{html.escape(format_metric_value(value))}</text>"
        )
    lower_better = metric_prefers_lower(metric_name)
    direction_badge = "lower better" if lower_better else "higher better"
    best_index = min(range(len(points)), key=lambda idx: ys[idx]) if lower_better else max(range(len(points)), key=lambda idx: ys[idx])
    best_point = points[best_index]

    return f"""
    <div class="chart-card">
      <div class="chart-card-head">
        <h4>{html.escape(metric_name)}</h4>
        <span class="chart-direction">{direction_badge}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" class="metric-chart" role="img" aria-label="{html.escape(metric_name)} line chart">
        <rect x="0" y="0" width="{width}" height="{height}" rx="12" ry="12"></rect>
        <g class="grid">{''.join(y_ticks)}</g>
        <line class="axis" x1="{padding_left}" y1="{height - padding_bottom}" x2="{width - padding_right}" y2="{height - padding_bottom}"></line>
        <line class="axis" x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}"></line>
        <polyline class="series" points="{polyline_points}"></polyline>
        <g class="points">{circles}</g>
        <g class="xlabels">{x_labels}</g>
      </svg>
      <div class="chart-foot">
        <span>best @ context {int(best_point[0])}</span>
        <span>{format_metric_value(best_point[1])}</span>
      </div>
    </div>
    """


def build_multi_series_chart_svg(
    series_map: dict[str, list[tuple[float, float]]],
    metric_name: str,
    x_label: str,
) -> str:
    width = 420
    height = 220
    padding_left = 48
    padding_right = 18
    padding_top = 18
    padding_bottom = 34
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    non_empty_series = {
        name: points for name, points in series_map.items() if len(points) >= 2
    }
    if not non_empty_series:
        return ""

    xs = [x for points in non_empty_series.values() for x, _ in points]
    ys = [y for points in non_empty_series.values() for _, y in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_x, max_x):
        max_x = min_x + 1.0
    if math.isclose(min_y, max_y):
        delta = 1.0 if math.isclose(min_y, 0.0) else abs(min_y) * 0.1
        min_y -= delta
        max_y += delta

    def x_to_svg(value: float) -> float:
        ratio = (value - min_x) / (max_x - min_x)
        return padding_left + ratio * plot_width

    def y_to_svg(value: float) -> float:
        ratio = (value - min_y) / (max_y - min_y)
        return padding_top + (1.0 - ratio) * plot_height

    palette = [
        "#b9512d",
        "#1f6f8b",
        "#3f7d20",
        "#8c4f9f",
        "#c47f00",
        "#cc5a71",
        "#2e8b57",
        "#5d5fef",
    ]
    sorted_series = sorted(
        non_empty_series.items(),
        key=lambda item: float(item[0]) if item[0].replace(".", "", 1).isdigit() else item[0],
    )
    paths = []
    legends = []
    point_groups = []
    for idx, (series_name, points) in enumerate(sorted_series):
        color = palette[idx % len(palette)]
        polyline_points = " ".join(
            f"{x_to_svg(x):.2f},{y_to_svg(y):.2f}" for x, y in points
        )
        circles = "".join(
            f"<circle cx='{x_to_svg(x):.2f}' cy='{y_to_svg(y):.2f}' r='3.2' fill='{color}'></circle>"
            for x, y in points
        )
        paths.append(
            f"<polyline class='series' style='stroke: {color};' points='{polyline_points}'></polyline>"
        )
        point_groups.append(f"<g class='points'>{circles}</g>")
        legends.append(
            "<span class='chart-legend-item'>"
            f"<span class='chart-swatch' style='background:{color};'></span>"
            f"context {html.escape(series_name)}"
            "</span>"
        )

    x_values = sorted(set(xs))
    x_labels = "".join(
        f"<text x='{x_to_svg(x):.2f}' y='{height - 10}' text-anchor='middle'>{int(x)}</text>"
        for x in x_values
    )
    y_ticks = []
    for tick_id in range(4):
        value = min_y + (max_y - min_y) * tick_id / 3
        y = y_to_svg(value)
        y_ticks.append(
            f"<line x1='{padding_left}' y1='{y:.2f}' x2='{width - padding_right}' y2='{y:.2f}'></line>"
            f"<text x='{padding_left - 8}' y='{y + 4:.2f}' text-anchor='end'>{html.escape(format_metric_value(value))}</text>"
        )

    lower_better = metric_prefers_lower(metric_name)
    direction_badge = "lower better" if lower_better else "higher better"
    return f"""
    <div class="chart-card">
      <div class="chart-card-head">
        <h4>{html.escape(metric_name)}</h4>
        <span class="chart-direction">{direction_badge}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" class="metric-chart" role="img" aria-label="{html.escape(metric_name)} by training step chart">
        <rect x="0" y="0" width="{width}" height="{height}" rx="12" ry="12"></rect>
        <g class="grid">{''.join(y_ticks)}</g>
        <line class="axis" x1="{padding_left}" y1="{height - padding_bottom}" x2="{width - padding_right}" y2="{height - padding_bottom}"></line>
        <line class="axis" x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}"></line>
        {''.join(paths)}
        {''.join(point_groups)}
        <g class="xlabels">{x_labels}</g>
      </svg>
      <div class="chart-foot">
        <span>{html.escape(x_label)}</span>
        <span>{len(sorted_series)} context curves</span>
      </div>
      <div class="chart-legend">{''.join(legends)}</div>
    </div>
    """


def collect_validation_step_trends(
    validation_steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not validation_steps:
        return [], []

    metric_context_series: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    best_context_series: list[tuple[float, float]] = []
    best_aggregate_series: list[tuple[float, float]] = []
    context_values: set[str] = set()

    for item in validation_steps:
        step_value = float(parse_step_number(item["step"]))
        summary = item.get("summary", {})
        best_context = summary.get("best_aggregate_context_frames")
        best_aggregate = summary.get("best_aggregate_mean")
        if best_context not in (None, ""):
            try:
                best_context_series.append((step_value, float(best_context)))
            except (TypeError, ValueError):
                pass
        if best_aggregate not in (None, ""):
            try:
                best_aggregate_series.append((step_value, float(best_aggregate)))
            except (TypeError, ValueError):
                pass

        for row in item.get("curve_rows", []):
            context_value = row.get("context_frames")
            if context_value in (None, ""):
                continue
            context_values.add(context_value)
            for metric_name, raw_value in row.items():
                if metric_name in {"context_frames", "future_pair_count"}:
                    continue
                if raw_value in (None, ""):
                    continue
                try:
                    numeric_value = float(raw_value)
                except ValueError:
                    continue
                metric_context_series[metric_name][context_value].append((step_value, numeric_value))

    summary_cards = []
    if len(best_aggregate_series) >= 2:
        summary_cards.append(
            {
                "metric_name": "best_aggregate_mean",
                "series_map": {"best": best_aggregate_series},
            }
        )
    if len(best_context_series) >= 2:
        summary_cards.append(
            {
                "metric_name": "best_aggregate_context_frames",
                "series_map": {"best": best_context_series},
            }
        )

    trend_cards = summary_cards + [
        {
            "metric_name": metric_name,
            "series_map": dict(context_map),
        }
        for metric_name, context_map in sorted(metric_context_series.items())
    ]
    return trend_cards, sorted(
        context_values,
        key=lambda value: float(value) if value.replace(".", "", 1).isdigit() else value,
    )


def render_overview_table(benchmark_step_summaries: list[dict[str, Any]]) -> str:
    rows = []
    for item in benchmark_step_summaries:
        summary = item.get("summary", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(item['step'])}</td>"
            f"<td>{summary.get('num_generated', '-')}</td>"
            f"<td>{summary.get('num_failed', '-')}</td>"
            f"<td>{summary.get('success_rate', '-')}</td>"
            f"<td>{summary.get('dataset_cases_mvp-lab-openvidhd-0.4m-720p-48fps', '-')}</td>"
            f"<td>{summary.get('dataset_cases_kubric_tfds_movi-d', '-')}</td>"
            f"<td>{summary.get('dataset_cases_version_1_genesis_rigid_data_all_cases', '-')}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='7'>No benchmark summaries found.</td></tr>"
    return (
        "<table class='overview-table'>"
        "<thead><tr>"
        "<th>Step</th><th>Generated</th><th>Failed</th><th>Success</th>"
        "<th>OpenVid</th><th>MOVI-D</th><th>Genesis</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_validation_section(validation_steps: list[dict[str, Any]]) -> str:
    if not validation_steps:
        return "<p class='empty'>No validation summaries found.</p>"

    blocks = []
    for item in validation_steps:
        summary = item.get("summary", {})
        curve_rows = item.get("curve_rows", [])
        headline = (
            f"best aggregate context = {summary.get('best_aggregate_context_frames', '-')}, "
            f"best aggregate mean = {summary.get('best_aggregate_mean', '-')}"
        )
        chart_grid = "<p class='empty'>No numeric metrics available for charts.</p>"
        if curve_rows:
            metric_names = list(curve_rows[0].keys())
            table_header = "".join(f"<th>{html.escape(name)}</th>" for name in metric_names)
            table_rows = []
            for row in curve_rows:
                cells = "".join(
                    f"<td>{html.escape(str(row.get(name, '')))}</td>"
                    for name in metric_names
                )
                table_rows.append(f"<tr>{cells}</tr>")
            curve_table = (
                "<div class='curve-table-wrap'><table class='curve-table'>"
                f"<thead><tr>{table_header}</tr></thead>"
                f"<tbody>{''.join(table_rows)}</tbody></table></div>"
            )
            chart_metric_names = [
                name
                for name in metric_names
                if name not in {"context_frames", "future_pair_count"}
            ]
            chart_cards = []
            for metric_name in chart_metric_names:
                points = []
                for row in curve_rows:
                    x_raw = row.get("context_frames")
                    y_raw = row.get(metric_name)
                    if x_raw in (None, "") or y_raw in (None, ""):
                        continue
                    try:
                        points.append((float(x_raw), float(y_raw)))
                    except ValueError:
                        continue
                if points:
                    chart_cards.append(build_line_chart_svg(points, metric_name))
            if chart_cards:
                chart_grid = f"<div class='chart-grid'>{''.join(chart_cards)}</div>"
        else:
            curve_table = "<p class='empty'>No context_curve.csv found.</p>"
        blocks.append(
            "<section class='validation-card'>"
            f"<h3>{html.escape(item['step'])}</h3>"
            f"<p class='validation-headline'>{html.escape(headline)}</p>"
            f"{chart_grid}"
            f"{curve_table}"
            "</section>"
        )
    return "\n".join(blocks)


def render_validation_step_trends(validation_steps: list[dict[str, Any]]) -> str:
    trend_cards, context_values = collect_validation_step_trends(validation_steps)
    if not trend_cards:
        return "<p class='empty'>No multi-step validation trends found.</p>"

    context_badges = "".join(
        f"<span class='step-badge'>context {html.escape(value)}</span>"
        for value in context_values
    )
    chart_cards = []
    for item in trend_cards:
        chart_html = build_multi_series_chart_svg(
            item["series_map"],
            item["metric_name"],
            "x-axis: training step",
        )
        if chart_html:
            chart_cards.append(chart_html)
    if not chart_cards:
        return "<p class='empty'>Validation trends did not contain plottable metrics.</p>"

    return (
        "<div class='trend-head'>"
        "<p class='validation-headline'>Each chart uses training step as the x-axis. "
        "Context-specific lines come from validation context curves.</p>"
        f"<div class='step-badges'>{context_badges}</div>"
        "</div>"
        f"<div class='chart-grid chart-grid-wide'>{''.join(chart_cards)}</div>"
    )


def render_reference_column(title: str, path: str | None, is_image: bool = False) -> str:
    if not path:
        return (
            "<div class='video-slot reference-slot'>"
            f"<div class='slot-title'>{html.escape(title)}</div>"
            "<div class='missing'>Missing</div>"
            "</div>"
        )
    resolved_path = web_path(path)
    if is_image:
        media_html = f"<img src='{html.escape(resolved_path)}' loading='lazy' alt='{html.escape(title)}' />"
    else:
        media_html = (
            f"<video controls preload='none' muted playsinline>"
            f"<source src='{html.escape(resolved_path)}' type='video/mp4'>"
            "</video>"
        )
    return (
        "<div class='video-slot reference-slot'>"
        f"<div class='slot-title'>{html.escape(title)}</div>"
        f"{media_html}"
        "</div>"
    )


def parse_model_names(raw_value: str) -> list[str]:
    names = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not names:
        raise ValueError("compare_model_names must contain at least one model name.")
    return names


def safe_sample_key(dataset: Any, sample_id: Any) -> str:
    return f"{dataset}::{sample_id}"


COMPARE_SCOPE_LABELS = {
    "overall": "Overall",
    "kubric_tfds_movi-d": "MOVI-D",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": "OpenVid",
    "physics-iq-benchmark": "Physics-IQ",
    "vLAR-PhysInOne": "vLAR",
    "version_1_genesis_rigid_data_all_cases": "Genesis",
}


def scope_display_name(scope: str) -> str:
    return COMPARE_SCOPE_LABELS.get(scope, scope)


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_categorical_line_chart_svg(
    categories: list[str],
    series_map: dict[str, list[float | None]],
    metric_name: str,
    category_labels: list[str] | None = None,
) -> str:
    non_empty_series = {}
    for series_name, values in series_map.items():
        if any(value is not None for value in values):
            non_empty_series[series_name] = values
    if not non_empty_series or not categories:
        return ""

    display_labels = category_labels or categories
    width = max(420, 92 * len(categories))
    height = 240
    padding_left = 52
    padding_right = 20
    padding_top = 18
    padding_bottom = 56
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom

    ys = [
        float(value)
        for values in non_empty_series.values()
        for value in values
        if value is not None
    ]
    if not ys:
        return ""
    min_y = min(ys)
    max_y = max(ys)
    if math.isclose(min_y, max_y):
        delta = 1.0 if math.isclose(min_y, 0.0) else abs(min_y) * 0.1
        min_y -= delta
        max_y += delta

    if len(categories) == 1:
        x_positions = [padding_left + plot_width / 2]
    else:
        x_positions = [
            padding_left + idx * plot_width / (len(categories) - 1)
            for idx in range(len(categories))
        ]

    def y_to_svg(value: float) -> float:
        ratio = (value - min_y) / (max_y - min_y)
        return padding_top + (1.0 - ratio) * plot_height

    palette = [
        "#b9512d",
        "#1f6f8b",
        "#3f7d20",
        "#8c4f9f",
        "#c47f00",
        "#cc5a71",
    ]
    paths = []
    point_groups = []
    legends = []
    for idx, (series_name, values) in enumerate(non_empty_series.items()):
        color = palette[idx % len(palette)]
        point_pairs = [
            (x_positions[pos], float(value))
            for pos, value in enumerate(values)
            if value is not None
        ]
        if not point_pairs:
            continue
        polyline_points = " ".join(
            f"{x:.2f},{y_to_svg(y):.2f}" for x, y in point_pairs
        )
        circles = "".join(
            f"<circle cx='{x:.2f}' cy='{y_to_svg(y):.2f}' r='3.4' fill='{color}'></circle>"
            for x, y in point_pairs
        )
        paths.append(
            f"<polyline class='series' style='stroke: {color};' points='{polyline_points}'></polyline>"
        )
        point_groups.append(f"<g class='points'>{circles}</g>")
        legends.append(
            "<span class='chart-legend-item'>"
            f"<span class='chart-swatch' style='background:{color};'></span>"
            f"{html.escape(series_name)}"
            "</span>"
        )

    x_labels = "".join(
        f"<text x='{x:.2f}' y='{height - 10}' text-anchor='middle'>{html.escape(label)}</text>"
        for x, label in zip(x_positions, display_labels)
    )
    y_ticks = []
    for tick_id in range(4):
        value = min_y + (max_y - min_y) * tick_id / 3
        y = y_to_svg(value)
        y_ticks.append(
            f"<line x1='{padding_left}' y1='{y:.2f}' x2='{width - padding_right}' y2='{y:.2f}'></line>"
            f"<text x='{padding_left - 8}' y='{y + 4:.2f}' text-anchor='end'>{html.escape(format_metric_value(value))}</text>"
        )

    lower_better = metric_prefers_lower(metric_name)
    direction_badge = "lower better" if lower_better else "higher better"
    return f"""
    <div class="chart-card chart-card-wide">
      <div class="chart-card-head">
        <h4>{html.escape(metric_name)}</h4>
        <span class="chart-direction">{direction_badge}</span>
      </div>
      <svg viewBox="0 0 {width} {height}" class="metric-chart" role="img" aria-label="{html.escape(metric_name)} categorical comparison chart">
        <rect x="0" y="0" width="{width}" height="{height}" rx="12" ry="12"></rect>
        <g class="grid">{''.join(y_ticks)}</g>
        <line class="axis" x1="{padding_left}" y1="{height - padding_bottom}" x2="{width - padding_right}" y2="{height - padding_bottom}"></line>
        <line class="axis" x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{height - padding_bottom}"></line>
        {''.join(paths)}
        {''.join(point_groups)}
        <g class="xlabels">{x_labels}</g>
      </svg>
      <div class="chart-foot">
        <span>x-axis: dataset scope</span>
        <span>{len(categories)} scopes</span>
      </div>
      <div class="chart-legend">{''.join(legends)}</div>
    </div>
    """


def render_compare_metric_charts(
    rows: list[dict[str, str]],
    model_names: list[str] | None = None,
) -> str:
    if not rows:
        return "<p class='empty'>Comparison charts are not ready yet.</p>"

    categories = [str(row.get("scope", "")) for row in rows]
    category_labels = [scope_display_name(category) for category in categories]
    chart_model_names = model_names or ["base-ti2v-5b", "step-008000"]
    base_name = chart_model_names[0] if chart_model_names else "base"
    ft_name = chart_model_names[1] if len(chart_model_names) > 1 else "finetuned"
    metrics = [
        ("future_psnr", "PSNR"),
        ("future_ssim", "SSIM"),
        ("future_lpips", "LPIPS"),
        ("future_dino", "DINO"),
    ]

    cards = []
    for metric_key, metric_title in metrics:
        base_key = f"base_{metric_key}"
        ft_key = f"ft_{metric_key}"
        delta_key = f"delta_{metric_key}"
        compare_chart = build_categorical_line_chart_svg(
            categories,
            {
                base_name: [parse_optional_float(row.get(base_key)) for row in rows],
                ft_name: [parse_optional_float(row.get(ft_key)) for row in rows],
            },
            metric_title,
            category_labels=category_labels,
        )
        delta_chart = build_categorical_line_chart_svg(
            categories,
            {
                "delta (ft - base)": [
                    parse_optional_float(row.get(delta_key)) for row in rows
                ],
            },
            f"{metric_title} Delta",
            category_labels=category_labels,
        )
        if compare_chart:
            cards.append(compare_chart)
        if delta_chart:
            cards.append(delta_chart)

    if not cards:
        return "<p class='empty'>Comparison charts are not ready yet.</p>"
    return f"<div class='chart-grid chart-grid-wide'>{''.join(cards)}</div>"


def render_compare_analysis(compare_summary: dict[str, Any] | None) -> str:
    if not isinstance(compare_summary, dict):
        return "<p class='empty'>Comparison summary is not ready yet.</p>"

    rows = compare_summary.get("rows", [])
    if not rows:
        return "<p class='empty'>Comparison summary is not ready yet.</p>"

    overall = next((row for row in rows if row.get("scope") == "overall"), rows[0])
    dataset_rows = [row for row in rows if row.get("scope") != "overall"]
    metric_specs = [
        ("delta_future_psnr", "PSNR", False),
        ("delta_future_ssim", "SSIM", False),
        ("delta_future_lpips", "LPIPS", True),
        ("delta_future_dino", "DINO", False),
    ]

    headline_parts = []
    for key, label, lower_better in metric_specs:
        value = parse_optional_float(overall.get(key))
        if value is None:
            continue
        direction = "improved" if (value < 0 if lower_better else value > 0) else "dropped"
        headline_parts.append(f"{label} {direction} {format_metric_value(value)}")
    headline = "; ".join(headline_parts) if headline_parts else "No aggregate delta metrics found."

    bullets = []
    for key, label, lower_better in metric_specs:
        scored_rows = []
        for row in dataset_rows:
            value = parse_optional_float(row.get(key))
            if value is None:
                continue
            scored_rows.append((float(value), str(row.get("scope", "")), int(row.get("ft_num_success", 0) or 0)))
        if not scored_rows:
            continue
        best_row = min(scored_rows, key=lambda item: item[0]) if lower_better else max(scored_rows, key=lambda item: item[0])
        worst_row = max(scored_rows, key=lambda item: item[0]) if lower_better else min(scored_rows, key=lambda item: item[0])
        bullets.append(
            "<li>"
            f"{html.escape(label)}: best on <strong>{html.escape(scope_display_name(best_row[1]))}</strong> "
            f"({format_metric_value(best_row[0])}, n={best_row[2]}), "
            f"weakest on <strong>{html.escape(scope_display_name(worst_row[1]))}</strong> "
            f"({format_metric_value(worst_row[0])}, n={worst_row[2]})."
            "</li>"
        )

    sample_notes = []
    for row in dataset_rows:
        scope = str(row.get("scope", ""))
        count = int(row.get("ft_num_success", 0) or 0)
        if count <= 4:
            sample_notes.append(f"{scope_display_name(scope)} n={count}")
    note_html = ""
    if sample_notes:
        note_html = (
            "<p class='validation-headline'>Small-sample scopes should be treated cautiously: "
            f"{html.escape(', '.join(sample_notes))}.</p>"
        )

    return (
        f"<p class='validation-headline'>{html.escape(headline)}</p>"
        f"{note_html}"
        "<ul class='analysis-list'>"
        f"{''.join(bullets)}"
        "</ul>"
    )


def gather_compare_samples(
    benchmark_root: Path,
    portal_dir: Path,
    model_names: list[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    generated_root = benchmark_root / "generated_videos"
    runtime_root = benchmark_root / "runtime"
    asset_root = portal_dir / "assets" / "samples"

    per_sample: dict[str, dict[str, Any]] = {}
    model_summaries: list[dict[str, Any]] = []

    for model_name in model_names:
        model_dir = generated_root / model_name
        if not model_dir.is_dir():
            continue

        runtime_summary_path = runtime_root / model_name / "summary.json"
        summary_payload = read_json(runtime_summary_path) if runtime_summary_path.is_file() else {}
        model_summaries.append(
            {
                "model_name": model_name,
                "summary": summary_payload.get("summary", summary_payload),
            }
        )

        for json_path in sorted(model_dir.glob("*.json")):
            payload = read_json(json_path)
            sample_key = safe_sample_key(
                payload.get("dataset", "unknown"),
                payload.get("sample_id", json_path.stem),
            )
            sample_entry = per_sample.setdefault(
                sample_key,
                {
                    "dataset": payload.get("dataset", "unknown"),
                    "sample_id": payload.get("sample_id", json_path.stem),
                    "scenario": payload.get("scenario"),
                    "caption": payload.get("caption", ""),
                    "paths": payload.get("paths", {}),
                    "generation_params": payload.get("generation_params", {}),
                    "models": {},
                },
            )
            sample_entry["paths"] = payload.get("paths", sample_entry.get("paths", {}))
            sample_entry["generation_params"] = payload.get(
                "generation_params", sample_entry.get("generation_params", {})
            )
            sample_entry["models"][model_name] = {
                "status": payload.get("status"),
                "output_video_path": relative_to_root(
                    benchmark_root,
                    Path(payload["paths"]["output_video_path"]),
                )
                if payload.get("paths", {}).get("output_video_path")
                else None,
                "output_json_path": relative_to_root(
                    benchmark_root,
                    Path(payload["paths"]["output_json_path"]),
                )
                if payload.get("paths", {}).get("output_json_path")
                else None,
            }

    sample_cards: list[dict[str, Any]] = []
    for sample in sorted(
        per_sample.values(),
        key=lambda item: (str(item["dataset"]).lower(), str(item["sample_id"]).lower()),
    ):
        dataset_tag = str(sample["dataset"]).replace("/", "_")
        sample_tag = str(sample["sample_id"]).replace("/", "_")
        sample_asset_dir = asset_root / f"{dataset_tag}__{sample_tag}"
        source_paths = sample.get("paths", {})
        linked_assets = {}
        for source_key, asset_name in (
            ("context_video_path", "context_video.mp4"),
            ("future_gt_video_path", "future_gt_video.mp4"),
            ("full_video_path", "full_video.mp4"),
            ("first_frame_path", "first_frame.png"),
            ("meta_json_path", "meta.json"),
        ):
            raw_path = source_paths.get(source_key)
            if not raw_path:
                continue
            linked_name = ensure_symlink(Path(raw_path), sample_asset_dir / asset_name)
            if linked_name:
                linked_assets[source_key] = relative_to_root(
                    benchmark_root,
                    sample_asset_dir / linked_name,
                )
        sample_cards.append(
            {
                "dataset": sample["dataset"],
                "sample_id": sample["sample_id"],
                "scenario": sample.get("scenario"),
                "caption": sample.get("caption", ""),
                "generation_params": sample.get("generation_params", {}),
                "linked_assets": linked_assets,
                "models": sample.get("models", {}),
            }
        )

    return model_names, sample_cards, model_summaries


def load_compare_metrics_table(benchmark_root: Path) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    runtime_root = benchmark_root / "runtime"
    compare_dirs = sorted(path for path in runtime_root.glob("comparison_*") if path.is_dir())
    if not compare_dirs:
        return [], None
    compare_dir = compare_dirs[0]
    csv_path = compare_dir / "comparison_metrics.csv"
    summary_path = compare_dir / "comparison_summary.json"
    rows = read_csv_rows(csv_path) if csv_path.is_file() else []
    summary = read_json(summary_path) if summary_path.is_file() else None
    if summary is not None:
        summary["_compare_dir"] = str(compare_dir)
    return rows, summary


def render_compare_overview_table(model_summaries: list[dict[str, Any]]) -> str:
    rows = []
    for item in model_summaries:
        summary = item.get("summary", {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item['model_name']))}</td>"
            f"<td>{html.escape(str(summary.get('num_generated', '-')))}</td>"
            f"<td>{html.escape(str(summary.get('num_skipped_existing', '-')))}</td>"
            f"<td>{html.escape(str(summary.get('num_failed', '-')))}</td>"
            f"<td>{html.escape(str(summary.get('success_rate', '-')))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='5'>No model summaries found.</td></tr>"
    return (
        "<table class='overview-table'>"
        "<thead><tr>"
        "<th>Model</th><th>Generated</th><th>Skipped</th><th>Failed</th><th>Success</th>"
        "</tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def render_compare_metrics_table(rows: list[dict[str, str]], compare_summary: dict[str, Any] | None) -> str:
    if not rows:
        return "<p class='empty'>Comparison metrics are not ready yet.</p>"

    preferred_columns = [
        "scope",
        "base_num_success",
        "ft_num_success",
        "base_future_psnr",
        "ft_future_psnr",
        "delta_future_psnr",
        "base_future_ssim",
        "ft_future_ssim",
        "delta_future_ssim",
        "base_future_lpips",
        "ft_future_lpips",
        "delta_future_lpips",
        "base_future_dino",
        "ft_future_dino",
        "delta_future_dino",
    ]
    available_columns = [name for name in preferred_columns if name in rows[0]]
    header = "".join(f"<th>{html.escape(name)}</th>" for name in available_columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(name, '')))}</td>" for name in available_columns)
        body_rows.append(f"<tr>{cells}</tr>")
    compare_dir = compare_summary.get("_compare_dir") if isinstance(compare_summary, dict) else None
    note_html = (
        f"<p class='validation-headline'>Metrics loaded from {html.escape(compare_dir)}.</p>"
        if compare_dir
        else ""
    )
    return (
        f"{note_html}"
        "<div class='curve-table-wrap'><table class='curve-table'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def render_compare_sample_cards(model_names: list[str], samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "<p class='empty'>No benchmark samples found.</p>"

    cards = []
    for sample in samples:
        assets = sample.get("linked_assets", {})
        refs = [
            render_reference_column("Context", assets.get("context_video_path")),
            render_reference_column("Future GT", assets.get("future_gt_video_path")),
            render_reference_column("Full Video", assets.get("full_video_path")),
            render_reference_column("First Frame", assets.get("first_frame_path"), is_image=True),
        ]

        generated_columns = []
        for model_name in model_names:
            model_payload = sample.get("models", {}).get(model_name, {})
            video_path = model_payload.get("output_video_path")
            status = model_payload.get("status", "missing")
            if video_path:
                resolved_path = web_path(video_path)
                video_html = (
                    f"<video controls preload='none' muted playsinline>"
                    f"<source src='{html.escape(resolved_path)}' type='video/mp4'>"
                    "</video>"
                )
            else:
                video_html = "<div class='missing'>Missing</div>"
            generated_columns.append(
                "<div class='video-slot generated-slot'>"
                f"<div class='slot-title'>{html.escape(model_name)}</div>"
                f"<div class='slot-subtitle'>status: {html.escape(str(status))}</div>"
                f"{video_html}"
                "</div>"
            )

        caption = html.escape(sample.get("caption", ""))
        scenario = sample.get("scenario")
        scenario_html = (
            f"<div class='meta-row'><span class='meta-key'>Scenario</span>"
            f"<span class='meta-value'>{html.escape(str(scenario))}</span></div>"
            if scenario
            else ""
        )
        requested_frames = sample.get("generation_params", {}).get("requested_output_frames", "-")
        used_context = sample.get("generation_params", {}).get("used_context_frames", "-")
        cards.append(
            "<article class='sample-card' "
            f"data-dataset='{html.escape(str(sample['dataset']))}' "
            f"data-sample-id='{html.escape(str(sample['sample_id']))}' "
            f"data-caption='{caption.lower()}'>"
            "<div class='sample-header'>"
            f"<h3>{html.escape(str(sample['sample_id']))}</h3>"
            f"<span class='dataset-tag'>{html.escape(str(sample['dataset']))}</span>"
            "</div>"
            "<div class='sample-meta'>"
            f"<div class='meta-row'><span class='meta-key'>Context Frames</span><span class='meta-value'>{used_context}</span></div>"
            f"<div class='meta-row'><span class='meta-key'>Output Frames</span><span class='meta-value'>{requested_frames}</span></div>"
            f"{scenario_html}"
            "</div>"
            f"<p class='caption'>{caption}</p>"
            "<div class='reference-grid'>"
            f"{''.join(refs)}"
            "</div>"
            "<div class='generated-grid compare-grid'>"
            f"{''.join(generated_columns)}"
            "</div>"
            "</article>"
        )
    return "\n".join(cards)


def render_sample_cards(steps: list[str], samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "<p class='empty'>No benchmark samples found.</p>"

    cards = []
    for sample in samples:
        assets = sample.get("linked_assets", {})
        refs = [
            render_reference_column("Context", assets.get("context_video_path")),
            render_reference_column("Future GT", assets.get("future_gt_video_path")),
            render_reference_column("Full Video", assets.get("full_video_path")),
            render_reference_column("First Frame", assets.get("first_frame_path"), is_image=True),
        ]
        generated_columns = []
        for step in steps:
            video_path = sample.get("benchmark_steps", {}).get(step)
            if video_path:
                resolved_path = web_path(video_path)
                video_html = (
                    f"<video controls preload='none' muted playsinline>"
                    f"<source src='{html.escape(resolved_path)}' type='video/mp4'>"
                    "</video>"
                )
            else:
                video_html = "<div class='missing'>Missing</div>"
            generated_columns.append(
                "<div class='video-slot generated-slot'>"
                f"<div class='slot-title'>{html.escape(step)}</div>"
                f"{video_html}"
                "</div>"
            )

        caption = html.escape(sample.get("caption", ""))
        scenario = sample.get("scenario")
        scenario_html = (
            f"<div class='meta-row'><span class='meta-key'>Scenario</span>"
            f"<span class='meta-value'>{html.escape(str(scenario))}</span></div>"
            if scenario
            else ""
        )
        requested_frames = sample.get("generation_params", {}).get("requested_output_frames", "-")
        used_context = sample.get("generation_params", {}).get("used_context_frames", "-")
        cards.append(
            "<article class='sample-card' "
            f"data-dataset='{html.escape(str(sample['dataset']))}' "
            f"data-sample-id='{html.escape(str(sample['sample_id']))}' "
            f"data-caption='{caption.lower()}'>"
            "<div class='sample-header'>"
            f"<h3>{html.escape(str(sample['sample_id']))}</h3>"
            f"<span class='dataset-tag'>{html.escape(str(sample['dataset']))}</span>"
            "</div>"
            "<div class='sample-meta'>"
            f"<div class='meta-row'><span class='meta-key'>Context Frames</span><span class='meta-value'>{used_context}</span></div>"
            f"<div class='meta-row'><span class='meta-key'>Output Frames</span><span class='meta-value'>{requested_frames}</span></div>"
            f"{scenario_html}"
            "</div>"
            f"<p class='caption'>{caption}</p>"
            "<div class='reference-grid'>"
            f"{''.join(refs)}"
            "</div>"
            "<div class='generated-grid'>"
            f"{''.join(generated_columns)}"
            "</div>"
            "</article>"
        )
    return "\n".join(cards)


def build_html(
    steps: list[str],
    samples: list[dict[str, Any]],
    benchmark_step_summaries: list[dict[str, Any]],
    validation_steps: list[dict[str, Any]],
) -> str:
    datasets = sorted({str(sample["dataset"]) for sample in samples})
    dataset_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>"
        for name in datasets
    )
    step_badges = "".join(f"<span class='step-badge'>{html.escape(step)}</span>" for step in steps)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Eval Portal</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: #fffdf8;
      --ink: #1f1c18;
      --muted: #6f665d;
      --line: #ddd1bc;
      --accent: #b9512d;
      --accent-soft: #f3ddcc;
      --tag: #efe4d3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: radial-gradient(circle at top left, #fff7ea 0%, var(--bg) 45%, #eadfcf 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 28px 24px 60px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,253,248,0.96), rgba(246,234,214,0.96));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px 28px;
      box-shadow: 0 18px 40px rgba(54, 36, 17, 0.08);
      margin-bottom: 22px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: 34px;
      line-height: 1.08;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 15px;
    }}
    .step-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }}
    .step-badge {{
      background: var(--tag);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .section {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 22px;
      box-shadow: 0 12px 28px rgba(54, 36, 17, 0.06);
      margin-bottom: 22px;
    }}
    .section h2 {{
      font-size: 24px;
      margin-bottom: 14px;
    }}
    .overview-table, .curve-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .overview-table th, .overview-table td,
    .curve-table th, .curve-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .overview-table th, .curve-table th {{
      color: var(--muted);
      font-weight: 600;
      background: rgba(239, 228, 211, 0.55);
      position: sticky;
      top: 0;
    }}
    .validation-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 16px;
    }}
    .validation-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      background: #fffaf2;
    }}
    .validation-card h3 {{
      font-size: 20px;
      margin-bottom: 8px;
    }}
    .validation-headline {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .curve-table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      margin-top: 14px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-bottom: 12px;
    }}
    .chart-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fff;
    }}
    .chart-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .chart-card h4 {{
      font-size: 14px;
      line-height: 1.25;
      word-break: break-word;
    }}
    .chart-direction {{
      font-size: 11px;
      color: var(--muted);
      background: #f6efe5;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      white-space: nowrap;
    }}
    .metric-chart {{
      width: 100%;
      display: block;
    }}
    .metric-chart rect {{
      fill: #fffaf2;
      stroke: #eadfcd;
    }}
    .metric-chart .grid line {{
      stroke: #eadfcd;
      stroke-dasharray: 3 4;
    }}
    .metric-chart .grid text,
    .metric-chart .xlabels text {{
      fill: #7a7168;
      font-size: 10px;
    }}
    .metric-chart .axis {{
      stroke: #a59484;
      stroke-width: 1.25;
    }}
    .metric-chart .series {{
      fill: none;
      stroke: var(--accent);
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .metric-chart .points circle {{
      fill: var(--accent);
      stroke: #fff;
      stroke-width: 1.2;
    }}
    .chart-foot {{
      margin-top: 8px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .chart-legend {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
    }}
    .chart-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      background: #f9f1e6;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
    }}
    .chart-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: none;
    }}
    .chart-grid-wide {{
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    }}
    .trend-head {{
      margin-bottom: 12px;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .filters input, .filters select {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      min-width: 220px;
    }}
    .sample-grid {{
      display: grid;
      gap: 18px;
    }}
    .sample-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 18px;
      box-shadow: 0 14px 24px rgba(54, 36, 17, 0.05);
    }}
    .sample-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }}
    .sample-header h3 {{
      font-size: 22px;
      line-height: 1.18;
      word-break: break-word;
    }}
    .dataset-tag {{
      flex: none;
      background: var(--accent-soft);
      border: 1px solid #e5bc9f;
      color: #7a381f;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
    }}
    .sample-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta-row {{
      display: inline-flex;
      gap: 6px;
      align-items: baseline;
    }}
    .meta-key {{
      font-weight: 700;
      color: #6a4d39;
    }}
    .caption {{
      margin: 12px 0 0;
      color: #473e35;
      font-size: 14px;
      line-height: 1.45;
    }}
    .reference-grid, .generated-grid {{
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }}
    .reference-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .generated-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .video-slot {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: #fff;
    }}
    .slot-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    video, img {{
      width: 100%;
      border-radius: 10px;
      background: #0e0d0c;
      display: block;
    }}
    img {{
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #f8f3eb;
    }}
    .missing {{
      min-height: 124px;
      display: grid;
      place-items: center;
      border-radius: 10px;
      color: var(--muted);
      background: #f7f1e7;
      border: 1px dashed var(--line);
      font-size: 13px;
    }}
    .empty {{
      color: var(--muted);
      margin: 0;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Wan 2.2 Training-Time Eval Portal</h1>
      <p class="subtitle">Fixed benchmark samples are grouped by sample and aligned across training checkpoints. Validation context curves and aggregate metrics are shown above the sample gallery.</p>
      <div class="step-badges">{step_badges}</div>
    </section>

    <section class="section">
      <h2>Fixed Benchmark Overview</h2>
      {render_overview_table(benchmark_step_summaries)}
    </section>

    <section class="section">
      <h2>Validation Trends By Training Step</h2>
      {render_validation_step_trends(validation_steps)}
    </section>

    <section class="section">
      <h2>Validation Metrics</h2>
      <div class="validation-grid">
        {render_validation_section(validation_steps)}
      </div>
    </section>

    <section class="section">
      <h2>Benchmark Samples</h2>
      <div class="filters">
        <input id="searchBox" type="search" placeholder="Search sample id or caption">
        <select id="datasetFilter">
          <option value="">All datasets</option>
          {dataset_options}
        </select>
      </div>
      <div id="sampleGrid" class="sample-grid">
        {render_sample_cards(steps, samples)}
      </div>
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.sample-card'));

    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      const dataset = datasetFilter.value;
      for (const card of cards) {{
        const matchesDataset = !dataset || card.dataset.dataset === dataset;
        const haystack = `${{card.dataset.sampleId}} ${{card.dataset.caption}} ${{card.dataset.dataset}}`.toLowerCase();
        const matchesSearch = !search || haystack.includes(search);
        card.style.display = matchesDataset && matchesSearch ? '' : 'none';
      }}
    }}

    searchBox.addEventListener('input', applyFilters);
    datasetFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def build_compare_html(
    model_names: list[str],
    samples: list[dict[str, Any]],
    model_summaries: list[dict[str, Any]],
    compare_rows: list[dict[str, str]],
    compare_summary: dict[str, Any] | None,
) -> str:
    datasets = sorted({str(sample["dataset"]) for sample in samples})
    dataset_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>"
        for name in datasets
    )
    model_badges = "".join(f"<span class='step-badge'>{html.escape(name)}</span>" for name in model_names)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sample300 Compare Portal</title>
  <style>
    :root {{
      --bg: #f6f2e8;
      --panel: #fffdf8;
      --ink: #1f1c18;
      --muted: #6f665d;
      --line: #ddd1bc;
      --accent: #b9512d;
      --accent-soft: #f3ddcc;
      --tag: #efe4d3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      background: radial-gradient(circle at top left, #fff7ea 0%, var(--bg) 45%, #eadfcf 100%);
      color: var(--ink);
    }}
    .page {{
      max-width: 1800px;
      margin: 0 auto;
      padding: 28px 24px 60px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,253,248,0.96), rgba(246,234,214,0.96));
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 24px 28px;
      box-shadow: 0 18px 40px rgba(54, 36, 17, 0.08);
      margin-bottom: 22px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    h1 {{
      font-size: 34px;
      line-height: 1.08;
      letter-spacing: -0.02em;
    }}
    .subtitle {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 15px;
    }}
    .step-badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }}
    .step-badge {{
      background: var(--tag);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 13px;
      color: var(--muted);
    }}
    .section {{
      background: rgba(255, 253, 248, 0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px 22px;
      box-shadow: 0 12px 28px rgba(54, 36, 17, 0.06);
      margin-bottom: 22px;
    }}
    .section h2 {{
      font-size: 24px;
      margin-bottom: 14px;
    }}
    .overview-table, .curve-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .overview-table th, .overview-table td,
    .curve-table th, .curve-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .overview-table th, .curve-table th {{
      color: var(--muted);
      font-weight: 600;
      background: rgba(239, 228, 211, 0.55);
      position: sticky;
      top: 0;
    }}
    .curve-table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      margin-top: 14px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      margin-top: 14px;
    }}
    .chart-grid-wide {{
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    }}
    .chart-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fff;
    }}
    .chart-card-wide {{
      overflow-x: auto;
    }}
    .chart-card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }}
    .chart-card h4 {{
      font-size: 14px;
      line-height: 1.25;
      word-break: break-word;
    }}
    .chart-direction {{
      font-size: 11px;
      color: var(--muted);
      background: #f6efe5;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
      white-space: nowrap;
    }}
    .metric-chart {{
      width: 100%;
      display: block;
    }}
    .metric-chart rect {{
      fill: #fffaf2;
      stroke: #eadfcd;
    }}
    .metric-chart .grid line {{
      stroke: #eadfcd;
      stroke-dasharray: 3 4;
    }}
    .metric-chart .grid text,
    .metric-chart .xlabels text {{
      fill: #7a7168;
      font-size: 10px;
    }}
    .metric-chart .axis {{
      stroke: #a59484;
      stroke-width: 1.25;
    }}
    .metric-chart .series {{
      fill: none;
      stroke: var(--accent);
      stroke-width: 2.5;
      stroke-linecap: round;
      stroke-linejoin: round;
    }}
    .metric-chart .points circle {{
      fill: var(--accent);
      stroke: #fff;
      stroke-width: 1.2;
    }}
    .chart-foot {{
      margin-top: 8px;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .chart-legend {{
      margin-top: 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px 10px;
    }}
    .chart-legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      background: #f9f1e6;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 4px 8px;
    }}
    .chart-swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: none;
    }}
    .analysis-list {{
      margin: 12px 0 0;
      padding-left: 20px;
      color: #473e35;
      line-height: 1.5;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .filters input, .filters select {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      min-width: 220px;
    }}
    .sample-grid {{
      display: grid;
      gap: 18px;
    }}
    .sample-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--panel);
      padding: 18px;
      box-shadow: 0 14px 24px rgba(54, 36, 17, 0.05);
    }}
    .sample-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }}
    .sample-header h3 {{
      font-size: 22px;
      line-height: 1.18;
      word-break: break-word;
    }}
    .dataset-tag {{
      flex: none;
      background: var(--accent-soft);
      border: 1px solid #e5bc9f;
      color: #7a381f;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
    }}
    .sample-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .meta-row {{
      display: inline-flex;
      gap: 6px;
      align-items: baseline;
    }}
    .meta-key {{
      font-weight: 700;
      color: #6a4d39;
    }}
    .caption {{
      margin: 12px 0 0;
      color: #473e35;
      font-size: 14px;
      line-height: 1.45;
    }}
    .reference-grid, .generated-grid {{
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }}
    .reference-grid {{
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }}
    .generated-grid {{
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    }}
    .video-slot {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: #fff;
    }}
    .slot-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .slot-subtitle {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    video, img {{
      width: 100%;
      border-radius: 10px;
      background: #0e0d0c;
      display: block;
    }}
    img {{
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #f8f3eb;
    }}
    .missing {{
      min-height: 124px;
      display: grid;
      place-items: center;
      border-radius: 10px;
      color: var(--muted);
      background: #f7f1e7;
      border: 1px dashed var(--line);
      font-size: 13px;
    }}
    .empty {{
      color: var(--muted);
      margin: 0;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Wan 2.2 Sample300 Compare Portal</h1>
      <p class="subtitle">Each sample is grouped on one card with aligned context, GT, baseline output, and 8k-step output. Aggregate metrics are shown above once comparison finishes.</p>
      <div class="step-badges">{model_badges}</div>
    </section>

    <section class="section">
      <h2>Generation Overview</h2>
      {render_compare_overview_table(model_summaries)}
    </section>

    <section class="section">
      <h2>Comparison Metrics</h2>
      {render_compare_analysis(compare_summary)}
      {render_compare_metric_charts(compare_rows, model_names)}
      {render_compare_metrics_table(compare_rows, compare_summary)}
    </section>

    <section class="section">
      <h2>Benchmark Samples</h2>
      <div class="filters">
        <input id="searchBox" type="search" placeholder="Search sample id or caption">
        <select id="datasetFilter">
          <option value="">All datasets</option>
          {dataset_options}
        </select>
      </div>
      <div id="sampleGrid" class="sample-grid">
        {render_compare_sample_cards(model_names, samples)}
      </div>
    </section>
  </div>
  <script>
    const searchBox = document.getElementById('searchBox');
    const datasetFilter = document.getElementById('datasetFilter');
    const cards = Array.from(document.querySelectorAll('.sample-card'));

    function applyFilters() {{
      const search = searchBox.value.trim().toLowerCase();
      const dataset = datasetFilter.value;
      for (const card of cards) {{
        const matchesDataset = !dataset || card.dataset.dataset === dataset;
        const haystack = `${{card.dataset.sampleId}} ${{card.dataset.caption}} ${{card.dataset.dataset}}`.toLowerCase();
        const matchesSearch = !search || haystack.includes(search);
        card.style.display = matchesDataset && matchesSearch ? '' : 'none';
      }}
    }}

    searchBox.addEventListener('input', applyFilters);
    datasetFilter.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.benchmark_root is not None:
        benchmark_root = args.benchmark_root.resolve()
        portal_dir = benchmark_root / COMPARE_PORTAL_SUBDIR
        portal_dir.mkdir(parents=True, exist_ok=True)

        model_names = parse_model_names(args.compare_model_names)
        model_names, samples, model_summaries = gather_compare_samples(
            benchmark_root,
            portal_dir,
            model_names,
        )
        compare_rows, compare_summary = load_compare_metrics_table(benchmark_root)
        html_text = build_compare_html(
            model_names,
            samples,
            model_summaries,
            compare_rows,
            compare_summary,
        )
        index_path = portal_dir / "index.html"
        index_path.write_text(html_text, encoding="utf-8")
        payload = {
            "mode": "benchmark_compare",
            "model_names": model_names,
            "num_samples": len(samples),
            "num_compare_rows": len(compare_rows),
            "index_path": str(index_path),
        }
        (portal_dir / "build_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(index_path)
        return

    output_root = args.output_root.resolve()
    portal_dir = output_root / PORTAL_SUBDIR
    portal_dir.mkdir(parents=True, exist_ok=True)

    steps, samples, benchmark_step_summaries = gather_benchmark_samples(output_root, portal_dir)
    validation_steps = gather_validation_data(output_root)
    html_text = build_html(steps, samples, benchmark_step_summaries, validation_steps)
    index_path = portal_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")

    payload = {
        "mode": "training_eval",
        "steps": steps,
        "num_samples": len(samples),
        "num_validation_steps": len(validation_steps),
        "index_path": str(index_path),
    }
    (portal_dir / "build_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(index_path)


if __name__ == "__main__":
    main()
