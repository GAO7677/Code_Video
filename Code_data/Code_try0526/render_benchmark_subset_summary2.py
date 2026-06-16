#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from physv_eval.records import load_payload, metric_value


METRICS = [
    "official_pdi",
    "scale_component",
    "traj_component",
    "epsilon_rigidity",
    "vp_component",
    "wmreward_surprise",
    "cosmos_reason1",
    "vjepa_temporal_relation_raw_error",
    "vjepa_delta_relation_raw_error",
    "vjepa_delta_profile_error",
    "videophy2_auto_sa",
    "videophy2_auto_pc",
    "videophy2_auto_joint",
]

METRIC_TITLES = {
    "official_pdi": "Official PDI ↓",
    "scale_component": "Scale ↓",
    "traj_component": "Trajectory ↓",
    "epsilon_rigidity": "Rigidity ↓",
    "vp_component": "VP ↓",
    "wmreward_surprise": "WMReward Surprise ↓",
    "cosmos_reason1": "Cosmos ↑",
    "vjepa_temporal_relation_raw_error": "RelRaw ↓",
    "vjepa_delta_relation_raw_error": "DeltaRel ↓",
    "vjepa_delta_profile_error": "DeltaProf ↓",
    "videophy2_auto_sa": "SA ↑",
    "videophy2_auto_pc": "PC ↑",
    "videophy2_auto_joint": "Joint ↑",
}

SERVER_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/abc_report")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/benchmark_summary_subpage")
OUTPUT_PATHS = [
    SERVER_ROOT / "benchmark_summary" / "index.html",
    OUTPUT_ROOT / "index.html",
]

BENCH_SPECS = {
    "A": {
        "title": "A / PDI-Bench",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output"),
        "benchmark": "PDI-Bench",
        "method_key": "method",
    },
    "B": {
        "title": "B / Dataset_physV_B_benchmark",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark/output"),
        "benchmark": "Dataset_physV_B_benchmark",
        "method_key": "method_name",
    },
    "D": {
        "title": "D / Physics-IQ",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark/output"),
        "benchmark": "physics-iq-benchmark",
        "method_key": "method",
    },
    "E": {
        "title": "E / PhyGenBench",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench/output"),
        "benchmark": "phygenbench",
        "method_key": "method",
    },
}

EXCLUDED_METHODS = {"FLUX_1_Kontext"}


def fv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def metric_if_complete(payloads: list[dict[str, Any]], metric_name: str) -> float | None:
    values = [metric_value(payload, metric_name) for payload in payloads]
    if any(value is None for value in values):
        return None
    return mean_or_none([float(value) for value in values if value is not None])


def method_name(payload: dict[str, Any], method_key: str, json_path: Path) -> str:
    value = payload.get(method_key)
    if isinstance(value, str) and value:
        return value
    if method_key == "method_name":
        fallback = payload.get("method")
        if isinstance(fallback, str) and fallback:
            return fallback
    return json_path.parent.name


def collect_rows() -> dict[str, list[dict[str, Any]]]:
    bench_rows: dict[str, list[dict[str, Any]]] = {}
    for bench_id, spec in BENCH_SPECS.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        root: Path = spec["root"]
        benchmark_name = spec["benchmark"]
        key = spec["method_key"]
        for json_path in sorted(root.rglob("*.json")):
            payload = load_payload(json_path)
            if payload.get("benchmark") != benchmark_name:
                continue
            method = method_name(payload, key, json_path)
            if method in EXCLUDED_METHODS:
                continue
            grouped[method].append(payload)
        bench_rows[bench_id] = _finalize_rows(grouped)
    return bench_rows


def _finalize_rows(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payloads in sorted(grouped.items()):
        row = {
            "method": method,
            "count": len(payloads),
        }
        for metric in METRICS:
            row[metric] = metric_if_complete(payloads, metric)
        rows.append(row)
    return rows


def render_table(title: str, rows: list[dict[str, Any]]) -> str:
    headers = "".join(f"<th>{METRIC_TITLES[name]}</th>" for name in METRICS)
    body = []
    for row in rows:
        metric_cells = "".join(f"<td class='num'>{fv(row[name])}</td>" for name in METRICS)
        body.append(
            "<tr>"
            f"<td class='label-cell'>{row['method']}</td>"
            f"<td class='num'>{row['count']}</td>"
            f"{metric_cells}"
            "</tr>"
        )
    return f"""
    <section class="block">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>N</th>
            {headers}
          </tr>
        </thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def build_html() -> str:
    rows_by_bench = collect_rows()
    sections = []
    for bench_id in ["A", "B", "D", "E"]:
        spec = BENCH_SPECS[bench_id]
        sections.append(render_table(spec["title"], rows_by_bench.get(bench_id, [])))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Method Summary</title>
  <style>
    body {{ margin: 0; background: #0f1417; color: #eaf0f4; font-family: system-ui, sans-serif; }}
    .page {{ max-width: 1760px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; }}
    .sub {{ color: #9cb1c0; margin-bottom: 20px; line-height: 1.6; }}
    .block {{ margin: 22px 0 30px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: #162027; border: 1px solid #30414f; }}
    th, td {{ border-bottom: 1px solid #30414f; padding: 8px 10px; font-size: 12px; }}
    th {{ background: rgba(255,255,255,0.05); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }}
    .label-cell {{ font-weight: 700; white-space: nowrap; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Benchmark Method Summary</h1>
    <div class="sub">按方法统计各测试集均值。当前统计口径排除了 <code>FLUX_1_Kontext</code>，缺失指标保持空白，不补默认值。</div>
    {''.join(sections)}
  </div>
</body>
</html>"""


def main() -> None:
    html = build_html()
    for path in OUTPUT_PATHS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    print(OUTPUT_PATHS[0])


if __name__ == "__main__":
    main()
