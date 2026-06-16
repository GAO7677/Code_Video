#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from physv_eval.records import load_payload, metric_value


OUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/benchmark_summary_subpage")
OUT_HTML = OUT_ROOT / "index.html"

BENCHES = {
    "A": {
        "label": "A / PDI-Bench",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output"),
        "method_dirs": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        "pattern": "**/*.json",
        "group_fn": lambda p: p.parent.name,
    },
    "B": {
        "label": "B / Dataset_physV",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark/output"),
        "method_dirs": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        "pattern": "**/*.json",
    },
    "D": {
        "label": "D / Physics-IQ",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark/output"),
        "method_dirs": ["GT", "wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"],
        "pattern": "**/*.json",
        "group_fn": lambda p: p.parent.name,
    },
    "E": {
        "label": "E / PhyGenBench",
        "root": Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench/output"),
        "method_dirs": ["wan22-5B-TI2V", "VACE_1p3B_TI2V"],
        "pattern": "**/*.json",
        "group_fn": lambda p: p.parent.name,
    },
}

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


def fv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def metric_row(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    row = {"count": len(payloads)}
    for metric in METRICS:
        vals = [metric_value(payload, metric) for payload in payloads]
        row[metric] = mean([float(v) for v in vals if v is not None])
    return row


def collect_bench(bench_id: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    root = cfg["root"]
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for json_path in sorted(root.rglob("*.json")):
        payload = load_payload(json_path)
        if bench_id == "E" and payload.get("benchmark") != "phygenbench":
            continue
        if bench_id == "D" and payload.get("benchmark") != "physics-iq-benchmark":
            continue
        if bench_id == "A" and payload.get("benchmark") != "PDI-Bench":
            continue
        if bench_id == "A":
            category = json_path.parent.name
        elif bench_id == "B":
            category = json_path.parent.name
        elif bench_id == "D":
            category = payload.get("category") or "UNKNOWN"
        method = payload.get("method") or json_path.parts[-3]
        grouped[(str(category), str(method))].append(payload)

    for (category, method), payloads in sorted(grouped.items()):
        row = {
            "category": category,
            "method": method,
            **metric_row(payloads),
        }
        rows.append(row)
    return rows


def render_table(title: str, rows: list[dict[str, Any]]) -> str:
    headers = "".join(f"<th>{METRIC_TITLES[m]}</th>" for m in METRICS)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td class='label-cell'>{row['category']}</td>"
            f"<td class='label-cell'>{row['method']}</td>"
            f"<td class='num'>{row['count']}</td>"
            + "".join(f"<td class='num'>{fv(row[m])}</td>" for m in METRICS)
            + "</tr>"
        )
    return f"""
    <section class="block">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr>
            <th>Test set</th>
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
    sections = [render_table(cfg["label"], collect_bench(bench_id, cfg)) for bench_id, cfg in BENCHES.items()]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Benchmark Subset Summary</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #0f1417; color: #eaf0f4; }}
    .page {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 16px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .block {{ margin: 20px 0 28px; }}
    table {{ width: 100%; border-collapse: collapse; background: #162027; border: 1px solid #30414f; }}
    th, td {{ border-bottom: 1px solid #30414f; padding: 8px 10px; font-size: 12px; }}
    th {{ background: rgba(255,255,255,0.05); text-transform: uppercase; font-size: 10px; letter-spacing: 0.04em; }}
    .label-cell {{ font-weight: 700; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Benchmark Subset Summary</h1>
    <div>按测试集和方法统计均值。缺失指标保持空白，不补默认值。</div>
    {''.join(sections)}
  </div>
</body>
</html>"""


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(), encoding="utf-8")
    print(OUT_HTML)


if __name__ == "__main__":
    main()
