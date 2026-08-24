#!/usr/bin/env python3
"""Build per-metric best/worst case pages for PhysicIQ-style dashboards."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from build_physiciq_physrvg_worst_case_dashboard import load_dashboard_payload


DEFAULT_PHYSICIQ_SOURCE_PAGE = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq/index.html"
)
DEFAULT_PHYSICIQ_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq-metric-extremes"
)
DEFAULT_TEST5_SOURCE_PAGE = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "test5/index.html"
)
DEFAULT_TEST5_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "test5-metric-extremes"
)

PHYRVG_DISPLAY_ORDER = (
    "physrvg_test5_lora_off",
    "physrvg_test5_lora_on",
    "full_sa_physrvg_dit",
    "full_sa_physrvg_dit_gpu56",
    "full_sa_physrvg_vjepa_loss",
    "full_sa_physrvg_vjepa_loss_0613_b2g2",
    "full_sa_physrvg_latent_mask_loss",
    "full_sa_physrvg_object_xssc_loss",
)
PHYRVG_DISPLAY_INDEX = {
    key: index for index, key in enumerate(PHYRVG_DISPLAY_ORDER)
}

# Keep every PHYRVG-Full-SA variant adjacent in the metric extreme view.
# The key-prefix fallback keeps newly added Full-SA experiments in the same
# block without requiring a manual inventory update.
PHYRVG_FULL_SA_PLUS_ORDER = (
    "full_sa_physrvg_dit",
    "full_sa_physrvg_dit_gpu56",
    "full_sa_physrvg_phyco_kubric_0717_b4gacc1",
    "full_sa_physrvg_no_vjepa_0717_b2g2",
    "full_sa_physrvg_vjepa_loss",
    "full_sa_physrvg_vjepa_loss_0613_b2g2",
    "full_sa_physrvg_latent_mask_loss",
    "full_sa_physrvg_object_xssc_loss",
    "full_sa_physrvg_vjepa_rect384x672_0717_b2g2",
    "full_sa_physrvg_vjepa_rect384x672_0717_w0p3_b4gacc1",
    "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_b2gacc2",
    "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled",
)
PHYRVG_FULL_SA_PLUS_INDEX = {
    key: index for index, key in enumerate(PHYRVG_FULL_SA_PLUS_ORDER)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-page", type=Path, default=DEFAULT_PHYSICIQ_SOURCE_PAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PHYSICIQ_OUTPUT_DIR)
    parser.add_argument("--page-title", type=str, default="PhysicIQ · 每 case 每指标 best/worst")
    parser.add_argument(
        "--subtitle",
        type=str,
        default="按 source case 分组；每个 case 的每个指标都在所有生成结果里横向比较，展示 best/worst，视频懒加载。",
    )
    return parser.parse_args()


def is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def is_phyrvg_method(method: dict[str, Any]) -> bool:
    key = str(method.get("key", "")).lower()
    label = str(method.get("label", "")).upper()
    return "physrvg" in key or "phyrvg" in key or label.startswith("PHYRVG-")


def is_phyrvg_full_sa_plus_method(method: dict[str, Any]) -> bool:
    key = str(method.get("key", ""))
    label = str(method.get("label", "")).strip().upper()
    return (
        key in PHYRVG_FULL_SA_PLUS_INDEX
        or key.startswith("full_sa_physrvg")
        or label.startswith("PHYRVG-FULL-SA")
    )


def display_methods(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(indexed_method: tuple[int, dict[str, Any]]) -> tuple[int, int, str]:
        original_index, method = indexed_method
        key = str(method.get("key", ""))
        if is_phyrvg_full_sa_plus_method(method):
            return (1, PHYRVG_FULL_SA_PLUS_INDEX.get(key, 999), key)
        if is_phyrvg_method(method):
            return (0, PHYRVG_DISPLAY_INDEX.get(key, 999), key)
        return (2, original_index, key)

    return [method for _, method in sorted(enumerate(methods), key=sort_key)]


def format_value(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10:
        return f"{value:.2f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def format_step(step: int, step_kind: str) -> str:
    return f"infer {step}" if step_kind == "inference" else str(step)


def select_extremes(
    record: dict[str, Any],
    cases: list[dict[str, Any]],
    spec: dict[str, Any],
) -> dict[str, Any]:
    direction = str(spec["direction"])
    metric_key = str(spec["key"])
    candidates: list[dict[str, Any]] = []
    videos = record.get("videos", {})
    metrics = record.get("metrics", {})
    for case in cases:
        stem = str(case["stem"])
        case_metrics = metrics.get(stem, {})
        value = case_metrics.get(metric_key)
        video = videos.get(stem)
        if not is_number(value) or not isinstance(video, str) or not video:
            continue
        candidates.append(
            {
                "stem": stem,
                "value": float(value),
                "video": video,
            }
        )
    if not candidates:
        return {
            "complete": False,
            "count": 0,
            "expected": len(cases),
            "best": None,
            "worst": None,
            "span": None,
        }
    candidates.sort(
        key=lambda item: (
            item["value"] if direction == "lower" else -item["value"],
            item["stem"],
        )
    )
    best = dict(candidates[0])
    worst = dict(candidates[-1])
    best["rank"] = 1
    worst["rank"] = len(candidates)
    best["count"] = len(candidates)
    worst["count"] = len(candidates)
    span = abs(float(best["value"]) - float(worst["value"]))
    return {
        "complete": len(candidates) == len(cases),
        "count": len(candidates),
        "expected": len(cases),
        "best": best,
        "worst": worst,
        "span": span,
    }


def build_case_groups(
    cases: list[dict[str, Any]],
    record_rows: list[dict[str, Any]],
    metric_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups_by_stem: dict[str, dict[str, Any]] = {}
    for case in cases:
        stem = str(case["stem"])
        groups_by_stem[stem] = {
            "stem": stem,
            "prompt": str(case.get("prompt", "")),
            "gt": str(case.get("gt", "")),
            "context": str(case.get("context", "")),
            "total_hits": 0,
            "best_hits": 0,
            "worst_hits": 0,
            "metrics_with_hits": 0,
            "metric_hits": {
                str(spec["key"]): {
                    "best": [],
                    "worst": [],
                    "candidate_count": 0,
                    "span": None,
                }
                for spec in metric_specs
            },
        }

    for case in cases:
        stem = str(case["stem"])
        group = groups_by_stem[stem]
        for spec in metric_specs:
            metric_key = str(spec["key"])
            direction = str(spec.get("direction", "higher"))
            candidates: list[dict[str, Any]] = []
            for pair in record_rows:
                record = pair["record"]
                row = pair["row"]
                case_metrics = record.get("metrics", {}).get(stem, {})
                value = case_metrics.get(metric_key)
                video = record.get("videos", {}).get(stem)
                if not is_number(value) or not isinstance(video, str) or not video:
                    continue
                candidates.append(
                    {
                        "kind": "",
                        "metric_key": metric_key,
                        "metric_label": str(spec["label"]),
                        "direction": direction,
                        "method_key": str(row["method_key"]),
                        "method_label": str(row["method_label"]),
                        "step": int(row["step"]),
                        "step_kind": str(row["step_kind"]),
                        "row_label": str(row["row_label"]),
                        "color": str(row["color"]),
                        "rank": 0,
                        "count": 0,
                        "value": float(value),
                        "video": video,
                    }
                )
            metric = group["metric_hits"][metric_key]
            metric["candidate_count"] = len(candidates)
            if not candidates:
                continue
            candidates.sort(
                key=lambda item: (
                    item["value"] if direction == "lower" else -item["value"],
                    item["row_label"],
                    item["method_key"],
                    item["step"],
                )
            )
            best = dict(candidates[0])
            worst = dict(candidates[-1])
            best["kind"] = "best"
            worst["kind"] = "worst"
            best["rank"] = 1
            worst["rank"] = len(candidates)
            best["count"] = len(candidates)
            worst["count"] = len(candidates)
            metric["best"] = [best]
            metric["worst"] = [worst]
            metric["span"] = abs(float(best["value"]) - float(worst["value"]))
            group["total_hits"] += 2
            group["best_hits"] += 1
            group["worst_hits"] += 1

    for group in groups_by_stem.values():
        group["metrics_with_hits"] = sum(
            1
            for metric in group["metric_hits"].values()
            if metric["best"] or metric["worst"]
        )
        for metric in group["metric_hits"].values():
            for kind in ("best", "worst"):
                metric[kind].sort(
                    key=lambda hit: (
                        hit["metric_key"],
                        hit["method_label"],
                        hit["step"],
                        hit["value"],
                    )
                )

    return sorted(
        groups_by_stem.values(),
        key=lambda group: (-int(group["total_hits"]), str(group["stem"])),
    )


def build_page_data(payload: dict[str, Any]) -> dict[str, Any]:
    metric_specs = [
        {
            "key": str(spec["key"]),
            "label": str(spec["label"]),
            "direction": str(spec.get("direction", "higher")),
        }
        for spec in payload["metricSpecs"]
    ]
    cases = [
        {
            "stem": str(case["stem"]),
            "prompt": str(case.get("prompt", "")),
            "gt": str(case.get("gt", "")),
            "context": str(case.get("context", "")),
        }
        for case in payload["cases"]
    ]
    methods = display_methods(
        [
            {
                "key": str(method["key"]),
                "label": str(method["label"]),
                "color": str(method.get("color", "#52636d")),
            }
            for method in payload["methods"]
        ]
    )
    method_meta = {str(method["key"]): method for method in methods}
    method_order = {str(method["key"]): index for index, method in enumerate(methods)}
    records_with_data = [
        record
        for record in payload["records"]
        if isinstance(record.get("videos"), dict)
        and isinstance(record.get("metrics"), dict)
    ]
    records = sorted(
        records_with_data,
        key=lambda record: (
            method_order.get(str(record["method_key"]), 999),
            int(record["step"]),
            str(record.get("origin", "")),
        ),
    )
    rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for row_index, record in enumerate(records):
        step = int(record["step"])
        step_kind = str(record.get("step_kind", "training"))
        method_key = str(record["method_key"])
        method_label = str(
            record.get("method_label")
            or method_meta.get(method_key, {}).get("label", method_key)
        )
        color = str(method_meta.get(method_key, {}).get("color", "#52636d"))
        row = {
            "row_id": f"{method_key}::{step_kind}::{step}::{row_index}",
            "method_key": method_key,
            "method_label": method_label,
            "step": step,
            "step_kind": step_kind,
            "row_label": f"{method_label} · {format_step(step, step_kind)}",
            "color": color,
        }
        rows.append(row)
        record_rows.append({"row": row, "record": record})
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "page_title": str(payload.get("page_title", "")),
        "metric_specs": metric_specs,
        "cases": cases,
        "methods": [
            {
                "key": str(method["key"]),
                "label": str(method["label"]),
                "color": str(method["color"]),
                "steps": sorted(
                    {
                        int(record["step"])
                        for record in records
                        if str(record["method_key"]) == str(method["key"])
                    }
                ),
                "default_step": max(
                    [
                        int(record["step"])
                        for record in records
                        if str(record["method_key"]) == str(method["key"])
                    ]
                    or [0]
                ),
            }
            for method in methods
        ],
        "rows": rows,
        "case_groups": build_case_groups(cases, record_rows, metric_specs),
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root{--bg:#f3f5f6;--surface:#fff;--ink:#172126;--muted:#657278;--line:#d6dde0;
      --deep:#17323d;--deep2:#0f252d;--teal:#0b6e75;--green:#176b5c;--rust:#ad452f;
      --gold:#d59b27;--blue:#315c87;--shadow:0 9px 28px rgba(18,46,57,.09)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:
      radial-gradient(circle at 15% 0,#e7f1f2 0 24%,transparent 25%),var(--bg);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}a{color:inherit}
    .hero{padding:24px clamp(15px,4vw,56px) 22px;background:linear-gradient(115deg,var(--deep) 0 72%,var(--deep2) 72% 100%);color:#f7fbfc}
    .hero-top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
    .hero-top a{text-decoration:none;color:#a9d6da;font-size:12px;font-weight:850}
    .eyebrow{font:700 10px/1 "Arial Narrow",sans-serif;letter-spacing:.16em;text-transform:uppercase;color:#7fc4c9}
    h1{margin:17px 0 7px;font:850 clamp(28px,4.2vw,54px)/1 "Arial Narrow","Roboto Condensed",sans-serif;letter-spacing:-.025em}
    .hero p{max-width:1080px;margin:0;color:#bfd3d8;font-size:13px;line-height:1.6}
    .case-ribbon{display:grid;grid-template-columns:repeat(18,1fr);gap:3px;margin-top:22px;max-width:980px}
    .case-ribbon span{height:5px;background:#3a626d}.case-ribbon span:nth-child(3n+1){background:#d59b27}
    .case-ribbon span:nth-child(3n+2){background:#0b6e75}.case-ribbon span:nth-child(3n){background:#ad452f}
    .toolbar{position:sticky;top:0;z-index:20;padding:10px clamp(12px,3vw,36px);background:rgba(243,245,246,.98);
      border-bottom:1px solid var(--line);box-shadow:0 5px 18px rgba(18,46,57,.08)}
    .toolbar-inner{display:grid;grid-template-columns:minmax(260px,1.35fr) minmax(210px,1fr) minmax(130px,.58fr)
      minmax(160px,.72fr) minmax(130px,.55fr) minmax(180px,.85fr) repeat(2,96px);gap:8px;align-items:end}
    label{display:grid;gap:4px;color:var(--muted);font-size:10px;font-weight:850;letter-spacing:.04em}
    select,input,button{height:38px;border:1px solid var(--line);border-radius:4px;background:#fff;color:var(--ink);font:750 12px/1 inherit}
    select,input{width:100%;padding:0 9px}button{padding:0 12px;cursor:pointer}
    button:hover,select:hover,input:hover{border-color:#b8c6cb}
    button:focus-visible,select:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid var(--gold);outline-offset:2px}
    main{max-width:1880px;margin:auto;padding:18px clamp(10px,2vw,30px) 70px}
    .summary{display:grid;grid-template-columns:repeat(4,minmax(180px,1fr));gap:10px;margin-bottom:14px}
    .summary-card{padding:12px 13px;border:1px solid var(--line);border-top:4px solid var(--accent);background:var(--surface);box-shadow:var(--shadow)}
    .summary-card span{display:block;color:var(--muted);font-size:10px;font-weight:850;letter-spacing:.04em;text-transform:uppercase}
    .summary-card strong{display:block;margin-top:6px;font:850 18px/1.15 "Arial Narrow",sans-serif}
    .summary-card small{display:block;margin-top:5px;color:var(--muted);font-size:11px;line-height:1.45;overflow-wrap:anywhere}
    .section-head{display:flex;align-items:end;justify-content:space-between;gap:15px;margin:2px 0 12px}
    .section-head h2{margin:0;font:850 23px/1 "Arial Narrow",sans-serif}
    .section-head p{margin:0;color:var(--muted);font-size:11px;line-height:1.5;text-align:right}
    .groups{display:grid;gap:13px}
    .case-group{border:1px solid var(--line);border-left:6px solid var(--teal);background:var(--surface);box-shadow:var(--shadow)}
    .group-head{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:14px;padding:13px 14px 11px;border-bottom:1px solid var(--line);background:#fbfcfc}
    .case-title{margin:0;font-size:14px;font-weight:950;line-height:1.35;overflow-wrap:anywhere}
    .prompt{max-width:1100px;margin:6px 0 0;color:var(--muted);font-size:11px;line-height:1.5;overflow-wrap:anywhere}
    .chips{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px;align-content:start}
    .chip{padding:4px 7px;border:1px solid var(--line);border-radius:999px;background:#f5f8f8;color:var(--muted);font-size:10px;font-weight:850;white-space:nowrap}
    .chip.best{color:var(--green);border-color:#cfe3dd;background:#edf8f3}.chip.worst{color:var(--rust);border-color:#efd0c7;background:#fbefeb}
    .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:10px;padding:11px;background:#f6f8f8}
    .metric-tile{display:grid;gap:8px;align-content:start;padding:10px;border:1px solid var(--line);border-top:4px solid var(--blue);background:#fff}
    .metric-tile.empty{opacity:.72}.metric-title{display:flex;align-items:center;justify-content:space-between;gap:8px}
    .metric-title h3{margin:0;font-size:12px;font-weight:950;line-height:1.25}.metric-title span{color:var(--muted);font-size:10px;font-weight:850;white-space:nowrap}
    .bw{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
    .lane{display:grid;gap:7px;align-content:start}.lane-head{display:flex;align-items:center;justify-content:space-between;gap:8px;
      padding-bottom:4px;border-bottom:1px solid #e3e9eb;font-size:10px;font-weight:950;letter-spacing:.08em;text-transform:uppercase}
    .lane.best .lane-head{color:var(--green)}.lane.worst .lane-head{color:var(--rust)}
    .hit{--accent:#52636d;display:grid;gap:6px;padding:8px;border:1px solid #dce4e7;border-left:4px solid var(--accent);background:#fbfcfc}
    .hit-top{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:start}
    .method{min-width:0;font-size:11px;font-weight:950;line-height:1.35;overflow-wrap:anywhere}
    .rank{padding:3px 5px;border-radius:3px;background:#eef3f3;color:var(--muted);font-size:10px;font-weight:900;white-space:nowrap}
    .value{font:850 17px/1 "Arial Narrow",sans-serif;color:var(--accent)}
    .note{color:var(--muted);font-size:10px;line-height:1.35}.empty-lane{padding:16px 8px;border:1px dashed #d9e1e4;color:#91a0a6;font-size:10px;text-align:center;background:#fbfcfc}
    video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#0d171b}
    .empty{padding:42px 16px;border:1px dashed var(--line);background:#fff;color:var(--muted);font-size:12px;line-height:1.5}
    .footer{margin-top:16px;color:var(--muted);font-size:10px}
    @media(max-width:1180px){.toolbar-inner{grid-template-columns:repeat(3,minmax(0,1fr))}.metric-grid{grid-template-columns:1fr}}
    @media(max-width:720px){header{padding:18px 13px}.toolbar-inner{grid-template-columns:1fr 1fr}.summary{grid-template-columns:1fr 1fr}.group-head{grid-template-columns:1fr}.chips{justify-content:flex-start}.bw{grid-template-columns:1fr}}
    @media(max-width:500px){.summary{grid-template-columns:1fr}.toolbar-inner{grid-template-columns:1fr}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-top">
      <a href="../">← 返回 8844 总览</a>
      <a href="../physiciq-average-metrics/">PhysicIQ 平均指标</a>
      <a href="../test5-average-metrics/">test_5 平均指标</a>
      <a href="../physiciq/">PhysicIQ case 合并</a>
      <a href="../test5/">test_5 case 合并</a>
    </div>
    <div class="eyebrow">per-case generated-result extremes</div>
    <h1>__TITLE__</h1>
    <p>__SUBTITLE__</p>
    <div class="case-ribbon" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
  </header>
  <div class="toolbar">
    <div class="toolbar-inner">
      <label>source case<select id="case-select"></select></label>
      <label>指标<select id="metric-select"></select></label>
      <label>极值类型<select id="kind-select"><option value="all">best + worst</option><option value="best">只看 best</option><option value="worst">只看 worst</option></select></label>
      <label>排序<select id="sort-select"><option value="stem">case 名称</option><option value="metrics">可比指标数</option><option value="hits">极值卡片数</option><option value="best">best 数</option><option value="worst">worst 数</option></select></label>
      <label>展示<select id="limit-select"><option value="all">全部</option><option value="12">前 12 组</option><option value="24">前 24 组</option><option value="48">前 48 组</option></select></label>
      <label>搜索<input id="query" placeholder="stem / prompt"></label>
      <button id="replay" title="重播当前可见视频">↺ 重播</button>
      <button id="play" title="播放当前可见视频">▶ 播放</button>
    </div>
  </div>
  <main>
    <section class="summary" id="summary"></section>
    <section class="section-head">
      <div><h2 id="view-title"></h2></div>
      <p id="view-note"></p>
    </section>
    <section id="groups" class="groups"></section>
    <p class="footer">生成时间：__GENERATED__ · 对每个 source case 的每个指标，候选集合是该 case 的所有已生成视频结果；#1/N 是当前指标 best，#N/N 是当前指标 worst。</p>
  </main>
  <script>
    const D=__DATA__;
    const caseSelect=document.getElementById("case-select");
    const metricSelect=document.getElementById("metric-select");
    const kindSelect=document.getElementById("kind-select");
    const sortSelect=document.getElementById("sort-select");
    const limitSelect=document.getElementById("limit-select");
    const queryInput=document.getElementById("query");
    const summaryRoot=document.getElementById("summary");
    const groupsRoot=document.getElementById("groups");
    const viewTitle=document.getElementById("view-title");
    const viewNote=document.getElementById("view-note");
    const metricsByKey=Object.fromEntries(D.metric_specs.map(spec=>[spec.key,spec]));
    let videoObserver=null;
    const fmt=v=>{if(!Number.isFinite(v))return "—";const a=Math.abs(v);if(a>=10)return v.toFixed(2);if(a>=1)return v.toFixed(3);return v.toFixed(4)};
    const esc=s=>String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
    const clip=(s,n=190)=>{s=String(s??"").trim();return s.length>n?s.slice(0,n-1)+"…":s};
    const directionText=spec=>spec.direction==="lower"?"lower is better":"higher is better";
    function stepText(hit){return `${hit.step_kind==="inference"?"infer ":""}${hit.step}`;}
    function selectedMetricSpecs(){
      const key=metricSelect.value;
      return key==="__all__" ? D.metric_specs : D.metric_specs.filter(spec=>spec.key===key);
    }
    function filteredHits(metric, kind){
      if(kindSelect.value==="best" && kind==="worst")return [];
      if(kindSelect.value==="worst" && kind==="best")return [];
      return metric[kind] || [];
    }
    function groupHitsForView(group){
      let total=0, best=0, worst=0, metrics=0;
      selectedMetricSpecs().forEach(spec=>{
        const metric=group.metric_hits[spec.key] || {best:[],worst:[],candidate_count:0,span:null};
        const b=filteredHits(metric,"best").length;
        const w=filteredHits(metric,"worst").length;
        best+=b; worst+=w; total+=b+w;
        if(b+w>0)metrics+=1;
      });
      return {total,best,worst,metrics};
    }
    function sortGroups(groups){
      const mode=sortSelect.value;
      const sorted=[...groups];
      sorted.sort((a,b)=>{
        if(mode==="stem")return a.stem.localeCompare(b.stem);
        if(mode==="metrics")return b.metrics_with_hits-a.metrics_with_hits || b.total_hits-a.total_hits || a.stem.localeCompare(b.stem);
        if(mode==="best")return b.best_hits-a.best_hits || b.total_hits-a.total_hits || a.stem.localeCompare(b.stem);
        if(mode==="worst")return b.worst_hits-a.worst_hits || b.total_hits-a.total_hits || a.stem.localeCompare(b.stem);
        return b.total_hits-a.total_hits || b.metrics_with_hits-a.metrics_with_hits || a.stem.localeCompare(b.stem);
      });
      return sorted;
    }
    function visibleGroups(){
      const selectedCase=caseSelect.value;
      const query=queryInput.value.trim().toLowerCase();
      let groups=D.case_groups.filter(group=>{
        if(selectedCase!=="__all__" && group.stem!==selectedCase)return false;
        if(query && !(group.stem.toLowerCase().includes(query) || group.prompt.toLowerCase().includes(query)))return false;
        return selectedCase!=="__all__" || groupHitsForView(group).total>0 || metricSelect.value==="__all__";
      });
      groups=sortGroups(groups);
      const limit=limitSelect.value==="all" ? Infinity : Number(limitSelect.value);
      return groups.slice(0,limit);
    }
    function hitCard(hit, kind){
      const accent=kind==="best" ? "var(--green)" : "var(--rust)";
      return `<article class="hit" style="--accent:${esc(hit.color||accent)}">
        <div class="hit-top"><div class="method">${esc(hit.method_label)}</div><div class="rank">#${hit.rank}/${hit.count}</div></div>
        <div class="value">${fmt(hit.value)}</div>
        <div class="note">${esc(stepText(hit))} · ${kind==="best"?"best":"worst"} among ${hit.count} generated results</div>
        <video data-src="${esc(hit.video)}" preload="none" muted playsinline controls></video>
      </article>`;
    }
    function lane(metric, kind){
      const hits=filteredHits(metric,kind);
      const label=kind==="best"?"BEST":"WORST";
      return `<div class="lane ${kind}"><div class="lane-head"><span>${label}</span><span>${hits.length}</span></div>
        ${hits.length ? hits.map(hit=>hitCard(hit,kind)).join("") : '<div class="empty-lane">无可用结果</div>'}
      </div>`;
    }
    function metricTile(group, spec){
      const metric=group.metric_hits[spec.key] || {best:[],worst:[],candidate_count:0,span:null};
      const b=filteredHits(metric,"best").length;
      const w=filteredHits(metric,"worst").length;
      const candidateCount=metric.candidate_count||0;
      const span=Number.isFinite(metric.span)?fmt(metric.span):"—";
      return `<section class="metric-tile ${b+w===0?"empty":""}">
        <div class="metric-title"><h3>${esc(spec.label)}</h3><span>${directionText(spec)} · ${candidateCount} results · spread ${span}</span></div>
        <div class="bw">${lane(metric,"best")}${lane(metric,"worst")}</div>
      </section>`;
    }
    function groupCard(group){
      const stat=groupHitsForView(group);
      const metrics=selectedMetricSpecs();
      return `<article class="case-group">
        <div class="group-head">
          <div>
            <h2 class="case-title">${esc(group.stem)}</h2>
            ${group.prompt ? `<p class="prompt">${esc(clip(group.prompt))}</p>` : ""}
          </div>
          <div class="chips">
            <span class="chip">${stat.metrics}/${metrics.length} metrics</span>
            <span class="chip">${stat.total} extreme cards</span>
            <span class="chip best">${stat.best} best</span>
            <span class="chip worst">${stat.worst} worst</span>
          </div>
        </div>
        <div class="metric-grid">${metrics.map(spec=>metricTile(group,spec)).join("")}</div>
      </article>`;
    }
    function ensureObserver(){
      if(videoObserver!==null)return videoObserver;
      if(!("IntersectionObserver" in window)){videoObserver=false;return videoObserver;}
      videoObserver=new IntersectionObserver(entries=>{
        entries.forEach(entry=>{
          if(!entry.isIntersecting)return;
          const video=entry.target;
          if(!video.dataset.src)return;
          video.src=video.dataset.src;
          delete video.dataset.src;
          video.load();
          videoObserver.unobserve(video);
        });
      },{rootMargin:"700px 0px"});
      return videoObserver;
    }
    function observeVideos(){
      const observer=ensureObserver();
      document.querySelectorAll("video[data-src]").forEach(video=>{
        if(observer){observer.observe(video);}else{video.src=video.dataset.src;delete video.dataset.src;video.load();}
      });
    }
    function render(){
      const groups=visibleGroups();
      const metricLabel=metricSelect.value==="__all__" ? "全部指标" : metricsByKey[metricSelect.value].label;
      const totalHits=groups.reduce((sum,group)=>sum+groupHitsForView(group).total,0);
      const metrics=selectedMetricSpecs();
      viewTitle.textContent=`${metricLabel} · per-case generated-result best/worst`;
      viewNote.textContent=`同一个 source case 聚在一组；每个指标都在该 case 的所有生成结果中选出 best 和 worst。`;
      summaryRoot.innerHTML=[
        `<article class="summary-card" style="--accent:#0b6e75"><span>Source Cases</span><strong>${groups.length}/${D.case_groups.length}</strong><small>当前筛选后展示的 source case 组</small></article>`,
        `<article class="summary-card" style="--accent:#315c87"><span>Metrics</span><strong>${metrics.length}/${D.metric_specs.length}</strong><small>${esc(metricLabel)}</small></article>`,
        `<article class="summary-card" style="--accent:#176b5c"><span>Visible Extremes</span><strong>${totalHits}</strong><small>当前页面显示的 best/worst 极值卡片数</small></article>`,
        `<article class="summary-card" style="--accent:#ad452f"><span>Generated Rows</span><strong>${D.rows.length}</strong><small>参与横向比较的模型 / checkpoint 生成结果</small></article>`,
      ].join("");
      groupsRoot.innerHTML=groups.length ? groups.map(groupCard).join("") : '<div class="empty">没有匹配的 source case。可以放宽指标、命中类型或搜索条件。</div>';
      observeVideos();
      localStorage.setItem("metricExtremesControls",JSON.stringify({
        case:caseSelect.value,metric:metricSelect.value,kind:kindSelect.value,sort:sortSelect.value,limit:limitSelect.value,query:queryInput.value
      }));
    }
    function videos(){return [...document.querySelectorAll("video")]}
    caseSelect.add(new Option("全部 source cases","__all__"));
    D.case_groups.forEach(group=>caseSelect.add(new Option(`${group.stem} · ${group.metrics_with_hits}/${D.metric_specs.length} metrics`,group.stem)));
    metricSelect.add(new Option("全部指标","__all__"));
    D.metric_specs.forEach(spec=>metricSelect.add(new Option(`${spec.label} ${spec.direction==="lower"?"↓":"↑"}`,spec.key)));
    try{
      const saved=JSON.parse(localStorage.getItem("metricExtremesControls")||"{}");
      if([...caseSelect.options].some(opt=>opt.value===saved.case))caseSelect.value=saved.case;
      if([...metricSelect.options].some(opt=>opt.value===saved.metric))metricSelect.value=saved.metric;
      if([...kindSelect.options].some(opt=>opt.value===saved.kind))kindSelect.value=saved.kind;
      if([...sortSelect.options].some(opt=>opt.value===saved.sort))sortSelect.value=saved.sort;
      if([...limitSelect.options].some(opt=>opt.value===saved.limit))limitSelect.value=saved.limit;
      if(saved.query)queryInput.value=saved.query;
    }catch{}
    [caseSelect,metricSelect,kindSelect,sortSelect,limitSelect].forEach(el=>el.onchange=render);
    queryInput.oninput=()=>render();
    document.getElementById("replay").onclick=()=>videos().forEach(video=>{video.currentTime=0;video.play().catch(()=>{})});
    document.getElementById("play").onclick=()=>videos().forEach(video=>video.play().catch(()=>{}));
    render();
  </script>
</body>
</html>
'''


def build_dashboard(
    source_page: Path,
    output_dir: Path,
    *,
    page_title: str,
    subtitle: str,
) -> Path:
    payload = load_dashboard_payload(source_page.resolve())
    data = build_page_data(payload)
    data["page_title"] = page_title
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = HTML_TEMPLATE.replace("__DATA__", encoded)
    html = html.replace("__TITLE__", page_title)
    html = html.replace("__SUBTITLE__", subtitle)
    html = html.replace("__GENERATED__", data["generated_utc"])
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    print(
        build_dashboard(
            args.source_page,
            args.output_dir,
            page_title=args.page_title,
            subtitle=args.subtitle,
        )
    )


if __name__ == "__main__":
    main()
