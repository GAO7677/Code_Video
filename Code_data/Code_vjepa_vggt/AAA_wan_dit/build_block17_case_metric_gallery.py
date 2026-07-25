#!/usr/bin/env python3
"""Build a case-level Block 17 ablation metric gallery."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan")
TESTDATASET_ROOT = Path("/data/gaoya/AAA_test_video/0623/testdataset")

MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}

MODE_LABELS = {
    "whole_block": "Whole block bypass",
    "self_attn_zero": "Self-attention output = 0",
    "object_cross_attn": "Object cross-attention output = 0",
    "text_cross_attn_zero": "Text cross-attention output = 0",
    "ffn_zero": "FFN output = 0",
    "lora_off": "LoRA disabled",
}


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    path: tuple[str, str]
    higher_is_better: bool
    threshold: float
    decimals: int

    def value(self, payload: dict[str, Any]) -> float | None:
        parent = payload.get(self.path[0])
        value = parent.get(self.path[1]) if isinstance(parent, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None


METRICS = (
    Metric("piq_ctx", "Physics-IQ with context", ("physics_iq_with_context", "score"), True, 1.0, 2),
    Metric("piq_noctx", "Physics-IQ without context", ("physics_iq_without_context", "score"), True, 1.0, 2),
    Metric("pmf_ctx", "PMF with context", ("pmf_with_context", "score"), True, 0.02, 4),
    Metric("pmf_noctx", "PMF without context", ("pmf_without_context", "score"), True, 0.02, 4),
    Metric(
        "videophy2_sa",
        "VideoPhy2 semantic adherence (SA)",
        ("videophy2", "sa_score"),
        True,
        1.0,
        2,
    ),
    Metric(
        "videophy2_pc",
        "VideoPhy2 physical commonsense (PC)",
        ("videophy2", "pc_score"),
        True,
        1.0,
        2,
    ),
    Metric(
        "videophy2_joint",
        "VideoPhy2 joint pass",
        ("videophy2", "joint_pass"),
        True,
        1.0,
        0,
    ),
    Metric("wmreward", "WMReward surprise", ("wmreward", "surprise"), False, 0.002, 4),
    Metric(
        "subject",
        "VBench subject consistency",
        ("vbench_subject_consistency", "score"),
        True,
        0.002,
        4,
    ),
    Metric(
        "background",
        "VBench background consistency",
        ("vbench_background_consistency", "score"),
        True,
        0.002,
        4,
    ),
    Metric(
        "flicker",
        "VBench temporal flickering",
        ("vbench_temporal_flickering", "score"),
        True,
        0.0005,
        4,
    ),
    Metric(
        "smoothness",
        "VBench motion smoothness",
        ("vbench_motion_smoothness", "score"),
        True,
        0.0005,
        4,
    ),
    Metric(
        "dynamic",
        "VBench dynamic degree",
        ("vbench_dynamic_degree", "score"),
        True,
        0.05,
        4,
    ),
    Metric(
        "aesthetic",
        "VBench aesthetic quality",
        ("vbench_aesthetic_quality", "score"),
        True,
        0.01,
        4,
    ),
    Metric(
        "imaging",
        "VBench imaging quality",
        ("vbench_imaging_quality", "score"),
        True,
        0.01,
        4,
    ),
    Metric("cosmos", "Cosmos-Reason1", ("cosmos_reason1", "score"), True, 1.0, 2),
)


@dataclass(frozen=True)
class Comparison:
    comparison_id: str
    model: str
    mode: str
    baseline_dir: Path
    ablation_dir: Path

    @property
    def model_label(self) -> str:
        return MODEL_LABELS[self.model]

    @property
    def mode_label(self) -> str:
        return MODE_LABELS[self.mode]


def comparisons(root: Path) -> list[Comparison]:
    rvg_suffix = Path("physicIQ/physRVG_steps40_512x896_08_49f")
    return [
        Comparison(
            "wan_lora__whole_block",
            "wan_lora",
            "whole_block",
            root / "wan_lora/baseline",
            root / "wan_lora/whole_block_block17",
        ),
        Comparison(
            "wan_lora__self_attn_zero",
            "wan_lora",
            "self_attn_zero",
            root / "wan_lora/baseline",
            root / "wan_lora/self_attn_zero_block17",
        ),
        Comparison(
            "xssc__whole_block",
            "xssc",
            "whole_block",
            root / "xssc/baseline/results",
            root / "xssc/whole_block_block17/results",
        ),
        Comparison(
            "xssc__self_attn_zero",
            "xssc",
            "self_attn_zero",
            root / "xssc/baseline/results",
            root / "xssc/self_attn_zero_block17/results",
        ),
        Comparison(
            "xssc__object_cross_attn",
            "xssc",
            "object_cross_attn",
            root / "xssc/baseline/results",
            root / "xssc/object_cross_attn_block17/results",
        ),
        Comparison(
            "physrvg__whole_block",
            "physrvg",
            "whole_block",
            root / "PhyRVG/baseline" / rvg_suffix,
            root / "PhyRVG/whole_block_block17" / rvg_suffix,
        ),
        Comparison(
            "physrvg__self_attn_zero",
            "physrvg",
            "self_attn_zero",
            root / "PhyRVG/baseline" / rvg_suffix,
            root / "PhyRVG/self_attn_zero_block17" / rvg_suffix,
        ),
        Comparison(
            "physrvg__text_cross_attn_zero",
            "physrvg",
            "text_cross_attn_zero",
            root / "PhyRVG/baseline" / rvg_suffix,
            root / "PhyRVG/text_cross_attn_zero_block17" / rvg_suffix,
        ),
        Comparison(
            "physrvg__ffn_zero",
            "physrvg",
            "ffn_zero",
            root / "PhyRVG/baseline" / rvg_suffix,
            root / "PhyRVG/ffn_zero_block17" / rvg_suffix,
        ),
        Comparison(
            "physrvg__lora_off",
            "physrvg",
            "lora_off",
            root / "PhyRVG/baseline" / rvg_suffix,
            root / "PhyRVG/lora_off_block17" / rvg_suffix,
        ),
    ]


def load_cases(root: Path) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.json")):
        if path.name.startswith("eval_summary_"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        input_json = payload.get("input_json")
        if not isinstance(input_json, str):
            continue
        loaded[Path(input_json).stem] = payload
    return loaded


def numeric_direction(delta: float, epsilon: float = 1.0e-12) -> str:
    if delta > epsilon:
        return "up"
    if delta < -epsilon:
        return "down"
    return "same"


def quality_direction(quality_delta: float, threshold: float) -> str:
    if quality_delta >= threshold:
        return "improved"
    if quality_delta <= -threshold:
        return "declined"
    return "neutral"


def web_path(path_value: Any, result_root: Path) -> str | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    path = Path(path_value).expanduser()
    try:
        return "v2v_wan/" + path.resolve().relative_to(result_root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return "testdataset/" + path.resolve().relative_to(TESTDATASET_ROOT.resolve()).as_posix()
    except ValueError:
        return None


def case_metadata(
    baseline: dict[str, Any],
    ablation: dict[str, Any],
    result_root: Path,
) -> dict[str, Any]:
    prompt = baseline.get("input_caption") or baseline.get("input_prompt") or ""
    source = baseline.get("source_video")
    if not source:
        source = baseline.get("input_video")
    return {
        "case": Path(str(baseline["input_json"])).stem,
        "input_json": baseline.get("input_json"),
        "prompt": prompt,
        "source_video": web_path(source, result_root),
        "baseline_video": web_path(baseline.get("output_video"), result_root),
        "ablation_video": web_path(ablation.get("output_video"), result_root),
    }


def build_payload(result_root: Path) -> dict[str, Any]:
    comparison_payloads: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_featured: list[dict[str, Any]] = []

    for comparison in comparisons(result_root):
        baseline_cases = load_cases(comparison.baseline_dir)
        ablation_cases = load_cases(comparison.ablation_dir)
        shared_cases = sorted(set(baseline_cases) & set(ablation_cases))
        if len(shared_cases) != 67:
            raise RuntimeError(
                f"{comparison.comparison_id}: expected 67 shared cases, got {len(shared_cases)}"
            )

        summary_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        cases_payload: dict[str, dict[str, Any]] = {}
        for case in shared_cases:
            baseline = baseline_cases[case]
            ablation = ablation_cases[case]
            cases_payload[case] = case_metadata(baseline, ablation, result_root)
            for metric in METRICS:
                baseline_value = metric.value(baseline)
                ablation_value = metric.value(ablation)
                if baseline_value is None or ablation_value is None:
                    continue
                delta = ablation_value - baseline_value
                quality_delta = delta if metric.higher_is_better else -delta
                row = {
                    "comparison_id": comparison.comparison_id,
                    "model": comparison.model,
                    "model_label": comparison.model_label,
                    "mode": comparison.mode,
                    "mode_label": comparison.mode_label,
                    "case": case,
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "higher_is_better": metric.higher_is_better,
                    "threshold": metric.threshold,
                    "decimals": metric.decimals,
                    "baseline": baseline_value,
                    "ablation": ablation_value,
                    "delta": delta,
                    "quality_delta": quality_delta,
                    "normalized_quality_delta": quality_delta / metric.threshold,
                    "numeric_direction": numeric_direction(delta),
                    "quality_direction": quality_direction(quality_delta, metric.threshold),
                }
                comparison_rows.append(row)
                all_rows.append(row)

        for metric in METRICS:
            rows = [row for row in comparison_rows if row["metric"] == metric.key]
            summary_rows.append(
                {
                    "metric": metric.key,
                    "metric_label": metric.label,
                    "higher_is_better": metric.higher_is_better,
                    "threshold": metric.threshold,
                    "num_cases": len(rows),
                    "numeric_up": sum(row["numeric_direction"] == "up" for row in rows),
                    "numeric_down": sum(row["numeric_direction"] == "down" for row in rows),
                    "numeric_same": sum(row["numeric_direction"] == "same" for row in rows),
                    "quality_improved": sum(row["quality_direction"] == "improved" for row in rows),
                    "quality_declined": sum(row["quality_direction"] == "declined" for row in rows),
                    "quality_neutral": sum(row["quality_direction"] == "neutral" for row in rows),
                    "mean_delta": sum(row["delta"] for row in rows) / max(1, len(rows)),
                }
            )

        ranked = sorted(
            comparison_rows,
            key=lambda row: row["normalized_quality_delta"],
        )
        featured_rows: list[dict[str, Any]] = []
        for role, candidates in (
            ("strongest_decline", ranked),
            ("strongest_improvement", list(reversed(ranked))),
        ):
            selected = next(
                (
                    row
                    for row in candidates
                    if abs(row["normalized_quality_delta"]) >= 1.0
                ),
                candidates[0],
            )
            featured = {
                **selected,
                **cases_payload[selected["case"]],
                "role": role,
            }
            featured_rows.append(featured)
            all_featured.append(featured)

        comparison_payloads.append(
            {
                "comparison_id": comparison.comparison_id,
                "model": comparison.model,
                "model_label": comparison.model_label,
                "mode": comparison.mode,
                "mode_label": comparison.mode_label,
                "num_cases": len(shared_cases),
                "summary": summary_rows,
                "featured": featured_rows,
            }
        )

    return {
        "block": 17,
        "num_comparisons": len(comparison_payloads),
        "num_metrics": len(METRICS),
        "metrics": [
            {
                "key": metric.key,
                "label": metric.label,
                "higher_is_better": metric.higher_is_better,
                "threshold": metric.threshold,
                "decimals": metric.decimals,
            }
            for metric in METRICS
        ],
        "comparisons": comparison_payloads,
        "featured": all_featured,
        "rows": all_rows,
        "notes": {
            "delta": "ablation - baseline",
            "quality_delta": "delta for higher-is-better metrics; -delta for WMReward surprise",
            "video_phy2": (
                "All comparisons use generated_only_sa_pc_joint; joint_pass means "
                "SA>=4 and PC>=4."
            ),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "comparison_id",
        "model",
        "mode",
        "case",
        "metric",
        "metric_label",
        "higher_is_better",
        "threshold",
        "baseline",
        "ablation",
        "delta",
        "quality_delta",
        "normalized_quality_delta",
        "numeric_direction",
        "quality_direction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def ensure_link(path: Path, target: Path) -> None:
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"refusing to replace existing non-symlink: {path}")
    os.symlink(target, path, target_is_directory=True)


def page_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block 17 Case Metric Audit</title>
<style>
:root {{
  color-scheme: light;
  --ink:#202522; --muted:#68706b; --line:#d8ddd9; --paper:#f7f8f6;
  --panel:#fff; --good:#147a4b; --bad:#b43a32; --blue:#275f9b; --amber:#9a6414;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 Arial,sans-serif; }}
header {{ position:sticky; top:0; z-index:5; background:#fff; border-bottom:1px solid var(--line); }}
.bar {{ max-width:1900px; margin:auto; padding:14px 20px; display:flex; gap:18px; align-items:center; flex-wrap:wrap; }}
h1 {{ margin:0; font-size:22px; }}
.meta {{ color:var(--muted); }}
main {{ max-width:1900px; margin:auto; padding:18px 20px 50px; }}
h2 {{ margin:24px 0 10px; font-size:18px; }}
h3 {{ margin:0; font-size:15px; }}
.filters {{ display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 14px; }}
select,input {{ height:34px; border:1px solid #bfc6c1; background:#fff; padding:0 10px; border-radius:4px; color:var(--ink); }}
input {{ min-width:300px; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); background:#fff; }}
table {{ border-collapse:collapse; width:100%; min-width:1080px; }}
th,td {{ padding:8px 10px; border-bottom:1px solid #e6e9e6; text-align:right; white-space:nowrap; }}
th {{ position:sticky; top:0; background:#eef1ee; color:#3f4742; font-size:12px; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
.good {{ color:var(--good); font-weight:700; }}
.bad {{ color:var(--bad); font-weight:700; }}
.neutral {{ color:var(--muted); }}
.featured {{ display:grid; gap:14px; }}
.case {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.case-head {{ display:grid; grid-template-columns:minmax(260px,1fr) auto; gap:18px; padding:12px 14px; border-bottom:1px solid var(--line); }}
.case-title {{ overflow-wrap:anywhere; }}
.reason {{ text-align:right; }}
.prompt {{ color:#4d5650; margin-top:6px; }}
.videos {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; background:var(--line); }}
figure {{ margin:0; min-width:0; background:#fff; }}
video {{ width:100%; aspect-ratio:16/9; display:block; background:#111; object-fit:contain; }}
figcaption {{ padding:7px 10px; border-top:1px solid var(--line); }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); }}
.count {{ font-variant-numeric:tabular-nums; }}
.empty {{ padding:30px; text-align:center; color:var(--muted); background:#fff; border:1px solid var(--line); }}
@media (max-width:900px) {{
  .videos {{ grid-template-columns:1fr; }}
  .case-head {{ grid-template-columns:1fr; }}
  .reason {{ text-align:left; }}
  input {{ min-width:220px; width:100%; }}
}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>Block 17 Case Metric Audit</h1>
  <span class="meta">67 paired cases · 10 module ablations · 16 metrics</span>
</div></header>
<main>
  <div class="legend">
    <span class="good">绿色：质量改善</span>
    <span class="bad">红色：质量下降</span>
    <span>Delta = ablation - baseline；WMReward surprise 越低越好</span>
  </div>

  <h2>指标涨跌数量</h2>
  <div class="filters">
    <select id="summary-model"></select>
    <select id="summary-mode"></select>
  </div>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>模型 / 消融</th><th>指标</th>
      <th>数值上涨</th><th>数值下降</th><th>不变</th>
      <th>有效改善</th><th>有效下降</th><th>阈值内</th><th>平均Delta</th>
    </tr></thead>
    <tbody id="summary-body"></tbody>
  </table></div>

  <h2>典型Case</h2>
  <div class="filters">
    <select id="featured-model"></select>
    <select id="featured-mode"></select>
    <select id="featured-role">
      <option value="all">改善与下降</option>
      <option value="strongest_improvement">改善最大</option>
      <option value="strongest_decline">下降最大</option>
    </select>
  </div>
  <div id="featured" class="featured"></div>

  <h2>全部Case逐指标变化</h2>
  <div class="filters">
    <select id="detail-model"></select>
    <select id="detail-mode"></select>
    <select id="detail-metric"></select>
    <select id="detail-quality">
      <option value="all">全部质量方向</option>
      <option value="improved">有效改善</option>
      <option value="declined">有效下降</option>
      <option value="neutral">阈值内变化</option>
    </select>
    <input id="detail-search" placeholder="搜索case名称">
  </div>
  <div class="table-wrap"><table>
    <thead><tr>
      <th>Case</th><th>模型 / 消融</th><th>指标</th>
      <th>Baseline</th><th>Ablation</th><th>Delta</th>
      <th>数值方向</th><th>质量方向</th>
    </tr></thead>
    <tbody id="detail-body"></tbody>
  </table></div>
</main>
<script>
const DATA={data};
const comparisonById=Object.fromEntries(DATA.comparisons.map(x=>[x.comparison_id,x]));
const metricByKey=Object.fromEntries(DATA.metrics.map(x=>[x.key,x]));

function fmt(value, decimals) {{
  const n=Number(value);
  return Number.isFinite(n) ? n.toFixed(decimals) : "NA";
}}
function signed(value, decimals) {{
  const n=Number(value);
  return Number.isFinite(n) ? `${{n>=0?"+":""}}${{n.toFixed(decimals)}}` : "NA";
}}
function qualityClass(value) {{
  return value==="improved" ? "good" : value==="declined" ? "bad" : "neutral";
}}
function qualityText(value) {{
  return value==="improved" ? "改善" : value==="declined" ? "下降" : "阈值内";
}}
function numericText(value) {{
  return value==="up" ? "上涨" : value==="down" ? "下降" : "不变";
}}
function optionHtml(values, allLabel) {{
  return `<option value="all">${{allLabel}}</option>`+
    values.map(([value,label])=>`<option value="${{value}}">${{label}}</option>`).join("");
}}
const models=[...new Map(DATA.comparisons.map(x=>[x.model,x.model_label])).entries()];
const modes=[...new Map(DATA.comparisons.map(x=>[x.mode,x.mode_label])).entries()];
for (const id of ["summary-model","featured-model","detail-model"]) {{
  document.getElementById(id).innerHTML=optionHtml(models,"全部模型");
}}
for (const id of ["summary-mode","featured-mode","detail-mode"]) {{
  document.getElementById(id).innerHTML=optionHtml(modes,"全部模块");
}}
document.getElementById("detail-metric").innerHTML=
  optionHtml(DATA.metrics.map(x=>[x.key,x.label]),"全部指标");

function selected(id) {{ return document.getElementById(id).value; }}
function matchesComparison(item, model, mode) {{
  return (model==="all" || item.model===model) && (mode==="all" || item.mode===mode);
}}
function renderSummary() {{
  const model=selected("summary-model"), mode=selected("summary-mode");
  const rows=[];
  for (const cmp of DATA.comparisons.filter(x=>matchesComparison(x,model,mode))) {{
    for (const stat of cmp.summary) {{
      const metric=metricByKey[stat.metric];
      rows.push(`<tr>
        <td>${{cmp.model_label}} · ${{cmp.mode_label}}</td><td>${{stat.metric_label}}</td>
        <td class="count">${{stat.numeric_up}}</td><td class="count">${{stat.numeric_down}}</td><td class="count">${{stat.numeric_same}}</td>
        <td class="good count">${{stat.quality_improved}}</td><td class="bad count">${{stat.quality_declined}}</td><td class="neutral count">${{stat.quality_neutral}}</td>
        <td>${{signed(stat.mean_delta,metric.decimals)}}</td>
      </tr>`);
    }}
  }}
  document.getElementById("summary-body").innerHTML=rows.join("");
}}
function videoFigure(src,label) {{
  if (!src) return `<figure><div class="empty">视频路径不在展示根目录</div><figcaption>${{label}}</figcaption></figure>`;
  return `<figure><video controls muted preload="metadata" src="${{src}}"></video><figcaption>${{label}}</figcaption></figure>`;
}}
function renderFeatured() {{
  const model=selected("featured-model"), mode=selected("featured-mode"), role=selected("featured-role");
  const items=DATA.featured.filter(x=>matchesComparison(x,model,mode) && (role==="all" || x.role===role));
  document.getElementById("featured").innerHTML=items.map(item=>{{
    const metric=metricByKey[item.metric];
    const cls=qualityClass(item.quality_direction);
    const roleLabel=item.role==="strongest_improvement" ? "改善最大" : "下降最大";
    return `<article class="case">
      <div class="case-head">
        <div><h3 class="case-title">${{item.case}}</h3><div>${{item.model_label}} · ${{item.mode_label}}</div><div class="prompt">${{item.prompt||""}}</div></div>
        <div class="reason"><div>${{roleLabel}} · ${{item.metric_label}}</div>
          <div class="${{cls}}">${{fmt(item.baseline,metric.decimals)}} → ${{fmt(item.ablation,metric.decimals)}} (${{signed(item.delta,metric.decimals)}})</div>
        </div>
      </div>
      <div class="videos">
        ${{videoFigure(item.source_video,"GT / source")}}
        ${{videoFigure(item.baseline_video,"Baseline")}}
        ${{videoFigure(item.ablation_video,item.mode_label)}}
      </div>
    </article>`;
  }}).join("") || `<div class="empty">当前筛选条件下没有case</div>`;
}}
function renderDetails() {{
  const model=selected("detail-model"), mode=selected("detail-mode"), metricKey=selected("detail-metric");
  const quality=selected("detail-quality"), query=document.getElementById("detail-search").value.trim().toLowerCase();
  const rows=DATA.rows.filter(row=>
    matchesComparison(row,model,mode) &&
    (metricKey==="all" || row.metric===metricKey) &&
    (quality==="all" || row.quality_direction===quality) &&
    (!query || row.case.toLowerCase().includes(query))
  ).sort((a,b)=>Math.abs(b.normalized_quality_delta)-Math.abs(a.normalized_quality_delta)).slice(0,1000);
  document.getElementById("detail-body").innerHTML=rows.map(row=>{{
    const metric=metricByKey[row.metric], cls=qualityClass(row.quality_direction);
    return `<tr>
      <td>${{row.case}}</td><td>${{row.model_label}} · ${{row.mode_label}}</td><td>${{row.metric_label}}</td>
      <td>${{fmt(row.baseline,metric.decimals)}}</td><td>${{fmt(row.ablation,metric.decimals)}}</td>
      <td class="${{cls}}">${{signed(row.delta,metric.decimals)}}</td>
      <td>${{numericText(row.numeric_direction)}}</td><td class="${{cls}}">${{qualityText(row.quality_direction)}}</td>
    </tr>`;
  }}).join("");
}}
for (const id of ["summary-model","summary-mode"]) document.getElementById(id).addEventListener("change",renderSummary);
for (const id of ["featured-model","featured-mode","featured-role"]) document.getElementById(id).addEventListener("change",renderFeatured);
for (const id of ["detail-model","detail-mode","detail-metric","detail-quality"]) document.getElementById(id).addEventListener("change",renderDetails);
document.getElementById("detail-search").addEventListener("input",renderDetails);
renderSummary(); renderFeatured(); renderDetails();
</script>
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "block17_case_metric_gallery",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(result_root)
    write_csv(output_dir / "case_metric_changes.csv", payload["rows"])
    (output_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(page_html(payload), encoding="utf-8")
    ensure_link(output_dir / "v2v_wan", result_root)
    ensure_link(output_dir / "testdataset", TESTDATASET_ROOT)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "index": str(output_dir / "index.html"),
                "csv": str(output_dir / "case_metric_changes.csv"),
                "comparisons": payload["num_comparisons"],
                "rows": len(payload["rows"]),
                "featured": len(payload["featured"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
