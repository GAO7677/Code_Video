#!/usr/bin/env python3
"""Build interactive Block 17 curves for Solid Mechanics cases only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GALLERY = (
    Path(__file__).resolve().parent / "block17_case_metric_gallery"
)


def is_solid_case(case: str) -> bool:
    normalized = case.lower().replace("-", "_").replace(" ", "_")
    return "solid_mechanics" in normalized


def build_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    solid_rows = [row for row in manifest["rows"] if is_solid_case(row["case"])]
    solid_cases = sorted({row["case"] for row in solid_rows})

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in solid_rows:
        grouped[(row["comparison_id"], row["metric"])].append(row)

    summary = []
    for comparison in manifest["comparisons"]:
        for metric in manifest["metrics"]:
            rows = grouped[(comparison["comparison_id"], metric["key"])]
            if not rows:
                continue
            summary.append(
                {
                    "comparison_id": comparison["comparison_id"],
                    "model": comparison["model"],
                    "model_label": comparison["model_label"],
                    "mode": comparison["mode"],
                    "mode_label": comparison["mode_label"],
                    "metric": metric["key"],
                    "metric_label": metric["label"],
                    "mean_quality_delta": sum(row["quality_delta"] for row in rows)
                    / len(rows),
                    "mean_normalized_quality_delta": sum(
                        row["normalized_quality_delta"] for row in rows
                    )
                    / len(rows),
                    "improved": sum(
                        row["quality_direction"] == "improved" for row in rows
                    ),
                    "declined": sum(
                        row["quality_direction"] == "declined" for row in rows
                    ),
                    "neutral": sum(
                        row["quality_direction"] == "neutral" for row in rows
                    ),
                }
            )

    return {
        "block": 17,
        "num_cases": len(solid_cases),
        "cases": solid_cases,
        "metrics": manifest["metrics"],
        "comparisons": [
            {
                key: comparison[key]
                for key in (
                    "comparison_id",
                    "model",
                    "model_label",
                    "mode",
                    "mode_label",
                )
            }
            for comparison in manifest["comparisons"]
        ],
        "summary": summary,
        "rows": solid_rows,
    }


def page_html(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Block 17 Solid Mechanics Curves</title>
<style>
:root {{
  color-scheme:light; --ink:#202522; --muted:#68706b; --line:#d8ddd9;
  --paper:#f7f8f6; --panel:#fff; --green:#147a4b; --red:#b43a32;
  --blue:#2866a6; --orange:#c66a20; --violet:#7651a8; --cyan:#16838b;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font:14px/1.45 Arial,sans-serif; }}
header {{ position:sticky; top:0; z-index:4; background:#fff; border-bottom:1px solid var(--line); }}
.bar {{ max-width:1800px; margin:auto; padding:13px 20px; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }}
h1 {{ margin:0; font-size:21px; }}
a {{ color:var(--blue); }}
main {{ max-width:1800px; margin:auto; padding:18px 20px 48px; }}
h2 {{ margin:22px 0 9px; font-size:17px; }}
.note {{ color:var(--muted); }}
.filters {{ display:flex; gap:10px; flex-wrap:wrap; margin:10px 0; }}
select {{ height:34px; border:1px solid #bcc4bf; border-radius:4px; background:#fff; padding:0 10px; }}
.chart {{ background:#fff; border:1px solid var(--line); min-height:390px; overflow:auto; }}
svg {{ width:100%; min-width:920px; height:390px; display:block; }}
.grid {{ stroke:#e4e8e5; stroke-width:1; }}
.zero {{ stroke:#4d5650; stroke-width:1.4; stroke-dasharray:5 4; }}
.axis {{ fill:#606963; font-size:11px; }}
.line {{ fill:none; stroke-width:2.2; }}
.dot {{ stroke:#fff; stroke-width:1.2; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 12px; }}
.legend span::before {{ content:""; display:inline-block; width:18px; height:3px; margin:0 6px 3px 0; background:var(--c); }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); background:#fff; max-height:520px; }}
table {{ border-collapse:collapse; width:100%; min-width:1050px; }}
th,td {{ padding:7px 10px; border-bottom:1px solid #e7eae8; text-align:right; white-space:nowrap; }}
th {{ position:sticky; top:0; background:#eef1ee; font-size:12px; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align:left; }}
.good {{ color:var(--green); font-weight:700; }}
.bad {{ color:var(--red); font-weight:700; }}
.summary {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); border:1px solid var(--line); background:#fff; }}
.summary div {{ padding:10px 12px; border-right:1px solid var(--line); }}
.summary div:last-child {{ border-right:0; }}
.summary strong {{ display:block; font-size:20px; }}
@media(max-width:760px) {{
  .summary {{ grid-template-columns:1fr; }}
  .summary div {{ border-right:0; border-bottom:1px solid var(--line); }}
}}
</style>
</head>
<body>
<header><div class="bar">
  <h1>Block 17 · Solid Mechanics Curves</h1>
  <span class="note">39 / 67 cases</span>
  <a href="./">返回全部 case 页面</a>
</div></header>
<main>
  <p class="note">Quality Δ 已统一为越高越好：普通指标为 ablation − baseline，WMReward surprise 为 baseline − ablation。阈值归一化值 +1/−1 表示平均变化达到一个有效变化阈值。</p>

  <h2>Solid-only 模块影响曲线</h2>
  <div class="filters"><select id="summary-model"></select></div>
  <div id="summary-legend" class="legend"></div>
  <div id="summary-chart" class="chart"></div>

  <h2>逐 case 原始分数曲线</h2>
  <div class="filters">
    <select id="case-model"></select>
    <select id="case-mode"></select>
    <select id="case-metric"></select>
  </div>
  <div class="legend">
    <span style="--c:var(--blue)">Baseline</span>
    <span style="--c:var(--orange)">Ablation</span>
  </div>
  <div id="case-chart" class="chart"></div>

  <h2>逐 case 质量变化曲线</h2>
  <div class="legend">
    <span style="--c:var(--green)">改善</span>
    <span style="--c:var(--red)">下降</span>
    <span style="--c:#4d5650">零变化</span>
  </div>
  <div id="delta-chart" class="chart"></div>
  <div id="counts" class="summary"></div>

  <h2>39 个 Solid case 明细</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>编号</th><th>Case</th><th>Baseline</th><th>Ablation</th><th>数值 Δ</th><th>Quality Δ</th><th>质量方向</th></tr></thead>
    <tbody id="case-body"></tbody>
  </table></div>
</main>
<script>
const DATA={data};
const COLORS=["#2866a6","#c66a20","#7651a8","#16838b","#b64f68"];
const metricByKey=Object.fromEntries(DATA.metrics.map(x=>[x.key,x]));
const models=[...new Map(DATA.comparisons.map(x=>[x.model,x.model_label])).entries()];
const cases=DATA.cases;

function options(items) {{
  return items.map(([v,l])=>`<option value="${{v}}">${{l}}</option>`).join("");
}}
document.getElementById("summary-model").innerHTML=options(models);
document.getElementById("case-model").innerHTML=options(models);
document.getElementById("case-metric").innerHTML=options(DATA.metrics.map(x=>[x.key,x.label]));

function fmt(v,d=4) {{
  return Number.isFinite(Number(v)) ? Number(v).toFixed(d) : "NA";
}}
function signed(v,d=4) {{
  const n=Number(v); return Number.isFinite(n) ? `${{n>=0?"+":""}}${{n.toFixed(d)}}` : "NA";
}}
function escapeXml(s) {{
  return String(s).replace(/[&<>"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&apos;"}}[c]));
}}
function pathFor(values,x,y) {{
  return values.map((v,i)=>`${{i?"L":"M"}} ${{x(i)}} ${{y(v)}}`).join(" ");
}}
function chartSvg(series, labels, options={{}}) {{
  const width=1500,height=390,left=66,right=24,top=24,bottom=78;
  const values=series.flatMap(s=>s.values).filter(Number.isFinite);
  let min=options.min ?? Math.min(...values), max=options.max ?? Math.max(...values);
  if (options.includeZero) {{ min=Math.min(0,min); max=Math.max(0,max); }}
  if (max===min) {{ max+=1; min-=1; }}
  const pad=(max-min)*0.08; min-=pad; max+=pad;
  const x=i=>left+(labels.length===1?0:(width-left-right)*i/(labels.length-1));
  const y=v=>top+(height-top-bottom)*(max-v)/(max-min);
  let body="";
  for(let i=0;i<=5;i++) {{
    const v=min+(max-min)*i/5, yy=y(v);
    body+=`<line class="grid" x1="${{left}}" y1="${{yy}}" x2="${{width-right}}" y2="${{yy}}"/><text class="axis" x="${{left-8}}" y="${{yy+4}}" text-anchor="end">${{v.toFixed(options.decimals??2)}}</text>`;
  }}
  if(min<0&&max>0) body+=`<line class="zero" x1="${{left}}" y1="${{y(0)}}" x2="${{width-right}}" y2="${{y(0)}}"/>`;
  const labelStep=Math.max(1,Math.ceil(labels.length/20));
  labels.forEach((label,i)=>{{
    if(i%labelStep===0||i===labels.length-1) body+=`<text class="axis" x="${{x(i)}}" y="${{height-bottom+18}}" text-anchor="end" transform="rotate(-38 ${{x(i)}} ${{height-bottom+18}})">${{escapeXml(label)}}</text>`;
  }});
  series.forEach((s,si)=>{{
    const color=s.color||COLORS[si%COLORS.length];
    body+=`<path class="line" stroke="${{color}}" d="${{pathFor(s.values,x,y)}}"/>`;
    s.values.forEach((v,i)=>{{
      const tip=`${{labels[i]}} | ${{s.label}}: ${{fmt(v,4)}}`;
      body+=`<circle class="dot" fill="${{color}}" cx="${{x(i)}}" cy="${{y(v)}}" r="3.1"><title>${{escapeXml(tip)}}</title></circle>`;
    }});
  }});
  return `<svg viewBox="0 0 ${{width}} ${{height}}" role="img">${{body}}</svg>`;
}}
function selected(id) {{ return document.getElementById(id).value; }}
function comparisonOptions(model) {{
  const items=DATA.comparisons.filter(x=>x.model===model).map(x=>[x.comparison_id,x.mode_label]);
  document.getElementById("case-mode").innerHTML=options(items);
}}
function shortMetric(label) {{
  return label.replace("Physics-IQ ","PIQ ").replace("VBench ","").replace(" consistency","").replace(" quality","");
}}
function renderSummary() {{
  const model=selected("summary-model");
  const comparisons=DATA.comparisons.filter(x=>x.model===model);
  const labels=DATA.metrics.map(x=>shortMetric(x.label));
  const series=comparisons.map((cmp,i)=>({{
    label:cmp.mode_label,
    color:COLORS[i%COLORS.length],
    values:DATA.metrics.map(metric=>{{
      const row=DATA.summary.find(x=>x.comparison_id===cmp.comparison_id&&x.metric===metric.key);
      return row?.mean_normalized_quality_delta ?? 0;
    }})
  }}));
  document.getElementById("summary-legend").innerHTML=series.map(s=>`<span style="--c:${{s.color}}">${{s.label}}</span>`).join("");
  document.getElementById("summary-chart").innerHTML=chartSvg(series,labels,{{includeZero:true,decimals:1}});
}}
function selectedRows() {{
  const comparisonId=selected("case-mode"), metric=selected("case-metric");
  const byCase=new Map(DATA.rows.filter(x=>x.comparison_id===comparisonId&&x.metric===metric).map(x=>[x.case,x]));
  return cases.map(c=>byCase.get(c)).filter(Boolean);
}}
function renderCases() {{
  const rows=selectedRows(), metric=metricByKey[selected("case-metric")];
  const labels=rows.map((_,i)=>`S${{String(i+1).padStart(2,"0")}}`);
  document.getElementById("case-chart").innerHTML=chartSvg([
    {{label:"Baseline",color:"#2866a6",values:rows.map(x=>x.baseline)}},
    {{label:"Ablation",color:"#c66a20",values:rows.map(x=>x.ablation)}}
  ],labels,{{decimals:metric.decimals}});
  document.getElementById("delta-chart").innerHTML=chartSvg([
    {{label:"Quality Δ",color:"#147a4b",values:rows.map(x=>x.quality_delta)}}
  ],labels,{{includeZero:true,decimals:metric.decimals}});
  const counts={{
    improved:rows.filter(x=>x.quality_direction==="improved").length,
    declined:rows.filter(x=>x.quality_direction==="declined").length,
    neutral:rows.filter(x=>x.quality_direction==="neutral").length
  }};
  document.getElementById("counts").innerHTML=`<div><strong class="good">${{counts.improved}}</strong>有效改善</div><div><strong class="bad">${{counts.declined}}</strong>有效下降</div><div><strong>${{counts.neutral}}</strong>阈值内变化</div>`;
  document.getElementById("case-body").innerHTML=rows.map((row,i)=>{{
    const cls=row.quality_direction==="improved"?"good":row.quality_direction==="declined"?"bad":"";
    const text=row.quality_direction==="improved"?"改善":row.quality_direction==="declined"?"下降":"阈值内";
    return `<tr><td>S${{String(i+1).padStart(2,"0")}}</td><td>${{row.case}}</td><td>${{fmt(row.baseline,metric.decimals)}}</td><td>${{fmt(row.ablation,metric.decimals)}}</td><td>${{signed(row.delta,metric.decimals)}}</td><td class="${{cls}}">${{signed(row.quality_delta,metric.decimals)}}</td><td class="${{cls}}">${{text}}</td></tr>`;
  }}).join("");
}}
document.getElementById("summary-model").addEventListener("change",renderSummary);
document.getElementById("case-model").addEventListener("change",()=>{{comparisonOptions(selected("case-model"));renderCases();}});
document.getElementById("case-mode").addEventListener("change",renderCases);
document.getElementById("case-metric").addEventListener("change",renderCases);
comparisonOptions(selected("case-model"));
renderSummary();
renderCases();
</script>
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gallery-dir", type=Path, default=DEFAULT_GALLERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gallery_dir = args.gallery_dir.expanduser().resolve()
    manifest_path = gallery_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = build_payload(manifest)
    output_path = gallery_dir / "solid_curves.html"
    output_path.write_text(page_html(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "solid_cases": payload["num_cases"],
                "rows": len(payload["rows"]),
                "summary_rows": len(payload["summary"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
