#!/usr/bin/env python3
"""Build an interactive PhysicIQ regression audit against PhysRVG references."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_SOURCE_PAGE = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq/index.html"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq-vs-physrvg-worst-cases"
)

REFERENCE_KEYS = {
    "off": "physrvg_test5_lora_off",
    "on": "physrvg_test5_lora_on",
}
PRIMARY_METRIC_KEYS = [
    "videophy2_pc_raw",
    "cosmos_reason1",
    "physics_iq_with_context",
    "physics_iq_without_context",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-page", type=Path, default=DEFAULT_SOURCE_PAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_dashboard_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\s*const caseSelect=", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find dashboard payload in {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dashboard object in {path}")
    return payload


def is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def record_case_is_complete(
    record: dict[str, Any], stem: str, metric_keys: list[str]
) -> bool:
    values = record.get("metrics", {}).get(stem, {})
    return stem in record.get("videos", {}) and all(
        is_number(values.get(key)) for key in metric_keys
    )


def compact_record(
    record: dict[str, Any],
    metric_keys: list[str],
    case_stems: set[str],
) -> dict[str, Any]:
    metrics: dict[str, dict[str, float]] = {}
    videos: dict[str, str] = {}
    for stem in sorted(case_stems):
        source_values = record.get("metrics", {}).get(stem, {})
        values = {
            key: float(source_values[key])
            for key in metric_keys
            if is_number(source_values.get(key))
        }
        if values:
            metrics[stem] = values
        video = record.get("videos", {}).get(stem)
        if isinstance(video, str) and video:
            videos[stem] = video
    return {
        "method_key": str(record["method_key"]),
        "method_label": str(record["method_label"]),
        "step": int(record["step"]),
        "checkpoint_dir": str(record.get("checkpoint_dir", "")),
        "origin": str(record.get("origin", "")),
        "videos": videos,
        "metrics": metrics,
    }


def build_data(payload: dict[str, Any]) -> dict[str, Any]:
    metric_specs = [
        {
            "key": str(spec["key"]),
            "label": str(spec["label"]),
            "direction": str(spec.get("direction", "higher")),
            "primary": str(spec["key"]) in PRIMARY_METRIC_KEYS,
        }
        for spec in payload["metricSpecs"]
    ]
    metric_keys = [spec["key"] for spec in metric_specs]
    missing_primary = set(PRIMARY_METRIC_KEYS) - set(metric_keys)
    if missing_primary:
        raise ValueError(f"Missing primary metric specs: {sorted(missing_primary)}")

    cases = [
        {
            "stem": str(case["stem"]),
            "prompt": str(case.get("prompt", "")),
            "gt": str(case["gt"]),
            "context": str(case.get("context", "")),
            "solid": "Solid_Mechanics" in str(case["stem"]),
        }
        for case in payload["cases"]
    ]
    case_stems = {case["stem"] for case in cases}
    method_meta = {
        str(method["key"]): {
            "key": str(method["key"]),
            "label": str(method["label"]),
            "color": str(method.get("color", "#52636d")),
        }
        for method in payload["methods"]
    }

    records_by_key: dict[str, list[dict[str, Any]]] = {}
    reference_records: dict[str, dict[str, Any]] = {}
    for record in payload["records"]:
        key = str(record["method_key"])
        compact = compact_record(record, metric_keys, case_stems)
        if key == REFERENCE_KEYS["off"]:
            reference_records["off"] = compact
            continue
        if key == REFERENCE_KEYS["on"]:
            reference_records["on"] = compact
            continue
        primary_count = sum(
            record_case_is_complete(record, stem, PRIMARY_METRIC_KEYS)
            for stem in case_stems
        )
        if primary_count == 0:
            continue
        compact["primary_complete_cases"] = primary_count
        records_by_key.setdefault(key, []).append(compact)

    if set(reference_records) != {"off", "on"}:
        raise ValueError(
            f"Expected PhysRVG OFF/ON references, found {sorted(reference_records)}"
        )
    for name, record in reference_records.items():
        incomplete = [
            stem
            for stem in case_stems
            if not record_case_is_complete(record, stem, PRIMARY_METRIC_KEYS)
        ]
        if incomplete:
            raise ValueError(
                f"PhysRVG reference {name} missing primary data for {len(incomplete)} cases"
            )

    methods: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for key, method_records in records_by_key.items():
        method_records.sort(key=lambda item: int(item["step"]))
        full_records = [
            record
            for record in method_records
            if int(record["primary_complete_cases"]) == len(cases)
        ]
        default_record = full_records[-1] if full_records else method_records[-1]
        meta = method_meta.get(
            key,
            {"key": key, "label": method_records[-1]["method_label"], "color": "#52636d"},
        )
        methods.append(
            {
                **meta,
                "steps": [int(record["step"]) for record in method_records],
                "default_step": int(default_record["step"]),
            }
        )
        records.extend(method_records)

    methods.sort(key=lambda item: str(item["label"]).lower())
    records.sort(key=lambda item: (str(item["method_key"]), int(item["step"])))
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "primary_metric_keys": PRIMARY_METRIC_KEYS,
        "metric_specs": metric_specs,
        "cases": cases,
        "methods": methods,
        "records": records,
        "references": reference_records,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PhysicIQ · PhysRVG 最大劣势 case</title>
  <style>
    :root{--navy:#102733;--navy2:#173b49;--paper:#f9fbfb;--mist:#e8eef0;--ink:#18272d;
      --muted:#63747b;--line:#cbd6da;--rust:#b64b36;--rust2:#7f2d23;--teal:#0b6e75;
      --gold:#d59b27;--off:#315c87;--on:#0b6e4f;--shadow:0 8px 28px rgba(16,39,51,.09)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--mist);
      color:var(--ink);font-family:Inter,"Noto Sans SC",Arial,sans-serif}
    a{color:inherit}.hero{padding:28px clamp(18px,4vw,60px) 24px;color:#f7fbfc;
      background:linear-gradient(112deg,var(--navy) 0 68%,var(--navy2) 68% 100%)}
    .hero-top{display:flex;align-items:center;gap:14px}.back{text-decoration:none;font-weight:850;
      color:#b9dfe2}.eyebrow{font:700 11px/1.2 "Arial Narrow",sans-serif;letter-spacing:.15em;
      text-transform:uppercase;color:#87c6ca}.hero h1{max-width:980px;margin:17px 0 8px;
      font:800 clamp(28px,4.2vw,56px)/.98 "Arial Narrow","Roboto Condensed",sans-serif;
      letter-spacing:-.025em}.hero p{max-width:940px;margin:0;color:#c3d7dc;font-size:13px;line-height:1.65}
    .hero-rule{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:23px}
    .hero-rule span{height:5px;background:#335966}.hero-rule span:nth-child(1){background:#e2ad3f}
    .hero-rule span:nth-child(2){background:#ce6d50}.hero-rule span:nth-child(3){background:#4d8ea0}
    .hero-rule span:nth-child(4){background:#5aa077}
    .controls{position:sticky;top:0;z-index:20;display:grid;
      grid-template-columns:minmax(220px,1.5fr) 105px minmax(190px,1fr) minmax(180px,1fr) 110px;
      gap:9px;padding:11px clamp(12px,3vw,38px);background:rgba(249,251,251,.97);
      border-bottom:1px solid var(--line);box-shadow:0 5px 18px rgba(16,39,51,.08)}
    label{display:grid;gap:4px;color:var(--muted);font-size:10px;font-weight:850;letter-spacing:.04em}
    select{width:100%;height:38px;padding:0 9px;border:1px solid var(--line);border-radius:4px;
      background:#fff;color:var(--ink);font:700 12px/1.2 inherit}select:focus-visible,button:focus-visible,
      a:focus-visible{outline:3px solid #e6b64b;outline-offset:2px}
    main{max-width:1800px;margin:auto;padding:21px clamp(12px,3vw,38px) 70px}
    .section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;margin:9px 0 11px}
    .section-head h2{margin:0;font:800 22px/1.05 "Arial Narrow",sans-serif}.section-head p{
      max-width:760px;margin:0;color:var(--muted);font-size:12px;line-height:1.5;text-align:right}
    .metric-lanes{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:10px}
    .lane{background:var(--paper);border:1px solid var(--line);border-top:5px solid var(--rust);
      box-shadow:var(--shadow)}.lane h3{margin:0;padding:11px 12px;border-bottom:1px solid var(--line);
      font-size:13px}.lane ol{list-style:none;margin:0;padding:0}.lane li{display:grid;
      grid-template-columns:26px minmax(0,1fr) auto;gap:7px;align-items:center;padding:9px 11px;
      border-bottom:1px solid #e2e8ea;cursor:pointer}.lane li:last-child{border-bottom:0}
    .lane li:hover{background:#fff7ef}.rank{font:800 16px/1 "Arial Narrow",sans-serif;color:#93a0a5}
    .lane .who{min-width:0}.lane .method{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      font-size:11px;font-weight:850}.lane .case{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      color:var(--muted);font-size:10px}.gap-chip{padding:4px 6px;background:#fae5df;color:var(--rust2);
      border-radius:3px;font:850 12px/1 "Arial Narrow",sans-serif;white-space:nowrap}
    .overview{margin-top:24px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .table-wrap{overflow:auto;max-height:500px}table{width:100%;border-collapse:separate;border-spacing:0;
      font-size:11px;font-variant-numeric:tabular-nums}th,td{padding:8px 9px;border-right:1px solid #dce4e6;
      border-bottom:1px solid #dce4e6;vertical-align:top}th{position:sticky;top:0;z-index:2;
      background:#dfe8ea;color:#41545c;text-align:left}th:first-child,td:first-child{position:sticky;left:0;
      z-index:1;background:var(--paper);min-width:260px}th:first-child{z-index:3;background:#dfe8ea}
    td.metric-cell{min-width:220px;cursor:pointer}.metric-cell:hover{background:#fff7ef}
    .scheme-name{font-weight:900}.step{margin-top:2px;color:var(--muted);font-size:10px}
    .cell-gap{color:var(--rust2);font-weight:900}.cell-case{display:block;max-width:240px;margin-top:2px;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:10px}
    .audit{margin-top:27px}.audit-meta{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 12px}
    .audit-meta span{padding:5px 8px;border:1px solid var(--line);background:#f5f8f8;
      color:var(--muted);font-size:11px;font-weight:750}.cards{display:grid;gap:15px}
    .case-card{background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .case-head{display:grid;grid-template-columns:54px minmax(0,1fr) auto;gap:12px;align-items:start;
      padding:13px 15px;border-bottom:1px solid var(--line)}.case-index{font:900 30px/1 "Arial Narrow",sans-serif;
      color:#a6b2b6}.case-head h3{margin:0 0 4px;font-size:14px;overflow-wrap:anywhere}
    .case-head p{margin:0;color:var(--muted);font-size:11px;line-height:1.5}.gap-readout{
      min-width:130px;padding:7px 9px;border-left:4px solid var(--rust);background:#fae9e4}
    .gap-readout strong{display:block;color:var(--rust2);font:900 21px/1 "Arial Narrow",sans-serif}
    .gap-readout span{display:block;margin-top:4px;color:#74534c;font-size:9px;font-weight:850}
    .gap-rail{height:5px;background:#dfe5e7}.gap-rail span{display:block;height:100%;background:var(--rust)}
    .videos{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line)}
    .video-cell{min-width:0;padding:9px;background:var(--paper)}.video-label{display:flex;
      justify-content:space-between;gap:8px;min-height:32px;margin-bottom:6px;font-size:10px;font-weight:900}
    .video-label em{color:var(--muted);font-style:normal;font-weight:650;text-align:right}
    video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#0c1519}
    .metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);
      border-top:1px solid var(--line)}.metric-box{padding:10px 11px;background:#f6f9f9}
    .metric-box.focus{background:#fff4e6;box-shadow:inset 0 3px 0 var(--gold)}
    .metric-box h4{margin:0 0 7px;font-size:10px}.metric-box dl{display:grid;
      grid-template-columns:minmax(0,1fr) auto;gap:4px 8px;margin:0;font-size:10px}
    .metric-box dt{color:var(--muted)}.metric-box dd{margin:0;text-align:right;font-weight:850}
    .metric-box .bad{color:var(--rust2)}.empty{padding:30px;background:var(--paper);
      border:1px solid var(--line);color:var(--muted);text-align:center}
    .replay{position:fixed;right:22px;bottom:20px;z-index:30;height:48px;padding:0 17px;border:0;
      border-radius:24px;background:var(--rust);color:#fff;box-shadow:0 8px 24px rgba(127,45,35,.3);
      cursor:pointer;font-weight:900}.footer{margin-top:18px;color:var(--muted);font-size:10px}
    @media(max-width:1250px){.metric-lanes{grid-template-columns:repeat(2,1fr)}
      .videos,.metrics{grid-template-columns:repeat(2,1fr)}.controls{grid-template-columns:repeat(3,1fr)}}
    @media(max-width:720px){.hero{padding:20px 15px}.controls{position:relative;grid-template-columns:1fr 1fr}
      .controls label:first-child,.controls label:nth-child(4){grid-column:1/-1}.metric-lanes{grid-template-columns:1fr}
      .videos,.metrics{grid-template-columns:1fr}.case-head{grid-template-columns:42px 1fr}.gap-readout{
        grid-column:1/-1}.section-head{display:block}.section-head p{margin-top:6px;text-align:left}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-top"><a class="back" href="../">← 返回 8844 总览</a><span class="eyebrow">Physics regression audit</span></div>
    <h1>哪些 case 明显输给 PhysRVG？</h1>
    <p>覆盖所有已有方案。默认以同一 case、同一指标上 PhysRVG LoRA OFF / +LoRA 中表现更好者为参考；
      按指标方向计算原始劣势，不做归一化。负差值代表该方案反而优于参考，不列入“最大劣势”。</p>
    <div class="hero-rule"><span></span><span></span><span></span><span></span></div>
  </header>
  <nav class="controls" aria-label="筛选条件">
    <label>方案<select id="method"></select></label>
    <label>Step<select id="step"></select></label>
    <label>参考<select id="reference"><option value="series">PhysRVG 系列最佳</option>
      <option value="off">PhysRVG LoRA OFF</option><option value="on">PhysRVG +LoRA</option></select></label>
    <label>指标<select id="metric"></select></label>
    <label>范围<select id="scope"><option value="all">全部 67 case</option>
      <option value="solid">Solid Mechanics 39 case</option></select></label>
  </nav>
  <main>
    <section>
      <div class="section-head"><h2>四项主要指标 · 全局最大劣势</h2>
        <p>每条轨道使用各方案最新且主要指标完整的 step；点击任意条目进入对应方案与 case。</p></div>
      <div id="lanes" class="metric-lanes"></div>
    </section>
    <section class="overview">
      <div class="section-head" style="padding:14px 15px 3px"><h2>所有方案审计索引</h2>
        <p>每格展示该方案在对应主要指标上最差的 case 与原始劣势。</p></div>
      <div class="table-wrap"><table><thead id="overview-head"></thead><tbody id="overview-body"></tbody></table></div>
    </section>
    <section class="audit" id="audit">
      <div class="section-head"><h2 id="audit-title">最大劣势 case</h2>
        <p id="audit-note"></p></div>
      <div class="audit-meta"><span id="selection-method"></span><span id="selection-reference"></span>
        <span id="selection-scope"></span><label style="display:flex;align-items:center;gap:6px">显示
          <select id="topk" style="width:72px;height:29px"><option>3</option><option selected>5</option><option>10</option></select></label></div>
      <div id="cards" class="cards"></div>
    </section>
    <p class="footer">数据来自 PhysicIQ 合并页的逐 case 视频与正式指标；生成时间：__GENERATED__</p>
  </main>
  <button id="replay" class="replay" title="重新播放页面中所有视频">↺ 全部重播</button>
  <script>
    const D=__DATA__;
    const refs=D.references;
    const byStem=Object.fromEntries(D.cases.map(item=>[item.stem,item]));
    const methodSelect=document.getElementById('method');
    const stepSelect=document.getElementById('step');
    const referenceSelect=document.getElementById('reference');
    const metricSelect=document.getElementById('metric');
    const scopeSelect=document.getElementById('scope');
    const topkSelect=document.getElementById('topk');
    const specByKey=Object.fromEntries(D.metric_specs.map(spec=>[spec.key,spec]));
    const methodByKey=Object.fromEntries(D.methods.map(method=>[method.key,method]));
    const recordsByMethod={};
    D.records.forEach(record=>(recordsByMethod[record.method_key]??=[]).push(record));
    function fmt(v){if(!Number.isFinite(v))return '—';const a=Math.abs(v);return a>=10?v.toFixed(2):v.toFixed(3)}
    function casesInScope(){return D.cases.filter(item=>scopeSelect.value==='all'||item.solid)}
    function recordFor(methodKey,step){return (recordsByMethod[methodKey]||[]).find(r=>r.step===Number(step))}
    function defaultRecord(method){return recordFor(method.key,method.default_step)}
    function metricValue(record,stem,key){const value=record?.metrics?.[stem]?.[key];return Number.isFinite(value)?value:null}
    function referenceFor(stem,key,choice=referenceSelect.value){
      const spec=specByKey[key];const off=metricValue(refs.off,stem,key);const on=metricValue(refs.on,stem,key);
      if(choice==='off')return {key:'off',label:'PhysRVG LoRA OFF',value:off};
      if(choice==='on')return {key:'on',label:'PhysRVG +LoRA',value:on};
      if(off===null)return {key:'on',label:'PhysRVG +LoRA',value:on};
      if(on===null)return {key:'off',label:'PhysRVG LoRA OFF',value:off};
      const offBetter=spec.direction==='lower'?off<=on:off>=on;
      return offBetter?{key:'off',label:'PhysRVG LoRA OFF',value:off}:{key:'on',label:'PhysRVG +LoRA',value:on};
    }
    function comparison(record,item,key,choice=referenceSelect.value){
      const value=metricValue(record,item.stem,key);const ref=referenceFor(item.stem,key,choice);
      if(value===null||ref.value===null||!record.videos[item.stem])return null;
      const gap=specByKey[key].direction==='lower'?value-ref.value:ref.value-value;
      return {item,record,value,ref,gap};
    }
    function rankedForRecord(record,key,choice=referenceSelect.value){return casesInScope()
      .map(item=>comparison(record,item,key,choice)).filter(Boolean)
      .sort((a,b)=>b.gap-a.gap||a.item.stem.localeCompare(b.item.stem))}
    function latestRows(key,choice=referenceSelect.value){
      const rows=[];D.methods.forEach(method=>{const record=defaultRecord(method);if(!record)return;
        rankedForRecord(record,key,choice).forEach(row=>rows.push({...row,method}))});
      return rows.sort((a,b)=>b.gap-a.gap||a.method.label.localeCompare(b.method.label)||a.item.stem.localeCompare(b.item.stem));
    }
    function setMethod(methodKey,step,key,stem){methodSelect.value=methodKey;populateSteps();
      if(step!==undefined)stepSelect.value=String(step);metricSelect.value=key;renderAudit(stem);
      document.getElementById('audit').scrollIntoView({behavior:'smooth'});}
    function populateSteps(){const key=methodSelect.value;stepSelect.replaceChildren();
      if(key==='__all__'){stepSelect.add(new Option('各方案默认','default'));stepSelect.disabled=true;return}
      stepSelect.disabled=false;const method=methodByKey[key];method.steps.forEach(step=>stepSelect.add(new Option(`step ${step}`,String(step))));
      stepSelect.value=String(method.default_step)}
    function renderLanes(){const root=document.getElementById('lanes');root.replaceChildren();
      D.primary_metric_keys.forEach(key=>{const spec=specByKey[key];const lane=document.createElement('article');lane.className='lane';
        const rows=latestRows(key).filter(row=>row.gap>0).slice(0,5);lane.innerHTML=`<h3>${spec.label} ↑</h3><ol></ol>`;
        const list=lane.querySelector('ol');rows.forEach((row,index)=>{const li=document.createElement('li');
          li.innerHTML=`<span class="rank">${index+1}</span><div class="who"><div class="method">${row.method.label} · step ${row.record.step}</div>
            <div class="case">${row.item.stem}</div></div><span class="gap-chip">−${fmt(row.gap)}</span>`;
          li.onclick=()=>setMethod(row.method.key,row.record.step,key,row.item.stem);list.append(li)});root.append(lane)})}
    function renderOverview(){const head=document.getElementById('overview-head');const body=document.getElementById('overview-body');
      head.innerHTML='<tr><th>方案 · 默认 step</th>'+D.primary_metric_keys.map(key=>`<th>${specByKey[key].label} ↑</th>`).join('')+'</tr>';
      body.replaceChildren();D.methods.forEach(method=>{const record=defaultRecord(method);if(!record)return;const tr=document.createElement('tr');
        tr.innerHTML=`<td><div class="scheme-name" style="color:${method.color}">${method.label}</div><div class="step">step ${record.step}</div></td>`;
        D.primary_metric_keys.forEach(key=>{const row=rankedForRecord(record,key)[0];const td=document.createElement('td');td.className='metric-cell';
          if(!row){td.textContent='无完整数据'}else{td.innerHTML=`<span class="cell-gap">${row.gap>0?'−'+fmt(row.gap):'无劣势'}</span>
            <span class="cell-case" title="${row.item.stem}">${row.item.stem}</span>`;td.onclick=()=>setMethod(method.key,record.step,key,row.item.stem)}tr.append(td)});body.append(tr)})}
    function metricBoxes(row){return D.primary_metric_keys.map(key=>{const spec=specByKey[key];const value=metricValue(row.record,row.item.stem,key);
      const off=metricValue(refs.off,row.item.stem,key);const on=metricValue(refs.on,row.item.stem,key);const ref=referenceFor(row.item.stem,key);
      const gap=value===null||ref.value===null?null:(spec.direction==='lower'?value-ref.value:ref.value-value);
      return `<div class="metric-box ${key===metricSelect.value?'focus':''}"><h4>${spec.label} ${spec.direction==='lower'?'↓':'↑'}</h4><dl>
        <dt>方案</dt><dd>${fmt(value)}</dd><dt>PhysRVG OFF</dt><dd>${fmt(off)}</dd><dt>PhysRVG +LoRA</dt><dd>${fmt(on)}</dd>
        <dt>系列参考</dt><dd>${ref.label.replace('PhysRVG ','')}</dd><dt>原始劣势</dt><dd class="${gap>0?'bad':''}">${gap>0?'−'+fmt(gap):gap===null?'—':'无劣势'}</dd></dl></div>`}).join('')}
    function cardFor(row,index,maxGap){const article=document.createElement('article');article.className='case-card';
      const method=methodByKey[row.record.method_key];const width=maxGap>0?Math.max(0,Math.min(100,row.gap/maxGap*100)):0;
      article.innerHTML=`<div class="case-head"><div class="case-index">${String(index+1).padStart(2,'0')}</div><div><h3>${row.item.stem}</h3>
        <p>${row.item.prompt}</p></div><div class="gap-readout"><strong>${row.gap>0?'−'+fmt(row.gap):'无劣势'}</strong><span>${row.ref.label} − 方案</span></div></div>
        <div class="gap-rail"><span style="width:${width}%"></span></div><div class="videos">
        <div class="video-cell"><div class="video-label"><span>GT</span><em>49f · 30 FPS</em></div><video src="${row.item.gt}" muted playsinline controls preload="metadata"></video></div>
        <div class="video-cell"><div class="video-label" style="color:${method.color}"><span>${method.label}</span><em>step ${row.record.step}</em></div><video src="${row.record.videos[row.item.stem]}" muted playsinline controls preload="metadata"></video></div>
        <div class="video-cell"><div class="video-label" style="color:var(--off)"><span>PhysRVG LoRA OFF</span><em>finetuned DiT</em></div><video src="${refs.off.videos[row.item.stem]}" muted playsinline controls preload="metadata"></video></div>
        <div class="video-cell"><div class="video-label" style="color:var(--on)"><span>PhysRVG +LoRA</span><em>rank-32</em></div><video src="${refs.on.videos[row.item.stem]}" muted playsinline controls preload="metadata"></video></div>
        </div><div class="metrics">${metricBoxes(row)}</div>`;return article}
    function selectedRows(){const key=metricSelect.value;if(methodSelect.value==='__all__')return latestRows(key);
      const record=recordFor(methodSelect.value,stepSelect.value);return record?rankedForRecord(record,key):[]}
    function renderAudit(preferredStem){const key=metricSelect.value;const spec=specByKey[key];let rows=selectedRows();
      rows=rows.filter(row=>row.gap>0);if(preferredStem){const index=rows.findIndex(row=>row.item.stem===preferredStem);
        if(index>0)rows=[rows[index],...rows.slice(0,index),...rows.slice(index+1)]}
      const topk=Number(topkSelect.value);rows=rows.slice(0,topk);const cards=document.getElementById('cards');cards.replaceChildren();
      const methodLabel=methodSelect.value==='__all__'?'所有方案 · 各自默认 step':methodByKey[methodSelect.value].label+` · step ${stepSelect.value}`;
      document.getElementById('audit-title').textContent=`${spec.label} · 最大劣势 case`;
      document.getElementById('audit-note').textContent='劣势 = 参考 − 方案（WMReward 按 lower-better 反向计算）；只展示正劣势。';
      document.getElementById('selection-method').textContent=methodLabel;
      document.getElementById('selection-reference').textContent=referenceSelect.options[referenceSelect.selectedIndex].text;
      document.getElementById('selection-scope').textContent=scopeSelect.options[scopeSelect.selectedIndex].text;
      if(!rows.length){cards.innerHTML='<div class="empty">当前选择下没有正劣势且视频、指标完整的 case。</div>';return}
      const maxGap=rows[0].gap;rows.forEach((row,index)=>cards.append(cardFor(row,index,maxGap)))}
    function renderAll(){renderLanes();renderOverview();renderAudit()}
    methodSelect.add(new Option('全局最差（所有方案）','__all__'));D.methods.forEach(method=>methodSelect.add(new Option(method.label,method.key)));
    D.metric_specs.forEach((spec,index)=>{if(index===D.primary_metric_keys.length){const divider=new Option('──────── 其他指标 ────────','');divider.disabled=true;metricSelect.add(divider)}
      metricSelect.add(new Option(`${spec.primary?'主要 · ':''}${spec.label} ${spec.direction==='lower'?'↓':'↑'}`,spec.key))});
    methodSelect.value='__all__';metricSelect.value=D.primary_metric_keys[0];populateSteps();
    methodSelect.onchange=()=>{populateSteps();renderAudit()};stepSelect.onchange=()=>renderAudit();metricSelect.onchange=()=>renderAudit();
    topkSelect.onchange=()=>renderAudit();referenceSelect.onchange=renderAll;scopeSelect.onchange=renderAll;
    document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(video=>{video.currentTime=0;video.play().catch(()=>{})});
    renderAll();
  </script>
</body>
</html>
'''


def build_dashboard(source_page: Path, output_dir: Path) -> Path:
    data = build_data(load_dashboard_payload(source_page.resolve()))
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    html = HTML_TEMPLATE.replace("__DATA__", encoded).replace(
        "__GENERATED__", data["generated_utc"]
    )
    output_path = output_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_path = build_dashboard(args.source_page, args.output_dir)
    print(output_path)


if __name__ == "__main__":
    main()
