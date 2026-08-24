#!/usr/bin/env python3
"""Build a paired, representative-case audit for the four key metrics.

The page intentionally reuses the videos and per-case metrics already exposed
by the test5 and PhysicIQ dashboards.  It does not copy videos or checkpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
OUTPUT_DIR = HUB_ROOT / "physrvg-key-metric-case-audit"
SOURCE_PAGES = {
    "test5": HUB_ROOT / "test5" / "index.html",
    "physiciq": HUB_ROOT / "physiciq" / "index.html",
}

BASELINE_KEY = "physrvg_test5_lora_off"
KEY_METRICS = [
    {
        "key": "vbench_dynamic_degree",
        "label": "VBench degree",
        "short": "VBench degree",
        "kind": "behavior",
        "scale": 1.0,
        "note": "运动量信号；不单调判优",
    },
    {
        "key": "physics_iq_with_context",
        "label": "PhysicsIQ · context",
        "short": "PhysicsIQ",
        "kind": "quality",
        "scale": 100.0,
        "note": "越高越好",
    },
    {
        "key": "videophy2_joint_rate",
        "label": "VideoPhy · joint pass",
        "short": "VideoPhy",
        "kind": "quality",
        "scale": 1.0,
        "note": "越高越好",
    },
    {
        "key": "cosmos_reason1",
        "label": "Cosmos Reason",
        "short": "Cosmos Reason",
        "kind": "quality",
        "scale": 5.0,
        "note": "越高越好",
    },
]
KEYS = [item["key"] for item in KEY_METRICS]
QUALITY_KEYS = [item["key"] for item in KEY_METRICS if item["kind"] == "quality"]


def load_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\s*const caseSelect=", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find dashboard payload in {path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object payload in {path}")
    return payload


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def complete_count(record: dict[str, Any], stems: list[str]) -> int:
    count = 0
    for stem in stems:
        metrics = record.get("metrics", {}).get(stem, {})
        if all(is_number(metrics.get(key)) for key in KEYS):
            count += 1
    return count


def compact_dataset(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    raw_cases = [item for item in payload.get("cases", []) if isinstance(item, dict)]
    cases = [
        {
            "stem": str(item.get("stem", "")),
            "prompt": str(item.get("prompt", "")),
            "gt": str(item.get("gt", "")),
            "context": str(item.get("context", "")),
        }
        for item in raw_cases
        if str(item.get("stem", ""))
    ]
    stems = [item["stem"] for item in cases]

    methods: list[dict[str, Any]] = []
    for item in payload.get("methods", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", ""))
        if key:
            methods.append(
                {
                    "key": key,
                    "label": str(item.get("label", key)),
                    "color": str(item.get("color", "#557078")),
                }
            )

    records: list[dict[str, Any]] = []
    for raw in payload.get("records", []):
        if not isinstance(raw, dict):
            continue
        method_key = str(raw.get("method_key", ""))
        if not method_key:
            continue
        metrics: dict[str, dict[str, float]] = {}
        videos: dict[str, str] = {}
        raw_metrics = raw.get("metrics", {})
        raw_videos = raw.get("videos", {})
        if not isinstance(raw_metrics, dict):
            raw_metrics = {}
        if not isinstance(raw_videos, dict):
            raw_videos = {}
        for stem in stems:
            values = raw_metrics.get(stem, {})
            if isinstance(values, dict):
                compact_values = {
                    key: float(values[key])
                    for key in KEYS
                    if is_number(values.get(key))
                }
                if compact_values:
                    metrics[stem] = compact_values
            video = raw_videos.get(stem)
            if isinstance(video, str) and video:
                videos[stem] = video
        records.append(
            {
                "method_key": method_key,
                "method_label": str(raw.get("method_label", method_key)),
                "step": int(raw.get("step", 0)),
                "videos": videos,
                "metrics": metrics,
                "complete_count": complete_count(
                    {"metrics": metrics}, stems
                ),
            }
        )

    method_by_key = {item["key"]: item for item in methods}
    for record in records:
        method_by_key.setdefault(
            record["method_key"],
            {
                "key": record["method_key"],
                "label": record["method_label"],
                "color": "#557078",
            },
        )
    methods = list(method_by_key.values())
    records.sort(key=lambda item: (item["method_key"], item["step"]))

    # Pick a useful default without hard-coding a checkpoint that is absent.
    preferred_method = "full_sa_physrvg_vjepa_loss"
    preferred_step = 3000
    if not any(
        item["method_key"] == preferred_method and item["step"] == preferred_step
        for item in records
    ):
        preferred_method = next(
            (
                item["method_key"]
                for item in methods
                if item["key"] != BASELINE_KEY
                and any(r["method_key"] == item["key"] for r in records)
            ),
            "",
        )
        candidates = [
            item for item in records if item["method_key"] == preferred_method
        ]
        preferred_step = max(
            candidates,
            key=lambda item: (item["complete_count"], item["step"]),
        )["step"] if candidates else 0

    return {
        "name": name,
        "label": "test_5" if name == "test5" else "PhysicIQ · 67-case",
        "case_count": len(cases),
        "cases": cases,
        "methods": methods,
        "records": records,
        "default_method": preferred_method,
        "default_step": preferred_step,
    }


def build_payload() -> dict[str, Any]:
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "baseline_key": BASELINE_KEY,
        "metrics": KEY_METRICS,
        "quality_keys": QUALITY_KEYS,
        "datasets": {
            name: compact_dataset(name, load_payload(path))
            for name, path in SOURCE_PAGES.items()
        },
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PHYRVG · 四项重点指标 case 审计</title>
  <style>
    :root{
      --ink:#18252b;--muted:#65757b;--paper:#f5f7f6;--surface:#fff;
      --line:#d7e0df;--navy:#112d38;--navy2:#1b4853;--teal:#0a7477;
      --orange:#e49b37;--green:#13735b;--red:#ae4b3b;--violet:#6d5ba6;
      --shadow:0 10px 28px rgba(17,45,56,.10)
    }
    *{box-sizing:border-box}html{scroll-behavior:smooth}
    body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,"Noto Sans SC",Arial,sans-serif}
    button,select{font:inherit}a{color:inherit}
    .hero{padding:27px clamp(16px,4vw,58px) 23px;color:#f6fbfb;
      background:linear-gradient(118deg,var(--navy) 0 70%,var(--navy2) 70% 100%);border-bottom:5px solid var(--orange)}
    .hero-top{display:flex;align-items:center;justify-content:space-between;gap:16px}
    .eyebrow{font:850 10px/1 "Arial Narrow",sans-serif;letter-spacing:.18em;text-transform:uppercase;color:#8bc7c8}
    .back{color:#bee0df;text-decoration:none;font-size:12px;font-weight:850}
    h1{max-width:970px;margin:15px 0 9px;font:850 clamp(29px,4.5vw,57px)/.98 "Arial Narrow","Roboto Condensed",sans-serif;letter-spacing:-.035em}
    .hero p{max-width:1040px;margin:0;color:#c8dbdc;font-size:13px;line-height:1.65}
    .signal-rail{display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-top:22px}
    .signal-rail span{height:5px;background:#426771}.signal-rail span:nth-child(1){background:var(--violet)}
    .signal-rail span:nth-child(2){background:#49a5a4}.signal-rail span:nth-child(3){background:var(--orange)}.signal-rail span:nth-child(4){background:#70b58d}
    .controls{position:sticky;top:0;z-index:20;display:grid;grid-template-columns:145px minmax(300px,1fr) 190px auto;
      align-items:end;gap:10px;padding:11px clamp(12px,3vw,38px);background:rgba(245,247,246,.97);
      border-bottom:1px solid var(--line);box-shadow:0 5px 18px rgba(17,45,56,.08)}
    .control{display:grid;gap:4px}.control label{color:var(--muted);font:850 10px/1 "Arial Narrow",sans-serif;letter-spacing:.08em;text-transform:uppercase}
    select{height:38px;width:100%;padding:0 10px;border:1px solid var(--line);border-radius:4px;background:#fff;color:var(--ink);font-size:12px;font-weight:750}
    button{height:38px;padding:0 13px;border:1px solid var(--line);border-radius:4px;background:#fff;color:var(--ink);font-size:12px;font-weight:850;cursor:pointer}
    button.primary{background:var(--teal);border-color:var(--teal);color:#fff}button:hover{border-color:var(--orange)}
    button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #e7b44d;outline-offset:2px}
    main{max-width:1880px;margin:auto;padding:20px clamp(12px,3vw,38px) 70px}
    .notice{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:start;padding:12px 14px;margin-bottom:13px;
      background:#edf5f4;border:1px solid #c8dfdd;border-left:5px solid var(--teal);font-size:12px;line-height:1.6}
    .notice strong{color:var(--teal);font-size:12px;white-space:nowrap}.notice span{color:#456066}
    .summary{display:grid;grid-template-columns:repeat(6,minmax(125px,1fr));gap:9px;margin-bottom:17px}
    .summary-card{min-height:89px;padding:11px 12px;background:var(--surface);border:1px solid var(--line);border-top:4px solid var(--accent);box-shadow:var(--shadow)}
    .summary-card .label{display:block;color:var(--muted);font:850 10px/1.15 "Arial Narrow",sans-serif;letter-spacing:.07em;text-transform:uppercase}
    .summary-card strong{display:block;margin-top:7px;font:850 20px/1 "Arial Narrow",sans-serif}.summary-card small{display:block;margin-top:5px;color:var(--muted);font-size:10px;line-height:1.35}
    .summary-card.verdict{grid-column:span 2}.verdict .badge{display:inline-flex;margin-top:7px;padding:5px 8px;border-radius:3px;font-size:12px;font-weight:900}
    .badge.effective{color:#075d48;background:#dff1e9}.badge.ineffective{color:#8b3327;background:#f8e3df}.badge.mixed{color:#76540d;background:#fbf0d4}
    .section-head{display:flex;align-items:end;justify-content:space-between;gap:15px;margin:22px 0 10px}.section-head h2{margin:0;font:850 22px/1 "Arial Narrow",sans-serif}.section-head p{max-width:780px;margin:0;color:var(--muted);font-size:11px;line-height:1.5;text-align:right}
    .case-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:13px}.case-grid.empty{display:block}
    .case-card{overflow:hidden;background:var(--surface);border:1px solid var(--line);box-shadow:var(--shadow)}
    .case-head{display:flex;justify-content:space-between;gap:12px;padding:12px 13px 9px;border-bottom:1px solid var(--line);background:#fbfcfc}
    .case-index{color:var(--orange);font:900 12px/1 "Arial Narrow",sans-serif;letter-spacing:.08em}.case-title{margin:4px 0 0;font-size:14px;font-weight:900;line-height:1.25;overflow-wrap:anywhere}.case-prompt{margin:5px 0 0;color:var(--muted);font-size:11px;line-height:1.4;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
    .case-verdict{flex:0 0 auto;align-self:start;padding:5px 7px;border-radius:3px;font-size:10px;font-weight:900;white-space:nowrap}.case-verdict.effective{background:#dff1e9;color:#075d48}.case-verdict.ineffective{background:#f8e3df;color:#8b3327}.case-verdict.mixed{background:#fbf0d4;color:#76540d}
    .videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:10px 11px 0}.video-block{min-width:0}.video-label{display:flex;justify-content:space-between;gap:6px;margin-bottom:4px;color:var(--muted);font:850 10px/1.2 "Arial Narrow",sans-serif;letter-spacing:.03em}.video-label strong{color:var(--ink);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#10232a;border:1px solid #cbd6d6}
    .aux-videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:9px 11px 0}.aux-videos .video-label{font-size:9px}.aux-videos video{aspect-ratio:16/5;background:#e9efee}
    .metrics{margin:10px 11px 12px;border-top:3px solid var(--teal)}table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}th,td{padding:7px 6px;border-bottom:1px solid #e4eaea;text-align:right}th:first-child,td:first-child{text-align:left}thead th{color:var(--muted);font-size:9px;letter-spacing:.06em;text-transform:uppercase}tbody tr:last-child td{border-bottom:0}.metric-name{font-weight:850}.metric-note{display:block;color:var(--muted);font-size:9px;font-weight:500;margin-top:2px}.delta{font-weight:900}.delta.pos{color:var(--green)}.delta.neg{color:var(--red)}.delta.neutral{color:var(--muted)}.signal{color:var(--violet);font-size:10px;font-weight:850}
    .footer-note{margin-top:18px;padding:13px 14px;color:var(--muted);background:#fff;border:1px solid var(--line);font-size:11px;line-height:1.6}.footer-note code{padding:1px 4px;background:#eef3f2;color:var(--ink)}
    .pending{padding:18px;background:#fff;border:1px dashed #bdcccb;color:var(--muted);font-size:12px}.hidden{display:none!important}
    @media(max-width:1050px){.summary{grid-template-columns:repeat(3,minmax(130px,1fr))}.summary-card.verdict{grid-column:span 3}.controls{grid-template-columns:130px minmax(220px,1fr) 160px auto}}
    @media(max-width:720px){.hero{padding:22px 15px 19px}.controls{position:static;grid-template-columns:1fr 1fr;padding:10px 12px}.control.method{grid-column:span 2}.controls button{width:100%}.summary{grid-template-columns:repeat(2,minmax(130px,1fr))}.summary-card.verdict{grid-column:span 2}.case-grid{grid-template-columns:1fr}.section-head{display:block}.section-head p{text-align:left;margin-top:6px}.videos{grid-template-columns:1fr}.aux-videos{grid-template-columns:1fr 1fr}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-top"><span class="eyebrow">paired physics audit · 8844</span><a class="back" href="../">← 返回 8844 总览</a></div>
    <h1>Baseline vs scheme<br>四项重点指标的典型 case</h1>
    <p>同一 source case、同一 baseline、同一 checkpoint 的配对审计。视频用于核查，指标用于定位“在哪些 case 有效、在哪些 case 失效”。PhysicsIQ、VideoPhy、Cosmos Reason 用于质量结论；VBench degree 作为运动量行为信号单独报告。</p>
    <div class="signal-rail" aria-hidden="true"><span></span><span></span><span></span><span></span></div>
  </header>
  <div class="controls">
    <div class="control"><label for="dataset">测试集</label><select id="dataset"><option value="test5">test_5</option><option value="physiciq">PhysicIQ · 67-case</option></select></div>
    <div class="control method"><label for="method">对比方案</label><select id="method"></select></div>
    <div class="control"><label for="step">checkpoint step</label><select id="step"></select></div>
    <button class="primary" id="reload" type="button">重新读取页面</button>
  </div>
  <main>
    <div class="notice"><strong>判定口径</strong><span>“有效/无效”是逐 case 的方向性审计：质量结论要求三个质量指标中至少两个同向且平均标准化差异超过阈值；其余归为“权衡/不确定”。这不是显著性检验。VBench degree 不参与单调质量判定，因为更高 degree 只表示运动量更大，不必然表示更真实。</span></div>
    <section id="summary" class="summary" aria-live="polite"></section>
    <div id="content"></div>
    <div class="footer-note">数据快照生成于 <code id="generated"></code>。视频使用现有 8844 gallery 的相对路径并按卡片进入视口后懒加载；缺失指标不会按 0 计入平均值。若 watcher 新增结果，重新运行构建脚本后点击“重新读取页面”。</div>
  </main>
  <script>
  const PAYLOAD = __PAYLOAD__;
  const baselineKey = PAYLOAD.baseline_key;
  const metricSpecs = PAYLOAD.metrics;
  const qualityKeys = new Set(PAYLOAD.quality_keys);
  const state = {dataset:"test5", method:"", step:0};
  const $ = (id) => document.getElementById(id);
  const finite = (x) => typeof x === "number" && Number.isFinite(x);
  const esc = (x) => String(x ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
  const shortText = (x, n=115) => { const s=String(x||"").replace(/\s+/g," ").trim(); return s.length>n ? s.slice(0,n-1)+"…" : s; };
  const humanStem = (s) => String(s||"").replace(/^physicIQ_/i,"PhysicIQ · ").replace(/^0613pybullet_sample_/i,"PyBullet · sample ").replace(/_/g," ");
  const fmt = (key, value) => {
    if (!finite(value)) return "—";
    if (key === "videophy2_joint_rate") return value.toFixed(2);
    if (key === "vbench_dynamic_degree") return value.toFixed(2);
    return value.toFixed(2);
  };
  const fmtDelta = (key, value) => {
    if (!finite(value)) return "—";
    const sign=value>0?"+":"";
    return sign + (key === "videophy2_joint_rate" || key === "vbench_dynamic_degree" ? value.toFixed(2) : value.toFixed(2));
  };
  function dataset(){ return PAYLOAD.datasets[state.dataset]; }
  function methodMeta(key){ return dataset().methods.find(x=>x.key===key) || {key,label:key,color:"#557078"}; }
  function recordsFor(key){ return dataset().records.filter(x=>x.method_key===key).sort((a,b)=>a.step-b.step); }
  function getRecord(key, step){ return dataset().records.find(x=>x.method_key===key && Number(x.step)===Number(step)); }
  function countComplete(record){ return Number(record?.complete_count||0); }
  function methodGroup(label,key){
    const s=(label+" "+key).toLowerCase();
    if(s.includes("phaselock")) return "PhaseLock";
    if(s.includes("wmreward")) return "WMReward";
    if(s.includes("full_sa_physrvg") || s.includes("phyrvg-full-sa")) return "PHYRVG-Full-SA";
    if(s.includes("physrvg")) return "PHYRVG / reference";
    return "Other schemes";
  }
  function populateMethods(){
    const d=dataset();
    const options=d.methods.filter(m=>m.key!==baselineKey && d.records.some(r=>r.method_key===m.key));
    const groups={};
    options.forEach(m=>(groups[methodGroup(m.label,m.key)] ||= []).push(m));
    const preferred=state.method && options.some(x=>x.key===state.method) ? state.method : (d.default_method || options[0]?.key || "");
    state.method=preferred;
    $("method").innerHTML=Object.entries(groups).map(([group,items])=>`<optgroup label="${esc(group)}">${items.sort((a,b)=>a.label.localeCompare(b.label,"zh-CN")).map(m=>`<option value="${esc(m.key)}" ${m.key===state.method?"selected":""}>${esc(m.label)}</option>`).join("")}</optgroup>`).join("");
    populateSteps();
  }
  function populateSteps(){
    const rows=recordsFor(state.method);
    if(!rows.length){$("step").innerHTML="<option>pending</option>";state.step=0;return;}
    const preferred=rows.some(r=>Number(r.step)===Number(state.step)) ? state.step : (dataset().default_method===state.method ? dataset().default_step : rows.slice().sort((a,b)=>(countComplete(b)-countComplete(a)) || (b.step-a.step))[0].step);
    state.step=Number(preferred);
    $("step").innerHTML=rows.map(r=>`<option value="${r.step}" ${Number(r.step)===state.step?"selected":""}>step ${String(r.step).padStart(4,"0")} · ${countComplete(r)}/${dataset().case_count} complete</option>`).join("");
  }
  function pairedRows(base,candidate){
    if(!base || !candidate) return [];
    return dataset().cases.map(c=>{
      const b=base.metrics?.[c.stem]||{}, v=candidate.metrics?.[c.stem]||{};
      const complete=metricSpecs.every(m=>finite(b[m.key]) && finite(v[m.key]));
      if(!complete) return null;
      const deltas={};
      metricSpecs.forEach(m=>deltas[m.key]=v[m.key]-b[m.key]);
      const q=metricSpecs.filter(m=>qualityKeys.has(m.key)).map(m=>deltas[m.key]/m.scale);
      const qualityScore=q.reduce((a,x)=>a+x,0)/q.length;
      const qualityUp=q.filter(x=>x>1e-9).length;
      const qualityDown=q.filter(x=>x<-1e-9).length;
      let verdict="mixed";
      if(qualityUp>=2 && qualityScore>0.025) verdict="effective";
      else if(qualityDown>=2 && qualityScore<-0.025) verdict="ineffective";
      return {...c, baseValues:b, values:v, deltas, qualityScore, qualityUp, qualityDown, verdict, degreeDelta:deltas.vbench_dynamic_degree};
    }).filter(Boolean);
  }
  function aggregate(base,candidate){
    const out={};
    metricSpecs.forEach(m=>{
      const vals=dataset().cases.map(c=>({b:base?.metrics?.[c.stem]?.[m.key],v:candidate?.metrics?.[c.stem]?.[m.key]})).filter(x=>finite(x.b)&&finite(x.v));
      out[m.key]={n:vals.length,base:vals.length?vals.reduce((a,x)=>a+x.b,0)/vals.length:null,value:vals.length?vals.reduce((a,x)=>a+x.v,0)/vals.length:null,delta:vals.length?vals.reduce((a,x)=>a+x.v-x.b,0)/vals.length:null};
    });
    return out;
  }
  function summaryHtml(base,candidate,rows){
    const agg=aggregate(base,candidate); const q=metricSpecs.filter(m=>qualityKeys.has(m.key));
    const qd=q.map(m=>agg[m.key].delta/ m.scale).filter(finite); const qscore=qd.length?qd.reduce((a,x)=>a+x,0)/qd.length:0;
    const up=rows.filter(r=>r.qualityScore>0.025).length, down=rows.filter(r=>r.qualityScore<-0.025).length;
    const verdict=up>down && qscore>0.025?"effective":down>up && qscore<-0.025?"ineffective":"mixed";
    const verdictText=verdict==="effective"?"整体有效":verdict==="ineffective"?"整体无效":"整体权衡 / 不确定";
    const cards=metricSpecs.map((m,i)=>{const a=agg[m.key];const cls=!finite(a.delta)?"neutral":a.delta>1e-9?"pos":a.delta<-1e-9?"neg":"neutral";return `<div class="summary-card" style="--accent:${["#6d5ba6","#49a5a4","#e49b37","#70b58d"][i]}"><span class="label">${esc(m.short)}</span><strong>${fmtDelta(m.key,a.delta)}</strong><small>${a.n}/${dataset().case_count} paired · base ${fmt(m.key,a.base)} → ${fmt(m.key,a.value)}<br><span class="${m.kind==='behavior'?'signal':''}">${esc(m.note)}</span></small></div>`}).join("");
    return cards+`<div class="summary-card verdict" style="--accent:${verdict==='effective'?'#13735b':verdict==='ineffective'?'#ae4b3b':'#d39a2c'}"><span class="label">质量结论 · paired case</span><span class="badge ${verdict}">${verdictText}</span><small>完整四指标 ${rows.length}/${dataset().case_count} · case-level quality ↑ ${up} / ↓ ${down}<br>质量平均标准化差分 ${qscore>=0?"+":""}${qscore.toFixed(3)}；VBench degree 单独看作行为信号</small></div>`;
  }
  function videoTag(src, label, cls=""){ if(!src) return `<div class="video-block ${cls}"><div class="video-label"><strong>${esc(label)}</strong><span>pending</span></div><div class="pending">暂无视频</div></div>`; return `<div class="video-block ${cls}"><div class="video-label"><strong>${esc(label)}</strong><span>lazy</span></div><video controls preload="none" data-src="${esc(src)}" aria-label="${esc(label)}"></video></div>`; }
  function metricTable(row){
    const body=metricSpecs.map(m=>{const d=row.deltas[m.key];const cls=d>1e-9?"pos":d<-1e-9?"neg":"neutral";const signal=m.kind==='behavior'?`<span class="signal">行为信号</span>`:(d>1e-9?"↑":d<-1e-9?"↓":"—");return `<tr><td class="metric-name">${esc(m.label)}<span class="metric-note">${esc(m.note)}</span></td><td>${fmt(m.key,row.baseValues[m.key])}</td><td>${fmt(m.key,row.values[m.key])}</td><td class="delta ${cls}">${fmtDelta(m.key,d)} ${signal}</td></tr>`}).join("");
    return `<table><thead><tr><th>指标</th><th>baseline</th><th>方案</th><th>Δ</th></tr></thead><tbody>${body}</tbody></table>`;
  }
  function caseCard(row,index,base,candidate){
    const candLabel=methodMeta(state.method).label+" · step "+state.step;
    const title=humanStem(row.stem); const verdictText=row.verdict==='effective'?"case 有效":row.verdict==='ineffective'?"case 无效":"权衡 / 不确定";
    return `<div class="case-card"><div class="case-head"><div><div class="case-index">CASE ${String(index).padStart(2,"0")}</div><div class="case-title">${esc(title)}</div><p class="case-prompt">${esc(shortText(row.prompt))}</p></div><span class="case-verdict ${row.verdict}">${verdictText}</span></div><div class="videos">${videoTag(base.videos?.[row.stem],"baseline · step0")}${videoTag(candidate.videos?.[row.stem],shortText(candLabel,52))}</div><div class="aux-videos">${videoTag(row.context,"context · 8 frames","aux")}${videoTag(row.gt,"ground truth · 49 frames","aux")}</div><div class="metrics">${metricTable(row)}</div></div>`;
  }
  function selectRows(rows){
    const used=new Set();
    const top=rows.filter(r=>r.verdict==="effective").sort((a,b)=>b.qualityScore-a.qualityScore);
    const bottom=rows.filter(r=>r.verdict==="ineffective").sort((a,b)=>a.qualityScore-b.qualityScore);
    const mixed=rows.filter(r=>r.verdict==="mixed").sort((a,b)=>Math.abs(a.qualityScore)-Math.abs(b.qualityScore));
    const take=(list,n)=>{const out=[];for(const r of list){if(out.length>=n||used.has(r.stem))continue;out.push(r);used.add(r.stem)}return out};
    const effective=take(top,3); const ineffective=take(bottom,3); const tradeoff=take(mixed,2);
    return {effective,ineffective,tradeoff};
  }
  function section(title,desc,rows,base,candidate,cls){
    if(!rows.length) return `<section><div class="section-head"><h2>${title}</h2><p>${desc}</p></div><div class="pending">当前方案没有满足筛选条件的完整四指标 case；已有结果可能仍缺少某项指标。</div></section>`;
    return `<section><div class="section-head"><h2>${title}</h2><p>${desc}</p></div><div class="case-grid ${cls||''}">${rows.map((r,i)=>caseCard(r,i+1,base,candidate)).join("")}</div></section>`;
  }
  function hydrateVideos(){
    const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){const v=e.target;if(v.dataset.src){v.src=v.dataset.src;delete v.dataset.src;observer.unobserve(v);}}}),{rootMargin:"240px"});
    document.querySelectorAll("video[data-src]").forEach(v=>observer.observe(v));
  }
  function render(){
    const d=dataset(); const base=getRecord(baselineKey,0); const candidate=getRecord(state.method,state.step);
    const rows=pairedRows(base,candidate); $("generated").textContent=PAYLOAD.generated_utc;
    if(!base||!candidate){$("summary").innerHTML="<article class='summary-card verdict' style='--accent:#ae4b3b'><span class='label'>状态</span><span class='badge ineffective'>pending</span><small>baseline 或当前方案记录不存在</small></article>";$("content").innerHTML="<div class='pending'>等待可用 checkpoint。</div>";return;}
    $("summary").innerHTML=summaryHtml(base,candidate,rows);
    const selected=selectRows(rows);
    const method=methodMeta(state.method);
    $("content").innerHTML=`<div class="section-head"><h2>${esc(d.label)} · ${esc(method.label)} · step ${state.step}</h2><p>baseline：${esc(methodMeta(baselineKey).label)} · 固定 step0<br>候选：${esc(method.label)} · 完整配对 ${rows.length}/${d.case_count}</p></div>`+
      section("典型有效 case","质量指标中至少两项相对 baseline 提升，按标准化平均差分排序；卡片仍完整显示 degree 的变化。",selected.effective,base,candidate,"effective")+
      section("典型无效 case","质量指标中至少两项相对 baseline 退化，按标准化平均差分排序；用于定位方案的失败模式。",selected.ineffective,base,candidate,"ineffective")+
      section("典型权衡 / 不确定 case","质量指标方向不一致或平均差异接近 0；不要把单项提升误读为整体有效。",selected.tradeoff,base,candidate,"mixed");
    hydrateVideos();
  }
  $("dataset").addEventListener("change",e=>{state.dataset=e.target.value;state.method="";state.step=NaN;populateMethods();render();});
  $("method").addEventListener("change",e=>{state.method=e.target.value;state.step=NaN;populateSteps();render();});
  $("step").addEventListener("change",e=>{state.step=Number(e.target.value);render();});
  $("reload").addEventListener("click",()=>location.reload());
  populateMethods(); render();
  </script>
</body>
</html>
'''


def write_page() -> Path:
    payload = build_payload()
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Keep user prompts from terminating the embedding script tag.
    encoded = encoded.replace("</", "<\\/")
    body = HTML_TEMPLATE.replace("__PAYLOAD__", encoded)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "index.html"
    output.write_text(body, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(write_page())
