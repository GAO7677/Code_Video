#!/usr/bin/env python3
"""Build a unified, multi-select method comparison page for test5/PhysicIQ."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any

from build_xssc_lora_checkpoint_dashboard import (
    CASE_METRIC_SPECS,
    display_methods,
    is_phyrvg_method,
)


OUTPUT_PATH = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "test5-physiciq-method-compare/index.html"
)


def is_full_sa(method: dict[str, Any]) -> bool:
    key = str(method.get("key", ""))
    label = str(method.get("label", "")).upper()
    return key.startswith("full_sa_physrvg") or label.startswith("PHYRVG-FULL-SA")


def method_group(method: dict[str, Any]) -> str:
    if is_full_sa(method):
        return "PHYRVG-Full-SA"
    if is_phyrvg_method(method):
        return "PHYRVG / reference"
    return "Other training schemes"


def finite_values(
    record_metrics: dict[str, Any],
    cases: list[dict[str, Any]],
    metric_key: str,
) -> list[float]:
    values: list[float] = []
    for case in cases:
        stem = str(case.get("stem", ""))
        case_metrics = record_metrics.get(stem, {})
        if not isinstance(case_metrics, dict):
            continue
        value = case_metrics.get(metric_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def summarize_dataset(payload: dict[str, Any], label: str) -> dict[str, Any]:
    cases = payload.get("cases", [])
    raw_methods = payload.get("methods", [])
    raw_records = payload.get("records", [])
    cases = [case for case in cases if isinstance(case, dict)]
    methods = display_methods(
        [method for method in raw_methods if isinstance(method, dict)]
    )
    method_order = {
        str(method.get("key", "")): index for index, method in enumerate(methods)
    }
    rows: list[dict[str, Any]] = []
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        # Keep the new comparison page focused on training checkpoints for
        # now.  The original average pages retain these reference rows.
        if (
            str(record.get("step_kind", "")) == "inference"
            and int(record.get("step", 0)) == 40
        ):
            continue
        method_key = str(record.get("method_key", ""))
        method = next(
            (item for item in methods if str(item.get("key", "")) == method_key),
            None,
        )
        if method is None:
            method = {
                "key": method_key,
                "label": str(record.get("method_label", method_key)),
                "color": "#52636d",
            }
        record_metrics = record.get("metrics", {})
        if not isinstance(record_metrics, dict):
            record_metrics = {}
        metrics: dict[str, dict[str, Any]] = {}
        for spec in CASE_METRIC_SPECS:
            key = str(spec["key"])
            values = finite_values(record_metrics, cases, key)
            metrics[key] = {
                "count": len(values),
                "mean": sum(values) / len(values) if values else None,
            }
        rows.append(
            {
                "method_key": method_key,
                "method_label": str(method.get("label", method_key)),
                "color": str(method.get("color", "#52636d")),
                "group": method_group(method),
                "step": int(record.get("step", 0)),
                "step_kind": str(record.get("step_kind", "training")),
                "metrics": metrics,
            }
        )
    rows.sort(
        key=lambda row: (
            method_order.get(str(row["method_key"]), 999),
            int(row["step"]),
            str(row["step_kind"]),
        )
    )
    visible_method_keys = {str(row["method_key"]) for row in rows}
    method_payload = [
        {
            "key": str(method.get("key", "")),
            "label": str(method.get("label", method.get("key", ""))),
            "color": str(method.get("color", "#52636d")),
            "group": method_group(method),
        }
        for method in methods
        if str(method.get("key", "")) in visible_method_keys
    ]
    return {
        "key": label.lower(),
        "label": label,
        "case_count": len(cases),
        "methods": method_payload,
        "rows": rows,
        "metric_specs": [
            {
                "key": str(spec["key"]),
                "label": str(spec["label"]),
                "direction": str(spec["direction"]),
            }
            for spec in CASE_METRIC_SPECS
        ],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>test5 / PhysicIQ · 方案筛选指标对比</title>
  <style>
    :root{--ink:#162b32;--muted:#667980;--paper:#f4f7f6;--surface:#fff;
      --line:#d7e0e0;--teal:#0d6870;--teal-dark:#123842;--amber:#d39a2c;
      --green:#176b5c;--red:#a54332;--shadow:0 10px 28px rgba(18,52,60,.09)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:
      linear-gradient(135deg,#e8f1f0 0 14%,transparent 14%),var(--paper);color:var(--ink);
      font-family:Inter,"Noto Sans SC",Arial,sans-serif}
    header{padding:24px clamp(16px,4vw,56px) 20px;background:var(--teal-dark);color:#f5fbfb;
      border-bottom:5px solid var(--amber)}
    .kicker{font:800 10px/1 "Arial Narrow",sans-serif;letter-spacing:.2em;color:#89c9c8;text-transform:uppercase}
    h1{margin:13px 0 7px;font:850 clamp(26px,4vw,48px)/1 "Arial Narrow","Roboto Condensed",sans-serif;letter-spacing:-.025em}
    header p{max-width:920px;margin:0;color:#c4d8d9;font-size:13px;line-height:1.55}
    main{max-width:1900px;margin:auto;padding:16px clamp(10px,2.5vw,32px) 60px}
    .dataset-tabs{display:flex;gap:8px;margin-bottom:12px}
    .dataset-tabs button,.view-toggle button,.action{border:1px solid var(--line);background:#fff;color:var(--ink);
      border-radius:5px;padding:9px 15px;font:850 12px/1 inherit;cursor:pointer}
    .dataset-tabs button{border-top:4px solid var(--teal);min-width:160px;text-align:left}
    .dataset-tabs button.active{background:var(--teal);color:#fff;border-color:var(--teal);border-top-color:var(--amber)}
    .control-deck{display:grid;grid-template-columns:minmax(300px,1fr) minmax(300px,2.1fr);gap:12px;align-items:stretch}
    .panel{background:var(--surface);border:1px solid var(--line);box-shadow:var(--shadow);padding:13px}
    .panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:9px}
    .panel-head h2{margin:0;font-size:13px;letter-spacing:.04em}.panel-head span{color:var(--muted);font-size:11px}
    .search{height:34px;width:100%;padding:0 9px;border:1px solid var(--line);border-radius:4px;font:inherit;color:var(--ink)}
    .method-groups{display:grid;gap:9px;max-height:250px;overflow:auto;padding-right:3px}
    .method-group{display:grid;gap:5px}.method-group-title{color:var(--muted);font:850 10px/1 "Arial Narrow",sans-serif;letter-spacing:.1em;text-transform:uppercase}
    .method-grid{display:flex;flex-wrap:wrap;gap:5px}
    .method-chip{--chip:#52636d;position:relative;display:inline-flex;align-items:center;gap:5px;
      min-height:29px;padding:5px 8px 5px 7px;border:1px solid #d8e1e1;border-left:4px solid var(--chip);
      border-radius:3px;background:#fbfcfc;color:var(--ink);font-size:11px;line-height:1.2;cursor:pointer}
    .method-chip:has(input:checked){background:#e8f4f3;border-color:#9fc7c5;color:#075d63}
    .method-chip input{accent-color:var(--teal);margin:0}.method-chip em{font-style:normal;color:var(--muted);font-size:10px}
    .actions{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.action{padding:7px 10px;font-size:11px}
    .action.primary{background:var(--teal);border-color:var(--teal);color:#fff}.action:hover,.dataset-tabs button:hover,.view-toggle button:hover{border-color:var(--amber)}
    .summary{display:grid;grid-template-columns:repeat(4,minmax(120px,1fr));gap:8px;margin-bottom:12px}
    .summary-card{padding:11px 12px;border:1px solid var(--line);border-top:4px solid var(--accent);background:var(--surface);box-shadow:var(--shadow)}
    .summary-card span{display:block;color:var(--muted);font:850 10px/1 "Arial Narrow",sans-serif;letter-spacing:.08em;text-transform:uppercase}
    .summary-card strong{display:block;margin-top:6px;font:850 18px/1.1 "Arial Narrow",sans-serif}
    .summary-card small{display:block;margin-top:4px;color:var(--muted);font-size:10px;line-height:1.4}
    .table-panel{padding:0;overflow:hidden}.table-toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:12px 13px;border-bottom:1px solid var(--line);background:#fbfcfc}
    .table-toolbar strong{font-size:13px;margin-right:auto}.view-toggle{display:flex;gap:3px;padding:3px;border:1px solid var(--line);background:#eef3f2;border-radius:5px}.view-toggle button{padding:6px 9px;border:0;background:transparent}.view-toggle button.active{background:#fff;color:var(--teal);box-shadow:0 1px 3px rgba(18,52,60,.16)}
    .table-wrap{overflow:auto;max-height:calc(100vh - 390px);min-height:270px}
    table{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;font-size:11px;font-variant-numeric:tabular-nums}
    th,td{height:32px;padding:5px 8px;border-right:1px solid #e0e7e7;border-bottom:1px solid #e0e7e7;text-align:center;white-space:nowrap}
    thead th{position:sticky;top:0;z-index:4;background:#e6eeee;color:#40565c;font-weight:850}
    .method-col{position:sticky;left:0;z-index:3;min-width:235px;text-align:left;background:#fff;font-weight:850}
    .step-col{position:sticky;left:235px;z-index:3;min-width:76px;background:#fff;color:var(--muted)}
    thead .method-col,thead .step-col{z-index:5;background:#dce8e7}
    .method-name{display:flex;align-items:center;gap:7px}.swatch{width:7px;height:22px;display:inline-block;background:var(--method-color);flex:0 0 auto}
    td.best{background:#dff3e7;color:#075d37;font-weight:900}.best::before{content:"★ ";color:var(--green)}
    td.pending{color:#9aa6aa;font-size:10px}.group-start td{border-top:3px solid #b8c7c8}
    .empty{padding:42px;text-align:center;color:var(--muted);background:#fff}
    footer{padding:16px 0;color:var(--muted);font-size:10px}
    @media(max-width:980px){.control-deck{grid-template-columns:1fr}.table-wrap{max-height:none}.summary{grid-template-columns:repeat(2,1fr)}}
    @media(max-width:560px){.dataset-tabs{display:grid;grid-template-columns:1fr 1fr}.dataset-tabs button{min-width:0}.summary{grid-template-columns:1fr 1fr}.method-col{min-width:190px}.step-col{left:190px;min-width:65px}.table-wrap{font-size:10px}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  </style>
</head>
<body>
  <header>
    <div class="kicker">Metric selection desk · live static snapshot</div>
    <h1>方案筛选指标对比</h1>
    <p>选择一个或多个训练方案，集中查看 test5 / PhysicIQ 的 step 平均指标。数据来自已落盘的 case 结果；pending 表示该方案当前尚未完成全部 case 指标。</p>
  </header>
  <main>
    <nav class="dataset-tabs" aria-label="数据集">
      <button type="button" data-dataset="test5">test5 <span></span></button>
      <button type="button" data-dataset="physiciq">PhysicIQ <span></span></button>
    </nav>
    <section class="control-deck">
      <div class="panel">
        <div class="panel-head"><h2>方案选择</h2><span id="selection-count"></span></div>
        <input id="method-search" class="search" type="search" placeholder="搜索方案名称…" aria-label="搜索方案">
        <div id="method-groups" class="method-groups"></div>
        <div class="actions">
          <button type="button" class="action primary" id="select-all">全选</button>
          <button type="button" class="action" id="clear-all">清空</button>
          <button type="button" class="action" id="select-full-sa">只选 Full-SA</button>
        </div>
      </div>
      <div class="panel">
        <div class="panel-head"><h2 id="dataset-heading">test5</h2><span id="updated-at"></span></div>
        <div class="summary" id="summary"></div>
        <p style="margin:0;color:var(--muted);font-size:11px;line-height:1.55">表格按当前选择动态过滤。切换“按模型”可比较同一方案的不同 step；切换“按 Step”可横向比较不同方案。指标方向和原页面一致。</p>
      </div>
    </section>
    <section class="panel table-panel" style="margin-top:12px">
      <div class="table-toolbar"><strong id="table-title">test5 · 方案指标</strong><span id="table-count" style="color:var(--muted);font-size:11px"></span>
        <div class="view-toggle" role="group" aria-label="表格分组"><button type="button" data-view="model" class="active">按模型</button><button type="button" data-view="step">按 Step</button></div>
      </div>
      <div class="table-wrap" id="table-wrap"></div>
    </section>
    <footer><a href="../test5-average-metrics/">test5 全量指标</a> · <a href="../physiciq-average-metrics/">PhysicIQ 全量指标</a> · 页面由现有指标 watcher 增量刷新</footer>
  </main>
  <script>
    const D=__DATA__;
    const state={dataset:localStorage.getItem("methodCompareDataset")||"test5",view:localStorage.getItem("methodCompareView")||"model",query:"",selected:{}};
    const saved=JSON.parse(localStorage.getItem("methodCompareSelected")||"{}");
    Object.keys(D.datasets).forEach(key=>{
      const keys=D.datasets[key].methods.map(m=>m.key);
      const previous=Array.isArray(saved[key])?saved[key].filter(item=>keys.includes(item)):keys;
      state.selected[key]=new Set(previous);
    });
    if(!D.datasets[state.dataset])state.dataset="test5";
    const esc=(value)=>String(value??"").replace(/[&<>\"]/g,ch=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[ch]));
    const current=()=>D.datasets[state.dataset];
    const save=()=>{const out={};Object.entries(state.selected).forEach(([key,value])=>out[key]=[...value]);localStorage.setItem("methodCompareSelected",JSON.stringify(out));localStorage.setItem("methodCompareDataset",state.dataset);localStorage.setItem("methodCompareView",state.view)};
    const format=(value)=>{const n=Math.abs(value);return n>=10?value.toFixed(2):n>=1?value.toFixed(3):value.toFixed(4)};
    function renderTabs(){document.querySelectorAll("[data-dataset]").forEach(button=>{const d=D.datasets[button.dataset.dataset];button.classList.toggle("active",button.dataset.dataset===state.dataset);button.querySelector("span").textContent=`· ${d.case_count} cases`})}
    function renderMethods(){
      const dataset=current();const groups=new Map();const q=state.query.trim().toLowerCase();
      dataset.methods.forEach(method=>{if(q&&!`${method.label} ${method.key}`.toLowerCase().includes(q))return;if(!groups.has(method.group))groups.set(method.group,[]);groups.get(method.group).push(method)});
      const root=document.getElementById("method-groups");root.innerHTML="";
      groups.forEach((methods,group)=>{const section=document.createElement("section");section.className="method-group";section.innerHTML=`<div class="method-group-title">${esc(group)}</div><div class="method-grid"></div>`;const grid=section.querySelector(".method-grid");methods.forEach(method=>{const label=document.createElement("label");label.className="method-chip";label.style.setProperty("--chip",method.color);const checked=state.selected[state.dataset].has(method.key);label.innerHTML=`<input type="checkbox" data-method="${esc(method.key)}" ${checked?"checked":""}><span>${esc(method.label)}</span>`;grid.append(label)});root.append(section)});
      root.querySelectorAll("input[data-method]").forEach(input=>input.addEventListener("change",()=>{if(input.checked)state.selected[state.dataset].add(input.dataset.method);else state.selected[state.dataset].delete(input.dataset.method);save();render() }));
      document.getElementById("selection-count").textContent=`${state.selected[state.dataset].size}/${dataset.methods.length} 已选`;
    }
    function selectedRows(){const keys=state.selected[state.dataset];return current().rows.filter(row=>keys.has(row.method_key))}
    function renderTable(){
      const dataset=current();let rows=selectedRows();const methodIndex=new Map(dataset.methods.map((m,i)=>[m.key,i]));
      rows.sort((a,b)=>state.view==="model"?(methodIndex.get(a.method_key)||999)-(methodIndex.get(b.method_key)||999)||a.step-b.step:a.step-b.step||(methodIndex.get(a.method_key)||999)-(methodIndex.get(b.method_key)||999));
      const groups=new Map();rows.forEach(row=>{const key=state.view==="model"?row.method_key:String(row.step);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row)});
      const best=new Map();dataset.metric_specs.forEach(spec=>groups.forEach((group,groupKey)=>{const vals=group.map(row=>row.metrics[spec.key]).filter(stat=>stat&&stat.count===dataset.case_count&&Number.isFinite(stat.mean));if(!vals.length)return;const target=spec.direction==="lower"?Math.min(...vals.map(v=>v.mean)):Math.max(...vals.map(v=>v.mean));if(!best.has(groupKey))best.set(groupKey,{});best.get(groupKey)[spec.key]=target}));
      const root=document.getElementById("table-wrap");if(!rows.length){root.innerHTML='<div class="empty">没有选中的方案，请在上方勾选至少一个方法。</div>';document.getElementById("table-count").textContent="0 行";return}
      let html='<table><thead><tr><th class="method-col">方法</th><th class="step-col">Step</th>'+dataset.metric_specs.map(spec=>`<th>${esc(spec.label)} <span style="font-size:10px">${spec.direction==="lower"?"↓":"↑"}</span></th>`).join("")+'</tr></thead><tbody>';
      let previous=null;rows.forEach(row=>{const groupKey=state.view==="model"?row.method_key:String(row.step);const groupStart=groupKey!==previous;previous=groupKey;html+=`<tr class="${groupStart?"group-start":""}"><td class="method-col"><span class="method-name"><span class="swatch" style="--method-color:${esc(row.color)}"></span>${esc(row.method_label)}</span></td><td class="step-col">${row.step_kind==="inference"?"infer ":""}${row.step}</td>`;dataset.metric_specs.forEach(spec=>{const stat=row.metrics[spec.key]||{count:0,mean:null};const complete=stat.count===dataset.case_count&&Number.isFinite(stat.mean);const target=best.get(groupKey)?.[spec.key];const isBest=complete&&Number.isFinite(target)&&Math.abs(stat.mean-target)<=1e-9;html+=complete?`<td class="${isBest?"best":""}" data-value="${stat.mean}">${isBest?"★ ":""}${format(stat.mean)}</td>`:`<td class="pending">pending ${stat.count}/${dataset.case_count}</td>`});html+='</tr>'});html+='</tbody></table>';root.innerHTML=html;document.getElementById("table-count").textContent=`${rows.length} 行 · ${dataset.case_count} cases`;
    }
    function renderSummary(){const dataset=current(),rows=selectedRows();const complete=rows.reduce((sum,row)=>sum+dataset.metric_specs.filter(spec=>{const s=row.metrics[spec.key];return s&&s.count===dataset.case_count}).length,0);document.getElementById("summary").innerHTML=`<article class="summary-card" style="--accent:#0d6870"><span>Dataset</span><strong>${esc(dataset.label)}</strong><small>${dataset.case_count} 个 case</small></article><article class="summary-card" style="--accent:#d39a2c"><span>Selected schemes</span><strong>${state.selected[state.dataset].size}</strong><small>可多选方法</small></article><article class="summary-card" style="--accent:#176b5c"><span>Visible rows</span><strong>${rows.length}</strong><small>方法 / step</small></article><article class="summary-card" style="--accent:#a54332"><span>Complete cells</span><strong>${complete}</strong><small>完整 case 平均指标</small></article>`;document.getElementById("dataset-heading").textContent=dataset.label;document.getElementById("table-title").textContent=`${dataset.label} · 方案指标`;document.getElementById("updated-at").textContent=`${D.generated_utc} 更新`}
    function render(){renderTabs();renderMethods();renderSummary();renderTable();document.querySelectorAll("[data-view]").forEach(button=>button.classList.toggle("active",button.dataset.view===state.view));save()}
    document.querySelectorAll("[data-dataset]").forEach(button=>button.addEventListener("click",()=>{state.dataset=button.dataset.dataset;state.query="";document.getElementById("method-search").value="";render()}));
    document.querySelectorAll("[data-view]").forEach(button=>button.addEventListener("click",()=>{state.view=button.dataset.view;render()}));
    document.getElementById("method-search").addEventListener("input",event=>{state.query=event.target.value;renderMethods()});
    document.getElementById("select-all").addEventListener("click",()=>{current().methods.forEach(m=>state.selected[state.dataset].add(m.key));render()});
    document.getElementById("clear-all").addEventListener("click",()=>{state.selected[state.dataset].clear();render()});
    document.getElementById("select-full-sa").addEventListener("click",()=>{state.selected[state.dataset]=new Set(current().methods.filter(m=>m.group==="PHYRVG-Full-SA").map(m=>m.key));render()});
    render();
  </script>
</body>
</html>'''


def build_page(test5_payload: dict[str, Any], physiciq_payload: dict[str, Any]) -> Path:
    data = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {
            "test5": summarize_dataset(test5_payload, "test5"),
            "physiciq": summarize_dataset(physiciq_payload, "PhysicIQ"),
        },
    }
    encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    html_text = HTML_TEMPLATE.replace("__DATA__", encoded)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(f".{OUTPUT_PATH.name}.tmp")
    temporary.write_text(html_text, encoding="utf-8")
    temporary.replace(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    raise SystemExit("Import build_page(test5_payload, physiciq_payload) from the refresh loop.")
