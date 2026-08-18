#!/usr/bin/env python3
"""Build the 67-case Top-3 training methods vs PhysRVG comparison page."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from build_physiciq_physrvg_worst_case_dashboard import (
    DEFAULT_SOURCE_PAGE,
    PRIMARY_METRIC_KEYS,
    build_data,
    load_dashboard_payload,
)


DEFAULT_OUTPUT_DIR = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physiciq-top3-vs-physrvg-all-cases"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-page", type=Path, default=DEFAULT_SOURCE_PAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def average_tied_ranks(
    rows: list[dict[str, Any]], metric_key: str, direction: str
) -> None:
    sign = 1.0 if direction == "lower" else -1.0
    ordered = sorted(
        rows,
        key=lambda row: (
            sign * float(row["means"][metric_key]),
            str(row["label"]),
        ),
    )
    start = 0
    while start < len(ordered):
        end = start + 1
        value = float(ordered[start]["means"][metric_key])
        while end < len(ordered) and abs(
            float(ordered[end]["means"][metric_key]) - value
        ) < 1e-12:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for row in ordered[start:end]:
            row.setdefault("metric_ranks", {})[metric_key] = average_rank
        start = end


def rank_methods(data: dict[str, Any]) -> list[dict[str, Any]]:
    records = {
        (str(record["method_key"]), int(record["step"])): record
        for record in data["records"]
    }
    spec_by_key = {spec["key"]: spec for spec in data["metric_specs"]}
    rows: list[dict[str, Any]] = []
    for method in data["methods"]:
        record = records[(str(method["key"]), int(method["default_step"]))]
        means: dict[str, float] = {}
        for metric_key in PRIMARY_METRIC_KEYS:
            values = [
                record["metrics"][case["stem"]][metric_key]
                for case in data["cases"]
                if metric_key in record["metrics"].get(case["stem"], {})
            ]
            if len(values) != len(data["cases"]):
                raise ValueError(
                    f"Default record {method['key']} step {record['step']} has "
                    f"{len(values)}/{len(data['cases'])} cases for {metric_key}"
                )
            means[metric_key] = float(fmean(values))
        rows.append(
            {
                "key": str(method["key"]),
                "label": str(method["label"]),
                "color": str(method["color"]),
                "step": int(record["step"]),
                "means": means,
                "metric_ranks": {},
            }
        )
    for metric_key in PRIMARY_METRIC_KEYS:
        average_tied_ranks(
            rows,
            metric_key,
            str(spec_by_key[metric_key]["direction"]),
        )
    for row in rows:
        row["average_rank"] = float(fmean(row["metric_ranks"].values()))
    rows.sort(key=lambda row: (float(row["average_rank"]), str(row["label"])))
    for overall_rank, row in enumerate(rows, start=1):
        row["overall_rank"] = overall_rank
    return rows


def primary_only_metrics(
    metrics: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        stem: {
            key: float(values[key])
            for key in PRIMARY_METRIC_KEYS
            if key in values
        }
        for stem, values in metrics.items()
    }


def slim_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "method_key": str(record["method_key"]),
        "method_label": str(record["method_label"]),
        "step": int(record["step"]),
        "videos": dict(record["videos"]),
        "metrics": primary_only_metrics(record["metrics"]),
    }


def build_top3_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = build_data(payload)
    ranking = rank_methods(data)
    top3 = ranking[:3]
    record_lookup = {
        (str(record["method_key"]), int(record["step"])): record
        for record in data["records"]
    }
    top_records = [
        slim_record(record_lookup[(row["key"], int(row["step"]))])
        for row in top3
    ]
    reference_records = {
        name: slim_record(record) for name, record in data["references"].items()
    }
    spec_lookup = {spec["key"]: spec for spec in data["metric_specs"]}
    primary_specs = [spec_lookup[key] for key in PRIMARY_METRIC_KEYS]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_specs": primary_specs,
        "ranking": ranking,
        "top3": top3,
        "top_records": top_records,
        "references": reference_records,
        "cases": data["cases"],
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>PhysicIQ · 综合 Top 3 vs PhysRVG · 67 case</title>
  <style>
    :root{--ice:#edf3f4;--paper:#fbfcfc;--ink:#17262c;--muted:#65767d;--line:#cbd7da;
      --deep:#15323d;--gold:#d6a13a;--silver:#80959d;--bronze:#b16e46;--off:#315c87;
      --on:#0b6e4f;--good:#08745f;--bad:#a94434;--shadow:0 9px 28px rgba(18,46,57,.09)}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--ice);
      color:var(--ink);font-family:Inter,"Noto Sans SC",Arial,sans-serif}a{color:inherit}
    header{padding:24px clamp(15px,4vw,60px) 26px;background:var(--deep);color:#f7fbfc}
    .topline{display:flex;gap:16px;align-items:center}.topline a{color:#a9d6da;text-decoration:none;font-weight:850}
    .stamp{font:750 10px/1 "Arial Narrow",sans-serif;letter-spacing:.14em;text-transform:uppercase;color:#76b8bd}
    h1{margin:18px 0 7px;font:850 clamp(27px,4vw,50px)/1 "Arial Narrow","Roboto Condensed",sans-serif;
      letter-spacing:-.025em}header p{max-width:1050px;margin:0;color:#bed1d6;font-size:13px;line-height:1.6}
    .podium{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:22px}
    .podium-card{position:relative;min-height:100px;padding:13px 14px;background:#1e4350;border-top:5px solid var(--medal)}
    .podium-card .place{position:absolute;right:11px;top:7px;font:900 38px/1 "Arial Narrow",sans-serif;color:rgba(255,255,255,.12)}
    .podium-card h2{max-width:85%;margin:0 0 6px;font-size:13px;line-height:1.35}.podium-card p{margin:0;font-size:10px;color:#a9c1c7}
    .podium-card strong{display:inline-block;margin-top:9px;color:#fff;font:900 18px/1 "Arial Narrow",sans-serif}
    .toolbar{position:sticky;top:0;z-index:20;display:grid;grid-template-columns:auto minmax(300px,1fr) auto auto minmax(190px,.45fr);
      gap:8px;align-items:end;padding:10px clamp(12px,3vw,35px);background:rgba(251,252,252,.97);
      border-bottom:1px solid var(--line);box-shadow:0 5px 18px rgba(18,46,57,.08)}
    label{display:grid;gap:4px;color:var(--muted);font-size:10px;font-weight:850}select,button{height:38px;border:1px solid var(--line);
      border-radius:4px;background:#fff;color:var(--ink);font:750 12px/1 inherit}select{width:100%;padding:0 9px}
    button{padding:0 12px;cursor:pointer}.position{min-width:76px;padding-bottom:10px;color:var(--muted);font:850 11px/1 inherit;text-align:center}
    main{max-width:1920px;margin:auto;padding:20px clamp(10px,2vw,30px) 70px}
    .section-head{display:flex;align-items:end;justify-content:space-between;gap:15px;margin:4px 0 10px}
    .section-head h2{margin:0;font:850 21px/1 "Arial Narrow",sans-serif}.section-head p{max-width:760px;margin:0;color:var(--muted);
      font-size:11px;line-height:1.5;text-align:right}.case-meta{padding:12px 14px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .case-meta h3{margin:0 0 5px;font-size:14px;overflow-wrap:anywhere}.case-meta p{margin:0;color:var(--muted);font-size:11px;line-height:1.55}
    .video-scroll{overflow-x:auto;margin-top:9px;border:1px solid var(--line);box-shadow:var(--shadow)}
    .videos{display:grid;grid-template-columns:repeat(6,minmax(230px,1fr));min-width:1450px;gap:1px;background:var(--line)}
    .video-card{padding:8px;background:var(--paper)}.video-label{display:flex;justify-content:space-between;gap:8px;min-height:42px;
      padding:2px 3px 7px;font-size:10px;font-weight:900;line-height:1.35}.video-label span{max-width:78%}.video-label em{color:var(--muted);
      font-style:normal;text-align:right;font-weight:650}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#0d171b}
    .case-metrics{margin-top:9px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow);overflow:auto}
    table{width:100%;border-collapse:separate;border-spacing:0;font-size:11px;font-variant-numeric:tabular-nums}th,td{padding:8px 9px;
      border-right:1px solid #dce4e6;border-bottom:1px solid #dce4e6;text-align:right;white-space:nowrap}th{position:sticky;top:0;z-index:2;
      background:#dfe8ea;color:#40535a}th:first-child,td:first-child{text-align:left}.method-cell{font-weight:900;max-width:360px;overflow:hidden;text-overflow:ellipsis}
    .delta{display:block;margin-top:2px;font-size:9px;font-weight:850}.delta.good{color:var(--good)}.delta.bad{color:var(--bad)}
    .winner{background:#e5f3ee}.top-gaps{margin-top:24px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .top-gaps .section-head{padding:13px 14px 3px}.top10-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1px;background:var(--line);
      border-top:1px solid var(--line)}.gap-item{display:grid;grid-template-columns:38px minmax(0,1fr) auto;gap:10px;align-items:center;padding:11px 13px;
      background:var(--paper);cursor:pointer}.gap-item:hover,.gap-item.active{background:#fff3df}.gap-rank{font:900 25px/1 "Arial Narrow",sans-serif;color:#a0adb1}
    .gap-main{min-width:0}.gap-method{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px;font-weight:900}.gap-case{margin-top:3px;
      overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:9px}.gap-values{text-align:right;font-variant-numeric:tabular-nums}
    .gap-values strong{display:block;color:var(--bad);font:900 18px/1 "Arial Narrow",sans-serif}.gap-values span{display:block;margin-top:3px;
      color:var(--muted);font-size:9px}.gap-values .ahead{color:var(--good)}.all-cases{margin-top:24px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .all-cases .section-head{padding:13px 14px 3px}.all-wrap{max-height:650px;overflow:auto}.all-wrap th{top:0}.all-wrap th:first-child,
      .all-wrap td:first-child{position:sticky;left:0;z-index:1;background:var(--paper);min-width:315px;text-align:left}.all-wrap th:first-child{z-index:3;background:#dfe8ea}
    .all-wrap tbody tr{cursor:pointer}.all-wrap tbody tr:hover td{background:#fff6e8}.all-wrap tbody tr.active td{background:#f9edca}
    .case-name{display:block;max-width:410px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:800}.best-tag{display:block;margin-top:2px;
      color:var(--good);font-size:9px}.ranking{margin-top:24px;background:var(--paper);border:1px solid var(--line);box-shadow:var(--shadow)}
    .ranking .section-head{padding:13px 14px 3px}.rank-table{max-height:440px;overflow:auto}.rank-table tr.top3 td{background:#fff7df}
    .replay{position:fixed;right:21px;bottom:19px;z-index:30;height:48px;padding:0 17px;border:0;border-radius:24px;background:#b7553e;
      color:#fff;box-shadow:0 8px 22px rgba(105,43,31,.3);cursor:pointer;font-weight:900}.footer{margin-top:16px;color:var(--muted);font-size:10px}
    select:focus-visible,button:focus-visible,a:focus-visible{outline:3px solid var(--gold);outline-offset:2px}
    @media(max-width:900px){.podium{grid-template-columns:1fr}.top10-grid{grid-template-columns:1fr}.toolbar{position:relative;grid-template-columns:auto 1fr auto auto}.toolbar label:last-child{grid-column:1/-1}
      .section-head{display:block}.section-head p{margin-top:6px;text-align:left}}
    @media(max-width:560px){header{padding:18px 13px}.toolbar{grid-template-columns:1fr 1fr}.toolbar label{grid-column:1/-1}.position{padding:10px 0}}
    @media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
  </style>
</head>
<body>
  <header>
    <div class="topline"><a href="../">← 返回 8844 总览</a><a href="../physiciq-vs-physrvg-worst-cases/">最大劣势审计页</a>
      <span class="stamp">67-case five-lane comparison</span></div>
    <h1>综合 Top 3 × PhysRVG 系列</h1>
    <p>选拔规则：每个方案使用最新且四项主要指标完整的 checkpoint；先对 67 个 case 求指标均值，分别排名，
      并列值共享平均名次，再对四项名次取算术平均。下方固定展示胜出的三个方案与 PHYRVG-PhysRVG LoRA OFF / PHYRVG-PhysRVG +LoRA。</p>
    <div id="podium" class="podium"></div>
  </header>
  <nav class="toolbar">
    <button id="prev" title="上一个 case">← 上一个</button>
    <label>Case<select id="case"></select></label>
    <div id="position" class="position"></div>
    <button id="next" title="下一个 case">下一个 →</button>
    <label>全表指标<select id="metric"></select></label>
  </nav>
  <main>
    <section>
      <div class="section-head"><h2>当前 case · GT + 五组结果</h2><p>六路视频使用同一 case；右下角按钮可全部归零并同时播放。</p></div>
      <div class="case-meta"><h3 id="case-title"></h3><p id="prompt"></p></div>
      <div class="video-scroll"><div id="videos" class="videos"></div></div>
      <div class="case-metrics"><table><thead id="case-head"></thead><tbody id="case-body"></tbody></table></div>
    </section>
    <section class="top-gaps">
      <div class="section-head"><h2 id="top10-title">当前指标 · 原始差距 Top 10</h2>
        <p>每个 case 取 Top 3 中最大的 |方案 − PhysRVG 系列最佳|，再选 10 个不重复 case；点击条目切换上方六路视频。</p></div>
      <div id="top10" class="top10-grid"></div>
    </section>
    <section class="all-cases">
      <div class="section-head"><h2>全部 67 case 对比表</h2><p>由上方“全表指标”控制；点击一行打开对应六路视频。</p></div>
      <div class="all-wrap"><table><thead id="all-head"></thead><tbody id="all-body"></tbody></table></div>
    </section>
    <section class="ranking">
      <div class="section-head"><h2>18 个方案的综合排名依据</h2><p>均值先排名，再平均四项名次；此表固定记录本页 Top 3 的来源。</p></div>
      <div class="rank-table"><table><thead id="rank-head"></thead><tbody id="rank-body"></tbody></table></div>
    </section>
    <p class="footer">逐 case 正式指标与视频来自 PhysicIQ 合并页；生成时间：__GENERATED__</p>
  </main>
  <button id="replay" class="replay">↺ 全部重播</button>
  <script>
    const D=__DATA__;const caseSelect=document.getElementById('case');const metricSelect=document.getElementById('metric');
    const specByKey=Object.fromEntries(D.metric_specs.map(spec=>[spec.key,spec]));
    const topByKey=Object.fromEntries(D.top3.map(row=>[row.key,row]));const records=D.top_records;const refs=D.references;
    const medal=['var(--gold)','var(--silver)','var(--bronze)'];let selectedStem=D.cases[0].stem;
    function fmt(value){if(!Number.isFinite(value))return '—';return Math.abs(value)>=10?value.toFixed(2):value.toFixed(3)}
    function value(record,stem,key){const v=record.metrics?.[stem]?.[key];return Number.isFinite(v)?v:null}
    function seriesReference(stem,key){const off=value(refs.off,stem,key),on=value(refs.on,stem,key);const spec=specByKey[key];
      if(off===null)return {label:'PHYRVG-PhysRVG +LoRA',value:on};if(on===null)return {label:'PHYRVG-PhysRVG OFF',value:off};
      return spec.direction==='lower'?(off<=on?{label:'PHYRVG-PhysRVG OFF',value:off}:{label:'PHYRVG-PhysRVG +LoRA',value:on}):
        (off>=on?{label:'PHYRVG-PhysRVG OFF',value:off}:{label:'PHYRVG-PhysRVG +LoRA',value:on})}
    function deltaVsSeries(record,stem,key){const v=value(record,stem,key),ref=seriesReference(stem,key);if(v===null||ref.value===null)return null;
      return specByKey[key].direction==='lower'?ref.value-v:v-ref.value}
    function allMethods(){return [...records,{...refs.off,method_label:'PHYRVG-PhysRVG finetuned DiT · LoRA OFF',reference:'off'},
      {...refs.on,method_label:'PHYRVG-PhysRVG finetuned DiT + LoRA',reference:'on'}]}
    function methodColor(record,index){if(record.reference==='off')return 'var(--off)';if(record.reference==='on')return 'var(--on)';return topByKey[record.method_key].color||medal[index]}
    function renderPodium(){const root=document.getElementById('podium');D.top3.forEach((row,index)=>{const card=document.createElement('article');
      card.className='podium-card';card.style.setProperty('--medal',medal[index]);card.innerHTML=`<span class="place">${index+1}</span><h2>${row.label}</h2>
        <p>step ${row.step} · 四项名次 ${D.metric_specs.map(s=>fmt(row.metric_ranks[s.key])).join(' / ')}</p><strong>平均名次 ${fmt(row.average_rank)}</strong>`;root.append(card)})}
    function renderVideos(caseItem){const root=document.getElementById('videos');root.replaceChildren();
      const entries=[{label:'GT',sub:'49f · 30 FPS',path:caseItem.gt,color:'var(--ink)'},...allMethods().map((record,index)=>({
        label:record.method_label,sub:record.reference?'inference 40':`综合 #${topByKey[record.method_key].overall_rank} · step ${record.step}`,
        path:record.videos[caseItem.stem],color:methodColor(record,index)}))];entries.forEach(entry=>{const card=document.createElement('div');card.className='video-card';
        card.innerHTML=`<div class="video-label" style="color:${entry.color}"><span>${entry.label}</span><em>${entry.sub}</em></div>
          <video src="${entry.path}" muted playsinline controls preload="metadata"></video>`;root.append(card)})}
    function renderCaseMetrics(caseItem){const methods=allMethods();document.getElementById('case-head').innerHTML='<tr><th>方案</th>'+D.metric_specs.map(s=>`<th>${s.label} ↑</th>`).join('')+'</tr>';
      const body=document.getElementById('case-body');body.replaceChildren();const best={};D.metric_specs.forEach(spec=>{const vals=methods.map(r=>value(r,caseItem.stem,spec.key)).filter(Number.isFinite);
        best[spec.key]=spec.direction==='lower'?Math.min(...vals):Math.max(...vals)});methods.forEach((record,index)=>{const tr=document.createElement('tr');
        tr.innerHTML=`<td class="method-cell" style="color:${methodColor(record,index)}">${record.method_label}${record.reference?'':` · step ${record.step}`}</td>`;
        D.metric_specs.forEach(spec=>{const v=value(record,caseItem.stem,spec.key);const td=document.createElement('td');if(v===best[spec.key])td.className='winner';
          let delta='';if(!record.reference){const d=deltaVsSeries(record,caseItem.stem,spec.key);delta=`<span class="delta ${d>=0?'good':'bad'}">vs 系列 ${d>=0?'+':''}${fmt(d)}</span>`}
          td.innerHTML=`${fmt(v)}${delta}`;tr.append(td)});body.append(tr)})}
    function renderCase(stem){const item=D.cases.find(c=>c.stem===stem);if(!item)return;selectedStem=stem;document.querySelectorAll('video').forEach(v=>v.pause());
      document.getElementById('case-title').textContent=item.stem;document.getElementById('prompt').textContent=item.prompt;caseSelect.value=stem;
      document.getElementById('position').textContent=`${D.cases.indexOf(item)+1} / ${D.cases.length}`;renderVideos(item);renderCaseMetrics(item);
      document.querySelectorAll('#all-body tr').forEach(tr=>tr.classList.toggle('active',tr.dataset.stem===stem));
      document.querySelectorAll('#top10 .gap-item').forEach(item=>item.classList.toggle('active',item.dataset.stem===stem))}
    function renderTop10(){const key=metricSelect.value,spec=specByKey[key],rows=[];D.cases.forEach(item=>{const candidates=[];records.forEach(record=>{
        const schemeValue=value(record,item.stem,key),ref=seriesReference(item.stem,key);if(schemeValue===null||ref.value===null)return;
        const signed=spec.direction==='lower'?ref.value-schemeValue:schemeValue-ref.value;candidates.push({record,item,schemeValue,ref,signed,gap:Math.abs(signed)})});
        candidates.sort((a,b)=>b.gap-a.gap||a.record.method_label.localeCompare(b.record.method_label));if(candidates.length)rows.push(candidates[0])});
      rows.sort((a,b)=>b.gap-a.gap||a.record.method_label.localeCompare(b.record.method_label)||a.item.stem.localeCompare(b.item.stem));
      document.getElementById('top10-title').textContent=`${spec.label} · 原始差距 Top 10`;const root=document.getElementById('top10');root.replaceChildren();
      rows.slice(0,10).forEach((row,index)=>{const rank=topByKey[row.record.method_key].overall_rank;const card=document.createElement('article');card.className='gap-item';
        card.dataset.stem=row.item.stem;card.innerHTML=`<div class="gap-rank">${String(index+1).padStart(2,'0')}</div><div class="gap-main">
          <div class="gap-method" style="color:${topByKey[row.record.method_key].color}">综合 #${rank} · ${row.record.method_label} · step ${row.record.step}</div>
          <div class="gap-case" title="${row.item.stem}">${row.item.stem}</div></div><div class="gap-values"><strong class="${row.signed>=0?'ahead':''}">${fmt(row.gap)}</strong>
          <span>${row.signed>=0?'方案领先':'方案落后'} · ${fmt(row.schemeValue)} vs ${fmt(row.ref.value)} ${row.ref.label.replace('PhysRVG ','')}</span></div>`;
        card.onclick=()=>{renderCase(row.item.stem);window.scrollTo({top:document.querySelector('main').offsetTop-55,behavior:'smooth'})};root.append(card)});
      document.querySelectorAll('#top10 .gap-item').forEach(item=>item.classList.toggle('active',item.dataset.stem===selectedStem))}
    function renderAllCases(){const key=metricSelect.value,spec=specByKey[key],methods=allMethods();document.getElementById('all-head').innerHTML='<tr><th>Case</th>'+methods.map(r=>`<th>${r.method_label}</th>`).join('')+'<th>最佳</th></tr>';
      const body=document.getElementById('all-body');body.replaceChildren();D.cases.forEach(item=>{const vals=methods.map(r=>({record:r,value:value(r,item.stem,key)}));
        const sorted=vals.filter(x=>x.value!==null).sort((a,b)=>spec.direction==='lower'?a.value-b.value:b.value-a.value);const winner=sorted[0];const tr=document.createElement('tr');tr.dataset.stem=item.stem;
        tr.innerHTML=`<td><span class="case-name" title="${item.stem}">${item.stem}</span><span class="best-tag">${winner?.record.method_label||'—'}</span></td>`;
        vals.forEach(({record,value:v})=>{const td=document.createElement('td');if(record===winner?.record)td.className='winner';td.textContent=fmt(v);tr.append(td)});
        const bestTd=document.createElement('td');bestTd.textContent=winner?fmt(winner.value):'—';bestTd.className='winner';tr.append(bestTd);tr.onclick=()=>{renderCase(item.stem);window.scrollTo({top:document.querySelector('main').offsetTop-55,behavior:'smooth'})};body.append(tr)});
      renderTop10();renderCase(selectedStem)}
    function renderRanking(){document.getElementById('rank-head').innerHTML='<tr><th>#</th><th>方案 · step</th><th>综合平均名次</th>'+D.metric_specs.map(s=>`<th>${s.label}<br>均值 / 名次</th>`).join('')+'</tr>';
      const body=document.getElementById('rank-body');D.ranking.forEach(row=>{const tr=document.createElement('tr');if(row.overall_rank<=3)tr.className='top3';
        tr.innerHTML=`<td>${row.overall_rank}</td><td class="method-cell">${row.label} · step ${row.step}</td><td>${fmt(row.average_rank)}</td>`+
          D.metric_specs.map(s=>`<td>${fmt(row.means[s.key])} / #${fmt(row.metric_ranks[s.key])}</td>`).join('');body.append(tr)})}
    D.cases.forEach((item,index)=>caseSelect.add(new Option(`${String(index+1).padStart(2,'0')} · ${item.stem}`,item.stem)));
    D.metric_specs.forEach(spec=>metricSelect.add(new Option(`${spec.label} ↑`,spec.key)));metricSelect.value=D.metric_specs[0].key;
    caseSelect.onchange=()=>renderCase(caseSelect.value);metricSelect.onchange=renderAllCases;document.getElementById('prev').onclick=()=>{const i=D.cases.findIndex(c=>c.stem===selectedStem);renderCase(D.cases[(i-1+D.cases.length)%D.cases.length].stem)};
    document.getElementById('next').onclick=()=>{const i=D.cases.findIndex(c=>c.stem===selectedStem);renderCase(D.cases[(i+1)%D.cases.length].stem)};
    document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})});
    renderPodium();renderRanking();renderAllCases();
  </script>
</body>
</html>
'''


def build_dashboard(source_page: Path, output_dir: Path) -> Path:
    data = build_top3_data(load_dashboard_payload(source_page.resolve()))
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
