#!/usr/bin/env python3
"""Stage 1A GT-STC hyperparameter-search dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_stc_hyperparam_search/"
    "latest3350_top100_v1/stage1a_first10"
)
REPORT = ROOT / "search_ranking.json"


def _read() -> dict[str, Any]:
    return json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.is_file() else {}


def _delta(value: Any, reference: Any) -> float | None:
    try:
        return float(value) - float(reference)
    except (TypeError, ValueError):
        return None


def catalog() -> dict[str, Any]:
    report = _read()
    baseline = report.get("baseline") or {}
    baseline_trajectory = baseline.get("trajectory") or {}
    baseline_pixel = baseline.get("pixel") or {}
    rows = [baseline] + list(report.get("guided_ranking") or []) if baseline else []
    for rank, row in enumerate(rows):
        row["rank"] = 0 if row.get("variant") == "baseline" else rank
        trajectory = row.get("trajectory") or {}
        pixel = row.get("pixel") or {}
        row["delta_vs_baseline"] = {
            "ade_d0": _delta(trajectory.get("ade_d0"), baseline_trajectory.get("ade_d0")),
            "track_loss": _delta(
                trajectory.get("future_track_loss_score_0_100"),
                baseline_trajectory.get("future_track_loss_score_0_100"),
            ),
            "target_mse": _delta(
                pixel.get("target_tube_mse_0_1"),
                baseline_pixel.get("target_tube_mse_0_1"),
            ),
            "outside_mse": _delta(
                pixel.get("outside_object_mse_0_1"),
                baseline_pixel.get("outside_object_mse_0_1"),
            ),
        }
        row["video_ready"] = bool(
            row.get("video")
            and Path(str(row["video"])).is_file()
            and Path(str(row["video"])).stat().st_size > 0
        )
    acceptable = [
        row
        for row in rows
        if row.get("variant") != "baseline"
        and bool((row.get("trajectory") or {}).get("quality_pass"))
    ]
    balanced = min(
        acceptable,
        key=lambda row: (
            float((row.get("pixel") or {}).get("target_tube_mse_0_1", float("inf"))),
            float((row.get("trajectory") or {}).get("ade_d0", float("inf"))),
        ),
        default=None,
    )
    return {
        "ready": bool(report),
        "root": str(ROOT),
        "case": report.get("case"),
        "target": report.get("target"),
        "seed": 47326,
        "window": [0, 9],
        "head_group": "latest3350 S039 Top100",
        "selection_order": report.get("selection_order") or [],
        "planned": 12,
        "complete": sum(bool(row.get("video_ready")) for row in rows if row.get("variant") != "baseline"),
        "rows": rows,
        "trajectory_best": report.get("acceptable_winner"),
        "balanced": balanced,
    }


def asset(variant: str) -> Path | None:
    allowed = {
        str(row.get("variant")): Path(str(row.get("video")))
        for row in catalog().get("rows", [])
        if row.get("variant") and row.get("video")
    }
    path = allowed.get(variant)
    return path if path is not None and path.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT-STC · Hyperparameter Search</title><style>
:root{--ink:#14212d;--muted:#607382;--paper:#eaf0f2;--panel:#fbfdfe;--line:#afc0c9;--blue:#165e8b;--orange:#e77f28;--red:#b54141;--green:#17745d;--violet:#6551a3;--shadow:0 16px 44px #203b4e17}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#d7e2e6 1px,transparent 1px),linear-gradient(#d7e2e6 1px,transparent 1px),var(--paper);background-size:26px 26px;font:15px/1.5 "Avenir Next","Segoe UI",sans-serif}a{color:var(--blue)}header{padding:32px clamp(18px,5vw,72px);background:#eef5f7f3;border-bottom:1px solid var(--line)}.back,.mono{font:700 10px ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase}.back{display:inline-block;margin-bottom:22px}.eyebrow{color:var(--blue)}h1{max-width:1150px;margin:7px 0 12px;font:800 clamp(38px,6vw,78px)/.92 "Arial Narrow","Avenir Next Condensed",sans-serif;letter-spacing:-.045em}.lead{max-width:1050px;color:#40596a;font-size:17px}.thesis{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;max-width:1000px;margin-top:24px}.thesis i{height:30px;background:#c6d5db;border:1px solid #a9bbc4}.thesis i.on{background:var(--orange);border-color:#bd6317}.thesis i::after{content:attr(data-label);display:grid;place-items:center;height:100%;color:#fff;font:700 9px ui-monospace,monospace}main{max-width:1700px;margin:auto;padding:22px clamp(14px,4vw,58px) 80px}.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}.stat,.panel,.candidate{background:var(--panel);border:1px solid var(--line);box-shadow:var(--shadow)}.stat{padding:16px}.stat b{display:block;margin:4px 0;font:800 27px "Arial Narrow",sans-serif}.stat small{color:var(--muted)}.panel{margin-top:14px;padding:18px}.panel h2{margin:0 0 13px;font:800 28px "Arial Narrow",sans-serif}.verdicts{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:12px}.verdict{padding:17px;border-left:6px solid var(--blue);background:#f5fafc}.verdict.trade{border-color:var(--orange);background:#fff8ef}.verdict b{display:block;font-size:19px}.verdict .numbers{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px;font:12px ui-monospace,monospace}.scatter-wrap{overflow:auto}.scatter{position:relative;min-width:760px;height:380px;margin:18px 50px 35px 65px;border-left:2px solid var(--ink);border-bottom:2px solid var(--ink);background:repeating-linear-gradient(to right,transparent 0,transparent calc(20% - 1px),#d5e0e4 calc(20% - 1px),#d5e0e4 20%),repeating-linear-gradient(to bottom,transparent 0,transparent calc(25% - 1px),#d5e0e4 calc(25% - 1px),#d5e0e4 25%)}.dot{position:absolute;width:18px;height:18px;border-radius:50%;transform:translate(-50%,50%);border:3px solid #fff;box-shadow:0 1px 7px #243b4d88;cursor:pointer;background:var(--blue)}.dot.point{background:var(--violet)}.dot.combined{background:var(--orange)}.dot.baseline{background:var(--ink);width:24px;height:24px}.dot.fail{background:var(--red)}.dot::after{content:attr(data-label);position:absolute;left:14px;top:-5px;white-space:nowrap;color:var(--ink);font:700 10px ui-monospace,monospace}.axis-x,.axis-y{position:absolute;color:var(--muted);font:700 10px ui-monospace,monospace;text-transform:uppercase}.axis-x{right:0;bottom:-29px}.axis-y{left:-61px;top:0;writing-mode:vertical-rl;transform:rotate(180deg)}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}.toolbar button{padding:8px 12px;border:1px solid #91a8b4;background:#fff;color:var(--ink);cursor:pointer;font-weight:700}.toolbar button.on{background:var(--ink);color:#fff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:12px}.candidate{overflow:hidden;border-top:5px solid var(--blue)}.candidate.point{border-top-color:var(--violet)}.candidate.combined{border-top-color:var(--orange)}.candidate.fail{border-top-color:var(--red)}.candidate.best{outline:3px solid var(--green)}video{display:block;width:100%;aspect-ratio:16/9;background:#0d1c27}.caption{padding:12px 14px 15px}.caption h3{margin:0;font:800 20px "Arial Narrow",sans-serif}.tags{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}.tag{padding:2px 7px;background:#e4edf1;color:#526878;font:700 9px ui-monospace,monospace;text-transform:uppercase}.tag.pass{background:#dcefe8;color:var(--green)}.tag.fail{background:#f5dddd;color:var(--red)}.metrics{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px;font:11px ui-monospace,monospace;color:#405869}.metrics strong{color:var(--ink)}table{width:100%;border-collapse:collapse;min-width:930px}th,td{padding:8px 9px;border-bottom:1px solid #d5e0e4;text-align:left}th{color:var(--muted);font:700 10px ui-monospace,monospace;text-transform:uppercase}.table-wrap{overflow:auto}.positive{color:var(--green)}.negative{color:var(--red)}@media(max-width:800px){.verdicts{grid-template-columns:1fr}.thesis{grid-template-columns:repeat(5,minmax(45px,1fr))}}@media(prefers-reduced-motion:no-preference){.candidate{animation:up .25s ease both}@keyframes up{from{opacity:0;transform:translateY(6px)}}}
</style></head><body><header><a class="back" href="/">← 返回 8092 总入口</a><div class="eyebrow mono">Offline model selection · MSE / CoTracker never backpropagate</div><h1>轨迹收益，花了多少像素代价？</h1><p class="lead">固定 001460、object A、seed 47326、First10、latest3350 Top100，只搜索 Region / Point / Combined 与 λ。轨迹门控优先；MSE 是生成后选参指标，不参与去噪。</p><div class="thesis"><i class="on" data-label="0–9"></i><i data-label="10–19"></i><i data-label="20–29"></i><i data-label="30–39"></i><i data-label="MSE OFFLINE"></i></div></header><main><div id="summary" class="summary"></div><section class="panel"><h2>当前结论</h2><div id="verdicts" class="verdicts"></div></section><section class="panel"><h2>ADE–Target MSE 权衡图</h2><p>越靠左表示 GT 轨迹误差越小，越靠下表示对象区域像素误差越小；红色为轨迹门控失败。</p><div class="scatter-wrap"><div id="scatter" class="scatter"><span class="axis-y">Target-tube GT MSE ↑</span><span class="axis-x">GT Center-ADE / D0 →</span></div></div></section><section class="panel"><h2>全部视频</h2><div class="toolbar" id="filters"></div><div class="grid" id="grid"></div></section><section class="panel"><h2>精确排序</h2><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Variant</th><th>Gate</th><th>Track Loss</th><th>ADE/D0</th><th>ΔADE</th><th>Target MSE</th><th>ΔTarget MSE</th><th>Outside MSE</th></tr></thead><tbody id="tbody"></tbody></table></div></section></main><script>
const api='/api/gt-stc-hyperparam-search',$=x=>document.getElementById(x),F=(v,d=5)=>v==null?'N/A':Number(v).toFixed(d),P=(v,d=1)=>v==null?'N/A':`${v>=0?'+':''}${(100*Number(v)).toFixed(d)}%`,E=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));let D,filter='all';
function mode(r){return r.loss_mode||'baseline'}function tm(r){return r.trajectory||{}}function px(r){return r.pixel||{}}function delta(r){return r.delta_vs_baseline||{}}function card(r){const m=mode(r),t=tm(r),p=px(r),d=delta(r),best=D.trajectory_best?.variant===r.variant;return `<article class="candidate ${m} ${t.quality_pass===false?'fail':''} ${best?'best':''}"><video controls muted playsinline preload="metadata" src="${api}/asset?variant=${encodeURIComponent(r.variant)}"></video><div class="caption"><h3>${r.variant==='baseline'?'Baseline':`${m.toUpperCase()} · λ${r.guidance_scale}`}</h3><div class="tags"><span class="tag">${r.variant==='baseline'?'No guidance':'steps 0–9'}</span><span class="tag ${t.quality_pass?'pass':'fail'}">Gate ${t.quality_pass?'PASS':'FAIL'}</span>${best?'<span class="tag pass">轨迹第一</span>':''}</div><div class="metrics"><span>Track Loss <strong>${F(t.future_track_loss_score_0_100,2)}</strong></span><span>ADE/D0 <strong>${F(t.ade_d0)}</strong></span><span>ΔADE <strong>${F(d.ade_d0)}</strong></span><span>Target MSE <strong>${F(p.target_tube_mse_0_1,7)}</strong></span><span>ΔTarget <strong>${F(d.target_mse,7)}</strong></span><span>Outside MSE <strong>${F(p.outside_object_mse_0_1,7)}</strong></span></div></div></article>`}
function renderGrid(){const rows=D.rows.filter(r=>filter==='all'||mode(r)===filter);$('grid').innerHTML=rows.map(card).join('')}
function render(){const guided=D.rows.filter(r=>r.variant!=='baseline'),passed=guided.filter(r=>tm(r).quality_pass),base=D.rows.find(r=>r.variant==='baseline'),best=D.rows.find(r=>r.variant===D.trajectory_best?.variant),bal=D.balanced;$('summary').innerHTML=`<div class="stat"><span class="mono">Search matrix</span><b>${D.complete}/${D.planned}</b><small>全部候选视频和指标</small></div><div class="stat"><span class="mono">Trajectory gate</span><b>${passed.length}/${guided.length}</b><small>可接受候选</small></div><div class="stat"><span class="mono">Case / seed</span><b>1 / 1</b><small>当前仅 calibration evidence</small></div><div class="stat"><span class="mono">Window</span><b>0–9</b><small>前 10 个 denoising step</small></div>`;const verdict=(r,title,klass)=>{const t=tm(r),p=px(r),d=delta(r);return `<div class="verdict ${klass}"><span class="mono">${title}</span><b>${E(r.variant)}</b><div class="numbers"><span>ADE ${F(t.ade_d0)} (${P(d.ade_d0/base.trajectory.ade_d0)})</span><span>Target MSE ${F(p.target_tube_mse_0_1,7)} (${P(d.target_mse/base.pixel.target_tube_mse_0_1)})</span><span>Outside ${F(p.outside_object_mse_0_1,7)}</span></div></div>`};$('verdicts').innerHTML=verdict(best,'轨迹改善最大','')+verdict(bal,'最低对象 MSE 的门控通过候选','trade');const valid=D.rows.filter(r=>tm(r).raw_ade_d0!=null&&px(r).target_tube_mse_0_1!=null),xs=valid.map(r=>tm(r).raw_ade_d0),ys=valid.map(r=>px(r).target_tube_mse_0_1),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),pos=(v,a,b)=>8+84*(v-a)/Math.max(b-a,1e-9);$('scatter').querySelectorAll('.dot').forEach(x=>x.remove());for(const r of valid){const b=document.createElement('button');b.className=`dot ${mode(r)} ${tm(r).quality_pass?'':'fail'}`;b.style.left=pos(tm(r).raw_ade_d0,xmin,xmax)+'%';b.style.bottom=pos(px(r).target_tube_mse_0_1,ymin,ymax)+'%';b.dataset.label=r.variant==='baseline'?'Base':`${mode(r)[0].toUpperCase()}${r.guidance_scale}`;b.title=`${r.variant}\nADE ${F(tm(r).raw_ade_d0)}\nTarget MSE ${F(px(r).target_tube_mse_0_1,7)}`;b.onclick=()=>{filter=mode(r);renderGrid()};$('scatter').appendChild(b)}$('filters').innerHTML=['all','baseline','region','point','combined'].map(x=>`<button class="${filter===x?'on':''}" data-f="${x}">${x}</button>`).join('');$('filters').onclick=e=>{if(e.target.dataset.f){filter=e.target.dataset.f;$('filters').querySelectorAll('button').forEach(b=>b.classList.toggle('on',b.dataset.f===filter));renderGrid()}};renderGrid();$('tbody').innerHTML=D.rows.map(r=>{const t=tm(r),p=px(r),d=delta(r);return `<tr><td>${r.variant==='baseline'?'Base':r.rank}</td><td>${E(r.variant)}</td><td class="${t.quality_pass?'positive':'negative'}">${t.quality_pass?'PASS':'FAIL'}</td><td>${F(t.future_track_loss_score_0_100,2)}</td><td>${F(t.ade_d0)}</td><td>${F(d.ade_d0)}</td><td>${F(p.target_tube_mse_0_1,7)}</td><td>${F(d.target_mse,7)}</td><td>${F(p.outside_object_mse_0_1,7)}</td></tr>`}).join('')}
fetch(api+'/catalog?v='+Date.now(),{cache:'no-store'}).then(r=>r.json()).then(d=>{D=d;render()});
</script></body></html>'''

