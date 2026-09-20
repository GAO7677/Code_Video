#!/usr/bin/env python3
"""Build a static overlay viewer for the sampling invariance experiment."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT = Path("/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918")
RUN = PROJECT / "validation_20260920/resampling_train_compare_v2"
OUT = PROJECT / "validation_20260920/resampling_overlay_view"


def read_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    archive = np.load(RUN / "predictions.npz", allow_pickle=False)
    seeds = list(range(20260920, 20260940))
    cases = ["control_door_00", "control_door_01", "control_door_02", "control_door_03"]
    target = archive["target"][:, 0, :, :2]
    payload = {
        "schema": "resampling_overlay_v1",
        "title": "Future Query Predictor · Sampling overlay",
        "subtitle": "Same histories, same point clouds per seed; fixed sampling vs per-step resampling",
        "fps": 30,
        "observed_frames": list(range(8)),
        "future_frames": list(range(8, 49)),
        "cases": cases,
        "seeds": seeds,
        "seed_roles": {
            str(seed): ("legacy development probe" if seed <= 20260923 else "new unseen sampling seed")
            for seed in seeds
        },
        "target_xy": target.tolist(),
        "pred_xy": {"fixed": {}, "resample": {}},
        "case_metrics": read_csv(RUN / "resampling_case_metrics.csv"),
        "pair_metrics": read_csv(RUN / "resampling_pair_metrics.csv"),
        "stability": read_csv(RUN / "sampling_stability.csv"),
    }
    for arm, prefix in [("fixed", "dense_fixed"), ("resample", "dense_resample_train")]:
        for seed in seeds:
            payload["pred_xy"][arm][str(seed)] = archive[f"{prefix}_seed{seed}"][:, :, :2].tolist()
    (OUT / "data.json").write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    (OUT / "index.html").write_text(HTML)
    print(OUT)


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Future Query Predictor · Sampling Overlay</title>
<style>
:root{--ink:#17232a;--muted:#60717b;--line:#dbe3e7;--paper:#f5f8f9;--panel:#fff;--fixed:#d45c51;--resample:#167b83;--gt:#1c252b;--accent:#e4a52b}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
header{padding:28px clamp(18px,4vw,54px) 22px;background:#102e35;color:#f7fbfa;border-bottom:4px solid var(--accent)}
header h1{margin:0;font-size:clamp(23px,3vw,34px);letter-spacing:.01em}header p{margin:8px 0 0;color:#bed0d2;max-width:900px}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:18px;max-width:1500px;margin:20px auto;padding:0 18px 48px}
.panel{background:var(--panel);border:1px solid var(--line);box-shadow:0 5px 18px #1939420d;border-radius:8px;padding:16px}
.controls{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:end}.control{display:grid;gap:4px;color:var(--muted);font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase}.control select,.control input{height:34px;border:1px solid #b9c7cc;border-radius:5px;background:white;color:var(--ink);padding:0 9px;font-size:13px;text-transform:none;letter-spacing:0}.check{display:flex;align-items:center;gap:6px;height:34px;color:var(--ink);font-size:13px;font-weight:600;text-transform:none;letter-spacing:0}.check input{width:15px;height:15px}
.canvas-wrap{margin-top:14px;border:1px solid var(--line);background:#fcfdfd;position:relative;aspect-ratio:1.52/1;min-height:390px}.canvas-wrap canvas{display:block;width:100%;height:100%}.legend{display:flex;flex-wrap:wrap;gap:11px;margin-top:9px;color:var(--muted);font-size:12px}.legend span{display:inline-flex;align-items:center;gap:5px}.swatch{width:18px;height:3px;border-radius:3px;display:inline-block}.swatch.gt{background:var(--gt)}.swatch.fixed{background:var(--fixed)}.swatch.resample{background:var(--resample)}.swatch.faint{opacity:.25}
h2{margin:0 0 12px;font-size:16px}h3{margin:20px 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.metric-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.metric{border:1px solid var(--line);border-radius:6px;padding:9px;background:#fbfcfc}.metric b{display:block;font-size:18px}.metric small{color:var(--muted);font-size:11px}.note{color:var(--muted);font-size:12px;margin:10px 0 0}.status{display:inline-block;padding:4px 7px;border-radius:4px;background:#edf6f5;color:#176b70;font-size:11px;font-weight:700}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:6px}.table-wrap table{width:100%;border-collapse:collapse;font-size:11px;min-width:520px}.table-wrap th,.table-wrap td{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}.table-wrap th:first-child,.table-wrap td:first-child{text-align:left}.table-wrap tr:last-child td{border-bottom:0}.table-wrap th{background:#edf2f3;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em}.pass{color:#167b83;font-weight:700}.warn{color:#ad6815;font-weight:700}
.foot{grid-column:1/-1;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:14px}.error{color:#a53e32;padding:12px}
@media(max-width:980px){main{grid-template-columns:1fr}.foot{grid-column:auto}.canvas-wrap{min-height:320px}}
</style></head>
<body><header><h1>Future Query Predictor · Sampling Overlay</h1><p>轨迹叠加展示：固定采样与每步重采样。粗线是当前选择的 seed，淡线是同一 arm 的其他采样 seed，黑线是真值。所有 seed 共享同一历史和对应的静态几何池。</p></header>
<main><section class="panel"><h2>Trajectory overlay</h2><div class="controls">
<label class="control">View<select id="view"><option value="case">single case</option><option value="pair">pair response</option></select></label>
<label class="control" id="case-label">Case<select id="case"></select></label>
<label class="control" id="pair-label" hidden>Pair<select id="pair"></select></label>
<label class="control">Arm<select id="arm"><option value="resample">dense_resample_train</option><option value="fixed">dense_fixed</option></select></label>
<label class="control">Sampling seed<select id="seed"></select></label>
<label class="check"><input id="all-seeds" type="checkbox" checked> all seeds</label>
<label class="check"><input id="show-gt" type="checkbox" checked> ground truth</label>
</div><div class="canvas-wrap"><canvas id="plot"></canvas></div><div class="legend"><span><i class="swatch gt"></i>GT</span><span><i class="swatch fixed"></i>fixed</span><span><i class="swatch resample"></i>resample</span><span><i class="swatch faint"></i>other seeds</span></div><p class="note" id="plot-note"></p></section>
<aside class="panel"><h2 id="summary-title">Selected case</h2><div id="status" class="status">loading</div><div class="metric-grid" id="metrics"></div><h3>Sampling stability</h3><div class="table-wrap"><table><thead><tr><th>case</th><th>mean</th><th>p95</th><th>max</th></tr></thead><tbody id="stability-body"></tbody></table></div><h3>Pair response</h3><div class="table-wrap"><table><thead><tr><th>pair</th><th>GT mm</th><th>R fixed</th><th>R resample</th></tr></thead><tbody id="pair-body"></tbody></table></div><p class="note">02--03 是相同未来的负对照；R 越低越接近真实场景响应。采样种子评测复用四个已消耗的 development history，不构成 OOD 结论。</p></aside>
<div class="foot">Projection: world X/Y plane; RGB7 is the observed endpoint and RGB8--RGB48 are future predictions. Data source: validation_20260920/resampling_train_compare_v2.</div></main>
<script>
const $=id=>document.getElementById(id);let D=null;
const fmt=(v,d=1)=>Number.isFinite(+v)?(+v).toFixed(d):'—';
function populate(){
 $('case').innerHTML=D.cases.map((x,i)=>`<option value="${i}">${x}</option>`).join('');
 const pairs=[];for(let i=0;i<D.cases.length;i++)for(let j=i+1;j<D.cases.length;j++)pairs.push({id:`${String(i).padStart(2,'0')}--${String(j).padStart(2,'0')}`,a:i,b:j});
 $('pair').innerHTML=pairs.map(p=>`<option value="${p.id}">${D.cases[p.a]} vs ${D.cases[p.b]}</option>`).join('');
 $('seed').innerHTML=D.seeds.map(s=>`<option value="${s}">${s} · ${D.seed_roles[String(s)]}</option>`).join('');$('seed').value='20260924';
 renderStability();renderPairs();render();
}
function updateControls(){const pair=$('view').value==='pair';$('case-label').hidden=pair;$('pair-label').hidden=!pair;}
function rangeFor(indices){let xs=[],ys=[];for(const i of indices){const t=D.target_xy[i];xs.push(...t.map(p=>p[0]));ys.push(...t.map(p=>p[1]));for(const arm of ['fixed','resample'])for(const s of D.seeds){const a=D.pred_xy[arm][String(s)][i];xs.push(...a.map(p=>p[0]));ys.push(...a.map(p=>p[1]));}}let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);const px=(xmax-xmin)*.1||.1,py=(ymax-ymin)*.1||.1;return [xmin-px,xmax+px,ymin-py,ymax+py];}
function drawLine(ctx,arr,xf,yf,color,width,alpha=1,dash=[]){ctx.save();ctx.strokeStyle=color;ctx.globalAlpha=alpha;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();arr.forEach((p,i)=>i?ctx.lineTo(xf(p[0]),yf(p[1])):ctx.moveTo(xf(p[0]),yf(p[1])));ctx.stroke();ctx.restore();}
function drawPoint(ctx,x,y,color,r=4){ctx.fillStyle=color;ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();}
function render(){if(!D)return;updateControls();const canvas=$('plot'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const W=rect.width,H=rect.height;const pair=$('view').value==='pair';let indices=[];let labels=[];if(pair){const [a,b]=$('pair').value.split('--').map(Number);indices=[a,b];labels=[D.cases[a],D.cases[b]];}else{indices=[+$('case').value];labels=[D.cases[indices[0]]];}const [xmin,xmax,ymin,ymax]=rangeFor(indices),mx=52,my=30,bx=W-mx,by=H-my,xf=x=>mx+(x-xmin)/(xmax-xmin)*(bx-mx),yf=y=>by-(y-ymin)/(ymax-ymin)*(by-my);ctx.fillStyle='#fcfdfd';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#dfe7e9';ctx.lineWidth=1;for(let i=0;i<6;i++){const x=mx+(bx-mx)*i/5,y=my+(by-my)*i/5;ctx.beginPath();ctx.moveTo(x,my);ctx.lineTo(x,by);ctx.stroke();ctx.beginPath();ctx.moveTo(mx,y);ctx.lineTo(bx,y);ctx.stroke();}ctx.fillStyle='#60717b';ctx.font='11px system-ui';ctx.fillText(`x ${xmin.toFixed(2)} .. ${xmax.toFixed(2)} m`,mx,17);ctx.fillText(`y ${ymin.toFixed(2)} .. ${ymax.toFixed(2)} m`,W-145,17);
 const arm=$('arm').value,seed=+$('seed').value,all=$('all-seeds').checked,showGT=$('show-gt').checked,colors=['#d45c51','#167b83'];indices.forEach((idx,k)=>{const color=colors[k%2];if(showGT)drawLine(ctx,D.target_xy[idx],xf,yf,'#1c252b',3,1);if(all)for(const s of D.seeds)if(s!==seed)drawLine(ctx,D.pred_xy[arm][String(s)][idx],xf,yf,color,1.25,.16);drawLine(ctx,D.pred_xy[arm][String(seed)][idx],xf,yf,color,2.8,1);const t=D.target_xy[idx][0];drawPoint(ctx,xf(t[0]),yf(t[1]),'#1c252b',4);const p=D.pred_xy[arm][String(seed)][idx][0];drawPoint(ctx,xf(p[0]),yf(p[1]),color,3);ctx.fillStyle=color;ctx.font='700 12px system-ui';ctx.fillText(labels[k],xf(p[0])+7,yf(p[1])-8);});
 $('plot-note').textContent=pair?`Pair overlay: ${labels.join(' vs ')} · arm ${arm} · seed ${seed} · black is each case GT.`:`${labels[0]} · arm ${arm} · seed ${seed} (${D.seed_roles[String(seed)]}) · lines start at RGB8; RGB7 is the observed endpoint.`;renderMetrics(indices,seed,arm);
}
function renderMetrics(indices,seed,arm){if(indices.length!==1){$('summary-title').textContent='Pair overlay';$('status').textContent='selected pair';$('metrics').innerHTML='<div class="metric"><b>—</b><small>pair details below</small></div>';return;}const idx=indices[0],caseName=D.cases[idx],rows=D.case_metrics.filter(r=>r.arm===(arm==='fixed'?'dense_fixed':'dense_resample_train')&&r.seed===String(seed)&&r.case===caseName);const r=rows[0]||{};$('summary-title').textContent=caseName;$('status').textContent=`${arm==='fixed'?'fixed':'per-step resample'} · seed ${seed}`;$('metrics').innerHTML=[['ADE',fmt(+r.ADE_m*1000)+' mm'],['FDE',fmt(+r.FDE_m*1000)+' mm'],['radius ADE',fmt(r.radius_normalized_ADE,3)],['max penetration',fmt(+r.penetration_max_depth_m*1000)+' mm']].map(x=>`<div class="metric"><b>${x[1]}</b><small>${x[0]}</small></div>`).join('');}
function renderStability(){const rows=D.stability;const grouped={};rows.forEach(r=>(grouped[r.arm]??=[]).push(r));$('stability-body').innerHTML=D.cases.map(c=>{const f=grouped.dense_fixed.find(r=>r.case===c),s=grouped.dense_resample_train.find(r=>r.case===c);return `<tr><td>${c.replace('control_door_','')}</td><td>${fmt(+s.S_case_mean_m*1000)}</td><td>${fmt(+s.S_case_p95_m*1000)}</td><td>${fmt(+s.S_case_max_m*1000)}</td></tr>`}).join('');}
function renderPairs(){const rows=D.pair_metrics.filter(r=>r.seed==='20260924'&&r.nondegenerate==='True');const map={};rows.forEach(r=>{const key=`${r.case_a.replace('control_door_','')}--${r.case_b.replace('control_door_','')}`;(map[key]??={})[r.arm]=r;});const keys=['00--01','00--02','00--03','01--02','01--03'];$('pair-body').innerHTML=keys.map(k=>{const r=map[k]||{};return `<tr><td>${k}</td><td>${r.dense_fixed?fmt(+r.dense_fixed.D_gt_m*1000):'0'}</td><td>${r.dense_fixed?fmt(r.dense_fixed.R_delta,2):'—'}</td><td>${r.dense_resample_train?fmt(r.dense_resample_train.R_delta,2):'—'}</td></tr>`}).join('')+`<tr><td>02--03</td><td>0</td><td class="warn">negative</td><td class="pass">negative</td></tr>`;}
['view','case','pair','arm','seed','all-seeds','show-gt'].forEach(id=>$(id).addEventListener('change',render));window.addEventListener('resize',render);fetch('data.json?ts='+Date.now()).then(r=>r.json()).then(d=>{D=d;populate()}).catch(e=>{$('plot-note').innerHTML='<span class="error">Cannot load data.json: '+e+'</span>'});
</script></body></html>'''


if __name__ == "__main__":
    main()
