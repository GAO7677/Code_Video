#!/usr/bin/env python3
"""Interactive latent-frame viewer for all-block, per-head Q@K results."""

from __future__ import annotations

import argparse
import json
import math
import urllib.parse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_headwise_50case")
WIDTH = 896
HEIGHT = 512
MODELS = {
    "gt": {"label": "GT teacher-forced", "step": 29, "color": "#ffc447"},
    "lora": {"label": "LoRA step-000500", "step": 39, "color": "#4fc3f7"},
    "baseline": {"label": "Wan2.2 Baseline", "step": 39, "color": "#ef6f78"},
}


def completed_cases(model: str) -> list[str]:
    case_root = ROOT / model / "cases"
    if not case_root.is_dir():
        return []
    return sorted(
        path.name
        for path in case_root.iterdir()
        if path.is_dir()
        and (path / "predicted_tracks.npz").is_file()
        and (path / "cotracker_pseudo_gt.npz").is_file()
    )


def validate_selection(model: str, case: str, block: int, head: int) -> Path:
    if model not in MODELS:
        raise ValueError("unknown model")
    if not 0 <= block < 30 or not 0 <= head < 24:
        raise ValueError("block/head out of range")
    if case not in completed_cases(model):
        raise ValueError("case is unavailable for this model")
    return ROOT / model / "cases" / case


def finite_list(array: np.ndarray) -> list:
    def sanitize(value):
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value if not isinstance(value, float) or math.isfinite(value) else None

    return sanitize(array.round(3).tolist())


def aggregate_metrics(rows: list[dict], region_type: str) -> dict | None:
    selected = [row for row in rows if row.get("region_type") == region_type and row.get("comparisons", 0) > 0]
    comparisons = sum(int(row["comparisons"]) for row in selected)
    if not comparisons:
        return None
    result = {"comparisons": comparisons}
    for key in ("pck8", "pck16", "pck32", "mean_error_px"):
        result[key] = round(
            sum(float(row[key]) * int(row["comparisons"]) for row in selected) / comparisons,
            3,
        )
    return result


@lru_cache(maxsize=256)
def track_payload(model: str, case: str, block: int, head: int) -> bytes:
    case_dir = validate_selection(model, case, block, head)
    step = MODELS[model]["step"]
    key = f"qk_head{head:02d}_layer{block:02d}_step{step:03d}_predictions"
    with np.load(case_dir / "predicted_tracks.npz", allow_pickle=False) as data:
        if key not in data:
            raise ValueError(f"missing array: {key}")
        predictions = data[key].astype(np.float32)
    with np.load(case_dir / "cotracker_pseudo_gt.npz", allow_pickle=False) as data:
        anchors = data["latent_anchor_frames"].astype(np.int32)
        tracks = data["tracks"].astype(np.float32)
        visibility = data["visibility"].astype(bool)
    anchor_tracks = tracks[anchors]
    anchor_visibility = visibility[anchors]
    rows = json.loads((case_dir / "metrics.json").read_text())
    method = f"qk_head{head:02d}"
    selected_rows = [
        row for row in rows
        if row.get("method") == method
        and int(row.get("layer", -1)) == block
        and int(row.get("step_index", -1)) == step
    ]
    payload = {
        "model": model,
        "model_label": MODELS[model]["label"],
        "case": case,
        "block": block,
        "head": head,
        "step": step,
        "color": MODELS[model]["color"],
        "anchors": anchors.tolist(),
        "predictions": finite_list(predictions),
        "gt": finite_list(anchor_tracks),
        "visibility": anchor_visibility.tolist(),
        "metrics": {
            "objects": aggregate_metrics(selected_rows, "object"),
            "background": aggregate_metrics(selected_rows, "background"),
        },
    }
    return json.dumps(payload, allow_nan=True, separators=(",", ":")).encode()


def source_video(case_dir: Path) -> Path:
    manifest = json.loads((case_dir / "manifest.json").read_text())
    value = manifest.get("gt_video") or manifest.get("context_video")
    if not value:
        raise ValueError("source video missing from manifest")
    return Path(value)


@lru_cache(maxsize=512)
def anchor_frame(model: str, case: str, latent: int) -> bytes:
    case_dir = validate_selection(model, case, 0, 0)
    with np.load(case_dir / "cotracker_pseudo_gt.npz", allow_pickle=False) as data:
        anchors = data["latent_anchor_frames"].astype(np.int32)
    if not 0 <= latent < len(anchors):
        raise ValueError("latent index out of range")
    capture = cv2.VideoCapture(str(source_video(case_dir)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(anchors[latent]))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError("cannot decode source frame")
    source_height, source_width = frame.shape[:2]
    scale = min(WIDTH / source_width, HEIGHT / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    left = (WIDTH - resized_width) // 2
    top = (HEIGHT - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    ok, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("cannot encode source frame")
    return encoded.tobytes()


def catalog_payload() -> bytes:
    models = []
    all_cases: set[str] = set()
    for key, config in MODELS.items():
        cases = completed_cases(key)
        all_cases.update(cases)
        models.append({"key": key, "label": config["label"], "step": config["step"], "cases": cases})
    return json.dumps({
        "models": models,
        "cases": sorted(all_cases),
        "blocks": 30,
        "heads": 24,
        "latent_frames": 7,
        "canvas": [WIDTH, HEIGHT],
    }, separators=(",", ":")).encode()


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Latent Q@K Block × Head Viewer</title><style>
:root{--bg:#101614;--panel:#19211e;--ink:#edf2e9;--muted:#9baaa3;--line:#34413c;--accent:#ffc447}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 85% 5%,#31483e 0,transparent 32rem),var(--bg);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}header{padding:28px clamp(16px,4vw,56px) 20px;border-bottom:1px solid var(--line)}.eyebrow{font:700 11px monospace;letter-spacing:.2em;color:var(--accent)}h1{font-family:Georgia,serif;font-size:clamp(30px,4vw,56px);font-weight:400;margin:8px 0}.intro{color:var(--muted);max-width:1000px;line-height:1.55}.layout{display:grid;grid-template-columns:minmax(0,1fr) 310px;gap:18px;padding:20px clamp(12px,3vw,40px) 48px}.stage{min-width:0}.canvas-wrap{position:relative;background:#050807;border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:0 24px 70px #0008}canvas{display:block;width:100%;aspect-ratio:7/4}.badge{position:absolute;top:12px;left:12px;background:#0a0f0dcc;border:1px solid #ffffff25;border-radius:8px;padding:8px 11px;font:700 12px monospace}.timeline{display:grid;grid-template-columns:repeat(7,1fr);gap:7px;margin-top:12px}.tick{padding:10px 4px;border:1px solid var(--line);border-radius:8px;text-align:center;color:var(--muted);cursor:pointer;background:var(--panel);font:600 11px monospace}.tick.active{color:#111;background:var(--accent);border-color:var(--accent)}.playbar{display:flex;gap:9px;margin-top:10px}.playbar button{flex:1}.controls{background:#151d1aee;border:1px solid var(--line);border-radius:14px;padding:17px;height:max-content;position:sticky;top:14px}.field{margin-bottom:14px}.field label{display:block;color:var(--muted);font:700 11px monospace;letter-spacing:.08em;margin-bottom:6px}select,input,button{width:100%;border:1px solid #46534e;background:#202a26;color:var(--ink);border-radius:8px;padding:10px;font:600 13px inherit}input[type=range]{padding:6px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:9px}.metric{border-top:1px solid var(--line);padding-top:13px;margin-top:13px}.metric h3{font-size:12px;color:var(--accent);margin:0 0 8px}.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;font:12px monospace}.metric-grid span{color:var(--muted)}.legend{font-size:12px;line-height:1.7;color:var(--muted);margin-top:14px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.white{border:2px solid white}.color{background:var(--accent)}#status{font:12px monospace;color:var(--muted);margin-top:10px}@media(max-width:900px){.layout{grid-template-columns:1fr}.controls{position:static;order:-1}.layout{display:flex;flex-direction:column}.controls{order:-1}}
</style></head><body><header><div class="eyebrow">DIFFTRACK · TRUE LATENT ANCHORS</div><h1>Block × Head trajectory microscope</h1><div class="intro">Each view uses the exact latent timeline: latent 0–6 maps to source frames 0, 4, 8, 12, 16, 20, 24. White rings are pseudo-GT; colored points and trails are the selected Q@K block/head. No temporal interpolation.</div></header><div class="layout"><section class="stage"><div class="canvas-wrap"><canvas id="canvas" width="896" height="512"></canvas><div class="badge" id="badge"></div></div><div class="timeline" id="timeline"></div><div class="playbar"><button id="prev">Previous latent</button><button id="play">Play latent timeline</button><button id="next">Next latent</button></div><div id="status"></div></section><aside class="controls"><div class="field"><label>MODEL</label><select id="model"></select></div><div class="field"><label>CASE</label><select id="case"></select></div><div class="pair"><div class="field"><label>BLOCK <b id="blockValue">00</b></label><input id="block" type="range" min="0" max="29" value="0"></div><div class="field"><label>HEAD <b id="headValue">00</b></label><input id="head" type="range" min="0" max="23" value="0"></div></div><div class="metric"><h3>OBJECT METRICS</h3><div class="metric-grid" id="objects"></div></div><div class="metric"><h3>BACKGROUND METRICS</h3><div class="metric-grid" id="background"></div></div><div class="legend"><div><i class="dot white"></i>pseudo-GT at anchor</div><div><i class="dot color"></i>selected Q@K prediction</div><div>Thin connector: instantaneous error</div><div>Trails stop at the selected latent frame</div></div></aside></div>
<script>
const $=s=>document.querySelector(s), canvas=$('#canvas'),ctx=canvas.getContext('2d');let catalog,data,latent=0,timer=null,token=0;const pad=n=>String(n).padStart(2,'0');
async function init(){catalog=await fetch('/api/catalog').then(r=>r.json());catalog.models.forEach(m=>{const o=new Option(`${m.label} (${m.cases.length}/50)`,m.key);o.disabled=!m.cases.length;$('#model').add(o)});const first=catalog.models.find(m=>m.cases.length);$('#model').value=first.key;fillCases();buildTimeline();await loadTrack()}
function currentModel(){return catalog.models.find(m=>m.key===$('#model').value)}function fillCases(){const old=$('#case').value;$('#case').innerHTML='';currentModel().cases.forEach(c=>$('#case').add(new Option(c.replace(/^case_\d+_/,''),c)));if(currentModel().cases.includes(old))$('#case').value=old}
function buildTimeline(){const box=$('#timeline');box.innerHTML='';for(let i=0;i<7;i++){const b=document.createElement('button');b.className='tick';b.onclick=()=>setLatent(i);box.appendChild(b)}}
async function loadTrack(){const mine=++token,model=$('#model').value,cs=$('#case').value,b=$('#block').value,h=$('#head').value;$('#status').textContent='Loading saved Q@K result...';try{const d=await fetch(`/api/track?model=${model}&case=${encodeURIComponent(cs)}&block=${b}&head=${h}`).then(async r=>{if(!r.ok)throw new Error(await r.text());return r.json()});if(mine!==token)return;data=d;document.documentElement.style.setProperty('--accent',d.color);showMetrics('objects',d.metrics.objects);showMetrics('background',d.metrics.background);await draw();$('#status').textContent=`Saved result: block ${pad(d.block)} · head ${pad(d.head)} · step ${pad(d.step)}`}catch(e){$('#status').textContent=e.message}}
function showMetrics(id,m){const el=$('#'+id);el.innerHTML=m?`<span>PCK@8</span><b>${m.pck8.toFixed(1)}%</b><span>PCK@16</span><b>${m.pck16.toFixed(1)}%</b><span>PCK@32</span><b>${m.pck32.toFixed(1)}%</b><span>Mean error</span><b>${m.mean_error_px.toFixed(1)}px</b>`:'<span>No valid points</span>'}
function valid(p){return p&&Number.isFinite(p[0])&&Number.isFinite(p[1])}function line(points,color,width){const q=points.filter(valid);if(q.length<2)return;ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);q.slice(1).forEach(p=>ctx.lineTo(p[0],p[1]));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
async function draw(){if(!data)return;const img=new Image(),src=`/api/frame?model=${data.model}&case=${encodeURIComponent(data.case)}&latent=${latent}`;await new Promise((ok,bad)=>{img.onload=ok;img.onerror=bad;img.src=src});ctx.drawImage(img,0,0,896,512);ctx.fillStyle='#07100db8';ctx.fillRect(0,0,896,54);ctx.font='bold 16px monospace';ctx.fillStyle='#fff';ctx.fillText(`${data.model_label} · B${pad(data.block)} H${pad(data.head)} · latent ${latent} / source frame ${data.anchors[latent]}`,18,23);ctx.font='12px monospace';ctx.fillStyle='#b8c5bf';ctx.fillText(`fixed diffusion step S${pad(data.step)} · exact anchor, no interpolation`,18,43);for(let p=0;p<data.predictions[0].length;p++){line(data.gt.slice(0,latent+1).map((f,i)=>data.visibility[i][p]?f[p]:null),'#ffffff99',1.2);line(data.predictions.slice(0,latent+1).map(f=>f[p]),data.color+'cc',2)}for(let p=0;p<data.predictions[latent].length;p++){const g=data.visibility[latent][p]?data.gt[latent][p]:null,q=data.predictions[latent][p];if(valid(g)&&valid(q)){ctx.beginPath();ctx.moveTo(g[0],g[1]);ctx.lineTo(q[0],q[1]);ctx.strokeStyle='#9aa49f88';ctx.lineWidth=1;ctx.stroke()}if(valid(g)){ctx.beginPath();ctx.arc(g[0],g[1],6,0,Math.PI*2);ctx.fillStyle='#0a0e0c';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}if(valid(q)){ctx.beginPath();ctx.arc(q[0],q[1],4.5,0,Math.PI*2);ctx.fillStyle=data.color;ctx.fill();ctx.strokeStyle='#101613';ctx.lineWidth=1.5;ctx.stroke()}}$('#badge').textContent=`latent ${latent}  →  frame ${data.anchors[latent]}`;[...document.querySelectorAll('.tick')].forEach((b,i)=>{b.classList.toggle('active',i===latent);b.textContent=`L${i} · F${data.anchors[i]}`})}
function setLatent(i){latent=(i+7)%7;draw()}$('#prev').onclick=()=>setLatent(latent-1);$('#next').onclick=()=>setLatent(latent+1);$('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('#play').textContent='Play latent timeline'}else{timer=setInterval(()=>setLatent(latent+1),700);$('#play').textContent='Pause'}};$('#model').onchange=()=>{fillCases();loadTrack()};$('#case').onchange=loadTrack;['block','head'].forEach(id=>$('#'+id).oninput=()=>{$('#'+id+'Value').textContent=pad($('#'+id).value);clearTimeout(window[id+'Delay']);window[id+'Delay']=setTimeout(loadTrack,120)});init();
</script></body></html>'''


BLOCKS_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>All-block latent overlay contact sheet</title><style>
:root{--bg:#ece8dc;--paper:#f8f5eb;--ink:#15201c;--muted:#66726c;--line:#cfc8b8;--accent:#d45436}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 88% 2%,#efc69d 0,transparent 30rem),var(--bg);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}header{padding:34px clamp(16px,4vw,62px) 22px;border-bottom:1px solid var(--line)}.eyebrow{font:700 11px monospace;letter-spacing:.2em;color:var(--accent)}h1{margin:8px 0 7px;font:400 clamp(32px,5vw,62px)/1 Georgia,serif}.intro{max-width:1000px;color:var(--muted);line-height:1.55}.toolbar{position:sticky;top:0;z-index:10;display:grid;grid-template-columns:1.1fr 1.6fr 1fr .8fr auto;gap:12px;align-items:end;padding:13px clamp(16px,4vw,62px);background:#ece8dcf2;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}label{display:block;margin-bottom:5px;color:var(--muted);font:700 10px monospace;letter-spacing:.1em}select,input,button{width:100%;border:1px solid #bcb4a3;border-radius:8px;background:#fffdf7;color:var(--ink);padding:9px;font:600 13px inherit}button{width:auto;min-width:120px;cursor:pointer;background:var(--ink);color:white}.head-field{display:grid;grid-template-columns:1fr 42px;gap:8px;align-items:center}.head-field b{font:700 15px monospace;text-align:center}.status{padding:12px clamp(16px,4vw,62px) 0;color:var(--muted);font:12px monospace}main{padding:16px clamp(12px,3vw,40px) 60px}.block{margin-bottom:17px;background:rgba(248,245,235,.93);border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 10px 28px #5d574716}.block-head{display:flex;justify-content:space-between;align-items:center;padding:10px 13px;border-bottom:1px solid var(--line)}.block-head h2{margin:0;font:700 14px monospace}.rank{display:inline-block;margin-right:9px;color:var(--accent)}.metrics{color:var(--muted);font:11px monospace}.latent-grid{display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));gap:1px;background:#2d3531;overflow-x:auto}.latent{position:relative;min-width:150px;background:#080c0a}.latent canvas{display:block;width:100%;aspect-ratio:7/4}.tag{position:absolute;left:6px;top:6px;padding:4px 6px;border-radius:5px;background:#07100dca;color:white;font:700 10px monospace}.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font:12px monospace}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.gt{border:2px solid white;background:#222}.pred{background:var(--accent)}@media(max-width:820px){.toolbar{grid-template-columns:1fr 1fr}.toolbar button{width:100%}main{padding-inline:7px}.latent-grid{grid-template-columns:repeat(7,150px)}}
</style></head><body><header><div class="eyebrow">DIFFTRACK · 30 BLOCKS ON ONE PAGE</div><h1>Latent overlay contact sheet</h1><div class="intro">Each block groups all seven true latent anchors in one row. All 30 blocks are ranked by object PCK for the selected case, model, and head. Select one global head to compare the same head consistently across blocks.</div><div class="legend"><span><i class="dot gt"></i>pseudo-GT</span><span><i class="dot pred"></i>Q@K prediction</span><span>anchors: F0 · F4 · F8 · F12 · F16 · F20 · F24</span></div></header><section class="toolbar"><div><label>MODEL</label><select id="model"></select></div><div><label>CASE</label><select id="case"></select></div><div><label>HEAD</label><div class="head-field"><input id="head" type="range" min="0" max="23" value="0"><b id="headValue">H00</b></div></div><div><label>SORT BY OBJECT PCK</label><select id="sortMetric"><option value="pck32">PCK@32</option><option value="pck16">PCK@16</option><option value="pck8">PCK@8</option></select></div><button id="reload">Render all blocks</button></section><div class="status" id="status">Loading catalog...</div><main id="blocks"></main>
<script>
const $=s=>document.querySelector(s),CW=280,CH=160,SX=CW/896,SY=CH/512;let catalog,renderToken=0;const pad=n=>String(n).padStart(2,'0'),valid=p=>p&&Number.isFinite(p[0])&&Number.isFinite(p[1]);
async function init(){catalog=await fetch('/api/catalog',{cache:'no-store'}).then(r=>r.json());catalog.models.forEach(m=>{const o=new Option(`${m.label} (${m.cases.length}/50)`,m.key);o.disabled=!m.cases.length;$('#model').add(o)});const first=catalog.models.find(m=>m.cases.length);$('#model').value=first.key;fillCases();await renderAll()}
function currentModel(){return catalog.models.find(m=>m.key===$('#model').value)}function fillCases(){const old=$('#case').value;$('#case').innerHTML='';currentModel().cases.forEach(c=>$('#case').add(new Option(c.replace(/^case_\d+_/,''),c)));if(currentModel().cases.includes(old))$('#case').value=old}
function loadImage(src){return new Promise((ok,bad)=>{const im=new Image();im.onload=()=>ok(im);im.onerror=bad;im.src=src})}function line(ctx,points,color,width){const q=points.filter(valid);if(q.length<2)return;ctx.beginPath();ctx.moveTo(q[0][0]*SX,q[0][1]*SY);q.slice(1).forEach(p=>ctx.lineTo(p[0]*SX,p[1]*SY));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(canvas,img,data,latent){const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0,CW,CH);for(let p=0;p<data.predictions[0].length;p++){line(ctx,data.gt.slice(0,latent+1).map((f,i)=>data.visibility[i][p]?f[p]:null),'#ffffffb8',.8);line(ctx,data.predictions.slice(0,latent+1).map(f=>f[p]),data.color+'dd',1.35)}for(let p=0;p<data.predictions[latent].length;p++){const g=data.visibility[latent][p]?data.gt[latent][p]:null,q=data.predictions[latent][p];if(valid(g)&&valid(q)){ctx.beginPath();ctx.moveTo(g[0]*SX,g[1]*SY);ctx.lineTo(q[0]*SX,q[1]*SY);ctx.strokeStyle='#b0b8b288';ctx.lineWidth=.65;ctx.stroke()}if(valid(g)){ctx.beginPath();ctx.arc(g[0]*SX,g[1]*SY,3.2,0,Math.PI*2);ctx.fillStyle='#111';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke()}if(valid(q)){ctx.beginPath();ctx.arc(q[0]*SX,q[1]*SY,2.6,0,Math.PI*2);ctx.fillStyle=data.color;ctx.fill();ctx.strokeStyle='#111';ctx.lineWidth=.7;ctx.stroke()}}}
function metricText(data,key){const m=data.metrics.objects,label=key.replace('pck','PCK@');return m?`${label} ${m[key].toFixed(1)}% · PCK@32 ${m.pck32.toFixed(1)}% · error ${m.mean_error_px.toFixed(1)}px`:'no valid object points'}
async function renderAll(){const mine=++renderToken,model=$('#model').value,cs=$('#case').value,head=Number($('#head').value),sortKey=$('#sortMetric').value;$('#status').textContent=`Loading 30 saved block results for ${cs} · H${pad(head)}...`;$('#blocks').innerHTML='';try{const frames=await Promise.all([...Array(7)].map((_,i)=>loadImage(`/api/frame?model=${model}&case=${encodeURIComponent(cs)}&latent=${i}`)));const tracks=await Promise.all([...Array(30)].map((_,b)=>fetch(`/api/track?model=${model}&case=${encodeURIComponent(cs)}&block=${b}&head=${head}`,{cache:'no-store'}).then(async r=>{if(!r.ok)throw new Error(await r.text());return r.json()})));if(mine!==renderToken)return;document.documentElement.style.setProperty('--accent',tracks[0].color);const ranked=tracks.map((data,block)=>({data,block,score:data.metrics.objects?.[sortKey]??-Infinity})).sort((a,b)=>b.score-a.score||a.block-b.block);const frag=document.createDocumentFragment();ranked.forEach(({data,block},rank)=>{const article=document.createElement('article');article.className='block';article.innerHTML=`<div class="block-head"><h2><span class="rank">RANK ${pad(rank+1)}</span>BLOCK ${pad(block)} · HEAD ${pad(head)} · STEP ${pad(data.step)}</h2><span class="metrics">${metricText(data,sortKey)}</span></div><div class="latent-grid"></div>`;const grid=article.querySelector('.latent-grid');frames.forEach((img,i)=>{const cell=document.createElement('div');cell.className='latent';const canvas=document.createElement('canvas');canvas.width=CW;canvas.height=CH;const tag=document.createElement('span');tag.className='tag';tag.textContent=`L${i} / F${data.anchors[i]}`;cell.append(canvas,tag);grid.append(cell);draw(canvas,img,data,i)});frag.append(article)});$('#blocks').append(frag);$('#status').textContent=`Ranked 30 blocks by object ${sortKey.replace('pck','PCK@')} ↓ · 210 overlays · ${currentModel().label} · ${cs} · H${pad(head)}`}catch(e){$('#status').textContent=`Render failed: ${e.message}`}}
$('#model').onchange=()=>{fillCases();renderAll()};$('#case').onchange=renderAll;$('#sortMetric').onchange=renderAll;$('#head').oninput=()=>{$('#headValue').textContent='H'+pad($('#head').value);clearTimeout(window.headDelay);window.headDelay=setTimeout(renderAll,180)};$('#reload').onclick=renderAll;init();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache" if content_type.startswith("text/html") else "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if parsed.path in ("/", "/all-blocks"):
                self.send_bytes(BLOCKS_PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/single":
                self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/catalog":
                self.send_bytes(catalog_payload(), "application/json")
            elif parsed.path == "/api/track":
                self.send_bytes(track_payload(
                    query["model"][0], query["case"][0],
                    int(query["block"][0]), int(query["head"][0]),
                ), "application/json")
            elif parsed.path == "/api/frame":
                self.send_bytes(anchor_frame(
                    query["model"][0], query["case"][0], int(query["latent"][0]),
                ), "image/jpeg")
            else:
                self.send_bytes(b"not found", "text/plain", 404)
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as error:
            self.send_bytes(str(error).encode(), "text/plain; charset=utf-8", 400)

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Latent viewer: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
