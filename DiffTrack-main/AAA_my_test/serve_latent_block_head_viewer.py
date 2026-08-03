#!/usr/bin/env python3
"""Interactive latent-frame viewer for all-block, per-head Q@K results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_headwise_50case")
ALL_STEPS_ROOT = Path("/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case")
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


def allstep_completed_cases(model: str) -> list[str]:
    case_root = ALL_STEPS_ROOT / model / "cases"
    if not case_root.is_dir():
        return []
    return sorted(
        path.name for path in case_root.iterdir()
        if path.is_dir()
        and (path / "complete.json").is_file()
        and (path / "predicted_tracks.npz").is_file()
        and (path / "metrics.json").is_file()
    )


def validate_allstep_selection(model: str, case: str, block: int, head: int, step: int) -> Path:
    if model not in MODELS:
        raise ValueError("unknown model")
    if not 0 <= block < 30 or not 0 <= head < 24 or not 0 <= step < 40:
        raise ValueError("step/block/head out of range")
    if case not in allstep_completed_cases(model):
        raise ValueError("case is unavailable for this model")
    return ALL_STEPS_ROOT / model / "cases" / case


def allstep_catalog_payload() -> bytes:
    top = json.loads((ALL_STEPS_ROOT / "top_combinations.json").read_text())
    models = []
    for key, config in MODELS.items():
        best = top[key][0]
        models.append({
            "key": key,
            "label": config["label"],
            "color": config["color"],
            "cases": allstep_completed_cases(key),
            "best": {
                "step": int(best["step"]),
                "block": int(best["layer"]),
                "head": int(best["head"]),
                "pck32": float(best["macro_pck32"]),
                "error": float(best["macro_mean_error_px"]),
            },
        })
    return json.dumps({
        "models": models,
        "steps": 40,
        "blocks": 30,
        "heads": 24,
        "latent_frames": 7,
        "combinations": 4_320_000,
    }, separators=(",", ":")).encode()


def summary_row(row: dict) -> dict:
    return {
        "model": row["model"],
        "scope": row["scope"],
        "step": int(row["step"]),
        "block": int(row["layer"]),
        "head": int(row["head"]),
        "timestep": float(row["timestep"]),
        "sigma": float(row["sigma"]),
        "total_cases": 50,
        "valid_cases": int(row["cases"]),
        "comparisons": int(row["comparisons"]),
        **{
            key: float(row[key])
            for key in (
                "macro_pck8", "macro_pck16", "macro_pck32", "macro_mean_error_px",
                "pooled_pck8", "pooled_pck16", "pooled_pck32", "pooled_mean_error_px",
            )
        },
    }


@lru_cache(maxsize=256)
def allstep_rankings_payload(model: str, scope: str, step: int) -> bytes:
    if model not in MODELS or scope not in ("objects", "background") or not 0 <= step < 40:
        raise ValueError("invalid all-step ranking selection")
    rows = []
    with (ALL_STEPS_ROOT / "block_step_head_summary.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["model"] == model and row["scope"] == scope and int(row["step"]) == step:
                rows.append(summary_row(row))
    return json.dumps({
        "model": model,
        "model_label": MODELS[model]["label"],
        "scope": scope,
        "step": step,
        "rows": rows,
    }, separators=(",", ":")).encode()


@lru_cache(maxsize=64)
def allstep_profile_payload(model: str, scope: str, metric: str) -> bytes:
    allowed = {"macro_pck8", "macro_pck16", "macro_pck32", "macro_mean_error_px"}
    if model not in MODELS or scope not in ("objects", "background") or metric not in allowed:
        raise ValueError("invalid all-step profile selection")
    best: dict[int, dict] = {}
    lower_is_better = metric == "macro_mean_error_px"
    with (ALL_STEPS_ROOT / "block_step_head_summary.csv").open() as handle:
        for raw in csv.DictReader(handle):
            if raw["model"] != model or raw["scope"] != scope:
                continue
            row = summary_row(raw)
            previous = best.get(row["step"])
            if previous is None or (
                row[metric] < previous[metric] if lower_is_better else row[metric] > previous[metric]
            ):
                best[row["step"]] = row
    return json.dumps({"metric": metric, "rows": [best[step] for step in range(40)]}, separators=(",", ":")).encode()


@lru_cache(maxsize=128)
def allstep_global_rankings_payload(model: str, scope: str, metric: str, page: int) -> bytes:
    allowed = {"macro_pck8", "macro_pck16", "macro_pck32", "macro_mean_error_px"}
    if model not in MODELS or scope not in ("objects", "background") or metric not in allowed or page < 0:
        raise ValueError("invalid global ranking selection")
    rows = []
    with (ALL_STEPS_ROOT / "block_step_head_summary.csv").open() as handle:
        for raw in csv.DictReader(handle):
            if raw["model"] == model and raw["scope"] == scope:
                rows.append(summary_row(raw))
    if metric == "macro_mean_error_px":
        rows.sort(key=lambda row: (row[metric], -row["macro_pck32"], row["step"], row["block"], row["head"]))
    else:
        rows.sort(key=lambda row: (-row[metric], row["macro_mean_error_px"], row["step"], row["block"], row["head"]))
    page_size = 100
    start = page * page_size
    selected = rows[start:start + page_size]
    for index, row in enumerate(selected, start=start + 1):
        row["global_rank"] = index
    return json.dumps({
        "model": model,
        "scope": scope,
        "metric": metric,
        "page": page,
        "page_size": page_size,
        "total": len(rows),
        "pages": math.ceil(len(rows) / page_size),
        "rows": selected,
    }, separators=(",", ":")).encode()


@lru_cache(maxsize=64)
def combined_rankings_payload(scope: str, metric: str, page: int) -> bytes:
    allowed = {"macro_pck8", "macro_pck16", "macro_pck32", "macro_mean_error_px"}
    if scope not in ("objects", "background") or metric not in allowed or page < 0:
        raise ValueError("invalid combined ranking selection")
    integer_fields = {"step", "block", "head", "models", "valid_cases", "total_cases", "comparisons"}
    rows = []
    with (ALL_STEPS_ROOT / "three_model_combined_summary.csv").open() as handle:
        for raw in csv.DictReader(handle):
            if raw["scope"] != scope:
                continue
            rows.append({
                key: int(value) if key in integer_fields else float(value) if key != "scope" else value
                for key, value in raw.items()
                if key != "rank_macro_pck32"
            })
    if metric == "macro_mean_error_px":
        rows.sort(key=lambda row: (row[metric], -row["macro_pck32"], -row["worst_model_macro_pck32"]))
    else:
        rows.sort(key=lambda row: (-row[metric], row["macro_mean_error_px"], -row["worst_model_macro_pck32"]))
    page_size = 100
    start = page * page_size
    selected = rows[start:start + page_size]
    for index, row in enumerate(selected, start=start + 1):
        row["combined_rank"] = index
    return json.dumps({
        "scope": scope,
        "metric": metric,
        "page": page,
        "page_size": page_size,
        "total": len(rows),
        "pages": math.ceil(len(rows) / page_size),
        "rows": selected,
    }, separators=(",", ":")).encode()


@lru_cache(maxsize=256)
def combined_selection_payload(scope: str, metric: str, step: int, head: int) -> bytes:
    allowed = {"macro_pck8", "macro_pck16", "macro_pck32", "macro_mean_error_px"}
    if scope not in ("objects", "background") or metric not in allowed or not 0 <= step < 40 or not 0 <= head < 24:
        raise ValueError("invalid combined overlay selection")
    integer_fields = {"step", "block", "head", "models", "valid_cases", "total_cases", "comparisons"}
    rows = []
    with (ALL_STEPS_ROOT / "three_model_combined_summary.csv").open() as handle:
        for raw in csv.DictReader(handle):
            if raw["scope"] != scope:
                continue
            rows.append({
                key: int(value) if key in integer_fields else float(value) if key != "scope" else value
                for key, value in raw.items()
                if key != "rank_macro_pck32"
            })
    if metric == "macro_mean_error_px":
        rows.sort(key=lambda row: (row[metric], -row["macro_pck32"], -row["worst_model_macro_pck32"]))
    else:
        rows.sort(key=lambda row: (-row[metric], row["macro_mean_error_px"], -row["worst_model_macro_pck32"]))
    selected = []
    for rank, row in enumerate(rows, start=1):
        if row["step"] == step and row["head"] == head:
            row["combined_rank"] = rank
            selected.append(row)
    selected.sort(key=lambda row: row["combined_rank"])
    return json.dumps({
        "scope": scope,
        "metric": metric,
        "step": step,
        "head": head,
        "total": len(rows),
        "rows": selected,
    }, separators=(",", ":")).encode()


@lru_cache(maxsize=24)
def allstep_tracks_payload(model: str, case: str, step: int, head: int) -> bytes:
    case_dir = validate_allstep_selection(model, case, 0, head, step)
    with np.load(case_dir / "cotracker_pseudo_gt.npz", allow_pickle=False) as data:
        anchors = data["latent_anchor_frames"].astype(np.int32)
        tracks = data["tracks"].astype(np.float32)[anchors]
        visibility = data["visibility"].astype(bool)[anchors]
    rows = json.loads((case_dir / "metrics.json").read_text())
    method = f"qk_head{head:02d}"
    by_block = {
        block: [
            row for row in rows
            if row.get("method") == method
            and int(row.get("layer", -1)) == block
            and int(row.get("step_index", -1)) == step
        ]
        for block in range(30)
    }
    block_payloads = []
    with np.load(case_dir / "predicted_tracks.npz", allow_pickle=False) as data:
        for block in range(30):
            key = f"qk_head{head:02d}_layer{block:02d}_step{step:03d}_predictions"
            if key not in data:
                raise ValueError(f"missing array: {key}")
            block_payloads.append({
                "block": block,
                "predictions": finite_list(data[key].astype(np.float32)),
                "metrics": {
                    "objects": aggregate_metrics(by_block[block], "object"),
                    "background": aggregate_metrics(by_block[block], "background"),
                },
            })
    return json.dumps({
        "model": model,
        "model_label": MODELS[model]["label"],
        "case": case,
        "step": step,
        "head": head,
        "color": MODELS[model]["color"],
        "anchors": anchors.tolist(),
        "gt": finite_list(tracks),
        "visibility": visibility.tolist(),
        "tracks": block_payloads,
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


PORTAL_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DiffTrack validation atlas</title><style>
:root{--ink:#17211e;--paper:#f3efe4;--line:#cfc7b7;--red:#d45538;--green:#285e4b}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 84% 8%,#efc498 0,transparent 31rem),linear-gradient(145deg,#f5f1e7,#dfe9e2);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}main{max-width:1220px;margin:auto;padding:clamp(40px,8vw,110px) 24px}.eyebrow{font:700 12px monospace;letter-spacing:.22em;color:var(--red)}h1{font:400 clamp(52px,9vw,112px)/.9 Georgia,serif;margin:14px 0 24px;max-width:950px}.lead{max-width:800px;color:#59655f;font-size:17px;line-height:1.7}.stats{display:flex;gap:24px;flex-wrap:wrap;margin:28px 0 42px;font:700 12px monospace}.stats b{font-size:24px;color:var(--green);display:block}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}.card{display:flex;min-height:260px;flex-direction:column;justify-content:space-between;padding:28px;border:1px solid var(--line);border-radius:18px;background:#fffdf7d9;color:inherit;text-decoration:none;box-shadow:0 22px 60px #4e55491a;transition:.2s transform,.2s box-shadow}.card:hover{transform:translateY(-5px);box-shadow:0 28px 70px #4e55492c}.num{font:700 12px monospace;color:var(--red)}h2{font:400 clamp(29px,4vw,48px) Georgia,serif;margin:18px 0 10px}.card p{color:#68736d;line-height:1.55;max-width:490px}.go{font:700 12px monospace;letter-spacing:.12em;color:var(--green)}.foot{margin-top:25px}.foot a{color:#4f5d57;margin-right:18px;font:12px monospace}@media(max-width:760px){.grid{grid-template-columns:1fr}h1{font-size:53px}}
</style></head><body><main><div class="eyebrow">DIFFTRACK · 50-CASE VALIDATION</div><h1>Trajectory attention atlas</h1><p class="lead">One entrance for the complete GT teacher-forced, LoRA, and Wan2.2 Baseline analysis. Explore exact latent overlays or compare every Block × Head combination averaged across the full case set.</p><div class="stats"><span><b>3</b>models</span><span><b>50</b>cases each</span><span><b>30 × 24</b>block-head combinations</span><span><b>7</b>true latent anchors</span></div><section class="grid"><a class="card" href="/overlays?v=5"><div><span class="num">01 / VISUAL EVIDENCE</span><h2>Latent overlays</h2><p>All 30 blocks on one page. Seven exact latent anchors per block, ranked by per-case PCK for the selected model and head.</p></div><span class="go">OPEN OVERLAY CONTACT SHEET →</span></a><a class="card" href="/rankings?v=1"><div><span class="num">02 / AGGREGATE EVIDENCE</span><h2>Combination rankings</h2><p>Every model ranked independently over all cases. Inspect 30×24 heatmaps and complete 720-row tables for Object or Background metrics.</p></div><span class="go">OPEN 50-CASE RANKINGS →</span></a></section><div class="foot"><a href="/single">Single latent microscope</a><a href="/downloads/block_head_summary.csv">Download CSV</a><a href="/downloads/combination_rankings.json">Download JSON</a></div></main></body></html>'''


RANKINGS_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>50-case Block × Head rankings</title><style>
:root{--bg:#111815;--panel:#1a231f;--ink:#edf1e9;--muted:#9ba9a2;--line:#35423d;--gold:#f2bb4d}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 90% 0,#30483d 0,transparent 32rem),var(--bg);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}header{padding:30px clamp(16px,4vw,58px) 22px;border-bottom:1px solid var(--line)}nav a{color:var(--muted);font:12px monospace;margin-right:17px}.eyebrow{margin-top:24px;color:var(--gold);font:700 11px monospace;letter-spacing:.2em}h1{font:400 clamp(38px,6vw,76px)/1 Georgia,serif;margin:9px 0}.intro{max-width:970px;color:var(--muted);line-height:1.6}.controls{position:sticky;top:0;z-index:10;display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:12px;padding:13px clamp(16px,4vw,58px);background:#111815ed;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}label{display:block;color:var(--muted);font:700 10px monospace;margin-bottom:5px;letter-spacing:.1em}select{width:100%;padding:10px;border:1px solid #48564f;border-radius:8px;background:#202b26;color:var(--ink);font:600 13px inherit}main{padding:20px clamp(12px,3vw,42px) 70px}.model-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px}.model-card{border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:15px;cursor:pointer}.model-card.active{border-color:var(--gold);box-shadow:inset 0 0 0 1px var(--gold)}.model-card small{color:var(--muted);font:11px monospace}.model-card b{display:block;font:700 20px monospace;margin:8px 0 4px}.model-card span{color:var(--gold);font:12px monospace}.section{background:#171f1c;border:1px solid var(--line);border-radius:13px;margin-bottom:18px;overflow:hidden}.section-head{display:flex;justify-content:space-between;gap:15px;padding:13px 16px;border-bottom:1px solid var(--line)}h2{font:700 14px monospace;margin:0}.meta{color:var(--muted);font:11px monospace}.heat-wrap{overflow:auto;padding:12px}.heat{display:grid;grid-template-columns:48px repeat(24,minmax(30px,1fr));gap:3px;min-width:900px}.axis,.cell{display:flex;align-items:center;justify-content:center;height:27px;border-radius:4px;font:9px monospace}.axis{color:var(--muted)}.cell{border:0;cursor:pointer;color:white;text-shadow:0 1px 2px #000}.cell:hover{outline:2px solid white;z-index:2}.table-wrap{max-height:720px;overflow:auto}table{border-collapse:collapse;width:100%;font:12px monospace}th{position:sticky;top:0;background:#202a26;color:var(--gold);text-align:right;padding:10px;border-bottom:1px solid var(--line)}th:nth-child(-n+3),td:nth-child(-n+3){text-align:center}td{text-align:right;padding:8px 10px;border-bottom:1px solid #29342f}tr:hover{background:#24302b}.best{color:var(--gold);font-weight:700}.download{color:var(--gold);font:11px monospace;margin-left:12px}@media(max-width:800px){.model-cards{grid-template-columns:1fr}.controls{grid-template-columns:1fr}main{padding-inline:7px}.section-head{flex-direction:column}}
</style></head><body><header><nav><a href="/">← Portal</a><a href="/overlays">Latent overlays</a></nav><div class="eyebrow">ALL CASES · MACRO AVERAGE</div><h1>Block × Head rankings</h1><p class="intro">Each case contributes equally after comparison-weighted merging of its regions. Models are ranked independently; object-valid case count is retained explicitly when a generated case has no valid object points.</p></header><section class="controls"><div><label>MODEL</label><select id="model"></select></div><div><label>SCOPE</label><select id="scope"><option value="object">Object</option><option value="background">Background</option></select></div><div><label>RANKING METRIC</label><select id="metric"><option value="macro_pck32">Macro PCK@32</option><option value="macro_pck16">Macro PCK@16</option><option value="macro_pck8">Macro PCK@8</option><option value="macro_mean_error_px">Macro mean error</option></select></div></section><main><div class="model-cards" id="cards"></div><section class="section"><div class="section-head"><h2>30 × 24 PERFORMANCE MAP</h2><span class="meta" id="heatMeta"></span></div><div class="heat-wrap"><div class="heat" id="heat"></div></div></section><section class="section"><div class="section-head"><h2>COMPLETE COMBINATION ORDER</h2><span><span class="meta" id="tableMeta"></span><a class="download" href="/downloads/block_head_summary.csv">CSV</a><a class="download" href="/downloads/combination_rankings.json">JSON</a></span></div><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Block</th><th>Head</th><th>Valid cases</th><th>Macro PCK@8</th><th>Macro PCK@16</th><th>Macro PCK@32</th><th>Macro error</th><th>Pooled PCK@32</th></tr></thead><tbody id="rows"></tbody></table></div></section></main><script>
const $=s=>document.querySelector(s);let payload;const pad=n=>String(n).padStart(2,'0');const labels={macro_pck8:'Macro PCK@8',macro_pck16:'Macro PCK@16',macro_pck32:'Macro PCK@32',macro_mean_error_px:'Macro error'};function model(){return payload.models[$('#model').value]}function descending(){return $('#metric').value!=='macro_mean_error_px'}function ordered(){const key=$('#metric').value,rows=[...model().scopes[$('#scope').value]];return rows.sort((a,b)=>(descending()?b[key]-a[key]:a[key]-b[key])||a.macro_mean_error_px-b.macro_mean_error_px||a.block-b.block||a.head-b.head)}function fmt(v,key){return key.includes('error')?`${v.toFixed(2)}px`:`${v.toFixed(2)}%`}
async function init(){payload=await fetch('/api/rankings',{cache:'no-store'}).then(r=>r.json());Object.entries(payload.models).forEach(([key,m])=>$('#model').add(new Option(`${m.label} (${m.total_cases}/50)`,key)));render()}
function renderCards(){const box=$('#cards');box.innerHTML='';Object.entries(payload.models).forEach(([key,m])=>{const best=[...m.scopes.object].sort((a,b)=>b.macro_pck32-a.macro_pck32||a.macro_mean_error_px-b.macro_mean_error_px)[0],card=document.createElement('div');card.className='model-card'+(key===$('#model').value?' active':'');card.innerHTML=`<small>${m.label} · ${m.total_cases}/50 inference cases</small><b>B${pad(best.block)} × H${pad(best.head)}</b><span>best macro Object PCK@32 ${best.macro_pck32.toFixed(2)}%</span>`;card.onclick=()=>{$('#model').value=key;render()};box.append(card)})}
function renderHeat(rows){const key=$('#metric').value,values=rows.map(r=>r[key]),lo=Math.min(...values),hi=Math.max(...values),map=new Map(rows.map(r=>[`${r.block}-${r.head}`,r]));const heat=$('#heat');heat.innerHTML='<span class="axis">B\\H</span>'+[...Array(24)].map((_,h)=>`<span class="axis">H${pad(h)}</span>`).join('');for(let b=0;b<30;b++){heat.insertAdjacentHTML('beforeend',`<span class="axis">B${pad(b)}</span>`);for(let h=0;h<24;h++){const r=map.get(`${b}-${h}`),raw=(r[key]-lo)/(hi-lo||1),quality=descending()?raw:1-raw,hue=8+quality*125,light=31+quality*13,cell=document.createElement('button');cell.className='cell';cell.style.background=`hsl(${hue} 62% ${light}%)`;cell.textContent=fmt(r[key],key).replace('%','').replace('px','');cell.title=`B${pad(b)} H${pad(h)} · ${labels[key]} ${fmt(r[key],key)} · valid ${r.valid_cases}/${r.total_cases}`;cell.onclick=()=>document.querySelector(`[data-combo="${b}-${h}"]`).scrollIntoView({behavior:'smooth',block:'center'});heat.append(cell)}}$('#heatMeta').textContent=`${labels[key]} · ${fmt(lo,key)} to ${fmt(hi,key)} · green is better`}
function renderTable(rows){const key=$('#metric').value,tbody=$('#rows');tbody.innerHTML=rows.map((r,i)=>`<tr data-combo="${r.block}-${r.head}"><td class="${i<3?'best':''}">${i+1}</td><td>B${pad(r.block)}</td><td>H${pad(r.head)}</td><td>${r.valid_cases}/${r.total_cases}</td><td>${r.macro_pck8.toFixed(2)}%</td><td>${r.macro_pck16.toFixed(2)}%</td><td>${r.macro_pck32.toFixed(2)}%</td><td>${r.macro_mean_error_px.toFixed(2)}px</td><td>${r.pooled_pck32.toFixed(2)}%</td></tr>`).join('');$('#tableMeta').textContent=`720 combinations · ${model().label} · ${$('#scope').value} · ${labels[key]} ${descending()?'↓':'↑'}`}
function render(){renderCards();const rows=ordered();renderHeat(rows);renderTable(rows)}['model','scope','metric'].forEach(id=>$('#'+id).onchange=render);init();
</script></body></html>'''


EXPERIMENTS_PORTAL_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DiffTrack validation atlas</title><style>
:root{--ink:#17211e;--paper:#f3efe4;--line:#cfc7b7;--red:#d45538;--green:#285e4b}*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 84% 8%,#efc498 0,transparent 31rem),linear-gradient(145deg,#f5f1e7,#dfe9e2);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}main{max-width:1280px;margin:auto;padding:clamp(40px,7vw,92px) 24px}.eyebrow{font:700 12px monospace;letter-spacing:.22em;color:var(--red)}h1{font:400 clamp(48px,8vw,102px)/.9 Georgia,serif;margin:14px 0 24px;max-width:1050px}.lead{max-width:900px;color:#59655f;font-size:17px;line-height:1.7}.stats{display:flex;gap:24px;flex-wrap:wrap;margin:28px 0 38px;font:700 12px monospace}.stats b{font-size:24px;color:var(--green);display:block}.group-title{display:flex;align-items:center;gap:12px;margin:32px 0 13px;font:700 12px monospace;letter-spacing:.15em}.group-title:after{content:"";height:1px;background:var(--line);flex:1}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{display:flex;min-height:225px;flex-direction:column;justify-content:space-between;padding:25px;border:1px solid var(--line);border-radius:18px;background:#fffdf7d9;color:inherit;text-decoration:none;box-shadow:0 18px 48px #4e554916;transition:.2s transform,.2s box-shadow}.card:hover{transform:translateY(-5px);box-shadow:0 25px 65px #4e55492b}.card.new{border-color:#7ca08e;background:linear-gradient(145deg,#f9fff9,#eef4e9)}.num{font:700 11px monospace;color:var(--red)}.new .num{color:var(--green)}h2{font:400 clamp(27px,3vw,42px) Georgia,serif;margin:15px 0 8px}.card p{color:#68736d;line-height:1.55;max-width:520px}.go{font:700 11px monospace;letter-spacing:.11em;color:var(--green)}.foot{margin-top:25px}.foot a{color:#4f5d57;margin-right:18px;font:12px monospace}@media(max-width:760px){.grid{grid-template-columns:1fr}h1{font-size:53px}}
</style></head><body><main><div class="eyebrow">DIFFTRACK · 50-CASE VALIDATION</div><h1>Trajectory attention atlas</h1><p class="lead">Explore two aligned experiments for GT teacher-forced, LoRA, and Wan2.2 Baseline. The fixed-step atlas is the original view; the full atlas expands every result across all 40 diffusion steps.</p><div class="stats"><span><b>3</b>models</span><span><b>50</b>cases each</span><span><b>40 × 30 × 24</b>step-block-head grid</span><span><b>4.32M</b>saved combinations</span></div><div class="group-title">FULL 40-STEP EXPERIMENT</div><section class="grid"><a class="card new" href="/all-steps/overlays?v=1"><div><span class="num">01 / STEP-AWARE VISUAL EVIDENCE</span><h2>All-step overlays</h2><p>Select any S000-S039, case, and head. Compare all 30 blocks and seven exact latent anchors on one page.</p></div><span class="go">OPEN ALL-STEP OVERLAYS</span></a><a class="card new" href="/all-steps/rankings?v=1"><div><span class="num">02 / COMPLETE AGGREGATE EVIDENCE</span><h2>Step × Block × Head rankings</h2><p>Navigate the 40-step profile, inspect a 30×24 heatmap at any step, and rank all 720 block-head pairs.</p></div><span class="go">OPEN ALL-STEP RANKINGS</span></a></section><div class="group-title">ORIGINAL FIXED-STEP EXPERIMENT</div><section class="grid"><a class="card" href="/overlays?v=5"><div><span class="num">03 / FIXED-STEP VISUAL EVIDENCE</span><h2>Latent overlays</h2><p>Original fixed steps: GT S029, LoRA S039, and Baseline S039. All 30 blocks on one page.</p></div><span class="go">OPEN FIXED-STEP OVERLAYS</span></a><a class="card" href="/rankings?v=1"><div><span class="num">04 / FIXED-STEP AGGREGATE EVIDENCE</span><h2>Block × Head rankings</h2><p>Original 30×24 heatmaps and complete 720-row tables for Object or Background metrics.</p></div><span class="go">OPEN FIXED-STEP RANKINGS</span></a></section><div class="foot"><a href="/downloads/all-step-summary.csv">Download all-step CSV</a><a href="/downloads/all-step-results.md">Results</a><a href="/downloads/all-step-validation.json">Validation</a></div></main></body></html>'''


ALL_STEPS_RANKINGS_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All-step Block × Head rankings</title><style>
:root{--bg:#101714;--panel:#18221e;--ink:#edf2ea;--muted:#96a69f;--line:#33423c;--gold:#f0bc52;--mint:#76d2a5}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 91% 0,#315242 0,transparent 34rem),var(--bg);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}header{padding:28px clamp(16px,4vw,58px) 20px;border-bottom:1px solid var(--line)}nav a{color:var(--muted);font:12px monospace;margin-right:17px}.eyebrow{margin-top:22px;color:var(--mint);font:700 11px monospace;letter-spacing:.2em}h1{font:400 clamp(38px,6vw,72px)/1 Georgia,serif;margin:9px 0}.intro{max-width:980px;color:var(--muted);line-height:1.6}.controls{position:sticky;top:0;z-index:10;display:grid;grid-template-columns:1.25fr .75fr 1fr 1.35fr;gap:12px;padding:13px clamp(16px,4vw,58px);background:#101714ed;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}label{display:block;color:var(--muted);font:700 10px monospace;margin-bottom:5px;letter-spacing:.1em}select,input{width:100%;padding:9px;border:1px solid #48584f;border-radius:8px;background:#202c27;color:var(--ink);font:600 13px inherit}.step-field{display:grid;grid-template-columns:1fr 48px;gap:8px;align-items:center}.step-field b{font:700 13px monospace;color:var(--mint)}main{padding:18px clamp(12px,3vw,42px) 70px}.model-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:17px}.model-card{border:1px solid var(--line);border-radius:12px;background:var(--panel);padding:14px;cursor:pointer}.model-card.active{border-color:var(--mint);box-shadow:inset 0 0 0 1px var(--mint)}.model-card small{color:var(--muted);font:11px monospace}.model-card b{display:block;font:700 18px monospace;margin:8px 0 4px}.model-card span{color:var(--mint);font:11px monospace}.section{background:#161f1b;border:1px solid var(--line);border-radius:13px;margin-bottom:17px;overflow:hidden}.section-head{display:flex;justify-content:space-between;gap:15px;padding:12px 15px;border-bottom:1px solid var(--line)}h2{font:700 13px monospace;margin:0}.meta{color:var(--muted);font:11px monospace}.profile{display:grid;grid-template-columns:repeat(40,minmax(24px,1fr));gap:3px;min-width:1040px;padding:12px}.step-cell{height:58px;border:0;border-radius:5px;color:white;cursor:pointer;font:9px monospace;display:flex;flex-direction:column;justify-content:flex-end;padding:4px;text-shadow:0 1px 2px #000}.step-cell.active{outline:2px solid white;outline-offset:1px}.scroll{overflow:auto}.heat{display:grid;grid-template-columns:48px repeat(24,minmax(30px,1fr));gap:3px;min-width:900px;padding:12px}.axis,.cell{display:flex;align-items:center;justify-content:center;height:27px;border-radius:4px;font:9px monospace}.axis{color:var(--muted)}.cell{border:0;cursor:pointer;color:white;text-shadow:0 1px 2px #000}.cell:hover{outline:2px solid white}.table-wrap{max-height:700px;overflow:auto}table{border-collapse:collapse;width:100%;font:12px monospace}th{position:sticky;top:0;background:#202b26;color:var(--gold);text-align:right;padding:9px;border-bottom:1px solid var(--line)}td{text-align:right;padding:8px 10px;border-bottom:1px solid #29352f}th:nth-child(-n+4),td:nth-child(-n+4){text-align:center}tr:hover{background:#24312b}.best{color:var(--gold);font-weight:700}.download{color:var(--mint);font:11px monospace;margin-left:12px}.pager{display:flex;align-items:center;gap:8px}.pager button{border:1px solid #48584f;border-radius:6px;background:#202c27;color:var(--ink);padding:6px 10px;cursor:pointer}.pager button:disabled{opacity:.35;cursor:default}@media(max-width:820px){.model-cards{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}.section-head{flex-direction:column}}
</style></head><body><header><nav><a href="/">Portal</a><a href="/all-steps/overlays">All-step overlays</a><a href="/rankings">Fixed-step rankings</a></nav><div class="eyebrow">ALL 40 STEPS · ALL 30 BLOCKS · ALL 24 HEADS</div><h1>Step × Block × Head rankings</h1><p class="intro">The first global table combines GT, LoRA, and Baseline with equal model weight. The second table ranks the selected model. Both cover all 28,800 step-block-head combinations.</p></header><section class="controls"><div><label>MODEL</label><select id="model"></select></div><div><label>SCOPE</label><select id="scope"><option value="objects">Object</option><option value="background">Background</option></select></div><div><label>RANKING METRIC</label><select id="metric"><option value="macro_pck32">Macro PCK@32</option><option value="macro_pck16">Macro PCK@16</option><option value="macro_pck8">Macro PCK@8</option><option value="macro_mean_error_px">Macro mean error</option></select></div><div><label>DIFFUSION STEP</label><div class="step-field"><input id="step" type="range" min="0" max="39" value="39"><b id="stepValue">S039</b></div></div></section><main><div class="model-cards" id="cards"></div><section class="section"><div class="section-head"><h2>40-STEP BEST-COMBINATION PROFILE</h2><span class="meta" id="profileMeta"></span></div><div class="scroll"><div class="profile" id="profile"></div></div></section><section class="section"><div class="section-head"><h2>THREE-MODEL COMBINED GLOBAL RANKING · EQUAL MODEL WEIGHT</h2><span class="pager"><a class="download" href="/downloads/three-model-combined.csv">CSV</a><button id="combinedPrev">Previous</button><span class="meta" id="combinedMeta"></span><button id="combinedNext">Next</button></span></div><div class="table-wrap"><table><thead><tr><th>Combined rank</th><th>Step</th><th>Block</th><th>Head</th><th>Valid</th><th>Mean PCK@8</th><th>Mean PCK@16</th><th>Mean PCK@32</th><th>Mean error</th><th>Worst model</th><th>GT</th><th>LoRA</th><th>Baseline</th></tr></thead><tbody id="combinedRows"></tbody></table></div></section><section class="section"><div class="section-head"><h2>SELECTED MODEL · CROSS-STEP GLOBAL RANKING</h2><span class="pager"><button id="globalPrev">Previous</button><span class="meta" id="globalMeta"></span><button id="globalNext">Next</button></span></div><div class="table-wrap"><table><thead><tr><th>Global rank</th><th>Step</th><th>Block</th><th>Head</th><th>Valid</th><th>PCK@8</th><th>PCK@16</th><th>PCK@32</th><th>Error</th><th>Pooled PCK@32</th></tr></thead><tbody id="globalRows"></tbody></table></div></section><section class="section"><div class="section-head"><h2>SELECTED STEP · 30 × 24 PERFORMANCE MAP</h2><span class="meta" id="heatMeta"></span></div><div class="scroll"><div class="heat" id="heat"></div></div></section><section class="section"><div class="section-head"><h2>SELECTED STEP · 720 BLOCK-HEAD COMBINATIONS</h2><span><span class="meta" id="tableMeta"></span><a class="download" href="/downloads/all-step-summary.csv">CSV</a></span></div><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Step</th><th>Block</th><th>Head</th><th>Valid</th><th>PCK@8</th><th>PCK@16</th><th>PCK@32</th><th>Error</th><th>Pooled PCK@32</th></tr></thead><tbody id="rows"></tbody></table></div></section></main><script>
const $=s=>document.querySelector(s);let catalog,rows=[],profile=[],globalData={rows:[],page:0,pages:1,total:0},combinedData={rows:[],page:0,pages:1,total:0};const pad=(n,w=2)=>String(n).padStart(w,'0'),labels={macro_pck8:'Macro PCK@8',macro_pck16:'Macro PCK@16',macro_pck32:'Macro PCK@32',macro_mean_error_px:'Macro error'};function metric(){return $('#metric').value}function descending(){return metric()!=='macro_mean_error_px'}function fmt(v){return metric().includes('error')?`${v.toFixed(2)}px`:`${v.toFixed(2)}%`}function currentModel(){return catalog.models.find(m=>m.key===$('#model').value)}
async function init(){catalog=await fetch('/api/all-steps/catalog',{cache:'no-store'}).then(r=>r.json());catalog.models.forEach(m=>$('#model').add(new Option(`${m.label} (${m.cases.length}/50)`,m.key)));$('#step').value=currentModel().best.step;await load()}
async function load(){const m=$('#model').value,s=$('#scope').value,t=$('#step').value,k=metric();$('#stepValue').textContent='S'+pad(t,3);const [rank,prof,global,combined]=await Promise.all([fetch(`/api/all-steps/rankings?model=${m}&scope=${s}&step=${t}`,{cache:'no-store'}).then(r=>r.json()),fetch(`/api/all-steps/profile?model=${m}&scope=${s}&metric=${k}`,{cache:'no-store'}).then(r=>r.json()),fetch(`/api/all-steps/global-rankings?model=${m}&scope=${s}&metric=${k}&page=${globalData.page}`,{cache:'no-store'}).then(r=>r.json()),fetch(`/api/all-steps/combined-rankings?scope=${s}&metric=${k}&page=${combinedData.page}`,{cache:'no-store'}).then(r=>r.json())]);rows=rank.rows;profile=prof.rows;globalData=global;combinedData=combined;render()}
function renderCards(){const box=$('#cards');box.innerHTML='';catalog.models.forEach(m=>{const b=m.best,d=document.createElement('div');d.className='model-card'+(m.key===$('#model').value?' active':'');d.innerHTML=`<small>${m.label} · global Object best</small><b>S${pad(b.step,3)} × L${pad(b.block)} × H${pad(b.head)}</b><span>Macro PCK@32 ${b.pck32.toFixed(2)}% · ${b.error.toFixed(2)}px</span>`;d.onclick=()=>{$('#model').value=m.key;$('#step').value=b.step;load()};box.append(d)})}
function color(v,lo,hi){let q=(v-lo)/(hi-lo||1);if(!descending())q=1-q;return `hsl(${8+q*125} 62% ${31+q*13}%)`}
function renderProfile(){const vals=profile.map(r=>r[metric()]),lo=Math.min(...vals),hi=Math.max(...vals),box=$('#profile');box.innerHTML='';profile.forEach(r=>{const b=document.createElement('button');b.className='step-cell'+(r.step==$('#step').value?' active':'');b.style.background=color(r[metric()],lo,hi);b.innerHTML=`<b>S${pad(r.step,3)}</b><span>${fmt(r[metric()])}</span>`;b.title=`S${pad(r.step,3)} · L${pad(r.block)} H${pad(r.head)} · ${labels[metric()]} ${fmt(r[metric()])}`;b.onclick=()=>{$('#step').value=r.step;load()};box.append(b)});$('#profileMeta').textContent=`best of 720 at each step · ${labels[metric()]}`}
function ordered(){return [...rows].sort((a,b)=>(descending()?b[metric()]-a[metric()]:a[metric()]-b[metric()])||a.macro_mean_error_px-b.macro_mean_error_px)}
function renderHeat(){const vals=rows.map(r=>r[metric()]),lo=Math.min(...vals),hi=Math.max(...vals),map=new Map(rows.map(r=>[`${r.block}-${r.head}`,r])),box=$('#heat');box.innerHTML='<span class="axis">L\\H</span>'+[...Array(24)].map((_,h)=>`<span class="axis">H${pad(h)}</span>`).join('');for(let l=0;l<30;l++){box.insertAdjacentHTML('beforeend',`<span class="axis">L${pad(l)}</span>`);for(let h=0;h<24;h++){const r=map.get(`${l}-${h}`),b=document.createElement('button');b.className='cell';b.style.background=color(r[metric()],lo,hi);b.textContent=fmt(r[metric()]).replace('%','').replace('px','');b.title=`S${pad(r.step,3)} L${pad(l)} H${pad(h)} · ${labels[metric()]} ${fmt(r[metric()])}`;b.onclick=()=>document.querySelector(`[data-combo="${l}-${h}"]`).scrollIntoView({behavior:'smooth',block:'center'});box.append(b)}}$('#heatMeta').textContent=`S${pad($('#step').value,3)} · ${fmt(lo)} to ${fmt(hi)} · green is better`}
function renderTable(){const data=ordered();$('#rows').innerHTML=data.map((r,i)=>`<tr data-combo="${r.block}-${r.head}"><td class="${i<3?'best':''}">${i+1}</td><td>S${pad(r.step,3)}</td><td>L${pad(r.block)}</td><td>H${pad(r.head)}</td><td>${r.valid_cases}/50</td><td>${r.macro_pck8.toFixed(2)}%</td><td>${r.macro_pck16.toFixed(2)}%</td><td>${r.macro_pck32.toFixed(2)}%</td><td>${r.macro_mean_error_px.toFixed(2)}px</td><td>${r.pooled_pck32.toFixed(2)}%</td></tr>`).join('');$('#tableMeta').textContent=`S${pad($('#step').value,3)} · ${currentModel().label} · ${$('#scope').value}`}
function renderGlobal(){const data=globalData.rows;$('#globalRows').innerHTML=data.map(r=>`<tr><td class="${r.global_rank<=3?'best':''}">${r.global_rank}</td><td>S${pad(r.step,3)}</td><td>L${pad(r.block)}</td><td>H${pad(r.head)}</td><td>${r.valid_cases}/50</td><td>${r.macro_pck8.toFixed(2)}%</td><td>${r.macro_pck16.toFixed(2)}%</td><td>${r.macro_pck32.toFixed(2)}%</td><td>${r.macro_mean_error_px.toFixed(2)}px</td><td>${r.pooled_pck32.toFixed(2)}%</td></tr>`).join('');$('#globalMeta').textContent=`page ${globalData.page+1}/${globalData.pages} · ranks ${data[0]?.global_rank??0}-${data.at(-1)?.global_rank??0} of ${globalData.total}`;$('#globalPrev').disabled=globalData.page<=0;$('#globalNext').disabled=globalData.page>=globalData.pages-1}
function renderCombined(){const data=combinedData.rows;$('#combinedRows').innerHTML=data.map(r=>`<tr><td class="${r.combined_rank<=3?'best':''}">${r.combined_rank}</td><td>S${pad(r.step,3)}</td><td>L${pad(r.block)}</td><td>H${pad(r.head)}</td><td>${r.valid_cases}/${r.total_cases}</td><td>${r.macro_pck8.toFixed(2)}%</td><td>${r.macro_pck16.toFixed(2)}%</td><td>${r.macro_pck32.toFixed(2)}%</td><td>${r.macro_mean_error_px.toFixed(2)}px</td><td>${r.worst_model_macro_pck32.toFixed(2)}%</td><td>${r.gt_macro_pck32.toFixed(2)}%</td><td>${r.lora_macro_pck32.toFixed(2)}%</td><td>${r.baseline_macro_pck32.toFixed(2)}%</td></tr>`).join('');$('#combinedMeta').textContent=`page ${combinedData.page+1}/${combinedData.pages} · ranks ${data[0]?.combined_rank??0}-${data.at(-1)?.combined_rank??0} of ${combinedData.total}`;$('#combinedPrev').disabled=combinedData.page<=0;$('#combinedNext').disabled=combinedData.page>=combinedData.pages-1}
function render(){renderCards();renderProfile();renderCombined();renderGlobal();renderHeat();renderTable()}function resetGlobal(){globalData.page=0;combinedData.page=0}$('#model').onchange=()=>{globalData.page=0;$('#step').value=currentModel().best.step;load()};['scope','metric'].forEach(id=>$('#'+id).onchange=()=>{resetGlobal();load()});$('#step').oninput=()=>{$('#stepValue').textContent='S'+pad($('#step').value,3);clearTimeout(window.stepDelay);window.stepDelay=setTimeout(load,140)};$('#globalPrev').onclick=()=>{if(globalData.page>0){globalData.page--;load()}};$('#globalNext').onclick=()=>{if(globalData.page<globalData.pages-1){globalData.page++;load()}};$('#combinedPrev').onclick=()=>{if(combinedData.page>0){combinedData.page--;load()}};$('#combinedNext').onclick=()=>{if(combinedData.page<combinedData.pages-1){combinedData.page++;load()}};init();
</script></body></html>'''


ALL_STEPS_OVERLAYS_PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>All-step latent overlays</title><style>
:root{--bg:#ece8dc;--paper:#f8f5eb;--ink:#15201c;--muted:#66726c;--line:#cfc8b8;--accent:#d45436}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 88% 2%,#efc69d 0,transparent 30rem),var(--bg);color:var(--ink);font-family:"Avenir Next","Trebuchet MS",sans-serif}header{padding:30px clamp(16px,4vw,62px) 20px;border-bottom:1px solid var(--line)}nav a{color:var(--muted);font:12px monospace;margin-right:17px}.eyebrow{margin-top:20px;font:700 11px monospace;letter-spacing:.2em;color:var(--accent)}h1{margin:8px 0 7px;font:400 clamp(32px,5vw,62px)/1 Georgia,serif}.intro{max-width:1050px;color:var(--muted);line-height:1.55}.toolbar{position:sticky;top:0;z-index:10;display:grid;grid-template-columns:1fr 1.4fr 1.15fr 1fr .75fr 1fr auto;gap:10px;align-items:end;padding:12px clamp(12px,3vw,48px);background:#ece8dcf2;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}label{display:block;margin-bottom:5px;color:var(--muted);font:700 10px monospace}select,input,button{width:100%;border:1px solid #bcb4a3;border-radius:8px;background:#fffdf7;color:var(--ink);padding:9px;font:600 13px inherit}button{width:auto;min-width:110px;cursor:pointer;background:var(--ink);color:white}.range{display:grid;grid-template-columns:1fr 45px;gap:7px;align-items:center}.range b{font:700 13px monospace;text-align:center}.status{padding:12px clamp(16px,4vw,62px) 0;color:var(--muted);font:12px monospace}main{padding:16px clamp(12px,3vw,40px) 60px}.block{margin-bottom:17px;background:#f8f5ebed;border:1px solid var(--line);border-radius:13px;overflow:hidden;box-shadow:0 10px 28px #5d574716}.block-head{display:flex;justify-content:space-between;align-items:center;padding:10px 13px;border-bottom:1px solid var(--line)}.block-head h2{margin:0;font:700 13px monospace}.rank{margin-right:9px;color:var(--accent)}.metrics{color:var(--muted);font:11px monospace}.latent-grid{display:grid;grid-template-columns:repeat(7,minmax(150px,1fr));gap:1px;background:#2d3531;overflow-x:auto}.latent{position:relative;min-width:150px;background:#080c0a}.latent canvas{display:block;width:100%;aspect-ratio:7/4}.tag{position:absolute;left:6px;top:6px;padding:4px 6px;border-radius:5px;background:#07100dca;color:white;font:700 10px monospace}.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:8px;color:var(--muted);font:12px monospace}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.gt{border:2px solid white;background:#222}.pred{background:var(--accent)}@media(max-width:1000px){.toolbar{grid-template-columns:1fr 1fr}.toolbar button{width:100%}main{padding-inline:7px}.latent-grid{grid-template-columns:repeat(7,150px)}}
</style></head><body><header><nav><a href="/">Portal</a><a href="/all-steps/rankings">All-step rankings</a><a href="/overlays">Fixed-step overlays</a></nav><div class="eyebrow">THREE-MODEL COMBINED GLOBAL RANK ORDER</div><h1>All-step latent overlay atlas</h1><div class="intro">Rows are ordered by the equal-weight GT + LoRA + Baseline global ranking across all 28,800 Step × Block × Head combinations. The selected model and case control the trajectories being drawn; they do not change the global order.</div><div class="legend"><span><i class="dot gt"></i>pseudo-GT</span><span><i class="dot pred"></i>selected model Q@K</span><span>row rank: three-model combined global rank</span><span>anchors: F0 · F4 · F8 · F12 · F16 · F20 · F24</span></div></header><section class="toolbar"><div><label>OVERLAY MODEL</label><select id="model"></select></div><div><label>CASE</label><select id="case"></select></div><div><label>STEP</label><div class="range"><input id="step" type="range" min="0" max="39" value="39"><b id="stepValue">S039</b></div></div><div><label>HEAD</label><div class="range"><input id="head" type="range" min="0" max="23" value="0"><b id="headValue">H00</b></div></div><div><label>SCOPE</label><select id="scope"><option value="objects">Object</option><option value="background">Background</option></select></div><div><label>COMBINED RANK METRIC</label><select id="sortMetric"><option value="macro_pck32">Macro PCK@32</option><option value="macro_pck16">Macro PCK@16</option><option value="macro_pck8">Macro PCK@8</option><option value="macro_mean_error_px">Macro mean error</option></select></div><button id="reload">Render</button></section><div class="status" id="status">Loading combined global ranking...</div><main id="blocks"></main><script>
const $=s=>document.querySelector(s),CW=280,CH=160,SX=CW/896,SY=CH/512;let catalog,renderToken=0;const pad=(n,w=2)=>String(n).padStart(w,'0'),valid=p=>p&&Number.isFinite(p[0])&&Number.isFinite(p[1]);function currentModel(){return catalog.models.find(m=>m.key===$('#model').value)}
async function init(){const [cat,best]=await Promise.all([fetch('/api/all-steps/catalog',{cache:'no-store'}).then(r=>r.json()),fetch('/api/all-steps/combined-rankings?scope=objects&metric=macro_pck32&page=0',{cache:'no-store'}).then(r=>r.json())]);catalog=cat;catalog.models.forEach(m=>$('#model').add(new Option(`${m.label} (${m.cases.length}/50)`,m.key)));fillCases();$('#step').value=best.rows[0].step;$('#head').value=best.rows[0].head;syncLabels();await renderAll()}
function fillCases(){const old=$('#case').value;$('#case').innerHTML='';currentModel().cases.forEach(c=>$('#case').add(new Option(c.replace(/^case_\d+_/,''),c)));if(currentModel().cases.includes(old))$('#case').value=old}function syncLabels(){$('#stepValue').textContent='S'+pad($('#step').value,3);$('#headValue').textContent='H'+pad($('#head').value)}function loadImage(src){return new Promise((ok,bad)=>{const im=new Image();im.onload=()=>ok(im);im.onerror=bad;im.src=src})}function line(ctx,points,color,width){const q=points.filter(valid);if(q.length<2)return;ctx.beginPath();ctx.moveTo(q[0][0]*SX,q[0][1]*SY);q.slice(1).forEach(p=>ctx.lineTo(p[0]*SX,p[1]*SY));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function draw(canvas,img,data,track,latent){const ctx=canvas.getContext('2d');ctx.drawImage(img,0,0,CW,CH);for(let p=0;p<track.predictions[0].length;p++){line(ctx,data.gt.slice(0,latent+1).map((f,i)=>data.visibility[i][p]?f[p]:null),'#ffffffb8',.8);line(ctx,track.predictions.slice(0,latent+1).map(f=>f[p]),data.color+'dd',1.35)}for(let p=0;p<track.predictions[latent].length;p++){const g=data.visibility[latent][p]?data.gt[latent][p]:null,q=track.predictions[latent][p];if(valid(g)&&valid(q)){ctx.beginPath();ctx.moveTo(g[0]*SX,g[1]*SY);ctx.lineTo(q[0]*SX,q[1]*SY);ctx.strokeStyle='#b0b8b288';ctx.lineWidth=.65;ctx.stroke()}if(valid(g)){ctx.beginPath();ctx.arc(g[0]*SX,g[1]*SY,3.2,0,Math.PI*2);ctx.fillStyle='#111';ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1.2;ctx.stroke()}if(valid(q)){ctx.beginPath();ctx.arc(q[0]*SX,q[1]*SY,2.6,0,Math.PI*2);ctx.fillStyle=data.color;ctx.fill();ctx.strokeStyle='#111';ctx.lineWidth=.7;ctx.stroke()}}}
function metricText(track,combo,key,scope){const caseMetric=track.metrics[scope],caseKey=key.replace('macro_',''),label=key.replace('macro_','').replace('pck','PCK@');const combined=key.includes('error')?`${combo[key].toFixed(2)}px`:`${combo[key].toFixed(2)}%`,caseValue=caseMetric?(caseKey.includes('error')?`${caseMetric[caseKey].toFixed(1)}px`:`${caseMetric[caseKey].toFixed(1)}%`):'n/a';return `3-model ${label} ${combined} · worst PCK@32 ${combo.worst_model_macro_pck32.toFixed(2)}% · selected case ${caseValue}`}
async function renderAll(){const mine=++renderToken,m=$('#model').value,cs=$('#case').value,s=Number($('#step').value),h=Number($('#head').value),scope=$('#scope').value,key=$('#sortMetric').value;$('#status').textContent=`Loading combined ranks + 30 overlays · S${pad(s,3)} · H${pad(h)}...`;$('#blocks').innerHTML='';try{const [frames,data,combined]=await Promise.all([Promise.all([...Array(7)].map((_,i)=>loadImage(`/api/frame?model=${m}&case=${encodeURIComponent(cs)}&latent=${i}`))),fetch(`/api/all-steps/tracks?model=${m}&case=${encodeURIComponent(cs)}&step=${s}&head=${h}`,{cache:'no-store'}).then(async r=>{if(!r.ok)throw new Error(await r.text());return r.json()}),fetch(`/api/all-steps/combined-selection?scope=${scope}&metric=${key}&step=${s}&head=${h}`,{cache:'no-store'}).then(async r=>{if(!r.ok)throw new Error(await r.text());return r.json()})]);if(mine!==renderToken)return;document.documentElement.style.setProperty('--accent',data.color);const comboByBlock=new Map(combined.rows.map(r=>[r.block,r])),ranked=data.tracks.map(track=>({track,combo:comboByBlock.get(track.block)})).sort((a,b)=>a.combo.combined_rank-b.combo.combined_rank),frag=document.createDocumentFragment();ranked.forEach(({track,combo})=>{const article=document.createElement('article');article.className='block';article.innerHTML=`<div class="block-head"><h2><span class="rank">COMBINED GLOBAL #${combo.combined_rank}</span>STEP ${pad(s,3)} · BLOCK ${pad(track.block)} · HEAD ${pad(h)}</h2><span class="metrics">${metricText(track,combo,key,scope)}</span></div><div class="latent-grid"></div>`;const grid=article.querySelector('.latent-grid');frames.forEach((img,i)=>{const cell=document.createElement('div'),canvas=document.createElement('canvas'),tag=document.createElement('span');cell.className='latent';canvas.width=CW;canvas.height=CH;tag.className='tag';tag.textContent=`L${i} / F${data.anchors[i]}`;cell.append(canvas,tag);grid.append(cell);draw(canvas,img,data,track,i)});frag.append(article)});$('#blocks').append(frag);$('#status').textContent=`Combined global order · ${scope} · ${key} · 30 blocks · ${data.model_label} overlay · S${pad(s,3)} H${pad(h)} · ${cs}`}catch(e){$('#status').textContent=`Render failed: ${e.message}`}}
$('#model').onchange=()=>{fillCases();renderAll()};$('#case').onchange=renderAll;['scope','sortMetric'].forEach(id=>$('#'+id).onchange=renderAll);['step','head'].forEach(id=>$('#'+id).oninput=()=>{syncLabels();clearTimeout(window[id+'Delay']);window[id+'Delay']=setTimeout(renderAll,180)});$('#reload').onclick=renderAll;init();
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
            if parsed.path == "/":
                self.send_bytes(EXPERIMENTS_PORTAL_PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/all-steps/overlays":
                self.send_bytes(ALL_STEPS_OVERLAYS_PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/all-steps/rankings":
                self.send_bytes(ALL_STEPS_RANKINGS_PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path in ("/overlays", "/all-blocks"):
                self.send_bytes(BLOCKS_PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/rankings":
                self.send_bytes(RANKINGS_PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/single":
                self.send_bytes(PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/api/catalog":
                self.send_bytes(catalog_payload(), "application/json")
            elif parsed.path == "/api/all-steps/catalog":
                self.send_bytes(allstep_catalog_payload(), "application/json")
            elif parsed.path == "/api/all-steps/rankings":
                self.send_bytes(allstep_rankings_payload(
                    query["model"][0], query["scope"][0], int(query["step"][0]),
                ), "application/json")
            elif parsed.path == "/api/all-steps/profile":
                self.send_bytes(allstep_profile_payload(
                    query["model"][0], query["scope"][0], query["metric"][0],
                ), "application/json")
            elif parsed.path == "/api/all-steps/global-rankings":
                self.send_bytes(allstep_global_rankings_payload(
                    query["model"][0], query["scope"][0],
                    query["metric"][0], int(query.get("page", ["0"])[0]),
                ), "application/json")
            elif parsed.path == "/api/all-steps/combined-rankings":
                self.send_bytes(combined_rankings_payload(
                    query["scope"][0], query["metric"][0],
                    int(query.get("page", ["0"])[0]),
                ), "application/json")
            elif parsed.path == "/api/all-steps/combined-selection":
                self.send_bytes(combined_selection_payload(
                    query["scope"][0], query["metric"][0],
                    int(query["step"][0]), int(query["head"][0]),
                ), "application/json")
            elif parsed.path == "/api/all-steps/tracks":
                self.send_bytes(allstep_tracks_payload(
                    query["model"][0], query["case"][0],
                    int(query["step"][0]), int(query["head"][0]),
                ), "application/json")
            elif parsed.path == "/api/rankings":
                self.send_bytes((ROOT / "combination_rankings.json").read_bytes(), "application/json")
            elif parsed.path == "/downloads/block_head_summary.csv":
                self.send_bytes((ROOT / "block_head_summary.csv").read_bytes(), "text/csv; charset=utf-8")
            elif parsed.path == "/downloads/combination_rankings.json":
                self.send_bytes((ROOT / "combination_rankings.json").read_bytes(), "application/json")
            elif parsed.path == "/downloads/all-step-summary.csv":
                self.send_bytes((ALL_STEPS_ROOT / "block_step_head_summary.csv").read_bytes(), "text/csv; charset=utf-8")
            elif parsed.path == "/downloads/all-step-results.md":
                self.send_bytes((ALL_STEPS_ROOT / "RESULTS.md").read_bytes(), "text/markdown; charset=utf-8")
            elif parsed.path == "/downloads/all-step-validation.json":
                self.send_bytes((ALL_STEPS_ROOT / "validation.json").read_bytes(), "application/json")
            elif parsed.path == "/downloads/three-model-combined.csv":
                self.send_bytes((ALL_STEPS_ROOT / "three_model_combined_summary.csv").read_bytes(), "text/csv; charset=utf-8")
            elif parsed.path == "/downloads/three-model-combined.md":
                self.send_bytes((ALL_STEPS_ROOT / "THREE_MODEL_COMBINED_RESULTS.md").read_bytes(), "text/markdown; charset=utf-8")
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
