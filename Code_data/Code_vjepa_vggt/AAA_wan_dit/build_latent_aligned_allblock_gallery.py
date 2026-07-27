#!/usr/bin/env python3
"""Build an interactive all-block gallery aligned to a jointly decoded latent."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from analyze_multiblock_ball_query_heads import (
    ROLE_LABELS,
    _feature_rows,
    _rank01,
    _role_scores,
)
from moving_query_attention import FEATURE_NAMES


CASE = "0613pybullet_sample_001460_w002"
MODEL = "wan_lora"
GRID = (13, 16, 28)
FIXED_QUERY_TIME = 2
FIXED_B_QUERY_TIME = 3
GALLERY_FEATURE_NAMES = FEATURE_NAMES + (
    "adjacent_frame_mass",
    "adjacent_trajectory_enrichment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _classify(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    scores = _role_scores(features)
    scores["T_long"] = scores["T"].copy()
    scores["T_adj"] = 0.7 * _rank01(
        features["adjacent_trajectory_enrichment"]
    ) + 0.3 * _rank01(features["adjacent_frame_mass"])
    scores["T"] = np.maximum(scores["T_long"], scores["T_adj"])
    roles = list(ROLE_LABELS)
    matrix = np.stack([scores[role] for role in roles], axis=1)
    order = np.argsort(matrix, axis=1)
    return {
        "scores": scores,
        "primary": np.asarray([roles[int(value)] for value in order[:, -1]]),
        "secondary": np.asarray([roles[int(value)] for value in order[:, -2]]),
        "margin": (
            np.take_along_axis(matrix, order[:, -1:], axis=1)[:, 0]
            - np.take_along_axis(matrix, order[:, -2:-1], axis=1)[:, 0]
        ),
    }


def _trajectory_tokens(query_coords: np.ndarray) -> list[np.ndarray]:
    output = []
    for time in range(GRID[0]):
        current = query_coords[query_coords[:, 0] == time]
        output.append(
            current[:, 0] * GRID[1] * GRID[2]
            + current[:, 1] * GRID[2]
            + current[:, 2]
        )
    return output


def _adjacent_features(
    attention: np.ndarray,
    *,
    query_time: int,
    trajectory_tokens: list[np.ndarray],
) -> dict[str, np.ndarray]:
    heads, frames, grid_h, grid_w = attention.shape
    token_count = frames * grid_h * grid_w
    adjacent_times = [
        time
        for time in (query_time - 1, query_time + 1)
        if 0 <= time < frames and len(trajectory_tokens[time])
    ]
    if not adjacent_times:
        zeros = np.zeros(heads, dtype=np.float64)
        return {
            "adjacent_frame_mass": zeros,
            "adjacent_trajectory_enrichment": zeros.copy(),
        }
    temporal = attention.sum(axis=(2, 3)).astype(np.float64)
    trajectory_ids = np.unique(
        np.concatenate([trajectory_tokens[time] for time in adjacent_times])
    )
    flat = attention.reshape(heads, token_count).astype(np.float64)
    trajectory_mass = flat[:, trajectory_ids].sum(1)
    return {
        "adjacent_frame_mass": temporal[:, adjacent_times].sum(1),
        "adjacent_trajectory_enrichment": trajectory_mass
        / (len(trajectory_ids) / token_count),
    }


def _moving_features(
    attention: np.ndarray,
    query_coords: np.ndarray,
    valid_query_times: np.ndarray,
) -> dict[str, np.ndarray]:
    trajectory_tokens = _trajectory_tokens(query_coords)
    rows = []
    for query_time in np.flatnonzero(valid_query_times):
        current_coords = query_coords[query_coords[:, 0] == query_time]
        features, _ = _feature_rows(
            attention[:, query_time],
            query_coords=current_coords,
            trajectory_tokens=trajectory_tokens,
        )
        features.update(
            _adjacent_features(
                attention[:, query_time],
                query_time=int(query_time),
                trajectory_tokens=trajectory_tokens,
            )
        )
        rows.append(features)
    if not rows:
        raise ValueError("moving track has no valid query times")
    return {
        name: np.stack([row[name] for row in rows]).mean(axis=0)
        for name in GALLERY_FEATURE_NAMES
    }


def _block_data(
    attention: np.ndarray,
    query_coords: tuple[np.ndarray, np.ndarray],
    valid_query_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if attention.shape != (2, 24, 13, 13, 16, 28):
        raise ValueError(f"unexpected attention shape: {attention.shape}")
    if valid_query_times.shape != (2, 13):
        raise ValueError(f"unexpected validity shape: {valid_query_times.shape}")
    fixed_coords = query_coords[0][
        query_coords[0][:, 0] == FIXED_QUERY_TIME
    ]
    fixed_features, _ = _feature_rows(
        attention[0, :, FIXED_QUERY_TIME],
        query_coords=fixed_coords,
        trajectory_tokens=_trajectory_tokens(query_coords[0]),
    )
    fixed_features.update(
        _adjacent_features(
            attention[0, :, FIXED_QUERY_TIME],
            query_time=FIXED_QUERY_TIME,
            trajectory_tokens=_trajectory_tokens(query_coords[0]),
        )
    )
    moving_features = [
        _moving_features(
            attention[track_index],
            query_coords[track_index],
            valid_query_times[track_index],
        )
        for track_index in range(2)
    ]
    fixed_maps_a = attention[0, :, FIXED_QUERY_TIME].astype(np.float32)
    fixed_maps_b = attention[1, :, FIXED_B_QUERY_TIME].astype(np.float32)
    moving_maps = np.stack(
        [
            np.stack(
                [attention[track, :, time, time] for time in range(GRID[0])],
                axis=1,
            )
            for track in range(2)
        ],
        axis=0,
    ).astype(np.float32)
    temporal_mass = attention.astype(np.float32).sum(axis=(4, 5))
    return (
        fixed_maps_a,
        fixed_maps_b,
        moving_maps,
        temporal_mass,
        [
            {"features": fixed_features, **_classify(fixed_features)},
            {
                "features": moving_features[0],
                **_classify(moving_features[0]),
            },
            {
                "features": moving_features[1],
                **_classify(moving_features[1]),
            },
        ],
    )


def _write_float32(path: Path, array: np.ndarray) -> None:
    np.ascontiguousarray(array, dtype="<f4").tofile(path)


def _render_full_matrix_images(
    key_mass: np.ndarray, *, block: int, output_dir: Path
) -> None:
    if key_mass.shape != (24, 512, 512):
        raise ValueError(f"unexpected full matrix shape: {key_mass.shape}")
    block_dir = output_dir / f"block{block:02d}"
    block_dir.mkdir(parents=True, exist_ok=True)
    boundaries = [
        int(round(frame * (16 * 28) * 512 / (13 * 16 * 28) - 0.5))
        for frame in range(1, 13)
    ]
    for head in range(24):
        matrix = key_mass[head].astype(np.float32)
        positive = matrix[matrix > 0]
        epsilon = float(positive.min()) * 0.5 if positive.size else 1.0e-12
        display = np.log10(np.maximum(matrix, epsilon))
        low, high = np.percentile(display[np.isfinite(display)], [1.0, 99.8])
        if high <= low:
            high = low + 1.0
        normalized = np.asarray(
            np.clip((display - low) / (high - low), 0.0, 1.0) * 255.0,
            dtype=np.uint8,
        )
        image = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
        for boundary in boundaries:
            cv2.line(image, (boundary, 0), (boundary, 511), (255, 255, 255), 1)
            cv2.line(image, (0, boundary), (511, boundary), (255, 255, 255), 1)
        path = block_dir / f"head{head:02d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write {path}")


def _role_record(
    *,
    block: int,
    head: int,
    protocol: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    primary = str(result["primary"][head])
    secondary = str(result["secondary"][head])
    scores = {
        role: float(result["scores"][role][head])
        for role in (*ROLE_LABELS, "T_adj", "T_long")
    }
    display_primary = primary
    if (
        primary == "S"
        and scores["T_adj"] >= 0.8
        and scores["S"] - scores["T_adj"] <= 0.1
    ):
        display_primary = "S+T_adj"
    elif (
        primary == "T"
        and scores["T_adj"] >= scores["T_long"]
        and scores["S"] >= 0.8
        and scores["T_adj"] - scores["S"] <= 0.1
    ):
        display_primary = "T_adj+S"
    return {
        "block": block,
        "head": head,
        "protocol": protocol,
        "primary": primary,
        "display_primary": display_primary,
        "primary_name": ROLE_LABELS[primary],
        "secondary": secondary,
        "secondary_name": ROLE_LABELS[secondary],
        "margin": float(result["margin"][head]),
        "features": {
            name: float(result["features"][name][head])
            for name in GALLERY_FEATURE_NAMES
        },
        "scores": scores,
    }


def _all_token_s_records(
    *,
    block: int,
    time_matrix: np.ndarray,
    exact_self_mass: np.ndarray,
    same_frame_win_rate: np.ndarray,
) -> list[dict[str, Any]]:
    if time_matrix.shape != (24, 13, 13):
        raise ValueError(f"unexpected all-token time matrix: {time_matrix.shape}")
    if exact_self_mass.shape != (24, 13):
        raise ValueError(f"unexpected exact-self shape: {exact_self_mass.shape}")
    if same_frame_win_rate.shape != (24, 13):
        raise ValueError(
            f"unexpected same-frame win-rate shape: {same_frame_win_rate.shape}"
        )
    records = []
    for head in range(24):
        matrix = time_matrix[head].astype(np.float64)
        diagonal = np.diag(matrix)
        same = float(diagonal.mean())
        other = float(
            (matrix.sum() - diagonal.sum())
            / (matrix.shape[0] * (matrix.shape[1] - 1))
        )
        if same <= other:
            continue
        enrichment = same / max(other, 1.0e-30)
        confidence = (same - other) / max(same + other, 1.0e-30)
        records.append(
            {
                "block": block,
                "head": head,
                "protocol": "all_token",
                "primary": "S",
                "display_primary": "S_all",
                "primary_name": "全部时空 query 的帧内空间偏好",
                "secondary": "non-S",
                "secondary_name": "未在此二分类协议中细分",
                "margin": confidence,
                "features": {
                    "same_frame_nonself_mass": same,
                    "other_frame_mean_mass": other,
                    "same_frame_enrichment": enrichment,
                    "same_frame_win_rate": float(
                        same_frame_win_rate[head].mean()
                    ),
                    "exact_self_mass": float(exact_self_mass[head].mean()),
                },
                "scores": {
                    "S_all": confidence,
                    "same_frame_enrichment": enrichment,
                },
            }
        )
    return records


def _page(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan+LoRA latent-aligned attention</title>
<style>
:root{{--bg:#f3f4f1;--ink:#202421;--line:#c9cec9;--panel:#fff;--accent:#0d7155}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}
header{{padding:14px 22px;border-bottom:1px solid var(--line);background:#fff;position:sticky;top:0;z-index:3;box-shadow:0 2px 8px rgba(25,35,29,.08)}}
h1{{font-size:22px;margin:0 0 7px}} p{{margin:5px 0;line-height:1.45}} main{{padding:16px 20px 30px}}
.page-link{{display:inline-block;margin-top:5px;color:var(--accent);font-weight:700;text-decoration:none}} .page-link:hover{{text-decoration:underline}}
.controls{{display:flex;flex-wrap:wrap;align-items:end;gap:16px;padding:9px 0 0}}
label{{display:grid;gap:5px;font-weight:700}} select,input[type=range]{{min-width:180px}}
.value{{font-variant-numeric:tabular-nums;color:var(--accent)}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:5px;padding:8px;min-width:0}}
.panel h2{{font-size:15px;margin:0 0 7px}} canvas{{display:block;width:100%;height:auto;background:#111;image-rendering:pixelated}}
.reference{{max-width:560px;margin-bottom:14px}} .head-grid{{display:grid;grid-template-columns:1fr;gap:10px}}
.head-card{{background:#fff;border:1px solid var(--line);border-radius:5px;padding:10px;min-width:0}}
.head-card h2{{font-size:17px;margin:0 0 7px;display:flex;align-items:center;flex-wrap:wrap;gap:6px}} .head-card h3{{font-size:12px;margin:0 0 5px;line-height:1.3}}
.badge{{display:inline-block;border:1px solid #aeb6b1;border-radius:4px;padding:2px 5px;font-size:11px;font-weight:700;background:#f2f4f2}}
.role-S{{background:#e2f3e9;border-color:#72ad88}} .role-T{{background:#e2eef8;border-color:#739fbe}}
.role-P{{background:#fff0cc;border-color:#c6a24d}} .role-C{{background:#f8e2df;border-color:#bd7f76}} .role-G{{background:#e9e9e9;border-color:#999}}
.head-panels{{display:grid;grid-template-columns:repeat(6,minmax(170px,1fr));gap:6px;align-items:start}}
.matrix{{aspect-ratio:1}}
.full-matrix{{display:block;width:100%;height:auto;aspect-ratio:1;image-rendering:pixelated;background:#111}}
.fixed-strip{{margin-top:8px;border-top:1px solid #d9ddd9;padding-top:7px}} .fixed-strip canvas{{width:100%;height:auto;image-rendering:pixelated}}
.legend{{height:8px;margin-top:5px;background:linear-gradient(90deg,#30123b,#38598c,#1f9e89,#9fda3a,#fde725)}}
code{{font-family:ui-monospace,SFMono-Regular,monospace}}
@media(max-width:1450px){{.head-panels{{grid-template-columns:repeat(3,1fr)}}}} @media(max-width:760px){{.head-panels{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<header>
<h1>Wan+LoRA：双球独立 moving-query attention</h1>
<p>球 A 与球 B 使用两条身份锁定轨迹；球 B 在 t0–t2 不可见，对应 query 行保持为空。</p>
<p>空间 attention 保持原生 <code>16×28</code>，每格直接画成 <code>32×32</code> 像素块；无双线性、双三次或时间插值。</p>
<p>All-token Q→K 使用全部 5824 个 query/key token 的精确 softmax，再连续池化为 <code>512×512</code>；白线分隔 13 个 latent 时刻。</p>
<a class="page-link" href="grouped_by_role.html">切换到按 Head 类别跨 Block 查看</a>
<div class="controls">
  <label>Block<select id="block"></select></label>
  <label>Latent 时刻 <span class="value" id="latentValue"></span><input id="latent" type="range" min="0" max="12" value="2"></label>
  <label>该 latent 对应的视频帧 <span class="value" id="phaseValue"></span><input id="phase" type="range" min="0" max="3" value="3"></label>
</div>
</header>
<main>
<section class="panel reference"><h2 id="frameTitle"></h2><canvas id="frame" width="896" height="512"></canvas></section>
<div class="head-grid" id="headGrid"></div>
</main>
<script>
const META={payload};
const blockEl=document.getElementById("block");
for(const b of META.blocks) blockEl.add(new Option(`Block ${{String(b).padStart(2,"0")}}`,b));
const cache=new Map();
async function loadBlock(block){{
  if(cache.has(block)) return cache.get(block);
  const prefix=`data/block${{String(block).padStart(2,"0")}}`;
  const [fa,fb,ma,mb,ta,tb,at]=await Promise.all([
    fetch(prefix+"_fixed_A.f32").then(r=>r.arrayBuffer()),
    fetch(prefix+"_fixed_B.f32").then(r=>r.arrayBuffer()),
    fetch(prefix+"_moving_A.f32").then(r=>r.arrayBuffer()),
    fetch(prefix+"_moving_B.f32").then(r=>r.arrayBuffer()),
    fetch(prefix+"_temporal_A.f32").then(r=>r.arrayBuffer()),
    fetch(prefix+"_temporal_B.f32").then(r=>r.arrayBuffer()),
    fetch(prefix+"_all_token_temporal.f32").then(r=>r.arrayBuffer())
  ]);
  const value={{fixedA:new Float32Array(fa),fixedB:new Float32Array(fb),movingA:new Float32Array(ma),movingB:new Float32Array(mb),temporalA:new Float32Array(ta),temporalB:new Float32Array(tb),allTokenTemporal:new Float32Array(at)}};
  cache.set(block,value); return value;
}}
function turbo(x){{
  x=Math.max(0,Math.min(1,x));
  const stops=[[48,18,59],[56,89,140],[31,158,137],[159,218,58],[253,231,37]];
  const p=x*(stops.length-1),i=Math.min(stops.length-2,Math.floor(p)),a=p-i;
  return stops[i].map((v,j)=>Math.round(v*(1-a)+stops[i+1][j]*a));
}}
function range(array,offset,count){{
  let lo=Infinity,hi=-Infinity;
  for(let i=0;i<count;i++){{const v=array[offset+i];if(!Number.isFinite(v))continue;if(v<lo)lo=v;if(v>hi)hi=v;}}
  return [lo,hi];
}}
function role(block,head,protocol){{return META.roles.find(x=>x.block===block&&x.head===head&&x.protocol===protocol);}}
function frameIndex(t,p){{return t===0?0:1+4*(t-1)+p;}}
function drawOverlay(canvas,image,array,base,lo,hi,coords,t){{
  const c=canvas.getContext("2d"); c.imageSmoothingEnabled=false;c.clearRect(0,0,896,512);c.drawImage(image,0,0);
  if(!Number.isFinite(array[base])){{
    c.fillStyle="rgba(40,44,42,.72)";c.fillRect(0,0,896,512);c.fillStyle="#fff";c.font="bold 28px Arial";c.fillText("query unavailable",320,265);return;
  }}
  c.globalAlpha=.62;
  for(let y=0;y<16;y++)for(let x=0;x<28;x++){{
    const v=array[base+y*28+x],n=hi>lo?(v-lo)/(hi-lo):0,col=turbo(n);
    c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(x*32,y*32,32,32);
  }}
  c.globalAlpha=1;c.strokeStyle="#fff";c.lineWidth=3;
  const here=coords.filter(q=>q[0]===t);
  if(here.length){{
    const ys=here.map(q=>q[1]),xs=here.map(q=>q[2]);
    c.strokeRect(Math.min(...xs)*32,Math.min(...ys)*32,(Math.max(...xs)-Math.min(...xs)+1)*32,(Math.max(...ys)-Math.min(...ys)+1)*32);
  }}
}}
function drawMatrix(canvas,array,base,highlightFixed){{
  const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.clearRect(0,0,520,520);
  const [lo,hi]=range(array,base,169);
  for(let q=0;q<13;q++)for(let k=0;k<13;k++){{
    const value=array[base+q*13+k];
    if(!Number.isFinite(value)){{c.fillStyle="#555";c.fillRect(k*40,q*40,40,40);continue;}}
    const n=hi>lo?(value-lo)/(hi-lo):0,col=turbo(n);c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(k*40,q*40,40,40);
  }}
  c.strokeStyle="rgba(255,255,255,.38)";c.lineWidth=1;
  for(let i=0;i<=13;i++){{c.beginPath();c.moveTo(i*40,0);c.lineTo(i*40,520);c.stroke();c.beginPath();c.moveTo(0,i*40);c.lineTo(520,i*40);c.stroke();}}
  c.fillStyle="#fff";c.font="12px Arial";c.fillText("Q time ↓",5,15);c.fillText("K time →",445,15);
  if(highlightFixed){{c.strokeStyle="#ffdf4d";c.lineWidth=5;c.strokeRect(2,FIXED_QUERY_TIME*40+2,516,36);c.fillStyle="#ffdf4d";c.font="bold 13px Arial";c.fillText("fixed Q=t2",5,FIXED_QUERY_TIME*40+16);}}
}}
function drawFixedStrip(canvas,array,base,lo,hi,queryTime){{
  const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.fillStyle="#111";c.fillRect(0,0,1456,88);
  const scale=4,top=24,frameWidth=28*scale;
  for(let t=0;t<13;t++){{
    for(let y=0;y<16;y++)for(let x=0;x<28;x++){{
      const value=array[base+t*16*28+y*28+x],n=hi>lo?(value-lo)/(hi-lo):0,col=turbo(n);
      c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(t*frameWidth+x*scale,top+y*scale,scale,scale);
    }}
    c.fillStyle=t===queryTime?"#ffdf4d":"#fff";c.font=t===queryTime?"bold 13px Arial":"12px Arial";c.fillText(`K=t${{t}}`,t*frameWidth+5,16);
    if(t>0){{c.strokeStyle="rgba(255,255,255,.8)";c.lineWidth=1;c.beginPath();c.moveTo(t*frameWidth,top);c.lineTo(t*frameWidth,88);c.stroke();}}
  }}
  c.strokeStyle="#ffdf4d";c.lineWidth=3;c.strokeRect(queryTime*frameWidth+1,top+1,frameWidth-2,62);
}}
function buildHeadCards(block){{
  let html="";
  for(let h=0;h<24;h++){{
    const f=role(block,h,"fixed_A"),ma=role(block,h,"moving_A"),mb=role(block,h,"moving_B");
    html+=`<article class="head-card">
      <h2>Block ${{String(block).padStart(2,"0")}} · Head ${{String(h).padStart(2,"0")}}
        <span class="badge role-${{f.primary}}" title="${{f.primary_name}}">Fixed A: ${{f.display_primary}}</span>
        <span class="badge role-${{ma.primary}}" title="${{ma.primary_name}}">Moving A: ${{ma.display_primary}}</span>
        <span class="badge role-${{mb.primary}}" title="${{mb.primary_name}}">Moving B: ${{mb.display_primary}}</span>
      </h2>
      <div class="head-panels">
        <div><h3>Fixed ball A Q(t=2) → K(t)</h3><canvas id="fixed-${{h}}" width="896" height="512"></canvas><div class="legend"></div></div>
        <div><h3>Moving ball A Q(t) → K(t)</h3><canvas id="moving-a-${{h}}" width="896" height="512"></canvas><div class="legend"></div></div>
        <div><h3>Moving ball B Q(t) → K(t)</h3><canvas id="moving-b-${{h}}" width="896" height="512"></canvas><div class="legend"></div></div>
        <div><h3>Ball A Q-time × K-time</h3><canvas class="matrix" id="matrix-a-${{h}}" width="520" height="520"></canvas><div class="legend"></div></div>
        <div><h3>Ball B Q-time × K-time</h3><canvas class="matrix" id="matrix-b-${{h}}" width="520" height="520"></canvas><div class="legend"></div></div>
        <div><h3>All-token Q-time × K-time · exact-self removed</h3><canvas class="matrix" id="matrix-all-${{h}}" width="520" height="520"></canvas><div class="legend"></div></div>
        <div><h3>All-token Q→K · 5824→512 bins</h3><img class="full-matrix" loading="lazy" src="full_qk/block${{String(block).padStart(2,"0")}}/head${{String(h).padStart(2,"0")}}.png"></div>
      </div>
      <div class="fixed-strip"><h3>Fixed ball A Q(t=2) → all K frames · t0…t12 concatenated</h3><canvas id="fixed-strip-${{h}}" width="1456" height="88"></canvas><div class="legend"></div></div>
      <div class="fixed-strip"><h3>Fixed ball B Q(t=3) → all K frames · t0…t12 concatenated</h3><canvas id="fixed-b-strip-${{h}}" width="1456" height="88"></canvas><div class="legend"></div></div>
    </article>`;
  }}
  document.getElementById("headGrid").innerHTML=html;
}}
let renderToken=0;
async function render(){{
  const token=++renderToken;
  const block=+blockEl.value,t=+document.getElementById("latent").value;
  const phaseEl=document.getElementById("phase");phaseEl.max=t===0?0:3;if(+phaseEl.value>+phaseEl.max)phaseEl.value=phaseEl.max;
  const phase=+phaseEl.value,frame=frameIndex(t,phase),data=await loadBlock(block);
  document.getElementById("latentValue").textContent=`t${{t}}`;
  document.getElementById("phaseValue").textContent=t===0?"唯一帧":`${{phase+1}}/4`;
  document.getElementById("frameTitle").textContent=`最终生成视频：frame ${{frame}}（对应 latent t${{t}}）`;
  const image=new Image();image.src=`generated_frames/frame_${{String(frame).padStart(3,"0")}}.png`;await image.decode();
  if(token!==renderToken)return;
  buildHeadCards(block);
  const fc=document.getElementById("frame").getContext("2d");fc.imageSmoothingEnabled=false;fc.drawImage(image,0,0);
  const mapSize=13*16*28;
  for(let head=0;head<24;head++){{
    const headBase=head*mapSize;
    const [flo,fhi]=range(data.fixedA,headBase,mapSize),[fblo,fbhi]=range(data.fixedB,headBase,mapSize),[alo,ahi]=range(data.movingA,headBase,mapSize),[blo,bhi]=range(data.movingB,headBase,mapSize);
    drawOverlay(document.getElementById(`fixed-${{head}}`),image,data.fixedA,headBase+t*16*28,flo,fhi,META.track_query_coords[0],FIXED_QUERY_TIME);
    drawOverlay(document.getElementById(`moving-a-${{head}}`),image,data.movingA,headBase+t*16*28,alo,ahi,META.track_query_coords[0],t);
    drawOverlay(document.getElementById(`moving-b-${{head}}`),image,data.movingB,headBase+t*16*28,blo,bhi,META.track_query_coords[1],t);
    drawMatrix(document.getElementById(`matrix-a-${{head}}`),data.temporalA,head*169,true);
    drawMatrix(document.getElementById(`matrix-b-${{head}}`),data.temporalB,head*169,false);
    drawMatrix(document.getElementById(`matrix-all-${{head}}`),data.allTokenTemporal,head*169,false);
    drawFixedStrip(document.getElementById(`fixed-strip-${{head}}`),data.fixedA,headBase,flo,fhi,FIXED_QUERY_TIME);
    drawFixedStrip(document.getElementById(`fixed-b-strip-${{head}}`),data.fixedB,headBase,fblo,fbhi,FIXED_B_QUERY_TIME);
  }}
}}
const FIXED_QUERY_TIME=2;
const FIXED_B_QUERY_TIME=3;
blockEl.addEventListener("input",render);document.getElementById("phase").addEventListener("input",render);
document.getElementById("latent").addEventListener("input",()=>{{const t=+document.getElementById("latent").value;document.getElementById("phase").value=t===0?0:3;render();}});
render();
</script>
</body></html>"""


def _grouped_page(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan+LoRA heads grouped by role</title>
<style>
:root{{--bg:#f3f4f1;--ink:#202421;--line:#c9cec9;--accent:#0d7155}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}
header{{padding:13px 20px;background:#fff;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5;box-shadow:0 2px 8px rgba(25,35,29,.08)}}
h1{{font-size:21px;margin:0 0 5px}} p{{margin:4px 0;line-height:1.4}} main{{padding:14px 18px 30px}}
.page-link{{display:inline-block;margin-top:4px;color:var(--accent);font-weight:700;text-decoration:none}} .page-link:hover{{text-decoration:underline}}
.controls{{display:flex;flex-wrap:wrap;align-items:end;gap:14px;padding-top:9px}} label{{display:grid;gap:4px;font-weight:700}}
select,input[type=range]{{min-width:170px}} .value{{color:var(--accent);font-variant-numeric:tabular-nums}}
.head-grid{{display:grid;grid-template-columns:1fr;gap:10px}} .head-card{{background:#fff;border:1px solid var(--line);border-radius:5px;padding:9px;min-width:0}}
.head-card h2{{font-size:16px;margin:0 0 7px;display:flex;align-items:center;flex-wrap:wrap;gap:5px}}
.badge{{border:1px solid #aeb6b1;border-radius:4px;padding:2px 5px;font-size:11px;background:#f2f4f2}}
.role-S{{background:#e2f3e9;border-color:#72ad88}} .role-T{{background:#e2eef8;border-color:#739fbe}}
.role-P{{background:#fff0cc;border-color:#c6a24d}} .role-C{{background:#f8e2df;border-color:#bd7f76}} .role-G{{background:#e9e9e9;border-color:#999}}
.head-panels{{display:grid;grid-template-columns:repeat(6,minmax(170px,1fr));gap:6px;align-items:start}}
.head-card h3{{font-size:12px;margin:0 0 4px}} canvas,.full-matrix{{display:block;width:100%;height:auto;background:#111;image-rendering:pixelated}}
.matrix,.full-matrix{{aspect-ratio:1}} .fixed-strip{{margin-top:7px;border-top:1px solid #ddd;padding-top:6px}}
.legend{{height:7px;margin-top:4px;background:linear-gradient(90deg,#30123b,#38598c,#1f9e89,#9fda3a,#fde725)}}
.placeholder{{height:90px;display:grid;place-items:center;color:#69736e;background:#f7f8f7}}
@media(max-width:1450px){{.head-panels{{grid-template-columns:repeat(3,1fr)}}}} @media(max-width:760px){{.head-panels{{grid-template-columns:1fr 1fr}}}}
</style>
</head>
<body>
<header>
<h1>按 Head 类别跨 Block 分组</h1>
<p>同一协议、同一主类别的 Head 集中展示；按主类别相对最强竞争类别的分差 Δ 降序。S 与相邻轨迹传播接近时标为 S+T_adj。</p>
<p>All-token S 对全部5824个 query等权统计；逐 query移除 exact-self并重新归一化，当平均同帧质量大于平均其他单帧质量时归入 S_all。</p>
<a class="page-link" href="index.html">返回按 Block 查看</a>
<div class="controls">
  <label>分类协议<select id="protocol">
    <option value="fixed_A">Fixed ball A</option>
    <option value="moving_A">Moving ball A</option>
    <option value="moving_B">Moving ball B</option>
    <option value="all_token" selected>All-token S</option>
  </select></label>
  <label>主类别<select id="category">
    <option value="S" selected>S · 帧内空间</option><option value="T">T · 球轨迹传播</option>
    <option value="P">P · 固定位置时间对齐</option><option value="C">C · 首帧/历史上下文</option>
    <option value="G">G · 全局聚合</option>
  </select></label>
  <label>Latent <span class="value" id="latentValue"></span><input id="latent" type="range" min="0" max="12" value="3"></label>
  <label>视频帧 <span class="value" id="phaseValue"></span><input id="phase" type="range" min="0" max="3" value="3"></label>
  <strong id="count"></strong>
</div>
</header>
<main><div class="head-grid" id="headGrid"></div></main>
<script>
const META={payload}, FIXED_QUERY_TIME=2, FIXED_B_QUERY_TIME=3;
const protocolEl=document.getElementById("protocol"),categoryEl=document.getElementById("category");
const latentEl=document.getElementById("latent"),phaseEl=document.getElementById("phase"),gridEl=document.getElementById("headGrid");
const cache=new Map(),imageCache=new Map(),visibleCards=new Set();let observer,renderEpoch=0;
async function loadBlock(block){{
  if(cache.has(block))return cache.get(block);
  const p=`data/block${{String(block).padStart(2,"0")}}`;
  const buffers=await Promise.all(["fixed_A","fixed_B","moving_A","moving_B","temporal_A","temporal_B","all_token_temporal"].map(name=>fetch(`${{p}}_${{name}}.f32`).then(r=>r.arrayBuffer())));
  const value={{fixedA:new Float32Array(buffers[0]),fixedB:new Float32Array(buffers[1]),movingA:new Float32Array(buffers[2]),movingB:new Float32Array(buffers[3]),temporalA:new Float32Array(buffers[4]),temporalB:new Float32Array(buffers[5]),allTokenTemporal:new Float32Array(buffers[6])}};
  cache.set(block,value);return value;
}}
async function loadFrame(index){{
  if(imageCache.has(index))return imageCache.get(index);
  const image=new Image();image.src=`generated_frames/frame_${{String(index).padStart(3,"0")}}.png`;await image.decode();imageCache.set(index,image);return image;
}}
function turbo(x){{x=Math.max(0,Math.min(1,x));const s=[[48,18,59],[56,89,140],[31,158,137],[159,218,58],[253,231,37]],p=x*(s.length-1),i=Math.min(s.length-2,Math.floor(p)),a=p-i;return s[i].map((v,j)=>Math.round(v*(1-a)+s[i+1][j]*a));}}
function range(a,o,n){{let lo=Infinity,hi=-Infinity;for(let i=0;i<n;i++){{const v=a[o+i];if(!Number.isFinite(v))continue;if(v<lo)lo=v;if(v>hi)hi=v;}}return[lo,hi];}}
function role(block,head,protocol){{return META.roles.find(x=>x.block===block&&x.head===head&&x.protocol===protocol);}}
function frameIndex(t,p){{return t===0?0:1+4*(t-1)+p;}}
function drawOverlay(canvas,image,array,base,lo,hi,coords,t){{
  const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.clearRect(0,0,896,512);c.drawImage(image,0,0);
  if(!Number.isFinite(array[base])){{c.fillStyle="rgba(40,44,42,.72)";c.fillRect(0,0,896,512);c.fillStyle="#fff";c.font="bold 28px Arial";c.fillText("query unavailable",320,265);return;}}
  c.globalAlpha=.62;for(let y=0;y<16;y++)for(let x=0;x<28;x++){{const v=array[base+y*28+x],n=hi>lo?(v-lo)/(hi-lo):0,col=turbo(n);c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(x*32,y*32,32,32);}}
  c.globalAlpha=1;c.strokeStyle="#fff";c.lineWidth=3;const here=coords.filter(q=>q[0]===t);
  if(here.length){{const ys=here.map(q=>q[1]),xs=here.map(q=>q[2]);c.strokeRect(Math.min(...xs)*32,Math.min(...ys)*32,(Math.max(...xs)-Math.min(...xs)+1)*32,(Math.max(...ys)-Math.min(...ys)+1)*32);}}
}}
function drawMatrix(canvas,array,base,highlight){{
  const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.clearRect(0,0,520,520);const [lo,hi]=range(array,base,169);
  for(let q=0;q<13;q++)for(let k=0;k<13;k++){{const v=array[base+q*13+k];if(!Number.isFinite(v)){{c.fillStyle="#555";}}else{{const n=hi>lo?(v-lo)/(hi-lo):0,col=turbo(n);c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;}}c.fillRect(k*40,q*40,40,40);}}
  c.strokeStyle="rgba(255,255,255,.38)";c.lineWidth=1;for(let i=0;i<=13;i++){{c.beginPath();c.moveTo(i*40,0);c.lineTo(i*40,520);c.stroke();c.beginPath();c.moveTo(0,i*40);c.lineTo(520,i*40);c.stroke();}}
  if(highlight){{c.strokeStyle="#ffdf4d";c.lineWidth=5;c.strokeRect(2,82,516,36);}}
}}
function drawStrip(canvas,array,base,lo,hi,queryTime){{
  const c=canvas.getContext("2d");c.imageSmoothingEnabled=false;c.fillStyle="#111";c.fillRect(0,0,1456,88);const scale=4,top=24,fw=112;
  for(let t=0;t<13;t++){{for(let y=0;y<16;y++)for(let x=0;x<28;x++){{const v=array[base+t*448+y*28+x],n=hi>lo?(v-lo)/(hi-lo):0,col=turbo(n);c.fillStyle=`rgb(${{col[0]}},${{col[1]}},${{col[2]}})`;c.fillRect(t*fw+x*scale,top+y*scale,scale,scale);}}c.fillStyle=t===queryTime?"#ffdf4d":"#fff";c.font="12px Arial";c.fillText(`K=t${{t}}`,t*fw+5,16);}}
}}
function cardHtml(record){{
  const b=record.block,h=record.head,f=role(b,h,"fixed_A"),a=role(b,h,"moving_A"),m=role(b,h,"moving_B"),bs=String(b).padStart(2,"0"),hs=String(h).padStart(2,"0");
  const current=record.protocol==="all_token"?`<span class="badge role-S">All-token: S_all</span>`:"";
  const evidence=record.protocol==="all_token"?`<span class="badge">same=${{record.features.same_frame_nonself_mass.toFixed(3)}} · other=${{record.features.other_frame_mean_mass.toFixed(3)}} · E=${{record.features.same_frame_enrichment.toFixed(2)}}</span>`:"";
  return `<article class="head-card" data-block="${{b}}" data-head="${{h}}"><h2>Block ${{bs}} · Head ${{hs}} ${{current}}<span class="badge role-${{f.primary}}">Fixed A: ${{f.display_primary}}</span><span class="badge role-${{a.primary}}">Moving A: ${{a.display_primary}}</span><span class="badge role-${{m.primary}}">Moving B: ${{m.display_primary}}</span><span class="badge">当前协议 Δ=${{record.margin.toFixed(3)}}</span>${{evidence}}</h2><div class="placeholder">滚动到此处加载可视化</div></article>`;
}}
function cardBody(block,head){{
  const b=String(block).padStart(2,"0"),h=String(head).padStart(2,"0");
  return `<div class="head-panels">
  <div><h3>Fixed A</h3><canvas data-kind="fixed" width="896" height="512"></canvas><div class="legend"></div></div>
  <div><h3>Moving A</h3><canvas data-kind="movingA" width="896" height="512"></canvas><div class="legend"></div></div>
  <div><h3>Moving B</h3><canvas data-kind="movingB" width="896" height="512"></canvas><div class="legend"></div></div>
  <div><h3>Ball A Q-time × K-time</h3><canvas class="matrix" data-kind="temporalA" width="520" height="520"></canvas><div class="legend"></div></div>
  <div><h3>Ball B Q-time × K-time</h3><canvas class="matrix" data-kind="temporalB" width="520" height="520"></canvas><div class="legend"></div></div>
  <div><h3>All-token Q-time × K-time · exact-self removed</h3><canvas class="matrix" data-kind="allTokenTemporal" width="520" height="520"></canvas><div class="legend"></div></div>
  <div><h3>All-token Q→K</h3><img class="full-matrix" loading="lazy" src="full_qk/block${{b}}/head${{h}}.png"></div></div>
  <div class="fixed-strip"><h3>Fixed ball A Q(t=2) → all K frames · t0…t12 concatenated</h3><canvas data-kind="stripA" width="1456" height="88"></canvas><div class="legend"></div></div>
  <div class="fixed-strip"><h3>Fixed ball B Q(t=3) → all K frames · t0…t12 concatenated</h3><canvas data-kind="stripB" width="1456" height="88"></canvas><div class="legend"></div></div>`;
}}
async function renderCard(card,epoch){{
  const block=+card.dataset.block,head=+card.dataset.head,t=+latentEl.value,phase=+phaseEl.value,frame=frameIndex(t,phase);
  if(!card.querySelector("[data-kind]")){{card.querySelector(".placeholder").outerHTML=cardBody(block,head);}}
  const [data,image]=await Promise.all([loadBlock(block),loadFrame(frame)]);if(epoch!==renderEpoch||!visibleCards.has(card))return;
  const size=13*16*28,base=head*size,[flo,fhi]=range(data.fixedA,base,size),[fblo,fbhi]=range(data.fixedB,base,size),[alo,ahi]=range(data.movingA,base,size),[blo,bhi]=range(data.movingB,base,size);
  drawOverlay(card.querySelector('[data-kind="fixed"]'),image,data.fixedA,base+t*448,flo,fhi,META.track_query_coords[0],2);
  drawOverlay(card.querySelector('[data-kind="movingA"]'),image,data.movingA,base+t*448,alo,ahi,META.track_query_coords[0],t);
  drawOverlay(card.querySelector('[data-kind="movingB"]'),image,data.movingB,base+t*448,blo,bhi,META.track_query_coords[1],t);
  drawMatrix(card.querySelector('[data-kind="temporalA"]'),data.temporalA,head*169,true);drawMatrix(card.querySelector('[data-kind="temporalB"]'),data.temporalB,head*169,false);
  drawMatrix(card.querySelector('[data-kind="allTokenTemporal"]'),data.allTokenTemporal,head*169,false);
  drawStrip(card.querySelector('[data-kind="stripA"]'),data.fixedA,base,flo,fhi,FIXED_QUERY_TIME);
  drawStrip(card.querySelector('[data-kind="stripB"]'),data.fixedB,base,fblo,fbhi,FIXED_B_QUERY_TIME);
}}
function renderVisible(){{const epoch=++renderEpoch;for(const card of visibleCards)renderCard(card,epoch);}}
function buildGroup(){{
  if(observer)observer.disconnect();visibleCards.clear();const protocol=protocolEl.value;
  if(protocol==="all_token"){{categoryEl.value="S";categoryEl.disabled=true;}}else{{categoryEl.disabled=false;}}
  const category=categoryEl.value;
  const rows=META.roles.filter(x=>x.protocol===protocol&&x.primary===category).sort((a,b)=>b.margin-a.margin||a.block-b.block||a.head-b.head);
  document.getElementById("count").textContent=`${{rows.length}} 个 Head`;gridEl.innerHTML=rows.map(cardHtml).join("");
  observer=new IntersectionObserver(entries=>{{for(const entry of entries){{if(entry.isIntersecting){{visibleCards.add(entry.target);renderCard(entry.target,renderEpoch);}}else visibleCards.delete(entry.target);}}}},{{rootMargin:"800px 0px"}});
  gridEl.querySelectorAll(".head-card").forEach(card=>observer.observe(card));
}}
function updateLabels(){{const t=+latentEl.value;phaseEl.max=t===0?0:3;if(+phaseEl.value>+phaseEl.max)phaseEl.value=phaseEl.max;document.getElementById("latentValue").textContent=`t${{t}}`;document.getElementById("phaseValue").textContent=t===0?"frame 0":`${{+phaseEl.value+1}}/4`;}}
protocolEl.addEventListener("change",buildGroup);categoryEl.addEventListener("change",buildGroup);phaseEl.addEventListener("input",()=>{{updateLabels();renderVisible();}});
latentEl.addEventListener("input",()=>{{phaseEl.value=+latentEl.value===0?0:3;updateLabels();renderVisible();}});updateLabels();buildGroup();
</script>
</body></html>"""


def main() -> None:
    args = parse_args()
    root = args.capture_root.expanduser().resolve()
    query_map_path = args.query_map.expanduser().resolve()
    query_map = json.loads(query_map_path.read_text(encoding="utf-8"))
    output = args.output_dir.expanduser().resolve()
    data_dir = output / "data"
    frame_output = output / "generated_frames"
    full_qk_output = output / "full_qk"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_output.mkdir(parents=True, exist_ok=True)
    full_qk_output.mkdir(parents=True, exist_ok=True)

    generated_video = root / "generated" / f"{CASE}.mp4"
    if not generated_video.is_file():
        raise FileNotFoundError(generated_video)
    stale_vae_frames = output / "vae_frames"
    if stale_vae_frames.is_dir():
        shutil.rmtree(stale_vae_frames)
    capture = cv2.VideoCapture(str(generated_video))
    generated_frame_count = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (512, 896):
            raise ValueError(
                f"generated frame {generated_frame_count} has shape {frame.shape}"
            )
        path = frame_output / f"frame_{generated_frame_count:03d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"failed to write {path}")
        generated_frame_count += 1
    capture.release()
    if generated_frame_count != 49:
        raise ValueError(
            f"generated video has {generated_frame_count} frames, expected 49"
        )

    roles = []
    query_coords_ref: tuple[np.ndarray, np.ndarray] | None = None
    valid_query_times_ref: np.ndarray | None = None
    for block in range(30):
        summary_path = (
            root
            / "attention"
            / f"block{block:02d}"
            / "matrices"
            / MODEL
            / CASE
            / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        entry = summary["steps"][0]
        npz_path = summary_path.parent / entry["directory"] / entry["maps_npz"]
        with np.load(npz_path) as arrays:
            attention = arrays["attention"].astype(np.float32)
            selected_heads = arrays["selected_heads"].astype(np.int64)
            valid_query_times = arrays["valid_query_times"].astype(np.bool_)
            query_coords = (
                arrays["track_0_query_coords"].astype(np.int64),
                arrays["track_1_query_coords"].astype(np.int64),
            )
            track_names = arrays["track_names"].astype(str).tolist()
        full_path = (
            summary_path.parent
            / entry["directory"]
            / entry["full_matrix_npz"]
        )
        with np.load(full_path) as arrays:
            key_mass = arrays["key_mass"].astype(np.float32)
            required = {
                "time_matrix_no_exact_self",
                "exact_self_mass",
                "same_frame_win_rate",
            }
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError(
                    f"{full_path} lacks all-token temporal statistics: "
                    f"{sorted(missing)}"
                )
            all_token_temporal = arrays[
                "time_matrix_no_exact_self"
            ].astype(np.float32)
            exact_self_mass = arrays["exact_self_mass"].astype(np.float32)
            same_frame_win_rate = arrays[
                "same_frame_win_rate"
            ].astype(np.float32)
        if not np.array_equal(selected_heads, np.arange(24)):
            raise ValueError(f"{npz_path} does not contain heads 0..23 in order")
        if query_coords_ref is None:
            query_coords_ref = query_coords
            valid_query_times_ref = valid_query_times
        elif not all(
            np.array_equal(first, second)
            for first, second in zip(query_coords_ref, query_coords)
        ):
            raise ValueError(f"query coordinates differ at block {block}")
        elif not np.array_equal(valid_query_times_ref, valid_query_times):
            raise ValueError(f"valid query times differ at block {block}")
        fixed_a, fixed_b, moving, temporal, results = _block_data(
            attention, query_coords, valid_query_times
        )
        prefix = data_dir / f"block{block:02d}"
        _write_float32(prefix.with_name(prefix.name + "_fixed_A.f32"), fixed_a)
        _write_float32(prefix.with_name(prefix.name + "_fixed_B.f32"), fixed_b)
        _write_float32(prefix.with_name(prefix.name + "_moving_A.f32"), moving[0])
        _write_float32(prefix.with_name(prefix.name + "_moving_B.f32"), moving[1])
        _write_float32(prefix.with_name(prefix.name + "_temporal_A.f32"), temporal[0])
        _write_float32(prefix.with_name(prefix.name + "_temporal_B.f32"), temporal[1])
        _write_float32(
            prefix.with_name(prefix.name + "_all_token_temporal.f32"),
            all_token_temporal,
        )
        _render_full_matrix_images(
            key_mass, block=block, output_dir=full_qk_output
        )
        for head in range(24):
            for protocol, result in zip(
                ("fixed_A", "moving_A", "moving_B"), results
            ):
                roles.append(
                    _role_record(
                        block=block,
                        head=head,
                        protocol=protocol,
                        result=result,
                    )
                )
        roles.extend(
            _all_token_s_records(
                block=block,
                time_matrix=all_token_temporal,
                exact_self_mass=exact_self_mass,
                same_frame_win_rate=same_frame_win_rate,
            )
        )

    assert query_coords_ref is not None and valid_query_times_ref is not None
    metadata = {
        "case": CASE,
        "model": MODEL,
        "blocks": list(range(30)),
        "heads": list(range(24)),
        "denoise_step_one_based": 25,
        "cfg_branch": "positive",
        "latent_grid": list(GRID),
        "fixed_query_time": FIXED_QUERY_TIME,
        "fixed_b_query_time": FIXED_B_QUERY_TIME,
        "fixed_query_coords": query_coords_ref[0][
            query_coords_ref[0][:, 0] == FIXED_QUERY_TIME
        ].tolist(),
        "fixed_b_query_coords": query_coords_ref[1][
            query_coords_ref[1][:, 0] == FIXED_B_QUERY_TIME
        ].tolist(),
        "track_names": track_names,
        "track_query_coords": [coords.tolist() for coords in query_coords_ref],
        "valid_query_times": valid_query_times_ref.tolist(),
        "query_source_frame": "frame=4*latent_time",
        "normalization": (
            "separate min-max over all 13x16x28 values for each "
            "block/head/query protocol"
        ),
        "spatial_rendering": "16x28 cells drawn directly as 32x32 RGB blocks",
        "interpolation": "none",
        "all_token_matrix": {
            "token_count": 5824,
            "display_bins": 512,
            "query_sampling": "none",
            "softmax": "exact over all 5824 key tokens",
            "pooling": "contiguous_token_block_mean converted to key mass",
            "display_scale": "per-head log10 with 1.0/99.8 percentile clipping",
        },
        "all_token_s_protocol": {
            "query_sampling": "none; all 5824 query tokens are equally weighted",
            "exact_self": "removed per query, then remaining attention renormalized",
            "time_matrix_shape": [24, 13, 13],
            "same_frame": "mean diagonal mass over all query times",
            "other_frame": "mean mass of one off-diagonal key time",
            "decision": "S_all iff same_frame > other_frame",
            "confidence": "(same_frame-other_frame)/(same_frame+other_frame)",
            "sorting": "confidence descending",
        },
        "overlay_background": {
            "type": "final_generated_video_frames",
            "video": str(generated_video),
            "frame_count": generated_frame_count,
            "frame_shape_hw": [512, 896],
        },
        "query_map": str(query_map_path),
        "query_identity_policy": [
            track["identity_policy"] for track in query_map["tracks"]
        ],
        "roles": roles,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "head_roles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "block",
                "head",
                "protocol",
                "primary",
                "primary_name",
                "secondary",
                "secondary_name",
                "margin",
            ],
        )
        writer.writeheader()
        for row in roles:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    (output / "index.html").write_text(_page(metadata), encoding="utf-8")
    (output / "grouped_by_role.html").write_text(
        _grouped_page(metadata), encoding="utf-8"
    )
    print(f"[gallery] wrote {output / 'index.html'}")
    print(f"[gallery] wrote {output / 'grouped_by_role.html'}")


if __name__ == "__main__":
    main()
