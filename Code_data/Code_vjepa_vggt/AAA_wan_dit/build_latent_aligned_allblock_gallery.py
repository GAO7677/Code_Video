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
GALLERY_ROLE_LABELS = {
    **ROLE_LABELS,
    "ST": "帧内空间与相邻轨迹联合",
}
ST_MIN_SCORE = 0.70
ST_MIN_BALANCE = 0.80
PREVIOUS_MIN_MASS = 0.05
PREVIOUS_MIN_SHARE = 0.80
PREVIOUS_MIN_CONSISTENCY = 0.80


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
    primary = np.asarray(
        [roles[int(value)] for value in order[:, -1]], dtype=object
    )
    secondary = np.asarray(
        [roles[int(value)] for value in order[:, -2]], dtype=object
    )
    margin = (
        np.take_along_axis(matrix, order[:, -1:], axis=1)[:, 0]
        - np.take_along_axis(matrix, order[:, -2:-1], axis=1)[:, 0]
    )
    st_score = (
        2.0
        * scores["S"]
        * scores["T_adj"]
        / np.maximum(scores["S"] + scores["T_adj"], 1.0e-30)
    )
    scores["ST"] = st_score
    balance = np.minimum(scores["S"], scores["T_adj"]) / np.maximum(
        np.maximum(scores["S"], scores["T_adj"]), 1.0e-30
    )
    competitor_labels = ("P", "C", "G", "T")
    competitor_values = np.stack(
        [
            scores["P"],
            scores["C"],
            scores["G"],
            scores["T_long"],
        ],
        axis=1,
    )
    competitor_order = np.argmax(competitor_values, axis=1)
    competitor_score = np.take_along_axis(
        competitor_values, competitor_order[:, None], axis=1
    )[:, 0]
    st_mask = (
        (scores["S"] >= ST_MIN_SCORE)
        & (scores["T_adj"] >= ST_MIN_SCORE)
        & (balance >= ST_MIN_BALANCE)
        & (scores["T_adj"] > scores["T_long"])
        & (st_score > competitor_score)
    )
    primary[st_mask] = "ST"
    secondary[st_mask] = np.asarray(competitor_labels, dtype=object)[
        competitor_order[st_mask]
    ]
    margin[st_mask] = st_score[st_mask] - competitor_score[st_mask]
    return {
        "scores": scores,
        "primary": primary,
        "secondary": secondary,
        "margin": margin,
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
        for role in (*GALLERY_ROLE_LABELS, "T_adj", "T_long")
    }
    return {
        "block": block,
        "head": head,
        "protocol": protocol,
        "primary": primary,
        "display_primary": primary,
        "primary_name": GALLERY_ROLE_LABELS[primary],
        "secondary": secondary,
        "secondary_name": GALLERY_ROLE_LABELS[secondary],
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


def _previous_trajectory_metrics(
    attention: np.ndarray,
    *,
    query_coords: np.ndarray,
    valid_query_times: np.ndarray,
) -> dict[str, np.ndarray]:
    if attention.shape != (24, 13, 13, 16, 28):
        raise ValueError(f"unexpected track attention shape: {attention.shape}")
    token_count = int(np.prod(GRID))
    previous_mass = []
    next_mass = []
    previous_enrichment = []
    next_enrichment = []
    previous_wins = []
    for query_time in range(1, GRID[0] - 1):
        if not (
            valid_query_times[query_time - 1]
            and valid_query_times[query_time]
            and valid_query_times[query_time + 1]
        ):
            continue
        previous_coords = query_coords[
            query_coords[:, 0] == query_time - 1
        ]
        next_coords = query_coords[query_coords[:, 0] == query_time + 1]
        previous = attention[
            :,
            query_time,
            query_time - 1,
            previous_coords[:, 1],
            previous_coords[:, 2],
        ].sum(1)
        following = attention[
            :,
            query_time,
            query_time + 1,
            next_coords[:, 1],
            next_coords[:, 2],
        ].sum(1)
        previous_enriched = previous / (
            len(previous_coords) / token_count
        )
        next_enriched = following / (len(next_coords) / token_count)
        previous_mass.append(previous)
        next_mass.append(following)
        previous_enrichment.append(previous_enriched)
        next_enrichment.append(next_enriched)
        previous_wins.append(previous_enriched > next_enriched)
    if not previous_mass:
        raise ValueError("track has no query times with both adjacent keys")
    previous_mass_mean = np.stack(previous_mass).mean(0)
    next_mass_mean = np.stack(next_mass).mean(0)
    previous_enrichment_mean = np.stack(previous_enrichment).mean(0)
    next_enrichment_mean = np.stack(next_enrichment).mean(0)
    previous_share = previous_enrichment_mean / np.maximum(
        previous_enrichment_mean + next_enrichment_mean, 1.0e-30
    )
    consistency = np.stack(previous_wins).mean(0)
    return {
        "previous_mass": previous_mass_mean,
        "next_mass": next_mass_mean,
        "previous_enrichment": previous_enrichment_mean,
        "next_enrichment": next_enrichment_mean,
        "previous_share": previous_share,
        "consistency": consistency,
    }


def _previous_trajectory_records(
    *,
    block: int,
    metrics_a: dict[str, np.ndarray],
    metrics_b: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    records = []
    for head in range(24):
        qualifies = []
        for metrics in (metrics_a, metrics_b):
            qualifies.append(
                bool(
                    metrics["previous_mass"][head] >= PREVIOUS_MIN_MASS
                    and metrics["previous_share"][head] >= PREVIOUS_MIN_SHARE
                    and metrics["consistency"][head]
                    >= PREVIOUS_MIN_CONSISTENCY
                )
            )
        if not any(qualifies):
            continue
        stable_both = all(qualifies)
        active_metrics = [
            metrics
            for metrics, active in zip((metrics_a, metrics_b), qualifies)
            if active
        ]
        confidence = min(
            float(
                (2.0 * metrics["previous_share"][head] - 1.0)
                * metrics["consistency"][head]
            )
            for metrics in active_metrics
        )
        scope = "A+B" if stable_both else ("A-only" if qualifies[0] else "B-only")
        features = {}
        for label, metrics in (("A", metrics_a), ("B", metrics_b)):
            for name, values in metrics.items():
                features[f"{label}_{name}"] = float(values[head])
        records.append(
            {
                "block": block,
                "head": head,
                "protocol": "previous_trajectory",
                "primary": "TP",
                "display_primary": f"T_prev {scope}",
                "primary_name": "只回看前一 latent 时刻的同一物体轨迹",
                "secondary": "non-TP",
                "secondary_name": "不满足前向轨迹协议",
                "margin": confidence,
                "stable_both": stable_both,
                "scope": scope,
                "features": features,
                "scores": {"T_prev": confidence},
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
.role-ST{{background:#dff2ef;border-color:#3f9d8a}}
.role-TP{{background:#e8e3f7;border-color:#8979b8}}
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
.category-summary{{display:flex;flex-wrap:wrap;gap:6px;padding-top:8px}} .summary-chip{{border:1px solid var(--line);border-radius:4px;background:#fff;padding:4px 8px;font-weight:700;cursor:pointer}} .summary-chip.active{{border-color:var(--accent);color:var(--accent);background:#e8f3ef}}
.head-grid{{display:grid;grid-template-columns:1fr;gap:10px}} .head-card{{background:#fff;border:1px solid var(--line);border-radius:5px;padding:9px;min-width:0}}
.head-card h2{{font-size:16px;margin:0 0 7px;display:flex;align-items:center;flex-wrap:wrap;gap:5px}}
.head-card a{{color:var(--accent);font-size:12px;text-decoration:none}} .head-card a:hover{{text-decoration:underline}}
.badge{{border:1px solid #aeb6b1;border-radius:4px;padding:2px 5px;font-size:11px;background:#f2f4f2}}
.role-S{{background:#e2f3e9;border-color:#72ad88}} .role-T{{background:#e2eef8;border-color:#739fbe}}
.role-ST{{background:#dff2ef;border-color:#3f9d8a}}
.role-TP{{background:#e8e3f7;border-color:#8979b8}}
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
<h1>Head 功能分类与跨 Block 可视化</h1>
<p>核心分类以 Fixed A、Moving A、Moving B 是否一致分层：338个一致 Head用于稳定类别分析，382个协议分歧 Head单独检查。诊断视图 S_all 与 T_prev 独立保留。</p>
<a class="page-link" href="index.html">返回按 Block 查看</a>
<a class="page-link" href="consistent_heads_distribution.html">查看338个一致 Head 的 Block 分布</a>
<div class="controls">
  <label>分类视图<select id="protocol">
    <option value="consistent" selected>三协议一致 · 338</option>
    <option value="disagreement">协议分歧 · 382</option>
    <option value="fixed_A">Fixed ball A</option>
    <option value="moving_A">Moving ball A</option>
    <option value="moving_B">Moving ball B</option>
    <option value="all_token">诊断 · All-token S</option>
    <option value="previous_trajectory">诊断 · T_prev</option>
  </select></label>
  <label>主类别<select id="category">
    <option value="ALL">全部类别</option><option value="S" selected>S · 帧内空间</option><option value="ST">ST · 帧内+相邻轨迹</option><option value="TP">T_prev · 前一时刻轨迹</option><option value="T">T · 球轨迹传播</option>
    <option value="P">P · 固定位置时间对齐</option><option value="C">C · 首帧/历史上下文</option>
    <option value="G">G · 全局聚合</option>
  </select></label>
  <label>Latent <span class="value" id="latentValue"></span><input id="latent" type="range" min="0" max="12" value="3"></label>
  <label>视频帧 <span class="value" id="phaseValue"></span><input id="phase" type="range" min="0" max="3" value="3"></label>
  <strong id="count"></strong>
</div>
<div class="category-summary" id="categorySummary"></div>
</header>
<main><div class="head-grid" id="headGrid"></div></main>
<script>
const META={payload}, FIXED_QUERY_TIME=2, FIXED_B_QUERY_TIME=3;
const protocolEl=document.getElementById("protocol"),categoryEl=document.getElementById("category");
const latentEl=document.getElementById("latent"),phaseEl=document.getElementById("phase"),gridEl=document.getElementById("headGrid");
const urlParams=new URLSearchParams(window.location.search);
const requestedView=urlParams.get("view")||urlParams.get("protocol"),requestedCategory=urlParams.get("category");
if([...protocolEl.options].some(option=>option.value===requestedView))protocolEl.value=requestedView;
if([...categoryEl.options].some(option=>option.value===requestedCategory))categoryEl.value=requestedCategory;
const cache=new Map(),imageCache=new Map(),visibleCards=new Set();let observer,renderEpoch=0;
const roleIndex=new Map(META.roles.map(row=>[`${{row.block}}:${{row.head}}:${{row.protocol}}`,row]));
function role(block,head,protocol){{return roleIndex.get(`${{block}}:${{head}}:${{protocol}}`);}}
const CONSISTENT=[],DISAGREEMENT=[];
for(let block=0;block<30;block++)for(let head=0;head<24;head++){{
  const entries=["fixed_A","moving_A","moving_B"].map(protocol=>role(block,head,protocol)),labels=[...new Set(entries.map(row=>row.primary))],meanMargin=entries.reduce((sum,row)=>sum+row.margin,0)/entries.length;
  if(labels.length===1)CONSISTENT.push({{block,head,protocol:"consistent",primary:labels[0],display_primary:labels[0],margin:meanMargin,min_margin:Math.min(...entries.map(row=>row.margin))}});
  else DISAGREEMENT.push({{block,head,protocol:"disagreement",primary:"ALL",display_primary:labels.join("/"),margin:meanMargin,label_count:labels.length}});
}}
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
  const stable=record.protocol==="consistent"?`<span class="badge role-${{record.primary}}">三协议一致: ${{record.primary}}</span><span class="badge">平均 Δ=${{record.margin.toFixed(3)}} · 最小 Δ=${{record.min_margin.toFixed(3)}}</span>`:"";
  const disputed=record.protocol==="disagreement"?`<span class="badge">协议分歧: ${{record.display_primary}} · 平均 Δ=${{record.margin.toFixed(3)}}</span>`:"";
  const current=record.protocol==="all_token"?`<span class="badge role-S">All-token: S_all</span>`:"";
  const evidence=record.protocol==="all_token"?`<span class="badge">same=${{record.features.same_frame_nonself_mass.toFixed(3)}} · other=${{record.features.other_frame_mean_mass.toFixed(3)}} · E=${{record.features.same_frame_enrichment.toFixed(2)}}</span>`:"";
  const previous=record.protocol==="previous_trajectory"?`<span class="badge role-TP">${{record.display_primary}}</span><span class="badge">A: mass=${{record.features.A_previous_mass.toFixed(3)}} share=${{record.features.A_previous_share.toFixed(3)}} · B: mass=${{record.features.B_previous_mass.toFixed(3)}} share=${{record.features.B_previous_share.toFixed(3)}}</span>`:"";
  const detail=b===3&&h===20?`<a href="head_details/block03_head20/">查看全部Q时刻</a>`:"";
  const protocolMargin=["fixed_A","moving_A","moving_B"].includes(record.protocol)?`<span class="badge">当前协议 Δ=${{record.margin.toFixed(3)}}</span>`:"";
  return `<article class="head-card" data-block="${{b}}" data-head="${{h}}"><h2>Block ${{bs}} · Head ${{hs}} ${{stable}}${{disputed}}${{current}}${{previous}}<span class="badge role-${{f.primary}}">Fixed A: ${{f.display_primary}}</span><span class="badge role-${{a.primary}}">Moving A: ${{a.display_primary}}</span><span class="badge role-${{m.primary}}">Moving B: ${{m.display_primary}}</span>${{protocolMargin}}${{evidence}}${{detail}}</h2><div class="placeholder">滚动到此处加载可视化</div></article>`;
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
const CATEGORY_DEFS={{
  ALL:"全部类别",S:"S · 帧内空间",ST:"ST · 帧内+相邻轨迹",T:"T · 轨迹传播",
  P:"P · 固定位置",C:"C · 历史/context",G:"G · 全局聚合",TP:"T_prev · 前一时刻轨迹"
}};
function allowedCategories(view){{
  if(view==="all_token")return["S"];
  if(view==="previous_trajectory")return["TP"];
  if(view==="disagreement")return["ALL"];
  return["ALL","S","ST","T","P","C","G"];
}}
function configureCategories(preferred){{
  const allowed=allowedCategories(protocolEl.value),candidate=preferred&&allowed.includes(preferred)?preferred:(allowed.includes(categoryEl.value)?categoryEl.value:allowed[0]);
  categoryEl.innerHTML=allowed.map(value=>`<option value="${{value}}">${{CATEGORY_DEFS[value]}}</option>`).join("");
  categoryEl.value=candidate;categoryEl.disabled=allowed.length===1;
}}
function sourceRows(view){{
  if(view==="consistent")return CONSISTENT;
  if(view==="disagreement")return DISAGREEMENT;
  return META.roles.filter(row=>row.protocol===view);
}}
function updateCategorySummary(rows,category){{
  const allowed=allowedCategories(protocolEl.value),summary=document.getElementById("categorySummary");
  summary.innerHTML=allowed.map(value=>{{const count=value==="ALL"?rows.length:rows.filter(row=>row.primary===value).length;return `<button class="summary-chip${{value===category?" active":""}}" data-category="${{value}}">${{CATEGORY_DEFS[value]}} · ${{count}}</button>`;}}).join("");
  summary.querySelectorAll("button").forEach(button=>button.addEventListener("click",()=>{{categoryEl.value=button.dataset.category;buildGroup();}}));
}}
function buildGroup(){{
  if(observer)observer.disconnect();visibleCards.clear();const view=protocolEl.value,category=categoryEl.value,allRows=sourceRows(view);let rows=category==="ALL"?[...allRows]:allRows.filter(row=>row.primary===category);
  const order={{S:0,ST:1,T:2,P:3,C:4,G:5}};
  rows.sort((a,b)=>view==="previous_trajectory"?(Number(b.stable_both)-Number(a.stable_both)||b.margin-a.margin||a.block-b.block||a.head-b.head):view==="disagreement"?(b.label_count-a.label_count||a.margin-b.margin||a.block-b.block||a.head-b.head):category==="ALL"?((order[a.primary]??99)-(order[b.primary]??99)||b.margin-a.margin||a.block-b.block||a.head-b.head):(b.margin-a.margin||a.block-b.block||a.head-b.head));
  updateCategorySummary(allRows,category);document.getElementById("count").textContent=`${{rows.length}} / ${{allRows.length}} 个 Head`;gridEl.innerHTML=rows.map(cardHtml).join("");
  observer=new IntersectionObserver(entries=>{{for(const entry of entries){{if(entry.isIntersecting){{visibleCards.add(entry.target);renderCard(entry.target,renderEpoch);}}else visibleCards.delete(entry.target);}}}},{{rootMargin:"800px 0px"}});
  gridEl.querySelectorAll(".head-card").forEach(card=>observer.observe(card));
}}
function updateLabels(){{const t=+latentEl.value;phaseEl.max=t===0?0:3;if(+phaseEl.value>+phaseEl.max)phaseEl.value=phaseEl.max;document.getElementById("latentValue").textContent=`t${{t}}`;document.getElementById("phaseValue").textContent=t===0?"frame 0":`${{+phaseEl.value+1}}/4`;}}
protocolEl.addEventListener("change",()=>{{configureCategories(null);buildGroup();}});categoryEl.addEventListener("change",buildGroup);phaseEl.addEventListener("input",()=>{{updateLabels();renderVisible();}});
latentEl.addEventListener("input",()=>{{phaseEl.value=+latentEl.value===0?0:3;updateLabels();renderVisible();}});configureCategories(requestedCategory);updateLabels();buildGroup();
</script>
</body></html>"""


def _consistent_distribution_page(metadata: dict[str, Any]) -> str:
    protocols = ("fixed_A", "moving_A", "moving_B")
    by_head: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in metadata["roles"]:
        if row["protocol"] not in protocols:
            continue
        by_head.setdefault((row["block"], row["head"]), []).append(row)
    cells = []
    for block in range(30):
        for head in range(24):
            rows = by_head[(block, head)]
            categories = {row["primary"] for row in rows}
            consistent = len(categories) == 1
            cells.append(
                {
                    "block": block,
                    "head": head,
                    "category": next(iter(categories)) if consistent else None,
                    "margin": float(np.mean([row["margin"] for row in rows])),
                    "labels": {
                        row["protocol"]: row["primary"] for row in rows
                    },
                }
            )
    payload = json.dumps(cells, ensure_ascii=False, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Protocol-consistent head distribution</title>
<style>
:root{{--bg:#f3f4f1;--ink:#202421;--line:#c9cec9;--accent:#0d7155}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px Arial,sans-serif;letter-spacing:0}}
header{{position:sticky;top:0;z-index:3;padding:12px 18px;background:#fff;border-bottom:1px solid var(--line);box-shadow:0 2px 8px rgba(25,35,29,.08)}}
h1{{font-size:21px;margin:0 0 5px}} p{{margin:4px 0;line-height:1.4}} a{{color:var(--accent);font-weight:700;text-decoration:none}}
main{{padding:14px 18px 30px}} section{{margin-bottom:20px}} h2{{font-size:17px;margin:0 0 7px}}
.filters{{display:flex;flex-wrap:wrap;gap:10px;margin-top:9px}} label{{display:flex;align-items:center;gap:5px;font-weight:700}}
.swatch{{width:13px;height:13px;border:1px solid #777}} .canvas-wrap{{position:relative;overflow-x:auto;background:#fff;border:1px solid var(--line);border-radius:5px;padding:8px}}
canvas{{display:block;min-width:900px;width:100%;height:auto}} #matrix{{max-width:1120px}} #stacked{{max-width:1120px}}
#tooltip{{position:fixed;z-index:8;display:none;pointer-events:none;background:#202421;color:#fff;padding:7px 9px;border-radius:4px;line-height:1.45;box-shadow:0 2px 8px rgba(0,0,0,.25)}}
.summary{{display:flex;flex-wrap:wrap;gap:7px;margin:8px 0}} .chip{{padding:4px 7px;border:1px solid var(--line);border-radius:4px;background:#fff;font-weight:700}}
</style>
</head>
<body>
<header>
<h1>338个跨协议一致 Head 的 Block 分布</h1>
<p>矩阵横轴是 Block 00–29，纵轴是 Head 00–23；彩色单元表示 Fixed A、Moving A、Moving B主类别一致，灰色表示不一致。</p>
<a href="grouped_by_role.html">返回按类别分组页面</a>
<div class="filters" id="filters"></div>
</header>
<main>
<div class="summary" id="summary"></div>
<section><h2>Block × Head 定位矩阵</h2><div class="canvas-wrap"><canvas id="matrix" width="930" height="545"></canvas></div></section>
<section><h2>每个 Block 的一致类别数量</h2><div class="canvas-wrap"><canvas id="stacked" width="930" height="330"></canvas></div></section>
</main>
<div id="tooltip"></div>
<script>
const CELLS={payload};
const CATEGORIES=["S","ST","T","P","C","G"];
const COLORS={{S:"#55a873",ST:"#3e9f91",T:"#4f88b5",P:"#d3ad42",C:"#c57f75",G:"#858b88"}};
const NAMES={{S:"帧内空间",ST:"帧内+相邻轨迹",T:"轨迹传播",P:"固定位置",C:"历史/context",G:"全局聚合"}};
const enabled=new Set(CATEGORIES),tooltip=document.getElementById("tooltip");
const filters=document.getElementById("filters");
for(const category of CATEGORIES){{const label=document.createElement("label");label.innerHTML=`<input type="checkbox" checked value="${{category}}"><span class="swatch" style="background:${{COLORS[category]}}"></span>${{category}} · ${{NAMES[category]}}`;label.querySelector("input").addEventListener("change",event=>{{event.target.checked?enabled.add(category):enabled.delete(category);draw();}});filters.appendChild(label);}}
const counts=Object.fromEntries(CATEGORIES.map(category=>[category,CELLS.filter(cell=>cell.category===category).length]));
document.getElementById("summary").innerHTML=CATEGORIES.map(category=>`<span class="chip"><span style="color:${{COLORS[category]}}">■</span> ${{category}} ${{counts[category]}}</span>`).join("")+`<span class="chip">一致 338 / 720</span>`;
function drawMatrix(){{
  const canvas=document.getElementById("matrix"),c=canvas.getContext("2d"),left=48,top=31,cw=28,ch=20;c.clearRect(0,0,canvas.width,canvas.height);c.fillStyle="#fff";c.fillRect(0,0,canvas.width,canvas.height);c.font="10px Arial";c.textAlign="center";c.fillStyle="#333";
  for(let block=0;block<30;block++)c.fillText(String(block).padStart(2,"0"),left+block*cw+cw/2,18);
  c.textAlign="right";for(let head=0;head<24;head++)c.fillText(String(head).padStart(2,"0"),left-7,top+head*ch+14);
  for(const cell of CELLS){{const x=left+cell.block*cw,y=top+cell.head*ch;c.fillStyle=cell.category&&enabled.has(cell.category)?COLORS[cell.category]:"#e2e5e3";c.fillRect(x+1,y+1,cw-2,ch-2);}}
  c.strokeStyle="#5e6661";c.strokeRect(left,top,30*cw,24*ch);canvas._layout={{left,top,cw,ch}};
}}
function drawStacked(){{
  const canvas=document.getElementById("stacked"),c=canvas.getContext("2d"),left=48,top=20,bottom=42,h=canvas.height-top-bottom,bw=24,gap=4;c.clearRect(0,0,canvas.width,canvas.height);c.fillStyle="#fff";c.fillRect(0,0,canvas.width,canvas.height);c.strokeStyle="#c9cec9";c.font="10px Arial";c.textAlign="right";
  for(let n=0;n<=24;n+=6){{const y=top+h-h*n/24;c.beginPath();c.moveTo(left,y);c.lineTo(left+30*(bw+gap),y);c.stroke();c.fillStyle="#444";c.fillText(String(n),left-7,y+3);}}
  c.textAlign="center";for(let block=0;block<30;block++){{let used=0;for(const category of CATEGORIES){{if(!enabled.has(category))continue;const value=CELLS.filter(cell=>cell.block===block&&cell.category===category).length;c.fillStyle=COLORS[category];const bh=h*value/24;c.fillRect(left+block*(bw+gap),top+h-used-bh,bw,bh);used+=bh;}}c.fillStyle="#333";c.fillText(String(block).padStart(2,"0"),left+block*(bw+gap)+bw/2,canvas.height-20);}}
}}
function draw(){{drawMatrix();drawStacked();}}
document.getElementById("matrix").addEventListener("mousemove",event=>{{const canvas=event.currentTarget,r=canvas.getBoundingClientRect(),sx=canvas.width/r.width,sy=canvas.height/r.height,{{left,top,cw,ch}}=canvas._layout,x=(event.clientX-r.left)*sx,y=(event.clientY-r.top)*sy,block=Math.floor((x-left)/cw),head=Math.floor((y-top)/ch);if(block<0||block>=30||head<0||head>=24){{tooltip.style.display="none";return;}}const cell=CELLS.find(value=>value.block===block&&value.head===head);tooltip.innerHTML=cell.category?`Block ${{String(block).padStart(2,"0")}} · Head ${{String(head).padStart(2,"0")}}<br>一致类别：${{cell.category}} · ${{NAMES[cell.category]}}<br>三协议平均分差：${{cell.margin.toFixed(3)}}`:`Block ${{String(block).padStart(2,"0")}} · Head ${{String(head).padStart(2,"0")}}<br>协议不一致：Fixed ${{cell.labels.fixed_A}} / Moving A ${{cell.labels.moving_A}} / Moving B ${{cell.labels.moving_B}}`;tooltip.style.display="block";tooltip.style.left=`${{event.clientX+14}}px`;tooltip.style.top=`${{event.clientY+14}}px`;}});document.getElementById("matrix").addEventListener("mouseleave",()=>tooltip.style.display="none");draw();
</script>
</body>
</html>"""


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
        previous_metrics = [
            _previous_trajectory_metrics(
                attention[track],
                query_coords=query_coords[track],
                valid_query_times=valid_query_times[track],
            )
            for track in range(2)
        ]
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
        roles.extend(
            _previous_trajectory_records(
                block=block,
                metrics_a=previous_metrics[0],
                metrics_b=previous_metrics[1],
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
        "object_role_protocol": {
            "roles": GALLERY_ROLE_LABELS,
            "st_role": {
                "name": GALLERY_ROLE_LABELS["ST"],
                "minimum_s_score": ST_MIN_SCORE,
                "minimum_t_adj_score": ST_MIN_SCORE,
                "minimum_balance": ST_MIN_BALANCE,
                "requires_t_adj_greater_than_t_long": True,
                "score": "harmonic_mean(S, T_adj)",
                "competition": "max(P, C, G, T_long)",
            },
        },
        "previous_trajectory_protocol": {
            "query_times": (
                "only query times where q-1, q, and q+1 are all valid"
            ),
            "comparison": (
                "same-object trajectory-token enrichment at K=q-1 versus K=q+1"
            ),
            "minimum_previous_mass": PREVIOUS_MIN_MASS,
            "minimum_previous_share": PREVIOUS_MIN_SHARE,
            "minimum_consistency": PREVIOUS_MIN_CONSISTENCY,
            "stable_both": "criteria pass independently for ball A and ball B",
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
    previous_rows = [
        row for row in roles if row["protocol"] == "previous_trajectory"
    ]
    with (output / "previous_trajectory_heads.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "block",
            "head",
            "scope",
            "stable_both",
            "confidence",
            "A_previous_mass",
            "A_previous_share",
            "A_consistency",
            "B_previous_mass",
            "B_previous_share",
            "B_consistency",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            previous_rows,
            key=lambda value: (
                -int(value["stable_both"]),
                -float(value["margin"]),
                int(value["block"]),
                int(value["head"]),
            ),
        ):
            features = row["features"]
            writer.writerow(
                {
                    "block": row["block"],
                    "head": row["head"],
                    "scope": row["scope"],
                    "stable_both": row["stable_both"],
                    "confidence": row["margin"],
                    "A_previous_mass": features["A_previous_mass"],
                    "A_previous_share": features["A_previous_share"],
                    "A_consistency": features["A_consistency"],
                    "B_previous_mass": features["B_previous_mass"],
                    "B_previous_share": features["B_previous_share"],
                    "B_consistency": features["B_consistency"],
                }
            )
    (output / "index.html").write_text(_page(metadata), encoding="utf-8")
    (output / "grouped_by_role.html").write_text(
        _grouped_page(metadata), encoding="utf-8"
    )
    (output / "consistent_heads_distribution.html").write_text(
        _consistent_distribution_page(metadata), encoding="utf-8"
    )
    print(f"[gallery] wrote {output / 'index.html'}")
    print(f"[gallery] wrote {output / 'grouped_by_role.html'}")
    print(f"[gallery] wrote {output / 'consistent_heads_distribution.html'}")


if __name__ == "__main__":
    main()
