#!/usr/bin/env python3
"""Three-case Baseline vs Top100-M1 guidance pilot for the port-8092 viewer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
RUNTIME_MANIFEST = ROOT / "stage4_runtime/stage4_manifest.json"
VIDEO_ROOT = ROOT / "training_free_top100_m1_guidance_v1"
CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_000331_w001",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
)
SEED = 47326
REGION = "object_A"
VARIANT = "single_object__object_A__m1_all_time__top100__pag0p5"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _samples() -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    for row in _read_json(RUNTIME_MANIFEST).get("samples") or []:
        if isinstance(row, dict):
            result[(str(row.get("case")), int(row.get("seed", -1)))] = row
    return result


def _variant_dir(case: str) -> Path:
    return VIDEO_ROOT / case / f"seed_{SEED:05d}" / VARIANT


def catalog() -> dict[str, Any]:
    samples = _samples()
    rows = []
    for case in CASES:
        sample = samples.get((case, SEED), {})
        directory = _variant_dir(case)
        manifest = _read_json(directory / "manifest.json")
        complete = _read_json(directory / "complete.json")
        error_path = directory / "error.txt"
        baseline = Path(str(sample.get("baseline_video") or ""))
        generated = directory / "generated.mp4"
        if complete and generated.is_file():
            state = "complete"
        elif error_path.is_file():
            state = "error"
        elif directory.is_dir():
            state = "running"
        else:
            state = "pending"
        audit = manifest.get("audit") if isinstance(manifest.get("audit"), dict) else {}
        relative = audit.get("relative_perturbation_l2_by_step") or {}
        finite_relative = [float(value) for value in relative.values()]
        rows.append(
            {
                "case": case,
                "seed": SEED,
                "caption": str(sample.get("caption") or ""),
                "region": REGION,
                "state": state,
                "baseline_ready": baseline.is_file(),
                "guidance_ready": generated.is_file(),
                "pag_scale": float(manifest.get("pag_scale", 0.5)),
                "cfg_scale": float(manifest.get("cfg_scale", 5.0)),
                "m1_time_scope": str(manifest.get("m1_time_scope") or "all_time"),
                "selected_head_count": int(manifest.get("selected_head_count") or 100),
                "mean_relative_prediction_delta": (
                    sum(finite_relative) / len(finite_relative) if finite_relative else None
                ),
                "error": (
                    error_path.read_text(encoding="utf-8", errors="replace")[-1200:]
                    if error_path.is_file()
                    else ""
                ),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "completed": sum(row["state"] == "complete" for row in rows),
        "total": len(rows),
        "equation": "ε = εu + 5(εc−εu) + 0.5(εc−εM1)",
        "m1": "Top100 heads: Y′[R] = Y[R] − A[R,R]V_R; all latent times; no renormalization",
    }


def asset(kind: str, case: str) -> Path | None:
    if case not in CASES:
        return None
    if kind == "baseline":
        sample = _samples().get((case, SEED), {})
        path = Path(str(sample.get("baseline_video") or ""))
    elif kind == "guidance":
        path = _variant_dir(case) / "generated.mp4"
    else:
        return None
    return path if path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top100 M1 Guidance Pilot</title><style>
:root{--paper:#e9edf0;--ink:#182128;--card:#f9fbfc;--line:#a8b4bd;--blue:#28667c;--orange:#d26a3a;--muted:#63717a;--dark:#172b34;--ok:#31735f}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#28667c16 1px,transparent 1px),linear-gradient(#28667c16 1px,transparent 1px),var(--paper);background-size:28px 28px;font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif}a{color:var(--blue)}header,main{width:min(1760px,calc(100% - 28px));margin:auto}header{padding:26px 0 12px}.eyebrow{margin-top:20px;color:var(--orange);font:800 12px ui-monospace,monospace;letter-spacing:.18em}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:8px 0;font-size:clamp(42px,7vw,92px);line-height:.87;letter-spacing:-.06em;text-transform:uppercase}.lead{max-width:1080px;font-size:16px;line-height:1.65}.formula{display:inline-block;padding:10px 14px;background:var(--dark);color:#eef8fb;font:13px ui-monospace,monospace}.status{margin:16px 0 0;display:flex;gap:8px;align-items:center;font:700 12px ui-monospace,monospace}.status i{width:10px;height:10px;border-radius:50%;background:var(--orange)}.case{margin:18px 0 34px;padding:16px;background:#f9fbfce8;border:1px solid var(--line);box-shadow:8px 8px 0 #28667c18}.case-head{display:grid;grid-template-columns:minmax(280px,.7fr) 1.3fr auto;gap:18px;align-items:start;border-bottom:2px solid var(--ink);padding-bottom:12px}.case-head h2{margin:4px 0;font-size:25px;overflow-wrap:anywhere}.case-no{color:var(--orange);font:800 12px ui-monospace,monospace}.caption{line-height:1.55;color:var(--muted)}.badge{padding:8px 11px;border:1px solid currentColor;font:800 11px ui-monospace,monospace;text-transform:uppercase}.badge.complete{color:var(--ok)}.badge.running{color:var(--orange)}.videos{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:14px}.video{position:relative;padding:10px;border:1px solid var(--line);background:var(--card)}.video.guidance{border-top:6px solid var(--orange)}.video.baseline{border-top:6px solid var(--blue)}.video h3{display:flex;justify-content:space-between;gap:10px;margin:0 0 8px;font-size:16px}.video h3 span{color:var(--muted);font:700 10px ui-monospace,monospace}.video video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#0c1114}.pending{display:grid;place-items:center;width:100%;aspect-ratio:16/9;background:repeating-linear-gradient(-45deg,#dce3e7,#dce3e7 12px,#edf1f3 12px,#edf1f3 24px);color:var(--muted);font:800 13px ui-monospace,monospace}.facts{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}.facts span{padding:5px 8px;background:#e5ebee;font:11px ui-monospace,monospace}.error{white-space:pre-wrap;color:#9d3426;font:11px ui-monospace,monospace}.actions{margin-top:14px;display:flex;gap:8px}.actions button{padding:9px 12px;border:1px solid var(--line);background:white;font-weight:800;cursor:pointer}.footer{padding:5px 0 50px;color:var(--muted);font-size:11px}@media(max-width:800px){.case-head,.videos{grid-template-columns:1fr}header,main{width:calc(100% - 12px)}}
</style></head><body><header><a href="/">← 返回 8092 总入口</a> · <a href="/object-query-information-flow-stage4?v=1">Stage 4 消融矩阵</a><div class="eyebrow">TRAINING-FREE / CLEAN − PERTURBED / THREE-CASE PILOT</div><h1>把 M1 差分<br>变成方向</h1><p class="lead">同一 case、同一 seed、同一首帧。左侧是未干预 Baseline，右侧在每个去噪步额外计算一次 latest3350 Top100 M1 扰动预测，并把 clean−M1 差分以 λ=0.5 加回 CFG。这里比较的是生成结果，不把像素变化自动解释为物理改善。</p><div class="formula">ε = εu + 5(εc−εu) + 0.5(εc−εM1)</div><div id="status" class="status"><i></i><span>读取中…</span></div></header><main id="cases"></main><div class="footer">视频按接近视口时加载；页面每 20 秒检查一次结果。M1=R K/V→R Query，All-time，Top100，post-softmax A·V contribution subtraction，无重新归一化。</div><script>
const api='/api/top100-m1-guidance-pilot',esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const media=(kind,c)=>`${api}/asset?${new URLSearchParams({kind,case:c})}`;
function video(kind,row){const ready=kind==='baseline'?row.baseline_ready:row.guidance_ready,label=kind==='baseline'?'Baseline · no intervention':'Top100 M1 Guidance · λ=0.5';return `<article class="video ${kind}"><h3>${label}<span>${kind==='baseline'?'shared reference':'clean − M1'}</span></h3>${ready?`<video controls muted loop playsinline preload="none" data-src="${esc(media(kind,row.case))}"></video>`:`<div class="pending">${kind==='baseline'?'BASELINE MISSING':row.state.toUpperCase()}</div>`}<div class="facts"><span>seed ${row.seed}</span><span>${esc(row.region)}</span><span>M1 ${esc(row.m1_time_scope)}</span><span>Top ${row.selected_head_count}</span>${row.mean_relative_prediction_delta==null?'':`<span>mean ‖Δε‖/‖εc‖ ${row.mean_relative_prediction_delta.toFixed(5)}</span>`}</div>${row.error?`<details><summary>错误日志</summary><pre class="error">${esc(row.error)}</pre></details>`:''}</article>`}
function lazy(){const io=new IntersectionObserver(xs=>xs.forEach(x=>{if(x.isIntersecting){const v=x.target;v.src=v.dataset.src;v.load();io.unobserve(v)}}),{rootMargin:'500px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function render(data){document.querySelector('#status span').textContent=`${data.completed}/${data.total} guidance 完成 · ${new Date(data.generated_at_utc).toLocaleTimeString()}`;document.querySelector('#status i').style.background=data.completed===data.total?'var(--ok)':'var(--orange)';document.querySelector('#cases').innerHTML=data.rows.map((r,i)=>`<section class="case"><div class="case-head"><div><div class="case-no">CASE ${String(i+1).padStart(2,'0')} / SEED ${r.seed}</div><h2>${esc(r.case)}</h2></div><div class="caption">${esc(r.caption||'同 seed Baseline 与 Top100-M1 guidance 对照')}</div><span class="badge ${esc(r.state)}">${esc(r.state)}</span></div><div class="videos">${video('baseline',r)}${video('guidance',r)}</div><div class="actions"><button data-replay="${i}">同步重播这一组</button></div></section>`).join('');document.querySelectorAll('[data-replay]').forEach(b=>b.onclick=()=>b.closest('.case').querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));lazy()}
async function load(){const d=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());render(d)}load().catch(e=>document.querySelector('#status span').textContent=`读取失败：${e}`);setInterval(()=>load().catch(()=>{}),20000);
</script></body></html>'''
