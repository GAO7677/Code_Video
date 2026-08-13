#!/usr/bin/env python3
"""Three-case Baseline vs controlled Top100-M1/M2/M3 guidance comparison."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
RUNTIME_MANIFEST = ROOT / "stage4_runtime/stage4_manifest.json"
GRID_MANIFEST = ROOT / "training_free_top100_m23_guidance_v1/guidance_grid_manifest.json"
VIDEO_ROOTS = {
    "m1": ROOT / "training_free_top100_m1_guidance_v1",
    "m2": ROOT / "training_free_top100_m23_guidance_v1",
    "m3": ROOT / "training_free_top100_m23_guidance_v1",
}
CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_000331_w001",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
)
SEEDS = (47326, 42)
PAG_SCALES = (0.5, 1.0)
REGION = "object_A"
FLOWS = {
    "m1": {"label": "M1 · R→R", "mask_mode": "self_only"},
    "m2": {"label": "M2 · C→R", "mask_mode": "incoming_only"},
    "m3": {"label": "M3 · R→C", "mask_mode": "outgoing_only"},
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _samples() -> dict[tuple[str, int], dict[str, Any]]:
    result = {}
    manifest_path = GRID_MANIFEST if GRID_MANIFEST.is_file() else RUNTIME_MANIFEST
    for row in _read_json(manifest_path).get("samples") or []:
        if isinstance(row, dict):
            result[(str(row.get("case")), int(row.get("seed", -1)))] = row
    return result


def _scale_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _variant_dir(case: str, flow: str, seed: int, pag_scale: float) -> Path:
    variant = (
        f"single_object__object_A__{flow}_all_time__top100"
        f"__pag{_scale_tag(pag_scale)}"
    )
    if flow == "m1" and seed == 47326 and pag_scale == 0.5:
        root = VIDEO_ROOTS["m1"]
    else:
        root = ROOT / "training_free_top100_m23_guidance_v1"
    return root / case / f"seed_{seed:05d}" / variant


def _flow_record(case: str, flow: str, seed: int, pag_scale: float) -> dict[str, Any]:
    directory = _variant_dir(case, flow, seed, pag_scale)
    manifest = _read_json(directory / "manifest.json")
    complete = _read_json(directory / "complete.json")
    error_path = directory / "error.txt"
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
    return {
        "flow": flow,
        "label": FLOWS[flow]["label"],
        "mask_mode": str(manifest.get("mask_mode") or FLOWS[flow]["mask_mode"]),
        "state": state,
        "ready": generated.is_file(),
        "pag_scale": float(manifest.get("pag_scale", pag_scale)),
        "cfg_scale": float(manifest.get("cfg_scale", 5.0)),
        "time_scope": str(
            manifest.get("time_scope") or manifest.get("m1_time_scope") or "all_time"
        ),
        "selected_head_count": int(manifest.get("selected_head_count") or 100),
        "modified_head_events": int(audit.get("modified_head_events") or 0),
        "mean_relative_prediction_delta": (
            sum(finite_relative) / len(finite_relative) if finite_relative else None
        ),
        "error": (
            error_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            if error_path.is_file()
            else ""
        ),
    }


def catalog() -> dict[str, Any]:
    samples = _samples()
    rows = []
    for case in CASES:
        for seed in SEEDS:
            sample = samples.get((case, seed), {})
            baseline = Path(str(sample.get("baseline_video") or ""))
            for pag_scale in PAG_SCALES:
                guidance = {
                    flow: _flow_record(case, flow, seed, pag_scale) for flow in FLOWS
                }
                rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "pag_scale": pag_scale,
                        "caption": str(sample.get("caption") or ""),
                        "region": REGION,
                        "baseline_ready": baseline.is_file(),
                        "guidance": guidance,
                    }
                )
    completed = sum(
        record["state"] == "complete"
        for row in rows
        for record in row["guidance"].values()
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "completed": completed,
        "total": len(CASES) * len(SEEDS) * len(PAG_SCALES) * len(FLOWS),
        "cases": len(CASES),
        "seeds": list(SEEDS),
        "pag_scales": list(PAG_SCALES),
        "equation": "ε = εu + 5(εc−εu) + λ(εc−εflow)",
        "controls": (
            "within each case x seed x lambda: same first frame, prompt, latest3350 "
            "Top100, CFG=5, 40 steps, all-time, post-softmax A@V subtraction"
        ),
    }


def asset(
    kind: str, case: str, seed: int = 47326, pag_scale: float = 0.5
) -> Path | None:
    if case not in CASES:
        return None
    if seed not in SEEDS or pag_scale not in PAG_SCALES:
        return None
    if kind == "baseline":
        sample = _samples().get((case, seed), {})
        path = Path(str(sample.get("baseline_video") or ""))
    elif kind in FLOWS:
        path = _variant_dir(case, kind, seed, pag_scale) / "generated.mp4"
    else:
        return None
    return path if path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Top100 M1/M2/M3 Guidance Pilot</title><style>
:root{--paper:#e9edf0;--ink:#182128;--card:#f9fbfc;--line:#a8b4bd;--blue:#28667c;--orange:#d26a3a;--muted:#63717a;--dark:#172b34;--ok:#31735f}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#28667c16 1px,transparent 1px),linear-gradient(#28667c16 1px,transparent 1px),var(--paper);background-size:28px 28px;font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif}a{color:var(--blue)}header,main{width:min(1900px,calc(100% - 28px));margin:auto}header{padding:26px 0 12px}.eyebrow{margin-top:20px;color:var(--orange);font:800 12px ui-monospace,monospace;letter-spacing:.18em}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:8px 0;font-size:clamp(42px,7vw,92px);line-height:.87;letter-spacing:-.06em;text-transform:uppercase}.lead{max-width:1280px;font-size:16px;line-height:1.65}.formula{display:inline-block;padding:10px 14px;background:var(--dark);color:#eef8fb;font:13px ui-monospace,monospace}.status{margin:16px 0 0;display:flex;gap:8px;align-items:center;font:700 12px ui-monospace,monospace}.status i{width:10px;height:10px;border-radius:50%;background:var(--orange)}.case{margin:18px 0 34px;padding:16px;background:#f9fbfce8;border:1px solid var(--line);box-shadow:8px 8px 0 #28667c18}.case-head{display:grid;grid-template-columns:minmax(280px,.7fr) 1.3fr;gap:18px;align-items:start;border-bottom:2px solid var(--ink);padding-bottom:12px}.case-head h2{margin:4px 0;font-size:25px;overflow-wrap:anywhere}.case-no{color:var(--orange);font:800 12px ui-monospace,monospace}.caption{line-height:1.55;color:var(--muted)}.videos{display:grid;grid-template-columns:repeat(4,minmax(280px,1fr));gap:10px;margin-top:14px;overflow-x:auto}.video{position:relative;padding:10px;border:1px solid var(--line);background:var(--card)}.video.m1{border-top:6px solid #d26a3a}.video.m2{border-top:6px solid #8367a8}.video.m3{border-top:6px solid #31735f}.video.baseline{border-top:6px solid var(--blue)}.video h3{display:flex;justify-content:space-between;gap:10px;margin:0 0 8px;font-size:16px}.video h3 span{color:var(--muted);font:700 10px ui-monospace,monospace}.video video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#0c1114}.pending{display:grid;place-items:center;width:100%;aspect-ratio:16/9;background:repeating-linear-gradient(-45deg,#dce3e7,#dce3e7 12px,#edf1f3 12px,#edf1f3 24px);color:var(--muted);font:800 13px ui-monospace,monospace}.facts{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}.facts span{padding:5px 8px;background:#e5ebee;font:11px ui-monospace,monospace}.error{white-space:pre-wrap;color:#9d3426;font:11px ui-monospace,monospace}.actions{margin-top:14px;display:flex;gap:8px}.actions button{padding:9px 12px;border:1px solid var(--line);background:white;font-weight:800;cursor:pointer}.footer{padding:5px 0 50px;color:var(--muted);font-size:11px}@media(max-width:1100px){.videos{grid-template-columns:repeat(2,minmax(280px,1fr))}}@media(max-width:650px){.case-head,.videos{grid-template-columns:1fr}header,main{width:calc(100% - 12px)}}
</style></head><body><header><a href="/">← 返回 8092 总入口</a> · <a href="/object-query-information-flow-stage4?v=1">Stage 4 消融矩阵</a> · <a href="/top100-m1-token-communication?v=1">查看 M1 Token 通信位置</a><div class="eyebrow">TRAINING-FREE / 3 CASES × 2 SEEDS × 2 LAMBDAS × 3 FLOWS</div><h1>同一把尺，<br>比较种子与强度</h1><p class="lead">每个 case×seed×λ 都按 Baseline、M1、M2、M3 四列展示。固定首帧、prompt、latest3350 Top100、CFG=5、40 步和 All-time；只改变 seed、λ 或被扰动的信息流。M1 删除 R→R，M2 删除 C→R，M3 删除 R→C；均为 post-softmax A·V contribution subtraction 且不重新归一化。</p><div class="formula">εguided(Mi,λ) = εu + 5(εc−εu) + λ(εc−εMi), &nbsp; λ∈{0.5,1}</div><div id="status" class="status"><i></i><span>读取中…</span></div></header><main id="cases"></main><div class="footer">同一 seed 的 λ=0.5/1 共用同一 Baseline；视频接近视口时才加载，页面每 20 秒自动检查新结果。视觉变化本身不等于物理质量改善。</div><script>
const api='/api/top100-m1-guidance-pilot',esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const media=(kind,row)=>`${api}/asset?${new URLSearchParams({kind,case:row.case,seed:row.seed,pag_scale:row.pag_scale})}`;
function baseline(row){return `<article class="video baseline"><h3>Baseline<span>shared for both λ</span></h3>${row.baseline_ready?`<video controls muted loop playsinline preload="none" data-src="${esc(media('baseline',row))}"></video>`:'<div class="pending">BASELINE MISSING</div>'}<div class="facts"><span>seed ${row.seed}</span><span>λ-independent</span><span>no intervention</span></div></article>`}
function guidance(flow,row){const r=row.guidance[flow];return `<article class="video ${flow}"><h3>${esc(r.label)} Guidance<span>clean − ${flow.toUpperCase()}</span></h3>${r.ready?`<video controls muted loop playsinline preload="none" data-src="${esc(media(flow,row))}"></video>`:`<div class="pending">${esc(r.state.toUpperCase())}</div>`}<div class="facts"><span>seed ${row.seed}</span><span>λ ${row.pag_scale}</span><span>${esc(row.region)}</span><span>${esc(r.time_scope)}</span><span>Top ${r.selected_head_count}</span><span>${esc(r.mask_mode)}</span>${r.modified_head_events?`<span>${r.modified_head_events} head-events</span>`:''}${r.mean_relative_prediction_delta==null?'':`<span>mean ‖Δε‖/‖εc‖ ${r.mean_relative_prediction_delta.toFixed(5)}</span>`}</div>${r.error?`<details><summary>错误日志</summary><pre class="error">${esc(r.error)}</pre></details>`:''}</article>`}
function lazy(){const io=new IntersectionObserver(xs=>xs.forEach(x=>{if(x.isIntersecting){const v=x.target;v.src=v.dataset.src;v.load();io.unobserve(v)}}),{rootMargin:'500px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function render(data){document.querySelector('#status span').textContent=`${data.completed}/${data.total} guidance 完成 · ${data.cases} cases · seeds ${data.seeds.join('/')} · λ ${data.pag_scales.join('/')} · ${new Date(data.generated_at_utc).toLocaleTimeString()}`;document.querySelector('#status i').style.background=data.completed===data.total?'var(--ok)':'var(--orange)';document.querySelector('#cases').innerHTML=data.rows.map((r,i)=>`<section class="case"><div class="case-head"><div><div class="case-no">CONDITION ${String(i+1).padStart(2,'0')} / SEED ${r.seed} / λ ${r.pag_scale}</div><h2>${esc(r.case)}</h2></div><div class="caption">${esc(r.caption||'同一 case×seed×λ，仅改变 M1/M2/M3 扰动信息流')}</div></div><div class="videos">${baseline(r)}${guidance('m1',r)}${guidance('m2',r)}${guidance('m3',r)}</div><div class="actions"><button data-replay="${i}">同步重播这一组</button></div></section>`).join('');document.querySelectorAll('[data-replay]').forEach(b=>b.onclick=()=>b.closest('.case').querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));lazy()}
async function load(){const d=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());render(d)}load().catch(e=>document.querySelector('#status span').textContent=`读取失败：${e}`);setInterval(()=>load().catch(()=>{}),20000);
</script></body></html>'''
