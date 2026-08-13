#!/usr/bin/env python3
"""Dedicated dashboard for the Training-Free M1 control experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
MATRIX_PATH = CODE_DIR / "tf1_matrix.json"
ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_control_v1"
)
EXPERIMENT_ROOT = ROOT.parent
SOFT_ROOT = ROOT / "soft_scaling"
CONTRAST_ROOT = ROOT / "contrast_raw"
POSITIVE_ROOTS = (
    EXPERIMENT_ROOT / "training_free_top100_m1_guidance_v1",
    EXPERIMENT_ROOT / "training_free_top100_m23_guidance_v1",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _matrix() -> dict[str, Any]:
    return _read_json(MATRIX_PATH)


def _samples(matrix: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    source = _read_json(Path(str(matrix.get("source_manifest") or "")))
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in source.get("samples") or []:
        if isinstance(row, dict):
            result[(str(row.get("case")), int(row.get("seed", -1)))] = row
    return result


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _soft_dir(case: str, seed: int, value: float) -> Path:
    variant = (
        "single_object__object_A__m1_all_time__top100"
        f"__alpha_{_tag(value)}"
    )
    return SOFT_ROOT / case / f"seed_{seed:05d}" / variant


def _new_contrast_dir(case: str, seed: int, value: float) -> Path:
    variant = (
        "single_object__object_A__m1_all_time__top100"
        f"__pag{_tag(value)}"
    )
    return CONTRAST_ROOT / case / f"seed_{seed:05d}" / variant


def _positive_contrast_dir(case: str, seed: int, value: float) -> Path | None:
    variant = (
        "single_object__object_A__m1_all_time__top100"
        f"__pag{_tag(value)}"
    )
    for root in POSITIVE_ROOTS:
        candidate = root / case / f"seed_{seed:05d}" / variant
        if (candidate / "generated.mp4").is_file():
            return candidate
    return None


def _state(directory: Path) -> str:
    if (directory / "complete.json").is_file() and (
        directory / "generated.mp4"
    ).is_file():
        return "complete"
    if (directory / "error.txt").is_file():
        return "error"
    if directory.is_dir():
        return "running"
    return "pending"


def _record(
    family: str,
    case: str,
    seed: int,
    value: float,
    baseline: Path,
) -> dict[str, Any]:
    if value == 0:
        directory = baseline.parent
        manifest = _read_json(directory / "manifest.json")
        state = "complete" if baseline.is_file() else "pending"
        ready = baseline.is_file()
        reused = True
    elif family == "soft":
        directory = _soft_dir(case, seed, value)
        manifest = _read_json(directory / "manifest.json")
        state = _state(directory)
        ready = (directory / "generated.mp4").is_file()
        reused = False
    elif value > 0:
        found = _positive_contrast_dir(case, seed, value)
        directory = found or (
            POSITIVE_ROOTS[-1]
            / case
            / f"seed_{seed:05d}"
            / (
                "single_object__object_A__m1_all_time__top100"
                f"__pag{_tag(value)}"
            )
        )
        manifest = _read_json(directory / "manifest.json")
        state = _state(directory)
        ready = (directory / "generated.mp4").is_file()
        reused = True
    else:
        directory = _new_contrast_dir(case, seed, value)
        manifest = _read_json(directory / "manifest.json")
        state = _state(directory)
        ready = (directory / "generated.mp4").is_file()
        reused = False

    audit = manifest.get("audit") if isinstance(manifest.get("audit"), dict) else {}
    relative = audit.get("relative_perturbation_l2_by_step") or {}
    finite_relative = [float(item) for item in relative.values()]
    error_path = directory / "error.txt"
    return {
        "family": family,
        "value": float(value),
        "state": state,
        "ready": ready,
        "reused": reused,
        "directory": str(directory),
        "modified_head_events": int(audit.get("modified_head_events") or 0),
        "selected_head_count": int(manifest.get("selected_head_count") or 100),
        "mean_relative_prediction_delta": (
            sum(finite_relative) / len(finite_relative) if finite_relative else None
        ),
        "decomposition_max_abs_error": audit.get("decomposition_max_abs_error"),
        "noop_mismatch_count": audit.get("noop_mismatch_count"),
        "error": (
            error_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            if error_path.is_file()
            else ""
        ),
    }


def catalog() -> dict[str, Any]:
    matrix = _matrix()
    samples = _samples(matrix)
    cases = [str(value) for value in matrix.get("cases") or []]
    seeds = [int(value) for value in matrix.get("seeds") or []]
    alphas = [float(value) for value in matrix.get("soft_scaling", {}).get("alphas") or []]
    lambdas = [
        float(value)
        for value in matrix.get("contrast_guidance", {}).get("lambdas") or []
    ]
    rows = []
    for case in cases:
        for seed in seeds:
            sample = samples.get((case, seed), {})
            baseline = Path(str(sample.get("baseline_video") or ""))
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "caption": str(sample.get("caption") or ""),
                    "baseline_ready": baseline.is_file(),
                    "soft": [
                        _record("soft", case, seed, value, baseline) for value in alphas
                    ],
                    "contrast": [
                        _record("contrast", case, seed, value, baseline)
                        for value in lambdas
                    ],
                }
            )
    intervention_records = [
        record
        for row in rows
        for family in ("soft", "contrast")
        for record in row[family]
        if record["value"] != 0
    ]
    tf0_pass = _read_json(ROOT / "tf0/PASS.json")
    tf0_fail = _read_json(ROOT / "tf0/FAIL.json")
    if tf0_pass:
        tf0_state = "pass"
    elif tf0_fail:
        tf0_state = "fail"
    elif any(record["state"] in {"running", "complete"} for record in intervention_records):
        tf0_state = "running"
    else:
        tf0_state = "pending"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": matrix.get("experiment_id"),
        "rows": rows,
        "cases": cases,
        "seeds": seeds,
        "alphas": alphas,
        "lambdas": lambdas,
        "tf0": {
            "state": tf0_state,
            "report": tf0_pass or tf0_fail,
        },
        "progress": {
            "interventions_complete": sum(
                record["state"] == "complete" for record in intervention_records
            ),
            "interventions_total": len(intervention_records),
            "baselines_complete": sum(row["baseline_ready"] for row in rows),
            "baselines_total": len(rows),
        },
        "controls": matrix.get("sampling") or {},
        "ranking_sha256": matrix.get("head_ranking_sha256"),
    }


def asset(family: str, case: str, seed: int, value: float) -> Path | None:
    matrix = _matrix()
    if case not in matrix.get("cases", []) or seed not in matrix.get("seeds", []):
        return None
    samples = _samples(matrix)
    sample = samples.get((case, seed), {})
    baseline = Path(str(sample.get("baseline_video") or ""))
    allowed = {
        "soft": [float(item) for item in matrix.get("soft_scaling", {}).get("alphas") or []],
        "contrast": [
            float(item)
            for item in matrix.get("contrast_guidance", {}).get("lambdas") or []
        ],
    }
    if family not in allowed or value not in allowed[family]:
        return None
    if value == 0:
        path = baseline
    elif family == "soft":
        path = _soft_dir(case, seed, value) / "generated.mp4"
    elif value > 0:
        directory = _positive_contrast_dir(case, seed, value)
        path = directory / "generated.mp4" if directory else Path("")
    else:
        path = _new_contrast_dir(case, seed, value) / "generated.mp4"
    return path if path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training-Free M1 Control</title><style>
:root{--ice:#e8f0f2;--paper:#f8fbfc;--ink:#102733;--line:#9babb3;--negative:#b64255;--positive:#147d84;--zero:#c39a25;--muted:#5e7079;--navy:#102f40;--ok:#267158}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#102f400c 1px,transparent 1px),linear-gradient(#102f400c 1px,transparent 1px),var(--ice);background-size:24px 24px;font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif}a{color:var(--positive);font-weight:700}header,main,footer{width:min(1880px,calc(100% - 28px));margin:auto}header{padding:24px 0 18px}.eyebrow{margin-top:22px;color:var(--negative);font:800 12px ui-monospace,monospace;letter-spacing:.18em}.hero{display:grid;grid-template-columns:minmax(360px,.8fr) minmax(520px,1.2fr);gap:34px;align-items:end}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:8px 0 0;font-size:clamp(42px,7vw,94px);line-height:.86;letter-spacing:-.065em}.lead{margin:0;font-size:17px;line-height:1.68;max-width:960px}.equation{margin-top:18px;padding:12px 15px;background:var(--navy);color:#eef8fa;font:13px/1.7 ui-monospace,monospace}.rail{margin-top:22px;display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);background:var(--paper)}.rail span{position:relative;padding:13px 8px 11px;text-align:center;font:800 11px ui-monospace,monospace;border-right:1px solid var(--line)}.rail span:last-child{border:0}.rail span:nth-child(-n+2){color:var(--negative)}.rail span:nth-child(3){color:#80640f}.rail span:nth-child(n+4){color:var(--positive)}.rail span:before{content:"";position:absolute;left:50%;top:-6px;width:11px;height:11px;transform:translateX(-50%) rotate(45deg);background:currentColor}.toolbar{position:sticky;top:0;z-index:5;display:flex;gap:10px;align-items:end;flex-wrap:wrap;padding:12px;margin-top:18px;background:#e8f0f2ee;border:1px solid var(--line);backdrop-filter:blur(8px)}label{display:grid;gap:4px;color:var(--muted);font:800 10px ui-monospace,monospace;letter-spacing:.08em}select,button{min-height:36px;border:1px solid var(--line);background:white;color:var(--ink);padding:7px 10px;font-weight:800}button{cursor:pointer}.status{margin-left:auto;font:800 11px ui-monospace,monospace}.gate{display:grid;grid-template-columns:200px 1fr;gap:18px;align-items:center;margin:18px 0;padding:15px;border:1px solid var(--line);background:var(--paper)}.gate strong{font:900 28px "Arial Black",sans-serif}.gate.pass strong{color:var(--ok)}.gate.fail strong{color:var(--negative)}.gate.running strong{color:var(--zero)}.gate p{margin:0;line-height:1.55;color:var(--muted)}.method{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:18px 0}.method article{padding:16px;background:#f8fbfce8;border:1px solid var(--line)}.method h2{margin:0 0 7px;font-size:22px}.method p{margin:5px 0;color:var(--muted);line-height:1.55}.method code{font:12px ui-monospace,monospace;color:var(--ink)}.experiment{margin:20px 0 34px}.experiment-head{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:end;border-bottom:3px solid var(--ink);padding-bottom:9px}.experiment-head h2{margin:0;font-size:29px}.experiment-head p{margin:0;text-align:right;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(5,minmax(250px,1fr));gap:9px;margin-top:11px;overflow-x:auto;padding-bottom:5px}.card{min-width:250px;padding:9px;background:var(--paper);border:1px solid var(--line);border-top:6px solid var(--zero)}.card.negative{border-top-color:var(--negative)}.card.positive{border-top-color:var(--positive)}.card h3{display:flex;justify-content:space-between;gap:8px;margin:1px 0 8px;font-size:16px}.badge{font:800 10px ui-monospace,monospace;color:var(--muted)}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#081217}.pending{display:grid;place-items:center;width:100%;aspect-ratio:16/9;background:repeating-linear-gradient(-45deg,#dbe4e7,#dbe4e7 12px,#edf2f4 12px,#edf2f4 24px);font:900 12px ui-monospace,monospace;color:var(--muted)}.facts{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.facts span{padding:4px 6px;background:#e8eef0;font:10px ui-monospace,monospace}.card details{margin-top:7px;font:11px ui-monospace,monospace}.error{white-space:pre-wrap;color:#9d273b}.empty{padding:40px;text-align:center;border:1px dashed var(--line);color:var(--muted)}footer{padding:0 0 46px;color:var(--muted);font-size:11px}@media(max-width:950px){.hero,.method,.experiment-head{grid-template-columns:1fr}.experiment-head p{text-align:left}.status{width:100%;margin-left:0}}@media(max-width:620px){header,main,footer{width:calc(100% - 12px)}.gate{grid-template-columns:1fr}.cards{grid-template-columns:repeat(5,84vw)}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><header><a href="/">← 返回 8092 总入口</a> · <a href="/top100-m1-guidance-pilot?v=3">旧 M1/M2/M3 Guidance</a> · <a href="/top100-m1-token-communication?v=1">M1 Token 通信</a><div class="eyebrow">LATEST3350 / TOP100 / OBJECT A / ALL-TIME</div><div class="hero"><h1>一条 R→R 通道，<br>两种控制方式。</h1><p class="lead">固定同一首帧、prompt、seed、对象 tube、Top100 heads、CFG=5 和 40 个去噪步。第一行直接缩放对象内部的 <b>A[R,R]V[R]</b>；第二行使用 clean 与 M1-knockout prediction 的差作为 guidance。负值靠近 knockout，正值增强或远离 knockout。</p></div><div class="equation">Soft: Y_R(α) = Y_R + α·A[R,R]V[R]<br>Contrast: εguided = εu + 5(εc−εu) + λ(εc−εc,M1)</div><div class="rail"><span>−1<br>KNOCKOUT</span><span>−0.5<br>WEAKEN / APPROACH</span><span>0<br>BASELINE</span><span>+0.5<br>ENHANCE / REPEL</span><span>+1<br>DOUBLE / REPEL</span></div><div class="toolbar"><label>CASE<select id="case"></select></label><label>SEED<select id="seed"></select></label><button id="refresh">刷新进度</button><button id="replay">同步重播</button><span id="status" class="status">读取中…</span></div></header><main><section id="gate" class="gate running"><strong>TF-0 …</strong><p>读取硬门控状态。</p></section><section class="method"><article><h2>A · M1 Soft Scaling</h2><p><code>conditional + unconditional</code> 两个 CFG 分支同时修改。α=−1 必须复现 Stage-3 M1 knockout；α=0 必须是数值 no-op。</p></article><article><h2>B · M1 Contrast Guidance</h2><p>仅额外计算 conditional M1-knockout prediction。λ=−1 不等于完整 knockout，因为差分项没有乘 CFG=5。</p></article></section><div id="experiments"><div class="empty">读取实验矩阵…</div></div></main><footer>只加载当前 case×seed 的十个槽位，且 MP4 接近视口才发起请求。所有变化均是 vs 同 seed Baseline 的干预效应，不自动等同于物理质量改善。</footer><script>
const api='/api/training-free-m1-control',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null;
const key=r=>`${r.case}::${r.seed}`,klass=v=>v<0?'negative':v>0?'positive':'zero',fmt=v=>(v>0?'+':'')+Number(v).toFixed(v%1?1:0),media=(family,row,r)=>`${api}/asset?${new URLSearchParams({family,case:row.case,seed:row.seed,value:r.value})}`;
function options(node,items,label){const old=node.value;node.innerHTML=items.map(x=>`<option value="${esc(x)}">${esc(label?label(x):x)}</option>`).join('');if([...node.options].some(o=>o.value===old))node.value=old}
function lazy(){const io=new IntersectionObserver(xs=>xs.forEach(x=>{if(x.isIntersecting){const v=x.target;v.src=v.dataset.src;v.load();io.unobserve(v)}}),{rootMargin:'500px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function card(family,row,r){const label=family==='soft'?`α ${fmt(r.value)}`:`λ ${fmt(r.value)}`,meaning=r.value===0?'共同 Baseline':family==='soft'?(r.value===-1?'完整 M1 knockout':r.value<0?'削弱 M1 contribution':'增强 M1 contribution'):(r.value<0?'向 M1 knockout 靠近':'远离 M1 knockout');return `<article class="card ${klass(r.value)}"><h3>${esc(label)}<span class="badge">${esc(r.state.toUpperCase())}</span></h3>${r.ready?`<video controls muted loop playsinline preload="none" data-src="${esc(media(family,row,r))}"></video>`:`<div class="pending">${esc(r.state.toUpperCase())}</div>`}<div class="facts"><span>${esc(meaning)}</span><span>${r.reused?'reused':'new run'}</span><span>Top ${r.selected_head_count}</span>${r.modified_head_events?`<span>${r.modified_head_events} events</span>`:''}${r.mean_relative_prediction_delta==null?'':`<span>mean ‖Δε‖/‖εc‖ ${Number(r.mean_relative_prediction_delta).toFixed(5)}</span>`}</div>${r.error?`<details><summary>错误日志</summary><pre class="error">${esc(r.error)}</pre></details>`:''}</article>`}
function section(title,note,family,row){return `<section class="experiment"><div class="experiment-head"><h2>${esc(title)}</h2><p>${esc(note)}</p></div><div class="cards">${row[family].map(r=>card(family,row,r)).join('')}</div></section>`}
function render(){const row=data.rows.find(r=>r.case===$('case').value&&String(r.seed)===$('seed').value);if(!row){$('experiments').innerHTML='<div class="empty">该 case×seed 不在冻结矩阵中</div>';return}const g=data.tf0.state,gate=$('gate');gate.className=`gate ${g}`;gate.innerHTML=g==='pass'?'<strong>TF-0 PASS</strong><p>α=0 no-op、α=−1 knockout 等价、Top100/13-anchor/dose 覆盖均已通过硬门控；TF-1 可以继续。</p>':g==='fail'?'<strong>TF-0 FAIL</strong><p>硬门控失败，后续队列应停止。展开运行日志定位具体不一致。</p>':'<strong>TF-0 RUNNING</strong><p>正在验证 α=0 与 α=−1；通过前不把后续生成解释为正式 TF-1 结果。</p>';$('status').textContent=`${data.progress.interventions_complete}/${data.progress.interventions_total} interventions · Baseline ${data.progress.baselines_complete}/${data.progress.baselines_total} · ${new Date(data.generated_at_utc).toLocaleTimeString()}`;$('experiments').innerHTML=section('A · Soft Scaling','两条 CFG 分支同时缩放；α=−1 是完整 knockout','soft',row)+section('B · Conditional Contrast Guidance','raw prediction difference；λ=−1 不是完整 knockout','contrast',row);lazy()}
async function load(){const oldCase=$('case').value,oldSeed=$('seed').value;data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());options($('case'),data.cases);if(oldCase&&data.cases.includes(oldCase))$('case').value=oldCase;options($('seed'),data.seeds.map(String));if(oldSeed&&data.seeds.map(String).includes(oldSeed))$('seed').value=oldSeed;if(q.get('case')&&data.cases.includes(q.get('case')))$('case').value=q.get('case');if(q.get('seed')&&data.seeds.map(String).includes(q.get('seed')))$('seed').value=q.get('seed');render()}
$('case').addEventListener('change',render);$('seed').addEventListener('change',render);$('refresh').addEventListener('click',load);$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load().catch(e=>$('status').textContent=`读取失败：${e}`);setInterval(()=>load().catch(()=>{}),20000);
</script></body></html>'''
