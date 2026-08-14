#!/usr/bin/env python3
"""Case-grouped live dashboard for the test_5 Phase-B/Phase-D experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_direct_enhancement_v2/test5_20case_5seed"
)
MANIFEST_PATH = ROOT / "test5_phase_bd_manifest.json"
SELECTION_PATH = ROOT / "phase_b_selection.json"
STATUS_PATH = ROOT / "pipeline_status.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _asset_id(phase: str, alpha: float, start: int, end: int) -> str:
    return f"{phase}_a{_tag(alpha)}_w{start:02d}_{end:02d}"


def _variant_dir(
    phase: str, case: str, seed: int, alpha: float, start: int, end: int
) -> Path:
    variant = (
        "single_object__object_A__m1_all_time__top100"
        f"__alpha_{_tag(alpha)}__denoise_{start:02d}_{end:02d}"
    )
    return ROOT / phase / case / f"seed_{seed:05d}" / variant


def _complete(directory: Path) -> bool:
    return (directory / "complete.json").is_file() and (
        directory / "generated.mp4"
    ).is_file()


def _variant(
    *, phase: str, alpha: float, start: int, end: int, case: str, seed: int
) -> dict[str, Any]:
    directory = _variant_dir(phase, case, seed, alpha, start, end)
    manifest = _read_json(directory / "manifest.json")
    return {
        "asset_id": _asset_id(phase, alpha, start, end),
        "phase": phase,
        "alpha": float(alpha),
        "denoise_start": int(start),
        "denoise_end": int(end),
        "ready": _complete(directory),
        "selected_head_count": int(manifest.get("selected_head_count") or 100),
        "applied_head_events": int(
            manifest.get("audit", {}).get("applied_head_events")
            if isinstance(manifest.get("audit"), dict)
            else manifest.get("applied_head_events") or 0
        ),
        "reused_from_phase_b": bool(manifest.get("reused_from_phase_b")),
    }


def _manifest() -> dict[str, Any]:
    return _read_json(MANIFEST_PATH)


def catalog() -> dict[str, Any]:
    manifest = _manifest()
    selection = _read_json(SELECTION_PATH)
    selected_alpha_raw = selection.get("selected_alpha")
    selected_alpha = (
        float(selected_alpha_raw) if selected_alpha_raw is not None else None
    )
    phase_b_alphas = [
        float(value) for value in manifest.get("phase_b", {}).get("alphas") or []
    ]
    samples = [row for row in manifest.get("samples") or [] if isinstance(row, dict)]

    case_order: list[str] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    phase_b_complete = 0
    phase_d_complete = 0
    baseline_complete = 0
    available_assets = 0

    for sample in samples:
        case = str(sample.get("case") or "")
        seed = int(sample.get("seed", -1))
        if not case or seed < 0:
            continue
        if case not in grouped:
            case_order.append(case)
            grouped[case] = []
        baseline = Path(str(sample.get("baseline_video") or ""))
        baseline_ready = baseline.is_file()
        baseline_complete += int(baseline_ready)
        available_assets += int(baseline_ready)

        phase_b = [
            _variant(
                phase="phase_b",
                alpha=alpha,
                start=0,
                end=39,
                case=case,
                seed=seed,
            )
            for alpha in phase_b_alphas
        ]
        phase_b_complete += sum(item["ready"] for item in phase_b)

        phase_d: list[dict[str, Any]] = []
        if selected_alpha is not None:
            phase_d = [
                _variant(
                    phase="phase_d",
                    alpha=selected_alpha,
                    start=start,
                    end=end,
                    case=case,
                    seed=seed,
                )
                for start, end in ((0, 9), (0, 19))
            ]
            phase_d_complete += sum(item["ready"] for item in phase_d)
        available_assets += sum(item["ready"] for item in phase_b + phase_d)
        grouped[case].append(
            {
                "case": case,
                "seed": seed,
                "caption": str(sample.get("caption") or ""),
                "baseline_ready": baseline_ready,
                "phase_b": phase_b,
                "phase_d": phase_d,
            }
        )

    cases = []
    for case in case_order:
        rows = grouped[case]
        cases.append(
            {
                "case": case,
                "caption": rows[0]["caption"] if rows else "",
                "seeds": rows,
                "available_assets": sum(
                    int(row["baseline_ready"])
                    + sum(item["ready"] for item in row["phase_b"] + row["phase_d"])
                    for row in rows
                ),
            }
        )

    status = _read_json(STATUS_PATH)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest.get("experiment_id"),
        "cases": cases,
        "case_names": case_order,
        "seeds": [int(value) for value in manifest.get("seeds") or []],
        "head_scope": manifest.get("head_scope"),
        "selected_alpha": selected_alpha,
        "selection_ready": selected_alpha is not None,
        "selection_rule": selection.get("selection_rule", ""),
        "pipeline": {
            "stage": status.get("stage", "unknown"),
            "state": status.get("state", "unknown"),
            "physical_gpu": status.get("physical_gpu"),
        },
        "progress": {
            "case_count": len(cases),
            "sample_count": len(samples),
            "baseline_complete": baseline_complete,
            "baseline_total": len(samples),
            "phase_b_complete": phase_b_complete,
            "phase_b_total": len(samples) * len(phase_b_alphas),
            "phase_d_complete": phase_d_complete,
            "phase_d_total": len(samples) * 2,
            "available_assets": available_assets,
        },
    }


def asset(case: str, seed: int, asset_id: str) -> Path | None:
    manifest = _manifest()
    samples = [row for row in manifest.get("samples") or [] if isinstance(row, dict)]
    sample = next(
        (
            row
            for row in samples
            if str(row.get("case")) == case and int(row.get("seed", -1)) == seed
        ),
        None,
    )
    if sample is None:
        return None
    if asset_id == "baseline":
        path = Path(str(sample.get("baseline_video") or ""))
        return path if path.is_file() else None

    allowed: list[tuple[str, float, int, int]] = []
    for alpha in manifest.get("phase_b", {}).get("alphas") or []:
        allowed.append(("phase_b", float(alpha), 0, 39))
    selection = _read_json(SELECTION_PATH)
    if selection.get("selected_alpha") is not None:
        alpha = float(selection["selected_alpha"])
        allowed.extend(("phase_d", alpha, 0, end) for end in (9, 19))
    for phase, alpha, start, end in allowed:
        if _asset_id(phase, alpha, start, end) != asset_id:
            continue
        path = _variant_dir(phase, case, seed, alpha, start, end) / "generated.mp4"
        return path if _complete(path.parent) else None
    return None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase B/D · M1 Direct Enhancement</title><style>
:root{--bg:#edf2f1;--paper:#fbfcf8;--ink:#172b30;--muted:#617176;--line:#9dada9;--navy:#12333c;--b:#187d85;--d:#c4673a;--base:#718086;--ok:#2a725b;--wait:#9a7920}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#12333c0a 1px,transparent 1px),linear-gradient(#12333c0a 1px,transparent 1px),var(--bg);background-size:24px 24px;font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif}a{color:var(--b);font-weight:800}header,main,footer{width:min(1880px,calc(100% - 28px));margin:auto}header{padding:23px 0 13px}.nav{font-size:13px}.eyebrow{margin-top:22px;color:var(--d);font:900 12px ui-monospace,monospace;letter-spacing:.17em}.hero{display:grid;grid-template-columns:minmax(420px,.95fr) minmax(520px,1.05fr);gap:38px;align-items:end}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:7px 0 0;font-size:clamp(43px,6vw,82px);line-height:.9;letter-spacing:-.06em}.lead{margin:0;max-width:900px;font-size:17px;line-height:1.66}.equation{margin-top:18px;padding:12px 15px;background:var(--navy);color:#eff8f5;font:13px/1.7 ui-monospace,monospace}.timeline{display:grid;grid-template-columns:repeat(40,1fr);height:22px;margin-top:13px;border:1px solid var(--line);background:white}.timeline i{position:relative;border-right:1px solid #dce4e1}.timeline i:nth-child(-n+10){background:#d879482b}.timeline i:nth-child(n+11):nth-child(-n+20){background:#d8a14821}.timeline i:nth-child(n+21){background:#21858b13}.timeline-labels{display:grid;grid-template-columns:1fr 1fr 2fr;margin:5px 0 0;font:800 10px ui-monospace,monospace;color:var(--muted)}.timeline-labels span:nth-child(2){text-align:center}.timeline-labels span:last-child{text-align:right}.toolbar{position:sticky;top:0;z-index:6;display:flex;align-items:end;gap:10px;flex-wrap:wrap;margin-top:18px;padding:12px;border:1px solid var(--line);background:#edf2f1ed;backdrop-filter:blur(9px)}label{display:grid;gap:4px;color:var(--muted);font:900 10px ui-monospace,monospace;letter-spacing:.08em}select,button{min-height:37px;padding:7px 11px;border:1px solid var(--line);background:white;color:var(--ink);font-weight:900}button{cursor:pointer}.status{margin-left:auto;font:800 11px/1.45 ui-monospace,monospace}.summary{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:18px 0}.stat{padding:13px;background:var(--paper);border:1px solid var(--line)}.stat b{display:block;font:900 25px "Arial Black",sans-serif}.stat span{color:var(--muted);font:800 10px ui-monospace,monospace}.methods{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:12px 0 22px}.methods article{padding:15px;border:1px solid var(--line);background:var(--paper)}.methods h2{margin:0 0 6px;font-size:21px}.methods p{margin:4px 0;color:var(--muted);line-height:1.5}.methods code{font:12px ui-monospace,monospace}.case-head{display:grid;grid-template-columns:1fr minmax(280px,.8fr);gap:20px;align-items:end;padding:13px 0 10px;border-bottom:4px solid var(--ink)}.case-head h2{margin:0;font-size:clamp(24px,3vw,40px);overflow-wrap:anywhere}.case-head p{margin:0;text-align:right;color:var(--muted);line-height:1.4}.seed-row{margin:14px 0 25px}.seed-title{display:flex;align-items:baseline;gap:10px;padding-bottom:7px;border-bottom:1px solid var(--line)}.seed-title h3{margin:0;font-size:20px}.seed-title span{color:var(--muted);font:800 10px ui-monospace,monospace}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:9px;margin-top:9px}.card{min-width:0;padding:9px;border:1px solid var(--line);border-top:6px solid var(--base);background:var(--paper)}.card.phase_b{border-top-color:var(--b)}.card.phase_d{border-top-color:var(--d)}.card.winner{outline:2px solid #d69b35;outline-offset:-4px}.card-head{display:flex;justify-content:space-between;gap:10px;align-items:start;margin-bottom:8px}.card h3{margin:0;font-size:16px}.badge{flex:none;padding:3px 5px;background:#e5ece9;color:var(--muted);font:900 9px ui-monospace,monospace}.winner .badge{background:#f1d797;color:#5d4610}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#081317}.facts{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}.facts span{padding:4px 6px;background:#e8eeeb;font:10px ui-monospace,monospace}.meaning{margin:8px 1px 1px;color:var(--muted);font-size:12px;line-height:1.45}.empty{padding:34px;text-align:center;border:1px dashed var(--line);color:var(--muted);background:#f8faf7}.loading{min-height:220px;display:grid;place-items:center;color:var(--muted)}footer{padding:12px 0 42px;color:var(--muted);font-size:11px}@media(max-width:980px){.hero,.case-head,.methods{grid-template-columns:1fr}.case-head p{text-align:left}.summary{grid-template-columns:repeat(2,1fr)}.status{width:100%;margin-left:0}}@media(max-width:620px){header,main,footer{width:calc(100% - 12px)}.grid{grid-template-columns:1fr}.summary{grid-template-columns:1fr 1fr}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
</style></head><body><header><div class="nav"><a href="/">← 返回 8092 总入口</a> · <a href="/training-free-m1-control?v=1">旧 M1 双向控制台</a></div><div class="eyebrow">LATEST3350 TOP100 / TEST_5 / PHASE B + D</div><div class="hero"><h1>直接加强<br>对象内部通信。</h1><p class="lead">每次只选择一个 case，并把该 case 的 5 个 seed 分行展示；每行只放已生成的 Baseline 与全部可用方案。这样既保持严格控制变量，也不会让 20 个 case 的视频同时加载。</p></div><div class="equation">M<sub>RR</sub> = Σ A[R<sub>tq</sub>,R<sub>tk</sub>]V[R<sub>tk</sub>]　　Y<sub>R</sub> ← Y<sub>R</sub> + α·M<sub>RR</sub><br>固定：同一 input / prompt / seed / object_A tube / latest3350 Top100 / CFG=5 / 40 sampling steps；conditional 与 unconditional 分支同步修改。</div><div class="timeline" id="timeline"></div><div class="timeline-labels"><span>First10 · 0–9</span><span>First20 · 0–19</span><span>Full40 · 0–39</span></div><div class="toolbar"><label>CASE<select id="case"></select></label><button id="previous">上一 case</button><button id="next">下一 case</button><button id="refresh">刷新结果</button><button id="replay">同步重播</button><span id="status" class="status">读取中…</span></div></header><main><section id="summary" class="summary"></section><section class="methods"><article><h2>Phase B · 选择增益</h2><p><code>α ∈ {0.1, 0.25}, window=0..39</code>。先完整生成两组，再用生成后的 GT full-frame MSE 与 GT-relative CoTracker Center-ADE/D0 选一个 α；指标不参与生成。</p></article><article><h2>Phase D · 选择时窗</h2><p><code>α = Phase-B winner, window ∈ {0..9, 0..19, 0..39}</code>。只改变介入的去噪步范围；Full40 直接复用获胜的 Phase-B 视频，因此页面只显示一次。</p></article></section><section id="content" class="loading">正在读取生成目录…</section></main><footer>视频采用视口懒加载；自动刷新只在新结果出现时重绘。页面不为尚未生成的方案保留空卡片。</footer><script>
const api='/api/training-free-m1-phase-bd',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,lastSignature='';
$('timeline').innerHTML='<i></i>'.repeat(40);
function media(row,id){return `${api}/asset?${new URLSearchParams({case:row.case,seed:row.seed,asset_id:id})}`}
function lazy(){const io=new IntersectionObserver(xs=>xs.forEach(x=>{if(x.isIntersecting){const v=x.target;v.src=v.dataset.src;v.load();io.unobserve(v)}}),{rootMargin:'450px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function card(row,v){const isBase=v.asset_id==='baseline',phase=isBase?'baseline':v.phase,label=isBase?'Baseline':v.phase==='phase_b'?`Phase B · α=${v.alpha} · Full40`:`Phase D · α=${v.alpha} · First${v.denoise_end+1}`,winner=!isBase&&data.selection_ready&&v.phase==='phase_b'&&Number(v.alpha)===Number(data.selected_alpha),meaning=isBase?'同 seed 未干预生成；所有方案的控制组。':v.phase==='phase_b'?`在全部 40 个去噪步增强 Top100 heads 的对象内部 R→R contribution。`:`只在去噪步 ${v.denoise_start}–${v.denoise_end} 增强相同 R→R contribution，其余步骤保持 Baseline。`;return `<article class="card ${esc(phase)} ${winner?'winner':''}"><div class="card-head"><h3>${esc(label)}</h3><span class="badge">${winner?'WINNER / D FULL40':isBase?'CONTROL':'READY'}</span></div><video controls muted loop playsinline preload="none" data-src="${esc(media(row,v.asset_id))}"></video><div class="facts"><span>${isBase?'α=0':`α=${v.alpha}`}</span><span>${isBase?'no intervention':`steps ${v.denoise_start}–${v.denoise_end}`}</span><span>${isBase?'same seed':`Top${v.selected_head_count}`}</span>${v.reused_from_phase_b?'<span>reused Phase B</span>':''}</div><p class="meaning">${esc(meaning)}</p></article>`}
function render(){const current=data.cases.find(x=>x.case===$('case').value);if(!current){$('content').innerHTML='<div class="empty">该 case 尚无可展示结果</div>';return}const p=data.progress,sel=data.selection_ready?`α=${data.selected_alpha}`:'待 Phase B 完成';$('summary').innerHTML=`<article class="stat"><b>${p.case_count}</b><span>CASES</span></article><article class="stat"><b>${p.sample_count}</b><span>CASE × SEED SAMPLES</span></article><article class="stat"><b>${p.phase_b_complete}/${p.phase_b_total}</b><span>PHASE B VIDEOS</span></article><article class="stat"><b>${p.phase_d_complete}/${p.phase_d_total}</b><span>PHASE D NEW VIDEOS</span></article><article class="stat"><b>${esc(sel)}</b><span>PHASE B WINNER</span></article>`;$('status').textContent=`${data.pipeline.stage} · ${data.pipeline.state} · GPU ${data.pipeline.physical_gpu??'—'} · ${new Date(data.generated_at_utc).toLocaleTimeString()}`;const rows=current.seeds.map(row=>{const variants=[];if(row.baseline_ready)variants.push({asset_id:'baseline'});variants.push(...row.phase_b.filter(v=>v.ready),...row.phase_d.filter(v=>v.ready));return `<section class="seed-row"><div class="seed-title"><h3>Seed ${row.seed}</h3><span>${variants.length} READY · ${esc(row.caption)}</span></div>${variants.length?`<div class="grid">${variants.map(v=>card(row,v)).join('')}</div>`:'<div class="empty">该 seed 暂无已生成视频</div>'}</section>`}).join('');$('content').className='';$('content').innerHTML=`<div class="case-head"><h2>${esc(current.case)}</h2><p>${esc(current.caption)}<br>${current.available_assets} 个当前可用视频</p></div>${rows}`;q.set('case',current.case);history.replaceState(null,'',`${location.pathname}?${q}`);lazy()}
function selectOffset(delta){const names=data.case_names,i=Math.max(0,names.indexOf($('case').value)),next=(i+delta+names.length)%names.length;$('case').value=names[next];render();scrollTo({top:0,behavior:'smooth'})}
async function load(force=false){const old=$('case').value||q.get('case');const next=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());const signature=JSON.stringify([next.progress,next.selected_alpha,next.pipeline.stage,next.pipeline.state]);data=next;$('case').innerHTML=data.cases.map(x=>`<option value="${esc(x.case)}">${esc(x.case)} · ${x.available_assets} ready</option>`).join('');if(old&&data.case_names.includes(old))$('case').value=old;if(force||signature!==lastSignature){lastSignature=signature;render()}else $('status').textContent=`${data.pipeline.stage} · ${data.pipeline.state} · GPU ${data.pipeline.physical_gpu??'—'} · ${new Date(data.generated_at_utc).toLocaleTimeString()}`}
$('case').addEventListener('change',render);$('previous').addEventListener('click',()=>selectOffset(-1));$('next').addEventListener('click',()=>selectOffset(1));$('refresh').addEventListener('click',()=>load(true));$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load(true).catch(e=>$('status').textContent=`读取失败：${e}`);setInterval(()=>load(false).catch(()=>{}),20000);
</script></body></html>'''
