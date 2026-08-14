#!/usr/bin/env python3
"""Live dashboard for the multi-object M1 contrast-guidance search."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_multi_object_search_v1"
)
MANIFEST_PATH = ROOT / "search_manifest.json"
GUIDED_ROOT = ROOT / "guided"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def _variant(scale: float, start: int, end: int) -> str:
    return (
        "multi_object_blockdiag__m1_all_time__top100"
        f"__pag{_tag(scale)}__denoise_{start:02d}_{end:02d}"
    )


def _directory(case: str, seed: int, scale: float, start: int, end: int) -> Path:
    return GUIDED_ROOT / case / f"seed_{seed:05d}" / _variant(scale, start, end)


def _manifest() -> dict[str, Any]:
    return _read_json(MANIFEST_PATH)


def _sample_map(manifest: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.get("samples") or []:
        if not isinstance(row, dict):
            continue
        try:
            result[(str(row["case"]), int(row["seed"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _record(case: str, seed: int, scale: float, start: int, end: int) -> dict[str, Any]:
    directory = _directory(case, seed, scale, start, end)
    video = directory / "generated.mp4"
    complete = directory / "complete.json"
    run_manifest = _read_json(directory / "manifest.json")
    audit = run_manifest.get("audit") if isinstance(run_manifest.get("audit"), dict) else {}
    block = audit.get("block_diagonal") if isinstance(audit.get("block_diagonal"), dict) else {}
    deltas = audit.get("perturbation_delta_l2_by_step") or {}
    finite_deltas: list[float] = []
    if isinstance(deltas, dict):
        for value in deltas.values():
            try:
                finite_deltas.append(float(value))
            except (TypeError, ValueError):
                pass
    error_path = directory / "error.json"
    if not error_path.is_file():
        error_path = directory / "error.txt"
    if complete.is_file() and video.is_file():
        state = "complete"
    elif error_path.is_file():
        state = "error"
    elif directory.is_dir():
        state = "running"
    else:
        state = "pending"
    return {
        "scale": scale,
        "window": [start, end],
        "window_key": f"{start:02d}_{end:02d}",
        "variant": _variant(scale, start, end),
        "state": state,
        "ready": state == "complete",
        "object_count": int(run_manifest.get("object_count") or 0),
        "object_regions": list(run_manifest.get("object_regions") or []),
        "modified_head_events": int(audit.get("modified_head_events") or 0),
        "expected_modified_head_events": int(audit.get("expected_modified_head_events") or 0),
        "deleted_pairs_per_head": int(block.get("deleted_pair_count_per_head") or 0),
        "overlap_token_count": int(block.get("overlap_token_count") or 0),
        "duplicate_pair_subtractions_prevented": int(
            block.get("duplicate_pair_subtractions_prevented") or 0
        ),
        "mean_prediction_delta_l2": (
            sum(finite_deltas) / len(finite_deltas) if finite_deltas else None
        ),
        "error": (
            error_path.read_text(encoding="utf-8", errors="replace")[-1000:]
            if error_path.is_file()
            else ""
        ),
    }


def catalog() -> dict[str, Any]:
    manifest = _manifest()
    samples = _sample_map(manifest)
    grid = manifest.get("search_grid") if isinstance(manifest.get("search_grid"), dict) else {}
    scales = [float(value) for value in grid.get("pag_scales") or []]
    windows = [
        [int(value[0]), int(value[1])]
        for value in grid.get("guidance_windows_inclusive") or []
        if isinstance(value, list) and len(value) == 2
    ]
    cases = list(dict.fromkeys(case for case, _ in samples))
    seeds = [int(value) for value in manifest.get("seeds") or []]
    rows: list[dict[str, Any]] = []
    case_complete = {case: 0 for case in cases}
    total_complete = 0
    total_error = 0
    for case in cases:
        for seed in seeds:
            sample = samples.get((case, seed))
            if sample is None:
                continue
            records = [
                _record(case, seed, scale, start, end)
                for start, end in windows
                for scale in scales
            ]
            completed = sum(record["ready"] for record in records)
            errors = sum(record["state"] == "error" for record in records)
            total_complete += completed
            total_error += errors
            case_complete[case] += completed
            baseline = Path(str(sample.get("baseline_video") or ""))
            rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "caption": str(sample.get("caption") or ""),
                    "objects": [
                        str(region.get("region_name") or "")
                        for region in sample.get("regions") or []
                        if isinstance(region, dict)
                    ],
                    "baseline_ready": baseline.is_file(),
                    "records": records,
                    "progress": {
                        "complete": completed,
                        "expected": len(scales) * len(windows),
                        "errors": errors,
                    },
                }
            )
    expected = len(rows) * len(scales) * len(windows)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest.get("experiment_id"),
        "cases": cases,
        "seeds": seeds,
        "scales": scales,
        "windows": windows,
        "rows": rows,
        "controlled": manifest.get("controlled") or {},
        "progress": {
            "guided_complete": total_complete,
            "guided_expected": expected,
            "baseline_complete": sum(row["baseline_ready"] for row in rows),
            "baseline_expected": len(rows),
            "errors": total_error,
            "by_case": [
                {
                    "case": case,
                    "complete": case_complete[case],
                    "expected": len(seeds) * len(scales) * len(windows),
                }
                for case in cases
            ],
        },
    }


def asset(
    kind: str,
    case: str,
    seed: int,
    scale: float = 0.0,
    start: int = 0,
    end: int = 0,
) -> Path | None:
    manifest = _manifest()
    samples = _sample_map(manifest)
    sample = samples.get((case, seed))
    if sample is None:
        return None
    if kind == "baseline":
        path = Path(str(sample.get("baseline_video") or ""))
        return path if path.is_file() else None
    grid = manifest.get("search_grid") if isinstance(manifest.get("search_grid"), dict) else {}
    allowed_scales = [float(value) for value in grid.get("pag_scales") or []]
    allowed_windows = [
        (int(value[0]), int(value[1]))
        for value in grid.get("guidance_windows_inclusive") or []
        if isinstance(value, list) and len(value) == 2
    ]
    if kind != "guided" or scale not in allowed_scales or (start, end) not in allowed_windows:
        return None
    directory = _directory(case, seed, scale, start, end)
    path = directory / "generated.mp4"
    return path if (directory / "complete.json").is_file() and path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multi-object M1 Guidance Search</title><style>
:root{--bg:#dce7e9;--paper:#f5f9f8;--ink:#142a31;--muted:#62747a;--line:#93a8ad;--deep:#173d48;--cyan:#117f88;--red:#b1455b;--amber:#c2901e;--complete:#247457}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:linear-gradient(90deg,#173d480c 1px,transparent 1px),linear-gradient(#173d480c 1px,transparent 1px),var(--bg);background-size:22px 22px;color:var(--ink);font-family:"Aptos Narrow","Noto Sans CJK SC",Arial,sans-serif}a{color:var(--cyan);font-weight:800}header,main,footer{width:min(1800px,calc(100% - 28px));margin:auto}header{padding:24px 0 18px}.eyebrow{margin-top:20px;color:var(--red);font:900 11px ui-monospace,monospace;letter-spacing:.18em}.hero{display:grid;grid-template-columns:minmax(380px,.82fr) minmax(560px,1.18fr);gap:34px;align-items:end}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:8px 0 0;font-size:clamp(46px,7vw,96px);line-height:.85;letter-spacing:-.07em}.lead{margin:0;max-width:940px;font-size:17px;line-height:1.65}.equation{margin-top:18px;padding:13px 16px;background:var(--deep);color:#edf8f8;font:13px/1.65 ui-monospace,monospace}.equation b{color:#73d5d5}.window-ruler{display:grid;grid-template-columns:5fr 5fr 10fr 20fr;margin-top:16px;border:1px solid var(--line);background:var(--paper)}.window-ruler span{position:relative;padding:10px;text-align:center;border-right:1px solid var(--line);font:800 10px ui-monospace,monospace}.window-ruler span:last-child{border:0}.window-ruler span:after{content:"";position:absolute;left:0;bottom:0;height:4px;width:100%;background:var(--cyan);opacity:calc(.35 + var(--i)*.15)}.toolbar{position:sticky;top:0;z-index:6;display:flex;gap:9px;align-items:end;flex-wrap:wrap;padding:11px;margin-top:16px;border:1px solid var(--line);background:#dce7e9ef;backdrop-filter:blur(9px)}label{display:grid;gap:4px;color:var(--muted);font:900 10px ui-monospace,monospace;letter-spacing:.08em}select,button{min-height:37px;padding:7px 10px;border:1px solid var(--line);background:white;color:var(--ink);font-weight:800}select#case{max-width:min(650px,75vw)}button{cursor:pointer}.status{margin-left:auto;font:800 11px ui-monospace,monospace}.summary{display:grid;grid-template-columns:1.15fr repeat(3,1fr);gap:9px;margin:18px 0}.summary article{min-height:100px;padding:13px 15px;border:1px solid var(--line);background:var(--paper)}.summary strong{display:block;font:900 31px "Arial Black",sans-serif}.summary span{color:var(--muted);font:11px/1.5 ui-monospace,monospace}.summary .case-summary strong{font-size:19px;line-height:1.25}.baseline{margin:20px 0 24px;padding:13px;border:1px solid var(--line);background:var(--paper)}.section-head{display:flex;align-items:end;justify-content:space-between;gap:16px;padding-bottom:8px;border-bottom:3px solid var(--ink)}.section-head h2{margin:0;font-size:27px}.section-head p{margin:0;color:var(--muted);font:11px ui-monospace,monospace}.baseline-grid{display:grid;grid-template-columns:minmax(300px,580px) 1fr;gap:14px;margin-top:11px}.baseline-copy{padding:8px 5px}.baseline-copy h3{margin:0 0 8px;font-size:20px}.baseline-copy p{margin:5px 0;color:var(--muted);line-height:1.55}.window{margin:26px 0 36px}.window-label{display:flex;align-items:end;gap:18px;padding-bottom:9px;border-bottom:3px solid var(--deep)}.window-label .step{font:900 45px/.9 "Arial Black",sans-serif;color:var(--cyan)}.window-label h2{margin:0;font-size:26px}.window-label p{margin:3px 0 0;color:var(--muted)}.cards{display:grid;grid-template-columns:repeat(4,minmax(265px,1fr));gap:10px;margin-top:11px}.card{padding:9px;border:1px solid var(--line);border-top:6px solid var(--amber);background:var(--paper)}.card.negative{border-top-color:var(--red)}.card.positive{border-top-color:var(--cyan)}.card h3{display:flex;justify-content:space-between;gap:8px;margin:1px 0 8px;font-size:16px}.badge{color:var(--complete);font:900 10px ui-monospace,monospace}video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#071318}.facts{display:flex;gap:5px;flex-wrap:wrap;margin-top:8px}.facts span{padding:4px 6px;background:#e3ebec;font:10px ui-monospace,monospace}.card details{margin-top:7px;color:var(--muted);font:11px/1.5 ui-monospace,monospace}.empty{margin-top:11px;padding:24px;border:1px dashed var(--line);background:#f5f9f880;color:var(--muted);text-align:center}.case-progress{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;margin:14px 0 25px}.case-progress a{display:grid;gap:5px;padding:8px;border:1px solid var(--line);background:var(--paper);color:var(--ink);text-decoration:none;font:10px ui-monospace,monospace;overflow:hidden}.case-progress a.active{outline:3px solid var(--cyan);outline-offset:-3px}.case-progress b{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.meter{height:5px;background:#d1dcde}.meter i{display:block;height:100%;background:var(--cyan)}footer{padding:0 0 45px;color:var(--muted);font:11px/1.6 ui-monospace,monospace}@media(max-width:1100px){.hero,.summary,.baseline-grid{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,minmax(260px,1fr))}.case-progress{grid-template-columns:repeat(3,1fr)}.status{width:100%;margin-left:0}}@media(max-width:650px){header,main,footer{width:calc(100% - 12px)}.cards,.case-progress{grid-template-columns:1fr}.window-label{align-items:start}.summary{grid-template-columns:1fr 1fr}.summary .case-summary{grid-column:1/-1}}@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style></head><body><header><a href="/">← 返回 8092 总入口</a> · <a href="/training-free-m1-control?v=2">单对象 M1 控制</a><div class="eyebrow">LATEST3350 TOP100 · BLOCK-DIAGONAL MULTI-OBJECT M1 · LIVE SEARCH</div><div class="hero"><h1>每个对象，<br>切自己的线。</h1><p class="lead">同一次生成里，同时删除每个对象 tube 内部的 R→R contribution；不同对象之间的 Rᵢ↔Rⱼ 通信继续保留。选择一个 case 和 seed 后，页面把共同 Baseline 与当前已经生成的全部 scale × guidance-window 方案放在同一张比较板上；未生成项不占空卡位。</p></div><div class="equation">ε = εᵤ + 5(εc−εᵤ) + λ(εc−εc,<b>M1-multi</b>) &nbsp; | &nbsp; Y′[Rᵢ] = Y[Rᵢ] − A[Rᵢ,Rᵢ]V[Rᵢ]<br>保留 A[Rᵢ,Rⱼ]V[Rⱼ], i≠j；窗口外严格使用 clean CFG。</div><div class="window-ruler"><span style="--i:0">STEP 0–4</span><span style="--i:1">STEP 0–9</span><span style="--i:2">STEP 0–19</span><span style="--i:3">STEP 0–39</span></div><div class="toolbar"><label>CASE<select id="case"></select></label><label>SEED<select id="seed"></select></label><button id="refresh">刷新已生成结果</button><button id="replay">同步重播</button><span id="status" class="status">读取中…</span></div></header><main><section id="summary" class="summary"></section><nav id="caseProgress" class="case-progress"></nav><section id="baseline" class="baseline"></section><div id="windows"></div></main><footer>页面每 20 秒重新扫描 complete.json。只加载当前 case×seed 且已经完成的视频，并在接近视口时才请求 MP4，避免 1600 个搜索结果同时拖慢页面。</footer><script>
const api='/api/training-free-m1-multi-object-search',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,initialized=false;
const fmt=v=>(Number(v)>0?'+':'')+Number(v).toFixed(Math.abs(Number(v))===1?0:1),klass=v=>Number(v)<0?'negative':'positive';
function opts(node,items){const old=node.value;node.innerHTML=items.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');if([...node.options].some(o=>o.value===old))node.value=old}
function url(kind,row,r=null){const p={kind,case:row.case,seed:row.seed};if(r)Object.assign(p,{scale:r.scale,start:r.window[0],end:r.window[1]});return `${api}/asset?${new URLSearchParams(p)}`}
function lazy(){const io=new IntersectionObserver(xs=>xs.forEach(x=>{if(x.isIntersecting){const v=x.target;v.src=v.dataset.src;v.load();io.unobserve(v)}}),{rootMargin:'650px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function syncUrl(){const u=new URL(location.href);u.searchParams.set('case',$('case').value);u.searchParams.set('seed',$('seed').value);u.searchParams.set('v','1');history.replaceState(null,'',u)}
function card(row,r){const direction=r.scale<0?'靠近 M1-multi knockout':'远离 M1-multi knockout';return `<article class="card ${klass(r.scale)}"><h3>λ ${fmt(r.scale)}<span class="badge">COMPLETE</span></h3><video controls muted loop playsinline preload="none" data-src="${esc(url('guided',row,r))}"></video><div class="facts"><span>${esc(direction)}</span><span>step ${r.window[0]}–${r.window[1]}</span><span>${r.object_count||row.objects.length} objects</span>${r.modified_head_events?`<span>${r.modified_head_events} head-events</span>`:''}</div><details><summary>精确运行审计</summary><div>deleted pairs/head: ${r.deleted_pairs_per_head||'N/A'}<br>overlap tokens: ${r.overlap_token_count}<br>duplicate subtraction prevented: ${r.duplicate_pair_subtractions_prevented}<br>mean ‖εc−εM1‖₂: ${r.mean_prediction_delta_l2==null?'N/A':Number(r.mean_prediction_delta_l2).toFixed(3)}</div></details></article>`}
function render(){const caseName=$('case').value,seed=Number($('seed').value),row=data.rows.find(x=>x.case===caseName&&x.seed===seed);if(!row)return;syncUrl();const pct=Math.round(100*data.progress.guided_complete/Math.max(1,data.progress.guided_expected));$('status').textContent=`Guided ${data.progress.guided_complete}/${data.progress.guided_expected} · Baseline ${data.progress.baseline_complete}/${data.progress.baseline_expected} · errors ${data.progress.errors} · ${new Date(data.generated_at_utc).toLocaleTimeString()}`;$('summary').innerHTML=`<article class="case-summary"><strong>${esc(row.case)}</strong><span>${esc(row.objects.join(' + '))}<br>${esc(row.caption)}</span></article><article><strong>${row.progress.complete}/${row.progress.expected}</strong><span>当前 case×seed 已生成方案</span></article><article><strong>${pct}%</strong><span>全搜索 guidance 完成率</span></article><article><strong>${data.progress.errors}</strong><span>当前错误记录</span></article>`;$('caseProgress').innerHTML=data.progress.by_case.map(x=>`<a class="${x.case===row.case?'active':''}" href="?case=${encodeURIComponent(x.case)}&seed=${row.seed}&v=1"><b>${esc(x.case)}</b><span>${x.complete}/${x.expected}</span><div class="meter"><i style="width:${100*x.complete/Math.max(1,x.expected)}%"></i></div></a>`).join('');$('baseline').innerHTML=`<div class="section-head"><h2>共同 Baseline · seed ${row.seed}</h2><p>λ=0；同 seed、同首帧、同 prompt、同初始噪声</p></div><div class="baseline-grid">${row.baseline_ready?`<video controls muted loop playsinline preload="metadata" src="${esc(url('baseline',row))}"></video>`:'<div class="empty">Baseline 尚未落盘</div>'}<div class="baseline-copy"><h3>${esc(row.objects.length)} 个对象同时接受独立 M1 guidance</h3><p>每个方案都与左侧同一个 Baseline 比较。负 λ 把 prediction 推向多对象 M1 knockout；正 λ 沿相反方向增强差分。MSE 与 CoTracker trajectory loss 只在生成后用于选参，不进入 guidance。</p></div></div>`;const blocks=data.windows.map((w,i)=>{const ready=row.records.filter(r=>r.window[0]===w[0]&&r.window[1]===w[1]&&r.ready).sort((a,b)=>a.scale-b.scale);if(!ready.length)return '';return `<section class="window"><div class="window-label"><span class="step">${String(i+1).padStart(2,'0')}</span><div><h2>Guidance step ${w[0]}–${w[1]}</h2><p>${ready.length}/${data.scales.length} 个 scale 已完成 · 其余去噪步保持 clean CFG</p></div></div><div class="cards">${ready.map(r=>card(row,r)).join('')}</div></section>`}).join('');$('windows').innerHTML=blocks||'<div class="empty">该 case×seed 尚未生成 guidance 视频；Baseline 仍可查看。</div>';lazy()}
async function load(){data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());const oldCase=$('case').value,oldSeed=$('seed').value;opts($('case'),data.cases);opts($('seed'),data.seeds.map(String));if(!initialized){if(q.get('case')&&data.cases.includes(q.get('case')))$('case').value=q.get('case');if(q.get('seed')&&data.seeds.map(String).includes(q.get('seed')))$('seed').value=q.get('seed');initialized=true}else{if(data.cases.includes(oldCase))$('case').value=oldCase;if(data.seeds.map(String).includes(oldSeed))$('seed').value=oldSeed}render()}
$('case').addEventListener('change',render);$('seed').addEventListener('change',render);$('refresh').addEventListener('click',load);$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load().catch(e=>$('status').textContent=`读取失败：${e}`);setInterval(()=>load().catch(()=>{}),20000);
</script></body></html>'''
