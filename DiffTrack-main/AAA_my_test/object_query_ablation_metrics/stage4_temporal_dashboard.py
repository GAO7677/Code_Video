#!/usr/bin/env python3
"""Live Stage-4 temporal information-flow video dashboard."""

from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
VIDEO_ROOT = ROOT / "stage4_temporal_v1"
RUNTIME_ROOT = ROOT / "stage4_runtime"
RUNTIME_MANIFEST = RUNTIME_ROOT / "stage4_manifest.json"
RUNTIME_SUMMARY = RUNTIME_ROOT / "runtime_summary.json"
METRICS_ROOT = ROOT / "stage4_metrics"

HEAD_ORDER = (
    "top100",
    "bottom100",
    "random100_layer_matched_draw0",
    "all720",
)
HEAD_LABELS = {
    "top100": "latest3350 Top100",
    "bottom100": "latest3350 Bottom100",
    "random100_layer_matched_draw0": "Layer-matched Random100",
    "all720": "All720",
}
HEAD_COUNTS = {"top100": 100, "bottom100": 100, "random100_layer_matched_draw0": 100, "all720": 720}
FLOW_ORDER = ("self", "incoming", "outgoing")
FLOW_DEFINITIONS = {
    "self": {
        "id": "M1",
        "flow": "R K/V → R Query",
        "formula": "Y′[R] = Y[R] − A[R,R]V_R",
        "diagnosis": "对象 tube 内部状态、身份与运动连续性。",
    },
    "incoming": {
        "id": "M2",
        "flow": "C K/V → R Query",
        "formula": "Y′[R] = Y[R] − A[R,C]V_C",
        "diagnosis": "环境和其他对象输入目标对象的约束。",
    },
    "outgoing": {
        "id": "M3",
        "flow": "R K/V → C Query",
        "formula": "Y′[C] = Y[C] − A[C,R]V_R",
        "diagnosis": "目标对象向其他对象和背景广播的状态。",
    },
}
DIRECTION_ORDER = ("only", "same", "future", "past")
DIRECTION_DEFINITIONS = {
    "only": {"label": "All-time", "predicate": "all tₖ", "meaning": "删除该矩阵块的全部 K 时刻"},
    "same": {"label": "Same", "predicate": "tₖ = t_q", "meaning": "只删除同一 latent 时刻"},
    "future": {"label": "Future", "predicate": "tₖ < t_q", "meaning": "切断历史 K/V 向未来 Query"},
    "past": {"label": "Past", "predicate": "tₖ > t_q", "meaning": "未来 K/V 向过去 Query 的反向控制"},
}

_lock = threading.Lock()
_signature_value: tuple[int, ...] | None = None
_catalog_value: dict[str, Any] | None = None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _samples() -> list[dict[str, Any]]:
    rows = _load_json(RUNTIME_MANIFEST).get("samples") or []
    return [row for row in rows if isinstance(row, dict)]


def _targets(sample: dict[str, Any]) -> list[dict[str, str]]:
    objects = [
        str(row["region_name"])
        for row in sample.get("regions") or []
        if isinstance(row, dict) and row.get("region_type") == "object"
    ]
    result = [
        {"key": f"single_object::{name}", "label": name, "scope": "single_object", "region": name}
        for name in objects
    ]
    if len(objects) > 1:
        result.append({"key": "all_objects::", "label": "All objects", "scope": "all_objects", "region": ""})
    return result


def _mode_parts(mode: str) -> tuple[str, str] | None:
    for flow in FLOW_ORDER:
        prefix = f"{flow}_"
        if mode.startswith(prefix):
            direction = mode[len(prefix) :]
            if direction in DIRECTION_ORDER:
                return flow, direction
    return None


def _record(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path.parent / "manifest.json")
    mode = str(payload.get("mask_mode") or "")
    parts = _mode_parts(mode)
    scope = str(payload.get("head_scope") or "")
    if parts is None or scope not in HEAD_ORDER or not (path.parent / "generated.mp4").is_file():
        return None
    flow, direction = parts
    target_scope = str(payload.get("target_scope") or "")
    region = str(payload.get("region") or "")
    target_key = f"single_object::{region}" if target_scope == "single_object" else "all_objects::"
    audit = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
    return {
        "case": str(payload.get("case") or ""),
        "seed": int(payload.get("seed", -1)),
        "variant_id": str(payload.get("variant_id") or path.parent.name),
        "target_key": target_key,
        "target_scope": target_scope,
        "region": region or None,
        "mask_mode": mode,
        "flow": flow,
        "direction": direction,
        "head_scope": scope,
        "head_label": HEAD_LABELS[scope],
        "head_count": int(payload.get("selected_head_count") or HEAD_COUNTS[scope]),
        "token_count": len(audit.get("query_token_indices") or []),
        "modified_head_events": int(audit.get("modified_head_events") or 0),
        "dose_ready": (path.parent / "dose_metrics.npz").is_file(),
    }


def _records() -> list[dict[str, Any]]:
    result = []
    if VIDEO_ROOT.is_dir():
        for complete in sorted(VIDEO_ROOT.rglob("complete.json")):
            record = _record(complete)
            if record is not None:
                result.append(record)
    return result


def _signature() -> tuple[int, ...]:
    complete = sorted(VIDEO_ROOT.rglob("complete.json")) if VIDEO_ROOT.is_dir() else []
    latest = max((path.stat().st_mtime_ns for path in complete), default=0)
    fast = sorted((METRICS_ROOT / "head_scope_baseline_fast").rglob("report.json")) if METRICS_ROOT.is_dir() else []
    trajectory = sorted((METRICS_ROOT / "head_scope_trajectory").rglob("report.json")) if METRICS_ROOT.is_dir() else []
    survival = sorted((METRICS_ROOT / "head_scope_trajectory").rglob("object_survival_report.json")) if METRICS_ROOT.is_dir() else []
    complete25 = sorted((METRICS_ROOT / "head_scope_complete25").rglob("report.json")) if METRICS_ROOT.is_dir() else []
    return len(complete), latest, len(fast), len(trajectory), len(survival), len(complete25)


def _expected_by_case(samples: list[dict[str, Any]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for sample in samples:
        case = str(sample["case"])
        target_count = len(_targets(sample))
        result[case] += target_count * 27
        if case == "0613pybullet_sample_001460_w002":
            result[case] += target_count * 9
    result["0613pybullet_sample_001460_w002"] += 27
    return result


def _build_catalog() -> dict[str, Any]:
    samples_raw = _samples()
    records = _records()
    summary = _load_json(RUNTIME_SUMMARY)
    total_expected = int(summary.get("total_generation_cells") or 999)
    by_case = Counter(row["case"] for row in records)
    by_head = Counter(row["head_scope"] for row in records)
    by_flow = Counter(row["flow"] for row in records)
    by_direction = Counter(row["direction"] for row in records)
    expected_case = _expected_by_case(samples_raw)
    samples = [
        {
            "case": str(row["case"]),
            "seed": int(row["seed"]),
            "caption": str(row.get("caption") or ""),
            "targets": _targets(row),
            "baseline_ready": Path(str(row.get("baseline_video") or "")).is_file(),
        }
        for row in samples_raw
    ]
    fast_count = len(list((METRICS_ROOT / "head_scope_baseline_fast").rglob("report.json"))) if METRICS_ROOT.is_dir() else 0
    trajectory_count = len(list((METRICS_ROOT / "head_scope_trajectory").rglob("report.json"))) if METRICS_ROOT.is_dir() else 0
    survival_count = len(list((METRICS_ROOT / "head_scope_trajectory").rglob("object_survival_report.json"))) if METRICS_ROOT.is_dir() else 0
    complete25_count = len(list((METRICS_ROOT / "head_scope_complete25").rglob("report.json"))) if METRICS_ROOT.is_dir() else 0
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "latest3350 Stage 4 temporal validation",
        "records": records,
        "samples": samples,
        "case_count": len({row["case"] for row in samples}),
        "seed_count": len({row["seed"] for row in samples}),
        "progress": {
            "completed": len(records),
            "expected": total_expected,
            "by_case": [
                {"key": case, "completed": by_case[case], "expected": expected}
                for case, expected in expected_case.items()
            ],
            "by_head": [
                {
                    "key": scope,
                    "label": HEAD_LABELS[scope],
                    "completed": by_head[scope],
                    "expected": 324 if scope != "all720" else 27,
                }
                for scope in HEAD_ORDER
            ],
            "by_flow": [
                {"key": flow, "label": FLOW_DEFINITIONS[flow]["id"], "completed": by_flow[flow], "expected": 333}
                for flow in FLOW_ORDER
            ],
            "by_direction": [
                {
                    "key": direction,
                    "label": DIRECTION_DEFINITIONS[direction]["label"],
                    "completed": by_direction[direction],
                    "expected": 81 if direction == "only" else 306,
                }
                for direction in DIRECTION_ORDER
            ],
        },
        "head_scopes": [
            {"key": scope, "label": HEAD_LABELS[scope], "count": HEAD_COUNTS[scope]}
            for scope in HEAD_ORDER
        ],
        "flows": FLOW_DEFINITIONS,
        "flow_order": list(FLOW_ORDER),
        "directions": DIRECTION_DEFINITIONS,
        "direction_order": list(DIRECTION_ORDER),
        "dose_count": sum(bool(row["dose_ready"]) for row in records),
        "metrics": {
            "fast": fast_count,
            "trajectory": trajectory_count,
            "survival": survival_count,
            "complete25": complete25_count,
        },
    }


def catalog() -> dict[str, Any]:
    global _catalog_value, _signature_value
    signature = _signature()
    with _lock:
        if _catalog_value is None or signature != _signature_value:
            _catalog_value = _build_catalog()
            _signature_value = signature
        return _catalog_value


def _sample(case: str, seed: int) -> dict[str, Any] | None:
    return next(
        (row for row in _samples() if str(row.get("case")) == case and int(row.get("seed", -1)) == seed),
        None,
    )


def _variant_dir(case: str, seed: int, variant: str) -> Path | None:
    if not case or not variant or seed < 0:
        return None
    base = VIDEO_ROOT.resolve()
    candidate = (VIDEO_ROOT / case / f"seed_{seed:05d}" / variant).resolve()
    if base not in candidate.parents:
        return None
    manifest = _load_json(candidate / "manifest.json")
    if (
        str(manifest.get("case")) != case
        or int(manifest.get("seed", -1)) != seed
        or str(manifest.get("variant_id")) != variant
    ):
        return None
    return candidate


def asset(kind: str, case: str, seed: int, variant: str = "") -> Path | None:
    if kind == "baseline":
        sample = _sample(case, seed)
        path = Path(str(sample.get("baseline_video") or "")) if sample else None
    elif kind == "ablation":
        directory = _variant_dir(case, seed, variant)
        path = directory / "generated.mp4" if directory else None
    else:
        path = None
    return path if path is not None and path.is_file() else None


@lru_cache(maxsize=512)
def _dose(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    path = Path(path_text)
    manifest = _load_json(path.parent / "manifest.json")
    mask = np.zeros((30, 24), dtype=bool)
    for row in manifest.get("selected_entries") or []:
        mask[int(row["block"]), int(row["head"])] = True
    result: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as payload:
        for key in (
            "attention_mass",
            "attention_mass_query_sum",
            "removed_value_norm",
            "removed_value_norm_query_sum",
            "removed_to_output_ratio",
            "target_query_count",
        ):
            if key not in payload:
                continue
            values = np.asarray(payload[key], dtype=np.float64)
            selected = values[:, :, mask] if values.shape[-2:] == (30, 24) else values
            finite = selected[np.isfinite(selected)]
            result[key] = {
                "mean": float(finite.mean()) if finite.size else None,
                "median": float(np.median(finite)) if finite.size else None,
                "p95": float(np.quantile(finite, 0.95)) if finite.size else None,
                "finite_count": int(finite.size),
            }
    return result


def dose(case: str, seed: int, variant: str) -> dict[str, Any] | None:
    directory = _variant_dir(case, seed, variant)
    path = directory / "dose_metrics.npz" if directory else None
    if path is None or not path.is_file():
        return None
    return _dose(str(path), path.stat().st_mtime_ns)


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stage 4 · Object Query 时序信息流</title><style>
:root{--paper:#eee9dc;--ink:#17201e;--card:#fffdf8;--line:#b9b19f;--green:#176654;--rust:#b64a31;--blue:#315c86;--purple:#7056a2;--muted:#6c675d;--dark:#142820}*{box-sizing:border-box}body{margin:0;background:linear-gradient(115deg,#b64a3118,transparent 38%),radial-gradient(circle at 94% 2%,#315c8622,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}a{color:var(--green)}header,main{width:min(1880px,calc(100% - 24px));margin:auto}header{padding:24px 0 6px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(42px,6vw,78px);line-height:.95;letter-spacing:-.045em;margin:12px 0}.eyebrow{color:var(--rust);font-weight:900;font-size:12px;letter-spacing:.17em}.lead{max-width:1120px;line-height:1.65}.toolbar{position:sticky;top:0;z-index:20;display:flex;gap:9px;align-items:end;flex-wrap:wrap;padding:11px;margin:17px 0;background:#f8f3e8ed;border:1px solid var(--line);backdrop-filter:blur(12px)}label{display:grid;gap:4px;font-size:10px;font-weight:900}select,button{padding:9px 11px;border:1px solid var(--line);background:white;color:var(--ink);font-weight:800}select{min-width:190px}button{cursor:pointer}.status{font-size:11px;color:var(--muted);padding:8px 0}section{margin:18px 0;padding:17px;background:#ffffff9e;border:1px solid var(--line);border-radius:3px 20px 3px 3px}section h2{margin:0 0 10px;font-size:27px}.metric-row,.pill-row{display:flex;gap:8px;flex-wrap:wrap}.metric{min-width:160px;padding:11px;background:var(--card);border:1px solid var(--line)}.metric b{display:block;font-size:23px}.metric span,.pill{font-size:11px;color:var(--muted)}.pill{padding:7px 10px;border:1px solid var(--line);border-radius:99px;background:var(--card)}.progress-grid{display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:7px;margin-top:12px}.progress{display:grid;grid-template-columns:minmax(145px,1fr) 3fr 78px;gap:8px;align-items:center;font-size:11px}.track{height:9px;background:#ddd5c7;border-radius:99px;overflow:hidden}.track i{display:block;height:100%;background:linear-gradient(90deg,var(--rust),var(--green))}.baseline{max-width:520px}.baseline video,.video-card video{display:block;width:100%;aspect-ratio:16/9;background:#121817;object-fit:contain}.matrix-key{display:grid;grid-template-columns:100px 155px 1fr 1.2fr;min-width:760px;border-top:1px solid var(--line);border-left:1px solid var(--line)}.matrix-key>*{padding:9px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);font-size:11px}.matrix-key .head{background:var(--dark);color:white;font-weight:900}.flow-band{margin:24px 0 32px}.flow-head{display:grid;grid-template-columns:minmax(130px,.6fr) 1fr 1.2fr;gap:10px;align-items:end;border-bottom:2px solid var(--ink);padding-bottom:8px}.flow-head h2{margin:0}.flow-head .formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}.direction-row{margin-top:14px}.direction-head{display:flex;gap:12px;align-items:baseline;margin-bottom:7px}.direction-head h3{margin:0;font-size:20px}.direction-head span{font-size:11px;color:var(--muted)}.video-grid{display:grid;grid-template-columns:repeat(4,minmax(230px,1fr));gap:10px}.video-card{padding:9px;background:var(--card);border:1px solid var(--line);border-top:5px solid var(--accent,var(--green));border-radius:2px 15px 2px 2px}.video-card[data-scope="top100"]{--accent:var(--green)}.video-card[data-scope="bottom100"]{--accent:var(--rust)}.video-card[data-scope="random100_layer_matched_draw0"]{--accent:var(--purple)}.video-card[data-scope="all720"]{--accent:var(--blue)}.video-card h4{margin:0 0 7px;font-size:14px}.caption{font-size:11px;line-height:1.5;margin-top:7px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}.video-card details{font-size:11px;margin-top:7px}.dose{padding:7px;background:#eee9dc;line-height:1.65}.empty{padding:20px;border:1px dashed var(--line);color:var(--muted)}.scroll{overflow:auto}.footer{padding:8px 0 50px;color:var(--muted);font-size:11px}@media(max-width:1250px){.video-grid{grid-template-columns:repeat(3,minmax(220px,1fr))}}@media(max-width:850px){.video-grid,.progress-grid{grid-template-columns:repeat(2,minmax(190px,1fr))}.flow-head{grid-template-columns:1fr}.matrix-key{grid-template-columns:80px 120px 1fr 1fr}}@media(max-width:560px){header,main{width:calc(100% - 10px)}.video-grid,.progress-grid{grid-template-columns:1fr}.progress{grid-template-columns:1fr}.toolbar label,.toolbar select{width:100%}}
</style></head><body><header><a href="/">返回 8092 总入口</a> · <a href="/object-query-information-flow-validation?v=2">返回 Stage 1–3</a> · <a href="/object-query-information-flow-stage4-representatives?v=1">查看代表性正例与反例</a><div class="eyebrow">LATEST3350 · STAGE 4 TEMPORAL VALIDATION</div><h1>时间方向不是<br>一个附注</h1><p class="lead">固定同一 object tube、seed 和 head ranking，把 M1/M2/M3 拆成 All-time、Same、Future、Past。每行是一个精确的信息流切口；每列并排 Top100、Bottom100、layer-matched Random100 与 All720。只展示已完成视频。</p><div class="toolbar"><label>Case<select id="case"></select></label><label>Seed<select id="seed"></select></label><label>Target<select id="target"></select></label><button id="refresh">刷新进度</button><button id="replay">同步重播</button><span id="status" class="status">读取中…</span></div></header><main><section><h2>实时生成矩阵</h2><div id="metrics" class="metric-row"></div><div id="progress" class="progress-grid"></div></section><section><h2>精确时间谓词</h2><div class="scroll"><div id="directionKey" class="matrix-key"></div></div><div class="pill-row" style="margin-top:10px"><span class="pill">40 denoising steps</span><span class="pill">conditional + unconditional</span><span class="pill">post-softmax contribution subtraction</span><span class="pill">no renormalization</span></div></section><section><h2>共同 Baseline</h2><div id="baseline" class="baseline"></div></section><div id="gallery"></div><section><h2>结果指标状态</h2><p id="metricStatus"></p><p class="status">Dose 表示实际删除的 attention / A·V 贡献；轨迹、外观、背景和对象存活指标在生成矩阵结束后分片补全。</p></section><div class="footer">页面每 30 秒扫描完成标记。只有当前 case/seed/target 的视频进入 DOM，并在接近视口时请求 MP4。</div></main><script>
const api='/api/object-query-information-flow-stage4',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,sig='';
const pct=(a,b)=>b?`${(100*a/b).toFixed(1)}%`:'—',num=(v,d=4)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'—';
function metric(label,value,note){return `<div class="metric"><b>${esc(value)}</b><span>${esc(label)} · ${esc(note||'')}</span></div>`}
function options(el,rows,value,label,wanted){const previous=wanted??el.value;el.innerHTML=rows.map(x=>`<option value="${esc(value(x))}">${esc(label(x))}</option>`).join('');if([...el.options].some(o=>o.value===String(previous)))el.value=String(previous)}
function selectedSample(){return data.samples.find(x=>x.case===$('case').value&&String(x.seed)===$('seed').value)}
function filters(first=false){const cases=[...new Set(data.samples.map(x=>x.case))],wc=first?(q.get('case')||cases[0]):$('case').value;options($('case'),cases,x=>x,x=>x,wc);const seeds=data.samples.filter(x=>x.case===$('case').value).map(x=>x.seed),ws=first?(q.get('seed')||47326):$('seed').value;options($('seed'),seeds,x=>x,x=>`seed ${x}`,ws);const sample=selectedSample(),wt=first?(q.get('target')||sample?.targets[0]?.key):$('target').value;options($('target'),sample?.targets||[],x=>x.key,x=>x.label,wt)}
function media(kind,row={}){const p=new URLSearchParams({kind,case:row.case||$('case').value,seed:String(row.seed??$('seed').value),variant:row.variant_id||''});return `${api}/asset?${p}`}
function lazy(){const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){const v=e.target;if(v.dataset.src){v.src=v.dataset.src;delete v.dataset.src;v.load()}io.unobserve(v)}}),{rootMargin:'400px'});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}
function progress(x){return `<div class="progress"><span>${esc(x.label||x.key)}</span><div class="track"><i style="width:${pct(x.completed,x.expected)}"></i></div><b>${x.completed}/${x.expected}</b></div>`}
function doseBox(r){return `<details data-case="${esc(r.case)}" data-seed="${r.seed}" data-variant="${esc(r.variant_id)}"><summary>实际删除 dose · 点击加载</summary><div class="dose">尚未加载</div></details>`}
function card(r){return `<article class="video-card" data-scope="${esc(r.head_scope)}"><h4>${esc(r.head_label)} · ${r.head_count} heads</h4><video controls muted playsinline preload="none" data-src="${esc(media('ablation',r))}"></video><div class="caption"><span class="mono">${esc(r.mask_mode)}</span><br>R tokens ${r.token_count} · modified events ${r.modified_head_events.toLocaleString()}</div>${r.dose_ready?doseBox(r):''}</article>`}
async function loadDose(el){if(el.dataset.loaded)return;el.dataset.loaded='1';const p=new URLSearchParams({case:el.dataset.case,seed:el.dataset.seed,variant:el.dataset.variant}),box=el.querySelector('.dose');try{const d=await fetch(`${api}/dose?${p}`,{cache:'no-store'}).then(r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()});const rows=[['attention_mass','Attention mass/head'],['attention_mass_query_sum','Attention mass query-sum/head'],['removed_value_norm','Removed A·V norm/head'],['removed_value_norm_query_sum','Removed A·V query-sum/head'],['removed_to_output_ratio','Removed/output ratio'],['target_query_count','Target Query count']];box.innerHTML=rows.map(([k,l])=>`<b>${l}</b>：mean ${num(d[k]?.mean)} · median ${num(d[k]?.median)} · P95 ${num(d[k]?.p95)} · N ${d[k]?.finite_count??0}<br>`).join('')}catch(e){box.textContent=`读取失败：${e}`;el.dataset.loaded=''}}
function renderStatic(){const ds=data.direction_order.map(k=>[k,data.directions[k]]);$('directionKey').innerHTML='<div class="head">名称</div><div class="head">谓词</div><div class="head">精确删除范围</div><div class="head">诊断含义</div>'+ds.map(([k,d])=>`<div><b>${esc(d.label)}</b></div><div class="mono">${esc(d.predicate)}</div><div>${esc(d.meaning)}</div><div>${k==='future'?'因果方向候选':k==='past'?'反向时间控制':k==='same'?'同帧关系':'总效应基准'}</div>`).join('')}
function render(){const p=data.progress,s=selectedSample();$('metrics').innerHTML=metric('全部生成',`${p.completed}/${p.expected}`,pct(p.completed,p.expected))+metric('设计规模',`${data.case_count} cases × ${data.seed_count} seeds`,'9 case-seed')+metric('Dose',`${data.dose_count}/${p.completed}`,'已写入')+metric('结果报告',Object.values(data.metrics).reduce((a,b)=>a+b,0),'当前文件数');$('progress').innerHTML=[...p.by_case,...p.by_head].map(progress).join('');$('baseline').innerHTML=s?.baseline_ready?`<video controls muted playsinline preload="metadata" src="${esc(media('baseline',s))}"></video><div class="caption">未消融 Baseline · ${esc(s.case)} · seed ${s.seed}</div>`:'<div class="empty">Baseline 不可用</div>';const rows=data.records.filter(r=>r.case===$('case').value&&String(r.seed)===$('seed').value&&r.target_key===$('target').value),hi=Object.fromEntries(data.head_scopes.map((x,i)=>[x.key,i]));$('gallery').innerHTML=data.flow_order.map(flow=>{const f=data.flows[flow],directionRows=data.direction_order.map(direction=>{const rs=rows.filter(r=>r.flow===flow&&r.direction===direction).sort((a,b)=>hi[a.head_scope]-hi[b.head_scope]);if(!rs.length)return'';const d=data.directions[direction];return `<div class="direction-row"><div class="direction-head"><h3>${esc(d.label)}</h3><span class="mono">${esc(d.predicate)}</span><span>${esc(d.meaning)}</span></div><div class="video-grid">${rs.map(card).join('')}</div></div>`}).join('');return directionRows?`<section class="flow-band"><div class="flow-head"><h2>${esc(f.id)} · ${esc(f.flow)}</h2><div class="formula">${esc(f.formula)}</div><div>${esc(f.diagnosis)}</div></div>${directionRows}</section>`:''}).join('')||'<div class="empty">当前 target 尚无完成视频。</div>';$('metricStatus').innerHTML=`Fast <b>${data.metrics.fast}</b> · Trajectory <b>${data.metrics.trajectory}</b> · Survival <b>${data.metrics.survival}</b> · Complete25 <b>${data.metrics.complete25}</b>`;document.querySelectorAll('details[data-variant]').forEach(x=>x.addEventListener('toggle',()=>{if(x.open)loadDose(x)}));lazy();const u=new URL(location.href);u.searchParams.set('case',$('case').value);u.searchParams.set('seed',$('seed').value);u.searchParams.set('target',$('target').value);history.replaceState(null,'',u);$('status').textContent=`${p.completed}/${p.expected} · ${pct(p.completed,p.expected)} · 更新 ${new Date(data.generated_at_utc).toLocaleTimeString()}`}
async function load(first=false){const next=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json()),nextSig=`${next.progress.completed}:${Object.values(next.metrics).join(':')}`;data=next;if(first){renderStatic();filters(true);render()}else{const c=$('case').value,s=$('seed').value,t=$('target').value;filters(false);if([...$('case').options].some(x=>x.value===c))$('case').value=c;filters(false);if([...$('seed').options].some(x=>x.value===s))$('seed').value=s;filters(false);if([...$('target').options].some(x=>x.value===t))$('target').value=t;if(nextSig!==sig)render()}sig=nextSig}
$('case').addEventListener('change',()=>{filters(false);render()});$('seed').addEventListener('change',()=>{filters(false);render()});$('target').addEventListener('change',render);$('refresh').addEventListener('click',()=>load(false));$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load(true).catch(e=>$('status').textContent=`读取失败：${e}`);setInterval(()=>load(false).catch(()=>{}),30000);
</script></body></html>'''
