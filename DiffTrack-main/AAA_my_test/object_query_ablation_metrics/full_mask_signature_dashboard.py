#!/usr/bin/env python3
"""Read-only comparison dashboard for the full-mask signature pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/full_mask_signature_pilot_v1"
)
OVERLAY_CATALOG = ROOT / "overlay_catalog.json"
STAGE3_ROOT = ROOT.parent / "stage3_discovery_videos"
TEMPORAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
    "visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
)
SEED = 47326
MODES = ("self_only", "incoming_only", "outgoing_only")
SCOPES = ("top100", "bottom100")


def _units() -> list[dict[str, Any]]:
    try:
        raw = json.loads(OVERLAY_CATALOG.read_text(encoding="utf-8"))["units"]
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return []
    result = []
    for unit in raw:
        partition = unit.get("partition") or {}
        result.append({
            "case": str(unit["case"]),
            "seed": int(unit["seed"]),
            "object_names": partition.get("object_names", []),
            "signature_token_counts": partition.get("signature_token_counts", {}),
            "shared_signature_token_counts": partition.get("shared_signature_token_counts", {}),
            "union_token_count": partition.get("union_token_count", 0),
        })
    return result


def _new_path(case: str, scope: str, mode: str) -> Path:
    return ROOT / case / f"seed_{SEED:05d}" / f"fullmask_signature__{mode}__{scope}_s039r3350/generated.mp4"


def _old_path(case: str, scope: str, mode: str) -> Path:
    if case == "0613pybullet_sample_001460_w002":
        return TEMPORAL_ROOT / case / f"seed_{SEED:05d}" / f"all_objects__all_objects__{mode}__{scope}_s039r3350/generated.mp4"
    return STAGE3_ROOT / case / f"seed_{SEED:05d}" / f"all_objects__all_objects__{mode}__{scope}_s039r3350_expv1/generated.mp4"


def catalog() -> dict[str, Any]:
    units = _units()
    variants = []
    for unit in units:
        case = unit["case"]
        for scope in SCOPES:
            for mode in MODES:
                new = _new_path(case, scope, mode)
                old = _old_path(case, scope, mode)
                manifest = new.parent / "manifest.json"
                audit = None
                if manifest.is_file():
                    try:
                        payload = json.loads(manifest.read_text(encoding="utf-8"))
                        audit = {
                            "selected_head_count": payload.get("selected_head_count"),
                            "modified_head_events": (payload.get("runtime_audit") or {}).get("modified_head_events"),
                            "dose_finite_events": (payload.get("runtime_audit") or {}).get("dose_finite_events"),
                        }
                    except (OSError, ValueError, json.JSONDecodeError):
                        audit = None
                variants.append({
                    "case": case,
                    "scope": scope,
                    "mode": mode,
                    "old_sparse_ready": old.is_file(),
                    "full_mask_ready": new.is_file(),
                    "audit": audit,
                })
    return {
        "schema_version": 1,
        "seed": SEED,
        "units": units,
        "variants": variants,
        "generated": sum(row["full_mask_ready"] for row in variants),
        "expected": len(variants),
    }


def asset(kind: str, case: str, scope: str = "", mode: str = "") -> Path | None:
    known_cases = {unit["case"] for unit in _units()}
    if case not in known_cases:
        return None
    if kind == "baseline":
        try:
            raw = json.loads(OVERLAY_CATALOG.read_text(encoding="utf-8"))["units"]
            value = next(row["baseline_video"] for row in raw if row["case"] == case)
            candidate = Path(str(value))
        except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError):
            return None
    elif kind == "overlay":
        candidate = ROOT / "overlays" / case / f"seed_{SEED:05d}/full_mask_signatures.mp4"
    elif kind in {"old_sparse", "full_mask"} and scope in SCOPES and mode in MODES:
        candidate = _old_path(case, scope, mode) if kind == "old_sparse" else _new_path(case, scope, mode)
    else:
        return None
    return candidate if candidate.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整对象区域消融 · Full-mask Signature</title><style>
:root{--ink:#0b1117;--paper:#f2efe7;--card:#fffdf7;--line:#b8b4aa;--red:#d62e3d;--blue:#075b83;--gold:#b06b00;--muted:#5d656b;--green:#087d58}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Sans SC","IBM Plex Sans",system-ui,sans-serif}header{padding:24px clamp(18px,4vw,62px);background:var(--ink);color:#fff}.crumb{color:#9bdff6;font:12px ui-monospace,monospace}h1{font-size:clamp(30px,5vw,66px);line-height:.95;letter-spacing:-.05em;margin:18px 0 12px;max-width:1000px}.lead{max-width:980px;line-height:1.65;color:#cdd8df}.stamp{display:inline-block;border:1px solid #4c6473;padding:6px 9px;font:11px ui-monospace,monospace;color:#b9dbe8}main{width:min(1800px,calc(100% - 28px));margin:auto;padding:20px 0 80px}.definitions{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}.definition{border-top:5px solid var(--blue);background:var(--card);padding:15px;border-left:1px solid var(--line);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.definition:nth-child(2){border-top-color:var(--gold)}.definition:nth-child(3){border-top-color:var(--red)}.definition h2{margin:0 0 7px;font-size:18px}.formula{font:12px/1.6 ui-monospace,monospace;background:#e8e4da;padding:8px;overflow-wrap:anywhere}.definition p{font-size:13px;line-height:1.55;color:var(--muted)}.warning{padding:13px 16px;border:1px solid #d59b48;background:#fff0cb;font-size:13px;line-height:1.6;margin:13px 0}.bar{display:flex;align-items:end;justify-content:space-between;gap:14px;flex-wrap:wrap;border:1px solid var(--line);padding:13px;background:var(--card)}.controls{display:flex;gap:8px;flex-wrap:wrap}.controls label{font:10px ui-monospace,monospace;color:var(--muted);display:grid;gap:4px}.controls select,.controls button{height:38px;border:1px solid #858078;background:#fffdf7;padding:0 10px;font-weight:700}.progress b{font:25px ui-monospace,monospace;color:var(--green)}.case-meta{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}.badge{border:1px solid var(--line);background:var(--card);padding:6px 9px;font:11px ui-monospace,monospace}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.video{border:1px solid var(--line);background:var(--card);min-width:0}.video header{padding:10px 12px;background:#e2ded3;color:var(--ink);display:flex;justify-content:space-between;font:11px ui-monospace,monospace}.video.new{border:2px solid var(--red)}.video.new header{background:#d62e3d;color:#fff}.video video{display:block;width:100%;aspect-ratio:1280/704;background:#14191e}.pending{display:grid;place-items:center;aspect-ratio:1280/704;color:var(--muted);font:12px ui-monospace,monospace;background:#ebe8e0}.overlay{margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:10px}.notes{border:1px solid var(--line);background:var(--card);padding:14px}.notes h3{margin-top:0}.notes li{margin:7px 0;font-size:13px;line-height:1.5}.audit{font:11px/1.55 ui-monospace,monospace;color:var(--muted)}@media(max-width:1050px){.definitions,.grid{grid-template-columns:1fr}.overlay{grid-template-columns:1fr}} </style></head><body><header><a class="crumb" href="/">← 返回 8092 总入口</a><h1>从八个点，扩展到完整对象。</h1><p class="lead">同一个 case、seed=47326、latest3350 head ranking 和 40-step 采样设置下，对比 Baseline、旧 sparse tube 消融与新 full SAM2-mask signature 消融。页面只加载当前选择的一组，避免视频矩阵阻塞浏览器。</p><span class="stamp">FULL MASK · DISJOINT SIGNATURES · NO SOFTMAX RENORMALIZATION</span></header><main><section class="definitions"><article class="definition"><h2>M1 · 对象内部自通信</h2><div class="formula">删除 Σ<sub>S≠0</sub> A[R<sub>S</sub>,R<sub>S</sub>]V[R<sub>S</sub>]</div><p>S 是精确成员签名：R_A、R_B、R_AB…各自成为独立 block；R_A↔R_B、R_A↔R_AB 等跨签名边保留。</p></article><article class="definition"><h2>M2 · 环境输入对象</h2><div class="formula">删除 A[R_all,C]V[C]</div><p>R_all 是所有完整对象 mask token 的并集，C=Ω\R_all。切断背景/其他非对象 token 对对象 Query 的输入。</p></article><article class="definition"><h2>M3 · 对象广播环境</h2><div class="formula">删除 A[C,R_all]V[R_all]</div><p>切断完整对象 K/V 对非对象 Query 的输出；对象区域内部所有 signature 之间的通信仍保留。</p></article></section><div class="warning"><b>控制变量边界：</b>Top/Bottom、Flow、case 与 seed 均严格固定；但旧 sparse 与新 full-mask 的 M1 不只是覆盖面积不同——旧 all_objects M1 删除稀疏并集 R→R（含跨对象边），新 M1 删除 signature 对角块并保留跨 signature 边。因此 M1 的旧/新差异不能单独归因于“点数变多”；M2/M3 的算子相同，主要变化是 R 的覆盖定义。</div><section class="bar"><div class="controls"><label>CASE<select id="case"></select></label><label>HEAD GROUP<select id="scope"><option value="top100">Top100</option><option value="bottom100">Bottom100</option></select></label><label>FLOW<select id="mode"><option value="self_only">M1 · R→R</option><option value="incoming_only">M2 · C→R</option><option value="outgoing_only">M3 · R→C</option></select></label><button id="play">同步播放</button><button id="pause">暂停</button></div><div class="progress"><b id="progress">0 / 18</b><br><span>full-mask videos ready</span></div></section><div id="meta" class="case-meta"></div><section id="videos" class="grid"></section><section class="overlay"><article class="video"><header><b>完整 Mask Signature Overlay</b><span>红色 = shared signature</span></header><video id="overlay" controls muted loop playsinline preload="metadata"></video></article><aside class="notes"><h3>怎么读这一组</h3><ul><li><b>Baseline</b>：同 case、同 seed 的无消融生成。</li><li><b>旧 sparse tube</b>：每对象 8 个 CoTracker 点量化成 token tube。</li><li><b>新 full mask</b>：每帧冻结 SAM2 mask 覆盖到 22×40 token 网格；边界只要相交即纳入。</li><li>红色格是多个对象共享的 token，例如 R_AB；新 M1 把它作为独立 self block。</li></ul><div id="audit" class="audit"></div></aside></section></main><script>
const api='/api/object-query-full-mask-signature',E=id=>document.getElementById(id),esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data;
const url=(kind,extra={})=>`${api}/asset?${new URLSearchParams({kind,case:E('case').value,scope:E('scope').value,mode:E('mode').value,...extra})}`;
function card(kind,title,ready,accent=''){return `<article class="video ${accent}"><header><b>${title}</b><span>${ready?'READY':'PENDING'}</span></header>${ready?`<video data-sync controls muted loop playsinline preload="metadata" src="${url(kind)}"></video>`:'<div class="pending">尚未生成</div>'}</article>`}
function render(){const c=E('case').value,s=E('scope').value,m=E('mode').value,u=data.units.find(x=>x.case===c),v=data.variants.find(x=>x.case===c&&x.scope===s&&x.mode===m);E('meta').innerHTML=[`objects ${u.object_names.length}`,`union ${u.union_token_count} tokens`,...Object.entries(u.signature_token_counts).map(([k,n])=>`${k}: ${n}`)].map(x=>`<span class="badge">${esc(x)}</span>`).join('');E('videos').innerHTML=card('baseline','BASELINE · no ablation',true)+card('old_sparse','旧 Sparse Tube',v.old_sparse_ready)+card('full_mask','新 Full Mask Signature',v.full_mask_ready,'new');const ov=E('overlay');ov.src=url('overlay');E('audit').textContent=`shared: ${JSON.stringify(u.shared_signature_token_counts)} · runtime audit: ${JSON.stringify(v.audit||'pending')}`;const q=new URL(location.href).searchParams;q.set('case',c);q.set('scope',s);q.set('mode',m);history.replaceState(null,'',location.pathname+'?'+q)}
function sync(){const vs=[...document.querySelectorAll('video')];vs.forEach(v=>v.currentTime=0);Promise.allSettled(vs.map(v=>v.play()))}function pause(){document.querySelectorAll('video').forEach(v=>v.pause())}
async function load(){data=await fetch(api+'/catalog',{cache:'no-store'}).then(r=>r.json());E('progress').textContent=`${data.generated} / ${data.expected}`;E('case').innerHTML=data.units.map(u=>`<option value="${esc(u.case)}">${esc(u.case)}</option>`).join('');const q=new URL(location.href).searchParams;if(data.units.some(u=>u.case===q.get('case')))E('case').value=q.get('case');if(['top100','bottom100'].includes(q.get('scope')))E('scope').value=q.get('scope');if(['self_only','incoming_only','outgoing_only'].includes(q.get('mode')))E('mode').value=q.get('mode');render()}
['case','scope','mode'].forEach(x=>E(x).onchange=render);E('play').onclick=sync;E('pause').onclick=pause;load();
</script></body></html>'''
