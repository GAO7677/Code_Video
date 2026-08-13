#!/usr/bin/env python3
"""Interactive audited token-communication viewer for the Top100-M1 pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/"
    "training_free_top100_m1_guidance_v1/token_communication_overlays"
)
CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_000331_w001",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
)
SEED = 47326


def _read(case: str) -> dict[str, Any]:
    if case not in CASES:
        return {}
    path = ROOT / case / f"seed_{SEED:05d}" / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def catalog() -> dict[str, Any]:
    rows = []
    for case in CASES:
        manifest = _read(case)
        rows.append(
            {
                "case": case,
                "ready": bool(manifest),
                "seed": SEED,
                "region": manifest.get("region", "object_A"),
                "head_scope": manifest.get("head_scope", "latest3350_top100"),
                "selected_head_count": manifest.get("selected_head_count", 100),
                "m1_time_scope": manifest.get("m1_time_scope", "all_time"),
                "grid": manifest.get("grid", {}),
                "communication": manifest.get("communication", {}),
                "anchors": manifest.get("anchors", []),
            }
        )
    return {
        "protocol": "top100_m1_token_communication_overlay_v1",
        "rows": rows,
        "ready": sum(bool(row["ready"]) for row in rows),
        "total": len(rows),
    }


def asset(kind: str, case: str, anchor: int = -1) -> Path | None:
    manifest = _read(case)
    if not manifest:
        return None
    if kind in {"baseline", "guidance", "overlay_video"}:
        key = {"baseline": "baseline_video", "guidance": "guidance_video"}.get(
            kind, "overlay_video"
        )
        path = Path(str(manifest.get(key, "")))
    elif kind in {"query", "key"} and 0 <= anchor < len(manifest["anchors"]):
        filename = manifest["anchors"][anchor][f"{kind}_image"]
        path = ROOT / case / f"seed_{SEED:05d}" / filename
    else:
        return None
    return path if path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top100 M1 Token Communication</title><style>
:root{--paper:#e7edf0;--white:#f8fbfc;--ink:#10212a;--muted:#5e7079;--line:#9dafb8;--matrix:#092934;--query:#f36b35;--key:#00a8b5;--violet:#7656d8;--same:#ed9c35;--past:#3987b7;--ok:#28745f}*{box-sizing:border-box}body{margin:0;background:linear-gradient(90deg,#0a58650b 1px,transparent 1px),linear-gradient(#0a58650b 1px,transparent 1px),var(--paper);background-size:24px 24px;color:var(--ink);font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif}button,select,input{font:inherit}a{color:#126879}header,main{width:min(1780px,calc(100% - 28px));margin:auto}header{padding:24px 0 16px}.eyebrow{margin-top:22px;color:var(--query);font:800 12px ui-monospace,monospace;letter-spacing:.18em}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{max-width:1200px;margin:8px 0 12px;font-size:clamp(42px,6.6vw,92px);line-height:.9;letter-spacing:-.055em}.lead{max-width:1150px;font-size:16px;line-height:1.65}.equation{display:inline-block;padding:10px 13px;background:var(--matrix);color:#eaf5f6;font:12px ui-monospace,monospace}.warning{margin-top:12px;padding:10px 13px;border-left:5px solid var(--query);background:#fff7f1;font-size:13px;line-height:1.55}.toolbar{position:sticky;top:0;z-index:5;display:grid;grid-template-columns:1.5fr 1fr auto;gap:14px;align-items:end;margin:10px 0 18px;padding:12px;background:#e7edf0ed;border:1px solid var(--line);backdrop-filter:blur(9px)}label{display:block;margin-bottom:5px;color:var(--muted);font:800 10px ui-monospace,monospace;letter-spacing:.1em}select,input[type=range]{width:100%}.frame-readout{min-width:170px;padding:10px 14px;background:var(--matrix);color:white;font:800 14px ui-monospace,monospace}.layout{display:grid;grid-template-columns:minmax(440px,.92fr) minmax(540px,1.08fr);gap:16px}.panel{padding:14px;border:1px solid var(--line);background:#f8fbfce8;box-shadow:7px 7px 0 #0b586312}.panel h2{margin:0 0 5px;font-size:22px}.panel-note{margin:0 0 12px;color:var(--muted);line-height:1.5;font-size:13px}.query-image{display:block;width:100%;background:#071116;border-top:7px solid var(--query)}.token-list{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.token-list span{padding:5px 7px;background:#fde5da;color:#8b361d;font:700 10px ui-monospace,monospace}.matrix-wrap{overflow:auto;background:var(--matrix);padding:12px}.axis{color:#bed3d8;font:700 10px ui-monospace,monospace}.matrix{display:grid;grid-template-columns:44px repeat(13,minmax(25px,1fr));gap:3px;min-width:520px}.mcell{aspect-ratio:1;border:0;color:white;font:800 9px ui-monospace,monospace;cursor:pointer;opacity:.78}.mcell:hover,.mcell:focus-visible{opacity:1;outline:3px solid white;outline-offset:1px}.mcell.future{background:var(--violet)}.mcell.same{background:var(--same)}.mcell.past{background:var(--past)}.mcell.selected{opacity:1;box-shadow:inset 0 0 0 3px white}.corner,.rowlab,.collab{display:grid;place-items:center;color:#cae0e4;font:800 9px ui-monospace,monospace}.legend{display:flex;gap:12px;flex-wrap:wrap;margin-top:10px;color:var(--muted);font-size:11px}.legend i{display:inline-block;width:10px;height:10px;margin-right:4px}.keys{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:18px 0}.key-card{border:1px solid var(--line);background:var(--white);overflow:hidden}.key-card.same{box-shadow:0 0 0 4px var(--same)}.key-card img{display:block;width:100%;aspect-ratio:16/9;object-fit:cover;border-top:5px solid var(--key)}.key-meta{padding:7px}.key-meta b{display:block;font:800 11px ui-monospace,monospace}.key-meta small{color:var(--muted)}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 44px}.video-card{padding:10px;border:1px solid var(--line);background:var(--white)}.video-card h3{margin:0 0 8px;font-size:15px}.video-card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#081317}.video-card:first-child{border-top:7px solid var(--same)}.video-card:nth-child(2){border-top:7px solid #477989}.video-card:nth-child(3){border-top:7px solid var(--query)}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin:14px 0}.fact{padding:10px;background:#dfe8eb}.fact b{display:block;font:900 19px ui-monospace,monospace}.fact span{font-size:11px;color:var(--muted)}.empty{padding:40px;background:white;border:1px solid var(--line)}footer{padding:0 0 50px;color:var(--muted);font-size:11px}@media(max-width:1000px){.layout,.videos{grid-template-columns:1fr}.keys{grid-template-columns:repeat(3,1fr)}}@media(max-width:650px){header,main{width:calc(100% - 12px)}.toolbar{grid-template-columns:1fr}.keys{grid-template-columns:repeat(2,1fr)}.facts{grid-template-columns:repeat(2,1fr)}}
</style></head><body><header><a href="/">← 8092 总入口</a> · <a href="/top100-m1-guidance-pilot?v=1">Baseline × Guidance</a><div class="eyebrow">AUDITED TOKEN CELLS / M1 ALL-TIME / LATEST3350 TOP100</div><h1>一条对象 Tube，<br>169 条被切断的时间通信</h1><p class="lead">选择一个 Query 时刻。橙色大图展示该帧作为接收端的真实对象 token；青色缩略图展示 13 个 K/V 来源时刻的真实 token。矩阵每一行对应一个 Query，每一列对应一个被删除的 K/V 来源。</p><div class="equation">Y[R_tq,h] ← Y[R_tq,h] − Σ_tk A[R_tq,R_tk,h]V[R_tk,h] &nbsp; h ∈ latest3350 Top100</div><div class="warning"><b>严格时间语义：</b>这是 13 个 latent 时刻 F00,F04,…,F48，不是 49 个独立 attention 时刻。坐标来自本次 guidance manifest 的运行审计；每格对应视频上的 32×32 px latent token。其他帧的 K/V 只画在它自己的视频帧上，不投影到当前 Query 帧。</div></header><main><section class="toolbar"><div><label for="case">CASE</label><select id="case"></select></div><div><label for="anchor">QUERY LATENT FRAME</label><input id="anchor" type="range" min="0" max="12" value="0" step="1"></div><div id="readout" class="frame-readout">Q F00 ← 13 K/V frames</div></section><div id="app"><div class="empty">读取 token 审计数据…</div></div><footer>橙色 = Query receiver；青色 = K/V source；紫色 = 历史 K/V → 未来 Query；金色 = 同帧；蓝色 = 未来 K/V → 过去 Query。位置相同不代表不同 head 的响应值相同；Top100 heads 共享这些 token 对，但各自 A·V 数值不同。</footer></main><script>
const API='/api/top100-m1-token-communication',esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let DATA=[],row=null,q=0;
const url=(kind,anchor=-1)=>`${API}/asset?${new URLSearchParams({kind,case:row.case,anchor})}`;
function direction(q,k){return k<q?'future':k===q?'same':'past'}function directionText(q,k){return k<q?'历史→未来':k===q?'同帧':'未来→过去'}
function matrix(){let h='<div class="matrix"><div class="corner">Q↓ K→</div>';for(let k=0;k<13;k++)h+=`<div class="collab">F${String(k*4).padStart(2,'0')}</div>`;for(let i=0;i<13;i++){h+=`<div class="rowlab">F${String(i*4).padStart(2,'0')}</div>`;for(let k=0;k<13;k++){const d=direction(i,k);h+=`<button class="mcell ${d} ${i===q?'selected':''}" data-q="${i}" title="Q F${i*4} ← K/V F${k*4} · ${directionText(i,k)}">${i===q?'×':'·'}</button>`}}return h+'</div>'}
function render(){if(!row||!row.ready){document.querySelector('#app').innerHTML='<div class="empty">该 case 的 overlay 尚未生成。</div>';return}const a=row.anchors[q],tokens=a.tokens.map(t=>`<span>#${t.token_index} · (y${t.grid_y},x${t.grid_x}) · [${t.pixel_bbox_xyxy.join(',')}]</span>`).join('');document.querySelector('#readout').textContent=`Q F${String(a.pixel_frame).padStart(2,'0')} ← 13 K/V frames`;document.querySelector('#app').innerHTML=`<div class="facts"><div class="fact"><b>13</b><span>latent Query frames</span></div><div class="fact"><b>169</b><span>active tq←tk pairs</span></div><div class="fact"><b>${a.token_count}</b><span>current Query tokens</span></div><div class="fact"><b>100×40</b><span>heads × denoising steps</span></div></div><div class="layout"><section class="panel"><h2>当前 Query：R_F${String(a.pixel_frame).padStart(2,'0')}</h2><p class="panel-note">橙色单元格是当前帧中被修改的 Query rows；坐标为 (grid y,x) 与像素 bbox [x0,y0,x1,y1]。</p><img class="query-image" src="${url('query',q)}" alt="Query token overlay"><div class="token-list">${tokens}</div></section><section class="panel"><h2>13 × 13 删除通信矩阵</h2><p class="panel-note">当前高亮行表示这个 Query 接收端；该行 13 列全部被删除。点击任意矩阵行切换 Query。</p><div class="matrix-wrap">${matrix()}</div><div class="legend"><span><i style="background:var(--violet)"></i>历史 K/V → 未来 Query</span><span><i style="background:var(--same)"></i>同帧</span><span><i style="background:var(--past)"></i>未来 K/V → 过去 Query</span></div></section></div><section class="keys">${row.anchors.map((k,i)=>`<article class="key-card ${i===q?'same':''}"><img loading="lazy" src="${url('key',i)}" alt="K/V F${k.pixel_frame} token overlay"><div class="key-meta"><b>K/V R_F${String(k.pixel_frame).padStart(2,'0')} → Q F${String(a.pixel_frame).padStart(2,'0')}</b><small>${directionText(q,i)} · ${k.token_count} tokens</small></div></article>`).join('')}</section><section class="videos"><article class="video-card"><h3>13-anchor token overlay</h3><video controls muted loop playsinline preload="metadata" src="${url('overlay_video')}"></video></article><article class="video-card"><h3>同 seed Baseline · tube 来源</h3><video controls muted loop playsinline preload="none" src="${url('baseline')}"></video></article><article class="video-card"><h3>Top100-M1 Guidance · λ=0.5</h3><video controls muted loop playsinline preload="none" src="${url('guidance')}"></video></article></section>`;document.querySelectorAll('[data-q]').forEach(b=>b.onclick=()=>{q=+b.dataset.q;document.querySelector('#anchor').value=q;render()})}
function selectCase(){row=DATA[+document.querySelector('#case').value];q=0;document.querySelector('#anchor').value=0;render()}async function load(){const d=await fetch(`${API}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());DATA=d.rows;const s=document.querySelector('#case');s.innerHTML=DATA.map((r,i)=>`<option value="${i}">${esc(r.case)}</option>`).join('');selectCase()}document.querySelector('#case').onchange=selectCase;document.querySelector('#anchor').oninput=e=>{q=+e.target.value;render()};document.addEventListener('keydown',e=>{if(!row||!['ArrowLeft','ArrowRight'].includes(e.key))return;q=Math.max(0,Math.min(12,q+(e.key==='ArrowRight'?1:-1)));document.querySelector('#anchor').value=q;render()});load().catch(e=>document.querySelector('#app').innerHTML=`<div class="empty">读取失败：${esc(e)}</div>`);
</script></body></html>'''
