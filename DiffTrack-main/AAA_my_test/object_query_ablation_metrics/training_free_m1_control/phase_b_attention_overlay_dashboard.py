#!/usr/bin/env python3
"""Dashboard for the seed-90094 Phase-B Top100 attention overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_direct_enhancement_v2/"
    "seed90094_top100_fixed_f04_trajectory_overlays"
)
WINDOWS = ("all40", "first10", "first20", "last20")


def _payload(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def catalog() -> dict[str, Any]:
    result = _payload(ROOT / "overlay_manifest.json")
    if result:
        return result
    protocol = _payload(ROOT / "protocol.json")
    return {
        "protocol": protocol.get("protocol", "phase_b_seed90094_top100_fixed_f04_trajectory_overlay_v2"),
        "case": "0613pybullet_sample_001460_w002",
        "seed": 90094,
        "default_window": "all40",
        "windows": [
            {"id": "all40", "label": "40 Steps Mean · S000-S039"},
            {"id": "first10", "label": "First10 · S000-S009"},
            {"id": "first20", "label": "First20 · S000-S019"},
            {"id": "last20", "label": "Last20 · S020-S039"},
        ],
        "anchor_pixel_frames": list(range(0, 49, 4)),
        "query_latent_frame": protocol.get("query_latent_frame", 1),
        "query_pixel_frame": protocol.get("query_pixel_frame", 4),
        "query_token_count": protocol.get("query_token_count", 0),
        "records": [
            {
                "variant_id": str(row.get("id")),
                "label": str(row.get("label")),
                "token_source": str(row.get("token_source")),
                "alpha": row.get("alpha"),
                "source_video": str(row.get("video")),
                "ready": False,
            }
            for row in protocol.get("variants", [])
        ],
    }


def asset(kind: str, variant_id: str, window: str = "", latent: int = -1) -> Path | None:
    data = catalog()
    record = next(
        (row for row in data.get("records", []) if row.get("variant_id") == variant_id),
        None,
    )
    if not isinstance(record, dict):
        return None
    if kind == "video":
        path = Path(str(record.get("display_video") or record.get("source_video") or ""))
        return path if path.is_file() else None
    if kind == "source_video":
        path = Path(str(record.get("source_video") or ""))
        return path if path.is_file() else None
    if kind != "overlay" or window not in WINDOWS or latent not in range(13):
        return None
    names = (record.get("images") or {}).get(window) or []
    if len(names) != 13:
        return None
    name = str(names[latent])
    if Path(name).name != name or not name.endswith(".jpg"):
        return None
    path = ROOT / "overlays" / window / variant_id / name
    return path if path.is_file() else None


def page() -> str:
    return PAGE


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phase-B Top100 Object Query Trajectory</title><style>
:root{--bg:#e9eeec;--paper:#fbfcf8;--ink:#14272d;--muted:#627278;--line:#9daca8;--navy:#0f313b;--cyan:#18838c;--orange:#d06c3d;--violet:#74539b}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#0f313b0b 1px,transparent 1px),linear-gradient(#0f313b0b 1px,transparent 1px),var(--bg);background-size:24px 24px;font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif}a{color:var(--cyan);font-weight:900}header,main,footer{width:min(2460px,calc(100% - 24px));margin:auto}header{padding:20px 0 13px}.eyebrow{margin-top:18px;color:var(--orange);font:900 11px ui-monospace,monospace;letter-spacing:.16em}h1,h2,h3{font-family:"Arial Black","Noto Sans CJK SC",sans-serif}h1{margin:7px 0 10px;font-size:clamp(38px,5vw,68px);line-height:.94;letter-spacing:-.045em}.lead{max-width:1260px;margin:0;color:#31464c;font-size:16px;line-height:1.65}.formula{margin:15px 0;padding:12px 15px;background:var(--navy);color:#edf7f4;font:12px/1.7 ui-monospace,monospace;overflow:auto}.toolbar{position:sticky;top:0;z-index:10;display:flex;gap:10px;align-items:end;flex-wrap:wrap;padding:11px;border:1px solid var(--line);background:#e9eeecef;backdrop-filter:blur(9px)}label{display:grid;gap:4px;color:var(--muted);font:900 10px ui-monospace,monospace}select,button{min-height:38px;padding:7px 11px;border:1px solid var(--line);background:white;color:var(--ink);font-weight:900}.status{margin-left:auto;font:800 11px ui-monospace,monospace}.notes{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:15px 0}.notes article{padding:12px;border:1px solid var(--line);background:var(--paper)}.notes b{display:block;margin-bottom:4px}.notes p{margin:0;color:var(--muted);font-size:12px;line-height:1.5}.row{margin:14px 0 22px;border:1px solid var(--line);background:var(--paper)}.row-head{display:grid;grid-template-columns:minmax(260px,.55fr) minmax(420px,1.45fr);gap:18px;align-items:end;padding:12px;border-bottom:1px solid var(--line)}.row-head h2{margin:0;font-size:21px}.row-head p{margin:3px 0 0;color:var(--muted);font:11px/1.45 ui-monospace,monospace}.facts{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.facts span{padding:4px 6px;background:#e7eeeb;font:10px ui-monospace,monospace}.row.sparse{border-top:7px solid var(--cyan)}.row.full{border-top:7px solid var(--violet)}.row.baseline{border-top:7px solid #68787e}.strip{display:grid;grid-template-columns:300px repeat(13,242px);gap:8px;align-items:start;padding:9px;overflow-x:auto}.video-card,.heat{margin:0}.video-card{position:sticky;left:9px;z-index:3;padding:7px;background:var(--paper);box-shadow:8px 0 12px #14272d18}.video-card video{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#091418}.video-card figcaption,.heat figcaption{margin-bottom:5px;font:900 10px ui-monospace,monospace}.heat img{display:block;width:242px;aspect-ratio:1280/704;object-fit:cover;background:#10191c;border:1px solid #34494e}.pending{padding:70px 20px;text-align:center;color:var(--muted);border:1px dashed var(--line)}footer{padding:8px 0 40px;color:var(--muted);font-size:11px}@media(max-width:900px){header,main,footer{width:calc(100% - 12px)}.notes{grid-template-columns:1fr 1fr}.row-head{grid-template-columns:1fr}.status{width:100%;margin-left:0}.strip{grid-template-columns:250px repeat(13,215px)}.heat img{width:215px}.video-card{position:static}}@media(max-width:560px){.notes{grid-template-columns:1fr}}
</style></head><body><header><a href="/">← 返回 8092 总入口</a> · <a href="/training-free-m1-phase-bd?v=2">返回 Phase B/D 视频页</a><div class="eyebrow">0613PYBULLET_SAMPLE_001460_W002 / SEED 90094 / LATEST3350 TOP100</div><h1>Fixed-F04 Object Query<br>13-Latent Trajectory</h1><p class="lead">每行是一组完整视频结果：Baseline、Sparse 8-point α=0.1/0.25、SAM2 Full-mask α=0.1/0.25。固定 Baseline F04 / latent 1 的完整小球 SAM2 Query，对全部 K00–K12 求响应，并叠加到对应 RGB 帧 F00/F04/…/F48。</p><div id="formula" class="formula">读取精确计算定义…</div><div class="toolbar"><label>DENOISING AGGREGATION<select id="window"></select></label><button id="refresh">刷新产物</button><button id="replay">同步重播</button><span id="status" class="status">读取中…</span></div></header><main><section class="notes"><article><b>固定 Query</b><p>五行统一使用 Baseline F04 / latent 1 的 SAM2 object_A 完整 mask tokens；仅 F04 白框标出 Query。</p></article><article><b>Head / CFG 聚合</b><p>对固定 Query tokens 求和；latest3350 Top100 物理 layer-head 等权平均，再平均 conditional 与 unconditional。</p></article><article><b>跨帧轨迹</b><p>每列计算同一个 Q_F04→K_t；softmax 分母始终包含完整 13×22×40 Keys，之后才切出 K_t。</p></article><article><b>颜色可比性</b><p>同一去噪分段的五行与 13 帧共享一个 q99.5 色标；不按单图重新拉伸。</p></article></section><section id="content" class="pending">正在读取 capture / overlay 产物…</section></main><footer>热力图表示固定 F04 Query 的注意力概率响应，不等于 A@V 信息内容，也不直接表示生成质量。</footer><script>
const api='/api/training-free-m1-phase-b-attention',q=new URL(location.href).searchParams,$=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null,windowId=q.get('window')||'all40';
function asset(kind,row,latent=-1){return `${api}/asset?${new URLSearchParams({kind,variant_id:row.variant_id,window:windowId,latent})}`}
function render(){const allowed=data.windows.map(x=>x.id);if(!allowed.includes(windowId))windowId=data.default_window||'all40';$('window').innerHTML=data.windows.map(x=>`<option value="${esc(x.id)}">${esc(x.label)}</option>`).join('');$('window').value=windowId;const scale=data.shared_color_scale_q995?.[windowId];$('formula').textContent=`${data.formula||'H=mean attention'} · softmax key domain: ${data.softmax_key_domain||'all keys'} · shared q99.5 scale=${scale===undefined?'pending':Number(scale).toExponential(4)}`;const ready=data.records.filter(x=>x.ready).length;$('status').textContent=`${ready}/${data.records.length||5} rows ready · ${data.case} · seed ${data.seed}`;$('content').className='';$('content').innerHTML=data.records.map(row=>{const cls=row.variant_id==='baseline'?'baseline':row.token_source==='sam2_full_mask'?'full':'sparse',alpha=row.alpha===null||row.alpha===undefined?'0':row.alpha,frames=data.anchor_pixel_frames||Array.from({length:13},(_,i)=>i*4),heat=frames.map((frame,i)=>`<figure class="heat"><figcaption>K${String(i).padStart(2,'0')} · F${String(frame).padStart(2,'0')} · Q=F${String(data.query_pixel_frame??4).padStart(2,'0')} · |R_q|=${data.query_token_count??'—'}</figcaption>${row.ready?`<img loading="lazy" src="${asset('overlay',row,i)}">`:'<div class="pending">PENDING</div>'}</figure>`).join(''),mae=row.replay_audit?.decoded_frame_mae_0_255;return `<article class="row ${cls}"><div class="row-head"><div><h2>${esc(row.label||row.variant_id)}</h2><div class="facts"><span>${esc(row.token_source)}</span><span>α=${esc(alpha)}</span><span>Top100</span><span>Q=F04 fixed</span><span>CFG both</span>${mae===undefined?'':`<span>replay↔history MAE=${Number(mae).toFixed(3)}/255</span>`}</div></div><p>${row.variant_id==='baseline'?'未干预控制组。':row.token_source==='sam2_full_mask'?'生成增强使用逐帧完整 SAM2 object_A token。':'生成增强使用 8 个 CoTracker 稀疏点。'} 测量统一固定 F04 Full-mask Query，并观察其在 K00–K12 的跨帧响应；Attention 与 overlay 来自同一次 replay。</p></div><div class="strip"><figure class="video-card"><figcaption>Attention capture replay</figcaption><video controls muted loop playsinline preload="metadata" src="${asset('video',row)}"></video></figure>${heat}</div></article>`}).join('')||'<div class="pending">尚无协议记录</div>';q.set('window',windowId);history.replaceState(null,'',`${location.pathname}?${q}`)}
async function load(){data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());render()}$('window').addEventListener('change',e=>{windowId=e.target.value;render()});$('refresh').addEventListener('click',()=>load().catch(e=>$('status').textContent=`读取失败：${e}`));$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load().catch(e=>$('status').textContent=`读取失败：${e}`);
</script></body></html>'''
