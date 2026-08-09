"""Standalone dashboard component mounted by the existing port-8092 server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import OUTPUT_ROOT


REPORT_PATH = OUTPUT_ROOT / "report.json"

OPERATOR_DEFINITIONS = {
    "self_only": {
        "id": "M1", "blocks": "S=0", "flow": "R K/V ─X→ R Query",
        "formula": "Y′R=I·VC；Y′C=O·VR+B·VC",
        "meaning": "只删除 R 内部 Value 对 R Query 的贡献。",
    },
    "incoming_only": {
        "id": "M2", "blocks": "I=0", "flow": "C K/V ─X→ R Query",
        "formula": "Y′R=S·VR；Y′C=O·VR+B·VC",
        "meaning": "只切断背景、另一对象及 R 外 token 向 R 接收端输入。",
    },
    "outgoing_only": {
        "id": "M3", "blocks": "O=0", "flow": "R K/V ─X→ C Query",
        "formula": "Y′R=S·VR+I·VC；Y′C=B·VC",
        "meaning": "R 自身读取不变，只切断 R Value 向其余 token 广播。",
    },
    "query_row": {
        "id": "M4", "blocks": "S=I=0", "flow": "全部 K/V ─X→ R Query",
        "formula": "Y′R=0；Y′C=O·VR+B·VC",
        "meaning": "将选中 head 对 R Query 的整行 A@V 更新置零；不删除残差中的 R token。",
    },
    "key_value_column": {
        "id": "M5", "blocks": "S=O=0", "flow": "R Value ─X→ 全部 Query",
        "formula": "Y′R=I·VC；Y′C=B·VC",
        "meaning": "保持原 softmax A，只删除 R 的 Value 贡献且不重归一化；等价于只令 VR=0。",
    },
    "cross_boundary": {
        "id": "M6", "blocks": "I=O=0", "flow": "C→R 与 R→C 均切断",
        "formula": "Y′R=S·VR；Y′C=B·VC",
        "meaning": "隔离 R/C 边界，但保留 R 内部与 C 内部通信。",
    },
    "row_and_column": {
        "id": "M7", "blocks": "S=I=O=0", "flow": "所有涉及 R 的流向均切断",
        "formula": "Y′R=0；Y′C=B·VC",
        "meaning": "选中 head 中 R 不再接收或发送；其他 heads、残差、FFN、cross-attention 仍保留。",
    },
    "literal_kv_zero": {
        "id": "C1", "blocks": "K′R=V′R=0", "flow": "R 列重新进入 softmax",
        "formula": "A′=softmax(QK′ᵀ/√d)；Y′=A′V′",
        "meaning": "R 列 logit 变为 0 但仍占概率质量；会重新路由注意力，计算上不等价于 M5。",
    },
}


def _prune(value: Any) -> Any:
    """Remove long per-frame arrays from the API; videos/images carry their audit trail."""
    if isinstance(value, list):
        return [_prune(item) for item in value]
    if not isinstance(value, dict):
        return value
    omitted = {
        "series", "candidate_contact_by_frame", "candidate_mask_gap_px",
        "dino_cosine_by_anchor", "dino_anchor_frames", "lpips_by_frame",
        "outside_object_lpips_by_frame", "per_frame_flow_epe_mean_px",
        "motion_profile_mean_magnitude_px", "gt_surface_distance_m",
    }
    return {key: _prune(item) for key, item in value.items() if key not in omitted}


def catalog() -> dict[str, Any]:
    if not REPORT_PATH.is_file():
        return {
            "ready": False,
            "reason": "指标报告仍在生成；轨迹与 SAM2 中间量已完成。",
            "output_root": str(OUTPUT_ROOT),
        }
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    payload = _prune(payload)
    payload["ready"] = True
    payload["operator_definitions"] = OPERATOR_DEFINITIONS
    payload["reference_note"] = (
        "Baseline 是同 seed 未消融生成视频；GT 列中中心/速度/接触使用 states.npz "
        "simulator GT，点轨迹、mask、外观、像素与光流使用 source render 前 49 帧。"
    )
    return payload


def asset(relative_path: str) -> Path | None:
    if not relative_path or Path(relative_path).is_absolute():
        return None
    root = OUTPUT_ROOT.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def input_video(video_id: str) -> Path | None:
    if not REPORT_PATH.is_file():
        return None
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if video_id == "baseline":
        path = Path(report["references"]["baseline"])
    elif video_id == "source_gt_video":
        path = Path(report["references"]["source_gt_video"])
    else:
        row = next((item for item in report.get("records", []) if item.get("id") == video_id), None)
        path = Path(row["assets"]["input_video"]) if row else Path("/__missing__")
    return path if path.is_file() else None


def page() -> str:
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Object Query Ablation Metrics</title><style>
:root{--paper:#eee8dc;--ink:#17241f;--line:#c3b59e;--card:#fffdf7;--deep:#173f36;--rust:#b54d35;--gold:#d19a38;--blue:#2d6674}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 3% 0,#df724a42,transparent 32rem),radial-gradient(circle at 97% 1%,#328e7a3c,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:9;padding:14px 22px;background:#eee8dcf3;border-bottom:1px solid var(--line);backdrop-filter:blur(12px)}header h1{margin:3px 0;font-size:clamp(28px,4.2vw,54px);line-height:1}.lead{margin:6px 0;max-width:1500px}.tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap}select,button{padding:8px 10px;border:1px solid var(--line);background:#fff;font-weight:900}.status,.mono{font:12px ui-monospace,SFMono-Regular,monospace}main{width:min(100% - 18px,2400px);margin:auto;padding:15px 0 65px}section{margin:13px 0;padding:14px;border:1px solid var(--line);border-radius:15px;background:#fffaf1e8;box-shadow:0 12px 30px #604b2e14}h2{margin:0 0 8px}.note{border-left:7px solid var(--gold)}.reference-grid,.row-grid,.percept-grid{display:grid;gap:10px}.reference-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.row-grid{grid-template-columns:repeat(5,minmax(0,1fr))}.percept-grid{grid-template-columns:repeat(4,minmax(0,1fr))}figure{margin:0;padding:7px;border:1px solid #d7cbb8;background:#fff;min-width:0}video,img{display:block;width:100%;background:#101715}video{aspect-ratio:1280/704}figcaption{padding:7px 2px 1px;font-weight:800;overflow-wrap:anywhere}.protocol-row{border-top:8px solid var(--deep)}.protocol-row.tube{border-top-color:var(--rust)}.formula{padding:9px;margin:8px 0;background:#f1eadc;border-left:4px solid var(--gold);line-height:1.55}.pill{display:inline-block;padding:4px 8px;margin:2px;border:1px solid var(--line);border-radius:99px;background:#fff;font:11px ui-monospace,monospace}.scroll{overflow:auto;border:1px solid var(--line);background:#fff}table{border-collapse:collapse;width:100%;min-width:1250px;font-variant-numeric:tabular-nums}th,td{padding:8px 9px;border-right:1px solid #ded5c6;border-bottom:1px solid #ded5c6;vertical-align:top}th{background:var(--deep);color:#fff;position:sticky;top:0}td.value{text-align:center;font-family:ui-monospace,monospace}.priority{font-weight:900}.P0{color:#9e2f26}.P1{color:#a36b08}.P2{color:#17645a}.P3{color:#56616b}.result-block{margin:14px 0}.result-scroll{overflow:auto;max-height:720px;border:1px solid var(--line);background:#fff}.result-table{min-width:7600px;font-size:11px}.result-table th{min-width:190px}.result-table th:first-child{min-width:390px;left:0;z-index:5}.result-table td:first-child{position:sticky;left:0;background:#fffdf7;z-index:2;min-width:390px}.metric-head b,.metric-head small{display:block}.metric-head small{margin-top:5px;color:#dcebe5;font-weight:500;line-height:1.35}.metric-head .prio{color:#f2cf7d;font:10px ui-monospace,monospace}.result-table tr.selected td{background:#fff0c9}.result-table tr.selected td:first-child{background:#ffe3a1}.ref-baseline{color:var(--deep);font-weight:900}.ref-gt{color:var(--rust);font-weight:900}.matrix{display:grid;grid-template-columns:80px repeat(2,minmax(220px,1fr));gap:1px;background:var(--line);border:1px solid var(--line)}.matrix>div{padding:8px;background:#fff}.matrix .head{background:var(--deep);color:#fff;font-weight:900}.waiting{min-height:280px;display:grid;place-items:center;text-align:center;color:#766f64}.small{font-size:12px;color:#625c52;line-height:1.55}@media(max-width:1500px){.row-grid{grid-template-columns:repeat(3,minmax(0,1fr))}.percept-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:850px){header{position:static}.reference-grid,.row-grid,.percept-grid{grid-template-columns:1fr}.matrix{grid-template-columns:1fr}.matrix .head{display:none}}
</style></head><body><header><a href="/">返回总览</a> · <a href="/wan22-ti2v-legacy-physiciq67-samples?v=1">返回原消融页</a><h1>Top100 Object Query<br>Ablation Metrics</h1><p class="lead">0613pybullet_sample_001460_w002 · seed 47326 · 1 Baseline + 24 Fixed + 24 Tube。每个指标同时区分“相对同 seed Baseline 的因果变化”和“相对 GT 的物理/视觉误差”。</p><div class="tools"><label>Target <select id="target"><option value="object_A">Object A · sphere</option><option value="object_B">Object B · box</option><option value="all_objects">All objects</option></select></label><label>Operator <select id="operator"></select></label><label>Metric object <select id="object"><option value="object_A">Object A · sphere</option><option value="object_B">Object B · box</option></select></label><button id="replay">同步从头播放</button><button id="refresh">刷新</button><span id="status" class="status">读取中</span></div></header><main><section class="note"><h2>参照与诊断口径</h2><p id="referenceNote"></p><div class="reference-grid"><figure><video controls muted playsinline preload="metadata" src="/api/object-query-ablation-metrics/input-video?id=baseline"></video><figcaption>Baseline · seed 47326 · 未消融生成结果</figcaption></figure><figure><video controls muted playsinline preload="metadata" src="/api/object-query-ablation-metrics/input-video?id=source_gt_video"></video><figcaption>GT source render · 前 49 帧；精确动力学中心与接触来自 states.npz</figcaption></figure></div></section><section><h2>消融矩阵定义</h2><p><span class="pill">A=softmax(QKᵀ/√d)</span><span class="pill">Y=AV</span><span class="pill">R=被选 object-query token</span><span class="pill">C=N∖R</span></p><div class="matrix"><div class="head">块</div><div class="head">矩阵区域</div><div class="head">信息流</div><div>S</div><div>A[R,R]</div><div>R K/V → R Query</div><div>I</div><div>A[R,C]</div><div>C K/V → R Query</div><div>O</div><div>A[C,R]</div><div>R K/V → C Query</div><div>B</div><div>A[C,C]</div><div>C K/V → C Query（始终保留）</div></div><p class="small">Fixed：R 只含 F00 稀疏点映射到 latent t=0 的 token。Tube：在未消融 Baseline 上冻结 CoTracker 轨迹，将同一组点在 F00/F04/…/F48 的 13 个 latent anchors 合并为时空集合。两者均在 S000–S039 全 40 个去噪步、两个 CFG 分支、同一 Top100 heads 上执行；Tube 改变了覆盖时间和 token 数，因此不是等剂量对照。</p></section><div id="rows"></div><section><h2>实际感知量可视化</h2><p class="small">每张图由真实计算输入生成：候选 crop、参照 crop、DINO patch 1−cos、LPIPS spatial map；按 F00/F12/F24/F36/F48 排列。</p><div id="percept" class="percept-grid"></div></section><section><h2>消融结果矩阵</h2><p class="small">结果拆成四张独立表：Fixed vs Baseline、Fixed vs GT、Tube vs Baseline、Tube vs GT。每行是一组明确的 Target × Operator；每列是一个指标。Metric object 下拉框决定对象级指标读取 Object A 或 Object B。GT 表中，中心/速度/接触读取 simulator states，其余对象级与视觉指标读取 source render；不适用于该 reference 的指标显示为「—」。</p><div id="resultMatrices"></div></section><section><h2>指标定义 · 按优先级</h2><div class="scroll"><table><thead><tr><th>#</th><th>优先级 / 指标</th><th>定义</th><th>计算形式</th><th>读法</th><th>实际计算量 / Overlay</th></tr></thead><tbody id="metricDefinitions"></tbody></table></div></section></main><script>
const api='/api/object-query-ablation-metrics',e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),q=new URL(location.href).searchParams,$=id=>document.getElementById(id);let data;
const f=v=>{if(v===null||v===undefined||Number.isNaN(Number(v)))return'—';return Math.abs(Number(v))>=100?Number(v).toFixed(2):Number(v).toFixed(5)};
const tuple=(o,keys)=>keys.map(k=>`${k}=${f(o?.[k])}`).join(' · ');
const assetLabel={trajectory:'Trajectory：CoTracker 点/中心/轨迹 + simulator center',mask:'Mask：SAM2 mask/质心/接触间距',raft:'RAFT：两套 flow + EPE map',perceptual:'Perceptual：对齐 crop + DINO/LPIPS map',pixel:'Pixel：帧差 + outside-object mask',input:'Input：实际生成视频与 VBench manifest'};
function rec(protocol){const target=$('target').value,mode=$('operator').value;return data.records.find(r=>r.protocol===protocol&&r.mask_mode===mode&&(target==='all_objects'?r.target_scope==='all_objects':r.target_scope==='single_object'&&r.region===target))}
function media(rel,kind='video'){const url=`${api}/asset?path=${encodeURIComponent(rel)}`;return kind==='image'?`<img loading="lazy" src="${url}">`:`<video controls muted playsinline preload="metadata" src="${url}"></video>`}
function input(id){return `<video controls muted playsinline preload="metadata" src="${api}/input-video?id=${encodeURIComponent(id)}"></video>`}
function metricValue(r,id,obj,ref){if(!r)return null;const o=r.objects[obj],track=ref==='baseline'?o.baseline_reference:o.source_video_reference,shape=ref==='baseline'?o.shape_vs_baseline:o.shape_vs_source,per=o.perceptual[ref==='baseline'?'baseline':'source_gt_video'],raft=r.raft[obj][ref==='baseline'?'vs_baseline':'vs_source'],pix=r.pixel[ref==='baseline'?'vs_baseline':'vs_source'],gt=o.simulator_gt_reference,lp=r.outside_object_lpips[ref==='baseline'?'baseline':'source_gt_video'];switch(id){case'gt_center_ade_change':return ref==='gt'?gt.center_ade_change_vs_baseline_norm:null;case'baseline_center_ade':return ref==='baseline'?track.center_ade_norm:null;case'gt_velocity_error_change':return ref==='gt'?gt.velocity_vector_error_change_vs_baseline_px_per_frame:null;case'contact_time_error_change':return ref==='gt'?r.interaction.contact_time_error_change_frames:null;case'post_contact_velocity_error_change':return ref==='gt'?r.interaction.post_contact_velocity_error_change_px_per_frame:null;case'other_object_center_ade':return ref==='baseline'?r.other_object?.center_ade_norm:null;case'center_fde':return ref==='baseline'?track.center_fde_norm:gt.center_fde_norm;case'object_normalized_pck':return tuple(track.pck_normalized,['0.05','0.1','0.2']);case'native_pck':return tuple(track.pck_native,['16','32','64']);case'point_ade':return track.point_ade_norm;case'velocity_decomposition':return tuple(ref==='gt'?gt:track,['velocity_speed_error_px_per_frame','velocity_direction_error_deg','velocity_vector_error_px_per_frame']);case'shape_iou':return shape.center_aligned_iou_mean;case'shape_geometry':return tuple(shape,['area_log_ratio_error_mean','aspect_log_ratio_error_mean','circularity_error_mean']);case'raft_roi_epe':return raft.flow_epe_mean_px;case'raft_motion_ratio':return raft.motion_magnitude_ratio;case'object_dino_similarity':return per.dino_cosine_mean;case'object_lpips':return per.lpips_mean;case'outside_object_lpips':return lp.outside_object_lpips_mean;case'raw_mask_iou':return shape.raw_iou_mean;case'vbench_subject_consistency':return ref==='baseline'?tuple(r.vbench.vbench_subject_consistency,['score','delta']):null;case'vbench_motion_smoothness':return ref==='baseline'?tuple(r.vbench.vbench_motion_smoothness,['score','delta']):null;case'vbench_dynamic_degree':return ref==='baseline'?tuple(r.vbench.vbench_dynamic_degree,['score','delta']):null;case'vbench_quality_suite':return ref==='baseline'?['vbench_background_consistency','vbench_temporal_flickering','vbench_imaging_quality','vbench_aesthetic_quality'].map(k=>`${k.replace('vbench_','')}=${f(r.vbench[k]?.score)} (Δ${f(r.vbench[k]?.delta)})`).join(' · '):null;case'full_frame_similarity':return tuple(pix,['ssim_mean','psnr_db','mae_0_1']);case'temporal_delta_mae':return pix.temporal_delta_mae_0_1;default:return null}}
function display(v){return typeof v==='string'?e(v):f(v)}
function protocolRow(protocol,r){const op=data.operator_definitions[$('operator').value],label=protocol==='fixed'?'Fixed Q00':'Tube Q00–Q12',rdef=protocol==='fixed'?'R_fixed：F00 稀疏点，仅 latent t=0':'R_tube：Baseline 冻结轨迹，13 个 latent anchors 联合集合';if(!r)return `<section class="protocol-row ${protocol}"><h2>${label}</h2><div class="waiting">该组合未生成</div></section>`;return `<section class="protocol-row ${protocol}"><h2>${label} · ${e(op.id)} · Top100</h2><div class="formula"><b>ID：</b><span class="mono">${e(r.id)}</span><br><b>R：</b>${e(rdef)}<br><b>切断：</b>${e(op.blocks)} · ${e(op.flow)}<br><b>精确计算：</b><span class="mono">${e(op.formula)}</span><br><b>理论诊断：</b>${e(op.meaning)}</div><div class="row-grid"><figure>${input(r.id)}<figcaption>${label} 消融生成视频</figcaption></figure><figure>${media(r.assets.trajectory)}<figcaption>Trajectory overlay · CoTracker 点/轨迹 + simulator center</figcaption></figure><figure>${media(r.assets.mask)}<figcaption>Mask overlay · SAM2 mask、质心、接触间距</figcaption></figure><figure>${media(r.assets.raft)}<figcaption>RAFT overlay · flow 与 EPE 差异</figcaption></figure><figure>${media(r.assets.pixel)}<figcaption>Pixel/outside overlay · 像素差与外部区域</figcaption></figure></div></section>`}
function resultTable(protocol,ref,obj){const chosen=rec(protocol)?.id,rows=data.records.filter(r=>r.protocol===protocol).sort((a,b)=>a.id.localeCompare(b.id));const refLabel=ref==='baseline'?'Baseline':'GT',head=`<tr><th>实验 ID</th>${data.metric_definitions.map(m=>`<th class="metric-head" title="定义：${e(m.definition)}&#10;公式：${e(m.formula)}&#10;读法：${e(m.direction)}"><span class="prio">#${m.rank} · ${e(m.priority)}</span><b>${e(m.name)}</b><small>${e(m.direction)}</small></th>`).join('')}</tr>`,body=rows.map(r=>`<tr class="${r.id===chosen?'selected':''}"><td><b>${e(r.operator_id)} · ${e(r.target_scope==='all_objects'?'all_objects':r.region)}</b><br><span class="mono">${e(r.id)}</span><br><span class="${ref==='baseline'?'ref-baseline':'ref-gt'}">vs ${refLabel}</span></td>${data.metric_definitions.map(m=>`<td class="value" title="${e(m.name)}：${e(m.direction)}；${e(assetLabel[m.asset]||m.asset)}">${display(metricValue(r,m.id,obj,ref))}</td>`).join('')}</tr>`).join('');return `<div class="result-block"><h3>${protocol==='fixed'?'Fixed R_fixed':'Tube R_tube'} · vs ${refLabel} · 24 experiments</h3><div class="result-scroll"><table class="result-table"><thead>${head}</thead><tbody>${body}</tbody></table></div></div>`}
function render(){const fixed=rec('fixed'),tube=rec('tube'),op=data.operator_definitions[$('operator').value],obj=$('object').value;$('rows').innerHTML=protocolRow('fixed',fixed)+protocolRow('tube',tube);$('percept').innerHTML=[['Fixed · Baseline',fixed,'baseline'],['Fixed · GT source',fixed,'source_gt_video'],['Tube · Baseline',tube,'baseline'],['Tube · GT source',tube,'source_gt_video']].map(([label,r,ref])=>r?`<figure>${media(r.assets.perceptual[obj][ref],'image')}<figcaption>${e(label)} · ${e(obj)} · ${e(r.id)}</figcaption></figure>`:'').join('');$('resultMatrices').innerHTML=resultTable('fixed','baseline',obj)+resultTable('fixed','gt',obj)+resultTable('tube','baseline',obj)+resultTable('tube','gt',obj);$('metricDefinitions').innerHTML=data.metric_definitions.map(m=>`<tr><td>${m.rank}</td><td><span class="priority ${e(m.priority)}">${e(m.priority)}</span><br><b>${e(m.name)}</b></td><td>${e(m.definition)}</td><td class="mono">${e(m.formula)}</td><td>${e(m.direction)}</td><td>${e(assetLabel[m.asset]||m.asset)}</td></tr>`).join('');const u=new URL(location.href);u.searchParams.set('target',$('target').value);u.searchParams.set('operator',$('operator').value);u.searchParams.set('object',obj);history.replaceState(null,'',u);$('status').textContent=`${data.video_count} videos · ${data.ablation_count} ablations · ${op.id}`}
async function load(){data=await fetch(`${api}/catalog?v=${Date.now()}`,{cache:'no-store'}).then(r=>r.json());if(!data.ready){$('status').textContent=data.reason;$('rows').innerHTML=`<section class="waiting">${e(data.reason)}</section>`;return}$('referenceNote').textContent=data.reference_note;const modes=Object.entries(data.operator_definitions);$('operator').innerHTML=modes.map(([k,v])=>`<option value="${e(k)}">${e(v.id)} · ${e(k)}</option>`).join('');for(const id of ['target','operator','object'])if(q.get(id)&&[...$(id).options].some(o=>o.value===q.get(id)))$(id).value=q.get(id);render()}
['target','operator','object'].forEach(id=>$(id).addEventListener('change',render));$('refresh').addEventListener('click',load);$('replay').addEventListener('click',()=>document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.play().catch(()=>{})}));load();
</script></body></html>'''
