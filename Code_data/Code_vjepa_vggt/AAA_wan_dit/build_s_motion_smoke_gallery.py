#!/usr/bin/env python3
"""Render the Motion Impact smoke report with synchronized videos and raw metrics."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INVENTORY = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_smoke/inventory.json"
)
DEFAULT_METRICS = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_smoke/results/per_video_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed/motion-n-analysis/smoke"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clean_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean_value(value.item())
    return value


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Motion Impact Smoke · 指标与视频证据</title>
<style>
:root{--bg:#111518;--panel:#1b2125;--panel2:#222a2f;--line:#3b464d;--text:#edf2f4;--muted:#aab4ba;--teal:#66cbb4;--amber:#e3b765;--red:#ef837c;--blue:#7fb1e3}
*{box-sizing:border-box}body{margin:0;padding-bottom:58px;background:var(--bg);color:var(--text);font:13px/1.48 Arial,"Noto Sans SC",sans-serif;letter-spacing:0}
header{position:sticky;top:0;z-index:8;padding:11px 16px;background:#111518f2;border-bottom:1px solid var(--line)}
.top,.controls,.row-head,.metric-strip{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.top{justify-content:space-between}h1,h2,h3,p{margin:0}h1{font-size:19px}h2{font-size:19px;margin:25px 0 8px}h3{font-size:14px}.muted{color:var(--muted)}
.controls{margin-top:8px}label{display:grid;gap:2px;color:var(--muted);font-size:10px}select,button{border:1px solid var(--line);background:#283137;color:var(--text);padding:6px 9px}button{cursor:pointer}a{color:var(--teal);font-weight:700;text-decoration:none}
main{padding:15px 16px;max-width:1680px;margin:auto}.definitions{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:8px}.definition{background:var(--panel);border:1px solid var(--line);padding:11px}.definition b{color:var(--teal)}code{display:block;margin-top:7px;padding:7px;background:#0e1113;color:#dce8e5;white-space:normal}
.protocol{margin-top:8px;padding:10px;border-left:3px solid var(--amber);background:#29261e;color:#e9dfc8}.scales{margin-top:8px;border-collapse:collapse}.scales td,.scales th{border:1px solid var(--line);padding:4px 7px;text-align:left}
.gt-grid{display:grid;grid-template-columns:minmax(300px,560px);gap:8px}.group{margin-top:28px;border-top:3px solid var(--teal)}.row{margin-top:13px}.row-head{justify-content:space-between;padding:7px 8px;background:#20272b;border-left:3px solid var(--teal)}.row-actions{display:flex;gap:5px}.row-actions button{padding:3px 7px;font-size:11px}
.videos{display:grid;grid-template-columns:repeat(5,minmax(240px,1fr));gap:7px;overflow-x:auto}.card{min-width:240px;background:var(--panel);border:1px solid var(--line)}.card h3{padding:6px 8px;background:var(--panel2)}video{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#050607;cursor:pointer}.metric-strip{padding:6px 7px}.chip{display:grid;gap:1px;min-width:74px;padding:4px 6px;background:#12171a;border:1px solid #323d43}.chip small{color:var(--muted);font-size:9px}.chip strong{font-size:13px}.positive{color:var(--teal)}.negative{color:var(--red)}
.card-meta{padding:0 7px 7px;color:var(--muted);font-size:10px}details{border-top:1px solid var(--line);padding:6px 7px}summary{cursor:pointer;color:var(--blue)}.raw{width:100%;margin-top:5px;border-collapse:collapse;font-size:10px}.raw td{padding:2px 4px;border-bottom:1px solid #30393f}.raw td:last-child{text-align:right;font-family:monospace}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:8px}.charts img{display:block;width:100%;background:#fff}.footerbar{position:fixed;left:0;right:0;bottom:0;z-index:10;display:flex;align-items:center;gap:7px;padding:9px 16px;background:#171d21f2;border-top:1px solid var(--line)}.footerbar span{margin-left:auto;color:var(--muted)}
@media(max-width:900px){.definitions,.charts{grid-template-columns:1fr}.videos{grid-template-columns:repeat(4,minmax(250px,85vw))}}
</style></head><body>
<header><div class="top"><h1>Motion Impact Smoke · 指标与视频证据</h1><a href="../">返回正式分析进度</a></div>
<div class="controls"><label>模型<select id="model"></select></label><label>深度实验 seed<select id="seed"></select></label><span class="muted" id="case"></span></div></header>
<main>
<section><h2>三个指标如何计算</h2><div class="definitions">
<article class="definition"><b>Motion Impact ↑：消融改变 baseline 运动的强度</b>
<p>先计算四个相对 baseline 的归一化误差：完整 RAFT 光流向量、Top-5% 强运动曲线、去除背景相机漂移后的物体轨迹、物体速度曲线。</p>
<code>r = error(ablation, baseline) / max(RMS(baseline), ε)<br>Impact = mean[ log(1 + r_flow), log(1 + r_top5), log(1 + r_traj), log(1 + r_speed) ]</code>
<p class="muted">ε：光流向量、Top-5% 曲线和速度为 0.002，轨迹为 0.01。光流 x/y 已分别除以图像宽/高。越大只表示“运动变化越大”，不表示视频更好或更符合物理。</p></article>
<article class="definition"><b>Impact/head ↑：控制消融数量后的近似敏感度</b>
<code>Impact/head = Motion Impact / 被消融 head 数</code>
<p>用于辅助比较 32-head、40/58/61-head 和 64-head 配置。Transformer 存在非线性和 head 交互，因此它不是单个 head 的因果贡献，也不能替代等数量消融。</p></article>
<article class="definition"><b>GT gain ↑：相对 baseline 是否更接近 GT 运动</b>
<p>对每个生成视频计算到 GT 的七项固定尺度距离，再与相同模型、seed、case 的 baseline 比较。</p>
<code>D_GT = mean_j log(1 + error_j / scale_j)<br>GT gain = D_GT(baseline) - D_GT(ablation)</code>
<p><span class="positive">正值：比 baseline 更接近 GT</span>；<span class="negative">负值：比 baseline 更远</span>。缺失分项按 log(1+10) 惩罚。它仍是视觉运动相似度，不是物理定律判定器。</p></article>
</div>
<div class="protocol"><b>对齐协议：</b>所有视频统一 49 帧、30 FPS、896×512；内部中心裁剪到 7:4 后以 256×448 提取特征。从 context 最后一帧 frame 7 开始评价。物体点来自逐 case 的实例区域，CoTracker 轨迹减去背景点的中位位移；PhyCo 使用数据集实例分割，PyBullet/Physics-IQ 使用 frame 7 的 SAM2 区域。</div>
<table class="scales"><thead><tr><th>GT 距离分项</th><th>固定 scale</th></tr></thead><tbody>
<tr><td>RAFT 向量 RMSE</td><td>0.005</td></tr><tr><td>Top-5% 光流曲线 RMSE</td><td>0.010</td></tr>
<tr><td>物体轨迹 RMSE</td><td>0.050</td></tr><tr><td>物体速度曲线 RMSE</td><td>0.005</td></tr>
<tr><td>物体加速度曲线 RMSE</td><td>0.002</td></tr><tr><td>背景漂移绝对误差</td><td>0.005</td></tr>
<tr><td>物体可见率绝对误差</td><td>0.250</td></tr></tbody></table>
</section>
<section><h2>GT 参考视频</h2><div class="gt-grid" id="gt"></div></section>
<div id="content"></div>
<section><h2>汇总热力图</h2><p class="muted">本 smoke 只有一个 case；数值用于核查实现和观察具体视频，不构成统计显著性结论。</p>
<div class="charts"><img src="s_feature_motion_heatmaps.png" alt="S feature metrics"><img src="s_depth_motion_heatmaps.png" alt="S depth metrics"></div></section>
</main>
<div class="footerbar"><button id="playAll">播放当前页</button><button id="replayAll">从头播放</button><button id="pauseAll">暂停</button><span>单击视频也可播放/暂停；不自动循环</span></div>
<script>
const DATA=__DATA__;const MODELS={wan_lora:"Wan+LoRA",xssc:"Wan+xSSC",physrvg:"PhysRVG"};
const q=id=>document.getElementById(id);let syncTimer=null,activeVideos=[];
function fmt(v,d=3){return v===null||v===undefined||Number.isNaN(Number(v))?"—":Number(v).toFixed(d)}
function entry(filter){return DATA.entries.find(e=>Object.entries(filter).every(([k,v])=>e[k]===v))}
function metric(e){return e?DATA.metrics[e.entry_id]:null}
function rawRows(m){const specs=[["RAFT vector vs baseline","flow_vector_rmse_baseline"],["Top-5% flow vs baseline","flow_top05_curve_rmse_baseline"],["Object trajectory vs baseline","object_trajectory_rmse_baseline"],["Object speed vs baseline","object_speed_curve_rmse_baseline"],["RAFT vector vs GT","flow_vector_rmse_gt"],["Top-5% flow vs GT","flow_top05_curve_rmse_gt"],["Object trajectory vs GT","object_trajectory_rmse_gt"],["Object speed vs GT","object_speed_curve_rmse_gt"],["Object acceleration vs GT","object_acceleration_curve_rmse_gt"],["Background drift vs GT","background_drift_abs_error_gt"],["Object visibility vs GT","object_visibility_abs_error_gt"],["D_GT","plausibility_distance_gt"]];return specs.map(([label,key])=>`<tr><td>${label}</td><td>${fmt(m[key],5)}</td></tr>`).join("")}
function card(e,label,isGT=false){if(!e)return`<article class="card"><h3>${label}</h3><div class="card-meta">该 smoke 配置不存在</div></article>`;const m=metric(e),heads=e.head_count||0,gain=m?m.gt_gain_vs_baseline:null,gainClass=gain>0?"positive":gain<0?"negative":"";return`<article class="card"><h3>${label}</h3><video muted playsinline preload="metadata" src="${e.media_url}"></video>${isGT?`<div class="card-meta">49 frames @ 30 FPS · 仅作为参考，不定义 Impact</div>`:`<div class="metric-strip"><span class="chip"><small>Motion Impact ↑</small><strong>${fmt(m.impact_score)}</strong></span><span class="chip"><small>Impact/head ↑</small><strong>${heads?fmt(m.impact_score/heads,5):"—"}</strong></span><span class="chip"><small>GT gain ↑</small><strong class="${gainClass}">${gain>0?"+":""}${fmt(gain)}</strong></span></div><div class="card-meta">heads=${heads} · seed=${e.seed} · ${e.denoise_step_range?`steps ${e.denoise_step_range[0]}–${e.denoise_step_range[1]}`:"未消融 baseline"}</div><details><summary>查看原始误差分项</summary><table class="raw">${rawRows(m)}</table></details>`}</article>`}
function row(title,cards){return`<section class="row"><div class="row-head"><h3>${title}</h3><div class="row-actions"><button data-row="play">播放本行</button><button data-row="replay">重播本行</button><button data-row="pause">暂停</button></div></div><div class="videos">${cards.join("")}</div></section>`}
function baseline(model,seed){return entry({family:"baseline",model,seed})}
function render(){stopSync();const model=q("model").value,seed=Number(q("seed").value),gt=DATA.entries.find(e=>e.family==="gt");const featureSeed=851;let out=`<section class="group"><h2>${MODELS[model]} · S 子类别（seed 851）</h2>`;out+=row("全去噪阶段 [0,40)",[card(gt,"GT",true),card(baseline(model,featureSeed),"Baseline"),card(entry({family:"s_feature",model,seed:featureSeed,subtype:"local_enrichment"}),"Local-enrichment S · 32 heads"),card(entry({family:"s_feature",model,seed:featureSeed,subtype:"same_frame_mass"}),"Same-frame-mass S · 32 heads"),card(entry({family:"s_feature",model,seed:featureSeed,subtype:"local_same_union"}),"Local + Same union · 64 heads")]);out+=`</section><section class="group"><h2>${MODELS[model]} · S 深度（seed ${seed}）</h2>`;for(const [start,end] of [[0,10],[10,20]])out+=row(`去噪阶段 [${start},${end})`,[card(gt,"GT",true),card(baseline(model,seed),"Baseline"),card(entry({family:"s_depth",model,seed,depth_stratum:"early",denoise_key:`${start}-${end}`}),"Early B00–09"),card(entry({family:"s_depth",model,seed,depth_stratum:"middle",denoise_key:`${start}-${end}`}),"Middle B10–19"),card(entry({family:"s_depth",model,seed,depth_stratum:"late",denoise_key:`${start}-${end}`}),"Late B20–29")]);out+="</section>";q("content").innerHTML=out;bindVideos()}
function bindVideos(){document.querySelectorAll("video").forEach(v=>v.onclick=()=>v.paused?v.play().catch(()=>{}):v.pause())}
function stopSync(){if(syncTimer!==null)clearInterval(syncTimer);syncTimer=null;activeVideos=[]}
function play(videos,replay){stopSync();activeVideos=videos;if(replay)videos.forEach(v=>v.currentTime=0);videos.forEach(v=>v.play().catch(()=>{}));syncTimer=setInterval(()=>{if(activeVideos.length<2||activeVideos[0].readyState<2)return;for(const v of activeVideos.slice(1))if(v.readyState>=2&&Math.abs(v.currentTime-activeVideos[0].currentTime)>.12)v.currentTime=activeVideos[0].currentTime},250)}
document.addEventListener("click",ev=>{const b=ev.target.closest("[data-row]");if(!b)return;const videos=[...b.closest(".row").querySelectorAll("video")];if(b.dataset.row==="pause"){videos.forEach(v=>v.pause());stopSync()}else play(videos,b.dataset.row==="replay")});
q("playAll").onclick=()=>play([...document.querySelectorAll("video")],false);q("replayAll").onclick=()=>play([...document.querySelectorAll("video")],true);q("pauseAll").onclick=()=>{document.querySelectorAll("video").forEach(v=>v.pause());stopSync()};
q("model").innerHTML=Object.entries(MODELS).map(([k,v])=>`<option value="${k}">${v}</option>`).join("");q("seed").innerHTML=[851,3278].map(x=>`<option>${x}</option>`).join("");q("case").textContent=`case: ${DATA.case_id}`;q("model").onchange=render;q("seed").onchange=render;q("gt").innerHTML=card(DATA.entries.find(e=>e.family==="gt"),"Ground Truth · 49f @ 30 FPS",true);render();bindVideos();
</script></body></html>"""


def main() -> None:
    args = parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    metrics_frame = pd.read_csv(args.metrics)
    metrics = {
        str(record["entry_id"]): {
            key: clean_value(value) for key, value in record.items()
        }
        for record in metrics_frame.to_dict(orient="records")
    }
    media_dir = args.output / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, original in enumerate(inventory["entries"]):
        entry = dict(original)
        source = Path(entry["source"]["path"]).resolve()
        media_path = media_dir / f"video_{index:03d}.mp4"
        if media_path.is_symlink() or media_path.exists():
            if media_path.resolve() != source:
                media_path.unlink()
        if not media_path.exists():
            media_path.symlink_to(source)
        stage = entry.get("denoise_step_range")
        entry["denoise_key"] = (
            f"{int(stage[0])}-{int(stage[1])}" if stage is not None else None
        )
        entry["media_url"] = f"media/{media_path.name}"
        entries.append(entry)
    data = {
        "case_id": inventory["case_id"],
        "entries": entries,
        "metrics": metrics,
    }
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace(
        "</", "<\\/"
    )
    atomic_write(args.output / "index.html", PAGE.replace("__DATA__", serialized))
    print(
        f"[s-motion-smoke-gallery] entries={len(entries)} "
        f"metrics={len(metrics)} output={args.output / 'index.html'}"
    )


if __name__ == "__main__":
    main()
