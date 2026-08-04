#!/usr/bin/env python3
"""Extend the existing 8790 atlas with stable-head all-token Q/K matrices."""

from __future__ import annotations

import argparse
import csv
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from AAA_my_test import serve_latent_block_head_viewer as base


ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_rank_extremes_alltoken_qk_5case"
)
ALL720_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_uniform_diagonal_5case"
)
NEIGHBOR_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_5case"
)
NEIGHBOR_HEAT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_heatmaps_case001"
)
HEAD_ZERO_ROOT = Path(
    "/data/gaoya/agent-data/outputs/top5_pck_head_zero_ablation_5case"
)
EXTREME30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme30_all720_head_zero_ablation_test5"
)
EXTREME100_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme100_all720_head_zero_ablation_test5"
)
EXTREME_ZERO_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_top30_bottom30_head_zero_ablation_test5"
)
INTERVAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "three_model_joint_interval_samples_alltoken_qk_case001"
)
BALANCED_INTERVAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "three_model_balanced_interval_samples_alltoken_qk_case001"
)
PCK_OVERLAY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_overlay_case001_s039"
)
BALANCED_VIDEO_PATHS = {
    "gt": Path(
        "/data/gaoya/agent-data/outputs/wan22_ti2v_5b_gt_real_sam2_regions_steps40/"
        "cases/case_001_ball_roll/gt.mp4"
    ),
    "lora": Path(
        "/data/gaoya/agent-data/outputs/"
        "wan_openvid_0613pybullet_lora_step000500_sam2_regions_steps40/"
        "cases/case_001_ball_roll/generated.mp4"
    ),
    "baseline": Path(
        "/data/gaoya/agent-data/outputs/wan22_ti2v_5b_baseline_sam2_regions_steps40/"
        "cases/case_001_ball_roll/generated.mp4"
    ),
}
SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case"
)
RANKING = json.loads((ROOT / "ranking_top_bottom30.json").read_text(encoding="utf-8"))
TOP_COMBINATIONS = tuple((item["block"], item["head"]) for item in RANKING["top"])
BOTTOM_COMBINATIONS = tuple((item["block"], item["head"]) for item in RANKING["bottom"])
COMBINATIONS = TOP_COMBINATIONS + BOTTOM_COMBINATIONS
PCK32 = {
    (item["block"], item["head"]): item["pck32"]
    for group in (RANKING["top"], RANKING["bottom"])
    for item in group
}
CASE_KEYS = tuple(f"case_{index:03d}" for index in range(1, 6))
MODELS = (("gt", "GT teacher-forced"), ("lora", "LoRA step-000500"), ("baseline", "Wan2.2 Baseline"))


BASE_HANDLER = next(
    value for value in vars(base).values()
    if isinstance(value, type)
    and value.__module__ == base.__name__
    and issubclass(value, BaseHTTPRequestHandler)
    and value is not BaseHTTPRequestHandler
)


PORTAL = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DiffTrack 可视化总览</title><style>
:root{--paper:#eee9dc;--ink:#17211e;--card:#fffdf8;--line:#b8b09f;--rust:#b64a31;--teal:#176654}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 5% 0,#d7764b35,transparent 36rem),radial-gradient(circle at 95% 5%,#4b9a8035,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1500px,calc(100% - 28px));margin:auto;padding:35px 0 70px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(42px,7vw,90px);line-height:.9;margin:8px 0 18px;letter-spacing:-.05em}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.16em;font-size:12px}.lead{max-width:980px;line-height:1.7}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px}.card{display:flex;min-height:245px;flex-direction:column;justify-content:space-between;padding:22px;background:var(--card);border:1px solid var(--line);color:inherit;text-decoration:none;border-radius:3px 28px 3px 3px}.card.new{background:#13251f;color:white;border-color:#13251f}.card h2{font-size:29px;margin:8px 0}.card p{line-height:1.55;opacity:.76}.go{font-size:12px;font-weight:900;letter-spacing:.08em;color:var(--rust)}.new .go{color:#e39a78}@media(max-width:1000px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">DIFFTRACK · 三模型注意力与轨迹图谱</div><h1>从轨迹指标<br>读懂注意力</h1><p class="lead">汇总 GT Teacher-Forced、LoRA 与 Wan2.2 Baseline 的 50-case 轨迹指标、全时间步排名，以及代表 case 的 all-token Q@K 热力图和生成视频。</p><section class="grid">
<a class="card new" href="/balanced-interval-heatmaps?v=3"><div><span>01 / BALANCED 分箱</span><h2>Balanced 区间热力图</h2><p>按 Balanced Diagonal 分箱，展示代表 Head、五项分数，以及三个模型的视频与热力图。</p></div><span class="go">打开 BALANCED 分箱页面</span></a>
<a class="card" href="/joint-interval-heatmaps?v=2"><div><span>02 / JOINT 分箱</span><h2>Joint 区间热力图</h2><p>按 Joint 分数分箱，每个区间选择 PCK 从低到高的代表 Head。</p></div><span class="go">打开 JOINT 分箱页面</span></a>
<a class="card" href="/uniform-diagonal-curves?v=3"><div><span>03 / 720 组合分布</span><h2>指标曲线与数量分布</h2><p>查看 30 个 Block × 24 个 Head 的指标曲线、PCK 关系和区间数量。</p></div><span class="go">打开指标分布页面</span></a>
<a class="card new" href="/neighbor-diagonal-ranking?v=4"><div><span>04 / 全帧块空间对角线</span><h2>Block × Head 严格对角线矩阵</h2><p>S039 下展示 30 × 24 性能矩阵，并按“相邻三帧均衡 × 全部帧块空间对角线纯度”排序 720 个 Head；支持三模型切换和对应 Q@K 热力图。</p></div><span class="go">打开严格对角线矩阵</span></a>
<a class="card" href="/all-token-qk?v=7"><div><span>04 / 排名两端</span><h2>Top30 与 Bottom30</h2><p>固定 S039，三模型并排展示 PCK@32 排名前后各 30 个组合。</p></div><span class="go">打开 ALL-TOKEN 热力图</span></a>
<a class="card" href="/all-steps/overlays?v=3"><div><span>05 / 全时间步轨迹</span><h2>轨迹 Overlay 图谱</h2><p>按照三模型综合全局排名，浏览 GT 与 Q@K 预测轨迹的逐帧叠加结果。</p></div><span class="go">打开轨迹叠加页面</span></a>
<a class="card" href="/all-steps/rankings?v=4"><div><span>06 / 全时间步指标</span><h2>Step × Block × Head 排名</h2><p>覆盖 28,800 个组合的三模型综合排名、单模型排名和性能热力图。</p></div><span class="go">打开全时间步排名</span></a>
<a class="card new" href="/pck-extreme-head-zero-ablation?v=1"><div><span>07 / 分阶段消融</span><h2>PCK Top30 / Bottom30 输出置零</h2><p>对比 Wan2.2 Baseline 与 Wan+LoRA 的高低 PCK Head 在四个十步阶段及全程置零后的生成视频。</p></div><span class="go">打开极值 Head 消融页面</span></a>
<a class="card new" href="/pck-extreme30-head-zero-ablation?v=1"><div><span>08 / ALL 720 消融</span><h2>全组合 PCK Top30 / Bottom30</h2><p>从全部 720 个 Block-Head 选择排名两端，各 30 个 Head 分阶段同时置零。</p></div><span class="go">打开 ALL-720 极值消融</span></a>
<a class="card new" href="/pck-extreme100-head-zero-ablation?v=1"><div><span>09 / ALL 720 扩大消融</span><h2>全组合 PCK Top100 / Bottom100</h2><p>从全部 720 个 Block-Head 选择排名两端，各 100 个 Head 分阶段同时置零。</p></div><span class="go">打开 TOP100 / BOTTOM100 消融</span></a>
<a class="card" href="/rankings?v=2"><div><span>07 / 固定时间步指标</span><h2>Block × Head 排名</h2><p>比较固定时间步下三个模型各自的 720 个 Block-Head 组合。</p></div><span class="go">打开固定步排名</span></a>
<a class="card" href="/single?v=2"><div><span>08 / 单组合检查</span><h2>轨迹显微镜</h2><p>选择模型、案例、Block 和 Head，逐潜空间帧检查 GT 与 Q@K 轨迹。</p></div><span class="go">打开单组合检查器</span></a>
<a class="card new" href="/pck-extreme-overlays?v=1"><div><span>09 / PCK 高低对比</span><h2>Top6 与 Bottom6 轨迹视频</h2><p>在原视频真实锚点帧上，并排比较三个模型的 GT 与 Q@K 轨迹，直观看到 PCK 高低差异。</p></div><span class="go">打开 PCK OVERLAY 视频</span></a>
</section></main></body></html>'''


PAGE_S039 = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S039 PCK 排名两端 Q@K 热力图</title><style>
:root{--paper:#ece7da;--ink:#18221e;--panel:#101714;--cream:#fffdf7;--rust:#c65738;--teal:#287a67;--line:#aaa392}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#d28b6330,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(2200px,calc(100% - 24px));margin:auto;padding:24px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}.top{display:flex;justify-content:space-between;gap:25px;align-items:end}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.14em;font-size:11px}h1{font-size:clamp(38px,5vw,72px);line-height:.92;margin:7px 0}.note{max-width:1000px;line-height:1.55;color:#59635e}.back{color:var(--ink);font-weight:900}.controls{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin:20px 0;padding:14px;background:var(--cream);border:1px solid var(--line)}label{font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:9px;background:white;border:1px solid #5f665f;font-weight:800}.section-title{display:flex;align-items:baseline;justify-content:space-between;margin:28px 0 9px;border-bottom:2px solid var(--ink)}.section-title h2{font-size:30px;margin:0 0 5px}.section-title span{font-size:11px;font-weight:900;color:#68726d}.rank-row{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px;padding-bottom:12px}.card{background:var(--panel);color:white;padding:8px;border-radius:3px 18px 3px 3px;min-width:0}.card h3{margin:2px 2px 7px;font:700 18px Georgia,serif}.card h3 span{display:block;margin-top:3px;color:#a9b8b0;font:700 10px "Trebuchet MS",sans-serif}.card img{display:block;width:100%;aspect-ratio:3/1;background:#050806;object-fit:contain}.status{font-size:12px;font-weight:900;color:var(--teal)}@media(max-width:1000px){.rank-row{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:650px){.controls{grid-template-columns:1fr}.top{display:block}.rank-row{grid-template-columns:1fr}}
</style></head><body><main><div class="top"><div><div class="eyebrow">5 个案例 · 仅 S039 · TOP/BOTTOM 各 30 个组合</div><h1>最终时间步的<br>PCK@32 排名两端</h1></div><a class="back" href="/">返回可视化总览</a></div><p class="note">每张卡固定展示 S039，内部从左到右依次为 GT teacher-forced、LoRA、Wan2.2 Baseline。Top 30 放在第一行，Bottom 30 放在第二行，均按 PCK@32 排名横向排列。</p><section class="controls"><label>案例<select id="case"></select></label><label>矩阵类型<select id="kind"><option value="attention">log10 Softmax 注意力</option><option value="raw">原始 QK / sqrt(d)</option><option value="temporal">7×7 时间注意力</option></select></label></section><p class="status" id="status"></p><section><div class="section-title"><h2>PCK@32 最高的 30 个</h2><span>排名第 1 至第 30</span></div><div class="rank-row" id="topRow"></div></section><section><div class="section-title"><h2>PCK@32 最低的 30 个</h2><span>由最低至倒数第 30</span></div><div class="rank-row" id="bottomRow"></div></section></main><script>
const q=id=>document.getElementById(id);let DATA;function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}async function init(){DATA=await fetch('/api/all-token/catalog').then(r=>r.json());q('case').innerHTML=DATA.cases.map(c=>`<option>${esc(c)}</option>`).join('');q('case').addEventListener('change',render);q('kind').addEventListener('change',render);render()}function cards(group){const c=q('case').value,k=q('kind').value;return DATA.combinations.filter(x=>x.group===group).map((x,i)=>{const b=String(x.block).padStart(2,'0'),h=String(x.head).padStart(2,'0'),score=`#${i+1} · 平均 PCK@32 ${Number(x.pck32).toFixed(3)}`,src=`/api/all-token/s039-strip?case=${encodeURIComponent(c)}&block=${x.block}&head=${x.head}&kind=${k}&v=2`;return`<article class="card"><h3>L${b} / H${h}<span>${score}</span></h3><img loading="lazy" src="${src}" alt="L${b} H${h} S039 three-model QK"></article>`}).join('')}function render(){q('status').textContent=`${q('case').value} · S039 · 未完成的模型标记为“待生成”。`;q('topRow').innerHTML=cards('Top stable');q('bottomRow').innerHTML=cards('Bottom PCK@32')}init();
</script></body></html>'''


CURVE_PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>720 个组合的均匀对角线指标</title><style>
:root{--paper:#ece7da;--ink:#18221e;--card:#fffdf7;--rust:#c65738;--line:#aaa392}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 5% 0,#d28b6335,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1700px,calc(100% - 24px));margin:auto;padding:28px 0 60px}h1{font:700 clamp(40px,6vw,78px)/.92 Georgia,serif;letter-spacing:-.04em;margin:8px 0}.eyebrow{color:var(--rust);font-size:11px;font-weight:900;letter-spacing:.15em}.lead{max-width:1050px;line-height:1.6;color:#59635e}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:18px 0}.stats span{padding:14px;background:var(--card);border:1px solid var(--line)}.stats b{display:block;font:700 28px Georgia,serif}.plot{display:block;width:100%;background:white;border:1px solid var(--line)}.links{display:flex;gap:16px;margin:14px 0}.links a{color:var(--ink);font-weight:900}.back{float:right}@media(max-width:700px){.stats{grid-template-columns:1fr 1fr}}
</style></head><body><main><a class="back" href="/">返回可视化总览</a><div class="eyebrow">S039 · 30 BLOCKS × 24 HEADS · 3 MODELS × 5 CASES</div><h1>均匀对角线指标<br>分布</h1><p class="lead">Joint score = 帧内同空间对角带质量 × 对角质量在7帧间的归一化熵。上图展示排序曲线与 Joint–PCK 关系；下图统计 720 个 block-head 组合落入各指标区间的数量。</p><section class="stats"><span><b>720</b>个 Block-Head 组合</span><span><b>10,800</b>条案例-模型记录</span><span><b>0.581</b>Pearson r</span><span><b>0.614</b>Spearman rho</span></section><img class="plot" src="/api/all720/curve?v=2" alt="All 720 uniform diagonal curves"><img class="plot" style="margin-top:18px" src="/api/all720/count-distribution?v=1" alt="All 720 metric count distributions"><div class="links"><a href="/downloads/all720-uniform-diagonal.csv">下载 720 行明细 CSV</a><a href="/downloads/all720-count-distribution.csv">下载区间数量 CSV</a><a href="/joint-interval-heatmaps?v=1">Joint 分箱代表热力图</a><a href="/balanced-interval-heatmaps?v=1">Balanced 分箱代表热力图</a><a href="/all-token-qk?v=6">Top30 / Bottom30 热力图</a></div></main></body></html>'''


HEAD_ZERO_PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top5 PCK Head 分阶段消融</title><style>
:root{--paper:#e7e0d2;--ink:#18211d;--card:#fffdf7;--rust:#bc5033;--line:#9d9687}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 6% 0,#cc76533b,transparent 36rem),linear-gradient(130deg,#e7e0d2,#f5f0e6);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(2380px,calc(100% - 24px));margin:auto;padding:26px 0 70px}h1{font:700 clamp(38px,5vw,74px)/.94 Georgia,serif;letter-spacing:-.045em;margin:8px 0}.eyebrow{color:var(--rust);font-size:11px;font-weight:900;letter-spacing:.14em}.lead{max-width:1250px;line-height:1.6;color:#5c655f}.heads{display:flex;flex-wrap:wrap;gap:7px;margin:18px 0 26px}.heads span{background:var(--ink);color:#fff;padding:8px 11px;font-weight:900;font-size:12px}.status{position:sticky;top:8px;z-index:5;background:#fffdf7e8;border:1px solid var(--line);padding:10px 14px;backdrop-filter:blur(8px)}.case{margin-top:32px;border-top:3px solid var(--ink);padding-top:14px}.case-head{display:flex;align-items:center;justify-content:space-between;gap:12px}.case h2{font:700 27px Georgia,serif;margin:0 0 8px;overflow-wrap:anywhere}.replay{border:0;background:var(--rust);color:white;padding:9px 13px;font-weight:900;cursor:pointer;white-space:nowrap}.model{margin:15px 0 24px}.model h3{margin:0 0 8px;color:var(--rust);font-size:15px;letter-spacing:.08em}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px}.card{background:var(--card);border:1px solid var(--line);padding:8px}.card strong{display:block;margin-bottom:6px;font-size:12px}.card small{display:block;color:#69716c;margin-top:6px}video,.pending{display:block;width:100%;aspect-ratio:896/512;background:#111}.pending{display:grid;place-items:center;color:#c8c2b5;font-weight:900}.back{float:right;color:var(--ink);font-weight:900}@media(max-width:1500px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:760px){.grid{grid-template-columns:1fr}.back{float:none;display:block;margin-bottom:12px}.case-head{align-items:flex-start}}
</style></head><body><main><a class="back" href="/">返回可视化总览</a><div class="eyebrow">WAN2.2 BASELINE × WAN+LORA · TOY5 + TEST_5 · TOP5 同时置零</div><h1>去噪阶段决定了<br>这些 Head 在做什么</h1><p class="lead">每组依次展示未消融、S00–09、S10–19、S20–29、S30–39 和 S00–39。视频播放一次后停止；点击每个 case 的按钮可将该组所有已完成视频从头同时播放。生成期间页面每 10 秒刷新一次可用结果。</p><div class="heads"><span>L09/H13 · PCK 87.769</span><span>L07/H08 · PCK 87.058</span><span>L06/H02 · PCK 86.661</span><span>L21/H07 · PCK 86.509</span><span>L13/H05 · PCK 84.399</span></div><div class="status" id="status">正在读取结果...</div><div id="cases"></div></main><script>
const MODELS=[['baseline','Wan2.2 Baseline'],['lora','Wan + LoRA']];const STAGES=[['original','Original','无消融'],['steps_00_10','S00–09','早期'],['steps_10_20','S10–19','中前期'],['steps_20_30','S20–29','中后期'],['steps_30_40','S30–39','后期'],['steps_00_40','S00–39','全程']];let signature='';const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function replay(button){const videos=button.closest('.case').querySelectorAll('video');videos.forEach(v=>{v.pause();v.currentTime=0});videos.forEach(v=>v.play().catch(()=>{}))}function render(data){const next=JSON.stringify(data.cases.map(c=>[c.case_key,c.available]));document.getElementById('status').textContent=`${data.ready_videos}/${data.expected_videos} 个视频已就绪 · ${data.complete_cases}/${data.cases.length} 个 case 双模型完整`;if(next===signature)return;signature=next;document.getElementById('cases').innerHTML=data.cases.map(c=>`<section class="case"><div class="case-head"><div><h2>${esc(c.case_key)}</h2><small>${esc(c.group)}</small></div><button class="replay" onclick="replay(this)">本组全部从头播放</button></div>${MODELS.map(([m,label])=>`<div class="model"><h3>${label}</h3><div class="grid">${STAGES.map(([v,title,note])=>{const ready=(c.available[m]||[]).includes(v),src=`/api/top5-head-zero/video?model=${m}&case=${encodeURIComponent(c.case_key)}&variant=${v}`;return`<article class="card"><strong>${title}</strong>${ready?`<video controls muted playsinline preload="metadata" src="${src}"></video>`:`<div class="pending">生成中</div>`}<small>${note}</small></article>`}).join('')}</div></div>`).join('')}</section>`).join('')}async function refresh(){try{render(await fetch('/api/top5-head-zero/catalog?v='+Date.now()).then(r=>r.json()))}catch(e){document.getElementById('status').textContent='结果读取失败，将自动重试'}}refresh();setInterval(refresh,10000);
</script></body></html>'''

EXTREME_ZERO_PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCK Top30 / Bottom30 分阶段消融</title><style>
:root{--paper:#e7dfd0;--ink:#17211d;--card:#fffdf7;--hot:#b74328;--cold:#206d7a;--line:#999184}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 5% 0,#cf684035,transparent 34rem),linear-gradient(125deg,#e7dfd0,#f4eee2);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(2380px,calc(100% - 24px));margin:auto;padding:26px 0 90px}h1{font:700 clamp(38px,5vw,74px)/.94 Georgia,serif;letter-spacing:-.045em;margin:8px 0}.eyebrow{color:var(--hot);font-size:11px;font-weight:900;letter-spacing:.14em}.lead{max-width:1300px;line-height:1.6;color:#5d655f}.toolbar{position:sticky;top:8px;z-index:5;display:flex;align-items:center;gap:12px;background:#fffdf7eb;border:1px solid var(--line);padding:10px 14px;backdrop-filter:blur(8px)}.toolbar select{min-width:min(720px,65vw);padding:8px;background:white;border:1px solid var(--line);font-weight:800}.status{margin-left:auto;font-size:12px;font-weight:800}.case{margin-top:34px;border-top:3px solid var(--ink);padding-top:14px}.case h2{font:700 27px Georgia,serif;margin:0;overflow-wrap:anywhere}.replay-fixed{position:fixed;right:24px;bottom:22px;z-index:20;border:0;background:var(--hot);color:#fff;padding:13px 18px;font-weight:900;cursor:pointer;box-shadow:0 8px 30px #37211855}.model{margin:18px 0 28px}.model>h3{font-size:17px;margin:0 0 9px}.band{margin:10px 0}.band-title{font-size:11px;font-weight:900;letter-spacing:.12em;margin:0 0 6px}.band-title.top{color:var(--hot)}.band-title.bottom{color:var(--cold)}.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.original-grid{display:grid;grid-template-columns:minmax(220px,1fr) repeat(4,minmax(0,1fr));gap:8px}.card{background:var(--card);border:1px solid var(--line);padding:8px}.card strong{display:block;margin-bottom:6px;font-size:12px}.card small{display:block;color:#69716c;margin-top:5px}video,.pending{display:block;width:100%;aspect-ratio:896/512;background:#111}.pending{display:grid;place-items:center;color:#c9c3b7;font-weight:900}.selection{margin:12px 0}.selection summary{cursor:pointer;font-weight:900}.back{float:right;color:var(--ink);font-weight:900}@media(max-width:1450px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:760px){.grid,.original-grid{grid-template-columns:1fr}.toolbar{align-items:stretch;flex-direction:column}.toolbar select{min-width:100%;width:100%}.status{margin-left:0}.back{float:none;display:block;margin-bottom:12px}.replay-fixed{right:12px;bottom:12px}}
</style></head><body><main><a class="back" href="/">返回可视化总览</a><div class="eyebrow">20 UNIQUE TEST_5 CASES · WAN2.2 BASELINE × WAN+LORA</div><h1>PCK 极值 Head<br>分阶段输出置零</h1><p class="lead">从 70 个 common T-head 中，按每个 Head 在三模型、50-case、所有时间步中的最佳 PCK@32 排序。每次只显示一个 case；下拉框切换后 URL 会记录 case，可直接分享或刷新。视频不循环，右下角按钮固定显示并同步重播当前 case。</p><details class="selection"><summary>查看 Top30 / Bottom30 Head 列表</summary><div id="selection"></div></details><div class="toolbar"><label>Case <select id="caseSelect"></select></label><span class="status" id="status">正在读取结果...</span></div><div id="cases"></div></main><button class="replay-fixed" onclick="replayCurrent()">当前 Case 全部从头播放</button><script>
const MODELS=[['baseline','Wan2.2 Baseline'],['lora','Wan + LoRA']];const STAGES=[['steps_00_10','S00–09'],['steps_10_20','S10–19'],['steps_20_30','S20–29'],['steps_30_40','S30–39'],['steps_00_40','S00–39']];let signature='',selected=new URLSearchParams(location.search).get('case')||'';const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function replayCurrent(){const videos=document.querySelectorAll('#cases video');videos.forEach(v=>{v.pause();v.currentTime=0});videos.forEach(v=>v.play().catch(()=>{}))}function media(c,m,v,title,note){const ready=(c.available[m]||[]).includes(v),src=`/api/pck-extreme-head-zero/video?model=${m}&case=${encodeURIComponent(c.case_key)}&variant=${v}`;return`<article class="card"><strong>${title}</strong>${ready?`<video controls muted playsinline preload="metadata" src="${src}"></video>`:`<div class="pending">生成中</div>`}<small>${note}</small></article>`}function caseHtml(c){return`<section class="case"><h2>${esc(c.case_key)}</h2>${MODELS.map(([m,label])=>`<div class="model"><h3>${label}</h3><div class="original-grid">${media(c,m,'original','Original','无消融')}</div><div class="band"><div class="band-title top">TOP30 同时置零</div><div class="grid">${STAGES.map(([v,t])=>media(c,m,'top30_'+v,t,'Top30')).join('')}</div></div><div class="band"><div class="band-title bottom">BOTTOM30 同时置零</div><div class="grid">${STAGES.map(([v,t])=>media(c,m,'bottom30_'+v,t,'Bottom30')).join('')}</div></div></div>`).join('')}</section>`}function render(data){if(!data.cases.some(c=>c.case_key===selected))selected=data.cases[0]?.case_key||'';const select=document.getElementById('caseSelect');if(select.options.length!==data.cases.length){select.innerHTML=data.cases.map(c=>`<option value="${esc(c.case_key)}">${esc(c.case_key)}</option>`).join('');select.onchange=()=>{selected=select.value;history.replaceState(null,'','?case='+encodeURIComponent(selected));signature='';render(data)}}select.value=selected;document.getElementById('status').textContent=`${data.ready_videos}/${data.expected_videos} 已就绪 · ${data.complete_cases}/${data.cases.length} case 完整`;if(data.selection){document.getElementById('selection').innerHTML=['top30','bottom30'].map(g=>`<p><b>${g.toUpperCase()}</b> · ${data.selection[g].map(x=>`L${String(x.block).padStart(2,'0')}/H${String(x.head).padStart(2,'0')} (${Number(x.macro_pck32).toFixed(2)})`).join(' · ')}</p>`).join('')}const c=data.cases.find(x=>x.case_key===selected),next=JSON.stringify([selected,c?.available]);if(next===signature)return;signature=next;document.getElementById('cases').innerHTML=c?caseHtml(c):''}async function refresh(){try{render(await fetch('/api/pck-extreme-head-zero/catalog?v='+Date.now()).then(r=>r.json()))}catch(e){document.getElementById('status').textContent='读取失败，将自动重试'}}refresh();setInterval(refresh,10000);
</script></body></html>'''

EXTREME30_PAGE = (
    EXTREME_ZERO_PAGE
    .replace("<title>PCK Top30 / Bottom30 分阶段消融</title>", "<title>All-720 PCK Top30 / Bottom30 分阶段消融</title>")
    .replace("从 70 个 common T-head 中", "从全部 720 个 block-head 中")
    .replace("/api/pck-extreme-head-zero/", "/api/pck-extreme30-head-zero/")
    .replace("PCK 极值 Head<br>分阶段输出置零", "All-720 PCK 极值 Head<br>分阶段输出置零")
)

EXTREME100_PAGE = (
    EXTREME30_PAGE
    .replace("Top30 / Bottom30", "Top100 / Bottom100")
    .replace("TOP30", "TOP100")
    .replace("BOTTOM30", "BOTTOM100")
    .replace("Top30", "Top100")
    .replace("Bottom30", "Bottom100")
    .replace("top30_", "top100_")
    .replace("bottom30_", "bottom100_")
    .replace("['top30','bottom30']", "['top100','bottom100']")
    .replace("/api/pck-extreme30-head-zero/", "/api/pck-extreme100-head-zero/")
)

INTERVAL_PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Joint 分区间 Head 热力图</title><style>
:root{--paper:#e8e1d2;--ink:#17201c;--card:#fffdf7;--rust:#bb4d30;--line:#9d9687;--muted:#69716c}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 4% 0,#ca755a3b,transparent 35rem),linear-gradient(120deg,#e8e1d2,#f3efe5);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(2280px,calc(100% - 24px));margin:auto;padding:26px 0 70px}h1{font:700 clamp(38px,5vw,72px)/.94 Georgia,serif;letter-spacing:-.04em;margin:8px 0}.eyebrow{color:var(--rust);font-size:11px;font-weight:900;letter-spacing:.15em}.lead{max-width:1100px;color:var(--muted);line-height:1.6}.back{float:right;color:var(--ink);font-weight:900}.section{margin-top:34px}.section-title{display:flex;align-items:end;justify-content:space-between;border-bottom:2px solid var(--ink);margin-bottom:10px}.section-title h2{font:700 30px Georgia,serif;margin:0 0 6px}.section-title span{font-size:11px;font-weight:900;letter-spacing:.12em;margin-bottom:8px}.grid{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);padding:10px;box-shadow:0 5px 18px #2b241711}.card h3{font:700 21px Georgia,serif;margin:0}.sample{float:right;color:var(--rust);font:900 10px "Trebuchet MS",sans-serif;letter-spacing:.1em}.score{display:grid;grid-template-columns:1fr 1fr;gap:3px 8px;margin:7px 0;color:var(--muted);font-size:10px}.score b{color:var(--ink)}.card img{display:block;width:100%;border:1px solid #c9c1b2;background:#111}.links{display:flex;gap:18px;margin-top:20px}.links a{color:var(--ink);font-weight:900}@media(max-width:1500px){.grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:760px){.grid{grid-template-columns:1fr}.back{float:none;display:block;margin-bottom:12px}}
</style></head><body><main><a class="back" href="/uniform-diagonal-curves?v=2">返回指标分布</a><div class="eyebrow">案例 001 · S039 · 9 个 JOINT 区间 · 每区间 6 个 PCK 分层样本</div><h1>查看每个<br>Joint 区间</h1><p class="lead">Joint 按 0.1 分箱，每个非空区间按 PCK@32 从低到高等分抽取 6 个 Block-Head。每张图从左到右为 GT teacher-forced、LoRA、Wan2.2 Baseline；白线标出 7 个潜空间帧边界，颜色表示 log10 Softmax 注意力质量。</p><div id="sections"></div><div class="links"><a href="/downloads/joint-interval-selection.csv">下载抽样明细 CSV</a><a href="/uniform-diagonal-curves?v=2">指标分布曲线</a></div></main><script>
const esc=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const f=x=>Number(x).toFixed(3);async function init(){const data=await fetch('/api/joint-interval/catalog').then(r=>r.json()),groups=Object.groupBy(data.combinations,x=>x.interval);document.getElementById('sections').innerHTML=Object.entries(groups).map(([interval,rows])=>`<section class="section"><div class="section-title"><h2>Joint ${esc(interval)}</h2><span>${rows[0].interval_count} 个组合</span></div><div class="grid">${rows.map(x=>{const b=String(x.block).padStart(2,'0'),h=String(x.head).padStart(2,'0'),src=`/api/joint-interval/strip?block=${x.block}&head=${x.head}&v=1`;return`<article class="card"><h3>L${b} / H${h}<span class="sample">SAMPLE ${x.sample_index}/6</span></h3><div class="score"><span>PCK@32 <b>${f(x.pck32)}</b></span><span>Joint <b>${f(x.joint)}</b></span><span>Diag mass <b>${f(x.diagonal_mass)}</b></span><span>Frame entropy <b>${f(x.diagonal_frame_entropy)}</b></span><span>Balanced <b>${f(x.balanced_diagonal)}</b></span></div><a href="${src}" target="_blank"><img loading="lazy" src="${src}" alt="L${b} H${h} 三模型注意力"></a></article>`}).join('')}</div></section>`).join('')}init();
</script></body></html>'''


BALANCED_INTERVAL_PAGE = (
    INTERVAL_PAGE
    .replace("<title>Joint 分区间 Head 热力图</title>", "<title>Balanced 分区间 Head 热力图</title>")
    .replace(
        "9 JOINT BINS · 6 PCK-STRATIFIED HEADS PER BIN",
        "9 BALANCED BINS · UP TO 6 PCK-STRATIFIED HEADS PER BIN",
    )
    .replace("查看每个<br>Joint 区间", "查看每个<br>Balanced 区间")
    .replace(
        "Joint 按 0.1 分箱，每个非空区间按 PCK@32 从低到高等分抽取 6 个 Block-Head。",
        "Balanced Diagonal 按 0.05 分箱，每个区间按 PCK@32 从低到高等分抽取最多 6 个 block-head；样本不足 6 个时全部展示。",
    )
    .replace(
        '<div id="sections"></div>',
        '''<section style="margin:28px 0 42px"><div class="section-title"><h2>三模型视频</h2><span>案例 001 · 40 步 · 随机种子 42</span></div><div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px"><article class="card"><h3>GT teacher-forced</h3><p style="font-size:11px;color:var(--muted)">Teacher-Forcing 使用的真实目标视频</p><video controls muted loop playsinline preload="metadata" style="display:block;width:100%;background:#111" src="/api/balanced-interval/video?model=gt"></video></article><article class="card"><h3>LoRA</h3><p style="font-size:11px;color:var(--muted)">step-000500 · 40 步生成</p><video controls muted loop playsinline preload="metadata" style="display:block;width:100%;background:#111" src="/api/balanced-interval/video?model=lora"></video></article><article class="card"><h3>Wan2.2 Baseline</h3><p style="font-size:11px;color:var(--muted)">基础模型 · 40 步生成</p><video controls muted loop playsinline preload="metadata" style="display:block;width:100%;background:#111" src="/api/balanced-interval/video?model=baseline"></video></article></div></section><div id="sections"></div>''',
    )
    .replace("Joint ${esc(interval)}", "Balanced ${esc(interval)}")
    .replace("${x.sample_index}/6", "${x.sample_index}/${rows.length}")
    .replace("/api/joint-interval", "/api/balanced-interval")
    .replace(
        "/downloads/joint-interval-selection.csv",
        "/downloads/balanced-interval-selection.csv",
    )
)


PCK_OVERLAY_PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PCK@32 高低组合轨迹对比</title><style>
:root{--paper:#ece7da;--ink:#17211e;--card:#fffdf7;--rust:#c65738;--teal:#176654;--line:#aaa392;--muted:#64706a}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 4% 0,#c6573830,transparent 35rem),radial-gradient(circle at 95% 4%,#17665425,transparent 32rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(2200px,calc(100% - 24px));margin:auto;padding:26px 0 70px}h1,h2,h3{font-family:Georgia,"Noto Serif CJK SC",serif}.back{float:right;color:var(--ink);font-weight:900}.eyebrow{color:var(--rust);font-size:11px;font-weight:900;letter-spacing:.15em}h1{font-size:clamp(40px,6vw,76px);line-height:.93;letter-spacing:-.04em;margin:8px 0}.lead{max-width:1100px;color:var(--muted);line-height:1.65}.formula{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:22px 0 36px}.formula article{background:var(--card);border:1px solid var(--line);padding:16px}.formula b{display:block;font:700 22px Georgia,serif;margin-bottom:7px}.formula p{margin:0;color:var(--muted);line-height:1.5}.section{margin-top:34px}.section-title{display:flex;align-items:end;justify-content:space-between;border-bottom:2px solid var(--ink);margin-bottom:11px}.section-title h2{font-size:32px;margin:0 0 6px}.section-title span{font-size:11px;font-weight:900;letter-spacing:.1em;margin-bottom:8px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);padding:11px;box-shadow:0 7px 20px #2d261715}.card h3{font-size:23px;margin:0 0 5px}.rank{float:right;color:var(--rust);font:900 11px "Trebuchet MS",sans-serif}.scores{display:grid;grid-template-columns:repeat(3,1fr);gap:4px 10px;margin:7px 0 10px;color:var(--muted);font-size:11px}.scores b{color:var(--ink)}video{display:block;width:100%;background:#0b100e;border:1px solid #333}.note{font-size:10px;color:var(--muted);line-height:1.45;margin:8px 0 0}.links{display:flex;gap:18px;margin-top:22px}.links a{color:var(--ink);font-weight:900}@media(max-width:1200px){.grid{grid-template-columns:1fr 1fr}}@media(max-width:750px){.formula,.grid{grid-template-columns:1fr}.back{float:none;display:block;margin-bottom:12px}.scores{grid-template-columns:1fr 1fr}}
</style></head><body><main><a class="back" href="/">返回可视化总览</a><div class="eyebrow">S039 · OBJECTS · 50 CASES · 3 MODELS · ORIGINAL VIDEO ANCHORS</div><h1>PCK@32 高低组合<br>轨迹 Overlay 对比</h1><p class="lead">白色圆环与白色轨迹是可见伪 GT，彩色点与轨迹是所选 Q@K Head 的预测。视频只使用原视频第 0、4、8、12、16、20、24 帧，对应7个真实潜空间锚点；不做时间插值。</p><section class="formula"><article><b>1. 单点误差</b><p>对有效 future 锚点，计算预测位置与 GT 位置的二维欧氏像素距离。</p></article><article><b>2. 案例 PCK@32</b><p>误差不超过32像素的有效比较次数 ÷ 该案例全部有效比较次数 × 100%。</p></article><article><b>3. 页面 Macro PCK@32</b><p>先在每个案例内合并目标区域，再让50个案例等权；最后对 GT、LoRA、Baseline 三模型等权平均。</p></article></section><div id="sections"></div><div class="links"><a href="/downloads/pck-extreme-selection.csv">下载 Top6/Bottom6 明细 CSV</a><a href="/all-steps/rankings?v=4">查看全部组合排名</a></div></main><script>
const f=x=>Number(x).toFixed(2);const pad=x=>String(x).padStart(2,'0');function cards(rows){return rows.map(x=>{const video=`/api/pck-extremes/video?group=${x.group}&rank=${x.rank}`,poster=`/api/pck-extremes/poster?group=${x.group}&rank=${x.rank}`;return`<article class="card"><h3>L${pad(x.block)} / H${pad(x.head)}<span class="rank">${x.group==='top'?'最高':'最低'} #${x.rank}</span></h3><div class="scores"><span>综合 Macro <b>${f(x.macro_pck32)}%</b></span><span>最差模型 <b>${f(x.worst_model_macro_pck32)}%</b></span><span>平均误差 <b>${f(x.macro_mean_error_px)}px</b></span><span>GT <b>${f(x.gt_macro_pck32)}%</b></span><span>LoRA <b>${f(x.lora_macro_pck32)}%</b></span><span>Baseline <b>${f(x.baseline_macro_pck32)}%</b></span></div><video controls muted loop playsinline preload="metadata" poster="${poster}" src="${video}"></video><p class="note">当前案例 PCK@32：GT ${f(x.case_pck32.gt)}% · LoRA ${f(x.case_pck32.lora)}% · Baseline ${f(x.case_pck32.baseline)}%</p></article>`}).join('')}async function init(){const data=await fetch('/api/pck-extremes/catalog').then(r=>r.json()),top=data.items.filter(x=>x.group==='top'),bottom=data.items.filter(x=>x.group==='bottom');document.getElementById('sections').innerHTML=`<section class="section"><div class="section-title"><h2>高 PCK Top 6</h2><span>三模型50-CASE MACRO PCK@32 由高到低</span></div><div class="grid">${cards(top)}</div></section><section class="section"><div class="section-title"><h2>低 PCK Bottom 6</h2><span>三模型50-CASE MACRO PCK@32 由低到高</span></div><div class="grid">${cards(bottom)}</div></section>`}init();
</script></body></html>'''


def catalog() -> dict:
    cases = sorted(
        path.name for path in (SOURCE_ROOT / "gt" / "cases").glob("case_*")
        if path.is_dir() and path.name.startswith(CASE_KEYS)
    )
    available = {}
    for model, _ in MODELS:
        available[model] = {}
        for case in cases:
            blocks = []
            for block, head in COMBINATIONS:
                group = "bottom30" if (block, head) in BOTTOM_COMBINATIONS else "top30"
                path = ROOT / group / model / "cases" / case / "all_token_qk" / f"block{block:02d}_selected_qk.npz"
                if path.is_file():
                    blocks.append(block)
            blocks = sorted(set(blocks))
            if blocks:
                available[model][case] = blocks
    return {
        "cases": cases,
        "models": [{"name": name, "label": label} for name, label in MODELS],
        "combinations": [
            {
                "block": block,
                "head": head,
                "group": "Bottom PCK@32" if (block, head) in BOTTOM_COMBINATIONS else "Top stable",
                "pck32": PCK32[(block, head)],
            }
            for block, head in COMBINATIONS
        ],
        "steps": list(range(40)),
        "available": available,
    }


def matrix_png(params: dict[str, list[str]]) -> bytes:
    model = params.get("model", [""])[0]
    case = params.get("case", [""])[0]
    block = int(params.get("block", ["-1"])[0])
    head = int(params.get("head", ["-1"])[0])
    step = int(params.get("step", ["-1"])[0])
    kind = params.get("kind", ["attention"])[0]
    if model not in dict(MODELS) or (block, head) not in COMBINATIONS:
        raise ValueError("invalid model or combination")
    group = "bottom30" if (block, head) in BOTTOM_COMBINATIONS else "top30"
    path = ROOT / group / model / "cases" / case / "all_token_qk" / f"block{block:02d}_selected_qk.npz"
    with np.load(path, allow_pickle=False) as data:
        heads = data["selected_heads"].astype(int).tolist()
        steps = data["steps_zero_based"].astype(int).tolist()
        hi = heads.index(head)
        si = steps.index(step)
        if kind == "raw":
            values = data["raw_qk_mean"][si, hi].astype(np.float32)
            limit = max(float(np.percentile(np.abs(values), 99.5)), 1e-8)
            cmap, vmin, vmax = "coolwarm", -limit, limit
        elif kind == "temporal":
            values = data["temporal_matrix"][si, hi].astype(np.float32)
            values = np.log10(np.maximum(values, 1e-8))
            cmap, vmin, vmax = "magma", float(np.percentile(values, 2)), float(np.percentile(values, 99.5))
        else:
            values = data["softmax_attention_mass"][si, hi].astype(np.float32)
            values = np.log10(np.maximum(values, 1e-8))
            cmap, vmin, vmax = "magma", float(np.percentile(values, 2)), float(np.percentile(values, 99.5))
    if vmax <= vmin:
        vmax = vmin + 1e-8
    fig, axis = plt.subplots(figsize=(7.2, 7.2), dpi=120)
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
    bins = values.shape[0]
    temporal_bins = 7
    for boundary in (time * bins / temporal_bins - 0.5 for time in range(1, temporal_bins)):
        axis.axhline(boundary, color="white", linewidth=.3, alpha=.7)
        axis.axvline(boundary, color="white", linewidth=.3, alpha=.7)
    axis.set_xlabel("key-token bin")
    axis.set_ylabel("query-token bin")
    axis.set_title(f"{dict(MODELS)[model]} · L{block:02d}/H{head:02d} · S{step:03d}")
    fig.colorbar(image, ax=axis, fraction=.046)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def contact_sheet_png(params: dict[str, list[str]]) -> bytes:
    case = params.get("case", [""])[0]
    block = int(params.get("block", ["-1"])[0])
    head = int(params.get("head", ["-1"])[0])
    kind = params.get("kind", ["attention"])[0]
    if (block, head) not in COMBINATIONS or kind not in {"attention", "raw", "temporal"}:
        raise ValueError("invalid combination or matrix type")
    tile = 184
    gap = 5
    label_width = 176
    group_header = 42
    columns, rows = 8, 5
    group_height = group_header + rows * (tile + gap) + 16
    width = label_width + columns * (tile + gap) + 15
    height = len(MODELS) * group_height + 18
    canvas = Image.new("RGB", (width, height), (10, 15, 13))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    any_available = False
    for model_index, (model, label) in enumerate(MODELS):
        top = 10 + model_index * group_height
        draw.text((12, top + 10), label, fill=(235, 239, 236), font=font)
        group = "bottom30" if (block, head) in BOTTOM_COMBINATIONS else "top30"
        path = ROOT / group / model / "cases" / case / "all_token_qk" / f"block{block:02d}_selected_qk.npz"
        if not path.is_file():
            draw.text((label_width, top + 10), "PENDING COMPUTE", fill=(218, 153, 117), font=font)
            continue
        any_available = True
        with np.load(path, allow_pickle=False) as data:
            heads = data["selected_heads"].astype(int).tolist()
            head_index = heads.index(head)
            steps = data["steps_zero_based"].astype(int).tolist()
            if kind == "raw":
                values = data["raw_qk_mean"][:, head_index].astype(np.float32)
                limit = max(float(np.percentile(np.abs(values), 99.5)), 1e-8)
                normalized = np.clip((values + limit) / (2.0 * limit), 0.0, 1.0)
                colormap = matplotlib.colormaps["coolwarm"]
            else:
                key = "temporal_matrix" if kind == "temporal" else "softmax_attention_mass"
                values = data[key][:, head_index].astype(np.float32)
                values = np.log10(np.maximum(values, 1e-8))
                low = float(np.percentile(values, 2.0))
                high = max(float(np.percentile(values, 99.5)), low + 1e-8)
                normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
                colormap = matplotlib.colormaps["magma"]
        for step_index, step in enumerate(steps):
            row, column = divmod(step_index, columns)
            x = label_width + column * (tile + gap)
            y = top + group_header + row * (tile + gap)
            rgba = colormap(normalized[step_index], bytes=True)
            image = Image.fromarray(rgba[:, :, :3], mode="RGB").resize(
                (tile, tile), Image.Resampling.NEAREST
            )
            image_draw = ImageDraw.Draw(image)
            source_bins = normalized.shape[-1]
            for time_index in range(1, 7):
                boundary = int(round(time_index * tile / 7.0))
                image_draw.line((boundary, 0, boundary, tile), fill=(255, 255, 255), width=1)
                image_draw.line((0, boundary, tile, boundary), fill=(255, 255, 255), width=1)
            image_draw.rectangle((2, 2, 37, 15), fill=(8, 12, 10))
            image_draw.text((5, 4), f"S{step:03d}", fill=(255, 255, 255), font=font)
            canvas.paste(image, (x, y))
        draw.text(
            (12, top + group_header),
            "5 rows x 8 steps\nwhite lines =\nlatent boundaries",
            fill=(155, 173, 164),
            font=font,
            spacing=5,
        )
    if not any_available:
        draw.text((label_width, 12), "No model output available yet. Refresh after a case completes.", fill=(218, 153, 117), font=font)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def s039_strip_png(params: dict[str, list[str]]) -> bytes:
    case = params.get("case", [""])[0]
    block = int(params.get("block", ["-1"])[0])
    head = int(params.get("head", ["-1"])[0])
    kind = params.get("kind", ["attention"])[0]
    if (block, head) not in COMBINATIONS or kind not in {"attention", "raw", "temporal"}:
        raise ValueError("invalid combination or matrix type")
    tile, header, gap = 300, 34, 5
    canvas = Image.new("RGB", (3 * tile + 2 * gap, tile + header), (8, 12, 10))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    group = "bottom30" if (block, head) in BOTTOM_COMBINATIONS else "top30"
    for model_index, (model, label) in enumerate(MODELS):
        x = model_index * (tile + gap)
        draw.text((x + 7, 11), label, fill=(235, 239, 236), font=font)
        path = ROOT / group / model / "cases" / case / "all_token_qk" / f"block{block:02d}_selected_qk.npz"
        if not path.is_file():
            draw.rectangle((x, header, x + tile - 1, header + tile - 1), fill=(31, 40, 35))
            draw.text((x + 100, header + 142), "PENDING", fill=(218, 153, 117), font=font)
            continue
        with np.load(path, allow_pickle=False) as data:
            head_index = data["selected_heads"].astype(int).tolist().index(head)
            step_index = data["steps_zero_based"].astype(int).tolist().index(39)
            if kind == "raw":
                values = data["raw_qk_mean"][step_index, head_index].astype(np.float32)
                limit = max(float(np.percentile(np.abs(values), 99.5)), 1e-8)
                normalized = np.clip((values + limit) / (2.0 * limit), 0.0, 1.0)
                colormap = matplotlib.colormaps["coolwarm"]
            else:
                key = "temporal_matrix" if kind == "temporal" else "softmax_attention_mass"
                values = data[key][step_index, head_index].astype(np.float32)
                values = np.log10(np.maximum(values, 1e-8))
                low = float(np.percentile(values, 2.0))
                high = max(float(np.percentile(values, 99.5)), low + 1e-8)
                normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
                colormap = matplotlib.colormaps["magma"]
        rgba = colormap(normalized, bytes=True)
        image = Image.fromarray(rgba[:, :, :3], mode="RGB").resize(
            (tile, tile), Image.Resampling.NEAREST
        )
        image_draw = ImageDraw.Draw(image)
        for time_index in range(1, 7):
            boundary = int(round(time_index * tile / 7.0))
            image_draw.line((boundary, 0, boundary, tile), fill=(255, 255, 255), width=1)
            image_draw.line((0, boundary, tile, boundary), fill=(255, 255, 255), width=1)
        canvas.paste(image, (x, header))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def send_file_with_range(handler, path: Path, content_type: str) -> None:
    size = path.stat().st_size
    range_header = handler.headers.get("Range")
    if not range_header:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                handler.wfile.write(chunk)
        return
    unit, value = range_header.split("=", 1)
    if unit.strip() != "bytes" or "," in value:
        raise ValueError(f"unsupported range: {range_header}")
    start_text, end_text = value.split("-", 1)
    start = int(start_text) if start_text else max(0, size - int(end_text))
    end = int(end_text) if end_text else size - 1
    start, end = max(0, start), min(size - 1, end)
    if start > end:
        raise ValueError(f"invalid range: {range_header}")
    handler.send_response(206)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(end - start + 1))
    handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    handler.send_header("Accept-Ranges", "bytes")
    handler.end_headers()
    with path.open("rb") as stream:
        stream.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            handler.wfile.write(chunk)
            remaining -= len(chunk)


def extreme30_catalog() -> dict:
    list_path = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
    cases, seen = [], set()
    variants = ["original"] + [
        f"{group}_{stage}"
        for group in ("top30", "bottom30")
        for stage in ("steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40")
    ]
    ready = complete = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        case_key = Path(line).stem if line else ""
        if not case_key or case_key in seen:
            continue
        seen.add(case_key)
        available = {}
        for model in ("baseline", "lora"):
            case_root = EXTREME30_ROOT / model / "cases" / case_key
            available[model] = [variant for variant in variants if (case_root / f"{variant}.mp4").is_file()]
            ready += len(available[model])
        if all(len(available[model]) == len(variants) for model in ("baseline", "lora")):
            complete += 1
        cases.append({"case_key": case_key, "available": available})
    selection_path = EXTREME30_ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    return {"cases": cases, "selection": selection, "ready_videos": ready, "expected_videos": len(cases) * 22, "complete_cases": complete}


def extreme100_catalog() -> dict:
    list_path = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
    variants = ["original"] + [f"{group}_{stage}" for group in ("top100", "bottom100") for stage in ("steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40")]
    cases, seen = [], set()
    ready = complete = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        case_key = Path(line.strip()).stem
        if not case_key or case_key in seen:
            continue
        seen.add(case_key)
        available = {}
        for model in ("baseline", "lora"):
            case_root = EXTREME100_ROOT / model / "cases" / case_key
            available[model] = [variant for variant in variants if (case_root / f"{variant}.mp4").is_file()]
            ready += len(available[model])
        if all(len(available[model]) == len(variants) for model in ("baseline", "lora")):
            complete += 1
        cases.append({"case_key": case_key, "available": available})
    selection_path = EXTREME100_ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    return {"cases": cases, "selection": selection, "ready_videos": ready, "expected_videos": len(cases) * 22, "complete_cases": complete}


def head_zero_catalog() -> dict:
    initial = [
        "case_001_ball_roll", "case_002_puck_slide", "case_003_capsule_slide",
        "case_004_cylinder_topple", "case_005_box_slide",
    ]
    list_path = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
    test5 = []
    seen = set()
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        stem = Path(line).stem
        if stem not in seen:
            seen.add(stem)
            test5.append(stem)
    stages = ["original", "steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40"]
    cases = []
    ready = 0
    complete = 0
    for case_key in [*initial, *test5]:
        available = {}
        for model in ("baseline", "lora"):
            case_root = HEAD_ZERO_ROOT / model / "cases" / case_key
            available[model] = [stage for stage in stages if (case_root / f"{stage}.mp4").is_file()]
            ready += len(available[model])
        if all(len(available[model]) == len(stages) for model in ("baseline", "lora")):
            complete += 1
        cases.append({"case_key": case_key, "group": "ToyDataset" if case_key in initial else "test_5.txt", "available": available})
    return {"cases": cases, "ready_videos": ready, "expected_videos": len(cases) * 12, "complete_cases": complete}


def extreme_zero_catalog() -> dict:
    list_path = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
    cases, seen = [], set()
    variants = ["original"] + [f"{group}_{stage}" for group in ("top30", "bottom30") for stage in ("steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40")]
    ready = complete = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        stem = Path(line.strip()).stem
        if not stem or stem in seen:
            continue
        seen.add(stem)
        available = {}
        for model in ("baseline", "lora"):
            case_root = EXTREME_ZERO_ROOT / model / "cases" / stem
            available[model] = [v for v in variants if (case_root / f"{v}.mp4").is_file()]
            ready += len(available[model])
        if all(len(available[m]) == len(variants) for m in ("baseline", "lora")):
            complete += 1
        cases.append({"case_key": stem, "available": available})
    selection_path = EXTREME_ZERO_ROOT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else None
    return {"cases": cases, "ready_videos": ready, "expected_videos": len(cases) * 22, "complete_cases": complete, "selection": selection}


def neighbor_diagonal_catalog():
    summary = NEIGHBOR_ROOT / "all720_neighbor_diagonal_summary.csv"
    records = []
    if summary.is_file():
        with summary.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(
                    {
                        "block": int(row["block"]),
                        "head": int(row["head"]),
                        "strict_rank": int(row["strict_rank"]),
                        "pck32": float(row["pck32"]),
                        "mass": float(row["neighbor3_diagonal_mass"]),
                        "uniformity": float(row["neighbor3_diagonal_uniformity"]),
                        "joint": float(row["neighbor3_joint"]),
                        "balanced": float(row["neighbor3_balanced_diagonal"]),
                        "strict_score": float(
                            row["neighbor3_allblock_diagonal_score"]
                        ),
                        "allblock_purity": float(row["allblock_diagonal_purity"]),
                        "allblock_min_purity": float(
                            row["allblock_min_diagonal_purity"]
                        ),
                        "balanced_std": float(row["neighbor3_balanced_diagonal_std"]),
                        "gt_balanced": float(row["gt_neighbor3_balanced_diagonal"]),
                        "lora_balanced": float(row["lora_neighbor3_balanced_diagonal"]),
                        "baseline_balanced": float(row["baseline_neighbor3_balanced_diagonal"]),
                        "gt_strict": float(
                            row["gt_neighbor3_allblock_diagonal_score"]
                        ),
                        "lora_strict": float(
                            row["lora_neighbor3_allblock_diagonal_score"]
                        ),
                        "baseline_strict": float(
                            row["baseline_neighbor3_allblock_diagonal_score"]
                        ),
                        "gt_allblock_purity": float(
                            row["gt_allblock_diagonal_purity"]
                        ),
                        "lora_allblock_purity": float(
                            row["lora_allblock_diagonal_purity"]
                        ),
                        "baseline_allblock_purity": float(
                            row["baseline_allblock_diagonal_purity"]
                        ),
                        "gt_allblock_min_purity": float(
                            row["gt_allblock_min_diagonal_purity"]
                        ),
                        "lora_allblock_min_purity": float(
                            row["lora_allblock_min_diagonal_purity"]
                        ),
                        "baseline_allblock_min_purity": float(
                            row["baseline_allblock_min_diagonal_purity"]
                        ),
                        "gt_pck32": float(row["gt_pck32"]),
                        "lora_pck32": float(row["lora_pck32"]),
                        "baseline_pck32": float(row["baseline_pck32"]),
                    }
                )
    status = {
        model: (NEIGHBOR_ROOT / "status" / f"{model}.complete").is_file()
        for model in ("gt", "lora", "baseline")
    }
    return {
        "records": records,
        "model_complete": status,
        "rendered_heatmaps": len(list((NEIGHBOR_HEAT_ROOT / "web").glob("*.png"))),
        "pipeline_complete": (NEIGHBOR_ROOT / "status/pipeline.complete").is_file(),
        "ranking_metric": (
            "neighbor-3 balanced diagonal × weakest all-frame block diagonal purity"
        ),
    }


NEIGHBOR_DIAGONAL_PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>720 Head Strict All-Block Diagonal Ranking</title><style>
:root{--ink:#17231e;--paper:#eee8d9;--card:#fffdf7;--line:#c7bca6;--red:#b64c34;--green:#176b61}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 7% 0,#efb87b55,transparent 30rem),radial-gradient(circle at 96% 5%,#6fa38c55,transparent 36rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:5;padding:16px 22px;background:#eee8d9ee;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}h1{margin:0;font-size:clamp(25px,4vw,45px)}header p{margin:6px 0;max-width:1100px}.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}select,button,a.download{padding:8px 12px;border:1px solid var(--line);background:#fff;color:var(--ink);font-weight:800;text-decoration:none}.status{font:12px ui-monospace,monospace}main{max-width:1800px;margin:auto;padding:18px}.curve{display:block;width:100%;max-height:850px;object-fit:contain;background:#fff;border:1px solid var(--line)}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;margin-top:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:5px 18px 5px 5px;padding:12px;box-shadow:0 10px 28px #4e47391c}.card h2{margin:0 0 8px;font:900 19px "Trebuchet MS",sans-serif}.rank{color:var(--red)}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.pill{padding:5px 7px;border-radius:99px;background:#e9e1d1;font:11px ui-monospace,monospace}.card img{display:block;width:100%;background:#171a18;border:1px solid var(--line)}.pending{padding:60px 20px;text-align:center;border:1px dashed var(--line);background:var(--card)}.matrix-section{margin-top:18px;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}.matrix-head{display:flex;justify-content:space-between;gap:15px;align-items:end;padding:13px 15px;border-bottom:1px solid var(--line)}.matrix-head h2{margin:0;font:900 17px "Trebuchet MS",sans-serif}.heat-controls{display:flex;gap:9px;flex-wrap:wrap}.heat-controls label{font:700 10px ui-monospace,monospace}.scroll{overflow:auto}.heat{display:grid;grid-template-columns:48px repeat(24,minmax(34px,1fr));gap:3px;min-width:980px;padding:12px}.axis,.cell{display:flex;align-items:center;justify-content:center;height:29px;border-radius:4px;font:9px ui-monospace,monospace}.axis{color:#6d756f}.cell{border:0;min-width:0;padding:1px;cursor:pointer;color:white;text-shadow:0 1px 2px #000}.cell:hover{outline:2px solid #17231e;outline-offset:1px}.pager{display:flex;justify-content:center;gap:12px;align-items:center;margin:22px 0}@media(max-width:1000px){.grid{grid-template-columns:1fr}header{position:static}main{padding:10px}}
</style></head><body><header><h1>720 Head · 全帧块空间对角线严格排序</h1><p>严格主分数 = 相邻三帧对角线均衡 × 全部 7 个目标帧块中的最弱空间对角线纯度，并在内部 query 帧、3 模型 × 5 case 上取均值。高分要求本帧和相邻帧对角线均衡，同时所有 49 个帧块内部都只沿空间对角线亮；金线标出每个帧块的空间对角线，青框标出相邻三帧区域。</p><div class="tools"><label>排序 <select id="sort"><option value="strict_score">Strict Combined</option><option value="allblock_purity">全帧块对角纯度</option><option value="allblock_min_purity">最弱帧块纯度</option><option value="balanced">Neighbor Balanced</option><option value="uniformity">纯均匀度</option><option value="joint">质量 × 均匀度</option><option value="mass">三帧总质量</option><option value="pck32">PCK@32</option></select></label><button id="refresh">手动刷新</button><a class="download" href="/downloads/all720-neighbor-diagonal.csv">下载 720 行 CSV</a><span id="status" class="status">等待结果</span></div></header><main><img id="curve" class="curve" src="/api/neighbor-diagonal/curve" onerror="this.style.display='none'"><section class="matrix-section"><div class="matrix-head"><div><h2>当前时间步 S039 · 30 × 24 性能图</h2><span class="status" id="heatMeta">等待 720-head 汇总</span></div><div class="heat-controls"><label>模型 <select id="heatModel"><option value="combined">三模型综合</option><option value="gt">GT teacher-forced</option><option value="lora">LoRA</option><option value="baseline">Wan2.2 Baseline</option></select></label><label>指标 <select id="heatMetric"><option value="strict_score">Strict Score</option><option value="allblock_purity">全帧块对角纯度</option><option value="allblock_min_purity">最弱帧块纯度</option><option value="balanced">Neighbor Balanced</option><option value="pck32">PCK</option></select></label></div></div><div class="scroll"><div class="heat" id="heat"><span class="pending">计算中</span></div></div></section><section id="grid" class="grid"><div class="pending">计算中</div></section><div class="pager"><button id="prev">上一页</button><b id="page"></b><button id="next">下一页</button></div></main><script>
const f=(x,n=4)=>Number(x).toFixed(n),pageSize=24;let records=[],page=0,state={};const heatLabels={strict_score:"Strict Score",allblock_purity:"All-block purity",allblock_min_purity:"Weakest-block purity",balanced:"Neighbor Balanced",pck32:"PCK@32"};function heatValue(r){const model=document.getElementById("heatModel").value,metric=document.getElementById("heatMetric").value;if(model==="combined")return r[metric];const suffix={strict_score:"strict",allblock_purity:"allblock_purity",allblock_min_purity:"allblock_min_purity",balanced:"balanced",pck32:"pck32"}[metric];return r[`${model}_${suffix}`]}function heatColor(value,low,high){const t=high>low?Math.max(0,Math.min(1,(value-low)/(high-low))):.5;return `hsl(${12+120*t} 58% ${34+7*t}%)`}function renderHeat(){const box=document.getElementById("heat");if(!records.length){box.innerHTML=`<span class="pending">三模型计算完成后显示 30 × 24 矩阵</span>`;return}const metric=document.getElementById("heatMetric").value,model=document.getElementById("heatModel").value,values=records.map(heatValue).filter(Number.isFinite),low=Math.min(...values),high=Math.max(...values),map=new Map(records.map(r=>[`${r.block}-${r.head}`,r]));box.innerHTML=`<span class="axis">L\H</span>`+[...Array(24)].map((_,h)=>`<span class="axis">H${String(h).padStart(2,"0")}</span>`).join("");for(let l=0;l<30;l++){box.insertAdjacentHTML("beforeend",`<span class="axis">L${String(l).padStart(2,"0")}</span>`);for(let h=0;h<24;h++){const r=map.get(`${l}-${h}`),button=document.createElement("button"),value=r?heatValue(r):NaN;button.className="cell";if(Number.isFinite(value)){button.style.background=heatColor(value,low,high);button.textContent=f(value,metric==="pck32"?1:3);button.title=`S039 · L${String(l).padStart(2,"0")} H${String(h).padStart(2,"0")} · ${heatLabels[metric]} ${f(value)}`;button.onclick=()=>{const index=ordered().findIndex(x=>x.block===l&&x.head===h);if(index>=0){page=Math.floor(index/pageSize);render();requestAnimationFrame(()=>document.querySelector(`[data-combo="${l}-${h}"]`)?.scrollIntoView({behavior:"smooth",block:"center"}))}}}else{button.disabled=true;button.textContent="-"}box.append(button)}}document.getElementById("heatMeta").textContent=`S039 · ${model==="combined"?"三模型综合":model} · ${heatLabels[metric]} · ${f(low)} to ${f(high)} · 绿色表示更高`}function ordered(){const key=document.getElementById('sort').value;return [...records].sort((a,b)=>b[key]-a[key])}function render(){renderHeat();const rows=ordered(),pages=Math.max(1,Math.ceil(rows.length/pageSize));page=Math.min(page,pages-1);const shown=rows.slice(page*pageSize,(page+1)*pageSize);document.getElementById('page').textContent=`${page+1} / ${pages}`;document.getElementById('grid').innerHTML=shown.length?shown.map((r,i)=>`<article class="card" data-combo="${r.block}-${r.head}"><h2><span class="rank">#${page*pageSize+i+1}</span> · L${String(r.block).padStart(2,'0')} / H${String(r.head).padStart(2,'0')}</h2><div class="metrics"><span class="pill">Strict ${f(r.strict_score)}</span><span class="pill">All-block purity ${f(r.allblock_purity)}</span><span class="pill">Min purity ${f(r.allblock_min_purity)}</span><span class="pill">Neighbor balanced ${f(r.balanced)}</span><span class="pill">Uniformity ${f(r.uniformity)}</span><span class="pill">Mass ${f(r.mass)}</span><span class="pill">Joint ${f(r.joint)}</span><span class="pill">PCK@32 ${f(r.pck32,2)}</span><span class="pill">Strict GT/L/BL ${f(r.gt_strict)}/${f(r.lora_strict)}/${f(r.baseline_strict)}</span></div><img loading="lazy" src="/api/neighbor-diagonal/strip?block=${r.block}&head=${r.head}" alt="L${r.block} H${r.head} attention"></article>`).join(''):`<div class="pending">三模型 720-head 指标与热力图正在生成。点击手动刷新查看进度。</div>`}async function load(){state=await fetch('/api/neighbor-diagonal/catalog',{cache:'no-store'}).then(r=>r.json());records=state.records;document.getElementById('status').textContent=`排名 ${records.length}/720 · 热力图 ${state.rendered_heatmaps}/720 · GT ${state.model_complete.gt?'完成':'运行中'} · LoRA ${state.model_complete.lora?'完成':'运行中'} · Baseline ${state.model_complete.baseline?'完成':'运行中'}`;render()}document.getElementById('heatModel').addEventListener('change',renderHeat);document.getElementById('heatMetric').addEventListener('change',renderHeat);document.getElementById('sort').addEventListener('change',()=>{page=0;render()});document.getElementById('refresh').addEventListener('click',load);document.getElementById('prev').addEventListener('click',()=>{page=Math.max(0,page-1);render();scrollTo(0,0)});document.getElementById('next').addEventListener('click',()=>{page++;render();scrollTo(0,0)});load().catch(e=>document.getElementById('status').textContent=e);
</script></body></html>'''


class Handler(BASE_HANDLER):
    protocol_version = "HTTP/1.1"

    def send_payload(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def send_video_file(self, path: Path) -> None:
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get("Range")
        if range_header:
            if not range_header.startswith("bytes=") or "," in range_header:
                self.send_error(416, "unsupported byte range")
                return
            bounds = range_header[6:].split("-", 1)
            if bounds[0]:
                start = int(bounds[0])
                end = min(int(bounds[1]) if bounds[1] else size - 1, size - 1)
            else:
                length = min(int(bounds[1]), size)
                start, end = size - length, size - 1
            if start < 0 or start >= size or end < start:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=3600")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                return self.send_payload(PORTAL.encode(), "text/html; charset=utf-8")
            if parsed.path == "/all-token-qk":
                return self.send_payload(PAGE_S039.encode(), "text/html; charset=utf-8")
            if parsed.path == "/uniform-diagonal-curves":
                return self.send_payload(CURVE_PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/neighbor-diagonal-ranking":
                return self.send_payload(
                    NEIGHBOR_DIAGONAL_PAGE.encode(), "text/html; charset=utf-8"
                )
            if parsed.path == "/top5-head-zero-ablation":
                return self.send_payload(HEAD_ZERO_PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/pck-extreme-head-zero-ablation":
                return self.send_payload(EXTREME_ZERO_PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/pck-extreme30-head-zero-ablation":
                return self.send_payload(EXTREME30_PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/pck-extreme100-head-zero-ablation":
                return self.send_payload(EXTREME100_PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/joint-interval-heatmaps":
                return self.send_payload(INTERVAL_PAGE.encode(), "text/html; charset=utf-8")
            if parsed.path == "/balanced-interval-heatmaps":
                return self.send_payload(
                    BALANCED_INTERVAL_PAGE.encode(), "text/html; charset=utf-8"
                )
            if parsed.path == "/pck-extreme-overlays":
                return self.send_payload(
                    PCK_OVERLAY_PAGE.encode(), "text/html; charset=utf-8"
                )
            if parsed.path == "/api/pck-extremes/catalog":
                return self.send_payload(
                    (PCK_OVERLAY_ROOT / "catalog.json").read_bytes(),
                    "application/json; charset=utf-8",
                )
            if parsed.path in ("/api/pck-extremes/video", "/api/pck-extremes/poster"):
                query = parse_qs(parsed.query)
                group = query.get("group", [""])[0]
                rank = int(query.get("rank", ["0"])[0])
                if group not in ("top", "bottom") or not 1 <= rank <= 6:
                    raise ValueError("invalid PCK extreme selection")
                folder = "videos" if parsed.path.endswith("video") else "posters"
                suffix = ".mp4" if folder == "videos" else ".jpg"
                matches = list((PCK_OVERLAY_ROOT / folder).glob(f"{group}_rank{rank:02d}_*{suffix}"))
                if len(matches) != 1:
                    raise FileNotFoundError(f"missing PCK overlay: {group} rank {rank}")
                if folder == "videos":
                    return self.send_video_file(matches[0])
                return self.send_payload(matches[0].read_bytes(), "image/jpeg")
            if parsed.path == "/api/joint-interval/catalog":
                return self.send_payload(
                    (INTERVAL_ROOT / "selected_heads.json").read_bytes(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/api/balanced-interval/catalog":
                return self.send_payload(
                    (BALANCED_INTERVAL_ROOT / "selected_heads.json").read_bytes(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/api/balanced-interval/video":
                model = parse_qs(parsed.query).get("model", [""])[0]
                if model not in BALANCED_VIDEO_PATHS:
                    raise ValueError(f"unknown video model: {model}")
                return self.send_video_file(BALANCED_VIDEO_PATHS[model])
            if parsed.path == "/api/joint-interval/strip":
                query = parse_qs(parsed.query)
                block = int(query["block"][0])
                head = int(query["head"][0])
                return self.send_payload(
                    (INTERVAL_ROOT / "web" / f"block{block:02d}_head{head:02d}.png").read_bytes(),
                    "image/png",
                )
            if parsed.path == "/api/balanced-interval/strip":
                query = parse_qs(parsed.query)
                block = int(query["block"][0])
                head = int(query["head"][0])
                return self.send_payload(
                    (BALANCED_INTERVAL_ROOT / "web" / f"block{block:02d}_head{head:02d}.png").read_bytes(),
                    "image/png",
                )
            if parsed.path == "/api/all-token/catalog":
                payload = json.dumps(catalog(), ensure_ascii=False).encode()
                return self.send_payload(payload, "application/json; charset=utf-8")
            if parsed.path == "/api/all-token/matrix":
                return self.send_payload(matrix_png(parse_qs(parsed.query)), "image/png")
            if parsed.path == "/api/all-token/contact-sheet":
                return self.send_payload(contact_sheet_png(parse_qs(parsed.query)), "image/png")
            if parsed.path == "/api/all-token/s039-strip":
                return self.send_payload(s039_strip_png(parse_qs(parsed.query)), "image/png")
            if parsed.path == "/api/all720/curve":
                return self.send_payload(
                    (ALL720_ROOT / "all720_uniform_diagonal_curves.png").read_bytes(),
                    "image/png",
                )
            if parsed.path == "/api/all720/count-distribution":
                return self.send_payload(
                    (ALL720_ROOT / "all720_metric_count_distribution.png").read_bytes(),
                    "image/png",
                )
            if parsed.path == "/api/neighbor-diagonal/catalog":
                return self.send_payload(
                    json.dumps(neighbor_diagonal_catalog(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/api/neighbor-diagonal/curve":
                return self.send_payload(
                    (NEIGHBOR_ROOT / "all720_neighbor_diagonal_curves.png").read_bytes(),
                    "image/png",
                )
            if parsed.path == "/api/neighbor-diagonal/strip":
                query = parse_qs(parsed.query)
                block = int(query["block"][0])
                head = int(query["head"][0])
                if not 0 <= block < 30 or not 0 <= head < 24:
                    raise ValueError("invalid neighbor diagonal block/head")
                return self.send_payload(
                    (NEIGHBOR_HEAT_ROOT / "web" / f"block{block:02d}_head{head:02d}.png").read_bytes(),
                    "image/png",
                )
            if parsed.path == "/api/top5-head-zero/video":
                query = parse_qs(parsed.query)
                model = query.get("model", [""])[0]
                case = query.get("case", [""])[0]
                variant = query.get("variant", [""])[0]
                allowed_variants = {
                    "original", "steps_00_10", "steps_10_20", "steps_20_30",
                    "steps_30_40", "steps_00_40",
                }
                if model not in {"baseline", "lora"} or not case or case != Path(case).name or case in {".", ".."} or variant not in allowed_variants:
                    raise ValueError("invalid top5-head-zero video request")
                return send_file_with_range(
                    self, HEAD_ZERO_ROOT / model / "cases" / case / f"{variant}.mp4", "video/mp4"
                )
            if parsed.path == "/api/top5-head-zero/catalog":
                return self.send_payload(
                    json.dumps(head_zero_catalog(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/api/pck-extreme-head-zero/video":
                query = parse_qs(parsed.query)
                model = query.get("model", [""])[0]
                case = query.get("case", [""])[0]
                variant = query.get("variant", [""])[0]
                allowed = {"original"} | {f"{group}_{stage}" for group in ("top30", "bottom30") for stage in ("steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40")}
                if model not in {"baseline", "lora"} or not case or case != Path(case).name or case in {".", ".."} or variant not in allowed:
                    raise ValueError("invalid PCK extreme head-zero video request")
                return send_file_with_range(self, EXTREME_ZERO_ROOT / model / "cases" / case / f"{variant}.mp4", "video/mp4")
            if parsed.path == "/api/pck-extreme-head-zero/catalog":
                return self.send_payload(
                    json.dumps(extreme_zero_catalog(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/api/pck-extreme30-head-zero/video":
                query = parse_qs(parsed.query)
                model = query.get("model", [""])[0]
                case = query.get("case", [""])[0]
                variant = query.get("variant", [""])[0]
                allowed = {"original"} | {f"{group}_{stage}" for group in ("top30", "bottom30") for stage in ("steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40")}
                if model not in {"baseline", "lora"} or not case or case != Path(case).name or case in {".", ".."} or variant not in allowed:
                    raise ValueError("invalid all720 PCK extreme head-zero video request")
                return send_file_with_range(self, EXTREME30_ROOT / model / "cases" / case / f"{variant}.mp4", "video/mp4")
            if parsed.path == "/api/pck-extreme30-head-zero/catalog":
                return self.send_payload(
                    json.dumps(extreme30_catalog(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/api/pck-extreme100-head-zero/video":
                query = parse_qs(parsed.query)
                model = query.get("model", [""])[0]
                case = query.get("case", [""])[0]
                variant = query.get("variant", [""])[0]
                allowed = {"original"} | {f"{group}_{stage}" for group in ("top100", "bottom100") for stage in ("steps_00_10", "steps_10_20", "steps_20_30", "steps_30_40", "steps_00_40")}
                if model not in {"baseline", "lora"} or not case or case != Path(case).name or case in {".", ".."} or variant not in allowed:
                    raise ValueError("invalid all720 PCK extreme100 head-zero video request")
                return send_file_with_range(self, EXTREME100_ROOT / model / "cases" / case / f"{variant}.mp4", "video/mp4")
            if parsed.path == "/api/pck-extreme100-head-zero/catalog":
                return self.send_payload(
                    json.dumps(extreme100_catalog(), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            if parsed.path == "/downloads/all720-uniform-diagonal.csv":
                return self.send_payload(
                    (ALL720_ROOT / "all720_uniform_diagonal_summary.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                )
            if parsed.path == "/downloads/all720-count-distribution.csv":
                return self.send_payload(
                    (ALL720_ROOT / "all720_metric_count_distribution.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                )
            if parsed.path == "/downloads/all720-neighbor-diagonal.csv":
                return self.send_payload(
                    (NEIGHBOR_ROOT / "all720_neighbor_diagonal_summary.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                )
            if parsed.path == "/downloads/joint-interval-selection.csv":
                return self.send_payload(
                    (INTERVAL_ROOT / "selected_heads.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                )
            if parsed.path == "/downloads/balanced-interval-selection.csv":
                return self.send_payload(
                    (BALANCED_INTERVAL_ROOT / "selected_heads.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                )
            if parsed.path == "/downloads/pck-extreme-selection.csv":
                return self.send_payload(
                    (PCK_OVERLAY_ROOT / "selection.csv").read_bytes(),
                    "text/csv; charset=utf-8",
                )
            return super().do_GET()
        except (FileNotFoundError, ValueError) as error:
            return self.send_payload(str(error).encode(), "text/plain; charset=utf-8", 404)
        except Exception as error:
            return self.send_payload(repr(error).encode(), "text/plain; charset=utf-8", 500)


PCK_COMPARISON_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_high_low_overlay_case001_s039"
)


_previous_do_get = Handler.do_GET


def _do_get_with_pck_overlay(self) -> None:
    parsed = urlparse(self.path)
    route = parsed.path.rstrip("/")
    if route == "/pck-overlay-comparison":
        page = (PCK_COMPARISON_ROOT / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)
        return
    prefix = "/pck-overlay-comparison/assets/"
    if parsed.path.startswith(prefix):
        name = Path(parsed.path[len(prefix) :]).name
        path = PCK_COMPARISON_ROOT / name
        if not path.is_file():
            self.send_error(404, "PCK overlay asset not found")
            return
        if path.suffix.lower() == ".mp4":
            return send_file_with_range(self, path, "video/mp4")
        body = path.read_bytes()
        content_type = "application/json; charset=utf-8" if path.suffix == ".json" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
    _previous_do_get(self)


Handler.do_GET = _do_get_with_pck_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    print(f"serving DiffTrack atlas at http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
