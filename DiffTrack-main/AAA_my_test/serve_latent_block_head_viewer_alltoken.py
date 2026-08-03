#!/usr/bin/env python3
"""Extend the existing 8790 atlas with stable-head all-token Q/K matrices."""

from __future__ import annotations

import argparse
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
SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case"
)
TOP_COMBINATIONS = ((20, 9), (20, 17), (26, 7), (18, 11), (24, 6), (19, 0))
BOTTOM_COMBINATIONS = ((0, 9), (1, 17), (1, 23), (8, 13), (1, 22), (2, 9))
COMBINATIONS = TOP_COMBINATIONS + BOTTOM_COMBINATIONS
BOTTOM_PCK32 = {
    (0, 9): 0.187529,
    (1, 17): 0.779056,
    (1, 23): 1.199664,
    (8, 13): 2.475224,
    (1, 22): 3.007380,
    (2, 9): 3.478668,
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


PORTAL = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DiffTrack atlas</title><style>
:root{--paper:#eee9dc;--ink:#17211e;--card:#fffdf8;--line:#b8b09f;--rust:#b64a31;--teal:#176654}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 5% 0,#d7764b35,transparent 36rem),radial-gradient(circle at 95% 5%,#4b9a8035,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1400px,calc(100% - 28px));margin:auto;padding:35px 0 70px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}h1{font-size:clamp(42px,7vw,90px);line-height:.9;margin:8px 0 18px;letter-spacing:-.05em}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.16em;font-size:12px}.lead{max-width:900px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:28px}.card{display:flex;min-height:260px;flex-direction:column;justify-content:space-between;padding:22px;background:var(--card);border:1px solid var(--line);color:inherit;text-decoration:none;border-radius:3px 28px 3px 3px}.card.new{background:#13251f;color:white;border-color:#13251f}.card h2{font-size:30px;margin:8px 0}.card p{line-height:1.55;opacity:.76}.go{font-size:12px;font-weight:900;letter-spacing:.08em;color:var(--rust)}.new .go{color:#e39a78}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="eyebrow">DIFFTRACK · THREE-MODEL ATTENTION ATLAS</div><h1>Motion lives<br>inside attention.</h1><p class="lead">50-case 指标排名，以及 5 个代表 case 上 Top stable 6 与 Bottom PCK@32 6 的 all-token Q@K 精确 softmax 热力图。</p><section class="grid"><a class="card new" href="/all-token-qk?v=3"><div><span>01 / RANK EXTREMES</span><h2>Top vs Bottom Q@K</h2><p>三模型并排；12 个组合、40 个时间步，包含 Raw QK、Softmax attention 和 7×7 temporal matrix。</p></div><span class="go">OPEN ALL-TOKEN MATRICES</span></a><a class="card" href="/all-steps/overlays?v=2"><div><span>02 / TRAJECTORIES</span><h2>All-step overlays</h2><p>按三模型 combined global ranking 浏览 GT 与 Q@K 轨迹。</p></div><span class="go">OPEN OVERLAYS</span></a><a class="card" href="/all-steps/rankings?v=3"><div><span>03 / METRICS</span><h2>Global rankings</h2><p>完整 Step × Block × Head 指标、profile 与跨模型综合排名。</p></div><span class="go">OPEN RANKINGS</span></a></section></main></body></html>'''


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stable-head all-token Q@K</title><style>
:root{--paper:#ece7da;--ink:#18221e;--panel:#101714;--cream:#fffdf7;--rust:#c65738;--teal:#287a67;--line:#aaa392}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#d28b6330,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1800px,calc(100% - 24px));margin:auto;padding:24px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}.top{display:flex;justify-content:space-between;gap:25px;align-items:end}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.14em;font-size:11px}h1{font-size:clamp(38px,5vw,72px);line-height:.92;margin:7px 0}.note{max-width:800px;line-height:1.55;color:#59635e}.back{color:var(--ink);font-weight:900}.controls{display:grid;grid-template-columns:2fr 1fr 1fr 1.3fr;gap:10px;margin:20px 0;padding:14px;background:var(--cream);border:1px solid var(--line)}label{font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}select,input{display:block;width:100%;margin-top:5px;padding:9px;background:white;border:1px solid #5f665f;font-weight:800}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.card{background:var(--panel);color:white;padding:11px;border-radius:3px 22px 3px 3px;min-width:0}.card h2{font-size:20px;margin:2px 0 8px}.matrix{display:block;width:100%;aspect-ratio:1;background:#050806;object-fit:contain}.meta{font-size:11px;color:#b6c2bc;margin-top:8px;line-height:1.5}.pending{display:grid;place-items:center;aspect-ratio:1;background:#202924;color:#d8b39e;font-weight:900}.status{font-size:12px;font-weight:900;color:var(--teal)}@media(max-width:1050px){.grid{grid-template-columns:1fr}.controls{grid-template-columns:1fr 1fr}}@media(max-width:600px){.controls{grid-template-columns:1fr}.top{display:block}}
</style></head><body><main><div class="top"><div><div class="eyebrow">3136 ALL TOKENS · EXACT SOFTMAX · 512 POOLED BINS</div><h1>Stable-head<br>Q@K matrices</h1></div><a class="back" href="/">Back to atlas</a></div><p class="note">每个原始 query token 对全部 3136 个 key token 计算 QK/√d 与精确 softmax，之后才汇总到 512×512。白线表示 7 个 latent time 的边界。</p><section class="controls"><label>Case<select id="case"></select></label><label>Stable combination<select id="combo"></select></label><label>Denoising step<input id="step" type="range" min="0" max="39" value="39"><span id="stepText">S039</span></label><label>Matrix<select id="kind"><option value="attention">log10 Softmax attention</option><option value="raw">Raw QK / sqrt(d)</option><option value="temporal">7×7 temporal attention</option></select></label></section><p class="status" id="status"></p><section class="grid" id="grid"></section></main><script>
const MODELS=[['gt','GT teacher-forced'],['lora','LoRA step-000500'],['baseline','Wan2.2 Baseline']],q=id=>document.getElementById(id);let DATA;function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}async function init(){DATA=await fetch('/api/all-token/catalog').then(r=>r.json());q('case').innerHTML=DATA.cases.map(c=>`<option>${esc(c)}</option>`).join('');q('combo').innerHTML=DATA.combinations.map(x=>`<option value="${x.block},${x.head}">L${String(x.block).padStart(2,'0')} / H${String(x.head).padStart(2,'0')}</option>`).join('');q('combo').value='20,9';for(const id of ['case','combo','kind'])q(id).addEventListener('change',render);q('step').addEventListener('input',()=>{q('stepText').textContent='S'+String(q('step').value).padStart(3,'0');render()});render()}function render(){const c=q('case').value,[b,h]=q('combo').value.split(','),s=q('step').value,k=q('kind').value;const done=MODELS.filter(([m])=>DATA.available[m]?.[c]?.includes(Number(b))).length;q('status').textContent=`${done}/3 model outputs available for ${c} · L${b}/H${h} · S${String(s).padStart(3,'0')}`;q('grid').innerHTML=MODELS.map(([m,label])=>{const available=DATA.available[m]?.[c]?.includes(Number(b));if(!available)return`<article class="card"><h2>${label}</h2><div class="pending">PENDING COMPUTE</div><div class="meta">该 worker 完成此 case 后会自动出现。</div></article>`;const src=`/api/all-token/matrix?model=${m}&case=${encodeURIComponent(c)}&block=${b}&head=${h}&step=${s}&kind=${k}&v=1`;return`<article class="card"><h2>${label}</h2><img class="matrix" src="${src}" alt="${label} QK"><div class="meta">L${b} / H${h} · S${String(s).padStart(3,'0')} · ${k==='attention'?'exact softmax then pooled':k==='raw'?'pooled raw QK mean':'exact attention mass by latent time'}</div></article>`}).join('')}init();
</script></body></html>'''


PAGE_ALL = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Top and bottom Q@K heatmaps</title><style>
:root{--paper:#ece7da;--ink:#18221e;--panel:#101714;--cream:#fffdf7;--rust:#c65738;--teal:#287a67;--line:#aaa392}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#d28b6330,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1900px,calc(100% - 24px));margin:auto;padding:24px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}.top{display:flex;justify-content:space-between;gap:25px;align-items:end}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.14em;font-size:11px}h1{font-size:clamp(38px,5vw,72px);line-height:.92;margin:7px 0}.note{max-width:1000px;line-height:1.55;color:#59635e}.back{color:var(--ink);font-weight:900}.controls{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin:20px 0;padding:14px;background:var(--cream);border:1px solid var(--line)}label{font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:9px;background:white;border:1px solid #5f665f;font-weight:800}.sheet{margin:14px 0 24px;padding:12px;background:var(--panel);color:white;border-radius:3px 24px 3px 3px}.sheet h2{display:flex;justify-content:space-between;align-items:baseline;margin:2px 0 10px;font-size:23px}.sheet h2 span{font:700 11px "Trebuchet MS",sans-serif;color:#aab8b1;letter-spacing:.07em}.sheet img{display:block;width:100%;min-height:260px;background:#050806;object-fit:contain}.status{font-size:12px;font-weight:900;color:var(--teal)}@media(max-width:650px){.controls{grid-template-columns:1fr}.top{display:block}.sheet{padding:7px}.sheet h2{font-size:18px}}
</style></head><body><main><div class="top"><div><div class="eyebrow">5 CASES · TOP 6 + BOTTOM 6 · ALL 40 STEPS · THREE MODELS</div><h1>PCK@32 extremes<br>Q@K contact sheets</h1></div><a class="back" href="/">Back to atlas</a></div><p class="note">Top stable 6 来自跨模型多时间步稳定性分析；Bottom 6 按三模型等权、S010-S039 Object Macro PCK@32 均值升序选取。每个组合按三个模型分区，并以 5×8 网格完整展示 S000-S039。</p><section class="controls"><label>Case<select id="case"></select></label><label>Matrix type<select id="kind"><option value="attention">log10 Softmax attention</option><option value="raw">Raw QK / sqrt(d)</option><option value="temporal">7×7 temporal attention</option></select></label></section><p class="status" id="status"></p><section id="sheets"></section></main><script>
const q=id=>document.getElementById(id);let DATA;function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}async function init(){DATA=await fetch('/api/all-token/catalog').then(r=>r.json());q('case').innerHTML=DATA.cases.map(c=>`<option>${esc(c)}</option>`).join('');q('case').addEventListener('change',render);q('kind').addEventListener('change',render);render()}function render(){const c=q('case').value,k=q('kind').value,ready=Object.keys(DATA.available).reduce((n,m)=>n+(DATA.available[m]?.[c]?1:0),0);q('status').textContent=`${c} · outputs are populated incrementally; refresh as workers finish.`;q('sheets').innerHTML=DATA.combinations.map(x=>{const b=String(x.block).padStart(2,'0'),h=String(x.head).padStart(2,'0'),score=x.pck32==null?'multi-step stable':`mean PCK@32 ${Number(x.pck32).toFixed(3)}`,src=`/api/all-token/contact-sheet?case=${encodeURIComponent(c)}&block=${x.block}&head=${x.head}&kind=${k}&v=3`;return`<article class="sheet"><h2>${x.group} · L${b} / H${h}<span>${score} · S000-S039 · GT / LoRA / BASELINE</span></h2><img loading="lazy" src="${src}" alt="L${b} H${h} all-step QK contact sheet"></article>`}).join('')}init();
</script></body></html>'''


PAGE_S039 = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>S039 top and bottom Q@K</title><style>
:root{--paper:#ece7da;--ink:#18221e;--panel:#101714;--cream:#fffdf7;--rust:#c65738;--teal:#287a67;--line:#aaa392}*{box-sizing:border-box}body{margin:0;background:linear-gradient(135deg,#d28b6330,transparent 34rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(2200px,calc(100% - 24px));margin:auto;padding:24px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif}.top{display:flex;justify-content:space-between;gap:25px;align-items:end}.eyebrow{color:var(--rust);font-weight:900;letter-spacing:.14em;font-size:11px}h1{font-size:clamp(38px,5vw,72px);line-height:.92;margin:7px 0}.note{max-width:1000px;line-height:1.55;color:#59635e}.back{color:var(--ink);font-weight:900}.controls{display:grid;grid-template-columns:2fr 1fr;gap:10px;margin:20px 0;padding:14px;background:var(--cream);border:1px solid var(--line)}label{font-size:10px;font-weight:900;letter-spacing:.1em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:9px;background:white;border:1px solid #5f665f;font-weight:800}.section-title{display:flex;align-items:baseline;justify-content:space-between;margin:28px 0 9px;border-bottom:2px solid var(--ink)}.section-title h2{font-size:30px;margin:0 0 5px}.section-title span{font-size:11px;font-weight:900;color:#68726d}.rank-row{display:grid;grid-template-columns:repeat(6,minmax(235px,1fr));gap:8px;overflow-x:auto;padding-bottom:8px}.card{background:var(--panel);color:white;padding:8px;border-radius:3px 18px 3px 3px;min-width:235px}.card h3{margin:2px 2px 7px;font:700 18px Georgia,serif}.card h3 span{display:block;margin-top:3px;color:#a9b8b0;font:700 10px "Trebuchet MS",sans-serif}.card img{display:block;width:100%;aspect-ratio:3/1;background:#050806;object-fit:contain}.status{font-size:12px;font-weight:900;color:var(--teal)}@media(max-width:650px){.controls{grid-template-columns:1fr}.top{display:block}.rank-row{grid-template-columns:repeat(6,270px)}}
</style></head><body><main><div class="top"><div><div class="eyebrow">5 CASES · S039 ONLY · THREE MODELS PER CARD</div><h1>PCK@32 extremes<br>at the final step</h1></div><a class="back" href="/">Back to atlas</a></div><p class="note">每张卡固定展示 S039，内部从左到右依次为 GT teacher-forced、LoRA、Wan2.2 Baseline。Top stable 6 放在第一行，Bottom PCK@32 6 放在第二行。</p><section class="controls"><label>Case<select id="case"></select></label><label>Matrix type<select id="kind"><option value="attention">log10 Softmax attention</option><option value="raw">Raw QK / sqrt(d)</option><option value="temporal">7×7 temporal attention</option></select></label></section><p class="status" id="status"></p><section><div class="section-title"><h2>Top stable 6</h2><span>CROSS-MODEL · MULTI-STEP STABILITY</span></div><div class="rank-row" id="topRow"></div></section><section><div class="section-title"><h2>Bottom PCK@32 6</h2><span>THREE-MODEL MEAN · S010-S039 OBJECT PCK@32</span></div><div class="rank-row" id="bottomRow"></div></section></main><script>
const q=id=>document.getElementById(id);let DATA;function esc(x){return String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}async function init(){DATA=await fetch('/api/all-token/catalog').then(r=>r.json());q('case').innerHTML=DATA.cases.map(c=>`<option>${esc(c)}</option>`).join('');q('case').addEventListener('change',render);q('kind').addEventListener('change',render);render()}function cards(group){const c=q('case').value,k=q('kind').value;return DATA.combinations.filter(x=>x.group===group).map(x=>{const b=String(x.block).padStart(2,'0'),h=String(x.head).padStart(2,'0'),score=x.pck32==null?'stable across models/steps':`mean PCK@32 ${Number(x.pck32).toFixed(3)}`,src=`/api/all-token/s039-strip?case=${encodeURIComponent(c)}&block=${x.block}&head=${x.head}&kind=${k}&v=1`;return`<article class="card"><h3>L${b} / H${h}<span>${score}</span></h3><img loading="lazy" src="${src}" alt="L${b} H${h} S039 three-model QK"></article>`}).join('')}function render(){q('status').textContent=`${q('case').value} · S039 · incomplete models are marked PENDING.`;q('topRow').innerHTML=cards('Top stable');q('bottomRow').innerHTML=cards('Bottom PCK@32')}init();
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
                group = "bottom" if (block, head) in BOTTOM_COMBINATIONS else "top"
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
                "pck32": BOTTOM_PCK32.get((block, head)),
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
    group = "bottom" if (block, head) in BOTTOM_COMBINATIONS else "top"
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
        group = "bottom" if (block, head) in BOTTOM_COMBINATIONS else "top"
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
    group = "bottom" if (block, head) in BOTTOM_COMBINATIONS else "top"
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


class Handler(BASE_HANDLER):
    def send_payload(self, payload: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                return self.send_payload(PORTAL.encode(), "text/html; charset=utf-8")
            if parsed.path == "/all-token-qk":
                return self.send_payload(PAGE_S039.encode(), "text/html; charset=utf-8")
            if parsed.path == "/api/all-token/catalog":
                payload = json.dumps(catalog(), ensure_ascii=False).encode()
                return self.send_payload(payload, "application/json; charset=utf-8")
            if parsed.path == "/api/all-token/matrix":
                return self.send_payload(matrix_png(parse_qs(parsed.query)), "image/png")
            if parsed.path == "/api/all-token/contact-sheet":
                return self.send_payload(contact_sheet_png(parse_qs(parsed.query)), "image/png")
            if parsed.path == "/api/all-token/s039-strip":
                return self.send_payload(s039_strip_png(parse_qs(parsed.query)), "image/png")
            return super().do_GET()
        except (FileNotFoundError, ValueError) as error:
            return self.send_payload(str(error).encode(), "text/plain; charset=utf-8", 404)
        except Exception as error:
            return self.send_payload(repr(error).encode(), "text/plain; charset=utf-8", 500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    print(f"serving DiffTrack atlas at http://{args.host}:{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
