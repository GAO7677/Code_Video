#!/usr/bin/env python3
"""Build a focused S-head gallery without dose-control replicates."""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path


DEFAULT_MANIFEST = (
    "/head-role-dose-control-pilot/manifest.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/s-head-ablation"
)
SOURCE_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/head-role-dose-control-pilot/manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def index_page(first_case: str) -> str:
    escaped = html.escape(first_case)
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0;url=cases/{escaped}/">
<title>S Head Ablation</title></head>
<body><a href="cases/{escaped}/">进入 S Head 消融页</a></body></html>"""


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__CASE_ID__ · S Head Ablation</title>
<style>
:root{--bg:#0f1316;--panel:#191f23;--line:#374047;--text:#edf1f3;--muted:#a8b2b8;--teal:#62c9b3;--amber:#e0ad63;--blue:#7aaee5;--pending:#d2b46c}
*{box-sizing:border-box}body{margin:0;padding-bottom:52px;background:var(--bg);color:var(--text);font:13px/1.45 Arial,"Noto Sans SC",sans-serif;letter-spacing:0}
header{position:sticky;top:0;z-index:5;padding:11px 16px;background:#0f1316f2;border-bottom:1px solid var(--line)}
.top,.controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap}.top{justify-content:space-between}.controls{margin-top:8px}
h1,h2,h3,p{margin:0}h1{font-size:18px;overflow-wrap:anywhere}h2{font-size:23px}.links{display:flex;gap:12px}.links a{color:var(--teal);font-weight:700;text-decoration:none}
label{display:grid;gap:2px;color:var(--muted);font-size:10px}select{max-width:560px;padding:6px 8px;border:1px solid var(--line);background:#242c31;color:var(--text)}#case{min-width:min(560px,82vw)}
button{padding:5px 9px;border:1px solid var(--line);background:#273036;color:var(--text);cursor:pointer}.status{color:var(--teal);font-weight:700}.prompt{margin-top:7px;color:var(--muted)}
main{padding:14px 16px}.references{display:grid;grid-template-columns:repeat(2,minmax(260px,448px));gap:8px;margin-top:8px}
.model{margin-top:32px;border-top:4px solid var(--teal)}.model-title{padding:10px 0;border-bottom:1px solid var(--line)}.group{margin-top:20px}.group-title{display:flex;align-items:baseline;gap:12px;padding:6px 0}.group-title h3{font-size:16px}.group-title p{color:var(--muted)}
.stage{margin-top:11px}.stage-head{display:flex;align-items:center;gap:8px;padding:5px 8px;background:#20272b;border-left:3px solid var(--teal)}.stage-head h3{font-size:14px}.actions{display:flex;gap:5px;margin-left:auto}.actions button{padding:3px 8px;font-size:11px}
.videos{display:grid;grid-template-columns:repeat(4,minmax(260px,448px));gap:7px;overflow-x:auto}.cell{border:1px solid var(--line);background:var(--panel)}.cell h3{padding:6px 8px;background:#252d32;font-size:13px}
.cell.feature{border-color:#4c8f83}.cell.feature h3{background:#23433d;color:#a0e3d5}.cell.union{border-color:#ad7742}.cell.union h3{background:#46301f;color:#ffd09a}.cell.depth{border-color:#537da5}.cell.depth h3{background:#24394d;color:#b9d9f7}
video{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#050607}.meta{padding:5px 8px;color:var(--muted);font-size:10px;min-height:31px}.missing{display:grid;place-items:center;aspect-ratio:7/4;color:var(--pending);background:#1c2226}
.playbar{position:fixed;left:0;right:0;bottom:0;z-index:8;padding:9px 16px;background:#171c20f2;border-top:1px solid var(--line);color:var(--muted)}
@media(max-width:700px){.references{grid-template-columns:1fr}}
</style></head><body>
<header><div class="top"><h1 id="title">__CASE_ID__</h1><span id="status" class="status">读取中</span><div class="links"><a href="/head-role-dose-control-pilot/cases/">完整页</a><a href="/multiseed/motion-n-analysis/">进度页</a></div></div>
<div class="controls"><label>Case<select id="case"></select></label><label>Seed<select id="seed"></select></label><button id="reload">刷新结果</button></div>
<p class="prompt" id="prompt"></p></header>
<main><section><h2>输入与参考</h2><div class="references" id="references"></div></section><div id="content"></div></main>
<div class="playbar">本页仅展示 S-feature 与 S-depth；不包含任何 dose-control replicate。</div>
<script>
const CASE_ID="__CASE_ID__",MANIFEST="__MANIFEST__";let D,R,C,syncTimer=null,activeRow=null;
const q=id=>document.getElementById(id);
function options(id,values,label=x=>x){q(id).innerHTML=values.map(x=>`<option value="${x}">${label(x)}</option>`).join("")}
function selection(){return{seed:Number(q("seed").value)}}
function media(record,label,kind=""){const meta=record?record.kind==="reference"?"原始输入":record.kind==="baseline"?"同模型、同 seed 未消融":`${record.subset_id} · k=${record.k} · steps ${record.start}-${record.end}`:"Pending";return`<article class="cell ${kind}"><h3>${label}</h3>${record?`<video muted playsinline preload="none" src="${record.video}"></video>`:`<div class="missing">该配置仍在生成或当前 seed 无结果</div>`}<div class="meta">${meta}</div></article>`}
function rowHead(label){return`<div class="stage-head"><h3>${label}</h3><div class="actions"><button data-action="play">播放本行</button><button data-action="replay">从头播放</button><button data-action="pause">暂停</button></div></div>`}
function baseline(model){const s=selection();return R.find(r=>r.kind==="baseline"&&r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed)}
function featureStages(){return(D.s_feature_step_ranges||[]).map(x=>x.map(Number))}
function feature(model,subtype,stage){const s=selection();return R.find(r=>r.kind==="s_feature_split"&&r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed&&r.feature_subtype===subtype&&r.start===stage[0]&&r.end===stage[1])}
function depth(model,id,stage){const s=selection();return R.find(r=>r.kind==="s_depth"&&r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed&&r.subset_id===id&&r.start===stage[0]&&r.end===stage[1])}
function depthSubsets(){const order={early:0,middle:1,late:2};return Object.entries(D.s_depth_subsets||{}).map(([id,x])=>({id,...x})).sort((a,b)=>(order[a.depth_stratum]??99)-(order[b.depth_stratum]??99))}
function depthStages(){return(D.s_depth_step_ranges||[]).map(x=>x.map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1])}
function depthLabel(x){const names={early:"Early",middle:"Middle",late:"Late"};return`${names[x.depth_stratum]} · B${String(x.block_start_inclusive).padStart(2,"0")}–B${String(x.block_end_exclusive-1).padStart(2,"0")} · ${x.k} heads`}
function renderReferences(){q("references").innerHTML=media(C.context_url?{video:C.context_url,kind:"reference"}:null,"8 帧 Context")+media(C.source_url?{video:C.source_url,kind:"reference"}:null,"Source / GT");q("prompt").textContent=C.caption}
function stopSync(){if(syncTimer!==null)clearInterval(syncTimer);syncTimer=null;activeRow=null}
function render(){stopSync();const subsets=depthSubsets(),stages=depthStages();let out="";for(const model of D.models){const base=baseline(model);out+=`<section class="model"><div class="model-title"><h2>${D.model_labels[model]} · seed ${selection().seed}</h2></div>`;out+=`<div class="group"><div class="group-title"><h3>S 分类消融</h3><p>同一对照下比较 32 / 32 / 64 heads</p></div>`;for(const stage of featureStages()){out+=`<section class="stage">${rowHead(`去噪阶段 ${stage[0]}–${stage[1]}`)}<div class="videos">`;out+=media(base,"Baseline");out+=media(feature(model,"local_enrichment",stage),"Local-enrichment S (32)","feature");out+=media(feature(model,"same_frame_mass",stage),"Same-frame-mass S (32)","feature");out+=media(feature(model,"local_same_union",stage),"Local + Same 联合 (64)","union");out+="</div></section>"}out+="</div>";out+=`<div class="group"><div class="group-title"><h3>S 深度消融</h3><p>Early / Middle / Late 全部 S heads</p></div>`;for(const stage of stages){out+=`<section class="stage">${rowHead(`去噪阶段 ${stage[0]}–${stage[1]}`)}<div class="videos">`;out+=media(base,"Baseline");for(const item of subsets)out+=media(depth(model,item.id,stage),depthLabel(item),"depth");out+="</div></section>"}out+="</div></section>"}q("content").innerHTML=out}
function rowVideos(button){const row=button.closest(".stage");return row?[...row.querySelectorAll("video")]:[]}
function playRow(button,replay){stopSync();const videos=rowVideos(button);if(!videos.length)return;if(replay)for(const v of videos)v.currentTime=0;for(const v of videos)v.play().catch(()=>{});activeRow=button.closest(".stage");syncTimer=setInterval(()=>{const vs=activeRow?[...activeRow.querySelectorAll("video")]:[];if(vs.length<2||vs[0].readyState<2)return;for(const v of vs.slice(1))if(v.readyState>=2&&Math.abs(v.currentTime-vs[0].currentTime)>0.12)v.currentTime=vs[0].currentTime},250)}
function pauseRow(button){const row=button.closest(".stage");if(row===activeRow)stopSync();for(const v of rowVideos(button))v.pause()}
document.addEventListener("click",event=>{const button=event.target.closest("[data-action]");if(!button)return;if(button.dataset.action==="play")playRow(button,false);else if(button.dataset.action==="replay")playRow(button,true);else pauseRow(button)});
async function progress(){const get=path=>fetch(`${path}?t=${Date.now()}`).then(x=>x.ok?x.json():null).catch(()=>null),[depth,feature,union,phased]=await Promise.all([get("/head-role-dose-control-pilot/s-depth-progress.json"),get("/head-role-dose-control-pilot/s-feature-progress.json"),get("/head-role-dose-control-pilot/s-feature-union-progress.json"),get("/head-role-dose-control-pilot/s-feature-phased-progress.json")]);const text=(x,fallback)=>x?`${(x.state_counts||{}).complete||0}/${x.expected_tasks}`:fallback;q("status").textContent=`S-feature ${text(feature,"6/6")} · union ${text(union,"0/3")} · phased ${text(phased,"0/18")} · S-depth ${text(depth,"35/36")}`}
fetch(`${MANIFEST}?t=${Date.now()}`).then(x=>x.json()).then(data=>{D=data;R=data.records;C=data.cases.find(x=>x.id===CASE_ID);options("case",D.cases.map(x=>x.id));q("case").value=CASE_ID;q("case").onchange=()=>location.href=`../${encodeURIComponent(q("case").value)}/`;const seeds=[...new Set(R.filter(r=>["baseline","s_feature_split","s_depth"].includes(r.kind)).map(r=>r.seed))].sort((a,b)=>a-b);options("seed",seeds);q("seed").value=seeds.includes(851)?"851":String(seeds[0]);q("seed").onchange=render;q("reload").onclick=()=>location.reload();renderReferences();render();progress();setInterval(progress,30000)}).catch(error=>q("status").textContent=`加载失败：${error}`);
</script></body></html>"""


def main() -> None:
    args = parse_args()
    source = args.source_manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    import json

    payload = json.loads(source.read_text(encoding="utf-8"))
    cases = [str(case["id"]) for case in payload["cases"]]
    if not cases:
        raise RuntimeError("The source manifest has no cases")
    page = PAGE.replace("__MANIFEST__", DEFAULT_MANIFEST)
    for case_id in cases:
        atomic_write(
            output / "cases" / case_id / "index.html",
            page.replace("__CASE_ID__", html.escape(case_id)),
        )
    atomic_write(output / "index.html", index_page(cases[0]))
    print(
        f"[s-head-gallery] cases={len(cases)} output={output} "
        "included=baseline,s_feature_split,s_depth excluded=ablation"
    )


if __name__ == "__main__":
    main()
