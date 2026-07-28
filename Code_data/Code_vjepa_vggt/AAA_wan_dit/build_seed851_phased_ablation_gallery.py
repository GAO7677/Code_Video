#!/usr/bin/env python3
"""Build the seed-851 grouped phased-ablation comparison page."""

from __future__ import annotations

from pathlib import Path


GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed"
)
OUTPUT = GALLERY_ROOT / "seed851" / "index.html"


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seed 851 · 分阶段 Head 消融</title>
<style>
:root{color-scheme:dark;--bg:#121416;--panel:#191c1f;--line:#343a40;--muted:#a4abb3;--accent:#55b8a6}
*{box-sizing:border-box}body{margin:0;padding-bottom:58px;background:var(--bg);color:#f3f5f7;font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;padding:12px 18px;background:#121416f2;border-bottom:1px solid var(--line)}
h1,h2,p{margin:0}h1{font-size:21px}h2{font-size:17px}.sub{margin-top:4px;color:var(--muted)}
.status{color:var(--accent);font-weight:650}.toolbar{display:flex;gap:12px;align-items:center;margin-top:8px}
button{border:1px solid var(--line);background:#24282c;color:#fff;padding:5px 9px;cursor:pointer}
.playbar{position:fixed;z-index:8;left:0;right:0;bottom:0;display:grid;grid-template-columns:auto auto auto auto minmax(180px,1fr) auto;gap:8px;align-items:center;padding:9px 18px;background:#171a1df2;border-top:1px solid var(--line);backdrop-filter:blur(8px)}
.playbar input{width:100%;accent-color:var(--accent)}.playbar-time{min-width:92px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
main{padding:14px 18px}.refs{display:grid;grid-template-columns:repeat(2,minmax(280px,448px));gap:10px;margin-bottom:20px}
.group{margin:0 0 22px}.group-title{display:flex;gap:10px;align-items:baseline;margin-bottom:7px}
.group-note{color:var(--muted)}
table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{border:1px solid var(--line);padding:6px;vertical-align:top}
thead th{background:#202429}th:first-child{width:220px;text-align:left}.variant{background:#181b1e}
video{display:block;width:100%;aspect-ratio:7/4;background:#000}.missing{display:grid;place-items:center;aspect-ratio:7/4;background:#24282c;color:#8f969e}
figure{margin:0}figcaption{padding-top:4px;color:var(--muted)}.target-count{display:block;color:var(--muted);font-weight:400}
@media(max-width:900px){.refs{grid-template-columns:1fr}table{min-width:900px}.group{overflow-x:auto}}
</style></head><body>
<header><h1>Seed 851 · 分阶段 Head 消融</h1>
<p class="sub">每个 Head 类别单独成组；组内每行一个模型，每列为全程或一个去噪阶段。区间均为半开区间。</p>
<div class="toolbar"><button id="refresh" type="button">立即刷新</button><span class="status" id="status">读取中</span><span class="sub" id="updated"></span></div>
</header><main><p class="sub" id="prompt"></p><div class="refs" id="refs"></div><div id="groups"></div></main>
<div class="playbar"><button id="play-all" type="button">全部播放</button><button id="replay-all" type="button">从头播放</button><button id="pause-all" type="button">全部暂停</button><button id="reset-all" type="button">回到开头</button><input id="timeline" type="range" min="0" max="1000" value="0" aria-label="统一播放进度"><span class="playbar-time" id="play-time">00:00 / 00:00</span></div>
<script>
const SEED="851";
const GROUPS=[
 {title:"S-all",note:"全部公共稳定空间局部 Head",variants:["baseline","S","S_steps00_10","S_steps00_15","S_steps05_10","S_steps05_15","S_steps10_20","S_steps20_30","S_steps30_40"]},
 {title:"T-all",note:"全部公共稳定运动轨迹 Head",variants:["baseline","T","T_steps00_10","T_steps00_15","T_steps05_10","T_steps05_15","T_steps10_20","T_steps20_30","T_steps30_40"]},
 {title:"P-all",note:"全部公共稳定固定位置时序 Head",variants:["baseline","P","P_steps00_10","P_steps10_20","P_steps20_30","P_steps30_40"]},
 {title:"C-all",note:"全部公共稳定上下文/历史 Head",variants:["baseline","C","C_steps00_10","C_steps00_15","C_steps05_10","C_steps05_15","C_steps10_20","C_steps20_30","C_steps30_40"]},
 {title:"G-all",note:"全部公共稳定全局聚合 Head",variants:["baseline","G","G_steps00_10","G_steps10_20","G_steps20_30","G_steps30_40"]},
 {title:"S score 前10",note:"score_S 排名前 10 个公共 Head",variants:["baseline","S_top10","S_top10_steps00_10","S_top10_steps10_20","S_top10_steps20_30","S_top10_steps30_40"]},
 {title:"S score 后10",note:"score_S 排名后 10 个公共 Head",variants:["baseline","S_bottom10","S_bottom10_steps00_10","S_bottom10_steps10_20","S_bottom10_steps20_30","S_bottom10_steps30_40"]}
];
let DATA=null;
let seeking=false;
const q=id=>document.getElementById(id);
const media=src=>src?`../${src}`:null;
function videoCard(src,label){return src?`<figure><video muted playsinline preload="metadata" src="${media(src)}"></video><figcaption>${label}</figcaption></figure>`:`<div class="missing">等待生成</div>`}
function refPath(item,key){if(!item[key])return null;if(key==="source_video")return `media/references/${item.id}__source_video_49f.mp4`;const ext=item[key].split(".").pop();return `media/references/${item.id}__${key}.${ext}`}
function variantLabel(v){return DATA.role_names[v]||v}
function render(){
 const item=DATA.cases[0],videos=DATA.videos[item.id][SEED];
 q("prompt").textContent=`Prompt: ${item.prompt}`;
 q("refs").innerHTML=videoCard(refPath(item,"source_video"),"Source / GT")+videoCard(refPath(item,"context_video"),"8-frame context");
 q("groups").innerHTML=GROUPS.map(group=>{
  const head=group.variants.map(v=>{
   const count=v==="baseline"?"":`<span class="target-count">${DATA.target_counts[v]} Heads</span>`;
   return `<th>${variantLabel(v)}${count}</th>`;
  }).join("");
  const rows=DATA.models.map(m=>{
   const cells=group.variants.map(v=>`<td class="video-cell" data-variant="${v}" data-model="${m}">${videoCard(videos[m][v],variantLabel(v))}</td>`).join("");
   return `<tr><th class="variant">${DATA.model_names[m]}</th>${cells}</tr>`;
  }).join("");
  return `<section class="group"><div class="group-title"><h2>${group.title}</h2><span class="group-note">${group.note}</span></div><table><thead><tr><th>模型</th>${head}</tr></thead><tbody>${rows}</tbody></table></section>`;
 }).join("");
 updateStatus();
}
function refreshCells(){
 const item=DATA.cases[0],videos=DATA.videos[item.id][SEED];
 document.querySelectorAll(".video-cell").forEach(cell=>{
  if(cell.querySelector("video"))return;
  const src=videos[cell.dataset.model][cell.dataset.variant];
  if(src)cell.innerHTML=videoCard(src,DATA.model_names[cell.dataset.model]);
 });
 updateStatus();
}
function videos(){return [...document.querySelectorAll("video")].filter(v=>Number.isFinite(v.duration)&&v.duration>0)}
function formatTime(seconds){
 const value=Math.max(0,Math.floor(Number.isFinite(seconds)?seconds:0));
 return `${String(Math.floor(value/60)).padStart(2,"0")}:${String(value%60).padStart(2,"0")}`;
}
function syncTimeLabel(){
 const list=videos(),master=list[0];
 if(!master){q("play-time").textContent="00:00 / 00:00";return}
 if(!seeking)q("timeline").value=String(Math.round(1000*master.currentTime/master.duration));
 q("play-time").textContent=`${formatTime(master.currentTime)} / ${formatTime(master.duration)}`;
}
function seekAll(fraction){videos().forEach(v=>{v.currentTime=Math.min(v.duration,Math.max(0,v.duration*fraction))})}
function playAll(){
 const list=videos();
 if(!list.length)return;
 const fraction=Number(q("timeline").value)/1000;
 seekAll(fraction);
 list.forEach(v=>{v.loop=false;v.play().catch(()=>{})});
}
function pauseAll(){videos().forEach(v=>v.pause())}
function resetAll(){pauseAll();q("timeline").value="0";seekAll(0);syncTimeLabel()}
function replayAll(){q("timeline").value="0";seekAll(0);playAll()}
function updateStatus(){
 const item=DATA.cases[0],videos=DATA.videos[item.id][SEED];
 const variants=GROUPS.flatMap(g=>g.variants);
 const done=variants.reduce((n,v)=>n+DATA.models.filter(m=>videos[m][v]).length,0);
 q("status").textContent=`Seed 851 视频 ${done}/${variants.length*DATA.models.length}`;
 q("updated").textContent=`更新于 ${new Date().toLocaleTimeString()}`;
}
async function load(){
 try{
  const response=await fetch(`../manifest.json?t=${Date.now()}`,{cache:"no-store"});
  if(!response.ok)throw new Error(`HTTP ${response.status}`);
  const next=await response.json();
  if(!next.seeds.map(String).includes(SEED))throw new Error("manifest 缺少 seed 851");
  DATA=next;
  if(!q("groups").children.length)render();else refreshCells();
 }catch(error){q("status").textContent=`刷新失败: ${error.message}`}
}
q("refresh").addEventListener("click",load);
q("play-all").addEventListener("click",playAll);
q("replay-all").addEventListener("click",replayAll);
q("pause-all").addEventListener("click",pauseAll);
q("reset-all").addEventListener("click",resetAll);
q("timeline").addEventListener("pointerdown",()=>{seeking=true});
q("timeline").addEventListener("input",event=>{seekAll(Number(event.target.value)/1000);syncTimeLabel()});
q("timeline").addEventListener("change",()=>{seeking=false;syncTimeLabel()});
setInterval(syncTimeLabel,250);
load();setInterval(load,10000);
</script></body></html>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HTML, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
