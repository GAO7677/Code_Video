#!/usr/bin/env python3
"""Build the S/T/C phased all-Head comparison page for six seeds."""

from __future__ import annotations

from pathlib import Path


GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/multiseed"
)
OUTPUT = GALLERY_ROOT / "stc-phased" / "index.html"


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S/T/C All-Head 分阶段消融</title>
<style>
:root{color-scheme:dark;--bg:#121416;--line:#343a40;--muted:#a4abb3;--accent:#55b8a6}
*{box-sizing:border-box}body{margin:0;padding-bottom:58px;background:var(--bg);color:#f3f5f7;font:14px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:3;padding:12px 18px;background:#121416f2;border-bottom:1px solid var(--line)}
h1,h2,p{margin:0}h1{font-size:21px}h2{font-size:17px}.sub{margin-top:4px;color:var(--muted)}
.toolbar{display:flex;gap:10px;align-items:center;margin-top:8px}.status{color:var(--accent);font-weight:650}
button,select{border:1px solid var(--line);background:#24282c;color:#fff;padding:5px 9px}
button{cursor:pointer}.playbar{position:fixed;z-index:8;left:0;right:0;bottom:0;display:grid;grid-template-columns:auto auto auto auto minmax(180px,1fr) auto;gap:8px;align-items:center;padding:9px 18px;background:#171a1df2;border-top:1px solid var(--line);backdrop-filter:blur(8px)}
.playbar input{width:100%;accent-color:var(--accent)}.playbar-time{min-width:92px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
main{padding:14px 18px}.refs{display:grid;grid-template-columns:repeat(2,minmax(280px,448px));gap:10px;margin-bottom:20px}
.group{margin:0 0 22px;overflow-x:auto}.group-title{display:flex;gap:10px;align-items:baseline;margin-bottom:7px}.group-note{color:var(--muted)}
table{width:100%;min-width:1180px;border-collapse:collapse;table-layout:fixed}th,td{border:1px solid var(--line);padding:6px;vertical-align:top}
thead th{background:#202429}th:first-child{width:140px;text-align:left}.model{background:#181b1e}
video{display:block;width:100%;aspect-ratio:7/4;background:#000}.missing{display:grid;place-items:center;aspect-ratio:7/4;background:#24282c;color:#8f969e}
figure{margin:0}figcaption{padding-top:4px;color:var(--muted)}.target-count{display:block;color:var(--muted);font-weight:400}
@media(max-width:900px){.refs{grid-template-columns:1fr}}
</style></head><body>
<header><h1>S/T/C All-Head 分阶段消融</h1>
<p class="sub">每个类别单独成组；每行一个模型，每列为 Baseline 或一个去噪区间。所有区间均为半开区间。</p>
<div class="toolbar"><label>Seed <select id="seed"></select></label><button id="refresh" type="button">立即刷新</button><span class="status" id="status">读取中</span><span class="sub" id="updated"></span></div>
</header><main><p class="sub" id="prompt"></p><div class="refs" id="refs"></div><div id="groups"></div></main>
<div class="playbar"><button id="play-all" type="button">全部播放</button><button id="replay-all" type="button">从头播放</button><button id="pause-all" type="button">全部暂停</button><button id="reset-all" type="button">回到开头</button><input id="timeline" type="range" min="0" max="1000" value="0" aria-label="统一播放进度"><span class="playbar-time" id="play-time">00:00 / 00:00</span></div>
<script>
const SEEDS=["851","3278","11395","20379","28221","32098"];
const GROUPS=[
 {title:"S-all",note:"全部公共稳定空间局部 Head",variants:["baseline","S_steps00_10","S_steps00_15","S_steps05_10","S_steps05_15","S_steps10_20","S_steps20_30","S_steps30_40"]},
 {title:"T-all",note:"全部公共稳定运动轨迹 Head",variants:["baseline","T_steps00_10","T_steps00_15","T_steps05_10","T_steps05_15","T_steps10_20","T_steps20_30","T_steps30_40"]},
 {title:"C-all",note:"全部公共稳定上下文/历史 Head",variants:["baseline","C_steps00_10","C_steps00_15","C_steps05_10","C_steps05_15","C_steps10_20","C_steps20_30","C_steps30_40"]}
];
let DATA=null,seeking=false;
const q=id=>document.getElementById(id);
const media=src=>src?`../${src}`:null;
function card(src,label){return src?`<figure><video muted playsinline preload="metadata" src="${media(src)}"></video><figcaption>${label}</figcaption></figure>`:`<div class="missing">等待生成</div>`}
function refPath(item,key){if(!item[key])return null;if(key==="source_video")return `media/references/${item.id}__source_video_49f.mp4`;const ext=item[key].split(".").pop();return `media/references/${item.id}__${key}.${ext}`}
function label(v){return DATA.role_names[v]||v}
function render(){
 pauseAll();
 const seed=q("seed").value,item=DATA.cases[0],videos=DATA.videos[item.id][seed];
 q("prompt").textContent=`Prompt: ${item.prompt}`;
 q("refs").innerHTML=card(refPath(item,"source_video"),"Source / GT")+card(refPath(item,"context_video"),"8-frame context");
 q("groups").innerHTML=GROUPS.map(group=>{
  const head=group.variants.map(v=>`<th>${label(v)}${v==="baseline"?"":`<span class="target-count">${DATA.target_counts[v]} Heads</span>`}</th>`).join("");
  const rows=DATA.models.map(m=>{
   const cells=group.variants.map(v=>`<td class="video-cell" data-variant="${v}" data-model="${m}">${card(videos[m][v],label(v))}</td>`).join("");
   return `<tr><th class="model">${DATA.model_names[m]}</th>${cells}</tr>`;
  }).join("");
  return `<section class="group"><div class="group-title"><h2>${group.title}</h2><span class="group-note">${group.note}</span></div><table><thead><tr><th>模型</th>${head}</tr></thead><tbody>${rows}</tbody></table></section>`;
 }).join("");
 updateStatus();
}
function refreshCells(){
 const seed=q("seed").value,item=DATA.cases[0],videos=DATA.videos[item.id][seed];
 document.querySelectorAll(".video-cell").forEach(cell=>{
  if(cell.querySelector("video"))return;
  const src=videos[cell.dataset.model][cell.dataset.variant];
  if(src)cell.innerHTML=card(src,label(cell.dataset.variant));
 });
 updateStatus();
}
function updateStatus(){
 const seed=q("seed").value,item=DATA.cases[0],videos=DATA.videos[item.id][seed];
 const variants=GROUPS.flatMap(g=>g.variants);
 const done=variants.reduce((n,v)=>n+DATA.models.filter(m=>videos[m][v]).length,0);
 q("status").textContent=`Seed ${seed} 视频 ${done}/${variants.length*DATA.models.length}`;
 q("updated").textContent=`更新于 ${new Date().toLocaleTimeString()}`;
}
function videos(){return [...document.querySelectorAll("video")].filter(v=>Number.isFinite(v.duration)&&v.duration>0)}
function fmt(seconds){const x=Math.max(0,Math.floor(Number.isFinite(seconds)?seconds:0));return `${String(Math.floor(x/60)).padStart(2,"0")}:${String(x%60).padStart(2,"0")}`}
function syncTime(){const list=videos(),master=list[0];if(!master){q("play-time").textContent="00:00 / 00:00";return}if(!seeking)q("timeline").value=String(Math.round(1000*master.currentTime/master.duration));q("play-time").textContent=`${fmt(master.currentTime)} / ${fmt(master.duration)}`}
function seekAll(f){videos().forEach(v=>{v.currentTime=Math.min(v.duration,Math.max(0,v.duration*f))})}
function playAll(){const list=videos();if(!list.length)return;seekAll(Number(q("timeline").value)/1000);list.forEach(v=>{v.loop=false;v.play().catch(()=>{})})}
function pauseAll(){videos().forEach(v=>v.pause())}
function resetAll(){pauseAll();q("timeline").value="0";seekAll(0);syncTime()}
function replayAll(){q("timeline").value="0";seekAll(0);playAll()}
async function load(){
 try{
  const response=await fetch(`../manifest.json?t=${Date.now()}`,{cache:"no-store"});
  if(!response.ok)throw new Error(`HTTP ${response.status}`);
  const first=!DATA;DATA=await response.json();
  if(first){q("seed").innerHTML=SEEDS.map(s=>`<option value="${s}">${s}</option>`).join("");render()}else refreshCells();
 }catch(error){q("status").textContent=`刷新失败: ${error.message}`}
}
q("seed").addEventListener("change",render);q("refresh").addEventListener("click",load);
q("play-all").addEventListener("click",playAll);q("replay-all").addEventListener("click",replayAll);q("pause-all").addEventListener("click",pauseAll);q("reset-all").addEventListener("click",resetAll);
q("timeline").addEventListener("pointerdown",()=>{seeking=true});q("timeline").addEventListener("input",event=>{seekAll(Number(event.target.value)/1000);syncTime()});q("timeline").addEventListener("change",()=>{seeking=false;syncTime()});
setInterval(syncTime,250);load();setInterval(load,10000);
</script></body></html>
"""


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HTML, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
