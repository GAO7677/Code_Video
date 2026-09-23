'use strict';
// Display-only encoded overlays; estimation files are never changed here.
const overlayVideo=document.createElement('video');
overlayVideo.id='overlayVideo';overlayVideo.controls=true;overlayVideo.loop=true;
overlayVideo.playsInline=true;overlayVideo.preload='metadata';
overlayVideo.style.cssText='display:block;width:100%;height:auto;background:#172020';
$('overlayCanvas').before(overlayVideo);
const videoToolbar=document.createElement('div');
videoToolbar.style.cssText='padding:10px;background:#eef2ef;display:flex;gap:12px;align-items:center;flex-wrap:wrap';
videoToolbar.innerHTML='<button id="toggleVideoMode" type="button">切换逐帧交互</button><label>视频速度 <select id="videoRate"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select></label><a id="videoDownload" download>下载 MP4</a><span id="videoNote"></span>';
document.querySelector('.canvas-stage').before(videoToolbar);
let videoMode=true,videoKey='';
function syncVideo(){
 if(!data)return;
 $('overlayCanvas').hidden=videoMode;$('overlayCanvas').style.display=videoMode?'none':'';
 overlayVideo.hidden=!videoMode;overlayVideo.style.display=videoMode?'block':'none';
 document.querySelector('.transport').hidden=videoMode;document.querySelector('.transport').style.display=videoMode?'none':'';
 $('canvasKey').style.display=videoMode?'none':'';
 $('layerControls').hidden=videoMode||stage!=='rollout';
 $('evalCamera').disabled=videoMode;
 if(videoMode)$('evalCamera').checked=true;
 $('toggleVideoMode').textContent=videoMode?'切换逐帧交互':'切换视频播放';
 document.querySelector('.layer-reference p:nth-of-type(2)').textContent=videoMode?'视频模式：rollout叠加在真实RGB0–48运动背景，未来RGB和GT相机仅用于冻结后的评测展示。geometry阶段保持RGB7背景。图层已写入MP4，切回逐帧交互可开关。A/B/C与GT几何本轮NOT_RUN。':'逐帧模式：未来轨迹叠加在RGB7静止背景，可切换图层及估计/GT评测相机投影。失败无预测轨迹。A/B/C与GT几何本轮NOT_RUN。';
 $('videoNote').textContent=videoMode?(stage==='rollout'?'完整RGB0–48背景；未来RGB/GT相机仅评测。绿GT、蓝CV、红D。':'当前阶段预渲染视频；图层已写入MP4。'):'';
 const key=data.id+'/'+stage;
 if(videoMode&&key!==videoKey){
  overlayVideo.pause();videoKey=key;
  overlayVideo.src='overlay_videos_v2/'+key+'.mp4';overlayVideo.load();
  overlayVideo.playbackRate=+$('videoRate').value;
  $('videoDownload').href=overlayVideo.src;
 }
 if(!videoMode)overlayVideo.pause();
 if(videoMode){$('loadingOverlay').hidden=true;videoFrame()}
}
function videoFrame(){if(!videoMode)return;const t=Math.min(stage==='rollout'||stage==='geometry'?48:7,Math.floor(overlayVideo.currentTime*30));$('frameBadge').textContent=stage==='geometry'?'RGB7 HOLD':`RGB${t} · VIDEO`;$('scopeBadge').textContent=stage==='rollout'?'EVALUATION':'ESTIMATOR';$('frameReadout').innerHTML=metric('视频时间',fmt(overlayVideo.currentTime)+' s')+(stage==='rollout'?metric('D逐帧误差 m',t>=8?fmt(data.evaluation.trajectory_metrics?.frame_errors_m[t-8]):'Context'):'')}
overlayVideo.ontimeupdate=videoFrame;
overlayVideo.onerror=()=>{$('videoNote').textContent='视频加载失败，可切换逐帧交互查看。'};
$('videoRate').onchange=()=>{overlayVideo.playbackRate=+$('videoRate').value};
$('toggleVideoMode').onclick=()=>{++playRequest;playing=false;videoMode=!videoMode;syncVideo();if(!videoMode)draw()};
const drawWithoutVideo=draw;
draw=async function(){await drawWithoutVideo();syncVideo()};
syncVideo();
