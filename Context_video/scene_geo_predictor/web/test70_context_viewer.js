'use strict';
const $=id=>document.getElementById(id),ctx=$('overlayCanvas').getContext('2d');
let records=[],data,stage='context',frame=0,filter='all',playing=false,last=0,ticket=0;
// Bound decoded images to the selected case, not the full 70-case collection.
const images=new Map(),decoded=new Map();let meshPaths={},playRequest=0;
function frameUrl(r,s,t){return s==='depth'?`diagnostics/${r.id}/depth_${t}.png`:s==='geometry'?`diagnostics/${r.id}/mesh.png`:`inputs/${r.id}/rgb_${String(s==='rollout'?7:t).padStart(2,'0')}.png`}
function warm(r,s){return Promise.all(Array.from({length:s==='rollout'||s==='geometry'?1:8},(_,t)=>image(frameUrl(r,s,t))))}
function segmentsPath(segments){const p=new Path2D();for(const seg of segments||[]){if(seg.length===2&&seg.flat().every(Number.isFinite)){p.moveTo(...seg[0]);p.lineTo(...seg[1])}}return p}
function setHtml(id,value){const el=$(id);if(el.innerHTML!==value)el.innerHTML=value}
const titles={context:'Context RGB',mask:'Grounding DINO + SAM2',state:'Center / radius / p7 / v7',depth:'VGGT depth',geometry:'Observed mesh + local completion',rollout:'Zero-omega PyBullet'};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>v==null?'NOT_RUN':Number(v).toFixed(3),metric=(k,v)=>`<div class="metric-row"><span>${esc(k)}</span><strong>${esc(v)}</strong></div>`;
const limit=()=>stage==='rollout'?41:stage==='geometry'?0:7;
const active=k=>document.querySelector(`[data-layer="${k}"]`).checked;
function list(){const rows=records.filter(r=>(filter==='all'||r.rollout_status===filter||(filter==='FAIL'&&r.rollout_status==='UNKNOWN'))&&r.case_id.includes($('caseSearch').value));$('visibleCaseCount').textContent=rows.length;$('caseList').innerHTML=rows.map(r=>`<button class="case-row ${r===data?'is-active':''}" data-case="${r.id}"><strong>${esc(r.case_id)}</strong><span class="case-ade">${esc(r.rollout_status)}</span><small>${r.id}</small><small>${r.evaluation?.trajectory_metrics?fmt(r.evaluation.trajectory_metrics.ADE_m)+' m':'未取得有效轨迹指标'}</small></button>`).join('')}
function image(url){if(!images.has(url)){const promise=(async()=>{const im=new Image();im.src=url;await im.decode();if(images.get(url)===promise)decoded.set(url,im);return im})();images.set(url,promise);promise.catch(()=>{if(images.get(url)===promise)images.delete(url)})}return images.get(url)}
function line(points,color,width=2){if(!points?.length)return;ctx.beginPath();let first=true;for(const p of points){if(!p||!p.every(Number.isFinite)){first=true;continue}if(first){ctx.moveTo(...p);first=false}else ctx.lineTo(...p)}ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
async function load(id){++playRequest;data=records.find(r=>r.id===id);playing=false;images.clear();decoded.clear();meshPaths={mesh:segmentsPath(data.mesh_segments),completion:segmentsPath(data.completion_segments)};$('caseTitle').textContent=data.case_id;$('familyBadge').textContent=data.id;$('variantText').textContent=data.rollout_status;list();frame=stage==='rollout'?41:0;await draw();warm(data,stage).catch(()=>{})}
async function selectStage(s){++playRequest;stage=s;frame=s==='rollout'?41:0;playing=false;await draw();warm(data,stage).catch(()=>{})}
async function draw(){if(!data)return;const token=++ticket,r=data,t=Math.min(frame,7),evalCam=$('evalCamera').checked;
 const url=frameUrl(r,stage,t);
 try{if(!decoded.has(url)){$('loadingOverlay').textContent='读取当前阶段帧…';$('loadingOverlay').hidden=false}const im=decoded.get(url)||await image(url);if(token!==ticket)return;ctx.clearRect(0,0,640,360);ctx.drawImage(im,0,0,640,360);
 if(stage==='mask'){r.mask_contours[t].forEach(c=>line(c.concat([c[0]]),'#37aef5'));if(t===7){const b=r.prompt_box;ctx.strokeStyle='#ffdc32';ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1])}}
 if(stage==='state'&&r.center_uv){line(r.center_uv.slice(0,t+1),'#ffdc32');for(const p of r.center_uv.slice(0,t+1)){if(p.every(Number.isFinite)){ctx.beginPath();ctx.arc(...p,3,0,Math.PI*2);ctx.strokeStyle='#ffdc32';ctx.stroke()}}}
 const keys=[];
 if(stage==='rollout'){
  const paths=[['gt',evalCam?r.evaluation.gt_future_uv:null,'GT · eval only','#42df91'],['cv',evalCam?r.evaluation.cv_eval_uv:r.cv_future_uv,'CV','#37aef5'],['D',evalCam?r.evaluation.pred_eval_uv:r.new_future_uv,'D · zero omega','#ff514c']];
  for(const [key,points,name,color] of paths)if(active(key)){if(points){line(points.slice(0,frame),'#111',4);line(points.slice(0,frame),color);keys.push([name,color])}else keys.push([name+(key==='gt'&&!evalCam?' · 需启用评测投影':' · NOT_RUN'),color])}
  if(active('mesh')){ctx.strokeStyle='#f453dc';ctx.lineWidth=1;ctx.stroke(meshPaths.mesh);keys.push(['Observed mesh boundary','#f453dc'])}
  if(active('completion')){ctx.strokeStyle='#ffdd44';ctx.lineWidth=1;ctx.stroke(meshPaths.completion);keys.push(['Inferred plane · 推断','#ffdd44'])}
 }
 setHtml('canvasKey',keys.map(([n,c])=>`<span class="key-item"><i class="key-dot" style="color:${c}"></i>${esc(n)}</span>`).join(''));$('loadingOverlay').hidden=true;
 }catch(e){$('loadingOverlay').textContent='加载失败 '+e.message;throw e}
 $('timeline').max=limit();$('timeline').value=frame;$('timeline').disabled=stage==='geometry';$('frameBadge').textContent=`RGB${stage==='rollout'?frame+7:stage==='geometry'?7:frame}${stage==='rollout'?' · RGB7 HOLD':''}`;$('timeOutput').textContent=fmt(frame/30)+' s';$('playPause').textContent=playing?'Ⅱ':'▶';
 document.querySelectorAll('[data-stage]').forEach(b=>b.classList.toggle('is-active',b.dataset.stage===stage));$('layerControls').hidden=stage!=='rollout';$('inspectorStep').textContent=titles[stage];$('stageStatus').textContent=stage==='rollout'?r.rollout_status:stage==='state'?r.state.status:stage==='geometry'?r.geometry.status:'OBSERVED';
 $('scopeBadge').textContent=stage==='rollout'&&evalCam?'EVALUATION':'ESTIMATOR';let rows='',note='';const s=r.state,m=r.evaluation.state_metrics;
 if(stage==='context'){rows=metric('输入','RGB0–7 + 时间戳')+metric('目标短语',r.target_detection.prompt)+metric('支持范围',r.support_reason||'单球非关节候选');note='全部70例；范围来自运行前context筛查，不是根据GT或预测结果筛选。'}
 if(stage==='mask'){rows=metric('定位','DINO RGB7唯一框')+metric('跟踪','SAM2双向传播')+metric('外轮廓数',r.mask_contours[t].length);note='蓝色：原始SAM2边界；黄色：RGB7提示框。未用球形先验修整mask。'}
 if(stage==='state'){rows=metric('状态',s.status)+metric('radius (m)',fmt(s.radius))+metric('p7',JSON.stringify(s.p7??null))+metric('v7',JSON.stringify(s.v7??null))+metric('radius CV',fmt(s.radius_cv))+metric('相对RMS',fmt(s.relative_rms))+metric('p7 GT error',fmt(m?.p7_error_m))+metric('v7 GT error',fmt(m?.v7_vector_error_mps))+metric('radius GT error',fmt(m?.radius_error_m));note=JSON.stringify(s.reason||s.reasons||'')+'；失败拟合值仅诊断，不进入rollout。';}
 if(stage==='depth'){rows=metric('模型','VGGT-1B')+metric('固定scale','5.819486884015457');note='保留原始坐标与深度；每clip八帧共用2–98%色域。固定scale来自旧pilot，尚不代表test70米制准确。'}
 if(stage==='geometry'){rows=metric('三角形',r.geometry.triangles??'NOT_RUN')+metric('几何用途',r.geometry.diagnostic_only?'仅诊断':'待验证碰撞输入')+metric('推断像素',r.completion_audit?.inferred_pixels??0)+metric('重力',r.gravity.gravity.status)+metric('g vector',JSON.stringify(r.gravity.gravity.vector));note='紫色：可见mesh；黄色：局部平面补全。厚度UNKNOWN；不移动球、不添加默认地面。非支持场景可能包含其他运动物体。'}
 if(stage==='rollout'){const tm=r.evaluation.trajectory_metrics;rows=metric('D',r.rollout_status)+metric('ADE / FDE',fmt(tm?.ADE_m)+' / '+fmt(tm?.FDE_m))+metric('同输入CV ADE',fmt(tm?.CV_ADE_m))+metric('初始穿透 m',fmt(r.initial_penetration_m))+metric('初始接触',r.initial_contacts??'NOT_RUN')+metric('接触帧',r.contacts?.frame_contacts??'NOT_RUN')+metric('首接触 s',fmt(r.contacts?.first_contact_time_s))+metric('最大穿透 m',fmt(r.contacts?.max_penetration_vs_rollout_geometry_m))+metric('contact accuracy','NOT_EVALUATED');note=r.failure?JSON.stringify(r.failure):'已推进41帧。EXECUTED不等于准确性PASS。A/B/C本轮NOT_RUN。';}
 setHtml('stageSummary',`<h1>${titles[stage]}</h1><p>${esc(note)}</p><div class="metric-stack">${rows}</div><details><summary>逐例原始审计</summary><a href="cases/${r.id}.json">case JSON</a> · <a href="estimates/${r.id}/result.json">估计字段与来源</a></details>`);
 $('boundaryText').textContent='输入只读context及冻结mask；目标短语、固定尺度、下方平面重力与局部补全是显式prior。GT全部在预测冻结后读取。';$('frameReadout').innerHTML=metric('时间',fmt(frame/30)+' s')+(stage==='rollout'?metric('D逐帧误差 m',frame?fmt(r.evaluation.trajectory_metrics?.frame_errors_m[frame-1]):'初始化'):'');
}
$('caseList').onclick=e=>{const b=e.target.closest('[data-case]');if(b)load(b.dataset.case)};$('caseSearch').oninput=list;
document.querySelectorAll('[data-filter]').forEach(b=>b.onclick=()=>{filter=b.dataset.filter;document.querySelectorAll('[data-filter]').forEach(x=>x.classList.toggle('is-active',x===b));list()});
document.querySelectorAll('[data-stage]').forEach(b=>b.onclick=()=>selectStage(b.dataset.stage));document.querySelectorAll('[data-layer]').forEach(b=>b.onchange=draw);$('evalCamera').onchange=draw;
$('timeline').oninput=e=>{++playRequest;frame=+e.target.value;playing=false;draw()};$('previousFrame').onclick=()=>{++playRequest;playing=false;frame=Math.max(0,frame-1);draw()};$('nextFrame').onclick=()=>{++playRequest;playing=false;frame=Math.min(limit(),frame+1);draw()};
$('playPause').onclick=async()=>{const request=++playRequest;if(playing){playing=false;await draw();return}if(!limit())return;$('playPause').textContent='…';try{await warm(data,stage);if(request!==playRequest)return;playing=true;last=performance.now();if(frame===limit())frame=0;await draw()}catch(e){$('playPause').textContent='▶';$('loadingOverlay').hidden=false;$('loadingOverlay').textContent='预加载失败：'+e.message}};
$('playbackFps').onchange=()=>{last=performance.now()};
async function tick(now){try{const interval=1000/+$('playbackFps').value;if(playing&&now-last>=interval){const steps=Math.floor((now-last)/interval);last+=steps*interval;frame=(frame+steps)%(limit()+1);await draw()}}finally{requestAnimationFrame(tick)}}
(async()=>{const d=await(await fetch('viewer_data.json')).json();records=d.records;$('summaryAde').textContent=fmt(d.summary.ADE_m)+'m';$('summaryFde').textContent=fmt(d.summary.FDE_m)+'m';$('summaryContact').textContent=(d.summary.rollout.EXECUTED||0)+'/70';$('summaryContact').nextElementSibling.textContent='EXECUTED';await load(records[0].id);requestAnimationFrame(tick)})();
