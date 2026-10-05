from pathlib import Path
import shutil
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_sweep_bench/20260929_unified_evaluation')
B=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261005_test70_v2_approved_motion_render/bench_backup');B.mkdir(exist_ok=True)
for name in ['bench.js','bench.html']:
 if not (B/name).exists():shutil.copy2(R/name,B/name)
helper=r'''// Live overlay of the four approved Test70 v2 motion revisions.
const motionRoot='/test70-approved-render/';
const motionLabels={SCENE_DOOR_FRAME_BALL:'门洞宽度（球） · 暖色挂画客厅',SCENE_DOOR_FRAME:'门洞宽度（木块） · 暖灰石材住宅玄关',V2V_OBSTACLE_SIZE:'固定门宽0.46 m／小球半径 · 经典橱柜厨房',V2V_PENDULUM_CABINET:'固定摆锤／柜体位置 · 简洁石材房间'};
let motionStatus=null,motionLoadError=null;
async function withMotionRevision(original){
 try{
  const replies=await Promise.all(['gallery.json','status.json'].map(f=>fetch(motionRoot+f+'?t='+Date.now(),{cache:'no-store'})));
  if(replies.some(r=>!r.ok))throw Error('渲染进度暂不可用');
  const [gallery,status]=await Promise.all(replies.map(r=>r.json()));
  if(gallery.cases.length!==20)throw Error('新版20例清单尚未齐全');
  const state=new Map(status.cases.map(c=>[c.case,c])), groups=new Map();
  for(const c of gallery.cases){
   if(!motionLabels[c.family])throw Error('未知场景组');
   if(!groups.has(c.family))groups.set(c.family,[]);
   const progress=state.get(c.case)||{},ready=Boolean(c.complete),active=progress.gpu!==null&&progress.gpu!==undefined;
   const stateText=ready?'新版完整视频已完成':progress.error?(active?'重试中':'任务异常／待重试'):active?'渲染中':'等待渲染';
   const detail=ready?'90/90帧':`${progress.frame||0}/90帧${c.poster?' · 已有首帧预览':''}`;
   const prompt=c.family==='V2V_PENDULUM_CABINET'?'摆长1.2 m，释放角45°，柜体中心等距排列，无底座；立柱与横梁保留。':c.family==='V2V_OBSTACLE_SIZE'?'固定门洞宽0.46 m，小球半径为0.123／0.169／0.215／0.261／0.307 m。':'门洞融入墙体，门高2.1 m，门扇固定打开100°。';
   groups.get(c.family).push([c.case,{family:c.family,variant:c.value+' m',control_label:c.variable+' = '+c.value+' m',prompt,gt_url:ready?motionRoot+c.video+'?v=approved-motion':'',poster_url:c.poster?motionRoot+c.poster+'?v=approved-motion':'',pending:!ready,render_state:stateText+' · '+detail}]);
  }
  if(groups.size!==4||[...groups.values()].some(g=>g.length!==5))throw Error('新版场景分组不完整');
  const cases={},added=new Set();
  for(const [id,info] of Object.entries(original.cases)){
   if(groups.has(info.family)){if(!added.has(info.family)){for(const [newId,newInfo] of groups.get(info.family))cases[newId]=newInfo;added.add(info.family);}}
   else cases[id]=info;
  }
  if(Object.keys(cases).length!==70)throw Error('新版清单样本数不等于70');
  motionStatus=status;motionLoadError=null;
  return {...original,cases,family_labels:{...original.family_labels,...motionLabels}};
 }catch(error){motionLoadError=error.message;return original;}
}
function updateBenchFigure(figure,id,info){
 const video=figure.querySelector('video');
 const url=info.pending?'':info.gt_url||'media/GT/'+encodeURIComponent(id)+'.mp4';
 if((video.getAttribute('src')||'')!==url){video.pause();if(url)video.src=url;else{video.removeAttribute('src');video.load();}figure.querySelectorAll('.error').forEach(e=>e.remove());}
 if((video.getAttribute('poster')||'')!==(info.poster_url||'')){if(info.poster_url)video.poster=info.poster_url;else video.removeAttribute('poster');}
 video.controls=!info.pending;video.dataset.caseId=id;
 video.setAttribute('aria-label',id+(info.pending?' 新版首帧预览':' GT参考视频'));
 const placeholder=figure.querySelector('.pending-placeholder');placeholder.hidden=!(info.pending&&!info.poster_url);video.hidden=info.pending&&!info.poster_url;
 const caption=figure.querySelector('figcaption');caption.replaceChildren(document.createTextNode(id+(info.variant?' · 变体 '+info.variant:'')));
 if(info.control_label)caption.append(make('small',info.control_label));
 if(info.render_state){const note=make('small',info.render_state);note.className='render-state';caption.append(note);}
 caption.append(make('small',info.prompt||'暂无提示词'));
}
function createBenchFigure(id,info){
 const f=make('figure'),v=make('video');v.muted=true;v.playsInline=true;v.preload='none';
 v.addEventListener('error',()=>{if(v.hasAttribute('src')&&!f.querySelector('.error')){const e=make('small','参考视频暂不可用');e.className='error';f.append(e);}});
 const placeholder=make('div','新版视频渲染中 · 首帧尚未生成');placeholder.className='pending-placeholder';placeholder.hidden=true;
 f.append(v,placeholder,make('figcaption'));updateBenchFigure(f,id,info);return f;
}
function updateMotionNote(){
 const note=document.getElementById('v2-revision-status');if(!note)return;
 if(motionLoadError){note.textContent='本轮新版状态读取失败，当前保留上一发布版：'+motionLoadError;return;}
 if(!motionStatus)return;
 const counts=motionStatus.active_per_gpu||{};
 note.textContent=`本轮4组20例写实渲染：完整视频 ${motionStatus.completed}/20；GPU0/1/2并发 ${counts['0']||0}/${counts['1']||0}/${counts['2']||0}。未完成项显示首帧与进度，不播放旧版视频。`+(motionStatus.error?' 部分任务失败或重试中，请查看进度详情。':'');
 note.append(document.createTextNode(' '),link('渲染进度详情',motionRoot));
}
'''
(R/'bench-motion.js').write_text(helper)
p=R/'bench.js';s=(B/'bench.js').read_text()
s=s.replace("const data=await r.json(),cases=Object.entries(data.cases),families=new Map();", "const original=await r.json(),data=key==='test70-v2'?await withMotionRevision(original):original,cases=Object.entries(data.cases),families=new Map();")
a=s.index('  for(const [id,info] of items){const f=');b=s.index('\n  let generation=0;',a)
s=s[:a]+"  for(const [id,info] of items)grid.append(createBenchFigure(id,info));"+s[b:]
s=s.replace("grid.querySelectorAll('video')].map", "grid.querySelectorAll('video[src]')].map")
s=s.replace("message.textContent=failures?", "message.textContent=results.length===0?'本组尚无完整视频，当前仅显示首帧预览':failures?")
s=s.replace('包含新版坡度住宅背景、缝隙暖光和多米诺26°近景。','本轮更新两组真实门洞、第10组固定门宽改变球半径、第12组固定摆锤改变柜体位置；未完成项显示首帧及进度。')
s=s.replace("?'完整新版数据集 →'", "?'上一发布版完整数据集 →'").replace("link('Bench 说明 PDF'", "link('上一发布版说明 PDF'").replace("link('Markdown＋图片'", "link('上一发布版 Markdown＋图片'")
a=s.index('async function refreshTest70V2(){');b=s.index('setInterval(refreshTest70V2',a)
s=s[:a]+'''async function refreshTest70V2(){
 if(document.body.dataset.loaded!=='true')return;
 try{
  const response=await fetch('test70_v2_bench.json?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error(response.status);const data=await withMotionRevision(await response.json());
  for(const category of document.querySelectorAll('#test70-v2 .category')){
   const family=category.dataset.family,items=Object.entries(data.cases).filter(([,i])=>i.family===family),grid=category.querySelector('.videos');
   const current=[...grid.querySelectorAll('video')].map(v=>v.dataset.caseId);
   if(current.join('|')!==items.map(([id])=>id).join('|'))grid.replaceChildren(...items.map(([id,info])=>createBenchFigure(id,info)));
   else for(const [id,info] of items){const figure=[...grid.querySelectorAll('figure')].find(f=>f.querySelector('video').dataset.caseId===id);updateBenchFigure(figure,id,info);}
   const label=data.family_labels?.[family];if(label){category.querySelector('h3').textContent=label+' · '+family+'（'+items.length+'例）';const a=document.querySelector('#directory a[href="#'+category.id+'"]');if(a)a.textContent=label+' · '+family+' · '+items.length;}
  }
  updateMotionNote();
  const n=document.getElementById('v2-refresh-status');if(n)n.textContent='新版清单与渲染进度已同步：'+new Date().toLocaleTimeString();
 }catch(e){const n=document.getElementById('v2-refresh-status');if(n)n.textContent='新版清单同步失败：'+e.message;}
}
''' +s[b:];p.write_text(s)
p=R/'bench.html';s=(B/'bench.html').read_text().replace('</style>', '.pending-placeholder{aspect-ratio:7/4;display:flex;align-items:center;justify-content:center;background:#edf1f3;color:#52656e;padding:20px;text-align:center}.pending-placeholder[hidden],video[hidden]{display:none}.render-state{color:#926000;font-weight:600}#v2-revision-status{padding:12px;background:#fff4db;line-height:1.7}</style>')
s=s.replace('<script src="bench.js?v=20261005-v2-autorefresh"></script>', '<script src="bench-motion.js?v=20261005-approved-motion"></script><script src="bench.js?v=20261005-approved-motion"></script>')
p.write_text(s)
print('Updated bench live overlay, case status and refresh')
