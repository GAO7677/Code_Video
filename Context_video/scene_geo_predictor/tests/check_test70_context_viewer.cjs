// Browser smoke audit of every case/stage and both projection modes.
const {spawn}=require('node:child_process'),fs=require('node:fs');
const root=process.argv[2],url=process.argv[3];
if(!root||!url)throw Error('usage: node check_test70_context_viewer.cjs OUTPUT URL');
const b=spawn('/data/gaoya/agent-data/cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell',['--no-sandbox','--disable-gpu','--disable-dev-shm-usage','--remote-debugging-pipe'],{stdio:['ignore','ignore','pipe','pipe','pipe']});
let seq=0,buf='';const pending=new Map(),errors=[];
b.stdio[4].on('data',c=>{buf+=c;let i;while((i=buf.indexOf('\0'))>=0){const m=JSON.parse(buf.slice(0,i));buf=buf.slice(i+1);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result)}if(m.method==='Runtime.exceptionThrown')errors.push(m.params)}});
function cdp(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++seq;pending.set(id,{resolve,reject});b.stdio[3].write(JSON.stringify({id,method,params,sessionId})+'\0')})}
(async()=>{
 const {targetId}=await cdp('Target.createTarget',{url:'about:blank'});
 const {sessionId}=await cdp('Target.attachToTarget',{targetId,flatten:true});
 const call=(m,p)=>cdp(m,p,sessionId);
 const ev=async expression=>{const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
 await call('Runtime.enable');await call('Page.enable');
 await call('Emulation.setDeviceMetricsOverride',{width:1600,height:1100,deviceScaleFactor:1,mobile:false});
 await call('Page.navigate',{url});
 await ev(`new Promise((resolve,reject)=>{let n=0,t=setInterval(()=>{if(document.querySelector('#caseList')?.children.length===70&&document.querySelector('#loadingOverlay').hidden){clearInterval(t);resolve(true)}else if(++n>300){clearInterval(t);reject('timeout')}},100)})`);
 if(process.env.VIDEO_AUDIT){
  if(process.env.LAZY_AUDIT){
   const network=await ev(`performance.getEntriesByType('resource').map(r=>({url:r.name,bytes:r.transferSize}))`);
   if(network.some(r=>/viewer_data\.json|\.png(?:\?|$)/.test(r.url)))throw Error('Video startup fetched full dataset or canvas images');
   if(!network.some(r=>r.url.includes('viewer_index.json')))throw Error('Missing lazy index');
   console.log('PASS lazy startup: no full dataset or PNG requests',network);
  }
  for(let i=0;i<70;i++)await ev(`(async()=>{await load('clip_${String(i).padStart(3,'0')}');for(const s of ['context','mask','state','depth','geometry','rollout']){await selectStage(s);const v=$('overlayVideo');await new Promise((resolve,reject)=>{let n=0,t=setInterval(()=>{if(v.error){clearInterval(t);reject(v.error.message)}else if(v.readyState>=2){clearInterval(t);resolve()}else if(++n>100){clearInterval(t);reject('video load timeout')}},50)});const expected=['geometry','rollout'].includes(s)?49/30:8/30;if(Math.abs(v.duration-expected)>.04)throw Error('duration mismatch');v.currentTime=Math.max(0,v.duration-.08);await new Promise((resolve,reject)=>{let n=0,t=setInterval(()=>{if(!v.seeking&&v.readyState>=2){clearInterval(t);resolve()}else if(++n>100){clearInterval(t);reject('seek timeout')}},30)})}})()`);
  const playback=await ev(`(async()=>{await load('clip_000');await selectStage('rollout');const v=$('overlayVideo');v.muted=true;v.playbackRate=.5;await v.play();await new Promise(r=>setTimeout(r,1000));const t=v.currentTime;v.pause();if(t<.2||t>.9)throw Error('playback did not advance at half speed');$('toggleVideoMode').click();if(!$('overlayVideo').hidden)throw Error('interactive toggle');$('toggleVideoMode').click();return {advanced_seconds:t,rate:.5}})()`);
  const shot=await call('Page.captureScreenshot',{format:'png'});fs.writeFileSync(root+'/video_browser_screenshot.png',Buffer.from(shot.data,'base64'));
  if(errors.length)throw Error(JSON.stringify(errors));fs.writeFileSync(root+'/video_browser_check.json',JSON.stringify({status:'PASS',cases:70,videos:420,seek:'PASS',playback,errors},null,2));
  console.log('PASS 420 videos loaded/seeked; native playback and mode toggle');await cdp('Browser.close');return;
 }
 if(process.env.PLAYBACK_AUDIT){
  const result=await ev(`(async()=>{
   await load('clip_000');await selectStage('mask');
   const original=HTMLImageElement.prototype.decode;
   let decodes=0,paints=0;const gaps=[];let previous=0;
   HTMLImageElement.prototype.decode=async function(){decodes++;await new Promise(r=>setTimeout(r,150));return original.call(this)};
   const paint=ctx.drawImage.bind(ctx);ctx.drawImage=(...a)=>{const now=performance.now();if(previous)gaps.push(now-previous);previous=now;paints++;return paint(...a)};
   $('playbackFps').value='30';$('playPause').click();
   await new Promise(r=>setTimeout(r,2500));if(playing)$('playPause').click();
   await new Promise(r=>setTimeout(r,200));
   HTMLImageElement.prototype.decode=original;ctx.drawImage=paint;
   return {paints,decodes,max_gap_ms:Math.max(0,...gaps),status:paints>=35&&decodes<=8?'PASS':'FAIL'};
  })()`);
  fs.writeFileSync(root+'/playback_'+process.env.PLAYBACK_AUDIT+'.json',JSON.stringify(result,null,2));console.log(result);
  await cdp('Browser.close');if(result.status!=='PASS')process.exitCode=1;return;
 }
 for(let i=0;i<70;i++)await ev(`(async()=>{await load('clip_${String(i).padStart(3,'0')}');for(const s of ['context','mask','state','depth','geometry','rollout']){await selectStage(s);if(!$('loadingOverlay').hidden)throw Error('loading');if(!$('stageSummary').textContent)throw Error('empty inspector')} $('evalCamera').checked=true;await draw();$('evalCamera').checked=false;await draw();if(document.querySelectorAll('[data-layer]').length!==5)throw Error('layers')})()`);
 await ev(`(async()=>{await load('clip_000');await selectStage('rollout');$('evalCamera').checked=true;await draw()})()`);
 const shot=await call('Page.captureScreenshot',{format:'png'});
 fs.writeFileSync(root+'/browser_screenshot.png',Buffer.from(shot.data,'base64'));
 if(errors.length)throw Error(JSON.stringify(errors));
 fs.writeFileSync(root+'/browser_check.json',JSON.stringify({status:'PASS',cases:70,stages:6,projection_modes:2,errors},null,2));
 console.log('PASS 70 cases x 6 stages, estimate/eval projection modes');await cdp('Browser.close');
})().catch(e=>{console.error(e);b.kill();process.exitCode=1});
