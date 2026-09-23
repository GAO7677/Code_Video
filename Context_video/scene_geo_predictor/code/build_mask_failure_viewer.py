"""Read-only diagnostics of frozen 36-case masks, not a refinement experiment."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from context_rgb_pybullet_common import crop_transform,resize_crop_mask,dump_json,sha256_file

ROOT=Path('/data/gaoya/agent-data/outputs/context_generic_pilot36_20260922_v1')
OLD=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
OUT=ROOT/'failure_diagnostics_20260923_v1'

def main():
    cv2.setNumThreads(2);OUT.mkdir(exist_ok=False)
    for n,h in json.loads((ROOT/'estimate_freeze.json').read_text()).items():
        if sha256_file(ROOT/n)!=h:raise ValueError('source freeze changed')
    rows=[]
    for mapping in json.loads((ROOT/'evaluation_mapping.json').read_text()):
        key=mapping['id'];folder=ROOT/'estimates'/key;result=json.loads((folder/'result.json').read_text());dst=OUT/key;dst.mkdir()
        with np.load(folder/'tracking.npz') as a:masks=a['masks'];prompt=a['prompt']
        with np.load(OLD/'vision_v3/estimates'/mapping['case_id']/'estimate.npz') as a:circles=a['sphere_circles_source']
        rgb=np.stack([np.array(Image.open(ROOT/'inputs'/key/f'rgb_{t:02d}.png')) for t in range(8)])
        gray=np.stack([cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb]).astype(float)
        change=np.abs(gray[7]-np.median(gray[:7],axis=0));heat=cv2.applyColorMap(np.uint8(np.clip(change/max(change.max(),1)*255,0,255)),cv2.COLORMAP_INFERNO)[:,:,::-1]
        Image.fromarray(heat).save(dst/'motion.png')
        frames=[];trans=crop_transform(masks.shape[1:]);h,w=masks.shape[1:]
        for t,mask in enumerate(masks):
            pm=resize_crop_mask(mask,trans);contours,_=cv2.findContours(pm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            areas=[float(cv2.contourArea(c)) for c in contours];circ=[float(4*np.pi*cv2.contourArea(c)/max(cv2.arcLength(c,True)**2,1)) for c in contours]
            reason='碎裂：外轮廓数 != 1' if len(contours)!=1 else ('球形检查：圆形度 < 0.65' if circ[0]<.65 else '通过二维检查')
            y,x=np.where(mask);box=[max(0,int(x.min())-18),max(0,int(y.min())-18),min(w,int(x.max())+19),min(h,int(y.max())+19)] if len(x) else [0,0,w,h]
            # Show exactly the processed-resolution mask used by the gate, inverse-mapped for display.
            up=cv2.resize(pm.astype(np.uint8),(w,trans['resized_hw'][0]*h//trans['resized_hw'][0]),interpolation=cv2.INTER_NEAREST)
            overlay=rgb[t].copy();overlay[mask]=(overlay[mask]*.6+np.array([255,60,100])*.4).astype(np.uint8)
            cs,_=cv2.findContours(up,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(overlay,cs,-1,(40,230,255),1)
            if t==7:
                b=prompt.astype(int);cv2.rectangle(overlay,tuple(b[:2]),tuple(b[2:]),(255,170,40),1)
            old=rgb[t].copy();cx,cy,r=circles[t];cv2.circle(old,(round(float(cx)),round(float(cy))),round(float(r)),(255,210,60),1)
            edge=cv2.Canny(cv2.cvtColor(rgb[t],cv2.COLOR_RGB2GRAY),60,140);edge_rgb=np.repeat(edge[:,:,None],3,axis=2)
            for name,img in [('new',overlay),('old',old),('edge',edge_rgb)]:Image.fromarray(img).save(dst/f'{name}_{t}.png')
            frames.append({'frame':t,'crop':box,'component_count':len(contours),'contour_areas_processed_px':areas,'circularities':circ,'gate':reason,'mask_area_source_px':int(mask.sum())})
        state=result['state'];reason=state.get('reason',str(state.get('reasons')))
        group='碎裂' if 'fragmented' in reason else ('球形检查' if 'non_sphere' in reason else '3D拟合')
        rows.append({**mapping,'group':group,'reason':reason,'frames':frames,'state':state,'target':result.get('target_detection'),
                     'first_2d_failure':next((f['frame'] for f in frames if f['gate']!='通过二维检查'),None)})
    dump_json(OUT/'data.json',rows)
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>36例失败定位</title>
<style>body{font:16px system-ui;background:#111827;color:#e5e7eb;max-width:1240px;margin:24px auto;padding:0 16px}a{color:#93c5fd}.controls{position:sticky;top:0;background:#182232;padding:12px;z-index:2}select,button,input{font:inherit;margin:4px}canvas{width:100%;background:#050b12}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.panel{background:#202b3c;padding:10px}pre{white-space:pre-wrap;max-height:350px;overflow:auto}td,th{padding:8px;border-bottom:1px solid #445}table{width:100%;border-collapse:collapse}.bad{color:#ff8794} @media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>36例：失败发生在哪里？</h1><p>14例mask碎裂 · 17例球形检查失败 · 5例3D拟合失败。这里只读取冻结结果，没有后处理或重新预测。</p>
<p>旧v3的黄色圆仅作历史对照，不是GT；灰度边缘仅供观察，不是新圆拟合结果。不能只凭拒绝码断言目标选错或全部由阴影造成。</p>
<div class="controls"><select id="filter"><option>全部</option><option>碎裂</option><option>球形检查</option><option>3D拟合</option></select><select id="case"></select><button id="first">跳到首个失败帧</button><button id="play">播放/暂停</button><input id="time" type="range" min="0" max="7" value="7"><b id="frame"></b></div>
<h2 id="title"></h2><p id="reason" class="bad"></p>
<div class="grid"><div class="panel">新SAM2：粉色mask；青色实际准入轮廓；橙色RGB7提示框<canvas id="full" width="640" height="360"></canvas></div><div class="panel">运动差分热图（固定RGB7 vs RGB0–6中值）<canvas id="motion" width="640" height="360"></canvas></div><div class="panel">旧v3 RGB边缘圆（仅历史对照）<canvas id="oldfull" width="640" height="360"></canvas></div></div>
<h3>相同区域放大</h3><div class="grid"><div class="panel">新mask与碎片<canvas id="zoom" width="360" height="300"></canvas></div><div class="panel">RGB边缘观察<canvas id="edges" width="360" height="300"></canvas></div><div class="panel">旧边缘圆<canvas id="oldzoom" width="360" height="300"></canvas></div></div>
<h3>八帧二维准入检查（按实际VGGT预处理分辨率计算）</h3><table><thead><tr><th>帧</th><th>分量</th><th>圆形度</th><th>结果</th></tr></thead><tbody id="metrics"></tbody></table>
<h3>三维拟合：只有5例有输出，其余未执行，不填零</h3><pre id="state"></pre><h3>运动候选框与评分</h3><pre id="target"></pre>
<p>半径未知联合拟合不使用0.11 m、不固定高度。已有3D候选也可能退化；FAIL不是成功预测。</p><p><a href="../">完整36例重跑页面</a> · <a href="/overlay_viewer_v3/">旧v3页面</a> · <a href="data.json">诊断数据</a></p>
<script>let data,rows,current,playing=false,serial=0;const sel=document.querySelector('#case'),time=document.querySelector('#time');
async function paint(id,url,crop,token){const im=new Image();im.src=url;await im.decode();if(token!==serial)return;let c=document.getElementById(id),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.imageSmoothingEnabled=false;if(crop){let [a,b,z,w]=crop;let s=Math.min(c.width/(z-a),c.height/(w-b));x.drawImage(im,a,b,z-a,w-b,(c.width-(z-a)*s)/2,(c.height-(w-b)*s)/2,(z-a)*s,(w-b)*s)}else x.drawImage(im,0,0,c.width,c.height)}
async function draw(){let token=++serial;current=rows[+sel.value];let r=current,t=+time.value,f=r.frames[t];document.querySelector('#title').textContent=r.case_id;document.querySelector('#reason').textContent=r.reason;document.querySelector('#frame').textContent='RGB'+t+' · '+f.gate;document.querySelector('#metrics').innerHTML=r.frames.map(f=>`<tr><td>RGB${f.frame}</td><td>${f.component_count}</td><td>${f.circularities.map(x=>x.toFixed(3)).join(', ')}</td><td>${f.gate}</td></tr>`).join('');document.querySelector('#state').textContent=JSON.stringify(r.state,null,2);document.querySelector('#target').textContent=JSON.stringify(r.target,null,2);await Promise.all([paint('full',r.id+'/new_'+t+'.png',null,token),paint('motion',r.id+'/motion.png',null,token),paint('oldfull',r.id+'/old_'+t+'.png',null,token),paint('zoom',r.id+'/new_'+t+'.png',f.crop,token),paint('edges',r.id+'/edge_'+t+'.png',f.crop,token),paint('oldzoom',r.id+'/old_'+t+'.png',f.crop,token)])}
function filter(){let value=document.querySelector('#filter').value;rows=data.filter(r=>value==='全部'||r.group===value);sel.innerHTML=rows.map((r,i)=>`<option value="${i}">${r.case_id} · ${r.group}</option>`).join('');draw()}
fetch('data.json').then(r=>r.json()).then(d=>{data=d;filter()});sel.onchange=draw;time.oninput=draw;document.querySelector('#filter').onchange=filter;document.querySelector('#first').onclick=()=>{time.value=current.first_2d_failure??7;draw()};document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing&&data){time.value=(+time.value+1)%8;draw()}},350);</script>''')
    print('built',len(rows),'cases',OUT)

if __name__=='__main__':main()
