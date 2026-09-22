"""Evaluation-only silhouette diagnosis and lightweight all-case context viewer."""
import argparse
import json
from pathlib import Path
import numpy as np
from skimage.measure import find_contours
from context_rgb_pybullet_common import dump_json, sha256_file
from run_sphere_state_gate import ROOT

def silhouette(center,k,radius=.11):
    axis=center/np.linalg.norm(center)
    u=np.cross(axis,[0,1,0]); u/=np.linalg.norm(u); v=np.cross(axis,u)
    a=np.arcsin(radius/np.linalg.norm(center)); theta=np.linspace(0,2*np.pi,181)
    rays=np.cos(a)*axis+np.sin(a)*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
    pixels=rays@k.T
    return (pixels[:,:2]/pixels[:,2,None]).tolist()

def run(output,gt_root):
    for name,digest in json.loads((output/'freeze.json').read_text()).items():
        if sha256_file(output/name)!=digest: raise ValueError('freeze mismatch')
    manifest=json.loads((gt_root/'pilot_manifest.json').read_text())
    evaluation=json.loads((output/'evaluation.json').read_text())
    rows=[]
    for record in manifest['records']:
        key=record['key']; est=json.loads((output/'estimates'/f'{key}.json').read_text())
        if est['status']!='EXECUTED': continue
        calibration=json.loads((ROOT/'vision_inputs'/key/'calibration.json').read_text())
        k=np.array(calibration['intrinsic_K']); e=np.array(calibration['world_to_camera_3x4'])
        with np.load(Path(record['sample_dir'])/'raw/states_xyzw.npz',allow_pickle=False) as data:
            names=list(data['object_names'].astype(str)); assert names.count('pilot_ball')==1
            i=names.index('pilot_ball'); gt=data['positions'][:8,i]; gv=data['linear_velocities'][7,i]
        with np.load(ROOT/'vision_v1/sam2_masks'/f'{key}.npz',allow_pickle=False) as data: masks=data['masks']
        with np.load(ROOT/'vision_v3/estimates'/key/'estimate.npz',allow_pickle=False) as old:
            old_p=old['estimated_p7']; old_v=old['estimated_v7']; old_circles=old['sphere_circles_source']
        frames=[]
        for t,mask in enumerate(masks):
            contours=[c[:,::-1] for c in find_contours(mask.astype(float),.5)]
            pixels=np.concatenate(contours)
            rays=np.column_stack((pixels,np.ones(len(pixels))))@np.linalg.inv(k).T
            rays/=np.linalg.norm(rays,axis=1)[:,None]
            true_camera=e[:,:3]@gt[t]+e[:,3]
            estimated_camera=e[:,:3]@np.array(est['centers'][t])+e[:,3]
            distance=np.linalg.norm(np.cross(rays,true_camera),axis=1)-.11
            frames.append({'rgb':f'../vision_inputs/{key}/rgb_{t:02d}.png',
                'mask_contours':[c.tolist() for c in contours], 'gt_silhouette_eval_only':silhouette(true_camera,k),
                'estimated_silhouette':silhouette(estimated_camera,k), 'old_circle':old_circles[t].tolist(),
                'gt_surface_residual_median_m':float(np.median(distance)),
                'gt_surface_residual_abs_median_m':float(np.median(np.abs(distance))),
                'fit':est['frame_fit'][t]})
        rows.append({'case_id':key,'family':record['family'],'group_id':record['group_id'],'frames':frames,
            'old_v3_p7_error_m':float(np.linalg.norm(old_p-gt[7])),
            'old_v3_v7_error_mps':float(np.linalg.norm(old_v-gv)),
            'new':next(r for r in evaluation['records'] if r['case_id']==key)['methods']['sphere/robust_linear']})
    payload={'rows':rows,'summary':evaluation['summary'],'families':evaluation['families'],
             'old_v3_p7_median_m':float(np.median([r['old_v3_p7_error_m'] for r in rows])),
             'old_v3_v7_median_mps':float(np.median([r['old_v3_v7_error_mps'] for r in rows]))}
    dump_json(output/'diagnosis.json',payload)
    html='''<!doctype html><meta charset="utf-8"><title>Sphere state gate — FAIL</title>
<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;background:#111827;color:#e5e7eb}select,button{font:inherit;padding:8px}canvas{width:960px;max-width:100%;image-rendering:pixelated}pre{white-space:pre-wrap}a{color:#93c5fd}</style>
<h1>P1 状态恢复：FAIL / 36 cases · 12 history groups</h1>
<p>自动 RGB 提示沿用旧协议。仅 CPU 使用冻结 SAM2 masks；不使用水平支撑约束。估计冻结后才读取 GT。</p>
<p>主方法：完整 mask 射线球拟合 + 鲁棒线性运动。p7 中位 0.423 m / p90 0.789 m；v7 向量中位 0.290 m/s；方向中位 7.12°。P2–P6 NOT_RUN，旧页面未替换。</p>
<p>轮廓：<span style="color:#fb7185">粉色 SAM2</span> / <span style="color:#34d399">绿色 GT 球投影（仅评测）</span> / <span style="color:#60a5fa">蓝色新拟合球</span> / <span style="color:#facc15">黄色旧 RGB 边缘圆</span>。放大显示，不代表新增像素精度。</p>
<select id="cases"></select> <button id="play">播放/暂停</button> <input id="time" type="range" min="0" max="7" value="7"><span id="frame"></span>
<canvas id="view" width="640" height="360"></canvas><pre id="details"></pre>
<p><a href="report.md">报告</a> · <a href="evaluation.json">逐例/分组评测</a> · <a href="diagnosis.json">轮廓诊断</a> · <a href="freeze.json">冻结 hashes</a></p>
<script>
let data,playing=false,serial=0;const sel=document.querySelector('#cases'),slider=document.querySelector('#time'),canvas=document.querySelector('#view'),ctx=canvas.getContext('2d');
function line(points,color){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=.7;ctx.stroke()}
async function draw(){let ticket=++serial,r=data.rows[+sel.value],t=+slider.value,f=r.frames[t],im=new Image();im.src=f.rgb;await im.decode();if(ticket!==serial)return;ctx.drawImage(im,0,0);f.mask_contours.forEach(p=>line(p,'#fb7185'));line(f.gt_silhouette_eval_only,'#34d399');line(f.estimated_silhouette,'#60a5fa');let [x,y,rad]=f.old_circle;ctx.beginPath();ctx.arc(x,y,rad,0,2*Math.PI);ctx.strokeStyle='#facc15';ctx.stroke();document.querySelector('#frame').textContent='RGB'+t;document.querySelector('#details').textContent=JSON.stringify({case:r.case_id,old_v3_p7_error_m:r.old_v3_p7_error_m,new:r.new,frame_fit:f.fit,gt_surface_residual_median_m:f.gt_surface_residual_median_m},(k,v)=>k==='residuals_m'?undefined:v,2)}
fetch('diagnosis.json').then(r=>r.json()).then(d=>{data=d;sel.innerHTML=d.rows.map((r,i)=>`<option value="${i}">${r.case_id}</option>`).join('');draw()});sel.onchange=draw;slider.oninput=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing&&data){slider.value=(+slider.value+1)%8;draw()}},300);
</script>'''
    (output/'index.html').write_text(html)
    print(json.dumps({k:v for k,v in payload.items() if k.startswith('old_')},indent=2))
    for key in ['phase1_support_edge_g03_v0620','phase1_aperture_g01_v0460']:
        r=next(r for r in rows if r['case_id']==key)
        print(key, json.dumps({'new':r['new'],'old_p7':r['old_v3_p7_error_m'],
              'rgb7_contour_gt_residual_m':r['frames'][7]['gt_surface_residual_median_m']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--gt-root',type=Path,required=True)
    a=p.parse_args();run(a.output,a.gt_root)
