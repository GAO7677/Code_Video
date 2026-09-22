"""Observation-only rollout first; post-freeze evaluation and all-case viewer second."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from context_rgb_pybullet_common import dump_json,sha256_file
from generic_bullet_input import run

def rollout(root):
    gravity=root/'lower_plane_gravity_v1'
    for n,h in json.loads((gravity/'freeze.json').read_text()).items():
        if sha256_file(gravity/n)!=h:raise ValueError('gravity freeze mismatch')
    for n,h in json.loads((root/'estimate_freeze.json').read_text()).items():
        if sha256_file(root/n)!=h:raise ValueError('estimate freeze mismatch')
    out=root/'rollouts';out.mkdir(exist_ok=False);physics=json.loads((root/'protocol.json').read_text())['physics']
    for r in json.loads((gravity/'candidates.json').read_text()):
        folder=root/'estimates'/r['id'];est=json.loads((folder/'result.json').read_text());state=est['state']
        if state['status']!='ESTIMATED':result={'status':'FAIL','reason':est.get('reason',state.get('reason',state.get('reasons','state_not_run'))),'step_calls':0}
        elif not (folder/'collision_primitive.json').exists():result={'status':'FAIL','reason':'geometry_unavailable','step_calls':0}
        else:result=run(state,[json.loads((folder/'collision_primitive.json').read_text())],physics,r['gravity'])
        dump_json(out/(r['id']+'.json'),result)
    dump_json(root/'rollout_freeze.json',{str(p.relative_to(root)):sha256_file(p) for p in sorted(out.glob('*.json'))})

def evaluate(root):
    for n,h in json.loads((root/'rollout_freeze.json').read_text()).items():
        if sha256_file(root/n)!=h:raise ValueError('rollout freeze mismatch')
    base=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
    gtroot=Path('/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5/samples')
    gravity={r['id']:r for r in json.loads((root/'lower_plane_gravity_v1/candidates.json').read_text())}
    records=[]
    for m in json.loads((root/'evaluation_mapping.json').read_text()):
        key=m['id'];case=m['case_id'];folder=root/'estimates'/key
        est=json.loads((folder/'result.json').read_text());roll=json.loads((root/'rollouts'/f'{key}.json').read_text());g=gravity[key]['gravity']
        calibration=json.loads((base/'vision_inputs'/case/'calibration.json').read_text());E=np.array(calibration['world_to_camera_3x4']);K=np.array(calibration['intrinsic_K'])
        with np.load(gtroot/case/'raw/states_xyzw.npz') as a:
            names=list(a['object_names'].astype(str));assert names.count('pilot_ball')==1;i=names.index('pilot_ball');gt=a['positions'][:49,i];v=a['linear_velocities'][7,i]
        with np.load(folder/'vggt_raw.npz') as a:e=a['extrinsic'][0];k=a['intrinsic']
        scale=est['scale'];gtcam=gt@E[:,:3].T+E[:,3]
        def project(x):
            p=x@K.T;return (p[:,:2]/p[:,2,None]).tolist()
        row={**m,'state_status':est['state']['status'],'gravity_status':g['status'],'rollout_status':roll['status'],
             'failure':roll.get('reason'),'geometry_status':est['geometry']['status'],'state_metrics':None,'trajectory_metrics':None,
             'gravity_error_deg':None,'initial_penetration_m':roll.get('initial_penetration_m'),
             'initial_contacts':roll.get('initial_contact_count'),'gt_future_uv':project(gtcam[8:49]),'new_future_uv':None,'old_future_uv':None}
        if g['status']=='ESTIMATED':
            down=e[:,:3]@np.array(g['vector'])/9.81;true=E[:,:3]@np.array([0,0,-1]);row['gravity_error_deg']=float(np.degrees(np.arccos(np.clip(down@true,-1,1))))
        state=est['state']
        if 'radius' in state:
            p=e[:,:3]@state['p7']+e[:,3]*scale;ve=e[:,:3]@state['v7']
            row['state_metrics']={'p7_error_m':float(np.linalg.norm(p-gtcam[7])),'v7_error_mps':float(np.linalg.norm(ve-E[:,:3]@v)),
                                  'radius_m':state['radius'],'radius_error_m':abs(state['radius']-.11),'admitted':state['status']=='ESTIMATED'}
        if roll['status']=='EXECUTED':
            pred=np.array(roll['positions'])@e[:,:3].T+e[:,3]*scale;err=np.linalg.norm(pred-gtcam[8:49],axis=1)
            row['trajectory_metrics']={'ADE_m':float(err.mean()),'FDE_m':float(err[-1]),'contact_frames':sum(bool(c) for c in roll['contacts']),
                                       'contact_accuracy':'NOT_EVALUATED','step_calls':roll['step_calls']};row['new_future_uv']=project(pred)
        old=base/'evaluation_v2/rollouts'/case/'D_estimated_state_estimated_geometry__omega_gt_observed.npz'
        if old.exists():
            with np.load(old) as a:op=a['positions']
            row['old_future_uv']=project(op@E[:,:3].T+E[:,3]);row['old_D_ADE_m']=float(np.linalg.norm(op-gt[8:49],axis=1).mean())
        # Store observed estimated boundary for context overlay; GT camera is NOT used here.
        row['mask_contours']=None
        track=folder/'tracking.npz'
        if track.exists():
            import cv2
            with np.load(track) as a:masks=a['masks']
            row['mask_contours']=[[c[:,0,:].tolist() for c in cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]] for mask in masks]
        records.append(row)
    summary={'cases':len(records),'state_counts':dict(Counter(r['state_status'] for r in records)),
             'gravity_counts':dict(Counter(r['gravity_status'] for r in records)),'rollout_counts':dict(Counter(r['rollout_status'] for r in records)),
             'failure_counts':dict(Counter(str(r['failure']) for r in records if r['failure']))}
    dump_json(root/'viewer_data.json',{'summary':summary,'records':records});print(json.dumps(summary,indent=2))
    (root/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>36-case full generic rerun</title>
<style>body{font:16px system-ui;max-width:1100px;margin:25px auto;background:#111827;color:#e5e7eb}a{color:#93c5fd}canvas{max-width:100%;width:960px}select,button{font:inherit;padding:6px}pre{white-space:pre-wrap}td,th{padding:7px;border-bottom:1px solid #445}table{border-collapse:collapse;width:100%}</style>
<h1>旧v3 36例 · 完整新逻辑重跑</h1><p>运动候选→SAM2→未知半径球拟合→固定尺度5.819487→观测mesh→下方平面重力prior→zero-omega Bullet。旧v3未覆盖。</p>
<p>绿色GT仅评测；蓝色新zero-omega轨迹；橙色旧D（GT omega，不是公平独立因果对照）。失败分支不显示伪造轨迹。未来线叠加在RGB7静态背景上，不是未来视频。</p>
<pre id="summary"></pre><select id="case"></select><input id="frame" type="range" min="0" max="7" value="7"><button id="play">播放context</button><canvas id="canvas" width="640" height="360"></canvas><pre id="detail"></pre><table><thead><tr><th>case</th><th>state</th><th>gravity</th><th>rollout</th><th>reason</th></tr></thead><tbody id="rows"></tbody></table><p><a href="viewer_data.json">逐例完整指标</a> · <a href="report.md">报告</a></p>
<script>let data,playing=false,serial=0;const select=document.querySelector('#case'),slider=document.querySelector('#frame'),canvas=document.querySelector('#canvas'),ctx=canvas.getContext('2d');function line(points,color){if(!points)return;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke()}
async function draw(){let token=++serial,r=data.records[+select.value],t=+slider.value,im=new Image();im.src='inputs/'+r.id+'/rgb_'+String(t).padStart(2,'0')+'.png';await im.decode();if(token!==serial)return;ctx.drawImage(im,0,0,640,360);if(r.mask_contours)r.mask_contours[t].forEach(c=>line(c,'#ff6688'));if(t===7){line(r.old_future_uv,'#ffb74d');line(r.gt_future_uv,'#44dd88');line(r.new_future_uv,'#55aaff')}document.querySelector('#detail').textContent=JSON.stringify(r,(k,v)=>k.endsWith('_uv')||k==='mask_contours'?undefined:v,2)}
fetch('viewer_data.json').then(r=>r.json()).then(d=>{data=d;document.querySelector('#summary').textContent=JSON.stringify(d.summary,null,2);select.innerHTML=d.records.map((r,i)=>`<option value="${i}">${r.case_id} · ${r.rollout_status}</option>`).join('');document.querySelector('#rows').innerHTML=d.records.map(r=>`<tr><td>${r.case_id}</td><td>${r.state_status}</td><td>${r.gravity_status}</td><td>${r.rollout_status}</td><td>${JSON.stringify(r.failure)}</td></tr>`).join('');draw()});select.onchange=draw;slider.oninput=draw;document.querySelector('#play').onclick=()=>playing=!playing;setInterval(()=>{if(playing&&data){slider.value=(+slider.value+1)%8;draw()}},300);</script>''')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['rollout','evaluate']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();(rollout if a.phase=='rollout' else evaluate)(a.root)
