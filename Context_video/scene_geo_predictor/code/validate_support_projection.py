"""CPU-only frozen-input support repair evaluation; GT is scoring only."""
import json,copy
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(2)
import evaluate_context_rgb_pybullet as ev
from fit_context_collision_primitives import fit_case
from build_context_overlay_viewer import project_points,primitive_wireframe

ROOT=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
OUT=ROOT/'support_projection_fix_20260922_v1'
def read(p):return json.loads(p.read_text())
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2))

def main():
    OUT.mkdir(exist_ok=False)
    recheck,pilot,generator=ev.load_engine()
    records={r['key']:r for r in read(ev.DEFAULT_DATA/'pilot_manifest.json')['records']}
    selected=[r for r in read(ROOT/'vision_inputs/manifest.json')['records'] if r['family']=='support_edge']
    write(OUT/'protocol.json',dict(case_count=12,input='frozen context estimates and calibration; no GT for fitting',
        change='top pixel rays intersect contact plane before footprint fitting',
        gates='1 mm penetration; required upward left support; no horizontal shift or expansion',
        modes=['strict_C','C_contact_aligned','strict_D'],omega='GT diagnostic',gpu=False))
    # Complete and freeze all fits before reading evaluation targets.
    for row in selected:fit_case(ROOT/'vision_v3',row,OUT/'primitives',support_ray_projection=True)
    results=[]
    for row in selected:
        cid=row['case_id'];case,seed=recheck.reconstruct_case(pilot,generator,records[cid])
        raw=recheck.load_raw_state(ev.DEFAULT_DATA/'samples'/cid);i=raw['dynamic_index']
        frozen=read(ROOT/'evaluation_v2/cases'/cid/'evaluation.json')
        payload=read(OUT/'primitives/cases'/cid/'primitives.json')
        geometry=ev.geometry_errors(case.blueprint,payload)
        result=dict(case_id=cid,geometry_before=frozen['geometry_evaluation']['measurements'],geometry_after=geometry['measurements'],modes={})
        cal=read(ROOT/'vision_inputs'/cid/'calibration.json');base=cv2.imread(str(ROOT/'vision_inputs'/cid/'rgb_07.png'))
        for prim in payload['primitives']:
            if 'platform' not in prim['name']:continue
            for seg in primitive_wireframe(prim,cal,estimated=True):cv2.polylines(base,[np.rint(seg).astype(np.int32)],False,(255,60,230),1)
        for letter,policy in [('C','strict'),('C','support_aligned'),('D','strict')]:
            mode=letter+'_'+policy
            old=next(v for k,v in frozen['modes'].items() if k.startswith(letter+'_') and k.endswith('gt_observed'))
            state=dict(position=np.asarray(old['initial_position_m']),quaternion=np.asarray(raw['quats'][7,i]),linear_velocity=np.asarray(old['initial_velocity_mps']),angular_velocity=np.asarray(old['initial_angular_velocity_radps']))
            try:
                pred=ev.rollout(generator,case,seed,'estimated',payload['primitives'],state,initialization_policy=policy,confirmed_support_names=('estimated_left_platform',))
                metrics=ev.trajectory_metrics(pred,raw['positions'][8:49,i],raw['linear_velocities'][8:49,i])
                metrics['initialization']=pred['initialization'];ev.save_rollout(OUT/'rollouts'/cid/(mode+'.npz'),pred)
                uv,_=project_points(np.vstack([pred['initialization']['after_position_m'],pred['positions']]),cal)
                color=(40,180,255) if letter=='C' else (40,40,255)
                cv2.polylines(base,[np.rint(uv).astype(np.int32)],False,color,2)
            except ev.InitialContactError as error:
                metrics=dict(status='BLOCKED',initialization=error.audit,ADE_m=None,FDE_m=None,api_step_calls=0)
            metrics['old_ADE_m']=old['ADE_m'];metrics['old_event']=old['event']
            result['modes'][mode]=metrics
        cv2.imwrite(str(OUT/(cid+'.jpg')),base)
        write(OUT/'cases'/f'{cid}.json',result);results.append(result)
        print(cid,{k:(v.get('status','PASS'),v['ADE_m']) for k,v in result['modes'].items()},flush=True)
    write(OUT/'results.json',results)
    rows=''.join('<tr><td>'+r['case_id']+'</td>'+''.join('<td>'+('BLOCKED: '+v['initialization']['reason'] if v['ADE_m'] is None else f"ADE {v['ADE_m']:.4f} m (old {v['old_ADE_m']:.4f})")+'</td>' for v in r['modes'].values())+'</tr>' for r in results)
    images=''.join('<h3>'+r['case_id']+'</h3><img src="'+r['case_id']+'.jpg"><p><a href="cases/'+r['case_id']+'.json">初始化、接触事件与几何误差</a></p>' for r in results)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px/1.6 sans-serif;max-width:1200px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}img{width:640px;max-width:100%}</style><h1>2026-09-22 平台支撑修复验证</h1><p>12 个平台 case，CPU。GT 仅评测及 omega 对照；未替换 VGGT、未使用 GT 平台拟合。紫框：新估计平台；橙线：C 显式接触对齐；红线：D。轨迹背景固定 RGB7，非未来视频。BLOCKED 无轨迹，不移动球去填补横向缺失支撑。</p><table><tr><th>Case</th><th>严格 C</th><th>C 接触对齐</th><th>严格 D</th></tr>'+rows+'</table>'+images)

if __name__=='__main__':main()

