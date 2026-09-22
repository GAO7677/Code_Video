"""Rerun all 144 displayed GT-omega arms; compare with frozen viewer v2."""
import copy,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
import pybullet as p
import evaluate_context_rgb_pybullet as ev
from build_context_overlay_viewer import project_points,json_points

ROOT=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
OUT=ROOT/'all36_recheck_20260922_v1'
VIEW=ROOT/'overlay_viewer_v3'
def read(path):return json.loads(path.read_text())
def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def run():
    OUT.mkdir(exist_ok=False);VIEW.mkdir(exist_ok=False)
    source=ROOT/'overlay_viewer_v2';shutil.copytree(source/'data',VIEW/'data');(VIEW/'assets').symlink_to(ROOT/'overlay_viewer_v1/assets',target_is_directory=True)
    for f in ['index.html','styles.css','app.js']:shutil.copy2(source/f,VIEW/f)
    protocol=dict(cases=36,displayed_arms=144,omega='GT observed, identical to viewer v2',geometry='frozen primitives_v3; experimental ray projection OFF',
                  inference='frozen vision_v3; no GPU/model rerun',strict_C_also_checked=True,comparison_tolerance_m=1e-6,
                  valid_means='initialization admitted, not guaranteed accurate trajectory',cpu_threads=2)
    write(OUT/'protocol.json',protocol)
    recheck,pilot,generator=ev.load_engine();records={r['key']:r for r in read(ev.DEFAULT_DATA/'pilot_manifest.json')['records']}
    manifest=read(VIEW/'data/manifest.json');allrows=[]
    for idx,record in enumerate(manifest['records'],1):
        cid=record['case_id'];case,seed=recheck.reconstruct_case(pilot,generator,records[cid]);raw=recheck.load_raw_state(ev.DEFAULT_DATA/'samples'/cid);di=raw['dynamic_index']
        frozen=read(ROOT/'evaluation_v2/cases'/cid/'evaluation.json');payload=read(source/'data/cases'/f'{cid}.json');cal=read(ROOT/'vision_inputs'/cid/'calibration.json')
        primitives=read(ROOT/'primitives_v3/cases'/cid/'primitives.json')['primitives'];rows=[]
        for letter in 'ABCD':
            name=next(k for k in frozen['modes'] if k.startswith(letter+'_') and k.endswith('gt_observed'));old=frozen['modes'][name];previous=copy.deepcopy(payload['rollout'][letter])
            state=dict(position=np.array(old['initial_position_m']),quaternion=raw['quats'][7,di],linear_velocity=np.array(old['initial_velocity_mps']),angular_velocity=np.array(old['initial_angular_velocity_radps']))
            geometry='gt' if letter in 'AB' else 'estimated'
            # v2 used explicit support alignment only on the 27 initially
            # overlapping C cases. Reproduce that exact policy, no new fallback.
            policy=previous.get('initialization',{}).get('policy','strict')
            supports=()
            if policy=='support_aligned':
                supports=tuple(previous['initialization']['confirmed_support_names'])
            calls=[0];actual=p.stepSimulation
            def counted(*a,**kw):calls[0]+=1;return actual(*a,**kw)
            p.stepSimulation=counted
            strict_audit=None
            try:
                if letter=='C' and policy!='strict':
                    try:ev.rollout(generator,case,seed,geometry,primitives,state)
                    except ev.InitialContactError as err:strict_audit=err.audit
                    assert calls[0]==0
                try:
                    pred=ev.rollout(generator,case,seed,geometry,primitives,state,initialization_policy=policy,confirmed_support_names=supports)
                    assert calls[0]==328
                    metrics=ev.trajectory_metrics(pred,raw['positions'][8:49,di],raw['linear_velocities'][8:49,di]);metrics['initialization']=pred['initialization']
                    ev.save_rollout(OUT/'rollouts'/cid/(name+'.npz'),pred)
                except ev.InitialContactError as err:
                    assert calls[0]==0
                    pred=None;metrics=dict(trajectory_metric_status='BLOCKED_INITIAL_CONTACT',ADE_m=None,FDE_m=None,initialization=err.audit,api_step_calls=0)
            finally:p.stepSimulation=actual
            change='';difference=None
            if pred is None:change='still_blocked' if previous['status'].startswith('BLOCKED') else 'newly_rejected'
            elif previous['status'].startswith('BLOCKED'):change='newly_admitted'
            else:
                oldpath=ROOT/'evaluation_v2/rollouts'/cid/(name+'.npz')
                if previous.get('initialization',{}).get('status')=='CONTACT_ALIGNED':oldpath=ROOT/'contact_alignment_v1/rollouts'/cid/(name+'__contact_aligned.npz')
                reference=np.load(oldpath)['positions'].astype(float);difference=float(np.max(np.linalg.norm(pred['positions']-reference,axis=1)))
                change='unchanged' if difference<=1e-6 else 'trajectory_changed'
            row=dict(case_id=cid,arm=letter,policy=policy,old_status=previous['status'],old_ADE_m=previous['ADE_m'],change=change,max_position_difference_m=difference,
                     new_metrics=metrics,actual_step_calls=calls[0],strict_admission=strict_audit)
            rows.append(row);allrows.append(row)
            v=payload['rollout'][letter];v.update(status=metrics['trajectory_metric_status'],initialization=metrics['initialization'],regression=row['change'],ADE_m=metrics['ADE_m'],FDE_m=metrics['FDE_m'])
            if pred is None:
                v.update(points_px=[],radius_px=[],position_error_m=[],contact_outcome='NOT_RUN',contact_match=None,first_contact_time_s=None,first_contact_path_index=None,max_penetration_m=None,post_contact_velocity_error={'status':'NOT_RUN'},included_in_valid_aggregate=False)
            else:
                xyz=np.vstack([pred['initialization']['after_position_m'],pred['positions']]);uv,z=project_points(xyz,cal);event=pred['event'];ei=event['first_interaction_future_index_30hz']
                v.update(points_px=json_points(uv),radius_px=np.clip(float(cal['intrinsic_K'][0][0])*.11/np.maximum(z,1e-5),2,80).tolist(),position_error_m=metrics['position_error_by_future_frame_m'],contact_outcome=event['category'],contact_match=event['category']==frozen['modes']['A_gt_state_gt_geometry__omega_gt_observed']['event']['category'],first_contact_time_s=event['first_interaction_time_after_rgb7_s'],first_contact_path_index=None if ei is None else ei+1,max_penetration_m=event['max_penetration_m'])
        write(OUT/'cases'/f'{cid}.json',rows);write(VIEW/'data/cases'/f'{cid}.json',payload)
        print(idx,cid,[(r['arm'],r['change']) for r in rows],flush=True)
    summary=dict(status='EXECUTED',protocol=protocol,changes=dict(Counter(r['change'] for r in allrows)),by_arm={a:dict(Counter(r['change'] for r in allrows if r['arm']==a)) for a in 'ABCD'},newly_rejected=[dict(case_id=r['case_id'],arm=r['arm'],reason=r['new_metrics']['initialization']['reason']) for r in allrows if r['change']=='newly_rejected'],max_unchanged_delta_m=max(r['max_position_difference_m'] or 0 for r in allrows if r['change']=='unchanged'))
    write(OUT/'summary.json',summary);write(OUT/'all_modes.json',allrows)
    ds=[r for r in allrows if r['arm']=='D' and r['new_metrics']['ADE_m'] is not None]
    manifest['summary']['D_ADE_m']=float(np.mean([r['new_metrics']['ADE_m'] for r in ds]));manifest['summary']['D_FDE_m']=float(np.mean([r['new_metrics']['FDE_m'] for r in ds]));manifest['rerun_summary']=summary
    write(VIEW/'data/manifest.json',manifest)
    banner='<div style="padding:12px;background:#fff2c4">36 case 已用新准入代码重跑；几何实验关闭。<a href="../all36_recheck_20260922_v1/">回归审计</a> · <a href="../overlay_viewer_v2/">旧版 v2</a></div>'
    html=(VIEW/'index.html').read_text().replace('<body>','<body>'+banner);(VIEW/'index.html').write_text(html)
    table=''.join('<tr><td>'+r['case_id']+'</td><td>'+r['arm']+'</td><td>'+r['change']+'</td><td>'+str(r['old_ADE_m'])+'</td><td>'+str(r['new_metrics']['ADE_m'])+'</td><td>'+r['new_metrics']['initialization'].get('reason','')+'</td></tr>' for r in allrows)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:14px/1.6 sans-serif;padding:24px}td,th{border:1px solid #ccc;padding:6px}table{border-collapse:collapse}</style><h1>36 case × A/B/C/D 新代码回归审计</h1><p>GT omega；VGGT 输入及原几何冻结；C 延续 v2 显式对齐策略。unchanged 是轨迹数值未变，不代表预测正确。newly_rejected 是新规则拦截初始化错误，不代表已恢复几何。</p><a href="../overlay_viewer_v3/">打开全部新轨迹</a><pre>'+json.dumps(summary,indent=2,ensure_ascii=False)+'</pre><table><tr><th>case</th><th>arm</th><th>change</th><th>old ADE</th><th>new ADE</th><th>reason</th></tr>'+table+'</table>')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
