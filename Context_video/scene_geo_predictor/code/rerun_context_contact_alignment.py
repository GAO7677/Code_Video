"""Re-run only frozen pilot modes with invalid initial overlap; keep legacy immutable."""
from __future__ import annotations
import argparse
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import pybullet as p
import evaluate_context_rgb_pybullet as ev
from build_context_overlay_viewer import project_points, json_points


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def run(root, output, viewer):
    if output.exists() or viewer.exists():
        raise FileExistsError('Refusing to overwrite versioned output')
    output.mkdir(parents=True)
    viewer.mkdir(parents=True)
    source=root/'overlay_viewer_v1'
    shutil.copytree(source/'data',viewer/'data')
    (viewer/'assets').symlink_to(source/'assets',target_is_directory=True)
    web=Path(__file__).resolve().parents[1]/'web/context_overlay_viewer'
    for name in ('index.html','styles.css','app.js'):
        shutil.copy2(web/name,viewer/name)
    manifest=read(viewer/'data/manifest.json')
    manifest.update(initialization_protocol='strict_or_explicit_support_aligned_v1',
                    legacy_viewer='../overlay_viewer_v1/',default_case_id='phase1_aperture_g01_v0460')
    write(output/'protocol.json',dict(initial_penetration_tolerance_m=.001,
          max_alignment_m=.055,contact_slop_m=.00001,policy='upward_only_confirmed_horizontal_support',
          unchanged=['horizontal_position','quaternion','linear_velocity','angular_velocity','geometry','solver','target'],
          pre_roll_steps=0,observed_support_source='frozen A oracle RGB7 contact semantics; diagnostic C only',
          aggregation='aligned C is separate from strict C; D headline remains frozen original',
          selection='frozen evaluation_v2 initial penetration > 1 mm; all 3 omega policies',gpu_used=False))
    recheck,pilot,generator=ev.load_engine()
    records={r['key']:r for r in read(ev.DEFAULT_DATA/'pilot_manifest.json')['records']}
    runs=[]
    affected=[]
    for record in manifest['records']:
        cid=record['case_id']
        frozen=read(root/'evaluation_v2/cases'/cid/'evaluation.json')
        selected={k:m for k,m in frozen['modes'].items() if m['event']['initial_penetration_m']>.001}
        if not selected:
            continue
        affected.append(cid)
        case,seed=recheck.reconstruct_case(pilot,generator,records[cid])
        raw=recheck.load_raw_state(ev.DEFAULT_DATA/'samples'/cid)
        di=raw['dynamic_index']
        target=np.asarray(raw['positions'][8:49,di])
        target_velocity=np.asarray(raw['linear_velocities'][8:49,di])
        primitives=read(root/'primitives_v3/cases'/cid/'primitives.json')['primitives']
        oracle=frozen['modes']['A_gt_state_gt_geometry__omega_gt_observed']
        semantics=set(oracle['event']['initial_contact_semantics']) & {'ground','left_platform'}
        support_names=tuple(x['name'] for x in primitives if ev.semantic_body(x['name'],frozen['family']) in semantics)
        payload=read(viewer/'data/cases'/f'{cid}.json')
        calibration=read(root/'vision_inputs'/cid/'calibration.json')
        for name,old in selected.items():
            # This pilot's selected rows are C. Do not accidentally grant oracle
            # support evidence to estimated-state B/D if future inputs change.
            if not name.startswith('C_gt_state_estimated_geometry__'):
                raise ValueError(f'Unreviewed mode requiring initialization repair: {name}')
            state=dict(position=np.asarray(old['initial_position_m']),
                       quaternion=np.asarray(raw['quats'][7,di]),
                       linear_velocity=np.asarray(old['initial_velocity_mps']),
                       angular_velocity=np.asarray(old['initial_angular_velocity_radps']))
            calls=[0]
            actual_step=p.stepSimulation
            def counted_step(*args,**kwargs):
                calls[0]+=1
                return actual_step(*args,**kwargs)
            p.stepSimulation=counted_step
            try:
                try:
                    ev.rollout(generator,case,seed,'estimated',primitives,state)
                    raise AssertionError('Strict overlap gate did not reject')
                except ev.InitialContactError as error:
                    strict=error.audit
                assert calls[0]==0, calls
                try:
                    result=ev.rollout(generator,case,seed,'estimated',primitives,state,
                                      initialization_policy='support_aligned',confirmed_support_names=support_names)
                    assert calls[0]==41*8, calls
                    audit=result['initialization']
                    metrics=ev.trajectory_metrics(result,target,target_velocity)
                    ev.save_rollout(output/'rollouts'/cid/(name+'__contact_aligned.npz'),result)
                    event_index=oracle['event']['first_interaction_future_index_30hz']
                    if event_index is None:
                        post={'status':'NOT_APPLICABLE'}
                    else:
                        errors=np.linalg.norm(result['linear_velocities'][event_index:]-target_velocity[event_index:],axis=1)
                        post=dict(status='EXECUTED',mean_mps=float(errors.mean()),max_mps=float(errors.max()),by_frame_mps=errors.tolist())
                    metrics.update(initialization=audit,post_contact_velocity_error=post,
                         contact_outcome_match_oracle=result['event']['category']==oracle['event']['category'],
                         protocol='contact_aligned_diagnostic',included_in_strict_C_aggregate=False)
                except ev.InitialContactError as error:
                    assert calls[0]==0
                    audit=error.audit
                    result=None
                    metrics=dict(trajectory_metric_status='BLOCKED_INITIAL_CONTACT',initialization=audit,
                                 ADE_m=None,FDE_m=None,api_step_calls=0,post_contact_velocity_error={'status':'NOT_RUN'})
            finally:
                p.stepSimulation=actual_step
            row=dict(case_id=cid,original_mode=name,strict_admission=strict,metrics=metrics,
                     actual_api_step_calls=calls[0],legacy_ADE_m=old['ADE_m'],legacy_FDE_m=old['FDE_m'])
            write(output/'cases'/cid/(name+'.json'),row)
            runs.append(row)
            if not name.endswith('gt_observed'):
                continue
            view=payload['rollout']['C']
            view.update(initialization=audit,strict_admission=strict,legacy_metrics=dict(ADE_m=old['ADE_m'],FDE_m=old['FDE_m']),
                        protocol='contact_aligned_diagnostic',label='C · Contact-aligned geometry diagnostic',
                        status=metrics['trajectory_metric_status'],included_in_valid_aggregate=False)
            if result is None:
                view.update(points_px=[],radius_px=[],position_error_m=[],ADE_m=None,FDE_m=None,
                            contact_outcome='NOT_RUN',contact_match=None,first_contact_time_s=None,
                            first_contact_path_index=None,max_penetration_m=None,post_contact_velocity_error={'status':'NOT_RUN'})
            else:
                positions=np.vstack([audit['after_position_m'],result['positions']])
                uv,depth=project_points(positions,calibration)
                radius=np.clip(float(calibration['intrinsic_K'][0][0])*.11/np.maximum(depth,1e-5),2,80)
                e=result['event']
                index=e['first_interaction_future_index_30hz']
                view.update(points_px=json_points(uv),radius_px=radius.tolist(),position_error_m=metrics['position_error_by_future_frame_m'],
                            ADE_m=metrics['ADE_m'],FDE_m=metrics['FDE_m'],contact_outcome=e['category'],
                            contact_match=metrics['contact_outcome_match_oracle'],first_contact_time_s=e['first_interaction_time_after_rgb7_s'],
                            first_contact_path_index=None if index is None else index+1,max_penetration_m=e['max_penetration_m'],
                            post_contact_velocity_error=metrics['post_contact_velocity_error'])
        write(viewer/'data/cases'/f'{cid}.json',payload)
        print(f'RERUN {len(affected)} {cid}: {payload["rollout"]["C"]["initialization"]["status"]}',flush=True)
    report=dict(status='EXECUTED',affected_case_count=len(affected),affected_cases=affected,mode_count=len(runs),
                aligned_modes=sum(r['metrics']['initialization']['status']=='CONTACT_ALIGNED' for r in runs),
                blocked_modes=sum(r['metrics']['initialization']['status']=='BLOCKED_INITIAL_CONTACT' for r in runs),
                strict_rejections=len(runs),strict_rejection_step_calls=0,
                max_post_alignment_initial_penetration_m=max(r['metrics']['initialization']['after_penetration_m'] for r in runs if r['metrics']['initialization']['status']=='CONTACT_ALIGNED'),
                max_rollout_penetration_m=max(r['metrics']['event']['max_penetration_m'] for r in runs if 'event' in r['metrics']))
    manifest['contact_alignment_report']=report
    write(viewer/'data/manifest.json',manifest)
    write(output/'report.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--viewer',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.output,args.viewer)
