"""Post-freeze A/B/C/D zero-omega diagnostics, including bounded C alignment."""
import argparse,copy,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
from context_rgb_pybullet_common import dump_json,sha256_file
from run_grounded_generic_pilot36 import verify
from generic_bullet_input import run as bullet
import evaluate_context_rgb_pybullet as legacy

BASE=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')

def project(points,k):
    a=np.asarray(points)@k.T
    return [[float(v) for v in row[:2]/row[2]] if row[2]>1e-6 else [None,None] for row in a]


def run(source,out):
    for f in ['estimate_freeze.json','rollout_freeze.json']:verify(source,f)
    verify(source/'lower_plane_gravity_v1','freeze.json')
    out.mkdir(exist_ok=False)
    for name in ['inputs','diagnostics']:(out/name).symlink_to(source/name,target_is_directory=True)
    protocol=json.loads((source/'protocol.json').read_text());physics=protocol['physics']
    dump_json(out/'protocol.json',{'source':str(source),'source_estimate_freeze_sha256':sha256_file(source/'estimate_freeze.json'),
        'omega':'zero for all arms','gravity':'same frozen estimated gravity for all arms, NOT GT gravity',
        'physics':physics,'A':'GT p7/v7/radius + GT geometry',
        'B':'estimated p7/v7/radius + GT geometry','C_strict':'GT p7/v7/radius + estimated completed mesh',
        'C_aligned':'C with explicit upward support-only penetration correction <=55mm; separate diagnostic',
        'D':'estimated p7/v7/radius + estimated completed mesh',
        'alignment_support_evidence':'GT-input A initial normal contacts, evaluation only',
        'coordinate_registration':'GT camera extrinsic -> VGGT camera0 rigid frame; no scale fitting or trajectory alignment',
        'contact_metric':'API-call sampled generic support/obstacle roles from normals; no family-derived semantic labels',
        'not_strict_causal':'state, reconstructed geometry, scale and gravity share visual estimates',
        'gt_boundary':'GT first read only after verified frozen estimates', 'threads':2,'gpu_used':False})
    recheck,pilot,generator=legacy.load_engine()
    manifest={r['key']:r for r in json.loads((legacy.DEFAULT_DATA/'pilot_manifest.json').read_text())['records']}
    gravity={x['id']:x['gravity'] for x in json.loads((source/'lower_plane_gravity_v1/candidates.json').read_text())}
    data=json.loads((source/'viewer_data.json').read_text());records=data['records'];start=time.perf_counter()
    for row in records:
        key=row['id'];cid=row['case_id'];folder=source/'estimates'/key
        est=json.loads((folder/'result.json').read_text());mesh=json.loads((folder/'collision_primitive.json').read_text())
        case,seed=recheck.reconstruct_case(pilot,generator,manifest[cid])
        raw=recheck.load_raw_state(legacy.DEFAULT_DATA/'samples'/cid);di=raw['dynamic_index']
        gtpos=raw['positions'][7,di];gtvel=raw['linear_velocities'][7,di]
        target=raw['positions'][8:49,di];target_v=raw['linear_velocities'][8:49,di]
        cal=json.loads((BASE/'vision_inputs'/cid/'calibration.json').read_text());E=np.array(cal['world_to_camera_3x4']);K=np.array(cal['intrinsic_K'])
        with np.load(folder/'vggt_raw.npz') as z:e0=z['extrinsic'][0]
        scale=est['scale'];R=e0[:,:3].T@E[:,:3];t=e0[:,:3].T@(E[:,3]-e0[:,3]*scale)
        q=Rotation.from_matrix(R).as_quat().tolist()
        def to_est(x):return np.asarray(x)@R.T+t
        def camera(x):return np.asarray(x)@e0[:,:3].T+e0[:,3]*scale
        gtball=next(obj for obj in case.blueprint.objects if obj.name=='pilot_ball' and obj.dynamic)
        radius=float(gtball.size['radius'])  # Required GT field; never default a missing radius.
        gtstate={'status':'ESTIMATED','shape':'sphere','radius':radius,'p7':to_est(gtpos).tolist(),
                 'v7':(R@gtvel).tolist(),'omega':[0,0,0]}
        estimated=copy.deepcopy(est['state']);estimated['omega']=[0,0,0]
        g=gravity[key];up=-np.array(g['vector'])/np.linalg.norm(g['vector'])
        def builder(api,client):
            _,names=legacy.add_gt_geometry(api,client,generator,case.blueprint,seed)
            for body in names:
                pos,rot=api.getBasePositionAndOrientation(body,physicsClientId=client)
                newpos,newrot=api.multiplyTransforms(t.tolist(),q,pos,rot)
                api.resetBasePositionAndOrientation(body,newpos,newrot,physicsClientId=client)
                # Shared fixed material protocol, not GT material inference.
                api.changeDynamics(body,-1,lateralFriction=physics['friction'],restitution=physics['restitution'],
                                   rollingFriction=0,spinningFriction=0,physicsClientId=client)
        row['abcd']={};confirmed=False
        for arm in ['A','B','C_strict','C_aligned','D']:
            state=estimated if arm in ['B','D'] else gtstate
            builder_arg=builder if arm in ['A','B'] else None
            primitives=[] if builder_arg else [mesh]
            align={'up':up.tolist(),'confirmed_support':confirmed,'aligned':arm=='C_aligned'} if arm.startswith('C') else None
            result=bullet(state,primitives,physics,g,geometry_builder=builder_arg,
                          diagnostic_alignment=align,log_api_contacts=True)
            if arm=='A':
                confirmed=any(abs(c['distance'])<=.001 and np.dot(c['normal'],up)>=.99 for c in result.get('initial_contacts_detailed',[]))
            path=out/'rollouts'/key/(arm+'.json');dump_json(path,result)
            m={'status':result['status'],'reason':result.get('reason'),'step_calls':result.get('step_calls',0),
               'radius_m':state.get('radius'),'omega':'zero','ADE_m':None,'FDE_m':None,'points_px':[],
               'position_error_m':[],'contact_outcome':'NOT_RUN','contact_match':None,'first_contact_time_s':None,
               'max_penetration_m':None,'initialization':result.get('initialization',{
                   'policy':'strict','status':'UNCHANGED' if result['status']=='EXECUTED' else 'BLOCKED',
                   'before_penetration_m':result.get('initial_penetration_m'),
                   'after_penetration_m':result.get('initial_penetration_m'),'displacement_m':[0,0,0],
                   'reason':result.get('reason')}),'rollout_file':str(path.relative_to(out))}
            if result['status']=='EXECUTED':
                positions=np.array(result['positions']);predcam=camera(positions);gtcam=target@E[:,:3].T+E[:,3]
                errors=np.linalg.norm(predcam-gtcam,axis=1)
                initial=result.get('initialization',{}).get('after_position_m',state['p7'])
                allcam=camera(np.vstack([initial,positions]));uv=project(allcam,K)
                initial_roles={role(c,up) for c in result['initial_contacts_detailed']}
                role_samples=[{role(c,up) for c in sample} for sample in result['api_contacts']]
                allroles=set().union(*role_samples)
                first_new=next((i for i,roles in enumerate(role_samples) if roles-initial_roles),None)
                first_any=next((i for i,roles in enumerate(role_samples) if roles),None)
                support_lost=next((i for i,roles in enumerate(role_samples) if 'support' in initial_roles and 'support' not in roles),None)
                outcomes=[]
                if 'obstacle' in allroles:outcomes.append('obstacle_contact')
                if support_lost is not None:outcomes.append('initial_support_lost')
                if 'support' in allroles:outcomes.append('support_contact')
                if not outcomes:outcomes=['no_contact']
                penetration=max([max(0.,-c['distance']) for sample in result['api_contacts'] for c in sample],default=0.)
                m.update(ADE_m=float(errors.mean()),FDE_m=float(errors[-1]),position_error_m=errors.tolist(),points_px=uv,
                    radius_px=np.clip(K[0,0]*state['radius']/np.maximum(allcam[:,2],1e-6),2,80).tolist(),
                    contact_outcome=' + '.join(outcomes),initial_contact_roles=sorted(initial_roles),
                    first_contact_time_s=None if first_any is None else (first_any+1)/240,
                    first_new_contact_time_s=None if first_new is None else (first_new+1)/240,
                    initial_support_loss_time_s=None if support_lost is None else (support_lost+1)/240,
                    max_penetration_m=penetration,contact_frames=sum(bool(c) for c in result['contacts']),
                    velocity_error_mps=float(np.linalg.norm(np.array(result['velocities'])@e0[:,:3].T-target_v@E[:,:3].T,axis=1).mean()))
                if arm=='D':
                    original=json.loads((source/'rollouts'/f'{key}.json').read_text())
                    m['max_difference_from_frozen_D_m']=float(np.max(np.linalg.norm(positions-np.array(original['positions']),axis=1)))
                    assert m['max_difference_from_frozen_D_m']<1e-8,'D replay changed'
            row['abcd'][arm]=m
        for m in row['abcd'].values():
            if m['status']=='EXECUTED' and row['abcd']['A']['status']=='EXECUTED':m['contact_match']=m['contact_outcome']==row['abcd']['A']['contact_outcome']
        row['gt_geometry_segments_uv']=gt_wire(case.blueprint,generator,E,K)
        row['gt_geometry_note']='actual static box boundaries; plane.urdf has no finite wire border'
        row['ground_truth_radius_m']=radius
        dump_json(out/'cases'/f'{key}.json',row)
        print(cid,[(a,m['status'],m['ADE_m']) for a,m in row['abcd'].items()],flush=True)
    summary={'cases':len(records),'groups':len({(r['family'],r['group_id']) for r in records}),'arms':{}}
    for arm in ['A','B','C_strict','C_aligned','D']:
        ms=[r['abcd'][arm] for r in records];done=[m for m in ms if m['status']=='EXECUTED']
        summary['arms'][arm]={'status_counts':dict(Counter(m['status'] for m in ms)),
            'ADE_mean_m':float(np.mean([m['ADE_m'] for m in done])) if done else None,
            'FDE_mean_m':float(np.mean([m['FDE_m'] for m in done])) if done else None,
            'aligned':sum(m['initialization'].get('status')=='CONTACT_ALIGNED' for m in ms),
            'blocked_reasons':dict(Counter(str(m['reason']) for m in ms if m['status']!='EXECUTED'))}
    dump_json(out/'viewer_data.json',{'summary':summary,'records':records})
    dump_json(out/'rollout_freeze.json',{str(x.relative_to(out)):sha256_file(x) for x in sorted((out/'rollouts').glob('*/*.json'))})
    dump_json(out/'runtime.json',{'seconds':time.perf_counter()-start,'cpu_threads':2,'gpu':False})
    print(json.dumps(summary,indent=2))


def role(contact,up):
    return 'support' if np.dot(contact['normal'],up)>=.7 else 'obstacle'


def gt_wire(blueprint,generator,E,K):
    signs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)];segments=[]
    for obj in blueprint.objects:
        if obj.dynamic or obj.metadata.get('visual_only') or not all(k in obj.size for k in ['hx','hy','hz']):continue
        q=generator.legacy._quat_from_euler_deg(list(obj.orientation_euler_deg));R=np.array(p.getMatrixFromQuaternion(q)).reshape(3,3)
        xyz=(signs*np.array([obj.size[k] for k in ['hx','hy','hz']]))@R.T+obj.position
        uv=project(xyz@E[:,:3].T+E[:,3],K)
        segments.extend([[uv[a],uv[b]] for a,b in edges if all(v is not None for v in uv[a]+uv[b])])
    return segments


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    args=a.parse_args();run(args.source,args.output)
