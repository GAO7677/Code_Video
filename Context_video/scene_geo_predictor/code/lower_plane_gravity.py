"""Explicit upright-camera / lower-image horizontal-plane gravity prior."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform,resize_crop_mask
from check_generic_gravity import planes
from generic_bullet_input import run

def estimate(root,out):
    out.mkdir(exist_ok=False)
    dump_json(out/'protocol.json',{'source':'lower_image_plane_prior','assumptions':['roughly upright camera','camera above visible horizontal lower-image plane'],
        'lower_image_fraction':.35,'min_consensus_fraction':.35,'max_down_vs_image_down_deg':60,'max_frame_drift_deg':3,
        'validation_max_signed_gravity_error_deg':5,'gravity_mps2':9.81,'no_state_or_geometry_alignment':True})
    for n,h in json.loads((root/'estimate_freeze.json').read_text()).items():
        if sha256_file(root/n)!=h:raise ValueError('source modified')
    scale=json.loads((root/'protocol.json').read_text())['scale_config']['meters_per_reconstruction_unit'];rows=[]
    for folder in sorted((root/'estimates').iterdir()):
        if not (folder/'tracking.npz').exists():
            rows.append({'id':folder.name,'frames':[],'gravity':{'status':'UNKNOWN','source':'lower_image_plane_prior','vector':None,'reason':'target_tracking_unavailable'}})
            continue
        with np.load(folder/'vggt_raw.npz') as a:d=a['depth'][...,0];k=a['intrinsic'];e=a['extrinsic']
        with np.load(folder/'tracking.npz') as a:m=a['masks']
        trans=crop_transform(m.shape[1:]);m=np.stack([resize_crop_mask(x,trans) for x in m]);union=cv2.dilate(np.any(m,axis=0).astype(np.uint8),np.ones((9,9),np.uint8))>0
        h,w=d.shape[1:];yy,xx=np.mgrid[:h,:w];pixels=np.stack([xx,yy,np.ones_like(xx)],-1);frames=[]
        for t in [0,7]:
            valid=(yy>=.65*h)&~union&np.isfinite(d[t])&(d[t]>0)
            cam=(pixels@np.linalg.inv(k[t]).T)*d[t,...,None]*scale;world=(cam-e[t,:,3]*scale)@e[t,:,:3];ref=world@e[0,:,:3].T+e[0,:,3]*scale
            pts=ref[valid];rng=np.random.default_rng(42);pts=pts[rng.choice(len(pts),min(8000,len(pts)),replace=False)]
            candidates=planes(pts,42);eligible=[]
            for p in candidates:
                down=-np.array(p['normal_camera_facing']);p['down_vs_image_down_deg']=float(np.degrees(np.arccos(np.clip(down[1],-1,1))))
                if p['fraction_of_all']>=.35 and down[1]>=.5:eligible.append(p)
            frames.append({'frame':t,'candidates':candidates,'selected':eligible[0] if eligible else None})
        row={'id':folder.name,'frames':frames,'gravity':{'status':'UNKNOWN','source':'lower_image_plane_prior','vector':None}}
        if all(f['selected'] is not None for f in frames):
            vectors=[-np.array(f['selected']['normal_camera_facing']) for f in frames]
            drift=float(np.degrees(np.arccos(np.clip(vectors[0]@vectors[1],-1,1))));row['drift_deg']=drift
            if drift<=3:
                down=np.mean(vectors,axis=0);down/=np.linalg.norm(down)
                row['gravity'].update(status='ESTIMATED',vector=(e[0,:,:3].T@down*9.81).tolist(),camera0_down=down.tolist())
        rows.append(row)
    dump_json(out/'candidates.json',rows);dump_json(out/'freeze.json',{n:sha256_file(out/n) for n in ['protocol.json','candidates.json']})

def evaluate(root,out):
    for n,h in json.loads((out/'freeze.json').read_text()).items():
        if sha256_file(out/n)!=h:raise ValueError('freeze mismatch')
    rows=json.loads((out/'candidates.json').read_text());mapping={x['id']:x for x in json.loads((root/'evaluation_mapping.json').read_text())}
    for row in rows:
        camera=json.loads((Path(mapping[row['id']]['source_sample'])/'videos/rgb_cycles.json').read_text())['camera']
        f=np.array(camera['target'])-camera['location'];f/=np.linalg.norm(f);right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);rot=np.stack([right,-np.cross(right,f),f])
        gt=rot@np.array([0,0,-1]);row['signed_error_deg']=None
        if row['gravity']['status']=='ESTIMATED':row['signed_error_deg']=float(np.degrees(np.arccos(np.clip(np.array(row['gravity']['camera0_down'])@gt,-1,1))))
    passed=all(r['signed_error_deg'] is not None and r['signed_error_deg']<=5 for r in rows)
    dump_json(out/'validation.json',{'six_case_prior_gate':'PASS' if passed else 'FAIL','rows':rows})
    print(json.dumps({'gate':passed,'rows':[{k:r.get(k) for k in ['id','drift_deg','signed_error_deg']} for r in rows]},indent=2))
    if passed:
        physics=json.loads((root/'protocol.json').read_text())['physics']
        for row in rows:
            folder=root/'estimates'/row['id'];state=json.loads((folder/'result.json').read_text())['state'];mesh=json.loads((folder/'collision_primitive.json').read_text())
            result=run(state,[mesh],physics,row['gravity']);dump_json(out/(row['id']+'_rollout.json'),result)
            print(row['id'],result['status'],result.get('step_calls'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['estimate','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2)
    (estimate if a.phase=='estimate' else evaluate)(a.root,a.output)
