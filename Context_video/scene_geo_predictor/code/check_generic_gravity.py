"""Observation-only dominant-plane probe, frozen before independent gravity evaluation."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform,resize_crop_mask

def planes(points,seed):
    rng=np.random.default_rng(seed);remaining=points.copy();out=[]
    for rank in range(3):
        if len(remaining)<300:break
        best=None;count=0
        for _ in range(150):
            a,b,c=remaining[rng.choice(len(remaining),3,replace=False)]
            n=np.cross(b-a,c-a);norm=np.linalg.norm(n)
            if norm<1e-8:continue
            n/=norm;d=-n@a;inside=abs(remaining@n+d)<.03
            if inside.sum()>count:best=inside;count=int(inside.sum())
        if count<300:break
        p=remaining[best];center=np.mean(p,axis=0);_,_,v=np.linalg.svd(p-center,full_matrices=False);n=v[-1]
        # Sign is only camera-facing, explicitly NOT gravity-down.
        if n@center>0:n=-n
        d=-n@center;inside=abs(remaining@n+d)<.03;p=remaining[inside]
        out.append({'normal_camera_facing':n.tolist(),'offset':float(d),'points':len(p),
                    'fraction_of_all':len(p)/len(points),'rms_m':float(np.sqrt(np.mean((p@n+d)**2)))})
        remaining=remaining[~inside]
    return out

def estimate(root,out):
    out.mkdir(exist_ok=False)
    protocol={'plane_residual_m':.03,'ransac_trials':150,'max_planes':3,'sample_points':8000,
              'minimum_points':300,'seed':42,'frames':[0,7],
              'predeclared_validation':{'axis_error_max_deg':5,'frame_stability_max_deg':3},
              'candidate':'largest consensus plane; sign camera-facing, not asserted gravity',
              'admission':'Require observation-side horizontal classification and gravity sign; largest plane alone insufficient',
              'no_new_priors':'No image-down, upright-camera, floor-is-largest, support or family assumption'}
    dump_json(out/'protocol.json',protocol)
    for key,digest in json.loads((root/'estimate_freeze.json').read_text()).items():
        if sha256_file(root/key)!=digest:raise ValueError('source freeze mismatch')
    scale=json.loads((root/'protocol.json').read_text())['scale_config']['meters_per_reconstruction_unit']
    rows=[]
    for folder in sorted((root/'estimates').iterdir()):
        with np.load(folder/'vggt_raw.npz') as a:depth=a['depth'][...,0];k=a['intrinsic'];e=a['extrinsic']
        with np.load(folder/'tracking.npz') as a:masks=a['masks']
        transform=crop_transform(masks.shape[1:]);m=np.stack([resize_crop_mask(x,transform) for x in masks]);union=cv2.dilate(np.any(m,axis=0).astype(np.uint8),np.ones((9,9),np.uint8))>0
        h,w=depth.shape[1:];yy,xx=np.mgrid[:h,:w];pixels=np.stack([xx,yy,np.ones_like(xx)],-1)
        frames=[]
        for t in [0,7]:
            valid=~union&np.isfinite(depth[t])&(depth[t]>0)
            cam=(pixels@np.linalg.inv(k[t]).T)*depth[t,...,None]*scale
            world=(cam-e[t,:,3]*scale)@e[t,:,:3]
            ref=world@e[0,:,:3].T+e[0,:,3]*scale
            points=ref[valid];rng=np.random.default_rng(42);points=points[rng.choice(len(points),min(8000,len(points)),replace=False)]
            frames.append({'frame':t,'planes':planes(points,42)})
        row={'id':folder.name,'frames':frames,'gravity_status':'UNKNOWN','reason':'plane geometry lacks observed horizontal label and down sign'}
        if all(x['planes'] for x in frames):
            a=np.array(frames[0]['planes'][0]['normal_camera_facing']);b=np.array(frames[1]['planes'][0]['normal_camera_facing'])
            row['dominant_axis_frame_drift_deg']=float(np.degrees(np.arccos(np.clip(abs(a@b),0,1))))
        rows.append(row)
    dump_json(out/'candidates.json',rows)
    dump_json(out/'freeze.json',{n:sha256_file(out/n) for n in ['protocol.json','candidates.json']})

def evaluate(root,out):
    for n,d in json.loads((out/'freeze.json').read_text()).items():
        if sha256_file(out/n)!=d:raise ValueError('candidate freeze mismatch')
    mapping={r['id']:r for r in json.loads((root/'evaluation_mapping.json').read_text())}
    rows=json.loads((out/'candidates.json').read_text())
    for row in rows:
        sample=Path(mapping[row['id']]['source_sample']);meta=json.loads((sample/'videos/rgb_cycles.json').read_text())
        camera=meta['camera'];eye=np.array(camera['location']);f=np.array(camera['target'])-eye;f/=np.linalg.norm(f)
        right=np.cross(f,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,f);rotation=np.stack([right,-up,f])
        gravity=rotation@np.array([0,0,-1.]);row['GT_down_eval_only']=gravity.tolist()
        for frame in row['frames']:
            for plane in frame['planes']:
                n=np.array(plane['normal_camera_facing']);plane['unsigned_GT_gravity_axis_error_deg']=float(np.degrees(np.arccos(np.clip(abs(n@gravity),0,1))))
        row['dominant_axis_within_5deg']=all(f['planes'] and f['planes'][0]['unsigned_GT_gravity_axis_error_deg']<=5 for f in row['frames'])
    dump_json(out/'evaluation.json',{'status':'EXECUTED','adopt_prior':False,'records':rows,'reason':'no observation-side horizontal/sign disambiguation; GT not used to pick planes'})
    for r in rows:print(r['id'],'drift',r.get('dominant_axis_frame_drift_deg'),'GT errors',[[round(p['unsigned_GT_gravity_axis_error_deg'],2) for p in f['planes']] for f in r['frames']])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['estimate','evaluate']);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2)
    (estimate if a.phase=='estimate' else evaluate)(a.root,a.output)
