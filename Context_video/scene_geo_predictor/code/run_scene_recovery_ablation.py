"""Fresh frozen recovery run; no GT/case-name reads before evaluation."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform,resize_crop_mask
from run_grounded_generic_pilot36 import verify
from observation_scene_recovery import silhouette_depth_sphere
from generic_bullet_input import run

def main(source,out):
    cv2.setNumThreads(2)
    for name in ['input_freeze.json','depth_freeze.json','estimate_freeze.json','rollout_freeze.json']:verify(source,name)
    out.mkdir(exist_ok=False)
    (out/'inputs').symlink_to(source/'inputs',target_is_directory=True)
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(recovery_method='silhouette_depth_sphere_v1',parent_baseline=str(source),
        frozen_changes={'sphere':'joint unknown radius/center from contour tangency and confidence top half depth',
        'geometry':'unchanged frozen baseline','camera':'unchanged frozen VGGT','scale':'unchanged fixed baseline',
        'admission':{'silhouette_relative_rms':.1,'surface_relative_rms':.15,'motion_rms_over_radius':.25},
        'evaluation':'all predeclared 30 sphere candidates; no GT parameter selection; common-valid and coverage separately'})
    dump_json(out/'protocol.json',protocol)
    for folder in sorted((source/'estimates').iterdir()):
        dst=out/'estimates'/folder.name;dst.mkdir(parents=True)
        for name in ['vggt_raw.npz','tracking.npz','mask_result.json','collision_primitive.json']:
            (dst/name).symlink_to(folder/name)
        row=json.loads((folder/'result.json').read_text())
        if not row['support_reason']:
            with np.load(dst/'vggt_raw.npz') as a:d,k,e,confidence=a['depth'][...,0],a['intrinsic'],a['extrinsic'],a['depth_conf']
            if confidence.ndim==4:confidence=confidence[...,0]
            with np.load(dst/'tracking.npz') as a:m=a['masks']
            tr=crop_transform(m.shape[1:]);m=np.stack([resize_crop_mask(x,tr) for x in m])
            times=json.loads((out/'inputs'/folder.name/'timestamps.json').read_text())['time_s']
            try:row['state']=silhouette_depth_sphere(d,confidence,k,e,m,row['scale'],times)
            except ValueError as exc:row['state']={'status':'FAIL','reason':str(exc)}
            row['sources']['state']='silhouette tangency + confidence-selected depth, unknown shared radius'
        row['status']=row['state']['status'];dump_json(dst/'result.json',row)
        print(folder.name,row['state']['status'],row['state'].get('reasons'),flush=True)
    dump_json(out/'input_freeze.json',{'protocol.json':sha256_file(out/'protocol.json')})
    dump_json(out/'depth_freeze.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'estimates').glob('*/vggt_raw.npz'))})
    dump_json(out/'estimate_freeze.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'estimates').glob('*/*'))})
    shutil.copytree(source/'lower_plane_gravity_v1',out/'lower_plane_gravity_v1')
    verify(out/'lower_plane_gravity_v1','freeze.json')
    for g in json.loads((out/'lower_plane_gravity_v1/candidates.json').read_text()):
        folder=out/'estimates'/g['id'];row=json.loads((folder/'result.json').read_text())
        if row['state']['status']!='ESTIMATED':r={'status':row['state']['status'],'step_calls':0,'reason':row['state'].get('reason',row['state'].get('reasons'))}
        else:r=run(row['state'],[json.loads((folder/'collision_primitive.json').read_text())],protocol['physics'],g['gravity'],log_api_contacts=True)
        dump_json(out/'rollouts'/(g['id']+'.json'),r)
    dump_json(out/'rollout_freeze.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'rollouts').glob('*.json'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.source,a.output)
