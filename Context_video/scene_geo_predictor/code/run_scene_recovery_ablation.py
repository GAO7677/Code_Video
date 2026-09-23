"""Fresh frozen recovery run; no GT/case-name reads before evaluation."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform,resize_crop_mask
from run_grounded_generic_pilot36 import verify
from observation_scene_recovery import silhouette_depth_sphere, fixed_camera, regularize_planes
from generic_context_geometry import finite_mesh_observed
from generic_bullet_input import run

def main(source,out,use_fixed_camera=False,planes=False):
    cv2.setNumThreads(2)
    for name in ['input_freeze.json','depth_freeze.json','estimate_freeze.json','rollout_freeze.json']:verify(source,name)
    out.mkdir(exist_ok=False)
    (out/'inputs').symlink_to(source/'inputs',target_is_directory=True)
    protocol=json.loads((source/'protocol.json').read_text())
    protocol.update(recovery_method='silhouette_depth_sphere_front_cap_v2',parent_baseline=str(source),
        frozen_changes={'sphere':'joint unknown radius/center from contour tangency and confidence top half depth',
        'geometry':'unchanged frozen baseline','camera':'unchanged frozen VGGT','scale':'unchanged fixed baseline',
        'sphere_front_constraint':'center_z >= p90 selected front surface z; no backside optimum',
        'admission':{'silhouette_relative_rms':.1,'surface_relative_rms':.15,'motion_rms_over_radius':.25},
        'evaluation':'all predeclared 30 sphere candidates; no GT parameter selection; common-valid and coverage separately'})
    dump_json(out/'protocol.json',protocol)
    protocol['fixed_camera_enabled']=use_fixed_camera
    protocol['plane_regularization_enabled']=planes
    if use_fixed_camera:
        protocol['frozen_changes']['camera']='static feature p90 drift <=1.5px; common median K/reference E; rebuild observed mesh'
    if planes:protocol['frozen_changes']['geometry']='observed plane regularization <=1cm, faces/holes unchanged'
    protocol['code_hashes']={name:sha256_file(Path(__file__).parent/name) for name in ['run_scene_recovery_ablation.py','observation_scene_recovery.py']}
    dump_json(out/'protocol.json',protocol)
    for folder in sorted((source/'estimates').iterdir()):
        dst=out/'estimates'/folder.name;dst.mkdir(parents=True)
        for name in ['tracking.npz','mask_result.json']:
            (dst/name).symlink_to(folder/name)
        row=json.loads((folder/'result.json').read_text())
        camera_error=None
        with np.load(folder/'vggt_raw.npz') as a:arrays={n:a[n] for n in a.files}
        with np.load(dst/'tracking.npz') as a:original_masks=a['masks']
        tr=crop_transform(original_masks.shape[1:]);processed=np.stack([resize_crop_mask(x,tr) for x in original_masks])
        if use_fixed_camera:
            rgb=[cv2.imread(str(out/'inputs'/folder.name/f'rgb_{t:02d}.png')) for t in range(8)]
            try:
                arrays['intrinsic'],arrays['extrinsic'],row['camera_audit']=fixed_camera(rgb,original_masks,arrays['intrinsic'],arrays['extrinsic'])
            except ValueError as exc:camera_error=str(exc);row['camera_audit']={'status':'FAIL','reason':camera_error}
            np.savez_compressed(dst/'vggt_raw.npz',**arrays)
            (dst/'vggt_model_raw.npz').symlink_to(folder/'vggt_raw.npz')
        else:(dst/'vggt_raw.npz').symlink_to(folder/'vggt_raw.npz')
        if use_fixed_camera and not camera_error:
            mesh,_=finite_mesh_observed(arrays['depth'][...,0],arrays['intrinsic'],arrays['extrinsic'],processed,row['scale'],complete_local_planes=True)
        else:mesh=json.loads((folder/'collision_primitive.json').read_text())
        if planes:mesh=regularize_planes(mesh)
        dump_json(dst/'collision_primitive.json',mesh)
        row['geometry'].update(triangles=len(mesh['faces']))
        if camera_error:row['geometry'].update(status='FAIL',reason=camera_error)
        if not row['support_reason']:
            with np.load(dst/'vggt_raw.npz') as a:d,k,e,confidence=a['depth'][...,0],a['intrinsic'],a['extrinsic'],a['depth_conf']
            if confidence.ndim==4:confidence=confidence[...,0]
            with np.load(dst/'tracking.npz') as a:m=a['masks']
            tr=crop_transform(m.shape[1:]);m=np.stack([resize_crop_mask(x,tr) for x in m])
            times=json.loads((out/'inputs'/folder.name/'timestamps.json').read_text())['time_s']
            try:row['state']=silhouette_depth_sphere(d,confidence,k,e,m,row['scale'],times)
            except ValueError as exc:row['state']={'status':'FAIL','reason':str(exc)}
            row['sources']['state']='silhouette tangency + confidence-selected depth, unknown shared radius'
            if camera_error:row['state']={'status':'FAIL','reason':camera_error}
        row['status']=row['state']['status'];dump_json(dst/'result.json',row)
        print(folder.name,row['state']['status'],row['state'].get('reasons'),flush=True)
    input_hashes={str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'inputs').glob('*/*'))}
    input_hashes['protocol.json']=sha256_file(out/'protocol.json')
    dump_json(out/'input_freeze.json',input_hashes)
    dump_json(out/'depth_freeze.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'estimates').glob('*/vggt_raw.npz'))})
    dump_json(out/'estimate_freeze.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'estimates').glob('*/*'))})
    if use_fixed_camera:
        from lower_plane_gravity import estimate
        estimate(out,out/'lower_plane_gravity_v1')
    else:shutil.copytree(source/'lower_plane_gravity_v1',out/'lower_plane_gravity_v1')
    verify(out/'lower_plane_gravity_v1','freeze.json')
    for g in json.loads((out/'lower_plane_gravity_v1/candidates.json').read_text()):
        folder=out/'estimates'/g['id'];row=json.loads((folder/'result.json').read_text())
        if row['state']['status']!='ESTIMATED':r={'status':row['state']['status'],'step_calls':0,'reason':row['state'].get('reason',row['state'].get('reasons'))}
        else:r=run(row['state'],[json.loads((folder/'collision_primitive.json').read_text())],protocol['physics'],g['gravity'],log_api_contacts=True)
        dump_json(out/'rollouts'/(g['id']+'.json'),r)
    dump_json(out/'rollout_freeze.json',{str(p.relative_to(out)):sha256_file(p) for p in sorted((out/'rollouts').glob('*.json'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-camera',action='store_true');p.add_argument('--planes',action='store_true')
    a=p.parse_args();main(a.source,a.output,a.fixed_camera,a.planes)
