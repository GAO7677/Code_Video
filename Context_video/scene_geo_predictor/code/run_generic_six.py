"""Six anonymous context clips, VGGT/SAM2 only; strict observation-only estimation."""
import argparse
import json
import os
from pathlib import Path
import sys
import shutil
import time
import cv2
import numpy as np
from PIL import Image
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform,resize_crop_mask
from generic_context_geometry import motion_prompt,sphere_fit,finite_mesh

def prepare(root,config):
    selected=[0,15,30,40,45,50]
    protocol={'selection':'Before any inference: first ordinal of each of six visually screened single-sphere nonjoint layouts; no result replacement',
        'selected_ordinals':selected,'screened_only':'70 anonymous RGB0 thumbnails; no state/blueprint/caption read',
        'excluded_visual_ordinals':{'5-14,20-29':'non-sphere','35-39':'multiple potentially dynamic objects','55-69':'visible joints/suspension'},
        'scale_config':json.loads(config.read_text()),'main_omega':'zero','threads':2,
        'gates':{'motion_min_pixel_change':12,'motion_candidate_ambiguity_ratio':.8,'camera_translation_pixels':3,
                 'mask_circularity_min':.65,'radius_cv_max':.25,'sphere_relative_rms_max':.15,'sphere_condition_max':1e5,
                 'mesh_max_relative_depth_jump':.03,'initial_overlap_m':.001},
        'gravity':'Normals alone cannot uniquely label horizontal/sign; UNKNOWN unless independent observation evidence resolves ambiguity. No default floor, Z-up or support.',
        'frame':'VGGT reference-camera frame scaled uniformly; not asserted Z-up',
        'physics':{'gravity_magnitude':9.81,'mass':1.,'friction':.35,'restitution':.25,'linear_damping':0.,'angular_damping':0.,
                   'fixedTimeStep':1/240,'numSubSteps':8,'calls_per_output':8,'outputs':41,'fps':30},
        'status':'PREDECLARED'}
    if (root/'protocol.json').exists():raise FileExistsError('protocol already frozen')
    dump_json(root/'protocol.json',protocol)
    mapping=json.loads((root/'curator_mapping.json').read_text()); chosen=[]
    for j,ordinal in enumerate(selected):
        identifier=f'clip_{j:02d}';src=root/'screening'/f'candidate_{ordinal:03d}';dst=root/'inputs'/identifier
        shutil.copytree(src,dst)
        chosen.append({'id':identifier,**mapping[ordinal]})
    dump_json(root/'evaluation_mapping.json',chosen)
    hashes={str(p.relative_to(root)):sha256_file(p) for p in sorted((root/'inputs').glob('*/*'))}
    hashes['protocol.json']=sha256_file(root/'protocol.json');dump_json(root/'input_freeze.json',hashes)

def run(root):
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-7f6fbc40-3594-2c34-8557-422621355ff9':raise ValueError('expected authorized GPU6 UUID')
    for key,digest in json.loads((root/'input_freeze.json').read_text()).items():
        if sha256_file(root/key)!=digest:raise ValueError('input freeze mismatch')
    protocol=json.loads((root/'protocol.json').read_text());scale=protocol['scale_config']['meters_per_reconstruction_unit']
    sys.path.insert(0,'/home/gaoya/vggt_official')
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    clips=sorted((root/'inputs').iterdir()); results={}
    out=root/'estimates';out.mkdir(exist_ok=False)
    model=VGGT.from_pretrained('/data/gaoya/ckpt/facebook-VGGT-1B').to('cuda:0').eval().requires_grad_(False)
    for clip in clips:
        dst=out/clip.name;dst.mkdir();started=time.perf_counter()
        images=load_and_preprocess_images([str(clip/f'rgb_{i:02d}.png') for i in range(8)]).to('cuda:0')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):pred=model(images)
        e,k=pose_encoding_to_extri_intri(pred['pose_enc'],images.shape[-2:])
        arrays={name:pred[name].detach().float().cpu().numpy().squeeze(0) for name in ['depth','depth_conf','world_points','world_points_conf']}
        arrays.update(extrinsic=e.detach().float().cpu().numpy().squeeze(0),intrinsic=k.detach().float().cpu().numpy().squeeze(0))
        np.savez_compressed(dst/'vggt_raw.npz',**arrays)
        results[clip.name]={'vggt_seconds':time.perf_counter()-started}
        del pred,images,e,k,arrays;torch.cuda.empty_cache();print('VGGT',clip.name,flush=True)
    del model;torch.cuda.empty_cache()
    sys.path.insert(0,'/home/gaoya/Code_Video/phys_state_video/src')
    from phys_state_video.mask_tracking import SAM2VideoMaskTracker
    tracker=SAM2VideoMaskTracker(device='cuda:0',model_id=None,
        model_cfg='/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml',
        checkpoint_path=Path('/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt'))
    for clip in clips:
        dst=out/clip.name;row=results[clip.name]
        row.update(status='FAIL',scale=scale,scale_source=protocol['scale_config']['scale_source'],
          sources={'input':'anonymous RGB0..7 and timestamps','camera':'VGGT prediction','target':'temporal motion candidate then SAM2',
          'radius':'joint visible 3D surface sphere fit','velocity':'robust linear centers fit','geometry':'VGGT observed finite surface',
          'omega':'fixed zero baseline','physics':'fixed protocol'},
          fallback={'used':False,'gt':False,'blueprint':False,'family':False,'default_radius':False,'default_support':False,'position_alignment':False},
          state={'status':'NOT_RUN'},geometry={'status':'NOT_RUN'},gravity={'status':'UNKNOWN'},rollout={'status':'NOT_RUN'})
        rgb=np.stack([np.array(Image.open(clip/f'rgb_{i:02d}.png').convert('RGB')) for i in range(8)])
        try:
            box,audit=motion_prompt(rgb);row['target_detection']=audit
            started=time.perf_counter()
            masks=tracker.track_from_boxes(rgb.transpose(0,3,1,2).astype(np.float32)/255,prompt_frame_idx=7,boxes_xyxy=box[None])[:,0].astype(bool)
            row['sam2_seconds']=time.perf_counter()-started
            np.savez_compressed(dst/'tracking.npz',masks=masks,prompt=box)
            transform=crop_transform(rgb.shape[1:3]);processed=np.stack([resize_crop_mask(m,transform) for m in masks])
            with np.load(dst/'vggt_raw.npz') as raw:
                depth=raw['depth'][...,0];k=raw['intrinsic'];e=raw['extrinsic']
            times=json.loads((clip/'timestamps.json').read_text())['time_s']
            # Independent state/geometry diagnostics still run when one fails.
            try:row['state']=sphere_fit(depth,k,e,processed,scale,times)
            except ValueError as exc:row['state']={'status':'FAIL','reason':str(exc)}
            try:
                mesh,gravity=finite_mesh(depth,k,e,processed,scale)
                dump_json(dst/'collision_primitive.json',mesh);row['geometry']={'status':'ESTIMATED','triangles':len(mesh['faces']),'thickness':'UNKNOWN'};row['gravity']=gravity
            except ValueError as exc:row['geometry']={'status':'FAIL','reason':str(exc)}
            row['status']='UNKNOWN' if row['state']['status']=='ESTIMATED' and row['geometry']['status']=='ESTIMATED' else 'FAIL'
            row['rollout']={'status':'NOT_RUN','reason':'gravity direction UNKNOWN; incomplete admissible physics input'}
        except ValueError as exc:row['reason']=str(exc)
        dump_json(dst/'result.json',row);print('RESULT',clip.name,row['status'],row['state'].get('reasons',row['state'].get('reason','')),flush=True)
    freeze={str(p.relative_to(root)):sha256_file(p) for p in sorted(out.glob('*/*'))}
    freeze['protocol.json']=sha256_file(root/'protocol.json');dump_json(root/'estimate_freeze.json',freeze)
    print('FROZEN',sha256_file(root/'estimate_freeze.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','estimate']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scale-config',type=Path,default=Path(__file__).resolve().parents[1]/'configs/context_reconstruction_scale.json');a=p.parse_args()
    prepare(a.root,a.scale_config) if a.phase=='prepare' else run(a.root)
