"""Frozen 12-case, context-only DA/VDA diagnostic against rendered GT depth."""
import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(2)

ROOT=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
OUT=ROOT/'depth_comparison_12_v1'
DA=Path('/data/gaoya/agent-data/third_party/Depth-Anything')
VDA=Path('/data/gaoya/agent-data/third_party/Video-Depth-Anything')
WEIGHTS={'da':Path('/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14/pytorch_model.bin'),
         'vda':Path('/data/gaoya/ckpt/depth-anything-Video-Depth-Anything-Large/video_depth_anything_vitl.pth')}

def read(p):return json.loads(p.read_text())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def cases():return read(OUT/'protocol.json')['cases']
def rgb(cid):return np.stack([cv2.cvtColor(cv2.imread(str(ROOT/'vision_inputs'/cid/f'rgb_{i:02d}.png')),cv2.COLOR_BGR2RGB) for i in range(8)])

def prepare():
    OUT.mkdir(exist_ok=False)
    selected=[]
    for family,variants in [('aperture',['0280','0460','0720','0460']),('deflector',['28000','58000','86000','28000']),('support_edge',['0080','0300','0620','0620'])]:
        selected += [f'phase1_{family}_g{i:02d}_v{v}' for i,v in enumerate(variants)]
    write(OUT/'protocol.json',dict(cases=selected,selection='one per history group per family, prespecified before DA/VDA evaluation',frames=list(range(8)),
         inference_input='RGB0-7 only',gpu=6,input_size=518,da_precision='FP32',vda_precision='official autocast FP16',
         vda_padding='repeat RGB7 to 32 frames; no future RGB',models=['da','vda','vggt'],
         metric_protocol='GT inverse-depth affine alignment on RGB0 checkerboard calibration pixels (0.2<Z<15m); evaluate RGB1-7 opposite checkerboard pixels, no per-frame alignment; GT fit is diagnostic, NOT deployable metric prediction',
         metrics=['AbsRel','RMSE_m','delta1','invalid_aligned_inverse_fraction','static_inverse_temporal_std'],
         regions=['all','objects','static_objects'],training=False,fallback=False))

def infer(model_name):
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no fallback')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='6':raise RuntimeError('Expected authorized physical GPU6')
    device=torch.cuda.get_device_properties(0)
    sys.path.insert(0,str(DA if model_name=='da' else VDA))
    if model_name=='da':
        from depth_anything.dpt import DepthAnything
        from depth_anything.util.transform import Resize,NormalizeImage,PrepareForNet
        from torchvision.transforms import Compose
        previous=os.getcwd();os.chdir(DA)
        try:model=DepthAnything(read(WEIGHTS['da'].parent/'config.json'))
        finally:os.chdir(previous)
        transform=Compose([Resize(width=518,height=518,resize_target=False,keep_aspect_ratio=True,ensure_multiple_of=14,resize_method='lower_bound',image_interpolation_method=cv2.INTER_CUBIC),NormalizeImage(mean=[.485,.456,.406],std=[.229,.224,.225]),PrepareForNet()])
    else:
        from video_depth_anything.video_depth import VideoDepthAnything
        model=VideoDepthAnything(encoder='vitl',features=256,out_channels=[256,512,1024,1024],metric=False)
    model.load_state_dict(torch.load(WEIGHTS[model_name],map_location='cpu',weights_only=True),strict=True)
    model=model.eval().cuda()
    info=dict(model=model_name,weight=str(WEIGHTS[model_name]),gpu=str(device),torch_version=torch.__version__,strict_checkpoint_load=True,records=[])
    for cid in cases():
        dest=OUT/'predictions'/model_name/(cid+'.npz')
        if dest.exists():raise FileExistsError(dest)
        frames=rgb(cid);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            if model_name=='da':
                values=[]
                for frame in frames:
                    x=transform({'image':frame.astype(np.float32)/255})['image']
                    pred=model(torch.from_numpy(x).unsqueeze(0).cuda())
                    values.append(torch.nn.functional.interpolate(pred[:,None],size=(360,640),mode='bilinear',align_corners=True)[0,0].cpu().numpy())
                depth=np.stack(values)
            else:depth,_=model.infer_video_depth(frames,target_fps=30,input_size=518,device='cuda',fp32=False)
        torch.cuda.synchronize()
        if depth.shape!=(8,360,640) or not np.isfinite(depth).all():raise ValueError('invalid output')
        dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,relative_inverse_depth=depth.astype(np.float32))
        row=dict(case_id=cid,seconds=time.perf_counter()-start,peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated()/2**30)
        info['records'].append(row);write(OUT/f'{model_name}_execution.json',info);print(model_name,row,flush=True)

def ground_truth():
    # EGL is explicitly bound to GPU6; no default GPU renderer is allowed.
    if os.environ.get('EGL_DEVICE_ID')!='6':raise RuntimeError('Set EGL_DEVICE_ID=6')
    import prepare_context_rgb_pybullet as prep
    pilot,generator,renderer=prep.load_engine()
    records={r['key']:r for r in read(prep.DATA_DEFAULT/'pilot_manifest.json')['records']}
    real_class=renderer.RealismPreviewRenderer
    captures=[]
    class Capture(real_class):
        def __init__(self,*args,**kwargs):
            kwargs.update(capture_depth_frames=True,capture_instance_masks=True)
            super().__init__(*args,**kwargs);captures.append(self)
    renderer.RealismPreviewRenderer=Capture
    for cid in cases():
        case,seed=prep.reconstruct_case(pilot,generator,records[cid]);names=[x.name for x in case.blueprint.objects if not x.metadata.get('visual_only')]
        pos,quat,_=prep.load_observed_pose(prep.DATA_DEFAULT/'samples'/cid,names)
        frames=prep.render_observed_prefix(renderer,case.blueprint,seed,pos,quat,OUT/'gt_rgb'/cid,640,360,'indoor_natural')
        original=rgb(cid);difference=np.abs(frames.astype(float)-original)
        audit=dict(max_rgb_error=float(difference.max()),mean_rgb_error=float(difference.mean()),exact_rgb_equal=bool(np.array_equal(frames,original)))
        write(OUT/'gt'/f'{cid}_audit.json',audit)
        if not audit['exact_rgb_equal']:raise RuntimeError(f'GT render mismatch {cid}: {audit}; stop for review')
        obj=captures.pop();depth=np.stack(obj.depth_frames);mask=np.stack(obj.mask_frames)
        path=OUT/'gt'/f'{cid}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,depth=depth,mask=mask)
        write(OUT/'gt'/f'{cid}_ids.json',obj.instance_ids)
        print('GT_RGB_EXACT',cid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','da','vda','gt']);args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='gt':ground_truth()
    else:infer(args.mode)
