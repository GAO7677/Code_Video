"""Shape-agnostic motion-candidate + SAM2 diagnostic, frozen before GT evaluation."""
import argparse,json,os,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from context_rgb_pybullet_common import dump_json,sha256_file

DATA=Path('/data/gaoya/AAA_test_video/physv_v2v_0819')
GPU='GPU-7f6fbc40-3594-2c34-8557-422621355ff9'

def detect(rgb):
    """Same thresholds/ranking as current motion_prompt, preserving rejected audits."""
    gray=np.stack([cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb]).astype(np.float32)
    shifts=[cv2.phaseCorrelate(gray[0],x)[0] for x in gray[1:]]
    change=abs(gray[-1]-np.median(gray[:-1],axis=0));med=np.median(change);mad=np.median(abs(change-med))
    threshold=max(12.,float(med+6*1.4826*mad))
    binary=cv2.morphologyEx((change>threshold).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(binary);candidates=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i])
        if area<12 or area>change.size*.1:continue
        candidates.append({'box':[max(0,x-w*.2),max(0,y-h*.2),min(rgb.shape[2]-1,x+w*1.2),min(rgb.shape[1]-1,y+h*1.2)],
                           'score':float(change[labels==i].sum()),'area':area})
    candidates.sort(key=lambda x:x['score'],reverse=True)
    audit={'threshold':threshold,'translation_shifts':shifts,'candidates':candidates,'shape_prior':False,'selected':None,'status':'ACCEPTED'}
    if np.max(np.linalg.norm(shifts,axis=1))>3:audit.update(status='UNKNOWN',reason='camera_motion_exceeds_translation_gate')
    elif not candidates:audit.update(status='FAIL',reason='no_motion_candidate')
    elif len(candidates)>1 and candidates[1]['score']>.8*candidates[0]['score']:audit.update(status='UNKNOWN',reason='multiple_motion_candidates')
    else:audit['selected']=0
    return audit,change

def prepare(root):
    root.mkdir(exist_ok=False)
    listing=DATA/'testjsons/physv_v2v_0819_all_cycles_test70_ctx8.txt'
    paths=[Path(p) for p in listing.read_text().splitlines() if p.strip()];assert len(paths)==len(set(paths))==70
    mapping=[]
    protocol={'date':'2026-09-23','selection':'all 70 entries in official test70 list, exact order, no replacement',
              'input':'anonymous RGB0..7 at native resolution plus timestamps; curator passes no captions/names/GT',
              'detector':'existing RGB7 vs median RGB0..6 motion components; no shape prior',
              'detector_parameters':{'minimum_difference':12,'mad_multiplier':6*1.4826,'closing_kernel':3,'min_component_pixels':12,
                'max_area_fraction':.1,'box_expansion_per_side':.2,'ambiguity_ratio':.8,'camera_translation_gate_px':3},
              'sam2':'existing SAM2.1 Hiera large; RGB7 box; forward/reverse tracking; no mask shape/fragment rejection or repair',
              'target_policy':'one dominant motion target; ambiguous multiple targets UNKNOWN, no GT choice',
              'downstream':'no sphere fitting, VGGT, Bullet, Predictor, Utonia or DiT',
              'evaluation':'freeze first; match selected target to one GT actor via RGB7 mask IoU, fixed actor across frames; no future frames',
              'quality':'descriptive native resolution, GT target pixel area/extent, Laplacian variance and contrast; no causal claim from correlation',
              'eval_bands':{'mean_IoU_good':.8,'mean_IoU_usable':.5,'very_small_equivalent_diameter_px':20},
              'gpu_uuid':GPU,'cpu_threads':2}
    dump_json(root/'protocol.json',protocol)
    for i,path in enumerate(paths):
        # Curator reads routing only; metadata is not passed to inference.
        payload=json.loads(path.read_text());video=Path(payload['input_video_8f']);key=f'clip_{i:03d}';dest=root/'inputs'/key;dest.mkdir(parents=True)
        cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
        if fps<=0:raise ValueError(f'missing fps {key}')
        for t in range(8):
            ok,bgr=cap.read()
            if not ok:raise ValueError(f'missing RGB{t} {key}')
            Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(dest/f'rgb_{t:02d}.png')
        cap.release();dump_json(dest/'timestamps.json',{'time_s':(np.arange(8)/fps).tolist(),'fps':fps})
        mapping.append({'id':key,'routing_json':str(path),'source_context':str(video),'source_sample':str(video.parent.parent),
                        'source_context_sha256':sha256_file(video)})
    dump_json(root/'evaluation_mapping.json',mapping)
    files=[root/'protocol.json',*sorted((root/'inputs').glob('*/*'))]
    dump_json(root/'input_freeze.json',{str(p.relative_to(root)):sha256_file(p) for p in files})
    print('PREPARED 70',flush=True)

def mask_stats(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8));areas=stats[1:,cv2.CC_STAT_AREA].astype(int)
    y,x=np.where(mask)
    return {'area_pixels':int(mask.sum()),'components':n-1,'component_areas':sorted(areas.tolist(),reverse=True),
            'largest_component_fraction':float(areas.max()/areas.sum()) if len(areas) else None,
            'bbox':[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None}

def infer(root):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('authorized GPU6 UUID required')
    for n,h in json.loads((root/'input_freeze.json').read_text()).items():
        if sha256_file(root/n)!=h:raise ValueError('input hash changed')
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,'/home/gaoya/Code_Video/phys_state_video/src')
    from phys_state_video.mask_tracking import SAM2VideoMaskTracker
    tracker=SAM2VideoMaskTracker(device='cuda:0',model_id=None,
        model_cfg='/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml',
        checkpoint_path=Path('/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt'))
    out=root/'predictions';out.mkdir(exist_ok=False);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for folder in sorted((root/'inputs').iterdir()):
        dst=out/folder.name;dst.mkdir();rgb=np.stack([np.array(Image.open(folder/f'rgb_{t:02d}.png').convert('RGB')) for t in range(8)])
        audit,change=detect(rgb);row={'id':folder.name,'resolution_wh':[rgb.shape[2],rgb.shape[1]],'detection':audit,
            'status':'NOT_RUN','fallback_used':False,'shape_prior':False,'mask_postprocessing':False}
        heat=cv2.applyColorMap(np.uint8(np.clip(change/max(float(change.max()),1)*255,0,255)),cv2.COLORMAP_INFERNO)
        Image.fromarray(heat[:,:,::-1]).save(dst/'motion.png')
        if audit['status']=='ACCEPTED':
            box=np.array(audit['candidates'][0]['box'],np.float32);started=time.perf_counter()
            try:
                masks=tracker.track_from_boxes(rgb.transpose(0,3,1,2).astype(np.float32)/255,prompt_frame_idx=7,boxes_xyxy=box[None])[:,0].astype(bool)
                np.savez_compressed(dst/'tracking.npz',masks=masks,prompt=box)
                row.update(status='EXECUTED',sam2_seconds=time.perf_counter()-started,frames=[mask_stats(m) for m in masks])
                row['empty_frames']=sum(not m.any() for m in masks)
            except (RuntimeError,ValueError) as exc:
                row.update(status='FAIL',reason=str(exc));torch.cuda.empty_cache()
        else:row.update(status=audit['status'],reason=audit['reason'])
        dump_json(dst/'result.json',row);print(folder.name,row['status'],row.get('reason',''),flush=True)
    dump_json(root/'runtime.json',{'elapsed_s':time.perf_counter()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                                  'gpu_uuid':GPU,'cpu_threads':2,'sam2_weights_sha256':sha256_file(Path('/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt'))})
    files=[root/'runtime.json',root/'protocol.json',*sorted(out.glob('*/*'))]
    dump_json(root/'prediction_freeze.json',{str(p.relative_to(root)):sha256_file(p) for p in files})
    print('FROZEN',sha256_file(root/'prediction_freeze.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','infer']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2)
    (prepare if a.phase=='prepare' else infer)(a.root)
