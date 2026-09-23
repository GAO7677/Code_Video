"""Reuse identical SAM2 tracker with strictly unique RGB7 text box; no GT inputs."""
import argparse,json,os,sys,time,shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from context_rgb_pybullet_common import dump_json,sha256_file
from test70_mask_diagnostic import mask_stats,GPU,detect

def run(a):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU
    import torch
    torch.set_num_threads(2);cv2.setNumThreads(2)
    for n,h in json.loads((a.grounding/'freeze.json').read_text()).items():
        assert sha256_file(a.grounding/n)==h, n
    rows=json.loads((a.grounding/'results.json').read_text());ids=sorted({r['id'] for r in rows})
    assert len(ids)==a.expected
    a.output.mkdir(exist_ok=True)
    (a.output/'inputs').symlink_to(a.source/'inputs',target_is_directory=True)
    shutil.copyfile(a.source/'evaluation_mapping.json',a.output/'evaluation_mapping.json')
    dump_json(a.output/'protocol.json',{'date':'2026-09-23','source':str(a.source),'grounding':str(a.grounding),'cases':a.expected,
        'detector':'single declared text phrase per group; RGB7 exactly one raw box; no top1/NMS/motion fallback',
        'sam2':'same SAM2.1 Hiera large local tracker, RGB7 box, bidirectional context propagation',
        'input':'RGB0-7 and declared group text only; GT is evaluation-only after freeze','shape_gate':False,'mask_postprocessing':False,
        'vg gt/physics':'NOT_RUN','grounding_freeze_sha256':sha256_file(a.grounding/'freeze.json')})
    sys.path.insert(0,'/home/gaoya/Code_Video/phys_state_video/src')
    from phys_state_video.mask_tracking import SAM2VideoMaskTracker
    tracker=SAM2VideoMaskTracker(device='cuda:0',model_id=None,model_cfg='/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml',checkpoint_path=Path('/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt'))
    out=a.output/'predictions';out.mkdir(exist_ok=False);start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    hashes={}
    for key in ids:
        rr=sorted([r for r in rows if r['id']==key],key=lambda r:r['frame']);assert [r['frame'] for r in rr]==list(range(8))
        imgs=[]
        for r in rr:
            src=a.source/'inputs'/key/f'rgb_{r["frame"]:02d}.png';h=sha256_file(src);assert h==r['rgb_sha256'];hashes[str(src)]=h;imgs.append(np.array(Image.open(src).convert('RGB')))
        rgb=np.stack(imgs);r=rr[7];dst=out/key;dst.mkdir()
        candidates=[{'box':b,'score':s} for b,s in zip(r['boxes'],r['scores'])]
        audit={'candidates':candidates,'selected':0 if len(candidates)==1 else None,'prompt':r['prompt'],'source':'Grounding DINO RGB7','all_context_counts':[x['count'] for x in rr]}
        result={'id':key,'status':'UNKNOWN','reason':'non_unique_grounding_box','resolution_wh':[rgb.shape[2],rgb.shape[1]],'detection':audit,'fallback_used':False,'shape_prior':False,'mask_postprocessing':False}
        _,change=detect(rgb) # visualization only: never used to select text boxes
        heat=cv2.applyColorMap(np.uint8(np.clip(change/max(float(change.max()),1)*255,0,255)),cv2.COLORMAP_INFERNO)
        Image.fromarray(heat[:,:,::-1]).save(dst/'motion.png')
        if len(candidates)==1:
            box=np.asarray(candidates[0]['box'],np.float32);ts=time.perf_counter()
            try:
                masks=tracker.track_from_boxes(rgb.transpose(0,3,1,2).astype(np.float32)/255,prompt_frame_idx=7,boxes_xyxy=box[None])[:,0].astype(bool)
                np.savez_compressed(dst/'tracking.npz',masks=masks,prompt=box)
                result.update(status='EXECUTED',reason=None,sam2_seconds=time.perf_counter()-ts,frames=[mask_stats(m) for m in masks],empty_frames=sum(not m.any() for m in masks))
            except (RuntimeError,ValueError) as exc:result.update(status='FAIL',reason=str(exc));torch.cuda.empty_cache()
        dump_json(dst/'result.json',result);print(key,result['status'],result.get('reason'),flush=True)
    dump_json(a.output/'input_rgb_hashes.json',hashes)
    dump_json(a.output/'runtime.json',{'elapsed_s':time.perf_counter()-start,'gpu_uuid':GPU,'cpu_threads':2,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})
    dump_json(a.output/'prediction_freeze.json',{str(p.relative_to(a.output)):sha256_file(p) for p in [a.output/'protocol.json',a.output/'runtime.json',a.output/'input_rgb_hashes.json',*sorted(out.glob('*/*'))]})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--grounding',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=int,required=True);run(p.parse_args())
