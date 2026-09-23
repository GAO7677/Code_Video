"""Explicit group-text tuning; RGB only; retain all raw Grounding DINO boxes."""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from context_rgb_pybullet_common import dump_json,sha256_file

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-7f6fbc40-3594-2c34-8557-422621355ff9':raise ValueError('GPU6 required')
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,'/home/gaoya/Grounded-SAM-2-main')
    from grounding_dino.groundingdino import _C
    from grounding_dino.groundingdino.models import build_model
    from grounding_dino.groundingdino.util.slconfig import SLConfig
    from grounding_dino.groundingdino.util.misc import clean_state_dict
    from grounding_dino.groundingdino.util.utils import get_phrases_from_posmap
    import grounding_dino.groundingdino.datasets.transforms as T
    from torchvision.ops import box_convert
    cfgpath=Path('/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/GroundingDINO_SwinT_OGC.cfg.py')
    weights=Path('/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/groundingdino_swint_ogc.pth')
    c=json.loads(a.config.read_text());groups=[g for g in c['groups'] if not a.groups or g['id'] in a.groups]
    assert groups and all(len(g['terms'])==1 for g in groups)
    a.output.mkdir(parents=True,exist_ok=False)
    dump_json(a.output/'protocol.json',{'config':c,'frames':a.frames,'groups':[g['id'] for g in groups],
        'thresholds':{'box':.35,'text':.25},'nms':False,'top1':False,'gt_read':False,'engine':'local Grounded-SAM-2 original GDINO CUDA extension',
        'config_sha256':sha256_file(a.config),'weights_sha256':sha256_file(weights),'model_config_sha256':sha256_file(cfgpath)})
    cfg=SLConfig.fromfile(str(cfgpath));cfg.device='cuda:0';model=build_model(cfg)
    ckpt=torch.load(weights,map_location='cpu',weights_only=False)
    incompatible=model.load_state_dict(clean_state_dict(ckpt['model']),strict=False)
    dump_json(a.output/'load_audit.json',{'missing_keys':incompatible.missing_keys,'unexpected_keys':incompatible.unexpected_keys})
    if incompatible.missing_keys:raise ValueError('Model weights missing keys; no partial inference')
    del ckpt
    model=model.eval().to('cuda:0')
    transform=T.Compose([T.RandomResize([800],max_size=1333),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    rows=[];start=time.perf_counter()
    for g in groups:
        tiles=[];caption=g['grounding_text'];tokenized=model.tokenizer(caption)
        for clip in g['anonymous_clip_ids']:
            for t in a.frames:
                src=a.inputs/clip/f'rgb_{t:02d}.png';im=Image.open(src).convert('RGB');w,h=im.size
                tensor,_=transform(im,None);ts=time.perf_counter()
                with torch.inference_mode():output=model(tensor[None].to('cuda:0'),captions=[caption])
                logits=output['pred_logits'][0].sigmoid().cpu();boxes=output['pred_boxes'][0].cpu();scores=logits.max(-1).values;keep=scores>.35
                xyxy=box_convert(boxes[keep],in_fmt='cxcywh',out_fmt='xyxy')*torch.tensor([w,h,w,h]);ss=scores[keep]
                phrases=[get_phrases_from_posmap(l>.25,tokenized,model.tokenizer) for l in logits[keep]]
                row={'id':clip,'frame':t,'group':g['id'],'prompt':caption,'boxes':xyxy.tolist(),'scores':ss.tolist(),'phrases':phrases,
                    'count':int(keep.sum()),'count_status':'UNIQUE' if int(keep.sum())==1 else 'NO_DETECTION' if not keep.any() else 'MULTIPLE',
                    'identity_review':'PENDING','rgb_sha256':sha256_file(src),'inference_s':time.perf_counter()-ts}
                rows.append(row);dst=a.output/g['id']/clip;dst.mkdir(parents=True,exist_ok=True);dump_json(dst/f'rgb_{t:02d}.json',row)
                d=ImageDraw.Draw(im)
                for b,s in zip(xyxy.tolist(),ss.tolist()):d.rectangle(b,outline='lime',width=2);d.text((b[0],max(0,b[1]-12)),f'{s:.3f}',fill='yellow')
                im.save(dst/f'rgb_{t:02d}.jpg',quality=94)
                tile=im.resize((448,256));canvas=Image.new('RGB',(448,282),'#182132');canvas.paste(tile,(0,26));ImageDraw.Draw(canvas).text((5,5),f'{clip} RGB{t} | {caption} | {row["count_status"]}',fill='white');tiles.append(canvas)
                print(clip,t,caption,row['count'],[round(x,3) for x in row['scores']],flush=True)
        cols=2 if len(a.frames)==1 else len(a.frames);montage=Image.new('RGB',(448*cols,282*((len(tiles)+cols-1)//cols)),'#182132')
        for i,tile in enumerate(tiles):montage.paste(tile,((i%cols)*448,(i//cols)*282))
        montage.save(a.output/f'{g["id"]}.jpg',quality=92)
    dump_json(a.output/'results.json',rows)
    dump_json(a.output/'summary.json',{'frames':len(rows),'unique':sum(r['count']==1 for r in rows),'zero':sum(r['count']==0 for r in rows),'multiple':sum(r['count']>1 for r in rows),'elapsed_s':time.perf_counter()-start,'identity_review':'PENDING'})
    dump_json(a.output/'freeze.json',{str(p.relative_to(a.output)):sha256_file(p) for p in sorted(a.output.rglob('*.json'))})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--inputs',type=Path,default=Path('/data/gaoya/agent-data/outputs/test70_motion_sam2_20260923_v1/inputs'));p.add_argument('--output',type=Path,required=True);p.add_argument('--frames',type=int,nargs='+',default=[7]);p.add_argument('--groups',nargs='*');run(p.parse_args())
