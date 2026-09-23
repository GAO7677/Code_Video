"""Post-freeze paired mask diagnostics and explicit old/new visualization."""
import argparse,json,csv
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform,resize_crop_mask
from test70_mask_diagnostic import mask_stats

def gate(mask):
    pm=resize_crop_mask(mask,crop_transform(mask.shape));cs,_=cv2.findContours(pm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cir=[float(4*np.pi*cv2.contourArea(c)/max(cv2.arcLength(c,True)**2,1)) for c in cs]
    return {'external_components':len(cs),'circularity':cir,'status':'fragmented' if len(cs)!=1 else 'non_sphere' if cir[0]<.65 else 'PASS_2D'}

def run(a):
    for n,h in json.loads((a.root/'prediction_freeze.json').read_text()).items():assert sha256_file(a.root/n)==h
    routes=json.loads((a.root/'evaluation_mapping.json').read_text());records=[];out=a.root/'comparison';out.mkdir(exist_ok=False)
    for route in routes:
        key=route['id'];new=json.loads((a.root/'predictions'/key/'result.json').read_text());nm=None;om=None
        npz=a.root/'predictions'/key/'tracking.npz'
        if npz.exists():
            with np.load(npz) as z:nm=z['masks']
        olddir=a.old/('estimates' if a.pilot else 'predictions')/key
        if (olddir/'tracking.npz').exists():
            with np.load(olddir/'tracking.npz') as z:om=z['masks'];ob=z['prompt']
        old=json.loads((olddir/'result.json').read_text());row={'id':key,'case_id':route.get('case_id',Path(route.get('source_sample','')).name),'status':new['status'],'prompt':new['detection']['prompt'],
            'grounding_counts':new['detection']['all_context_counts'],'old_status':old['status'] if not a.pilot else old.get('state',{}).get('status'),
            'old_mask_exists':om is not None,'new_mask_exists':nm is not None,'frames':[],'new_mean_iou':None,'old_mean_iou':None,'gt_status':'NOT_AVAILABLE'}
        if not a.pilot:
            nd=json.loads((a.root/'evaluation_v2'/key/'details.json').read_text());od=json.loads((a.old/'evaluation_v2'/key/'details.json').read_text())
            row.update(new_mean_iou=nd['mean_iou'],old_mean_iou=od['mean_iou'],gt_status=nd['gt_status'],new_actor=nd.get('matched_gt_actor'),old_actor=od.get('matched_gt_actor'))
            row['paired_comparable']=nd['mean_iou'] is not None and od['mean_iou'] is not None and nd.get('matched_gt_actor')==od.get('matched_gt_actor')
            row['delta_iou']=nd['mean_iou']-od['mean_iou'] if row['paired_comparable'] else None
        dst=out/key;dst.mkdir()
        for t in range(8):
            rgb=np.array(Image.open(a.root/'inputs'/key/f'rgb_{t:02d}.png'));f={'frame':t,'old':mask_stats(om[t]) if om is not None else None,'new':mask_stats(nm[t]) if nm is not None else None}
            if a.pilot:f.update(old_gate=gate(om[t]) if om is not None else None,new_gate=gate(nm[t]) if nm is not None else None)
            if not a.pilot:f.update(old_iou=od.get('frame_iou',[None]*8)[t] if od.get('frame_iou') else None,new_iou=nd.get('frame_iou',[None]*8)[t] if nd.get('frame_iou') else None)
            union=np.zeros(rgb.shape[:2],bool)
            for label,masks,color in [('old',om,[255,150,40]),('new',nm,[255,65,110])]:
                im=rgb.copy()
                if masks is not None:
                    m=masks[t];union|=m;im[m]=(.65*im[m]+.35*np.array(color)).astype(np.uint8)
                    cs,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,cs,-1,color,1)
                else:m=np.zeros(rgb.shape[:2],bool)
                if t==7:
                    boxes=[ob] if label=='old' and om is not None else [x['box'] for x in new['detection']['candidates']] if label=='new' else []
                    for box in boxes:
                        b=np.asarray(box,dtype=int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(60,255,80),1)
                Image.fromarray(im).save(dst/f'{label}_{t}.jpg',quality=94);Image.fromarray(m.astype(np.uint8)*255).save(dst/f'{label}_mask_{t}.png')
            yy,xx=np.where(union)
            f['crop']=[max(0,int(xx.min())-16),max(0,int(yy.min())-16),min(rgb.shape[1],int(xx.max())+17),min(rgb.shape[0],int(yy.max())+17)] if len(xx) else [0,0,rgb.shape[1],rgb.shape[0]]
            row['frames'].append(f)
        if a.pilot:
            for label in ['old','new']:
                statuses=[f[f'{label}_gate']['status'] if f[f'{label}_gate'] else 'NOT_RUN' for f in row['frames']]
                row[label+'_2d_status']=next((s for s in statuses if s!='PASS_2D'),'PASS_2D')
        dump_json(dst/'details.json',row);records.append(row)
    summary={'cases':len(records),'new_executed':sum(r['status']=='EXECUTED' for r in records),'old_masks':sum(r['old_mask_exists'] for r in records),
        'pilot':a.pilot,'no_mask_repair':True,'physics':'NOT_RUN','state_fit':'NOT_RUN'}
    if a.pilot:
        for label in ['old','new']:summary[label+'_2d_status']=dict(Counter(r[label+'_2d_status'] for r in records))
    else:
        pairs=[r for r in records if r['paired_comparable']];summary['paired_count']=len(pairs)
        for label in ['old','new']:summary[label+'_paired_mean_iou']=float(np.mean([r[label+'_mean_iou'] for r in pairs])) if pairs else None
        summary['improved_gt_0p01']=sum(r['delta_iou']>.01 for r in pairs);summary['regressed_lt_minus0p01']=sum(r['delta_iou']<-.01 for r in pairs);summary['stable_within_0p01']=sum(abs(r['delta_iou'])<=.01 for r in pairs)
        good=[r for r in pairs if r['old_mean_iou']>=.8];summary['previous_good_count']=len(good);summary['previous_good_still_ge0p8']=sum(r['new_mean_iou']>=.8 for r in good)
        summary['new_scorable_count']=sum(r['new_mean_iou'] is not None for r in records);summary['new_scorable_mean_iou']=float(np.mean([r['new_mean_iou'] for r in records if r['new_mean_iou'] is not None]))
    dump_json(a.root/'comparison_data.json',{'summary':summary,'records':records})
    fields=['id','case_id','status','prompt','old_mean_iou','new_mean_iou','delta_iou','gt_status','old_2d_status','new_2d_status']
    with (out/'per_case.csv').open('w') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(records)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args();cv2.setNumThreads(2);run(a)
