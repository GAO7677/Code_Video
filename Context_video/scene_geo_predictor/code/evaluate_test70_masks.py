"""Post-freeze actor-matched mask diagnostics and native-resolution overlay artifacts."""
import argparse,json,csv
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from context_rgb_pybullet_common import dump_json,sha256_file

DATA=Path('/data/gaoya/AAA_test_video/physv_v2v_0819')

def contour_list(mask):
    return [c[:,0].tolist() for c in cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]]

def bbox(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)] if len(x) else None

def axis_audit(masks,centers):
    errors=[[],[]]
    for j,actor in enumerate(masks):
        for t,mask in enumerate(actor):
            if not mask.any():continue
            y,x=np.where(mask);point=np.array([x.mean(),y.mean()]);target=centers[t,j,:2]
            if not np.isfinite(target).all():continue
            errors[0].append(float(np.linalg.norm(point-target)))
            point[1]=mask.shape[0]-1-point[1];errors[1].append(float(np.linalg.norm(point-target)))
    if not errors[0]:raise ValueError('GT axis audit has no visible actors')
    scores=[float(np.median(x)) for x in errors]
    if abs(scores[0]-scores[1])<1:raise ValueError(f'GT axis ambiguous: {scores}')
    flip=scores[1]<scores[0]
    return masks[:,:,::-1,:].copy() if flip else masks,{'transform':'vertical_flip' if flip else 'identity',
             'raw_centroid_projection_median_px':scores[0],'flipped_centroid_projection_median_px':scores[1],
             'selection':'GT masks vs saved GT projected centers only; independent of predicted masks'}

def run(root):
    for n,h in json.loads((root/'prediction_freeze.json').read_text()).items():
        if sha256_file(root/n)!=h:raise ValueError('prediction freeze mismatch')
    out=root/'evaluation_v2';out.mkdir(exist_ok=False);records=[]
    for route in json.loads((root/'evaluation_mapping.json').read_text()):
        key=route['id'];folder=root/'predictions'/key;result=json.loads((folder/'result.json').read_text());sample=Path(route['source_sample']);case=sample.name
        rgb=np.stack([np.array(Image.open(root/'inputs'/key/f'rgb_{t:02d}.png').convert('RGB')) for t in range(8)])
        row={'id':key,'case_id':case,'status':result['status'],'reason':result.get('reason'),'resolution_wh':result['resolution_wh'],
             'target_detection':result['detection'],'gt_status':'BLOCKED','mean_iou':None,'frames':[],
             'predicted_shape':'raw mask only; no class or shape prior','mask_frames':result.get('frames'),'fallback_used':False}
        pred=None
        if (folder/'tracking.npz').exists():
            with np.load(folder/'tracking.npz') as a:pred=a['masks']
        truth=DATA/'physv_v2v_0819_cycles_aligned_truth_v1/cases'/case
        gm=None;selected=None
        try:
            with np.load(truth/'dynamic_masks.npz',allow_pickle=False) as a:gm=a['masks_thw'][:,:8];names=list(a['object_names'].astype(str))
            with np.load(truth/'trajectory_pixels.npz',allow_pickle=False) as a:centers=a['centers_tnc'][:8];center_names=list(a['object_names'].astype(str))
            if names!=center_names or gm.shape[2:]!=rgb.shape[1:3]:raise ValueError('GT shape or name mismatch')
            gm,axis=axis_audit(gm,centers);row['gt_axis_audit']=axis
            row['gt_status']='EVALUATED';row['gt_actor_names']=names;row['gt_actor_count']=len(names)
            # Matching is evaluation-only; freeze the identity at the prompt frame.
            if pred is not None:
                scores=[float(np.sum(pred[7]&m[7])/max(np.sum(pred[7]|m[7]),1)) for m in gm]
                selected=int(np.argmax(scores));row['matched_gt_actor']=names[selected];row['prompt_frame_actor_ious']=scores
                row['identity_match_uncertain']=max(scores)<.1
                target=gm[selected];ious=[];recalls=[];precisions=[];quality=[]
                b=result['detection']['candidates'][0]['box'];x0,y0,x1,y1=np.array(b).astype(int)
                promptmask=np.zeros(target[7].shape,bool);promptmask[max(0,y0):max(0,y1),max(0,x0):max(0,x1)]=True
                row['prompt_gt_coverage']=float(np.sum(promptmask&target[7])/max(target[7].sum(),1))
                row['prompt_area_over_gt_area']=float(promptmask.sum()/max(target[7].sum(),1))
                for t,(pm,gt) in enumerate(zip(pred,target)):
                    intersection=int(np.sum(pm&gt));union=int(np.sum(pm|gt));iou=intersection/union if union else None
                    precision=intersection/int(pm.sum()) if pm.any() else 0.;recall=intersection/int(gt.sum()) if gt.any() else None
                    ious.append(iou);precisions.append(precision);recalls.append(recall)
                    box=bbox(gt);q={'gt_area_px':int(gt.sum()),'gt_equivalent_diameter_px':float(2*np.sqrt(gt.sum()/np.pi)),'gt_bbox':box}
                    if box:
                        x,y,z,w=box;pad=4;gray=cv2.cvtColor(rgb[t],cv2.COLOR_RGB2GRAY)
                        roi=gray[max(0,y-pad):min(gray.shape[0],w+pad),max(0,x-pad):min(gray.shape[1],z+pad)]
                        q.update(laplacian_variance=float(cv2.Laplacian(roi,cv2.CV_64F).var()),intensity_std=float(roi.std()),bbox_min_side_px=min(z-x,w-y))
                    quality.append(q)
                valid=[x for x in ious if x is not None]
                row.update(mean_iou=float(np.mean(valid)) if valid else None,frame_iou=ious,
                           mean_precision=float(np.mean(precisions)),mean_recall=float(np.mean([x for x in recalls if x is not None])) if any(x is not None for x in recalls) else None,
                           quality_frames=quality)
                row['quality']={field:float(np.median([q[field] for q in quality if field in q])) for field in ['gt_area_px','gt_equivalent_diameter_px','laplacian_variance','intensity_std','bbox_min_side_px'] if any(field in q for q in quality)}
                row['diagnostic_category']=('low_overlap_identity_uncertain' if row['identity_match_uncertain'] else
                    'prompt_partial_coverage' if row['prompt_gt_coverage']<.5 else
                    'good_mask' if row['mean_iou'] is not None and row['mean_iou']>=.8 else
                    'mask_error_despite_target_covered')
            else:row['diagnostic_category']='detection_not_admitted'
            meta=json.loads((sample/'metadata.json').read_text());row['eval_group']=str(meta.get('source_group'))+'/'+str(meta.get('task_type'))+'/'+str(meta.get('control',{}).get('variable'))
            row['unannotated_dynamic_actors']=sorted(k for k,v in meta.get('actors',{}).items() if v.get('dynamic') and k not in names)
            if row['unannotated_dynamic_actors'] and row.get('identity_match_uncertain'):
                row['unmatched_annotation_overlap_diagnostic']={k:row.get(k) for k in ['matched_gt_actor','mean_iou','frame_iou','mean_precision','mean_recall']}
                for k in ['mean_iou','frame_iou','mean_precision','mean_recall','matched_gt_actor','prompt_gt_coverage','prompt_area_over_gt_area']:row[k]=None
                row.update(gt_status='BLOCKED_TARGET_COVERAGE',gt_reason='Dynamic actors missing from GT masks and no confident annotated identity match; cannot score selected target.',diagnostic_category='GT_target_coverage_incomplete')
                selected=None
        except (FileNotFoundError,ValueError,KeyError) as exc:
            gm=None;row.update(gt_status='BLOCKED',gt_reason=str(exc),diagnostic_category='GT_evaluation_blocked')
        dst=out/key;dst.mkdir()
        for t,image in enumerate(rgb):
            painted=image.copy();frame={'pred_contours':[],'gt_contours':[],'gt_others':[]}
            if pred is not None:
                Image.fromarray(pred[t].astype(np.uint8)*255).save(dst/f'mask_{t}.png')
                frame['pred_contours']=contour_list(pred[t]);painted[pred[t]]=(.72*painted[pred[t]]+.28*np.array([255,60,90])).astype(np.uint8)
                cs,_=cv2.findContours(pred[t].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(painted,cs,-1,(255,70,100),1)
            if gm is not None:
                for j,m in enumerate(gm):
                    cs=contour_list(m[t]);frame['gt_contours' if j==selected else 'gt_others'].extend(cs)
                    contours=[np.array(c,dtype=np.int32)[:,None,:] for c in cs];cv2.drawContours(painted,contours,-1,(40,240,110) if j==selected else (160,180,230),1)
            if t==7:
                for j,c in enumerate(result['detection']['candidates'][:5]):
                    b=np.array(c['box']).astype(int);cv2.rectangle(painted,tuple(b[:2]),tuple(b[2:]),(255,180,30) if result['detection']['selected']==j else (160,160,160),1)
            # crop follows predicted object, GT only if prediction absent (explicit eval visualization).
            cropmask=pred[t] if pred is not None and pred[t].any() else (gm[:,t].any(axis=0) if gm is not None else None)
            box=bbox(cropmask) if cropmask is not None else None
            if box:
                x,y,z,w=box;pad=max(10,int(.15*max(z-x,w-y)));box=[max(0,x-pad),max(0,y-pad),min(image.shape[1],z+pad),min(image.shape[0],w+pad)]
            frame['crop']=box;frame['crop_source']='predicted_mask' if pred is not None and pred[t].any() else 'GT_visualization_only_or_full_frame'
            row['frames'].append(frame)
            Image.fromarray(painted).save(dst/f'overlay_{t}.jpg',quality=92)
        dump_json(dst/'details.json',row);records.append(row);print('EVAL',key,row['status'],row['mean_iou'],row.get('gt_reason',''),flush=True)
    valid=[r for r in records if r['mean_iou'] is not None];executed=[r for r in records if r['status']=='EXECUTED']
    def stats(rs):
        vals=[r['mean_iou'] for r in rs if r['mean_iou'] is not None]
        return {'count':len(rs),'evaluated_masks':len(vals),'mean_iou':float(np.mean(vals)) if vals else None,'median_iou':float(np.median(vals)) if vals else None}
    summary={'total':len(records),'status_counts':dict(Counter(r['status'] for r in records)),
             'detector_failures':dict(Counter(r['reason'] for r in records if r['status']!='EXECUTED')),
             'gt_status_counts':dict(Counter(r['gt_status'] for r in records)), 'mask_quality_conditional':stats(valid),
             'iou_bands':{'ge_0p8':sum(r['mean_iou']>=.8 for r in valid),'0p5_to_0p8':sum(.5<=r['mean_iou']<.8 for r in valid),'lt_0p5':sum(r['mean_iou']<.5 for r in valid)},
             'diagnostic_categories':dict(Counter(r['diagnostic_category'] for r in records)),
             'multi_gt_actor_cases':sum(r.get('gt_actor_count',0)>1 for r in records),
             'note':'single dominant target; GT actor matching for evaluation only. No mask-shape rejection; no causal image-quality conclusion.'}
    summary['by_resolution']={str(size):stats([r for r in records if str(r['resolution_wh'])==size]) for size in sorted({str(r['resolution_wh']) for r in records})}
    summary['by_target_size']={'equivalent_diameter_lt20':stats([r for r in valid if r['quality']['gt_equivalent_diameter_px']<20]),
                              'equivalent_diameter_ge20':stats([r for r in valid if r['quality']['gt_equivalent_diameter_px']>=20])}
    summary['quality_correlations']={}
    for field in ['gt_equivalent_diameter_px','laplacian_variance','intensity_std']:
        pairs=[(r['quality'][field],r['mean_iou']) for r in valid if field in r['quality']]
        rho=spearmanr(np.array(pairs)[:,0],np.array(pairs)[:,1]).statistic if len(pairs)>2 else float('nan')
        summary['quality_correlations'][field]={'n':len(pairs),'spearman_rho':float(rho) if np.isfinite(rho) else None,'causal':False}
    groups=defaultdict(list)
    for r in records:groups[r.get('eval_group','UNKNOWN')].append(r)
    summary['groups']={k:stats(v) for k,v in groups.items()}
    dump_json(out/'summary.json',summary)
    dump_json(root/'viewer_data_v2.json',{'summary':summary,'records':[{k:v for k,v in r.items() if k not in ['frames','target_detection','quality_frames','mask_frames']} for r in records]})
    fields=['id','case_id','status','reason','gt_status','mean_iou','mean_precision','mean_recall','prompt_gt_coverage','diagnostic_category','matched_gt_actor','eval_group']
    with (out/'per_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    print(json.dumps({k:v for k,v in summary.items() if k!='groups'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2);run(a.root)
