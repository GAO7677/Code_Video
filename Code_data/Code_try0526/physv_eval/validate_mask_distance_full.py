"""CPU rescore of cached, completed SAM2 extractions; never overwrites old metrics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from .single_case_rigidbench.mask_distance_full import POLICY_VERSION, score_masks, summarize


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def _evaluate(config_path):
    cfg=json.loads(config_path.read_text());out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    expected={'policy_version':POLICY_VERSION,'unilateral_empty':'maximum_pixel_distance',
              'both_empty':'zero_if_observation_valid','invalid_observation':'error',
              'normalization':'height','chamfer_reduction':'sum_of_two_directed_means',
              'full_reduction':'equal_actor_frame_mean'}
    for key,value in expected.items():
        if cfg.get(key)!=value:raise ValueError(f'Unsupported policy {key}: {cfg.get(key)}')
    sys.path.insert(0,str(Path(cfg['official_root'])/'src'))
    from rigidbench.eval.score.mask import l2_per_frame,chamfer_per_frame,iou_per_frame
    official={'l2_legacy':l2_per_frame,'chamfer_legacy':chamfer_per_frame,'iou':iou_per_frame}
    rows=[];checks=[];sources=[]
    names=[x['name'] for x in cfg['cases']]
    if len(names)!=len(set(names)):raise ValueError('Duplicate sample names')
    for case in cfg['cases']:
        name=case['name'];dest=out/name;dest.mkdir(exist_ok=True)
        frame_count=case['expected_frames']
        if sha256(case['video'])!=case['video_sha256']:raise ValueError(f'{name}: source video changed')
        with np.load(case['pred_masks'],allow_pickle=False) as f:pred=f['masks']
        with np.load(case['gt_masks'],allow_pickle=False) as f:
            all_gt=f['masks']
            active=case['active_actor_indices']
            if not active or len(active)!=len(set(active)) or any(i<0 or i>=all_gt.shape[1] for i in active):
                raise ValueError('Invalid active actor indices')
            if all_gt.shape[0]<frame_count:raise ValueError('Incomplete GT sequence')
            gt=all_gt[:frame_count,active]
        if pred.shape!=gt.shape or pred.shape[0]!=frame_count:raise ValueError(f'{name}: incomplete or mismatched masks')
        extraction=list(csv.DictReader(open(case['extraction_csv'])))
        if [int(x['frame']) for x in extraction]!=list(range(frame_count)):
            raise ValueError(f'{name}: missing or duplicate extraction frames')
        # These saved six tests are single-actor. Multi-actor math is tested independently.
        if pred.shape[1]!=1:raise ValueError('This extraction CSV contract is single-actor only')
        np.testing.assert_array_equal(pred.sum(axis=(1,2,3)),[int(x['pred_pixels']) for x in extraction])
        np.testing.assert_array_equal(gt.sum(axis=(1,2,3)),[int(x['gt_pixels']) for x in extraction])
        scores=score_masks(gt,pred,observed=np.ones(pred.shape[:2],dtype=bool))
        arrays={k:v for k,v in scores.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(dest/'per_actor.npz',**arrays)
        oracle={}
        for key,fn in official.items():
            oracle[key]=np.stack([fn(gt[:,n:n+1],pred[:,n:n+1]) for n in range(pred.shape[1])],axis=1)
            np.testing.assert_allclose(scores[key],oracle[key],rtol=1e-12,atol=1e-12,equal_nan=True)
        present=scores['gt_present'] & scores['pred_present']
        per_actor=[];per_frame=[]
        for t in range(frame_count):
            for n in range(pred.shape[1]):
                record={'frame':t,'actor_index':active[n],
                        'gt_present':bool(scores['gt_present'][t,n]),'pred_present':bool(scores['pred_present'][t,n])}
                for k in ('l2_legacy','chamfer_legacy','l2_full','chamfer_full','iou'):
                    v=float(scores[k][t,n]);record[k]=v if np.isfinite(v) else None
                per_actor.append(record)
            per_frame.append({'frame':t,**summarize(scores,t,t+1)})
        write_csv(dest/'per_actor.csv',per_actor);write_csv(dest/'per_frame.csv',per_frame)
        for window,(start,stop) in cfg['windows'].items():
            aggregate=summarize(scores,start,stop)
            # Independent official aggregation, including its two-stage NaN reduction.
            for key in official:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    old=float(np.nanmean(np.nanmean(oracle[key][start:stop],axis=1)))
                if np.isnan(old):assert aggregate[key] is None
                else:np.testing.assert_allclose(aggregate[key],old,rtol=1e-12,atol=1e-12)
            rows.append({'name':name,'case_id':case['case_id'],'method':case['method'],'noise':case['noise'],'window':window,**aggregate})
        for key in ('l2','chamfer'):
            # Valid observations preserve old formulas exactly.
            np.testing.assert_allclose(scores[key+'_full'][present],scores[key+'_legacy'][present],rtol=0,atol=0)
            assert np.isfinite(scores[key+'_full']).all()
            assert np.all(scores[key+'_full']>=0)
            assert np.all(scores[key+'_full']<=scores[key+'_penalty']+1e-12)
        # Destructively erase predictions only on frames where GT exists.
        erased=pred.copy();erased[scores['gt_present']]=False
        erased_scores=score_masks(gt,erased)
        for key in ('l2_full','chamfer_full'):
            assert np.all(erased_scores[key]>=scores[key]-1e-12)
        if scores['gt_present'].all():
            for key in ('l2','chamfer'):
                aggregate=summarize(scores)
                valid_sum=float(np.nansum(scores[key+'_legacy']))
                missing=int((~scores['pred_present']).sum())
                expected_mean=(valid_sum+missing*scores[key+'_penalty'])/present.size
                np.testing.assert_allclose(aggregate[key+'_full'],expected_mean,rtol=1e-12,atol=1e-12)
        checks.append({'name':name,'official_formula_match':True,'valid_distance_unchanged':True,
                       'finite_and_bounded':True,'deletion_never_improves':True,
                       'missing_frames':int((scores['gt_present'] & ~scores['pred_present']).sum())})
        sources.append({'name':name,**{k:sha256(case[k]) for k in ('pred_masks','gt_masks','extraction_csv','video')}})
        print('PASS',name,flush=True)
    write_csv(out/'summary.csv',rows)
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    audit={'status':'PASS','policy':expected,'case_count':len(checks),'checks':checks,'source_hashes':sources,
           'config_sha256':sha256(config_path),'numpy_version':np.__version__,
           'metric_code_sha256':sha256(Path(__file__).parent/'single_case_rigidbench/mask_distance_full.py'),
           'runner_sha256':sha256(__file__),
           'official_mask_code_sha256':sha256(Path(cfg['official_root'])/'src/rigidbench/eval/score/mask.py')}
    (out/'validation.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    print('COMPLETE',out,flush=True)


def run(config_path):
    cfg=json.loads(config_path.read_text())
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    status=out/'validation.json'
    status.write_text(json.dumps({'status':'RUNNING','config_sha256':sha256(config_path)}))
    try:
        _evaluate(config_path)
    except Exception as exc:
        status.write_text(json.dumps({'status':'FAILED','error_type':type(exc).__name__,
                                     'error':str(exc),'config_sha256':sha256(config_path)},indent=2))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);args=p.parse_args();run(args.config)
