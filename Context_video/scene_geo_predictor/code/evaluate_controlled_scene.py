"""Evaluate frozen predictors against actual single-variable simulation pairs."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from controlled_scene_eval import OUT
from evaluate_small_trial import truth
from future_query_predictor import Predictor
from predictor_pilot import build_batch
from prepare_small_trial import dump, sha
from trial_metrics import behaviour


def response_metrics(pred_a, pred_b, gt_a, gt_b):
    predicted, target = pred_b-pred_a, gt_b-gt_a
    pn = np.linalg.norm(predicted,axis=-1)
    gn = np.linalg.norm(target,axis=-1)
    denominator = np.linalg.norm(predicted)*np.linalg.norm(target)
    return dict(predicted_change_m=float(pn.mean()),gt_change_m=float(gn.mean()),
        response_error_m=float(np.linalg.norm(predicted-target,axis=-1).mean()),
        response_gain=float(pn.mean()/gn.mean()) if gn.mean()>1e-5 else None,
        response_cosine=float(np.clip(np.sum(predicted*target)/denominator,-1,1)) if denominator>1e-10 else None,
        gt_first_difference_frame=int(np.flatnonzero(gn>.001)[0]+8) if (gn>.001).any() else None,
        pred_first_difference_frame=int(np.flatnonzero(pn>.001)[0]+8) if (pn>.001).any() else None)


def evaluate(root):
    torch.set_num_threads(2)
    manifest = json.loads((root/'manifest.json').read_text())
    records = manifest['records']
    audit = json.loads((root/'simulation_audit.json').read_text())
    if not audit['all_contexts_match'] or audit['overlap_original_bank']:
        raise ValueError('Control simulation audit failed')
    batches,targets,truths,feature_hashes = [],[],[],{}
    for record in records:
        key = record['key']
        sample = root/'samples'/key
        proof = json.loads((sample/'replay.json').read_text())
        for name,digest in proof['output_sha256'].items():
            if sha(sample/name)!=digest:
                raise ValueError('Simulation/metadata changed')
        path = root/'scene_features'/key/'scene_tokens.npz'
        feature_hashes[key] = sha(path)
        fr = json.loads((path.parent/'report.json').read_text())
        ar = root/'aligned_scene'/key
        if not json.loads((ar/'report.json').read_text())['alignment']['accepted']:
            raise ValueError('Rejected alignment')
        if (sha(ar/'report.json')!=fr['alignment_report_sha256'] or
            sha(ar/'coarse_static_points.npz')!=fr['aligned_points_sha256'] or
            sha(root/'raw_probe'/key/'raw_vggt.npz')!=fr['raw_sha256']):
            raise ValueError('Feature provenance changed')
        batch,target,_ = build_batch(root,root/'scene_features',[key])
        # Match the fp16 cache representation used during training and validation.
        batch['scene_features'] = batch['scene_features'].half()
        batches.append(batch); targets.append(target[0,0].numpy()); truths.append(truth(root,record))
    groups = {}
    for family in ('barrier','door','gap'):
        ids = [i for i,r in enumerate(records) if r['family']==family]
        groups[family] = ids
        renders = [json.loads((root/'render_context8'/records[i]['key']/'cycles_frames/render_metadata.json').read_text()) for i in ids]
        for field in ('resolution','samples','engine','exposure','hdri','lighting_preset','material_assignments'):
            if not all(r[field]==renders[0][field] for r in renders):
                raise ValueError(f'Render settings differ within {family}: {field}')
        for i in ids[1:]:
            for field in ('motion','last_position','last_velocity','future_dt','object_mask'):
                if not torch.equal(batches[ids[0]][field],batches[i][field]):
                    raise ValueError(f'Predictor motion differs within {family}')
            a=root/'observed_context'/records[ids[0]]['key']/'context_geometry.npz'
            b=root/'observed_context'/records[i]['key']/'context_geometry.npz'
            with np.load(a) as x,np.load(b) as y:
                for field in ('camera_K','camera_world_to_view'):
                    if not np.array_equal(x[field],y[field]):
                        raise ValueError('Rendered camera differs within group')
    results,arrays = {},{}
    arrays['GT'] = np.stack(targets)
    for mode, checkpoint in manifest['checkpoints'].items():
        if sha(checkpoint['path'])!=checkpoint['sha256']:
            raise ValueError('Frozen checkpoint changed')
        weights = load_file(checkpoint['path'])
        model = Predictor(weights['motion_mean'],weights['motion_std'],mode).eval()
        model.load_state_dict(weights)
        with torch.inference_mode():
            predictions = np.stack([model(b)[0,0].numpy() for b in batches])
        arrays[mode] = predictions
        rows = []
        for record,pred,t in zip(records,predictions,truths):
            history,target,radius,boxes = t
            rows.append({**record,**behaviour(pred,history,target,radius,boxes)})
        comparisons = {}
        for family,ids in groups.items():
            pairs=[]
            for a,b in combinations(ids,2):
                metrics=response_metrics(predictions[a],predictions[b],targets[a],targets[b])
                if mode=='constant' and metrics['predicted_change_m']>1e-7:
                    raise ValueError('Constant control must be scene invariant')
                pairs.append(dict(a=records[a]['key'],b=records[b]['key'],**metrics))
            comparisons[family]=dict(pairs=pairs,
                mean_ADE_m=float(np.mean([rows[i]['ADE_m'] for i in ids])),
                mean_response_error_m=float(np.mean([r['response_error_m'] for r in pairs])),
                mean_predicted_change_m=float(np.mean([r['predicted_change_m'] for r in pairs])),
                mean_gt_change_m=float(np.mean([r['gt_change_m'] for r in pairs])))
        results[mode]=dict(rows=rows,groups=comparisons)
    output=root/'reports'
    output.mkdir(exist_ok=True)
    np.savez_compressed(output/'predictions.npz',keys=np.array([r['key'] for r in records]),**arrays)
    report=dict(results=results,manifest_sha256=sha(root/'manifest.json'),feature_sha256=feature_hashes,
        render_and_motion_invariants_verified=True,
        inference_source_sha256={str(p):sha(p) for p in
            (Path(__file__),Path(__file__).with_name('future_query_predictor.py'),Path(__file__).with_name('predictor_pilot.py'))},
        weights_updated=False,scene_inputs='Observed VGGT/SAM2/Utonia only; static GT used in metrics, never forward',
        response_definition='All 6 unordered pairs per family; mean norm of predicted trajectory difference minus GT difference',
        limitations=manifest['limitations']+['Pairs share cases and are not independent statistical replicates.',
            'Event measures are 30Hz geometry/motion proxies, not simulator impulse labels.'])
    dump(output/'diagnostics.json',report)
    print(json.dumps({m:{f:{k:v for k,v in r.items() if k!='pairs'} for f,r in d['groups'].items()}
                      for m,d in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=OUT)
    evaluate(p.parse_args().root)
