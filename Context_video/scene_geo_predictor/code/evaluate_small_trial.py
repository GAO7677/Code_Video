"""Post-selection validation diagnostics, with clearly labelled event proxies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from future_query_predictor import Predictor
from prepare_small_trial import ROOT, dump, sha
from train_small_trial import Cases, minibatch
from trial_metrics import behaviour


def truth(root, record):
    sample = root/'samples'/record['key']
    actors = json.loads((sample/'metadata.json').read_text())['actors']
    with np.load(sample/'raw/states_xyzw.npz', allow_pickle=False) as archive:
        names = archive['object_names'].tolist()
        dynamic = [i for i,n in enumerate(names) if actors[n]['dynamic']]
        if len(dynamic) != 1:
            raise ValueError('Single-object evaluator only')
        j = dynamic[0]
        history, target = archive['positions'][:8,j], archive['positions'][8:49,j]
        boxes = []
        for i,name in enumerate(names):
            if actors[name]['dynamic']:
                continue
            if actors[name]['shape'] != 'box':
                raise ValueError('Unsupported evaluation geometry')
            size = actors[name]['size_m']
            boxes.append((archive['positions'][0,i], np.array([size[k] for k in ('hx','hy','hz')]), archive['quats'][0,i]))
        actor = actors[names[j]]
        radius = actor['size_m']['radius'] if actor['shape'] == 'sphere' else None
    return history, target, radius, boxes


def summarize(rows):
    result = {}
    for family in ['all']+sorted({r['family'] for r in rows}):
        selected = rows if family == 'all' else [r for r in rows if r['family'] == family]
        metrics = {}
        for name in ('ADE_m', 'FDE_m', 'velocity_MAE_mps', 'pred_penetration_frame_rate',
                     'gt_penetration_frame_rate', 'contact_onset_classification_correct',
                     'contact_onset_time_error_s', 'post_contact_velocity_angle_deg'):
            values = [r[name] for r in selected if r.get(name) is not None]
            metrics[name] = dict(mean=float(np.mean(values)) if values else None, n=len(values))
        for name in ('stop_proxy', 'reversal_proxy', 'descent_15cm_proxy'):
            pairs = [(r['pred_'+name], r['gt_'+name]) for r in selected if r['gt_'+name] is not None]
            metrics[name] = dict(n=len(pairs), TP=sum(p and g for p,g in pairs), FP=sum(p and not g for p,g in pairs),
                                 TN=sum(not p and not g for p,g in pairs), FN=sum(not p and g for p,g in pairs))
        result[family] = metrics
    return result


def main(root):
    torch.set_num_threads(2)
    manifest = json.loads((root/'manifest.json').read_text())
    records = [r for r in manifest['records'] if r['split'] == 'val']
    dataset = Cases(root, records,
                    require_versioned_scene_cache=bool(manifest.get('training', {}).get(
                        'require_versioned_scene_cache', False)))
    truths = [truth(root, r) for r in records]
    results = {}
    for mode in ('constant','visual'):
        run = root/'runs'/mode
        complete = json.loads((run/'complete.json').read_text())
        weights = load_file(str(run/'best.safetensors'))
        model = Predictor(weights['motion_mean'], weights['motion_std'], mode).eval()
        model.load_state_dict(weights)
        rows, swaps = [], []
        for index, record in enumerate(records):
            batch, _ = minibatch(dataset, [index], 'cpu')
            family_ids = [i for i,r in enumerate(records) if r['family'] == record['family']]
            donor = family_ids[(family_ids.index(index)+1)%len(family_ids)]
            other, _ = minibatch(dataset, [donor], 'cpu')
            changed = dict(batch)
            for field in ('scene_xyz','scene_features','scene_mask'):
                changed[field] = other[field]
            with torch.no_grad():
                prediction, swapped = model(batch)[0,0].numpy(), model(changed)[0,0].numpy()
            history, target, radius, boxes = truths[index]
            rows.append(dict(key=record['key'], family=record['family'], **behaviour(prediction,history,target,radius,boxes)))
            swaps.append(dict(key=record['key'], donor=records[donor]['key'],
                prediction_change_m=float(np.linalg.norm(prediction-swapped,axis=-1).mean()),
                original_ADE_m=rows[-1]['ADE_m'], swapped_ADE_m=float(np.linalg.norm(swapped-target,axis=-1).mean())))
        if not np.isclose(np.mean([r['ADE_m'] for r in rows]), complete['best_validation']['ADE_m'], atol=1e-5):
            raise ValueError('Reloaded validation differs from training receipt')
        results[mode] = dict(summary=summarize(rows), rows=rows, same_family_scene_swap=swaps,
                             best_checkpoint_sha256=sha(run/'best.safetensors'))
    cv_rows = []
    for record, (history,target,radius,boxes) in zip(records,truths):
        prediction = history[-1]+(history[-1]-history[-2])*np.arange(1,42)[:,None]
        cv_rows.append(dict(key=record['key'],family=record['family'],**behaviour(prediction,history,target,radius,boxes)))
    results['constant_velocity'] = dict(summary=summarize(cv_rows), rows=cv_rows)
    dump(root/'reports/validation_diagnostics.json', dict(results=results,
        definitions=dict(contact='30Hz signed-distance onset at <=5mm; sustained support is not a new event; includes recontact',
            penetration='sphere-versus-individual-static-OBBs and ground; distance below -5mm; no gap filling',
            stop='speed below 0.1m/s for >=3 intervals; motion-derived proxy',
            reversal='velocity opposite final observed XY motion by >0.1m/s for >=3 intervals; proxy',
            descent='future center lower than observed last center by >0.15m; proxy, not semantic fall classifier',
            post_contact='velocity angle over up to3 intervals following GT 30Hz onset; undefined below0.05m/s'),
        limitations=['Development validation only; single seed.',
            'Sub-frame impacts may be missed: contact metrics are geometry-derived proxies, not 240Hz impulse labels.',
            'Non-sphere contact/penetration is intentionally unsupported without predicted orientation.',
            'Same-family swaps are OOD sensitivity checks, not physically matched counterfactual truth.'],
        static_gt_used_only_in_evaluation=True, paired_training=False))
    print(json.dumps({k:v['summary']['all'] for k,v in results.items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT/'small_trial_120')
    main(p.parse_args().root)
