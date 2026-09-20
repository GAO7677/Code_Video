"""Matched minibatch trials with train-only statistics and resumable validation."""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate
from safetensors.torch import load_file, save_file

from future_query_predictor import Predictor, common, interval_velocity, objective
from predictor_pilot import build_batch
from prepare_small_trial import ROOT, dump, sha
from probe_vggt import validate_gpu


class Cases(Dataset):
    def __init__(self, root, records, require_versioned_scene_cache=False):
        self.root, self.records = root, records
        self.require_versioned_scene_cache = bool(require_versioned_scene_cache)

    def __len__(self):
        return len(self.records)

    @lru_cache(maxsize=128)
    def __getitem__(self, index):
        row = self.records[index]
        batch, target, _ = build_batch(
            self.root, self.root/'scene_features', [row['key']],
            require_versioned_scene_cache=self.require_versioned_scene_cache)
        item = {k:v[0] for k,v in batch.items()}
        item['scene_features'] = item['scene_features'].half()
        return item, target[0]


def sample_indices(size, step, batch_size, seed):
    if size % batch_size:
        raise ValueError('This balanced pilot requires whole batches per epoch')
    batches = size//batch_size
    generator = torch.Generator().manual_seed(seed+step//batches)
    order = torch.randperm(size, generator=generator)
    offset = step % batches*batch_size
    return order[offset:offset+batch_size].tolist()


def minibatch(dataset, indices, device):
    batch, target = default_collate([dataset[i] for i in indices])
    return {k:v.to(device) for k,v in batch.items()}, target.to(device)


@torch.no_grad()
def evaluate(model, dataset, device, chunk=10):
    model.eval()
    rows, positions = [], []
    for begin in range(0, len(dataset), chunk):
        indices = list(range(begin, min(len(dataset), begin+chunk)))
        batch, target = minibatch(dataset, indices, device)
        pred = model(batch)
        error = (pred-target).norm(dim=-1)
        pv = interval_velocity(pred, batch['last_position'], batch['future_dt'])
        tv = interval_velocity(target, batch['last_position'], batch['future_dt'])
        velocity_error = (pv-tv).abs().mean((1,2,3))
        position_loss = torch.nn.functional.smooth_l1_loss(pred,target,reduction='none').mean((1,2,3))
        velocity_loss = torch.nn.functional.smooth_l1_loss(pv,tv,reduction='none').mean((1,2,3))
        for j, i in enumerate(indices):
            rows.append(dict(key=dataset.records[i]['key'], family=dataset.records[i]['family'],
                ADE_m=float(error[j].mean()), FDE_m=float(error[j, :, -1].mean()),
                velocity_MAE_mps=float(velocity_error[j]), position_loss=float(position_loss[j]),
                velocity_loss=float(velocity_loss[j]), physical_loss=float(position_loss[j]+.2*velocity_loss[j])))
        positions.append(pred.cpu())
    metrics = ('ADE_m', 'FDE_m', 'velocity_MAE_mps', 'position_loss', 'velocity_loss', 'physical_loss')
    summary = {name:float(np.mean([row[name] for row in rows])) for name in metrics}
    summary['families'] = {f:{name:float(np.mean([r[name] for r in rows if r['family'] == f]))
                              for name in metrics} for f in sorted({r['family'] for r in rows})}
    return summary, rows, torch.cat(positions)


def atomic_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    temp.replace(path)


def train(root, mode, device, resume=False):
    manifest = json.loads((root/'manifest.json').read_text())
    config = manifest['training']
    require_versioned = bool(config.get('require_versioned_scene_cache', False))
    trainset = Cases(root, [r for r in manifest['records'] if r['split'] == 'train'], require_versioned)
    valset = Cases(root, [r for r in manifest['records'] if r['split'] == 'val'], require_versioned)
    sources = {name:sha(ROOT/name) for name in ('future_query_predictor.py', 'train_small_trial.py', 'predictor_pilot.py')}
    sources[str(common.__file__)] = sha(common.__file__)
    feature_hashes = {r['key']:sha(root/'scene_features'/r['key']/'scene_tokens.npz') for r in manifest['records']}
    feature_report_hashes = {r['key']:sha(root/'scene_features'/r['key']/'report.json') for r in manifest['records']}
    supervision_hashes = {}
    for row in manifest['records']:
        sample = root/'samples'/row['key']
        replay = json.loads((sample/'replay.json').read_text())
        actual = sha(sample/'raw/states_xyzw.npz')
        if replay['motion_hash'] != row['motion_hash'] or actual != replay['output_sha256']['raw/states_xyzw.npz']:
            raise ValueError('Supervision no longer matches verified source replay')
        supervision_hashes[row['key']] = actual
    strict_future_grid = bool(config.get('strict_future_grid', True))
    contract = dict(scene_mode=mode, strict_future_grid=strict_future_grid,
                    manifest_sha256=sha(root/'manifest.json'), source_sha256=sources,
                    feature_sha256=feature_hashes, feature_report_sha256=feature_report_hashes,
                    supervision_sha256=supervision_hashes)
    output = root/'runs'/mode
    if output.exists() and not resume:
        raise FileExistsError('Existing training output requires explicit --resume')
    state = None
    if resume:
        state = torch.load(output/'last.pt', map_location=device, weights_only=False)
        saved_contract = dict(state['contract'])
        # Checkpoints written before the strict 30-FPS contract did not carry
        # this metadata. Their training stream already used the fixed grid, so
        # treat a missing field as the protocol default while preserving all
        # other provenance checks.
        if 'strict_future_grid' not in saved_contract or 'feature_report_sha256' not in saved_contract:
            legacy_contract = dict(contract)
            if 'strict_future_grid' not in saved_contract:
                legacy_contract.pop('strict_future_grid')
            if 'feature_report_sha256' not in saved_contract:
                legacy_contract.pop('feature_report_sha256')
            compatible = saved_contract == legacy_contract
        else:
            compatible = saved_contract == contract
        if not compatible:
            raise ValueError('Resume data/source/protocol changed')
    output.mkdir(parents=True, exist_ok=True)
    values = torch.stack([trainset[i][0]['motion'][0] for i in range(len(trainset))])
    mean, std = values.mean(0), values.std(0, unbiased=False).clamp_min(1e-3)
    save_file(dict(motion_mean=mean, motion_std=std), str(output/'train_only_stats.safetensors'))
    model = Predictor(mean, std, mode, seed=config['seed'],
                      strict_future_grid=strict_future_grid).to(device)
    optimizer = torch.optim.AdamW(model.active_parameters(), lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    start, best = 0, float('inf')
    best_weights, best_record = None, None
    if resume:
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start, best = state['step'], state['best_ADE_m']
        best_weights = {k:v.cpu() for k,v in state['best_weights'].items()}
        best_record = state['best_record']
        save_file(best_weights, str(output/'best.safetensors'))
        dump(output/'best.json', best_record)
        # Current network is deterministic, but preserve RNG state for exact continuation.
        torch.set_rng_state(state['cpu_rng'].cpu())
        if device != 'cpu':
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
        # Discard only uncheckpointed log entries from an interrupted attempt.
        log = output/'metrics.jsonl'
        if log.exists():
            entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            kept = [e for e in entries if e['step'] <= start]
            temp = log.with_suffix('.tmp')
            temp.write_text(''.join(json.dumps(e, allow_nan=False)+'\n' for e in kept))
            temp.replace(log)
    elif (output/'metrics.jsonl').exists():
        raise FileExistsError('Refusing to overwrite metrics')
    dump(output/'contract.json', contract)
    if start == 0:
        initial, _, _ = evaluate(model, valset, device)
        dump(output/'initial_validation.json', initial)
    for step in range(start, config['max_steps']):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = sample_indices(len(trainset), step, config['effective_batch'], config['seed'])
        total_loss = 0.
        for begin in range(0, len(indices), 10):
            part = indices[begin:begin+10]
            batch, target = minibatch(trainset, part, device)
            loss = objective(model(batch), target, batch)*len(part)/len(indices)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {mode} step {step+1}')
            loss.backward()
            total_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(list(model.active_parameters()), 1., error_if_nonfinite=True)
        optimizer.step()
        entry = dict(step=step+1, training_batch_loss=total_loss)
        checkpoint_step = (step+1) % config['validate_every'] == 0 or step+1 == config['max_steps']
        if checkpoint_step:
            validation, _, _ = evaluate(model, valset, device)
            entry['validation'] = validation
            if validation['ADE_m'] < best:
                best = validation['ADE_m']
                best_weights = {k:v.detach().cpu().contiguous().clone() for k,v in model.state_dict().items()}
                best_record = dict(step=step+1, validation=validation)
                save_file(best_weights, str(output/'best.safetensors'))
                dump(output/'best.json', best_record)
            print(mode, step+1, 'VAL_ADE', validation['ADE_m'], flush=True)
        with (output/'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(entry, allow_nan=False)+'\n')
        if checkpoint_step:
            atomic_checkpoint(output/'last.pt', dict(step=step+1, best_ADE_m=best, contract=contract,
                best_weights=best_weights, best_record=best_record, model=model.state_dict(),
                optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device) if device != 'cpu' else None))
    final_val, _, _ = evaluate(model, valset, device)
    model.load_state_dict(load_file(str(output/'best.safetensors'), device=device))
    best_val, rows, prediction = evaluate(model, valset, device)
    train_metrics, _, _ = evaluate(model, trainset, device)
    np.savez_compressed(output/'validation_predictions.npz', prediction=prediction.numpy(),
                        keys=np.array([r['key'] for r in rows]))
    dump(output/'complete.json', dict(mode=mode, steps=config['max_steps'], effective_batch=config['effective_batch'],
        best_validation=best_val, final_validation=final_val, training_at_best=train_metrics,
        validation_rows=rows, test_evaluated=False, validation_role='development, not blind test',
        paired_training=False, dit_loaded=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'small_trial_120')
    parser.add_argument('--mode', choices=['constant', 'visual', 'motion_only', 'geometry_only'], required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True, timeout=30).stdout
    validate_gpu(args.gpu_uuid, os.environ.get('CUDA_VISIBLE_DEVICES'), inventory)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one allowed GPU')
    torch.set_num_threads(2)
    train(args.root, args.mode, 'cuda:0', args.resume)
