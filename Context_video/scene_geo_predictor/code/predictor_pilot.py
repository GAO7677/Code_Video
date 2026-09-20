"""Three-case integration/overfit diagnostic, explicitly not an independent test."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from probe_vggt import validate_gpu
from scene_token_cache import sha256


def build_batch(root, features, cases, require_versioned_scene_cache=False):
    import torch
    from future_query_predictor import motion_features

    rows, records = [], []
    for case in cases:
        context_path = root/'observed_context'/case/'context_geometry.npz'
        report = json.loads((context_path.parent/'report.json').read_text())
        scene_path = features/case/'scene_tokens.npz'
        scene_report = json.loads((scene_path.parent/'report.json').read_text())
        if require_versioned_scene_cache and scene_report.get('cache_schema') != 'observed_utonia_tokens_v2':
            raise ValueError('Scene cache is legacy/unverified; regenerate with cache_schema v2')
        if sha256(scene_path) != scene_report['output_sha256']:
            raise ValueError('Visual feature cache hash changed')
        if scene_report['static_scene_gt_used'] or scene_report['future_rgb_used']:
            raise ValueError('Scene cache is not an observation-only visual input')
        with np.load(context_path, allow_pickle=False) as data:
            geometry = {k: data[k] for k in data.files}
        if len(report['dynamic_names']) != 1:
            raise ValueError('This integration pilot is single-object only')
        aligned_report = json.loads((root/'aligned_scene'/case/'report.json').read_text())
        if sha256(context_path) != aligned_report['provenance']['geometry']['sha256']:
            raise ValueError('Motion context and visual scale anchor do not match')
        # Future states are loaded only here as targets; never by the scene extractor.
        label_path = Path(report['sample'])/'raw/states_xyzw.npz'
        with np.load(label_path, allow_pickle=False) as labels:
            names = labels['object_names'].astype(str).tolist()
            ids = [names.index(n) for n in report['dynamic_names']]
            if not np.allclose(labels['positions'][:8, ids], geometry['positions_world'], atol=1e-6, rtol=0):
                raise ValueError('Supervision and observed motion are from different samples')
            times = labels['frame_times'][:49]
            if times.shape != (49,) or not np.allclose(times, np.arange(49)/30, atol=1e-6, rtol=0):
                raise ValueError('RGB0-48 supervision must use the observed 30 FPS timeline')
            target = labels['positions'][8:49, ids].transpose(1, 0, 2).astype(np.float32)
        with np.load(scene_path, allow_pickle=False) as data:
            scene = {k: data[k] for k in ('scene_xyz', 'scene_features', 'scene_mask')}
        rows.append(dict(positions=geometry['positions_world'].astype(np.float32),
                         size=geometry['size_m'].astype(np.float32),
                         times=geometry['frame_times'].astype(np.float32), target=target, **scene))
        records.append(dict(case=case, role='development_fit_only', context_sha256=sha256(context_path),
                            scene_sha256=sha256(scene_path), supervision_sha256=sha256(label_path),
                            input_pixels_sha256=scene_report['input_pixels_sha256']))
    motion, last, velocity = motion_features(
        torch.from_numpy(np.stack([r['positions'] for r in rows])),
        torch.from_numpy(np.stack([r['size'] for r in rows])),
        torch.from_numpy(np.stack([r['times'] for r in rows])))
    n = max(len(r['scene_xyz']) for r in rows)
    batch = dict(motion=motion, last_position=last, last_velocity=velocity,
        object_mask=torch.ones(len(rows), 1, dtype=torch.bool),
        future_dt=torch.arange(1, 42).float()[None].expand(len(rows), -1)/30,
        scene_xyz=torch.zeros(len(rows), n, 3), scene_features=torch.zeros(len(rows), n, 1386),
        scene_mask=torch.zeros(len(rows), n, dtype=torch.bool))
    for i, row in enumerate(rows):
        count = len(row['scene_xyz'])
        for name in ('scene_xyz', 'scene_features', 'scene_mask'):
            batch[name][i, :count] = torch.from_numpy(row[name])
    target = torch.from_numpy(np.stack([r['target'] for r in rows]))
    if not torch.isfinite(target).all() or not all(torch.isfinite(v).all() for v in batch.values()):
        raise ValueError('Nonfinite training inputs or targets')
    return batch, target, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True, capture_output=True, check=True, timeout=30).stdout
    index = validate_gpu(args.gpu_uuid, os.environ.get('CUDA_VISIBLE_DEVICES'), inventory)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1 <= args.steps <= 2000 or len(args.cases) != 3:
        raise ValueError('This executable is a bounded three-case integration pilot')
    import torch
    from safetensors.torch import save_file
    from future_query_predictor import Predictor, objective

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one permitted GPU')
    torch.set_num_threads(2)
    batch, target, records = build_batch(args.root, args.features, args.cases)
    mean = batch['motion'].mean((0, 1))
    std = batch['motion'].std((0, 1), unbiased=False).clamp_min(1e-3)
    args.output.mkdir(parents=True)
    save_file(dict(motion_mean=mean, motion_std=std), str(args.output/'fit_only_stats.safetensors'))
    batch = {k: v.to('cuda:0') for k, v in batch.items()}
    target = target.to('cuda:0')
    summary, curves, predictions = {}, {}, {}
    cv = batch['last_position'][:, :, None]+batch['last_velocity'][:, :, None]*batch['future_dt'][:, None, :, None]
    for mode in ('constant', 'visual'):
        model = Predictor(mean, std, scene_mode=mode).to('cuda:0')
        optimizer = torch.optim.AdamW(model.active_parameters(), lr=3e-4, weight_decay=.01)
        losses, gradients = [], []
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            initial = float(objective(model(batch), target, batch))
        losses.append(initial)
        for step in range(1, args.steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = objective(prediction, target, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{mode} step{step}: nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(list(model.active_parameters()), 1., error_if_nonfinite=True)
            if step in (1, 2, 5, args.steps):
                gradient = model.scene_projection[-1].weight.grad
                gradients.append(dict(step=step, total_unclipped_norm=float(norm),
                                      scene_projection_gradient_norm=float(gradient.norm()) if gradient is not None else 0.))
            optimizer.step()
            losses.append(float(loss.detach()))
            if step % 25 == 0:
                print(mode, step, losses[-1], flush=True)
        model.eval()
        with torch.no_grad():
            prediction = model(batch)
            final = float(objective(prediction, target, batch))
            error = (prediction-target).norm(dim=-1)
            changed = {**batch, 'scene_xyz': batch['scene_xyz'].roll(1, 0),
                       'scene_features': batch['scene_features'].roll(1, 0),
                       'scene_mask': batch['scene_mask'].roll(1, 0)}
            perturbation = float((model(changed)-prediction).norm(dim=-1).mean())
        summary[mode] = dict(initial_loss=initial, final_loss=final,
            loss_reduction_fraction=1-final/initial, training_ADE_m=float(error.mean()),
            training_FDE_m=float(error[:, :, -1].mean()), per_case_ADE_m=error.mean((1, 2)).cpu().tolist(),
            scene_swap_prediction_change_m=perturbation, gradient_probes=gradients,
            steps=args.steps, effective_batch=3, parameters=sum(p.numel() for p in model.parameters()),
            elapsed_seconds=time.monotonic()-start, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        curves[mode] = losses+[final]
        predictions[mode] = prediction.cpu().numpy()
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                  str(args.output/f'{mode}_step{args.steps:04d}.safetensors'))
        del optimizer, model
    passed = all(s['loss_reduction_fraction'] > .5 for s in summary.values())
    passed &= summary['constant']['scene_swap_prediction_change_m'] < 1e-7
    passed &= any(g['scene_projection_gradient_norm'] > 0 for g in summary['visual']['gradient_probes'])
    result = dict(status='integration_fit_complete', smoke_gate_passed=bool(passed),
        interpretation='3 training examples only; no held-out validation/test and no generalization claim',
        training_modes='same architecture/init; constant scene content control versus observed visual scene',
        input_contract='RGB8-derived scene, observed dynamic position/size, actual camera assistance; no static GT/material/pose inputs',
        missing_state_scope='observed angular motion and inter-object dynamics are not modelled by this single-object pilot',
        future_targets_only_in_loss=True, events_evaluated=False, paired_loss=False,
        dit_loaded=False, pretrained_encoders_loaded_during_training=False,
        source_sha256={name: sha256(args.root/name) for name in ('predictor_pilot.py', 'future_query_predictor.py')},
        records=records, physical_gpu_index=index, gpu_uuid=args.gpu_uuid, summary=summary)
    (args.output/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (args.output/'loss_curves.json').write_text(json.dumps(curves, indent=2, allow_nan=False)+'\n')
    np.savez_compressed(args.output/'fit_predictions.npz', target=target.cpu().numpy(),
                        cv=cv.cpu().numpy(), **predictions)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for mode, values in curves.items():
        axes[0].plot(values, label=mode)
    axes[0].set(xlabel='Optimizer step', ylabel='Position + 0.2 interval-velocity loss', title='Training fit only: 3 cases')
    axes[0].legend()
    for i, case in enumerate(args.cases):
        for label, value in [('GT', target.cpu().numpy()), ('CV', cv.cpu().numpy()), *predictions.items()]:
            axes[i+1].plot(value[i, 0, :, 0], value[i, 0, :, 2], label=label)
        axes[i+1].set(xlabel='World X (m)', ylabel='World Z (m)', title=case)
        axes[i+1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output/'training_fit.png', dpi=140)
    plt.close(fig)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
