"""Freeze a balanced subset and replay the original bank before RGB rendering."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
BANK = ROOT.parent/'original_pipeline_experiment_20260917'
SEED = 2026091801


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def actor_metadata(o):
    return dict(object_id=o.family_key, role=o.role, dynamic=o.dynamic,
                shape=o.shape, size_m=o.size, mass_kg=o.mass, friction=o.friction,
                restitution=o.restitution, initial_position_m=o.position,
                initial_linear_velocity_mps=o.linear_velocity,
                initial_angular_velocity_radps=o.angular_velocity)


def select(records):
    selected = []
    for split, count in (('train', 30), ('val', 10)):
        for family in ('barrier', 'door', 'gap'):
            pool = [(i, r) for i, r in enumerate(records)
                    if r['split'] == split and r['family'] == family]
            pool.sort(key=lambda pair: hashlib.sha256(f'{SEED}:{pair[1]["key"]}'.encode()).hexdigest())
            cells = set()
            chosen = []
            for index, row in pool:
                if row['lineage_cell'] in cells:
                    continue
                cells.add(row['lineage_cell'])
                chosen.append(dict(row, bank_index=index))
                if len(chosen) == count:
                    break
            if len(chosen) != count:
                raise ValueError(f'Insufficient distinct cells for {split}/{family}')
            selected.extend(chosen)
    for field in ('key', 'lineage_cell', 'context_hash', 'motion_hash'):
        a = {r[field] for r in selected if r['split'] == 'train'}
        b = {r[field] for r in selected if r['split'] == 'val'}
        if a & b:
            raise ValueError(f'Train/validation overlap: {field}')
    return selected


def plan(output):
    manifest_path = BANK/'data/manifest.json'
    source = json.loads(manifest_path.read_text())
    for path, expected in source['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Original generation code changed: {path}')
    rows = select(source['records'])
    # Report covariate proximity without selecting using future outcomes.
    nearest = {}
    for family in ('barrier', 'door', 'gap'):
        train = np.array([r['u'] for r in rows if r['family'] == family and r['split'] == 'train'])
        val = np.array([r['u'] for r in rows if r['family'] == family and r['split'] == 'val'])
        distance = np.linalg.norm(val[:, None]-train[None], axis=-1).min(1)
        nearest[family] = dict(min=float(distance.min()), median=float(np.median(distance)),
                               definition='Euclidean distance in source unit-cube controls, not motion error')
    value = dict(schema='visual_small_trial_v1', selection_seed=SEED, records=rows,
        counts=dict(Counter(r['split'] for r in rows)), source_manifest_sha256=sha(manifest_path),
        source_bank_sha256=sha(BANK/'data/bank.safetensors'), source_code=source['source_sha256'],
        selection='source train/val only; balanced families; one example per cell; hash order; no outcome selection',
        validation_role='development validation, not a new blind test', nearest_control_distance=nearest,
        training=dict(modes=['constant', 'visual'], seed=42, max_steps=500, effective_batch=30,
                      validate_every=25, optimizer='AdamW', lr=3e-4, weight_decay=.01,
                      loss='position SmoothL1 + 0.2 interval velocity SmoothL1',
                      checkpoint_selection='minimum validation ADE; report final separately',
                      material_input=False, paired_loss=False, dit=False))
    path = output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Refusing to change the frozen manifest')
    else:
        dump(path, value)
    return value


def replay(output, limit=0):
    import torch
    from safetensors import safe_open
    sys.path.insert(0, str(BANK))
    import experiment as e
    torch.set_num_threads(2)
    e.g.SIM_DURATION_S = 49/30
    manifest = plan(output)
    records = manifest['records'][:limit or None]
    with safe_open(str(BANK/'data/bank.safetensors'), framework='pt', device='cpu') as bank:
        for record in records:
            dest = output/'samples'/record['key']
            receipt = dest/'replay.json'
            if receipt.exists():
                old = json.loads(receipt.read_text())
                for name, digest in old['output_sha256'].items():
                    if sha(dest/name) != digest:
                        raise ValueError(f'Replayed artifact changed: {dest/name}')
                continue
            if dest.exists():
                raise FileExistsError(f'Incomplete replay requires inspection: {dest}')
            original = json.loads((BANK/'data/blueprints'/f'{record["key"]}.json').read_text())
            case = e.make_case(record['family'], original['sample_key'], np.asarray(record['u']))
            if json.loads(json.dumps(asdict(case.blueprint))) != original:
                raise ValueError(f'Blueprint mismatch: {record["key"]}')
            objects, pos, vel, quat, corrections = e.audit.replay(case, record['simulation_seed'])
            packed = e.pack(case, objects, pos, vel, quat)
            context_hash = hashlib.sha256(packed['node'].numpy().tobytes()).hexdigest()
            motion_hash = hashlib.sha256(packed['target_position'].numpy().tobytes()+packed['node'].numpy().tobytes()).hexdigest()
            if context_hash != record['context_hash'] or motion_hash != record['motion_hash']:
                raise ValueError(f'Replay hash mismatch: {record["key"]}')
            for name in ('node', 'target_position', 'last_position', 'history_dt', 'future_dt'):
                if not torch.equal(packed[name], bank.get_slice(name)[record['bank_index']]):
                    raise ValueError(f'Bank mismatch: {record["key"]}/{name}')
            if corrections != record['source_rebound_corrections']:
                raise ValueError('Simulation correction count changed')
            if any(o.metadata.get('visual_only') for o in case.blueprint.objects):
                raise ValueError('Visual-only actor export is outside this trial')
            camera = case.blueprint.camera
            metadata = dict(schema_version='visual_small_trial_replay_v1', sample_id=record['key'],
                family_key=case.family_key, seed=record['simulation_seed'], split=record['split'],
                simulation=dict(fps=30, sim_hz=e.g.SIM_HZ, frame_count=49, pre_roll_s=case.blueprint.pre_roll_s),
                camera=dict(intrinsics=dict(yfov_deg=camera.yfov_deg),
                            extrinsics=dict(eye=camera.eye, target=camera.target, up=camera.up)),
                actors={o.name:actor_metadata(o) for o in objects},
                scenario_spec=case.blueprint.metadata, source=str(e.audit.SOURCE),
                rebound_corrections=corrections)
            dest.mkdir(parents=True)
            (dest/'raw').mkdir()
            dump(dest/'metadata.json', metadata)
            # Future truth stays in supervision storage. The renderer sees only RGB0-7 poses.
            np.savez_compressed(dest/'raw/states_xyzw.npz', positions=pos, linear_velocities=vel,
                quats=quat, object_names=np.array([o.name for o in objects]), frame_times=np.arange(49)/30)
            trajectory = dict(object_names=[o.name for o in objects], frame_times_s=(np.arange(8)/30).tolist())
            for j, o in enumerate(objects):
                trajectory[o.name+'_positions'] = pos[:8, j].tolist()
                trajectory[o.name+'_rotations'] = quat[:8, j][:, [3, 0, 1, 2]].tolist()
            dump(dest/'raw/observed_trajectories.json', trajectory)
            dump(receipt, dict(key=record['key'], bank_index=record['bank_index'],
                 context_hash=context_hash, motion_hash=motion_hash, exact_replay=True,
                 output_sha256={name:sha(dest/name) for name in
                    ('metadata.json', 'raw/states_xyzw.npz', 'raw/observed_trajectories.json')}))
            print('REPLAY_OK', record['key'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['plan', 'replay'])
    parser.add_argument('--output', type=Path, default=ROOT/'small_trial_120')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps(plan(args.output)['counts']))
    else:
        replay(args.output, args.limit)
