"""Frozen single-variable evaluation generation; never used for training."""
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from prepare_small_trial import ROOT, BANK, actor_metadata, dump, sha

VALUES = {'barrier': (.40, .65, .90, 1.15), 'door': (.38, .50, .62, .74),
          'gap': (.06, .20, .40, .70)}
VARIABLE = {'barrier': 'barrier_center_x_m', 'door': 'opening_width_m', 'gap': 'gap_width_m'}
SEED = 2026091803
OUT = ROOT/'controlled_scene_eval_12'


def generator():
    sys.path.insert(0, str(BANK))
    import experiment as e
    e.g.SIM_DURATION_S = 49/30
    return e


def make_case(e, family, key, value):
    u = np.full(5, .5)
    if family == 'door':
        u[0] = (value-.38)/.36
    elif family == 'gap':
        u[0] = (value-.06)/.64
    case = e.make_case(family, key, u)
    if family == 'barrier':
        objects = tuple(replace(o, position=(value, *o.position[1:]))
                        if o.name == 'puck_barrier' else o for o in case.blueprint.objects)
        metadata = {**case.blueprint.metadata, 'controlled_variable': VARIABLE[family],
                    VARIABLE[family]: value}
        case = replace(case, controlled_variable=VARIABLE[family], controlled_value=value,
                       controlled_value_label=f'x={value:.2f} m', units='m',
                       blueprint=replace(case.blueprint, objects=objects, metadata=metadata))
    return case


def records():
    return [dict(key=f'control_{family}_{i:02d}', family=family, split='control_eval',
                 variable=VARIABLE[family], value=value, units='m', simulation_seed=SEED,
                 generalization='OOD barrier translation' if family == 'barrier' else 'in-range geometry')
            for family, values in VALUES.items() for i, value in enumerate(values)]


def plan(root):
    e = generator()
    sources = [Path(__file__), BANK/'experiment.py', Path(e.audit.__file__),
               e.audit.SOURCE/'scripts/generate_v2v_context_demos.py']
    run = ROOT/'small_trial_120/runs'
    checkpoint = {mode:dict(path=str(run/mode/'best.safetensors'),
                           sha256=sha(run/mode/'best.safetensors'),
                           step=json.loads((run/mode/'best.json').read_text())['step'])
                  for mode in ('constant', 'visual')}
    manifest = dict(schema='single_variable_scene_eval_v1', records=records(),
        source_sha256={str(p):sha(p) for p in sources}, checkpoints=checkpoint,
        selection='Four predeclared values per family; midpoint motion controls; no outcome filtering',
        role='Post-selection controlled diagnostic, not training, checkpoint selection or a blind benchmark',
        fixed_motion_controls=dict(speed_u=.5, dx_u=.5, dy_u=.5, heading_u=.5),
        training=False, paired_loss=False, dit=False,
        limitations=['One motion context per family; not a population-level generalization estimate.',
                     'Barrier center translation was absent from the training sampling scheme.',
                     'Original puck rebound correction retained, not claimed to be raw Bullet dynamics.'])
    path = root/'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Frozen protocol/source/checkpoint changed')
    else:
        dump(path, manifest)
    return manifest


def generate(root):
    manifest = plan(root)
    e = generator()
    old = json.loads((BANK/'data/manifest.json').read_text())['records']
    old_contexts = {r['context_hash'] for r in old}
    group_context, audits = {}, []
    for row in manifest['records']:
        case = make_case(e, row['family'], row['key'], row['value'])
        objects, pos, vel, quat, corrections = e.audit.replay(case, SEED)
        j, = [i for i,o in enumerate(objects) if o.dynamic]
        assert pos.shape[0] == 49 and all(np.isfinite(x).all() for x in (pos,vel,quat))
        packed = e.pack(case, objects, pos, vel, quat)
        ch = hashlib.sha256(packed['node'].numpy().tobytes()).hexdigest()
        if ch in old_contexts:
            raise ValueError('Control context overlaps original training/validation/test bank')
        context = (pos[:8,j], vel[:8,j], quat[:8,j], asdict(objects[j]), asdict(case.blueprint.camera))
        if row['family'] in group_context:
            ref = group_context[row['family']]
            if not all(np.array_equal(a,b) for a,b in zip(context[:3], ref[:3])) or context[3:] != ref[3:]:
                raise ValueError('Controlled group has different observation motion/object/camera')
        else:
            group_context[row['family']] = context
        dest = root/'samples'/row['key']
        receipt = dest/'replay.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            for name, digest in saved['output_sha256'].items():
                if sha(dest/name) != digest:
                    raise ValueError('Existing control artifact changed')
            with np.load(dest/'raw/states_xyzw.npz') as archive:
                assert np.array_equal(pos, archive['positions'])
        else:
            if dest.exists():
                raise FileExistsError(dest)
            camera = case.blueprint.camera
            metadata = dict(schema_version='controlled_scene_eval_v1', sample_id=row['key'],
                family_key=case.family_key, seed=SEED, split=row['split'],
                simulation=dict(fps=30, sim_hz=e.g.SIM_HZ, frame_count=49, pre_roll_s=case.blueprint.pre_roll_s),
                camera=dict(intrinsics=dict(yfov_deg=camera.yfov_deg),
                            extrinsics=dict(eye=camera.eye,target=camera.target,up=camera.up)),
                actors={o.name:actor_metadata(o) for o in objects}, scenario_spec=case.blueprint.metadata,
                source=str(e.audit.SOURCE), rebound_corrections=corrections)
            dump(dest/'metadata.json', metadata)
            dump(dest/'blueprint.json', asdict(case.blueprint))
            (dest/'raw').mkdir()
            np.savez_compressed(dest/'raw/states_xyzw.npz', positions=pos,linear_velocities=vel,
                quats=quat, object_names=np.array([o.name for o in objects]),frame_times=np.arange(49)/30)
            trajectory = dict(object_names=[o.name for o in objects], frame_times_s=(np.arange(8)/30).tolist())
            for i,o in enumerate(objects):
                trajectory[o.name+'_positions'] = pos[:8,i].tolist()
                trajectory[o.name+'_rotations'] = quat[:8,i][:,[3,0,1,2]].tolist()
            dump(dest/'raw/observed_trajectories.json', trajectory)
            dump(receipt, dict(key=row['key'],context_hash=ch,original_simulator_used=True,
                original_bank_replay=False,rebound_corrections=corrections,
                output_sha256={name:sha(dest/name) for name in ('metadata.json','blueprint.json',
                    'raw/states_xyzw.npz','raw/observed_trajectories.json')}))
        audits.append(dict(key=row['key'],context_hash=ch,corrections=corrections,
                           group_context_bitwise_identical=True))
        print('CONTROL_READY',row['key'],row['value'],flush=True)
    dump(root/'simulation_audit.json',dict(records=audits,all_contexts_match=True,
        overlap_original_bank=False,outcome_selection=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['plan','generate'])
    parser.add_argument('--root',type=Path,default=OUT)
    args = parser.parse_args()
    (plan if args.stage == 'plan' else generate)(args.root)
