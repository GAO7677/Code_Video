"""Publish observed RGB and simulated GT, never predictor outputs."""
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from prepare_small_trial import ROOT, dump, sha

OUT = ROOT/'data_samples_view'
HUB = Path('/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/generic-full-2175-lora1500-lineage/p4-v2-data-samples')


def case_payload(root, row, asset_name):
    key = row['key']
    sample = root/'samples'/key
    proof = json.loads((sample/'replay.json').read_text())
    for name, digest in proof['output_sha256'].items():
        if sha(sample/name) != digest:
            raise ValueError(f'Changed simulation: {key}/{name}')
    context = root/'observed_context'/key
    report = json.loads((context/'report.json').read_text())
    with np.load(sample/'raw/states_xyzw.npz', allow_pickle=False) as data:
        names = data['object_names'].tolist()
        index = names.index(report['dynamic_names'][0])
        position, velocity = data['positions'][:, index], data['linear_velocities'][:, index]
        quats = data['quats'][:, index]
        times = data['frame_times']
    if position.shape != (49, 3) or not np.allclose(times, np.arange(49)/30, atol=1e-7):
        raise ValueError('Expected RGB0-48 at 30 FPS')
    with np.load(context/'context_geometry.npz', allow_pickle=False) as data:
        if not np.array_equal(data['positions_world'][:, 0], position[:8]):
            raise ValueError('Observed context disagrees with simulated GT')
        k, rt, size = data['camera_K'], data['camera_world_to_view'], data['size_m'][0]
    camera = position@rt[:, :3].T+rt[:, 3]
    if (camera[:, 2] <= 0).any():
        raise ValueError('Trajectory lies behind camera')
    pixel = camera@k.T
    pixel = pixel[:, :2]/pixel[:, 2:]
    assets = OUT/'assets'/asset_name/key
    assets.parent.mkdir(parents=True, exist_ok=True)
    source = root/'context_inputs'/key
    if assets.is_symlink():
        if assets.resolve() != source.resolve():
            raise ValueError('Unexpected existing asset link')
    elif assets.exists():
        raise FileExistsError(assets)
    else:
        assets.symlink_to(source, target_is_directory=True)
    for t in range(8):
        if not (assets/f'rgb_{t:02d}.png').is_file():
            raise FileNotFoundError(key)
    value = row.get('value')
    if value is None:
        u = row['u'][0]
        value = {'barrier':30+60*u, 'door':.38+.36*u, 'gap':.06+.64*u}[row['family']]
    payload = dict(key=key, family=row['family'], value=value, split=row['split'],
        position=position.tolist(), velocity=velocity.tolist(), pixels=pixel.tolist(), size=size.tolist(),
        frames=[f'assets/{asset_name}/{key}/rgb_{t:02d}.png' for t in range(8)],
        speed=float(np.linalg.norm(velocity[0])), heading_deg=float(np.degrees(np.arctan2(velocity[0, 1], velocity[0, 0]))),
        initial_position=position[0].tolist(), state_sha256=sha(sample/'raw/states_xyzw.npz'),
        source_root=str(root), corrections=proof.get('source_rebound_corrections', None))
    return payload, dict(position=position[:8], velocity=velocity[:8], quats=quats[:8], K=k, RT=rt)


def main():
    OUT.mkdir(exist_ok=True)
    groups = []
    for kind, root in [('control', ROOT/'controlled_scene_eval_12'), ('train', ROOT/'small_trial_120')]:
        manifest = json.loads((root/'manifest.json').read_text())
        for family in ('barrier', 'door', 'gap'):
            rows = [r for r in manifest['records'] if r['family'] == family and
                    (kind == 'control' or r['split'] == 'train')]
            if kind == 'train':
                rows.sort(key=lambda row: row['u'][0])
                rows = [rows[i] for i in np.linspace(0, len(rows)-1, 4, dtype=int)]
            cases, contexts = zip(*(case_payload(root, r, kind) for r in rows))
            if kind == 'control':
                for context in contexts[1:]:
                    if not all(np.array_equal(context[k], contexts[0][k]) for k in context):
                        raise ValueError('Control observations or cameras differ')
            same = []
            for a, b in combinations(range(4), 2):
                if np.array_equal(np.array(cases[a]['position'])[8:], np.array(cases[b]['position'])[8:]):
                    same.append([a, b])
            groups.append(dict(kind=kind, family=family, cases=cases, identical_future_pairs=same,
                context_equal_verified=kind == 'control'))
    dump(OUT/'data.json', dict(groups=groups, model_predictions=False,
        rgb_frames=list(range(8)), gt_frames=list(range(49)), fps=30,
        control_in_training=False, train_selection='Four evenly spaced geometry ranks from the existing 30-per-family training subset; no outcome selection',
        limitation='Only three single-object families; raw RGB beyond frame 7 is unavailable here'))
    HUB.parent.mkdir(parents=True, exist_ok=True)
    if HUB.is_symlink():
        if HUB.resolve() != OUT.resolve():
            raise ValueError('Existing page has a different target')
    elif HUB.exists():
        raise FileExistsError(HUB)
    else:
        HUB.symlink_to(OUT, target_is_directory=True)
    print('http://localhost:8844/generic-full-2175-lora1500-lineage/p4-v2-data-samples/')


if __name__ == '__main__':
    main()
