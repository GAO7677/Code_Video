"""Fixed stratified sample from the full physical bank, with exact replay checks."""
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
import numpy as np
from scipy.spatial import ConvexHull
import torch
from safetensors import safe_open

from prepare_small_trial import ROOT, BANK, dump, sha

OUT = ROOT/'bank4200_samples_view'
SEED = 2026091901
HUB = Path('/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/generic-full-2175-lora1500-lineage/p4-v2-bank4200-samples')


def parameters(row):
    u = row['u']
    if row['family'] == 'barrier':
        return dict(geometry=30+60*u[0], unit='deg', speed=1.1+.9*u[2], dx=-.15+.30*u[1], dy=-.20+.40*u[4], heading=-10+20*u[3])
    if row['family'] == 'door':
        return dict(geometry=.38+.36*u[0], unit='m', speed=1.4+.9*u[2], dx=-.15+.30*u[1], dy=-.16+.32*u[4], heading=-8+16*u[3])
    return dict(geometry=.06+.64*u[0], unit='m', speed=1.3+1.3*u[2], dx=-.12+.24*u[1], dy=-.12+.24*u[4], heading=-6+12*u[3])


def plot_case(row, objects, pos, quat, g):
    j = next(i for i, o in enumerate(objects) if o.dynamic)
    trajectory = pos[:, j]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    corners = []
    for i, o in enumerate(objects):
        if o.dynamic:
            continue
        if o.shape != 'box':
            raise ValueError('Unexpected static geometry')
        half = np.array([o.size[k] for k in ('hx', 'hy', 'hz')])
        rotation = np.array(g.p.getMatrixFromQuaternion(quat[0, i])).reshape(3, 3)
        xyz = np.array(list(itertools.product((-1, 1), repeat=3)))*half
        corners.append(xyz@rotation.T+pos[0, i])
    for ax, dimensions, view in zip(axes, ((0, 1), (0, 2)), ('Top view (XY)', 'Side projection (XZ)')):
        d = list(dimensions)
        for xyz in corners:
            xy = xyz[:, d]
            hull = ConvexHull(xy)
            ax.add_patch(Polygon(xy[hull.vertices], facecolor='#cbd5da', edgecolor='#758b96', linewidth=.9, alpha=.72))
        ax.plot(*trajectory[:8, d].T, color='#008baf', lw=2.3, label='Observed 0-7', zorder=5)
        ax.plot(*trajectory[7:, d].T, color='#188748', lw=2.3, label='Future GT 8-48', zorder=5)
        ax.scatter(*trajectory[7, d], marker='o', c='#e59321', s=32, edgecolors='white', zorder=6, label='Context end')
        ax.scatter(*trajectory[-1, d], marker='x', c='#a32952', s=42, zorder=6, label='Final state')
        obj = objects[j]
        p = trajectory[0, d]
        if obj.shape == 'sphere' or dimensions == (0, 1):
            ax.add_patch(Circle(p, obj.size['radius'], facecolor='#56a4c3', edgecolor='#126482', alpha=.6, zorder=4))
        else:
            ax.add_patch(Rectangle(p-[obj.size['radius'], obj.size['height']/2], 2*obj.size['radius'], obj.size['height'], facecolor='#56a4c3', alpha=.6, zorder=4))
        if dimensions == (0, 2):
            ax.axhline(0, color='#657981', lw=1)
        extent = np.array([obj.size['radius'], obj.size['radius'] if dimensions == (0, 1) or obj.shape == 'sphere' else obj.size['height']/2])
        combined = np.concatenate([trajectory[:, d], np.stack((p-extent, p+extent)),
                                   *(xyz[:, d] for xyz in corners)])
        lo, hi = combined.min(0), combined.max(0)
        margin = np.maximum((hi-lo)*.08, .15)
        ax.set(xlim=(lo[0]-margin[0], hi[0]+margin[0]), ylim=(lo[1]-margin[1], hi[1]+margin[1]),
               xlabel='X (m)', ylabel='Y (m)' if dimensions == (0, 1) else 'Z (m)', title=view)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
    fig.suptitle(row['key']+' | simulator geometry + GT, not RGB', fontsize=11)
    fig.tight_layout(rect=(0, .10, 1, .93))
    fig.savefig(OUT/'images'/f"{row['key']}.png", dpi=130)
    plt.close(fig)


def main():
    torch.set_num_threads(2)
    manifest_path = BANK/'data/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Original generator source changed: '+path)
    records = manifest['records']
    if len(records) != 4200:
        raise ValueError('Unexpected bank size')
    chosen = []
    for split in ('train', 'val', 'test'):
        for family in ('barrier', 'door', 'gap'):
            pool = [(i, r) for i, r in enumerate(records) if r['split'] == split and r['family'] == family]
            pool.sort(key=lambda pair: hashlib.sha256(f"{SEED}:{pair[1]['key']}".encode()).hexdigest())
            chosen.extend(dict(r, bank_index=i) for i, r in pool[:4])
    selection = dict(seed=SEED, source_manifest_sha256=sha(manifest_path), source_bank_sha256=sha(BANK/'data/bank.safetensors'),
        method='Four lowest seeded key hashes in each split x family, from all 4200 records; no outcome selection',
        records=chosen)
    if (OUT/'selection.json').exists() and json.loads((OUT/'selection.json').read_text()) != selection:
        raise ValueError('Existing frozen selection differs')
    dump(OUT/'selection.json', selection)
    (OUT/'images').mkdir(exist_ok=True)
    sys.path.insert(0, str(BANK))
    import experiment as e
    e.g.SIM_DURATION_S = 49/30
    output = []
    with safe_open(str(BANK/'data/bank.safetensors'), framework='pt', device='cpu') as bank:
        for row in chosen:
            key = row['key']
            blueprint = json.loads((BANK/'data/blueprints'/f'{key}.json').read_text())
            case = e.make_case(row['family'], blueprint['sample_key'], np.array(row['u']))
            if json.loads(json.dumps(asdict(case.blueprint))) != blueprint:
                raise ValueError('Blueprint mismatch: '+key)
            objects, pos, vel, quat, corrections = e.audit.replay(case, row['simulation_seed'])
            packed = e.pack(case, objects, pos, vel, quat)
            for field in ('node', 'target_position', 'last_position', 'history_dt', 'future_dt'):
                if not torch.equal(packed[field], bank.get_slice(field)[row['bank_index']]):
                    raise ValueError('Replay does not match bank: '+key+'/'+field)
            if corrections != row['source_rebound_corrections']:
                raise ValueError('Rebound correction count changed')
            context_hash = hashlib.sha256(packed['node'].numpy().tobytes()).hexdigest()
            motion_hash = hashlib.sha256(packed['target_position'].numpy().tobytes()+packed['node'].numpy().tobytes()).hexdigest()
            if context_hash != row['context_hash'] or motion_hash != row['motion_hash']:
                raise ValueError('Replay fingerprint mismatch')
            plot_case(row, objects, pos, quat, e.g)
            j = next(i for i, o in enumerate(objects) if o.dynamic)
            output.append(dict(**row, params=parameters(row), positions=pos[:, j].tolist(),
                velocities=vel[:, j].tolist(), initial_position=pos[0, j].tolist(),
                image=f'images/{key}.png', replay_exact=True,
                image_sha256=sha(OUT/'images'/f'{key}.png')))
            print('VERIFIED', key, flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    colors = dict(train='#4777a5', val='#c1872b', test='#9b5688')
    for ax, family in zip(axes, ('barrier', 'door', 'gap')):
        for split, color in colors.items():
            pool = [parameters(r) for r in records if r['family'] == family and r['split'] == split]
            ax.scatter([p['geometry'] for p in pool], [p['speed'] for p in pool], s=5, alpha=.23, c=color, label=split)
            selected = [parameters(r) for r in chosen if r['family'] == family and r['split'] == split]
            ax.scatter([p['geometry'] for p in selected], [p['speed'] for p in selected], s=55, facecolors='none', edgecolors=color, lw=1.5)
        ax.set(title=family, xlabel='Angle (deg)' if family == 'barrier' else 'Width (m)', ylabel='Initial speed (m/s)')
        ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT/'coverage.png', dpi=130)
    plt.close(fig)
    dump(OUT/'data.json', dict(total=4200, counts=dict(Counter(r['split'] for r in records)),
        sampled=len(output), seed=SEED, records=output,
        bank_sha256=selection['source_bank_sha256'], all_replays_match=True, predictions=False,
        test_role='Original test split already used for development analysis, not a new blind test'))
    if HUB.is_symlink():
        if HUB.resolve() != OUT.resolve():
            raise ValueError('Unexpected existing page target')
    elif HUB.exists():
        raise FileExistsError(HUB)
    else:
        HUB.symlink_to(OUT, target_is_directory=True)
    print('http://localhost:8844/generic-full-2175-lora1500-lineage/p4-v2-bank4200-samples/')


if __name__ == '__main__':
    main()
