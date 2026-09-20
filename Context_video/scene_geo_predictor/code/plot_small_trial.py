"""Curves and observation-frame overlays for development validation only."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prepare_small_trial import ROOT
from evaluate_small_trial import truth


def project(points, k, rt):
    camera = points@rt[:,:3].T+rt[:,3]
    pixels = camera@k.T
    uv = pixels[:,:2]/pixels[:,2:]
    uv[camera[:,2] <= 0] = np.nan
    return uv


def main(root):
    output = root/'reports'
    output.mkdir(exist_ok=True)
    records = [r for r in json.loads((root/'manifest.json').read_text())['records'] if r['split'] == 'val']
    colors = dict(constant='#176ca4',visual='#bd5923',GT='#228445')
    fig,axes = plt.subplots(1,3,figsize=(14,3.8))
    predictions, complete = {}, {}
    for mode in ('constant','visual'):
        run = root/'runs'/mode
        logs = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        validation = [r for r in logs if 'validation' in r]
        axes[0].plot([r['step'] for r in logs],[r['training_batch_loss'] for r in logs],color=colors[mode],label=mode,alpha=.8)
        for ax,metric in zip(axes[1:],('physical_loss','ADE_m')):
            ax.plot([r['step'] for r in validation],[r['validation'][metric] for r in validation],color=colors[mode],label=mode)
        complete[mode] = json.loads((run/'complete.json').read_text())
        with np.load(run/'validation_predictions.npz',allow_pickle=False) as archive:
            predictions[mode] = dict(zip(archive['keys'].tolist(),archive['prediction'][:,0]))
    for ax,title in zip(axes,('Training batch loss','Validation physical loss','Validation ADE (m)')):
        ax.set(title=title,xlabel='Optimizer step')
        ax.grid(alpha=.18)
        ax.legend()
    fig.suptitle('90 training / 30 development validation; not an independent test')
    fig.tight_layout()
    fig.savefig(output/'curves.png',dpi=150)
    plt.close(fig)
    for record in records:
        key = record['key']
        history,target,_,_ = truth(root,record)
        with np.load(root/'observed_context'/key/'context_geometry.npz') as geometry:
            k,rt = geometry['camera_K'],geometry['camera_world_to_view']
        rgb = Image.open(root/'context_inputs'/key/'rgb_07.png')
        fig,axes = plt.subplots(1,2,figsize=(13,4.6),gridspec_kw={'width_ratios':[1.6,1]})
        axes[0].imshow(rgb)
        vertical = 2 if record['family'] == 'gap' else 1
        trajectories = dict(GT=target,constant=predictions['constant'][key],visual=predictions['visual'][key])
        for label,trajectory in trajectories.items():
            points = np.concatenate((history[-1:],trajectory))
            pixels = project(points,k,rt)
            axes[0].plot(pixels[:,0],pixels[:,1],color=colors[label],label=label,linewidth=1.8)
            axes[1].plot(points[:,0],points[:,vertical],color=colors[label],label=label,linewidth=1.8)
        axes[0].set(xlim=(0,rgb.width),ylim=(rgb.height,0),title='RGB7 + future center trajectories')
        axes[0].axis('off')
        axis_name = 'Z' if vertical == 2 else 'Y'
        axes[1].set(xlabel='World X (m)',ylabel=f'World {axis_name} (m)',title=f'Full future trajectory, X/{axis_name} projection')
        axes[1].grid(alpha=.2)
        axes[1].legend()
        fig.suptitle(key+' | development validation; prediction is not a rendered future video')
        fig.tight_layout()
        fig.savefig(output/f'{key}.png',dpi=120)
        plt.close(fig)
    table = ''.join('<tr><td>'+mode+'</td>'+''.join(f'<td>{complete[mode][part][metric]:.6f}</td>' for part,metric in
        [('training_at_best','ADE_m'),('best_validation','ADE_m'),('best_validation','FDE_m'),('best_validation','physical_loss')])+'</tr>' for mode in ('constant','visual'))
    links = ''.join(f'<option value="{html.escape(r["key"])}.png">{html.escape(r["key"])}</option>' for r in records)
    page = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visual Scene | 120-case Trial</title><style>body{font:15px system-ui;margin:24px auto;max-width:1240px;padding:0 16px;color:#222;background:#fff}h1{font-size:24px}table{border-collapse:collapse;width:100%;margin:20px 0;display:block;overflow-x:auto}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}img{display:block;width:100%;height:auto}select{font:inherit;padding:8px;max-width:100%;margin:16px 0}small{color:#666}</style>
<h1>Visual Scene / 120-case Trial</h1><p>90 train &middot; 30 development validation &middot; seed 42</p>
<small>Not an independent test. Best checkpoints selected by validation ADE. Event proxies are recorded separately.</small>
<table><thead><tr><th>Arm</th><th>Train ADE (m)</th><th>Val ADE (m)</th><th>Val FDE (m)</th><th>Val physical loss</th></tr></thead><tbody>'''+table+'''</tbody></table>
<img src="curves.png" alt="Training and validation curves"><select id="case" aria-label="Validation case">'''+links+'''</select>
<img id="overlay" alt="GT and predicted future center trajectories on the last observation"><script>const s=document.getElementById('case'),i=document.getElementById('overlay');s.onchange=()=>i.src=s.value;s.onchange();</script>'''
    (output/'index.html').write_text(page)
    print(output/'index.html')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'small_trial_120')
    main(p.parse_args().root)
