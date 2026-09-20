"""Compare frozen rounds without selecting models on controlled cases."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from prepare_small_trial import ROOT, dump
from plot_small_trial import project

OUT = ROOT/'two_round_release'
ARMS = {
    'Original visual': (ROOT/'small_trial_120', ROOT/'controlled_scene_eval_12'),
    'Round 1 scale': (OUT/'round1/train', OUT/'round1/control'),
    'Round 2 local': (OUT/'round2/train', OUT/'round2/control'),
}


def read(path):
    return json.loads(path.read_text())


def main():
    report = OUT/'reports'
    report.mkdir(exist_ok=True)
    results = {}
    curves, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, (train, control) in ARMS.items():
        complete = read(train/'runs/visual/complete.json')
        control_metrics = read(control/'reports/diagnostics.json')['results']['visual']
        events = read(train/'reports/validation_diagnostics.json')['results']['visual']['summary']['all']
        results[label] = dict(best_step=read(train/'runs/visual/best.json')['step'],
            validation=complete['best_validation'], training=complete['training_at_best'],
            events=events, control=control_metrics,
            same_future_false_responses=[dict(family=f, **p) for f, g in control_metrics['groups'].items()
                for p in g['pairs'] if p['gt_change_m'] < 1e-5])
        entries = [json.loads(line) for line in (train/'runs/visual/metrics.jsonl').read_text().splitlines()]
        vals = [r for r in entries if 'validation' in r]
        axes[0].plot([r['step'] for r in vals], [100*r['validation']['ADE_m'] for r in vals], label=label)
        axes[1].plot([r['step'] for r in vals], [r['validation']['physical_loss'] for r in vals], label=label)
    for ax, title in zip(axes, ('Development validation ADE (cm)', 'Position + 0.2 velocity loss')):
        ax.set(xlabel='Training step', title=title)
        ax.grid(alpha=.2)
        ax.legend()
    curves.tight_layout()
    curves.savefig(report/'curves.png', dpi=140)
    plt.close(curves)
    selected = min(results, key=lambda key: results[key]['validation']['ADE_m'])
    dump(report/'comparison.json', dict(results=results, selected=selected,
        selection_rule='Lowest 30-case development validation ADE; controlled cases do not select weights',
        new_training_runs=2, independent_test=False,
        limitations=['Single seed, 90 training and 30 development validation cases.',
            'Only one fixed motion context for each controlled scene family.',
            'Round 2 was designed using development diagnostics, not an independent blind test.',
            'Event metrics are 30 Hz proxies, and penetration omits unsupported puck geometry.']))
    rows = []
    for label, value in results.items():
        v = value['validation']
        rows.append(f"| {label} | {value['best_step']} | {100*value['training']['ADE_m']:.2f} | {100*v['ADE_m']:.2f} | {100*v['FDE_m']:.2f} | {v['physical_loss']:.5f} |")
    markdown = '# Two-round Results\n\n'
    markdown += '| Version | Best step | Train ADE cm | Val ADE cm | Val FDE cm | Physical loss |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)
    markdown += f'\n\nSelected by development validation ADE: **{selected}**. No independent test claim.\n\n'
    markdown += '| Version | Family | Control ADE cm | Response error cm |\n|---|---|---:|---:|\n'
    html_rows = []
    for label, value in results.items():
        for family, group in value['control']['groups'].items():
            ade, response = 100*group['mean_ADE_m'], 100*group['mean_response_error_m']
            markdown += f'| {label} | {family} | {ade:.2f} | {response:.2f} |\n'
            html_rows.append(f'<tr><td>{label}</td><td>{family}</td><td>{ade:.2f}</td><td>{response:.2f}</td></tr>')
    markdown += '\n## Identical-GT Future Pairs\n\n'
    for label, value in results.items():
        for pair in value['same_future_false_responses']:
            markdown += f"- {label}, {pair['a']} / {pair['b']}: false predicted difference {100*pair['predicted_change_m']:.2f} cm.\n"
    markdown += '\nSee comparison.json for complete event metrics and all control pairs. No third training run.\n'
    (report/'results.md').write_text(markdown)
    records = read(ARMS['Original visual'][1]/'manifest.json')['records']
    arrays = {}
    for label, (_, control) in ARMS.items():
        with np.load(control/'reports/predictions.npz') as data:
            if data['keys'].tolist() != [r['key'] for r in records]:
                raise ValueError('Comparison record ordering changed')
            arrays[label] = data['visual']
            arrays['GT'] = data['GT']
    colors = {'GT':'#17864b', 'Original visual':'#4f66b4', 'Round 1 scale':'#d18620', 'Round 2 local':'#c03e57'}
    for family in ('barrier', 'door', 'gap'):
        ids = [i for i, r in enumerate(records) if r['family'] == family]
        fig, axes = plt.subplots(4, 2, figsize=(13, 15), gridspec_kw={'width_ratios':[1.6, 1]})
        paths = []
        vertical = 2 if family == 'gap' else 1
        for row, i in enumerate(ids):
            record = records[i]
            control = ARMS['Original visual'][1]
            rgb = Image.open(control/'context_inputs'/record['key']/'rgb_07.png')
            with np.load(control/'observed_context'/record['key']/'context_geometry.npz') as data:
                k, rt, last = data['camera_K'], data['camera_world_to_view'], data['positions_world'][-1, 0]
            axes[row, 0].imshow(rgb)
            for label, color in colors.items():
                trajectory = np.concatenate((last[None], arrays[label][i]))
                uv = project(trajectory, k, rt)
                axes[row, 0].plot(*uv.T, color=color, label=label, linewidth=1.6)
                axes[row, 1].plot(trajectory[:, 0], trajectory[:, vertical], color=color, label=label)
                paths.append(trajectory[:, [0, vertical]])
            axes[row, 0].set(xlim=(0, rgb.width), ylim=(rgb.height, 0), title=f"{record['variable']} = {record['value']:.2f} m")
            axes[row, 0].axis('off')
            axes[row, 1].set(xlabel='World X (m)', ylabel='World Z (m)' if vertical == 2 else 'World Y (m)')
            axes[row, 1].grid(alpha=.2)
            axes[row, 1].legend(fontsize=8)
        points = np.concatenate(paths)
        low, high = points.min(0), points.max(0)
        margin = np.maximum((high-low)*.05, .02)
        for row in range(4):
            axes[row, 1].set(xlim=(low[0]-margin[0], high[0]+margin[0]), ylim=(low[1]-margin[1], high[1]+margin[1]))
        fig.suptitle(f'{family}: identical observed motion, different geometry\nRGB7 trajectory overlays; not generated videos')
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(report/f'{family}.png', dpi=120)
        plt.close(fig)
    blocks = ''.join(f'<section><h2>{f}</h2><img src="{f}.png" alt="{f} matched-scene trajectory comparison"></section>' for f in ('barrier', 'door', 'gap'))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-round predictor review</title><style>body{margin:24px auto;padding:0 16px;max-width:1240px;font:15px system-ui;color:#202020;background:white}h1{font-size:26px}h2{font-size:21px}img{display:block;width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:24px}.scroll{overflow-x:auto}table{border-collapse:collapse;min-width:640px;width:100%}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}p{line-height:1.5}</style><h1>Two-round Predictor Review</h1><p>90 train / 30 development validation / 12 controlled diagnostic cases / one seed / no DiT</p>'''
    page += f'<p>Selected by validation ADE: <strong>{selected}</strong>. Controlled cases are development diagnostics, not an independent test.</p><img src="curves.png" alt="Validation curves by training step">'
    page += '<div class="scroll"><table><tr><th>Version</th><th>Family</th><th>Control ADE cm</th><th>Response error cm</th></tr>'+''.join(html_rows)+'</table></div>'+blocks+'</html>'
    (report/'index.html').write_text(page)
    print(markdown)


if __name__ == '__main__':
    main()
