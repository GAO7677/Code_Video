"""Group-wise actual RGB overlays and controlled trajectory-response plots."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from controlled_scene_eval import OUT
from evaluate_small_trial import truth
from plot_small_trial import project


def plot(root):
    report=root/'reports'
    manifest=json.loads((root/'manifest.json').read_text())
    records=manifest['records']
    metrics=json.loads((report/'diagnostics.json').read_text())['results']
    with np.load(report/'predictions.npz') as data:
        assert data['keys'].tolist()==[r['key'] for r in records]
        predictions={k:data[k] for k in ('GT','constant','visual')}
    colors={'GT':'#228445','constant':'#176ca4','visual':'#bd5923'}
    response_rows=[]
    for family in ('barrier','door','gap'):
        ids=[i for i,r in enumerate(records) if r['family']==family]
        fig,axes=plt.subplots(4,2,figsize=(13,15),gridspec_kw={'width_ratios':[1.6,1]})
        trajectories=[]
        for row,i in enumerate(ids):
            record=records[i]; key=record['key']
            history,*_=truth(root,record)
            rgb=Image.open(root/'context_inputs'/key/'rgb_07.png')
            with np.load(root/'observed_context'/key/'context_geometry.npz') as data:
                k,rt=data['camera_K'],data['camera_world_to_view']
            vertical=2 if family=='gap' else 1
            axes[row,0].imshow(rgb)
            for mode in colors:
                path=np.concatenate((history[-1:],predictions[mode][i]))
                uv=project(path,k,rt)
                axes[row,0].plot(*uv.T,color=colors[mode],label=mode,linewidth=1.8)
                axes[row,1].plot(path[:,0],path[:,vertical],color=colors[mode],label=mode)
                trajectories.append(path[:,[0,vertical]])
            axes[row,0].set(xlim=(0,rgb.width),ylim=(rgb.height,0),
                title=f'{record["variable"]} = {record["value"]:.2f} m')
            axes[row,0].axis('off')
            axes[row,1].set(xlabel='World X (m)',ylabel='World Z (m)' if vertical==2 else 'World Y (m)')
            axes[row,1].grid(alpha=.2)
            axes[row,1].legend()
        points=np.concatenate(trajectories)
        low,high=points.min(0),points.max(0)
        margin=np.maximum((high-low)*.05,.02)
        for row in range(4):
            axes[row,1].set(xlim=(low[0]-margin[0],high[0]+margin[0]),ylim=(low[1]-margin[1],high[1]+margin[1]))
        fig.suptitle(f'{family}: identical RGB0-7 motion; one static geometry variable\nGT / constant / visual, frozen step 450; RGB7 overlays, not generated video')
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(report/f'{family}_overlays.png',dpi=120)
        plt.close(fig)
        # Subtract the same reference variant at each future time, in world coordinates.
        fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey='col')
        times=np.arange(8,49)/30
        for row,i in enumerate(ids[1:]):
            for mode in colors:
                delta=predictions[mode][i]-predictions[mode][ids[0]]
                for dim in range(3):
                    axes[row,dim].plot(times,delta[:,dim],color=colors[mode],label=mode)
                    axes[row,dim].set(title=f'{records[i]["value"]:.2f} - {records[ids[0]]["value"]:.2f} m | delta {"XYZ"[dim]}',
                                      xlabel='Time from RGB0 (s)',ylabel='Trajectory difference (m)')
                    axes[row,dim].grid(alpha=.2)
        axes[0,0].legend()
        fig.suptitle(f'{family}: matched physical response, relative to first variant')
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(report/f'{family}_response.png',dpi=120)
        plt.close(fig)
        for mode in ('constant','visual'):
            group=metrics[mode]['groups'][family]
            response_rows.append(f'<tr><td>{family}</td><td>{mode}</td>'+
                ''.join(f'<td>{100*group[k]:.2f}</td>' for k in
                    ('mean_ADE_m','mean_predicted_change_m','mean_gt_change_m','mean_response_error_m'))+'</tr>')
    blocks=''.join(f'<section><h2>{family}</h2><img src="{family}_overlays.png" alt="{family} four single-variable RGB overlays"><img src="{family}_response.png" alt="{family} predicted and true trajectory differences"></section>' for family in ('barrier','door','gap'))
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Controlled Scene Evaluation</title>
<style>body{margin:24px auto;padding:0 16px;max-width:1240px;font:15px system-ui;color:#202020;background:#fff}h1{font-size:26px}h2{font-size:21px}img{display:block;width:100%;height:auto}section{border-top:1px solid #ddd;margin-top:24px}table{border-collapse:collapse;min-width:700px;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #ddd}.scroll{overflow-x:auto}p{line-height:1.5}</style></head><body>
<h1>Controlled Scene Evaluation</h1><p>12 cases / 3 fixed motion contexts / frozen step 450 / no retraining</p>
<p>Door and gap widths are within the training control ranges. Barrier translation is out of distribution. One context per family; paired comparisons are not independent replicates.</p>
<div class="scroll"><table><thead><tr><th>Family</th><th>Arm</th><th>ADE (cm)</th><th>Pred change (cm)</th><th>GT change (cm)</th><th>Response error (cm)</th></tr></thead><tbody>'''+''.join(response_rows)+'''</tbody></table></div>'''+blocks+'''</body></html>'''
    (report/'index.html').write_text(page)
    print(report/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=OUT)
    plot(p.parse_args().root)
