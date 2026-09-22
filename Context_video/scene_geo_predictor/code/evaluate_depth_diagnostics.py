"""GT-aligned diagnostic only: frozen RGB0 calibration, RGB1-7 evaluation."""
import json
from pathlib import Path
import numpy as np
import cv2
from compare_depth_diagnostics import ROOT,OUT,read,write,cases,rgb
cv2.setNumThreads(2)


def metrics(pred,gt,mask,invalid):
    if not mask.any():return {'status':'NO_PIXELS'}
    p=pred[mask];g=gt[mask]
    return dict(status='EXECUTED',pixels=int(mask.sum()),AbsRel=float(np.mean(np.abs(p-g)/g)),
                RMSE_m=float(np.sqrt(np.mean((p-g)**2))),delta1=float(np.mean(np.maximum(p/g,g/p)<1.25)),
                invalid_aligned_inverse_fraction=float(np.mean(invalid[mask])))


def run():
    all_rows=[]
    assets=OUT/'comparison_assets';assets.mkdir(exist_ok=True)
    yy,xx=np.indices((360,640));checker=(xx+yy)%2==0
    for cid in cases():
        gt_archive=np.load(OUT/'gt'/f'{cid}.npz');gt=gt_archive['depth'].astype(float);seg=gt_archive['mask']
        ids=read(OUT/'gt'/f'{cid}_ids.json');ball_id=ids['pilot_ball'];valid=(gt>.2)&(gt<15)&np.isfinite(gt)
        calibration=valid[0]&checker
        test=valid&~checker[None];test[0]=False
        objects=(seg>0)&(seg!=ball_id)
        # Fixed-camera static pixels with stable GT range, excluding all observed
        # dynamic ball footprints. No optical-flow or estimated masks in scoring.
        ball_union=np.any(seg==ball_id,axis=0)
        stable=(np.ptp(gt,axis=0)<1e-4)&~ball_union&np.all(valid,axis=0)
        raw=np.load(ROOT/'vision_v1/raw_vggt'/f'{cid}.npz')['depth'].squeeze(-1)
        vggt=np.stack([cv2.resize(x.astype(np.float32),(640,360),interpolation=cv2.INTER_LINEAR) for x in raw])
        maps={'vggt':1/np.maximum(vggt,1e-6)}
        for model in ['da','vda']:
            maps[model]=np.load(OUT/'predictions'/model/(cid+'.npz'))['relative_inverse_depth']
        panels=[]
        frames=rgb(cid)
        def color(depth):
            v=np.clip((1/np.maximum(depth,.2)-1/15)/(1/.2-1/15),0,1)
            return cv2.applyColorMap((v*255).astype(np.uint8),cv2.COLORMAP_TURBO)
        for model,inv in maps.items():
            inv=inv.astype(float)
            if not np.isfinite(inv).all():raise ValueError(f'nonfinite {model}/{cid}')
            design=np.column_stack([inv[0][calibration],np.ones(calibration.sum())])
            coef=np.linalg.lstsq(design,1/gt[0][calibration],rcond=None)[0]
            if coef[0]<=0:raise ValueError(f'Unexpected inverse-depth ordering: {model}/{cid}')
            aligned_inv=inv*coef[0]+coef[1];invalid=aligned_inv<=0
            pred=np.clip(1/np.maximum(aligned_inv,1/15),.2,15)
            regions={'all':test,'objects':test&(seg>0),'static_objects':test&objects}
            row=dict(case_id=cid,family=cid.split('_')[1] if 'support_edge' not in cid else 'support_edge',model=model,
                     alignment_scale=float(coef[0]),alignment_shift=float(coef[1]),
                     alignment_uses_GT=True,metrics={k:metrics(pred,gt,m,invalid) for k,m in regions.items()},
                     static_inverse_temporal_std=float(np.mean(np.std(aligned_inv[:,stable],axis=0))))
            # Existing VGGT scale is evaluated separately, never labelled a fair
            # raw-metric comparison with relative-only DA and VDA.
            if model=='vggt':
                scale=read(ROOT/'vision_v3/estimates'/cid/'report.json')['vggt_depth_scale']['metric_scale']
                row['existing_sphere_scaled_metrics']=metrics(vggt*scale,gt,test,np.zeros_like(test))
            all_rows.append(row)
            target=assets/cid;target.mkdir(exist_ok=True)
            for f in range(8):
                colored=color(pred[f]);overlay=cv2.addWeighted(cv2.cvtColor(frames[f],cv2.COLOR_RGB2BGR),.35,colored,.65,0)
                cv2.imwrite(str(target/f'{model}_{f}.webp'),overlay)
            error=np.clip(np.abs(pred[7]-gt[7])/gt[7],0,1)
            cv2.imwrite(str(target/f'{model}_error7.webp'),cv2.applyColorMap((error*255).astype(np.uint8),cv2.COLORMAP_INFERNO))
        for f in range(8):
            target=assets/cid
            cv2.imwrite(str(target/f'rgb_{f}.webp'),cv2.cvtColor(frames[f],cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(target/f'gt_{f}.webp'),color(gt[f]))
        print('SCORED',cid,flush=True)
    summary={}
    for model in maps:
        rows=[r for r in all_rows if r['model']==model]
        summary[model]={region:{key:float(np.mean([r['metrics'][region][key] for r in rows])) for key in ['AbsRel','RMSE_m','delta1','invalid_aligned_inverse_fraction']} for region in regions}
        summary[model]['static_inverse_temporal_std']=float(np.mean([r['static_inverse_temporal_std'] for r in rows]))
    write(OUT/'comparison.json',dict(status='EXECUTED',aggregation='case macro mean',summary=summary,rows=all_rows))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
