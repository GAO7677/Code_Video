"""Create isolated experiment views; reuse immutable observations, not labels in forward."""
import argparse
import json
from pathlib import Path
import numpy as np

from prepare_small_trial import ROOT,dump,sha
from align_observed_depth import crop_transform,resize_crop,world_rays
from observed_surface_scale import estimate_scale

OUT=ROOT/'two_round_release'


def prepare(source,dest):
    source=source.resolve();dest=dest.resolve()
    manifest=json.loads((source/'manifest.json').read_text())
    if (dest/'manifest.json').exists():
        if json.loads((dest/'manifest.json').read_text())!=manifest:
            raise ValueError('Experiment manifest changed')
    else:
        dump(dest/'manifest.json',manifest)
    for folder in ('samples','observed_context','context_inputs','observed_masks','raw_probe','render_context8','reviews'):
        link=dest/folder
        if link.exists():
            if link.resolve()!=(source/folder).resolve():raise ValueError('Unexpected source link')
        else:link.symlink_to(source/folder,target_is_directory=True)
    if (source/'simulation_audit.json').exists() and not (dest/'simulation_audit.json').exists():
        (dest/'simulation_audit.json').symlink_to(source/'simulation_audit.json')
    summaries=[]
    for row in manifest['records']:
        key=row['key']; out=dest/'aligned_scene'/key
        old_dir=source/'aligned_scene'/key
        old=json.loads((old_dir/'report.json').read_text())
        geometry_path=source/'observed_context'/key/'context_geometry.npz'
        raw_path=source/'raw_probe'/key/'raw_vggt.npz'
        mask_path=source/'observed_masks'/key/'observed_masks.npz'
        for p,field in ((geometry_path,'geometry'),(raw_path,'raw'),(mask_path,'masks')):
            if sha(p)!=old['provenance'][field]['sha256']: raise ValueError('Original inputs changed')
        if out.exists():raise FileExistsError(out)
        with np.load(geometry_path) as data: geometry={k:data[k] for k in data.files}
        with np.load(raw_path) as data: depth=data['depth'][...,0].astype(float)
        with np.load(mask_path) as data:
            masks=data['per_object']; accepted=data['accepted']
        if masks.shape[1]!=1 or not accepted.all():raise ValueError('Invalid reviewed masks')
        transform=crop_transform(masks.shape[2:])
        mask=np.stack([resize_crop(x[0],transform,is_mask=True) for x in masks])
        with np.load(old_dir/'coarse_static_points.npz') as data: arrays={k:data[k] for k in data.files}
        result=estimate_scale(depth,mask,geometry,arrays['K_processed'])
        scale=result['scale_midpoint']
        origin,rays=world_rays(arrays['K_processed'],arrays['RT'],depth.shape[1:])
        xyz=origin+rays[None]*depth[...,None]*scale
        xyz[~arrays['static_valid']]=0
        arrays['world_midpoint']=xyz.astype(np.float32)
        arrays['scale_interval']=np.array([scale,scale])
        out.mkdir(parents=True)
        np.savez_compressed(out/'coarse_static_points.npz',**arrays)
        report={**old,'schema':'observed_surface_scale_v1','baseline_alignment':old['alignment'],
            'alignment':result,'adapter_sha256':sha(Path(__file__).with_name('observed_surface_scale.py')),
            'output_npz_sha256':sha(out/'coarse_static_points.npz'),
            'scale_interval_interpretation':result['scale_interval_interpretation']}
        dump(out/'report.json',report)
        summaries.append(dict(key=key,old_scale=old['alignment']['scale_midpoint'],new_scale=scale,
                              frame_span=result['frame_scale_relative_span']))
        print('SCALE_READY',key,round(scale/old['alignment']['scale_midpoint'],5),flush=True)
    dump(dest/'scale_summary.json',dict(records=summaries,static_gt_used=False,future_used=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.dest)
