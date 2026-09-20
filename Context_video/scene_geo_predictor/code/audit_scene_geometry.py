"""CPU-only physics-surface audit of existing observation scene caches."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import cKDTree

from align_observed_depth import world_rays, ray_aabb_depth, morph3
from controlled_scene_eval import OUT
from prepare_small_trial import dump, sha
from scene_token_cache import point_inputs
from trial_metrics import rotation_xyzw


def reference(origin, rays, boxes):
    """Nearest physical box or z=0 plane, not a full rendered-scene Z buffer."""
    depth=np.full(rays.shape[:-1],np.inf)
    labels=np.full(depth.shape,-1,dtype=int)
    floor=np.full_like(depth,np.inf)
    np.divide(-origin[2],rays[...,2],out=floor,where=rays[...,2]<-1e-12)
    ground=np.isfinite(floor)&(floor>0)
    depth[ground]=floor[ground]; labels[ground]=len(boxes)
    for i,box in enumerate(boxes):
        rotation=box['rotation']
        local_origin=(origin-box['center'])@rotation
        near,_,hit=ray_aabb_depth(local_origin,rays@rotation,np.zeros(3),2*box['half'])
        hit &= near<depth
        depth[hit]=near[hit]; labels[hit]=i
    points=origin+rays*np.where(np.isfinite(depth),depth,0)[...,None]
    return points,labels,depth


def interior_labels(labels,count):
    result=np.full_like(labels,-1)
    for i in range(count):
        keep=labels==i
        for _ in range(2):
            keep=morph3(keep,dilate=False)
        result[keep]=i
    return result


def critical_region(family,points,labels,boxes,value):
    selected=np.zeros(labels.shape,dtype=bool)
    for i,box in enumerate(boxes):
        name=box['name']; p=points
        if family=='barrier':
            keep=(p[...,2]<.30)&(np.abs(p[...,1])<.45)
        elif family=='door' and name in ('door_frame_left','door_frame_right'):
            keep=(np.abs(np.abs(p[...,1])-value/2)<.10)&(p[...,2]>.02)&(p[...,2]<.55)
        elif family=='gap' and name in ('left_platform','right_platform'):
            edge=box['center'][0]+(box['half'][0] if name=='left_platform' else -box['half'][0])
            keep=(np.abs(p[...,0]-edge)<.10)&(np.abs(p[...,1])<.35)&(np.abs(p[...,2]-.48)<.005)
        else:
            continue
        selected |= (labels==i)&keep
    return selected


def stats(values):
    values=np.asarray(values)
    if not len(values):
        return dict(n=0,median=None,p90=None,mean=None)
    return dict(n=int(len(values)),median=float(np.median(values)),
                p90=float(np.quantile(values,.9)),mean=float(values.mean()))


def open_volume(family,points,boxes,value):
    """Static diagnostic volumes only; points are not an occupancy field."""
    if family=='door':
        box=next(b for b in boxes if b['name']=='door_frame_left')
        x=box['center'][0]; hx=box['half'][0]
        return ((np.abs(points[:,0]-x)<hx-.02)&(np.abs(points[:,1])<value/2-.02)
                &(points[:,2]>.05)&(points[:,2]<.95))
    if family=='gap':
        box=next(b for b in boxes if b['name']=='left_platform')
        edge=box['center'][0]+box['half'][0]
        return ((points[:,0]>edge+.02)&(points[:,0]<edge+value-.02)&
                (np.abs(points[:,1])<.30)&(np.abs(points[:,2]-.48)<.03))
    return np.zeros(len(points),dtype=bool)


def coverage(target,points):
    if not len(target) or not len(points):
        return dict(target_count=len(target),point_count=len(points),distance_m=stats([]),
                    within_2cm=0.,within_5cm=0.,within_10cm=0.)
    distance=cKDTree(points).query(target,workers=1)[0]
    return dict(target_count=len(target),point_count=len(points),distance_m=stats(distance),
                within_2cm=float((distance<=.02).mean()),within_5cm=float((distance<=.05).mean()),
                within_10cm=float((distance<=.10).mean()))


def boxes_for(root,key):
    sample=root/'samples'/key
    actors=json.loads((sample/'metadata.json').read_text())['actors']
    with np.load(sample/'raw/states_xyzw.npz') as data:
        names=data['object_names'].tolist()
        boxes=[]
        for i,name in enumerate(names):
            a=actors[name]
            if a['dynamic']:
                continue
            if a['shape']!='box':
                raise ValueError('Only finite static OBBs supported')
            if not np.array_equal(data['positions'][:8,i],np.broadcast_to(data['positions'][0,i],(8,3))):
                raise ValueError('Nonstatic geometry in reference')
            boxes.append(dict(name=name,center=data['positions'][0,i].astype(float),
                half=np.array([a['size_m'][v] for v in ('hx','hy','hz')]),
                rotation=rotation_xyzw(data['quats'][0,i])))
    return boxes


def draw_boxes(ax,boxes,axes):
    from scipy.spatial import ConvexHull
    corners=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for box in boxes:
        pts=((corners*box['half'])@box['rotation'].T+box['center'])[:,axes]
        hull=ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices],fill=False,edgecolor='#242424',linewidth=1.2))


def case_audit(root,row,out):
    key=row['key']; family=row['family']
    aligned_path=root/'aligned_scene'/key/'coarse_static_points.npz'
    raw_path=root/'raw_probe'/key/'raw_vggt.npz'
    token_path=root/'scene_features'/key/'scene_tokens.npz'
    receipt=json.loads((token_path.parent/'report.json').read_text())
    for path,field in ((aligned_path,'aligned_points_sha256'),(raw_path,'raw_sha256'),(token_path,'output_sha256')):
        if sha(path)!=receipt[field]:
            raise ValueError('Cached input changed')
    with np.load(aligned_path) as data:
        aligned={k:data[k] for k in data.files}
    with np.load(raw_path) as data:
        raw={k:data[k] for k in ('images','depth_conf','depth')}
    with np.load(token_path) as data:
        tokens={k:data[k] for k in ('scene_xyz','source_flat_indices','scene_mask')}
    point,sample_ids,_,_=point_inputs(aligned,raw,65536)
    world=aligned['world_midpoint']; valid=aligned['static_valid']
    h,w=valid.shape[1:]; size=h*w
    flat=world.reshape(-1,3)
    token_ids=tokens['source_flat_indices'][tokens['scene_mask']]
    if not np.array_equal(flat[token_ids],tokens['scene_xyz'][tokens['scene_mask']]):
        raise ValueError('Token representative positions changed')
    if not np.isin(token_ids,sample_ids).all() or not np.array_equal(point['coord'],flat[sample_ids]):
        raise ValueError('Cannot reproduce encoder source sample')
    origin,rays=world_rays(aligned['K_processed'],aligned['RT'],(h,w))
    boxes=boxes_for(root,key)
    truth,labels,depth=reference(origin,rays,boxes)
    interior=interior_labels(labels,len(boxes))
    critical=critical_region(family,truth,labels,boxes,row['value'])
    ref_flat=truth.reshape(-1,3)
    critical_flat=np.tile(critical.reshape(-1),8)
    static_inside=np.tile((interior>=0).reshape(-1),8)
    stages={'aligned':np.flatnonzero(valid.reshape(-1)),'encoder_input':sample_ids,'token_anchors':token_ids}
    stage_results={}
    reference_targets=ref_flat[critical.reshape(-1)&valid.any(0).reshape(-1)]
    for name,ids in stages.items():
        selected=ids[static_inside[ids]]
        delta=flat[selected]-ref_flat[selected%size]
        crit=ids[critical_flat[ids]]
        stage_results[name]=dict(total_count=len(ids),surface_error_m=stats(np.linalg.norm(delta,axis=-1)),
            signed_camera_depth_error_m=stats(delta@aligned['RT'][:,:3][2]),
            critical_count=len(crit),critical_source_pixels=len(np.unique(crit%size)),
            critical_actual_coverage=coverage(reference_targets,flat[crit]),
            critical_oracle_coordinate_coverage=coverage(reference_targets,ref_flat[crit%size]),
            open_volume_point_count=int(open_volume(family,flat[ids],boxes,row['value']).sum()) if family!='barrier' else None)
    per_actor=[]
    for i,box in enumerate(boxes):
        chosen=valid&(interior[None]==i)
        delta=(world-truth[None])[chosen]
        depths=np.broadcast_to(depth,valid.shape)[chosen]
        estimated=((world-origin)@aligned['RT'][:,:3][2])[chosen]
        per_actor.append(dict(name=box['name'],surface_error_m=stats(np.linalg.norm(delta,axis=-1)),
            camera_depth_bias_m=stats(estimated-depths)))
    selected=valid&(interior[None]>=0)
    vg=raw['depth'][...,0][selected].astype(float)
    gt=np.broadcast_to(depth,valid.shape)[selected]
    best_scale=float(np.dot(vg,gt)/np.dot(vg,vg))
    residual=vg*best_scale-gt
    ray_norm=np.broadcast_to(np.linalg.norm(rays,axis=-1),valid.shape)[selected]
    scale=dict(used=float(aligned['scale_interval'].mean()),accepted_interval=aligned['scale_interval'].tolist(),
        oracle_least_squares_scale=best_scale,oracle_scale_only_surface_error_m=stats(np.abs(residual)*ray_norm),
        purpose='GT-fit diagnostic only; never applied to caches or predictor; no shift fitted')
    face=None
    if family in ('door','gap'):
        axis=0 if family=='door' else 2
        box=next(b for b in boxes if b['name']==('door_frame_left' if family=='door' else 'left_platform'))
        value=box['center'][axis]+(-1 if family=='door' else 1)*box['half'][axis]
        face_mask=selected&(np.abs(truth[None,...,axis]-value)<1e-5)
        observed=world[...,axis][face_mask]
        face=dict(name='door_front_X' if family=='door' else 'platform_top_Z',reference_m=float(value),
                  estimated_m=stats(observed),bias_m=stats(observed-value))
    result=dict(key=key,family=family,value=row['value'],stages=stage_results,actors=per_actor,
                scale_diagnostic=scale,face_diagnostic=face,critical_reference_pixels=len(reference_targets),
                provenance={str(p):sha(p) for p in (aligned_path,raw_path,token_path,
                    root/'samples'/key/'metadata.json',root/'samples'/key/'raw/states_xyzw.npz')},
                sample_source_mapping_verified=True)
    dump(out/(key+'.json'),result)
    plot_case(out,row,world,valid,raw,truth,labels,critical,boxes,stages,size,stage_results)
    print('AUDIT',key,'surface_median_m',stage_results['aligned']['surface_error_m']['median'],
          'critical_tokens',stage_results['token_anchors']['critical_count'],flush=True)
    return result


def plot_case(out,row,world,valid,raw,truth,labels,critical,boxes,stages,size,metrics):
    key=row['key']; h,w=valid.shape[1:]
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    rgb=raw['images'][7].transpose(1,2,0)
    axes[0,0].imshow(rgb)
    y,x=np.where(critical)
    axes[0,0].scatter(x,y,s=1,c='#ec3a78',alpha=.6)
    ids=stages['token_anchors']%size
    relevant=labels.reshape(-1)[ids]<len(boxes)
    relevant &= labels.reshape(-1)[ids]>=0
    axes[0,0].scatter(ids[relevant]%w,ids[relevant]//w,s=7,facecolors='none',edgecolors='#19cfdd',linewidths=.5)
    axes[0,0].set(title='RGB7: pink critical surfaces / cyan token source pixels',xlim=(0,w),ylim=(h,0))
    axes[0,0].axis('off')
    error=np.linalg.norm(world[7]-truth,axis=-1)
    mask=valid[7]&(interior_labels(labels,len(boxes))>=0)
    view=np.where(mask,error,np.nan)
    im=axes[0,1].imshow(view,cmap='magma',vmin=0,vmax=.5)
    axes[0,1].set_title('Known collider interior: same-ray error (m)')
    axes[0,1].axis('off');fig.colorbar(im,ax=axes[0,1],fraction=.046)
    axes[0,2].imshow(rgb)
    axes[0,2].contour((labels>=0)&(labels<len(boxes)),levels=[.5],colors=['#f2aa00'],linewidths=.6)
    axes[0,2].set_title('Physics-only projected collider reference')
    axes[0,2].axis('off')
    dims=[0,2] if row['family']=='gap' else [0,1]
    source_roi=(labels>=0)&(labels<len(boxes))
    for ax,(name,ids) in zip(axes[1],stages.items()):
        chosen=ids[source_roi.reshape(-1)[ids%size]]
        if len(chosen)>10000:
            chosen=chosen[np.linspace(0,len(chosen)-1,10000,dtype=int)]
        pts=world.reshape(-1,3)[chosen]
        ax.scatter(pts[:,dims[0]],pts[:,dims[1]],s=2,c='#176ca4',alpha=.4)
        gt=truth[source_roi]
        ax.scatter(gt[::8,dims[0]],gt[::8,dims[1]],s=2,c='#e17d22',alpha=.3)
        draw_boxes(ax,boxes,dims)
        ax.set(title=f'{name}: {metrics[name]["critical_count"]} critical anchors',
               xlabel='World X (m)',ylabel='World '+('Z' if dims[1]==2 else 'Y')+' (m)')
        if row['family']=='gap': ax.set(xlim=(-1.7,2.3),ylim=(-.2,1.1))
        elif row['family']=='door': ax.set(xlim=(-.2,1.7),ylim=(-1.8,1.8))
        else: ax.set(xlim=(-.5,2.1),ylim=(-1.1,1.1))
        ax.grid(alpha=.15)
    fig.suptitle(key+' | blue predicted points / orange physics reference / black box outlines\nReference excludes room props; token anchors are not the full feature receptive fields')
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/(key+'.png'),dpi=120)
    plt.close(fig)


def main(root):
    out=root/'geometry_audit'
    out.mkdir(exist_ok=True)
    rows=json.loads((root/'manifest.json').read_text())['records']
    results=[case_audit(root,row,out) for row in rows]
    dump(out/'summary.json',dict(records=results,source_sha256=sha(Path(__file__)),
        scope='CPU-only evaluation; existing model/caches unchanged; no future geometry used',
        limitations=['Reference includes static physics boxes and plane, not a full rendered-scene Z buffer.',
                     'Dynamic regions use existing dilated SAM masks, not GT segmentation.',
                     'Surface accuracy excludes 2-pixel collider silhouette bands; decorative occluders remain possible.',
                     'Critical regions are predeclared static geometry bands, not selected by model errors.',
                     'Token anchor coverage is not proof of what the full Utonia feature encodes.',
                     'Points inside an expected empty opening are suspicious geometry, not an occupancy/solid prediction.',
                     'Oracle-coordinate coverage holds sampling fixed and removes position error only.']))
    table=''.join('<tr><td>'+r['key']+'</td><td>'+f'{100*r["stages"]["aligned"]["surface_error_m"]["median"]:.2f}'+'</td><td>'+str(r['stages']['token_anchors']['critical_count'])+'</td><td>'+f'{100*r["stages"]["token_anchors"]["critical_actual_coverage"]["within_5cm"]:.1f}'+'</td></tr>' for r in results)
    pictures=''.join(f'<h2>{r["key"]}</h2><a href="{r["key"]}.png"><img src="{r["key"]}.png" alt="{r["key"]} geometry audit"></a>' for r in results)
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene Geometry Audit</title><style>body{font:15px system-ui;max-width:1400px;margin:24px auto;padding:0 16px;color:#222}h1{font-size:26px}h2{font-size:20px}img{width:100%;height:auto}table{border-collapse:collapse;min-width:650px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow-x:auto}</style><h1>Scene Geometry Audit</h1><p>12 controlled cases / existing caches / no training / physics-only surface reference</p><p>Reference excludes decorative room props. Token anchor coverage does not measure the full feature receptive field.</p><div class="scroll"><table><tr><th>Case</th><th>Median surface error (cm)</th><th>Critical token anchors</th><th>Critical 5cm coverage (%)</th></tr>'''+table+'</table></div>'+pictures)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=OUT)
    main(p.parse_args().root)
