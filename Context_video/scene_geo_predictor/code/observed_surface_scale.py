"""Metric scale from observed sphere/upright-cylinder surfaces, no static GT."""
import numpy as np
from align_observed_depth import world_rays, morph3


def ray_surface_depth(origin,rays,center,size):
    """Scope: equal-size sphere, otherwise circular upright cylinder."""
    q=np.asarray(origin)-center
    size=np.asarray(size)
    if np.any(size<=0) or not np.isclose(size[0],size[1],atol=1e-6):
        raise ValueError('Unsupported observed shape dimensions')
    sphere=np.allclose(size,size[0],atol=1e-6)
    axes=3 if sphere else 2
    radius=size[0]/2
    a=np.sum(rays[...,:axes]**2,axis=-1)
    b=np.sum(rays[...,:axes]*q[:axes],axis=-1)
    c=np.sum(q[:axes]**2)-radius**2
    disc=b*b-a*c
    safe=np.maximum(a,1e-20)
    near=(-b-np.sqrt(np.maximum(disc,0)))/safe
    far=(-b+np.sqrt(np.maximum(disc,0)))/safe
    hit=(disc>=0)&(a>1e-12)
    if not sphere:
        parallel=a<=1e-12
        near=np.where(parallel,-np.inf,near); far=np.where(parallel,np.inf,far)
        hit=(hit| (parallel&(c<=0)))
        dz=rays[...,2]
        znear=np.full(dz.shape,-np.inf); zfar=np.full(dz.shape,np.inf)
        nonparallel=np.abs(dz)>1e-12
        lo=np.zeros_like(dz); hi=np.zeros_like(dz)
        np.divide(-size[2]/2-q[2],dz,out=lo,where=nonparallel)
        np.divide(size[2]/2-q[2],dz,out=hi,where=nonparallel)
        znear[nonparallel]=np.minimum(lo,hi)[nonparallel]
        zfar[nonparallel]=np.maximum(lo,hi)[nonparallel]
        hit &= nonparallel | (abs(q[2])<=size[2]/2)
        near=np.maximum(near,znear);far=np.minimum(far,zfar)
    hit &= (near>0)&np.isfinite(near)&(far>=near)
    return np.where(hit,near,0),hit


def estimate_scale(depth,masks,geometry,intrinsic):
    if depth.shape!=masks.shape or depth.shape[0]!=8:
        raise ValueError('Expected eight matched depth/mask frames for one object')
    if geometry['positions_world'].shape!=(8,1,3) or geometry['size_m'].shape!=(1,3):
        raise ValueError('One observed sphere or upright cylinder is supported')
    if not np.isfinite(depth).all() or np.any(depth<=0):
        raise ValueError('Invalid predicted depth')
    origin,rays=world_rays(intrinsic,geometry['camera_world_to_view'],depth.shape[1:])
    ratios=[];frames=[]
    for t in range(8):
        interior=morph3(morph3(masks[t],dilate=False),dilate=False)
        near,hit=ray_surface_depth(origin,rays,geometry['positions_world'][t,0],geometry['size_m'][0])
        chosen=interior&hit
        fraction=float(chosen.sum()/max(1,interior.sum()))
        if chosen.sum()<8 or fraction<.85:
            raise ValueError(f'RGB{t}: insufficient surface anchors ({chosen.sum()}, {fraction})')
        ratio=near[chosen]/depth[t][chosen]
        ratios.append(ratio)
        frames.append(dict(frame=t,points=int(chosen.sum()),hit_fraction=fraction,
                           median_scale=float(np.median(ratio))))
    scales=np.array([r['median_scale'] for r in frames])
    scale=float(np.median(scales))
    spread=float(np.ptp(scales)/scale)
    if not np.isfinite(scale) or scale<=0 or spread>.10:
        raise ValueError('Unstable observed-surface scale')
    return dict(scale_midpoint=scale,accepted=True,per_frame=frames,
        frame_scale_relative_span=spread,method='median of frame median front-surface depth ratios',
        shape_scope='sphere or already-validated upright circular cylinder, inferred from observed size',
        static_scene_gt_used=False,future_state_used=False,
        scale_interval_interpretation='point estimate duplicated for cache schema; not a confidence interval')
