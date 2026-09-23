"""Explicit local planar prior for target-occluded holes; no scene metadata."""
import cv2
import numpy as np


def complete_occluded_depth(depth, target_region, intrinsic):
    """Only fill unknown pixels enclosed by well observed, locally coplanar data.

    Works in reconstruction units; thresholds are relative to observed depth.
    A surface inferred here is never labelled observed or guaranteed support.
    """
    z=np.asarray(depth,dtype=float).copy();h,w=z.shape
    valid=np.isfinite(z)&(z>0);candidate=~valid&target_region.astype(bool)
    count,labels,stats,centroids=cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
    filled=np.zeros((h,w),bool);audit=[]
    yy,xx=np.mgrid[:h,:w];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(intrinsic).T
    for label in range(1,count):
        region=labels==label;area=int(region.sum());x,y,bw,bh,_=stats[label]
        row={'component':label,'pixels':area,'status':'UNKNOWN','reason':None}
        audit.append(row)
        width=max(3,int(np.ceil(np.sqrt(area)*.3)))
        ring=(cv2.dilate(region.astype(np.uint8),np.ones((2*width+1,2*width+1),np.uint8))>0)&~region
        if x==0 or y==0 or x+bw>=w or y+bh>=h:
            row['reason']='image_boundary';continue
        coverage=float(valid[ring].mean()) if ring.any() else 0.
        row['ring_coverage']=coverage
        if coverage<.9 or (ring&valid).sum()<40:
            row['reason']='insufficient_surrounding_observation';continue
        support=ring&valid
        angles=np.arctan2(yy[support]-centroids[label,1],xx[support]-centroids[label,0])
        sectors=np.minimum(7,((angles+np.pi)/(2*np.pi)*8).astype(int))
        if np.any(np.bincount(sectors,minlength=8)<3):
            row['reason']='one_sided_support';continue
        points=rays[support]*z[support,None]
        center=np.median(points,axis=0);_,_,v=np.linalg.svd(points-center,full_matrices=False);normal=v[-1]
        offset=-float(normal@np.mean(points,axis=0))
        errors=np.abs(points@normal+offset);depth_scale=float(np.median(z[support]))
        rms=float(np.sqrt(np.mean(errors**2)))
        row.update(relative_plane_rms=rms/depth_scale,relative_p95=float(np.percentile(errors,95)/depth_scale),
                   normal_camera=normal.tolist(),offset_reconstruction_units=offset,support_pixels=int(support.sum()))
        if rms/depth_scale>.005 or np.percentile(errors,95)/depth_scale>.01:
            row['reason']='nonplanar_or_visible_edge';continue
        denom=rays[region]@normal
        if np.any(np.abs(denom)<1e-6):
            row['reason']='grazing_plane';continue
        inferred=-offset/denom
        if np.any(~np.isfinite(inferred)) or np.any(inferred<=0):
            row['reason']='invalid_intersection';continue
        # Reject extrapolation outside local depth extent; never extend an infinite plane.
        lo,hi=np.min(z[support]),np.max(z[support]);margin=.01*depth_scale
        if np.any(inferred<lo-margin) or np.any(inferred>hi+margin):
            row['reason']='unsupported_depth_extrapolation';continue
        z[region]=inferred;filled[region]=True
        row.update(status='INFERRED',reason='explicit_local_planar_prior')
    return z,filled,{'mode':'local_plane_completion','is_observation':False,
        'limits':{'ring_coverage_min':.9,'relative_rms_max':.005,'relative_p95_max':.01,
                  'required_angular_sectors':8,'modify_observed_pixels':False},
        'candidate_pixels':int(candidate.sum()),'inferred_pixels':int(filled.sum()),'components':audit}
