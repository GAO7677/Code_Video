"""Complete occlusion per observed connected surface, never per target hole."""
import cv2
import numpy as np


def complete_surface_depth(depth, target_region, intrinsic):
    z = np.asarray(depth, float).copy()
    valid = np.isfinite(z) & (z > 0)
    candidate = ~valid & target_region.astype(bool)
    yy, xx = np.indices(z.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(intrinsic).T
    xyz = rays * z[..., None]
    # Surface normals reject accidental plane intersections across depth edges.
    dx = np.zeros_like(xyz); dy = np.zeros_like(xyz)
    dx[:,1:-1] = xyz[:,2:] - xyz[:,:-2]
    dy[1:-1] = xyz[2:] - xyz[:-2]
    normal = np.cross(dx, dy)
    length = np.linalg.norm(normal, axis=-1)
    normal /= np.maximum(length[...,None], 1e-12)
    interior = cv2.erode(valid.astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    available = interior & (length > 1e-10)
    owners = np.zeros(z.shape, np.int32)
    values = np.zeros(z.shape)
    rows = []
    rng = np.random.default_rng(42)
    cosine = np.cos(np.deg2rad(20))
    for iteration in range(16):
        y, x = np.where(available)
        if len(y) < 80:
            break
        ids = rng.choice(len(y), min(3000, len(y)), replace=False)
        points = xyz[y[ids], x[ids]]
        ns = normal[y[ids], x[ids]]
        tol = .005 * float(np.median(z[available]))
        best = None; count = 0
        for _ in range(160):
            a,b,c = points[rng.choice(len(points),3,replace=False)]
            n = np.cross(b-a,c-a); size = np.linalg.norm(n)
            if size < 1e-10:
                continue
            n /= size; offset = -n@a
            score = int(((abs(points@n+offset) <= tol) & (abs(ns@n) >= cosine)).sum())
            if score > count:
                count, best = score, (n, offset)
        if best is None or count < 20:
            break
        n, offset = best
        compatible = available & (abs(xyz@n+offset) <= tol) & (abs(normal@n) >= cosine)
        # Each connected patch owns its own finite silhouette, not a union of
        # disconnected but coplanar platforms.
        nc, labels, stats, _ = cv2.connectedComponentsWithStats(compatible.astype(np.uint8))
        available[compatible] = False
        for label in range(1,nc):
            if stats[label,cv2.CC_STAT_AREA] < 80:
                continue
            patch = labels == label
            pts = xyz[patch]
            center = pts.mean(axis=0)
            _,_,vh = np.linalg.svd(pts-center,full_matrices=False)
            pn = vh[-1]; po = -pn@center
            scale = float(np.median(z[patch]))
            residual = abs(pts@pn+po)
            row = {'surface':len(rows)+1, 'status':'OBSERVED', 'reason':'no_bounded_occlusion',
                   'support_pixels':int(patch.sum()), 'normal_camera':pn.tolist(),
                   'offset_reconstruction_units':float(po),
                   'relative_plane_rms':float(np.sqrt(np.mean(residual**2))/scale),
                   'relative_p95':float(np.percentile(residual,95)/scale)}
            rows.append(row)
            if row['relative_plane_rms'] > .005 or row['relative_p95'] > .01:
                row.update(status='UNKNOWN',reason='nonplanar_patch')
                continue
            py,px = np.where(patch)
            hull = cv2.convexHull(np.c_[px,py].astype(np.int32))
            bounded = np.zeros(z.shape,np.uint8)
            cv2.fillConvexPoly(bounded,hull,1)
            # Hull is only an explicit occlusion-continuation envelope. Never
            # replace observed pixels or fill unrelated visible missing regions.
            region = candidate & (bounded > 0)
            row['boundary_hull_pixels'] = hull[:,0].tolist()
            row['source_pixels_xy'] = np.c_[px,py].tolist()
            if not region.any():
                continue
            denom = rays[region]@pn
            inferred = np.divide(-po,denom,out=np.full_like(denom,np.nan),where=abs(denom)>1e-6)
            good = np.isfinite(inferred) & (inferred>0) & (inferred>=z[patch].min()-.01*scale) & (inferred<=z[patch].max()+.01*scale)
            ry,rx = np.where(region); ry,rx,inferred = ry[good],rx[good],inferred[good]
            values[ry,rx] = inferred
            owners[ry,rx] += 1
            row.update(status='PROPOSED',reason='bounded_surface_occlusion_prior',proposed_pixels=len(ry))
    filled = candidate & (owners == 1)
    z[filled] = values[filled]
    return z, filled, {'mode':'finite_surface_first_completion','is_observation':False,
        'candidate_pixels':int(candidate.sum()),'inferred_pixels':int(filled.sum()),
        'ambiguous_pixels':int((owners>1).sum()),'components':rows,
        'limits':{'modify_observed_pixels':False,'fill_outside_target':False,
                  'min_connected_points':80,'normal_angle_deg':20,'relative_rms_max':.005,
                  'relative_p95_max':.01,'ransac_trials':160,'max_plane_extractions':16,'seed':42},
        'prior':'convex continuation inside one connected observed surface hull; hidden concavities remain uncertain'}
