"""Trusted local finite-plane completion for target-occluded depth."""
import cv2
import numpy as np


def complete_trusted_planes(depth, target_region, intrinsic):
    z = np.asarray(depth, float).copy()
    valid = np.isfinite(z) & (z > 0)
    target = target_region.astype(bool)
    candidate = ~valid & target
    h, w = z.shape
    yy, xx = np.indices(z.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(intrinsic).T
    xyz = rays * np.where(valid, z, 0)[..., None]
    dx = np.zeros_like(xyz); dy = np.zeros_like(xyz)
    dx[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dy[1:-1] = xyz[2:] - xyz[:-2]
    normals = np.cross(dx, dy)
    nrm = np.linalg.norm(normals, axis=-1)
    normals /= np.maximum(nrm[..., None], 1e-12)
    # Trusted support is local, smooth, and away from depth discontinuities.
    zgrad = np.zeros_like(z)
    zgrad[:, 1:-1] = np.maximum(zgrad[:, 1:-1], np.abs(z[:, 2:] - z[:, :-2]))
    zgrad[1:-1] = np.maximum(zgrad[1:-1], np.abs(z[2:] - z[:-2]))
    trusted = valid & (nrm > 1e-8)
    trusted &= zgrad <= .012 * np.maximum(z, np.nanmedian(z[valid]))
    outputs = np.zeros_like(z); owners = np.zeros_like(z, np.int8)
    rows = []
    rng = np.random.default_rng(42)
    count, labels, stats, cents = cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
    for hole_id in range(1, count):
        hole = labels == hole_id
        area = int(stats[hole_id, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        radius = max(4, int(np.ceil(np.sqrt(area) * .35)))
        ring = (cv2.dilate(hole.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0) & ~hole
        support = ring & trusted
        row_base = {'hole_component': hole_id, 'hole_pixels': area,
                    'support_pixels': int(support.sum()), 'status': 'UNKNOWN'}
        if support.sum() < 40:
            row_base['reason'] = 'insufficient_trusted_support'; rows.append(row_base); continue
        # Fit only the local ring. This prevents a distant coplanar floor/wall
        # from producing a hull that crosses an intervening edge.
        sy, sx = np.where(support)
        points = xyz[sy, sx]; pnorm = normals[sy, sx]
        scale = float(np.median(z[support])); tol = .006 * scale
        remaining = np.ones(len(points), bool)
        found = 0
        for _ in range(6):
            ids = np.where(remaining)[0]
            if len(ids) < 40: break
            sample = ids[rng.choice(len(ids), min(1200, len(ids)), replace=False)]
            best = None; best_score = 0
            for _ in range(180):
                a,b,c = points[rng.choice(sample, 3, replace=False)]
                n = np.cross(b-a, c-a); ln = np.linalg.norm(n)
                if ln < 1e-10: continue
                n /= ln; d = -n @ a
                score = ((np.abs(points[sample] @ n + d) <= tol) &
                         (np.abs(pnorm[sample] @ n) >= np.cos(np.deg2rad(18)))).sum()
                if score > best_score: best_score, best = int(score), (n,d)
            if best is None or best_score < 30: break
            n,d = best
            inlier = remaining & (np.abs(points@n+d) <= tol) & (np.abs(pnorm@n) >= np.cos(np.deg2rad(18)))
            if inlier.sum() < 40: break
            center = points[inlier].mean(0); _,_,vh = np.linalg.svd(points[inlier]-center, full_matrices=False)
            n = vh[-1]; d = -n@center
            inlier = remaining & (np.abs(points@n+d) <= tol) & (np.abs(pnorm@n) >= np.cos(np.deg2rad(18)))
            iy, ix = sy[inlier], sx[inlier]
            # Fit only connected local support, preserving true edges.
            bitmap = np.zeros((h,w), np.uint8); bitmap[iy,ix] = 1
            nc, comps, cstats, _ = cv2.connectedComponentsWithStats(bitmap)
            for comp in range(1,nc):
                if cstats[comp,cv2.CC_STAT_AREA] < 40: continue
                patch = comps == comp
                py,px = np.where(patch)
                center2 = np.array([px.mean(),py.mean()])
                hull = cv2.convexHull(np.c_[px,py].astype(np.int32))
                envelope = np.zeros((h,w), np.uint8); cv2.fillConvexPoly(envelope,hull,1)
                region = hole & (envelope > 0)
                if not region.any(): continue
                denom = rays[region] @ n
                values = np.divide(-d, denom, out=np.full_like(denom,np.nan), where=np.abs(denom)>1e-6)
                good = np.isfinite(values) & (values>0) & (values>=z[patch].min()-.01*scale) & (values<=z[patch].max()+.01*scale)
                ry,rx=np.where(region); ry,rx,values=ry[good],rx[good],values[good]
                outputs[ry,rx]=values; owners[ry,rx]+=1; found += len(ry)
                row_base.setdefault('patches',[]).append({'support_pixels':int(patch.sum()),'proposed_pixels':int(len(ry)),
                    'normal_camera':n.tolist(),'plane_rms_m':float(np.sqrt(np.mean((points[inlier]@n+d)**2))*1.0)})
            remaining[inlier] = False
        row_base.update(status='PROPOSED' if found else 'UNKNOWN', reason='local_trusted_surface' if found else 'no_local_surface',
                        proposed_pixels=int(found)); rows.append(row_base)
    filled = candidate & (owners == 1)
    z[filled] = outputs[filled]
    return z, filled, {'mode':'trusted_local_finite_plane_v1','is_observation':False,
        'candidate_pixels':int(candidate.sum()),'inferred_pixels':int(filled.sum()),
        'ambiguous_pixels':int((owners>1).sum()),'components':rows,
        'limits':{'modify_observed_pixels':False,'fill_outside_target':False,'local_ring_only':True,
                  'depth_gradient_rel':.012,'normal_angle_deg':18,'ransac_trials':180,'seed':42},
        'prior':'multi-surface local continuation from trusted smooth points'}
