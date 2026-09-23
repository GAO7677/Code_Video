"""Local observed-plane segmentation for bounded target-occlusion completion.

No state, scene labels, physical size or GT inputs. No observed vertex is moved.
This is an explicit planar continuation prior, not an observation of hidden space.
"""
import cv2
import numpy as np


def complete_segmented_depth(depth, target_region, intrinsic):
    z = np.asarray(depth, dtype=float).copy()
    valid = np.isfinite(z) & (z > 0)
    target = target_region.astype(bool)
    candidate = ~valid & target
    count, labels, stats, centers = cv2.connectedComponentsWithStats(candidate.astype(np.uint8))
    yy, xx = np.indices(z.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(intrinsic).T
    inferred_mask = np.zeros(z.shape, bool)
    rows = []
    rng = np.random.default_rng(42)
    for label in range(1, count):
        region = labels == label
        x, y, width, height, area = stats[label]
        row = {'component': label, 'pixels': int(area), 'status': 'UNKNOWN'}
        rows.append(row)
        if x == 0 or y == 0 or x + width >= z.shape[1] or y + height >= z.shape[0]:
            row['reason'] = 'image_boundary'
            continue
        radius = max(3, int(np.ceil(np.sqrt(area) * .3)))
        ring = (cv2.dilate(region.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8)) > 0) & ~region
        # Fusion already removes the dynamic mask in each source frame. Pixels
        # occluded in RGB0 may be observed static surfaces in later frames.
        support = ring & valid
        coverage = support.sum() / max(1, ring.sum())
        row['ring_coverage'] = float(coverage)
        if coverage < .9 or support.sum() < 40:
            row['reason'] = 'insufficient_surrounding_observation'
            continue
        points = rays[support] * z[support, None]
        depth_scale = float(np.median(z[support]))
        # Keep the old residual tolerance; segment competing surfaces instead
        # of fitting one plane to all depths in the surrounding ring.
        tolerance = .005 * depth_scale
        subset = points[rng.choice(len(points), min(1000, len(points)), replace=False)]
        best = None
        best_count = 0
        for _ in range(160):
            a, b, c = subset[rng.choice(len(subset), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            normal /= length
            offset = -normal @ a
            n = int((abs(subset @ normal + offset) <= tolerance).sum())
            if n > best_count:
                best_count, best = n, (normal, offset)
        if best is None:
            row['reason'] = 'no_plane_candidate'
            continue
        normal, offset = best
        inliers = abs(points @ normal + offset) <= tolerance
        for _ in range(3):
            if inliers.sum() < 40:
                break
            center = points[inliers].mean(axis=0)
            _, _, vh = np.linalg.svd(points[inliers]-center, full_matrices=False)
            normal = vh[-1]
            offset = -normal @ center
            inliers = abs(points @ normal + offset) <= tolerance
        row['plane_inlier_fraction'] = float(inliers.mean())
        if inliers.mean() < .65 or inliers.sum() < 40:
            row['reason'] = 'no_dominant_local_plane'
            continue
        selected = np.zeros_like(valid)
        selected[support] = inliers
        # Disconnected coplanar surfaces must not be joined across an edge.
        ncomp, components, sizes, _ = cv2.connectedComponentsWithStats(selected.astype(np.uint8))
        largest = 1 + int(np.argmax(sizes[1:, cv2.CC_STAT_AREA])) if ncomp > 1 else 0
        selected = components == largest if largest else np.zeros_like(valid)
        angle = np.arctan2(yy[support]-centers[label, 1], xx[support]-centers[label, 0])
        sectors = np.minimum(7, ((angle+np.pi)/(2*np.pi)*8).astype(int))
        totals = np.bincount(sectors, minlength=8)
        hits = np.bincount(sectors[selected[support]], minlength=8)
        fractions = hits / np.maximum(totals, 1)
        row['plane_sector_fractions'] = fractions.tolist()
        if np.any(hits < 3) or np.any(fractions < .6):
            row['reason'] = 'plane_not_surrounding_hole'
            continue
        border = (cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~region
        # Missing neighboring depths are unknown, not observed conflicting
        # surfaces. Coverage is checked separately on the surrounding ring.
        observed_border = border & valid
        row['boundary_observed_fraction'] = float(valid[border].mean())
        row['boundary_plane_fraction'] = float(selected[observed_border].mean()) if observed_border.any() else 0.
        if not observed_border.any() or row['boundary_plane_fraction'] < .8:
            row['reason'] = 'visible_boundary_or_competing_surface'
            continue
        pts = rays[selected] * z[selected, None]
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts-center, full_matrices=False)
        normal = vh[-1]
        offset = -normal @ center
        error = abs(pts @ normal + offset)
        row.update(relative_plane_rms=float(np.sqrt(np.mean(error**2))/depth_scale),
                   relative_p95=float(np.percentile(error, 95)/depth_scale),
                   normal_camera=normal.tolist(), offset_reconstruction_units=float(offset),
                   support_pixels=int(selected.sum()))
        if row['relative_plane_rms'] > .005 or row['relative_p95'] > .01:
            row['reason'] = 'nonplanar_selected_surface'
            continue
        denominator = rays[region] @ normal
        if np.any(abs(denominator) < 1e-6):
            row['reason'] = 'grazing_plane'
            continue
        values = -offset / denominator
        lo, hi = np.min(z[selected]), np.max(z[selected])
        if np.any(~np.isfinite(values)) or np.any(values <= 0) or np.any(values < lo-.01*depth_scale) or np.any(values > hi+.01*depth_scale):
            row['reason'] = 'unsupported_depth_extrapolation'
            continue
        z[region] = values
        inferred_mask[region] = True
        row.update(status='INFERRED', reason='segmented_surrounding_plane_prior')
    return z, inferred_mask, {'mode': 'segmented_local_plane_completion', 'is_observation': False,
        'candidate_pixels': int(candidate.sum()), 'inferred_pixels': int(inferred_mask.sum()),
        'limits': {'ring_coverage_min': .9, 'plane_fraction_min': .65, 'sector_fraction_min': .6,
                   'boundary_fraction_min': .8, 'relative_rms_max': .005, 'relative_p95_max': .01,
                   'modify_observed_pixels': False, 'random_seed': 42, 'ransac_trials': 160},
        'components': rows}
