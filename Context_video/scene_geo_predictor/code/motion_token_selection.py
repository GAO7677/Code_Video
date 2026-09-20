"""Observed-motion corridor selection mixed with global scene coverage."""
import numpy as np


def select_tokens(xyz, positions, size, times, budget=1792):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or positions.shape != (8, 1, 3) or size.shape != (1, 3) or times.shape != (8,):
        raise ValueError('Invalid observation/token shapes')
    if not all(np.isfinite(a).all() for a in (xyz, positions, size, times)) or not (size > 0).all() or not (np.diff(times) > 0).all():
        raise ValueError('Invalid observed geometry')
    if budget < 2 or not len(xyz):
        raise ValueError('Invalid token budget')
    n = min(budget, len(xyz))
    start = positions[-1, 0]
    velocity = (positions[-1, 0]-positions[-2, 0])/(times[-1]-times[-2])
    segment = velocity*(41/30)
    radius = float(size.max()/2+.20)
    fraction = np.clip((xyz-start)@segment/max(float(segment@segment), 1e-12), 0, 1)
    distance = np.linalg.norm(xyz-start-fraction[:, None]*segment, axis=1)
    candidates = np.flatnonzero(distance <= radius)
    count = min(n//2, len(candidates))
    local = []
    if count:
        coords = xyz[candidates]
        nearest = np.full(len(candidates), np.inf)
        available = np.ones(len(candidates), dtype=bool)
        current = int(np.argmin(np.linalg.norm(coords-start, axis=1)))
        # Farthest-point selection avoids spending the local budget on duplicate frames.
        for _ in range(count):
            local.append(int(candidates[current]))
            available[current] = False
            nearest = np.minimum(nearest, np.sum((coords-coords[current])**2, axis=1))
            current = int(np.argmax(np.where(available, nearest, -1)))
    remaining = np.setdiff1d(np.arange(len(xyz)), np.array(local, dtype=np.int64))
    global_ids = remaining[np.linspace(0, len(remaining)-1, n-count, dtype=np.int64)]
    selected = np.sort(np.r_[np.array(local, dtype=np.int64), global_ids])
    return selected, dict(method='half_global_half_observed_cv_corridor_fps_v1',
        local_tokens=count, global_tokens=n-count, local_candidates=len(candidates),
        radius_m=radius, horizon_s=41/30, observed_velocity_mps=velocity.tolist(),
        future_trajectory_used=False, static_gt_used=False,
        limitation='CV is a sampling prior, not a collision rollout; half the budget stays global')
