"""Versioned missing-mask penalties; does not change official metric registration.

Input masks have shape (T, N, H, W), with persistent actor identity along N.
Only completed, valid observations may be scored. An empty mask means a real
empty extraction, never a missing file, skipped frame, or worker exception.
A caller must validate those upstream conditions; ``observed`` can enforce an
explicit per-actor-frame completion map. Nonempty masks are NOT certified to
track the right object by this metric.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

POLICY_VERSION = 'mask_distance_full_v1'


def _binary_masks(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 4 or any(dim == 0 for dim in arr.shape):
        raise ValueError(f'{name} must be nonempty (T,N,H,W), got {arr.shape}')
    if arr.dtype == np.bool_:
        return arr
    if not (np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)):
        raise TypeError(f'{name} must be boolean or numeric binary masks')
    if not np.isfinite(arr).all() or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f'{name} must contain only finite 0/1 values; threshold logits upstream')
    return arr.astype(bool)


def score_masks(gt: np.ndarray, pred: np.ndarray, *, observed: np.ndarray | None = None) -> dict:
    """Return per-actor-frame arrays without dropping empty masks.

    Spatial units match the official implementation: distance divided by H;
    Chamfer is the SUM of its two directed means. A unilateral empty mask gets
    the spatial upper bound, D/H or 2D/H. Both empty gets zero distance.
    When GT is nonempty, deleting a prediction can never improve these scores.
    Legacy arrays retain NaNs so their original conditional means can be audited.
    """
    gt = _binary_masks(gt, 'gt')
    pred = _binary_masks(pred, 'pred')
    if gt.shape != pred.shape:
        raise ValueError(f'Mask shapes differ: {gt.shape} vs {pred.shape}')
    t_count, n_count, height, width = gt.shape
    if observed is not None:
        observed = np.asarray(observed)
        if observed.dtype != np.bool_ or observed.shape != (t_count, n_count):
            raise ValueError('observed must be boolean (T,N)')
        if not observed.all():
            raise ValueError('Incomplete observations: repair extraction; do not score as empty masks')
    diagonal = float(np.hypot(height - 1, width - 1))
    l2_limit = diagonal / height
    chamfer_limit = 2 * l2_limit
    shape = (t_count, n_count)
    gt_present = gt.any(axis=(-2, -1))
    pred_present = pred.any(axis=(-2, -1))
    unilateral = gt_present ^ pred_present
    l2_full = np.where(unilateral, l2_limit, 0.0)
    chamfer_full = np.where(unilateral, chamfer_limit, 0.0)
    l2_legacy = np.full(shape, np.nan)
    chamfer_legacy = np.full(shape, np.nan)
    intersection = (gt & pred).sum(axis=(-2, -1))
    union = (gt | pred).sum(axis=(-2, -1))
    iou = np.ones(shape, dtype=float)
    np.divide(intersection, union, out=iou, where=union != 0)
    for t, n in np.argwhere(gt_present & pred_present):
        # Float64 centroid arithmetic and float32 point clouds match upstream.
        p1 = np.column_stack(np.where(gt[t, n]))
        p2 = np.column_stack(np.where(pred[t, n]))
        l2 = float(np.linalg.norm(p1.mean(axis=0) - p2.mean(axis=0)) / height)
        p1 = p1.astype(np.float32)
        p2 = p2.astype(np.float32)
        d1, _ = cKDTree(p1).query(p2)
        d2, _ = cKDTree(p2).query(p1)
        chamfer = float((d1.mean() + d2.mean()) / height)
        l2_full[t, n] = l2_legacy[t, n] = l2
        chamfer_full[t, n] = chamfer_legacy[t, n] = chamfer
    return {
        'policy_version': POLICY_VERSION,
        'height': height, 'width': width,
        'l2_penalty': l2_limit, 'chamfer_penalty': chamfer_limit,
        'gt_present': gt_present, 'pred_present': pred_present,
        'l2_full': l2_full, 'chamfer_full': chamfer_full,
        'l2_legacy': l2_legacy, 'chamfer_legacy': chamfer_legacy,
        'iou': iou,
    }


def _legacy_mean(values: np.ndarray) -> float | None:
    """Match upstream nanmean over actors, then nanmean over frames, without warnings."""
    valid = np.isfinite(values)
    counts = valid.sum(axis=1)
    frame_means = np.divide(
        np.where(valid, values, 0).sum(axis=1), counts,
        out=np.full(values.shape[0], np.nan), where=counts > 0,
    )
    surviving = frame_means[np.isfinite(frame_means)]
    return float(surviving.mean()) if len(surviving) else None


def summarize(scores: dict, start: int = 0, stop: int | None = None) -> dict:
    """Summarize a fixed window [start, stop); full means give each actor-frame equal weight."""
    frame_count = scores['gt_present'].shape[0]
    stop = frame_count if stop is None else stop
    if not (isinstance(start, int) and isinstance(stop, int) and 0 <= start < stop <= frame_count):
        raise ValueError(f'Invalid frame window [{start}, {stop}) for {frame_count} frames')
    gt = scores['gt_present'][start:stop]
    pred = scores['pred_present'][start:stop]
    gt_count = int(gt.sum())
    missing = int((gt & ~pred).sum())
    result = {
        'policy_version': POLICY_VERSION, 'start': start, 'stop': stop,
        'frames': stop - start, 'actors': gt.shape[1], 'actor_frames': gt.size,
        'gt_present_actor_frames': gt_count,
        'paired_nonempty_actor_frames': int((gt & pred).sum()),
        'missing_actor_frames': missing,
        'false_positive_actor_frames': int((~gt & pred).sum()),
        'both_empty_actor_frames': int((~gt & ~pred).sum()),
        'mask_missing_rate': missing / gt_count if gt_count else None,
        'l2_penalty': scores['l2_penalty'], 'chamfer_penalty': scores['chamfer_penalty'],
    }
    for key in ('l2_full', 'chamfer_full', 'iou'):
        result[key] = float(scores[key][start:stop].mean())
    for key in ('l2_legacy', 'chamfer_legacy'):
        result[key] = _legacy_mean(scores[key][start:stop])
    return result
