from __future__ import annotations

import numpy as np

from rigidbench.eval.score.identity import compute_iddrift

from .common import as_frames, as_tracks, as_visibility


def score_case(
    gt_frames,
    pred_frames,
    gt_tracks,
    pred_tracks,
    visibility,
    actor_offsets,
    dinov2_model,
    device: str = "cuda",
) -> dict:
    """Return DINO identity drift from frames, tracks and visibility.

    Frames are uint8 RGB (T,H,W,3), tracks are pixel coordinates (N,T,2),
    visibility is bool (N,T), and actor_offsets is int64 (A+1,).
    """
    gt_f = as_frames(gt_frames, "gt_frames")
    pred_f = as_frames(pred_frames, "pred_frames")
    gt_t = as_tracks(gt_tracks, "gt_tracks")
    pred_t = as_tracks(pred_tracks, "pred_tracks")
    if gt_t.shape != pred_t.shape:
        raise ValueError("gt_tracks and pred_tracks must have the same shape")
    vis = as_visibility(visibility, gt_t.shape[:2])
    if vis is None:
        raise ValueError("visibility is required for iddrift")
    offsets = np.asarray(actor_offsets, dtype="int64")
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != gt_t.shape[0]:
        raise ValueError("actor_offsets must be [0,...,N] for the N track points")
    return compute_iddrift(gt_f, pred_f, gt_t, pred_t, vis, offsets, dinov2_model, device)
