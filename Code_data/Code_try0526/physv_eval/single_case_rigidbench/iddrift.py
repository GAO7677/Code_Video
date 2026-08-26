from __future__ import annotations

import numpy as np

from rigidbench.eval.score.identity import compute_iddrift

from .common import as_frames, as_tracks, as_visibility, load_video_rgb
from .prediction import extract_tracks


def score_case(
    gt_frames,
    pred_video,
    gt_tracks,
    visibility,
    actor_offsets,
    dinov2_model,
    cotracker_model,
    device: str = "cuda",
) -> dict:
    """Return DINO identity drift from GT inputs and a generated video.

    CoTracker extracts prediction tracks internally from the generated video;
    DINOv2 is supplied by the metric worker and reused across cases.
    """
    gt_f = as_frames(gt_frames, "gt_frames")
    gt_t = as_tracks(gt_tracks, "gt_tracks")
    pred_f = as_frames(load_video_rgb(pred_video), "pred_frames")
    pred_tracks, pred_visibility = extract_tracks(pred_video, gt_t, cotracker_model)
    pred_t = as_tracks(pred_tracks, "pred_tracks")
    T = min(len(gt_f), len(pred_f), gt_t.shape[1], pred_t.shape[1])
    gt_f, pred_f = gt_f[:T], pred_f[:T]
    gt_t, pred_t = gt_t[:, :T], pred_t[:, :T]
    vis = as_visibility(np.asarray(visibility)[:, :T], (gt_t.shape[0], T))
    vis = vis & pred_visibility[:, :T]
    offsets = np.asarray(actor_offsets, dtype="int64")
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != gt_t.shape[0]:
        raise ValueError("actor_offsets must be [0,...,N] for the N track points")
    return compute_iddrift(gt_f, pred_f, gt_t, pred_t, vis, offsets, dinov2_model, device)
