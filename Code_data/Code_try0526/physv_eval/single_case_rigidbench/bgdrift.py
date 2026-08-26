from __future__ import annotations

from rigidbench.eval.score.background import compute_bgdrift, detect_bg_corners, track_points_cotracker3

from .common import as_frames, as_masks


def score_case(pred_frames, foreground_mask, cotracker_model, device: str = "cuda") -> dict:
    """Return BG-Drift from RGB frames and a foreground mask.

    Frames are uint8 RGB (T,H,W,3). ``foreground_mask`` is bool (H,W), or a
    bool mask batch (T,1,H,W)/(T,H,W); only frame 0 is used to select the
    static background, matching the current RigidBench pipeline.
    """
    frames = as_frames(pred_frames, "pred_frames")
    masks = as_masks(foreground_mask, "foreground_mask")
    if masks.shape[0] > 1:
        fg = masks[0].any(axis=0)
    else:
        fg = masks[0].any(axis=0)
    corners = detect_bg_corners(frames[0], fg)
    if corners is None:
        return {"bgdrift": float("nan")}
    tracks, confidence = track_points_cotracker3(frames, corners, cotracker_model, device)
    return {"bgdrift": compute_bgdrift(tracks, confidence, frames.shape[1])}
