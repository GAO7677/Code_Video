from __future__ import annotations

from rigidbench.eval.score.background import compute_bgdrift, detect_bg_corners, track_points_cotracker3

from .common import as_frames, as_masks, load_video_rgb
from .prediction import extract_masks


def score_case(
    pred_video,
    gt_mask,
    sam2_model,
    cotracker_model,
    active_actor_indices=None,
    device: str = "cuda",
) -> dict:
    """Return BG-Drift from GT mask supervision and a generated video.

    SAM2 obtains the generated-video foreground mask internally from the GT
    first-frame active masks.  CoTracker then tracks background corners.
    """
    frames = as_frames(load_video_rgb(pred_video), "pred_frames")
    masks = as_masks(extract_masks(pred_video, gt_mask, sam2_model, active_actor_indices), "pred_mask")
    if masks.shape[0] > 1:
        fg = masks[0].any(axis=0)
    else:
        fg = masks[0].any(axis=0)
    corners = detect_bg_corners(frames[0], fg)
    if corners is None:
        return {"bgdrift": float("nan")}
    tracks, confidence = track_points_cotracker3(frames, corners, cotracker_model, device)
    return {"bgdrift": compute_bgdrift(tracks, confidence, frames.shape[1])}
