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
    masks = as_masks(
        extract_masks(pred_video, gt_mask, sam2_model, active_actor_indices, frames=frames),
        "pred_mask",
    )
    return score_from_frames_and_masks(frames, masks, cotracker_model, device)


def score_from_frames_and_masks(
    pred_frames: np.ndarray,
    pred_masks: np.ndarray,
    cotracker_model,
    device: str = "cuda",
) -> dict:
    """Compute BG-Drift from already extracted prediction frames and masks.

    The numerical path is identical to :func:`score_case`; this helper lets a
    grouped evaluator reuse one SAM2 propagation for the mask metrics and
    BG-Drift without decoding or propagating the same case again.
    """
    frames = as_frames(pred_frames, "pred_frames")
    masks = as_masks(pred_masks, "pred_mask")
    if masks.shape[0] > 1:
        fg = masks[0].any(axis=0)
    else:
        fg = masks[0].any(axis=0)
    corners = detect_bg_corners(frames[0], fg)
    if corners is None:
        return {"bgdrift": float("nan")}
    tracks, confidence = track_points_cotracker3(frames, corners, cotracker_model, device)
    return {"bgdrift": compute_bgdrift(tracks, confidence, frames.shape[1])}
