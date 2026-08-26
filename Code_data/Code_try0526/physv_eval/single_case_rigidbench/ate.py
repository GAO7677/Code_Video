from __future__ import annotations

import argparse
import numpy as np

from rigidbench.eval.score.track import ate_per_frame, compute_ate_scalar

from .common import as_tracks, as_visibility, cli_print, load_npz_array
from .prediction import extract_tracks, load_cotracker_model


def score_case(gt_tracks, pred_video, image_height: int, cotracker_model, visibility=None) -> dict:
    """Return ATE from GT tracks and a generated video.

    CoTracker is run internally using GT first-frame query points.
    """
    gt = as_tracks(gt_tracks, "gt_tracks")
    pred_tracks, pred_visibility = extract_tracks(pred_video, gt, cotracker_model)
    pred = as_tracks(pred_tracks, "pred_tracks")
    T = min(gt.shape[1], pred.shape[1])
    gt, pred = gt[:, :T], pred[:, :T]
    if image_height <= 0:
        raise ValueError("image_height must be positive")
    if visibility is None:
        vis = np.ones(gt.shape[:2], dtype=bool)
    else:
        visibility = np.asarray(visibility)[:, :T]
        vis = as_visibility(visibility, (gt.shape[0], T))
    vis = vis & pred_visibility[:, :T]
    result = compute_ate_scalar(gt, pred, int(image_height), vis)
    result["per_frame"] = ate_per_frame(gt, pred, int(image_height), vis)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one case's normalized 2D ATE")
    parser.add_argument("--gt-tracks", required=True)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--image-height", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    with np.load(args.gt_tracks, allow_pickle=False) as data:
        gt = data["tracks"]
        visibility = data["visibility"] if "visibility" in data.files else None
    model = load_cotracker_model(args.device)
    cli_print(score_case(gt, args.pred_video, args.image_height, model, visibility))
