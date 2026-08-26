from __future__ import annotations

import argparse

from rigidbench.eval.score.track import ate_per_frame, compute_ate_scalar

from .common import as_tracks, as_visibility, cli_print, load_npz_array


def score_case(gt_tracks, pred_tracks, image_height: int, visibility=None) -> dict:
    """Return ATE/ATE-std from pixel tracks shaped (N,T,2), normalized by H."""
    gt = as_tracks(gt_tracks, "gt_tracks")
    pred = as_tracks(pred_tracks, "pred_tracks")
    if gt.shape != pred.shape:
        raise ValueError(f"gt_tracks and pred_tracks must have the same shape, got {gt.shape} vs {pred.shape}")
    if image_height <= 0:
        raise ValueError("image_height must be positive")
    vis = as_visibility(visibility, gt.shape[:2])
    result = compute_ate_scalar(gt, pred, int(image_height), vis)
    result["per_frame"] = ate_per_frame(gt, pred, int(image_height), vis)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one case's normalized 2D ATE")
    parser.add_argument("--gt-tracks", required=True)
    parser.add_argument("--pred-tracks", required=True)
    parser.add_argument("--image-height", required=True, type=int)
    args = parser.parse_args()
    gt = load_npz_array(args.gt_tracks, "tracks")
    pred = load_npz_array(args.pred_tracks, "tracks")
    cli_print(score_case(gt, pred, args.image_height))
