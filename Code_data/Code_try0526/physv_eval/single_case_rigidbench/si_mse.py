from __future__ import annotations

import argparse

from rigidbench.eval.score.depth import compute_si_mse

from .common import as_depth, cli_print, load_npz_array


def score_case(gt_depth, pred_disparity) -> dict:
    """Return SI-MSE from GT depth and predicted disparity, both (T,H,W).

    GT depth is positive metric depth in meters. Prediction is positive
    inverse-depth/disparity; it is affine-aligned to GT depth per video by the
    official implementation, so callers must not pre-normalize it.
    """
    gt = as_depth(gt_depth, "gt_depth")
    pred = as_depth(pred_disparity, "pred_disparity")
    if gt.shape[0] != pred.shape[0]:
        raise ValueError(f"gt_depth and pred_disparity must have the same T, got {gt.shape} vs {pred.shape}")
    return {"si_mse": compute_si_mse(gt, pred)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one case's scale-invariant depth MSE")
    parser.add_argument("--gt-depth", required=True)
    parser.add_argument("--pred-disparity", required=True)
    args = parser.parse_args()
    cli_print(score_case(load_npz_array(args.gt_depth, "depth"), load_npz_array(args.pred_disparity, "depth")))
