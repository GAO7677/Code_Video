from __future__ import annotations

import argparse

from rigidbench.eval.score.depth import compute_si_mse

from .common import as_depth, cli_print, load_npz_array
from .prediction import extract_disparity, load_vda_model


def score_case(gt_depth, pred_video, vda_model, device: str = "cuda") -> dict:
    """Return SI-MSE from GT depth and a generated video.

    VDA predicts the positive inverse-depth/disparity internally; it is
    affine-aligned to GT depth by the official implementation.
    """
    gt = as_depth(gt_depth, "gt_depth")
    pred_disparity = extract_disparity(pred_video, vda_model, device)
    pred = as_depth(pred_disparity, "pred_disparity")
    T = min(gt.shape[0], pred.shape[0])
    gt, pred = gt[:T], pred[:T]
    return {"si_mse": compute_si_mse(gt, pred)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one case's scale-invariant depth MSE")
    parser.add_argument("--gt-depth", required=True)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = load_vda_model(args.device)
    cli_print(score_case(load_npz_array(args.gt_depth, "depth"), args.pred_video, model, args.device))
