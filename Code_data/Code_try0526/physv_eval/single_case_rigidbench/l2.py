from __future__ import annotations

import argparse

from rigidbench.eval.score.mask import l2_per_frame

from .common import cli_print, load_npz_array
from .mask_metric_common import score_mask_metric
from .prediction import extract_masks, load_sam2_model


def score_case(gt_mask, pred_video, sam2_model, active_actor_indices=None) -> dict:
    """Return centroid L2/H from GT masks and a generated video."""
    pred_mask = extract_masks(pred_video, gt_mask, sam2_model, active_actor_indices)
    if active_actor_indices is not None:
        gt_mask = gt_mask[:, active_actor_indices]
    T = min(len(gt_mask), len(pred_mask))
    gt_mask, pred_mask = gt_mask[:T], pred_mask[:T]
    out = score_mask_metric(gt_mask, pred_mask, l2_per_frame)
    return {"l2": out["value"], "per_frame": out["per_frame"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one case's normalized mask centroid L2")
    parser.add_argument("--gt-mask", required=True)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = load_sam2_model(args.device)
    cli_print(score_case(load_npz_array(args.gt_mask, "masks", "mask"), args.pred_video, model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
