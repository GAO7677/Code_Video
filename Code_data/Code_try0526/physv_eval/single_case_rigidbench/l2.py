from __future__ import annotations

import argparse

from rigidbench.eval.score.mask import l2_per_frame

from .common import cli_print, load_npz_array
from .mask_metric_common import score_mask_metric


def score_case(gt_mask, pred_mask) -> dict:
    """Return centroid L2/H; masks are bool arrays shaped (T,N,H,W)."""
    out = score_mask_metric(gt_mask, pred_mask, l2_per_frame)
    return {"l2": out["value"], "per_frame": out["per_frame"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one case's normalized mask centroid L2")
    parser.add_argument("--gt-mask", required=True)
    parser.add_argument("--pred-mask", required=True)
    args = parser.parse_args()
    cli_print(score_case(load_npz_array(args.gt_mask, "masks", "mask"), load_npz_array(args.pred_mask, "masks", "mask")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
