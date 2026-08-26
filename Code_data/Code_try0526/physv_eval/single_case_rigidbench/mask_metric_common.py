from __future__ import annotations

import numpy as np

from .common import as_masks, check_same_mask_shape


def score_mask_metric(gt_mask: np.ndarray, pred_mask: np.ndarray, per_frame_fn) -> dict:
    gt = as_masks(gt_mask, "gt_mask")
    pred = as_masks(pred_mask, "pred_mask")
    check_same_mask_shape(gt, pred)
    per_actor = np.stack(
        [per_frame_fn(gt[:, i : i + 1], pred[:, i : i + 1]) for i in range(gt.shape[1])],
        axis=0,
    )
    per_frame = np.nanmean(per_actor, axis=0)
    return {"value": float(np.nanmean(per_frame)), "per_frame": per_frame}
