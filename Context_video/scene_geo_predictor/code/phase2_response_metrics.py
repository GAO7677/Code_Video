"""Versioned response metrics for the Phase 2 seed-42 recheck.

The original response metric is a time-aligned vector error.  Magnitude-only
ratios are reported alongside it under explicit names and never replace it.
"""
from __future__ import annotations

import numpy as np


PAIR_R_THRESHOLD = 0.25
RADIUS_M = 0.11
EQUAL_D_PRED_THRESHOLD_M = 0.05 * RADIUS_M


def vector_pair_metrics(pred_a: np.ndarray, pred_b: np.ndarray,
                        gt_a: np.ndarray, gt_b: np.ndarray) -> dict:
    """Compute the frozen, time-aligned pair metric for [T,3] trajectories."""
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    gt_a = np.asarray(gt_a, dtype=np.float64)
    gt_b = np.asarray(gt_b, dtype=np.float64)
    if not (pred_a.shape == pred_b.shape == gt_a.shape == gt_b.shape and pred_a.ndim == 2 and pred_a.shape[1] == 3):
        raise ValueError("pair trajectories must all have shape [T,3]")
    true_delta = gt_a - gt_b
    pred_delta = pred_a - pred_b
    d_gt_t = np.linalg.norm(true_delta, axis=-1)
    d_pred_t = np.linalg.norm(pred_delta, axis=-1)
    e_delta_t = np.linalg.norm(pred_delta - true_delta, axis=-1)
    d_gt = float(d_gt_t.mean())
    d_pred = float(d_pred_t.mean())
    e_delta = float(e_delta_t.mean())
    equal = bool(d_gt <= 1e-8)
    return {
        "D_gt_m": d_gt,
        "D_pred_m": d_pred,
        "E_delta_m": e_delta,
        "R_delta": None if equal else e_delta / d_gt,
        "response_magnitude_ratio": None if equal else d_pred / d_gt,
        "response_magnitude_error": None if equal else abs(d_pred - d_gt) / d_gt,
        "equal_future": equal,
        "equal_future_absolute_prediction_delta_m": d_pred if equal else None,
        "strong_response_pass": bool((not equal) and e_delta / d_gt <= PAIR_R_THRESHOLD),
        "equal_future_pass": bool(equal and d_pred <= EQUAL_D_PRED_THRESHOLD_M),
    }


def synthetic_selftests() -> dict:
    """Fixed fixtures that distinguish direction, timing, and common shifts."""
    gt_a = np.asarray([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], dtype=np.float64)
    gt_b = np.zeros_like(gt_a)
    same = vector_pair_metrics(gt_a, gt_b, gt_a, gt_b)
    same_pred = vector_pair_metrics(gt_b, gt_b, gt_a, gt_b)
    opposite = vector_pair_metrics(gt_b, gt_a, gt_a, gt_b)
    shifted_a = gt_a + np.asarray([0., 3., 0.])
    shifted_b = gt_b + np.asarray([0., 3., 0.])
    common_shift = vector_pair_metrics(shifted_a, shifted_b, gt_a, gt_b)
    common_shift_ade = float(np.linalg.norm(shifted_a - gt_a, axis=-1).mean())
    # Same total magnitude but the response occurs at the wrong time.
    timing_a = np.asarray([[0., 0., 0.], [0., 0., 0.], [3., 0., 0.]], dtype=np.float64)
    timing_b = np.zeros_like(timing_a)
    timing = vector_pair_metrics(timing_a, timing_b, gt_a, gt_b)
    equal = vector_pair_metrics(gt_b, gt_b, gt_b, gt_b)
    checks = {
        "prediction_equals_gt_zero_error": same["E_delta_m"] == 0.0 and same["R_delta"] == 0.0,
        "same_prediction_non_degenerate_R_is_one": same_pred["R_delta"] == 1.0,
        "opposite_direction_magnitude_same_but_R_is_two": abs(opposite["response_magnitude_ratio"] - 1.0) < 1e-12 and abs(opposite["response_magnitude_error"]) < 1e-12 and abs(opposite["R_delta"] - 2.0) < 1e-12,
        "timing_shift_detected_by_vector_error": timing["response_magnitude_error"] == 0.0 and timing["R_delta"] > 0.0,
        "common_shift_keeps_pair_error_but_not_absolute_ADE": common_shift["E_delta_m"] == 0.0 and common_shift_ade > 0.0,
        "equal_future_is_explicit_na": equal["R_delta"] is None and equal["response_magnitude_ratio"] is None and equal["equal_future_absolute_prediction_delta_m"] == 0.0,
    }
    return {"status": "EXECUTED", "all_pass": all(checks.values()), "checks": checks,
            "fixtures": {"same": same, "same_prediction": same_pred, "opposite": opposite,
                          "timing_shift": timing, "common_shift": common_shift,
                          "common_shift_ADE_m": common_shift_ade, "equal_future": equal}}


__all__ = ["vector_pair_metrics", "synthetic_selftests", "PAIR_R_THRESHOLD", "RADIUS_M", "EQUAL_D_PRED_THRESHOLD_M"]
