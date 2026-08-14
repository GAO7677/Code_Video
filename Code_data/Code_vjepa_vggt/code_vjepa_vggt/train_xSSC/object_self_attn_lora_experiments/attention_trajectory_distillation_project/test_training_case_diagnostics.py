from __future__ import annotations

import numpy as np

from run_training_case_diagnostics import (
    anchor_frame_indices,
    make_contact_sheet,
    mask_iou,
    signed_difference_image,
)


def test_anchor_frame_indices_follow_wan_4n_plus_1_mapping():
    assert anchor_frame_indices(13, 49) == list(range(0, 49, 4))


def test_mask_iou_handles_exact_disjoint_and_partial_masks():
    left = np.zeros((4, 4), dtype=np.uint8)
    right = np.zeros((4, 4), dtype=np.uint8)
    left[:2, :2] = 1
    right[:2, 1:3] = 1
    assert mask_iou(left, left) == 1.0
    assert mask_iou(left, np.zeros_like(left)) == 0.0
    assert np.isclose(mask_iou(left, right), 2.0 / 6.0)


def test_contact_sheet_and_signed_difference_have_stable_shapes():
    frames = [np.zeros((20, 30, 3), dtype=np.uint8) for _ in range(5)]
    assert make_contact_sheet(frames, columns=2).shape == (60, 60, 3)
    values = np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32)
    rendered = signed_difference_image(values, (12, 8), vmax=1.0)
    assert rendered.shape == (8, 12, 3)
    assert rendered.dtype == np.uint8
