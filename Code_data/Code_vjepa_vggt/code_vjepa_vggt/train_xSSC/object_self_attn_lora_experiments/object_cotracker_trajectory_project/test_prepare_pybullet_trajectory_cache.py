from __future__ import annotations

import numpy as np
import torch

from prepare_pybullet_trajectory_cache import (
    TRACK_HEIGHT,
    TRACK_WIDTH,
    prepare_tracker_inputs,
)


def test_prepare_tracker_inputs_scales_each_axis_once() -> None:
    frames = np.zeros((2, 512, 896, 3), dtype=np.uint8)
    points = np.asarray(
        [
            [0.0, 0.0],
            [895.0, 511.0],
            [447.5, 255.5],
        ],
        dtype=np.float32,
    )
    video, queries = prepare_tracker_inputs(
        frames,
        points,
        anchor_frame=4,
        device=torch.device("cpu"),
    )

    assert video.shape == (1, 2, 3, TRACK_HEIGHT, TRACK_WIDTH)
    torch.testing.assert_close(queries[0, :, 0], torch.full((3,), 4.0))
    expected = torch.tensor(
        [
            [0.0, 0.0],
            [TRACK_WIDTH - 1.0, TRACK_HEIGHT - 1.0],
            [(TRACK_WIDTH - 1.0) / 2.0, (TRACK_HEIGHT - 1.0) / 2.0],
        ]
    )
    torch.testing.assert_close(queries[0, :, 1:], expected)
