from __future__ import annotations

import numpy as np
import torch

from prepare_pybullet_trajectory_cache import (
    TRACK_HEIGHT,
    TRACK_WIDTH,
    atomic_json,
    prepare_tracker_inputs,
    select_maximum_distinct_detections,
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


def test_atomic_json_uses_process_scoped_temporary_file(tmp_path) -> None:
    target = tmp_path / "cache_config.json"
    atomic_json(target, {"status": "complete", "count": 3})

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob(".cache_config.json.*.tmp"))


def test_distinct_detection_selection_drops_lower_score_conflict() -> None:
    box_a = np.asarray([10.0, 10.0, 30.0, 30.0], dtype=np.float32)
    box_b = np.asarray([60.0, 10.0, 80.0, 30.0], dtype=np.float32)
    detections = [
        [{"box": box_a, "score": 0.9}],
        [{"box": box_a.copy(), "score": 0.8}],
        [{"box": box_b, "score": 0.7}],
    ]

    phrase_indices, candidate_indices, selected = (
        select_maximum_distinct_detections(detections)
    )

    assert phrase_indices == (0, 2)
    assert candidate_indices == (0, 0)
    assert selected == [detections[0][0], detections[2][0]]


def test_distinct_detection_selection_skips_phrase_without_candidates() -> None:
    visible = {"box": np.asarray([10.0, 10.0, 30.0, 30.0]), "score": 0.9}

    phrase_indices, candidate_indices, selected = (
        select_maximum_distinct_detections([[], [visible]])
    )

    assert phrase_indices == (1,)
    assert candidate_indices == (0,)
    assert selected == [visible]
