from __future__ import annotations

import json

import numpy as np

from AAA_my_test.wan_context_point_guidance.attention_audit_visualization import (
    write_step_attention_audit,
)


def _capture(heatmap: np.ndarray, local_shift: float) -> dict[str, object]:
    time_count = heatmap.shape[0]
    return {
        "heatmap": heatmap,
        "frame_mass": heatmap.sum(axis=(1, 2)),
        "localized_mass": np.full(time_count, 0.02 + local_shift, dtype=np.float32),
        "peak_distance_tokens": np.full(time_count, 2.0 - local_shift, dtype=np.float32),
        "peak_hit_rate_2sigma": np.full(time_count, 0.4 + local_shift, dtype=np.float32),
    }


def test_writes_five_panel_attention_audit(tmp_path) -> None:
    rng = np.random.default_rng(7)
    source = rng.integers(0, 255, size=(13, 64, 96, 3), dtype=np.uint8)
    predicted = rng.integers(0, 255, size=(13, 64, 96, 3), dtype=np.uint8)
    pre_heat = rng.random((13, 4, 6), dtype=np.float32) * 0.01
    post_heat = pre_heat.copy()
    post_heat[:, 1:3, 2:4] += 0.01
    tracks = np.full((13, 2, 2), (48.0, 32.0), dtype=np.float32)
    visibility = np.ones((13, 2), dtype=bool)
    report = write_step_attention_audit(
        tmp_path,
        5,
        source,
        np.arange(13, dtype=np.int64) * 4,
        tracks,
        visibility,
        {"pre": _capture(pre_heat, 0.0), "post": _capture(post_heat, 0.1)},
        predicted,
        1,
        2.0,
        1.5,
    )
    step = tmp_path / "step_05"
    assert (step / "attention_comparison.mp4").stat().st_size > 0
    assert (step / "raw_attention_maps.npz").stat().st_size > 0
    assert (step / "complete.json").is_file()
    payload = json.loads((step / "metrics.json").read_text(encoding="utf-8"))
    assert report["summary"]["loss_change"] == -0.5
    assert payload["normalization"].startswith("global softmax")
    assert len(payload["frames"]) == 13
