from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from run_training_case_diagnostics import (
    anchor_frame_indices,
    make_contact_sheet,
    mask_iou,
    noise_sweep_config,
    noise_sweep_html,
    signed_difference_image,
    sweep_value_code,
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


def test_noise_sweep_config_pairs_probe_level_and_timestep():
    args = SimpleNamespace(
        sweep_training_timesteps=[100, 300, 500, 700, 900],
        sweep_probe_noise_levels=[0.1, 0.2],
        sweep_probe_timesteps=[100, 200],
    )
    config = noise_sweep_config(args)
    assert [row["id"] for row in config["training_stages"]] == [
        "train_0100",
        "train_0300",
        "train_0500",
        "train_0700",
        "train_0900",
    ]
    assert config["probe_settings"] == (
        {"noise_level": 0.1, "timestep": 100.0, "id": "probe_010"},
        {"noise_level": 0.2, "timestep": 200.0, "id": "probe_020"},
    )
    assert sweep_value_code("probe", 0.1) == "probe_010"


def test_noise_sweep_config_rejects_unpaired_probe_settings():
    args = SimpleNamespace(
        sweep_training_timesteps=[100],
        sweep_probe_noise_levels=[0.1, 0.2],
        sweep_probe_timesteps=[100],
    )
    try:
        noise_sweep_config(args)
    except ValueError as error:
        assert "equal lengths" in str(error)
    else:
        raise AssertionError("unpaired Probe settings must fail")


def test_noise_sweep_html_contains_all_stage_probe_combinations():
    stages = [
        {
            "id": f"train_0{t}",
            "training_timestep": float(t),
            "scheduler_sigma": float(t) / 1000,
            "weighted_loss": 0.1,
            "raw_v_mse": 0.1,
            "shared_training_noise_seed": 4300,
            "peak_gpu_memory_mib": 1024,
        }
        for t in (100, 300, 500, 700, 900)
    ]
    probes = [
        {
            "id": f"probe_0{int(level * 100)}",
            "noise_level": level,
            "timestep": level * 1000,
            "scheduler_sigma": level,
            "shared_probe_noise_seed": 5201,
        }
        for level in (0.1, 0.2)
    ]
    comparisons = []
    for stage in stages:
        for probe in probes:
            comparisons.append(
                {
                    "training_stage_id": stage["id"],
                    "probe_setting_id": probe["id"],
                    "heatmap_kl_student_teacher": 0.01,
                    "trajectory_huber": 0.001,
                    "weighted_auxiliary_loss": 0.0011,
                    "gradient_to_first_pass_v_pred_norm": 0.02,
                    "peak_gpu_memory_mib": 2048,
                }
            )
    rendered = noise_sweep_html(
        {
            "training_stages": stages,
            "probe_settings": probes,
            "comparisons": comparisons,
        }
    )
    assert rendered.count('class="sweep-stage"') == 5
    assert rendered.count('class="sweep-probe"') == 10
    assert rendered.count("corresponding frame concatenation") == 20
    assert "noise_sweep/comparisons/train_0100/probe_010" in rendered
