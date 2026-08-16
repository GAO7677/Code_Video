from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from run_training_case_diagnostics import (
    anchor_frame_indices,
    compute_latent_mask_comparison,
    compute_probe_weighting_comparison,
    make_contact_sheet,
    mask_iou,
    noise_sweep_config,
    noise_sweep_html,
    render_equal_pck_timeline,
    render_heatmap_timelines,
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


def test_probe_comparison_keeps_linear_and_gamma30_losses_distinct():
    teacher = torch.tensor(
        [[[0.8, 0.2], [0.3, 0.7]]], dtype=torch.float32
    )
    student = torch.tensor(
        [[[0.5, 0.5], [0.4, 0.6]]], dtype=torch.float32
    )
    comparison = compute_probe_weighting_comparison(
        teacher,
        student,
        grid=(1, 1, 2),
        pck_linear_weights=torch.tensor([0.4, 0.6]),
        pck_weight_power=30.0,
        trajectory_huber_delta=0.05,
    )
    per_head = comparison["per_head_kl_teacher_student"]
    linear_weights = comparison["pck_linear_weights"]
    sharpened_weights = comparison["pck_sharpened_weights"]
    assert torch.allclose(
        comparison["pck_linear_head_kl_teacher_student"],
        (per_head * linear_weights).sum(),
    )
    assert torch.allclose(
        comparison["pck_sharpened_head_kl_teacher_student"],
        (per_head * sharpened_weights).sum(),
    )
    assert float(sharpened_weights.max() / sharpened_weights.min()) > 1.0e5


def test_latent_mask_comparison_conditions_each_frame_and_reaches_student():
    teacher = torch.tensor(
        [
            [
                [0.10, 0.20, 0.05, 0.05, 0.08, 0.12, 0.20, 0.20],
                [0.05, 0.05, 0.20, 0.20, 0.10, 0.10, 0.15, 0.15],
            ]
        ]
    )
    student = torch.tensor(
        [
            [
                [0.20, 0.10, 0.05, 0.05, 0.12, 0.08, 0.20, 0.20],
                [0.08, 0.12, 0.20, 0.10, 0.05, 0.15, 0.15, 0.15],
            ]
        ],
        requires_grad=True,
    )
    occupancy = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.5], [1.0, 0.0]]]]
    )

    comparison = compute_latent_mask_comparison(
        teacher,
        student,
        object_token_occupancy_bthw=occupancy,
        grid=(2, 2, 2),
        head_weights=torch.tensor([0.7, 0.3]),
        source_frame=0,
        lambda_mask=0.01,
    )

    assert comparison["valid"].tolist() == [[False, True]]
    assert torch.allclose(
        comparison["student_region_attention"].sum(dim=-1),
        torch.ones(1, 2),
    )
    assert torch.allclose(comparison["target"].sum(dim=-1), torch.ones(1, 2))
    assert torch.allclose(
        comparison["loss"], 0.01 * comparison["raw_soft_ce"]
    )
    comparison["loss"].backward()
    assert student.grad is not None and float(student.grad.norm()) > 0.0


def test_latent_mask_comparison_accepts_explicit_framewise_head_maps():
    teacher = torch.tensor(
        [[[[0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.6, 0.1]]]],
        dtype=torch.float32,
    )
    student = torch.tensor(
        [[[[0.6, 0.2, 0.1, 0.1], [0.2, 0.2, 0.5, 0.1]]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    occupancy = torch.tensor(
        [[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]]]
    )

    comparison = compute_latent_mask_comparison(
        teacher,
        student,
        object_token_occupancy_bthw=occupancy,
        grid=(2, 2, 2),
        head_weights=torch.ones(1),
        source_frame=0,
        lambda_mask=0.01,
    )

    assert torch.allclose(
        comparison["student_region_attention"], student[:, 0]
    )
    comparison["loss"].backward()
    assert student.grad is not None and float(student.grad.norm()) > 0.0


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
                    "pck_sharpened_media_complete": True,
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
    assert rendered.count("equal_vs_pck_gamma30_top100_timeline.jpg") == 10
    assert rendered.count("pck_gamma30/teacher_student_five_panel.mp4") == 10
    assert rendered.count("Inputs and frame-by-frame media") == 10
    assert "noise_sweep/comparisons/train_0100/probe_010" in rendered


def test_heatmap_timelines_form_one_row_per_role_with_shared_frame_order():
    teacher = np.zeros((1, 13, 4, 7), dtype=np.float32)
    student = np.zeros_like(teacher)
    for index in range(13):
        teacher[0, index, index % 4, index % 7] = 1.0
        student[0, index, (index + 1) % 4, (index + 2) % 7] = 1.0
    with TemporaryDirectory() as directory:
        files = render_heatmap_timelines(
            Path(directory), teacher, student, pixel_frames=49
        )
        teacher_image = cv2.imread(str(Path(directory) / files["teacher_timeline"]))
        student_image = cv2.imread(str(Path(directory) / files["student_timeline"]))
        combined_image = cv2.imread(str(Path(directory) / files["combined_timeline"]))
    assert teacher_image.shape == (96, 13 * 160, 3)
    assert student_image.shape == teacher_image.shape
    assert combined_image.shape == (96 * 2 + 6, 13 * 160, 3)


def test_equal_pck_timeline_has_six_rows_under_one_scale():
    maps = []
    for offset in range(6):
        values = np.zeros((1, 13, 4, 7), dtype=np.float32)
        for index in range(13):
            values[0, index, (index + offset) % 4, (index + offset) % 7] = 1.0
        maps.append(values)
    with TemporaryDirectory() as directory:
        filename = render_equal_pck_timeline(
            Path(directory),
            teacher_equal=maps[0],
            student_equal=maps[1],
            teacher_pck_linear=maps[2],
            student_pck_linear=maps[3],
            teacher_pck_sharpened=maps[4],
            student_pck_sharpened=maps[5],
            pck_weight_power=30.0,
            pixel_frames=49,
        )
        image = cv2.imread(str(Path(directory) / filename))
    assert filename == "equal_vs_pck_gamma30_top100_timeline.jpg"
    assert image.shape == (96 * 6 + 6 * 5, 13 * 160, 3)
