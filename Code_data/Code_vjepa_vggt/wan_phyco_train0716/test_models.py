from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

import wan_phyco_train0716.property_maps as property_maps
from wan_phyco_train0716.models import WanPhyCoMultiControlNet
from wan_phyco_train0716.property_maps import build_null_property_map, build_pybullet_property_map


def test_zero_initialized_controller_is_identity() -> None:
    model = WanPhyCoMultiControlNet(wan_dim=64, hidden_dim=16, block_ids=(1,), patch_size=(1, 2, 2))
    hidden = torch.randn(1, 32, 64)
    maps = torch.randn(1, 12, 1, 8, 8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    residual = model.residual(1, hidden, maps, valid)
    assert torch.count_nonzero(residual) == 0
    residual.square().mean().backward()
    assert model.branches["rigid"].zero_outputs["1"].weight.grad is not None


def test_null_control_has_expected_contract() -> None:
    result = build_null_property_map(height=16, width=24)
    assert result.maps.shape == (12, 1, 16, 24)
    assert not bool(result.branch_valid.any())


def test_pybullet_rigid_and_motion_maps() -> None:
    result = build_pybullet_property_map(
        {
            "camera": {
                "eye": [0.0, -3.0, 1.4],
                "target": [0.0, 0.0, 0.4],
                "up": [0.0, 0.0, 1.0],
                "yfov_deg": 50.0,
            },
            "objects": [
                {
                    "position": [0.0, 0.0, 0.5],
                    "size": {"radius": 0.2},
                    "restitution": 0.7,
                    "friction": 0.3,
                    "linear_velocity": [1.0, 0.0, 0.0],
                }
            ],
        },
        height=32,
        width=48,
    )
    assert result.maps.shape == (12, 1, 32, 48)
    assert abs(result.maps[0].max().item() - 0.7) < 1.0e-6
    assert abs(result.maps[1].max().item() - 0.3) < 1.0e-6
    assert abs(result.maps[7].max().item() - 0.2) < 1.0e-6
    assert result.maps[10].min().item() == -1.0
    assert result.maps[11].max().item() == 1.0
    assert result.branch_valid.tolist() == [True, False, True]


def _build_synthetic_kubric(metadata: dict, segmentation: np.ndarray):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        original = property_maps._first_video_frame
        property_maps._first_video_frame = lambda _: segmentation
        try:
            return property_maps.build_kubric_property_map(path, height=8, width=8)
        finally:
            property_maps._first_video_frame = original


def test_kubric_deformation_uses_phyco_ranges() -> None:
    result = _build_synthetic_kubric(
        {
            "object_data": {
                "type": ["soft_ball"],
                "segmentation_color": [[255, 0, 0]],
                "friction": [0.5],
                "restitution": [0.2],
                "neo_hookean_mu": [330.0],
                "neo_hookean_lambda": [350.0],
                "neo_hookean_damping": [0.25],
                "use_neo_hookean": [True],
            }
        },
        np.full((8, 8, 3), [255, 0, 0], dtype=np.uint8),
    )
    assert result.maps.shape == (12, 1, 8, 8)
    assert abs(result.maps[3].max().item() - 0.5) < 1.0e-6
    assert abs(result.maps[4].max().item() - 0.5) < 1.0e-6
    assert abs(result.maps[5].max().item() - 0.25) < 1.0e-6
    assert result.maps[6].max().item() == 1.0
    assert result.branch_valid.tolist() == [True, True, False]


def test_kubric_velocity_alias_and_force_are_typed() -> None:
    segmentation = np.zeros((8, 8, 3), dtype=np.uint8)
    segmentation[:, :4] = [255, 0, 0]
    segmentation[:, 4:] = [0, 255, 0]
    common = {
        "object_data": {
            "type": ["dome", "sliding_object"],
            "position": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.2]],
            "segmentation_color": [[255, 0, 0], [0, 255, 0]],
            "friction": [1.0, 0.3],
            "restitution": [0.0, 0.0],
        },
        "applied_velocities_simulator": [{
            "object_name": "brick",
            "velocity_point_world": [1.0, 0.0, 0.2],
            "velocity_magnitude": 2.0,
            "velocity_vector_world": [1.0, 0.0, 0.0],
        }],
        "applied_velocities_image": [{"velocity_arrow_unit_vector": [0.0, 1.0]}],
    }
    velocity = _build_synthetic_kubric(common, segmentation)
    assert abs(velocity.maps[7].max().item() - 0.4) < 1.0e-6
    assert velocity.maps[10].min().item() == -1.0
    assert velocity.diagnostics["binding_action_point"] == 1

    force_metadata = dict(common)
    force_metadata.pop("applied_velocities_simulator")
    force_metadata.pop("applied_velocities_image")
    force_metadata["object_data"] = dict(common["object_data"], type=["dome", "brick"])
    force_metadata["min_force"] = 200.0
    force_metadata["max_force"] = 450.0
    force_metadata["applied_forces_simulator"] = [{
        "object_name": "brick",
        "force_point_world": [1.0, 0.0, 0.2],
        "force_magnitude": 325.0,
        "force_vector_world": [1.0, 0.0, 0.0],
    }]
    force_metadata["applied_forces_image"] = [{
        "image_coordinates": [0.0, 0.0],
        "force_end_image_coordinates": [1.0, 0.0],
    }]
    force = _build_synthetic_kubric(force_metadata, segmentation)
    assert abs(force.maps[7].max().item() - 0.5) < 1.0e-6
    assert force.maps[10].max().item() == 1.0
    assert force.maps[11].max().item() == 1.0
    assert force.diagnostics["binding_object_type"] == 1
