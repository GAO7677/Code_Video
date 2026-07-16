from __future__ import annotations

import torch

from wan_phyco_train0716.models import WanPhyCoMultiControlNet
from wan_phyco_train0716.property_maps import build_null_property_map, build_pybullet_property_map


def test_zero_initialized_controller_is_identity() -> None:
    model = WanPhyCoMultiControlNet(wan_dim=64, hidden_dim=16, block_ids=(1,), patch_size=(1, 2, 2))
    hidden = torch.randn(1, 32, 64)
    maps = torch.randn(1, 9, 1, 8, 8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    residual = model.residual(1, hidden, maps, valid)
    assert torch.count_nonzero(residual) == 0
    residual.square().mean().backward()
    assert model.branches["rigid"].zero_outputs["1"].weight.grad is not None


def test_null_control_has_expected_contract() -> None:
    result = build_null_property_map(height=16, width=24)
    assert result.maps.shape == (9, 1, 16, 24)
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
    assert result.maps.shape == (9, 1, 32, 48)
    assert abs(result.maps[0].max().item() - 0.7) < 1.0e-6
    assert abs(result.maps[1].max().item() - 0.3) < 1.0e-6
    assert result.branch_valid.tolist() == [True, False, True]
