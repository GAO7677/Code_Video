import torch

from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.capture_phase_b_top100_attention_overlays import (
    fixed_query_head_maps,
)


def test_fixed_query_head_maps_preserve_all_key_frames_and_query_mass():
    torch.manual_seed(7)
    grid = (3, 2, 2)
    qh = torch.randn(2, 3, 12, 4)
    kh = torch.randn(2, 3, 12, 4)

    maps = fixed_query_head_maps(qh, kh, (4, 5), grid)

    assert maps.shape == (3, 3, 2, 2)
    torch.testing.assert_close(maps.sum(dim=(1, 2, 3)), torch.full((3,), 2.0))


def test_fixed_query_head_maps_uniform_attention_is_uniform_over_keys():
    grid = (2, 1, 3)
    qh = torch.zeros(1, 1, 6, 2)
    kh = torch.zeros_like(qh)

    maps = fixed_query_head_maps(qh, kh, (0, 1, 2), grid)

    torch.testing.assert_close(maps, torch.full((1, 2, 1, 3), 0.5))
