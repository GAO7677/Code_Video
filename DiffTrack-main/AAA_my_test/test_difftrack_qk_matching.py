#!/usr/bin/env python3
"""Numerical checks for the DiffTrack-compatible Wan Q/K adapter."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from AAA_my_test.difftrack_qk_matching import match_query_points
from utils.matching import corr_to_matches


def reference_match(
    source_q: torch.Tensor,
    source_k: torch.Tensor,
    target_q: torch.Tensor,
    target_k: torch.Tensor,
    points: torch.Tensor,
    grid_hw: tuple[int, int],
    pixel_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = grid_hw
    pixel_height, pixel_width = pixel_hw
    scale = math.sqrt(source_q.shape[-1])
    forward = torch.einsum("h i d, h j d -> h i j", source_q.permute(1, 0, 2), target_k.permute(1, 0, 2)) / scale
    reverse = torch.einsum("h i d, h j d -> h i j", target_q.permute(1, 0, 2), source_k.permute(1, 0, 2)) / scale
    forward = forward.softmax(dim=-1).mean(dim=0)
    reverse = reverse.softmax(dim=-1).mean(dim=0)
    correlation = 0.5 * (forward.transpose(0, 1) + reverse)
    x_target, y_target, _, _, _ = corr_to_matches(
        correlation.reshape(1, height, width, height, width).unsqueeze(1),
        get_maximum=True,
        do_softmax=True,
        device=correlation.device,
    )
    mapping = torch.stack((x_target, y_target), dim=1).reshape(1, 2, height, width)
    stride_x = pixel_width / width
    stride_y = pixel_height / height
    margin = pixel_width / (64 * stride_x)
    grid = points.clone()
    grid[:, 0] = (grid[:, 0] / stride_x) / (width - margin) * 2 - 1
    grid[:, 1] = (grid[:, 1] / stride_y) / (height - margin) * 2 - 1
    sampled = F.grid_sample(mapping, grid.view(1, len(grid), 1, 2), align_corners=True)
    result = sampled[0, :, :, 0].transpose(0, 1)
    result[:, 0] *= stride_x
    result[:, 1] *= stride_y
    return result


class DiffTrackMatchingTest(unittest.TestCase):
    def test_matches_repository_dense_mapping_path(self) -> None:
        torch.manual_seed(7)
        grid_hw = (3, 4)
        pixel_hw = (96, 128)
        tensors = [torch.randn(12, 2, 5) for _ in range(4)]
        points = torch.tensor([[32.0, 32.0], [72.0, 48.0], [96.0, 64.0]])
        actual, probabilities = match_query_points(
            *tensors, points, grid_hw, pixel_hw
        )
        expected = reference_match(*tensors, points, grid_hw, pixel_hw)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            probabilities.sum(dim=-1), torch.ones(len(points)), atol=1e-6, rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
