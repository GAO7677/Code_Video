#!/usr/bin/env python3
"""DiffTrack-compatible dense Q/K matching for arbitrary video token grids."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.matching import corr_to_matches


IMPLEMENTATION = "utils.matching.corr_to_matches+torch.grid_sample"


def _query_grid(
    query_points: torch.Tensor,
    grid_hw: tuple[int, int],
    pixel_hw: tuple[int, int],
) -> tuple[torch.Tensor, float, float]:
    height, width = grid_hw
    pixel_height, pixel_width = pixel_hw
    stride_x = pixel_width / width
    stride_y = pixel_height / height
    token_points = query_points.to(dtype=torch.float32).clone()
    token_points[:, 0] /= stride_x
    token_points[:, 1] /= stride_y

    # Preserve DiffTrack's grid_sample normalization, generalized to the Wan grid.
    margin = pixel_width / (64 * stride_x)
    token_points[:, 0] = token_points[:, 0] / (width - margin) * 2 - 1
    token_points[:, 1] = token_points[:, 1] / (height - margin) * 2 - 1
    return token_points.view(1, len(token_points), 1, 2), stride_x, stride_y


def match_query_points(
    source_q: torch.Tensor,
    source_k: torch.Tensor,
    target_q: torch.Tensor,
    target_k: torch.Tensor,
    query_points: torch.Tensor,
    grid_hw: tuple[int, int],
    pixel_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DiffTrack's symmetric dense matching and sample it at query points.

    Q/K tensors use ``[spatial, heads, head_dim]``. Returned tracks are pixel
    coordinates and probabilities are ``[query, target_spatial]``.
    """
    height, width = grid_hw
    spatial = height * width
    expected = (spatial, source_q.shape[1], source_q.shape[2])
    if any(tuple(tensor.shape) != expected for tensor in (source_q, source_k, target_q, target_k)):
        raise ValueError("DiffTrack matching expects equal [spatial, heads, head_dim] Q/K tensors")

    scale = math.sqrt(source_q.shape[-1])
    forward = torch.einsum("ihd,jhd->hij", source_q, target_k) / scale
    reverse = torch.einsum("jhd,ihd->hji", target_q, source_k) / scale
    forward = forward.softmax(dim=-1).mean(dim=0)
    reverse = reverse.softmax(dim=-1).mean(dim=0)

    # DiffTrack lays the dense correlation out as [target_y, target_x, source_y, source_x].
    correlation = 0.5 * (forward.transpose(0, 1) + reverse)
    corr4d = correlation.reshape(1, height, width, height, width).unsqueeze(1)
    x_target, y_target, _, _, _ = corr_to_matches(
        corr4d,
        get_maximum=True,
        do_softmax=True,
        device=correlation.device,
    )
    mapping = torch.stack((x_target, y_target), dim=1).reshape(1, 2, height, width)
    query_grid, stride_x, stride_y = _query_grid(
        query_points.to(correlation.device), grid_hw, pixel_hw
    )
    sampled = F.grid_sample(mapping, query_grid, mode="bilinear", align_corners=True)
    tracks = sampled[0, :, :, 0].transpose(0, 1)
    tracks[:, 0] *= stride_x
    tracks[:, 1] *= stride_y

    sampled_correlation = F.grid_sample(
        correlation.reshape(1, spatial, height, width),
        query_grid,
        mode="bilinear",
        align_corners=True,
    )
    probabilities = sampled_correlation[0, :, :, 0].transpose(0, 1).softmax(dim=-1)
    return tracks, probabilities
