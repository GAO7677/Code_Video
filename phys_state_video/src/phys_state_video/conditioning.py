from __future__ import annotations

from dataclasses import dataclass

from .config import ConditioningConfig
from .schemas import StateIndex
from .utils import require_torch

torch = require_torch()


@dataclass(slots=True)
class ConditionBundle:
    maps: torch.Tensor
    memory_tokens: torch.Tensor


def _meshgrid(height: int, width: int, device):
    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    return torch.meshgrid(ys, xs, indexing="ij")


def _draw_heatmap(center_x, center_y, sigma, height, width, device):
    yy, xx = _meshgrid(height, width, device)
    return torch.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma))


def build_condition_bundle(
    future_states: torch.Tensor,
    future_boxes: torch.Tensor,
    appearance: torch.Tensor,
    config: ConditioningConfig | None = None,
) -> ConditionBundle:
    config = config or ConditioningConfig()
    batch, num_steps, num_objects, _ = future_states.shape
    height = config.frame_height
    width = config.frame_width
    vel_channels = 2 if config.include_velocity_maps else 0
    existence_channels = 1 if config.include_existence_map else 0
    channels = 4 + vel_channels + existence_channels
    maps = torch.zeros(
        (batch, num_steps, channels, height, width),
        device=future_states.device,
        dtype=future_states.dtype,
    )

    yy, xx = _meshgrid(height, width, future_states.device)
    yy = yy.view(1, 1, 1, height, width)
    xx = xx.view(1, 1, 1, height, width)

    centers_x = future_states[..., StateIndex.CENTER_X].unsqueeze(-1).unsqueeze(-1) * width
    centers_y = future_states[..., StateIndex.CENTER_Y].unsqueeze(-1).unsqueeze(-1) * height
    sigma_sq = 2.0 * config.heatmap_sigma * config.heatmap_sigma
    heatmaps = torch.exp(-((xx - centers_x) ** 2 + (yy - centers_y) ** 2) / sigma_sq)
    maps[:, :, 0] = heatmaps.max(dim=2).values

    x0 = torch.clamp((future_boxes[..., 0] * width).floor(), 0, width - 1)
    y0 = torch.clamp((future_boxes[..., 1] * height).floor(), 0, height - 1)
    x1 = torch.clamp((future_boxes[..., 2] * width).ceil(), 1, width)
    y1 = torch.clamp((future_boxes[..., 3] * height).ceil(), 1, height)

    x1 = torch.maximum(x1, x0 + 1)
    y1 = torch.maximum(y1, y0 + 1)

    bbox_mask = (
        (xx >= x0.unsqueeze(-1).unsqueeze(-1))
        & (xx < x1.unsqueeze(-1).unsqueeze(-1))
        & (yy >= y0.unsqueeze(-1).unsqueeze(-1))
        & (yy < y1.unsqueeze(-1).unsqueeze(-1))
    )
    mask = bbox_mask.to(future_states.dtype)

    maps[:, :, 1] = bbox_mask.any(dim=2).to(future_states.dtype)
    maps[:, :, 2] = (
        mask * future_states[..., StateIndex.DEPTH].unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=2)
    maps[:, :, 3] = (
        mask * future_states[..., StateIndex.VISIBILITY].unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=2)

    channel_offset = 4
    if config.include_existence_map:
        maps[:, :, channel_offset] = (
            mask * future_states[..., StateIndex.EXISTENCE].unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=2)
        channel_offset += 1
    if config.include_velocity_maps:
        maps[:, :, channel_offset] = (
            mask * future_states[..., StateIndex.VEL_X].unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=2)
        maps[:, :, channel_offset + 1] = (
            mask * future_states[..., StateIndex.VEL_Y].unsqueeze(-1).unsqueeze(-1)
        ).sum(dim=2)

    last_scale = future_states[:, -1, :, StateIndex.LOG_SCALE:StateIndex.LOG_SCALE + 1]
    last_conf = future_states[:, -1, :, StateIndex.CONFIDENCE:StateIndex.CONFIDENCE + 1]
    memory_tokens = torch.cat([appearance, last_scale, last_conf], dim=-1)
    return ConditionBundle(maps=maps, memory_tokens=memory_tokens)
