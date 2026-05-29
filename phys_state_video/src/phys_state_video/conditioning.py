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
    maps = torch.zeros((batch, num_steps, channels, height, width), device=future_states.device, dtype=future_states.dtype)

    for b_idx in range(batch):
        for t_idx in range(num_steps):
            for o_idx in range(num_objects):
                state = future_states[b_idx, t_idx, o_idx]
                box = future_boxes[b_idx, t_idx, o_idx]
                cx = state[StateIndex.CENTER_X] * width
                cy = state[StateIndex.CENTER_Y] * height
                heatmap = _draw_heatmap(cx, cy, config.heatmap_sigma, height, width, future_states.device)

                x0 = int(torch.clamp(box[0] * width, 0, width - 1).item())
                y0 = int(torch.clamp(box[1] * height, 0, height - 1).item())
                x1 = int(torch.clamp(box[2] * width, x0 + 1, width).item())
                y1 = int(torch.clamp(box[3] * height, y0 + 1, height).item())

                maps[b_idx, t_idx, 0] = torch.maximum(maps[b_idx, t_idx, 0], heatmap)
                maps[b_idx, t_idx, 1, y0:y1, x0:x1] = 1.0
                maps[b_idx, t_idx, 2, y0:y1, x0:x1] = state[StateIndex.DEPTH]
                maps[b_idx, t_idx, 3, y0:y1, x0:x1] = state[StateIndex.VISIBILITY]
                channel_offset = 4
                if config.include_existence_map:
                    maps[b_idx, t_idx, channel_offset, y0:y1, x0:x1] = state[StateIndex.EXISTENCE]
                    channel_offset += 1
                if config.include_velocity_maps:
                    maps[b_idx, t_idx, channel_offset, y0:y1, x0:x1] = state[StateIndex.VEL_X]
                    maps[b_idx, t_idx, channel_offset + 1, y0:y1, x0:x1] = state[StateIndex.VEL_Y]

    last_scale = future_states[:, -1, :, StateIndex.LOG_SCALE:StateIndex.LOG_SCALE + 1]
    last_conf = future_states[:, -1, :, StateIndex.CONFIDENCE:StateIndex.CONFIDENCE + 1]
    memory_tokens = torch.cat([appearance, last_scale, last_conf], dim=-1)
    return ConditionBundle(maps=maps, memory_tokens=memory_tokens)
