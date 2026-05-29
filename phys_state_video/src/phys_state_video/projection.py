from __future__ import annotations

from .config import ProjectionConfig
from .schemas import StateIndex
from .utils import require_torch


class ConfidenceAwareProjector:
    def __init__(self, config: ProjectionConfig | None = None):
        self.config = config or ProjectionConfig()

    def project(self, future_states):
        torch = require_torch()
        projected = future_states.clone()
        _, num_steps, _, _ = projected.shape

        for step in range(1, num_steps):
            prev = projected[:, step - 1]
            curr = projected[:, step]

            prev_center = prev[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
            curr_center = curr[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1]
            prev_vel = prev[..., StateIndex.VEL_X:StateIndex.VEL_Y + 1]
            pred_center = prev_center + prev_vel
            center_delta = curr_center - pred_center
            conf = curr[..., StateIndex.CONFIDENCE:StateIndex.CONFIDENCE + 1]
            low_conf = (conf < self.config.low_confidence_threshold).float()
            center_norm = torch.linalg.norm(center_delta, dim=-1, keepdim=True)
            smooth_mask = (center_norm > self.config.max_position_delta).float() * low_conf
            curr_center = curr_center - self.config.velocity_smooth_weight * smooth_mask * center_delta
            curr[..., StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = curr_center
            curr[..., StateIndex.VEL_X:StateIndex.VEL_Y + 1] = curr_center - prev_center

            prev_depth = prev[..., StateIndex.DEPTH]
            prev_depth_vel = prev[..., StateIndex.DEPTH_VEL]
            pred_depth = prev_depth + prev_depth_vel
            curr_depth = curr[..., StateIndex.DEPTH]
            depth_delta = curr_depth - pred_depth
            curr[..., StateIndex.DEPTH] = curr_depth - self.config.depth_smooth_weight * low_conf.squeeze(-1) * depth_delta
            curr[..., StateIndex.DEPTH_VEL] = curr[..., StateIndex.DEPTH] - prev_depth

            prev_vis = prev[..., StateIndex.VISIBILITY]
            curr_vis = curr[..., StateIndex.VISIBILITY]
            turned_off = (prev_vis > self.config.visibility_on_threshold) & (curr_vis < self.config.visibility_off_threshold)
            turned_on = (prev_vis < self.config.visibility_off_threshold) & (curr_vis > self.config.visibility_on_threshold)
            curr[..., StateIndex.VISIBILITY] = torch.where(
                turned_off | turned_on,
                0.5 * (prev_vis + curr_vis),
                curr_vis,
            )

            depth_change = torch.abs(curr[..., StateIndex.DEPTH] - prev[..., StateIndex.DEPTH])
            scale_change = torch.abs(curr[..., StateIndex.LOG_SCALE] - prev[..., StateIndex.LOG_SCALE])
            scale_mask = (
                (depth_change < 0.1)
                & (scale_change > self.config.max_scale_delta)
                & (conf.squeeze(-1) >= self.config.low_confidence_threshold)
            )
            target_scale = prev[..., StateIndex.LOG_SCALE]
            curr[..., StateIndex.LOG_SCALE] = torch.where(
                scale_mask,
                (1.0 - self.config.scale_depth_weight) * curr[..., StateIndex.LOG_SCALE]
                + self.config.scale_depth_weight * target_scale,
                curr[..., StateIndex.LOG_SCALE],
            )
            projected[:, step] = curr
        return projected
