from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TeacherStudentSlices:
    context_tokens: torch.Tensor
    future_tokens: torch.Tensor


def split_context_future_tokens(
    full_tokens: torch.Tensor,
    *,
    context_latent_frames: int,
) -> TeacherStudentSlices:
    if full_tokens.ndim != 4:
        raise ValueError(f"full_tokens must have shape [B,T,O,D], got {list(full_tokens.shape)}")
    context_latent_frames = int(context_latent_frames)
    if context_latent_frames <= 0 or context_latent_frames >= int(full_tokens.shape[1]):
        raise ValueError(
            f"context_latent_frames must be in [1, T-1], got {context_latent_frames} for T={int(full_tokens.shape[1])}"
        )
    return TeacherStudentSlices(
        context_tokens=full_tokens[:, :context_latent_frames].contiguous(),
        future_tokens=full_tokens[:, context_latent_frames:].contiguous(),
    )


def future_object_valid_mask(
    future_boxes: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if future_boxes.ndim != 4:
        raise ValueError(f"future_boxes must have shape [B,T,O,4], got {list(future_boxes.shape)}")
    valid = ((future_boxes[..., 2] - future_boxes[..., 0]) > eps) & ((future_boxes[..., 3] - future_boxes[..., 1]) > eps)
    return valid.any(dim=1 if future_boxes.shape[2] == valid.shape[2] else 2)


def build_future_track_summary(
    future_boxes: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    future_latent_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if future_boxes.ndim != 4:
        raise ValueError(f"future_boxes must have shape [B,T,O,4], got {list(future_boxes.shape)}")
    height, width = int(image_hw[0]), int(image_hw[1])
    batch, future_frames, objects, _ = future_boxes.shape
    if future_frames % max(int(future_latent_frames), 1) != 0:
        raise ValueError(
            f"future_frames={future_frames} must be divisible by future_latent_frames={future_latent_frames}"
        )
    group = future_frames // int(future_latent_frames)
    grouped = future_boxes.view(batch, int(future_latent_frames), group, objects, 4)
    valid = ((grouped[..., 2] - grouped[..., 0]) > 1.0e-6) & ((grouped[..., 3] - grouped[..., 1]) > 1.0e-6)
    valid_perm = valid.bool().permute(0, 1, 3, 2)
    any_valid = valid_perm.any(dim=-1)
    first_idx = valid_perm.float().argmax(dim=-1)
    last_idx = group - 1 - valid_perm.flip(dims=[-1]).float().argmax(dim=-1)
    first_idx = torch.where(any_valid, first_idx, torch.zeros_like(first_idx))
    last_idx = torch.where(any_valid, last_idx, torch.zeros_like(last_idx))
    grouped_perm = grouped.permute(0, 1, 3, 2, 4)
    gather_first = first_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 4).long()
    gather_last = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 4).long()
    first = torch.gather(grouped_perm, dim=3, index=gather_first).squeeze(3)
    last = torch.gather(grouped_perm, dim=3, index=gather_last).squeeze(3)
    first_center = 0.5 * (first[..., :2] + first[..., 2:])
    last_center = 0.5 * (last[..., :2] + last[..., 2:])
    last_center_xy = torch.stack(
        [
            last_center[..., 0] * float(width),
            last_center[..., 1] * float(height),
        ],
        dim=-1,
    )
    first_center_xy = torch.stack(
        [
            first_center[..., 0] * float(width),
            first_center[..., 1] * float(height),
        ],
        dim=-1,
    )
    last_center_norm = torch.stack(
        [
            last_center_xy[..., 0] / max(float(width - 1), 1.0),
            last_center_xy[..., 1] / max(float(height - 1), 1.0),
        ],
        dim=-1,
    ).clamp(0.0, 1.0)
    delta_norm = torch.stack(
        [
            (last_center_xy[..., 0] - first_center_xy[..., 0]) / max(float(width - 1), 1.0),
            (last_center_xy[..., 1] - first_center_xy[..., 1]) / max(float(height - 1), 1.0),
        ],
        dim=-1,
    )
    summary = torch.cat([last_center_norm, delta_norm], dim=-1)
    summary_valid = any_valid
    return summary, summary_valid


def group_future_boxes(
    future_boxes: torch.Tensor,
    *,
    future_latent_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if future_boxes.ndim != 4:
        raise ValueError(f"future_boxes must have shape [B,T,O,4], got {list(future_boxes.shape)}")
    batch, future_frames, objects, _ = future_boxes.shape
    if future_frames % max(int(future_latent_frames), 1) != 0:
        raise ValueError(
            f"future_frames={future_frames} must be divisible by future_latent_frames={future_latent_frames}"
        )
    group = future_frames // int(future_latent_frames)
    grouped = future_boxes.view(batch, int(future_latent_frames), group, objects, 4)
    valid = ((grouped[..., 2] - grouped[..., 0]) > 1.0e-6) & ((grouped[..., 3] - grouped[..., 1]) > 1.0e-6)
    valid_perm = valid.bool().permute(0, 1, 3, 2)
    any_valid = valid_perm.any(dim=-1)
    last_idx = group - 1 - valid_perm.flip(dims=[-1]).float().argmax(dim=-1)
    last_idx = torch.where(any_valid, last_idx, torch.zeros_like(last_idx))
    grouped_perm = grouped.permute(0, 1, 3, 2, 4)
    gather_last = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, 4).long()
    grouped_last = torch.gather(grouped_perm, dim=3, index=gather_last).squeeze(3).contiguous()
    return grouped_last, any_valid
