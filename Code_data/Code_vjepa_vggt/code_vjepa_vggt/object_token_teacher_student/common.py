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
    # valid is [B,T,O]; reduce over the time axis (dim=1) to get a per-slot
    # "object ever visible in the future window" mask of shape [B,O].
    return valid.any(dim=1)


def last_valid_context_anchor(
    context_boxes: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an inference-available geometry anchor from context boxes.

    Returns the last valid context box per slot (and the matching track-summary
    center) so the Stage 2 future heads can predict displacement relative to the
    last observed object state instead of leaking the future GT box.

    Args:
        context_boxes: normalized xyxy boxes, shape [B, T_ctx, O, 4].
        image_hw: (H, W) used to match build_future_track_summary normalization.

    Returns:
        base_box_xyxy: [B, O, 4] last valid box (neutral box where no valid frame).
        base_track_summary: [B, O, 4] = [center_norm_x, center_norm_y, 0, 0].
        valid: [B, O] float mask, 1 where a valid context box exists.
    """
    if context_boxes.ndim != 4:
        raise ValueError(f"context_boxes must have shape [B,T,O,4], got {list(context_boxes.shape)}")
    height, width = int(image_hw[0]), int(image_hw[1])
    batch, frames, objects, _ = context_boxes.shape
    valid = ((context_boxes[..., 2] - context_boxes[..., 0]) > eps) & (
        (context_boxes[..., 3] - context_boxes[..., 1]) > eps
    )  # [B,T,O]
    valid_bot = valid.permute(0, 2, 1)  # [B,O,T]
    any_valid = valid_bot.any(dim=-1)  # [B,O]
    # last valid frame index per (b,o): T-1 - argmax over the time-reversed mask.
    last_idx = frames - 1 - valid_bot.flip(dims=[-1]).float().argmax(dim=-1)  # [B,O]
    last_idx = torch.where(any_valid, last_idx, torch.zeros_like(last_idx)).long()
    boxes_bot = context_boxes.permute(0, 2, 1, 3)  # [B,O,T,4]
    gather_idx = last_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 4)
    last_box = torch.gather(boxes_bot, dim=2, index=gather_idx).squeeze(2)  # [B,O,4]
    neutral = last_box.new_tensor([0.45, 0.45, 0.55, 0.55]).view(1, 1, 4)
    valid_f = any_valid.unsqueeze(-1).to(dtype=last_box.dtype)
    base_box_xyxy = last_box * valid_f + neutral * (1.0 - valid_f)
    center = 0.5 * (base_box_xyxy[..., :2] + base_box_xyxy[..., 2:])  # [B,O,2] in [0,1]
    center_norm = torch.stack(
        [
            (center[..., 0] * float(width)) / max(float(width - 1), 1.0),
            (center[..., 1] * float(height)) / max(float(height - 1), 1.0),
        ],
        dim=-1,
    ).clamp(0.0, 1.0)
    base_track_summary = torch.cat([center_norm, torch.zeros_like(center_norm)], dim=-1)  # [B,O,4]
    return base_box_xyxy, base_track_summary, any_valid.to(dtype=last_box.dtype)


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
