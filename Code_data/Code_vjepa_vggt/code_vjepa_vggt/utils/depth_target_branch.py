from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class DepthTargetComparison:
    state_depth_latent: torch.Tensor
    depth_anything_latent: torch.Tensor
    state_depth_framewise: torch.Tensor
    depth_anything_framewise: torch.Tensor
    depth_anything_maps_norm: torch.Tensor
    matched_boxes: torch.Tensor
    matched_boxes_valid: torch.Tensor
    frame_indices_for_latent: torch.Tensor


def group_last(values: torch.Tensor, latent_frames: int) -> torch.Tensor:
    group = int(values.shape[1]) // int(latent_frames)
    if int(values.shape[1]) % int(latent_frames) != 0:
        raise ValueError(
            f"context value frames {int(values.shape[1])} not divisible by latent_frames={latent_frames}"
        )
    new_shape = (values.shape[0], int(latent_frames), group) + tuple(values.shape[2:])
    return values.view(*new_shape)[:, :, -1]


def latent_last_frame_indices(num_frames: int, latent_frames: int) -> torch.Tensor:
    if int(num_frames) % int(latent_frames) != 0:
        raise ValueError(f"num_frames={num_frames} not divisible by latent_frames={latent_frames}")
    group = int(num_frames) // int(latent_frames)
    return torch.arange(int(latent_frames), dtype=torch.long) * group + (group - 1)


def gather_object_values(values: torch.Tensor, object_indices: torch.Tensor) -> torch.Tensor:
    gather_idx = object_indices[:, None, :, None].expand(-1, values.shape[1], -1, values.shape[-1])
    return torch.gather(values, 2, gather_idx)


def gather_object_boxes(boxes_xyxy: torch.Tensor, object_indices: torch.Tensor) -> torch.Tensor:
    gather_idx = object_indices[:, None, :, None].expand(-1, boxes_xyxy.shape[1], -1, boxes_xyxy.shape[-1])
    return torch.gather(boxes_xyxy, 2, gather_idx)


def normalize_depth_maps_percentile(depth_maps: torch.Tensor, q_low: float = 5.0, q_high: float = 95.0) -> torch.Tensor:
    if depth_maps.ndim != 4:
        raise ValueError(f"depth_maps must have shape [B,T,H,W], got {list(depth_maps.shape)}")
    outputs: list[torch.Tensor] = []
    for batch_idx in range(int(depth_maps.shape[0])):
        d = depth_maps[batch_idx].detach().float().cpu().numpy()
        finite = np.isfinite(d)
        if not finite.any():
            outputs.append(torch.zeros_like(depth_maps[batch_idx], dtype=torch.float32))
            continue
        vals = d[finite]
        lo = float(np.percentile(vals, q_low))
        hi = float(np.percentile(vals, q_high))
        if hi - lo < 1.0e-6:
            hi = lo + 1.0
        norm = np.clip((np.where(finite, d, lo) - lo) / (hi - lo + 1.0e-6), 0.0, 1.0)
        outputs.append(torch.from_numpy(norm).float())
    return torch.stack(outputs, dim=0).to(device=depth_maps.device)


def pool_depth_from_boxes_median(
    depth_maps_norm: torch.Tensor,
    boxes_xyxy_norm: torch.Tensor,
    boxes_valid: torch.Tensor,
) -> torch.Tensor:
    if depth_maps_norm.ndim != 4:
        raise ValueError(f"depth_maps_norm must have shape [B,T,H,W], got {list(depth_maps_norm.shape)}")
    if boxes_xyxy_norm.ndim != 4:
        raise ValueError(f"boxes_xyxy_norm must have shape [B,T,O,4], got {list(boxes_xyxy_norm.shape)}")
    bsz, num_frames, height, width = depth_maps_norm.shape
    _, box_frames, num_objects, _ = boxes_xyxy_norm.shape
    if box_frames != num_frames:
        raise ValueError(f"frame mismatch between depth_maps={num_frames} and boxes={box_frames}")
    out = torch.zeros((bsz, num_frames, num_objects, 1), dtype=torch.float32, device=depth_maps_norm.device)
    for b in range(bsz):
        for t in range(num_frames):
            depth_hw = depth_maps_norm[b, t].detach().float().cpu().numpy()
            for o in range(num_objects):
                if not bool(boxes_valid[b, t, o].item()):
                    continue
                x0, y0, x1, y1 = boxes_xyxy_norm[b, t, o].detach().float().cpu().tolist()
                px0 = int(np.floor(x0 * width))
                py0 = int(np.floor(y0 * height))
                px1 = int(np.ceil(x1 * width))
                py1 = int(np.ceil(y1 * height))
                px0 = max(0, min(px0, width - 1))
                py0 = max(0, min(py0, height - 1))
                px1 = max(px0 + 1, min(px1, width))
                py1 = max(py0 + 1, min(py1, height))
                roi = depth_hw[py0:py1, px0:px1]
                finite = np.isfinite(roi)
                if finite.any():
                    scalar = float(np.median(roi[finite]))
                else:
                    cx = int(round(0.5 * (px0 + px1 - 1)))
                    cy = int(round(0.5 * (py0 + py1 - 1)))
                    scalar = float(depth_hw[max(0, min(cy, height - 1)), max(0, min(cx, width - 1))])
                out[b, t, o, 0] = scalar
    return out


def scalar_depth_to_box_map(
    depth_scalars: torch.Tensor,
    boxes_xyxy_norm: torch.Tensor,
    boxes_valid: torch.Tensor,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    if depth_scalars.ndim != 4:
        raise ValueError(f"depth_scalars must have shape [B,T,O,1], got {list(depth_scalars.shape)}")
    height, width = int(image_hw[0]), int(image_hw[1])
    bsz, num_frames, num_objects, _ = depth_scalars.shape
    canvas = torch.zeros((bsz, num_frames, height, width), dtype=torch.float32)
    for b in range(bsz):
        for t in range(num_frames):
            for o in range(num_objects):
                if not bool(boxes_valid[b, t, o].item()):
                    continue
                x0, y0, x1, y1 = boxes_xyxy_norm[b, t, o].detach().float().cpu().tolist()
                px0 = int(np.floor(x0 * width))
                py0 = int(np.floor(y0 * height))
                px1 = int(np.ceil(x1 * width))
                py1 = int(np.ceil(y1 * height))
                px0 = max(0, min(px0, width - 1))
                py0 = max(0, min(py0, height - 1))
                px1 = max(px0 + 1, min(px1, width))
                py1 = max(py0 + 1, min(py1, height))
                canvas[b, t, py0:py1, px0:px1] = float(depth_scalars[b, t, o, 0].item())
    return canvas


def scalar_depth_map_to_rgb(depth_hw: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth_hw)
    if finite.any():
        vals = depth_hw[finite]
        lo = float(vals.min())
        hi = float(vals.max())
        if hi - lo < 1.0e-6:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    norm = np.clip((np.where(finite, depth_hw, lo) - lo) / (hi - lo + 1.0e-6), 0.0, 1.0)
    gray = (norm * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def build_depth_target_comparison(
    *,
    context_states: torch.Tensor,
    context_boxes: torch.Tensor,
    object_indices: torch.Tensor,
    depth_target_state_index: int,
    depth_maps_norm: torch.Tensor,
    latent_frames: int,
) -> DepthTargetComparison:
    matched_state_depth = gather_object_values(
        context_states[..., depth_target_state_index : depth_target_state_index + 1],
        object_indices,
    )
    matched_boxes = gather_object_boxes(context_boxes, object_indices)
    matched_boxes_valid = (
        ((matched_boxes[..., 2] - matched_boxes[..., 0]) > 1.0e-6)
        & ((matched_boxes[..., 3] - matched_boxes[..., 1]) > 1.0e-6)
    )
    state_depth_latent = group_last(matched_state_depth, latent_frames)
    depth_anything_framewise = pool_depth_from_boxes_median(depth_maps_norm, matched_boxes, matched_boxes_valid)
    depth_anything_latent = group_last(depth_anything_framewise, latent_frames)
    state_depth_framewise = matched_state_depth
    frame_indices = latent_last_frame_indices(int(context_states.shape[1]), int(latent_frames))
    return DepthTargetComparison(
        state_depth_latent=state_depth_latent,
        depth_anything_latent=depth_anything_latent,
        state_depth_framewise=state_depth_framewise,
        depth_anything_framewise=depth_anything_framewise,
        depth_anything_maps_norm=depth_maps_norm,
        matched_boxes=matched_boxes,
        matched_boxes_valid=matched_boxes_valid,
        frame_indices_for_latent=frame_indices,
    )
