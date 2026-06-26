from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.infer_v_newtrain_context_video_wan import (
    _load_v_newtrain_state_into_model,
    _resolve_checkpoint_file,
    _tensor_video_to_pil_list,
    build_model,
)
from code_vjepa_vggt.inspect_cotracker_vggt_geometry import (
    colorize_scalar_video,
    draw_box_rgb,
    draw_point_rgb,
)


BOX_GT_COLOR = (214, 40, 40)
BOX_PRED_COLOR = (42, 157, 143)
TRACK_GT_COLOR = (247, 127, 0)
TRACK_PRED_COLOR = (39, 125, 161)

DEFAULT_DATASET_ROOT = (
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/"
    "industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500"
)
DEFAULT_WAN_ROOT = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
DEFAULT_JEPA_CKPT = "/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth"
DEFAULT_COTRACKER_CKPT = "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def _build_contact_sheet(
    frames_thwc_uint8: np.ndarray,
    *,
    title: str,
    cols: int = 4,
    pad: int = 10,
) -> Image.Image:
    frames = np.asarray(frames_thwc_uint8, dtype=np.uint8)
    if frames.ndim != 4:
        raise ValueError(f"expected [T,H,W,C], got {frames.shape}")
    total, height, width, _ = frames.shape
    cols = max(1, min(int(cols), int(total)))
    rows = int(np.ceil(float(total) / float(cols)))
    title_h = 34
    label_h = 24
    canvas_w = cols * width + (cols + 1) * pad
    canvas_h = title_h + rows * (height + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(246, 244, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 8), title, fill=(25, 25, 25))
    for frame_idx in range(total):
        row = frame_idx // cols
        col = frame_idx % cols
        x0 = pad + col * (width + pad)
        y0 = title_h + pad + row * (height + label_h)
        frame = Image.fromarray(frames[frame_idx])
        canvas.paste(frame, (x0, y0))
        draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(180, 180, 180), width=1)
        draw.text((x0, y0 + height + 4), f"frame {frame_idx}", fill=(40, 40, 40))
    return canvas


def _write_contact_sheet(path: Path, frames_thwc_uint8: np.ndarray, *, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _build_contact_sheet(frames_thwc_uint8, title=title)
    image.save(path, format="PNG")
    return path


def _write_scalar_panel(
    path: Path,
    values: np.ndarray,
    *,
    title: str,
    cell_w: int = 220,
    row_h: int = 56,
    pad: int = 16,
) -> Path:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"expected [T,N], got {arr.shape}")
    rows, cols = arr.shape
    valid = np.isfinite(arr)
    if np.any(valid):
        lo = float(np.min(arr[valid]))
        hi = float(np.max(arr[valid]))
    else:
        lo, hi = 0.0, 1.0
    if hi - lo < 1.0e-6:
        hi = lo + 1.0
    title_h = 42
    label_w = 86
    canvas_w = label_w + cols * cell_w + (cols + 1) * pad
    canvas_h = title_h + rows * row_h + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(246, 244, 238))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 10), title, fill=(20, 20, 20))
    for r in range(rows):
        y0 = title_h + pad + r * row_h
        draw.text((pad, y0 + 16), f"frame {r}", fill=(40, 40, 40))
        for c in range(cols):
            x0 = label_w + pad + c * cell_w
            v = float(arr[r, c])
            norm = max(0.0, min(1.0, (v - lo) / (hi - lo)))
            color = tuple(int(x) for x in colorize_scalar_video(np.array([[[norm]]], dtype=np.float32))[0, 0, 0])
            draw.rounded_rectangle((x0, y0, x0 + cell_w - 12, y0 + row_h - 10), radius=8, fill=color, outline=(180, 180, 180))
            text_fill = (255, 255, 255) if norm > 0.45 else (20, 20, 20)
            draw.text((x0 + 10, y0 + 16), f"{v:.6f}", fill=text_fill)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return [Path(path).expanduser().resolve() for path in args.checkpoints]
    if not args.checkpoint_dir:
        raise ValueError("one of --checkpoints or --checkpoint-dir is required")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    step_dirs = sorted(path for path in checkpoint_dir.glob("step-*") if path.is_dir())
    if not step_dirs:
        raise FileNotFoundError(f"no step-* directories found under {checkpoint_dir}")
    return step_dirs


def _checkpoint_label(checkpoint_path: Path) -> str:
    resolved = _resolve_checkpoint_file(checkpoint_path)
    parent = resolved.parent.name
    if parent.startswith("step-"):
        return parent
    return resolved.stem


def _build_model_args(args: argparse.Namespace) -> Any:
    class _Args:
        pass

    model_args = _Args()
    model_args.device = args.device
    model_args.wan_root = args.wan_root
    model_args.lora_rank = int(args.lora_rank)
    model_args.context_frames = int(args.num_context_frames)
    model_args.disable_object_branch = False
    model_args.object_num_queries = int(args.object_num_queries)
    model_args.aux_max_objects = int(args.aux_max_objects)
    model_args.jepa_ckpt_path = args.jepa_ckpt_path
    model_args.jepa_input_size = int(args.jepa_input_size)
    model_args.jepa_patch_size = int(args.jepa_patch_size)
    model_args.jepa_tubelet_size = int(args.jepa_tubelet_size)
    model_args.cotracker_checkpoint = args.cotracker_checkpoint
    model_args.cotracker_input_h = int(args.cotracker_input_h)
    model_args.cotracker_input_w = int(args.cotracker_input_w)
    model_args.cotracker_window_len = int(args.cotracker_window_len)
    model_args.object_pooler_latent_dim = int(args.object_pooler_latent_dim)
    model_args.cond_proj_dim = int(args.cond_proj_dim)
    model_args.jepa_window_radius = int(args.jepa_window_radius)
    model_args.latent_window_radius = int(args.latent_window_radius)
    return model_args


def _norm_box_to_px(box_xyxy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    scale = np.array([width, height, width, height], dtype=np.float32)
    return np.asarray(box_xyxy, dtype=np.float32) * scale


def _summary_to_px(track_summary_xydxdy: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_hw
    center = np.array(
        [
            float(track_summary_xydxdy[0]) * max(width - 1, 1),
            float(track_summary_xydxdy[1]) * max(height - 1, 1),
        ],
        dtype=np.float32,
    )
    delta = np.array(
        [
            float(track_summary_xydxdy[2]) * max(width - 1, 1),
            float(track_summary_xydxdy[3]) * max(height - 1, 1),
        ],
        dtype=np.float32,
    )
    start = center - delta
    return center, start


def _xy_to_px(xy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    return np.array(
        [
            float(xy[0]) * max(width - 1, 1),
            float(xy[1]) * max(height - 1, 1),
        ],
        dtype=np.float32,
    )


def _decode_summary_background_frames(
    pipe,
    clean_prefix_latents: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        pipe.load_models_to_device(["vae"])
        decoded = pipe.vae.decode_framewise(
            clean_prefix_latents.to(device=pipe.device, dtype=pipe.torch_dtype),
            device=pipe.device,
        )
        pipe.load_models_to_device([])
    return np.stack(
        [
            tensor_frame_to_uint8_hwc(decoded[0, :, frame_idx])
            for frame_idx in range(int(decoded.shape[2]))
        ],
        axis=0,
    )


def _render_box_overlay(
    background_frames: np.ndarray,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
    pred_box_xyxy: np.ndarray,
    pred_box_valid: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = min(int(background_frames.shape[0]), int(gt_box_xyxy.shape[0]), int(pred_box_xyxy.shape[0]))
    for latent_idx in range(latent_frames):
        frame = np.asarray(background_frames[latent_idx], dtype=np.uint8).copy()
        for obj_idx in range(gt_box_xyxy.shape[1]):
            if bool(gt_box_valid[latent_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(gt_box_xyxy[latent_idx, obj_idx], image_hw),
                    BOX_GT_COLOR,
                    f"gt{obj_idx}",
                )
            if bool(pred_box_valid[latent_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(pred_box_xyxy[latent_idx, obj_idx], image_hw),
                    BOX_PRED_COLOR,
                    f"pred{obj_idx}",
                )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_native_box_overlay(
    context_video: torch.Tensor,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
    pred_box_xyxy: np.ndarray,
    pred_box_valid: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    source_frames = min(
        int(context_video.shape[1]),
        int(gt_box_xyxy.shape[0]),
        int(pred_box_xyxy.shape[0]),
    )
    for frame_idx in range(source_frames):
        frame = tensor_frame_to_uint8_hwc(context_video[:, frame_idx]).copy()
        for obj_idx in range(gt_box_xyxy.shape[1]):
            if bool(gt_box_valid[frame_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(gt_box_xyxy[frame_idx, obj_idx], image_hw),
                    BOX_GT_COLOR,
                    f"gt{obj_idx}",
                )
            if bool(pred_box_valid[frame_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(pred_box_xyxy[frame_idx, obj_idx], image_hw),
                    BOX_PRED_COLOR,
                    f"pred{obj_idx}",
                )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _tracks_to_radius_boxes(
    tracks_xy: np.ndarray,
    *,
    image_hw: tuple[int, int],
    radius_px: float = 12.0,
) -> np.ndarray:
    height, width = image_hw
    boxes = np.zeros((tracks_xy.shape[0], tracks_xy.shape[1], 4), dtype=np.float32)
    boxes[..., 0] = (tracks_xy[..., 0] - float(radius_px)) / max(float(width), 1.0)
    boxes[..., 1] = (tracks_xy[..., 1] - float(radius_px)) / max(float(height), 1.0)
    boxes[..., 2] = (tracks_xy[..., 0] + float(radius_px)) / max(float(width), 1.0)
    boxes[..., 3] = (tracks_xy[..., 1] + float(radius_px)) / max(float(height), 1.0)
    return np.clip(boxes, 0.0, 1.0)


def _render_track_overlay(
    background_frames: np.ndarray,
    gt_track_summary: np.ndarray,
    gt_track_valid: np.ndarray,
    pred_track_summary: np.ndarray,
    pred_track_valid: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = min(int(background_frames.shape[0]), int(gt_track_summary.shape[0]), int(pred_track_summary.shape[0]))
    for latent_idx in range(latent_frames):
        frame = np.asarray(background_frames[latent_idx], dtype=np.uint8).copy()
        for obj_idx in range(gt_track_summary.shape[1]):
            if bool(pred_track_valid[latent_idx, obj_idx]):
                pred_center, pred_start = _summary_to_px(pred_track_summary[latent_idx, obj_idx], image_hw)
                draw_point_rgb(frame, pred_center, TRACK_PRED_COLOR, f"pred{obj_idx}", radius=5)
                draw_point_rgb(frame, pred_start, TRACK_PRED_COLOR, f"s{obj_idx}", radius=3)
            if bool(gt_track_valid[latent_idx, obj_idx]):
                gt_center, gt_start = _summary_to_px(gt_track_summary[latent_idx, obj_idx], image_hw)
                draw_point_rgb(frame, gt_center, TRACK_GT_COLOR, f"gt{obj_idx}", radius=5)
                draw_point_rgb(frame, gt_start, TRACK_GT_COLOR, f"gs{obj_idx}", radius=3)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_native_track_overlay(
    context_video: torch.Tensor,
    gt_centers_xy: np.ndarray,
    gt_track_valid: np.ndarray,
    pred_centers_xy: np.ndarray,
    pred_track_valid: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    source_frames = min(
        int(context_video.shape[1]),
        int(gt_centers_xy.shape[0]),
        int(pred_centers_xy.shape[0]),
    )
    for frame_idx in range(source_frames):
        frame = tensor_frame_to_uint8_hwc(context_video[:, frame_idx]).copy()
        for obj_idx in range(gt_centers_xy.shape[1]):
            if bool(pred_track_valid[frame_idx, obj_idx]):
                draw_point_rgb(
                    frame,
                    _xy_to_px(pred_centers_xy[frame_idx, obj_idx], image_hw),
                    TRACK_PRED_COLOR,
                    f"pred{obj_idx}",
                    radius=5,
                )
            if bool(gt_track_valid[frame_idx, obj_idx]):
                draw_point_rgb(
                    frame,
                    _xy_to_px(gt_centers_xy[frame_idx, obj_idx], image_hw),
                    TRACK_GT_COLOR,
                    f"gt{obj_idx}",
                    radius=5,
                )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_object_context_heatmap(
    object_context: torch.Tensor,
    object_valid_mask: np.ndarray,
    *,
    latent_frames: int,
    num_slots: int,
    width_repeat: int = 24,
) -> np.ndarray:
    if object_context.ndim != 3:
        raise ValueError(f"object_context must have shape [B, T*O, D], got {list(object_context.shape)}")
    tokens = object_context[0].detach().float().cpu()
    if int(tokens.shape[0]) != int(latent_frames) * int(num_slots):
        raise ValueError(
            f"object_context token count mismatch: got {int(tokens.shape[0])}, expected {int(latent_frames) * int(num_slots)}"
        )
    values = tokens.abs().mean(dim=-1).view(int(latent_frames), int(num_slots)).numpy()
    if object_valid_mask.shape[-1] == num_slots:
        values = values * object_valid_mask.astype(np.float32)[None, :]
    values = np.repeat(values[..., None], max(1, int(width_repeat)), axis=2)
    heat = colorize_scalar_video(values.astype(np.float32))
    return heat


def _render_matrix_heatmap(
    values_2d: np.ndarray,
    *,
    title: str,
    width_repeat: int = 24,
) -> np.ndarray:
    values = np.asarray(values_2d, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"expected [T,N], got {values.shape}")
    repeated = np.repeat(values[..., None], max(1, int(width_repeat)), axis=2)
    heat = colorize_scalar_video(repeated.astype(np.float32))
    return heat


def _annotate_video_with_lines(
    frames: np.ndarray,
    *,
    title: str,
    lines_per_frame: list[list[str]],
) -> np.ndarray:
    if len(frames) != len(lines_per_frame):
        raise ValueError(f"frame count mismatch: frames={len(frames)}, lines={len(lines_per_frame)}")
    out: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8).copy())
        draw = ImageDraw.Draw(image)
        width, height = image.size
        panel_h = max(20, 16 * (len(lines_per_frame[idx]) + 1))
        draw.rectangle((4, 4, min(width - 4, 520), min(height - 4, 4 + panel_h)), fill=(0, 0, 0))
        draw.text((10, 8), title, fill=(255, 255, 255))
        for line_idx, line in enumerate(lines_per_frame[idx]):
            draw.text((10, 24 + 14 * line_idx), line, fill=(255, 255, 255))
        out.append(np.array(image))
    return np.stack(out, axis=0)


def _compute_framewise_box_losses(
    *,
    pred_box_xyxy: np.ndarray,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pred_box_xyxy.shape != gt_box_xyxy.shape:
        raise ValueError(f"box shape mismatch: pred={pred_box_xyxy.shape}, gt={gt_box_xyxy.shape}")
    T, O = pred_box_xyxy.shape[:2]
    center_l1 = np.zeros((T, 1), dtype=np.float32)
    wh_l1 = np.zeros((T, 1), dtype=np.float32)
    iou_loss = np.zeros((T, 1), dtype=np.float32)
    for t in range(T):
        valid = gt_box_valid[t]
        if not np.any(valid):
            continue
        pred = pred_box_xyxy[t, valid]
        gt = gt_box_xyxy[t, valid]
        pred_center = 0.5 * (pred[..., :2] + pred[..., 2:])
        gt_center = 0.5 * (gt[..., :2] + gt[..., 2:])
        pred_wh = np.clip(pred[..., 2:] - pred[..., :2], 1.0e-4, None)
        gt_wh = np.clip(gt[..., 2:] - gt[..., :2], 1.0e-4, None)
        center_l1[t, 0] = float(np.abs(pred_center - gt_center).mean())
        wh_l1[t, 0] = float(np.abs(pred_wh - gt_wh).mean())
        inter_x0 = np.maximum(pred[..., 0], gt[..., 0])
        inter_y0 = np.maximum(pred[..., 1], gt[..., 1])
        inter_x1 = np.minimum(pred[..., 2], gt[..., 2])
        inter_y1 = np.minimum(pred[..., 3], gt[..., 3])
        inter_w = np.clip(inter_x1 - inter_x0, 0.0, None)
        inter_h = np.clip(inter_y1 - inter_y0, 0.0, None)
        inter = inter_w * inter_h
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        union = np.clip(pred_area + gt_area - inter, 1.0e-6, None)
        iou_loss[t, 0] = float((1.0 - inter / union).mean())
    return center_l1, wh_l1, iou_loss


def _compute_framewise_track_losses(
    *,
    pred_tracks_native: np.ndarray,
    gt_tracks_native: np.ndarray,
    gt_tracks_valid: np.ndarray,
    gt_boxes_native: np.ndarray,
    gt_boxes_valid: np.ndarray,
    image_hw: tuple[int, int],
    radius_px: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    if pred_tracks_native.shape != gt_tracks_native.shape:
        raise ValueError(f"track shape mismatch: pred={pred_tracks_native.shape}, gt={gt_tracks_native.shape}")
    pred_boxes = _tracks_to_radius_boxes(pred_tracks_native, image_hw=image_hw, radius_px=radius_px)
    track_l1 = np.zeros((pred_tracks_native.shape[0], 1), dtype=np.float32)
    track_iou = np.zeros((pred_tracks_native.shape[0], 1), dtype=np.float32)
    for t in range(pred_tracks_native.shape[0]):
        valid = gt_tracks_valid[t] & gt_boxes_valid[t]
        if valid.any():
            track_l1[t, 0] = float(np.abs(pred_tracks_native[t] - gt_tracks_native[t])[valid].mean())
            pred_box = pred_boxes[t, valid]
            gt_box = gt_boxes_native[t, valid]
            inter_x0 = np.maximum(pred_box[..., 0], gt_box[..., 0])
            inter_y0 = np.maximum(pred_box[..., 1], gt_box[..., 1])
            inter_x1 = np.minimum(pred_box[..., 2], gt_box[..., 2])
            inter_y1 = np.minimum(pred_box[..., 3], gt_box[..., 3])
            inter_w = np.clip(inter_x1 - inter_x0, 0.0, None)
            inter_h = np.clip(inter_y1 - inter_y0, 0.0, None)
            inter = inter_w * inter_h
            pred_area = np.clip(pred_box[..., 2] - pred_box[..., 0], 0.0, None) * np.clip(
                pred_box[..., 3] - pred_box[..., 1], 0.0, None
            )
            gt_area = np.clip(gt_box[..., 2] - gt_box[..., 0], 0.0, None) * np.clip(
                gt_box[..., 3] - gt_box[..., 1], 0.0, None
            )
            union = np.clip(pred_area + gt_area - inter, 1.0e-6, None)
            track_iou[t, 0] = float((1.0 - (inter / union)).mean())
    return track_l1, track_iou


def _render_depth_panel(
    gt_depth: np.ndarray,
    gt_depth_valid: np.ndarray,
    pred_depth: np.ndarray,
) -> np.ndarray:
    valid_values = []
    if gt_depth_valid.any():
        valid_values.append(gt_depth[gt_depth_valid])
        valid_values.append(pred_depth[gt_depth_valid])
    if valid_values:
        concat = np.concatenate(valid_values, axis=0)
        lo = float(np.min(concat))
        hi = float(np.max(concat))
        if hi - lo < 1.0e-6:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    gt_map = np.where(gt_depth_valid, gt_depth, lo)
    pred_map = np.where(gt_depth_valid, pred_depth, lo)
    gt_vis = colorize_scalar_video(((gt_map - lo) / (hi - lo + 1.0e-6)).astype(np.float32))
    pred_vis = colorize_scalar_video(((pred_map - lo) / (hi - lo + 1.0e-6)).astype(np.float32))
    frames: list[np.ndarray] = []
    for t in range(gt_vis.shape[0]):
        left = gt_vis[t]
        right = pred_vis[t]
        pad = np.full((left.shape[0], 24, 3), 245, dtype=np.uint8)
        panel = np.concatenate([left, pad, right], axis=1)
        frames.append(panel)
    return np.stack(frames, axis=0)


def _render_depth_overlay(
    background_frames: np.ndarray,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
    pred_box_xyxy: np.ndarray,
    pred_box_valid: np.ndarray,
    gt_depth: np.ndarray,
    gt_depth_valid: np.ndarray,
    pred_depth: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    total_frames = min(
        int(background_frames.shape[0]),
        int(gt_box_xyxy.shape[0]),
        int(pred_box_xyxy.shape[0]),
        int(gt_depth.shape[0]),
        int(pred_depth.shape[0]),
    )
    for frame_idx in range(total_frames):
        frame = np.asarray(background_frames[frame_idx], dtype=np.uint8).copy()
        for obj_idx in range(gt_box_xyxy.shape[1]):
            if bool(gt_box_valid[frame_idx, obj_idx]):
                gt_box_px = _norm_box_to_px(gt_box_xyxy[frame_idx, obj_idx], image_hw)
                draw_box_rgb(frame, gt_box_px, BOX_GT_COLOR, f"gt{obj_idx}")
            if bool(pred_box_valid[frame_idx, obj_idx]):
                pred_box_px = _norm_box_to_px(pred_box_xyxy[frame_idx, obj_idx], image_hw)
                draw_box_rgb(frame, pred_box_px, BOX_PRED_COLOR, f"pred{obj_idx}")
            if bool(gt_depth_valid[frame_idx, obj_idx]):
                label_xy = _norm_box_to_px(gt_box_xyxy[frame_idx, obj_idx], image_hw)[:2]
                draw_point_rgb(frame, label_xy, TRACK_GT_COLOR, f"dgt={float(gt_depth[frame_idx, obj_idx]):.3f}", radius=3)
                if bool(pred_box_valid[frame_idx, obj_idx]):
                    pred_xy = _norm_box_to_px(pred_box_xyxy[frame_idx, obj_idx], image_hw)[:2]
                    draw_point_rgb(frame, pred_xy, TRACK_PRED_COLOR, f"dp={float(pred_depth[frame_idx, obj_idx]):.3f}", radius=3)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _overlay_metric_lines(
    frames: np.ndarray,
    *,
    title: str,
    per_frame_metrics: list[list[str]],
) -> np.ndarray:
    return _annotate_video_with_lines(
        frames,
        title=title,
        lines_per_frame=per_frame_metrics,
    )


def _compute_aux_metrics(
    *,
    pred_track_summary: torch.Tensor,
    gt_track_summary: torch.Tensor,
    gt_track_valid: torch.Tensor,
    pred_box_xyxy: torch.Tensor,
    gt_box_xyxy: torch.Tensor,
    gt_box_valid: torch.Tensor,
    pred_depth: torch.Tensor | None,
    gt_depth: torch.Tensor | None,
    gt_depth_valid: torch.Tensor | None,
    track_box_loss: torch.Tensor,
    track_iou_loss: torch.Tensor,
) -> dict[str, float]:
    track_aux_loss = (((pred_track_summary - gt_track_summary).abs()) * gt_track_valid.unsqueeze(-1)).sum()
    track_aux_loss = track_aux_loss / (
        gt_track_valid.unsqueeze(-1).sum().clamp_min(1.0) * pred_track_summary.shape[-1]
    )
    box_weights = gt_box_valid.unsqueeze(-1).to(dtype=pred_box_xyxy.dtype, device=pred_box_xyxy.device)
    box_denom = gt_box_valid.sum().clamp_min(1.0)
    pred_center = 0.5 * (pred_box_xyxy[..., :2] + pred_box_xyxy[..., 2:])
    gt_center = 0.5 * (gt_box_xyxy[..., :2] + gt_box_xyxy[..., 2:])
    pred_wh = (pred_box_xyxy[..., 2:] - pred_box_xyxy[..., :2]).clamp_min(1.0e-4)
    gt_wh = (gt_box_xyxy[..., 2:] - gt_box_xyxy[..., :2]).clamp_min(1.0e-4)
    center_l1 = (((pred_center - gt_center).abs()) * box_weights[..., :2]).sum() / (box_denom * 2.0)
    wh_l1 = (((pred_wh - gt_wh).abs()) * box_weights[..., :2]).sum() / (box_denom * 2.0)
    inter_x0 = torch.maximum(pred_box_xyxy[..., 0], gt_box_xyxy[..., 0])
    inter_y0 = torch.maximum(pred_box_xyxy[..., 1], gt_box_xyxy[..., 1])
    inter_x1 = torch.minimum(pred_box_xyxy[..., 2], gt_box_xyxy[..., 2])
    inter_y1 = torch.minimum(pred_box_xyxy[..., 3], gt_box_xyxy[..., 3])
    inter_w = (inter_x1 - inter_x0).clamp_min(0.0)
    inter_h = (inter_y1 - inter_y0).clamp_min(0.0)
    inter = inter_w * inter_h
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    gt_area = gt_wh[..., 0] * gt_wh[..., 1]
    union = (pred_area + gt_area - inter).clamp_min(1.0e-6)
    iou = inter / union
    iou_loss = ((1.0 - iou) * gt_box_valid.to(dtype=iou.dtype, device=iou.device)).sum() / box_denom
    box_aux_loss = center_l1 + 0.5 * wh_l1 + 0.5 * iou_loss
    depth_aux_loss = pred_track_summary.new_zeros(())
    if pred_depth is not None and gt_depth is not None and gt_depth_valid is not None:
        depth_pred = pred_depth
        depth_gt = gt_depth
        depth_valid = gt_depth_valid
        if depth_valid.ndim == depth_pred.ndim - 1:
            depth_valid = depth_valid.unsqueeze(-1)
        depth_aux_loss = (((depth_pred - depth_gt).abs()) * depth_valid).sum()
        depth_aux_loss = depth_aux_loss / (
            depth_valid.sum().clamp_min(1.0) * max(int(depth_pred.shape[-1]), 1)
        )
    return {
        "train/loss_main": float("nan"),
        "train/loss_track_aux": float(track_aux_loss.detach().item()),
        "train/loss_box_aux": float(box_aux_loss.detach().item()),
        "train/loss_depth_aux": float(depth_aux_loss.detach().item()),
        "train/track_box_loss": float(track_box_loss.detach().item()),
        "train/track_iou_loss": float(track_iou_loss.detach().item()),
    }


def _compute_regularizer_metrics(
    *,
    object_context: torch.Tensor,
    object_latent_tokens: torch.Tensor,
    track_delta: torch.Tensor,
    box_center_delta: torch.Tensor,
    box_log_scale: torch.Tensor,
) -> dict[str, float]:
    return {
        "train/loss_object_context_reg": float(object_context.detach().square().mean().item()),
        "train/loss_track_anchor_reg": float(track_delta.detach().abs().mean().item()),
        "train/loss_box_anchor_reg": float((box_center_delta.detach().abs().mean() + box_log_scale.detach().abs().mean()).item()),
        "train/object_context_abs_max": float(object_context.detach().abs().max().item()),
        "train/object_context_abs_mean": float(object_context.detach().abs().mean().item()),
        "train/object_latent_tokens_abs_max": float(object_latent_tokens.detach().abs().max().item()),
    }


def _render_scalar_strip(
    values: np.ndarray,
    *,
    title: str,
    labels: list[str] | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"scalar strip expects [T,N], got {arr.shape}")
    frames = colorize_scalar_video(arr)
    out: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        canvas_img = Image.fromarray(frame.copy())
        draw = ImageDraw.Draw(canvas_img)
        draw.text((8, 8), title, fill=(255, 255, 255))
        draw.text((8, 24), f"frame {idx}", fill=(255, 255, 255))
        if labels is not None:
            for col_idx, label in enumerate(labels):
                if col_idx >= arr.shape[1]:
                    break
                draw.text((8 + col_idx * 120, canvas_img.size[1] - 18), str(label), fill=(255, 255, 255))
        out.append(np.array(canvas_img))
    return np.stack(out, axis=0)


def _compute_framewise_track_summary_losses(
    *,
    pred_track_summary: np.ndarray,
    gt_track_summary: np.ndarray,
    gt_track_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pred_track_summary.shape != gt_track_summary.shape:
        raise ValueError(
            f"track summary shape mismatch: pred={pred_track_summary.shape}, gt={gt_track_summary.shape}"
        )
    T = pred_track_summary.shape[0]
    center_l1 = np.zeros((T, 1), dtype=np.float32)
    delta_l1 = np.zeros((T, 1), dtype=np.float32)
    total = np.zeros((T, 1), dtype=np.float32)
    for t in range(T):
        valid = gt_track_valid[t]
        if not np.any(valid):
            continue
        pred = pred_track_summary[t, valid]
        gt = gt_track_summary[t, valid]
        center_l1[t, 0] = float(np.abs(pred[..., :2] - gt[..., :2]).mean())
        delta_l1[t, 0] = float(np.abs(pred[..., 2:4] - gt[..., 2:4]).mean())
        total[t, 0] = float(center_l1[t, 0] + 0.25 * delta_l1[t, 0])
    return center_l1, delta_l1, total


def _compute_framewise_box_summary_losses(
    *,
    pred_box_xyxy: np.ndarray,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pred_box_xyxy.shape != gt_box_xyxy.shape:
        raise ValueError(f"box shape mismatch: pred={pred_box_xyxy.shape}, gt={gt_box_xyxy.shape}")
    T = pred_box_xyxy.shape[0]
    center_l1 = np.zeros((T, 1), dtype=np.float32)
    wh_l1 = np.zeros((T, 1), dtype=np.float32)
    iou_loss = np.zeros((T, 1), dtype=np.float32)
    total = np.zeros((T, 1), dtype=np.float32)
    for t in range(T):
        valid = gt_box_valid[t]
        if not np.any(valid):
            continue
        pred = pred_box_xyxy[t, valid]
        gt = gt_box_xyxy[t, valid]
        pred_center = 0.5 * (pred[..., :2] + pred[..., 2:])
        gt_center = 0.5 * (gt[..., :2] + gt[..., 2:])
        pred_wh = np.clip(pred[..., 2:] - pred[..., :2], 1.0e-4, None)
        gt_wh = np.clip(gt[..., 2:] - gt[..., :2], 1.0e-4, None)
        center_l1[t, 0] = float(np.abs(pred_center - gt_center).mean())
        wh_l1[t, 0] = float(np.abs(pred_wh - gt_wh).mean())
        inter_x0 = np.maximum(pred[..., 0], gt[..., 0])
        inter_y0 = np.maximum(pred[..., 1], gt[..., 1])
        inter_x1 = np.minimum(pred[..., 2], gt[..., 2])
        inter_y1 = np.minimum(pred[..., 3], gt[..., 3])
        inter_w = np.clip(inter_x1 - inter_x0, 0.0, None)
        inter_h = np.clip(inter_y1 - inter_y0, 0.0, None)
        inter = inter_w * inter_h
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        gt_area = gt_wh[..., 0] * gt_wh[..., 1]
        union = np.clip(pred_area + gt_area - inter, 1.0e-6, None)
        iou_loss[t, 0] = float((1.0 - inter / union).mean())
        total[t, 0] = float(center_l1[t, 0] + 0.5 * wh_l1[t, 0] + 0.5 * iou_loss[t, 0])
    return center_l1, wh_l1, iou_loss, total


def _compute_framewise_depth_losses(
    *,
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    gt_depth_valid: np.ndarray,
) -> np.ndarray:
    if pred_depth.shape != gt_depth.shape:
        raise ValueError(f"depth shape mismatch: pred={pred_depth.shape}, gt={gt_depth.shape}")
    T = pred_depth.shape[0]
    depth_l1 = np.zeros((T, 1), dtype=np.float32)
    for t in range(T):
        valid = gt_depth_valid[t]
        if not np.any(valid):
            continue
        depth_l1[t, 0] = float(np.abs(pred_depth[t] - gt_depth[t])[valid].mean())
    return depth_l1


def _compute_box_decomposition_metrics(
    *,
    active_box_xyxy: torch.Tensor,
    pred_box_xyxy: torch.Tensor,
    gt_box_xyxy: torch.Tensor,
    gt_box_valid: torch.Tensor,
) -> dict[str, float]:
    weights = gt_box_valid.unsqueeze(-1).to(dtype=pred_box_xyxy.dtype, device=pred_box_xyxy.device)
    denom = gt_box_valid.sum().clamp_min(1.0)

    active_center = 0.5 * (active_box_xyxy[..., :2] + active_box_xyxy[..., 2:])
    pred_center = 0.5 * (pred_box_xyxy[..., :2] + pred_box_xyxy[..., 2:])
    gt_center = 0.5 * (gt_box_xyxy[..., :2] + gt_box_xyxy[..., 2:])

    active_wh = (active_box_xyxy[..., 2:] - active_box_xyxy[..., :2]).clamp_min(1.0e-4)
    pred_wh = (pred_box_xyxy[..., 2:] - pred_box_xyxy[..., :2]).clamp_min(1.0e-4)
    gt_wh = (gt_box_xyxy[..., 2:] - gt_box_xyxy[..., :2]).clamp_min(1.0e-4)

    active_center_l1 = (((active_center - gt_center).abs()) * weights[..., :2]).sum() / (denom * 2.0)
    active_wh_l1 = (((active_wh - gt_wh).abs()) * weights[..., :2]).sum() / (denom * 2.0)
    pred_center_l1 = (((pred_center - gt_center).abs()) * weights[..., :2]).sum() / (denom * 2.0)
    pred_wh_l1 = (((pred_wh - gt_wh).abs()) * weights[..., :2]).sum() / (denom * 2.0)
    pred_vs_active_center_l1 = (((pred_center - active_center).abs()) * weights[..., :2]).sum() / (denom * 2.0)
    pred_vs_active_wh_l1 = (((pred_wh - active_wh).abs()) * weights[..., :2]).sum() / (denom * 2.0)

    return {
        "train/active_box_center_l1": float(active_center_l1.detach().item()),
        "train/active_box_wh_l1": float(active_wh_l1.detach().item()),
        "train/pred_box_center_l1": float(pred_center_l1.detach().item()),
        "train/pred_box_wh_l1": float(pred_wh_l1.detach().item()),
        "train/pred_vs_active_box_center_l1": float(pred_vs_active_center_l1.detach().item()),
        "train/pred_vs_active_box_wh_l1": float(pred_vs_active_wh_l1.detach().item()),
    }


def _run_case_for_checkpoint(
    *,
    model,
    sample: dict[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    checkpoint_label: str,
    sample_index: int,
    fps: int,
    export_aux_visuals: bool = True,
) -> dict[str, Any]:
    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video_single = sample["context_video"].to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video_single.shape[-2]), int(context_video_single.shape[-1]))
    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = model._build_object_query_priors(sample, image_hw=image_hw)
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = ((context_video_single.unsqueeze(0).permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    with torch.no_grad():
        cotracker_out = model.cotracker_adapter(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )
        tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
            cotracker_out.tracks,
            cotracker_out.visibility,
            cotracker_out.confidence,
            max_objects=model.aux_max_objects,
            points_per_object=model.object_num_queries,
        )
        jepa_dtype = next(model.jepa_adapter.parameters()).dtype
        jepa_out = model.jepa_adapter(context_video_single.unsqueeze(0).to(dtype=jepa_dtype))
        preprocessed_context_video = pipe.preprocess_video(_tensor_video_to_pil_list(context_video_single))
        clean_prefix_latents = pipe.vae.encode(
            preprocessed_context_video,
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        ).to(dtype=pipe.torch_dtype, device=device)
        object_out = model.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=clean_prefix_latents,
            tracks=tracks_grouped,
            visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=box_prior_xyxy,
        frame_valid_mask=None,
    )
        object_aux_out = model.object_aux_heads(
            object_out.object_latent_tokens,
            object_out.active_track_summary,
            object_out.active_box_xyxy,
        )
        object_context = model.object_adapter(
            object_out.object_latent_tokens,
            object_valid_mask=object_valid_mask,
        )

        gt_boxes = sample["context_boxes"].unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
        center_tracks_native, center_track_valid = model._object_center_tracks_from_grouped(
            tracks_grouped,
            visibility_grouped,
            confidence_grouped,
            object_valid_mask=object_valid_mask,
        )
        from code_vjepa_vggt.utils.track_supervision import (
            align_tracks_to_boxes,
            track_box_iou_loss,
            track_box_l1_loss,
        )

        track_alignment = align_tracks_to_boxes(
            tracks=center_tracks_native,
            gt_boxes=gt_boxes,
            image_hw=image_hw,
        )
        track_box_loss = track_box_l1_loss(
            tracks=center_tracks_native,
            matched_gt_centers=track_alignment.matched_gt_centers,
            matched_gt_valid=track_alignment.matched_gt_valid
            * center_track_valid.to(dtype=track_alignment.matched_gt_valid.dtype),
        )
        track_iou_loss = track_box_iou_loss(
            tracks=center_tracks_native,
            gt_boxes=gt_boxes,
            matched_gt_indices=track_alignment.matched_gt_indices,
            image_hw=image_hw,
            radius_px=12.0,
        )
        native_pred_box_xyxy = model.object_pooler._boxes_from_tracks(
            tracks_grouped,
            visibility_grouped,
            confidence_grouped,
            image_hw=image_hw,
            target_frames=None,
            box_prior_xyxy=box_prior_xyxy,
        )
        latent_frames = int(object_out.object_latent_tokens.shape[1])
        gt_valid_full = (track_alignment.matched_gt_valid > 0.5) & center_track_valid
        gt_track_summary, gt_track_valid = model._group_track_summary(
            track_alignment.matched_gt_centers,
            gt_valid_full,
            image_hw=image_hw,
            latent_frames=latent_frames,
        )
        matched_gt_boxes = model._gather_matched_gt_features(gt_boxes, track_alignment.matched_gt_indices)
        matched_gt_box_valid = ((matched_gt_boxes[..., 2] - matched_gt_boxes[..., 0]) > 1.0e-6) & (
            (matched_gt_boxes[..., 3] - matched_gt_boxes[..., 1]) > 1.0e-6
        )
        gt_box_xyxy, gt_box_valid = model._group_box_targets(
            matched_gt_boxes,
            matched_gt_box_valid,
            latent_frames,
        )

        gt_depth = None
        gt_depth_valid = None
        pred_depth = None
        if model.depth_target_state_index is not None and model.lambda_depth_aux > 0.0:
            gt_states = sample["context_states"].unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
            matched_gt_depth = model._gather_matched_gt_features(
                gt_states[..., model.depth_target_state_index : model.depth_target_state_index + 1],
                track_alignment.matched_gt_indices,
            )
            gt_depth = model._group_last(matched_gt_depth, latent_frames)
            pred_depth = object_aux_out.pred_depth
            gt_depth_valid = gt_box_valid.unsqueeze(-1)
        summary_background = _decode_summary_background_frames(pipe, clean_prefix_latents)

    case_stem = f"case_{sample_index:05d}__{checkpoint_label}"
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def _export_sheet(
        frames: np.ndarray,
        *,
        stem: str,
        title: str,
    ) -> str:
        sheet_path = _write_contact_sheet(
            assets_dir / f"{case_stem}__{stem}_sheet.png",
            frames,
            title=title,
        )
        return str(sheet_path.relative_to(output_dir))

    gt_track_summary_np = gt_track_summary[0].detach().float().cpu().numpy()
    gt_track_valid_np = gt_track_valid[0].detach().cpu().numpy() > 0.5
    gt_box_xyxy_np = gt_box_xyxy[0].detach().float().cpu().numpy()
    gt_box_valid_np = gt_box_valid[0].detach().cpu().numpy() > 0.5
    pred_track_summary_np = object_aux_out.pred_track_summary[0].detach().float().cpu().numpy()
    pred_box_xyxy_np = object_aux_out.pred_box_xyxy[0].detach().float().cpu().numpy()
    pred_valid_np = object_valid_mask[0].detach().float().cpu().numpy() > 0.5
    pred_track_valid_np = np.broadcast_to(pred_valid_np[None, :], (pred_track_summary_np.shape[0], pred_track_summary_np.shape[1]))
    pred_box_valid_np = np.broadcast_to(pred_valid_np[None, :], (pred_box_xyxy_np.shape[0], pred_box_xyxy_np.shape[1]))
    gt_track_native_np = track_alignment.matched_gt_centers[0].detach().float().cpu().numpy()
    gt_track_native_valid_np = gt_valid_full[0].detach().cpu().numpy() > 0.5
    pred_track_native_np = center_tracks_native[0].detach().float().cpu().numpy()
    pred_track_native_valid_np = center_track_valid[0].detach().cpu().numpy() > 0.5
    gt_box_native_np = matched_gt_boxes[0].detach().float().cpu().numpy()
    gt_box_native_valid_np = matched_gt_box_valid[0].detach().cpu().numpy() > 0.5
    pred_box_native_np = native_pred_box_xyxy[0].detach().float().cpu().numpy()
    pred_box_native_valid_np = (
        ((visibility_grouped * confidence_grouped).clamp_min(0.0) > 1.0e-6).any(dim=3)[0].detach().cpu().numpy() > 0.5
    ) & pred_valid_np[None, :]

    box_video = _render_box_overlay(
        summary_background,
        gt_box_xyxy_np,
        gt_box_valid_np,
        pred_box_xyxy_np,
        pred_box_valid_np,
        image_hw,
    )

    track_video = _render_track_overlay(
        summary_background,
        gt_track_summary_np,
        gt_track_valid_np,
        pred_track_summary_np,
        pred_track_valid_np,
        image_hw,
    )

    native_box_video = _render_native_box_overlay(
        sample["context_video"],
        gt_box_native_np,
        gt_box_native_valid_np,
        pred_box_native_np,
        pred_box_native_valid_np,
        image_hw,
    )
    native_box_sheet = _export_sheet(
        native_box_video,
        stem="native_box_overlay",
        title=f"{checkpoint_label} case {sample_index} native frame box overlay",
    )

    native_track_video = _render_native_track_overlay(
        sample["context_video"],
        gt_track_native_np,
        gt_track_native_valid_np,
        pred_track_native_np,
        pred_track_native_valid_np,
        image_hw,
    )
    native_track_sheet = _export_sheet(
        native_track_video,
        stem="native_track_overlay",
        title=f"{checkpoint_label} case {sample_index} native frame track overlay",
    )

    depth_sheet_rel = None
    depth_overlay_sheet_rel = None
    if gt_depth is not None and pred_depth is not None and gt_depth_valid is not None:
        gt_depth_np = gt_depth[0, ..., 0].detach().float().cpu().numpy()
        pred_depth_np = pred_depth[0, ..., 0].detach().float().cpu().numpy()
        gt_depth_valid_np = gt_depth_valid[0, ..., 0].detach().cpu().numpy() > 0.5
        if export_aux_visuals:
            depth_video = _render_depth_panel(gt_depth_np, gt_depth_valid_np, pred_depth_np)
            depth_sheet_rel = _export_sheet(
                depth_video,
                stem="depth_panel",
                title=f"{checkpoint_label} case {sample_index} depth panel",
            )

    track_center_frame_l1, track_delta_frame_l1, track_total_frame_l1 = _compute_framewise_track_summary_losses(
        pred_track_summary=pred_track_summary_np,
        gt_track_summary=gt_track_summary_np,
        gt_track_valid=gt_track_valid_np,
    )
    box_center_frame_l1, box_wh_frame_l1, box_iou_frame_l1, box_total_frame_l1 = _compute_framewise_box_summary_losses(
        pred_box_xyxy=pred_box_xyxy_np,
        gt_box_xyxy=gt_box_xyxy_np,
        gt_box_valid=gt_box_valid_np,
    )
    track_native_l1, track_native_iou = _compute_framewise_track_losses(
        pred_tracks_native=pred_track_native_np,
        gt_tracks_native=gt_track_native_np,
        gt_tracks_valid=gt_track_native_valid_np,
        gt_boxes_native=gt_box_native_np,
        gt_boxes_valid=gt_box_native_valid_np,
        image_hw=image_hw,
    )
    depth_frame_l1 = None
    if gt_depth is not None and pred_depth is not None and gt_depth_valid is not None:
        depth_frame_l1 = _compute_framewise_depth_losses(
            pred_depth=pred_depth[0, ..., 0].detach().float().cpu().numpy(),
            gt_depth=gt_depth[0, ..., 0].detach().float().cpu().numpy(),
            gt_depth_valid=gt_depth_valid[0, ..., 0].detach().cpu().numpy() > 0.5,
        )

    track_overlay_metrics = [
        [
            f"track_total={float(track_total_frame_l1[t, 0]):.6f}",
            f"track_center={float(track_center_frame_l1[t, 0]):.6f}",
            f"track_delta={float(track_delta_frame_l1[t, 0]):.6f}",
        ]
        for t in range(track_total_frame_l1.shape[0])
    ]
    box_overlay_metrics = [
        [
            f"box_total={float(box_total_frame_l1[t, 0]):.6f}",
            f"box_center={float(box_center_frame_l1[t, 0]):.6f}",
            f"box_wh={float(box_wh_frame_l1[t, 0]):.6f}",
            f"box_iou={float(box_iou_frame_l1[t, 0]):.6f}",
        ]
        for t in range(box_total_frame_l1.shape[0])
    ]
    box_sheet = None
    track_sheet = None
    track_summary_scalar_sheet = None
    box_summary_scalar_sheet = None
    track_native_scalar_sheet = None
    track_native_iou_sheet = None
    box_center_scalar_sheet = None
    box_iou_scalar_sheet = None
    depth_scalar_rel = None
    if export_aux_visuals:
        track_video = _overlay_metric_lines(
            track_video,
            title="track aux on decoded summary latent",
            per_frame_metrics=track_overlay_metrics,
        )
        box_video = _overlay_metric_lines(
            box_video,
            title="box aux on decoded summary latent",
            per_frame_metrics=box_overlay_metrics,
        )
        if depth_frame_l1 is not None:
            depth_overlay_video = _render_depth_overlay(
                summary_background,
                gt_box_xyxy_np,
                gt_box_valid_np,
                pred_box_xyxy_np,
                pred_box_valid_np,
                gt_depth_np,
                gt_depth_valid_np,
                pred_depth_np,
                image_hw,
            )
            depth_overlay_video = _overlay_metric_lines(
                depth_overlay_video,
                title="depth aux on decoded summary latent",
                per_frame_metrics=[
                    [f"depth_l1={float(depth_frame_l1[t, 0]):.6f}"]
                    for t in range(depth_frame_l1.shape[0])
                ],
            )
            depth_overlay_sheet_rel = _export_sheet(
                depth_overlay_video,
                stem="depth_overlay",
                title=f"{checkpoint_label} case {sample_index} depth overlay",
            )

        box_sheet = _export_sheet(
            box_video,
            stem="box_overlay",
            title=f"{checkpoint_label} case {sample_index} box overlay",
        )
        track_sheet = _export_sheet(
            track_video,
            stem="track_overlay",
            title=f"{checkpoint_label} case {sample_index} track overlay",
        )

        track_summary_scalar_sheet = str(
            _write_scalar_panel(
                assets_dir / f"{case_stem}__track_summary_loss_sheet.png",
                track_total_frame_l1,
                title=f"{checkpoint_label} case {sample_index} track summary loss",
            ).relative_to(output_dir)
        )
        box_summary_scalar_sheet = str(
            _write_scalar_panel(
                assets_dir / f"{case_stem}__box_summary_loss_sheet.png",
                box_total_frame_l1,
                title=f"{checkpoint_label} case {sample_index} box summary loss",
            ).relative_to(output_dir)
        )
        track_native_scalar_sheet = str(
            _write_scalar_panel(
                assets_dir / f"{case_stem}__native_track_loss_sheet.png",
                track_native_l1,
                title=f"{checkpoint_label} case {sample_index} native track loss",
            ).relative_to(output_dir)
        )
        track_native_iou_sheet = str(
            _write_scalar_panel(
                assets_dir / f"{case_stem}__native_track_iou_loss_sheet.png",
                track_native_iou,
                title=f"{checkpoint_label} case {sample_index} native track IoU loss",
            ).relative_to(output_dir)
        )
        box_center_scalar_sheet = str(
            _write_scalar_panel(
                assets_dir / f"{case_stem}__box_center_loss_sheet.png",
                box_center_frame_l1,
                title=f"{checkpoint_label} case {sample_index} box center loss",
            ).relative_to(output_dir)
        )
        box_iou_scalar_sheet = str(
            _write_scalar_panel(
                assets_dir / f"{case_stem}__box_iou_loss_sheet.png",
                box_iou_frame_l1,
                title=f"{checkpoint_label} case {sample_index} box IoU loss",
            ).relative_to(output_dir)
        )
        if depth_frame_l1 is not None:
            depth_scalar_rel = str(
                _write_scalar_panel(
                    assets_dir / f"{case_stem}__depth_loss_sheet.png",
                    depth_frame_l1,
                    title=f"{checkpoint_label} case {sample_index} depth loss",
                ).relative_to(output_dir)
            )

    metrics = _compute_aux_metrics(
        pred_track_summary=object_aux_out.pred_track_summary,
        gt_track_summary=gt_track_summary,
        gt_track_valid=gt_track_valid.to(dtype=object_aux_out.pred_track_summary.dtype),
        pred_box_xyxy=object_aux_out.pred_box_xyxy,
        gt_box_xyxy=gt_box_xyxy,
        gt_box_valid=gt_box_valid.to(dtype=object_aux_out.pred_box_xyxy.dtype),
        pred_depth=pred_depth,
        gt_depth=gt_depth,
        gt_depth_valid=gt_depth_valid.to(dtype=pred_depth.dtype) if gt_depth_valid is not None and pred_depth is not None else None,
        track_box_loss=track_box_loss,
        track_iou_loss=track_iou_loss,
    )
    metrics.update(
        _compute_box_decomposition_metrics(
            active_box_xyxy=object_out.active_box_xyxy,
            pred_box_xyxy=object_aux_out.pred_box_xyxy,
            gt_box_xyxy=gt_box_xyxy,
            gt_box_valid=gt_box_valid,
        )
    )
    metrics.update(
        _compute_regularizer_metrics(
            object_context=object_context,
            object_latent_tokens=object_out.object_latent_tokens,
            track_delta=object_aux_out.track_delta,
            box_center_delta=object_aux_out.box_center_delta,
            box_log_scale=object_aux_out.box_log_scale,
        )
    )
    metrics.update(
        {
            "train/loss_track_center_aux": float(track_center_frame_l1.mean().item()),
            "train/loss_track_delta_aux": float(track_delta_frame_l1.mean().item()),
        }
    )
    return {
        "sample_index": int(sample_index),
        "video_path": sample["video_path"],
        "caption": sample["caption"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "checkpoint": str(_resolve_checkpoint_file(checkpoint_path)),
        "checkpoint_label": checkpoint_label,
        "box_overlay_sheet": box_sheet,
        "track_overlay_sheet": track_sheet,
        "native_box_overlay_sheet": native_box_sheet,
        "native_track_overlay_sheet": native_track_sheet,
        "track_summary_loss_sheet": track_summary_scalar_sheet,
        "box_summary_loss_sheet": box_summary_scalar_sheet,
        "native_track_loss_sheet": track_native_scalar_sheet,
        "native_track_iou_loss_sheet": track_native_iou_sheet,
        "box_center_loss_sheet": box_center_scalar_sheet,
        "box_iou_loss_sheet": box_iou_scalar_sheet,
        "depth_panel_sheet": depth_sheet_rel,
        "depth_overlay_sheet": depth_overlay_sheet_rel,
        "depth_loss_sheet": depth_scalar_rel,
        "metrics": metrics,
        "shapes": {
            "gt_track_summary": list(gt_track_summary.shape),
            "gt_box_xyxy": list(gt_box_xyxy.shape),
            "gt_track_native": list(track_alignment.matched_gt_centers.shape),
            "gt_box_native": list(matched_gt_boxes.shape),
            "gt_depth": None if gt_depth is None else list(gt_depth.shape),
            "pred_track_summary": list(object_aux_out.pred_track_summary.shape),
            "pred_box_xyxy": list(object_aux_out.pred_box_xyxy.shape),
            "pred_track_native": list(center_tracks_native.shape),
            "pred_box_native": list(native_pred_box_xyxy.shape),
            "pred_depth": None if pred_depth is None else list(pred_depth.shape),
            "query_points_prior": list(query_points_prior.shape),
            "tracks_grouped": list(tracks_grouped.shape),
            "object_tokens": list(object_out.object_latent_tokens.shape),
            "track_summary_loss": list(track_total_frame_l1.shape),
            "box_summary_loss": list(box_total_frame_l1.shape),
            "native_track_loss": list(track_native_l1.shape),
            "native_track_iou_loss": list(track_native_iou.shape),
            "box_center_loss": list(box_center_frame_l1.shape),
            "box_iou_loss": list(box_iou_frame_l1.shape),
            "depth_loss": None if depth_frame_l1 is None else list(depth_frame_l1.shape),
        },
        "object_context_abs_max": float(object_context.detach().abs().max().item()),
        "active_box_xyxy": object_out.active_box_xyxy[0].detach().float().cpu().numpy().tolist(),
    }


def _build_report(
    *,
    results_by_case: list[dict[str, Any]],
    summary_by_checkpoint: list[dict[str, Any]],
    output_dir: Path,
    native_only: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics.json"
    summary_payload = {
        "summary_by_checkpoint": summary_by_checkpoint,
        "cases": results_by_case,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _image_block(*, image_src: str | None, image_alt: str, caption: str, sheet: bool = False) -> str:
        klass = ' class="sheet"' if sheet else ""
        if image_src is not None:
            return f"""
          <figure{klass}>
            <img loading="lazy" src="{image_src}" alt="{image_alt}">
            <figcaption>{caption}</figcaption>
          </figure>
"""
        return ""

    def _metric_row(label: str, value: Any) -> str:
        if value is None:
            shown = "-"
        else:
            try:
                shown = f"{float(value):.4f}"
            except Exception:
                shown = str(value)
        return f"<tr><th>{label}</th><td>{shown}</td></tr>"

    summary_rows = []
    for item in summary_by_checkpoint:
        summary_rows.append(
            f"""
      <tr>
        <td>{item['checkpoint_label']}</td>
        <td>{item['checkpoint']}</td>
        <td>{item['mean_loss_total']:.4f}</td>
        <td>{item['mean_track_aux']:.6f}</td>
        <td>{item['mean_box_aux']:.6f}</td>
        <td>{item['mean_depth_aux']:.6f}</td>
        <td>{item['mean_track_box']:.6f}</td>
        <td>{item['mean_track_iou']:.6f}</td>
      </tr>
"""
        )

    case_blocks: list[str] = []
    for case_result in results_by_case:
        checkpoint_cards = []
        for item in case_result["checkpoints"]:
            metrics = item["metrics"]
            metric_table = f"""
          <table class="metric-table">
            <tbody>
              {_metric_row("loss_track_aux", metrics.get("train/loss_track_aux"))}
              {_metric_row("loss_box_aux", metrics.get("train/loss_box_aux"))}
              {_metric_row("loss_depth_aux", metrics.get("train/loss_depth_aux"))}
              {_metric_row("track_box_loss", metrics.get("train/track_box_loss"))}
              {_metric_row("track_iou_loss", metrics.get("train/track_iou_loss"))}
              {_metric_row("loss_track_center_aux", metrics.get("train/loss_track_center_aux"))}
              {_metric_row("loss_track_delta_aux", metrics.get("train/loss_track_delta_aux"))}
              {_metric_row("loss_object_context_reg", metrics.get("train/loss_object_context_reg"))}
              {_metric_row("loss_track_anchor_reg", metrics.get("train/loss_track_anchor_reg"))}
              {_metric_row("loss_box_anchor_reg", metrics.get("train/loss_box_anchor_reg"))}
              {_metric_row("object_context_abs_max", metrics.get("train/object_context_abs_max"))}
              {_metric_row("object_latent_tokens_abs_max", metrics.get("train/object_latent_tokens_abs_max"))}
            </tbody>
          </table>
"""
            if native_only:
                visual_grid = f"""
          <div class="video-grid">
            {_image_block(image_src=item['native_track_overlay_sheet'], image_alt='native track sheet', caption='Track overlay on original context frames', sheet=True)}
            {_image_block(image_src=item['native_box_overlay_sheet'], image_alt='native box sheet', caption='Box overlay on original context frames', sheet=True)}
          </div>
"""
            else:
                visual_grid = f"""
          <div class="video-grid">
            {_image_block(image_src=item['track_overlay_sheet'], image_alt='track sheet', caption='Track aux on decoded summary latent', sheet=True)}
            {_image_block(image_src=item['box_overlay_sheet'], image_alt='box sheet', caption='Box aux on decoded summary latent', sheet=True)}
            {_image_block(image_src=item['native_track_overlay_sheet'], image_alt='native track sheet', caption='Native 8-frame track 逐帧静态图', sheet=True)}
            {_image_block(image_src=item['native_box_overlay_sheet'], image_alt='native box sheet', caption='Native 8-frame box 逐帧静态图', sheet=True)}
            {_image_block(image_src=item['track_summary_loss_sheet'], image_alt='track summary loss', caption='Track summary loss panel', sheet=True)}
            {_image_block(image_src=item['box_summary_loss_sheet'], image_alt='box summary loss', caption='Box summary loss panel', sheet=True)}
            {_image_block(image_src=item['native_track_loss_sheet'], image_alt='native track loss', caption='Native track L1 panel', sheet=True)}
            {_image_block(image_src=item['native_track_iou_loss_sheet'], image_alt='native track iou loss', caption='Native track IoU panel', sheet=True)}
            {_image_block(image_src=item['box_center_loss_sheet'], image_alt='box center loss', caption='Box center loss panel', sheet=True)}
            {_image_block(image_src=item['box_iou_loss_sheet'], image_alt='box iou loss', caption='Box IoU loss panel', sheet=True)}
            {_image_block(image_src=item.get('depth_loss_sheet'), image_alt='depth loss', caption='Depth loss framewise panel')}
            {_image_block(image_src=item['depth_overlay_sheet'], image_alt='depth overlay sheet', caption='Depth aux on decoded summary latent')}
            {_image_block(image_src=item['depth_panel_sheet'], image_alt='depth sheet', caption='Depth aux 逐帧静态图', sheet=True)}
          </div>
"""
            checkpoint_cards.append(
                f"""
        <article class="checkpoint-card">
          <h3>{item['checkpoint_label']}</h3>
          <p class="ckpt-path">{item['checkpoint']}</p>
          <p><b>Losses:</b> main={item['metrics']['train/loss_main']:.4f}, track_aux={item['metrics']['train/loss_track_aux']:.4f}, box_aux={item['metrics']['train/loss_box_aux']:.4f}, depth_aux={item['metrics']['train/loss_depth_aux']:.4f}</p>
          <p><b>Color legend:</b> yellow/orange = GT track center, blue = pred track center, red = GT box, teal/green = pred box</p>
          {visual_grid}
          {metric_table}
        </article>
"""
            )
        case_blocks.append(
            f"""
  <section class="case">
    <h2>{case_result.get('case_group', 'Case')} / index {case_result['sample_index']}</h2>
    <p><b>Caption:</b> {case_result['caption']}</p>
    <p><b>Sample ID:</b> {case_result.get('sample_id', '-')}</p>
    <p><b>Context frames:</b> {case_result['context_frame_indices']}</p>
    <p><b>Video:</b> {case_result['video_path']}</p>
    <div class="checkpoint-grid">
      {''.join(checkpoint_cards)}
    </div>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>v_newtrain Aux Loss Comparison</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f0ece1; }}
    .case {{ margin-top: 36px; padding-top: 12px; border-top: 2px solid #d8d1c2; }}
    .checkpoint-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 18px; align-items: start; }}
    .checkpoint-card {{ background: #fff; border: 1px solid #ddd; padding: 14px; }}
    .video-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 12px; align-items: start; }}
    .video-grid video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    .video-grid img {{ width: 100%; border: 1px solid #ccc; background: #fff; }}
    .sheet {{ grid-column: span 2; }}
    .ckpt-path {{ font-size: 12px; color: #555; word-break: break-all; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    .metric-table {{ margin-top: 12px; width: 100%; }}
    .metric-table th {{ width: 40%; background: #faf7ef; }}
    .metric-table td, .metric-table th {{ padding: 6px 8px; }}
  </style>
</head>
<body>
  <h1>v_newtrain Train Aux Loss Comparison</h1>
  <p>这页默认聚焦原始 context frame 上的 overlay，对没有直接画回原图的可视化面板不再展示。相关 loss / regularizer 统一以四位小数数值表给出，方便直接横向比较。</p>
  <p><b>Color legend:</b> yellow/orange = GT track center, blue = pred track center, red = GT box, teal/green = pred box。</p>
  <h2>Checkpoint Summary</h2>
  <table>
    <thead>
      <tr>
        <th>checkpoint</th>
        <th>path</th>
        <th>mean total</th>
        <th>mean track_aux</th>
        <th>mean box_aux</th>
        <th>mean depth_aux</th>
        <th>mean track_box_l1</th>
        <th>mean track_iou</th>
      </tr>
    </thead>
    <tbody>
      {''.join(summary_rows)}
    </tbody>
  </table>
  {''.join(case_blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wan-root", default=DEFAULT_WAN_ROOT)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--jepa-ckpt-path", default=DEFAULT_JEPA_CKPT)
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument("--cotracker-checkpoint", default=DEFAULT_COTRACKER_CKPT)
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Only export contact-sheet images and the HTML report; skip mp4 video generation.",
    )
    args = parser.parse_args()

    checkpoints = _resolve_checkpoints(args)
    dataset = PhysStateEpisodeDataset(
        root=args.dataset_root,
        split=args.split,
        resolution=(int(args.height), int(args.width)),
        num_context_frames=int(args.num_context_frames),
        context_fraction=0.5,
        random_context_frames=False,
        seed=42,
    )
    samples = {int(sample_index): dataset[int(sample_index)] for sample_index in args.indices}
    model = build_model(_build_model_args(args))
    model.to(torch.device(args.device))
    model.eval()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_case: list[dict[str, Any]] = []
    per_case_records = {
        int(sample_index): {
            "sample_index": int(sample_index),
            "caption": samples[int(sample_index)]["caption"],
            "video_path": samples[int(sample_index)]["video_path"],
            "context_frame_indices": samples[int(sample_index)]["context_frame_indices"].tolist(),
            "checkpoints": [],
        }
        for sample_index in args.indices
    }
    summary_by_checkpoint: list[dict[str, Any]] = []

    for checkpoint_path in checkpoints:
        load_info = _load_v_newtrain_state_into_model(model, checkpoint_path)
        checkpoint_label = _checkpoint_label(checkpoint_path)
        metric_rows = []
        for sample_index in args.indices:
            item = _run_case_for_checkpoint(
                model=model,
                sample=samples[int(sample_index)],
                checkpoint_path=checkpoint_path,
                checkpoint_label=checkpoint_label,
                sample_index=int(sample_index),
                output_dir=output_dir,
                fps=int(args.fps),
            )
            item["load_info"] = load_info
            per_case_records[int(sample_index)]["checkpoints"].append(item)
            metric_rows.append(item["metrics"])
        summary_by_checkpoint.append(
            {
                "checkpoint_label": checkpoint_label,
                "checkpoint": str(_resolve_checkpoint_file(checkpoint_path)),
                "mean_track_aux": float(np.mean([row["train/loss_track_aux"] for row in metric_rows])),
                "mean_box_aux": float(np.mean([row["train/loss_box_aux"] for row in metric_rows])),
                "mean_depth_aux": float(np.mean([row["train/loss_depth_aux"] for row in metric_rows])),
                "mean_track_box": float(np.mean([row["train/track_box_loss"] for row in metric_rows])),
                "mean_track_iou": float(np.mean([row["train/track_iou_loss"] for row in metric_rows])),
            }
        )

    for sample_index in args.indices:
        results_by_case.append(per_case_records[int(sample_index)])

    html_path = _build_report(
        results_by_case=results_by_case,
        summary_by_checkpoint=summary_by_checkpoint,
        output_dir=output_dir,
    )
    print(f"aux loss comparison report: {html_path}")


if __name__ == "__main__":
    main()
