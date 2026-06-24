from __future__ import annotations

import argparse
import http.server
import json
import socketserver
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
    ensure_browser_video,
    write_mp4,
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


def _render_box_overlay(
    context_video: torch.Tensor,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
    pred_box_xyxy: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = int(gt_box_xyxy.shape[0])
    source_frames = int(context_video.shape[1])
    latent_to_source = np.linspace(0, max(source_frames - 1, 0), latent_frames).round().astype(np.int64)
    for latent_idx in range(latent_frames):
        src_idx = int(latent_to_source[latent_idx])
        frame = tensor_frame_to_uint8_hwc(context_video[:, src_idx]).copy()
        for obj_idx in range(gt_box_xyxy.shape[1]):
            if bool(gt_box_valid[latent_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(gt_box_xyxy[latent_idx, obj_idx], image_hw),
                    BOX_GT_COLOR,
                    f"gt{obj_idx}",
                )
            draw_box_rgb(
                frame,
                _norm_box_to_px(pred_box_xyxy[latent_idx, obj_idx], image_hw),
                BOX_PRED_COLOR,
                f"pred{obj_idx}",
            )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_track_overlay(
    context_video: torch.Tensor,
    gt_track_summary: np.ndarray,
    gt_track_valid: np.ndarray,
    pred_track_summary: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = int(gt_track_summary.shape[0])
    source_frames = int(context_video.shape[1])
    latent_to_source = np.linspace(0, max(source_frames - 1, 0), latent_frames).round().astype(np.int64)
    for latent_idx in range(latent_frames):
        src_idx = int(latent_to_source[latent_idx])
        frame = tensor_frame_to_uint8_hwc(context_video[:, src_idx]).copy()
        for obj_idx in range(gt_track_summary.shape[1]):
            pred_center, pred_start = _summary_to_px(pred_track_summary[latent_idx, obj_idx], image_hw)
            draw_point_rgb(frame, pred_center, TRACK_PRED_COLOR, f"pred{obj_idx}", radius=5)
            draw_point_rgb(frame, pred_start, TRACK_PRED_COLOR, f"s{obj_idx}", radius=3)
            if bool(gt_track_valid[latent_idx, obj_idx]):
                gt_center, gt_start = _summary_to_px(gt_track_summary[latent_idx, obj_idx], image_hw)
                draw_point_rgb(frame, gt_center, TRACK_GT_COLOR, f"gt{obj_idx}", radius=5)
                draw_point_rgb(frame, gt_start, TRACK_GT_COLOR, f"gs{obj_idx}", radius=3)
        frames.append(frame)
    return np.stack(frames, axis=0)


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
    box_aux_loss = (((pred_box_xyxy - gt_box_xyxy).abs()) * gt_box_valid.unsqueeze(-1)).sum()
    box_aux_loss = box_aux_loss / (
        gt_box_valid.unsqueeze(-1).sum().clamp_min(1.0) * pred_box_xyxy.shape[-1]
    )
    depth_aux_loss = pred_track_summary.new_zeros(())
    if pred_depth is not None and gt_depth is not None and gt_depth_valid is not None:
        depth_aux_loss = (((pred_depth - gt_depth).abs()) * gt_depth_valid.unsqueeze(-1)).sum()
        depth_aux_loss = depth_aux_loss / (
            gt_depth_valid.unsqueeze(-1).sum().clamp_min(1.0) * pred_depth.shape[-1]
        )
    return {
        "train/loss_track_aux": float(track_aux_loss.detach().item()),
        "train/loss_box_aux": float(box_aux_loss.detach().item()),
        "train/loss_depth_aux": float(depth_aux_loss.detach().item()),
        "train/track_box_loss": float(track_box_loss.detach().item()),
        "train/track_iou_loss": float(track_iou_loss.detach().item()),
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
) -> dict[str, Any]:
    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video_single = sample["context_video"].to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video_single.shape[-2]), int(context_video_single.shape[-1]))
    query_points_prior, object_valid_mask = model._build_object_query_priors(sample, image_hw=image_hw)
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = ((context_video_single.unsqueeze(0).permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    with torch.no_grad():
        cotracker_out = model.cotracker_adapter(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
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
            frame_valid_mask=None,
        )
        object_aux_out = model.object_aux_heads(
            object_out.object_latent_tokens,
            object_out.active_track_summary,
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

    case_stem = f"case_{sample_index:05d}__{checkpoint_label}"
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    gt_track_summary_np = gt_track_summary[0].detach().float().cpu().numpy()
    gt_track_valid_np = gt_track_valid[0].detach().cpu().numpy() > 0.5
    gt_box_xyxy_np = gt_box_xyxy[0].detach().float().cpu().numpy()
    gt_box_valid_np = gt_box_valid[0].detach().cpu().numpy() > 0.5
    pred_track_summary_np = object_aux_out.pred_track_summary[0].detach().float().cpu().numpy()
    pred_box_xyxy_np = object_aux_out.pred_box_xyxy[0].detach().float().cpu().numpy()

    box_video = _render_box_overlay(
        sample["context_video"],
        gt_box_xyxy_np,
        gt_box_valid_np,
        pred_box_xyxy_np,
        image_hw,
    )
    box_raw = assets_dir / f"{case_stem}__box_overlay.mp4"
    write_mp4(box_raw, box_video, fps=fps)
    box_browser = ensure_browser_video(box_raw)
    box_sheet = _write_contact_sheet(
        assets_dir / f"{case_stem}__box_overlay_sheet.png",
        box_video,
        title=f"{checkpoint_label} case {sample_index} box overlay",
    )

    track_video = _render_track_overlay(
        sample["context_video"],
        gt_track_summary_np,
        gt_track_valid_np,
        pred_track_summary_np,
        image_hw,
    )
    track_raw = assets_dir / f"{case_stem}__track_overlay.mp4"
    write_mp4(track_raw, track_video, fps=fps)
    track_browser = ensure_browser_video(track_raw)
    track_sheet = _write_contact_sheet(
        assets_dir / f"{case_stem}__track_overlay_sheet.png",
        track_video,
        title=f"{checkpoint_label} case {sample_index} track overlay",
    )

    depth_video_rel = None
    depth_sheet_rel = None
    if gt_depth is not None and pred_depth is not None and gt_depth_valid is not None:
        gt_depth_np = gt_depth[0, ..., 0].detach().float().cpu().numpy()
        pred_depth_np = pred_depth[0, ..., 0].detach().float().cpu().numpy()
        gt_depth_valid_np = gt_depth_valid[0, ..., 0].detach().cpu().numpy() > 0.5
        depth_video = _render_depth_panel(gt_depth_np, gt_depth_valid_np, pred_depth_np)
        depth_raw = assets_dir / f"{case_stem}__depth_panel.mp4"
        write_mp4(depth_raw, depth_video, fps=max(1, min(fps, 8)))
        depth_browser = ensure_browser_video(depth_raw)
        depth_video_rel = str(depth_browser.relative_to(output_dir))
        depth_sheet = _write_contact_sheet(
            assets_dir / f"{case_stem}__depth_panel_sheet.png",
            depth_video,
            title=f"{checkpoint_label} case {sample_index} depth panel",
        )
        depth_sheet_rel = str(depth_sheet.relative_to(output_dir))

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
    return {
        "sample_index": int(sample_index),
        "video_path": sample["video_path"],
        "caption": sample["caption"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "checkpoint": str(_resolve_checkpoint_file(checkpoint_path)),
        "checkpoint_label": checkpoint_label,
        "box_overlay_video": str(box_browser.relative_to(output_dir)),
        "track_overlay_video": str(track_browser.relative_to(output_dir)),
        "box_overlay_sheet": str(box_sheet.relative_to(output_dir)),
        "track_overlay_sheet": str(track_sheet.relative_to(output_dir)),
        "depth_panel_video": depth_video_rel,
        "depth_panel_sheet": depth_sheet_rel,
        "metrics": metrics,
        "shapes": {
            "gt_track_summary": list(gt_track_summary.shape),
            "gt_box_xyxy": list(gt_box_xyxy.shape),
            "gt_depth": None if gt_depth is None else list(gt_depth.shape),
            "pred_track_summary": list(object_aux_out.pred_track_summary.shape),
            "pred_box_xyxy": list(object_aux_out.pred_box_xyxy.shape),
            "pred_depth": None if pred_depth is None else list(pred_depth.shape),
            "query_points_prior": list(query_points_prior.shape),
            "tracks_grouped": list(tracks_grouped.shape),
            "object_tokens": list(object_out.object_latent_tokens.shape),
        },
    }


def _build_report(
    *,
    results_by_case: list[dict[str, Any]],
    summary_by_checkpoint: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics.json"
    summary_payload = {
        "summary_by_checkpoint": summary_by_checkpoint,
        "cases": results_by_case,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_rows = []
    for item in summary_by_checkpoint:
        summary_rows.append(
            f"""
      <tr>
        <td>{item['checkpoint_label']}</td>
        <td>{item['checkpoint']}</td>
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
            depth_block = ""
            if item["depth_panel_video"] is not None:
                depth_block = f"""
          <figure>
            <video controls preload="none" playsinline src="{item['depth_panel_video']}"></video>
            <figcaption>Depth aux: GT depth vs Pred depth</figcaption>
          </figure>
          <figure class="sheet">
            <img loading="lazy" src="{item['depth_panel_sheet']}" alt="depth sheet">
            <figcaption>Depth aux 逐帧静态图</figcaption>
          </figure>
"""
            checkpoint_cards.append(
                f"""
        <article class="checkpoint-card">
          <h3>{item['checkpoint_label']}</h3>
          <p class="ckpt-path">{item['checkpoint']}</p>
          <p><b>Losses:</b> track_aux={item['metrics']['train/loss_track_aux']:.6f}, box_aux={item['metrics']['train/loss_box_aux']:.6f}, depth_aux={item['metrics']['train/loss_depth_aux']:.6f}</p>
          <p><b>Track alignment:</b> l1={item['metrics']['train/track_box_loss']:.6f}, iou={item['metrics']['train/track_iou_loss']:.6f}</p>
          <div class="video-grid">
            <figure>
              <video controls preload="none" playsinline src="{item['track_overlay_video']}"></video>
              <figcaption>Track aux: GT summary vs Pred summary</figcaption>
            </figure>
            <figure>
              <video controls preload="none" playsinline src="{item['box_overlay_video']}"></video>
              <figcaption>Box aux: GT box(red) vs Pred box(green)</figcaption>
            </figure>
            <figure class="sheet">
              <img loading="lazy" src="{item['track_overlay_sheet']}" alt="track sheet">
              <figcaption>Track aux 逐帧静态图</figcaption>
            </figure>
            <figure class="sheet">
              <img loading="lazy" src="{item['box_overlay_sheet']}" alt="box sheet">
              <figcaption>Box aux 逐帧静态图</figcaption>
            </figure>
            {depth_block}
          </div>
          <pre>{json.dumps({'metrics': item['metrics'], 'shapes': item['shapes']}, indent=2, ensure_ascii=False)}</pre>
        </article>
"""
            )
        case_blocks.append(
            f"""
  <section class="case">
    <h2>Case {case_result['sample_index']}</h2>
    <p><b>Caption:</b> {case_result['caption']}</p>
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
    pre {{ background: #fafafa; border: 1px solid #ddd; padding: 12px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>v_newtrain Train Aux Loss Comparison</h1>
  <p>这页只展示当前 `v_newtrain` 训练里真实参与 `train/loss_track_aux`、`train/loss_box_aux`、`train/loss_depth_aux` 计算的量。每一列对应一个 checkpoint，同一批 case 放在同一页横向对比。</p>
  <h2>Checkpoint Summary</h2>
  <table>
    <thead>
      <tr>
        <th>checkpoint</th>
        <th>path</th>
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
    parser.add_argument("--port", type=int, default=8812)
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

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", int(args.port)), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{int(args.port)}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
