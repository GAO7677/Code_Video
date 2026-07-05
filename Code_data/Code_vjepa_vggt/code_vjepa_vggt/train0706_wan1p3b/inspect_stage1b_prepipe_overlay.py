from __future__ import annotations

# Run command example:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
# CUDA_VISIBLE_DEVICES=7 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/inspect_stage1b_prepipe_overlay.py \
#   --steps step-002500 step-007000 \
#   --input-jsons \
#     /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000336_w001.json \
#     /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json \
#     /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/phyco_kubric_ball_drop_soft_v4_2025-09-05_0144a4.json

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.infer_context_video_wan import (
    _draw_box_rgb,
    _draw_point_rgb,
    _ensure_browser_video,
    _overlay_mask,
    _write_mp4,
)
from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.train0706_wan1p3b import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0706
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
    read_video_uniform,
)


DEFAULT_CHECKPOINT_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_diffsynth_native0706_wan21_13b/run_gpu0235_20260703/checkpoints"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v_1p3b/"
    "train_stage1b_diffsynth_native0706_wan21_13b/prepipe_overlays"
)

OBJECT_COLORS = [
    (230, 57, 70),
    (29, 78, 216),
    (46, 125, 50),
    (245, 158, 11),
]
PROMPT_COLOR = (255, 140, 0)
TRACK_LINE_THICKNESS = 2
DUMMY_BOX_XYXY = (0.45, 0.45, 0.55, 0.55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the train0706 pre-pipe processing on selected json cases. "
            "Outputs context overlays with viewer-grounding boxes, query points, and "
            "CoTracker point tracks, plus object_context stats before pipe()."
        )
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--steps", nargs="+", required=True, help="checkpoint step names, e.g. step-002500 step-007000")
    parser.add_argument("--input-jsons", nargs="+", required=True, help="input case json paths")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--spatial-only", action="store_true", help="skip object_context/VGGT/JEPA and only export box/query/track overlays")
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B")
    parser.add_argument("--diffsynth-root", default="/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
    parser.add_argument(
        "--lora-checkpoint",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/smoke/raw_phys_state_lora_continue/checkpoints/step-000002/checkpoint.safetensors",
    )
    parser.add_argument(
        "--stage1a-init-from",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token/step_0003000.pt",
    )
    parser.add_argument("--grounding-device", default=None)
    parser.add_argument("--aux-device", default=None, help="optional device for live VGGT forward, e.g. cuda:1")
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--sam2-segment-len", type=int, default=8)
    parser.add_argument("--grounding-proposal-source", default="gdino_only")
    parser.add_argument("--grounding-motion-score-ratio", type=float, default=0.15)
    parser.add_argument(
        "--grounding-text-prompt",
        default="box . cube . block . cylinder . capsule . sphere . ball .",
    )
    parser.add_argument("--grounding-extra-prompt-terms", default="")
    parser.add_argument("--grounding-disable-caption-terms", action="store_true", default=True)
    parser.add_argument("--grounding-gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--grounding-prompt-frame-mode", default="first")
    parser.add_argument("--grounding-track-dedupe-iou-threshold", type=float, default=0.75)
    parser.add_argument("--grounding-container-suppress-ratio-threshold", type=float, default=0.95)
    parser.add_argument("--grounding-container-suppress-min-contained", type=int, default=2)
    parser.add_argument("--grounding-container-suppress-min-area-ratio", type=float, default=1.5)
    parser.add_argument("--grounding-container-suppress-small-iou-threshold", type=float, default=0.7)
    return parser.parse_args()


def _resolve_runtime_device(device_arg: str) -> str:
    if str(device_arg).strip() and str(device_arg).strip().lower() != "cuda":
        return str(device_arg).strip()
    return infer0706._resolve_launch_device()


def _build_infer_args(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
) -> argparse.Namespace:
    old_argv = list(sys.argv)
    try:
        sys.argv = [
            old_argv[0],
            "--checkpoint",
            str(checkpoint_dir),
            "--context-video",
            "/tmp/dummy_context.mp4",
            "--prompt",
            "dummy",
            "--output-dir",
            str(output_dir),
        ]
        infer_args = infer0706.parse_args()
    finally:
        sys.argv = old_argv

    infer_args.checkpoint = str(checkpoint_dir)
    infer_args.output_dir = str(output_dir)
    infer_args.device = _resolve_runtime_device(args.device)
    infer_args.height = int(args.height)
    infer_args.width = int(args.width)
    infer_args.context_frames = int(args.context_frames)
    infer_args.num_frames = int(args.num_frames)
    infer_args.fps = int(args.fps)
    infer_args.seed = int(args.seed)
    infer_args.wan_root = str(args.wan_root)
    infer_args.diffsynth_root = str(args.diffsynth_root)
    infer_args.lora_checkpoint = str(args.lora_checkpoint)
    infer_args.stage1a_init_from = str(args.stage1a_init_from)
    infer_args.grounding_device = args.grounding_device
    infer_args.vggt_cache_root = args.vggt_cache_root
    infer_args.initialize_model_on_cpu = bool(args.initialize_model_on_cpu)
    return infer_args


def _build_viewer_grounding_provider(args: argparse.Namespace) -> ViewerGroundingBoxProvider:
    grounding_device = str(args.grounding_device or "cpu")
    include_caption_terms = not bool(args.grounding_disable_caption_terms)
    return ViewerGroundingBoxProvider(
        device=grounding_device,
        segment_len=int(args.sam2_segment_len),
        max_objects=int(args.aux_max_objects),
        points_per_object=int(args.object_num_queries),
        proposal_source=str(args.grounding_proposal_source),
        motion_score_ratio=float(args.grounding_motion_score_ratio),
        text_prompt=str(args.grounding_text_prompt),
        extra_prompt_terms=str(args.grounding_extra_prompt_terms),
        include_caption_terms=include_caption_terms,
        gdino_box_threshold=float(args.grounding_gdino_box_threshold),
        gdino_text_threshold=float(args.grounding_gdino_text_threshold),
        prompt_frame_mode=str(args.grounding_prompt_frame_mode),
        track_dedupe_iou_threshold=float(args.grounding_track_dedupe_iou_threshold),
        container_suppress_ratio_threshold=float(args.grounding_container_suppress_ratio_threshold),
        container_suppress_min_contained=int(args.grounding_container_suppress_min_contained),
        container_suppress_min_area_ratio=float(args.grounding_container_suppress_min_area_ratio),
        container_suppress_small_iou_threshold=float(args.grounding_container_suppress_small_iou_threshold),
    )


def _build_cotracker_adapter(args: argparse.Namespace) -> CoTrackerAdapter:
    cotracker_device = str(args.aux_device or _resolve_runtime_device(args.device))
    return CoTrackerAdapter(
        checkpoint_path=str(args.cotracker_checkpoint),
        num_queries=int(args.aux_max_objects) * int(args.object_num_queries),
        device=cotracker_device,
        input_hw=(int(args.cotracker_input_h), int(args.cotracker_input_w)),
        window_len=int(args.cotracker_window_len),
    )


def _build_priors_from_grounding_sample(
    grounding_sample,
    *,
    aux_max_objects: int,
    object_num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped_queries = torch.from_numpy(grounding_sample.grouped_queries_px).float()
    object_valid_mask = torch.from_numpy(grounding_sample.object_valid_mask).float()
    context_boxes_norm = torch.from_numpy(grounding_sample.context_boxes_norm).float()
    prompt_frame_idx = int(getattr(grounding_sample, "prompt_frame_idx", 0))

    flat = grouped_queries.view(1, int(aux_max_objects) * int(object_num_queries), 2)

    box_priors = []
    frame_ids = []
    valid_frames = int(context_boxes_norm.shape[0])
    for object_idx in range(int(aux_max_objects)):
        is_valid = bool(object_valid_mask[object_idx].item() > 0.5)
        first_valid_frame = 0
        box = None
        if is_valid:
            for frame_idx in range(valid_frames):
                candidate = context_boxes_norm[frame_idx, object_idx]
                if bool((candidate[2] - candidate[0] > 1.0e-6) and (candidate[3] - candidate[1] > 1.0e-6)):
                    first_valid_frame = frame_idx
                    box = candidate
                    break
        if box is None:
            box = torch.tensor(DUMMY_BOX_XYXY, dtype=torch.float32)
            first_valid_frame = prompt_frame_idx if is_valid else 0
        box_priors.append(box.to(dtype=torch.float32))
        frame_ids.extend([float(first_valid_frame)] * int(object_num_queries))

    box_prior_xyxy = torch.stack(box_priors, dim=0).view(1, int(aux_max_objects), 4)
    frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.float32).view(
        1, int(aux_max_objects) * int(object_num_queries), 1
    )
    object_valid = object_valid_mask.view(1, int(aux_max_objects))
    return flat, frame_ids_tensor, object_valid, box_prior_xyxy


def _load_context_video_for_mode(
    *,
    video_path: Path,
    target_context_frames: int,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if sampling_mode == "uniform":
        frames, frame_indices = read_video_uniform(video_path, target_context_frames)
    else:
        frames, frame_indices = read_video_prefix(video_path, target_context_frames)
    if int(frames.shape[0]) < int(target_context_frames):
        raise RuntimeError(
            f"context video {video_path} only provides {int(frames.shape[0])} frames, "
            f"smaller than required num_context_frames={int(target_context_frames)}"
        )
    if int(frames.shape[0]) > int(target_context_frames):
        frames = frames[:target_context_frames]
        frame_indices = frame_indices[:target_context_frames]
    return frames, frame_indices


def _context_tensor_to_uint8(context_video_single: torch.Tensor) -> np.ndarray:
    frames = context_video_single.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return frames


def _draw_track_trail(
    image: np.ndarray,
    track_points: np.ndarray,
    *,
    visible_mask: np.ndarray,
    color_rgb: tuple[int, int, int],
    upto_index: int,
) -> None:
    if upto_index <= 0:
        return
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    for idx in range(1, upto_index + 1):
        if not bool(visible_mask[idx]) or not bool(visible_mask[idx - 1]):
            continue
        pt0 = track_points[idx - 1]
        pt1 = track_points[idx]
        x0, y0 = [int(round(v)) for v in pt0.tolist()]
        x1, y1 = [int(round(v)) for v in pt1.tolist()]
        cv2.line(image, (x0, y0), (x1, y1), color_bgr, TRACK_LINE_THICKNESS, cv2.LINE_AA)


def _render_overlay_video(
    *,
    context_frames: np.ndarray,
    prompt_frame_idx: int,
    object_tracks: list,
    grouped_queries_px: np.ndarray,
    cotracker_tracks: np.ndarray,
    cotracker_visibility: np.ndarray,
) -> np.ndarray:
    rendered = []
    num_frames = int(context_frames.shape[0])
    num_objects = int(grouped_queries_px.shape[0])
    points_per_object = int(grouped_queries_px.shape[1]) if grouped_queries_px.ndim >= 3 else 0

    for frame_idx in range(num_frames):
        frame = context_frames[frame_idx].copy()
        for obj_idx, track in enumerate(object_tracks):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            if frame_idx < int(track.masks_thw.shape[0]):
                frame = _overlay_mask(frame, track.masks_thw[frame_idx], color, alpha=0.28)
            if frame_idx < int(track.boxes_t4.shape[0]):
                _draw_box_rgb(frame, track.boxes_t4[frame_idx].astype(np.float32), color, f"sam{obj_idx}")
            if frame_idx == int(prompt_frame_idx):
                _draw_box_rgb(frame, track.box_prompt_xyxy.astype(np.float32), PROMPT_COLOR, f"prompt{obj_idx}")

        for obj_idx in range(num_objects):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            for point_idx in range(points_per_object):
                flat_idx = obj_idx * points_per_object + point_idx
                track_t2 = cotracker_tracks[:, flat_idx].astype(np.float32)
                vis_t = cotracker_visibility[:, flat_idx] > 0.5
                _draw_track_trail(
                    frame,
                    track_t2,
                    visible_mask=vis_t,
                    color_rgb=color,
                    upto_index=frame_idx,
                )
                if frame_idx == 0:
                    _draw_point_rgb(
                        frame,
                        grouped_queries_px[obj_idx, point_idx].astype(np.float32),
                        color,
                        "",
                        radius=4,
                    )
                if bool(vis_t[frame_idx]):
                    label = f"o{obj_idx}" if point_idx == 0 else ""
                    _draw_point_rgb(
                        frame,
                        track_t2[frame_idx],
                        color,
                        label,
                        radius=4,
                    )
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def _save_prompt_frame_preview(
    *,
    context_frames: np.ndarray,
    prompt_frame_idx: int,
    object_tracks: list,
    grouped_queries_px: np.ndarray,
    output_path: Path,
) -> None:
    frame = context_frames[int(prompt_frame_idx)].copy()
    for obj_idx, track in enumerate(object_tracks):
        color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
        _draw_box_rgb(frame, track.box_prompt_xyxy.astype(np.float32), PROMPT_COLOR, f"prompt{obj_idx}")
        _draw_box_rgb(frame, track.boxes_t4[min(prompt_frame_idx, track.boxes_t4.shape[0] - 1)].astype(np.float32), color, f"sam{obj_idx}")
    if int(prompt_frame_idx) == 0:
        for obj_idx in range(int(grouped_queries_px.shape[0])):
            color = OBJECT_COLORS[obj_idx % len(OBJECT_COLORS)]
            for point_idx in range(int(grouped_queries_px.shape[1])):
                _draw_point_rgb(frame, grouped_queries_px[obj_idx, point_idx].astype(np.float32), color, "", radius=4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _build_spatial_debug(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
) -> dict[str, object]:
    pipe = model.pipe
    device = torch.device(pipe.device)
    image_hw = (int(context_video_single.shape[-2]), int(context_video_single.shape[-1]))
    sample = {
        "context_video": context_video_single,
        "num_context_frames": int(context_video_single.shape[1]),
        "caption": prompt,
        "video_path": video_path,
    }

    frames_tchw_01 = (
        ((context_video_single.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )
    if model.viewer_grounding is None:
        raise RuntimeError("viewer grounding provider is not initialized")
    grounding_sample = model.viewer_grounding.build_sample(
        frames_tchw_01=frames_tchw_01,
        caption=prompt,
        image_hw=image_hw,
    )

    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = model._build_object_query_priors(
        sample,
        image_hw=image_hw,
    )
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)

    context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    cotracker_out = model._run_cotracker(
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

    if model.vggt_cache_root:
        vggt_out = load_vggt_cache(sample, model.vggt_cache_root, allow_missing=False)
        if vggt_out is None:
            raise RuntimeError(f"VGGT cache missing for {video_path}")
    else:
        vggt_device = getattr(model.vggt_adapter, "device_obj", device)
        vggt_out = model.vggt_adapter(
            frames_bthwc_01.to(vggt_device),
            query_points_prior=query_points_prior.to(vggt_device),
            query_image_hw=image_hw,
        )
        for attr_name in (
            "query_points",
            "tracks",
            "visibility",
            "confidence",
            "pose_enc",
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
            "dense_patch_tokens",
        ):
            attr_value = getattr(vggt_out, attr_name, None)
            if isinstance(attr_value, torch.Tensor):
                setattr(vggt_out, attr_name, attr_value.to(device))

    jepa_out = model._run_jepa(context_video)
    context_latents = infer0706._encode_context_latents(pipe, context_video_single)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=context_latents,
        tracks=tracks_grouped,
        visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=box_prior_xyxy,
        vggt_world_points=getattr(vggt_out, "world_points", None),
        vggt_world_points_conf=getattr(vggt_out, "world_points_conf", None),
        vggt_depth=getattr(vggt_out, "depth", None),
        vggt_depth_conf=getattr(vggt_out, "depth_conf", None),
        vggt_dense_patch_tokens=getattr(vggt_out, "dense_patch_tokens", None),
        vggt_patch_grid_hw=getattr(vggt_out, "patch_grid_hw", None),
        vggt_geometry_image_hw=getattr(vggt_out, "input_hw", None)
        if getattr(vggt_out, "input_hw", None) is not None
        else getattr(vggt_out, "image_hw", None),
        frame_valid_mask=None,
    )
    object_context = model.object_adapter(
        object_out.object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )

    grouped_queries_px = grounding_sample.grouped_queries_px.astype(np.float32)
    cotracker_tracks = cotracker_out.tracks[0].detach().cpu().numpy().astype(np.float32)
    cotracker_visibility = cotracker_out.visibility[0].detach().cpu().numpy().astype(np.float32)
    cotracker_confidence = cotracker_out.confidence[0].detach().cpu().numpy().astype(np.float32)

    return {
        "grounding_sample": grounding_sample,
        "grouped_queries_px": grouped_queries_px,
        "query_points_prior": query_points_prior[0].detach().float().cpu().numpy().astype(np.float32),
        "query_frame_ids": query_frame_ids[0, :, 0].detach().float().cpu().numpy().astype(np.int32),
        "object_valid_mask": object_valid_mask[0].detach().float().cpu().numpy().astype(np.float32),
        "box_prior_xyxy": box_prior_xyxy[0].detach().float().cpu().numpy().astype(np.float32),
        "cotracker_tracks": cotracker_tracks,
        "cotracker_visibility": cotracker_visibility,
        "cotracker_confidence": cotracker_confidence,
        "tracks_grouped_shape": list(tracks_grouped.shape),
        "visibility_grouped_shape": list(visibility_grouped.shape),
        "confidence_grouped_shape": list(confidence_grouped.shape),
        "context_latents_shape": list(context_latents.shape),
        "object_latent_tokens_shape": list(object_out.object_latent_tokens.shape),
        "object_context_shape": list(object_context.shape),
        "object_context_abs_mean": float(object_context.detach().abs().mean().item()),
        "object_context_abs_max": float(object_context.detach().abs().max().item()),
        "object_context_mean": float(object_context.detach().mean().item()),
        "object_context_std": float(object_context.detach().float().std().item()),
    }


def _build_spatial_debug_standalone(
    *,
    viewer_grounding: ViewerGroundingBoxProvider,
    cotracker: CoTrackerAdapter,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
    aux_max_objects: int,
    object_num_queries: int,
) -> dict[str, object]:
    image_hw = (int(context_video_single.shape[-2]), int(context_video_single.shape[-1]))
    frames_tchw_01 = (
        ((context_video_single.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )
    grounding_sample = viewer_grounding.build_sample(
        frames_tchw_01=frames_tchw_01,
        caption=prompt,
        image_hw=image_hw,
    )
    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = _build_priors_from_grounding_sample(
        grounding_sample,
        aux_max_objects=int(aux_max_objects),
        object_num_queries=int(object_num_queries),
    )
    context_video = context_video_single.unsqueeze(0)
    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    cotracker_device = cotracker.device_obj
    cotracker_out = cotracker(
        frames_bthwc_01.to(cotracker_device),
        query_points_prior=query_points_prior.to(cotracker_device),
        query_frame_ids=query_frame_ids.to(cotracker_device),
        query_image_hw=image_hw,
    )
    cotracker_tracks = cotracker_out.tracks[0].detach().float().cpu().numpy().astype(np.float32)
    cotracker_visibility = cotracker_out.visibility[0].detach().float().cpu().numpy().astype(np.float32)
    cotracker_confidence = cotracker_out.confidence[0].detach().float().cpu().numpy().astype(np.float32)
    grouped_queries_px = grounding_sample.grouped_queries_px.astype(np.float32)
    return {
        "grounding_sample": grounding_sample,
        "grouped_queries_px": grouped_queries_px,
        "query_points_prior": query_points_prior[0].detach().float().cpu().numpy().astype(np.float32),
        "query_frame_ids": query_frame_ids[0, :, 0].detach().float().cpu().numpy().astype(np.int32),
        "object_valid_mask": object_valid_mask[0].detach().float().cpu().numpy().astype(np.float32),
        "box_prior_xyxy": box_prior_xyxy[0].detach().float().cpu().numpy().astype(np.float32),
        "cotracker_tracks": cotracker_tracks,
        "cotracker_visibility": cotracker_visibility,
        "cotracker_confidence": cotracker_confidence,
        "tracks_grouped_shape": None,
        "visibility_grouped_shape": None,
        "confidence_grouped_shape": None,
        "context_latents_shape": None,
        "object_latent_tokens_shape": None,
        "object_context_shape": None,
        "object_context_abs_mean": None,
        "object_context_abs_max": None,
        "object_context_mean": None,
        "object_context_std": None,
        "step_invariant_spatial_only": True,
        "video_path": str(video_path),
    }


def _case_output_name(input_json_path: Path) -> str:
    return input_json_path.stem


def _render_case(
    *,
    model,
    infer_args: argparse.Namespace | None,
    input_json_path: Path,
    sampling_mode: str,
    output_dir: Path,
    viewer_grounding: ViewerGroundingBoxProvider | None = None,
    cotracker: CoTrackerAdapter | None = None,
    spatial_only: bool = False,
    aux_max_objects: int = 4,
    object_num_queries: int = 8,
) -> dict[str, object]:
    payload = core._load_input_json(input_json_path)
    input_video = core._resolve_input_video(payload, input_json_path)
    input_caption = core._ensure_str_field(payload, "input_caption", input_json_path)

    context_video_path = Path(input_video).expanduser().resolve()
    raw_frames, frame_indices = _load_context_video_for_mode(
        video_path=context_video_path,
        target_context_frames=int(infer_args.context_frames),
        sampling_mode=sampling_mode,
    )
    context_video_single = preprocess_video_rgb_uint8(
        raw_frames,
        (int(infer_args.height), int(infer_args.width)),
    )
    context_frames = _context_tensor_to_uint8(context_video_single)

    with torch.no_grad():
        if spatial_only:
            if viewer_grounding is None or cotracker is None:
                raise RuntimeError("viewer_grounding and cotracker are required in spatial_only mode")
            debug = _build_spatial_debug_standalone(
                viewer_grounding=viewer_grounding,
                cotracker=cotracker,
                context_video_single=context_video_single,
                prompt=str(input_caption),
                video_path=str(context_video_path),
                aux_max_objects=int(aux_max_objects),
                object_num_queries=int(object_num_queries),
            )
        else:
            debug = _build_spatial_debug(
                model,
                context_video_single=context_video_single,
                prompt=str(input_caption),
                video_path=str(context_video_path),
            )

    grounding_sample = debug["grounding_sample"]
    overlay_video = _render_overlay_video(
        context_frames=context_frames,
        prompt_frame_idx=int(grounding_sample.prompt_frame_idx),
        object_tracks=grounding_sample.object_tracks,
        grouped_queries_px=debug["grouped_queries_px"],
        cotracker_tracks=debug["cotracker_tracks"],
        cotracker_visibility=debug["cotracker_visibility"],
    )
    overlay_raw = output_dir / "prepipe_overlay.mp4"
    fps = int(infer_args.fps) if infer_args is not None else 30
    _write_mp4(overlay_raw, overlay_video, fps=fps)
    overlay_browser = _ensure_browser_video(overlay_raw)

    prompt_preview = output_dir / "prompt_frame_preview.png"
    _save_prompt_frame_preview(
        context_frames=context_frames,
        prompt_frame_idx=int(grounding_sample.prompt_frame_idx),
        object_tracks=grounding_sample.object_tracks,
        grouped_queries_px=debug["grouped_queries_px"],
        output_path=prompt_preview,
    )

    object_tracks_meta = []
    for track in grounding_sample.object_tracks:
        object_tracks_meta.append(
            {
                "phrase": str(track.phrase),
                "score": float(track.score),
                "prompt_box_xyxy": track.box_prompt_xyxy.astype(np.float32).tolist(),
                "boxes_t4_shape": list(track.boxes_t4.shape),
                "masks_thw_shape": list(track.masks_thw.shape),
            }
        )

    result = {
        "input_json": str(input_json_path),
        "context_video": str(context_video_path),
        "input_caption": str(input_caption),
        "sampling_mode": str(sampling_mode),
        "frame_indices": frame_indices.tolist(),
        "context_frames_shape": list(context_frames.shape),
        "prompt_frame_idx": int(grounding_sample.prompt_frame_idx),
        "prompt_mode": str(grounding_sample.prompt_mode),
        "prior_source": str(grounding_sample.prior_source),
        "object_tracks": object_tracks_meta,
        "grounding_debug": dict(grounding_sample.debug),
        "query_points_shape": list(debug["query_points_prior"].shape),
        "query_points_prior": debug["query_points_prior"].tolist(),
        "query_frame_ids": debug["query_frame_ids"].tolist(),
        "object_valid_mask": debug["object_valid_mask"].tolist(),
        "box_prior_xyxy": debug["box_prior_xyxy"].tolist(),
        "cotracker_tracks_shape": list(debug["cotracker_tracks"].shape),
        "cotracker_visibility_shape": list(debug["cotracker_visibility"].shape),
        "cotracker_confidence_shape": list(debug["cotracker_confidence"].shape),
        "tracks_grouped_shape": debug["tracks_grouped_shape"],
        "visibility_grouped_shape": debug["visibility_grouped_shape"],
        "confidence_grouped_shape": debug["confidence_grouped_shape"],
        "context_latents_shape": debug["context_latents_shape"],
        "object_latent_tokens_shape": debug["object_latent_tokens_shape"],
        "object_context_shape": debug["object_context_shape"],
        "object_context_abs_mean": debug["object_context_abs_mean"],
        "object_context_abs_max": debug["object_context_abs_max"],
        "object_context_mean": debug["object_context_mean"],
        "object_context_std": debug["object_context_std"],
        "spatial_only": bool(spatial_only),
        "step_invariant_spatial_only": bool(debug.get("step_invariant_spatial_only", False)),
        "overlay_video": str(overlay_browser),
        "prompt_frame_preview": str(prompt_preview),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _write_step_index(step_dir: Path, step_name: str, results: list[dict[str, object]]) -> None:
    cards = []
    for result in results:
        overlay_rel = Path(result["overlay_video"]).relative_to(step_dir)
        preview_rel = Path(result["prompt_frame_preview"]).relative_to(step_dir)
        object_context_shape = result.get("object_context_shape")
        object_context_abs_mean = result.get("object_context_abs_mean")
        if object_context_shape is None or object_context_abs_mean is None:
            context_line = "<p><b>object_context:</b> spatial-only mode, not computed</p>"
        else:
            context_line = (
                f"<p><b>object_context_shape:</b> {object_context_shape} | "
                f"<b>abs_mean:</b> {float(object_context_abs_mean):.6f}</p>"
            )
        cards.append(
            f"""
<section class="card">
  <h2>{Path(str(result['input_json'])).stem}</h2>
  <p><b>context_video:</b> {result['context_video']}</p>
  <p><b>prompt_frame_idx:</b> {result['prompt_frame_idx']} | <b>prior_source:</b> {result['prior_source']}</p>
  {context_line}
  <img src="{preview_rel.as_posix()}" alt="prompt preview">
  <video controls preload="none" playsinline src="{overlay_rel.as_posix()}"></video>
</section>
"""
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{step_name} pre-pipe overlay</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f7f6f2; color: #1f2937; }}
    .card {{ background: #fff; border: 1px solid #ddd; padding: 16px; margin-bottom: 18px; }}
    img, video {{ width: 100%; max-width: 960px; display: block; margin-top: 10px; border: 1px solid #ddd; background: #000; }}
  </style>
</head>
<body>
  <h1>{step_name} pre-pipe overlay</h1>
  <p>这页展示送入 pipe() 之前的 spatial 前处理：viewer grounding box、query points、以及 CoTracker tracks。注意这些 spatial 结果主要依赖 frozen 模块，所以不同 step 的 overlay 理论上应保持一致；step 差异主要体现在 object_context 张量统计。</p>
  {''.join(cards)}
</body>
</html>
"""
    (step_dir / "index.html").write_text(html, encoding="utf-8")


def _write_root_index(output_root: Path, root_results: dict[str, list[dict[str, object]]]) -> None:
    items = []
    for step_name, results in root_results.items():
        items.append(
            f"<li><a href=\"{step_name}/index.html\">{step_name}</a> ({len(results)} cases)</li>"
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>train0706 pre-pipe overlay</title>
</head>
<body>
  <h1>train0706 pre-pipe overlay</h1>
  <ul>
    {''.join(items)}
  </ul>
</body>
</html>
"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = args.checkpoint_root.expanduser().resolve()

    input_jsons = [Path(item).expanduser().resolve() for item in args.input_jsons]
    root_results: dict[str, list[dict[str, object]]] = {}
    spatial_only_infer_args = argparse.Namespace(
        context_frames=int(args.context_frames),
        height=int(args.height),
        width=int(args.width),
        fps=int(args.fps),
    )
    viewer_grounding = None
    cotracker = None
    if args.spatial_only:
        viewer_grounding = _build_viewer_grounding_provider(args)
        cotracker = _build_cotracker_adapter(args)

    for step_name in args.steps:
        checkpoint_dir = checkpoint_root / step_name
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"checkpoint step not found: {checkpoint_dir}")
        step_dir = output_root / step_name
        step_dir.mkdir(parents=True, exist_ok=True)

        infer_args = None
        model = None
        if not args.spatial_only:
            infer_args = _build_infer_args(args, checkpoint_dir=checkpoint_dir, output_dir=step_dir)
            torch.manual_seed(int(infer_args.seed))
            np.random.seed(int(infer_args.seed))
            print(f"[load] {checkpoint_dir}")
            model, model_args, load_info = infer0706._build_runtime_model(infer_args)
            _ = model_args, load_info
            model.pipe.dit.eval()
            if args.aux_device and model.vggt_adapter is not None:
                aux_device = torch.device(str(args.aux_device))
                model.vggt_adapter = model.vggt_adapter.to(aux_device)
                model.vggt_adapter.device_obj = aux_device
                if getattr(model.vggt_adapter, "model", None) is not None:
                    model.vggt_adapter.model = model.vggt_adapter.model.to(aux_device)

        step_results: list[dict[str, object]] = []
        try:
            for input_json_path in input_jsons:
                case_name = _case_output_name(input_json_path)
                case_dir = step_dir / case_name
                case_dir.mkdir(parents=True, exist_ok=True)
                print(f"[case] {step_name} :: {input_json_path}")
                result = _render_case(
                    model=model,
                    infer_args=infer_args if infer_args is not None else spatial_only_infer_args,
                    input_json_path=input_json_path,
                    sampling_mode=str(args.sampling_mode),
                    output_dir=case_dir,
                    viewer_grounding=viewer_grounding,
                    cotracker=cotracker,
                    spatial_only=bool(args.spatial_only),
                    aux_max_objects=int(args.aux_max_objects),
                    object_num_queries=int(args.object_num_queries),
                )
                step_results.append(result)
        finally:
            if model is not None:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        _write_step_index(step_dir, step_name, step_results)
        root_results[step_name] = step_results

    _write_root_index(output_root, root_results)
    print(f"[done] outputs under {output_root}")


if __name__ == "__main__":
    main()
