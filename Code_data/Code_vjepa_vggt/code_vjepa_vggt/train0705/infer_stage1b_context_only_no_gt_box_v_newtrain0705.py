from __future__ import annotations

# Run command example:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
# CUDA_VISIBLE_DEVICES=7 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py \
#   --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
#   --context-video /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_001460/source_video/context_video_8f.mp4 \
#   --prompt "f5 sample 001460 industrial rigid body simulation sphere box" \
#   --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/inference_review/step-001000 \
#   --sampling-steps 12
#
# Guided example:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
# CUDA_VISIBLE_DEVICES=6 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py \
#   --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
#   --context-video /path/to/context_video_8f.mp4 \
#   --prompt "your prompt" \
#   --output-dir /data/gaoya/agent-data/outputs/train0705_vjepa_demo \
#   --sampling-steps 40 \
#   --vjepa-preset ladder_s20 \
#   --vjepa-device cuda:0

"""
Stage1B context-only no-GT-box inference for the train0705 DiffSynth-native run.

This is based on the official DiffSynth example:
  examples/wanvideo/model_inference/Wan2.2-TI2V-5B.py

Example run command:
  env PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main" \
  CUDA_VISIBLE_DEVICES="7" \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py \
    --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
    --context-video /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_001460/source_video/context_video_8f.mp4 \
    --prompt "f5 sample 001460 industrial rigid body simulation sphere box" \
    --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/inference_review/step-001000 \
    --sampling-steps 12

The script keeps the inference core on top of `diffsynth`, but reconstructs the
same object-conditioning path used during train0705:

  context video -> viewer grounding pseudo boxes -> CoTracker / VGGT / JEPA ->
  ObjectTubeProjector -> ObjectConditionAdapter -> Wan object branch

It loads three weight sources exactly like training:
  1. Wan 2.2 base from --wan-root
  2. Frozen base LoRA from --lora-checkpoint
  3. Frozen Stage1A object_pooler / object_aux_heads from --stage1a-init-from
  4. Stage1B trainable object_adapter + DiT object-branch weights from --checkpoint
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _read_cli_arg_value(argv: list[str], names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        if name not in argv:
            continue
        index = argv.index(name)
        if index + 1 < len(argv):
            return argv[index + 1]
    return default


_DEFAULT_DIFFSYNTH_ROOT = "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main"
_SELECTED_DIFFSYNTH_ROOT = _read_cli_arg_value(
    sys.argv,
    ("--diffsynth-root", "--diffsynth_root"),
    os.environ.get("DIFFSYNTH_ROOT", _DEFAULT_DIFFSYNTH_ROOT),
)
if _SELECTED_DIFFSYNTH_ROOT:
    os.environ["DIFFSYNTH_ROOT"] = _SELECTED_DIFFSYNTH_ROOT
    if _SELECTED_DIFFSYNTH_ROOT not in sys.path:
        sys.path.insert(0, _SELECTED_DIFFSYNTH_ROOT)

from diffsynth.utils.data import save_video

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_free.vjepa_guidance import WanVJEPAConfig, apply_train0705_preset
from code_vjepa_vggt.context_wan_v_newtrain import ContextAwareWanVideoPipeline
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset
from code_vjepa_vggt.train0705 import train_stage1b_context_only_no_gt_box_v_newtrain as t0705
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix


def _summarize_string_list(values, *, sample_limit: int = 8):
    values = list(values or [])
    return {
        "count": len(values),
        "sample": values[:sample_limit],
    }


def _summarize_load_info(load_info: dict) -> dict:
    summarized = {}
    for name, info in load_info.items():
        summarized[name] = {
            "loaded_count": int(info.get("loaded_count", 0)),
            "selected_source_keys": int(info.get("selected_source_keys", 0)),
            "missing_keys": _summarize_string_list(info.get("missing_keys", [])),
            "unexpected_keys": _summarize_string_list(info.get("unexpected_keys", [])),
            "skipped_shape_mismatch_count": len(info.get("skipped_shape_mismatch", [])),
            "skipped_shape_mismatch_sample": list(info.get("skipped_shape_mismatch", []))[:4],
        }
    return summarized


def _resolve_launch_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def _resolve_aux_device(args: argparse.Namespace) -> str | None:
    raw_value = getattr(args, "aux_device", None)
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    return value or None


_VJEPA_RUNTIME_ARG_NAMES = (
    "vjepa_preset",
    "enable_vjepa_guidance",
    "vjepa_device",
    "vjepa_model",
    "vjepa_ckpt",
    "vjepa_guidance_mode",
    "vjepa_motion_mask_mode",
    "vjepa_guidance_steps",
    "vjepa_min_step_percent",
    "vjepa_max_step_percent",
    "vjepa_target_step_indices",
    "vjepa_target_timesteps",
    "vjepa_latent_step_size",
    "vjepa_inner_k",
    "vjepa_backtracking",
    "vjepa_backtracking_taps",
    "vjepa_line_search_taps",
    "vjepa_preview_downsample_factor",
    "vjepa_preview_frame_stride",
    "vjepa_window_size",
    "vjepa_context_frames",
    "vjepa_stride",
    "vjepa_reduction",
    "vjepa_grad_norm_mode",
    "vjepa_max_grad_norm",
    "vjepa_max_correction_ratio",
    "vjepa_stay_close_max_video_l1",
    "vjepa_artifact_guard_mode",
    "vjepa_use_spectral_guidance",
    "vjepa_spectral_source",
    "vjepa_spectral_lowpass_ratio",
    "vjepa_spectral_normalize_percentile",
    "vjepa_spectral_weight_floor",
    "vjepa_spectral_weight_scale",
    "vjepa_spectral_mask_dilation",
)


def add_vjepa_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--vjepa-preset", type=str, default=None)
    parser.add_argument("--enable-vjepa-guidance", action="store_true")
    parser.add_argument("--vjepa-device", type=str, default=None)
    parser.add_argument("--vjepa-model", type=str, default="vith")
    parser.add_argument("--vjepa-ckpt", default="/data/gaoya/ckpt/VJEPA2/vith.pt")
    parser.add_argument("--vjepa-guidance-mode", choices=["surprise", "context_anchored"], default="context_anchored")
    parser.add_argument(
        "--vjepa-motion-mask-mode",
        choices=["per_frame", "temporal_union", "temporal_union_except_first"],
        default="temporal_union_except_first",
    )
    parser.add_argument("--vjepa-guidance-steps", type=int, default=12)
    parser.add_argument("--vjepa-min-step-percent", type=float, default=0.35)
    parser.add_argument("--vjepa-max-step-percent", type=float, default=0.80)
    parser.add_argument("--vjepa-target-step-indices", type=int, nargs="*", default=None)
    parser.add_argument("--vjepa-target-timesteps", type=int, nargs="*", default=None)
    parser.add_argument("--vjepa-latent-step-size", type=float, default=0.20)
    parser.add_argument("--vjepa-inner-k", type=int, default=1)
    parser.add_argument("--vjepa-backtracking", action="store_true")
    parser.add_argument("--vjepa-backtracking-taps", type=float, nargs="*", default=None)
    parser.add_argument("--vjepa-line-search-taps", type=float, nargs="*", default=None)
    parser.add_argument("--vjepa-preview-downsample-factor", type=int, default=4)
    parser.add_argument("--vjepa-preview-frame-stride", type=int, default=1)
    parser.add_argument("--vjepa-window-size", type=int, default=24)
    parser.add_argument("--vjepa-context-frames", type=int, default=8)
    parser.add_argument("--vjepa-stride", type=int, default=4)
    parser.add_argument("--vjepa-reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--vjepa-grad-norm-mode", choices=["rms", "l2", "none"], default="rms")
    parser.add_argument("--vjepa-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--vjepa-max-correction-ratio", type=float, default=0.05)
    parser.add_argument("--vjepa-stay-close-max-video-l1", type=float, default=0.03)
    parser.add_argument(
        "--vjepa-artifact-guard-mode",
        choices=["none", "video_l1_backoff"],
        default="video_l1_backoff",
    )
    parser.add_argument("--vjepa-use-spectral-guidance", action="store_true")
    parser.add_argument("--vjepa-spectral-source", type=str, default="temporal_lowpass_residual")
    parser.add_argument("--vjepa-spectral-lowpass-ratio", type=float, default=0.18)
    parser.add_argument("--vjepa-spectral-normalize-percentile", type=float, default=95.0)
    parser.add_argument("--vjepa-spectral-weight-floor", type=float, default=0.25)
    parser.add_argument("--vjepa-spectral-weight-scale", type=float, default=1.0)
    parser.add_argument("--vjepa-spectral-mask-dilation", type=int, default=0)
    return parser


def apply_vjepa_preset_if_requested(args: argparse.Namespace) -> None:
    preset_name = getattr(args, "vjepa_preset", None)
    if not preset_name:
        return
    apply_train0705_preset(args, str(preset_name))


def _build_vjepa_config_from_args(args: argparse.Namespace) -> WanVJEPAConfig:
    max_grad_norm = args.vjepa_max_grad_norm
    if max_grad_norm is not None and float(max_grad_norm) <= 0:
        max_grad_norm = None
    max_correction_ratio = args.vjepa_max_correction_ratio
    if max_correction_ratio is not None and float(max_correction_ratio) <= 0:
        max_correction_ratio = None
    stay_close_max_video_l1 = args.vjepa_stay_close_max_video_l1
    if stay_close_max_video_l1 is not None and float(stay_close_max_video_l1) <= 0:
        stay_close_max_video_l1 = None
    return WanVJEPAConfig(
        guidance_steps=int(args.vjepa_guidance_steps),
        min_step_percent=float(args.vjepa_min_step_percent),
        max_step_percent=float(args.vjepa_max_step_percent),
        latent_step_size=float(args.vjepa_latent_step_size),
        preview_downsample_factor=int(args.vjepa_preview_downsample_factor),
        preview_frame_stride=int(args.vjepa_preview_frame_stride),
        window_size=int(args.vjepa_window_size),
        context_frames=int(args.vjepa_context_frames),
        stride=int(args.vjepa_stride),
        reduction=str(args.vjepa_reduction),
        gradient_normalization=str(args.vjepa_grad_norm_mode),
        max_grad_norm=max_grad_norm,
        max_correction_ratio=max_correction_ratio,
        stay_close_max_video_l1=stay_close_max_video_l1,
        artifact_guard_mode=str(args.vjepa_artifact_guard_mode),
        guidance_mode=str(args.vjepa_guidance_mode),
        motion_mask_mode=str(args.vjepa_motion_mask_mode),
        use_spectral_guidance=bool(args.vjepa_use_spectral_guidance),
        spectral_source=str(args.vjepa_spectral_source),
        spectral_lowpass_ratio=float(args.vjepa_spectral_lowpass_ratio),
        spectral_normalize_percentile=float(args.vjepa_spectral_normalize_percentile),
        spectral_weight_floor=float(args.vjepa_spectral_weight_floor),
        spectral_weight_scale=float(args.vjepa_spectral_weight_scale),
        spectral_mask_dilation=int(args.vjepa_spectral_mask_dilation),
    )


def summarize_vjepa_args(args: argparse.Namespace) -> dict | None:
    apply_vjepa_preset_if_requested(args)
    if not bool(getattr(args, "enable_vjepa_guidance", False)):
        return None
    config = _build_vjepa_config_from_args(args)
    return {
        "enabled": True,
        "preset": str(getattr(args, "vjepa_preset", "") or ""),
        "device": str(args.vjepa_device or args.device),
        "model": str(args.vjepa_model),
        "checkpoint": str(args.vjepa_ckpt),
        "target_step_indices": [int(v) for v in (args.vjepa_target_step_indices or [])],
        "target_timesteps": [int(v) for v in (args.vjepa_target_timesteps or [])],
        "inner_k": int(args.vjepa_inner_k),
        "backtracking": bool(args.vjepa_backtracking),
        "backtracking_taps": [float(v) for v in (args.vjepa_backtracking_taps or [])],
        "line_search_taps": [float(v) for v in (args.vjepa_line_search_taps or [])],
        "config": {
            "guidance_mode": str(config.guidance_mode),
            "motion_mask_mode": str(config.motion_mask_mode),
            "guidance_steps": int(config.guidance_steps),
            "min_step_percent": float(config.min_step_percent),
            "max_step_percent": float(config.max_step_percent),
            "latent_step_size": float(config.latent_step_size),
            "preview_downsample_factor": int(config.preview_downsample_factor),
            "preview_frame_stride": int(config.preview_frame_stride),
            "window_size": int(config.window_size),
            "context_frames": int(config.context_frames),
            "stride": int(config.stride),
            "reduction": str(config.reduction),
            "gradient_normalization": str(config.gradient_normalization),
            "max_grad_norm": config.max_grad_norm,
            "max_correction_ratio": config.max_correction_ratio,
            "stay_close_max_video_l1": config.stay_close_max_video_l1,
            "artifact_guard_mode": str(config.artifact_guard_mode),
            "use_spectral_guidance": bool(config.use_spectral_guidance),
            "spectral_source": str(config.spectral_source),
            "spectral_lowpass_ratio": float(config.spectral_lowpass_ratio),
            "spectral_normalize_percentile": float(config.spectral_normalize_percentile),
            "spectral_weight_floor": float(config.spectral_weight_floor),
            "spectral_weight_scale": float(config.spectral_weight_scale),
            "spectral_mask_dilation": int(config.spectral_mask_dilation),
        },
    }


def configure_runtime_pipe_vjepa(pipe: ContextAwareWanVideoPipeline, args: argparse.Namespace) -> None:
    apply_vjepa_preset_if_requested(args)
    if not bool(getattr(args, "enable_vjepa_guidance", False)):
        return
    pipe.configure_vjepa(
        vjepa_model_name=str(args.vjepa_model),
        vjepa_checkpoint_path=Path(args.vjepa_ckpt).expanduser().resolve()
        if args.vjepa_ckpt is not None
        else None,
        vjepa_device=str(args.vjepa_device or args.device),
        vjepa_config=_build_vjepa_config_from_args(args),
        enable_vjepa_guidance=True,
    )
    pipe.configure_target_timesteps([int(value) for value in (args.vjepa_target_timesteps or [])])
    pipe.configure_target_step_indices([int(value) for value in (args.vjepa_target_step_indices or [])])
    pipe.vjepa_inner_k = max(1, int(args.vjepa_inner_k))
    pipe.set_backtracking(
        bool(args.vjepa_backtracking),
        [float(value) for value in (args.vjepa_backtracking_taps or [])] or None,
    )
    pipe.set_line_search_taps([float(value) for value in (args.vjepa_line_search_taps or [])] or None)


def _tensor_video_to_pil_list(context_video_single: torch.Tensor):
    from PIL import Image

    frames = context_video_single.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _build_model_args(args: argparse.Namespace) -> argparse.Namespace:
    parser = t0705.build_parser()
    model_args = parser.parse_args([])
    jepa_tubelet_size = max(1, int(getattr(args, "jepa_tubelet_size", 2)))
    configured_context_frames = max(int(args.context_frames), jepa_tubelet_size)

    model_args.diffsynth_root = str(args.diffsynth_root)
    model_args.wan_root = str(args.wan_root)
    model_args.height = int(args.height)
    model_args.width = int(args.width)
    model_args.num_frames = int(args.num_frames)
    # Keep JEPA on the video backbone path even when inference requests a
    # single context frame. The actual context clip length is still governed by
    # the loaded video frames and may remain 1; this only affects module setup.
    model_args.fixed_num_context_frames = configured_context_frames
    model_args.max_train_steps = 1
    model_args.num_epochs = 1
    model_args.output_path = str(args.output_dir)
    model_args.dataset_type = "wan_ti2v"
    model_args.report_to = "none"
    model_args.initialize_model_on_cpu = bool(getattr(args, "initialize_model_on_cpu", False))

    model_args.lora_base_model = "dit"
    model_args.lora_target_modules = "q,k,v,o,ffn.0,ffn.2"
    model_args.lora_rank = int(args.lora_rank)
    model_args.lora_alpha = int(args.lora_alpha)
    model_args.lora_checkpoint = str(args.lora_checkpoint)
    model_args.extra_inputs = "input_image"

    object_branch_enabled = not bool(getattr(args, "disable_object_branch", False))
    model_args.enable_object_branch = object_branch_enabled
    model_args.freeze_non_object_trainables = object_branch_enabled
    model_args.train_object_adapter = object_branch_enabled
    model_args.train_object_dit_branch = object_branch_enabled
    model_args.train_object_pooler = False
    model_args.train_object_aux_heads = False

    model_args.object_num_queries = int(args.object_num_queries)
    model_args.aux_max_objects = int(args.aux_max_objects)
    model_args.object_pooler_latent_dim = int(args.object_pooler_latent_dim)
    model_args.cond_proj_dim = int(args.cond_proj_dim)
    model_args.jepa_window_radius = int(args.jepa_window_radius)
    model_args.latent_window_radius = int(args.latent_window_radius)
    model_args.object_gate_init = float(args.object_gate_init)

    model_args.lambda_main = 1.0
    model_args.lambda_track_aux = 0.0
    model_args.lambda_box_aux = 0.0
    model_args.lambda_depth_aux = 0.0
    model_args.lambda_object_context_reg = 0.0

    model_args.stage1a_init_from = str(args.stage1a_init_from)
    model_args.jepa_ckpt_path = str(args.jepa_ckpt_path)
    model_args.jepa_input_size = int(args.jepa_input_size)
    model_args.jepa_patch_size = int(args.jepa_patch_size)
    model_args.jepa_tubelet_size = int(args.jepa_tubelet_size)
    model_args.cotracker_checkpoint = str(args.cotracker_checkpoint)
    model_args.cotracker_input_h = int(args.cotracker_input_h)
    model_args.cotracker_input_w = int(args.cotracker_input_w)
    model_args.cotracker_window_len = int(args.cotracker_window_len)
    model_args.vggt_model_path = str(args.vggt_model_path)
    model_args.vggt_input_h = int(args.vggt_input_h)
    model_args.vggt_input_w = int(args.vggt_input_w)
    model_args.vggt_cache_root = None if args.vggt_cache_root is None else str(args.vggt_cache_root)
    model_args.object_aux_devices = _resolve_aux_device(args)

    model_args.grounding_device = None if args.grounding_device is None else str(args.grounding_device)
    model_args.sam2_segment_len = int(args.sam2_segment_len)
    model_args.grounding_proposal_source = str(args.grounding_proposal_source)
    model_args.grounding_motion_score_ratio = float(args.grounding_motion_score_ratio)
    model_args.grounding_text_prompt = str(args.grounding_text_prompt)
    model_args.grounding_extra_prompt_terms = str(args.grounding_extra_prompt_terms)
    model_args.grounding_caption_prompt_mode = str(
        getattr(args, "grounding_caption_prompt_mode", "known_terms")
    )
    model_args.grounding_caption_max_phrases = int(
        getattr(args, "grounding_caption_max_phrases", 4)
    )
    model_args.grounding_caption_min_score = float(
        getattr(args, "grounding_caption_min_score", 4.0)
    )
    model_args.grounding_disable_caption_terms = bool(args.grounding_disable_caption_terms)
    model_args.grounding_gdino_box_threshold = float(args.grounding_gdino_box_threshold)
    model_args.grounding_gdino_text_threshold = float(args.grounding_gdino_text_threshold)
    model_args.grounding_prompt_frame_mode = str(args.grounding_prompt_frame_mode)
    model_args.grounding_track_dedupe_iou_threshold = float(args.grounding_track_dedupe_iou_threshold)
    model_args.grounding_container_suppress_ratio_threshold = float(
        args.grounding_container_suppress_ratio_threshold
    )
    model_args.grounding_container_suppress_min_contained = int(
        args.grounding_container_suppress_min_contained
    )
    model_args.grounding_container_suppress_min_area_ratio = float(
        args.grounding_container_suppress_min_area_ratio
    )
    model_args.grounding_container_suppress_small_iou_threshold = float(
        args.grounding_container_suppress_small_iou_threshold
    )

    return tvn.prepare_args(model_args)


def _build_runtime_model(args: argparse.Namespace):
    apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    accelerator = SimpleNamespace(device=torch.device(args.device))
    model = t0705.build_model(model_args, accelerator)

    if model_args.enable_object_branch:
        stage1a_info = tvn._load_filtered_checkpoint_into_model(
            model,
            model_args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        stage1b_info = tvn._load_filtered_checkpoint_into_model(
            model,
            args.checkpoint,
            include_prefixes=("object_adapter.",),
            include_substrings=(
                "object_embedding",
                ".object_cross_attn.",
                ".object_gate",
                ".norm4.",
            ),
        )
    else:
        stage1a_info = {"skipped": True, "reason": "disable_object_branch"}
        stage1b_info = {"skipped": True, "reason": "disable_object_branch"}
    target_device = torch.device(args.device)
    model.to(target_device)
    model.pipe.to(device=target_device, dtype=model.pipe.torch_dtype)
    for aux_name in ("cotracker_adapter", "jepa_adapter", "vggt_adapter"):
        aux_module = getattr(model, aux_name, None)
        if aux_module is not None and hasattr(aux_module, "device_obj"):
            aux_module.device_obj = target_device
    aux_device = _resolve_aux_device(args)
    if aux_device and getattr(model, "vggt_adapter", None) is not None:
        resolved_aux_device = torch.device(aux_device)
        model.vggt_adapter = model.vggt_adapter.to(resolved_aux_device)
        model.vggt_adapter.device_obj = resolved_aux_device
        if getattr(model.vggt_adapter, "model", None) is not None:
            model.vggt_adapter.model = model.vggt_adapter.model.to(resolved_aux_device)
    model.eval()
    configure_runtime_pipe_vjepa(model.pipe, args)
    return model, model_args, {
        "stage1a_info": stage1a_info,
        "stage1b_info": stage1b_info,
    }


def _encode_context_latents(
    pipe: ContextAwareWanVideoPipeline,
    context_video_single: torch.Tensor,
) -> torch.Tensor:
    context_pil = _tensor_video_to_pil_list(context_video_single)
    preprocessed = pipe.preprocess_video(context_pil)
    return pipe.vae.encode(
        preprocessed,
        device=pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ).to(dtype=pipe.torch_dtype, device=pipe.device)


def _build_object_context(
    model: t0705.ContextOnlyNoGTBoxWanModule,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    if not bool(getattr(model, "enable_object_branch", False)):
        return None, {"enabled": False}

    pipe = model.pipe
    device = torch.device(pipe.device)
    context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

    sample = {
        "context_video": context_video_single,
        "num_context_frames": int(context_video_single.shape[1]),
        "caption": prompt,
        "video_path": video_path,
    }
    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = (
        model._build_object_query_priors(sample, image_hw=image_hw)
    )
    query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)

    frames_bthwc_01 = (
        (context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)
    cotracker_out = model._run_cotracker(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_frame_ids=query_frame_ids,
        query_image_hw=image_hw,
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

    tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model.aux_max_objects,
        points_per_object=model.object_num_queries,
    )
    jepa_out = model._run_jepa(context_video)
    clean_prefix_latents = _encode_context_latents(pipe, context_video_single)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=clean_prefix_latents,
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
        bbox_xyxy=object_out.active_box_xyxy,
    )
    debug = {
        "query_points_shape": list(query_points_prior.shape),
        "query_frame_ids_shape": list(query_frame_ids.shape),
        "object_valid_mask_shape": list(object_valid_mask.shape),
        "object_valid_count": float(object_valid_mask.sum().item()),
        "box_prior_shape": list(box_prior_xyxy.shape),
        "tracks_shape": list(cotracker_out.tracks.shape),
        "object_latent_tokens_shape": list(object_out.object_latent_tokens.shape),
        "object_context_shape": list(object_context.shape),
        "clean_prefix_latents_shape": list(clean_prefix_latents.shape),
    }
    return object_context, debug


def _apply_object_context_ablation(
    object_context: torch.Tensor | None,
    *,
    mode: str = "none",
    random_seed: int | None = None,
    random_scale: float = 1.0,
    slot_count: int | None = None,
    keep_slot_ids: list[int] | tuple[int, ...] | None = None,
    scale_factor: float = 1.0,
    token_norm_max: float | None = None,
) -> tuple[torch.Tensor | None, dict[str, object]]:
    mode_norm = str(mode).strip().lower()
    if object_context is None:
        return None, {
            "mode": mode_norm,
            "applied": False,
            "disabled_object_branch": True,
            "random_seed": None if random_seed is None else int(random_seed),
            "random_scale": float(random_scale),
            "scale_factor": float(scale_factor),
            "token_norm_max": None if token_norm_max is None else float(token_norm_max),
        }

    base = object_context.detach().float()
    debug = {
        "mode": mode_norm,
        "input_shape": list(object_context.shape),
        "input_mean": float(base.mean().item()),
        "input_std": float(base.std(unbiased=False).item()),
        "input_abs_mean": float(base.abs().mean().item()),
        "input_abs_max": float(base.abs().max().item()),
        "random_seed": None if random_seed is None else int(random_seed),
        "random_scale": float(random_scale),
        "slot_count": None if slot_count is None else int(slot_count),
        "keep_slot_ids": None if keep_slot_ids is None else [int(v) for v in keep_slot_ids],
        "scale_factor": float(scale_factor),
        "token_norm_max": None if token_norm_max is None else float(token_norm_max),
    }
    ablation_applied = False
    if mode_norm in ("", "none"):
        ablated = object_context
    else:
        if mode_norm == "zero":
            ablated = torch.zeros_like(object_context)
            ablation_applied = True
        elif mode_norm == "keep_slot":
            if slot_count is None or int(slot_count) <= 0:
                raise ValueError("slot_count must be positive when mode=keep_slot")
            if not keep_slot_ids:
                raise ValueError("keep_slot_ids must be non-empty when mode=keep_slot")
            slot_count = int(slot_count)
            keep_slot_ids = sorted({int(v) for v in keep_slot_ids})
            seq_len = int(object_context.shape[1])
            if seq_len % slot_count != 0:
                raise ValueError(
                    f"object_context sequence length {seq_len} is not divisible by slot_count={slot_count}"
                )
            time_steps = seq_len // slot_count
            invalid = [slot for slot in keep_slot_ids if slot < 0 or slot >= slot_count]
            if invalid:
                raise ValueError(f"keep_slot_ids out of range for slot_count={slot_count}: {invalid}")
            keep_mask = torch.zeros(
                (1, time_steps, slot_count, 1),
                device=object_context.device,
                dtype=object_context.dtype,
            )
            for slot_id in keep_slot_ids:
                keep_mask[:, :, int(slot_id), :] = 1
            keep_mask = keep_mask.view(1, seq_len, 1)
            ablated = object_context * keep_mask
            debug["time_steps"] = int(time_steps)
            ablation_applied = True
        elif mode_norm == "random":
            generator = None
            if random_seed is not None:
                generator = torch.Generator(device=object_context.device)
                generator.manual_seed(int(random_seed))
            random_fp32 = torch.randn(
                tuple(object_context.shape),
                device=object_context.device,
                dtype=torch.float32,
                generator=generator,
            )
            input_mean = float(base.mean().item())
            input_std = float(base.std(unbiased=False).item())
            target_std = input_std if np.isfinite(input_std) and input_std > 1.0e-6 else 1.0
            random_fp32 = random_fp32 * (target_std * float(random_scale)) + input_mean
            ablated = random_fp32.to(device=object_context.device, dtype=object_context.dtype)
            ablation_applied = True
        else:
            raise ValueError(f"unsupported object_context ablation mode: {mode}")

    postprocess_applied = False
    if float(scale_factor) != 1.0:
        ablated = ablated * float(scale_factor)
        postprocess_applied = True

    token_norm_max_value = None if token_norm_max is None else float(token_norm_max)
    if token_norm_max_value is not None:
        if token_norm_max_value <= 0:
            raise ValueError("token_norm_max must be positive when provided")
        ablated_fp32 = ablated.float()
        token_norm = torch.linalg.norm(ablated_fp32, dim=-1, keepdim=True)
        safe_norm = torch.clamp(token_norm, min=1.0e-12)
        scale = torch.clamp(token_norm_max_value / safe_norm, max=1.0)
        ablated = (ablated_fp32 * scale).to(dtype=object_context.dtype)
        postprocess_applied = True
        debug["token_norm_pre_max"] = float(token_norm.max().item())
        debug["token_norm_pre_mean"] = float(token_norm.mean().item())
        token_norm_post = torch.linalg.norm(ablated.float(), dim=-1, keepdim=True)
        debug["token_norm_post_max"] = float(token_norm_post.max().item())
        debug["token_norm_post_mean"] = float(token_norm_post.mean().item())

    ablated_fp32 = ablated.detach().float()
    debug.update(
        {
            "applied": bool(ablation_applied or postprocess_applied),
            "ablation_applied": bool(ablation_applied),
            "postprocess_applied": bool(postprocess_applied),
            "output_mean": float(ablated_fp32.mean().item()),
            "output_std": float(ablated_fp32.std(unbiased=False).item()),
            "output_abs_mean": float(ablated_fp32.abs().mean().item()),
            "output_abs_max": float(ablated_fp32.abs().max().item()),
        }
    )
    return ablated, debug


def _load_context_video(
    *,
    video_path: Path,
    target_context_frames: int,
):
    frames, frame_indices = read_video_prefix(video_path, target_context_frames)
    if int(frames.shape[0]) <= 0:
        raise RuntimeError(f"context video {video_path} does not provide any readable frames")
    if int(frames.shape[0]) > int(target_context_frames):
        frames = frames[:target_context_frames]
        frame_indices = frame_indices[:target_context_frames]
    return frames, frame_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run train0705 stage1b context-only no-GT-box inference on a single context-video case. "
            "Built on DiffSynth Wan2.2 inference with the same object-conditioning path as training."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir or checkpoint.safetensors")
    parser.add_argument("--context-video", required=True, help="Context video mp4")
    parser.add_argument("--prompt", required=True, help="Prompt / caption for generation")
    parser.add_argument("--output-dir", required=True, help="Directory for video + json outputs")
    parser.add_argument("--wan-root", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument(
        "--diffsynth-root",
        default="/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
    )
    parser.add_argument(
        "--lora-checkpoint",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors",
    )
    parser.add_argument(
        "--stage1a-init-from",
        default="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt",
    )
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument("--object-gate-init", type=float, default=0.1)
    parser.add_argument(
        "--jepa-ckpt-path",
        default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
    )
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--vggt-model-path", default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--vggt-input-h", type=int, default=420)
    parser.add_argument("--vggt-input-w", type=int, default=728)
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--aux-device", default=None, help="Optional device for JEPA/CoTracker/VGGT, e.g. cuda:1")
    parser.add_argument("--grounding-device", default=None)
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
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    parser.add_argument(
        "--object-context-ablation",
        choices=["none", "zero", "random"],
        default="none",
        help="Replace the final object_context fed into Wan DiT for ablation.",
    )
    parser.add_argument(
        "--object-context-random-seed",
        type=int,
        default=None,
        help="Optional RNG seed used when --object-context-ablation=random.",
    )
    parser.add_argument(
        "--object-context-random-scale",
        type=float,
        default=1.0,
        help="Std multiplier used when --object-context-ablation=random.",
    )
    parser.add_argument(
        "--object-context-scale-factor",
        type=float,
        default=1.0,
        help="Optional multiplicative factor applied to the final object_context after ablation.",
    )
    parser.add_argument(
        "--object-context-token-norm-max",
        type=float,
        default=None,
        help="Optional per-token L2 norm clamp applied to the final object_context after ablation.",
    )
    add_vjepa_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_vjepa_preset_if_requested(args)
    args.device = _resolve_launch_device()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, model_args, load_info = _build_runtime_model(args)
    pipe = model.pipe
    pipe.dit.eval()

    context_video_path = Path(args.context_video).expanduser().resolve()
    frames, frame_indices = _load_context_video(
        video_path=context_video_path,
        target_context_frames=int(args.context_frames),
    )
    effective_context_frames = int(frames.shape[0])
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
    )
    context_pil = _tensor_video_to_pil_list(context_video_single)

    with torch.no_grad():
        object_context_raw, object_debug = _build_object_context(
            model,
            context_video_single=context_video_single,
            prompt=str(args.prompt),
            video_path=str(context_video_path),
        )
        object_context, ablation_debug = _apply_object_context_ablation(
            object_context_raw,
            mode=str(args.object_context_ablation),
            random_seed=args.object_context_random_seed,
            random_scale=float(args.object_context_random_scale),
            scale_factor=float(args.object_context_scale_factor),
            token_norm_max=args.object_context_token_norm_max,
        )
        object_debug["object_context_ablation"] = ablation_debug
        pipe_kwargs = dict(
            prompt=str(args.prompt),
            negative_prompt="",
            context_video=context_pil,
            seed=int(args.seed),
            tiled=True,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            num_inference_steps=int(args.sampling_steps),
            cfg_scale=float(args.cfg_scale),
        )
        if bool(getattr(model, "enable_object_branch", False)):
            pipe_kwargs["object_context"] = object_context
        video = pipe(**pipe_kwargs)

    checkpoint_path = Path(tvn._resolve_checkpoint_file(args.checkpoint)).resolve()
    checkpoint_tag = checkpoint_path.parent.name
    output_video = output_dir / f"{checkpoint_tag}.mp4"
    save_video(video, str(output_video), fps=int(args.fps), quality=int(args.quality))

    result = {
        "checkpoint": str(checkpoint_path),
        "output_video": str(output_video),
        "context_video": str(context_video_path),
        "prompt": str(args.prompt),
        "frame_indices": frame_indices.tolist(),
        "requested_context_frames": int(args.context_frames),
        "effective_context_frames": effective_context_frames,
        "model_device": str(args.device),
        "aux_device": _resolve_aux_device(args),
        "model_args": {
            "height": int(model_args.height),
            "width": int(model_args.width),
            "num_frames": int(model_args.num_frames),
            "context_frames": effective_context_frames,
            "enable_object_branch": bool(model_args.enable_object_branch),
            "lora_checkpoint": str(model_args.lora_checkpoint),
            "stage1a_init_from": str(model_args.stage1a_init_from),
        },
        "object_context_ablation": {
            "mode": str(args.object_context_ablation),
            "random_seed": args.object_context_random_seed,
            "random_scale": float(args.object_context_random_scale),
            "scale_factor": float(args.object_context_scale_factor),
            "token_norm_max": args.object_context_token_norm_max,
        },
        "vjepa": summarize_vjepa_args(args),
    }
    (output_dir / f"{checkpoint_tag}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
