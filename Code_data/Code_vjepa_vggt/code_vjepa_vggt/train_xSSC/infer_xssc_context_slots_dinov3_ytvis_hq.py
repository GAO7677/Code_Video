#!/usr/bin/env python3
"""Inference with DINOv3 YTVIS-HQ xSSC slot-512 context slots.

This entry point mirrors ``infer_xssc_context_slots_dinov3.py`` but keeps the
YTVIS-HQ xSSC preprocessing semantics: no MOVi-C/SAM2 AMG pseudo boxes are
used, and frame-0 slots are initialized by ``NormalShared(batch_size)``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
if str(TRAIN_XSSC_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_XSSC_ROOT))

from code_vjepa_vggt.train_xSSC import infer_xssc_context_slots as infer_old
from code_vjepa_vggt.train_xSSC import train_xssc_context_slots_dinov3 as train


infer_base = infer_old.infer_base
batch_base = infer_old.batch_base
_ORIGINAL_BUILD_MODEL_ARGS = infer_old._ORIGINAL_BUILD_MODEL_ARGS

DINOV3_XSSC_ROOT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
DEFAULT_YTVIS_HQ_XSSC_CONFIG = str(
    DINOV3_XSSC_ROOT
    / "upstream/config-randsfq/rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
)
DEFAULT_YTVIS_HQ_XSSC_CHECKPOINT = (
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/"
    "rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512/42/step-015000.pth"
)
DEFAULT_YTVIS_HQ_NUM_SLOTS = 7


def _env(name: str, default) -> str:
    return os.environ.get(name, str(default))


def _copy_dinov3_xssc_defaults(model_args) -> None:
    parser = train.build_parser()
    defaults = parser.parse_args([])
    for action in parser._actions:
        dest = action.dest
        if dest == "help" or hasattr(model_args, dest):
            continue
        if dest.startswith("xssc_") or dest.startswith("dinov3_"):
            setattr(model_args, dest, getattr(defaults, dest))


def _build_model_args(args):
    model_args = _ORIGINAL_BUILD_MODEL_ARGS(args)
    _copy_dinov3_xssc_defaults(model_args)
    model_args.xssc_root = _env("XSSC_ROOT", train.DEFAULT_XSSC_ROOT)
    model_args.xssc_config = _env("XSSC_CONFIG", DEFAULT_YTVIS_HQ_XSSC_CONFIG)
    model_args.xssc_checkpoint = _env(
        "XSSC_CHECKPOINT",
        DEFAULT_YTVIS_HQ_XSSC_CHECKPOINT,
    )
    model_args.xssc_checkpoint_latest_dir = _env(
        "XSSC_CHECKPOINT_LATEST_DIR",
        str(Path(DEFAULT_YTVIS_HQ_XSSC_CHECKPOINT).parent),
    )
    model_args.dinov3_root = _env("DINOV3_ROOT", train.DEFAULT_DINOV3_ROOT)
    model_args.dinov3_checkpoint = _env(
        "DINOV3_CHECKPOINT",
        train.DEFAULT_DINOV3_CHECKPOINT,
    )
    model_args.xssc_box_source = "zeros"
    model_args.xssc_box_cache_dir = ""
    model_args.xssc_filter_empty_amg = False
    model_args.xssc_empty_amg_max_resample_attempts = 0
    model_args.xssc_input_size = int(os.environ.get("XSSC_INPUT_SIZE", "256"))
    model_args.xssc_max_time_steps = int(os.environ.get("XSSC_MAX_TIME_STEPS", "64"))
    model_args.object_lora_rank = int(os.environ.get("OBJECT_LORA_RANK", "32"))
    model_args.object_lora_alpha = float(os.environ.get("OBJECT_LORA_ALPHA", "32"))
    model_args.object_lora_dropout = 0.0
    model_args.xssc_slot_track_dropout = 0.0
    model_args.fixed_num_context_frames = train.base.XSSC_NUM_CONTEXT_FRAMES
    model_args.no_context_ratio = 0.0
    return model_args


@torch.no_grad()
def _extract_ytvis_hq_slots(model, video: torch.Tensor) -> torch.Tensor:
    """Run DINOv3 xSSC recurrence with YTVIS-HQ no-box initialization."""
    model.xssc.eval()
    batch, time_steps, _, _, _ = video.shape
    flat_video = video.flatten(0, 1)
    autocast_enabled = flat_video.device.type == "cuda"
    with torch.autocast(
        device_type=flat_video.device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        feature = model.xssc.encode_backbone(flat_video).detach()
        encoded = feature.permute(0, 2, 3, 1)
        encoded = model.xssc.encode_posit_embed(encoded).flatten(1, 2)
        encoded = model.xssc.encode_project(encoded)
        encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])

        slots = None
        for frame_id in range(time_steps):
            if frame_id == 0:
                query = model.xssc.initializ(batch)
            else:
                query = model.xssc.transit(slots, encoded[:, : frame_id + 1])
            num_iter = None if frame_id == 0 else 1
            current_slots, _ = model.xssc.aggregat(
                encoded[:, frame_id],
                query,
                num_iter=num_iter,
            )
            current_slots = current_slots[:, None]
            slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
    if slots is None:
        raise RuntimeError("xSSC received zero context frames")
    return slots


def _build_runtime_model(args):
    infer_base.apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    expected_slots = int(os.environ.get("XSSC_EXPECTED_NUM_SLOTS", DEFAULT_YTVIS_HQ_NUM_SLOTS))
    previous_expected_slots = train.DINOV3_XSSC_NUM_SLOTS
    train.DINOV3_XSSC_NUM_SLOTS = expected_slots
    try:
        model = train.build_model(
            model_args,
            SimpleNamespace(device=torch.device(args.device)),
        )
    finally:
        train.DINOV3_XSSC_NUM_SLOTS = previous_expected_slots

    model._extract_xssc_slots = MethodType(_extract_ytvis_hq_slots, model)
    checkpoint = train.tvn._resolve_checkpoint_file(args.checkpoint)
    load_info = train.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint,
        include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
        include_substrings=(".object_cross_attn.", ".object_gate"),
    )
    expected_count = sum(1 for _, parameter in model.named_parameters() if parameter.requires_grad)
    if load_info["loaded_count"] != expected_count or load_info["skipped_shape_mismatch"]:
        raise RuntimeError(
            "Incomplete or incompatible DINOv3 YTVIS-HQ xSSC Wan checkpoint: "
            f"loaded={load_info['loaded_count']}/{expected_count}, "
            f"shape_mismatch={len(load_info['skipped_shape_mismatch'])}"
        )

    target_device = torch.device(args.device)
    model.to(target_device)
    model.pipe.to(device=target_device, dtype=model.pipe.torch_dtype)
    model.eval()
    model.aux_max_objects = model.xssc_num_slots
    model.object_adapter = nn.Identity()
    infer_base.configure_runtime_pipe_vjepa(model.pipe, args)
    return model, model_args, {
        "stage1a_info": {
            "skipped": True,
            "reason": "DINOv3 YTVIS-HQ xSSC replaces the Stage1A/Stage1B object frontend",
        },
        "stage1b_info": load_info,
        "xssc_info": load_info,
    }


@torch.no_grad()
def _build_object_context(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    del prompt, video_path
    pipe = model.pipe
    context_video = context_video_single.unsqueeze(0).to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    preprocess_mode = os.environ.get("XSSC_PREPROCESS_MODE", "center_crop")
    xssc_video, preprocess_debug = infer_old._preprocess_xssc_with_mode(
        model,
        context_video,
        mode=preprocess_mode,
    )
    slots = model._extract_xssc_slots(xssc_video)
    perturb_config = infer_old._get_slot_perturb_config()
    slots, perturb_debug = infer_old._apply_slot_perturbation(slots, **perturb_config)
    temporal_mode = os.environ.get("XSSC_SLOT_TEMPORAL_MODE", "full").strip().lower()
    original_time_steps = int(slots.shape[1])
    if temporal_mode in ("", "full"):
        time_ids = torch.arange(original_time_steps, device=model.time_embedding.weight.device)
        temporal_debug = {
            "mode": "full",
            "input_time_steps": original_time_steps,
            "selected_time_indices": list(range(original_time_steps)),
            "time_embedding_ids": list(range(original_time_steps)),
        }
    elif temporal_mode == "last_time0":
        slots = slots[:, -1:, :, :]
        time_ids = torch.zeros(1, device=model.time_embedding.weight.device, dtype=torch.long)
        temporal_debug = {
            "mode": temporal_mode,
            "input_time_steps": original_time_steps,
            "selected_time_indices": [original_time_steps - 1],
            "time_embedding_ids": [0],
        }
    elif temporal_mode == "last_time7":
        slots = slots[:, -1:, :, :]
        time_id = min(train.base.XSSC_NUM_CONTEXT_FRAMES - 1, model.xssc_max_time_steps - 1)
        time_ids = torch.tensor([time_id], device=model.time_embedding.weight.device, dtype=torch.long)
        temporal_debug = {
            "mode": temporal_mode,
            "input_time_steps": original_time_steps,
            "selected_time_indices": [original_time_steps - 1],
            "time_embedding_ids": [int(time_id)],
        }
    else:
        raise ValueError(
            "Unsupported XSSC_SLOT_TEMPORAL_MODE: "
            f"{temporal_mode}. Expected full, last_time0, or last_time7."
        )

    time_steps = int(slots.shape[1])
    if time_steps > model.xssc_max_time_steps:
        raise ValueError(
            f"Context length {time_steps} exceeds xssc_max_time_steps={model.xssc_max_time_steps}"
        )
    target_dtype = model.slot_norm.weight.dtype
    slots_for_projection = slots.to(device=model.slot_norm.weight.device, dtype=target_dtype)
    tokens = model.slot_projector(model.slot_norm(slots_for_projection))
    time_ids = time_ids.to(device=tokens.device)
    time_tokens = model.time_embedding(time_ids).view(1, time_steps, 1, -1)
    tokens = tokens + time_tokens.to(dtype=tokens.dtype)
    batch, _, num_slots, hidden_dim = tokens.shape
    object_context = tokens.reshape(batch, time_steps * num_slots, hidden_dim)

    slots_float = slots.detach().float()
    context_float = object_context.detach().float()
    debug = {
        "enabled": True,
        "variant": "dinov3_ytvis_hq_slot512_no_box",
        "context_video_shape": list(context_video.shape),
        "xssc_video_shape": list(xssc_video.shape),
        "xssc_box_source": "none",
        "xssc_boxes_shape": None,
        "xssc_slots_shape": list(slots.shape),
        "object_context_shape": list(object_context.shape),
        "object_valid_count": float(model.xssc_num_slots),
        "xssc_slots_finite": bool(torch.isfinite(slots_float).all().item()),
        "object_context_finite": bool(torch.isfinite(context_float).all().item()),
        "xssc_slots_abs_mean": float(slots_float.abs().mean().item()),
        "object_context_abs_mean": float(context_float.abs().mean().item()),
        "object_context_abs_max": float(context_float.abs().max().item()),
        "xssc_preprocess": preprocess_debug,
        "xssc_slot_perturbation": perturb_debug,
        "xssc_slot_temporal_mode": temporal_debug,
    }
    if not debug["xssc_slots_finite"] or not debug["object_context_finite"]:
        raise FloatingPointError(f"non-finite DINOv3 YTVIS-HQ xSSC conditioning values: {debug}")
    return object_context, debug


def _install_runtime_hooks() -> None:
    infer_base.t0705 = train
    infer_base._build_model_args = _build_model_args
    infer_base._build_runtime_model = _build_runtime_model
    infer_base._build_object_context = _build_object_context


def main() -> None:
    batch_base._install_kubric_runtime_hooks = _install_runtime_hooks
    batch_base.main()


if __name__ == "__main__":
    main()
