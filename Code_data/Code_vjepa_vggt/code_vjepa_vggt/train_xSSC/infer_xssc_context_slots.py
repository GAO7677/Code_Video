#!/usr/bin/env python3
"""Batch inference for xSSC context-slot object-conditioning checkpoints."""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer_base,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
)
from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train


_ORIGINAL_BUILD_MODEL_ARGS = infer_base._build_model_args


def _build_model_args(args):
    model_args = _ORIGINAL_BUILD_MODEL_ARGS(args)
    model_args.xssc_root = os.environ.get("XSSC_ROOT", train.DEFAULT_XSSC_ROOT)
    model_args.xssc_config = os.environ.get("XSSC_CONFIG", train.DEFAULT_XSSC_CONFIG)
    model_args.xssc_checkpoint = os.environ.get(
        "XSSC_CHECKPOINT", train.DEFAULT_XSSC_CHECKPOINT
    )
    model_args.xssc_input_size = int(os.environ.get("XSSC_INPUT_SIZE", "256"))
    model_args.xssc_max_time_steps = int(os.environ.get("XSSC_MAX_TIME_STEPS", "64"))
    model_args.object_lora_rank = int(os.environ.get("OBJECT_LORA_RANK", "32"))
    model_args.object_lora_alpha = float(os.environ.get("OBJECT_LORA_ALPHA", "32"))
    model_args.object_lora_dropout = 0.0
    model_args.xssc_slot_track_dropout = 0.0
    model_args.fixed_num_context_frames = train.XSSC_NUM_CONTEXT_FRAMES
    model_args.no_context_ratio = 0.0
    return model_args


def _build_runtime_model(args):
    infer_base.apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    model = train.build_model(
        model_args,
        SimpleNamespace(device=torch.device(args.device)),
    )
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
            "Incomplete or incompatible xSSC checkpoint: "
            f"loaded={load_info['loaded_count']}/{expected_count}, "
            f"shape_mismatch={len(load_info['skipped_shape_mismatch'])}"
        )

    target_device = torch.device(args.device)
    model.to(target_device)
    model.pipe.to(device=target_device, dtype=model.pipe.torch_dtype)
    model.eval()
    model.aux_max_objects = model.xssc_num_slots
    # The shared batch runner configures one legacy adapter-only runtime knob.
    # xSSC has no adapter MLP, so expose a harmless placeholder for that contract.
    model.object_adapter = nn.Identity()
    infer_base.configure_runtime_pipe_vjepa(model.pipe, args)
    return model, model_args, {
        "stage1a_info": {
            "skipped": True,
            "reason": "xSSC replaces the Stage1A/Stage1B object frontend",
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
    object_context, slots = model._build_object_context(context_video)
    slots_float = slots.detach().float()
    context_float = object_context.detach().float()
    debug = {
        "enabled": True,
        "context_video_shape": list(context_video.shape),
        "xssc_slots_shape": list(slots.shape),
        "object_context_shape": list(object_context.shape),
        "object_valid_count": float(model.xssc_num_slots),
        "xssc_slots_finite": bool(torch.isfinite(slots_float).all().item()),
        "object_context_finite": bool(torch.isfinite(context_float).all().item()),
        "xssc_slots_abs_mean": float(slots_float.abs().mean().item()),
        "object_context_abs_mean": float(context_float.abs().mean().item()),
        "object_context_abs_max": float(context_float.abs().max().item()),
    }
    if not debug["xssc_slots_finite"] or not debug["object_context_finite"]:
        raise FloatingPointError(f"non-finite xSSC conditioning values: {debug}")
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
