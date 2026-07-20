#!/usr/bin/env python3
"""Batch inference for xSSC context-slot object-conditioning checkpoints."""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer_base,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
)
from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train


_ORIGINAL_BUILD_MODEL_ARGS = infer_base._build_model_args


def _normalize_xssc_frames(frames: torch.Tensor, model) -> torch.Tensor:
    frames = (frames + 1.0).mul(127.5).clamp(0.0, 255.0)
    mean = frames.new_tensor(train.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = frames.new_tensor(train.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    return (frames - mean) / std


def _preprocess_xssc_with_mode(
    model,
    context_video: torch.Tensor,
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    mode_norm = str(mode).strip().lower()
    aliases = {
        "": "center_crop",
        "center": "center_crop",
        "center_crop": "center_crop",
        "left": "left_crop",
        "left_crop": "left_crop",
        "right": "right_crop",
        "right_crop": "right_crop",
        "pad": "resize_pad_square",
        "letterbox": "resize_pad_square",
        "resize_pad": "resize_pad_square",
        "resize_pad_square": "resize_pad_square",
    }
    if mode_norm not in aliases:
        raise ValueError(
            "Unsupported XSSC_PREPROCESS_MODE="
            f"{mode}. Expected center_crop, left_crop, right_crop, resize_pad_square."
        )
    mode_norm = aliases[mode_norm]
    frames = context_video.permute(0, 2, 1, 3, 4).float()
    batch, time_steps, channels, height, width = frames.shape
    input_size = int(model.xssc_input_size)
    debug: dict[str, object] = {
        "mode": mode_norm,
        "input_context_shape": list(context_video.shape),
        "source_hw_after_wan_cover_crop": [int(height), int(width)],
        "xssc_input_size": input_size,
    }
    if mode_norm == "center_crop":
        xssc_video = model._preprocess_xssc(context_video)
        debug["crop_yxhw"] = [
            int((int(height) - min(int(height), int(width))) // 2),
            int((int(width) - min(int(height), int(width))) // 2),
            int(min(int(height), int(width))),
            int(min(int(height), int(width))),
        ]
        return xssc_video, debug

    if mode_norm in {"left_crop", "right_crop"}:
        crop_size = min(int(height), int(width))
        top = (int(height) - crop_size) // 2
        left = 0 if mode_norm == "left_crop" else int(width) - crop_size
        cropped = frames[..., top : top + crop_size, left : left + crop_size]
        flat = cropped.reshape(batch * time_steps, channels, crop_size, crop_size)
        flat = F.interpolate(
            flat,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        flat = _normalize_xssc_frames(flat, model)
        debug["crop_yxhw"] = [int(top), int(left), int(crop_size), int(crop_size)]
        return flat.view(batch, time_steps, channels, input_size, input_size), debug

    flat = frames.reshape(batch * time_steps, channels, int(height), int(width))
    scale = min(input_size / float(height), input_size / float(width))
    resized_h = max(1, int(round(float(height) * scale)))
    resized_w = max(1, int(round(float(width) * scale)))
    resized = F.interpolate(
        flat,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    padded = resized.new_full(
        (batch * time_steps, channels, input_size, input_size),
        -1.0,
    )
    pad_top = (input_size - resized_h) // 2
    pad_left = (input_size - resized_w) // 2
    padded[:, :, pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    padded = _normalize_xssc_frames(padded, model)
    debug.update(
        {
            "resize_hw": [int(resized_h), int(resized_w)],
            "pad_top_left": [int(pad_top), int(pad_left)],
            "pad_value_before_imagenet_norm": "black(-1 in [-1,1], 0 in uint8)",
        }
    )
    return padded.view(batch, time_steps, channels, input_size, input_size), debug


def _get_slot_perturb_config() -> dict[str, object]:
    mode = os.environ.get("XSSC_SLOT_PERTURB", "none").strip().lower()
    seed_text = os.environ.get("XSSC_SLOT_PERTURB_SEED", "").strip()
    return {
        "mode": mode,
        "seed": None if not seed_text else int(seed_text),
        "noise_std": float(os.environ.get("XSSC_SLOT_NOISE_STD", "1.0")),
        "drop_prob": float(os.environ.get("XSSC_SLOT_DROP_PROB", "0.5")),
    }


def _apply_slot_perturbation(
    slots: torch.Tensor,
    *,
    mode: str,
    seed: int | None,
    noise_std: float,
    drop_prob: float,
) -> tuple[torch.Tensor, dict[str, object]]:
    mode_norm = str(mode).strip().lower()
    before = slots.detach().float()
    debug: dict[str, object] = {
        "mode": mode_norm,
        "seed": None if seed is None else int(seed),
        "noise_std": float(noise_std),
        "drop_prob": float(drop_prob),
        "input_abs_mean": float(before.abs().mean().item()),
        "input_abs_max": float(before.abs().max().item()),
    }
    if mode_norm in ("", "none"):
        debug["applied"] = False
        return slots, debug

    generator = None
    if seed is not None:
        generator = torch.Generator(device=slots.device)
        generator.manual_seed(int(seed))

    if mode_norm == "zero":
        perturbed = torch.zeros_like(slots)
    elif mode_norm == "noise":
        noise = torch.randn(
            tuple(slots.shape),
            device=slots.device,
            dtype=torch.float32,
            generator=generator,
        )
        input_std = before.std(unbiased=False).clamp_min(1.0e-6)
        noise = noise * (input_std * float(noise_std))
        perturbed = (slots.float() + noise).to(dtype=slots.dtype)
    elif mode_norm == "shuffle_time":
        time_steps = int(slots.shape[1])
        order = torch.randperm(time_steps, device=slots.device, generator=generator)
        perturbed = slots[:, order]
        debug["time_order"] = [int(v) for v in order.detach().cpu().tolist()]
    elif mode_norm == "shuffle_slot":
        slot_count = int(slots.shape[2])
        order = torch.randperm(slot_count, device=slots.device, generator=generator)
        perturbed = slots[:, :, order]
        debug["slot_order"] = [int(v) for v in order.detach().cpu().tolist()]
    elif mode_norm == "drop_slot":
        if not 0.0 <= float(drop_prob) < 1.0:
            raise ValueError(f"XSSC_SLOT_DROP_PROB must be in [0, 1), got {drop_prob}")
        keep = torch.rand(
            (int(slots.shape[0]), int(slots.shape[2])),
            device=slots.device,
            generator=generator,
        ) >= float(drop_prob)
        empty_rows = ~keep.any(dim=1)
        if bool(empty_rows.any()):
            replacement = torch.randint(
                int(slots.shape[2]),
                (int(empty_rows.sum().item()),),
                device=slots.device,
                generator=generator,
            )
            keep[empty_rows, replacement] = True
        perturbed = slots * keep[:, None, :, None].to(dtype=slots.dtype)
        debug["actual_drop_fraction"] = float((~keep).float().mean().item())
    else:
        raise ValueError(
            "Unsupported XSSC_SLOT_PERTURB mode: "
            f"{mode}. Expected none, zero, noise, shuffle_time, shuffle_slot, or drop_slot."
        )

    after = perturbed.detach().float()
    debug.update(
        {
            "applied": True,
            "output_abs_mean": float(after.abs().mean().item()),
            "output_abs_max": float(after.abs().max().item()),
            "mean_abs_delta": float((after - before).abs().mean().item()),
        }
    )
    return perturbed, debug


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
    preprocess_mode = os.environ.get("XSSC_PREPROCESS_MODE", "center_crop")
    xssc_video, preprocess_debug = _preprocess_xssc_with_mode(
        model,
        context_video,
        mode=preprocess_mode,
    )
    slots = model._extract_xssc_slots(xssc_video)
    perturb_config = _get_slot_perturb_config()
    slots, perturb_debug = _apply_slot_perturbation(slots, **perturb_config)
    time_steps = int(slots.shape[1])
    if time_steps > model.xssc_max_time_steps:
        raise ValueError(
            f"Context length {time_steps} exceeds xssc_max_time_steps={model.xssc_max_time_steps}"
        )
    target_dtype = model.slot_norm.weight.dtype
    slots_for_projection = slots.to(device=model.slot_norm.weight.device, dtype=target_dtype)
    tokens = model.slot_projector(model.slot_norm(slots_for_projection))
    time_ids = torch.arange(time_steps, device=tokens.device)
    time_tokens = model.time_embedding(time_ids).view(1, time_steps, 1, -1)
    tokens = tokens + time_tokens.to(dtype=tokens.dtype)
    batch, _, num_slots, hidden_dim = tokens.shape
    object_context = tokens.reshape(batch, time_steps * num_slots, hidden_dim)
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
        "xssc_preprocess": preprocess_debug,
        "xssc_slot_perturbation": perturb_debug,
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
