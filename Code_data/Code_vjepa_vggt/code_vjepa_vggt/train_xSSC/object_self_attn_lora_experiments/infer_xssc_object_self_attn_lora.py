#!/usr/bin/env python3
"""Config-bound inference for xSSC object/self-attention LoRA experiments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(TRAIN_XSSC_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_XSSC_ROOT))

import infer_xssc_context_slots_dinov3 as infer_dinov3
import train_xssc_object_self_attn_lora as train
import train_xssc_object_self_attn_lora_slot_dedup as train_slot_dedup


infer_base = infer_dinov3.infer_base
batch_base = infer_dinov3.batch_base
_ORIGINAL_BUILD_MODEL_ARGS = infer_dinov3._ORIGINAL_BUILD_MODEL_ARGS
_MANIFEST_CACHE: dict[Path, dict] = {}


def _resolve_experiment_manifest(checkpoint: str | os.PathLike) -> Path:
    configured = os.environ.get("EXPERIMENT_CONFIG", "").strip()
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EXPERIMENT_CONFIG does not exist: {path}")
        return path

    checkpoint_file = train.tvn._resolve_checkpoint_file(checkpoint).resolve()
    candidates = [
        parent / "resolved_experiment_config.json"
        for parent in checkpoint_file.parents
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find resolved_experiment_config.json above checkpoint "
        f"{checkpoint_file}. Set EXPERIMENT_CONFIG explicitly."
    )


def _load_resolved_config(checkpoint: str | os.PathLike) -> tuple[dict, Path]:
    manifest_path = _resolve_experiment_manifest(checkpoint)
    if manifest_path not in _MANIFEST_CACHE:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("resolved_config")
        if not isinstance(config, dict):
            raise ValueError(
                f"Manifest has no resolved_config object: {manifest_path}"
            )
        mode = config.get("adaptation", {}).get("mode")
        if mode not in train.SELF_ATTN_ADAPTATION_MODES:
            raise ValueError(
                f"Unsupported adaptation mode {mode!r} in {manifest_path}"
            )
        _MANIFEST_CACHE[manifest_path] = config
    return _MANIFEST_CACHE[manifest_path], manifest_path


def _apply_config_to_model_args(model_args, config: dict) -> None:
    paths = config["paths"]
    model = config["model"]
    adaptation = config["adaptation"]
    conditioning = config["conditioning"]
    filters = conditioning["amg_filters"]

    model_args.wan_root = paths["wan_root"]
    model_args.lora_checkpoint = paths["pretrained_lora_checkpoint"]
    model_args.lora_target_modules = model["pretrained_lora_target_modules"]
    model_args.lora_rank = int(model["pretrained_lora_rank"])
    model_args.lora_alpha = float(model["pretrained_lora_alpha"])
    model_args.xssc_root = paths["xssc_root"]
    model_args.xssc_config = paths["xssc_config"]
    model_args.xssc_checkpoint = paths["xssc_checkpoint"]
    model_args.xssc_checkpoint_latest_dir = str(
        Path(paths["xssc_checkpoint"]).parent
    )
    model_args.dinov3_root = paths["dinov3_root"]
    model_args.dinov3_checkpoint = paths["dinov3_checkpoint"]
    model_args.xssc_sam2_config = paths["sam2_config"]
    model_args.xssc_sam2_checkpoint = paths["sam2_checkpoint"]
    model_args.xssc_box_source = conditioning["xssc_box_source"]
    model_args.xssc_box_cache_dir = os.environ.get(
        "XSSC_BOX_CACHE_DIR",
        paths["xssc_box_cache_dir"],
    )
    model_args.xssc_filter_empty_amg = False
    model_args.xssc_empty_amg_max_resample_attempts = 0
    model_args.xssc_input_size = int(model["xssc_input_size"])
    model_args.xssc_max_time_steps = int(model["xssc_max_time_steps"])
    model_args.fixed_num_context_frames = int(model["fixed_num_context_frames"])
    model_args.no_context_ratio = 0.0

    model_args.object_lora_rank = int(adaptation["object_lora_rank"])
    model_args.object_lora_alpha = float(adaptation["object_lora_alpha"])
    model_args.object_lora_dropout = float(adaptation["object_lora_dropout"])
    model_args.xssc_slot_track_dropout = 0.0
    model_args.self_attn_adaptation_mode = adaptation["mode"]
    model_args.pretrained_lora_expected_modules = int(
        model["pretrained_lora_expected_modules"]
    )
    model_args.self_attn_expected_num_blocks = int(
        model["self_attn_expected_num_blocks"]
    )
    model_args.self_attn_expected_num_heads = int(
        model["self_attn_expected_num_heads"]
    )
    model_args.self_attn_lora_rank = int(adaptation["self_attn_lora_rank"])
    model_args.self_attn_lora_alpha = float(adaptation["self_attn_lora_alpha"])
    model_args.self_attn_lora_dropout = float(
        adaptation["self_attn_lora_dropout"]
    )
    model_args.head_selection_config = paths["head_selection_config"]
    model_args.head_selection_subset_id = adaptation["head_selection_subset_id"]
    model_args.head_selection_expected_role = adaptation[
        "head_selection_expected_role"
    ]
    model_args.head_selection_feature_subtype = adaptation[
        "head_selection_feature_subtype"
    ]
    model_args.head_selection_expected_num_heads = int(
        adaptation["head_selection_expected_num_heads"]
    )
    model_args.object_gate_init = float(adaptation["object_gate_init"])
    model_args.lambda_main = float(conditioning["lambda_main"])
    model_args.lambda_object_context_reg = float(
        conditioning["lambda_object_context_reg"]
    )

    slot_dedup = conditioning.get("slot_dedup", {})
    model_args.xssc_slot_dedup_mode = str(slot_dedup.get("mode", "none"))
    model_args.xssc_slot_dedup_similarity_threshold = float(
        slot_dedup.get("similarity_threshold", 0.94)
    )
    model_args.xssc_slot_dedup_similarity_metric = str(
        slot_dedup.get("similarity_metric", "mean_frame_cosine")
    )
    model_args.xssc_slot_dedup_min_keep = int(slot_dedup.get("min_keep", 1))

    for key, value in filters.items():
        setattr(model_args, f"xssc_amg_{key}", value)


def _build_model_args(args):
    model_args = _ORIGINAL_BUILD_MODEL_ARGS(args)
    config, manifest_path = _load_resolved_config(args.checkpoint)
    _apply_config_to_model_args(model_args, config)
    model_args._experiment_manifest_path = str(manifest_path)
    return model_args


def _build_runtime_model(args):
    infer_base.apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    config, manifest_path = _load_resolved_config(args.checkpoint)
    slot_dedup_mode = str(
        config.get("conditioning", {}).get("slot_dedup", {}).get("mode", "none")
    )
    train_impl = train_slot_dedup if slot_dedup_mode != "none" else train
    model = train_impl.build_model(
        model_args,
        SimpleNamespace(device=torch.device(args.device)),
    )
    checkpoint = train.tvn._resolve_checkpoint_file(args.checkpoint)
    identity_info = None
    if (
        model.self_attn_adaptation_mode
        in train.HEAD_SELECTIVE_ADAPTATION_MODES
    ):
        identity_info = train.validate_head_selection_resume_checkpoint(
            model,
            checkpoint,
        )

    load_info = train.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint,
        include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
        include_substrings=(
            ".object_cross_attn.",
            ".object_gate",
            ".self_attn.",
        ),
    )
    expected_count = sum(
        1 for _, parameter in model.named_parameters() if parameter.requires_grad
    )
    if (
        load_info["loaded_count"] != expected_count
        or load_info["skipped_shape_mismatch"]
    ):
        raise RuntimeError(
            "Incomplete or incompatible object/self-attention checkpoint: "
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
    runtime_info = {
        "stage1a_info": {
            "skipped": True,
            "reason": "DINOv3 xSSC replaces the legacy object frontend",
        },
        "stage1b_info": load_info,
        "xssc_info": load_info,
        "experiment_info": {
            "manifest": str(manifest_path),
            "name": config["experiment"]["name"],
            "adaptation_mode": config["adaptation"]["mode"],
            "slot_dedup_mode": slot_dedup_mode,
            "slot_dedup": config.get("conditioning", {}).get("slot_dedup"),
            "head_identity": identity_info,
            "pretrained_lora_checkpoint": config["paths"][
                "pretrained_lora_checkpoint"
            ],
        },
    }
    return model, model_args, runtime_info


@torch.no_grad()
def _build_object_context(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    """Build inference tokens with the same optional slot de-dup as training."""
    del prompt, video_path
    pipe = model.pipe
    context_video = context_video_single.unsqueeze(0).to(
        device=pipe.device,
        dtype=pipe.torch_dtype,
    )
    preprocess_mode = os.environ.get("XSSC_PREPROCESS_MODE", "center_crop")
    xssc_video, preprocess_debug = infer_dinov3.infer_old._preprocess_xssc_with_mode(
        model,
        context_video,
        mode=preprocess_mode,
    )
    boxes = model._build_xssc_boxes(xssc_video)
    slots = model._extract_xssc_slots(xssc_video, boxes)
    perturb_config = infer_dinov3.infer_old._get_slot_perturb_config()
    slots, perturb_debug = infer_dinov3.infer_old._apply_slot_perturbation(
        slots,
        **perturb_config,
    )

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
        time_id = min(
            infer_dinov3.train.base.XSSC_NUM_CONTEXT_FRAMES - 1,
            model.xssc_max_time_steps - 1,
        )
        time_ids = torch.tensor(
            [time_id],
            device=model.time_embedding.weight.device,
            dtype=torch.long,
        )
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

    slot_dedup_mode = str(getattr(model, "xssc_slot_dedup_mode", "none"))
    slots, slot_keep_mask, dedup_stats = train_slot_dedup.deduplicate_xssc_slot_tracks(
        slots,
        mode=slot_dedup_mode,
        threshold=float(getattr(model, "xssc_slot_dedup_similarity_threshold", 0.94)),
        similarity_metric=str(
            getattr(model, "xssc_slot_dedup_similarity_metric", "mean_frame_cosine")
        ),
        min_keep=int(getattr(model, "xssc_slot_dedup_min_keep", 1)),
    )
    time_steps = int(slots.shape[1])
    if time_steps > model.xssc_max_time_steps:
        raise ValueError(
            f"Context length {time_steps} exceeds xssc_max_time_steps={model.xssc_max_time_steps}"
        )
    target_dtype = model.slot_norm.weight.dtype
    slots_for_projection = slots.to(device=model.slot_norm.weight.device, dtype=target_dtype)
    tokens = model.slot_projector(model.slot_norm(slots_for_projection))
    time_tokens = model.time_embedding(time_ids.to(device=tokens.device)).view(
        1,
        time_steps,
        1,
        -1,
    )
    tokens = tokens + time_tokens.to(dtype=tokens.dtype)
    tokens = train_slot_dedup._apply_slot_track_dropout_to_available_tokens(
        model,
        tokens,
        slot_keep_mask.to(device=tokens.device),
    )
    batch, _, num_slots, hidden_dim = tokens.shape
    object_context = tokens.reshape(batch, time_steps * num_slots, hidden_dim)

    slots_float = slots.detach().float()
    context_float = object_context.detach().float()
    boxes_float = boxes.detach().float()
    debug = {
        "enabled": True,
        "context_video_shape": list(context_video.shape),
        "xssc_video_shape": list(xssc_video.shape),
        "xssc_boxes_shape": list(boxes.shape),
        "xssc_boxes_abs_mean": float(boxes_float.abs().mean().item()),
        "xssc_amg_selected_counts": list(
            getattr(model, "_last_xssc_amg_selected_counts", [])
        ),
        "xssc_slots_shape": list(slots.shape),
        "object_context_shape": list(object_context.shape),
        "object_valid_count": float(slot_keep_mask.float().sum(dim=1).mean().item()),
        "xssc_slots_finite": bool(torch.isfinite(slots_float).all().item()),
        "object_context_finite": bool(torch.isfinite(context_float).all().item()),
        "xssc_slots_abs_mean": float(slots_float.abs().mean().item()),
        "object_context_abs_mean": float(context_float.abs().mean().item()),
        "object_context_abs_max": float(context_float.abs().max().item()),
        "xssc_preprocess": preprocess_debug,
        "xssc_slot_perturbation": perturb_debug,
        "xssc_slot_temporal_mode": temporal_debug,
        "xssc_slot_dedup": {"mode": slot_dedup_mode, **dedup_stats},
    }
    if not debug["xssc_slots_finite"] or not debug["object_context_finite"]:
        raise FloatingPointError(f"non-finite DINOv3 xSSC conditioning values: {debug}")
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
