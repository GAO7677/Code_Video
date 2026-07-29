#!/usr/bin/env python3
"""Config-bound inference for Object-only, Full-SA, S-head, and T-head runs."""

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
    model = train.build_model(
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
            "head_identity": identity_info,
            "pretrained_lora_checkpoint": config["paths"][
                "pretrained_lora_checkpoint"
            ],
        },
    }
    return model, model_args, runtime_info


def _install_runtime_hooks() -> None:
    infer_base.t0705 = train
    infer_base._build_model_args = _build_model_args
    infer_base._build_runtime_model = _build_runtime_model
    infer_base._build_object_context = infer_dinov3._build_object_context


def main() -> None:
    batch_base._install_kubric_runtime_hooks = _install_runtime_hooks
    batch_base.main()


if __name__ == "__main__":
    main()
