#!/usr/bin/env python3
"""Batch v2v inference for Scheme-D object-tube checkpoints."""
from __future__ import annotations

import os
import types
from types import SimpleNamespace

import torch

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer_base,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_entity_id_binding_v2v as entity_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch_base,
)
from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler import train


_ORIGINAL_OBJECT_BUILDER = kubric_infer._build_object_context


def _install_scheme_d_model_args(model_args) -> None:
    model_args.tube_num_tokens = int(os.environ.get("SCHEME_D_TUBE_NUM_TOKENS", "4"))
    model_args.tube_hidden_dim = int(os.environ.get("SCHEME_D_TUBE_HIDDEN_DIM", "256"))
    model_args.tube_num_heads = int(os.environ.get("SCHEME_D_TUBE_NUM_HEADS", "8"))
    model_args.tube_num_layers = int(os.environ.get("SCHEME_D_TUBE_NUM_LAYERS", "2"))
    model_args.tube_motion_tokens = int(
        os.environ.get("SCHEME_D_TUBE_MOTION_TOKENS", "4")
    )
    model_args.tube_motion_fourier_bands = int(
        os.environ.get("SCHEME_D_TUBE_MOTION_FOURIER_BANDS", "4")
    )
    model_args.tube_object_attn_dim = int(
        os.environ.get("SCHEME_D_TUBE_OBJECT_ATTN_DIM", "256")
    )
    model_args.tube_object_attn_heads = int(
        os.environ.get("SCHEME_D_TUBE_OBJECT_ATTN_HEADS", "8")
    )
    model_args.tube_latent_dim = int(os.environ.get("SCHEME_D_TUBE_LATENT_DIM", "48"))
    model_args.tube_modality_dropout_prob = 0.0
    model_args.object_block_ids = os.environ.get(
        "SCHEME_D_OBJECT_BLOCK_IDS", "8,11,14,17,20,23"
    )
    model_args.train_object_pooler = False
    model_args.train_object_adapter = False
    model_args.train_object_dit_branch = False


def _build_runtime_model(args):
    infer_base.apply_vjepa_preset_if_requested(args)
    model_args = _build_model_args(args)
    model = train.build_model(
        model_args,
        SimpleNamespace(device=torch.device(args.device)),
    )
    tube_info = train.load_scheme_d_trainables(model, args.checkpoint)
    target_device = torch.device(args.device)
    model.to(target_device)
    model.pipe.to(device=target_device, dtype=model.pipe.torch_dtype)
    for aux_name in ("cotracker_adapter", "jepa_adapter"):
        aux_module = getattr(model, aux_name, None)
        if aux_module is not None and hasattr(aux_module, "device_obj"):
            aux_module.device_obj = target_device
    model.eval()
    infer_base.configure_runtime_pipe_vjepa(model.pipe, args)
    return model, model_args, {
        "stage1a_info": {
            "skipped": True,
            "reason": "Scheme-D has no legacy Stage1A checkpoint",
        },
        "stage1b_info": tube_info,
        "scheme_d_info": tube_info,
    }


def _build_model_args(args):
    model_args = kubric_infer._build_model_args(args)
    _install_scheme_d_model_args(model_args)
    return model_args


def _build_object_context_with_binding(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    if not bool(getattr(model, "enable_object_branch", False)):
        return _ORIGINAL_OBJECT_BUILDER(
            model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
    prompt_context = entity_infer._encode_prompt_context(model.pipe, str(prompt))
    original_query_builder = model._build_object_query_priors
    binding_debug: dict[str, object] = {"enabled": False}

    def query_builder_with_binding(self, sample, *, image_hw):
        nonlocal binding_debug
        outputs = original_query_builder(sample, image_hw=image_hw)
        binding_debug = entity_infer._install_binding_for_grounded_slots(
            self,
            prompt=str(prompt),
            prompt_context=prompt_context,
            object_valid_mask=outputs[2],
        )
        return outputs

    model._build_object_query_priors = types.MethodType(
        query_builder_with_binding,
        model,
    )
    try:
        object_context, debug = _ORIGINAL_OBJECT_BUILDER(
            model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
        binding_debug["adapter_metrics"] = (
            model.object_adapter.pop_entity_binding_metrics()
        )
        diagnostics = model.object_pooler.pop_diagnostics()
        if diagnostics is not None:
            debug["tube_resampler"] = {
                "source_tokens_per_object": diagnostics.source_tokens_per_object,
                "output_tokens_per_object": diagnostics.output_tokens_per_object,
                "motion_tokens_per_object": diagnostics.motion_tokens_per_object,
                "valid_objects": diagnostics.valid_objects,
                "jepa_frames": diagnostics.jepa_frames,
                "latent_frames": diagnostics.latent_frames,
                "track_frames": diagnostics.track_frames,
                "active_dit_blocks": list(model.object_block_ids),
                "vggt_enabled": False,
            }
        debug["entity_id_binding"] = binding_debug
        return object_context, debug
    finally:
        model._build_object_query_priors = original_query_builder
        model.object_adapter.clear_entity_binding_context()


def _install_runtime_hooks() -> None:
    kubric_infer.trainmod = train
    infer_base.t0705 = train
    infer_base._build_model_args = _build_model_args
    infer_base._build_runtime_model = _build_runtime_model
    infer_base._build_object_context = _build_object_context_with_binding


def main() -> None:
    batch_base._install_kubric_runtime_hooks = _install_runtime_hooks
    batch_base.main()


if __name__ == "__main__":
    main()
