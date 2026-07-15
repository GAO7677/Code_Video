#!/usr/bin/env python3
"""Train Scheme-D learned object-tube tokens with replay preservation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

import code_vjepa_vggt.train_v_newtrain as tvn
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_base
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve as replay
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve_entity_id_binding as entity_train
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig
from code_vjepa_vggt.models.object_entity_id_binder import (
    EntityIDBindingObjectConditionAdapter,
)
from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler.models import (
    ObjectTubeResampler,
    install_bottleneck_object_cross_attention,
    parse_block_ids,
)

from diffsynth.diffusion import ModelLogger


# The existing JSON inference builder swaps its ``trainmod`` reference to this
# module and expects these data-path helpers on that module.
prepare_jepa_context_video = kubric_base.prepare_jepa_context_video
compact_object_context_valid_slots = kubric_base.compact_object_context_valid_slots


class SchemeDObjectTubeWanModule(entity_train.EntityIDBindingReplayPreserveWanModule):
    def __init__(
        self,
        *args,
        tube_num_tokens: int = 4,
        tube_hidden_dim: int = 256,
        tube_num_heads: int = 8,
        tube_num_layers: int = 2,
        tube_motion_tokens: int = 4,
        tube_motion_fourier_bands: int = 4,
        tube_object_attn_dim: int = 256,
        tube_object_attn_heads: int = 8,
        tube_latent_dim: int = 48,
        tube_modality_dropout_prob: float = 0.10,
        object_block_ids: str = "8,11,14,17,20,23",
        debug_print_tube_shapes: bool = False,
        tube_shape_trace_path: str | None = None,
        **kwargs,
    ) -> None:
        # A non-empty cache path prevents the legacy constructor from loading
        # live VGGT. This project never reads that sentinel after construction.
        kwargs["vggt_cache_root"] = "/__scheme_d_vggt_disabled__"
        kwargs["train_vggt"] = False
        # Build all temporary parent object modules at the final Scheme-D width
        # instead of allocating the legacy 4096-dimensional adapter first.
        kwargs["cond_proj_dim"] = int(tube_hidden_dim)
        super().__init__(*args, **kwargs)
        if not self.enable_object_branch:
            return

        self.vggt_cache_root = None
        self.vggt_runner = None
        self.vggt_adapter = None
        old_pooler = self.object_pooler
        old_adapter = self.entity_bound_adapter
        device = next(old_pooler.parameters()).device
        jepa_dim = int(old_pooler.jepa_proj.in_features)
        object_dim = int(tube_hidden_dim)
        text_embedding = getattr(self.pipe.dit, "text_embedding", None)
        text_linear = (
            next(
                (
                    module
                    for module in text_embedding.modules()
                    if isinstance(module, nn.Linear)
                ),
                None,
            )
            if isinstance(text_embedding, nn.Module)
            else None
        )
        if text_linear is None:
            raise RuntimeError("cannot infer Wan T5 context dimension")
        entity_text_dim = int(text_linear.in_features)

        self.object_pooler = ObjectTubeResampler(
            jepa_dim=jepa_dim,
            latent_dim=int(tube_latent_dim),
            output_dim=object_dim,
            hidden_dim=int(tube_hidden_dim),
            num_output_tokens=int(tube_num_tokens),
            num_heads=int(tube_num_heads),
            num_layers=int(tube_num_layers),
            num_motion_tokens=int(tube_motion_tokens),
            motion_fourier_bands=int(tube_motion_fourier_bands),
            max_objects=int(self.aux_max_objects),
            max_points=int(self.object_num_queries),
            modality_dropout_prob=float(tube_modality_dropout_prob),
            min_box_px=float(getattr(old_pooler, "min_box_px", 16.0)),
        ).to(device=device)
        self.object_adapter = EntityIDBindingObjectConditionAdapter(
            dim=object_dim,
            num_slots=int(self.aux_max_objects),
            max_time_steps=max(int(tube_num_tokens), 8),
            output_gate_init=float(torch.sigmoid(old_adapter.output_gate_logit.detach()).item()),
            entity_text_dim=entity_text_dim,
            entity_bottleneck_dim=int(self.entity_binding_bottleneck_dim),
            entity_gate_init=float(self.entity_binding_gate_init),
            entity_dropout_prob=float(self.entity_binding_dropout_prob),
            entity_residual_max_ratio=float(self.entity_binding_residual_max_ratio),
        ).to(device=device)
        self.object_adapter.mlp_residual_max_ratio = (
            float(self.object_adapter_mlp_residual_max_ratio)
            if float(self.object_adapter_mlp_residual_max_ratio) > 0.0
            else None
        )
        self.object_aux_heads = nn.Identity().to(device=device)
        self.tube_num_tokens = int(tube_num_tokens)
        self.tube_motion_tokens = int(tube_motion_tokens)
        self.tube_motion_fourier_bands = int(tube_motion_fourier_bands)
        self.tube_object_attn_dim = int(tube_object_attn_dim)
        self.tube_object_attn_heads = int(tube_object_attn_heads)
        self.tube_latent_dim = int(tube_latent_dim)
        self.object_block_ids = parse_block_ids(
            object_block_ids,
            num_blocks=len(self.pipe.dit.blocks),
        )
        self.object_block_layout = install_bottleneck_object_cross_attention(
            self.pipe.dit,
            self.object_block_ids,
            object_dim=object_dim,
            inner_dim=self.tube_object_attn_dim,
            num_heads=self.tube_object_attn_heads,
        )
        for name, param in self.pipe.dit.named_parameters():
            if ".object_cross_attn." in name or ".object_gate" in name or ".norm4." in name:
                param.requires_grad = bool(self.train_object_dit_branch)
        self.object_pooler.requires_grad_(bool(self.train_object_pooler))
        self.object_adapter.requires_grad_(bool(self.train_object_adapter))
        self.debug_print_tube_shapes = bool(debug_print_tube_shapes)
        self.tube_shape_trace_path = (
            None
            if tube_shape_trace_path is None
            else Path(tube_shape_trace_path).expanduser().resolve()
        )
        self._tube_forward_index = 0

    def _run_vggt(self, *args, **kwargs):
        return None

    def _run_main_loss_with_trace(
        self,
        pipe,
        inputs_shared,
        inputs_posi,
        object_context,
    ):
        # Legacy no-context dropout creates O zero tokens. Keep the Scheme-D
        # memory layout stable at K*O even when no object evidence is supplied.
        if (
            object_context is not None
            and object_context.ndim == 3
            and int(object_context.shape[1]) == int(self.aux_max_objects)
            and not bool(torch.count_nonzero(object_context.detach()).item())
        ):
            object_context = object_context.repeat_interleave(
                int(self.tube_num_tokens), dim=1
            )
        return super()._run_main_loss_with_trace(
            pipe,
            inputs_shared,
            inputs_posi,
            object_context,
        )

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        total, metrics = super()._compute_object_losses(
            pipe,
            inputs_shared,
            inputs_posi,
        )
        diagnostics = self.object_pooler.pop_diagnostics()
        shape_trace = self.object_pooler.pop_shape_trace()
        if diagnostics is not None:
            metrics.update(
                {
                    "train/tube_source_tokens_per_object": float(
                        diagnostics.source_tokens_per_object
                    ),
                    "train/tube_output_tokens_per_object": float(
                        diagnostics.output_tokens_per_object
                    ),
                    "train/tube_motion_tokens_per_object": float(
                        diagnostics.motion_tokens_per_object
                    ),
                    "train/tube_valid_objects": float(diagnostics.valid_objects),
                    "train/tube_jepa_frames": float(diagnostics.jepa_frames),
                    "train/tube_latent_frames": float(diagnostics.latent_frames),
                    "train/tube_track_frames": float(diagnostics.track_frames),
                }
            )
        metrics["train/tube_active_dit_blocks"] = float(len(self.object_block_ids))
        metrics["train/tube_vggt_enabled"] = 0.0
        self._tube_forward_index += 1
        if shape_trace is not None and (
            self.debug_print_tube_shapes or self.tube_shape_trace_path is not None
        ):
            raw_sample = inputs_shared.get("raw_sample", {})
            valid_objects = int(round(float(metrics.get("train/object_count", 0.0))))
            shape_trace.update(
                {
                    "21_entity_bound_adapter_input_B_K_O_D": [
                        1,
                        self.tube_num_tokens,
                        int(self.aux_max_objects),
                        int(self.object_adapter.dim),
                    ],
                    "22_object_adapter_output_B_KxO_D": [
                        1,
                        self.tube_num_tokens * int(self.aux_max_objects),
                        int(self.object_adapter.dim),
                    ],
                    "23_compacted_object_memory_B_KxOvalid_D": [
                        1,
                        self.tube_num_tokens * valid_objects,
                        int(self.object_adapter.dim),
                    ],
                    "24_wan_clean_prefix_latents": list(
                        inputs_shared.get("clean_prefix_latents").shape
                    ),
                    "25_wan_full_video_latents": list(
                        inputs_shared.get("input_latents").shape
                    ),
                    "26_t5_text_context": list(inputs_posi.get("context").shape),
                    "27_object_cross_attn_q_weight": list(
                        self.pipe.dit.blocks[self.object_block_ids[0]]
                        .object_cross_attn.q.weight.shape
                    ),
                    "28_object_cross_attn_k_weight": list(
                        self.pipe.dit.blocks[self.object_block_ids[0]]
                        .object_cross_attn.k.weight.shape
                    ),
                }
            )
            record = {
                "forward_index": self._tube_forward_index,
                "dataset_source": self._dataset_source(inputs_shared),
                "video_path": str(raw_sample.get("video_path", "")),
                "caption": str(raw_sample.get("caption", "")),
                "active_dit_blocks": list(self.object_block_ids),
                "shape_trace": shape_trace,
            }
            if self.debug_print_tube_shapes:
                print("[scheme-d-shapes] " + json.dumps(record, ensure_ascii=False))
            if self.tube_shape_trace_path is not None and int(os.environ.get("LOCAL_RANK", "0")) == 0:
                self.tube_shape_trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.tube_shape_trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = entity_train.build_parser()
    parser.description = "Scheme-D learned object-tube resampler training."
    group = parser.add_argument_group("scheme_d_object_tube")
    group.add_argument("--tube_num_tokens", type=int, default=4)
    group.add_argument("--tube_hidden_dim", type=int, default=256)
    group.add_argument("--tube_num_heads", type=int, default=8)
    group.add_argument("--tube_num_layers", type=int, default=2)
    group.add_argument("--tube_motion_tokens", type=int, default=4)
    group.add_argument("--tube_motion_fourier_bands", type=int, default=4)
    group.add_argument("--tube_object_attn_dim", type=int, default=256)
    group.add_argument("--tube_object_attn_heads", type=int, default=8)
    group.add_argument("--tube_latent_dim", type=int, default=48)
    group.add_argument("--tube_modality_dropout_prob", type=float, default=0.10)
    group.add_argument(
        "--object_block_ids",
        default="8,11,14,17,20,23",
        help="Comma-separated Wan DiT blocks retaining object cross-attention.",
    )
    group.add_argument("--debug_print_tube_shapes", action="store_true")
    group.add_argument("--tube_shape_trace_path", default=None)
    return parser


def build_model(args: argparse.Namespace, accelerator) -> SchemeDObjectTubeWanModule:
    original_factory = replay.ReplayPreserveNoGTBoxWanModule

    def factory(*model_args, **model_kwargs):
        return SchemeDObjectTubeWanModule(
            *model_args,
            **model_kwargs,
            entity_binding_enabled=not getattr(
                args, "disable_entity_id_binding", False
            ),
            entity_binding_sources=getattr(
                args, "entity_binding_sources", "pybullet,kubric"
            ),
            entity_binding_bottleneck_dim=getattr(
                args, "entity_binding_bottleneck_dim", 256
            ),
            entity_binding_gate_init=getattr(
                args, "entity_binding_gate_init", 0.1
            ),
            entity_binding_dropout_prob=getattr(
                args, "entity_binding_dropout_prob", 0.0
            ),
            entity_binding_residual_max_ratio=getattr(
                args, "entity_binding_residual_max_ratio", 0.1
            ),
            entity_binding_randomize_ids=(
                not getattr(args, "disable_entity_binding_id_randomization", False)
            ),
            debug_print_entity_binding=getattr(args, "debug_print_entity_binding", False),
            tube_num_tokens=getattr(args, "tube_num_tokens", 4),
            tube_hidden_dim=getattr(args, "tube_hidden_dim", 256),
            tube_num_heads=getattr(args, "tube_num_heads", 8),
            tube_num_layers=getattr(args, "tube_num_layers", 2),
            tube_motion_tokens=getattr(args, "tube_motion_tokens", 4),
            tube_motion_fourier_bands=getattr(
                args, "tube_motion_fourier_bands", 4
            ),
            tube_object_attn_dim=getattr(args, "tube_object_attn_dim", 256),
            tube_object_attn_heads=getattr(args, "tube_object_attn_heads", 8),
            tube_latent_dim=getattr(args, "tube_latent_dim", 48),
            tube_modality_dropout_prob=getattr(
                args, "tube_modality_dropout_prob", 0.10
            ),
            object_block_ids=getattr(
                args, "object_block_ids", "8,11,14,17,20,23"
            ),
            debug_print_tube_shapes=getattr(args, "debug_print_tube_shapes", False),
            tube_shape_trace_path=getattr(args, "tube_shape_trace_path", None),
        )

    replay.ReplayPreserveNoGTBoxWanModule = factory
    try:
        model = replay.build_model(args, accelerator)
    finally:
        replay.ReplayPreserveNoGTBoxWanModule = original_factory
    if not isinstance(model, SchemeDObjectTubeWanModule):
        raise TypeError(f"unexpected model type: {type(model).__name__}")
    return model


def audit_scheme_d_checkpoint(
    model: SchemeDObjectTubeWanModule,
    checkpoint_path: Path,
) -> dict[str, Any]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

        def exactly_one(suffix: str) -> str:
            matches = [key for key in keys if key.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"invalid Scheme-D checkpoint: expected one tensor ending "
                    f"with {suffix!r}, found {matches}"
                )
            return matches[0]

        query_key = exactly_one("object_pooler.output_queries")
        query_shape = tuple(handle.get_tensor(query_key).shape)
        expected_query_shape = (
            int(model.tube_num_tokens),
            int(model.object_pooler.hidden_dim),
        )
        if query_shape != expected_query_shape:
            raise RuntimeError(
                "Scheme-D K/hidden mismatch: "
                f"checkpoint={query_shape}, model={expected_query_shape}"
            )
        for block_id in model.object_block_ids:
            exactly_one(f"blocks.{block_id}.object_gate")
            exactly_one(f"blocks.{block_id}.object_cross_attn.k.weight")
        exactly_one("object_adapter.entity_id_embed.weight")
        legacy_embedding_keys = [key for key in keys if "object_embedding." in key]
        if legacy_embedding_keys:
            raise RuntimeError(
                "Scheme-D v1/v2 checkpoint is incompatible with v3 bottleneck "
                "object attention; legacy object embedding tensors found: "
                f"{legacy_embedding_keys}"
            )
        q_key = exactly_one(
            f"blocks.{model.object_block_ids[0]}.object_cross_attn.q.weight"
        )
        k_key = exactly_one(
            f"blocks.{model.object_block_ids[0]}.object_cross_attn.k.weight"
        )
        q_shape = tuple(handle.get_tensor(q_key).shape)
        k_shape = tuple(handle.get_tensor(k_key).shape)
        expected_q_shape = (
            int(model.tube_object_attn_dim),
            int(model.pipe.dit.dim),
        )
        expected_k_shape = (
            int(model.tube_object_attn_dim),
            int(model.object_pooler.output_dim),
        )
        if q_shape != expected_q_shape or k_shape != expected_k_shape:
            raise RuntimeError(
                "Scheme-D bottleneck attention mismatch: "
                f"checkpoint_qk={(q_shape, k_shape)}, "
                f"model_qk={(expected_q_shape, expected_k_shape)}"
            )
    return {
        "architecture_version": 3,
        "checkpoint": str(checkpoint_path),
        "tube_query_shape": list(query_shape),
        "object_attention_q_shape": list(q_shape),
        "object_attention_k_shape": list(k_shape),
        "object_block_ids": list(model.object_block_ids),
    }


def load_scheme_d_trainables(model, checkpoint: str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "checkpoint.safetensors"
    elif checkpoint_path.name == "training_state.pt":
        checkpoint_path = Path(tvn.resolve_lora_checkpoint_for_resume(str(checkpoint_path)))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Scheme-D checkpoint not found: {checkpoint_path}")
    audit = audit_scheme_d_checkpoint(model, checkpoint_path)
    load_info = tvn._load_filtered_checkpoint_into_model(
        model,
        str(checkpoint_path),
        include_prefixes=("object_pooler.", "object_adapter."),
        include_substrings=(
            "object_embedding",
            ".object_cross_attn.",
            ".object_gate",
            ".norm4.",
        ),
    )
    load_info["scheme_d_audit"] = audit
    return load_info


def _parameter_stats(named_parameters) -> dict[str, int]:
    named = list(named_parameters)
    return {
        "parameter_tensors": len(named),
        "elements": sum(int(param.numel()) for _, param in named),
        "trainable_tensors": sum(int(param.requires_grad) for _, param in named),
        "trainable_elements": sum(
            int(param.numel()) for _, param in named if param.requires_grad
        ),
    }


def build_module_report(
    model: SchemeDObjectTubeWanModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    object_markers = (
        "object_embedding",
        ".object_cross_attn.",
        ".object_gate",
        ".norm4.",
    )
    dit_named = list(model.pipe.dit.named_parameters())
    groups = {
        "dit_object_branch": _parameter_stats(
            (name, param)
            for name, param in dit_named
            if any(marker in name for marker in object_markers)
        ),
        "dit_base_lora": _parameter_stats(
            (name, param)
            for name, param in dit_named
            if "lora_" in name
            and not any(marker in name for marker in object_markers)
        ),
        "dit_base_without_lora_or_object": _parameter_stats(
            (name, param)
            for name, param in dit_named
            if "lora_" not in name
            and not any(marker in name for marker in object_markers)
        ),
        "object_tube_resampler": _parameter_stats(
            model.object_pooler.named_parameters()
        ),
        "entity_object_adapter": _parameter_stats(
            model.object_adapter.named_parameters()
        ),
        "object_aux_heads": _parameter_stats(
            model.object_aux_heads.named_parameters()
        ),
    }
    for report_name, attr_name in (
        ("wan_vae", "vae"),
        ("wan_text_encoder", "text_encoder"),
    ):
        module = getattr(model.pipe, attr_name, None)
        if isinstance(module, nn.Module):
            groups[report_name] = _parameter_stats(module.named_parameters())
    for report_name, attr_name in (
        ("vjepa", "jepa_runner"),
        ("cotracker", "cotracker_runner"),
    ):
        runner = getattr(model, attr_name, None)
        module = getattr(runner, "module", None)
        if isinstance(module, nn.Module):
            groups[report_name] = _parameter_stats(module.named_parameters())

    all_trainable = list(model.trainable_modules())
    return {
        "architecture": {
            "version": 3,
            "tube_tokens_per_object": int(model.tube_num_tokens),
            "tube_hidden_dim": int(model.object_pooler.hidden_dim),
            "object_memory_dim": int(model.object_pooler.output_dim),
            "motion_tokens_per_object": int(model.tube_motion_tokens),
            "motion_fourier_bands": int(model.tube_motion_fourier_bands),
            "object_attention_dim": int(model.tube_object_attn_dim),
            "object_attention_heads": int(model.tube_object_attn_heads),
            "maximum_objects": int(model.aux_max_objects),
            "points_per_object": int(model.object_num_queries),
            "active_object_dit_blocks": list(model.object_block_ids),
            "vggt_enabled": False,
            "legacy_stage1a_enabled": False,
        },
        "weight_sources": {
            "wan_base_vae_t5_dit": {
                "path": str(args.wan_root),
                "frozen": True,
            },
            "base_wan_lora": {
                "path": str(args.lora_checkpoint),
                "frozen": True,
            },
            "vjepa": {
                "path": str(args.jepa_ckpt_path),
                "frozen": True,
            },
            "cotracker": {
                "path": str(args.cotracker_checkpoint),
                "frozen": True,
            },
            "grounding_dino": {
                "path": "/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/groundingdino_swint_ogc.pth",
                "frozen": True,
            },
            "sam2": {
                "path": "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt",
                "frozen": True,
            },
            "legacy_stage1a": {
                "path": None,
                "loaded": False,
            },
            "scheme_d_stage1b": {
                "path": args.stage2_init_from or args.stage2_resume_from,
                "loaded": bool(args.stage2_init_from or args.stage2_resume_from),
            },
        },
        "parameter_groups": groups,
        "optimizer_unique_trainable_tensors": len(all_trainable),
        "optimizer_unique_trainable_elements": sum(
            int(param.numel()) for param in all_trainable
        ),
    }


def main() -> None:
    args = tvn.prepare_args(build_parser().parse_args())
    if args.stage1a_init_from is not None:
        raise ValueError(
            "Scheme-D does not load the legacy Stage1A checkpoint; omit "
            "--stage1a_init_from"
        )
    if args.stage2_init_from is not None and args.stage2_resume_from is not None:
        raise ValueError("--stage2_init_from and --stage2_resume_from are mutually exclusive")
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)
    dataset = replay.build_dataset(args)
    if accelerator.is_main_process and hasattr(dataset, "dataset_stats"):
        accelerator.print(f"Replay mixture: {dataset.dataset_stats}")
    model = build_model(args, accelerator)
    stage2_source = args.stage2_init_from or args.stage2_resume_from
    if stage2_source is not None:
        info = load_scheme_d_trainables(model, stage2_source)
        mode = "model-only initialization" if args.stage2_init_from else "resume"
        accelerator.print(
            f"Loaded Scheme-D {mode}: source={stage2_source} "
            f"loaded={info['loaded_count']} "
            f"shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )
    module_report = build_module_report(model, args)
    if accelerator.is_main_process:
        report_path = Path(args.output_path).expanduser().resolve() / "scheme_d_module_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(module_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        accelerator.print(
            "[scheme-d-module-report] "
            + json.dumps(module_report, ensure_ascii=False)
        )
    replay.base._log_stage_summary(accelerator, model, args)
    accelerator.print(
        "Scheme-D: "
        f"v3, K={args.tube_num_tokens}, hidden={args.tube_hidden_dim}, "
        f"motion_tokens={args.tube_motion_tokens}, "
        f"object_attn={args.tube_object_attn_dim}, "
        f"layers={args.tube_num_layers}, blocks={model.object_block_ids}, "
        "VGGT=disabled, legacy_stage1a=disabled"
    )
    disabled_val = HeadOnlyValConfig(
        enabled=False,
        split="val",
        every_steps=None,
        num_batches=1,
    )
    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict[str, Any] = {}
    try:
        tvn.train_loop(
            accelerator,
            dataset,
            model,
            model_logger,
            args,
            runtime_state=runtime_state,
            headonly_val_dataloader=None,
            headonly_val_config=disabled_val,
        )
    except (KeyboardInterrupt, tvn.TrainingInterrupted) as exc:
        checkpoint_root = tvn.get_checkpoint_dir(args)
        model_logger.save_model(
            accelerator,
            model,
            tvn.training_checkpoint_file(checkpoint_root, "interrupted-latest"),
        )
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get("progress", {})
        if optimizer is not None and scheduler is not None:
            tvn.save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=progress.get("global_step", 0),
                epoch_id=progress.get("epoch_id", 0),
                batch_in_epoch=progress.get("batch_in_epoch", 0),
                model_logger=model_logger,
                state_path=tvn.training_state_file(
                    checkpoint_root,
                    "interrupted-latest",
                ),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc
    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
