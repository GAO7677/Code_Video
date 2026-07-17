#!/usr/bin/env python3
"""Train Scheme-E object tokens through gated masked joint attention."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from safetensors import safe_open

import code_vjepa_vggt.train_v_newtrain as tvn
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve as replay
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig
from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler import train as scheme_d
from code_vjepa_vggt.train0717_scheme_e_object_joint_self_attention.models import (
    install_bottleneck_object_joint_self_attention,
)
from diffsynth.diffusion import ModelLogger


# Scheme-D's JSON-native inference builder swaps its train-module reference to
# this module and resolves these helpers dynamically.
prepare_jepa_context_video = scheme_d.prepare_jepa_context_video
compact_object_context_valid_slots = scheme_d.compact_object_context_valid_slots


class SchemeEObjectJointSelfAttentionWanModule(scheme_d.SchemeDObjectTubeWanModule):
    def __init__(
        self,
        *args,
        object_block_ids: str = "8,14,20",
        joint_gate_init: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, object_block_ids=object_block_ids, **kwargs)
        if not self.enable_object_branch:
            return
        self.joint_gate_init = float(joint_gate_init)
        self.object_block_layout = install_bottleneck_object_joint_self_attention(
            self.pipe.dit,
            self.object_block_ids,
            object_dim=int(self.object_pooler.output_dim),
            inner_dim=int(self.tube_object_attn_dim),
            num_heads=int(self.tube_object_attn_heads),
            gate_init=self.joint_gate_init,
        )
        for name, param in self.pipe.dit.named_parameters():
            if ".object_cross_attn." in name or ".object_gate" in name or ".norm4." in name:
                param.requires_grad = bool(self.train_object_dit_branch)

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        total, metrics = super()._compute_object_losses(pipe, inputs_shared, inputs_posi)
        traces = []
        for block_id in self.object_block_ids:
            module = self.pipe.dit.blocks[block_id].object_cross_attn
            trace = module.pop_trace() if hasattr(module, "pop_trace") else None
            if trace is not None:
                traces.append({"block_id": int(block_id), **trace})
        if traces:
            first = traces[0]
            metrics.update(
                {
                    "train/joint_self_attention_active_blocks": float(len(traces)),
                    "train/joint_self_attention_video_tokens": float(first["video_shape"][1]),
                    "train/joint_self_attention_object_tokens": float(first["object_shape"][1]),
                    "train/joint_self_attention_total_tokens": float(first["joint_shape"][1]),
                    "train/joint_self_attention_active_batch_items": float(
                        first["active_batch_items"]
                    ),
                }
            )
            if self.debug_print_tube_shapes:
                print(
                    "[scheme-e-joint-shapes] "
                    + json.dumps({"forward": self._tube_forward_index, "blocks": traces})
                )
            if self.tube_shape_trace_path is not None and os.environ.get("LOCAL_RANK", "0") == "0":
                with self.tube_shape_trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "scheme_e_joint_self_attention",
                                "forward": self._tube_forward_index,
                                "blocks": traces,
                            }
                        )
                        + "\n"
                    )
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = scheme_d.build_parser()
    parser.description = "Scheme-E gated masked object joint-attention training."
    parser.set_defaults(object_block_ids="8,14,20")
    group = parser.add_argument_group("scheme_e_joint_self_attention")
    group.add_argument(
        "--joint_gate_init",
        type=float,
        default=0.0,
        help="Scalar tanh gate initialization for each joint self-attention adapter.",
    )
    return parser


def build_model(args: argparse.Namespace, accelerator) -> SchemeEObjectJointSelfAttentionWanModule:
    original_factory = replay.ReplayPreserveNoGTBoxWanModule

    def factory(*model_args, **model_kwargs):
        return SchemeEObjectJointSelfAttentionWanModule(
            *model_args,
            **model_kwargs,
            entity_binding_enabled=not getattr(args, "disable_entity_id_binding", False),
            entity_binding_sources=getattr(args, "entity_binding_sources", "pybullet,kubric"),
            entity_binding_bottleneck_dim=getattr(args, "entity_binding_bottleneck_dim", 256),
            entity_binding_gate_init=getattr(args, "entity_binding_gate_init", 0.1),
            entity_binding_dropout_prob=getattr(args, "entity_binding_dropout_prob", 0.0),
            entity_binding_residual_max_ratio=getattr(args, "entity_binding_residual_max_ratio", 0.1),
            entity_binding_randomize_ids=(
                not getattr(args, "disable_entity_binding_id_randomization", False)
            ),
            debug_print_entity_binding=getattr(args, "debug_print_entity_binding", False),
            tube_num_tokens=getattr(args, "tube_num_tokens", 4),
            tube_hidden_dim=getattr(args, "tube_hidden_dim", 256),
            tube_num_heads=getattr(args, "tube_num_heads", 8),
            tube_num_layers=getattr(args, "tube_num_layers", 2),
            tube_motion_tokens=getattr(args, "tube_motion_tokens", 4),
            tube_motion_fourier_bands=getattr(args, "tube_motion_fourier_bands", 4),
            tube_object_attn_dim=getattr(args, "tube_object_attn_dim", 256),
            tube_object_attn_heads=getattr(args, "tube_object_attn_heads", 8),
            tube_latent_dim=getattr(args, "tube_latent_dim", 48),
            tube_modality_dropout_prob=getattr(args, "tube_modality_dropout_prob", 0.05),
            object_block_ids=getattr(args, "object_block_ids", "8,14,20"),
            debug_print_tube_shapes=getattr(args, "debug_print_tube_shapes", False),
            tube_shape_trace_path=getattr(args, "tube_shape_trace_path", None),
            joint_gate_init=getattr(args, "joint_gate_init", 0.0),
        )

    replay.ReplayPreserveNoGTBoxWanModule = factory
    try:
        model = replay.build_model(args, accelerator)
    finally:
        replay.ReplayPreserveNoGTBoxWanModule = original_factory
    if not isinstance(model, SchemeEObjectJointSelfAttentionWanModule):
        raise TypeError(f"unexpected model type: {type(model).__name__}")
    return model


def _checkpoint_file(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "checkpoint.safetensors"
    elif path.name == "training_state.pt":
        path = Path(tvn.resolve_lora_checkpoint_for_resume(str(path)))
    if not path.is_file():
        raise FileNotFoundError(f"Scheme-E checkpoint not found: {path}")
    return path


def audit_scheme_e_checkpoint(
    model: SchemeEObjectJointSelfAttentionWanModule,
    checkpoint_path: Path,
) -> dict[str, Any]:
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())

        def exactly_one(suffix: str) -> str:
            matches = [key for key in keys if key.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"invalid Scheme-E checkpoint: expected one tensor ending "
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
                f"Scheme-E tube query mismatch: {query_shape} != {expected_query_shape}"
            )
        for block_id in model.object_block_ids:
            exactly_one(f"blocks.{block_id}.object_gate")
            exactly_one(f"blocks.{block_id}.object_cross_attn.video_in.weight")
            exactly_one(f"blocks.{block_id}.object_cross_attn.object_in.weight")
            exactly_one(f"blocks.{block_id}.object_cross_attn.object_update_norm.weight")
            exactly_one(f"blocks.{block_id}.object_cross_attn.q.weight")
        exactly_one("object_adapter.entity_id_embed.weight")
        exactly_one("object_adapter.entity_text_context_up.weight")
        first = model.object_block_ids[0]
        video_in_shape = tuple(
            handle.get_tensor(
                exactly_one(f"blocks.{first}.object_cross_attn.video_in.weight")
            ).shape
        )
        object_in_shape = tuple(
            handle.get_tensor(
                exactly_one(f"blocks.{first}.object_cross_attn.object_in.weight")
            ).shape
        )
        expected_video = (int(model.tube_object_attn_dim), int(model.pipe.dit.dim))
        expected_object = (
            int(model.tube_object_attn_dim),
            int(model.object_pooler.output_dim),
        )
        if video_in_shape != expected_video or object_in_shape != expected_object:
            raise RuntimeError(
                "Scheme-E joint attention mismatch: "
                f"checkpoint={(video_in_shape, object_in_shape)} "
                f"model={(expected_video, expected_object)}"
            )
    return {
        "architecture": "scheme_e_gated_masked_object_joint_attention",
        "architecture_version": 3,
        "checkpoint": str(checkpoint_path),
        "tube_query_shape": list(query_shape),
        "video_projection_shape": list(video_in_shape),
        "object_projection_shape": list(object_in_shape),
        "object_block_ids": list(model.object_block_ids),
    }


def load_scheme_e_trainables(model, checkpoint: str) -> dict[str, Any]:
    checkpoint_path = _checkpoint_file(checkpoint)
    audit = audit_scheme_e_checkpoint(model, checkpoint_path)
    load_info = tvn._load_filtered_checkpoint_into_model(
        model,
        str(checkpoint_path),
        include_prefixes=("object_pooler.", "object_adapter."),
        include_substrings=(
            ".object_cross_attn.",
            ".object_gate",
            ".norm4.",
        ),
    )
    load_info["scheme_e_audit"] = audit
    return load_info


# Scheme-D's inference wrapper calls this legacy name on the injected train
# module. Keep the alias local to Scheme-E rather than changing the old code.
load_scheme_d_trainables = load_scheme_e_trainables


def build_module_report(
    model: SchemeEObjectJointSelfAttentionWanModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = scheme_d.build_module_report(model, args)
    report["architecture"].update(
        {
            "name": "scheme_e_gated_masked_object_joint_attention",
            "version": 3,
            "injection_type": "masked_joint_attention",
            "injection_position": "after_wan_self_attention_before_text_cross_attention",
            "attention_mask_policy": (
                "object_reads_video_and_object_then_video_reads_object_only"
            ),
            "added_video_to_video_attention": False,
            "joint_gate_init": float(model.joint_gate_init),
            "symmetric_text_object_entity_id_binding": True,
            "active_object_dit_blocks": list(model.object_block_ids),
        }
    )
    report["weight_sources"].pop("scheme_d_stage1b", None)
    report["weight_sources"]["scheme_e_stage1b"] = {
        "path": args.stage2_init_from or args.stage2_resume_from,
        "loaded": bool(args.stage2_init_from or args.stage2_resume_from),
    }
    return report


def main() -> None:
    args = tvn.prepare_args(build_parser().parse_args())
    if args.stage1a_init_from is not None:
        raise ValueError("Scheme-E does not load legacy Stage1A; omit --stage1a_init_from")
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
        info = load_scheme_e_trainables(model, stage2_source)
        mode = "model-only initialization" if args.stage2_init_from else "resume"
        accelerator.print(
            f"Loaded Scheme-E {mode}: source={stage2_source} "
            f"loaded={info['loaded_count']} "
            f"shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )

    module_report = build_module_report(model, args)
    if accelerator.is_main_process:
        report_path = Path(args.output_path).expanduser().resolve() / "scheme_e_module_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(module_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        accelerator.print("[scheme-e-module-report] " + json.dumps(module_report))
    replay.base._log_stage_summary(accelerator, model, args)
    accelerator.print(
        "Scheme-E: "
        f"masked-joint-attention, K={args.tube_num_tokens}, "
        f"hidden={args.tube_hidden_dim}, joint_dim={args.tube_object_attn_dim}, "
        f"blocks={model.object_block_ids}, gate_init={args.joint_gate_init}"
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
                state_path=tvn.training_state_file(checkpoint_root, "interrupted-latest"),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc
    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
