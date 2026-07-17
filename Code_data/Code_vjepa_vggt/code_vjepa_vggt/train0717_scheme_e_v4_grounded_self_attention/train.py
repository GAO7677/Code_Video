#!/usr/bin/env python3
"""Train Scheme-E v4 grouped grounded routing on top of frozen Wan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import code_vjepa_vggt.train_v_newtrain as tvn
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_base
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve as replay
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve_entity_id_binding as entity_train
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig
from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler import train as scheme_d
from code_vjepa_vggt.train0717_scheme_e_v4_grounded_self_attention.models import (
    GroundedObjectCondition,
    TrainableGroupedGroundedAttention,
    install_grouped_grounded_self_attention,
)
from code_vjepa_vggt.train0717_scheme_e_v4_grounded_self_attention.prototype_grouped_grounded_self_attention import (
    masks_to_assignment_targets,
    masks_to_spatial_bias,
)
from diffsynth.diffusion import ModelLogger


prepare_jepa_context_video = scheme_d.prepare_jepa_context_video
compact_object_context_valid_slots = scheme_d.compact_object_context_valid_slots


class SchemeEV4GroundedWanModule(scheme_d.SchemeDObjectTubeWanModule):
    def __init__(
        self,
        *args,
        grounded_gate_init: float = 0.01,
        noun_key_gate_init: float = 0.1,
        assignment_loss_weight: float = 0.1,
        spatial_bias_strength: float = 0.5,
        spatial_bias_dropout_prob: float = 0.25,
        evidence_rms_reference: float = 0.01,
        evidence_active_threshold: float = 1.0e-3,
        grounded_metrics_trace_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.grounded_metrics_trace_path = (
            None
            if grounded_metrics_trace_path is None
            else Path(grounded_metrics_trace_path).expanduser().resolve()
        )
        if not self.enable_object_branch:
            return
        text_embedding = getattr(self.pipe.dit, "text_embedding", None)
        text_linear = next(
            (
                module
                for module in text_embedding.modules()
                if isinstance(module, torch.nn.Linear)
            ),
            None,
        )
        if text_linear is None:
            raise RuntimeError("cannot infer Wan T5 input dimension")
        self.grounded_gate_init = float(grounded_gate_init)
        self.noun_key_gate_init = float(noun_key_gate_init)
        self.assignment_loss_weight = float(assignment_loss_weight)
        self.spatial_bias_strength = float(spatial_bias_strength)
        self.spatial_bias_dropout_prob = float(spatial_bias_dropout_prob)
        self.evidence_rms_reference = float(evidence_rms_reference)
        self.evidence_active_threshold = float(evidence_active_threshold)
        self.object_block_layout = install_grouped_grounded_self_attention(
            self.pipe.dit,
            self.object_block_ids,
            object_dim=int(self.object_pooler.output_dim),
            text_dim=int(text_linear.in_features),
            inner_dim=int(self.tube_object_attn_dim),
            gate_init=self.grounded_gate_init,
            noun_key_gate_init=self.noun_key_gate_init,
            assignment_loss_weight=self.assignment_loss_weight,
            evidence_rms_reference=self.evidence_rms_reference,
            evidence_active_threshold=self.evidence_active_threshold,
            spatial_bias_dropout_p=self.spatial_bias_dropout_prob,
        )
        for name, parameter in self.pipe.dit.named_parameters():
            if ".object_cross_attn." in name:
                parameter.requires_grad = bool(self.train_object_dit_branch)
        self.object_adapter.requires_grad_(False)
        self._v4_pooler_capture: dict[str, torch.Tensor] | None = None
        self._v4_grounding_sample = None
        self._v4_last_metrics: dict[str, float] = {}
        self._v4_last_binding: list[dict[str, Any]] = []
        self._v4_pooler_hook = self.object_pooler.register_forward_hook(
            self._capture_pooler_output,
            with_kwargs=True,
        )

    def _capture_pooler_output(self, module, args, kwargs, output) -> None:
        raw = output.object_latent_tokens
        query_template = module.output_proj(module.output_norm(module.output_queries))
        query_template = query_template[None, :, None, :].to(
            device=raw.device,
            dtype=raw.dtype,
        )
        valid = kwargs["object_valid_mask"].to(device=raw.device) > 0.5
        content_delta = (raw - query_template) * valid[:, None, :, None].to(raw.dtype)
        visibility = kwargs["visibility"].detach().float()
        confidence = kwargs["confidence"].detach().float()
        evidence_confidence = (visibility * confidence).mean(dim=(1, 3))
        self._v4_pooler_capture = {
            "content_delta": content_delta.permute(0, 2, 1, 3).contiguous(),
            "valid_mask": valid,
            "evidence_confidence": evidence_confidence,
        }

    def _build_object_query_priors(self, sample: dict, *, image_hw: tuple[int, int]):
        original = self.viewer_grounding.build_sample
        captured: dict[str, Any] = {}

        def capture(*args, **kwargs):
            value = original(*args, **kwargs)
            captured["sample"] = value
            return value

        self.viewer_grounding.build_sample = capture
        try:
            output = super()._build_object_query_priors(sample, image_hw=image_hw)
        finally:
            self.viewer_grounding.build_sample = original
        self._v4_grounding_sample = captured.get("sample")
        return output

    def _noun_features(
        self,
        pipe,
        inputs_shared: dict,
        inputs_posi: dict,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_context = inputs_posi["context"]
        objects = int(valid_mask.shape[1])
        features = prompt_context.new_zeros(
            prompt_context.shape[0], objects, prompt_context.shape[-1]
        )
        matched = torch.zeros(
            prompt_context.shape[0],
            objects,
            dtype=torch.bool,
            device=prompt_context.device,
        )
        grounding = self._v4_grounding_sample
        if grounding is None:
            return features, matched
        prompt = str(inputs_shared.get("raw_sample", {}).get("caption", ""))
        prompt_ids, prompt_mask = pipe.tokenizer(
            prompt,
            return_mask=True,
            add_special_tokens=True,
        )
        valid_length = int(prompt_mask[0].sum().item())
        token_ids = [int(value) for value in prompt_ids[0, :valid_length].tolist()]
        tracks = list(getattr(grounding, "object_tracks", []) or [])
        used_spans: set[tuple[int, int]] = set()
        for slot_id in range(objects):
            if not bool(valid_mask[0, slot_id]) or slot_id >= len(tracks):
                continue
            phrase = str(getattr(tracks[slot_id], "phrase", ""))
            pooled, _, _, span = entity_train._pool_unique_phrase_span_from_prompt_context(
                prompt_token_ids=token_ids,
                prompt_context=prompt_context,
                tokenizer=pipe.tokenizer,
                phrase=phrase,
                used_spans=used_spans,
            )
            if pooled is None or span is None:
                continue
            features[0, slot_id] = pooled[0]
            matched[0, slot_id] = True
        return features, matched

    def _aligned_masks(
        self,
        inputs_shared: dict,
        *,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = inputs_shared["input_latents"]
        batch, _, latent_frames, latent_h, latent_w = latents.shape
        grid_h, grid_w = int(latent_h // 2), int(latent_w // 2)
        objects = int(valid_mask.shape[1])
        aligned = latents.new_zeros(
            batch,
            latent_frames,
            grid_h,
            grid_w,
            objects,
            dtype=torch.float32,
        )
        context_latents = inputs_shared.get("clean_prefix_latents")
        known_frames = 0 if context_latents is None else int(context_latents.shape[2])
        grounding = self._v4_grounding_sample
        tracks = [] if grounding is None else list(
            getattr(grounding, "object_tracks", []) or []
        )
        for slot_id in range(min(objects, len(tracks))):
            if not bool(valid_mask[0, slot_id]) or known_frames == 0:
                continue
            masks = torch.as_tensor(
                getattr(tracks[slot_id], "masks_thw"),
                dtype=torch.float32,
                device=latents.device,
            )
            if masks.ndim != 3 or int(masks.shape[0]) == 0:
                continue
            time_ids = torch.linspace(
                0,
                int(masks.shape[0]) - 1,
                known_frames,
                device=latents.device,
            ).round().long()
            selected = masks[time_ids, None]
            resized = F.interpolate(
                selected,
                size=(grid_h, grid_w),
                mode="area",
            )[:, 0]
            aligned[0, :known_frames, :, :, slot_id] = resized
        flattened = aligned.flatten(1, 3)
        known = torch.zeros(
            batch,
            latent_frames,
            grid_h,
            grid_w,
            dtype=torch.bool,
            device=latents.device,
        )
        known[:, :known_frames] = True
        return flattened, known.flatten(1, 3)

    def _build_grounded_condition(
        self,
        pipe,
        inputs_shared: dict,
        inputs_posi: dict,
    ) -> GroundedObjectCondition:
        capture = self._v4_pooler_capture
        if capture is None:
            latents = inputs_shared["input_latents"]
            objects = int(self.aux_max_objects)
            content_delta = latents.new_zeros(
                latents.shape[0],
                objects,
                int(self.tube_num_tokens),
                int(self.object_pooler.output_dim),
            )
            valid_mask = torch.zeros(
                latents.shape[0], objects, dtype=torch.bool, device=latents.device
            )
            evidence_confidence = torch.zeros_like(valid_mask, dtype=torch.float32)
        else:
            content_delta = capture["content_delta"]
            valid_mask = capture["valid_mask"]
            evidence_confidence = capture["evidence_confidence"]
        noun_features, noun_matched = self._noun_features(
            pipe,
            inputs_shared,
            inputs_posi,
            valid_mask,
        )
        aligned_masks, known_mask = self._aligned_masks(
            inputs_shared,
            valid_mask=valid_mask,
        )
        spatial_bias = masks_to_spatial_bias(
            aligned_masks,
            known_mask,
            strength=self.spatial_bias_strength,
        )
        targets = masks_to_assignment_targets(
            aligned_masks,
            valid_mask,
            known_token_mask=known_mask,
        )
        return GroundedObjectCondition(
            content_delta=content_delta,
            valid_mask=valid_mask,
            evidence_confidence=evidence_confidence,
            noun_features=noun_features,
            noun_matched_mask=noun_matched,
            spatial_bias=spatial_bias,
            known_token_mask=known_mask,
            assignment_targets=targets,
        )

    def _run_main_loss_with_trace(
        self,
        pipe,
        inputs_shared,
        inputs_posi,
        object_context,
    ):
        condition = self._build_grounded_condition(pipe, inputs_shared, inputs_posi)
        tracks = list(
            getattr(self._v4_grounding_sample, "object_tracks", []) or []
        )
        self._v4_last_binding = []
        for slot_id in range(int(condition.valid_mask.shape[1])):
            track = tracks[slot_id] if slot_id < len(tracks) else None
            self._v4_last_binding.append(
                {
                    "slot_id": slot_id,
                    "phrase": str(getattr(track, "phrase", "")),
                    "valid": bool(condition.valid_mask[0, slot_id].item()),
                    "noun_matched": bool(
                        condition.noun_matched_mask[0, slot_id].item()
                    ),
                    "evidence_confidence": float(
                        condition.evidence_confidence[0, slot_id].detach().item()
                    ),
                }
            )
        diffusion_loss = kubric_base.flow_match_context_sft_loss(
            pipe,
            **inputs_shared,
            **inputs_posi,
            object_context=condition,
        )
        assignment_losses = []
        traces = []
        for block_id in self.object_block_ids:
            module = self.pipe.dit.blocks[block_id].object_cross_attn
            if not isinstance(module, TrainableGroupedGroundedAttention):
                continue
            auxiliary = module.pop_assignment_loss()
            trace = module.pop_trace()
            if auxiliary is not None:
                assignment_losses.append(auxiliary)
            if trace is not None:
                traces.append({"block_id": int(block_id), **trace})
        assignment_loss = (
            torch.stack(assignment_losses).mean()
            if assignment_losses
            else diffusion_loss.new_zeros(())
        )
        self._v4_last_metrics = {
            "train/loss_diffusion": float(diffusion_loss.detach().item()),
            "train/loss_assignment_weighted": float(assignment_loss.detach().item()),
            "train/grounded_valid_objects": float(condition.valid_mask.sum().item()),
            "train/grounded_matched_nouns": float(
                condition.noun_matched_mask.sum().item()
            ),
            "train/grounded_known_tokens": float(
                condition.known_token_mask.sum().item()
            ),
        }
        if traces:
            first = traces[0]
            for key in (
                "evidence_gate_mean",
                "background_assignment_mass",
                "assignment_entropy_mean",
                "content_logit_std",
                "spatial_bias_std",
                "residual_ratio_mean",
                "residual_ratio_p95",
                "residual_ratio_max",
                "context_residual_ratio",
                "future_residual_ratio",
            ):
                value = first.get(key)
                if value is not None:
                    self._v4_last_metrics[f"train/grounded_{key}"] = float(value)
            if self.debug_print_tube_shapes:
                print(
                    "[scheme-e-v4-grounded] "
                    + json.dumps({"forward": self._tube_forward_index + 1, "blocks": traces})
                )
            if self.tube_shape_trace_path is not None and os.environ.get("LOCAL_RANK", "0") == "0":
                self.tube_shape_trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.tube_shape_trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "kind": "scheme_e_v4_grouped_grounded",
                                "forward": self._tube_forward_index + 1,
                                "blocks": traces,
                            }
                        )
                        + "\n"
                    )
        return diffusion_loss + assignment_loss, None

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        self._v4_pooler_capture = None
        self._v4_grounding_sample = None
        self._v4_last_metrics = {}
        self._v4_last_binding = []
        total, metrics = super()._compute_object_losses(
            pipe,
            inputs_shared,
            inputs_posi,
        )
        metrics.update(self._v4_last_metrics)
        if self.grounded_metrics_trace_path is not None:
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            base_path = self.grounded_metrics_trace_path
            rank_path = base_path.with_name(
                f"{base_path.stem}.rank{rank}{base_path.suffix or '.jsonl'}"
            )
            raw_sample = inputs_shared.get("raw_sample", {})
            record = {
                "forward_index": int(self._tube_forward_index),
                "rank": rank,
                "dataset_source": self._dataset_source(inputs_shared),
                "video_path": str(raw_sample.get("video_path", "")),
                "caption": str(raw_sample.get("caption", "")),
                "slots": self._v4_last_binding,
                "metrics": {key: float(value) for key, value in metrics.items()},
            }
            rank_path.parent.mkdir(parents=True, exist_ok=True)
            with rank_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = scheme_d.build_parser()
    parser.description = "Scheme-E v4 grouped grounded self-attention training."
    parser.set_defaults(object_block_ids="14", disable_entity_id_binding=True)
    group = parser.add_argument_group("scheme_e_v4_grounded")
    group.add_argument("--grounded_gate_init", type=float, default=0.01)
    group.add_argument("--noun_key_gate_init", type=float, default=0.1)
    group.add_argument("--assignment_loss_weight", type=float, default=0.1)
    group.add_argument("--spatial_bias_strength", type=float, default=0.5)
    group.add_argument("--spatial_bias_dropout_prob", type=float, default=0.25)
    group.add_argument("--evidence_rms_reference", type=float, default=0.01)
    group.add_argument("--evidence_active_threshold", type=float, default=1.0e-3)
    group.add_argument("--grounded_metrics_trace_path", default=None)
    return parser


def build_model(args: argparse.Namespace, accelerator) -> SchemeEV4GroundedWanModule:
    original_factory = replay.ReplayPreserveNoGTBoxWanModule

    def factory(*model_args, **model_kwargs):
        return SchemeEV4GroundedWanModule(
            *model_args,
            **model_kwargs,
            entity_binding_enabled=False,
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
            object_block_ids=getattr(args, "object_block_ids", "14"),
            debug_print_tube_shapes=getattr(args, "debug_print_tube_shapes", False),
            tube_shape_trace_path=getattr(args, "tube_shape_trace_path", None),
            grounded_gate_init=getattr(args, "grounded_gate_init", 0.01),
            noun_key_gate_init=getattr(args, "noun_key_gate_init", 0.1),
            assignment_loss_weight=getattr(args, "assignment_loss_weight", 0.1),
            spatial_bias_strength=getattr(args, "spatial_bias_strength", 0.5),
            spatial_bias_dropout_prob=getattr(args, "spatial_bias_dropout_prob", 0.25),
            evidence_rms_reference=getattr(args, "evidence_rms_reference", 0.01),
            evidence_active_threshold=getattr(args, "evidence_active_threshold", 1.0e-3),
            grounded_metrics_trace_path=getattr(
                args, "grounded_metrics_trace_path", None
            ),
        )

    replay.ReplayPreserveNoGTBoxWanModule = factory
    try:
        model = replay.build_model(args, accelerator)
    finally:
        replay.ReplayPreserveNoGTBoxWanModule = original_factory
    if not isinstance(model, SchemeEV4GroundedWanModule):
        raise TypeError(f"unexpected model type: {type(model).__name__}")
    return model


def build_module_report(
    model: SchemeEV4GroundedWanModule,
    args: argparse.Namespace,
) -> dict[str, Any]:
    report = scheme_d.build_module_report(model, args)
    report["architecture"].update(
        {
            "name": "scheme_e_v4_grouped_grounded_self_attention_stage_adapter",
            "version": 4,
            "injection_position": "after_wan_self_attention_before_text_cross_attention",
            "routing": "shared_video_to_object_with_fixed_background",
            "active_object_dit_blocks": list(model.object_block_ids),
            "grounded_gate_init": model.grounded_gate_init,
            "noun_key_gate_init": model.noun_key_gate_init,
            "assignment_loss_weight": model.assignment_loss_weight,
            "spatial_bias_strength": model.spatial_bias_strength,
            "spatial_bias_dropout_prob": model.spatial_bias_dropout_prob,
            "content_delta_policy": "subtract_direct_learned_query_template_smoke_mvp",
            "legacy_entity_adapter_used": False,
        }
    )
    report["weight_sources"].pop("scheme_d_stage1b", None)
    report["weight_sources"]["scheme_e_v4"] = {"path": None, "loaded": False}
    return report


def main() -> None:
    args = tvn.prepare_args(build_parser().parse_args())
    if args.stage1a_init_from or args.stage2_init_from or args.stage2_resume_from:
        raise ValueError("Scheme-E v4 smoke starts fresh; omit all resume/init checkpoints")
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)
    dataset = replay.build_dataset(args)
    if accelerator.is_main_process and hasattr(dataset, "dataset_stats"):
        accelerator.print(f"Replay mixture: {dataset.dataset_stats}")
    model = build_model(args, accelerator)
    report = build_module_report(model, args)
    if accelerator.is_main_process:
        report_path = Path(args.output_path).expanduser().resolve() / "scheme_e_v4_module_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        accelerator.print("[scheme-e-v4-module-report] " + json.dumps(report))
    replay.base._log_stage_summary(accelerator, model, args)
    accelerator.print(
        "Scheme-E v4: grouped-grounded routing, "
        f"K={args.tube_num_tokens}, blocks={model.object_block_ids}, "
        f"gate={args.grounded_gate_init}, assignment={args.assignment_loss_weight}"
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
    finally:
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
