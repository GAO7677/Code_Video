"""Stage1B context-only *no-GT-box* trainer on the DiffSynth-native v_newtrain framework.

This script ports
    object_token_teacher_student/run_train_teacher_student_stage1b_context_only_no_gt_box.sh
    (-> ContextOnlyInjectionNoGTBoxTrainer)
onto the same DiffSynth-native training framework used by
    run_train_v_newtrain_gpu2367.sh  (-> train_v_newtrain.WanTrainingModule)

It reuses `train_v_newtrain.py` wholesale (pipeline construction via
`ContextAwareWanVideoPipeline`, `enable_object_condition_branch`, CoTracker /
VGGT / JEPA adapters, ObjectTubeProjector -> ObjectConditionAdapter ->
`flow_match_context_sft_loss`, checkpoint / resume helpers) and changes only
what the "no GT box" variant requires:

  1. The object query-point / box priors come from a viewer-style GDINO + SAM2
     pseudo-box pipeline (`ViewerGroundingBoxProvider`) instead of the dataset
     GT boxes (`sample["context_boxes"]`).
  2. All GT-box aux losses (track / box / depth) are dropped. Only the main
     flow-match loss on the injected `object_context` is optimized, matching the
     source config where lambda_track_aux = lambda_box_aux = lambda_depth_aux = 0.

Trainable set: the DiT object-injection branch (object_embedding /
object_cross_attn / object_gate / norm4) + ObjectConditionAdapter. The base Wan
DiT, its LoRA (loaded from the raw-phys stage), the VAE, the text encoder, the
frozen Stage1A token builder (ObjectTubeProjector + ObjectAuxHeads) are all
frozen.

The frozen Stage1A token builder weights are initialized from the Stage1A `.pt`
checkpoint via ``--stage1a_init_from`` (object_pooler.* / object_aux_heads.*),
while the base Wan LoRA is loaded (frozen) from ``--lora_checkpoint``. These are
two independent sources, which is why this script provides its own ``main`` and
does not reuse ``train_v_newtrain.main`` (whose ``--head_resume_from`` would
overwrite ``lora_checkpoint``).
"""
from __future__ import annotations

import argparse

import torch

# Importing train_v_newtrain triggers its top-of-module
# ``sys.path.insert(0, --diffsynth_root)`` shim (it reads --diffsynth_root
# straight from sys.argv), so the DiffSynth-Studio checkout passed on the
# command line is the one used for every `diffsynth` import below.
import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.context_wan_v_newtrain import flow_match_context_sft_loss
from code_vjepa_vggt.data.kubric_no_gt_box_dataset import KubricNoGTBoxDataset
from code_vjepa_vggt.headonly_val_loss import HeadOnlyValConfig
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.train0710querypoints.gt_mask_query_repair import GTMaskRepairConfig
from code_vjepa_vggt.train0710querypoints.gt_mask_query_repair import repair_grouped_queries_with_gt_masks
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache

from diffsynth.diffusion import ModelLogger


_DUMMY_BOX_XYXY = (0.45, 0.45, 0.55, 0.55)


def _summarize_object_branch_trace(trace_layers: list[dict] | None) -> dict[str, float]:
    if not trace_layers:
        return {
            "max_gated_to_x_ratio_l2": 0.0,
            "mean_gated_to_x_ratio_l2": 0.0,
            "max_pre_guard_gated_to_x_ratio_l2": 0.0,
            "mean_pre_guard_gated_to_x_ratio_l2": 0.0,
            "max_ratio_block_id": -1.0,
            "guard_applied_layer_count": 0.0,
            "guard_scale_min": 1.0,
        }
    ratios = [float(layer.get("gated_to_x_ratio_l2", 0.0)) for layer in trace_layers]
    pre_guard_ratios = [
        float(layer.get("pre_guard_gated_to_x_ratio_l2", layer.get("gated_to_x_ratio_l2", 0.0)))
        for layer in trace_layers
    ]
    max_idx = max(range(len(ratios)), key=ratios.__getitem__)
    guard_infos = [
        layer.get("object_ratio_guard", {})
        for layer in trace_layers
        if isinstance(layer.get("object_ratio_guard", {}), dict)
    ]
    guard_applied = [info for info in guard_infos if bool(info.get("applied", False))]
    guard_scales = [float(info.get("scale", 1.0)) for info in guard_infos]
    return {
        "max_gated_to_x_ratio_l2": float(max(ratios)),
        "mean_gated_to_x_ratio_l2": float(sum(ratios) / max(len(ratios), 1)),
        "max_pre_guard_gated_to_x_ratio_l2": float(max(pre_guard_ratios)),
        "mean_pre_guard_gated_to_x_ratio_l2": float(sum(pre_guard_ratios) / max(len(pre_guard_ratios), 1)),
        "max_ratio_block_id": float(trace_layers[max_idx].get("block_id", -1)),
        "guard_applied_layer_count": float(len(guard_applied)),
        "guard_scale_min": float(min(guard_scales)) if guard_scales else 1.0,
    }


def prepare_jepa_context_video(
    context_video: torch.Tensor,
    *,
    latent_frames: int,
    tubelet_size: int = 2,
) -> tuple[torch.Tensor, dict[str, int | bool]]:
    """Pad JEPA input time so JEPA output frames align with latent time."""
    if context_video.ndim != 5:
        raise ValueError(f"context_video must be [B,C,T,H,W], got {list(context_video.shape)}")
    latent_frames = max(int(latent_frames), 1)
    tubelet_size = max(int(tubelet_size), 1)
    input_frames = int(context_video.shape[2])
    target_frames = max(input_frames, tubelet_size)
    while (target_frames // tubelet_size) % latent_frames != 0:
        target_frames += 1
    if target_frames == input_frames:
        return context_video, {
            "duplicated_for_jepa": False,
            "input_context_frames": input_frames,
            "jepa_context_frames": input_frames,
            "latent_frames": latent_frames,
            "tubelet_size": tubelet_size,
            "padded_context_frames": 0,
        }
    pad_count = target_frames - input_frames
    pad_frames = context_video[:, :, -1:, :, :].expand(-1, -1, pad_count, -1, -1)
    aligned = torch.cat([context_video, pad_frames], dim=2)
    return aligned, {
        "duplicated_for_jepa": True,
        "input_context_frames": input_frames,
        "jepa_context_frames": int(aligned.shape[2]),
        "latent_frames": latent_frames,
        "tubelet_size": tubelet_size,
        "padded_context_frames": pad_count,
    }


def compact_object_context_valid_slots(
    object_context: torch.Tensor,
    object_valid_mask: torch.Tensor,
) -> torch.Tensor | None:
    """Physically remove invalid slot tokens before object cross-attention."""
    if object_context.ndim != 3:
        raise ValueError(
            f"object_context must be [B,T*O,D], got {list(object_context.shape)}"
        )
    if object_valid_mask.ndim != 2:
        raise ValueError(
            f"object_valid_mask must be [B,O], got {list(object_valid_mask.shape)}"
        )
    batch, sequence_length, dim = object_context.shape
    if int(object_valid_mask.shape[0]) != int(batch):
        raise ValueError("object_context and object_valid_mask batch sizes differ")
    slots = int(object_valid_mask.shape[1])
    if slots <= 0 or int(sequence_length) % slots != 0:
        raise ValueError(
            f"object_context sequence length {sequence_length} is not divisible by slots={slots}"
        )
    time_steps = int(sequence_length) // slots
    valid_ids = [
        torch.nonzero(object_valid_mask[b] > 0.5, as_tuple=False).flatten()
        for b in range(int(batch))
    ]
    valid_counts = [int(ids.numel()) for ids in valid_ids]
    if not valid_counts or max(valid_counts) == 0:
        return None
    if len(set(valid_counts)) != 1:
        raise ValueError(
            "compact object context requires equal valid-slot counts in a batch; "
            f"got {valid_counts}"
        )
    context_bto = object_context.view(int(batch), time_steps, slots, int(dim))
    compacted = [context_bto[b, :, ids, :] for b, ids in enumerate(valid_ids)]
    return torch.stack(compacted, dim=0).reshape(
        int(batch), time_steps * valid_counts[0], int(dim)
    )


class ContextOnlyNoGTBoxWanModule(tvn.WanTrainingModule):
    """WanTrainingModule variant that sources object priors from viewer grounding."""

    def __init__(self, *args, grounding_config: dict | None = None, **kwargs) -> None:
        self.lambda_object_gate_reg = float(kwargs.pop("lambda_object_gate_reg", 0.0))
        self.object_gate_reg_target = float(kwargs.pop("object_gate_reg_target", 0.20))
        self.object_slot_dropout_prob = float(kwargs.pop("object_slot_dropout_prob", 0.0))
        self.full_slot_loss_weight = float(kwargs.pop("full_slot_loss_weight", 1.0))
        self.compact_object_context_slots = bool(
            kwargs.pop("compact_object_context_slots", False)
        )
        self.lambda_object_adapter_mlp_reg = float(
            kwargs.pop("lambda_object_adapter_mlp_reg", 0.0)
        )
        self.object_adapter_mlp_reg_target = float(
            kwargs.pop("object_adapter_mlp_reg_target", 3.0)
        )
        self.object_adapter_mlp_residual_max_ratio = float(
            kwargs.pop("object_adapter_mlp_residual_max_ratio", 0.0)
        )
        if not 0.0 <= self.object_slot_dropout_prob <= 1.0:
            raise ValueError("object_slot_dropout_prob must be in [0, 1]")
        if self.full_slot_loss_weight <= 0.0:
            raise ValueError("full_slot_loss_weight must be positive")
        if self.lambda_object_adapter_mlp_reg < 0.0:
            raise ValueError("lambda_object_adapter_mlp_reg must be non-negative")
        if self.object_adapter_mlp_reg_target <= 0.0:
            raise ValueError("object_adapter_mlp_reg_target must be positive")
        if self.object_adapter_mlp_residual_max_ratio < 0.0:
            raise ValueError("object_adapter_mlp_residual_max_ratio must be non-negative")
        self.object_branch_train_trace = bool(kwargs.pop("object_branch_train_trace", False))
        self.object_branch_ratio_guard_max_ratio = float(
            kwargs.pop("object_branch_ratio_guard_max_ratio", 0.0)
        )
        self.object_branch_ratio_guard_max_block_id = int(
            kwargs.pop("object_branch_ratio_guard_max_block_id", -1)
        )
        self._jepa_tubelet_size = int(kwargs.get("jepa_tubelet_size", 2))
        super().__init__(*args, **kwargs)
        self.viewer_grounding: ViewerGroundingBoxProvider | None = None
        self.gt_mask_query_repair = GTMaskRepairConfig(
            enabled=bool((grounding_config or {}).get("grounding_gt_mask_query_repair", False)),
            oversample_factor=int((grounding_config or {}).get("grounding_gt_mask_oversample_factor", 4)),
            min_visible_ratio=float((grounding_config or {}).get("grounding_gt_mask_min_visible_ratio", 0.60)),
            min_in_mask_ratio=float((grounding_config or {}).get("grounding_gt_mask_min_in_mask_ratio", 0.60)),
            color_tolerance=int((grounding_config or {}).get("grounding_gt_mask_color_tolerance", 18)),
        )
        if self.enable_object_branch:
            self.object_adapter.mlp_residual_max_ratio = (
                float(self.object_adapter_mlp_residual_max_ratio)
                if float(self.object_adapter_mlp_residual_max_ratio) > 0.0
                else None
            )
            active_dit = getattr(self.pipe, "dit", None)
            if active_dit is not None and hasattr(active_dit, "_object_branch_ratio_guard_max_ratio"):
                active_dit._object_branch_ratio_guard_max_ratio = (
                    float(self.object_branch_ratio_guard_max_ratio)
                    if float(self.object_branch_ratio_guard_max_ratio) > 0.0
                    else None
                )
                active_dit._object_branch_ratio_guard_max_block_id = (
                    int(self.object_branch_ratio_guard_max_block_id)
                    if int(self.object_branch_ratio_guard_max_block_id) >= 0
                    else None
                )
            cfg = dict(grounding_config or {})
            grounding_device = str(cfg.get("grounding_device") or self.pipe.device)
            self.viewer_grounding = ViewerGroundingBoxProvider(
                device=grounding_device,
                segment_len=int(cfg.get("sam2_segment_len", 8)),
                max_objects=int(self.aux_max_objects),
                points_per_object=int(self.object_num_queries),
                proposal_source=str(cfg.get("grounding_proposal_source", "gdino_only")),
                motion_score_ratio=float(cfg.get("grounding_motion_score_ratio", 0.15)),
                text_prompt=str(
                    cfg.get(
                        "grounding_text_prompt",
                        "box . cube . block . cylinder . capsule . sphere . ball .",
                    )
                ),
                extra_prompt_terms=str(cfg.get("grounding_extra_prompt_terms", "")),
                include_caption_terms=not bool(cfg.get("grounding_disable_caption_terms", True)),
                gdino_box_threshold=float(cfg.get("grounding_gdino_box_threshold", 0.20)),
                gdino_text_threshold=float(cfg.get("grounding_gdino_text_threshold", 0.15)),
                prompt_frame_mode=str(cfg.get("grounding_prompt_frame_mode", "first")),
                track_dedupe_iou_threshold=float(cfg.get("grounding_track_dedupe_iou_threshold", 0.75)),
                container_suppress_ratio_threshold=float(
                    cfg.get("grounding_container_suppress_ratio_threshold", 0.95)
                ),
                container_suppress_min_contained=int(
                    cfg.get("grounding_container_suppress_min_contained", 2)
                ),
                container_suppress_min_area_ratio=float(
                    cfg.get("grounding_container_suppress_min_area_ratio", 1.5)
                ),
                container_suppress_small_iou_threshold=float(
                    cfg.get("grounding_container_suppress_small_iou_threshold", 0.7)
                ),
            )

    def _compute_object_gate_regularizer(self, pipe) -> tuple[torch.Tensor, dict[str, float]]:
        gate_abs_means = []
        gate_abs_maxes = []
        for block in getattr(pipe.dit, "blocks", []):
            object_gate = getattr(block, "object_gate", None)
            if object_gate is None:
                continue
            gate_tanh_abs = torch.tanh(object_gate.float()).abs()
            gate_abs_means.append(gate_tanh_abs.mean())
            gate_abs_maxes.append(gate_tanh_abs.max())
        if not gate_abs_means:
            zero = torch.zeros((), device=pipe.device, dtype=pipe.torch_dtype)
            return zero, {
                "train/object_gate_tanh_abs_mean": 0.0,
                "train/object_gate_tanh_abs_max": 0.0,
            }
        gate_abs_mean = torch.stack(gate_abs_means).mean()
        gate_abs_max = torch.stack(gate_abs_maxes).max()
        gate_excess = torch.relu(torch.stack(gate_abs_maxes) - float(self.object_gate_reg_target))
        gate_reg = gate_excess.square().mean().to(device=pipe.device, dtype=pipe.torch_dtype)
        return gate_reg, {
            "train/object_gate_tanh_abs_mean": float(gate_abs_mean.detach().item()),
            "train/object_gate_tanh_abs_max": float(gate_abs_max.detach().item()),
        }

    def _consume_object_adapter_mlp_regularizer(
        self,
        pipe,
        object_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ratio, diagnostics = self.object_adapter.pop_mlp_diagnostics(object_valid_mask)
        if ratio is None:
            zero = torch.zeros((), device=pipe.device, dtype=pipe.torch_dtype)
            return zero, {
                "train/object_adapter_mlp_residual_ratio_mean": 0.0,
                "train/object_adapter_mlp_residual_ratio_max": 0.0,
                "train/object_adapter_mlp_cap_applied_fraction": 0.0,
                "train/object_adapter_mlp_cap_scale_min": 1.0,
            }
        excess = torch.relu(ratio - float(self.object_adapter_mlp_reg_target))
        regularizer = excess.square().mean().to(device=pipe.device, dtype=pipe.torch_dtype)
        return regularizer, {
            "train/object_adapter_mlp_residual_ratio_mean": diagnostics["mean_ratio"],
            "train/object_adapter_mlp_residual_ratio_max": diagnostics["max_ratio"],
            "train/object_adapter_mlp_cap_applied_fraction": diagnostics["cap_applied_fraction"],
            "train/object_adapter_mlp_cap_scale_min": diagnostics["cap_scale_min"],
        }

    def _run_main_loss_with_trace(self, pipe, inputs_shared, inputs_posi, object_context):
        active_dit = getattr(pipe, "dit", None)
        trace_layers = None
        trace_enabled = bool(
            self.object_branch_train_trace
            or float(self.object_branch_ratio_guard_max_ratio) > 0.0
        )
        if active_dit is not None and hasattr(active_dit, "_object_branch_trace_collect") and trace_enabled:
            active_dit._object_branch_trace_collect = True
            active_dit._object_branch_trace_buffer = []
        try:
            loss_main = flow_match_context_sft_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
                object_context=object_context,
            )
            if active_dit is not None and hasattr(active_dit, "_object_branch_trace_buffer"):
                trace_layers = getattr(active_dit, "_object_branch_trace_buffer", None)
        finally:
            if active_dit is not None and hasattr(active_dit, "_object_branch_trace_collect"):
                active_dit._object_branch_trace_collect = False
                active_dit._object_branch_trace_buffer = None
        return loss_main, trace_layers

    def _apply_object_slot_dropout(self, object_valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        original_count = int((object_valid_mask > 0.5).sum().item())
        sampled_mask = object_valid_mask
        dropout_applied = False
        if original_count > 1 and self.object_slot_dropout_prob > 0.0:
            if bool(torch.rand((), device=object_valid_mask.device) < self.object_slot_dropout_prob):
                valid_ids = torch.nonzero(object_valid_mask[0] > 0.5, as_tuple=False).flatten()
                keep_count = int(torch.randint(1, original_count, (), device=object_valid_mask.device).item())
                keep_ids = valid_ids[torch.randperm(original_count, device=object_valid_mask.device)[:keep_count]]
                sampled_mask = torch.zeros_like(object_valid_mask)
                sampled_mask[0, keep_ids] = object_valid_mask[0, keep_ids]
                dropout_applied = True
        sampled_count = int((sampled_mask > 0.5).sum().item())
        full_slot_sample = original_count == int(self.aux_max_objects) and sampled_count == original_count
        return sampled_mask, {
            "train/object_count_before_dropout": float(original_count),
            "train/object_count_after_dropout": float(sampled_count),
            "train/object_slot_dropout_applied": 1.0 if dropout_applied else 0.0,
            "train/object_full_slot_sample": 1.0 if full_slot_sample else 0.0,
            "train/object_main_loss_weight": float(self.full_slot_loss_weight if full_slot_sample else 1.0),
        }

    # ------------------------------------------------------------------
    # Override 1: query priors come from viewer grounding, not GT boxes.
    # Returns the exact same 4-tuple / shapes as the base implementation:
    #   flat            [1, total_object_queries, 2]  pixel query points
    #   frame_ids       [1, total_object_queries, 1]  prompt-frame index per query
    #   object_valid    [1, aux_max_objects]          1.0 / 0.0
    #   box_prior_xyxy  [1, aux_max_objects, 4]        normalized xyxy
    # ------------------------------------------------------------------
    def _build_object_query_priors(self, sample: dict, *, image_hw: tuple[int, int]):
        if self.viewer_grounding is None:
            raise RuntimeError("viewer grounding provider is not initialized")

        height, width = int(image_hw[0]), int(image_hw[1])
        num_context_frames = int(sample["num_context_frames"])
        caption = str(sample.get("caption", ""))

        # sample["context_video"]: [C, Tc, H, W] in [-1, 1] -> [Tc, C, H, W] in [0, 1]
        context_video = sample["context_video"]
        valid_frames = max(int(num_context_frames), 1)
        frames_tchw_01 = (
            ((context_video[:, :valid_frames].permute(1, 0, 2, 3).float() + 1.0) / 2.0)
            .clamp(0.0, 1.0)
            .cpu()
            .numpy()
        )

        grounding_sample = self.viewer_grounding.build_sample(
            frames_tchw_01=frames_tchw_01,
            caption=caption,
            image_hw=(height, width),
        )
        self._last_grounding_debug = dict(getattr(grounding_sample, "debug", {}) or {})

        repair_debug = {"applied": False, "reason": "disabled"}
        if bool(self.gt_mask_query_repair.enabled):
            repaired_queries_px, repair_debug = repair_grouped_queries_with_gt_masks(
                sample=sample,
                image_hw=(height, width),
                frames_bthwc_01=torch.from_numpy(frames_tchw_01).permute(0, 2, 3, 1).unsqueeze(0).float(),
                grouped_queries_px=grounding_sample.grouped_queries_px,
                object_valid_mask=grounding_sample.object_valid_mask,
                object_tracks=getattr(grounding_sample, "object_tracks", []),
                prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
                points_per_object=int(self.object_num_queries),
                run_cotracker=self._run_cotracker,
                config=self.gt_mask_query_repair,
            )
            grounding_sample.grouped_queries_px = repaired_queries_px
            if isinstance(getattr(grounding_sample, "debug", None), dict):
                grounding_sample.debug["gt_mask_query_repair"] = repair_debug
                self._last_grounding_debug = dict(grounding_sample.debug)

        # grouped_queries_px: [aux_max_objects, object_num_queries, 2] (pixels)
        grouped_queries = torch.from_numpy(grounding_sample.grouped_queries_px).float()
        object_valid_mask = torch.from_numpy(grounding_sample.object_valid_mask).float()
        # context_boxes_norm: [Tc, aux_max_objects, 4] (normalized xyxy)
        context_boxes_norm = torch.from_numpy(grounding_sample.context_boxes_norm).float()
        prompt_frame_idx = int(getattr(grounding_sample, "prompt_frame_idx", 0))

        flat = grouped_queries.view(1, self.total_object_queries, 2)

        box_priors = []
        frame_ids = []
        for object_idx in range(int(self.aux_max_objects)):
            is_valid = bool(object_valid_mask[object_idx].item() > 0.5)
            first_valid_frame = 0
            box = None
            if is_valid:
                for frame_idx in range(min(valid_frames, int(context_boxes_norm.shape[0]))):
                    candidate = context_boxes_norm[frame_idx, object_idx]
                    if bool(
                        (candidate[2] - candidate[0] > 1.0e-6)
                        and (candidate[3] - candidate[1] > 1.0e-6)
                    ):
                        first_valid_frame = frame_idx
                        box = candidate
                        break
            if box is None:
                box = torch.tensor(_DUMMY_BOX_XYXY, dtype=torch.float32)
                first_valid_frame = prompt_frame_idx if is_valid else 0
            box_priors.append(box.to(dtype=torch.float32))
            frame_ids.extend([float(first_valid_frame)] * int(self.object_num_queries))

        box_prior_xyxy = torch.stack(box_priors, dim=0).view(1, int(self.aux_max_objects), 4)
        frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.float32).view(
            1, self.total_object_queries, 1
        )
        object_valid = object_valid_mask.view(1, int(self.aux_max_objects))
        return flat, frame_ids_tensor, object_valid, box_prior_xyxy

    # ------------------------------------------------------------------
    # Override 2: object_context main loss only, no GT-box aux losses.
    # Mirrors the object_context-producing half of the base
    # _compute_object_losses (context_video -> priors -> CoTracker/VGGT/JEPA ->
    # ObjectTubeProjector -> ObjectConditionAdapter -> flow_match loss) and drops
    # the entire sample["context_boxes"] aux-supervision block.
    # ------------------------------------------------------------------
    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if not self.enable_object_branch:
            return flow_match_context_sft_loss(pipe, **inputs_shared, **inputs_posi), {}

        sample = inputs_shared["raw_sample"]
        num_context_frames = int(sample.get("num_context_frames", 0))
        context_frame_indices = sample.get("context_frame_indices", None)
        if isinstance(context_frame_indices, torch.Tensor) and int(context_frame_indices.numel()) > 0:
            sampled_ctx_last_index = float(context_frame_indices.max().item())
        else:
            sampled_ctx_last_index = -1.0
        ctx_max_length = float(sample.get("ctx_max_length", -1))
        if num_context_frames <= 0:
            object_context = torch.zeros(
                (1, int(self.aux_max_objects), int(self.object_adapter.dim)),
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
            object_gate_reg, gate_metrics = self._compute_object_gate_regularizer(pipe)
            object_adapter_mlp_reg = object_context.new_zeros(())
            adapter_mlp_metrics = {
                "train/object_adapter_mlp_residual_ratio_mean": 0.0,
                "train/object_adapter_mlp_residual_ratio_max": 0.0,
                "train/object_adapter_mlp_cap_applied_fraction": 0.0,
                "train/object_adapter_mlp_cap_scale_min": 1.0,
            }
            if self.lambda_main > 0.0:
                loss_main, trace_layers = self._run_main_loss_with_trace(
                    pipe,
                    inputs_shared,
                    inputs_posi,
                    object_context,
                )
            else:
                loss_main = object_context.new_zeros(())
                trace_layers = None
            object_context_reg = object_context.new_zeros(())
            trace_summary = _summarize_object_branch_trace(trace_layers)
            total = (
                self.lambda_main * loss_main
                + self.lambda_object_context_reg * object_context_reg
                + self.lambda_object_gate_reg * object_gate_reg
                + self.lambda_object_adapter_mlp_reg * object_adapter_mlp_reg
            )
            metrics = {
                "train/loss_total": float(total.detach().item()),
                "train/loss_main": float(loss_main.detach().item()),
                "train/loss_object_context_reg": float(object_context_reg.detach().item()),
                "train/loss_object_gate_reg": float(object_gate_reg.detach().item()),
                "train/loss_object_adapter_mlp_reg": float(
                    object_adapter_mlp_reg.detach().item()
                ),
                "train/object_count": 0.0,
                "train/object_count_before_dropout": 0.0,
                "train/object_count_after_dropout": 0.0,
                "train/object_slot_dropout_applied": 0.0,
                "train/object_full_slot_sample": 0.0,
                "train/object_main_loss_weight": 1.0,
                "train/object_latent_tokens_abs_max": 0.0,
                "train/object_context_abs_max": 0.0,
                "train/object_context_abs_mean": 0.0,
                "train/object_branch_max_gated_to_x_ratio_l2": trace_summary["max_gated_to_x_ratio_l2"],
                "train/object_branch_mean_gated_to_x_ratio_l2": trace_summary["mean_gated_to_x_ratio_l2"],
                "train/object_branch_max_pre_guard_gated_to_x_ratio_l2": trace_summary["max_pre_guard_gated_to_x_ratio_l2"],
                "train/object_branch_mean_pre_guard_gated_to_x_ratio_l2": trace_summary["mean_pre_guard_gated_to_x_ratio_l2"],
                "train/object_branch_max_ratio_block_id": trace_summary["max_ratio_block_id"],
                "train/object_branch_guard_applied_layer_count": trace_summary["guard_applied_layer_count"],
                "train/object_branch_guard_scale_min": trace_summary["guard_scale_min"],
                "train/object_branch_ratio_guard_enabled": 1.0 if float(self.object_branch_ratio_guard_max_ratio) > 0.0 else 0.0,
                "train/ctx_max_length": ctx_max_length,
                "train/sampled_ctx_last_index": sampled_ctx_last_index,
                "train/sampled_ctx_num_frames": 0.0,
            }
            metrics.update(gate_metrics)
            metrics.update(adapter_mlp_metrics)
            return total, metrics

        context_video = sample["context_video"].unsqueeze(0).to(
            device=pipe.device, dtype=pipe.torch_dtype
        )
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

        query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = (
            self._build_object_query_priors(sample, image_hw=image_hw)
        )
        query_points_prior = query_points_prior.to(device=pipe.device, dtype=pipe.torch_dtype)
        query_frame_ids = query_frame_ids.to(device=pipe.device, dtype=pipe.torch_dtype)
        object_valid_mask = object_valid_mask.to(device=pipe.device, dtype=pipe.torch_dtype)
        box_prior_xyxy = box_prior_xyxy.to(device=pipe.device, dtype=pipe.torch_dtype)
        object_valid_mask, slot_metrics = self._apply_object_slot_dropout(object_valid_mask)

        frames_bthwc_01 = (
            (context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        cotracker_out = self._run_cotracker(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )

        vggt_out = None
        if self.vggt_cache_root:
            vggt_out = load_vggt_cache(sample, self.vggt_cache_root, allow_missing=False)
            if vggt_out is None:
                raise RuntimeError(
                    "VGGT cache root is set but no cache found for sample "
                    f"{sample.get('video_path', '<unknown>')}"
                )
        else:
            vggt_out = self._run_vggt(
                frames_bthwc_01,
                query_points_prior=query_points_prior,
                query_image_hw=image_hw,
            )

        tracks_grouped, visibility_grouped, confidence_grouped = self._group_tracks_to_objects(
            cotracker_out.tracks,
            cotracker_out.visibility,
            cotracker_out.confidence,
            max_objects=self.aux_max_objects,
            points_per_object=self.object_num_queries,
        )
        context_latents = inputs_shared["clean_prefix_latents"]
        jepa_input_video, jepa_ctx_fix = prepare_jepa_context_video(
            context_video,
            latent_frames=int(context_latents.shape[2]),
            tubelet_size=int(self._jepa_tubelet_size),
        )
        jepa_out = self._run_jepa(jepa_input_video)
        object_out = self.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latents,
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
        object_context = self.object_adapter(
            object_out.object_latent_tokens,
            object_valid_mask=object_valid_mask,
            bbox_xyxy=object_out.active_box_xyxy,
        )
        object_context_for_dit = (
            compact_object_context_valid_slots(object_context, object_valid_mask)
            if self.compact_object_context_slots
            else object_context
        )
        object_adapter_mlp_reg, adapter_mlp_metrics = (
            self._consume_object_adapter_mlp_regularizer(pipe, object_valid_mask)
        )

        object_gate_reg, gate_metrics = self._compute_object_gate_regularizer(pipe)
        if self.lambda_main > 0.0:
            loss_main, trace_layers = self._run_main_loss_with_trace(
                pipe,
                inputs_shared,
                inputs_posi,
                object_context_for_dit,
            )
        else:
            loss_main = object_context.new_zeros(())
            trace_layers = None
        object_context_reg = object_context.square().mean()
        trace_summary = _summarize_object_branch_trace(trace_layers)
        main_loss_weight = float(slot_metrics["train/object_main_loss_weight"])

        total = (
            self.lambda_main * main_loss_weight * loss_main
            + self.lambda_object_context_reg * object_context_reg
            + self.lambda_object_gate_reg * object_gate_reg
            + self.lambda_object_adapter_mlp_reg * object_adapter_mlp_reg
        )

        object_context_abs = object_context.detach().abs()
        object_latent_tokens_abs = object_out.object_latent_tokens.detach().abs()
        metrics = {
            "train/loss_total": float(total.detach().item()),
            "train/loss_main": float(loss_main.detach().item()),
            "train/loss_object_context_reg": float(object_context_reg.detach().item()),
            "train/loss_object_gate_reg": float(object_gate_reg.detach().item()),
            "train/loss_object_adapter_mlp_reg": float(
                object_adapter_mlp_reg.detach().item()
            ),
            "train/object_count": float(object_valid_mask.sum().item()),
            "train/object_latent_tokens_abs_max": float(object_latent_tokens_abs.max().item()),
            "train/object_context_abs_max": float(object_context_abs.max().item()),
            "train/object_context_abs_mean": float(object_context_abs.mean().item()),
            "train/object_context_attention_tokens": float(
                0 if object_context_for_dit is None else object_context_for_dit.shape[1]
            ),
            "train/object_context_compact_slots": float(self.compact_object_context_slots),
            "train/object_branch_max_gated_to_x_ratio_l2": trace_summary["max_gated_to_x_ratio_l2"],
            "train/object_branch_mean_gated_to_x_ratio_l2": trace_summary["mean_gated_to_x_ratio_l2"],
            "train/object_branch_max_pre_guard_gated_to_x_ratio_l2": trace_summary["max_pre_guard_gated_to_x_ratio_l2"],
            "train/object_branch_mean_pre_guard_gated_to_x_ratio_l2": trace_summary["mean_pre_guard_gated_to_x_ratio_l2"],
            "train/object_branch_max_ratio_block_id": trace_summary["max_ratio_block_id"],
            "train/object_branch_guard_applied_layer_count": trace_summary["guard_applied_layer_count"],
            "train/object_branch_guard_scale_min": trace_summary["guard_scale_min"],
            "train/object_branch_ratio_guard_enabled": 1.0 if float(self.object_branch_ratio_guard_max_ratio) > 0.0 else 0.0,
            "train/jepa_input_frames": float(jepa_ctx_fix["jepa_context_frames"]),
            "train/jepa_padding_frames": float(jepa_ctx_fix["padded_context_frames"]),
            "train/ctx_max_length": ctx_max_length,
            "train/sampled_ctx_last_index": sampled_ctx_last_index,
            "train/sampled_ctx_num_frames": float(num_context_frames),
        }
        metrics.update(gate_metrics)
        metrics.update(adapter_mlp_metrics)
        metrics.update(slot_metrics)
        return total, metrics


# ----------------------------------------------------------------------
# Argument parsing: reuse the full v_newtrain parser and append the
# viewer-grounding knobs + a dedicated Stage1A init flag.
# ----------------------------------------------------------------------
def _patch_general_config_conflict_handler() -> None:
    """Make tvn.wan_parser tolerant of duplicate options.

    WAN_2p2's ``add_general_config`` defines some options (e.g. ``--wandb_project``)
    that ``tvn.wan_parser`` also re-declares explicitly. The older DiffSynth
    checkout the reference script used did not, so this only surfaces against the
    WAN_2p2 framework. We wrap ``tvn.add_general_config`` so the parser it returns
    resolves duplicate option strings (later declaration wins) instead of raising
    ``argparse.ArgumentError``. train_v_newtrain.py is left untouched.
    """
    if getattr(tvn, "_stage1b_conflict_patched", False):
        return
    _orig_add_general_config = tvn.add_general_config

    def _patched_add_general_config(parser: argparse.ArgumentParser):
        parser = _orig_add_general_config(parser)
        parser.conflict_handler = "resolve"
        parser._optionals.conflict_handler = "resolve"
        return parser

    tvn.add_general_config = _patched_add_general_config
    tvn._stage1b_conflict_patched = True


def build_parser() -> argparse.ArgumentParser:
    _patch_general_config_conflict_handler()
    parser = tvn.wan_parser()
    parser.description = (
        "Stage1B context-only NO-GT-BOX training on the DiffSynth-native "
        "v_newtrain framework (viewer GDINO+SAM2 pseudo boxes)."
    )
    for action in parser._actions:
        if action.dest == "dataset_type":
            action.choices = [*action.choices, "kubric_no_gt_box"]
            break

    group = parser.add_argument_group("stage1b_no_gt_box")
    group.add_argument(
        "--stage1a_init_from",
        default=None,
        help="Stage1A checkpoint (.pt) providing frozen object_pooler / object_aux_heads weights.",
    )
    group.add_argument("--grounding_device", default=None)
    group.add_argument("--sam2_segment_len", type=int, default=8)
    group.add_argument("--grounding_proposal_source", default="gdino_only")
    group.add_argument("--grounding_motion_score_ratio", type=float, default=0.15)
    group.add_argument(
        "--grounding_text_prompt",
        default="box . cube . block . cylinder . capsule . sphere . ball .",
    )
    group.add_argument("--grounding_extra_prompt_terms", default="")
    group.add_argument(
        "--grounding_disable_caption_terms",
        action="store_true",
        default=True,
        help="Disable caption-derived prompt terms for viewer grounding (default: on).",
    )
    group.add_argument(
        "--grounding_enable_caption_terms",
        dest="grounding_disable_caption_terms",
        action="store_false",
        help="Enable caption-derived prompt terms for viewer grounding.",
    )
    group.add_argument("--grounding_gdino_box_threshold", type=float, default=0.20)
    group.add_argument("--grounding_gdino_text_threshold", type=float, default=0.15)
    group.add_argument("--grounding_prompt_frame_mode", default="first")
    group.add_argument("--grounding_track_dedupe_iou_threshold", type=float, default=0.75)
    group.add_argument("--grounding_container_suppress_ratio_threshold", type=float, default=0.95)
    group.add_argument("--grounding_container_suppress_min_contained", type=int, default=2)
    group.add_argument("--grounding_container_suppress_min_area_ratio", type=float, default=1.5)
    group.add_argument("--grounding_container_suppress_small_iou_threshold", type=float, default=0.7)
    group.add_argument("--grounding_gt_mask_query_repair", action="store_true", default=False)
    group.add_argument("--grounding_gt_mask_oversample_factor", type=int, default=4)
    group.add_argument("--grounding_gt_mask_min_visible_ratio", type=float, default=0.60)
    group.add_argument("--grounding_gt_mask_min_in_mask_ratio", type=float, default=0.60)
    group.add_argument("--grounding_gt_mask_color_tolerance", type=int, default=18)
    group.add_argument(
        "--object_branch_train_trace",
        action="store_true",
        help="Enable per-step object-branch ratio diagnostics during training.",
    )
    group.add_argument(
        "--object_branch_ratio_guard_max_ratio",
        type=float,
        default=0.0,
        help="If >0, cap per-block gated object residual L2 ratio to this value during training.",
    )
    group.add_argument(
        "--object_branch_ratio_guard_max_block_id",
        type=int,
        default=-1,
        help="Optional inclusive max block id for the training-time ratio guard; negative means all blocks.",
    )
    group.add_argument(
        "--lambda_object_gate_reg",
        type=float,
        default=0.0,
        help="Weight for penalizing oversized tanh(object_gate) values.",
    )
    group.add_argument(
        "--object_gate_reg_target",
        type=float,
        default=0.20,
        help="No penalty below this tanh(object_gate) magnitude target.",
    )
    group.add_argument(
        "--object_slot_dropout_prob",
        type=float,
        default=0.0,
        help="Probability of keeping a random non-empty subset of detected object slots.",
    )
    group.add_argument(
        "--full_slot_loss_weight",
        type=float,
        default=1.0,
        help="Main-loss multiplier for undropped samples using every configured object slot.",
    )
    group.add_argument(
        "--compact_object_context_slots",
        action="store_true",
        help="Physically remove invalid or dropped slot tokens before DiT cross-attention.",
    )
    group.add_argument(
        "--lambda_object_adapter_mlp_reg",
        type=float,
        default=0.0,
        help="Weight for penalizing ObjectConditionAdapter MLP residual ratios above the target.",
    )
    group.add_argument(
        "--object_adapter_mlp_reg_target",
        type=float,
        default=3.0,
        help="No adapter MLP residual-ratio penalty below this per-token RMS ratio.",
    )
    group.add_argument(
        "--object_adapter_mlp_residual_max_ratio",
        type=float,
        default=0.0,
        help="If >0, cap the adapter MLP residual to this per-token RMS ratio before out_norm.",
    )

    kubric_group = parser.add_argument_group("kubric_no_gt_box_dataset")
    kubric_group.add_argument(
        "--kubric_root",
        type=str,
        default=None,
        help="Root directory of the extracted Kubric/PhyCo dataset.",
    )
    kubric_group.add_argument(
        "--kubric_split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="Stable hash split used by KubricNoGTBoxDataset.",
    )
    kubric_group.add_argument(
        "--kubric_sampling_strategy",
        type=str,
        default="prefix",
        choices=["prefix", "uniform"],
        help="How to select the target num_frames from each raw rgba.mp4.",
    )
    kubric_group.add_argument(
        "--kubric_scenario",
        action="append",
        default=None,
        help="Optional scenario filter; may be passed multiple times.",
    )
    kubric_group.add_argument(
        "--kubric_cache_root",
        type=str,
        default="/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset",
        help="Cache root for KubricNoGTBoxDataset indices.",
    )
    kubric_group.add_argument(
        "--kubric_split_train_ratio",
        type=float,
        default=0.9,
        help="Stable-hash train ratio used when kubric_split is train/val/test.",
    )
    kubric_group.add_argument(
        "--kubric_split_val_ratio",
        type=float,
        default=0.05,
        help="Stable-hash val ratio used when kubric_split is train/val/test.",
    )
    kubric_group.add_argument(
        "--kubric_init_scan_limit",
        type=int,
        default=None,
        help="Optional limit applied after indexing, useful for smoke tests.",
    )
    kubric_group.add_argument(
        "--kubric_max_retry_samples",
        type=int,
        default=8,
        help="How many neighboring samples to retry when a decoded video is invalid.",
    )
    return parser


def _grounding_config_from_args(args: argparse.Namespace) -> dict:
    return {
        "grounding_device": args.grounding_device,
        "sam2_segment_len": args.sam2_segment_len,
        "grounding_proposal_source": args.grounding_proposal_source,
        "grounding_motion_score_ratio": args.grounding_motion_score_ratio,
        "grounding_text_prompt": args.grounding_text_prompt,
        "grounding_extra_prompt_terms": args.grounding_extra_prompt_terms,
        "grounding_disable_caption_terms": args.grounding_disable_caption_terms,
        "grounding_gdino_box_threshold": args.grounding_gdino_box_threshold,
        "grounding_gdino_text_threshold": args.grounding_gdino_text_threshold,
        "grounding_prompt_frame_mode": args.grounding_prompt_frame_mode,
        "grounding_track_dedupe_iou_threshold": args.grounding_track_dedupe_iou_threshold,
        "grounding_container_suppress_ratio_threshold": args.grounding_container_suppress_ratio_threshold,
        "grounding_container_suppress_min_contained": args.grounding_container_suppress_min_contained,
        "grounding_container_suppress_min_area_ratio": args.grounding_container_suppress_min_area_ratio,
        "grounding_container_suppress_small_iou_threshold": args.grounding_container_suppress_small_iou_threshold,
        "grounding_gt_mask_query_repair": args.grounding_gt_mask_query_repair,
        "grounding_gt_mask_oversample_factor": args.grounding_gt_mask_oversample_factor,
        "grounding_gt_mask_min_visible_ratio": args.grounding_gt_mask_min_visible_ratio,
        "grounding_gt_mask_min_in_mask_ratio": args.grounding_gt_mask_min_in_mask_ratio,
        "grounding_gt_mask_color_tolerance": args.grounding_gt_mask_color_tolerance,
    }


def build_model(args: argparse.Namespace, accelerator) -> ContextOnlyNoGTBoxWanModule:
    """Same construction as train_v_newtrain.build_model, plus grounding_config."""
    grounding_config = _grounding_config_from_args(args)
    grounding_config["grounding_device"] = args.grounding_device or str(accelerator.device)
    return ContextOnlyNoGTBoxWanModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_sampling_profile=args.context_sampling_profile,
        min_context_frames=args.min_context_frames,
        max_context_ratio=args.max_context_ratio,
        context_frame_choices=args.context_frame_choices,
        context_length_sampling=args.context_length_sampling,
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
        ctx_max_length=args.ctx_max_length,
        enable_object_branch=args.enable_object_branch,
        object_num_queries=args.object_num_queries,
        aux_max_objects=args.aux_max_objects,
        jepa_ckpt_path=args.jepa_ckpt_path,
        jepa_input_size=args.jepa_input_size,
        jepa_patch_size=args.jepa_patch_size,
        jepa_tubelet_size=args.jepa_tubelet_size,
        cotracker_checkpoint=args.cotracker_checkpoint,
        cotracker_input_h=args.cotracker_input_h,
        cotracker_input_w=args.cotracker_input_w,
        cotracker_window_len=args.cotracker_window_len,
        vggt_model_path=args.vggt_model_path,
        vggt_input_h=args.vggt_input_h,
        vggt_input_w=args.vggt_input_w,
        vggt_cache_root=args.vggt_cache_root,
        object_aux_devices=args.object_aux_devices,
        train_vggt=args.train_vggt,
        object_pooler_latent_dim=args.object_pooler_latent_dim,
        cond_proj_dim=args.cond_proj_dim,
        jepa_window_radius=args.jepa_window_radius,
        latent_window_radius=args.latent_window_radius,
        object_track_delta_scale=args.object_track_delta_scale,
        object_track_gate_init=args.object_track_gate_init,
        object_box_delta_scale=args.object_box_delta_scale,
        object_box_wh_log_scale=args.object_box_wh_log_scale,
        object_box_wh_max_scale=args.object_box_wh_max_scale,
        object_min_box_px=args.object_min_box_px,
        object_gate_init=args.object_gate_init,
        lambda_main=args.lambda_main,
        lambda_track_aux=args.lambda_track_aux,
        lambda_box_aux=args.lambda_box_aux,
        lambda_depth_aux=args.lambda_depth_aux,
        lambda_track_box_aux=args.lambda_track_box_aux,
        lambda_track_iou_aux=args.lambda_track_iou_aux,
        lambda_track_anchor_reg=args.lambda_track_anchor_reg,
        lambda_box_anchor_reg=args.lambda_box_anchor_reg,
        lambda_object_context_reg=args.lambda_object_context_reg,
        lambda_object_gate_reg=args.lambda_object_gate_reg,
        train_object_pooler=args.train_object_pooler,
        train_object_aux_heads=args.train_object_aux_heads,
        train_object_adapter=args.train_object_adapter,
        train_object_dit_branch=args.train_object_dit_branch,
        freeze_non_object_trainables=args.freeze_non_object_trainables,
        depth_target_state_index=args.depth_target_state_index,
        depth_target_source=args.depth_target_source,
        depth_anything_cache_root=args.depth_anything_cache_root,
        object_gate_reg_target=args.object_gate_reg_target,
        object_slot_dropout_prob=args.object_slot_dropout_prob,
        full_slot_loss_weight=args.full_slot_loss_weight,
        compact_object_context_slots=args.compact_object_context_slots,
        lambda_object_adapter_mlp_reg=args.lambda_object_adapter_mlp_reg,
        object_adapter_mlp_reg_target=args.object_adapter_mlp_reg_target,
        object_adapter_mlp_residual_max_ratio=args.object_adapter_mlp_residual_max_ratio,
        object_branch_train_trace=args.object_branch_train_trace,
        object_branch_ratio_guard_max_ratio=args.object_branch_ratio_guard_max_ratio,
        object_branch_ratio_guard_max_block_id=args.object_branch_ratio_guard_max_block_id,
        grounding_config=grounding_config,
    )


def _log_stage_summary(accelerator, model: ContextOnlyNoGTBoxWanModule, args: argparse.Namespace) -> None:
    """Print which folders are loaded (frozen) and which modules are trained/frozen.

    Reads the actual ``requires_grad`` state after construction + init loads, so
    the report reflects reality rather than intent.
    """
    if not accelerator.is_main_process:
        return

    def _count(params) -> tuple[int, int]:
        n_mods = 0
        n_elems = 0
        for p in params:
            n_mods += 1
            n_elems += int(p.numel())
        return n_mods, n_elems

    dit = model.pipe.dit
    obj_branch_tokens = ("object_embedding", ".object_cross_attn.", ".object_gate", ".norm4.")

    dit_branch_trainable = [
        p for n, p in dit.named_parameters()
        if p.requires_grad and any(tok in n for tok in obj_branch_tokens)
    ]
    dit_lora_trainable = [
        p for n, p in dit.named_parameters()
        if p.requires_grad and "lora_" in n and all(tok not in n for tok in obj_branch_tokens)
    ]
    dit_lora_total = [p for n, p in dit.named_parameters() if "lora_" in n]
    adapter_trainable = [p for p in model.object_adapter.parameters() if p.requires_grad] if model.object_adapter is not None else []
    pooler_trainable = [p for p in model.object_pooler.parameters() if p.requires_grad] if model.object_pooler is not None else []
    aux_trainable = [p for p in model.object_aux_heads.parameters() if p.requires_grad] if model.object_aux_heads is not None else []

    _, dit_branch_elems = _count(dit_branch_trainable)
    _, adapter_elems = _count(adapter_trainable)
    _, lora_elems = _count(dit_lora_trainable)
    total_trainable_elems = sum(int(p.numel()) for p in model.trainable_modules())

    lines = []
    lines.append("=" * 78)
    lines.append("Stage1B context-only NO-GT-BOX (DiffSynth-native v_newtrain) — 模块概览")
    lines.append("=" * 78)
    lines.append("加载文件夹 (frozen loads):")
    lines.append(f"  - 基础 Wan LoRA (raw-phys, 冻结) <- --lora_checkpoint")
    lines.append(f"      {args.lora_checkpoint}")
    lines.append(f"      LoRA params in DiT: {len(dit_lora_total)} (trainable now: {len(dit_lora_trainable)})")
    lines.append(f"  - Stage1A token builder (object_pooler / object_aux_heads, 冻结) <- --stage1a_init_from")
    lines.append(f"      {args.stage1a_init_from}")
    lines.append(f"  - Wan2.2 base (DiT/VAE/T5, 冻结) <- --wan_root")
    lines.append(f"      {args.wan_root}")
    lines.append("训练模块 (trainable):")
    lines.append(f"  - DiT object 注入分支 (object_embedding/object_cross_attn/object_gate/norm4): "
                 f"{len(dit_branch_trainable)} params, {dit_branch_elems:,} elems")
    lines.append(f"  - ObjectConditionAdapter: {len(adapter_trainable)} params, {adapter_elems:,} elems")
    lines.append(f"  - 可训练参数总量: {total_trainable_elems:,} elems")
    lines.append("稳定化 / 诊断配置:")
    lines.append(
        f"  - object_branch_train_trace={bool(args.object_branch_train_trace)} "
        f"| lambda_object_context_reg={float(args.lambda_object_context_reg):.6g} "
        f"| lambda_object_gate_reg={float(args.lambda_object_gate_reg):.6g}"
    )
    lines.append(
        f"  - object_gate_reg_target={float(args.object_gate_reg_target):.4f} "
        f"| object_branch_ratio_guard_max_ratio={float(args.object_branch_ratio_guard_max_ratio):.4f} "
        f"| object_branch_ratio_guard_max_block_id={int(args.object_branch_ratio_guard_max_block_id)}"
    )
    lines.append(
        f"  - object_slot_dropout_prob={float(args.object_slot_dropout_prob):.4f} "
        f"| full_slot_loss_weight={float(args.full_slot_loss_weight):.4f}"
    )
    lines.append(
        f"  - lambda_object_adapter_mlp_reg={float(args.lambda_object_adapter_mlp_reg):.6g} "
        f"| object_adapter_mlp_reg_target={float(args.object_adapter_mlp_reg_target):.4f} "
        f"| object_adapter_mlp_residual_max_ratio="
        f"{float(args.object_adapter_mlp_residual_max_ratio):.4f}"
    )
    lines.append("冻结模块 (frozen):")
    lines.append(f"  - Wan DiT base + LoRA, VAE, Text encoder")
    lines.append(f"  - ObjectTubeProjector (object_pooler): trainable params now = {len(pooler_trainable)}")
    lines.append(f"  - ObjectAuxHeads (object_aux_heads): trainable params now = {len(aux_trainable)}")
    lines.append(f"  - JEPA / CoTracker / VGGT adapters (frozen feature extractors)")
    lines.append("=" * 78)
    accelerator.print("\n".join(lines))


def build_dataset(args: argparse.Namespace):
    if args.dataset_type != "kubric_no_gt_box":
        return tvn.build_dataset(args)
    if not args.kubric_root:
        raise ValueError("--kubric_root is required when dataset_type=kubric_no_gt_box")
    return KubricNoGTBoxDataset(
        root=args.kubric_root,
        split=args.kubric_split,
        resolution=(args.height, args.width),
        num_frames=args.num_frames,
        num_context_frames=args.fixed_num_context_frames,
        sampling_strategy=args.kubric_sampling_strategy,
        seed=42,
        scenarios=args.kubric_scenario,
        init_scan_limit=args.kubric_init_scan_limit,
        cache_root=args.kubric_cache_root,
        split_train_ratio=args.kubric_split_train_ratio,
        split_val_ratio=args.kubric_split_val_ratio,
        max_retry_samples=args.kubric_max_retry_samples,
    )


def build_headonly_val_config(args: argparse.Namespace) -> HeadOnlyValConfig:
    if args.dataset_type == "kubric_no_gt_box":
        return HeadOnlyValConfig(
            enabled=False,
            split="val",
            every_steps=None,
            num_batches=max(1, int(getattr(args, "headonly_val_loss_num_batches", 8))),
        )
    return tvn.build_headonly_val_config(args)


def build_headonly_val_dataset(args: argparse.Namespace, config: HeadOnlyValConfig):
    if args.dataset_type == "kubric_no_gt_box":
        return None
    return tvn.build_headonly_val_dataset(args, config)


def build_headonly_val_dataloader(dataset, args: argparse.Namespace):
    if dataset is None:
        return None
    return tvn.build_headonly_val_dataloader(dataset, args)


def main() -> None:
    parser = build_parser()
    args = tvn.prepare_args(parser.parse_args())
    previous_handlers = tvn.install_interrupt_handlers()

    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    # NOTE: unlike train_v_newtrain.main we do NOT overwrite args.lora_checkpoint
    # on resume. The base Wan LoRA (raw-phys) is FROZEN, so it is never written
    # into the saved checkpoint (only trainable weights are exported). It must
    # therefore always be (re)loaded from --lora_checkpoint by the module
    # constructor. On resume, the stage2 *trainable* weights are restored below
    # via _load_filtered_checkpoint_into_model, and optimizer/step are restored
    # by train_loop from training_state.pt (keyed on args.stage2_resume_from).
    if args.stage2_resume_from is not None and accelerator.is_main_process:
        accelerator.print(
            f"👉 Resuming stage2 training from state {args.stage2_resume_from} "
            f"(base LoRA stays loaded from --lora_checkpoint)."
        )

    dataset = build_dataset(args)
    headonly_val_config = build_headonly_val_config(args)
    headonly_val_dataset = build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = build_headonly_val_dataloader(headonly_val_dataset, args)

    model = build_model(args, accelerator)

    if args.stage1a_init_from is not None:
        init_info = tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded Stage1A token builder init: "
                f"selected_source_keys={init_info['selected_source_keys']}, "
                f"loaded_count={init_info['loaded_count']}, "
                f"shape_mismatch={len(init_info['skipped_shape_mismatch'])}"
            )

    if args.stage2_resume_from is not None:
        resume_info = tvn._load_filtered_checkpoint_into_model(
            model,
            tvn.resolve_lora_checkpoint_for_resume(args.stage2_resume_from),
            include_prefixes=("object_adapter.",),
            include_substrings=(
                "object_embedding",
                ".object_cross_attn.",
                ".object_gate",
                ".norm4.",
            ),
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded stage2 trainable initialization: "
                f"selected_source_keys={resume_info['selected_source_keys']}, "
                f"loaded_count={resume_info['loaded_count']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    _log_stage_summary(accelerator, model, args)

    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict = {}

    try:
        if args.task in ("sft:data_process", "direct_distill:data_process"):
            tvn.launch_data_process_task(accelerator, dataset, model, model_logger, args=args)
        else:
            tvn.train_loop(
                accelerator,
                dataset,
                model,
                model_logger,
                args,
                runtime_state=runtime_state,
                headonly_val_dataloader=headonly_val_dataloader,
                headonly_val_config=headonly_val_config,
            )
    except (KeyboardInterrupt, tvn.TrainingInterrupted) as exc:
        interrupted_checkpoint_path = tvn.training_checkpoint_file(
            tvn.get_checkpoint_dir(args), "interrupted-latest"
        )
        accelerator.print(
            f"Training interrupted at step {model_logger.num_steps}. Saving interrupt checkpoint."
        )
        model_logger.save_model(accelerator, model, interrupted_checkpoint_path)
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get(
            "progress",
            {"global_step": 0, "epoch_id": 0, "batch_in_epoch": 0},
        )
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
                    tvn.get_checkpoint_dir(args), "interrupted-latest"
                ),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc

    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
