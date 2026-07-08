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
from code_vjepa_vggt.utils.vggt_cache import load_vggt_cache

from diffsynth.diffusion import ModelLogger


_DUMMY_BOX_XYXY = (0.45, 0.45, 0.55, 0.55)


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


class ContextOnlyNoGTBoxWanModule(tvn.WanTrainingModule):
    """WanTrainingModule variant that sources object priors from viewer grounding."""

    def __init__(self, *args, grounding_config: dict | None = None, **kwargs) -> None:
        self._jepa_tubelet_size = int(kwargs.get("jepa_tubelet_size", 2))
        super().__init__(*args, **kwargs)
        self.viewer_grounding: ViewerGroundingBoxProvider | None = None
        if self.enable_object_branch:
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
            if self.lambda_main > 0.0:
                loss_main = flow_match_context_sft_loss(
                    pipe,
                    **inputs_shared,
                    **inputs_posi,
                    object_context=object_context,
                )
            else:
                loss_main = object_context.new_zeros(())
            object_context_reg = object_context.new_zeros(())
            total = (
                self.lambda_main * loss_main
                + self.lambda_object_context_reg * object_context_reg
            )
            metrics = {
                "train/loss_total": float(total.detach().item()),
                "train/loss_main": float(loss_main.detach().item()),
                "train/loss_object_context_reg": float(object_context_reg.detach().item()),
                "train/object_count": 0.0,
                "train/object_latent_tokens_abs_max": 0.0,
                "train/object_context_abs_max": 0.0,
                "train/object_context_abs_mean": 0.0,
                "train/ctx_max_length": ctx_max_length,
                "train/sampled_ctx_last_index": sampled_ctx_last_index,
                "train/sampled_ctx_num_frames": 0.0,
            }
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
        )

        if self.lambda_main > 0.0:
            loss_main = flow_match_context_sft_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
                object_context=object_context,
            )
        else:
            loss_main = object_context.new_zeros(())
        object_context_reg = object_context.square().mean()

        total = (
            self.lambda_main * loss_main
            + self.lambda_object_context_reg * object_context_reg
        )

        object_context_abs = object_context.detach().abs()
        object_latent_tokens_abs = object_out.object_latent_tokens.detach().abs()
        metrics = {
            "train/loss_total": float(total.detach().item()),
            "train/loss_main": float(loss_main.detach().item()),
            "train/loss_object_context_reg": float(object_context_reg.detach().item()),
            "train/object_count": float(object_valid_mask.sum().item()),
            "train/object_latent_tokens_abs_max": float(object_latent_tokens_abs.max().item()),
            "train/object_context_abs_max": float(object_context_abs.max().item()),
            "train/object_context_abs_mean": float(object_context_abs.mean().item()),
            "train/jepa_input_frames": float(jepa_ctx_fix["jepa_context_frames"]),
            "train/jepa_padding_frames": float(jepa_ctx_fix["padded_context_frames"]),
            "train/ctx_max_length": ctx_max_length,
            "train/sampled_ctx_last_index": sampled_ctx_last_index,
            "train/sampled_ctx_num_frames": float(num_context_frames),
        }
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
        train_object_pooler=args.train_object_pooler,
        train_object_aux_heads=args.train_object_aux_heads,
        train_object_adapter=args.train_object_adapter,
        train_object_dit_branch=args.train_object_dit_branch,
        freeze_non_object_trainables=args.freeze_non_object_trainables,
        depth_target_state_index=args.depth_target_state_index,
        depth_target_source=args.depth_target_source,
        depth_anything_cache_root=args.depth_anything_cache_root,
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
