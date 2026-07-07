"""Compare-ablation training entry for train0705 Stage1B context-only no-GT-box.

This file intentionally lives under ``train0705/compare_ablation`` so ablation-
specific training switches can evolve without touching the original training
entrypoint used for the main experiments.

Added training-time switches:
  - ``--disable-cotracker``
  - ``--disable-jepa``
  - ``--disable-vggt``

The Stage1A-init ablation does not need a new switch: launch without
``--stage1a_init_from`` and the Stage1A token-builder initialization is skipped.
"""
from __future__ import annotations

import argparse

import torch

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerOutput
from code_vjepa_vggt.adapters.jepa_adapter import JEPAAdapterOutput
from code_vjepa_vggt.train0705 import train_stage1b_context_only_no_gt_box_v_newtrain as base0705
from diffsynth.diffusion import ModelLogger


_DISABLE_VGGT_SENTINEL = "__compare_ablation_disable_vggt__"


class ContextOnlyNoGTBoxWanModuleAblation(base0705.ContextOnlyNoGTBoxWanModule):
    def __init__(
        self,
        *args,
        disable_cotracker: bool = False,
        disable_jepa: bool = False,
        disable_vggt: bool = False,
        **kwargs,
    ) -> None:
        self.disable_cotracker = bool(disable_cotracker)
        self.disable_jepa = bool(disable_jepa)
        self.disable_vggt = bool(disable_vggt)
        self._jepa_crop_size = int(kwargs.get("jepa_input_size", 384))
        self._jepa_patch_size = int(kwargs.get("jepa_patch_size", 16))
        self._jepa_tubelet_size = int(kwargs.get("jepa_tubelet_size", 2))

        if self.disable_cotracker:
            kwargs["cotracker_checkpoint"] = None
        if self.disable_vggt:
            kwargs["train_vggt"] = False
            if not kwargs.get("vggt_cache_root"):
                kwargs["vggt_cache_root"] = _DISABLE_VGGT_SENTINEL

        super().__init__(*args, **kwargs)

        self._jepa_embed_dim = (
            int(self.object_pooler.jepa_proj.in_features) if self.object_pooler is not None else 0
        )

        if self.disable_cotracker:
            self.cotracker_adapter = None
            self.cotracker_runner = None
        if self.disable_jepa:
            self.jepa_adapter = None
            self.jepa_runner = None
        if self.disable_vggt:
            self.vggt_adapter = None
            self.vggt_cache_root = None

    def _build_static_cotracker_output(
        self,
        frames_bthwc_01: torch.Tensor,
        *,
        query_points_prior: torch.Tensor,
    ) -> CoTrackerOutput:
        batch_size, frames, height, width, _ = frames_bthwc_01.shape
        tracks = query_points_prior.unsqueeze(1).expand(batch_size, frames, -1, -1).contiguous()
        visibility = torch.ones(
            batch_size,
            frames,
            query_points_prior.shape[1],
            device=query_points_prior.device,
            dtype=query_points_prior.dtype,
        )
        confidence = torch.ones_like(visibility)
        return CoTrackerOutput(
            query_points=query_points_prior,
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            image_hw=(height, width),
            input_hw=(height, width),
            used_model=False,
        )

    def _build_zero_jepa_output(self, context_video: torch.Tensor) -> JEPAAdapterOutput:
        batch_size, _, frames, _, _ = context_video.shape
        token_t = max(1, int(frames) // max(int(self._jepa_tubelet_size), 1))
        token_h = max(1, int(self._jepa_crop_size) // max(int(self._jepa_patch_size), 1))
        token_w = max(1, int(self._jepa_crop_size) // max(int(self._jepa_patch_size), 1))
        patch_tokens = torch.zeros(
            batch_size,
            token_t,
            token_h,
            token_w,
            int(self._jepa_embed_dim),
            device=context_video.device,
            dtype=context_video.dtype,
        )
        return JEPAAdapterOutput(
            patch_tokens=patch_tokens,
            input_hw=(int(self._jepa_crop_size), int(self._jepa_crop_size)),
            token_grid_hw=(token_h, token_w),
            token_grid_t=token_t,
        )

    def _run_cotracker(self, frames_bthwc_01, *, query_points_prior, query_frame_ids, query_image_hw):
        if self.disable_cotracker:
            return self._build_static_cotracker_output(
                frames_bthwc_01,
                query_points_prior=query_points_prior,
            )
        return super()._run_cotracker(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=query_image_hw,
        )

    def _run_jepa(self, context_video):
        if self.disable_jepa:
            return self._build_zero_jepa_output(context_video)
        return super()._run_jepa(context_video)

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if not self.enable_object_branch:
            return super()._compute_object_losses(pipe, inputs_shared, inputs_posi)

        sample = inputs_shared["raw_sample"]
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
        if not self.disable_vggt:
            if self.vggt_cache_root:
                vggt_out = base0705.load_vggt_cache(sample, self.vggt_cache_root, allow_missing=False)
                if vggt_out is None:
                    raise RuntimeError(
                        "VGGT cache root is set but no cache found for sample "
                        f"{sample.get('video_path', '<unknown>')}"
                    )
            else:
                if self.vggt_adapter is None:
                    raise RuntimeError("VGGT adapter is not initialized while disable_vggt is false.")
                vggt_out = self.vggt_adapter(
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
        jepa_out = self._run_jepa(context_video)
        context_latents = inputs_shared["clean_prefix_latents"]
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
            loss_main = base0705.flow_match_context_sft_loss(
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
        }
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = base0705.build_parser()
    group = parser.add_argument_group("compare_ablation")
    group.add_argument("--disable-cotracker", action="store_true", default=False)
    group.add_argument("--disable-jepa", action="store_true", default=False)
    group.add_argument("--disable-vggt", action="store_true", default=False)
    return parser


def build_model(args: argparse.Namespace, accelerator) -> ContextOnlyNoGTBoxWanModuleAblation:
    grounding_config = base0705._grounding_config_from_args(args)
    grounding_config["grounding_device"] = args.grounding_device or str(accelerator.device)
    return ContextOnlyNoGTBoxWanModuleAblation(
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
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
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
        disable_cotracker=args.disable_cotracker,
        disable_jepa=args.disable_jepa,
        disable_vggt=args.disable_vggt,
    )


def _log_ablation_summary(accelerator, args: argparse.Namespace) -> None:
    if not accelerator.is_main_process:
        return
    lines = [
        "=" * 78,
        "compare_ablation switches",
        "=" * 78,
        f"  - disable_cotracker: {bool(args.disable_cotracker)}",
        f"  - disable_jepa: {bool(args.disable_jepa)}",
        f"  - disable_vggt: {bool(args.disable_vggt)}",
        f"  - stage1a_init_from: {args.stage1a_init_from}",
        "=" * 78,
    ]
    accelerator.print("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = tvn.prepare_args(parser.parse_args())
    previous_handlers = tvn.install_interrupt_handlers()

    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    if args.stage2_resume_from is not None and accelerator.is_main_process:
        accelerator.print(
            f"👉 Resuming stage2 training from state {args.stage2_resume_from} "
            f"(base LoRA stays loaded from --lora_checkpoint)."
        )

    dataset = tvn.build_dataset(args)
    headonly_val_config = tvn.build_headonly_val_config(args)
    headonly_val_dataset = tvn.build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = tvn.build_headonly_val_dataloader(headonly_val_dataset, args)

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

    _log_ablation_summary(accelerator, args)
    base0705._log_stage_summary(accelerator, model, args)

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
        raise exc

    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
