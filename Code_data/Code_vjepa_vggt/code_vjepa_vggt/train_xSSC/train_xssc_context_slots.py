"""Train Wan2.2 with frozen xSSC context slots as object cross-attention tokens.

This file was copied from train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py
and intentionally lives in train_xSSC. It replaces the complete Stage1A/Stage1B
token-building frontend (grounding, CoTracker, VGGT, JEPA, ObjectTubeProjector,
ObjectAuxHeads, and ObjectConditionAdapter) with:

    context video -> frozen RandSFQ2 -> slotz [B, Tc, 7, 256]
                  -> LayerNorm + Linear(256, Wan dim) + time embedding
                  -> [B, Tc * 7, Wan dim] -> object cross-attention

Only the slot projection, time embedding, and Wan object cross-attention branch
are trainable. The Wan base, physical-state LoRA, xSSC, and DINO backbone stay
frozen. Only context frames are passed to xSSC, so no future information leaks
into the conditioning path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing train_v_newtrain installs the DiffSynth path shim selected by the
# --diffsynth_root command-line argument.
import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.context_wan_v_newtrain import (
    enable_object_condition_branch,
    flow_match_context_sft_loss,
)

from diffsynth.diffusion import ModelLogger


XSSC_IMAGENET_MEAN = (123.675, 116.28, 103.53)
XSSC_IMAGENET_STD = (58.395, 57.12, 57.375)


def _load_xssc_model(
    *,
    xssc_root: str,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Build RandSFQ2 from its official config and strictly load its checkpoint."""
    root = Path(xssc_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"xSSC root does not exist: {root}")
    if not config.is_file():
        raise FileNotFoundError(f"xSSC config does not exist: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"xSSC checkpoint does not exist: {checkpoint}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config)
    model = build_from_config(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported xSSC checkpoint object: {type(state)!r}")
    if state and all(str(key).startswith("m.") for key in state):
        state = {str(key)[2:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)

    slot_dim = int(cfg.emb_dim)
    num_slots = int(cfg.max_num)
    # The xSSC decoder is only a pretraining objective. Slot conditioning needs
    # encode_backbone/encode_project/initializ/aggregat/transit, not reconstruction.
    model.decode = None
    model.requires_grad_(False)
    model.eval()
    model.to(device=device)
    return model, slot_dim, num_slots


class XSSCContextSlotsWanModule(tvn.WanTrainingModule):
    """Wan training module conditioned directly on frozen context-only xSSC slots."""

    def __init__(
        self,
        *args,
        xssc_root: str,
        xssc_config: str,
        xssc_checkpoint: str,
        xssc_input_size: int = 256,
        xssc_max_time_steps: int = 64,
        object_gate_init: float = 0.1,
        lambda_main: float = 1.0,
        lambda_object_context_reg: float = 0.0,
        **kwargs,
    ) -> None:
        # The parent must not construct the legacy object frontend. We inject only
        # the Wan cross-attention branch after the base/LoRA pipeline is ready.
        kwargs["enable_object_branch"] = False
        kwargs["freeze_non_object_trainables"] = True
        kwargs["train_object_pooler"] = False
        kwargs["train_object_aux_heads"] = False
        kwargs["train_object_adapter"] = False
        kwargs["train_object_dit_branch"] = False
        super().__init__(
            *args,
            object_gate_init=object_gate_init,
            lambda_main=lambda_main,
            lambda_object_context_reg=lambda_object_context_reg,
            **kwargs,
        )

        self.enable_object_branch = True
        self.lambda_main = float(lambda_main)
        self.lambda_object_context_reg = float(lambda_object_context_reg)
        self.xssc_input_size = int(xssc_input_size)
        self.xssc_max_time_steps = int(xssc_max_time_steps)

        dit = enable_object_condition_branch(
            self.pipe.dit,
            object_gate_init=float(object_gate_init),
            reinitialize_object_branch=True,
        )
        # xSSC tokens are projected directly to dit.dim, so the old text-dimension
        # object_embedding would be both redundant and shape-incompatible.
        dit.object_embedding = None
        for name, param in dit.named_parameters():
            is_object_branch = any(
                token in name
                for token in (".object_cross_attn.", ".object_gate", ".norm4.")
            )
            param.requires_grad = is_object_branch

        model_device = dit.patch_embedding.weight.device
        model_dtype = dit.patch_embedding.weight.dtype
        self.xssc, self.xssc_slot_dim, self.xssc_num_slots = _load_xssc_model(
            xssc_root=xssc_root,
            config_path=xssc_config,
            checkpoint_path=xssc_checkpoint,
            device=model_device,
        )
        if self.xssc_slot_dim != 256:
            raise ValueError(f"Expected 256-d xSSC slots, got {self.xssc_slot_dim}")

        hidden_dim = int(dit.dim)
        self.slot_norm = nn.LayerNorm(self.xssc_slot_dim)
        self.slot_projector = nn.Linear(self.xssc_slot_dim, hidden_dim)
        self.time_embedding = nn.Embedding(self.xssc_max_time_steps, hidden_dim)
        nn.init.normal_(self.slot_projector.weight, std=0.02)
        nn.init.zeros_(self.slot_projector.bias)
        nn.init.normal_(self.time_embedding.weight, std=0.02)
        self.slot_norm.to(device=model_device, dtype=model_dtype)
        self.slot_projector.to(device=model_device, dtype=model_dtype)
        self.time_embedding.to(device=model_device, dtype=model_dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        # nn.Module.train() is recursive; force the frozen xSSC transition dropout
        # off after every mode switch.
        self.xssc.eval()
        return self

    def trainable_modules(self) -> list[nn.Parameter]:
        params = list(self.slot_norm.parameters())
        params.extend(self.slot_projector.parameters())
        params.extend(self.time_embedding.parameters())
        params.extend(
            param
            for name, param in self.pipe.dit.named_parameters()
            if any(token in name for token in (".object_cross_attn.", ".object_gate", ".norm4."))
        )
        unique: list[nn.Parameter] = []
        seen: set[int] = set()
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        return unique

    def _preprocess_xssc(self, context_video: torch.Tensor) -> torch.Tensor:
        """Convert [B,C,T,H,W] in [-1,1] to xSSC [B,T,C,256,256]."""
        frames = context_video.permute(0, 2, 1, 3, 4).float()
        batch, time_steps, channels, height, width = frames.shape
        crop_size = min(int(height), int(width))
        top = (int(height) - crop_size) // 2
        left = (int(width) - crop_size) // 2
        frames = frames[..., top : top + crop_size, left : left + crop_size]
        frames = frames.reshape(batch * time_steps, channels, crop_size, crop_size)
        frames = F.interpolate(
            frames,
            size=(self.xssc_input_size, self.xssc_input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        frames = (frames + 1.0).mul(127.5).clamp(0.0, 255.0)
        mean = frames.new_tensor(XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = frames.new_tensor(XSSC_IMAGENET_STD).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames.view(batch, time_steps, channels, self.xssc_input_size, self.xssc_input_size)

    @torch.no_grad()
    def _extract_xssc_slots(self, video: torch.Tensor) -> torch.Tensor:
        """Run the frozen RandSFQ2 encoder/slot recurrence without its decoder."""
        self.xssc.eval()
        batch, time_steps, channels, height, width = video.shape
        flat_video = video.flatten(0, 1)
        autocast_enabled = flat_video.device.type == "cuda"
        with torch.autocast(device_type=flat_video.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            feature = self.xssc.encode_backbone(flat_video).detach()
            encoded = feature.permute(0, 2, 3, 1).flatten(1, 2)
            encoded = self.xssc.encode_posit_embed(encoded)
            encoded = self.xssc.encode_project(encoded)
            encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])

            slots = None
            for frame_id in range(time_steps):
                if frame_id == 0:
                    query = self.xssc.initializ(batch)
                else:
                    query = self.xssc.transit(slots, encoded[:, : frame_id + 1])
                num_iter = None if frame_id == 0 else 1
                current_slots, _ = self.xssc.aggregat(
                    encoded[:, frame_id], query, num_iter=num_iter
                )
                current_slots = current_slots[:, None]
                slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
        if slots is None:
            raise RuntimeError("xSSC received zero context frames")
        return slots

    def _build_object_context(self, context_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xssc_video = self._preprocess_xssc(context_video)
        slots = self._extract_xssc_slots(xssc_video)
        time_steps = int(slots.shape[1])
        if time_steps > self.xssc_max_time_steps:
            raise ValueError(
                f"Context length {time_steps} exceeds xssc_max_time_steps={self.xssc_max_time_steps}"
            )
        target_dtype = self.slot_norm.weight.dtype
        slots_for_projection = slots.to(device=self.slot_norm.weight.device, dtype=target_dtype)
        tokens = self.slot_projector(self.slot_norm(slots_for_projection))
        time_ids = torch.arange(time_steps, device=tokens.device)
        time_tokens = self.time_embedding(time_ids).view(1, time_steps, 1, -1)
        tokens = tokens + time_tokens.to(dtype=tokens.dtype)
        batch, _, num_slots, hidden_dim = tokens.shape
        return tokens.reshape(batch, time_steps * num_slots, hidden_dim), slots

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        sample = inputs_shared["raw_sample"]
        num_context_frames = max(1, int(sample["num_context_frames"]))
        context_video = sample["context_video"][:, :num_context_frames].unsqueeze(0)
        context_video = context_video.to(device=pipe.device, dtype=pipe.torch_dtype)
        object_context, slots = self._build_object_context(context_video)

        loss_main = flow_match_context_sft_loss(
            pipe,
            **inputs_shared,
            **inputs_posi,
            object_context=object_context,
        )
        object_context_reg = object_context.square().mean()
        total = self.lambda_main * loss_main + self.lambda_object_context_reg * object_context_reg
        context_abs = object_context.detach().abs()
        slot_abs = slots.detach().abs()
        metrics = {
            "train/loss_total": float(total.detach().item()),
            "train/loss_main": float(loss_main.detach().item()),
            "train/loss_object_context_reg": float(object_context_reg.detach().item()),
            "train/xssc_context_frames": float(num_context_frames),
            "train/xssc_slots_per_frame": float(self.xssc_num_slots),
            "train/xssc_token_count": float(object_context.shape[1]),
            "train/xssc_slot_abs_mean": float(slot_abs.mean().item()),
            "train/object_context_abs_max": float(context_abs.max().item()),
            "train/object_context_abs_mean": float(context_abs.mean().item()),
        }
        return total, metrics


def _patch_general_config_conflict_handler() -> None:
    if getattr(tvn, "_xssc_conflict_patched", False):
        return
    original = tvn.add_general_config

    def patched(parser: argparse.ArgumentParser):
        parser = original(parser)
        parser.conflict_handler = "resolve"
        parser._optionals.conflict_handler = "resolve"
        return parser

    tvn.add_general_config = patched
    tvn._xssc_conflict_patched = True


def build_parser() -> argparse.ArgumentParser:
    _patch_general_config_conflict_handler()
    parser = tvn.wan_parser()
    parser.description = "Train Wan2.2 with frozen context-only xSSC slots."
    group = parser.add_argument_group("xssc_context_slots")
    group.add_argument("--xssc_root", required=True)
    group.add_argument("--xssc_config", required=True)
    group.add_argument("--xssc_checkpoint", required=True)
    group.add_argument("--xssc_input_size", type=int, default=256)
    group.add_argument("--xssc_max_time_steps", type=int, default=64)
    return parser


def build_model(args: argparse.Namespace, accelerator) -> XSSCContextSlotsWanModule:
    return XSSCContextSlotsWanModule(
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
        object_gate_init=args.object_gate_init,
        lambda_main=args.lambda_main,
        lambda_object_context_reg=args.lambda_object_context_reg,
        xssc_root=args.xssc_root,
        xssc_config=args.xssc_config,
        xssc_checkpoint=args.xssc_checkpoint,
        xssc_input_size=args.xssc_input_size,
        xssc_max_time_steps=args.xssc_max_time_steps,
    )


def _log_stage_summary(accelerator, model: XSSCContextSlotsWanModule, args: argparse.Namespace) -> None:
    if not accelerator.is_main_process:
        return
    dit_params = [
        param
        for name, param in model.pipe.dit.named_parameters()
        if param.requires_grad
        and any(token in name for token in (".object_cross_attn.", ".object_gate", ".norm4."))
    ]
    projector_params = list(model.slot_norm.parameters()) + list(model.slot_projector.parameters())
    projector_params += list(model.time_embedding.parameters())
    total = sum(param.numel() for param in model.trainable_modules())
    lines = [
        "=" * 78,
        "xSSC context-slot object conditioning",
        "=" * 78,
        f"Frozen Wan base: {args.wan_root}",
        f"Frozen physical-state LoRA: {args.lora_checkpoint}",
        f"Frozen xSSC checkpoint: {args.xssc_checkpoint}",
        f"xSSC shape: [B, Tc, {model.xssc_num_slots}, {model.xssc_slot_dim}]",
        f"Object token shape: [B, Tc*{model.xssc_num_slots}, {model.pipe.dit.dim}]",
        f"Trainable projector/time params: {sum(p.numel() for p in projector_params):,}",
        f"Trainable DiT object-branch params: {sum(p.numel() for p in dit_params):,}",
        f"Total trainable params: {total:,}",
        "Legacy Stage1A/Grounding/CoTracker/VGGT/JEPA modules: not constructed",
        "=" * 78,
    ]
    accelerator.print("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = tvn.prepare_args(parser.parse_args())
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    dataset = tvn.build_dataset(args)
    headonly_val_config = tvn.build_headonly_val_config(args)
    headonly_val_dataset = tvn.build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = tvn.build_headonly_val_dataloader(headonly_val_dataset, args)
    model = build_model(args, accelerator)

    if args.stage2_resume_from is not None:
        resume_info = tvn._load_filtered_checkpoint_into_model(
            model,
            tvn.resolve_lora_checkpoint_for_resume(args.stage2_resume_from),
            include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
            include_substrings=(".object_cross_attn.", ".object_gate", ".norm4."),
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded xSSC-object resume weights: "
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
            f"Training interrupted at step {model_logger.num_steps}; saving checkpoint."
        )
        model_logger.save_model(accelerator, model, interrupted_checkpoint_path)
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get(
            "progress", {"global_step": 0, "epoch_id": 0, "batch_in_epoch": 0}
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
