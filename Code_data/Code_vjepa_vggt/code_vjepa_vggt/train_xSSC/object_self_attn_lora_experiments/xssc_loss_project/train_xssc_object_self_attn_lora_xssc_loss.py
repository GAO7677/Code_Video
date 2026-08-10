"""Train Wan Full-SA LoRA with a frozen xSSC future-slot feature loss.

This keeps the existing Wan2.2 + merged OpenVid LoRA, no-object, Full-SA
training path intact.  The only additional objective is computed as follows:

    DiT v prediction -> reconstructed latent x0 -> frozen Tiny VAE ->
    frozen xSSC slots -> future-frame cosine distance to GT xSSC slots.

The xSSC parameters and Tiny VAE parameters stay frozen, but the prediction
branch deliberately retains gradients with respect to the reconstructed x0.
The auxiliary term follows the same normalized scheduler-timestep weighting
used by the existing V-JEPA loss experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
for _path in (EXPERIMENT_ROOT, TRAIN_XSSC_ROOT, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import code_vjepa_vggt.context_wan_v_newtrain as context_wan
import train_xssc_context_slots as official_xssc
import train_xssc_object_self_attn_lora as core
from vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss import (
    _load_tiny_vae,
)


VALID_XSSC_LOSS_BACKENDS = ("dinov3_movic", "official_dinov2")


class XSSCFeatureLossWanModule(core.DINOv3XSSCContextSlotsWanModule):
    """Existing no-object Full-SA module plus differentiable frozen-xSSC loss."""

    def __init__(
        self,
        *args,
        xssc_loss_backend: str,
        xssc_loss_weight: float,
        xssc_loss_future_start_frame: int,
        xssc_loss_backbone_chunk_size: int,
        xssc_loss_gradient_diagnostics_every_n_forwards: int,
        tiny_vae_root: str,
        tiny_vae_checkpoint: str,
        tiny_vae_parallel: bool,
        **kwargs,
    ) -> None:
        xssc_root = str(kwargs["xssc_root"])
        xssc_config = str(kwargs["xssc_config"])
        xssc_checkpoint = str(kwargs["xssc_checkpoint"])
        dinov3_root = str(kwargs["dinov3_root"])
        dinov3_checkpoint = str(kwargs["dinov3_checkpoint"])
        sam2_config = str(kwargs["xssc_sam2_config"])
        sam2_checkpoint = str(kwargs["xssc_sam2_checkpoint"])
        box_cache_dir = kwargs.get("xssc_box_cache_dir")
        amg_filter_args = kwargs.get("xssc_amg_filter_args")

        super().__init__(*args, **kwargs)
        if self.enable_object_branch:
            raise ValueError("xSSC feature-loss experiments require no-object mode")
        if self.self_attn_adaptation_mode != "full_sa":
            raise ValueError("xSSC feature-loss experiments require Full-SA adaptation")

        self.xssc_loss_backend = str(xssc_loss_backend)
        self.xssc_loss_weight = float(xssc_loss_weight)
        self.xssc_loss_future_start_frame = int(xssc_loss_future_start_frame)
        self.xssc_loss_backbone_chunk_size = int(xssc_loss_backbone_chunk_size)
        self.xssc_loss_gradient_diagnostics_every_n_forwards = int(
            xssc_loss_gradient_diagnostics_every_n_forwards
        )
        self.tiny_vae_parallel = bool(tiny_vae_parallel)
        self._xssc_loss_forward_count = 0
        if self.xssc_loss_backend not in VALID_XSSC_LOSS_BACKENDS:
            raise ValueError(
                f"xssc_loss_backend must be one of {VALID_XSSC_LOSS_BACKENDS}, "
                f"got {self.xssc_loss_backend!r}"
            )
        if self.xssc_loss_weight <= 0.0:
            raise ValueError("xssc_loss_weight must be positive")
        if self.xssc_loss_future_start_frame < 0:
            raise ValueError("xssc_loss_future_start_frame must be non-negative")
        if self.xssc_loss_backbone_chunk_size <= 0:
            raise ValueError("xssc_loss_backbone_chunk_size must be positive")
        if self.xssc_loss_gradient_diagnostics_every_n_forwards <= 0:
            raise ValueError(
                "xssc_loss_gradient_diagnostics_every_n_forwards must be positive"
            )

        model_device = self.pipe.dit.patch_embedding.weight.device
        if self.xssc_loss_backend == "dinov3_movic":
            encoder, slot_dim, num_slots = core._load_dinov3_xssc_model(
                xssc_root=xssc_root,
                config_path=xssc_config,
                checkpoint_path=xssc_checkpoint,
                dinov3_root=dinov3_root,
                dinov3_checkpoint=dinov3_checkpoint,
                device=model_device,
            )
            if amg_filter_args is None:
                raise ValueError("DINOv3 MOVi-C xSSC loss requires AMG filter settings")
            box_builder = core.AMGBoxBuilder(
                sam2_config=sam2_config,
                sam2_checkpoint=sam2_checkpoint,
                cache_dir=box_cache_dir,
                filter_args=amg_filter_args,
            )
        else:
            encoder, slot_dim, num_slots = official_xssc._load_xssc_model(
                xssc_root=xssc_root,
                config_path=xssc_config,
                checkpoint_path=xssc_checkpoint,
                device=model_device,
            )
            box_builder = None

        encoder.requires_grad_(False)
        encoder.eval()
        # Frozen auxiliaries remain unregistered: DDP must not broadcast them and
        # LoRA-only checkpoints must not traverse or serialize them.
        object.__setattr__(self, "_xssc_loss_encoder", encoder)
        object.__setattr__(self, "_xssc_loss_box_builder", box_builder)
        self.xssc_loss_slot_dim = int(slot_dim)
        self.xssc_loss_num_slots = int(num_slots)

        tiny_vae, tiny_vae_apply = _load_tiny_vae(
            tiny_vae_root,
            tiny_vae_checkpoint,
            model_device,
            self.pipe.torch_dtype,
        )
        object.__setattr__(self, "_tiny_vae", tiny_vae)
        object.__setattr__(self, "_tiny_vae_apply", tiny_vae_apply)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_tiny_vae"):
            self._tiny_vae.eval()
        if hasattr(self, "_xssc_loss_encoder"):
            self._xssc_loss_encoder.eval()
        return self

    @staticmethod
    def _restore_condition_latents(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        captured_inputs: dict,
    ) -> torch.Tensor:
        context_indices = context_wan.resolve_context_latent_indices_from_frames(
            raw_frame_indices=captured_inputs.get("context_frame_indices"),
            raw_num_frames=captured_inputs.get("num_frames"),
            latent_length=int(target_x0.shape[2]),
        )
        if context_indices:
            return context_wan.apply_clean_latents_at_indices(
                pred_x0,
                target_x0,
                context_indices,
            )
        prefix_length = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
            num_clean_prefix_latents=captured_inputs.get(
                "num_clean_prefix_latents"
            ),
        )
        if prefix_length > 0:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, :prefix_length] = target_x0[:, :, :prefix_length]
        elif "first_frame_latents" in captured_inputs:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, 0:1] = target_x0[:, :, 0:1]
        return pred_x0

    def _decode_tiny_vae_raw(self, latents: torch.Tensor) -> torch.Tensor:
        """Differentiably decode Wan latents to Tiny-VAE [B,T,3,H,W] video."""
        latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
        device_type = latent_ntchw.device.type
        with torch.autocast(
            device_type=device_type,
            dtype=self.pipe.torch_dtype,
            enabled=device_type == "cuda",
        ):
            video = self._tiny_vae_apply(
                self._tiny_vae.decoder,
                latent_ntchw,
                self.tiny_vae_parallel,
                False,
            )
            if self._tiny_vae.patch_size > 1:
                video = F.pixel_shuffle(video, self._tiny_vae.patch_size)
        skip_trim = (
            self._tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0
        )
        if not skip_trim:
            video = video[:, self._tiny_vae.frames_to_trim :]
        return video

    def _preprocess_xssc_loss(self, video: torch.Tensor) -> torch.Tensor:
        """Convert Tiny-VAE [B,T,C,H,W] in [0,1] to xSSC input."""
        if video.ndim != 5 or int(video.shape[2]) != 3:
            raise ValueError(
                f"Tiny VAE video must be [B,T,3,H,W], got {tuple(video.shape)}"
            )
        frames = video.float()
        batch, time_steps, channels, height, width = frames.shape
        crop_size = min(int(height), int(width))
        top = (int(height) - crop_size) // 2
        left = (int(width) - crop_size) // 2
        frames = frames[..., top : top + crop_size, left : left + crop_size]
        frames = F.interpolate(
            frames.reshape(batch * time_steps, channels, crop_size, crop_size),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        frames = frames * 255.0
        mean = frames.new_tensor(official_xssc.XSSC_IMAGENET_MEAN).view(
            1, 3, 1, 1
        )
        std = frames.new_tensor(official_xssc.XSSC_IMAGENET_STD).view(
            1, 3, 1, 1
        )
        frames = (frames - mean) / std
        return frames.view(batch, time_steps, channels, 256, 256)

    def _encode_xssc_backbone(self, flat_video: torch.Tensor) -> torch.Tensor:
        chunks = []
        chunk_size = self.xssc_loss_backbone_chunk_size
        for start in range(0, int(flat_video.shape[0]), chunk_size):
            current = flat_video[start : start + chunk_size]
            if torch.is_grad_enabled() and current.requires_grad:
                output = checkpoint(
                    self._xssc_loss_encoder.encode_backbone,
                    current,
                    use_reentrant=False,
                )
            else:
                output = self._xssc_loss_encoder.encode_backbone(current)
            chunks.append(output)
        return torch.cat(chunks, dim=0)

    def _extract_xssc_loss_slots(
        self,
        video: torch.Tensor,
        initial_query: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Frozen xSSC forward that preserves gradients to ``video``."""
        encoder = self._xssc_loss_encoder
        encoder.eval()
        batch, time_steps = int(video.shape[0]), int(video.shape[1])
        flat_video = video.flatten(0, 1)
        autocast_enabled = flat_video.device.type == "cuda"
        with torch.autocast(
            device_type=flat_video.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            feature = self._encode_xssc_backbone(flat_video)
            feature_height, feature_width = int(feature.shape[-2]), int(feature.shape[-1])
            encoded = feature.permute(0, 2, 3, 1)
            encoded = encoder.encode_posit_embed(encoded).flatten(1, 2)
            encoded = encoder.encode_project(encoded)
            encoded = encoded.view(
                batch,
                time_steps,
                encoded.shape[1],
                encoded.shape[2],
            )
            query0 = initial_query.to(device=encoded.device, dtype=encoded.dtype)
            slot_frames = []
            attention_frames = []
            slots_so_far = None
            for frame_id in range(time_steps):
                query = (
                    query0
                    if frame_id == 0
                    else encoder.transit(
                        slots_so_far,
                        encoded[:, : frame_id + 1],
                    )
                )
                current_slots, current_attention = encoder.aggregat(
                    encoded[:, frame_id],
                    query,
                    num_iter=None if frame_id == 0 else 1,
                )
                slot_frames.append(current_slots)
                slots_so_far = torch.stack(slot_frames, dim=1)
                if return_attention:
                    attention_frames.append(
                        current_attention.view(
                            batch,
                            self.xssc_loss_num_slots,
                            feature_height,
                            feature_width,
                        )
                    )
        attentions = (
            torch.stack(attention_frames, dim=1) if return_attention else None
        )
        return slots_so_far, attentions

    def _make_shared_initial_query(
        self,
        target_xssc_video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time_steps = int(target_xssc_video.shape[0]), int(
            target_xssc_video.shape[1]
        )
        if self.xssc_loss_backend == "dinov3_movic":
            box_builder = self._xssc_loss_box_builder
            generator = box_builder._generator
            if generator is not None:
                generator.predictor.model.to(device=target_xssc_video.device)
            boxes = box_builder(
                target_xssc_video,
                self.xssc_loss_num_slots,
            )
            generator = box_builder._generator
            if generator is not None:
                generator.predictor.reset_predictor()
                generator.predictor.model.to(device="cpu")
                torch.cuda.empty_cache()
            query = self._xssc_loss_encoder.initializ(boxes[:, 0])
            valid_slots = boxes[:, 0].abs().sum(dim=-1) > 0
        else:
            query = self._xssc_loss_encoder.initializ(batch)
            valid_slots = torch.ones(
                batch,
                self.xssc_loss_num_slots,
                device=target_xssc_video.device,
                dtype=torch.bool,
            )
        if int(time_steps) <= self.xssc_loss_future_start_frame:
            raise ValueError(
                "Decoded video has no future xSSC frames: "
                f"T={time_steps}, start={self.xssc_loss_future_start_frame}"
            )
        return query.detach(), valid_slots

    def _xssc_feature_loss(
        self,
        pred_x0_latents: torch.Tensor,
        target_x0_latents: torch.Tensor,
        *,
        return_visuals: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor] | None]:
        with torch.no_grad():
            target_raw = self._decode_tiny_vae_raw(target_x0_latents)
            target_video = target_raw.clamp(0.0, 1.0)
            target_input = self._preprocess_xssc_loss(target_video)
            initial_query, valid_slots = self._make_shared_initial_query(target_input)
            target_slots, target_attention = self._extract_xssc_loss_slots(
                target_input,
                initial_query,
                return_attention=return_visuals,
            )
            target_slots = target_slots.detach()

        pred_raw = self._decode_tiny_vae_raw(pred_x0_latents)
        pred_clipped = pred_raw.clamp(0.0, 1.0)
        # Straight-through clipping keeps the frozen encoder in its trained
        # range without silently zeroing all out-of-range prediction gradients.
        pred_video = pred_raw + (pred_clipped - pred_raw).detach()
        pred_input = self._preprocess_xssc_loss(pred_video)
        pred_slots, pred_attention = self._extract_xssc_loss_slots(
            pred_input,
            initial_query,
            return_attention=return_visuals,
        )
        if pred_slots.shape != target_slots.shape:
            raise RuntimeError(
                "xSSC slot shape mismatch: "
                f"pred={tuple(pred_slots.shape)}, target={tuple(target_slots.shape)}"
            )

        start = self.xssc_loss_future_start_frame
        pred_future = F.normalize(pred_slots[:, start:].float(), dim=-1)
        target_future = F.normalize(target_slots[:, start:].float(), dim=-1)
        cosine_distance = 1.0 - (pred_future * target_future).sum(dim=-1)
        future_valid = valid_slots[:, None, :].expand_as(cosine_distance)
        valid_count = int(future_valid.sum().item())
        if valid_count:
            feature_loss = cosine_distance[future_valid].mean()
            cosine_similarity = 1.0 - feature_loss.detach()
        else:
            # Preserve graph connectivity while safely skipping an AMG sample
            # for which no usable first-frame boxes were detected.
            feature_loss = cosine_distance.sum() * 0.0
            cosine_similarity = feature_loss.detach()

        metrics = {
            "train/loss_xssc": float(feature_loss.detach().item()),
            "train/xssc_cosine_similarity": float(cosine_similarity.item()),
            "train/xssc_valid_slot_fraction": float(valid_slots.float().mean().item()),
            "train/xssc_future_frames": float(pred_future.shape[1]),
            "train/xssc_pred_below_zero_fraction": float(
                (pred_raw.detach() < 0.0).float().mean().item()
            ),
            "train/xssc_pred_above_one_fraction": float(
                (pred_raw.detach() > 1.0).float().mean().item()
            ),
        }
        visuals = None
        if return_visuals:
            visuals = {
                "target_video": target_video.detach(),
                "pred_video": pred_clipped.detach(),
                "target_slots": target_slots.detach(),
                "pred_slots": pred_slots.detach(),
                "target_attention": target_attention.detach(),
                "pred_attention": pred_attention.detach(),
                "valid_slots": valid_slots.detach(),
            }
        return feature_loss, metrics, visuals

    @staticmethod
    def _normalized_xssc_timestep_weight(
        pipe,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Return scheduler training weight normalized to unit global mean."""
        raw_weight = pipe.scheduler.training_weight(timestep).detach().float()
        all_weights = pipe.scheduler.linear_timesteps_weights.detach().float()
        normalizer = all_weights.mean()
        normalized = raw_weight / normalizer.clamp_min(1e-12)
        return (
            normalized.to(device=pipe.device, dtype=torch.float32),
            float(raw_weight.item()),
            float(normalizer.item()),
        )

    @staticmethod
    def _output_gradient_diagnostics(
        main_loss: torch.Tensor,
        weighted_aux_loss: torch.Tensor,
        model_output: torch.Tensor,
    ) -> dict[str, float]:
        main_grad = torch.autograd.grad(
            main_loss,
            model_output,
            retain_graph=True,
            create_graph=False,
        )[0].detach().float()
        aux_grad = torch.autograd.grad(
            weighted_aux_loss,
            model_output,
            retain_graph=True,
            create_graph=False,
        )[0].detach().float()
        main_norm = torch.linalg.vector_norm(main_grad)
        aux_norm = torch.linalg.vector_norm(aux_grad)
        cosine = (main_grad * aux_grad).sum() / (
            main_norm * aux_norm
        ).clamp_min(1e-20)
        return {
            "train/xssc_grad_diag_applied": 1.0,
            "train/grad_v_main_norm": float(main_norm.item()),
            "train/grad_v_xssc_norm": float(aux_norm.item()),
            "train/grad_v_xssc_to_main_ratio": float(
                (aux_norm / main_norm.clamp_min(1e-20)).item()
            ),
            "train/grad_v_main_xssc_cosine": float(cosine.item()),
        }

    def forward(self, data, inputs=None):
        if isinstance(data, list):
            return super().forward(data, inputs=inputs)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs,
            self.pipe.device,
            self.pipe.torch_dtype,
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss, metrics = self._compute_object_losses(
            self.pipe,
            inputs[0],
            inputs[1],
        )
        self.last_train_metrics = metrics
        return loss

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        captured: list[dict] = []
        original_model_fn = pipe.model_fn

        def capture_model_fn(*args, **kwargs):
            output = original_model_fn(*args, **kwargs)
            captured.append(
                {
                    "model_output": output,
                    "latents": kwargs.get("latents"),
                    "timestep": kwargs.get("timestep"),
                    "inputs": kwargs,
                }
            )
            return output

        pipe.model_fn = capture_model_fn
        try:
            total, metrics = super()._compute_object_losses(
                pipe,
                inputs_shared,
                inputs_posi,
            )
        finally:
            pipe.model_fn = original_model_fn

        if len(captured) != 1:
            raise RuntimeError(
                "Expected exactly one DiT forward in flow-matching loss, "
                f"captured {len(captured)}"
            )
        record = captured[0]
        if not isinstance(record["latents"], torch.Tensor):
            raise RuntimeError("DiT forward did not receive latent x_t")
        if not isinstance(record["model_output"], torch.Tensor):
            raise RuntimeError("DiT forward did not return a tensor v prediction")

        sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler,
            record["timestep"],
        ).to(device=record["latents"].device, dtype=record["latents"].dtype)
        while sigma.ndim < record["latents"].ndim:
            sigma = sigma.unsqueeze(-1)
        pred_x0 = record["latents"] - sigma * record["model_output"]
        target_x0 = inputs_shared["input_latents"]
        pred_x0 = self._restore_condition_latents(
            pred_x0,
            target_x0,
            record["inputs"],
        )
        loss_xssc, xssc_metrics, _ = self._xssc_feature_loss(
            pred_x0,
            target_x0,
        )
        (
            timestep_weight,
            raw_timestep_weight,
            timestep_weight_normalizer,
        ) = self._normalized_xssc_timestep_weight(
            pipe,
            record["timestep"],
        )
        weighted_aux = (
            self.xssc_loss_weight
            * timestep_weight.to(device=loss_xssc.device)
            * loss_xssc
        )

        self._xssc_loss_forward_count += 1
        metrics["train/xssc_grad_diag_applied"] = 0.0
        if (
            self._xssc_loss_forward_count
            % self.xssc_loss_gradient_diagnostics_every_n_forwards
            == 0
        ):
            metrics.update(
                self._output_gradient_diagnostics(
                    total,
                    weighted_aux,
                    record["model_output"],
                )
            )
        total = total + weighted_aux
        metrics.update(xssc_metrics)
        metrics["train/xssc_loss_weight"] = self.xssc_loss_weight
        metrics["train/xssc_timestep_weight_raw"] = raw_timestep_weight
        metrics["train/xssc_timestep_weight_normalizer"] = (
            timestep_weight_normalizer
        )
        metrics["train/xssc_timestep_weight"] = float(
            timestep_weight.detach().item()
        )
        metrics["train/xssc_weighted_contribution"] = float(
            weighted_aux.detach().item()
        )
        metrics["train/xssc_sigma"] = float(sigma.detach().flatten()[0].item())
        metrics["train/loss_total"] = float(total.detach().item())
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description += " Adds frozen xSSC future-slot cosine loss."
    group = parser.add_argument_group("xssc_feature_loss")
    group.add_argument(
        "--xssc_loss_backend",
        choices=VALID_XSSC_LOSS_BACKENDS,
        required=True,
    )
    group.add_argument("--xssc_loss_weight", type=float, required=True)
    group.add_argument(
        "--xssc_loss_future_start_frame",
        type=int,
        default=8,
    )
    group.add_argument(
        "--xssc_loss_backbone_chunk_size",
        type=int,
        default=2,
    )
    group.add_argument(
        "--xssc_loss_gradient_diagnostics_every_n_forwards",
        type=int,
        default=400,
    )
    group.add_argument("--tiny_vae_root", required=True)
    group.add_argument("--tiny_vae_checkpoint", required=True)
    group.add_argument("--tiny_vae_parallel", action="store_true")
    return parser


def build_model(args: argparse.Namespace, accelerator):
    return core.build_model(
        args,
        accelerator,
        model_class=XSSCFeatureLossWanModule,
        extra_model_kwargs={
            "xssc_loss_backend": args.xssc_loss_backend,
            "xssc_loss_weight": args.xssc_loss_weight,
            "xssc_loss_future_start_frame": args.xssc_loss_future_start_frame,
            "xssc_loss_backbone_chunk_size": args.xssc_loss_backbone_chunk_size,
            "xssc_loss_gradient_diagnostics_every_n_forwards": (
                args.xssc_loss_gradient_diagnostics_every_n_forwards
            ),
            "tiny_vae_root": args.tiny_vae_root,
            "tiny_vae_checkpoint": args.tiny_vae_checkpoint,
            "tiny_vae_parallel": args.tiny_vae_parallel,
        },
    )


def log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    core._log_stage_summary(accelerator, model, args)
    if accelerator.is_main_process:
        accelerator.print(
            "Frozen xSSC auxiliary loss: "
            f"backend={args.xssc_loss_backend}, "
            f"checkpoint={args.xssc_checkpoint}, "
            f"weight={args.xssc_loss_weight:g}, "
            f"future_frames=[{args.xssc_loss_future_start_frame}, "
            f"{args.num_frames - 1}], slots={model.xssc_loss_num_slots}, "
            f"slot_dim={model.xssc_loss_slot_dim}, "
            f"backbone_chunk={args.xssc_loss_backbone_chunk_size}, "
            "scheduler_timestep_weight=normalized_global_mean, "
            f"DiT-gradient-checkpointing-offload={args.use_gradient_checkpointing_offload}, "
            f"Tiny-VAE differentiable decode=True (parallel={args.tiny_vae_parallel}), "
            "object branch=False"
        )


def main() -> None:
    core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        log_stage_summary_fn=log_stage_summary,
    )


if __name__ == "__main__":
    main()
