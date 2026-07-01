from __future__ import annotations

from typing import Any

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer

from .oracle_encoder import OracleObjectTokenEncoder
from .runtime_stage1_common import Stage1OracleMixin


class OracleInjectionTrainer(Stage1OracleMixin, ContextVideoTrainer):
    """Stage 1 trainer: inject full-video oracle object context into Wan."""

    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)
        self.oracle_encoder = OracleObjectTokenEncoder(self)

        # Freeze the teacher object-token builder path. Stage 1 only trains Wan's
        # object-conditioning injection modules against the denoising objective.
        self.object_pooler.eval().requires_grad_(False)
        self.object_aux_heads.eval().requires_grad_(False)
        self.object_adapter.train().requires_grad_(True)
        self.jepa_adapter.eval().requires_grad_(False)
        self.vggt_adapter.eval().requires_grad_(False)
        if self.cotracker_adapter is not None:
            self.cotracker_adapter.eval().requires_grad_(False)

        self.bundle.freeze_parts(
            freeze_vae=True,
            freeze_text_encoder=True,
            freeze_dit=True,
            freeze_lora=True,
        )
        if self.bundle.dit is not None:
            self.bundle.dit.train()

    def trainable_parameters(self):
        if self.bundle.dit is None:
            self.bundle.ensure_dit_loaded()
        params = [param for param in self.object_adapter.parameters() if param.requires_grad]
        if self.bundle.dit is not None:
            for name, param in self.bundle.dit.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    "object_embedding." in name
                    or ".object_cross_attn." in name
                    or ".object_gate" in name
                    or ".norm4." in name
                ):
                    params.append(param)
        return params

    def export_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        # Stage1B/1C use full-video oracle tokens for object context. The default
        # context path in super()._prepare_batch() runs JEPA + CoTracker +
        # object_pooler + object_adapter — all of which are frozen and whose
        # results are immediately overwritten by the oracle path below. Running
        # them is pure waste. Instead, extract only what the Wan forward pass
        # actually needs (VAE latents + text context) and build everything else
        # from the oracle encoder directly.
        videos = batch["video"].to(self.device_obj)
        context_videos = batch["context_video"].to(self.device_obj)
        captions = list(batch["caption"])
        num_context_frames = batch["num_context_frames"].to(self.device_obj).long()

        text_ctx = self._encode_text(captions)
        full_latents = self._encode_video_latents(videos)
        context_latents = self._encode_video_latents(context_videos)

        oracle_batch = self._build_oracle_stage1_batch(batch)

        debug: dict[str, Any] = {
            "video": list(videos.shape),
            "context_video": list(context_videos.shape),
            "num_context_frames": num_context_frames.detach().cpu().tolist(),
            "full_latents": [list(t.shape) for t in full_latents],
            "context_latents": [list(t.shape) for t in context_latents],
            "object_context": list(oracle_batch.oracle_object_context.shape),
            "object_latent_tokens": list(oracle_batch.oracle_object_latent_tokens.shape),
            "teacher_student_stage1": {
                "mode": "oracle_full_video_object_context",
                "oracle_object_latent_tokens": list(oracle_batch.oracle_object_latent_tokens.shape),
                "oracle_object_context": list(oracle_batch.oracle_object_context.shape),
            },
        }

        return {
            "videos": videos,
            "context_videos": context_videos,
            "captions": captions,
            "num_context_frames": num_context_frames,
            "full_latents": full_latents,
            "context_latents": context_latents,
            "text_context": text_ctx,
            "object_context": oracle_batch.oracle_object_context.to(self.device_obj),
            "object_latent_tokens": oracle_batch.oracle_object_latent_tokens.to(self.device_obj),
            "object_aux_out": oracle_batch.object_aux_out,
            "object_tokens": oracle_batch.object_tokens.to(self.device_obj),
            "track_box_loss": None,
            "track_iou_loss": None,
            "gt_track_summary": oracle_batch.gt_track_summary,
            "gt_track_valid": oracle_batch.gt_track_valid,
            "gt_box_xyxy": oracle_batch.gt_box_xyxy,
            "gt_box_valid": oracle_batch.gt_box_valid,
            "gt_depth": oracle_batch.gt_depth,
            "gt_depth_valid": oracle_batch.gt_depth_valid,
            "debug": debug,
        }
