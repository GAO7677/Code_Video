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
        self.object_adapter.eval().requires_grad_(False)
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
        params = []
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
        prepared = super()._prepare_batch(batch)
        oracle_batch = self._build_oracle_stage1_batch(batch)
        prepared["object_context"] = oracle_batch.oracle_object_context.to(
            device=self.device_obj,
            dtype=prepared["object_context"].dtype,
        )
        prepared["object_latent_tokens"] = oracle_batch.oracle_object_latent_tokens.to(
            device=self.device_obj,
            dtype=prepared["object_latent_tokens"].dtype,
        )
        prepared["object_aux_out"] = oracle_batch.object_aux_out
        prepared["object_tokens"] = oracle_batch.object_tokens.to(
            device=self.device_obj,
            dtype=prepared["object_tokens"].dtype,
        )
        prepared["debug"]["teacher_student_stage1"] = {
            "mode": "oracle_full_video_object_context",
            "oracle_object_latent_tokens": list(oracle_batch.oracle_object_latent_tokens.shape),
            "oracle_object_context": list(oracle_batch.oracle_object_context.shape),
            "default_context_object_context": list(prepared["debug"]["object_context"]),
        }
        prepared["debug"]["object_context"] = list(oracle_batch.oracle_object_context.shape)
        prepared["debug"]["object_latent_tokens"] = list(oracle_batch.oracle_object_latent_tokens.shape)
        return prepared
