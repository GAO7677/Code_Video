from __future__ import annotations

from typing import Any

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer


class ContextOnlyInjectionTrainer(ContextVideoTrainer):
    """Stage1B context-only trainer: consume context object tokens directly, without predictor or future tokens."""

    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)

        # Keep the Stage1A token builder fixed. This stage only teaches the
        # adapter and Wan object-injection branch to consume context-only tokens.
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
        unique = []
        seen = set()
        for param in params:
            if not param.requires_grad:
                continue
            key = id(param)
            if key in seen:
                continue
            seen.add(key)
            unique.append(param)
        return unique

    def export_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = super()._prepare_batch(batch)
        debug = dict(prepared.get("debug", {}))
        debug["teacher_student_stage1"] = {
            "mode": "context_only_object_context",
            "context_object_latent_tokens": list(prepared["object_latent_tokens"].shape),
            "context_object_context": list(prepared["object_context"].shape),
            "future_token_predictor": False,
            "oracle_full_video_replacement": False,
        }
        prepared["debug"] = debug
        return prepared
