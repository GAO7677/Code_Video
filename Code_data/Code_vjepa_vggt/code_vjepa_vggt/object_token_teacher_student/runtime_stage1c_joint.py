from __future__ import annotations

from typing import Any

import torch

from .runtime_stage1 import OracleInjectionTrainer


class Stage1CJointTrainer(OracleInjectionTrainer):
    """Stage1C joint finetune: keep oracle injection but reopen small token-side modules."""

    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)

        joint_cfg = cfg.get("stage1c", {})
        train_object_pooler_tail = bool(joint_cfg.get("train_object_pooler_tail", True))
        train_object_aux_heads = bool(joint_cfg.get("train_object_aux_heads", True))
        train_wan_lora = bool(joint_cfg.get("train_wan_lora", False))

        self.object_adapter.train().requires_grad_(True)

        if train_object_pooler_tail:
            for name in ("modal_refine", "out_norm", "jepa_router_score", "latent_router_score", "track_geometry_router_score", "appearance_router_score"):
                module = getattr(self.object_pooler, name, None)
                if module is not None:
                    module.train().requires_grad_(True)
        if train_object_aux_heads:
            self.object_aux_heads.train().requires_grad_(True)
        if train_wan_lora and self.bundle.dit is not None:
            for name, param in self.bundle.dit.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True

    def trainable_parameters(self):
        params = list(super().trainable_parameters())
        params.extend([p for p in self.object_adapter.parameters() if p.requires_grad])
        params.extend([p for p in self.object_aux_heads.parameters() if p.requires_grad])
        params.extend([p for p in self.object_pooler.parameters() if p.requires_grad])
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
