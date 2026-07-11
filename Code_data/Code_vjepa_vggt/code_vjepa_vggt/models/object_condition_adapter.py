from __future__ import annotations

import torch
import torch.nn as nn


class ObjectConditionAdapter(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        num_slots: int = 8,
        max_time_steps: int = 64,
        output_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.max_time_steps = int(max_time_steps)
        self.slot_embed = nn.Embedding(self.num_slots, self.dim)
        self.time_embed = nn.Embedding(self.max_time_steps, self.dim)
        # Projects normalized xyxy bbox [0,1] coords into token space.
        # Zero-initialized so spatial grounding starts neutral and grows with
        # training — mirrors the object_gate zero-init pattern in Wan DiT.
        self.bbox_embed = nn.Linear(4, self.dim, bias=False)
        nn.init.zeros_(self.bbox_embed.weight)
        self.norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.out_norm = nn.LayerNorm(self.dim)
        gate_init = torch.tensor(float(output_gate_init)).clamp(1.0e-4, 1.0 - 1.0e-4)
        self.output_gate_logit = nn.Parameter(torch.logit(gate_init))
        self.mlp_residual_max_ratio: float | None = None
        self._last_mlp_residual_ratio: torch.Tensor | None = None
        self._last_mlp_cap_scale: torch.Tensor | None = None

    def pop_mlp_diagnostics(self) -> tuple[torch.Tensor | None, dict[str, float]]:
        ratio = self._last_mlp_residual_ratio
        scale = self._last_mlp_cap_scale
        self._last_mlp_residual_ratio = None
        self._last_mlp_cap_scale = None
        if ratio is None:
            return None, {
                "mean_ratio": 0.0,
                "max_ratio": 0.0,
                "cap_applied_fraction": 0.0,
                "cap_scale_min": 1.0,
            }
        ratio_detached = ratio.detach().float()
        if scale is None:
            scale_detached = torch.ones_like(ratio_detached)
        else:
            scale_detached = scale.detach().float()
        return ratio, {
            "mean_ratio": float(ratio_detached.mean().item()),
            "max_ratio": float(ratio_detached.max().item()),
            "cap_applied_fraction": float((scale_detached < 0.9999).float().mean().item()),
            "cap_scale_min": float(scale_detached.min().item()),
        }

    def forward(
        self,
        object_latent_tokens: torch.Tensor,
        *,
        object_valid_mask: torch.Tensor | None = None,
        bbox_xyxy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        object_latent_tokens: [B, T, O, dim]
        object_valid_mask:    [B, O]   float, 1 = valid slot
        bbox_xyxy:            [B, T, O, 4]  normalized xyxy in [0, 1]; optional
        Returns: [B, T*O, dim]
        """
        batch, time_steps, slots, dim = object_latent_tokens.shape
        if int(slots) > self.num_slots:
            raise ValueError(f"slots={slots} exceeds configured num_slots={self.num_slots}")
        if int(time_steps) > self.max_time_steps:
            raise ValueError(f"time_steps={time_steps} exceeds configured max_time_steps={self.max_time_steps}")
        slot_ids = torch.arange(slots, device=object_latent_tokens.device)
        time_ids = torch.arange(time_steps, device=object_latent_tokens.device)
        slot_bias = self.slot_embed(slot_ids).view(1, 1, slots, dim)
        time_bias = self.time_embed(time_ids).view(1, time_steps, 1, dim)
        x = torch.nan_to_num(object_latent_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        x = x + slot_bias + time_bias
        if bbox_xyxy is not None:
            # [B, T, O, 4] -> [B, T, O, dim]; clamp to guard against degenerate boxes
            bbox = bbox_xyxy.to(dtype=x.dtype, device=x.device).clamp(0.0, 1.0)
            x = x + self.bbox_embed(bbox)
        x = self.norm(x)
        mlp_residual = self.mlp(x)
        x_rms = x.float().square().mean(dim=-1).clamp_min(1.0e-12).sqrt()
        mlp_rms = mlp_residual.float().square().mean(dim=-1).clamp_min(1.0e-12).sqrt()
        mlp_ratio = mlp_rms / x_rms
        cap_scale = torch.ones_like(mlp_ratio)
        if self.mlp_residual_max_ratio is not None and float(self.mlp_residual_max_ratio) > 0.0:
            cap_scale = (
                float(self.mlp_residual_max_ratio) / mlp_ratio.clamp_min(1.0e-12)
            ).clamp(max=1.0)
            mlp_residual = mlp_residual * cap_scale.unsqueeze(-1).to(dtype=mlp_residual.dtype)
        self._last_mlp_residual_ratio = mlp_ratio
        self._last_mlp_cap_scale = cap_scale
        x = x + mlp_residual
        x = self.out_norm(x)
        x = x * torch.sigmoid(self.output_gate_logit).to(dtype=x.dtype, device=x.device)
        if object_valid_mask is not None:
            slot_mask = object_valid_mask[:, None, :, None].to(dtype=x.dtype, device=x.device)
            x = x * slot_mask
        return x.view(batch, time_steps * slots, dim)
