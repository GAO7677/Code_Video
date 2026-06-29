from __future__ import annotations

from typing import Any

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer

from .oracle_encoder import OracleObjectTokenEncoder
from .runtime_stage1_common import Stage1OracleBatch, Stage1OracleMixin


class FullTokenTeacherTrainer(Stage1OracleMixin, ContextVideoTrainer):
    """Stage1A trainer: learn full-video object tokens without Wan denoising."""

    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=False, device=device)
        self.oracle_encoder = OracleObjectTokenEncoder(self)

        # Stage1A does not optimize or even use Wan denoising.
        self.bundle.freeze_parts(
            freeze_vae=True,
            freeze_text_encoder=True,
            freeze_dit=True,
            freeze_lora=True,
        )
        if self.bundle.dit is not None:
            self.bundle.dit.eval().requires_grad_(False)

        # Freeze the perception/backbone path. Only train token builder + aux heads.
        self.jepa_adapter.eval().requires_grad_(False)
        self.vggt_adapter.eval().requires_grad_(False)
        if self.cotracker_adapter is not None:
            self.cotracker_adapter.eval().requires_grad_(False)
        self.object_adapter.eval().requires_grad_(False)

    def trainable_parameters(self):
        params = list(self.object_pooler.parameters()) + list(self.object_aux_heads.parameters())
        return [param for param in params if param.requires_grad]

    def export_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    def _prepare_stage1a_batch(self, batch: dict[str, Any]) -> Stage1OracleBatch:
        prepared = self._build_oracle_stage1_batch(batch)
        prepared.debug["video"] = list(batch["video"].shape)
        prepared.debug["trainable_modules"] = [
            "object_pooler",
            "object_aux_heads",
        ]
        return prepared

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        prepared = self._prepare_stage1a_batch(batch)
        aux = prepared.object_aux_out
        loss_total = prepared.oracle_object_latent_tokens.new_zeros(())
        loss_track = loss_total.clone()
        loss_box = loss_total.clone()
        loss_depth = loss_total.clone()

        if prepared.gt_track_summary is not None and prepared.gt_track_valid is not None:
            weights = prepared.gt_track_valid.unsqueeze(-1).to(
                dtype=aux.pred_track_summary.dtype,
                device=aux.pred_track_summary.device,
            )
            denom = weights.sum().clamp_min(1.0) * aux.pred_track_summary.shape[-1]
            loss_track = ((aux.pred_track_summary - prepared.gt_track_summary).abs() * weights).sum() / denom

        if prepared.gt_box_xyxy is not None and prepared.gt_box_valid is not None:
            weights = prepared.gt_box_valid.unsqueeze(-1).to(
                dtype=aux.pred_box_xyxy.dtype,
                device=aux.pred_box_xyxy.device,
            )
            denom = weights.sum().clamp_min(1.0) * aux.pred_box_xyxy.shape[-1]
            loss_box = ((aux.pred_box_xyxy - prepared.gt_box_xyxy).abs() * weights).sum() / denom

        if prepared.gt_depth is not None and prepared.gt_depth_valid is not None:
            weights = prepared.gt_depth_valid.unsqueeze(-1).to(
                dtype=aux.pred_depth.dtype,
                device=aux.pred_depth.device,
            )
            denom = weights.sum().clamp_min(1.0) * aux.pred_depth.shape[-1]
            loss_depth = ((aux.pred_depth - prepared.gt_depth).abs() * weights).sum() / denom

        loss_cfg = self.cfg.get("loss", {})
        lambda_track = float(loss_cfg.get("lambda_track_aux", 1.0))
        lambda_box = float(loss_cfg.get("lambda_box_aux", 1.0))
        lambda_depth = float(loss_cfg.get("lambda_depth_aux", 1.0))
        loss_total = (
            lambda_track * loss_track
            + lambda_box * loss_box
            + lambda_depth * loss_depth
        )

        self.last_loss_breakdown = {
            "loss_total": loss_total,
            "loss_full_track": loss_track,
            "loss_full_box": loss_box,
            "loss_full_depth": loss_depth,
        }
        self.last_train_metrics = {
            "train/loss_total": float(loss_total.detach().item()),
            "train/loss_full_track": float(loss_track.detach().item()),
            "train/loss_full_box": float(loss_box.detach().item()),
            "train/loss_full_depth": float(loss_depth.detach().item()),
            "train/object_tokens_abs_max": float(prepared.object_tokens.detach().abs().max().item()),
            "train/object_latent_tokens_abs_max": float(prepared.oracle_object_latent_tokens.detach().abs().max().item()),
            "train/object_context_abs_max": float(prepared.oracle_object_context.detach().abs().max().item()),
            "train/track_delta_abs_mean": float(aux.track_delta.detach().abs().mean().item()),
            "train/box_center_delta_abs_mean": float(aux.box_center_delta.detach().abs().mean().item()),
            "train/box_log_scale_abs_mean": float(aux.box_log_scale.detach().abs().mean().item()),
        }
        return loss_total

    @torch.no_grad()
    def inspect_one_batch(self) -> dict[str, Any]:
        loader = self.build_dataloader(num_workers=0)
        batch = next(iter(loader))
        prepared = self._prepare_stage1a_batch(batch)
        return prepared.debug
