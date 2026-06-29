from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from code_vjepa_vggt.train_v_newtrain import _sample_points_from_box
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer

from .common import (
    build_future_track_summary,
    future_object_valid_mask,
    group_future_boxes,
    split_context_future_tokens,
)
from .future_heads import FutureObjectHeads
from .losses import compute_predictor_losses
from .oracle_encoder import OracleObjectTokenEncoder
from .predictor import FutureObjectTokenPredictor


@dataclass
class PredictorPreparedBatch:
    teacher_context_tokens: torch.Tensor
    teacher_future_tokens: torch.Tensor
    pred_future_tokens: torch.Tensor
    pred_future_track_summary: torch.Tensor
    pred_future_box_xyxy: torch.Tensor
    gt_future_track_summary: torch.Tensor
    gt_future_track_valid: torch.Tensor
    gt_future_box_xyxy: torch.Tensor
    gt_future_box_valid: torch.Tensor
    object_valid_mask: torch.Tensor


class TeacherStudentPredictorTrainer(ContextVideoTrainer):
    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)
        model_cfg = cfg["model"]
        loss_cfg = cfg.get("loss", {})
        dim = int(model_cfg.get("cond_proj_dim", 4096))
        self.predictor = FutureObjectTokenPredictor(
            dim=dim,
            max_context_latent_frames=int(model_cfg.get("predictor_max_context_latent_frames", 8)),
            max_future_latent_frames=int(model_cfg.get("predictor_max_future_latent_frames", 8)),
            num_slots=int(model_cfg.get("sam2_max_objects", self.max_objects)),
        ).to(self.device_obj)
        self.future_heads = FutureObjectHeads(
            dim=dim,
            track_delta_scale=float(loss_cfg.get("future_track_delta_scale", 0.06)),
            box_delta_scale=float(loss_cfg.get("future_box_delta_scale", 0.06)),
            box_wh_log_scale=float(loss_cfg.get("future_box_wh_log_scale", 1.25)),
            box_wh_max_scale=float(loss_cfg.get("future_box_wh_max_scale", 2.0)),
        ).to(self.device_obj)
        self.oracle_encoder = OracleObjectTokenEncoder(self)
        self.lambda_future_token = float(loss_cfg.get("lambda_future_token", 1.0))
        self.lambda_future_track = float(loss_cfg.get("lambda_future_track", 0.1))
        self.lambda_future_box = float(loss_cfg.get("lambda_future_box", 0.1))
        self._teacher_student_current_batch: dict[str, Any] | None = None

        self.object_pooler.eval().requires_grad_(False)
        self.object_aux_heads.eval().requires_grad_(False)
        self.object_adapter.eval().requires_grad_(False)
        if self.bundle.dit is not None:
            self.bundle.dit.eval().requires_grad_(False)
        self.bundle.freeze_parts(
            freeze_vae=True,
            freeze_text_encoder=True,
            freeze_dit=True,
            freeze_lora=True,
        )

    def trainable_parameters(self):
        params = list(self.predictor.parameters()) + list(self.future_heads.parameters())
        return [param for param in params if param.requires_grad]

    def export_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    def _maybe_build_query_priors(
        self,
        context_videos: torch.Tensor,
        num_context_frames: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str], list[str], list[dict[str, Any]]]:
        if self.enable_sam2_priors:
            return super()._maybe_build_query_priors(context_videos, num_context_frames, captions)
        batch = self._teacher_student_current_batch
        if batch is None or "context_boxes" not in batch:
            return super()._maybe_build_query_priors(context_videos, num_context_frames, captions)
        context_boxes = batch["context_boxes"].to(context_videos.device)
        batch_size = int(context_boxes.shape[0])
        grouped_queries = torch.zeros(
            batch_size,
            int(self.max_objects),
            int(self.points_per_object),
            2,
            dtype=context_videos.dtype,
            device=context_videos.device,
        )
        object_valid_mask = torch.zeros(
            batch_size,
            int(self.max_objects),
            dtype=context_videos.dtype,
            device=context_videos.device,
        )
        prior_debugs: list[dict[str, Any]] = []
        for batch_idx in range(batch_size):
            valid_frames = int(num_context_frames[batch_idx].item())
            for object_idx in range(int(self.max_objects)):
                first_box = None
                for frame_idx in range(valid_frames):
                    candidate = context_boxes[batch_idx, frame_idx, object_idx]
                    if bool((candidate[2] - candidate[0] > 1.0e-6) and (candidate[3] - candidate[1] > 1.0e-6)):
                        first_box = candidate
                        break
                if first_box is None:
                    continue
                points = _sample_points_from_box(first_box.detach().float().cpu(), int(self.points_per_object)).to(
                    device=context_videos.device,
                    dtype=context_videos.dtype,
                )
                points[:, 0] *= float(context_videos.shape[-1])
                points[:, 1] *= float(context_videos.shape[-2])
                grouped_queries[batch_idx, object_idx] = points
                object_valid_mask[batch_idx, object_idx] = 1.0
            prior_debugs.append(
                {
                    "strategy": "gt_box_queries",
                    "prior_source": "gt_box_queries",
                    "object_count": int(object_valid_mask[batch_idx].sum().item()),
                    "valid_frames": valid_frames,
                }
            )
        return (
            grouped_queries,
            object_valid_mask,
            ["gt_box_queries"] * batch_size,
            ["gt_box_queries"] * batch_size,
            prior_debugs,
        )

    def _prepare_predictor_batch(self, batch: dict[str, Any]) -> PredictorPreparedBatch:
        with torch.no_grad():
            self._teacher_student_current_batch = batch
            try:
                context_prepared = self._prepare_batch(batch)
            finally:
                self._teacher_student_current_batch = None
        oracle_out = self.oracle_encoder.forward_from_batch(batch, use_full_video_as_context=True)
        context_tokens = context_prepared["object_latent_tokens"].detach()
        context_latent_frames = int(context_tokens.shape[1])
        slices = split_context_future_tokens(
            oracle_out.object_latent_tokens,
            context_latent_frames=context_latent_frames,
        )
        future_latent_frames = int(slices.future_tokens.shape[1])
        future_boxes = batch["future_boxes"][:, :, : int(self.max_objects)].to(self.device_obj)
        object_valid_mask = future_object_valid_mask(future_boxes)
        predictor_out = self.predictor(
            context_tokens,
            future_latent_frames=future_latent_frames,
            object_valid_mask=object_valid_mask,
        )
        gt_future_track_summary, gt_future_track_valid = build_future_track_summary(
            future_boxes,
            image_hw=(int(batch["video"].shape[-2]), int(batch["video"].shape[-1])),
            future_latent_frames=future_latent_frames,
        )
        gt_future_box_xyxy, gt_future_box_valid = group_future_boxes(
            future_boxes,
            future_latent_frames=future_latent_frames,
        )
        base_track_summary = gt_future_track_summary.detach()
        base_box_xyxy = gt_future_box_xyxy.detach()
        future_head_out = self.future_heads(
            predictor_out.pred_future_tokens,
            base_track_summary=base_track_summary,
            base_box_xyxy=base_box_xyxy,
        )
        return PredictorPreparedBatch(
            teacher_context_tokens=context_tokens,
            teacher_future_tokens=slices.future_tokens,
            pred_future_tokens=predictor_out.pred_future_tokens,
            pred_future_track_summary=future_head_out.pred_track_summary,
            pred_future_box_xyxy=future_head_out.pred_box_xyxy,
            gt_future_track_summary=gt_future_track_summary,
            gt_future_track_valid=gt_future_track_valid,
            gt_future_box_xyxy=gt_future_box_xyxy,
            gt_future_box_valid=gt_future_box_valid,
            object_valid_mask=object_valid_mask,
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        prepared = self._prepare_predictor_batch(batch)
        loss_out = compute_predictor_losses(
            pred_future_tokens=prepared.pred_future_tokens,
            teacher_future_tokens=prepared.teacher_future_tokens,
            pred_future_track_summary=prepared.pred_future_track_summary,
            gt_future_track_summary=prepared.gt_future_track_summary,
            gt_future_track_valid=prepared.gt_future_track_valid,
            pred_future_box_xyxy=prepared.pred_future_box_xyxy,
            gt_future_box_xyxy=prepared.gt_future_box_xyxy,
            gt_future_box_valid=prepared.gt_future_box_valid,
            lambda_token=self.lambda_future_token,
            lambda_track=self.lambda_future_track,
            lambda_box=self.lambda_future_box,
        )
        self.last_loss_breakdown = {
            "loss_total": loss_out.loss_total,
            "loss_future_token": loss_out.loss_token,
            "loss_future_track": loss_out.loss_track,
            "loss_future_box": loss_out.loss_box,
        }
        self.last_train_metrics = {
            "train/loss_total": float(loss_out.loss_total.detach().item()),
            "train/loss_future_token": float(loss_out.loss_token.detach().item()),
            "train/loss_future_track": float(loss_out.loss_track.detach().item()),
            "train/loss_future_box": float(loss_out.loss_box.detach().item()),
            "train/object_context_abs_max": 0.0,
            "train/object_latent_tokens_abs_max": float(prepared.teacher_future_tokens.detach().abs().max().item()),
            "train/pred_future_tokens_abs_max": float(prepared.pred_future_tokens.detach().abs().max().item()),
        }
        return loss_out.loss_total
