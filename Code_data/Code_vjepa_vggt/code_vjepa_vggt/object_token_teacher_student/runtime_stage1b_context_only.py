from __future__ import annotations

from typing import Any

import torch

from code_vjepa_vggt.train_v_newtrain import _sample_points_from_box
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer


class ContextOnlyInjectionTrainer(ContextVideoTrainer):
    """Stage1B context-only trainer: consume context object tokens directly, without predictor or future tokens."""

    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)
        self._teacher_student_current_batch: dict[str, Any] | None = None

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

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._teacher_student_current_batch = batch
        try:
            prepared = super()._prepare_batch(batch)
        finally:
            self._teacher_student_current_batch = None
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
