from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from code_vjepa_vggt.train_v_newtrain import _sample_points_from_box
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer

from .oracle_encoder import OracleObjectTokenEncoder


@dataclass
class Stage1OracleBatch:
    oracle_object_latent_tokens: torch.Tensor
    oracle_object_context: torch.Tensor
    object_aux_out: Any
    object_tokens: torch.Tensor
    object_valid_mask: torch.Tensor
    gt_track_summary: torch.Tensor | None
    gt_track_valid: torch.Tensor | None
    gt_box_xyxy: torch.Tensor | None
    gt_box_valid: torch.Tensor | None
    gt_depth: torch.Tensor | None
    gt_depth_valid: torch.Tensor | None
    debug: dict[str, Any]


class Stage1OracleMixin:
    oracle_encoder: OracleObjectTokenEncoder
    device_obj: torch.device
    max_objects: int
    points_per_object: int
    object_pooler: Any
    object_aux_heads: Any
    vggt_adapter: Any
    cotracker_adapter: Any
    cfg: dict[str, Any]
    enable_sam2_priors: bool
    _teacher_student_current_batch: dict[str, Any] | None = None

    def _maybe_build_query_priors(
        self,
        context_videos: torch.Tensor,
        num_context_frames: torch.Tensor,
        captions: list[str],
    ):
        # The context path (VGGT / CoTracker) requires query-point priors. With
        # SAM2 priors disabled we derive them from the GT context boxes, exactly
        # as the Stage2 trainer does, so Stage1B/1C can run the default context
        # branch without a SAM2 detector.
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

    def _align_tracks_to_boxes_full(self, tracks: torch.Tensor, gt_boxes: torch.Tensor, *, image_hw: tuple[int, int]):
        from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes

        return align_tracks_to_boxes(
            tracks=tracks,
            gt_boxes=gt_boxes,
            image_hw=image_hw,
        )

    def _build_full_object_targets(
        self,
        *,
        batch: dict[str, Any],
        full_object_latent_tokens: torch.Tensor,
        full_tracks_grouped: torch.Tensor,
        full_visibility_grouped: torch.Tensor,
        full_confidence_grouped: torch.Tensor,
        full_track_image_hw: tuple[int, int],
        object_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if "context_boxes" not in batch or "future_boxes" not in batch:
            return None, None, None, None, None, None
        full_boxes = torch.cat([batch["context_boxes"], batch["future_boxes"]], dim=1).to(self.device_obj)
        full_states = None
        if "context_states" in batch and "future_states" in batch:
            full_states = torch.cat([batch["context_states"], batch["future_states"]], dim=1).to(self.device_obj)

        scale_x = float(batch["video"].shape[-1]) / float(full_track_image_hw[1])
        scale_y = float(batch["video"].shape[-2]) / float(full_track_image_hw[0])
        tracks_native = full_tracks_grouped.clone()
        tracks_native[..., 0] *= scale_x
        tracks_native[..., 1] *= scale_y
        center_tracks_native, center_track_valid = ContextVideoTrainer._object_center_tracks_from_grouped(
            tracks_native,
            full_visibility_grouped,
            full_confidence_grouped,
            object_valid_mask=object_valid_mask,
        )
        track_alignment = self._align_tracks_to_boxes_full(
            center_tracks_native,
            full_boxes,
            image_hw=(int(batch["video"].shape[-2]), int(batch["video"].shape[-1])),
        )
        latent_frames = int(full_object_latent_tokens.shape[1])
        gt_valid_full = (track_alignment.matched_gt_valid > 0.5) & center_track_valid
        gt_track_summary, gt_track_valid = ContextVideoTrainer._group_track_summary(
            track_alignment.matched_gt_centers,
            gt_valid_full,
            image_hw=(int(batch["video"].shape[-2]), int(batch["video"].shape[-1])),
            latent_frames=latent_frames,
        )
        matched_gt_boxes = ContextVideoTrainer._gather_matched_gt_features(
            full_boxes,
            track_alignment.matched_gt_indices,
        )
        matched_gt_box_valid = (
            ((matched_gt_boxes[..., 2] - matched_gt_boxes[..., 0]) > 1.0e-6)
            & ((matched_gt_boxes[..., 3] - matched_gt_boxes[..., 1]) > 1.0e-6)
        )
        gt_box_xyxy, gt_box_valid = ContextVideoTrainer._group_box_targets(
            matched_gt_boxes,
            matched_gt_box_valid,
            latent_frames,
        )

        gt_depth = None
        gt_depth_valid = None
        depth_target_index = self.cfg.get("loss", {}).get(
            "depth_target_state_index",
            self.cfg["model"].get("depth_target_state_index"),
        )
        if depth_target_index is not None and full_states is not None:
            depth_target_index = int(depth_target_index)
            if depth_target_index < 0 or depth_target_index >= int(full_states.shape[-1]):
                raise ValueError(
                    f"depth_target_state_index={depth_target_index} is out of range for "
                    f"full_states shape {list(full_states.shape)}"
                )
            matched_gt_depth = ContextVideoTrainer._gather_matched_gt_features(
                full_states[..., depth_target_index : depth_target_index + 1],
                track_alignment.matched_gt_indices,
            )
            gt_depth = ContextVideoTrainer._group_last(matched_gt_depth, latent_frames)
            gt_depth_valid = gt_box_valid
        return gt_track_summary, gt_track_valid, gt_box_xyxy, gt_box_valid, gt_depth, gt_depth_valid

    def _build_oracle_stage1_batch(self, batch: dict[str, Any]) -> Stage1OracleBatch:
        # The oracle encoder runs the frozen perception backbone under no_grad
        # internally, but keeps the (possibly trainable) object_pooler /
        # object_adapter in the autograd graph. We therefore must NOT wrap this
        # call in no_grad, otherwise Stage1B/1C would silently freeze those
        # modules even though they are listed as trainable.
        oracle_out = self.oracle_encoder.forward_from_batch(
            batch,
            use_full_video_as_context=True,
        )

        aux_outputs = []
        object_tokens = []
        tracks_grouped_all = []
        visibility_grouped_all = []
        confidence_grouped_all = []
        object_valid_masks = []
        track_image_hw_ref = None
        for sample in oracle_out.samples:
            object_out = sample.object_out
            # aux_heads run in the same autograd context as the pooler; they are
            # trainable in Stage1A/1C and frozen (eval + requires_grad_(False))
            # in Stage1B. That is controlled by the trainer, not here.
            aux_out = self.object_aux_heads(
                object_out.object_latent_tokens,
                object_out.active_track_summary,
                object_out.active_box_xyxy,
            )
            aux_outputs.append(aux_out)
            object_tokens.append(object_out.object_tokens)
            tracks_grouped_all.append(sample.tracks_grouped)
            visibility_grouped_all.append(sample.visibility_grouped)
            confidence_grouped_all.append(sample.confidence_grouped)
            object_valid_masks.append(sample.object_valid_mask)
            track_image_hw_ref = sample.track_image_hw

        pred_track_summary = torch.cat([u.pred_track_summary for u in aux_outputs], dim=0)
        pred_box_xyxy = torch.cat([u.pred_box_xyxy for u in aux_outputs], dim=0)
        pred_depth = torch.cat([u.pred_depth for u in aux_outputs], dim=0)
        pred_box_wh = torch.cat([u.pred_box_wh for u in aux_outputs], dim=0)
        track_delta = torch.cat([u.track_delta for u in aux_outputs], dim=0)
        box_center_delta = torch.cat([u.box_center_delta for u in aux_outputs], dim=0)
        box_log_scale = torch.cat([u.box_log_scale for u in aux_outputs], dim=0)

        class _PackedAux:
            pass

        packed_aux = _PackedAux()
        packed_aux.pred_track_summary = pred_track_summary
        packed_aux.pred_box_xyxy = pred_box_xyxy
        packed_aux.pred_depth = pred_depth
        packed_aux.pred_box_wh = pred_box_wh
        packed_aux.track_delta = track_delta
        packed_aux.box_center_delta = box_center_delta
        packed_aux.box_log_scale = box_log_scale

        object_valid_mask_batch = torch.cat(object_valid_masks, dim=0)
        gt_track_summary, gt_track_valid, gt_box_xyxy, gt_box_valid, gt_depth, gt_depth_valid = self._build_full_object_targets(
            batch=batch,
            full_object_latent_tokens=oracle_out.object_latent_tokens,
            full_tracks_grouped=torch.cat(tracks_grouped_all, dim=0),
            full_visibility_grouped=torch.cat(visibility_grouped_all, dim=0),
            full_confidence_grouped=torch.cat(confidence_grouped_all, dim=0),
            full_track_image_hw=track_image_hw_ref,
            object_valid_mask=object_valid_mask_batch,
        )

        debug = {
            "mode": "stage1_oracle_full_token",
            "oracle_object_latent_tokens": list(oracle_out.object_latent_tokens.shape),
            "oracle_object_context": list(oracle_out.object_context.shape),
            "object_aux_pred_track_summary": list(pred_track_summary.shape),
            "object_aux_pred_box_xyxy": list(pred_box_xyxy.shape),
            "object_aux_pred_depth": list(pred_depth.shape),
            "object_valid_mask": list(object_valid_mask_batch.shape),
        }
        if oracle_out.samples:
            repair_debug = getattr(oracle_out.samples[0], "query_repair_debug", None)
            if repair_debug is not None:
                debug["query_repair_debug"] = repair_debug
        if gt_track_summary is not None:
            debug["gt_track_summary"] = list(gt_track_summary.shape)
        if gt_box_xyxy is not None:
            debug["gt_box_xyxy"] = list(gt_box_xyxy.shape)
        if gt_depth is not None:
            debug["gt_depth"] = list(gt_depth.shape)

        return Stage1OracleBatch(
            oracle_object_latent_tokens=oracle_out.object_latent_tokens,
            oracle_object_context=oracle_out.object_context,
            object_aux_out=packed_aux,
            object_tokens=torch.cat(object_tokens, dim=0),
            object_valid_mask=object_valid_mask_batch,
            gt_track_summary=gt_track_summary,
            gt_track_valid=gt_track_valid,
            gt_box_xyxy=gt_box_xyxy,
            gt_box_valid=gt_box_valid,
            gt_depth=gt_depth,
            gt_depth_valid=gt_depth_valid,
            debug=debug,
        )
