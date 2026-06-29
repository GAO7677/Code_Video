from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

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
        with torch.no_grad():
            oracle_out = self.oracle_encoder.forward_from_batch(
                batch,
                use_full_video_as_context=True,
            )

        full_video = batch["video"].to(self.device_obj)
        full_boxes = torch.cat([batch["context_boxes"], batch["future_boxes"]], dim=1).to(self.device_obj)
        batch_size = int(full_video.shape[0])
        aux_outputs = []
        object_tokens = []
        tracks_grouped_all = []
        visibility_grouped_all = []
        confidence_grouped_all = []
        object_valid_masks = []
        track_image_hw_ref = None
        for sample_idx in range(batch_size):
            sample_video = full_video[sample_idx : sample_idx + 1]
            sample_boxes = full_boxes[sample_idx : sample_idx + 1]
            image_hw = (int(sample_video.shape[-2]), int(sample_video.shape[-1]))
            query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = self.oracle_encoder._build_object_query_priors(
                sample_boxes,
                image_hw=image_hw,
            )
            frames_bthwc = sample_video.permute(0, 2, 3, 4, 1).float()
            frames_bthwc = (frames_bthwc + 1.0) / 2.0
            jepa_adapter = self.oracle_encoder._oracle_jepa_adapter(int(sample_video.shape[2]))
            jepa_out = jepa_adapter(sample_video)
            full_latents = self._encode_video_latents(sample_video)[0].unsqueeze(0).to(self.device_obj)
            vggt_out = self.vggt_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=image_hw,
            )
            cotracker_out = None
            if self.cotracker_adapter is not None:
                cotracker_out = self.cotracker_adapter(
                    frames_bthwc,
                    query_points_prior=query_points_prior,
                    query_frame_ids=query_frame_ids,
                    query_image_hw=image_hw,
                )
            if cotracker_out is not None:
                tracks = cotracker_out.tracks
                visibility = cotracker_out.visibility
                confidence = cotracker_out.confidence
                track_image_hw = cotracker_out.image_hw
            else:
                tracks = vggt_out.tracks
                visibility = vggt_out.visibility
                confidence = vggt_out.confidence
                track_image_hw = vggt_out.image_hw
            track_image_hw_ref = track_image_hw
            tracks_grouped, visibility_grouped, confidence_grouped = ContextVideoTrainer._group_tracks_to_objects(
                tracks,
                visibility,
                confidence,
                max_objects=self.max_objects,
                points_per_object=self.points_per_object,
            )
            object_out = self.object_pooler(
                jepa_patch_tokens=jepa_out.patch_tokens,
                context_latents=full_latents,
                tracks=tracks_grouped,
                visibility=visibility_grouped,
                confidence=confidence_grouped,
                track_image_hw=track_image_hw,
                object_valid_mask=object_valid_mask,
                box_prior_xyxy=box_prior_xyxy,
                vggt_world_points=vggt_out.world_points,
                vggt_world_points_conf=vggt_out.world_points_conf,
                vggt_depth=vggt_out.depth,
                vggt_depth_conf=vggt_out.depth_conf,
                vggt_dense_patch_tokens=vggt_out.dense_patch_tokens,
                vggt_patch_grid_hw=vggt_out.patch_grid_hw,
                vggt_geometry_image_hw=vggt_out.image_hw,
                frame_valid_mask=None,
            )
            aux_out = self.object_aux_heads(
                object_out.object_latent_tokens,
                object_out.active_track_summary,
                object_out.active_box_xyxy,
            )
            aux_outputs.append(aux_out)
            object_tokens.append(object_out.object_tokens)
            tracks_grouped_all.append(tracks_grouped)
            visibility_grouped_all.append(visibility_grouped)
            confidence_grouped_all.append(confidence_grouped)
            object_valid_masks.append(object_valid_mask)

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
