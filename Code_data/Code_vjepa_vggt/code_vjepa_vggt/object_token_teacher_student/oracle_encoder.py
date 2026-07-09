from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.train0710querypoints.gt_box_query_repair import GTBoxRepairConfig
from code_vjepa_vggt.train0710querypoints.gt_box_query_repair import repair_grouped_queries_with_aligned_gt_boxes
from code_vjepa_vggt.train_v_newtrain import _sample_points_from_box


@dataclass
class OracleSampleArtifacts:
    """Per-sample intermediates produced by a single oracle forward.

    Stage 1 reuses these for aux heads / target building so the heavy frozen
    perception path (JEPA + VGGT + CoTracker + VAE) only runs once per sample.
    """

    object_out: Any
    object_context: torch.Tensor
    tracks_grouped: torch.Tensor
    visibility_grouped: torch.Tensor
    confidence_grouped: torch.Tensor
    track_image_hw: tuple[int, int]
    object_valid_mask: torch.Tensor
    query_repair_debug: dict[str, Any] | None = None


@dataclass
class OracleTokenOutput:
    object_latent_tokens: torch.Tensor
    object_context: torch.Tensor
    samples: list[OracleSampleArtifacts] = field(default_factory=list)


class OracleObjectTokenEncoder:
    def __init__(
        self,
        trainer,
    ) -> None:
        self.trainer = trainer
        self._oracle_jepa_adapters: dict[int, JEPAPatchAdapter] = {}
        model_cfg = trainer.cfg.get("model", {})
        self.gt_box_query_repair = GTBoxRepairConfig(
            enabled=bool(model_cfg.get("grounding_gt_box_query_repair", False)),
            oversample_factor=int(model_cfg.get("grounding_gt_box_oversample_factor", 4)),
            min_visible_ratio=float(model_cfg.get("grounding_gt_box_min_visible_ratio", 0.60)),
            min_in_box_ratio=float(model_cfg.get("grounding_gt_box_min_in_box_ratio", 0.60)),
        )

    def _oracle_jepa_adapter(self, total_frames: int) -> JEPAPatchAdapter:
        total_frames = int(total_frames)
        configured_context_frames = int(self.trainer.cfg["data"]["num_context_frames"])
        if total_frames == configured_context_frames:
            return self.trainer.jepa_adapter
        adapter = self._oracle_jepa_adapters.get(total_frames)
        if adapter is not None:
            return adapter
        model_cfg = self.trainer.cfg["model"]
        adapter = JEPAPatchAdapter(
            ckpt_path=str(model_cfg["je_pa_ckpt_dir"]) + "/original/model.pth",
            device=str(self.trainer.device_obj),
            crop_size=int(model_cfg["jepa_input_size"]),
            num_frames=total_frames,
            patch_size=int(model_cfg["jepa_patch_size"]),
            tubelet_size=int(model_cfg["jepa_tubelet_size"]),
            use_activation_checkpointing=bool(model_cfg.get("jepa_activation_checkpointing", False)),
            trainable=False,
        ).to(self.trainer.device_obj)
        self._oracle_jepa_adapters[total_frames] = adapter
        return adapter

    def _build_object_query_priors(
        self,
        full_boxes: torch.Tensor,
        *,
        image_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, total_frames, _, _ = full_boxes.shape
        height, width = int(image_hw[0]), int(image_hw[1])
        all_query_points = []
        all_query_frame_ids = []
        all_object_valid_masks = []
        all_box_priors = []
        for b in range(batch):
            grouped_points = []
            query_frame_ids = []
            object_valid_mask = []
            box_priors = []
            for object_idx in range(int(self.trainer.max_objects)):
                first_valid_frame = None
                first_box = None
                for frame_idx in range(total_frames):
                    candidate = full_boxes[b, frame_idx, object_idx]
                    if bool((candidate[2] - candidate[0] > 1.0e-6) and (candidate[3] - candidate[1] > 1.0e-6)):
                        first_valid_frame = frame_idx
                        first_box = candidate
                        break
                if first_box is None:
                    object_valid_mask.append(0.0)
                    cx = 0.5 * float(width)
                    cy = 0.5 * float(height)
                    points = torch.tensor(
                        [[cx, cy]] * int(self.trainer.points_per_object),
                        dtype=torch.float32,
                        device=full_boxes.device,
                    )
                    box_priors.append(torch.tensor([0.45, 0.45, 0.55, 0.55], dtype=torch.float32, device=full_boxes.device))
                    query_frame_ids.extend([0.0] * int(self.trainer.points_per_object))
                else:
                    object_valid_mask.append(1.0)
                    points = _sample_points_from_box(first_box, int(self.trainer.points_per_object)).to(
                        device=full_boxes.device,
                        dtype=torch.float32,
                    )
                    points[:, 0] *= float(width)
                    points[:, 1] *= float(height)
                    box_priors.append(first_box.to(device=full_boxes.device, dtype=torch.float32))
                    query_frame_ids.extend([float(first_valid_frame)] * int(self.trainer.points_per_object))
                grouped_points.append(points)
            all_query_points.append(torch.stack(grouped_points, dim=0).view(1, int(self.trainer.total_object_queries), 2))
            all_query_frame_ids.append(
                torch.tensor(query_frame_ids, dtype=torch.float32, device=full_boxes.device).view(1, int(self.trainer.total_object_queries), 1)
            )
            all_object_valid_masks.append(
                torch.tensor(object_valid_mask, dtype=torch.float32, device=full_boxes.device).view(1, int(self.trainer.max_objects))
            )
            all_box_priors.append(torch.stack(box_priors, dim=0).view(1, int(self.trainer.max_objects), 4))
        query_points_prior = torch.cat(all_query_points, dim=0)        # [B, N_total, 2]
        query_frame_ids_tensor = torch.cat(all_query_frame_ids, dim=0) # [B, N_total, 1]
        object_valid_mask_tensor = torch.cat(all_object_valid_masks, dim=0)  # [B, N_obj]
        box_prior_xyxy = torch.cat(all_box_priors, dim=0)              # [B, N_obj, 4]
        return query_points_prior, query_frame_ids_tensor, object_valid_mask_tensor, box_prior_xyxy

    @torch.no_grad()
    def _run_frozen_perception(
        self,
        batch_video: torch.Tensor,
        batch_boxes: torch.Tensor,
    ) -> dict[str, Any]:
        """Run frozen perception backbone for a whole batch under no_grad.

        batch_video: [B, C, T, H, W]
        batch_boxes: [B, T, N_obj, 4]
        """
        image_hw = (int(batch_video.shape[-2]), int(batch_video.shape[-1]))
        query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = self._build_object_query_priors(
            batch_boxes,
            image_hw=image_hw,
        )
        repair_debug: dict[str, Any] = {"applied": False, "reason": "not_requested"}
        frames_bthwc = batch_video.permute(0, 2, 3, 4, 1).float()
        frames_bthwc = (frames_bthwc + 1.0) / 2.0

        if bool(self.gt_box_query_repair.enabled) and self.trainer.cotracker_adapter is not None:
            repaired_queries_batches: list[torch.Tensor] = []
            repair_items: list[dict[str, Any]] = []
            for batch_idx in range(int(batch_video.shape[0])):
                grouped_queries_px = query_points_prior[batch_idx].detach().float().cpu().view(
                    int(self.trainer.max_objects),
                    int(self.trainer.points_per_object),
                    2,
                ).numpy()
                valid_mask_np = object_valid_mask[batch_idx].detach().float().cpu().numpy()
                repaired_np, repair_item = repair_grouped_queries_with_aligned_gt_boxes(
                    gt_boxes_tn4_norm=batch_boxes[batch_idx].detach().float().cpu().numpy(),
                    image_hw=image_hw,
                    frames_bthwc_01=frames_bthwc[batch_idx : batch_idx + 1],
                    grouped_queries_px=grouped_queries_px,
                    object_valid_mask=valid_mask_np,
                    points_per_object=int(self.trainer.points_per_object),
                    run_cotracker=self.trainer.cotracker_adapter,
                    config=self.gt_box_query_repair,
                )
                repaired_queries_batches.append(
                    torch.from_numpy(np.asarray(repaired_np, dtype=np.float32)).view(1, int(self.trainer.total_object_queries), 2)
                )
                repair_items.append(repair_item)
            query_points_prior = torch.cat(repaired_queries_batches, dim=0).to(
                device=batch_video.device,
                dtype=batch_video.dtype,
            )
            repair_debug = {
                "applied": True,
                "batch_items": repair_items,
            }

        jepa_adapter = self._oracle_jepa_adapter(int(batch_video.shape[2]))
        jepa_out = jepa_adapter(batch_video)

        # VAE encode each sample; _encode_video_latents operates on [C, T, H, W]
        context_latents_list = []
        for b in range(int(batch_video.shape[0])):
            lat = self.trainer._encode_video_latents(batch_video[b : b + 1])[0].unsqueeze(0)
            context_latents_list.append(lat.to(self.trainer.device_obj))
        context_latents = torch.cat(context_latents_list, dim=0)  # [B, C_lat, T_lat, H_lat, W_lat]

        if self.trainer.vggt_adapter.model is not None:
            vggt_out = self.trainer.vggt_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=image_hw,
            )
        else:
            from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackOutput
            _B, _T = frames_bthwc.shape[0], frames_bthwc.shape[1]
            _N = query_points_prior.shape[1]
            _dev = frames_bthwc.device
            vggt_out = VGGTTrackOutput(
                query_points=query_points_prior,
                tracks=torch.zeros(_B, _T, _N, 2, device=_dev),
                visibility=torch.ones(_B, _T, _N, device=_dev),
                confidence=torch.ones(_B, _T, _N, device=_dev),
                image_hw=image_hw,
                used_model=False,
            )

        cotracker_out = None
        if self.trainer.cotracker_adapter is not None:
            cotracker_out = self.trainer.cotracker_adapter(
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

        tracks_grouped, visibility_grouped, confidence_grouped = self.trainer._group_tracks_to_objects(
            tracks,
            visibility,
            confidence,
            max_objects=self.trainer.max_objects,
            points_per_object=self.trainer.points_per_object,
        )
        return {
            "jepa_out": jepa_out,
            "context_latents": context_latents,
            "vggt_out": vggt_out,
            "tracks_grouped": tracks_grouped,
            "visibility_grouped": visibility_grouped,
            "confidence_grouped": confidence_grouped,
            "track_image_hw": track_image_hw,
            "object_valid_mask": object_valid_mask,
            "box_prior_xyxy": box_prior_xyxy,
            "query_points_prior": query_points_prior,
            "query_repair_debug": repair_debug,
        }

    def forward_from_batch(
        self,
        batch: dict,
        *,
        use_full_video_as_context: bool,
    ) -> OracleTokenOutput:
        """Build oracle object tokens from the full video (batched)."""
        if not use_full_video_as_context:
            raise NotImplementedError("only full-video oracle token extraction is implemented")
        full_video = batch["video"].to(self.trainer.device_obj)
        full_boxes = torch.cat([batch["context_boxes"], batch["future_boxes"]], dim=1).to(self.trainer.device_obj)

        perception = self._run_frozen_perception(full_video, full_boxes)
        vggt_out = perception["vggt_out"]

        object_out = self.trainer.object_pooler(
            jepa_patch_tokens=perception["jepa_out"].patch_tokens,
            context_latents=perception["context_latents"],
            tracks=perception["tracks_grouped"],
            visibility=perception["visibility_grouped"],
            confidence=perception["confidence_grouped"],
            track_image_hw=perception["track_image_hw"],
            object_valid_mask=perception["object_valid_mask"],
            box_prior_xyxy=perception["box_prior_xyxy"],
            vggt_world_points=vggt_out.world_points,
            vggt_world_points_conf=vggt_out.world_points_conf,
            vggt_depth=vggt_out.depth,
            vggt_depth_conf=vggt_out.depth_conf,
            vggt_dense_patch_tokens=vggt_out.dense_patch_tokens,
            vggt_patch_grid_hw=vggt_out.patch_grid_hw,
            vggt_geometry_image_hw=vggt_out.image_hw,
            frame_valid_mask=None,
        )
        # Subsample full_boxes [B, T_all, N_obj_data, 4] to latent-frame resolution
        # and clip the slot dimension to match the adapter's num_slots, which may
        # differ from the dataset's max_objects (e.g. num_slots=4, max_objects=6).
        T_lat = int(object_out.object_latent_tokens.shape[1])
        O_tok = int(object_out.object_latent_tokens.shape[2])  # adapter's slot count
        T_all = int(full_boxes.shape[1])
        lat_indices = torch.linspace(0, T_all - 1, T_lat, device=full_boxes.device).long()
        full_boxes_lat = full_boxes[:, lat_indices, :O_tok]  # [B, T_lat, O_tok, 4]

        object_context = self.trainer.object_adapter(
            object_out.object_latent_tokens,
            object_valid_mask=perception["object_valid_mask"],
            bbox_xyxy=full_boxes_lat,
        )
        # Build per-sample OracleSampleArtifacts for compatibility with Stage1 callers
        batch_size = int(full_video.shape[0])
        samples: list[OracleSampleArtifacts] = []
        for b in range(batch_size):
            samples.append(OracleSampleArtifacts(
                object_out=object_out,
                object_context=object_context[b : b + 1],
                tracks_grouped=perception["tracks_grouped"][b : b + 1],
                visibility_grouped=perception["visibility_grouped"][b : b + 1],
                confidence_grouped=perception["confidence_grouped"][b : b + 1],
                track_image_hw=perception["track_image_hw"],
                object_valid_mask=perception["object_valid_mask"][b : b + 1],
                query_repair_debug=perception.get("query_repair_debug"),
            ))
        return OracleTokenOutput(
            object_latent_tokens=object_out.object_latent_tokens,
            object_context=object_context,
            samples=samples,
        )
