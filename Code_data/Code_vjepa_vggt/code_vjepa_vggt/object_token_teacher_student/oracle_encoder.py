from __future__ import annotations

from dataclasses import dataclass

import torch

from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.train_v_newtrain import _sample_points_from_box


@dataclass
class OracleTokenOutput:
    object_latent_tokens: torch.Tensor
    object_context: torch.Tensor


class OracleObjectTokenEncoder:
    def __init__(
        self,
        trainer,
    ) -> None:
        self.trainer = trainer
        self._oracle_jepa_adapters: dict[int, JEPAPatchAdapter] = {}

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
        if batch != 1:
            raise NotImplementedError("first oracle version only supports per-sample query-prior construction inside batched loop")
        height, width = int(image_hw[0]), int(image_hw[1])
        grouped_points = []
        query_frame_ids = []
        object_valid_mask = []
        box_priors = []
        for object_idx in range(int(self.trainer.max_objects)):
            first_valid_frame = None
            first_box = None
            for frame_idx in range(total_frames):
                candidate = full_boxes[0, frame_idx, object_idx]
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
        query_points_prior = torch.stack(grouped_points, dim=0).view(1, int(self.trainer.total_object_queries), 2)
        query_frame_ids_tensor = torch.tensor(query_frame_ids, dtype=torch.float32, device=full_boxes.device).view(
            1, int(self.trainer.total_object_queries), 1
        )
        object_valid_mask_tensor = torch.tensor(object_valid_mask, dtype=torch.float32, device=full_boxes.device).view(
            1, int(self.trainer.max_objects)
        )
        box_prior_xyxy = torch.stack(box_priors, dim=0).view(1, int(self.trainer.max_objects), 4)
        return query_points_prior, query_frame_ids_tensor, object_valid_mask_tensor, box_prior_xyxy

    @torch.no_grad()
    def forward_from_batch(
        self,
        batch: dict,
        *,
        use_full_video_as_context: bool,
    ) -> OracleTokenOutput:
        if not use_full_video_as_context:
            raise NotImplementedError("only full-video oracle token extraction is implemented in the first version")
        full_video = batch["video"].to(self.trainer.device_obj)
        full_boxes = torch.cat([batch["context_boxes"], batch["future_boxes"]], dim=1).to(self.trainer.device_obj)
        batch_size = int(full_video.shape[0])
        token_outputs = []
        context_outputs = []
        for sample_idx in range(batch_size):
            sample_video = full_video[sample_idx : sample_idx + 1]
            sample_boxes = full_boxes[sample_idx : sample_idx + 1]
            image_hw = (int(sample_video.shape[-2]), int(sample_video.shape[-1]))
            query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = self._build_object_query_priors(
                sample_boxes,
                image_hw=image_hw,
            )
            frames_bthwc = sample_video.permute(0, 2, 3, 4, 1).float()
            frames_bthwc = (frames_bthwc + 1.0) / 2.0
            jepa_adapter = self._oracle_jepa_adapter(int(sample_video.shape[2]))
            jepa_out = jepa_adapter(sample_video)
            context_latents = self.trainer._encode_video_latents(sample_video)[0].unsqueeze(0).to(self.trainer.device_obj)
            vggt_out = self.trainer.vggt_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=image_hw,
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
            object_out = self.trainer.object_pooler(
                jepa_patch_tokens=jepa_out.patch_tokens,
                context_latents=context_latents,
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
            object_context = self.trainer.object_adapter(
                object_out.object_latent_tokens,
                object_valid_mask=object_valid_mask,
            )
            token_outputs.append(object_out.object_latent_tokens)
            context_outputs.append(object_context)
        return OracleTokenOutput(
            object_latent_tokens=torch.cat(token_outputs, dim=0),
            object_context=torch.cat(context_outputs, dim=0),
        )
