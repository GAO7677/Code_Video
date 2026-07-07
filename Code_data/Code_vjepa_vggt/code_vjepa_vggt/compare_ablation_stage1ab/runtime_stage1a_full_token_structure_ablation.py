from __future__ import annotations

from typing import Any

import torch

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackOutput
from code_vjepa_vggt.object_token_teacher_student.oracle_encoder import OracleObjectTokenEncoder
from code_vjepa_vggt.object_token_teacher_student.runtime_stage1a_full_token import FullTokenTeacherTrainer

from .structure_ablation_shared import (
    StructureAblationObjectTubeProjector,
    VALID_STAGE1AB_ABLATIONS,
)


class _NullJEPAOutput:
    def __init__(self) -> None:
        self.patch_tokens = None


class StructureAblationOracleObjectTokenEncoder(OracleObjectTokenEncoder):
    def __init__(
        self,
        trainer,
        *,
        disable_jepa: bool,
        disable_vggt: bool,
    ) -> None:
        super().__init__(trainer)
        self.disable_jepa = bool(disable_jepa)
        self.disable_vggt = bool(disable_vggt)

    @torch.no_grad()
    def _run_frozen_perception(
        self,
        batch_video: torch.Tensor,
        batch_boxes: torch.Tensor,
    ) -> dict[str, Any]:
        image_hw = (int(batch_video.shape[-2]), int(batch_video.shape[-1]))
        query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = self._build_object_query_priors(
            batch_boxes,
            image_hw=image_hw,
        )
        frames_bthwc = batch_video.permute(0, 2, 3, 4, 1).float()
        frames_bthwc = (frames_bthwc + 1.0) / 2.0

        jepa_out = _NullJEPAOutput()
        if not self.disable_jepa:
            jepa_adapter = self._oracle_jepa_adapter(int(batch_video.shape[2]))
            jepa_out = jepa_adapter(batch_video)

        context_latents_list = []
        for b in range(int(batch_video.shape[0])):
            lat = self.trainer._encode_video_latents(batch_video[b : b + 1])[0].unsqueeze(0)
            context_latents_list.append(lat.to(self.trainer.device_obj))
        context_latents = torch.cat(context_latents_list, dim=0)

        if not self.disable_vggt and self.trainer.vggt_adapter is not None and self.trainer.vggt_adapter.model is not None:
            vggt_out = self.trainer.vggt_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=image_hw,
            )
        else:
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
        }


class StructureAblationFullTokenTeacherTrainer(FullTokenTeacherTrainer):
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        structure_ablation_type: str,
        build_optimizer: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        ablation = str(structure_ablation_type).strip().lower()
        if ablation not in VALID_STAGE1AB_ABLATIONS:
            raise ValueError(
                f"unsupported structure_ablation_type={structure_ablation_type!r}; "
                f"expected one of {VALID_STAGE1AB_ABLATIONS}"
            )
        self.structure_ablation_type = ablation
        self.disable_jepa = ablation == "wo_jepa"
        self.disable_vggt = ablation == "wo_vggt"

        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)

        self.object_pooler = StructureAblationObjectTubeProjector.from_existing(
            self.object_pooler,
            disable_cotracker=False,
            disable_jepa=self.disable_jepa,
            disable_vggt=self.disable_vggt,
        )
        if self.disable_jepa:
            self.jepa_adapter = None
        if self.disable_vggt:
            self.vggt_adapter = None

        self.oracle_encoder = StructureAblationOracleObjectTokenEncoder(
            self,
            disable_jepa=self.disable_jepa,
            disable_vggt=self.disable_vggt,
        )

    def _prepare_stage1a_batch(self, batch: dict[str, Any]):
        prepared = super()._prepare_stage1a_batch(batch)
        prepared.debug["structure_ablation_type"] = self.structure_ablation_type
        prepared.debug["disable_jepa"] = self.disable_jepa
        prepared.debug["disable_vggt"] = self.disable_vggt
        return prepared
