from __future__ import annotations

from typing import Any

import torch

from .runtime_stage1b_context_only import ContextOnlyInjectionTrainer
from .viewer_grounding_box_provider import ViewerGroundingBoxProvider, ViewerGroundingSample


class ContextOnlyInjectionNoGTBoxTrainer(ContextOnlyInjectionTrainer):
    """Stage1B context-only trainer that uses viewer-style GDINO+SAM2 pseudo boxes."""

    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__(cfg=cfg, build_optimizer=build_optimizer, device=device)
        model_cfg = cfg["model"]
        grounding_device = str(model_cfg.get("grounding_device", str(self.device_obj)))
        include_caption_terms = not bool(model_cfg.get("grounding_disable_caption_terms", True))
        self.viewer_grounding = ViewerGroundingBoxProvider(
            device=grounding_device,
            segment_len=int(model_cfg.get("sam2_segment_len", 8)),
            max_objects=int(self.max_objects),
            points_per_object=int(self.points_per_object),
            proposal_source=str(model_cfg.get("grounding_proposal_source", "gdino_only")),
            motion_score_ratio=float(model_cfg.get("grounding_motion_score_ratio", 0.15)),
            text_prompt=str(model_cfg.get("grounding_text_prompt", "box . cube . block . cylinder . capsule . sphere . ball .")),
            extra_prompt_terms=str(model_cfg.get("grounding_extra_prompt_terms", "")),
            include_caption_terms=include_caption_terms,
            gdino_box_threshold=float(model_cfg.get("grounding_gdino_box_threshold", 0.20)),
            gdino_text_threshold=float(model_cfg.get("grounding_gdino_text_threshold", 0.15)),
            prompt_frame_mode=str(model_cfg.get("grounding_prompt_frame_mode", "first")),
            track_dedupe_iou_threshold=float(model_cfg.get("grounding_track_dedupe_iou_threshold", 0.75)),
            container_suppress_ratio_threshold=float(model_cfg.get("grounding_container_suppress_ratio_threshold", 0.95)),
            container_suppress_min_contained=int(model_cfg.get("grounding_container_suppress_min_contained", 2)),
            container_suppress_min_area_ratio=float(model_cfg.get("grounding_container_suppress_min_area_ratio", 1.5)),
            container_suppress_small_iou_threshold=float(model_cfg.get("grounding_container_suppress_small_iou_threshold", 0.7)),
        )
        self._cached_query_points_grouped: torch.Tensor | None = None
        self._cached_object_valid_mask: torch.Tensor | None = None
        self._cached_prior_sources: list[str] | None = None
        self._cached_prompt_modes: list[str] | None = None
        self._cached_prior_debugs: list[dict[str, Any]] | None = None

    def _clear_grounding_cache(self) -> None:
        self._cached_query_points_grouped = None
        self._cached_object_valid_mask = None
        self._cached_prior_sources = None
        self._cached_prompt_modes = None
        self._cached_prior_debugs = None

    def _maybe_build_query_priors(
        self,
        context_videos: torch.Tensor,
        num_context_frames: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str], list[str], list[dict[str, Any]]]:
        if self._cached_query_points_grouped is not None:
            return (
                self._cached_query_points_grouped,
                self._cached_object_valid_mask,
                list(self._cached_prior_sources or []),
                list(self._cached_prompt_modes or []),
                list(self._cached_prior_debugs or []),
            )
        return super()._maybe_build_query_priors(context_videos, num_context_frames, captions)

    def _build_viewer_grounding_batch(self, batch: dict[str, Any]) -> tuple[dict[str, Any], list[ViewerGroundingSample]]:
        context_videos = batch["context_video"].to(self.device_obj)
        captions = list(batch["caption"])
        num_context_frames = batch["num_context_frames"].to(self.device_obj).long()
        batch_size = int(context_videos.shape[0])
        target_context_frames = int(context_videos.shape[2])
        image_hw = (int(context_videos.shape[-2]), int(context_videos.shape[-1]))

        grouped_queries = []
        object_valid_masks = []
        prior_sources: list[str] = []
        prompt_modes: list[str] = []
        prior_debugs: list[dict[str, Any]] = []
        pseudo_context_boxes = torch.zeros(
            batch_size,
            target_context_frames,
            int(self.max_objects),
            4,
            dtype=context_videos.dtype,
            device=self.device_obj,
        )
        samples: list[ViewerGroundingSample] = []

        for batch_idx in range(batch_size):
            valid_frames = int(num_context_frames[batch_idx].item())
            frames_tchw_01 = ((context_videos[batch_idx, :, :valid_frames].permute(1, 0, 2, 3).float() + 1.0) / 2.0).detach().cpu().numpy()
            sample = self.viewer_grounding.build_sample(
                frames_tchw_01=frames_tchw_01,
                caption=captions[batch_idx],
                image_hw=image_hw,
            )
            grouped_queries.append(torch.from_numpy(sample.grouped_queries_px))
            object_valid_masks.append(torch.from_numpy(sample.object_valid_mask))
            prior_sources.append(sample.prior_source)
            prompt_modes.append(sample.prompt_mode)
            prior_debugs.append(sample.debug)
            pseudo_context_boxes[batch_idx, : valid_frames] = torch.from_numpy(sample.context_boxes_norm).to(
                device=self.device_obj,
                dtype=context_videos.dtype,
            )
            samples.append(sample)

        self._cached_query_points_grouped = torch.stack(grouped_queries, dim=0).to(device=self.device_obj, dtype=context_videos.dtype)
        self._cached_object_valid_mask = torch.stack(object_valid_masks, dim=0).to(device=self.device_obj, dtype=context_videos.dtype)
        self._cached_prior_sources = prior_sources
        self._cached_prompt_modes = prompt_modes
        self._cached_prior_debugs = prior_debugs

        pseudo_batch = dict(batch)
        pseudo_batch["context_boxes"] = pseudo_context_boxes
        if "future_boxes" in batch:
            pseudo_batch["future_boxes"] = torch.zeros_like(batch["future_boxes"])
        return pseudo_batch, samples

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pseudo_batch, samples = self._build_viewer_grounding_batch(batch)
        try:
            prepared = super()._prepare_batch(pseudo_batch)
        finally:
            self._clear_grounding_cache()
        debug = dict(prepared.get("debug", {}))
        stage1_debug = dict(debug.get("teacher_student_stage1", {}))
        stage1_debug.update(
            {
                "mode": "context_only_object_context_viewer_grounding",
                "box_source": "viewer_grounding_gdino_sam2",
                "depends_on_dataset_boxes": False,
                "grounding_object_counts": [sample.debug["object_count"] for sample in samples],
            }
        )
        debug["teacher_student_stage1"] = stage1_debug
        debug["grounding_samples"] = [sample.debug for sample in samples]
        prepared["debug"] = debug
        return prepared
