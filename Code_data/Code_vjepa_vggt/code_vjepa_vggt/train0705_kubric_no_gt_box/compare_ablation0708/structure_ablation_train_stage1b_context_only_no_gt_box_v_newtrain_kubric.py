"""Kubric Stage1B structure-ablation training entry for compare_ablation0708.

This file is intentionally separate from the formal Kubric training entry so
ablation-specific structural changes do not affect the main experiment path.

Currently supported structural ablations:
  - ``wo_jepa``: remove the JEPA branch from object-token construction
  - ``wo_vggt``: remove the VGGT geometry / depth branch
"""
from __future__ import annotations

import argparse

import torch

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.models.object_tokens import ObjectTokenOutput, ObjectTubeProjector
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as base0708,
)
from diffsynth.diffusion import ModelLogger


_STRUCTURE_ABLATIONS = ("none", "wo_jepa", "wo_vggt")


def _copy_requires_grad_by_name(source: torch.nn.Module, target: torch.nn.Module) -> None:
    source_flags = {
        name: bool(param.requires_grad)
        for name, param in source.named_parameters()
    }
    for name, param in target.named_parameters():
        if name in source_flags:
            param.requires_grad = source_flags[name]


class StructureAblationObjectTubeProjector(ObjectTubeProjector):
    def __init__(
        self,
        *,
        jepa_dim: int,
        latent_dim: int,
        out_dim: int,
        vggt_dense_dim: int,
        jepa_window_radius: int,
        latent_window_radius: int,
        min_box_px: float,
        disable_jepa: bool = False,
        disable_vggt: bool = False,
    ) -> None:
        super().__init__(
            jepa_dim=jepa_dim,
            latent_dim=latent_dim,
            out_dim=out_dim,
            vggt_dense_dim=vggt_dense_dim,
            jepa_window_radius=jepa_window_radius,
            latent_window_radius=latent_window_radius,
            min_box_px=min_box_px,
        )
        self.disable_jepa = bool(disable_jepa)
        self.disable_vggt = bool(disable_vggt)

        if self.disable_jepa:
            del self.jepa_proj
            del self.jepa_router_score
        if self.disable_vggt:
            del self.vggt_geom_point_proj
            del self.depth_proj

    @classmethod
    def from_existing(
        cls,
        source: ObjectTubeProjector,
        *,
        disable_jepa: bool,
        disable_vggt: bool,
    ) -> "StructureAblationObjectTubeProjector":
        jepa_dim = int(source.jepa_proj.in_features)
        latent_dim = int(source.latent_proj.in_features)
        out_dim = int(source.out_dim)
        if hasattr(source, "vggt_geom_point_proj") and source.vggt_geom_point_proj is not None:
            first_linear = source.vggt_geom_point_proj[0]
            vggt_dense_dim = int(first_linear.in_features) - 7
        else:
            vggt_dense_dim = 2048

        device = source.latent_proj.weight.device
        dtype = source.latent_proj.weight.dtype
        replacement = cls(
            jepa_dim=jepa_dim,
            latent_dim=latent_dim,
            out_dim=out_dim,
            vggt_dense_dim=vggt_dense_dim,
            jepa_window_radius=int(source.jepa_window_radius),
            latent_window_radius=int(source.latent_window_radius),
            min_box_px=float(source.min_box_px),
            disable_jepa=disable_jepa,
            disable_vggt=disable_vggt,
        ).to(device=device, dtype=dtype)
        replacement.load_state_dict(source.state_dict(), strict=False)
        replacement.train(source.training)
        _copy_requires_grad_by_name(source, replacement)
        return replacement

    def forward(
        self,
        jepa_patch_tokens: torch.Tensor | None,
        context_latents: torch.Tensor,
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        track_image_hw: tuple[int, int],
        object_valid_mask: torch.Tensor | None = None,
        box_prior_xyxy: torch.Tensor | None = None,
        vggt_world_points: torch.Tensor | None = None,
        vggt_world_points_conf: torch.Tensor | None = None,
        vggt_depth: torch.Tensor | None = None,
        vggt_depth_conf: torch.Tensor | None = None,
        vggt_dense_patch_tokens: torch.Tensor | None = None,
        vggt_patch_grid_hw: tuple[int, int] | None = None,
        vggt_geometry_image_hw: tuple[int, int] | None = None,
        frame_valid_mask: torch.Tensor | None = None,
    ) -> ObjectTokenOutput:
        return_dtype = context_latents.dtype
        with torch.autocast(device_type=context_latents.device.type, enabled=False):
            context_latents = torch.nan_to_num(context_latents.float(), nan=0.0, posinf=0.0, neginf=0.0)
            tracks = torch.nan_to_num(tracks.float(), nan=0.0, posinf=0.0, neginf=0.0)
            visibility = torch.nan_to_num(visibility.float(), nan=0.0, posinf=0.0, neginf=0.0)
            confidence = torch.nan_to_num(confidence.float(), nan=0.0, posinf=0.0, neginf=0.0)
            tracks, visibility, confidence = self._ensure_grouped_tracks(tracks, visibility, confidence)
            feature_device = tracks.device
            if object_valid_mask is not None:
                object_valid_mask = torch.nan_to_num(
                    object_valid_mask.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if jepa_patch_tokens is not None:
                jepa_patch_tokens = torch.nan_to_num(
                    jepa_patch_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_depth is not None:
                vggt_depth = torch.nan_to_num(
                    vggt_depth.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_depth_conf is not None:
                vggt_depth_conf = torch.nan_to_num(
                    vggt_depth_conf.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_dense_patch_tokens is not None:
                vggt_dense_patch_tokens = torch.nan_to_num(
                    vggt_dense_patch_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)

            batch, src_frames, objects, points, _ = tracks.shape
            latent_frames = int(context_latents.shape[2])

            latent_time_idx = self._time_indices(src_frames, latent_frames, tracks.device)
            latent_tracks = tracks[:, latent_time_idx]
            latent_visibility = visibility[:, latent_time_idx]
            latent_confidence = confidence[:, latent_time_idx]
            latent_frame_valid_mask = (
                frame_valid_mask[:, latent_time_idx]
                if frame_valid_mask is not None
                else None
            )
            flat_latent_tracks, _, _ = self._flatten_point_axis(latent_tracks)

            latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
            latent_local = self._pool_feature_grid(
                latent_grid,
                flat_latent_tracks,
                image_hw=track_image_hw,
                window_radius=self.latent_window_radius,
            )
            latent_local = self._restore_point_axis(latent_local, objects, points)
            point_weights_lat = self._point_weights(
                latent_visibility,
                latent_confidence,
                frame_valid_mask=latent_frame_valid_mask,
            )
            latent_local = self._aggregate_points(latent_local, point_weights_lat)
            self._ensure_latent_proj(int(latent_local.shape[-1]), latent_local.device)
            latent_latent_tokens = self.latent_proj(latent_local)

            if self.disable_jepa:
                jepa_latent_tokens = torch.zeros_like(latent_latent_tokens)
                appearance_latent_tokens = latent_latent_tokens
            else:
                if jepa_patch_tokens is None:
                    raise ValueError("jepa_patch_tokens is required unless JEPA is structurally removed.")
                jepa_time_idx = self._time_indices(
                    src_frames,
                    int(jepa_patch_tokens.shape[1]),
                    tracks.device,
                )
                jepa_tracks = tracks[:, jepa_time_idx]
                flat_jepa_tracks, _, _ = self._flatten_point_axis(jepa_tracks)
                jepa_local = self._pool_feature_grid(
                    jepa_patch_tokens,
                    flat_jepa_tracks,
                    image_hw=track_image_hw,
                    window_radius=self.jepa_window_radius,
                )
                if int(jepa_local.shape[1]) % max(latent_frames, 1) != 0:
                    raise ValueError(
                        f"JEPA frames ({jepa_local.shape[1]}) must be divisible by latent frames ({latent_frames})"
                    )
                jepa_valid = frame_valid_mask[:, jepa_time_idx] if frame_valid_mask is not None else None
                jepa_local = self._temporal_group_mean(
                    jepa_local,
                    latent_frames,
                    frame_valid_mask=jepa_valid,
                )
                jepa_local = self._restore_point_axis(jepa_local, objects, points)
                jepa_local = self._aggregate_points(jepa_local, point_weights_lat)
                expected_jepa_dim = int(self.jepa_proj.in_features)
                if int(jepa_local.shape[-1]) != expected_jepa_dim:
                    actual_jepa_dim = int(jepa_local.shape[-1])
                    if actual_jepa_dim % expected_jepa_dim == 0:
                        fold = actual_jepa_dim // expected_jepa_dim
                        jepa_local = jepa_local.reshape(
                            *jepa_local.shape[:-1],
                            fold,
                            expected_jepa_dim,
                        ).mean(dim=-2)
                    else:
                        self._ensure_jepa_proj(actual_jepa_dim, jepa_local.device)
                jepa_latent_tokens = self.jepa_proj(jepa_local)
                appearance_latent_tokens = self._pair_fuse(
                    jepa_latent_tokens,
                    latent_latent_tokens,
                    self.jepa_router_score,
                    self.latent_router_score,
                )

            motion_xy = latent_tracks
            motion_xy = self._resize_tracks_xy(
                motion_xy,
                src_hw=track_image_hw,
                dst_hw=track_image_hw,
                align_corners=False,
            )
            motion_xy_norm = torch.stack(
                [
                    motion_xy[..., 0] / max(float(track_image_hw[1] - 1), 1.0),
                    motion_xy[..., 1] / max(float(track_image_hw[0] - 1), 1.0),
                ],
                dim=-1,
            ).clamp(0.0, 1.0)
            motion_delta = motion_xy_norm.clone()
            motion_delta[:, 1:] = motion_xy_norm[:, 1:] - motion_xy_norm[:, :-1]
            motion_delta[:, 0] = 0.0
            motion_local = torch.cat(
                [
                    motion_xy_norm,
                    motion_delta,
                    latent_visibility.unsqueeze(-1),
                    latent_confidence.unsqueeze(-1),
                ],
                dim=-1,
            )
            motion_point_tokens = self.motion_point_proj(motion_local)
            motion_latent_tokens = self._point_attention_pool(
                motion_point_tokens,
                self.motion_router_score,
                point_weights_lat,
            )

            center_tracks, center_track_valid = self._center_tracks_from_grouped(
                latent_tracks,
                latent_visibility,
                latent_confidence,
            )
            active_track_summary = self._track_summary(
                center_tracks,
                center_track_valid.to(dtype=center_tracks.dtype),
                center_track_valid.to(dtype=center_tracks.dtype),
                image_hw=track_image_hw,
                target_frames=latent_frames,
                frame_valid_mask=latent_frame_valid_mask,
            )
            active_box_xyxy = self._boxes_from_tracks(
                latent_tracks,
                latent_visibility,
                latent_confidence,
                image_hw=track_image_hw,
                target_frames=latent_frames,
                box_prior_xyxy=box_prior_xyxy,
                min_box_px=self.min_box_px,
            )
            box_geom_latent_tokens = self.track_geom_proj(active_track_summary)

            depth_latent_tokens = None
            geom_latent_tokens = box_geom_latent_tokens
            vggt_geom_tokens = None
            if not self.disable_vggt and vggt_dense_patch_tokens is not None:
                geometry_patch_hw = (
                    tuple(int(v) for v in vggt_patch_grid_hw)
                    if vggt_patch_grid_hw is not None
                    else self._grid_feature_hw(vggt_dense_patch_tokens)
                )
                geometry_tracks = self._resize_tracks_xy(
                    tracks.reshape(batch, src_frames, objects * points, 2),
                    src_hw=track_image_hw,
                    dst_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                    align_corners=False,
                ).view(batch, src_frames, objects, points, 2)
                geometry_tracks = geometry_tracks[:, latent_time_idx]
                patch_tracks = self._resize_tracks_xy(
                    geometry_tracks.reshape(batch, latent_frames, objects * points, 2),
                    src_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                    dst_hw=geometry_patch_hw,
                    align_corners=False,
                ).view(batch, latent_frames, objects, points, 2)
                flat_patch_tracks, _, _ = self._flatten_point_axis(patch_tracks)
                geom_local = self._pool_feature_grid(
                    vggt_dense_patch_tokens[:, latent_time_idx],
                    flat_patch_tracks,
                    image_hw=geometry_patch_hw,
                    window_radius=0,
                )
                geom_local = self._restore_point_axis(geom_local, objects, points)
                flat_geometry_tracks, _, _ = self._flatten_point_axis(geometry_tracks)
                if vggt_depth is not None:
                    depth_local = self._pool_feature_grid(
                        vggt_depth[:, latent_time_idx],
                        flat_geometry_tracks,
                        image_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                        window_radius=0,
                    ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                    depth_local = self._restore_point_axis(depth_local, objects, points)
                else:
                    depth_local = geom_local.new_zeros(*geom_local.shape[:-1], 1)
                geom_point_features = torch.cat(
                    [geom_local, depth_local, motion_local],
                    dim=-1,
                )
                geom_point_tokens = self.vggt_geom_point_proj(geom_point_features)
                geom_latent_tokens = self._point_attention_pool(
                    geom_point_tokens,
                    self.geom_router_score,
                    point_weights_lat,
                )
                vggt_geom_tokens = geom_latent_tokens.mean(dim=1)

            track_geom_latent_tokens = self._pair_fuse(
                motion_latent_tokens,
                geom_latent_tokens,
                self.motion_router_score,
                self.geom_router_score,
            )

            if not self.disable_vggt and vggt_depth is not None:
                geometry_image_hw = (
                    tuple(int(v) for v in vggt_geometry_image_hw)
                    if vggt_geometry_image_hw is not None
                    else tuple(int(v) for v in track_image_hw)
                )
                geometry_tracks = self._resize_tracks_xy(
                    tracks.reshape(batch, src_frames, objects * points, 2),
                    src_hw=track_image_hw,
                    dst_hw=geometry_image_hw,
                    align_corners=False,
                )
                geometry_tracks = geometry_tracks.view(batch, src_frames, objects, points, 2)
                geometry_tracks = geometry_tracks[:, latent_time_idx]
                flat_geometry_tracks, _, _ = self._flatten_point_axis(geometry_tracks)
                depth_local = self._pool_feature_grid(
                    vggt_depth[:, latent_time_idx],
                    flat_geometry_tracks,
                    image_hw=geometry_image_hw,
                    window_radius=0,
                ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                depth_local = self._restore_point_axis(depth_local, objects, points)
                if vggt_depth_conf is not None:
                    depth_conf_local = self._pool_feature_grid(
                        vggt_depth_conf[:, latent_time_idx].unsqueeze(-1),
                        flat_geometry_tracks,
                        image_hw=geometry_image_hw,
                        window_radius=0,
                    ).clamp(0.0, 1.0)
                    depth_conf_local = self._restore_point_axis(depth_conf_local, objects, points)
                else:
                    depth_conf_local = torch.ones_like(depth_local)
                depth_local = self._aggregate_points(depth_local, point_weights_lat)
                depth_conf_local = self._aggregate_points(depth_conf_local, point_weights_lat)
                depth_latent_tokens = self.depth_proj(
                    torch.cat([depth_local, depth_conf_local], dim=-1)
                )

            object_latent_tokens = self._modality_fuse(
                track_geom_latent_tokens,
                appearance_latent_tokens,
            )

            if object_valid_mask is not None:
                slot_mask = object_valid_mask[:, None, :, None].to(
                    dtype=object_latent_tokens.dtype,
                    device=object_latent_tokens.device,
                )
                object_latent_tokens = object_latent_tokens * slot_mask
                jepa_latent_tokens = jepa_latent_tokens * slot_mask
                latent_latent_tokens = latent_latent_tokens * slot_mask
                geom_latent_tokens = geom_latent_tokens * slot_mask
                track_geom_latent_tokens = track_geom_latent_tokens * slot_mask
                active_track_summary = active_track_summary * slot_mask
                active_box_xyxy = active_box_xyxy * slot_mask
                motion_latent_tokens = motion_latent_tokens * slot_mask
                if depth_latent_tokens is not None:
                    depth_latent_tokens = depth_latent_tokens * slot_mask

            object_tokens = object_latent_tokens.mean(dim=1)
            jepa_tokens = jepa_latent_tokens.mean(dim=1)
            latent_tokens = latent_latent_tokens.mean(dim=1)
            geom_tokens = geom_latent_tokens.mean(dim=1)
            motion_tokens = motion_latent_tokens.mean(dim=1)

            return ObjectTokenOutput(
                object_tokens=object_tokens.to(dtype=return_dtype),
                object_latent_tokens=object_latent_tokens.to(dtype=return_dtype),
                jepa_tokens=jepa_tokens.to(dtype=return_dtype),
                jepa_latent_tokens=jepa_latent_tokens.to(dtype=return_dtype),
                latent_tokens=latent_tokens.to(dtype=return_dtype),
                latent_latent_tokens=latent_latent_tokens.to(dtype=return_dtype),
                geom_tokens=geom_tokens.to(dtype=return_dtype),
                track_geom_latent_tokens=track_geom_latent_tokens.to(dtype=return_dtype),
                vggt_geom_tokens=None if vggt_geom_tokens is None else vggt_geom_tokens.to(dtype=return_dtype),
                depth_latent_tokens=None if depth_latent_tokens is None else depth_latent_tokens.to(dtype=return_dtype),
                world_latent_tokens=None,
                motion_latent_tokens=motion_tokens.to(dtype=return_dtype),
                active_track_summary=active_track_summary.to(dtype=return_dtype),
                active_box_xyxy=active_box_xyxy.to(dtype=return_dtype),
            )


class StructureAblationContextOnlyNoGTBoxWanModule(base0708.ContextOnlyNoGTBoxWanModule):
    def __init__(
        self,
        *args,
        structure_ablation_type: str = "none",
        **kwargs,
    ) -> None:
        ablation = str(structure_ablation_type).strip().lower()
        if ablation not in _STRUCTURE_ABLATIONS:
            raise ValueError(
                f"unsupported structure_ablation_type={structure_ablation_type!r}; "
                f"expected one of {_STRUCTURE_ABLATIONS}"
            )
        self.structure_ablation_type = ablation
        self.disable_jepa = ablation == "wo_jepa"
        self.disable_vggt = ablation == "wo_vggt"

        super().__init__(*args, **kwargs)

        if not self.enable_object_branch or self.object_pooler is None:
            return

        self.object_pooler = StructureAblationObjectTubeProjector.from_existing(
            self.object_pooler,
            disable_jepa=self.disable_jepa,
            disable_vggt=self.disable_vggt,
        )

        if self.disable_jepa:
            self.jepa_adapter = None
            self.jepa_runner = None
        if self.disable_vggt:
            self.vggt_adapter = None
            self.vggt_cache_root = None

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if not self.enable_object_branch:
            return super()._compute_object_losses(pipe, inputs_shared, inputs_posi)

        sample = inputs_shared["raw_sample"]
        num_context_frames = int(sample.get("num_context_frames", 0))
        context_frame_indices = sample.get("context_frame_indices", None)
        if isinstance(context_frame_indices, torch.Tensor) and int(context_frame_indices.numel()) > 0:
            sampled_ctx_last_index = float(context_frame_indices.max().item())
        else:
            sampled_ctx_last_index = -1.0
        ctx_max_length = float(sample.get("ctx_max_length", -1))
        if num_context_frames <= 0:
            object_context = torch.zeros(
                (1, int(self.aux_max_objects), int(self.object_adapter.dim)),
                device=pipe.device,
                dtype=pipe.torch_dtype,
            )
            if self.lambda_main > 0.0:
                loss_main = base0708.flow_match_context_sft_loss(
                    pipe,
                    **inputs_shared,
                    **inputs_posi,
                    object_context=object_context,
                )
            else:
                loss_main = object_context.new_zeros(())
            object_context_reg = object_context.new_zeros(())
            total = (
                self.lambda_main * loss_main
                + self.lambda_object_context_reg * object_context_reg
            )
            metrics = {
                "train/loss_total": float(total.detach().item()),
                "train/loss_main": float(loss_main.detach().item()),
                "train/loss_object_context_reg": float(object_context_reg.detach().item()),
                "train/object_count": 0.0,
                "train/object_latent_tokens_abs_max": 0.0,
                "train/object_context_abs_max": 0.0,
                "train/object_context_abs_mean": 0.0,
                "train/jepa_input_frames": 0.0,
                "train/jepa_padding_frames": 0.0,
                "train/ctx_max_length": ctx_max_length,
                "train/sampled_ctx_last_index": sampled_ctx_last_index,
                "train/sampled_ctx_num_frames": 0.0,
            }
            return total, metrics

        context_video = sample["context_video"].unsqueeze(0).to(
            device=pipe.device, dtype=pipe.torch_dtype
        )
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

        query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = (
            self._build_object_query_priors(sample, image_hw=image_hw)
        )
        query_points_prior = query_points_prior.to(device=pipe.device, dtype=pipe.torch_dtype)
        query_frame_ids = query_frame_ids.to(device=pipe.device, dtype=pipe.torch_dtype)
        object_valid_mask = object_valid_mask.to(device=pipe.device, dtype=pipe.torch_dtype)
        box_prior_xyxy = box_prior_xyxy.to(device=pipe.device, dtype=pipe.torch_dtype)

        frames_bthwc_01 = (
            (context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        cotracker_out = self._run_cotracker(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )

        vggt_out = None
        if not self.disable_vggt:
            if self.vggt_cache_root:
                vggt_out = base0708.load_vggt_cache(sample, self.vggt_cache_root, allow_missing=False)
                if vggt_out is None:
                    raise RuntimeError(
                        "VGGT cache root is set but no cache found for sample "
                        f"{sample.get('video_path', '<unknown>')}"
                    )
            else:
                vggt_out = self._run_vggt(
                    frames_bthwc_01,
                    query_points_prior=query_points_prior,
                    query_image_hw=image_hw,
                )

        tracks_grouped, visibility_grouped, confidence_grouped = self._group_tracks_to_objects(
            cotracker_out.tracks,
            cotracker_out.visibility,
            cotracker_out.confidence,
            max_objects=self.aux_max_objects,
            points_per_object=self.object_num_queries,
        )
        context_latents = inputs_shared["clean_prefix_latents"]

        jepa_ctx_fix = {
            "jepa_context_frames": 0,
            "padded_context_frames": 0,
        }
        jepa_patch_tokens = None
        if not self.disable_jepa:
            jepa_input_video, jepa_ctx_fix = base0708.prepare_jepa_context_video(
                context_video,
                latent_frames=int(context_latents.shape[2]),
                tubelet_size=int(self._jepa_tubelet_size),
            )
            jepa_out = self._run_jepa(jepa_input_video)
            jepa_patch_tokens = jepa_out.patch_tokens

        object_out = self.object_pooler(
            jepa_patch_tokens=jepa_patch_tokens,
            context_latents=context_latents,
            tracks=tracks_grouped,
            visibility=visibility_grouped,
            confidence=confidence_grouped,
            track_image_hw=image_hw,
            object_valid_mask=object_valid_mask,
            box_prior_xyxy=box_prior_xyxy,
            vggt_world_points=getattr(vggt_out, "world_points", None),
            vggt_world_points_conf=getattr(vggt_out, "world_points_conf", None),
            vggt_depth=getattr(vggt_out, "depth", None),
            vggt_depth_conf=getattr(vggt_out, "depth_conf", None),
            vggt_dense_patch_tokens=getattr(vggt_out, "dense_patch_tokens", None),
            vggt_patch_grid_hw=getattr(vggt_out, "patch_grid_hw", None),
            vggt_geometry_image_hw=getattr(vggt_out, "input_hw", None)
            if getattr(vggt_out, "input_hw", None) is not None
            else getattr(vggt_out, "image_hw", None),
            frame_valid_mask=None,
        )
        object_context = self.object_adapter(
            object_out.object_latent_tokens,
            object_valid_mask=object_valid_mask,
        )

        if self.lambda_main > 0.0:
            loss_main = base0708.flow_match_context_sft_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
                object_context=object_context,
            )
        else:
            loss_main = object_context.new_zeros(())
        object_context_reg = object_context.square().mean()

        total = (
            self.lambda_main * loss_main
            + self.lambda_object_context_reg * object_context_reg
        )

        object_context_abs = object_context.detach().abs()
        object_latent_tokens_abs = object_out.object_latent_tokens.detach().abs()
        metrics = {
            "train/loss_total": float(total.detach().item()),
            "train/loss_main": float(loss_main.detach().item()),
            "train/loss_object_context_reg": float(object_context_reg.detach().item()),
            "train/object_count": float(object_valid_mask.sum().item()),
            "train/object_latent_tokens_abs_max": float(object_latent_tokens_abs.max().item()),
            "train/object_context_abs_max": float(object_context_abs.max().item()),
            "train/object_context_abs_mean": float(object_context_abs.mean().item()),
            "train/jepa_input_frames": float(jepa_ctx_fix["jepa_context_frames"]),
            "train/jepa_padding_frames": float(jepa_ctx_fix["padded_context_frames"]),
            "train/ctx_max_length": ctx_max_length,
            "train/sampled_ctx_last_index": sampled_ctx_last_index,
            "train/sampled_ctx_num_frames": float(num_context_frames),
        }
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = base0708.build_parser()
    group = parser.add_argument_group("structure_ablation0708")
    group.add_argument(
        "--structure_ablation_type",
        default="none",
        choices=_STRUCTURE_ABLATIONS,
        help="Structure ablation type: none / wo_jepa / wo_vggt.",
    )
    return parser


def build_model(args: argparse.Namespace, accelerator) -> StructureAblationContextOnlyNoGTBoxWanModule:
    grounding_config = base0708._grounding_config_from_args(args)
    grounding_config["grounding_device"] = args.grounding_device or str(accelerator.device)
    return StructureAblationContextOnlyNoGTBoxWanModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_sampling_profile=args.context_sampling_profile,
        min_context_frames=args.min_context_frames,
        max_context_ratio=args.max_context_ratio,
        context_frame_choices=args.context_frame_choices,
        context_length_sampling=args.context_length_sampling,
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
        ctx_max_length=args.ctx_max_length,
        enable_object_branch=args.enable_object_branch,
        object_num_queries=args.object_num_queries,
        aux_max_objects=args.aux_max_objects,
        jepa_ckpt_path=args.jepa_ckpt_path,
        jepa_input_size=args.jepa_input_size,
        jepa_patch_size=args.jepa_patch_size,
        jepa_tubelet_size=args.jepa_tubelet_size,
        cotracker_checkpoint=args.cotracker_checkpoint,
        cotracker_input_h=args.cotracker_input_h,
        cotracker_input_w=args.cotracker_input_w,
        cotracker_window_len=args.cotracker_window_len,
        vggt_model_path=args.vggt_model_path,
        vggt_input_h=args.vggt_input_h,
        vggt_input_w=args.vggt_input_w,
        vggt_cache_root=args.vggt_cache_root,
        object_aux_devices=args.object_aux_devices,
        train_vggt=args.train_vggt,
        object_pooler_latent_dim=args.object_pooler_latent_dim,
        cond_proj_dim=args.cond_proj_dim,
        jepa_window_radius=args.jepa_window_radius,
        latent_window_radius=args.latent_window_radius,
        object_track_delta_scale=args.object_track_delta_scale,
        object_track_gate_init=args.object_track_gate_init,
        object_box_delta_scale=args.object_box_delta_scale,
        object_box_wh_log_scale=args.object_box_wh_log_scale,
        object_box_wh_max_scale=args.object_box_wh_max_scale,
        object_min_box_px=args.object_min_box_px,
        object_gate_init=args.object_gate_init,
        lambda_main=args.lambda_main,
        lambda_track_aux=args.lambda_track_aux,
        lambda_box_aux=args.lambda_box_aux,
        lambda_depth_aux=args.lambda_depth_aux,
        lambda_track_box_aux=args.lambda_track_box_aux,
        lambda_track_iou_aux=args.lambda_track_iou_aux,
        lambda_track_anchor_reg=args.lambda_track_anchor_reg,
        lambda_box_anchor_reg=args.lambda_box_anchor_reg,
        lambda_object_context_reg=args.lambda_object_context_reg,
        train_object_pooler=args.train_object_pooler,
        train_object_aux_heads=args.train_object_aux_heads,
        train_object_adapter=args.train_object_adapter,
        train_object_dit_branch=args.train_object_dit_branch,
        freeze_non_object_trainables=args.freeze_non_object_trainables,
        depth_target_state_index=args.depth_target_state_index,
        depth_target_source=args.depth_target_source,
        depth_anything_cache_root=args.depth_anything_cache_root,
        grounding_config=grounding_config,
        structure_ablation_type=args.structure_ablation_type,
    )


def _log_structure_ablation_summary(accelerator, model, args: argparse.Namespace) -> None:
    if not accelerator.is_main_process:
        return
    lines = [
        "=" * 78,
        "structure_ablation0708 switches",
        "=" * 78,
        f"  - structure_ablation_type: {args.structure_ablation_type}",
        f"  - stage1a_init_from: {args.stage1a_init_from}",
        f"  - JEPA present: {model.jepa_adapter is not None or model.jepa_runner is not None}",
        f"  - VGGT present: {model.vggt_adapter is not None or bool(model.vggt_cache_root)}",
        "=" * 78,
    ]
    accelerator.print("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = tvn.prepare_args(parser.parse_args())
    previous_handlers = tvn.install_interrupt_handlers()

    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    if args.stage2_resume_from is not None and accelerator.is_main_process:
        accelerator.print(
            f"👉 Resuming stage2 training from state {args.stage2_resume_from} "
            f"(base LoRA stays loaded from --lora_checkpoint)."
        )

    dataset = base0708.build_dataset(args)
    headonly_val_config = base0708.build_headonly_val_config(args)
    headonly_val_dataset = base0708.build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = base0708.build_headonly_val_dataloader(headonly_val_dataset, args)

    model = build_model(args, accelerator)

    if args.stage1a_init_from is not None:
        init_info = tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded Stage1A token builder init: "
                f"selected_source_keys={init_info['selected_source_keys']}, "
                f"loaded_count={init_info['loaded_count']}, "
                f"shape_mismatch={len(init_info['skipped_shape_mismatch'])}"
            )

    if args.stage2_resume_from is not None:
        resume_info = tvn._load_filtered_checkpoint_into_model(
            model,
            tvn.resolve_lora_checkpoint_for_resume(args.stage2_resume_from),
            include_prefixes=("object_adapter.",),
            include_substrings=(
                "object_embedding",
                ".object_cross_attn.",
                ".object_gate",
                ".norm4.",
            ),
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded stage2 trainable initialization: "
                f"selected_source_keys={resume_info['selected_source_keys']}, "
                f"loaded_count={resume_info['loaded_count']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    _log_structure_ablation_summary(accelerator, model, args)
    base0708._log_stage_summary(accelerator, model, args)

    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict = {}

    try:
        if args.task in ("sft:data_process", "direct_distill:data_process"):
            tvn.launch_data_process_task(accelerator, dataset, model, model_logger, args=args)
        else:
            tvn.train_loop(
                accelerator,
                dataset,
                model,
                model_logger,
                args,
                runtime_state=runtime_state,
                headonly_val_dataloader=headonly_val_dataloader,
                headonly_val_config=headonly_val_config,
            )
    except (KeyboardInterrupt, tvn.TrainingInterrupted) as exc:
        interrupted_checkpoint_path = tvn.training_checkpoint_file(
            tvn.get_checkpoint_dir(args), "interrupted-latest"
        )
        accelerator.print(
            f"Training interrupted at step {model_logger.num_steps}. Saving interrupt checkpoint."
        )
        model_logger.save_model(accelerator, model, interrupted_checkpoint_path)
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get(
            "progress",
            {"global_step": 0, "epoch_id": 0, "batch_in_epoch": 0},
        )
        if optimizer is not None and scheduler is not None:
            tvn.save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=progress.get("global_step", 0),
                epoch_id=progress.get("epoch_id", 0),
                batch_in_epoch=progress.get("batch_in_epoch", 0),
                model_logger=model_logger,
                state_path=tvn.training_state_file(
                    tvn.get_checkpoint_dir(args), "interrupted-latest"
                ),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise exc

    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
