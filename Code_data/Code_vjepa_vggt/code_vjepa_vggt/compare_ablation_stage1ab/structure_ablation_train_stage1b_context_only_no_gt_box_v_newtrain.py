"""Structure-ablation training entry for train0705 Stage1B context-only no-GT-box.

This file is separate from both:
  - the original formal training entry, and
  - the earlier signal-style ablation entry.

In `compare_ablation_stage1ab` this Stage1B copy is intended to consume a
matching ablated Stage1A checkpoint, so the resulting `w/o JEPA` / `w/o VGGT`
experiments are full-pipeline Stage1A+1B removals instead of Stage1B-only
injection ablations.

The goal here is *structural* ablation:
  - w/o CoTracker: remove the CoTracker module and the motion branch from the
    object token builder. Static viewer-grounding query priors are used only as
    slot anchors for local pooling.
  - w/o JEPA: remove the JEPA module and the JEPA appearance branch.
  - w/o VGGT: remove the VGGT module and the VGGT geometry/depth branch.
  - No Stage1A init: same structure as baseline, but skip Stage1A checkpoint
    initialization by not passing ``--stage1a_init_from``.
"""
from __future__ import annotations

import argparse

import torch

import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.models.object_tokens import ObjectTokenOutput, ObjectTubeProjector
from code_vjepa_vggt.train0705 import train_stage1b_context_only_no_gt_box_v_newtrain as base0705
from diffsynth.diffusion import ModelLogger


_STRUCTURE_ABLATIONS = ("none", "wo_cotracker", "wo_jepa", "wo_vggt")


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
        disable_cotracker: bool = False,
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
        self.disable_cotracker = bool(disable_cotracker)
        self.disable_jepa = bool(disable_jepa)
        self.disable_vggt = bool(disable_vggt)

        if self.disable_jepa:
            del self.jepa_proj
            del self.jepa_router_score
        if self.disable_cotracker:
            del self.motion_point_proj
            del self.motion_router_score
        if self.disable_vggt:
            del self.vggt_geom_point_proj
            del self.depth_proj

    @classmethod
    def from_existing(
        cls,
        source: ObjectTubeProjector,
        *,
        disable_cotracker: bool,
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
            disable_cotracker=disable_cotracker,
            disable_jepa=disable_jepa,
            disable_vggt=disable_vggt,
        ).to(device=device, dtype=dtype)
        replacement.load_state_dict(source.state_dict(), strict=False)
        replacement.train(source.training)
        _copy_requires_grad_by_name(source, replacement)
        return replacement

    @staticmethod
    def _expand_box_prior_over_time(
        box_prior_xyxy: torch.Tensor | None,
        *,
        target_frames: int,
    ) -> torch.Tensor:
        if box_prior_xyxy is None:
            raise ValueError("box_prior_xyxy is required for structure ablation without CoTracker.")
        if box_prior_xyxy.ndim == 3:
            return box_prior_xyxy[:, None].expand(-1, int(target_frames), -1, -1)
        if box_prior_xyxy.ndim == 4:
            if int(box_prior_xyxy.shape[1]) == int(target_frames):
                return box_prior_xyxy
            if int(box_prior_xyxy.shape[1]) == 1:
                return box_prior_xyxy.expand(-1, int(target_frames), -1, -1)
        raise ValueError(
            "box_prior_xyxy must have shape [B,O,4], [B,1,O,4], or [B,T,O,4], "
            f"got {list(box_prior_xyxy.shape)}"
        )

    @staticmethod
    def _box_summary_from_boxes(
        boxes_xyxy: torch.Tensor,
        *,
        object_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        center_x = 0.5 * (boxes_xyxy[..., 0] + boxes_xyxy[..., 2])
        center_y = 0.5 * (boxes_xyxy[..., 1] + boxes_xyxy[..., 3])
        center_xy = torch.stack([center_x, center_y], dim=-1)
        delta_xy = torch.zeros_like(center_xy)
        if object_valid_mask is None:
            valid = torch.ones(
                boxes_xyxy.shape[0],
                boxes_xyxy.shape[1],
                boxes_xyxy.shape[2],
                1,
                device=boxes_xyxy.device,
                dtype=boxes_xyxy.dtype,
            )
        else:
            valid = object_valid_mask
            if valid.ndim == 2:
                valid = valid[:, None].expand(-1, boxes_xyxy.shape[1], -1)
            valid = valid.unsqueeze(-1).to(device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)
        return torch.cat([center_xy, delta_xy, valid, valid], dim=-1)

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
        return_dtype = (
            jepa_patch_tokens.dtype if jepa_patch_tokens is not None else context_latents.dtype
        )
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
            if vggt_world_points is not None:
                vggt_world_points = torch.nan_to_num(
                    vggt_world_points.float(), nan=0.0, posinf=0.0, neginf=0.0
                ).to(device=feature_device)
            if vggt_world_points_conf is not None:
                vggt_world_points_conf = torch.nan_to_num(
                    vggt_world_points_conf.float(), nan=0.0, posinf=0.0, neginf=0.0
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
            if int(src_frames) % max(latent_frames, 1) != 0:
                raise ValueError(
                    f"track frames ({src_frames}) must be divisible by latent frames ({latent_frames}) "
                    "for latent-time conditioning"
                )

            point_weights_lat = self._point_weights(
                visibility,
                confidence,
                target_frames=latent_frames,
                frame_valid_mask=frame_valid_mask,
            )

            latent_time_idx = self._time_indices(src_frames, latent_frames, tracks.device)
            latent_tracks = tracks[:, latent_time_idx]
            flat_latent_tracks, _, _ = self._flatten_point_axis(latent_tracks)
            latent_grid = context_latents.permute(0, 2, 3, 4, 1).contiguous()
            latent_local = self._pool_feature_grid(
                latent_grid,
                flat_latent_tracks,
                image_hw=track_image_hw,
                window_radius=self.latent_window_radius,
            )
            latent_local = self._restore_point_axis(latent_local, objects, points)
            latent_local = self._aggregate_points(latent_local, point_weights_lat)
            self._ensure_latent_proj(int(latent_local.shape[-1]), latent_local.device)
            latent_latent_tokens = self.latent_proj(latent_local)

            if self.disable_jepa:
                jepa_latent_tokens = torch.zeros_like(latent_latent_tokens)
                appearance_latent_tokens = latent_latent_tokens
            else:
                if jepa_patch_tokens is None:
                    raise ValueError("jepa_patch_tokens is required unless JEPA is structurally removed.")
                jepa_time_idx = self._time_indices(src_frames, int(jepa_patch_tokens.shape[1]), tracks.device)
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
                jepa_local = self._temporal_group_mean(jepa_local, latent_frames, frame_valid_mask=jepa_valid)
                jepa_local = self._restore_point_axis(jepa_local, objects, points)
                jepa_local = self._aggregate_points(jepa_local, point_weights_lat)
                expected_jepa_dim = int(self.jepa_proj.in_features)
                if int(jepa_local.shape[-1]) != expected_jepa_dim:
                    actual_jepa_dim = int(jepa_local.shape[-1])
                    if actual_jepa_dim % expected_jepa_dim == 0:
                        fold = actual_jepa_dim // expected_jepa_dim
                        jepa_local = jepa_local.reshape(
                            *jepa_local.shape[:-1], fold, expected_jepa_dim
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

            if self.disable_cotracker:
                active_box_xyxy = self._expand_box_prior_over_time(
                    box_prior_xyxy,
                    target_frames=latent_frames,
                ).float()
                active_track_summary = self._box_summary_from_boxes(
                    active_box_xyxy,
                    object_valid_mask=object_valid_mask,
                ).float()
                box_geom_latent_tokens = self.track_geom_proj(active_track_summary)
                motion_latent_tokens = None
            else:
                motion_xy = tracks[:, latent_time_idx]
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
                        visibility[:, latent_time_idx].unsqueeze(-1),
                        confidence[:, latent_time_idx].unsqueeze(-1),
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
                    tracks,
                    visibility,
                    confidence,
                )
                active_track_summary = self._track_summary(
                    center_tracks,
                    center_track_valid.to(dtype=center_tracks.dtype),
                    center_track_valid.to(dtype=center_tracks.dtype),
                    image_hw=track_image_hw,
                    target_frames=latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                active_box_xyxy = self._boxes_from_tracks(
                    tracks,
                    visibility,
                    confidence,
                    image_hw=track_image_hw,
                    target_frames=latent_frames,
                    box_prior_xyxy=box_prior_xyxy,
                    min_box_px=self.min_box_px,
                )
                box_geom_latent_tokens = self.track_geom_proj(active_track_summary)

            vggt_geom_latent_tokens = None
            vggt_geom_tokens = None
            geom_latent_tokens = box_geom_latent_tokens
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
                patch_tracks = self._resize_tracks_xy(
                    geometry_tracks.reshape(batch, src_frames, objects * points, 2),
                    src_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                    dst_hw=geometry_patch_hw,
                    align_corners=False,
                ).view(batch, src_frames, objects, points, 2)
                flat_patch_tracks, _, _ = self._flatten_point_axis(patch_tracks)
                geom_local = self._pool_feature_grid(
                    vggt_dense_patch_tokens,
                    flat_patch_tracks,
                    image_hw=geometry_patch_hw,
                    window_radius=0,
                )
                geom_local = self._restore_point_axis(geom_local, objects, points)
                geom_local = self._temporal_group_mean_grouped(
                    geom_local,
                    latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                if vggt_depth is not None:
                    flat_geometry_tracks, _, _ = self._flatten_point_axis(geometry_tracks)
                    depth_local = self._pool_feature_grid(
                        vggt_depth,
                        flat_geometry_tracks,
                        image_hw=vggt_geometry_image_hw if vggt_geometry_image_hw is not None else track_image_hw,
                        window_radius=0,
                    ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                    depth_local = self._restore_point_axis(depth_local, objects, points)
                    depth_local = self._temporal_group_mean_grouped(
                        depth_local,
                        latent_frames,
                        frame_valid_mask=frame_valid_mask,
                    )
                else:
                    depth_local = geom_local.new_zeros(*geom_local.shape[:-1], 1)

                if self.disable_cotracker:
                    motion_local_lat = geom_local.new_zeros(
                        geom_local.shape[0],
                        geom_local.shape[1],
                        geom_local.shape[2],
                        geom_local.shape[3],
                        6,
                    )
                else:
                    motion_local_lat = motion_local
                    if int(motion_local_lat.shape[1]) != int(latent_frames):
                        motion_local_lat = self._temporal_group_mean_grouped(
                            motion_local_lat,
                            latent_frames,
                            frame_valid_mask=frame_valid_mask,
                        )

                geom_point_features = torch.cat(
                    [geom_local, depth_local, motion_local_lat],
                    dim=-1,
                )
                geom_point_tokens = self.vggt_geom_point_proj(geom_point_features)
                vggt_geom_latent_tokens = self._point_attention_pool(
                    geom_point_tokens,
                    self.geom_router_score,
                    point_weights_lat,
                )
                vggt_geom_tokens = vggt_geom_latent_tokens.mean(dim=1)
                geom_latent_tokens = vggt_geom_latent_tokens

            if self.disable_cotracker:
                if vggt_geom_latent_tokens is not None:
                    track_geom_latent_tokens = self._pair_fuse(
                        box_geom_latent_tokens,
                        vggt_geom_latent_tokens,
                        self.track_geometry_router_score,
                        self.geom_router_score,
                    )
                else:
                    track_geom_latent_tokens = box_geom_latent_tokens
                    geom_latent_tokens = box_geom_latent_tokens
            else:
                track_geom_latent_tokens = self._pair_fuse(
                    motion_latent_tokens,
                    geom_latent_tokens,
                    self.motion_router_score,
                    self.geom_router_score,
                )

            depth_latent_tokens = None
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
                ).view(batch, src_frames, objects, points, 2)
                flat_geometry_tracks, _, _ = self._flatten_point_axis(geometry_tracks)
                depth_local = self._pool_feature_grid(
                    vggt_depth,
                    flat_geometry_tracks,
                    image_hw=geometry_image_hw,
                    window_radius=0,
                ).clamp(-self.vggt_depth_clip, self.vggt_depth_clip)
                depth_local = self._restore_point_axis(depth_local, objects, points)
                if vggt_depth_conf is not None:
                    depth_conf_local = self._pool_feature_grid(
                        vggt_depth_conf.unsqueeze(-1),
                        flat_geometry_tracks,
                        image_hw=geometry_image_hw,
                        window_radius=0,
                    ).clamp(0.0, 1.0)
                    depth_conf_local = self._restore_point_axis(depth_conf_local, objects, points)
                else:
                    depth_conf_local = torch.ones_like(depth_local)
                depth_local = self._temporal_group_mean_grouped(
                    depth_local,
                    latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                depth_conf_local = self._temporal_group_mean_grouped(
                    depth_conf_local,
                    latent_frames,
                    frame_valid_mask=frame_valid_mask,
                )
                depth_local = self._aggregate_points(depth_local, point_weights_lat)
                depth_conf_local = self._aggregate_points(depth_conf_local, point_weights_lat)
                depth_latent_tokens = self.depth_proj(torch.cat([depth_local, depth_conf_local], dim=-1))

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
                if motion_latent_tokens is not None:
                    motion_latent_tokens = motion_latent_tokens * slot_mask
                if depth_latent_tokens is not None:
                    depth_latent_tokens = depth_latent_tokens * slot_mask

            object_tokens = object_latent_tokens.mean(dim=1)
            jepa_tokens = jepa_latent_tokens.mean(dim=1)
            latent_tokens = latent_latent_tokens.mean(dim=1)
            geom_tokens = geom_latent_tokens.mean(dim=1)
            motion_tokens = None if motion_latent_tokens is None else motion_latent_tokens.mean(dim=1)

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
                motion_latent_tokens=None if motion_tokens is None else motion_tokens.to(dtype=return_dtype),
                active_track_summary=active_track_summary.to(dtype=return_dtype),
                active_box_xyxy=active_box_xyxy.to(dtype=return_dtype),
            )


class StructureAblationContextOnlyNoGTBoxWanModule(base0705.ContextOnlyNoGTBoxWanModule):
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
        self.disable_cotracker = ablation == "wo_cotracker"
        self.disable_jepa = ablation == "wo_jepa"
        self.disable_vggt = ablation == "wo_vggt"

        super().__init__(*args, **kwargs)

        if not self.enable_object_branch or self.object_pooler is None:
            return

        self.object_pooler = StructureAblationObjectTubeProjector.from_existing(
            self.object_pooler,
            disable_cotracker=self.disable_cotracker,
            disable_jepa=self.disable_jepa,
            disable_vggt=self.disable_vggt,
        )

        if self.disable_cotracker:
            self.cotracker_adapter = None
            self.cotracker_runner = None
        if self.disable_jepa:
            self.jepa_adapter = None
            self.jepa_runner = None
        if self.disable_vggt:
            self.vggt_adapter = None
            self.vggt_cache_root = None

    def _build_static_anchor_tracks(
        self,
        query_points_prior: torch.Tensor,
        *,
        src_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(query_points_prior.shape[0])
        grouped = query_points_prior.view(
            batch_size,
            int(self.aux_max_objects),
            int(self.object_num_queries),
            2,
        )
        tracks = grouped.unsqueeze(1).expand(-1, int(src_frames), -1, -1, -1).contiguous()
        visibility = torch.ones(
            batch_size,
            int(src_frames),
            int(self.aux_max_objects),
            int(self.object_num_queries),
            device=query_points_prior.device,
            dtype=query_points_prior.dtype,
        )
        confidence = torch.ones_like(visibility)
        return tracks, visibility, confidence

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if not self.enable_object_branch:
            return super()._compute_object_losses(pipe, inputs_shared, inputs_posi)

        sample = inputs_shared["raw_sample"]
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

        if self.disable_cotracker:
            tracks_grouped, visibility_grouped, confidence_grouped = self._build_static_anchor_tracks(
                query_points_prior,
                src_frames=int(frames_bthwc_01.shape[1]),
            )
        else:
            cotracker_out = self._run_cotracker(
                frames_bthwc_01,
                query_points_prior=query_points_prior,
                query_frame_ids=query_frame_ids,
                query_image_hw=image_hw,
            )
            tracks_grouped, visibility_grouped, confidence_grouped = self._group_tracks_to_objects(
                cotracker_out.tracks,
                cotracker_out.visibility,
                cotracker_out.confidence,
                max_objects=self.aux_max_objects,
                points_per_object=self.object_num_queries,
            )

        vggt_out = None
        if not self.disable_vggt:
            if self.vggt_cache_root:
                vggt_out = base0705.load_vggt_cache(sample, self.vggt_cache_root, allow_missing=False)
                if vggt_out is None:
                    raise RuntimeError(
                        "VGGT cache root is set but no cache found for sample "
                        f"{sample.get('video_path', '<unknown>')}"
                    )
            else:
                if self.vggt_adapter is None:
                    raise RuntimeError("VGGT adapter is not initialized while VGGT ablation is disabled.")
                vggt_out = self.vggt_adapter(
                    frames_bthwc_01,
                    query_points_prior=query_points_prior,
                    query_image_hw=image_hw,
                )

        jepa_patch_tokens = None
        if not self.disable_jepa:
            jepa_out = self._run_jepa(context_video)
            jepa_patch_tokens = jepa_out.patch_tokens

        context_latents = inputs_shared["clean_prefix_latents"]
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
            loss_main = base0705.flow_match_context_sft_loss(
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
        }
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = base0705.build_parser()
    group = parser.add_argument_group("structure_ablation")
    group.add_argument(
        "--structure_ablation_type",
        default="none",
        choices=_STRUCTURE_ABLATIONS,
        help="Structure ablation type: none / wo_cotracker / wo_jepa / wo_vggt.",
    )
    return parser


def build_model(args: argparse.Namespace, accelerator) -> StructureAblationContextOnlyNoGTBoxWanModule:
    grounding_config = base0705._grounding_config_from_args(args)
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
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
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
        "structure_ablation switches",
        "=" * 78,
        f"  - structure_ablation_type: {args.structure_ablation_type}",
        f"  - stage1a_init_from: {args.stage1a_init_from}",
        f"  - JEPA present: {model.jepa_adapter is not None or model.jepa_runner is not None}",
        f"  - CoTracker present: {model.cotracker_adapter is not None or model.cotracker_runner is not None}",
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

    dataset = tvn.build_dataset(args)
    headonly_val_config = tvn.build_headonly_val_config(args)
    headonly_val_dataset = tvn.build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = tvn.build_headonly_val_dataloader(headonly_val_dataset, args)

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
    base0705._log_stage_summary(accelerator, model, args)

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
