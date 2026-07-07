from __future__ import annotations

import torch

from code_vjepa_vggt.models.object_tokens import ObjectTokenOutput, ObjectTubeProjector


VALID_STAGE1AB_ABLATIONS = ("wo_jepa", "wo_vggt")


def copy_requires_grad_by_name(source: torch.nn.Module, target: torch.nn.Module) -> None:
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
        copy_requires_grad_by_name(source, replacement)
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
