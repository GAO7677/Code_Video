from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.adapters.jepa_adapter import JEPAPatchAdapter
from code_vjepa_vggt.adapters.sam2_motion import (
    GroundingDINOTextDetector,
    SAM2MotionTracker,
    build_motion_prompt_box,
)
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data import BallBlockVideoDataset, PhysStateEpisodeDataset
from code_vjepa_vggt.models.context_fuser import ContextTokenFuser
from code_vjepa_vggt.models.object_tokens import ObjectTubeProjector
from code_vjepa_vggt.models.wan_context_model import WanContextVideoModel
from code_vjepa_vggt.training.flow_match import WanFlowMatchScheduler
from code_vjepa_vggt.utils.masks import (
    broadcast_latent_mask,
    collate_video_batch,
    expand_context_latents_to_full,
    latent_frame_mask,
)
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior
from code_vjepa_vggt.utils.track_supervision import (
    align_tracks_to_boxes,
    track_box_iou_loss,
    track_box_l1_loss,
)


def _build_multi_object_prompt(caption: str) -> str:
    caption_lower = str(caption).lower()
    ordered = ["sphere", "ball", "block", "box", "cube", "cylinder", "capsule"]
    found = []
    for token in ordered:
        if token in caption_lower and token not in found:
            found.append(token)
    if not found:
        return str(caption)
    return " . ".join(found) + " ."


@dataclass
class TrainerState:
    step: int = 0


class ContextVideoTrainer(nn.Module):
    def __init__(self, cfg: dict[str, Any], build_optimizer: bool = True, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        if device is not None:
            self.device_obj = torch.device(device)
        elif torch.cuda.is_available():
            self.device_obj = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device_obj = torch.device("cpu")
        self.build_optimizer = build_optimizer

        model_cfg = cfg["model"]
        data_cfg = cfg["data"]

        self.bundle = WanContextVideoModel(
            ckpt_dir=model_cfg["wan_ckpt_dir"],
            task=model_cfg["wan_task"],
            device=str(self.device_obj),
            load_dit=build_optimizer,
            lora_rank=int(model_cfg.get("wan_lora_rank", 0)),
            lora_alpha=int(model_cfg.get("wan_lora_alpha", 0)),
            lora_dropout=float(model_cfg.get("wan_lora_dropout", 0.0)),
            lora_init=str(model_cfg.get("wan_lora_init", "gaussian")),
        )
        self.bundle.freeze_parts(
            freeze_vae=bool(model_cfg["freeze_vae"]),
            freeze_text_encoder=bool(model_cfg["freeze_text_encoder"]),
            freeze_dit=bool(model_cfg["freeze_wan_dit"]),
        )
        if self.bundle.dit is not None:
            self.bundle.dit.train(mode=build_optimizer)

        cond_dim = int(model_cfg.get("cond_proj_dim", self.bundle.config.text_dim if hasattr(self.bundle.config, "text_dim") else 4096))
        if hasattr(self.bundle.config, "text_dim") and cond_dim != int(self.bundle.config.text_dim):
            raise ValueError(f"cond_proj_dim must match Wan text_dim={self.bundle.config.text_dim}, got {cond_dim}")

        self.jepa_adapter = JEPAPatchAdapter(
            ckpt_path=str(Path(model_cfg["je_pa_ckpt_dir"]) / "original" / "model.pth"),
            device=str(self.device_obj),
            crop_size=int(model_cfg["jepa_input_size"]),
            num_frames=int(data_cfg["num_context_frames"]),
            patch_size=int(model_cfg["jepa_patch_size"]),
            tubelet_size=int(model_cfg["jepa_tubelet_size"]),
        ).to(self.device_obj)
        self.vggt_adapter = VGGTTrackAdapter(
            model_path=model_cfg.get("vggt_model_path"),
            num_queries=int(model_cfg["object_num_queries"]),
            device=str(self.device_obj),
            input_hw=tuple(model_cfg["vggt_input_hw"]),
        ).to(self.device_obj)
        self.track_source = str(model_cfg.get("track_source", "vggt")).strip().lower()
        if self.track_source not in {"vggt", "cotracker"}:
            raise ValueError(f"unsupported track_source: {self.track_source}")
        self.cotracker_adapter = None
        if self.track_source == "cotracker":
            self.cotracker_adapter = CoTrackerAdapter(
                checkpoint_path=model_cfg.get("cotracker_checkpoint"),
                num_queries=int(model_cfg["object_num_queries"]),
                device=str(self.device_obj),
                input_hw=tuple(model_cfg.get("cotracker_input_hw", [384, 512])),
                window_len=int(model_cfg.get("cotracker_window_len", 60)),
            ).to(self.device_obj)
        latent_dim = int(getattr(self.bundle.config, "in_dim", 16))
        self.object_pooler = ObjectTubeProjector(
            jepa_dim=self.jepa_adapter.encoder.backbone.embed_dim,
            latent_dim=latent_dim,
            out_dim=cond_dim,
            jepa_window_radius=int(model_cfg["jepa_window_radius"]),
            latent_window_radius=int(model_cfg["latent_window_radius"]),
        ).to(self.device_obj)
        self.context_fuser = ContextTokenFuser(
            text_dim=cond_dim,
            max_context_len=self.bundle.config.text_len,
            min_text_tokens=int(model_cfg.get("min_text_tokens", 64)),
        ).to(self.device_obj)
        self.scheduler = WanFlowMatchScheduler(num_train_timesteps=int(self.bundle.config.num_train_timesteps))

        self.enable_sam2_priors = bool(model_cfg.get("enable_sam2_priors", False))
        self.sam2_prior_strategy = str(model_cfg.get("sam2_prior_strategy", "single")).strip().lower()
        self.sam2_tracker = None
        self.text_detector = None
        if self.enable_sam2_priors:
            self.sam2_tracker = SAM2MotionTracker(
                device=str(self.device_obj),
                segment_len=int(model_cfg.get("sam2_segment_len", 8)),
                enable_text_prompt=bool(model_cfg.get("sam2_enable_text_prompt", True)),
            )
            if self.sam2_prior_strategy in {"grounded_text_multi", "text_multi", "grounded_text"}:
                self.text_detector = GroundingDINOTextDetector(
                    device=str(self.device_obj),
                    max_boxes=int(model_cfg.get("sam2_max_objects", 4)),
                )

        self.dataset = self._build_dataset()
        self.state = TrainerState()

    def _build_dataset(self):
        data_cfg = self.cfg["data"]
        dataset_type = data_cfg.get("dataset_type", "ball_block_json")
        if dataset_type == "ball_block_json":
            return BallBlockVideoDataset(
                root=data_cfg["root"],
                num_frames=data_cfg["num_frames"],
                num_context_frames=data_cfg["num_context_frames"],
                resolution=tuple(data_cfg["resolution"]),
                sampling_mode=str(data_cfg.get("sampling_mode", "uniform")),
            )
        if dataset_type == "phys_state_episode":
            return PhysStateEpisodeDataset(
                root=data_cfg["root"],
                split=data_cfg["split"],
                resolution=tuple(data_cfg["resolution"]),
                num_context_frames=int(data_cfg["num_context_frames"]),
                context_fraction=float(data_cfg.get("context_fraction", 0.5)),
                random_context_frames=bool(data_cfg.get("random_context_frames", True)),
                seed=int(self.cfg.get("experiment", {}).get("seed", 42)),
            )
        raise ValueError(f"unsupported dataset_type: {dataset_type}")

    def build_dataloader(self, num_workers: int | None = None) -> DataLoader:
        data_cfg = self.cfg["data"]
        return DataLoader(
            self.dataset,
            batch_size=int(data_cfg["batch_size"]),
            shuffle=True,
            num_workers=int(data_cfg["num_workers"] if num_workers is None else num_workers),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_video_batch,
        )

    def trainable_parameters(self):
        self.bundle.ensure_dit_loaded()
        params = list(self.bundle.dit.parameters())
        params += list(self.context_fuser.parameters())
        params += list(self.object_pooler.parameters())
        return [param for param in params if param.requires_grad]

    def export_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, param in self.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if name in trainable_names
        }

    def _encode_text(self, captions: list[str]) -> list[torch.Tensor]:
        with torch.no_grad():
            ctx = self.bundle.text_encoder(captions, self.bundle.text_encoder.device)
        return [u.to(self.device_obj) for u in ctx]

    def _encode_video_latents(self, videos_bcthw: torch.Tensor) -> list[torch.Tensor]:
        videos_list = [u.to(self.device_obj) for u in videos_bcthw]
        with torch.no_grad():
            zs = self.bundle.vae.encode(videos_list)
        return zs

    @staticmethod
    def _pad_time_axis(tensor: torch.Tensor, target_t: int, *, fill_mode: str) -> torch.Tensor:
        current_t = int(tensor.shape[1] if tensor.ndim == 4 else tensor.shape[0])
        if current_t >= target_t:
            return tensor
        pad_t = target_t - current_t
        if tensor.ndim == 4:
            if fill_mode == "repeat_last":
                pad = tensor[:, -1:].expand(-1, pad_t, -1, -1).contiguous()
            else:
                pad = tensor.new_zeros(tensor.shape[0], pad_t, tensor.shape[2], tensor.shape[3])
            return torch.cat([tensor, pad], dim=1)
        if fill_mode == "zeros":
            pad = tensor.new_zeros((pad_t,) + tuple(tensor.shape[1:]))
        else:
            pad = tensor[-1:].expand(pad_t, *tensor.shape[1:]).contiguous()
        return torch.cat([tensor, pad], dim=0)

    @staticmethod
    def _shape_list(tensors: list[torch.Tensor]) -> list[list[int]]:
        return [list(t.shape) for t in tensors]

    @staticmethod
    def _frame_valid_mask(max_context_frames: int, num_context_frames: torch.Tensor, device: torch.device) -> torch.Tensor:
        frame_ids = torch.arange(max_context_frames, device=device).view(1, -1)
        return frame_ids < num_context_frames.view(-1, 1)

    def _maybe_build_query_priors(
        self,
        context_videos: torch.Tensor,
        num_context_frames: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor | None, list[str], list[str], list[dict[str, Any]]]:
        if self.sam2_tracker is None:
            return None, [], [], []

        priors = []
        prior_sources: list[str] = []
        prompt_modes: list[str] = []
        prior_debugs: list[dict[str, Any]] = []
        for batch_idx in range(context_videos.shape[0]):
            valid_frames = int(num_context_frames[batch_idx].item())
            frames_tchw_01 = ((context_videos[batch_idx, :, :valid_frames].permute(1, 0, 2, 3).float() + 1.0) / 2.0).detach().cpu().numpy()
            prompt_frame_idx = max(valid_frames - 1, 0)
            try:
                query_points_px, prior_source, prompt_mode, prior_debug = self._build_query_prior_for_sample(
                    frames_tchw_01=frames_tchw_01,
                    prompt_frame_idx=prompt_frame_idx,
                    caption=captions[batch_idx],
                )
                priors.append(torch.from_numpy(query_points_px))
                prior_sources.append(prior_source)
                prompt_modes.append(prompt_mode)
                prior_debugs.append(
                    {
                        "prompt_frame_idx": int(prompt_frame_idx),
                        "valid_frames": int(valid_frames),
                        **prior_debug,
                    }
                )
            except Exception:
                return None, [], [], []
        if not priors:
            return None, [], [], []
        stacked = torch.stack(priors, dim=0).to(device=self.device_obj, dtype=context_videos.dtype)
        return stacked, prior_sources, prompt_modes, prior_debugs

    def _build_query_prior_for_sample(
        self,
        *,
        frames_tchw_01: Any,
        prompt_frame_idx: int,
        caption: str,
    ) -> tuple[Any, str, str, dict[str, Any]]:
        if self.sam2_prior_strategy in {"grounded_text_multi", "text_multi", "grounded_text"}:
            return self._build_multi_object_query_prior(
                frames_tchw_01=frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                caption=caption,
            )

        motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
        sam_out = self.sam2_tracker.track(
            frames_tchw_01,
            prompt_frame_idx=prompt_frame_idx,
            prompt_box_xyxy=motion_prompt_box_xyxy,
            caption=caption,
        )
        query_points_px, prior_source = build_vggt_query_prior(
            sam_out.masks_thw,
            sam_out.boxes_t4,
            num_queries=self.vggt_adapter.num_queries,
        )
        return query_points_px, prior_source, sam_out.prompt_mode, {
            "strategy": self.sam2_prior_strategy,
            "prompt_text": sam_out.prompt_text,
            "object_count": 1,
            "used_fallback": bool("fallback" in sam_out.prompt_mode or sam_out.prompt_mode.startswith("proxy_box")),
            "prior_source": prior_source,
        }

    def _build_multi_object_query_prior(
        self,
        *,
        frames_tchw_01: np.ndarray,
        prompt_frame_idx: int,
        caption: str,
    ) -> tuple[np.ndarray, str, str, dict[str, Any]]:
        max_objects = int(self.cfg["model"].get("sam2_max_objects", 4))
        text_prompt = _build_multi_object_prompt(caption)
        detected_boxes = None
        prompt_mode = "caption_gdino_multi"
        used_fallback = False
        detector_error = ""
        if self.text_detector is not None and text_prompt.strip():
            try:
                detection = self.text_detector.detect(
                    frames_tchw_01[int(prompt_frame_idx)],
                    text_prompt,
                    guidance_box_xyxy=None,
                )
                if detection.boxes_xyxy.shape[0] > 0:
                    detected_boxes = detection.boxes_xyxy[:max_objects]
                    prompt_mode = detection.prompt_mode
            except Exception as exc:
                used_fallback = True
                detector_error = f"{type(exc).__name__}: {exc}"
        if detected_boxes is None or detected_boxes.shape[0] == 0:
            used_fallback = True
            motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
            sam_out = self.sam2_tracker.track(
                frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                prompt_box_xyxy=motion_prompt_box_xyxy,
                caption=caption,
            )
            query_points_px, prior_source = build_vggt_query_prior(
                sam_out.masks_thw,
                sam_out.boxes_t4,
                num_queries=self.vggt_adapter.num_queries,
            )
            return query_points_px, prior_source, sam_out.prompt_mode, {
                "strategy": self.sam2_prior_strategy,
                "prompt_text": text_prompt if text_prompt.strip() else sam_out.prompt_text,
                "object_count": 1,
                "used_fallback": used_fallback,
                "prior_source": prior_source,
                "detector_error": detector_error,
            }

        per_object_queries = []
        object_count = min(int(detected_boxes.shape[0]), int(self.vggt_adapter.num_queries))
        detected_boxes = detected_boxes[:object_count]
        base = self.vggt_adapter.num_queries // max(object_count, 1)
        remainder = max(0, self.vggt_adapter.num_queries - base * object_count)
        for obj_idx, box_xyxy in enumerate(detected_boxes):
            sam_out = self.sam2_tracker.track(
                frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                prompt_box_xyxy=box_xyxy.astype(np.float32),
                caption="",
            )
            alloc = base + (1 if obj_idx < remainder else 0)
            if alloc <= 0:
                continue
            query_points_px, _ = build_vggt_query_prior(
                sam_out.masks_thw,
                sam_out.boxes_t4,
                num_queries=alloc,
            )
            if query_points_px.shape[0] > 0:
                per_object_queries.append(query_points_px)
        if not per_object_queries:
            motion_prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx)
            sam_out = self.sam2_tracker.track(
                frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                prompt_box_xyxy=motion_prompt_box_xyxy,
                caption=caption,
            )
            query_points_px, prior_source = build_vggt_query_prior(
                sam_out.masks_thw,
                sam_out.boxes_t4,
                num_queries=self.vggt_adapter.num_queries,
            )
            return query_points_px, prior_source, "grounded_text_empty_fallback", {
                "strategy": self.sam2_prior_strategy,
                "prompt_text": text_prompt,
                "object_count": 0,
                "used_fallback": True,
                "prior_source": prior_source,
                "detector_error": detector_error,
            }

        query_points = np.concatenate(per_object_queries, axis=0)[: self.vggt_adapter.num_queries].astype(np.float32)
        if query_points.shape[0] < self.vggt_adapter.num_queries:
            extra = query_points[-1:].repeat(self.vggt_adapter.num_queries - query_points.shape[0], axis=0)
            query_points = np.concatenate([query_points, extra], axis=0)
        prior_source = f"grounded_sam_objects{object_count}"
        return query_points.astype(np.float32), prior_source, f"{prompt_mode}_objects{object_count}", {
            "strategy": self.sam2_prior_strategy,
            "prompt_text": text_prompt,
            "object_count": object_count,
            "used_fallback": used_fallback,
            "prior_source": prior_source,
            "detector_error": detector_error,
        }

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        videos = batch["video"].to(self.device_obj)
        context_videos = batch["context_video"].to(self.device_obj)
        captions = list(batch["caption"])
        num_context_frames = batch["num_context_frames"].to(self.device_obj).long()
        target_context_frames = int(self.cfg["data"]["num_context_frames"])
        if context_videos.shape[2] < target_context_frames:
            pad_t = target_context_frames - context_videos.shape[2]
            pad = context_videos[:, :, -1:].expand(-1, -1, pad_t, -1, -1).contiguous()
            context_videos = torch.cat([context_videos, pad], dim=2)
            if "context_boxes" in batch:
                batch["context_boxes"] = torch.stack(
                    [self._pad_time_axis(value, target_context_frames, fill_mode="zeros") for value in batch["context_boxes"]],
                    dim=0,
                )
            if "context_states" in batch:
                batch["context_states"] = torch.stack(
                    [self._pad_time_axis(value, target_context_frames, fill_mode="zeros") for value in batch["context_states"]],
                    dim=0,
                )
        frame_valid_mask = self._frame_valid_mask(context_videos.shape[2], num_context_frames, self.device_obj)

        text_ctx = self._encode_text(captions)
        full_latents = self._encode_video_latents(videos)
        context_latents = self._encode_video_latents(context_videos)
        context_latent_batch = torch.stack(context_latents, dim=0)

        jepa_out = self.jepa_adapter(context_videos)
        frames_bthwc = context_videos.permute(0, 2, 3, 4, 1).float()
        frames_bthwc = (frames_bthwc + 1.0) / 2.0
        query_points_prior, sam_prior_sources, sam_prompt_modes, sam_prior_debug = self._maybe_build_query_priors(
            context_videos=context_videos,
            num_context_frames=num_context_frames,
            captions=captions,
        )
        vggt_out = self.vggt_adapter(
            frames_bthwc,
            query_points_prior=query_points_prior,
            query_image_hw=(context_videos.shape[-2], context_videos.shape[-1]) if query_points_prior is not None else None,
        )
        cotracker_out = None
        if self.cotracker_adapter is not None:
            cotracker_out = self.cotracker_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=(context_videos.shape[-2], context_videos.shape[-1]) if query_points_prior is not None else None,
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
        object_out = self.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latent_batch,
            tracks=tracks,
            visibility=visibility,
            confidence=confidence,
            track_image_hw=track_image_hw,
            vggt_world_points=vggt_out.world_points,
            vggt_world_points_conf=vggt_out.world_points_conf,
            vggt_depth=vggt_out.depth,
            vggt_depth_conf=vggt_out.depth_conf,
            frame_valid_mask=frame_valid_mask,
        )
        fused_context = self.context_fuser(text_ctx, object_out.object_tokens)

        track_alignment = None
        track_box_loss = None
        track_iou_loss = None
        tracks_native = None
        if "context_boxes" in batch:
            context_boxes = batch["context_boxes"].to(self.device_obj)
            scale_x = float(context_videos.shape[-1]) / float(track_image_hw[1])
            scale_y = float(context_videos.shape[-2]) / float(track_image_hw[0])
            tracks_native = tracks.clone()
            tracks_native[..., 0] *= scale_x
            tracks_native[..., 1] *= scale_y
            track_alignment = align_tracks_to_boxes(
                tracks=tracks_native,
                gt_boxes=context_boxes,
                image_hw=(context_videos.shape[-2], context_videos.shape[-1]),
            )
            track_box_loss = track_box_l1_loss(
                tracks=tracks_native,
                matched_gt_centers=track_alignment.matched_gt_centers,
                matched_gt_valid=track_alignment.matched_gt_valid,
            )
            track_iou_loss = track_box_iou_loss(
                tracks=tracks_native,
                gt_boxes=context_boxes,
                matched_gt_indices=track_alignment.matched_gt_indices,
                image_hw=(context_videos.shape[-2], context_videos.shape[-1]),
                radius_px=float(self.cfg.get("loss", {}).get("track_iou_radius_px", 12.0)),
            )

        debug = {
            "说明": {
                "context_video": "输入给 JEPA / VGGT / VAE 的上下文视频片段，batch 内可变长度会被 padding 到同一长度。",
                "sam2_prior": "如果开启，则先用 SAM2 在 context clip 上找到目标，再从 frame0 的 mask 或 box 采样 query points 作为 VGGT 的先验。",
                "sam2_prior_strategy": "可选单目标 motion/text prompt，或 Grounded-SAM 文本检测多目标，再分别采样 query points 给 VGGT。",
                "jepa_patch_tokens": "V-JEPA 对 context video 编码后的局部 patch token 网格 [B,Tj,Hj,Wj,Dj]。",
                "vggt_tracks": "VGGT 根据 query priors 或默认 queries 预测的 query-point tracks [B,Tctx,K,2]。",
                "cotracker_tracks": "如果 track_source=cotracker，则同一批 query points 会额外送入 CoTracker，object pooling 和 box 辅助约束都改用 CoTracker 轨迹。",
                "vggt_dense_geometry": "VGGT 还能输出 pose / depth / world_points；当前版本已把 world_points + depth 沿轨迹采样后并入 object geometry token。",
                "object_tokens": "沿着轨迹从 JEPA 局部 tube、VAE latent、轨迹几何特征、以及 VGGT dense geometry 中池化并投影得到的 object state tokens [B,K,D]。",
                "fused_context": "送入 Wan DiT cross-attention 的条件 token，等于 text tokens + 选出的 object tokens。",
            },
            "video": list(videos.shape),
            "context_video": list(context_videos.shape),
            "num_context_frames": num_context_frames.detach().cpu().tolist(),
            "context_frame_valid_mask": list(frame_valid_mask.shape),
            "text_context": self._shape_list(text_ctx),
            "full_latents": self._shape_list(full_latents),
            "context_latents": self._shape_list(context_latents),
            "jepa_patch_tokens": list(jepa_out.patch_tokens.shape),
            "vggt_query_points": list(vggt_out.query_points.shape),
            "vggt_tracks": list(vggt_out.tracks.shape),
            "vggt_visibility": list(vggt_out.visibility.shape),
            "vggt_confidence": list(vggt_out.confidence.shape),
            "vggt_pose_enc": list(vggt_out.pose_enc.shape) if vggt_out.pose_enc is not None else None,
            "vggt_depth": list(vggt_out.depth.shape) if vggt_out.depth is not None else None,
            "vggt_depth_conf": list(vggt_out.depth_conf.shape) if vggt_out.depth_conf is not None else None,
            "vggt_world_points": list(vggt_out.world_points.shape) if vggt_out.world_points is not None else None,
            "vggt_world_points_conf": list(vggt_out.world_points_conf.shape) if vggt_out.world_points_conf is not None else None,
            "track_source": self.track_source,
            "active_tracks": list(tracks.shape),
            "active_visibility": list(visibility.shape),
            "active_confidence": list(confidence.shape),
            "active_track_image_hw": list(track_image_hw),
            "object_tokens": list(object_out.object_tokens.shape),
            "object_jepa_tokens": list(object_out.jepa_tokens.shape),
            "object_latent_tokens": list(object_out.latent_tokens.shape),
            "object_geom_tokens": list(object_out.geom_tokens.shape),
            "object_vggt_geom_tokens": list(object_out.vggt_geom_tokens.shape) if object_out.vggt_geom_tokens is not None else None,
            "fused_context": self._shape_list(fused_context),
            "vggt_used_model": bool(vggt_out.used_model),
            "vggt_track_image_hw": list(vggt_out.image_hw),
            "video_path": batch["video_path"][0] if isinstance(batch["video_path"], list) else batch["video_path"],
            "frame_indices": batch["frame_indices"][0].tolist() if isinstance(batch["frame_indices"], torch.Tensor) and batch["frame_indices"].ndim == 2 else batch["frame_indices"],
            "caption": captions[0] if captions else "",
            "sam2_prior_strategy": self.sam2_prior_strategy,
            "sam_prior_sources": sam_prior_sources,
            "sam_prompt_modes": sam_prompt_modes,
            "sam_prior_debug": sam_prior_debug,
        }
        if query_points_prior is not None:
            debug["sam_query_points"] = list(query_points_prior.shape)
        if cotracker_out is not None:
            debug["cotracker_query_points"] = list(cotracker_out.query_points.shape)
            debug["cotracker_tracks"] = list(cotracker_out.tracks.shape)
            debug["cotracker_visibility"] = list(cotracker_out.visibility.shape)
            debug["cotracker_confidence"] = list(cotracker_out.confidence.shape)
            debug["cotracker_input_hw"] = list(cotracker_out.input_hw)
            debug["cotracker_used_model"] = bool(cotracker_out.used_model)
        if "context_boxes" in batch:
            debug["context_boxes"] = list(batch["context_boxes"].shape)
            debug["future_boxes"] = list(batch["future_boxes"].shape)
            debug["context_states"] = list(batch["context_states"].shape)
            debug["future_states"] = list(batch["future_states"].shape)
            debug["appearance"] = list(batch["appearance"].shape)
            debug["camera"] = list(batch["camera"].shape)
            if track_alignment is not None and track_box_loss is not None and track_iou_loss is not None and tracks_native is not None:
                debug["matched_gt_indices"] = list(track_alignment.matched_gt_indices.shape)
                debug["matched_gt_centers"] = list(track_alignment.matched_gt_centers.shape)
                debug["matched_gt_valid"] = list(track_alignment.matched_gt_valid.shape)
                debug["track_pair_cost"] = list(track_alignment.pair_cost.shape)
                debug["track_box_l1_loss"] = float(track_box_loss.item())
                debug["track_box_iou_loss"] = float(track_iou_loss.item())
                debug["tracks_native_xy"] = list(tracks_native.shape)

        return {
            "videos": videos,
            "context_videos": context_videos,
            "captions": captions,
            "num_context_frames": num_context_frames,
            "full_latents": full_latents,
            "context_latents": context_latents,
            "fused_context": fused_context,
            "track_box_loss": track_box_loss,
            "track_iou_loss": track_iou_loss,
            "debug": debug,
        }

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        self.bundle.ensure_dit_loaded()
        prepared = self._prepare_batch(batch)
        videos = prepared["videos"]
        num_context_frames = prepared["num_context_frames"]
        full_latents = prepared["full_latents"]
        context_latents = prepared["context_latents"]
        fused_context = prepared["fused_context"]
        track_box_loss = prepared["track_box_loss"]
        track_iou_loss = prepared["track_iou_loss"]
        dit_param = next(self.bundle.dit.parameters())
        dit_dtype = dit_param.dtype
        dit_device = dit_param.device

        losses = []
        for sample_idx, latent_clean in enumerate(full_latents):
            latent_clean = latent_clean.to(device=dit_device, dtype=dit_dtype)
            noise = torch.randn_like(latent_clean)
            timestep_id = torch.randint(0, len(self.scheduler.timesteps), (1,), device=self.device_obj)
            timestep = self.scheduler.timesteps[timestep_id.cpu()].to(device=dit_device, dtype=dit_dtype)
            x_t = self.scheduler.add_noise(latent_clean, noise, timestep.cpu())

            context_mask_t, future_mask_t = latent_frame_mask(
                num_video_frames=videos.shape[2],
                num_context_frames=int(num_context_frames[sample_idx].item()),
                vae_stride_t=self.bundle.config.vae_stride[0],
                device=dit_device,
            )
            context_mask = broadcast_latent_mask(context_mask_t, latent_clean)
            future_mask = broadcast_latent_mask(future_mask_t, latent_clean)
            context_clean_full = expand_context_latents_to_full(
                context_latents[sample_idx].to(device=dit_device, dtype=dit_dtype),
                latent_clean,
            )
            x_t = context_mask * context_clean_full + (1.0 - context_mask) * x_t

            seq_len = x_t.shape[1] * x_t.shape[2] * x_t.shape[3] // (
                self.bundle.config.patch_size[1] * self.bundle.config.patch_size[2]
            )
            t_tokens = torch.full((1, seq_len), float(timestep.item()), device=dit_device, dtype=dit_dtype)
            cond_context = fused_context[sample_idx].to(device=dit_device, dtype=dit_dtype)
            pred = self.bundle.dit(
                [x_t],
                t=t_tokens,
                context=[cond_context],
                seq_len=seq_len,
                y=None,
            )[0]

            target = self.scheduler.training_target(latent_clean, noise, timestep)
            denom = future_mask.sum().clamp_min(1.0)
            loss_main = ((pred - target) ** 2 * future_mask).sum() / denom
            loss_main = loss_main * self.scheduler.training_weight(
                timestep,
                device=loss_main.device,
                dtype=loss_main.dtype,
            )
            losses.append(loss_main)

        loss = torch.stack(losses).mean()
        if track_box_loss is not None:
            loss = loss + float(self.cfg.get("loss", {}).get("lambda_vggt_align", 0.0)) * track_box_loss
        if track_iou_loss is not None:
            loss = loss + float(self.cfg.get("loss", {}).get("lambda_vggt_iou", 0.0)) * track_iou_loss
        return loss

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        loss = self.forward(batch)
        self.state.step += 1
        return {"loss": float(loss.detach().item())}

    @torch.no_grad()
    def inspect_one_batch(self) -> dict[str, Any]:
        loader = self.build_dataloader(num_workers=0)
        batch = next(iter(loader))
        prepared = self._prepare_batch(batch)
        return prepared["debug"]

    @torch.no_grad()
    def write_inspection_report(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        debug = self.inspect_one_batch()

        json_path = output_path / "shape_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)

        video_src = Path(debug["video_path"])
        video_link = output_path / video_src.name
        if not video_link.exists():
            video_link.symlink_to(video_src)

        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Object-Centric Wan Context Report</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f5ef; color: #222; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; overflow-x: auto; white-space: pre-wrap; }}
    video {{ width: 100%; max-width: 720px; border: 1px solid #ccc; background: #000; }}
  </style>
</head>
<body>
  <h1>Object-Centric Wan Context Report</h1>
  <p>中文说明：可变长度 context 先编码成 JEPA / VAE 表征；若开启 object prior，则优先尝试用 Grounded-SAM 根据 caption 在 prompt frame 上检测物体，再把每个物体的 SAM2 mask/track 采样成 VGGT query points；若文本检测不可用或失败，则退回 motion-based SAM2 prior。随后沿轨迹池化 JEPA 与 latent 局部特征形成 object tokens，再与文本条件一起送入 Wan DiT。</p>
  <div class="grid">
    <div>
      <h2>Source Video</h2>
      <video controls src="./{video_src.name}"></video>
    </div>
    <div>
      <h2>Shape Report</h2>
      <pre>{json.dumps(debug, indent=2, ensure_ascii=False)}</pre>
    </div>
  </div>
</body>
</html>
"""
        html_path = output_path / "index.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        return html_path

    def train(self) -> None:
        from accelerate import Accelerator

        from code_vjepa_vggt.training.runner import launch_training_task

        opt_cfg = self.cfg["optimization"]
        accelerator = Accelerator(
            gradient_accumulation_steps=int(opt_cfg.get("grad_accum_steps", 1)),
            mixed_precision=str(opt_cfg.get("mixed_precision", "no")),
        )
        launch_training_task(
            accelerator,
            self,
            learning_rate=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            num_workers=int(self.cfg["data"]["num_workers"]),
            save_every=int(self.cfg["logging"]["save_every"]),
            max_steps=int(opt_cfg["max_steps"]),
            grad_accum_steps=int(opt_cfg.get("grad_accum_steps", 1)),
            max_grad_norm=float(opt_cfg.get("max_grad_norm", 0.0)),
        )
