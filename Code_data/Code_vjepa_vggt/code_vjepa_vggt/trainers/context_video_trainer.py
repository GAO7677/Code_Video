from __future__ import annotations

import json
import os
import time
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
    build_motion_prompt_boxes,
)
from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.data import BallBlockVideoDataset, PhysStateEpisodeDataset
from code_vjepa_vggt.models.object_aux_heads import ObjectAuxHeadOutput, ObjectAuxHeads
from code_vjepa_vggt.models.object_condition_adapter import ObjectConditionAdapter
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
        debug_init = os.environ.get("CODEX_DEBUG_TRAINER_INIT", "").strip() not in {"", "0", "false", "False"}
        init_t0 = time.perf_counter()
        def _debug_log(message: str) -> None:
            if debug_init:
                elapsed = time.perf_counter() - init_t0
                print(f"[trainer_init +{elapsed:.2f}s] {message}", flush=True)
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

        _debug_log("build WanContextVideoModel")
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
        _debug_log("freeze Wan parts")
        self.bundle.freeze_parts(
            freeze_vae=bool(model_cfg["freeze_vae"]),
            freeze_text_encoder=bool(model_cfg["freeze_text_encoder"]),
            freeze_dit=bool(model_cfg["freeze_wan_dit"]),
            freeze_lora=bool(model_cfg.get("freeze_wan_lora", False)),
        )
        if self.bundle.dit is not None:
            self.bundle.dit.train(mode=build_optimizer and not bool(model_cfg["freeze_wan_dit"]))

        cond_dim = int(model_cfg.get("cond_proj_dim", self.bundle.config.text_dim if hasattr(self.bundle.config, "text_dim") else 4096))
        if hasattr(self.bundle.config, "text_dim") and cond_dim != int(self.bundle.config.text_dim):
            raise ValueError(f"cond_proj_dim must match Wan text_dim={self.bundle.config.text_dim}, got {cond_dim}")
        self.max_objects = int(model_cfg.get("sam2_max_objects", 4))
        self.points_per_object = int(model_cfg.get("object_num_queries", 8))
        self.total_object_queries = int(self.max_objects * self.points_per_object)

        _debug_log("build JEPA adapter")
        self.jepa_adapter = JEPAPatchAdapter(
            ckpt_path=str(Path(model_cfg["je_pa_ckpt_dir"]) / "original" / "model.pth"),
            device=str(self.device_obj),
            crop_size=int(model_cfg["jepa_input_size"]),
            num_frames=int(data_cfg["num_context_frames"]),
            patch_size=int(model_cfg["jepa_patch_size"]),
            tubelet_size=int(model_cfg["jepa_tubelet_size"]),
            use_activation_checkpointing=bool(model_cfg.get("jepa_activation_checkpointing", False)),
            trainable=bool(model_cfg.get("train_jepa", False)),
        ).to(self.device_obj)
        _debug_log("build VGGT adapter")
        self.vggt_adapter = VGGTTrackAdapter(
            model_path=model_cfg.get("vggt_model_path"),
            num_queries=self.total_object_queries,
            device=str(self.device_obj),
            input_hw=tuple(model_cfg["vggt_input_hw"]),
            trainable=bool(model_cfg.get("train_vggt", False)),
        ).to(self.device_obj)
        self.track_source = str(model_cfg.get("track_source", "cotracker")).strip().lower()
        if self.track_source not in {"vggt", "cotracker"}:
            raise ValueError(f"unsupported track_source: {self.track_source}")
        self.cotracker_adapter = None
        if self.track_source == "cotracker":
            _debug_log("build CoTracker adapter")
            self.cotracker_adapter = CoTrackerAdapter(
                checkpoint_path=model_cfg.get("cotracker_checkpoint"),
                num_queries=self.total_object_queries,
                device=str(self.device_obj),
                input_hw=tuple(model_cfg.get("cotracker_input_hw", [384, 512])),
                window_len=int(model_cfg.get("cotracker_window_len", 60)),
            ).to(self.device_obj)
        latent_dim = int(getattr(self.bundle.config, "in_dim", 16))
        latent_dim = int(model_cfg.get("object_pooler_latent_dim", latent_dim))
        self.object_pooler_latent_dim = latent_dim
        _debug_log("build object pooler")
        self.object_pooler = ObjectTubeProjector(
            jepa_dim=self.jepa_adapter.encoder.backbone.embed_dim,
            latent_dim=latent_dim,
            out_dim=cond_dim,
            jepa_window_radius=int(model_cfg["jepa_window_radius"]),
            latent_window_radius=int(model_cfg["latent_window_radius"]),
        ).to(self.device_obj)
        if bool(model_cfg.get("freeze_object_pooler", False)):
            self.object_pooler.eval().requires_grad_(False)
        _debug_log("build object aux heads")
        self.object_aux_heads = ObjectAuxHeads(
            dim=cond_dim,
            track_delta_scale=float(model_cfg.get("track_head_delta_scale", 0.25)),
            box_delta_scale=float(model_cfg.get("box_head_delta_scale", 1.0)),
            box_wh_log_scale=float(model_cfg.get("box_head_wh_log_scale", 1.25)),
            box_wh_max_scale=float(model_cfg.get("box_head_wh_max_scale", 2.0)),
            track_gate_init=float(model_cfg.get("track_head_gate_init", 0.5)),
            box_center_gate_init=float(model_cfg.get("box_head_center_gate_init", 0.5)),
            box_size_gate_init=float(model_cfg.get("box_head_size_gate_init", 0.5)),
        ).to(self.device_obj)
        depth_lambda = float(self.cfg.get("loss", {}).get("lambda_depth_aux", 0.0))
        depth_target_index = self.cfg.get("loss", {}).get(
            "depth_target_state_index",
            model_cfg.get("depth_target_state_index"),
        )
        if depth_lambda <= 0.0 or depth_target_index is None:
            self.object_aux_heads.depth_head.eval().requires_grad_(False)
        _debug_log("build object condition adapter")
        self.object_adapter = ObjectConditionAdapter(
            dim=cond_dim,
            num_slots=self.max_objects,
            max_time_steps=int(model_cfg.get("max_object_time_steps", 64)),
        ).to(self.device_obj)
        self.scheduler = WanFlowMatchScheduler(num_train_timesteps=int(self.bundle.config.num_train_timesteps))
        self.init_wan_lora_from_checkpoint = model_cfg.get("init_wan_lora_from_checkpoint")
        if self.init_wan_lora_from_checkpoint is not None:
            _debug_log("load initial Wan LoRA checkpoint")
            self.bundle.load_lora_checkpoint(
                self.init_wan_lora_from_checkpoint,
                strict=bool(model_cfg.get("init_wan_lora_strict", True)),
                zero_missing=bool(model_cfg.get("init_wan_lora_zero_missing", False)),
            )
        if hasattr(self.object_pooler, "_ensure_jepa_proj"):
            _debug_log("ensure JEPA projection")
            self.object_pooler._ensure_jepa_proj(
                int(self.jepa_adapter.encoder.backbone.embed_dim),
                self.device_obj,
            )

        self.enable_sam2_priors = bool(model_cfg.get("enable_sam2_priors", False))
        self.sam2_prior_strategy = str(model_cfg.get("sam2_prior_strategy", "single")).strip().lower()
        self.sam2_tracker = None
        self.text_detector = None
        if self.enable_sam2_priors:
            _debug_log("build SAM2 priors")
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
        _debug_log("dataset built")
        self.state = TrainerState()
        self.last_train_metrics: dict[str, float] = {}
        self.last_loss_breakdown: dict[str, torch.Tensor] = {}

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
                init_scan_limit=data_cfg.get("init_scan_limit"),
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
        params += list(self.object_pooler.parameters())
        params += list(self.object_aux_heads.parameters())
        params += list(self.object_adapter.parameters())
        if getattr(self.jepa_adapter, "trainable", False):
            params += list(self.jepa_adapter.parameters())
        if getattr(self.vggt_adapter, "trainable", False):
            params += list(self.vggt_adapter.parameters())
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

    @staticmethod
    def _latent_window_size(num_frames: int, latent_frames: int) -> int:
        if int(num_frames) % max(int(latent_frames), 1) != 0:
            raise RuntimeError(
                f"num_frames={num_frames} must be divisible by latent_frames={latent_frames} "
                "for latent-time object conditioning"
            )
        return int(num_frames) // max(int(latent_frames), 1)

    @classmethod
    def _group_last(cls, values: torch.Tensor, latent_frames: int) -> torch.Tensor:
        group = cls._latent_window_size(int(values.shape[1]), int(latent_frames))
        new_shape = (values.shape[0], int(latent_frames), group) + tuple(values.shape[2:])
        return values.view(new_shape)[:, :, -1]

    @classmethod
    def _group_track_summary(
        cls,
        centers_xy: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        image_hw: tuple[int, int],
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = cls._latent_window_size(int(centers_xy.shape[1]), int(latent_frames))
        height, width = int(image_hw[0]), int(image_hw[1])
        centers_xy = centers_xy.view(centers_xy.shape[0], int(latent_frames), group, centers_xy.shape[2], 2)
        valid_mask = valid_mask.view(valid_mask.shape[0], int(latent_frames), group, valid_mask.shape[2])
        first_xy = centers_xy[:, :, 0]
        last_xy = centers_xy[:, :, -1]
        last_xy_norm = torch.stack(
            [
                last_xy[..., 0] / max(float(width - 1), 1.0),
                last_xy[..., 1] / max(float(height - 1), 1.0),
            ],
            dim=-1,
        ).clamp(0.0, 1.0)
        delta_xy_norm = torch.stack(
            [
                (last_xy[..., 0] - first_xy[..., 0]) / max(float(width - 1), 1.0),
                (last_xy[..., 1] - first_xy[..., 1]) / max(float(height - 1), 1.0),
            ],
            dim=-1,
        )
        out = torch.cat([last_xy_norm, delta_xy_norm], dim=-1)
        out_valid = valid_mask.any(dim=2)
        return out, out_valid

    @staticmethod
    def _group_tracks_to_objects(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        *,
        max_objects: int,
        points_per_object: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected_queries = int(max_objects) * int(points_per_object)
        if int(tracks.shape[2]) != expected_queries:
            raise ValueError(
                f"flat query count mismatch: got {int(tracks.shape[2])}, expected {expected_queries} "
                f"(max_objects={max_objects}, points_per_object={points_per_object})"
            )
        tracks_grouped = tracks.view(tracks.shape[0], tracks.shape[1], int(max_objects), int(points_per_object), 2)
        visibility_grouped = visibility.view(visibility.shape[0], visibility.shape[1], int(max_objects), int(points_per_object))
        confidence_grouped = confidence.view(confidence.shape[0], confidence.shape[1], int(max_objects), int(points_per_object))
        return tracks_grouped, visibility_grouped, confidence_grouped

    @staticmethod
    def _object_center_tracks_from_grouped(
        tracks: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        object_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = (visibility * confidence).clamp_min(0.0)
        denom = weights.sum(dim=3, keepdim=True).clamp_min(1.0e-6)
        centers = (tracks * weights.unsqueeze(-1)).sum(dim=3) / denom
        valid = weights.sum(dim=3) > 1.0e-6
        if object_valid_mask is not None:
            slot_valid = object_valid_mask[:, None, :].to(dtype=valid.dtype, device=valid.device) > 0.5
            valid = valid & slot_valid
        return centers, valid

    @staticmethod
    def _gather_matched_gt_features(values: torch.Tensor, matched_gt_indices: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError(f"expected values with shape [B,T,G,D], got {list(values.shape)}")
        gather_idx = matched_gt_indices[:, None, :, None].expand(-1, values.shape[1], -1, values.shape[-1])
        return torch.gather(values, 2, gather_idx)

    @staticmethod
    def _gather_matched_gt_mask(values: torch.Tensor, matched_gt_indices: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"expected values with shape [B,T,G], got {list(values.shape)}")
        gather_idx = matched_gt_indices[:, None, :].expand(-1, values.shape[1], -1)
        return torch.gather(values, 2, gather_idx)

    @classmethod
    def _group_box_targets(
        cls,
        boxes_xyxy: torch.Tensor,
        valid_mask: torch.Tensor,
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        group = cls._latent_window_size(int(boxes_xyxy.shape[1]), int(latent_frames))
        grouped_boxes = boxes_xyxy.view(boxes_xyxy.shape[0], int(latent_frames), group, boxes_xyxy.shape[2], 4)[:, :, -1]
        grouped_valid = valid_mask.view(valid_mask.shape[0], int(latent_frames), group, valid_mask.shape[2]).any(dim=2)
        return grouped_boxes, grouped_valid

    def _maybe_build_query_priors(
        self,
        context_videos: torch.Tensor,
        num_context_frames: torch.Tensor,
        captions: list[str],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, list[str], list[str], list[dict[str, Any]]]:
        if not self.enable_sam2_priors or self.sam2_tracker is None:
            batch_size = int(context_videos.shape[0])
            prior_debugs = [
                {
                    "strategy": "disabled",
                    "prior_source": "uniform_queries",
                    "object_count": 0,
                }
                for _ in range(batch_size)
            ]
            return None, None, ["uniform_queries"] * batch_size, ["disabled"] * batch_size, prior_debugs
        if self.sam2_tracker is None:
            raise RuntimeError("SAM2 tracker is required to build query priors")

        priors = []
        object_valid_masks = []
        prior_sources: list[str] = []
        prompt_modes: list[str] = []
        prior_debugs: list[dict[str, Any]] = []
        for batch_idx in range(context_videos.shape[0]):
            valid_frames = int(num_context_frames[batch_idx].item())
            frames_tchw_01 = ((context_videos[batch_idx, :, :valid_frames].permute(1, 0, 2, 3).float() + 1.0) / 2.0).detach().cpu().numpy()
            prompt_frame_idx = max(valid_frames - 1, 0)
            query_points_px, object_valid_mask, prior_source, prompt_mode, prior_debug = self._build_query_prior_for_sample(
                frames_tchw_01=frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                caption=captions[batch_idx],
            )
            priors.append(torch.from_numpy(query_points_px))
            object_valid_masks.append(torch.from_numpy(object_valid_mask))
            prior_sources.append(prior_source)
            prompt_modes.append(prompt_mode)
            prior_debugs.append(
                {
                    "prompt_frame_idx": int(prompt_frame_idx),
                    "valid_frames": int(valid_frames),
                    **prior_debug,
                }
            )
        stacked = torch.stack(priors, dim=0).to(device=self.device_obj, dtype=context_videos.dtype)
        object_valid = torch.stack(object_valid_masks, dim=0).to(device=self.device_obj, dtype=context_videos.dtype)
        return stacked, object_valid, prior_sources, prompt_modes, prior_debugs

    def _build_query_prior_for_sample(
        self,
        *,
        frames_tchw_01: Any,
        prompt_frame_idx: int,
        caption: str,
    ) -> tuple[Any, Any, str, str, dict[str, Any]]:
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
            num_queries=self.points_per_object,
        )
        grouped = np.repeat(query_points_px[None, :, :], self.max_objects, axis=0).astype(np.float32)
        valid_mask = np.zeros((self.max_objects,), dtype=np.float32)
        valid_mask[0] = 1.0
        return grouped, valid_mask, prior_source, sam_out.prompt_mode, {
            "strategy": self.sam2_prior_strategy,
            "prompt_text": sam_out.prompt_text,
            "object_count": 1,
            "prior_source": prior_source,
        }

    def _build_multi_object_query_prior(
        self,
        *,
        frames_tchw_01: np.ndarray,
        prompt_frame_idx: int,
        caption: str,
    ) -> tuple[np.ndarray, np.ndarray, str, str, dict[str, Any]]:
        max_objects = self.max_objects
        text_prompt = _build_multi_object_prompt(caption)
        detected_boxes = None
        prompt_mode = "caption_gdino_multi"
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
                detector_error = f"{type(exc).__name__}: {exc}"
        if detected_boxes is None or detected_boxes.shape[0] == 0:
            motion_multi = build_motion_prompt_boxes(frames_tchw_01, max_boxes=max_objects)
            detected_boxes = motion_multi.boxes_xyxy[:max_objects]
            prompt_mode = motion_multi.prompt_mode
            if detected_boxes.shape[0] == 0:
                raise RuntimeError(
                    "failed to build any multi-object query priors from GroundingDINO or motion fallback; "
                    f"prompt_frame_idx={prompt_frame_idx}, text_prompt={text_prompt!r}, detector_error={detector_error}"
                )

        object_count = min(int(detected_boxes.shape[0]), int(max_objects))
        detected_boxes = detected_boxes[:object_count]
        grouped_queries = np.zeros((max_objects, self.points_per_object, 2), dtype=np.float32)
        object_valid_mask = np.zeros((max_objects,), dtype=np.float32)
        for obj_idx, box_xyxy in enumerate(detected_boxes):
            sam_out = self.sam2_tracker.track(
                frames_tchw_01,
                prompt_frame_idx=prompt_frame_idx,
                prompt_box_xyxy=box_xyxy.astype(np.float32),
                caption="",
            )
            query_points_px, _ = build_vggt_query_prior(
                sam_out.masks_thw,
                sam_out.boxes_t4,
                num_queries=self.points_per_object,
            )
            if query_points_px.shape[0] == 0:
                continue
            if query_points_px.shape[0] < self.points_per_object:
                extra = query_points_px[-1:].repeat(self.points_per_object - query_points_px.shape[0], axis=0)
                query_points_px = np.concatenate([query_points_px, extra], axis=0)
            grouped_queries[obj_idx] = query_points_px[: self.points_per_object].astype(np.float32)
            object_valid_mask[obj_idx] = 1.0
        if float(object_valid_mask.sum()) <= 0.0:
            raise RuntimeError(
                "SAM2 failed to produce any query priors after GroundingDINO detections; "
                f"prompt_frame_idx={prompt_frame_idx}, text_prompt={text_prompt!r}, detector_error={detector_error}"
            )
        prior_source = f"grounded_sam_objects{object_count}"
        return grouped_queries.astype(np.float32), object_valid_mask.astype(np.float32), prior_source, f"{prompt_mode}_objects{object_count}", {
            "strategy": self.sam2_prior_strategy,
            "prompt_text": text_prompt,
            "object_count": object_count,
            "prior_source": prior_source,
            "detector_error": detector_error,
        }

    def _prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        videos = batch["video"].to(self.device_obj)
        context_videos = batch["context_video"].to(self.device_obj)
        captions = list(batch["caption"])
        num_context_frames = batch["num_context_frames"].to(self.device_obj).long()
        target_context_frames = int(self.cfg["data"]["num_context_frames"])
        if int(context_videos.shape[2]) != target_context_frames:
            raise RuntimeError(
                f"context_videos must be fixed-length before training; "
                f"got T={int(context_videos.shape[2])}, expected {target_context_frames}"
            )
        frame_valid_mask = self._frame_valid_mask(context_videos.shape[2], num_context_frames, self.device_obj)

        text_ctx = self._encode_text(captions)
        full_latents = self._encode_video_latents(videos)
        context_latents = self._encode_video_latents(context_videos)
        context_latent_batch = torch.stack(context_latents, dim=0)

        jepa_out = self.jepa_adapter(context_videos)
        frames_bthwc = context_videos.permute(0, 2, 3, 4, 1).float()
        frames_bthwc = (frames_bthwc + 1.0) / 2.0
        query_points_grouped, object_valid_mask, sam_prior_sources, sam_prompt_modes, sam_prior_debug = self._maybe_build_query_priors(
            context_videos=context_videos,
            num_context_frames=num_context_frames,
            captions=captions,
        )
        query_points_prior = None
        if query_points_grouped is not None:
            query_points_prior = query_points_grouped.view(
                query_points_grouped.shape[0],
                self.total_object_queries,
                2,
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
        tracks_grouped, visibility_grouped, confidence_grouped = self._group_tracks_to_objects(
            tracks,
            visibility,
            confidence,
            max_objects=self.max_objects,
            points_per_object=self.points_per_object,
        )
        object_out = self.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latent_batch,
            tracks=tracks_grouped,
            visibility=visibility_grouped,
            confidence=confidence_grouped,
            track_image_hw=track_image_hw,
            object_valid_mask=object_valid_mask,
            vggt_world_points=vggt_out.world_points,
            vggt_world_points_conf=vggt_out.world_points_conf,
            vggt_depth=vggt_out.depth,
            vggt_depth_conf=vggt_out.depth_conf,
            vggt_geometry_image_hw=vggt_out.image_hw,
            frame_valid_mask=frame_valid_mask,
        )
        object_aux_out = self.object_aux_heads(
            object_out.object_latent_tokens,
            object_out.active_track_summary,
        )
        object_context = self.object_adapter(object_out.object_latent_tokens)

        track_alignment = None
        track_box_loss = None
        track_iou_loss = None
        tracks_native = None
        center_tracks_native = None
        gt_track_summary = None
        gt_track_valid = None
        gt_box_xyxy = None
        gt_box_valid = None
        gt_depth = None
        gt_depth_valid = None
        depth_target_index = self.cfg.get("loss", {}).get(
            "depth_target_state_index",
            self.cfg["model"].get("depth_target_state_index"),
        )
        if "context_boxes" in batch:
            context_boxes = batch["context_boxes"].to(self.device_obj)
            scale_x = float(context_videos.shape[-1]) / float(track_image_hw[1])
            scale_y = float(context_videos.shape[-2]) / float(track_image_hw[0])
            tracks_native = tracks_grouped.clone()
            tracks_native[..., 0] *= scale_x
            tracks_native[..., 1] *= scale_y
            center_tracks_native, center_track_valid = self._object_center_tracks_from_grouped(
                tracks_native,
                visibility_grouped,
                confidence_grouped,
                object_valid_mask=object_valid_mask,
            )
            track_alignment = align_tracks_to_boxes(
                tracks=center_tracks_native,
                gt_boxes=context_boxes,
                image_hw=(context_videos.shape[-2], context_videos.shape[-1]),
            )
            track_box_loss = track_box_l1_loss(
                tracks=center_tracks_native,
                matched_gt_centers=track_alignment.matched_gt_centers,
                matched_gt_valid=track_alignment.matched_gt_valid * center_track_valid.to(dtype=track_alignment.matched_gt_valid.dtype),
            )
            track_iou_loss = track_box_iou_loss(
                tracks=center_tracks_native,
                gt_boxes=context_boxes,
                matched_gt_indices=track_alignment.matched_gt_indices,
                image_hw=(context_videos.shape[-2], context_videos.shape[-1]),
                radius_px=float(self.cfg.get("loss", {}).get("track_iou_radius_px", 12.0)),
            )
            latent_frames = int(object_out.object_latent_tokens.shape[1])
            gt_valid_full = (track_alignment.matched_gt_valid > 0.5) & frame_valid_mask.unsqueeze(-1) & center_track_valid
            gt_track_summary, gt_track_valid = self._group_track_summary(
                track_alignment.matched_gt_centers,
                gt_valid_full,
                image_hw=(context_videos.shape[-2], context_videos.shape[-1]),
                latent_frames=latent_frames,
            )
            matched_gt_boxes = self._gather_matched_gt_features(
                context_boxes,
                track_alignment.matched_gt_indices,
            )
            matched_gt_box_valid = (
                ((matched_gt_boxes[..., 2] - matched_gt_boxes[..., 0]) > 1.0e-6)
                & ((matched_gt_boxes[..., 3] - matched_gt_boxes[..., 1]) > 1.0e-6)
                & frame_valid_mask.unsqueeze(-1)
            )
            gt_box_xyxy, gt_box_valid = self._group_box_targets(
                matched_gt_boxes,
                matched_gt_box_valid,
                latent_frames,
            )
            if depth_target_index is not None and "context_states" in batch:
                context_states = batch["context_states"].to(self.device_obj)
                depth_target_index = int(depth_target_index)
                if depth_target_index < 0 or depth_target_index >= int(context_states.shape[-1]):
                    raise ValueError(
                        f"depth_target_state_index={depth_target_index} is out of range for "
                        f"context_states shape {list(context_states.shape)}"
                    )
                matched_gt_depth = self._gather_matched_gt_features(
                    context_states[..., depth_target_index : depth_target_index + 1],
                    track_alignment.matched_gt_indices,
                )
                gt_depth = self._group_last(matched_gt_depth, latent_frames)
                gt_depth_valid = gt_box_valid

        debug = {
            "说明": {
                "context_video": "输入给 JEPA / VGGT / VAE 的上下文视频片段，batch 内可变长度会被 padding 到同一长度。",
                "sam2_prior": "如果开启，则先用 SAM2 在 context clip 上找到目标，再从 frame0 的 mask 或 box 采样 query points 作为 VGGT 的先验。",
                "sam2_prior_strategy": "可选单目标 motion/text prompt，或 Grounded-SAM 文本检测多目标，再分别采样 query points 给 VGGT。",
                "jepa_patch_tokens": "V-JEPA 对 context video 编码后的局部 patch token 网格 [B,Tj,Hj,Wj,Dj]。",
                "vggt_tracks": "VGGT 根据 query priors 或默认 queries 预测的 query-point tracks [B,Tctx,K,2]。",
                "cotracker_tracks": "如果 track_source=cotracker，则同一批 query points 会额外送入 CoTracker，object pooling 和 box 辅助约束都改用 CoTracker 轨迹。",
                "vggt_dense_geometry": "VGGT 还能输出 pose / depth / world_points；当前版本已把 world_points + depth 沿轨迹采样后并入 object geometry token。",
                "object_latent_tokens": "主条件改成 latent-time object token [B,T_lat,K,D]，先与文本分开，后续由 Wan block 的独立 object cross-attn 消化。",
                "object_context": "object_latent_tokens 加入 time/slot adapter 后展平成 [B,T_lat*K,D]，送入 Wan gated object cross-attn。",
                "object_aux_heads": "训练期额外从 object_latent_tokens 预测 track summary / box / depth，用于监督，不作为推理必须输出。",
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
            "active_tracks_flat": list(tracks.shape),
            "active_tracks_grouped": list(tracks_grouped.shape),
            "active_visibility_flat": list(visibility.shape),
            "active_visibility_grouped": list(visibility_grouped.shape),
            "active_confidence_flat": list(confidence.shape),
            "active_confidence_grouped": list(confidence_grouped.shape),
            "active_track_image_hw": list(track_image_hw),
            "object_tokens": list(object_out.object_tokens.shape),
            "object_jepa_tokens": list(object_out.jepa_tokens.shape),
            "object_latent_tokens": list(object_out.object_latent_tokens.shape),
            "object_geom_tokens": list(object_out.geom_tokens.shape),
            "object_vggt_geom_tokens": list(object_out.vggt_geom_tokens.shape) if object_out.vggt_geom_tokens is not None else None,
            "object_context": list(object_context.shape),
            "object_aux_pred_track_summary": list(object_aux_out.pred_track_summary.shape),
            "object_aux_pred_box_xyxy": list(object_aux_out.pred_box_xyxy.shape),
            "object_aux_pred_depth": list(object_aux_out.pred_depth.shape),
            "vggt_used_model": bool(vggt_out.used_model),
            "vggt_track_image_hw": list(vggt_out.image_hw),
            "video_path": batch["video_path"][0] if isinstance(batch["video_path"], list) else batch["video_path"],
            "frame_indices": batch["frame_indices"][0].tolist() if isinstance(batch["frame_indices"], torch.Tensor) and batch["frame_indices"].ndim == 2 else batch["frame_indices"],
            "caption": captions[0] if captions else "",
            "sam2_prior_strategy": self.sam2_prior_strategy,
            "sam_prior_sources": sam_prior_sources,
            "sam_prompt_modes": sam_prompt_modes,
            "sam_prior_debug": sam_prior_debug,
            "max_objects": self.max_objects,
            "points_per_object": self.points_per_object,
        }
        if query_points_grouped is not None:
            debug["sam_query_points_grouped"] = list(query_points_grouped.shape)
        if query_points_prior is not None:
            debug["sam_query_points_flat"] = list(query_points_prior.shape)
        if object_valid_mask is not None:
            debug["object_valid_mask"] = list(object_valid_mask.shape)
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
                debug["tracks_native_xy_grouped"] = list(tracks_native.shape)
            if center_tracks_native is not None:
                debug["center_tracks_native_xy"] = list(center_tracks_native.shape)
            if gt_track_summary is not None and gt_track_valid is not None:
                debug["gt_track_summary"] = list(gt_track_summary.shape)
                debug["gt_track_valid"] = list(gt_track_valid.shape)
            if gt_box_xyxy is not None and gt_box_valid is not None:
                debug["gt_box_xyxy"] = list(gt_box_xyxy.shape)
                debug["gt_box_valid"] = list(gt_box_valid.shape)
            if gt_depth is not None and gt_depth_valid is not None:
                debug["gt_depth"] = list(gt_depth.shape)
                debug["gt_depth_valid"] = list(gt_depth_valid.shape)
                debug["depth_target_state_index"] = int(depth_target_index)

        return {
            "videos": videos,
            "context_videos": context_videos,
            "captions": captions,
            "num_context_frames": num_context_frames,
            "full_latents": full_latents,
            "context_latents": context_latents,
            "text_context": text_ctx,
            "object_context": object_context,
            "object_latent_tokens": object_out.object_latent_tokens,
            "object_aux_out": object_aux_out,
            "object_tokens": object_out.object_tokens,
            "track_box_loss": track_box_loss,
            "track_iou_loss": track_iou_loss,
            "gt_track_summary": gt_track_summary,
            "gt_track_valid": gt_track_valid,
            "gt_box_xyxy": gt_box_xyxy,
            "gt_box_valid": gt_box_valid,
            "gt_depth": gt_depth,
            "gt_depth_valid": gt_depth_valid,
            "debug": debug,
        }

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        self.bundle.ensure_dit_loaded()
        prepared = self._prepare_batch(batch)
        videos = prepared["videos"]
        num_context_frames = prepared["num_context_frames"]
        full_latents = prepared["full_latents"]
        context_latents = prepared["context_latents"]
        text_context = prepared["text_context"]
        object_context = prepared["object_context"]
        object_latent_tokens = prepared["object_latent_tokens"]
        object_aux_out: ObjectAuxHeadOutput = prepared["object_aux_out"]
        object_tokens = prepared["object_tokens"]
        track_box_loss = prepared["track_box_loss"]
        track_iou_loss = prepared["track_iou_loss"]
        gt_track_summary = prepared["gt_track_summary"]
        gt_track_valid = prepared["gt_track_valid"]
        gt_box_xyxy = prepared["gt_box_xyxy"]
        gt_box_valid = prepared["gt_box_valid"]
        gt_depth = prepared["gt_depth"]
        gt_depth_valid = prepared["gt_depth_valid"]
        dit_param = next(self.bundle.dit.parameters())
        dit_dtype = dit_param.dtype
        dit_device = dit_param.device
        text_context_abs_max = max(float(ctx.detach().abs().max().item()) for ctx in text_context)
        object_context_abs_max = float(object_context.detach().abs().max().item())
        fused_context_abs_max = max(text_context_abs_max, object_context_abs_max)
        object_tokens_abs_max = float(object_tokens.detach().abs().max().item())
        object_latent_tokens_abs_max = float(object_latent_tokens.detach().abs().max().item())

        losses = []
        pred_abs_max_values: list[float] = []
        latent_abs_max_values: list[float] = []
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
            latent_abs_max_values.append(float(x_t.detach().abs().max().item()))

            seq_len = x_t.shape[1] * x_t.shape[2] * x_t.shape[3] // (
                self.bundle.config.patch_size[1] * self.bundle.config.patch_size[2]
            )
            t_tokens = torch.full((1, seq_len), float(timestep.item()), device=dit_device, dtype=dit_dtype)
            text_ctx_sample = text_context[sample_idx].to(device=dit_device, dtype=dit_dtype)
            object_ctx_sample = object_context[sample_idx].to(device=dit_device, dtype=dit_dtype)
            pred = self.bundle.dit(
                [x_t],
                t=t_tokens,
                context=None,
                text_context=[text_ctx_sample],
                object_context=[object_ctx_sample],
                seq_len=seq_len,
                y=None,
            )[0]
            if not torch.isfinite(pred).all():
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            pred_abs_max_values.append(float(pred.detach().abs().max().item()))

            target = self.scheduler.training_target(latent_clean, noise, timestep)
            if not torch.isfinite(target).all():
                raise RuntimeError(
                    f"non-finite target detected at sample_idx={sample_idx}, "
                    f"target_min={float(torch.nan_to_num(target).min().item())}, "
                    f"target_max={float(torch.nan_to_num(target).max().item())}"
                )
            future_mask_sum = future_mask.sum().clamp_min(1.0)
            masked_mse = ((pred - target) ** 2 * future_mask).sum()
            loss_main = masked_mse / future_mask.sum().clamp_min(1.0)
            loss_main = loss_main / float(pred.numel() / max(int(future_mask_sum.item()), 1))
            loss_main = loss_main * self.scheduler.training_weight(
                timestep,
                device=loss_main.device,
                dtype=loss_main.dtype,
            )
            if not torch.isfinite(loss_main).all():
                raise RuntimeError(
                    f"non-finite loss_main detected at sample_idx={sample_idx}, "
                    f"pred_finite={bool(torch.isfinite(pred).all())}, "
                    f"target_finite={bool(torch.isfinite(target).all())}"
                )
            losses.append(loss_main)

        loss_main = torch.stack(losses).mean()
        if track_box_loss is not None:
            track_box_loss = torch.nan_to_num(track_box_loss, nan=0.0, posinf=0.0, neginf=0.0)
        if track_iou_loss is not None:
            track_iou_loss = torch.nan_to_num(track_iou_loss, nan=0.0, posinf=0.0, neginf=0.0)
        track_aux_loss = loss_main.new_zeros(())
        box_aux_loss = loss_main.new_zeros(())
        depth_aux_loss = loss_main.new_zeros(())
        if gt_track_summary is not None and gt_track_valid is not None:
            pred_track_summary = torch.nan_to_num(object_aux_out.pred_track_summary, nan=0.0, posinf=0.0, neginf=0.0)
            weights = gt_track_valid.unsqueeze(-1).to(dtype=pred_track_summary.dtype, device=pred_track_summary.device)
            denom = weights.sum().clamp_min(1.0) * pred_track_summary.shape[-1]
            track_aux_loss = ((pred_track_summary - gt_track_summary).abs() * weights).sum() / denom
        if gt_box_xyxy is not None and gt_box_valid is not None:
            pred_box_xyxy = torch.nan_to_num(object_aux_out.pred_box_xyxy, nan=0.0, posinf=0.0, neginf=0.0)
            weights = gt_box_valid.unsqueeze(-1).to(dtype=pred_box_xyxy.dtype, device=pred_box_xyxy.device)
            denom = weights.sum().clamp_min(1.0) * pred_box_xyxy.shape[-1]
            box_aux_loss = ((pred_box_xyxy - gt_box_xyxy).abs() * weights).sum() / denom
        if gt_depth is not None and gt_depth_valid is not None:
            pred_depth = torch.nan_to_num(object_aux_out.pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
            weights = gt_depth_valid.unsqueeze(-1).to(dtype=pred_depth.dtype, device=pred_depth.device)
            denom = weights.sum().clamp_min(1.0) * pred_depth.shape[-1]
            depth_aux_loss = ((pred_depth - gt_depth).abs() * weights).sum() / denom

        loss_cfg = self.cfg.get("loss", {})
        lambda_main = float(loss_cfg.get("lambda_main", 1.0))
        lambda_track_aux = float(loss_cfg.get("lambda_track_aux", 0.1))
        lambda_box_aux = float(loss_cfg.get("lambda_box_aux", 0.1))
        lambda_depth_aux = float(loss_cfg.get("lambda_depth_aux", 0.0))
        loss = (
            lambda_main * loss_main
            + lambda_track_aux * track_aux_loss
            + lambda_box_aux * box_aux_loss
            + lambda_depth_aux * depth_aux_loss
        )
        self.last_loss_breakdown = {
            "loss_main": loss_main,
            "track_aux_loss": track_aux_loss,
            "box_aux_loss": box_aux_loss,
            "depth_aux_loss": depth_aux_loss,
            "loss_total": loss,
            "lambda_main": loss_main.new_tensor(lambda_main),
            "lambda_track_aux": loss_main.new_tensor(lambda_track_aux),
            "lambda_box_aux": loss_main.new_tensor(lambda_box_aux),
            "lambda_depth_aux": loss_main.new_tensor(lambda_depth_aux),
        }
        if not torch.isfinite(loss).all():
            raise RuntimeError(
                "non-finite total loss detected; "
                f"loss_main_mean={float(loss_main.item())}, "
                f"track_aux_loss={float(track_aux_loss.item())}, "
                f"box_aux_loss={float(box_aux_loss.item())}, "
                f"depth_aux_loss={float(depth_aux_loss.item())}, "
                f"track_box_loss={None if track_box_loss is None else float(track_box_loss.item())}, "
                f"track_iou_loss={None if track_iou_loss is None else float(track_iou_loss.item())}"
            )
        self.last_train_metrics = {
            "train/loss_main": float(loss_main.item()),
            "train/loss_total": float(loss.item()),
            "train/loss_track_aux": float(track_aux_loss.item()),
            "train/loss_box_aux": float(box_aux_loss.item()),
            "train/loss_depth_aux": float(depth_aux_loss.item()),
            "train/object_tokens_abs_max": float(object_tokens_abs_max),
            "train/object_latent_tokens_abs_max": float(object_latent_tokens_abs_max),
            "train/text_context_abs_max": float(text_context_abs_max),
            "train/object_context_abs_max": float(object_context_abs_max),
            "train/fused_context_abs_max": float(fused_context_abs_max),
            "train/pred_abs_max": float(max(pred_abs_max_values) if pred_abs_max_values else 0.0),
            "train/x_t_abs_max": float(max(latent_abs_max_values) if latent_abs_max_values else 0.0),
            "train/track_box_loss": float(track_box_loss.item()) if track_box_loss is not None else 0.0,
            "train/track_iou_loss": float(track_iou_loss.item()) if track_iou_loss is not None else 0.0,
        }
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

    def train(self, resume_checkpoint: str | Path | None = None, init_from: str | Path | None = None) -> None:
        from accelerate import Accelerator
        from accelerate.utils import DistributedDataParallelKwargs

        from code_vjepa_vggt.training.runner import launch_training_task

        opt_cfg = self.cfg["optimization"]
        accelerator = Accelerator(
            gradient_accumulation_steps=int(opt_cfg.get("grad_accum_steps", 1)),
            mixed_precision=str(opt_cfg.get("mixed_precision", "no")),
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
        )
        launch_training_task(
            accelerator,
            self,
            optimizer_type=str(opt_cfg.get("optimizer_type", "adamw")),
            learning_rate=float(opt_cfg["lr"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            betas=tuple(float(value) for value in opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1.0e-8)),
            num_workers=int(self.cfg["data"]["num_workers"]),
            save_every=int(self.cfg["logging"]["save_every"]),
            max_steps=int(opt_cfg["max_steps"]),
            grad_accum_steps=int(opt_cfg.get("grad_accum_steps", 1)),
            max_grad_norm=float(opt_cfg.get("max_grad_norm", 0.0)),
            resume_checkpoint=resume_checkpoint,
            init_from=init_from,
        )
