from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from code_vjepa_vggt.adapters.sam2_motion import (
    DetectionPromptOutput,
    MotionPromptBoxesOutput,
    SAM2TrackOutput,
    build_motion_prompt_box,
    build_motion_prompt_boxes,
)
from code_vjepa_vggt.models.object_tokens import ObjectTokenOutput, ObjectTubeProjector
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer, _build_multi_object_prompt
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import (
    broadcast_latent_mask,
    collate_video_batch,
    expand_context_latents_to_full,
    latent_frame_mask,
)
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior
from code_vjepa_vggt.utils.track_supervision import (
    TrackBoxAlignment,
    align_tracks_to_boxes,
    track_box_iou_loss,
    track_box_l1_loss,
)


GT_PALETTE = ["#d62828", "#f77f00", "#fcbf49", "#2a9d8f", "#277da1", "#6a4c93"]
OBJ_PALETTE = ["#0077b6", "#00b4d8", "#40916c", "#8338ec", "#ff006e", "#fb5607", "#8ac926", "#3a86ff"]


@dataclass
class QueryPriorObjectDebug:
    prompt_box_xyxy: np.ndarray
    prompt_mode: str
    prompt_text: str
    masks_thw: np.ndarray
    boxes_t4: np.ndarray
    query_points_px: np.ndarray
    allocated_queries: int


@dataclass
class QueryPriorDebug:
    strategy: str
    prompt_frame_idx: int
    prompt_text: str
    prior_source: str
    object_records: list[QueryPriorObjectDebug]
    query_points_px: np.ndarray
    fallback_note: str


@dataclass
class TrainingArtifacts:
    batch: dict[str, Any]
    videos: torch.Tensor
    context_videos: torch.Tensor
    captions: list[str]
    num_context_frames: torch.Tensor
    frame_valid_mask: torch.Tensor
    text_ctx: list[torch.Tensor]
    full_latents: list[torch.Tensor]
    context_latents: list[torch.Tensor]
    jepa_patch_tokens: torch.Tensor
    query_points_prior: torch.Tensor | None
    sam_prior_sources: list[str]
    sam_prompt_modes: list[str]
    sam_prior_debug: list[dict[str, Any]]
    vggt_out: Any
    cotracker_out: Any | None
    active_tracks: torch.Tensor
    active_visibility: torch.Tensor
    active_confidence: torch.Tensor
    active_track_image_hw: tuple[int, int]
    object_out: ObjectTokenOutput
    fused_context: list[torch.Tensor]
    track_alignment: TrackBoxAlignment | None
    track_box_loss: torch.Tensor | None
    track_iou_loss: torch.Tensor | None
    query_prior_visual: QueryPriorDebug | None


def _tensor_frame_to_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def _np_frame01_to_uint8(frame_chw_01: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_chw_01, dtype=np.float32)
    return np.transpose((frame.clip(0.0, 1.0) * 255.0).round().astype(np.uint8), (1, 2, 0))


def _to_pil(image_hwc: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image_hwc, dtype=np.uint8))


def _draw_gt_boxes(draw: ImageDraw.ImageDraw, gt_boxes_k4: torch.Tensor, width: int, height: int) -> None:
    for obj_idx, box in enumerate(gt_boxes_k4.detach().cpu().tolist()):
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            continue
        color = GT_PALETTE[obj_idx % len(GT_PALETTE)]
        draw.rectangle([x0 * width, y0 * height, x1 * width, y1 * height], outline=color, width=3)
        draw.text((x0 * width + 3, y0 * height + 3), f"gt{obj_idx}", fill=color)


def _draw_boxes_px(draw: ImageDraw.ImageDraw, boxes_xyxy: list[np.ndarray], labels: list[str] | None = None) -> None:
    labels = labels or []
    for idx, box in enumerate(boxes_xyxy):
        if box.shape != (4,):
            continue
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        if x1 <= x0 or y1 <= y0:
            continue
        color = OBJ_PALETTE[idx % len(OBJ_PALETTE)]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=4)
        label = labels[idx] if idx < len(labels) else f"obj{idx}"
        draw.text((x0 + 3, max(y0 + 3, 0)), label, fill=color)


def _draw_mask_points(draw: ImageDraw.ImageDraw, mask_hw: np.ndarray, color: str) -> None:
    ys, xs = np.where(np.asarray(mask_hw) > 0)
    if xs.size == 0:
        return
    step = max(1, xs.size // 800)
    for x, y in zip(xs[::step], ys[::step]):
        draw.point((float(x), float(y)), fill=color)


def _draw_query_points(draw: ImageDraw.ImageDraw, points_k2: np.ndarray) -> None:
    for idx, point in enumerate(np.asarray(points_k2, dtype=np.float32)):
        x, y = float(point[0]), float(point[1])
        color = OBJ_PALETTE[idx % len(OBJ_PALETTE)]
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=color, width=3)
        draw.text((x + 5, y + 3), f"q{idx}", fill=color)


def _draw_track_points(
    draw: ImageDraw.ImageDraw,
    tracks_k2: torch.Tensor,
    visibility_k: torch.Tensor,
    *,
    labels: list[str] | None = None,
) -> None:
    labels = labels or []
    for idx, point in enumerate(tracks_k2.detach().cpu().tolist()):
        x, y = float(point[0]), float(point[1])
        color = OBJ_PALETTE[idx % len(OBJ_PALETTE)]
        radius = 5
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=color, width=3)
        suffix = ""
        if float(visibility_k[idx].item()) < 0.5:
            suffix = " inv"
        label = labels[idx] if idx < len(labels) else f"t{idx}"
        draw.text((x + 5, y - 5), f"{label}{suffix}", fill=color)


def _scale_tracks_xy(tracks: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    out = tracks.clone()
    out[..., 0] *= float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    out[..., 1] *= float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    return out


def _scale_points_xy(points: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32).copy()
    out[..., 0] *= float(dst_hw[1]) / max(float(src_hw[1]), 1.0)
    out[..., 1] *= float(dst_hw[0]) / max(float(src_hw[0]), 1.0)
    return out


def _select_time_indices(num_frames: int, num_show: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames <= num_show:
        return list(range(num_frames))
    raw = np.linspace(0, num_frames - 1, num_show)
    return [int(round(v)) for v in raw.tolist()]


def _to_numpy_image_maybe(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        if image.ndim == 3 and image.shape[0] in {1, 3}:
            return _tensor_frame_to_uint8(image)
        return image.detach().cpu().numpy()
    return np.asarray(image)


class TrainingFlowInspector:
    def __init__(
        self,
        config_path: str | Path,
        *,
        sample_index: int = 0,
        device: str | torch.device | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_yaml_config(self.config_path)
        self.sample_index = int(sample_index)
        self.device = device
        self.trainer = ContextVideoTrainer(self.cfg, build_optimizer=False, device=device)
        self._batch: dict[str, Any] | None = None
        self._artifacts: TrainingArtifacts | None = None
        self._forward_metrics: dict[str, Any] | None = None

    @property
    def batch(self) -> dict[str, Any]:
        if self._batch is None:
            sample = self.trainer.dataset[self.sample_index]
            self._batch = collate_video_batch([sample])
        return self._batch

    def describe(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "sample_index": self.sample_index,
            "track_source": self.trainer.track_source,
            "sam2_prior_strategy": self.trainer.sam2_prior_strategy,
            "enable_sam2_priors": self.trainer.enable_sam2_priors,
            "caption": self.batch["caption"][0],
            "video_path": self.batch["video_path"][0],
            "num_context_frames": int(self.batch["num_context_frames"][0].item()),
            "dataset_type": self.cfg["data"]["dataset_type"],
        }

    def _collect_query_prior_visual(
        self,
        *,
        frames_tchw_01: np.ndarray,
        prompt_frame_idx: int,
        caption: str,
    ) -> QueryPriorDebug | None:
        if not self.trainer.enable_sam2_priors or self.trainer.sam2_tracker is None:
            return None

        strategy = self.trainer.sam2_prior_strategy
        fallback_note = ""
        object_records: list[QueryPriorObjectDebug] = []

        if strategy in {"grounded_text_multi", "text_multi", "grounded_text"}:
            text_prompt = _build_multi_object_prompt(caption)
            prompt_mode = "caption_gdino_multi"
            detector_error = ""
            detected_boxes = None
            if self.trainer.text_detector is not None and text_prompt.strip():
                try:
                    detection: DetectionPromptOutput = self.trainer.text_detector.detect(
                        frames_tchw_01[int(prompt_frame_idx)],
                        text_prompt,
                        guidance_box_xyxy=None,
                    )
                    if detection.boxes_xyxy.shape[0] > 0:
                        max_objects = int(self.cfg["model"].get("sam2_max_objects", 4))
                        detected_boxes = detection.boxes_xyxy[:max_objects]
                        prompt_mode = detection.prompt_mode
                except Exception as exc:  # pragma: no cover - diagnostic path
                    detector_error = f"{type(exc).__name__}: {exc}"
            if detected_boxes is None or detected_boxes.shape[0] == 0:
                motion_multi: MotionPromptBoxesOutput = build_motion_prompt_boxes(
                    frames_tchw_01,
                    max_boxes=int(self.cfg["model"].get("sam2_max_objects", 4)),
                )
                detected_boxes = motion_multi.boxes_xyxy
                prompt_frame_idx = int(motion_multi.prompt_frame_idx)
                prompt_mode = motion_multi.prompt_mode
                fallback_note = detector_error or "GroundingDINO 没给出框，回退到 motion prompt boxes"
            if detected_boxes is None or detected_boxes.shape[0] == 0:
                return QueryPriorDebug(
                    strategy=strategy,
                    prompt_frame_idx=int(prompt_frame_idx),
                    prompt_text=text_prompt,
                    prior_source="empty",
                    object_records=[],
                    query_points_px=np.zeros((0, 2), dtype=np.float32),
                    fallback_note=fallback_note or "没有检测到任何对象",
                )

            object_count = min(int(detected_boxes.shape[0]), int(self.trainer.vggt_adapter.num_queries))
            base = self.trainer.vggt_adapter.num_queries // max(object_count, 1)
            remainder = max(0, self.trainer.vggt_adapter.num_queries - base * object_count)
            sampled_queries = []
            for obj_idx, box_xyxy in enumerate(detected_boxes[:object_count]):
                sam_out: SAM2TrackOutput = self.trainer.sam2_tracker.track(
                    frames_tchw_01,
                    prompt_frame_idx=int(prompt_frame_idx),
                    prompt_box_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                    caption="",
                )
                alloc = base + (1 if obj_idx < remainder else 0)
                query_points_px, _ = build_vggt_query_prior(
                    sam_out.masks_thw,
                    sam_out.boxes_t4,
                    num_queries=alloc,
                )
                sampled_queries.append(query_points_px)
                object_records.append(
                    QueryPriorObjectDebug(
                        prompt_box_xyxy=np.asarray(box_xyxy, dtype=np.float32),
                        prompt_mode=sam_out.prompt_mode,
                        prompt_text=text_prompt,
                        masks_thw=sam_out.masks_thw,
                        boxes_t4=sam_out.boxes_t4,
                        query_points_px=query_points_px,
                        allocated_queries=int(alloc),
                    )
                )
            query_points_px = np.concatenate(sampled_queries, axis=0)[: self.trainer.vggt_adapter.num_queries].astype(np.float32)
            prior_source = f"grounded_sam_objects{len(object_records)}"
            return QueryPriorDebug(
                strategy=strategy,
                prompt_frame_idx=int(prompt_frame_idx),
                prompt_text=text_prompt,
                prior_source=prior_source,
                object_records=object_records,
                query_points_px=query_points_px,
                fallback_note=fallback_note,
            )

        prompt_box_xyxy = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=int(prompt_frame_idx))
        sam_out = self.trainer.sam2_tracker.track(
            frames_tchw_01,
            prompt_frame_idx=int(prompt_frame_idx),
            prompt_box_xyxy=prompt_box_xyxy,
            caption=caption,
        )
        query_points_px, prior_source = build_vggt_query_prior(
            sam_out.masks_thw,
            sam_out.boxes_t4,
            num_queries=self.trainer.vggt_adapter.num_queries,
        )
        object_records.append(
            QueryPriorObjectDebug(
                prompt_box_xyxy=np.asarray(prompt_box_xyxy, dtype=np.float32),
                prompt_mode=sam_out.prompt_mode,
                prompt_text=sam_out.prompt_text,
                masks_thw=sam_out.masks_thw,
                boxes_t4=sam_out.boxes_t4,
                query_points_px=query_points_px,
                allocated_queries=int(self.trainer.vggt_adapter.num_queries),
            )
        )
        return QueryPriorDebug(
            strategy=strategy,
            prompt_frame_idx=int(prompt_frame_idx),
            prompt_text=sam_out.prompt_text,
            prior_source=prior_source,
            object_records=object_records,
            query_points_px=query_points_px,
            fallback_note="",
        )

    @torch.no_grad()
    def collect_artifacts(self) -> TrainingArtifacts:
        if self._artifacts is not None:
            return self._artifacts

        batch = self.batch
        videos = batch["video"].to(self.trainer.device_obj)
        context_videos = batch["context_video"].to(self.trainer.device_obj)
        captions = list(batch["caption"])
        num_context_frames = batch["num_context_frames"].to(self.trainer.device_obj).long()
        target_context_frames = int(self.cfg["data"]["num_context_frames"])
        mutable_batch = dict(batch)
        if int(context_videos.shape[2]) != target_context_frames:
            raise RuntimeError(
                f"context_videos must be fixed-length before inspection; "
                f"got T={int(context_videos.shape[2])}, expected {target_context_frames}"
            )

        frame_valid_mask = self.trainer._frame_valid_mask(context_videos.shape[2], num_context_frames, self.trainer.device_obj)
        text_ctx = self.trainer._encode_text(captions)
        full_latents = self.trainer._encode_video_latents(videos)
        context_latents = self.trainer._encode_video_latents(context_videos)
        context_latent_batch = torch.stack(context_latents, dim=0)

        jepa_out = self.trainer.jepa_adapter(context_videos)
        frames_bthwc = context_videos.permute(0, 2, 3, 4, 1).float()
        frames_bthwc = (frames_bthwc + 1.0) / 2.0
        query_points_prior, sam_prior_sources, sam_prompt_modes, sam_prior_debug = self.trainer._maybe_build_query_priors(
            context_videos=context_videos,
            num_context_frames=num_context_frames,
            captions=captions,
        )
        vggt_out = self.trainer.vggt_adapter(
            frames_bthwc,
            query_points_prior=query_points_prior,
            query_image_hw=(context_videos.shape[-2], context_videos.shape[-1]) if query_points_prior is not None else None,
        )
        cotracker_out = None
        if self.trainer.cotracker_adapter is not None:
            cotracker_out = self.trainer.cotracker_adapter(
                frames_bthwc,
                query_points_prior=query_points_prior,
                query_image_hw=(context_videos.shape[-2], context_videos.shape[-1]) if query_points_prior is not None else None,
            )

        if cotracker_out is not None:
            active_tracks = cotracker_out.tracks
            active_visibility = cotracker_out.visibility
            active_confidence = cotracker_out.confidence
            active_track_image_hw = tuple(cotracker_out.image_hw)
        else:
            active_tracks = vggt_out.tracks
            active_visibility = vggt_out.visibility
            active_confidence = vggt_out.confidence
            active_track_image_hw = tuple(vggt_out.image_hw)

        object_out = self.trainer.object_pooler(
            jepa_patch_tokens=jepa_out.patch_tokens,
            context_latents=context_latent_batch,
            tracks=active_tracks,
            visibility=active_visibility,
            confidence=active_confidence,
            track_image_hw=active_track_image_hw,
            vggt_world_points=vggt_out.world_points,
            vggt_world_points_conf=vggt_out.world_points_conf,
            vggt_depth=vggt_out.depth,
            vggt_depth_conf=vggt_out.depth_conf,
            vggt_geometry_image_hw=vggt_out.image_hw,
            frame_valid_mask=frame_valid_mask,
        )
        fused_context = self.trainer.context_fuser(text_ctx, object_out.object_tokens)

        track_alignment = None
        track_box_loss = None
        track_iou_loss = None
        if "context_boxes" in mutable_batch:
            context_boxes = mutable_batch["context_boxes"].to(self.trainer.device_obj)
            scale_x = float(context_videos.shape[-1]) / float(active_track_image_hw[1])
            scale_y = float(context_videos.shape[-2]) / float(active_track_image_hw[0])
            tracks_native = active_tracks.clone()
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

        valid_frames = int(num_context_frames[0].item())
        frames_tchw_01 = ((context_videos[0, :, :valid_frames].permute(1, 0, 2, 3).float() + 1.0) / 2.0).detach().cpu().numpy()
        query_prior_visual = self._collect_query_prior_visual(
            frames_tchw_01=frames_tchw_01,
            prompt_frame_idx=max(valid_frames - 1, 0),
            caption=captions[0],
        )

        self._artifacts = TrainingArtifacts(
            batch=mutable_batch,
            videos=videos,
            context_videos=context_videos,
            captions=captions,
            num_context_frames=num_context_frames,
            frame_valid_mask=frame_valid_mask,
            text_ctx=text_ctx,
            full_latents=full_latents,
            context_latents=context_latents,
            jepa_patch_tokens=jepa_out.patch_tokens,
            query_points_prior=query_points_prior,
            sam_prior_sources=sam_prior_sources,
            sam_prompt_modes=sam_prompt_modes,
            sam_prior_debug=sam_prior_debug,
            vggt_out=vggt_out,
            cotracker_out=cotracker_out,
            active_tracks=active_tracks,
            active_visibility=active_visibility,
            active_confidence=active_confidence,
            active_track_image_hw=active_track_image_hw,
            object_out=object_out,
            fused_context=fused_context,
            track_alignment=track_alignment,
            track_box_loss=track_box_loss,
            track_iou_loss=track_iou_loss,
            query_prior_visual=query_prior_visual,
        )
        return self._artifacts

    @torch.no_grad()
    def run_forward_dry_run(self) -> dict[str, Any]:
        if self._forward_metrics is None:
            loss = self.trainer.forward(self.batch)
            self._forward_metrics = {
                "loss": float(loss.detach().item()),
                **self.trainer.last_train_metrics,
            }
        return self._forward_metrics

    def _native_hw(self) -> tuple[int, int]:
        artifacts = self.collect_artifacts()
        return (int(artifacts.context_videos.shape[-2]), int(artifacts.context_videos.shape[-1]))

    def _active_tracks_native(self) -> torch.Tensor:
        artifacts = self.collect_artifacts()
        native_hw = self._native_hw()
        if tuple(artifacts.active_track_image_hw) == native_hw:
            return artifacts.active_tracks.detach().cpu()
        return _scale_tracks_xy(artifacts.active_tracks.detach().cpu(), artifacts.active_track_image_hw, native_hw)

    def _vggt_tracks_native(self) -> torch.Tensor:
        artifacts = self.collect_artifacts()
        native_hw = self._native_hw()
        if tuple(artifacts.vggt_out.image_hw) == native_hw:
            return artifacts.vggt_out.tracks.detach().cpu()
        return _scale_tracks_xy(artifacts.vggt_out.tracks.detach().cpu(), artifacts.vggt_out.image_hw, native_hw)

    def plot_step1_training_inputs(self, num_show: int = 4) -> plt.Figure:
        artifacts = self.collect_artifacts()
        full_video = artifacts.videos[0].detach().cpu()
        context_video = artifacts.context_videos[0].detach().cpu()
        full_indices = _select_time_indices(int(full_video.shape[1]), num_show)
        context_indices = _select_time_indices(int(artifacts.num_context_frames[0].item()), num_show)
        fig, axes = plt.subplots(2, max(len(full_indices), len(context_indices)), figsize=(4.2 * num_show, 7.2))
        if num_show == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for col, frame_idx in enumerate(full_indices):
            image = _to_pil(_tensor_frame_to_uint8(full_video[:, frame_idx]))
            axes[0, col].imshow(image)
            axes[0, col].set_title(f"full video t={frame_idx}")
            axes[0, col].axis("off")
        for col, frame_idx in enumerate(context_indices):
            image = _to_pil(_tensor_frame_to_uint8(context_video[:, frame_idx]))
            draw = ImageDraw.Draw(image)
            if "context_boxes" in artifacts.batch:
                _draw_gt_boxes(draw, artifacts.batch["context_boxes"][0, frame_idx], image.size[0], image.size[1])
            axes[1, col].imshow(image)
            axes[1, col].set_title(f"context t={frame_idx}")
            axes[1, col].axis("off")
        for col in range(len(full_indices), axes.shape[1]):
            axes[0, col].axis("off")
        for col in range(len(context_indices), axes.shape[1]):
            axes[1, col].axis("off")
        fig.suptitle("Step 1. 训练输入: full video 与 context video", fontsize=16)
        fig.tight_layout()
        return fig

    def plot_step2_query_priors(self) -> plt.Figure:
        artifacts = self.collect_artifacts()
        qvis = artifacts.query_prior_visual
        if qvis is None:
            raise RuntimeError("当前配置没有启用 SAM2 priors，无法展示 Step 2。")

        frames_tchw_01 = ((artifacts.context_videos[0, :, : int(artifacts.num_context_frames[0].item())].permute(1, 0, 2, 3).float() + 1.0) / 2.0).detach().cpu().numpy()
        prompt_frame = _to_pil(_np_frame01_to_uint8(frames_tchw_01[qvis.prompt_frame_idx]))
        prompt_draw = ImageDraw.Draw(prompt_frame)
        _draw_boxes_px(prompt_draw, [record.prompt_box_xyxy for record in qvis.object_records], [f"prompt{idx}" for idx, _ in enumerate(qvis.object_records)])

        frame0_mask_img = _to_pil(_np_frame01_to_uint8(frames_tchw_01[0]))
        mask_draw = ImageDraw.Draw(frame0_mask_img)
        for idx, record in enumerate(qvis.object_records):
            _draw_mask_points(mask_draw, record.masks_thw[0], OBJ_PALETTE[idx % len(OBJ_PALETTE)])
            _draw_boxes_px(mask_draw, [record.boxes_t4[0]], [f"sam{idx}"])

        query_img = _to_pil(_np_frame01_to_uint8(frames_tchw_01[0]))
        query_draw = ImageDraw.Draw(query_img)
        _draw_query_points(query_draw, qvis.query_points_px)

        fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
        axes[0].imshow(prompt_frame)
        axes[0].set_title(f"Prompt Frame t={qvis.prompt_frame_idx}\n{qvis.strategy}")
        axes[1].imshow(frame0_mask_img)
        axes[1].set_title("SAM2 frame0 masks / boxes")
        axes[2].imshow(query_img)
        axes[2].set_title(f"Sampled query priors\n{qvis.prior_source}")
        for ax in axes:
            ax.axis("off")
        subtitle = qvis.prompt_text.strip() or artifacts.captions[0]
        note = qvis.fallback_note.strip()
        fig.suptitle(f"Step 2. Grounded-SAM / SAM2 prior -> query points\n{textwrap_trim(subtitle, 120)}", fontsize=15)
        if note:
            fig.text(0.02, 0.02, f"fallback: {note}", fontsize=10)
        fig.tight_layout()
        return fig

    def plot_step3_tracks_and_geometry(self, num_show: int = 4) -> plt.Figure:
        artifacts = self.collect_artifacts()
        context_video = artifacts.context_videos[0].detach().cpu()
        indices = _select_time_indices(int(artifacts.num_context_frames[0].item()), num_show)
        active_tracks = self._active_tracks_native()
        vggt_tracks = self._vggt_tracks_native()

        rows = 3 if artifacts.vggt_out.depth is not None and artifacts.vggt_out.world_points is not None else 2
        fig, axes = plt.subplots(rows, len(indices), figsize=(4.2 * len(indices), 4.2 * rows))
        if rows == 2:
            axes = np.asarray(axes).reshape(2, len(indices))

        for col, frame_idx in enumerate(indices):
            active_img = _to_pil(_tensor_frame_to_uint8(context_video[:, frame_idx]))
            active_draw = ImageDraw.Draw(active_img)
            _draw_track_points(active_draw, active_tracks[0, frame_idx], artifacts.active_visibility[0, frame_idx].detach().cpu())
            axes[0, col].imshow(active_img)
            axes[0, col].set_title(f"active tracks t={frame_idx}")
            axes[0, col].axis("off")

            vggt_img = _to_pil(_tensor_frame_to_uint8(context_video[:, frame_idx]))
            vggt_draw = ImageDraw.Draw(vggt_img)
            _draw_track_points(vggt_draw, vggt_tracks[0, frame_idx], artifacts.vggt_out.visibility[0, frame_idx].detach().cpu())
            axes[1, col].imshow(vggt_img)
            axes[1, col].set_title(f"VGGT tracks t={frame_idx}")
            axes[1, col].axis("off")

        if rows == 3:
            depth = artifacts.vggt_out.depth[0, indices[0]].detach().cpu()
            if depth.ndim == 3:
                depth = depth[..., 0]
            world = artifacts.vggt_out.world_points[0, indices[0]].detach().cpu()
            if world.ndim == 3 and world.shape[-1] == 3:
                world = torch.linalg.norm(world, dim=-1)
            axes[2, 0].imshow(depth, cmap="magma")
            axes[2, 0].set_title("VGGT depth")
            axes[2, 0].axis("off")
            if len(indices) > 1:
                axes[2, 1].imshow(world, cmap="viridis")
                axes[2, 1].set_title("VGGT world norm")
                axes[2, 1].axis("off")
            for col in range(2, len(indices)):
                axes[2, col].axis("off")

        fig.suptitle(
            f"Step 3. Tracks 与几何分支\nactive source = {self.trainer.track_source}, VGGT geometry 仍供 object pooler 使用",
            fontsize=15,
        )
        fig.tight_layout()
        return fig

    def plot_step4_object_pooler(self) -> plt.Figure:
        artifacts = self.collect_artifacts()
        names = ["jepa_tokens", "latent_tokens", "geom_tokens", "object_tokens"]
        tensors = [
            artifacts.object_out.jepa_tokens[0],
            artifacts.object_out.latent_tokens[0],
            artifacts.object_out.geom_tokens[0],
            artifacts.object_out.object_tokens[0],
        ]
        if artifacts.object_out.vggt_geom_tokens is not None:
            names.insert(3, "vggt_geom_tokens")
            tensors.insert(3, artifacts.object_out.vggt_geom_tokens[0])
        norms = np.stack([torch.linalg.norm(t.detach().cpu(), dim=-1).numpy() for t in tensors], axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        im = axes[0].imshow(norms, aspect="auto", cmap="YlGnBu")
        axes[0].set_yticks(range(len(names)))
        axes[0].set_yticklabels(names)
        axes[0].set_xlabel("object index")
        axes[0].set_title("per-object token norm")
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        visibility = artifacts.active_visibility[0].detach().cpu().numpy()
        confidence = artifacts.active_confidence[0].detach().cpu().numpy()
        axes[1].plot(visibility.mean(axis=0), label="visibility mean")
        axes[1].plot(confidence.mean(axis=0), label="confidence mean")
        axes[1].set_title("active track quality before pooler")
        axes[1].set_xlabel("object index")
        axes[1].legend()

        fig.suptitle("Step 4. Object Pooler: JEPA / latent / geom -> object tokens", fontsize=15)
        fig.tight_layout()
        return fig

    def plot_step5_fused_context(self, max_tokens: int = 96) -> plt.Figure:
        artifacts = self.collect_artifacts()
        text_ctx = artifacts.text_ctx[0].detach().cpu()
        object_tokens = artifacts.object_out.object_tokens[0].detach().cpu()
        fused = artifacts.fused_context[0].detach().cpu()
        keep = min(max_tokens, int(fused.shape[0]))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
        fused_heat = fused[:keep].abs().mean(dim=-1, keepdim=True).T.numpy()
        axes[0].imshow(fused_heat, aspect="auto", cmap="rocket")
        axes[0].set_title("fused_context token energy")
        axes[0].set_xlabel("token index")
        axes[0].set_yticks([])

        summary_lines = [
            f"text tokens: {int(text_ctx.shape[0])}",
            f"object tokens before fuse: {int(object_tokens.shape[0])}",
            f"fused tokens: {int(fused.shape[0])}",
            f"max context len: {int(self.trainer.context_fuser.max_context_len)}",
            f"min text tokens: {int(self.trainer.context_fuser.min_text_tokens)}",
            f"object gate: {float(self.trainer.context_fuser.object_gate.detach().cpu().item()):.4f}",
        ]
        axes[1].axis("off")
        axes[1].text(0.0, 0.95, "\n".join(summary_lines), va="top", fontsize=12, family="monospace")

        fig.suptitle("Step 5. Context Fuser: text tokens + object tokens -> fused context", fontsize=15)
        fig.tight_layout()
        return fig

    def plot_step6_track_supervision(self, frame_idx: int | None = None) -> plt.Figure:
        artifacts = self.collect_artifacts()
        if "context_boxes" not in artifacts.batch or artifacts.track_alignment is None:
            raise RuntimeError("当前样本没有 context_boxes，无法展示 GT 对齐。")

        valid_frames = int(artifacts.num_context_frames[0].item())
        if frame_idx is None:
            frame_idx = max(valid_frames - 1, 0)
        native_tracks = self._active_tracks_native()
        image = _to_pil(_tensor_frame_to_uint8(artifacts.context_videos[0, :, frame_idx].detach().cpu()))
        draw = ImageDraw.Draw(image)
        _draw_gt_boxes(draw, artifacts.batch["context_boxes"][0, frame_idx], image.size[0], image.size[1])
        labels = [f"q{idx}->gt{int(artifacts.track_alignment.matched_gt_indices[0, idx].item())}" for idx in range(native_tracks.shape[2])]
        _draw_track_points(draw, native_tracks[0, frame_idx], artifacts.active_visibility[0, frame_idx].detach().cpu(), labels=labels)

        pair_cost = artifacts.track_alignment.pair_cost[0].detach().cpu().numpy()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].imshow(image)
        axes[0].set_title(f"GT vs active tracks (t={frame_idx})")
        axes[0].axis("off")
        im = axes[1].imshow(pair_cost, aspect="auto", cmap="mako_r")
        axes[1].set_title("track -> GT pair cost")
        axes[1].set_xlabel("gt index")
        axes[1].set_ylabel("track index")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        if artifacts.track_box_loss is not None and artifacts.track_iou_loss is not None:
            fig.text(
                0.02,
                0.02,
                f"track_box_l1={float(artifacts.track_box_loss.item()):.4f} | track_iou_loss={float(artifacts.track_iou_loss.item()):.4f}",
                fontsize=11,
            )
        fig.suptitle("Step 6. 训练监督: active tracks 与 GT boxes 对齐", fontsize=15)
        fig.tight_layout()
        return fig

    def plot_step7_wan_loss_path(self) -> plt.Figure:
        artifacts = self.collect_artifacts()
        metrics = self.run_forward_dry_run()
        latent_clean = artifacts.full_latents[0].detach().cpu()
        context_latent = artifacts.context_latents[0].detach().cpu()
        context_mask_t, future_mask_t = latent_frame_mask(
            num_video_frames=int(artifacts.videos.shape[2]),
            num_context_frames=int(artifacts.num_context_frames[0].item()),
            vae_stride_t=self.trainer.bundle.config.vae_stride[0],
            device=latent_clean.device,
        )
        context_mask = broadcast_latent_mask(context_mask_t, latent_clean).detach().cpu().numpy()
        future_mask = broadcast_latent_mask(future_mask_t, latent_clean).detach().cpu().numpy()
        expanded_context = expand_context_latents_to_full(context_latent, latent_clean).detach().cpu()

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        axes[0].imshow(context_mask[0], aspect="auto", cmap="Greens")
        axes[0].set_title("context latent mask")
        axes[0].set_xlabel("time")
        axes[0].set_yticks([])

        axes[1].imshow(future_mask[0], aspect="auto", cmap="Oranges")
        axes[1].set_title("future latent mask")
        axes[1].set_xlabel("time")
        axes[1].set_yticks([])

        summary = [
            f"loss={metrics['loss']:.6f}",
            f"loss_main={metrics['train/loss_main']:.6f}",
            f"pred_abs_max={metrics['train/pred_abs_max']:.4f}",
            f"x_t_abs_max={metrics['train/x_t_abs_max']:.4f}",
            f"object_tokens_abs_max={metrics['train/object_tokens_abs_max']:.4f}",
            f"fused_context_abs_max={metrics['train/fused_context_abs_max']:.4f}",
            f"context_latent_shape={list(context_latent.shape)}",
            f"full_latent_shape={list(latent_clean.shape)}",
            f"expanded_context_shape={list(expanded_context.shape)}",
        ]
        axes[2].axis("off")
        axes[2].text(0.0, 0.95, "\n".join(summary), va="top", fontsize=11, family="monospace")

        fig.suptitle("Step 7. Wan 训练前向: latent mask / noise / loss dry-run", fontsize=15)
        fig.tight_layout()
        return fig

    def plot_cotracker_vggt_geometry_alignment_compare(self, frame_idx: int | None = None) -> plt.Figure:
        artifacts = self.collect_artifacts()
        if artifacts.cotracker_out is None:
            raise RuntimeError("当前样本没有走 CoTracker active tracks，无法做该对比。")
        if artifacts.vggt_out.depth is None or artifacts.vggt_out.world_points is None:
            raise RuntimeError("VGGT 几何输出不存在，无法做该对比。")

        valid_frames = int(artifacts.num_context_frames[0].item())
        if frame_idx is None:
            frame_idx = max(valid_frames - 1, 0)

        native_hw = tuple(int(v) for v in artifacts.active_track_image_hw)
        geometry_hw = tuple(int(v) for v in artifacts.vggt_out.image_hw)
        tracks_native = artifacts.active_tracks[0, frame_idx].detach().cpu()
        old_tracks = ObjectTubeProjector._resize_tracks_xy(
            tracks_native.unsqueeze(0),
            src_hw=native_hw,
            dst_hw=geometry_hw,
            align_corners=True,
        )[0].detach().cpu().numpy()
        new_tracks = ObjectTubeProjector._resize_tracks_xy(
            tracks_native.unsqueeze(0),
            src_hw=native_hw,
            dst_hw=geometry_hw,
            align_corners=False,
        )[0].detach().cpu().numpy()
        delta = np.linalg.norm(new_tracks - old_tracks, axis=-1)

        depth = artifacts.vggt_out.depth[0, frame_idx].detach().cpu()
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_img = depth.numpy()
        world = artifacts.vggt_out.world_points[0, frame_idx].detach().cpu()
        if world.ndim == 3 and world.shape[-1] == 3:
            world_img = torch.linalg.norm(world, dim=-1).numpy()
        else:
            world_img = np.asarray(world)

        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
        panels = [
            (depth_img, "VGGT depth"),
            (world_img, "VGGT world norm"),
        ]
        for ax, (panel, title) in zip(axes, panels):
            ax.imshow(panel, cmap="magma" if title == "VGGT depth" else "viridis")
            ax.scatter(old_tracks[:, 0], old_tracks[:, 1], s=46, c="#d62828", label="before", zorder=3)
            ax.scatter(new_tracks[:, 0], new_tracks[:, 1], s=30, c="#2a9d8f", label="after", zorder=4)
            for idx in range(old_tracks.shape[0]):
                ax.annotate(
                    "",
                    xy=(new_tracks[idx, 0], new_tracks[idx, 1]),
                    xytext=(old_tracks[idx, 0], old_tracks[idx, 1]),
                    arrowprops=dict(arrowstyle="->", color="#111111", lw=1.0),
                )
            ax.set_title(title)
            ax.axis("off")
        axes[0].legend(loc="lower right")
        fig.suptitle(
            "CoTracker active tracks -> VGGT geometry sampling\n"
            f"before=corner-aligned assumption, after=pixel-center aligned to VGGT resize | "
            f"mean shift={float(delta.mean()):.4f}px, max shift={float(delta.max()):.4f}px",
            fontsize=14,
        )
        fig.tight_layout()
        return fig


def textwrap_trim(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
