from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera_motion import CameraMotionResult, estimate_global_camera_motion
from .proxy_state import extract_primary_track
from .schemas import STATE_DIM, StateIndex
from .utils import require_torch

torch = require_torch()


@dataclass(slots=True)
class MaskTrackOutputs:
    masks: np.ndarray
    boxes: np.ndarray
    states: np.ndarray
    appearance: np.ndarray
    camera_full: np.ndarray
    camera_motion: CameraMotionResult
    prompt_boxes_xyxy: np.ndarray
    prompt_frame_idx: int
    prompt_mode: str


@dataclass(slots=True)
class DetectionPromptOutputs:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    phrases: list[str]
    prompt_frame_idx: int
    prompt_mode: str


def _mask_to_box_xyxy(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask > 0)
    if rows.size == 0 or cols.size == 0:
        return np.zeros((4,), dtype=np.float32)
    x0 = float(cols.min())
    y0 = float(rows.min())
    x1 = float(cols.max())
    y1 = float(rows.max())
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def _normalize_box(box_xyxy: np.ndarray, *, width: int, height: int) -> np.ndarray:
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return box_xyxy.astype(np.float32) / scale


def _box_area_ratio_xyxy(box_xyxy: np.ndarray, *, width: int, height: int) -> float:
    bw = max(float(box_xyxy[2] - box_xyxy[0]), 0.0)
    bh = max(float(box_xyxy[3] - box_xyxy[1]), 0.0)
    return float((bw * bh) / max(float(width * height), 1.0))


def _box_iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = [float(v) for v in box_a]
    bx0, by0, bx1, by1 = [float(v) for v in box_b]
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(inter_x1 - inter_x0, 0.0)
    inter_h = max(inter_y1 - inter_y0, 0.0)
    inter = inter_w * inter_h
    area_a = max(ax1 - ax0, 0.0) * max(ay1 - ay0, 0.0)
    area_b = max(bx1 - bx0, 0.0) * max(by1 - by0, 0.0)
    union = max(area_a + area_b - inter, 1e-6)
    return float(inter / union)


def _encode_mask_appearance(frames_tchw: np.ndarray, masks_tnhw: np.ndarray) -> np.ndarray:
    num_frames, _, height, width = frames_tchw.shape
    num_objects = int(masks_tnhw.shape[1])
    appearance = np.zeros((num_objects, 64), dtype=np.float32)
    for obj_idx in range(num_objects):
        for frame_idx in range(num_frames):
            mask = masks_tnhw[frame_idx, obj_idx] > 0
            if not np.any(mask):
                continue
            frame = frames_tchw[frame_idx]
            pixels = frame[:, mask]
            if pixels.size == 0:
                continue
            appearance[obj_idx, 0:3] = pixels.mean(axis=1)
            appearance[obj_idx, 3:6] = pixels.std(axis=1)
            box = _mask_to_box_xyxy(mask.astype(np.uint8))
            appearance[obj_idx, 6] = float((box[2] - box[0]) / max(width, 1))
            appearance[obj_idx, 7] = float((box[3] - box[1]) / max(height, 1))
            appearance[obj_idx, 8] = 1.0
            hist_r, _ = np.histogram(pixels[0], bins=8, range=(0.0, 1.0), density=True)
            hist_g, _ = np.histogram(pixels[1], bins=8, range=(0.0, 1.0), density=True)
            hist_b, _ = np.histogram(pixels[2], bins=8, range=(0.0, 1.0), density=True)
            appearance[obj_idx, 16:24] = hist_r.astype(np.float32)
            appearance[obj_idx, 24:32] = hist_g.astype(np.float32)
            appearance[obj_idx, 32:40] = hist_b.astype(np.float32)
            break
    return appearance


def masks_to_states_and_boxes(masks_tnhw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if masks_tnhw.ndim != 4:
        raise ValueError(f"expected masks with shape [T, N, H, W], got {tuple(masks_tnhw.shape)}")
    num_frames, num_objects, height, width = masks_tnhw.shape
    boxes = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    states = np.zeros((num_frames, num_objects, STATE_DIM), dtype=np.float32)
    mask_areas = masks_tnhw.reshape(num_frames, num_objects, -1).sum(axis=-1).astype(np.float32)
    visible_areas = mask_areas[mask_areas > 0.5]
    base_area = float(np.median(visible_areas)) if visible_areas.size > 0 else 1.0

    prev_centers: list[np.ndarray | None] = [None for _ in range(num_objects)]
    prev_depths = [1.0 for _ in range(num_objects)]
    prev_log_scales = [0.0 for _ in range(num_objects)]
    for frame_idx in range(num_frames):
        for obj_idx in range(num_objects):
            mask = masks_tnhw[frame_idx, obj_idx] > 0
            area = float(mask.sum())
            if area <= 0.5:
                states[frame_idx, obj_idx, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = (
                    prev_centers[obj_idx] if prev_centers[obj_idx] is not None else 0.0
                )
                states[frame_idx, obj_idx, StateIndex.DEPTH] = prev_depths[obj_idx]
                states[frame_idx, obj_idx, StateIndex.LOG_SCALE] = prev_log_scales[obj_idx]
                states[frame_idx, obj_idx, StateIndex.VISIBILITY] = 0.0
                states[frame_idx, obj_idx, StateIndex.EXISTENCE] = 0.0
                states[frame_idx, obj_idx, StateIndex.CONFIDENCE] = 0.0
                continue

            rows, cols = np.where(mask)
            cx = float(cols.mean() / max(width, 1))
            cy = float(rows.mean() / max(height, 1))
            center = np.asarray([cx, cy], dtype=np.float32)
            box_xyxy = _mask_to_box_xyxy(mask.astype(np.uint8))
            boxes[frame_idx, obj_idx] = _normalize_box(box_xyxy, width=width, height=height)
            log_scale = float(np.log(max(area / float(height * width), 1e-6)))
            depth = float(np.sqrt(max(base_area, 1e-6) / max(area, 1e-6)))
            velocity = np.zeros((2,), dtype=np.float32)
            depth_velocity = 0.0
            if prev_centers[obj_idx] is not None:
                velocity = center - prev_centers[obj_idx]
                depth_velocity = depth - prev_depths[obj_idx]
            states[frame_idx, obj_idx, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = center
            states[frame_idx, obj_idx, StateIndex.DEPTH] = depth
            states[frame_idx, obj_idx, StateIndex.LOG_SCALE] = log_scale
            states[frame_idx, obj_idx, StateIndex.VEL_X:StateIndex.VEL_Y + 1] = velocity
            states[frame_idx, obj_idx, StateIndex.DEPTH_VEL] = depth_velocity
            states[frame_idx, obj_idx, StateIndex.VISIBILITY] = 1.0
            states[frame_idx, obj_idx, StateIndex.EXISTENCE] = 1.0
            states[frame_idx, obj_idx, StateIndex.CONFIDENCE] = 1.0
            prev_centers[obj_idx] = center
            prev_depths[obj_idx] = depth
            prev_log_scales[obj_idx] = log_scale
    return states.astype(np.float32), boxes.astype(np.float32)


def _save_frames_to_dir(frames_tchw: np.ndarray, frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames_tchw):
        rgb = np.transpose(np.clip(frame, 0.0, 1.0), (1, 2, 0))
        bgr = cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), bgr)
        if not ok:
            raise RuntimeError(f"failed to write frame {idx} to {frame_dir}")


def build_proxy_prompt_box(
    frames_tchw: np.ndarray,
    *,
    prompt_frame_idx: int,
    history_window: int = 8,
) -> np.ndarray:
    start = max(0, int(prompt_frame_idx) - int(history_window) + 1)
    clip = frames_tchw[start : prompt_frame_idx + 1]
    track = extract_primary_track(clip)
    box = track.boxes[-1, 0].copy()
    height = int(frames_tchw.shape[2])
    width = int(frames_tchw.shape[3])
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return (box * scale).astype(np.float32)


class GroundingDINOTextDetector:
    def __init__(
        self,
        *,
        repo_root: str | Path = "/home/gaoya/Grounded-SAM-2-main",
        config_path: str | Path = "/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/GroundingDINO_SwinT_OGC.cfg.py",
        checkpoint_path: str | Path = "/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/groundingdino_swint_ogc.pth",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        max_boxes: int = 4,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.max_boxes = int(max_boxes)
        self._model = None
        self._predict = None
        self._model_api = None
        self._phrase_blacklist = (
            "static shot",
            "no camera movement",
            "camera movement",
            "plain wall",
            "background",
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        from grounding_dino.groundingdino.util.inference import Model, load_model, predict

        self._model = load_model(
            str(self.config_path),
            str(self.checkpoint_path),
            device=self.device,
        )
        self._predict = predict
        self._model_api = Model

    @staticmethod
    def _cxcywh_to_xyxy_pixels(box_cxcywh: np.ndarray, *, width: int, height: int) -> np.ndarray:
        cx, cy, bw, bh = [float(value) for value in box_cxcywh]
        x0 = (cx - 0.5 * bw) * width
        y0 = (cy - 0.5 * bh) * height
        x1 = (cx + 0.5 * bw) * width
        y1 = (cy + 0.5 * bh) * height
        return np.asarray([x0, y0, x1, y1], dtype=np.float32)

    def _filter_candidates(
        self,
        *,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        phrases: list[str],
        width: int,
        height: int,
        guidance_box_xyxy: np.ndarray | None,
        max_area_ratio: float = 0.72,
        min_guidance_iou: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        kept_boxes: list[np.ndarray] = []
        kept_scores: list[float] = []
        kept_phrases: list[str] = []
        for box_xyxy, score, phrase in zip(boxes_xyxy, scores, phrases):
            phrase_norm = str(phrase).strip().lower()
            if any(blocked in phrase_norm for blocked in self._phrase_blacklist):
                continue
            if _box_area_ratio_xyxy(box_xyxy, width=width, height=height) > max_area_ratio:
                continue
            kept_boxes.append(np.asarray(box_xyxy, dtype=np.float32))
            kept_scores.append(float(score))
            kept_phrases.append(str(phrase))

        if not kept_boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                [],
            )

        boxes_arr = np.stack(kept_boxes, axis=0).astype(np.float32)
        scores_arr = np.asarray(kept_scores, dtype=np.float32)
        phrases_arr = kept_phrases
        if guidance_box_xyxy is not None:
            overlaps = np.asarray(
                [_box_iou_xyxy(box_xyxy, guidance_box_xyxy) for box_xyxy in boxes_arr],
                dtype=np.float32,
            )
            if np.any(overlaps >= float(min_guidance_iou)):
                keep = overlaps >= float(min_guidance_iou)
                boxes_arr = boxes_arr[keep]
                scores_arr = scores_arr[keep]
                phrases_arr = [phrase for phrase, flag in zip(phrases_arr, keep.tolist()) if flag]
        return boxes_arr, scores_arr, phrases_arr

    def detect(
        self,
        frame_chw: np.ndarray,
        caption: str,
        *,
        guidance_box_xyxy: np.ndarray | None = None,
    ) -> DetectionPromptOutputs:
        self._load()
        if not caption.strip():
            raise ValueError("caption-based detection requires a non-empty caption")
        rgb = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
        bgr = cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        processed = self._model_api.preprocess_image(bgr).to(self.device)
        boxes, logits, phrases = self._predict(
            model=self._model,
            image=processed,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        scores_np = logits.detach().cpu().numpy().astype(np.float32)
        if boxes_np.size == 0:
            return DetectionPromptOutputs(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                phrases=[],
                prompt_frame_idx=0,
                prompt_mode="caption_gdino",
            )
        order = np.argsort(-scores_np)
        if self.max_boxes > 0:
            order = order[: self.max_boxes]
        height, width = bgr.shape[:2]
        xyxy = np.stack(
            [self._cxcywh_to_xyxy_pixels(boxes_np[idx], width=width, height=height) for idx in order],
            axis=0,
        )
        phrases_sorted = [str(phrases[idx]) for idx in order]
        scores_sorted = scores_np[order]
        xyxy, scores_sorted, phrases_sorted = self._filter_candidates(
            boxes_xyxy=xyxy,
            scores=scores_sorted,
            phrases=phrases_sorted,
            width=width,
            height=height,
            guidance_box_xyxy=guidance_box_xyxy,
        )
        return DetectionPromptOutputs(
            boxes_xyxy=xyxy.astype(np.float32),
            scores=scores_sorted.astype(np.float32),
            phrases=phrases_sorted,
            prompt_frame_idx=0,
            prompt_mode="caption_gdino",
        )


def build_caption_prompt_boxes(
    frames_tchw: np.ndarray,
    *,
    prompt_frame_idx: int,
    caption: str,
    detector: GroundingDINOTextDetector,
    guidance_box_xyxy: np.ndarray | None = None,
) -> DetectionPromptOutputs:
    outputs = detector.detect(
        frames_tchw[int(prompt_frame_idx)],
        caption,
        guidance_box_xyxy=guidance_box_xyxy,
    )
    return DetectionPromptOutputs(
        boxes_xyxy=outputs.boxes_xyxy.astype(np.float32),
        scores=outputs.scores.astype(np.float32),
        phrases=list(outputs.phrases),
        prompt_frame_idx=int(prompt_frame_idx),
        prompt_mode="caption_gdino",
    )


class SAM2VideoMaskTracker:
    def __init__(
        self,
        *,
        device: str = "cuda",
        model_id: str | None = "facebook/sam2.1-hiera-small",
        model_cfg: str | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.model_cfg = model_cfg
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._video_predictor = None

    def _resolve_model_cfg(self) -> str:
        if self.model_cfg is None:
            raise ValueError("local SAM2 build requires model_cfg")
        model_cfg = str(self.model_cfg)
        if model_cfg.startswith("configs/"):
            return model_cfg
        cfg_path = Path(model_cfg)
        if cfg_path.is_file():
            parts = cfg_path.parts
            for idx, part in enumerate(parts):
                if part == "configs" and idx + 1 < len(parts):
                    return "/".join(parts[idx:])
            name = cfg_path.name
            if name.startswith("sam2.1_"):
                return f"configs/sam2.1/{name}"
            if name.startswith("sam2_"):
                return f"configs/sam2/{name}"
            return name
        return model_cfg

    def _build_video_predictor(self):
        from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_hf

        if self._video_predictor is not None:
            return self._video_predictor
        if self.checkpoint_path is not None:
            if self.model_cfg is None:
                raise ValueError("local SAM2 build requires both model_cfg and checkpoint_path")
            predictor = build_sam2_video_predictor(
                self._resolve_model_cfg(),
                str(self.checkpoint_path),
                device=self.device,
            )
        elif self.model_id:
            predictor = build_sam2_video_predictor_hf(self.model_id, device=self.device)
        else:
            raise ValueError("SAM2 build requires either model_id or local model_cfg + checkpoint_path")
        predictor.fill_hole_area = 0
        self._video_predictor = predictor
        return predictor

    def track_from_boxes(
        self,
        frames_tchw: np.ndarray,
        *,
        prompt_frame_idx: int,
        boxes_xyxy: np.ndarray,
    ) -> np.ndarray:
        predictor = self._build_video_predictor()
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
        if boxes_xyxy.ndim == 1:
            boxes_xyxy = boxes_xyxy[None]
        if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[1] != 4:
            raise ValueError(f"expected boxes_xyxy [N,4], got {tuple(boxes_xyxy.shape)}")

        num_frames = int(frames_tchw.shape[0])
        num_objects = int(boxes_xyxy.shape[0])
        forward_masks = np.zeros((num_frames, num_objects, frames_tchw.shape[2], frames_tchw.shape[3]), dtype=np.uint8)
        reverse_masks = np.zeros_like(forward_masks)
        with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmp_dir:
            frame_dir = Path(tmp_dir)
            _save_frames_to_dir(frames_tchw, frame_dir)
            with torch.inference_mode():
                for direction in ("forward", "reverse"):
                    state = predictor.init_state(
                        video_path=str(frame_dir),
                        offload_video_to_cpu=True,
                        async_loading_frames=False,
                    )
                    for obj_idx, box in enumerate(boxes_xyxy, start=1):
                        predictor.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=int(prompt_frame_idx),
                            obj_id=int(obj_idx),
                            box=box,
                        )
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        state,
                        start_frame_idx=int(prompt_frame_idx),
                        reverse=(direction == "reverse"),
                    ):
                        for local_idx, out_obj_id in enumerate(out_obj_ids):
                            mask = (out_mask_logits[local_idx] > 0.0).detach().cpu().numpy().squeeze(0).astype(np.uint8)
                            target = reverse_masks if direction == "reverse" else forward_masks
                            target[int(out_frame_idx), int(out_obj_id) - 1] = mask
        merged = forward_masks.copy()
        before_prompt = np.arange(num_frames) < int(prompt_frame_idx)
        merged[before_prompt] = reverse_masks[before_prompt]
        merged[int(prompt_frame_idx)] = np.maximum(forward_masks[int(prompt_frame_idx)], reverse_masks[int(prompt_frame_idx)])
        return merged.astype(np.uint8)


def build_mask_track_outputs(
    frames_tchw: np.ndarray,
    *,
    prompt_frame_idx: int,
    prompt_boxes_xyxy: np.ndarray,
    prompt_mode: str,
    tracker: SAM2VideoMaskTracker,
) -> MaskTrackOutputs:
    masks = tracker.track_from_boxes(
        frames_tchw,
        prompt_frame_idx=int(prompt_frame_idx),
        boxes_xyxy=prompt_boxes_xyxy,
    )
    states, boxes = masks_to_states_and_boxes(masks)
    appearance = _encode_mask_appearance(frames_tchw, masks)
    camera_motion = estimate_global_camera_motion(frames_tchw)
    return MaskTrackOutputs(
        masks=masks.astype(np.uint8),
        boxes=boxes.astype(np.float32),
        states=states.astype(np.float32),
        appearance=appearance.astype(np.float32),
        camera_full=camera_motion.camera_features.astype(np.float32),
        camera_motion=camera_motion,
        prompt_boxes_xyxy=np.asarray(prompt_boxes_xyxy, dtype=np.float32),
        prompt_frame_idx=int(prompt_frame_idx),
        prompt_mode=str(prompt_mode),
    )


def load_prompt_boxes_from_json(path: str | Path) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    boxes_by_key: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None]
        if array.ndim != 2 or array.shape[1] != 4:
            raise ValueError(f"invalid box payload for {key!r}: expected [N,4], got {tuple(array.shape)}")
        boxes_by_key[str(key)] = array
    return boxes_by_key
