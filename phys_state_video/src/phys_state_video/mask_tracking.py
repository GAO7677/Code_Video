from __future__ import annotations

import json
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
