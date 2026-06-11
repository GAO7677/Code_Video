from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


SAM2_REPO_ROOT = Path('/home/gaoya/Grounded-SAM-2-main')
if str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))


@dataclass
class SAM2TrackOutput:
    prompt_box_xyxy: np.ndarray
    masks_thw: np.ndarray
    boxes_t4: np.ndarray
    boxes_norm_t4: np.ndarray


def _box_center(box_xyxy: np.ndarray) -> np.ndarray:
    return np.asarray([(box_xyxy[0] + box_xyxy[2]) * 0.5, (box_xyxy[1] + box_xyxy[3]) * 0.5], dtype=np.float32)


def _save_frames_to_dir(frames_tchw_01: np.ndarray, frame_dir: Path) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames_tchw_01):
        rgb = np.transpose(np.clip(frame, 0.0, 1.0), (1, 2, 0))
        bgr = cv2.cvtColor((rgb * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(str(frame_dir / f"{idx:05d}.jpg"), bgr)
        if not ok:
            raise RuntimeError(f"failed to write frame {idx} to {frame_dir}")


def _mask_to_box_xyxy(mask_hw: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask_hw > 0)
    if rows.size == 0 or cols.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())], dtype=np.float32)


def build_motion_prompt_box(frames_tchw_01: np.ndarray, prompt_frame_idx: int, history_window: int = 8) -> np.ndarray:
    start = max(0, int(prompt_frame_idx) - int(history_window) + 1)
    clip = frames_tchw_01[start : prompt_frame_idx + 1]
    num_frames, _, height, width = clip.shape
    grays = []
    for frame in clip:
        image = np.transpose((frame * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grays.append(cv2.GaussianBlur(gray, (5, 5), 0))

    ref = grays[0]
    prev = ref
    prev_center: np.ndarray | None = None
    prev_box: np.ndarray | None = None
    prev_area = 0.0
    frame_area = float(height * width)
    min_area = max(24.0, frame_area * 0.00018)
    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_big = np.ones((7, 7), dtype=np.uint8)
    best_box = np.zeros((4,), dtype=np.float32)

    for gray in grays:
        diff_ref = cv2.absdiff(gray, ref)
        diff_prev = cv2.absdiff(gray, prev)
        motion = cv2.max(diff_ref, diff_prev)
        _, mask = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
            center = _box_center(box)
            score = area
            if prev_center is not None:
                dx = (center[0] - prev_center[0]) / max(width, 1)
                dy = (center[1] - prev_center[1]) / max(height, 1)
                score -= 0.35 * frame_area * float(np.hypot(dx, dy))
            if prev_box is not None:
                x0 = max(box[0], prev_box[0])
                y0 = max(box[1], prev_box[1])
                x1 = min(box[2], prev_box[2])
                y1 = min(box[3], prev_box[3])
                inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                union = max(area + prev_area - inter, 1e-6)
                score += 0.12 * frame_area * (inter / union)
            if best_score is None or score > best_score:
                best_score = score
                best_box = box
                prev_box = box
                prev_area = area
                prev_center = center
        prev = gray
    return best_box.astype(np.float32)


class SAM2MotionTracker:
    def __init__(
        self,
        *,
        device: str = 'cuda',
        model_cfg: str = '/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml',
        checkpoint_path: str = '/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt',
    ) -> None:
        self.device = device
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self._predictor = None

    def _resolve_model_cfg(self) -> str:
        cfg_path = Path(self.model_cfg)
        if cfg_path.is_file():
            parts = cfg_path.parts
            for idx, part in enumerate(parts):
                if part == 'configs' and idx + 1 < len(parts):
                    return '/'.join(parts[idx:])
            name = cfg_path.name
            if name.startswith('sam2.1_'):
                return f'configs/sam2.1/{name}'
            if name.startswith('sam2_'):
                return f'configs/sam2/{name}'
            return name
        return str(self.model_cfg)

    def _build(self):
        if self._predictor is not None:
            return self._predictor
        from sam2.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            self._resolve_model_cfg(),
            str(self.checkpoint_path),
            device=self.device,
        )
        predictor.fill_hole_area = 0
        self._predictor = predictor
        return predictor

    def track(self, frames_tchw_01: np.ndarray, prompt_frame_idx: int, prompt_box_xyxy: np.ndarray) -> SAM2TrackOutput:
        predictor = self._build()
        num_frames, _, height, width = frames_tchw_01.shape
        forward_masks = np.zeros((num_frames, height, width), dtype=np.uint8)
        reverse_masks = np.zeros_like(forward_masks)

        with tempfile.TemporaryDirectory(prefix='sam2_frames_') as tmp_dir:
            frame_dir = Path(tmp_dir)
            _save_frames_to_dir(frames_tchw_01, frame_dir)
            with torch.inference_mode():
                for direction in ('forward', 'reverse'):
                    state = predictor.init_state(
                        video_path=str(frame_dir),
                        offload_video_to_cpu=True,
                        async_loading_frames=False,
                    )
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=int(prompt_frame_idx),
                        obj_id=1,
                        box=prompt_box_xyxy.astype(np.float32),
                    )
                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        state,
                        start_frame_idx=int(prompt_frame_idx),
                        reverse=(direction == 'reverse'),
                    ):
                        mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze(0).astype(np.uint8)
                        target = reverse_masks if direction == 'reverse' else forward_masks
                        target[int(out_frame_idx)] = mask

        masks = forward_masks.copy()
        before_prompt = np.arange(num_frames) < int(prompt_frame_idx)
        masks[before_prompt] = reverse_masks[before_prompt]
        masks[int(prompt_frame_idx)] = np.maximum(forward_masks[int(prompt_frame_idx)], reverse_masks[int(prompt_frame_idx)])
        boxes_t4 = np.stack([_mask_to_box_xyxy(mask) for mask in masks], axis=0).astype(np.float32)
        scale = np.asarray([width, height, width, height], dtype=np.float32)
        boxes_norm_t4 = boxes_t4 / scale
        return SAM2TrackOutput(
            prompt_box_xyxy=prompt_box_xyxy.astype(np.float32),
            masks_thw=masks.astype(np.uint8),
            boxes_t4=boxes_t4,
            boxes_norm_t4=boxes_norm_t4.astype(np.float32),
        )
