from __future__ import annotations

import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from code_vjepa_vggt.utils.object_priors import sample_points_from_mask


SAM2_REPO_ROOT = Path('/home/gaoya/Grounded-SAM-2-main')
if str(SAM2_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM2_REPO_ROOT))


@dataclass
class SAM2TrackOutput:
    prompt_box_xyxy: np.ndarray
    masks_thw: np.ndarray
    boxes_t4: np.ndarray
    boxes_norm_t4: np.ndarray
    prompt_mode: str = "proxy_box"
    prompt_text: str = ""


@dataclass
class DetectionPromptOutput:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    phrases: list[str]
    prompt_mode: str


@dataclass
class MotionPromptBoxesOutput:
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    prompt_frame_idx: int
    prompt_mode: str


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


def _box_valid(box_xyxy: np.ndarray) -> bool:
    if box_xyxy.shape != (4,):
        return False
    return bool(float(box_xyxy[2] - box_xyxy[0]) > 1e-6 and float(box_xyxy[3] - box_xyxy[1]) > 1e-6)


def _frame_to_rgb_uint8(frame_chw_01: np.ndarray) -> np.ndarray:
    return np.transpose(
        (np.clip(frame_chw_01, 0.0, 1.0) * 255.0).round().astype(np.uint8),
        (1, 2, 0),
    )


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


def _caption_to_object_prompt(caption: str) -> str:
    tokens = re.findall(r"[a-zA-Z]+", str(caption).lower())
    preferred = ("sphere", "capsule", "cylinder", "box", "cube", "ball", "block")
    for token in reversed(tokens):
        if token in preferred:
            return f"{token}."
    if tokens:
        return f"{tokens[-1]}."
    return ""


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


def _motion_mask_for_prompt_frame(
    frames_tchw_01: np.ndarray,
    prompt_frame_idx: int,
    history_window: int = 8,
) -> tuple[np.ndarray, float]:
    start = max(0, int(prompt_frame_idx) - int(history_window) + 1)
    clip = frames_tchw_01[start : int(prompt_frame_idx) + 1]
    if clip.shape[0] < 2:
        height, width = int(frames_tchw_01.shape[-2]), int(frames_tchw_01.shape[-1])
        return np.zeros((height, width), dtype=np.uint8), 0.0

    grays = []
    for frame in clip:
        image = np.transpose((frame * 255.0).clip(0, 255).astype(np.uint8), (1, 2, 0))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grays.append(cv2.GaussianBlur(gray, (5, 5), 0))

    ref = grays[0]
    prev = ref
    best_mask = np.zeros_like(ref, dtype=np.uint8)
    best_score = 0.0
    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_big = np.ones((7, 7), dtype=np.uint8)
    for gray in grays[1:]:
        diff_ref = cv2.absdiff(gray, ref)
        diff_prev = cv2.absdiff(gray, prev)
        motion = cv2.max(diff_ref, diff_prev)
        _, mask = cv2.threshold(motion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)
        score = float((mask > 0).sum())
        if score > best_score:
            best_score = score
            best_mask = mask.astype(np.uint8)
        prev = gray
    return best_mask, best_score


def _extract_motion_components(mask_hw: np.ndarray) -> list[dict[str, np.ndarray | float]]:
    mask_u8 = (mask_hw > 0).astype(np.uint8)
    if int(mask_u8.sum()) <= 0:
        return []

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    total_fg_area = max(int(mask_u8.sum()), 1)
    min_area = max(24, int(round(total_fg_area * 0.02)))
    components: list[dict[str, np.ndarray | float]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp_mask = (labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
        max_dist = float(dist.max())
        safe_thresh = max(1.5, 0.25 * max_dist)
        safe_mask = dist >= safe_thresh
        safe_area = float(safe_mask.sum())
        confidence = safe_area / max(float(area), 1.0)
        score = max(float(area) * max(confidence, 1.0e-3), 1.0)
        x, y, w, h = cv2.boundingRect(comp_mask)
        box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
        components.append(
            {
                "mask": comp_mask,
                "box": box,
                "area": float(area),
                "confidence": float(confidence),
                "score": float(score),
            }
        )
    return sorted(components, key=lambda item: float(item["score"]), reverse=True)


def build_motion_prompt_boxes(
    frames_tchw_01: np.ndarray,
    *,
    history_window: int = 8,
    max_boxes: int = 4,
    top_frames: int = 3,
) -> MotionPromptBoxesOutput:
    num_frames, _, height, width = frames_tchw_01.shape
    if num_frames <= 0:
        return MotionPromptBoxesOutput(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            prompt_frame_idx=0,
            prompt_mode="motion_empty",
        )

    candidate_frames: list[tuple[float, int, np.ndarray]] = []
    for frame_idx in range(1, num_frames):
        mask_hw, score = _motion_mask_for_prompt_frame(
            frames_tchw_01,
            prompt_frame_idx=frame_idx,
            history_window=history_window,
        )
        if score > 0:
            candidate_frames.append((float(score), int(frame_idx), mask_hw))

    if not candidate_frames:
        fallback_idx = max(num_frames // 2, 0)
        fallback_box = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=fallback_idx, history_window=history_window)
        if _box_valid(fallback_box):
            return MotionPromptBoxesOutput(
                boxes_xyxy=fallback_box.reshape(1, 4).astype(np.float32),
                scores=np.asarray([1.0], dtype=np.float32),
                prompt_frame_idx=int(fallback_idx),
                prompt_mode="motion_single",
            )
        return MotionPromptBoxesOutput(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            prompt_frame_idx=int(fallback_idx),
            prompt_mode="motion_empty",
        )

    candidate_frames.sort(key=lambda item: item[0], reverse=True)
    prompt_frame_idx = int(candidate_frames[0][1])
    boxes: list[np.ndarray] = []
    scores: list[float] = []
    for motion_score, frame_idx, mask_hw in candidate_frames[: max(1, int(top_frames))]:
        components = _extract_motion_components(mask_hw)
        for component in components:
            box = np.asarray(component["box"], dtype=np.float32)
            score = float(component["score"]) + 0.01 * float(motion_score)
            if any(_box_iou_xyxy(box, existing) >= 0.75 for existing in boxes):
                continue
            boxes.append(box)
            scores.append(score)
            if len(boxes) >= max_boxes:
                break
        if len(boxes) >= max_boxes:
            break

    if not boxes:
        fallback_box = build_motion_prompt_box(frames_tchw_01, prompt_frame_idx=prompt_frame_idx, history_window=history_window)
        if _box_valid(fallback_box):
            return MotionPromptBoxesOutput(
                boxes_xyxy=fallback_box.reshape(1, 4).astype(np.float32),
                scores=np.asarray([candidate_frames[0][0]], dtype=np.float32),
                prompt_frame_idx=prompt_frame_idx,
                prompt_mode="motion_single",
            )
        return MotionPromptBoxesOutput(
            boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            prompt_frame_idx=prompt_frame_idx,
            prompt_mode="motion_empty",
        )

    boxes_arr = np.stack(boxes, axis=0).astype(np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)
    order = np.argsort(-scores_arr)
    boxes_arr = boxes_arr[order][:max_boxes]
    scores_arr = scores_arr[order][:max_boxes]
    return MotionPromptBoxesOutput(
        boxes_xyxy=boxes_arr.astype(np.float32),
        scores=scores_arr.astype(np.float32),
        prompt_frame_idx=prompt_frame_idx,
        prompt_mode="motion_multi",
    )


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
            "background",
            "plain wall",
            "wall",
            "floor",
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
        if guidance_box_xyxy is not None and _box_valid(guidance_box_xyxy):
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
        frame_chw_01: np.ndarray,
        text_prompt: str,
        *,
        guidance_box_xyxy: np.ndarray | None = None,
    ) -> DetectionPromptOutput:
        self._load()
        if not text_prompt.strip():
            return DetectionPromptOutput(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                phrases=[],
                prompt_mode="empty_text",
            )
        bgr = cv2.cvtColor(_frame_to_rgb_uint8(frame_chw_01), cv2.COLOR_RGB2BGR)
        processed = self._model_api.preprocess_image(bgr).to(self.device)
        boxes, logits, phrases = self._predict(
            model=self._model,
            image=processed,
            caption=text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
        scores_np = logits.detach().cpu().numpy().astype(np.float32)
        if boxes_np.size == 0:
            return DetectionPromptOutput(
                boxes_xyxy=np.zeros((0, 4), dtype=np.float32),
                scores=np.zeros((0,), dtype=np.float32),
                phrases=[],
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
        return DetectionPromptOutput(
            boxes_xyxy=xyxy.astype(np.float32),
            scores=scores_sorted.astype(np.float32),
            phrases=phrases_sorted,
            prompt_mode="caption_gdino",
        )


class SAM2MotionTracker:
    def __init__(
        self,
        *,
        device: str = 'cuda',
        model_cfg: str = '/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml',
        checkpoint_path: str = '/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt',
        segment_len: int = 8,
        enable_text_prompt: bool = True,
    ) -> None:
        self.device = device
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.segment_len = int(segment_len)
        self.enable_text_prompt = bool(enable_text_prompt)
        self._predictor = None
        self._image_model = None
        self._image_predictor = None
        self._text_detector = None

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

    def _build_image_predictor(self):
        if self._image_predictor is not None:
            return self._image_predictor
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        image_model = build_sam2(
            self._resolve_model_cfg(),
            str(self.checkpoint_path),
            device=self.device,
        )
        self._image_model = image_model
        self._image_predictor = SAM2ImagePredictor(image_model)
        return self._image_predictor

    def _build_text_detector(self):
        if self._text_detector is not None:
            return self._text_detector
        self._text_detector = GroundingDINOTextDetector(
            device=self.device,
            max_boxes=1,
        )
        return self._text_detector

    def _select_prompt(
        self,
        frames_tchw_01: np.ndarray,
        *,
        prompt_frame_idx: int,
        caption: str,
        guidance_box_xyxy: np.ndarray,
    ) -> tuple[np.ndarray, str, str]:
        if self.enable_text_prompt:
            text_prompt = _caption_to_object_prompt(caption)
            if text_prompt:
                detection = self._build_text_detector().detect(
                    frames_tchw_01[int(prompt_frame_idx)],
                    text_prompt,
                    guidance_box_xyxy=guidance_box_xyxy,
                )
                if detection.boxes_xyxy.shape[0] > 0:
                    return detection.boxes_xyxy[0].astype(np.float32), detection.prompt_mode, text_prompt
                raise RuntimeError("GroundingDINO did not return any boxes for the provided text prompt")
        return guidance_box_xyxy.astype(np.float32), "proxy_box", ""

    def _refine_box_to_mask(self, frame_chw_01: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray | None:
        if not _box_valid(box_xyxy):
            return None
        image_predictor = self._build_image_predictor()
        image_predictor.set_image(_frame_to_rgb_uint8(frame_chw_01))
        masks, scores, _ = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_xyxy.astype(np.float32),
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        if masks.ndim == 3:
            best_idx = int(np.argmax(scores.reshape(-1)))
            mask = masks[best_idx]
        elif masks.ndim == 2:
            mask = masks
        else:
            return None
        mask = (mask > 0).astype(np.uint8)
        if int(mask.sum()) <= 0:
            return None
        return mask

    def _sample_points_from_mask(self, mask_hw: np.ndarray, num_points: int) -> tuple[np.ndarray, np.ndarray]:
        points = sample_points_from_mask(mask_hw, num_points, avoid_edges=True)
        if points.shape[0] <= 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        labels = np.ones((points.shape[0],), dtype=np.int32)
        return points.astype(np.float32), labels

    def _propagate_segment(
        self,
        *,
        predictor,
        frame_dir: Path,
        frames_tchw_01: np.ndarray,
        anchor_idx: int,
        current_box_xyxy: np.ndarray,
        reverse: bool,
        target_masks: np.ndarray,
    ) -> tuple[int, np.ndarray] | None:
        anchor_mask = self._refine_box_to_mask(frames_tchw_01[int(anchor_idx)], current_box_xyxy)
        prompt_variants: list[tuple[str, dict[str, np.ndarray]]] = []
        if anchor_mask is not None:
            prompt_variants.append(("mask", {"mask": anchor_mask.astype(np.uint8)}))
            num_points = max(4, min(8, int(self.segment_len)))
            points, labels = self._sample_points_from_mask(anchor_mask, num_points=num_points)
            if points.shape[0] > 0:
                prompt_variants.append(("points", {"points": points.astype(np.float32), "labels": labels.astype(np.int32)}))
        prompt_variants.append(("box", {"box": current_box_xyxy.astype(np.float32)}))
        for prompt_mode, prompt_kwargs in prompt_variants:
            state = predictor.init_state(
                video_path=str(frame_dir),
                offload_video_to_cpu=True,
                async_loading_frames=False,
            )
            if prompt_mode == "mask":
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=int(anchor_idx),
                    obj_id=1,
                    mask=prompt_kwargs["mask"],
                )
            elif prompt_mode in {"points", "center_point"}:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(anchor_idx),
                    obj_id=1,
                    points=prompt_kwargs["points"],
                    labels=prompt_kwargs["labels"],
                )
            else:
                predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(anchor_idx),
                    obj_id=1,
                    box=prompt_kwargs["box"],
                )

            seen_frame_indices: list[int] = []
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                state,
                start_frame_idx=int(anchor_idx),
                max_frame_num_to_track=int(self.segment_len),
                reverse=reverse,
            ):
                if len(out_obj_ids) == 0:
                    continue
                mask = (out_mask_logits[0] > 0.0).detach().cpu().numpy().squeeze(0).astype(np.uint8)
                target_masks[int(out_frame_idx)] = mask
                seen_frame_indices.append(int(out_frame_idx))

            if seen_frame_indices:
                next_anchor_idx = min(seen_frame_indices) if reverse else max(seen_frame_indices)
                next_box_xyxy = _mask_to_box_xyxy(target_masks[next_anchor_idx])
                return next_anchor_idx, next_box_xyxy

        raise RuntimeError("SAM2 tracking failed to propagate any masks")

    def track(
        self,
        frames_tchw_01: np.ndarray,
        prompt_frame_idx: int,
        prompt_box_xyxy: np.ndarray,
        *,
        caption: str = "",
    ) -> SAM2TrackOutput:
        predictor = self._build()
        num_frames, _, height, width = frames_tchw_01.shape
        forward_masks = np.zeros((num_frames, height, width), dtype=np.uint8)
        reverse_masks = np.zeros_like(forward_masks)
        prompt_box_xyxy, prompt_mode, prompt_text = self._select_prompt(
            frames_tchw_01,
            prompt_frame_idx=int(prompt_frame_idx),
            caption=caption,
            guidance_box_xyxy=prompt_box_xyxy.astype(np.float32),
        )

        with tempfile.TemporaryDirectory(prefix='sam2_frames_') as tmp_dir:
            frame_dir = Path(tmp_dir)
            _save_frames_to_dir(frames_tchw_01, frame_dir)
            with torch.inference_mode():
                for direction in ('forward', 'reverse'):
                    reverse = direction == 'reverse'
                    target = reverse_masks if reverse else forward_masks
                    current_anchor_idx = int(prompt_frame_idx)
                    current_box_xyxy = prompt_box_xyxy.astype(np.float32).copy()
                    while _box_valid(current_box_xyxy):
                        segment_out = self._propagate_segment(
                            predictor=predictor,
                            frame_dir=frame_dir,
                            frames_tchw_01=frames_tchw_01,
                            anchor_idx=current_anchor_idx,
                            current_box_xyxy=current_box_xyxy,
                            reverse=reverse,
                            target_masks=target,
                        )
                        if segment_out is None:
                            break
                        next_anchor_idx, next_box_xyxy = segment_out
                        if next_anchor_idx == current_anchor_idx:
                            break
                        current_anchor_idx = int(next_anchor_idx)
                        current_box_xyxy = next_box_xyxy.astype(np.float32)
                        if reverse and current_anchor_idx <= 0:
                            break
                        if (not reverse) and current_anchor_idx >= num_frames - 1:
                            break

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
            prompt_mode=prompt_mode,
            prompt_text=prompt_text,
        )
