from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schemas import STATE_DIM, StateIndex, box_center


@dataclass(slots=True)
class ProxyTrack:
    frames: np.ndarray
    boxes: np.ndarray
    states: np.ndarray
    appearance: np.ndarray
    visible_fraction: float


def read_video_frames(
    video_path: str | Path,
    resize_height: int | None = None,
    resize_width: int | None = None,
    max_frames: int | None = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize_height is not None and resize_width is not None:
            frame = cv2.resize(frame, (resize_width, resize_height),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1)))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {video_path}")
    return np.stack(frames, axis=0)


def _extract_primary_boxes(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray]:
    num_frames, _, height, width = frames.shape
    grays = []
    for frame in frames:
        image = np.transpose((frame * 255.0).clip(0, 255).astype(np.uint8),
                             (1, 2, 0))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grays.append(cv2.GaussianBlur(gray, (5, 5), 0))

    ref = grays[0]
    prev = ref
    prev_center: tuple[float, float] | None = None
    prev_box: np.ndarray | None = None
    prev_area = 0.0

    boxes = np.zeros((num_frames, 4), dtype=np.float32)
    areas = np.zeros((num_frames,), dtype=np.float32)
    visibility = np.zeros((num_frames,), dtype=np.float32)
    confidence = np.zeros((num_frames,), dtype=np.float32)

    frame_area = float(height * width)
    min_area = max(24.0, frame_area * 0.00018)
    kernel_small = np.ones((3, 3), dtype=np.uint8)
    kernel_big = np.ones((7, 7), dtype=np.uint8)

    for idx, gray in enumerate(grays):
        diff_ref = cv2.absdiff(gray, ref)
        diff_prev = cv2.absdiff(gray, prev)
        motion = cv2.max(diff_ref, diff_prev)
        _, mask = cv2.threshold(motion, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_box = None
        best_area = 0.0
        best_score = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
            center = box_center(box)
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
                best_box = box
                best_area = area
                best_score = score

        if best_box is None:
            if prev_box is not None:
                boxes[idx] = prev_box
                areas[idx] = prev_area
                confidence[idx] = 0.25
            visibility[idx] = 0.0
        else:
            boxes[idx] = best_box
            areas[idx] = best_area
            visibility[idx] = 1.0
            confidence[idx] = 1.0
            prev_box = best_box
            prev_area = best_area
            center = box_center(best_box)
            prev_center = (float(center[0]), float(center[1]))
        prev = gray

    scale = np.asarray([width, height, width, height], dtype=np.float32)
    boxes_norm = boxes / scale
    return boxes_norm, areas, visibility, confidence


def _encode_appearance(frames: np.ndarray, boxes: np.ndarray,
                       visibility: np.ndarray) -> np.ndarray:
    appearance = np.zeros((1, 64), dtype=np.float32)
    for frame, box, visible in zip(frames, boxes, visibility):
        if visible < 0.5:
            continue
        _, height, width = frame.shape
        x0 = max(int(box[0] * width), 0)
        y0 = max(int(box[1] * height), 0)
        x1 = min(int(box[2] * width), width)
        y1 = min(int(box[3] * height), height)
        crop = frame[:, y0:y1, x0:x1]
        if crop.size == 0:
            continue
        flat = crop.reshape(3, -1)
        appearance[0, 0:3] = flat.mean(axis=1)
        appearance[0, 3:6] = flat.std(axis=1)
        appearance[0, 6] = float((x1 - x0) / max(width, 1))
        appearance[0, 7] = float((y1 - y0) / max(height, 1))
        appearance[0, 8] = 1.0
        return appearance
    return appearance


def extract_primary_track(frames: np.ndarray) -> ProxyTrack:
    boxes, areas, visibility, confidence = _extract_primary_boxes(frames)
    num_frames = frames.shape[0]
    states = np.zeros((num_frames, 1, STATE_DIM), dtype=np.float32)

    visible_areas = areas[visibility > 0.5]
    if visible_areas.size == 0:
        base_area = 1e-4
    else:
        base_area = float(np.median(visible_areas))

    prev_center = None
    prev_depth = 1.0
    for idx in range(num_frames):
        box = boxes[idx]
        center = box_center(box)
        area = max(float((box[2] - box[0]) * (box[3] - box[1])), 1e-6)
        depth = float(np.sqrt(max(base_area, 1e-6) / area))
        velocity = np.zeros((2,), dtype=np.float32)
        depth_velocity = 0.0
        if prev_center is not None:
            velocity = center - prev_center
            depth_velocity = depth - prev_depth
        states[idx, 0, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = center
        states[idx, 0, StateIndex.DEPTH] = depth
        states[idx, 0, StateIndex.LOG_SCALE] = float(np.log(area))
        states[idx, 0, StateIndex.VEL_X:StateIndex.VEL_Y + 1] = velocity
        states[idx, 0, StateIndex.DEPTH_VEL] = depth_velocity
        states[idx, 0, StateIndex.VISIBILITY] = visibility[idx]
        states[idx, 0, StateIndex.EXISTENCE] = 1.0
        states[idx, 0, StateIndex.CONFIDENCE] = confidence[idx]
        prev_center = center
        prev_depth = depth

    appearance = _encode_appearance(frames, boxes, visibility)
    return ProxyTrack(
        frames=frames.astype(np.float32),
        boxes=boxes[:, None, :].astype(np.float32),
        states=states.astype(np.float32),
        appearance=appearance.astype(np.float32),
        visible_fraction=float(np.mean(visibility > 0.5)),
    )
