from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np

from .schemas import FrameObjectState, box_center, box_log_scale, compute_box_from_mask


def compute_scale_depth_consistency(log_scale: float, depth: float, eps: float = 1e-6) -> float:
    log_scale_arr = np.asarray(log_scale, dtype=np.float32)
    depth_arr = np.asarray(depth, dtype=np.float32)
    return log_scale_arr + 2.0 * np.log(np.clip(depth_arr, eps, None))


def _safe_mean_depth(depth_map: Optional[np.ndarray], mask: Optional[np.ndarray], box: np.ndarray) -> float:
    if depth_map is None:
        return 1.0
    if mask is not None and mask.any():
        values = depth_map[mask.astype(bool)]
        if values.size > 0:
            return float(np.median(values))
    x0, y0, x1, y1 = box.astype(int)
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, depth_map.shape[1] - 1)
    y1 = min(y1, depth_map.shape[0] - 1)
    crop = depth_map[y0:y1 + 1, x0:x1 + 1]
    if crop.size == 0:
        return 1.0
    return float(np.median(crop))


@dataclass(slots=True)
class AnnotationPseudoStateExtractor:
    image_height: int
    image_width: int
    normalize: bool = True

    def extract(
        self,
        annotations_per_frame: Iterable[Iterable[Mapping[str, object]]],
        depth_maps: Optional[Iterable[np.ndarray]] = None,
        appearance_by_track: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        frames = list(annotations_per_frame)
        if depth_maps is None:
            depth_list = [None] * len(frames)
        else:
            depth_list = list(depth_maps)
        track_ids = sorted({int(obj["track_id"]) for frame in frames for obj in frame})
        track_to_index = {track_id: idx for idx, track_id in enumerate(track_ids)}
        num_frames = len(frames)
        num_tracks = len(track_ids)

        states = np.zeros((num_frames, num_tracks, 10), dtype=np.float32)
        boxes = np.zeros((num_frames, num_tracks, 4), dtype=np.float32)
        appearances: List[np.ndarray] = []

        for track_id in track_ids:
            if appearance_by_track and track_id in appearance_by_track:
                appearances.append(np.asarray(appearance_by_track[track_id], dtype=np.float32))
            else:
                appearances.append(np.zeros((64,), dtype=np.float32))

        prev_centers = np.zeros((num_tracks, 2), dtype=np.float32)
        prev_depths = np.ones((num_tracks,), dtype=np.float32)
        prev_valid = np.zeros((num_tracks,), dtype=bool)

        for frame_idx, frame_objects in enumerate(frames):
            depth_map = depth_list[frame_idx] if frame_idx < len(depth_list) else None
            for obj in frame_objects:
                track_id = int(obj["track_id"])
                obj_idx = track_to_index[track_id]

                if "bbox" in obj:
                    box = np.asarray(obj["bbox"], dtype=np.float32)
                elif "mask" in obj:
                    box = compute_box_from_mask(np.asarray(obj["mask"], dtype=np.uint8))
                else:
                    raise ValueError("each annotation must provide either bbox or mask")

                mask = np.asarray(obj["mask"], dtype=np.uint8) if "mask" in obj else None
                center = box_center(box)
                depth = float(obj.get("depth", _safe_mean_depth(depth_map, mask, box)))
                visibility = float(obj.get("visibility", 1.0))
                existence = float(obj.get("existence", 1.0))
                confidence = float(obj.get("confidence", 1.0))

                if self.normalize:
                    center = center / np.asarray([self.image_width, self.image_height], dtype=np.float32)
                    box = box / np.asarray([self.image_width, self.image_height, self.image_width, self.image_height], dtype=np.float32)
                log_scale = box_log_scale(box)

                velocity = np.zeros((2,), dtype=np.float32)
                depth_velocity = 0.0
                if prev_valid[obj_idx]:
                    velocity = center - prev_centers[obj_idx]
                    depth_velocity = depth - float(prev_depths[obj_idx])

                states[frame_idx, obj_idx, 0:2] = center
                states[frame_idx, obj_idx, 2] = depth
                states[frame_idx, obj_idx, 3] = log_scale
                states[frame_idx, obj_idx, 4:6] = velocity
                states[frame_idx, obj_idx, 6] = depth_velocity
                states[frame_idx, obj_idx, 7] = visibility
                states[frame_idx, obj_idx, 8] = existence
                states[frame_idx, obj_idx, 9] = confidence
                boxes[frame_idx, obj_idx] = box

                prev_centers[obj_idx] = center
                prev_depths[obj_idx] = depth
                prev_valid[obj_idx] = True

        scale_depth = compute_scale_depth_consistency(states[..., 3], states[..., 2])
        return {
            "track_ids": np.asarray(track_ids, dtype=np.int32),
            "states": states,
            "boxes": boxes,
            "appearance": np.stack(appearances, axis=0),
            "scale_depth_consistency": scale_depth.astype(np.float32),
        }


def build_frame_object_states(
    track_ids: np.ndarray,
    state_vectors: np.ndarray,
    boxes: np.ndarray,
    appearances: Optional[np.ndarray] = None,
) -> List[FrameObjectState]:
    outputs: List[FrameObjectState] = []
    for idx, track_id in enumerate(track_ids.tolist()):
        appearance = None if appearances is None else appearances[idx]
        outputs.append(
            FrameObjectState(
                track_id=int(track_id),
                center=state_vectors[idx, 0:2],
                depth=float(state_vectors[idx, 2]),
                log_scale=float(state_vectors[idx, 3]),
                velocity=state_vectors[idx, 4:6],
                depth_velocity=float(state_vectors[idx, 6]),
                visibility=float(state_vectors[idx, 7]),
                existence=float(state_vectors[idx, 8]),
                confidence=float(state_vectors[idx, 9]),
                bbox=boxes[idx],
                appearance=appearance,
                scale_depth_consistency=compute_scale_depth_consistency(
                    float(state_vectors[idx, 3]), float(state_vectors[idx, 2])
                ),
            )
        )
    return outputs
