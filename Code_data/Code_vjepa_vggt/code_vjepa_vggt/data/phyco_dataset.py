"""PhyCo raw-tar dataset adapter for the VJEPA phys-state training path.

Detailed Chinese documentation lives in `phyco_dataset.md` next to this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tarfile
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_SPLIT_NAMES = {"train", "val", "test", "all"}
_SUPPORT_TOKENS = ("platform", "wall", "table", "floor", "ground", "dome", "base")
_OCCLUDER_TOKENS = ("pillar", "post", "pole", "occluder")
_SPHERE_TOKENS = ("ball", "sphere")
_BOX_TOKENS = ("cube", "box", "brick", "block", "jenga", "platform", "wall", "table", "dome")
_CYLINDER_TOKENS = ("cylinder",)
_CAPSULE_TOKENS = ("capsule",)
_PUCK_TOKENS = ("puck", "disc", "disk")


@dataclass(slots=True)
class PhyCoSampleRecord:
    scenario: str
    tar_path: str
    date: str
    sample_id: str
    prompt: str

    @property
    def key(self) -> str:
        return f"{self.scenario}/{self.date}/{self.sample_id}"

    @property
    def member_prefix(self) -> str:
        return f"{self.date}/{self.sample_id}"


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _member_path(prefix: str, name: str) -> str:
    return f"{prefix}/{name}"


def _safe_list_value(values: Any, index: int, default: Any) -> Any:
    if not isinstance(values, list) or index < 0 or index >= len(values):
        return default
    value = values[index]
    return default if value is None else value


def _type_name_to_shape(type_name: str) -> str:
    name = str(type_name).lower()
    if any(token in name for token in _SPHERE_TOKENS):
        return "sphere"
    if any(token in name for token in _CAPSULE_TOKENS):
        return "capsule"
    if any(token in name for token in _CYLINDER_TOKENS):
        return "cylinder"
    if any(token in name for token in _PUCK_TOKENS):
        return "puck"
    if any(token in name for token in _BOX_TOKENS):
        return "box"
    return "box"


def _role_name(type_name: str, motion_score: float) -> str:
    name = str(type_name).lower()
    if any(token in name for token in _OCCLUDER_TOKENS):
        return "occluder"
    if any(token in name for token in _SUPPORT_TOKENS):
        return "support"
    if motion_score > 1.0e-3:
        return "dynamic"
    return "support"


def _shape_one_hot_index(shape_name: str) -> int:
    return {
        "sphere": 0,
        "box": 1,
        "cylinder": 2,
        "capsule": 3,
        "puck": 4,
    }.get(shape_name, 1)


def _role_one_hot_index(role_name: str) -> int:
    return {
        "dynamic": 0,
        "support": 1,
        "occluder": 2,
    }.get(role_name, 1)


def _build_camera_vector(metadata: dict[str, Any], *, width: int, height: int) -> np.ndarray:
    camera_diversity = metadata.get("camera_diversity", {})
    eye = np.asarray(camera_diversity.get("camera_position", [0.0, -8.0, 4.5]), dtype=np.float32)
    forward = -eye.astype(np.float64)
    norm = float(np.linalg.norm(forward))
    if norm < 1.0e-8:
        forward = np.asarray([0.0, 1.0, -0.5], dtype=np.float64)
        norm = float(np.linalg.norm(forward))
    forward = (forward / norm).astype(np.float32)
    yfov_deg = float(metadata.get("camera_yfov_deg", 50.0))
    yfov = math.radians(yfov_deg)
    aspect = float(width) / float(height)
    fx = 0.5 * float(width) / (math.tan(yfov * 0.5) * aspect)
    fy = 0.5 * float(height) / math.tan(yfov * 0.5)
    return np.asarray(
        [
            fx / float(width),
            fy / float(height),
            0.5,
            0.5,
            float(eye[0]),
            float(eye[1]),
            float(eye[2]),
            float(-forward[0]),
            float(-forward[1]),
            float(-forward[2]),
        ],
        dtype=np.float32,
    )


def _read_video_rgb(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return np.stack(frames, axis=0)


def _resize_video_tchw(frames_tchw: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    video = torch.from_numpy(frames_tchw).float()
    video = F.interpolate(video, size=resolution, mode="bilinear", align_corners=False)
    return video.numpy()


def _sample_frame_indices(total_frames: int, target_frames: int) -> np.ndarray:
    if total_frames < target_frames:
        raise RuntimeError(f"raw video has only {total_frames} frames but target_frames={target_frames}")
    return np.linspace(0, total_frames - 1, num=target_frames).round().astype(np.int64)


def _extract_member_to_path(tar_path: Path, member_name: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        return output_path
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tarfile.open(tar_path, "r:gz") as tf:
        handle = tf.extractfile(member_name)
        if handle is None:
            raise FileNotFoundError(f"missing member {member_name} in {tar_path}")
        with open(tmp_path, "wb") as f:
            f.write(handle.read())
    tmp_path.replace(output_path)
    return output_path


def _load_tar_json(tar_path: Path, member_name: str) -> dict[str, Any]:
    with tarfile.open(tar_path, "r:gz") as tf:
        handle = tf.extractfile(member_name)
        if handle is None:
            raise FileNotFoundError(f"missing member {member_name} in {tar_path}")
        return json.load(handle)


def _compute_boxes_and_areas(
    segmentation_rgb_thwc: np.ndarray,
    segmentation_colors: np.ndarray,
    *,
    color_tolerance: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_frames, height, width, _ = segmentation_rgb_thwc.shape
    num_objects = int(segmentation_colors.shape[0])
    boxes = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    areas = np.zeros((num_frames, num_objects), dtype=np.float32)
    if num_objects == 0:
        return boxes, areas

    colors_i16 = segmentation_colors.astype(np.int16)
    for frame_idx in range(num_frames):
        frame = segmentation_rgb_thwc[frame_idx].astype(np.int16)
        best_dist = np.full((height, width), 32767, dtype=np.int16)
        best_idx = np.full((height, width), -1, dtype=np.int16)
        for start in range(0, num_objects, 16):
            end = min(start + 16, num_objects)
            chunk = colors_i16[start:end]
            diff = np.abs(frame[:, :, None, :] - chunk[None, None, :, :])
            dist = diff.max(axis=-1)
            local_best = dist.argmin(axis=-1).astype(np.int16)
            local_dist = dist.min(axis=-1).astype(np.int16)
            update = local_dist < best_dist
            best_dist[update] = local_dist[update]
            best_idx[update] = (local_best + start)[update]
        valid = (best_idx >= 0) & (best_dist <= int(color_tolerance))
        for obj_idx in range(num_objects):
            mask = valid & (best_idx == obj_idx)
            rows, cols = np.where(mask)
            if rows.size == 0 or cols.size == 0:
                continue
            x0 = float(cols.min())
            y0 = float(rows.min())
            x1 = float(cols.max() + 1)
            y1 = float(rows.max() + 1)
            boxes[frame_idx, obj_idx] = np.asarray([x0, y0, x1, y1], dtype=np.float32)
            areas[frame_idx, obj_idx] = float(rows.size)
    return boxes, areas


def _motion_scores(boxes_xyxy: np.ndarray, areas: np.ndarray) -> np.ndarray:
    num_frames, num_objects, _ = boxes_xyxy.shape
    scores = np.zeros((num_objects,), dtype=np.float32)
    for obj_idx in range(num_objects):
        visible = areas[:, obj_idx] > 0.5
        if not np.any(visible):
            continue
        centers = 0.5 * (boxes_xyxy[:, obj_idx, 0:2] + boxes_xyxy[:, obj_idx, 2:4])
        visible_ids = np.flatnonzero(visible)
        path = 0.0
        for left, right in zip(visible_ids[:-1], visible_ids[1:]):
            path += float(np.linalg.norm(centers[right] - centers[left]))
        area_mean = float(areas[visible, obj_idx].mean())
        scores[obj_idx] = path + 0.05 * float(visible.sum()) + 1.0e-4 * area_mean
    return scores


def _build_object_state_arrays(
    boxes_xyxy: np.ndarray,
    areas: np.ndarray,
    *,
    image_hw: tuple[int, int],
    selected_indices: list[int],
    max_objects: int,
) -> tuple[np.ndarray, np.ndarray]:
    total_frames = int(boxes_xyxy.shape[0])
    height, width = int(image_hw[0]), int(image_hw[1])
    states = np.zeros((total_frames, max_objects, 10), dtype=np.float32)
    boxes_norm = np.zeros((total_frames, max_objects, 4), dtype=np.float32)
    scale = np.asarray([width, height, width, height], dtype=np.float32)

    for slot_idx, object_idx in enumerate(selected_indices[:max_objects]):
        visible = areas[:, object_idx] > 0.5
        if not np.any(visible):
            continue
        visible_areas = areas[visible, object_idx]
        base_area = float(np.median(visible_areas)) if visible_areas.size > 0 else 1.0
        prev_center = np.zeros((2,), dtype=np.float32)
        prev_depth = 1.0
        prev_log_scale = float(np.log(max(base_area / float(height * width), 1.0e-6)))
        has_prev = False
        for frame_idx in range(total_frames):
            states[frame_idx, slot_idx, 8] = 1.0
            states[frame_idx, slot_idx, 9] = 1.0
            area = float(areas[frame_idx, object_idx])
            if area <= 0.5:
                if has_prev:
                    states[frame_idx, slot_idx, 0:2] = prev_center
                    states[frame_idx, slot_idx, 2] = prev_depth
                    states[frame_idx, slot_idx, 3] = prev_log_scale
                continue
            box_px = boxes_xyxy[frame_idx, object_idx]
            box_norm = box_px / scale
            boxes_norm[frame_idx, slot_idx] = box_norm.astype(np.float32)
            center = np.asarray(
                [
                    0.5 * float(box_norm[0] + box_norm[2]),
                    0.5 * float(box_norm[1] + box_norm[3]),
                ],
                dtype=np.float32,
            )
            log_scale = float(np.log(max(area / float(height * width), 1.0e-6)))
            depth = float(np.sqrt(max(base_area, 1.0e-6) / max(area, 1.0e-6)))
            velocity = np.zeros((2,), dtype=np.float32)
            depth_velocity = 0.0
            if has_prev:
                velocity = center - prev_center
                depth_velocity = depth - prev_depth
            states[frame_idx, slot_idx, 0:2] = center
            states[frame_idx, slot_idx, 2] = depth
            states[frame_idx, slot_idx, 3] = log_scale
            states[frame_idx, slot_idx, 4:6] = velocity
            states[frame_idx, slot_idx, 6] = depth_velocity
            states[frame_idx, slot_idx, 7] = 1.0
            prev_center = center
            prev_depth = depth
            prev_log_scale = log_scale
            has_prev = True
    return states, boxes_norm


def _build_appearance(
    metadata: dict[str, Any],
    *,
    selected_indices: list[int],
    motion_scores: np.ndarray,
    max_objects: int,
    appearance_dim: int,
) -> np.ndarray:
    object_data = metadata.get("object_data", {})
    types = list(object_data.get("type", []))
    colors = list(object_data.get("color", []))
    masses = list(object_data.get("mass", []))
    frictions = list(object_data.get("friction", []))
    scales = list(object_data.get("scale", []))

    appearance = np.zeros((max_objects, appearance_dim), dtype=np.float32)
    for slot_idx, object_idx in enumerate(selected_indices[:max_objects]):
        type_name = str(_safe_list_value(types, object_idx, "box"))
        motion = float(motion_scores[object_idx]) if object_idx < len(motion_scores) else 0.0
        role_name = _role_name(type_name, motion)
        shape_name = _type_name_to_shape(type_name)
        color = np.asarray(_safe_list_value(colors, object_idx, [0.0, 0.0, 0.0]), dtype=np.float32)
        scale = np.asarray(_safe_list_value(scales, object_idx, [1.0, 1.0, 1.0]), dtype=np.float32)
        appearance[slot_idx, _shape_one_hot_index(shape_name)] = 1.0
        appearance[slot_idx, 5 + _role_one_hot_index(role_name)] = 1.0
        appearance[slot_idx, 8:11] = color[:3]
        appearance[slot_idx, 11] = float(np.max(scale))
        appearance[slot_idx, 12] = float(np.min(scale))
        appearance[slot_idx, 13] = float(np.prod(scale))
        appearance[slot_idx, 14] = float(_safe_list_value(masses, object_idx, 1.0))
        appearance[slot_idx, 15] = float(_safe_list_value(frictions, object_idx, 1.0))
    return appearance


class PhyCoEpisodeDataset(Dataset):
    """Read raw PhyCo tarballs and expose phys_state-style episodes.

    This class is intentionally aligned to the output contract of
    PhysStateEpisodeDataset. The current implementation derives boxes from
    segmentation video and uses box-area depth as a proxy, so it is suitable for
    `run_train_v_newtrain_gpu67.sh`-style training where `lambda_depth_aux=0.0`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: tuple[int, int],
        num_context_frames: int = 8,
        num_future_frames: int = 16,
        max_objects: int = 6,
        context_fraction: float = 0.5,
        random_context_frames: bool = True,
        seed: int = 42,
        scenarios: list[str] | None = None,
        init_scan_limit: int | None = None,
        cache_root: str | Path = "/data/gaoya/agent-data/cache/phyco_vjepa_dataset",
        split_train_ratio: float = 0.9,
        split_val_ratio: float = 0.05,
        color_tolerance: int = 18,
        appearance_dim: int = 16,
    ) -> None:
        self.root = Path(root)
        self.split = str(split).strip().lower()
        if self.split not in _SPLIT_NAMES:
            raise ValueError(f"unsupported split={split!r}, expected one of {_SPLIT_NAMES}")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.num_context_frames = int(num_context_frames)
        self.num_future_frames = int(num_future_frames)
        self.total_frames = self.num_context_frames + self.num_future_frames
        self.max_objects = int(max_objects)
        self.context_fraction = float(context_fraction)
        self.random_context_frames = bool(random_context_frames)
        self.seed = int(seed)
        self.init_scan_limit = None if init_scan_limit is None else max(int(init_scan_limit), 1)
        self.cache_root = Path(cache_root)
        self.raw_cache_root = self.cache_root / "raw"
        self.episode_cache_root = self.cache_root / "episodes"
        self.index_cache_root = self.cache_root / "indices"
        self.scenarios = sorted({item.strip() for item in scenarios or [] if item.strip()})
        self.split_train_ratio = float(split_train_ratio)
        self.split_val_ratio = float(split_val_ratio)
        self.color_tolerance = int(color_tolerance)
        self.appearance_dim = int(appearance_dim)
        if not 0.0 < self.split_train_ratio < 1.0:
            raise ValueError(f"split_train_ratio must be in (0,1), got {self.split_train_ratio}")
        if not 0.0 <= self.split_val_ratio < 1.0:
            raise ValueError(f"split_val_ratio must be in [0,1), got {self.split_val_ratio}")
        if self.split_train_ratio + self.split_val_ratio >= 1.0:
            raise ValueError("split_train_ratio + split_val_ratio must be < 1.0")

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.samples = self._build_index()
        if self.init_scan_limit is not None:
            self.samples = self.samples[: self.init_scan_limit]
        if not self.samples:
            raise RuntimeError(f"no PhyCo samples found for split={self.split} under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _max_context_len(self, total_frames: int) -> int:
        return max(1, min(total_frames, int(total_frames * self.context_fraction)))

    def _select_context_indices(self, total_frames: int, idx: int) -> torch.Tensor:
        max_context_len = self._max_context_len(total_frames)
        if max_context_len < self.num_context_frames:
            raise RuntimeError(
                f"sample idx={idx} only has max_context_len={max_context_len}, "
                f"smaller than required num_context_frames={self.num_context_frames}"
            )
        if not self.random_context_frames:
            return torch.arange(self.num_context_frames, dtype=torch.long)
        max_start = max_context_len - self.num_context_frames
        generator = torch.Generator()
        generator.manual_seed(self.seed + idx)
        start = int(torch.randint(0, max_start + 1, (1,), generator=generator).item()) if max_start > 0 else 0
        return torch.arange(start, start + self.num_context_frames, dtype=torch.long)

    def _index_cache_path(self) -> Path:
        scenario_key = ",".join(self.scenarios) if self.scenarios else "__all__"
        config_key = (
            f"root={self.root.resolve()}|split={self.split}|scenario={scenario_key}|"
            f"train={self.split_train_ratio:.6f}|val={self.split_val_ratio:.6f}"
        )
        digest = hashlib.sha1(config_key.encode("utf-8")).hexdigest()[:16]
        return self.index_cache_root / f"phyco_index_{digest}.json"

    def _sample_split_name(self, key: str) -> str:
        u = _stable_unit_interval(key)
        if u < self.split_train_ratio:
            return "train"
        if u < self.split_train_ratio + self.split_val_ratio:
            return "val"
        return "test"

    def _build_index(self) -> list[PhyCoSampleRecord]:
        index_path = self._index_cache_path()
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            return [PhyCoSampleRecord(**item) for item in payload["samples"]]

        scenario_dirs = sorted(path for path in self.root.iterdir() if path.is_dir())
        if self.scenarios:
            scenario_set = set(self.scenarios)
            scenario_dirs = [path for path in scenario_dirs if path.name in scenario_set]

        samples: list[PhyCoSampleRecord] = []
        for scenario_dir in scenario_dirs:
            scenario = scenario_dir.name
            caption_path = scenario_dir / "common_caption_cosmos.txt"
            prompt = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else scenario.replace("_", " ")
            for tar_path in sorted(scenario_dir.glob("*.tar.gz")):
                date = tar_path.name.replace(".tar.gz", "")
                per_sample_files: dict[str, set[str]] = {}
                with tarfile.open(tar_path, "r:gz") as tf:
                    for name in tf.getnames():
                        parts = name.split("/")
                        if len(parts) != 3:
                            continue
                        sample_id = parts[1]
                        file_name = parts[2]
                        if not sample_id or not file_name:
                            continue
                        bucket = per_sample_files.setdefault(sample_id, set())
                        bucket.add(file_name)
                for sample_id, files in sorted(per_sample_files.items()):
                    if not {"metadata.json", "rgba.mp4", "segmentation.mp4"}.issubset(files):
                        continue
                    record = PhyCoSampleRecord(
                        scenario=scenario,
                        tar_path=str(tar_path),
                        date=date,
                        sample_id=sample_id,
                        prompt=prompt,
                    )
                    if self.split != "all" and self._sample_split_name(record.key) != self.split:
                        continue
                    samples.append(record)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "split": self.split,
            "scenarios": self.scenarios,
            "samples": [asdict(record) for record in samples],
        }
        tmp_path = index_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(index_path)
        return samples

    def _cache_key(self, record: PhyCoSampleRecord) -> str:
        cfg = (
            f"{record.key}|ctx={self.num_context_frames}|future={self.num_future_frames}|"
            f"obj={self.max_objects}|res={self.resolution[0]}x{self.resolution[1]}|tol={self.color_tolerance}"
        )
        digest = hashlib.sha1(cfg.encode("utf-8")).hexdigest()[:16]
        return f"{record.scenario}__{record.date}__{record.sample_id}__{digest}"

    def _episode_cache_paths(self, record: PhyCoSampleRecord) -> tuple[Path, Path]:
        stem = self._cache_key(record)
        return self.episode_cache_root / f"{stem}.npz", self.episode_cache_root / f"{stem}.json"

    def _ensure_raw_member(self, record: PhyCoSampleRecord, name: str) -> Path:
        tar_path = Path(record.tar_path)
        member_name = _member_path(record.member_prefix, name)
        output_path = self.raw_cache_root / record.scenario / record.date / record.sample_id / name
        return _extract_member_to_path(tar_path, member_name, output_path)

    def _process_record(self, record: PhyCoSampleRecord) -> tuple[Path, Path]:
        cache_npz_path, cache_json_path = self._episode_cache_paths(record)
        if cache_npz_path.is_file() and cache_json_path.is_file():
            return cache_npz_path, cache_json_path

        tar_path = Path(record.tar_path)
        metadata = _load_tar_json(tar_path, _member_path(record.member_prefix, "metadata.json"))
        rgba_path = self._ensure_raw_member(record, "rgba.mp4")
        segmentation_path = self._ensure_raw_member(record, "segmentation.mp4")

        rgb_full = _read_video_rgb(rgba_path)
        segmentation_full = _read_video_rgb(segmentation_path)
        if rgb_full.shape[0] != segmentation_full.shape[0]:
            total = min(int(rgb_full.shape[0]), int(segmentation_full.shape[0]))
            rgb_full = rgb_full[:total]
            segmentation_full = segmentation_full[:total]

        sampled_frame_indices = _sample_frame_indices(int(rgb_full.shape[0]), self.total_frames)
        rgb = rgb_full[sampled_frame_indices]
        segmentation = segmentation_full[sampled_frame_indices]
        source_height = int(rgb.shape[1])
        source_width = int(rgb.shape[2])

        object_data = metadata.get("object_data", {})
        segmentation_ids = list(object_data.get("segmentation_id", []))
        segmentation_colors = np.asarray(object_data.get("segmentation_color", []), dtype=np.uint8)
        object_types = list(object_data.get("type", []))
        if segmentation_colors.ndim != 2 or segmentation_colors.shape[-1] != 3:
            raise RuntimeError(f"invalid segmentation colors for sample {record.key}")

        boxes_xyxy, areas = _compute_boxes_and_areas(
            segmentation,
            segmentation_colors,
            color_tolerance=self.color_tolerance,
        )
        motion_scores = _motion_scores(boxes_xyxy, areas)
        ranking = list(range(len(segmentation_ids)))
        ranking.sort(
            key=lambda obj_idx: (
                float(motion_scores[obj_idx]),
                0.0 if _role_name(_safe_list_value(object_types, obj_idx, "box"), float(motion_scores[obj_idx])) == "dynamic" else -1.0,
                float(areas[:, obj_idx].mean()) if obj_idx < areas.shape[1] else 0.0,
            ),
            reverse=True,
        )
        selected_indices = [obj_idx for obj_idx in ranking if float(areas[:, obj_idx].sum()) > 0.5][: self.max_objects]
        if not selected_indices:
            raise RuntimeError(f"no visible objects found in sample {record.key}")

        states, boxes_norm = _build_object_state_arrays(
            boxes_xyxy,
            areas,
            image_hw=(source_height, source_width),
            selected_indices=selected_indices,
            max_objects=self.max_objects,
        )
        appearance = _build_appearance(
            metadata,
            selected_indices=selected_indices,
            motion_scores=motion_scores,
            max_objects=self.max_objects,
            appearance_dim=self.appearance_dim,
        )

        frames_tchw = np.transpose(rgb.astype(np.float32) / 255.0, (0, 3, 1, 2))
        frames_tchw = _resize_video_tchw(frames_tchw, self.resolution)
        camera_vec = _build_camera_vector(metadata, width=source_width, height=source_height)
        camera_full = np.repeat(camera_vec[None, :], self.total_frames, axis=0)

        context_slice = slice(0, self.num_context_frames)
        future_slice = slice(self.num_context_frames, self.total_frames)
        cache_npz_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = cache_npz_path.with_suffix(".npz.tmp")
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(
                f,
                context_frames=frames_tchw[context_slice].astype(np.float32),
                future_frames=frames_tchw[future_slice].astype(np.float32),
                context_states=states[context_slice].astype(np.float32),
                future_states=states[future_slice].astype(np.float32),
                context_boxes=boxes_norm[context_slice].astype(np.float32),
                future_boxes=boxes_norm[future_slice].astype(np.float32),
                full_frames=frames_tchw.astype(np.float32),
                full_states=states.astype(np.float32),
                full_boxes=boxes_norm.astype(np.float32),
                appearance=appearance.astype(np.float32),
                camera=camera_full[context_slice].astype(np.float32),
                camera_full=camera_full.astype(np.float32),
            )
        Path(str(tmp_npz)).replace(cache_npz_path)

        selected_types = [_safe_list_value(object_types, obj_idx, "box") for obj_idx in selected_indices]
        selected_seg_ids = [_safe_list_value(segmentation_ids, obj_idx, -1) for obj_idx in selected_indices]
        cache_meta = {
            "prompt": record.prompt,
            "scenario": record.scenario,
            "tar_path": record.tar_path,
            "date": record.date,
            "sample_id": record.sample_id,
            "sample_key": record.key,
            "split": self.split,
            "raw_frame_count": int(rgb_full.shape[0]),
            "sampled_frame_indices": sampled_frame_indices.tolist(),
            "selected_object_indices": selected_indices,
            "selected_object_types": selected_types,
            "selected_segmentation_ids": selected_seg_ids,
            "depth_mode": "box_area_proxy",
            "camera_mode": "origin_lookat_proxy",
        }
        tmp_json = cache_json_path.with_suffix(".json.tmp")
        tmp_json.write_text(json.dumps(cache_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_json.replace(cache_json_path)
        return cache_npz_path, cache_json_path

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.samples[idx]
        cache_npz_path, cache_json_path = self._process_record(record)
        payload = np.load(cache_npz_path)
        meta = json.loads(cache_json_path.read_text(encoding="utf-8"))

        context_frames = torch.from_numpy(payload["context_frames"]).float()
        future_frames = torch.from_numpy(payload["future_frames"]).float()
        all_frames = torch.cat([context_frames, future_frames], dim=0)
        total_frames = int(all_frames.shape[0])
        context_indices = self._select_context_indices(total_frames, idx)

        video = all_frames.permute(1, 0, 2, 3).contiguous()
        video = video * 2.0 - 1.0
        context_video = video[:, context_indices].contiguous()

        all_boxes = torch.cat(
            [
                torch.from_numpy(payload["context_boxes"]).float(),
                torch.from_numpy(payload["future_boxes"]).float(),
            ],
            dim=0,
        )
        all_states = torch.cat(
            [
                torch.from_numpy(payload["context_states"]).float(),
                torch.from_numpy(payload["future_states"]).float(),
            ],
            dim=0,
        )
        context_boxes = all_boxes[context_indices].contiguous()
        context_states = all_states[context_indices].contiguous()

        return {
            "video": video,
            "context_video": context_video,
            "caption": meta["prompt"],
            "video_path": f"{record.tar_path}:{record.member_prefix}/rgba.mp4",
            "frame_indices": torch.arange(total_frames, dtype=torch.long),
            "context_frame_indices": context_indices,
            "num_context_frames": int(context_indices.numel()),
            "metadata": meta,
            "context_boxes": context_boxes,
            "future_boxes": torch.from_numpy(payload["future_boxes"]).float(),
            "context_states": context_states,
            "future_states": torch.from_numpy(payload["future_states"]).float(),
            "appearance": torch.from_numpy(payload["appearance"]).float(),
            "camera": torch.from_numpy(payload["camera"]).float(),
        }

    def materialize_cache(self, limit: int | None = None) -> dict[str, Any]:
        processed = 0
        errors: list[dict[str, str]] = []
        target = self.samples if limit is None else self.samples[: max(int(limit), 0)]
        for record in target:
            try:
                self._process_record(record)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                errors.append({"sample_key": record.key, "error": str(exc)})
        return {
            "processed": processed,
            "requested": len(target),
            "errors": errors,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test or pre-materialize PhyCo episodes.")
    parser.add_argument("--root", default="/data/gaoya/dataset/nnsriram97-phyco_kubric")
    parser.add_argument("--split", default="train", choices=sorted(_SPLIT_NAMES))
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--num-future-frames", type=int, default=16)
    parser.add_argument("--max-objects", type=int, default=6)
    parser.add_argument("--cache-root", default="/data/gaoya/agent-data/cache/phyco_vjepa_dataset")
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--init-scan-limit", type=int, default=4)
    parser.add_argument("--materialize-limit", type=int, default=0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    dataset = PhyCoEpisodeDataset(
        root=args.root,
        split=args.split,
        resolution=(args.height, args.width),
        num_context_frames=args.num_context_frames,
        num_future_frames=args.num_future_frames,
        max_objects=args.max_objects,
        cache_root=args.cache_root,
        scenarios=args.scenario,
        init_scan_limit=args.init_scan_limit,
        random_context_frames=False,
    )
    summary: dict[str, Any] = {
        "split": args.split,
        "num_samples": len(dataset),
        "cache_root": str(Path(args.cache_root)),
    }
    if args.materialize_limit > 0:
        summary["materialize"] = dataset.materialize_cache(limit=args.materialize_limit)
    sample = dataset[0]
    summary["sample0"] = {
        "caption": sample["caption"],
        "video_shape": list(sample["video"].shape),
        "context_video_shape": list(sample["context_video"].shape),
        "context_boxes_shape": list(sample["context_boxes"].shape),
        "future_boxes_shape": list(sample["future_boxes"].shape),
        "context_states_shape": list(sample["context_states"].shape),
        "future_states_shape": list(sample["future_states"].shape),
        "appearance_shape": list(sample["appearance"].shape),
        "camera_shape": list(sample["camera"].shape),
        "video_path": sample["video_path"],
        "sample_key": sample["metadata"].get("sample_key"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
