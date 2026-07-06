from __future__ import annotations

import hashlib
import json
import re
import zipfile
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, sample_frame_indices


_SPLIT_NAMES = {"train", "val", "test", "all"}
_FRAME_RE = re.compile(r"^(?P<scene>[^/]+)/(?P<camera>CineCamera_[^/]+)/rgb/(?P<index>\d+)\.(jpg|jpeg|png)$")


@dataclass(slots=True)
class PhysInOneNoGTBoxRecord:
    top_split: str
    physics_group: str
    scene_name: str
    camera_name: str
    archive_path: str
    sample_name: str
    caption: str
    frame_names: list[str]
    frame_count_hint: int | None = None

    @property
    def key(self) -> str:
        return f"{self.top_split}/{self.physics_group}/{self.scene_name}/{self.camera_name}"


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _read_json_from_zip(zf: zipfile.ZipFile, member: str) -> dict[str, Any]:
    with zf.open(member) as handle:
        return json.loads(handle.read().decode("utf-8"))


def _read_text_from_zip(zf: zipfile.ZipFile, member: str) -> str:
    with zf.open(member) as handle:
        return handle.read().decode("utf-8").strip()


def _caption_from_metadata(metadata: dict[str, Any], default: str) -> str:
    for key in ("input_caption", "caption", "prompt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    physics_types = metadata.get("physics_types")
    if isinstance(physics_types, list) and physics_types:
        joined = ", ".join(str(item).strip() for item in physics_types if str(item).strip())
        if joined:
            return f"A realistic physics video showing {joined}."
    group_name = metadata.get("group_name")
    if isinstance(group_name, str) and group_name.strip():
        return group_name.strip()
    return default


def _frame_count_hint_from_metadata(metadata: dict[str, Any] | None) -> int | None:
    if metadata is None:
        return None
    sequence_info = metadata.get("sequence_info", {})
    if isinstance(sequence_info, dict):
        for key in ("total_frames", "end_frame"):
            value = sequence_info.get(key)
            if isinstance(value, int) and value > 0:
                return int(value)
            if isinstance(value, float) and value > 0:
                return int(value)
    for key in ("future_frames", "context_frames", "frame_count", "num_frames"):
        value = metadata.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)
    return None


class PhysInOneNoGTBoxDataset(Dataset):
    """PhysInOne adapter for the train0705 no-GT-box branch.

    Each sample corresponds to one camera view inside one scene zip. The loader
    reads the JPEG frames directly from the archive and returns the same fields
    the current train0705 pipeline expects:

    - video
    - context_video
    - caption
    - video_path
    - frame_indices
    - context_frame_indices
    - num_context_frames
    - metadata
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: tuple[int, int],
        *,
        num_frames: int = 24,
        num_context_frames: int = 8,
        sampling_strategy: str = "prefix",
        seed: int = 42,
        camera_names: list[str] | None = None,
        init_scan_limit: int | None = None,
        cache_root: str | Path = "/data/gaoya/agent-data/cache/phisinone_no_gt_box_dataset",
        max_retry_samples: int = 8,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = str(split).strip().lower()
        if self.split not in _SPLIT_NAMES:
            raise ValueError(f"unsupported split={split!r}, expected one of {_SPLIT_NAMES}")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        self.seed = int(seed)
        self.camera_names = sorted({item.strip() for item in camera_names or [] if item and item.strip()})
        self.init_scan_limit = None if init_scan_limit is None else max(int(init_scan_limit), 1)
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.index_cache_root = self.cache_root / "indices"
        self.max_retry_samples = max(1, int(max_retry_samples))

        if not self.root.is_dir():
            raise FileNotFoundError(f"PhysInOne root not found: {self.root}")
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if self.num_context_frames <= 0:
            raise ValueError(f"num_context_frames must be positive, got {self.num_context_frames}")
        if self.num_context_frames > self.num_frames:
            raise ValueError(
                f"num_context_frames={self.num_context_frames} exceeds num_frames={self.num_frames}"
            )
        if self.sampling_strategy not in {"prefix", "uniform"}:
            raise ValueError(
                f"unsupported sampling_strategy={self.sampling_strategy!r}, expected prefix/uniform"
            )

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.samples = self._build_index()
        if self.init_scan_limit is not None:
            self.samples = self.samples[: self.init_scan_limit]
        if not self.samples:
            raise RuntimeError(f"no PhysInOne samples found for split={self.split} under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _index_cache_path(self) -> Path:
        camera_key = ",".join(self.camera_names) if self.camera_names else "__all__"
        config_key = (
            f"root={self.root.resolve()}|split={self.split}|camera={camera_key}|"
            f"num_frames={self.num_frames}|ctx={self.num_context_frames}|sampling={self.sampling_strategy}"
        )
        digest = hashlib.sha1(config_key.encode("utf-8")).hexdigest()[:16]
        return self.index_cache_root / f"phisinone_index_{digest}.json"

    def _sample_frame_indices(self, frame_count: int) -> np.ndarray:
        if frame_count < self.num_frames:
            raise RuntimeError(
                f"camera view has only {frame_count} frames, smaller than requested {self.num_frames}"
            )
        if self.sampling_strategy == "uniform":
            return sample_frame_indices(frame_count, self.num_frames)
        return np.arange(self.num_frames, dtype=np.int64)

    def _build_index(self) -> list[PhysInOneNoGTBoxRecord]:
        index_path = self._index_cache_path()
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            return [PhysInOneNoGTBoxRecord(**item) for item in payload["samples"]]

        samples: list[PhysInOneNoGTBoxRecord] = []
        archive_paths = sorted(self.root.rglob("*.zip"))
        for archive_path in archive_paths:
            try:
                rel_parts = archive_path.relative_to(self.root).parts
            except ValueError:
                rel_parts = archive_path.parts
            if not rel_parts:
                continue
            top_split = rel_parts[0]
            if self.split != "all" and top_split.strip().lower() != self.split:
                continue

            with zipfile.ZipFile(archive_path) as zf:
                names = zf.namelist()
                caption = None
                for candidate in (
                    next((name for name in names if name.endswith("/caption.txt")), None),
                    next((name for name in names if name.endswith("caption.txt")), None),
                ):
                    if candidate is not None:
                        text = _read_text_from_zip(zf, candidate)
                        if text:
                            caption = text
                            break

                metadata = None
                metadata_member = next(
                    (
                        name
                        for name in names
                        if name.endswith(f"{Path(archive_path).stem}.json")
                    ),
                    None,
                )
                if metadata_member is None:
                    metadata_member = next(
                        (
                            name
                            for name in names
                            if name.endswith(".json")
                            and "recorder_stats" not in name
                            and "blender_CineCamera" not in name
                        ),
                        None,
                    )
                if metadata_member is not None:
                    try:
                        metadata = _read_json_from_zip(zf, metadata_member)
                    except Exception:
                        metadata = None

                scenes: dict[tuple[str, str], list[str]] = {}
                for name in names:
                    match = _FRAME_RE.match(name)
                    if match is None:
                        continue
                    camera_name = match.group("camera")
                    if self.camera_names and camera_name not in self.camera_names:
                        continue
                    scene_name = match.group("scene")
                    key = (scene_name, camera_name)
                    scenes.setdefault(key, []).append(name)

                for (scene_name, camera_name), frame_names in sorted(scenes.items()):
                    frame_names = sorted(
                        frame_names,
                        key=lambda item: int(Path(item).stem),
                    )
                    frame_count = len(frame_names)
                    if frame_count < self.num_frames:
                        continue
                    sample_id = f"{scene_name}__{camera_name}"
                    prompt_default = f"PhysInOne scene {scene_name} on {camera_name}"
                    record = PhysInOneNoGTBoxRecord(
                        top_split=top_split,
                        physics_group=rel_parts[1] if len(rel_parts) > 1 else "Unknown",
                        scene_name=scene_name,
                        camera_name=camera_name,
                        archive_path=str(archive_path),
                        sample_name=sample_id,
                        caption=_caption_from_metadata(metadata or {}, caption or prompt_default),
                        frame_names=frame_names,
                        frame_count_hint=_frame_count_hint_from_metadata(metadata),
                    )
                    samples.append(record)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "split": self.split,
            "samples": [asdict(record) for record in samples],
        }
        tmp_path = index_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(index_path)
        return samples

    def _load_frame(self, zf: zipfile.ZipFile, member: str) -> np.ndarray:
        with zf.open(member) as handle:
            with Image.open(handle) as img:
                return np.asarray(img.convert("RGB"))

    def _load_sample(self, record: PhysInOneNoGTBoxRecord) -> dict[str, Any]:
        archive_path = Path(record.archive_path)
        with zipfile.ZipFile(archive_path) as zf:
            frame_indices_np = self._sample_frame_indices(len(record.frame_names))
            selected_members = [record.frame_names[int(idx)] for idx in frame_indices_np.tolist()]
            frames = [self._load_frame(zf, member) for member in selected_members]

        frames_np = np.stack(frames, axis=0)
        video = preprocess_video_rgb_uint8(frames_np, self.resolution, value_range="minus_one_to_one")
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        virtual_video_path = (
            self.root
            / "_virtual"
            / record.top_split
            / record.physics_group
            / f"{record.sample_name}.mp4"
        )
        metadata = {
            "sample_key": record.key,
            "sample_name": record.sample_name,
            "stem": record.sample_name,
            "top_split": record.top_split,
            "physics_group": record.physics_group,
            "scene_name": record.scene_name,
            "camera_name": record.camera_name,
            "source_zip": str(archive_path),
            "source_frame_count": int(len(record.frame_names)),
            "sampled_frame_indices": frame_indices_np.tolist(),
            "sampling_strategy": self.sampling_strategy,
            "fps": 30,
            "resolution": [int(self.resolution[0]), int(self.resolution[1])],
            "caption": record.caption,
            "frame_count_hint": record.frame_count_hint,
            "sequence_info": {
                "total_frames": int(len(record.frame_names)),
                "start_frame": 0,
                "end_frame": int(len(record.frame_names) - 1),
                "frame_rate": 30.0,
            },
        }
        return {
            "video": video,
            "context_video": video[:, context_indices].contiguous(),
            "caption": record.caption,
            "video_path": str(virtual_video_path),
            "sample_name": record.sample_name,
            "stem": record.sample_name,
            "frame_indices": torch.arange(self.num_frames, dtype=torch.long),
            "context_frame_indices": context_indices,
            "num_context_frames": int(self.num_context_frames),
            "metadata": metadata,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(min(self.max_retry_samples, len(self.samples))):
            record = self.samples[(idx + attempt) % len(self.samples)]
            try:
                return self._load_sample(record)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(
            f"failed to load PhysInOne sample after {self.max_retry_samples} attempts"
        ) from last_error
