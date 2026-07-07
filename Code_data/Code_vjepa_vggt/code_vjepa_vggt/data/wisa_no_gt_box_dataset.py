from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, sample_frame_indices


_SPLIT_NAMES = {"train", "val", "test", "all"}


@dataclass(slots=True)
class WisaNoGTBoxRecord:
    video_name: str
    video_path: str
    prompt: str
    label: str
    metadata_index: int
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration: float | None = None
    frame_count_hint: int | None = None

    @property
    def key(self) -> str:
        return self.video_name


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _caption_from_entry(entry: dict[str, Any], default: str) -> str:
    for key in ("captions", "caption", "prompt", "description"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            parts = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if parts:
                return " ".join(parts[:4])
    return default


def _frame_count_hint_from_entry(entry: dict[str, Any]) -> int | None:
    fps = _coerce_float(entry.get("fps"))
    duration = _coerce_float(entry.get("duration"))
    if fps is None or duration is None:
        return None
    if fps <= 0.0 or duration <= 0.0:
        return None
    return max(int(round(fps * duration)), 1)


class WisaNoGTBoxDataset(Dataset):
    """WISA-80K raw-video adapter for train0705 no-GT-box training.

    Expected layout:

    - ``root/data/wisa-80k.json`` metadata downloaded from Hugging Face.
    - Actual mp4 files stored under ``videos_root``. By default this is
      ``<root>/videos`` because the current HF dataset repo only contains
      metadata, not the videos themselves.

    Returned fields intentionally match the no-GT-box branch contract:

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
        videos_root: str | Path | None = None,
        metadata_path: str | Path | None = None,
        num_frames: int = 24,
        num_context_frames: int = 8,
        sampling_strategy: str = "prefix",
        seed: int = 42,
        labels: list[str] | None = None,
        init_scan_limit: int | None = None,
        cache_root: str | Path = "/data/gaoya/agent-data/cache/wisa_no_gt_box_dataset",
        split_train_ratio: float = 0.9,
        split_val_ratio: float = 0.05,
        max_retry_samples: int = 8,
    ) -> None:
        self.root = Path(root)
        self.split = str(split).strip().lower()
        if self.split not in _SPLIT_NAMES:
            raise ValueError(f"unsupported split={split!r}, expected one of {_SPLIT_NAMES}")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.videos_root = (
            Path(videos_root) if videos_root is not None else self.root / "videos"
        )
        self.metadata_path = (
            Path(metadata_path) if metadata_path is not None else self.root / "data" / "wisa-80k.json"
        )
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        self.seed = int(seed)
        self.labels = sorted({item.strip() for item in labels or [] if item and item.strip()})
        self.init_scan_limit = None if init_scan_limit is None else max(int(init_scan_limit), 1)
        self.cache_root = Path(cache_root)
        self.index_cache_root = self.cache_root / "indices"
        self.split_train_ratio = float(split_train_ratio)
        self.split_val_ratio = float(split_val_ratio)
        self.max_retry_samples = max(1, int(max_retry_samples))
        self._video_name_to_path: dict[str, Path] | None = None

        if not self.root.is_dir():
            raise FileNotFoundError(f"wisa root not found: {self.root}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"wisa metadata json not found: {self.metadata_path}. "
                f"Expected the HF metadata file under {self.root / 'data' / 'wisa-80k.json'}."
            )
        if not self.videos_root.is_dir():
            raise FileNotFoundError(
                f"wisa videos root not found: {self.videos_root}. "
                "The current HF dataset repo only ships metadata; place mp4 files under this "
                "directory or pass --wisa_videos_root."
            )
        if self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")
        if self.num_context_frames <= 0:
            raise ValueError(
                f"num_context_frames must be positive, got {self.num_context_frames}"
            )
        if self.num_context_frames > self.num_frames:
            raise ValueError(
                f"num_context_frames={self.num_context_frames} exceeds num_frames={self.num_frames}"
            )
        if self.sampling_strategy not in {"prefix", "uniform"}:
            raise ValueError(
                f"unsupported sampling_strategy={self.sampling_strategy!r}, expected prefix/uniform"
            )
        if not 0.0 < self.split_train_ratio < 1.0:
            raise ValueError(
                f"split_train_ratio must be in (0,1), got {self.split_train_ratio}"
            )
        if not 0.0 <= self.split_val_ratio < 1.0:
            raise ValueError(
                f"split_val_ratio must be in [0,1), got {self.split_val_ratio}"
            )
        if self.split_train_ratio + self.split_val_ratio >= 1.0:
            raise ValueError("split_train_ratio + split_val_ratio must be < 1.0")

        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.samples = self._build_index()
        if self.init_scan_limit is not None:
            self.samples = self.samples[: self.init_scan_limit]
        if not self.samples:
            label_msg = f", labels={self.labels}" if self.labels else ""
            raise RuntimeError(
                f"no wisa samples found for split={self.split}{label_msg} under {self.root}. "
                f"Check whether mp4 files matching video_name are present under {self.videos_root}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _index_cache_path(self) -> Path:
        label_key = ",".join(self.labels) if self.labels else "__all__"
        config_key = (
            f"root={self.root.resolve()}|split={self.split}|labels={label_key}|"
            f"metadata={self.metadata_path.resolve()}|videos_root={self.videos_root.resolve()}|"
            f"num_frames={self.num_frames}|ctx={self.num_context_frames}|"
            f"train={self.split_train_ratio:.6f}|val={self.split_val_ratio:.6f}"
        )
        digest = hashlib.sha1(config_key.encode("utf-8")).hexdigest()[:16]
        return self.index_cache_root / f"wisa_index_{digest}.json"

    def _sample_split_name(self, key: str) -> str:
        u = _stable_unit_interval(key)
        if u < self.split_train_ratio:
            return "train"
        if u < self.split_train_ratio + self.split_val_ratio:
            return "val"
        return "test"

    def _build_video_name_index(self) -> dict[str, Path]:
        if self._video_name_to_path is not None:
            return self._video_name_to_path
        mapping: dict[str, Path] = {}
        for path in sorted(self.videos_root.rglob("*.mp4")):
            mapping.setdefault(path.name, path)
        self._video_name_to_path = mapping
        return mapping

    def _resolve_video_path(self, video_name: str) -> Path | None:
        exact_path = self.videos_root / video_name
        if exact_path.is_file():
            return exact_path
        return self._build_video_name_index().get(video_name)

    def _build_index(self) -> list[WisaNoGTBoxRecord]:
        index_path = self._index_cache_path()
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            return [WisaNoGTBoxRecord(**item) for item in payload["samples"]]

        payload = _read_json(self.metadata_path)
        if not isinstance(payload, list):
            raise RuntimeError(
                f"expected {self.metadata_path} to contain a JSON array, got {type(payload).__name__}"
            )

        label_filter = set(self.labels)
        samples: list[WisaNoGTBoxRecord] = []
        for metadata_index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                continue
            video_name = entry.get("video_name")
            if not isinstance(video_name, str) or not video_name.strip():
                continue
            video_name = video_name.strip()

            label = str(entry.get("label", "") or "").strip()
            if label_filter and label not in label_filter:
                continue

            frame_count_hint = _frame_count_hint_from_entry(entry)
            if frame_count_hint is not None and frame_count_hint < self.num_frames:
                continue

            video_path = self._resolve_video_path(video_name)
            if video_path is None:
                continue

            prompt_default = label.replace("_", " ") if label else "physics video"
            record = WisaNoGTBoxRecord(
                video_name=video_name,
                video_path=str(video_path),
                prompt=_caption_from_entry(entry, prompt_default),
                label=label,
                metadata_index=metadata_index,
                width=_coerce_int(entry.get("width")),
                height=_coerce_int(entry.get("height")),
                fps=_coerce_float(entry.get("fps")),
                duration=_coerce_float(entry.get("duration")),
                frame_count_hint=frame_count_hint,
            )
            if self.split != "all" and self._sample_split_name(record.key) != self.split:
                continue
            samples.append(record)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "root": str(self.root),
            "videos_root": str(self.videos_root),
            "metadata_path": str(self.metadata_path),
            "split": self.split,
            "samples": [asdict(record) for record in samples],
        }
        tmp_path = index_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(cache_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(index_path)
        return samples

    def _frame_indices(self, frame_count: int) -> np.ndarray:
        if frame_count < self.num_frames:
            raise RuntimeError(
                f"video has only {frame_count} frames, smaller than requested {self.num_frames}"
            )
        if self.sampling_strategy == "uniform":
            return sample_frame_indices(frame_count, self.num_frames)
        return np.arange(self.num_frames, dtype=np.int64)

    def _load_sample(self, record: WisaNoGTBoxRecord) -> dict[str, Any]:
        video_path = Path(record.video_path)
        vr = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(vr)
        frame_indices_np = self._frame_indices(frame_count)
        frames = vr.get_batch(frame_indices_np).asnumpy()
        video = preprocess_video_rgb_uint8(frames, self.resolution, value_range="minus_one_to_one")
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        metadata = {
            "sample_key": record.key,
            "video_name": record.video_name,
            "label": record.label,
            "metadata_index": record.metadata_index,
            "source_video_path": str(video_path),
            "source_frame_count": int(frame_count),
            "frame_count_hint": record.frame_count_hint,
            "sampled_frame_indices": frame_indices_np.tolist(),
            "sampling_strategy": self.sampling_strategy,
            "width": record.width,
            "height": record.height,
            "fps": record.fps,
            "duration": record.duration,
        }
        return {
            "video": video,
            "context_video": video[:, context_indices].contiguous(),
            "caption": record.prompt,
            "video_path": str(video_path),
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
            f"failed to load wisa sample after {self.max_retry_samples} attempts"
        ) from last_error
