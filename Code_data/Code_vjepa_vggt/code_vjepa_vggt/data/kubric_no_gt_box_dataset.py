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
class KubricNoGTBoxRecord:
    scenario: str
    date: str
    sample_id: str
    video_path: str
    prompt: str
    frame_count_hint: int | None = None

    @property
    def key(self) -> str:
        return f"{self.scenario}/{self.date}/{self.sample_id}"


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _caption_from_metadata(metadata: dict[str, Any], default: str) -> str:
    for key in ("input_caption", "caption", "prompt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    simulation_type = metadata.get("simulation_type")
    if isinstance(simulation_type, str) and simulation_type.strip():
        return simulation_type.strip().replace("_", " ")
    return default


def _build_caption(sample_dir: Path, metadata: dict[str, Any] | None, scenario_default: str) -> str:
    candidates = [
        sample_dir / "caption.txt",
        sample_dir.parent.parent / "common_caption_cosmos.txt",
    ]
    candidates.extend(sorted(sample_dir.parent.parent.glob("common_caption_cosmos*.txt")))
    for candidate in candidates:
        if candidate.is_file():
            text = _read_text(candidate)
            if text:
                return text
    if metadata is not None:
        return _caption_from_metadata(metadata, scenario_default)
    return scenario_default


def _frame_count_hint_from_metadata(metadata: dict[str, Any] | None) -> int | None:
    if metadata is None:
        return None
    rendering_efficiency = metadata.get("rendering_efficiency", {})
    if not isinstance(rendering_efficiency, dict):
        return None
    for key in ("total_frames", "frames_rendered"):
        value = rendering_efficiency.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, float) and value > 0:
            return int(value)
    return None


class KubricNoGTBoxDataset(Dataset):
    """Raw Kubric/PhyCo adapter for train0705 no-GT-box training.

    This dataset intentionally returns only the fields that the train0705
    no-GT-box branch consumes:

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
        scenarios: list[str] | None = None,
        init_scan_limit: int | None = None,
        cache_root: str | Path = "/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset",
        split_train_ratio: float = 0.9,
        split_val_ratio: float = 0.05,
        max_retry_samples: int = 8,
    ) -> None:
        self.root = Path(root)
        self.split = str(split).strip().lower()
        if self.split not in _SPLIT_NAMES:
            raise ValueError(f"unsupported split={split!r}, expected one of {_SPLIT_NAMES}")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        self.seed = int(seed)
        self.scenarios = sorted({item.strip() for item in scenarios or [] if item and item.strip()})
        self.init_scan_limit = None if init_scan_limit is None else max(int(init_scan_limit), 1)
        self.cache_root = Path(cache_root)
        self.index_cache_root = self.cache_root / "indices"
        self.split_train_ratio = float(split_train_ratio)
        self.split_val_ratio = float(split_val_ratio)
        self.max_retry_samples = max(1, int(max_retry_samples))

        if not self.root.is_dir():
            raise FileNotFoundError(f"kubric root not found: {self.root}")
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
            raise RuntimeError(f"no kubric samples found for split={self.split} under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _index_cache_path(self) -> Path:
        scenario_key = ",".join(self.scenarios) if self.scenarios else "__all__"
        config_key = (
            f"root={self.root.resolve()}|split={self.split}|scenario={scenario_key}|"
            f"num_frames={self.num_frames}|ctx={self.num_context_frames}|"
            f"train={self.split_train_ratio:.6f}|val={self.split_val_ratio:.6f}"
        )
        digest = hashlib.sha1(config_key.encode("utf-8")).hexdigest()[:16]
        return self.index_cache_root / f"kubric_index_{digest}.json"

    def _sample_split_name(self, key: str) -> str:
        u = _stable_unit_interval(key)
        if u < self.split_train_ratio:
            return "train"
        if u < self.split_train_ratio + self.split_val_ratio:
            return "val"
        return "test"

    def _build_index(self) -> list[KubricNoGTBoxRecord]:
        index_path = self._index_cache_path()
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            return [KubricNoGTBoxRecord(**item) for item in payload["samples"]]

        scenario_dirs = sorted(
            path
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )
        if self.scenarios:
            scenario_set = set(self.scenarios)
            scenario_dirs = [path for path in scenario_dirs if path.name in scenario_set]

        samples: list[KubricNoGTBoxRecord] = []
        for scenario_dir in scenario_dirs:
            scenario = scenario_dir.name
            scenario_default = scenario.replace("_", " ")
            for date_dir in sorted(
                path for path in scenario_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
            ):
                for sample_dir in sorted(
                    path for path in date_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
                ):
                    video_path = sample_dir / "rgba.mp4"
                    if not video_path.is_file():
                        continue
                    metadata_path = sample_dir / "metadata.json"
                    metadata = _read_json(metadata_path) if metadata_path.is_file() else None
                    frame_count_hint = _frame_count_hint_from_metadata(metadata)
                    if frame_count_hint is not None and frame_count_hint < self.num_frames:
                        continue
                    prompt = _build_caption(sample_dir, metadata, scenario_default)
                    record = KubricNoGTBoxRecord(
                        scenario=scenario,
                        date=date_dir.name,
                        sample_id=sample_dir.name,
                        video_path=str(video_path),
                        prompt=prompt,
                        frame_count_hint=frame_count_hint,
                    )
                    if self.split != "all" and self._sample_split_name(record.key) != self.split:
                        continue
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

    def _frame_indices(self, frame_count: int) -> np.ndarray:
        if frame_count < self.num_frames:
            raise RuntimeError(
                f"video has only {frame_count} frames, smaller than requested {self.num_frames}"
            )
        if self.sampling_strategy == "uniform":
            return sample_frame_indices(frame_count, self.num_frames)
        return np.arange(self.num_frames, dtype=np.int64)

    def _load_sample(self, record: KubricNoGTBoxRecord) -> dict[str, Any]:
        video_path = Path(record.video_path)
        vr = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(vr)
        frame_indices_np = self._frame_indices(frame_count)
        frames = vr.get_batch(frame_indices_np).asnumpy()
        video = preprocess_video_rgb_uint8(frames, self.resolution, value_range="minus_one_to_one")
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        metadata = {
            "sample_key": record.key,
            "scenario": record.scenario,
            "date": record.date,
            "sample_id": record.sample_id,
            "source_video_path": str(video_path),
            "source_frame_count": int(frame_count),
            "frame_count_hint": record.frame_count_hint,
            "sampled_frame_indices": frame_indices_np.tolist(),
            "sampling_strategy": self.sampling_strategy,
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
            f"failed to load kubric sample after {self.max_retry_samples} attempts"
        ) from last_error
