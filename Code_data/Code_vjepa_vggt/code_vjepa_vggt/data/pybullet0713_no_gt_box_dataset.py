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


@dataclass(slots=True, frozen=True)
class PyBullet0713Record:
    case_id: str
    family_key: str
    video_path: str
    manifest_path: str
    meta_path: str | None
    caption: str
    negative_prompt: str

    @property
    def key(self) -> str:
        return f"{self.family_key}/{self.case_id}"


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _object_phrase_from_object(item: dict[str, Any], fallback_index: int) -> str:
    phrase = _first_nonempty(item.get("object_phrase"), item.get("phrase"))
    if phrase:
        return phrase
    noun = _first_nonempty(item.get("object_noun"), item.get("family_key"), item.get("shape"))
    if noun:
        material = _clean_text(item.get("material_phrase"))
        if material:
            article = "an" if material[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            return f"{article} {material} {noun}".strip()
        return noun
    return f"slot{fallback_index}"


def _entity_slots_from_meta(
    meta: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    objects = meta.get("objects", []) if isinstance(meta, dict) else []
    slots: list[dict[str, Any]] = []
    if isinstance(objects, list):
        for index, item in enumerate(objects):
            if not isinstance(item, dict):
                continue
            noun = _first_nonempty(
                item.get("object_noun"),
                item.get("family_key"),
                item.get("shape"),
                f"slot{index}",
            )
            phrase = _object_phrase_from_object(item, index)
            slots.append(
                {
                    "entity_id": int(index),
                    "slot_name": f"{noun}#{index}",
                    "object_noun": noun,
                    "object_phrase": phrase,
                    "source": "metadata_object",
                    "dynamic": bool(item.get("dynamic", False)),
                    "role": _clean_text(item.get("role")),
                    "shape": _clean_text(item.get("shape")),
                    "name": _clean_text(item.get("name")),
                    "material_key": _clean_text(item.get("material_key")),
                    "material_phrase": _clean_text(item.get("material_phrase")),
                }
            )
    if slots:
        return slots

    phrases = manifest.get("object_phrases", [])
    nouns = manifest.get("object_nouns", [])
    dynamic_phrases = set(str(value) for value in manifest.get("dynamic_object_phrases", []) or [])
    if isinstance(phrases, list):
        for index, phrase_value in enumerate(phrases):
            phrase = _clean_text(phrase_value)
            noun = _clean_text(nouns[index] if isinstance(nouns, list) and index < len(nouns) else "")
            if not noun:
                noun = f"slot{index}"
            slots.append(
                {
                    "entity_id": int(index),
                    "slot_name": f"{noun}#{index}",
                    "object_noun": noun,
                    "object_phrase": phrase or noun,
                    "source": "manifest_object_phrase",
                    "dynamic": phrase in dynamic_phrases,
                    "role": "",
                    "shape": "",
                    "name": "",
                    "material_key": "",
                    "material_phrase": "",
                }
            )
    return slots


class PyBullet0713NoGTBoxDataset(Dataset):
    """0713 PyBullet case-manifest adapter for no-GT-box Stage1B training.

    The returned sample schema intentionally matches ``PyBulletRawNoGTBoxDataset``
    and ``KubricNoGTBoxDataset`` so existing no-GT-box trainers can consume it.
    The richer per-object metadata is kept under ``metadata["entity_slots"]`` for
    text-binding fallback and visualization.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        resolution: tuple[int, int],
        *,
        num_frames: int = 49,
        num_context_frames: int = 8,
        sampling_strategy: str = "prefix",
        families: list[str] | tuple[str, ...] | None = None,
        init_scan_limit: int | None = None,
        split_train_ratio: float = 0.9,
        split_val_ratio: float = 0.05,
        max_retry_samples: int = 8,
    ) -> None:
        self.root = Path(root)
        self.split = str(split).strip().lower()
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.num_frames = int(num_frames)
        self.num_context_frames = int(num_context_frames)
        self.sampling_strategy = str(sampling_strategy).strip().lower()
        self.families = sorted(
            {
                str(value).strip()
                for value in (families or [])
                if str(value).strip()
            }
        )
        self.init_scan_limit = None if init_scan_limit is None else max(1, int(init_scan_limit))
        self.split_train_ratio = float(split_train_ratio)
        self.split_val_ratio = float(split_val_ratio)
        self.max_retry_samples = max(1, int(max_retry_samples))

        if not self.root.is_dir():
            raise FileNotFoundError(f"0713 PyBullet root not found: {self.root}")
        if self.split not in _SPLIT_NAMES:
            raise ValueError(f"unsupported split={split!r}, expected one of {_SPLIT_NAMES}")
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

        self.samples = self._build_index()
        if self.init_scan_limit is not None:
            self.samples = self.samples[: self.init_scan_limit]
        if not self.samples:
            raise RuntimeError(f"no 0713 PyBullet samples found for split={self.split} under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_split_name(self, key: str) -> str:
        value = _stable_unit_interval(key)
        if value < self.split_train_ratio:
            return "train"
        if value < self.split_train_ratio + self.split_val_ratio:
            return "val"
        return "test"

    def _manifest_paths(self) -> list[Path]:
        manifest_path = self.root / "manifest.json"
        if manifest_path.is_file():
            rows = _read_json(manifest_path)
            if not isinstance(rows, list):
                raise ValueError(f"manifest must contain a list: {manifest_path}")
            paths: list[Path] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                output_root = _clean_text(row.get("output_root"))
                case_id = _clean_text(row.get("case_id") or row.get("sample_key"))
                family_key = _clean_text(row.get("family_key"))
                if output_root:
                    paths.append(Path(output_root) / "case_manifest.json")
                elif case_id and family_key:
                    paths.append(self.root / "cases" / family_key / case_id / "case_manifest.json")
            deduped = list(dict.fromkeys(paths))
            if deduped:
                return sorted(deduped)
        return sorted((self.root / "cases").glob("F*/*/case_manifest.json"))

    def _build_index(self) -> list[PyBullet0713Record]:
        family_filter = set(self.families)
        samples: list[PyBullet0713Record] = []
        for manifest_path in self._manifest_paths():
            if not manifest_path.is_file():
                continue
            manifest = _read_json(manifest_path)
            if not isinstance(manifest, dict):
                continue
            case_id = _first_nonempty(
                manifest.get("case_id"),
                manifest.get("sample_key"),
                manifest_path.parent.name,
            )
            family_key = _first_nonempty(
                manifest.get("family_key"),
                manifest_path.parent.parent.name,
            )
            if family_filter and family_key not in family_filter:
                continue
            if self.split != "all" and self._sample_split_name(f"{family_key}/{case_id}") != self.split:
                continue
            video_path = Path(_clean_text(manifest.get("video")))
            if not video_path.is_absolute():
                video_path = manifest_path.parent / video_path
            if not video_path.is_file():
                continue
            meta_text = _clean_text(manifest.get("meta"))
            meta_path = Path(meta_text) if meta_text else None
            if meta_path is not None and not meta_path.is_absolute():
                meta_path = manifest_path.parent / meta_path
            caption = _first_nonempty(
                manifest.get("input_caption"),
                manifest.get("caption"),
                manifest.get("short_caption"),
                case_id.replace("_", " "),
            )
            samples.append(
                PyBullet0713Record(
                    case_id=case_id,
                    family_key=family_key,
                    video_path=str(video_path),
                    manifest_path=str(manifest_path),
                    meta_path=str(meta_path) if meta_path is not None and meta_path.is_file() else None,
                    caption=caption,
                    negative_prompt=_clean_text(manifest.get("negative_prompt")),
                )
            )
        return samples

    def _frame_indices(self, frame_count: int) -> np.ndarray:
        if frame_count < self.num_frames:
            raise RuntimeError(
                f"video has only {frame_count} frames, smaller than requested {self.num_frames}"
            )
        if self.sampling_strategy == "uniform":
            return sample_frame_indices(frame_count, self.num_frames)
        return np.arange(self.num_frames, dtype=np.int64)

    def _load_sample(self, record: PyBullet0713Record) -> dict[str, Any]:
        video_path = Path(record.video_path)
        manifest = _read_json(Path(record.manifest_path))
        meta = _read_json(Path(record.meta_path)) if record.meta_path else None
        if not isinstance(meta, dict):
            meta = None
        vr = VideoReader(str(video_path), ctx=cpu(0))
        frame_count = len(vr)
        frame_indices_np = self._frame_indices(frame_count)
        frames = vr.get_batch(frame_indices_np).asnumpy()
        video = preprocess_video_rgb_uint8(
            frames,
            self.resolution,
            value_range="minus_one_to_one",
        )
        context_indices = torch.arange(self.num_context_frames, dtype=torch.long)
        metadata = {
            "dataset_name": "pybullet0713",
            "sample_key": record.key,
            "case_id": record.case_id,
            "family_key": record.family_key,
            "source_video_path": str(video_path),
            "source_frame_count": int(frame_count),
            "sampled_frame_indices": frame_indices_np.tolist(),
            "sampling_strategy": self.sampling_strategy,
            "manifest_path": record.manifest_path,
            "meta_path": record.meta_path,
            "negative_prompt": record.negative_prompt,
            "object_nouns": manifest.get("object_nouns", []),
            "object_phrases": manifest.get("object_phrases", []),
            "dynamic_object_phrases": manifest.get("dynamic_object_phrases", []),
            "static_object_phrases": manifest.get("static_object_phrases", []),
            "entity_slots": _entity_slots_from_meta(meta, manifest),
            "raw_manifest": manifest,
            "raw_meta": meta,
        }
        return {
            "video": video,
            "context_video": video[:, context_indices].contiguous(),
            "caption": record.caption,
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
            f"failed to load 0713 PyBullet sample after {self.max_retry_samples} attempts"
        ) from last_error

    @property
    def dataset_stats(self) -> dict[str, Any]:
        families: dict[str, int] = {}
        for record in self.samples:
            families[record.family_key] = families.get(record.family_key, 0) + 1
        return {
            "dataset": "pybullet0713_no_gt_box",
            "root": str(self.root),
            "split": self.split,
            "num_samples": len(self.samples),
            "families": families,
            "num_frames": self.num_frames,
            "num_context_frames": self.num_context_frames,
            "sampling_strategy": self.sampling_strategy,
            "resolution": list(self.resolution),
        }
