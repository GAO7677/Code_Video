"""Dataset for oracle-state Wan adapter training."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_DIR.parent
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

from dataset import WAN_SPATIAL_DIVISIBILITY
from diffsynth.core.data.operators import ImageCropAndResize
from motion_complexity import (
    build_inverse_frequency_weights,
    infer_motion_complexity,
    parse_motion_complexity_filter,
    summarize_motion_complexity,
)
from window_interactions import infer_window_interactions


def parse_int_filter(value: str | Sequence[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    return {int(item) for item in items if item}


def parse_str_filter(value: str | Sequence[str] | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    return {item for item in items if item}


def parse_optional_str_list(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value]
    return [item for item in items if item]


def load_prompt(meta: Dict[str, object]) -> str:
    prompt = str(meta.get("prompt", "")).strip()
    if prompt:
        return prompt
    return "a rigid object motion scene"


def _normalize_dataset_spec_item(item, default_repeat: int) -> Dict[str, object]:
    if isinstance(item, str):
        return {
            "path": item,
            "repeat": default_repeat,
        }
    if not isinstance(item, dict):
        raise TypeError(
            f"State-adapter dataset spec must be a string path or dict, got {type(item).__name__}."
        )
    spec = dict(item)
    if "path" not in spec:
        raise ValueError(f"State-adapter dataset spec is missing required key 'path': {spec}")
    spec.setdefault("repeat", default_repeat)
    return spec


def parse_dataset_root_specs(dataset_root: str | Sequence[object], default_repeat: int) -> List[Dict[str, object]]:
    if isinstance(dataset_root, (list, tuple)):
        return [_normalize_dataset_spec_item(item, default_repeat) for item in dataset_root]

    if isinstance(dataset_root, str):
        stripped = dataset_root.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            data = json.loads(stripped)
            if isinstance(data, dict) and "datasets" in data:
                data = data["datasets"]
            if not isinstance(data, list):
                data = [data]
            return [_normalize_dataset_spec_item(item, default_repeat) for item in data]

        if stripped.endswith(".json") and os.path.isfile(stripped):
            with open(stripped, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and "datasets" in data:
                data = data["datasets"]
            if not isinstance(data, list):
                data = [data]
            return [_normalize_dataset_spec_item(item, default_repeat) for item in data]

    return [_normalize_dataset_spec_item(dataset_root, default_repeat)]


def is_summary_samples_root(root: Path) -> bool:
    return root.is_dir() and (root / "summary.json").is_file() and (root / "train").is_dir()


def infer_summary_sample_dataset(path: Path) -> str:
    text = str(path).lower()
    if "movi" in text:
        return "movi-d"
    return "genesis"


class OracleStateWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        height: int,
        width: int,
        dataset_repeat: int = 1,
        max_pixels: int = 1024 * 1024,
        use_normalized_state: bool = True,
        motion_complexity_filter: str = "",
        rebalance_motion_complexity: bool = False,
        motion_complexity_rebalance_strength: float = 1.0,
        object_count_filter: str = "",
        future_collision_type_filter: str = "",
        future_collision_bucket_filter: str = "",
    ):
        self.dataset_root = dataset_root
        self.dataset_repeat = int(dataset_repeat)
        self.use_normalized_state = bool(use_normalized_state)
        self.motion_complexity_filter = parse_motion_complexity_filter(motion_complexity_filter)
        self.rebalance_motion_complexity = bool(rebalance_motion_complexity)
        self.motion_complexity_rebalance_strength = float(motion_complexity_rebalance_strength)
        self.object_count_filter = parse_int_filter(object_count_filter)
        self.future_collision_type_filter = parse_str_filter(future_collision_type_filter)
        self.future_collision_bucket_filter = parse_str_filter(future_collision_bucket_filter)
        self.load_from_cache = False
        self.frame_processor = ImageCropAndResize(
            height=height,
            width=width,
            max_pixels=max_pixels,
            height_division_factor=WAN_SPATIAL_DIVISIBILITY,
            width_division_factor=WAN_SPATIAL_DIVISIBILITY,
        )

        self.dataset_specs = parse_dataset_root_specs(dataset_root, default_repeat=1)
        self.dataset_sources = self._discover_window_sources(self.dataset_specs)
        all_window_dirs = [item["window_dir"] for item in self.dataset_sources]
        if not all_window_dirs:
            raise FileNotFoundError(f"No oracle window data found under {self.dataset_root}")
        self.window_records = self._build_window_records(self.dataset_sources)
        self.window_dirs = [record["window_dir"] for record in self.window_records]
        self.motion_complexity_labels = [str(record["motion_complexity"]["label"]) for record in self.window_records]
        self.motion_complexity_summary = summarize_motion_complexity(self.motion_complexity_labels)
        self.object_count_summary = self._summarize_field("object_count")
        self.future_collision_type_summary = self._summarize_field("future_collision_type_bucket")
        self.future_collision_bucket_summary = self._summarize_field("future_collision_bucket")
        self.dataset_source_summary = self._summarize_field("dataset_source")
        self.sample_weights = self._build_sample_weights() if self.rebalance_motion_complexity else None
        if not self.window_dirs:
            raise FileNotFoundError(
                f"No oracle window data matched motion_complexity_filter={sorted(self.motion_complexity_filter)} "
                f"under {self.dataset_root}"
            )

    def __len__(self) -> int:
        return len(self.window_dirs) * self.dataset_repeat

    def _load_motion_complexity(self, window_dir: Path, meta: Dict[str, object]) -> Dict[str, object]:
        existing = meta.get("motion_complexity")
        if isinstance(existing, dict) and "label" in existing:
            return existing

        with np.load(window_dir / "state_pair.npz") as payload:
            if "y_state_norm" in payload:
                state_norm = np.asarray(payload["y_state_norm"]).astype(np.float32)
            elif "y_state" in payload:
                state_norm = np.asarray(payload["y_state"]).astype(np.float32)
            else:
                raise KeyError(f"No y_state_norm/y_state found in {window_dir / 'state_pair.npz'}")
            visibility = payload["y_visibility"] if "y_visibility" in payload else None
            return infer_motion_complexity(state_norm=state_norm, visibility_mask=visibility)

    def _discover_window_sources(self, dataset_specs: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
        discovered: List[Dict[str, object]] = []
        for spec in dataset_specs:
            root = Path(str(spec["path"]))
            source_name = str(spec.get("name") or root.name or root)
            source_repeat = max(1, int(spec.get("repeat", 1)))
            if is_summary_samples_root(root):
                window_dirs = self._discover_window_dirs_from_summary_root(root, spec)
            else:
                window_dirs = sorted(path.parent for path in root.rglob("pair_meta.json"))
            if not window_dirs:
                raise FileNotFoundError(f"No oracle window data found under {root}")
            for _ in range(source_repeat):
                for window_dir in window_dirs:
                    discovered.append(
                        {
                            "dataset_root": root,
                            "dataset_source": source_name,
                            "window_dir": window_dir,
                        }
                    )
        return discovered

    def _discover_window_dirs_from_summary_root(self, root: Path, spec: Dict[str, object]) -> List[Path]:
        split_name = str(spec.get("summary_split", "train")).strip() or "train"
        split_root = root / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Summary split directory does not exist: {split_root}")

        simulator_types = set(parse_optional_str_list(spec.get("simulator_types")))
        collision_buckets = set(parse_optional_str_list(spec.get("collision_buckets")))
        object_count_buckets = set(parse_optional_str_list(spec.get("object_count_buckets")))
        allowed_datasets = set(parse_optional_str_list(spec.get("summary_datasets")))
        if not allowed_datasets:
            allowed_datasets = {"genesis"}
        if not collision_buckets:
            collision_buckets = {"no_collision", "env_only"}

        window_dirs: List[Path] = []
        seen: set[Path] = set()
        for samples_path in sorted(split_root.rglob("samples.txt")):
            collision_bucket = samples_path.parent.name
            object_count_bucket = samples_path.parent.parent.name if samples_path.parent.parent else ""
            simulator_type = samples_path.parent.parent.parent.name if samples_path.parent.parent.parent else ""
            if collision_buckets and collision_bucket not in collision_buckets:
                continue
            if object_count_buckets and object_count_bucket not in object_count_buckets:
                continue
            if simulator_types and simulator_type not in simulator_types:
                continue

            for raw_line in samples_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                window_dir = Path(line)
                if not window_dir.is_absolute():
                    window_dir = (samples_path.parent / window_dir).resolve()
                sample_dataset = infer_summary_sample_dataset(window_dir)
                if allowed_datasets and sample_dataset not in allowed_datasets:
                    continue
                pair_meta_path = window_dir / "pair_meta.json"
                if not pair_meta_path.is_file():
                    continue
                if window_dir in seen:
                    continue
                seen.add(window_dir)
                window_dirs.append(window_dir)
        return window_dirs

    def _build_window_records(self, window_sources: Sequence[Dict[str, object]]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for source_info in window_sources:
            window_dir = Path(str(source_info["window_dir"]))
            meta = json.loads((window_dir / "pair_meta.json").read_text(encoding="utf-8"))
            motion_complexity = self._load_motion_complexity(window_dir, meta)
            window_interactions = self._load_window_interactions(meta)
            label = str(motion_complexity["label"])
            if self.motion_complexity_filter and label not in self.motion_complexity_filter:
                continue
            object_count = int(window_interactions.get("object_count", 0))
            future_collision_type_bucket = str(window_interactions.get("future_window", {}).get("collision_type_bucket", ""))
            future_collision_bucket = str(window_interactions.get("future_bucket", ""))
            if self.object_count_filter and object_count not in self.object_count_filter:
                continue
            if self.future_collision_type_filter and future_collision_type_bucket not in self.future_collision_type_filter:
                continue
            if self.future_collision_bucket_filter and future_collision_bucket not in self.future_collision_bucket_filter:
                continue
            records.append(
                {
                    "dataset_source": str(source_info["dataset_source"]),
                    "dataset_root": Path(str(source_info["dataset_root"])),
                    "window_dir": window_dir,
                    "motion_complexity": motion_complexity,
                    "window_interactions": window_interactions,
                    "meta": meta,
                }
            )
        return records

    def _load_window_interactions(self, meta: Dict[str, object]) -> Dict[str, object]:
        existing = meta.get("window_interactions")
        if isinstance(existing, dict) and "future_bucket" in existing:
            return existing
        return infer_window_interactions(meta)

    def _summarize_field(self, field_name: str) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for record in self.window_records:
            if field_name == "object_count":
                key = str(int(record["window_interactions"].get("object_count", 0)))
            elif field_name == "future_collision_type_bucket":
                key = str(record["window_interactions"].get("future_window", {}).get("collision_type_bucket", ""))
            elif field_name == "future_collision_bucket":
                key = str(record["window_interactions"].get("future_bucket", ""))
            elif field_name == "dataset_source":
                key = str(record.get("dataset_source", ""))
            else:
                key = str(record.get(field_name, ""))
            summary[key] = int(summary.get(key, 0)) + 1
        return dict(sorted(summary.items(), key=lambda item: item[0]))

    def _build_sample_weights(self) -> List[float]:
        base_weights = build_inverse_frequency_weights(
            self.motion_complexity_labels,
            strength=self.motion_complexity_rebalance_strength,
        )
        if self.dataset_repeat <= 1:
            return base_weights
        expanded: List[float] = []
        for _ in range(self.dataset_repeat):
            expanded.extend(base_weights)
        return expanded

    def _load_frames(self, frame_paths: Sequence[str]) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for path in frame_paths:
            image = Image.open(path).convert("RGB")
            frames.append(self.frame_processor(image))
        return frames

    def __getitem__(self, index: int) -> Dict[str, object]:
        record = self.window_records[index % len(self.window_records)]
        window_dir = record["window_dir"]
        meta = record["meta"]
        motion_complexity = record["motion_complexity"]
        window_interactions = record["window_interactions"]
        state = torch.load  # silence lint in environments without np typing
        del state
        with torch.no_grad():
            npz = torch.from_numpy  # silence lint
            del npz

        with np.load(window_dir / "state_pair.npz") as payload:
            state_key = "y_state_norm" if self.use_normalized_state else "y_state_raw"
            future_state = torch.from_numpy(np.asarray(payload[state_key]).copy()).float()

        context_paths = meta["x_frame_paths"]
        future_paths = meta["y_frame_paths"]
        context_video = self._load_frames(context_paths)
        future_video = self._load_frames(future_paths)
        video = context_video + future_video

        return {
            "video": video,
            "prompt": load_prompt(meta),
            "context_video": context_video,
            "oracle_state": future_state,
            "future_len": int(meta["future_len"]),
            "context_len": int(meta["context_len"]),
            "window_dir": str(window_dir),
            "motion_complexity": str(motion_complexity["label"]),
            "motion_complexity_bucket_id": int(motion_complexity["bucket_id"]),
            "motion_complexity_score": float(motion_complexity["score"]),
            "motion_complexity_metrics": dict(motion_complexity.get("metrics", {})),
            "object_count": int(window_interactions.get("object_count", 0)),
            "future_collision_bucket": str(window_interactions.get("future_bucket", "")),
            "future_collision_type_bucket": str(window_interactions.get("future_window", {}).get("collision_type_bucket", "")),
            "future_collision_episode_count": int(window_interactions.get("future_window", {}).get("collision_episode_count", 0)),
            "future_object_environment_count": int(window_interactions.get("future_window", {}).get("object_environment_count", 0)),
            "future_object_object_count": int(window_interactions.get("future_window", {}).get("object_object_count", 0)),
            "window_interactions": dict(window_interactions),
            "dataset_source": str(record.get("dataset_source", "")),
        }
