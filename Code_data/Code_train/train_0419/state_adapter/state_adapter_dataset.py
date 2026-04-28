"""Dataset for oracle-state Wan adapter training."""

from __future__ import annotations

import json
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


def load_prompt(meta: Dict[str, object]) -> str:
    prompt = str(meta.get("prompt", "")).strip()
    if prompt:
        return prompt
    return "a rigid object motion scene"


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
        self.dataset_root = Path(dataset_root)
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

        all_window_dirs = sorted(path.parent for path in self.dataset_root.rglob("pair_meta.json"))
        if not all_window_dirs:
            raise FileNotFoundError(f"No oracle window data found under {self.dataset_root}")
        self.window_records = self._build_window_records(all_window_dirs)
        self.window_dirs = [record["window_dir"] for record in self.window_records]
        self.motion_complexity_labels = [str(record["motion_complexity"]["label"]) for record in self.window_records]
        self.motion_complexity_summary = summarize_motion_complexity(self.motion_complexity_labels)
        self.object_count_summary = self._summarize_field("object_count")
        self.future_collision_type_summary = self._summarize_field("future_collision_type_bucket")
        self.future_collision_bucket_summary = self._summarize_field("future_collision_bucket")
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

    def _build_window_records(self, window_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for window_dir in window_dirs:
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
        }
