"""Dataset for oracle-state Wan adapter training."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN0419_ROOT = SCRIPT_DIR.parent
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

from dataset import WAN_SPATIAL_DIVISIBILITY
from diffsynth.core.data.operators import ImageCropAndResize


def stable_hash_id(text: str, vocab_size: int) -> int:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(vocab_size)


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
        object_vocab_size: int = 65536,
        text_vocab_size: int = 4096,
        use_normalized_state: bool = True,
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_repeat = int(dataset_repeat)
        self.object_vocab_size = int(object_vocab_size)
        self.text_vocab_size = int(text_vocab_size)
        self.use_normalized_state = bool(use_normalized_state)
        self.load_from_cache = False
        self.frame_processor = ImageCropAndResize(
            height=height,
            width=width,
            max_pixels=max_pixels,
            height_division_factor=WAN_SPATIAL_DIVISIBILITY,
            width_division_factor=WAN_SPATIAL_DIVISIBILITY,
        )

        self.window_dirs = sorted(path.parent for path in self.dataset_root.rglob("pair_meta.json"))
        if not self.window_dirs:
            raise FileNotFoundError(f"No oracle window data found under {self.dataset_root}")

    def __len__(self) -> int:
        return len(self.window_dirs) * self.dataset_repeat

    def _load_frames(self, frame_paths: Sequence[str]) -> List[Image.Image]:
        frames: List[Image.Image] = []
        for path in frame_paths:
            image = Image.open(path).convert("RGB")
            frames.append(self.frame_processor(image))
        return frames

    def __getitem__(self, index: int) -> Dict[str, object]:
        window_dir = self.window_dirs[index % len(self.window_dirs)]
        meta = json.loads((window_dir / "pair_meta.json").read_text(encoding="utf-8"))
        state = torch.load  # silence lint in environments without np typing
        del state
        with torch.no_grad():
            npz = torch.from_numpy  # silence lint
            del npz
        import numpy as np

        payload = np.load(window_dir / "state_pair.npz")
        state_key = "y_state_norm" if self.use_normalized_state else "y_state_raw"
        future_state = torch.from_numpy(payload[state_key]).float()
        future_visibility = torch.from_numpy(payload["y_visibility"]).float()

        objects = meta.get("objects", [])
        object_id_tokens = []
        role_tokens = []
        source_tokens = []
        category_tokens = []
        for obj in objects:
            obj = dict(obj) if isinstance(obj, dict) else {}
            object_id_tokens.append(
                stable_hash_id(
                    obj.get("source_object_id", obj.get("object_id", "unknown")),
                    self.object_vocab_size,
                )
            )
            role_tokens.append(
                stable_hash_id(obj.get("role", "unknown"), self.text_vocab_size)
            )
            source_tokens.append(
                stable_hash_id(
                    obj.get("source_tag", obj.get("dataset_source", "unknown")),
                    self.text_vocab_size,
                )
            )
            category_tokens.append(
                stable_hash_id(obj.get("category", obj.get("name", "unknown")), self.text_vocab_size)
            )

        num_objects = int(future_state.shape[1])
        while len(object_id_tokens) < num_objects:
            object_id_tokens.append(stable_hash_id(f"pad_object_{len(object_id_tokens)}", self.object_vocab_size))
            role_tokens.append(stable_hash_id("unknown", self.text_vocab_size))
            source_tokens.append(stable_hash_id("unknown", self.text_vocab_size))
            category_tokens.append(stable_hash_id("unknown", self.text_vocab_size))

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
            "oracle_visibility": future_visibility,
            "oracle_object_id_tokens": torch.tensor(object_id_tokens[:num_objects], dtype=torch.long),
            "oracle_role_tokens": torch.tensor(role_tokens[:num_objects], dtype=torch.long),
            "oracle_source_tokens": torch.tensor(source_tokens[:num_objects], dtype=torch.long),
            "oracle_category_tokens": torch.tensor(category_tokens[:num_objects], dtype=torch.long),
            "future_len": int(meta["future_len"]),
            "context_len": int(meta["context_len"]),
            "window_dir": str(window_dir),
        }
