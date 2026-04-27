"""Dataset for oracle-state Wan adapter training."""

from __future__ import annotations

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
    ):
        self.dataset_root = Path(dataset_root)
        self.dataset_repeat = int(dataset_repeat)
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
        }
