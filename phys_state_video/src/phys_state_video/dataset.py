from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .schemas import EpisodeArrays
from .utils import hash_prompt_tokens, require_torch


class NpzEpisodeDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if self.root.is_file():
            self.files = [self.root]
        else:
            self.files = sorted(self.root.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz episodes found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> EpisodeArrays:
        path = self.files[index]
        payload = np.load(path, allow_pickle=False)
        meta_path = path.with_suffix(".json")
        prompt = ""
        if meta_path.exists():
            prompt = json.loads(meta_path.read_text()).get("prompt", "")
        episode = EpisodeArrays(
            context_frames=payload["context_frames"].astype(np.float32),
            future_frames=payload["future_frames"].astype(np.float32),
            context_states=payload["context_states"].astype(np.float32),
            future_states=payload["future_states"].astype(np.float32),
            context_boxes=payload["context_boxes"].astype(np.float32),
            future_boxes=payload["future_boxes"].astype(np.float32),
            appearance=payload["appearance"].astype(np.float32),
            camera=payload["camera"].astype(np.float32),
            prompt=prompt,
        )
        episode.validate()
        return episode


def collate_episodes(batch: List[EpisodeArrays]) -> Dict[str, object]:
    torch = require_torch()
    prompt_token_lists = [hash_prompt_tokens(item.prompt, 4096) for item in batch]
    max_prompt_len = max(len(tokens) for tokens in prompt_token_lists)
    prompt_token_ids = np.zeros((len(batch), max_prompt_len), dtype=np.int64)
    prompt_token_mask = np.zeros((len(batch), max_prompt_len), dtype=np.float32)
    for idx, tokens in enumerate(prompt_token_lists):
        prompt_token_ids[idx, :len(tokens)] = np.asarray(tokens, dtype=np.int64)
        prompt_token_mask[idx, :len(tokens)] = 1.0
    return {
        "context_frames": torch.from_numpy(np.stack([item.context_frames for item in batch], axis=0)),
        "future_frames": torch.from_numpy(np.stack([item.future_frames for item in batch], axis=0)),
        "context_states": torch.from_numpy(np.stack([item.context_states for item in batch], axis=0)),
        "future_states": torch.from_numpy(np.stack([item.future_states for item in batch], axis=0)),
        "context_boxes": torch.from_numpy(np.stack([item.context_boxes for item in batch], axis=0)),
        "future_boxes": torch.from_numpy(np.stack([item.future_boxes for item in batch], axis=0)),
        "appearance": torch.from_numpy(np.stack([item.appearance for item in batch], axis=0)),
        "camera": torch.from_numpy(np.stack([item.camera for item in batch], axis=0)),
        "prompt_token_ids": torch.from_numpy(prompt_token_ids),
        "prompt_token_mask": torch.from_numpy(prompt_token_mask),
        "prompts": [item.prompt for item in batch],
    }
