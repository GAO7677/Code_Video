from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(slots=True)
class PredictorEpisodeArrays:
    context_frames: np.ndarray
    context_states: np.ndarray
    future_states: np.ndarray
    appearance: np.ndarray
    camera: np.ndarray
    prompt: str = ""


@dataclass(slots=True)
class PredictorFullEpisodeArrays:
    full_frames: np.ndarray
    full_states: np.ndarray
    full_boxes: np.ndarray
    appearance: np.ndarray
    camera: np.ndarray
    legacy_context_steps: int
    legacy_future_steps: int
    prompt: str = ""


class NpzPredictorDataset:
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

    def __getitem__(self, index: int) -> PredictorEpisodeArrays:
        path = self.files[index]
        meta_path = path.with_suffix(".json")
        prompt = ""
        if meta_path.exists():
            prompt = json.loads(meta_path.read_text()).get("prompt", "")
        with np.load(path, allow_pickle=False) as payload:
            return PredictorEpisodeArrays(
                context_frames=payload["context_frames"].astype(np.float32),
                context_states=payload["context_states"].astype(np.float32),
                future_states=payload["future_states"].astype(np.float32),
                appearance=payload["appearance"].astype(np.float32),
                camera=payload["camera"].astype(np.float32),
                prompt=prompt,
            )


class NpzPredictorFullDataset:
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

    def __getitem__(self, index: int) -> PredictorFullEpisodeArrays:
        path = self.files[index]
        meta_path = path.with_suffix(".json")
        prompt = ""
        if meta_path.exists():
            prompt = json.loads(meta_path.read_text()).get("prompt", "")
        with np.load(path, allow_pickle=False) as payload:
            if "full_frames" in payload:
                full_frames = payload["full_frames"].astype(np.float32)
            else:
                full_frames = np.concatenate(
                    [
                        payload["context_frames"].astype(np.float32),
                        payload["future_frames"].astype(np.float32),
                    ],
                    axis=0,
                )
            if "full_states" in payload:
                full_states = payload["full_states"].astype(np.float32)
            else:
                full_states = np.concatenate(
                    [
                        payload["context_states"].astype(np.float32),
                        payload["future_states"].astype(np.float32),
                    ],
                    axis=0,
                )
            if "full_boxes" in payload:
                full_boxes = payload["full_boxes"].astype(np.float32)
            else:
                full_boxes = np.concatenate(
                    [
                        payload["context_boxes"].astype(np.float32),
                        payload["future_boxes"].astype(np.float32),
                    ],
                    axis=0,
                )

            appearance = payload["appearance"].astype(np.float32)
            legacy_context_steps = int(payload["context_frames"].shape[0])
            legacy_future_steps = int(payload["future_states"].shape[0])
            if "camera_full" in payload:
                camera = payload["camera_full"].astype(np.float32)
            else:
                camera = payload["camera"].astype(np.float32)
                if camera.shape[0] < full_frames.shape[0]:
                    if camera.shape[0] <= 0:
                        raise ValueError(f"{path} has empty camera array and no camera_full")
                    pad = np.repeat(camera[-1:], repeats=full_frames.shape[0] - camera.shape[0], axis=0)
                    camera = np.concatenate([camera, pad.astype(np.float32)], axis=0)
                elif camera.shape[0] > full_frames.shape[0]:
                    camera = camera[: full_frames.shape[0]]

            return PredictorFullEpisodeArrays(
                full_frames=full_frames,
                full_states=full_states,
                full_boxes=full_boxes,
                appearance=appearance,
                camera=camera.astype(np.float32),
                legacy_context_steps=legacy_context_steps,
                legacy_future_steps=legacy_future_steps,
                prompt=prompt,
            )


def _pad_and_stack_time_major(arrays: List[np.ndarray], *, pad_value: float = 0.0) -> np.ndarray:
    if not arrays:
        raise ValueError("arrays must be non-empty")
    max_steps = max(int(item.shape[0]) for item in arrays)
    sample_shape = arrays[0].shape[1:]
    dtype = arrays[0].dtype
    stacked = np.full((len(arrays), max_steps, *sample_shape), pad_value, dtype=dtype)
    for index, item in enumerate(arrays):
        steps = int(item.shape[0])
        stacked[index, :steps] = item
    return stacked


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


def collate_predictor_episodes(batch: List[PredictorEpisodeArrays]) -> Dict[str, object]:
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
        "context_states": torch.from_numpy(np.stack([item.context_states for item in batch], axis=0)),
        "future_states": torch.from_numpy(np.stack([item.future_states for item in batch], axis=0)),
        "appearance": torch.from_numpy(np.stack([item.appearance for item in batch], axis=0)),
        "camera": torch.from_numpy(np.stack([item.camera for item in batch], axis=0)),
        "prompt_token_ids": torch.from_numpy(prompt_token_ids),
        "prompt_token_mask": torch.from_numpy(prompt_token_mask),
        "prompts": [item.prompt for item in batch],
    }


def collate_predictor_full_episodes(batch: List[PredictorFullEpisodeArrays]) -> Dict[str, object]:
    torch = require_torch()
    prompt_token_lists = [hash_prompt_tokens(item.prompt, 4096) for item in batch]
    max_prompt_len = max(len(tokens) for tokens in prompt_token_lists)
    prompt_token_ids = np.zeros((len(batch), max_prompt_len), dtype=np.int64)
    prompt_token_mask = np.zeros((len(batch), max_prompt_len), dtype=np.float32)
    for idx, tokens in enumerate(prompt_token_lists):
        prompt_token_ids[idx, :len(tokens)] = np.asarray(tokens, dtype=np.int64)
        prompt_token_mask[idx, :len(tokens)] = 1.0
    full_lengths = np.asarray([item.full_frames.shape[0] for item in batch], dtype=np.int64)
    return {
        "full_frames": torch.from_numpy(_pad_and_stack_time_major([item.full_frames for item in batch])),
        "full_states": torch.from_numpy(_pad_and_stack_time_major([item.full_states for item in batch])),
        "full_boxes": torch.from_numpy(_pad_and_stack_time_major([item.full_boxes for item in batch])),
        "appearance": torch.from_numpy(np.stack([item.appearance for item in batch], axis=0)),
        "camera": torch.from_numpy(_pad_and_stack_time_major([item.camera for item in batch])),
        "full_lengths": torch.from_numpy(full_lengths),
        "legacy_context_steps": torch.as_tensor([item.legacy_context_steps for item in batch], dtype=torch.int64),
        "legacy_future_steps": torch.as_tensor([item.legacy_future_steps for item in batch], dtype=torch.int64),
        "prompt_token_ids": torch.from_numpy(prompt_token_ids),
        "prompt_token_mask": torch.from_numpy(prompt_token_mask),
        "prompts": [item.prompt for item in batch],
    }
