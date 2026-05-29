# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _to_tensor(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).float()
    return torch.as_tensor(value, dtype=torch.float32)


def _ensure_batch_first(tensor):
    if tensor is None:
        return None
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError("token tensor must have shape [L, C] or [B, L, C]")


def canonicalize_state_condition(state_condition):
    if state_condition is None:
        return None
    if isinstance(state_condition, (str, Path)):
        state_condition = load_state_condition(state_condition)

    if isinstance(state_condition, torch.Tensor):
        return {"state_tokens": _ensure_batch_first(state_condition)}

    if isinstance(state_condition, np.ndarray):
        return {"state_tokens": _ensure_batch_first(_to_tensor(state_condition))}

    if not isinstance(state_condition, dict):
        raise TypeError(
            "state_condition must be a path, tensor, ndarray, or dict.")

    payload = {key: _to_tensor(value) for key, value in state_condition.items()}

    if payload.get("predicted_states") is not None and payload.get(
            "state_tokens") is None:
        pred = payload["predicted_states"]
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
        if pred.dim() != 4:
            raise ValueError(
                "predicted_states must have shape [T, N, D] or [B, T, N, D]")
        payload["state_tokens"] = pred.flatten(1, 2)

    state_tokens = payload.get("state_tokens")
    if state_tokens is not None:
        payload["state_tokens"] = _ensure_batch_first(state_tokens)

    memory_tokens = payload.get("memory_tokens")
    if memory_tokens is not None:
        payload["memory_tokens"] = _ensure_batch_first(memory_tokens)

    condition_maps = payload.get("condition_maps")
    if condition_maps is not None:
        if condition_maps.dim() == 4:
            condition_maps = condition_maps.unsqueeze(0)
        if condition_maps.dim() != 5:
            raise ValueError(
                "condition_maps must have shape [T, C, H, W] or [B, T, C, H, W]"
            )
        payload["condition_maps"] = condition_maps

    if not any(
            payload.get(key) is not None
            for key in ("state_tokens", "memory_tokens", "condition_maps")):
        raise ValueError(
            "state_condition must contain one of: state_tokens, memory_tokens, condition_maps, predicted_states"
        )
    return payload


def load_state_condition(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    if path.suffix in {".pt", ".pth", ".bin"}:
        return torch.load(path, map_location="cpu")
    raise ValueError(f"unsupported state condition format: {path.suffix}")


class WanObjectStateAdapter(nn.Module):

    def __init__(self,
                 model_dim,
                 state_token_dim=None,
                 memory_token_dim=None,
                 map_token_dim=None,
                 hidden_dim=None):
        super().__init__()
        self.model_dim = model_dim
        self.state_token_dim = state_token_dim
        self.memory_token_dim = memory_token_dim
        self.map_token_dim = map_token_dim
        hidden_dim = hidden_dim or model_dim

        self.state_token_encoder = self._build_encoder(state_token_dim,
                                                       hidden_dim)
        self.memory_token_encoder = self._build_encoder(memory_token_dim,
                                                        hidden_dim)
        self.map_token_encoder = self._build_encoder(map_token_dim, hidden_dim)

    def _build_encoder(self, input_dim, hidden_dim):
        if input_dim is None:
            return None
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.model_dim),
        )

    def get_config(self):
        return {
            "model_dim": self.model_dim,
            "state_token_dim": self.state_token_dim,
            "memory_token_dim": self.memory_token_dim,
            "map_token_dim": self.map_token_dim,
        }

    def forward(self, state_condition):
        payload = canonicalize_state_condition(state_condition)
        tokens = []

        state_tokens = payload.get("state_tokens")
        if state_tokens is not None:
            if self.state_token_encoder is None:
                raise RuntimeError("state_token_encoder is not initialized")
            tokens.append(self.state_token_encoder(state_tokens))

        memory_tokens = payload.get("memory_tokens")
        if memory_tokens is not None:
            if self.memory_token_encoder is None:
                raise RuntimeError("memory_token_encoder is not initialized")
            tokens.append(self.memory_token_encoder(memory_tokens))

        condition_maps = payload.get("condition_maps")
        if condition_maps is not None:
            if self.map_token_encoder is None:
                raise RuntimeError("map_token_encoder is not initialized")
            pooled_maps = condition_maps.flatten(-2).mean(dim=-1)
            tokens.append(self.map_token_encoder(pooled_maps))

        if not tokens:
            return None
        return torch.cat(tokens, dim=1)
