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


def _flatten_condition_maps_to_tokens(condition_maps):
    if condition_maps is None:
        return None
    if condition_maps.dim() != 5:
        raise ValueError(
            "condition_maps must have shape [B, T, C, H, W] before flattening")
    batch, steps, channels, height, width = condition_maps.shape
    return condition_maps.permute(0, 1, 3, 4, 2).contiguous().view(
        batch, steps * height * width, channels)


def _build_1d_sincos_positions(length, dim, device, dtype):
    if length <= 0:
        raise ValueError(f"position length must be positive, got {length}")
    if dim <= 0:
        raise ValueError(f"position dim must be positive, got {dim}")
    positions = torch.arange(length, device=device, dtype=torch.float32)
    half_dim = dim // 2
    if half_dim == 0:
        return positions.unsqueeze(-1).to(dtype)
    exponent = torch.arange(half_dim, device=device, dtype=torch.float32)
    exponent = exponent / max(half_dim - 1, 1)
    omega = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * exponent)
    angles = positions.unsqueeze(-1) * omega.unsqueeze(0)
    embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embeddings.shape[-1] < dim:
        embeddings = torch.cat(
            [embeddings, torch.zeros(length, dim - embeddings.shape[-1], device=device, dtype=embeddings.dtype)],
            dim=-1,
        )
    return embeddings[:, :dim].to(dtype)


def _build_condition_map_positional_tokens(steps, height, width, dim, device, dtype):
    if min(steps, height, width) <= 0:
        raise ValueError(
            f"steps, height, width must be positive, got steps={steps}, height={height}, width={width}"
        )
    t_dim = max(dim // 3, 1)
    h_dim = max(dim // 3, 1)
    w_dim = max(dim - t_dim - h_dim, 1)
    while t_dim + h_dim + w_dim > dim:
        w_dim -= 1
    t_pos = _build_1d_sincos_positions(steps, t_dim, device=device, dtype=dtype).view(steps, 1, 1, t_dim)
    h_pos = _build_1d_sincos_positions(height, h_dim, device=device, dtype=dtype).view(1, height, 1, h_dim)
    w_pos = _build_1d_sincos_positions(width, w_dim, device=device, dtype=dtype).view(1, 1, width, w_dim)
    t_pos = t_pos.expand(steps, height, width, t_dim)
    h_pos = h_pos.expand(steps, height, width, h_dim)
    w_pos = w_pos.expand(steps, height, width, w_dim)
    return torch.cat([t_pos, h_pos, w_pos], dim=-1).view(1, steps * height * width, dim)


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
            map_tokens = _flatten_condition_maps_to_tokens(condition_maps)
            encoded_map_tokens = self.map_token_encoder(map_tokens)
            position_tokens = _build_condition_map_positional_tokens(
                steps=condition_maps.shape[1],
                height=condition_maps.shape[3],
                width=condition_maps.shape[4],
                dim=encoded_map_tokens.shape[-1],
                device=encoded_map_tokens.device,
                dtype=encoded_map_tokens.dtype,
            )
            tokens.append(encoded_map_tokens + position_tokens)

        if not tokens:
            return None
        return torch.cat(tokens, dim=1)
