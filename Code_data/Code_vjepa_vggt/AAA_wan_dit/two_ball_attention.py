#!/usr/bin/env python3
"""Exact attention maps for two identity-locked moving-object query tracks."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ball_query_attention import _query_indices
from self_attention_matrix import (
    MatrixCaptureConfig,
    _as_heads,
    pool_full_attention_matrix,
)


@torch.no_grad()
def two_track_attention_maps(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    track_coords: tuple[tuple[tuple[int, int, int], ...], ...],
    grid: tuple[int, int, int],
    selected_heads: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    times, grid_h, grid_w = grid
    if token_count != times * grid_h * grid_w:
        raise ValueError(f"Q/K have {token_count} tokens, expected grid {grid}")
    if max(selected_heads) >= heads:
        raise ValueError(f"selected head outside 0..{heads - 1}")

    output = np.full(
        (
            len(track_coords),
            len(selected_heads),
            times,
            times,
            grid_h,
            grid_w,
        ),
        np.nan,
        dtype=np.float16,
    )
    valid = np.zeros((len(track_coords), times), dtype=np.bool_)
    head_index = torch.tensor(
        selected_heads, device=q_heads.device, dtype=torch.long
    )
    scale = 1.0 / math.sqrt(float(head_dim))
    for track_index, coords in enumerate(track_coords):
        grouped = {
            time: tuple(coord for coord in coords if int(coord[0]) == time)
            for time in range(times)
        }
        for query_time, current in grouped.items():
            if not current:
                continue
            indices = torch.tensor(
                _query_indices(current, grid),
                device=q_heads.device,
                dtype=torch.long,
            )
            query = q_heads.index_select(1, indices)
            scores = torch.einsum("hqd,hkd->hqk", query, k_heads)
            probabilities = torch.softmax(
                scores.float() * scale, dim=-1
            ).mean(dim=1)
            output[track_index, :, query_time] = (
                probabilities.index_select(0, head_index)
                .reshape(len(selected_heads), times, grid_h, grid_w)
                .cpu()
                .numpy()
                .astype(np.float16)
            )
            valid[track_index, query_time] = True
    return output, valid


class TwoBallAttentionRecorder:
    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        track_names: tuple[str, ...],
        track_coords: tuple[tuple[tuple[int, int, int], ...], ...],
        selected_heads: tuple[int, ...],
        query_preview: Path,
    ) -> None:
        config.validate()
        if len(track_names) != len(track_coords):
            raise ValueError("track names and coordinates differ in length")
        if len(track_names) != 2:
            raise ValueError("two-ball recorder requires exactly two tracks")
        self.config = config
        self.model_label = str(model_label)
        self.output_root = output_root.expanduser().resolve()
        self.track_names = tuple(str(value) for value in track_names)
        self.track_coords = track_coords
        self.selected_heads = tuple(int(value) for value in selected_heads)
        self.query_preview = query_preview.expanduser().resolve()
        self.grid: tuple[int, int, int] | None = None
        self.active = False
        self.current_step: int | None = None
        self.case_key: str | None = None
        self.case_metadata: dict[str, Any] = {}
        self.captures: dict[int, dict[str, Any]] = {}

    def begin_case(
        self, case_key: str, *, metadata: dict[str, Any] | None = None
    ) -> None:
        self.case_key = str(case_key)
        self.case_metadata = dict(metadata or {})
        self.captures = {}

    def set_grid(self, grid: tuple[int, int, int]) -> None:
        candidate = tuple(int(value) for value in grid)
        for coords in self.track_coords:
            if coords:
                _query_indices(coords, candidate)
        self.grid = candidate

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if self.grid is None:
            raise RuntimeError("latent grid is not configured")
        attention, valid = two_track_attention_maps(
            q,
            k,
            num_heads=int(num_heads),
            track_coords=self.track_coords,
            grid=self.grid,
            selected_heads=self.selected_heads,
        )
        print(
            f"[two-ball-attn] block={self.config.block_id} step={step} "
            "pooling all 5824 Q/K tokens to 512 bins",
            flush=True,
        )
        block_mean, key_mass, full_metadata = pool_full_attention_matrix(
            q,
            k,
            num_heads=int(num_heads),
            output_bins=int(self.config.output_bins),
            query_chunk=int(self.config.query_chunk),
        )
        self.captures[int(step)] = {
            "attention": attention,
            "valid": valid,
            "block_mean": block_mean,
            "key_mass": key_mass,
            "full_metadata": full_metadata,
        }

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("case/grid missing")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(f"missing two-ball captures for steps {missing}")
        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.query_preview, case_dir / "two_ball_query_preview.jpg")
        entries = []
        for step in self.config.step_numbers:
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(exist_ok=True)
            capture = self.captures[int(step)]
            attention = capture["attention"]
            valid = capture["valid"]
            name = f"block{self.config.block_id:02d}_two_ball_maps.npz"
            arrays: dict[str, np.ndarray] = {
                "attention": attention,
                "valid_query_times": valid,
                "selected_heads": np.asarray(
                    self.selected_heads, dtype=np.int64
                ),
                "track_names": np.asarray(self.track_names),
            }
            for track_index, coords in enumerate(self.track_coords):
                arrays[f"track_{track_index}_query_coords"] = np.asarray(
                    coords, dtype=np.int16
                ).reshape(-1, 3)
            np.savez_compressed(step_dir / name, **arrays)
            full_name = (
                f"block{self.config.block_id:02d}_all_token_matrix.npz"
            )
            np.savez_compressed(
                step_dir / full_name,
                block_mean=capture["block_mean"].astype(np.float32),
                key_mass=capture["key_mass"].astype(np.float32),
            )
            entries.append(
                {
                    "step_number_one_based": int(step),
                    "directory": step_dir.name,
                    "maps_npz": name,
                    "full_matrix_npz": full_name,
                    "full_matrix_metadata": capture["full_metadata"],
                }
            )
        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "latent_grid": list(self.grid),
            "track_names": list(self.track_names),
            "selected_heads": list(self.selected_heads),
            "attention_shape": (
                "track, head, query_time, key_time, key_row, key_column"
            ),
            "missing_query_policy": "NaN rows plus valid_query_times mask",
            "softmax": "exact over all key tokens",
            "full_matrix_capture": (
                "all 5824 query/key tokens, exact softmax, contiguous 512-bin pooling"
            ),
            "case_metadata": self.case_metadata,
            "steps": entries,
        }
        path = case_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[two-ball-attn] wrote {path}", flush=True)
        return path
