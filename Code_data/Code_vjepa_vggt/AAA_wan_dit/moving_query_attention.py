#!/usr/bin/env python3
"""Compact exact attention features for per-frame moving-object queries."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ball_query_attention import _query_indices
from self_attention_matrix import MatrixCaptureConfig, _as_heads


FEATURE_NAMES = (
    "entropy",
    "same_frame_mass",
    "local_mass",
    "first_frame_mass",
    "history_bias",
    "mean_time_distance",
    "aligned_enrichment",
    "cross_ball_enrichment",
)


def moving_query_coords(
    trajectory: list[dict[str, float]],
    *,
    frame_shape: tuple[int, int],
    grid: tuple[int, int, int] = (13, 16, 28),
) -> tuple[tuple[int, int, int], ...]:
    frame_h, frame_w = frame_shape
    times, grid_h, grid_w = grid
    if len(trajectory) != times:
        raise ValueError(f"trajectory has {len(trajectory)} points, expected {times}")
    coords = []
    for time, point in enumerate(trajectory):
        row = int(float(point["cy"]) * grid_h / frame_h)
        column = int(float(point["cx"]) * grid_w / frame_w)
        row0 = min(max(row - 1, 0), grid_h - 2)
        column0 = min(max(column - 1, 0), grid_w - 2)
        coords.extend(
            (
                (time, row0, column0),
                (time, row0, column0 + 1),
                (time, row0 + 1, column0),
                (time, row0 + 1, column0 + 1),
            )
        )
    return tuple(coords)


@torch.no_grad()
def moving_query_features(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    query_coords: tuple[tuple[int, int, int], ...],
    grid: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    times, grid_h, grid_w = grid
    if token_count != times * grid_h * grid_w:
        raise ValueError(f"Q/K have {token_count} tokens, expected grid {grid}")

    grouped: dict[int, list[tuple[int, int, int]]] = {}
    for coord in query_coords:
        grouped.setdefault(int(coord[0]), []).append(coord)
    if tuple(sorted(grouped)) != tuple(range(times)):
        raise ValueError("moving queries must cover every latent time")
    per_time = {len(coords) for coords in grouped.values()}
    if len(per_time) != 1:
        raise ValueError("each latent time must use the same number of queries")

    ordered = tuple(coord for time in range(times) for coord in grouped[time])
    indices = torch.tensor(
        _query_indices(ordered, grid), device=q_heads.device, dtype=torch.long
    )
    queries_per_time = len(grouped[0])
    query = q_heads.index_select(1, indices).reshape(
        heads, times, queries_per_time, head_dim
    )
    scores = torch.einsum("htqd,hkd->htqk", query, k_heads)
    probabilities = torch.softmax(
        scores.float() * (1.0 / math.sqrt(float(head_dim))), dim=-1
    ).mean(dim=2)

    key_ids = torch.arange(token_count, device=q_heads.device)
    key_times = torch.div(key_ids, grid_h * grid_w, rounding_mode="floor")
    key_rows = torch.div(
        key_ids % (grid_h * grid_w), grid_w, rounding_mode="floor"
    )
    key_columns = key_ids % grid_w
    trajectory_mask = torch.zeros(
        (times, token_count), dtype=torch.bool, device=q_heads.device
    )
    local_mask = torch.zeros_like(trajectory_mask)
    aligned_mask = torch.zeros_like(trajectory_mask)
    for time in range(times):
        coords = grouped[time]
        rows = [coord[1] for coord in coords]
        columns = [coord[2] for coord in coords]
        trajectory_mask[time] = (
            (key_times == time)
            & (key_rows[:, None] == torch.tensor(rows, device=q_heads.device)).any(1)
            & (
                key_columns[:, None]
                == torch.tensor(columns, device=q_heads.device)
            ).any(1)
        )
        local_mask[time] = (
            (key_times == time)
            & (key_rows >= max(0, min(rows) - 1))
            & (key_rows < min(grid_h, max(rows) + 2))
            & (key_columns >= max(0, min(columns) - 1))
            & (key_columns < min(grid_w, max(columns) + 2))
        )
        aligned_mask[time] = (
            (key_times != time)
            & (key_rows[:, None] == torch.tensor(rows, device=q_heads.device)).any(1)
            & (
                key_columns[:, None]
                == torch.tensor(columns, device=q_heads.device)
            ).any(1)
        )

    values: dict[str, list[torch.Tensor]] = {name: [] for name in FEATURE_NAMES}
    for query_time in range(times):
        probability = probabilities[:, query_time]
        same = key_times == query_time
        first = key_times == 0
        past = key_times < query_time
        future = key_times > query_time
        cross_trajectory = trajectory_mask.clone()
        cross_trajectory[query_time] = False
        cross_trajectory = cross_trajectory.any(0)
        aligned = aligned_mask[query_time]
        values["entropy"].append(
            -(probability.clamp_min(1.0e-30) * probability.clamp_min(1.0e-30).log())
            .sum(1)
            / math.log(token_count)
        )
        values["same_frame_mass"].append(probability[:, same].sum(1))
        values["local_mass"].append(probability[:, local_mask[query_time]].sum(1))
        values["first_frame_mass"].append(probability[:, first].sum(1))
        values["history_bias"].append(
            probability[:, past].sum(1) - probability[:, future].sum(1)
        )
        values["mean_time_distance"].append(
            (
                probability
                * (key_times - query_time).abs().to(probability.dtype)[None]
            ).sum(1)
        )
        values["aligned_enrichment"].append(
            probability[:, aligned].sum(1)
            / (float(aligned.sum()) / token_count)
        )
        values["cross_ball_enrichment"].append(
            probability[:, cross_trajectory].sum(1)
            / (float(cross_trajectory.sum()) / token_count)
        )
    return {
        name: torch.stack(items, dim=1).mean(1).cpu().numpy().astype(np.float32)
        for name, items in values.items()
    }


class MovingQueryFeatureRecorder:
    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        query_coords: tuple[tuple[int, int, int], ...],
        query_preview: Path,
    ) -> None:
        config.validate()
        self.config = config
        self.model_label = model_label
        self.output_root = output_root.expanduser().resolve()
        self.query_coords = query_coords
        self.query_preview = query_preview.expanduser().resolve()
        self.grid: tuple[int, int, int] | None = None
        self.active = False
        self.current_step: int | None = None
        self.case_key: str | None = None
        self.case_metadata: dict[str, Any] = {}
        self.captures: dict[int, dict[str, np.ndarray]] = {}

    def begin_case(self, case_key: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.case_key = str(case_key)
        self.case_metadata = dict(metadata or {})
        self.captures = {}

    def set_grid(self, grid: tuple[int, int, int]) -> None:
        candidate = tuple(int(value) for value in grid)
        _query_indices(self.query_coords, candidate)
        self.grid = candidate

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if self.grid is None:
            raise RuntimeError("latent grid is not configured")
        self.captures[int(step)] = moving_query_features(
            q,
            k,
            num_heads=num_heads,
            query_coords=self.query_coords,
            grid=self.grid,
        )

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("case/grid missing")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(f"missing moving-query captures for steps {missing}")
        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.query_preview, case_dir / "moving_query_preview.jpg")
        entries = []
        for step in self.config.step_numbers:
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(exist_ok=True)
            name = f"block{self.config.block_id:02d}_moving_query_features.npz"
            np.savez_compressed(
                step_dir / name,
                **self.captures[int(step)],
                query_coords=np.asarray(self.query_coords, dtype=np.int64),
            )
            entries.append(
                {
                    "step_number_one_based": int(step),
                    "directory": step_dir.name,
                    "features_npz": name,
                }
            )
        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "latent_grid": list(self.grid),
            "query_sampling": "2x2 moving-object tokens at every latent time",
            "query_time_reduction": "features computed per query time, then mean over 13 times",
            "softmax": "exact over all key tokens",
            "query_coords": [list(coord) for coord in self.query_coords],
            "case_metadata": self.case_metadata,
            "steps": entries,
        }
        path = case_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[moving-query-attn] wrote {path}", flush=True)
        return path
