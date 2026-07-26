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
from self_attention_matrix import (
    MatrixCaptureConfig,
    _as_heads,
    pool_full_attention_matrix,
)


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


def explicit_moving_query_coords(
    coords_per_time: list[list[list[int]]],
    *,
    grid: tuple[int, int, int] = (13, 16, 28),
) -> tuple[tuple[int, int, int], ...]:
    times, grid_h, grid_w = grid
    if len(coords_per_time) != times:
        raise ValueError(
            f"query_coords_per_time has {len(coords_per_time)} entries, expected {times}"
        )
    coords: list[tuple[int, int, int]] = []
    for time, entries in enumerate(coords_per_time):
        for raw in entries:
            if len(raw) == 2:
                row, column = (int(value) for value in raw)
            elif len(raw) == 3:
                raw_time, row, column = (int(value) for value in raw)
                if raw_time != time:
                    raise ValueError(
                        f"query coordinate time {raw_time} does not match group {time}"
                    )
            else:
                raise ValueError(f"invalid query coordinate: {raw}")
            if not (0 <= row < grid_h and 0 <= column < grid_w):
                raise ValueError(
                    f"query coordinate {(time, row, column)} outside grid {grid}"
                )
            coords.append((time, row, column))
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
    valid_times = tuple(sorted(grouped))
    if not valid_times:
        raise ValueError("moving queries contain no visible latent times")
    probabilities = {}
    scale = 1.0 / math.sqrt(float(head_dim))
    for time in valid_times:
        indices = torch.tensor(
            _query_indices(tuple(grouped[time]), grid),
            device=q_heads.device,
            dtype=torch.long,
        )
        query = q_heads.index_select(1, indices)
        scores = torch.einsum("hqd,hkd->hqk", query, k_heads)
        probabilities[time] = torch.softmax(
            scores.float() * scale, dim=-1
        ).mean(dim=1)

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
    for time in valid_times:
        coords = grouped[time]
        rows = [coord[1] for coord in coords]
        columns = [coord[2] for coord in coords]
        spatial_ids = torch.tensor(
            [row * grid_w + column for row, column in zip(rows, columns)],
            device=q_heads.device,
        )
        key_spatial_ids = key_rows * grid_w + key_columns
        selected_spatial = (
            key_spatial_ids[:, None] == spatial_ids[None, :]
        ).any(dim=1)
        trajectory_mask[time] = (
            (key_times == time) & selected_spatial
        )
        local_mask[time] = (
            (key_times == time)
            & (key_rows >= max(0, min(rows) - 1))
            & (key_rows < min(grid_h, max(rows) + 2))
            & (key_columns >= max(0, min(columns) - 1))
            & (key_columns < min(grid_w, max(columns) + 2))
        )
        aligned_mask[time] = (key_times != time) & selected_spatial

    values: dict[str, list[torch.Tensor]] = {name: [] for name in FEATURE_NAMES}
    for query_time in valid_times:
        probability = probabilities[query_time]
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


@torch.no_grad()
def moving_query_attention_maps(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    query_coords: tuple[tuple[int, int, int], ...],
    grid: tuple[int, int, int],
    selected_heads: tuple[int, ...],
) -> np.ndarray:
    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    times, grid_h, grid_w = grid
    if token_count != times * grid_h * grid_w:
        raise ValueError(f"Q/K have {token_count} tokens, expected grid {grid}")
    if max(selected_heads) >= heads:
        raise ValueError(f"selected head outside 0..{heads - 1}")
    grouped = {
        time: tuple(coord for coord in query_coords if coord[0] == time)
        for time in range(times)
    }
    if any(not coords for coords in grouped.values()):
        raise ValueError("moving map queries must cover every latent time")
    scale = 1.0 / math.sqrt(float(head_dim))
    rows = []
    for time in range(times):
        indices = torch.tensor(
            _query_indices(grouped[time], grid),
            device=q_heads.device,
            dtype=torch.long,
        )
        query = q_heads.index_select(1, indices)
        scores = torch.einsum("hqd,hkd->hqk", query, k_heads)
        rows.append(torch.softmax(scores.float() * scale, dim=-1).mean(dim=1))
    probabilities = torch.stack(rows, dim=1)
    head_index = torch.tensor(
        selected_heads, device=q_heads.device, dtype=torch.long
    )
    return (
        probabilities.index_select(0, head_index)
        .reshape(len(selected_heads), times, times, grid_h, grid_w)
        .cpu()
        .numpy()
        .astype(np.float16)
    )


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
            "query_sampling": (
                "per-frame moving-object tokens; count may vary by latent time"
            ),
            "query_tokens_per_time": [
                sum(1 for coord in self.query_coords if int(coord[0]) == time)
                for time in range(int(self.grid[0]))
            ],
            "query_time_reduction": (
                "features computed per visible query time, then mean over valid times"
            ),
            "valid_query_times": [
                time
                for time in range(int(self.grid[0]))
                if any(int(coord[0]) == time for coord in self.query_coords)
            ],
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


class MovingQueryMapRecorder(MovingQueryFeatureRecorder):
    def __init__(self, *, selected_heads: tuple[int, ...], **kwargs) -> None:
        super().__init__(**kwargs)
        if not selected_heads or len(set(selected_heads)) != len(selected_heads):
            raise ValueError("selected map heads must be non-empty and unique")
        self.selected_heads = tuple(int(head) for head in selected_heads)
        self.captures: dict[int, np.ndarray] = {}

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if self.grid is None:
            raise RuntimeError("latent grid is not configured")
        self.captures[int(step)] = moving_query_attention_maps(
            q,
            k,
            num_heads=num_heads,
            query_coords=self.query_coords,
            grid=self.grid,
            selected_heads=self.selected_heads,
        )

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("case/grid missing")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(f"missing moving-query maps for steps {missing}")
        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.query_preview, case_dir / "moving_query_preview.jpg")
        entries = []
        for step in self.config.step_numbers:
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(exist_ok=True)
            name = f"block{self.config.block_id:02d}_moving_query_maps.npz"
            np.savez_compressed(
                step_dir / name,
                attention=self.captures[int(step)],
                selected_heads=np.asarray(self.selected_heads, dtype=np.int64),
                query_coords=np.asarray(self.query_coords, dtype=np.int64),
            )
            entries.append(
                {
                    "step_number_one_based": int(step),
                    "directory": step_dir.name,
                    "maps_npz": name,
                }
            )
        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "latent_grid": list(self.grid),
            "selected_heads": list(self.selected_heads),
            "attention_shape": (
                "selected_head, query_time, key_time, key_row, key_column"
            ),
            "query_sampling": "moving-object tokens at every latent time",
            "softmax": "exact over all key tokens",
            "query_coords": [list(coord) for coord in self.query_coords],
            "steps": entries,
        }
        path = case_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[moving-query-map] wrote {path}", flush=True)
        return path


class MovingQueryMapAndFullRecorder(MovingQueryMapRecorder):
    """Capture moving-query maps and the pooled full QK matrix in one forward."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.full_captures: dict[int, dict[str, Any]] = {}

    def begin_case(self, case_key: str, *, metadata: dict[str, Any] | None = None) -> None:
        super().begin_case(case_key, metadata=metadata)
        self.full_captures = {}

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        super().capture(q=q, k=k, num_heads=num_heads)
        block_mean, key_mass, metadata = pool_full_attention_matrix(
            q,
            k,
            num_heads=int(num_heads),
            output_bins=int(self.config.output_bins),
            query_chunk=int(self.config.query_chunk),
        )
        self.full_captures[int(step)] = {
            "block_mean": block_mean,
            "key_mass": key_mass,
            "metadata": metadata,
        }

    def finalize_case(self) -> Path:
        path = super().finalize_case()
        summary = json.loads(path.read_text(encoding="utf-8"))
        for entry in summary["steps"]:
            step = int(entry["step_number_one_based"])
            capture = self.full_captures[step]
            name = f"block{self.config.block_id:02d}_all_heads_token_matrix.npz"
            np.savez_compressed(
                path.parent / entry["directory"] / name,
                block_mean=capture["block_mean"].astype(np.float32),
                key_mass=capture["key_mass"].astype(np.float32),
            )
            entry["full_matrix_npz"] = name
            entry["full_matrix_metadata"] = capture["metadata"]
        summary["full_matrix_capture"] = (
            "same Q/K forward as moving-query maps; pooled to configured bins"
        )
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[moving-query-map+full] updated {path}", flush=True)
        return path
