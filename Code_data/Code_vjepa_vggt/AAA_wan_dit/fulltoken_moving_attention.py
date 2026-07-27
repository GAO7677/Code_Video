#!/usr/bin/env python3
"""Exact compact all-token and moving-trajectory self-attention statistics."""

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


FULL_FEATURE_NAMES = (
    "entropy",
    "same_frame_mass",
    "local_enrichment",
    "context_enrichment",
    "history_bias",
    "mean_time_distance",
    "aligned_enrichment",
    "exact_self_mass",
)

OBJECT_FEATURE_NAMES = (
    "entropy",
    "same_frame_mass",
    "context_enrichment",
    "history_bias",
    "mean_time_distance",
    "trajectory_enrichment",
    "shift_enrichment",
    "shuffle_enrichment",
    "trajectory_selectivity_log2",
    "fixed_position_enrichment",
)


def _group_coords(
    coords: tuple[tuple[int, int, int], ...],
    *,
    grid: tuple[int, int, int],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    times, grid_h, grid_w = grid
    grouped: list[list[tuple[int, int]]] = [[] for _ in range(times)]
    for time, row, column in coords:
        if not (0 <= time < times and 0 <= row < grid_h and 0 <= column < grid_w):
            raise ValueError(f"trajectory coordinate {(time, row, column)} outside {grid}")
        point = (int(row), int(column))
        if point not in grouped[int(time)]:
            grouped[int(time)].append(point)
    return tuple(tuple(points) for points in grouped)


def _matched_shuffle(
    grouped: tuple[tuple[tuple[int, int], ...], ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Roll trajectory shapes in time while preserving each target token count."""

    output = []
    times = len(grouped)
    for target_time, target in enumerate(grouped):
        if not target:
            output.append(())
            continue
        source = next(
            (
                grouped[(target_time + offset) % times]
                for offset in range(1, times + 1)
                if grouped[(target_time + offset) % times]
            ),
            target,
        )
        count = len(target)
        output.append(tuple(source[index % len(source)] for index in range(count)))
    return tuple(output)


def _trajectory_indices(
    grouped: tuple[tuple[tuple[int, int], ...], ...],
    *,
    grid: tuple[int, int, int],
) -> tuple[torch.Tensor, ...]:
    _, _, grid_w = grid
    return tuple(
        torch.tensor(
            [time * grid[1] * grid_w + row * grid_w + column for row, column in points],
            dtype=torch.long,
        )
        for time, points in enumerate(grouped)
    )


def _mean_valid(values: torch.Tensor, valid: torch.Tensor, dim: tuple[int, ...]) -> torch.Tensor:
    weights = valid.to(values.dtype)
    numerator = torch.where(valid, values, torch.zeros_like(values)).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


@torch.no_grad()
def fulltoken_moving_statistics(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    trajectory_coords: tuple[tuple[int, int, int], ...],
    grid: tuple[int, int, int],
    query_chunk: int = 64,
) -> dict[str, np.ndarray]:
    """Reduce exact softmax(QK) online without retaining the token matrix."""

    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    if q_heads.shape != k_heads.shape:
        raise ValueError(f"Q/K shape mismatch: {q_heads.shape} vs {k_heads.shape}")
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    times, grid_h, grid_w = (int(value) for value in grid)
    spatial = grid_h * grid_w
    if token_count != times * spatial:
        raise ValueError(f"Q/K have {token_count} tokens, expected grid {grid}")
    if query_chunk <= 0:
        raise ValueError("query_chunk must be positive")

    device = q_heads.device
    scale = 1.0 / math.sqrt(float(head_dim))
    token_ids = torch.arange(token_count, device=device, dtype=torch.long)
    key_times = torch.div(token_ids, spatial, rounding_mode="floor")
    full_by_time = torch.zeros(
        (heads, times, len(FULL_FEATURE_NAMES)),
        device=device,
        dtype=torch.float32,
    )
    temporal_sum = torch.zeros(
        (heads, times, times), device=device, dtype=torch.float32
    )
    query_counts = torch.zeros(times, device=device, dtype=torch.float32)
    k_t = k_heads.transpose(-1, -2)

    for start in range(0, token_count, int(query_chunk)):
        stop = min(start + int(query_chunk), token_count)
        chunk_ids = token_ids[start:stop]
        chunk = int(stop - start)
        query_times = torch.div(chunk_ids, spatial, rounding_mode="floor")
        spatial_ids = chunk_ids % spatial
        rows = torch.div(spatial_ids, grid_w, rounding_mode="floor")
        columns = spatial_ids % grid_w
        scores = torch.matmul(q_heads[:, start:stop], k_t) * scale
        probabilities = torch.softmax(scores.float(), dim=-1)
        temporal = probabilities.view(heads, chunk, times, spatial).sum(dim=-1)

        temporal_sum.index_add_(1, query_times, temporal)
        query_counts.index_add_(
            0, query_times, torch.ones(chunk, device=device, dtype=torch.float32)
        )
        chunk_index = torch.arange(chunk, device=device)
        same_frame = temporal[:, chunk_index, query_times]
        context = temporal[:, :, : min(2, times)].sum(dim=-1)
        past_mask = key_times.view(1, 1, -1) < query_times.view(1, -1, 1)
        future_mask = key_times.view(1, 1, -1) > query_times.view(1, -1, 1)
        past = (probabilities * past_mask).sum(dim=-1)
        future = (probabilities * future_mask).sum(dim=-1)
        time_distance = (
            probabilities
            * (key_times.view(1, 1, -1) - query_times.view(1, -1, 1))
            .abs()
            .to(torch.float32)
        ).sum(dim=-1)
        entropy = -(
            probabilities.clamp_min(1.0e-30)
            * probabilities.clamp_min(1.0e-30).log()
        ).sum(dim=-1) / math.log(float(token_count))
        exact_self = probabilities.gather(
            2, chunk_ids.view(1, chunk, 1).expand(heads, chunk, 1)
        ).squeeze(-1)

        all_times = torch.arange(times, device=device, dtype=torch.long)
        aligned_indices = (
            all_times.view(1, times) * spatial + spatial_ids.view(chunk, 1)
        )
        aligned_valid = all_times.view(1, times) != query_times.view(chunk, 1)
        aligned_mass = (
            probabilities.gather(
                2, aligned_indices.view(1, chunk, times).expand(heads, chunk, times)
            )
            * aligned_valid.view(1, chunk, times)
        ).sum(dim=-1)
        aligned_enrichment = aligned_mass / ((times - 1) / float(token_count))

        offsets = torch.tensor(
            ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1),
             (1, -1), (1, 0), (1, 1)),
            device=device,
            dtype=torch.long,
        )
        local_rows = rows.view(chunk, 1) + offsets[:, 0].view(1, 9)
        local_columns = columns.view(chunk, 1) + offsets[:, 1].view(1, 9)
        local_valid = (
            (local_rows >= 0)
            & (local_rows < grid_h)
            & (local_columns >= 0)
            & (local_columns < grid_w)
        )
        local_indices = (
            query_times.view(chunk, 1) * spatial
            + local_rows.clamp(0, grid_h - 1) * grid_w
            + local_columns.clamp(0, grid_w - 1)
        )
        local_mass = (
            probabilities.gather(
                2, local_indices.view(1, chunk, 9).expand(heads, chunk, 9)
            )
            * local_valid.view(1, chunk, 9)
        ).sum(dim=-1)
        local_enrichment = local_mass / (
            local_valid.sum(dim=1).clamp_min(1).to(torch.float32).view(1, chunk)
            / float(token_count)
        )
        context_enrichment = context / (min(2, times) / float(times))
        feature_values = torch.stack(
            (
                entropy,
                same_frame,
                local_enrichment,
                context_enrichment,
                past - future,
                time_distance,
                aligned_enrichment,
                exact_self,
            ),
            dim=-1,
        )
        full_by_time.index_add_(1, query_times, feature_values)
        del scores, probabilities, temporal, feature_values

    temporal_matrix = temporal_sum / query_counts.view(1, times, 1).clamp_min(1.0)
    full_by_time /= query_counts.view(1, times, 1).clamp_min(1.0)
    predicted = torch.arange(times, device=device) >= min(2, times)
    full_features = torch.empty(
        (heads, len(FULL_FEATURE_NAMES)), device=device, dtype=torch.float32
    )
    for feature_index, name in enumerate(FULL_FEATURE_NAMES):
        use = predicted if name == "context_enrichment" else torch.ones_like(predicted)
        full_features[:, feature_index] = full_by_time[:, use, feature_index].mean(dim=1)

    grouped = _group_coords(trajectory_coords, grid=grid)
    trajectory_valid_times = torch.tensor(
        [bool(points) for points in grouped],
        device=device,
        dtype=torch.bool,
    )
    shifted = tuple(
        tuple(((row + grid_h // 2) % grid_h, (column + grid_w // 2) % grid_w)
              for row, column in points)
        for points in grouped
    )
    shuffled = _matched_shuffle(grouped)
    trajectory_cpu = _trajectory_indices(grouped, grid=grid)
    shift_cpu = _trajectory_indices(shifted, grid=grid)
    shuffle_cpu = _trajectory_indices(shuffled, grid=grid)
    trajectory = tuple(indices.to(device) for indices in trajectory_cpu)
    shift = tuple(indices.to(device) for indices in shift_cpu)
    shuffle = tuple(indices.to(device) for indices in shuffle_cpu)

    object_temporal = torch.full(
        (heads, times, times), torch.nan, device=device, dtype=torch.float32
    )
    trajectory_enrichment = torch.full_like(object_temporal, torch.nan)
    shift_enrichment = torch.full_like(object_temporal, torch.nan)
    shuffle_enrichment = torch.full_like(object_temporal, torch.nan)
    fixed_enrichment = torch.full_like(object_temporal, torch.nan)
    object_by_time = torch.full(
        (heads, times, len(OBJECT_FEATURE_NAMES)),
        torch.nan,
        device=device,
        dtype=torch.float32,
    )

    for query_time in range(times):
        query_indices = trajectory[query_time]
        if not int(query_indices.numel()):
            continue
        probabilities = torch.softmax(
            torch.matmul(q_heads.index_select(1, query_indices), k_t).float() * scale,
            dim=-1,
        )
        query_count = int(query_indices.numel())
        temporal = probabilities.view(
            heads, query_count, times, spatial
        ).sum(dim=-1).mean(dim=1)
        object_temporal[:, query_time] = temporal
        object_entropy = -(
            probabilities.clamp_min(1.0e-30)
            * probabilities.clamp_min(1.0e-30).log()
        ).sum(dim=-1).mean(dim=1) / math.log(float(token_count))

        for key_time in range(times):
            key_time_mass = temporal[:, key_time].clamp_min(1.0e-20)
            for indices, target in (
                (trajectory[key_time], trajectory_enrichment),
                (shift[key_time], shift_enrichment),
                (shuffle[key_time], shuffle_enrichment),
            ):
                if not int(indices.numel()):
                    continue
                mass = probabilities.index_select(2, indices).sum(dim=-1).mean(dim=1)
                conditional = mass / key_time_mass
                target[:, query_time, key_time] = conditional / (
                    int(indices.numel()) / float(spatial)
                )

            spatial_query = query_indices % spatial
            fixed_indices = key_time * spatial + spatial_query
            fixed_mass = probabilities.gather(
                2,
                fixed_indices.view(1, query_count, 1).expand(
                    heads, query_count, 1
                ),
            ).squeeze(-1).mean(dim=1)
            fixed_enrichment[:, query_time, key_time] = (
                fixed_mass / key_time_mass
            ) / (1.0 / float(spatial))

        past = temporal[:, :query_time].sum(dim=1)
        future = temporal[:, query_time + 1 :].sum(dim=1)
        context = temporal[:, : min(2, times)].sum(dim=1)
        distance = (
            temporal
            * (
                torch.arange(times, device=device, dtype=torch.float32)
                - float(query_time)
            ).abs().view(1, times)
        ).sum(dim=1)
        offdiag = torch.arange(times, device=device) != query_time
        traj = torch.nanmean(
            trajectory_enrichment[:, query_time, offdiag], dim=1
        )
        shift_value = torch.nanmean(
            shift_enrichment[:, query_time, offdiag], dim=1
        )
        shuffle_value = torch.nanmean(
            shuffle_enrichment[:, query_time, offdiag], dim=1
        )
        matched_key_times = offdiag & trajectory_valid_times
        fixed_value = fixed_enrichment[
            :, query_time, matched_key_times
        ].mean(dim=1)
        selectivity = torch.log2(
            (traj.clamp_min(1.0e-8))
            / ((0.5 * (shift_value + shuffle_value)).clamp_min(1.0e-8))
        )
        object_by_time[:, query_time] = torch.stack(
            (
                object_entropy,
                temporal[:, query_time],
                context / (min(2, times) / float(times)),
                past - future,
                distance,
                traj,
                shift_value,
                shuffle_value,
                selectivity,
                fixed_value,
            ),
            dim=-1,
        )

    object_features = torch.nanmean(object_by_time, dim=1)
    predicted_times = torch.arange(times, device=device) >= min(2, times)
    for name in ("context_enrichment", "history_bias"):
        feature_index = OBJECT_FEATURE_NAMES.index(name)
        object_features[:, feature_index] = torch.nanmean(
            object_by_time[:, predicted_times, feature_index], dim=1
        )
    return {
        "trajectory_valid_times": trajectory_valid_times.cpu().numpy(),
        "temporal_matrix": temporal_matrix.cpu().numpy().astype(np.float32),
        "full_features": full_features.cpu().numpy().astype(np.float32),
        "full_features_by_query_time": full_by_time.cpu().numpy().astype(np.float32),
        "object_temporal_mass": object_temporal.cpu().numpy().astype(np.float32),
        "trajectory_enrichment": trajectory_enrichment.cpu().numpy().astype(np.float32),
        "shift_enrichment": shift_enrichment.cpu().numpy().astype(np.float32),
        "shuffle_enrichment": shuffle_enrichment.cpu().numpy().astype(np.float32),
        "fixed_position_enrichment": fixed_enrichment.cpu().numpy().astype(np.float32),
        "object_features": object_features.cpu().numpy().astype(np.float32),
        "object_features_by_query_time": object_by_time.cpu().numpy().astype(np.float32),
    }


class FullTokenMovingRecorder:
    """Recorder compatible with the existing DiffSynth and PhysRVG hooks."""

    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        trajectory_coords: tuple[tuple[int, int, int], ...],
        query_preview: Path,
        compact_storage: bool = False,
    ) -> None:
        config.validate()
        self.config = config
        self.model_label = str(model_label)
        self.output_root = output_root.expanduser().resolve()
        self.trajectory_coords = trajectory_coords
        self.query_preview = query_preview.expanduser().resolve()
        self.compact_storage = bool(compact_storage)
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
        _group_coords(self.trajectory_coords, grid=candidate)
        self.grid = candidate

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if self.grid is None:
            raise RuntimeError("latent grid is not configured")
        if int(step) in self.captures:
            raise RuntimeError(
                f"duplicate block {self.config.block_id} capture at step {step}"
            )
        self.captures[int(step)] = fulltoken_moving_statistics(
            q,
            k,
            num_heads=num_heads,
            trajectory_coords=self.trajectory_coords,
            grid=self.grid,
            query_chunk=int(self.config.query_chunk),
        )

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("case/grid missing")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(
                f"block {self.config.block_id} missing captures for steps {missing}"
            )
        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        preview_name = "query_preview.jpg"
        shutil.copy2(self.query_preview, case_dir / preview_name)
        steps = tuple(int(step) for step in self.config.step_numbers)
        arrays: dict[str, np.ndarray] = {
            "steps_one_based": np.asarray(steps, dtype=np.int16),
            "trajectory_coords": np.asarray(self.trajectory_coords, dtype=np.int16),
            "full_feature_names": np.asarray(FULL_FEATURE_NAMES),
            "object_feature_names": np.asarray(OBJECT_FEATURE_NAMES),
        }
        stored_keys = (
            {
                "trajectory_valid_times",
                "full_features",
                "object_features_by_query_time",
            }
            if self.compact_storage
            else set(self.captures[steps[0]])
        )
        for key in stored_keys:
            arrays[key] = np.stack(
                [self.captures[step][key] for step in steps], axis=0
            )
        npz_name = f"block{self.config.block_id:02d}_fulltoken_moving.npz"
        np.savez_compressed(case_dir / npz_name, **arrays)
        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "steps_one_based": list(steps),
            "latent_grid": list(self.grid),
            "query_chunk": int(self.config.query_chunk),
            "compact_storage": self.compact_storage,
            "stored_array_keys": sorted(stored_keys),
            "softmax": "exact over all key tokens for every query token",
            "full_token_temporal_matrix": "raw attention mass; every query-time row sums to one",
            "trajectory_nulls": {
                "shift": "half-grid circular spatial shift with matched token count",
                "shuffle": "one-step cyclic time shuffle with matched token count",
            },
            "trajectory_coords": [list(coord) for coord in self.trajectory_coords],
            "trajectory_valid_times": [
                bool(any(coord[0] == time for coord in self.trajectory_coords))
                for time in range(self.grid[0])
            ],
            "trajectory_valid_time_count": len(
                {coord[0] for coord in self.trajectory_coords}
            ),
            "query_tokens_per_time": [
                sum(1 for coord in self.trajectory_coords if coord[0] == time)
                for time in range(self.grid[0])
            ],
            "full_feature_names": list(FULL_FEATURE_NAMES),
            "object_feature_names": list(OBJECT_FEATURE_NAMES),
            "feature_npz": npz_name,
            "query_preview": preview_name,
            "case_metadata": self.case_metadata,
        }
        path = case_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[fulltoken-moving] wrote {path}", flush=True)
        return path
