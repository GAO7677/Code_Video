#!/usr/bin/env python3
"""Exact per-query same-frame attention statistics for Wan self-attention."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_attention_matrix import MatrixCaptureConfig, _as_heads


@torch.no_grad()
def exact_same_frame_mass(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    grid: tuple[int, int, int],
    query_chunk: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return exact same-frame mass for every Head and original query token."""

    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    if q_heads.shape != k_heads.shape:
        raise ValueError(
            f"self-attention Q/K shapes differ: {list(q_heads.shape)} vs "
            f"{list(k_heads.shape)}"
        )
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    frames, grid_h, grid_w = (int(value) for value in grid)
    spatial_tokens = grid_h * grid_w
    if frames * spatial_tokens != token_count:
        raise ValueError(
            f"grid {grid} contains {frames * spatial_tokens} tokens, "
            f"but Q/K contain {token_count}"
        )

    device = q_heads.device
    same_frame = torch.empty(
        (heads, token_count), device=device, dtype=torch.float32
    )
    scale = 1.0 / math.sqrt(float(head_dim))
    k_t = k_heads.transpose(-1, -2)
    for start in range(0, token_count, int(query_chunk)):
        stop = min(start + int(query_chunk), token_count)
        scores = torch.matmul(q_heads[:, start:stop], k_t) * scale
        probability = torch.softmax(scores.float(), dim=-1)
        frame_mass = probability.view(
            heads, stop - start, frames, spatial_tokens
        ).sum(-1)
        query_frames = (
            torch.arange(start, stop, device=device, dtype=torch.long)
            // spatial_tokens
        )
        gather_index = query_frames.view(1, -1, 1).expand(heads, -1, 1)
        same_frame[:, start:stop] = frame_mass.gather(
            2, gather_index
        ).squeeze(2)
        del scores, probability, frame_mass, gather_index, query_frames

    metadata = {
        "num_heads": heads,
        "token_count": token_count,
        "head_dim": head_dim,
        "latent_grid": [frames, grid_h, grid_w],
        "tokens_per_frame": spatial_tokens,
        "query_chunk": int(query_chunk),
        "softmax_axis": "all_key_tokens",
        "query_sampling": "none",
        "statistic": (
            "For every original query token, sum exact softmax probability "
            "over all key tokens in the same latent frame."
        ),
    }
    return same_frame.cpu().numpy().astype(np.float32), metadata


class ExactSpatiotemporalQueryRecorder:
    """Recorder compatible with the existing DiffSynth and PhysRVG hooks."""

    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.model_label = str(model_label)
        self.output_root = output_root.expanduser().resolve()
        self.grid = grid
        self.active = False
        self.current_step: int | None = None
        self.case_key: str | None = None
        self.case_metadata: dict[str, Any] = {}
        self.captures: dict[int, dict[str, Any]] = {}

    def begin_case(
        self,
        case_key: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.active = False
        self.current_step = None
        self.case_key = str(case_key)
        self.case_metadata = dict(metadata or {})
        self.captures = {}

    def set_grid(self, grid: tuple[int, int, int]) -> None:
        candidate = tuple(int(value) for value in grid)
        if self.grid is not None and self.grid != candidate:
            raise RuntimeError(f"attention grid changed from {self.grid} to {candidate}")
        self.grid = candidate

    @torch.no_grad()
    def capture(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        num_heads: int,
    ) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if step in self.captures:
            raise RuntimeError(
                f"captured Block {self.config.block_id} step {step} more than once"
            )
        if self.grid is None:
            raise RuntimeError("latent token grid is not configured")
        print(
            f"[same-cross] model={self.model_label} case={self.case_key} "
            f"block={self.config.block_id} step={step} q={list(q.shape)} "
            f"k={list(k.shape)} heads={num_heads}",
            flush=True,
        )
        same_frame, metadata = exact_same_frame_mass(
            q,
            k,
            num_heads=int(num_heads),
            grid=self.grid,
            query_chunk=int(self.config.query_chunk),
        )
        self.captures[int(step)] = {
            "same_frame_mass": same_frame,
            "metadata": metadata,
        }

    def finalize_case(self) -> Path:
        if self.case_key is None:
            raise RuntimeError("begin_case must be called before finalize_case")
        missing = sorted(set(self.config.step_numbers) - set(self.captures))
        if missing:
            raise RuntimeError(
                f"missing Block {self.config.block_id} captures for steps {missing}"
            )
        if self.grid is None:
            raise RuntimeError("latent grid is missing")

        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        entries: list[dict[str, Any]] = []
        for step in self.config.step_numbers:
            capture = self.captures[int(step)]
            step_dir = case_dir / f"step_{step:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            npz_path = step_dir / "same_frame_mass_per_query.npz"
            same_frame = capture["same_frame_mass"]
            np.savez_compressed(
                npz_path,
                same_frame_mass=same_frame.astype(np.float32),
                different_frame_mass=(1.0 - same_frame).astype(np.float32),
            )
            metadata = capture["metadata"]
            entries.append(
                {
                    "step_number_one_based": int(step),
                    "step_index_zero_based": int(step - 1),
                    "directory": step_dir.name,
                    "statistics_npz": npz_path.name,
                    "same_frame_mass_mean_per_head": same_frame.mean(1).tolist(),
                    "metadata": metadata,
                }
            )

        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "step_numbers_one_based": list(self.config.step_numbers),
            "latent_grid": list(self.grid),
            "token_order": "time-major, then row-major height and width",
            "query_sampling": "none; all original query tokens are retained",
            "softmax": "exact over all key tokens for each query and Head",
            "case_metadata": self.case_metadata,
            "steps": entries,
        }
        summary_path = case_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[same-cross] wrote {summary_path}", flush=True)
        return summary_path
