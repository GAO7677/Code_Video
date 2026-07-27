#!/usr/bin/env python3
"""Selected-head all-token QK and softmax-attention matrix capture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from self_attention_matrix import MatrixCaptureConfig, _as_heads


@torch.no_grad()
def pool_selected_qk_matrices(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    num_heads: int,
    selected_heads: tuple[int, ...],
    output_bins: int = 512,
    query_chunk: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Pool raw QK scores and exact-softmax attention over all Q/K tokens."""

    q_heads = _as_heads(q.detach(), num_heads=num_heads)
    k_heads = _as_heads(k.detach(), num_heads=num_heads)
    if q_heads.shape != k_heads.shape:
        raise ValueError(f"Q/K shape mismatch: {q_heads.shape} vs {k_heads.shape}")
    heads, token_count, head_dim = (int(value) for value in q_heads.shape)
    if not selected_heads or len(set(selected_heads)) != len(selected_heads):
        raise ValueError("selected_heads must be non-empty and unique")
    if min(selected_heads) < 0 or max(selected_heads) >= heads:
        raise ValueError(f"selected head outside 0..{heads - 1}")
    bins = min(int(output_bins), token_count)
    if bins <= 0 or query_chunk <= 0:
        raise ValueError("output_bins and query_chunk must be positive")

    device = q_heads.device
    head_ids = torch.tensor(selected_heads, device=device, dtype=torch.long)
    q_selected = q_heads.index_select(0, head_ids)
    k_selected = k_heads.index_select(0, head_ids)
    selected_count = len(selected_heads)
    token_ids = torch.arange(token_count, device=device, dtype=torch.long)
    token_bins = torch.div(token_ids * bins, token_count, rounding_mode="floor")
    bin_counts = torch.bincount(token_bins, minlength=bins).to(torch.float32)
    raw_sum = torch.zeros(
        (selected_count, bins, bins), device=device, dtype=torch.float32
    )
    attention_sum = torch.zeros_like(raw_sum)
    k_t = k_selected.transpose(-1, -2)
    scale = 1.0 / math.sqrt(float(head_dim))

    for start in range(0, token_count, int(query_chunk)):
        stop = min(start + int(query_chunk), token_count)
        chunk = stop - start
        scores = torch.matmul(q_selected[:, start:stop], k_t).float() * scale
        probabilities = torch.softmax(scores, dim=-1)
        key_index = token_bins.view(1, 1, token_count).expand(
            selected_count, chunk, token_count
        )
        raw_key_sum = torch.zeros(
            (selected_count, chunk, bins), device=device, dtype=torch.float32
        )
        attention_key_sum = torch.zeros_like(raw_key_sum)
        raw_key_sum.scatter_add_(2, key_index, scores)
        attention_key_sum.scatter_add_(2, key_index, probabilities)
        raw_sum.index_add_(1, token_bins[start:stop], raw_key_sum)
        attention_sum.index_add_(
            1, token_bins[start:stop], attention_key_sum
        )
        del scores, probabilities, key_index, raw_key_sum, attention_key_sum

    raw_denominator = (
        bin_counts.view(1, bins, 1) * bin_counts.view(1, 1, bins)
    ).clamp_min(1.0)
    query_denominator = bin_counts.view(1, bins, 1).clamp_min(1.0)
    raw_mean = raw_sum / raw_denominator
    attention_mass = attention_sum / query_denominator
    metadata = {
        "num_heads": heads,
        "selected_heads": selected_count,
        "token_count": token_count,
        "head_dim": head_dim,
        "output_bins": bins,
        "query_chunk": int(query_chunk),
    }
    return (
        raw_mean.cpu().numpy().astype(np.float16),
        attention_mass.cpu().numpy().astype(np.float16),
        metadata,
    )


class SelectedQKMatrixRecorder:
    """Recorder compatible with existing DiffSynth and PhysRVG Q/K hooks."""

    def __init__(
        self,
        *,
        config: MatrixCaptureConfig,
        model_label: str,
        output_root: Path,
        selected_heads: tuple[int, ...],
        role_by_head: dict[int, list[str]],
    ) -> None:
        config.validate()
        self.config = config
        self.model_label = str(model_label)
        self.output_root = output_root.expanduser().resolve()
        self.selected_heads = tuple(int(value) for value in selected_heads)
        self.role_by_head = {
            int(head): tuple(str(role) for role in roles)
            for head, roles in role_by_head.items()
        }
        self.grid: tuple[int, int, int] | None = None
        self.active = False
        self.current_step: int | None = None
        self.case_key: str | None = None
        self.case_metadata: dict[str, Any] = {}
        self.captures: dict[int, tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}

    def begin_case(self, case_key: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.case_key = str(case_key)
        self.case_metadata = dict(metadata or {})
        self.captures = {}

    def set_grid(self, grid: tuple[int, int, int]) -> None:
        self.grid = tuple(int(value) for value in grid)

    @torch.no_grad()
    def capture(self, *, q: torch.Tensor, k: torch.Tensor, num_heads: int) -> None:
        step = self.current_step
        if not self.active or step is None or step not in self.config.step_numbers:
            return
        if int(step) in self.captures:
            raise RuntimeError(
                f"duplicate block {self.config.block_id} capture at step {step}"
            )
        self.captures[int(step)] = pool_selected_qk_matrices(
            q,
            k,
            num_heads=num_heads,
            selected_heads=self.selected_heads,
            output_bins=int(self.config.output_bins),
            query_chunk=int(self.config.query_chunk),
        )

    def finalize_case(self) -> Path:
        if self.case_key is None or self.grid is None:
            raise RuntimeError("case/grid missing")
        steps = tuple(int(step) for step in self.config.step_numbers)
        missing = sorted(set(steps) - set(self.captures))
        if missing:
            raise RuntimeError(
                f"block {self.config.block_id} missing captures for steps {missing}"
            )
        case_dir = self.output_root / self.model_label / self.case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        raw = np.stack([self.captures[step][0] for step in steps], axis=0)
        attention = np.stack(
            [self.captures[step][1] for step in steps], axis=0
        )
        metadata = self.captures[steps[0]][2]
        name = f"block{self.config.block_id:02d}_selected_qk.npz"
        np.savez_compressed(
            case_dir / name,
            steps_one_based=np.asarray(steps, dtype=np.int16),
            selected_heads=np.asarray(self.selected_heads, dtype=np.int16),
            raw_qk_mean=raw,
            softmax_attention_mass=attention,
        )
        summary = {
            "model": self.model_label,
            "case": self.case_key,
            "block_id": int(self.config.block_id),
            "selected_heads": list(self.selected_heads),
            "role_by_head": {
                str(head): list(roles) for head, roles in self.role_by_head.items()
            },
            "steps_one_based": list(steps),
            "latent_grid": list(self.grid),
            "matrix_npz": name,
            "raw_qk": "mean QK^T/sqrt(d) over each pooled query/key bin",
            "softmax_attention": (
                "mean exact-softmax attention mass from each pooled query bin "
                "to each pooled key bin"
            ),
            "metadata": metadata,
            "case_metadata": self.case_metadata,
        }
        path = case_dir / "summary.json"
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[selected-qk] wrote {path}", flush=True)
        return path
