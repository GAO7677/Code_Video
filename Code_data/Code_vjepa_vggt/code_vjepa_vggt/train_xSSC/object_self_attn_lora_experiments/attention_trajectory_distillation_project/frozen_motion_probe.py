"""Differentiable fixed-query QK correspondence utilities.

This module deliberately contains no Wan model-loading code.  It implements the
small, testable part of Frozen Motion Probe distillation:

* capture selected self-attention Q/K heads without changing attention output;
* retain every physical head and form equal/PCK-weighted aggregate heatmaps;
* turn each latent-frame heatmap into a differentiable soft-argmax trajectory;
* compare teacher and student head maps with PCK-weighted KL(teacher || student).

The caller owns the frozen DiT and is responsible for running the teacher under
``torch.no_grad()`` while retaining gradients for the student probe input.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def ordered_head_pairs(
    selected_heads_by_block: Mapping[int, Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    pairs = tuple(
        (int(block_id), int(head_id))
        for block_id, head_ids in sorted(selected_heads_by_block.items())
        for head_id in sorted({int(value) for value in head_ids})
    )
    if not pairs:
        raise ValueError("selected_heads_by_block is empty")
    return pairs


def load_pck_head_weights(
    selection_metadata: Mapping[str, Any],
    selected_heads_by_block: Mapping[int, Sequence[int]],
    *,
    score_key: str = "pck32",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and align normalized PCK weights to collector head order."""
    pairs = ordered_head_pairs(selected_heads_by_block)
    source_value = selection_metadata.get("selection_source")
    if not source_value:
        raise KeyError("head-selection metadata has no selection_source")
    source_path = Path(str(source_value)).expanduser()
    if not source_path.is_absolute():
        config_path = Path(str(selection_metadata.get("config_path", ".")))
        source_path = config_path.parent / source_path
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"PCK score source does not exist: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    expected_step = int(selection_metadata.get("ranking_step", -1))
    if int(payload.get("ranking_step", -2)) != expected_step:
        raise ValueError(
            "PCK score source ranking step mismatch: "
            f"source={payload.get('ranking_step')}, selection={expected_step}"
        )
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise TypeError("PCK score source entries must be a list")
    scores_by_pair: dict[tuple[int, int], float] = {}
    ranked_pairs: list[tuple[int, int]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError(f"invalid PCK entry: {entry!r}")
        if int(entry.get("step", expected_step)) != expected_step:
            continue
        pair = (int(entry["block"]), int(entry["head"]))
        if pair in scores_by_pair:
            raise ValueError(f"duplicate PCK entry for head {pair}")
        score = float(entry[score_key])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"invalid {score_key}={score} for head {pair}")
        scores_by_pair[pair] = score
        ranked_pairs.append(pair)
    missing = [pair for pair in pairs if pair not in scores_by_pair]
    if missing:
        raise KeyError(f"PCK score source is missing selected heads: {missing[:12]}")
    declared_count = int(selection_metadata.get("num_heads", len(pairs)))
    if declared_count != len(pairs):
        raise ValueError(
            f"selected head count mismatch: metadata={declared_count}, actual={len(pairs)}"
        )
    top_pairs = set(ranked_pairs[:declared_count])
    if top_pairs != set(pairs):
        raise ValueError("selected heads do not match the PCK source Top-N identity")
    raw_scores = torch.tensor(
        [scores_by_pair[pair] for pair in pairs], dtype=torch.float64
    )
    score_sum = raw_scores.sum()
    if not bool(torch.isfinite(score_sum)) or float(score_sum) <= 0.0:
        raise ValueError("PCK scores must have a positive finite sum")
    weights = (raw_scores / score_sum).to(torch.float32)
    audit = {
        "score_key": str(score_key),
        "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "ranking_step": expected_step,
        "completed_runs_at_selection": int(
            payload.get("completed_runs_at_selection", -1)
        ),
        "head_pairs": [list(pair) for pair in pairs],
        "raw_scores": raw_scores.tolist(),
        "normalized_weights": weights.tolist(),
        "score_sum": float(score_sum),
        "score_min": float(raw_scores.min()),
        "score_max": float(raw_scores.max()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
    return weights, audit


def assert_no_lora_modules(module: nn.Module, *, label: str) -> None:
    """Fail closed if a supposedly baseline module contains LoRA state."""
    suspicious = []
    for name, _ in module.named_modules():
        lowered = name.lower()
        if "lora_a" in lowered or "lora_b" in lowered or "head_lora" in lowered:
            suspicious.append(name)
    for name, _ in module.named_parameters():
        lowered = name.lower()
        if "lora_a" in lowered or "lora_b" in lowered or "head_lora" in lowered:
            suspicious.append(name)
    if suspicious:
        raise RuntimeError(
            f"{label} must be LoRA-free, but found: {sorted(set(suspicious))[:12]}"
        )


def fixed_query_head_probabilities(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    head_indices: Sequence[int],
    query_rows: torch.Tensor,
    num_heads: int,
    fixed_query: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return selected-head Q->K distributions as ``[B, H_selected, S]``.

    ``q`` and ``k`` are the RoPE-applied tensors received by Wan's attention
    kernel, with shape ``[B, S, num_heads * head_dim]``.  Multiple fixed object
    query rows are averaged.  Averaging (rather than summing) keeps every head's
    result a probability distribution and is equivalent up to a constant to the
    legacy visualization's query-row sum.
    """
    if q.ndim != 3 or k.shape != q.shape:
        raise ValueError(f"expected matching [B,S,D] Q/K, got {q.shape}/{k.shape}")
    if not head_indices:
        raise ValueError("head_indices must not be empty")
    if q.shape[-1] % int(num_heads):
        raise ValueError(
            f"Q/K width {q.shape[-1]} is not divisible by num_heads={num_heads}"
        )
    rows = torch.as_tensor(query_rows, dtype=torch.long, device=q.device).flatten()
    if rows.numel() == 0:
        raise ValueError("fixed object query row set is empty")
    if int(rows.min()) < 0 or int(rows.max()) >= q.shape[1]:
        raise IndexError(
            f"query rows [{int(rows.min())}, {int(rows.max())}] outside S={q.shape[1]}"
        )
    heads = torch.as_tensor(head_indices, dtype=torch.long, device=q.device)
    if int(heads.min()) < 0 or int(heads.max()) >= int(num_heads):
        raise IndexError(
            f"selected heads [{int(heads.min())}, {int(heads.max())}] "
            f"outside num_heads={num_heads}"
        )
    head_dim = q.shape[-1] // int(num_heads)
    qh = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    kh = k.reshape(k.shape[0], k.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
    selected_q = qh.index_select(1, heads).index_select(2, rows)
    if fixed_query is not None:
        fixed_query = torch.as_tensor(fixed_query, device=q.device)
        if fixed_query.shape != selected_q.shape:
            raise ValueError(
                "fixed GT query representation shape mismatch: "
                f"expected {selected_q.shape}, got {fixed_query.shape}"
            )
        selected_q = fixed_query.to(dtype=selected_q.dtype)
    selected_k = kh.index_select(1, heads)
    logits = torch.matmul(selected_q.float(), selected_k.float().transpose(-1, -2))
    probabilities = torch.softmax(logits / math.sqrt(float(head_dim)), dim=-1)
    return probabilities.mean(dim=2)


class TopHeadQKCollector:
    """Collect differentiable fixed-query maps from configured Wan blocks."""

    def __init__(
        self,
        *,
        selected_heads_by_block: Mapping[int, Sequence[int]],
        query_rows: torch.Tensor,
        grid: tuple[int, int, int],
        expected_num_heads: int,
        fixed_query_by_block: Mapping[int, torch.Tensor] | None = None,
    ) -> None:
        self.selected_heads_by_block = {
            int(block): tuple(sorted({int(head) for head in heads}))
            for block, heads in selected_heads_by_block.items()
            if heads
        }
        self.query_rows = torch.as_tensor(query_rows, dtype=torch.long).flatten()
        self.grid = tuple(map(int, grid))
        self.expected_num_heads = int(expected_num_heads)
        self.fixed_query_by_block = {
            int(block): query
            for block, query in (fixed_query_by_block or {}).items()
        }
        self._head_probabilities: list[torch.Tensor] = []
        self._query_representations: list[torch.Tensor] = []
        self._captured_heads = 0

        if math.prod(self.grid) <= 0:
            raise ValueError(f"invalid probe token grid: {self.grid}")
        if self.query_rows.numel() == 0:
            raise ValueError("query_rows must not be empty")
        selected_count = sum(len(heads) for heads in self.selected_heads_by_block.values())
        if selected_count <= 0:
            raise ValueError("selected_heads_by_block is empty")

    def record(self, *, block_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        heads = self.selected_heads_by_block.get(int(block_id), ())
        if not heads:
            return
        if q.shape[1] != math.prod(self.grid):
            raise RuntimeError(
                f"probe grid {self.grid} has {math.prod(self.grid)} tokens, "
                f"but block {block_id} received {q.shape[1]}"
            )
        probabilities = fixed_query_head_probabilities(
            q,
            k,
            head_indices=heads,
            query_rows=self.query_rows,
            num_heads=self.expected_num_heads,
            fixed_query=self.fixed_query_by_block.get(int(block_id)),
        )
        head_dim = q.shape[-1] // self.expected_num_heads
        rows = self.query_rows.to(device=q.device)
        head_tensor = torch.as_tensor(heads, dtype=torch.long, device=q.device)
        qh = q.reshape(
            q.shape[0],
            q.shape[1],
            self.expected_num_heads,
            head_dim,
        ).permute(0, 2, 1, 3)
        self._query_representations.append(
            qh.index_select(1, head_tensor).index_select(2, rows)
        )
        self._head_probabilities.append(probabilities)
        self._captured_heads += len(heads)

    def finalize(self) -> torch.Tensor:
        """Return a normalized aggregate heatmap with shape ``[B,F,H,W]``."""
        probabilities = self.finalize_head_probabilities().mean(dim=1)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return probabilities.reshape(probabilities.shape[0], *self.grid)

    def finalize_head_probabilities(self) -> torch.Tensor:
        """Return every captured physical head as ``[B,H_selected,S]``.

        This form is useful when the Q/K map must be an explicit activation-
        checkpoint output.  The caller can concatenate blocks and give every
        physical layer-head equal weight.
        """
        expected = sum(len(heads) for heads in self.selected_heads_by_block.values())
        if self._captured_heads != expected:
            raise RuntimeError(
                f"captured {self._captured_heads} selected heads, expected {expected}"
            )
        if not self._head_probabilities:
            raise RuntimeError("frozen probe captured no Q/K maps")
        return torch.cat(self._head_probabilities, dim=1)

    def finalize_query_representations(self) -> torch.Tensor:
        """Return selected fixed-row Q vectors as ``[B,H,Q,D]``."""
        if not self._query_representations:
            raise RuntimeError("frozen probe captured no query representations")
        return torch.cat(self._query_representations, dim=1)


@contextmanager
def capture_wan_self_attention_qk(
    dit: nn.Module,
    collector: TopHeadQKCollector,
) -> Iterable[None]:
    """Temporarily capture Q/K while leaving Wan attention math unchanged."""
    originals: list[tuple[nn.Module, object]] = []
    try:
        for block_id in collector.selected_heads_by_block:
            try:
                attention = dit.blocks[int(block_id)].self_attn.attn
            except (AttributeError, IndexError) as exc:
                raise RuntimeError(f"Wan block {block_id} has no self-attention kernel") from exc
            original_forward = attention.forward

            def wrapped_forward(
                module,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                *,
                _block_id: int = int(block_id),
                _original=original_forward,
            ):
                collector.record(block_id=_block_id, q=q, k=k)
                return _original(q, k, v)

            originals.append((attention, original_forward))
            attention.forward = MethodType(wrapped_forward, attention)
        yield
    finally:
        for attention, original_forward in originals:
            attention.forward = original_forward


def normalize_head_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 3:
        raise ValueError(
            f"expected [B,H,S] head probabilities, got {probabilities.shape}"
        )
    normalized = probabilities.float().clamp_min(1e-12)
    return normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def aggregate_head_probabilities(
    probabilities: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    head_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    normalized = normalize_head_probabilities(probabilities)
    if normalized.shape[-1] != math.prod(grid):
        raise ValueError(
            f"head token count {normalized.shape[-1]} does not match grid={grid}"
        )
    if head_weights is None:
        aggregate = normalized.mean(dim=1)
    else:
        weights = torch.as_tensor(
            head_weights, device=normalized.device, dtype=normalized.dtype
        ).flatten()
        if weights.numel() != normalized.shape[1]:
            raise ValueError(
                f"head weight count {weights.numel()} != heads {normalized.shape[1]}"
            )
        if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
            raise ValueError("head weights must be finite and non-negative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("head weights must have a positive sum")
        weights = weights / weights.sum()
        aggregate = (normalized * weights.reshape(1, -1, 1)).sum(dim=1)
    aggregate = aggregate / aggregate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return aggregate.reshape(aggregate.shape[0], *map(int, grid))


def teacher_student_head_kl(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Return per-example, per-head ``KL(Teacher || Student)`` as ``[B,H]``."""
    if student_probabilities.shape != teacher_probabilities.shape:
        raise ValueError(
            "Student/Teacher head probability shape mismatch: "
            f"{student_probabilities.shape}/{teacher_probabilities.shape}"
        )
    student = normalize_head_probabilities(student_probabilities)
    teacher = normalize_head_probabilities(teacher_probabilities.detach())
    return (teacher * (teacher.log() - student.log())).sum(dim=-1)


def pck_weighted_teacher_student_head_kl(
    student_probabilities: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    head_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_head = teacher_student_head_kl(
        student_probabilities,
        teacher_probabilities,
    )
    weights = torch.as_tensor(
        head_weights, device=per_head.device, dtype=per_head.dtype
    ).flatten()
    if weights.numel() != per_head.shape[1]:
        raise ValueError(f"head weight count {weights.numel()} != heads {per_head.shape[1]}")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("head weights must be finite and non-negative")
    if float(weights.sum()) <= 0.0:
        raise ValueError("head weights must have a positive sum")
    weights = weights / weights.sum()
    return (per_head * weights.reshape(1, -1)).sum(dim=1).mean(), per_head


def flatten_heatmap_distribution(heatmap: torch.Tensor) -> torch.Tensor:
    if heatmap.ndim != 4:
        raise ValueError(f"expected [B,F,H,W] heatmap, got {heatmap.shape}")
    distribution = heatmap.float().flatten(1).clamp_min(1e-12)
    return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def student_teacher_heatmap_kl(
    student_heatmap: torch.Tensor,
    teacher_heatmap: torch.Tensor,
) -> torch.Tensor:
    """Compute the requested KL(student_attention || teacher_attention)."""
    student = flatten_heatmap_distribution(student_heatmap)
    teacher = flatten_heatmap_distribution(teacher_heatmap.detach())
    return (student * (student.log() - teacher.log())).sum(dim=-1).mean()


def heatmap_soft_argmax_trajectory(heatmap: torch.Tensor) -> torch.Tensor:
    """Return normalized ``(x,y)`` coordinates for every latent frame.

    Output shape is ``[B,F,2]`` and coordinates lie in ``[0,1]``.  Each frame
    is spatially renormalized so temporal attention mass cannot move its point.
    """
    if heatmap.ndim != 4:
        raise ValueError(f"expected [B,F,H,W] heatmap, got {heatmap.shape}")
    batch, frames, height, width = heatmap.shape
    spatial = heatmap.float().reshape(batch, frames, height * width).clamp_min(1e-12)
    spatial = spatial / spatial.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    ys = torch.linspace(0.0, 1.0, height, device=heatmap.device, dtype=spatial.dtype)
    xs = torch.linspace(0.0, 1.0, width, device=heatmap.device, dtype=spatial.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = (spatial * grid_x.reshape(1, 1, -1)).sum(dim=-1)
    y = (spatial * grid_y.reshape(1, 1, -1)).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def trajectory_huber_loss(
    student_heatmap: torch.Tensor,
    teacher_heatmap: torch.Tensor,
    *,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(delta) <= 0.0:
        raise ValueError("trajectory Huber delta must be positive")
    student_trajectory = heatmap_soft_argmax_trajectory(student_heatmap)
    teacher_trajectory = heatmap_soft_argmax_trajectory(teacher_heatmap.detach())
    loss = F.huber_loss(
        student_trajectory,
        teacher_trajectory,
        delta=float(delta),
        reduction="mean",
    )
    return loss, student_trajectory, teacher_trajectory


def blend_with_fixed_probe_noise(
    x0: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    noise_level: float,
) -> torch.Tensor:
    """Flow-style fixed probe corruption ``(1-sigma)x0 + sigma*epsilon``."""
    sigma = float(noise_level)
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"probe_noise_level must be in [0,1], got {sigma}")
    if epsilon.shape != x0.shape:
        raise ValueError(f"x0/epsilon shape mismatch: {x0.shape}/{epsilon.shape}")
    return (1.0 - sigma) * x0 + sigma * epsilon


def query_rows_from_mask(
    mask: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    query_latent_frame: int,
    object_index: int = 0,
) -> torch.Tensor:
    """Map a GT object mask to flattened fixed Wan query-token rows.

    Accepted layouts are ``[H,W]``, ``[T,H,W]`` and ``[O,T,H,W]``.  When a
    pixel-frame mask sequence is supplied, latent frame ``t`` maps to pixel
    frame ``min(4*t, T-1)``, matching Wan's 4n+1 video/latent convention.
    """
    mask = torch.as_tensor(mask)
    if mask.ndim == 4:
        if not 0 <= int(object_index) < mask.shape[0]:
            raise IndexError(f"object_index={object_index} outside O={mask.shape[0]}")
        mask = mask[int(object_index)]
    if mask.ndim == 3:
        pixel_frame = min(4 * int(query_latent_frame), mask.shape[0] - 1)
        mask = mask[pixel_frame]
    if mask.ndim != 2:
        raise ValueError(f"GT object mask must be [H,W], [T,H,W] or [O,T,H,W], got {mask.shape}")
    frames, height, width = map(int, grid)
    if not 0 <= int(query_latent_frame) < frames:
        raise IndexError(
            f"query_latent_frame={query_latent_frame} outside latent frames={frames}"
        )
    # Match the established full-mask query rule: a latent cell is selected
    # when any object-mask pixel intersects that cell.
    resized = F.adaptive_max_pool2d(
        mask.float().reshape(1, 1, *mask.shape),
        output_size=(height, width),
    )[0, 0] > 0.5
    y, x = resized.nonzero(as_tuple=True)
    if y.numel() == 0:
        raise ValueError("GT object mask maps to zero Wan tokens")
    spatial = y * width + x
    return spatial.to(torch.long) + int(query_latent_frame) * height * width


def query_rows_from_points(
    points_xy: torch.Tensor,
    *,
    grid: tuple[int, int, int],
    query_latent_frame: int,
    image_size: tuple[int, int] | None,
    object_index: int = 0,
) -> torch.Tensor:
    """Map GT tracking points to unique flattened Wan query-token rows."""
    points = torch.as_tensor(points_xy).float()
    if points.ndim == 4:  # [O,T,P,2]
        points = points[int(object_index)]
    if points.ndim == 3:  # [T,P,2]
        point_frame = min(4 * int(query_latent_frame), points.shape[0] - 1)
        points = points[point_frame]
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"GT points must end as [P,2], got {points.shape}")
    frames, height, width = map(int, grid)
    if not 0 <= int(query_latent_frame) < frames:
        raise IndexError(
            f"query_latent_frame={query_latent_frame} outside latent frames={frames}"
        )
    if points.numel() == 0:
        raise ValueError("GT tracking point set is empty")
    if float(points.abs().max()) <= 1.5:
        normalized = points
    else:
        if image_size is None:
            raise ValueError("pixel-space GT points require image_size=(height,width)")
        image_height, image_width = map(float, image_size)
        normalized = points / points.new_tensor((image_width, image_height))
    x = (normalized[:, 0].clamp(0.0, 1.0 - 1e-7) * width).floor().to(torch.long)
    y = (normalized[:, 1].clamp(0.0, 1.0 - 1e-7) * height).floor().to(torch.long)
    rows = y * width + x + int(query_latent_frame) * height * width
    return torch.unique(rows, sorted=True)
