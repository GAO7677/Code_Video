#!/usr/bin/env python3
"""Direct point-trajectory interventions on selected Wan attention rows."""

from __future__ import annotations

import math
from typing import Any

import torch

from AAA_my_test.wan_context_point_guidance.protocol_core import gaussian_targets


VALID_DIRECTIONS = ("context_to_future", "future_to_context", "bidirectional")


def point_attention_targets(
    point_rows_tn: torch.Tensor,
    visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    context_times: tuple[int, ...],
    future_times: tuple[int, ...],
    direction: str,
    sigma_tokens: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unique Query rows and same-ID Gaussian Key targets.

    Duplicate point tracks can quantize to the same Query token. Their targets
    are averaged so each Query row is replaced exactly once.
    """
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"unknown direct-attention direction: {direction}")
    rows = point_rows_tn.to(device=device, dtype=torch.long)
    visibility = visibility_tn.to(device=device, dtype=torch.bool)
    time_count, point_count = rows.shape
    token_height, token_width = (int(value) for value in token_hw)
    frame_tokens = token_height * token_width
    sequence = time_count * frame_tokens

    directed_times: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    if direction in {"context_to_future", "bidirectional"}:
        directed_times.append((context_times, future_times))
    if direction in {"future_to_context", "bidirectional"}:
        directed_times.append((future_times, context_times))

    targets_by_query: dict[int, list[torch.Tensor]] = {}
    for query_times, key_times in directed_times:
        for query_time in query_times:
            for point_index in range(point_count):
                if not bool(visibility[query_time, point_index]):
                    continue
                visible_keys = [
                    key_time
                    for key_time in key_times
                    if bool(visibility[key_time, point_index])
                ]
                if not visible_keys:
                    continue
                target = torch.zeros(sequence, device=device, dtype=torch.float32)
                mixture_weight = 1.0 / float(len(visible_keys))
                for key_time in visible_keys:
                    local = gaussian_targets(
                        rows[key_time, point_index : point_index + 1],
                        (token_height, token_width),
                        sigma_tokens,
                        device,
                    )[0]
                    start = int(key_time) * frame_tokens
                    target[start : start + frame_tokens] += mixture_weight * local
                query_row = int(query_time) * frame_tokens + int(
                    rows[query_time, point_index]
                )
                targets_by_query.setdefault(query_row, []).append(target)

    if not targets_by_query:
        raise RuntimeError("no visible point rows for direct attention intervention")
    query_rows = torch.as_tensor(
        sorted(targets_by_query), device=device, dtype=torch.long
    )
    targets = torch.stack(
        [torch.stack(targets_by_query[int(row)]).mean(dim=0) for row in query_rows],
        dim=0,
    )
    targets = targets / targets.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return query_rows, targets


def intervene_attention_rows(
    q_brhd: torch.Tensor,
    k_bnhd: torch.Tensor,
    v_bnhd: torch.Tensor,
    targets_rn: torch.Tensor,
    tv_budget: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Move each attention row toward its target by a matched TV budget.

    The returned output has shape ``B,R,H,D``.  The interpolation coefficient
    is chosen per batch/head/row so ``0.5*|A'-A|_1`` equals ``tv_budget`` when
    the target is far enough, and is otherwise safely capped at the target.
    """
    if not 0.0 < float(tv_budget) <= 1.0:
        raise ValueError("attention TV budget must lie in (0, 1]")
    if q_brhd.ndim != 4 or k_bnhd.ndim != 4 or v_bnhd.ndim != 4:
        raise ValueError("Q/K/V must have B,R-or-N,H,D shapes")
    if k_bnhd.shape != v_bnhd.shape:
        raise ValueError("K and V shapes must match")
    if q_brhd.shape[0] != k_bnhd.shape[0] or q_brhd.shape[2:] != k_bnhd.shape[2:]:
        raise ValueError("Q and K batch/head/dim shapes must match")
    if targets_rn.shape != (q_brhd.shape[1], k_bnhd.shape[1]):
        raise ValueError(
            f"target shape {tuple(targets_rn.shape)} does not match "
            f"R/N={q_brhd.shape[1]}/{k_bnhd.shape[1]}"
        )

    scale = 1.0 / math.sqrt(float(q_brhd.shape[-1]))
    logits = torch.einsum("brhd,bnhd->bhrn", q_brhd, k_bnhd).float() * scale
    before = torch.softmax(logits, dim=-1)
    target = targets_rn.to(device=before.device, dtype=before.dtype)[None, None]
    max_tv = 0.5 * (target - before).abs().sum(dim=-1, keepdim=True)
    blend = (float(tv_budget) / max_tv.clamp_min(1.0e-12)).clamp(max=1.0)
    after = before + blend * (target - before)
    after = after / after.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    output = torch.einsum("bhrn,bnhd->brhd", after.to(v_bnhd.dtype), v_bnhd)
    before_output = torch.einsum(
        "bhrn,bnhd->brhd", before.to(v_bnhd.dtype), v_bnhd
    )
    actual_tv = 0.5 * (after - before).abs().sum(dim=-1)
    target_ce_before = -(target * before.clamp_min(1.0e-12).log()).sum(dim=-1)
    target_ce_after = -(target * after.clamp_min(1.0e-12).log()).sum(dim=-1)
    return output, {
        "before": before,
        "after": after,
        "actual_tv": actual_tv,
        "target_ce_before": target_ce_before,
        "target_ce_after": target_ce_after,
        "av_delta_rms": (output.float() - before_output.float())
        .square()
        .mean(dim=-1)
        .sqrt(),
        "blend": blend.squeeze(-1),
    }


def tensor_mean(value: torch.Tensor) -> float:
    return float(value.detach().float().mean().cpu())


def serializable_intervention_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        name: tensor_mean(value)
        for name, value in metrics.items()
        if isinstance(value, torch.Tensor)
        and name not in {"before", "after"}
    }
