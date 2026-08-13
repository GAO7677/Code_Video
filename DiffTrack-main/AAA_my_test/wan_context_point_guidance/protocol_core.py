#!/usr/bin/env python3
"""Pure protocol utilities for equal-budget Wan point guidance."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def load_head_groups(
    ranking_path: Path,
    scope_path: Path,
    requested: Iterable[str] = ("top100", "bottom100", "random100"),
) -> dict[str, list[dict[str, Any]]]:
    """Load disjoint Top/Bottom/layer-matched Random head groups."""
    ranking_payload = json.loads(Path(ranking_path).read_text(encoding="utf-8"))
    entries = list(ranking_payload.get("entries") or [])
    pairs = [(int(row["block"]), int(row["head"])) for row in entries]
    expected = {(block, head) for block in range(30) for head in range(24)}
    if len(entries) != 720 or len(set(pairs)) != 720 or set(pairs) != expected:
        raise ValueError(f"ranking is not a complete 30x24 matrix: {ranking_path}")
    by_pair = {
        (int(row["block"]), int(row["head"])): dict(row) for row in entries
    }
    scope_payload = json.loads(Path(scope_path).read_text(encoding="utf-8"))
    random_pairs = scope_payload["head_scopes"][
        "random100_layer_matched_draw0"
    ]["pairs"]
    groups = {
        "top100": [dict(row) for row in entries[:100]],
        "bottom100": [dict(row) for row in entries[-100:]],
        "random100": [by_pair[(int(pair[0]), int(pair[1]))] for pair in random_pairs],
    }
    requested_names = tuple(str(value) for value in requested)
    unknown = set(requested_names) - set(groups)
    if unknown:
        raise ValueError(f"unknown head groups: {sorted(unknown)}")
    pair_sets = {
        name: {(int(row["block"]), int(row["head"])) for row in rows}
        for name, rows in groups.items()
    }
    if any(len(rows) != 100 for rows in pair_sets.values()):
        raise ValueError("each head group must contain exactly 100 unique heads")
    for left, right in (("top100", "bottom100"), ("top100", "random100"), ("bottom100", "random100")):
        if pair_sets[left] & pair_sets[right]:
            raise ValueError(f"head groups overlap: {left}, {right}")
    top_layers: dict[int, int] = {}
    random_layers: dict[int, int] = {}
    for block, _ in pair_sets["top100"]:
        top_layers[block] = top_layers.get(block, 0) + 1
    for block, _ in pair_sets["random100"]:
        random_layers[block] = random_layers.get(block, 0) + 1
    if top_layers != random_layers:
        raise ValueError("Random100 is not layer-matched to Top100")
    return {name: groups[name] for name in requested_names}


def points_to_token_rows(
    tracks_tn2: np.ndarray,
    pixel_hw: tuple[int, int],
    token_hw: tuple[int, int],
) -> torch.Tensor:
    pixel_height, pixel_width = (int(value) for value in pixel_hw)
    token_height, token_width = (int(value) for value in token_hw)
    tracks = torch.as_tensor(np.asarray(tracks_tn2), dtype=torch.float32)
    x = torch.floor(tracks[..., 0] * token_width / pixel_width).long()
    y = torch.floor(tracks[..., 1] * token_height / pixel_height).long()
    x.clamp_(0, token_width - 1)
    y.clamp_(0, token_height - 1)
    return y * token_width + x


def gaussian_targets(
    centers_n: torch.Tensor,
    token_hw: tuple[int, int],
    sigma_tokens: float,
    device: torch.device,
) -> torch.Tensor:
    height, width = (int(value) for value in token_hw)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    centers_n = centers_n.to(device=device)
    centers = torch.stack(
        (
            (centers_n % width).float(),
            torch.div(centers_n, width, rounding_mode="floor").float(),
        ),
        dim=-1,
    )
    squared_distance = (centers[:, None, :] - xy[None, :, :]).square().sum(-1)
    target = torch.exp(
        -0.5 * squared_distance / max(float(sigma_tokens) ** 2, 1.0e-6)
    )
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def valid_correspondence_count(
    visibility_tn: torch.Tensor,
    query_times: tuple[int, ...],
    key_times: tuple[int, ...],
) -> int:
    visibility = visibility_tn.bool()
    count = 0
    for query_time in query_times:
        key_visible = torch.stack(
            [visibility[key_time] for key_time in key_times], dim=0
        ).any(dim=0)
        count += int((visibility[query_time] & key_visible).sum().item())
    return count


def global_context_point_loss(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    point_rows_tn: torch.Tensor,
    visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    query_times: tuple[int, ...],
    key_times: tuple[int, ...],
    sigma_tokens: float,
) -> torch.Tensor:
    """Global-spatiotemporal CE from future point Queries to context point Keys.

    The denominator is Wan's complete T*H*W key sequence.  The target is an
    equal mixture of Gaussian neighborhoods around the same tracked point in
    every visible context latent.
    """
    token_height, token_width = (int(value) for value in token_hw)
    frame_tokens = token_height * token_width
    time_count, point_count = point_rows_tn.shape
    if q_bshd.shape[1] != time_count * frame_tokens:
        raise ValueError(
            f"token mismatch: q={tuple(q_bshd.shape)}, T/H/W="
            f"{time_count}/{token_height}/{token_width}"
        )
    if q_bshd.shape != k_bshd.shape:
        raise ValueError("Q and K must have matching selected-head shapes")
    if not query_times or not key_times:
        raise ValueError("query_times and key_times must both be non-empty")
    if set(query_times) & set(key_times):
        raise ValueError("future query times and context key times must be disjoint")
    q = q_bshd.view(
        q_bshd.shape[0], time_count, frame_tokens, q_bshd.shape[2], q_bshd.shape[3]
    )
    k = k_bshd.view(
        k_bshd.shape[0], time_count * frame_tokens, k_bshd.shape[2], k_bshd.shape[3]
    )
    rows = point_rows_tn.to(device=q.device, dtype=torch.long)
    visibility = visibility_tn.to(device=q.device, dtype=torch.bool)
    point_indices = torch.arange(point_count, device=q.device)
    scale = math.sqrt(float(q_bshd.shape[-1]))
    terms: list[torch.Tensor] = []
    for query_time in query_times:
        key_counts = torch.stack(
            [visibility[key_time] for key_time in key_times], dim=0
        ).sum(dim=0)
        valid = visibility[query_time] & (key_counts > 0)
        if not bool(valid.any()):
            continue
        query_vectors = q[:, query_time, rows[query_time], :, :].float()
        logits = torch.einsum("bnhd,bkhd->bhnk", query_vectors, k.float()) / scale
        log_normalizer = torch.logsumexp(logits, dim=-1)
        expected_logit = torch.zeros_like(log_normalizer)
        for key_time in key_times:
            key_visible = visibility[key_time]
            if not bool(key_visible.any()):
                continue
            target = gaussian_targets(
                rows[key_time], (token_height, token_width), sigma_tokens, q.device
            )
            start = key_time * frame_tokens
            stop = start + frame_tokens
            local_expectation = (
                logits[..., start:stop] * target[None, None, :, :]
            ).sum(dim=-1)
            mixture_weight = (
                key_visible.float() / key_counts.clamp_min(1).float()
            )[None, None, :]
            expected_logit = expected_logit + local_expectation * mixture_weight
        ce = log_normalizer - expected_logit
        terms.append(ce[..., point_indices[valid]].mean())
    if not terms:
        raise RuntimeError("no visible future-to-context point correspondences")
    return torch.stack(terms).mean()


def global_forward_point_loss(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    point_rows_tn: torch.Tensor,
    visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    context_query_times: tuple[int, ...],
    future_key_times: tuple[int, ...],
    sigma_tokens: float,
) -> torch.Tensor:
    """Global CE from observed-context point Queries to future point Keys.

    Each context point Query competes over Wan's complete T*H*W key sequence.
    Its target is an equal mixture of Gaussian neighborhoods around the same
    CoTracker point at every visible future latent time.
    """
    token_height, token_width = (int(value) for value in token_hw)
    frame_tokens = token_height * token_width
    time_count, point_count = point_rows_tn.shape
    if q_bshd.shape[1] != time_count * frame_tokens:
        raise ValueError(
            f"token mismatch: q={tuple(q_bshd.shape)}, T/H/W="
            f"{time_count}/{token_height}/{token_width}"
        )
    if q_bshd.shape != k_bshd.shape:
        raise ValueError("Q and K must have matching selected-head shapes")
    if not context_query_times or not future_key_times:
        raise ValueError("context query and future key times must both be non-empty")
    if set(context_query_times) & set(future_key_times):
        raise ValueError("context query and future key times must be disjoint")
    q = q_bshd.view(
        q_bshd.shape[0], time_count, frame_tokens, q_bshd.shape[2], q_bshd.shape[3]
    )
    k = k_bshd.view(
        k_bshd.shape[0], time_count * frame_tokens, k_bshd.shape[2], k_bshd.shape[3]
    )
    rows = point_rows_tn.to(device=q.device, dtype=torch.long)
    visibility = visibility_tn.to(device=q.device, dtype=torch.bool)
    point_indices = torch.arange(point_count, device=q.device)
    scale = math.sqrt(float(q_bshd.shape[-1]))
    future_counts = torch.stack(
        [visibility[future_time] for future_time in future_key_times], dim=0
    ).sum(dim=0)
    terms: list[torch.Tensor] = []
    for context_time in context_query_times:
        valid = visibility[context_time] & (future_counts > 0)
        if not bool(valid.any()):
            continue
        query_vectors = q[:, context_time, rows[context_time], :, :].float()
        logits = torch.einsum("bnhd,bkhd->bhnk", query_vectors, k.float()) / scale
        log_normalizer = torch.logsumexp(logits, dim=-1)
        expected_logit = torch.zeros_like(log_normalizer)
        for future_time in future_key_times:
            future_visible = visibility[future_time]
            if not bool(future_visible.any()):
                continue
            target = gaussian_targets(
                rows[future_time],
                (token_height, token_width),
                sigma_tokens,
                q.device,
            )
            start = future_time * frame_tokens
            stop = start + frame_tokens
            local_expectation = (
                logits[..., start:stop] * target[None, None, :, :]
            ).sum(dim=-1)
            mixture_weight = (
                future_visible.float() / future_counts.clamp_min(1).float()
            )[None, None, :]
            expected_logit = expected_logit + local_expectation * mixture_weight
        ce = log_normalizer - expected_logit
        terms.append(ce[..., point_indices[valid]].mean())
    if not terms:
        raise RuntimeError("no visible context-to-future point correspondences")
    return torch.stack(terms).mean()


def forward_attention_audit_sums(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    point_rows_tn: torch.Tensor,
    visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    context_query_times: tuple[int, ...],
    future_key_times: tuple[int, ...],
    sigma_tokens: float,
) -> dict[str, torch.Tensor]:
    """Summarize the exact global attention used by forward point guidance.

    Returned values are additive sums so captures from different blocks can be
    combined without giving sparse blocks extra weight.  Heatmaps use the same
    context-query/head pairs at every time, preserving real cross-time
    attention mass.  Localization metrics additionally require the tracked
    point to be visible in the evaluated frame.
    """
    token_height, token_width = (int(value) for value in token_hw)
    frame_tokens = token_height * token_width
    time_count, point_count = point_rows_tn.shape
    if q_bshd.shape != k_bshd.shape:
        raise ValueError("Q and K must have matching selected-head shapes")
    if q_bshd.shape[1] != time_count * frame_tokens:
        raise ValueError("Q/K token count does not match T*H*W")
    if not context_query_times or not future_key_times:
        raise ValueError("context query and future key times must both be non-empty")
    if set(context_query_times) & set(future_key_times):
        raise ValueError("context query and future key times must be disjoint")

    device = q_bshd.device
    q = q_bshd.detach().view(
        q_bshd.shape[0], time_count, frame_tokens, q_bshd.shape[2], q_bshd.shape[3]
    )
    k = k_bshd.detach().view(
        k_bshd.shape[0], time_count * frame_tokens, k_bshd.shape[2], k_bshd.shape[3]
    )
    rows = point_rows_tn.to(device=device, dtype=torch.long)
    visibility = visibility_tn.to(device=device, dtype=torch.bool)
    future_counts = torch.stack(
        [visibility[future_time] for future_time in future_key_times], dim=0
    ).sum(dim=0)
    scale = math.sqrt(float(q_bshd.shape[-1]))
    heatmap_sum = torch.zeros(
        (time_count, token_height, token_width), device=device, dtype=torch.float32
    )
    heatmap_pair_count = torch.zeros(time_count, device=device, dtype=torch.float32)
    frame_mass_sum = torch.zeros(time_count, device=device, dtype=torch.float32)
    localized_mass_sum = torch.zeros(time_count, device=device, dtype=torch.float32)
    peak_distance_sum = torch.zeros(time_count, device=device, dtype=torch.float32)
    peak_hit_sum = torch.zeros(time_count, device=device, dtype=torch.float32)
    metric_pair_count = torch.zeros(time_count, device=device, dtype=torch.float32)
    radius = max(2.0 * float(sigma_tokens), 1.0)
    yy, xx = torch.meshgrid(
        torch.arange(token_height, device=device, dtype=torch.float32),
        torch.arange(token_width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    flat_xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)

    for context_time in context_query_times:
        valid_query = visibility[context_time] & (future_counts > 0)
        if not bool(valid_query.any()):
            continue
        point_ids = torch.nonzero(valid_query, as_tuple=False).flatten()
        query_vectors = q[:, context_time, rows[context_time, point_ids], :, :].float()
        logits = torch.einsum("bnhd,bkhd->bhnk", query_vectors, k.float()) / scale
        probabilities = torch.softmax(logits, dim=-1)
        batch_count, head_count, selected_count = probabilities.shape[:3]
        all_pair_count = float(batch_count * head_count * selected_count)
        for time_index in range(time_count):
            start = time_index * frame_tokens
            stop = start + frame_tokens
            frame_probabilities = probabilities[..., start:stop]
            heatmap_sum[time_index] += frame_probabilities.sum((0, 1, 2)).view(
                token_height, token_width
            )
            heatmap_pair_count[time_index] += all_pair_count
            frame_mass_sum[time_index] += frame_probabilities.sum()

            visible_in_frame = visibility[time_index, point_ids]
            if not bool(visible_in_frame.any()):
                continue
            local_probabilities = frame_probabilities[..., visible_in_frame, :]
            local_point_ids = point_ids[visible_in_frame]
            local_rows = rows[time_index, local_point_ids]
            centers = torch.stack(
                (
                    (local_rows % token_width).float(),
                    torch.div(local_rows, token_width, rounding_mode="floor").float(),
                ),
                dim=-1,
            )
            local_mask = (
                (centers[:, None, :] - flat_xy[None, :, :]).square().sum(-1)
                <= radius**2
            )
            localized_mass_sum[time_index] += (
                local_probabilities * local_mask[None, None].float()
            ).sum()
            peaks = local_probabilities.argmax(dim=-1)
            peak_xy = torch.stack(
                (
                    (peaks % token_width).float(),
                    torch.div(peaks, token_width, rounding_mode="floor").float(),
                ),
                dim=-1,
            )
            distances = (peak_xy - centers[None, None]).square().sum(-1).sqrt()
            peak_distance_sum[time_index] += distances.sum()
            peak_hit_sum[time_index] += (distances <= radius).float().sum()
            metric_pair_count[time_index] += float(
                batch_count * head_count * int(visible_in_frame.sum())
            )

    if not bool(heatmap_pair_count.any()):
        raise RuntimeError("no visible context point Queries for attention audit")
    return {
        "heatmap_sum": heatmap_sum,
        "heatmap_pair_count": heatmap_pair_count,
        "frame_mass_sum": frame_mass_sum,
        "localized_mass_sum": localized_mass_sum,
        "peak_distance_sum": peak_distance_sum,
        "peak_hit_sum": peak_hit_sum,
        "metric_pair_count": metric_pair_count,
    }


def fixed_mutable_rms_delta(
    gradient: torch.Tensor,
    context_latent_frames: int,
    update_rms: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return -normalized gradient with an exact RMS budget on mutable latents."""
    if gradient.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W] gradient, got {gradient.shape}")
    if not 0 <= int(context_latent_frames) < int(gradient.shape[2]):
        raise ValueError("context_latent_frames must leave at least one mutable latent")
    if float(update_rms) <= 0:
        raise ValueError("update_rms must be positive")
    gradient_float = gradient.detach().float().clone()
    gradient_float[:, :, : int(context_latent_frames)] = 0
    mutable = gradient_float[:, :, int(context_latent_frames) :]
    raw_rms = mutable.square().mean().sqrt()
    if not bool(torch.isfinite(raw_rms)) or float(raw_rms) <= 1.0e-12:
        raise FloatingPointError(f"invalid mutable gradient RMS: {float(raw_rms)}")
    delta = -gradient_float / raw_rms * float(update_rms)
    actual_rms = (
        delta[:, :, int(context_latent_frames) :].square().mean().sqrt()
    )
    context_max = float(
        delta[:, :, : int(context_latent_frames)].abs().max().item()
        if context_latent_frames
        else 0.0
    )
    return delta.to(dtype=gradient.dtype), {
        "raw_mutable_gradient_rms": float(raw_rms.item()),
        "requested_mutable_update_rms": float(update_rms),
        "actual_mutable_update_rms": float(actual_rms.item()),
        "context_update_abs_max": context_max,
    }


def transform_points_stretch_to_cover_crop(
    tracks_tn2: np.ndarray,
    source_hw: tuple[int, int],
    stretched_hw: tuple[int, int],
    crop_hw: tuple[int, int],
    output_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Map legacy stretch coordinates into the exact Wan cover-crop geometry."""
    source_h, source_w = (int(value) for value in source_hw)
    stretched_h, stretched_w = (int(value) for value in stretched_hw)
    crop_h, crop_w = (int(value) for value in crop_hw)
    output_h, output_w = (
        (crop_h, crop_w)
        if output_hw is None
        else tuple(int(value) for value in output_hw)
    )
    scale = max(crop_h / float(source_h), crop_w / float(source_w))
    resized_h = max(crop_h, int(round(source_h * scale)))
    resized_w = max(crop_w, int(round(source_w * scale)))
    top = max(0, (resized_h - crop_h) // 2)
    left = max(0, (resized_w - crop_w) // 2)
    points = np.asarray(tracks_tn2, dtype=np.float32)
    source_x = (points[..., 0] + 0.5) * source_w / stretched_w - 0.5
    source_y = (points[..., 1] + 0.5) * source_h / stretched_h - 0.5
    crop_x = (source_x + 0.5) * resized_w / source_w - 0.5 - left
    crop_y = (source_y + 0.5) * resized_h / source_h - 0.5 - top
    output_x = (crop_x + 0.5) * output_w / crop_w - 0.5
    output_y = (crop_y + 0.5) * output_h / crop_h - 0.5
    transformed = np.stack((output_x, output_y), axis=-1).astype(np.float32)
    in_frame = (
        (transformed[..., 0] >= 0)
        & (transformed[..., 0] < output_w)
        & (transformed[..., 1] >= 0)
        & (transformed[..., 1] < output_h)
    )
    transformed[..., 0] = np.clip(transformed[..., 0], 0, output_w - 1)
    transformed[..., 1] = np.clip(transformed[..., 1], 0, output_h - 1)
    return transformed, in_frame, {
        "source_height": source_h,
        "source_width": source_w,
        "resized_height": resized_h,
        "resized_width": resized_w,
        "crop_top": top,
        "crop_left": left,
        "crop_height": crop_h,
        "crop_width": crop_w,
        "output_height": output_h,
        "output_width": output_w,
    }
