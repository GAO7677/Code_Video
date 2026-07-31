#!/usr/bin/env python3
"""Train object/self-attention LoRA with xSSC slot-track de-duplication.

Only the xSSC object-token preparation is changed from
train_xssc_object_self_attn_lora.py:

    xSSC slots [B,T,S,512]
    -> compute slot-track similarity
    -> cluster duplicate slots
    -> merge or mask duplicate slots
    -> project to Wan object tokens

The Wan/xSSC construction, datasets, optimizer, checkpointing, DDP loop, and
wandb logging are reused from the base experiment script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F

EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
if str(TRAIN_XSSC_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_XSSC_ROOT))

import train_xssc_object_self_attn_lora as base_train


DEDUP_MODES = ("none", "mask", "merge")
DEDUP_SIMILARITY_METRICS = ("mean_frame_cosine", "pooled_cosine")
_ORIGINAL_BUILD_PARSER = base_train.build_parser
_ORIGINAL_BUILD_MODEL = base_train.build_model
_ORIGINAL_LOG_STAGE_SUMMARY = base_train._log_stage_summary


@torch.no_grad()
def compute_slot_track_similarity(
    slots: torch.Tensor,
    *,
    metric: str = "mean_frame_cosine",
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return [B,S,S] cosine similarity between xSSC slot tracks."""
    if slots.ndim != 4:
        raise ValueError(f"slots must be [B,T,S,D], got {tuple(slots.shape)}")
    if metric not in DEDUP_SIMILARITY_METRICS:
        raise ValueError(
            f"metric must be one of {DEDUP_SIMILARITY_METRICS}, got {metric!r}"
        )
    slots_float = slots.detach().float()
    if metric == "pooled_cosine":
        tracks = F.normalize(slots_float.mean(dim=1), dim=-1, eps=eps)
        similarity = torch.matmul(tracks, tracks.transpose(1, 2))
    else:
        normalized = F.normalize(slots_float, dim=-1, eps=eps)
        similarity = torch.einsum("btsd,btud->bsu", normalized, normalized)
        similarity = similarity / max(1, int(slots.shape[1]))
    eye = torch.eye(int(slots.shape[2]), device=slots.device, dtype=torch.bool)
    similarity[:, eye] = 1.0
    return similarity.clamp(-1.0, 1.0)


def _connected_components_from_similarity(
    sim: torch.Tensor,
    *,
    threshold: float,
    min_keep: int,
) -> list[list[int]]:
    slot_count = int(sim.shape[0])
    sim_cpu = sim.detach().float().cpu()
    visited = [False] * slot_count
    groups: list[list[int]] = []
    for start in range(slot_count):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        group: list[int] = []
        while stack:
            current = stack.pop()
            group.append(current)
            neighbors = torch.nonzero(sim_cpu[current] >= float(threshold), as_tuple=False)
            for item in neighbors.flatten().tolist():
                index = int(item)
                if not visited[index]:
                    visited[index] = True
                    stack.append(index)
        groups.append(sorted(group))

    min_keep = max(1, min(int(min_keep), slot_count))
    while len(groups) < min_keep:
        largest_index = max(range(len(groups)), key=lambda idx: len(groups[idx]))
        largest = groups[largest_index]
        if len(largest) <= 1:
            break
        groups[largest_index] = largest[:-1]
        groups.append([largest[-1]])
    return sorted(groups, key=lambda group: group[0])


@torch.no_grad()
def deduplicate_xssc_slot_tracks(
    slots: torch.Tensor,
    *,
    mode: str,
    threshold: float,
    similarity_metric: str = "mean_frame_cosine",
    min_keep: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Merge or mask duplicate slot tracks while preserving [B,T,S,D] shape.

    ``keep_mask`` marks representative slots. Callers should apply it again
    after adding time embeddings so masked duplicate tokens do not reappear.
    """
    if mode not in DEDUP_MODES:
        raise ValueError(f"mode must be one of {DEDUP_MODES}, got {mode!r}")
    if mode == "none":
        batch, _, slot_count, _ = slots.shape
        keep_mask = torch.ones(batch, slot_count, device=slots.device, dtype=torch.bool)
        stats = {
            "enabled": 0.0,
            "threshold": float(threshold),
            "retained_slots_mean": float(slot_count),
            "duplicate_fraction_mean": 0.0,
            "groups_per_sample_mean": float(slot_count),
            "max_group_size_mean": 1.0,
            "mean_offdiag_similarity": 0.0,
            "mean_duplicate_pair_similarity": 0.0,
        }
        return slots, keep_mask, stats

    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0,1], got {threshold}")
    similarity = compute_slot_track_similarity(slots, metric=similarity_metric)
    deduped = slots.clone()
    keep_mask = torch.zeros(
        int(slots.shape[0]),
        int(slots.shape[2]),
        device=slots.device,
        dtype=torch.bool,
    )
    retained_counts: list[float] = []
    group_counts: list[float] = []
    max_group_sizes: list[float] = []
    duplicate_pair_sims: list[float] = []
    offdiag_sims: list[float] = []

    for batch_id in range(int(slots.shape[0])):
        sim_item = similarity[batch_id]
        eye = torch.eye(int(slots.shape[2]), device=slots.device, dtype=torch.bool)
        offdiag_sims.append(float(sim_item[~eye].mean().item()))
        groups = _connected_components_from_similarity(
            sim_item,
            threshold=float(threshold),
            min_keep=int(min_keep),
        )
        retained_counts.append(float(len(groups)))
        group_counts.append(float(len(groups)))
        max_group_sizes.append(float(max(len(group) for group in groups)))
        deduped[batch_id].zero_()
        for group in groups:
            rep = int(group[0])
            keep_mask[batch_id, rep] = True
            if len(group) > 1:
                indices = torch.as_tensor(group, device=slots.device, dtype=torch.long)
                if mode == "merge":
                    deduped[batch_id, :, rep, :] = slots[batch_id, :, indices, :].mean(dim=1)
                else:
                    deduped[batch_id, :, rep, :] = slots[batch_id, :, rep, :]
                pairs = []
                for i, left in enumerate(group):
                    for right in group[i + 1 :]:
                        pairs.append(float(sim_item[int(left), int(right)].item()))
                if pairs:
                    duplicate_pair_sims.extend(pairs)
            else:
                deduped[batch_id, :, rep, :] = slots[batch_id, :, rep, :]

    slot_count = float(slots.shape[2])
    retained = torch.as_tensor(retained_counts, dtype=torch.float32)
    stats = {
        "enabled": 1.0,
        "threshold": float(threshold),
        "retained_slots_mean": float(retained.mean().item()),
        "duplicate_fraction_mean": float(((slot_count - retained) / slot_count).mean().item()),
        "groups_per_sample_mean": float(sum(group_counts) / max(1, len(group_counts))),
        "max_group_size_mean": float(sum(max_group_sizes) / max(1, len(max_group_sizes))),
        "mean_offdiag_similarity": float(sum(offdiag_sims) / max(1, len(offdiag_sims))),
        "mean_duplicate_pair_similarity": (
            float(sum(duplicate_pair_sims) / len(duplicate_pair_sims))
            if duplicate_pair_sims
            else 0.0
        ),
    }
    return deduped, keep_mask, stats


def _apply_slot_track_dropout_to_available_tokens(self, tokens: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    if self.training and self.xssc_slot_track_dropout > 0.0:
        batch, _, num_slots, _ = tokens.shape
        random_keep = torch.rand(batch, num_slots, device=tokens.device) >= self.xssc_slot_track_dropout
        keep = random_keep & available.to(device=tokens.device)
        empty_rows = ~keep.any(dim=1)
        if bool(empty_rows.any()):
            for row in torch.nonzero(empty_rows, as_tuple=False).flatten().tolist():
                candidates = torch.nonzero(available[int(row)], as_tuple=False).flatten()
                if int(candidates.numel()) == 0:
                    candidates = torch.arange(num_slots, device=tokens.device)
                choice = candidates[torch.randint(int(candidates.numel()), (1,), device=tokens.device)]
                keep[int(row)] = False
                keep[int(row), int(choice.item())] = True
        keep_scale = keep.to(dtype=tokens.dtype) / (1.0 - self.xssc_slot_track_dropout)
        self._last_slot_dropout_fraction = float((available & ~keep).float().mean().item())
        self._last_retained_slots_per_sample = float(keep.float().sum(dim=1).mean().item())
        return tokens * keep_scale[:, None, :, None]

    keep = available.to(device=tokens.device, dtype=tokens.dtype)
    self._last_slot_dropout_fraction = 0.0
    self._last_retained_slots_per_sample = float(keep.float().sum(dim=1).mean().item())
    return tokens * keep[:, None, :, None]


def _build_object_context_with_dedup(self, context_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xssc_video = self._preprocess_xssc(context_video)
    boxes = self._build_xssc_boxes(xssc_video)
    slots = self._extract_xssc_slots(xssc_video, boxes)
    slots, slot_keep_mask, dedup_stats = deduplicate_xssc_slot_tracks(
        slots,
        mode=self.xssc_slot_dedup_mode,
        threshold=self.xssc_slot_dedup_similarity_threshold,
        similarity_metric=self.xssc_slot_dedup_similarity_metric,
        min_keep=self.xssc_slot_dedup_min_keep,
    )
    self._last_slot_dedup_stats = dedup_stats
    time_steps = int(slots.shape[1])
    if time_steps > self.xssc_max_time_steps:
        raise ValueError(
            f"Context length {time_steps} exceeds xssc_max_time_steps={self.xssc_max_time_steps}"
        )
    target_dtype = self.slot_norm.weight.dtype
    slots_for_projection = slots.to(device=self.slot_norm.weight.device, dtype=target_dtype)
    tokens = self.slot_projector(self.slot_norm(slots_for_projection))
    time_ids = torch.arange(time_steps, device=tokens.device)
    time_tokens = self.time_embedding(time_ids).view(1, time_steps, 1, -1)
    tokens = tokens + time_tokens.to(dtype=tokens.dtype)
    tokens = _apply_slot_track_dropout_to_available_tokens(
        self,
        tokens,
        slot_keep_mask.to(device=tokens.device),
    )
    batch, _, num_slots, hidden_dim = tokens.shape
    return tokens.reshape(batch, time_steps * num_slots, hidden_dim), slots


def _wrap_compute_object_losses(model) -> None:
    original = model._compute_object_losses

    def _compute_object_losses_with_dedup(self, pipe, inputs_shared, inputs_posi):
        total, metrics = original(pipe, inputs_shared, inputs_posi)
        stats = getattr(self, "_last_slot_dedup_stats", {})
        prefix = "train/xssc_slot_dedup_"
        for key, value in stats.items():
            metrics[f"{prefix}{key}"] = float(value)
        metrics[f"{prefix}mode_merge"] = float(self.xssc_slot_dedup_mode == "merge")
        metrics[f"{prefix}mode_mask"] = float(self.xssc_slot_dedup_mode == "mask")
        return total, metrics

    model._compute_object_losses = MethodType(_compute_object_losses_with_dedup, model)


def build_parser() -> argparse.ArgumentParser:
    parser = _ORIGINAL_BUILD_PARSER()
    group = parser.add_argument_group("xssc_slot_track_dedup")
    group.add_argument(
        "--xssc_slot_dedup_mode",
        choices=DEDUP_MODES,
        default="none",
        help="none keeps all xSSC slots; mask zeros duplicate slots; merge averages duplicate groups into the representative slot.",
    )
    group.add_argument("--xssc_slot_dedup_similarity_threshold", type=float, default=0.94)
    group.add_argument(
        "--xssc_slot_dedup_similarity_metric",
        choices=DEDUP_SIMILARITY_METRICS,
        default="mean_frame_cosine",
    )
    group.add_argument("--xssc_slot_dedup_min_keep", type=int, default=1)
    return parser


def build_model(args: argparse.Namespace, accelerator):
    model = _ORIGINAL_BUILD_MODEL(args, accelerator)
    model.xssc_slot_dedup_mode = str(args.xssc_slot_dedup_mode)
    model.xssc_slot_dedup_similarity_threshold = float(
        args.xssc_slot_dedup_similarity_threshold
    )
    model.xssc_slot_dedup_similarity_metric = str(args.xssc_slot_dedup_similarity_metric)
    model.xssc_slot_dedup_min_keep = int(args.xssc_slot_dedup_min_keep)
    model._last_slot_dedup_stats: dict[str, float] = {
        "enabled": float(model.xssc_slot_dedup_mode != "none"),
        "threshold": model.xssc_slot_dedup_similarity_threshold,
        "retained_slots_mean": float(model.xssc_num_slots),
        "duplicate_fraction_mean": 0.0,
        "groups_per_sample_mean": float(model.xssc_num_slots),
        "max_group_size_mean": 1.0,
        "mean_offdiag_similarity": 0.0,
        "mean_duplicate_pair_similarity": 0.0,
    }
    model._build_object_context = MethodType(_build_object_context_with_dedup, model)
    _wrap_compute_object_losses(model)
    return model


def _log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    _ORIGINAL_LOG_STAGE_SUMMARY(accelerator, model, args)
    if not accelerator.is_main_process:
        return
    accelerator.print(
        "\n".join(
            [
                "xSSC slot-track de-duplication:",
                f"  mode={model.xssc_slot_dedup_mode}",
                f"  similarity_metric={model.xssc_slot_dedup_similarity_metric}",
                f"  similarity_threshold={model.xssc_slot_dedup_similarity_threshold:g}",
                f"  min_keep={model.xssc_slot_dedup_min_keep}",
                "  shape contract: [B,T,11,512] -> mask/merge duplicates -> [B,88,3072]",
            ]
        )
    )


def main() -> None:
    base_train.build_parser = build_parser
    base_train.build_model = build_model
    base_train._log_stage_summary = _log_stage_summary
    base_train.main()


if __name__ == "__main__":
    main()
