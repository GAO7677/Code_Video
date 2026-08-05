#!/usr/bin/env python3
"""Capture per-point object-query attention using the Head-wise PCK protocol."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from AAA_my_test.object_query_attention_capture import (
    _after_probabilities,
    ranked_heads,
)


DEFAULT_REGION_CACHE = Path(
    "/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions/"
    "case_test100_51_048_0613pybullet_sample_001460_w002"
)
LATENT_FRAMES = 13
LATENT_HEIGHT = 16
LATENT_WIDTH = 28
SPATIAL_TOKENS = LATENT_HEIGHT * LATENT_WIDTH
TARGET_HEIGHT = 512
TARGET_WIDTH = 896
QUERY_LATENT_FRAME = 1
QUERY_PIXEL_FRAME = 4
ALL_TOKEN_SIZE = 416
ALL_TOKEN_BIN = (LATENT_FRAMES * SPATIAL_TOKENS) // ALL_TOKEN_SIZE
ALL_TOKEN_QUERY_CHUNK = ALL_TOKEN_BIN * 32


def pck_query_regions() -> tuple[list[dict], np.ndarray]:
    cache_dir = Path(os.environ.get("OBJECT_QUERY_REGION_CACHE", DEFAULT_REGION_CACHE))
    metadata = json.loads((cache_dir / "regions.json").read_text(encoding="utf-8"))
    with np.load(cache_dir / "regions.npz") as arrays:
        query_points = arrays["query_points"].astype(np.float32)
        masks = arrays["masks_rhw"].astype(np.uint8)
        context_frame = arrays["context_frame_rgb"].astype(np.uint8)
    if context_frame.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
        raise RuntimeError(
            f"Expected SAM2 cache resolution {(TARGET_HEIGHT, TARGET_WIDTH)}, "
            f"got {context_frame.shape[:2]}"
        )
    if int(metadata.get("query_context_frame", -1)) != QUERY_PIXEL_FRAME:
        raise RuntimeError(
            f"Expected PCK query context frame {QUERY_PIXEL_FRAME}, "
            f"got {metadata.get('query_context_frame')}"
        )
    regions = []
    for region_index, region in enumerate(metadata["regions"]):
        if region.get("region_type") != "object":
            continue
        point_start = int(region["point_start"])
        point_end = int(region["point_end"])
        points = query_points[point_start:point_end].copy()
        token_x = np.floor(points[:, 0] * LATENT_WIDTH / TARGET_WIDTH).astype(np.int64)
        token_y = np.floor(points[:, 1] * LATENT_HEIGHT / TARGET_HEIGHT).astype(np.int64)
        token_x = np.clip(token_x, 0, LATENT_WIDTH - 1)
        token_y = np.clip(token_y, 0, LATENT_HEIGHT - 1)
        local_indices = token_y * LATENT_WIDTH + token_x
        sequence_indices = QUERY_LATENT_FRAME * SPATIAL_TOKENS + local_indices
        regions.append(
            {
                "name": str(region["region_name"]),
                "phrase": str(region.get("region_phrase") or region["region_name"]),
                "points": points,
                "mask": masks[region_index],
                "token_indices": sequence_indices.astype(np.int64),
            }
        )
    if not regions:
        raise RuntimeError(f"No object regions in {cache_dir}")
    return regions, context_frame


def _configure(owner) -> None:
    owner.object_capture_root = Path(os.environ["OBJECT_QUERY_CAPTURE_ROOT"])
    owner.object_capture_step = int(os.environ.get("QK_ATTENTION_CAPTURE_STEP", "39"))
    owner.object_capture_all_steps = (
        os.environ.get("OBJECT_QUERY_CAPTURE_ALL_STEPS", "0") == "1"
    )
    owner.object_capture_case = os.environ.get("QK_ATTENTION_CAPTURE_CASE", "case")
    owner.object_ranked_heads = (
        {} if owner.object_capture_all_steps else ranked_heads(owner.object_capture_step)
    )
    owner.object_query_regions, owner.object_query_context_frame = pck_query_regions()
    owner.object_capture_entries = {}
    owner.object_all_token_entries = {}


def _capture_all_token(owner, qh, kh, heads, group, block, scale) -> None:
    sequence = qh.shape[2]
    if sequence != ALL_TOKEN_SIZE * ALL_TOKEN_BIN:
        raise RuntimeError(
            f"Cannot pool {sequence} tokens into {ALL_TOKEN_SIZE} all-token bins"
        )
    for head in heads:
        entry = owner.object_all_token_entries.setdefault(
            (group, block, head),
            {
                "before": torch.zeros((ALL_TOKEN_SIZE, ALL_TOKEN_SIZE)),
                "after": torch.zeros((ALL_TOKEN_SIZE, ALL_TOKEN_SIZE)),
                "count": 0,
            },
        )
        for start in range(0, sequence, ALL_TOKEN_QUERY_CHUNK):
            end = min(start + ALL_TOKEN_QUERY_CHUNK, sequence)
            query_indices = torch.arange(start, end, device=qh.device)
            logits = (
                torch.matmul(
                    qh[:, head, start:end], kh[:, head].transpose(-1, -2)
                ).float() * scale
            )
            before = torch.softmax(logits, dim=-1)
            after = _after_probabilities(owner, logits, before, query_indices)
            query_bins = (end - start) // ALL_TOKEN_BIN
            pooled_shape = (
                before.shape[0], query_bins, ALL_TOKEN_BIN,
                ALL_TOKEN_SIZE, ALL_TOKEN_BIN,
            )
            pooled_before = (
                before.reshape(pooled_shape).mean(dim=(0, 2, 4)).detach().cpu()
            )
            pooled_after = (
                after.reshape(pooled_shape).mean(dim=(0, 2, 4)).detach().cpu()
            )
            bin_start = start // ALL_TOKEN_BIN
            bin_end = end // ALL_TOKEN_BIN
            entry["before"][bin_start:bin_end] += pooled_before
            entry["after"][bin_start:bin_end] += pooled_after
        entry["count"] += 1


def _capture(owner, q, k, groups, block: int) -> None:
    if not owner.object_capture_all_steps and owner.current_step != owner.object_capture_step:
        return
    capture_step = int(owner.current_step)
    group = (owner.group or "").split("_step_", 1)[0]
    selected = owner.object_ranked_heads.get(group, {})
    active_heads = sorted(set(groups.get(owner.group or "", ())))
    heads = (
        active_heads
        if owner.object_capture_all_steps
        else [head for head in active_heads if (block, head) in selected]
    )
    if not heads:
        return
    batch, sequence, channels = q.shape
    if sequence != LATENT_FRAMES * SPATIAL_TOKENS:
        raise RuntimeError(f"Expected 5824 tokens, got {sequence}")
    num_heads = channels // 128
    head_dim = channels // num_heads
    qh = q.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
    kh = k.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(head_dim)
    if owner.object_capture_all_steps:
        selected_qh = qh[:, heads]
        selected_kh = kh[:, heads]
        for region in owner.object_query_regions:
            indices = torch.as_tensor(
                region["token_indices"], device=q.device, dtype=torch.long
            )
            logits = (
                torch.matmul(
                    selected_qh[:, :, indices],
                    selected_kh.transpose(-1, -2),
                )
                .float()
                .mul(scale)
            )
            before = torch.softmax(logits, dim=-1)
            before_tensors = before.mean(dim=0).reshape(
                len(heads),
                len(indices),
                LATENT_FRAMES,
                LATENT_HEIGHT,
                LATENT_WIDTH,
            ).detach().cpu()
            for head_index, head in enumerate(heads):
                before_tensor = before_tensors[head_index]
                key = (group, capture_step, block, head, region["name"])
                entry = owner.object_capture_entries.setdefault(
                    key,
                    {
                        "before": torch.zeros_like(before_tensor),
                        "after": torch.zeros_like(before_tensor),
                        "count": 0,
                        "pck32": float("nan"),
                        "region": region,
                    },
                )
                entry["before"] += before_tensor
                entry["after"] += before_tensor
                entry["count"] += 1
        return
    if not owner.object_capture_all_steps:
        _capture_all_token(owner, qh, kh, heads, group, block, scale)
    for head in heads:
        for region in owner.object_query_regions:
            indices = torch.as_tensor(
                region["token_indices"], device=q.device, dtype=torch.long
            )
            logits = (
                torch.matmul(qh[:, head, indices], kh[:, head].transpose(-1, -2))
                .float()
                .mul(scale)
            )
            before = torch.softmax(logits, dim=-1)
            after = _after_probabilities(owner, logits, before, region["token_indices"])
            before_tensor = before.mean(dim=0).reshape(
                len(indices), LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH
            ).detach().cpu()
            after_tensor = after.mean(dim=0).reshape(
                len(indices), LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH
            ).detach().cpu()
            key = (group, capture_step, block, head, region["name"])
            entry = owner.object_capture_entries.setdefault(
                key,
                {
                    "before": torch.zeros_like(before_tensor),
                    "after": torch.zeros_like(after_tensor),
                    "count": 0,
                    "pck32": (
                        selected[(block, head)]["pck32"]
                        if (block, head) in selected
                        else float("nan")
                    ),
                    "region": region,
                },
            )
            entry["before"] += before_tensor
            entry["after"] += after_tensor
            entry["count"] += 1


def _flush(owner, group: str | None) -> None:
    if not group:
        return
    prefix = group.split("_step_", 1)[0]
    owner.object_capture_root.mkdir(parents=True, exist_ok=True)
    written = []
    keys = [key for key in owner.object_capture_entries if key[0] == prefix]
    for key in keys:
        _group, capture_step, block, head, region_name = key
        entry = owner.object_capture_entries.pop(key)
        count = max(1, int(entry["count"]))
        before = (entry["before"] / count).numpy()
        after = (entry["after"] / count).numpy()
        region = entry["region"]
        filename = (
            f"{owner.object_capture_case}__{prefix}__{region_name}"
            f"__step{capture_step:02d}__b{block:02d}_h{head:02d}.npz"
        )
        np.savez_compressed(
            owner.object_capture_root / filename,
            before=before,
            after=after,
            delta=after - before,
            query_points=region["points"],
            query_mask=region["mask"],
            query_token_indices=region["token_indices"],
            query_context_frame=owner.object_query_context_frame,
            query_latent_frame=np.int32(QUERY_LATENT_FRAME),
            query_pixel_frame=np.int32(QUERY_PIXEL_FRAME),
            region_name=np.asarray(region_name),
            region_phrase=np.asarray(region["phrase"]),
            protocol=np.asarray("headwise_pck_sam2_context_f04"),
            pck32=np.float32(entry["pck32"]),
            block=np.int32(block),
            head=np.int32(head),
            step=np.int32(capture_step),
        )
        written.append(filename)
    all_token_root = owner.object_capture_root / "all_token"
    all_token_root.mkdir(parents=True, exist_ok=True)
    all_token_keys = [
        key for key in owner.object_all_token_entries if key[0] == prefix
    ]
    for key in all_token_keys:
        _group, block, head = key
        entry = owner.object_all_token_entries.pop(key)
        count = max(1, int(entry["count"]))
        before = (entry["before"] / count).numpy()
        after = (entry["after"] / count).numpy()
        filename = (
            f"{owner.object_capture_case}__{prefix}"
            f"__step{owner.object_capture_step:02d}__b{block:02d}_h{head:02d}.npz"
        )
        np.savez_compressed(
            all_token_root / filename,
            before=before,
            after=after,
            delta=after - before,
            block=np.int32(block),
            head=np.int32(head),
            step=np.int32(owner.object_capture_step),
            pooled_tokens=np.int32(ALL_TOKEN_SIZE),
            source_tokens=np.int32(LATENT_FRAMES * SPATIAL_TOKENS),
        )
    if written:
        manifest = owner.object_capture_root / f"{prefix}__manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "protocol": "headwise_pck_sam2_context_f04",
                    "group": prefix,
                    "files": written,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def install_qk_capture(worker) -> None:
    cls = worker.AdaptiveQKLogitNoise
    original_init = cls.__init__
    original_attention = cls._attention
    original_flush = cls.flush_capture

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _configure(self)

    def attention(self, q, k, v, original, groups, block):
        _capture(self, q, k, groups, block)
        return original_attention(self, q, k, v, original, groups, block)

    def flush(self, group):
        original_flush(self, group)
        _flush(self, group)

    cls.__init__ = init
    cls._attention = attention
    cls.flush_capture = flush
