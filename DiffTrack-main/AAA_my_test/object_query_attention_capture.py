#!/usr/bin/env python3
"""Capture object-query attention for PCK Top/Bottom10 heads."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


RANKING_CSV = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
SAMPLE_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/"
    "raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/"
    "F5_drop_support/sample_001460"
)
LATENT_FRAMES = 13
LATENT_HEIGHT = 16
LATENT_WIDTH = 28
SPATIAL_TOKENS = LATENT_HEIGHT * LATENT_WIDTH
TARGET_HEIGHT = 512
TARGET_WIDTH = 896


def ranked_heads(step: int) -> dict[str, dict[tuple[int, int], dict]]:
    selection_path = os.environ.get("OBJECT_QUERY_RANKING_SELECTION", "").strip()
    if selection_path:
        payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))

        def selected(group: str):
            rows = payload[f"{group}_step_{step:02d}"][:10]
            return {
                (int(row["block"]), int(row["head"])): {
                    "block": int(row["block"]),
                    "head": int(row["head"]),
                    "pck32": float(row["macro_pck32"]),
                }
                for row in rows
            }

        return {
            "top100": selected("top100"),
            "bottom100": selected("bottom100"),
        }
    rows = []
    with RANKING_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") == "objects" and int(row["step"]) == step:
                rows.append(
                    {
                        "block": int(row["block"]),
                        "head": int(row["head"]),
                        "pck32": float(row["macro_pck32"]),
                    }
                )
    rows.sort(key=lambda item: (-item["pck32"], item["block"], item["head"]))
    if len(rows) != 720:
        raise RuntimeError(f"Expected 720 heads at step {step}, got {len(rows)}")
    return {
        "top100": {(row["block"], row["head"]): row for row in rows[:10]},
        "bottom100": {
            (row["block"], row["head"]): row
            for row in sorted(
                rows[-10:], key=lambda item: (item["pck32"], item["block"], item["head"])
            )
        },
    }


def object_query_layout() -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    meta = json.loads((SAMPLE_ROOT / "meta.json").read_text(encoding="utf-8"))
    states = np.load(SAMPLE_ROOT / "states.npz", allow_pickle=True)
    names = [str(name) for name in states["object_names"]]
    object_index = names.index("drop_ball")
    radius = float(
        next(obj for obj in meta["objects"] if obj["name"] == "drop_ball")["size"]["radius"]
    )
    eye = np.asarray(meta["camera"]["eye"], dtype=np.float64)
    target = np.asarray(meta["camera"]["target"], dtype=np.float64)
    up = np.asarray(meta["camera"]["up"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    source_width, source_height = map(float, meta["resolution"])
    focal = (source_height * 0.5) / math.tan(
        math.radians(float(meta["camera"]["yfov_deg"])) * 0.5
    )
    resize_scale = max(TARGET_WIDTH / source_width, TARGET_HEIGHT / source_height)
    crop_x = (source_width * resize_scale - TARGET_WIDTH) * 0.5
    crop_y = (source_height * resize_scale - TARGET_HEIGHT) * 0.5
    masks = np.zeros((LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH), dtype=np.uint8)
    boxes = np.zeros((LATENT_FRAMES, 4), dtype=np.float32)
    indices = []
    for latent_frame in range(LATENT_FRAMES):
        source_frame = min(latent_frame * 4, states["positions"].shape[0] - 1)
        relative = states["positions"][source_frame, object_index].astype(np.float64) - eye
        depth = float(np.dot(relative, forward))
        center_x = source_width * 0.5 + focal * float(np.dot(relative, right)) / depth
        center_y = source_height * 0.5 - focal * float(np.dot(relative, camera_up)) / depth
        radius_px = focal * radius / depth
        center_x = center_x * resize_scale - crop_x
        center_y = center_y * resize_scale - crop_y
        radius_px *= resize_scale
        boxes[latent_frame] = (
            center_x - radius_px,
            center_y - radius_px,
            center_x + radius_px,
            center_y + radius_px,
        )
        yy, xx = np.mgrid[0:LATENT_HEIGHT, 0:LATENT_WIDTH]
        token_x = (xx + 0.5) * TARGET_WIDTH / LATENT_WIDTH
        token_y = (yy + 0.5) * TARGET_HEIGHT / LATENT_HEIGHT
        mask = (token_x - center_x) ** 2 + (token_y - center_y) ** 2 <= radius_px**2
        if not mask.any():
            nearest = np.argmin((token_x - center_x) ** 2 + (token_y - center_y) ** 2)
            mask.flat[int(nearest)] = True
        masks[latent_frame] = mask.astype(np.uint8)
        local = np.flatnonzero(mask.reshape(-1))
        indices.append((latent_frame * SPATIAL_TOKENS + local).astype(int).tolist())
    return indices, masks, boxes


def _after_probabilities(owner, logits: torch.Tensor, before: torch.Tensor, query_indices):
    mode = str(getattr(owner, "noise_mode", "head_output_zero"))
    if mode == "probability_mono_scale":
        exponent = 1.0 + float(getattr(owner, "attention_alpha", 0.0))
        values = before.clamp_min(1e-12).pow(exponent)
        return values / values.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if mode == "probability_zero":
        return torch.zeros_like(before)
    if mode == "probability_uniform":
        return torch.full_like(before, 1.0 / before.shape[-1])
    if mode in {
        "probability_temporal_causal",
        "probability_strict_past",
        "probability_strict_future",
        "probability_exclude_current",
        "probability_context_only",
    }:
        key_frames = torch.arange(before.shape[-1], device=before.device) // SPATIAL_TOKENS
        query_frames = torch.as_tensor(query_indices, device=before.device) // SPATIAL_TOKENS
        if mode == "probability_temporal_causal":
            allowed = key_frames.unsqueeze(0) <= query_frames.unsqueeze(1)
        elif mode == "probability_strict_past":
            allowed = key_frames.unsqueeze(0) < query_frames.unsqueeze(1)
        elif mode == "probability_strict_future":
            allowed = key_frames.unsqueeze(0) > query_frames.unsqueeze(1)
        elif mode == "probability_exclude_current":
            allowed = key_frames.unsqueeze(0) != query_frames.unsqueeze(1)
        else:
            context_frames = int(
                os.environ.get("ATTENTION_MASK_CONTEXT_LATENT_FRAMES", "2")
            )
            allowed = (key_frames < context_frames).unsqueeze(0).expand(
                len(query_frames), -1
            )
        result = torch.zeros_like(before)
        valid_rows = allowed.any(dim=-1)
        if valid_rows.any():
            result[:, valid_rows] = torch.softmax(
                logits[:, valid_rows].masked_fill(
                    ~allowed[valid_rows].unsqueeze(0), -torch.inf
                ),
                dim=-1,
            )
        return result
    return before


def _configure(owner) -> None:
    owner.object_capture_root = Path(os.environ["OBJECT_QUERY_CAPTURE_ROOT"])
    owner.object_capture_step = int(os.environ.get("QK_ATTENTION_CAPTURE_STEP", "39"))
    owner.object_capture_case = os.environ.get("QK_ATTENTION_CAPTURE_CASE", "case")
    owner.object_ranked_heads = ranked_heads(owner.object_capture_step)
    owner.object_query_indices, owner.object_query_masks, owner.object_query_boxes = (
        object_query_layout()
    )
    owner.object_capture_entries = {}


def _capture(owner, q, k, groups, block: int, after_is_before: bool = False) -> None:
    if owner.current_step != owner.object_capture_step:
        return
    group = (owner.group or "").split("_step_", 1)[0]
    selected = owner.object_ranked_heads.get(group, {})
    active_heads = sorted(set(groups.get(owner.group or "", ())))
    heads = [head for head in active_heads if (block, head) in selected]
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
    for head in heads:
        before_frames = []
        after_frames = []
        for query_indices in owner.object_query_indices:
            index = torch.as_tensor(query_indices, device=q.device, dtype=torch.long)
            logits = torch.matmul(qh[:, head, index], kh[:, head].transpose(-1, -2)).float() * scale
            before = torch.softmax(logits, dim=-1)
            after = before if after_is_before else _after_probabilities(
                owner, logits, before, query_indices
            )
            before_frames.append(
                before.mean(dim=(0, 1)).reshape(
                    LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH
                ).detach().cpu()
            )
            after_frames.append(
                after.mean(dim=(0, 1)).reshape(
                    LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH
                ).detach().cpu()
            )
        key = (group, block, head)
        before_tensor = torch.stack(before_frames)
        after_tensor = torch.stack(after_frames)
        entry = owner.object_capture_entries.setdefault(
            key,
            {
                "before": torch.zeros_like(before_tensor),
                "after": torch.zeros_like(after_tensor),
                "count": 0,
                "pck32": selected[(block, head)]["pck32"],
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
    for key in [key for key in owner.object_capture_entries if key[0] == prefix]:
        _group, block, head = key
        entry = owner.object_capture_entries.pop(key)
        before = (entry["before"] / max(1, entry["count"])).numpy()
        after = (entry["after"] / max(1, entry["count"])).numpy()
        filename = (
            f"{owner.object_capture_case}__{prefix}__step{owner.object_capture_step:02d}"
            f"__b{block:02d}_h{head:02d}.npz"
        )
        np.savez_compressed(
            owner.object_capture_root / filename,
            before=before,
            after=after,
            delta=after - before,
            query_masks=owner.object_query_masks,
            query_boxes=owner.object_query_boxes,
            pck32=np.float32(entry["pck32"]),
            block=np.int32(block),
            head=np.int32(head),
            step=np.int32(owner.object_capture_step),
        )
        written.append(filename)
    if written:
        manifest = owner.object_capture_root / f"{prefix}__manifest.json"
        manifest.write_text(
            json.dumps({"group": prefix, "files": written}, indent=2) + "\n",
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


def install_head_zero_capture(adaptive):
    original_cls = adaptive.StepAdaptiveExtremeHeadZeroer

    class ObjectQueryHeadZeroer(original_cls):
        def __init__(self, pipe, groups):
            super().__init__(pipe, groups)
            self.noise_mode = "head_output_zero"
            self.attention_alpha = 0.0
            _configure(self)
            self.object_original_forwards = []
            wanted_blocks = {
                block
                for group in self.object_ranked_heads.values()
                for block, _head in group
            }
            models = [pipe.dit]
            if getattr(pipe, "dit2", None) is not None and pipe.dit2 is not pipe.dit:
                models.append(pipe.dit2)
            for model in models:
                for block in wanted_blocks:
                    attn = model.blocks[block].self_attn.attn
                    original = attn.forward
                    self.object_original_forwards.append((attn, original))

                    def wrapped(q, k, v, *, _original=original, _block=block):
                        group = self.group or ""
                        prefix = group.split("_step_", 1)[0]
                        heads = {
                            group: [
                                head
                                for candidate_block, head in self.object_ranked_heads.get(prefix, {})
                                if candidate_block == _block
                            ]
                        }
                        _capture(self, q, k, heads, _block, after_is_before=True)
                        return _original(q, k, v)

                    attn.forward = wrapped

        def flush_object_capture(self, group):
            _flush(self, group)

        def remove(self):
            for attn, original in self.object_original_forwards:
                attn.forward = original
            super().remove()

    adaptive.StepAdaptiveExtremeHeadZeroer = ObjectQueryHeadZeroer
    adaptive.base.ExtremeHeadZeroer = ObjectQueryHeadZeroer
