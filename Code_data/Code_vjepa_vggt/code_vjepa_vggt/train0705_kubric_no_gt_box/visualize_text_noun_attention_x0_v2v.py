#!/usr/bin/env python3
"""Capture Wan text cross-attention for object nouns and overlay it on context/x0 videos."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as base,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8


def _pop_option(argv: list[str], name: str, default: str) -> str:
    index = 1
    values: list[str] = []
    while index < len(argv):
        token = argv[index]
        if token == name:
            if index + 1 >= len(argv):
                raise SystemExit(f"{name} requires a value")
            values.append(argv[index + 1])
            del argv[index : index + 2]
            continue
        if token.startswith(f"{name}="):
            values.append(token.split("=", 1)[1])
            del argv[index]
            continue
        index += 1
    if not values:
        return default
    if len(set(values)) != 1:
        raise SystemExit(f"conflicting {name} values: {values}")
    return values[-1]


CAPTURE_SPEC = _pop_option(sys.argv, "--attention-capture-progress-indices", "auto5")
QUERY_CHUNK = int(_pop_option(sys.argv, "--attention-query-chunk", "256"))
TOP_LAYERS = int(_pop_option(sys.argv, "--attention-top-layers", "3"))
SAVE_ALL_ATTENTION_MAPS = bool(
    int(_pop_option(sys.argv, "--attention-save-all-maps", "0"))
)
SHARED_CONTEXT_FUTURE_SCALE = bool(
    int(_pop_option(sys.argv, "--attention-shared-context-future-scale", "0"))
)
BOUNDARY_COPY_ENABLED = bool(
    int(_pop_option(sys.argv, "--attention-boundary-copy-enabled", "0"))
)
BOUNDARY_COPY_SOURCE_LATENT = int(
    _pop_option(sys.argv, "--attention-boundary-copy-source-latent", "1")
)
BOUNDARY_COPY_TARGET_LATENT = int(
    _pop_option(sys.argv, "--attention-boundary-copy-target-latent", "2")
)
FFMPEG = Path(
    _pop_option(
        sys.argv,
        "--attention-ffmpeg",
        "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg",
    )
)


NOUN_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed": {
        "pillows": ("pillows",),
        "table": ("table",),
        "grabber_tools": ("grabber tools", "tools"),
        "tennis_ball": ("tennis ball", "ball"),
        "block": ("block",),
    },
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end": {
        "pillows": ("pillows",),
        "table": ("table",),
        "grabber_tools": ("grabber tools", "tools"),
        "tennis_ball": ("tennis ball", "ball"),
        "block": ("block",),
    },
    "physicIQ_057_Solid_Mechanics_0179_perspective-center_trimmed-two-balls-pass": {
        "tabletop": ("tabletop",),
        "pipes": ("pipes",),
        "tennis_ball": ("tennis ball", "ball"),
    },
}


def _capture_indices(spec: str, steps: int) -> list[int]:
    if steps <= 0:
        raise ValueError("sampling_steps must be positive")
    if spec.strip().lower() == "auto5":
        values = [0, steps // 4, steps // 2, (3 * steps) // 4, steps - 1]
    else:
        values = [int(value.strip()) for value in spec.split(",") if value.strip()]
    return sorted({max(0, min(int(value), steps - 1)) for value in values})


def _find_subsequence(sequence: list[int], pattern: list[int]) -> list[int]:
    if not pattern:
        return []
    positions: list[int] = []
    for start in range(0, len(sequence) - len(pattern) + 1):
        if sequence[start : start + len(pattern)] == pattern:
            positions.extend(range(start, start + len(pattern)))
    return positions


def _resolve_noun_tokens(pipe, prompt: str, noun_spec: dict[str, tuple[str, ...]]) -> dict[str, dict[str, Any]]:
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    valid_length = int(mask[0].sum().item())
    full_ids = [int(value) for value in ids[0, :valid_length].tolist()]
    tokenizer = pipe.tokenizer.tokenizer
    clean = pipe.tokenizer._clean(prompt) if pipe.tokenizer.clean else prompt
    details: dict[str, dict[str, Any]] = {}
    for noun, aliases in noun_spec.items():
        positions: set[int] = set()
        alias_ids: dict[str, list[int]] = {}
        for alias in aliases:
            clean_alias = pipe.tokenizer._clean(alias) if pipe.tokenizer.clean else alias
            encoded = tokenizer(clean_alias, add_special_tokens=False).input_ids
            encoded = [int(value) for value in encoded]
            alias_ids[alias] = encoded
            positions.update(_find_subsequence(full_ids, encoded))
        if not positions:
            raise RuntimeError(
                f"cannot locate noun {noun!r} aliases={aliases} in prompt tokens; prompt={clean!r}"
            )
        ordered = sorted(positions)
        details[noun] = {
            "aliases": list(aliases),
            "token_positions": ordered,
            "token_ids": [full_ids[index] for index in ordered],
            "tokens": tokenizer.convert_ids_to_tokens([full_ids[index] for index in ordered]),
            "alias_token_ids": alias_ids,
        }
    return details


class AttentionRecorder:
    def __init__(
        self,
        *,
        noun_details: dict[str, dict[str, Any]],
        capture_indices: list[int],
        query_chunk: int,
    ) -> None:
        self.noun_details = noun_details
        self.capture_indices = set(int(value) for value in capture_indices)
        self.query_chunk = int(query_chunk)
        self.active = False
        self.intervention_active = False
        self.step_index = -1
        self.grid: tuple[int, int, int] | None = None
        self.maps: dict[tuple[int, int, str], np.ndarray] = {}
        self._original_forwards: list[tuple[Any, Any]] = []
        self.intervention_calls = 0

    def install(self, dit) -> None:
        for layer_id, block in enumerate(dit.blocks):
            cross_attn = getattr(block, "cross_attn", None)
            attn = getattr(cross_attn, "attn", None)
            if attn is None:
                raise RuntimeError(f"Wan block {layer_id} has no text cross-attention module")
            original = attn.forward

            def wrapped(q, k, v, *, _original=original, _layer_id=layer_id):
                q_for_attention = self.apply_boundary_copy(q)
                output = _original(q_for_attention, k, v)
                if self.active:
                    self.capture(_layer_id, q_for_attention, k)
                return output

            attn.forward = wrapped
            self._original_forwards.append((attn, original))

    def restore(self) -> None:
        for module, original in self._original_forwards:
            module.forward = original
        self._original_forwards.clear()

    def apply_boundary_copy(self, q: torch.Tensor) -> torch.Tensor:
        if not (BOUNDARY_COPY_ENABLED and self.intervention_active):
            return q
        if self.grid is None:
            raise RuntimeError("attention grid was not configured before boundary intervention")
        frames, grid_h, grid_w = self.grid
        source = int(BOUNDARY_COPY_SOURCE_LATENT)
        target = int(BOUNDARY_COPY_TARGET_LATENT)
        if not (0 <= source < frames and 0 <= target < frames):
            raise RuntimeError(
                f"boundary latent indices source={source} target={target} exceed grid frames={frames}"
            )
        spatial = int(grid_h * grid_w)
        source_slice = slice(source * spatial, (source + 1) * spatial)
        target_slice = slice(target * spatial, (target + 1) * spatial)
        copied = q.clone()
        copied[:, target_slice] = q[:, source_slice]
        self.intervention_calls += 1
        return copied

    @torch.no_grad()
    def capture(self, layer_id: int, q: torch.Tensor, k: torch.Tensor) -> None:
        if self.grid is None:
            raise RuntimeError("attention grid was not configured before capture")
        frames, grid_h, grid_w = self.grid
        expected_queries = int(frames * grid_h * grid_w)
        if int(q.shape[0]) != 1:
            raise RuntimeError(f"expected batch=1 attention, got {list(q.shape)}")
        if int(q.shape[1]) != expected_queries:
            raise RuntimeError(
                f"video query count {q.shape[1]} does not match grid {self.grid}={expected_queries}"
            )
        num_heads = int(getattr(getattr(q, "_attention_module", None), "num_heads", 0))
        if num_heads <= 0:
            num_heads = 24
        if int(q.shape[-1]) % num_heads != 0:
            raise RuntimeError(f"attention dim {q.shape[-1]} is not divisible by heads={num_heads}")
        head_dim = int(q.shape[-1]) // num_heads
        keys = k.view(1, int(k.shape[1]), num_heads, head_dim).permute(0, 2, 1, 3).float()
        key_t = keys.transpose(-1, -2)
        noun_chunks: dict[str, list[torch.Tensor]] = defaultdict(list)
        for start in range(0, int(q.shape[1]), self.query_chunk):
            stop = min(start + self.query_chunk, int(q.shape[1]))
            queries = (
                q[:, start:stop]
                .view(1, stop - start, num_heads, head_dim)
                .permute(0, 2, 1, 3)
                .float()
            )
            scores = torch.matmul(queries, key_t) / math.sqrt(float(head_dim))
            probs = torch.softmax(scores, dim=-1)
            for noun, details in self.noun_details.items():
                token_ids = torch.as_tensor(
                    details["token_positions"], device=probs.device, dtype=torch.long
                )
                pooled = probs.index_select(-1, token_ids).mean(dim=-1).mean(dim=1)[0]
                noun_chunks[noun].append(pooled.cpu())
            del scores, probs, queries
        for noun, chunks in noun_chunks.items():
            flat = torch.cat(chunks, dim=0)
            array = flat.reshape(frames, grid_h, grid_w).numpy().astype(np.float16)
            self.maps[(int(self.step_index), int(layer_id), noun)] = array


class ModelFnRecorder:
    def __init__(
        self,
        *,
        pipe,
        attention: AttentionRecorder,
        capture_indices: list[int],
        cfg_scale: float,
        total_steps: int,
    ) -> None:
        self.pipe = pipe
        self.attention = attention
        self.capture_indices = set(int(value) for value in capture_indices)
        self.cfg_scale = float(cfg_scale)
        self.total_steps = int(total_steps)
        self.call_index = 0
        self.calls_per_step = 1 if abs(self.cfg_scale - 1.0) < 1.0e-8 else 2
        self.positive: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
        self.x0_latents: dict[int, torch.Tensor] = {}
        self.original = pipe.model_fn

    def install(self) -> None:
        self.pipe.model_fn = self

    def restore(self) -> None:
        self.pipe.model_fn = self.original

    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        step_index = self.call_index // self.calls_per_step
        phase_index = self.call_index % self.calls_per_step
        positive = phase_index == 0
        selected = step_index in self.capture_indices and step_index < self.total_steps
        latents = kwargs.get("latents")
        dit = kwargs.get("dit")
        if positive and (selected or BOUNDARY_COPY_ENABLED):
            patch = tuple(int(value) for value in getattr(dit, "patch_size", (1, 2, 2)))
            self.attention.grid = (
                int(latents.shape[2]) // patch[0],
                int(latents.shape[3]) // patch[1],
                int(latents.shape[4]) // patch[2],
            )
        self.attention.intervention_active = bool(positive and BOUNDARY_COPY_ENABLED)
        if selected and positive:
            self.attention.active = True
            self.attention.step_index = int(step_index)
        else:
            self.attention.active = False
        output = self.original(*args, **kwargs)
        self.attention.active = False
        self.attention.intervention_active = False
        clean_prefix = kwargs.get("clean_prefix_latents")
        clean_prefix_cpu = None if clean_prefix is None else clean_prefix.detach().cpu()
        if selected and positive:
            self.positive[int(step_index)] = (
                output.detach().cpu(),
                latents.detach().cpu(),
                clean_prefix_cpu,
            )
        if selected and (self.calls_per_step == 1 or not positive):
            if self.calls_per_step == 1:
                positive_output = output.detach().cpu()
                latent_cpu = latents.detach().cpu()
                prefix_cpu = clean_prefix_cpu
                combined = positive_output
            else:
                positive_output, latent_cpu, prefix_cpu = self.positive[int(step_index)]
                negative_output = output.detach().cpu()
                combined = negative_output + self.cfg_scale * (positive_output - negative_output)
            sigma = float(self.pipe.scheduler.sigmas[int(step_index)].item())
            x0 = latent_cpu.float() - sigma * combined.float()
            if prefix_cpu is not None:
                prefix_len = int(prefix_cpu.shape[2])
                x0[:, :, :prefix_len] = prefix_cpu.float()
            self.x0_latents[int(step_index)] = x0.to(dtype=torch.float16)
        self.call_index += 1
        return output


def _video_tensor_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"expected [C,T,H,W] video tensor, got {list(video.shape)}")
    array = (
        ((video.detach().float().clamp(-1, 1) + 1.0) * 127.5)
        .byte()
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in array]


def _decode_x0(pipe, x0_latents: dict[int, torch.Tensor]) -> dict[int, list[np.ndarray]]:
    pipe.load_models_to_device(["vae"])
    vae_dtype = next(pipe.vae.model.parameters()).dtype
    decoded: dict[int, list[np.ndarray]] = {}
    for step_index, latent_cpu in sorted(x0_latents.items()):
        latent = torch.nan_to_num(latent_cpu.float()).to(device=pipe.device, dtype=vae_dtype)
        video = pipe.vae.decode(
            latent,
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        decoded[step_index] = _video_tensor_to_uint8(video)
        del latent, video
        torch.cuda.empty_cache()
    return decoded


def _context_frames(
    *,
    source_video: str,
    context_frames: int,
    sampling_mode: str,
    height: int,
    width: int,
    crop_height: int,
    crop_width: int,
) -> tuple[list[np.ndarray], list[int]]:
    frames, frame_indices = base._load_context_video_for_mode(
        video_path=Path(source_video).expanduser().resolve(),
        target_context_frames=int(context_frames),
        sampling_mode=sampling_mode,
    )
    tensor = preprocess_video_rgb_uint8(
        frames,
        (int(height), int(width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(crop_height), int(crop_width)),
    )
    return _video_tensor_to_uint8(tensor), [int(value) for value in frame_indices]


def _temporal_resize_lowres(attn: np.ndarray, output_frames: int) -> np.ndarray:
    tensor = torch.from_numpy(attn.astype(np.float32))[None, None]
    resized = F.interpolate(
        tensor,
        size=(int(output_frames), int(attn.shape[1]), int(attn.shape[2])),
        mode="trilinear",
        align_corners=True,
    )
    return resized[0, 0].numpy()


def _heat_overlay(frame: np.ndarray, heat_lowres: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((heat_lowres - low) / max(high - low, 1.0e-12), 0.0, 1.0)
    heat = cv2.resize(
        (normalized * 255.0).astype(np.uint8),
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.56, color, 0.44, 0.0)


def _label(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    label_height = 26 * len(lines) + 8
    out = cv2.copyMakeBorder(
        frame,
        label_height,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    for index, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (8, 22 + 26 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def _write_h264(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        raise ValueError(f"cannot write empty video: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".temporary.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {temporary}")
    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError(f"inconsistent frame size in {path}: {frame.shape[:2]}")
        writer.write(frame)
    writer.release()
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(temporary),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    temporary.unlink()


def _score_layers(
    recorder: AttentionRecorder,
    *,
    nouns: list[str],
    capture_indices: list[int],
    total_frames: int,
    context_frames: int,
    layer_count: int,
) -> tuple[dict[str, list[int]], list[dict[str, object]]]:
    rankings: dict[str, list[int]] = {}
    rows: list[dict[str, object]] = []
    for noun in nouns:
        scores: list[tuple[float, int]] = []
        for layer_id in range(layer_count):
            maps = [
                recorder.maps[(step, layer_id, noun)].astype(np.float32)
                for step in capture_indices
            ]
            averaged = np.mean(maps, axis=0)
            aligned = _temporal_resize_lowres(averaged, total_frames)[:context_frames]
            flat = aligned.reshape(-1)
            top_count = max(1, int(round(flat.size * 0.01)))
            score = float(np.partition(flat, -top_count)[-top_count:].mean())
            mean = float(flat.mean())
            peak = float(flat.max())
            scores.append((score, layer_id))
            rows.append(
                {
                    "noun": noun,
                    "layer_id": layer_id,
                    "context_top1pct_mean": score,
                    "context_mean": mean,
                    "context_peak": peak,
                    "peak_to_mean": peak / max(mean, 1.0e-12),
                }
            )
        rankings[noun] = [layer for _, layer in sorted(scores, reverse=True)[:TOP_LAYERS]]
    return rankings, rows


def _normalized_spatial_map(array: np.ndarray) -> np.ndarray:
    flat = np.maximum(array.astype(np.float64).reshape(-1), 0.0)
    total = float(flat.sum())
    if total <= 0.0:
        return np.full_like(flat, 1.0 / float(flat.size))
    return flat / total


def _boundary_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p = _normalized_spatial_map(source)
    q = _normalized_spatial_map(target)
    midpoint = 0.5 * (p + q)
    eps = 1.0e-12
    js = 0.5 * float(np.sum(p * np.log((p + eps) / (midpoint + eps))))
    js += 0.5 * float(np.sum(q * np.log((q + eps) / (midpoint + eps))))
    cosine = float(np.dot(p, q) / max(np.linalg.norm(p) * np.linalg.norm(q), eps))
    height, width = source.shape
    yy, xx = np.mgrid[0:height, 0:width]
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    source_x = float(np.sum(p * xx.reshape(-1)) / x_scale)
    source_y = float(np.sum(p * yy.reshape(-1)) / y_scale)
    target_x = float(np.sum(q * xx.reshape(-1)) / x_scale)
    target_y = float(np.sum(q * yy.reshape(-1)) / y_scale)
    centroid_jump = math.hypot(target_x - source_x, target_y - source_y)
    source_mass = float(np.maximum(source.astype(np.float64), 0.0).sum())
    target_mass = float(np.maximum(target.astype(np.float64), 0.0).sum())
    return {
        "js_divergence": js,
        "cosine_similarity": cosine,
        "centroid_source_x": source_x,
        "centroid_source_y": source_y,
        "centroid_target_x": target_x,
        "centroid_target_y": target_y,
        "centroid_jump": centroid_jump,
        "source_attention_mass": source_mass,
        "target_attention_mass": target_mass,
        "target_to_source_mass_ratio": target_mass / max(source_mass, eps),
    }


def _render_outputs(
    *,
    output_dir: Path,
    case_stem: str,
    prompt: str,
    context_frames_bgr: list[np.ndarray],
    context_source_indices: list[int],
    x0_videos: dict[int, list[np.ndarray]],
    recorder: AttentionRecorder,
    noun_details: dict[str, dict[str, Any]],
    capture_indices: list[int],
    total_steps: int,
    fps: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    first_video = next(iter(x0_videos.values()))
    total_frames = len(first_video)
    context_count = len(context_frames_bgr)
    layer_count = len({key[1] for key in recorder.maps})
    expected = len(capture_indices) * layer_count * len(noun_details)
    if len(recorder.maps) != expected:
        raise RuntimeError(f"captured {len(recorder.maps)} maps, expected {expected}")
    rankings, score_rows = _score_layers(
        recorder,
        nouns=list(noun_details),
        capture_indices=capture_indices,
        total_frames=total_frames,
        context_frames=context_count,
        layer_count=layer_count,
    )
    score_path = output_dir / "layer_scores.csv"
    with score_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    boundary_rows: list[dict[str, object]] = []
    boundary_summary_rows: list[dict[str, object]] = []
    source_latent = int(BOUNDARY_COPY_SOURCE_LATENT)
    target_latent = int(BOUNDARY_COPY_TARGET_LATENT)
    if not (0 <= source_latent < recorder.grid[0] and 0 <= target_latent < recorder.grid[0]):
        raise RuntimeError(
            f"boundary metric indices source={source_latent} target={target_latent} "
            f"exceed attention grid {recorder.grid}"
        )
    score_lookup = {
        (str(row["noun"]), int(row["layer_id"])): row for row in score_rows
    }
    for noun in noun_details:
        for layer_id in range(layer_count):
            group: list[dict[str, float]] = []
            for step in capture_indices:
                raw = recorder.maps[(step, layer_id, noun)].astype(np.float32)
                metrics = _boundary_metrics(raw[source_latent], raw[target_latent])
                group.append(metrics)
                boundary_rows.append(
                    {
                        "noun": noun,
                        "layer_id": layer_id,
                        "progress_index": step,
                        "remaining_steps": total_steps - step,
                        "source_latent_time_index": source_latent,
                        "target_latent_time_index": target_latent,
                        **metrics,
                    }
                )
            layer_score = score_lookup[(noun, layer_id)]
            boundary_summary_rows.append(
                {
                    "noun": noun,
                    "layer_id": layer_id,
                    "context_top1pct_mean": layer_score["context_top1pct_mean"],
                    "context_mean": layer_score["context_mean"],
                    "context_peak": layer_score["context_peak"],
                    "js_mean": float(np.mean([item["js_divergence"] for item in group])),
                    "js_max": float(np.max([item["js_divergence"] for item in group])),
                    "cosine_mean": float(np.mean([item["cosine_similarity"] for item in group])),
                    "cosine_min": float(np.min([item["cosine_similarity"] for item in group])),
                    "centroid_jump_mean": float(np.mean([item["centroid_jump"] for item in group])),
                    "centroid_jump_max": float(np.max([item["centroid_jump"] for item in group])),
                    "mass_ratio_mean": float(np.mean([item["target_to_source_mass_ratio"] for item in group])),
                    "mass_ratio_max": float(np.max([item["target_to_source_mass_ratio"] for item in group])),
                }
            )
    boundary_path = output_dir / "boundary_attention_metrics_per_step.csv"
    with boundary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boundary_rows[0]))
        writer.writeheader()
        writer.writerows(boundary_rows)
    boundary_summary_path = output_dir / "boundary_attention_metrics_summary.csv"
    with boundary_summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(boundary_summary_rows[0]))
        writer.writeheader()
        writer.writerows(boundary_summary_rows)

    x0_paths: dict[str, str] = {}
    for step_index, frames in sorted(x0_videos.items()):
        remaining = total_steps - int(step_index)
        labeled = [
            _label(frame, [f"predicted x0 | denoise remaining {remaining}", f"output frame {idx:02d}"])
            for idx, frame in enumerate(frames)
        ]
        path = output_dir / "predicted_x0" / f"pred_x0_remaining_{remaining:02d}_h264.mp4"
        _write_h264(path, labeled, fps)
        x0_paths[str(step_index)] = str(path)

    selected_raw: dict[str, np.ndarray] = {}
    noun_outputs: dict[str, object] = {}
    for noun, top_layers in rankings.items():
        noun_dir = output_dir / f"noun_{noun}"
        noun_dir.mkdir(parents=True, exist_ok=True)
        layer_outputs: list[dict[str, object]] = []
        context_contact_rows: list[np.ndarray] = []
        future_contact_rows: list[np.ndarray] = []
        for rank, layer_id in enumerate(top_layers, start=1):
            maps_by_step = {
                step: recorder.maps[(step, layer_id, noun)].astype(np.float32)
                for step in capture_indices
            }
            for step, array in maps_by_step.items():
                selected_raw[f"{noun}__layer_{layer_id:02d}__progress_{step:02d}"] = array.astype(np.float16)
            averaged = np.mean(list(maps_by_step.values()), axis=0)
            context_aligned = _temporal_resize_lowres(averaged, total_frames)[:context_count]
            context_low = float(np.percentile(context_aligned, 2.0))
            context_high = float(np.percentile(context_aligned, 99.0))
            aligned_by_step = {
                step: _temporal_resize_lowres(array, total_frames)
                for step, array in maps_by_step.items()
            }
            future_values = np.concatenate(
                [array[context_count:].reshape(-1) for array in aligned_by_step.values()]
            )
            future_low = float(np.percentile(future_values, 2.0))
            future_high = float(np.percentile(future_values, 99.0))
            if SHARED_CONTEXT_FUTURE_SCALE:
                shared_values = np.concatenate(
                    [context_aligned.reshape(-1), future_values]
                )
                context_low = future_low = float(np.percentile(shared_values, 2.0))
                context_high = future_high = float(np.percentile(shared_values, 99.0))
            context_overlay = []
            for frame_index, (frame, heat) in enumerate(zip(context_frames_bgr, context_aligned)):
                over = _heat_overlay(frame, heat, context_low, context_high)
                over = _label(
                    over,
                    [
                        f"noun={noun} | context top{rank} layer={layer_id}",
                        f"context frame {frame_index:02d} | source frame {context_source_indices[frame_index]}",
                    ],
                )
                context_overlay.append(over)
            context_path = noun_dir / f"context_rank{rank}_layer{layer_id:02d}_h264.mp4"
            _write_h264(context_path, context_overlay, fps)

            future_composite: list[np.ndarray] = []
            for frame_index in range(context_count, total_frames):
                rows = []
                for step in capture_indices:
                    remaining = total_steps - int(step)
                    over = _heat_overlay(
                        x0_videos[step][frame_index],
                        aligned_by_step[step][frame_index],
                        future_low,
                        future_high,
                    )
                    over = cv2.resize(over, (448, 256), interpolation=cv2.INTER_AREA)
                    over = _label(
                        over,
                        [
                            f"noun={noun} layer={layer_id} remaining={remaining}",
                            f"future output frame {frame_index:02d}",
                        ],
                    )
                    rows.append(over)
                future_composite.append(np.concatenate(rows, axis=0))
            future_path = noun_dir / f"future_x0_rank{rank}_layer{layer_id:02d}_h264.mp4"
            _write_h264(future_path, future_composite, fps)

            boundary_contact_rows = []
            for step in capture_indices:
                remaining = total_steps - int(step)
                source_heat = aligned_by_step[step][context_count - 1]
                target_heat = aligned_by_step[step][context_count]
                source_overlay = _heat_overlay(
                    context_frames_bgr[-1], source_heat, context_low, context_high
                )
                target_overlay = _heat_overlay(
                    x0_videos[step][context_count], target_heat, future_low, future_high
                )
                source_overlay = cv2.resize(source_overlay, (448, 256), interpolation=cv2.INTER_AREA)
                target_overlay = cv2.resize(target_overlay, (448, 256), interpolation=cv2.INTER_AREA)
                source_overlay = _label(
                    source_overlay,
                    [f"noun={noun} layer={layer_id} remaining={remaining}", "context frame 07"],
                )
                target_overlay = _label(
                    target_overlay,
                    [f"noun={noun} layer={layer_id} remaining={remaining}", "future frame 08"],
                )
                boundary_contact_rows.append(np.concatenate([source_overlay, target_overlay], axis=1))
            boundary_contact = noun_dir / f"boundary_rank{rank}_layer{layer_id:02d}_shared_scale.jpg"
            cv2.imwrite(
                str(boundary_contact),
                np.concatenate(boundary_contact_rows, axis=0),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            context_indices = [0, min(3, context_count - 1), context_count - 1]
            context_contact_rows.append(
                np.concatenate(
                    [cv2.resize(context_overlay[index], (448, 256), interpolation=cv2.INTER_AREA) for index in context_indices],
                    axis=1,
                )
            )
            final_step = capture_indices[-1]
            final_aligned = aligned_by_step[final_step]
            future_indices = [
                min(total_frames - 1, value)
                for value in (context_count, 16, 24, 32, 40, 48)
            ]
            final_tiles = []
            for frame_index in future_indices:
                over = _heat_overlay(
                    x0_videos[final_step][frame_index],
                    final_aligned[frame_index],
                    future_low,
                    future_high,
                )
                over = cv2.resize(over, (448, 256), interpolation=cv2.INTER_AREA)
                final_tiles.append(
                    _label(over, [f"noun={noun} layer={layer_id}", f"final x0 frame {frame_index:02d}"])
                )
            future_contact_rows.append(np.concatenate(final_tiles, axis=1))
            score_row = next(
                row for row in score_rows if row["noun"] == noun and int(row["layer_id"]) == layer_id
            )
            layer_outputs.append(
                {
                    "rank": rank,
                    "layer_id": layer_id,
                    "score": score_row,
                    "context_video": str(context_path),
                    "future_x0_video": str(future_path),
                    "boundary_shared_scale_contact": str(boundary_contact),
                    "context_scale": [context_low, context_high],
                    "future_scale": [future_low, future_high],
                }
            )
        context_contact = noun_dir / "context_top3_contact.jpg"
        future_contact = noun_dir / "future_final_x0_top3_contact.jpg"
        cv2.imwrite(str(context_contact), np.concatenate(context_contact_rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(future_contact), np.concatenate(future_contact_rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        noun_outputs[noun] = {
            "tokenization": noun_details[noun],
            "top_layers": top_layers,
            "layers": layer_outputs,
            "context_contact": str(context_contact),
            "future_contact": str(future_contact),
        }

    npz_path = output_dir / "selected_top3_attention_maps_fp16.npz"
    np.savez_compressed(npz_path, **selected_raw)
    all_maps_path = None
    if SAVE_ALL_ATTENTION_MAPS:
        all_maps_path = output_dir / "all_attention_maps_fp16.npz"
        all_raw = {
            f"{noun}__layer_{layer_id:02d}__progress_{step:02d}": array.astype(np.float16)
            for (step, layer_id, noun), array in recorder.maps.items()
        }
        np.savez_compressed(all_maps_path, **all_raw)
    manifest = {
        "case": case_stem,
        "prompt": prompt,
        "attention_direction": "video_query_to_positive_text_key",
        "pooling": "mean over attention heads, subword tokens, aliases, and repeated noun occurrences",
        "top_layer_metric": "mean of top 1 percent raw attention values over aligned context frames, averaged over captured denoise steps",
        "latent_to_video_alignment": "trilinear temporal interpolation with align_corners=True from 13 latent frames to 49 video frames",
        "capture_progress_indices": capture_indices,
        "capture_remaining_steps": [total_steps - value for value in capture_indices],
        "context_frames": context_count,
        "output_frames": total_frames,
        "layer_count": layer_count,
        "top_layer_count": TOP_LAYERS,
        "query_chunk": QUERY_CHUNK,
        "shared_context_future_scale": SHARED_CONTEXT_FUTURE_SCALE,
        "boundary_metrics": {
            "source_latent_time_index": source_latent,
            "target_latent_time_index": target_latent,
            "per_step_csv": str(boundary_path),
            "summary_csv": str(boundary_summary_path),
            "js_log_base": "natural",
            "centroid_coordinates": "normalized independently to [0,1] in x and y",
        },
        "cross_attention_boundary_intervention": {
            "enabled": BOUNDARY_COPY_ENABLED,
            "cfg_branch": "positive_only",
            "source_latent_time_index": BOUNDARY_COPY_SOURCE_LATENT,
            "target_latent_time_index": BOUNDARY_COPY_TARGET_LATENT,
            "scope": "all denoise steps and all DiT text cross-attention layers",
            "operation": "replace target q rows with source q rows before full text-key attention",
            "intervention_calls": recorder.intervention_calls,
        },
        "layer_scores_csv": str(score_path),
        "predicted_x0_videos": x0_paths,
        "raw_selected_maps": str(npz_path),
        "raw_all_maps": None if all_maps_path is None else str(all_maps_path),
        "nouns": noun_outputs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "nouns": noun_outputs}


_original_run_single_case = base._run_single_case_in_process
_original_has_complete_output = base._has_complete_existing_output


def _attention_dir(output_video: Path) -> Path:
    return output_video.with_name(f"{output_video.stem}_text_noun_attention")


def _has_complete_attention_output(output_video: Path, output_json: Path) -> bool:
    if not _original_has_complete_output(output_video, output_json):
        return False
    manifest = _attention_dir(output_video) / "manifest.json"
    if not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("nouns")) and int(payload.get("top_layer_count", 0)) == TOP_LAYERS


def _run_single_case_with_attention(*args, **kwargs):
    model = kwargs["model"]
    pipe = model.pipe
    input_json_path = Path(kwargs["input_json_path"])
    case_stem = input_json_path.stem
    noun_spec = NOUN_SPECS.get(case_stem)
    if noun_spec is None:
        raise RuntimeError(f"no noun specification configured for {case_stem}")
    prompt = str(kwargs.get("input_caption", ""))
    sampling_steps = int(kwargs.get("sampling_steps", 40))
    cfg_scale = float(kwargs.get("cfg_scale", 5.0))
    capture_indices = _capture_indices(CAPTURE_SPEC, sampling_steps)
    noun_details = _resolve_noun_tokens(pipe, prompt, noun_spec)
    attention = AttentionRecorder(
        noun_details=noun_details,
        capture_indices=capture_indices,
        query_chunk=QUERY_CHUNK,
    )
    model_fn = ModelFnRecorder(
        pipe=pipe,
        attention=attention,
        capture_indices=capture_indices,
        cfg_scale=cfg_scale,
        total_steps=sampling_steps,
    )
    attention.install(pipe.dit)
    model_fn.install()
    try:
        result, logs = _original_run_single_case(*args, **kwargs)
    finally:
        model_fn.restore()
        attention.restore()
    if set(model_fn.x0_latents) != set(capture_indices):
        raise RuntimeError(
            f"predicted-x0 snapshots mismatch: got={sorted(model_fn.x0_latents)} expected={capture_indices}"
        )
    x0_videos = _decode_x0(pipe, model_fn.x0_latents)
    context_bgr, source_indices = _context_frames(
        source_video=str(kwargs["source_video"]),
        context_frames=int(kwargs["context_frames"]),
        sampling_mode=str(kwargs["sampling_mode"]),
        height=int(kwargs["height"]),
        width=int(kwargs["width"]),
        crop_height=int(kwargs["input_cover_crop_height"]),
        crop_width=int(kwargs["input_cover_crop_width"]),
    )
    attention_output = _render_outputs(
        output_dir=_attention_dir(Path(kwargs["output_video"])),
        case_stem=case_stem,
        prompt=prompt,
        context_frames_bgr=context_bgr,
        context_source_indices=source_indices,
        x0_videos=x0_videos,
        recorder=attention,
        noun_details=noun_details,
        capture_indices=capture_indices,
        total_steps=sampling_steps,
        fps=int(kwargs["fps"]),
    )
    result["text_noun_attention"] = attention_output
    logs.append(
        "[text-noun-attention] "
        f"capture_indices={capture_indices} nouns={list(noun_details)} "
        f"manifest={attention_output['manifest']}"
    )
    return result, logs


base._run_single_case_in_process = _run_single_case_with_attention
base._has_complete_existing_output = _has_complete_attention_output


if __name__ == "__main__":
    base.main()
