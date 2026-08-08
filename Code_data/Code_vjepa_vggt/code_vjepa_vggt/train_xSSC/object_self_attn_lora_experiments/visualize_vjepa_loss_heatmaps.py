#!/usr/bin/env python3
"""Render local flow/V-JEPA loss contributions for fixed samples and timesteps.

The visualization uses the same latent noise, context restoration, V-JEPA
sampling, and timestep weighting semantics as the formal V-JEPA training path.
It intentionally runs two independent read-only model workers; no optimizer or
training state is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRAIN_XSSC_DIR = PROJECT_ROOT / "code_vjepa_vggt/train_xSSC"
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_CONFIG = SCRIPT_DIR / "configs/formal_full_sa_no_object_gpu27_vjepa_loss.json"
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/"
    "full_sa_no_object_gpu01_formal_vjepa_loss/20260805T180305Z/"
    "checkpoints/interrupted-latest"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_vjepa_loss_heatmaps"
)
OUTPUT_HEIGHT = 256
OUTPUT_WIDTH = 448
OUTPUT_FPS = 6
OUTPUT_QUALITY = 7
VJEPA_PATCH_SIZE = 16
VJEPA_TUBELET_SIZE = 2
VJEPA_INPUT_MODES = ("center_crop", "native_rect")
CASE_SELECTION_MODES = ("mixture", "pybullet_multiobject")
MODEL_CONDITIONS = ("step03463_lora", "no_step03463_lora")
DEFAULT_CENTER_RUN_TAG = "step03463_seed3463_retry2"
DEFAULT_COMPARISON_PAGE = "comparison_step03463.html"

for _path in (PROJECT_ROOT, TRAIN_XSSC_DIR, DIFFSYNTH_ROOT, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from diffsynth.utils.data import save_video

import code_vjepa_vggt.context_wan_v_newtrain as context_flow
from code_vjepa_vggt.data import pybullet0713_no_gt_box_dataset as pybullet_data
from code_vjepa_vggt.train_xSSC import visualize_training_xt_v_x0_dinov3 as vis_single
from code_vjepa_vggt.train_xSSC.object_self_attn_lora_experiments import (
    launch_from_config,
    train_xssc_object_self_attn_lora_vjepa_loss as vjepa_train,
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _safe_name(value: str) -> str:
    value = "".join(char if char.isalnum() or char in "-_." else "_" for char in value)
    return value[:120] or "sample"


def _select_cases(dataset, *, count: int, seed: int) -> list[dict[str, Any]]:
    """Sample by the configured source mixture, then uniformly inside a source."""
    rng = np.random.default_rng(int(seed))
    source_names = list(dataset.source_names)
    source_lengths = [int(value) for value in dataset.source_lengths]
    source_probabilities = np.asarray(dataset.source_probabilities, dtype=np.float64)
    offsets = np.cumsum([0, *source_lengths[:-1]], dtype=np.int64)
    cases = []
    for position in range(int(count)):
        source_id = int(rng.choice(len(source_names), p=source_probabilities))
        local_index = int(rng.integers(0, source_lengths[source_id]))
        cases.append(
            {
                "position": position,
                "global_index": int(offsets[source_id] + local_index),
                "source_id": source_id,
                "source_name": source_names[source_id],
                "local_index": local_index,
                "case_seed": int(seed + 1009 * (position + 1)),
            }
        )
    return cases


def _select_pybullet_multiobject_cases(
    dataset,
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic random PyBullet cases using metadata only."""
    source_names = list(dataset.source_names)
    if "pybullet" not in source_names:
        raise RuntimeError(f"Dataset has no PyBullet source: {source_names}")
    source_id = source_names.index("pybullet")
    source_dataset = dataset.datasets[source_id]
    records = getattr(source_dataset, "samples", None)
    if not isinstance(records, list) or not records:
        raise RuntimeError("PyBullet dataset does not expose a non-empty samples index")
    source_offset = sum(int(value) for value in dataset.source_lengths[:source_id])
    candidates: list[dict[str, Any]] = []
    for local_index, record in enumerate(records):
        manifest_path = Path(record.manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        meta = None
        if record.meta_path:
            meta_path = Path(record.meta_path)
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            meta = None
        entity_slots = pybullet_data._entity_slots_from_meta(meta, manifest)
        if len(entity_slots) < 2:
            continue
        candidates.append(
            {
                "global_index": int(source_offset + local_index),
                "source_id": int(source_id),
                "source_name": "pybullet",
                "local_index": int(local_index),
                "sample_key": str(record.key),
                "case_id": str(record.case_id),
                "family_key": str(record.family_key),
                "object_count": len(entity_slots),
                "entity_slots": entity_slots,
                "manifest_path": str(manifest_path),
                "meta_path": str(record.meta_path) if record.meta_path else None,
            }
        )
    if len(candidates) < int(count):
        raise RuntimeError(
            f"Need {count} multi-object PyBullet cases, found {len(candidates)}"
        )
    rng = np.random.default_rng(int(seed))
    selected_ids = rng.choice(len(candidates), size=int(count), replace=False).tolist()
    selected = []
    for position, candidate_id in enumerate(selected_ids):
        case = dict(candidates[int(candidate_id)])
        case.update(
            {
                "position": int(position),
                "case_seed": int(seed + 1009 * (position + 1)),
            }
        )
        selected.append(case)
    return selected


def _validate_case_plan(
    payload: dict[str, Any],
    *,
    dataset,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if payload.get("selection_mode") != "pybullet_multiobject":
        raise ValueError(f"Unexpected case-plan selection mode: {payload.get('selection_mode')}")
    if int(payload.get("seed", -1)) != int(seed):
        raise ValueError(f"Case-plan seed mismatch: {payload.get('seed')} vs {seed}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != int(count):
        raise ValueError(
            f"Case-plan count mismatch: {len(cases) if isinstance(cases, list) else None} vs {count}"
        )
    normalized = []
    for expected_position, raw_case in enumerate(cases):
        case = dict(raw_case)
        if int(case.get("position", -1)) != expected_position:
            raise ValueError(f"Invalid case-plan position at row {expected_position}")
        if case.get("source_name") != "pybullet" or int(case.get("object_count", 0)) < 2:
            raise ValueError(f"Case-plan row is not multi-object PyBullet: {case}")
        global_index = int(case["global_index"])
        if not 0 <= global_index < len(dataset):
            raise ValueError(f"Case-plan global index is out of range: {global_index}")
        normalized.append(case)
    return normalized


def _cases_from_args(dataset, args: argparse.Namespace, *, local_rank: int) -> list[dict[str, Any]]:
    if args.case_selection == "mixture":
        return _select_cases(dataset, count=int(args.num_cases), seed=int(args.seed))
    plan_path = Path(args.case_plan).expanduser().resolve() if args.case_plan else None
    if plan_path is not None and plan_path.is_file():
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        return _validate_case_plan(
            payload,
            dataset=dataset,
            count=int(args.num_cases),
            seed=int(args.seed),
        )
    cases = _select_pybullet_multiobject_cases(
        dataset,
        count=int(args.num_cases),
        seed=int(args.seed),
    )
    if plan_path is not None and local_rank == 0:
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        _json_dump(
            plan_path,
            {
                "selection_mode": "pybullet_multiobject",
                "seed": int(args.seed),
                "num_cases": int(args.num_cases),
                "minimum_object_count": 2,
                "cases": cases,
            },
        )
    return cases


def _schedule_choices(pipe, *, sigma_min: float, sigma_max: float) -> list[dict[str, Any]]:
    total = len(pipe.scheduler.timesteps)
    sigmas = pipe.scheduler.sigmas[:total].detach().float().cpu()
    timesteps = pipe.scheduler.timesteps[:total]
    raw_weights = (
        pipe.scheduler.linear_timesteps_weights[:total]
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )
    if raw_weights.numel() != total:
        raise RuntimeError(
            f"Scheduler weight count mismatch: {raw_weights.numel()} vs {total}"
        )
    gate = (sigmas >= float(sigma_min)) & (sigmas <= float(sigma_max))
    eligible = torch.where(gate)[0].tolist()
    if len(eligible) < 3:
        raise RuntimeError(f"Need at least three V-JEPA sigma-gated timesteps, got {len(eligible)}")
    normalizer = float(raw_weights[gate].mean().item())
    ordered = sorted(eligible, key=lambda index: (float(raw_weights[index]), index))
    selected = []
    for quantile, label in ((0.10, "low_weight"), (0.50, "mid_weight"), (0.90, "high_weight")):
        offset = int(round(quantile * (len(ordered) - 1)))
        index = ordered[offset]
        if selected and index == selected[-1]["scheduler_index"]:
            for candidate in ordered:
                if candidate not in {item["scheduler_index"] for item in selected}:
                    index = candidate
                    break
        selected.append(
            {
                "label": label,
                "scheduler_index": int(index),
                "timestep": float(timesteps[index].item()),
                "sigma": float(sigmas[index].item()),
                "raw_timestep_weight": float(raw_weights[index].item()),
                "normalized_timestep_weight": float(raw_weights[index].item() / max(normalizer, 1e-12)),
                "sigma_gate_normalizer": normalizer,
            }
        )
    return selected


def _fixed_vjepa_frames(
    module,
    *,
    time_steps: int,
    context_cutoff: int,
    seed: int,
) -> tuple[torch.Tensor, bool]:
    """Make mixed sampling deterministic and reuse it for every timestep."""
    sampling = module.vjepa_frame_sampling
    if sampling == "full":
        if time_steps <= 0:
            raise ValueError("V-JEPA full-video input requires at least one frame")
        indices = torch.arange(
            time_steps,
            device=module.pipe.device,
            dtype=torch.long,
        )
        # Keep all real frames. The final frame is repeated only to complete
        # the last size-2 temporal tubelet when the video length is odd.
        if indices.numel() % VJEPA_TUBELET_SIZE:
            indices = torch.cat((indices, indices[-1:]))
        return indices, False

    if sampling == "local":
        use_local = True
    elif sampling == "global":
        use_local = False
    else:
        use_local = random.Random(int(seed)).random() < module.vjepa_local_sampling_probability
    if use_local and time_steps >= module.vjepa_num_frames:
        max_start = time_steps - module.vjepa_num_frames
        desired_start = context_cutoff - module.vjepa_local_context_frames + 1
        start = max(0, min(max_start, desired_start))
        indices = torch.arange(
            start,
            start + module.vjepa_num_frames,
            device=module.pipe.device,
            dtype=torch.long,
        )
        return indices, True
    indices = torch.linspace(
        0,
        time_steps - 1,
        steps=module.vjepa_num_frames,
        device=module.pipe.device,
    ).round().to(torch.long)
    return indices, False


def _supervised_mask(
    *,
    latent_length: int,
    context_latent_indices: list[int],
    num_clean_prefix_latents: int,
    has_first_frame_latents: bool,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(latent_length, dtype=torch.bool, device=device)
    for index in context_latent_indices:
        if 0 <= int(index) < latent_length:
            mask[int(index)] = False
    if num_clean_prefix_latents > 0:
        mask[:num_clean_prefix_latents] = False
    elif has_first_frame_latents and latent_length > 0:
        mask[0] = False
    if not bool(mask.any()):
        raise RuntimeError("The sample has no supervised latent positions")
    return mask


def _project_latent_map(value: torch.Tensor, *, time_steps: int) -> torch.Tensor:
    value = value[None, None].float()
    projected = F.interpolate(
        value,
        size=(int(time_steps), OUTPUT_HEIGHT, OUTPUT_WIDTH),
        mode="trilinear",
        align_corners=False,
    )
    return projected[0, 0]


def _preprocess_vjepa(
    module,
    video: torch.Tensor,
    frame_indices: torch.Tensor,
    *,
    input_mode: str,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Apply the configured spatial transform and report its patch grid."""
    if input_mode == "center_crop":
        prepared = module._preprocess_vjepa(video, frame_indices)
    elif input_mode == "native_rect":
        if video.ndim != 5 or int(video.shape[2]) != 3:
            raise ValueError(
                f"Tiny VAE video must be [B,T,3,H,W], got {tuple(video.shape)}"
            )
        frames = video.index_select(1, frame_indices).float()
        batch, selected_frames, channels, source_height, source_width = frames.shape
        input_height = int(module.vjepa_input_size)
        input_width = int(
            round(
                (input_height * source_width / source_height) / VJEPA_PATCH_SIZE
            )
            * VJEPA_PATCH_SIZE
        )
        input_width = max(VJEPA_PATCH_SIZE, input_width)
        frames = F.interpolate(
            frames.reshape(
                batch * selected_frames,
                channels,
                source_height,
                source_width,
            ),
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        mean = frames.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = frames.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        prepared = frames.view(
            batch,
            selected_frames,
            channels,
            input_height,
            input_width,
        ).permute(0, 2, 1, 3, 4).contiguous()
    else:
        raise ValueError(f"Unsupported V-JEPA input mode: {input_mode}")

    input_height, input_width = (int(value) for value in prepared.shape[-2:])
    if input_height % VJEPA_PATCH_SIZE or input_width % VJEPA_PATCH_SIZE:
        raise RuntimeError(
            "V-JEPA input dimensions must be divisible by its patch size: "
            f"input={input_height}x{input_width}, patch={VJEPA_PATCH_SIZE}"
        )
    return (
        prepared,
        (input_height, input_width),
        (input_height // VJEPA_PATCH_SIZE, input_width // VJEPA_PATCH_SIZE),
    )


def _project_vjepa_token_map(
    token_map: torch.Tensor,
    *,
    frame_indices: torch.Tensor,
    output_time: int,
    source_height: int,
    source_width: int,
    input_mode: str,
) -> torch.Tensor:
    """Project token losses through the selected V-JEPA spatial geometry."""
    if input_mode not in VJEPA_INPUT_MODES:
        raise ValueError(f"Unsupported V-JEPA input mode: {input_mode}")
    if input_mode == "center_crop":
        resize_short = round((256.0 / 224.0) * 384)
        scale = resize_short / min(source_height, source_width)
        resized_height = max(384, round(source_height * scale))
        resized_width = max(384, round(source_width * scale))
        top = (resized_height - 384) // 2
        left = (resized_width - 384) // 2
    result = torch.zeros(
        output_time,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        dtype=torch.float32,
    )
    for token_id, frame_value in enumerate(frame_indices.detach().cpu().tolist()):
        if not 0 <= int(frame_value) < output_time:
            continue
        if input_mode == "native_rect":
            full = F.interpolate(
                token_map[token_id][None, None].float(),
                size=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        else:
            crop = F.interpolate(
                token_map[token_id][None, None].float(),
                size=(384, 384),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            canvas = torch.zeros(resized_height, resized_width, dtype=torch.float32)
            canvas[top : top + 384, left : left + 384] = crop
            full = F.interpolate(
                canvas[None, None],
                size=(OUTPUT_HEIGHT, OUTPUT_WIDTH),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        result[int(frame_value)] = full
    return result


def _normalize_density(value: torch.Tensor, scalar: torch.Tensor | float) -> torch.Tensor:
    scalar_value = float(scalar.detach().float().item() if torch.is_tensor(scalar) else scalar)
    value = value.float().clamp_min(0.0)
    mean_value = float(value.mean().item())
    if scalar_value <= 0.0 or mean_value <= 1e-20:
        return torch.zeros_like(value)
    return value * (scalar_value / mean_value)


def _vjepa_maps(
    module,
    *,
    target_raw: torch.Tensor,
    pred_raw: torch.Tensor,
    target_features: torch.Tensor,
    frame_indices: torch.Tensor,
    context_cutoff: int,
    input_mode: str,
    input_hw: tuple[int, int],
    spatial_grid: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
    pred_clipped = pred_raw.clamp(0.0, 1.0)
    pred_input, pred_input_hw, pred_spatial_grid = _preprocess_vjepa(
        module,
        pred_clipped,
        frame_indices,
        input_mode=input_mode,
    )
    if pred_input_hw != input_hw or pred_spatial_grid != spatial_grid:
        raise RuntimeError(
            "Target/prediction V-JEPA geometry mismatch: "
            f"target={input_hw}/{spatial_grid}, "
            f"pred={pred_input_hw}/{pred_spatial_grid}"
        )
    pred_features = module._encode_vjepa(pred_input)
    if pred_features.shape != target_features.shape:
        raise RuntimeError(
            f"V-JEPA feature shape mismatch: pred={tuple(pred_features.shape)}, "
            f"target={tuple(target_features.shape)}"
        )
    selected_frames = int(frame_indices.numel())
    tubelet_size = VJEPA_TUBELET_SIZE
    temporal_tokens = selected_frames // tubelet_size
    if int(pred_features.shape[1]) % temporal_tokens:
        raise RuntimeError("V-JEPA feature tokens do not divide into temporal groups")
    spatial_tokens = int(pred_features.shape[1]) // temporal_tokens
    pred_features = F.normalize(pred_features.float(), dim=-1)
    target_features = F.normalize(target_features.float(), dim=-1)
    error = (pred_features - target_features).square().sum(dim=-1)
    error = error.view(error.shape[0], temporal_tokens, spatial_tokens)
    tubes = frame_indices.view(temporal_tokens, tubelet_size)
    future_mask = (tubes > int(context_cutoff)).all(dim=1)
    if not bool(future_mask.any()):
        raise RuntimeError("V-JEPA frame selection has no future-only tubelets")
    feature_loss = error[:, future_mask].mean()
    spatial_height, spatial_width = spatial_grid
    if spatial_height * spatial_width != spatial_tokens:
        raise RuntimeError(
            "V-JEPA token count does not match its input grid: "
            f"tokens={spatial_tokens}, grid={spatial_height}x{spatial_width}"
        )
    selected_token_map = torch.zeros(
        selected_frames,
        spatial_height,
        spatial_width,
        device=error.device,
        dtype=torch.float32,
    )
    for tubelet_id in torch.where(future_mask)[0].detach().cpu().tolist():
        token_map = error[0, tubelet_id].view(spatial_height, spatial_width)
        for frame_id in range(tubelet_id * tubelet_size, tubelet_id * tubelet_size + tubelet_size):
            selected_token_map[frame_id] = token_map
    frame_map = _project_vjepa_token_map(
        selected_token_map,
        frame_indices=frame_indices,
        output_time=int(target_raw.shape[1]),
        source_height=int(target_raw.shape[-2]),
        source_width=int(target_raw.shape[-1]),
        input_mode=input_mode,
    ).to(device=pred_raw.device)
    range_map = (
        F.relu(-pred_raw).square() + F.relu(pred_raw - 1.0).square()
    ).mean(dim=2)[0]
    range_map = F.interpolate(
        range_map[None, None].float(),
        size=(int(target_raw.shape[1]), OUTPUT_HEIGHT, OUTPUT_WIDTH),
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    range_loss = (
        F.relu(-pred_raw).square().mean()
        + F.relu(pred_raw - 1.0).square().mean()
    )
    return (
        frame_map,
        range_map,
        float(feature_loss.detach().item()),
        float(range_loss.detach().item()),
        float(future_mask.float().mean().item()),
    )


def _as_rgb_frames(video: torch.Tensor) -> list[np.ndarray]:
    frames = (
        video[0]
        .detach()
        .float()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [frame for frame in frames]


def _resize_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [
        cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        for frame in frames
    ]


def _write_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(path), fps=OUTPUT_FPS, quality=OUTPUT_QUALITY)


def _vmax(maps: list[np.ndarray]) -> float:
    values = np.concatenate([item.reshape(-1) for item in maps])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return 1.0
    return max(float(np.percentile(values, 99.5)), float(values.max()) * 1e-6, 1e-12)


def _future_vjepa_frame_indices(
    frame_indices: torch.Tensor,
    *,
    context_cutoff: int,
) -> list[int]:
    if int(frame_indices.numel()) % VJEPA_TUBELET_SIZE:
        raise RuntimeError(
            "V-JEPA frame selection is not divisible by its tubelet size: "
            f"frames={int(frame_indices.numel())}, tubelet={VJEPA_TUBELET_SIZE}"
        )
    tubes = frame_indices.view(-1, VJEPA_TUBELET_SIZE)
    future_tubes = (tubes > int(context_cutoff)).all(dim=1)
    selected_tensor = tubes[future_tubes].reshape(-1)
    # The full-video path may repeat the final real frame for tubelet
    # alignment. Report each decoded frame once in visualizations/statistics.
    selected_tensor = torch.unique_consecutive(selected_tensor)
    selected = selected_tensor.detach().cpu().tolist()
    if not selected:
        raise RuntimeError("V-JEPA frame selection has no future-only feature-loss frames")
    return [int(value) for value in selected]


def _project_latent_temporal_signal(
    value: torch.Tensor,
    *,
    time_steps: int,
) -> torch.Tensor:
    projected = F.interpolate(
        value[None, None, :, None, None].float(),
        size=(int(time_steps), 1, 1),
        mode="trilinear",
        align_corners=False,
    )
    return projected[0, 0, :, 0, 0]


def _flow_frame_usage(
    supervised: torch.Tensor,
    *,
    time_steps: int,
) -> tuple[list[int], list[str]]:
    activity = _project_latent_temporal_signal(
        supervised.float(),
        time_steps=time_steps,
    ).detach().cpu().numpy()
    active = np.flatnonzero(activity > 1e-6).astype(np.int64).tolist()
    labels = []
    for value in activity:
        if value <= 1e-6:
            labels.append("flow=masked")
        elif value >= 1.0 - 1e-6:
            labels.append("flow=supervised")
        else:
            labels.append("flow=mixed")
    if not active:
        raise RuntimeError("Projected flow supervision has no active video frames")
    return [int(value) for value in active], labels


def _spatial_stats(
    density: np.ndarray,
    *,
    active_frame_indices: list[int],
    profile_size: tuple[int, int] = (24, 42),
) -> dict[str, Any]:
    """Summarize normalized spatial mass so case comparisons ignore loss scale."""
    if density.ndim != 3:
        raise ValueError(f"Expected [T,H,W] density, got {density.shape}")
    selected = [
        int(index)
        for index in active_frame_indices
        if 0 <= int(index) < int(density.shape[0])
    ]
    if not selected:
        raise ValueError("Spatial statistics require at least one active frame")
    profile = np.asarray(density[selected], dtype=np.float64).sum(axis=0)
    profile = np.where(np.isfinite(profile), np.maximum(profile, 0.0), 0.0)
    total = float(profile.sum())
    height, width = profile.shape
    if total <= 1e-20:
        normalized = np.full((height, width), 1.0 / float(height * width), dtype=np.float64)
    else:
        normalized = profile / total
    y_coords = (np.arange(height, dtype=np.float64) + 0.5) / float(height)
    x_coords = (np.arange(width, dtype=np.float64) + 0.5) / float(width)
    centroid_x = float((normalized.sum(axis=0) * x_coords).sum())
    centroid_y = float((normalized.sum(axis=1) * y_coords).sum())
    positive = normalized[normalized > 0.0]
    entropy = float(-(positive * np.log(positive)).sum() / math.log(float(height * width)))
    active_area = float((normalized > (1.0 / float(height * width))).mean())
    grid = F.adaptive_avg_pool2d(
        torch.from_numpy(normalized.astype(np.float32))[None, None],
        profile_size,
    )[0, 0].numpy().astype(np.float64)
    grid /= max(float(grid.sum()), 1e-20)
    return {
        "active_frame_indices": selected,
        "centroid_xy_normalized": [centroid_x, centroid_y],
        "normalized_entropy": entropy,
        "active_area_fraction_above_uniform": active_area,
        "profile_grid_shape": [int(profile_size[0]), int(profile_size[1])],
        "profile_grid": grid.astype(np.float32).tolist(),
    }


def _overlay_video(
    frames: list[np.ndarray],
    density: np.ndarray,
    *,
    vmax: float,
    header: str,
    frame_labels: list[str] | None = None,
) -> list[np.ndarray]:
    if len(frames) != int(density.shape[0]):
        raise ValueError(
            f"Frame/density length mismatch: frames={len(frames)}, density={density.shape}"
        )
    if frame_labels is not None and len(frame_labels) != len(frames):
        raise ValueError(
            f"Frame-label length mismatch: labels={len(frame_labels)}, frames={len(frames)}"
        )
    rendered = []
    for frame_index, (frame, values) in enumerate(zip(frames, density)):
        normalized = np.clip(values / max(vmax, 1e-12), 0.0, 1.0).astype(np.float32)
        heat_bgr = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        heat = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        base = frame.astype(np.float32)
        support = values > 0.0
        alpha = (0.12 + 0.70 * np.sqrt(normalized)) * support.astype(np.float32)
        alpha = alpha[..., None]
        blended = base * (1.0 - alpha) + heat * alpha
        canvas = np.clip(blended, 0.0, 255.0).astype(np.uint8)
        cv2.rectangle(canvas, (0, 0), (OUTPUT_WIDTH, 37), (20, 24, 28), -1)
        cv2.putText(
            canvas,
            header,
            (7, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (245, 248, 250),
            1,
            cv2.LINE_AA,
        )
        frame_label = (
            frame_labels[frame_index] if frame_labels is not None else "loss=projected"
        )
        cv2.putText(
            canvas,
            f"frame={frame_index:02d} | {frame_label}",
            (7, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (210, 221, 214),
            1,
            cv2.LINE_AA,
        )
        rendered.append(canvas)
    return rendered


def _build_case(
    *,
    model,
    dataset,
    case: dict[str, Any],
    schedule_choices: list[dict[str, Any]],
    case_dir: Path,
    vjepa_input_mode: str,
    model_condition: str,
) -> dict[str, Any]:
    case_seed = int(case["case_seed"])
    _seed_all(case_seed)
    sample = dataset[int(case["global_index"])]
    prepared = model._prepare_pipeline_sample(sample)
    inputs_shared, inputs_posi = prepared[0], prepared[1]
    input_latents = inputs_shared["input_latents"]
    pipe = model.pipe
    target_raw = model._decode_tiny_vae_raw(input_latents)
    if target_raw.ndim != 5 or int(target_raw.shape[0]) != 1:
        raise RuntimeError(f"Unexpected Tiny-VAE target shape: {tuple(target_raw.shape)}")
    target_video = target_raw.clamp(0.0, 1.0)
    output_time = int(target_raw.shape[1])
    context_cutoff = model._context_frame_cutoff(inputs_shared, output_time)
    model_frame_indices, sampling_local = _fixed_vjepa_frames(
        model,
        time_steps=output_time,
        context_cutoff=context_cutoff,
        seed=case_seed + 7001,
    )
    display_frame_indices = torch.arange(
        output_time,
        device=model_frame_indices.device,
        dtype=torch.long,
    )
    vjepa_loss_frame_indices = _future_vjepa_frame_indices(
        model_frame_indices,
        context_cutoff=context_cutoff,
    )
    vjepa_loss_frame_set = set(vjepa_loss_frame_indices)
    vjepa_frame_labels = [
        "vjepa_feature=used; range=all"
        if frame_index in vjepa_loss_frame_set
        else "vjepa_feature=unused; range=all"
        for frame_index in range(output_time)
    ]
    with torch.no_grad():
        target_input, vjepa_input_hw, vjepa_spatial_grid = _preprocess_vjepa(
            model,
            target_video,
            model_frame_indices,
            input_mode=vjepa_input_mode,
        )
        target_features = model._encode_vjepa(target_input).detach()
    target_frames = _resize_frames(_as_rgb_frames(target_video))
    variants: list[dict[str, Any]] = []
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for choice in schedule_choices:
        _seed_all(case_seed)
        generator = torch.Generator(device=input_latents.device)
        generator.manual_seed(case_seed)
        noise = torch.randn(
            tuple(input_latents.shape),
            device=input_latents.device,
            dtype=input_latents.dtype,
            generator=generator,
        )
        timestep_index = int(choice["scheduler_index"])
        timestep = pipe.scheduler.timesteps[timestep_index : timestep_index + 1].to(
            device=pipe.device,
            dtype=pipe.torch_dtype,
        )
        (
            latent_xt,
            training_target,
            context_latent_indices,
            num_clean_prefix_latents,
            _clean_prefix_latents,
        ) = vis_single.apply_training_noise(
            pipe=pipe,
            inputs_shared=inputs_shared,
            input_latents=input_latents,
            noise=noise,
            timestep=timestep,
        )
        model_inputs = dict(inputs_shared)
        model_inputs.update(inputs_posi)
        model_inputs["latents"] = latent_xt
        with torch.no_grad():
            velocity = pipe.model_fn(
                **models,
                **model_inputs,
                timestep=timestep,
            )
            pred_x0_raw = context_flow._predict_x0_from_diffsynth_flow(
                scheduler=pipe.scheduler,
                latent_xt=latent_xt,
                model_output=velocity,
                timestep=timestep,
            )
            pred_x0 = model._restore_condition_latents(
                pred_x0_raw,
                input_latents,
                model_inputs,
            )
            latent_error = (velocity.float() - training_target.float()).square().mean(dim=1)[0]
            supervised = _supervised_mask(
                latent_length=int(latent_error.shape[0]),
                context_latent_indices=context_latent_indices,
                num_clean_prefix_latents=int(num_clean_prefix_latents),
                has_first_frame_latents="first_frame_latents" in inputs_shared,
                device=latent_error.device,
            )
            latent_error = latent_error * supervised[:, None, None]
            main_mse = latent_error[supervised].mean()
            main_scalar = main_mse * float(choice["raw_timestep_weight"])
            flow_loss_frame_indices, flow_frame_labels = _flow_frame_usage(
                supervised,
                time_steps=output_time,
            )
            pred_raw = model._decode_tiny_vae_raw(pred_x0)
            feature_map, range_map, feature_loss, range_loss, future_fraction = _vjepa_maps(
                model,
                target_raw=target_raw,
                pred_raw=pred_raw,
                target_features=target_features,
                frame_indices=model_frame_indices,
                context_cutoff=context_cutoff,
                input_mode=vjepa_input_mode,
                input_hw=vjepa_input_hw,
                spatial_grid=vjepa_spatial_grid,
            )
            feature_density = _normalize_density(feature_map, feature_loss)
            range_density = _normalize_density(range_map, range_loss)
            aux_weight = float(model.vjepa_loss_weight) * float(
                choice["normalized_timestep_weight"]
            )
            feature_weighted_density = aux_weight * feature_density
            range_weighted_density = (
                aux_weight * float(model.vjepa_range_penalty_weight) * range_density
            )
            aux_map = feature_weighted_density + range_weighted_density
            feature_weighted_scalar = aux_weight * feature_loss
            range_weighted_scalar = (
                aux_weight * float(model.vjepa_range_penalty_weight) * range_loss
            )
            aux_scalar = feature_weighted_scalar + range_weighted_scalar
            main_density = _normalize_density(
                _project_latent_map(latent_error, time_steps=output_time),
                main_scalar,
            )
            total_density = main_density + aux_map
            total_scalar = float(main_scalar.item()) + aux_scalar
            pred_frames = _resize_frames(_as_rgb_frames(pred_raw.clamp(0.0, 1.0)))
            density_means = {
                "flow": float(main_density.mean().item()),
                "vjepa_feature": float(feature_weighted_density.mean().item()),
                "vjepa": float(aux_map.mean().item()),
                "total": float(total_density.mean().item()),
            }
            expected_means = {
                "flow": float(main_scalar.item()),
                "vjepa_feature": feature_weighted_scalar,
                "vjepa": aux_scalar,
                "total": total_scalar,
            }
            for map_name in ("flow", "vjepa_feature", "vjepa", "total"):
                if not math.isclose(
                    density_means[map_name],
                    expected_means[map_name],
                    rel_tol=2e-4,
                    abs_tol=1e-10,
                ):
                    raise RuntimeError(
                        f"{map_name} density mean does not recover its scalar: "
                        f"map={density_means[map_name]:.8g}, "
                        f"scalar={expected_means[map_name]:.8g}"
                    )
        variants.append(
            {
                "choice": choice,
                "scalar": {
                    "main_v_mse": float(main_mse.item()),
                    "main_weighted": float(main_scalar.item()),
                    "vjepa_feature": feature_loss,
                    "vjepa_range": range_loss,
                    "vjepa_feature_weighted": feature_weighted_scalar,
                    "vjepa_range_weighted": range_weighted_scalar,
                    "vjepa_weighted_aux": aux_scalar,
                    "total_weighted": total_scalar,
                    "future_token_fraction": future_fraction,
                    "density_mean_check": density_means,
                },
                "density": {
                    "flow": main_density.detach().cpu().numpy().astype(np.float32),
                    "vjepa_feature": (
                        feature_weighted_density.detach().cpu().numpy().astype(np.float32)
                    ),
                    "vjepa": aux_map.detach().cpu().numpy().astype(np.float32),
                    "total": total_density.detach().cpu().numpy().astype(np.float32),
                },
                "pred_frames": pred_frames,
                "flow_loss_frame_indices": flow_loss_frame_indices,
                "flow_frame_labels": flow_frame_labels,
            }
        )
        del latent_xt, training_target, velocity, pred_x0_raw, pred_x0, pred_raw
        torch.cuda.empty_cache()

    scales = {
        name: _vmax([variant["density"][name] for variant in variants])
        for name in ("flow", "vjepa_feature", "vjepa", "total")
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_video(target_frames, case_dir / "target_tiny_vae.mp4")
    _write_video(target_frames, case_dir / "source_frames.mp4")
    variant_records = []
    raw_sample = sample.get("metadata", {})
    case_label = str(raw_sample.get("sample_key", f"{case['source_name']}_{case['global_index']}"))
    for variant in variants:
        choice = variant["choice"]
        variant_dir = case_dir / str(choice["label"])
        variant_dir.mkdir(parents=True, exist_ok=True)
        scalar = variant["scalar"]
        videos = {}
        video_specs = (
            (
                "flow",
                "Flow weighted v-MSE",
                variant["flow_frame_labels"],
            ),
            (
                "vjepa_feature",
                "V-JEPA feature MSE",
                vjepa_frame_labels,
            ),
            (
                "vjepa",
                "V-JEPA auxiliary",
                vjepa_frame_labels,
            ),
            (
                "total",
                "Weighted total",
                [
                    (
                        f"{flow_label}; "
                        f"vjepa_feature={'used' if frame_index in vjepa_loss_frame_set else 'unused'}"
                    )
                    for frame_index, flow_label in enumerate(variant["flow_frame_labels"])
                ],
            ),
        )
        for name, title, frame_labels in video_specs:
            header = (
                f"{title} | {choice['label']} | sigma={choice['sigma']:.3f} "
                f"w={choice['normalized_timestep_weight']:.3f}"
            )
            path = variant_dir / f"{name}_overlay.mp4"
            _write_video(
                _overlay_video(
                    target_frames,
                    variant["density"][name],
                    vmax=scales[name],
                    header=header,
                    frame_labels=frame_labels,
                ),
                path,
            )
            videos[name] = str(path.relative_to(case_dir))
        pred_path = variant_dir / "pred_x0.mp4"
        _write_video(variant["pred_frames"], pred_path)
        variant_records.append(
            {
                "label": choice["label"],
                "scheduler_index": choice["scheduler_index"],
                "timestep": choice["timestep"],
                "sigma": choice["sigma"],
                "raw_timestep_weight": choice["raw_timestep_weight"],
                "normalized_timestep_weight": choice["normalized_timestep_weight"],
                "sigma_gate_normalizer": choice["sigma_gate_normalizer"],
                "scalar": scalar,
                "videos": videos,
                "pred_video": str(pred_path.relative_to(case_dir)),
                "spatial_stats": {
                    "flow": _spatial_stats(
                        variant["density"]["flow"],
                        active_frame_indices=variant["flow_loss_frame_indices"],
                    ),
                    "vjepa_feature": _spatial_stats(
                        variant["density"]["vjepa_feature"],
                        active_frame_indices=vjepa_loss_frame_indices,
                    ),
                    "total": _spatial_stats(
                        variant["density"]["total"],
                        active_frame_indices=list(range(output_time)),
                    ),
                },
            }
        )
    record = {
        "case_id": _safe_name(case_label),
        "case_label": case_label,
        "case_position": int(case["position"]),
        "source": str(case["source_name"]),
        "global_index": int(case["global_index"]),
        "local_index": int(case["local_index"]),
        "case_seed": case_seed,
        "context_cutoff_frame": int(context_cutoff),
        "vjepa_frame_indices": [
            int(value) for value in display_frame_indices.cpu().tolist()
        ],
        "vjepa_model_frame_indices": [
            int(value) for value in model_frame_indices.cpu().tolist()
        ],
        "vjepa_loss_frame_indices": vjepa_loss_frame_indices,
        "vjepa_sampling_local": bool(sampling_local),
        "vjepa_frame_sampling": str(model.vjepa_frame_sampling),
        "vjepa_padded_frame_count": int(
            model_frame_indices.numel() - display_frame_indices.numel()
        ),
        "vjepa_model_frame_count": int(model_frame_indices.numel()),
        "vjepa_temporal_tokens": int(model_frame_indices.numel()) // VJEPA_TUBELET_SIZE,
        "vjepa_input_mode": vjepa_input_mode,
        "vjepa_input_size": [int(value) for value in vjepa_input_hw],
        "vjepa_token_grid": [
            int(model_frame_indices.numel()) // VJEPA_TUBELET_SIZE,
            *[int(value) for value in vjepa_spatial_grid],
        ],
        "model_condition": model_condition,
        "base_pretrained_lora_merged": True,
        "step03463_lora_loaded": model_condition == "step03463_lora",
        "selected_object_count": int(
            case.get("object_count", len(raw_sample.get("entity_slots", [])))
        ),
        "selected_entity_slots": case.get(
            "entity_slots", raw_sample.get("entity_slots", [])
        ),
        "target_video": "target_tiny_vae.mp4",
        "source_video": "source_frames.mp4",
        "loss_input_description": (
            "Tiny-VAE decoded target video. Flow v-MSE is projected from latent "
            "supervision; V-JEPA receives all decoded frames. An odd final frame "
            "is repeated only for tubelet alignment."
        ),
        "color_scales_p99_5": scales,
        "metadata": raw_sample,
        "variants": variant_records,
    }
    _json_dump(case_dir / "manifest.json", record)
    return record


def _worker_main(args: argparse.Namespace, train_argv: list[str]) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "2"))
    device = torch.device(f"cuda:{local_rank}")
    visible = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if set(visible) != {"6", "7"} or "4" in visible:
        raise RuntimeError(f"Expected CUDA_VISIBLE_DEVICES=6,7, got {visible}")
    _seed_all(int(args.seed) + local_rank)
    parser = vjepa_train.build_parser()
    train_args = vjepa_train.core.tvn.prepare_args(parser.parse_args(train_argv))
    dataset = vjepa_train.core.base.build_dataset(train_args)
    cases = _cases_from_args(dataset, args, local_rank=local_rank)
    if args.model_condition == "no_step03463_lora":
        train_args.self_attn_adaptation_mode = "object_only"
    model = vjepa_train.build_model(train_args, SimpleNamespace(device=device))
    merged_count = len(model.merged_pretrained_lora_modules)
    expected_merged_count = int(train_args.pretrained_lora_expected_modules)
    if merged_count != expected_merged_count:
        raise RuntimeError(
            "Base pretrained LoRA was not fully merged into Wan: "
            f"merged={merged_count}/{expected_merged_count}"
        )
    self_attn_lora_names = [
        name
        for name, _ in model.pipe.dit.named_parameters()
        if vjepa_train.core._is_full_self_attn_lora_parameter(name)
    ]
    checkpoint_loaded = False
    loaded_count = 0
    if args.model_condition == "step03463_lora":
        if not self_attn_lora_names:
            raise RuntimeError("Step-3463 condition has no self-attention LoRA parameters")
        checkpoint = vjepa_train.core.tvn._resolve_checkpoint_file(
            Path(args.checkpoint).resolve()
        )
        load_info = vjepa_train.core.tvn._load_filtered_checkpoint_into_model(
            model,
            checkpoint,
            include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
            include_substrings=(".self_attn.",),
        )
        expected_count = sum(
            1 for _, parameter in model.named_parameters() if parameter.requires_grad
        )
        if load_info["loaded_count"] != expected_count or load_info["skipped_shape_mismatch"]:
            raise RuntimeError(
                "Incomplete step-3463 checkpoint load: "
                f"loaded={load_info['loaded_count']}/{expected_count}, "
                f"shape_mismatch={len(load_info['skipped_shape_mismatch'])}"
            )
        checkpoint_loaded = True
        loaded_count = int(load_info["loaded_count"])
    elif self_attn_lora_names:
        raise RuntimeError(
            "no_step03463_lora must not contain self-attention LoRA parameters: "
            f"{self_attn_lora_names[:8]}"
        )
    model.eval()
    schedule_choices = _schedule_choices(
        model.pipe,
        sigma_min=float(train_args.vjepa_sigma_min),
        sigma_max=float(train_args.vjepa_sigma_max),
    )
    output_root = Path(args.output).resolve()
    rank_root = output_root / f"rank{local_rank}"
    rank_root.mkdir(parents=True, exist_ok=True)
    _json_dump(
        rank_root / "model_init.json",
        {
            "model_condition": args.model_condition,
            "base_pretrained_lora_checkpoint": str(train_args.lora_checkpoint),
            "base_pretrained_lora_merged_modules": merged_count,
            "self_attn_lora_parameter_tensors": len(self_attn_lora_names),
            "step03463_checkpoint_loaded": checkpoint_loaded,
            "step03463_loaded_tensors": loaded_count,
        },
    )
    for case in cases:
        if int(case["position"]) % world_size != local_rank:
            continue
        case_dir = rank_root / f"case_{int(case['position']):02d}_{case['source_name']}"
        record = _build_case(
            model=model,
            dataset=dataset,
            case=case,
            schedule_choices=schedule_choices,
            case_dir=case_dir,
            vjepa_input_mode=args.vjepa_input_mode,
            model_condition=args.model_condition,
        )
        print(
            f"rank={local_rank} completed case={record['case_id']} "
            f"source={record['source']} seed={record['case_seed']}",
            flush=True,
        )


def _pairwise_spatial_comparison(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute case-to-case similarity on normalized spatial profiles."""
    if not records:
        return {}
    variants_by_record = [
        {str(variant["label"]): variant for variant in record["variants"]}
        for record in records
    ]
    shared_labels = set(variants_by_record[0])
    for variants in variants_by_record[1:]:
        shared_labels &= set(variants)
    if not shared_labels:
        raise RuntimeError("No shared timestep labels available for spatial comparison")

    case_positions = [int(record["case_position"]) for record in records]
    case_ids = [str(record["case_id"]) for record in records]
    results: dict[str, Any] = {
        "definition": (
            "Profiles are spatially normalized before comparison. Flow aggregates the "
            "projected supervised latent v-MSE; V-JEPA feature uses future-only "
            "tubelet frames and excludes the all-frame range penalty."
        ),
        "case_positions": case_positions,
        "case_ids": case_ids,
        "timesteps": {},
    }
    for label in sorted(shared_labels):
        per_kind: dict[str, Any] = {}
        for kind in ("flow", "vjepa_feature", "total"):
            profiles = []
            for variants in variants_by_record:
                stats = variants[label]["spatial_stats"][kind]
                profile = np.asarray(stats["profile_grid"], dtype=np.float64).reshape(-1)
                profile = np.maximum(profile, 0.0)
                profile /= max(float(profile.sum()), 1e-20)
                profiles.append(profile)
            matrix = np.stack(profiles, axis=0)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            cosine = (matrix @ matrix.T) / np.maximum(norms * norms.T, 1e-20)
            cosine = np.clip(cosine, -1.0, 1.0)
            midpoint = 0.5 * (matrix[:, None, :] + matrix[None, :, :])
            safe_matrix = np.maximum(matrix, 1e-20)
            safe_midpoint = np.maximum(midpoint, 1e-20)
            js = 0.5 * (
                (matrix[:, None, :] * np.log(safe_matrix[:, None, :] / safe_midpoint)).sum(axis=-1)
                + (matrix[None, :, :] * np.log(safe_matrix[None, :, :] / safe_midpoint)).sum(axis=-1)
            )
            upper = np.triu_indices(len(records), k=1)
            cosine_values = cosine[upper]
            js_values = js[upper]
            per_kind[kind] = {
                "cosine_similarity": cosine.astype(np.float32).tolist(),
                "jensen_shannon_divergence": js.astype(np.float32).tolist(),
                "mean_pairwise_cosine_similarity": float(cosine_values.mean()),
                "min_pairwise_cosine_similarity": float(cosine_values.min()),
                "max_pairwise_cosine_similarity": float(cosine_values.max()),
                "mean_pairwise_js_divergence": float(js_values.mean()),
                "max_pairwise_js_divergence": float(js_values.max()),
            }
        results["timesteps"][label] = per_kind
    return results


def _build_index(output_root: Path, *, config_path: Path, checkpoint: Path, seed: int) -> None:
    records = []
    for manifest_path in sorted(output_root.glob("rank*/case_*/manifest.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        prefix = manifest_path.parent.relative_to(output_root).as_posix()
        record["target_video"] = f"{prefix}/{record['target_video']}"
        record["source_video"] = f"{prefix}/{record['source_video']}"
        for variant in record["variants"]:
            for name, relative_path in list(variant["videos"].items()):
                variant["videos"][name] = f"{prefix}/{relative_path}"
            variant["pred_video"] = f"{prefix}/{variant['pred_video']}"
        records.append(record)
    records.sort(key=lambda item: int(item["case_position"]))
    index_payload = {
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "seed": int(seed),
        "vjepa_input_mode": records[0]["vjepa_input_mode"] if records else None,
        "vjepa_input_size": records[0]["vjepa_input_size"] if records else None,
        "vjepa_token_grid": records[0]["vjepa_token_grid"] if records else None,
        "model_condition": records[0]["model_condition"] if records else None,
        "base_pretrained_lora_merged": (
            records[0]["base_pretrained_lora_merged"] if records else None
        ),
        "step03463_lora_loaded": (
            records[0]["step03463_lora_loaded"] if records else None
        ),
        "spatial_comparison": _pairwise_spatial_comparison(records),
        "records": records,
    }
    _json_dump(output_root / "index.json", index_payload)
    vjepa_mode = records[0]["vjepa_input_mode"] if records else "unknown"
    vjepa_size = records[0]["vjepa_input_size"] if records else [0, 0]
    vjepa_grid = records[0]["vjepa_token_grid"] if records else [0, 0, 0]
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA Loss Heatmaps</title>
<style>
:root {{ --bg:#121718; --panel:#1b2221; --line:#3c4a43; --text:#eef4ed; --muted:#aab8ad; --accent:#b8e986; --warm:#f3b562; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }}
header {{ padding:18px 22px 14px; border-bottom:1px solid var(--line); background:#18201e; }}
h1 {{ margin:0 0 5px; font:700 24px/1.1 Georgia,serif; }} h2 {{ margin:0; font-size:18px; }}
.muted {{ color:var(--muted); }} main {{ max-width:1700px; margin:0 auto; padding:18px 22px 30px; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:end; padding:0 0 16px; }}
label {{ display:grid; gap:4px; color:var(--muted); font-size:12px; }} select,button,input[type=range] {{ accent-color:var(--accent); }}
select,button {{ min-height:34px; border:1px solid var(--line); border-radius:5px; background:#25302b; color:var(--text); padding:6px 10px; }}
button {{ cursor:pointer; }} button:hover {{ border-color:var(--accent); }}
.case-head {{ display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:8px; margin:10px 0 13px; }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }} figure {{ margin:0; min-width:0; }} figcaption {{ min-height:38px; color:var(--muted); font-size:12px; }}
video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#050606; border:1px solid var(--line); }}
.source video {{ border-color:var(--warm); }} .facts {{ display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); margin:14px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.fact {{ padding:8px 10px; border-right:1px solid var(--line); }} .fact:last-child {{ border-right:0; }} .fact b {{ display:block; color:var(--accent); font-size:15px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:9px; align-items:center; margin:14px 0; }} .controls input {{ min-width:260px; flex:1; }}
.note {{ color:var(--muted); margin:12px 0 0; }} code {{ color:#d4edae; }}
@media(max-width:1100px) {{ .grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .facts {{ grid-template-columns:repeat(3,minmax(120px,1fr)); }} }}
@media(max-width:650px) {{ main,header {{ padding-left:12px; padding-right:12px; }} .grid {{ grid-template-columns:1fr; }} .facts {{ grid-template-columns:repeat(2,minmax(120px,1fr)); }} }}
</style></head><body>
<header><h1>V-JEPA Loss Heatmaps</h1><div class="muted">Step 3463 · {vjepa_mode} {vjepa_size[0]}×{vjepa_size[1]} · token grid {'×'.join(str(value) for value in vjepa_grid)} · fixed case seed/noise/full-video V-JEPA input</div></header>
<main><div class="toolbar">
<label>Case<select id="case"></select></label><label>Overlay<select id="kind"><option value="flow">Flow loss</option><option value="vjepa">V-JEPA auxiliary</option><option value="total">Weighted total</option></select></label>
<button id="play">Play</button><button id="pause">Pause</button><button id="replay">Replay</button>
</div><div class="case-head"><h2 id="title"></h2><span class="muted" id="sampling"></span></div><div class="facts" id="facts"></div>
<div class="grid" id="grid"></div><div class="controls"><span class="muted" id="frame">frame 0</span><input id="seek" type="range" min="0" max="48" value="0" step="0.01"></div>
<p class="note">Each overlay is a local loss-contribution density on the decoded target video. The three timestep columns share one color scale per loss type within the selected case; the scale is the 99.5th percentile and clipped values are saturated. Flow is projected from latent v-MSE. V-JEPA is the true normalized future-token feature contribution plus its range penalty. The weighted total is the numerical sum of the two contribution maps.</p>
</main><script>
const DATA={payload}; let current=null; const vids=()=>[...document.querySelectorAll('video')];
const casePick=document.getElementById('case'), kindPick=document.getElementById('kind'), grid=document.getElementById('grid'), facts=document.getElementById('facts'), title=document.getElementById('title'), sampling=document.getElementById('sampling'), seek=document.getElementById('seek'), frame=document.getElementById('frame');
DATA.forEach((item,i)=>{{const o=document.createElement('option');o.value=i;o.textContent=item.source+' · '+item.case_id;casePick.appendChild(o);}});
function fmt(v){{return Number(v).toExponential(4)}}
function renderFacts(item){{const rows=[]; item.variants.forEach(v=>rows.push('<div class="fact"><span>'+v.label+' · sigma '+Number(v.sigma).toFixed(3)+' · w '+Number(v.normalized_timestep_weight).toFixed(3)+'</span><b>flow '+fmt(v.scalar.main_weighted)+'</b><b>V-JEPA '+fmt(v.scalar.vjepa_weighted_aux)+'</b><b>total '+fmt(v.scalar.total_weighted)+'</b></div>')); facts.innerHTML=rows.join('');}}
function load(){{current=DATA[Number(casePick.value)||0]; title.textContent=current.source+' · '+current.case_label; const scale=current.color_scales_p99_5[kindPick.value]; const samplingMode=current.vjepa_frame_sampling||'full'; const rawFrames=Number(current.vjepa_frame_indices?.length||0); const paddedFrames=Number(current.vjepa_padded_frame_count||0); const tokenCount=Number(current.vjepa_temporal_tokens||(current.vjepa_token_grid&&current.vjepa_token_grid[0])||0); sampling.textContent='seed '+current.case_seed+' · V-JEPA '+samplingMode+' · '+rawFrames+' raw frames'+(paddedFrames?' (+ '+paddedFrames+' padded)':'')+' · '+tokenCount+' tubelets · color p99.5 '+fmt(scale); renderFacts(current); const labels={{flow:'Flow loss',vjepa:'V-JEPA auxiliary',total:'Weighted total'}}; grid.innerHTML='<figure class="source"><figcaption>Target video used for loss comparison</figcaption><video controls muted playsinline preload="metadata" src="'+current.target_video+'"></video></figure>'+current.variants.map(v=>'<figure><figcaption><b>'+v.label+'</b> · '+labels[kindPick.value]+'<br>sigma '+Number(v.sigma).toFixed(4)+' · raw w '+fmt(v.raw_timestep_weight)+' · normalized w '+fmt(v.normalized_timestep_weight)+'</figcaption><video controls muted playsinline preload="metadata" src="'+v.videos[kindPick.value]+'"></video></figure>').join(''); seek.value=0; frame.textContent='frame 0'; }}
function sync(action){{vids().forEach(v=>action(v));}}
document.getElementById('play').onclick=()=>sync(v=>v.play().catch(()=>{{}})); document.getElementById('pause').onclick=()=>sync(v=>v.pause()); document.getElementById('replay').onclick=()=>sync(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}});
casePick.onchange=load; kindPick.onchange=load; seek.oninput=()=>{{const t=Number(seek.value);sync(v=>v.currentTime=t);frame.textContent='frame '+Math.round(t);}};
grid.addEventListener('loadedmetadata',()=>{{const first=vids()[0];if(first)seek.max=Math.max(0,first.duration||8);}},true); grid.addEventListener('timeupdate',e=>{{if(e.target.tagName==='VIDEO'){{seek.value=e.target.currentTime;frame.textContent='frame '+Math.round(e.target.currentTime*6);}}}},true); load();
</script></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")


def _prefix_record_paths(record: dict[str, Any], run_tag: str) -> dict[str, Any]:
    result = json.loads(json.dumps(record))
    result["target_video"] = f"{run_tag}/{result['target_video']}"
    result["source_video"] = f"{run_tag}/{result['source_video']}"
    for variant in result["variants"]:
        for name, relative_path in list(variant["videos"].items()):
            variant["videos"][name] = f"{run_tag}/{relative_path}"
        variant["pred_video"] = f"{run_tag}/{variant['pred_video']}"
    return result


def _build_comparison_page(
    output_root: Path,
    *,
    center_run_tag: str,
    native_run_tag: str,
    page_name: str,
) -> Path:
    center_index_path = output_root / center_run_tag / "index.json"
    native_index_path = output_root / native_run_tag / "index.json"
    if not center_index_path.is_file():
        raise FileNotFoundError(f"Center-crop baseline index not found: {center_index_path}")
    center_index = json.loads(center_index_path.read_text(encoding="utf-8"))
    native_index = json.loads(native_index_path.read_text(encoding="utf-8"))
    center_records = {
        int(record["case_position"]): _prefix_record_paths(record, center_run_tag)
        for record in center_index["records"]
    }
    native_records = {
        int(record["case_position"]): _prefix_record_paths(record, native_run_tag)
        for record in native_index["records"]
    }
    if center_records.keys() != native_records.keys():
        raise RuntimeError(
            "Center/native case positions differ: "
            f"center={sorted(center_records)}, native={sorted(native_records)}"
        )

    pairs = []
    identity_fields = (
        "case_id",
        "case_label",
        "source",
        "global_index",
        "local_index",
        "case_seed",
        "context_cutoff_frame",
        "vjepa_frame_indices",
        "vjepa_loss_frame_indices",
        "vjepa_frame_sampling",
        "vjepa_model_frame_indices",
        "vjepa_model_frame_count",
        "vjepa_temporal_tokens",
        "vjepa_padded_frame_count",
    )
    schedule_fields = (
        "scheduler_index",
        "timestep",
        "sigma",
        "raw_timestep_weight",
        "normalized_timestep_weight",
        "sigma_gate_normalizer",
    )
    for position in sorted(center_records):
        center = center_records[position]
        native = native_records[position]
        for field in identity_fields:
            if center[field] != native[field]:
                raise RuntimeError(
                    f"Case {position} differs in {field}: "
                    f"center={center[field]!r}, native={native[field]!r}"
                )
        center_variants = {item["label"]: item for item in center["variants"]}
        native_variants = {item["label"]: item for item in native["variants"]}
        if center_variants.keys() != native_variants.keys():
            raise RuntimeError(f"Case {position} timestep labels differ")
        for label in center_variants:
            center_variant = center_variants[label]
            native_variant = native_variants[label]
            for field in schedule_fields:
                if center_variant[field] != native_variant[field]:
                    raise RuntimeError(
                        f"Case {position}/{label} differs in {field}: "
                        f"center={center_variant[field]!r}, native={native_variant[field]!r}"
                    )
            center_flow = float(center_variant["scalar"]["main_weighted"])
            native_flow = float(native_variant["scalar"]["main_weighted"])
            if not math.isclose(center_flow, native_flow, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError(
                    f"Case {position}/{label} flow scalar changed: "
                    f"center={center_flow:.9g}, native={native_flow:.9g}"
                )
        center.setdefault("vjepa_input_mode", "center_crop")
        center.setdefault("vjepa_input_size", [384, 384])
        center.setdefault("vjepa_token_grid", [8, 24, 24])
        pairs.append({"center": center, "native": native})

    payload = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA Input Geometry Comparison</title>
<style>
:root {{ --bg:#111615; --surface:#19201e; --line:#3a4841; --text:#edf3ee; --muted:#a9b7ae; --lime:#b8e986; --amber:#f1b45c; --cyan:#78c8d2; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:linear-gradient(135deg,#111615 0%,#17201d 55%,#111615 100%); color:var(--text); font:14px/1.45 "IBM Plex Sans",Verdana,sans-serif; min-height:100vh; }}
header {{ padding:18px 24px 15px; border-bottom:1px solid var(--line); background:#151c1a; }}
h1 {{ margin:0 0 5px; font:700 24px/1.1 Georgia,serif; letter-spacing:0; }} h2 {{ margin:0; font-size:18px; letter-spacing:0; }}
.muted {{ color:var(--muted); }} main {{ max-width:1680px; margin:0 auto; padding:18px 24px 30px; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:end; padding-bottom:16px; }}
label {{ display:grid; gap:4px; color:var(--muted); font-size:12px; }} select,input[type=range] {{ accent-color:var(--lime); }}
select,.icon {{ min-height:36px; border:1px solid var(--line); border-radius:5px; background:#222c28; color:var(--text); padding:6px 10px; }}
.icon {{ width:38px; padding:0; cursor:pointer; font-size:16px; }} .icon:hover {{ border-color:var(--lime); }}
.case-head {{ display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:8px; margin:8px 0 12px; }}
.grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }} figure {{ margin:0; min-width:0; }} figcaption {{ min-height:53px; color:var(--muted); font-size:12px; }} figcaption b {{ color:var(--text); font-size:14px; }}
video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#050606; border:1px solid var(--line); }} .target video {{ border-color:var(--amber); }} .native video {{ border-color:var(--cyan); }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 16px; font-variant-numeric:tabular-nums; }} th,td {{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; font-weight:500; }} td strong {{ color:var(--lime); }}
.controls {{ display:flex; gap:10px; align-items:center; margin:15px 0 0; }} .controls input {{ flex:1; min-width:220px; }} .frame {{ width:72px; color:var(--muted); font-variant-numeric:tabular-nums; }}
@media(max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} figcaption {{ min-height:0; margin:10px 0 5px; }} }}
@media(max-width:620px) {{ main,header {{ padding-left:12px; padding-right:12px; }} table {{ font-size:11px; }} th,td {{ padding:7px 4px; }} }}
</style></head><body>
<header><h1>V-JEPA Input Geometry Comparison</h1><div class="muted">Step 3463 checkpoint | same cases, seeds, noise, and full-video V-JEPA input</div></header>
<main><div class="toolbar">
<label>Case<select id="case"></select></label>
<label>Loss<select id="kind"><option value="flow">Flow loss</option><option value="vjepa">V-JEPA auxiliary</option><option value="total">Weighted total</option></select></label>
<label>Timestep weight<select id="weight"><option value="low_weight">Low</option><option value="mid_weight">Mid</option><option value="high_weight">High</option></select></label>
<button class="icon" id="play" title="Play all" aria-label="Play all">&#9654;</button><button class="icon" id="pause" title="Pause all" aria-label="Pause all">&#10074;&#10074;</button><button class="icon" id="replay" title="Replay all" aria-label="Replay all">&#8634;</button>
</div><div class="case-head"><h2 id="title"></h2><span class="muted" id="sampling"></span></div>
<table><thead><tr><th>V-JEPA input</th><th>Flow</th><th>V-JEPA</th><th>Total</th><th id="scaleHead">Color p99.5</th></tr></thead><tbody id="facts"></tbody></table>
<div class="grid" id="grid"></div><div class="controls"><span class="frame" id="frame">frame 0</span><input id="seek" type="range" min="0" max="8.2" value="0" step="0.01"></div>
</main><script>
const DATA={payload}; const $=id=>document.getElementById(id); const casePick=$('case'),kindPick=$('kind'),weightPick=$('weight'),grid=$('grid'),facts=$('facts'),title=$('title'),sampling=$('sampling'),seek=$('seek'),frame=$('frame');
const vids=()=>[...document.querySelectorAll('video')]; const fmt=v=>Number(v).toExponential(5); const shape=v=>v.join('x');
DATA.forEach((pair,i)=>{{const r=pair.native,o=document.createElement('option');o.value=i;o.textContent=r.source+' | '+r.case_id;casePick.appendChild(o);}});
function variant(record){{return record.variants.find(v=>v.label===weightPick.value);}}
function load(){{const pair=DATA[Number(casePick.value)||0],center=pair.center,native=pair.native,cv=variant(center),nv=variant(native),kind=kindPick.value; title.textContent=native.source+' | '+native.case_label; const centerFrames=Number(center.vjepa_model_frame_count||center.vjepa_frame_indices?.length||0); const nativeFrames=Number(native.vjepa_model_frame_count||native.vjepa_frame_indices?.length||0); const centerPadded=Number(center.vjepa_padded_frame_count||0); const nativePadded=Number(native.vjepa_padded_frame_count||0); const centerMode=center.vjepa_frame_sampling||'full'; const nativeMode=native.vjepa_frame_sampling||'full'; sampling.textContent='seed '+native.case_seed+' | center '+centerMode+' '+centerFrames+' frames'+(centerPadded?' (+ '+centerPadded+' padded)':'')+' | native '+nativeMode+' '+nativeFrames+' frames'+(nativePadded?' (+ '+nativePadded+' padded)':''); $('scaleHead').textContent=kindPick.options[kindPick.selectedIndex].text+' p99.5'; facts.innerHTML='<tr><td>Center crop '+shape(center.vjepa_input_size)+' | '+shape(center.vjepa_token_grid)+'</td><td>'+fmt(cv.scalar.main_weighted)+'</td><td>'+fmt(cv.scalar.vjepa_weighted_aux)+'</td><td><strong>'+fmt(cv.scalar.total_weighted)+'</strong></td><td>'+fmt(center.color_scales_p99_5[kind])+'</td></tr><tr><td>Native rect '+shape(native.vjepa_input_size)+' | '+shape(native.vjepa_token_grid)+'</td><td>'+fmt(nv.scalar.main_weighted)+'</td><td>'+fmt(nv.scalar.vjepa_weighted_aux)+'</td><td><strong>'+fmt(nv.scalar.total_weighted)+'</strong></td><td>'+fmt(native.color_scales_p99_5[kind])+'</td></tr>'; const label=kindPick.options[kindPick.selectedIndex].text,weight=weightPick.options[weightPick.selectedIndex].text; grid.innerHTML='<figure class="target"><figcaption><b>Target video</b><br>Decoded Tiny-VAE target</figcaption><video controls muted playsinline preload="metadata" src="'+center.target_video+'"></video></figure><figure><figcaption><b>Center crop 384x384</b><br>'+label+' | '+weight+' weight</figcaption><video controls muted playsinline preload="metadata" src="'+cv.videos[kind]+'"></video></figure><figure class="native"><figcaption><b>Native rect 384x672</b><br>'+label+' | '+weight+' weight</figcaption><video controls muted playsinline preload="metadata" src="'+nv.videos[kind]+'"></video></figure>'; seek.value=0; frame.textContent='frame 0';}}
function sync(action){{vids().forEach(video=>action(video));}} $('play').onclick=()=>sync(video=>video.play().catch(()=>{{}})); $('pause').onclick=()=>sync(video=>video.pause()); $('replay').onclick=()=>sync(video=>{{video.currentTime=0;video.play().catch(()=>{{}})}});
casePick.onchange=load; kindPick.onchange=load; weightPick.onchange=load; seek.oninput=()=>{{const t=Number(seek.value);sync(video=>video.currentTime=t);frame.textContent='frame '+Math.round(t*6);}};
grid.addEventListener('loadedmetadata',()=>{{const first=vids()[0];if(first)seek.max=Math.max(0,first.duration||8.2);}},true); grid.addEventListener('timeupdate',event=>{{if(event.target.tagName==='VIDEO'){{seek.value=event.target.currentTime;frame.textContent='frame '+Math.round(event.target.currentTime*6);}}}},true); load();
</script></body></html>"""
    page_path = output_root / page_name
    page_path.write_text(page, encoding="utf-8")
    return page_path


def _build_model_comparison_page(
    output_root: Path,
    *,
    step_run_tag: str,
    no_step_run_tag: str,
    page_name: str,
) -> Path:
    step_index = json.loads(
        (output_root / step_run_tag / "index.json").read_text(encoding="utf-8")
    )
    no_step_index = json.loads(
        (output_root / no_step_run_tag / "index.json").read_text(encoding="utf-8")
    )
    if step_index.get("model_condition") != "step03463_lora":
        raise RuntimeError(f"Unexpected step model condition: {step_index.get('model_condition')}")
    if no_step_index.get("model_condition") != "no_step03463_lora":
        raise RuntimeError(
            f"Unexpected no-step model condition: {no_step_index.get('model_condition')}"
        )
    step_records = {
        int(record["case_position"]): _prefix_record_paths(record, step_run_tag)
        for record in step_index["records"]
    }
    no_step_records = {
        int(record["case_position"]): _prefix_record_paths(record, no_step_run_tag)
        for record in no_step_index["records"]
    }
    if step_records.keys() != no_step_records.keys():
        raise RuntimeError(
            "Step/no-step case positions differ: "
            f"step={sorted(step_records)}, no_step={sorted(no_step_records)}"
        )
    identity_fields = (
        "case_id",
        "case_label",
        "source",
        "global_index",
        "local_index",
        "case_seed",
        "context_cutoff_frame",
        "vjepa_frame_indices",
        "vjepa_frame_sampling",
        "vjepa_input_mode",
        "vjepa_input_size",
        "vjepa_token_grid",
        "vjepa_model_frame_indices",
        "vjepa_model_frame_count",
        "vjepa_temporal_tokens",
        "vjepa_padded_frame_count",
        "selected_object_count",
        "selected_entity_slots",
    )
    schedule_fields = (
        "label",
        "scheduler_index",
        "timestep",
        "sigma",
        "raw_timestep_weight",
        "normalized_timestep_weight",
        "sigma_gate_normalizer",
    )
    pairs = []
    for position in sorted(step_records):
        step = step_records[position]
        no_step = no_step_records[position]
        for field in identity_fields:
            if step[field] != no_step[field]:
                raise RuntimeError(
                    f"Case {position} differs in {field}: "
                    f"step={step[field]!r}, no_step={no_step[field]!r}"
                )
        step_variants = {item["label"]: item for item in step["variants"]}
        no_step_variants = {item["label"]: item for item in no_step["variants"]}
        if step_variants.keys() != no_step_variants.keys():
            raise RuntimeError(f"Case {position} timestep labels differ")
        for label in step_variants:
            for field in schedule_fields:
                if step_variants[label][field] != no_step_variants[label][field]:
                    raise RuntimeError(f"Case {position}/{label} differs in {field}")
        pairs.append({"step": step, "no_step": no_step})

    payload = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Step 3463 LoRA Multi-Object Comparison</title>
<style>
:root {{ --bg:#111615; --surface:#19201e; --line:#3a4841; --text:#edf3ee; --muted:#a9b7ae; --lime:#b8e986; --amber:#f1b45c; --cyan:#78c8d2; }}
* {{ box-sizing:border-box; }} body {{ margin:0; min-height:100vh; background:linear-gradient(135deg,#111615,#17201d 55%,#111615); color:var(--text); font:14px/1.45 "IBM Plex Sans",Verdana,sans-serif; }}
header {{ padding:18px 24px 15px; border-bottom:1px solid var(--line); background:#151c1a; }} h1 {{ margin:0 0 5px; font:700 24px/1.1 Georgia,serif; letter-spacing:0; }} h2 {{ margin:0; font-size:18px; letter-spacing:0; }}
.muted {{ color:var(--muted); }} main {{ max-width:1680px; margin:0 auto; padding:18px 24px 30px; }} .toolbar {{ display:flex; flex-wrap:wrap; gap:10px 16px; align-items:end; padding-bottom:16px; }}
label {{ display:grid; gap:4px; color:var(--muted); font-size:12px; }} select,input[type=range] {{ accent-color:var(--lime); }} select,.icon {{ min-height:36px; border:1px solid var(--line); border-radius:5px; background:#222c28; color:var(--text); padding:6px 10px; }} .icon {{ width:38px; padding:0; cursor:pointer; font-size:16px; }} .icon:hover {{ border-color:var(--lime); }}
.case-head {{ display:flex; flex-wrap:wrap; align-items:baseline; justify-content:space-between; gap:8px; margin:8px 0 2px; }} .objects {{ min-height:22px; margin-bottom:10px; color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }} figure {{ margin:0; min-width:0; }} figcaption {{ min-height:54px; color:var(--muted); font-size:12px; }} figcaption b {{ color:var(--text); font-size:14px; }} video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#050606; border:1px solid var(--line); }} .target video {{ border-color:var(--amber); }} .no-step video {{ border-color:var(--cyan); }}
table {{ width:100%; border-collapse:collapse; margin:12px 0 16px; font-variant-numeric:tabular-nums; }} th,td {{ padding:8px 10px; text-align:right; border-bottom:1px solid var(--line); }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-size:12px; font-weight:500; }} td strong {{ color:var(--lime); }}
.controls {{ display:flex; gap:10px; align-items:center; margin-top:15px; }} .controls input {{ flex:1; min-width:220px; }} .frame {{ width:72px; color:var(--muted); font-variant-numeric:tabular-nums; }}
@media(max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} figcaption {{ min-height:0; margin:10px 0 5px; }} }} @media(max-width:620px) {{ main,header {{ padding-left:12px; padding-right:12px; }} table {{ font-size:11px; }} th,td {{ padding:7px 4px; }} }}
</style></head><body>
<header><h1>Step 3463 LoRA Multi-Object Comparison</h1><div class="muted">PyBullet | six multi-object cases | native V-JEPA 384x672 | shared cases, noise and timesteps</div></header>
<main><div class="toolbar">
<label>Case<select id="case"></select></label><label>View<select id="view"><option value="loss">Loss overlay</option><option value="x0">Predicted x0</option></select></label>
<label id="lossLabel">Loss<select id="kind"><option value="flow">Flow loss</option><option value="vjepa_feature">V-JEPA feature MSE</option><option value="total">Weighted total</option></select></label>
<label>Timestep weight<select id="weight"><option value="low_weight">Low</option><option value="mid_weight">Mid</option><option value="high_weight">High</option></select></label>
<button class="icon" id="play" title="Play all" aria-label="Play all">&#9654;</button><button class="icon" id="pause" title="Pause all" aria-label="Pause all">&#10074;&#10074;</button><button class="icon" id="replay" title="Replay all" aria-label="Replay all">&#8634;</button>
</div><div class="case-head"><h2 id="title"></h2><span class="muted" id="sampling"></span></div><div class="objects" id="objects"></div>
<table><thead><tr><th>Model condition</th><th>Flow</th><th>V-JEPA feature</th><th>Total</th><th id="scaleHead">Color p99.5</th></tr></thead><tbody id="facts"></tbody></table>
<div class="grid" id="grid"></div><div class="controls"><span class="frame" id="frame">frame 0</span><input id="seek" type="range" min="0" max="8.2" value="0" step="0.01"></div>
</main><script>
const DATA={payload}; const $=id=>document.getElementById(id); const casePick=$('case'),viewPick=$('view'),kindPick=$('kind'),weightPick=$('weight'),grid=$('grid'),facts=$('facts'),title=$('title'),sampling=$('sampling'),objects=$('objects'),seek=$('seek'),frame=$('frame');
const vids=()=>[...document.querySelectorAll('video')]; const fmt=v=>Number(v).toExponential(5); DATA.forEach((pair,i)=>{{const r=pair.step,o=document.createElement('option');o.value=i;o.textContent=r.case_id+' | '+r.selected_object_count+' objects';casePick.appendChild(o);}}); function variant(record){{return record.variants.find(v=>v.label===weightPick.value);}}
function load(){{const pair=DATA[Number(casePick.value)||0],step=pair.step,noStep=pair.no_step,sv=variant(step),nv=variant(noStep),kind=kindPick.value,isLoss=viewPick.value==='loss'; title.textContent=step.case_label; const stepMode=step.vjepa_frame_sampling||'full'; const noStepMode=noStep.vjepa_frame_sampling||'full'; const stepFrames=Number(step.vjepa_model_frame_count||step.vjepa_frame_indices?.length||0); const noStepFrames=Number(noStep.vjepa_model_frame_count||noStep.vjepa_frame_indices?.length||0); const stepPadded=Number(step.vjepa_padded_frame_count||0); const noStepPadded=Number(noStep.vjepa_padded_frame_count||0); sampling.textContent='seed '+step.case_seed+' | step '+stepMode+' '+stepFrames+' frames'+(stepPadded?' (+ '+stepPadded+' padded)':'')+' | no-step '+noStepMode+' '+noStepFrames+' frames'+(noStepPadded?' (+ '+noStepPadded+' padded)':'')+' | feature-loss tubelets '+step.vjepa_loss_frame_indices.join(',')+' | sigma '+Number(sv.sigma).toFixed(4)+' | w '+Number(sv.normalized_timestep_weight).toFixed(4); objects.textContent=step.selected_object_count+' objects | '+step.selected_entity_slots.map(x=>x.object_phrase||x.object_noun).join(' | '); $('lossLabel').style.display=isLoss?'grid':'none'; $('scaleHead').textContent=isLoss?kindPick.options[kindPick.selectedIndex].text+' p99.5':'Input structure'; facts.innerHTML='<tr><td>step03463_lora</td><td>'+fmt(sv.scalar.main_weighted)+'</td><td>'+fmt(sv.scalar.vjepa_feature_weighted)+'</td><td><strong>'+fmt(sv.scalar.total_weighted)+'</strong></td><td>'+(isLoss?fmt(step.color_scales_p99_5[kind]):'base merge + step LoRA')+'</td></tr><tr><td>no_step03463_lora</td><td>'+fmt(nv.scalar.main_weighted)+'</td><td>'+fmt(nv.scalar.vjepa_feature_weighted)+'</td><td><strong>'+fmt(nv.scalar.total_weighted)+'</strong></td><td>'+(isLoss?fmt(noStep.color_scales_p99_5[kind]):'base merge only')+'</td></tr>'; const viewLabel=isLoss?kindPick.options[kindPick.selectedIndex].text:'Predicted x0',stepSrc=isLoss?sv.videos[kind]:sv.pred_video,noStepSrc=isLoss?nv.videos[kind]:nv.pred_video; grid.innerHTML='<figure class="target"><figcaption><b>Loss-input target</b><br>Tiny-VAE decoded target frames</figcaption><video controls muted playsinline preload="metadata" src="'+step.target_video+'"></video></figure><figure><figcaption><b>step03463_lora</b><br>'+viewLabel+' | '+weightPick.options[weightPick.selectedIndex].text+' weight</figcaption><video controls muted playsinline preload="metadata" src="'+stepSrc+'"></video></figure><figure class="no-step"><figcaption><b>no_step03463_lora</b><br>'+viewLabel+' | base pretrained LoRA merged</figcaption><video controls muted playsinline preload="metadata" src="'+noStepSrc+'"></video></figure>'; seek.value=0; frame.textContent='frame 0';}}
function sync(action){{vids().forEach(video=>action(video));}} $('play').onclick=()=>sync(video=>video.play().catch(()=>{{}})); $('pause').onclick=()=>sync(video=>video.pause()); $('replay').onclick=()=>sync(video=>{{video.currentTime=0;video.play().catch(()=>{{}})}}); casePick.onchange=load; viewPick.onchange=load; kindPick.onchange=load; weightPick.onchange=load; seek.oninput=()=>{{const t=Number(seek.value);sync(video=>video.currentTime=t);frame.textContent='frame '+Math.round(t*6);}}; grid.addEventListener('loadedmetadata',()=>{{const first=vids()[0];if(first)seek.max=Math.max(0,first.duration||8.2);}},true); grid.addEventListener('timeupdate',event=>{{if(event.target.tagName==='VIDEO'){{seek.value=event.target.currentTime;frame.textContent='frame '+Math.round(event.target.currentTime*6);}}}},true); load();
</script></body></html>"""
    page_path = output_root / page_name
    page_path.write_text(page, encoding="utf-8")
    return page_path


def _build_all_frames_model_comparison_page(
    output_root: Path,
    *,
    step_run_tag: str,
    no_step_run_tag: str,
    page_name: str,
) -> Path:
    """Build a synchronized six-case gallery with normalized spatial statistics."""
    step_index = json.loads(
        (output_root / step_run_tag / "index.json").read_text(encoding="utf-8")
    )
    no_step_index = json.loads(
        (output_root / no_step_run_tag / "index.json").read_text(encoding="utf-8")
    )
    step_records = {
        int(record["case_position"]): _prefix_record_paths(record, step_run_tag)
        for record in step_index["records"]
    }
    no_step_records = {
        int(record["case_position"]): _prefix_record_paths(record, no_step_run_tag)
        for record in no_step_index["records"]
    }
    if step_records.keys() != no_step_records.keys():
        raise RuntimeError(
            "All-frame model comparison case positions differ: "
            f"step={sorted(step_records)}, no_step={sorted(no_step_records)}"
        )
    pairs = []
    for position in sorted(step_records):
        step = step_records[position]
        no_step = no_step_records[position]
        for field in (
            "case_id",
            "case_label",
            "case_seed",
            "global_index",
            "vjepa_frame_indices",
            "vjepa_loss_frame_indices",
            "vjepa_frame_sampling",
            "vjepa_input_mode",
            "vjepa_input_size",
            "vjepa_token_grid",
            "vjepa_model_frame_indices",
            "vjepa_model_frame_count",
            "vjepa_temporal_tokens",
            "vjepa_padded_frame_count",
        ):
            if step[field] != no_step[field]:
                raise RuntimeError(
                    f"All-frame case {position} differs in {field}: "
                    f"step={step[field]!r}, no_step={no_step[field]!r}"
                )
        pairs.append({"step": step, "no_step": no_step})

    payload = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PyBullet Multi-Object Loss Gallery</title>
<style>
:root {
  --bg:#101513; --panel:#17201c; --line:#3a4b40; --text:#edf5ee;
  --muted:#a5b6aa; --lime:#b8e986; --amber:#f1b45c; --cyan:#79c8d2;
}
* { box-sizing:border-box; }
body { margin:0; background:linear-gradient(135deg,#101513,#1a241f 56%,#101513);
  color:var(--text); font:14px/1.45 "IBM Plex Sans",Verdana,sans-serif; }
header { padding:18px 24px 15px; border-bottom:1px solid var(--line);
  background:#141c18; }
h1 { margin:0 0 5px; font:700 24px/1.1 Georgia,serif; }
h2 { margin:0; font-size:18px; }
.muted { color:var(--muted); }
main { max-width:1900px; margin:0 auto; padding:18px 24px 36px; }
.toolbar { display:flex; flex-wrap:wrap; align-items:end; gap:10px 16px;
  padding-bottom:14px; border-bottom:1px solid var(--line); }
label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
select,.icon { min-height:36px; border:1px solid var(--line); border-radius:5px;
  background:#222e27; color:var(--text); padding:6px 10px; }
select { accent-color:var(--lime); }
.icon { width:38px; padding:0; cursor:pointer; font-size:16px; }
.icon:hover { border-color:var(--lime); }
.status { margin:14px 0 8px; color:var(--muted); }
.case { border-top:1px solid var(--line); padding:16px 0 18px; }
.case-title { display:flex; flex-wrap:wrap; justify-content:space-between;
  align-items:baseline; gap:8px; margin-bottom:8px; }
.case-title b { color:var(--text); font-size:16px; }
.objects { color:var(--muted); font-size:12px; margin-bottom:10px; }
.videos { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
figure { margin:0; min-width:0; }
figcaption { min-height:42px; color:var(--muted); font-size:12px; }
figcaption b { color:var(--text); font-size:13px; }
video { display:block; width:100%; aspect-ratio:16/9; object-fit:contain;
  background:#050606; border:1px solid var(--line); }
.target video { border-color:var(--amber); }
.flow video { border-color:#d67e57; }
.feature video { border-color:var(--lime); }
.total video { border-color:var(--cyan); }
.stats { margin:18px 0 0; padding-top:14px; border-top:1px solid var(--line); }
.stats-head { display:flex; flex-wrap:wrap; align-items:baseline; gap:12px; }
.stats-head b { font-size:16px; }
.metric { margin:10px 0; color:var(--muted); }
.matrix-wrap { overflow:auto; }
table { border-collapse:collapse; width:max-content; min-width:520px;
  font-variant-numeric:tabular-nums; }
th,td { padding:5px 8px; border:1px solid var(--line); text-align:right; }
th { color:var(--muted); font-size:11px; font-weight:500; }
th:first-child,td:first-child { text-align:left; }
td { min-width:62px; }
.controls { display:flex; gap:10px; align-items:center; margin-top:14px; }
.controls input { flex:1; min-width:220px; accent-color:var(--lime); }
.frame { width:76px; color:var(--muted); font-variant-numeric:tabular-nums; }
code { color:#d3edac; }
@media(max-width:1250px) { .videos { grid-template-columns:repeat(2,minmax(0,1fr)); } }
@media(max-width:680px) {
  main,header { padding-left:12px; padding-right:12px; }
  .videos { grid-template-columns:1fr; }
}
</style>
</head>
<body>
<header>
  <h1>PyBullet Multi-Object Loss Gallery</h1>
  <div class="muted">Six shared cases | native rectangular V-JEPA input 384x672 | complete 49-frame timeline</div>
</header>
<main>
  <div class="toolbar">
    <label>Model condition
      <select id="condition">
        <option value="step">step03463_lora</option>
        <option value="no_step">no_step03463_lora</option>
      </select>
    </label>
    <label>Loss profile
      <select id="kind">
        <option value="flow">Flow weighted v-MSE</option>
        <option value="vjepa_feature">V-JEPA feature MSE</option>
        <option value="total">Weighted total</option>
      </select>
    </label>
    <label>Timestep weight
      <select id="weight">
        <option value="low_weight">Low</option>
        <option value="mid_weight">Mid</option>
        <option value="high_weight">High</option>
      </select>
    </label>
    <button class="icon" id="play" title="Play all videos" aria-label="Play all">&#9654;</button>
    <button class="icon" id="pause" title="Pause all videos" aria-label="Pause all">&#10074;&#10074;</button>
    <button class="icon" id="replay" title="Replay all videos" aria-label="Replay all">&#8634;</button>
  </div>
  <div class="status" id="status"></div>
  <div id="gallery"></div>
  <section class="stats">
    <div class="stats-head"><b>Cross-case spatial distribution</b><span class="muted" id="statsMeta"></span></div>
    <div class="metric" id="metric"></div>
    <div class="matrix-wrap"><table id="matrix"></table></div>
  </section>
  <div class="controls">
    <span class="frame" id="frame">frame 0</span>
    <input id="seek" type="range" min="0" max="8.2" value="0" step="0.01">
  </div>
</main>
<script>
const DATA=__PAYLOAD__;
const $=id=>document.getElementById(id);
const condition=$("condition"),kind=$("kind"),weight=$("weight");
const gallery=$("gallery"),status=$("status"),statsMeta=$("statsMeta");
const metric=$("metric"),matrix=$("matrix"),seek=$("seek"),frame=$("frame");
const labels={flow:"Flow weighted v-MSE",vjepa_feature:"V-JEPA feature MSE",total:"Weighted total"};
const videos=()=>[...document.querySelectorAll("video")];
function getRecord(pair){return pair[condition.value];}
function getVariant(record){return record.variants.find(value=>value.label===weight.value);}
function objectText(record){return record.selected_entity_slots.map(value=>value.object_phrase||value.object_noun).join(" | ");}
function profile(record){return getVariant(record).spatial_stats[kind.value].profile_grid.flat();}
function normalize(values){const clipped=values.map(value=>Math.max(Number(value),0));const total=clipped.reduce((a,b)=>a+b,0)||1;return clipped.map(value=>value/total);}
function cosine(a,b){let dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return dot/(Math.sqrt(aa*bb)||1);}
function jsDivergence(a,b){const eps=1e-20;let first=0,second=0;for(let i=0;i<a.length;i++){const mid=0.5*(a[i]+b[i]);if(a[i]>0)first+=0.5*a[i]*Math.log(Math.max(a[i],eps)/Math.max(mid,eps));if(b[i]>0)second+=0.5*b[i]*Math.log(Math.max(b[i],eps)/Math.max(mid,eps));}return first+second;}
function matrixColor(value){const clipped=Math.max(0,Math.min(1,Number(value)));const hue=18+clipped*105;return "hsl("+hue+" 58% "+(20+clipped*35)+"%)";}
function renderStats(){
  const records=DATA.map(getRecord),profiles=records.map(record=>normalize(profile(record))),n=records.length;
  const cosineMatrix=[],jsMatrix=[];for(let row=0;row<n;row++){cosineMatrix[row]=[];jsMatrix[row]=[];for(let col=0;col<n;col++){cosineMatrix[row][col]=cosine(profiles[row],profiles[col]);jsMatrix[row][col]=jsDivergence(profiles[row],profiles[col]);}}
  let sum=0,maxJs=0,count=0;for(let row=0;row<n;row++)for(let col=row+1;col<n;col++){sum+=cosineMatrix[row][col];maxJs=Math.max(maxJs,jsMatrix[row][col]);count++;}
  statsMeta.textContent=labels[kind.value]+" | normalized spatial profile | "+weight.value;
  metric.textContent="Mean pairwise cosine similarity "+(sum/Math.max(count,1)).toFixed(4)+" | maximum pairwise Jensen-Shannon divergence "+maxJs.toFixed(4)+" | profiles aggregate only frames participating in the selected loss.";
  let html="<thead><tr><th>case</th>"+records.map(record=>"<th>"+record.case_position+"</th>").join("")+"</tr></thead><tbody>";
  for(let row=0;row<n;row++){html+="<tr><th>"+records[row].case_position+" · "+records[row].case_id+"</th>";for(let col=0;col<n;col++){html+='<td style="background:'+matrixColor(cosineMatrix[row][col])+'">'+cosineMatrix[row][col].toFixed(3)+"</td>";}html+="</tr>";}
  matrix.innerHTML=html+"</tbody>";
}
function render(){
  const records=DATA.map(getRecord),selectedWeight=weight.value;
  status.textContent="Showing "+labels[kind.value]+" on "+records.length+" complete videos | selected timestep "+selectedWeight+" | V-JEPA input is full-video; only future tubelets contribute; Flow is latent v-MSE projected to the 49-frame target.";
  gallery.innerHTML=records.map(record=>{
    const variant=getVariant(record),stats=variant.spatial_stats[kind.value];
    const statsText="centroid ("+stats.centroid_xy_normalized.map(value=>Number(value).toFixed(3)).join(", ")+") | entropy "+Number(stats.normalized_entropy).toFixed(3)+" | area>uniform "+Number(stats.active_area_fraction_above_uniform).toFixed(3);
    return '<section class="case"><div class="case-title"><b>case '+record.case_position+" | "+record.case_label+" | seed "+record.case_seed+"</b><span class=\"muted\">sigma "+Number(variant.sigma).toFixed(4)+" | normalized weight "+Number(variant.normalized_timestep_weight).toFixed(4)+" | "+statsText+"</span></div><div class=\"objects\">"+record.selected_object_count+" objects | "+objectText(record)+"</div><div class=\"videos\">"+
      '<figure class="target"><figcaption><b>Loss-input target</b><br>decoded Tiny-VAE target</figcaption><video controls muted playsinline preload="metadata" src="'+record.target_video+'"></video></figure>'+
      '<figure class="flow"><figcaption><b>Flow</b><br>weighted v-MSE projected to target frames</figcaption><video controls muted playsinline preload="metadata" src="'+variant.videos.flow+'"></video></figure>'+
      '<figure class="feature"><figcaption><b>V-JEPA feature</b><br>full-video input; future-only tubelets carry feature heat</figcaption><video controls muted playsinline preload="metadata" src="'+variant.videos.vjepa_feature+'"></video></figure>'+
      '<figure class="total"><figcaption><b>Weighted total</b><br>flow + V-JEPA auxiliary contribution</figcaption><video controls muted playsinline preload="metadata" src="'+variant.videos.total+'"></video></figure></div></section>';
  }).join("");
  renderStats();seek.value=0;frame.textContent="frame 0";
}
function sync(action){videos().forEach(video=>action(video));}
$("play").onclick=()=>sync(video=>video.play().catch(()=>{}));
$("pause").onclick=()=>sync(video=>video.pause());
$("replay").onclick=()=>sync(video=>{video.currentTime=0;video.play().catch(()=>{});});
condition.onchange=render;kind.onchange=render;weight.onchange=render;
seek.oninput=()=>{const time=Number(seek.value);sync(video=>video.currentTime=time);frame.textContent="frame "+Math.round(time*6);};
gallery.addEventListener("loadedmetadata",event=>{if(event.target.tagName==="VIDEO")seek.max=Math.max(0,event.target.duration||8.2);},true);
gallery.addEventListener("timeupdate",event=>{if(event.target.tagName==="VIDEO"){seek.value=event.target.currentTime;frame.textContent="frame "+Math.round(event.target.currentTime*6);}},true);
render();
</script>
</body>
</html>
"""
    page_path = output_root / page_name
    page_path.write_text(page.replace("__PAYLOAD__", payload), encoding="utf-8")
    return page_path


def _parent_main(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    raw, _ = launch_from_config.load_config(config_path)
    resolved = launch_from_config.validate_config(raw, config_path.parent)
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.model_condition == "step03463_lora":
        checkpoint_file = vjepa_train.core.tvn._resolve_checkpoint_file(checkpoint)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(checkpoint_file)
    resolved["launch"]["gpu_set"] = "6,7"
    resolved["launch"]["num_processes"] = 2
    resolved["launch"]["main_process_port"] = int(args.accelerate_port)
    resolved["checkpointing"]["resume_from"] = (
        str(checkpoint) if args.model_condition == "step03463_lora" else None
    )
    from datetime import datetime, timezone

    run_tag = args.run_tag or f"step03463_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_root = args.output_root.expanduser().resolve() / run_tag
    output_root.mkdir(parents=True, exist_ok=False)
    _json_dump(
        output_root / "run_spec.json",
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint),
            "gpu_set": "6,7",
            "num_cases": int(args.num_cases),
            "seed": int(args.seed),
            "vjepa_input_mode": args.vjepa_input_mode,
            "case_selection": args.case_selection,
            "case_plan": str(args.case_plan.expanduser().resolve()) if args.case_plan else None,
            "model_condition": args.model_condition,
            "base_pretrained_lora_checkpoint": resolved["paths"][
                "pretrained_lora_checkpoint"
            ],
            "base_pretrained_lora_merge_required": True,
            "step03463_checkpoint_loaded": args.model_condition == "step03463_lora",
            "interpretation": "local_loss_contribution_density",
        },
    )
    command = launch_from_config.build_command(resolved, output_root)
    train_script_tokens = {
        str(launch_from_config.TRAIN_SCRIPT),
        str(launch_from_config.VJEPA_LOSS_TRAIN_SCRIPT),
    }
    script_index = next(
        (index for index, token in enumerate(command) if token in train_script_tokens),
        None,
    )
    if script_index is None:
        raise RuntimeError("Could not find the V-JEPA training script in launch command")
    worker_prefix = [
        str(SCRIPT_DIR / "visualize_vjepa_loss_heatmaps.py"),
        "--as-worker",
        "--output",
        str(output_root),
        "--checkpoint",
        str(checkpoint),
        "--num-cases",
        str(args.num_cases),
        "--seed",
        str(args.seed),
        "--vjepa-input-mode",
        args.vjepa_input_mode,
        "--case-selection",
        args.case_selection,
        "--model-condition",
        args.model_condition,
    ]
    if args.case_plan is not None:
        worker_prefix.extend(["--case-plan", str(args.case_plan.expanduser().resolve())])
    worker_command = command[:script_index] + worker_prefix + command[script_index + 1 :]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "6,7",
            "PYTHONNOUSERSITE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": os.pathsep.join(
                [str(PROJECT_ROOT), str(DIFFSYNTH_ROOT), str(TRAIN_XSSC_DIR), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
        }
    )
    print("GPU 6,7 visualization launch:", flush=True)
    print(" ".join(subprocess.list2cmdline([str(item)]) for item in worker_command), flush=True)
    subprocess.run(worker_command, env=env, check=True)
    _build_index(output_root, config_path=config_path, checkpoint=checkpoint, seed=int(args.seed))
    serve_root = output_root
    if args.vjepa_input_mode == "native_rect" and args.compare_run_tag:
        comparison_path = _build_comparison_page(
            args.output_root.expanduser().resolve(),
            center_run_tag=args.compare_run_tag,
            native_run_tag=run_tag,
            page_name=args.comparison_page,
        )
        serve_root = args.output_root.expanduser().resolve()
        print(f"Comparison page: {comparison_path}", flush=True)
    if args.compare_model_run_tag:
        if args.model_condition != "no_step03463_lora":
            raise ValueError(
                "--compare-model-run-tag must be used while generating no_step03463_lora"
            )
        model_comparison_path = _build_model_comparison_page(
            args.output_root.expanduser().resolve(),
            step_run_tag=args.compare_model_run_tag,
            no_step_run_tag=run_tag,
            page_name=args.model_comparison_page,
        )
        serve_root = args.output_root.expanduser().resolve()
        print(f"Model comparison page: {model_comparison_path}", flush=True)
        all_frames_path = _build_all_frames_model_comparison_page(
            args.output_root.expanduser().resolve(),
            step_run_tag=args.compare_model_run_tag,
            no_step_run_tag=run_tag,
            page_name=args.all_frames_comparison_page,
        )
        print(f"All-frame gallery page: {all_frames_path}", flush=True)
    print(f"Visualization artifacts: {output_root}", flush=True)
    print(
        f"Foreground server command: {sys.executable} -m http.server 8787 "
        f"--bind 127.0.0.1 --directory {serve_root}",
        flush=True,
    )


def _parse_parent_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize formal V-JEPA local loss maps")
    parser.add_argument("config", type=Path, default=DEFAULT_CONFIG, nargs="?")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3463)
    parser.add_argument("--accelerate-port", type=int, default=29567)
    parser.add_argument(
        "--vjepa-input-mode",
        choices=VJEPA_INPUT_MODES,
        default="center_crop",
    )
    parser.add_argument("--case-selection", choices=CASE_SELECTION_MODES, default="mixture")
    parser.add_argument("--case-plan", type=Path, default=None)
    parser.add_argument("--model-condition", choices=MODEL_CONDITIONS, default="step03463_lora")
    parser.add_argument("--compare-run-tag", default=None)
    parser.add_argument("--comparison-page", default=DEFAULT_COMPARISON_PAGE)
    parser.add_argument("--compare-model-run-tag", default=None)
    parser.add_argument(
        "--model-comparison-page",
        default="comparison_step03463_pybullet_multiobject.html",
    )
    parser.add_argument(
        "--all-frames-comparison-page",
        default="comparison_step03463_pybullet_multiobject_all_frames.html",
    )
    args = parser.parse_args()
    if args.num_cases <= 0:
        parser.error("--num-cases must be positive")
    if Path(args.comparison_page).name != args.comparison_page:
        parser.error("--comparison-page must be a file name")
    if Path(args.model_comparison_page).name != args.model_comparison_page:
        parser.error("--model-comparison-page must be a file name")
    if Path(args.all_frames_comparison_page).name != args.all_frames_comparison_page:
        parser.error("--all-frames-comparison-page must be a file name")
    if args.case_selection == "pybullet_multiobject" and args.case_plan is None:
        parser.error("--case-plan is required for pybullet_multiobject selection")
    return args


def _parse_worker_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--as-worker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vjepa-input-mode", choices=VJEPA_INPUT_MODES, required=True)
    parser.add_argument("--case-selection", choices=CASE_SELECTION_MODES, required=True)
    parser.add_argument("--case-plan", type=Path, default=None)
    parser.add_argument("--model-condition", choices=MODEL_CONDITIONS, required=True)
    known, rest = parser.parse_known_args()
    if known.num_cases <= 0:
        parser.error("--num-cases must be positive")
    return known, rest


def main() -> None:
    if "--as-worker" in sys.argv:
        args, rest = _parse_worker_args()
        _worker_main(args, rest)
    else:
        _parent_main(_parse_parent_args())


if __name__ == "__main__":
    main()
