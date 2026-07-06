"""
Batch Wan2.2 TI2V inference over a txt file that lists one input json per line.

Examples

Baseline:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py \
    --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name wan2p2_ti2v5b_baseline \
    --backend official \
    --size 704*1280 \
    --frame-num 49 \
    --sampling-steps 40 \
    --cfg-scale 5.0 \
    --fps 30 \
    --seed 42 \
    --offload-model \
    --vjepa-preset baseline

Guided:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py \
    --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
    --model-name wan2p2_ti2v5b_target_w24_s15_ratio_0025 \
    --backend official \
    --size 704*1280 \
    --frame-num 49 \
    --sampling-steps 40 \
    --cfg-scale 5.0 \
    --fps 30 \
    --seed 42 \
    --offload-model \
    --vjepa-preset target_w24_s15_ratio_0025 \
    --vjepa-ckpt /data/gaoya/ckpt/VJEPA2/vith.pt

Outputs default to:
- /data/gaoya/AAA_test_video/0623/test/v2v/basemodel/<model-name>
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw
from tqdm import tqdm

from code_vjepa_free.vjepa_guidance import WanVJEPAConfig, apply_train0705_preset
from code_vjepa_free.vjepa_guidance.motion_masks import extract_motion_mask_thw
from code_vjepa_free.vjepa_guidance.mask_video_viz import (
    render_background_overlay_video,
    render_binary_mask_video,
    render_motion_overlay_video,
    write_mp4_h264,
)
from code_vjepa_free.vjepa_guidance.build_trace_viewer import build_html as build_trace_viewer_html
from code_vjepa_free.vjepa_guidance.build_trace_viewer import collect_cases as collect_trace_cases
from code_vjepa_free.vjepa_guidance.spectral_guidance import (
    compute_temporal_lowpass_residual_map,
    dilate_mask_thw,
)
from code_vjepa_free.vjepa_guidance.vjepa_surprise import build_context_future_clip
from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root
from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_OFFICIAL_WAN_ROOT,
    WanTI2VArgs,
    _ensure_official_wan_imports,
    build_run_manifest,
    cleanup_pipeline,
    convert_official_video_to_thwc,
    ensure_cuda_env,
    ensure_firstframe_image,
    ensure_str_field,
    load_json,
    normalize_sample_solver,
    patch_wanmodel_from_pretrained_defaults,
    read_list_file,
    resolve_default_cfg_scale,
    resolve_default_frame_num,
    resolve_default_sample_shift,
    resolve_default_sampling_steps,
    resolve_official_wan_root,
    save_video_np,
    write_json,
)
from code_vjepa_free.vjepa_guidance.vjepa_surprise import VJEPASurpriseEnergy
from code_vjepa_free.vjepa_guidance.wan_latent_guidance import (
    decode_preview_video,
    pick_guidance_step_indices,
    predict_x0_from_flow_model_output,
)

_ensure_official_wan_imports()
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS  # type: ignore  # noqa: E402
from wan.textimage2video import WanTI2V  # type: ignore  # noqa: E402
from wan.utils.fm_solvers import (  # type: ignore  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # type: ignore  # noqa: E402
from wan.utils.utils import best_output_size, masks_like  # type: ignore  # noqa: E402


DEFAULT_OUTPUT_BASE_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v/basemodel")
DEFAULT_MODEL_NAME = "wan2p2_ti2v5B"
DEFAULT_VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_gradient(gradient: torch.Tensor, mode: str = "rms", eps: float = 1e-6) -> torch.Tensor:
    if mode == "none":
        return gradient
    if mode == "rms":
        scale = gradient.pow(2).mean().sqrt().clamp_min(eps)
        return gradient / scale
    if mode == "l2":
        scale = gradient.norm().clamp_min(eps)
        return gradient / scale
    raise ValueError(f"Unsupported gradient normalization mode: {mode}")


def _pixel_frames_to_latent_len(pixel_frames: int, temporal_stride: int) -> int:
    if pixel_frames <= 0:
        return 0
    temporal_stride = max(1, int(temporal_stride))
    return (pixel_frames - 1) // temporal_stride + 1


def _pixel_frames_to_preview_len(pixel_frames: int, preview_frame_stride: int) -> int:
    if pixel_frames <= 0:
        return 0
    preview_frame_stride = max(1, int(preview_frame_stride))
    return (pixel_frames - 1) // preview_frame_stride + 1


def _video_btchw_to_u8(video_btchw: torch.Tensor) -> np.ndarray:
    if video_btchw.ndim != 5:
        raise ValueError(f"Expected [B,C,T,H,W], got {tuple(video_btchw.shape)}")
    if int(video_btchw.shape[0]) != 1:
        raise ValueError(f"Expected batch size 1 for mask extraction, got {tuple(video_btchw.shape)}")
    video = video_btchw[0].detach().float().clamp(-1.0, 1.0)
    video = ((video + 1.0) * 127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).contiguous().cpu().numpy()


def _motion_mask_from_preview_video(
    preview_video: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    preview_u8 = _video_btchw_to_u8(preview_video)
    temporal_union = mode in {"temporal_union", "temporal_union_except_first"}
    mask_np = extract_motion_mask_thw(
        preview_u8,
        method="background_residual",
        temporal_union=temporal_union,
        temporal_union_exclude_first_frame=(mode == "temporal_union_except_first"),
        temporal_union_zero_first_frame=(mode == "temporal_union_except_first"),
    )
    return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0)


def _preview_video_to_u8(preview_video: torch.Tensor) -> np.ndarray:
    video = preview_video.detach().float().clamp(-1.0, 1.0)
    if video.ndim == 5:
        if int(video.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected preview video [C,T,H,W], got {tuple(video.shape)}")
    video_u8 = ((video + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy()
    return np.transpose(video_u8, (1, 2, 3, 0))


def _weight_map_to_u8(weight_bthw: torch.Tensor) -> np.ndarray:
    weight = weight_bthw.detach().float().cpu()
    if weight.ndim == 4:
        if int(weight.shape[0]) != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(weight.shape)}")
        weight = weight[0]
    if weight.ndim != 3:
        raise ValueError(f"Expected [T,H,W], got {tuple(weight.shape)}")
    max_value = float(weight.max().item()) if weight.numel() > 0 else 0.0
    if max_value <= 1.0e-6:
        normalized = torch.zeros_like(weight)
    else:
        normalized = (weight / max_value).clamp(0.0, 1.0)
    gray = (normalized * 255.0).round().to(torch.uint8).numpy()
    return np.repeat(gray[..., None], 3, axis=-1)


def _overlay_mask_to_video(preview_video_u8: np.ndarray, mask_thw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    motion_overlay = render_motion_overlay_video(preview_video_u8, mask_thw)
    background_overlay = render_background_overlay_video(preview_video_u8, mask_thw)
    binary_video = render_binary_mask_video(mask_thw)
    return binary_video, motion_overlay, background_overlay


def _video_thwc_u8_to_pil_frames(video_thwc_u8: np.ndarray) -> list[Image.Image]:
    if video_thwc_u8.ndim != 4:
        raise ValueError(f"Expected [T,H,W,3] video array, got {tuple(video_thwc_u8.shape)}")
    return [Image.fromarray(frame) for frame in video_thwc_u8.astype(np.uint8)]


def _latent_norm_to_pil_frames(latent: torch.Tensor, *, upscale: int = 8) -> tuple[list[Image.Image], dict[str, float]]:
    if latent.ndim == 5:
        if latent.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for latent trace export, got {tuple(latent.shape)}")
        latent = latent[0]
    if latent.ndim != 4:
        raise ValueError(f"Expected latent tensor [C,T,H,W], got {tuple(latent.shape)}")
    norm_map = latent.detach().float().cpu().pow(2).mean(dim=0).sqrt()
    min_value = float(norm_map.min().item())
    max_value = float(norm_map.max().item())
    scale = max(max_value - min_value, 1e-6)
    frames: list[Image.Image] = []
    for frame_idx in range(norm_map.shape[0]):
        frame = ((norm_map[frame_idx] - min_value) / scale * 255.0).round().to(torch.uint8).numpy()
        image = Image.fromarray(frame, mode="L").resize(
            (frame.shape[1] * upscale, frame.shape[0] * upscale),
            Image.Resampling.NEAREST,
        )
        frames.append(image.convert("RGB"))
    return frames, {"latent_norm_min": min_value, "latent_norm_max": max_value}


def _pick_strip_indices(num_frames: int, max_frames: int) -> list[int]:
    if num_frames <= 0:
        return []
    if num_frames <= max_frames:
        return list(range(num_frames))
    positions = torch.linspace(0, num_frames - 1, steps=max_frames)
    return [int(round(value.item())) for value in positions]


def _save_frame_strip(
    frames: list[Image.Image],
    output_path: Path,
    *,
    max_frames: int = 8,
    tile_height: int = 160,
) -> None:
    if not frames:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = _pick_strip_indices(len(frames), max_frames=max_frames)
    selected = []
    for frame_idx in indices:
        frame = frames[frame_idx].convert("RGB")
        new_width = max(1, int(round(frame.width * (tile_height / max(frame.height, 1)))))
        selected.append((frame.resize((new_width, tile_height), Image.Resampling.BILINEAR), frame_idx))

    padding = 10
    label_height = 22
    width = padding + sum(frame.width + padding for frame, _ in selected)
    height = tile_height + label_height + 2 * padding
    canvas = Image.new("RGB", (width, height), color=(248, 244, 238))
    draw = ImageDraw.Draw(canvas)
    x = padding
    for frame, frame_idx in selected:
        canvas.paste(frame, (x, padding))
        draw.text((x, padding + tile_height + 4), f"t={frame_idx}", fill=(40, 40, 40))
        x += frame.width + padding
    canvas.save(output_path)


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _apply_context_anchored_guidance(
    *,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
    vae,
    context_frames_pixel: torch.Tensor,
    energy_obj: VJEPASurpriseEnergy,
    config: WanVJEPAConfig,
    predicted_future_ref: torch.Tensor,
    vae_temporal_stride: int,
    motion_mask_mode: str = "per_frame",
    trace_hook: Optional[Callable[..., None]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_for_grad = latent_xt.detach().float().requires_grad_(True)
    model_output = model_output.detach().float()
    preview_frame_stride = max(1, int(config.preview_frame_stride))

    x0_pred = predict_x0_from_flow_model_output(
        scheduler=scheduler,
        latent_xt=latent_for_grad,
        model_output=model_output,
        timestep=timestep,
    )
    full_video = decode_preview_video(
        vae=vae,
        x0_latent=x0_pred,
        preview_downsample_factor=int(config.preview_downsample_factor),
        preview_frame_stride=preview_frame_stride,
    )

    n_ctx = int(config.context_frames)
    preview_ctx_frames = _pixel_frames_to_preview_len(n_ctx, preview_frame_stride)
    if int(full_video.shape[2]) <= preview_ctx_frames:
        raise ValueError(
            "Decoded preview has "
            f"{int(full_video.shape[2])} frames, need > preview_context_frames={preview_ctx_frames}"
        )
    generated_future = full_video[:, :, preview_ctx_frames:]
    ctx = context_frames_pixel.to(device=generated_future.device, dtype=generated_future.dtype)
    clip = build_context_future_clip(
        context_btchw=ctx,
        future_btchw=generated_future,
        window_size=int(config.window_size),
        context_frames=n_ctx,
    )
    future_motion_mask_thw = _motion_mask_from_preview_video(
        full_video,
        mode=str(motion_mask_mode),
    )[:, preview_ctx_frames:].to(device=generated_future.device, dtype=generated_future.dtype)
    full_motion_mask_thw = _motion_mask_from_preview_video(
        full_video,
        mode=str(motion_mask_mode),
    ).to(device=full_video.device, dtype=full_video.dtype)
    future_energy_mask_thw = future_motion_mask_thw
    full_energy_mask_thw = full_motion_mask_thw
    spectral_future_weight_thw = None
    full_spectral_weight_thw = None
    if bool(getattr(config, "use_spectral_guidance", False)):
        spectral_source = str(getattr(config, "spectral_source", "temporal_lowpass_residual"))
        if spectral_source != "temporal_lowpass_residual":
            raise ValueError(f"Unsupported spectral_source: {spectral_source}")
        spectral_future_weight_thw = compute_temporal_lowpass_residual_map(
            full_video,
            future_start_idx=int(preview_ctx_frames),
            lowpass_ratio=float(getattr(config, "spectral_lowpass_ratio", 0.18)),
            normalize_percentile=float(getattr(config, "spectral_normalize_percentile", 95.0)),
        ).to(device=generated_future.device, dtype=generated_future.dtype)
        spectral_future_weight_thw = (
            float(getattr(config, "spectral_weight_floor", 0.25))
            + float(getattr(config, "spectral_weight_scale", 1.0)) * spectral_future_weight_thw
        )
        dilation = max(0, int(getattr(config, "spectral_mask_dilation", 0)))
        if dilation > 0:
            future_energy_mask_thw = dilate_mask_thw(future_motion_mask_thw, dilation).to(
                device=generated_future.device,
                dtype=generated_future.dtype,
            )
            full_energy_mask_thw = dilate_mask_thw(full_motion_mask_thw, dilation).to(
                device=full_video.device,
                dtype=full_video.dtype,
            )
        full_spectral_weight_thw = torch.zeros_like(full_energy_mask_thw)
        full_spectral_weight_thw[:, preview_ctx_frames:] = spectral_future_weight_thw.to(
            device=full_spectral_weight_thw.device,
            dtype=full_spectral_weight_thw.dtype,
        )
    energy = energy_obj.context_anchored(
        clip,
        window_size=int(config.window_size),
        context_frames=n_ctx,
        predicted_future_ref=predicted_future_ref,
        future_motion_mask_thw=future_energy_mask_thw,
        future_extra_weight_thw=spectral_future_weight_thw,
        motion_mask_mode=str(motion_mask_mode),
    )

    gradient = torch.autograd.grad(energy, latent_for_grad, retain_graph=False, create_graph=False)[0]
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    raw_grad_norm = float(gradient.norm().item())
    if config.max_grad_norm is not None and raw_grad_norm > float(config.max_grad_norm):
        gradient = gradient * (float(config.max_grad_norm) / max(raw_grad_norm, 1e-6))
    gradient = _normalize_gradient(
        gradient,
        mode=str(config.gradient_normalization),
    )

    if trace_hook is not None:
        trace_hook(
            x0_pred=x0_pred.detach(),
            preview_video=full_video.detach(),
            future_motion_mask_thw=future_energy_mask_thw.detach(),
            full_motion_mask_thw=full_energy_mask_thw.detach(),
            spectral_weight_thw=full_spectral_weight_thw.detach() if full_spectral_weight_thw is not None else None,
            energy=float(energy.detach().item()),
            raw_grad_norm=raw_grad_norm,
            normalized_grad_rms=float(gradient.detach().pow(2).mean().sqrt().item()),
            preview_context_frames=int(preview_ctx_frames),
            motion_mask_mode=str(motion_mask_mode),
        )

    n_ctx_latent = _pixel_frames_to_latent_len(n_ctx, vae_temporal_stride)
    if 0 < n_ctx_latent < gradient.shape[1]:
        gradient[:, :n_ctx_latent] = 0.0
    elif 0 < n_ctx_latent < gradient.shape[2]:
        gradient[:, :, :n_ctx_latent] = 0.0

    corrected_step_size = float(config.latent_step_size)
    correction = corrected_step_size * gradient
    latent_l2 = float(latent_xt.detach().float().norm().item())

    ratio_cap_applied = False
    ratio_cap_scale = 1.0
    if config.max_correction_ratio is not None and float(config.max_correction_ratio) > 0 and latent_l2 > 0:
        correction_l2_raw = float(correction.detach().norm().item())
        max_allowed_l2 = float(config.max_correction_ratio) * latent_l2
        if correction_l2_raw > max_allowed_l2:
            ratio_cap_scale = max_allowed_l2 / max(correction_l2_raw, 1e-6)
            correction = correction * ratio_cap_scale
            corrected_step_size *= ratio_cap_scale
            ratio_cap_applied = True

    artifact_guard_mode = str(getattr(config, "artifact_guard_mode", "none"))
    stay_close_max_video_l1 = getattr(config, "stay_close_max_video_l1", None)
    artifact_guard_applied = False
    artifact_guard_backoff_steps = 0
    artifact_guard_video_l1 = None
    if (
        artifact_guard_mode == "video_l1_backoff"
        and stay_close_max_video_l1 is not None
        and float(stay_close_max_video_l1) > 0
        and corrected_step_size > 0
    ):
        base_video = full_video.detach()
        trial_correction = correction.detach().clone()
        for _attempt in range(5):
            with torch.no_grad():
                trial_latent = latent_xt.detach().float() - trial_correction
                trial_x0 = predict_x0_from_flow_model_output(
                    scheduler=scheduler,
                    latent_xt=trial_latent,
                    model_output=model_output,
                    timestep=timestep,
                )
                trial_video = decode_preview_video(
                    vae=vae,
                    x0_latent=trial_x0,
                    preview_downsample_factor=int(config.preview_downsample_factor),
                    preview_frame_stride=preview_frame_stride,
                )
                artifact_guard_video_l1 = float((trial_video - base_video).abs().mean().item())
            if artifact_guard_video_l1 <= float(stay_close_max_video_l1):
                correction = trial_correction
                corrected_step_size = float(corrected_step_size) * (0.5 ** artifact_guard_backoff_steps)
                break
            trial_correction = trial_correction * 0.5
            artifact_guard_backoff_steps += 1
            artifact_guard_applied = True
        else:
            correction = trial_correction
            corrected_step_size = float(corrected_step_size) * (0.5 ** artifact_guard_backoff_steps)
            with torch.no_grad():
                trial_latent = latent_xt.detach().float() - correction
                trial_x0 = predict_x0_from_flow_model_output(
                    scheduler=scheduler,
                    latent_xt=trial_latent,
                    model_output=model_output,
                    timestep=timestep,
                )
                trial_video = decode_preview_video(
                    vae=vae,
                    x0_latent=trial_x0,
                    preview_downsample_factor=int(config.preview_downsample_factor),
                    preview_frame_stride=preview_frame_stride,
                )
                artifact_guard_video_l1 = float((trial_video - base_video).abs().mean().item())

    corrected = latent_xt.detach().float() - correction
    corrected = corrected.to(dtype=latent_xt.dtype)

    correction_l2 = float(correction.detach().norm().item())
    stats = {
        "energy": float(energy.detach().item()),
        "grad_rms": float(gradient.detach().pow(2).mean().sqrt().item()),
        "latent_rms": float(latent_xt.detach().float().pow(2).mean().sqrt().item()),
        "correction_l2": correction_l2,
        "latent_l2": latent_l2,
        "correction_ratio": correction_l2 / max(latent_l2, 1e-6),
        "preview_frames": float(full_video.shape[2]),
        "preview_height": float(full_video.shape[3]),
        "preview_width": float(full_video.shape[4]),
        "preview_context_frames": float(preview_ctx_frames),
        "motion_mask_coverage": float(future_motion_mask_thw.mean().item()),
        "step_size_used": float(corrected_step_size),
        "ratio_cap_applied": float(ratio_cap_applied),
        "ratio_cap_scale": float(ratio_cap_scale),
        "artifact_guard_applied": float(artifact_guard_applied),
        "artifact_guard_backoff_steps": float(artifact_guard_backoff_steps),
    }
    if artifact_guard_video_l1 is not None:
        stats["artifact_guard_video_l1"] = float(artifact_guard_video_l1)
    return corrected, stats


class WanTI2VContextAnchoredVJEPA(WanTI2V):
    def __init__(
        self,
        *args,
        vjepa_model_name: str = "vith",
        vjepa_checkpoint_path: Optional[str] = None,
        vjepa_device: Optional[str | torch.device] = None,
        vjepa_config: Optional[WanVJEPAConfig] = None,
        enable_vjepa_guidance: bool = False,
        vjepa_target_step_indices: Optional[list[int]] = None,
        motion_mask_mode: str = "per_frame",
        **kwargs,
    ):
        checkpoint_dir = kwargs.get("checkpoint_dir")
        convert_model_dtype = kwargs.get("convert_model_dtype", False)
        super().__init__(*args, **kwargs)
        self.vjepa_model_name = str(vjepa_model_name)
        self.vjepa_checkpoint_path = vjepa_checkpoint_path
        self.vjepa_device = torch.device(vjepa_device) if vjepa_device is not None else self.device
        self.vjepa_config = vjepa_config or WanVJEPAConfig()
        self.enable_vjepa_guidance = bool(enable_vjepa_guidance)
        self.vjepa_target_step_indices = [int(value) for value in (vjepa_target_step_indices or [])]
        self.motion_mask_mode = str(motion_mask_mode)
        self._vjepa_energy: Optional[VJEPASurpriseEnergy] = None
        self.last_vjepa_trace: list[dict[str, object]] = []
        self.last_vjepa_step_artifacts: list[dict[str, object]] = []
        self.anchor_mode = "repeated_first_frame"
        self.trace_intermediates_enabled = False
        self.trace_max_strip_frames = 8
        self.trace_case_dir: Path | None = None
        self.trace_case_payload: dict[str, Any] | None = None
        self.trace_fps = 16

        if any(parameter.is_meta for parameter in self.model.parameters()):
            if checkpoint_dir is None:
                raise ValueError("Wan model contains meta tensors, but checkpoint_dir is unavailable for reload.")
            logging.info("Reloading WanModel with low_cpu_mem_usage=False to materialize meta tensors.")
            reloaded_model = self.model.__class__.from_pretrained(
                checkpoint_dir,
                low_cpu_mem_usage=False,
            )
            reloaded_model.eval().requires_grad_(False)
            if convert_model_dtype:
                reloaded_model.to(self.param_dtype)
            self.model = reloaded_model

        vae_module = getattr(self.vae, "model", None)
        if vae_module is not None:
            for parameter in vae_module.parameters():
                parameter.requires_grad_(False)
            vae_module.eval()

    def configure_trace(self, *, enabled: bool, max_strip_frames: int, fps: int) -> None:
        self.trace_intermediates_enabled = bool(enabled)
        self.trace_max_strip_frames = max(2, int(max_strip_frames))
        self.trace_fps = max(1, int(fps))

    def set_trace_case(
        self,
        *,
        case_dir: Path | None,
        sample_id: str | None,
        prompt: str | None,
        output_video_path: Path | None,
        source_json: str | None,
        fps: int,
    ) -> None:
        self.trace_case_dir = case_dir if self.trace_intermediates_enabled else None
        self.trace_fps = int(fps)
        if self.trace_case_dir is None:
            self.trace_case_payload = None
            return
        self.trace_case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "sample_id": sample_id,
            "prompt": prompt,
            "source_json": source_json,
            "fps": int(fps),
        }
        if output_video_path is not None:
            payload["output_video_path"] = str(output_video_path)
            payload["final_video"] = "final_video.mp4"
        self.trace_case_payload = payload
        if output_video_path is not None and output_video_path.exists():
            _safe_symlink_or_copy(output_video_path, self.trace_case_dir / "final_video.mp4")
        (self.trace_case_dir / "case.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_trace_case_payload(self, extra: dict[str, Any]) -> None:
        if self.trace_case_dir is None:
            return
        payload = dict(self.trace_case_payload or {})
        payload.update(extra)
        self.trace_case_payload = payload
        (self.trace_case_dir / "case.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _trace_guidance_step(
        self,
        *,
        step_idx: int,
        timestep: int,
        x0_pred: torch.Tensor,
        preview_video: torch.Tensor,
        future_motion_mask_thw: torch.Tensor,
        full_motion_mask_thw: torch.Tensor,
        spectral_weight_thw: torch.Tensor | None,
        energy: float,
        raw_grad_norm: float,
        normalized_grad_rms: float,
        preview_context_frames: int,
        motion_mask_mode: str,
    ) -> None:
        if self.trace_case_dir is None:
            return
        step_dir = self.trace_case_dir / f"step_{step_idx:02d}_t{timestep:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        preview_u8 = _preview_video_to_u8(preview_video)
        future_mask = future_motion_mask_thw.detach().float().cpu().numpy()
        full_mask = full_motion_mask_thw.detach().float().cpu().numpy()
        if future_mask.ndim == 4 and future_mask.shape[0] == 1:
            future_mask = future_mask[0]
        if full_mask.ndim == 4 and full_mask.shape[0] == 1:
            full_mask = full_mask[0]

        motion_mask_u8 = render_binary_mask_video(full_mask)
        motion_overlay_u8 = render_motion_overlay_video(preview_u8, full_mask)
        background_overlay_u8 = render_background_overlay_video(preview_u8, full_mask)

        preview_frames = _video_thwc_u8_to_pil_frames(preview_u8)
        motion_mask_frames = _video_thwc_u8_to_pil_frames(motion_mask_u8)
        motion_overlay_frames = _video_thwc_u8_to_pil_frames(motion_overlay_u8)
        background_overlay_frames = _video_thwc_u8_to_pil_frames(background_overlay_u8)

        fps = max(1, self.trace_fps // max(1, self.vjepa_config.preview_frame_stride))
        write_mp4_h264(step_dir / "preview_video.mp4", preview_u8, fps=fps)
        write_mp4_h264(step_dir / "motion_mask_video.mp4", motion_mask_u8, fps=fps)
        write_mp4_h264(step_dir / "motion_overlay_video.mp4", motion_overlay_u8, fps=fps)
        write_mp4_h264(step_dir / "background_overlay_video.mp4", background_overlay_u8, fps=fps)
        spectral_mean = None
        if spectral_weight_thw is not None:
            spectral_u8 = _weight_map_to_u8(spectral_weight_thw)
            spectral_frames = _video_thwc_u8_to_pil_frames(spectral_u8)
            write_mp4_h264(step_dir / "spectral_weight_video.mp4", spectral_u8, fps=fps)
            _save_frame_strip(
                spectral_frames,
                step_dir / "spectral_weight_strip.png",
                max_frames=self.trace_max_strip_frames,
                tile_height=140,
            )
            spectral_mean = float(spectral_weight_thw.detach().float().mean().item())

        _save_frame_strip(preview_frames, step_dir / "preview_strip.png", max_frames=self.trace_max_strip_frames, tile_height=140)
        _save_frame_strip(motion_mask_frames, step_dir / "motion_mask_strip.png", max_frames=self.trace_max_strip_frames, tile_height=140)
        _save_frame_strip(motion_overlay_frames, step_dir / "motion_overlay_strip.png", max_frames=self.trace_max_strip_frames, tile_height=140)
        _save_frame_strip(background_overlay_frames, step_dir / "background_overlay_strip.png", max_frames=self.trace_max_strip_frames, tile_height=140)

        latent_frames, latent_stats = _latent_norm_to_pil_frames(x0_pred)
        latent_u8 = np.stack([np.asarray(frame.convert("RGB")) for frame in latent_frames], axis=0)
        write_mp4_h264(step_dir / "x0_latent_norm.mp4", latent_u8, fps=fps)
        _save_frame_strip(
            latent_frames,
            step_dir / "x0_latent_norm_strip.png",
            max_frames=self.trace_max_strip_frames,
            tile_height=140,
        )

        stats = {
            "step_idx": int(step_idx),
            "timestep": int(timestep),
            "energy": float(energy),
            "raw_grad_norm": float(raw_grad_norm),
            "normalized_grad_rms": float(normalized_grad_rms),
            "preview_frames": len(preview_frames),
            "preview_height": preview_frames[0].height if preview_frames else 0,
            "preview_width": preview_frames[0].width if preview_frames else 0,
            "preview_context_frames": int(preview_context_frames),
            "motion_mask_mode": str(motion_mask_mode),
            "motion_mask_coverage_full": float(full_mask.mean()),
            "motion_mask_coverage_future": float(future_mask.mean()),
            "spectral_weight_mean": spectral_mean,
            **latent_stats,
        }
        (step_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._write_trace_case_payload(
            {
                "selected_guidance_steps": list(self.vjepa_target_step_indices or []),
                "trace_enabled": True,
            }
        )

    def _ensure_vjepa_energy(self) -> VJEPASurpriseEnergy:
        if self._vjepa_energy is None:
            self._vjepa_energy = VJEPASurpriseEnergy(
                model_name=self.vjepa_model_name,
                device=self.vjepa_device,
                local_torchhub=True,
                checkpoint_path=self.vjepa_checkpoint_path,
            )
        return self._vjepa_energy

    def _prepare_sampling_context(self, input_prompt, n_prompt, seed, offload_model):
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [tensor.to(self.device) for tensor in context]
            context_null = [tensor.to(self.device) for tensor in context_null]

        return context, context_null, seed_g

    def _create_scheduler(self, sample_solver, sampling_steps, shift):
        if sample_solver == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = scheduler.timesteps
        elif sample_solver == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(
                scheduler,
                device=self.device,
                sigmas=sampling_sigmas,
            )
        else:
            raise NotImplementedError(f"Unsupported solver: {sample_solver}")
        return scheduler, timesteps

    def _decode_latents(self, zs):
        if self.rank == 0:
            return self.vae.decode(zs)
        return None

    def _select_guidance_steps(self, timesteps) -> tuple[set[int], dict[int, int]]:
        if self.vjepa_target_step_indices:
            selected = {
                max(0, min(int(step_idx), len(timesteps) - 1))
                for step_idx in self.vjepa_target_step_indices
            }
        else:
            selected = set(
                pick_guidance_step_indices(
                    total_steps=len(timesteps),
                    count=int(self.vjepa_config.guidance_steps),
                    min_step_percent=float(self.vjepa_config.min_step_percent),
                    max_step_percent=float(self.vjepa_config.max_step_percent),
                )
            )
        mapping = {
            int(step_idx): int(round(float(timesteps[step_idx].detach().cpu().item())))
            for step_idx in sorted(selected)
        }
        return selected, mapping

    def generate_vjepa(
        self,
        input_prompt: str,
        img: Optional[Image.Image],
        size=(1280, 704),
        max_area=704 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        self.last_vjepa_trace = []
        self.last_vjepa_step_artifacts = []
        if img is None:
            raise ValueError("Wan TI2V guidance requires an input image.")
        if not self.enable_vjepa_guidance or int(self.vjepa_config.guidance_steps) <= 0:
            return super().generate(
                input_prompt=input_prompt,
                img=img,
                size=size,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model,
            )

        final_latents = self._sample_i2v_latents(
            input_prompt=input_prompt,
            img=img,
            max_area=max_area,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
        )
        videos = self._decode_latents([final_latents])
        return videos[0]

    def _sample_i2v_latents(
        self,
        input_prompt,
        img,
        max_area,
        frame_num,
        shift,
        sample_solver,
        sampling_steps,
        guide_scale,
        n_prompt,
        seed,
        offload_model,
    ):
        input_height, input_width = img.height, img.width
        down_h = self.patch_size[1] * self.vae_stride[1]
        down_w = self.patch_size[2] * self.vae_stride[2]
        output_width, output_height = best_output_size(input_width, input_height, down_w, down_h, max_area)

        scale = max(output_width / input_width, output_height / input_height)
        img = img.resize((round(input_width * scale), round(input_height * scale)), Image.LANCZOS)
        x1 = (img.width - output_width) // 2
        y1 = (img.height - output_height) // 2
        img = img.crop((x1, y1, x1 + output_width, y1 + output_height))
        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        frames = frame_num
        seq_len = ((frames - 1) // self.vae_stride[0] + 1) * (
            output_height // self.vae_stride[1]
        ) * (output_width // self.vae_stride[2]) // (self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        context, context_null, seed_g = self._prepare_sampling_context(
            input_prompt=input_prompt,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model,
        )

        noise = torch.randn(
            self.vae.model.z_dim,
            (frames - 1) // self.vae_stride[0] + 1,
            output_height // self.vae_stride[1],
            output_width // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device,
        )
        fixed_latent = self.vae.encode([img_tensor])[0]
        _, mask2 = masks_like([noise], zero=True)
        latent = (1.0 - mask2[0]) * fixed_latent + mask2[0] * noise

        arg_c = {"context": [context[0]], "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        scheduler, timesteps = self._create_scheduler(
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            shift=shift,
        )
        selected_steps, timestep_map = self._select_guidance_steps(timesteps)
        energy_fn = self._ensure_vjepa_energy()
        n_ctx = int(self.vjepa_config.context_frames)
        future_frames = int(self.vjepa_config.window_size) - n_ctx
        if future_frames <= 0:
            raise ValueError(
                f"window_size={self.vjepa_config.window_size} must exceed context_frames={n_ctx}"
            )
        context_frames_pixel = img_tensor.repeat(1, n_ctx, 1, 1).unsqueeze(0).detach()
        placeholder = torch.zeros(
            1,
            3,
            future_frames,
            output_height,
            output_width,
            device=context_frames_pixel.device,
            dtype=context_frames_pixel.dtype,
        )
        precompute_clip = build_context_future_clip(
            context_btchw=context_frames_pixel,
            future_btchw=placeholder,
            window_size=int(self.vjepa_config.window_size),
            context_frames=n_ctx,
        )
        predicted_future_ref = energy_fn.precompute_future_prediction(
            precompute_clip,
            window_size=int(self.vjepa_config.window_size),
            context_frames=n_ctx,
        )

        logging.info(
            "V-JEPA guidance enabled: mode=context_anchored steps=%s timesteps=%s anchor=%s",
            sorted(int(step) for step in selected_steps),
            {str(step): int(timestep) for step, timestep in sorted(timestep_map.items())},
            self.anchor_mode,
        )

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)
        inner_k = max(1, int(getattr(self.vjepa_config, "inner_k", 1)))
        recompute_noise_pred_after_guidance = bool(
            getattr(self.vjepa_config, "recompute_noise_pred_after_guidance", False)
        )

        def _predict_cfg_noise(current_latent: torch.Tensor, timestep_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            latent_model_input = [current_latent.to(self.device)]
            timestep = torch.stack([timestep_value]).to(self.device)

            temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
            temp_ts = torch.cat(
                [temp_ts, temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep]
            )
            timestep = temp_ts.unsqueeze(0)

            noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
            if offload_model:
                torch.cuda.empty_cache()
            noise_pred_uncond = self.model(
                latent_model_input,
                t=timestep,
                **arg_null,
            )[0]
            if offload_model:
                torch.cuda.empty_cache()

            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
            return noise_pred, timestep

        with torch.amp.autocast("cuda", dtype=self.param_dtype), no_sync():
            for step_index, timestep_value in enumerate(tqdm(timesteps)):
                if (offload_model or self.init_on_cpu) and next(self.model.parameters()).device.type != "cuda":
                    self.model.to(self.device)
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    noise_pred, timestep = _predict_cfg_noise(latent, timestep_value)

                latent_xt = latent
                if step_index in selected_steps:
                    if next(self.model.parameters()).device.type == "cuda":
                        self.model.cpu()
                        torch.cuda.empty_cache()
                    for inner_index in range(inner_k):
                        with torch.enable_grad():
                            latent_xt, stats = _apply_context_anchored_guidance(
                                latent_xt=latent_xt,
                                model_output=noise_pred,
                                timestep=timestep_value,
                                scheduler=scheduler,
                                vae=self.vae,
                                context_frames_pixel=context_frames_pixel,
                                energy_obj=energy_fn,
                                config=self.vjepa_config,
                                predicted_future_ref=predicted_future_ref,
                                vae_temporal_stride=int(self.vae_stride[0]),
                                motion_mask_mode=self.motion_mask_mode,
                                trace_hook=(
                                    lambda **kwargs: self._trace_guidance_step(
                                        step_idx=step_index,
                                        timestep=int(timestep_value.detach().cpu().item()),
                                        **kwargs,
                                    )
                                    if self.trace_intermediates_enabled and self.trace_case_dir is not None
                                    else None
                                ),
                            )
                        self.last_vjepa_step_artifacts.append(
                            {
                                "step_index": int(step_index),
                                "inner_index": int(inner_index),
                                "timestep": int(timestep_value.detach().cpu().item()),
                                "preview_video": None,
                                "future_motion_mask": None,
                            }
                        )
                        trace_row = {
                            "step_index": int(step_index),
                            "inner_index": int(inner_index),
                            "inner_k": int(inner_k),
                            "timestep": int(timestep_value.detach().cpu().item()),
                            "energy": float(stats["energy"]),
                            "grad_rms": float(stats["grad_rms"]),
                            "correction_l2": float(stats["correction_l2"]),
                            "latent_l2": float(stats["latent_l2"]),
                            "correction_ratio": float(stats["correction_ratio"]),
                            "step_size_used": float(stats["step_size_used"]),
                            "ratio_cap_applied": bool(stats["ratio_cap_applied"]),
                            "ratio_cap_scale": float(stats["ratio_cap_scale"]),
                            "artifact_guard_applied": bool(stats.get("artifact_guard_applied", 0.0)),
                            "artifact_guard_backoff_steps": int(stats.get("artifact_guard_backoff_steps", 0.0)),
                            "artifact_guard_video_l1": (
                                float(stats["artifact_guard_video_l1"])
                                if "artifact_guard_video_l1" in stats
                                else None
                            ),
                            "preview_frames": int(stats["preview_frames"]),
                            "preview_context_frames": int(stats["preview_context_frames"]),
                            "preview_height": int(stats["preview_height"]),
                            "preview_width": int(stats["preview_width"]),
                            "preview_frame_stride": int(self.vjepa_config.preview_frame_stride),
                            "motion_mask_mode": str(self.motion_mask_mode),
                            "motion_mask_coverage": float(stats.get("motion_mask_coverage", 0.0)),
                            "recompute_noise_pred_after_guidance": bool(recompute_noise_pred_after_guidance),
                        }
                        self.last_vjepa_trace.append(trace_row)
                        logging.info(
                            "V-JEPA[context_anchored] step=%d inner=%d/%d timestep=%d energy=%.6f grad_rms=%.6f "
                            "corr_l2=%.4f latent_l2=%.1f corr_ratio=%.5f step_used=%.4f ratio_cap=%s "
                            "guard=%s guard_l1=%s preview=%dx%dx%d stride=%d recompute_noise=%s",
                            trace_row["step_index"],
                            trace_row["inner_index"] + 1,
                            trace_row["inner_k"],
                            trace_row["timestep"],
                            trace_row["energy"],
                            trace_row["grad_rms"],
                            trace_row["correction_l2"],
                            trace_row["latent_l2"],
                            trace_row["correction_ratio"],
                            trace_row["step_size_used"],
                            trace_row["ratio_cap_applied"],
                            trace_row["artifact_guard_applied"],
                            (
                                f"{trace_row['artifact_guard_video_l1']:.6f}"
                                if trace_row["artifact_guard_video_l1"] is not None
                                else "NA"
                            ),
                            trace_row["preview_frames"],
                            trace_row["preview_height"],
                            trace_row["preview_width"],
                            trace_row["preview_frame_stride"],
                            trace_row["motion_mask_mode"],
                            trace_row["motion_mask_coverage"],
                            trace_row["recompute_noise_pred_after_guidance"],
                        )
                    if recompute_noise_pred_after_guidance:
                        if next(self.model.parameters()).device.type != "cuda":
                            self.model.to(self.device)
                            torch.cuda.empty_cache()
                        with torch.no_grad():
                            noise_pred, timestep = _predict_cfg_noise(latent_xt, timestep_value)

                with torch.no_grad():
                    temp_x0 = scheduler.step(
                        noise_pred.unsqueeze(0),
                        timestep_value,
                        latent_xt.unsqueeze(0),
                        return_dict=False,
                        generator=seed_g,
                    )[0]
                    latent = temp_x0.squeeze(0)
                    latent = (1.0 - mask2[0]) * fixed_latent + mask2[0] * latent

            if offload_model:
                self.model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

        del scheduler
        if offload_model:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return latent


class OfficialWanTI2VVJEPAWrapper:
    def __init__(
        self,
        *,
        model: WanTI2VContextAnchoredVJEPA,
        resolved_wan_root: Path,
        max_area: int,
        guidance_enabled: bool,
        vjepa_preset: str,
        motion_mask_mode: str,
    ) -> None:
        self.model = model
        self.resolved_wan_root = resolved_wan_root
        self.max_area = int(max_area)
        self.guidance_enabled = bool(guidance_enabled)
        self.vjepa_preset = str(vjepa_preset)
        self.motion_mask_mode = str(motion_mask_mode)
        self.last_vjepa_trace: list[dict[str, object]] = []
        self.anchor_mode = getattr(model, "anchor_mode", "repeated_first_frame")

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        input_image: Image.Image,
        height: int,
        width: int,
        num_frames: int,
        cfg_scale: float,
        num_inference_steps: int,
        sample_shift: float,
        sample_solver: str,
        offload_model: bool,
    ) -> np.ndarray:
        solver = normalize_sample_solver(sample_solver)
        video = self.model.generate_vjepa(
            input_prompt=prompt,
            img=input_image,
            size=(int(width), int(height)),
            max_area=self.max_area,
            frame_num=int(num_frames),
            shift=float(sample_shift),
            sample_solver=solver,
            sampling_steps=int(num_inference_steps),
            guide_scale=float(cfg_scale),
            n_prompt=str(negative_prompt),
            seed=int(seed),
            offload_model=bool(offload_model),
        )
        self.last_vjepa_trace = list(getattr(self.model, "last_vjepa_trace", []))
        return convert_official_video_to_thwc(video)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run official Wan2.2 TI2V on input jsons listed in a txt file, with optional V-JEPA guidance."
    )
    parser.add_argument(
        "--input-list",
        default="/data/gaoya/AAA_test_video/0623/testjsons/test_100.txt",
        help="Text file containing one input json path per line.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output directory for mp4/json files.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--wan-root",
        default=str(DEFAULT_OFFICIAL_WAN_ROOT),
        help="Official Wan2.2 TI2V checkpoint directory.",
    )
    parser.add_argument("--backend", default="official", choices=["official", "legacy"])
    parser.add_argument("--size", default="704*1280", choices=["704*1280", "1280*704"])
    parser.add_argument("--frame-num", type=int, default=25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Override the default negative prompt. Use '' for empty.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--convert-model-dtype", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--vjepa-preset", default="baseline")
    parser.add_argument(
        "--disable-vjepa-guidance",
        action="store_true",
        help="Force-disable V-JEPA guidance even when a guided preset is selected.",
    )
    parser.add_argument("--vjepa-ckpt", type=Path, default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--vjepa-device-id", type=int, default=None)
    parser.add_argument("--vjepa-model", default=None, choices=["vith", "vitg", "vitg384"])
    parser.add_argument("--vjepa-guidance-mode", default=None, choices=["surprise", "context_anchored"])
    parser.add_argument("--vjepa-guidance-steps", type=int, default=None)
    parser.add_argument("--vjepa-min-step-percent", type=float, default=None)
    parser.add_argument("--vjepa-max-step-percent", type=float, default=None)
    parser.add_argument("--vjepa-target-step-indices", type=int, nargs="*", default=None)
    parser.add_argument("--vjepa-latent-step-size", type=float, default=None)
    parser.add_argument("--vjepa-inner-k", type=int, default=None)
    parser.add_argument("--vjepa-backtracking", action="store_true")
    parser.add_argument("--vjepa-preview-downsample-factor", type=int, default=None)
    parser.add_argument("--vjepa-preview-frame-stride", type=int, default=None)
    parser.add_argument("--vjepa-window-size", type=int, default=None)
    parser.add_argument("--vjepa-context-frames", type=int, default=None)
    parser.add_argument("--vjepa-stride", type=int, default=None)
    parser.add_argument("--vjepa-reduction", default=None, choices=["mean", "max"])
    parser.add_argument("--vjepa-grad-norm-mode", default=None, choices=["none", "rms", "l2"])
    parser.add_argument("--vjepa-max-grad-norm", type=float, default=None)
    parser.add_argument("--vjepa-max-correction-ratio", type=float, default=None)
    parser.add_argument("--vjepa-stay-close-max-video-l1", type=float, default=None)
    parser.add_argument("--vjepa-artifact-guard-mode", default=None, choices=["none", "video_l1_backoff"])
    parser.add_argument("--vjepa-recompute-noise-pred-after-guidance", action="store_true")
    parser.add_argument("--vjepa-use-spectral-guidance", action="store_true")
    parser.add_argument(
        "--vjepa-spectral-source",
        default="temporal_lowpass_residual",
        choices=["temporal_lowpass_residual"],
    )
    parser.add_argument("--vjepa-spectral-lowpass-ratio", type=float, default=0.18)
    parser.add_argument("--vjepa-spectral-normalize-percentile", type=float, default=95.0)
    parser.add_argument("--vjepa-spectral-weight-floor", type=float, default=0.25)
    parser.add_argument("--vjepa-spectral-weight-scale", type=float, default=1.0)
    parser.add_argument("--vjepa-spectral-mask-dilation", type=int, default=0)
    parser.add_argument(
        "--motion-mask-mode",
        default="temporal_union_except_first",
        choices=["per_frame", "temporal_union", "temporal_union_except_first"],
    )
    parser.add_argument("--trace-intermediates", action="store_true")
    parser.add_argument("--trace-root", type=Path, default=Path("/data/gaoya/agent-data/outputs/vjepa_guidance_trace"))
    parser.add_argument("--trace-max-strip-frames", type=int, default=8)
    parser.add_argument("--trace-build-html", action="store_true")
    parser.add_argument("--trace-output-html", type=Path, default=None)
    return parser.parse_args()


def _resolve_vjepa_settings(cli_args: argparse.Namespace) -> tuple[argparse.Namespace, WanVJEPAConfig]:
    override_fields = {
        "vjepa_model": cli_args.vjepa_model,
        "vjepa_guidance_mode": cli_args.vjepa_guidance_mode,
        "vjepa_guidance_steps": cli_args.vjepa_guidance_steps,
        "vjepa_min_step_percent": cli_args.vjepa_min_step_percent,
        "vjepa_max_step_percent": cli_args.vjepa_max_step_percent,
        "vjepa_target_step_indices": list(cli_args.vjepa_target_step_indices) if cli_args.vjepa_target_step_indices is not None else None,
        "vjepa_latent_step_size": cli_args.vjepa_latent_step_size,
        "vjepa_inner_k": cli_args.vjepa_inner_k,
        "vjepa_backtracking": bool(cli_args.vjepa_backtracking),
        "vjepa_preview_downsample_factor": cli_args.vjepa_preview_downsample_factor,
        "vjepa_preview_frame_stride": cli_args.vjepa_preview_frame_stride,
        "vjepa_window_size": cli_args.vjepa_window_size,
        "vjepa_context_frames": cli_args.vjepa_context_frames,
        "vjepa_stride": cli_args.vjepa_stride,
        "vjepa_reduction": cli_args.vjepa_reduction,
        "vjepa_grad_norm_mode": cli_args.vjepa_grad_norm_mode,
        "vjepa_max_grad_norm": cli_args.vjepa_max_grad_norm,
        "vjepa_max_correction_ratio": cli_args.vjepa_max_correction_ratio,
        "vjepa_stay_close_max_video_l1": cli_args.vjepa_stay_close_max_video_l1,
        "vjepa_artifact_guard_mode": cli_args.vjepa_artifact_guard_mode,
        "vjepa_recompute_noise_pred_after_guidance": bool(cli_args.vjepa_recompute_noise_pred_after_guidance),
        "motion_mask_mode": cli_args.motion_mask_mode,
    }
    apply_train0705_preset(cli_args, cli_args.vjepa_preset)

    if cli_args.disable_vjepa_guidance:
        cli_args.enable_vjepa_guidance = False
    if not bool(cli_args.enable_vjepa_guidance):
        cli_args.vjepa_guidance_steps = 0
    if override_fields["vjepa_model"] is not None:
        cli_args.vjepa_model = str(override_fields["vjepa_model"])
    if override_fields["vjepa_guidance_mode"] is not None:
        cli_args.vjepa_guidance_mode = str(override_fields["vjepa_guidance_mode"])
    if override_fields["vjepa_guidance_steps"] is not None:
        cli_args.vjepa_guidance_steps = int(override_fields["vjepa_guidance_steps"])
    if override_fields["vjepa_min_step_percent"] is not None:
        cli_args.vjepa_min_step_percent = float(override_fields["vjepa_min_step_percent"])
    if override_fields["vjepa_max_step_percent"] is not None:
        cli_args.vjepa_max_step_percent = float(override_fields["vjepa_max_step_percent"])
    if override_fields["vjepa_target_step_indices"] is not None:
        cli_args.vjepa_target_step_indices = [int(value) for value in override_fields["vjepa_target_step_indices"]]
    if override_fields["vjepa_latent_step_size"] is not None:
        cli_args.vjepa_latent_step_size = float(override_fields["vjepa_latent_step_size"])
    if override_fields["vjepa_inner_k"] is not None:
        cli_args.vjepa_inner_k = int(override_fields["vjepa_inner_k"])
    if override_fields["vjepa_backtracking"]:
        cli_args.vjepa_backtracking = True
    if override_fields["vjepa_preview_downsample_factor"] is not None:
        cli_args.vjepa_preview_downsample_factor = int(override_fields["vjepa_preview_downsample_factor"])
    if override_fields["vjepa_preview_frame_stride"] is not None:
        cli_args.vjepa_preview_frame_stride = int(override_fields["vjepa_preview_frame_stride"])
    if override_fields["vjepa_window_size"] is not None:
        cli_args.vjepa_window_size = int(override_fields["vjepa_window_size"])
    if override_fields["vjepa_context_frames"] is not None:
        cli_args.vjepa_context_frames = int(override_fields["vjepa_context_frames"])
    if override_fields["vjepa_stride"] is not None:
        cli_args.vjepa_stride = int(override_fields["vjepa_stride"])
    if override_fields["vjepa_reduction"] is not None:
        cli_args.vjepa_reduction = str(override_fields["vjepa_reduction"])
    if override_fields["vjepa_grad_norm_mode"] is not None:
        cli_args.vjepa_grad_norm_mode = str(override_fields["vjepa_grad_norm_mode"])
    if override_fields["vjepa_max_grad_norm"] is not None:
        cli_args.vjepa_max_grad_norm = float(override_fields["vjepa_max_grad_norm"])
    if override_fields["vjepa_max_correction_ratio"] is not None:
        cli_args.vjepa_max_correction_ratio = float(override_fields["vjepa_max_correction_ratio"])
    if override_fields["vjepa_stay_close_max_video_l1"] is not None:
        cli_args.vjepa_stay_close_max_video_l1 = float(override_fields["vjepa_stay_close_max_video_l1"])
    if override_fields["vjepa_artifact_guard_mode"] is not None:
        cli_args.vjepa_artifact_guard_mode = str(override_fields["vjepa_artifact_guard_mode"])
    if override_fields["vjepa_recompute_noise_pred_after_guidance"]:
        cli_args.vjepa_recompute_noise_pred_after_guidance = True

    if cli_args.vjepa_backtracking:
        raise ValueError(
            "Backtracking is not supported in wanti2v.py after the minimal-fix rewrite. "
            "Use fixed-step presets instead."
        )
    if bool(cli_args.enable_vjepa_guidance) and str(cli_args.vjepa_guidance_mode) != "context_anchored":
        raise ValueError(
            "wanti2v.py only supports V-JEPA guidance_mode=context_anchored in the current path."
        )

    stay_close_max_video_l1 = cli_args.vjepa_stay_close_max_video_l1
    if stay_close_max_video_l1 is not None and float(stay_close_max_video_l1) <= 0:
        stay_close_max_video_l1 = None
    max_correction_ratio = cli_args.vjepa_max_correction_ratio
    if max_correction_ratio is not None and float(max_correction_ratio) <= 0:
        max_correction_ratio = None

    vjepa_config = WanVJEPAConfig(
        guidance_steps=int(cli_args.vjepa_guidance_steps),
        min_step_percent=float(cli_args.vjepa_min_step_percent),
        max_step_percent=float(cli_args.vjepa_max_step_percent),
        latent_step_size=float(cli_args.vjepa_latent_step_size),
        inner_k=max(1, int(cli_args.vjepa_inner_k)),
        backtracking=bool(cli_args.vjepa_backtracking),
        preview_downsample_factor=int(cli_args.vjepa_preview_downsample_factor),
        preview_frame_stride=int(cli_args.vjepa_preview_frame_stride),
        window_size=int(cli_args.vjepa_window_size),
        context_frames=int(cli_args.vjepa_context_frames),
        stride=int(cli_args.vjepa_stride),
        reduction=str(cli_args.vjepa_reduction),
        gradient_normalization=str(cli_args.vjepa_grad_norm_mode),
        max_grad_norm=float(cli_args.vjepa_max_grad_norm) if cli_args.vjepa_max_grad_norm is not None else None,
        max_correction_ratio=float(max_correction_ratio) if max_correction_ratio is not None else None,
        stay_close_max_video_l1=(
            float(stay_close_max_video_l1)
            if stay_close_max_video_l1 is not None
            else None
        ),
        artifact_guard_mode=str(cli_args.vjepa_artifact_guard_mode),
        guidance_mode=str(cli_args.vjepa_guidance_mode),
        recompute_noise_pred_after_guidance=bool(cli_args.vjepa_recompute_noise_pred_after_guidance),
        use_spectral_guidance=bool(cli_args.vjepa_use_spectral_guidance),
        spectral_source=str(cli_args.vjepa_spectral_source),
        spectral_lowpass_ratio=float(cli_args.vjepa_spectral_lowpass_ratio),
        spectral_normalize_percentile=float(cli_args.vjepa_spectral_normalize_percentile),
        spectral_weight_floor=float(cli_args.vjepa_spectral_weight_floor),
        spectral_weight_scale=float(cli_args.vjepa_spectral_weight_scale),
        spectral_mask_dilation=max(0, int(cli_args.vjepa_spectral_mask_dilation)),
    )
    return cli_args, vjepa_config


def build_vjepa_aware_pipeline(args: WanTI2VArgs, cli_args: argparse.Namespace, vjepa_config: WanVJEPAConfig):
    if args.backend == "legacy":
        raise ValueError("V-JEPA TI2V guidance is only wired for backend=official.")

    resolved_wan_root = resolve_official_wan_root(args.wan_root)
    patch_wanmodel_from_pretrained_defaults()
    cfg = WAN_CONFIGS["ti2v-5B"]
    max_area = int(MAX_AREA_CONFIGS[args.size])
    if cli_args.vjepa_device_id is None:
        vjepa_device = None
    elif int(cli_args.vjepa_device_id) < 0:
        vjepa_device = "cpu"
    else:
        vjepa_device = f"cuda:{int(cli_args.vjepa_device_id)}"

    model = WanTI2VContextAnchoredVJEPA(
        config=cfg,
        checkpoint_dir=str(resolved_wan_root),
        device_id=int(cli_args.device_id),
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=bool(args.t5_cpu),
        convert_model_dtype=bool(args.convert_model_dtype),
        vjepa_model_name=str(cli_args.vjepa_model),
        vjepa_checkpoint_path=str(cli_args.vjepa_ckpt) if cli_args.vjepa_ckpt is not None else None,
        vjepa_device=vjepa_device,
        vjepa_config=vjepa_config,
        enable_vjepa_guidance=bool(cli_args.enable_vjepa_guidance),
        vjepa_target_step_indices=list(cli_args.vjepa_target_step_indices or []),
        motion_mask_mode=str(cli_args.motion_mask_mode),
    )
    model.configure_trace(
        enabled=bool(cli_args.trace_intermediates),
        max_strip_frames=int(cli_args.trace_max_strip_frames),
        fps=int(cli_args.fps),
    )
    return OfficialWanTI2VVJEPAWrapper(
        model=model,
        resolved_wan_root=resolved_wan_root,
        max_area=max_area,
        guidance_enabled=bool(cli_args.enable_vjepa_guidance),
        vjepa_preset=str(cli_args.vjepa_preset),
        motion_mask_mode=str(cli_args.motion_mask_mode),
    )


def _run_pipe_once(
    *,
    pipe,
    prompt: str,
    negative_prompt: str,
    seed: int,
    input_image: Image.Image,
    height: int,
    width: int,
    num_frames: int,
    cfg_scale: float,
    num_inference_steps: int,
    sample_shift: float,
    sample_solver: str,
    offload_model: bool,
) -> np.ndarray:
    with torch.no_grad():
        return pipe(
            prompt=prompt,
            negative_prompt=str(negative_prompt),
            seed=int(seed),
            input_image=input_image,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            cfg_scale=float(cfg_scale),
            num_inference_steps=int(num_inference_steps),
            sample_shift=float(sample_shift),
            sample_solver=normalize_sample_solver(sample_solver),
            offload_model=bool(offload_model),
        )


def run_single_case_vjepa(
    *,
    pipe,
    args: WanTI2VArgs,
    cli_args: argparse.Namespace,
    input_json_path: Path,
    payload: dict[str, Any],
    firstframe_path: Path,
    output_video: Path,
    trace_case_dir: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    input_caption = ensure_str_field(payload, "input_caption", input_json_path)
    height_str, width_str = args.size.split("*", maxsplit=1)
    height, width = int(height_str), int(width_str)

    image = Image.open(firstframe_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    logs = [
        f"[case] input_json={input_json_path}",
        f"[case] input_image={firstframe_path}",
        f"[case] input_caption={input_caption}",
        f"[case] wan_root={args.wan_root}",
        f"[case] resolved_wan_root={pipe.resolved_wan_root}",
        f"[case] backend={args.backend}",
        f"[case] sample_solver={normalize_sample_solver(args.sample_solver)}",
        f"[case] negative_prompt={args.negative_prompt}",
        f"[case] vjepa_preset={cli_args.vjepa_preset}",
        f"[case] enable_vjepa_guidance={bool(cli_args.enable_vjepa_guidance)}",
        f"[case] motion_mask_mode={cli_args.motion_mask_mode}",
        f"[case] vjepa_inner_k={int(cli_args.vjepa_inner_k)}",
        f"[case] vjepa_preview_frame_stride={int(cli_args.vjepa_preview_frame_stride)}",
        f"[case] vjepa_artifact_guard_mode={cli_args.vjepa_artifact_guard_mode}",
        f"[case] vjepa_stay_close_max_video_l1={cli_args.vjepa_stay_close_max_video_l1}",
        f"[case] vjepa_recompute_noise_pred_after_guidance={bool(cli_args.vjepa_recompute_noise_pred_after_guidance)}",
        f"[case] vjepa_use_spectral_guidance={bool(cli_args.vjepa_use_spectral_guidance)}",
        f"[case] vjepa_spectral_source={cli_args.vjepa_spectral_source}",
        f"[case] vjepa_spectral_lowpass_ratio={float(cli_args.vjepa_spectral_lowpass_ratio)}",
        f"[case] vjepa_spectral_weight_floor={float(cli_args.vjepa_spectral_weight_floor)}",
        f"[case] vjepa_spectral_weight_scale={float(cli_args.vjepa_spectral_weight_scale)}",
        f"[case] vjepa_spectral_mask_dilation={int(cli_args.vjepa_spectral_mask_dilation)}",
    ]

    if trace_case_dir is not None and hasattr(pipe, "model") and hasattr(pipe.model, "set_trace_case"):
        pipe.model.set_trace_case(
            case_dir=trace_case_dir,
            sample_id=input_json_path.stem,
            prompt=input_caption,
            output_video_path=output_video,
            source_json=str(input_json_path),
            fps=int(args.fps),
        )

    used_offload = bool(args.offload_model)
    try:
        video = _run_pipe_once(
            pipe=pipe,
            prompt=input_caption,
            negative_prompt=str(args.negative_prompt),
            seed=int(args.seed),
            input_image=image,
            height=int(height),
            width=int(width),
            num_frames=int(args.frame_num),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.sampling_steps),
            sample_shift=float(args.sample_shift),
            sample_solver=str(args.sample_solver),
            offload_model=used_offload,
        )
    except RuntimeError as exc:
        if used_offload or "out of memory" not in str(exc).lower():
            raise
        logs.append("[case] retry=oom_with_offload_model")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        video = _run_pipe_once(
            pipe=pipe,
            prompt=input_caption,
            negative_prompt=str(args.negative_prompt),
            seed=int(args.seed),
            input_image=image,
            height=int(height),
            width=int(width),
            num_frames=int(args.frame_num),
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.sampling_steps),
            sample_shift=float(args.sample_shift),
            sample_solver=str(args.sample_solver),
            offload_model=True,
        )
        used_offload = True

    save_video_np(video, output_video, fps=int(args.fps))
    if trace_case_dir is not None and hasattr(pipe, "model") and hasattr(pipe.model, "_write_trace_case_payload"):
        _safe_symlink_or_copy(output_video, trace_case_dir / "final_video.mp4")
        pipe.model._write_trace_case_payload(
            {
                "output_video_path": str(output_video),
                "trace_case_dir": str(trace_case_dir),
                "final_video": "final_video.mp4",
            }
        )

    result = {
        "input_json": str(input_json_path),
        "input_image": str(firstframe_path),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "method": output_video.parent.name,
        "seed": int(args.seed),
        "step": int(args.sampling_steps),
        "guidance": float(args.cfg_scale),
        "sample_shift": float(args.sample_shift),
        "sample_solver": normalize_sample_solver(args.sample_solver),
        "backend": str(args.backend),
        "negative_prompt": str(args.negative_prompt),
        "offload_model": bool(used_offload),
        "ckpt": str(pipe.resolved_wan_root),
        "vjepa_preset": str(cli_args.vjepa_preset),
        "enable_vjepa_guidance": bool(cli_args.enable_vjepa_guidance),
        "vjepa_guidance_mode": str(cli_args.vjepa_guidance_mode),
        "vjepa_model": str(cli_args.vjepa_model),
        "vjepa_ckpt": str(cli_args.vjepa_ckpt) if cli_args.vjepa_ckpt is not None else None,
        "vjepa_target_step_indices": list(cli_args.vjepa_target_step_indices or []),
        "vjepa_inner_k": int(cli_args.vjepa_inner_k),
        "vjepa_backtracking": bool(cli_args.vjepa_backtracking),
        "vjepa_preview_frame_stride": int(cli_args.vjepa_preview_frame_stride),
        "vjepa_artifact_guard_mode": str(cli_args.vjepa_artifact_guard_mode),
        "vjepa_stay_close_max_video_l1": (
            float(cli_args.vjepa_stay_close_max_video_l1)
            if cli_args.vjepa_stay_close_max_video_l1 is not None
            else None
        ),
        "vjepa_recompute_noise_pred_after_guidance": bool(cli_args.vjepa_recompute_noise_pred_after_guidance),
        "vjepa_use_spectral_guidance": bool(cli_args.vjepa_use_spectral_guidance),
        "vjepa_spectral_source": str(cli_args.vjepa_spectral_source),
        "vjepa_spectral_lowpass_ratio": float(cli_args.vjepa_spectral_lowpass_ratio),
        "vjepa_spectral_normalize_percentile": float(cli_args.vjepa_spectral_normalize_percentile),
        "vjepa_spectral_weight_floor": float(cli_args.vjepa_spectral_weight_floor),
        "vjepa_spectral_weight_scale": float(cli_args.vjepa_spectral_weight_scale),
        "vjepa_spectral_mask_dilation": int(cli_args.vjepa_spectral_mask_dilation),
        "motion_mask_mode": str(cli_args.motion_mask_mode),
        "vjepa_anchor_mode": getattr(pipe, "anchor_mode", "repeated_first_frame"),
        "vjepa_trace": list(getattr(pipe, "last_vjepa_trace", [])),
    }
    return result, logs


def main() -> None:
    cli_args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(cli_args.log_level).upper()),
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    ensure_cuda_env()
    set_seed(int(cli_args.seed))

    if cli_args.backend != "official":
        raise ValueError("This script currently supports backend=official only.")

    cli_args, vjepa_config = _resolve_vjepa_settings(cli_args)
    if bool(cli_args.enable_vjepa_guidance) and int(cli_args.frame_num) <= int(cli_args.vjepa_context_frames):
        raise ValueError(
            f"frame_num={cli_args.frame_num} must exceed context_frames={cli_args.vjepa_context_frames}"
        )

    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root=DEFAULT_OUTPUT_BASE_ROOT,
        model_name=model_name,
    )

    args = WanTI2VArgs(
        input_list=Path(cli_args.input_list).expanduser().resolve(),
        output_root=output_root,
        model_name=model_name,
        wan_root=Path(cli_args.wan_root).expanduser().resolve(),
        backend=str(cli_args.backend),
        size=str(cli_args.size),
        frame_num=resolve_default_frame_num(cli_args.frame_num),
        fps=int(cli_args.fps),
        seed=int(cli_args.seed),
        sample_solver=str(cli_args.sample_solver),
        sampling_steps=resolve_default_sampling_steps(cli_args.sampling_steps),
        sample_shift=resolve_default_sample_shift(cli_args.sample_shift),
        cfg_scale=resolve_default_cfg_scale(cli_args.cfg_scale),
        negative_prompt=(
            cli_args.negative_prompt
            if cli_args.negative_prompt is not None
            else DEFAULT_NEGATIVE_PROMPT
        ),
        offload_model=bool(cli_args.offload_model),
        t5_cpu=bool(cli_args.t5_cpu),
        convert_model_dtype=bool(cli_args.convert_model_dtype),
        force=bool(cli_args.force),
    )

    json_paths = read_list_file(args.input_list)
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(args, json_paths)
    manifest["vjepa"] = {
        "preset": str(cli_args.vjepa_preset),
        "enabled": bool(cli_args.enable_vjepa_guidance),
        "model": str(cli_args.vjepa_model),
        "checkpoint": str(cli_args.vjepa_ckpt) if cli_args.vjepa_ckpt is not None else None,
        "device_id": cli_args.vjepa_device_id,
        "guidance_mode": str(cli_args.vjepa_guidance_mode),
        "guidance_steps": int(cli_args.vjepa_guidance_steps),
        "target_step_indices": list(cli_args.vjepa_target_step_indices or []),
        "latent_step_size": float(cli_args.vjepa_latent_step_size),
        "inner_k": int(cli_args.vjepa_inner_k),
        "backtracking": bool(cli_args.vjepa_backtracking),
        "preview_downsample_factor": int(cli_args.vjepa_preview_downsample_factor),
        "preview_frame_stride": int(cli_args.vjepa_preview_frame_stride),
        "window_size": int(cli_args.vjepa_window_size),
        "context_frames": int(cli_args.vjepa_context_frames),
        "max_correction_ratio": float(cli_args.vjepa_max_correction_ratio)
        if cli_args.vjepa_max_correction_ratio is not None
        else None,
        "stay_close_max_video_l1": float(cli_args.vjepa_stay_close_max_video_l1)
        if cli_args.vjepa_stay_close_max_video_l1 is not None
        else None,
        "artifact_guard_mode": str(cli_args.vjepa_artifact_guard_mode),
        "recompute_noise_pred_after_guidance": bool(cli_args.vjepa_recompute_noise_pred_after_guidance),
        "anchor_mode": "repeated_first_frame",
        "use_spectral_guidance": bool(cli_args.vjepa_use_spectral_guidance),
        "spectral_source": str(cli_args.vjepa_spectral_source),
        "spectral_lowpass_ratio": float(cli_args.vjepa_spectral_lowpass_ratio),
        "spectral_normalize_percentile": float(cli_args.vjepa_spectral_normalize_percentile),
        "spectral_weight_floor": float(cli_args.vjepa_spectral_weight_floor),
        "spectral_weight_scale": float(cli_args.vjepa_spectral_weight_scale),
        "spectral_mask_dilation": int(cli_args.vjepa_spectral_mask_dilation),
    }
    with (args.output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    prepared_cases: list[tuple[Path, dict[str, object], Path]] = []
    for input_json_path in json_paths:
        payload = load_json(input_json_path)
        try:
            ensure_str_field(payload, "input_video", input_json_path)
            ensure_str_field(payload, "input_caption", input_json_path)
            payload, firstframe_path = ensure_firstframe_image(input_json_path, payload)
        except Exception as exc:
            print(f"[skip] {input_json_path.stem}: {exc}")
            continue
        prepared_cases.append((input_json_path, payload, firstframe_path))

    pipe = build_vjepa_aware_pipeline(args, cli_args, vjepa_config)
    try:
        for input_json_path, payload, firstframe_path in prepared_cases:
            sample_stem = input_json_path.stem
            output_video = args.output_root / f"{sample_stem}.mp4"
            output_json = args.output_root / f"{sample_stem}.json"
            trace_case_dir = None
            if bool(cli_args.trace_intermediates):
                trace_case_dir = cli_args.trace_root.expanduser().resolve() / model_name / sample_stem

            if output_video.exists() and output_json.exists() and not args.force:
                print(f"[skip] {sample_stem}")
                continue

            try:
                result, case_logs = run_single_case_vjepa(
                    pipe=pipe,
                    args=args,
                    cli_args=cli_args,
                    input_json_path=input_json_path,
                    payload=payload,
                    firstframe_path=firstframe_path,
                    output_video=output_video,
                    trace_case_dir=trace_case_dir,
                )
            except Exception as exc:
                print(f"[error] {sample_stem}: {exc}")
                continue

            write_json(output_json, result)
            for log_line in case_logs:
                print(log_line)
            print(f"[done] {sample_stem}")

        if bool(cli_args.trace_intermediates) and bool(cli_args.trace_build_html):
            trace_root = (cli_args.trace_root.expanduser().resolve() / model_name).resolve()
            trace_root.mkdir(parents=True, exist_ok=True)
            output_html = (
                cli_args.trace_output_html.expanduser().resolve()
                if cli_args.trace_output_html is not None
                else trace_root / "index.html"
            )
            cases = collect_trace_cases(trace_root)
            build_trace_viewer_html(trace_root, output_html, f"V-JEPA Guidance Trace Viewer - {model_name}", cases)
            print(f"[trace-html] {output_html}")
    finally:
        cleanup_pipeline(pipe)


if __name__ == "__main__":
    main()
