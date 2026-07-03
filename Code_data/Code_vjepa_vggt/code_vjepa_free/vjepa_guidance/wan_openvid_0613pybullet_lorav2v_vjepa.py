from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root, resolve_runtime_root
from code_vjepa_vggt.train0419_reference import batch_eval_lora as core

try:
    from .vjepa_surprise import VJEPASurpriseEnergy, build_context_future_clip
    from .wan_latent_guidance import WanVJEPAConfig, pick_guidance_step_indices
except ImportError:
    from vjepa_surprise import VJEPASurpriseEnergy, build_context_future_clip
    from wan_latent_guidance import WanVJEPAConfig, pick_guidance_step_indices

"""
OpenVid 10000 step + 0613pybullet 500 step LoRA + V-JEPA guidance batch eval.

Example:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=0,1 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py \
  --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_100.txt \
  --model-name wan_openvid_0613pybullet_lorav2v_step000500_vjepa \
  --device cuda:0 \
  --vjepa-device cuda:1 \
  --num-frames 49 \
  --vjepa-guidance-steps 2 \
  --vjepa-latent-step-size 0.01
"""


def _resolve_lora_path(weights_root: Path) -> Path:
    checkpoint_path = weights_root.expanduser().resolve() / "checkpoint.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint not found under weights-root: {checkpoint_path}")
    return checkpoint_path


def _diffsynth_sigma_for_timestep(scheduler, timestep: torch.Tensor | int) -> torch.Tensor:
    if isinstance(timestep, torch.Tensor):
        timestep_scalar = timestep.detach().float().reshape(-1)[0].cpu()
    else:
        timestep_scalar = torch.tensor(float(timestep))
    timestep_id = torch.argmin((scheduler.timesteps - timestep_scalar).abs())
    return scheduler.sigmas[timestep_id]


def _predict_x0_from_diffsynth_flow(
    *,
    scheduler,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
) -> torch.Tensor:
    sigma_t = _diffsynth_sigma_for_timestep(scheduler, timestep)
    sigma_t = sigma_t.to(device=latent_xt.device, dtype=latent_xt.dtype)
    while sigma_t.ndim < latent_xt.ndim:
        sigma_t = sigma_t.unsqueeze(-1)
    return latent_xt - sigma_t * model_output


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


def _video_tensor_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for video trace export, got {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor [C,T,H,W], got {tuple(video.shape)}")
    video_u8 = ((video.detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8).cpu()
    frames: list[Image.Image] = []
    for frame_idx in range(video_u8.shape[1]):
        frame = video_u8[:, frame_idx].permute(1, 2, 0).contiguous().numpy()
        frames.append(Image.fromarray(frame))
    return frames


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


def _apply_diffsynth_vjepa_guidance(
    *,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
    preview_decoder,
    energy_fn,
    config: WanVJEPAConfig,
    trace_hook: Optional[Callable[..., None]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    latent_for_grad = latent_xt.detach().float().requires_grad_(True)
    model_output = model_output.detach().float()

    x0_pred = _predict_x0_from_diffsynth_flow(
        scheduler=scheduler,
        latent_xt=latent_for_grad,
        model_output=model_output,
        timestep=timestep,
    )
    preview_video = preview_decoder(
        x0_pred,
        preview_downsample_factor=config.preview_downsample_factor,
        preview_frame_stride=config.preview_frame_stride,
    )
    energy = energy_fn(
        preview_video,
        window_size=config.window_size,
        context_frames=config.context_frames,
        stride=config.stride,
        reduction=config.reduction,
    )
    gradient = torch.autograd.grad(energy, latent_for_grad, retain_graph=False, create_graph=False)[0]
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    raw_grad_norm = float(gradient.norm().item())
    if config.max_grad_norm is not None and raw_grad_norm > config.max_grad_norm:
        gradient = gradient * (config.max_grad_norm / max(raw_grad_norm, 1e-6))
    gradient = _normalize_gradient(gradient, mode=config.gradient_normalization)
    if trace_hook is not None:
        trace_hook(
            x0_pred=x0_pred.detach(),
            preview_video=preview_video.detach(),
            energy=float(energy.detach().item()),
            raw_grad_norm=raw_grad_norm,
            normalized_grad_rms=float(gradient.detach().pow(2).mean().sqrt().item()),
        )
    corrected = latent_xt.detach().float() - config.latent_step_size * gradient
    corrected = corrected.to(dtype=latent_xt.dtype)

    stats = {
        "energy": float(energy.detach().item()),
        "grad_rms": float(gradient.detach().pow(2).mean().sqrt().item()),
        "latent_rms": float(latent_xt.detach().float().pow(2).mean().sqrt().item()),
        "preview_frames": float(preview_video.shape[2]),
        "preview_height": float(preview_video.shape[3]),
        "preview_width": float(preview_video.shape[4]),
    }
    return corrected, stats


def _apply_context_anchored_guidance(
    *,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
    full_decoder: Callable[..., torch.Tensor],
    context_frames_pixel: torch.Tensor,
    energy_obj,
    config: WanVJEPAConfig,
    predicted_future_ref: Optional[torch.Tensor],
    trace_hook: Optional[Callable[..., None]] = None,
    line_search_taps: Optional[list[float]] = None,
    backtracking: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Context-anchored guidance.

    Instead of measuring the *self-consistency* of the generated clip (the old
    ``_apply_diffsynth_vjepa_guidance`` path), this decodes the predicted clean
    video, takes only its generated (future) frames, prepends the *real*
    conditioning frames, and drives the generated future toward the feature-space
    continuation V-JEPA forecasts from that real context.

    ``context_frames_pixel`` is a fixed [1,3,Tc,H,W] tensor in [-1,1] (decoded once
    from the real context latents). ``predicted_future_ref`` is the precomputed
    V-JEPA future prediction; if None it is computed on the fly each call.
    """
    latent_for_grad = latent_xt.detach().float().requires_grad_(True)
    model_output = model_output.detach().float()

    x0_pred = _predict_x0_from_diffsynth_flow(
        scheduler=scheduler,
        latent_xt=latent_for_grad,
        model_output=model_output,
        timestep=timestep,
    )

    # Decode the full predicted clean video (keeps graph to latent_for_grad).
    full_video = full_decoder(x0_pred)  # [1,3,T,H,W] in [-1,1]

    n_ctx = int(config.context_frames)
    total_frames = full_video.shape[2]
    if total_frames <= n_ctx:
        raise ValueError(
            f"Decoded video has {total_frames} frames, need > context_frames={n_ctx}"
        )
    # Generated (future) portion of the decoded prediction. Gradients flow here.
    generated_future = full_video[:, :, n_ctx:]

    ctx = context_frames_pixel.to(device=generated_future.device, dtype=generated_future.dtype)

    clip = build_context_future_clip(
        context_btchw=ctx,
        future_btchw=generated_future,
        window_size=config.window_size,
        context_frames=n_ctx,
    )

    energy = energy_obj.context_anchored(
        clip,
        window_size=config.window_size,
        context_frames=n_ctx,
        predicted_future_ref=predicted_future_ref,
    )

    gradient = torch.autograd.grad(energy, latent_for_grad, retain_graph=False, create_graph=False)[0]
    gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    raw_grad_norm = float(gradient.norm().item())
    if config.max_grad_norm is not None and raw_grad_norm > config.max_grad_norm:
        gradient = gradient * (config.max_grad_norm / max(raw_grad_norm, 1e-6))
    gradient = _normalize_gradient(gradient, mode=config.gradient_normalization)

    # Do not modify the clean-prefix latents: zero the gradient on context region.
    n_ctx_latent = _pixel_frames_to_latent_len(n_ctx)
    if n_ctx_latent > 0 and n_ctx_latent < gradient.shape[2]:
        gradient[:, :, :n_ctx_latent] = 0.0

    line_search: dict[str, float] = {}
    if line_search_taps:
        # Decisive overshoot-vs-noise test: re-evaluate the anchored energy after
        # stepping the latent by -tap*gradient for several taps. If ANY tap lowers
        # energy below E(0), the gradient direction is descending (overshoot); if
        # none do, the direction itself is not a descent direction (noise/flat).
        with torch.no_grad():
            for tap in line_search_taps:
                trial_latent = (latent_xt.detach().float() - tap * gradient).to(dtype=latent_xt.dtype)
                trial_x0 = _predict_x0_from_diffsynth_flow(
                    scheduler=scheduler,
                    latent_xt=trial_latent.float(),
                    model_output=model_output,
                    timestep=timestep,
                )
                trial_video = full_decoder(trial_x0)
                trial_future = trial_video[:, :, n_ctx:]
                trial_clip = build_context_future_clip(
                    context_btchw=ctx,
                    future_btchw=trial_future,
                    window_size=config.window_size,
                    context_frames=n_ctx,
                )
                trial_energy = float(
                    energy_obj.context_anchored(
                        trial_clip,
                        window_size=config.window_size,
                        context_frames=n_ctx,
                        predicted_future_ref=predicted_future_ref,
                    ).item()
                )
                line_search[f"tap_{tap:g}"] = trial_energy

    if trace_hook is not None:
        trace_hook(
            x0_pred=x0_pred.detach(),
            preview_video=full_video.detach(),
            energy=float(energy.detach().item()),
            raw_grad_norm=raw_grad_norm,
            normalized_grad_rms=float(gradient.detach().pow(2).mean().sqrt().item()),
        )

    corrected_step_size = config.latent_step_size
    if backtracking and line_search:
        # Pick the tap that most lowers the anchored energy below E(0). If none of
        # the taps beat the base energy, take NO step (better to skip than climb).
        base_e = float(energy.detach().item())
        best_tap = None
        best_e = base_e
        for key, trial_e in line_search.items():
            tap_val = float(key.split("_", 1)[1])
            if trial_e < best_e:
                best_e = trial_e
                best_tap = tap_val
        corrected_step_size = best_tap if best_tap is not None else 0.0

    correction = corrected_step_size * gradient
    corrected = latent_xt.detach().float() - correction
    corrected = corrected.to(dtype=latent_xt.dtype)

    # Phase 0b: quantify how much the guidance actually writes to the latent.
    # correction_l2   = ||step * grad|| (absolute displacement applied this step)
    # latent_l2       = ||latent_xt||   (scale of the thing we're perturbing)
    # correction_ratio= relative perturbation (correction_l2 / latent_l2); a tiny
    #                   ratio means the denoiser will almost certainly wash it out.
    correction_l2 = float(correction.detach().norm().item())
    latent_l2 = float(latent_xt.detach().float().norm().item())
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
        "step_size_used": float(corrected_step_size),
    }
    if line_search:
        stats["line_search"] = line_search
        stats["line_search_base_energy"] = float(energy.detach().item())
    return corrected, stats


def _pixel_frames_to_latent_len(pixel_frames: int) -> int:
    """Wan VAE temporal compression: latent_T = (pixel_T - 1) // 4 + 1.

    The clean prefix occupies the first ``_pixel_frames_to_latent_len(Tc)`` latent
    frames; guidance must not touch them since they are overwritten each step.
    """
    if pixel_frames <= 0:
        return 0
    return (pixel_frames - 1) // 4 + 1


class ContextAwareWanVideoPipelineVJEPA(core.ContextAwareWanVideoPipeline):
    @staticmethod
    def from_pretrained_vjepa(
        *,
        wan_root: Path,
        device: str,
        lora_path: Path | None,
        vjepa_model_name: str,
        vjepa_checkpoint_path: Path | None,
        vjepa_device: str | None,
        vjepa_config: WanVJEPAConfig,
        enable_vjepa_guidance: bool,
    ) -> "ContextAwareWanVideoPipelineVJEPA":
        pipe = core.ContextAwareWanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=core.build_model_configs(wan_root),
            tokenizer_config=core.ModelConfig(path=str(core.find_tokenizer_path(wan_root))),
        )
        pipe.__class__ = ContextAwareWanVideoPipelineVJEPA
        pipe.configure_vjepa(
            vjepa_model_name=vjepa_model_name,
            vjepa_checkpoint_path=vjepa_checkpoint_path,
            vjepa_device=vjepa_device,
            vjepa_config=vjepa_config,
            enable_vjepa_guidance=enable_vjepa_guidance,
        )
        if lora_path is not None:
            pipe.load_lora(pipe.dit, str(lora_path), alpha=1.0)
        return pipe

    def configure_vjepa(
        self,
        *,
        vjepa_model_name: str,
        vjepa_checkpoint_path: Path | None,
        vjepa_device: str | None,
        vjepa_config: WanVJEPAConfig,
        enable_vjepa_guidance: bool,
    ) -> None:
        self.vjepa_model_name = vjepa_model_name
        self.vjepa_checkpoint_path = str(vjepa_checkpoint_path) if vjepa_checkpoint_path is not None else None
        self.vjepa_device = torch.device(vjepa_device) if vjepa_device is not None else torch.device(self.device)
        self.vjepa_config = vjepa_config
        self.enable_vjepa_guidance = enable_vjepa_guidance
        self._vjepa_energy: VJEPASurpriseEnergy | None = None
        self.vjepa_inner_k: int = 1  # repetitions of guidance per selected step

        for parameter in self.vae.model.parameters():
            parameter.requires_grad_(False)
        self.vae.model.eval()
        self.trace_intermediates_enabled = False
        self.trace_max_strip_frames = 8
        self.trace_case_dir: Path | None = None
        self.trace_case_payload: dict[str, Any] | None = None
        self.trace_fps = 16
        self.vjepa_target_timestep_values: list[int] = []
        self.vjepa_target_step_indices: list[int] = []
        # Externally-provided context-anchored anchor (context pixels + future ref).
        # When set, guidance uses these instead of decoding the clean prefix, so the
        # guidance and the probe optimize/measure the exact same energy.
        self._external_anchor_context_pixel: torch.Tensor | None = None
        self._external_anchor_future_ref: torch.Tensor | None = None

    def set_external_anchor(
        self,
        *,
        context_frames_pixel: torch.Tensor | None,
        predicted_future_ref: torch.Tensor | None,
    ) -> None:
        """Inject a fixed context-anchored reference shared with an external probe.

        ``context_frames_pixel`` is [1,3,Tc,H,W] in [-1,1]; ``predicted_future_ref``
        is the precomputed V-JEPA future prediction. Pass (None, None) to clear and
        fall back to decoding the clean prefix internally.
        """
        self._external_anchor_context_pixel = context_frames_pixel
        self._external_anchor_future_ref = predicted_future_ref

    def set_line_search_taps(self, taps: Optional[list[float]]) -> None:
        """Enable a diagnostic line search at each guidance step (logging only)."""
        self._line_search_taps = list(taps) if taps else None

    def set_backtracking(self, enabled: bool, taps: Optional[list[float]] = None) -> None:
        """Enable backtracking: at each guidance step pick the tap that most lowers
        the anchored energy (take no step if none beat E(0)). Robust to the sharp,
        shallow energy basin -- avoids the fixed-step overshoot."""
        self._backtracking = bool(enabled)
        self._backtracking_taps = list(taps) if taps else [0.002, 0.005, 0.01, 0.02]

    def configure_trace(
        self,
        *,
        enabled: bool,
        max_strip_frames: int,
    ) -> None:
        self.trace_intermediates_enabled = enabled
        self.trace_max_strip_frames = max(2, int(max_strip_frames))

    def configure_target_timesteps(self, timestep_values: list[int]) -> None:
        self.vjepa_target_timestep_values = [int(value) for value in timestep_values]

    def configure_target_step_indices(self, step_indices: list[int]) -> None:
        self.vjepa_target_step_indices = [int(value) for value in step_indices]

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
        energy: float,
        raw_grad_norm: float,
        normalized_grad_rms: float,
    ) -> None:
        if self.trace_case_dir is None:
            return
        step_dir = self.trace_case_dir / f"step_{step_idx:02d}_t{timestep:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        preview_frames = _video_tensor_to_pil_frames(preview_video)
        core.save_video(
            preview_frames,
            str(step_dir / "preview_video.mp4"),
            fps=max(1, self.trace_fps // max(1, self.vjepa_config.preview_frame_stride)),
            quality=6,
        )
        _save_frame_strip(
            preview_frames,
            step_dir / "preview_strip.png",
            max_frames=self.trace_max_strip_frames,
            tile_height=140,
        )

        latent_frames, latent_stats = _latent_norm_to_pil_frames(x0_pred)
        core.save_video(
            latent_frames,
            str(step_dir / "x0_latent_norm.mp4"),
            fps=max(1, self.trace_fps // max(1, self.vjepa_config.preview_frame_stride)),
            quality=6,
        )
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
            **latent_stats,
        }
        (step_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _select_guidance_steps(self) -> tuple[set[int], dict[int, int]]:
        if self.vjepa_target_step_indices:
            selected = {
                max(0, min(int(step_idx), len(self.scheduler.timesteps) - 1))
                for step_idx in self.vjepa_target_step_indices
            }
            mapping = {
                int(step_idx): int(round(float(self.scheduler.timesteps[step_idx].detach().cpu().item())))
                for step_idx in sorted(selected)
            }
            return selected, mapping

        if self.vjepa_target_timestep_values:
            scheduler_values = self.scheduler.timesteps.detach().cpu().tolist()
            selected: set[int] = set()
            mapping: dict[int, int] = {}
            for target in self.vjepa_target_timestep_values:
                best_idx = min(
                    range(len(scheduler_values)),
                    key=lambda idx: abs(float(scheduler_values[idx]) - float(target)),
                )
                selected.add(int(best_idx))
                mapping[int(best_idx)] = int(round(float(scheduler_values[best_idx])))
            return selected, mapping

        selected = set(
            pick_guidance_step_indices(
                total_steps=len(self.scheduler.timesteps),
                count=self.vjepa_config.guidance_steps,
                min_step_percent=self.vjepa_config.min_step_percent,
                max_step_percent=self.vjepa_config.max_step_percent,
            )
        )
        mapping = {
            int(step_idx): int(round(float(self.scheduler.timesteps[step_idx].detach().cpu().item())))
            for step_idx in sorted(selected)
        }
        return selected, mapping

    def _ensure_vjepa_energy(self) -> VJEPASurpriseEnergy:
        if not self.enable_vjepa_guidance:
            raise RuntimeError("V-JEPA guidance is disabled for this pipeline instance.")
        if self._vjepa_energy is None:
            logging.info("Loading V-JEPA energy model: %s", self.vjepa_model_name)
            self._vjepa_energy = VJEPASurpriseEnergy(
                model_name=self.vjepa_model_name,
                device=self.vjepa_device,
                local_torchhub=True,
                checkpoint_path=self.vjepa_checkpoint_path,
            )
        return self._vjepa_energy

    def _decode_preview_video(
        self,
        x0_latent: torch.Tensor,
        *,
        preview_downsample_factor: int,
        preview_frame_stride: int,
        tiled: bool,
        tile_size: tuple[int, int],
        tile_stride: tuple[int, int],
        framewise_decoding: bool,
        restore_model_names: tuple[str, ...],
    ) -> torch.Tensor:
        preview_latent = x0_latent
        if preview_frame_stride > 1:
            preview_latent = preview_latent[:, :, ::preview_frame_stride].contiguous()
        if preview_downsample_factor > 1:
            preview_latent = F.interpolate(
                preview_latent,
                scale_factor=(1.0, 1.0 / preview_downsample_factor, 1.0 / preview_downsample_factor),
                mode="trilinear",
                align_corners=False,
            )

        self.load_models_to_device(["vae"])
        try:
            vae_dtype = next(self.vae.model.parameters()).dtype
            preview_latent = preview_latent.to(device=self.device, dtype=vae_dtype)
            if framewise_decoding:
                preview_video = self.vae.decode_framewise(preview_latent, device=self.device)
            else:
                preview_video = self.vae.decode(
                    preview_latent,
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
            return preview_video.clamp_(-1, 1)
        finally:
            self.load_models_to_device(list(restore_model_names))

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_image: Optional[Image.Image] = None,
        end_image: Optional[Image.Image] = None,
        input_video: Optional[list[Image.Image]] = None,
        context_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        input_audio=None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video=None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        camera_control_direction: Optional[
            Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        vap_video: Optional[list[Image.Image]] = None,
        vap_prompt: Optional[str] = " ",
        negative_vap_prompt: Optional[str] = " ",
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        switch_DiT_boundary: Optional[float] = 0.875,
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        motion_bucket_id: Optional[int] = None,
        longcat_video: Optional[list[Image.Image]] = None,
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        wantodance_music_path: Optional[str] = None,
        wantodance_reference_image: Optional[Image.Image] = None,
        wantodance_fps: Optional[float] = 30,
        wantodance_keyframes: Optional[list[Image.Image]] = None,
        wantodance_keyframes_mask: Optional[list[int]] = None,
        framewise_decoding: bool = False,
        progress_bar_cmd=tqdm,
        output_type: Optional[Literal["quantized", "floatpoint"]] = "quantized",
    ):
        if not self.enable_vjepa_guidance or self.vjepa_config.guidance_steps <= 0:
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                input_image=input_image,
                end_image=end_image,
                input_video=input_video,
                context_video=context_video,
                denoising_strength=denoising_strength,
                input_audio=input_audio,
                audio_embeds=audio_embeds,
                audio_sample_rate=audio_sample_rate,
                s2v_pose_video=s2v_pose_video,
                s2v_pose_latents=s2v_pose_latents,
                motion_video=motion_video,
                control_video=control_video,
                reference_image=reference_image,
                camera_control_direction=camera_control_direction,
                camera_control_speed=camera_control_speed,
                camera_control_origin=camera_control_origin,
                vace_video=vace_video,
                vace_video_mask=vace_video_mask,
                vace_reference_image=vace_reference_image,
                vace_scale=vace_scale,
                animate_pose_video=animate_pose_video,
                animate_face_video=animate_face_video,
                animate_inpaint_video=animate_inpaint_video,
                animate_mask_video=animate_mask_video,
                vap_video=vap_video,
                vap_prompt=vap_prompt,
                negative_vap_prompt=negative_vap_prompt,
                seed=seed,
                rand_device=rand_device,
                height=height,
                width=width,
                num_frames=num_frames,
                cfg_scale=cfg_scale,
                cfg_merge=cfg_merge,
                switch_DiT_boundary=switch_DiT_boundary,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                motion_bucket_id=motion_bucket_id,
                longcat_video=longcat_video,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
                tea_cache_l1_thresh=tea_cache_l1_thresh,
                tea_cache_model_id=tea_cache_model_id,
                wantodance_music_path=wantodance_music_path,
                wantodance_reference_image=wantodance_reference_image,
                wantodance_fps=wantodance_fps,
                wantodance_keyframes=wantodance_keyframes,
                wantodance_keyframes_mask=wantodance_keyframes_mask,
                framewise_decoding=framewise_decoding,
                progress_bar_cmd=progress_bar_cmd,
                output_type=output_type,
            )

        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )
        if context_video is not None and input_image is None and len(context_video) > 0:
            input_image = context_video[0]

        inputs_posi = {
            "prompt": prompt,
            "vap_prompt": vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "negative_vap_prompt": negative_vap_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "context_video": context_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "longcat_video": longcat_video,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
            "vap_video": vap_video,
            "wantodance_music_path": wantodance_music_path,
            "wantodance_reference_image": wantodance_reference_image,
            "wantodance_fps": wantodance_fps,
            "wantodance_keyframes": wantodance_keyframes,
            "wantodance_keyframes_mask": wantodance_keyframes_mask,
            "framewise_decoding": framewise_decoding,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit,
                self,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )

        energy_fn = self._ensure_vjepa_energy()
        selected_steps, selected_step_timestep_map = self._select_guidance_steps()
        self._write_trace_case_payload(
            {
                "selected_guidance_steps": sorted(int(step) for step in selected_steps),
                "selected_guidance_timestep_map": {
                    str(step): int(timestep) for step, timestep in sorted(selected_step_timestep_map.items())
                },
                "target_step_indices": [int(value) for value in self.vjepa_target_step_indices],
                "target_timestep_values": [int(value) for value in self.vjepa_target_timestep_values],
                "num_inference_steps": int(num_inference_steps),
                "vjepa_config": {
                    "guidance_steps": int(self.vjepa_config.guidance_steps),
                    "min_step_percent": float(self.vjepa_config.min_step_percent),
                    "max_step_percent": float(self.vjepa_config.max_step_percent),
                    "latent_step_size": float(self.vjepa_config.latent_step_size),
                    "preview_downsample_factor": int(self.vjepa_config.preview_downsample_factor),
                    "preview_frame_stride": int(self.vjepa_config.preview_frame_stride),
                    "window_size": int(self.vjepa_config.window_size),
                    "context_frames": int(self.vjepa_config.context_frames),
                    "stride": int(self.vjepa_config.stride),
                    "reduction": str(self.vjepa_config.reduction),
                    "gradient_normalization": str(self.vjepa_config.gradient_normalization),
                    "max_grad_norm": self.vjepa_config.max_grad_norm,
                },
            }
        )

        # --- Context-anchored guidance setup (guidance_mode == "context_anchored") ---
        # Decode the real context latents to pixel space once, and precompute the
        # V-JEPA future prediction from that fixed real context. Both are held fixed
        # for the whole denoising trajectory.
        guidance_mode = getattr(self.vjepa_config, "guidance_mode", "surprise")
        context_frames_pixel: Optional[torch.Tensor] = None
        predicted_future_ref: Optional[torch.Tensor] = None
        if guidance_mode == "context_anchored" and selected_steps:
            n_ctx = int(self.vjepa_config.context_frames)
            future_frames = int(self.vjepa_config.window_size) - n_ctx
            # Prefer an externally-injected anchor (set by the probe harness) so the
            # guidance optimizes the EXACT same energy the probe measures. Fall back
            # to decoding the clean-prefix latent only if none was provided.
            external_ctx = getattr(self, "_external_anchor_context_pixel", None)
            external_ref = getattr(self, "_external_anchor_future_ref", None)
            if external_ctx is not None and external_ref is not None:
                context_frames_pixel = external_ctx.detach()
                predicted_future_ref = external_ref.detach()
                logging.info(
                    "Context-anchored guidance using EXTERNAL anchor: ctx_frames=%d future_frames=%d ref_shape=%s",
                    context_frames_pixel.shape[2],
                    future_frames,
                    tuple(predicted_future_ref.shape),
                )
            elif inputs_shared.get("clean_prefix_latents") is not None:
                with torch.no_grad():
                    prefix_latents = inputs_shared["clean_prefix_latents"].detach()
                    context_video_pixel = self._decode_preview_video(
                        prefix_latents,
                        preview_downsample_factor=1,
                        preview_frame_stride=1,
                        tiled=tiled,
                        tile_size=tile_size,
                        tile_stride=tile_stride,
                        framewise_decoding=framewise_decoding,
                        restore_model_names=tuple(self.in_iteration_models),
                    )  # [1,3,Tc,H,W]
                    context_frames_pixel = context_video_pixel.detach()
                    placeholder = torch.zeros(
                        context_frames_pixel.shape[0],
                        context_frames_pixel.shape[1],
                        future_frames,
                        context_frames_pixel.shape[3],
                        context_frames_pixel.shape[4],
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
                    "Context-anchored guidance decoded-prefix anchor: ctx_frames=%d future_frames=%d ref_shape=%s",
                    context_frames_pixel.shape[2],
                    future_frames,
                    tuple(predicted_future_ref.shape),
                )

        self.load_models_to_device(self.in_iteration_models)
        active_model_names = self.in_iteration_models
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep_cpu in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if (
                timestep_cpu.item() < switch_DiT_boundary * 1000
                and self.dit2 is not None
                and models["dit"] is not self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                active_model_names = self.in_iteration_models_2
                models["dit"] = self.dit2
                models["vace"] = self.vace2

            timestep = timestep_cpu.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_nega,
                        timestep=timestep,
                    )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            if progress_id in selected_steps:
                inner_k = max(1, int(getattr(self, "vjepa_inner_k", 1)))
                use_anchored = (
                    guidance_mode == "context_anchored"
                    and context_frames_pixel is not None
                    and predicted_future_ref is not None
                )
                for _inner in range(inner_k):
                    with torch.enable_grad():
                        if use_anchored:
                            inputs_shared["latents"], stats = _apply_context_anchored_guidance(
                                latent_xt=inputs_shared["latents"],
                                model_output=noise_pred,
                                timestep=timestep_cpu,
                                scheduler=self.scheduler,
                                full_decoder=lambda x0_pred: self._decode_preview_video(
                                    x0_pred,
                                    preview_downsample_factor=self.vjepa_config.preview_downsample_factor,
                                    preview_frame_stride=1,
                                    tiled=tiled,
                                    tile_size=tile_size,
                                    tile_stride=tile_stride,
                                    framewise_decoding=framewise_decoding,
                                    restore_model_names=active_model_names,
                                ),
                                context_frames_pixel=context_frames_pixel,
                                energy_obj=energy_fn,
                                config=self.vjepa_config,
                                predicted_future_ref=predicted_future_ref,
                                line_search_taps=(
                                    getattr(self, "_line_search_taps", None)
                                    or (getattr(self, "_backtracking_taps", None)
                                        if getattr(self, "_backtracking", False) else None)
                                ),
                                backtracking=getattr(self, "_backtracking", False),
                                trace_hook=(
                                    lambda *, x0_pred, preview_video, energy, raw_grad_norm, normalized_grad_rms: self._trace_guidance_step(
                                        step_idx=progress_id,
                                        timestep=int(timestep_cpu.item()),
                                        x0_pred=x0_pred,
                                        preview_video=preview_video,
                                        energy=energy,
                                        raw_grad_norm=raw_grad_norm,
                                        normalized_grad_rms=normalized_grad_rms,
                                    )
                                    if self.trace_intermediates_enabled and _inner == 0
                                    else None
                                ),
                            )
                        else:
                            inputs_shared["latents"], stats = _apply_diffsynth_vjepa_guidance(
                                latent_xt=inputs_shared["latents"],
                                model_output=noise_pred,
                                timestep=timestep_cpu,
                                scheduler=self.scheduler,
                                preview_decoder=lambda x0_pred, preview_downsample_factor, preview_frame_stride: self._decode_preview_video(
                                    x0_pred,
                                    preview_downsample_factor=preview_downsample_factor,
                                    preview_frame_stride=preview_frame_stride,
                                    tiled=tiled,
                                    tile_size=tile_size,
                                    tile_stride=tile_stride,
                                    framewise_decoding=framewise_decoding,
                                    restore_model_names=active_model_names,
                                ),
                                energy_fn=energy_fn,
                                config=self.vjepa_config,
                                trace_hook=(
                                    lambda *, x0_pred, preview_video, energy, raw_grad_norm, normalized_grad_rms: self._trace_guidance_step(
                                        step_idx=progress_id,
                                        timestep=int(timestep_cpu.item()),
                                        x0_pred=x0_pred,
                                        preview_video=preview_video,
                                        energy=energy,
                                        raw_grad_norm=raw_grad_norm,
                                        normalized_grad_rms=normalized_grad_rms,
                                    )
                                    if self.trace_intermediates_enabled and _inner == 0
                                    else None
                                ),
                            )
                logging.info(
                    "V-JEPA[%s] step=%d timestep=%d inner_k=%d energy=%.6f grad_rms=%.6f "
                    "corr_l2=%.4f latent_l2=%.1f corr_ratio=%.5f step_used=%.4f preview=%dx%dx%d",
                    guidance_mode,
                    progress_id,
                    int(timestep_cpu.item()),
                    inner_k,
                    stats["energy"],
                    stats["grad_rms"],
                    stats.get("correction_l2", float("nan")),
                    stats.get("latent_l2", float("nan")),
                    stats.get("correction_ratio", float("nan")),
                    stats.get("step_size_used", float("nan")),
                    int(stats["preview_frames"]),
                    int(stats["preview_height"]),
                    int(stats["preview_width"]),
                )
                if stats.get("line_search"):
                    base_e = stats.get("line_search_base_energy", float("nan"))
                    taps_str = "  ".join(
                        f"{k}={v:.6f}({'DOWN' if v < base_e else 'up'})"
                        for k, v in stats["line_search"].items()
                    )
                    logging.info(
                        "V-JEPA[%s] step=%d LINE-SEARCH base_E=%.6f  %s",
                        guidance_mode, progress_id, base_e, taps_str,
                    )

            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if inputs_shared.get("clean_prefix_latents") is not None:
                prefix_len = inputs_shared["clean_prefix_latents"].shape[2]
                inputs_shared["latents"][:, :, :prefix_len] = inputs_shared["clean_prefix_latents"]
            elif "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

        if vace_reference_image is not None or (
            animate_pose_video is not None and animate_face_video is not None
        ):
            if vace_reference_image is not None and isinstance(vace_reference_image, list):
                trim_frames = len(vace_reference_image)
            else:
                trim_frames = 1
            inputs_shared["latents"] = inputs_shared["latents"][:, :, trim_frames:]

        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        self.load_models_to_device(["vae"])
        if framewise_decoding:
            video = self.vae.decode_framewise(inputs_shared["latents"], device=self.device)
        else:
            video = self.vae.decode(
                inputs_shared["latents"],
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the fixed 0613 pybullet Wan LoRA checkpoint with optional V-JEPA latent guidance. "
            "This keeps the original batch_eval_lora data plumbing and swaps in a DiffSynth Wan+V-JEPA pipeline."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True, help="step-* dir containing checkpoint.safetensors")
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--vjepa-device", type=str, default=None)
    parser.add_argument("--height", type=int, default=core.DEFAULT_SINGLE_CASE_HEIGHT)
    parser.add_argument("--width", type=int, default=core.DEFAULT_SINGLE_CASE_WIDTH)
    parser.add_argument("--num-frames", type=int, default=core.DEFAULT_SINGLE_CASE_NUM_FRAMES)
    parser.add_argument("--context-frames", type=int, default=core.DEFAULT_SINGLE_CASE_CONTEXT_FRAMES)
    parser.add_argument("--fps", type=int, default=core.DEFAULT_SINGLE_CASE_FPS)
    parser.add_argument("--num-inference-steps", type=int, default=core.DEFAULT_SINGLE_CASE_NUM_INFERENCE_STEPS)
    parser.add_argument("--cfg-scale", type=float, default=core.DEFAULT_SINGLE_CASE_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SINGLE_CASE_SEED)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative-prompt", type=str, default=core.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--conditioning-mode", choices=["context_aware", "input_image_only"], default="context_aware")
    parser.add_argument("--context-resize-mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable-vjepa-guidance", action="store_true")
    parser.add_argument("--vjepa-model", type=str, default="vith")
    parser.add_argument("--vjepa-ckpt", type=Path, default=Path("/data/gaoya/ckpt/VJEPA2/vith.pt"))
    parser.add_argument("--vjepa-guidance-steps", type=int, default=2)
    parser.add_argument("--vjepa-min-step-percent", type=float, default=0.35)
    parser.add_argument("--vjepa-max-step-percent", type=float, default=0.65)
    parser.add_argument("--vjepa-target-step-indices", type=int, nargs="*", default=None)
    parser.add_argument("--vjepa-target-timesteps", type=int, nargs="*", default=None)
    parser.add_argument("--vjepa-latent-step-size", type=float, default=0.01)
    parser.add_argument("--vjepa-preview-downsample-factor", type=int, default=4)
    parser.add_argument("--vjepa-preview-frame-stride", type=int, default=2)
    parser.add_argument("--vjepa-window-size", type=int, default=16)
    parser.add_argument("--vjepa-context-frames", type=int, default=8)
    parser.add_argument("--vjepa-stride", type=int, default=4)
    parser.add_argument("--vjepa-reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--vjepa-grad-norm-mode", choices=["rms", "l2", "none"], default="rms")
    parser.add_argument("--vjepa-max-grad-norm", type=float, default=10.0)
    parser.add_argument("--trace-intermediates", action="store_true")
    parser.add_argument("--trace-root", type=Path, default=None)
    parser.add_argument("--trace-max-strip-frames", type=int, default=8)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def build_core_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    model_name = str(cli_args.model_name).strip()
    output_root = resolve_output_root(
        explicit_output_root=cli_args.output_root,
        base_output_root="/data/gaoya/AAA_test_video/0623/test/v2v/loramodel",
        model_name=model_name,
    )
    runtime_root = resolve_runtime_root(
        explicit_runtime_root=cli_args.runtime_root,
        base_runtime_root="/data/gaoya/AAA_test_video/0623/test/v2v/loramodel",
        model_name=model_name,
    )
    return argparse.Namespace(
        wan_root=cli_args.wan_root.expanduser().resolve(),
        output_root=output_root,
        runtime_root=runtime_root,
        weights_root=cli_args.weights_root.expanduser().resolve(),
        lora_path=_resolve_lora_path(cli_args.weights_root),
        input_json_list_path=cli_args.input_json_list_path.expanduser().resolve(),
        meta_list_path=None,
        meta_json_path=None,
        context_path=None,
        output_video_path=None,
        prompt=None,
        sample_id=core.DEFAULT_SINGLE_CASE_SAMPLE_ID,
        dataset_name=core.DEFAULT_SINGLE_CASE_DATASET_NAME,
        future_gt_path=None,
        full_video_path=None,
        first_frame_path=None,
        context_resize_mode=cli_args.context_resize_mode,
        conditioning_mode=cli_args.conditioning_mode,
        device=cli_args.device,
        height=int(cli_args.height),
        width=int(cli_args.width),
        fps=int(cli_args.fps),
        num_frames=int(cli_args.num_frames),
        context_frames=int(cli_args.context_frames),
        num_inference_steps=int(cli_args.num_inference_steps),
        cfg_scale=float(cli_args.cfg_scale),
        seed=int(cli_args.seed),
        quality=int(cli_args.quality),
        model_name=model_name,
        negative_prompt=str(cli_args.negative_prompt),
        overwrite=bool(cli_args.overwrite),
        limit=cli_args.limit,
        no_metadata=False,
        multi_gpu=bool(cli_args.multi_gpu),
        num_shards=int(cli_args.num_shards),
        shard_id=int(cli_args.shard_id),
        worker=bool(cli_args.worker),
    )


def build_vjepa_config(cli_args: argparse.Namespace) -> WanVJEPAConfig:
    max_grad_norm = cli_args.vjepa_max_grad_norm
    if max_grad_norm is not None and max_grad_norm <= 0:
        max_grad_norm = None
    return WanVJEPAConfig(
        guidance_steps=int(cli_args.vjepa_guidance_steps),
        min_step_percent=float(cli_args.vjepa_min_step_percent),
        max_step_percent=float(cli_args.vjepa_max_step_percent),
        latent_step_size=float(cli_args.vjepa_latent_step_size),
        preview_downsample_factor=int(cli_args.vjepa_preview_downsample_factor),
        preview_frame_stride=int(cli_args.vjepa_preview_frame_stride),
        window_size=int(cli_args.vjepa_window_size),
        context_frames=int(cli_args.vjepa_context_frames),
        stride=int(cli_args.vjepa_stride),
        reduction=str(cli_args.vjepa_reduction),
        gradient_normalization=str(cli_args.vjepa_grad_norm_mode),
        max_grad_norm=max_grad_norm,
    )


def _build_pipeline_with_vjepa(cli_args: argparse.Namespace):
    vjepa_config = build_vjepa_config(cli_args)
    vjepa_device = cli_args.vjepa_device or cli_args.device

    def _builder(wan_root: Path, device: str, lora_path: Path | None):
        pipe = ContextAwareWanVideoPipelineVJEPA.from_pretrained_vjepa(
            wan_root=wan_root,
            device=device,
            lora_path=lora_path,
            vjepa_model_name=str(cli_args.vjepa_model),
            vjepa_checkpoint_path=cli_args.vjepa_ckpt.expanduser().resolve() if cli_args.vjepa_ckpt is not None else None,
            vjepa_device=vjepa_device,
            vjepa_config=vjepa_config,
            enable_vjepa_guidance=not cli_args.disable_vjepa_guidance,
        )
        pipe.configure_trace(
            enabled=bool(cli_args.trace_intermediates),
            max_strip_frames=int(cli_args.trace_max_strip_frames),
        )
        pipe.configure_target_timesteps(
            [int(value) for value in (cli_args.vjepa_target_timesteps or [])]
        )
        pipe.configure_target_step_indices(
            [int(value) for value in (cli_args.vjepa_target_step_indices or [])]
        )
        return pipe

    return _builder


def run_generation_with_optional_trace(
    args: argparse.Namespace,
    cli_args: argparse.Namespace,
    generated_dir: Path,
    metadata_dir: Path,
) -> None:
    if not cli_args.trace_intermediates:
        core.run_generation(args, generated_dir, metadata_dir)
        return

    trace_root = (
        cli_args.trace_root.expanduser().resolve()
        if cli_args.trace_root is not None
        else (args.runtime_root / "trace_viewer" / args.model_name).resolve()
    )
    trace_root.mkdir(parents=True, exist_ok=True)

    cases = core.collect_cases_from_args(args)
    generated_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    core.write_run_manifest(args, metadata_dir)
    method_name = core.build_method_name(args.lora_path)

    per_case_jsonl = core.per_case_jsonl_path(metadata_dir, args.model_name, args.num_shards, args.shard_id)
    if args.overwrite and per_case_jsonl.exists():
        per_case_jsonl.unlink()
    existing_entries = core.load_jsonl(per_case_jsonl) if per_case_jsonl.exists() else []
    entries_by_index: dict[int, dict[str, Any]] = {}
    for entry in existing_entries:
        entries_by_index[core.entry_sort_index(entry)] = entry

    indexed_cases = list(enumerate(cases))
    shard_cases = [(idx, row) for idx, row in indexed_cases if idx % args.num_shards == args.shard_id]
    print(
        f"[worker] shard_id={args.shard_id}/{args.num_shards}, "
        f"num_cases={len(shard_cases)}, device={args.device}, seed={args.seed}, "
        f"context_frames={args.context_frames}, trace_root={trace_root}"
    )

    pipe = core.build_pipeline(args.wan_root, args.device, args.lora_path)

    for index, row in shard_cases:
        output_override = row.get("output_path_override")
        output_path = (
            Path(output_override)
            if isinstance(output_override, str) and output_override
            else generated_dir / row["output_name"]
        )
        sidecar_path = output_path.with_suffix(".json")
        context_path = Path(row["context_path"])
        core.assert_exists(context_path, "Context video")
        trace_case_dir = trace_root / f"{index:04d}_{core.sanitize_filename(str(row['sample_id']))}"
        if hasattr(pipe, "set_trace_case"):
            pipe.set_trace_case(
                case_dir=trace_case_dir,
                sample_id=str(row["sample_id"]),
                prompt=str(row["caption"]),
                output_video_path=output_path,
                source_json=str(row["meta_path"]),
                fps=int(args.fps),
            )

        if output_path.exists() and not args.overwrite:
            print(f"[skip][shard {args.shard_id}] {row['output_name']} | seed={args.seed}")
            if row.get("simple_input_json_mode"):
                case_payload = core.build_simple_result_payload(
                    input_json_path=Path(str(row["meta_path"])).expanduser().resolve(),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=output_path,
                    method=method_name,
                    seed=args.seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=args.lora_path,
                    status="skipped_existing",
                )
            else:
                case_payload = core.build_case_metadata(
                    args=args,
                    row=row,
                    index=index,
                    seed=args.seed,
                    output_path=output_path,
                    used_context_frames=1 if args.conditioning_mode == "input_image_only" else args.context_frames,
                    status="skipped_existing",
                )
            core.write_json(sidecar_path, case_payload)
            entries_by_index[index] = case_payload
            continue

        print(
            f"[generate][shard {args.shard_id}] {row['dataset']}::{row['sample_id']} "
            f"-> {row['output_name']} | seed={args.seed}"
        )
        try:
            first_frame_path = None
            raw_first_frame_path = row.get("source_paths", {}).get("first_frame_path")
            if isinstance(raw_first_frame_path, str) and raw_first_frame_path:
                first_frame_path = Path(raw_first_frame_path)
            video, used_context_frames = core.generate_one_video(
                pipe=pipe,
                context_path=context_path,
                first_frame_path=first_frame_path,
                prompt=row["caption"],
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                context_frames=args.context_frames,
                output_num_frames=args.requested_output_frames,
                context_resize_mode=row.get("context_resize_mode", "crop"),
                conditioning_mode=args.conditioning_mode,
            )
            core.save_video(video, str(output_path), fps=args.fps, quality=args.quality)
            if hasattr(pipe, "set_trace_case"):
                pipe.set_trace_case(
                    case_dir=trace_case_dir,
                    sample_id=str(row["sample_id"]),
                    prompt=str(row["caption"]),
                    output_video_path=output_path,
                    source_json=str(row["meta_path"]),
                    fps=int(args.fps),
                )
            if row.get("simple_input_json_mode"):
                case_payload = core.build_simple_result_payload(
                    input_json_path=Path(str(row["meta_path"])).expanduser().resolve(),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=output_path,
                    method=method_name,
                    seed=args.seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=args.lora_path,
                    status="generated",
                )
            else:
                case_payload = core.build_case_metadata(
                    args=args,
                    row=row,
                    index=index,
                    seed=args.seed,
                    output_path=output_path,
                    used_context_frames=used_context_frames,
                    status="generated",
                )
        except Exception as exc:
            print(f"[error][shard {args.shard_id}] {row['output_name']} | {exc}")
            if row.get("simple_input_json_mode"):
                case_payload = core.build_simple_result_payload(
                    input_json_path=Path(str(row["meta_path"])).expanduser().resolve(),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=output_path,
                    method=method_name,
                    seed=args.seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=args.lora_path,
                    status="failed",
                    error=repr(exc),
                )
            else:
                case_payload = core.build_case_metadata(
                    args=args,
                    row=row,
                    index=index,
                    seed=args.seed,
                    output_path=output_path,
                    used_context_frames=0,
                    status="failed",
                    error=repr(exc),
                )
        core.write_json(sidecar_path, case_payload)
        entries_by_index[index] = case_payload
    core.write_jsonl(
        per_case_jsonl,
        [entries_by_index[idx] for idx in sorted(entries_by_index)],
    )


def main() -> None:
    cli_args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(cli_args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_core_args(cli_args)

    core.assert_exists(args.wan_root, "Wan root")
    core.assert_exists(args.weights_root, "Weights root")
    core.assert_exists(args.lora_path, "LoRA checkpoint")
    core.assert_exists(args.input_json_list_path, "Input json list path")
    if not cli_args.disable_vjepa_guidance:
        core.assert_exists(cli_args.vjepa_ckpt.expanduser().resolve(), "V-JEPA checkpoint")
    core.validate_args(args)

    aligned_height, aligned_width = core.align_generation_size(args.height, args.width)
    if (aligned_height, aligned_width) != (args.height, args.width):
        print(
            "[size_align] Adjusting generation size from "
            f"{args.height}x{args.width} to {aligned_height}x{aligned_width}."
        )
        args.height = aligned_height
        args.width = aligned_width

    args.requested_output_frames = int(args.num_frames)
    aligned_num_frames = core.align_generation_num_frames(args.num_frames)
    if aligned_num_frames != args.num_frames:
        print(
            "[time_align] Adjusting generation length from "
            f"{args.num_frames} to {aligned_num_frames} to satisfy 4n+1, "
            f"while saving only the first {args.requested_output_frames} frames."
        )
        args.num_frames = aligned_num_frames

    generated_dir = args.output_root
    metadata_dir = args.runtime_root / "metadata" / args.model_name
    summary_json_path = args.runtime_root / "summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    original_build_pipeline = core.build_pipeline
    original_build_method_name = core.build_method_name
    core.build_pipeline = _build_pipeline_with_vjepa(cli_args)
    core.build_method_name = lambda lora_path: f"{original_build_method_name(lora_path)}_vjepa"
    try:
        effective_num_shards = args.num_shards
        if args.multi_gpu and not args.worker:
            effective_num_shards = core.launch_multi_gpu_workers(args, generated_dir, metadata_dir)
        else:
            run_generation_with_optional_trace(args, cli_args, generated_dir, metadata_dir)
    finally:
        core.build_pipeline = original_build_pipeline
        core.build_method_name = original_build_method_name

    merged_jsonl = core.merge_shard_jsonl_files(metadata_dir, args.model_name, effective_num_shards)
    summary_entries_path = merged_jsonl
    if summary_entries_path is None:
        summary_entries_path = core.per_case_jsonl_path(
            metadata_dir,
            args.model_name,
            effective_num_shards,
            args.shard_id,
        )
    summary_entries = core.load_jsonl(summary_entries_path)
    eval_csv_path = core.infer_eval_csv_path(args.output_root, args.model_name)
    summary_entries, num_entries_with_metrics = core.augment_entries_with_eval_metrics(
        summary_entries,
        eval_csv_path,
    )
    if summary_entries_path is not None and summary_entries:
        core.write_jsonl(summary_entries_path, summary_entries)
    payload = {
        "model_name": args.model_name,
        "weights_root": str(args.weights_root),
        "lora_path": str(args.lora_path),
        "generated_dir": str(generated_dir),
        "metadata_dir": str(metadata_dir),
        "runtime_root": str(args.runtime_root),
        "input_json_list_path": str(args.input_json_list_path),
        "eval_csv": str(eval_csv_path) if eval_csv_path is not None else None,
        "num_entries_with_metrics": num_entries_with_metrics,
        "summary": core.build_summary(summary_entries),
        "selected_videos": core.find_selected_video_paths(generated_dir, summary_entries),
        "vjepa": {
            "enabled": not cli_args.disable_vjepa_guidance,
            "device": cli_args.vjepa_device or cli_args.device,
            "model": cli_args.vjepa_model,
            "ckpt": str(cli_args.vjepa_ckpt.expanduser().resolve()) if cli_args.vjepa_ckpt is not None else None,
            "config": vars(build_vjepa_config(cli_args)),
        },
    }
    core.write_json(summary_json_path, payload)
    print(summary_json_path)


if __name__ == "__main__":
    main()
