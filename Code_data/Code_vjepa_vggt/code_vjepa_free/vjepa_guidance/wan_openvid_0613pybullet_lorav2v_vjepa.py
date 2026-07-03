from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from code_vjepa_vggt.AAAinfer.utils.named_paths import resolve_output_root, resolve_runtime_root
from code_vjepa_vggt.train0419_reference import batch_eval_lora as core

try:
    from .vjepa_surprise import VJEPASurpriseEnergy
    from .wan_latent_guidance import WanVJEPAConfig, pick_guidance_step_indices
except ImportError:
    from vjepa_surprise import VJEPASurpriseEnergy
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


def _apply_diffsynth_vjepa_guidance(
    *,
    latent_xt: torch.Tensor,
    model_output: torch.Tensor,
    timestep: torch.Tensor | int,
    scheduler,
    preview_decoder,
    energy_fn,
    config: WanVJEPAConfig,
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

        for parameter in self.vae.model.parameters():
            parameter.requires_grad_(False)
        self.vae.model.eval()

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
        selected_steps = set(
            pick_guidance_step_indices(
                total_steps=len(self.scheduler.timesteps),
                count=self.vjepa_config.guidance_steps,
                min_step_percent=self.vjepa_config.min_step_percent,
                max_step_percent=self.vjepa_config.max_step_percent,
            )
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
                with torch.enable_grad():
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
                    )
                logging.info(
                    "V-JEPA step=%d timestep=%d energy=%.6f grad_rms=%.6f preview=%dx%dx%d",
                    progress_id,
                    int(timestep_cpu.item()),
                    stats["energy"],
                    stats["grad_rms"],
                    int(stats["preview_frames"]),
                    int(stats["preview_height"]),
                    int(stats["preview_width"]),
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
    parser.add_argument("--vjepa-latent-step-size", type=float, default=0.01)
    parser.add_argument("--vjepa-preview-downsample-factor", type=int, default=4)
    parser.add_argument("--vjepa-preview-frame-stride", type=int, default=2)
    parser.add_argument("--vjepa-window-size", type=int, default=16)
    parser.add_argument("--vjepa-context-frames", type=int, default=8)
    parser.add_argument("--vjepa-stride", type=int, default=4)
    parser.add_argument("--vjepa-reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--vjepa-grad-norm-mode", choices=["rms", "l2", "none"], default="rms")
    parser.add_argument("--vjepa-max-grad-norm", type=float, default=10.0)
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
        return ContextAwareWanVideoPipelineVJEPA.from_pretrained_vjepa(
            wan_root=wan_root,
            device=device,
            lora_path=lora_path,
            vjepa_model_name=str(cli_args.vjepa_model),
            vjepa_checkpoint_path=cli_args.vjepa_ckpt.expanduser().resolve() if cli_args.vjepa_ckpt is not None else None,
            vjepa_device=vjepa_device,
            vjepa_config=vjepa_config,
            enable_vjepa_guidance=not cli_args.disable_vjepa_guidance,
        )

    return _builder


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
            core.run_generation(args, generated_dir, metadata_dir)
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
