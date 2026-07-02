from __future__ import annotations

"""
Wan2.1 T2V 1.3B is text-to-video only. It does not accept a first-frame image.

Guided run:

CUDA_VISIBLE_DEVICES=0,5 /data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_vjepa.py \
    --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
    --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
    --output /data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/wan21_1p3b_vjepa.mp4 \
    --height 480 \
    --width 832 \
    --num_frames 24 \
    --num_inference_steps 10 \
    --guidance_scale 6 \
    --flow_shift 8 \
    --device_id 1 \
    --vjepa_device_id 0 \
    --vjepa_model vith \
    --vjepa_ckpt /data/gaoya/ckpt/VJEPA2/vith.pt

Baseline run:

CUDA_VISIBLE_DEVICES=0 /data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_vjepa.py \
    --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
    --prompt "Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement." \
    --output /data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/wan21_1p3b_baseline.mp4 \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --num_inference_steps 10 \
    --guidance_scale 6 \
    --flow_shift 8 \
    --device_id 0 \
    --disable_vjepa_guidance
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from transformers import AutoTokenizer, UMT5EncoderModel

try:
    from .vjepa_surprise import VJEPASurpriseEnergy
    from .wan_latent_guidance import (
        WanVJEPAConfig,
        apply_vjepa_latent_guidance_with_decoder,
        pick_guidance_step_indices,
    )
except ImportError:
    from vjepa_surprise import VJEPASurpriseEnergy
    from wan_latent_guidance import (
        WanVJEPAConfig,
        apply_vjepa_latent_guidance_with_decoder,
        pick_guidance_step_indices,
    )


DEFAULT_CKPT_DIR = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers")
DEFAULT_VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
DEFAULT_OUTPUT_PATH = Path("/data/gaoya/agent-data/outputs/vjepa_guidance/wan21_t2v_1_3b_vjepa.mp4")
DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
    "fight intensely on a spotlighted stage."
)


def parse_torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_id: Optional[int]) -> torch.device:
    if torch.cuda.is_available():
        visible_count = torch.cuda.device_count()
        if device_id is None:
            return torch.device("cuda")
        if device_id < 0 or device_id >= visible_count:
            raise ValueError(
                f"Requested device_id={device_id}, but only {visible_count} CUDA device(s) are visible. "
                "Check CUDA_VISIBLE_DEVICES and the local device index."
            )
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


class WanT2V13BVJEPA(WanPipeline):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,
        *,
        vjepa_model_name: str = "vith",
        vjepa_checkpoint_path: Optional[str] = None,
        vjepa_device: Optional[str | torch.device] = None,
        vjepa_config: Optional[WanVJEPAConfig] = None,
        enable_vjepa_guidance: bool = True,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            transformer=transformer,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )
        self.vjepa_model_name = vjepa_model_name
        self.vjepa_checkpoint_path = vjepa_checkpoint_path
        self.vjepa_device = torch.device(vjepa_device) if vjepa_device is not None else self._execution_device
        self.vjepa_config = vjepa_config or WanVJEPAConfig()
        self.enable_vjepa_guidance = enable_vjepa_guidance
        self._vjepa_energy: VJEPASurpriseEnergy | None = None

        for parameter in self.vae.parameters():
            parameter.requires_grad_(False)
        self.vae.eval()

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

    def _vae_latent_stats(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=dtype)
        latents_mean = latents_mean.view(1, self.vae.config.z_dim, 1, 1, 1)
        latents_std = latents_std.view(1, self.vae.config.z_dim, 1, 1, 1)
        return latents_mean, latents_std

    def _destandardize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean, latents_std = self._vae_latent_stats(device=latents.device, dtype=latents.dtype)
        return latents * latents_std + latents_mean

    def _decode_preview_video(
        self,
        x0_latent: torch.Tensor,
        *,
        preview_downsample_factor: int = 2,
        preview_frame_stride: int = 1,
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

        vae_device = next(self.vae.parameters()).device
        preview_latent = preview_latent.to(device=vae_device, dtype=self.vae.dtype)
        preview_latent = self._destandardize_latents(preview_latent)
        preview_video = self.vae.decode(preview_latent, return_dict=False)[0]
        if preview_video.ndim != 5:
            raise RuntimeError(f"Expected preview decode [B,C,T,H,W], got {tuple(preview_video.shape)}")
        return preview_video

    def _decode_final_video(self, latents: torch.Tensor, output_type: str):
        vae_device = next(self.vae.parameters()).device
        latents = latents.to(device=vae_device, dtype=self.vae.dtype)
        latents = self._destandardize_latents(latents)
        video = self.vae.decode(latents, return_dict=False)[0]
        return self.video_processor.postprocess_video(video, output_type=output_type)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 6.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict[str, Any]], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        if not self.enable_vjepa_guidance or self.vjepa_config.guidance_steps <= 0:
            logging.info("V-JEPA guidance disabled; using default WanPipeline sampling.")
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                guidance_scale_2=guidance_scale_2,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logging.warning(
                "`num_frames - 1` has to be divisible by %d. Rounding to the nearest valid value.",
                self.vae_scale_factor_temporal,
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        selected_steps = set(
            pick_guidance_step_indices(
                total_steps=len(timesteps),
                count=self.vjepa_config.guidance_steps,
                min_step_percent=self.vjepa_config.min_step_percent,
                max_step_percent=self.vjepa_config.max_step_percent,
            )
        )
        energy_fn = self._ensure_vjepa_energy()
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        mask = torch.ones_like(latents, dtype=torch.float32, device=device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_idx, timestep in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = timestep

                if boundary_timestep is None or timestep >= boundary_timestep:
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                else:
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2

                latent_model_input = latents.to(transformer_dtype)
                if self.config.expand_timesteps:
                    temp_ts = (mask[0][0][:, ::2, ::2] * timestep).flatten()
                    timestep_input = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep_input = timestep.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep_input,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep_input,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                if step_idx in selected_steps:
                    with torch.enable_grad():
                        latents, stats = apply_vjepa_latent_guidance_with_decoder(
                            latent_xt=latents,
                            model_output=noise_pred,
                            timestep=timestep,
                            scheduler=self.scheduler,
                            preview_decoder=self._decode_preview_video,
                            energy_fn=energy_fn,
                            config=self.vjepa_config,
                        )
                    logging.info(
                        "V-JEPA step=%d timestep=%d energy=%.6f grad_rms=%.6f preview=%dx%dx%d",
                        step_idx,
                        int(timestep.item()),
                        stats["energy"],
                        stats["grad_rms"],
                        int(stats["preview_frames"]),
                        int(stats["preview_height"]),
                        int(stats["preview_width"]),
                    )

                latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for key in callback_on_step_end_tensor_inputs:
                        callback_kwargs[key] = locals()[key]
                    callback_outputs = callback_on_step_end(self, step_idx, timestep, callback_kwargs)
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop(
                            "negative_prompt_embeds",
                            negative_prompt_embeds,
                        )

                if step_idx == len(timesteps) - 1 or (
                    (step_idx + 1) > num_warmup_steps and (step_idx + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        self._current_timestep = None

        if output_type != "latent":
            video = self._decode_final_video(latents, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan2.1 T2V 1.3B with optional V-JEPA latent guidance.")
    parser.add_argument("--ckpt_dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--vjepa_device_id", type=int, default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--transformer_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--vae_dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--disable_vjepa_guidance", action="store_true")
    parser.add_argument("--vjepa_model", type=str, default="vith")
    parser.add_argument("--vjepa_ckpt", type=Path, default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--vjepa_guidance_steps", type=int, default=6)
    parser.add_argument("--vjepa_min_step_percent", type=float, default=0.2)
    parser.add_argument("--vjepa_max_step_percent", type=float, default=0.8)
    parser.add_argument("--vjepa_latent_step_size", type=float, default=0.02)
    parser.add_argument("--preview_downsample_factor", type=int, default=2)
    parser.add_argument("--preview_frame_stride", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--reduction", choices=["mean", "max"], default="mean")
    parser.add_argument("--gradient_normalization", choices=["rms", "l2", "none"], default="rms")
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--log_level", type=str, default="INFO")
    return parser


def build_vjepa_config(args: argparse.Namespace) -> WanVJEPAConfig:
    max_grad_norm = args.max_grad_norm
    if max_grad_norm is not None and max_grad_norm <= 0:
        max_grad_norm = None

    return WanVJEPAConfig(
        guidance_steps=args.vjepa_guidance_steps,
        min_step_percent=args.vjepa_min_step_percent,
        max_step_percent=args.vjepa_max_step_percent,
        latent_step_size=args.vjepa_latent_step_size,
        preview_downsample_factor=args.preview_downsample_factor,
        preview_frame_stride=args.preview_frame_stride,
        window_size=args.window_size,
        context_frames=args.context_frames,
        stride=args.stride,
        reduction=args.reduction,
        gradient_normalization=args.gradient_normalization,
        max_grad_norm=max_grad_norm,
    )


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.ckpt_dir.is_dir():
        raise FileNotFoundError(f"Wan checkpoint dir not found: {args.ckpt_dir}")
    if not args.disable_vjepa_guidance and not args.vjepa_ckpt.exists():
        raise FileNotFoundError(f"V-JEPA checkpoint not found: {args.vjepa_ckpt}")

    set_seed(args.seed)

    main_device = resolve_device(args.device_id)
    vjepa_device = resolve_device(args.vjepa_device_id) if args.vjepa_device_id is not None else main_device
    transformer_dtype = parse_torch_dtype(args.transformer_dtype)
    vae_dtype = parse_torch_dtype(args.vae_dtype)

    logging.info("Main device: %s", main_device)
    logging.info("V-JEPA device: %s", vjepa_device)
    logging.info("Loading Wan2.1 T2V 1.3B from %s", args.ckpt_dir)

    vae = AutoencoderKLWan.from_pretrained(
        str(args.ckpt_dir),
        subfolder="vae",
        torch_dtype=vae_dtype,
        local_files_only=True,
    )
    base_pipe = WanPipeline.from_pretrained(
        str(args.ckpt_dir),
        vae=vae,
        torch_dtype=transformer_dtype,
        local_files_only=True,
    )
    pipe = WanT2V13BVJEPA(
        tokenizer=base_pipe.tokenizer,
        text_encoder=base_pipe.text_encoder,
        vae=base_pipe.vae,
        scheduler=base_pipe.scheduler,
        transformer=base_pipe.transformer,
        transformer_2=getattr(base_pipe, "transformer_2", None),
        boundary_ratio=getattr(base_pipe.config, "boundary_ratio", None),
        expand_timesteps=getattr(base_pipe.config, "expand_timesteps", False),
        vjepa_model_name=args.vjepa_model,
        vjepa_checkpoint_path=str(args.vjepa_ckpt),
        vjepa_device=vjepa_device,
        vjepa_config=build_vjepa_config(args),
        enable_vjepa_guidance=not args.disable_vjepa_guidance,
    )
    del base_pipe
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=args.flow_shift)

    if args.cpu_offload:
        if main_device.type != "cuda":
            raise ValueError("--cpu_offload requires CUDA.")
        pipe.enable_model_cpu_offload(gpu_id=args.device_id)
    else:
        pipe.to(main_device)
        if not args.disable_vjepa_guidance and vjepa_device != main_device:
            logging.info("Moving VAE to %s for preview decode and final decode.", vjepa_device)
            pipe.vae.to(vjepa_device)

    generator_device = main_device.type if main_device.type == "cpu" else str(main_device)
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)

    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="pil",
        max_sequence_length=args.max_sequence_length,
    )

    frames = output.frames[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    saved_path = export_to_video(frames, output_video_path=str(args.output), fps=args.fps)
    logging.info("Saved video to %s", saved_path)


if __name__ == "__main__":
    main()
