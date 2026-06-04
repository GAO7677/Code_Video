from __future__ import annotations

import gc
import math
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import require_torch
from .wan_state_v2_helpers import (
    align_wan_frame_num,
    apply_clean_prefix_to_latents,
    build_state_condition_payload_from_condition_maps,
    filter_state_condition_payload_for_adapter,
    flatten_condition_maps_to_state_tokens,
    resample_condition_maps_to_steps,
    resample_state_tokens_to_steps,
)

torch = require_torch()


DEFAULT_WAN_REPO_ROOT = Path("/home/gaoya/Code_Video/Wan2.2-main")


def resolve_wan_repo_root(wan_repo_root: str | Path | None = None) -> Path:
    root = Path(wan_repo_root) if wan_repo_root is not None else DEFAULT_WAN_REPO_ROOT
    if not root.exists():
        raise FileNotFoundError(f"Wan repo root does not exist: {root}")
    return root


def _ensure_wan_importable(wan_repo_root: str | Path | None = None) -> Path:
    root = resolve_wan_repo_root(wan_repo_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def load_wan_modules(wan_repo_root: str | Path | None = None) -> dict[str, Any]:
    _ensure_wan_importable(wan_repo_root)

    from wan_.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    from wan_.image2video import WanI2V
    from wan_.modules.t5 import T5EncoderModel
    from wan_.modules.vae2_1 import Wan2_1_VAE
    from wan_.modules.vae2_2 import Wan2_2_VAE

    return {
        "WAN_CONFIGS": WAN_CONFIGS,
        "SIZE_CONFIGS": SIZE_CONFIGS,
        "MAX_AREA_CONFIGS": MAX_AREA_CONFIGS,
        "SUPPORTED_SIZES": SUPPORTED_SIZES,
        "WanI2V": WanI2V,
        "T5EncoderModel": T5EncoderModel,
        "Wan2_1_VAE": Wan2_1_VAE,
        "Wan2_2_VAE": Wan2_2_VAE,
    }


def _normalize_video_range(frames: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(frames):
        if float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0:
            return frames * 2.0 - 1.0
        return frames.clamp(-1.0, 1.0)
    return frames.float().div(127.5).sub(1.0).clamp(-1.0, 1.0)


def _build_prefix_condition_mask(
    total_frames: int,
    context_steps: int,
    lat_h: int,
    lat_w: int,
    device,
) -> torch.Tensor:
    if total_frames <= 0:
        raise ValueError(f"total_frames must be positive, got {total_frames}")
    if context_steps <= 0 or context_steps > total_frames:
        raise ValueError(f"context_steps must be in [1, total_frames], got {context_steps} for {total_frames}")
    mask = torch.zeros(1, total_frames, lat_h, lat_w, device=device)
    mask[:, :context_steps] = 1
    mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
    mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
    return mask.transpose(1, 2)[0]


def _align_wan_frame_num(frame_num: int) -> int:
    return align_wan_frame_num(frame_num)


def _build_prefix_latent_noise_mask(
    noise_latent: torch.Tensor,
    context_steps: int,
    temporal_stride: int,
) -> torch.Tensor:
    if noise_latent.ndim != 4:
        raise ValueError(f"expected noise latent with shape [C, T, H, W], got {tuple(noise_latent.shape)}")
    if temporal_stride <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    keep_steps = 1 + max(context_steps - 1, 0) // temporal_stride
    keep_steps = min(keep_steps, noise_latent.shape[1])
    noise_mask = torch.ones_like(noise_latent)
    noise_mask[:, :keep_steps] = 0
    return noise_mask


def _apply_clean_prefix_to_latent(
    latents: torch.Tensor,
    clean_prefix_latents: torch.Tensor,
) -> torch.Tensor:
    return apply_clean_prefix_to_latents(latents, clean_prefix_latents)


def _build_separated_timestep(
    timestep_value: torch.Tensor,
    seq_len: int,
    latent_steps: int,
    prefix_len: int,
    lat_h: int,
    lat_w: int,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if latent_steps <= 0:
        raise ValueError(f"latent_steps must be positive, got {latent_steps}")
    token_count_per_latent = lat_h * lat_w // 4
    if token_count_per_latent <= 0:
        raise ValueError(f"invalid latent grid for Wan patching: lat_h={lat_h}, lat_w={lat_w}")
    clean_tokens = prefix_len * token_count_per_latent
    future_tokens = (latent_steps - prefix_len) * token_count_per_latent
    tokens = torch.cat(
        [
            torch.zeros((clean_tokens,), dtype=torch.float32, device=timestep_value.device),
            torch.full((future_tokens,), float(timestep_value.item()), dtype=torch.float32, device=timestep_value.device),
        ]
    )
    if tokens.numel() > seq_len:
        raise ValueError(f"timestep token count {tokens.numel()} exceeds seq_len {seq_len}")
    if tokens.numel() < seq_len:
        tokens = torch.cat([tokens, tokens.new_full((seq_len - tokens.numel(),), float(timestep_value.item()))])
    return tokens.unsqueeze(0)


def _pad_future_state_tokens(state_tokens: torch.Tensor, target_steps: int) -> torch.Tensor:
    if state_tokens.ndim != 3:
        raise ValueError(f"expected state tokens with shape [B, T, D], got {tuple(state_tokens.shape)}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if state_tokens.shape[1] == target_steps:
        return state_tokens
    if state_tokens.shape[1] > target_steps:
        return state_tokens[:, :target_steps]
    pad_steps = target_steps - state_tokens.shape[1]
    pad_source = state_tokens[:, -1:] if state_tokens.shape[1] > 0 else torch.zeros(
        state_tokens.shape[0],
        1,
        state_tokens.shape[2],
        dtype=state_tokens.dtype,
        device=state_tokens.device,
    )
    return torch.cat([state_tokens, pad_source.expand(-1, pad_steps, -1)], dim=1)


def _resample_state_tokens_to_steps(state_tokens: torch.Tensor, target_steps: int) -> torch.Tensor:
    return resample_state_tokens_to_steps(state_tokens, target_steps)


def _resample_condition_maps_to_steps(condition_maps: torch.Tensor, target_steps: int) -> torch.Tensor:
    return resample_condition_maps_to_steps(condition_maps, target_steps)


def _flatten_condition_maps_to_state_tokens(condition_maps: torch.Tensor) -> torch.Tensor:
    return flatten_condition_maps_to_state_tokens(condition_maps)


def _filter_state_condition_payload_for_adapter(
    state_condition: dict[str, torch.Tensor],
    adapter,
) -> dict[str, torch.Tensor]:
    return filter_state_condition_payload_for_adapter(state_condition, adapter)


def _resample_video_latents_to_frame_steps(latent_clip: torch.Tensor, target_steps: int) -> torch.Tensor:
    if latent_clip.ndim != 4:
        raise ValueError(f"expected latent clip with shape [C, T, H, W], got {tuple(latent_clip.shape)}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    channels, clip_steps, lat_h, lat_w = latent_clip.shape
    flattened = latent_clip.permute(0, 2, 3, 1).contiguous().view(1, channels * lat_h * lat_w, clip_steps)
    resized = torch.nn.functional.interpolate(
        flattened,
        size=target_steps,
        mode="linear",
        align_corners=False,
    )
    resized = resized.view(channels, lat_h, lat_w, target_steps)
    return resized.permute(3, 0, 1, 2).contiguous()


def _resize_video_frames(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"expected frames with shape [T, 3, H, W], got {tuple(frames.shape)}")
    return torch.nn.functional.interpolate(
        frames.float(),
        size=(out_h, out_w),
        mode="bicubic",
        align_corners=False,
    ).to(frames.dtype)


def _encode_video_prefix_latents(
    vae,
    frames: torch.Tensor,
) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"expected video frames with shape [T, 3, H, W], got {tuple(frames.shape)}")
    video_cthw = frames.permute(1, 0, 2, 3).contiguous()
    return vae.encode([video_cthw])[0]


@dataclass(slots=True)
class WanLatentExtractor:
    ckpt_dir: str | Path
    wan_repo_root: str | Path | None = None
    task: str = "i2v-A14B"
    device: str = "cuda"

    def __post_init__(self) -> None:
        modules = load_wan_modules(self.wan_repo_root)
        self._wan_configs = modules["WAN_CONFIGS"]
        if self.task not in self._wan_configs:
            raise ValueError(f"unsupported Wan task: {self.task}")
        task_config = self._wan_configs[self.task]
        vae_checkpoint = task_config.vae_checkpoint
        vae_dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        vae_class = modules["Wan2_1_VAE"]
        if "2.2" in str(vae_checkpoint):
            vae_class = modules["Wan2_2_VAE"]
        self.vae = vae_class(vae_pth=str(Path(self.ckpt_dir) / vae_checkpoint), dtype=vae_dtype, device=self.device)
        self.temporal_stride = int(task_config.vae_stride[0])

    def encode_context_frames_raw(self, context_frames: torch.Tensor) -> torch.Tensor:
        if context_frames.ndim != 5:
            raise ValueError(
                f"expected context frames with shape [B, K, 3, H, W], got {tuple(context_frames.shape)}"
            )
        normalized = _normalize_video_range(context_frames.to(self.device))
        videos = [normalized[b].permute(1, 0, 2, 3).contiguous() for b in range(context_frames.shape[0])]
        with torch.no_grad():
            encoded = self.vae.encode(videos)

        latent_slices = []
        for latent in encoded:
            if latent.ndim != 4:
                raise ValueError(f"expected Wan latent with shape [C, T, H, W], got {tuple(latent.shape)}")
            latent_slices.append(latent.permute(1, 0, 2, 3).contiguous())
        return torch.stack(latent_slices, dim=0)

    def encode_context_frames(self, context_frames: torch.Tensor) -> torch.Tensor:
        batch, context_steps = context_frames.shape[:2]
        raw_latents = self.encode_context_frames_raw(context_frames)
        aligned = []
        for latent in raw_latents:
            latent_cthw = latent.permute(1, 0, 2, 3).contiguous()
            aligned.append(_resample_video_latents_to_frame_steps(latent_cthw, context_steps))
        return torch.stack(aligned, dim=0)


@dataclass(slots=True)
class WanImageToVideoBackend:
    ckpt_dir: str | Path
    wan_repo_root: str | Path | None = None
    task: str = "i2v-A14B"
    device: str = "cuda"
    state_adapter_ckpt: str | Path | None = None

    def __post_init__(self) -> None:
        if not str(self.device).startswith("cuda"):
            raise ValueError("WanImageToVideoBackend currently requires a CUDA device")
        modules = load_wan_modules(self.wan_repo_root)
        self._wan_configs = modules["WAN_CONFIGS"]
        self._max_area_configs = modules["MAX_AREA_CONFIGS"]
        self._supported_sizes = modules["SUPPORTED_SIZES"]
        if self.task not in self._wan_configs:
            raise ValueError(f"unsupported Wan task: {self.task}")
        device_id = int(str(self.device).split(":")[1]) if ":" in str(self.device) else 0
        self.pipeline = modules["WanI2V"](
            config=self._wan_configs[self.task],
            checkpoint_dir=str(self.ckpt_dir),
            device_id=device_id,
            rank=0,
        )

    def generate(
        self,
        prompt: str,
        context_frames: torch.Tensor,
        size: str,
        frame_num: int,
        state_tokens: torch.Tensor | None = None,
        memory_tokens: torch.Tensor | None = None,
        condition_maps: torch.Tensor | None = None,
        sample_solver: str = "unipc",
        sampling_steps: int = 40,
        guide_scale: float = 5.0,
        shift: float = 5.0,
        negative_prompt: str = "",
        seed: int = -1,
        offload_model: bool = True,
        state_scale: float = 1.0,
        state_guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        if size not in self._supported_sizes[self.task]:
            raise ValueError(
                f"unsupported size '{size}' for task '{self.task}', expected one of {self._supported_sizes[self.task]}"
            )
        if context_frames.ndim != 4:
            raise ValueError(f"expected context frames with shape [K, 3, H, W], got {tuple(context_frames.shape)}")
        if context_frames.shape[0] >= frame_num:
            raise ValueError(
                f"frame_num must be larger than context length so there is future to generate, got {frame_num=} and K={context_frames.shape[0]}"
            )
        if state_tokens is None and memory_tokens is None and condition_maps is None:
            raise ValueError("at least one of state_tokens, memory_tokens, condition_maps must be provided")
        condition_tokens = None if state_tokens is None else state_tokens.detach()
        if condition_tokens is not None and condition_tokens.ndim == 2:
            condition_tokens = condition_tokens.unsqueeze(0)
        condition_memory = None if memory_tokens is None else memory_tokens.detach()
        if condition_memory is not None and condition_memory.ndim == 2:
            condition_memory = condition_memory.unsqueeze(0)
        condition_maps_tensor = None if condition_maps is None else condition_maps.detach()
        if condition_maps_tensor is not None and condition_maps_tensor.ndim == 4:
            condition_maps_tensor = condition_maps_tensor.unsqueeze(0)
        if negative_prompt == "":
            negative_prompt = self.pipeline.sample_neg_prompt

        context_steps = int(context_frames.shape[0])
        requested_total_frames = int(frame_num)
        total_frames = _align_wan_frame_num(requested_total_frames)
        max_area = self._max_area_configs[size]
        context_frames = context_frames.to(self.pipeline.device)
        normalized_context = _normalize_video_range(context_frames)
        if condition_tokens is not None:
            condition_tokens = condition_tokens.to(self.pipeline.device)
        if condition_memory is not None:
            condition_memory = condition_memory.to(self.pipeline.device)
        if condition_maps_tensor is not None:
            condition_maps_tensor = condition_maps_tensor.to(self.pipeline.device)

        height, width = normalized_context.shape[-2:]
        aspect_ratio = height / width
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.pipeline.vae_stride[1] // self.pipeline.patch_size[1]
            * self.pipeline.patch_size[1]
        )
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.pipeline.vae_stride[2] // self.pipeline.patch_size[2]
            * self.pipeline.patch_size[2]
        )
        out_h = lat_h * self.pipeline.vae_stride[1]
        out_w = lat_w * self.pipeline.vae_stride[2]
        seq_len = ((total_frames - 1) // self.pipeline.vae_stride[0] + 1) * lat_h * lat_w // (
            self.pipeline.patch_size[1] * self.pipeline.patch_size[2]
        )
        seq_len = int(math.ceil(seq_len / self.pipeline.sp_size)) * self.pipeline.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.pipeline.device)
        seed_g.manual_seed(seed)

        noise = torch.randn(
            self.pipeline.vae.model.z_dim,
            (total_frames - 1) // self.pipeline.vae_stride[0] + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.pipeline.device,
        )

        resized_context = _resize_video_frames(normalized_context, out_h=out_h, out_w=out_w)
        clean_prefix_latents = _encode_video_prefix_latents(self.pipeline.vae, resized_context)
        prefix_len = int(clean_prefix_latents.shape[1])
        if prefix_len >= noise.shape[1]:
            raise ValueError(
                "context prefix covers all Wan latent steps, leaving no future latent step to denoise: "
                f"prefix_len={prefix_len}, latent_steps={noise.shape[1]}"
            )
        future_latent_steps = int(noise.shape[1] - prefix_len)
        if condition_maps_tensor is not None:
            condition_maps_tensor = _resample_condition_maps_to_steps(condition_maps_tensor, future_latent_steps)
            condition_tokens = _flatten_condition_maps_to_state_tokens(condition_maps_tensor)
        elif condition_tokens is not None:
            raise ValueError(
                "flattened state_tokens-only conditioning is no longer supported for wan_state_v2 clean-prefix "
                "inference because it loses spatial token layout. Pass condition_maps instead."
            )

        i2v_video = torch.zeros(
            3,
            total_frames,
            out_h,
            out_w,
            dtype=resized_context.dtype,
            device=self.pipeline.device,
        )
        i2v_video[:, :1] = resized_context[:1].permute(1, 0, 2, 3)
        i2v_latent = self.pipeline.vae.encode([i2v_video])[0]
        i2v_mask = _build_prefix_condition_mask(
            total_frames=total_frames,
            context_steps=1,
            lat_h=lat_h,
            lat_w=lat_w,
            device=self.pipeline.device,
        ).to(i2v_latent.dtype)
        y = torch.concat([i2v_mask, i2v_latent], dim=0)

        latent = _apply_clean_prefix_to_latent(noise, clean_prefix_latents)

        if not self.pipeline.t5_cpu:
            self.pipeline.text_encoder.model.to(self.pipeline.device)
            context = self.pipeline.text_encoder([prompt], self.pipeline.device)
            context_null = self.pipeline.text_encoder([negative_prompt], self.pipeline.device)
            if offload_model:
                self.pipeline.text_encoder.model.cpu()
        else:
            context = self.pipeline.text_encoder([prompt], torch.device("cpu"))
            context_null = self.pipeline.text_encoder([negative_prompt], torch.device("cpu"))
            context = [item.to(self.pipeline.device) for item in context]
            context_null = [item.to(self.pipeline.device) for item in context_null]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.pipeline.low_noise_model, "no_sync", noop_no_sync)
        no_sync_high_noise = getattr(self.pipeline.high_noise_model, "no_sync", noop_no_sync)

        with (
            torch.amp.autocast("cuda", dtype=self.pipeline.param_dtype),
            torch.no_grad(),
            no_sync_low_noise(),
            no_sync_high_noise(),
        ):
            if self.state_adapter_ckpt is None and self.pipeline.state_adapter is None:
                raise ValueError(
                    "state_condition was provided but no trained state adapter is loaded. "
                    "Pass state_adapter_ckpt when constructing WanImageToVideoBackend."
                )
            if condition_maps_tensor is None:
                raise ValueError(
                    "condition_maps is required for wan_state_v2 clean-prefix inference; state_tokens is now only a "
                    "derived compatibility view."
                )
            state_condition_payload = build_state_condition_payload_from_condition_maps(
                condition_maps_tensor,
                memory_tokens=condition_memory,
                include_condition_maps=True,
            )
            if self.state_adapter_ckpt is not None and self.pipeline.state_adapter is None:
                self.pipeline.load_state_adapter(
                    str(self.state_adapter_ckpt),
                    state_condition=state_condition_payload,
                )
            state_condition_payload = _filter_state_condition_payload_for_adapter(
                state_condition_payload,
                self.pipeline.state_adapter,
            )
            state_context = self.pipeline._build_state_context(
                state_condition_payload,
                offload_model,
            )
            boundary = self.pipeline.boundary * self.pipeline.num_train_timesteps

            if sample_solver == "unipc":
                from wan_.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.pipeline.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(sampling_steps, device=self.pipeline.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                from wan_.utils.fm_solvers import (
                    FlowDPMSolverMultistepScheduler,
                    get_sampling_sigmas,
                    retrieve_timesteps,
                )

                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.pipeline.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.pipeline.device, sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            arg_c = {
                "context": [context[0]],
                "seq_len": seq_len,
                "y": [y],
                "state_context": state_context,
                "state_scale": state_scale,
            }
            arg_text_only = {
                "context": [context[0]],
                "seq_len": seq_len,
                "y": [y],
                "state_context": None,
                "state_scale": 0.0,
            }
            arg_null = {
                "context": context_null,
                "seq_len": seq_len,
                "y": [y],
                "state_context": None,
                "state_scale": 0.0,
            }

            if offload_model:
                torch.cuda.empty_cache()

            for timestep_value in timesteps:
                latent_model_input = [latent.to(self.pipeline.device)]
                token_timestep = _build_separated_timestep(
                    timestep_value=timestep_value.to(self.pipeline.device),
                    seq_len=seq_len,
                    latent_steps=latent.shape[1],
                    prefix_len=prefix_len,
                    lat_h=latent.shape[2],
                    lat_w=latent.shape[3],
                )

                model = self.pipeline._prepare_model_for_timestep(timestep_value, boundary, offload_model)
                sample_guide_scale = guide_scale
                if isinstance(guide_scale, tuple):
                    sample_guide_scale = guide_scale[1] if timestep_value.item() >= boundary else guide_scale[0]

                noise_pred_text_state = model(latent_model_input, t=token_timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_text_only = model(latent_model_input, t=token_timestep, **arg_text_only)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = model(latent_model_input, t=token_timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = (
                    noise_pred_uncond
                    + sample_guide_scale * (noise_pred_text_only - noise_pred_uncond)
                    + state_guidance_scale * (noise_pred_text_state - noise_pred_text_only)
                )

                latent = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    timestep_value,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0].squeeze(0)
                latent = _apply_clean_prefix_to_latent(latent, clean_prefix_latents)

            if offload_model:
                self.pipeline.low_noise_model.cpu()
                self.pipeline.high_noise_model.cpu()
                if self.pipeline.state_adapter is not None:
                    self.pipeline.state_adapter.cpu()
                torch.cuda.empty_cache()

            video = self.pipeline.vae.decode([latent])[0]

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        return video[:, :requested_total_frames].detach().cpu()
