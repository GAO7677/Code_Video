from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .utils import require_torch
from .wan_runtime import ensure_wan_importable
from .wan_state_condition_bundles import (
    REQUIRED_STATE_ADAPTER_KEYS_I2V,
    REQUIRED_STATE_ADAPTER_KEYS_TI2V,
    StateConditionBundleRecord,
    discover_state_condition_bundles,
    is_i2v_state_adapter_checkpoint,
    is_ti2v_state_adapter_checkpoint,
    load_episode_npz,
    load_state_condition_npz,
)
from .wan_state_v2_helpers import (
    align_wan_frame_num as align_wan_frame_num_shared,
    apply_clean_prefix_to_latents as apply_clean_prefix_to_latents_shared,
    build_prefix_latent_mask as build_prefix_latent_mask_shared,
    build_prefix_timestep_tensor as build_prefix_timestep_tensor_shared,
    filter_state_condition_payload_for_adapter as filter_state_condition_payload_for_adapter_shared,
    normalize_video_range_shared,
    preprocess_ti2v_prefix_frames_shared,
    resize_and_center_crop_frames_shared,
    resample_condition_maps_to_steps as resample_condition_maps_to_steps_shared,
)

torch = require_torch()
F = torch.nn.functional


def align_wan_frame_num(frame_num: int) -> int:
    return align_wan_frame_num_shared(frame_num)


def to_frame_tensor(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(frames, torch.Tensor):
        tensor = frames.float()
    else:
        tensor = torch.from_numpy(np.asarray(frames)).float()
    if tensor.ndim != 4:
        raise ValueError(f"expected frames with shape [T, 3, H, W], got {tuple(tensor.shape)}")
    return tensor


def resize_and_center_crop_frames(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    return resize_and_center_crop_frames_shared(frames, out_h, out_w)


def normalize_video_range(frames: torch.Tensor) -> torch.Tensor:
    return normalize_video_range_shared(frames)


def preprocess_ti2v_prefix_frames(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    return preprocess_ti2v_prefix_frames_shared(frames, out_h, out_w)


def build_ti2v_training_video(
    context_frames: np.ndarray | torch.Tensor,
    future_frames: np.ndarray | torch.Tensor,
    frame_num: int | None = None,
) -> torch.Tensor:
    context_tensor = to_frame_tensor(context_frames)
    future_tensor = to_frame_tensor(future_frames)
    if context_tensor.shape[0] < 1:
        raise ValueError("context_frames must contain at least one frame")
    base_video = torch.cat([context_tensor[:1], future_tensor], dim=0)
    min_frame_num = align_wan_frame_num(int(base_video.shape[0]))
    target_frame_num = min_frame_num if frame_num in (None, 0) else align_wan_frame_num(int(frame_num))
    if target_frame_num < min_frame_num:
        raise ValueError(
            f"frame_num={frame_num} is too small for this sample: need at least {min_frame_num} frames after Wan alignment"
        )
    if base_video.shape[0] == target_frame_num:
        return base_video
    pad_count = target_frame_num - int(base_video.shape[0])
    pad_frame = base_video[-1:].expand(pad_count, -1, -1, -1)
    return torch.cat([base_video, pad_frame], dim=0)


def build_prefix_training_video(
    context_frames: np.ndarray | torch.Tensor,
    future_frames: np.ndarray | torch.Tensor,
    frame_num: int | None = None,
) -> torch.Tensor:
    context_tensor = to_frame_tensor(context_frames)
    future_tensor = to_frame_tensor(future_frames)
    base_video = torch.cat([context_tensor, future_tensor], dim=0)
    min_frame_num = align_wan_frame_num(int(base_video.shape[0]))
    target_frame_num = min_frame_num if frame_num in (None, 0) else align_wan_frame_num(int(frame_num))
    if target_frame_num < min_frame_num:
        raise ValueError(
            f"frame_num={frame_num} is too small for this sample: need at least {min_frame_num} frames after Wan alignment"
        )
    if base_video.shape[0] == target_frame_num:
        return base_video
    pad_count = target_frame_num - int(base_video.shape[0])
    pad_frame = base_video[-1:].expand(pad_count, -1, -1, -1)
    return torch.cat([base_video, pad_frame], dim=0)


def build_first_frame_mask(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4:
        raise ValueError(f"expected latent with shape [C, T, H, W], got {tuple(latent.shape)}")
    mask = torch.ones_like(latent)
    mask[:, 0] = 0
    return mask


def build_prefix_latent_mask(latent: torch.Tensor, prefix_len: int) -> torch.Tensor:
    return build_prefix_latent_mask_shared(latent, prefix_len)


def build_ti2v_timestep_tensor(mask: torch.Tensor, timestep: torch.Tensor, seq_len: int) -> torch.Tensor:
    if mask.ndim != 4:
        raise ValueError(f"expected mask with shape [C, T, H, W], got {tuple(mask.shape)}")
    if timestep.ndim != 1 or timestep.shape[0] != 1:
        raise ValueError(f"expected timestep with shape [1], got {tuple(timestep.shape)}")
    masked = (mask[0][:, ::2, ::2] * timestep).flatten()
    if masked.numel() > seq_len:
        raise ValueError(f"masked timestep token count {masked.numel()} exceeds seq_len {seq_len}")
    if masked.numel() < seq_len:
        masked = torch.cat([masked, masked.new_ones(seq_len - masked.numel()) * timestep])
    return masked.unsqueeze(0)


def build_prefix_timestep_tensor(mask: torch.Tensor, timestep: torch.Tensor, seq_len: int) -> torch.Tensor:
    return build_prefix_timestep_tensor_shared(mask, timestep, seq_len)


def compute_ti2v_seq_len(latent: torch.Tensor, patch_size: tuple[int, int]) -> int:
    if latent.ndim != 4:
        raise ValueError(f"expected latent with shape [C, T, H, W], got {tuple(latent.shape)}")
    _, latent_steps, lat_h, lat_w = latent.shape
    return latent_steps * lat_h * lat_w // (patch_size[0] * patch_size[1])


def apply_clean_prefix_to_latents(latent: torch.Tensor, clean_prefix_latents: torch.Tensor) -> torch.Tensor:
    return apply_clean_prefix_to_latents_shared(latent, clean_prefix_latents)

def resample_condition_maps_to_steps(condition_maps: torch.Tensor, target_steps: int) -> torch.Tensor:
    return resample_condition_maps_to_steps_shared(condition_maps, target_steps)


def filter_state_condition_payload_for_adapter(state_condition: dict[str, torch.Tensor], adapter) -> dict[str, torch.Tensor]:
    return filter_state_condition_payload_for_adapter_shared(state_condition, adapter)


def select_ti2v_state_adapter_parameters(pipeline) -> list[tuple[str, torch.nn.Parameter]]:
    if getattr(pipeline, "state_adapter", None) is None:
        raise RuntimeError("pipeline.state_adapter is not initialized")

    if hasattr(pipeline.text_encoder, "model"):
        pipeline.text_encoder.model.eval().requires_grad_(False)
    vae_module = getattr(pipeline.vae, "model", None)
    if vae_module is not None and hasattr(vae_module, "eval"):
        vae_module.eval().requires_grad_(False)
    if hasattr(pipeline.model, "enable_gradient_checkpointing"):
        pipeline.model.enable_gradient_checkpointing()
    pipeline.model.train()
    pipeline.model.requires_grad_(False)
    pipeline.state_adapter.train()
    pipeline.state_adapter.requires_grad_(True)

    trainable: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in pipeline.state_adapter.named_parameters():
        param.requires_grad_(True)
        trainable.append((f"state_adapter.{name}", param))
    for name, param in pipeline.model.named_parameters():
        if "state_adapter_" not in name:
            param.requires_grad_(False)
            continue
        param.requires_grad_(True)
        trainable.append((f"model.{name}", param))
    return trainable


def select_i2v_state_adapter_parameters(pipeline) -> list[tuple[str, torch.nn.Parameter]]:
    if getattr(pipeline, "state_adapter", None) is None:
        raise RuntimeError("pipeline.state_adapter is not initialized")

    if hasattr(pipeline.text_encoder, "model"):
        pipeline.text_encoder.model.eval().requires_grad_(False)
    vae_module = getattr(pipeline.vae, "model", None)
    if vae_module is not None and hasattr(vae_module, "eval"):
        vae_module.eval().requires_grad_(False)
    pipeline.low_noise_model.train().requires_grad_(False)
    pipeline.high_noise_model.train().requires_grad_(False)
    pipeline.state_adapter.train().requires_grad_(True)

    trainable: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in pipeline.state_adapter.named_parameters():
        param.requires_grad_(True)
        trainable.append((f"state_adapter.{name}", param))
    for module_name, module in (
        ("low_noise_model", pipeline.low_noise_model),
        ("high_noise_model", pipeline.high_noise_model),
    ):
        for name, param in module.named_parameters():
            if "state_adapter_" not in name:
                param.requires_grad_(False)
                continue
            param.requires_grad_(True)
            trainable.append((f"{module_name}.{name}", param))
    return trainable


class LocalWanFlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        sigmas = torch.linspace(1.0, 0.0, self.num_train_timesteps + 1, dtype=torch.float32)[:-1]
        self.sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps
        self.linear_timesteps_weights = self._build_training_weight()

    def _build_training_weight(self) -> torch.Tensor:
        steps = float(self.num_train_timesteps)
        x = self.timesteps
        weights = torch.exp(-2.0 * ((x - steps / 2.0) / steps) ** 2)
        weights = weights - weights.min()
        weights = weights * (steps / max(float(weights.sum()), 1e-6))
        return weights

    def sample_timestep(self, *, device, dtype) -> torch.Tensor:
        timestep_id = torch.randint(0, len(self.timesteps), (1,), device=device)
        return self.timesteps.to(device=device, dtype=dtype)[timestep_id]

    def _sigma_from_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        distance = (self.timesteps.to(device=timestep.device, dtype=timestep.dtype) - timestep).abs()
        index = int(torch.argmin(distance).item())
        return self.sigmas.to(device=timestep.device, dtype=timestep.dtype)[index]

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma_from_timestep(timestep)
        return (1.0 - sigma) * original_samples + sigma * noise

    def training_target(self, sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        distance = (self.timesteps.to(device=timestep.device, dtype=timestep.dtype) - timestep).abs()
        index = int(torch.argmin(distance).item())
        return self.linear_timesteps_weights.to(device=timestep.device, dtype=timestep.dtype)[index]


def serialize_ti2v_state_adapter_checkpoint(exported_bundle: dict[str, object], meta: dict[str, object]) -> dict[str, object]:
    payload = dict(exported_bundle)
    payload["trainer_meta"] = meta
    return payload


def serialize_i2v_state_adapter_checkpoint(exported_bundle: dict[str, object], meta: dict[str, object]) -> dict[str, object]:
    payload = dict(exported_bundle)
    payload["trainer_meta"] = meta
    return payload


def load_frozen_state_adapter_encoder(
    checkpoint_path: str | Path,
    *,
    wan_repo_root: str | Path = "/home/gaoya/Code_Video/Wan2.2-main",
    device: str = "cpu",
):
    ensure_wan_importable(wan_repo_root)
    from wan_.state_condition import WanObjectStateAdapter

    state_bundle = torch.load(str(checkpoint_path), map_location="cpu")
    adapter_config = dict(state_bundle["state_adapter_config"])
    adapter = WanObjectStateAdapter(**adapter_config)
    adapter.load_state_dict(state_bundle["state_adapter"])
    adapter.eval().requires_grad_(False)
    return adapter.to(device)
