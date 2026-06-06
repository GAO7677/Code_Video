from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

from .predictor_wan_state_v2 import resample_temporal_features
from .utils import require_torch

torch = require_torch()
F = torch.nn.functional


def normalize_video_range_shared(frames: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(frames):
        frames = frames.float()
    max_value = float(frames.max()) if frames.numel() > 0 else 1.0
    min_value = float(frames.min()) if frames.numel() > 0 else 0.0
    if min_value >= 0.0 and max_value <= 1.0:
        return frames * 2.0 - 1.0
    if min_value >= 0.0 and max_value <= 255.0:
        return frames / 127.5 - 1.0
    return frames.clamp(-1.0, 1.0)


def resize_and_center_crop_frames_shared(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"expected frames with shape [T, 3, H, W], got {tuple(frames.shape)}")
    _, _, in_h, in_w = frames.shape
    scale = max(out_h / max(in_h, 1), out_w / max(in_w, 1))
    resized_h = max(int(round(in_h * scale)), out_h)
    resized_w = max(int(round(in_w * scale)), out_w)
    resized = F.interpolate(frames, size=(resized_h, resized_w), mode="bilinear", align_corners=False)
    top = max((resized_h - out_h) // 2, 0)
    left = max((resized_w - out_w) // 2, 0)
    return resized[:, :, top : top + out_h, left : left + out_w].contiguous()


def preprocess_ti2v_prefix_frames_shared(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    normalized = normalize_video_range_shared(frames)
    return resize_and_center_crop_frames_shared(normalized, out_h=out_h, out_w=out_w)


def compute_latent_step_count(frame_steps: int, temporal_stride: int) -> int:
    if frame_steps <= 0:
        raise ValueError(f"frame_steps must be positive, got {frame_steps}")
    if temporal_stride <= 0:
        raise ValueError(f"temporal_stride must be positive, got {temporal_stride}")
    return 1 + max(frame_steps - 1, 0) // temporal_stride


def compute_future_latent_steps(context_steps: int, future_steps: int, temporal_stride: int) -> int:
    total = compute_latent_step_count(context_steps + future_steps, temporal_stride)
    context = compute_latent_step_count(context_steps, temporal_stride)
    future = total - context
    if future <= 0:
        raise ValueError(
            f"future latent steps must be positive, got context_steps={context_steps}, future_steps={future_steps}, "
            f"temporal_stride={temporal_stride}, total_latents={total}, context_latents={context}"
        )
    return future


def build_future_step_loss_mask(
    future_latent_steps: int,
    valid_future_latent_steps: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    if future_latent_steps <= 0:
        raise ValueError(f"future_latent_steps must be positive, got {future_latent_steps}")
    if valid_future_latent_steps <= 0 or valid_future_latent_steps > future_latent_steps:
        raise ValueError(
            f"valid_future_latent_steps must be in [1, {future_latent_steps}], got {valid_future_latent_steps}"
        )
    mask = torch.zeros((1, future_latent_steps, 1, 1), device=device, dtype=dtype)
    mask[:, :valid_future_latent_steps] = 1
    return mask


def resample_camera_to_latent_steps(camera: torch.Tensor, target_steps: int) -> torch.Tensor:
    return resample_temporal_features(camera, target_steps)


def split_context_future_camera(
    camera: torch.Tensor,
    context_steps: int,
    future_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if camera.ndim == 2:
        camera = camera.unsqueeze(0)
    if camera.ndim != 3:
        raise ValueError(f"expected camera [B, T, D] or [T, D], got {tuple(camera.shape)}")
    if context_steps <= 0 or future_steps <= 0:
        raise ValueError(f"context_steps and future_steps must be positive, got {context_steps} and {future_steps}")
    total_steps = int(camera.shape[1])
    if total_steps < context_steps:
        raise ValueError(
            f"camera length {total_steps} is smaller than required context_steps {context_steps}"
        )
    context_camera = camera[:, :context_steps]
    if total_steps >= context_steps + future_steps:
        future_camera = camera[:, context_steps : context_steps + future_steps]
    elif total_steps > context_steps:
        future_camera = camera[:, context_steps:]
        pad = future_steps - int(future_camera.shape[1])
        if pad > 0:
            future_camera = torch.cat([future_camera, future_camera[:, -1:].expand(-1, pad, -1)], dim=1)
    else:
        future_camera = context_camera[:, -1:].expand(-1, future_steps, -1)
    return context_camera.contiguous(), future_camera.contiguous()


def align_wan_frame_num(frame_num: int) -> int:
    if frame_num <= 0:
        raise ValueError(f"frame_num must be positive, got {frame_num}")
    remainder = (frame_num - 1) % 4
    if remainder == 0:
        return frame_num
    return frame_num + (4 - remainder)


def apply_clean_prefix_to_latents(latent: torch.Tensor, clean_prefix_latents: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4 or clean_prefix_latents.ndim != 4:
        raise ValueError(
            f"expected latent and clean_prefix_latents with shape [C, T, H, W], got {tuple(latent.shape)} and "
            f"{tuple(clean_prefix_latents.shape)}"
        )
    prefix_len = int(clean_prefix_latents.shape[1])
    if prefix_len <= 0 or prefix_len >= latent.shape[1]:
        raise ValueError(f"invalid prefix_len={prefix_len} for latent_steps={latent.shape[1]}")
    updated = latent.clone()
    updated[:, :prefix_len] = clean_prefix_latents
    return updated


def build_prefix_latent_mask(latent: torch.Tensor, prefix_len: int) -> torch.Tensor:
    if latent.ndim != 4:
        raise ValueError(f"expected latent with shape [C, T, H, W], got {tuple(latent.shape)}")
    if prefix_len <= 0 or prefix_len >= latent.shape[1]:
        raise ValueError(f"prefix_len must be in [1, {latent.shape[1] - 1}], got {prefix_len}")
    mask = torch.ones_like(latent)
    mask[:, :prefix_len] = 0
    return mask


def build_prefix_timestep_tensor(mask: torch.Tensor, timestep: torch.Tensor, seq_len: int) -> torch.Tensor:
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


def resample_condition_maps_to_steps(condition_maps: torch.Tensor, target_steps: int) -> torch.Tensor:
    if condition_maps.ndim == 4:
        condition_maps = condition_maps.unsqueeze(0)
    if condition_maps.ndim != 5:
        raise ValueError(
            f"expected condition_maps with shape [B, T, C, H, W] or [T, C, H, W], got {tuple(condition_maps.shape)}"
        )
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if condition_maps.shape[1] == target_steps:
        return condition_maps
    batch, steps, channels, height, width = condition_maps.shape
    flattened = condition_maps.permute(0, 2, 3, 4, 1).contiguous().view(batch, channels * height * width, steps)
    resized = F.interpolate(
        flattened,
        size=target_steps,
        mode="linear",
        align_corners=False,
    )
    return resized.view(batch, channels, height, width, target_steps).permute(0, 4, 1, 2, 3).contiguous()


def filter_state_condition_payload_for_adapter(
    state_condition: dict[str, torch.Tensor],
    adapter,
) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    if getattr(adapter, "memory_token_dim", None) is not None and state_condition.get("memory_tokens") is not None:
        filtered["memory_tokens"] = state_condition["memory_tokens"]
    if getattr(adapter, "map_token_dim", None) is not None and state_condition.get("condition_maps") is not None:
        filtered["condition_maps"] = state_condition["condition_maps"]
    if not filtered:
        raise ValueError(
            "state_condition payload does not match any initialized adapter branch. "
            "The v2 mainline only supports condition_maps + memory_tokens payloads."
        )
    return filtered


def build_state_condition_payload_from_condition_maps(
    condition_maps: torch.Tensor,
    memory_tokens: torch.Tensor | None = None,
    *,
    include_condition_maps: bool = False,
) -> dict[str, torch.Tensor]:
    if condition_maps.ndim == 4:
        condition_maps = condition_maps.unsqueeze(0)
    if condition_maps.ndim != 5:
        raise ValueError(
            f"expected condition_maps with shape [B, T, C, H, W] or [T, C, H, W], got {tuple(condition_maps.shape)}"
        )
    payload: dict[str, torch.Tensor] = {}
    if memory_tokens is not None:
        if memory_tokens.ndim == 2:
            memory_tokens = memory_tokens.unsqueeze(0)
        if memory_tokens.ndim != 3:
            raise ValueError(
                f"expected memory_tokens with shape [B, N, D] or [N, D], got {tuple(memory_tokens.shape)}"
            )
        payload["memory_tokens"] = memory_tokens
    if include_condition_maps:
        payload["condition_maps"] = condition_maps
    return payload


@dataclass(slots=True)
class WanPromptContextEncoder:
    ckpt_dir: str | Path
    wan_repo_root: str | Path | None = None
    task: str = "i2v-A14B"
    device: str = "cpu"
    text_encoder: Any = field(init=False, repr=False)
    context_dim: int = field(init=False)
    max_text_len: int = field(init=False)

    def __post_init__(self) -> None:
        from .wan_runtime import load_wan_modules

        modules = load_wan_modules(self.wan_repo_root)
        wan_configs = modules["WAN_CONFIGS"]
        if self.task not in wan_configs:
            raise ValueError(f"unsupported Wan task: {self.task}")
        task_config = wan_configs[self.task]
        checkpoint_dir = Path(self.ckpt_dir)
        checkpoint_path = checkpoint_dir / task_config.t5_checkpoint
        tokenizer_path = checkpoint_dir / task_config.t5_tokenizer
        resolved_tokenizer = str(tokenizer_path) if tokenizer_path.exists() else str(task_config.t5_tokenizer)
        dtype = torch.bfloat16 if str(self.device).startswith("cuda") else torch.float32
        self.text_encoder = modules["T5EncoderModel"](
            text_len=int(task_config.text_len),
            dtype=dtype,
            device=torch.device(self.device),
            checkpoint_path=str(checkpoint_path),
            tokenizer_path=resolved_tokenizer,
        )
        self.context_dim = int(self.text_encoder.model.dim)
        self.max_text_len = int(task_config.text_len)

    def encode_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoded = self.text_encoder(prompts, torch.device(self.device))
        batch = len(encoded)
        max_len = max((item.shape[0] for item in encoded), default=1)
        context = torch.zeros(
            batch,
            max_len,
            self.context_dim,
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros(batch, max_len, dtype=torch.float32, device=self.device)
        for index, item in enumerate(encoded):
            item = item.to(device=self.device, dtype=torch.float32)
            context[index, : item.shape[0]] = item
            mask[index, : item.shape[0]] = 1.0
        return context, mask


@dataclass(slots=True)
class MockLatentExtractor:
    latent_channels: int = 16
    latent_height: int = 8
    latent_width: int = 8
    temporal_stride: int = 4
    device: str = "cpu"

    def encode_context_frames_raw(self, context_frames: torch.Tensor) -> torch.Tensor:
        if context_frames.ndim != 5:
            raise ValueError(
                f"expected context frames with shape [B, K, 3, H, W], got {tuple(context_frames.shape)}"
            )
        batch, context_steps = context_frames.shape[:2]
        latent_steps = compute_latent_step_count(context_steps, self.temporal_stride)
        frames = context_frames.to(self.device).float()
        flattened = frames.permute(0, 2, 3, 4, 1).contiguous().view(
            batch,
            frames.shape[2] * frames.shape[3] * frames.shape[4],
            context_steps,
        )
        resized_time = F.interpolate(
            flattened,
            size=latent_steps,
            mode="linear",
            align_corners=False,
        )
        resized_time = resized_time.view(batch, 3, frames.shape[3], frames.shape[4], latent_steps).permute(0, 4, 1, 2, 3)
        spatial = F.interpolate(
            resized_time.reshape(batch * latent_steps, 3, frames.shape[3], frames.shape[4]),
            size=(self.latent_height, self.latent_width),
            mode="bilinear",
            align_corners=False,
        ).view(batch, latent_steps, 3, self.latent_height, self.latent_width)
        if self.latent_channels <= 3:
            return spatial[:, :, : self.latent_channels]
        repeats = (self.latent_channels + 2) // 3
        expanded = spatial.repeat(1, 1, repeats, 1, 1)
        return expanded[:, :, : self.latent_channels].contiguous()
