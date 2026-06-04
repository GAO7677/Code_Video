from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .predictor_wan_state_v2 import resample_temporal_features
from .utils import require_torch

torch = require_torch()
F = torch.nn.functional


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


def resample_camera_to_latent_steps(camera: torch.Tensor, target_steps: int) -> torch.Tensor:
    return resample_temporal_features(camera, target_steps)


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


def resample_state_tokens_to_steps(state_tokens: torch.Tensor, target_steps: int) -> torch.Tensor:
    if state_tokens.ndim == 2:
        state_tokens = state_tokens.unsqueeze(0)
    if state_tokens.ndim != 3:
        raise ValueError(f"expected state_tokens with shape [B, T, D] or [T, D], got {tuple(state_tokens.shape)}")
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if state_tokens.shape[1] == target_steps:
        return state_tokens
    resized = F.interpolate(
        state_tokens.transpose(1, 2),
        size=target_steps,
        mode="linear",
        align_corners=False,
    )
    return resized.transpose(1, 2).contiguous()


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


def flatten_condition_maps_to_state_tokens(condition_maps: torch.Tensor) -> torch.Tensor:
    if condition_maps.ndim == 4:
        condition_maps = condition_maps.unsqueeze(0)
    if condition_maps.ndim != 5:
        raise ValueError(
            f"expected condition_maps with shape [B, T, C, H, W] or [T, C, H, W], got {tuple(condition_maps.shape)}"
        )
    batch, steps, channels, height, width = condition_maps.shape
    return condition_maps.permute(0, 1, 3, 4, 2).contiguous().view(batch, steps * height * width, channels)


def filter_state_condition_payload_for_adapter(
    state_condition: dict[str, torch.Tensor],
    adapter,
) -> dict[str, torch.Tensor]:
    filtered: dict[str, torch.Tensor] = {}
    if getattr(adapter, "state_token_dim", None) is not None and state_condition.get("state_tokens") is not None:
        filtered["state_tokens"] = state_condition["state_tokens"]
    if getattr(adapter, "memory_token_dim", None) is not None and state_condition.get("memory_tokens") is not None:
        filtered["memory_tokens"] = state_condition["memory_tokens"]
    if getattr(adapter, "map_token_dim", None) is not None and state_condition.get("condition_maps") is not None:
        filtered["condition_maps"] = state_condition["condition_maps"]
    if not filtered:
        raise ValueError("state_condition payload does not match any initialized adapter branch")
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
    payload = {
        "state_tokens": flatten_condition_maps_to_state_tokens(condition_maps),
    }
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

    def __post_init__(self) -> None:
        from .wan_bridge import load_wan_modules

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
