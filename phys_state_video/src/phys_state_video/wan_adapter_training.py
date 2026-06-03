from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import require_torch

torch = require_torch()
F = torch.nn.functional


REQUIRED_STATE_ADAPTER_KEYS_TI2V = {
    "state_adapter_config",
    "state_adapter",
    "model_state_adapter",
}


@dataclass(slots=True)
class StateConditionBundleRecord:
    sample_id: str
    bundle_dir: Path
    episode_path: Path
    image_path: Path
    state_condition_path: Path
    meta_path: Path
    prompt_path: Path
    prompt: str
    meta: dict[str, object]


def align_wan_frame_num(frame_num: int) -> int:
    if frame_num <= 0:
        raise ValueError(f"frame_num must be positive, got {frame_num}")
    remainder = (frame_num - 1) % 4
    if remainder == 0:
        return frame_num
    return frame_num + (4 - remainder)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_condition_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def is_ti2v_state_adapter_checkpoint(state_bundle: dict[str, object]) -> bool:
    return REQUIRED_STATE_ADAPTER_KEYS_TI2V.issubset(state_bundle.keys())


def _discover_bundle_dirs(root: Path) -> list[Path]:
    manifest_path = root / "manifest.jsonl"
    if manifest_path.is_file():
        bundle_dirs: list[Path] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                state_condition_path = Path(record["state_condition_path"]).resolve()
                bundle_dirs.append(state_condition_path.parent)
        return bundle_dirs
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "state_condition.npz").is_file() and (path / "meta.json").is_file()
    )


def discover_state_condition_bundles(root: str | Path, limit: int = 0) -> list[StateConditionBundleRecord]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"state-condition root does not exist: {root}")
    bundle_dirs = _discover_bundle_dirs(root)
    if limit > 0:
        bundle_dirs = bundle_dirs[:limit]
    records: list[StateConditionBundleRecord] = []
    for bundle_dir in bundle_dirs:
        meta_path = bundle_dir / "meta.json"
        state_condition_path = bundle_dir / "state_condition.npz"
        image_path = bundle_dir / "input_image.png"
        prompt_path = bundle_dir / "prompt.txt"
        meta = _read_json(meta_path)
        episode_path = Path(str(meta["episode_path"])).resolve()
        prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.is_file() else str(meta.get("prompt", ""))
        records.append(
            StateConditionBundleRecord(
                sample_id=str(meta.get("sample_id", bundle_dir.name)),
                bundle_dir=bundle_dir.resolve(),
                episode_path=episode_path,
                image_path=image_path.resolve(),
                state_condition_path=state_condition_path.resolve(),
                meta_path=meta_path.resolve(),
                prompt_path=prompt_path.resolve(),
                prompt=prompt,
                meta=meta,
            )
        )
    if not records:
        raise FileNotFoundError(f"no state-condition bundles found under {root}")
    return records


def load_episode_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def to_frame_tensor(frames: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(frames, torch.Tensor):
        tensor = frames.float()
    else:
        tensor = torch.from_numpy(np.asarray(frames)).float()
    if tensor.ndim != 4:
        raise ValueError(f"expected frames with shape [T, 3, H, W], got {tuple(tensor.shape)}")
    return tensor


def resize_and_center_crop_frames(frames: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
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


def normalize_video_range(frames: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(frames):
        frames = frames.float()
    max_value = float(frames.max()) if frames.numel() > 0 else 1.0
    min_value = float(frames.min()) if frames.numel() > 0 else 0.0
    if min_value >= 0.0 and max_value <= 1.0:
        return frames * 2.0 - 1.0
    if min_value >= 0.0 and max_value <= 255.0:
        return frames / 127.5 - 1.0
    return frames.clamp(-1.0, 1.0)


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


def build_first_frame_mask(latent: torch.Tensor) -> torch.Tensor:
    if latent.ndim != 4:
        raise ValueError(f"expected latent with shape [C, T, H, W], got {tuple(latent.shape)}")
    mask = torch.ones_like(latent)
    mask[:, 0] = 0
    return mask


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


def compute_ti2v_seq_len(latent: torch.Tensor, patch_size: tuple[int, int]) -> int:
    if latent.ndim != 4:
        raise ValueError(f"expected latent with shape [C, T, H, W], got {tuple(latent.shape)}")
    _, latent_steps, lat_h, lat_w = latent.shape
    return latent_steps * lat_h * lat_w // (patch_size[0] * patch_size[1])


def select_ti2v_state_adapter_parameters(pipeline) -> list[tuple[str, torch.nn.Parameter]]:
    if getattr(pipeline, "state_adapter", None) is None:
        raise RuntimeError("pipeline.state_adapter is not initialized")

    if hasattr(pipeline.text_encoder, "model"):
        pipeline.text_encoder.model.eval().requires_grad_(False)
    pipeline.vae.eval().requires_grad_(False)
    pipeline.model.eval()
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
