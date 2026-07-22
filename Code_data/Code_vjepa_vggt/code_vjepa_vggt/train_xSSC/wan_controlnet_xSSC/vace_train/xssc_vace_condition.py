"""Frozen xSSC to VACE-condition adapter.

This file only changes the condition source for DiffSynth's Wan VACE branch.
It keeps the official VACE model and its residual injection path unchanged:

    full video -> frozen xSSC slots -> ctx-visible/future-masked xSSC VACE condition
    vace_context -> official VaceWanModel -> block-wise residual hints

The adapter emits a VACE-compatible tensor shaped ``[B, 96, Tz, Hvae, Wvae]``.
The channel split deliberately follows DiffSynth's official VACE condition:
32 channels are inactive/reactive condition latents and 64 channels are mask
channels.  Here those inactive/reactive latents are not VAE-encoded RGB/depth
videos; they are dense latent-like maps generated from frozen xSSC slots.

When ctx video is used as VACE reference, ctx frames are padded to a Wan
VAE-friendly length and encoded as one short video, then prepended to both the
denoising latent sequence and the xSSC VACE condition sequence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffsynth.diffusion.base_pipeline import PipelineUnit


XSSC_IMAGENET_MEAN = (123.675, 116.28, 103.53)
XSSC_IMAGENET_STD = (58.395, 57.12, 57.375)
DEFAULT_XSSC_ROOT = "/home/gaoya/Code_Video/xSSC-main"
DEFAULT_XSSC_CONFIG = f"{DEFAULT_XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py"
DEFAULT_XSSC_CHECKPOINT = "/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth"
DEFAULT_XSSC_CONTEXT_FRAMES = 8
DEFAULT_WAN_VAE_TEMPORAL_STRIDE = 4


def wan_latent_frame_count(num_frames: int) -> int:
    """Wan VAE temporal length for a video containing ``num_frames`` frames."""
    num_frames = int(num_frames)
    if num_frames <= 0:
        return 0
    return (num_frames + DEFAULT_WAN_VAE_TEMPORAL_STRIDE - 1) // DEFAULT_WAN_VAE_TEMPORAL_STRIDE


def pad_frame_count_to_wan_vae(num_frames: int) -> int:
    """Pad a ctx clip to ``4n+1`` frames so Wan VAE keeps the tail context."""
    num_frames = int(num_frames)
    if num_frames <= 0:
        return 0
    remainder = num_frames % DEFAULT_WAN_VAE_TEMPORAL_STRIDE
    target_remainder = 1
    if remainder == target_remainder:
        return num_frames
    return num_frames + ((target_remainder - remainder) % DEFAULT_WAN_VAE_TEMPORAL_STRIDE)


def reference_latent_frame_count(vace_reference_image) -> int:
    """Number of prepended Wan latent steps for an xSSC-VACE reference clip."""
    if vace_reference_image is None:
        return 0
    if isinstance(vace_reference_image, list):
        return wan_latent_frame_count(len(vace_reference_image))
    return 1


def load_xssc_model(
    *,
    xssc_root: str,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Build official RandSFQ2 from config and strictly load its checkpoint."""
    root = Path(xssc_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"xSSC root does not exist: {root}")
    if not config.is_file():
        raise FileNotFoundError(f"xSSC config does not exist: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"xSSC checkpoint does not exist: {checkpoint}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import timm
    from object_centric_bench.util import Config, build_from_config

    cfg = Config.fromfile(config)
    original_create_model = timm.create_model

    def create_model_offline(*args, **kwargs):
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    timm.create_model = create_model_offline
    try:
        model = build_from_config(cfg.model)
    finally:
        timm.create_model = original_create_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported xSSC checkpoint object: {type(state)!r}")
    if state and all(str(key).startswith("m.") for key in state):
        state = {str(key)[2:]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)

    slot_dim = int(cfg.emb_dim)
    num_slots = int(cfg.max_num)
    model.decode = None
    model.requires_grad_(False)
    model.eval()
    model.to(device=device)
    return model, slot_dim, num_slots


class XSSCVACEContextConditioner(nn.Module):
    """Convert official xSSC slots into a dense VACE condition tensor."""

    def __init__(
        self,
        *,
        xssc_root: str = DEFAULT_XSSC_ROOT,
        xssc_config: str = DEFAULT_XSSC_CONFIG,
        xssc_checkpoint: str = DEFAULT_XSSC_CHECKPOINT,
        xssc_input_size: int = 256,
        xssc_condition_frames: int = DEFAULT_XSSC_CONTEXT_FRAMES,
        temporal_stride: int = DEFAULT_WAN_VAE_TEMPORAL_STRIDE,
        vace_in_dim: int = 96,
        vace_video_dim: int = 32,
        query_dim: int = 256,
        slot_dropout: float = 0.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.xssc_input_size = int(xssc_input_size)
        self.xssc_condition_frames = int(xssc_condition_frames)
        self.temporal_stride = int(temporal_stride)
        self.vace_in_dim = int(vace_in_dim)
        self.vace_video_dim = int(vace_video_dim)
        self.vace_mask_dim = self.vace_in_dim - self.vace_video_dim
        self.vace_slot_latent_dim = self.vace_video_dim // 2
        self.query_dim = int(query_dim)
        self.slot_dropout = float(slot_dropout)
        if self.xssc_condition_frames <= 0:
            raise ValueError("--xssc_condition_frames must be positive")
        if self.temporal_stride <= 0:
            raise ValueError("--xssc_vae_temporal_stride must be positive")
        if not 0.0 <= self.slot_dropout < 1.0:
            raise ValueError("--xssc_slot_dropout must be in [0, 1)")
        if self.vace_video_dim <= 0 or self.vace_video_dim >= self.vace_in_dim:
            raise ValueError(
                f"vace_video_dim must be in (0, {self.vace_in_dim}), got {self.vace_video_dim}"
            )
        if self.vace_video_dim % 2 != 0:
            raise ValueError(f"vace_video_dim must be even, got {self.vace_video_dim}")

        self.xssc, self.xssc_slot_dim, self.xssc_num_slots = load_xssc_model(
            xssc_root=xssc_root,
            config_path=xssc_config,
            checkpoint_path=xssc_checkpoint,
            device=torch.device(device),
        )
        if self.xssc_slot_dim != 256:
            raise ValueError(f"Expected 256-d xSSC slots, got {self.xssc_slot_dim}")

        self.slot_norm = nn.LayerNorm(self.xssc_slot_dim)
        self.slot_key = nn.Linear(self.xssc_slot_dim, self.query_dim, bias=False)
        self.slot_value = nn.Linear(self.xssc_slot_dim, self.vace_slot_latent_dim, bias=False)
        self.coord_query = nn.Sequential(
            nn.Linear(4, self.query_dim),
            nn.SiLU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.video_norm = nn.GroupNorm(1, self.vace_video_dim)
        self.future_placeholder = nn.Parameter(
            torch.zeros(1, self.vace_slot_latent_dim, 1, 1, 1, device=device, dtype=dtype)
        )
        for module in (
            self.slot_norm,
            self.slot_key,
            self.slot_value,
            self.coord_query,
            self.video_norm,
        ):
            module.to(device=device, dtype=dtype)
        self.xssc.to(device=device)
        self.xssc.requires_grad_(False)
        self.xssc.eval()

        self.last_stats: dict[str, float] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.xssc.eval()
        return self

    def _preprocess_xssc(self, video: torch.Tensor) -> torch.Tensor:
        """Convert ``[B,C,T,H,W]`` in ``[-1,1]`` to xSSC ``[B,T,C,256,256]``."""
        frames = video.permute(0, 2, 1, 3, 4).float()
        batch, time_steps, channels, height, width = frames.shape
        crop_size = min(int(height), int(width))
        top = (int(height) - crop_size) // 2
        left = (int(width) - crop_size) // 2
        frames = frames[..., top : top + crop_size, left : left + crop_size]
        frames = frames.reshape(batch * time_steps, channels, crop_size, crop_size)
        frames = F.interpolate(
            frames,
            size=(self.xssc_input_size, self.xssc_input_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        frames = (frames + 1.0).mul(127.5).clamp(0.0, 255.0)
        mean = frames.new_tensor(XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = frames.new_tensor(XSSC_IMAGENET_STD).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames.view(batch, time_steps, channels, self.xssc_input_size, self.xssc_input_size)

    @torch.no_grad()
    def extract_slots(self, xssc_video: torch.Tensor) -> torch.Tensor:
        """Run frozen official xSSC recurrence and return ``[B,T,K,256]``."""
        self.xssc.eval()
        batch, time_steps, channels, height, width = xssc_video.shape
        flat_video = xssc_video.flatten(0, 1)
        autocast_enabled = flat_video.device.type == "cuda"
        with torch.autocast(
            device_type=flat_video.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            feature = self.xssc.encode_backbone(flat_video).detach()
            encoded = feature.permute(0, 2, 3, 1)
            encoded = self.xssc.encode_posit_embed(encoded).flatten(1, 2)
            encoded = self.xssc.encode_project(encoded)
            encoded = encoded.view(batch, time_steps, encoded.shape[1], encoded.shape[2])

            slots = None
            for frame_id in range(time_steps):
                if frame_id == 0:
                    query = self.xssc.initializ(batch)
                else:
                    query = self.xssc.transit(slots, encoded[:, : frame_id + 1])
                num_iter = None if frame_id == 0 else 1
                current_slots, _ = self.xssc.aggregat(
                    encoded[:, frame_id], query, num_iter=num_iter
                )
                current_slots = current_slots[:, None]
                slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
        if slots is None:
            raise RuntimeError("xSSC received zero frames")
        return slots

    def _group_slots_to_latent_time(
        self,
        slots: torch.Tensor,
        *,
        target_frames: int | None = None,
    ) -> torch.Tensor:
        """Wan VAE-style temporal grouping: frame 0, then 4-frame chunks."""
        chunks = [slots[:, :1]]
        for start in range(1, int(slots.shape[1]), self.temporal_stride):
            chunks.append(slots[:, start : start + self.temporal_stride].mean(dim=1, keepdim=True))
        aligned = torch.cat(chunks, dim=1)
        if target_frames is None:
            return aligned
        if aligned.shape[1] < target_frames:
            repeat = aligned[:, -1:].expand(-1, target_frames - aligned.shape[1], -1, -1)
            aligned = torch.cat([aligned, repeat], dim=1)
        elif aligned.shape[1] > target_frames:
            aligned = aligned[:, :target_frames]
        return aligned

    def _group_frame_mask_to_latent_time(
        self,
        frame_mask: torch.Tensor,
        *,
        target_frames: int,
    ) -> torch.Tensor:
        """Resize raw-frame visibility mask to latent time like official VACE."""
        mask = frame_mask[:, None, :, None, None].float()
        mask = F.interpolate(mask, size=(target_frames, 1, 1), mode="nearest-exact")
        return mask[:, 0, :, 0, 0].to(dtype=frame_mask.dtype)

    def _prepare_reference_slots(
        self,
        slots: torch.Tensor,
        *,
        reference_raw_frames: int,
        reference_frames: int,
    ) -> torch.Tensor:
        raw_frames = int(reference_raw_frames)
        if raw_frames <= 0 or reference_frames <= 0:
            return slots[:, :0]
        ctx_slots = slots[:, : min(self.xssc_condition_frames, slots.shape[1])]
        if ctx_slots.shape[1] < raw_frames:
            repeat = ctx_slots[:, -1:].expand(-1, raw_frames - ctx_slots.shape[1], -1, -1)
            ctx_slots = torch.cat([ctx_slots, repeat], dim=1)
        elif ctx_slots.shape[1] > raw_frames:
            ctx_slots = ctx_slots[:, :raw_frames]
        return self._group_slots_to_latent_time(ctx_slots, target_frames=reference_frames)

    def _coordinate_query(self, frames: int, height: int, width: int, device, dtype) -> torch.Tensor:
        t = torch.linspace(-1.0, 1.0, frames, device=device, dtype=torch.float32)
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
        tt, yy, xx = torch.meshgrid(t, y, x, indexing="ij")
        ones = torch.ones_like(tt)
        coords = torch.stack((tt, yy, xx, ones), dim=-1)
        return self.coord_query(coords.to(dtype=dtype))

    def _apply_slot_dropout(self, slots: torch.Tensor) -> torch.Tensor:
        if self.training and self.slot_dropout > 0.0:
            keep = torch.rand(slots.shape[:3], device=slots.device) >= self.slot_dropout
            empty = ~keep.any(dim=2)
            if bool(empty.any()):
                keep_flat = keep.view(-1, keep.shape[-1])
                empty_flat = empty.view(-1)
                replacement = torch.randint(
                    keep.shape[-1],
                    (int(empty_flat.sum().item()),),
                    device=slots.device,
                )
                keep_flat[empty_flat] = False
                keep_flat[empty_flat, replacement] = True
                keep = keep_flat.view_as(keep)
            slots = slots * keep[..., None].to(slots.dtype) / (1.0 - self.slot_dropout)
            self.last_stats["xssc_slot_dropout_fraction"] = float((~keep).float().mean().item())
        else:
            self.last_stats["xssc_slot_dropout_fraction"] = 0.0
        return slots

    def _slots_to_dense(
        self,
        slots: torch.Tensor,
        *,
        target_height: int,
        target_width: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slots = slots.to(device=self.slot_norm.weight.device, dtype=self.slot_norm.weight.dtype)
        normed = self.slot_norm(slots)
        keys = self.slot_key(normed)
        values = self.slot_value(normed)
        queries = self._coordinate_query(
            int(slots.shape[1]),
            target_height,
            target_width,
            device=slots.device,
            dtype=keys.dtype,
        )
        logits = torch.einsum("thwd,btkd->bthwk", queries, keys) * (self.query_dim ** -0.5)
        assignment = torch.softmax(logits, dim=-1)
        dense = torch.einsum("bthwk,btkc->bthwc", assignment, values)
        dense = dense.permute(0, 4, 1, 2, 3).contiguous().to(dtype=dtype)
        return dense, assignment

    def slots_to_vace_context(
        self,
        slots: torch.Tensor,
        *,
        target_frames: int,
        target_height: int,
        target_width: int,
        reference_frames: int = 0,
        reference_raw_frames: int = 0,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        reference_frames = int(reference_frames)
        reference_raw_frames = int(reference_raw_frames)
        video_target_frames = int(target_frames) - reference_frames
        if reference_frames < 0 or reference_raw_frames < 0:
            raise ValueError("reference_frames and reference_raw_frames must be non-negative")
        if video_target_frames <= 0:
            raise ValueError(
                f"target_frames={target_frames} must exceed reference_frames={reference_frames}"
            )
        slots = slots.to(device=self.slot_norm.weight.device, dtype=self.slot_norm.weight.dtype)
        slots = self._apply_slot_dropout(slots)

        batch = int(slots.shape[0])
        raw_frames = int(slots.shape[1])
        visible_frames = min(self.xssc_condition_frames, raw_frames)
        frame_mask = torch.zeros(batch, raw_frames, device=slots.device, dtype=slots.dtype)
        frame_mask[:, :visible_frames] = 1

        video_slots = self._group_slots_to_latent_time(
            slots,
            target_frames=video_target_frames,
        )
        vace_slot_mask = self._group_frame_mask_to_latent_time(
            frame_mask,
            target_frames=video_target_frames,
        ).to(device=slots.device, dtype=dtype)
        vace_slot, assignment = self._slots_to_dense(
            video_slots,
            target_height=target_height,
            target_width=target_width,
            dtype=dtype,
        )
        vace_slot_mask_5d = vace_slot_mask[:, None, :, None, None]
        placeholder = self.future_placeholder.to(
            device=vace_slot.device,
            dtype=vace_slot.dtype,
        ).expand(batch, -1, video_target_frames, target_height, target_width)
        vace_slot = vace_slot * vace_slot_mask_5d + placeholder * (1 - vace_slot_mask_5d)

        # Official VACE equivalent with xSSC slots replacing vace_video.
        inactive = vace_slot * (1 - vace_slot_mask_5d) + 0 * vace_slot_mask_5d
        reactive = vace_slot * vace_slot_mask_5d + 0 * (1 - vace_slot_mask_5d)
        vace_video_latents = torch.cat([inactive, reactive], dim=1)

        if reference_frames > 0:
            reference_slots = self._prepare_reference_slots(
                slots,
                reference_raw_frames=reference_raw_frames,
                reference_frames=reference_frames,
            )
            reference_dense, _ = self._slots_to_dense(
                reference_slots,
                target_height=target_height,
                target_width=target_width,
                dtype=dtype,
            )
            reference_latents = torch.cat([reference_dense, torch.zeros_like(reference_dense)], dim=1)
            vace_video_latents = torch.cat([reference_latents, vace_video_latents], dim=2)

        vace_video_latents = self.video_norm(
            vace_video_latents.to(dtype=self.video_norm.weight.dtype)
        ).to(dtype=dtype)

        video_mask_latents = vace_slot_mask[:, None, :, None, None].expand(
            batch,
            self.vace_mask_dim,
            video_target_frames,
            target_height,
            target_width,
        )
        if reference_frames > 0:
            reference_mask = torch.zeros(
                batch,
                self.vace_mask_dim,
                reference_frames,
                target_height,
                target_width,
                device=video_mask_latents.device,
                dtype=video_mask_latents.dtype,
            )
            mask = torch.cat([reference_mask, video_mask_latents], dim=2)
        else:
            mask = video_mask_latents
        vace_context = torch.cat([vace_video_latents, mask.to(dtype=dtype)], dim=1)

        usage = assignment.mean(dim=(0, 1, 2, 3))
        entropy = -(usage * usage.clamp_min(1e-8).log()).sum()
        self.last_stats.update(
            {
                "xssc_condition_latent_frames": float(target_frames),
                "xssc_condition_reference_frames": float(reference_frames),
                "xssc_condition_reference_raw_frames": float(reference_raw_frames),
                "xssc_condition_video_latent_frames": float(video_target_frames),
                "xssc_condition_visible_raw_frames": float(visible_frames),
                "xssc_condition_slots": float(slots.shape[2]),
                "xssc_condition_video_mask_mean": float(vace_slot_mask.float().mean().detach().cpu()),
                "xssc_assignment_usage_entropy": float(entropy.detach().cpu()),
                "xssc_assignment_max_prob": float(assignment.max(dim=-1).values.mean().detach().cpu()),
            }
        )
        return vace_context

    def forward(
        self,
        *,
        video: torch.Tensor,
        latent_like: torch.Tensor,
        reference_frames: int = 0,
        reference_raw_frames: int = 0,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,C,T,H,W], got {tuple(video.shape)}")
        if latent_like.ndim != 5:
            raise ValueError(f"latent_like must be [B,C,T,H,W], got {tuple(latent_like.shape)}")
        if int(video.shape[2]) < self.xssc_condition_frames:
            raise ValueError(
                f"Need at least {self.xssc_condition_frames} frames for xSSC, "
                f"got {video.shape[2]}"
            )
        xssc_video = self._preprocess_xssc(video)
        slots = self.extract_slots(xssc_video)
        target_frames = int(latent_like.shape[2])
        target_height = int(latent_like.shape[3])
        target_width = int(latent_like.shape[4])
        return self.slots_to_vace_context(
            slots,
            target_frames=target_frames,
            target_height=target_height,
            target_width=target_width,
            reference_frames=reference_frames,
            reference_raw_frames=reference_raw_frames,
            dtype=output_dtype,
        )


class XSSCVACENoiseInitializer(PipelineUnit):
    """Noise initializer matching whole-video ctx reference encoding."""

    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"),
            output_params=("noise",),
        )

    def process(self, pipe, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = wan_latent_frame_count(num_frames)
        reference_frames = reference_latent_frame_count(vace_reference_image)
        length += reference_frames
        shape = (
            1,
            pipe.vae.model.z_dim,
            length,
            height // pipe.vae.upsampling_factor,
            width // pipe.vae.upsampling_factor,
        )
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if reference_frames > 0:
            noise = torch.cat((noise[:, :, -reference_frames:], noise[:, :, :-reference_frames]), dim=2)
        return {"noise": noise}


class XSSCVACEReferenceVideoEmbedder(PipelineUnit):
    """Input-video embedder with whole-clip ctx reference semantics."""

    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "noise",
                "tiled",
                "tile_size",
                "tile_stride",
                "vace_reference_image",
                "framewise_decoding",
            ),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe,
        input_video,
        noise,
        tiled,
        tile_size,
        tile_stride,
        vace_reference_image,
        framewise_decoding,
    ):
        if input_video is None:
            return {"latents": noise}

        pipe.load_models_to_device(self.onload_model_names)
        input_video = pipe.preprocess_video(input_video)
        if framewise_decoding:
            input_latents = pipe.vae.encode_framewise(input_video, device=pipe.device)
        else:
            input_latents = pipe.vae.encode(
                input_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)

        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]
            reference_video = pipe.preprocess_video(vace_reference_image)
            reference_latents = pipe.vae.encode(
                reference_video,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.cat([reference_latents, input_latents], dim=2)

        if pipe.scheduler.training:
            if noise.shape != input_latents.shape:
                raise ValueError(
                    "xSSC-VACE noise/input_latents shape mismatch after reference-video "
                    f"prepend: noise={tuple(noise.shape)}, input_latents={tuple(input_latents.shape)}"
                )
            return {"latents": noise, "input_latents": input_latents}

        latents = pipe.scheduler.add_noise(
            input_latents,
            noise,
            timestep=pipe.scheduler.timesteps[0],
        )
        return {"latents": latents}


class XSSCVACEContextUnit(PipelineUnit):
    """Pipeline unit replacing DiffSynth's image/video VACE conditioner."""

    def __init__(self, conditioner: XSSCVACEContextConditioner):
        super().__init__(
            input_params=("input_video", "input_latents", "latents", "vace_reference_image", "vace_scale"),
            output_params=("vace_context", "vace_scale"),
        )
        self.conditioner = conditioner

    def process(self, pipe, input_video, input_latents, latents, vace_reference_image, vace_scale):
        if input_video is None:
            return {"vace_context": None, "vace_scale": vace_scale}
        latent_like = input_latents if input_latents is not None else latents
        if vace_reference_image is None:
            reference_raw_frames = 0
            reference_frames = 0
        elif isinstance(vace_reference_image, list):
            reference_raw_frames = len(vace_reference_image)
            reference_frames = wan_latent_frame_count(reference_raw_frames)
        else:
            reference_raw_frames = 1
            reference_frames = 1
        video = pipe.preprocess_video(input_video, torch_dtype=torch.float32, device=pipe.device)
        vace_context = self.conditioner(
            video=video,
            latent_like=latent_like,
            reference_frames=reference_frames,
            reference_raw_frames=reference_raw_frames,
            output_dtype=pipe.torch_dtype,
        )
        return {"vace_context": vace_context, "vace_scale": vace_scale}
