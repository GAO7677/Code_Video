"""Train Wan2.2 with frozen xSSC context slots as object cross-attention tokens.

This file was copied from train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py
and intentionally lives in train_xSSC. It replaces the complete Stage1A/Stage1B
token-building frontend (grounding, CoTracker, VGGT, JEPA, ObjectTubeProjector,
ObjectAuxHeads, and ObjectConditionAdapter) with:

    context video -> frozen RandSFQ2 -> slotz [B, Tc, 7, 256]
                  -> LayerNorm + Linear(256, Wan dim) + time embedding
                  -> [B, Tc * 7, Wan dim] -> object cross-attention

The object cross-attention base weights are initialized from Wan's text
cross-attention with the frozen physical-state LoRA baked in, then frozen. Only
its Q/K/V/O LoRA adapters, the slot projection, time embedding, and object gates
are trainable. The Wan base, physical-state LoRA, xSSC, and DINO backbone stay
frozen. Exactly eight context frames are passed to xSSC, so no future information
leaks into the conditioning path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model

# Importing train_v_newtrain installs the DiffSynth path shim selected by the
# --diffsynth_root command-line argument.
import code_vjepa_vggt.train_v_newtrain as tvn
from code_vjepa_vggt.context_wan_v_newtrain import (
    enable_object_condition_branch,
    flow_match_context_sft_loss,
)
from code_vjepa_vggt.data.mixed_replay_no_gt_box_dataset import (
    KubricReplayNoGTBoxDataset,
    OpenVidNoGTBoxDataset,
    WeightedNoGTBoxMixture,
)
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)

from diffsynth.diffusion import ModelLogger


XSSC_IMAGENET_MEAN = (123.675, 116.28, 103.53)
XSSC_IMAGENET_STD = (58.395, 57.12, 57.375)
XSSC_NUM_CONTEXT_FRAMES = 8
DEFAULT_XSSC_ROOT = "/home/gaoya/Code_Video/xSSC-main"
DEFAULT_XSSC_CONFIG = f"{DEFAULT_XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py"
DEFAULT_XSSC_CHECKPOINT = "/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth"


def _unwrap_linear(module: nn.Module) -> nn.Linear:
    base_layer = getattr(module, "base_layer", module)
    if not isinstance(base_layer, nn.Linear):
        raise TypeError(f"Expected Linear or PEFT-wrapped Linear, got {type(module)!r}")
    return base_layer


def _effective_linear_weight(module: nn.Module) -> torch.Tensor:
    """Return base weight plus any active, unmerged frozen PEFT adapters."""
    base_layer = _unwrap_linear(module)
    weight = base_layer.weight.detach().clone()
    if not hasattr(module, "get_delta_weight") or bool(getattr(module, "disable_adapters", False)):
        return weight
    merged_adapters = set(getattr(module, "merged_adapters", ()))
    for adapter_name in getattr(module, "active_adapters", ()):
        if adapter_name not in merged_adapters:
            weight.add_(module.get_delta_weight(adapter_name).to(weight))
    return weight


def _initialize_object_attention_from_text(block: nn.Module) -> None:
    """Copy effective text cross-attention weights into the new object branch."""
    with torch.no_grad():
        for name in ("q", "k", "v", "o"):
            source_module = getattr(block.cross_attn, name)
            source = _unwrap_linear(source_module)
            target = _unwrap_linear(getattr(block.object_cross_attn, name))
            target.weight.copy_(_effective_linear_weight(source_module))
            if target.bias is not None:
                if source.bias is None:
                    target.bias.zero_()
                else:
                    target.bias.copy_(source.bias)
        block.object_cross_attn.norm_q.load_state_dict(block.cross_attn.norm_q.state_dict())
        block.object_cross_attn.norm_k.load_state_dict(block.cross_attn.norm_k.state_dict())
        block.norm4.load_state_dict(block.norm3.state_dict())


def _inject_object_attention_lora(
    block: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    config = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=["q", "k", "v", "o"],
        bias="none",
    )
    block.object_cross_attn = inject_adapter_in_model(config, block.object_cross_attn)


def _is_trainable_object_dit_parameter(name: str) -> bool:
    is_object_lora = ".object_cross_attn." in name and (
        ".lora_A." in name or ".lora_B." in name
    )
    return is_object_lora or ".object_gate" in name


def merge_batched_pipeline_dicts(dicts: list[dict]) -> dict:
    """Merge independently preprocessed samples into a true tensor batch."""
    merged = {}
    for key in dicts[0]:
        values = [item[key] for item in dicts]
        if all(isinstance(value, torch.Tensor) for value in values):
            first = values[0]
            can_concat = all(
                value.ndim > 0
                and int(value.shape[0]) == 1
                and tuple(value.shape[1:]) == tuple(first.shape[1:])
                for value in values
            )
            if can_concat:
                merged[key] = torch.cat(values, dim=0)
            elif all(
                value.shape == first.shape and torch.equal(value, first)
                for value in values[1:]
            ):
                merged[key] = first
            else:
                raise ValueError(
                    f"Cannot batch tensor input {key!r}: "
                    f"{[tuple(value.shape) for value in values]}"
                )
        else:
            merged[key] = values[0]
    return merged


class GroupedBatchDataset(torch.utils.data.Dataset):
    """Group raw samples while retaining the parent's batch-size-one DataLoader."""

    def __init__(self, dataset, batch_size: int) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if len(self.dataset) <= 0:
            raise ValueError("Cannot batch an empty dataset")
        source_weights = getattr(self.dataset, "sample_weights", None)
        self.sample_weights = None
        if source_weights is not None:
            self.sample_weights = [
                sum(
                    float(source_weights[(start + offset) % len(self.dataset)])
                    for offset in range(self.batch_size)
                )
                for start in range(0, len(self.dataset), self.batch_size)
            ]
        self.load_from_cache = False
        self.dataset_stats = {
            "kind": "grouped_batch",
            "per_gpu_batch_size": self.batch_size,
            "num_groups": len(self),
            "source": getattr(self.dataset, "dataset_stats", None),
        }

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index: int) -> list[dict]:
        start = int(index) * self.batch_size
        return [
            self.dataset[(start + offset) % len(self.dataset)]
            for offset in range(self.batch_size)
        ]


def _load_xssc_model(
    *,
    xssc_root: str,
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Build RandSFQ2 from its official config and strictly load its checkpoint."""
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
    # DINO2ViT hard-codes pretrained=True, but the official xSSC checkpoint
    # contains the complete DINO state. Build the architecture offline and then
    # strictly restore every parameter from that checkpoint.
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
    # The xSSC decoder is only a pretraining objective. Slot conditioning needs
    # encode_backbone/encode_project/initializ/aggregat/transit, not reconstruction.
    model.decode = None
    model.requires_grad_(False)
    model.eval()
    model.to(device=device)
    return model, slot_dim, num_slots


class XSSCContextSlotsWanModule(tvn.WanTrainingModule):
    """Wan training module conditioned directly on frozen context-only xSSC slots."""

    def __init__(
        self,
        *args,
        xssc_root: str,
        xssc_config: str,
        xssc_checkpoint: str,
        xssc_input_size: int = 256,
        xssc_max_time_steps: int = 64,
        object_lora_rank: int = 32,
        object_lora_alpha: float = 32.0,
        object_lora_dropout: float = 0.0,
        xssc_slot_track_dropout: float = 0.0,
        object_gate_init: float = 0.1,
        lambda_main: float = 1.0,
        lambda_object_context_reg: float = 0.0,
        **kwargs,
    ) -> None:
        # The parent must not construct the legacy object frontend. We inject only
        # the Wan cross-attention branch after the base/LoRA pipeline is ready.
        kwargs["enable_object_branch"] = False
        kwargs["freeze_non_object_trainables"] = True
        kwargs["train_object_pooler"] = False
        kwargs["train_object_aux_heads"] = False
        kwargs["train_object_adapter"] = False
        kwargs["train_object_dit_branch"] = False
        kwargs["no_context_ratio"] = 0.0
        super().__init__(
            *args,
            object_gate_init=object_gate_init,
            lambda_main=lambda_main,
            lambda_object_context_reg=lambda_object_context_reg,
            **kwargs,
        )

        self.enable_object_branch = True
        self.lambda_main = float(lambda_main)
        self.lambda_object_context_reg = float(lambda_object_context_reg)
        self.xssc_input_size = int(xssc_input_size)
        self.xssc_max_time_steps = int(xssc_max_time_steps)
        self.object_lora_rank = int(object_lora_rank)
        self.object_lora_alpha = float(object_lora_alpha)
        self.object_lora_dropout = float(object_lora_dropout)
        self.xssc_slot_track_dropout = float(xssc_slot_track_dropout)
        self._last_slot_dropout_fraction = 0.0
        self._last_retained_slots_per_sample = float(self.xssc_num_slots) if hasattr(self, "xssc_num_slots") else 0.0
        if self.fixed_num_context_frames != XSSC_NUM_CONTEXT_FRAMES:
            raise ValueError(
                "xSSC training requires exactly "
                f"{XSSC_NUM_CONTEXT_FRAMES} context frames, got {self.fixed_num_context_frames}"
            )
        if self.object_lora_rank <= 0:
            raise ValueError(f"object_lora_rank must be positive, got {self.object_lora_rank}")
        if not 0.0 <= self.object_lora_dropout < 1.0:
            raise ValueError(
                f"object_lora_dropout must be in [0, 1), got {self.object_lora_dropout}"
            )
        if not 0.0 <= self.xssc_slot_track_dropout < 1.0:
            raise ValueError(
                "xssc_slot_track_dropout must be in [0, 1), got "
                f"{self.xssc_slot_track_dropout}"
            )

        dit = enable_object_condition_branch(
            self.pipe.dit,
            object_gate_init=float(object_gate_init),
            reinitialize_object_branch=True,
        )
        # xSSC tokens are projected directly to dit.dim, so the old text-dimension
        # object_embedding would be both redundant and shape-incompatible.
        dit.object_embedding = None
        for block in dit.blocks:
            _initialize_object_attention_from_text(block)
            _inject_object_attention_lora(
                block,
                rank=self.object_lora_rank,
                alpha=self.object_lora_alpha,
                dropout=self.object_lora_dropout,
            )
        for name, param in dit.named_parameters():
            param.requires_grad = _is_trainable_object_dit_parameter(name)

        model_device = dit.patch_embedding.weight.device
        model_dtype = dit.patch_embedding.weight.dtype
        self.xssc, self.xssc_slot_dim, self.xssc_num_slots = _load_xssc_model(
            xssc_root=xssc_root,
            config_path=xssc_config,
            checkpoint_path=xssc_checkpoint,
            device=model_device,
        )
        if self.xssc_slot_dim != 256:
            raise ValueError(f"Expected 256-d xSSC slots, got {self.xssc_slot_dim}")
        self._last_retained_slots_per_sample = float(self.xssc_num_slots)

        hidden_dim = int(dit.dim)
        self.slot_norm = nn.LayerNorm(self.xssc_slot_dim)
        self.slot_projector = nn.Linear(self.xssc_slot_dim, hidden_dim)
        self.time_embedding = nn.Embedding(self.xssc_max_time_steps, hidden_dim)
        nn.init.normal_(self.slot_projector.weight, std=0.02)
        nn.init.zeros_(self.slot_projector.bias)
        nn.init.normal_(self.time_embedding.weight, std=0.02)
        self.slot_norm.to(device=model_device, dtype=model_dtype)
        self.slot_projector.to(device=model_device, dtype=model_dtype)
        self.time_embedding.to(device=model_device, dtype=model_dtype)

    def train(self, mode: bool = True):
        super().train(mode)
        # nn.Module.train() is recursive; force the frozen xSSC transition dropout
        # off after every mode switch.
        self.xssc.eval()
        return self

    def _prepare_pipeline_sample(self, sample):
        inputs = self.get_pipeline_inputs(sample)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return inputs

    def _forward_sample_batch(self, samples: list[dict]) -> torch.Tensor:
        prepared = [self._prepare_pipeline_sample(sample) for sample in samples]
        shared = merge_batched_pipeline_dicts([item[0] for item in prepared])
        positive = merge_batched_pipeline_dicts([item[1] for item in prepared])
        raw_samples = [item[0]["raw_sample"] for item in prepared]
        shared["raw_sample"] = dict(raw_samples[0])
        shared["raw_sample"]["context_video"] = torch.stack(
            [sample["context_video"] for sample in raw_samples], dim=0
        )
        shared["raw_sample"]["num_context_frames"] = XSSC_NUM_CONTEXT_FRAMES
        batch_size = int(shared["input_latents"].shape[0])
        if batch_size != len(samples):
            raise RuntimeError(
                f"Prepared latent batch mismatch: got {batch_size}, expected {len(samples)}"
            )
        loss, metrics = self._compute_object_losses(self.pipe, shared, positive)
        metrics["train/batch_size_per_gpu"] = float(batch_size)
        self.last_train_metrics = metrics
        return loss

    def forward(self, data, inputs=None):
        if isinstance(data, list):
            if inputs is not None:
                raise ValueError("Batched raw samples cannot be combined with prepared inputs")
            return self._forward_sample_batch(data)
        return super().forward(data, inputs=inputs)

    def sample_context_spec(self, video, raw_sample=None):
        if raw_sample is None or "context_video" not in raw_sample:
            raise ValueError("xSSC training requires raw_sample.context_video")
        context_video = raw_sample["context_video"]
        if not isinstance(context_video, torch.Tensor) or context_video.ndim < 2:
            raise TypeError("raw_sample.context_video must be a [C,T,H,W] tensor")
        time_steps = int(context_video.shape[1])
        if time_steps != XSSC_NUM_CONTEXT_FRAMES:
            raise ValueError(
                f"Expected {XSSC_NUM_CONTEXT_FRAMES} context frames, got {time_steps}"
            )
        return self._finalize_context_spec(
            "fixed_full_context",
            range(XSSC_NUM_CONTEXT_FRAMES),
            ctx_max_length=XSSC_NUM_CONTEXT_FRAMES - 1,
        )

    def trainable_modules(self) -> list[nn.Parameter]:
        params = list(self.slot_norm.parameters())
        params.extend(self.slot_projector.parameters())
        params.extend(self.time_embedding.parameters())
        params.extend(
            param
            for name, param in self.pipe.dit.named_parameters()
            if _is_trainable_object_dit_parameter(name)
        )
        unique: list[nn.Parameter] = []
        seen: set[int] = set()
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        return unique

    def _preprocess_xssc(self, context_video: torch.Tensor) -> torch.Tensor:
        """Convert [B,C,T,H,W] in [-1,1] to xSSC [B,T,C,256,256]."""
        frames = context_video.permute(0, 2, 1, 3, 4).float()
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
    def _extract_xssc_slots(self, video: torch.Tensor) -> torch.Tensor:
        """Run the frozen RandSFQ2 encoder/slot recurrence without its decoder."""
        self.xssc.eval()
        batch, time_steps, channels, height, width = video.shape
        flat_video = video.flatten(0, 1)
        autocast_enabled = flat_video.device.type == "cuda"
        with torch.autocast(device_type=flat_video.device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
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
            raise RuntimeError("xSSC received zero context frames")
        return slots

    def _build_object_context(self, context_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xssc_video = self._preprocess_xssc(context_video)
        slots = self._extract_xssc_slots(xssc_video)
        time_steps = int(slots.shape[1])
        if time_steps > self.xssc_max_time_steps:
            raise ValueError(
                f"Context length {time_steps} exceeds xssc_max_time_steps={self.xssc_max_time_steps}"
            )
        target_dtype = self.slot_norm.weight.dtype
        slots_for_projection = slots.to(device=self.slot_norm.weight.device, dtype=target_dtype)
        tokens = self.slot_projector(self.slot_norm(slots_for_projection))
        time_ids = torch.arange(time_steps, device=tokens.device)
        time_tokens = self.time_embedding(time_ids).view(1, time_steps, 1, -1)
        tokens = tokens + time_tokens.to(dtype=tokens.dtype)
        if self.training and self.xssc_slot_track_dropout > 0.0:
            batch, _, num_slots, _ = tokens.shape
            keep = torch.rand(batch, num_slots, device=tokens.device) >= self.xssc_slot_track_dropout
            empty_rows = ~keep.any(dim=1)
            if bool(empty_rows.any()):
                replacement = torch.randint(num_slots, (int(empty_rows.sum().item()),), device=tokens.device)
                keep[empty_rows] = False
                keep[empty_rows, replacement] = True
            keep_scale = keep.to(dtype=tokens.dtype) / (1.0 - self.xssc_slot_track_dropout)
            tokens = tokens * keep_scale[:, None, :, None]
            self._last_slot_dropout_fraction = float((~keep).float().mean().item())
            self._last_retained_slots_per_sample = float(keep.float().sum(dim=1).mean().item())
        else:
            self._last_slot_dropout_fraction = 0.0
            self._last_retained_slots_per_sample = float(tokens.shape[2])
        batch, _, num_slots, hidden_dim = tokens.shape
        return tokens.reshape(batch, time_steps * num_slots, hidden_dim), slots

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        sample = inputs_shared["raw_sample"]
        num_context_frames = int(sample["num_context_frames"])
        if num_context_frames != XSSC_NUM_CONTEXT_FRAMES:
            raise ValueError(
                f"Expected {XSSC_NUM_CONTEXT_FRAMES} sampled context frames, "
                f"got {num_context_frames}"
            )
        context_video = sample["context_video"]
        if context_video.ndim == 4:
            context_video = context_video.unsqueeze(0)
        elif context_video.ndim != 5:
            raise ValueError(
                "context_video must be [C,T,H,W] or [B,C,T,H,W], "
                f"got shape={tuple(context_video.shape)}"
            )
        if int(context_video.shape[2]) != XSSC_NUM_CONTEXT_FRAMES:
            raise ValueError(
                f"Expected context_video T={XSSC_NUM_CONTEXT_FRAMES}, "
                f"got shape={tuple(context_video.shape)}"
            )
        context_video = context_video.to(device=pipe.device, dtype=pipe.torch_dtype)
        object_context, slots = self._build_object_context(context_video)

        loss_main = flow_match_context_sft_loss(
            pipe,
            **inputs_shared,
            **inputs_posi,
            object_context=object_context,
        )
        object_context_reg = object_context.square().mean()
        total = self.lambda_main * loss_main + self.lambda_object_context_reg * object_context_reg
        context_abs = object_context.detach().abs()
        slot_abs = slots.detach().abs()
        gate_abs_means = []
        gate_abs_maxes = []
        for block in self.pipe.dit.blocks:
            object_gate = getattr(block, "object_gate", None)
            if object_gate is None:
                continue
            gate_abs = torch.tanh(object_gate.detach().float()).abs()
            gate_abs_means.append(gate_abs.mean())
            gate_abs_maxes.append(gate_abs.max())
        gate_abs_mean = (
            float(torch.stack(gate_abs_means).mean().item()) if gate_abs_means else 0.0
        )
        gate_abs_max = (
            float(torch.stack(gate_abs_maxes).max().item()) if gate_abs_maxes else 0.0
        )
        metrics = {
            "train/loss_total": float(total.detach().item()),
            "train/loss_main": float(loss_main.detach().item()),
            "train/loss_object_context_reg": float(object_context_reg.detach().item()),
            "train/xssc_context_frames": float(num_context_frames),
            "train/xssc_slots_per_frame": float(self.xssc_num_slots),
            "train/xssc_token_count": float(object_context.shape[1]),
            "train/xssc_slot_abs_mean": float(slot_abs.mean().item()),
            "train/xssc_slot_dropout_fraction": self._last_slot_dropout_fraction,
            "train/xssc_retained_slots_per_sample": self._last_retained_slots_per_sample,
            "train/object_slot_dropout_applied": float(self._last_slot_dropout_fraction > 0.0),
            "train/object_count_before_dropout": float(self.xssc_num_slots),
            "train/object_count_after_dropout": self._last_retained_slots_per_sample,
            "train/object_context_abs_max": float(context_abs.max().item()),
            "train/object_context_abs_mean": float(context_abs.mean().item()),
            "train/object_gate_tanh_abs_mean": gate_abs_mean,
            "train/object_gate_tanh_abs_max": gate_abs_max,
        }
        return total, metrics


def _patch_general_config_conflict_handler() -> None:
    if getattr(tvn, "_xssc_conflict_patched", False):
        return
    original = tvn.add_general_config

    def patched(parser: argparse.ArgumentParser):
        parser = original(parser)
        parser.conflict_handler = "resolve"
        parser._optionals.conflict_handler = "resolve"
        return parser

    tvn.add_general_config = patched
    tvn._xssc_conflict_patched = True


def build_parser() -> argparse.ArgumentParser:
    _patch_general_config_conflict_handler()
    parser = tvn.wan_parser()
    parser.description = "Train Wan2.2 with frozen context-only xSSC slots."
    group = parser.add_argument_group("xssc_context_slots")
    group.add_argument("--xssc_root", default=DEFAULT_XSSC_ROOT)
    group.add_argument("--xssc_config", default=DEFAULT_XSSC_CONFIG)
    group.add_argument("--xssc_checkpoint", default=DEFAULT_XSSC_CHECKPOINT)
    group.add_argument("--xssc_input_size", type=int, default=256)
    group.add_argument("--xssc_max_time_steps", type=int, default=64)
    group.add_argument("--object_lora_rank", type=int, default=32)
    group.add_argument("--object_lora_alpha", type=float, default=32.0)
    group.add_argument("--object_lora_dropout", type=float, default=0.0)
    group.add_argument("--xssc_slot_track_dropout", type=float, default=0.0)
    group.add_argument("--train_batch_size", type=int, default=1)
    for action in parser._actions:
        if action.dest == "dataset_type" and "xssc_replay_mix" not in action.choices:
            action.choices = [*action.choices, "xssc_replay_mix"]
            break
    dataset = parser.add_argument_group("xssc_replay_mix")
    dataset.add_argument("--pybullet0713_root", type=str, default=None)
    dataset.add_argument("--pybullet0713_split", default="train", choices=["train", "val", "test", "all"])
    dataset.add_argument("--pybullet0713_sampling_strategy", default="prefix", choices=["prefix", "uniform"])
    dataset.add_argument("--pybullet0713_init_scan_limit", type=int, default=None)
    dataset.add_argument("--pybullet0713_family", action="append", default=None)
    dataset.add_argument("--pybullet0713_vae_cache_dir", type=str, default=None)
    dataset.add_argument("--pybullet0713_prompt_cache_dir", type=str, default=None)
    dataset.add_argument("--kubric_root", type=str, default=None)
    dataset.add_argument("--kubric_split", default="train", choices=["train", "val", "test", "all"])
    dataset.add_argument("--kubric_sampling_strategy", default="prefix", choices=["prefix", "uniform"])
    dataset.add_argument("--kubric_init_scan_limit", type=int, default=None)
    dataset.add_argument("--kubric_cache_root", default="/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset")
    dataset.add_argument("--kubric_replay_index_num_frames", type=int, default=69)
    dataset.add_argument("--kubric_replay_index_num_context_frames", type=int, default=20)
    dataset.add_argument("--openvid_root", type=str, default=None)
    dataset.add_argument("--openvid_max_samples", type=int, default=None)
    dataset.add_argument("--mixture_pybullet_ratio", type=float, default=0.30)
    dataset.add_argument("--mixture_kubric_ratio", type=float, default=0.30)
    dataset.add_argument("--mixture_openvid_ratio", type=float, default=0.40)
    return parser


def build_dataset(args: argparse.Namespace):
    if args.dataset_type != "xssc_replay_mix":
        return tvn.build_dataset(args)
    if (int(args.num_frames) - 1) % 4 != 0:
        raise ValueError("xssc_replay_mix num_frames must satisfy 4n+1")
    resolution = (int(args.height), int(args.width))
    source_probabilities = {
        "pybullet": float(args.mixture_pybullet_ratio),
        "kubric": float(args.mixture_kubric_ratio),
        "openvid": float(args.mixture_openvid_ratio),
    }
    if any(value < 0.0 for value in source_probabilities.values()):
        raise ValueError(
            f"xssc_replay_mix probabilities must be non-negative: {source_probabilities}"
        )
    if sum(source_probabilities.values()) <= 0.0:
        raise ValueError(
            f"xssc_replay_mix requires at least one positive source: {source_probabilities}"
        )

    datasets = []
    source_names = []
    active_probabilities = []
    if source_probabilities["pybullet"] > 0.0:
        if not args.pybullet0713_root:
            raise ValueError("Positive PyBullet ratio requires --pybullet0713_root")
        datasets.append(
            PyBullet0713NoGTBoxDataset(
                root=args.pybullet0713_root,
                split=args.pybullet0713_split,
                resolution=resolution,
                num_frames=args.num_frames,
                num_context_frames=args.fixed_num_context_frames,
                sampling_strategy=args.pybullet0713_sampling_strategy,
                families=args.pybullet0713_family,
                init_scan_limit=args.pybullet0713_init_scan_limit,
                vae_cache_dir=args.pybullet0713_vae_cache_dir,
                vae_checkpoint_path=Path(args.wan_root) / "Wan2.2_VAE.pth",
                prompt_cache_dir=args.pybullet0713_prompt_cache_dir,
                text_encoder_checkpoint_path=(
                    Path(args.wan_root) / "models_t5_umt5-xxl-enc-bf16.pth"
                ),
                tokenizer_path=args.tokenizer_path,
            )
        )
        source_names.append("pybullet")
        active_probabilities.append(source_probabilities["pybullet"])
    if source_probabilities["kubric"] > 0.0:
        if not args.kubric_root:
            raise ValueError("Positive Kubric ratio requires --kubric_root")
        datasets.append(
            KubricReplayNoGTBoxDataset(
                root=args.kubric_root,
                split=args.kubric_split,
                resolution=resolution,
                num_frames=args.num_frames,
                num_context_frames=args.fixed_num_context_frames,
                index_num_frames=args.kubric_replay_index_num_frames,
                index_num_context_frames=args.kubric_replay_index_num_context_frames,
                sampling_strategy=args.kubric_sampling_strategy,
                seed=42,
                init_scan_limit=args.kubric_init_scan_limit,
                cache_root=args.kubric_cache_root,
            )
        )
        source_names.append("kubric")
        active_probabilities.append(source_probabilities["kubric"])
    if source_probabilities["openvid"] > 0.0:
        if not args.openvid_root:
            raise ValueError("Positive OpenVid ratio requires --openvid_root")
        datasets.append(
            OpenVidNoGTBoxDataset(
                root=args.openvid_root,
                resolution=resolution,
                num_frames=args.num_frames,
                num_context_frames=args.fixed_num_context_frames,
                max_samples=args.openvid_max_samples,
            )
        )
        source_names.append("openvid")
        active_probabilities.append(source_probabilities["openvid"])
    return WeightedNoGTBoxMixture(
        datasets=tuple(datasets),
        source_names=tuple(source_names),
        source_probabilities=tuple(active_probabilities),
    )


def build_model(args: argparse.Namespace, accelerator) -> XSSCContextSlotsWanModule:
    return XSSCContextSlotsWanModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        context_sampling_profile=args.context_sampling_profile,
        min_context_frames=args.min_context_frames,
        max_context_ratio=args.max_context_ratio,
        context_frame_choices=args.context_frame_choices,
        context_length_sampling=args.context_length_sampling,
        context_reference_frames=args.context_reference_frames,
        context_reference_prefixes=args.context_reference_prefixes,
        prefix_context_ratio=args.prefix_context_ratio,
        first_frame_context_ratio=args.first_frame_context_ratio,
        sparse_context_ratio=args.sparse_context_ratio,
        random_context_ratio=args.random_context_ratio,
        no_context_ratio=args.no_context_ratio,
        fixed_num_context_frames=args.fixed_num_context_frames,
        ctx_max_length=args.ctx_max_length,
        object_gate_init=args.object_gate_init,
        lambda_main=args.lambda_main,
        lambda_object_context_reg=args.lambda_object_context_reg,
        xssc_root=args.xssc_root,
        xssc_config=args.xssc_config,
        xssc_checkpoint=args.xssc_checkpoint,
        xssc_input_size=args.xssc_input_size,
        xssc_max_time_steps=args.xssc_max_time_steps,
        object_lora_rank=args.object_lora_rank,
        object_lora_alpha=args.object_lora_alpha,
        object_lora_dropout=args.object_lora_dropout,
        xssc_slot_track_dropout=args.xssc_slot_track_dropout,
    )


def _log_stage_summary(accelerator, model: XSSCContextSlotsWanModule, args: argparse.Namespace) -> None:
    if not accelerator.is_main_process:
        return
    object_lora_params = [
        param
        for name, param in model.pipe.dit.named_parameters()
        if param.requires_grad
        and ".object_cross_attn." in name
        and (".lora_A." in name or ".lora_B." in name)
    ]
    object_gate_params = [
        param
        for name, param in model.pipe.dit.named_parameters()
        if param.requires_grad and ".object_gate" in name
    ]
    projector_params = list(model.slot_norm.parameters()) + list(model.slot_projector.parameters())
    projector_params += list(model.time_embedding.parameters())
    total = sum(param.numel() for param in model.trainable_modules())
    lines = [
        "=" * 78,
        "xSSC context-slot object conditioning",
        "=" * 78,
        f"Frozen Wan base: {args.wan_root}",
        f"Frozen physical-state LoRA: {args.lora_checkpoint}",
        f"Frozen xSSC checkpoint: {args.xssc_checkpoint}",
        "Context policy: fixed full 8-frame context (text-only disabled)",
        f"Per-GPU training batch size: {args.train_batch_size}",
        f"xSSC shape: [B, 8, {model.xssc_num_slots}, {model.xssc_slot_dim}]",
        f"Object token shape: [B, {8 * model.xssc_num_slots}, {model.pipe.dit.dim}]",
        "Object attention base: text cross-attention + physical LoRA, baked and frozen",
        f"Object LoRA: rank={model.object_lora_rank}, alpha={model.object_lora_alpha:g}, "
        f"dropout={model.object_lora_dropout:g}",
        f"xSSC slot-track dropout: {model.xssc_slot_track_dropout:g} "
        "(same slot mask across all 8 context frames)",
        f"Trainable projector/time params: {sum(p.numel() for p in projector_params):,}",
        f"Trainable object-attention LoRA params: {sum(p.numel() for p in object_lora_params):,}",
        f"Trainable object-gate params: {sum(p.numel() for p in object_gate_params):,}",
        f"Total trainable params: {total:,}",
        "Legacy Stage1A/Grounding/CoTracker/VGGT/JEPA modules: not constructed",
        "=" * 78,
    ]
    accelerator.print("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = tvn.prepare_args(parser.parse_args())
    if int(args.fixed_num_context_frames) != XSSC_NUM_CONTEXT_FRAMES:
        parser.error(
            f"--fixed_num_context_frames must be {XSSC_NUM_CONTEXT_FRAMES} for xSSC training"
        )
    if int(args.train_batch_size) <= 0:
        parser.error("--train_batch_size must be positive")
    args.no_context_ratio = 0.0
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    dataset = build_dataset(args)
    if int(args.train_batch_size) > 1:
        dataset = GroupedBatchDataset(dataset, args.train_batch_size)
    headonly_val_config = tvn.build_headonly_val_config(args)
    headonly_val_dataset = tvn.build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = tvn.build_headonly_val_dataloader(headonly_val_dataset, args)
    model = build_model(args, accelerator)

    if args.stage2_resume_from is not None:
        resume_info = tvn._load_filtered_checkpoint_into_model(
            model,
            tvn.resolve_lora_checkpoint_for_resume(args.stage2_resume_from),
            include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
            include_substrings=(".object_cross_attn.", ".object_gate"),
        )
        expected_resume_count = sum(
            1 for _, param in model.named_parameters() if param.requires_grad
        )
        if (
            resume_info["loaded_count"] != expected_resume_count
            or resume_info["skipped_shape_mismatch"]
        ):
            raise RuntimeError(
                "Incomplete or incompatible xSSC object-LoRA resume checkpoint: "
                f"loaded={resume_info['loaded_count']}/{expected_resume_count}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded xSSC-object resume weights: "
                f"loaded_count={resume_info['loaded_count']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    _log_stage_summary(accelerator, model, args)
    model_logger = ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict = {}

    try:
        if args.task in ("sft:data_process", "direct_distill:data_process"):
            tvn.launch_data_process_task(accelerator, dataset, model, model_logger, args=args)
        else:
            tvn.train_loop(
                accelerator,
                dataset,
                model,
                model_logger,
                args,
                runtime_state=runtime_state,
                headonly_val_dataloader=headonly_val_dataloader,
                headonly_val_config=headonly_val_config,
            )
    except (KeyboardInterrupt, tvn.TrainingInterrupted) as exc:
        interrupted_checkpoint_path = tvn.training_checkpoint_file(
            tvn.get_checkpoint_dir(args), "interrupted-latest"
        )
        accelerator.print(
            f"Training interrupted at step {model_logger.num_steps}; saving checkpoint."
        )
        model_logger.save_model(accelerator, model, interrupted_checkpoint_path)
        optimizer = runtime_state.get("optimizer")
        scheduler = runtime_state.get("scheduler")
        progress = runtime_state.get(
            "progress", {"global_step": 0, "epoch_id": 0, "batch_in_epoch": 0}
        )
        if optimizer is not None and scheduler is not None:
            tvn.save_training_state(
                accelerator=accelerator,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=progress.get("global_step", 0),
                epoch_id=progress.get("epoch_id", 0),
                batch_in_epoch=progress.get("batch_in_epoch", 0),
                model_logger=model_logger,
                state_path=tvn.training_state_file(
                    tvn.get_checkpoint_dir(args), "interrupted-latest"
                ),
            )
        accelerator.end_training()
        tvn.restore_interrupt_handlers(previous_handlers)
        raise SystemExit(130) from exc

    accelerator.end_training()
    tvn.restore_interrupt_handlers(previous_handlers)


if __name__ == "__main__":
    main()
