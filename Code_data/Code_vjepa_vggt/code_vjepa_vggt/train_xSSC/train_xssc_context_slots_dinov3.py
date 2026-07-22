"""Train Wan2.2 with frozen DINOv3 xSSC MOVi-C context slots.

This entry point is based on ``train_xssc_context_slots.py`` but swaps the
frozen xSSC branch from official DINOv2 YTVIS slots to the latest DINOv3
MOVi-C slot-512 model:

    context video -> xSSC preprocessing -> frame-0 AMG pseudo boxes
                  -> frozen DINOv3 RandSFQ2 -> slotz [B, 8, 11, 512]
                  -> LayerNorm + Linear(512, Wan dim) + time embedding
                  -> [B, 8 * 11, Wan dim] -> object cross-attention

The AMG box flow matches the current visualization preprocessing: SAM2 AMG
masks are filtered by ``select_xssc_candidates()``, converted to normalized
xyxy boxes, then repeated across the context time axis for xSSC's bbox
condition. The Wan base, physical LoRA, xSSC, DINOv3 backbone, and SAM2 box
generator are frozen. Trainable parameters remain object-attention LoRA, object
gates, slot projection, slot LayerNorm, and time embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import train_xssc_context_slots as base

tvn = base.tvn

TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
DINOV3_XSSC_ROOT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
DEFAULT_XSSC_ROOT = str(DINOV3_XSSC_ROOT)
DEFAULT_XSSC_CONFIG = str(
    DINOV3_XSSC_ROOT
    / "upstream/config-randsfq/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
)
DEFAULT_XSSC_CHECKPOINT_DIR = (
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42"
)
DEFAULT_DINOV3_ROOT = str(DINOV3_XSSC_ROOT / "third_party/dinov3")
DEFAULT_DINOV3_CHECKPOINT = (
    "/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors"
)
DEFAULT_XSSC_BOX_CACHE_DIR = (
    "/data/gaoya/agent-data/cache/xssc_dinov3_context_amg_boxes"
)
DEFAULT_SAM2_CONFIG = "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT = (
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt"
)
DINOV3_XSSC_SLOT_DIM = 512
DINOV3_XSSC_NUM_SLOTS = 11


def _set_parser_default(parser: argparse.ArgumentParser, dest: str, value) -> None:
    for action in parser._actions:
        if action.dest == dest:
            action.default = value
            return
    raise KeyError(f"parser option {dest!r} not found")


def resolve_latest_xssc_checkpoint(value: str | os.PathLike, latest_dir: str | os.PathLike) -> str:
    text = str(value)
    if text not in {"latest", "auto", ""}:
        return str(Path(text).expanduser().resolve())
    root = Path(latest_dir).expanduser().resolve()
    checkpoints = sorted(
        root.glob("step-*.pth"),
        key=lambda path: int(path.stem.split("-")[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No step-*.pth checkpoints found under {root}")
    return str(checkpoints[-1])


def _prepend_once(path: Path) -> None:
    import sys

    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _load_dinov3_xssc_model(
    *,
    xssc_root: str,
    config_path: str,
    checkpoint_path: str,
    dinov3_root: str,
    dinov3_checkpoint: str,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Build DINOv3 RandSFQ2 and load all non-backbone xSSC weights."""
    root = Path(xssc_root).expanduser().resolve()
    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    dinov3_source = Path(dinov3_root).expanduser().resolve()
    dinov3_weight = Path(dinov3_checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"DINOv3 xSSC root does not exist: {root}")
    if not config.is_file():
        raise FileNotFoundError(f"DINOv3 xSSC config does not exist: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"DINOv3 xSSC checkpoint does not exist: {checkpoint}")
    if not (dinov3_source / "dinov3").is_dir():
        raise FileNotFoundError(f"DINOv3 source tree not found: {dinov3_source}")
    if not dinov3_weight.is_file():
        raise FileNotFoundError(f"DINOv3 pretrained checkpoint not found: {dinov3_weight}")

    _prepend_once(root / "upstream")
    _prepend_once(dinov3_source)

    old_dinov3_root = os.environ.get("DINOV3_ROOT")
    old_dinov3_checkpoint = os.environ.get("DINOV3_CHECKPOINT")
    os.environ["DINOV3_ROOT"] = str(dinov3_source)
    os.environ["DINOV3_CHECKPOINT"] = str(dinov3_weight)
    try:
        from object_centric_bench.util import Config, build_from_config

        cfg = Config.fromfile(config)
        model = build_from_config(cfg.model)
    finally:
        if old_dinov3_root is None:
            os.environ.pop("DINOV3_ROOT", None)
        else:
            os.environ["DINOV3_ROOT"] = old_dinov3_root
        if old_dinov3_checkpoint is None:
            os.environ.pop("DINOV3_CHECKPOINT", None)
        else:
            os.environ["DINOV3_CHECKPOINT"] = old_dinov3_checkpoint

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported xSSC checkpoint object: {type(state)!r}")
    if state and all(str(key).startswith("m.") for key in state):
        state = {str(key)[2:]: value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    bad_missing = [
        key for key in incompatible.missing_keys if not key.startswith("encode_backbone.")
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "DINOv3 xSSC checkpoint mismatch: "
            f"missing={bad_missing}, unexpected={incompatible.unexpected_keys}"
        )

    slot_dim = int(cfg.emb_dim)
    num_slots = int(cfg.max_num)
    if slot_dim != DINOV3_XSSC_SLOT_DIM or num_slots != DINOV3_XSSC_NUM_SLOTS:
        raise RuntimeError(
            "Unexpected DINOv3 xSSC geometry: "
            f"slot_dim={slot_dim}, num_slots={num_slots}"
        )
    model.decode = None
    model.requires_grad_(False)
    model.eval()
    model.to(device=device)
    return model, slot_dim, num_slots


def _normalized_xssc_first_frames_to_uint8(video: torch.Tensor) -> list[np.ndarray]:
    """Convert xSSC-normalized [B,T,3,256,256] first frames to RGB uint8."""
    mean = video.new_tensor(base.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = video.new_tensor(base.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    frames = video[:, 0].float() * std + mean
    frames = frames.clamp(0.0, 255.0).round().to(torch.uint8)
    return [
        frame.permute(1, 2, 0).detach().cpu().numpy()
        for frame in frames
    ]


def masks_to_repeated_boxes(masks: np.ndarray, num_slots: int, num_frames: int) -> np.ndarray:
    boxes = np.zeros((num_frames, num_slots, 4), dtype=np.float32)
    if masks.size == 0:
        return boxes
    height, width = masks.shape[-2:]
    for slot_id, mask in enumerate(masks[:num_slots]):
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        boxes[:, slot_id] = np.asarray(
            [
                xs.min() / width,
                ys.min() / height,
                (xs.max() + 1) / width,
                (ys.max() + 1) / height,
            ],
            dtype=np.float32,
        )
    return boxes


class AMGBoxBuilder:
    """Lazy SAM2 AMG -> xSSC pseudo-box builder with a disk cache."""

    def __init__(
        self,
        *,
        sam2_config: str,
        sam2_checkpoint: str,
        cache_dir: str | None,
        filter_args: argparse.Namespace,
    ) -> None:
        self.sam2_config = Path(sam2_config).expanduser().resolve()
        self.sam2_checkpoint = Path(sam2_checkpoint).expanduser().resolve()
        self.cache_dir = None if cache_dir is None else Path(cache_dir).expanduser().resolve()
        self.filter_args = filter_args
        self._generator = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, image: np.ndarray, num_slots: int, num_frames: int) -> Path | None:
        if self.cache_dir is None:
            return None
        payload = {
            "num_slots": int(num_slots),
            "num_frames": int(num_frames),
            "sam2_config": str(self.sam2_config),
            "sam2_checkpoint": str(self.sam2_checkpoint),
            "filters": vars(self.filter_args),
            "image_sha1": hashlib.sha1(image.tobytes()).hexdigest(),
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    def _get_generator(self, device: torch.device):
        if self._generator is not None:
            return self._generator
        if not self.sam2_config.is_file():
            raise FileNotFoundError(f"SAM2 config not found: {self.sam2_config}")
        if not self.sam2_checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.sam2_checkpoint}")
        import sys

        grounded_sam2_root = "/home/gaoya/Grounded-SAM-2-main"
        if grounded_sam2_root not in sys.path:
            sys.path.insert(0, grounded_sam2_root)
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
        from visualize_movi_c_sam2_amg import resolve_sam2_config_name

        sam2 = build_sam2(
            resolve_sam2_config_name(self.sam2_config),
            str(self.sam2_checkpoint),
            device=str(device),
            mode="eval",
        )
        self._generator = SAM2AutomaticMaskGenerator(sam2)
        return self._generator

    def __call__(self, video: torch.Tensor, num_slots: int) -> torch.Tensor:
        from visualize_movi_c_sam2_amg import select_xssc_candidates

        device = video.device
        num_frames = int(video.shape[1])
        boxes = []
        generator = None
        for image in _normalized_xssc_first_frames_to_uint8(video):
            cache_path = self._cache_key(image, num_slots, num_frames)
            if cache_path is not None and cache_path.is_file():
                boxes.append(np.load(cache_path).astype(np.float32))
                continue
            if generator is None:
                generator = self._get_generator(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                annotations = generator.generate(image)
            selected = select_xssc_candidates(
                annotations,
                image.shape[0] * image.shape[1],
                self.filter_args,
                image=image,
            )
            if selected:
                masks = np.stack(
                    [item["segmentation"].astype(bool) for item in selected],
                    axis=0,
                )
            else:
                masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
            item_boxes = masks_to_repeated_boxes(masks, num_slots, num_frames)
            if cache_path is not None:
                temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
                try:
                    with temp_path.open("wb") as handle:
                        np.save(handle, item_boxes)
                    temp_path.replace(cache_path)
                except OSError:
                    pass
            boxes.append(item_boxes)
        return torch.from_numpy(np.stack(boxes, axis=0)).to(device=device, dtype=torch.float32)


class DINOv3XSSCContextSlotsWanModule(base.XSSCContextSlotsWanModule):
    """Wan training module conditioned on frozen DINOv3 MOVi-C xSSC slots."""

    def __init__(
        self,
        *args,
        xssc_root: str,
        xssc_config: str,
        xssc_checkpoint: str,
        dinov3_root: str,
        dinov3_checkpoint: str,
        xssc_box_source: str = "amg",
        xssc_box_cache_dir: str | None = DEFAULT_XSSC_BOX_CACHE_DIR,
        xssc_sam2_config: str = DEFAULT_SAM2_CONFIG,
        xssc_sam2_checkpoint: str = DEFAULT_SAM2_CHECKPOINT,
        xssc_amg_filter_args: argparse.Namespace | None = None,
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
        kwargs["enable_object_branch"] = False
        kwargs["freeze_non_object_trainables"] = True
        kwargs["train_object_pooler"] = False
        kwargs["train_object_aux_heads"] = False
        kwargs["train_object_adapter"] = False
        kwargs["train_object_dit_branch"] = False
        kwargs["no_context_ratio"] = 0.0
        tvn.WanTrainingModule.__init__(
            self,
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
        self.xssc_box_source = str(xssc_box_source)
        self._last_slot_dropout_fraction = 0.0
        self._last_retained_slots_per_sample = 0.0

        if self.fixed_num_context_frames != base.XSSC_NUM_CONTEXT_FRAMES:
            raise ValueError(
                "DINOv3 xSSC training requires exactly "
                f"{base.XSSC_NUM_CONTEXT_FRAMES} context frames, got "
                f"{self.fixed_num_context_frames}"
            )
        if self.xssc_input_size != 256:
            raise ValueError(f"DINOv3 xSSC requires xssc_input_size=256, got {self.xssc_input_size}")
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
        if self.xssc_box_source not in {"amg", "zeros"}:
            raise ValueError(f"unsupported xssc_box_source={self.xssc_box_source!r}")

        dit = base.enable_object_condition_branch(
            self.pipe.dit,
            object_gate_init=float(object_gate_init),
            reinitialize_object_branch=True,
        )
        dit.object_embedding = None
        for block in dit.blocks:
            base._initialize_object_attention_from_text(block)
            base._inject_object_attention_lora(
                block,
                rank=self.object_lora_rank,
                alpha=self.object_lora_alpha,
                dropout=self.object_lora_dropout,
            )
        for name, param in dit.named_parameters():
            param.requires_grad = base._is_trainable_object_dit_parameter(name)

        model_device = dit.patch_embedding.weight.device
        model_dtype = dit.patch_embedding.weight.dtype
        self.xssc, self.xssc_slot_dim, self.xssc_num_slots = _load_dinov3_xssc_model(
            xssc_root=xssc_root,
            config_path=xssc_config,
            checkpoint_path=xssc_checkpoint,
            dinov3_root=dinov3_root,
            dinov3_checkpoint=dinov3_checkpoint,
            device=model_device,
        )
        self._last_retained_slots_per_sample = float(self.xssc_num_slots)

        self.xssc_box_builder = None
        if self.xssc_box_source == "amg":
            if xssc_amg_filter_args is None:
                raise ValueError("xssc_amg_filter_args is required when xssc_box_source='amg'")
            self.xssc_box_builder = AMGBoxBuilder(
                sam2_config=xssc_sam2_config,
                sam2_checkpoint=xssc_sam2_checkpoint,
                cache_dir=xssc_box_cache_dir,
                filter_args=xssc_amg_filter_args,
            )

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

    def _build_xssc_boxes(self, xssc_video: torch.Tensor) -> torch.Tensor:
        if self.xssc_box_source == "zeros":
            batch, time_steps = int(xssc_video.shape[0]), int(xssc_video.shape[1])
            return xssc_video.new_zeros(batch, time_steps, self.xssc_num_slots, 4)
        if self.xssc_box_builder is None:
            raise RuntimeError("xSSC AMG box builder is not initialized")
        return self.xssc_box_builder(xssc_video, self.xssc_num_slots).to(
            device=xssc_video.device,
            dtype=torch.float32,
        )

    @torch.no_grad()
    def _extract_xssc_slots(self, video: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        self.xssc.eval()
        batch, time_steps, _, _, _ = video.shape
        if tuple(boxes.shape[:3]) != (batch, time_steps, self.xssc_num_slots):
            raise ValueError(
                "xSSC boxes must be [B,T,S,4], got "
                f"video={tuple(video.shape)}, boxes={tuple(boxes.shape)}"
            )
        flat_video = video.flatten(0, 1)
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
            boxes = boxes.to(device=encoded.device, dtype=encoded.dtype)

            slots = None
            for frame_id in range(time_steps):
                if frame_id == 0:
                    query = self.xssc.initializ(boxes[:, 0, :, :])
                else:
                    query = self.xssc.transit(slots, encoded[:, : frame_id + 1])
                num_iter = None if frame_id == 0 else 1
                current_slots, _ = self.xssc.aggregat(
                    encoded[:, frame_id],
                    query,
                    num_iter=num_iter,
                )
                current_slots = current_slots[:, None]
                slots = current_slots if slots is None else torch.cat((slots, current_slots), dim=1)
        if slots is None:
            raise RuntimeError("xSSC received zero context frames")
        return slots

    def _build_object_context(self, context_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xssc_video = self._preprocess_xssc(context_video)
        boxes = self._build_xssc_boxes(xssc_video)
        slots = self._extract_xssc_slots(xssc_video, boxes)
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


def _amg_filter_args_from_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        max_selected=args.xssc_amg_max_selected,
        min_area_ratio=args.xssc_amg_min_area_ratio,
        max_area_ratio=args.xssc_amg_max_area_ratio,
        min_bbox_side=args.xssc_amg_min_bbox_side,
        background_area_ratio=args.xssc_amg_background_area_ratio,
        background_span_ratio=args.xssc_amg_background_span_ratio,
        border_area_ratio=args.xssc_amg_border_area_ratio,
        border_occupancy_ratio=args.xssc_amg_border_occupancy_ratio,
        opposite_edge_area_ratio=args.xssc_amg_opposite_edge_area_ratio,
        shadow_min_area_ratio=args.xssc_amg_shadow_min_area_ratio,
        shadow_max_luminance_ratio=args.xssc_amg_shadow_max_luminance_ratio,
        shadow_max_chromaticity_distance=args.xssc_amg_shadow_max_chromaticity_distance,
        shadow_max_gradient_mean=args.xssc_amg_shadow_max_gradient_mean,
        duplicate_iou=args.xssc_amg_duplicate_iou,
        duplicate_containment=args.xssc_amg_duplicate_containment,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = "Train Wan2.2 with frozen DINOv3 MOVi-C xSSC context slots."
    _set_parser_default(parser, "xssc_root", DEFAULT_XSSC_ROOT)
    _set_parser_default(parser, "xssc_config", DEFAULT_XSSC_CONFIG)
    _set_parser_default(parser, "xssc_checkpoint", "latest")

    group = parser.add_argument_group("dinov3_xssc_context_slots")
    group.add_argument("--xssc_checkpoint_latest_dir", default=DEFAULT_XSSC_CHECKPOINT_DIR)
    group.add_argument("--dinov3_root", default=DEFAULT_DINOV3_ROOT)
    group.add_argument("--dinov3_checkpoint", default=DEFAULT_DINOV3_CHECKPOINT)
    group.add_argument("--xssc_box_source", choices=("amg", "zeros"), default="amg")
    group.add_argument("--xssc_box_cache_dir", default=DEFAULT_XSSC_BOX_CACHE_DIR)
    group.add_argument("--xssc_sam2_config", default=DEFAULT_SAM2_CONFIG)
    group.add_argument("--xssc_sam2_checkpoint", default=DEFAULT_SAM2_CHECKPOINT)
    group.add_argument("--xssc_amg_max_selected", type=int, default=11)
    group.add_argument("--xssc_amg_min_area_ratio", type=float, default=0.004)
    group.add_argument("--xssc_amg_max_area_ratio", type=float, default=0.35)
    group.add_argument("--xssc_amg_min_bbox_side", type=float, default=7.0)
    group.add_argument("--xssc_amg_background_area_ratio", type=float, default=0.06)
    group.add_argument("--xssc_amg_background_span_ratio", type=float, default=0.75)
    group.add_argument("--xssc_amg_border_area_ratio", type=float, default=0.025)
    group.add_argument("--xssc_amg_border_occupancy_ratio", type=float, default=0.18)
    group.add_argument("--xssc_amg_opposite_edge_area_ratio", type=float, default=0.04)
    group.add_argument("--xssc_amg_shadow_min_area_ratio", type=float, default=0.03)
    group.add_argument("--xssc_amg_shadow_max_luminance_ratio", type=float, default=0.55)
    group.add_argument("--xssc_amg_shadow_max_chromaticity_distance", type=float, default=0.10)
    group.add_argument("--xssc_amg_shadow_max_gradient_mean", type=float, default=20.0)
    group.add_argument("--xssc_amg_duplicate_iou", type=float, default=0.70)
    group.add_argument("--xssc_amg_duplicate_containment", type=float, default=0.85)
    return parser


def build_model(args: argparse.Namespace, accelerator) -> DINOv3XSSCContextSlotsWanModule:
    xssc_checkpoint = resolve_latest_xssc_checkpoint(
        args.xssc_checkpoint,
        args.xssc_checkpoint_latest_dir,
    )
    args.xssc_checkpoint = xssc_checkpoint
    return DINOv3XSSCContextSlotsWanModule(
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
        dinov3_root=args.dinov3_root,
        dinov3_checkpoint=args.dinov3_checkpoint,
        xssc_box_source=args.xssc_box_source,
        xssc_box_cache_dir=args.xssc_box_cache_dir,
        xssc_sam2_config=args.xssc_sam2_config,
        xssc_sam2_checkpoint=args.xssc_sam2_checkpoint,
        xssc_amg_filter_args=_amg_filter_args_from_args(args),
        xssc_input_size=args.xssc_input_size,
        xssc_max_time_steps=args.xssc_max_time_steps,
        object_lora_rank=args.object_lora_rank,
        object_lora_alpha=args.object_lora_alpha,
        object_lora_dropout=args.object_lora_dropout,
        xssc_slot_track_dropout=args.xssc_slot_track_dropout,
    )


def _log_stage_summary(accelerator, model: DINOv3XSSCContextSlotsWanModule, args: argparse.Namespace) -> None:
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
        "DINOv3 xSSC MOVi-C context-slot object conditioning",
        "=" * 78,
        f"Frozen Wan base: {args.wan_root}",
        f"Frozen physical-state LoRA: {args.lora_checkpoint}",
        f"Frozen DINOv3 xSSC checkpoint: {args.xssc_checkpoint}",
        f"Frozen DINOv3 pretrained weight: {args.dinov3_checkpoint}",
        f"xSSC box source: {model.xssc_box_source}",
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
    if int(args.fixed_num_context_frames) != base.XSSC_NUM_CONTEXT_FRAMES:
        parser.error(
            f"--fixed_num_context_frames must be {base.XSSC_NUM_CONTEXT_FRAMES} for xSSC training"
        )
    if int(args.train_batch_size) <= 0:
        parser.error("--train_batch_size must be positive")
    args.no_context_ratio = 0.0
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    tvn.init_trackers(accelerator, args)

    dataset = base.build_dataset(args)
    if int(args.train_batch_size) > 1:
        dataset = base.GroupedBatchDataset(dataset, args.train_batch_size)
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
                "Incomplete or incompatible DINOv3 xSSC object-LoRA resume checkpoint: "
                f"loaded={resume_info['loaded_count']}/{expected_resume_count}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded DINOv3 xSSC-object resume weights: "
                f"loaded_count={resume_info['loaded_count']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    _log_stage_summary(accelerator, model, args)
    model_logger = base.ModelLogger(
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
