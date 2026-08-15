"""Configuration-driven Object-only, Full-SA, S-head, and T-head xSSC training.

All variants start from the same effective Wan initialization:

1. inject and load the OpenVid/MOVi-D/Genesis LoRA checkpoint;
2. merge that complete LoRA into the frozen Wan weights and unload its PEFT
   modules;
3. optionally construct the xSSC object branch;
4. add the configured zero-initialized self-attention delta adapter.

``object_only`` trains only the object branch and xSSC projection. ``full_sa``
adds ordinary rank-r LoRA to q/k/v/o in every self-attention layer.
``s_head`` and ``t_head`` add compact adapters whose support is restricted to
their configured heads. ``full_sa`` can also disable the complete object path,
leaving only the 30-layer self-attention LoRA. The common merged initialization
makes the step-0 forward identical across variants before their new adapters act.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

EXPERIMENT_ROOT = Path(__file__).resolve().parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
if str(TRAIN_XSSC_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_XSSC_ROOT))
import train_xssc_context_slots as base

tvn = base.tvn

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
SELF_ATTN_ADAPTATION_MODES = ("object_only", "full_sa", "s_head", "t_head")
SELF_ATTN_PROJECTIONS = ("q", "k", "v", "o")
HEAD_SELECTIVE_ADAPTATION_MODES = ("s_head", "t_head")
HEAD_SELECTION_IDENTITY_KEY = "head_selection_identity"
HEAD_SELECTION_CONFIG_SHA256_KEY = "head_selection_config_sha256"


def _set_child_module(root: nn.Module, qualified_name: str, child: nn.Module) -> None:
    parent_name, _, leaf_name = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf_name, child)


def merge_and_unload_pretrained_lora(
    model: nn.Module,
    *,
    expected_module_count: int,
) -> list[str]:
    """Merge injected PEFT LoRA modules into their base layers and remove them."""
    names = [
        name
        for name, module in model.named_modules()
        if name
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "get_base_layer")
        and callable(getattr(module, "merge", None))
    ]
    if len(names) != int(expected_module_count):
        raise RuntimeError(
            "Unexpected pretrained LoRA module count before merge: "
            f"found={len(names)}, expected={expected_module_count}"
        )

    # Replace deepest modules first so parent traversal remains valid.
    for name in sorted(names, key=lambda value: value.count("."), reverse=True):
        module = model.get_submodule(name)
        module.merge(safe_merge=True)
        base_layer = module.get_base_layer()
        base_layer.requires_grad_(False)
        _set_child_module(model, name, base_layer)

    remaining = [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and hasattr(module, "lora_B")
    ]
    if remaining:
        raise RuntimeError(f"Pretrained LoRA unload left wrapped modules: {remaining[:8]}")
    return sorted(names)


def load_head_selection_config(
    path: str | os.PathLike,
    *,
    expected_subset_id: str,
    expected_role: str,
    expected_feature_subtype: str,
    expected_num_heads: int,
    num_blocks: int,
    num_heads: int,
) -> tuple[dict[int, tuple[int, ...]], dict]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Head-selection config does not exist: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported head-selection schema in {config_path}")
    if payload.get("subset_id") != expected_subset_id:
        raise ValueError(
            "Head-selection subset mismatch: "
            f"config={payload.get('subset_id')!r}, expected={expected_subset_id!r}"
        )
    if payload.get("role") != expected_role:
        raise ValueError(
            "Head-selection role mismatch: "
            f"config={payload.get('role')!r}, expected={expected_role!r}"
        )
    if payload.get("feature_subtype") != expected_feature_subtype:
        raise ValueError(
            "Head-selection feature subtype mismatch: "
            f"config={payload.get('feature_subtype')!r}, "
            f"expected={expected_feature_subtype!r}"
        )

    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise TypeError("Head-selection config targets must be a list")
    pairs: list[tuple[int, int]] = []
    for item in targets:
        if not isinstance(item, dict):
            raise TypeError(f"Invalid head-selection target: {item!r}")
        block_id, head_id = int(item["block"]), int(item["head"])
        if not 0 <= block_id < int(num_blocks):
            raise ValueError(f"Head-selection block out of range: {(block_id, head_id)}")
        if not 0 <= head_id < int(num_heads):
            raise ValueError(f"Head-selection index out of range: {(block_id, head_id)}")
        pairs.append((block_id, head_id))
    if len(pairs) != int(expected_num_heads):
        raise ValueError(
            f"Expected {expected_num_heads} selected heads, "
            f"found {len(pairs)} in {config_path}"
        )
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"Duplicate block/head entries in {config_path}")
    if int(payload.get("num_heads", -1)) != len(pairs):
        raise ValueError(
            f"Declared num_heads does not match targets in {config_path}"
        )

    by_block: dict[int, list[int]] = {}
    for block_id, head_id in sorted(pairs):
        by_block.setdefault(block_id, []).append(head_id)
    if int(payload.get("num_blocks", -1)) != len(by_block):
        raise ValueError(
            f"Declared num_blocks does not match targets in {config_path}"
        )
    actual_histogram = {
        str(block_id): len(head_ids)
        for block_id, head_ids in sorted(by_block.items())
    }
    if payload.get("block_histogram") != actual_histogram:
        raise ValueError(
            f"Declared block_histogram does not match targets in {config_path}"
        )
    normalized = {
        block_id: tuple(sorted(head_ids))
        for block_id, head_ids in by_block.items()
    }
    metadata = dict(payload)
    metadata["config_path"] = str(config_path)
    metadata["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return normalized, metadata


def build_head_selection_identity(
    heads_by_block: dict[int, tuple[int, ...]],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    pairs = [
        (int(block_id), int(head_id))
        for block_id, head_ids in sorted(heads_by_block.items())
        for head_id in sorted(head_ids)
    ]
    if not pairs:
        raise ValueError("Head-selection identity cannot be empty")
    # NCCL does not support broadcasting torch.int16 buffers during DDP setup.
    return torch.tensor(pairs, dtype=torch.int32, device=device)


def build_sha256_identity(
    sha256_hex: str,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    try:
        digest = bytes.fromhex(str(sha256_hex))
    except ValueError as exc:
        raise ValueError(f"Invalid SHA256 digest: {sha256_hex!r}") from exc
    if len(digest) != 32:
        raise ValueError(f"SHA256 digest must contain 32 bytes, got {len(digest)}")
    return torch.tensor(list(digest), dtype=torch.uint8, device=device)


def _format_head_identity(identity: torch.Tensor) -> list[str]:
    values = identity.detach().to(device="cpu", dtype=torch.int64)
    if values.ndim != 2 or int(values.shape[1]) != 2:
        return [f"<invalid shape {tuple(values.shape)}>"]
    return [
        f"B{int(block_id):02d}H{int(head_id):02d}"
        for block_id, head_id in values.tolist()
    ]


def validate_head_selection_checkpoint_state(
    state_dict: dict[str, torch.Tensor],
    *,
    expected_identity: torch.Tensor,
    expected_config_sha256: torch.Tensor,
) -> dict[str, object]:
    normalized = {
        tvn._normalize_checkpoint_key(key): value
        for key, value in state_dict.items()
    }
    required = (
        HEAD_SELECTION_IDENTITY_KEY,
        HEAD_SELECTION_CONFIG_SHA256_KEY,
    )
    missing = [key for key in required if key not in normalized]
    if missing:
        raise RuntimeError(
            "Head-selective resume checkpoint is missing bound head identity "
            f"metadata: {missing}. Legacy S/T checkpoints cannot be resumed "
            "safely because equal tensor shapes do not prove equal BxxHxx targets."
        )

    found_identity = normalized[HEAD_SELECTION_IDENTITY_KEY].detach().to(
        device="cpu", dtype=torch.int32
    )
    expected_identity = expected_identity.detach().to(
        device="cpu", dtype=torch.int32
    )
    if not torch.equal(found_identity, expected_identity):
        raise RuntimeError(
            "Head-selective resume checkpoint targets different heads. "
            f"checkpoint={_format_head_identity(found_identity)}, "
            f"current={_format_head_identity(expected_identity)}"
        )

    found_sha256 = normalized[HEAD_SELECTION_CONFIG_SHA256_KEY].detach().to(
        device="cpu", dtype=torch.uint8
    )
    expected_config_sha256 = expected_config_sha256.detach().to(
        device="cpu", dtype=torch.uint8
    )
    if not torch.equal(found_sha256, expected_config_sha256):
        found_hex = bytes(found_sha256.tolist()).hex()
        expected_hex = bytes(expected_config_sha256.tolist()).hex()
        raise RuntimeError(
            "Head-selective resume checkpoint was produced from a different "
            "head-selection config: "
            f"checkpoint_sha256={found_hex}, current_sha256={expected_hex}"
        )
    return {
        "num_heads": int(expected_identity.shape[0]),
        "config_sha256": bytes(expected_config_sha256.tolist()).hex(),
    }


def validate_head_selection_resume_checkpoint(
    model: "DINOv3XSSCContextSlotsWanModule",
    checkpoint_path: str | os.PathLike,
) -> dict[str, object]:
    if model.self_attn_adaptation_mode not in HEAD_SELECTIVE_ADAPTATION_MODES:
        raise ValueError(
            "Head-selection checkpoint validation is only valid for "
            f"{HEAD_SELECTIVE_ADAPTATION_MODES}"
        )
    state_dict = tvn._load_trainable_state(checkpoint_path)
    return validate_head_selection_checkpoint_state(
        state_dict,
        expected_identity=getattr(model, HEAD_SELECTION_IDENTITY_KEY),
        expected_config_sha256=getattr(
            model, HEAD_SELECTION_CONFIG_SHA256_KEY
        ),
    )


def checkpoint_saver_only_on_sync(save_fn):
    """Suppress checkpoint writes during non-sync gradient micro-steps."""

    def wrapped(*args, **kwargs):
        accelerator = kwargs.get("accelerator")
        if accelerator is None:
            raise TypeError("Checkpoint saver requires accelerator as a keyword")
        if not bool(accelerator.sync_gradients):
            return None
        return save_fn(*args, **kwargs)

    return wrapped


def run_train_loop_with_synced_checkpoint_saves(*args, **kwargs):
    """Run the shared loop while saving only at optimizer-update boundaries."""
    original_save = tvn.save_training_checkpoint_bundle
    tvn.save_training_checkpoint_bundle = checkpoint_saver_only_on_sync(
        original_save
    )
    try:
        return tvn.train_loop(*args, **kwargs)
    finally:
        tvn.save_training_checkpoint_bundle = original_save


class HeadSelectiveLoRALinear(nn.Module):
    """Compact LoRA residual supported only on selected attention heads."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        selected_heads: tuple[int, ...],
        num_heads: int,
        rank: int,
        alpha: float,
        dropout: float,
        projection: str,
    ) -> None:
        super().__init__()
        if projection not in SELF_ATTN_PROJECTIONS:
            raise ValueError(f"Unsupported self-attention projection: {projection}")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                f"Head-selective LoRA requires nn.Linear, got {type(base_layer)!r}"
            )
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if base_layer.in_features % num_heads or base_layer.out_features % num_heads:
            raise ValueError(
                "Projection dimensions must be divisible by num_heads: "
                f"in={base_layer.in_features}, out={base_layer.out_features}, "
                f"heads={num_heads}"
            )

        heads = tuple(sorted(set(int(value) for value in selected_heads)))
        if not heads or heads[0] < 0 or heads[-1] >= num_heads:
            raise ValueError(f"Invalid selected heads: {heads}")
        head_dim = base_layer.out_features // num_heads
        if projection == "o":
            head_dim = base_layer.in_features // num_heads
        channels = [
            head_id * head_dim + offset
            for head_id in heads
            for offset in range(head_dim)
        ]

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.projection = projection
        self.selected_heads = heads
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.register_buffer(
            "selected_channels",
            torch.tensor(channels, dtype=torch.long, device=base_layer.weight.device),
            persistent=False,
        )

        factory_kwargs = {
            "device": base_layer.weight.device,
            "dtype": base_layer.weight.dtype,
        }
        selected_dim = len(channels)
        if projection == "o":
            self.head_lora_A = nn.Linear(
                selected_dim, rank, bias=False, **factory_kwargs
            )
            self.head_lora_B = nn.Linear(
                rank, base_layer.out_features, bias=False, **factory_kwargs
            )
        else:
            self.head_lora_A = nn.Linear(
                base_layer.in_features, rank, bias=False, **factory_kwargs
            )
            self.head_lora_B = nn.Linear(
                rank, selected_dim, bias=False, **factory_kwargs
            )
        nn.init.kaiming_uniform_(self.head_lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.head_lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.projection == "o":
            selected = x.index_select(-1, self.selected_channels)
            delta = self.head_lora_B(
                self.head_lora_A(self.dropout(selected))
            )
            return result + delta * self.scaling

        delta = self.head_lora_B(self.head_lora_A(self.dropout(x)))
        expanded = torch.zeros_like(result).index_copy(
            -1, self.selected_channels, delta * self.scaling
        )
        return result + expanded


def inject_full_self_attn_lora(
    dit: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    config = LoraConfig(
        r=int(rank),
        lora_alpha=float(alpha),
        lora_dropout=float(dropout),
        target_modules=list(SELF_ATTN_PROJECTIONS),
        bias="none",
    )
    for block in dit.blocks:
        block.self_attn = inject_adapter_in_model(config, block.self_attn)


def inject_head_selective_lora(
    dit: nn.Module,
    *,
    heads_by_block: dict[int, tuple[int, ...]],
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    for block_id, selected_heads in heads_by_block.items():
        self_attn = dit.blocks[block_id].self_attn
        for projection in SELF_ATTN_PROJECTIONS:
            layer = getattr(self_attn, projection)
            setattr(
                self_attn,
                projection,
                HeadSelectiveLoRALinear(
                    layer,
                    selected_heads=selected_heads,
                    num_heads=int(self_attn.num_heads),
                    rank=int(rank),
                    alpha=float(alpha),
                    dropout=float(dropout),
                    projection=projection,
                ),
            )


def _is_full_self_attn_lora_parameter(name: str) -> bool:
    return ".self_attn." in name and (
        ".lora_A." in name or ".lora_B." in name
    )


def _is_head_selective_lora_parameter(name: str) -> bool:
    return ".self_attn." in name and (
        ".head_lora_A." in name or ".head_lora_B." in name
    )


class EmptyAMGConditionError(RuntimeError):
    """Raised when AMG filtering leaves a training sample without any boxes."""

    def __init__(self, counts: list[int]) -> None:
        self.counts = [int(value) for value in counts]
        self.bad_indices = [
            index for index, value in enumerate(self.counts) if int(value) <= 0
        ]
        super().__init__(
            "empty AMG xSSC condition for batch indices "
            f"{self.bad_indices}; selected_counts={self.counts}"
        )


def _count_nonempty_boxes(item_boxes: np.ndarray) -> int:
    if item_boxes.size == 0:
        return 0
    frame0 = np.asarray(item_boxes[0], dtype=np.float32)
    wh = frame0[:, 2:4] - frame0[:, 0:2]
    valid = (wh[:, 0] > 0.0) & (wh[:, 1] > 0.0)
    return int(valid.sum())


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
        self.last_selected_counts: list[int] = []
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
        selected_counts = []
        generator = None
        for image in _normalized_xssc_first_frames_to_uint8(video):
            cache_path = self._cache_key(image, num_slots, num_frames)
            if cache_path is not None and cache_path.is_file():
                item_boxes = np.load(cache_path).astype(np.float32)
                selected_counts.append(_count_nonempty_boxes(item_boxes))
                boxes.append(item_boxes)
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
            selected_counts.append(_count_nonempty_boxes(item_boxes))
            if cache_path is not None:
                temp_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
                try:
                    with temp_path.open("wb") as handle:
                        np.save(handle, item_boxes)
                    temp_path.replace(cache_path)
                except OSError:
                    pass
            boxes.append(item_boxes)
        self.last_selected_counts = selected_counts
        return torch.from_numpy(np.stack(boxes, axis=0)).to(device=device, dtype=torch.float32)


class DINOv3XSSCContextSlotsWanModule(base.XSSCContextSlotsWanModule):
    """Wan training module conditioned on frozen DINOv3 MOVi-C xSSC slots."""

    def _initialize_frozen_dit(self, dit: nn.Module) -> list[str]:
        merged_modules = merge_and_unload_pretrained_lora(
            dit,
            expected_module_count=self.pretrained_lora_expected_modules,
        )
        self.base_dit_initialization = "Wan2.2 TI2V 5B + merged OpenVid LoRA"
        return merged_modules

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
        xssc_filter_empty_amg: bool = False,
        xssc_empty_amg_max_resample_attempts: int = 20,
        xssc_input_size: int = 256,
        xssc_max_time_steps: int = 64,
        object_lora_rank: int = 32,
        object_lora_alpha: float = 32.0,
        object_lora_dropout: float = 0.0,
        xssc_slot_track_dropout: float = 0.0,
        self_attn_adaptation_mode: str = "object_only",
        pretrained_lora_expected_modules: int = 300,
        self_attn_expected_num_blocks: int = 30,
        self_attn_expected_num_heads: int = 24,
        self_attn_lora_rank: int = 32,
        self_attn_lora_alpha: float = 32.0,
        self_attn_lora_dropout: float = 0.0,
        head_selection_config: str | None = None,
        head_selection_subset_id: str = "S_same_full59",
        head_selection_expected_role: str = "S",
        head_selection_feature_subtype: str = "same_frame_mass",
        head_selection_expected_num_heads: int = 59,
        enable_object_branch: bool = True,
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

        self.enable_object_branch = bool(enable_object_branch)
        self.lambda_main = float(lambda_main)
        self.lambda_object_context_reg = float(lambda_object_context_reg)
        self.xssc_input_size = int(xssc_input_size)
        self.xssc_max_time_steps = int(xssc_max_time_steps)
        self.object_lora_rank = int(object_lora_rank)
        self.object_lora_alpha = float(object_lora_alpha)
        self.object_lora_dropout = float(object_lora_dropout)
        self.xssc_slot_track_dropout = float(xssc_slot_track_dropout)
        self.self_attn_adaptation_mode = str(self_attn_adaptation_mode)
        self.pretrained_lora_expected_modules = int(pretrained_lora_expected_modules)
        self.self_attn_expected_num_blocks = int(self_attn_expected_num_blocks)
        self.self_attn_expected_num_heads = int(self_attn_expected_num_heads)
        self.self_attn_lora_rank = int(self_attn_lora_rank)
        self.self_attn_lora_alpha = float(self_attn_lora_alpha)
        self.self_attn_lora_dropout = float(self_attn_lora_dropout)
        self.head_selection_config = head_selection_config
        self.head_selection_subset_id = str(head_selection_subset_id)
        self.head_selection_expected_role = str(head_selection_expected_role)
        self.head_selection_feature_subtype = str(head_selection_feature_subtype)
        self.head_selection_expected_num_heads = int(head_selection_expected_num_heads)
        self.selected_heads_by_block: dict[int, tuple[int, ...]] = {}
        self.head_selection_metadata: dict = {}
        self.merged_pretrained_lora_modules: list[str] = []
        self.xssc_box_source = str(xssc_box_source)
        self.xssc_filter_empty_amg = bool(xssc_filter_empty_amg)
        self.xssc_empty_amg_max_resample_attempts = int(
            xssc_empty_amg_max_resample_attempts
        )
        self.xssc_empty_amg_resample_dataset = None
        self._xssc_empty_amg_resample_cdf = None
        self._last_xssc_amg_selected_counts: list[int] = []
        self._last_empty_amg_resample_count = 0
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
        if self.enable_object_branch and self.object_lora_rank <= 0:
            raise ValueError(f"object_lora_rank must be positive, got {self.object_lora_rank}")
        if self.self_attn_adaptation_mode not in SELF_ATTN_ADAPTATION_MODES:
            raise ValueError(
                "self_attn_adaptation_mode must be one of "
                f"{SELF_ATTN_ADAPTATION_MODES}, got {self.self_attn_adaptation_mode!r}"
            )
        if self.pretrained_lora_expected_modules <= 0:
            raise ValueError("pretrained_lora_expected_modules must be positive")
        if self.self_attn_expected_num_blocks <= 0:
            raise ValueError("self_attn_expected_num_blocks must be positive")
        if self.self_attn_expected_num_heads <= 0:
            raise ValueError("self_attn_expected_num_heads must be positive")
        if self.self_attn_lora_rank <= 0:
            raise ValueError(
                f"self_attn_lora_rank must be positive, got {self.self_attn_lora_rank}"
            )
        if not 0.0 <= self.self_attn_lora_dropout < 1.0:
            raise ValueError(
                "self_attn_lora_dropout must be in [0, 1), got "
                f"{self.self_attn_lora_dropout}"
            )
        if self.head_selection_expected_num_heads <= 0:
            raise ValueError("head_selection_expected_num_heads must be positive")
        if not self.head_selection_expected_role:
            raise ValueError("head_selection_expected_role must not be empty")
        if not self.head_selection_feature_subtype:
            raise ValueError("head_selection_feature_subtype must not be empty")
        if self.enable_object_branch and not 0.0 <= self.object_lora_dropout < 1.0:
            raise ValueError(
                f"object_lora_dropout must be in [0, 1), got {self.object_lora_dropout}"
            )
        if self.enable_object_branch and not 0.0 <= self.xssc_slot_track_dropout < 1.0:
            raise ValueError(
                "xssc_slot_track_dropout must be in [0, 1), got "
                f"{self.xssc_slot_track_dropout}"
            )
        if self.enable_object_branch and self.xssc_box_source not in {"amg", "zeros"}:
            raise ValueError(f"unsupported xssc_box_source={self.xssc_box_source!r}")
        if self.xssc_empty_amg_max_resample_attempts < 0:
            raise ValueError(
                "xssc_empty_amg_max_resample_attempts must be non-negative, got "
                f"{self.xssc_empty_amg_max_resample_attempts}"
            )

        dit = self.pipe.dit
        self.merged_pretrained_lora_modules = self._initialize_frozen_dit(dit)
        if len(dit.blocks) != self.self_attn_expected_num_blocks:
            raise RuntimeError(
                "Unexpected Wan block count: "
                f"found={len(dit.blocks)}, expected={self.self_attn_expected_num_blocks}"
            )
        found_num_heads = {int(block.self_attn.num_heads) for block in dit.blocks}
        if found_num_heads != {self.self_attn_expected_num_heads}:
            raise RuntimeError(
                "Unexpected Wan self-attention head counts: "
                f"found={sorted(found_num_heads)}, "
                f"expected={self.self_attn_expected_num_heads}"
            )

        if self.enable_object_branch:
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
        else:
            dit = self.pipe.dit
            unexpected_object_modules = [
                name
                for name, _ in dit.named_modules()
                if "object_cross_attn" in name or name.endswith(".norm4")
            ]
            if unexpected_object_modules:
                raise RuntimeError(
                    "Object-disabled mode unexpectedly contains object modules: "
                    f"{unexpected_object_modules[:8]}"
                )

        if self.self_attn_adaptation_mode == "full_sa":
            inject_full_self_attn_lora(
                dit,
                rank=self.self_attn_lora_rank,
                alpha=self.self_attn_lora_alpha,
                dropout=self.self_attn_lora_dropout,
            )
        elif self.self_attn_adaptation_mode in HEAD_SELECTIVE_ADAPTATION_MODES:
            if self.head_selection_config is None:
                raise ValueError(
                    "head_selection_config is required for head-selective adaptation"
                )
            (
                self.selected_heads_by_block,
                self.head_selection_metadata,
            ) = load_head_selection_config(
                self.head_selection_config,
                expected_subset_id=self.head_selection_subset_id,
                expected_role=self.head_selection_expected_role,
                expected_feature_subtype=self.head_selection_feature_subtype,
                expected_num_heads=self.head_selection_expected_num_heads,
                num_blocks=len(dit.blocks),
                num_heads=self.self_attn_expected_num_heads,
            )
            identity_device = dit.patch_embedding.weight.device
            self.register_buffer(
                HEAD_SELECTION_IDENTITY_KEY,
                build_head_selection_identity(
                    self.selected_heads_by_block,
                    device=identity_device,
                ),
                persistent=True,
            )
            self.register_buffer(
                HEAD_SELECTION_CONFIG_SHA256_KEY,
                build_sha256_identity(
                    self.head_selection_metadata["config_sha256"],
                    device=identity_device,
                ),
                persistent=True,
            )
            inject_head_selective_lora(
                dit,
                heads_by_block=self.selected_heads_by_block,
                rank=self.self_attn_lora_rank,
                alpha=self.self_attn_lora_alpha,
                dropout=self.self_attn_lora_dropout,
            )

        for name, param in dit.named_parameters():
            trainable = (
                self.enable_object_branch
                and base._is_trainable_object_dit_parameter(name)
            )
            if self.self_attn_adaptation_mode == "full_sa":
                trainable = trainable or _is_full_self_attn_lora_parameter(name)
            elif self.self_attn_adaptation_mode in HEAD_SELECTIVE_ADAPTATION_MODES:
                trainable = trainable or _is_head_selective_lora_parameter(name)
            param.requires_grad = trainable

        expected_full_lora_tensors = (
            self.self_attn_expected_num_blocks * len(SELF_ATTN_PROJECTIONS) * 2
        )
        full_lora_tensors = sum(
            1
            for name, param in dit.named_parameters()
            if param.requires_grad and _is_full_self_attn_lora_parameter(name)
        )
        head_lora_tensors = sum(
            1
            for name, param in dit.named_parameters()
            if param.requires_grad and _is_head_selective_lora_parameter(name)
        )
        if self.self_attn_adaptation_mode == "object_only":
            if full_lora_tensors or head_lora_tensors:
                raise RuntimeError("Object-only mode unexpectedly has trainable self-attention LoRA")
        elif self.self_attn_adaptation_mode == "full_sa":
            if full_lora_tensors != expected_full_lora_tensors or head_lora_tensors:
                raise RuntimeError(
                    "Full-SA trainable tensor mismatch: "
                    f"full={full_lora_tensors}/{expected_full_lora_tensors}, "
                    f"head={head_lora_tensors}"
                )
        else:
            expected_head_lora_tensors = (
                len(self.selected_heads_by_block) * len(SELF_ATTN_PROJECTIONS) * 2
            )
            if head_lora_tensors != expected_head_lora_tensors or full_lora_tensors:
                raise RuntimeError(
                    "Head-selective trainable tensor mismatch: "
                    f"head={head_lora_tensors}/{expected_head_lora_tensors}, "
                    f"full={full_lora_tensors}"
                )

        self.xssc = None
        self.xssc_slot_dim = 0
        self.xssc_num_slots = 0
        self.xssc_box_builder = None
        self.slot_norm = None
        self.slot_projector = None
        self.time_embedding = None
        if self.enable_object_branch:
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

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if self.xssc is not None:
            self.xssc.eval()
        return self

    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        exported = super().export_trainable_state_dict(
            state_dict,
            remove_prefix=remove_prefix,
        )
        if self.self_attn_adaptation_mode not in HEAD_SELECTIVE_ADAPTATION_MODES:
            return exported
        for source_key in (
            HEAD_SELECTION_IDENTITY_KEY,
            HEAD_SELECTION_CONFIG_SHA256_KEY,
        ):
            matches = [
                key
                for key in state_dict
                if tvn._normalize_checkpoint_key(key) == source_key
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "Model state must contain exactly one head identity buffer "
                    f"{source_key!r}, found {matches}"
                )
            state_key = matches[0]
            output_key = source_key
            if remove_prefix is not None and output_key.startswith(remove_prefix):
                output_key = output_key[len(remove_prefix) :]
            exported[output_key] = state_dict[state_key]
        return exported

    def set_empty_amg_resample_dataset(self, dataset) -> None:
        self.xssc_empty_amg_resample_dataset = dataset
        sample_weights = getattr(dataset, "sample_weights", None)
        if sample_weights is None:
            self._xssc_empty_amg_resample_cdf = None
            return
        weights = torch.as_tensor(sample_weights, dtype=torch.float64)
        if weights.ndim != 1 or int(weights.numel()) != len(dataset):
            raise ValueError("dataset.sample_weights must be a 1D vector matching dataset length")
        total = float(weights.sum().item())
        if total <= 0.0:
            raise ValueError("dataset.sample_weights must have positive sum")
        self._xssc_empty_amg_resample_cdf = torch.cumsum(weights, dim=0)

    def _sample_empty_amg_replacement(self) -> dict:
        dataset = self.xssc_empty_amg_resample_dataset
        if dataset is None:
            raise RuntimeError("empty AMG resampling requested but no dataset is attached")
        if self._xssc_empty_amg_resample_cdf is not None:
            cdf = self._xssc_empty_amg_resample_cdf
            draw = torch.rand((), dtype=torch.float64) * cdf[-1]
            index = int(torch.searchsorted(cdf, draw, right=False).item())
            index = min(index, len(dataset) - 1)
        else:
            index = int(torch.randint(len(dataset), (1,)).item())
        return dataset[index]

    def _build_xssc_boxes(self, xssc_video: torch.Tensor) -> torch.Tensor:
        if self.xssc_box_source == "zeros":
            batch, time_steps = int(xssc_video.shape[0]), int(xssc_video.shape[1])
            return xssc_video.new_zeros(batch, time_steps, self.xssc_num_slots, 4)
        if self.xssc_box_builder is None:
            raise RuntimeError("xSSC AMG box builder is not initialized")
        boxes = self.xssc_box_builder(xssc_video, self.xssc_num_slots).to(
            device=xssc_video.device,
            dtype=torch.float32,
        )
        self._last_xssc_amg_selected_counts = list(self.xssc_box_builder.last_selected_counts)
        if self.training and self.xssc_filter_empty_amg:
            if any(count <= 0 for count in self._last_xssc_amg_selected_counts):
                raise EmptyAMGConditionError(self._last_xssc_amg_selected_counts)
        return boxes

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
        if not self.enable_object_branch:
            raise RuntimeError("Object context requested while object branch is disabled")
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

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        if not self.enable_object_branch:
            loss_main = base.flow_match_context_sft_loss(
                pipe,
                **inputs_shared,
                **inputs_posi,
            )
            return loss_main, {
                "train/loss_total": float(loss_main.detach().item()),
                "train/loss_main": float(loss_main.detach().item()),
            }
        total, metrics = super()._compute_object_losses(pipe, inputs_shared, inputs_posi)
        counts = self._last_xssc_amg_selected_counts
        if counts:
            values = [float(value) for value in counts]
            metrics["train/xssc_amg_selected_masks_min"] = float(min(values))
            metrics["train/xssc_amg_selected_masks_mean"] = float(sum(values) / len(values))
            metrics["train/xssc_amg_selected_masks_max"] = float(max(values))
        metrics["train/xssc_filter_empty_amg"] = float(self.xssc_filter_empty_amg)
        metrics["train/xssc_empty_amg_resample_count"] = float(
            self._last_empty_amg_resample_count
        )
        return total, metrics

    def _forward_sample_batch_with_empty_amg_filter(self, samples: list[dict]) -> torch.Tensor:
        current = list(samples)
        resample_count = 0
        max_attempts = self.xssc_empty_amg_max_resample_attempts
        while True:
            self._last_empty_amg_resample_count = resample_count
            try:
                return self._forward_sample_batch(current)
            except EmptyAMGConditionError as exc:
                if not self.training or not self.xssc_filter_empty_amg:
                    raise
                if resample_count >= max_attempts:
                    raise RuntimeError(
                        "Exceeded empty-AMG resample budget: "
                        f"attempts={resample_count}, max_attempts={max_attempts}, "
                        f"last_selected_counts={exc.counts}"
                    ) from exc
                for bad_index in exc.bad_indices:
                    current[bad_index] = self._sample_empty_amg_replacement()
                    resample_count += 1
                    if resample_count >= max_attempts:
                        break

    def forward(self, data, inputs=None):
        if (
            self.enable_object_branch
            and self.training
            and self.xssc_filter_empty_amg
            and inputs is None
            and self.xssc_box_source == "amg"
        ):
            samples = data if isinstance(data, list) else [data]
            return self._forward_sample_batch_with_empty_amg_filter(samples)
        return super().forward(data, inputs=inputs)

    def trainable_modules(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.enable_object_branch:
            params.extend(self.slot_norm.parameters())
            params.extend(self.slot_projector.parameters())
            params.extend(self.time_embedding.parameters())
        params.extend(
            param for param in self.pipe.dit.parameters() if param.requires_grad
        )
        unique: list[nn.Parameter] = []
        seen: set[int] = set()
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            unique.append(param)
        return unique


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
    group.add_argument(
        "--xssc_filter_empty_amg",
        action="store_true",
        default=False,
        help="During training, discard and resample samples whose AMG filters select no masks.",
    )
    group.add_argument("--xssc_empty_amg_max_resample_attempts", type=int, default=20)
    group.add_argument(
        "--self_attn_adaptation_mode",
        choices=SELF_ATTN_ADAPTATION_MODES,
        default="object_only",
    )
    group.add_argument(
        "--disable_object_branch",
        action="store_true",
        help="Do not construct or load SAM2, xSSC, object projection, or object cross-attention.",
    )
    group.add_argument("--pretrained_lora_expected_modules", type=int, default=300)
    group.add_argument("--self_attn_expected_num_blocks", type=int, default=30)
    group.add_argument("--self_attn_expected_num_heads", type=int, default=24)
    group.add_argument("--self_attn_lora_rank", type=int, default=32)
    group.add_argument("--self_attn_lora_alpha", type=float, default=32.0)
    group.add_argument("--self_attn_lora_dropout", type=float, default=0.0)
    group.add_argument("--head_selection_config", default=None)
    group.add_argument(
        "--head_selection_subset_id",
        default="S_same_full59",
    )
    group.add_argument("--head_selection_expected_role", default="S")
    group.add_argument(
        "--head_selection_feature_subtype",
        default="same_frame_mass",
    )
    group.add_argument("--head_selection_expected_num_heads", type=int, default=59)
    group.add_argument("--expected_trainable_params", type=int, default=None)
    group.add_argument("--experiment_seed", type=int, default=42)
    return parser


def build_model(
    args: argparse.Namespace,
    accelerator,
    *,
    model_class: type[DINOv3XSSCContextSlotsWanModule] = DINOv3XSSCContextSlotsWanModule,
    extra_model_kwargs: dict | None = None,
) -> DINOv3XSSCContextSlotsWanModule:
    xssc_checkpoint = args.xssc_checkpoint
    if not args.disable_object_branch:
        xssc_checkpoint = resolve_latest_xssc_checkpoint(
            args.xssc_checkpoint,
            args.xssc_checkpoint_latest_dir,
        )
    args.xssc_checkpoint = xssc_checkpoint
    return model_class(
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
        xssc_filter_empty_amg=args.xssc_filter_empty_amg,
        xssc_empty_amg_max_resample_attempts=args.xssc_empty_amg_max_resample_attempts,
        xssc_input_size=args.xssc_input_size,
        xssc_max_time_steps=args.xssc_max_time_steps,
        object_lora_rank=args.object_lora_rank,
        object_lora_alpha=args.object_lora_alpha,
        object_lora_dropout=args.object_lora_dropout,
        xssc_slot_track_dropout=args.xssc_slot_track_dropout,
        self_attn_adaptation_mode=args.self_attn_adaptation_mode,
        pretrained_lora_expected_modules=args.pretrained_lora_expected_modules,
        self_attn_expected_num_blocks=args.self_attn_expected_num_blocks,
        self_attn_expected_num_heads=args.self_attn_expected_num_heads,
        self_attn_lora_rank=args.self_attn_lora_rank,
        self_attn_lora_alpha=args.self_attn_lora_alpha,
        self_attn_lora_dropout=args.self_attn_lora_dropout,
        head_selection_config=args.head_selection_config,
        head_selection_subset_id=args.head_selection_subset_id,
        head_selection_expected_role=args.head_selection_expected_role,
        head_selection_feature_subtype=args.head_selection_feature_subtype,
        head_selection_expected_num_heads=args.head_selection_expected_num_heads,
        enable_object_branch=not args.disable_object_branch,
        **(extra_model_kwargs or {}),
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
    full_sa_params = [
        param
        for name, param in model.pipe.dit.named_parameters()
        if param.requires_grad and _is_full_self_attn_lora_parameter(name)
    ]
    selected_head_params = [
        param
        for name, param in model.pipe.dit.named_parameters()
        if param.requires_grad and _is_head_selective_lora_parameter(name)
    ]
    projector_params = []
    if model.enable_object_branch:
        projector_params.extend(model.slot_norm.parameters())
        projector_params.extend(model.slot_projector.parameters())
        projector_params.extend(model.time_embedding.parameters())
    total = sum(param.numel() for param in model.trainable_modules())
    selected_head_count = sum(
        len(heads) for heads in model.selected_heads_by_block.values()
    )
    lines = [
        "=" * 78,
        "DINOv3 xSSC object/self-attention LoRA experiment",
        "=" * 78,
        f"Frozen Wan architecture/components: {args.wan_root}",
        f"Frozen DiT initialization: {model.base_dit_initialization}",
        f"Initialization LoRA modules removed: "
        f"{len(model.merged_pretrained_lora_modules)}",
        f"Object branch enabled: {model.enable_object_branch}",
        f"Frozen DINOv3 xSSC checkpoint: "
        f"{args.xssc_checkpoint if model.enable_object_branch else 'not loaded'}",
        f"Frozen DINOv3 pretrained weight: "
        f"{args.dinov3_checkpoint if model.enable_object_branch else 'not loaded'}",
        f"Self-attention adaptation mode: {model.self_attn_adaptation_mode}",
        f"Experiment seed: {args.experiment_seed}",
        f"Self-attention LoRA: rank={model.self_attn_lora_rank}, "
        f"alpha={model.self_attn_lora_alpha:g}, "
        f"dropout={model.self_attn_lora_dropout:g}",
        f"Selected {model.head_selection_expected_role} heads: "
        f"{selected_head_count} across "
        f"{len(model.selected_heads_by_block)} blocks",
        f"Head-selection config SHA256: "
        f"{model.head_selection_metadata.get('config_sha256', 'not-applicable')}",
        f"xSSC box source: {model.xssc_box_source if model.enable_object_branch else 'disabled'}",
        "Context policy: fixed full 8-frame context (text-only disabled)",
        f"Per-GPU training batch size: {args.train_batch_size}",
        f"xSSC shape: "
        f"{'[B, 8, ' + str(model.xssc_num_slots) + ', ' + str(model.xssc_slot_dim) + ']' if model.enable_object_branch else 'not constructed'}",
        f"Object token shape: "
        f"{'[B, ' + str(8 * model.xssc_num_slots) + ', ' + str(model.pipe.dit.dim) + ']' if model.enable_object_branch else 'not constructed'}",
        "Object attention base: "
        + ("text cross-attention + physical LoRA, baked and frozen" if model.enable_object_branch else "not constructed"),
        "Object LoRA: "
        + (
            f"rank={model.object_lora_rank}, alpha={model.object_lora_alpha:g}, "
            f"dropout={model.object_lora_dropout:g}"
            if model.enable_object_branch
            else "not constructed"
        ),
        "xSSC slot-track dropout: "
        + (
            f"{model.xssc_slot_track_dropout:g} "
            "(same slot mask across all 8 context frames)"
            if model.enable_object_branch
            else "not applicable"
        ),
        f"Trainable projector/time params: {sum(p.numel() for p in projector_params):,}",
        f"Trainable object-attention LoRA params: {sum(p.numel() for p in object_lora_params):,}",
        f"Trainable object-gate params: {sum(p.numel() for p in object_gate_params):,}",
        f"Trainable full self-attention LoRA params: "
        f"{sum(p.numel() for p in full_sa_params):,}",
        f"Trainable compact selected-head LoRA params: "
        f"{sum(p.numel() for p in selected_head_params):,}",
        f"Total trainable params: {total:,}",
        "Legacy Stage1A/Grounding/CoTracker/VGGT/JEPA modules: not constructed",
        "=" * 78,
    ]
    accelerator.print("\n".join(lines))


def main(
    *,
    build_parser_fn=build_parser,
    build_model_fn=build_model,
    build_dataset_fn=base.build_dataset,
    log_stage_summary_fn=_log_stage_summary,
    require_pretrained_lora: bool = True,
) -> None:
    parser = build_parser_fn()
    args = tvn.prepare_args(parser.parse_args())
    if int(args.fixed_num_context_frames) != base.XSSC_NUM_CONTEXT_FRAMES:
        parser.error(
            f"--fixed_num_context_frames must be {base.XSSC_NUM_CONTEXT_FRAMES} for this experiment"
        )
    if int(args.train_batch_size) <= 0:
        parser.error("--train_batch_size must be positive")
    if require_pretrained_lora and args.lora_checkpoint is None:
        parser.error(
            "--lora_checkpoint is required: all experiment modes must start from "
            "the same pretrained OpenVid LoRA"
        )
    if (
        args.self_attn_adaptation_mode in HEAD_SELECTIVE_ADAPTATION_MODES
        and args.head_selection_config is None
    ):
        parser.error("--head_selection_config is required for head-selective mode")
    if (
        args.disable_object_branch
        and args.self_attn_adaptation_mode
        not in ("full_sa", *HEAD_SELECTIVE_ADAPTATION_MODES)
    ):
        parser.error(
            "--disable_object_branch requires a self-attention adaptation mode"
        )
    if args.disable_object_branch and args.xssc_filter_empty_amg:
        parser.error(
            "--xssc_filter_empty_amg cannot be used when the object branch is disabled"
        )
    args.no_context_ratio = 0.0
    previous_handlers = tvn.install_interrupt_handlers()
    accelerator = tvn.build_accelerator(args)
    from accelerate.utils import set_seed

    set_seed(int(args.experiment_seed), device_specific=True)
    tvn.init_trackers(accelerator, args)

    dataset = build_dataset_fn(args)
    raw_train_dataset = dataset
    if int(args.train_batch_size) > 1:
        dataset = base.GroupedBatchDataset(dataset, args.train_batch_size)
    headonly_val_config = tvn.build_headonly_val_config(args)
    headonly_val_dataset = tvn.build_headonly_val_dataset(args, headonly_val_config)
    headonly_val_dataloader = tvn.build_headonly_val_dataloader(headonly_val_dataset, args)
    model = build_model_fn(args, accelerator)
    actual_trainable_params = sum(
        param.numel() for param in model.trainable_modules()
    )
    if (
        args.expected_trainable_params is not None
        and actual_trainable_params != int(args.expected_trainable_params)
    ):
        raise RuntimeError(
            "Trainable parameter count mismatch: "
            f"actual={actual_trainable_params:,}, "
            f"expected={int(args.expected_trainable_params):,}"
        )
    if not args.disable_object_branch and args.xssc_filter_empty_amg:
        model.set_empty_amg_resample_dataset(raw_train_dataset)

    if args.stage2_resume_from is not None:
        resume_checkpoint = tvn.resolve_lora_checkpoint_for_resume(
            args.stage2_resume_from
        )
        if args.self_attn_adaptation_mode in HEAD_SELECTIVE_ADAPTATION_MODES:
            identity_info = validate_head_selection_resume_checkpoint(
                model,
                resume_checkpoint,
            )
            accelerator.print(
                "Validated resume head identity: "
                f"num_heads={identity_info['num_heads']}, "
                f"config_sha256={identity_info['config_sha256']}"
            )
        resume_info = tvn._load_filtered_checkpoint_into_model(
            model,
            resume_checkpoint,
            include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
            include_substrings=(
                ".object_cross_attn.",
                ".object_gate",
                ".self_attn.",
            ),
        )
        expected_resume_count = sum(
            1 for _, param in model.named_parameters() if param.requires_grad
        )
        if (
            resume_info["loaded_count"] != expected_resume_count
            or resume_info["skipped_shape_mismatch"]
        ):
            raise RuntimeError(
                "Incomplete or incompatible object/self-attention resume checkpoint: "
                f"loaded={resume_info['loaded_count']}/{expected_resume_count}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )
        if accelerator.is_main_process:
            accelerator.print(
                "Loaded object/self-attention resume weights: "
                f"loaded_count={resume_info['loaded_count']}, "
                f"shape_mismatch={len(resume_info['skipped_shape_mismatch'])}"
            )

    log_stage_summary_fn(accelerator, model, args)
    model_logger = base.ModelLogger(
        tvn.get_checkpoint_dir(args),
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    runtime_state: dict = {}

    try:
        if args.task in ("sft:data_process", "direct_distill:data_process"):
            tvn.launch_data_process_task(accelerator, dataset, model, model_logger, args=args)
        else:
            run_train_loop_with_synced_checkpoint_saves(
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
