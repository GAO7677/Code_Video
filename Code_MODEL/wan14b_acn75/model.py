"""CUDA-only Pusa/Wan2.2 A14B initialization and fresh experiment LoRAs.

The upstream Pusa rank-512 adapters are merged at fixed scale into frozen
weights. Only the new, explicitly enumerated LoRAs are trainable. Upstream
source is vendored unchanged and verified before import. No model parameters
or checkpoint tensors are initialized on CPU; CPU copies occur only when
serializing the small experiment adapters.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import types
from typing import Any

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from safetensors import safe_open
from safetensors.torch import save_file


DEFAULT_TARGETS = (
    "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
    "cross_attn.q", "cross_attn.k", "cross_attn.v", "cross_attn.o",
    "ffn.0", "ffn.2",
)
ARCHITECTURE = dict(
    dim=5120, in_dim=16, ffn_dim=13824, out_dim=16, text_dim=4096,
    freq_dim=256, eps=1e-6, patch_size=(1, 2, 2), num_heads=40,
    num_layers=40, has_image_input=False,
)
VENDOR_ROOT = Path(__file__).resolve().parent.parent / "Pusa-VidGen"
EXPECTED_R32_PARAMS = 153_354_240


def load_upstream(vendor_root: str | Path = VENDOR_ROOT):
    """Import only the verified upstream model and utility, not DiffSynth."""
    vendor_root = Path(vendor_root)
    manifest = json.loads((vendor_root / "WAN14B_VENDOR_MANIFEST.json").read_text())
    for name, item in manifest["files"].items():
        actual = hashlib.sha256((vendor_root / name).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise RuntimeError(f"Modified upstream source: {name}")
    package_name = "_wan14b_pusa_upstream"
    package_path = vendor_root / "PusaV1/diffsynth/models"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_path)]
        sys.modules[package_name] = package
    module_name = package_name + ".wan_video_pusa"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            module_name, package_path / "wan_video_pusa.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name], manifest


def _header(path: Path) -> tuple[dict, str]:
    with path.open("rb") as stream:
        length_raw = stream.read(8)
        if len(length_raw) != 8:
            raise ValueError(f"Truncated safetensors: {path}")
        length = struct.unpack("<Q", length_raw)[0]
        if length > 32 * 1024 * 1024:
            raise ValueError(f"Unexpected safetensors header size: {path}")
        raw = stream.read(length)
    header = json.loads(raw)
    header.pop("__metadata__", None)
    if max(item["data_offsets"][1] for item in header.values()) + length + 8 != path.stat().st_size:
        raise ValueError(f"Safetensors file length mismatch: {path}")
    return header, hashlib.sha256(length_raw + raw).hexdigest()


class TensorFiles:
    """Header-indexed files; tensors are read directly onto the requested GPU."""
    def __init__(self, paths, device):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Checkpoint tensor loading requires a CUDA device")
        self.index = {}
        self.audit = []
        self.handles = {}
        self.stack = contextlib.ExitStack()
        try:
            for path in map(Path, paths):
                header, digest = _header(path)
                for name, item in header.items():
                    if name in self.index:
                        raise ValueError(f"Duplicate tensor key: {name}")
                    self.index[name] = (path, tuple(item["shape"]), item["dtype"])
                self.audit.append(dict(path=str(path.resolve()), size_bytes=path.stat().st_size,
                                       header_sha256=digest, tensor_count=len(header)))
                self.handles[path] = self.stack.enter_context(
                    safe_open(str(path), framework="pt", device=str(self.device)))
        except BaseException:
            self.stack.close()
            raise

    def get(self, name):
        path, shape, _ = self.index[name]
        tensor = self.handles[path].get_tensor(name)
        if tensor.device != self.device or tuple(tensor.shape) != shape:
            raise RuntimeError(f"Wrong loaded tensor device/shape: {name}")
        return tensor

    def close(self):
        self.stack.close()


class ExperimentLoRALinear(nn.Module):
    """PEFT-equivalent LoRA math with a frozen, already Pusa-merged base."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float,
                 generator: torch.Generator):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.scaling = float(alpha) / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                               device="meta", dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                               device="meta", dtype=torch.float32)
        self.lora_A.to_empty(device=base.weight.device)
        self.lora_B.to_empty(device=base.weight.device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5), generator=generator)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.base_layer(x)
        delta = self.lora_B(self.lora_A(self.dropout(x).to(self.lora_A.weight.dtype)))
        return base + delta.to(base.dtype) * self.scaling


def _resolve_parent(module: nn.Module, name: str):
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _check_sources(model, base: TensorFiles, pusa: TensorFiles):
    expected = {name: tuple(value.shape) for name, value in model.named_parameters()}
    if set(expected) != set(base.index):
        raise ValueError(f"Base tensor keys mismatch: missing={sorted(set(expected)-set(base.index))[:8]}, "
                         f"unexpected={sorted(set(base.index)-set(expected))[:8]}")
    for name, shape in expected.items():
        if shape != base.index[name][1]:
            raise ValueError(f"Base tensor shape mismatch for {name}: {base.index[name][1]} != {shape}")
    pusa_expected = {}
    for block_index in range(ARCHITECTURE["num_layers"]):
        for target in DEFAULT_TARGETS:
            linear = model.get_submodule(f"blocks.{block_index}.{target}")
            prefix = f"blocks.{block_index}.{target}"
            pusa_expected[prefix + ".lora_A.default.weight"] = (512, linear.in_features)
            pusa_expected[prefix + ".lora_B.default.weight"] = (linear.out_features, 512)
    if set(pusa_expected) != set(pusa.index):
        raise ValueError("Pusa must contain exactly the official 800 adapter tensors per expert")
    for name, shape in pusa_expected.items():
        if shape != pusa.index[name][1]:
            raise ValueError(f"Pusa tensor shape mismatch: {name}")


@torch.no_grad()
def _load_parameters(module, prefix, base, pusa, scale, consumed):
    merged = 0
    for local_name, meta_param in list(module.named_parameters()):
        name = prefix + local_name
        tensor = base.get(name).to(dtype=torch.bfloat16)
        a_name = name.removesuffix(".weight") + ".lora_A.default.weight"
        b_name = name.removesuffix(".weight") + ".lora_B.default.weight"
        if a_name in pusa.index:
            # Match upstream GeneralLoRAFromPeft: W + scale * B @ A.
            # The upstream pretrained adapter has rank=alpha=512 (ratio one).
            a = pusa.get(a_name).float()
            b = pusa.get(b_name).float()
            tensor = torch.addmm(tensor.float(), b, a, alpha=scale).to(torch.bfloat16)
            del a, b
            merged += 1
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite initialized weight: {name}")
        parent, leaf = _resolve_parent(module, local_name)
        setattr(parent, leaf, nn.Parameter(tensor, requires_grad=False))
        consumed.add(name)
    return merged


def _inject_block(block, targets, rank, alpha, dropout, generator):
    for target in targets:
        parent, leaf = _resolve_parent(block, target)
        base = getattr(parent, leaf)
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Configured LoRA target is not Linear: {target}")
        setattr(parent, leaf, ExperimentLoRALinear(base, rank, alpha, dropout, generator))


def shard_block(block, mesh, policy=None):
    """Shard trainable FP32 LoRAs separately from the frozen BF16 block.

    Torch 2.7 FSDP2 requires a single original dtype per communication group.
    Its list-module API groups all A/B linears into one collective, while the
    subsequent block wrapper excludes those already-owned parameters.
    """
    policy = policy or MixedPrecisionPolicy(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    lora_modules = []
    lora_parameter_ids = set()
    for module in block.modules():
        if isinstance(module, ExperimentLoRALinear):
            lora_modules.extend((module.lora_A, module.lora_B))
            lora_parameter_ids.update(id(p) for child in (module.lora_A, module.lora_B)
                                      for p in child.parameters())
    if not lora_modules:
        raise ValueError("Cannot shard a training block with no experiment LoRAs")
    for param in block.parameters():
        trainable = id(param) in lora_parameter_ids
        expected_dtype = torch.float32 if trainable else torch.bfloat16
        if param.dtype != expected_dtype or param.requires_grad != trainable:
            raise ValueError("FSDP block must have trainable FP32 LoRAs and frozen BF16 base")
    kwargs = dict(mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(lora_modules, **kwargs)
    fully_shard(block, **kwargs)
    return block


def _build_expert(model_cfg, expert, device, mesh, use_fsdp, upstream):
    with torch.device("meta"):
        model = upstream.WanModelPusa(**ARCHITECTURE).to(dtype=torch.bfloat16)
    model.requires_grad_(False)
    # Upstream RoPE tables are ordinary attributes, so regenerate on CUDA.
    with torch.device(device):
        model.freqs = upstream.precompute_freqs_cis_3d(
            ARCHITECTURE["dim"] // ARCHITECTURE["num_heads"])
    paths = sorted((Path(model_cfg["base_dir"]) / f"{expert}_noise_model").glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No {expert} expert safetensors in {model_cfg['base_dir']}")
    base = TensorFiles(paths, device)
    pusa = TensorFiles([Path(model_cfg["pusa_dir"]) / f"{expert}_noise_pusa.safetensors"], device)
    rank = int(model_cfg.get("lora_rank", 32))
    alpha = float(model_cfg.get("lora_alpha", 64))
    dropout = float(model_cfg.get("lora_dropout", 0))
    targets = tuple(model_cfg.get("lora_targets", DEFAULT_TARGETS))
    if rank < 1 or len(targets) != len(set(targets)) or not set(targets) <= set(DEFAULT_TARGETS):
        raise ValueError("Invalid LoRA rank or target list")
    if not 0 <= dropout < 1:
        raise ValueError("LoRA dropout must be in [0, 1)")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(model_cfg.get("lora_init_seed", 42)) + (0 if expert == "high" else 1))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                  cast_forward_inputs=False)
    shard_kwargs = dict(mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    consumed = set()
    merged = 0
    try:
        _check_sources(model, base, pusa)
        scale = float(model_cfg.get(f"{expert}_pusa_scale", 1.0))
        if not math.isfinite(scale):
            raise ValueError("Pusa merge scale must be finite")
        for index, block in enumerate(model.blocks):
            merged += _load_parameters(block, f"blocks.{index}.", base, pusa, scale, consumed)
            _inject_block(block, targets, rank, alpha, dropout, generator)
            if use_fsdp:
                shard_block(block, mesh, policy)
        # The non-block modules are small enough to load before root sharding.
        for name, child in model.named_children():
            if name != "blocks":
                merged += _load_parameters(child, name + ".", base, pusa, scale, consumed)
        if consumed != set(base.index) or merged != 400:
            raise RuntimeError("Incomplete base loading or Pusa merge")
        if use_fsdp:
            fully_shard(model, **shard_kwargs)
        model_audit = dict(expert=expert, base_files=base.audit, pusa_files=pusa.audit,
                           base_tensor_count=len(consumed), pusa_merged_modules=merged,
                           pusa_scale=scale, lora_rank=rank, lora_alpha=alpha,
                           lora_dropout=dropout, lora_targets=list(targets))
    finally:
        base.close()
        pusa.close()
    for name, param in model.named_parameters():
        local = param.to_local() if isinstance(param, DTensor) else param
        if local.device != device or local.is_meta:
            raise RuntimeError(f"Model parameter was not materialized on CUDA: {name}")
    return model, model_audit


class PusaDualExpert(nn.Module):
    def __init__(self, experts, gradient_checkpointing=True):
        super().__init__()
        self.experts = nn.ModuleDict(experts)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, latents, timesteps, text, expert="high"):
        if expert not in self.experts:
            raise ValueError(f"Unknown expert {expert}")
        if timesteps.shape != (latents.shape[0], latents.shape[2]):
            raise ValueError("Per-latent timesteps must have shape [B, T]")
        if latents.ndim != 5 or latents.shape[1] != 16:
            raise ValueError("Expected Wan14B latents [B,16,T,H,W]")
        if text.ndim != 3 or text.shape[0] != latents.shape[0] or text.shape[-1] != 4096:
            raise ValueError("Text embeddings must have shape [B,L,4096]")
        if not latents.is_cuda or text.device != latents.device or timesteps.device != latents.device:
            raise ValueError("All model inputs must be on the same CUDA device")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.experts[expert](
                latents.to(torch.bfloat16), timestep=timesteps.float(),
                context=text.to(torch.bfloat16),
                use_gradient_checkpointing=self.gradient_checkpointing,
                use_gradient_checkpointing_offload=False,
            )

    def set_requires_gradient_sync(self, sync: bool):
        for expert in self.experts.values():
            if isinstance(expert, FSDPModule):
                expert.set_requires_gradient_sync(sync, recurse=True)

    def set_is_last_backward(self, is_last_backward: bool):
        for expert in self.experts.values():
            if isinstance(expert, FSDPModule):
                expert.set_is_last_backward(is_last_backward)

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm, error_if_nonfinite=True):
        grads = []
        sharded = False
        mesh = None
        for param in self.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad = param.grad
            if isinstance(grad, DTensor):
                if len(grad.placements) != 1 or not isinstance(grad.placements[0], Shard):
                    raise ValueError("Gradient clipping currently requires one-dimensional FSDP sharding")
                sharded = True
                mesh = grad.device_mesh
                grad = grad.to_local()
            grads.append(grad)
        device = next(self.parameters()).device
        total = torch.zeros((), device=device, dtype=torch.float32)
        for grad in grads:
            total.add_(grad.float().square().sum())
        if sharded:
            dist.all_reduce(total, group=mesh.get_group())
        norm = total.sqrt()
        if error_if_nonfinite and not torch.isfinite(norm).item():
            raise RuntimeError("Non-finite global gradient norm")
        coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1)
        for grad in grads:
            grad.mul_(coefficient)
        return norm


def trainable_parameter_manifest(model):
    result = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if not (name.endswith(".lora_A.weight") or name.endswith(".lora_B.weight")):
                raise RuntimeError(f"Unexpected trainable base parameter: {name}")
            result.append(dict(name=name, shape=list(param.shape), numel=param.numel(), dtype=str(param.dtype)))
    return result


def build_model(config: dict, device, mesh=None):
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Only CUDA model materialization is permitted")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    cfg = config["model"]
    if cfg.get("base_dtype", "bfloat16") != "bfloat16" or cfg.get("lora_dtype", "float32") != "float32":
        raise ValueError("This implementation requires BF16 base weights and FP32 trainable LoRAs")
    use_fsdp = bool(config.get("runtime", {}).get("fsdp", True))
    if use_fsdp and (mesh is None or not dist.is_initialized()):
        raise ValueError("FSDP2 requires initialized distributed process group and device mesh")
    upstream, provenance = load_upstream(cfg.get("vendor_root", VENDOR_ROOT))
    experts = {}
    expert_audits = {}
    for expert in ("high", "low"):
        experts[expert], expert_audits[expert] = _build_expert(cfg, expert, device, mesh, use_fsdp, upstream)
    model = PusaDualExpert(experts, bool(cfg.get("gradient_checkpointing", True)))
    manifest = trainable_parameter_manifest(model)
    for expert in ("high", "low"):
        subset = [item for item in manifest if item["name"].startswith(f"experts.{expert}.")]
        count = sum(item["numel"] for item in subset)
        expert_audits[expert].update(trainable_parameters=count, trainable_tensors=len(subset),
                                    trainable_modules=len(subset) // 2)
        if int(cfg.get("lora_rank", 32)) == 32 and tuple(cfg.get("lora_targets", DEFAULT_TARGETS)) == DEFAULT_TARGETS:
            if count != EXPECTED_R32_PARAMS or len(subset) != 800:
                raise RuntimeError("Unexpected rank32 trainable parameter count")
    audit = dict(
        upstream=provenance, experts=expert_audits, trainable=manifest,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        materialization="meta_to_cuda_per_block", sharding="fsdp2" if use_fsdp else "none",
        lightx2v_loaded=False,
    )
    identity = dict(
        architecture=ARCHITECTURE, upstream_commit=provenance["commit"],
        base_revision=cfg.get("base_revision"), pusa_revision=cfg.get("pusa_revision"),
        lora_rank=int(cfg.get("lora_rank", 32)), lora_alpha=float(cfg.get("lora_alpha", 64)),
        lora_dropout=float(cfg.get("lora_dropout", 0)),
        lora_targets=list(cfg.get("lora_targets", DEFAULT_TARGETS)),
        pusa_scales={name: expert_audits[name]["pusa_scale"] for name in experts},
        source_headers={name: {
            kind: [item["header_sha256"] for item in expert_audits[name][kind]]
            for kind in ("base_files", "pusa_files")
        } for name in experts},
    )
    # JSON canonicalization also normalizes tuples before metadata comparison.
    model.initialization_identity = json.loads(json.dumps(identity, sort_keys=True))
    audit["initialization_identity"] = model.initialization_identity
    return model, audit


def _adapter_parameters(model):
    return {name: param for name, param in model.named_parameters() if param.requires_grad}


@torch.no_grad()
def save_adapter(model, path: str | Path, metadata: dict[str, Any] | None = None):
    """All ranks call; only rank zero writes the gathered experiment adapter."""
    path = Path(path)
    rank = dist.get_rank() if dist.is_initialized() else 0
    tensors = {}
    manifest = trainable_parameter_manifest(model)
    for name, param in _adapter_parameters(model).items():
        value = param.full_tensor() if isinstance(param, DTensor) else param
        if rank == 0:
            tensors[name] = value.detach().float().cpu().contiguous()
        del value
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        payload = dict(metadata or {}, schema="wan14b_experiment_lora_v1", trainable=manifest,
                       initialization_identity=model.initialization_identity)
        save_file(tensors, str(temporary), metadata={"audit": json.dumps(payload, sort_keys=True)})
        temporary.replace(path)
    if dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def load_adapter(model, path: str | Path, strict=True):
    """All ranks call; CUDA-read full matrices then copy each local FSDP shard."""
    parameters = _adapter_parameters(model)
    device = next(model.parameters()).device
    source = TensorFiles([Path(path)], device)
    try:
        stored = source.handles[Path(path)].metadata() or {}
        metadata = json.loads(stored.get("audit", "{}"))
        if strict and (metadata.get("schema") != "wan14b_experiment_lora_v1" or
                       metadata.get("initialization_identity") != model.initialization_identity):
            raise ValueError("Experiment adapter initialization identity does not match this model")
        if strict and set(source.index) != set(parameters):
            raise ValueError("Experiment adapter keys differ from the current trainable parameter set")
        loaded = 0
        for name, param in parameters.items():
            if name not in source.index:
                continue
            value = source.get(name).to(dtype=param.dtype)
            if tuple(value.shape) != tuple(param.shape):
                raise ValueError(f"Experiment adapter shape mismatch: {name}")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Non-finite adapter tensor: {name}")
            if isinstance(param, DTensor):
                value = distribute_tensor(value, device_mesh=param.device_mesh, placements=param.placements)
                param.to_local().copy_(value.to_local())
            else:
                param.copy_(value)
            loaded += 1
        return dict(path=str(Path(path).resolve()), loaded_tensors=loaded, files=source.audit)
    finally:
        source.close()
