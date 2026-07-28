"""Runtime-only ablations for the official PhysRVG Wan transformer."""

from __future__ import annotations

import types
from dataclasses import asdict, dataclass
from typing import Any

import torch


ABLATION_MODES = (
    "baseline",
    "whole_block",
    "self_attn_zero",
    "self_attn_head_zero",
    "text_cross_attn_zero",
    "ffn_zero",
    "lora_off",
)

EXPECTED_LORA_MODULES = (
    "attn1.to_k",
    "attn1.to_out.0",
    "attn1.to_q",
    "attn1.to_v",
    "attn2.to_k",
    "attn2.to_out.0",
    "attn2.to_q",
    "attn2.to_v",
    "ffn.net.0.proj",
    "ffn.net.2",
)


@dataclass(frozen=True)
class PhysRVGAblationSpec:
    mode: str = "baseline"
    block_id: int | None = None
    head_id: int | None = None

    def validate(self, num_blocks: int) -> None:
        if self.mode not in ABLATION_MODES:
            raise ValueError(
                f"Unsupported ablation mode {self.mode!r}; expected one of {ABLATION_MODES}"
            )
        if self.mode == "baseline":
            if self.block_id is not None or self.head_id is not None:
                raise ValueError("baseline must not specify block/head ids")
            return
        if self.block_id is None:
            raise ValueError(f"{self.mode} requires a block id")
        if not 0 <= self.block_id < num_blocks:
            raise ValueError(
                f"block id must be in [0, {num_blocks - 1}], got {self.block_id}"
            )
        if self.mode == "self_attn_head_zero":
            if self.head_id is None:
                raise ValueError("self_attn_head_zero requires a head id")
        elif self.head_id is not None:
            raise ValueError(f"{self.mode} must not specify a head id")

    @property
    def tag(self) -> str:
        if self.mode == "baseline":
            return "baseline"
        if self.mode == "self_attn_head_zero":
            return (
                f"{self.mode}_block{self.block_id:02d}_head{self.head_id:02d}"
            )
        return f"{self.mode}_block{self.block_id:02d}"


class _ForwardCounter:
    def __init__(self) -> None:
        self.count = 0


class _DenoiseStepGate:
    def __init__(
        self,
        *,
        start: int,
        end: int,
        total_steps: int,
        calls_per_step: int,
    ) -> None:
        if not 0 <= start < end <= total_steps:
            raise ValueError(
                "active denoise step range must satisfy "
                f"0 <= start < end <= {total_steps}, got [{start}, {end})"
            )
        if calls_per_step <= 0:
            raise ValueError("calls_per_step must be positive")
        self.start = start
        self.end = end
        self.total_steps = total_steps
        self.calls_per_step = calls_per_step
        self.forward_calls = 0
        self.active = False

    @property
    def step_index(self) -> int:
        return (self.forward_calls // self.calls_per_step) % self.total_steps


def _install_denoise_step_gate(
    module: torch.nn.Module,
    gate: _DenoiseStepGate,
) -> None:
    if hasattr(module, "_physrvg_step_gate_original_forward"):
        raise RuntimeError("Denoise step gate is already installed")
    original_forward = module.forward
    module._physrvg_step_gate_original_forward = original_forward

    def forward_with_step_gate(
        self: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ):
        del self
        gate.active = gate.start <= gate.step_index < gate.end
        try:
            return original_forward(*args, **kwargs)
        finally:
            gate.active = False
            gate.forward_calls += 1

    module.forward = types.MethodType(forward_with_step_gate, module)


def _resolve_blocks(transformer: torch.nn.Module) -> tuple[torch.nn.ModuleList, str]:
    candidates = (
        ("transformer", transformer),
        ("transformer.base_model.model", getattr(getattr(transformer, "base_model", None), "model", None)),
        ("transformer.model", getattr(transformer, "model", None)),
        ("transformer.module", getattr(transformer, "module", None)),
    )
    for path, candidate in candidates:
        blocks = getattr(candidate, "blocks", None)
        if blocks is not None:
            return blocks, path
    raise TypeError(
        "Could not find PhysRVG transformer blocks; expected .blocks or "
        ".base_model.model.blocks"
    )


def _replace_with_zero(
    module: torch.nn.Module,
    counter: _ForwardCounter,
) -> None:
    if hasattr(module, "_physrvg_ablation_original_forward"):
        raise RuntimeError(f"Ablation already installed on {type(module).__name__}")
    module._physrvg_ablation_original_forward = module.forward

    def zero_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, args, kwargs
        counter.count += 1
        return torch.zeros_like(hidden_states)

    module.forward = types.MethodType(zero_forward, module)


def _replace_block_with_identity(
    block: torch.nn.Module,
    counter: _ForwardCounter,
) -> None:
    if hasattr(block, "_physrvg_ablation_original_forward"):
        raise RuntimeError(f"Ablation already installed on {type(block).__name__}")
    block._physrvg_ablation_original_forward = block.forward

    def identity_forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self, args, kwargs
        counter.count += 1
        return hidden_states

    block.forward = types.MethodType(identity_forward, block)


def _zero_projection_input_head(
    projection: torch.nn.Module,
    *,
    num_heads: int,
    head_id: int,
    counter: _ForwardCounter,
) -> None:
    if not 0 <= head_id < num_heads:
        raise ValueError(f"head id must be in [0, {num_heads - 1}], got {head_id}")
    if hasattr(projection, "_physrvg_ablation_original_forward"):
        raise RuntimeError(
            f"Ablation already installed on {type(projection).__name__}"
        )
    original_forward = projection.forward
    projection._physrvg_ablation_original_forward = original_forward

    def forward_with_head_zero(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self
        if hidden_states.shape[-1] % num_heads != 0:
            raise RuntimeError(
                f"attention width {hidden_states.shape[-1]} is not divisible "
                f"by {num_heads} heads"
            )
        head_dim = hidden_states.shape[-1] // num_heads
        per_head = hidden_states.reshape(
            *hidden_states.shape[:-1], num_heads, head_dim
        ).clone()
        per_head[..., head_id, :] = 0
        counter.count += 1
        return original_forward(
            per_head.reshape_as(hidden_states), *args, **kwargs
        )

    projection.forward = types.MethodType(
        forward_with_head_zero,
        projection,
    )


def _zero_projection_input_heads(
    projection: torch.nn.Module,
    *,
    num_heads: int,
    head_ids: tuple[int, ...],
    counter: _ForwardCounter,
    step_gate: _DenoiseStepGate | None = None,
) -> None:
    if not head_ids:
        raise ValueError("head_ids must not be empty")
    if len(head_ids) != len(set(head_ids)):
        raise ValueError(f"head_ids contain duplicates: {head_ids}")
    for head_id in head_ids:
        if not 0 <= head_id < num_heads:
            raise ValueError(
                f"head id must be in [0, {num_heads - 1}], got {head_id}"
            )
    if hasattr(projection, "_physrvg_ablation_original_forward"):
        raise RuntimeError(
            f"Ablation already installed on {type(projection).__name__}"
        )
    original_forward = projection.forward
    projection._physrvg_ablation_original_forward = original_forward

    def forward_with_heads_zero(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        del self
        if step_gate is not None and not step_gate.active:
            return original_forward(hidden_states, *args, **kwargs)
        if hidden_states.shape[-1] % num_heads != 0:
            raise RuntimeError(
                f"attention width {hidden_states.shape[-1]} is not divisible "
                f"by {num_heads} heads"
            )
        head_dim = hidden_states.shape[-1] // num_heads
        per_head = hidden_states.reshape(
            *hidden_states.shape[:-1], num_heads, head_dim
        ).clone()
        per_head[..., list(head_ids), :] = 0
        counter.count += len(head_ids)
        return original_forward(
            per_head.reshape_as(hidden_states), *args, **kwargs
        )

    projection.forward = types.MethodType(forward_with_heads_zero, projection)


def _disable_block_lora(
    block: torch.nn.Module,
    counter: _ForwardCounter,
) -> list[str]:
    disabled: list[str] = []
    for name, module in block.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        enable_adapters = getattr(module, "enable_adapters", None)
        if not callable(enable_adapters):
            raise TypeError(f"LoRA module {name!r} has no enable_adapters method")
        enable_adapters(enabled=False)
        disabled.append(name)

    if tuple(sorted(disabled)) != EXPECTED_LORA_MODULES:
        raise RuntimeError(
            "Unexpected PhysRVG LoRA coverage in target block: "
            f"expected {EXPECTED_LORA_MODULES}, found {tuple(sorted(disabled))}"
        )
    disabled.sort()

    def count_block_calls(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
    ) -> None:
        counter.count += 1

    block.register_forward_pre_hook(count_block_calls)
    return disabled


def install_physrvg_ablation(
    transformer: torch.nn.Module,
    spec: PhysRVGAblationSpec,
    *,
    expected_num_blocks: int = 30,
) -> dict[str, object]:
    """Install one ablation after the full DiT checkpoint and LoRA are loaded."""

    blocks, blocks_path = _resolve_blocks(transformer)
    num_blocks = len(blocks)
    if num_blocks != expected_num_blocks:
        raise ValueError(
            f"Expected {expected_num_blocks} PhysRVG blocks, found {num_blocks}"
        )
    spec.validate(num_blocks)

    counter = _ForwardCounter()
    disabled_module: str | None = None
    disabled_lora_modules: list[str] = []
    semantics: str | None = None
    num_heads: int | None = None

    if spec.mode != "baseline":
        block = blocks[spec.block_id]
        block_prefix = f"{blocks_path}.blocks.{spec.block_id}"
        if spec.mode == "whole_block":
            disabled_module = block_prefix
            semantics = "block_output=block_input"
            _replace_block_with_identity(block, counter)
        elif spec.mode == "self_attn_zero":
            disabled_module = f"{block_prefix}.attn1"
            semantics = "attn1_output=zeros_like(attn1_input)"
            _replace_with_zero(block.attn1, counter)
        elif spec.mode == "self_attn_head_zero":
            num_heads = int(block.attn1.heads)
            disabled_module = (
                f"{block_prefix}.attn1.attention_output_head[{spec.head_id}]"
            )
            semantics = (
                "selected_attn1_head_output_zero_before_to_out_projection"
            )
            _zero_projection_input_head(
                block.attn1.to_out[0],
                num_heads=num_heads,
                head_id=int(spec.head_id),
                counter=counter,
            )
        elif spec.mode == "text_cross_attn_zero":
            disabled_module = f"{block_prefix}.attn2"
            semantics = "attn2_output=zeros_like(attn2_input)"
            _replace_with_zero(block.attn2, counter)
        elif spec.mode == "ffn_zero":
            disabled_module = f"{block_prefix}.ffn"
            semantics = "ffn_output=zeros_like(ffn_input)"
            _replace_with_zero(block.ffn, counter)
        elif spec.mode == "lora_off":
            disabled_module = block_prefix
            semantics = "disable_all_PEFT_LoRA_modules_in_target_block"
            disabled_lora_modules = _disable_block_lora(block, counter)

    metadata = asdict(spec)
    metadata.update(
        {
            "tag": spec.tag,
            "num_dit_blocks": num_blocks,
            "blocks_path": blocks_path,
            "disabled_module": disabled_module,
            "disabled_lora_modules": disabled_lora_modules,
            "semantics": semantics,
            "num_attention_heads": num_heads,
            "installation_point": "after_full_dit_and_lora_load",
        }
    )
    transformer._physrvg_ablation_metadata = metadata
    transformer._physrvg_ablation_counter = counter
    return metadata


def get_ablation_call_count(transformer: torch.nn.Module) -> int | None:
    counter = getattr(transformer, "_physrvg_ablation_counter", None)
    if counter is None:
        return None
    return int(counter.count)


def install_grouped_physrvg_head_ablation(
    transformer: torch.nn.Module,
    *,
    category: str,
    targets: list[tuple[int, int]],
    expected_num_blocks: int = 30,
    active_step_range: tuple[int, int] | None = None,
    total_steps: int = 40,
    calls_per_step: int = 1,
) -> dict[str, object]:
    """Zero selected attn1 heads, including multiple heads per block."""

    blocks, blocks_path = _resolve_blocks(transformer)
    num_blocks = len(blocks)
    if num_blocks != expected_num_blocks:
        raise ValueError(
            f"Expected {expected_num_blocks} PhysRVG blocks, found {num_blocks}"
        )
    normalized_category = category.strip().upper()
    if (
        not normalized_category
        or not normalized_category.replace("_", "").isalnum()
    ):
        raise ValueError(f"Unsupported grouped head category {category!r}")
    if not targets:
        raise ValueError("Grouped head ablation requires at least one target")
    normalized_targets = [(int(block), int(head)) for block, head in targets]
    if len(normalized_targets) != len(set(normalized_targets)):
        raise ValueError("Grouped head ablation targets contain duplicates")
    counter = _ForwardCounter()
    step_gate = None
    if active_step_range is not None:
        step_gate = _DenoiseStepGate(
            start=int(active_step_range[0]),
            end=int(active_step_range[1]),
            total_steps=int(total_steps),
            calls_per_step=int(calls_per_step),
        )
        _install_denoise_step_gate(transformer, step_gate)
    target_metadata = []
    targets_by_block: dict[int, list[int]] = {}
    for block_id, head_id in normalized_targets:
        targets_by_block.setdefault(block_id, []).append(head_id)
    for block_id, head_ids in sorted(targets_by_block.items()):
        if not 0 <= block_id < num_blocks:
            raise ValueError(
                f"block id must be in [0, {num_blocks - 1}], got {block_id}"
            )
        block = blocks[block_id]
        num_heads = int(block.attn1.heads)
        normalized_head_ids = tuple(sorted(head_ids))
        _zero_projection_input_heads(
            block.attn1.to_out[0],
            num_heads=num_heads,
            head_ids=normalized_head_ids,
            counter=counter,
            step_gate=step_gate,
        )
        for head_id in normalized_head_ids:
            target_metadata.append(
                {
                    "block_id": block_id,
                    "head_id": head_id,
                    "num_attention_heads": num_heads,
                    "disabled_module": (
                        f"{blocks_path}.blocks.{block_id}.attn1."
                        f"attention_output_head[{head_id}]"
                    ),
                }
            )

    metadata: dict[str, object] = {
        "mode": "self_attn_grouped_head_zero",
        "category": normalized_category,
        "tag": f"self_attn_grouped_head_zero_category_{normalized_category.lower()}",
        "targets": target_metadata,
        "num_targets": len(target_metadata),
        "num_target_blocks": len(targets_by_block),
        "num_dit_blocks": num_blocks,
        "blocks_path": blocks_path,
        "semantics": (
            "selected_attn1_head_outputs_zero_before_to_out_projection"
        ),
        "installation_point": "after_full_dit_and_lora_load",
        "active_denoise_step_range": (
            None if step_gate is None else [step_gate.start, step_gate.end]
        ),
        "denoise_step_interval_semantics": (
            None if step_gate is None else "[start,end)"
        ),
        "total_denoise_steps": int(total_steps),
        "transformer_forward_calls_per_denoise_step": int(calls_per_step),
    }
    transformer._physrvg_ablation_metadata = metadata
    transformer._physrvg_ablation_counter = counter
    if step_gate is not None:
        transformer._physrvg_denoise_step_gate = step_gate
    return metadata
