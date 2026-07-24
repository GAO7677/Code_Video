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

    def validate(self, num_blocks: int) -> None:
        if self.mode not in ABLATION_MODES:
            raise ValueError(
                f"Unsupported ablation mode {self.mode!r}; expected one of {ABLATION_MODES}"
            )
        if self.mode == "baseline":
            if self.block_id is not None:
                raise ValueError("baseline must not specify a block id")
            return
        if self.block_id is None:
            raise ValueError(f"{self.mode} requires a block id")
        if not 0 <= self.block_id < num_blocks:
            raise ValueError(
                f"block id must be in [0, {num_blocks - 1}], got {self.block_id}"
            )

    @property
    def tag(self) -> str:
        if self.mode == "baseline":
            return "baseline"
        return f"{self.mode}_block{self.block_id:02d}"


class _ForwardCounter:
    def __init__(self) -> None:
        self.count = 0


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
