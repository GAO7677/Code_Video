"""Runtime ablations for Wan DiT blocks without editing model source files."""

from __future__ import annotations

import json
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


ABLATION_MODES = (
    "baseline",
    "whole_block",
    "self_attn_zero",
    "self_attn_head_zero",
    "object_cross_attn",
)


@dataclass(frozen=True)
class DiTAblationSpec:
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
                raise ValueError("baseline mode must not specify block/head ids")
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


def _whole_block_identity(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    del self, args, kwargs
    return x


def _attention_zero(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    del self, args, kwargs
    return torch.zeros_like(x)


def _replace_forward(module: torch.nn.Module, replacement) -> None:
    if hasattr(module, "_aaa_wan_dit_original_forward"):
        raise RuntimeError(f"Ablation is already installed on {type(module).__name__}")
    module._aaa_wan_dit_original_forward = module.forward
    module.forward = types.MethodType(replacement, module)


def _zero_projection_input_head(
    projection: torch.nn.Module,
    *,
    num_heads: int,
    head_id: int,
    counter: _ForwardCounter,
) -> None:
    if not 0 <= head_id < num_heads:
        raise ValueError(f"head id must be in [0, {num_heads - 1}], got {head_id}")
    if hasattr(projection, "_aaa_wan_dit_original_forward"):
        raise RuntimeError(
            f"Ablation is already installed on {type(projection).__name__}"
        )
    original_forward = projection.forward
    projection._aaa_wan_dit_original_forward = original_forward

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


def install_dit_ablation(
    dit: torch.nn.Module,
    spec: DiTAblationSpec,
    *,
    expected_num_blocks: int | None = 30,
) -> dict[str, object]:
    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise TypeError("Expected Wan DiT with a .blocks collection")

    num_blocks = len(blocks)
    if expected_num_blocks is not None and num_blocks != expected_num_blocks:
        raise ValueError(
            f"Expected {expected_num_blocks} Wan DiT blocks, found {num_blocks}"
        )
    spec.validate(num_blocks)

    disabled_module = None
    counter = _ForwardCounter()
    num_heads: int | None = None
    if spec.mode != "baseline":
        block = blocks[spec.block_id]
        if spec.mode == "whole_block":
            disabled_module = f"blocks.{spec.block_id}"
            _replace_forward(block, _whole_block_identity)
        elif spec.mode == "self_attn_zero":
            self_attn = getattr(block, "self_attn", None)
            if self_attn is None:
                raise AttributeError(f"Wan block {spec.block_id} has no self_attn")
            disabled_module = f"blocks.{spec.block_id}.self_attn"
            _replace_forward(self_attn, _attention_zero)
        elif spec.mode == "self_attn_head_zero":
            self_attn = getattr(block, "self_attn", None)
            if self_attn is None:
                raise AttributeError(f"Wan block {spec.block_id} has no self_attn")
            num_heads = int(getattr(self_attn, "num_heads"))
            disabled_module = (
                f"blocks.{spec.block_id}.self_attn.attn_output_head"
                f"[{spec.head_id}]"
            )
            _zero_projection_input_head(
                self_attn.o,
                num_heads=num_heads,
                head_id=int(spec.head_id),
                counter=counter,
            )
        elif spec.mode == "object_cross_attn":
            object_cross_attn = getattr(block, "object_cross_attn", None)
            if object_cross_attn is None:
                raise AttributeError(
                    f"Wan block {spec.block_id} has no object_cross_attn; "
                    "this mode is only valid for the xSSC model"
                )
            disabled_module = f"blocks.{spec.block_id}.object_cross_attn"
            _replace_forward(object_cross_attn, _attention_zero)

    metadata = asdict(spec)
    if spec.mode == "self_attn_zero":
        attention_semantics = "self_attention_output=zeros_like(query)"
    elif spec.mode == "self_attn_head_zero":
        attention_semantics = (
            "selected_self_attention_head_output_zero_before_output_projection"
        )
    elif spec.mode == "object_cross_attn":
        attention_semantics = "object_cross_attention_output=zeros_like(query)"
    else:
        attention_semantics = None
    metadata.update(
        {
            "tag": spec.tag,
            "num_dit_blocks": num_blocks,
            "disabled_module": disabled_module,
            "whole_block_semantics": "x_out=x_in",
            "attention_semantics": attention_semantics,
            "num_attention_heads": num_heads,
            "installation_point": "self_attention_output_projection_input",
        }
    )
    dit._aaa_wan_dit_ablation = metadata
    dit._aaa_wan_dit_head_ablation_counter = counter
    return metadata


def get_dit_head_ablation_call_count(dit: torch.nn.Module) -> int | None:
    counter = getattr(dit, "_aaa_wan_dit_head_ablation_counter", None)
    if counter is None:
        return None
    return int(counter.count)


def cli_value(args: list[str], option: str) -> str | None:
    try:
        index = args.index(option)
    except ValueError:
        return None
    if index + 1 >= len(args):
        raise ValueError(f"{option} requires a value")
    return args[index + 1]


def cli_path(args: list[str], option: str) -> Path | None:
    value = cli_value(args, option)
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def annotate_result_files(
    roots: list[Path | None],
    metadata: dict[str, object],
    *,
    negative_prompt: str | None = None,
) -> dict[str, int]:
    """Add the exact ablation spec to generated JSON and JSONL artifacts."""
    json_paths: set[Path] = set()
    jsonl_paths: set[Path] = set()
    for root in roots:
        if root is None or not root.is_dir():
            continue
        json_paths.update(root.rglob("*.json"))
        jsonl_paths.update(root.rglob("*.jsonl"))

    json_count = 0
    for path in sorted(json_paths):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            continue
        payload["dit_ablation"] = metadata
        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary_path.replace(path)
        json_count += 1

    jsonl_count = 0
    for path in sorted(jsonl_paths):
        annotated_lines: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    payload["dit_ablation"] = metadata
                    if negative_prompt is not None:
                        payload["negative_prompt"] = negative_prompt
                annotated_lines.append(json.dumps(payload, ensure_ascii=False))
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for line in annotated_lines:
                handle.write(line + "\n")
        temporary_path.replace(path)
        jsonl_count += 1

    return {"json_files": json_count, "jsonl_files": jsonl_count}
