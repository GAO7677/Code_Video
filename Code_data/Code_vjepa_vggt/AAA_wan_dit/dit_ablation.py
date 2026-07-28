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


def _install_denoise_step_gate_on_block_entry(
    first_block: torch.nn.Module,
    gate: _DenoiseStepGate,
) -> None:
    if hasattr(first_block, "_aaa_wan_dit_step_gate_original_forward"):
        raise RuntimeError("Denoise step gate is already installed")
    original_forward = first_block.forward
    first_block._aaa_wan_dit_step_gate_original_forward = original_forward

    def forward_with_step_gate_entry(
        self: torch.nn.Module,
        *args: Any,
        **kwargs: Any,
    ):
        del self
        gate.active = gate.start <= gate.step_index < gate.end
        try:
            return original_forward(*args, **kwargs)
        finally:
            gate.forward_calls += 1

    first_block.forward = types.MethodType(
        forward_with_step_gate_entry,
        first_block,
    )


class DynamicGroupedHeadAblator:
    """Switch grouped Head-zero targets without rebuilding the Wan pipeline."""

    def __init__(
        self,
        dit: torch.nn.Module,
        *,
        expected_num_blocks: int | None = 30,
    ) -> None:
        blocks = getattr(dit, "blocks", None)
        if blocks is None:
            raise TypeError("Expected Wan DiT with a .blocks collection")
        self.dit = dit
        self.num_blocks = len(blocks)
        if (
            expected_num_blocks is not None
            and self.num_blocks != expected_num_blocks
        ):
            raise ValueError(
                f"Expected {expected_num_blocks} Wan DiT blocks, "
                f"found {self.num_blocks}"
            )
        self.num_heads_by_block: dict[int, int] = {}
        self.active_heads_by_block: dict[int, tuple[int, ...]] = {}
        self.call_count = 0
        self.metadata: dict[str, object] = {}
        for block_id, block in enumerate(blocks):
            self_attn = getattr(block, "self_attn", None)
            if self_attn is None:
                raise AttributeError(f"Wan block {block_id} has no self_attn")
            num_heads = int(getattr(self_attn, "num_heads"))
            self.num_heads_by_block[block_id] = num_heads
            self._install_projection_wrapper(self_attn.o, block_id)
        self.set_targets(category=None, targets=[])
        dit._aaa_wan_dit_dynamic_head_ablator = self

    def _install_projection_wrapper(
        self,
        projection: torch.nn.Module,
        block_id: int,
    ) -> None:
        if hasattr(projection, "_aaa_wan_dit_original_forward"):
            raise RuntimeError(
                f"Ablation is already installed on {type(projection).__name__}"
            )
        original_forward = projection.forward
        projection._aaa_wan_dit_original_forward = original_forward
        controller = self

        def forward_with_dynamic_head_zero(
            module: torch.nn.Module,
            hidden_states: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            del module
            head_ids = controller.active_heads_by_block.get(block_id, ())
            if not head_ids:
                return original_forward(hidden_states, *args, **kwargs)
            num_heads = controller.num_heads_by_block[block_id]
            if hidden_states.shape[-1] % num_heads != 0:
                raise RuntimeError(
                    f"attention width {hidden_states.shape[-1]} is not "
                    f"divisible by {num_heads} heads"
                )
            head_dim = hidden_states.shape[-1] // num_heads
            per_head = hidden_states.reshape(
                *hidden_states.shape[:-1], num_heads, head_dim
            ).clone()
            per_head[..., list(head_ids), :] = 0
            controller.call_count += len(head_ids)
            return original_forward(
                per_head.reshape_as(hidden_states), *args, **kwargs
            )

        projection.forward = types.MethodType(
            forward_with_dynamic_head_zero,
            projection,
        )

    def set_targets(
        self,
        *,
        category: str | None,
        targets: list[tuple[int, int]],
    ) -> dict[str, object]:
        normalized_targets = [(int(block), int(head)) for block, head in targets]
        if len(normalized_targets) != len(set(normalized_targets)):
            raise ValueError("Dynamic grouped Head targets contain duplicates")
        targets_by_block: dict[int, list[int]] = {}
        for block_id, head_id in normalized_targets:
            if not 0 <= block_id < self.num_blocks:
                raise ValueError(
                    f"block id must be in [0, {self.num_blocks - 1}], "
                    f"got {block_id}"
                )
            num_heads = self.num_heads_by_block[block_id]
            if not 0 <= head_id < num_heads:
                raise ValueError(
                    f"head id must be in [0, {num_heads - 1}], got {head_id}"
                )
            targets_by_block.setdefault(block_id, []).append(head_id)

        self.active_heads_by_block = {
            block_id: tuple(sorted(head_ids))
            for block_id, head_ids in targets_by_block.items()
        }
        self.call_count = 0
        target_metadata = [
            {
                "block_id": block_id,
                "head_id": head_id,
                "num_attention_heads": self.num_heads_by_block[block_id],
                "disabled_module": (
                    f"blocks.{block_id}.self_attn.attn_output_head[{head_id}]"
                ),
            }
            for block_id, head_ids in sorted(self.active_heads_by_block.items())
            for head_id in head_ids
        ]
        if category is None:
            metadata: dict[str, object] = {
                "mode": "baseline",
                "category": None,
                "tag": "baseline",
                "targets": [],
                "num_targets": 0,
                "num_target_blocks": 0,
                "num_dit_blocks": self.num_blocks,
                "attention_semantics": "no_head_output_modified",
                "installation_point": (
                    "self_attention_output_projection_input_dynamic_wrapper"
                ),
            }
        else:
            normalized_category = category.strip().upper()
            if (
                not normalized_category
                or not normalized_category.replace("_", "").isalnum()
            ):
                raise ValueError(
                    f"Unsupported grouped Head category {category!r}"
                )
            if not target_metadata:
                raise ValueError(
                    "A non-baseline dynamic ablation requires targets"
                )
            metadata = {
                "mode": "self_attn_grouped_head_zero",
                "category": normalized_category,
                "tag": (
                    "self_attn_consistent_head_zero_category_"
                    f"{normalized_category.lower()}"
                ),
                "targets": target_metadata,
                "num_targets": len(target_metadata),
                "num_target_blocks": len(self.active_heads_by_block),
                "num_dit_blocks": self.num_blocks,
                "attention_semantics": (
                    "selected_self_attention_head_outputs_zero_before_"
                    "output_projection"
                ),
                "installation_point": (
                    "self_attention_output_projection_input_dynamic_wrapper"
                ),
            }
        self.metadata = metadata
        self.dit._aaa_wan_dit_ablation = metadata
        return metadata


def install_dynamic_grouped_head_ablator(
    dit: torch.nn.Module,
    *,
    expected_num_blocks: int | None = 30,
) -> DynamicGroupedHeadAblator:
    return DynamicGroupedHeadAblator(
        dit,
        expected_num_blocks=expected_num_blocks,
    )


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
    _zero_projection_input_heads(
        projection,
        num_heads=num_heads,
        head_ids=(head_id,),
        counter=counter,
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


def install_grouped_head_ablation(
    dit: torch.nn.Module,
    *,
    category: str,
    targets: list[tuple[int, int]],
    expected_num_blocks: int | None = 30,
    active_step_range: tuple[int, int] | None = None,
    total_steps: int = 40,
    calls_per_step: int = 2,
) -> dict[str, object]:
    """Zero selected self-attention heads, including multiple heads per block."""

    blocks = getattr(dit, "blocks", None)
    if blocks is None:
        raise TypeError("Expected Wan DiT with a .blocks collection")
    num_blocks = len(blocks)
    if expected_num_blocks is not None and num_blocks != expected_num_blocks:
        raise ValueError(
            f"Expected {expected_num_blocks} Wan DiT blocks, found {num_blocks}"
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
        # DiffSynth model_fn_wan_video() bypasses dit.forward and iterates
        # dit.blocks directly. Block 0 is therefore the reliable model-call
        # boundary for both conditional and unconditional CFG passes.
        _install_denoise_step_gate_on_block_entry(blocks[0], step_gate)
    target_metadata = []
    targets_by_block: dict[int, list[int]] = {}
    for block_id, head_id in normalized_targets:
        targets_by_block.setdefault(block_id, []).append(head_id)
    for block_id, head_ids in sorted(targets_by_block.items()):
        if not 0 <= block_id < num_blocks:
            raise ValueError(
                f"block id must be in [0, {num_blocks - 1}], got {block_id}"
            )
        self_attn = getattr(blocks[block_id], "self_attn", None)
        if self_attn is None:
            raise AttributeError(f"Wan block {block_id} has no self_attn")
        num_heads = int(getattr(self_attn, "num_heads"))
        normalized_head_ids = tuple(sorted(head_ids))
        _zero_projection_input_heads(
            self_attn.o,
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
                        f"blocks.{block_id}.self_attn."
                        f"attn_output_head[{head_id}]"
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
        "attention_semantics": (
            "selected_self_attention_head_outputs_zero_before_output_projection"
        ),
        "installation_point": "self_attention_output_projection_input",
        "denoise_step_gate_entry_point": (
            None if step_gate is None else "blocks.0.forward"
        ),
        "active_denoise_step_range": (
            None if step_gate is None else [step_gate.start, step_gate.end]
        ),
        "denoise_step_interval_semantics": (
            None if step_gate is None else "[start,end)"
        ),
        "total_denoise_steps": int(total_steps),
        "dit_forward_calls_per_denoise_step": int(calls_per_step),
    }
    dit._aaa_wan_dit_ablation = metadata
    dit._aaa_wan_dit_head_ablation_counter = counter
    if step_gate is not None:
        dit._aaa_wan_dit_denoise_step_gate = step_gate
    return metadata


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
