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
    "object_cross_attn",
)


@dataclass(frozen=True)
class DiTAblationSpec:
    mode: str = "baseline"
    block_id: int | None = None

    def validate(self, num_blocks: int) -> None:
        if self.mode not in ABLATION_MODES:
            raise ValueError(
                f"Unsupported ablation mode {self.mode!r}; expected one of {ABLATION_MODES}"
            )
        if self.mode == "baseline":
            if self.block_id is not None:
                raise ValueError("baseline mode must not specify a block id")
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
        }
    )
    dit._aaa_wan_dit_ablation = metadata
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
