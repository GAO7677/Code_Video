"""Scheme-C entity-binding v2v inference matching the fresh training recipe."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_entity_id_binding_v2v as entity_v2v,
)


_REQUIRED_ENTITY_SUFFIXES = (
    "object_adapter.entity_binding_gate",
    "object_adapter.entity_id_embed.weight",
    "object_adapter.entity_text_down.weight",
    "object_adapter.entity_text_norm.bias",
    "object_adapter.entity_text_norm.weight",
    "object_adapter.entity_text_up.weight",
)


def _option_value(argv: list[str], option: str) -> str | None:
    try:
        index = argv.index(option)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        raise ValueError(f"{option} requires a value")
    return argv[index + 1]


def _checkpoint_file(weights_root: str) -> Path:
    path = Path(weights_root).expanduser().resolve()
    if path.is_dir():
        path = path / "checkpoint.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def audit_entity_checkpoint(weights_root: str) -> dict[str, object]:
    checkpoint = _checkpoint_file(weights_root)
    tensors: dict[str, dict[str, object]] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        available = list(handle.keys())
        for suffix in _REQUIRED_ENTITY_SUFFIXES:
            matches = [key for key in available if key.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected exactly one checkpoint tensor ending with {suffix!r}, "
                    f"found {matches}"
                )
            key = matches[0]
            tensor = handle.get_tensor(key)
            if not bool(torch.isfinite(tensor).all()):
                raise FloatingPointError(f"non-finite entity checkpoint tensor: {key}")
            tensors[key] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nonzero": int(torch.count_nonzero(tensor).item()),
                "numel": int(tensor.numel()),
                "abs_max": float(tensor.float().abs().max().item()),
            }
    text_up_key = next(key for key in tensors if key.endswith("entity_text_up.weight"))
    if int(tensors[text_up_key]["nonzero"]) == 0:
        raise RuntimeError(
            "entity_text_up.weight is still exactly zero; the checkpoint has not "
            "received an effective entity-binding optimizer update"
        )
    return {"checkpoint": str(checkpoint), "entity_tensors": tensors}


def _install_training_matched_defaults(argv: list[str]) -> None:
    defaults: tuple[tuple[str, str | None], ...] = (
        ("--grounding-text-prompt", ""),
        ("--grounding-enable-caption-terms", None),
        ("--grounding-caption-prompt-mode", "physical_noun_phrases"),
        ("--grounding-caption-max-phrases", "4"),
        ("--grounding-caption-min-score", "4.0"),
        ("--compact-object-context-slots", None),
        ("--object-adapter-mlp-residual-max-ratio", "3.0"),
        ("--object-branch-ratio-guard-max-ratio", "0.30"),
        ("--object-branch-ratio-guard-max-block-id", "-1"),
    )
    for option, value in defaults:
        if option in argv:
            continue
        argv.append(option)
        if value is not None:
            argv.append(value)


def main() -> None:
    weights_root = _option_value(sys.argv, "--weights-root")
    if weights_root is None:
        raise ValueError("--weights-root is required")
    audit = audit_entity_checkpoint(weights_root)
    print(f"[entity-checkpoint-audit] {audit}")
    _install_training_matched_defaults(sys.argv)
    entity_v2v.main()


if __name__ == "__main__":
    main()
