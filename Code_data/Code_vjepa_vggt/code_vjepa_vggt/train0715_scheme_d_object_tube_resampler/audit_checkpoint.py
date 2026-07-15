#!/usr/bin/env python3
"""Audit Scheme-D v3 trainable checkpoints without loading Wan."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open


ACTIVE_BLOCK_IDS = (8, 11, 14, 17, 20, 23)


def resolve_checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "checkpoint.safetensors"
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def expected_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {
        "object_pooler.output_queries": (4, 256),
        "object_pooler.motion_encoder.motion_queries": (4, 256),
        "object_pooler.spatial_proj.weight": (256, 18),
        "object_adapter.entity_id_embed.weight": (4, 256),
    }
    for block_id in ACTIVE_BLOCK_IDS:
        prefix = f"blocks.{block_id}.object_cross_attn"
        shapes.update(
            {
                f"{prefix}.q.weight": (256, 3072),
                f"{prefix}.k.weight": (256, 256),
                f"{prefix}.v.weight": (256, 256),
                f"{prefix}.o.weight": (3072, 256),
                f"blocks.{block_id}.object_gate": (1, 1, 3072),
            }
        )
    return shapes


def classify_tensor(name: str) -> str:
    if name.startswith("object_pooler."):
        return "object_pooler"
    if name.startswith("object_adapter."):
        return "object_adapter"
    if ".object_cross_attn." in name:
        return "dit_object_attention"
    if name.endswith(".object_gate"):
        return "dit_object_gate"
    if ".norm4." in name:
        return "dit_object_norm"
    return "other"


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_checkpoint(
    checkpoint: Path,
    *,
    compare: Path | None = None,
) -> dict[str, object]:
    checkpoint = resolve_checkpoint(checkpoint)
    compare = None if compare is None else resolve_checkpoint(compare)
    errors: list[str] = []
    group_stats: dict[str, dict[str, float | int]] = {}
    shape_contract = expected_shapes()
    tensors: dict[str, torch.Tensor] = {}

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            tensor = handle.get_tensor(key)
            tensors[key] = tensor
            if not bool(torch.isfinite(tensor).all()):
                errors.append(f"non-finite tensor: {key}")
            group = classify_tensor(key)
            stats = group_stats.setdefault(
                group,
                {
                    "tensor_count": 0,
                    "elements": 0,
                    "abs_max": 0.0,
                    "rms_numerator": 0.0,
                },
            )
            stats["tensor_count"] = int(stats["tensor_count"]) + 1
            stats["elements"] = int(stats["elements"]) + int(tensor.numel())
            stats["abs_max"] = max(
                float(stats["abs_max"]),
                float(tensor.detach().float().abs().max().item()),
            )
            stats["rms_numerator"] = float(stats["rms_numerator"]) + float(
                tensor.detach().float().square().sum().item()
            )

    for key, shape in shape_contract.items():
        tensor = tensors.get(key)
        if tensor is None:
            errors.append(f"missing required tensor: {key}")
        elif tuple(tensor.shape) != shape:
            errors.append(
                f"shape mismatch for {key}: {tuple(tensor.shape)} != {shape}"
            )
    legacy_keys = [key for key in tensors if "object_embedding." in key]
    if legacy_keys:
        errors.append(f"legacy object_embedding tensors present: {legacy_keys}")
    for group, stats in group_stats.items():
        elements = max(int(stats["elements"]), 1)
        stats["rms"] = (float(stats.pop("rms_numerator")) / elements) ** 0.5

    comparison: dict[str, dict[str, float | int]] = {}
    if compare is not None:
        with safe_open(str(compare), framework="pt", device="cpu") as previous:
            previous_keys = set(previous.keys())
            if previous_keys != set(tensors):
                errors.append("comparison checkpoint tensor keys differ")
            accumulators: dict[str, dict[str, float | int]] = {}
            for key, current_tensor in tensors.items():
                if key not in previous_keys:
                    continue
                previous_tensor = previous.get_tensor(key)
                if previous_tensor.shape != current_tensor.shape:
                    errors.append(f"comparison shape mismatch: {key}")
                    continue
                delta = current_tensor.float() - previous_tensor.float()
                group = classify_tensor(key)
                stats = accumulators.setdefault(
                    group,
                    {
                        "elements": 0,
                        "changed_elements": 0,
                        "abs_sum": 0.0,
                        "abs_max": 0.0,
                    },
                )
                stats["elements"] = int(stats["elements"]) + int(delta.numel())
                stats["changed_elements"] = int(stats["changed_elements"]) + int(
                    torch.count_nonzero(delta).item()
                )
                stats["abs_sum"] = float(stats["abs_sum"]) + float(
                    delta.abs().sum().item()
                )
                stats["abs_max"] = max(
                    float(stats["abs_max"]), float(delta.abs().max().item())
                )
            for group, stats in accumulators.items():
                elements = max(int(stats["elements"]), 1)
                comparison[group] = {
                    "elements": int(stats["elements"]),
                    "changed_elements": int(stats["changed_elements"]),
                    "changed_fraction": int(stats["changed_elements"]) / elements,
                    "abs_mean": float(stats["abs_sum"]) / elements,
                    "abs_max": float(stats["abs_max"]),
                }

    return {
        "status": "failed" if errors else "passed",
        "architecture_version": 3,
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "tensor_count": len(tensors),
        "elements": sum(int(tensor.numel()) for tensor in tensors.values()),
        "group_stats": group_stats,
        "compare_checkpoint": None if compare is None else str(compare),
        "comparison": comparison,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = audit_checkpoint(args.checkpoint, compare=args.compare)
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
