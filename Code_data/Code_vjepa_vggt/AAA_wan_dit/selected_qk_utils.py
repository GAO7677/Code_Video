#!/usr/bin/env python3
"""Selection and recorder construction helpers for all-token QK capture."""

from __future__ import annotations

import json
from pathlib import Path

from ball_query_attention import BallQueryRecorderGroup
from selected_qk_matrix import SelectedQKMatrixRecorder
from self_attention_matrix import MatrixCaptureConfig, parse_step_numbers


def load_selection(path: Path) -> dict:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, dict) or not samples:
        raise ValueError(f"selection contains no samples: {path}")
    return samples


def build_selected_qk_group(
    *,
    selection: dict,
    model_label: str,
    case_key: str,
    steps_text: str,
    output_root: Path,
    output_bins: int,
    query_chunk: int,
) -> BallQueryRecorderGroup | None:
    model_samples = selection.get(model_label, {})
    item = model_samples.get(case_key)
    if item is None:
        return None
    steps = parse_step_numbers(steps_text)
    by_block: dict[int, dict[int, list[str]]] = {}
    for role, pair in item["roles"].items():
        block = int(pair["block"])
        head = int(pair["head"])
        by_block.setdefault(block, {}).setdefault(head, []).append(str(role))
    recorders = []
    for block, role_by_head in sorted(by_block.items()):
        recorders.append(
            SelectedQKMatrixRecorder(
                config=MatrixCaptureConfig(
                    block_id=block,
                    step_numbers=steps,
                    output_bins=int(output_bins),
                    query_chunk=int(query_chunk),
                ),
                model_label=model_label,
                output_root=output_root / f"block{block:02d}" / "matrices",
                selected_heads=tuple(sorted(role_by_head)),
                role_by_head=role_by_head,
            )
        )
    return BallQueryRecorderGroup(recorders)
