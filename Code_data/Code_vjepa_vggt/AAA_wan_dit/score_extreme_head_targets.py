#!/usr/bin/env python3
"""Load cross-model common-S score-extreme Head targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


MODELS = ("wan_lora", "xssc", "physrvg")
GROUPS = ("top", "bottom")


def targets_for_score_group(
    selection_path: Path,
    group: str,
) -> tuple[str, list[tuple[int, int]], dict[str, object]]:
    path = selection_path.expanduser().resolve()
    normalized_group = group.strip().lower()
    if normalized_group not in GROUPS:
        raise ValueError(f"unknown score group {group!r}; expected one of {GROUPS}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"selection contains no samples: {path}")

    prefix = f"S_{normalized_group}"
    targets_by_model: dict[str, list[tuple[int, int]]] = {}
    labels_by_model: dict[str, list[str]] = {}
    for model in MODELS:
        model_samples = samples.get(model)
        if not isinstance(model_samples, dict) or len(model_samples) != 1:
            raise ValueError(f"{model} must contain exactly one selected case")
        item = next(iter(model_samples.values()))
        roles = item.get("roles", {})
        selected = [
            (label, (int(pair["block"]), int(pair["head"])))
            for label, pair in roles.items()
            if str(label).startswith(prefix)
        ]
        selected.sort(key=lambda item: item[0])
        labels_by_model[model] = [label for label, _ in selected]
        targets_by_model[model] = [pair for _, pair in selected]

    reference = targets_by_model[MODELS[0]]
    if len(reference) != 10 or len(set(reference)) != 10:
        raise ValueError(
            f"{normalized_group} selection must contain 10 unique targets, "
            f"found {len(reference)}"
        )
    for model in MODELS[1:]:
        if targets_by_model[model] != reference:
            raise ValueError(f"{model} {normalized_group} targets differ across models")
    category = f"S_{normalized_group}10".upper()
    source = {
        "kind": "common_s_cross_model_mean_score_extreme",
        "selection_path": str(path),
        "selection_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "score_group": normalized_group,
        "ranking_metric": "mean(score_S across wan_lora/xssc/physrvg)",
        "labels": labels_by_model[MODELS[0]],
        "num_targets": len(reference),
        "targets": [
            {"block": block, "head": head} for block, head in reference
        ],
        "representative_seed": int(payload["representative_seed"]),
        "case": str(payload["case"]),
    }
    return category, reference, source
