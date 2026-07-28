"""Load cross-model public stable Head targets from an aggregate CSV."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


MODELS = ("wan_lora", "xssc", "physrvg")
ROLES = ("S", "T", "P", "C", "G")
ROLE_GROUPS = {"ST": ("S", "T")}
ROLE_CHOICES = (*ROLES, *ROLE_GROUPS)
EXPECTED_COUNTS = {"S": 159, "T": 13, "P": 82, "C": 20, "G": 75}
EXPECTED_BLOCKS = 30
EXPECTED_HEADS = 24


def load_public_head_targets(
    report_path: Path,
    *,
    validate_common22_counts: bool = True,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    """Return heads with the same clear aggregate role in all three models."""

    path = report_path.expanduser().resolve()
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    expected_rows = len(MODELS) * EXPECTED_BLOCKS * EXPECTED_HEADS
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in {path}, found {len(rows)}")

    labels: dict[tuple[str, int, int], str] = {}
    for row in rows:
        model = str(row["model"])
        block = int(row["block"])
        head = int(row["head"])
        role = str(row["role"]).upper()
        if model not in MODELS:
            raise ValueError(f"Unexpected model {model!r} in {path}")
        if not 0 <= block < EXPECTED_BLOCKS or not 0 <= head < EXPECTED_HEADS:
            raise ValueError(f"Invalid target B{block}H{head} in {path}")
        if role not in {*ROLES, "M"}:
            raise ValueError(f"Unexpected role {role!r} in {path}")
        key = (model, block, head)
        if key in labels:
            raise ValueError(f"Duplicate aggregate row {key} in {path}")
        labels[key] = role

    targets = {role: [] for role in ROLES}
    disagreement_or_mixed = 0
    for block in range(EXPECTED_BLOCKS):
        for head in range(EXPECTED_HEADS):
            model_roles = [labels[(model, block, head)] for model in MODELS]
            if len(set(model_roles)) == 1 and model_roles[0] != "M":
                targets[model_roles[0]].append((block, head))
            else:
                disagreement_or_mixed += 1

    counts = {role: len(targets[role]) for role in ROLES}
    if validate_common22_counts and counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Common22 public Head counts changed: expected {EXPECTED_COUNTS}, "
            f"found {counts}"
        )
    total = sum(counts.values())
    if total + disagreement_or_mixed != EXPECTED_BLOCKS * EXPECTED_HEADS:
        raise RuntimeError("Public and non-public Heads do not partition the grid")

    source = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "models": list(MODELS),
        "roles": list(ROLES),
        "selection": "same non-M aggregate role across all three models",
        "counts": counts,
        "num_public_heads": total,
        "num_non_public_heads": disagreement_or_mixed,
        "num_total_heads": EXPECTED_BLOCKS * EXPECTED_HEADS,
        "block_counts": dict(
            sorted(Counter(block for values in targets.values() for block, _ in values).items())
        ),
    }
    return targets, source


def targets_for_role(
    report_path: Path,
    role: str,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    normalized = role.strip().upper()
    if normalized not in ROLE_CHOICES:
        raise ValueError(f"Unknown role {role!r}; expected one of {ROLE_CHOICES}")
    targets, source = load_public_head_targets(report_path)
    selected_roles = ROLE_GROUPS.get(normalized, (normalized,))
    selected = sorted(
        target
        for selected_role in selected_roles
        for target in targets[selected_role]
    )
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"Role group {normalized} contains duplicate Head targets")
    source = {
        **source,
        "requested_role": normalized,
        "selected_roles": list(selected_roles),
        "selected_role_counts": {
            selected_role: len(targets[selected_role])
            for selected_role in selected_roles
        },
        "selected_target_count": len(selected),
        "selection_is_union": len(selected_roles) > 1,
    }
    return selected, source
