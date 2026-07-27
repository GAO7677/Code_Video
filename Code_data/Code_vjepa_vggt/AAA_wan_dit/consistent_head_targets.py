"""Load protocol-consistent Head categories from an attention gallery manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = ("S", "ST", "T", "P", "C", "G")
OBJECT_PROTOCOLS = ("fixed_A", "moving_A", "moving_B")
EXPECTED_COUNTS = {
    "S": 110,
    "ST": 6,
    "T": 40,
    "P": 41,
    "C": 21,
    "G": 120,
}
EXPECTED_NUM_BLOCKS = 30
EXPECTED_NUM_HEADS = 24


def _load_payload(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def load_consistent_category_targets(
    metadata_path: Path,
    *,
    validate_expected_counts: bool = True,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, Any]]:
    """Return heads whose primary role agrees across all object-query protocols."""

    path = metadata_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Classification metadata not found: {path}")
    payload, sha256 = _load_payload(path)
    roles = payload.get("roles")
    if not isinstance(roles, list):
        raise TypeError(f"{path} has no roles list")

    labels_by_head: dict[tuple[int, int], dict[str, str]] = defaultdict(dict)
    duplicate_rows: list[tuple[int, int, str]] = []
    for row in roles:
        if not isinstance(row, dict) or row.get("protocol") not in OBJECT_PROTOCOLS:
            continue
        block = int(row["block"])
        head = int(row["head"])
        protocol = str(row["protocol"])
        primary = str(row["primary"]).upper()
        if not 0 <= block < EXPECTED_NUM_BLOCKS:
            raise ValueError(f"Invalid block {block} in {path}")
        if not 0 <= head < EXPECTED_NUM_HEADS:
            raise ValueError(f"Invalid head {head} in {path}")
        if primary not in CATEGORIES:
            raise ValueError(f"Unknown primary category {primary!r} in {path}")
        key = (block, head)
        if protocol in labels_by_head[key]:
            duplicate_rows.append((block, head, protocol))
        labels_by_head[key][protocol] = primary

    if duplicate_rows:
        raise ValueError(f"Duplicate object-protocol rows: {duplicate_rows[:8]}")
    expected_grid_size = EXPECTED_NUM_BLOCKS * EXPECTED_NUM_HEADS
    if len(labels_by_head) != expected_grid_size:
        raise ValueError(
            f"Expected {expected_grid_size} block/head entries, "
            f"found {len(labels_by_head)}"
        )

    targets: dict[str, list[tuple[int, int]]] = {
        category: [] for category in CATEGORIES
    }
    disagreement_count = 0
    for target, protocol_labels in sorted(labels_by_head.items()):
        missing = set(OBJECT_PROTOCOLS) - set(protocol_labels)
        if missing:
            raise ValueError(f"{target} is missing protocols {sorted(missing)}")
        unique_labels = set(protocol_labels.values())
        if len(unique_labels) == 1:
            targets[next(iter(unique_labels))].append(target)
        else:
            disagreement_count += 1

    counts = {category: len(targets[category]) for category in CATEGORIES}
    if validate_expected_counts and counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Consistent-category counts changed: expected {EXPECTED_COUNTS}, "
            f"found {counts}"
        )
    if sum(counts.values()) + disagreement_count != expected_grid_size:
        raise RuntimeError("Consistent and disagreement heads do not partition the grid")

    source = {
        "path": str(path),
        "sha256": sha256,
        "case": payload.get("case"),
        "model": payload.get("model"),
        "denoise_step_one_based": payload.get("denoise_step_one_based"),
        "cfg_branch": payload.get("cfg_branch"),
        "protocols": list(OBJECT_PROTOCOLS),
        "selection": "identical primary category across all three protocols",
        "counts": counts,
        "num_consistent_heads": sum(counts.values()),
        "num_disagreement_heads": disagreement_count,
        "num_total_heads": expected_grid_size,
    }
    return targets, source


def targets_for_category(
    metadata_path: Path,
    category: str,
    *,
    validate_expected_counts: bool = True,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    normalized = category.strip().upper()
    if normalized not in CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; expected one of {CATEGORIES}")
    targets, source = load_consistent_category_targets(
        metadata_path,
        validate_expected_counts=validate_expected_counts,
    )
    return targets[normalized], source


def category_counts(metadata_path: Path) -> Counter[str]:
    targets, _ = load_consistent_category_targets(metadata_path)
    return Counter({category: len(values) for category, values in targets.items()})
