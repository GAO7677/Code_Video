"""Load a frozen explicit Head subset from a dose-control manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_matched_subset(
    path: Path,
    subset_id: str,
) -> tuple[str, list[tuple[int, int]], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    raw = resolved.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != 1:
        raise ValueError("Matched-subset manifest schema_version must be 1")
    subsets = payload.get("subsets")
    if not isinstance(subsets, dict) or subset_id not in subsets:
        raise ValueError(f"Unknown subset id {subset_id!r} in {resolved}")
    record = subsets[subset_id]
    if not isinstance(record, dict):
        raise ValueError(f"Subset {subset_id!r} must be an object")
    target_rows = record.get("targets")
    if not isinstance(target_rows, list) or not target_rows:
        raise ValueError(f"Subset {subset_id!r} has no targets")
    targets = [
        (int(item["block"]), int(item["head"]))
        for item in target_rows
    ]
    if len(targets) != len(set(targets)):
        raise ValueError(f"Subset {subset_id!r} contains duplicate targets")
    for block, head in targets:
        if not 0 <= block < 30 or not 0 <= head < 24:
            raise ValueError(f"Invalid Head target {(block, head)}")
    expected = int(record.get("k", -1))
    if expected != len(targets):
        raise ValueError(
            f"Subset {subset_id!r} records k={expected}, found {len(targets)} targets"
        )
    role = str(record.get("role", "")).upper()
    if role not in {"S", "T", "C"}:
        raise ValueError(f"Subset {subset_id!r} has invalid role {role!r}")
    source = {
        "kind": "frozen_count_and_depth_matched_subset",
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "subset_id": subset_id,
        **record,
    }
    return subset_id.upper(), targets, source
