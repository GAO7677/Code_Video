#!/usr/bin/env python3
"""Merge two completed CYCLES Utonia cache shards without duplicating tensors."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--shard-cache-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    final_dir = args.final_cache_dir.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    shards = [path.expanduser().resolve() for path in args.shard_cache_dir]
    if len(shards) < 2:
        raise ValueError("at least two Utonia shards are required")
    if final_dir.exists() and any(final_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty final cache: {final_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    rows = []
    seen: set[str] = set()
    for shard in shards:
        config = json.loads((shard / "cache_config.json").read_text(encoding="utf-8"))
        if config.get("status") != "complete":
            raise RuntimeError(f"Utonia shard is not complete: {shard}")
        shard_rows = _read_jsonl(shard / str(config.get("index_file", "index.jsonl")))
        if len(shard_rows) != int(config.get("completed_rows", -1)):
            raise RuntimeError(f"index/config mismatch in {shard}")
        configs.append(config)
        for row in shard_rows:
            logical_key = str(row["logical_key"])
            if logical_key in seen:
                raise RuntimeError(f"duplicate logical key: {logical_key}")
            seen.add(logical_key)
            relative = Path(str(row["scene_file"]))
            source = shard / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            _link(source, final_dir / relative)
            rows.append(dict(row))

    rows.sort(key=lambda row: str(row["logical_key"]))
    if len(rows) != 70:
        raise RuntimeError(f"expected 70 merged Utonia rows, got {len(rows)}")
    _atomic_jsonl(final_dir / "index.jsonl", rows)
    config = deepcopy(configs[0])
    config.update(
        {
            "cache_dir": str(final_dir),
            "dataset_root": str(dataset_root),
            "selected_rows": len(rows),
            "completed_rows": len(rows),
            "status": "complete",
            "merged_shards": [str(shard) for shard in shards],
            "merged_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _atomic_json(final_dir / "cache_config.json", config)
    (final_dir / "README.md").write_text(
        "# physv_v2v_0819 CYCLES Utonia scene cache\n\n"
        "70 entries merged from two GPU shards. Features use context frame 7,\n"
        "official VGGT crop preprocessing, Utonia full-upcast, and the formal\n"
        "GroundingDINO/SAM2/CoTracker/3D-safe dynamic filtering path.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(final_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
