#!/usr/bin/env python3
"""Build an isolated bench.py batch from the current motion-analysis inventory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_INVENTORY = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis/inventory.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_bench"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_video_link(link: Path, source: Path) -> None:
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise RuntimeError(f"Conflicting video link: {link}")
        return
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.symlink_to(source.resolve())


def main() -> None:
    args = parse_args()
    inventory_path = args.inventory.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    manifest_entries = []
    for entry in inventory["entries"]:
        if entry["kind"] != "generated":
            continue
        entry_id = str(entry["entry_id"])
        source_video = Path(entry["source"]["path"]).resolve()
        source_json = source_video.with_suffix(".json")
        if not source_video.is_file() or not source_json.is_file():
            raise FileNotFoundError(f"Missing source pair for {entry_id}: {source_video}")

        target_json = cases_root / f"{entry_id}.json"
        target_video = target_json.with_suffix(".mp4")
        ensure_video_link(target_video, source_video)
        metadata = {
            "entry_id": entry_id,
            "model": entry["model"],
            "seed": entry["seed"],
            "variant": entry["variant"],
            "role": entry["role"] or "baseline",
            "denoise_step_range": entry["denoise_step_range"],
            "source_json": str(source_json),
            "source_video": str(source_video),
            "source_cache_key": entry["source"]["cache_key"],
        }
        if target_json.is_file():
            current = json.loads(target_json.read_text(encoding="utf-8"))
            if current.get("_stc_bench") != metadata:
                raise RuntimeError(f"Existing batch entry has changed source: {target_json}")
        else:
            payload = json.loads(source_json.read_text(encoding="utf-8"))
            payload["_stc_bench"] = metadata
            payload["output_video"] = str(target_video)
            atomic_json(target_json, payload)
        manifest_entries.append(metadata)

    manifest = {
        "schema_version": 1,
        "inventory": str(inventory_path),
        "num_entries": len(manifest_entries),
        "entries": manifest_entries,
    }
    atomic_json(output_root / "batch_manifest.json", manifest)
    (output_root / "result_roots.txt").write_text(
        str(output_root) + "\n",
        encoding="utf-8",
    )
    print(f"[stc-bench-batch] entries={len(manifest_entries)} root={output_root}")


if __name__ == "__main__":
    main()
