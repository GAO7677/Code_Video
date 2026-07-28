#!/usr/bin/env python3
"""Build an isolated metric batch from the seed-851 phased ablation outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_st_phased_seed851/generated"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_st_phased_seed851_bench"
)
ROLE_PATTERN = re.compile(r"role-(S|T|ST|C)_steps(\d{2})_(\d{2})$")
SEED_PATTERN = re.compile(r"seed-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
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


def parse_source(source_root: Path, video: Path) -> dict[str, Any]:
    relative = video.relative_to(source_root)
    if len(relative.parts) < 4:
        raise ValueError(f"Unexpected generated-video path: {video}")
    model = relative.parts[0]
    seed_match = SEED_PATTERN.fullmatch(relative.parts[1])
    role_match = ROLE_PATTERN.fullmatch(relative.parts[2])
    if seed_match is None or role_match is None:
        raise ValueError(f"Cannot parse seed/role from: {video}")
    role, start, end = role_match.groups()
    return {
        "model": model,
        "seed": int(seed_match.group(1)),
        "variant": relative.parts[2],
        "role": role,
        "denoise_step_range": [int(start), int(end)],
        "relative_path": relative.as_posix(),
    }


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    missing_pairs = []
    for source_video in sorted(source_root.rglob("*.mp4")):
        source_json = source_video.with_suffix(".json")
        if not source_json.is_file():
            missing_pairs.append(str(source_video))
            continue

        metadata = parse_source(source_root, source_video)
        entry_id = "__".join(
            (
                metadata["model"],
                f"seed-{metadata['seed']:06d}",
                metadata["variant"],
                source_video.stem,
            )
        )
        target_json = cases_root / f"{entry_id}.json"
        target_video = target_json.with_suffix(".mp4")
        ensure_video_link(target_video, source_video)

        bench_metadata = {
            "entry_id": entry_id,
            **metadata,
            "source_json": str(source_json),
            "source_video": str(source_video),
        }
        if target_json.is_file():
            current = json.loads(target_json.read_text(encoding="utf-8"))
            if current.get("_stc_bench") != bench_metadata:
                raise RuntimeError(f"Existing batch entry changed: {target_json}")
        else:
            payload = json.loads(source_json.read_text(encoding="utf-8"))
            payload["_stc_bench"] = bench_metadata
            payload["output_video"] = str(target_video)
            atomic_json(target_json, payload)
        manifest_entries.append(bench_metadata)

    if missing_pairs:
        examples = "\n".join(missing_pairs[:10])
        raise FileNotFoundError(
            f"{len(missing_pairs)} videos have no sibling JSON. Examples:\n{examples}"
        )
    if not manifest_entries:
        raise RuntimeError(f"No generated video/JSON pairs found under {source_root}")

    atomic_json(
        output_root / "batch_manifest.json",
        {
            "schema_version": 1,
            "source_root": str(source_root),
            "num_entries": len(manifest_entries),
            "entries": manifest_entries,
        },
    )
    (output_root / "result_roots.txt").write_text(
        str(output_root) + "\n",
        encoding="utf-8",
    )
    print(
        f"[seed851-bench-batch] entries={len(manifest_entries)} "
        f"root={output_root}"
    )


if __name__ == "__main__":
    main()
