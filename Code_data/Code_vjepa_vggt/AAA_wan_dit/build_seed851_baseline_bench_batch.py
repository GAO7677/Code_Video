#!/usr/bin/env python3
"""Stage the matched seed-851 test5 baselines for metric evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/pass1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_common22_test5_seed851_baseline_bench"
)
MODEL_SUBDIRS = {
    "wan_lora": Path("wan_lora/seed-000851/generated/wan_lora"),
    "xssc": Path("xssc/seed-000851/generated/xssc/results"),
    "physrvg": Path(
        "physrvg/seed-000851/generated/physrvg/test5_unique20/"
        "physRVG_steps40_512x896_08_49f"
    ),
}


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


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)

    manifest_entries = []
    case_sets: dict[str, set[str]] = {}
    for model, relative_root in MODEL_SUBDIRS.items():
        model_root = source_root / relative_root
        videos = sorted(model_root.glob("*.mp4"))
        case_sets[model] = {video.stem for video in videos}
        if len(videos) != 20:
            raise RuntimeError(
                f"Expected 20 baseline videos for {model}, found {len(videos)}"
            )
        for source_video in videos:
            source_json = source_video.with_suffix(".json")
            if not source_json.is_file():
                raise FileNotFoundError(f"Missing baseline JSON: {source_json}")
            entry_id = f"{model}__seed-000851__baseline__{source_video.stem}"
            target_json = cases_root / f"{entry_id}.json"
            target_video = target_json.with_suffix(".mp4")
            ensure_video_link(target_video, source_video)
            metadata = {
                "entry_id": entry_id,
                "model": model,
                "seed": 851,
                "variant": "baseline",
                "role": "baseline",
                "denoise_step_range": None,
                "source_json": str(source_json),
                "source_video": str(source_video),
            }
            if target_json.is_file():
                current = json.loads(target_json.read_text(encoding="utf-8"))
                if current.get("_stc_bench") != metadata:
                    raise RuntimeError(f"Existing baseline changed: {target_json}")
            else:
                payload = json.loads(source_json.read_text(encoding="utf-8"))
                payload["_stc_bench"] = metadata
                payload["output_video"] = str(target_video)
                atomic_json(target_json, payload)
            manifest_entries.append(metadata)

    if len({frozenset(stems) for stems in case_sets.values()}) != 1:
        raise RuntimeError("The three baseline models do not share the same cases")

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
        f"[seed851-baseline-batch] entries={len(manifest_entries)} "
        f"root={output_root}"
    )


if __name__ == "__main__":
    main()
