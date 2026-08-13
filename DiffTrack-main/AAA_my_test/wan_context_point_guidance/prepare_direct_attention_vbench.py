#!/usr/bin/env python3
"""Build a clean VBench index for the direct-attention multicase pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_RESULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/"
    "direct_attention_tv_v1/firstframe_ti2v/generations"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/"
    "direct_attention_tv_v1/vbench_multicase"
)
DEFAULT_CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_001455_w000",
    "0613pybullet_sample_000336_w001",
    "phyco_kubric_ball_wall_collision_2025-08-08_00ac15",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px",
)
DEFAULT_SEEDS = (13248, 47326, 90094)
GROUPS = ("top100", "bottom100", "random100")
DIRECTIONS = ("context_to_future", "future_to_context", "bidirectional")
TARGETS = {
    "0613pybullet_sample_001460_w002": "object_A",
    "0613pybullet_sample_001455_w000": "object_A",
    "0613pybullet_sample_000336_w001": "object_B",
    "phyco_kubric_ball_wall_collision_2025-08-08_00ac15": "object_B",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px": "object_B",
}
VBENCH_FIELDS = (
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def variants(case: str) -> list[str]:
    target = TARGETS[case]
    return ["baseline"] + [
        f"{group}__{direction}__{target}"
        for group in GROUPS
        for direction in DIRECTIONS
    ]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args()

    result_root = args.result_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    index_root = output_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    expected_names: set[str] = set()

    for case in dict.fromkeys(args.cases):
        if case not in TARGETS:
            raise ValueError(f"unsupported pilot case: {case}")
        for seed in dict.fromkeys(args.seeds):
            for variant in variants(case):
                run_dir = result_root / case / f"seed_{seed:05d}" / variant
                video = run_dir / "generated.mp4"
                complete = run_dir / "complete.json"
                manifest_path = run_dir / "manifest.json"
                ready = video.is_file() and complete.is_file() and manifest_path.is_file()
                identity = f"{case}:{seed}:{variant}"
                name = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
                expected_names.add(name)
                record = {
                    "case": case,
                    "seed": int(seed),
                    "variant": variant,
                    "target": TARGETS[case],
                    "ready": bool(ready),
                    "run_dir": str(run_dir),
                    "video": str(video),
                    "result_json": None,
                }
                if ready:
                    manifest = read_json(manifest_path)
                    source_json = manifest.get("source_json")
                    if not isinstance(source_json, str) or not Path(source_json).is_file():
                        raise RuntimeError(f"missing source_json in {manifest_path}")
                    item_root = index_root / name
                    result_json = item_root / "result.json"
                    existing = read_json(result_json)
                    preserved = {
                        field: existing[field]
                        for field in VBENCH_FIELDS
                        if field in existing
                    }
                    payload = {
                        "input_json": str(Path(source_json).resolve()),
                        "output_video": str(video.resolve()),
                        "case": case,
                        "seed": int(seed),
                        "variant": variant,
                        "method_name": f"direct_attention__{variant}",
                        **preserved,
                    }
                    atomic_json(result_json, payload)
                    record["result_json"] = str(result_json)
                records.append(record)

    for child in index_root.iterdir():
        if child.is_dir() and child.name not in expected_names:
            for item in child.iterdir():
                if item.is_file() or item.is_symlink():
                    item.unlink()
            child.rmdir()

    snapshot = {
        "cases": list(dict.fromkeys(args.cases)),
        "seeds": list(dict.fromkeys(args.seeds)),
        "planned_video_count": len(records),
        "ready_video_count": sum(int(record["ready"]) for record in records),
        "index_root": str(index_root),
        "records": records,
    }
    atomic_json(output_root / "snapshot.json", snapshot)
    print(json.dumps({
        "planned_video_count": snapshot["planned_video_count"],
        "ready_video_count": snapshot["ready_video_count"],
        "index_root": snapshot["index_root"],
    }, indent=2))


if __name__ == "__main__":
    main()

