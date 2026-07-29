#!/usr/bin/env python3
"""Build the strict multi-case inventory for S-subtype and S-depth motion analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/head-role-dose-control-pilot/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
)
DEFAULT_REGION_ROOT = Path(
    "/data/gaoya/agent-data/cache/wan_dit_s_motion_sam2_regions"
)
DEFAULT_GT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan_dit_common22_test5_gt49f_896x512_bench/cases"
)
MODELS = ("wan_lora", "xssc", "physrvg")
FEATURE_SUBTYPES = ("local_enrichment", "same_frame_mass", "local_same_union")
FEATURE_STAGES = ((0, 10), (10, 20), (0, 40))
DEPTH_STAGES = ((0, 10), (10, 20), (0, 40))
DEPTH_SEEDS = (851, 3278)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region-root", type=Path, default=DEFAULT_REGION_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    material = f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "cache_key": hashlib.sha256(material).hexdigest()[:24],
    }


def region_fingerprint(path: Path) -> dict[str, Any]:
    path = path.resolve()
    files = [path / "regions.json", path / "regions.npz"]
    material = [str(path)]
    size = 0
    newest = 0
    for file_path in files:
        stat = file_path.stat()
        size += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
        material.extend((file_path.name, str(stat.st_size), str(stat.st_mtime_ns)))
    return {
        "path": str(path),
        "size_bytes": size,
        "mtime_ns": newest,
        "cache_key": hashlib.sha256("\0".join(material).encode()).hexdigest()[:24],
    }


def valid_video(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    gallery_root = manifest_path.parent.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = {str(case["id"]): case for case in payload["cases"]}
    depth_subsets = payload["s_depth_subsets"]
    records = payload["records"]

    region_by_case: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for case_id in cases:
        region_dir = args.region_root / case_id
        try:
            region_by_case[case_id] = region_fingerprint(region_dir)
        except FileNotFoundError:
            missing.append({"kind": "region_cache", "case_id": case_id, "path": str(region_dir)})

    entries: list[dict[str, Any]] = []
    for case_id in cases:
        if case_id not in region_by_case:
            continue
        gt_path = args.gt_root / f"gt__seed-000851__gt49f_896x512__{case_id}.mp4"
        if not valid_video(gt_path):
            missing.append({"kind": "gt", "case_id": case_id, "path": str(gt_path)})
            continue
        entries.append(
            {
                "entry_id": f"gt__{case_id}",
                "kind": "gt",
                "family": "gt",
                "case_id": case_id,
                "model": "gt",
                "seed": None,
                "variant": "gt",
                "subtype": None,
                "depth_stratum": None,
                "head_count": 0,
                "denoise_step_range": None,
                "source": fingerprint(gt_path),
                "region_cache": region_by_case[case_id],
            }
        )

    wanted_records: list[dict[str, Any]] = []
    for record in records:
        kind = record.get("kind")
        case_id = str(record.get("case_id"))
        model = str(record.get("model"))
        seed = int(record.get("seed", -1))
        stage = (int(record.get("start", -1)), int(record.get("end", -1)))
        if case_id not in region_by_case or model not in MODELS:
            continue
        if kind == "baseline" and seed in DEPTH_SEEDS:
            wanted_records.append(record)
        elif (
            kind == "s_feature_split"
            and seed == 851
            and record.get("feature_subtype") in FEATURE_SUBTYPES
            and stage in FEATURE_STAGES
        ):
            wanted_records.append(record)
        elif (
            kind == "s_depth"
            and seed in DEPTH_SEEDS
            and stage in DEPTH_STAGES
            and record.get("subset_id") in depth_subsets
        ):
            wanted_records.append(record)

    seen: set[str] = set()
    for record in wanted_records:
        case_id = str(record["case_id"])
        model = str(record["model"])
        seed = int(record["seed"])
        kind = str(record["kind"])
        start = int(record.get("start", -1))
        end = int(record.get("end", -1))
        if kind == "baseline":
            family = "baseline"
            variant = "baseline"
            subtype = None
            depth_stratum = None
        elif kind == "s_feature_split":
            family = "s_feature"
            subtype = str(record["feature_subtype"])
            depth_stratum = None
            variant = f"{subtype}__steps{start:02d}_{end:02d}"
        else:
            family = "s_depth"
            subtype = None
            depth_stratum = str(depth_subsets[record["subset_id"]]["depth_stratum"])
            variant = f"{depth_stratum}__steps{start:02d}_{end:02d}"
        entry_id = f"{family}__{model}__seed-{seed:06d}__{variant}__{case_id}"
        if entry_id in seen:
            raise RuntimeError(f"Duplicate inventory entry: {entry_id}")
        seen.add(entry_id)
        video_path = gallery_root / str(record["video"]).lstrip("/")
        if not valid_video(video_path):
            missing.append(
                {
                    "kind": kind,
                    "case_id": case_id,
                    "model": model,
                    "seed": seed,
                    "variant": variant,
                    "path": str(video_path),
                }
            )
            continue
        entries.append(
            {
                "entry_id": entry_id,
                "kind": "generated",
                "family": family,
                "case_id": case_id,
                "model": model,
                "seed": seed,
                "variant": variant,
                "subset_id": str(record["subset_id"]),
                "subtype": subtype,
                "depth_stratum": depth_stratum,
                "head_count": int(record.get("k", 0)),
                "denoise_step_range": None if family == "baseline" else [start, end],
                "source": fingerprint(video_path),
                "region_cache": region_by_case[case_id],
            }
        )

    expected = {
        "gt": len(cases),
        "baseline": len(cases) * len(MODELS) * len(DEPTH_SEEDS),
        "s_feature": (
            len(cases) * len(MODELS) * len(FEATURE_SUBTYPES) * len(FEATURE_STAGES)
        ),
        "s_depth": (
            len(cases)
            * len(MODELS)
            * len(depth_subsets)
            * len(DEPTH_STAGES)
            * len(DEPTH_SEEDS)
        ),
    }
    actual = {
        family: sum(entry["family"] == family for entry in entries)
        for family in expected
    }
    if not args.allow_incomplete and (actual != expected or missing):
        raise RuntimeError(
            "Inventory is incomplete: "
            f"actual={actual} expected={expected} missing={len(missing)}"
        )
    output = {
        "schema_version": 2,
        "manifest": str(manifest_path),
        "cases": list(cases.values()),
        "models": list(MODELS),
        "feature_subtypes": list(FEATURE_SUBTYPES),
        "feature_stages": [list(stage) for stage in FEATURE_STAGES],
        "depth_stages": [list(stage) for stage in DEPTH_STAGES],
        "depth_seeds": list(DEPTH_SEEDS),
        "depth_subsets": depth_subsets,
        "entries": entries,
        "missing": missing,
        "counts": {"actual": actual, "expected": expected, "total": len(entries)},
    }
    output_path = args.output_root / "inventory.json"
    atomic_write_json(output_path, output)
    print(
        f"[s-motion-inventory] total={len(entries)} actual={actual} "
        f"expected={expected} missing={len(missing)}"
    )
    print(output_path)


if __name__ == "__main__":
    main()
