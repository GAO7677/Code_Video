#!/usr/bin/env python3
"""Select reproducible PhysicIQ67 runs and build lightweight viewer assets."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import (
    CASES,
    OUTPUT_ROOT,
    REGION_CACHE_ROOT,
    run_dir,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import object_queries
from AAA_my_test.sam2_region_query_utils import (
    load_region_cache,
    region_metadata,
    save_region_query_visualizations,
)


VISUAL_ROOT = OUTPUT_ROOT / "visual_samples"
MANIFEST_PATH = VISUAL_ROOT / "samples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--selection-seed", type=int, default=20260808)
    parser.add_argument("--refresh-selection", action="store_true")
    return parser.parse_args()


def completed_runs() -> dict[str, list[int]]:
    completed: dict[str, list[int]] = {}
    for path in sorted((OUTPUT_ROOT / "runs").glob("*/seed_*/complete.json")):
        run = path.parent
        if not (run / "metrics.npz").is_file() or not (run / "generated.mp4").is_file():
            continue
        completed.setdefault(run.parent.name, []).append(int(run.name.removeprefix("seed_")))
    return completed


def choose_samples(available: dict[str, list[int]], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cases = sorted(available)
    rng.shuffle(cases)
    pools = {case: list(seeds) for case, seeds in available.items()}
    for seeds in pools.values():
        rng.shuffle(seeds)
    selected: list[dict] = []
    while len(selected) < count and any(pools.values()):
        for case in cases:
            if pools[case] and len(selected) < count:
                selected.append({"case": case, "seed": pools[case].pop()})
    return selected


def provisional_s039_top_heads(limit: int = 100) -> tuple[list[dict], int]:
    ranking_path = OUTPUT_ROOT / "aggregate" / "ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    rows = [row for row in ranking["global_step_block_head"] if int(row["step"]) == 39]
    rows.sort(
        key=lambda row: (
            row.get("pck32") is None,
            -float(row.get("pck32") or -1.0),
            float(row.get("mean_error_px") or float("inf")),
            int(row["block"]),
            int(row["head"]),
        )
    )
    return rows[:limit], int(ranking.get("completed_runs", 0))


def matrix_records(metrics_path: Path, kind: str) -> list[dict]:
    with np.load(metrics_path) as arrays:
        correct = arrays["correct32"].astype(np.float64)
        comparisons = arrays["comparisons"].astype(np.float64)
        error_sum = arrays["error_sum"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_step_pck = np.where(comparisons > 0, 100.0 * correct / comparisons, np.nan)
        per_step_error = np.where(comparisons > 0, error_sum / comparisons, np.nan)
    if kind == "s039":
        pck, error, counts = per_step_pck[39], per_step_error[39], comparisons[39]
    elif kind == "all_steps_mean":
        pck = np.nanmean(per_step_pck, axis=0)
        error = np.nanmean(per_step_error, axis=0)
        counts = comparisons.sum(axis=0)
    else:
        raise ValueError(kind)
    return [
        {
            "block": block,
            "head": head,
            "pck32": None if not np.isfinite(pck[block, head]) else float(pck[block, head]),
            "mean_error_px": (
                None if not np.isfinite(error[block, head]) else float(error[block, head])
            ),
            "comparisons": int(counts[block, head]),
        }
        for block in range(30)
        for head in range(24)
    ]


def existing_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_selection(payload: dict) -> list[dict]:
    return [
        {"case": str(row["case"]), "seed": int(row["seed"])}
        for row in payload.get("samples", [])
    ]


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    available = completed_runs()
    previous = {} if args.refresh_selection else existing_manifest()
    selected = manifest_selection(previous)
    selected = [
        row for row in selected
        if int(row["seed"]) in available.get(str(row["case"]), [])
    ]
    if not selected:
        selected = choose_samples(available, args.count, args.selection_seed)
    if not selected:
        raise RuntimeError("No completed PhysicIQ67 runs are available")

    case_lookup = {case.key: case for case in CASES}
    if selected and len(previous.get("entries", [])) >= 100:
        entries = previous["entries"]
        completed_snapshot = int(previous.get("completed_runs_at_selection", 0))
    else:
        entries, completed_snapshot = provisional_s039_top_heads(100)
    VISUAL_ROOT.mkdir(parents=True, exist_ok=True)
    samples = []
    for selected_row in selected:
        case_key, seed = selected_row["case"], int(selected_row["seed"])
        case = case_lookup[case_key]
        cache = load_region_cache(REGION_CACHE_ROOT, case_key)
        _, query_regions = object_queries(cache)
        case_visual_root = VISUAL_ROOT / "regions" / case_key
        save_region_query_visualizations(case_visual_root, cache)
        output = run_dir(case_key, seed)
        samples.append(
            {
                "case": case_key,
                "seed": seed,
                "category": case.category,
                "caption": str(json.loads(case.json_path.read_text(encoding="utf-8"))["input_caption"]),
                "regions": [region_metadata(region) for region, _ in query_regions],
                "matrices": {
                    "s039": matrix_records(output / "metrics.npz", "s039"),
                    "all_steps_mean": matrix_records(output / "metrics.npz", "all_steps_mean"),
                },
            }
        )
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_seed": int(previous.get("selection_seed", args.selection_seed)),
        "requested_count": int(previous.get("requested_count", args.count)),
        "completed_runs_at_selection": completed_snapshot,
        "ranking_status": "provisional until 3350/3350 runs complete",
        "protocol": {
            "model": "Wan2.2 TI2V 5B Legacy DiffSynth",
            "resolution": [704, 1280],
            "query_pixel_frame": 0,
            "query_latent_index": 0,
            "latent_anchor_pixel_frames": list(range(0, 49, 4)),
            "points_per_object": 8,
            "attention": "S039 provisional Top100; per-frame spatial softmax then mean over object query points",
        },
        "entries": entries,
        "samples": samples,
    }
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(MANIFEST_PATH)
    print(f"wrote {MANIFEST_PATH} with {len(samples)} samples")


if __name__ == "__main__":
    main()
