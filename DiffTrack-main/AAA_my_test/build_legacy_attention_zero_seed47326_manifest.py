#!/usr/bin/env python3
"""Build the fixed nine-case seed-47326 attention-zero manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from AAA_my_test.build_legacy_ti2v_firstlatent_physiciq67_visual_samples import (
    MANIFEST_PATH as RANKING_MANIFEST,
    matrix_records,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import object_queries
from AAA_my_test.sam2_region_query_utils import load_region_cache, region_metadata


SEED = 47326
VISUAL_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples"
)
OUTPUT_ROOT = VISUAL_ROOT / "attention_zero_seed47326"
MANIFEST_PATH = OUTPUT_ROOT / "cases.json"
OLD_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50"
)
NEW_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50"
)
OLD_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_regions_704x1280"
)
NEW_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_regions_704x1280"
)
REQUEST_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/"
    "wan22_ti2v_legacy_attention_zero_seed47326_regions_704x1280"
)
INPUT_JSONS = tuple(
    Path(path)
    for path in (
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000301_w000.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000331_w001.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001455_w000.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_000336_w001.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json",
        "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json",
    )
)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def main() -> None:
    ranking = json.loads(RANKING_MANIFEST.read_text(encoding="utf-8"))
    entries = ranking["entries"][:100]
    if len(entries) != 100:
        raise RuntimeError("frozen ranking manifest does not contain Top100")
    samples = []
    for json_path in INPUT_JSONS:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        case = json_path.stem
        cache_root = next(
            (
                root
                for root in (OLD_CACHE_ROOT, NEW_CACHE_ROOT, REQUEST_CACHE_ROOT)
                if (root / case / "complete.json").is_file()
            ),
            None,
        )
        if cache_root is None:
            raise FileNotFoundError(f"missing F00 query cache for {case}")
        cache = load_region_cache(cache_root, case)
        if int(cache.metadata.get("query_context_frame", -1)) != 0:
            raise RuntimeError(f"{case}: expected query_context_frame=0")
        _, query_regions = object_queries(cache)
        baseline = first_existing(
            [
                OLD_OUTPUT_ROOT / "runs" / case / f"seed_{SEED:05d}" / "generated.mp4",
                NEW_OUTPUT_ROOT / "runs" / case / f"seed_{SEED:05d}" / "generated.mp4",
            ]
        )
        if baseline is None:
            baseline = OUTPUT_ROOT / "baselines" / case / "generated.mp4"
        metrics = first_existing(
            [
                OLD_OUTPUT_ROOT / "runs" / case / f"seed_{SEED:05d}" / "metrics.npz",
                NEW_OUTPUT_ROOT / "runs" / case / f"seed_{SEED:05d}" / "metrics.npz",
            ]
        )
        matrices = (
            {
                "s039": matrix_records(metrics, "s039"),
                "all_steps_mean": matrix_records(metrics, "all_steps_mean"),
            }
            if metrics is not None
            else {"s039": [], "all_steps_mean": []}
        )
        samples.append(
            {
                "case": case,
                "seed": SEED,
                "category": "Requested Legacy seed47326",
                "caption": str(payload["input_caption"]),
                "input_json": str(json_path),
                "baseline_video": str(baseline),
                "query_cache_dir": str(cache_root / case),
                "ablation_root": str(OUTPUT_ROOT / "ablations"),
                "sample_group": "requested_seed47326",
                "regions": [region_metadata(region) for region, _ in query_regions],
                "matrices": matrices,
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "completed_runs_at_selection": ranking["completed_runs_at_selection"],
        "ranking_status": ranking["ranking_status"],
        "entries": entries,
        "samples": samples,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(MANIFEST_PATH)
    print(f"wrote {MANIFEST_PATH}: {len(samples)} cases")


if __name__ == "__main__":
    main()
