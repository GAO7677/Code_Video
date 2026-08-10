#!/usr/bin/env python3
"""Build the strict six-seed manifest for the ten non-pilot legacy cases."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from AAA_my_test.build_legacy_ti2v_firstlatent_physiciq67_visual_samples import (
    provisional_s039_top_heads,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import object_queries
from AAA_my_test.sam2_region_query_utils import load_region_cache, region_metadata


SEEDS = (13248, 32466, 35075, 47326, 68613, 90094)
OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
SOURCE_MANIFEST = OUTPUT_BASE / "cases.json"
OUTPUT_MANIFEST = OUTPUT_BASE / "cases_other10_6seeds.json"
NEW_BASELINE_ROOT = OUTPUT_BASE / "multicase_multiseed_baselines"
OLD_RUN_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/runs"
)
PHYSICIQ67_RUN_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/runs"
)
PHYSICIQ67_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_regions_704x1280"
)
JSON_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
INPUT_JSONS = (
    JSON_ROOT / "0613pybullet_sample_000301_w000.json",
    JSON_ROOT / "0613pybullet_sample_000331_w001.json",
    JSON_ROOT / "0613pybullet_sample_000336_w001.json",
    JSON_ROOT / "0613pybullet_sample_001455_w000.json",
    JSON_ROOT
    / "physicIQ_008_Fluid_Dynamics_0128_perspective-center_trimmed-napkin-soak.json",
    JSON_ROOT
    / "physicIQ_009_Fluid_Dynamics_0131_perspective-center_trimmed-paint-on-glass.json",
    JSON_ROOT / "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
    JSON_ROOT
    / (
        "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-"
        "ball-and-block-fall_motion_to_end.json"
    ),
    JSON_ROOT
    / "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json",
    JSON_ROOT
    / (
        "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-"
        "ball-behind-rotating-paper.json"
    ),
)


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def source_states(source_video: Path) -> Path | None:
    candidates = (source_video.parent / "states.npz", source_video.parent.parent / "states.npz")
    return first_existing(list(candidates))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-ranking", action="store_true")
    parser.add_argument("--output-manifest", type=Path, default=OUTPUT_MANIFEST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if args.latest_ranking:
        entries, completed_runs = provisional_s039_top_heads(100)
        source["entries"] = entries
        source["completed_runs_at_selection"] = completed_runs
        source["ranking_status"] = (
            "final at 3350/3350 runs"
            if completed_runs >= 3350
            else "provisional until 3350/3350 runs complete"
        )
    if len(source.get("entries", [])) < 100:
        raise RuntimeError("source manifest does not contain the frozen Top100 ranking")
    source_samples = {str(row["case"]): row for row in source["samples"]}
    samples: list[dict] = []

    for json_path in INPUT_JSONS:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        case = json_path.stem
        template = source_samples.get(case)
        if template is not None:
            cache_dir = Path(str(template["query_cache_dir"]))
        else:
            cache_dir = PHYSICIQ67_CACHE_ROOT / case
        if not all((cache_dir / name).is_file() for name in ("complete.json", "regions.npz")):
            raise FileNotFoundError(f"{case}: missing query cache at {cache_dir}")
        cache = load_region_cache(cache_dir.parent, cache_dir.name)
        if int(cache.metadata.get("query_context_frame", -1)) != 0:
            raise RuntimeError(f"{case}: expected query_context_frame=0")
        _, query_regions = object_queries(cache)
        regions = [region_metadata(region) for region, _ in query_regions]
        if not regions:
            raise RuntimeError(f"{case}: no object regions")

        source_video = Path(str(payload["source_video"]))
        if not source_video.is_file():
            raise FileNotFoundError(f"{case}: missing source video {source_video}")
        states = source_states(source_video)

        for seed in SEEDS:
            baseline = first_existing(
                [
                    OLD_RUN_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
                    PHYSICIQ67_RUN_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
                    OUTPUT_BASE / "baselines" / case / "generated.mp4"
                    if seed == 47326
                    else Path("/__not_a_baseline__"),
                ]
            )
            if baseline is None:
                baseline = NEW_BASELINE_ROOT / case / f"seed_{seed:05d}" / "generated.mp4"
            samples.append(
                {
                    "case": case,
                    "seed": seed,
                    "category": "Legacy object-query ten-case six-seed replication",
                    "caption": str(payload["input_caption"]),
                    "input_json": str(json_path),
                    "source_video": str(source_video),
                    "source_states": str(states) if states is not None else None,
                    "baseline_video": str(baseline),
                    "query_cache_dir": str(cache_dir),
                    "matrix_ablation_root": str(OUTPUT_BASE / "attention_matrix_ablations_v2"),
                    "temporal_tube_ablation_root": str(
                        OUTPUT_BASE / "attention_matrix_ablations_temporal_tube_v1"
                    ),
                    "sample_group": "other10_6seeds",
                    "regions": regions,
                    "matrices": {"s039": [], "all_steps_mean": []},
                }
            )

    if len(samples) != len(INPUT_JSONS) * len(SEEDS):
        raise RuntimeError(f"expected 60 samples, got {len(samples)}")
    output = {
        **source,
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": None,
        "seeds": list(SEEDS),
        "case_count": len(INPUT_JSONS),
        "sample_count": len(samples),
        "sample_group": "other10_6seeds",
        "samples": samples,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output_manifest)
    missing_baselines = sum(
        not Path(str(sample["baseline_video"])).is_file() for sample in samples
    )
    print(f"wrote {args.output_manifest}")
    print(
        f"cases={len(INPUT_JSONS)} samples={len(samples)} seeds={len(SEEDS)} "
        f"missing_baselines={missing_baselines} entries={len(output['entries'])}"
    )


if __name__ == "__main__":
    main()
