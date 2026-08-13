#!/usr/bin/env python3
"""Build the frozen 20-case x five-seed multi-object M1 search manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
OUTPUT_ROOT = EXPERIMENT_ROOT / "training_free_m1_multi_object_search_v1"
OUTPUT = OUTPUT_ROOT / "search_manifest.json"
HEAD_RANKING = EXPERIMENT_ROOT / "head_scopes_latest3350_with_random100.json"
SOURCE_TUBE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2/gt_tubes"
)
SEEDS = (13248, 32466, 47326, 68613, 90094)
PAG_SCALES = (-1.0, -0.5, 0.5, 1.0)
WINDOWS = ((0, 4), (0, 9), (0, 19), (0, 39))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def baseline_candidates(case: str, seed: int) -> list[Path]:
    seed_dir = f"seed_{seed:05d}"
    return [
        Path(
            "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/"
            f"runs/{case}/{seed_dir}/generated.mp4"
        ),
        Path(
            "/data/gaoya/agent-data/outputs/"
            "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/"
            f"runs/{case}/{seed_dir}/generated.mp4"
        ),
        Path(
            "/data/gaoya/agent-data/outputs/"
            "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
            "attention_zero_seed47326/multicase_multiseed_baselines/"
            f"{case}/{seed_dir}/generated.mp4"
        ),
        Path(
            "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
            "latest3350_top100_cotracker_sam2_v2/generations/"
            f"{case}/{seed_dir}/baseline/generated.mp4"
        ),
        OUTPUT_ROOT / "baselines" / case / seed_dir / "generated.mp4",
    ]


def select_baseline(case: str, seed: int) -> tuple[Path, bool]:
    candidates = baseline_candidates(case, seed)
    existing = next((path for path in candidates if path.is_file()), None)
    return (existing, True) if existing is not None else (candidates[-1], False)


def main() -> None:
    ranking = read_json(HEAD_RANKING)
    entries = list(ranking.get("entries") or [])
    if len(entries) != 720:
        raise RuntimeError(f"expected 720 ranked heads, got {len(entries)}")

    json_paths = [
        Path(line.strip())
        for line in INPUT_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(json_paths) != 20 or len({path.stem for path in json_paths}) != 20:
        raise RuntimeError(f"expected 20 unique cases, got {len(json_paths)}")

    samples: list[dict[str, Any]] = []
    existing_baselines = 0
    for json_path in json_paths:
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
        case = json_path.stem
        source_payload = read_json(json_path)
        source_video = Path(str(source_payload["source_video"])).expanduser().resolve()
        if not source_video.is_file():
            raise FileNotFoundError(source_video)

        tube_path = SOURCE_TUBE_ROOT / case / "tube.npz"
        tube_manifest_path = SOURCE_TUBE_ROOT / case / "manifest.json"
        tube_complete = SOURCE_TUBE_ROOT / case / "complete.json"
        if not all(path.is_file() for path in (tube_path, tube_manifest_path, tube_complete)):
            raise FileNotFoundError(f"{case}: incomplete frozen source tube")
        tube_manifest = read_json(tube_manifest_path)
        objects = list(tube_manifest.get("objects") or [])
        if not objects:
            raise RuntimeError(f"{case}: source tube contains no objects")

        for seed in SEEDS:
            baseline, reused = select_baseline(case, seed)
            existing_baselines += int(reused)
            samples.append(
                {
                    "case": case,
                    "seed": seed,
                    "input_json": str(json_path),
                    "source_video": str(source_video),
                    "caption": str(source_payload["input_caption"]),
                    "source_query_tube": str(tube_path),
                    "source_query_tube_manifest": str(tube_manifest_path),
                    "baseline_video": str(baseline),
                    "baseline_reused_at_manifest_build": reused,
                    "regions": [
                        {
                            "region_name": str(item["name"]),
                            "region_type": "object",
                            "region_phrase": str(item.get("phrase") or item["name"]),
                            "point_start": int(item["point_start"]),
                            "point_end": int(item["point_end"]),
                        }
                        for item in objects
                    ],
                }
            )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "training_free_m1_multi_object_blockdiag_search_v1",
        "input_list": str(INPUT_LIST),
        "case_count": len(json_paths),
        "seed_count": len(SEEDS),
        "sample_count": len(samples),
        "seeds": list(SEEDS),
        "excluded_seed": 35075,
        "head_ranking_path": str(HEAD_RANKING),
        "entries": entries[:100],
        "samples": samples,
        "search_grid": {
            "pag_scales": list(PAG_SCALES),
            "guidance_windows_inclusive": [list(window) for window in WINDOWS],
            "guided_videos_per_sample": len(PAG_SCALES) * len(WINDOWS),
            "guided_video_count": len(samples) * len(PAG_SCALES) * len(WINDOWS),
            "baseline_count": len(samples),
            "total_video_count_including_baselines": (
                len(samples) * (1 + len(PAG_SCALES) * len(WINDOWS))
            ),
        },
        "controlled": {
            "model": "Wan2.2-TI2V-5B Legacy DiffSynth",
            "head_scope": "latest3350 Top100",
            "flow": "M1 block-diagonal multi-object R_i->R_i",
            "preserved_cross_object_pairs": "A[R_i,R_j]V[R_j], i != j",
            "cfg_scale": 5.0,
            "sampling_steps": 40,
            "num_frames": 49,
            "height": 704,
            "width": 1280,
            "sample_shift": 5.0,
            "solver": "unipc",
        },
        "existing_baselines_at_build": existing_baselines,
        "missing_baselines_at_build": len(samples) - existing_baselines,
    }
    atomic_json(OUTPUT, payload)
    print(OUTPUT)
    print(
        f"cases={len(json_paths)} samples={len(samples)} "
        f"existing_baselines={existing_baselines} "
        f"missing_baselines={len(samples) - existing_baselines} "
        f"guided={payload['search_grid']['guided_video_count']}"
    )


if __name__ == "__main__":
    main()
