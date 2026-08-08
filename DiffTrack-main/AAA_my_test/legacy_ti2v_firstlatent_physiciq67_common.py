#!/usr/bin/env python3
"""Shared configuration for PhysicIQ67 legacy Wan2.2 TI2V first-latent PCK."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


EXPERIMENT_KEY = "wan22_ti2v_legacy_firstlatent_physiciq67_pck50"
OUTPUT_ROOT = Path(f"/data/gaoya/agent-data/outputs/{EXPERIMENT_KEY}")
REGION_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_regions_704x1280"
)
LEGACY_VIDEO_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/"
    "physicIQ/basemodel/wan2p2_ti2v5B_aligned49_steps40_512x896_49f_defaultnegprompt"
)
INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt")
SEEDS_FILE = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds.txt"
)
WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
CASE_MANIFEST = OUTPUT_ROOT / "physiciq67_cases.json"
TASKS_JSONL = OUTPUT_ROOT / "missing_tasks.jsonl"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    json_path: Path
    formal_video_path: Path
    formal_json_path: Path
    source_video: Path
    category: str

    def to_json(self) -> dict:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def read_seeds() -> list[int]:
    seeds = [int(line.strip()) for line in SEEDS_FILE.read_text().splitlines() if line.strip()]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise ValueError(f"expected 50 unique seeds in {SEEDS_FILE}, got {len(seeds)}")
    return seeds


def read_payload(case: CaseSpec) -> dict:
    return json.loads(case.json_path.read_text(encoding="utf-8"))


def _category_from_key(case_key: str) -> str:
    match = re.match(
        r"^physicIQ_(?:\d{3}_)?"
        r"(Fluid_Dynamics|Solid_Mechanics|Thermodynamics|Magnetism|Optics)_",
        case_key,
    )
    return match.group(1) if match else "unknown"


def _load_cases() -> tuple[CaseSpec, ...]:
    cases: list[CaseSpec] = []
    seen: set[str] = set()
    for line in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        json_path = Path(line)
        case_key = json_path.stem
        if case_key in seen:
            raise ValueError(f"duplicate case in {INPUT_LIST}: {case_key}")
        seen.add(case_key)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_video = Path(str(payload["source_video"]))
        formal_video_path = LEGACY_VIDEO_ROOT / f"{case_key}.mp4"
        formal_json_path = LEGACY_VIDEO_ROOT / f"{case_key}.json"
        missing = [
            str(path)
            for path in (json_path, source_video, formal_video_path, formal_json_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(f"{case_key}: missing required files: {missing}")
        cases.append(
            CaseSpec(
                key=case_key,
                json_path=json_path,
                formal_video_path=formal_video_path,
                formal_json_path=formal_json_path,
                source_video=source_video,
                category=_category_from_key(case_key),
            )
        )
    if len(cases) != 67:
        raise ValueError(f"expected 67 PhysicIQ cases in {INPUT_LIST}, got {len(cases)}")
    return tuple(cases)


CASES = _load_cases()


def run_dir(case_key: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / case_key / f"seed_{int(seed):05d}"


def heatmap_dir(case_key: str, seed: int) -> Path:
    return OUTPUT_ROOT / "heatmaps" / case_key / f"seed_{int(seed):05d}"


def all_tasks() -> list[tuple[CaseSpec, int]]:
    seeds = read_seeds()
    return [(case, seed) for case in CASES for seed in seeds]


def write_case_manifest() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_key": EXPERIMENT_KEY,
        "case_count": len(CASES),
        "seed_count": len(read_seeds()),
        "expected_runs": len(CASES) * len(read_seeds()),
        "input_list": str(INPUT_LIST),
        "formal_compare_root": str(LEGACY_VIDEO_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "region_cache_root": str(REGION_CACHE_ROOT),
        "cases": [case.to_json() for case in CASES],
    }
    CASE_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
