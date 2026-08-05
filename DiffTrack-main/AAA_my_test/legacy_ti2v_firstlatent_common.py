#!/usr/bin/env python3
"""Shared configuration for legacy Wan2.2 TI2V first-latent PCK experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50"
)
REGION_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_regions_704x1280"
)
LEGACY_VIDEO_ROOT = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/basemodel/wan2p2_ti2v5B_frame49"
)
SEEDS_FILE = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds.txt"
)
WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


@dataclass(frozen=True)
class CaseSpec:
    key: str
    json_path: Path
    old_region_cache: Path


_JSON_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
_OLD_CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/test100_51_grounded_sam2_regions"
)
CASES = (
    CaseSpec(
        "0613pybullet_sample_000301_w000",
        _JSON_ROOT / "0613pybullet_sample_000301_w000.json",
        _OLD_CACHE_ROOT / "case_test100_51_000_0613pybullet_sample_000301_w000",
    ),
    CaseSpec(
        "0613pybullet_sample_000331_w001",
        _JSON_ROOT / "0613pybullet_sample_000331_w001.json",
        _OLD_CACHE_ROOT / "case_test100_51_031_0613pybullet_sample_000331_w001",
    ),
    CaseSpec(
        "0613pybullet_sample_001455_w000",
        _JSON_ROOT / "0613pybullet_sample_001455_w000.json",
        _OLD_CACHE_ROOT / "case_test100_51_038_0613pybullet_sample_001455_w000",
    ),
    CaseSpec(
        "0613pybullet_sample_000336_w001",
        _JSON_ROOT / "0613pybullet_sample_000336_w001.json",
        _OLD_CACHE_ROOT / "case_test100_51_041_0613pybullet_sample_000336_w001",
    ),
    CaseSpec(
        "0613pybullet_sample_001460_w002",
        _JSON_ROOT / "0613pybullet_sample_001460_w002.json",
        _OLD_CACHE_ROOT / "case_test100_51_048_0613pybullet_sample_001460_w002",
    ),
    CaseSpec(
        "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed",
        _JSON_ROOT / "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
        _OLD_CACHE_ROOT
        / "case_test100_51_050_physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed",
    ),
)


def read_seeds() -> list[int]:
    seeds = [int(line.strip()) for line in SEEDS_FILE.read_text().splitlines() if line.strip()]
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise ValueError(f"expected 50 unique seeds in {SEEDS_FILE}, got {len(seeds)}")
    return seeds


def read_payload(case: CaseSpec) -> dict:
    return json.loads(case.json_path.read_text(encoding="utf-8"))


def run_dir(case_key: str, seed: int) -> Path:
    return OUTPUT_ROOT / "runs" / case_key / f"seed_{int(seed):05d}"


def heatmap_dir(case_key: str, seed: int) -> Path:
    return OUTPUT_ROOT / "heatmaps" / case_key / f"seed_{int(seed):05d}"


def all_tasks() -> list[tuple[CaseSpec, int]]:
    return [(case, seed) for case in CASES for seed in read_seeds()]
