#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed_sweep_case001460"
)
CURRENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
)
BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed_sweep_metrics_case001460"
)
INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json"
)
CASE_KEY = INPUT_JSON.stem
CRITERIA = (
    "strict_score", "allblock_purity", "allblock_min_purity", "balanced",
    "uniformity", "joint", "mass", "pck32",
)
STAGES = ("all_steps", "steps00_09")
PROFILES = (
    "alpha090", "alpha150", "zero", "uniform", "temporal_causal",
    "strict_past", "strict_future", "head_output_zero",
)
GROUPS = ("top100", "bottom100")

os.environ["ATTENTION_SEED_SWEEP_SOURCE_ROOT"] = str(SOURCE_ROOT)
os.environ["ATTENTION_SEED_SWEEP_BENCH_ROOT"] = str(BENCH_ROOT)
os.environ["ATTENTION_SEED_SWEEP_INPUT_JSON"] = str(INPUT_JSON)
os.environ["ATTENTION_SEED_SWEEP_CASE_KEY"] = CASE_KEY
spec = importlib.util.spec_from_file_location(
    "seed_benchmark_base", HERE / "prepare_attention_lora_seed_sweep_benchmark.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def methods():
    yield "original", {
        "criterion": "original", "stage": "original",
        "profile": "original", "group": "original",
    }
    for criterion in CRITERIA:
        for stage in STAGES:
            for profile in PROFILES:
                for group in GROUPS:
                    yield f"{criterion}__{stage}__{profile}__{group}", {
                        "criterion": criterion, "stage": stage,
                        "profile": profile, "group": group,
                    }


def source_video(seed: int, item: dict[str, str]):
    if item["stage"] == "original":
        seed_root = CURRENT_ROOT / "seeds" / f"seed_{seed:06d}"
        return seed_root / "original.mp4", seed_root / "all_steps" / "alpha090" / "complete"
    run_root = (
        SOURCE_ROOT / "seeds" / f"seed_{seed:06d}" / item["criterion"]
        / item["stage"] / item["profile"]
    )
    suffix = "steps_00_40" if item["stage"] == "all_steps" else "steps_00_10"
    video = run_root / "videos" / "lora" / "cases" / CASE_KEY / f"{item['group']}_{suffix}.mp4"
    return video, run_root / "complete"


base.methods = methods
base.source_video = source_video

if __name__ == "__main__":
    base.main()
