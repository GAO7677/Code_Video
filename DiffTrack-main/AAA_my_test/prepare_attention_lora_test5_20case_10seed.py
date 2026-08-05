#!/usr/bin/env python3
"""Prepare the reproducible 20-case x 10-seed attention sweep layout."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_10seed"
)
SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
)
REUSED_CASE = "0613pybullet_sample_001460_w002"
QUEUE_PATH = HERE / "attention_lora_test5_20case_10seed_queue.tsv"
NUM_SEEDS = 10
EXPECTED_PROFILES = 16


def unique_cases() -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw_line in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        case_key = path.stem
        if case_key in seen:
            continue
        seen.add(case_key)
        records.append((case_key, path))
    if len(records) != 20:
        raise RuntimeError(f"Expected 20 unique cases, found {len(records)}")
    return records


def common_seeds() -> list[int]:
    seeds = []
    for line in (SOURCE_ROOT / "seeds.txt").read_text(encoding="utf-8").splitlines():
        seed = int(line.strip())
        if seed not in seeds:
            seeds.append(seed)
        if len(seeds) == NUM_SEEDS:
            break
    if len(seeds) != NUM_SEEDS:
        raise RuntimeError(f"Expected {NUM_SEEDS} source seeds, found {len(seeds)}")
    return seeds


def source_seed_complete(seed: int) -> bool:
    seed_root = SOURCE_ROOT / "seeds" / f"seed_{seed:06d}"
    return (
        (seed_root / "original.mp4").is_file()
        and sum(1 for _ in seed_root.glob("*/*/complete")) == EXPECTED_PROFILES
    )


def prepare_case(case_key: str, input_json: Path, seeds: list[int]) -> list[int]:
    case_root = OUTPUT_ROOT / "cases" / case_key
    (case_root / "seeds").mkdir(parents=True, exist_ok=True)
    (case_root / "logs").mkdir(exist_ok=True)
    (case_root / "case_list.txt").write_text(
        f"{input_json}\n", encoding="utf-8"
    )
    (case_root / "seeds.txt").write_text(
        "".join(f"{seed}\n" for seed in seeds), encoding="utf-8"
    )
    reused = []
    if case_key == REUSED_CASE:
        for seed in seeds:
            if not source_seed_complete(seed):
                continue
            source = SOURCE_ROOT / "seeds" / f"seed_{seed:06d}"
            target = case_root / "seeds" / source.name
            if target.is_symlink() and target.resolve() != source.resolve():
                target.unlink()
            if not target.exists():
                target.symlink_to(source, target_is_directory=True)
            reused.append(seed)
    return reused


def main() -> None:
    cases = unique_cases()
    seeds = common_seeds()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "logs").mkdir(exist_ok=True)
    QUEUE_PATH.write_text(
        "".join(f"{case_key}\t{path}\n" for case_key, path in cases),
        encoding="utf-8",
    )
    reused: dict[str, list[int]] = {}
    for case_key, input_json in cases:
        reused_seeds = prepare_case(case_key, input_json, seeds)
        if reused_seeds:
            reused[case_key] = reused_seeds
    manifest = {
        "input_list": str(INPUT_LIST),
        "queue": str(QUEUE_PATH),
        "output_root": str(OUTPUT_ROOT),
        "num_unique_cases": len(cases),
        "num_seeds": len(seeds),
        "seeds": seeds,
        "profiles_per_seed": EXPECTED_PROFILES,
        "intervention_videos_per_seed": EXPECTED_PROFILES * 2,
        "expected_intervention_videos": len(cases)
        * len(seeds)
        * EXPECTED_PROFILES
        * 2,
        "reused_complete_seeds": reused,
        "cases": [
            {"case_key": case_key, "input_json": str(path)}
            for case_key, path in cases
        ],
    }
    (OUTPUT_ROOT / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
