#!/usr/bin/env python3
"""Incrementally prepare completed 50-seed videos for the 14-metric bench."""

from __future__ import annotations

import json
import os
from pathlib import Path


SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
)
BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_metrics_case001460"
)
CASE_KEY = "0613pybullet_sample_001460_w002"
INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "0613pybullet_sample_001460_w002.json"
)
PROFILES = (
    "alpha090",
    "alpha150",
    "zero",
    "uniform",
    "temporal_causal",
    "strict_past",
    "strict_future",
    "head_output_zero",
)
STAGES = ("all_steps", "steps00_09")
GROUPS = ("top100", "bottom100")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def methods():
    yield "original", {"stage": "original", "profile": "original", "group": "original"}
    for stage in STAGES:
        for profile in PROFILES:
            for group in GROUPS:
                yield f"{stage}__{profile}__{group}", {
                    "stage": stage,
                    "profile": profile,
                    "group": group,
                }


def source_video(seed: int, spec: dict[str, str]):
    seed_root = SOURCE_ROOT / "seeds" / f"seed_{seed:06d}"
    if spec["stage"] == "original":
        complete = seed_root / "all_steps" / "alpha090" / "complete"
        return seed_root / "original.mp4", complete
    suffix = "steps_00_40" if spec["stage"] == "all_steps" else "steps_00_10"
    run_root = seed_root / spec["stage"] / spec["profile"]
    video = (
        run_root
        / "videos"
        / "lora"
        / "cases"
        / CASE_KEY
        / f"{spec['group']}_{suffix}.mp4"
    )
    return video, run_root / "complete"


def main() -> None:
    seeds = [
        int(line)
        for line in (SOURCE_ROOT / "seeds.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_payload = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    prompt = str(case_payload.get("input_caption", CASE_KEY))
    methods_root = BENCH_ROOT / "methods"
    methods_root.mkdir(parents=True, exist_ok=True)
    method_paths = []
    counts = {}
    for method_index, (method_name, spec) in enumerate(methods()):
        method_root = methods_root / method_name
        method_root.mkdir(parents=True, exist_ok=True)
        method_paths.append(method_root)
        count = 0
        for seed in seeds:
            video, complete = source_video(seed, spec)
            if not complete.is_file() or not video.is_file() or not video.stat().st_size:
                continue
            stem = f"seed_{seed:06d}"
            link = method_root / f"{stem}.mp4"
            if link.is_symlink() and link.resolve() != video.resolve():
                link.unlink()
            if not link.exists():
                link.symlink_to(video)
            result_path = method_root / f"{stem}.json"
            existing = {}
            if result_path.is_file():
                try:
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
            existing.update(
                {
                    "input_json": str(INPUT_JSON),
                    "case_json": str(INPUT_JSON),
                    "output_video": str(link),
                    "source_experiment_video": str(video),
                    "case_key": CASE_KEY,
                    "prompt": prompt,
                    "caption": prompt,
                    "context_frames": 8,
                    "effective_context_frames": 8,
                    "seed": seed,
                    "method": method_name,
                    **spec,
                }
            )
            atomic_json(result_path, existing)
            count += 1
        counts[method_name] = count
        atomic_json(
            method_root / "batch_manifest.json",
            {
                "method": method_name,
                "method_index": method_index,
                "num_expected_seeds": len(seeds),
                "input_json_list_path": str(BENCH_ROOT / "input_json_allowlist.txt"),
                **spec,
            },
        )
    (BENCH_ROOT / "input_json_allowlist.txt").write_text(
        str(INPUT_JSON) + "\n", encoding="utf-8"
    )
    (BENCH_ROOT / "bench_methods.txt").write_text(
        "\n".join(str(path) for path in method_paths) + "\n", encoding="utf-8"
    )
    for gpu in range(4):
        assigned = [path for index, path in enumerate(method_paths) if index % 4 == gpu]
        (BENCH_ROOT / f"bench_methods_gpu{gpu}.txt").write_text(
            "\n".join(str(path) for path in assigned) + "\n", encoding="utf-8"
        )
    atomic_json(
        BENCH_ROOT / "prepared_status.json",
        {
            "case": CASE_KEY,
            "num_methods": len(method_paths),
            "num_seeds": len(seeds),
            "available_method_seed_videos": sum(counts.values()),
            "expected_method_seed_videos": len(method_paths) * len(seeds),
            "counts": counts,
        },
    )
    (BENCH_ROOT / "PREPARED").write_text("ready\n", encoding="utf-8")
    print(
        f"prepared {sum(counts.values())}/{len(method_paths) * len(seeds)} "
        f"method-seed videos"
    )


if __name__ == "__main__":
    main()
