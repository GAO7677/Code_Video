#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .render_sim_0705 import render_generated_case
from .scene_generators_0705 import generate_scenario_blueprint


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0713pybullet/direction_check")
VERTICAL_MOTION_TAGS = {"F5": "drop", "F8": "vertical_drop"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render paired direction checks for rigid F1-F10 cases.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed-base", type=int, default=20260718)
    return parser.parse_args()


def _seed_with_motion_tag(family_key: str, seed: int, motion_tag: str) -> int:
    for candidate in range(seed, seed + 500):
        blueprint = generate_scenario_blueprint(
            family_key,
            f"{family_key.lower()}_vertical_probe",
            candidate,
            "vertical",
        )
        if blueprint.tags[-1] == motion_tag:
            return candidate
    raise RuntimeError(f"could not find {family_key} seed with motion tag {motion_tag}")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    plans: list[tuple[str, str, int]] = []
    for family_index in range(1, 11):
        family_key = f"F{family_index}"
        pair_seed = args.seed_base + family_index * 1009
        plans.extend(
            [
                (family_key, "left_to_right", pair_seed),
                (family_key, "right_to_left", pair_seed),
            ]
        )
        if family_key in VERTICAL_MOTION_TAGS:
            vertical_seed = _seed_with_motion_tag(family_key, pair_seed, VERTICAL_MOTION_TAGS[family_key])
            plans.append((family_key, "vertical", vertical_seed))

    for index, (family_key, direction_mode, seed) in enumerate(plans, 1):
        case_id = f"direction_{family_key.lower()}_{direction_mode}"
        case_root = args.output_root / "cases" / family_key / case_id
        print(f"[{index:02d}/{len(plans):02d}] start {case_id}", flush=True)
        try:
            record = render_generated_case(
                family_key=family_key,
                sample_key=case_id,
                seed=seed,
                output_root=case_root,
                width=args.width,
                height=args.height,
                direction_mode=direction_mode,
            )
            records.append(
                {
                    "case_id": case_id,
                    "family_key": family_key,
                    "direction_mode": direction_mode,
                    "seed": seed,
                    "output_root": str(case_root),
                    "video": record["video"],
                    "meta": record["meta"],
                    "object_phrases_path": record.get("object_phrases_path", ""),
                    "caption": record.get("caption", ""),
                    "short_caption": record.get("short_caption", ""),
                    "object_nouns": record.get("object_nouns", []),
                    "object_phrases": record.get("object_phrases", []),
                    "dynamic_object_phrases": record.get("dynamic_object_phrases", []),
                    "static_object_phrases": record.get("static_object_phrases", []),
                    "negative_prompt": record.get("negative_prompt", ""),
                }
            )
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append({"case_id": case_id, "error": repr(exc)})
        print(f"[{index:02d}/{len(plans):02d}] done  {case_id}", flush=True)

    (args.output_root / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"cases": len(records), "failures": len(failures)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
