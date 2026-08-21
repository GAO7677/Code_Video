"""Audit PhysV pilot geometry before rendering artifacts.

The audit checks every object pair and object-ground pair after creation,
after pre-roll, and at the actual first recorded video frame.  It intentionally
does not render images, so it can reject invalid cases before a batch consumes
render time or produces misleading training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generate_difficulty_pilot import (
    DIFFICULTY_LEVELS,
    RAMP_INCLINE_CASES,
    RAMP_LENGTH_CONTROL_CASES,
    TABLE_ROLLOFF_CASES,
)
from .generate_v2v_context_demos import (
    audit_v2v_case_initialization,
    build_demo_cases,
)
from .render_sim_0705 import audit_blueprint_initialization
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
    generate_scenario_blueprint,
)


DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/physv_qa_fixed_20260819/reports/initialization_preflight.json"
)


def _difficulty_cases(seed_base: int, per_level: int):
    case_index = 0
    for level_key, level in DIFFICULTY_LEVELS.items():
        families = list(level["families"])
        for local_index in range(per_level):
            family_key = str(families[local_index % len(families)])
            case_id = f"difficulty_{level_key.lower()}_{family_key.lower()}_{local_index:03d}"
            seed = int(seed_base + case_index * 1009)
            case_index += 1
            yield case_id, family_key, seed, generate_scenario_blueprint(
                family_key=family_key,
                sample_key=case_id,
                seed=seed,
                size_scale=1.0,
                camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            )

    for extra in TABLE_ROLLOFF_CASES:
        table_height_m = float(extra["table_height_m"])
        case_id = (
            f"difficulty_l2_f11_h{int(round(table_height_m * 100)):03d}_"
            f"{extra['angle_label']}"
        )
        seed = int(seed_base + 88000)
        yield case_id, "F11", seed, generate_scenario_blueprint(
            family_key="F11",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            table_height_m=table_height_m,
            initial_speed_mps=1.25,
            travel_angle_deg=float(extra.get("travel_angle_deg", F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG)),
        )

    for extra in RAMP_INCLINE_CASES:
        case_id = f"difficulty_l2_f12_{extra['angle_label']}"
        seed = int(seed_base + 99000)
        yield case_id, "F12", seed, generate_scenario_blueprint(
            family_key="F12",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            ramp_angle_deg=float(extra["ramp_angle_deg"]),
        )

    for extra in RAMP_LENGTH_CONTROL_CASES:
        case_id = f"difficulty_l2_f12_length_{extra['length_label']}_a024"
        seed = int(seed_base + 99100)
        yield case_id, "F12_RAMP_LENGTH", seed, generate_scenario_blueprint(
            family_key="F12",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            ramp_angle_deg=float(extra["ramp_angle_deg"]),
            ramp_length_m=float(extra["ramp_length_m"]),
        )


def audit_suite(
    *,
    difficulty_seed_base: int = 20260817,
    v2v_seed_base: int = 20260819,
    per_level: int = 4,
    include_difficulty: bool = True,
    include_v2v: bool = True,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if include_difficulty:
        for case_id, family_key, seed, blueprint in _difficulty_cases(
            difficulty_seed_base,
            per_level,
        ):
            try:
                qa = audit_blueprint_initialization(blueprint=blueprint, seed=seed)
                rows.append(
                    {
                        "case_id": case_id,
                        "suite": "difficulty",
                        "family_key": family_key,
                        "seed": seed,
                        "passed": True,
                        "initialization_qa": qa,
                    }
                )
            except Exception as exc:  # Keep auditing the remaining cases.
                rows.append(
                    {
                        "case_id": case_id,
                        "suite": "difficulty",
                        "family_key": family_key,
                        "seed": seed,
                        "passed": False,
                        "error": repr(exc),
                    }
                )
    if include_v2v:
        for index, case in enumerate(build_demo_cases(v2v_seed_base)):
            seed = int(v2v_seed_base + index * 1009)
            try:
                qa = audit_v2v_case_initialization(case, seed=seed)
                rows.append(
                    {
                        "case_id": case.case_id,
                        "suite": "v2v",
                        "family_key": case.family_key,
                        "seed": seed,
                        "passed": True,
                        "initialization_qa": qa,
                    }
                )
            except Exception as exc:  # Keep auditing the remaining cases.
                rows.append(
                    {
                        "case_id": case.case_id,
                        "suite": "v2v",
                        "family_key": case.family_key,
                        "seed": seed,
                        "passed": False,
                        "error": repr(exc),
                    }
                )
    failures = [row for row in rows if not row["passed"]]
    return {
        "contract": {
            "stages": ["post_creation", "post_pre_roll", "video_frame_0"],
            "penetration_tolerance_m": 0.001,
            "contact_policy": "intended contact is allowed; overlap deeper than 1 mm is rejected",
        },
        "total_cases": len(rows),
        "passed_cases": len(rows) - len(failures),
        "failed_cases": len(failures),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--difficulty-seed-base", type=int, default=20260817)
    parser.add_argument("--v2v-seed-base", type=int, default=20260819)
    parser.add_argument("--per-level", type=int, default=4)
    parser.add_argument("--difficulty-only", action="store_true")
    parser.add_argument("--v2v-only", action="store_true")
    args = parser.parse_args()
    if args.per_level <= 0:
        raise ValueError("--per-level must be positive")
    if args.difficulty_only and args.v2v_only:
        raise ValueError("--difficulty-only and --v2v-only cannot be combined")

    report = audit_suite(
        difficulty_seed_base=args.difficulty_seed_base,
        v2v_seed_base=args.v2v_seed_base,
        per_level=args.per_level,
        include_difficulty=not args.v2v_only,
        include_v2v=not args.difficulty_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total_cases": report["total_cases"],
                "passed_cases": report["passed_cases"],
                "failed_cases": report["failed_cases"],
            },
            ensure_ascii=False,
        )
    )
    if report["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
