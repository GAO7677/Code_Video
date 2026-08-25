"""Generate a small RigidBench-inspired support-and-motion demo set.

The pilot keeps the scene mechanism explicit and varies one controlled
quantity within each family:

* desk/table roll-off: table height;
* incline/ramp release: ramp angle;
* pendulum: pendulum length.

The simulator and fast preview renderer are reused from the existing PhysV
V2V pipeline.  Artifacts are written under ``/data/gaoya/agent-data`` so this
pilot remains independent of the exported training/evaluation dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .generate_v2v_context_demos import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    DemoCase,
    _make_pendulum_case,
    _render_case,
)
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
    generate_scenario_blueprint,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/datasets/rigidbench_style_support_motion_demo_20260825"
)
SEED_BASE = 20260825


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value


def _support_case(
    *,
    case_id: str,
    family_key: str,
    title: str,
    description: str,
    controlled_variable: str,
    controlled_value: float,
    value_label: str,
    event_rule: str,
    blueprint,
) -> DemoCase:
    family_title = {
        "F11": "桌面滚落",
        "F12": "斜面释放",
    }[family_key]
    family_description = {
        "F11": "固定桌面与滚动物体，改变桌面高度，观察越过桌缘后的自由落体和落地反弹。",
        "F12": "固定木块和动态支撑斜面，改变坡角，观察重力驱动的滑动、翻滚和离开斜面。",
    }[family_key]
    return DemoCase(
        case_id=case_id,
        family_key=family_key,
        family_title=family_title,
        family_description=family_description,
        level="L2",
        title=title,
        description=description,
        controlled_variable=controlled_variable,
        controlled_value=float(controlled_value),
        controlled_value_label=value_label,
        units="m" if controlled_variable == "table_height_m" else "deg",
        blueprint=blueprint,
        event_rule=event_rule,
    )


def build_cases(seed_base: int = SEED_BASE) -> list[tuple[DemoCase, int, str]]:
    """Return six cases and their deterministic seeds.

    The ``reference_family`` field is recorded separately from the local
    family key so later viewers can distinguish the RigidBench design cue from
    the actual PhysV taxonomy used by the simulator.
    """

    cases: list[tuple[DemoCase, int, str]] = []

    table_specs = (
        ("rbstyle_desk_rolloff_h055", 0.55),
        ("rbstyle_desk_rolloff_h110", 1.10),
    )
    for index, (case_id, table_height) in enumerate(table_specs):
        seed = seed_base + index * 1009
        blueprint = generate_scenario_blueprint(
            family_key="F11",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            table_height_m=table_height,
            initial_speed_mps=1.25,
            travel_angle_deg=F11_SCREEN_RIGHT_TRAVEL_ANGLE_DEG,
        )
        cases.append(
            (
                _support_case(
                    case_id=case_id,
                    family_key="F11",
                    title=f"桌面滚落：桌高 {table_height:.2f} m",
                    description=(
                        "一颗红色滚动物体沿固定桌面运动并越过桌缘；"
                        "只改变桌面高度，后续自由落体和落地时刻随高度变化。"
                    ),
                    controlled_variable="table_height_m",
                    controlled_value=table_height,
                    value_label=f"{table_height:.2f} m",
                    event_rule="roller_crosses_table_edge_then_hits_floor",
                    blueprint=blueprint,
                ),
                seed,
                "RigidBench-style Desk/Table support and roll-off",
            )
        )

    ramp_specs = (
        ("rbstyle_incline_release_a12", 12.0),
        ("rbstyle_incline_release_a32", 32.0),
    )
    for offset, (case_id, ramp_angle) in enumerate(ramp_specs, start=2):
        seed = seed_base + offset * 1009
        blueprint = generate_scenario_blueprint(
            family_key="F12",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            ramp_angle_deg=ramp_angle,
        )
        cases.append(
            (
                _support_case(
                    case_id=case_id,
                    family_key="F12",
                    title=f"斜面释放：坡角 {ramp_angle:.0f}°",
                    description=(
                        "一个红色木块从动态支撑斜面上静止释放；"
                        "只改变斜面坡角，比较滑动加速、离开斜面和末态。"
                    ),
                    controlled_variable="ramp_angle_deg",
                    controlled_value=ramp_angle,
                    value_label=f"{ramp_angle:.0f}°",
                    event_rule="block_slides_down_incline_then_exits_ramp",
                    blueprint=blueprint,
                ),
                seed,
                "RigidBench-style ramp_collision / inclined support",
            )
        )

    pendulum_specs = (
        ("rbstyle_pendulum_length_l080", 0.80),
        ("rbstyle_pendulum_length_l150", 1.50),
    )
    for offset, (case_id, length) in enumerate(pendulum_specs, start=4):
        seed = seed_base + offset * 1009
        cases.append(
            (
                _make_pendulum_case(case_id, length),
                seed,
                "RigidBench-inspired constrained pendulum support",
            )
        )

    return cases


def generate_demo_set(
    *,
    output_root: Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    seed_base: int = SEED_BASE,
    overwrite: bool = False,
) -> dict[str, object]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    reference = {
        "source": "/data/gaoya/dataset/rigidbench-eval-v1.1/",
        "relevant_tasks": ["ramp_collision", "bounce", "free_fall"],
        "relevant_surfaces": ["Desk", "Table.002", "Table.005"],
        "note": "RigidBench is used as a scene-design reference; dynamics are regenerated by the local PhysV PyBullet pipeline.",
    }
    (output_root / "rigidbench_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for index, (case, seed, reference_family) in enumerate(build_cases(seed_base)):
        case_root = output_root / "artifacts" / case.family_key / case.case_id
        try:
            row = _render_case(
                case,
                seed=seed,
                output_root=case_root,
                width=width,
                height=height,
            )
            row.update(
                {
                    "pilot_index": index,
                    "family_key": case.family_key,
                    "case_title": case.title,
                    "controlled_variable": case.controlled_variable,
                    "controlled_value": case.controlled_value,
                    "reference_family": reference_family,
                    "reference_source": reference["source"],
                    "width": width,
                    "height": height,
                }
            )
            rows.append(row)
            print(f"rendered {case.case_id}", flush=True)
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append(
                {
                    "case_id": case.case_id,
                    "family_key": case.family_key,
                    "seed": seed,
                    "error": repr(exc),
                }
            )
            print(f"failed {case.case_id}: {exc!r}", flush=True)

    (output_root / "cases.jsonl").write_text(
        "".join(json.dumps(_json_safe(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_root / "reports").mkdir(parents=True, exist_ok=True)
    (output_root / "reports" / "failure_report.json").write_text(
        json.dumps(_json_safe(failures), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "requested": len(build_cases(seed_base)),
        "rendered": len(rows),
        "failed": len(failures),
        "families": {
            family: sum(row.get("family_key") == family for row in rows)
            for family in ("F11", "F12", "V2V_PENDULUM")
        },
        "reference": reference,
    }
    (output_root / "reports" / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = generate_demo_set(
        output_root=args.output_root,
        width=args.width,
        height=args.height,
        seed_base=args.seed_base,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
