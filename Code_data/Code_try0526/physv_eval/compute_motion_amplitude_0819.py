#!/usr/bin/env python3
"""Compute a physics-grounded within-group motion amplitude for PhysV cases."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_radius(actor: dict[str, Any]) -> float:
    size = actor.get("size_m", {})
    if actor.get("shape") == "sphere":
        return float(size.get("radius", 0.0))
    dimensions = [float(value) for value in size.values() if isinstance(value, (int, float))]
    return math.sqrt(sum(value * value for value in dimensions)) if dimensions else 0.0


def quaternion_path(rotations: np.ndarray) -> float:
    normalized = rotations / np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-12)
    dots = np.abs(np.sum(normalized[:-1] * normalized[1:], axis=1))
    return float(np.sum(2.0 * np.arccos(np.clip(dots, 0.0, 1.0))))


def case_metric(sample_dir: Path) -> dict[str, Any]:
    metadata = load_json(sample_dir / "metadata.json")
    # Use the exported per-object trajectory truth directly. Fixed scene geometry
    # remains in the file but is excluded by its non-dynamic role.
    arrays = np.load(sample_dir / "raw/trajectories.npz", allow_pickle=False)
    names = [str(value) for value in arrays["object_names"]]
    roles = [str(value) for value in arrays["object_roles"]]
    actors = metadata.get("actors", {})
    objects: list[dict[str, Any]] = []
    total_rigid_motion = 0.0
    for index, (name, role) in enumerate(zip(names, roles)):
        if not role.startswith("dynamic"):
            continue
        position_key = f"{name}_positions"
        rotation_key = f"{name}_rotations"
        positions = np.asarray(arrays[position_key], dtype=np.float64)
        trajectory_length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        rotations = np.asarray(arrays[rotation_key], dtype=np.float64)
        rotation_path = quaternion_path(rotations)
        radius = effective_radius(actors.get(name, {}))
        rotation_equivalent = radius * rotation_path
        rigid_motion = trajectory_length + rotation_equivalent
        total_rigid_motion += rigid_motion
        objects.append(
            {
                "object_name": name,
                "trajectory_length_m": trajectory_length,
                "rotation_path_rad": rotation_path,
                "effective_radius_m": radius,
                "rotation_equivalent_m": rotation_equivalent,
                "rigid_motion_length_m": rigid_motion,
            }
        )
    for item in objects:
        item["motion_contribution"] = (
            item["rigid_motion_length_m"] / total_rigid_motion if total_rigid_motion > 1e-12 else 0.0
        )
    return {
        "case_id": sample_dir.name,
        "source_group": metadata.get("source_group", ""),
        "family_key": metadata.get("family_key", ""),
        "control": metadata.get("control", {}),
        "objects": objects,
        "dynamic_object_count": len(objects),
        "total_rigid_motion_length_m": total_rigid_motion,
        "motion_amplitude_m": total_rigid_motion,
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output = (args.output_json or root / "reports/motion_amplitude.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = [case_metric(path) for path in sorted((root / "samples").iterdir()) if path.is_dir()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault(case["family_key"] or case["source_group"], []).append(case)
    group_summary: dict[str, Any] = {}
    for group, members in groups.items():
        values = np.asarray([member["motion_amplitude_m"] for member in members], dtype=np.float64)
        order = np.argsort(-values, kind="stable")
        minimum = float(values.min())
        maximum = float(values.max())
        median = float(np.median(values))
        for rank, index in enumerate(order, start=1):
            member = members[int(index)]
            value = float(values[int(index)])
            member["group_comparison"] = {
                "group": group,
                "rank_descending": rank,
                "case_count": len(members),
                "delta_from_group_min_m": value - minimum,
                "delta_from_group_min_pct": (value / minimum - 1.0) * 100.0 if minimum > 1e-12 else None,
                "relative_to_group_median": value / median if median > 1e-12 else None,
            }
        group_summary[group] = {
            "case_count": len(members),
            "min_m": minimum,
            "max_m": maximum,
            "range_m": maximum - minimum,
            "median_m": median,
            "coefficient_of_variation": float(values.std() / median) if median > 1e-12 else None,
            "case_ids_descending": [members[int(index)]["case_id"] for index in order],
        }
    report = {
        "schema_version": "physv_motion_amplitude_v3",
        "metric": {
            "name": "total_rigid_motion_length",
            "label_zh": "场景总刚体运动长度",
            "unit": "m",
            "formula": "sum_i [sum_t ||p_i(t+1)-p_i(t)||_2 + r_i * sum_t quaternion_angle(q_i(t), q_i(t+1))]",
            "interpretation": "把所有动态物体的平移轨迹和按有效半径折算的旋转轨迹直接相加；数值越大表示场景累计刚体运动幅度越大，不对物体求平均。",
        },
        "dataset_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "groups": group_summary,
        "cases": {case["case_id"]: case for case in cases},
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")
    print(f"cases={len(cases)} groups={len(groups)}")
    for group, summary in group_summary.items():
        print(f"{group}: range={summary['range_m']:.4f} m, cv={summary['coefficient_of_variation']:.4f}")


if __name__ == "__main__":
    main()
