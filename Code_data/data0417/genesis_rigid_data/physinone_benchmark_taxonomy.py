#!/usr/bin/env python3
# 用途：定义并导出 benchmark taxonomy 与统计结果。
"""Benchmark-oriented taxonomy for the PhysInOne dataset.

该脚本用于把 PhysInOne 原始物理标签整理成 benchmark 分类；输入为数据集根目录及标签映射规则，输出为终端摘要和可选 taxonomy/validation JSON 文件。

This module turns the raw PhysInOne physics tags into a smaller set of
benchmark groups so downstream evaluation can aggregate samples by motion type.

It provides:

- ``GROUPS``: ordered benchmark groups with descriptions and member types.
- ``TYPE_TO_GROUP``: direct mapping from a PhysInOne type to one benchmark group.
- helpers to parse PhysInOne sample names such as
  ``AccelPanelSpin_BalloonFloat__bg170__28WzBF_trajectory.zip``.
- optional validation against a local PhysInOne dataset root.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/vLAR-PhysInOne")


GROUPS: List[Dict[str, Any]] = [
    {
        "group_id": "A",
        "group_slug": "rigid_basic_motion",
        "group_name": "Rigid Basic Motion",
        "solver_focus": "Rigid-body dynamics with gravity, inertia, translation, rotation, projectile motion, and slope motion.",
        "types": [
            "CartMove",
            "VerticalFall",
            "ObliqueProjectile",
            "LinCarryInertia",
            "RotBoardInertia",
            "RotTurnableInertia",
            "FrictionStop",
            "RollDownSlope",
            "RollUpSlope",
        ],
    },
    {
        "group_id": "B",
        "group_slug": "rigid_contact_collision_spin",
        "group_name": "Rigid Contact Collision Spin",
        "solver_focus": "Rigid contact response, collision constraints, and geometry-induced spin.",
        "types": [
            "MovingHitsFixed",
            "MovingHitsMoving",
            "MovingHitsStationary",
            "AccelConcaveSpin",
            "AccelPanelSpin",
            "AccelSurfaceSpin",
            "UniformConcaveSpin",
            "UniformPanelSpin",
            "UniformSurfaceSpin",
        ],
    },
    {
        "group_id": "C",
        "group_slug": "articulated_constraints_pendulum",
        "group_name": "Articulated Constraints Pendulum",
        "solver_focus": "Articulated rigid bodies, pivots, coupled pendulums, and mechanical constraints.",
        "types": [
            "SimplePendulum",
            "DoublePendulum",
            "ChainSuspend",
            "SeesawCenterPivot",
            "SeesawOffsetPivot",
            "CrankPush",
            "CatapultLaunch",
        ],
    },
    {
        "group_id": "D",
        "group_slug": "deformable_elastic_plastic_support",
        "group_name": "Deformable Elastic Plastic Support",
        "solver_focus": "Elasticity, restitution, plastic dissipation, and support instability.",
        "types": [
            "ElasticCouple",
            "ElasticFall",
            "SpringCompress",
            "SpringStretch",
            "SpringboardRebound",
            "PlasticineFall",
            "StickSupportFail",
        ],
    },
    {
        "group_id": "E",
        "group_slug": "fracture_collapse_failure",
        "group_name": "Fracture Collapse Failure",
        "solver_focus": "Fracture, collapse, structural failure, and fragment propagation.",
        "types": [
            "ImpactFracture",
            "BlockWallCollapse",
            "MirrorFragmentReflect",
        ],
    },
    {
        "group_id": "F",
        "group_slug": "fluid_dominant",
        "group_name": "Fluid Dominant",
        "solver_focus": "Flowing liquid, free surfaces, transport, containers, and non-Newtonian behavior.",
        "types": [
            "DropInLiquid",
            "FloatOnLiquid",
            "JetLiquid",
            "LiquidAcrossUneven",
            "LiquidAlongContours",
            "LiquidCarryMovingObj",
            "LiquidHitFixedObj",
            "LiquidMultiTransfers",
            "LiquidRise",
            "LiquidTension",
            "LiquidThroughGrid",
            "LiquidTransfer",
            "MovingObjDriveLiquid",
            "NewtonianFluidFall",
            "NonNewtonianFluidFall",
            "LiquidRefraction",
        ],
    },
    {
        "group_id": "G",
        "group_slug": "granular_media",
        "group_name": "Granular Media",
        "solver_focus": "Discrete granular flow, accumulation, and pile formation.",
        "types": [
            "GranularFall",
        ],
    },
    {
        "group_id": "H",
        "group_slug": "adhesive_viscous_contact",
        "group_name": "Adhesive Viscous Contact",
        "solver_focus": "Adhesive contact, sticking, attachment, and detachment behavior.",
        "types": [
            "StickyFromObjects",
            "StickyToObjects",
        ],
    },
    {
        "group_id": "I",
        "group_slug": "wind_buoyancy_aerodynamics",
        "group_name": "Wind Buoyancy Aerodynamics",
        "solver_focus": "External force fields with wind, buoyancy, and simplified aerodynamic effects.",
        "types": [
            "BalloonFloat",
            "BalloonLift",
            "BalloonTether",
            "WindDeflectMotion",
            "WindGravityBalance",
            "WindPushOppDir",
            "WindPushSameDir",
            "WindPushStationary",
        ],
    },
    {
        "group_id": "J",
        "group_slug": "magnetic_force_fields",
        "group_name": "Magnetic Force Fields",
        "solver_focus": "Non-contact attraction and repulsion under external force fields.",
        "types": [
            "MagnetAttract",
            "MagnetRepel",
        ],
    },
    {
        "group_id": "K",
        "group_slug": "geometric_optics_reflection_occlusion",
        "group_name": "Geometric Optics Reflection Occlusion",
        "solver_focus": "Ray-like reflection, redirection, and occlusion-based physical reasoning.",
        "types": [
            "MirrorReflect",
            "DynMirrorRedirect",
            "FixedArrayRedirect",
            "FixedConcaveRedirect",
            "FixedConvexRedirect",
            "FixedPlanarRedirect",
            "LaserBlock",
        ],
    },
]


GROUP_ID_TO_GROUP: Dict[str, Dict[str, Any]] = {group["group_id"]: group for group in GROUPS}
TYPE_TO_GROUP: Dict[str, Dict[str, Any]] = {}
for group in GROUPS:
    for physics_type in group["types"]:
        if physics_type in TYPE_TO_GROUP:
            prev = TYPE_TO_GROUP[physics_type]["group_id"]
            raise ValueError(f"Duplicate taxonomy assignment for {physics_type}: {prev} and {group['group_id']}")
        TYPE_TO_GROUP[physics_type] = {
            "group_id": group["group_id"],
            "group_slug": group["group_slug"],
            "group_name": group["group_name"],
        }


def group_for_type(physics_type: str) -> Dict[str, Any]:
    """Return the benchmark group metadata for one PhysInOne physics type."""
    key = str(physics_type or "").strip()
    if key not in TYPE_TO_GROUP:
        raise KeyError(f"Unknown PhysInOne physics type: {physics_type!r}")
    return TYPE_TO_GROUP[key]


def normalize_sample_name(sample_name: str) -> str:
    """Normalize a PhysInOne sample name from a path, zip name, or bare stem."""
    name = Path(str(sample_name)).name
    if name.endswith(".zip"):
        name = name[:-4]
    if name.endswith("_trajectory"):
        name = name[: -len("_trajectory")]
    return name


def extract_physinone_types_from_sample_name(sample_name: str) -> List[str]:
    """Extract raw PhysInOne types from a sample name."""
    name = normalize_sample_name(sample_name)
    combo = name.split("__bg", 1)[0]
    return [item for item in combo.split("_") if item]


def benchmark_groups_for_types(physics_types: Sequence[str]) -> List[Dict[str, Any]]:
    """Return unique benchmark groups covered by a list of raw PhysInOne types."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for physics_type in physics_types:
        info = group_for_type(physics_type)
        if info["group_id"] in seen:
            continue
        seen.add(info["group_id"])
        out.append(
            {
                "group_id": info["group_id"],
                "group_slug": info["group_slug"],
                "group_name": info["group_name"],
            }
        )
    return out


def classify_sample_name(sample_name: str) -> Dict[str, Any]:
    """Parse a PhysInOne sample and return type-level and group-level labels."""
    physics_types = extract_physinone_types_from_sample_name(sample_name)
    return {
        "sample_name": Path(str(sample_name)).name,
        "sample_stem": normalize_sample_name(sample_name),
        "physics_types": physics_types,
        "benchmark_groups": benchmark_groups_for_types(physics_types),
    }


def scan_dataset_types(dataset_root: Path) -> Dict[str, Any]:
    """Scan a local PhysInOne root and validate taxonomy coverage."""
    zip_paths = sorted(dataset_root.rglob("*.zip"))
    split_type_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    discovered_types = set()
    split_counts: Dict[str, int] = defaultdict(int)

    for path in zip_paths:
        physics_types = extract_physinone_types_from_sample_name(path.name)
        split = path.parent.name if path.parent.name in {"SinglePhysics", "DoublePhysics", "TriplePhysics"} else "RootLevel"
        split_counts[split] += 1
        for physics_type in physics_types:
            discovered_types.add(physics_type)
            split_type_counts[split][physics_type] += 1

    uncovered = sorted(discovered_types - set(TYPE_TO_GROUP))
    unused = sorted(set(TYPE_TO_GROUP) - discovered_types)

    return {
        "dataset_root": str(dataset_root),
        "total_zip_files": len(zip_paths),
        "split_counts": dict(sorted(split_counts.items())),
        "discovered_types": sorted(discovered_types),
        "num_discovered_types": len(discovered_types),
        "taxonomy_type_count": len(TYPE_TO_GROUP),
        "uncovered_types": uncovered,
        "unused_taxonomy_types": unused,
        "split_type_counts": {
            split: dict(sorted(type_counts.items()))
            for split, type_counts in sorted(split_type_counts.items())
        },
    }


def export_taxonomy() -> Dict[str, Any]:
    """Return a JSON-serializable benchmark taxonomy payload."""
    return {
        "groups": GROUPS,
        "type_to_group": TYPE_TO_GROUP,
    }


def _print_human_summary(report: Dict[str, Any]) -> None:
    print(f"dataset_root={report['dataset_root']}")
    print(f"total_zip_files={report['total_zip_files']}")
    print(f"num_discovered_types={report['num_discovered_types']}")
    print(f"taxonomy_type_count={report['taxonomy_type_count']}")
    print(f"split_counts={report['split_counts']}")
    print(f"uncovered_types={report['uncovered_types']}")
    print(f"unused_taxonomy_types={report['unused_taxonomy_types']}")

    group_to_seen_types: Dict[str, List[str]] = defaultdict(list)
    for physics_type in report["discovered_types"]:
        info = group_for_type(physics_type)
        group_to_seen_types[info["group_id"]].append(physics_type)

    print("group_coverage:")
    for group in GROUPS:
        seen_types = group_to_seen_types.get(group["group_id"], [])
        print(
            f"  {group['group_id']} {group['group_slug']}: "
            f"{len(seen_types)} types -> {', '.join(seen_types)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhysInOne benchmark taxonomy helper.")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--sample_name",
        type=str,
        default="",
        help="Optional sample zip name or path to classify directly.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to write the taxonomy payload or validation payload as JSON.",
    )
    parser.add_argument(
        "--skip_scan",
        action="store_true",
        help="Only emit taxonomy / sample classification without scanning the dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.sample_name:
        payload = classify_sample_name(args.sample_name)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        if args.output_json is not None:
            args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return

    payload: Dict[str, Any] = {"taxonomy": export_taxonomy()}
    if not args.skip_scan:
        payload["validation"] = scan_dataset_types(args.dataset_root)
        _print_human_summary(payload["validation"])
    else:
        print(json.dumps(payload["taxonomy"], indent=2, ensure_ascii=True))

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
