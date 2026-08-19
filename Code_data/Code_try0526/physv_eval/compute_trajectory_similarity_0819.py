#!/usr/bin/env python3
"""Compute pairwise rigid-body trajectory similarity within each PhysV group."""

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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def effective_radius(actor: dict[str, Any]) -> float:
    size = actor.get("size_m", {})
    if actor.get("shape") == "sphere":
        return float(size.get("radius", 0.0))
    dimensions = [float(value) for value in size.values() if isinstance(value, (int, float))]
    return math.sqrt(sum(value * value for value in dimensions)) if dimensions else 0.0


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    result = q.copy()
    result[..., 1:] *= -1.0
    return result


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def relative_rotations(rotations: np.ndarray) -> np.ndarray:
    q = quat_normalize(rotations)
    initial_inverse = quat_conjugate(q[0])
    return quat_normalize(quat_multiply(np.broadcast_to(initial_inverse, q.shape), q))


def rotation_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(quat_normalize(a) * quat_normalize(b), axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def load_case(sample_dir: Path) -> dict[str, Any]:
    metadata = read_json(sample_dir / "metadata.json")
    actors = metadata.get("actors", {})
    arrays = np.load(sample_dir / "raw/trajectories.npz", allow_pickle=False)
    names = [str(value) for value in arrays["object_names"]]
    roles = [str(value) for value in arrays["object_roles"]]
    objects: dict[str, dict[str, Any]] = {}
    for name, role in zip(names, roles):
        if not role.startswith("dynamic"):
            continue
        positions = np.asarray(arrays[f"{name}_positions"], dtype=np.float64)
        rotations = np.asarray(arrays[f"{name}_rotations"], dtype=np.float64)
        relative_position = positions - positions[0]
        relative_rotation = relative_rotations(rotations)
        translation_path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
        rotation_path = float(np.sum(rotation_angles(rotations[:-1], rotations[1:])))
        radius = effective_radius(actors.get(name, {}))
        objects[name] = {
            "relative_position": relative_position,
            "relative_rotation": relative_rotation,
            "radius_m": radius,
            "rigid_path_m": translation_path + radius * rotation_path,
            "excursion_m": float(np.max(np.linalg.norm(relative_position, axis=1)))
            + radius * float(np.max(rotation_angles(relative_rotation, relative_rotation[:1]))),
        }
    return {
        "case_id": sample_dir.name,
        "family_key": metadata.get("family_key") or metadata.get("source_group", ""),
        "control": metadata.get("control", {}),
        "objects": objects,
    }


def object_distance(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float, float]:
    if first["relative_position"].shape != second["relative_position"].shape:
        raise ValueError("Trajectory arrays must have aligned frame counts and dimensions")
    translation_error = np.linalg.norm(first["relative_position"] - second["relative_position"], axis=1)
    rotation_error = rotation_angles(first["relative_rotation"], second["relative_rotation"])
    rotation_equivalent = 0.5 * (first["radius_m"] + second["radius_m"]) * rotation_error
    rigid_rmse = float(np.sqrt(np.mean(translation_error**2 + rotation_equivalent**2)))
    return rigid_rmse, float(np.sqrt(np.mean(translation_error**2))), float(np.sqrt(np.mean(rotation_equivalent**2)))


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output = (args.output_json or root / "reports/trajectory_similarity.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = [load_case(path) for path in sorted((root / "samples").iterdir()) if path.is_dir()]
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        groups.setdefault(case["family_key"], []).append(case)

    group_reports: dict[str, Any] = {}
    case_reports: dict[str, Any] = {}
    for group, members in groups.items():
        object_names = sorted(set.intersection(*(set(member["objects"]) for member in members)))
        object_stats: dict[str, dict[str, float]] = {}
        for name in object_names:
            paths = [member["objects"][name]["rigid_path_m"] for member in members]
            excursions = [member["objects"][name]["excursion_m"] for member in members]
            object_stats[name] = {
                "motion_weight": float(np.median(paths)),
                "normalization_scale_m": max(float(np.median(excursions)), 1e-3),
            }
        weight_sum = sum(stats["motion_weight"] for stats in object_stats.values())
        if weight_sum <= 1e-12:
            for stats in object_stats.values():
                stats["motion_weight"] = 1.0
            weight_sum = float(len(object_stats))
        for stats in object_stats.values():
            stats["normalized_weight"] = stats["motion_weight"] / weight_sum

        count = len(members)
        similarity = np.eye(count, dtype=np.float64)
        rigid_rmse = np.zeros((count, count), dtype=np.float64)
        pair_details: list[dict[str, Any]] = []
        for left in range(count):
            for right in range(left + 1, count):
                normalized_distance = 0.0
                absolute_distance = 0.0
                details: list[dict[str, Any]] = []
                for name in object_names:
                    total, translation, rotation = object_distance(
                        members[left]["objects"][name], members[right]["objects"][name]
                    )
                    weight = object_stats[name]["normalized_weight"]
                    scale = object_stats[name]["normalization_scale_m"]
                    normalized_distance += weight * total / scale
                    absolute_distance += weight * total
                    details.append(
                        {
                            "object_name": name,
                            "weight": weight,
                            "rigid_rmse_m": total,
                            "translation_rmse_m": translation,
                            "rotation_equivalent_rmse_m": rotation,
                        }
                    )
                score = math.exp(-normalized_distance)
                similarity[left, right] = similarity[right, left] = score
                rigid_rmse[left, right] = rigid_rmse[right, left] = absolute_distance
                pair_details.append(
                    {
                        "case_a": members[left]["case_id"],
                        "case_b": members[right]["case_id"],
                        "similarity": score,
                        "weighted_rigid_rmse_m": absolute_distance,
                        "objects": details,
                    }
                )

        case_ids = [member["case_id"] for member in members]
        for index, member in enumerate(members):
            peers = [
                {"case_id": case_ids[other], "similarity": float(similarity[index, other])}
                for other in range(count)
                if other != index
            ]
            peers.sort(key=lambda item: item["similarity"], reverse=True)
            case_reports[member["case_id"]] = {
                "group": group,
                "peer_similarities": peers,
                "mean_peer_similarity": float(np.mean([item["similarity"] for item in peers])) if peers else 1.0,
                "most_similar_case": peers[0] if peers else None,
                "least_similar_case": peers[-1] if peers else None,
            }
        upper = similarity[np.triu_indices(count, k=1)]
        group_reports[group] = {
            "case_ids": case_ids,
            "dynamic_object_names": object_names,
            "object_aggregation": "group-motion-weighted",
            "object_statistics": object_stats,
            "similarity_matrix": similarity.tolist(),
            "weighted_rigid_rmse_matrix_m": rigid_rmse.tolist(),
            "mean_pairwise_similarity": float(np.mean(upper)) if upper.size else 1.0,
            "min_pairwise_similarity": float(np.min(upper)) if upper.size else 1.0,
            "max_pairwise_similarity": float(np.max(upper)) if upper.size else 1.0,
            "pairs": pair_details,
        }

    report = {
        "schema_version": "physv_trajectory_similarity_v1",
        "dataset_root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": {
            "name": "motion_weighted_rigid_trajectory_similarity",
            "range": [0.0, 1.0],
            "formula": "exp(-sum_i group_motion_weight_i * rigid_trajectory_rmse_i / group_excursion_scale_i)",
            "alignment": "position and quaternion trajectories are aligned to each object's initial state",
            "interpretation": "1 means identical motion; lower values mean greater translation/rotation trajectory differences.",
        },
        "groups": group_reports,
        "cases": case_reports,
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")
    for group, summary in group_reports.items():
        print(
            f"{group}: mean={summary['mean_pairwise_similarity']:.4f}, "
            f"range=[{summary['min_pairwise_similarity']:.4f}, {summary['max_pairwise_similarity']:.4f}]"
        )


if __name__ == "__main__":
    main()
