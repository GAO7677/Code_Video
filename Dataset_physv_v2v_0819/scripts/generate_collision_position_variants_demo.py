#!/usr/bin/env python3
"""Render collision variants with independent per-object position changes.

Unlike the first collision demo, this generator does not apply one shared
translation to the whole scene.  Every dynamic object receives its own XY
position perturbation.  The incoming object velocity is then aimed toward the
perturbed target (with a small random angular offset), and PyBullet verifies
that the intended dynamic-object contact survives before RGB rendering.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from Dataset_physv_v2v_0819.scripts.generate_collision_variants_demo import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    _blueprint_from_meta,
    _contact_summary,
    _load_meta,
    _load_source_index,
)
from Dataset_physv_v2v_0819.scripts.render_sim_0705 import (  # noqa: E402
    audit_blueprint_initialization,
    render_blueprint_case,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_collision_position_variants_demo"
DEFAULT_SOURCE_CASES = (
    "0717_f2_attempt001614",
    "0717_f2_attempt001096",
    "0717_f3_attempt000307",
)


def _aim_velocity(
    base_velocity: tuple[float, float, float],
    origin_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    speed_scale: float,
    heading_jitter_deg: float,
) -> tuple[float, float, float]:
    base = np.asarray(base_velocity, dtype=np.float64)
    speed_xy = float(np.linalg.norm(base[:2]))
    if speed_xy < 1e-8:
        return tuple(float(value) for value in base)
    direction = np.asarray(target_xy, dtype=np.float64) - np.asarray(origin_xy, dtype=np.float64)
    if float(np.linalg.norm(direction)) < 1e-8:
        direction = base[:2]
    angle = math.atan2(float(direction[1]), float(direction[0])) + math.radians(heading_jitter_deg)
    horizontal = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    horizontal *= speed_xy * float(speed_scale)
    return (float(horizontal[0]), float(horizontal[1]), float(base[2]))


def _perturb_blueprint(
    blueprint,
    *,
    source_case_id: str,
    variant_index: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    incoming_name = "driver_0" if blueprint.family_key == "F2" else "lead_0"
    target_name = "target_0" if blueprint.family_key == "F2" else "mid_0"

    # Position deltas are sampled separately for every dynamic object.  The
    # ranges are deliberately small enough to keep the same camera framing,
    # but large enough to change relative placement visibly.
    position_delta: dict[str, np.ndarray] = {}
    for obj in blueprint.objects:
        if not obj.dynamic:
            continue
        x_limit, y_limit = (0.115, 0.075) if obj.name == target_name else (0.085, 0.060)
        position_delta[obj.name] = np.asarray(
            [rng.uniform(-x_limit, x_limit), rng.uniform(-y_limit, y_limit), 0.0],
            dtype=np.float64,
        )

    positions: dict[str, tuple[float, float]] = {}
    for obj in blueprint.objects:
        if obj.dynamic:
            base = np.asarray(obj.position, dtype=np.float64) + position_delta[obj.name]
            positions[obj.name] = (float(base[0]), float(base[1]))

    speed_scale = float(rng.uniform(0.90, 1.10))
    heading_jitter = float(rng.uniform(-4.0, 4.0))
    angular_scale = float(rng.uniform(0.92, 1.08))
    objects = []
    perturbations: list[dict[str, Any]] = []
    for obj in blueprint.objects:
        if not obj.dynamic:
            objects.append(obj)
            continue
        delta = position_delta[obj.name]
        base_position = np.asarray(obj.position, dtype=np.float64) + delta
        new_velocity = obj.linear_velocity
        new_angular_velocity = obj.angular_velocity
        if obj.name == incoming_name:
            new_velocity = _aim_velocity(
                obj.linear_velocity,
                positions[incoming_name],
                positions[target_name],
                speed_scale=speed_scale,
                heading_jitter_deg=heading_jitter,
            )
            new_angular_velocity = tuple(
                float(value) * angular_scale for value in obj.angular_velocity
            )
        objects.append(
            replace(
                obj,
                position=tuple(float(value) for value in base_position),
                linear_velocity=tuple(float(value) for value in new_velocity),
                angular_velocity=tuple(float(value) for value in new_angular_velocity),
                # Keep orientation unchanged so the original ground-contact
                # envelope remains valid; position and velocity are the
                # external variables being ablated here.
                metadata={
                    **obj.metadata,
                    "external_collision_position_variant": True,
                    "source_case_id": source_case_id,
                    "variant_index": variant_index,
                },
            )
        )
        perturbations.append(
            {
                "object": obj.name,
                "position_delta_m": [round(float(value), 6) for value in delta],
                "velocity_scale": round(float(speed_scale if obj.name == incoming_name else 1.0), 6),
                "heading_delta_deg": round(float(heading_jitter if obj.name == incoming_name else 0.0), 6),
                "angular_velocity_scale": round(float(angular_scale if obj.name == incoming_name else 1.0), 6),
            }
        )

    variant = replace(
        blueprint,
        sample_key=f"{source_case_id}__posv{variant_index:02d}",
        objects=tuple(objects),
        metadata={
            **blueprint.metadata,
            "external_collision_position_variant": True,
            "source_case_id": source_case_id,
            "variant_index": variant_index,
            "variant_seed": seed,
            "changed_variables": [
                "per_object_position_xy",
                "incoming_linear_velocity",
                "incoming_angular_velocity",
            ],
            "position_policy": "independent XY perturbation per dynamic object",
            "collision_preservation": "required dynamic-object contact verified by PyBullet",
        },
    )
    return variant, {
        "variant_seed": seed,
        "position_mode": "per_object_independent_xy",
        "incoming_object": incoming_name,
        "target_object": target_name,
        "incoming_speed_scale": round(speed_scale, 6),
        "incoming_heading_delta_deg": round(heading_jitter, 6),
        "incoming_angular_velocity_scale": round(angular_scale, 6),
        "objects": perturbations,
    }


def _find_valid_variant(blueprint, *, source_case_id: str, variant_index: int, seed: int):
    for attempt in range(100):
        candidate_seed = int(seed + attempt * 100003)
        variant, perturbations = _perturb_blueprint(
            blueprint,
            source_case_id=source_case_id,
            variant_index=variant_index,
            seed=candidate_seed,
        )
        # Reject an accidental near-global translation: each dynamic object
        # must have an independently sampled position delta.
        deltas = [tuple(item["position_delta_m"]) for item in perturbations["objects"]]
        if len(set(deltas)) < min(2, len(deltas)):
            continue
        try:
            audit_blueprint_initialization(blueprint=variant, seed=candidate_seed)
            collision = _contact_summary(variant, candidate_seed)
        except Exception:
            continue
        if collision["passed"]:
            return variant, perturbations, collision, candidate_seed
    raise RuntimeError(
        f"could not find a strict position-changing collision variant for {source_case_id} v{variant_index:02d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-case", action="append", dest="source_cases")
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_index = _load_source_index(source_root)
    source_cases = tuple(args.source_cases or DEFAULT_SOURCE_CASES)
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for source_offset, source_case_id in enumerate(source_cases):
        if source_case_id not in source_index:
            raise KeyError(f"source case not found: {source_case_id}")
        source_item = source_index[source_case_id]
        source_meta = _load_meta(source_item)
        source_blueprint = _blueprint_from_meta(source_meta, source_case_id)
        audit_blueprint_initialization(blueprint=source_blueprint, seed=int(source_item["seed"]))
        source_collision = _contact_summary(source_blueprint, int(source_item["seed"]))
        if not source_collision["passed"]:
            raise ValueError(f"source has no required collision: {source_case_id}: {source_collision}")
        source_records.append(
            {
                "source_case_id": source_case_id,
                "family_key": source_item.get("family_key"),
                "source_video": source_item.get("video"),
                "source_meta": source_item.get("meta"),
                "source_collision": source_collision,
            }
        )
        count_rng = np.random.default_rng(args.seed + source_offset * 1009 + 7919)
        variant_count = int(count_rng.integers(2, 4))
        print(f"{source_case_id}: independent-position mode; generating {variant_count} variants")
        for variant_index in range(1, variant_count + 1):
            base_seed = int(args.seed + source_offset * 1009 + variant_index * 37)
            case_root = output_root / "cases" / str(source_item["family_key"]) / f"{source_case_id}__posv{variant_index:02d}"
            existing_manifest = case_root / "case_manifest.json"
            if existing_manifest.is_file():
                manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
                generated.append(manifest)
                print(f"  reuse {case_root.name}")
                continue
            variant, perturbations, collision, variant_seed = _find_valid_variant(
                source_blueprint,
                source_case_id=source_case_id,
                variant_index=variant_index,
                seed=base_seed,
            )
            manifest = render_blueprint_case(
                blueprint=variant,
                seed=variant_seed,
                output_root=case_root,
                width=1280,
                height=720,
                scene_style="indoor_realistic",
            )
            manifest.update(
                {
                    "source_case_id": source_case_id,
                    "source_video": str(source_item["video"]),
                    "source_meta": str(source_item["meta"]),
                    "variant_index": variant_index,
                    "variant_seed": variant_seed,
                    "perturbations": perturbations,
                    "collision_summary": collision,
                }
            )
            existing_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            generated.append(manifest)
            delta_text = ", ".join(
                f"{item['object']}={item['position_delta_m']}"
                for item in perturbations["objects"]
            )
            print(
                f"  done {variant.sample_key}: contact frames="
                f"{collision['required_pair_contact_frames']} position_deltas={delta_text}"
            )

    (output_root / "source_records.json").write_text(
        json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "manifest.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "README.json").write_text(
        json.dumps(
            {
                "purpose": "demo multi-object collision variants with independent object positions; no VAE/prompt/Utonia cache generated",
                "source_root": str(source_root),
                "source_cases": list(source_cases),
                "num_generated": len(generated),
                "changed_variables": [
                    "per_object_position_xy",
                    "incoming_linear_velocity",
                    "incoming_angular_velocity",
                ],
                "unchanged_variables": [
                    "shape",
                    "size",
                    "mass",
                    "friction",
                    "restitution",
                    "damping",
                    "material",
                    "object_count",
                    "static_scene",
                    "camera",
                ],
                "collision_contract": "each variant retains its source family's required dynamic-object contact",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"generated={len(generated)} output_root={output_root}")


if __name__ == "__main__":
    main()
