#!/usr/bin/env python3
"""Render small, metadata-preserving multi-object collision variants.

The source cases are taken from the 0717 PyBullet dataset.  Each variant keeps
the object geometry, material, mass, friction, restitution, damping and camera
from its source case.  Only a shared XY translation and the incoming object's
initial velocity are perturbed.  A PyBullet preflight is run before rendering,
and a second simulation verifies that the intended dynamic-object contact is
still present.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from Dataset_physv_v2v_0819.scripts import generate_sim_preview_gallery as legacy  # noqa: E402
from Dataset_physv_v2v_0819.scripts.common_specs import (  # noqa: E402
    ObjectInstanceSpec,
    ScenarioBlueprint,
)
from Dataset_physv_v2v_0819.scripts.generate_external_variants_demo import (  # noqa: E402
    DEFAULT_SOURCE_ROOT,
    _blueprint_from_meta,
    _load_meta,
    _load_source_index,
    _rotate_xy,
)
from Dataset_physv_v2v_0819.scripts.render_sim_0705 import (  # noqa: E402
    audit_blueprint_initialization,
    blueprint_to_legacy_scenario,
    render_blueprint_case,
)


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_collision_variants_demo"

# These are valid under the current strict initializer and have an observed
# dynamic-object collision in the source trajectory.  F2 gives two-body
# contacts; F3 gives a three-object scene with a lead-to-middle contact.
DEFAULT_SOURCE_CASES = (
    "0717_f2_attempt001614",  # head-on ball -> box
    "0717_f2_attempt001096",  # oblique wheel -> rounded box
    "0717_f3_attempt000307",  # three-body chain scene
)


def _contact_summary(blueprint: ScenarioBlueprint, seed: int) -> dict[str, Any]:
    """Simulate without rendering and summarize dynamic-object contacts."""

    scenario = blueprint_to_legacy_scenario(blueprint, seed=seed)
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("could not connect to PyBullet DIRECT")
    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0.0, 0.0, -scenario.gravity)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / legacy.SIM_HZ,
            numSolverIterations=legacy.PHYSICS_SOLVER_ITERATIONS,
            numSubSteps=legacy.PHYSICS_SUB_STEPS,
            contactERP=legacy.PHYSICS_CONTACT_ERP,
            erp=legacy.PHYSICS_CONTACT_ERP,
        )
        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=scenario.floor_friction,
            restitution=scenario.floor_restitution,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )

        body_ids: dict[str, int] = {}
        for obj in scenario.objects:
            body_id = p.createMultiBody(
                baseMass=obj.mass if obj.dynamic else 0.0,
                baseCollisionShapeIndex=legacy._collision_shape(obj),
                basePosition=obj.position,
                baseOrientation=legacy._quat_from_euler_deg(obj.orientation_euler_deg),
            )
            p.changeDynamics(
                body_id,
                -1,
                restitution=obj.restitution,
                lateralFriction=obj.friction,
                linearDamping=obj.linear_damping,
                angularDamping=obj.angular_damping,
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            )
            p.resetBaseVelocity(
                body_id,
                linearVelocity=obj.linear_velocity,
                angularVelocity=obj.angular_velocity,
            )
            body_ids[obj.name] = int(body_id)

        names = list(body_ids)
        pair_counts: dict[str, int] = {}
        first_contact_s: dict[str, float] = {}
        max_impulse: dict[str, float] = {}
        pre_roll_steps = int(scenario.pre_roll_s * legacy.SIM_HZ)
        total_steps = int(3.0 * legacy.SIM_HZ)
        for step in range(pre_roll_steps + total_steps):
            p.stepSimulation()
            if step < pre_roll_steps:
                continue
            visible_time = (step - pre_roll_steps) / legacy.SIM_HZ
            for index, left_name in enumerate(names):
                for right_name in names[index + 1 :]:
                    contacts = p.getContactPoints(body_ids[left_name], body_ids[right_name])
                    if not contacts:
                        continue
                    pair = f"{left_name}__{right_name}"
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    first_contact_s.setdefault(pair, float(visible_time))
                    max_impulse[pair] = max(
                        max_impulse.get(pair, 0.0),
                        max(float(contact[9]) for contact in contacts),
                    )

        required_pair = (
            "driver_0__target_0" if blueprint.family_key == "F2" else "lead_0__mid_0"
        )
        return {
            "required_pair": required_pair,
            "required_pair_contact_frames": int(pair_counts.get(required_pair, 0)),
            "passed": bool(pair_counts.get(required_pair, 0)),
            "pair_contact_frames": pair_counts,
            "first_contact_s": first_contact_s,
            "max_normal_impulse": {
                key: round(float(value), 6) for key, value in max_impulse.items()
            },
        }
    finally:
        p.disconnect(client)


def _collision_perturbation(
    blueprint: ScenarioBlueprint,
    *,
    source_case_id: str,
    variant_index: int,
    seed: int,
) -> tuple[ScenarioBlueprint, dict[str, Any]]:
    """Apply conservative external perturbations that retain the contact path."""

    rng = np.random.default_rng(seed)
    group_delta = np.asarray(
        [rng.uniform(-0.045, 0.045), rng.uniform(-0.035, 0.035), 0.0],
        dtype=np.float64,
    )
    if blueprint.family_key == "F2":
        speed_scale = float(rng.uniform(0.94, 1.06))
        heading_delta = float(rng.uniform(-2.0, 2.0))
    else:
        speed_scale = float(rng.uniform(0.96, 1.04))
        heading_delta = float(rng.uniform(-1.25, 1.25))
    angular_scale = float(rng.uniform(0.96, 1.04))

    incoming_name = "driver_0" if blueprint.family_key == "F2" else "lead_0"
    objects: list[ObjectInstanceSpec] = []
    perturbations: list[dict[str, Any]] = []
    for obj in blueprint.objects:
        if not obj.dynamic:
            objects.append(obj)
            continue

        new_position = np.asarray(obj.position, dtype=np.float64) + group_delta
        new_velocity = np.asarray(obj.linear_velocity, dtype=np.float64)
        new_angular_velocity = np.asarray(obj.angular_velocity, dtype=np.float64)
        object_speed_scale = 1.0
        object_heading_delta = 0.0
        if obj.name == incoming_name:
            new_velocity = _rotate_xy(new_velocity, heading_delta) * speed_scale
            new_angular_velocity = new_angular_velocity * angular_scale
            object_speed_scale = speed_scale
            object_heading_delta = heading_delta

        # Keep the original orientation.  For boxes and cylinders this avoids
        # changing the source's ground contact envelope while still varying
        # the external position and incoming motion state.
        objects.append(
            replace(
                obj,
                position=tuple(float(value) for value in new_position),
                linear_velocity=tuple(float(value) for value in new_velocity),
                angular_velocity=tuple(float(value) for value in new_angular_velocity),
                metadata={
                    **obj.metadata,
                    "external_collision_variant": True,
                    "source_case_id": source_case_id,
                    "variant_index": variant_index,
                },
            )
        )
        perturbations.append(
            {
                "object": obj.name,
                "position_delta_m": [round(float(value), 6) for value in group_delta],
                "velocity_scale": round(float(object_speed_scale), 6),
                "heading_delta_deg": round(float(object_heading_delta), 6),
                "angular_velocity_scale": round(float(angular_scale if obj.name == incoming_name else 1.0), 6),
            }
        )

    variant = replace(
        blueprint,
        sample_key=f"{source_case_id}__colv{variant_index:02d}",
        objects=tuple(objects),
        metadata={
            **copy.deepcopy(blueprint.metadata),
            "external_collision_variant": True,
            "source_case_id": source_case_id,
            "variant_index": variant_index,
            "variant_seed": seed,
            "changed_variables": ["shared_position_xy", "incoming_linear_velocity", "incoming_angular_velocity"],
            "collision_preservation": "required dynamic-object contact verified by PyBullet",
        },
    )
    return variant, {
        "variant_seed": seed,
        "shared_position_delta_m": [round(float(value), 6) for value in group_delta],
        "incoming_object": incoming_name,
        "incoming_speed_scale": round(speed_scale, 6),
        "incoming_heading_delta_deg": round(heading_delta, 6),
        "incoming_angular_velocity_scale": round(angular_scale, 6),
        "objects": perturbations,
    }


def _find_valid_variant(
    blueprint: ScenarioBlueprint,
    *,
    source_case_id: str,
    variant_index: int,
    seed: int,
) -> tuple[ScenarioBlueprint, dict[str, Any], dict[str, Any], int]:
    for attempt in range(80):
        candidate_seed = int(seed + attempt * 100003)
        variant, perturbations = _collision_perturbation(
            blueprint,
            source_case_id=source_case_id,
            variant_index=variant_index,
            seed=candidate_seed,
        )
        try:
            qa = audit_blueprint_initialization(blueprint=variant, seed=candidate_seed)
            collision = _contact_summary(variant, candidate_seed)
        except Exception:
            continue
        if collision["passed"]:
            return variant, perturbations, collision, candidate_seed
    raise RuntimeError(
        f"could not find a strict, colliding variant for {source_case_id} v{variant_index:02d}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-case", action="append", dest="source_cases")
    parser.add_argument("--seed", type=int, default=20260825)
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
        # Fail early for a source that cannot be reproduced by the current
        # strict renderer; this prevents silently comparing incompatible cases.
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
        print(f"{source_case_id}: source collision verified; generating {variant_count} variants")
        for variant_index in range(1, variant_count + 1):
            base_seed = int(args.seed + source_offset * 1009 + variant_index * 37)
            case_root = output_root / "cases" / str(source_item["family_key"]) / f"{source_case_id}__colv{variant_index:02d}"
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
            print(
                f"  done {variant.sample_key}: required contact frames="
                f"{collision['required_pair_contact_frames']}"
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
                "purpose": "demo multi-object collision variants; no VAE/prompt/Utonia cache generated",
                "source_root": str(source_root),
                "source_cases": list(source_cases),
                "num_generated": len(generated),
                "families": {"F2": "two-body collision", "F3": "three-body collision scene"},
                "changed_variables": [
                    "shared_position_xy",
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
                "collision_contract": "each variant must retain the source family's required dynamic-object contact",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"generated={len(generated)} output_root={output_root}")


if __name__ == "__main__":
    main()
