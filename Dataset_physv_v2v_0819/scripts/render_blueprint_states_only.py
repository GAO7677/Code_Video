"""Generate PyBullet trajectory states without invoking the preview renderer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import generate_sim_preview_gallery as legacy
from .render_sim_0705 import (
    audit_blueprint_initialization,
    blueprint_to_legacy_scenario,
)


def _object_payload(obj, *, material_key: str) -> dict[str, object]:
    return {
        "name": obj.name,
        "shape": obj.shape,
        "size": {str(k): float(v) for k, v in obj.size.items()},
        "color": [float(v) for v in obj.color],
        "mass": float(obj.mass),
        "dynamic": bool(obj.dynamic),
        "position": [float(v) for v in obj.position],
        "orientation_euler_deg": [float(v) for v in obj.orientation_euler_deg],
        "linear_velocity": [float(v) for v in obj.linear_velocity],
        "angular_velocity": [float(v) for v in obj.angular_velocity],
        "restitution": float(obj.restitution),
        "friction": float(obj.friction),
        "linear_damping": float(obj.linear_damping),
        "angular_damping": float(obj.angular_damping),
        "role": obj.role,
        "material_key": material_key,
    }


def render_blueprint_states_only(
    *,
    blueprint,
    seed: int,
    output_root: Path,
) -> dict[str, str]:
    """Replay a blueprint and save the exact pose trajectory for Eevee.

    This deliberately avoids PyRender.  The later Blender pass is responsible
    for all RGB rendering, so old families cannot spend time in a second
    preview renderer before reaching the final visualization.
    """

    output_root.mkdir(parents=True, exist_ok=True)
    scenario = blueprint_to_legacy_scenario(blueprint, seed=seed)
    blueprint_material_keys = {obj.name: str(obj.material_key) for obj in blueprint.objects}
    initialization_qa = audit_blueprint_initialization(blueprint=blueprint, seed=seed)
    client = legacy.p.connect(legacy.p.DIRECT)
    if client < 0:
        raise RuntimeError("could not connect to PyBullet DIRECT")
    try:
        p = legacy.p
        p.resetSimulation()
        p.setAdditionalSearchPath(legacy.pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0.0, 0.0, -float(scenario.gravity))
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
            lateralFriction=float(scenario.floor_friction),
            restitution=float(scenario.floor_restitution),
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )
        body_ids: dict[str, int] = {}
        for obj in scenario.objects:
            body_id = p.createMultiBody(
                baseMass=float(obj.mass) if obj.dynamic else 0.0,
                baseCollisionShapeIndex=legacy._collision_shape(obj),
                basePosition=list(obj.position),
                baseOrientation=legacy._quat_from_euler_deg(obj.orientation_euler_deg),
            )
            p.changeDynamics(
                body_id,
                -1,
                restitution=float(obj.restitution),
                lateralFriction=float(obj.friction),
                linearDamping=float(obj.linear_damping),
                angularDamping=float(obj.angular_damping),
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            )
            p.resetBaseVelocity(
                body_id,
                linearVelocity=list(obj.linear_velocity),
                angularVelocity=list(obj.angular_velocity),
            )
            body_ids[obj.name] = int(body_id)

        qa_stages = [
            legacy.assert_initialization_contacts(
                body_ids,
                plane_id,
                stage="state_only_post_creation",
            )
        ]
        for _ in range(int(round(float(scenario.pre_roll_s) * legacy.SIM_HZ))):
            p.stepSimulation()
        qa_stages.append(
            legacy.assert_initialization_contacts(
                body_ids,
                plane_id,
                stage="state_only_post_pre_roll",
            )
        )

        total_steps = int(legacy.SIM_DURATION * legacy.SIM_HZ)
        frame_count = int(np.ceil(total_steps / legacy.RECORD_EVERY))
        count = len(scenario.objects)
        positions = np.zeros((frame_count, count, 3), dtype=np.float32)
        quats = np.zeros((frame_count, count, 4), dtype=np.float32)
        linear_velocities = np.zeros((frame_count, count, 3), dtype=np.float32)
        angular_velocities = np.zeros((frame_count, count, 3), dtype=np.float32)
        frame_index = 0
        for step_index in range(total_steps):
            if step_index % legacy.RECORD_EVERY != 0:
                p.stepSimulation()
                continue
            if frame_index == 0:
                qa_stages.append(
                    legacy.assert_initialization_contacts(
                        body_ids,
                        plane_id,
                        stage="state_only_video_frame_0",
                    )
                )
            for object_index, obj in enumerate(scenario.objects):
                body_id = body_ids[obj.name]
                pos, quat = p.getBasePositionAndOrientation(body_id)
                linvel, angvel = p.getBaseVelocity(body_id)
                positions[frame_index, object_index] = np.asarray(pos, dtype=np.float32)
                quats[frame_index, object_index] = np.asarray(quat, dtype=np.float32)
                linear_velocities[frame_index, object_index] = np.asarray(linvel, dtype=np.float32)
                angular_velocities[frame_index, object_index] = np.asarray(angvel, dtype=np.float32)
            frame_index += 1
            p.stepSimulation()
        positions = positions[:frame_index]
        quats = quats[:frame_index]
        linear_velocities = linear_velocities[:frame_index]
        angular_velocities = angular_velocities[:frame_index]
    finally:
        p.disconnect(client)

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    states_path = meta_dir / f"{blueprint.sample_key}_states.npz"
    np.savez_compressed(
        states_path,
        positions=positions,
        quats=quats,
        linear_velocities=linear_velocities,
        angular_velocities=angular_velocities,
        frame_times=np.arange(frame_index, dtype=np.float32) / legacy.FPS,
        object_names=np.asarray([obj.name for obj in scenario.objects], dtype=np.str_),
        object_roles=np.asarray([obj.role for obj in scenario.objects], dtype=np.str_),
        frame_width=np.asarray([1280], dtype=np.int32),
        frame_height=np.asarray([720], dtype=np.int32),
    )
    meta_path = meta_dir / f"{blueprint.sample_key}.json"
    meta = {
        "case_id": blueprint.sample_key,
        "key": blueprint.sample_key,
        "family_key": blueprint.family_key,
        "title": blueprint.title,
        "description": blueprint.description,
        "fps": legacy.FPS,
        "sim_hz": legacy.SIM_HZ,
        "duration_s": legacy.SIM_DURATION,
        "gravity": blueprint.gravity,
        "surface_key": blueprint.surface_key,
        "lighting_key": blueprint.lighting_key,
        "camera": {
            "eye": list(blueprint.camera.eye),
            "target": list(blueprint.camera.target),
            "up": list(blueprint.camera.up),
            "yfov_deg": blueprint.camera.yfov_deg,
        },
        "objects": [
            _object_payload(
                obj,
                material_key=blueprint_material_keys.get(obj.name, "rubber_dark"),
            )
            for obj in scenario.objects
        ],
        "states": str(states_path),
        "initialization_qa": {
            "contract": initialization_qa,
            "state_only_stages": qa_stages,
        },
        "state_only_staging": True,
        "blueprint": {
            "family_key": blueprint.family_key,
            "sample_key": blueprint.sample_key,
            "metadata": blueprint.metadata,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"meta": str(meta_path), "states": str(states_path)}
