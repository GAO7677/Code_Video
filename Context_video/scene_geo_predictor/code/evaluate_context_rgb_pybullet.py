"""Evaluate RGB-estimated state/geometry in the existing PyBullet protocol."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from context_rgb_pybullet_common import (
    BALL_RADIUS_M,
    FPS,
    FUTURE_FRAMES,
    SIM_HZ,
    dump_json,
    load_json,
    rolling_omega,
    velocity_errors,
)
from pybullet_context_rollout import _collision_shape_for_client


PROJECT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5")
STEPS_PER_OUTPUT = SIM_HZ // FPS
OMEGA_POLICIES = ("gt_observed", "zero", "pure_rolling_hypothesis")
COMBINATIONS = {
    "A_gt_state_gt_geometry": ("gt", "gt"),
    "B_estimated_state_gt_geometry": ("estimated", "gt"),
    "C_gt_state_estimated_geometry": ("gt", "estimated"),
    "D_estimated_state_estimated_geometry": ("estimated", "estimated"),
}


def load_engine():
    sys.path.insert(0, str(PROJECT))
    import recheck_pybullet_bridge as recheck

    pilot, generator, _ = recheck.load_engine()
    return recheck, pilot, generator


def configure_engine(p, client: int, generator, blueprint) -> dict[str, Any]:
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0.0, 0.0, -generator.EARTH_GRAVITY, physicsClientId=client)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / SIM_HZ,
        numSolverIterations=generator.legacy.PHYSICS_SOLVER_ITERATIONS,
        numSubSteps=int(blueprint.metadata.get("physics_sub_steps", STEPS_PER_OUTPUT)),
        contactERP=generator.legacy.PHYSICS_CONTACT_ERP,
        erp=generator.legacy.PHYSICS_CONTACT_ERP,
        physicsClientId=client,
    )
    return {
        key: value
        for key, value in p.getPhysicsEngineParameters(physicsClientId=client).items()
        if isinstance(value, (str, int, float, bool))
    }


def add_gt_geometry(p, client: int, generator, blueprint, seed: int) -> tuple[dict[str, int], dict[int, str]]:
    p.setAdditionalSearchPath(generator.pybullet_data.getDataPath(), physicsClientId=client)
    plane = int(p.loadURDF("plane.urdf", physicsClientId=client))
    surface = generator.build_surface_catalog()[blueprint.surface_key]
    floor_mu = float(np.clip(
        blueprint.metadata.get("floor_friction", surface.floor_friction_range.midpoint()), 0.01, 1.2
    ))
    p.changeDynamics(
        plane,
        -1,
        lateralFriction=floor_mu,
        restitution=0.02,
        activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        physicsClientId=client,
    )
    ids: dict[str, int] = {"floor": plane}
    body_names = {plane: "floor"}
    for obj in blueprint.objects:
        if obj.dynamic or obj.metadata.get("visual_only"):
            continue
        body = int(p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=_collision_shape_for_client(p, obj, client),
            basePosition=list(obj.position),
            baseOrientation=generator.legacy._quat_from_euler_deg(list(obj.orientation_euler_deg)),
            physicsClientId=client,
        ))
        p.changeDynamics(
            body,
            -1,
            restitution=float(obj.restitution),
            lateralFriction=float(obj.friction),
            rollingFriction=float(obj.metadata.get("rolling_friction", 0.0)),
            spinningFriction=float(obj.metadata.get("spinning_friction", 0.0)),
            linearDamping=float(obj.linear_damping),
            angularDamping=float(obj.angular_damping),
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            physicsClientId=client,
        )
        ids[obj.name] = body
        body_names[body] = obj.name
    if blueprint.metadata.get("constraints"):
        raise ValueError("pilot evaluator does not admit constrained scenes")
    for left, right in blueprint.metadata.get("disable_collision_pairs", []):
        p.setCollisionFilterPair(ids[str(left)], ids[str(right)], -1, -1, 0, physicsClientId=client)
    return ids, body_names


def add_estimated_geometry(p, client: int, primitives: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    ids: dict[str, int] = {}
    body_names: dict[int, str] = {}
    for primitive in primitives:
        if primitive.get("shape") != "box":
            raise ValueError(f"unsupported estimated shape: {primitive.get('shape')}")
        half_extents = primitive["size"]["half_extents_m"]
        shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client
        )
        body = int(p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            basePosition=primitive["position_m"],
            baseOrientation=primitive["orientation_xyzw"],
            physicsClientId=client,
        ))
        material = primitive["material"]
        p.changeDynamics(
            body,
            -1,
            lateralFriction=float(material["lateral_friction"]),
            restitution=float(material["restitution"]),
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            physicsClientId=client,
        )
        name = str(primitive["name"])
        ids[name] = body
        body_names[body] = name
    return ids, body_names


def add_dynamic_ball(
    p,
    client: int,
    generator,
    blueprint,
    seed: int,
    position: np.ndarray,
    quaternion: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> int:
    dynamic = [obj for obj in blueprint.objects if obj.name == "pilot_ball" and obj.dynamic]
    if len(dynamic) != 1:
        raise ValueError(f"expected one dynamic pilot_ball, got {len(dynamic)}")
    obj = dynamic[0]
    body = int(p.createMultiBody(
        baseMass=float(obj.mass),
        baseCollisionShapeIndex=_collision_shape_for_client(p, obj, client),
        basePosition=np.asarray(position, dtype=np.float64).tolist(),
        baseOrientation=np.asarray(quaternion, dtype=np.float64).tolist(),
        physicsClientId=client,
    ))
    dynamics = dict(
        restitution=float(obj.restitution),
        lateralFriction=float(obj.friction),
        rollingFriction=float(obj.metadata.get("rolling_friction", 0.0)),
        spinningFriction=float(obj.metadata.get("spinning_friction", 0.0)),
        linearDamping=float(obj.linear_damping),
        angularDamping=float(obj.angular_damping),
        activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        physicsClientId=client,
    )
    if "ccd_swept_sphere_radius_m" in obj.metadata:
        dynamics.update(
            ccdSweptSphereRadius=float(obj.metadata["ccd_swept_sphere_radius_m"]),
            contactProcessingThreshold=0.0,
        )
    p.changeDynamics(body, -1, **dynamics)
    p.resetBaseVelocity(
        body,
        linearVelocity=np.asarray(linear_velocity, dtype=np.float64).tolist(),
        angularVelocity=np.asarray(angular_velocity, dtype=np.float64).tolist(),
        physicsClientId=client,
    )
    return body


def semantic_body(name: str, family: str) -> str:
    name = str(name)
    if name in {"floor", "estimated_ground"}:
        return "ground"
    if family == "support_edge":
        if "left_platform" in name:
            return "left_platform"
        if "right_platform" in name:
            return "right_platform"
        return "support_structure"
    return "interaction_structure"


def contact_snapshot(p, client: int, ball: int, body_names: dict[int, str]) -> list[dict[str, Any]]:
    rows = []
    for point in p.getContactPoints(bodyA=ball, physicsClientId=client):
        body_b = int(point[2])
        rows.append({
            "body": body_names.get(body_b, f"body_{body_b}"),
            "distance_m": float(point[8]),
            "normal_force_n": float(point[9]),
        })
    return rows


def event_summary(
    family: str,
    initial_contacts: list[dict[str, Any]],
    contact_samples: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    initial_semantics = sorted({semantic_body(row["body"], family) for row in initial_contacts})
    all_semantics = sorted({
        semantic_body(row["body"], family)
        for sample in contact_samples for row in sample
    })
    first_by_semantic: dict[str, float] = {}
    max_penetration = 0.0
    max_non_ground_penetration = 0.0
    for api_index, sample in enumerate(contact_samples, start=1):
        for row in sample:
            semantic = semantic_body(row["body"], family)
            first_by_semantic.setdefault(semantic, api_index / SIM_HZ)
            penetration = max(0.0, -float(row["distance_m"]))
            max_penetration = max(max_penetration, penetration)
            if semantic != "ground":
                max_non_ground_penetration = max(max_non_ground_penetration, penetration)

    support_loss_time = None
    if family == "support_edge" and "left_platform" in initial_semantics:
        for api_index, sample in enumerate(contact_samples, start=1):
            semantics = {semantic_body(row["body"], family) for row in sample}
            if "left_platform" not in semantics:
                support_loss_time = api_index / SIM_HZ
                break
    if family == "support_edge":
        candidates = [
            value for key, value in first_by_semantic.items()
            if key in {"right_platform", "support_structure", "ground"}
        ]
        if support_loss_time is not None:
            candidates.append(support_loss_time)
        interaction_time = min(candidates, default=None)
        if "right_platform" in all_semantics:
            outcome = "right_platform_contact"
        elif "ground" in all_semantics and "ground" not in initial_semantics:
            outcome = "fell_to_ground"
        elif support_loss_time is not None:
            outcome = "left_support_lost_no_landing_within_horizon"
        else:
            outcome = "remained_on_left_support"
    else:
        interaction_time = first_by_semantic.get("interaction_structure")
        outcome = "structure_contact" if interaction_time is not None else "no_structure_contact"
    return {
        "category": outcome,
        "initial_contact_semantics": initial_semantics,
        "future_contact_semantics": all_semantics,
        "first_contact_time_by_semantic_s": first_by_semantic,
        "first_interaction_time_after_rgb7_s": interaction_time,
        "first_interaction_future_index_30hz": (
            None if interaction_time is None else min(FUTURE_FRAMES - 1, max(0, math.ceil(interaction_time * FPS) - 1))
        ),
        "first_support_loss_time_after_rgb7_s": support_loss_time,
        "max_penetration_m": max_penetration,
        "max_non_ground_penetration_m": max_non_ground_penetration,
        "contact_log_resolution_s": 1.0 / SIM_HZ,
        "hidden_engine_substeps_observed": False,
    }


def rollout(
    generator,
    case,
    seed: int,
    geometry_source: str,
    primitives: list[dict[str, Any]],
    state: dict[str, np.ndarray],
) -> dict[str, Any]:
    import pybullet as p

    client = int(p.connect(p.DIRECT))
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    started = time.perf_counter()
    try:
        engine = configure_engine(p, client, generator, case.blueprint)
        if geometry_source == "gt":
            ids, body_names = add_gt_geometry(p, client, generator, case.blueprint, seed)
        elif geometry_source == "estimated":
            ids, body_names = add_estimated_geometry(p, client, primitives)
        else:
            raise ValueError(geometry_source)
        ball = add_dynamic_ball(
            p, client, generator, case.blueprint, seed,
            state["position"], state["quaternion"], state["linear_velocity"], state["angular_velocity"],
        )
        ids["pilot_ball"] = ball
        body_names[ball] = "pilot_ball"
        p.performCollisionDetection(physicsClientId=client)
        initial = contact_snapshot(p, client, ball, body_names)
        positions, velocities, angular_velocities = [], [], []
        contact_samples: list[list[dict[str, Any]]] = []
        for _output_index in range(FUTURE_FRAMES):
            for _ in range(STEPS_PER_OUTPUT):
                p.stepSimulation(physicsClientId=client)
                contact_samples.append(contact_snapshot(p, client, ball, body_names))
            position, _ = p.getBasePositionAndOrientation(ball, physicsClientId=client)
            linear, angular = p.getBaseVelocity(ball, physicsClientId=client)
            positions.append(position)
            velocities.append(linear)
            angular_velocities.append(angular)
        return {
            "positions": np.asarray(positions, dtype=np.float64),
            "linear_velocities": np.asarray(velocities, dtype=np.float64),
            "angular_velocities": np.asarray(angular_velocities, dtype=np.float64),
            "initial_contacts": initial,
            "contact_samples": contact_samples,
            "event": event_summary(case.blueprint.metadata["pilot_family"], initial, contact_samples),
            "engine_parameters": engine,
            "api_step_calls": FUTURE_FRAMES * STEPS_PER_OUTPUT,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        p.disconnect(physicsClientId=client)


def recover_gt_omega(recheck, generator, case, seed: int, raw: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay only through RGB7 and independently verify the saved prefix."""
    import pybullet as p

    client = int(p.connect(p.DIRECT))
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        configure_engine(p, client, generator, case.blueprint)
        ids, body_names = add_gt_geometry(p, client, generator, case.blueprint, seed)
        dynamic_obj = next(obj for obj in case.blueprint.objects if obj.name == "pilot_ball")
        ball = add_dynamic_ball(
            p,
            client,
            generator,
            case.blueprint,
            seed,
            np.asarray(dynamic_obj.position),
            np.asarray(generator.legacy._quat_from_euler_deg(list(dynamic_obj.orientation_euler_deg))),
            np.asarray(dynamic_obj.linear_velocity),
            np.asarray(dynamic_obj.angular_velocity),
        )
        ids["pilot_ball"] = ball
        body_names[ball] = "pilot_ball"
        for _ in range(int(round(float(case.blueprint.pre_roll_s) * SIM_HZ))):
            p.stepSimulation(physicsClientId=client)
        prefix_position, prefix_velocity, prefix_quaternion, prefix_angular = [], [], [], []
        for frame_index in range(8):
            position, quaternion = p.getBasePositionAndOrientation(ball, physicsClientId=client)
            linear, angular = p.getBaseVelocity(ball, physicsClientId=client)
            prefix_position.append(position)
            prefix_velocity.append(linear)
            prefix_quaternion.append(quaternion)
            prefix_angular.append(angular)
            if frame_index < 7:
                for _ in range(STEPS_PER_OUTPUT):
                    p.stepSimulation(physicsClientId=client)
        dynamic_index = int(raw["dynamic_index"])
        raw_position = np.asarray(raw["positions"][:8, dynamic_index], dtype=np.float64)
        raw_velocity = np.asarray(raw["linear_velocities"][:8, dynamic_index], dtype=np.float64)
        raw_quaternion = np.asarray(raw["quats"][:8, dynamic_index], dtype=np.float64)
        prefix_position = np.asarray(prefix_position)
        prefix_velocity = np.asarray(prefix_velocity)
        prefix_quaternion = np.asarray(prefix_quaternion)
        quaternion_error = np.minimum(
            np.linalg.norm(prefix_quaternion - raw_quaternion, axis=1),
            np.linalg.norm(prefix_quaternion + raw_quaternion, axis=1),
        )
        report = {
            "status": "EXECUTED",
            "max_position_error_m": float(np.max(np.linalg.norm(prefix_position - raw_position, axis=1))),
            "max_velocity_error_mps": float(np.max(np.linalg.norm(prefix_velocity - raw_velocity, axis=1))),
            "max_quaternion_sign_invariant_l2": float(np.max(quaternion_error)),
            "passed_1e-5": bool(
                np.max(np.linalg.norm(prefix_position - raw_position, axis=1)) <= 1e-5
                and np.max(np.linalg.norm(prefix_velocity - raw_velocity, axis=1)) <= 1e-5
                and np.max(quaternion_error) <= 1e-5
            ),
            "source": "explicit-client replay of original prefix through RGB7",
        }
        if not report["passed_1e-5"]:
            raise ValueError(f"original prefix mismatch: {report}")
        return np.asarray(prefix_angular[7], dtype=np.float64), report
    finally:
        p.disconnect(physicsClientId=client)


def quaternion_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(np.dot(left, right))), 0.0, 1.0))))


def box_orientation_error_deg(left_yaw_deg: float, right_yaw_deg: float) -> float:
    """Orientation error for a rectangular box, modulo its 180-degree symmetry."""
    return abs(float((left_yaw_deg - right_yaw_deg + 90.0) % 180.0 - 90.0))


def canonical_gt_primitives(blueprint) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    objects = {obj.name: obj for obj in blueprint.objects}
    family = blueprint.metadata["pilot_family"]

    def bounds(obj):
        center = np.asarray(obj.position, dtype=np.float64)
        half = np.asarray([obj.size["hx"], obj.size["hy"], obj.size["hz"]], dtype=np.float64)
        return center - half, center + half

    def item(name: str, center: np.ndarray, half: np.ndarray, yaw_deg: float = 0.0):
        return {
            "name": name,
            "position_m": center.tolist(),
            "half_extents_m": half.tolist(),
            "orientation_xyzw": [0.0, 0.0, math.sin(math.radians(yaw_deg) / 2), math.cos(math.radians(yaw_deg) / 2)],
        }

    if family == "deflector":
        obj = objects["puck_barrier"]
        primitives = [item(
            "estimated_deflector",
            np.asarray(obj.position),
            np.asarray([obj.size["hx"], obj.size["hy"], obj.size["hz"]]),
            float(obj.orientation_euler_deg[2]),
        )]
        measurements = {
            "ground_top_z_m": 0.0,
            "normal_yaw_deg": float(obj.orientation_euler_deg[2]),
        }
    elif family == "support_edge":
        primitives = []
        for source_name, target_name in (
            ("left_platform", "estimated_left_platform"),
            ("right_platform", "estimated_right_platform"),
        ):
            obj = objects[source_name]
            primitives.append(item(
                target_name,
                np.asarray(obj.position),
                np.asarray([obj.size["hx"], obj.size["hy"], obj.size["hz"]]),
            ))
        left_high = bounds(objects["left_platform"])[1][0]
        right_low = bounds(objects["right_platform"])[0][0]
        measurements = {
            "ground_top_z_m": 0.0,
            "gap_edges_m": [float(left_high), float(right_low)],
            "gap_width_m": float(right_low - left_high),
            "top_z_m": float(bounds(objects["left_platform"])[1][2]),
        }
    elif family == "aperture":
        left_low = min(bounds(objects["door_wall_left"])[0][1], bounds(objects["door_frame_left"])[0][1])
        left_high = max(bounds(objects["door_wall_left"])[1][1], bounds(objects["door_frame_left"])[1][1])
        right_low = min(bounds(objects["door_wall_right"])[0][1], bounds(objects["door_frame_right"])[0][1])
        right_high = max(bounds(objects["door_wall_right"])[1][1], bounds(objects["door_frame_right"])[1][1])
        front_x = min(bounds(obj)[0][0] for obj in objects.values() if obj.name != "pilot_ball")
        back_x = max(bounds(obj)[1][0] for obj in objects.values() if obj.name != "pilot_ball")
        top_z = max(bounds(obj)[1][2] for obj in objects.values() if obj.name != "pilot_ball")
        opening_height = min(bounds(objects["door_frame_lintel"])[0][2], bounds(objects["door_wall_header"])[0][2])
        half_x = 0.5 * (back_x - front_x)
        center_x = 0.5 * (front_x + back_x)
        primitives = [
            item(
                "estimated_aperture_left",
                np.asarray([center_x, 0.5 * (left_low + left_high), 0.5 * opening_height]),
                np.asarray([half_x, 0.5 * (left_high - left_low), 0.5 * opening_height]),
            ),
            item(
                "estimated_aperture_right",
                np.asarray([center_x, 0.5 * (right_low + right_high), 0.5 * opening_height]),
                np.asarray([half_x, 0.5 * (right_high - right_low), 0.5 * opening_height]),
            ),
            item(
                "estimated_aperture_header",
                np.asarray([center_x, 0.5 * (left_low + right_high), 0.5 * (opening_height + top_z)]),
                np.asarray([half_x, 0.5 * (right_high - left_low), 0.5 * (top_z - opening_height)]),
            ),
        ]
        measurements = {
            "ground_top_z_m": 0.0,
            "front_surface_x_m": float(front_x),
            "gap_edges_y_m": [float(left_high), float(right_low)],
            "opening_width_m": float(right_low - left_high),
            "opening_height_m": float(opening_height),
            "top_z_m": float(top_z),
            "thickness_m": float(back_x - front_x),
        }
    else:
        raise ValueError(family)
    return primitives, measurements


def geometry_errors(blueprint, estimated_payload: dict[str, Any]) -> dict[str, Any]:
    gt_primitives, gt_measurements = canonical_gt_primitives(blueprint)
    estimated = {
        row["name"]: row for row in estimated_payload["primitives"] if row["name"] != "estimated_ground"
    }
    rows = []
    for gt in gt_primitives:
        row = estimated[gt["name"]]
        center_error = float(np.linalg.norm(np.asarray(row["position_m"]) - np.asarray(gt["position_m"])))
        half_est = np.asarray(row["size"]["half_extents_m"], dtype=np.float64)
        half_gt = np.asarray(gt["half_extents_m"], dtype=np.float64)
        size_error = float(np.linalg.norm(2.0 * (half_est - half_gt)))
        orientation_error = box_orientation_error_deg(
            float(row.get("orientation_yaw_deg", 0.0)),
            float(2.0 * math.degrees(math.atan2(gt["orientation_xyzw"][2], gt["orientation_xyzw"][3]))),
        )
        rows.append({
            "name": gt["name"],
            "center_error_m": center_error,
            "full_size_l2_error_m": size_error,
            "orientation_error_deg": orientation_error,
            "estimated": row,
            "gt_canonical": gt,
        })
    estimated_measurements = {
        "ground_top_z_m": estimated_payload["ground_fit"]["ground_top_z_m"],
        **{
            key: value for key, value in estimated_payload["family_fit"].items()
            if key in gt_measurements
        },
    }
    measurement_errors = {}
    for key, target in gt_measurements.items():
        value = estimated_measurements.get(key)
        if value is None:
            continue
        if isinstance(target, list):
            error = np.asarray(value, dtype=np.float64) - np.asarray(target, dtype=np.float64)
            measurement_errors[key] = {
                "estimated": value,
                "gt": target,
                "error": error.tolist(),
                "l2_error": float(np.linalg.norm(error)),
            }
        else:
            measurement_errors[key] = {
                "estimated": float(value),
                "gt": float(target),
                "absolute_error": abs(float(value) - float(target)),
            }
    return {
        "canonicalization": "aperture wall/frame boxes are compared as an equivalent left/right/header union",
        "per_primitive": rows,
        "mean_center_error_m": float(np.mean([row["center_error_m"] for row in rows])),
        "max_center_error_m": float(np.max([row["center_error_m"] for row in rows])),
        "mean_full_size_l2_error_m": float(np.mean([row["full_size_l2_error_m"] for row in rows])),
        "mean_orientation_error_deg": float(np.mean([row["orientation_error_deg"] for row in rows])),
        "measurements": measurement_errors,
    }


def trajectory_metrics(predicted: dict[str, Any], target_position: np.ndarray, target_velocity: np.ndarray) -> dict[str, Any]:
    position_error = np.linalg.norm(predicted["positions"] - target_position, axis=1)
    velocity_error = np.linalg.norm(predicted["linear_velocities"] - target_velocity, axis=1)
    return {
        "ADE_m": float(np.mean(position_error)),
        "FDE_m": float(position_error[-1]),
        "max_error_m": float(np.max(position_error)),
        "position_error_by_future_frame_m": position_error.tolist(),
        "velocity_error_by_future_frame_mps": velocity_error.tolist(),
        "mean_velocity_error_mps": float(np.mean(velocity_error)),
        "event": predicted["event"],
        "api_step_calls": predicted["api_step_calls"],
        "wall_seconds": predicted["elapsed_seconds"],
    }


def save_rollout(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=result["positions"].astype(np.float32),
        linear_velocities=result["linear_velocities"].astype(np.float32),
        angular_velocities=result["angular_velocities"].astype(np.float32),
    )


def evaluate_case(recheck, pilot, generator, record: dict[str, Any], data_root: Path,
                  vision_output: Path, primitive_root: Path, output: Path) -> dict[str, Any]:
    case_id = str(record["key"])
    case, seed = recheck.reconstruct_case(pilot, generator, record)
    raw = recheck.load_raw_state(data_root / "samples" / case_id)
    dynamic_index = int(raw["dynamic_index"])
    gt_p7 = np.asarray(raw["positions"][7, dynamic_index], dtype=np.float64)
    gt_q7 = np.asarray(raw["quats"][7, dynamic_index], dtype=np.float64)
    gt_v7 = np.asarray(raw["linear_velocities"][7, dynamic_index], dtype=np.float64)
    target_position = np.asarray(raw["positions"][8:49, dynamic_index], dtype=np.float64)
    target_velocity = np.asarray(raw["linear_velocities"][8:49, dynamic_index], dtype=np.float64)
    gt_omega, prefix_report = recover_gt_omega(recheck, generator, case, seed, raw)
    with np.load(vision_output / "estimates" / case_id / "estimate.npz", allow_pickle=False) as archive:
        estimated_p7 = archive["estimated_p7"].astype(np.float64)
        estimated_v7 = archive["estimated_v7"].astype(np.float64)
    primitive_payload = load_json(primitive_root / "cases" / case_id / "primitives.json")
    primitives = primitive_payload["primitives"]

    state_evaluation = {
        "estimated_p7_m": estimated_p7.tolist(),
        "gt_p7_m": gt_p7.tolist(),
        "position_error_m": float(np.linalg.norm(estimated_p7 - gt_p7)),
        "position_component_error_m": (estimated_p7 - gt_p7).tolist(),
        "estimated_v7_mps": estimated_v7.tolist(),
        "gt_v7_mps": gt_v7.tolist(),
        **velocity_errors(estimated_v7, gt_v7),
    }
    geometry_evaluation = geometry_errors(case.blueprint, primitive_payload)
    modes = {}
    rollout_cache = {}
    for combination, (state_source, geometry_source) in COMBINATIONS.items():
        p7 = gt_p7 if state_source == "gt" else estimated_p7
        v7 = gt_v7 if state_source == "gt" else estimated_v7
        for omega_policy in OMEGA_POLICIES:
            if omega_policy == "gt_observed":
                omega = gt_omega
            elif omega_policy == "zero":
                omega = np.zeros(3, dtype=np.float64)
            else:
                omega = rolling_omega(v7)
            mode_name = f"{combination}__omega_{omega_policy}"
            predicted = rollout(
                generator,
                case,
                seed,
                geometry_source,
                primitives,
                {
                    "position": p7,
                    "quaternion": gt_q7,
                    "linear_velocity": v7,
                    "angular_velocity": omega,
                },
            )
            rollout_cache[mode_name] = predicted
            metrics = trajectory_metrics(predicted, target_position, target_velocity)
            metrics.update({
                "combination": combination,
                "state_source": state_source,
                "geometry_source": geometry_source,
                "omega_policy": omega_policy,
                "initial_position_m": p7.tolist(),
                "initial_velocity_mps": v7.tolist(),
                "initial_angular_velocity_radps": omega.tolist(),
                "initial_angular_velocity_error_radps": float(np.linalg.norm(omega - gt_omega)),
                "orientation_policy": "GT q7; collision object is a rotationally symmetric sphere",
            })
            modes[mode_name] = metrics
            save_rollout(output / "rollouts" / case_id / f"{mode_name}.npz", predicted)

    oracle_name = "A_gt_state_gt_geometry__omega_gt_observed"
    oracle_event = modes[oracle_name]["event"]
    event_index = oracle_event["first_interaction_future_index_30hz"]
    for mode_name, metrics in modes.items():
        metrics["contact_outcome_match_oracle"] = metrics["event"]["category"] == oracle_event["category"]
        if event_index is None:
            metrics["post_contact_velocity_error"] = {
                "status": "NOT_APPLICABLE",
                "reason": "oracle has no interaction event within RGB8-RGB48",
            }
        else:
            errors = np.linalg.norm(
                rollout_cache[mode_name]["linear_velocities"][event_index:] - target_velocity[event_index:], axis=1
            )
            metrics["post_contact_velocity_error"] = {
                "status": "EXECUTED",
                "oracle_event_future_index": int(event_index),
                "mean_mps": float(np.mean(errors)),
                "max_mps": float(np.max(errors)),
                "final_mps": float(errors[-1]),
                "by_frame_mps": errors.tolist(),
            }

    primary = {letter: modes[f"{name}__omega_gt_observed"] for letter, name in zip("ABCD", COMBINATIONS)}
    decomposition = {
        "B_minus_A_ADE_m": primary["B"]["ADE_m"] - primary["A"]["ADE_m"],
        "C_minus_A_ADE_m": primary["C"]["ADE_m"] - primary["A"]["ADE_m"],
        "D_minus_A_ADE_m": primary["D"]["ADE_m"] - primary["A"]["ADE_m"],
        "D_interaction_residual_ADE_m": primary["D"]["ADE_m"] - primary["A"]["ADE_m"]
            - (primary["B"]["ADE_m"] - primary["A"]["ADE_m"])
            - (primary["C"]["ADE_m"] - primary["A"]["ADE_m"]),
        "omega_zero_minus_gt_D_ADE_m": modes["D_estimated_state_estimated_geometry__omega_zero"]["ADE_m"]
            - primary["D"]["ADE_m"],
        "omega_rolling_minus_gt_D_ADE_m": modes["D_estimated_state_estimated_geometry__omega_pure_rolling_hypothesis"]["ADE_m"]
            - primary["D"]["ADE_m"],
    }
    result = {
        "schema": "context_rgb8_pybullet_case_evaluation_v1",
        "status": "EXECUTED",
        "case_id": case_id,
        "family": record["family"],
        "group_id": record["group_id"],
        "geometry_value": record["geometry_value"],
        "geometry_units": record["geometry_units"],
        "state_evaluation": state_evaluation,
        "geometry_evaluation": geometry_evaluation,
        "gt_omega7_radps": gt_omega.tolist(),
        "prefix_replay": prefix_report,
        "modes": modes,
        "decomposition": decomposition,
        "target_future_used_for_estimation": False,
        "gt_used_only_after_vision_and_primitive_outputs_were_frozen": True,
    }
    dump_json(output / "cases" / case_id / "evaluation.json", result)
    return result


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    mode_names = list(cases[0]["modes"])
    modes = {}
    for mode_name in mode_names:
        rows = [case["modes"][mode_name] for case in cases]
        modes[mode_name] = {
            "ADE_mean_m": float(np.mean([row["ADE_m"] for row in rows])),
            "ADE_median_m": float(np.median([row["ADE_m"] for row in rows])),
            "FDE_mean_m": float(np.mean([row["FDE_m"] for row in rows])),
            "max_error_mean_m": float(np.mean([row["max_error_m"] for row in rows])),
            "contact_outcome_accuracy_vs_oracle": float(np.mean([row["contact_outcome_match_oracle"] for row in rows])),
        }
    state = {
        "position_error_mean_m": float(np.mean([case["state_evaluation"]["position_error_m"] for case in cases])),
        "position_error_median_m": float(np.median([case["state_evaluation"]["position_error_m"] for case in cases])),
        "velocity_vector_error_mean_mps": float(np.mean([case["state_evaluation"]["vector_error_mps"] for case in cases])),
        "velocity_magnitude_error_mean_mps": float(np.mean([case["state_evaluation"]["magnitude_error_mps"] for case in cases])),
        "velocity_direction_error_mean_deg": float(np.mean([
            case["state_evaluation"]["direction_error_deg"] for case in cases
            if case["state_evaluation"]["direction_error_deg"] is not None
        ])),
    }
    geometry = {
        "primitive_center_error_mean_m": float(np.mean([case["geometry_evaluation"]["mean_center_error_m"] for case in cases])),
        "primitive_size_error_mean_m": float(np.mean([case["geometry_evaluation"]["mean_full_size_l2_error_m"] for case in cases])),
        "orientation_error_mean_deg": float(np.mean([case["geometry_evaluation"]["mean_orientation_error_deg"] for case in cases])),
    }
    by_family = {}
    for family in ("aperture", "deflector", "support_edge"):
        subset = [case for case in cases if case["family"] == family]
        by_family[family] = {
            "case_count": len(subset),
            "state_position_error_mean_m": float(np.mean([row["state_evaluation"]["position_error_m"] for row in subset])),
            "state_velocity_error_mean_mps": float(np.mean([row["state_evaluation"]["vector_error_mps"] for row in subset])),
            "geometry_center_error_mean_m": float(np.mean([row["geometry_evaluation"]["mean_center_error_m"] for row in subset])),
            "A_ADE_mean_m": float(np.mean([row["modes"]["A_gt_state_gt_geometry__omega_gt_observed"]["ADE_m"] for row in subset])),
            "B_ADE_mean_m": float(np.mean([row["modes"]["B_estimated_state_gt_geometry__omega_gt_observed"]["ADE_m"] for row in subset])),
            "C_ADE_mean_m": float(np.mean([row["modes"]["C_gt_state_estimated_geometry__omega_gt_observed"]["ADE_m"] for row in subset])),
            "D_ADE_mean_m": float(np.mean([row["modes"]["D_estimated_state_estimated_geometry__omega_gt_observed"]["ADE_m"] for row in subset])),
        }
    return {"state": state, "geometry": geometry, "modes": modes, "by_family": by_family}


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.data_root.resolve()
    vision_output = args.vision_output.resolve()
    primitive_root = args.primitive_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True)
    fit_report = load_json(primitive_root / "fit_report.json")
    if fit_report.get("status") != "EXECUTED" or fit_report.get("executed_count") != 36:
        raise ValueError("all 36 primitive fits must pass before evaluation")
    manifest = load_json(data_root / "pilot_manifest.json")
    records = sorted(manifest["records"], key=lambda row: row["key"])
    if len(records) != 36:
        raise ValueError("expected fixed 36-case pilot")
    recheck, pilot, generator = load_engine()
    cases = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        case = evaluate_case(
            recheck, pilot, generator, record, data_root, vision_output, primitive_root, output
        )
        cases.append(case)
        d_ade = case["modes"]["D_estimated_state_estimated_geometry__omega_gt_observed"]["ADE_m"]
        print(f"EVAL_OK {index:02d}/36 {case['case_id']} D_ADE={d_ade:.6f}m", flush=True)
    report = {
        "schema": "context_rgb8_pybullet_evaluation_report_v1",
        "status": "EXECUTED",
        "case_count": len(cases),
        "rollout_count": len(cases) * len(COMBINATIONS) * len(OMEGA_POLICIES),
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate": aggregate(cases),
        "case_reports": [str(Path("cases") / case["case_id"] / "evaluation.json") for case in cases],
        "protocol": {
            "combinations": COMBINATIONS,
            "omega_policies": OMEGA_POLICIES,
            "future_frames": FUTURE_FRAMES,
            "output_fps": FPS,
            "fixed_time_step_s": 1.0 / SIM_HZ,
            "api_step_calls_per_output": STEPS_PER_OUTPUT,
            "contact_log_resolution_s": 1.0 / SIM_HZ,
            "engine_internal_substeps_not_exposed_in_contact_log": True,
            "gt_q7_policy": "used in all modes because the dynamic body is a rotationally symmetric sphere",
            "pure_rolling_is_grounded_hypothesis_not_truth": True,
            "cpu_only": True,
            "threads": 2,
        },
    }
    dump_json(output / "evaluation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--vision-output", type=Path, required=True)
    parser.add_argument("--primitive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "case_count": result["case_count"],
        "rollout_count": result["rollout_count"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))
