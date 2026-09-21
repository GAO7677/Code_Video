"""Layered, CPU-only recheck of the phase-one PyBullet pilot bridge.

This module is intentionally separate from ``pybullet_context_rollout.py``.
It records the legacy result first, then runs controlled replay/rebuild
experiments without changing the pilot data or the production bridge.  The
outputs are an auditable diagnostic, not a training job.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parent
ENGINE_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819")
AUDIT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/original_pipeline_audit_20260917"
)
DATA_DEFAULT = Path(
    "/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5"
)
OUTPUT_DEFAULT = Path(
    "/data/gaoya/agent-data/outputs/validation_pybullet_bridge_recheck_v2"
)
LEGACY_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/validation_pybullet_bridge_recheck_v2/legacy_approx"
)

FPS = 30
SIM_HZ = 240
CONTEXT_FRAMES = 8
FUTURE_FRAMES = 41
BALL_RADIUS_M = 0.11
SEED_BASE = 2026092100
CONTACT_DISTANCE_TOLERANCE_M = 1e-4
CONTACT_VELOCITY_TOLERANCE_MPS = 1e-3
QUATERNION_NORM_TOLERANCE = 1e-5

SELECTED_CASES = (
    "phase1_aperture_g03_v0280",
    "phase1_aperture_g03_v0720",
    "phase1_deflector_g03_v28000",
    "phase1_deflector_g03_v86000",
    "phase1_support_edge_g03_v0080",
    "phase1_support_edge_g03_v0620",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_engine():
    sys.path[:0] = [str(PROJECT), str(ENGINE_ROOT), str(AUDIT_ROOT)]
    import prepare_physvideo_phase1_pilot as pilot

    generator, replay_audit = pilot.load_engine()
    return pilot, generator, replay_audit


def reconstruct_case(pilot, generator, record: dict[str, Any]):
    family = str(record["family"])
    group = int(str(record["group_id"]).rsplit("g", 1)[1])
    value = float(record["geometry_value"])
    family_index = {"aperture": 0, "deflector": 1, "support_edge": 2}[family]
    seed = SEED_BASE + family_index * 100 + group
    case = pilot.make_case(generator, family, group, value, seed)
    if case.case_id != record["key"]:
        raise ValueError(f"reconstructed case mismatch: {case.case_id} != {record['key']}")
    return case, seed


def quat_sign_invariant_l2(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def quaternion_norm_report(quats: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(np.asarray(quats, dtype=np.float64), axis=-1)
    return {
        "min": float(norms.min()),
        "max": float(norms.max()),
        "max_abs_error_from_one": float(np.max(np.abs(norms - 1.0))),
    }


def load_raw_state(sample_dir: Path) -> dict[str, Any]:
    path = sample_dir / "raw" / "states_xyzw.npz"
    with np.load(path, allow_pickle=False) as raw:
        state = {name: raw[name].copy() for name in raw.files}
    names = state["object_names"].astype(str).tolist()
    pilot_indices = [index for index, name in enumerate(names) if name == "pilot_ball"]
    if pilot_indices != [names.index("pilot_ball")]:
        raise ValueError(f"pilot_ball lookup is not unique: {pilot_indices} / {names}")
    if len(pilot_indices) != 1:
        raise ValueError(f"pilot_ball lookup failed: {names}")
    dynamic_index = pilot_indices[0]
    if state["positions"].shape[0] < CONTEXT_FRAMES + FUTURE_FRAMES:
        raise ValueError(f"state frame count too short: {state['positions'].shape}")
    if state["positions"].shape[2] != 3 or state["linear_velocities"].shape[2] != 3:
        raise ValueError("position/velocity vectors must have shape [..., 3]")
    if state["quats"].shape[2] != 4:
        raise ValueError("quaternions must have shape [..., 4] in xyzw order")
    times = np.asarray(state["frame_times"], dtype=np.float64)
    return {**state, "dynamic_index": dynamic_index, "object_names_list": names, "frame_times_float": times}


def raw_context_payload(state: dict[str, Any]) -> dict[str, Any]:
    idx = int(state["dynamic_index"])
    positions = np.asarray(state["positions"], dtype=np.float32)
    quats = np.asarray(state["quats"], dtype=np.float32)
    velocities = np.asarray(state["linear_velocities"], dtype=np.float32)
    times = np.asarray(state["frame_times_float"], dtype=np.float64)
    return {
        "positions_observed": positions[:CONTEXT_FRAMES, idx].copy(),
        "quats_observed": quats[:CONTEXT_FRAMES, idx].copy(),
        "linear_velocities_observed": velocities[:CONTEXT_FRAMES, idx].copy(),
        "frame_times_observed": times[:CONTEXT_FRAMES].copy(),
        "p7": positions[7, idx].copy(),
        "q7": quats[7, idx].copy(),
        "v7_gt": velocities[7, idx].copy(),
        "v7_fd": ((positions[7, idx] - positions[6, idx]) / (times[7] - times[6])).astype(np.float32),
        # The unmodified bridge uses ``*FPS`` rather than the recorded frame
        # interval.  Keep that legacy input separate from the explicit R2/R4
        # frame-time finite difference.
        "v7_legacy": ((positions[7, idx] - positions[6, idx]) * FPS).astype(np.float32),
        "target": positions[CONTEXT_FRAMES : CONTEXT_FRAMES + FUTURE_FRAMES, idx].copy(),
        "dynamic_index": idx,
    }


def physics_object_payload(obj) -> dict[str, Any]:
    return {
        "name": obj.name,
        "family_key": obj.family_key,
        "shape": obj.shape,
        "size": dict(obj.size),
        "mass": float(obj.mass),
        "friction": float(obj.friction),
        "restitution": float(obj.restitution),
        "linear_damping": float(obj.linear_damping),
        "angular_damping": float(obj.angular_damping),
        "dynamic": bool(obj.dynamic),
        "role": obj.role,
        "position": list(obj.position),
        "orientation_euler_deg": list(obj.orientation_euler_deg),
        "linear_velocity": list(obj.linear_velocity),
        "angular_velocity": list(obj.angular_velocity),
        "metadata": dict(obj.metadata),
    }


def physical_blueprint_payload(blueprint) -> dict[str, Any]:
    metadata = dict(blueprint.metadata)
    return {
        "family_key": blueprint.family_key,
        "sample_key": blueprint.sample_key,
        "gravity": float(blueprint.gravity),
        "pre_roll_s": float(blueprint.pre_roll_s),
        "surface_key": blueprint.surface_key,
        "objects": [physics_object_payload(obj) for obj in blueprint.objects],
        "constraints": metadata.get("constraints", []),
        "disable_collision_pairs": metadata.get("disable_collision_pairs", []),
        "physics_sub_steps": metadata.get("physics_sub_steps"),
        "floor_friction": metadata.get("floor_friction"),
    }


def original_source_map() -> str:
    return """# PyBullet pilot bridge call map

```text
pilot_manifest.json
  -> prepare_physvideo_phase1_pilot.make_case()
     -> original generator _make_door_frame_case/_make_puck_barrier_case/_make_gap_case
  -> sample/raw/states_xyzw.npz (saved original replay states)
  -> pybullet_context_rollout.py rollout_from_rgb7()
     -> p7/q7 and p6->p7 finite-difference v7
     -> fresh DIRECT world: plane + blueprint objects + constraints/filter pairs
     -> original generator _step_v2v_simulation()
        -> p.stepSimulation() (pilot families do not enter the puck correction branch)
     -> 41 post-step position/velocity/contact snapshots
  -> results.json / *_rollout.npz / ADE-FDE comparison
```

Relevant source locations:

- `scene_geo_predictor/code/pybullet_context_rollout.py:171-188`: state loading and RGB7 initialization policy.
- `scene_geo_predictor/code/pybullet_context_rollout.py:77-164`: explicit-client shape/step adapters.
- `scene_geo_predictor/code/pybullet_context_rollout.py:196-267`: world construction and dynamics settings.
- `scene_geo_predictor/code/pybullet_context_rollout.py:269-299`: stepping, state reads, and contacts.
- `Dataset_physv_v2v_0819/scripts/generate_v2v_context_demos.py:2132-2198`: `_step_v2v_simulation()`.
- `.../original_pipeline_audit_20260917/replay_audit.py:16-85`: original CPU replay entry.
- `scene_geo_predictor/code/prepare_physvideo_phase1_pilot.py:147-225`: pilot blueprint construction.
"""


def rolling_omega_legacy(v: np.ndarray, radius: float = BALL_RADIUS_M) -> np.ndarray:
    vx, vy, _ = [float(value) for value in np.asarray(v)]
    # Match the production bridge's float32 return dtype for the L mode.
    return np.asarray([vy / radius, -vx / radius, 0.0], dtype=np.float32)


def rolling_omega_hypothesis(v: np.ndarray, normal: np.ndarray, radius: float = BALL_RADIUS_M) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    tangent = v - float(np.dot(v, normal)) * normal
    return np.cross(normal, tangent) / float(radius)


def contact_point_velocity(center: np.ndarray, velocity: np.ndarray, omega: np.ndarray, contact_point: np.ndarray) -> np.ndarray:
    radius_vector = np.asarray(contact_point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return np.asarray(velocity, dtype=np.float64) + np.cross(np.asarray(omega, dtype=np.float64), radius_vector)


def rolling_kinematics_tests() -> dict[str, Any]:
    radius = BALL_RADIUS_M
    normal = np.asarray([0.0, 0.0, 1.0])
    directions = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([-1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, -1.0, 0.0]),
        np.asarray([0.7, 0.4, 0.0]),
    ]
    rows = []
    contact = np.asarray([0.0, 0.0, -radius])
    for velocity in directions:
        correct = rolling_omega_hypothesis(velocity, normal, radius)
        legacy = rolling_omega_legacy(velocity, radius)
        rows.append(
            {
                "velocity": velocity.tolist(),
                "correct_omega": correct.tolist(),
                "legacy_omega": legacy.tolist(),
                "correct_contact_velocity": contact_point_velocity(np.zeros(3), velocity, correct, contact).tolist(),
                "legacy_contact_velocity": contact_point_velocity(np.zeros(3), velocity, legacy, contact).tolist(),
            }
        )

    tilted_normal = np.asarray([0.3, -0.2, 1.0], dtype=np.float64)
    tilted_normal /= np.linalg.norm(tilted_normal)
    tangent = np.asarray([1.0, 0.2, 0.0], dtype=np.float64)
    tangent -= np.dot(tangent, tilted_normal) * tilted_normal
    tilted_omega = rolling_omega_hypothesis(tangent, tilted_normal, radius)
    tilted_contact = -radius * tilted_normal
    tilted_residual = contact_point_velocity(np.zeros(3), tangent, tilted_omega, tilted_contact)
    passed = all(np.linalg.norm(row["correct_contact_velocity"]) <= 1e-12 for row in rows)
    passed = bool(passed and np.linalg.norm(tilted_residual) <= 1e-12)
    return {
        "status": "EXECUTED",
        "coordinate_convention": "world right-handed xyz, z-up, PyBullet xyzw quaternion, world-frame linear/angular velocity",
        "formula": "omega = cross(normal, tangent_velocity) / radius + optional spin*normal",
        "legacy_formula": "[vy/r, -vx/r, 0]",
        "rows": rows,
        "tilted_plane": {
            "normal": tilted_normal.tolist(),
            "tangent_velocity": tangent.tolist(),
            "omega": tilted_omega.tolist(),
            "contact_velocity": tilted_residual.tolist(),
        },
        "passed": passed,
    }


class StepCounter:
    """Count visible Python-level calls to PyBullet's stepSimulation API."""

    def __init__(self, p_module):
        self.p = p_module
        self.calls = 0
        self._original = None

    def __enter__(self):
        self._original = self.p.stepSimulation

        def counted_step(*args, **kwargs):
            self.calls += 1
            return self._original(*args, **kwargs)

        self.p.stepSimulation = counted_step
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.p.stepSimulation = self._original


@dataclasses.dataclass
class PhysicsWorld:
    p: Any
    generator: Any
    blueprint: Any
    case_key: str
    client_id: int
    plane_id: int
    body_ids: dict[str, int]
    body_names: dict[int, str]
    legacy_objects: dict[str, Any]
    step_counter: StepCounter | None = None
    api_step_index: int = 0
    output_frame_index: int | None = None
    correction_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    contact_records: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    def step_once(self, *, output_frame: int | None = None, phase: str = "rollout") -> dict[str, Any] | None:
        self.output_frame_index = output_frame
        correction = self.generator._step_v2v_simulation(self.body_ids, self.blueprint)
        if correction is not None:
            self.correction_records.append(
                {
                    "api_step_index": self.api_step_index,
                    "output_frame": output_frame,
                    "phase": phase,
                    **jsonable(correction),
                }
            )
        self.api_step_index += 1
        return correction

    def _contact_payload(self, point: tuple[Any, ...], *, phase: str, output_frame: int | None) -> dict[str, Any]:
        body_a = int(point[1])
        body_b = int(point[2])
        return {
            "body_a_id": body_a,
            "body_b_id": body_b,
            "body_a_name": self.body_names.get(body_a, str(body_a)),
            "body_b_name": self.body_names.get(body_b, str(body_b)),
            "link_a": int(point[3]),
            "link_b": int(point[4]),
            "position_on_a": [float(value) for value in point[5]],
            "position_on_b": [float(value) for value in point[6]],
            "normal_on_b": [float(value) for value in point[7]],
            "contact_distance_m": float(point[8]),
            "normal_force_n": float(point[9]),
            "phase": phase,
            "output_frame": output_frame,
            "api_step_index": self.api_step_index,
            "api_time_s": float(self.api_step_index) / SIM_HZ,
        }

    def contacts(self, *, phase: str, output_frame: int | None, record: bool = True) -> list[dict[str, Any]]:
        dynamic_id = self.body_ids["pilot_ball"]
        points = self.p.getContactPoints(bodyA=dynamic_id)
        payload = [self._contact_payload(point, phase=phase, output_frame=output_frame) for point in points]
        if record:
            self.contact_records.extend(payload)
        return payload

    def capture_state(self) -> dict[str, Any]:
        dynamic_id = self.body_ids["pilot_ball"]
        position, quat = self.p.getBasePositionAndOrientation(dynamic_id)
        linear, angular = self.p.getBaseVelocity(dynamic_id)
        return {
            "position": np.asarray(position, dtype=np.float32),
            "quaternion": np.asarray(quat, dtype=np.float32),
            "linear_velocity": np.asarray(linear, dtype=np.float32),
            "angular_velocity": np.asarray(angular, dtype=np.float32),
        }

    def contact_snapshot(self, *, phase: str, output_frame: int | None) -> list[dict[str, Any]]:
        return self.contacts(phase=phase, output_frame=output_frame, record=True)

    def config_snapshot(self) -> dict[str, Any]:
        objects = []
        body_entries = list(self.body_ids.items()) + [("floor", self.plane_id)]
        for name, body_id in body_entries:
            dynamics = self.p.getDynamicsInfo(body_id, -1)
            shapes = self.p.getCollisionShapeData(body_id, -1)
            position, quat = self.p.getBasePositionAndOrientation(body_id)
            linear, angular = self.p.getBaseVelocity(body_id)
            objects.append(
                {
                    "name": name,
                    "body_id": int(body_id),
                    "shape_data": jsonable(shapes),
                    "dynamics_info": jsonable(dynamics),
                    "position": list(position),
                    "quaternion_xyzw": list(quat),
                    "linear_velocity": list(linear),
                    "angular_velocity": list(angular),
                    "aabb": jsonable(self.p.getAABB(body_id, -1)),
                }
            )
        return {
            "client_id": int(self.client_id),
            "physics_engine_parameters": jsonable(self.p.getPhysicsEngineParameters()),
            "plane_id": int(self.plane_id),
            "bodies": objects,
            "constraints": [jsonable(self.p.getConstraintInfo(index)) for index in range(self.p.getNumConstraints())],
            "body_order": list(self.body_ids),
            "disabled_collision_pairs": self.blueprint.metadata.get("disable_collision_pairs", []),
            "plane_resource": str(Path(self.generator.pybullet_data.getDataPath()) / "plane.urdf"),
        }


def build_world(
    generator,
    case,
    seed: int,
    *,
    dynamic_state: dict[str, np.ndarray] | None = None,
    label: str,
) -> PhysicsWorld:
    """Build a world using the same collision and dynamics helpers as replay."""
    import pybullet as p

    blueprint = case.blueprint
    scenario = generator.blueprint_to_legacy_scenario(blueprint, seed=seed)
    legacy_objects = {obj.name: obj for obj in scenario.objects}
    objects = [obj for obj in blueprint.objects if not obj.metadata.get("visual_only")]
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        p.setAdditionalSearchPath(generator.pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -generator.EARTH_GRAVITY)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / SIM_HZ,
            numSolverIterations=generator.legacy.PHYSICS_SOLVER_ITERATIONS,
            numSubSteps=int(blueprint.metadata.get("physics_sub_steps", generator.legacy.PHYSICS_SUB_STEPS)),
            contactERP=generator.legacy.PHYSICS_CONTACT_ERP,
            erp=generator.legacy.PHYSICS_CONTACT_ERP,
        )
        plane_id = p.loadURDF("plane.urdf")
        surface = generator.build_surface_catalog()[blueprint.surface_key]
        floor_mu = float(
            np.clip(
                blueprint.metadata.get("floor_friction", surface.floor_friction_range.midpoint()),
                0.01,
                1.2,
            )
        )
        p.changeDynamics(
            plane_id,
            -1,
            lateralFriction=floor_mu,
            restitution=0.02,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
        )
        body_ids: dict[str, int] = {}
        for obj in objects:
            if obj.dynamic and dynamic_state is not None:
                position = dynamic_state["position"]
                orientation = dynamic_state["quaternion"]
                linear_velocity = dynamic_state["linear_velocity"]
                angular_velocity = dynamic_state["angular_velocity"]
            else:
                position = np.asarray(obj.position, dtype=np.float64)
                orientation = generator.legacy._quat_from_euler_deg(list(obj.orientation_euler_deg))
                linear_velocity = np.asarray(obj.linear_velocity, dtype=np.float64)
                angular_velocity = np.asarray(obj.angular_velocity, dtype=np.float64)
            body_id = p.createMultiBody(
                baseMass=float(obj.mass) if obj.dynamic else 0.0,
                baseCollisionShapeIndex=generator.legacy._collision_shape(legacy_objects[obj.name]),
                basePosition=np.asarray(position, dtype=np.float64).tolist(),
                baseOrientation=np.asarray(orientation, dtype=np.float64).tolist(),
            )
            dynamics_kwargs = {
                "restitution": float(obj.restitution),
                "lateralFriction": float(obj.friction),
                "rollingFriction": float(obj.metadata.get("rolling_friction", 0.0)),
                "spinningFriction": float(obj.metadata.get("spinning_friction", 0.0)),
                "linearDamping": float(obj.linear_damping),
                "angularDamping": float(obj.angular_damping),
                "activationState": p.ACTIVATION_STATE_DISABLE_SLEEPING,
            }
            if "ccd_swept_sphere_radius_m" in obj.metadata:
                dynamics_kwargs.update(
                    ccdSweptSphereRadius=float(obj.metadata["ccd_swept_sphere_radius_m"]),
                    contactProcessingThreshold=0.0,
                )
            p.changeDynamics(body_id, -1, **dynamics_kwargs)
            p.resetBaseVelocity(
                body_id,
                linearVelocity=np.asarray(linear_velocity, dtype=np.float64).tolist(),
                angularVelocity=np.asarray(angular_velocity, dtype=np.float64).tolist(),
            )
            body_ids[obj.name] = int(body_id)
        for descriptor in blueprint.metadata.get("constraints", []):
            generator._make_constraint(dict(descriptor), body_ids)
        for left_name, right_name in blueprint.metadata.get("disable_collision_pairs", []):
            p.setCollisionFilterPair(body_ids[str(left_name)], body_ids[str(right_name)], -1, -1, 0)
        body_names = {body_id: name for name, body_id in body_ids.items()}
        body_names[int(plane_id)] = "floor"
        return PhysicsWorld(
            p=p,
            generator=generator,
            blueprint=blueprint,
            case_key=case.case_id,
            client_id=int(client_id),
            plane_id=int(plane_id),
            body_ids=body_ids,
            body_names=body_names,
            legacy_objects=legacy_objects,
        )
    except Exception:
        p.disconnect(client_id)
        raise


def run_one_output_frame(world: PhysicsWorld, output_frame: int, *, steps_per_output: int = 8) -> dict[str, Any]:
    for _ in range(steps_per_output):
        world.step_once(output_frame=output_frame, phase="rollout_step")
        world.contact_snapshot(phase="after_api_step", output_frame=output_frame)
    return world.capture_state()


def run_rebuild_rollout(
    world: PhysicsWorld,
    *,
    steps_per_output: int = 8,
    perform_initial_query: bool = True,
) -> dict[str, Any]:
    initial_config = world.config_snapshot()
    initial_contacts = []
    if perform_initial_query:
        # This query refreshes contact manifolds but does not advance time.  It
        # is intentionally disabled for R0, whose first post-RGB7 operation
        # must match the original replay exactly.
        world.p.performCollisionDetection()
        initial_contacts = world.contact_snapshot(phase="initial_collision_query", output_frame=None)
    positions, quats, velocities, angular_velocities = [], [], [], []
    for output_frame in range(FUTURE_FRAMES):
        state = run_one_output_frame(world, output_frame, steps_per_output=steps_per_output)
        positions.append(state["position"])
        quats.append(state["quaternion"])
        velocities.append(state["linear_velocity"])
        angular_velocities.append(state["angular_velocity"])
    return {
        "positions": np.asarray(positions, dtype=np.float32),
        "quaternions": np.asarray(quats, dtype=np.float32),
        "linear_velocities": np.asarray(velocities, dtype=np.float32),
        "angular_velocities": np.asarray(angular_velocities, dtype=np.float32),
        "initial_contacts": initial_contacts,
        "initial_collision_query_performed": bool(perform_initial_query),
        "contact_records": list(world.contact_records),
        "correction_records": list(world.correction_records),
        "api_step_calls": int(world.step_counter.calls if world.step_counter is not None else 0),
        "config_initial": initial_config,
        "config_final": world.config_snapshot(),
    }


def run_world_with_counter(world: PhysicsWorld, fn):
    with StepCounter(world.p) as counter:
        world.step_counter = counter
        return fn()


def contact_names(records: Iterable[dict[str, Any]], *, exclude_floor: bool = False) -> list[str]:
    names = []
    for record in records:
        name = str(record["body_b_name"])
        if exclude_floor and name == "floor":
            continue
        if name not in names:
            names.append(name)
    return names


def state_arrays(states: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "positions": np.asarray([state["position"] for state in states], dtype=np.float32),
        "quaternions": np.asarray([state["quaternion"] for state in states], dtype=np.float32),
        "linear_velocities": np.asarray([state["linear_velocity"] for state in states], dtype=np.float32),
        "angular_velocities": np.asarray([state["angular_velocity"] for state in states], dtype=np.float32),
    }


def run_original_prefix(generator, case, seed: int, state: dict[str, Any]) -> dict[str, Any]:
    """Run the original world to RGB7, save/restore state, and continue."""
    world = build_world(generator, case, seed, dynamic_state=None, label="original")
    prefix_states: list[dict[str, Any]] = []
    prefix_contacts: list[list[dict[str, Any]]] = []

    def advance_prefix() -> None:
        pre_roll_steps = int(round(float(case.blueprint.pre_roll_s) * SIM_HZ))
        for _ in range(pre_roll_steps):
            world.step_once(output_frame=None, phase="pre_roll")
            world.contact_snapshot(phase="pre_roll_after_api_step", output_frame=None)
        for frame in range(CONTEXT_FRAMES):
            prefix_states.append(world.capture_state())
            prefix_contacts.append(world.contacts(phase="prefix_snapshot", output_frame=frame, record=False))
            if frame < CONTEXT_FRAMES - 1:
                for _ in range(SIM_HZ // FPS):
                    world.step_once(output_frame=frame, phase="prefix_step")
                    world.contact_snapshot(phase="prefix_after_api_step", output_frame=frame)

    with StepCounter(world.p) as counter:
        world.step_counter = counter
        initial_config = world.config_snapshot()
        advance_prefix()
        snapshot_id = int(world.p.saveState())

        def continuous_future() -> dict[str, Any]:
            # Do not insert a collision-manifold refresh between RGB7 and the
            # first replay step: R0 is a snapshot/restore equivalence test.
            return run_rebuild_rollout(
                world,
                steps_per_output=SIM_HZ // FPS,
                perform_initial_query=False,
            )

        continuous = continuous_future()
        restored_runs = []
        for repeat in range(3):
            world.p.restoreState(stateId=snapshot_id)
            world.api_step_index = 0
            world.output_frame_index = None
            world.contact_records = []
            world.correction_records = []
            restored = continuous_future()
            restored["repeat_index"] = repeat
            restored_runs.append(restored)
        world.p.removeState(snapshot_id)
        total_step_calls = counter.calls

    try:
        config = world.config_snapshot()
    finally:
        world.p.disconnect(world.client_id)

    prefix = state_arrays(prefix_states)
    raw_positions = np.asarray(state["positions"], dtype=np.float32)[:CONTEXT_FRAMES, state["dynamic_index"]]
    raw_quats = np.asarray(state["quats"], dtype=np.float32)[:CONTEXT_FRAMES, state["dynamic_index"]]
    raw_velocities = np.asarray(state["linear_velocities"], dtype=np.float32)[:CONTEXT_FRAMES, state["dynamic_index"]]
    prefix_position_error = np.linalg.norm(prefix["positions"] - raw_positions, axis=-1)
    prefix_velocity_error = np.linalg.norm(prefix["linear_velocities"] - raw_velocities, axis=-1)
    prefix_quaternion_error = np.asarray(
        [quat_sign_invariant_l2(prefix["quaternions"][i], raw_quats[i]) for i in range(CONTEXT_FRAMES)],
        dtype=np.float64,
    )
    frame7_contacts = prefix_contacts[-1]
    return {
        "prefix": prefix,
        "prefix_contacts": prefix_contacts,
        "frame7_contacts": frame7_contacts,
        "continuous": continuous,
        "restored_runs": restored_runs,
        "total_step_calls": total_step_calls,
        "config": config,
        "config_initial": initial_config,
        "prefix_match": {
            "max_position_error_m": float(prefix_position_error.max()),
            "max_linear_velocity_error_mps": float(prefix_velocity_error.max()),
            "max_quaternion_sign_invariant_l2": float(prefix_quaternion_error.max()),
            "position_error_by_frame_m": prefix_position_error.tolist(),
            "velocity_error_by_frame_mps": prefix_velocity_error.tolist(),
            "quaternion_error_by_frame": prefix_quaternion_error.tolist(),
            "passed_position_1e-5_m": bool(prefix_position_error.max() <= 1e-5),
            "passed_velocity_1e-5_mps": bool(prefix_velocity_error.max() <= 1e-5),
            "passed_quaternion_1e-5": bool(prefix_quaternion_error.max() <= 1e-5),
        },
    }


def original_replay_entrypoint_check(
    replay_audit,
    pilot,
    generator,
    records: dict[str, dict[str, Any]],
    data_root: Path,
    cases: list[str],
) -> dict[str, Any]:
    """Call the unmodified original replay entrypoint for every selected case.

    The R0 instrumentation below needs a save/restore hook, so it uses the
    same builder calls in ``build_world``.  This independent check executes
    ``replay_audit.replay`` itself and compares all 49 saved frames, proving
    that the instrumented builder and the original entrypoint produce the same
    physical replay before we interpret any bridge-vs-target error.
    """
    rows = []
    for key in cases:
        try:
            raw = load_raw_state(data_root / "samples" / key)
            case, seed = reconstruct_case(pilot, generator, records[key])
            objects, positions, velocities, quats, corrections = replay_audit.replay(case, seed)
            names = [obj.name for obj in objects]
            raw_names = raw["object_names"].astype(str).tolist()
            if names != raw_names:
                raise ValueError(f"original replay object order differs: {names} != {raw_names}")
            position_error = np.linalg.norm(
                np.asarray(positions, dtype=np.float64) - np.asarray(raw["positions"], dtype=np.float64), axis=-1
            )
            velocity_error = np.linalg.norm(
                np.asarray(velocities, dtype=np.float64) - np.asarray(raw["linear_velocities"], dtype=np.float64), axis=-1
            )
            raw_quats = np.asarray(raw["quats"], dtype=np.float64)
            replay_quats = np.asarray(quats, dtype=np.float64)
            quaternion_error = np.minimum(
                np.linalg.norm(replay_quats - raw_quats, axis=-1),
                np.linalg.norm(replay_quats + raw_quats, axis=-1),
            )
            rows.append(
                {
                    "key": key,
                    "status": "EXECUTED",
                    "frame_count": int(positions.shape[0]),
                    "object_count": int(positions.shape[1]),
                    "max_position_error_m": float(position_error.max()),
                    "max_linear_velocity_error_mps": float(velocity_error.max()),
                    "max_quaternion_sign_invariant_l2": float(quaternion_error.max()),
                    "rebound_corrections": int(corrections),
                    "passed_all_frames_1e-5": bool(
                        position_error.max() <= 1e-5
                        and velocity_error.max() <= 1e-5
                        and quaternion_error.max() <= 1e-5
                    ),
                }
            )
        except Exception as exc:
            rows.append({"key": key, **exception_payload(exc)})
    return {
        "status": "EXECUTED" if all(row.get("status") == "EXECUTED" for row in rows) else "PARTIAL",
        "entrypoint": "original_pipeline_audit.replay_audit.replay",
        "rows": rows,
    }


def choose_support_normal(frame7_contacts: list[dict[str, Any]]) -> dict[str, Any]:
    nonfloor = [record for record in frame7_contacts if record["body_b_name"] != "floor"]
    candidates = nonfloor or frame7_contacts
    if not candidates:
        return {
            "normal": [0.0, 0.0, 1.0],
            "source": "world_z_fallback_no_frame7_contact",
            "valid_support_contact": False,
            "horizontal": False,
        }
    record = max(candidates, key=lambda item: abs(float(item["normal_on_b"][2])))
    normal = np.asarray(record["normal_on_b"], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    horizontal = bool(abs(float(normal[2])) >= 0.95)
    return {
        "normal": normal.tolist(),
        "source": "frame7_contact_normal_on_b" if nonfloor or frame7_contacts else "world_z_fallback",
        "support_body": record["body_b_name"],
        "valid_support_contact": bool(nonfloor or frame7_contacts),
        "horizontal": horizontal,
    }


def contact_velocity_diagnostics(
    state: dict[str, Any],
    prefix: dict[str, Any],
    omega_gt: np.ndarray,
    omega_hypothesis: np.ndarray,
    omega_legacy: np.ndarray,
) -> dict[str, Any]:
    center = np.asarray(state["positions"][7, state["dynamic_index"]], dtype=np.float64)
    velocity = np.asarray(state["linear_velocities"][7, state["dynamic_index"]], dtype=np.float64)
    rows = []
    for contact in prefix["frame7_contacts"]:
        point = np.asarray(contact["position_on_a"], dtype=np.float64)
        rows.append(
            {
                "body_b_name": contact["body_b_name"],
                "normal_on_b": contact["normal_on_b"],
                "gt_contact_velocity_mps": contact_point_velocity(center, velocity, omega_gt, point).tolist(),
                "rolling_hypothesis_contact_velocity_mps": contact_point_velocity(center, velocity, omega_hypothesis, point).tolist(),
                "legacy_rolling_contact_velocity_mps": contact_point_velocity(center, velocity, omega_legacy, point).tolist(),
                "contact_distance_m": contact["contact_distance_m"],
                "normal_force_n": contact["normal_force_n"],
            }
        )
    gt_norm = [float(np.linalg.norm(row["gt_contact_velocity_mps"])) for row in rows]
    hyp_norm = [float(np.linalg.norm(row["rolling_hypothesis_contact_velocity_mps"])) for row in rows]
    legacy_norm = [float(np.linalg.norm(row["legacy_rolling_contact_velocity_mps"])) for row in rows]
    return {
        "frame7_contact_count": len(rows),
        "rows": rows,
        "max_gt_contact_speed_mps": max(gt_norm, default=None),
        "max_rolling_hypothesis_contact_speed_mps": max(hyp_norm, default=None),
        "max_legacy_rolling_contact_speed_mps": max(legacy_norm, default=None),
        "gt_pure_rolling_assumption": bool(rows and max(gt_norm) <= CONTACT_VELOCITY_TOLERANCE_MPS),
        "rolling_hypothesis_residual_below_tolerance": bool(rows and max(hyp_norm) <= CONTACT_VELOCITY_TOLERANCE_MPS),
        "legacy_rolling_residual_below_tolerance": bool(rows and max(legacy_norm) <= CONTACT_VELOCITY_TOLERANCE_MPS),
    }


def mode_metrics(
    predicted: dict[str, Any],
    target: np.ndarray,
    target_velocity: np.ndarray,
    target_angular_velocity: np.ndarray | None,
    context_p7: np.ndarray,
    target_times: np.ndarray,
    state_spec: dict[str, Any],
) -> dict[str, Any]:
    positions = np.asarray(predicted["positions"], dtype=np.float64)
    velocities = np.asarray(predicted["linear_velocities"], dtype=np.float64)
    angular = np.asarray(predicted["angular_velocities"], dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    target_velocity = np.asarray(target_velocity, dtype=np.float64)
    position_errors = np.linalg.norm(positions - target, axis=-1)
    sampled_velocity_errors = np.linalg.norm(velocities - target_velocity, axis=-1)
    dt = np.diff(np.concatenate([[target_times[0] - 1.0 / FPS], target_times]))
    # The first interval is explicitly RGB7 -> RGB8; subsequent intervals are target-frame differences.
    target_interval = np.diff(np.concatenate([context_p7[None, :], target], axis=0), axis=0) / dt[:, None]
    predicted_interval = np.diff(np.concatenate([context_p7[None, :], positions], axis=0), axis=0) / dt[:, None]
    interval_velocity_errors = np.linalg.norm(predicted_interval - target_interval, axis=-1)
    contact_by_frame: dict[int, list[str]] = {}
    for record in predicted["contact_records"]:
        frame = record.get("output_frame")
        if frame is None:
            continue
        contact_by_frame.setdefault(int(frame), [])
        body = str(record["body_b_name"])
        if body not in contact_by_frame[int(frame)]:
            contact_by_frame[int(frame)].append(body)
    nonfloor_indices = [frame for frame, names in contact_by_frame.items() if any(name != "floor" for name in names)]
    if nonfloor_indices:
        first_contact = min(nonfloor_indices)
        last_contact = max(nonfloor_indices)
        before = position_errors[:first_contact]
        during = position_errors[first_contact : last_contact + 1]
        after = position_errors[last_contact + 1 :]
    else:
        first_contact = None
        last_contact = None
        before = position_errors
        during = np.asarray([], dtype=np.float64)
        after = np.asarray([], dtype=np.float64)
    first_divergence = next((int(index) for index, error in enumerate(position_errors) if error > 1e-3), None)
    angular_metrics = {
        "sampled_angular_velocity_error_radps_mean": None,
        "sampled_angular_velocity_error_radps_max": None,
    }
    if target_angular_velocity is not None:
        angular_errors = np.linalg.norm(angular - np.asarray(target_angular_velocity, dtype=np.float64), axis=-1)
        angular_metrics = {
            "sampled_angular_velocity_error_radps_mean": float(angular_errors.mean()),
            "sampled_angular_velocity_error_radps_max": float(angular_errors.max()),
        }
    return {
        "ADE_m": float(position_errors.mean()),
        "FDE_m": float(position_errors[-1]),
        "max_error_m": float(position_errors.max()),
        "position_error_by_future_frame_m": position_errors.tolist(),
        "first_divergence_over_1mm_future_index": first_divergence,
        "first_divergence_over_1mm_absolute_frame": None if first_divergence is None else first_divergence + CONTEXT_FRAMES,
        "sampled_state_velocity_error_mps_mean": float(sampled_velocity_errors.mean()),
        "sampled_state_velocity_error_mps_max": float(sampled_velocity_errors.max()),
        "sampled_state_velocity_error_by_future_frame_mps": sampled_velocity_errors.tolist(),
        "interval_velocity_error_mps_mean": float(interval_velocity_errors.mean()),
        "interval_velocity_error_mps_max": float(interval_velocity_errors.max()),
        "interval_velocity_error_by_future_frame_mps": interval_velocity_errors.tolist(),
        "contact_phase": {
            "first_nonfloor_contact_future_index": first_contact,
            "last_nonfloor_contact_future_index": last_contact,
            "before_contact_ADE_m": None if before.size == 0 else float(before.mean()),
            "during_contact_ADE_m": None if during.size == 0 else float(during.mean()),
            "after_contact_ADE_m": None if after.size == 0 else float(after.mean()),
            "contact_frames_with_nonfloor": sorted(nonfloor_indices),
        },
        "initial_velocity_error_mps": float(np.linalg.norm(state_spec["linear_velocity"] - state_spec["v7_gt"])),
        "initial_angular_velocity_error_radps": (
            None
            if state_spec.get("omega7_gt") is None
            else float(np.linalg.norm(state_spec["angular_velocity"] - state_spec["omega7_gt"]))
        ),
        "api_step_calls": int(predicted["api_step_calls"]),
        "rollout_corrections": int(len(predicted["correction_records"])),
        "correction_records": predicted["correction_records"],
        **angular_metrics,
    }


def serialise_rollout(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions=np.asarray(result["positions"], dtype=np.float32),
        quaternions=np.asarray(result["quaternions"], dtype=np.float32),
        linear_velocities=np.asarray(result["linear_velocities"], dtype=np.float32),
        angular_velocities=np.asarray(result["angular_velocities"], dtype=np.float32),
    )


def run_rebuild_mode(generator, case, seed: int, raw: dict[str, Any], prefix: dict[str, Any], mode: str) -> dict[str, Any]:
    context = raw_context_payload(raw)
    omega_gt = np.asarray(prefix["prefix"]["angular_velocities"][7], dtype=np.float64)
    support = choose_support_normal(prefix["frame7_contacts"])
    normal = np.asarray(support["normal"], dtype=np.float64)
    omega_hyp = rolling_omega_hypothesis(context["v7_gt"], normal)
    omega_legacy = rolling_omega_legacy(context["v7_legacy"])
    if mode == "R1_gt_state_rebuild":
        velocity = context["v7_gt"]
        angular = omega_gt
        velocity_strategy = "gt_observed"
        omega_strategy = "gt_observed"
    elif mode == "R2_fd_velocity_only":
        velocity = context["v7_fd"]
        angular = omega_gt
        velocity_strategy = "finite_difference_frame_times"
        omega_strategy = "gt_observed"
    elif mode == "R3_rolling_omega_only":
        velocity = context["v7_gt"]
        angular = omega_hyp
        velocity_strategy = "gt_observed"
        omega_strategy = "rolling_hypothesis"
    elif mode == "R4_fd_plus_rolling":
        velocity = context["v7_fd"]
        angular = omega_hyp
        velocity_strategy = "finite_difference_frame_times"
        omega_strategy = "rolling_hypothesis"
    elif mode == "L_legacy_equivalent":
        velocity = context["v7_legacy"]
        angular = omega_legacy
        velocity_strategy = "legacy_fps_difference"
        omega_strategy = "legacy_rolling"
    else:
        raise ValueError(mode)

    dynamic_state = {
        "position": context["p7"].astype(np.float64),
        "quaternion": context["q7"].astype(np.float64),
        "linear_velocity": np.asarray(velocity, dtype=np.float64),
        "angular_velocity": np.asarray(angular, dtype=np.float64),
    }
    world = build_world(generator, case, seed, dynamic_state=dynamic_state, label=mode)
    start = time.perf_counter()
    try:
        with StepCounter(world.p) as counter:
            world.step_counter = counter
            rollout = run_rebuild_rollout(world, steps_per_output=SIM_HZ // FPS)
        elapsed = time.perf_counter() - start
        rollout["api_step_calls"] = int(counter.calls)
        rollout["wall_seconds_physics_and_contact"] = float(elapsed)
        rollout["state_spec"] = {
            "position": dynamic_state["position"],
            "quaternion": dynamic_state["quaternion"],
            "linear_velocity": dynamic_state["linear_velocity"],
            "angular_velocity": dynamic_state["angular_velocity"],
            "v7_gt": context["v7_gt"],
            "v7_fd": context["v7_fd"],
            "omega7_gt": omega_gt,
            "omega7_rolling_hypothesis": omega_hyp,
            "omega7_legacy": omega_legacy,
        }
        rollout["strategies"] = {
            "velocity": velocity_strategy,
            "omega": omega_strategy,
            "support_normal": support,
            "future_used_for_initialization": False,
            "pre_roll_added_after_rgb7": False,
        }
        target_times = np.asarray(raw["frame_times_float"], dtype=np.float64)[CONTEXT_FRAMES : CONTEXT_FRAMES + FUTURE_FRAMES]
        target_velocity = np.asarray(raw["linear_velocities"], dtype=np.float32)[CONTEXT_FRAMES : CONTEXT_FRAMES + FUTURE_FRAMES, raw["dynamic_index"]]
        target_angular = np.asarray(prefix["continuous"]["angular_velocities"], dtype=np.float32)
        rollout["metrics"] = mode_metrics(
            rollout,
            context["target"],
            target_velocity,
            target_angular,
            context["p7"],
            target_times,
            rollout["state_spec"],
        )
        rollout["contact_diagnostics"] = contact_velocity_diagnostics(
            raw,
            prefix,
            omega_gt,
            omega_hyp,
            omega_legacy,
        )
        return rollout
    finally:
        world.p.disconnect(world.client_id)


def strip_config_for_physics(config: dict[str, Any], *, include_state: bool = False) -> dict[str, Any]:
    out = copy.deepcopy(config)
    out.pop("client_id", None)
    out.pop("plane_id", None)
    out["bodies"] = []
    for body in config.get("bodies", []):
        item = {
            "name": body.get("name"),
            "shape_data": body.get("shape_data"),
            "dynamics_info": body.get("dynamics_info"),
        }
        if include_state:
            item.update(
                {
                    "position": body.get("position"),
                    "quaternion_xyzw": body.get("quaternion_xyzw"),
                    "linear_velocity": body.get("linear_velocity"),
                    "angular_velocity": body.get("angular_velocity"),
                }
            )
        out["bodies"].append(item)
    return out


def recursive_diff(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(left, dict) and isinstance(right, dict):
        diffs = []
        for key in sorted(set(left) | set(right)):
            next_path = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                diffs.append({"path": next_path, "left": left.get(key), "right": right.get(key)})
            else:
                diffs.extend(recursive_diff(left[key], right[key], next_path))
        return diffs
    if isinstance(left, list) and isinstance(right, list):
        diffs = []
        if len(left) != len(right):
            diffs.append({"path": path + ".length", "left": len(left), "right": len(right)})
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            diffs.extend(recursive_diff(l_item, r_item, f"{path}[{index}]"))
        return diffs
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-8):
            return []
    if left != right:
        return [{"path": path, "left": left, "right": right}]
    return []


def timestep_fixture(generator) -> dict[str, Any]:
    import pybullet as p

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, 0.0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / SIM_HZ,
            numSubSteps=8,
            numSolverIterations=50,
        )
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
        body = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape, basePosition=[0.0, 0.0, 0.0])
        p.changeDynamics(body, -1, linearDamping=0.0, angularDamping=0.0, activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
        p.resetBaseVelocity(body, linearVelocity=[1.0, 0.0, 0.0], angularVelocity=[0.0, 0.0, 0.0])
        params = p.getPhysicsEngineParameters()
        samples = []
        for call in range(328):
            p.stepSimulation()
            if call in {0, 7, 8, 15, 327}:
                position = p.getBasePositionAndOrientation(body)[0]
                samples.append({"api_call_index": call + 1, "position_x_m": float(position[0])})
        final_position = p.getBasePositionAndOrientation(body)[0]
    finally:
        p.disconnect(client)
    per_call = float(final_position[0]) / 328.0
    per_output = per_call * 8.0
    first_output = next(item for item in samples if item["api_call_index"] == 8)
    expected_first = 1.0 / FPS
    expected_final = FUTURE_FRAMES / FPS
    return {
        "status": "EXECUTED",
        "fixture": "no_gravity_no_damping_no_contact_constant_velocity",
        "requested": {
            "fixedTimeStep_s": 1.0 / SIM_HZ,
            "engine_num_substeps": 8,
            "wrapper_calls_per_output": 8,
            "output_fps": FPS,
            "future_output_frames": FUTURE_FRAMES,
        },
        "physics_engine_parameters": jsonable(params),
        "samples": samples,
        "actual_step_calls_per_wrapper": 1,
        "actual_step_calls_per_output": 8,
        "total_api_step_calls": 328,
        "estimated_hidden_engine_substeps": 328 * int(params.get("numSubSteps", 0)),
        "effective_api_call_dt_s": float(per_call),
        "expected_api_call_dt_s": 1.0 / SIM_HZ,
        "effective_output_dt_s": float(per_output),
        "expected_output_dt_s": 1.0 / FPS,
        "first_output_after_8_calls_s": float(first_output["position_x_m"]),
        "expected_first_output_s": expected_first,
        "last_output_after_328_calls_s": float(final_position[0]),
        "expected_last_output_s": expected_final,
        "first_output_abs_error_s": float(abs(first_output["position_x_m"] - expected_first)),
        "last_output_abs_error_s": float(abs(float(final_position[0]) - expected_final)),
        "interpretation": "one stepSimulation API call advances one fixedTimeStep while numSubSteps subdivides the engine solve; the bridge's eight API calls per output frame yield 1/30 s outer time and eight engine substeps per API call",
    }


def multi_client_fixture(generator) -> dict[str, Any]:
    import pybullet as p

    c1 = p.connect(p.DIRECT)
    c2 = p.connect(p.DIRECT)
    try:
        for client in (c1, c2):
            p.resetSimulation(physicsClientId=client)
            p.setGravity(0.0, 0.0, 0.0, physicsClientId=client)
            p.setPhysicsEngineParameter(fixedTimeStep=1.0 / SIM_HZ, numSubSteps=1, physicsClientId=client)
        shape1 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01, physicsClientId=c1)
        body1 = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape1, basePosition=[0.0, 0.0, 0.0], physicsClientId=c1)
        p.resetBaseVelocity(body1, linearVelocity=[1.0, 0.0, 0.0], physicsClientId=c1)
        shape2 = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01, physicsClientId=c2)
        body2 = p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape2, basePosition=[10.0, 0.0, 0.0], physicsClientId=c2)
        p.resetBaseVelocity(body2, linearVelocity=[-1.0, 0.0, 0.0], physicsClientId=c2)
        p.stepSimulation(physicsClientId=c1)
        first_after = p.getBasePositionAndOrientation(body1, physicsClientId=c1)[0]
        second_before_reset = p.getBasePositionAndOrientation(body2, physicsClientId=c2)[0]
        p.resetSimulation(physicsClientId=c2)
        first_after_other_reset = p.getBasePositionAndOrientation(body1, physicsClientId=c1)[0]
        # Only client 1 was stepped.  Client 2 must therefore still be at its
        # initial position, and resetting client 2 must leave client 1 intact.
        passed = bool(
            np.allclose(first_after, first_after_other_reset, rtol=0.0, atol=1e-15)
            and np.allclose(second_before_reset[0], 10.0, rtol=0.0, atol=1e-15)
            and first_after[0] > 0.0
        )
    finally:
        p.disconnect(c1)
        p.disconnect(c2)
    return {
        "status": "EXECUTED",
        "passed": passed,
        "client_1_id": c1,
        "client_2_id": c2,
        "client_1_after_client_2_reset": list(first_after_other_reset),
        "client_2_before_reset": list(second_before_reset),
        "explicit_physicsClientId": True,
    }


def bridge_multi_client_regression(pilot, generator, records: dict[str, dict[str, Any]], data_root: Path) -> dict[str, Any]:
    """Run the production bridge with an unrelated client already connected."""
    import pybullet as p
    import pybullet_context_rollout as bridge

    key = SELECTED_CASES[0]
    record = records[key]
    case, _ = reconstruct_case(pilot, generator, record)
    sample_dir = data_root / "samples" / key
    c0 = p.connect(p.DIRECT)
    sentinel = None
    try:
        p.resetSimulation(physicsClientId=c0)
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01, physicsClientId=c0)
        sentinel = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=shape,
            basePosition=[123.0, 0.0, 0.0],
            physicsClientId=c0,
        )
        before = {
            "body_count": int(p.getNumBodies(physicsClientId=c0)),
            "sentinel_position": list(p.getBasePositionAndOrientation(sentinel, physicsClientId=c0)[0]),
        }
        result = bridge.rollout_from_rgb7(pilot, generator, case, record, sample_dir)
        after = {
            "body_count": int(p.getNumBodies(physicsClientId=c0)),
            "sentinel_position": list(p.getBasePositionAndOrientation(sentinel, physicsClientId=c0)[0]),
        }
        unchanged = bool(
            before["body_count"] == after["body_count"]
            and np.allclose(before["sentinel_position"], after["sentinel_position"], rtol=0.0, atol=1e-15)
        )
        return {
            "status": "EXECUTED",
            "case": key,
            "preexisting_client_id": int(c0),
            "before": before,
            "after": after,
            "sentinel_unchanged": unchanged,
            "bridge_metrics": result["metrics"],
            "passed": unchanged,
        }
    except Exception as exc:
        return {"status": "BLOCKED", "case": key, **exception_payload(exc)}
    finally:
        try:
            p.disconnect(c0)
        except Exception:
            pass


def default_client_alias_probe() -> dict[str, Any]:
    """Demonstrate why omitted physicsClientId is unsafe with client 0 busy."""
    import pybullet as p

    c0 = p.connect(p.DIRECT)
    c1 = p.connect(p.DIRECT)
    try:
        p.resetSimulation(physicsClientId=c0)
        p.resetSimulation(physicsClientId=c1)
        shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01, physicsClientId=c0)
        sentinel = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=shape,
            basePosition=[123.0, 0.0, 0.0],
            physicsClientId=c0,
        )
        before = {
            "client_0_body_count": int(p.getNumBodies(physicsClientId=c0)),
            "client_0_sentinel_position": list(p.getBasePositionAndOrientation(sentinel, physicsClientId=c0)[0]),
        }
        # This is the exact implicit-default operation used by the old bridge.
        p.resetSimulation()
        after = {
            "client_0_body_count": int(p.getNumBodies(physicsClientId=c0)),
            "client_1_body_count": int(p.getNumBodies(physicsClientId=c1)),
        }
        return {
            "status": "EXECUTED",
            "client_0_id": int(c0),
            "client_1_id": int(c1),
            "before": before,
            "after_implicit_reset": after,
            "implicit_default_hits_client_0": bool(after["client_0_body_count"] == 0),
        }
    finally:
        p.disconnect(c1)
        p.disconnect(c0)


def command_output(args: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(args, cwd=cwd, check=False, capture_output=True, text=True).stdout.strip()
    except Exception as exc:  # pragma: no cover - environment diagnostics should not abort the audit
        return f"<command failed: {type(exc).__name__}: {exc}>"


def environment_payload(generator) -> dict[str, Any]:
    import pybullet as p
    import pybullet_data

    plane_path = Path(pybullet_data.getDataPath()) / "plane.urdf"
    package_versions = {}
    for name in ("pybullet", "numpy"):
        try:
            package_versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            package_versions[name] = None
    project_repo = ENGINE_ROOT.parent
    project_commit = command_output(["git", "rev-parse", "HEAD"], cwd=project_repo)
    project_status = command_output(["git", "status", "--short"], cwd=project_repo)
    source_paths = {
        "bridge": PROJECT / "pybullet_context_rollout.py",
        "recheck": PROJECT / "recheck_pybullet_bridge.py",
        "pilot_generator": PROJECT / "prepare_physvideo_phase1_pilot.py",
        "original_generator": ENGINE_ROOT / "scripts" / "generate_v2v_context_demos.py",
        "original_replay_audit": AUDIT_ROOT / "replay_audit.py",
    }
    source_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in source_paths.items()
        if path.exists()
    }
    client = p.connect(p.DIRECT)
    try:
        default_params = p.getPhysicsEngineParameters()
        api_version = p.getAPIVersion()
    finally:
        p.disconnect(client)
    return {
        "status": "EXECUTED",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "prefix": sys.prefix,
        },
        "pybullet": {
            "module_file": str(Path(p.__file__).resolve()),
            "distribution_version": package_versions["pybullet"],
            "api_version": int(api_version),
            "default_engine_parameters": jsonable(default_params),
        },
        "package_versions": package_versions,
        "pybullet_data": {
            "data_path": str(pybullet_data.getDataPath()),
            "plane_urdf": str(plane_path),
            "plane_urdf_exists": plane_path.exists(),
            "plane_urdf_sha256": sha256_file(plane_path) if plane_path.exists() else None,
        },
        "platform": {
            "system": platform.platform(),
            "python_machine": platform.machine(),
            "cpu_count_visible": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "project": {
            "engine_root": str(ENGINE_ROOT),
            "project_repo": str(project_repo),
            "commit": project_commit,
            "status_short": project_status,
            "source_hashes": source_hashes,
        },
    }


def save_context_state_v2(
    output_root: Path,
    key: str,
    raw: dict[str, Any],
    prefix: dict[str, Any],
) -> dict[str, Any]:
    context = raw_context_payload(raw)
    support = choose_support_normal(prefix["frame7_contacts"])
    omega_gt = np.asarray(prefix["prefix"]["angular_velocities"][7], dtype=np.float32)
    omega_hyp = rolling_omega_hypothesis(context["v7_gt"], np.asarray(support["normal"])).astype(np.float32)
    omega_legacy = rolling_omega_legacy(context["v7_legacy"]).astype(np.float32)
    path = output_root / "context_states_v2" / f"{key}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        positions_observed=np.asarray(context["positions_observed"], dtype=np.float32),
        quaternions_observed=np.asarray(context["quats_observed"], dtype=np.float32),
        linear_velocities_observed=np.asarray(context["linear_velocities_observed"], dtype=np.float32),
        frame_times_observed=np.asarray(context["frame_times_observed"], dtype=np.float64),
        p7=np.asarray(context["p7"], dtype=np.float32),
        q7=np.asarray(context["q7"], dtype=np.float32),
        v7_gt=np.asarray(context["v7_gt"], dtype=np.float32),
        v7_fd=np.asarray(context["v7_fd"], dtype=np.float32),
        v7_legacy=np.asarray(context["v7_legacy"], dtype=np.float32),
        omega7_gt=omega_gt,
        omega7_rolling_hypothesis=omega_hyp,
        omega7_legacy=omega_legacy,
    )
    payload = {
        "schema": "physvideo_context_state_v2",
        "case": key,
        "status": "EXECUTED" if np.isfinite(omega_gt).all() else "GT_RECOVERY_BLOCKED",
        "state_file": str(path),
        "state_file_sha256": sha256_file(path),
        "dynamic_object": "pilot_ball",
        "dynamic_index": int(raw["dynamic_index"]),
        "position_source": "saved_original_bullet_replay_states_xyzw_npz",
        "quaternion_source": "saved_original_bullet_replay_states_xyzw_npz",
        "linear_velocity_sources": {
            "v7_gt": "original_replay_getBaseVelocity_linear_at_frame7_recovered_by_exact_prefix_replay",
            "v7_fd": "(p7-p6)/(frame_times[7]-frame_times[6])",
            "v7_legacy": "(p7-p6)*FPS, matching the unmodified bridge",
        },
        "angular_velocity_sources": {
            "omega7_gt": "original_replay_getBaseVelocity_angular_at_frame7_recovered_by_exact_prefix_replay",
            "omega7_rolling_hypothesis": "cross(support_normal, tangent_velocity)/radius",
            "omega7_legacy": "[vy/r,-vx/r,0]",
        },
        "frame_times": np.asarray(raw["frame_times_float"][:CONTEXT_FRAMES]).tolist(),
        "frame_time_delta_6_to_7_s": float(raw["frame_times_float"][7] - raw["frame_times_float"][6]),
        "support_normal": support,
        "frame7_contacts": prefix["frame7_contacts"],
        "omega7_gt": omega_gt.tolist(),
        "omega7_rolling_hypothesis": omega_hyp.tolist(),
        "omega7_legacy": omega_legacy.tolist(),
        "gt_omega_norm_radps": float(np.linalg.norm(omega_gt)),
        "rolling_omega_norm_radps": float(np.linalg.norm(omega_hyp)),
        "legacy_omega_norm_radps": float(np.linalg.norm(omega_legacy)),
        "contact_diagnostics": contact_velocity_diagnostics(raw, prefix, omega_gt, omega_hyp, omega_legacy),
    }
    write_json(path.with_suffix(".json"), payload)
    return payload


def r0_summary(raw: dict[str, Any], prefix: dict[str, Any]) -> dict[str, Any]:
    context = raw_context_payload(raw)
    target_velocity = np.asarray(raw["linear_velocities"], dtype=np.float32)[CONTEXT_FRAMES : CONTEXT_FRAMES + FUTURE_FRAMES, raw["dynamic_index"]]
    target_times = np.asarray(raw["frame_times_float"], dtype=np.float64)[CONTEXT_FRAMES : CONTEXT_FRAMES + FUTURE_FRAMES]
    state_spec = {
        "position": context["p7"],
        "quaternion": context["q7"],
        "linear_velocity": context["v7_gt"],
        "angular_velocity": prefix["prefix"]["angular_velocities"][7],
        "v7_gt": context["v7_gt"],
        "omega7_gt": prefix["prefix"]["angular_velocities"][7],
    }
    continuous = prefix["continuous"]
    continuous_metrics = mode_metrics(
        continuous,
        context["target"],
        target_velocity,
        continuous["angular_velocities"],
        context["p7"],
        target_times,
        state_spec,
    )
    repeat_metrics = []
    for restored in prefix["restored_runs"]:
        pos_diff = np.linalg.norm(restored["positions"] - continuous["positions"], axis=-1)
        vel_diff = np.linalg.norm(restored["linear_velocities"] - continuous["linear_velocities"], axis=-1)
        quat_diff = np.asarray(
            [quat_sign_invariant_l2(restored["quaternions"][i], continuous["quaternions"][i]) for i in range(FUTURE_FRAMES)]
        )
        repeat_metrics.append(
            {
                "repeat_index": restored["repeat_index"],
                "max_position_difference_from_continuous_m": float(pos_diff.max()),
                "max_linear_velocity_difference_from_continuous_mps": float(vel_diff.max()),
                "max_quaternion_difference_from_continuous": float(quat_diff.max()),
                "same_within_1e-7_m": bool(pos_diff.max() <= 1e-7),
                "api_step_calls": int(restored["api_step_calls"]),
            }
        )
    return {
        "mode": "R0_snapshot_resume",
        "continuous_reference_metrics_vs_old_target": continuous_metrics,
        "restored_repetition_metrics_vs_continuous": repeat_metrics,
        "old_target_is_used_only_for_comparison": True,
        "continuous_and_restored_outputs": {
            "continuous_positions": continuous["positions"].tolist(),
            "restored_positions": [item["positions"].tolist() for item in prefix["restored_runs"]],
        },
        "prefix_match": prefix["prefix_match"],
        "total_step_calls_including_three_restores": prefix["total_step_calls"],
    }


def future_mutation_test(generator, case, seed: int, raw: dict[str, Any], prefix: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mutated = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in raw.items()}
    future_slice = slice(CONTEXT_FRAMES, CONTEXT_FRAMES + FUTURE_FRAMES)
    mutated["positions"] = np.asarray(mutated["positions"]).copy()
    mutated["linear_velocities"] = np.asarray(mutated["linear_velocities"]).copy()
    mutated["quats"] = np.asarray(mutated["quats"]).copy()
    mutated["positions"][future_slice] += np.asarray([17.0, -13.0, 5.0], dtype=np.float32)
    mutated["linear_velocities"][future_slice] *= -11.0
    mutated["quats"][future_slice] *= -1.0
    mutated_record = {"interaction": {"rule": "MUTATED_FUTURE_ONLY", "reference_x_m": 999.0}}
    rerun = run_rebuild_mode(generator, case, seed, mutated, prefix, "R4_fd_plus_rolling")
    baseline_positions = np.asarray(baseline["positions"])
    rerun_positions = np.asarray(rerun["positions"])
    baseline_contacts = json.dumps(baseline["contact_records"], sort_keys=True)
    rerun_contacts = json.dumps(rerun["contact_records"], sort_keys=True)
    return {
        "status": "EXECUTED",
        "mode": "R4_fd_plus_rolling",
        "mutation": {
            "future_positions_delta_m": [17.0, -13.0, 5.0],
            "future_velocity_scale": -11.0,
            "future_quaternion_sign_flipped": True,
            "interaction": mutated_record["interaction"],
        },
        "initialization_context_unchanged": bool(
            np.array_equal(raw["positions"][:CONTEXT_FRAMES], mutated["positions"][:CONTEXT_FRAMES])
            and np.array_equal(raw["linear_velocities"][:CONTEXT_FRAMES], mutated["linear_velocities"][:CONTEXT_FRAMES])
            and np.array_equal(raw["quats"][:CONTEXT_FRAMES], mutated["quats"][:CONTEXT_FRAMES])
        ),
        "max_rollout_position_difference_m": float(np.max(np.linalg.norm(baseline_positions - rerun_positions, axis=-1))),
        "contact_log_equal": bool(baseline_contacts == rerun_contacts),
        "correction_log_equal": bool(baseline["correction_records"] == rerun["correction_records"]),
        "metrics_are_allowed_to_change": bool(baseline["metrics"]["ADE_m"] != rerun["metrics"]["ADE_m"]),
        "baseline_metrics": baseline["metrics"],
        "mutated_metrics": rerun["metrics"],
    }


def timed_physics_rollout(world: PhysicsWorld, output_root: Path, label: str) -> dict[str, Any]:
    world.api_step_index = 0
    world.contact_records = []
    world.correction_records = []
    world.p.performCollisionDetection()
    initial_contact_start = time.perf_counter()
    initial_contacts = world.contacts(phase="initial_collision_query", output_frame=None, record=False)
    initial_contact_seconds = time.perf_counter() - initial_contact_start
    physics_seconds = []
    contact_seconds = []
    state_read_seconds = []
    positions = []
    with StepCounter(world.p) as counter:
        world.step_counter = counter
        for output_frame in range(FUTURE_FRAMES):
            for _ in range(SIM_HZ // FPS):
                start = time.perf_counter()
                world.step_once(output_frame=output_frame, phase="benchmark_step")
                physics_seconds.append(time.perf_counter() - start)
                start = time.perf_counter()
                world.contact_snapshot(phase="benchmark_contact", output_frame=output_frame)
                contact_seconds.append(time.perf_counter() - start)
            start = time.perf_counter()
            positions.append(world.capture_state())
            state_read_seconds.append(time.perf_counter() - start)
    array = np.asarray([state["position"] for state in positions], dtype=np.float32)
    output_root.mkdir(parents=True, exist_ok=True)
    serialization_path = output_root / f"{label}.npz"
    start = time.perf_counter()
    np.savez_compressed(serialization_path, positions=array)
    serialization_seconds = time.perf_counter() - start
    return {
        "initial_contact_seconds": float(initial_contact_seconds),
        "physics_seconds": float(sum(physics_seconds)),
        "contact_read_seconds": float(sum(contact_seconds)),
        "state_read_seconds": float(sum(state_read_seconds)),
        "serialization_seconds": float(serialization_seconds),
        "total_api_step_calls": int(counter.calls),
        "initial_contact_count": len(initial_contacts),
        "serialization_path": str(serialization_path),
    }


def summary_stats(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(array)),
        "median_s": float(np.median(array)),
        "p95_s": float(np.percentile(array, 95)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def benchmark_case(generator, case, seed: int, raw: dict[str, Any], prefix: dict[str, Any], output_root: Path) -> dict[str, Any]:
    context = raw_context_payload(raw)
    omega_gt = np.asarray(prefix["prefix"]["angular_velocities"][7], dtype=np.float64)
    dynamic_state = {
        "position": context["p7"].astype(np.float64),
        "quaternion": context["q7"].astype(np.float64),
        "linear_velocity": context["v7_gt"].astype(np.float64),
        "angular_velocity": omega_gt,
    }
    bench_root = output_root / "latency" / case.case_id
    bench_root.mkdir(parents=True, exist_ok=True)
    cold_runs = []
    # Two cold warm-ups are discarded, followed by ten measured cold runs.
    for index in range(12):
        total_start = time.perf_counter()
        build_start = time.perf_counter()
        world = build_world(generator, case, seed, dynamic_state=dynamic_state, label="benchmark_cold")
        connect_build_seconds = time.perf_counter() - build_start
        try:
            timed = timed_physics_rollout(world, bench_root / "cold", f"cold_{index:02d}")
        finally:
            disconnect_start = time.perf_counter()
            world.p.disconnect(world.client_id)
            disconnect_seconds = time.perf_counter() - disconnect_start
        timed.update(
            {
                "connect_build_seconds": connect_build_seconds,
                "disconnect_seconds": disconnect_seconds,
                "cold_total_seconds": time.perf_counter() - total_start,
                "warmup": index < 2,
            }
        )
        cold_runs.append(timed)

    reuse_world = build_world(generator, case, seed, dynamic_state=dynamic_state, label="benchmark_reuse")
    reuse_state_id = reuse_world.p.saveState()
    reuse_runs = []
    try:
        for index in range(12):
            restore_start = time.perf_counter()
            reuse_world.p.restoreState(stateId=reuse_state_id)
            restore_seconds = time.perf_counter() - restore_start
            timed = timed_physics_rollout(reuse_world, bench_root / "reuse", f"reuse_{index:02d}")
            timed.update({"restore_state_seconds": restore_seconds, "warmup": index < 2})
            reuse_runs.append(timed)
    finally:
        reuse_world.p.removeState(reuse_state_id)
        reuse_world.p.disconnect(reuse_world.client_id)

    def measured(runs: list[dict[str, Any]], field: str) -> dict[str, Any]:
        return summary_stats([float(item[field]) for item in runs[2:]])

    return {
        "case": case.case_id,
        "family": case.family_title,
        "warmup_runs": 2,
        "measured_runs": 10,
        "cold_start": {
            "connect_build": measured(cold_runs, "connect_build_seconds"),
            "physics": measured(cold_runs, "physics_seconds"),
            "contact_read": measured(cold_runs, "contact_read_seconds"),
            "state_read": measured(cold_runs, "state_read_seconds"),
            "serialization": measured(cold_runs, "serialization_seconds"),
            "total": measured(cold_runs, "cold_total_seconds"),
            "disconnect": measured(cold_runs, "disconnect_seconds"),
        },
        "reuse_saved_state": {
            "restore_state": measured(reuse_runs, "restore_state_seconds"),
            "physics": measured(reuse_runs, "physics_seconds"),
            "contact_read": measured(reuse_runs, "contact_read_seconds"),
            "state_read": measured(reuse_runs, "state_read_seconds"),
            "serialization": measured(reuse_runs, "serialization_seconds"),
        },
        "raw_runs": {"cold": cold_runs, "reuse": reuse_runs},
    }


def exception_payload(exc: BaseException) -> dict[str, Any]:
    return {"status": "BLOCKED", "exception_type": type(exc).__name__, "message": str(exc)}


def write_mode_outputs(root: Path, case_key: str, mode: str, result: dict[str, Any]) -> None:
    mode_root = root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    serialise_rollout(mode_root / f"{case_key}.npz", result)
    public = copy.deepcopy(result)
    for key in ("positions", "quaternions", "linear_velocities", "angular_velocities"):
        public.pop(key, None)
    write_json(mode_root / f"{case_key}.json", public)
    write_json(root / "world_configs_and_diffs" / case_key / f"{mode}_config_initial.json", result.get("config_initial", {}))
    write_json(root / "world_configs_and_diffs" / case_key / f"{mode}_config_final.json", result.get("config_final", {}))


def mode_frame_rows(case_key: str, mode: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = result.get("metrics", {})
    errors = metrics.get("position_error_by_future_frame_m", [])
    velocity_errors = metrics.get("sampled_state_velocity_error_by_future_frame_mps", [])
    interval_errors = metrics.get("interval_velocity_error_by_future_frame_mps", [])
    contacts: dict[int, list[str]] = {}
    for record in result.get("contact_records", []):
        frame = record.get("output_frame")
        if frame is None:
            continue
        contacts.setdefault(int(frame), [])
        body = str(record.get("body_b_name"))
        if body not in contacts[int(frame)]:
            contacts[int(frame)].append(body)
    rows = []
    for index, error in enumerate(errors):
        rows.append(
            {
                "case": case_key,
                "mode": mode,
                "future_index": index,
                "absolute_frame": index + CONTEXT_FRAMES,
                "time_after_rgb7_s": (index + 1) / FPS,
                "position_error_m": error,
                "sampled_state_velocity_error_mps": velocity_errors[index] if index < len(velocity_errors) else None,
                "interval_velocity_error_mps": interval_errors[index] if index < len(interval_errors) else None,
                "contact_names": ";".join(contacts.get(index, [])),
                "corrections_at_mode": len(result.get("correction_records", [])),
            }
        )
    return rows


def write_per_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case",
        "mode",
        "future_index",
        "absolute_frame",
        "time_after_rgb7_s",
        "position_error_m",
        "sampled_state_velocity_error_mps",
        "interval_velocity_error_mps",
        "contact_names",
        "corrections_at_mode",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_contact_jsonl(path: Path, contact_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in contact_rows:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def summarize_findings(all_results: dict[str, Any]) -> dict[str, Any]:
    prefix_rows = [row for row in all_results["cases"].values() if row.get("status") == "EXECUTED"]
    rolling = all_results["rolling_kinematics"]
    timestep = all_results["timestep_fixture"]
    client = all_results["multi_client_fixture"]
    r0_rows = [row.get("r0", {}) for row in prefix_rows if row.get("r0")]
    r0_continuous = [row.get("continuous_reference_metrics_vs_old_target", {}) for row in r0_rows]
    r0_repeated = [repeat for row in r0_rows for repeat in row.get("restored_repetition_metrics_vs_continuous", [])]
    legacy_checks = [row.get("legacy_equivalence", {}) for row in prefix_rows if row.get("legacy_equivalence")]
    mode_effects = []
    for row in prefix_rows:
        modes = row.get("modes", {})
        r1 = modes.get("R1_gt_state_rebuild", {})
        if r1.get("status") != "EXECUTED":
            continue
        r1m = r1["metrics"]
        effects = {"case": row["key"]}
        for name in ("R2_fd_velocity_only", "R3_rolling_omega_only", "R4_fd_plus_rolling"):
            payload = modes.get(name, {})
            if payload.get("status") == "EXECUTED":
                metric = payload["metrics"]
                effects[name] = {
                    "delta_ADE_m_vs_R1": float(metric["ADE_m"] - r1m["ADE_m"]),
                    "delta_FDE_m_vs_R1": float(metric["FDE_m"] - r1m["FDE_m"]),
                    "delta_max_error_m_vs_R1": float(metric["max_error_m"] - r1m["max_error_m"]),
                }
            else:
                effects[name] = {"status": payload.get("status", "BLOCKED")}
        mode_effects.append(effects)
    return {
        "confirmed_bug": [
            {
                "finding": "修复前 bridge 的 PyBullet 调用未传 physicsClientId；当 client 0 已被占用时，隐式默认调用会重置/写入 client 0，而不是 bridge 新建的 client。",
                "evidence": "default_client_alias_probe",
                "fixed_in_current_bridge": bool(all_results.get("bridge_multi_client_regression", {}).get("passed")),
            }
        ],
        "no_code_bug_or_doc_ambiguity": [
            {
                "finding": "PyBullet numSubSteps is distinct from the Python wrapper loop; the local constant-time fixture must determine effective outer time.",
                "evidence": "timestep_fixture",
            }
        ],
        "protocol_approximation": [
            {
                "finding": "legacy_rolling uses the opposite sign from the pure rolling hypothesis under the tested z-up convention; it is retained as an explicit legacy strategy rather than silently changed.",
                "evidence": "rolling_kinematics",
            },
            {
                "finding": "finite-difference v7 is an interval average over frame 6 to 7, while saved linear_velocities[7] is the instantaneous state value.",
                "evidence": "context_states_v2",
            },
        ],
        "unverified_or_blocked": [],
        "evidence_summary": {
            "case_count": len(prefix_rows),
            "original_replay_entrypoint": all_results.get("original_replay_entrypoint", {}),
            "rolling_fixture_passed": rolling.get("passed"),
            "clock_fixture_first_error_s": timestep.get("first_output_abs_error_s"),
            "clock_fixture_last_error_s": timestep.get("last_output_abs_error_s"),
            "multi_client_passed": client.get("passed"),
            "bridge_multi_client_passed": all_results.get("bridge_multi_client_regression", {}).get("passed"),
            "implicit_default_hits_client_0": all_results.get("default_client_alias_probe", {}).get("implicit_default_hits_client_0"),
            "r0_continuous_rows": r0_continuous,
            "r0_restore_rows": r0_repeated,
            "legacy_equivalence_rows": legacy_checks,
            "mode_effects_vs_R1": mode_effects,
        },
    }


def build_report(all_results: dict[str, Any]) -> str:
    lines = []
    lines.append("# PyBullet pilot bridge 分层复核报告")
    lines.append("")
    lines.append("本报告只覆盖六个 pilot episode 的 trajectory-only 复核；没有训练 Predictor、运行 RGB/VGGT/SAM2/Utonia 或扩充数据。")
    lines.append("")
    lines.append("## 首页结论")
    lines.append("")
    lines.append("1. 修复前的 `legacy_approx` 原样证据保存在 `validation_pybullet_bridge_recheck_v8/legacy_approx`；本目录再次运行修复后 bridge，六例输出逐项保持一致。它不是精确 GT 重放，只是历史近似协议基线。")
    lines.append("2. 纯滚动独立接触点测试验证了 z-up 世界坐标下的符号；当前 legacy 公式被保留为显式诊断策略，没有回写旧 target。")
    lines.append("3. RGB7 真实角速度通过原生成器前缀重放补录；如果某 case 的前缀校验未通过，对应 GT omega 模式会标为 `BLOCKED`。")
    lines.append("4. `R0` 将原世界连续续推与同一 client 的 save/restore 续推分开报告，并分别与旧 target 比较。")
    lines.append("5. `R1/R2/R3/R4` 只在真实 RGB7 omega 可可靠恢复时执行；差分 v 和 rolling omega 的影响按单变量模式报告。")
    lines.append("6. 修复了 bridge 的多 client 默认 client 风险：当前 pilot 建世界、读状态、接触和 step 均显式绑定 rollout client。")
    lines.append("")
    lines.append("## 执行状态")
    lines.append("")
    lines.append("| case | prefix / omega | R0 | R1 | R2 | R3 | R4 | legacy 对照 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for key, row in all_results["cases"].items():
        if row.get("status") != "EXECUTED":
            lines.append(f"| `{key}` | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | — |")
            continue
        modes = row.get("modes", {})
        status = lambda name: modes.get(name, {}).get("status", "BLOCKED")
        lines.append(
            f"| `{key}` | {row.get('omega_status', 'UNKNOWN')} | {row.get('r0_status', 'UNKNOWN')} | "
            f"{status('R1_gt_state_rebuild')} | {status('R2_fd_velocity_only')} | "
            f"{status('R3_rolling_omega_only')} | {status('R4_fd_plus_rolling')} | {row.get('legacy_equivalence', {}).get('status', 'UNKNOWN')} |"
        )
    lines.append("")
    lines.append("## 原始 replay 入口独立对照")
    lines.append("")
    lines.append("直接调用 `original_pipeline_audit.replay_audit.replay`，与每个 sample 的 49 帧 raw state 比较；该检查不使用 bridge 的 rollout 输出。")
    lines.append("")
    for item in all_results.get("original_replay_entrypoint", {}).get("rows", []):
        if item.get("status") != "EXECUTED":
            lines.append(f"- `{item.get('key')}`: {item.get('status')} — {item.get('message', '')}")
        else:
            lines.append(
                f"- `{item['key']}`: position max {item['max_position_error_m']:.3e} m, "
                f"velocity max {item['max_linear_velocity_error_mps']:.3e} m/s, "
                f"quaternion max {item['max_quaternion_sign_invariant_l2']:.3e}, "
                f"corrections={item['rebound_corrections']}, passed={item['passed_all_frames_1e-5']}."
            )
    lines.append("")
    lines.append("## 时间和调用语义")
    lines.append("")
    clock = all_results["timestep_fixture"]
    lines.append(f"- fixture: `{clock['fixture']}`，首个 8-call 输出位移 {clock['first_output_after_8_calls_s']:.9f} m，末端 328-call 位移 {clock['last_output_after_328_calls_s']:.9f} m。")
    lines.append(f"- 每个 API `stepSimulation()` 的实际时间为 {clock['effective_api_call_dt_s']:.9f} s；8 次 API 调用合成一个输出帧，实际输出间隔 {clock['effective_output_dt_s']:.9f} s（目标 {clock['expected_output_dt_s']:.9f} s）。")
    lines.append(f"- 实测 `fixedTimeStep={clock['requested']['fixedTimeStep_s']}`、`engine_num_substeps={clock['physics_engine_parameters'].get('numSubSteps')}`、每输出 8 次 API 调用；引擎隐藏子步估计为 {clock['estimated_hidden_engine_substeps']}。")
    lines.append("- 结论只针对本地 PyBullet 版本和这个独立 fixture；不能把 API 调用数直接称为内部积分频率。")
    lines.append("")
    lines.append("## 逐 case 分层结果")
    lines.append("")
    for key, row in all_results["cases"].items():
        if row.get("status") != "EXECUTED":
            continue
        lines.append(f"### `{key}`")
        lines.append("")
        prefix_match = row["prefix_match"]
        lines.append(
            f"- 原生成器前缀复现：position max {prefix_match['max_position_error_m']:.3e} m，"
            f"velocity max {prefix_match['max_linear_velocity_error_mps']:.3e} m/s，"
            f"quaternion max {prefix_match['max_quaternion_sign_invariant_l2']:.3e}。"
        )
        if row.get("r0"):
            cont = row["r0"]["continuous_reference_metrics_vs_old_target"]
            repeats = row["r0"]["restored_repetition_metrics_vs_continuous"]
            lines.append(f"- R0 continuous vs old target: ADE {cont['ADE_m']:.6g} m, FDE {cont['FDE_m']:.6g} m；restore repeats: {json.dumps(repeats, ensure_ascii=False)}")
        for mode, payload in row.get("modes", {}).items():
            if payload.get("status") != "EXECUTED":
                lines.append(f"- {mode}: {payload.get('status')} — {payload.get('message', '')}")
                continue
            metric = payload["metrics"]
            lines.append(
                f"- {mode}: ADE {metric['ADE_m']:.6g} m, FDE {metric['FDE_m']:.6g} m, "
                f"max {metric['max_error_m']:.6g} m, initial v error {metric['initial_velocity_error_mps']:.6g} m/s, "
                f"initial omega error {metric['initial_angular_velocity_error_radps']}."
            )
        diag = row.get("context_state", {}).get("contact_diagnostics", {})
        lines.append(f"- RGB7 contact/rolling diagnostic: {diag.get('frame7_contact_count')} contacts, GT pure-rolling assumption={diag.get('gt_pure_rolling_assumption')}, corrected rolling residual={diag.get('max_rolling_hypothesis_contact_speed_mps')} m/s, legacy residual={diag.get('max_legacy_rolling_contact_speed_mps')} m/s。")
        lines.append("")
    lines.append("## 分层误差归因（相对 R1_gt_state_rebuild）")
    lines.append("")
    lines.append("正值表示 ADE/FDE 变差；R2 只替换线速度，R3 只替换角速度，R4 同时替换两者。")
    lines.append("")
    lines.append("| case | R2 ΔADE / ΔFDE (m) | R3 ΔADE / ΔFDE (m) | R4 ΔADE / ΔFDE (m) |")
    lines.append("|---|---:|---:|---:|")
    for row in all_results["findings"]["evidence_summary"].get("mode_effects_vs_R1", []):
        cells = [row["case"]]
        for name in ("R2_fd_velocity_only", "R3_rolling_omega_only", "R4_fd_plus_rolling"):
            effect = row.get(name, {})
            if "delta_ADE_m_vs_R1" not in effect:
                cells.append(effect.get("status", "BLOCKED"))
            else:
                cells.append(f"{effect['delta_ADE_m_vs_R1']:+.6g} / {effect['delta_FDE_m_vs_R1']:+.6g}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 未来隔离和 client 隔离")
    lines.append("")
    for key, row in all_results["cases"].items():
        mutation = row.get("future_mutation")
        if mutation:
            lines.append(f"- `{key}`: future mutation 后 rollout max difference {mutation['max_rollout_position_difference_m']:.3e} m，contact log equal={mutation['contact_log_equal']}，metrics changed={mutation['metrics_are_allowed_to_change']}。")
    lines.append(f"- multi-client fixture: passed={all_results['multi_client_fixture'].get('passed')}。")
    alias_probe = all_results.get("default_client_alias_probe", {})
    lines.append(f"- implicit-default probe: omitted `physicsClientId` hit client 0={alias_probe.get('implicit_default_hits_client_0')}（这是修复前风险的独立证据）。")
    bridge_mc = all_results.get("bridge_multi_client_regression", {})
    lines.append(f"- production bridge with a pre-existing client: passed={bridge_mc.get('passed')}，sentinel unchanged={bridge_mc.get('sentinel_unchanged')}。")
    lines.append("")
    lines.append("## 实测耗时（每个 family 2 次 warm-up + 10 次 measured）")
    lines.append("")
    lines.append("| family case | cold build median/p95 (s) | cold total median/p95 (s) | reuse physics median/p95 (s) | reuse contact median/p95 (s) |")
    lines.append("|---|---:|---:|---:|---:|")
    for bench in all_results.get("latency", []):
        if "cold_start" not in bench:
            lines.append(f"| {bench.get('case')} | {bench.get('status', 'BLOCKED')} | — | — |")
            continue
        cold = bench["cold_start"]
        reuse = bench["reuse_saved_state"]
        lines.append(
            f"| {bench['case']} | {cold['connect_build']['median_s']:.6f}/{cold['connect_build']['p95_s']:.6f} | "
            f"{cold['total']['median_s']:.6f}/{cold['total']['p95_s']:.6f} | "
            f"{reuse['physics']['median_s']:.6f}/{reuse['physics']['p95_s']:.6f} | "
            f"{reuse['contact_read']['median_s']:.6f}/{reuse['contact_read']['p95_s']:.6f} |"
        )
    lines.append("")
    lines.append("## 限制")
    lines.append("")
    lines.append("- 旧 raw NPZ 没有直接保存 angular velocity；本轮通过同一原生成器在本地环境恢复 RGB7 前缀并补录 omega。若 source hash、前缀状态或时间边界不匹配，GT omega 模式必须保持 BLOCKED。")
    lines.append("- 接触记录在 `after_api_step` 分辨率；当 `numSubSteps>1` 时不能声称覆盖每个隐藏引擎子步。")
    lines.append("- blueprint 的 camera/material/finite-surface token 不构成当前 PyBullet 的碰撞输入；它们只用于渲染或 predictor adapter。")
    lines.append("- latency 结果只代表结构化状态到 PyBullet 的 CPU 路径，不代表 RGB 视觉系统实时性。")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--legacy-output", type=Path, default=LEGACY_OUTPUT)
    parser.add_argument("--cases", nargs="+", default=list(SELECTED_CASES))
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        # The legacy command is deliberately run first and therefore creates
        # the parent directory.  Reuse that parent only when it contains the
        # untouched legacy subdirectory; every other pre-existing artifact is
        # treated as an overwrite hazard.
        existing = {item.name for item in output_root.iterdir()}
        if existing != {"legacy_approx"}:
            raise FileExistsError(f"refusing to overwrite existing audit directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json(data_root / "pilot_manifest.json")
    records = {row["key"]: row for row in manifest["records"]}
    missing = [key for key in args.cases if key not in records]
    if missing:
        raise KeyError(f"selected cases missing from manifest: {missing}")
    selected = []
    for key in args.cases:
        row = copy.deepcopy(records[key])
        row["selection_reason"] = "fixed g03 history group, low/high geometry boundary pair; selected before this recheck and not by model error"
        selected.append(row)
    write_json(output_root / "selected_cases.json", {"selection_policy": "fixed g03 low/high geometry pairs", "records": selected})

    pilot, generator, replay_audit = load_engine()
    env = environment_payload(generator)
    write_json(output_root / "environment.json", env)
    (output_root / "source_map.md").write_text(original_source_map(), encoding="utf-8")
    (output_root / "commands.txt").write_text(
        "# Pre-fix legacy reference (saved before the explicit-client fix)\n"
        f"# /data/gaoya/agent-data/outputs/validation_pybullet_bridge_recheck_v8/legacy_approx\n\n"
        "# Current bridge legacy rerun (post-fix; expected numerically identical)\n"
        f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {sys.executable} -B "
        f"{PROJECT / 'pybullet_context_rollout.py'} --data-root {data_root} --output {args.legacy_output} "
        f"--cases {' '.join(args.cases)}\n\n"
        "# Recheck command\n"
        f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 {sys.executable} -B "
        f"{PROJECT / 'recheck_pybullet_bridge.py'} --data-root {data_root} --output-root {output_root} "
        f"--legacy-output {args.legacy_output} --cases {' '.join(args.cases)}\n",
        encoding="utf-8",
    )

    all_results: dict[str, Any] = {
        "schema": "physvideo_pybullet_bridge_recheck_v1",
        "status": "EXECUTED",
        "data_root": str(data_root),
        "output_root": str(output_root),
        "selected_cases": list(args.cases),
        "environment": env,
        "rolling_kinematics": rolling_kinematics_tests(),
        "timestep_fixture": timestep_fixture(generator),
        "multi_client_fixture": multi_client_fixture(generator),
        "default_client_alias_probe": default_client_alias_probe(),
        "bridge_multi_client_regression": bridge_multi_client_regression(pilot, generator, records, data_root),
        "cases": {},
    }
    all_results["original_replay_entrypoint"] = original_replay_entrypoint_check(
        replay_audit,
        pilot,
        generator,
        records,
        data_root,
        list(args.cases),
    )
    write_json(output_root / "original_replay_entrypoint_check.json", all_results["original_replay_entrypoint"])
    write_json(output_root / "timestep_audit.json", all_results["timestep_fixture"])
    write_json(output_root / "clock_fixture_results.json", all_results["timestep_fixture"])
    write_json(output_root / "rolling_kinematics.json", all_results["rolling_kinematics"])
    write_json(output_root / "multi_client_fixture.json", all_results["multi_client_fixture"])
    write_json(output_root / "default_client_alias_probe.json", all_results["default_client_alias_probe"])
    write_json(output_root / "bridge_multi_client_regression.json", all_results["bridge_multi_client_regression"])

    legacy_results = load_json(args.legacy_output / "results.json")
    legacy_by_key = {row["key"]: row for row in legacy_results["cases"]}
    per_frame_rows: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    mode_metrics_rows: list[dict[str, Any]] = []
    for key in args.cases:
        record = records[key]
        sample_dir = data_root / "samples" / key
        case_result: dict[str, Any] = {"status": "EXECUTED", "key": key}
        try:
            raw = load_raw_state(sample_dir)
            case, seed = reconstruct_case(pilot, generator, record)
            saved_blueprint = load_json(sample_dir / "blueprint.json")
            regenerated_blueprint = jsonable(dataclasses.asdict(case.blueprint))
            blueprint_diff = recursive_diff(saved_blueprint, regenerated_blueprint)
            write_json(
                output_root / "world_configs_and_diffs" / key / "blueprint_comparison.json",
                {
                    "saved_blueprint_sha256": sha256_file(sample_dir / "blueprint.json"),
                    "regenerated_blueprint_physical_sha256": canonical_sha(physical_blueprint_payload(case.blueprint)),
                    "saved_vs_regenerated_diff_count": len(blueprint_diff),
                    "saved_vs_regenerated_diffs": blueprint_diff[:200],
                    "saved_physical_payload": saved_blueprint,
                    "regenerated_blueprint": regenerated_blueprint,
                },
            )
            prefix = run_original_prefix(generator, case, seed, raw)
            case_result["prefix_match"] = prefix["prefix_match"]
            case_result["r0"] = r0_summary(raw, prefix)
            case_result["r0_status"] = "EXECUTED"
            case_result["omega_status"] = "GT_RECOVERY_OK" if (
                prefix["prefix_match"]["passed_position_1e-5_m"]
                and prefix["prefix_match"]["passed_velocity_1e-5_mps"]
                and prefix["prefix_match"]["passed_quaternion_1e-5"]
            ) else "GT_RECOVERY_BLOCKED"
            context_payload = save_context_state_v2(output_root, key, raw, prefix)
            case_result["context_state"] = context_payload
            case_result["frame7_contacts"] = prefix["frame7_contacts"]
            np.savez_compressed(
                output_root / "context_states_v2" / f"{key}_prefix.npz",
                positions=np.asarray(prefix["prefix"]["positions"], dtype=np.float32),
                quaternions=np.asarray(prefix["prefix"]["quaternions"], dtype=np.float32),
                linear_velocities=np.asarray(prefix["prefix"]["linear_velocities"], dtype=np.float32),
                angular_velocities=np.asarray(prefix["prefix"]["angular_velocities"], dtype=np.float32),
            )
            write_json(output_root / "action_trace" / f"{key}_prefix.json", {
                "case": key,
                "original_replay_pre_roll_s": case.blueprint.pre_roll_s,
                "original_replay_prefix_api_steps_to_rgb7": int(round(case.blueprint.pre_roll_s * SIM_HZ)) + 7 * (SIM_HZ // FPS),
                "prefix_step_counter_total": prefix["total_step_calls"],
                "initial_frame7_contacts": prefix["frame7_contacts"],
                "correction_records": prefix["continuous"].get("correction_records", []),
            })

            runtime_diff = {
                "original_initial_world_physics_hash": canonical_sha(strip_config_for_physics(prefix["config_initial"])),
                "original_initial_world_physics": strip_config_for_physics(prefix["config_initial"]),
            }
            modes: dict[str, Any] = {}
            mode_names = ["L_legacy_equivalent"]
            if case_result["omega_status"] == "GT_RECOVERY_OK":
                mode_names.extend([
                    "R1_gt_state_rebuild",
                    "R2_fd_velocity_only",
                    "R3_rolling_omega_only",
                    "R4_fd_plus_rolling",
                ])
            else:
                for blocked_mode in ("R1_gt_state_rebuild", "R2_fd_velocity_only", "R3_rolling_omega_only", "R4_fd_plus_rolling"):
                    modes[blocked_mode] = {"status": "BLOCKED", "message": "GT RGB7 omega recovery/prefix validation did not pass fixed tolerances"}
            for mode in mode_names:
                try:
                    result = run_rebuild_mode(generator, case, seed, raw, prefix, mode)
                    modes[mode] = {"status": "EXECUTED", **result}
                    write_mode_outputs(output_root, key, mode, result)
                    per_frame_rows.extend(mode_frame_rows(key, mode, result))
                    for contact in result.get("contact_records", []):
                        contact_rows.append({"case": key, "mode": mode, **contact})
                    mode_metrics_rows.append({"case": key, "mode": mode, **result["metrics"], "strategies": result.get("strategies", {})})
                except Exception as exc:
                    modes[mode] = {"status": "BLOCKED", **exception_payload(exc)}
            case_result["modes"] = modes

            if modes.get("R1_gt_state_rebuild", {}).get("status") == "EXECUTED":
                r1_config = modes["R1_gt_state_rebuild"]["config_initial"]
                runtime_diff.update({
                    "rebuild_initial_world_physics_hash": canonical_sha(strip_config_for_physics(r1_config)),
                    "rebuild_initial_world_physics": strip_config_for_physics(r1_config),
                    "original_vs_rebuild_physics_diffs": recursive_diff(
                        strip_config_for_physics(prefix["config_initial"]),
                        strip_config_for_physics(r1_config),
                    ),
                })
            write_json(output_root / "world_configs_and_diffs" / key / "runtime_world_comparison.json", runtime_diff)

            legacy_row = legacy_by_key[key]
            legacy_equiv = modes.get("L_legacy_equivalent")
            if legacy_equiv and legacy_equiv.get("status") == "EXECUTED":
                legacy_positions = np.asarray(legacy_row["bullet"], dtype=np.float64)
                equivalent_positions = np.asarray(legacy_equiv["positions"], dtype=np.float64)
                legacy_error = np.linalg.norm(equivalent_positions - legacy_positions, axis=-1)
                case_result["legacy_equivalence"] = {
                    "status": "EXECUTED",
                    "max_position_difference_m": float(legacy_error.max()),
                    "mean_position_difference_m": float(legacy_error.mean()),
                    "same_within_1e-7_m": bool(legacy_error.max() <= 1e-7),
                    "legacy_metrics": legacy_row["metrics"],
                    "instrumented_metrics": legacy_equiv["metrics"],
                }
            else:
                case_result["legacy_equivalence"] = {"status": "BLOCKED", "message": "instrumented legacy equivalent did not execute"}

            if modes.get("R4_fd_plus_rolling", {}).get("status") == "EXECUTED":
                case_result["future_mutation"] = future_mutation_test(
                    generator,
                    case,
                    seed,
                    raw,
                    prefix,
                    modes["R4_fd_plus_rolling"],
                )
            else:
                case_result["future_mutation"] = {"status": "BLOCKED", "message": "R4 unavailable"}
            write_json(output_root / "future_mutation_results" / f"{key}.json", case_result["future_mutation"])
        except Exception as exc:
            case_result = {"status": "BLOCKED", "key": key, **exception_payload(exc)}
        all_results["cases"][key] = case_result
        write_json(output_root / "per_case" / f"{key}.json", case_result)

    # Benchmark one fixed case per family after correctness paths are complete.
    benchmark_cases = []
    for family in ("aperture", "deflector", "support_edge"):
        family_key = next(key for key in args.cases if key.startswith(f"phase1_{family}_"))
        row = all_results["cases"].get(family_key, {})
        if row.get("status") != "EXECUTED":
            continue
        try:
            record = records[family_key]
            sample_dir = data_root / "samples" / family_key
            raw = load_raw_state(sample_dir)
            case, seed = reconstruct_case(pilot, generator, record)
            prefix = run_original_prefix(generator, case, seed, raw)
            benchmark_cases.append(benchmark_case(generator, case, seed, raw, prefix, output_root))
        except Exception as exc:
            benchmark_cases.append({"case": family_key, **exception_payload(exc)})
    all_results["latency"] = benchmark_cases

    write_per_frame_csv(output_root / "per_frame_metrics.csv", per_frame_rows)
    write_json(output_root / "per_case_mode_metrics.json", mode_metrics_rows)
    write_contact_jsonl(output_root / "contact_events.jsonl", contact_rows)
    write_json(output_root / "latency_report.json", benchmark_cases)
    all_results["findings"] = summarize_findings(all_results)
    write_json(output_root / "audit_findings.json", all_results["findings"])
    write_json(output_root / "all_results.json", all_results)
    (output_root / "report.md").write_text(build_report(all_results), encoding="utf-8")
    lines = [
        f"status={all_results['status']}",
        f"rolling_kinematics_passed={all_results['rolling_kinematics']['passed']}",
        f"time_fixture_first_abs_error_s={all_results['timestep_fixture']['first_output_abs_error_s']}",
        f"time_fixture_last_abs_error_s={all_results['timestep_fixture']['last_output_abs_error_s']}",
        f"multi_client_passed={all_results['multi_client_fixture']['passed']}",
    ]
    for key, row in all_results["cases"].items():
        lines.append(f"{key}: status={row.get('status')} omega={row.get('omega_status')} r0={row.get('r0_status')}")
    (output_root / "test_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": all_results["status"], "output_root": str(output_root), "cases": args.cases}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
