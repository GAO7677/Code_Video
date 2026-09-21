"""Generate the phase-one 36-episode physics-only structured-geometry pilot.

This generator deliberately reuses the repository's original Bullet replay and
its scene constructors, but creates a new versioned protocol with one unified
sphere, common materials, and three named interaction families.  It never
renders RGB, reads future labels while constructing geometry, or applies
post-observation controls.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
ENGINE_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819")
AUDIT_ROOT = Path("/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/original_pipeline_audit_20260917")
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v4")
PROTOCOL_VERSION = "physvideo_next_experiment_protocol_20260921_phase1_v4"
FPS = 30
SIM_HZ = 240
FRAME_COUNT = 49
GROUPS_PER_FAMILY = 4
VARIANTS_PER_GROUP = 3
BALL_RADIUS_M = 0.11
BALL_MASS_KG = 1.0
BALL_FRICTION = 0.35
BALL_RESTITUTION = 0.25
STATIC_FRICTION = 0.35
STATIC_RESTITUTION = 0.25
SIM_SEED_BASE = 2026092100
EPS_GT_M = 1e-6
STRONG_RESPONSE_RADIUS_FRACTION = 0.30

FAMILY_CONFIG = {
    "aperture": {
        "values": (0.28, 0.46, 0.72),
        "units": "m",
        "variable": "opening_width_m",
        "constructor": "_make_door_frame_case",
        "description": "finite door aperture: pass, edge scrape, or block",
    },
    "deflector": {
        "values": (28.0, 58.0, 86.0),
        "units": "deg",
        "variable": "barrier_normal_angle_deg",
        "constructor": "_make_puck_barrier_case",
        "description": "finite rectangular deflector: graze, deflect, or rebound",
    },
    "support_edge": {
        "values": (0.08, 0.30, 0.62),
        "units": "m",
        "variable": "gap_width_m",
        "constructor": "_make_gap_case",
        "description": "finite platform support edge: bridge, leave edge, or fall",
    },
}

# These controls are frozen before any replay result is inspected.  Each family
# uses the same four history-group rows; only the static geometry parameter is
# changed within a group.
HISTORY_CONTROLS = (
    {"speed_mps": 2.00, "heading_deg": 0.0, "offset_y_m": 0.10, "start_x_m": -1.40},
    {"speed_mps": 2.20, "heading_deg": 1.0, "offset_y_m": 0.14, "start_x_m": -1.40},
    {"speed_mps": 2.40, "heading_deg": -1.0, "offset_y_m": -0.08, "start_x_m": -1.40},
    {"speed_mps": 2.60, "heading_deg": 1.0, "offset_y_m": 0.12, "start_x_m": -1.40},
)

FAMILY_HISTORY_CONTROLS = {
    "aperture": HISTORY_CONTROLS,
    "deflector": (
        {"speed_mps": 2.20, "heading_deg": -3.0, "offset_y_m": 0.08, "start_x_m": -1.55},
        {"speed_mps": 2.50, "heading_deg": 2.0, "offset_y_m": 0.14, "start_x_m": -1.55},
        {"speed_mps": 2.80, "heading_deg": -2.0, "offset_y_m": -0.10, "start_x_m": -1.55},
        {"speed_mps": 3.10, "heading_deg": 4.0, "offset_y_m": 0.18, "start_x_m": -1.55},
    ),
    "support_edge": (
        {"speed_mps": 1.45, "heading_deg": -3.0, "offset_y_m": 0.08, "start_x_m": -1.55},
        {"speed_mps": 1.70, "heading_deg": 2.0, "offset_y_m": 0.14, "start_x_m": -1.55},
        {"speed_mps": 1.95, "heading_deg": -2.0, "offset_y_m": -0.10, "start_x_m": -1.55},
        {"speed_mps": 2.15, "heading_deg": 4.0, "offset_y_m": 0.18, "start_x_m": -1.55},
    ),
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_engine():
    sys.path[:0] = [str(ENGINE_ROOT), str(AUDIT_ROOT)]
    from scripts import generate_v2v_context_demos as generator
    import replay_audit

    generator.SIM_DURATION_S = FRAME_COUNT / FPS
    return generator, replay_audit


def _dynamic_control(family: str, control: dict[str, float], source_object):
    theta = math.radians(float(control["heading_deg"]))
    speed = float(control["speed_mps"])
    velocity = (speed * math.cos(theta), speed * math.sin(theta), 0.0)
    if family == "support_edge":
        z = 0.48 + BALL_RADIUS_M
    else:
        z = BALL_RADIUS_M
    angular = (velocity[1] / BALL_RADIUS_M, -velocity[0] / BALL_RADIUS_M, 0.0)
    return replace(
        source_object,
        name="pilot_ball",
        family_key="pilot_unified_sphere",
        shape="sphere",
        size={"radius": BALL_RADIUS_M},
        mass=BALL_MASS_KG,
        friction=BALL_FRICTION,
        restitution=BALL_RESTITUTION,
        linear_damping=0.01,
        angular_damping=0.02,
        role="dynamic",
        semantic_role="dynamic",
        position=(float(control["start_x_m"]), float(control["offset_y_m"]), z),
        linear_velocity=velocity,
        angular_velocity=angular,
        metadata={"appearance_group": "physvideo_phase1_unified_ball_v1"},
    )


def make_case(generator, family: str, group_index: int, value, seed: int):
    config = FAMILY_CONFIG[family]
    key = f"phase1_{family}_g{group_index:02d}_v{int(round(float(value) * 1000)):04d}"
    if family == "aperture":
        base = generator._make_door_frame_case(key, float(value), moving_object="ball")
    elif family == "deflector":
        base = generator._make_puck_barrier_case(key, float(value))
    elif family == "support_edge":
        base = generator._make_gap_case(key, float(value))
    else:  # pragma: no cover
        raise ValueError(family)
    control = dict(FAMILY_HISTORY_CONTROLS[family][group_index])
    objects = []
    for obj in base.blueprint.objects:
        if obj.dynamic:
            objects.append(_dynamic_control(family, control, obj))
        else:
            objects.append(replace(obj, friction=STATIC_FRICTION, restitution=STATIC_RESTITUTION))
    metadata = {
        **base.blueprint.metadata,
        "phase1_protocol": PROTOCOL_VERSION,
        "pilot_family": family,
        "controlled_variable": config["variable"],
        "controlled_value": float(value),
        "unified_object_protocol": {
            "shape": "sphere",
            "radius_m": BALL_RADIUS_M,
            "mass_kg": BALL_MASS_KG,
            "friction": BALL_FRICTION,
            "restitution": BALL_RESTITUTION,
            "gravity_mps2": 9.81,
        },
        "static_material_protocol": {
            "friction": STATIC_FRICTION,
            "restitution": STATIC_RESTITUTION,
            "floor_friction": STATIC_FRICTION,
        },
        "physics_sub_steps": 8,
        "floor_friction": STATIC_FRICTION,
        "post_observation_control_log": {
            "force_commands": [],
            "torque_commands": [],
            "pose_overwrites": [],
            "velocity_overwrites": [],
            "static_geometry_updates": [],
            "rebound_corrections": [],
            "status": "none_declared_in_original_replay",
        },
        "future_state_used_for_geometry": False,
        "future_event_used_for_geometry": False,
        "rgb_generated": False,
        "visual_frontend_extracted": False,
    }
    # The original constructor's contact contract reflects its source dynamic
    # object. Rebuild it after standardizing the ball and all static materials.
    provisional = replace(
        base.blueprint,
        family_key=f"PILOT_{family.upper()}",
        objects=tuple(objects),
        surface_key="residential_wood_floor",
        metadata=metadata,
    )
    metadata["initialization_contract"] = generator.build_contact_contract(provisional)
    blueprint = replace(provisional, metadata=metadata)
    generator._validate_blueprint(blueprint)
    family_key = f"PILOT_{family.upper()}"
    return replace(
        base,
        case_id=key,
        family_key=family_key,
        family_title=family,
        family_description=config["description"],
        title=f"Phase-one {family} value={value}",
        description=config["description"],
        controlled_variable=config["variable"],
        controlled_value=float(value),
        controlled_value_label=f"{config['variable']}={float(value):.4f}",
        units=config["units"],
        blueprint=blueprint,
    )


def _quat_matrix_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in quat]
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def finite_surfaces(record, objects, positions, quats, max_surfaces=48):
    """Extract finite box faces plus a finite local floor ROI from frame 0."""
    centers, normals, tangent_u, half_extents, names = [], [], [], [], []
    static_boxes = []
    object_names = [obj.name for obj in objects]
    for index, obj in enumerate(objects):
        if obj.dynamic:
            continue
        if obj.shape != "box":
            raise ValueError(f"unsupported static pilot shape {obj.shape}")
        half = np.asarray([obj.size[k] for k in ("hx", "hy", "hz")], dtype=np.float32)
        center = positions[0, index].astype(np.float32)
        rotation = _quat_matrix_xyzw(quats[0, index])
        static_boxes.append((center, half, rotation, obj.name))
        faces = ((0, -1, 1, 2), (0, 1, 1, 2),
                 (1, -1, 0, 2), (1, 1, 0, 2),
                 (2, -1, 0, 1), (2, 1, 0, 1))
        local_axis = np.eye(3, dtype=np.float32)
        for fixed, sign, axis_u, axis_v in faces:
            centers.append(center + rotation @ (local_axis[fixed] * sign * half[fixed]))
            normals.append(rotation @ (local_axis[fixed] * sign))
            tangent_u.append(rotation @ local_axis[axis_u])
            half_extents.append([half[axis_u], half[axis_v]])
            names.append(obj.name)
    if not static_boxes:
        raise ValueError("pilot case has no static boxes")
    mins = np.min([center[:2] - (rotation[:2, :] ** 2 @ half ** 2) ** 0.5 for center, half, rotation, _ in static_boxes], axis=0) - 1.5
    maxs = np.max([center[:2] + (rotation[:2, :] ** 2 @ half ** 2) ** 0.5 for center, half, rotation, _ in static_boxes], axis=0) + 1.5
    centers.append(np.asarray([(mins[0] + maxs[0]) / 2, (mins[1] + maxs[1]) / 2, 0.0], dtype=np.float32))
    normals.append(np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
    tangent_u.append(np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    half_extents.append([(maxs[0] - mins[0]) / 2, (maxs[1] - mins[1]) / 2])
    names.append("finite_floor_roi")
    if len(centers) > max_surfaces:
        raise ValueError(f"surface count {len(centers)} exceeds max {max_surfaces}")
    padded = {
        "surface_center": np.zeros((max_surfaces, 3), dtype=np.float32),
        "surface_normal": np.zeros((max_surfaces, 3), dtype=np.float32),
        "surface_tangent_u": np.zeros((max_surfaces, 3), dtype=np.float32),
        "surface_half_extents": np.zeros((max_surfaces, 2), dtype=np.float32),
        "surface_mask": np.zeros((max_surfaces,), dtype=bool),
        "surface_name": np.full((max_surfaces,), "__padding__", dtype="U64"),
    }
    n = len(centers)
    padded["surface_center"][:n] = np.asarray(centers, dtype=np.float32)
    padded["surface_normal"][:n] = np.asarray(normals, dtype=np.float32)
    padded["surface_tangent_u"][:n] = np.asarray(tangent_u, dtype=np.float32)
    padded["surface_half_extents"][:n] = np.asarray(half_extents, dtype=np.float32)
    padded["surface_mask"][:n] = True
    padded["surface_name"][:n] = np.asarray(names, dtype="U64")
    return padded


def interaction_metadata(family: str, blueprint, positions, dynamic_index):
    meta = blueprint.metadata
    if family == "aperture":
        x = float(meta["door_frame_center_x_m"])
        rule = "target_x_near_door_frame"
    elif family == "deflector":
        x = float(meta["barrier_center_x_m"])
        rule = "target_x_near_deflector"
    else:
        x = float(meta["left_platform_edge_x_m"])
        rule = "target_crosses_left_platform_edge"
    trajectory = positions[:, dynamic_index, 0]
    if family == "support_edge":
        candidate = np.flatnonzero(trajectory[8:] > x + 0.02)
    else:
        candidate = np.flatnonzero(np.abs(trajectory[8:] - x) <= 0.20)
    first = int(candidate[0] + 8) if len(candidate) else None
    relative_time = None if first is None else float((first - 7) / FPS)
    return {
        "rule": rule,
        "reference_x_m": x,
        "first_future_frame": first,
        "first_future_time_s": None if first is None else float(first / FPS),
        "first_future_time_after_rgb7_s": relative_time,
        "within_recommended_window_0p25_0p90_s": bool(first is not None and 0.25 <= relative_time <= 0.90),
    }


def write_episode(output: Path, record, replay_audit, generator, point_seed: int):
    dest = output / "samples" / record["key"]
    dest.joinpath("raw").mkdir(parents=True, exist_ok=False)
    dest.joinpath("geometry").mkdir(parents=True, exist_ok=False)
    objects, positions, velocities, quats, corrections = replay_audit.replay(record["case"], record["simulation_seed"])
    if positions.shape != (FRAME_COUNT, len(objects), 3) or velocities.shape != positions.shape or quats.shape != (FRAME_COUNT, len(objects), 4):
        raise ValueError(f"unexpected replay shape for {record['key']}")
    if not all(np.isfinite(x).all() for x in (positions, velocities, quats)):
        raise ValueError(f"non-finite replay for {record['key']}")
    dynamic = [i for i, obj in enumerate(objects) if obj.dynamic]
    if dynamic != [0] or objects[0].name != "pilot_ball":
        raise ValueError(f"expected one pilot_ball dynamic object: {record['key']}")
    if corrections != 0:
        raise ValueError(f"unexpected rebound correction in unified pilot: {record['key']}: {corrections}")
    surfaces = finite_surfaces(record, objects, positions, quats)
    from finite_surface_geometry import canonicalize_surfaces, sample_surface_points
    clean = canonicalize_surfaces(
        surfaces["surface_center"], surfaces["surface_normal"], surfaces["surface_tangent_u"],
        surfaces["surface_half_extents"], surfaces["surface_mask"],
    )
    points = sample_surface_points(clean, point_seed, count=512, resolution=10)
    np.savez_compressed(
        dest / "raw/states_xyzw.npz", positions=positions, linear_velocities=velocities,
        quats=quats, object_names=np.asarray([obj.name for obj in objects]),
        frame_times=np.arange(FRAME_COUNT, dtype=np.float32) / FPS,
    )
    np.savez_compressed(
        dest / "geometry/finite_surfaces.npz",
        surface_center=surfaces["surface_center"], surface_normal=clean["surface_normal"],
        surface_tangent_u=clean["surface_tangent_u"], surface_half_extents=surfaces["surface_half_extents"],
        surface_mask=surfaces["surface_mask"], surface_name=surfaces["surface_name"],
        point_xyz=points["point_xyz"], point_mask=np.ones(512, dtype=bool),
        point_source_surface=points["point_source_surface"],
    )
    dynamic_index = dynamic[0]
    context = np.concatenate((positions[:8, dynamic_index].reshape(-1),
                              velocities[:8, dynamic_index].reshape(-1),
                              quats[:8, dynamic_index].reshape(-1))).astype(np.float32)
    history_hash = hashlib.sha256(context.tobytes()).hexdigest()
    motion_context = np.concatenate((positions[:8, dynamic_index].reshape(-1),
                                     velocities[:8, dynamic_index].reshape(-1),
                                     np.full(3, BALL_RADIUS_M * 2, dtype=np.float32),
                                     np.arange(8, dtype=np.float32) / FPS)).astype(np.float32)
    motion_hash = hashlib.sha256(motion_context.tobytes()).hexdigest()
    target = positions[8:FRAME_COUNT, dynamic_index].astype(np.float32)
    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
    metadata = {
        **record["case"].blueprint.metadata,
        "sample_id": record["key"], "pair_group": record["group_id"], "split": "train_pilot",
        "family": record["family"], "geometry_value": record["geometry_value"],
        "simulation": {"fps": FPS, "sim_hz": SIM_HZ, "frame_count": FRAME_COUNT,
                        "pre_roll_s": float(record["case"].blueprint.pre_roll_s)},
        "actors": {obj.name: {
            "object_id": obj.family_key, "role": obj.role, "dynamic": obj.dynamic,
            "shape": obj.shape, "size_m": obj.size, "mass_kg": obj.mass,
            "friction": obj.friction, "restitution": obj.restitution,
            "position_m": obj.position, "linear_velocity_mps": obj.linear_velocity,
            "angular_velocity_radps": obj.angular_velocity,
        } for obj in objects},
        "source": str(replay_audit.SOURCE), "rebound_corrections": int(corrections),
        "rgb_generated": False, "vggt_sam2_utonia_extracted": False,
        "post_rgb7_external_action_status": "NONE_DECLARED",
        "full_state_saved": True,
    }
    dump(dest / "metadata.json", metadata)
    dump(dest / "blueprint.json", asdict(record["case"].blueprint))
    dump(dest / "raw/trajectories.json", {
        "object_names": [obj.name for obj in objects],
        "frame_times_s": (np.arange(FRAME_COUNT, dtype=np.float32) / FPS).tolist(),
        "dynamic_name": objects[dynamic_index].name,
        "observed_positions": positions[:8, dynamic_index].tolist(),
        "observed_velocities": velocities[:8, dynamic_index].tolist(),
        "future_positions": target.tolist(),
    })
    surface_report = {
        "schema": "finite_surface_geometry_v1",
        "valid_surface_count": int(surfaces["surface_mask"].sum()),
        "padded_surface_count": int(len(surfaces["surface_mask"])),
        "surface_names": surfaces["surface_name"][surfaces["surface_mask"]].tolist(),
        "surface_center_shape": [int(x) for x in surfaces["surface_center"].shape],
        "surface_normal_shape": [int(x) for x in surfaces["surface_normal"].shape],
        "surface_tangent_u_shape": [int(x) for x in surfaces["surface_tangent_u"].shape],
        "surface_half_extents_shape": [int(x) for x in surfaces["surface_half_extents"].shape],
        "point_valid_count": 512, "point_padded_token_count": 1792,
        "point_candidate_pool_hash": points["candidate_pool_hash"],
        "point_candidate_pool_count": int(points["candidate_pool_count"][0]),
        "point_sample_seed": int(point_seed),
        "finite_floor_roi": True,
        "future_used": False,
    }
    dump(dest / "geometry/report.json", surface_report)
    receipt = {
        "schema": "physvideo_phase1_replay_receipt_v1", "key": record["key"],
        "original_bullet_replay": True, "cpu_only": True, "gpu_used": False,
        "post_observation_controls": {"force": False, "torque": False, "pose": False,
                                       "velocity": False, "static_geometry": False,
                                       "rebound_correction": False},
        "output_sha256": {name: sha(dest / name) for name in (
            "metadata.json", "blueprint.json", "raw/states_xyzw.npz", "raw/trajectories.json",
            "geometry/finite_surfaces.npz", "geometry/report.json")},
    }
    dump(dest / "replay.json", receipt)
    return {
        **record,
        "history_hash": history_hash,
        "motion_hash": motion_hash,
        "target_hash": target_hash,
        "static_geometry_hash": json_sha([
            [obj.name, obj.position, obj.orientation_euler_deg, obj.size, obj.friction, obj.restitution]
            for obj in objects if not obj.dynamic
        ]),
        "rebound_corrections": int(corrections),
        "observed_motion": {"positions": positions[:8, dynamic_index].tolist(),
                             "velocities": velocities[:8, dynamic_index].tolist(),
                             "future_dt": (np.arange(1, 42, dtype=np.float32) / FPS).tolist()},
        "interaction": interaction_metadata(record["family"], record["case"].blueprint, positions, dynamic_index),
        "surface_report": surface_report,
    }


def pair_rows(records):
    rows = []
    grouped = {}
    for record in records:
        grouped.setdefault(record["group_id"], []).append(record)
    radius = BALL_RADIUS_M
    for group_id, values in sorted(grouped.items()):
        values = sorted(values, key=lambda row: row["geometry_value"])
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                left, right = values[left_index], values[right_index]
                with np.load(Path(left["sample_dir"]) / "raw/states_xyzw.npz", allow_pickle=False) as a:
                    ta = a["positions"][8:FRAME_COUNT, 0]
                with np.load(Path(right["sample_dir"]) / "raw/states_xyzw.npz", allow_pickle=False) as b:
                    tb = b["positions"][8:FRAME_COUNT, 0]
                d_gt = float(np.linalg.norm(ta - tb, axis=-1).mean())
                if d_gt <= EPS_GT_M:
                    layer = "equal_future"
                elif d_gt < STRONG_RESPONSE_RADIUS_FRACTION * radius:
                    layer = "weak_response"
                else:
                    layer = "strong_response"
                rows.append({
                    "family": left["family"], "group_id": group_id,
                    "split": "train_pilot", "geometry_a": left["geometry_value"],
                    "geometry_b": right["geometry_value"], "D_gt_m": d_gt,
                    "response_layer": layer, "history_equal": left["history_hash"] == right["history_hash"],
                    "motion_equal": left["motion_hash"] == right["motion_hash"],
                    "static_geometry_different": left["static_geometry_hash"] != right["static_geometry_hash"],
                    "event_a": left["interaction"], "event_b": right["interaction"],
                })
    return rows


def full_data_config():
    return {
        "schema": "physvideo_next_experiment_full_data_config_v1",
        "status": "IMPLEMENTED_NOT_RUN",
        "protocol_version": PROTOCOL_VERSION,
        "families": list(FAMILY_CONFIG),
        "histories_per_family": {"train": 32, "dev": 8, "locked_test": 8},
        "episodes_per_history": 3,
        "episodes": {"train": 288, "dev": 72, "locked_test": 72, "total": 432},
        "pilot": {"histories_per_family": 4, "episodes": 36, "role": "train_side_pilot_only"},
        "geometry_values": {name: list(cfg["values"]) for name, cfg in FAMILY_CONFIG.items()},
        "group_split": "history_group_frozen_before_replay; all variants stay together",
        "response_layers": {"equal_future": f"D_gt <= {EPS_GT_M} m",
                             "weak_response": f"{EPS_GT_M} < D_gt < {STRONG_RESPONSE_RADIUS_FRACTION} * radius",
                             "strong_response": f"D_gt >= {STRONG_RESPONSE_RADIUS_FRACTION} * radius"},
        "physics": {"engine": "original PyBullet replay_audit", "sim_hz": SIM_HZ, "fps": FPS,
                     "frames": FRAME_COUNT, "substeps": 8, "unified_sphere_radius_m": BALL_RADIUS_M,
                     "mass_kg": BALL_MASS_KG, "friction": BALL_FRICTION, "restitution": BALL_RESTITUTION,
                     "post_observation_external_controls": False},
        "representation": {"point_arm": {"valid_points": 512, "padded_tokens": 1792,
                                             "sampling": "one candidate per valid finite surface, then seeded global fill"},
                            "finite_surface_arm": {"max_tokens": 48,
                                                     "fields": ["center[3]", "normal[3]", "tangent_u[3]", "half_extents[2]", "valid"]}},
        "training": {"arms": ["motion_only", "point_geometry_resample", "finite_surface_geometry"],
                     "analytic_reference": "analytic_cv", "max_optimizer_steps_profile": 20,
                     "full_budget": {"epochs": 200, "history_group_batch": 4,
                                     "episodes_per_update": 12, "optimizer_steps": 4800,
                                     "seeds": [42, 43, 44], "status": "IMPLEMENTED_NOT_RUN"},
                     "optimizer": "AdamW(lr=3e-4, weight_decay=0.01, clip=1.0)",
                     "loss": "SmoothL1(position)+0.2*SmoothL1(interval_velocity)",
                     "normalization": "training observation motion only"},
        "evaluation": {"metrics": ["ADE_m", "FDE_m", "interval_velocity_MAE_mps", "D_gt_m", "D_pred_m", "E_delta_m", "R_delta", "equal_future_absolute_prediction_delta", "sampling_stability", "penetration_depth", "penetration_frame_rate", "event_before_after_error"],
                        "invalid_metric_policy": "N/A_or_NOT_RUN_not_zero", "short_profile_metrics_not_for_model_selection": True},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--point-seed", type=int, default=2026092111)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    generator, replay_audit = load_engine()
    records, groups = [], {}
    for family_index, family in enumerate(FAMILY_CONFIG):
        for group_index, control in enumerate(FAMILY_HISTORY_CONTROLS[family]):
            group_id = f"pilot_{family}_g{group_index:02d}"
            group_records = []
            for value_index, value in enumerate(FAMILY_CONFIG[family]["values"]):
                case = make_case(generator, family, group_index, value, SIM_SEED_BASE + family_index * 100 + group_index)
                record = {
                    "key": case.case_id, "family": family, "group_id": group_id,
                    "history_index": group_index, "geometry_index": value_index,
                    "geometry_value": float(value), "geometry_units": FAMILY_CONFIG[family]["units"],
                    "simulation_seed": SIM_SEED_BASE + family_index * 100 + group_index,
                    "case": case,
                    "sample_dir": str(output / "samples" / case.case_id),
                    "point_seed": args.point_seed + family_index * 10000 + group_index * 100 + value_index,
                    "control": control,
                }
                completed = write_episode(output, record, replay_audit, generator, record["point_seed"])
                group_records.append(completed); records.append(completed)
                print("PHASE1_EPISODE_READY", family, group_id, value, flush=True)
            if len({row["history_hash"] for row in group_records}) != 1 or len({row["motion_hash"] for row in group_records}) != 1:
                raise AssertionError(f"same-group observation input mismatch: {group_id}")
            if len({row["static_geometry_hash"] for row in group_records}) != VARIANTS_PER_GROUP:
                raise AssertionError(f"same-group geometry did not vary: {group_id}")
            groups[group_id] = group_records
            print("PHASE1_GROUP_READY", group_id, flush=True)
    rows = pair_rows(records)
    layer_counts = {}
    for row in rows:
        layer_counts[f"{row['family']}/{row['response_layer']}"] = layer_counts.get(f"{row['family']}/{row['response_layer']}", 0) + 1
    manifest_records = []
    for row in records:
        manifest_records.append({key: value for key, value in row.items() if key not in {"case", "observed_motion"}})
    manifest = {
        "schema": "physvideo_phase1_pilot_manifest_v1", "status": "EXECUTED",
        "protocol_version": PROTOCOL_VERSION, "role": "train_side_pilot_only",
        "families": list(FAMILY_CONFIG), "groups_total": len(groups),
        "episodes_total": len(records), "episodes_per_group": VARIANTS_PER_GROUP,
        "records": manifest_records, "pair_rows": rows, "response_layer_counts": layer_counts,
        "history_input_checks": {"all_group_history_equal": True, "all_group_motion_equal": True,
                                  "forward_fields": ["motion", "object_mask", "last_position", "last_velocity", "future_dt"]},
        "physics": {"engine": "original Bullet replay_audit", "source": str(replay_audit.SOURCE),
                     "cpu_only": True, "gpu_used": False, "sim_hz": SIM_HZ, "fps": FPS,
                     "frame_count": FRAME_COUNT, "rebound_corrections_total": int(sum(r["rebound_corrections"] for r in records))},
        "future_used_for_geometry": False, "future_used_for_input": False,
        "rgb_generated": False, "utonia_extracted": False,
        "full_data_config": full_data_config(),
        "source_sha256": {str(path): sha(path) for path in (PROJECT / "prepare_physvideo_phase1_pilot.py", ENGINE_ROOT / "scripts/generate_v2v_context_demos.py", AUDIT_ROOT / "replay_audit.py")},
    }
    dump(output / "physics_protocol.json", {"schema": PROTOCOL_VERSION, "status": "EXECUTED", "physics": manifest["physics"], "object_protocol": {"radius_m": BALL_RADIUS_M, "mass_kg": BALL_MASS_KG, "friction": BALL_FRICTION, "restitution": BALL_RESTITUTION}, "families": FAMILY_CONFIG, "history_controls": FAMILY_HISTORY_CONTROLS, "response_thresholds": {"epsilon_gt_m": EPS_GT_M, "strong_radius_fraction": STRONG_RESPONSE_RADIUS_FRACTION}, "post_observation_controls": "none_declared"})
    dump(output / "pilot_manifest.json", manifest)
    dump(output / "full_432_data_config.json", full_data_config())
    dump(output / "response_layer_summary.json", {"status": "EXECUTED", "counts": layer_counts, "pair_count": len(rows), "pair_definition": "mean_t norm(P_A(t)-P_B(t))", "thresholds": {"epsilon_gt_m": EPS_GT_M, "strong_radius_fraction": STRONG_RESPONSE_RADIUS_FRACTION}})
    (output / "generation_command.txt").write_text("CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B " + str(Path(__file__).resolve()) + " --output " + str(output) + "\n")
    print(json.dumps({"status": "EXECUTED", "output": str(output), "groups": len(groups), "episodes": len(records), "pairs": len(rows), "response_layer_counts": layer_counts}, indent=2))


if __name__ == "__main__":
    main()
