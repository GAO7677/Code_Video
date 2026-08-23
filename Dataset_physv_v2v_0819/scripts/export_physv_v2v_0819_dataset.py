"""Export PhysV V2V controls as a RigidBench-style physics dataset.

The source previews contain useful RGB videos, simulator states, and instance
IDs, but do not persist lossless RGB, depth, or contact records.  This exporter
replays the deterministic V2V, F11 table-height, and F12 incline cases and
writes one portable sample directory per case under ``samples/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .common_specs import CameraSpec, ScenarioBlueprint
from .caption_templates_0819 import CAPTION_FILES, CAPTION_SCHEMA_VERSION, attach_caption_metadata
from .caption_observations_0819 import derive_caption_observations
from .generate_difficulty_pilot import (
    ANALYSIS_QUESTION,
    RAMP_INCLINE_CASES,
    RAMP_LENGTH_CONTROL_CASES,
    TABLE_HEIGHT_CONTROL_CASES,
)
from .generate_v2v_context_demos import (
    FPS,
    SCENE_STYLE as V2V_SCENE_STYLE,
    V2V_QUESTION,
    DemoCase,
    _render_case as render_v2v_case,
    build_demo_cases,
)
from .group_invariants_0819 import audit_group_invariants
from .render_sim_0705 import render_blueprint_case
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    generate_scenario_blueprint,
)
from .taxonomy_0819 import TAXONOMY_DEFINITIONS, taxonomy_for_family

try:
    from . import generate_sim_preview_gallery as legacy
except ImportError:  # pragma: no cover - direct script fallback
    import generate_sim_preview_gallery as legacy


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_V2V_SEED_BASE = 20260819
DEFAULT_DIFFICULTY_SEED_BASE = 20260817
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
CONTEXT_FRAME_OPTIONS = (8, 16)
SCHEMA_VERSION = "physv_v2v_rigidbench_style_v2"
MOTION_THRESHOLD_MPS = 0.03


@dataclass(frozen=True)
class ExportCase:
    case_id: str
    source_group: str
    family_key: str
    task_type: str
    title: str
    description: str
    analysis_question: str
    controlled_variable: str
    controlled_value: float
    controlled_value_label: str
    units: str
    event_rule: str
    blueprint: ScenarioBlueprint
    seed: int
    scene_style: str
    v2v_case: DemoCase | None = None
    taxonomy: str = "Relation"


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _decode_name(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _make_f11_cases(seed_base: int) -> list[ExportCase]:
    cases: list[ExportCase] = []
    seed = int(seed_base + 88000)
    for extra in TABLE_HEIGHT_CONTROL_CASES:
        table_height_m = float(extra["table_height_m"])
        case_id = f"difficulty_l2_f11_h{int(round(table_height_m * 100)):03d}_{extra['angle_label']}"
        blueprint = generate_scenario_blueprint(
            family_key="F11",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            table_height_m=table_height_m,
            initial_speed_mps=1.25,
            travel_angle_deg=float(extra["travel_angle_deg"]),
        )
        cases.append(
            ExportCase(
                case_id=case_id,
                source_group="f11_table_height",
                family_key="F11",
                task_type="table_rolloff",
                title=blueprint.title,
                description=blueprint.description,
                analysis_question=ANALYSIS_QUESTION,
                controlled_variable="table_height_m",
                controlled_value=table_height_m,
                controlled_value_label=f"{table_height_m:.2f} m",
                units="m",
                event_rule="ball_crosses_table_right_edge",
                blueprint=blueprint,
                seed=seed,
                scene_style="indoor_realistic",
                taxonomy="Scene",
            )
        )
    return cases


def _make_f12_cases(seed_base: int) -> list[ExportCase]:
    cases: list[ExportCase] = []
    seed = int(seed_base + 99000)
    for extra in RAMP_INCLINE_CASES:
        angle_deg = float(extra["ramp_angle_deg"])
        case_id = f"difficulty_l2_f12_{extra['angle_label']}"
        blueprint = generate_scenario_blueprint(
            family_key="F12",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            ramp_angle_deg=angle_deg,
        )
        cases.append(
            ExportCase(
                case_id=case_id,
                source_group="f12_incline",
                family_key="F12",
                task_type="incline_release",
                title=blueprint.title,
                description=blueprint.description,
                analysis_question=ANALYSIS_QUESTION,
                controlled_variable="ramp_angle_deg",
                controlled_value=angle_deg,
                controlled_value_label=f"{angle_deg:.0f} deg",
                units="deg",
                event_rule="block_exits_ramp_lower_edge",
                blueprint=blueprint,
                seed=seed,
                scene_style="indoor_realistic",
                taxonomy="Scene",
            )
        )
    return cases


def _make_f12_length_cases(seed_base: int) -> list[ExportCase]:
    cases: list[ExportCase] = []
    seed = int(seed_base + 99100)
    for extra in RAMP_LENGTH_CONTROL_CASES:
        ramp_length_m = float(extra["ramp_length_m"])
        support_height_m = float(extra["ramp_support_height_m"])
        case_id = f"difficulty_l2_f12_length_{extra['length_label']}"
        base_blueprint = generate_scenario_blueprint(
            family_key="F12",
            sample_key=case_id,
            seed=seed,
            direction_mode="left_to_right",
            size_scale=1.0,
            camera_distance_scale=DEFAULT_CAMERA_DISTANCE_SCALE,
            ramp_length_m=ramp_length_m,
            ramp_support_height_m=support_height_m,
        )
        ramp_angle_deg = float(base_blueprint.metadata["ramp_angle_deg"])
        blueprint = replace(
            base_blueprint,
            title=(
                f"Red wooden block release on a {ramp_length_m:.2f} m incline "
                f"({ramp_angle_deg:.1f} degrees)"
            ),
            description=(
                "A red wooden block is released from rest on an incline whose high-end "
                "support height is fixed; changing the ramp length changes its slope."
            ),
            metadata={
                **base_blueprint.metadata,
                "ramp_angle_deg": ramp_angle_deg,
                "ramp_length_m": ramp_length_m,
                "controlled_variable": "ramp_length_m",
                "ramp_support_height_m": support_height_m,
                "ramp_length_control_group": "fixed_high_support_height",
            },
        )
        cases.append(
            ExportCase(
                case_id=case_id,
                source_group="f12_ramp_length",
                family_key="F12_RAMP_LENGTH",
                task_type="incline_length_release",
                title=blueprint.title,
                description=blueprint.description,
                analysis_question=ANALYSIS_QUESTION,
                controlled_variable="ramp_length_m",
                controlled_value=ramp_length_m,
                controlled_value_label=f"{ramp_length_m:.2f} m",
                units="m",
                event_rule="block_exits_ramp_lower_edge",
                blueprint=blueprint,
                seed=seed,
                scene_style="indoor_realistic",
                taxonomy="Scene",
            )
        )
    return cases


def _v2v_task_type(family_key: str) -> str:
    return {
        "V2V_GAP": "gap_rolloff",
        "V2V_OBSTACLE": "obstacle_collision",
        "V2V_OBSTACLE_SIZE": "obstacle_collision",
        "V2V_BOWL": "bowl_descent",
        "V2V_PENDULUM": "pendulum_swing",
        "V2V_PENDULUM_CABINET": "pendulum_cabinet_collision",
        "V2V_SEESAW": "seesaw_rotation",
        "V2V_DOMINO": "domino_chain",
        "SCENE_PUCK_BARRIER": "puck_barrier_collision",
        "SCENE_DOOR_FRAME": "door_frame_clearance",
        "SCENE_DOOR_FRAME_BALL": "door_frame_clearance_ball",
    }[family_key]


def _make_v2v_cases(seed_base: int) -> list[ExportCase]:
    cases: list[ExportCase] = []
    for index, demo in enumerate(build_demo_cases(seed_base)):
        source_group = {
            "V2V_OBSTACLE_SIZE": "v2v_obstacle_ball_size",
            "V2V_PENDULUM_CABINET": "v2v_pendulum_cabinet_height",
            "SCENE_PUCK_BARRIER": "scene_puck_barrier",
            "SCENE_DOOR_FRAME": "scene_door_frame",
            "SCENE_DOOR_FRAME_BALL": "scene_door_frame_ball",
        }.get(demo.family_key, "v2v_control")
        cases.append(
            ExportCase(
                case_id=demo.case_id,
                source_group=source_group,
                family_key=demo.family_key,
                task_type=_v2v_task_type(demo.family_key),
                title=demo.title,
                description=demo.description,
                analysis_question=V2V_QUESTION,
                controlled_variable=demo.controlled_variable,
                controlled_value=float(demo.controlled_value),
                controlled_value_label=demo.controlled_value_label,
                units=demo.units,
                event_rule=demo.event_rule,
                blueprint=demo.blueprint,
                # Keep the seed fixed within a control group.  The obstacle
                # group must differ only by the initial speed; other families
                # retain their existing per-case seed convention.
                seed=(
                    int(seed_base + 5000)
                    if demo.family_key in {
                        "V2V_OBSTACLE",
                        "V2V_OBSTACLE_SIZE",
                        "V2V_PENDULUM_CABINET",
                    }
                    else int(seed_base + index * 1009)
                ),
                scene_style=V2V_SCENE_STYLE,
                v2v_case=demo,
                taxonomy=taxonomy_for_family(demo.family_key),
            )
        )
    return cases


def build_export_cases(
    *,
    v2v_seed_base: int = DEFAULT_V2V_SEED_BASE,
    difficulty_seed_base: int = DEFAULT_DIFFICULTY_SEED_BASE,
) -> list[ExportCase]:
    cases = (
        _make_v2v_cases(v2v_seed_base)
        + _make_f11_cases(difficulty_seed_base)
        + _make_f12_cases(difficulty_seed_base)
        + _make_f12_length_cases(difficulty_seed_base)
    )
    ids = [case.case_id for case in cases]
    if len(cases) != 70 or len(set(ids)) != len(ids):
        raise RuntimeError(f"expected 70 unique V2V/F11/F12 cases, got {len(cases)}")
    audit_group_invariants(cases)
    return cases


def _remove_or_skip_sample(sample_dir: Path, *, overwrite: bool) -> bool:
    """Return True when a completed sample should be skipped."""
    if not sample_dir.exists():
        return False
    if overwrite:
        shutil.rmtree(sample_dir)
        return False
    if (sample_dir / "meta.json").exists() and (sample_dir / "manifest.json").exists():
        return True
    shutil.rmtree(sample_dir)
    return False


def _move(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.move(str(source), str(destination))


def _hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _frame_paths(frames_dir: Path) -> list[Path]:
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"no lossless RGB frames under {frames_dir}")
    expected = [f"{index:05d}.png" for index in range(len(paths))]
    if [path.name for path in paths] != expected:
        raise RuntimeError(f"frame filenames are not contiguous under {frames_dir}")
    return paths


def _load_bgr_frames(paths: Iterable[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read frame: {path}")
        frames.append(frame)
    return frames


def _write_context_videos(sample_dir: Path) -> tuple[Path, Path]:
    frame_paths = _frame_paths(sample_dir / "frames")
    if len(frame_paths) < CONTEXT_FRAME_OPTIONS[-1]:
        raise RuntimeError(f"{sample_dir.name}: fewer than 16 frames")
    frames = _load_bgr_frames(frame_paths[: CONTEXT_FRAME_OPTIONS[-1]])
    context_dir = sample_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    context8 = context_dir / "context8.mp4"
    context16 = context_dir / "context16.mp4"
    legacy._write_video_h264(context8, frames[: CONTEXT_FRAME_OPTIONS[0]])
    legacy._write_video_h264(context16, frames)
    return context8, context16


def _camera_payload(camera: CameraSpec, width: int, height: int) -> dict[str, object]:
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    camera_to_world = legacy._look_at(eye, target, up)
    fy = 0.5 * float(height) / math.tan(math.radians(float(camera.yfov_deg)) * 0.5)
    fx = fy
    return {
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(width) * 0.5,
            "cy": float(height) * 0.5,
            "width": int(width),
            "height": int(height),
            "yfov_deg": float(camera.yfov_deg),
        },
        "extrinsics": {
            "eye": eye.tolist(),
            "target": target.tolist(),
            "up": up.tolist(),
            "camera_to_world_pyrender": camera_to_world.tolist(),
            "world_to_camera_pyrender": np.linalg.inv(camera_to_world).tolist(),
            "coordinate_convention": "PyRender OpenGL camera pose; camera forward is local -Z.",
        },
    }


def _project_world_point(
    point: np.ndarray,
    camera: CameraSpec,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-8
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-8
    true_up = np.cross(right, forward)
    delta = np.asarray(point, dtype=np.float64) - eye
    depth = float(delta @ forward)
    if depth <= 1e-6:
        return None
    fy = 0.5 * float(height) / math.tan(math.radians(float(camera.yfov_deg)) * 0.5)
    fx = fy
    u = fx * float(delta @ right) / depth + 0.5 * width
    v = 0.5 * height - fy * float(delta @ true_up) / depth
    if not (0.0 <= u < width and 0.0 <= v < height):
        return None
    return int(round(u)), int(round(v))


def _palette_bgr(index: int) -> tuple[int, int, int]:
    colors = (
        (52, 75, 220),
        (65, 174, 74),
        (208, 148, 43),
        (194, 77, 167),
        (42, 165, 211),
        (132, 86, 225),
        (66, 190, 200),
        (95, 107, 255),
    )
    return colors[index % len(colors)]


def _write_mask_video(mask_ids: np.ndarray, output_path: Path) -> None:
    palette = np.zeros((256, 3), dtype=np.uint8)
    for object_id in range(1, 256):
        palette[object_id] = _palette_bgr(object_id - 1)
    legacy._write_video_h264(output_path, [palette[frame] for frame in mask_ids])


def _write_depth_video(depth: np.ndarray, output_path: Path) -> dict[str, float]:
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise RuntimeError("depth capture contains no valid pixels")
    low, high = np.quantile(depth[valid], (0.01, 0.99))
    high = max(float(high), float(low) + 1e-4)
    frames: list[np.ndarray] = []
    for frame, frame_valid in zip(depth, valid):
        scaled = np.clip((frame - low) / (high - low), 0.0, 1.0)
        visualization = cv2.applyColorMap(
            np.rint(scaled * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        visualization[~frame_valid] = 0
        frames.append(visualization)
    legacy._write_video_h264(output_path, frames)
    return {"visualization_low_m": float(low), "visualization_high_m": float(high)}


def _write_trajectory_video(
    *,
    frame_paths: list[Path],
    positions: np.ndarray,
    object_names: list[str],
    dynamic_indices: list[int],
    camera: CameraSpec,
    output_path: Path,
) -> None:
    frames: list[np.ndarray] = []
    trajectories: dict[int, list[tuple[int, int]]] = {index: [] for index in dynamic_indices}
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read frame: {frame_path}")
        height, width = frame.shape[:2]
        for color_index, object_index in enumerate(dynamic_indices):
            pixel = _project_world_point(positions[frame_index, object_index], camera, width, height)
            if pixel is None:
                continue
            history = trajectories[object_index]
            history.append(pixel)
            color = _palette_bgr(color_index)
            for left, right in zip(history[:-1], history[1:]):
                cv2.line(frame, left, right, color, 2, cv2.LINE_AA)
            cv2.circle(frame, pixel, 5, color, -1, cv2.LINE_AA)
        frames.append(frame)
    legacy._write_video_h264(output_path, frames)


def _write_contact_video(
    *,
    frame_paths: list[Path],
    contacts: list[dict[str, object]],
    camera: CameraSpec,
    output_path: Path,
) -> None:
    contacts_by_frame: dict[int, list[dict[str, object]]] = {}
    for record in contacts:
        contacts_by_frame.setdefault(int(record["frame"]), []).append(record)
    frames: list[np.ndarray] = []
    for frame_index, frame_path in enumerate(frame_paths):
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to read frame: {frame_path}")
        height, width = frame.shape[:2]
        for pair_index, record in enumerate(contacts_by_frame.get(frame_index, [])):
            color = _palette_bgr(pair_index)
            for contact in record["contacts"]:
                pixel = _project_world_point(np.asarray(contact["point"], dtype=np.float32), camera, width, height)
                if pixel is not None:
                    cv2.circle(frame, pixel, 5, color, 1, cv2.LINE_AA)
        frames.append(frame)
    legacy._write_video_h264(output_path, frames)


def _actor_masks(
    *,
    raw_dir: Path,
    mask_ids: np.ndarray,
    object_names: list[str],
    object_roles: list[str],
    dynamic_mask: np.ndarray,
    instance_id_by_name: dict[str, int],
) -> tuple[list[str], list[int]]:
    actor_indices = [index for index, dynamic in enumerate(dynamic_mask) if bool(dynamic)]
    actor_names = [object_names[index] for index in actor_indices]
    actor_ids = [instance_id_by_name[name] for name in actor_names]
    if not actor_names:
        raise RuntimeError("sample has no dynamic actors for RigidBench-style masks")
    temporary_array = raw_dir / ".actor_masks.npy"
    masks = np.lib.format.open_memmap(
        temporary_array,
        mode="w+",
        dtype=np.bool_,
        shape=(mask_ids.shape[0], len(actor_ids), mask_ids.shape[1], mask_ids.shape[2]),
    )
    try:
        for actor_index, object_id in enumerate(actor_ids):
            masks[:, actor_index] = mask_ids == object_id
        np.savez_compressed(
            raw_dir / "masks.npz",
            masks=masks,
            object_names=np.asarray(actor_names, dtype=np.str_),
            object_ids=np.asarray(actor_ids, dtype=np.uint8),
            object_roles=np.asarray([object_roles[index] for index in actor_indices], dtype=np.str_),
        )
    finally:
        del masks
        temporary_array.unlink(missing_ok=True)
    return actor_names, actor_ids


def _write_trajectories(
    *,
    raw_dir: Path,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    linear_velocities: np.ndarray,
    angular_velocities: np.ndarray,
    object_names: list[str],
    object_roles: list[str],
    frame_times: np.ndarray,
) -> None:
    payload: dict[str, np.ndarray] = {
        "object_names": np.asarray(object_names, dtype=np.str_),
        "object_roles": np.asarray(object_roles, dtype=np.str_),
        "frame_times_s": frame_times.astype(np.float32),
    }
    quats_wxyz = np.concatenate([quats_xyzw[..., 3:4], quats_xyzw[..., :3]], axis=-1)
    for index, name in enumerate(object_names):
        payload[f"{name}_positions"] = positions[:, index].astype(np.float32)
        payload[f"{name}_rotations"] = quats_wxyz[:, index].astype(np.float32)
        payload[f"{name}_linear_velocity"] = linear_velocities[:, index].astype(np.float32)
        payload[f"{name}_angular_velocity"] = angular_velocities[:, index].astype(np.float32)
    np.savez_compressed(raw_dir / "trajectories.npz", **payload)


def _first_true_frame(values: np.ndarray) -> int:
    hits = np.flatnonzero(values)
    return int(hits[0]) if len(hits) else -1


def _last_true_frame(values: np.ndarray) -> int:
    hits = np.flatnonzero(values)
    return int(hits[-1]) if len(hits) else -1


def _flatten_contact_records(
    contacts: list[dict[str, object]],
    object_index: dict[str, int],
) -> dict[str, np.ndarray]:
    frame_indices: list[int] = []
    object_a_indices: list[int] = []
    object_b_indices: list[int] = []
    points: list[list[float]] = []
    normals: list[list[float]] = []
    distances: list[float] = []
    forces: list[float] = []
    for record in contacts:
        for contact in record["contacts"]:
            frame_indices.append(int(record["frame"]))
            object_a_indices.append(int(object_index.get(str(record["obj_a"]), -1)))
            object_b_indices.append(int(object_index.get(str(record["obj_b"]), -1)))
            points.append([float(value) for value in contact["point"]])
            normals.append([float(value) for value in contact["normal"]])
            distances.append(float(contact["distance_m"]))
            forces.append(float(contact["normal_force_n"]))
    count = len(frame_indices)
    return {
        "contact_frame_indices": np.asarray(frame_indices, dtype=np.int32),
        "contact_object_a_indices": np.asarray(object_a_indices, dtype=np.int32),
        "contact_object_b_indices": np.asarray(object_b_indices, dtype=np.int32),
        "contact_points_m": np.asarray(points, dtype=np.float32).reshape(count, 3),
        "contact_normals_b_to_a": np.asarray(normals, dtype=np.float32).reshape(count, 3),
        "contact_distances_m": np.asarray(distances, dtype=np.float32),
        "contact_normal_forces_n": np.asarray(forces, dtype=np.float32),
    }


def _write_physics_supervision(
    *,
    sample_dir: Path,
    positions: np.ndarray,
    quats_xyzw: np.ndarray,
    linear_velocities: np.ndarray,
    angular_velocities: np.ndarray,
    frame_times: np.ndarray,
    object_names: list[str],
    object_roles: list[str],
    dynamic_mask: np.ndarray,
    mask_ids: np.ndarray,
    instance_id_by_name: dict[str, int],
    contacts: list[dict[str, object]],
) -> dict[str, object]:
    if len(frame_times) < 2:
        raise RuntimeError("physics supervision requires at least two frames")
    dt = float(np.median(np.diff(frame_times)))
    linear_acceleration = np.gradient(linear_velocities, dt, axis=0).astype(np.float32)
    angular_acceleration = np.gradient(angular_velocities, dt, axis=0).astype(np.float32)
    speed = np.linalg.norm(linear_velocities, axis=-1).astype(np.float32)
    angular_speed = np.linalg.norm(angular_velocities, axis=-1).astype(np.float32)
    displacement_from_start = np.linalg.norm(
        positions - positions[:1], axis=-1
    ).astype(np.float32)
    visible = np.stack(
        [np.any(mask_ids == instance_id_by_name[name], axis=(1, 2)) for name in object_names],
        axis=1,
    )
    contact_payload = _flatten_contact_records(contacts, {name: index for index, name in enumerate(object_names)})
    contact_count = np.zeros((len(frame_times), len(object_names)), dtype=np.int16)
    for frame, left, right in zip(
        contact_payload["contact_frame_indices"],
        contact_payload["contact_object_a_indices"],
        contact_payload["contact_object_b_indices"],
    ):
        if left >= 0:
            contact_count[frame, left] += 1
        if right >= 0:
            contact_count[frame, right] += 1
    in_contact = contact_count > 0
    moving = speed > MOTION_THRESHOLD_MPS
    first_motion = np.asarray([_first_true_frame(moving[:, index]) for index in range(len(object_names))], dtype=np.int32)
    last_motion = np.asarray([_last_true_frame(moving[:, index]) for index in range(len(object_names))], dtype=np.int32)
    first_contact = np.asarray([_first_true_frame(in_contact[:, index]) for index in range(len(object_names))], dtype=np.int32)
    quats_wxyz = np.concatenate([quats_xyzw[..., 3:4], quats_xyzw[..., :3]], axis=-1)
    np.savez_compressed(
        sample_dir / "physics_supervision.npz",
        frame_times_s=frame_times.astype(np.float32),
        positions_m=positions.astype(np.float32),
        rotations_wxyz=quats_wxyz.astype(np.float32),
        linear_velocity_mps=linear_velocities.astype(np.float32),
        angular_velocity_radps=angular_velocities.astype(np.float32),
        linear_acceleration_mps2=linear_acceleration,
        angular_acceleration_radps2=angular_acceleration,
        speed_mps=speed,
        angular_speed_radps=angular_speed,
        displacement_from_start_m=displacement_from_start,
        visible=visible.astype(np.bool_),
        in_contact=in_contact.astype(np.bool_),
        contact_count=contact_count,
        first_motion_frame=first_motion,
        last_motion_frame=last_motion,
        first_contact_frame=first_contact,
        object_names=np.asarray(object_names, dtype=np.str_),
        object_roles=np.asarray(object_roles, dtype=np.str_),
        dynamic_mask=dynamic_mask.astype(np.bool_),
        instance_ids=np.asarray([instance_id_by_name[name] for name in object_names], dtype=np.uint8),
        **contact_payload,
    )
    summaries = {}
    for index, name in enumerate(object_names):
        summaries[name] = {
            "dynamic": bool(dynamic_mask[index]),
            "first_motion_frame": int(first_motion[index]),
            "last_motion_frame": int(last_motion[index]),
            "first_contact_frame": int(first_contact[index]),
            "max_speed_mps": float(speed[:, index].max(initial=0.0)),
            "final_speed_mps": float(speed[-1, index]),
            "max_angular_speed_radps": float(angular_speed[:, index].max(initial=0.0)),
            "final_angular_speed_radps": float(angular_speed[-1, index]),
            "path_length_m": float(np.linalg.norm(np.diff(positions[:, index], axis=0), axis=1).sum()),
            "net_displacement_m": float(displacement_from_start[-1, index]),
            "visible_every_frame": bool(np.all(visible[:, index])),
        }
    supervision_json = {
        "schema_version": SCHEMA_VERSION,
        "state_coordinate_system": "PyBullet world coordinates in meters; z is up.",
        "rotation_convention": "wxyz in physics_supervision.npz and trajectories.npz; raw/states_xyzw.npz remains PyBullet xyzw.",
        "contact_normal_convention": "contact_normals_b_to_a points from obj_b to obj_a, following PyBullet.",
        "motion_threshold_mps": MOTION_THRESHOLD_MPS,
        "dt_s": dt,
        "contact_point_count": int(len(contact_payload["contact_frame_indices"])),
        "objects": summaries,
    }
    _write_json(sample_dir / "physics_supervision.json", supervision_json)
    return supervision_json


def _derive_event_frame(case: ExportCase, positions: np.ndarray, quats: np.ndarray, object_names: list[str]) -> int:
    index = {name: idx for idx, name in enumerate(object_names)}
    if case.family_key == "F11":
        threshold = float(case.blueprint.metadata["table_top_half_width_m"])
        frames = np.flatnonzero(positions[:, index["roller_0"], 0] > threshold)
    elif case.family_key in {"F12", "F12_RAMP_LENGTH"}:
        theta = math.radians(float(case.blueprint.metadata["ramp_angle_deg"]))
        ramp_edge = 0.5 * float(case.blueprint.metadata["ramp_length_m"]) * math.cos(theta)
        frames = np.flatnonzero(positions[:, index["block_0"], 0] > ramp_edge - 0.02)
    else:
        # V2V rendering already computes this rule in its simulator metadata.
        return -1
    return int(frames[0]) if len(frames) else -1


def _package_case(
    *,
    case: ExportCase,
    sample_dir: Path,
    render_manifest: dict[str, object],
    width: int,
    height: int,
) -> dict[str, object]:
    raw_dir = sample_dir / "raw"
    videos_dir = sample_dir / "videos"
    raw_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    render_video = Path(str(render_manifest["video"]))
    render_states = Path(str(render_manifest["states"]))
    render_mask_ids = Path(str(render_manifest["mask_ids"]))
    render_meta = Path(str(render_manifest["meta"]))
    _move(render_video, raw_dir / "source_video.mp4")
    _move(render_states, raw_dir / "states_xyzw.npz")
    _move(render_mask_ids, raw_dir / "instance_ids.npz")
    _hardlink_or_copy(raw_dir / "source_video.mp4", videos_dir / "rgb.mp4")
    _write_json(raw_dir / "simulator_render_metadata.json", json.loads(render_meta.read_text(encoding="utf-8")))

    stage_context8 = render_manifest.get("context_video")
    stage_context16 = render_manifest.get("context16_video")
    context_dir = sample_dir / "context"
    context_dir.mkdir(parents=True, exist_ok=True)
    if stage_context8 and stage_context16:
        _move(Path(str(stage_context8)), context_dir / "context8.mp4")
        _move(Path(str(stage_context16)), context_dir / "context16.mp4")
    else:
        _write_context_videos(sample_dir)

    state_payload = np.load(raw_dir / "states_xyzw.npz", allow_pickle=True)
    positions = np.asarray(state_payload["positions"], dtype=np.float32)
    quats = np.asarray(state_payload["quats"], dtype=np.float32)
    linear_velocities = np.asarray(state_payload["linear_velocities"], dtype=np.float32)
    angular_velocities = np.asarray(state_payload["angular_velocities"], dtype=np.float32)
    frame_times = np.asarray(state_payload["frame_times"], dtype=np.float32)
    object_names = [_decode_name(value) for value in state_payload["object_names"]]
    object_roles = [_decode_name(value) for value in state_payload["object_roles"]]
    if positions.shape[0] != len(_frame_paths(sample_dir / "frames")):
        raise RuntimeError(f"{case.case_id}: state/frame count mismatch")
    if positions.shape[:2] != quats.shape[:2] or positions.shape[:2] != linear_velocities.shape[:2]:
        raise RuntimeError(f"{case.case_id}: inconsistent state shapes")

    mask_payload = np.load(raw_dir / "instance_ids.npz", allow_pickle=True)
    mask_ids = np.asarray(mask_payload["instance_ids"], dtype=np.uint8)
    mask_names = [_decode_name(value) for value in mask_payload["object_names"]]
    mask_values = [int(value) for value in mask_payload["object_ids"]]
    instance_id_by_name = dict(zip(mask_names, mask_values))
    if mask_ids.shape[0] != len(frame_times) or mask_ids.shape[1:] != (height, width):
        raise RuntimeError(f"{case.case_id}: mask dimensions do not match rendered video")
    if set(object_names) != set(mask_names):
        raise RuntimeError(f"{case.case_id}: state/mask object names differ")

    object_by_name = {obj.name: obj for obj in case.blueprint.objects}
    if set(object_names) != set(object_by_name):
        raise RuntimeError(f"{case.case_id}: state/blueprint object names differ")
    dynamic_mask = np.asarray([bool(object_by_name[name].dynamic) for name in object_names], dtype=np.bool_)
    actor_names, actor_ids = _actor_masks(
        raw_dir=raw_dir,
        mask_ids=mask_ids,
        object_names=object_names,
        object_roles=object_roles,
        dynamic_mask=dynamic_mask,
        instance_id_by_name=instance_id_by_name,
    )
    _write_trajectories(
        raw_dir=raw_dir,
        positions=positions,
        quats_xyzw=quats,
        linear_velocities=linear_velocities,
        angular_velocities=angular_velocities,
        object_names=object_names,
        object_roles=object_roles,
        frame_times=frame_times,
    )

    contacts_path = sample_dir / "contacts.json"
    contacts = json.loads(contacts_path.read_text(encoding="utf-8"))
    if not isinstance(contacts, list):
        raise RuntimeError(f"{case.case_id}: contacts.json must contain a list")
    physics_summary = _write_physics_supervision(
        sample_dir=sample_dir,
        positions=positions,
        quats_xyzw=quats,
        linear_velocities=linear_velocities,
        angular_velocities=angular_velocities,
        frame_times=frame_times,
        object_names=object_names,
        object_roles=object_roles,
        dynamic_mask=dynamic_mask,
        mask_ids=mask_ids,
        instance_id_by_name=instance_id_by_name,
        contacts=contacts,
    )

    _write_mask_video(mask_ids, videos_dir / "masks.mp4")
    depth_payload = np.load(raw_dir / "depth.npz", allow_pickle=False)
    depth = np.asarray(depth_payload["depth"], dtype=np.float32)
    if depth.shape != (len(frame_times), height, width):
        raise RuntimeError(f"{case.case_id}: unexpected depth shape {depth.shape}")
    depth_visualization = _write_depth_video(depth, videos_dir / "depth.mp4")
    frame_paths = _frame_paths(sample_dir / "frames")
    dynamic_indices = [index for index, dynamic in enumerate(dynamic_mask) if dynamic]
    _write_trajectory_video(
        frame_paths=frame_paths,
        positions=positions,
        object_names=object_names,
        dynamic_indices=dynamic_indices,
        camera=case.blueprint.camera,
        output_path=videos_dir / "trajectory.mp4",
    )
    _write_contact_video(
        frame_paths=frame_paths,
        contacts=contacts,
        camera=case.blueprint.camera,
        output_path=videos_dir / "contacts.mp4",
    )
    raw_frames_dir = raw_dir / "frames"
    for frame_path in frame_paths:
        _hardlink_or_copy(frame_path, raw_frames_dir / frame_path.name)

    render_metadata = json.loads((raw_dir / "simulator_render_metadata.json").read_text(encoding="utf-8"))
    v2v_qa = render_metadata.get("qa", {}) if isinstance(render_metadata, dict) else {}
    raw_event_frame = v2v_qa.get("first_event_frame", -1)
    event_frame = (
        int(raw_event_frame)
        if raw_event_frame is not None
        else -1
    ) if case.v2v_case is not None else _derive_event_frame(case, positions, quats, object_names)
    event_time_s = float(frame_times[event_frame]) if 0 <= event_frame < len(frame_times) else None
    actor_details = {
        name: {
            "object_id": object_by_name[name].family_key,
            "role": object_by_name[name].role,
            "dynamic": bool(object_by_name[name].dynamic),
            "mass_kg": float(object_by_name[name].mass),
            "friction": float(object_by_name[name].friction),
            "restitution": float(object_by_name[name].restitution),
            "shape": object_by_name[name].shape,
            "size_m": dict(object_by_name[name].size),
            "initial_position_m": [float(value) for value in object_by_name[name].position],
            "initial_linear_velocity_mps": [float(value) for value in object_by_name[name].linear_velocity],
            "initial_angular_velocity_radps": [float(value) for value in object_by_name[name].angular_velocity],
            "instance_id": int(instance_id_by_name[name]),
        }
        for name in object_names
    }
    camera = _camera_payload(case.blueprint.camera, width, height)
    metadata = {
        "sample_id": case.case_id,
        "dataset": "PhysV V2V controls",
        "schema_version": SCHEMA_VERSION,
        "task_type": case.task_type,
        "split": "controlled_physics",
        "source_group": case.source_group,
        "family_key": case.family_key,
        "taxonomy": case.taxonomy,
        "taxonomy_definition": TAXONOMY_DEFINITIONS[case.taxonomy],
        "seed": case.seed,
        "scene_style": case.scene_style,
        "title": case.title,
        "scene_description_simulator_only": case.description,
        "gravity_mps2": [0.0, 0.0, -float(case.blueprint.gravity)],
        "simulation": {
            "sim_hz": 240,
            "fps": FPS,
            "frame_count": int(len(frame_times)),
            "duration_seconds": float(frame_times[-1] - frame_times[0]),
            "pre_roll_s": float(case.blueprint.pre_roll_s),
        },
        "conditioning": {
            "type": "video_context",
            "context_frame_options": list(CONTEXT_FRAME_OPTIONS),
            "context_duration_s": [round(value / FPS, 6) for value in CONTEXT_FRAME_OPTIONS],
            "target_video": "videos/rgb.mp4",
            "first_event_rule": case.event_rule,
            "first_event_frame": event_frame if event_frame >= 0 else None,
            "first_event_time_s": event_time_s,
            "event_after_context8": bool(event_frame >= CONTEXT_FRAME_OPTIONS[0]),
            "event_after_context16": bool(event_frame >= CONTEXT_FRAME_OPTIONS[-1]),
        },
        "control": {
            "variable": case.controlled_variable,
            "value": case.controlled_value,
            "value_label": case.controlled_value_label,
            "units": case.units,
        },
        "camera": camera,
        "actors": actor_details,
        "scenario_spec": _json_safe(case.blueprint.metadata),
        "analysis_prompt": case.analysis_question,
        "depth_visualization": depth_visualization,
        "initialization_qa": render_metadata.get("initialization_qa"),
    }
    caption_observations = derive_caption_observations(sample_dir, metadata)
    caption_bundle = attach_caption_metadata(metadata, caption_observations)
    manifest = {
        "sample_id": case.case_id,
        "task_type": case.task_type,
        "source_group": case.source_group,
        "family_key": case.family_key,
        "taxonomy": case.taxonomy,
        "seed": case.seed,
        "controlled_variable": case.controlled_variable,
        "controlled_value": case.controlled_value,
        "captions": {
            "specific": CAPTION_FILES["specific"],
            "abstract": CAPTION_FILES["abstract"],
            "bundle": CAPTION_FILES["bundle"],
        },
        "caption_observations": caption_observations,
        "objects": object_names,
        "dynamic_actors": actor_names,
    }
    wrapper = {
        "sample_id": case.case_id,
        "dataset": "PhysV V2V controls",
        "schema_version": SCHEMA_VERSION,
        "task": {
            "task_type": case.task_type,
            "source_group": case.source_group,
            "family_key": case.family_key,
            "taxonomy": case.taxonomy,
            "controlled_variable": case.controlled_variable,
            "controlled_value": case.controlled_value,
        },
        "video": {
            "frame_count": int(len(frame_times)),
            "width": int(width),
            "height": int(height),
            "fps": FPS,
            "duration_seconds": float(frame_times[-1] - frame_times[0]),
            "files": {
                "rgb": "videos/rgb.mp4",
                "masks": "videos/masks.mp4",
                "depth": "videos/depth.mp4",
                "trajectory": "videos/trajectory.mp4",
                "contacts": "videos/contacts.mp4",
                "context8": "context/context8.mp4",
                "context16": "context/context16.mp4",
            },
        },
        "content": {
            "caption_specific": CAPTION_FILES["specific"],
            "caption_abstract": CAPTION_FILES["abstract"],
            "captions": CAPTION_FILES["bundle"],
            "metadata": "metadata.json",
            "physics_supervision": "physics_supervision.npz",
            "physics_supervision_summary": "physics_supervision.json",
            "contacts": "contacts.json",
            "manifest": "manifest.json",
            "source_video": "raw/source_video.mp4",
            "raw_frames": "raw/frames",
            "lossless_frames": "frames",
            "masks": "raw/masks.npz",
            "instance_ids": "raw/instance_ids.npz",
            "depth": "raw/depth.npz",
            "trajectories": "raw/trajectories.npz",
            "raw_states_xyzw": "raw/states_xyzw.npz",
        },
        "mask_policy": "raw/masks.npz contains dynamic actors only; raw/instance_ids.npz contains all rendered simulator objects.",
        "status": {
            "captions": True,
            "lossless_frames": True,
            "metadata": True,
            "contacts": True,
            "masks": True,
            "depth": True,
            "trajectories": True,
            "physics_supervision": True,
        },
    }
    _write_json(sample_dir / "manifest.json", manifest)
    _write_json(sample_dir / "metadata.json", metadata)
    _write_json(sample_dir / "meta.json", wrapper)
    caption_dir = sample_dir / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    (caption_dir / "caption_specific.txt").write_text(caption_bundle["specific"] + "\n", encoding="utf-8")
    (caption_dir / "caption_abstract.txt").write_text(caption_bundle["abstract"] + "\n", encoding="utf-8")
    _write_json(
        caption_dir / "captions.json",
        {
            "schema_version": CAPTION_SCHEMA_VERSION,
            "source": "metadata.json",
            **caption_bundle,
        },
    )
    _write_json(
        sample_dir / "export_summary.json",
        {
            "sample_id": case.case_id,
            "frame_count": len(frame_times),
            "object_count": len(object_names),
            "dynamic_actor_count": len(actor_names),
            "contact_record_count": len(contacts),
            "contact_point_count": physics_summary["contact_point_count"],
            "event_first_frame": event_frame if event_frame >= 0 else None,
        },
    )
    return {
        "sample_id": case.case_id,
        "task_type": case.task_type,
        "source_group": case.source_group,
        "family_key": case.family_key,
        "taxonomy": case.taxonomy,
        "controlled_variable": case.controlled_variable,
        "controlled_value": case.controlled_value,
        "sample_dir": str(sample_dir),
        "frame_count": int(len(frame_times)),
        "dynamic_actor_count": len(actor_names),
        "contact_point_count": physics_summary["contact_point_count"],
    }


def _render_export_case(
    *,
    case: ExportCase,
    sample_dir: Path,
    width: int,
    height: int,
) -> dict[str, object]:
    staging_dir = sample_dir / ".render"
    if case.v2v_case is not None:
        return render_v2v_case(
            case.v2v_case,
            seed=case.seed,
            output_root=staging_dir,
            width=width,
            height=height,
            ground_truth_output_dir=sample_dir,
        )
    return render_blueprint_case(
        blueprint=case.blueprint,
        seed=case.seed,
        output_root=staging_dir,
        width=width,
        height=height,
        scene_style=case.scene_style,
        export_instance_masks=True,
        preserve_states=True,
        ground_truth_output_dir=sample_dir,
    )


def _validate_sample(sample_dir: Path) -> None:
    meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
    required = [
        "captions/caption_specific.txt",
        "captions/caption_abstract.txt",
        "captions/captions.json",
        "manifest.json",
        "metadata.json",
        "contacts.json",
        "physics_supervision.npz",
        "raw/source_video.mp4",
        "raw/masks.npz",
        "raw/instance_ids.npz",
        "raw/depth.npz",
        "raw/trajectories.npz",
        "videos/rgb.mp4",
        "videos/masks.mp4",
        "videos/depth.mp4",
        "videos/trajectory.mp4",
        "videos/contacts.mp4",
        "context/context8.mp4",
        "context/context16.mp4",
    ]
    missing = [path for path in required if not (sample_dir / path).exists()]
    if missing:
        raise RuntimeError(f"{sample_dir.name}: missing required files: {missing}")
    frames = _frame_paths(sample_dir / "frames")
    depth = np.load(sample_dir / "raw" / "depth.npz", allow_pickle=False)["depth"]
    masks = np.load(sample_dir / "raw" / "masks.npz", allow_pickle=False)["masks"]
    trajectories = np.load(sample_dir / "raw" / "trajectories.npz", allow_pickle=False)
    if len(frames) != int(meta["video"]["frame_count"]):
        raise RuntimeError(f"{sample_dir.name}: lossless frame count mismatch")
    if depth.shape[0] != len(frames) or masks.shape[0] != len(frames):
        raise RuntimeError(f"{sample_dir.name}: truth tensor time dimension mismatch")
    if "object_names" not in trajectories.files:
        raise RuntimeError(f"{sample_dir.name}: trajectories lack object_names")


def _write_dataset_files(output_root: Path, rows: list[dict[str, object]]) -> None:
    rows = sorted(rows, key=lambda row: str(row["sample_id"]))
    group_counts = {
        group: sum(1 for row in rows if row.get("source_group") == group)
        for group in (
            "v2v_control",
            "v2v_obstacle_ball_size",
            "v2v_pendulum_cabinet_height",
            "scene_puck_barrier",
            "scene_door_frame",
            "scene_door_frame_ball",
            "f11_table_height",
            "f12_incline",
            "f12_ramp_length",
        )
    }
    _write_json(
        output_root / "manifest.json",
        {
            "dataset": "PhysV V2V controls",
            "schema_version": SCHEMA_VERSION,
            "sample_count": len(rows),
            "included_groups": group_counts,
            "taxonomy_counts": {
                taxonomy: sum(1 for row in rows if row.get("taxonomy") == taxonomy)
                for taxonomy in TAXONOMY_DEFINITIONS
            },
            "excluded_groups": ["F11 direction variants"],
            "samples": rows,
        },
    )
    _write_json(
        output_root / "dataset_meta.json",
        {
            "dataset": "PhysV V2V controls",
            "schema_version": SCHEMA_VERSION,
            "description": "Deterministic PyBullet rigid-body continuation controls with RigidBench-style ground truth and metadata-driven captions.",
            "coordinate_system": "PyBullet world coordinates in meters; z is up.",
            "rgb": "Lossless PNG frames are direct renderer captures. MP4 files are convenience encodes.",
            "mask_policy": "masks.npz contains dynamic actors; instance_ids.npz contains all rendered simulator objects.",
            "depth": "raw/depth.npz contains PyRender Z-depth in scene meters; zero denotes background.",
            "contacts": "contacts.json records motion-relevant PyBullet contacts sampled at video frames.",
            "source_selection": "55 V2V/scene cases (including obstacle, pendulum-cabinet height, puck-barrier, wooden-crate door-frame, and ball door-frame controls), 5 F11 table-height cases, 5 F12 incline-angle cases, and 5 F12 fixed-high-support-height ramp-length cases; F11 direction variants excluded.",
            "taxonomy": TAXONOMY_DEFINITIONS,
            "captions": {
                "specific": "captions/caption_specific.txt exposes the controlled variable and value.",
                "abstract": "captions/caption_abstract.txt hides the controlled variable and value.",
                "bundle": "captions/captions.json contains both caption versions.",
            },
        },
    )
    readme = """# PhysV V2V 0819

This dataset contains 70 deterministic rigid-body video-continuation controls:
55 V2V/scene cases, including the original V2V controls, five pendulum-cabinet suspension-height controls, five ice-puck barrier-angle controls, five wooden-crate door-frame opening-width controls, and five ball door-frame opening-width controls; five F11 table-height controls; five F12 incline-angle controls; and five F12 fixed-high-support-height ramp-length controls.
F11 direction variants are intentionally excluded.
The puck-barrier and wooden-crate door-frame groups include low-resolution Cycles previews at 640x360; the ball door-frame and pendulum-cabinet groups include low-resolution Cycles previews at 896x512. The pendulum-cabinet group was rendered after PyBullet motion verification. The PyBullet source videos remain the full-resolution simulation reference.

The controls use three explicit taxonomy levels:

- `Scene`: static environment geometry changes; the moving object remains fixed.
- `Object`: the environment remains fixed; the moving object's geometry or initial state changes.
- `Relation`: the object and environment remain fixed; only their relative position, direction, or support relation changes.

Each `samples/<case_id>/` directory follows a RigidBench-inspired layout:

- `frames/`: lossless renderer RGB PNGs.
- `context/context8.mp4`, `context/context16.mp4`: V2V conditioning clips.
- `raw/depth.npz`: per-frame PyRender depth in meters.
- `raw/masks.npz`: boolean masks for dynamic actors.
- `raw/instance_ids.npz`: instance IDs for all rendered simulator objects.
- `raw/trajectories.npz`: per-object world trajectories with wxyz rotations.
- `contacts.json`: sampled PyBullet contact points, normals, distance, and normal force.
- `physics_supervision.npz`: aligned states, velocities, accelerations, visibility, contact arrays, and derived motion indices.
- `captions/caption_specific.txt`: caption with the controlled variable and value exposed.
- `captions/caption_abstract.txt`: caption with the controlled variable and value hidden.
- `captions/captions.json`: structured copy of both caption versions.
- `metadata.json`, `meta.json`, `manifest.json`: sample metadata and caption references.

The raw state file retains PyBullet xyzw quaternions; exported trajectory and
physics-supervision files use wxyz and declare this convention in metadata.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def export_dataset(
    *,
    output_root: Path,
    selected_case_ids: set[str] | None = None,
    overwrite: bool = False,
    overwrite_selected: bool = False,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    v2v_seed_base: int = DEFAULT_V2V_SEED_BASE,
    difficulty_seed_base: int = DEFAULT_DIFFICULTY_SEED_BASE,
) -> list[dict[str, object]]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_cases = build_export_cases(
        v2v_seed_base=v2v_seed_base,
        difficulty_seed_base=difficulty_seed_base,
    )
    invariant_report = audit_group_invariants(all_cases)
    _write_json(output_root / "reports" / "group_invariant_audit.json", invariant_report)
    cases = all_cases
    if selected_case_ids is not None:
        known = {case.case_id for case in cases}
        unknown = sorted(selected_case_ids - known)
        if unknown:
            raise ValueError(f"unknown case ids: {unknown}")
        cases = [case for case in cases if case.case_id in selected_case_ids]
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for case in cases:
        sample_dir = output_root / "samples" / case.case_id
        try:
            if _remove_or_skip_sample(
                sample_dir,
                overwrite=bool(overwrite_selected and selected_case_ids is not None),
            ):
                _validate_sample(sample_dir)
                existing_meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
                existing_manifest = json.loads(
                    (sample_dir / "manifest.json").read_text(encoding="utf-8")
                )
                rows.append(
                    {
                        "sample_id": case.case_id,
                        "status": "existing_validated",
                        "task_type": case.task_type,
                        "source_group": case.source_group,
                        "family_key": case.family_key,
                        "taxonomy": case.taxonomy,
                        "controlled_variable": case.controlled_variable,
                        "controlled_value": case.controlled_value,
                        "sample_dir": str(sample_dir),
                        "frame_count": int(existing_meta["video"]["frame_count"]),
                        "dynamic_actor_count": len(existing_manifest.get("dynamic_actors", [])),
                        "contact_point_count": int(
                            json.loads((sample_dir / "physics_supervision.json").read_text(encoding="utf-8"))[
                                "contact_point_count"
                            ]
                        ),
                    }
                )
                print(f"validated existing {case.case_id}", flush=True)
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)
            render_manifest = _render_export_case(
                case=case,
                sample_dir=sample_dir,
                width=width,
                height=height,
            )
            row = _package_case(
                case=case,
                sample_dir=sample_dir,
                render_manifest=render_manifest,
                width=width,
                height=height,
            )
            shutil.rmtree(sample_dir / ".render", ignore_errors=True)
            _validate_sample(sample_dir)
            rows.append(row)
            print(f"exported {case.case_id}", flush=True)
        except Exception as exc:  # pragma: no cover - batch guard
            failures.append({"sample_id": case.case_id, "error": repr(exc)})
            print(f"failed {case.case_id}: {exc!r}", flush=True)
    # Selected regeneration must preserve the other valid samples in the
    # root manifest, while dropping rows for obsolete case IDs.
    if selected_case_ids is not None:
        known_ids = {case.case_id for case in all_cases}
        manifest_path = output_root / "manifest.json"
        existing_rows: list[dict[str, object]] = []
        if manifest_path.exists():
            existing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(existing_payload, dict) and isinstance(existing_payload.get("samples"), list):
                existing_rows = [
                    row
                    for row in existing_payload["samples"]
                    if isinstance(row, dict) and str(row.get("sample_id")) in known_ids
                ]
        by_id = {str(row["sample_id"]): row for row in existing_rows}
        by_id.update({str(row["sample_id"]): row for row in rows})
        rows = list(by_id.values())
    _write_json(output_root / "reports" / "failure_report.json", failures)
    _write_dataset_files(output_root, rows)
    _write_json(
        output_root / "reports" / "summary.json",
        {
            "requested": len(cases),
            "exported_or_validated": len(rows),
            "failed": len(failures),
            "output_root": str(output_root),
        },
    )
    if failures:
        raise RuntimeError(f"dataset export failed for {len(failures)} case(s)")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--overwrite", action="store_true", help="Remove the complete output root first.")
    parser.add_argument("--overwrite-selected", action="store_true", help="Regenerate selected sample directories while preserving the rest of the dataset.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--v2v-seed-base", type=int, default=DEFAULT_V2V_SEED_BASE)
    parser.add_argument("--difficulty-seed-base", type=int, default=DEFAULT_DIFFICULTY_SEED_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("--width and --height must be positive")
    rows = export_dataset(
        output_root=args.output_root,
        selected_case_ids=set(args.case_ids) if args.case_ids else None,
        overwrite=args.overwrite,
        overwrite_selected=args.overwrite_selected,
        width=args.width,
        height=args.height,
        v2v_seed_base=args.v2v_seed_base,
        difficulty_seed_base=args.difficulty_seed_base,
    )
    print(json.dumps({"output_root": str(args.output_root), "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
