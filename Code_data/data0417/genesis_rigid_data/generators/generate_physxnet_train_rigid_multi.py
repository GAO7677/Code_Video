#!/usr/bin/env python3
# 用途：生成当前 train/rigid 多物体训练集。
"""Generate rigid-only multi-object training samples from PhysXNet assets.

该脚本用于从 /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet 生成多物体 rigid 训练样本；输入为 PhysXNet 资产、对象采样配置和输出参数，输出为 /data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_train_rigid_multi 下的样本目录、metadata 和缓存资产。

This is intentionally a lightweight train-data generator.  It keeps the same
sample file schema used by the current try1/try3 exports, but organizes samples
as train/rigid/<scene_composition>/<object_count_bucket>/... .
"""
from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from generators import try1_physxnet_benchmark as try1
from core.utils_io import write_json

SCENE_COMPOSITIONS = [
    "uniform_dynamic",
    "interaction_pair_plus_dynamic",
    "dual_interaction_groups",
    "omni_showcase",
]
COUNT_BUCKETS_BY_COMPOSITION = {
    "uniform_dynamic": ["count_01", "count_02", "count_03_04", "count_05_06"],
    "interaction_pair_plus_dynamic": ["count_02", "count_03_04", "count_05_06"],
    "dual_interaction_groups": ["count_04", "count_05_06"],
    "omni_showcase": ["count_03_04", "count_05_06"],
}
TRAIN_TARGET_COUNTS = {
    ("uniform_dynamic", "count_01"): 360,
    ("uniform_dynamic", "count_02"): 480,
    ("uniform_dynamic", "count_03_04"): 240,
    ("uniform_dynamic", "count_05_06"): 120,
    ("interaction_pair_plus_dynamic", "count_02"): 420,
    ("interaction_pair_plus_dynamic", "count_03_04"): 420,
    ("interaction_pair_plus_dynamic", "count_05_06"): 210,
    ("dual_interaction_groups", "count_04"): 270,
    ("dual_interaction_groups", "count_05_06"): 180,
    ("omni_showcase", "count_03_04"): 120,
    ("omni_showcase", "count_05_06"): 180,
}
ENVIRONMENT_SPECIAL_IDS = {"ground": -1}
PALETTE = [
    (0.82, 0.32, 0.24, 1.0),
    (0.22, 0.55, 0.86, 1.0),
    (0.92, 0.72, 0.20, 1.0),
    (0.30, 0.74, 0.38, 1.0),
    (0.62, 0.42, 0.88, 1.0),
    (0.90, 0.46, 0.68, 1.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhysXNet rigid-only train generator with multi-object scene layouts")
    parser.add_argument("--physx_root", type=Path, default=Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet"))
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--output_root", type=Path, default=Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_train_rigid_multi"))
    parser.add_argument("--cache_root", type=Path, default=None, help="Prepared asset cache. Defaults to <output_root>/_asset_cache")
    parser.add_argument("--object_ids", nargs="*", default=None, help="Optional fixed PhysXNet object id pool")
    parser.add_argument("--num_samples", type=int, default=0, help="If >0, sample this many scenes according to the train proportions")
    parser.add_argument("--one_case_each", action="store_true", help="Generate one case for every scene_composition/count_bucket combination")
    parser.add_argument("--scene_composition", choices=SCENE_COMPOSITIONS, default=None)
    parser.add_argument("--object_count_bucket", choices=sorted({b for v in COUNT_BUCKETS_BY_COMPOSITION.values() for b in v}), default=None)
    parser.add_argument("--samples_per_combo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260419)
    parser.add_argument("--steps", type=int, default=48, help="Exported frames before any duplicated initial still frame")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--substeps", type=int, default=40)
    parser.add_argument("--resolution", type=int, nargs=2, default=[960, 720])
    parser.add_argument("--default_friction", type=float, default=0.55)
    parser.add_argument("--physxnet_volume_threshold_m3", type=float, default=0.20, help="Objects with assembled bbox volume >= threshold stay static in-scene")
    parser.add_argument("--max_scene_sampling_attempts", type=int, default=24, help="Retry count for random object sampling when a scene composition needs more dynamic-capable objects")
    parser.add_argument("--object_scale_mult", type=float, default=1.0)
    parser.add_argument("--voxel_pitch", type=float, default=0.025)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Alias for --one_case_each with short 12-frame output")
    return parser.parse_args()

def sample_object_pool(args: argparse.Namespace, needed: int) -> List[str]:
    if args.object_ids:
        pool = [str(x) for x in args.object_ids]
    else:
        finaljson_dir = args.physx_root / args.version / "finaljson"
        pool = sorted(path.stem for path in finaljson_dir.glob("*.json"))
    if not pool:
        raise RuntimeError("No PhysXNet object ids found")
    rng = random.Random(int(args.seed))
    rng.shuffle(pool)
    return pool


def bucket_object_count(bucket: str, rng: random.Random) -> int:
    if bucket == "count_01":
        return 1
    if bucket == "count_02":
        return 2
    if bucket == "count_03_04":
        return rng.choice([3, 4])
    if bucket == "count_04":
        return 4
    if bucket == "count_05_06":
        return rng.choice([5, 6])
    raise ValueError(f"Unsupported bucket: {bucket}")


def build_plan(args: argparse.Namespace) -> List[Tuple[str, str, int]]:
    if args.smoke:
        args.one_case_each = True
        args.steps = min(int(args.steps), 12)
        args.samples_per_combo = max(1, int(args.samples_per_combo))
    combos: List[Tuple[str, str]] = []
    if args.scene_composition and args.object_count_bucket:
        combos = [(args.scene_composition, args.object_count_bucket)]
    elif args.scene_composition:
        combos = [(args.scene_composition, b) for b in COUNT_BUCKETS_BY_COMPOSITION[args.scene_composition]]
    elif args.object_count_bucket:
        combos = [(c, args.object_count_bucket) for c in SCENE_COMPOSITIONS if args.object_count_bucket in COUNT_BUCKETS_BY_COMPOSITION[c]]
    elif args.one_case_each or args.num_samples <= 0:
        combos = [(c, b) for c in SCENE_COMPOSITIONS for b in COUNT_BUCKETS_BY_COMPOSITION[c]]
    else:
        weighted = []
        for combo, count in TRAIN_TARGET_COUNTS.items():
            weighted.extend([combo] * int(count))
        rng = random.Random(int(args.seed) + 99)
        combos = [rng.choice(weighted) for _ in range(int(args.num_samples))]
    return [(c, b, idx) for idx, (c, b) in enumerate(combos * max(1, int(args.samples_per_combo)))]


def make_runner_args(args: argparse.Namespace, object_id: str) -> argparse.Namespace:
    ns = try1.build_argparser().parse_args([
        "--physx_root", str(args.physx_root),
        "--version", str(args.version),
        "--object_id", str(object_id),
        "--output_root", str(args.cache_root),
        "--simulator_mode", "rigid",
        "--steps", str(int(args.steps)),
        "--dt", str(float(args.dt)),
        "--substeps", str(int(args.substeps)),
        "--fps", str(int(args.fps)),
        "--disable_striker",
        "--prefer_existing_runtime_meshes",
    ])
    ns.object_scale_mult = float(args.object_scale_mult)
    ns.voxel_pitch = float(args.voxel_pitch)
    return ns


def prepare_object(args: argparse.Namespace, object_id: str) -> try1.PreparedObject:
    runner_args = make_runner_args(args, object_id)
    return try1.prepare_physxnet_object(
        physx_root=Path(runner_args.physx_root),
        version=runner_args.version,
        object_id=str(object_id),
        output_root=Path(runner_args.output_root),
        voxel_pitch=float(runner_args.voxel_pitch),
        json_override=Path(runner_args.json_override) if runner_args.json_override else None,
        object_scale_mult=float(runner_args.object_scale_mult),
        solver_family_override=runner_args.solver_family_override,
        all_parts_youngs_threshold_gpa=runner_args.all_parts_youngs_threshold_gpa,
        rigid_visual_double_sided_shell=True,
        simulator_mode="rigid",
    )


def _motion_group(motion_type: str) -> str:
    if motion_type == "static_rest":
        return "static"
    if motion_type in {"linear_slide_left", "linear_slide_right", "front_slide_in", "strike_static_left", "strike_static_right"}:
        return "linear_slide"
    if motion_type in {"diagonal_corner_left", "diagonal_corner_right", "side_throw_left", "side_throw_right"}:
        return "diagonal_throw"
    if motion_type in {"high_drop", "low_drop"}:
        return "gravity_drop"
    if motion_type in {"roll_left", "roll_right"}:
        return "rolling"
    return "dynamic"


def _object_bbox_volume_m3(metadata: Dict[str, Any]) -> float:
    bbox_min = np.asarray(metadata.get("object_bbox_min", [0.0, 0.0, 0.0]), dtype=np.float64)
    bbox_max = np.asarray(metadata.get("object_bbox_max", [0.0, 0.0, 0.0]), dtype=np.float64)
    extent = np.maximum(bbox_max - bbox_min, 1e-6)
    return float(np.prod(extent))


def _is_large_static(metadata: Dict[str, Any], threshold_m3: float) -> bool:
    if float(threshold_m3) <= 0.0:
        return False
    return _object_bbox_volume_m3(metadata) >= float(threshold_m3)


def _required_dynamic_objects(scene_composition: str) -> int:
    if scene_composition == "uniform_dynamic":
        return 0
    if scene_composition == "interaction_pair_plus_dynamic":
        return 1
    if scene_composition == "dual_interaction_groups":
        return 2
    if scene_composition == "omni_showcase":
        return 2
    return 0


def _sample_object_ids(pool: Sequence[str], count: int, rng: random.Random) -> List[str]:
    if count <= 0:
        return []
    if len(pool) < count:
        raise ValueError(f"Object pool too small: need {count}, got {len(pool)}")
    return [str(x) for x in rng.sample(list(pool), count)]


def _velocity_for_motion(motion_type: str, rng: random.Random) -> Tuple[np.ndarray, np.ndarray, float]:
    speed = rng.uniform(0.8, 1.8)
    if motion_type == "static_rest":
        return np.zeros(3), np.zeros(3), 0.0
    if motion_type in {"linear_slide_left", "strike_static_left"}:
        return np.array([speed, 0.0, 0.0]), np.array([0.0, 0.0, rng.uniform(-0.6, 0.6)]), 0.0
    if motion_type in {"linear_slide_right", "strike_static_right"}:
        return np.array([-speed, 0.0, 0.0]), np.array([0.0, 0.0, rng.uniform(-0.6, 0.6)]), 0.0
    if motion_type == "front_slide_in":
        return np.array([0.0, -speed, 0.0]), np.array([0.0, 0.0, rng.uniform(-0.8, 0.8)]), 0.0
    if motion_type == "diagonal_corner_left":
        return np.array([speed, -0.7 * speed, 0.0]), np.array([0.0, 0.0, 1.0]), 0.0
    if motion_type == "diagonal_corner_right":
        return np.array([-speed, -0.7 * speed, 0.0]), np.array([0.0, 0.0, -1.0]), 0.0
    if motion_type == "side_throw_left":
        return np.array([0.9 * speed, 0.55 * speed, 0.15]), np.array([0.0, 1.5, 0.0]), 0.04
    if motion_type == "side_throw_right":
        return np.array([-0.9 * speed, 0.55 * speed, 0.15]), np.array([0.0, -1.5, 0.0]), 0.04
    if motion_type == "high_drop":
        return np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0.0]), np.zeros(3), rng.uniform(0.35, 0.65)
    if motion_type == "low_drop":
        return np.zeros(3), np.zeros(3), rng.uniform(0.16, 0.28)
    if motion_type == "roll_left":
        return np.array([speed, 0.0, 0.0]), np.array([0.0, speed / 0.09, 0.0]), 0.0
    if motion_type == "roll_right":
        return np.array([-speed, 0.0, 0.0]), np.array([0.0, -speed / 0.09, 0.0]), 0.0
    return np.array([rng.uniform(-speed, speed), rng.uniform(-speed, speed), 0.0]), np.zeros(3), 0.0


def object_layout(
    scene_composition: str,
    count_bucket: str,
    prepared: Sequence[try1.PreparedObject],
    metadata_list: Sequence[Dict[str, Any]],
    threshold_m3: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], str]:
    rng = random.Random(seed)
    n = len(prepared)
    lane_y = np.linspace(-0.65, 0.65, max(n, 2))
    ordinary = ["linear_slide_left", "linear_slide_right", "front_slide_in", "high_drop", "roll_left", "roll_right"]
    initiator = ["strike_static_left", "strike_static_right", "front_slide_in", "diagonal_corner_left", "diagonal_corner_right", "side_throw_left", "side_throw_right"]
    objects: List[Dict[str, Any]] = []
    volume_by_idx = {
        idx: _object_bbox_volume_m3(metadata)
        for idx, metadata in enumerate(metadata_list)
    }
    large_static_idx = {
        idx
        for idx, metadata in enumerate(metadata_list)
        if _is_large_static(metadata, threshold_m3=threshold_m3)
    }
    dynamic_candidates = [idx for idx in range(n) if idx not in large_static_idx]
    if len(dynamic_candidates) < _required_dynamic_objects(scene_composition):
        raise ValueError(
            f"Not enough dynamic-capable objects for {scene_composition}: "
            f"need {_required_dynamic_objects(scene_composition)}, got {len(dynamic_candidates)} "
            f"under volume threshold {threshold_m3:.4f} m^3"
        )

    def add(idx: int, role: str, motion_type: str, pos: Sequence[float], motion_group: Optional[str] = None) -> None:
        sampled_motion_type = str(motion_type)
        large_static_override = bool(idx in large_static_idx)
        if large_static_override:
            role = "target" if role == "initiator" else role
            motion_type = "static_rest"
        lin, ang, z_extra = _velocity_for_motion(motion_type, rng)
        if role in {"target", "bystander"} and motion_type == "static_rest":
            lin[:] = 0.0
            ang[:] = 0.0
        objects.append(
            {
                "prepared_index": idx,
                "role": role,
                "motion_type": motion_type,
                "sampled_motion_type": sampled_motion_type,
                "motion_group": motion_group or _motion_group(motion_type),
                "position_xy": [float(pos[0]), float(pos[1])],
                "extra_z": float(z_extra),
                "linear_velocity": lin.astype(float).tolist(),
                "angular_velocity": ang.astype(float).tolist(),
                "yaw_deg": float(rng.uniform(-180.0, 180.0)),
                "bbox_volume_est_m3": float(volume_by_idx[idx]),
                "large_static_by_volume_threshold": large_static_override,
            }
        )

    if scene_composition == "uniform_dynamic":
        for idx in range(n):
            motion = rng.choice(ordinary) if idx not in large_static_idx else "static_rest"
            role = "initiator" if motion != "static_rest" else "target"
            add(idx, role, motion, [-0.45 + 0.3 * idx, lane_y[idx % len(lane_y)]])
        return objects, "uniform_dynamic"

    if scene_composition == "interaction_pair_plus_dynamic":
        target_idx = max(range(n), key=lambda idx: volume_by_idx[idx])
        initiator_candidates = [idx for idx in dynamic_candidates if idx != target_idx] or list(dynamic_candidates)
        initiator_idx = initiator_candidates[0]
        add(target_idx, "target", "static_rest", [0.0, 0.0])
        add(initiator_idx, "initiator", rng.choice(initiator), [0.9, 0.0])
        for idx in range(n):
            if idx in {target_idx, initiator_idx}:
                continue
            motion = rng.choice(ordinary) if idx not in large_static_idx else "static_rest"
            add(idx, "bystander", motion, [-0.55 + 0.28 * idx, lane_y[idx % len(lane_y)]])
        return objects, "pair_interaction_plus_dynamic"

    if scene_composition == "dual_interaction_groups":
        pairs = [(-0.45, -0.35), (0.45, 0.35)]
        initiator_indices = sorted(dynamic_candidates, key=lambda idx: volume_by_idx[idx])[:2]
        target_candidates = [idx for idx in range(n) if idx not in initiator_indices]
        target_indices = sorted(target_candidates, key=lambda idx: volume_by_idx[idx], reverse=True)[:2]
        if len(target_indices) < 2:
            raise ValueError("dual_interaction_groups requires at least 4 objects")
        for pair_idx, (x, y) in enumerate(pairs):
            target_idx = target_indices[pair_idx]
            init_idx = initiator_indices[pair_idx]
            add(target_idx, "target", "static_rest", [x, y])
            add(init_idx, "initiator", rng.choice(initiator), [x + 0.75, y])
            objects[-2]["motion_group"] = f"interaction_group_{pair_idx}"
            objects[-1]["motion_group"] = f"interaction_group_{pair_idx}"
        for idx in range(n):
            if idx in set(target_indices + initiator_indices):
                continue
            motion = rng.choice(ordinary) if idx not in large_static_idx else "static_rest"
            add(idx, "bystander", motion, [rng.uniform(-0.6, 0.6), rng.uniform(-0.7, 0.7)])
        return objects, "dual_interaction_groups"

    if scene_composition == "omni_showcase":
        required = ["high_drop", "linear_slide_left", "roll_right"]
        for idx in range(n):
            motion = required[idx] if idx < len(required) else rng.choice(ordinary + initiator)
            if idx in large_static_idx:
                motion = "static_rest"
            role = "initiator" if motion != "static_rest" else "target"
            angle = 2.0 * math.pi * idx / max(n, 1)
            add(idx, role, motion, [0.65 * math.cos(angle), 0.55 * math.sin(angle)])
        unique_groups = {obj["motion_group"] for obj in objects}
        if len(unique_groups) < min(3, n):
            raise ValueError(
                f"omni_showcase requires at least 3 motion groups, got {sorted(unique_groups)}"
            )
        return objects, "omni_showcase"

    raise ValueError(scene_composition)


def _add_physxnet_entity(gs: Any, scene: Any, rec: Dict[str, Any], obj_idx: int, friction: float) -> Any:
    prep: try1.PreparedObject = rec["prepared"]
    metadata = rec["metadata"]
    rigid_material_cfg = try1._default_entity_rigid_material(metadata, default_friction=friction)
    ent = scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(Path(prep.output_dir) / "rigid" / f"{prep.object_id}.urdf"),
            scale=1.0,
            pos=tuple(rec["pos"].tolist()),
            euler=(0.0, 0.0, float(rec["yaw_deg"])),
            visualization=True,
            collision=True,
            fixed=False,
            merge_fixed_links=False,
            prioritize_urdf_material=True,
            file_meshes_are_zup=True,
        ),
        material=try1._make_genesis_rigid_material(
            gs,
            rho=float(rigid_material_cfg["rho"]),
            friction=float(rigid_material_cfg["friction"]),
            restitution=float(rigid_material_cfg["restitution"]),
        ),
    )
    return ent


def _object_snapshot(entity: Any, gravity_vec: np.ndarray) -> Dict[str, Any]:
    snap = try1.rigid_entity_kinematic_snapshot(entity, gravity=gravity_vec)
    lin_k, rot_k, pot = try1.rigid_entity_energy_components(entity, gravity=gravity_vec)
    try:
        quat = np.asarray(try1._to_numpy(entity.get_quat()), dtype=np.float64).reshape(4)
    except Exception:
        quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    try:
        mass = float(entity.get_mass())
    except Exception:
        mass = 0.0
    return {
        "com_pos": snap.com_pos.astype(np.float32),
        "orientation_quat": quat.astype(np.float32),
        "linear_vel": snap.linear_vel.astype(np.float32),
        "angular_vel": snap.angular_vel.astype(np.float32),
        "kinetic": np.float32(float(lin_k) + float(rot_k)),
        "kinetic_trans": np.float32(lin_k),
        "kinetic_rot": np.float32(rot_k),
        "potential": np.float32(pot),
        "mass": np.float32(mass),
        "aabb": try1._entity_aabb_numpy(entity),
    }


def _scene_layout_path(output_root: Path, scene_composition: str, bucket: str, sample_name: str) -> Path:
    return output_root / "train" / "rigid" / scene_composition / bucket / sample_name


def generate_sample(
    *,
    args: argparse.Namespace,
    sample_index: int,
    scene_composition: str,
    bucket: str,
    object_ids: List[str],
) -> Path:
    import genesis as gs

    seed = int(args.seed) + sample_index * 1009 + sum(int("".join(ch for ch in oid if ch.isdigit()) or 0) for oid in object_ids)
    prepared = [prepare_object(args, oid) for oid in object_ids]
    prepared_metadata = [
        json.loads((Path(prep.output_dir) / "meta" / "metadata.json").read_text(encoding="utf-8"))
        for prep in prepared
    ]
    obj_cfgs, scene_type = object_layout(
        scene_composition,
        bucket,
        prepared,
        prepared_metadata,
        float(args.physxnet_volume_threshold_m3),
        seed,
    )

    try:
        gs.init()
    except Exception as exc:
        if "already initialized" not in str(exc).lower():
            raise
        try:
            gs.destroy()
        except Exception:
            pass
        gs.init()

    gravity_z = -9.81
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=float(args.dt), substeps=int(args.substeps), gravity=(0.0, 0.0, gravity_z), floor_height=0.0),
        rigid_options=gs.options.RigidOptions(dt=float(args.dt), gravity=(0.0, 0.0, gravity_z), max_collision_pairs=512),
        viewer_options=gs.options.ViewerOptions(camera_fov=35, camera_pos=(3.0, -2.4, 2.2), camera_lookat=(0.0, 0.0, 0.35)),
        vis_options=gs.options.VisOptions(visualize_mpm_boundary=False, visualize_sph_boundary=False),
        show_viewer=False,
    )
    scene.add_entity(morph=gs.morphs.Plane(), material=gs.materials.Rigid(rho=1200.0, friction=float(args.default_friction)))

    records = []
    entities = []
    for idx, cfg in enumerate(obj_cfgs):
        prep = prepared[int(cfg["prepared_index"])]
        metadata = prepared_metadata[int(cfg["prepared_index"])]
        z = float(metadata.get("grounding_offset_z", 0.0)) + 0.004 + float(cfg.get("extra_z", 0.0))
        rec = {
            "prepared": prep,
            "metadata": metadata,
            "pos": np.asarray([float(cfg["position_xy"][0]), float(cfg["position_xy"][1]), z], dtype=np.float64),
            "yaw_deg": float(cfg["yaw_deg"]),
            "linear_velocity": np.asarray(cfg["linear_velocity"], dtype=np.float64),
            "angular_velocity": np.asarray(cfg["angular_velocity"], dtype=np.float64),
            "role": str(cfg["role"]),
            "motion_type": str(cfg["motion_type"]),
            "sampled_motion_type": str(cfg.get("sampled_motion_type", cfg["motion_type"])),
            "motion_group": str(cfg["motion_group"]),
            "extra_z": float(cfg.get("extra_z", 0.0)),
            "bbox_volume_est_m3": float(cfg.get("bbox_volume_est_m3", _object_bbox_volume_m3(metadata))),
            "large_static_by_volume_threshold": bool(cfg.get("large_static_by_volume_threshold", False)),
        }
        records.append(rec)
        entities.append(_add_physxnet_entity(gs, scene, rec, idx, friction=float(args.default_friction)))

    lookat = np.mean(np.stack([rec["pos"] for rec in records], axis=0), axis=0)
    lookat[2] = max(0.25, float(lookat[2]))
    spread = max(1.0, float(np.max(np.linalg.norm(np.stack([rec["pos"][:2] for rec in records], axis=0) - lookat[:2], axis=1))) + 0.8)
    cam_pos = lookat + np.asarray([2.2 * spread, -2.4 * spread, 1.5 * spread], dtype=np.float64)
    cam_fov = 35
    cam = scene.add_camera(res=tuple(int(x) for x in args.resolution), pos=tuple(cam_pos.tolist()), lookat=tuple(lookat.tolist()), fov=cam_fov, GUI=False)
    scene.build()

    for ent, rec in zip(entities, records):
        aabb = try1._entity_aabb_numpy(ent)
        if aabb is not None:
            dz = 0.004 - float(aabb[0, 2])
            if abs(dz) > 1e-6 and float(rec["extra_z"]) <= 1e-8:
                rec["pos"][2] += dz
                ent.set_pos(tuple(rec["pos"].tolist()))
        lin = np.asarray(rec["linear_velocity"], dtype=np.float64)
        ang = np.asarray(rec["angular_velocity"], dtype=np.float64)
        if np.linalg.norm(lin) > 1e-8 or np.linalg.norm(ang) > 1e-8:
            try1._apply_rigid_entry_velocity(ent, linear=lin, angular=ang)

    camera_cfg = {
        "pos": cam_pos.astype(float).tolist(),
        "lookat": lookat.astype(float).tolist(),
        "fov": float(cam_fov),
        "res": [int(args.resolution[0]), int(args.resolution[1])],
    }
    cam_intrinsics = try1.camera_intrinsics_dict(cam, fallback_res=tuple(int(x) for x in args.resolution), fallback_fov_deg=float(cam_fov))
    save_every = max(1, int(round((1.0 / float(args.dt)) / float(args.fps))))
    total_steps = max(1, int(args.steps)) * save_every
    gravity_vec = np.asarray([0.0, 0.0, gravity_z], dtype=np.float64)

    rgb_frames: List[np.ndarray] = []
    depth_metric_frames: List[np.ndarray] = []
    depth_norm_frames: List[np.ndarray] = []
    seg_frames: List[np.ndarray] = []
    com_frames: List[np.ndarray] = []
    quat_frames: List[np.ndarray] = []
    lin_frames: List[np.ndarray] = []
    ang_frames: List[np.ndarray] = []
    kinetic_frames: List[np.float32] = []
    potential_frames: List[np.float32] = []
    total_energy_frames: List[np.float32] = []
    kinetic_trans_frames: List[np.float32] = []
    kinetic_rot_frames: List[np.float32] = []
    potential_gravity_frames: List[np.float32] = []
    trajectory_frames: List[np.ndarray] = []
    aabb_frames: List[List[Optional[np.ndarray]]] = []

    object_ids_arr = np.arange(len(records), dtype=np.int32)
    seg_ids_arr = object_ids_arr + 1

    def record_frame() -> None:
        rendered = cam.render(rgb=True, depth=True, segmentation=True, normal=False)
        rgb_raw, depth_raw, seg_raw = rendered[0], rendered[1], rendered[2]
        rgb = try1.rgb_to_uint8(rgb_raw)
        depth_metric = try1.metric_depth_map(depth_raw)
        depth_norm = try1.normalize_depth_map(depth_metric, near=float(cam_intrinsics["near"]), far=float(cam_intrinsics["far"]))
        seg_mapping = try1.build_segmentation_mapping(scene, entities, object_ids_arr.tolist())
        seg = try1.remap_segmentation(seg_raw, seg_mapping)
        snaps = [_object_snapshot(ent, gravity_vec) for ent in entities]
        com = np.stack([s["com_pos"] for s in snaps], axis=0).astype(np.float32)
        quat = np.stack([s["orientation_quat"] for s in snaps], axis=0).astype(np.float32)
        lin = np.stack([s["linear_vel"] for s in snaps], axis=0).astype(np.float32)
        ang = np.stack([s["angular_vel"] for s in snaps], axis=0).astype(np.float32)
        masses = np.asarray([s["mass"] for s in snaps], dtype=np.float32)
        k = np.float32(sum(float(s["kinetic"]) for s in snaps))
        kt = np.float32(sum(float(s["kinetic_trans"]) for s in snaps))
        kr = np.float32(sum(float(s["kinetic_rot"]) for s in snaps))
        p = np.float32(sum(float(s["potential"]) for s in snaps))
        rgb_frames.append(rgb)
        depth_metric_frames.append(depth_metric.astype(np.float32))
        depth_norm_frames.append(depth_norm.astype(np.float32))
        seg_frames.append(seg.astype(np.int32))
        com_frames.append(com)
        quat_frames.append(quat)
        lin_frames.append(lin)
        ang_frames.append(ang)
        kinetic_frames.append(k)
        potential_frames.append(p)
        total_energy_frames.append(np.float32(float(k) + float(p)))
        kinetic_trans_frames.append(kt)
        kinetic_rot_frames.append(kr)
        potential_gravity_frames.append(p)
        trajectory_frames.append(np.concatenate([com, lin * masses[:, None]], axis=1).astype(np.float32))
        aabb_frames.append([s["aabb"] for s in snaps])

    record_frame()
    for step_idx in range(total_steps):
        scene.step()
        if step_idx % save_every == 0:
            record_frame()

    rgb_arr = rgb_frames
    depth_metric_arr = np.stack(depth_metric_frames, axis=0).astype(np.float32)
    depth_norm_arr = np.stack(depth_norm_frames, axis=0).astype(np.float32)
    seg_arr = np.stack(seg_frames, axis=0).astype(np.int32)
    com_arr = np.stack(com_frames, axis=0).astype(np.float32)
    quat_arr = np.stack(quat_frames, axis=0).astype(np.float32)
    lin_arr = np.stack(lin_frames, axis=0).astype(np.float32)
    ang_arr = np.stack(ang_frames, axis=0).astype(np.float32)
    trajectory_arr = np.stack(trajectory_frames, axis=0).astype(np.float32)

    contact_graph_frames = []
    env_contact_series = []
    env_contact_impulse_series = []
    environment_contact_events: List[Dict[str, Any]] = []
    previous_env_contact = np.zeros((len(records), 1), dtype=np.uint8)
    for frame_idx, frame_aabbs in enumerate(aabb_frames):
        graph, env_contacts = try1._contact_graph_with_environment(frame_aabbs, object_ids=object_ids_arr.tolist(), ground_height=0.0)
        contact_graph_frames.append(graph)
        env_contact = np.zeros((len(records), 1), dtype=np.uint8)
        env_impulse = np.zeros((len(records), 1), dtype=np.float32)
        for env in env_contacts:
            obj_idx = int(env["object_idx"])
            env_contact[obj_idx, 0] = 1
            if frame_idx <= 0 or previous_env_contact[obj_idx, 0] != 0:
                continue
            environment_contact_events.append(
                {
                    "event_id": len(environment_contact_events),
                    "participants": [int(env["object_id"]), int(env["environment_id"])],
                    "object_indices": [obj_idx, -1],
                    "frame_idx": int(frame_idx),
                    "start_frame": int(frame_idx),
                    "peak_frame": int(frame_idx),
                    "end_frame": int(frame_idx),
                    "impulse_peak": 0.0,
                    "contact_duration": 1,
                    "environment_name": "ground",
                }
            )
        env_contact_series.append(env_contact)
        env_contact_impulse_series.append(env_impulse)
        previous_env_contact = env_contact
    contact_graph_arr = np.stack(contact_graph_frames, axis=0).astype(np.uint8)
    contact_impulse_arr = np.zeros_like(contact_graph_arr, dtype=np.float32)
    env_contact_arr = np.stack(env_contact_series, axis=0).astype(np.uint8)
    env_contact_impulse_arr = np.stack(env_contact_impulse_series, axis=0).astype(np.float32)
    frame_phase_arr, event_windows, collision_events = try1._summarize_contact_windows(contact_graph_arr, object_ids_arr)
    env_event_windows = try1.summarize_environment_contact_windows(environment_contact_events) if hasattr(try1, "summarize_environment_contact_windows") else []
    collision_events.extend(environment_contact_events)

    anchor_targets = try1.compute_anchor_targets(
        seg_frames=seg_arr,
        depth_metric_frames=depth_metric_arr,
        com_pos_frames=com_arr,
        object_ids=object_ids_arr,
        seg_ids=seg_ids_arr,
        camera_cfg=camera_cfg,
        cam_intrinsics=cam_intrinsics,
    )
    flow_arr = try1._build_flow_fallback(anchor_targets["com_uv"], anchor_targets["visibility_mask"], seg_arr)

    sample_name = f"sample_{sample_index:06d}"
    case_dir = _scene_layout_path(args.output_root, scene_composition, bucket, sample_name)
    try1.prepare_case_output_dirs(case_dir)
    for frame_idx, frame in enumerate(rgb_arr):
        imageio.imwrite(case_dir / "rgb" / f"frame_{frame_idx:03d}.png", frame)
    for frame_idx, frame in enumerate(depth_norm_arr):
        imageio.imwrite(case_dir / "depth" / f"frame_{frame_idx:03d}.png", try1.depth_to_uint8(frame))
    imageio.mimwrite(case_dir / "videos" / "rgb.mp4", [np.asarray(frame) for frame in rgb_arr], fps=int(args.fps), quality=8)
    imageio.mimwrite(case_dir / "videos" / "depth.mp4", [try1.depth_to_uint8(frame) for frame in depth_norm_arr], fps=int(args.fps), quality=8)
    try1.save_vis_video(case_dir / "visualizations" / "depth_vis.mp4", [try1.depth_to_vis(frame, near=float(cam_intrinsics["near"]), far=float(cam_intrinsics["far"])) for frame in depth_metric_arr], fps=int(args.fps))

    physics_dir = case_dir / "physics"
    np.save(physics_dir / "depth_metric.npy", depth_metric_arr)
    np.save(physics_dir / "depth_normalized.npy", depth_norm_arr)
    np.save(physics_dir / "seg.npy", seg_arr)
    np.save(physics_dir / "trajectory.npy", trajectory_arr)
    np.save(physics_dir / "contact_graph.npy", contact_graph_arr)
    np.save(physics_dir / "contact_impulse.npy", contact_impulse_arr)
    np.save(physics_dir / "env_contact.npy", env_contact_arr)
    np.save(physics_dir / "env_contact_impulse.npy", env_contact_impulse_arr)
    np.save(physics_dir / "frame_phase.npy", frame_phase_arr.astype(np.int8))
    np.save(physics_dir / "flow.npy", flow_arr.astype(np.float32))
    np.savez_compressed(physics_dir / "anchor_targets.npz", **anchor_targets)
    np.savez_compressed(
        physics_dir / "rigid_kinematics.npz",
        object_ids=object_ids_arr.astype(np.int32),
        seg_ids=seg_ids_arr.astype(np.int32),
        com_pos=com_arr,
        orientation_quat=quat_arr,
        linear_vel=lin_arr,
        angular_vel=ang_arr,
        com_uv=anchor_targets["com_uv"],
        bbox_xyxy=anchor_targets["bbox_xyxy"],
        visibility_mask=anchor_targets["visibility_mask"],
        center_depth=anchor_targets["center_depth"],
        kinetic_energy=np.asarray(kinetic_frames, dtype=np.float32),
        potential_energy=np.asarray(potential_frames, dtype=np.float32),
        total_energy=np.asarray(total_energy_frames, dtype=np.float32),
    )
    np.savez_compressed(
        physics_dir / "energy.npz",
        kinetic_trans=np.asarray(kinetic_trans_frames, dtype=np.float32),
        kinetic_rot=np.asarray(kinetic_rot_frames, dtype=np.float32),
        potential_gravity=np.asarray(potential_gravity_frames, dtype=np.float32),
        mechanical_total=np.asarray(total_energy_frames, dtype=np.float32),
    )
    write_json(physics_dir / "collision_events.json", collision_events)
    write_json(physics_dir / "event_windows.json", event_windows + env_event_windows)
    write_json(physics_dir / "env_collision_events.json", environment_contact_events)
    write_json(physics_dir / "env_event_windows.json", env_event_windows)
    write_json(physics_dir / "properties.json", {"object_ids": object_ids_arr.tolist(), "sampled_restitution": [None] * len(records), "effective_restitution_used": [None] * len(records)})

    objects_meta = []
    for idx, rec in enumerate(records):
        prep = rec["prepared"]
        meta = rec["metadata"]
        bbox_min = np.asarray(meta.get("object_bbox_min", [0, 0, 0]), dtype=np.float64)
        bbox_max = np.asarray(meta.get("object_bbox_max", [0, 0, 0]), dtype=np.float64)
        objects_meta.append(
            {
                "object_id": int(idx),
                "seg_id": int(idx + 1),
                "source_object_id": str(prep.object_id),
                "entity_type": "rigid_assembly",
                "role": str(rec["role"]),
                "motion_type": str(rec["motion_type"]),
                "sampled_motion_type": str(rec["sampled_motion_type"]),
                "motion_group": str(rec["motion_group"]),
                "object_motion_type": str(rec["motion_type"]),
                "object_motion_group": str(rec["motion_group"]),
                "source_tag": "physxnet_main",
                "dataset_source": "PhysXNet",
                "canonical_size": {"bbox_extent": np.maximum(bbox_max - bbox_min, 1e-6).astype(float).tolist()},
                "bbox_volume_est_m3": float(rec["bbox_volume_est_m3"]),
                "large_static_by_volume_threshold": bool(rec["large_static_by_volume_threshold"]),
                "volume_threshold_m3": float(args.physxnet_volume_threshold_m3),
                "initial_position": np.asarray(rec["pos"], dtype=float).tolist(),
                "initial_yaw_deg": float(rec["yaw_deg"]),
                "initial_linear_velocity": np.asarray(rec["linear_velocity"], dtype=float).tolist(),
                "initial_angular_velocity": np.asarray(rec["angular_velocity"], dtype=float).tolist(),
            }
        )

    scene_input = {
        "scene_id": sample_name,
        "sample_name": sample_name,
        "split": "train",
        "simulator_type": "rigid",
        "scene_composition": scene_composition,
        "object_count_bucket": bucket,
        "scene_type": scene_type,
        "motion_type": [obj["motion_type"] for obj in objects_meta],
        "volume_threshold_m3": float(args.physxnet_volume_threshold_m3),
        "camera": camera_cfg,
        "gravity": [0.0, 0.0, gravity_z],
        "objects": objects_meta,
    }
    write_json(case_dir / "scene_input.json", scene_input)
    try:
        output_relpath = str(case_dir.relative_to(args.output_root))
    except Exception:
        output_relpath = str(case_dir)
    metadata_payload = {
        "scene_id": sample_name,
        "output_relpath": output_relpath,
        "seed": int(seed),
        "split": "train",
        "family": "physxnet_multi_object_train",
        "simulator_type": "rigid",
        "scene_composition": scene_composition,
        "object_count_bucket": bucket,
        "scene_type": scene_type,
        "num_objects": int(len(records)),
        "frames": int(com_arr.shape[0]),
        "resolution": [int(args.resolution[0]), int(args.resolution[1])],
        "motion_category": scene_type,
        "motion_type": [obj["motion_type"] for obj in objects_meta],
        "volume_threshold_m3": float(args.physxnet_volume_threshold_m3),
        "convention": {
            "length_unit": "meter",
            "mass_unit": "kg",
            "time_unit": "second",
            "coordinate_system": "right-handed",
            "gravity_axis": "z_negative",
        },
        "simulation": {
            "engine": "Genesis",
            "engine_version": str(getattr(gs, "__version__", "unknown")),
            "dt": float(args.dt),
            "substeps": int(args.substeps),
            "steps_per_frame": int(save_every),
            "gravity": [0.0, 0.0, gravity_z],
        },
        "camera": camera_cfg,
        "camera_intrinsics": cam_intrinsics,
        "objects": objects_meta,
        "environment_entities": [{"name": "ground", "special_id": -1, "entity_type": "container"}],
        "outputs": {
            "metadata": "metadata.json",
            "scene_input": "scene_input.json",
            "rgb_video": "videos/rgb.mp4",
            "depth_video": "videos/depth.mp4",
            "depth_metric": "physics/depth_metric.npy",
            "depth_normalized": "physics/depth_normalized.npy",
            "segmentation": "physics/seg.npy",
            "trajectory": "physics/trajectory.npy",
            "flow": "physics/flow.npy",
            "anchor_targets": "physics/anchor_targets.npz",
            "rigid_kinematics": "physics/rigid_kinematics.npz",
            "energy": "physics/energy.npz",
            "properties": "physics/properties.json",
            "contact_graph": "physics/contact_graph.npy",
            "contact_impulse": "physics/contact_impulse.npy",
            "env_contact": "physics/env_contact.npy",
            "env_contact_impulse": "physics/env_contact_impulse.npy",
            "frame_phase": "physics/frame_phase.npy",
            "event_windows": "physics/event_windows.json",
            "env_collision_events": "physics/env_collision_events.json",
            "env_event_windows": "physics/env_event_windows.json",
            "depth_visualization_video": "visualizations/depth_vis.mp4",
        },
        "has_depth_metric": True,
        "has_seg": True,
        "has_contact_graph": True,
        "status": "ok",
    }
    write_json(case_dir / "metadata.json", metadata_payload)

    try:
        scene.destroy()
    except Exception:
        pass
    return case_dir


def write_dataset_format(output_root: Path) -> None:
    text = """# PhysXNet Rigid-Only Train Dataset Format

## Directory Layout

Training samples are organized only by `simulator_type + scene_composition + object_count_bucket`:

```text
train/
  rigid/
    uniform_dynamic/
      count_01/
      count_02/
      count_03_04/
      count_05_06/
    interaction_pair_plus_dynamic/
      count_02/
      count_03_04/
      count_05_06/
    dual_interaction_groups/
      count_04/
      count_05_06/
    omni_showcase/
      count_03_04/
      count_05_06/
```

`motion_type`, `role`, and `scene_type` are metadata fields only; they are not directory levels.

## Object Count Buckets

| bucket | meaning |
|---|---|
| `count_01` | 1 object |
| `count_02` | 2 objects |
| `count_03_04` | 3 or 4 objects |
| `count_04` | exactly 4 objects |
| `count_05_06` | 5 or 6 objects |

## Per-Sample Files

| path | shape / type | meaning |
|---|---|---|
| `metadata.json` | JSON | full schema, outputs, object roles, motion labels, camera, simulation |
| `scene_input.json` | JSON | compact scene construction input and labels |
| `videos/rgb.mp4` | video | RGB render |
| `videos/depth.mp4` | video | normalized depth visualization |
| `rgb/frame_*.png` | images | RGB frames |
| `depth/frame_*.png` | images | normalized depth frames |
| `physics/depth_metric.npy` | `[T,H,W] float32` | metric depth from camera render, meters |
| `physics/depth_normalized.npy` | `[T,H,W,1] float32` | depth normalized by camera near/far |
| `physics/seg.npy` | `[T,H,W] int32` | instance segmentation; background=0, object_id=k maps to seg_id=k+1 |
| `physics/flow.npy` | `[T-1,H,W,2] float32` | fallback optical flow; each visible object mask receives projected COM displacement from t to t+1 |
| `physics/trajectory.npy` | `[T,N,6] float32` | `[com_x,com_y,com_z,momentum_x,momentum_y,momentum_z]` |
| `physics/rigid_kinematics.npz` | NPZ | object kinematics; see fields below |
| `physics/anchor_targets.npz` | NPZ | 2D anchors aligned with segmentation |
| `physics/energy.npz` | NPZ | scene-level energy curves |
| `physics/contact_graph.npy` | `[T,N,N] uint8` | object-object AABB contact graph, symmetric, diagonal=0 |
| `physics/contact_impulse.npy` | `[T,N,N] float32` | placeholder impulse values; currently zeros |
| `physics/env_contact.npy` | `[T,N,1] uint8` | object-ground contact, ground special id = -1 |
| `physics/frame_phase.npy` | `[T] int8` | phase labels inferred from contacts |
| `physics/collision_events.json` | JSON list | object-object contact windows plus environment contact onsets after frame 0 |
| `physics/event_windows.json` | JSON list | contact windows; frame-0 support contact is excluded from collision windows |
| `physics/properties.json` | JSON | physical property export; runtime restitution is null if unavailable |

## `rigid_kinematics.npz`

| field | shape | meaning |
|---|---|---|
| `object_ids` | `[N]` | object ids; order matches `metadata.json.objects` |
| `seg_ids` | `[N]` | segmentation labels, `seg_id=object_id+1` |
| `com_pos` | `[T,N,3]` | world-space center of mass, meters |
| `orientation_quat` | `[T,N,4]` | quaternion in Genesis order |
| `linear_vel` | `[T,N,3]` | linear velocity, m/s |
| `angular_vel` | `[T,N,3]` | angular velocity, rad/s |
| `com_uv` | `[T,N,2]` | projected COM pixel coordinate |
| `bbox_xyxy` | `[T,N,4]` | 2D bbox from `seg.npy` |
| `visibility_mask` | `[T,N]` | whether object appears in segmentation |
| `center_depth` | `[T,N]` | median metric depth inside object mask |
| `kinetic_energy` | `[T]` | total kinetic energy |
| `potential_energy` | `[T]` | gravity potential energy |
| `total_energy` | `[T]` | kinetic + potential |

## Labels

Each object in `metadata.json.objects` and `scene_input.json.objects` includes:

- `object_id`, `seg_id`
- `entity_type`
- `role`: `initiator`, `target`, or `bystander`
- `motion_type`: per-object motion label such as `static_rest`, `front_slide_in`, `high_drop`, `roll_left`
- `motion_group`: grouped motion family
- `source_object_id`: original PhysXNet object id
- `source_tag`: currently `physxnet_main`

"""
    (output_root / "DATASET_FORMAT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.cache_root = (args.cache_root or (args.output_root / "_asset_cache")).resolve()
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    try1.EXPORT_CAMERA_RESOLUTION = (int(args.resolution[0]), int(args.resolution[1]))
    write_dataset_format(args.output_root)

    plan = build_plan(args)
    max_objects = max(bucket_object_count(bucket, random.Random(args.seed + i)) for i, (_, bucket, _) in enumerate(plan)) if plan else 1
    pool = sample_object_pool(args, max_objects * max(2, len(plan)))
    rng = random.Random(int(args.seed) + 500)
    manifest = {
        "name": "physxnet_rigid_train_multi",
        "split": "train",
        "simulator_type": "rigid",
        "layout": "train/rigid/<scene_composition>/<object_count_bucket>/<sample>",
        "records": [],
        "failed": [],
    }
    for global_idx, (composition, bucket, _) in enumerate(plan):
        count = bucket_object_count(bucket, random.Random(args.seed + global_idx * 17))
        last_error: Optional[Exception] = None
        success = False
        for attempt in range(max(1, int(args.max_scene_sampling_attempts))):
            object_ids = _sample_object_ids(pool, count, rng)
            try:
                sample_dir = generate_sample(
                    args=args,
                    sample_index=global_idx,
                    scene_composition=composition,
                    bucket=bucket,
                    object_ids=object_ids,
                )
                rel = str(sample_dir.relative_to(args.output_root))
                manifest["records"].append({
                    "scene_id": sample_dir.name,
                    "relative_path": rel,
                    "simulator_type": "rigid",
                    "scene_composition": composition,
                    "object_count_bucket": bucket,
                    "num_objects": count,
                    "object_ids": object_ids,
                    "sampling_attempt": attempt + 1,
                })
                print(f"[OK] {rel} objects={object_ids} attempt={attempt + 1}")
                success = True
                break
            except Exception as exc:
                last_error = exc
                try:
                    import genesis as gs
                    gs.destroy()
                except Exception:
                    pass
        if not success:
            manifest["failed"].append({
                "sample_index": global_idx,
                "scene_composition": composition,
                "object_count_bucket": bucket,
                "num_objects": count,
                "error": str(last_error) if last_error is not None else "unknown error",
            })
            print(f"[FAIL] idx={global_idx} comp={composition} bucket={bucket} err={last_error}")
    write_json(args.output_root / "dataset_manifest.json", manifest)
    try:
        import genesis as gs
        gs.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
