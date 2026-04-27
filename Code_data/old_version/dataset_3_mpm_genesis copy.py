#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Genesis dataset from PhysXNet articulated assets.

Design goals:
1) Reuse the corrected Y-up -> Z-up conversion and strict per-part export from
   physxnet_articulation_demo.py.
2) Follow the scene/dataset-construction style of
   genesis_demo_physxnet_urdf_loader_merge.py: asset bank -> scene sampling ->
   rendering/export.
3) Preserve dataset-provided physical parameters as much as Genesis allows:
   - rigid part masses / inertias / colors come from the exported URDF
   - soft parts reuse exact density / Young's modulus / Poisson ratio
4) Keep object scale, container scale, and placement mutually reasonable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np

from physxnet_articulation_demo import (
    _configure_genesis_rigid_entity_from_metadata,
    _default_entity_rigid_material,
    _make_genesis_rigid_material,
    _make_pbd_cloth_material_from_part,
    prepare_physxnet_object,
)


IMG_W, IMG_H = 960, 720
MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 4
CAMERA_PC_STRIDE = 2

TARGET_LONGEST_SIZE_RANGE = (0.18, 0.42)
STATIC_REST_PROB = 0.38
PREVIEW_FPS = 30

TOP_DROP_Z_RANGE = (1.00, 1.55)
TOP_TOSS_Z_RANGE = (0.95, 1.45)
FRONT_SLIDE_Z_RANGE = (0.16, 0.34)

STRIKE_SPEED_RANGE = (0.80, 1.35)
TOP_DROP_ANGVEL = 1.6
TOP_TOSS_ANGVEL = 2.2
FRONT_SLIDE_ANGVEL = 1.8

REST_CONTACT_MARGIN = 0.02
SLIDE_CONTACT_MARGIN_RANGE = (0.015, 0.035)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def weighted_choice(d: Dict[str, float]) -> str:
    keys = list(d.keys())
    probs = np.asarray(list(d.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(keys, p=probs))


def to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def safe_subsample_points(xyz: np.ndarray, max_points: int = 2048) -> np.ndarray:
    xyz = np.asarray(xyz)
    if len(xyz) <= max_points:
        return xyz.astype(np.float32, copy=False)
    idx = np.random.choice(len(xyz), size=max_points, replace=False)
    return xyz[idx].astype(np.float32, copy=False)


def _try_import_genesis():
    try:
        import genesis as gs
        return gs
    except Exception as e:
        raise RuntimeError(f"Failed to import genesis: {e}")


_GENESIS_INITIALIZED = False


def ensure_genesis_initialized(seed: int = 0) -> Any:
    global _GENESIS_INITIALIZED
    gs = _try_import_genesis()
    if not _GENESIS_INITIALIZED:
        gs.init(seed=int(seed), precision="32", logging_level="warning")
        _GENESIS_INITIALIZED = True
    return gs


def _try_call_methods(obj: Any, method_names: List[str], value: Any) -> bool:
    for name in method_names:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            try:
                fn(value)
                return True
            except Exception:
                try:
                    fn(tuple(np.asarray(value).tolist()))
                    return True
                except Exception:
                    pass
    return False


def apply_initial_motion_to_entity(ent: Any, linvel: List[float], angvel: List[float]) -> None:
    v = np.asarray(linvel, dtype=np.float32)
    w = np.asarray(angvel, dtype=np.float32)
    if np.linalg.norm(v) > 0:
        _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], v)
    if np.linalg.norm(w) > 0:
        _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], w)


def _get_entity_world_min_z(ent: Any) -> Optional[float]:
    # Prefer dense geometry query when available; this is the most reliable way
    # to detect initial floor penetration for articulated URDF entities.
    if hasattr(ent, "get_verts"):
        try:
            verts = to_numpy(ent.get_verts())
            if verts is not None and verts.size > 0:
                verts = verts.reshape(-1, 3)
                return float(np.min(verts[:, 2]))
        except Exception:
            pass
    if hasattr(ent, "get_particles_pos"):
        try:
            pts = to_numpy(ent.get_particles_pos())
            if pts is not None and pts.size > 0:
                pts = pts.reshape(-1, 3)
                return float(np.min(pts[:, 2]))
        except Exception:
            pass
    return None


def _lift_entity_if_penetrating_floor(ent: Any, floor_top_z: float, margin: float) -> float:
    min_z = _get_entity_world_min_z(ent)
    if min_z is None or (not np.isfinite(min_z)):
        return 0.0

    target_min_z = float(floor_top_z) + float(margin)
    dz = target_min_z - float(min_z)
    if dz <= 0.0:
        return 0.0

    if hasattr(ent, "get_pos"):
        try:
            pos = to_numpy(ent.get_pos()).reshape(-1)
            new_pos = np.asarray([float(pos[0]), float(pos[1]), float(pos[2] + dz)], dtype=np.float32)
            if _try_call_methods(ent, ["set_pos", "set_position"], new_pos):
                return float(dz)
        except Exception:
            pass
    return 0.0


def _enforce_entity_static(ent: Any, anchor_pos: Optional[Tuple[float, float, float]] = None) -> None:
    if hasattr(ent, "set_fixed"):
        try:
            ent.set_fixed(True)
        except Exception:
            pass
    if hasattr(ent, "set_kinematic"):
        try:
            ent.set_kinematic(True)
        except Exception:
            pass
    if anchor_pos is not None:
        _try_call_methods(ent, ["set_pos", "set_position"], np.asarray(anchor_pos, dtype=np.float32))
    _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], np.zeros(3, dtype=np.float32))
    _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], np.zeros(3, dtype=np.float32))


def create_genesis_rigid_material(gs: Any, mat_cfg: Dict[str, Any]):
    kwargs = {
        "rho": float(mat_cfg["rho"]),
        "friction": float(mat_cfg["friction"]),
    }
    if mat_cfg.get("restitution") is not None:
        kwargs["restitution"] = float(mat_cfg["restitution"])
    try:
        return gs.materials.Rigid(**kwargs)
    except TypeError:
        kwargs.pop("restitution", None)
        return gs.materials.Rigid(**kwargs)


def _default_rigid_material_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    densities = []
    frictions = []
    restitutions = []
    rigid_records = metadata.get("rigid_part_links", [])
    for rec in rigid_records:
        if rec.get("density_kgm3") is not None:
            densities.append(float(rec["density_kgm3"]))
        if rec.get("friction") is not None:
            frictions.append(float(rec["friction"]))
        if rec.get("restitution") is not None:
            restitutions.append(float(rec["restitution"]))

    density = float(np.median(densities)) if densities else 1000.0
    friction = float(np.median(frictions)) if frictions else 0.55
    restitution = float(np.median(restitutions)) if restitutions else 0.10
    return {
        "family": "Rigid",
        "name": "physxnet_articulation_root",
        "rho": density,
        "friction": float(np.clip(friction, 1e-2, 5.0)),
        "restitution": float(np.clip(restitution, 0.0, 1.2)),
    }


def _list_physxnet_object_ids(physx_root: Path, version: str, object_ids: Optional[List[str]], max_objects: int) -> List[str]:
    finaljson_dir = physx_root / version / "finaljson"
    if object_ids:
        ids = [str(x) for x in object_ids]
    else:
        ids = sorted([p.stem for p in finaljson_dir.glob("*.json")])
    if max_objects not in (None, 0):
        ids = ids[: int(max_objects)]
    return ids


def prepare_articulation_asset(
    physx_root: Path,
    version: str,
    object_id: str,
    prepared_asset_root: Path,
    voxel_pitch: float,
    object_scale_mult: float,
    force_rebuild: bool,
    solver_family_override: Optional[str] = None,
) -> Dict[str, Any]:
    asset_dir = prepared_asset_root / object_id
    meta_path = asset_dir / "meta" / "metadata.json"
    if force_rebuild or not meta_path.exists():
        prepare_physxnet_object(
            physx_root=physx_root,
            version=version,
            object_id=object_id,
            output_root=prepared_asset_root,
            voxel_pitch=voxel_pitch,
            object_scale_mult=object_scale_mult,
            solver_family_override=solver_family_override,
        )

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    bbox_min = np.asarray(metadata["object_bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(metadata["object_bbox_max"], dtype=np.float64)
    bbox_extents = np.maximum(bbox_max - bbox_min, 1e-6)

    rigid_count = len(metadata.get("rigid_part_links", []))
    soft_count = len(metadata.get("soft_parts", []))

    return {
        "asset_id": f"physxnet_articulation__{object_id}",
        "dataset_name": "physxnet_articulation",
        "object_id": str(object_id),
        "object_name": str(metadata.get("object_name", object_id)),
        "category": str(metadata.get("category", "unknown")),
        "asset_dir": str(asset_dir),
        "metadata_path": str(meta_path),
        "urdf_path": str(asset_dir / "rigid" / f"{object_id}.urdf"),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extents": bbox_extents.tolist(),
        "grounding_offset_z": float(metadata["grounding_offset_z"]),
        "material_override": _default_rigid_material_from_metadata(metadata),
        "rigid_part_count": rigid_count,
        "soft_part_count": soft_count,
        "metadata": metadata,
    }


def build_asset_bank(
    physx_root: Path,
    version: str,
    prepared_asset_root: Path,
    manifest_path: Path,
    voxel_pitch: float,
    object_scale_mult: float,
    object_ids: Optional[List[str]],
    max_objects: int,
    force_rebuild_assets: bool,
    solver_family_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_dir(prepared_asset_root)
    ensure_dir(manifest_path.parent)

    bank: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for object_id in _list_physxnet_object_ids(physx_root, version, object_ids, max_objects):
        try:
            bank.append(
                prepare_articulation_asset(
                    physx_root=physx_root,
                    version=version,
                    object_id=object_id,
                    prepared_asset_root=prepared_asset_root,
                    voxel_pitch=voxel_pitch,
                    object_scale_mult=object_scale_mult,
                    force_rebuild=force_rebuild_assets,
                    solver_family_override=solver_family_override,
                )
            )
        except Exception as e:
            failed.append({"object_id": object_id, "error": str(e)})
            print(f"[WARN] skip articulation asset {object_id}: {e}")

    manifest = {
        "dataset_name": "physxnet_articulation_dataset",
        "physx_root": str(physx_root),
        "version": version,
        "prepared_asset_root": str(prepared_asset_root),
        "n_assets": len(bank),
        "n_failed_assets": len(failed),
        "assets": bank,
        "failed_assets": failed,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] articulation assets: usable={len(bank)} failed={len(failed)}")
    return bank


def compute_bound_radius(extents: np.ndarray) -> float:
    extents = np.asarray(extents, dtype=np.float64)
    return float(np.linalg.norm(0.5 * extents))


def sample_container_for_objects(scaled_extents: List[np.ndarray]) -> Dict[str, Any]:
    max_ext = np.max(np.stack(scaled_extents, axis=0), axis=0)
    half_x = float(np.clip(max(0.95, 2.1 * max_ext[0] + 0.30), 0.95, 1.70))
    half_y = float(np.clip(max(1.05, 2.4 * max_ext[1] + 0.42), 1.05, 1.95))
    wall_height = float(np.clip(max(1.00, 2.8 * max_ext[2] + 0.42), 1.00, 2.10))
    return {
        "half_x": half_x,
        "half_y": half_y,
        "wall_thickness": 0.10,
        "wall_height": wall_height,
        "floor_thickness": 0.08,
        "center": [0.0, 0.0, 0.0],
        "front_keep_out": 0.38,
        "back_keep_out": 0.12,
        "side_keep_out": 0.08,
    }


def sample_camera(container_cfg: Dict[str, Any]) -> Dict[str, Any]:
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wh = container_cfg["wall_height"]
    cx, cy, cz = container_cfg["center"]
    return {
        "res": [IMG_W, IMG_H],
        "pos": [
            float(cx + np.random.uniform(-0.10, 0.10)),
            float(cy - hy - 1.80 + np.random.uniform(-0.15, 0.10)),
            float(cz + 0.70 * wh + np.random.uniform(-0.08, 0.10)),
        ],
        "lookat": [
            float(cx + np.random.uniform(-0.06, 0.06)),
            float(cy + np.random.uniform(0.12, 0.28)),
            float(cz + 0.24 * wh + np.random.uniform(-0.03, 0.08)),
        ],
        "fov": float(np.random.uniform(38.0, 46.0)),
        "GUI": False,
    }


def sample_spawn_xy(container_cfg: Dict[str, Any], half_x: float, half_y: float, bias_to_back: bool) -> Tuple[float, float]:
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]

    x_min = -hx + wt + container_cfg["side_keep_out"] + half_x
    x_max = hx - wt - container_cfg["side_keep_out"] - half_x
    y_front = -hy + wt + container_cfg["front_keep_out"] + half_y
    y_back = hy - wt - container_cfg["back_keep_out"] - half_y

    if bias_to_back:
        y_min = max(y_front, 0.05)
        y_max = y_back
    else:
        y_min = y_front
        y_max = y_back

    if x_min >= x_max:
        x_min, x_max = -0.05, 0.05
    if y_min >= y_max:
        y_min, y_max = 0.05, 0.15

    return float(np.random.uniform(x_min, x_max)), float(np.random.uniform(y_min, y_max))


def _random_angvel(scale: float) -> List[float]:
    axis = np.random.uniform(-1.0, 1.0, size=3)
    n = np.linalg.norm(axis)
    if n < 1e-6:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = axis / n
    mag = float(np.random.uniform(0.2, scale))
    return [float(x) for x in axis * mag]


def sample_motion_for_object(
    pattern: str,
    container_cfg: Dict[str, Any],
    scaled_extents: np.ndarray,
    grounding_offset_z: float,
    target_pos: Optional[np.ndarray] = None,
    index_in_scene: int = 0,
) -> Dict[str, Any]:
    half_x, half_y, half_z = [float(x) * 0.5 for x in scaled_extents]
    floor_top_z = float(container_cfg["center"][2]) + float(container_cfg["floor_thickness"])
    rest_z = floor_top_z + float(grounding_offset_z) + REST_CONTACT_MARGIN
    if pattern == "strike_static":
        if index_in_scene == 0:
            x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
            return {
                "motion_type": "static_rest",
                "init_pos": [x, y, rest_z],
                "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
                "init_linvel": [0.0, 0.0, 0.0],
                "init_angvel": [0.0, 0.0, 0.0],
            }

        strike_target = np.asarray(target_pos if target_pos is not None else [0.0, 0.20, rest_z], dtype=np.float64)
        start_x = -container_cfg["half_x"] + 0.12 + half_x
        start_y = float(np.clip(strike_target[1] + np.random.uniform(-0.08, 0.08), -0.05, 0.35))
        speed = float(np.random.uniform(*STRIKE_SPEED_RANGE))
        return {
            "motion_type": "strike_static",
            "init_pos": [start_x, start_y, rest_z + 0.005],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-0.4, 0.4))],
            "init_linvel": [speed, float(np.random.uniform(-0.10, 0.10)), 0.0],
            "init_angvel": _random_angvel(FRONT_SLIDE_ANGVEL),
        }

    if pattern == "front_slide_in":
        x = float(np.random.uniform(-0.55, 0.55))
        y = -container_cfg["half_y"] + 0.06 + half_y
        return {
            "motion_type": "front_slide_in",
            "init_pos": [x, y, rest_z + float(np.random.uniform(*SLIDE_CONTACT_MARGIN_RANGE))],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [float(np.random.uniform(-0.18, 0.18)), float(np.random.uniform(1.00, 1.75)), 0.0],
            "init_angvel": _random_angvel(FRONT_SLIDE_ANGVEL),
        }

    static_rest = np.random.rand() < STATIC_REST_PROB
    if static_rest:
        x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
        return {
            "motion_type": "static_rest",
            "init_pos": [x, y, rest_z],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [0.0, 0.0, 0.0],
            "init_angvel": [0.0, 0.0, 0.0],
        }

    x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=False)
    z = float(np.random.uniform(*TOP_DROP_Z_RANGE))
    mode = "top_drop"
    linvel = [float(np.random.uniform(-0.10, 0.10)), float(np.random.uniform(-0.10, 0.10)), 0.0]

    if np.random.rand() < 0.45:
        z = float(np.random.uniform(*TOP_TOSS_Z_RANGE))
        mode = "top_toss"
        linvel = [
            float(np.random.uniform(-0.25, 0.25)),
            float(np.random.uniform(-0.20, 0.20)),
            float(np.random.uniform(-0.65, -0.05)),
        ]

    return {
        "motion_type": mode,
        "init_pos": [x, y, z],
        "init_euler": [
            float(np.random.uniform(-0.4, 0.4)),
            float(np.random.uniform(-0.4, 0.4)),
            float(np.random.uniform(-math.pi, math.pi)),
        ],
        "init_linvel": linvel,
        "init_angvel": _random_angvel(TOP_TOSS_ANGVEL if mode == "top_toss" else TOP_DROP_ANGVEL),
    }


def scale_soft_parts(soft_parts: List[Dict[str, Any]], scene_scale: float) -> List[Dict[str, Any]]:
    scaled = []
    for rec in soft_parts:
        item = dict(rec)
        item["scene_scale"] = float(scene_scale)
        scaled.append(item)
    return scaled


def sample_scene(scene_index: int, asset_bank: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    set_seed(seed)
    n_objects = random.randint(2, 4)
    chosen_assets = random.sample(asset_bank, k=min(n_objects, len(asset_bank)))
    pattern = weighted_choice({
        "drop_cluster": 0.48,
        "strike_static": 0.34,
        "front_slide_in": 0.18,
    })

    scene_scales = []
    scaled_extents = []
    for asset in chosen_assets:
        ext = np.asarray(asset["bbox_extents"], dtype=np.float64)
        target_longest = float(np.random.uniform(*TARGET_LONGEST_SIZE_RANGE))
        scene_scale = target_longest / max(float(np.max(ext)), 1e-8)
        scene_scales.append(scene_scale)
        scaled_extents.append(ext * scene_scale)

    container_cfg = sample_container_for_objects(scaled_extents)
    camera_cfg = sample_camera(container_cfg)

    objects = []
    strike_target = None
    for idx, (asset, scene_scale, extents) in enumerate(zip(chosen_assets, scene_scales, scaled_extents)):
        # motion = sample_motion_for_object(
        #     pattern=pattern,
        #     container_cfg=container_cfg,
        #     scaled_extents=np.asarray(extents, dtype=np.float64),
        #     target_pos=strike_target,
        #     index_in_scene=idx,
        # )
        motion = sample_motion_for_object(
            pattern=pattern,
            container_cfg=container_cfg,
            scaled_extents=np.asarray(extents, dtype=np.float64),
            grounding_offset_z=float(asset["grounding_offset_z"]) * float(scene_scale),
            target_pos=strike_target,
            index_in_scene=idx,
        )
        if idx == 0:
            strike_target = np.asarray(motion["init_pos"], dtype=np.float64)

        objects.append(
            {
                "scene_object_id": idx,
                "asset_id": asset["asset_id"],
                "physx_object_id": asset["object_id"],
                "object_name": asset["object_name"],
                "category": asset["category"],
                "solver": "ArticulationRigid",
                "source_type": "physxnet_articulation",
                "pattern": pattern,
                "motion_type": motion["motion_type"],
                "geom": {
                    "shape": "urdf",
                    "urdf_file": asset["urdf_path"],
                    "scale": float(scene_scale),
                    "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
                    "bound_radius": compute_bound_radius(np.asarray(extents, dtype=np.float64)),
                },
                "material": dict(asset["material_override"]),
                "init_pos": [float(x) for x in motion["init_pos"]],
                "init_euler": [float(x) for x in motion["init_euler"]],
                "init_linvel": [float(x) for x in motion["init_linvel"]],
                "init_angvel": [float(x) for x in motion["init_angvel"]],
                "soft_parts": scale_soft_parts(asset["metadata"].get("soft_parts", []), scene_scale),
                "rigid_part_count": int(asset["rigid_part_count"]),
                "soft_part_count": int(asset["soft_part_count"]),
                "prepared_asset_dir": asset["asset_dir"],
                "prepared_metadata_path": asset["metadata_path"],
                "grounding_offset_z": float(asset["grounding_offset_z"]) * float(scene_scale),
            }
        )

    return {
        "scene_id": f"train_scene_{scene_index:06d}",
        "seed": seed,
        "family": "physxnet_articulation",
        "pattern": pattern,
        "container": container_cfg,
        "camera": camera_cfg,
        "background": {
            "name": "plain_open_container",
            "background_color": [0.96, 0.97, 0.99, 1.0],
            "ambient_light": [0.65, 0.65, 0.65],
        },
        "sim_options": {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 1e-3,# dt 是每个 step 的时间长度
            "substeps": 10,# substep_dt = dt / substeps
            "num_steps": 5000,# 总共推进 num_steps 个外层 step
        },# T=dt×num_steps 是整个模拟的时间长度
        "objects": objects,
    }


def add_container(gs: Any, scene: Any, container_cfg: Dict[str, Any]) -> Dict[str, Any]:
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]
    wh = container_cfg["wall_height"]
    ft = container_cfg["floor_thickness"]
    cx, cy, cz = container_cfg["center"]

    mat = gs.materials.Rigid(rho=1200.0, friction=0.98)
    container_entities: Dict[str, Dict[str, Any]] = {}
    floor_pos = (cx, cy, cz + ft / 2.0)
    container_entities["floor"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(2 * hx, 2 * hy, ft), pos=floor_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.70, 0.72, 0.76, 1.0)),
        ),
        "anchor_pos": floor_pos,
    }
    left_wall_pos = (cx - hx + wt / 2.0, cy, cz + ft + wh / 2.0)
    container_entities["left_wall"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(wt, 2 * hy, wh), pos=left_wall_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.80, 0.63, 0.61, 1.0)),
        ),
        "anchor_pos": left_wall_pos,
    }
    right_wall_pos = (cx + hx - wt / 2.0, cy, cz + ft + wh / 2.0)
    container_entities["right_wall"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(wt, 2 * hy, wh), pos=right_wall_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.62, 0.80, 0.67, 1.0)),
        ),
        "anchor_pos": right_wall_pos,
    }
    back_wall_pos = (cx, cy + hy - wt / 2.0, cz + ft + wh / 2.0)
    container_entities["back_wall"] = {
        "entity": scene.add_entity(
            morph=gs.morphs.Box(size=(2 * hx, wt, wh), pos=back_wall_pos, fixed=True),
            material=mat,
            surface=gs.surfaces.Default(color=(0.63, 0.71, 0.84, 1.0)),
        ),
        "anchor_pos": back_wall_pos,
    }
    return container_entities


def build_scene(scene_cfg: Dict[str, Any]):
    gs = ensure_genesis_initialized(seed=int(scene_cfg.get("seed", 0)))

    vis_options = gs.options.VisOptions(
        show_world_frame=False,
        show_link_frame=False,
        background_color=tuple(scene_cfg["background"]["background_color"]),
        ambient_light=tuple(scene_cfg["background"]["ambient_light"]),
        segmentation_level="entity",
        render_particle_as="sphere",
        particle_size_scale=1.0,
    )

    scene_kwargs: Dict[str, Any] = {
        "sim_options": gs.options.SimOptions(
            gravity=tuple(scene_cfg["sim_options"]["gravity"]),
            dt=scene_cfg["sim_options"]["dt"],
            substeps=scene_cfg["sim_options"]["substeps"],
        ),
        "vis_options": vis_options,
        "show_viewer": False,
    }

    try:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(
            dt=scene_cfg["sim_options"]["dt"],
            enable_collision=True,
            use_gjk_collision=True,
            batch_dofs_info=True,
            batch_joints_info=True,
            batch_links_info=True,
        )
    except Exception:
        try:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(
                enable_collision=True,
                batch_dofs_info=True,
                batch_joints_info=True,
                batch_links_info=True,
            )
        except Exception:
            pass

    if any(obj.get("soft_part_count", 0) > 0 for obj in scene_cfg["objects"]):
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(
            lower_bound=(-2.2, -2.2, -0.2),
            upper_bound=(2.2, 2.2, 2.8),
        )
        try:
            scene_kwargs["pbd_options"] = gs.options.PBDOptions()
        except Exception:
            pass
        if hasattr(gs.options, "CouplerOptions"):
            try:
                scene_kwargs["coupler_options"] = gs.options.CouplerOptions(rigid_mpm=True, rigid_pbd=True)
            except Exception:
                pass

    scene = gs.Scene(**scene_kwargs)
    container_entities = add_container(gs, scene, scene_cfg["container"])

    entities = []
    state_specs = []
    runtime_records = []

    for obj in scene_cfg["objects"]:
        obj_metadata = json.loads(Path(obj["prepared_metadata_path"]).read_text(encoding="utf-8"))
        rigid_material_cfg = _default_entity_rigid_material(
            obj_metadata,
            default_friction=float(obj["material"].get("friction", 0.55)),
        )
        material = _make_genesis_rigid_material(
            gs,
            rho=float(rigid_material_cfg["rho"]),
            friction=float(rigid_material_cfg["friction"]),
            restitution=float(rigid_material_cfg["restitution"]),
        )
        euler = tuple(obj["init_euler"])
        pos = tuple(obj["init_pos"])
        urdf_kwargs = dict(
            file=obj["geom"]["urdf_file"],
            scale=obj["geom"]["scale"],
            pos=pos,
            euler=euler,
            visualization=True,
            collision=True,
            fixed=False,
            merge_fixed_links=True,
            prioritize_urdf_material=True,
        )
        try:
            ent = scene.add_entity(
                morph=gs.morphs.URDF(**urdf_kwargs),
                material=material,
            )
        except TypeError:
            fallback_keys = ["file", "scale", "pos", "euler", "visualization", "collision", "fixed"]
            ent = scene.add_entity(
                morph=gs.morphs.URDF(**{k: urdf_kwargs[k] for k in fallback_keys}),
                material=material,
            )
        entities.append(ent)
        runtime_records.append({"entity": ent, "metadata": obj_metadata, "scene_object_id": obj["scene_object_id"]})
        state_specs.append(
            {
                "object_id": obj["scene_object_id"],
                "name": obj["object_name"],
                "solver": "Rigid",
                "entity": ent,
            }
        )

        for soft in obj.get("soft_parts", []):
            density = float(soft["density_kgm3"]) if soft.get("density_kgm3") is not None else 800.0
            youngs = float(soft["youngs_pa"]) if soft.get("youngs_pa") is not None else 1e7
            poisson = float(soft["poisson"]) if soft.get("poisson") is not None else 0.30
            scale = float(soft.get("scene_scale", obj["geom"]["scale"]))

            try:
                if soft["solver_family"] == "pbd_cloth":
                    soft_ent = scene.add_entity(
                        material=_make_pbd_cloth_material_from_part(
                            gs,
                            density=float(density),
                            friction=soft.get("friction", None),
                            youngs=soft.get("youngs_pa", None),
                            damping=soft.get("damping", None),
                        ),
                        morph=gs.morphs.Mesh(
                            file=soft["mesh_path"],
                            scale=scale,
                            pos=pos,
                            euler=euler,
                        ),
                        surface=gs.surfaces.Default(color=(0.20, 0.60, 0.88, 1.0), vis_mode="visual"),
                    )
                else:
                    if soft["solver_family"] == "mpm_elastoplastic":
                        try:
                            mpm_mat = gs.materials.MPM.ElastoPlastic(E=youngs, nu=poisson, rho=density)
                        except Exception:
                            mpm_mat = gs.materials.MPM.Elastic(E=youngs, nu=poisson, rho=density)
                    else:
                        mpm_mat = gs.materials.MPM.Elastic(E=youngs, nu=poisson, rho=density)
                    soft_ent = scene.add_entity(
                        material=mpm_mat,
                        morph=gs.morphs.Mesh(
                            file=soft["mesh_path"],
                            scale=scale,
                            pos=pos,
                            euler=euler,
                        ),
                        surface=gs.surfaces.Default(color=(0.86, 0.34, 0.34, 1.0), vis_mode="particle"),
                    )
                state_specs.append(
                    {
                        "object_id": f"{obj['scene_object_id']}_soft_{soft['part_id']}",
                        "name": f"{obj['object_name']}_soft_{soft['part_id']}",
                        "solver": soft["solver_family"],
                        "entity": soft_ent,
                    }
                )
            except Exception:
                continue

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )

    scene.build()

    for rec in container_entities.values():
        _enforce_entity_static(rec["entity"], rec["anchor_pos"])

    floor_top_z = float(scene_cfg["container"]["center"][2]) + float(scene_cfg["container"]["floor_thickness"])
    penetration_lift_records: List[Dict[str, Any]] = []
    for obj, ent in zip(scene_cfg["objects"], entities):
        lifted = _lift_entity_if_penetrating_floor(
            ent,
            floor_top_z=floor_top_z,
            margin=max(REST_CONTACT_MARGIN, 0.015),
        )
        penetration_lift_records.append(
            {
                "scene_object_id": int(obj["scene_object_id"]),
                "lift_dz": float(lifted),
            }
        )

    for obj, ent in zip(scene_cfg["objects"], entities):
        apply_initial_motion_to_entity(ent, obj["init_linvel"], obj["init_angvel"])

    for rec in runtime_records:
        rec["runtime_application"] = _configure_genesis_rigid_entity_from_metadata(
            rec["entity"],
            rec["metadata"],
            default_friction=0.55,
        )

    return scene, cam, entities, container_entities, state_specs, runtime_records, penetration_lift_records


def export_entity_state(ent: Any, state_spec: Dict[str, Any]) -> Dict[str, Any]:
    state = {
        "object_id": state_spec["object_id"],
        "solver": state_spec["solver"],
        "centroid": None,
        "quat": None,
        "vel": None,
        "ang": None,
        "pointcloud": None,
        "n_points": 0,
    }

    if hasattr(ent, "get_particles_pos"):
        try:
            pts = to_numpy(ent.get_particles_pos())
            if pts is not None and pts.size > 0:
                pts = pts.reshape(-1, 3)
                state["pointcloud"] = pts
                state["centroid"] = pts.mean(axis=0)
                state["n_points"] = int(len(pts))
                return state
        except Exception:
            pass

    if hasattr(ent, "get_verts"):
        try:
            verts = to_numpy(ent.get_verts())
            if verts is not None and verts.size > 0:
                verts = verts.reshape(-1, 3)
                state["pointcloud"] = verts
                state["centroid"] = verts.mean(axis=0)
                state["n_points"] = int(len(verts))
        except Exception:
            pass

    if hasattr(ent, "get_pos"):
        try:
            pos = to_numpy(ent.get_pos()).reshape(-1)
            state["centroid"] = pos[:3]
        except Exception:
            pass

    if hasattr(ent, "get_quat"):
        try:
            quat = to_numpy(ent.get_quat()).reshape(-1)
            state["quat"] = quat[:4]
        except Exception:
            pass

    if hasattr(ent, "get_vel"):
        try:
            vel = to_numpy(ent.get_vel()).reshape(-1)
            state["vel"] = vel[:3]
        except Exception:
            pass

    if hasattr(ent, "get_ang"):
        try:
            ang = to_numpy(ent.get_ang()).reshape(-1)
            state["ang"] = ang[:3]
        except Exception:
            pass

    return state


def save_depth_vis(depth: np.ndarray, out_path: Path) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if np.any(valid):
        d = depth[valid]
        lo = float(np.percentile(d, 5))
        hi = float(np.percentile(d, 95))
        hi = max(hi, lo + 1e-5)
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        gray = (255.0 * (1.0 - norm)).astype(np.uint8)
        vis[..., 0] = gray
        vis[..., 1] = gray
        vis[..., 2] = gray
    imageio.imwrite(out_path, vis)


def prepare_output_dirs(out_dir: Path) -> None:
    for sub in [
        "rgb",
        "depth",
        "depth_vis",
        "segmentation",
        "normal",
        "pointcloud",
        "object_pointcloud",
        "trajectories",
        "camera",
        "video",
    ]:
        ensure_dir(out_dir / sub)


def compute_preview_stride_and_fps(dt: float) -> Tuple[int, int]:
    dt = float(max(dt, 1e-6))
    stride = max(1, int(round(1.0 / (PREVIEW_FPS * dt))))
    return stride, int(PREVIEW_FPS)


def export_scene(scene_cfg: Dict[str, Any], dataset_root: Path) -> Dict[str, Any]:
    out_dir = dataset_root / "train" / scene_cfg["scene_id"]
    prepare_output_dirs(out_dir)
    (out_dir / "scene_input.json").write_text(json.dumps(scene_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    scene, cam, entities, container_entities, state_specs, runtime_records, penetration_lift_records = build_scene(scene_cfg)
    del entities

    try:
        np.save(out_dir / "camera" / "intrinsics.npy", to_numpy(cam.intrinsics))
    except Exception:
        pass
    try:
        np.save(out_dir / "camera" / "extrinsics.npy", to_numpy(cam.extrinsics))
    except Exception:
        pass

    traj_path = out_dir / "trajectories" / "objects_world.csv"
    frame_index_path = out_dir / "trajectories" / "frame_index.csv"

    preview_frames = []
    collision_detected = False
    num_steps = int(scene_cfg["sim_options"]["num_steps"])
    preview_stride, preview_fps = compute_preview_stride_and_fps(float(scene_cfg["sim_options"]["dt"]))

    with open(traj_path, "w", newline="", encoding="utf-8") as traj_csv, open(frame_index_path, "w", newline="", encoding="utf-8") as frame_csv:
        traj_writer = csv.writer(traj_csv)
        traj_writer.writerow([
            "frame", "object_id", "solver",
            "cx", "cy", "cz",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
            "n_points",
        ])

        frame_writer = csv.writer(frame_csv)
        frame_writer.writerow([
            "frame", "rgb_path", "depth_path", "depth_vis_path",
            "seg_path", "normal_path", "pointcloud_path",
        ])

        for t in range(num_steps):
            for rec in container_entities.values():
                _enforce_entity_static(rec["entity"], rec["anchor_pos"])
            scene.step()
            rgb, depth, seg, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

            rgb_path = out_dir / "rgb" / f"{t:06d}.png"
            depth_path = out_dir / "depth" / f"{t:06d}.npy"
            depth_vis_path = out_dir / "depth_vis" / f"{t:06d}.png"
            seg_path = out_dir / "segmentation" / f"{t:06d}.npy"
            normal_path = out_dir / "normal" / f"{t:06d}.npy"

            imageio.imwrite(rgb_path, rgb)
            np.save(depth_path, depth)
            save_depth_vis(depth, depth_vis_path)
            np.save(seg_path, seg)
            np.save(normal_path, normal)

            pc_name = ""
            if (t % CAMERA_PC_STRIDE) == 0:
                try:
                    pc, mask = cam.render_pointcloud(world_frame=True)
                    pc_path = out_dir / "pointcloud" / f"{t:06d}.npz"
                    np.savez_compressed(pc_path, xyz=pc, mask=mask)
                    pc_name = pc_path.name
                except Exception:
                    pc_name = ""

            frame_writer.writerow([
                t,
                rgb_path.name,
                depth_path.name,
                depth_vis_path.name,
                seg_path.name,
                normal_path.name,
                pc_name,
            ])

            if t % preview_stride == 0:
                preview_frames.append(rgb)

            for spec in state_specs:
                ent = spec["entity"]
                if spec["solver"] == "Rigid" and hasattr(ent, "detect_collision"):
                    try:
                        col = ent.detect_collision()
                        col_arr = np.asarray(col)
                        if col_arr.size > 0 and bool(np.any(col_arr)):
                            collision_detected = True
                    except Exception:
                        pass

                state = export_entity_state(ent, spec)
                c = state["centroid"] if state["centroid"] is not None else [np.nan, np.nan, np.nan]
                q = state["quat"] if state["quat"] is not None else [np.nan] * 4
                v = state["vel"] if state["vel"] is not None else [np.nan] * 3
                w = state["ang"] if state["ang"] is not None else [np.nan] * 3

                traj_writer.writerow([
                    t, state["object_id"], state["solver"],
                    float(c[0]), float(c[1]), float(c[2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                    float(v[0]), float(v[1]), float(v[2]),
                    float(w[0]), float(w[1]), float(w[2]),
                    int(state["n_points"]),
                ])

                if (t % OBJECT_PC_STRIDE) == 0 and state["pointcloud"] is not None:
                    xyz = safe_subsample_points(state["pointcloud"], max_points=MAX_OBJECT_PC)
                    np.savez_compressed(
                        out_dir / "object_pointcloud" / f"{t:06d}_{str(state['object_id']).replace('/', '_')}.npz",
                        xyz=xyz,
                        solver=state["solver"],
                        object_id=state["object_id"],
                    )

    if preview_frames:
        imageio.mimsave(out_dir / "video" / "preview.mp4", preview_frames, fps=preview_fps)

    scene_metadata = {
        "scene_id": scene_cfg["scene_id"],
        "seed": scene_cfg["seed"],
        "family": scene_cfg["family"],
        "pattern": scene_cfg["pattern"],
        "num_objects": len(scene_cfg["objects"]),
        "num_soft_parts": int(sum(obj.get("soft_part_count", 0) for obj in scene_cfg["objects"])),
        "sim_steps": num_steps,
        "dt": scene_cfg["sim_options"]["dt"],
        "substeps": scene_cfg["sim_options"]["substeps"],
        "preview_stride": preview_stride,
        "preview_fps": preview_fps,
        "collision_detected": collision_detected,
        "container": scene_cfg["container"],
        "objects": [
            {
                "scene_object_id": obj["scene_object_id"],
                "physx_object_id": obj["physx_object_id"],
                "object_name": obj["object_name"],
                "category": obj["category"],
                "asset_id": obj["asset_id"],
                "motion_type": obj["motion_type"],
                "bbox_extents": obj["geom"]["bbox_extents"],
                "scale": obj["geom"]["scale"],
                "material": obj["material"],
                "rigid_part_count": obj["rigid_part_count"],
                "soft_part_count": obj["soft_part_count"],
                "prepared_metadata_path": obj["prepared_metadata_path"],
                "runtime_application": next(
                    (
                        rec["runtime_application"]
                        for rec in runtime_records
                        if rec["scene_object_id"] == obj["scene_object_id"]
                    ),
                    None,
                ),
            }
            for obj in scene_cfg["objects"]
        ],
        "notes": [
            "Rigid articulation geometry comes from the strict per-part URDF exported by physxnet_articulation_demo.py.",
            "Rigid per-part mass/inertia/friction is preserved in the exported URDF and reapplied to Genesis links and joints after build when setters are available.",
            "Soft parts reuse exact density / Young's modulus / Poisson ratio from PhysXNet metadata whenever a compatible Genesis solver is available.",
            "Object and container scale are sampled jointly to keep scene size physically reasonable.",
        ],
        "initial_floor_penetration_lift": penetration_lift_records,
        "status": "ok",
    }
    (out_dir / "scene_metadata.json").write_text(json.dumps(scene_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        scene.destroy()
    except Exception:
        pass
    return scene_metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Genesis dataset from PhysXNet articulated objects.")
    parser.add_argument("--physx_root", type=str, required=True, help="PhysXNet root, e.g. /data/.../PhysXNet")
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--dataset_root", type=str, required=True, help="Output dataset root")
    parser.add_argument("--prepared_asset_root", type=str, default=None, help="Cache directory for converted per-object articulation assets")
    parser.add_argument("--n_scenes", type=int, default=10)
    parser.add_argument("--start_scene_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--voxel_pitch", type=float, default=0.025)
    parser.add_argument("--object_scale_mult", type=float, default=1.0)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--object_ids", type=str, nargs="*", default=None, help="Optional explicit PhysXNet object ids")
    parser.add_argument("--force_rebuild_assets", action="store_true")
    parser.add_argument("--solver_family_override", type=str, default=None, help="Override solver family for all objects")
    
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    dataset_root = Path(args.dataset_root)
    prepared_asset_root = Path(args.prepared_asset_root) if args.prepared_asset_root else dataset_root / "_prepared_assets"
    ensure_dir(dataset_root)
    ensure_dir(prepared_asset_root)

    asset_bank = build_asset_bank(
        physx_root=Path(args.physx_root),
        version=args.version,
        prepared_asset_root=prepared_asset_root,
        manifest_path=dataset_root / "asset_manifest.json",
        voxel_pitch=float(args.voxel_pitch),
        object_scale_mult=float(args.object_scale_mult),
        object_ids=args.object_ids,
        max_objects=int(args.max_objects),
        force_rebuild_assets=bool(args.force_rebuild_assets),
        solver_family_override=args.solver_family_override,
    )
    if not asset_bank:
        raise RuntimeError("No usable articulation assets were prepared.")

    scene_metas = []
    failed_scenes = []
    for local_idx in range(int(args.n_scenes)):
        scene_idx = int(args.start_scene_idx) + local_idx
        seed = int(args.seed) + scene_idx
        scene_cfg = sample_scene(scene_idx, asset_bank, seed=seed)
        print(f"[INFO] exporting {scene_cfg['scene_id']} | pattern={scene_cfg['pattern']} | n_objects={len(scene_cfg['objects'])}")
        try:
            scene_metas.append(export_scene(scene_cfg, dataset_root))
        except Exception as e:
            failed_scenes.append(
                {
                    "scene_id": scene_cfg["scene_id"],
                    "seed": seed,
                    "pattern": scene_cfg["pattern"],
                    "error": str(e),
                }
            )
            scene_dir = dataset_root / "train" / scene_cfg["scene_id"]
            if scene_dir.exists():
                shutil.rmtree(scene_dir, ignore_errors=True)
            print(f"[WARN] skip failed scene {scene_cfg['scene_id']}: {e}")

    dataset_manifest = {
        "dataset_name": "physxnet_articulation_dataset",
        "physx_root": str(args.physx_root),
        "version": args.version,
        "dataset_root": str(dataset_root),
        "prepared_asset_root": str(prepared_asset_root),
        "n_scenes": len(scene_metas),
        "start_scene_idx": int(args.start_scene_idx),
        "seed": int(args.seed),
        "image_size": [IMG_W, IMG_H],
        "target_longest_size_range_m": list(TARGET_LONGEST_SIZE_RANGE),
        "asset_manifest": str(dataset_root / "asset_manifest.json"),
        "scenes": scene_metas,
        "failed_scenes": failed_scenes,
    }
    (dataset_root / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()



'''
physx-net数据集仿真

CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/dataset_3_mpm_genesis.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --dataset_root /data/gaoya/AAA_test_video/Dataset_test/genesis_mpm \
  --n_scenes 10 \
  --max_objects 5 \
  --force_rebuild_assets \
  --solver_family_override "mpm"



  




python /home/gaoya/Code_Video/Code_data/1_localshow.py \
  --root /data/gaoya/AAA_test_video/Dataset_test/genesis_mpm \
  --host 0.0.0.0 \
  --port 8002




'''
