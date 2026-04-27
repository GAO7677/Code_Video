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
import trimesh

from physxnet_articulation_demo import (
    _configure_genesis_rigid_entity_from_metadata,
    _default_entity_rigid_material,
    _make_genesis_rigid_material,
    prepare_physxnet_object,
)


IMG_W, IMG_H = 960, 720
MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 4
CAMERA_PC_STRIDE = 2

TARGET_LONGEST_SIZE_RANGE = (0.18, 0.42)
STATIC_REST_PROB = 0.38
PREVIEW_FPS = 30
DATASET_OBJECT_COUNTS = [1, 2, 3, 4, 5, 10]

TOP_DROP_Z_RANGE = (1.00, 1.55)
TOP_TOSS_Z_RANGE = (0.95, 1.45)
FRONT_SLIDE_Z_RANGE = (0.16, 0.34)
SIDE_THROW_Z_RANGE = (0.75, 1.25)
DIAGONAL_ENTRY_Z_RANGE = (0.95, 1.60)

STRIKE_SPEED_RANGE = (0.80, 1.35)
TOP_DROP_ANGVEL = 1.6
TOP_TOSS_ANGVEL = 2.2
FRONT_SLIDE_ANGVEL = 1.8
DIAGONAL_THROW_ANGVEL = 2.6
SIDE_THROW_ANGVEL = 2.2

FRONT_SLIDE_VY_RANGE = (1.00, 1.75)
FRONT_SLIDE_VX_RANGE = (-0.18, 0.18)
FRONT_SLIDE_VZ_RANGE = (-0.05, 0.18)
DIAGONAL_THROW_VX_RANGE = (0.95, 1.65)
DIAGONAL_THROW_VY_RANGE = (1.00, 1.85)
DIAGONAL_THROW_VZ_RANGE = (-0.12, 0.42)
SIDE_THROW_VX_RANGE = (0.95, 1.40)
SIDE_THROW_VY_RANGE = (0.08, 0.26)
SIDE_THROW_VZ_RANGE = (0.45, 0.90)

REST_CONTACT_MARGIN = 0.02
SLIDE_CONTACT_MARGIN_RANGE = (0.015, 0.035)

DEFAULT_SOPHY_ROOT = Path("/data/gaoya/dataset/SOPHY_data")
DEFAULT_SOPHY_CATEGORIES = ("bag", "teddy_bear")

MPM_MOTION_CATEGORY_SPECS = [
    {"name": "top_drop_only", "label_zh": "下坠", "motion_modes": ["top_drop"], "scene_builder": "uniform_dynamic"},
    {"name": "top_toss_only", "label_zh": "上方抛掷", "motion_modes": ["top_toss"], "scene_builder": "uniform_dynamic"},
    {"name": "front_slide_only", "label_zh": "前向滑入", "motion_modes": ["front_slide_in"], "scene_builder": "uniform_dynamic"},
    {"name": "diagonal_left_only", "label_zh": "左前对角抛入", "motion_modes": ["diagonal_corner_left"], "scene_builder": "uniform_dynamic"},
    {"name": "diagonal_right_only", "label_zh": "右前对角抛入", "motion_modes": ["diagonal_corner_right"], "scene_builder": "uniform_dynamic"},
    {"name": "side_throw_left_only", "label_zh": "左侧抛入", "motion_modes": ["side_throw_left"], "scene_builder": "uniform_dynamic"},
    {"name": "side_throw_right_only", "label_zh": "右侧抛入", "motion_modes": ["side_throw_right"], "scene_builder": "uniform_dynamic"},
    {"name": "drop_toss_mix", "label_zh": "下坠与抛掷混合", "motion_modes": ["top_drop", "top_toss"], "scene_builder": "uniform_dynamic"},
    {"name": "front_diagonal_mix", "label_zh": "滑入与对角入场混合", "motion_modes": ["front_slide_in", "diagonal_corner_left", "diagonal_corner_right"], "scene_builder": "uniform_dynamic"},
    {
        "name": "omni_dynamic_mix",
        "label_zh": "多运动混合",
        "motion_modes": ["top_drop", "top_toss", "front_slide_in", "diagonal_corner_left", "diagonal_corner_right", "side_throw_left", "side_throw_right"],
        "scene_builder": "uniform_dynamic",
    },
    {
        "name": "ground_static_plus_dynamic",
        "label_zh": "地面静止加动态混合",
        "motion_modes": ["top_drop", "top_toss", "front_slide_in", "diagonal_corner_left", "diagonal_corner_right", "side_throw_left", "side_throw_right"],
        "scene_builder": "ground_static_plus_dynamic",
    },
]


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


def _parse_obj_bbox_extents(obj_path: Path) -> np.ndarray:
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
            mins = np.minimum(mins, xyz)
            maxs = np.maximum(maxs, xyz)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        raise ValueError(f"failed to read vertices from {obj_path}")
    return np.maximum(maxs - mins, 1e-6)


def _write_obj_mesh(out_path: Path, vertices: List[List[float]], faces: List[List[int]]) -> None:
    lines = []
    for v in vertices:
        lines.append(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
    for face in faces:
        lines.append("f " + " ".join(str(int(i)) for i in face) + "\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def _load_point_cloud_vertices(ply_path: Path) -> np.ndarray:
    obj = trimesh.load(ply_path)
    if not hasattr(obj, "vertices"):
        raise ValueError(f"{ply_path} does not contain vertices.")
    xyz = np.asarray(obj.vertices, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError(f"{ply_path} has invalid point cloud shape {xyz.shape}.")
    return xyz


def _bbox_stats_from_points(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    bbox_min = np.min(xyz, axis=0)
    bbox_max = np.max(xyz, axis=0)
    bbox_extents = np.maximum(bbox_max - bbox_min, 1e-6)
    return bbox_min, bbox_max, bbox_extents


def _euler_xyz_deg_to_rotmat(euler_deg: List[float] | Tuple[float, float, float]) -> np.ndarray:
    rx, ry, rz = [math.radians(float(v)) for v in euler_deg]
    T = trimesh.transformations.euler_matrix(rx, ry, rz, axes="sxyz")
    return np.asarray(T[:3, :3], dtype=np.float32)


def _transform_local_points_to_world(
    xyz_local: np.ndarray,
    scale: float,
    pos: Tuple[float, float, float],
    euler_deg: Tuple[float, float, float],
) -> np.ndarray:
    pts = np.asarray(xyz_local, dtype=np.float32).reshape(-1, 3) * float(scale)
    R = _euler_xyz_deg_to_rotmat(euler_deg)
    t = np.asarray(pos, dtype=np.float32).reshape(1, 3)
    return pts @ R.T + t


def _extract_obj_submeshes_by_material(obj_path: Path, out_dir: Path) -> Dict[str, Path]:
    vertices: List[List[float]] = []
    faces_by_material: Dict[str, List[List[int]]] = {}
    current_material = "material_0"

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue

            if line.startswith("usemtl "):
                current_material = line.strip().split(maxsplit=1)[1]
                faces_by_material.setdefault(current_material, [])
                continue

            if not line.startswith("f "):
                continue

            tokens = line.strip().split()[1:]
            face_idx: List[int] = []
            for tok in tokens:
                idx_str = tok.split("/")[0]
                if not idx_str:
                    continue
                idx = int(idx_str)
                if idx < 0:
                    idx = len(vertices) + idx + 1
                face_idx.append(idx)

            if len(face_idx) < 3:
                continue

            tri_faces = []
            if len(face_idx) == 3:
                tri_faces.append(face_idx)
            else:
                for j in range(1, len(face_idx) - 1):
                    tri_faces.append([face_idx[0], face_idx[j], face_idx[j + 1]])
            faces_by_material.setdefault(current_material, []).extend(tri_faces)

    ensure_dir(out_dir)
    out_paths: Dict[str, Path] = {}
    for material_name, faces in faces_by_material.items():
        used = sorted({idx for face in faces for idx in face})
        if not used:
            continue
        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(used, start=1)}
        sub_vertices = [vertices[old_idx - 1] for old_idx in used]
        sub_faces = [[remap[idx] for idx in face] for face in faces]
        out_path = out_dir / f"{material_name}.obj"
        _write_obj_mesh(out_path, sub_vertices, sub_faces)
        out_paths[material_name] = out_path
    return out_paths


def _ordered_sophy_material_items(mat_params: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    return [(str(name), dict(cfg)) for name, cfg in mat_params.items()]


def _sophy_material_model_from_params(cfg: Dict[str, Any]) -> str:
    plasticity = str(cfg.get("plasticity", "")).lower()
    if plasticity and plasticity != "none":
        return "elastoplastic"
    if cfg.get("sigma_y") is not None:
        return "elastoplastic"
    return "elastic"


def _build_sophy_soft_parts_from_sampled_points(
    sampled_xyz: np.ndarray,
    point_mat_labels: np.ndarray,
    point_part_labels: np.ndarray,
    material_groups: Dict[int, List[Dict[str, Any]]],
    particles_dir: Path,
) -> List[Dict[str, Any]]:
    ensure_dir(particles_dir)
    soft_parts: List[Dict[str, Any]] = []
    unique_mat_ids = sorted(int(v) for v in np.unique(point_mat_labels))
    for part_id, mat_id in enumerate(unique_mat_ids):
        mask = np.asarray(point_mat_labels == mat_id)
        if not np.any(mask):
            continue
        group = material_groups.get(mat_id, [])
        if not group:
            continue
        cfg = dict(group[0]["cfg"])
        part_xyz = np.asarray(sampled_xyz[mask], dtype=np.float32)
        part_point_labels = np.asarray(point_part_labels[mask], dtype=np.int64)
        part_names = [str(item["material_name"]) for item in group]
        mesh_paths = [str(item["submesh_path"]) for item in group if item.get("submesh_path")]
        particles_path = particles_dir / f"mat_{mat_id:03d}.npz"
        np.savez_compressed(
            particles_path,
            xyz=part_xyz,
            point_mat_labels=np.full((len(part_xyz),), mat_id, dtype=np.int32),
            point_part_labels=part_point_labels.astype(np.int32, copy=False),
        )
        soft_parts.append(
            {
                "part_id": int(part_id),
                "part_name": "__".join(part_names),
                "associated_material_names": part_names,
                "material_id": int(mat_id),
                "material_slot_candidates": [str(item["material_slot"]) for item in group],
                "mesh_path": mesh_paths[0] if mesh_paths else None,
                "mesh_path_candidates": mesh_paths,
                "particles_path": str(particles_path),
                "particle_source": "sampled_points",
                "n_particles": int(len(part_xyz)),
                "point_part_labels": sorted(int(v) for v in np.unique(part_point_labels)),
                "material_model": _sophy_material_model_from_params(cfg),
                "density_kgm3": cfg.get("rho"),
                "youngs_pa": cfg.get("E"),
                "poisson": cfg.get("nu"),
                "sigma_y_pa": cfg.get("sigma_y"),
                "elasticity": cfg.get("elasticity"),
                "plasticity": cfg.get("plasticity"),
                "material_parameters": cfg,
            }
        )
    return soft_parts


def _build_sophy_soft_parts_from_vecset(
    vecset: Dict[str, np.ndarray],
    material_groups: Dict[int, List[Dict[str, Any]]],
    particles_dir: Path,
) -> List[Dict[str, Any]]:
    ensure_dir(particles_dir)
    vols = np.asarray(vecset["vols"], dtype=np.float32)
    mat_vols = np.asarray(vecset["mat_vols"])
    part_vols = np.asarray(vecset["part_vols"]) if "part_vols" in vecset else np.full((len(vols),), -1, dtype=np.int32)
    soft_parts: List[Dict[str, Any]] = []
    unique_mat_ids = sorted(int(v) for v in np.unique(mat_vols))
    for part_id, mat_id in enumerate(unique_mat_ids):
        mask = np.asarray(mat_vols == mat_id)
        if not np.any(mask):
            continue
        group = material_groups.get(mat_id, [])
        if not group:
            continue
        cfg = dict(group[0]["cfg"])
        part_xyz = np.asarray(vols[mask], dtype=np.float32)
        part_point_labels = np.asarray(part_vols[mask], dtype=np.int64)
        part_names = [str(item["material_name"]) for item in group]
        mesh_paths = [str(item["submesh_path"]) for item in group if item.get("submesh_path")]
        particles_path = particles_dir / f"mat_{mat_id:03d}.npz"
        np.savez_compressed(
            particles_path,
            xyz=part_xyz,
            point_mat_labels=np.full((len(part_xyz),), mat_id, dtype=np.int32),
            point_part_labels=part_point_labels.astype(np.int32, copy=False),
        )
        soft_parts.append(
            {
                "part_id": int(part_id),
                "part_name": "__".join(part_names),
                "associated_material_names": part_names,
                "material_id": int(mat_id),
                "material_slot_candidates": [str(item["material_slot"]) for item in group],
                "mesh_path": mesh_paths[0] if mesh_paths else None,
                "mesh_path_candidates": mesh_paths,
                "particles_path": str(particles_path),
                "particle_source": "vecset_vols",
                "n_particles": int(len(part_xyz)),
                "point_part_labels": sorted(int(v) for v in np.unique(part_point_labels)),
                "material_model": _sophy_material_model_from_params(cfg),
                "density_kgm3": cfg.get("rho"),
                "youngs_pa": cfg.get("E"),
                "poisson": cfg.get("nu"),
                "sigma_y_pa": cfg.get("sigma_y"),
                "elasticity": cfg.get("elasticity"),
                "plasticity": cfg.get("plasticity"),
                "material_parameters": cfg,
            }
        )
    return soft_parts


def prepare_sophy_asset(
    sophy_root: Path,
    category: str,
    split: str,
    instance_id: str,
    prepared_asset_root: Path,
    force_rebuild: bool,
) -> Dict[str, Any]:
    instance_dir = prepared_asset_root / f"sophy_{category}_{instance_id}"
    meta_path = instance_dir / "meta.json"
    if force_rebuild or not meta_path.exists():
        sim_dir = sophy_root / category / "simulation_data" / split / category / instance_id
        data_dir = sophy_root / category / "data" / split / category / instance_id
        mesh_path = sim_dir / "material.obj"
        mat_params_path = sim_dir / "mat_params_new_v3.4.json"
        sampled_points_path = sim_dir / "sampled_points.ply"
        sampled_points_info_path = sim_dir / "sampled_points_info.npz"
        vecset_path = sim_dir / "vecset_v3.1.npz"
        data_static_path = data_dir / "data_static.json"
        if not mesh_path.exists():
            raise FileNotFoundError(mesh_path)
        if not mat_params_path.exists():
            raise FileNotFoundError(mat_params_path)
        if not data_static_path.exists():
            raise FileNotFoundError(data_static_path)

        ensure_dir(instance_dir)
        submesh_dir = instance_dir / "submeshes"
        particles_dir = instance_dir / "particles"
        submeshes = _extract_obj_submeshes_by_material(mesh_path, submesh_dir)
        mat_params = json.loads(mat_params_path.read_text(encoding="utf-8"))
        material_items = _ordered_sophy_material_items(mat_params)
        ordered_submeshes = sorted(
            submeshes.items(),
            key=lambda kv: int(kv[0].split("_")[-1]) if kv[0].split("_")[-1].isdigit() else kv[0],
        )
        material_groups: Dict[int, List[Dict[str, Any]]] = {}
        for (material_slot, submesh_path), (material_name, cfg) in zip(ordered_submeshes, material_items):
            mat_id = cfg.get("mat_id")
            if mat_id is None:
                continue
            material_groups.setdefault(int(mat_id), []).append(
                {
                    "material_name": str(material_name),
                    "cfg": dict(cfg),
                    "material_slot": str(material_slot),
                    "submesh_path": str(submesh_path),
                }
            )

        sampled_xyz = None
        point_mat_labels = None
        point_part_labels = None
        particle_source = None
        if sampled_points_path.exists() and sampled_points_info_path.exists():
            sampled_xyz = _load_point_cloud_vertices(sampled_points_path)
            sampled_info = np.load(sampled_points_info_path, allow_pickle=True)
            point_mat_labels = np.asarray(sampled_info["point_mat_labels"])
            point_part_labels = np.asarray(sampled_info["point_part_labels"])
            particle_source = "sampled_points"

        vecset = None
        if vecset_path.exists():
            vecset_npz = np.load(vecset_path, allow_pickle=True)
            vecset = {key: vecset_npz[key] for key in vecset_npz.files}
            if particle_source is None:
                particle_source = "vecset_vols"

        if sampled_xyz is not None and point_mat_labels is not None and point_part_labels is not None:
            soft_parts = _build_sophy_soft_parts_from_sampled_points(
                sampled_xyz=sampled_xyz,
                point_mat_labels=point_mat_labels,
                point_part_labels=point_part_labels,
                material_groups=material_groups,
                particles_dir=particles_dir,
            )
            bbox_min, bbox_max, bbox_extents = _bbox_stats_from_points(sampled_xyz)
        elif vecset is not None:
            soft_parts = _build_sophy_soft_parts_from_vecset(
                vecset=vecset,
                material_groups=material_groups,
                particles_dir=particles_dir,
            )
            bbox_min, bbox_max, bbox_extents = _bbox_stats_from_points(np.asarray(vecset["vols"], dtype=np.float32))
        else:
            raise FileNotFoundError(
                f"SOPHY asset {category}/{split}/{instance_id} is missing both sampled_points(.ply/.npz) and vecset_v3.1.npz."
            )

        metadata = {
            "dataset_name": "sophy",
            "category": category,
            "split": split,
            "instance_id": instance_id,
            "mesh_path": str(mesh_path),
            "data_static_path": str(data_static_path),
            "mat_params_path": str(mat_params_path),
            "sampled_points_path": str(sampled_points_path) if sampled_points_path.exists() else None,
            "sampled_points_info_path": str(sampled_points_info_path) if sampled_points_info_path.exists() else None,
            "vecset_path": str(vecset_path) if vecset_path.exists() else None,
            "particle_source": particle_source,
            "bbox_min": bbox_min.tolist(),
            "bbox_max": bbox_max.tolist(),
            "bbox_extents": bbox_extents.tolist(),
            "soft_parts": soft_parts,
            "raw_material_parameters": mat_params,
            "raw_material_items": [{"name": str(name), "cfg": dict(cfg)} for name, cfg in material_items],
            "available_material_ids": sorted(int(k) for k in material_groups.keys()),
            "sampled_point_count": int(len(sampled_xyz)) if sampled_xyz is not None else 0,
            "vecset_stats": (
                {
                    "loc": np.asarray(vecset.get("loc"), dtype=np.float32).tolist() if vecset is not None and "loc" in vecset else None,
                    "scale": float(vecset.get("scale")) if vecset is not None and "scale" in vecset else None,
                    "num_vols": int(len(vecset["vols"])) if vecset is not None and "vols" in vecset else 0,
                    "num_surfs": int(len(vecset["surfs"])) if vecset is not None and "surfs" in vecset else 0,
                    "num_nears": int(len(vecset["nears"])) if vecset is not None and "nears" in vecset else 0,
                }
                if vecset is not None
                else None
            ),
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    bbox_extents = np.asarray(metadata["bbox_extents"], dtype=np.float64)
    bbox_min = np.asarray(metadata.get("bbox_min", (-0.5 * bbox_extents)), dtype=np.float64)
    bbox_max = np.asarray(metadata.get("bbox_max", (0.5 * bbox_extents)), dtype=np.float64)
    return {
        "asset_id": f"sophy__{category}__{instance_id}",
        "dataset_name": "sophy",
        "object_id": str(instance_id),
        "object_name": str(instance_id),
        "category": str(category),
        "split": str(metadata["split"]),
        "asset_dir": str(instance_dir),
        "metadata_path": str(meta_path),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_extents": bbox_extents.tolist(),
        "grounding_offset_z": float(-bbox_min[2]),
        "material_override": None,
        "rigid_part_count": 0,
        "soft_part_count": int(len(metadata.get("soft_parts", []))),
        "metadata": metadata,
    }


def build_sophy_asset_bank(
    sophy_root: Path,
    prepared_asset_root: Path,
    force_rebuild_assets: bool,
    max_assets: int,
    categories: Tuple[str, ...] = DEFAULT_SOPHY_CATEGORIES,
) -> List[Dict[str, Any]]:
    bank: List[Dict[str, Any]] = []
    for category in categories:
        data_root = sophy_root / category / "data"
        if not data_root.exists():
            continue
        instance_paths = sorted(data_root.glob(f"*/{category}/*/data_static.json"))
        for data_static_path in instance_paths:
            split = data_static_path.parts[-4]
            instance_id = data_static_path.parts[-2]
            try:
                bank.append(
                    prepare_sophy_asset(
                        sophy_root=sophy_root,
                        category=category,
                        split=split,
                        instance_id=instance_id,
                        prepared_asset_root=prepared_asset_root,
                        force_rebuild=force_rebuild_assets,
                    )
                )
            except Exception as e:
                print(f"[WARN] skip SOPHY asset {category}/{split}/{instance_id}: {e}")
            if max_assets not in (None, 0) and len(bank) >= int(max_assets):
                return bank
    return bank


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
    """
    准备 PhysXNet 铰接资产（刚体+软体混合对象）
    
    从 PhysXNet 数据集中加载指定的铰接对象，进行必要的预处理（如果需要），
    并返回包含资产元数据、几何信息和材料属性的字典。
    
    Args:
        physx_root (Path): PhysXNet 数据集根目录路径
        version (str): 资产版本标识符（如 "version_1"）
        object_id (str): 对象的唯一标识符（如 "obj_001"）
        prepared_asset_root (Path): 预处理资产的输出根目录
        voxel_pitch (float): 体素网格的间距（单位：米），用于 MPM 软体部分的离散化
        object_scale_mult (float): 对象缩放倍数，用于调整对象大小
        force_rebuild (bool): 是否强制重新构建资产，忽略缓存
    
    Returns:
        Dict[str, Any]: 包含以下键的资产字典：
            - asset_id (str): 资产的唯一标识符
            - dataset_name (str): 数据集名称（"physxnet_articulation"）
            - object_id (str): 对象 ID
            - object_name (str): 对象的人类可读名称
            - category (str): 对象类别
            - asset_dir (str): 资产目录路径
            - metadata_path (str): 元数据 JSON 文件路径
            - urdf_path (str): URDF 文件路径（刚体部分）
            - bbox_min (list): 边界框最小坐标 [x, y, z]
            - bbox_max (list): 边界框最大坐标 [x, y, z]
            - bbox_extents (list): 边界框尺寸 [dx, dy, dz]
            - grounding_offset_z (float): Z 轴接地偏移量
            - material_override (dict): 材料属性覆盖配置
            - rigid_part_count (int): 刚体部分数量
            - soft_part_count (int): 软体部分数量
            - metadata (dict): 完整的元数据字典
    
    Raises:
        FileNotFoundError: 如果元数据文件不存在且 force_rebuild=False
        RuntimeError: 如果资产准备过程失败
    
    Notes:
        - 如果缓存的元数据存在且 force_rebuild=False，则直接加载缓存
        - 否则调用 prepare_physxnet_object() 进行完整的资产准备
        - 返回的资产字典可直接用于 Genesis 场景构建
    """
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
    sophy_root: Path,
    prepared_asset_root: Path,
    manifest_path: Path,
    voxel_pitch: float,
    object_scale_mult: float,
    object_ids: Optional[List[str]],
    max_objects: int,
    force_rebuild_assets: bool,
    asset_sources: Tuple[str, ...] = ("physx", "sophy"),
    solver_family_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ensure_dir(prepared_asset_root)
    ensure_dir(manifest_path.parent)

    bank: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    if "physx" in asset_sources:
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

    if "sophy" in asset_sources:
        sophy_bank = build_sophy_asset_bank(
            sophy_root=sophy_root,
            prepared_asset_root=prepared_asset_root,
            force_rebuild_assets=force_rebuild_assets,
            max_assets=max_objects,
        )
        bank.extend(sophy_bank)

    manifest = {
        "dataset_name": "genesis_mpm_mixed_dataset",
        "physx_root": str(physx_root),
        "sophy_root": str(sophy_root),
        "version": version,
        "prepared_asset_root": str(prepared_asset_root),
        "n_assets": len(bank),
        "n_failed_assets": len(failed),
        "assets": bank,
        "failed_assets": failed,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] MPM assets: usable={len(bank)} failed={len(failed)}")
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
    forced_mode: Optional[str] = None,
) -> Dict[str, Any]:
    half_x, half_y, half_z = [float(x) * 0.5 for x in scaled_extents]
    floor_top_z = float(container_cfg["center"][2]) + float(container_cfg["floor_thickness"])
    rest_z = floor_top_z + float(grounding_offset_z) + REST_CONTACT_MARGIN
    mode = forced_mode

    def _sample_front_band() -> Tuple[float, float]:
        x = float(np.random.uniform(-max(0.12, container_cfg["half_x"] * 0.70), max(0.12, container_cfg["half_x"] * 0.70)))
        y = -container_cfg["half_y"] + 0.06 + half_y
        return x, y

    def _sample_side_band(side: str, y_center: Optional[float] = None) -> Tuple[float, float]:
        x_mag = max(container_cfg["half_x"] - half_x - 0.05, 0.12)
        x = -x_mag if side == "left" else x_mag
        if y_center is None:
            y = float(np.random.uniform(-container_cfg["half_y"] * 0.35, container_cfg["half_y"] * 0.25))
        else:
            y = float(np.clip(y_center + np.random.uniform(-0.12, 0.12), -container_cfg["half_y"] * 0.45, container_cfg["half_y"] * 0.30))
        return x, y

    if pattern == "strike_static" and mode is None:
        mode = "static_rest" if index_in_scene == 0 else "strike_static_left"

    if mode == "static_rest":
        x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
        return {
            "motion_type": "static_rest",
            "init_pos": [x, y, rest_z],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [0.0, 0.0, 0.0],
            "init_angvel": [0.0, 0.0, 0.0],
        }

    if pattern == "strike_static":
        if index_in_scene == 0 and mode is None:
            x, y = sample_spawn_xy(container_cfg, half_x, half_y, bias_to_back=True)
            return {
                "motion_type": "static_rest",
                "init_pos": [x, y, rest_z],
                "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
                "init_linvel": [0.0, 0.0, 0.0],
                "init_angvel": [0.0, 0.0, 0.0],
            }

        strike_target = np.asarray(target_pos if target_pos is not None else [0.0, 0.20, rest_z], dtype=np.float64)
        side = "left" if mode != "strike_static_right" else "right"
        start_x = -container_cfg["half_x"] + 0.12 + half_x if side == "left" else container_cfg["half_x"] - 0.12 - half_x
        start_y = float(np.clip(strike_target[1] + np.random.uniform(-0.08, 0.08), -0.05, 0.35))
        speed = float(np.random.uniform(*STRIKE_SPEED_RANGE))
        x_vel = speed if side == "left" else -speed
        return {
            "motion_type": "strike_static_left" if side == "left" else "strike_static_right",
            "init_pos": [start_x, start_y, rest_z + 0.005],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [x_vel, float(np.random.uniform(-0.10, 0.10)), 0.0],
            "init_angvel": _random_angvel(FRONT_SLIDE_ANGVEL),
        }

    if mode == "front_slide_in":
        x, y = _sample_front_band()
        return {
            "motion_type": "front_slide_in",
            "init_pos": [x, y, rest_z + float(np.random.uniform(*SLIDE_CONTACT_MARGIN_RANGE))],
            "init_euler": [0.0, 0.0, float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [float(np.random.uniform(*FRONT_SLIDE_VX_RANGE)), float(np.random.uniform(*FRONT_SLIDE_VY_RANGE)), float(np.random.uniform(*FRONT_SLIDE_VZ_RANGE))],
            "init_angvel": _random_angvel(FRONT_SLIDE_ANGVEL),
        }

    if mode in {"diagonal_corner_left", "diagonal_corner_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side)
        z = max(float(np.random.uniform(*DIAGONAL_ENTRY_Z_RANGE)), rest_z + 0.15)
        vx = abs(float(np.random.uniform(*DIAGONAL_THROW_VX_RANGE)))
        if side == "right":
            vx = -vx
        return {
            "motion_type": mode,
            "init_pos": [x, y, z],
            "init_euler": [float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [vx, float(np.random.uniform(*DIAGONAL_THROW_VY_RANGE)), float(np.random.uniform(*DIAGONAL_THROW_VZ_RANGE))],
            "init_angvel": _random_angvel(DIAGONAL_THROW_ANGVEL),
        }

    if mode in {"side_throw_left", "side_throw_right"}:
        side = "left" if mode.endswith("left") else "right"
        x, y = _sample_side_band(side=side)
        z = max(float(np.random.uniform(*SIDE_THROW_Z_RANGE)), rest_z + 0.10)
        vx = abs(float(np.random.uniform(*SIDE_THROW_VX_RANGE)))
        if side == "right":
            vx = -vx
        return {
            "motion_type": mode,
            "init_pos": [x, y, z],
            "init_euler": [float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-0.35, 0.35)), float(np.random.uniform(-math.pi, math.pi))],
            "init_linvel": [vx, float(np.random.uniform(*SIDE_THROW_VY_RANGE)), float(np.random.uniform(*SIDE_THROW_VZ_RANGE))],
            "init_angvel": _random_angvel(SIDE_THROW_ANGVEL),
        }

    static_rest = mode is None and (np.random.rand() < STATIC_REST_PROB)
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

    if mode == "top_toss" or (mode is None and np.random.rand() < 0.45):
        z = float(np.random.uniform(*TOP_TOSS_Z_RANGE))
        mode = "top_toss"
        linvel = [
            float(np.random.uniform(-0.25, 0.25)),
            float(np.random.uniform(-0.20, 0.20)),
            float(np.random.uniform(-0.65, -0.05)),
        ]

    return {
        "motion_type": str(mode),
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


def get_sim_steps_for_motion_modes(motion_modes: List[str]) -> int:
    motion_modes = set(motion_modes)
    if motion_modes & {"side_throw_left", "side_throw_right"}:
        return 2400
    if motion_modes & {"diagonal_corner_left", "diagonal_corner_right"}:
        return 2200
    if motion_modes & {"top_toss"}:
        return 2000
    return 1800


def compute_num_steps_for_target_seconds(dt: float, target_seconds: Optional[float], fallback_steps: int) -> int:
    if target_seconds is None:
        return int(fallback_steps)
    return max(1, int(round(float(target_seconds) / max(float(dt), 1e-6))))


def build_scene_generation_plan(samples_per_category: int) -> List[Dict[str, Any]]:
    plan = []
    for category_spec in MPM_MOTION_CATEGORY_SPECS:
        if category_spec["scene_builder"] == "uniform_dynamic":
            for object_count in DATASET_OBJECT_COUNTS:
                for sample_idx in range(samples_per_category):
                    plan.append({"category_spec": category_spec, "object_count": int(object_count), "sample_index": int(sample_idx)})
        elif category_spec["scene_builder"] == "ground_static_plus_dynamic":
            for sample_idx in range(samples_per_category):
                plan.append({"category_spec": category_spec, "object_count": None, "sample_index": int(sample_idx)})
    return plan


def sample_scene_cfg(scene_id: int, scene_plan: Dict[str, Any], asset_bank: List[Dict[str, Any]], seed: int, target_seconds: Optional[float] = None) -> Dict[str, Any]:
    set_seed(seed)
    category_spec = scene_plan["category_spec"]
    object_count = scene_plan["object_count"]
    if object_count is None:
        n_objects = random.randint(2, min(4, len(asset_bank)))
    else:
        n_objects = min(int(object_count), len(asset_bank))

    chosen_assets = random.sample(asset_bank, k=max(1, n_objects))
    pattern = "strike_static" if category_spec["name"] == "ground_static_plus_dynamic" else category_spec["name"]

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
    motion_modes_present = []
    num_static_objects = 0
    for idx, (asset, scene_scale, extents) in enumerate(zip(chosen_assets, scene_scales, scaled_extents)):
        forced_mode = None
        if category_spec["scene_builder"] == "uniform_dynamic":
            forced_mode = category_spec["motion_modes"][idx % len(category_spec["motion_modes"])]
        elif category_spec["scene_builder"] == "ground_static_plus_dynamic":
            if idx == 0:
                forced_mode = "static_rest"
            else:
                forced_mode = category_spec["motion_modes"][(idx - 1) % len(category_spec["motion_modes"])]

        motion = sample_motion_for_object(
            pattern=pattern,
            container_cfg=container_cfg,
            scaled_extents=np.asarray(extents, dtype=np.float64),
            grounding_offset_z=float(asset["grounding_offset_z"]) * float(scene_scale),
            target_pos=strike_target,
            index_in_scene=idx,
            forced_mode=forced_mode,
        )
        if idx == 0:
            strike_target = np.asarray(motion["init_pos"], dtype=np.float64)
        motion_modes_present.append(str(motion["motion_type"]))
        if motion["motion_type"] == "static_rest":
            num_static_objects += 1

        obj_rec = {
            "scene_object_id": idx,
            "asset_id": asset["asset_id"],
            "source_object_id": asset["object_id"],
            "object_name": asset["object_name"],
            "category": asset["category"],
            "pattern": pattern,
            "motion_type": motion["motion_type"],
            "init_pos": [float(x) for x in motion["init_pos"]],
            "init_euler": [float(x) for x in motion["init_euler"]],
            "init_linvel": [float(x) for x in motion["init_linvel"]],
            "init_angvel": [float(x) for x in motion["init_angvel"]],
            "prepared_asset_dir": asset["asset_dir"],
            "prepared_metadata_path": asset["metadata_path"],
            "grounding_offset_z": float(asset["grounding_offset_z"]) * float(scene_scale),
            "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
            "scene_scale": float(scene_scale),
            "soft_parts": scale_soft_parts(asset["metadata"].get("soft_parts", []), scene_scale),
            "rigid_part_count": int(asset["rigid_part_count"]),
            "soft_part_count": int(asset["soft_part_count"]),
        }

        if asset["dataset_name"] == "physxnet_articulation":
            obj_rec.update(
                {
                    "solver": "ArticulationRigid",
                    "source_type": "physxnet_articulation",
                    "geom": {
                        "shape": "urdf",
                        "urdf_file": asset["urdf_path"],
                        "scale": float(scene_scale),
                        "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
                        "bound_radius": compute_bound_radius(np.asarray(extents, dtype=np.float64)),
                    },
                    "material": dict(asset["material_override"]),
                }
            )
        else:
            obj_rec.update(
                {
                    "solver": "SoftMPM",
                    "source_type": "sophy",
                    "geom": {
                        "shape": "mesh_collection",
                        "scale": float(scene_scale),
                        "bbox_extents": np.asarray(extents, dtype=np.float64).tolist(),
                        "bound_radius": compute_bound_radius(np.asarray(extents, dtype=np.float64)),
                    },
                    "material": None,
                }
            )

        objects.append(obj_rec)

    num_dynamic_objects = len(objects) - num_static_objects
    if object_count is None:
        count_dir = f"count_mixed_s{num_static_objects}_d{num_dynamic_objects}"
        scene_name_suffix = f"s{num_static_objects}_d{num_dynamic_objects}"
    else:
        count_dir = f"count_{int(object_count):02d}"
        scene_name_suffix = f"n{int(object_count):02d}"
    scene_id_str = f"{category_spec['name']}__{scene_name_suffix}__sample_{int(scene_plan['sample_index']):04d}"
    output_relpath = Path("train") / category_spec["name"] / count_dir / scene_id_str

    dt = 1e-3
    sim_steps = compute_num_steps_for_target_seconds(
        dt=dt,
        target_seconds=target_seconds,
        fallback_steps=get_sim_steps_for_motion_modes(motion_modes_present),
    )

    return {
        "scene_id": scene_id_str,
        "seed": seed,
        "family": "mpm_mixed_assets",
        "mpm_motion_category": category_spec["name"],
        "mpm_motion_label_zh": category_spec.get("label_zh", category_spec["name"]),
        "scene_builder": category_spec["scene_builder"],
        "object_count_bucket": object_count,
        "sample_index_in_bucket": int(scene_plan["sample_index"]),
        "num_static_objects": int(num_static_objects),
        "num_dynamic_objects": int(num_dynamic_objects),
        "motion_modes_present": sorted(set(motion_modes_present)),
        "output_relpath": str(output_relpath),
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
            "dt": dt,
            "substeps": 10,
            "num_steps": sim_steps,
        },
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


def _build_mpm_material_from_record(gs: Any, soft: Dict[str, Any]):
    density = float(soft["density_kgm3"])
    youngs = float(soft["youngs_pa"])
    poisson = float(soft["poisson"])
    material_model = str(soft.get("material_model", "")).lower()
    plasticity = str(soft.get("plasticity", "")).lower()
    sigma_y = soft.get("sigma_y_pa", None)
    use_elastoplastic = (
        material_model == "elastoplastic"
        or (plasticity not in {"", "none"} and plasticity != "elastic")
        or sigma_y is not None
    )
    if use_elastoplastic:
        return gs.materials.MPM.ElastoPlastic(E=youngs, nu=poisson, rho=density)
    return gs.materials.MPM.Elastic(E=youngs, nu=poisson, rho=density)


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

    scene_kwargs["rigid_options"] = gs.options.RigidOptions(
        dt=scene_cfg["sim_options"]["dt"],
        enable_collision=True,
        use_gjk_collision=True,
        batch_dofs_info=True,
        batch_joints_info=True,
        batch_links_info=True,
    )

    if any(obj.get("soft_part_count", 0) > 0 for obj in scene_cfg["objects"]):
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(
            lower_bound=(-2.2, -2.2, -0.2),
            upper_bound=(2.2, 2.2, 2.8),
        )
        if hasattr(gs.options, "CouplerOptions"):
            scene_kwargs["coupler_options"] = gs.options.CouplerOptions(rigid_mpm=True)

    scene = gs.Scene(**scene_kwargs)
    container_entities = add_container(gs, scene, scene_cfg["container"])

    entities = []
    state_specs = []
    runtime_records = []
    soft_particle_records = []

    for obj in scene_cfg["objects"]:
        primary_ent = None
        obj_metadata = json.loads(Path(obj["prepared_metadata_path"]).read_text(encoding="utf-8"))
        euler = tuple(obj["init_euler"])
        pos = tuple(obj["init_pos"])

        if obj["source_type"] == "physxnet_articulation":
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
            ent = scene.add_entity(
                morph=gs.morphs.URDF(**urdf_kwargs),
                material=material,
            )
            primary_ent = ent
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
            if soft.get("density_kgm3") is None or soft.get("youngs_pa") is None or soft.get("poisson") is None:
                runtime_records.append(
                    {
                        "scene_object_id": obj["scene_object_id"],
                        "soft_part_id": soft.get("part_id"),
                        "soft_part_skipped": True,
                        "reason": "missing_dataset_mpm_parameters",
                    }
                )
                continue

            density = float(soft["density_kgm3"])
            scale = float(soft.get("scene_scale", obj["geom"]["scale"]))

            mpm_mat = _build_mpm_material_from_record(gs, soft)
            particles_path = soft.get("particles_path")
            if particles_path:
                particle_npz = np.load(particles_path, allow_pickle=True)
                xyz_local = np.asarray(particle_npz["xyz"], dtype=np.float32)
                if xyz_local.size == 0:
                    runtime_records.append(
                        {
                            "scene_object_id": obj["scene_object_id"],
                            "soft_part_id": soft.get("part_id"),
                            "soft_part_skipped": True,
                            "reason": "empty_dataset_particle_cloud",
                        }
                    )
                    continue
                xyz_world = _transform_local_points_to_world(
                    xyz_local=xyz_local,
                    scale=scale,
                    pos=pos,
                    euler_deg=euler,
                )
                soft_ent = scene.add_entity(
                    material=mpm_mat,
                    morph=gs.morphs.Nowhere(n_particles=int(len(xyz_world))),
                    surface=gs.surfaces.Default(color=(0.86, 0.34, 0.34, 1.0), vis_mode="particle"),
                )
                soft_particle_records.append(
                    {
                        "entity": soft_ent,
                        "world_particles": xyz_world,
                        "linvel": np.asarray(obj["init_linvel"], dtype=np.float32),
                    }
                )
            else:
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
            if primary_ent is None:
                primary_ent = soft_ent
            state_specs.append(
                {
                    "object_id": f"{obj['scene_object_id']}_soft_{soft['part_id']}",
                    "name": f"{obj['object_name']}_soft_{soft['part_id']}",
                    "solver": "mpm_elastoplastic" if str(soft.get("material_model", "")).lower() == "elastoplastic" or soft.get("sigma_y_pa") is not None else "mpm_elastic",
                    "entity": soft_ent,
                }
            )

        entities.append(primary_ent)

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )

    scene.build()

    for rec in soft_particle_records:
        rec["entity"].active = True
        rec["entity"].set_particles_active(True)
        rec["entity"].set_position(rec["world_particles"])
        if np.linalg.norm(rec["linvel"]) > 0:
            rec["entity"].set_velocity(rec["linvel"])

    for rec in container_entities.values():
        _enforce_entity_static(rec["entity"], rec["anchor_pos"])

    floor_top_z = float(scene_cfg["container"]["center"][2]) + float(scene_cfg["container"]["floor_thickness"])
    penetration_lift_records: List[Dict[str, Any]] = []
    for obj, ent in zip(scene_cfg["objects"], entities):
        if ent is None:
            penetration_lift_records.append(
                {
                    "scene_object_id": int(obj["scene_object_id"]),
                    "lift_dz": 0.0,
                }
            )
            continue
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
        if ent is None:
            continue
        apply_initial_motion_to_entity(ent, obj["init_linvel"], obj["init_angvel"])

    for rec in runtime_records:
        if "entity" not in rec or "metadata" not in rec:
            continue
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
        pts = to_numpy(ent.get_particles_pos())
        if pts is not None and pts.size > 0:
            pts = pts.reshape(-1, 3)
            state["pointcloud"] = pts
            state["centroid"] = pts.mean(axis=0)
            state["n_points"] = int(len(pts))
            return state

    if hasattr(ent, "get_verts"):
        verts = to_numpy(ent.get_verts())
        if verts is not None and verts.size > 0:
            verts = verts.reshape(-1, 3)
            state["pointcloud"] = verts
            state["centroid"] = verts.mean(axis=0)
            state["n_points"] = int(len(verts))

    if hasattr(ent, "get_pos"):
        pos = to_numpy(ent.get_pos()).reshape(-1)
        state["centroid"] = pos[:3]

    if hasattr(ent, "get_quat"):
        quat = to_numpy(ent.get_quat()).reshape(-1)
        state["quat"] = quat[:4]

    if hasattr(ent, "get_vel"):
        vel = to_numpy(ent.get_vel()).reshape(-1)
        state["vel"] = vel[:3]

    if hasattr(ent, "get_ang"):
        ang = to_numpy(ent.get_ang()).reshape(-1)
        state["ang"] = ang[:3]

    return state


def save_depth_vis(depth: np.ndarray, out_path: Path) -> None:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if np.any(valid):
        d = np.asarray(depth[valid], dtype=np.float32).reshape(-1)
        if d.size == 0:
            imageio.imwrite(out_path, vis)
            return
        lo = float(np.min(d))
        hi = float(np.max(d))
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


def compute_preview_stride_and_fps(dt: float, num_steps: int, target_seconds: Optional[float]) -> Tuple[int, int]:
    if target_seconds is None:
        dt = float(max(dt, 1e-6))
        stride = max(1, int(round(1.0 / (PREVIEW_FPS * dt))))
        return stride, int(PREVIEW_FPS)
    target_frames = max(1.0, float(target_seconds) * float(PREVIEW_FPS))
    stride = max(1, int(round(float(num_steps) / target_frames)))
    return stride, int(PREVIEW_FPS)


def export_scene(scene_cfg: Dict[str, Any], dataset_root: Path) -> Dict[str, Any]:
    out_dir = dataset_root / scene_cfg.get("output_relpath", str(Path("train") / scene_cfg["scene_id"]))
    prepare_output_dirs(out_dir)
    (out_dir / "scene_input.json").write_text(json.dumps(scene_cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    scene, cam, entities, container_entities, state_specs, runtime_records, penetration_lift_records = build_scene(scene_cfg)
    del entities

    np.save(out_dir / "camera" / "intrinsics.npy", to_numpy(cam.intrinsics))
    np.save(out_dir / "camera" / "extrinsics.npy", to_numpy(cam.extrinsics))

    traj_path = out_dir / "trajectories" / "objects_world.csv"
    frame_index_path = out_dir / "trajectories" / "frame_index.csv"
    object_pc_index_path = out_dir / "trajectories" / "object_pointcloud_index.csv"

    preview_frames = []
    collision_detected = False
    num_steps = int(scene_cfg["sim_options"]["num_steps"])
    preview_stride, preview_fps = compute_preview_stride_and_fps(
        float(scene_cfg["sim_options"]["dt"]),
        num_steps,
        scene_cfg.get("target_seconds"),
    )

    with open(traj_path, "w", newline="", encoding="utf-8") as traj_csv, open(frame_index_path, "w", newline="", encoding="utf-8") as frame_csv, open(object_pc_index_path, "w", newline="", encoding="utf-8") as object_pc_csv:
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
        object_pc_writer = csv.writer(object_pc_csv)
        object_pc_writer.writerow([
            "frame", "object_id", "solver", "pointcloud_path",
            "cx", "cy", "cz",
            "qx", "qy", "qz", "qw",
            "vx", "vy", "vz",
            "wx", "wy", "wz",
            "n_points_raw", "n_points_saved", "coordinate_frame",
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
                pc, mask = cam.render_pointcloud(world_frame=True)
                pc_path = out_dir / "pointcloud" / f"{t:06d}.npz"
                np.savez_compressed(pc_path, xyz=pc, mask=mask)
                pc_name = pc_path.name

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
                    col = ent.detect_collision()
                    col_arr = np.asarray(col)
                    if col_arr.size > 0 and bool(np.any(col_arr)):
                        collision_detected = True

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
                    object_pc_name = f"{t:06d}_obj{str(state['object_id']).replace('/', '_')}.npz"
                    np.savez_compressed(
                        out_dir / "object_pointcloud" / object_pc_name,
                        xyz=xyz,
                        solver=state["solver"],
                        object_id=state["object_id"],
                        frame=int(t),
                        centroid=np.asarray(c, dtype=np.float32),
                        quat=np.asarray(q, dtype=np.float32),
                        vel=np.asarray(v, dtype=np.float32),
                        ang=np.asarray(w, dtype=np.float32),
                        n_points_raw=int(state["n_points"]),
                        n_points_saved=int(len(xyz)),
                        coordinate_frame="world",
                    )
                    object_pc_writer.writerow([
                        t, state["object_id"], state["solver"], object_pc_name,
                        float(c[0]), float(c[1]), float(c[2]),
                        float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                        float(v[0]), float(v[1]), float(v[2]),
                        float(w[0]), float(w[1]), float(w[2]),
                        int(state["n_points"]), int(len(xyz)), "world",
                    ])

    if preview_frames:
        imageio.mimsave(out_dir / "video" / "preview.mp4", preview_frames, fps=preview_fps)

    default_parameter_usage = {
        "container_rigid_material": {"rho": 1200.0, "friction": 0.98},
        "camera_sampling": "sampled by script, not from source dataset",
        "object_scale_sampling": "sampled by script from target_longest_size_range_m",
        "rigid_default_friction_when_missing": 0.55,
        "soft_parts_missing_density_youngs_poisson": "skipped instead of default-filled",
        "sophy_particle_preference": "prefer sampled_points.ply + sampled_points_info.npz; fallback to vecset_v3.1.npz when sampled points are unavailable",
        "sophy_material_slot_mapping": "usemtl material_N is mapped to mat_params_new_v3.4.json entry order, then consolidated by material id for particle groups",
        "sigma_y_usage": "preserved in metadata and used only to choose elastic vs elastoplastic; Genesis MPM constructor here does not receive sigma_y directly",
    }

    scene_metadata = {
        "scene_id": scene_cfg["scene_id"],
        "output_relpath": scene_cfg.get("output_relpath"),
        "seed": scene_cfg["seed"],
        "family": scene_cfg["family"],
        "mpm_motion_category": scene_cfg.get("mpm_motion_category"),
        "mpm_motion_label_zh": scene_cfg.get("mpm_motion_label_zh"),
        "scene_builder": scene_cfg.get("scene_builder"),
        "object_count_bucket": scene_cfg.get("object_count_bucket"),
        "sample_index_in_bucket": scene_cfg.get("sample_index_in_bucket"),
        "pattern": scene_cfg["pattern"],
        "num_objects": len(scene_cfg["objects"]),
        "num_static_objects": scene_cfg.get("num_static_objects", 0),
        "num_dynamic_objects": scene_cfg.get("num_dynamic_objects", len(scene_cfg["objects"])),
        "num_soft_parts": int(sum(obj.get("soft_part_count", 0) for obj in scene_cfg["objects"])),
        "motion_modes_present": scene_cfg.get("motion_modes_present", []),
        "sim_steps": num_steps,
        "target_seconds": scene_cfg.get("target_seconds"),
        "dt": scene_cfg["sim_options"]["dt"],
        "substeps": scene_cfg["sim_options"]["substeps"],
        "preview_stride": preview_stride,
        "preview_fps": preview_fps,
        "collision_detected": collision_detected,
        "container": scene_cfg["container"],
        "objects": [
            {
                "scene_object_id": obj["scene_object_id"],
                "source_object_id": obj["source_object_id"],
                "object_name": obj["object_name"],
                "category": obj["category"],
                "asset_id": obj["asset_id"],
                "source_type": obj["source_type"],
                "motion_type": obj["motion_type"],
                "bbox_extents": obj["bbox_extents"],
                "scale": obj["scene_scale"],
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
        "default_parameter_usage": default_parameter_usage,
        "notes": [
            "Rigid articulation geometry comes from the strict per-part URDF exported by physxnet_articulation_demo.py.",
            "SOPHY soft particles reuse dataset-provided sampled_points.ply and sampled_points_info.npz whenever available, instead of re-sampling meshes inside Genesis.",
            "SOPHY vecset_v3.1.npz is preserved in metadata and used as a fallback particle source when sampled_points are unavailable.",
            "Rigid per-part mass/inertia/friction is preserved in the exported URDF and reapplied to Genesis links and joints after build when setters are available.",
            "MPM soft parts reuse dataset-provided density / Young's modulus / Poisson ratio, and SOPHY material JSON is also preserved in scene metadata.",
            "Object and container scale are sampled jointly to keep scene size physically reasonable.",
        ],
        "exports": {
            "files": {
                "rgb": "rgb/<frame:06d>.png",
                "depth": "depth/<frame:06d>.npy",
                "depth_vis": "depth_vis/<frame:06d>.png",
                "segmentation": "segmentation/<frame:06d>.npy",
                "normal": "normal/<frame:06d>.npy",
                "scene_pointcloud": "pointcloud/<frame:06d>.npz",
                "object_pointcloud": "object_pointcloud/<frame:06d>_obj<object_id>.npz",
                "frame_index_csv": "trajectories/frame_index.csv",
                "object_trajectory_csv": "trajectories/objects_world.csv",
                "object_pointcloud_index_csv": "trajectories/object_pointcloud_index.csv",
                "camera_intrinsics": "camera/intrinsics.npy",
                "camera_extrinsics": "camera/extrinsics.npy",
                "preview_video": "video/preview.mp4",
            },
            "object_pointcloud_npz_keys": [
                "xyz", "object_id", "frame", "solver", "centroid",
                "quat", "vel", "ang", "n_points_raw", "n_points_saved", "coordinate_frame",
            ],
        },
        "initial_floor_penetration_lift": penetration_lift_records,
        "status": "ok",
    }
    (out_dir / "scene_metadata.json").write_text(json.dumps(scene_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    scene.destroy()
    return scene_metadata


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Genesis MPM dataset from PhysXNet articulation assets and SOPHY soft assets.")
    parser.add_argument("--physx_root", type=str, required=True, help="PhysXNet root, e.g. /data/.../PhysXNet")
    parser.add_argument("--sophy_root", type=str, default=str(DEFAULT_SOPHY_ROOT), help="SOPHY root, e.g. /data/.../SOPHY_data")
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--dataset_root", type=str, required=True, help="Output dataset root")
    parser.add_argument("--prepared_asset_root", type=str, default=None, help="Cache directory for converted per-object articulation assets")
    parser.add_argument("--samples_per_category", type=int, default=1, help="Number of samples for each motion-category/count bucket.")
    parser.add_argument("--start_scene_idx", type=int, default=0, help="Start index within the generated scene plan.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target_seconds", type=float, default=3.0, help="Target physical duration in seconds. Also used to keep preview video duration close to physical duration.")
    parser.add_argument("--voxel_pitch", type=float, default=0.025)
    parser.add_argument("--object_scale_mult", type=float, default=1.0)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--object_ids", type=str, nargs="*", default=None, help="Optional explicit PhysXNet object ids")
    parser.add_argument("--asset_sources", type=str, nargs="+", choices=["physx", "sophy"], default=["physx", "sophy"], help="Choose which asset sources to include. Use only `sophy` to avoid mixing articulated PhysX rigid bodies.")
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
        sophy_root=Path(args.sophy_root),
        prepared_asset_root=prepared_asset_root,
        manifest_path=dataset_root / "asset_manifest.json",
        voxel_pitch=float(args.voxel_pitch),
        object_scale_mult=float(args.object_scale_mult),
        object_ids=args.object_ids,
        max_objects=int(args.max_objects),
        force_rebuild_assets=bool(args.force_rebuild_assets),
        asset_sources=tuple(args.asset_sources),
        solver_family_override=args.solver_family_override,
    )
    if not asset_bank:
        raise RuntimeError("No usable articulation assets were prepared.")

    scene_plan_list = build_scene_generation_plan(max(1, int(args.samples_per_category)))
    scene_metas = []
    failed_scenes = []
    for local_idx, scene_plan in enumerate(scene_plan_list[int(args.start_scene_idx):], start=int(args.start_scene_idx)):
        scene_idx = local_idx
        seed = int(args.seed) + scene_idx
        scene_cfg = sample_scene_cfg(
            scene_idx,
            scene_plan=scene_plan,
            asset_bank=asset_bank,
            seed=seed,
            target_seconds=float(args.target_seconds) if args.target_seconds is not None else None,
        )
        print(f"[INFO] exporting {scene_cfg['scene_id']} | pattern={scene_cfg['pattern']} | n_objects={len(scene_cfg['objects'])}")
        scene_metas.append(export_scene(scene_cfg, dataset_root))

    dataset_manifest = {
        "dataset_name": "genesis_mpm_mixed_dataset",
        "physx_root": str(args.physx_root),
        "sophy_root": str(args.sophy_root),
        "version": args.version,
        "dataset_root": str(dataset_root),
        "prepared_asset_root": str(prepared_asset_root),
        "n_scenes": len(scene_metas),
        "start_scene_idx": int(args.start_scene_idx),
        "seed": int(args.seed),
        "samples_per_category": int(args.samples_per_category),
        "target_seconds": float(args.target_seconds) if args.target_seconds is not None else None,
        "object_count_buckets": DATASET_OBJECT_COUNTS,
        "mpm_motion_category_specs": MPM_MOTION_CATEGORY_SPECS,
        "image_size": [IMG_W, IMG_H],
        "target_longest_size_range_m": list(TARGET_LONGEST_SIZE_RANGE),
        "asset_manifest": str(dataset_root / "asset_manifest.json"),
        "asset_sources": list(args.asset_sources),
        "scenes": scene_metas,
        "failed_scenes": failed_scenes,
    }
    (dataset_root / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()




'''
CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/dataset_3_mpm_genesis.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --sophy_root /data/gaoya/dataset/SOPHY_data \
  --version version_1 \
  --dataset_root /data/gaoya/AAA_test_video/Dataset_physV/Genesis_mpm_sophy_only \
  --prepared_asset_root /data/gaoya/AAA_test_video/Dataset_physV/Genesis_mpm_sophy_only/_prepared_assets \
  --samples_per_category 1 \
  --max_objects 5 \
  --asset_sources sophy \
  --target_seconds 3 \
  --force_rebuild_assets \
  --solver_family_override "mpm"




'''
