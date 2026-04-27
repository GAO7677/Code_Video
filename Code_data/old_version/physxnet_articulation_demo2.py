#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相比于上一版没有 softpart
PhysXNet -> Genesis-ready dataset + strict-per-part demo simulation

This version focuses on three requirements:
1) Coordinates are converted from PhysXNet Y-up into Genesis Z-up.
2) Rigid parts are exported and colored strictly per original JSON part, not merged by group.
3) Per-part physical parameters are kept exact from the JSON whenever the solver supports them.

Important notes:
- For rigid bodies, Genesis/URDF consumes per-link mass / inertia, and this script also reapplies per-link mass and
  friction to the built Genesis entity whenever the runtime exposes the corresponding setters.
- Young's modulus / Poisson's ratio are consumed directly by the soft solvers. For rigid bodies they are preserved in
  metadata only, because rigid contact models do not take them as runtime coefficients.
- If the JSON does not provide friction / restitution / damping, this script does NOT invent them as "exact JSON"
  parameters beyond the minimum runtime fallback friction needed by Genesis.
- For articulated groups with multiple child parts, a mass-light carrier link is created for the group joint,
  and each original rigid child part is attached to that carrier via a fixed joint. This preserves part-level
  colors, meshes, and inertial properties.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from dataclasses import dataclass, asdict, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
from dataset_3_utils_dataset import build_urdf_from_json_file

import imageio.v2 as imageio
import numpy as np
import trimesh







import re
from typing import Any, Dict, Optional

def parse_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None

def norm_text(x: Any) -> str:
    s = "" if x is None else str(x).lower()
    s = s.replace("_", " ").replace("-", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def has_any(text: str, keywords) -> bool:
    return any(k in text for k in keywords)

# 1) 刚体关键词：最高优先级
HARD_KWS = [
    "wood", "plywood",
    "plastic", "abs",
    "metal", "steel", "stainless",
    "aluminum", "aluminium", "iron",
    "glass", "tempered glass",
    "ceramic", "stone", "concrete"
]

# 2) 液体
LIQUID_KWS = [
    "water", "liquid", "solution", "beverage", "drink", "soup",
    "perfume", "oil", "shampoo", "conditioner", "detergent",
    "ketchup", "sauce", "condiment", "juice"
]

# 3) 颗粒 / 砂土
GRANULAR_KWS = [
    "sand", "soil", "powder", "granule", "granular",
    "grain", "rice", "beans", "salt", "sugar"
]

# 4) 雪
SNOW_KWS = ["snow", "frost"]

# 5) 布/纸/皮
CLOTH_BASE_KWS = [
    "fabric", "cloth", "textile", "leather", "felt", "canvas", "paper"
]

# 只有同时像“薄片/布片”时才判 cloth
THIN_SHEET_HINT_KWS = [
    "cover", "curtain", "sheet", "page", "label", "wrapper",
    "lining", "towel", "flag"
]

# 6) 体软体
SOFT_ELASTIC_KWS = [
    "foam", "sponge", "rubber", "silicone", "latex", "gel", "cushion"
]

# 7) 可塑软体
SOFT_PLASTIC_KWS = [
    "clay", "dough", "putty", "wax", "soap"
]

def choose_genesis_material_type(part: Dict[str, Any]) -> Dict[str, str]:
    """
    只判断:
      - solver_family
      - material_ctor
    不改、不截断任何 json 里的数值参数。
    """

    material = norm_text(part.get("material"))
    name = norm_text(part.get("name"))
    movement = norm_text(part.get("Movement_description"))
    text = " | ".join([material, name, movement])

    E_gpa = parse_float(part.get("Young's Modulus (GPa)"))

    # A. 刚体关键词最高优先级
    # 例如: "wood with fabric", "fabric-covered wood", "plastic and metal"
    if has_any(text, HARD_KWS):
        return {
            "solver_family": "rigid",
            "material_ctor": "gs.materials.Rigid",
            "reason": "hard material keyword has highest priority"
        }

    # B. 液体
    if has_any(text, LIQUID_KWS):
        return {
            "solver_family": "sph",
            "material_ctor": "gs.materials.SPH.Liquid",
            "reason": "liquid keyword matched"
        }

    # C. 颗粒
    if has_any(text, GRANULAR_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.Sand",
            "reason": "granular keyword matched"
        }

    # D. 雪
    if has_any(text, SNOW_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.Snow",
            "reason": "snow keyword matched"
        }

    # E. 薄布 / 薄纸
    if has_any(text, CLOTH_BASE_KWS) and has_any(text, THIN_SHEET_HINT_KWS):
        return {
            "solver_family": "pbd",
            "material_ctor": "gs.materials.PBD.Cloth",
            "reason": "cloth/paper + thin-sheet hint matched"
        }

    # F. 体软弹性
    if has_any(text, SOFT_ELASTIC_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.Elastic",
            "reason": "soft elastic keyword matched"
        }

    # G. 可塑软体
    if has_any(text, SOFT_PLASTIC_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.ElastoPlastic",
            "reason": "soft plastic keyword matched"
        }

    # H. 语义不清时，再用 E 做 fallback
    if E_gpa is not None:
        if E_gpa >= 5.0:
            return {
                "solver_family": "rigid",
                "material_ctor": "gs.materials.Rigid",
                "reason": "fallback by high Young's modulus"
            }
        elif E_gpa >= 0.05:
            return {
                "solver_family": "mpm",
                "material_ctor": "gs.materials.MPM.Elastic",
                "reason": "fallback by medium Young's modulus"
            }
        else:
            return {
                "solver_family": "mpm",
                "material_ctor": "gs.materials.MPM.ElastoPlastic",
                "reason": "fallback by low Young's modulus"
            }

    # I. 最终兜底
    return {
        "solver_family": "rigid",
        "material_ctor": "gs.materials.Rigid",
        "reason": "default fallback"
    }





# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_optional_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        s = str(x).strip().lower()
        if not s:
            return None
        try:
            return float(s.split()[0])
        except Exception:
            return None


def parse_density_to_kgm3(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    s = str(value).strip().lower()
    try:
        num = float(s.split()[0])
    except Exception:
        return default
    if "g/cm" in s:
        return num * 1000.0
    if "kg/m" in s:
        return num
    return num


def parse_modulus_to_pa(value: Any, name: str, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    s = str(value).strip().lower()
    try:
        num = float(s.split()[0])
    except Exception:
        return default
    if "gpa" in name or "(gpa)" in name or "GPa" in name:
        return num * 1e9
    if "mpa" in name or "(mpa)" in name or "MPa" in name:
        return num * 1e6
    if "kpa" in name or "(kpa)" in name or "kPa" in name:
        return num * 1e3
    # Bare numeric values under the PhysXNet key "Young's Modulus (GPa)" are interpreted as GPa.
    return num * 1e9


def parse_dimension_to_meters(dim_str: str) -> Optional[np.ndarray]:
    if not dim_str:
        return None
    s = str(dim_str).strip().split()[0]
    toks = s.replace("×", "*").split("*")
    vals = []
    for t in toks:
        try:
            vals.append(float(t) / 100.0)
        except Exception:
            return None
    if len(vals) != 3:
        return None
    return np.asarray(vals, dtype=np.float64)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():  # type: ignore[name-defined]
            child.tail = indent  # type: ignore[name-defined]
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def sanitize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    for fn in [
        "remove_unreferenced_vertices",
        "remove_duplicate_faces",
        "remove_degenerate_faces",
        "merge_vertices",
    ]:
        try:
            getattr(mesh, fn)()
        except Exception:
            pass
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("empty mesh")
    return mesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    return sanitize_mesh(mesh)


def merge_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    if len(meshes) == 0:
        raise ValueError("no meshes to merge")
    if len(meshes) == 1:
        return meshes[0].copy()
    return sanitize_mesh(trimesh.util.concatenate([m.copy() for m in meshes]))


def mesh_volume_fallback(mesh: trimesh.Trimesh) -> float:
    try:
        if mesh.is_volume and float(mesh.volume) > 1e-12:
            return float(abs(mesh.volume))
    except Exception:
        pass
    candidates: List[float] = []
    ext = np.maximum(np.asarray(mesh.extents, dtype=np.float64), 1e-6)
    bbox_vol = float(np.prod(ext))

    try:
        hull = mesh.convex_hull
        hull_vol = float(abs(hull.volume))
        if hull_vol > 1e-12:
            candidates.append(hull_vol)
    except Exception:
        pass

    try:
        area = float(mesh.area)
        thickness = float(np.clip(np.min(ext) * 0.35, 5e-4, 0.05))
        shell_vol = area * thickness
        if shell_vol > 1e-12:
            candidates.append(shell_vol)
    except Exception:
        pass

    candidates.append(bbox_vol * 0.015)
    return float(max(min(candidates), 1e-8))


def yup_to_zup_vec(v: List[float]) -> List[float]:
    """Rotate +90 deg around X: (x, y, z) -> (x, -z, y)."""
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return [x, -z, y]


YUP_TO_ZUP_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def yup_to_zup_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) @ YUP_TO_ZUP_ROT.T
    return sanitize_mesh(mesh)


def shift_mesh(mesh: trimesh.Trimesh, shift: np.ndarray) -> trimesh.Trimesh:
    out = mesh.copy()
    out.apply_translation(np.asarray(shift, dtype=np.float64))
    return sanitize_mesh(out)


# -----------------------------------------------------------------------------
# Material mapping and exact JSON parameters
# -----------------------------------------------------------------------------


@dataclass
class PartPhysical:
    part_id: int
    name: str
    material_name: str
    density_kgm3: Optional[float]
    youngs_pa: Optional[float]
    poisson: Optional[float]
    friction: Optional[float]
    restitution: Optional[float]
    damping: Optional[float]
    solver_family: str
    simulator_material: str
    material_ctor: str
    priority_rank: int
    basic_description: str
    functional_description: str
    movement_description: str
    mesh_path: str
    json_exact_parameters: Dict[str, Any]


CLASSIFICATION_POLICY_VERSION = "v3_parts_physical_drives_runtime_ctor_no_soft_parts"


def _choice_to_solver_tuple(choice: Dict[str, str]) -> Tuple[str, str, str]:
    material_ctor = str(choice.get("material_ctor", "gs.materials.Rigid"))
    if material_ctor == "gs.materials.Rigid":
        return "rigid", "rigid", material_ctor
    if material_ctor == "gs.materials.SPH.Liquid":
        return "sph_liquid", "liquid", material_ctor
    if material_ctor == "gs.materials.PBD.Cloth":
        return "pbd_cloth", "cloth", material_ctor
    if material_ctor == "gs.materials.MPM.Elastic":
        return "mpm_elastic", "elastic", material_ctor
    if material_ctor == "gs.materials.MPM.ElastoPlastic":
        return "mpm_elastoplastic", "elastoplastic", material_ctor
    if material_ctor == "gs.materials.MPM.Sand":
        return "mpm_sand", "sand", material_ctor
    if material_ctor == "gs.materials.MPM.Snow":
        return "mpm_snow", "snow", material_ctor
    if material_ctor == "gs.materials.MPM.Liquid":
        return "mpm_liquid", "liquid", material_ctor
    return "rigid", "rigid", "gs.materials.Rigid"


def _collapse_choice_for_override(choice: Dict[str, str], solver_family_override: Optional[str]) -> Dict[str, str]:
    if solver_family_override is None:
        return choice
    ov = str(solver_family_override).strip().lower()
    if ov in ["", "none", "auto", "hybrid"]:
        return choice
    if ov == "rigid":
        return {
            "solver_family": "rigid",
            "material_ctor": "gs.materials.Rigid",
            "reason": "forced rigid override",
        }
    if ov == "pbd_cloth":
        return {
            "solver_family": "pbd",
            "material_ctor": "gs.materials.PBD.Cloth",
            "reason": "forced pbd_cloth override",
        }
    if ov == "mpm":
        ctor = str(choice.get("material_ctor", "gs.materials.Rigid"))
        if ctor == "gs.materials.Rigid":
            return choice
        if ctor == "gs.materials.SPH.Liquid":
            return {
                "solver_family": "mpm",
                "material_ctor": "gs.materials.MPM.Liquid",
                "reason": "mpm override collapsed SPH liquid to MPM liquid",
            }
        if ctor == "gs.materials.PBD.Cloth":
            return {
                "solver_family": "mpm",
                "material_ctor": "gs.materials.MPM.Elastic",
                "reason": "mpm override collapsed cloth to MPM elastic",
            }
        return choice
    return choice


def solver_from_json(
    part: Dict[str, Any],
    solver_family_override: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    基于 part 的 material/name/Young's Modulus 预测 Genesis 求解器族与材料构造器。

    返回:
      (solver_family, simulator_material, material_ctor)

    其中:
      - solver_family: rigid / sph_liquid / pbd_cloth / mpm_elastic / mpm_elastoplastic / mpm_sand / mpm_snow / mpm_liquid
      - simulator_material: rigid / liquid / cloth / elastic / elastoplastic / sand / snow
      - material_ctor: 例如 gs.materials.MPM.Elastic
    """
    choice = choose_genesis_material_type(part)
    # choice = _collapse_choice_for_override(choice, solver_family_override=solver_family_override)
    return _choice_to_solver_tuple(choice)


def _first_present_key(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return None


def build_part_physical(
    part: Dict[str, Any],
    mesh_path: Path,
    solver_family_override: Optional[str] = None,
) -> PartPhysical:
    part_id = int(part.get("label", -1))
    density_kgm3 = parse_density_to_kgm3(part.get("density"), default=None)
    youngs_pa = parse_modulus_to_pa(part.get("Young's Modulus (GPa)"),name = "Young's Modulus (GPa)", default=None)
    poisson = safe_optional_float(part.get("Poisson's Ratio"))
    friction = safe_optional_float(_first_present_key(part, ["Friction Coefficient", "friction", "coefficient_of_friction"]))
    restitution = safe_optional_float(_first_present_key(part, ["Restitution", "restitution"]))
    damping = safe_optional_float(_first_present_key(part, ["Damping", "damping"]))
    solver_family, simulator_material, material_ctor = solver_from_json(part, solver_family_override=solver_family_override)
    print(f"solver_family={solver_family},simulator_material={simulator_material},material_ctor={material_ctor}")

    return PartPhysical(
        part_id=part_id,
        name=str(part.get("name", f"part_{part_id}")),
        material_name=str(part.get("material", "Unknown")),
        density_kgm3=density_kgm3,
        youngs_pa=youngs_pa,
        poisson=poisson,
        friction=friction,
        restitution=restitution,
        damping=damping,
        solver_family=solver_family,
        simulator_material=simulator_material,
        material_ctor=material_ctor,
        priority_rank=int(part.get("priority_rank", 0)),
        basic_description=str(part.get("Basic_description", "")),
        functional_description=str(part.get("Functional_description", "")),
        movement_description=str(part.get("Movement_description", "")),
        mesh_path=str(mesh_path),
        json_exact_parameters={
            "density": part.get("density", None),
            "youngs_modulus_gpa": part.get("Young's Modulus (GPa)", None),
            "poissons_ratio": part.get("Poisson's Ratio", None),
            "friction": _first_present_key(part, ["Friction Coefficient", "friction", "coefficient_of_friction"]),
            "restitution": _first_present_key(part, ["Restitution", "restitution"]),
            "damping": _first_present_key(part, ["Damping", "damping"]),
            "predicted_solver_family": solver_family,
            "predicted_material_ctor": material_ctor,
        },
    )


# -----------------------------------------------------------------------------
# PhysXNet parsing
# -----------------------------------------------------------------------------


def convert_joint_params_yup_to_zup(params: List[float]) -> List[float]:
    params = list(params)
    if len(params) >= 3:
        params[:3] = yup_to_zup_vec(params[:3])
    if len(params) >= 6:
        params[3:6] = yup_to_zup_vec(params[3:6])
    return params


@dataclass
class GroupRecord:
    group_id: str
    child_labels: List[int]
    parent_group: str
    params: List[float]
    joint_type: str


@dataclass
class PreparedObject:
    object_id: str
    object_name: str
    category: str
    dimension_m: Optional[List[float]]
    object_scale: float
    base_part_labels: List[int]
    rigid_group_carriers: List[Dict[str, Any]]
    rigid_part_links: List[Dict[str, Any]]
    floating_parts: List[Dict[str, Any]]
    output_dir: str
    urdf_path: Optional[str]
    preview_video: Optional[str]
    grounding_offset_z: float
    object_bbox_min: List[float]
    object_bbox_max: List[float]


def parse_group_info(group_info: Dict[str, Any]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {"base_group": group_info.get("0", [])}
    movable_groups: List[GroupRecord] = []
    for key, value in group_info.items():
        if key == "0":
            continue
        child_labels = list(value[0]) if isinstance(value[0], list) else [int(value[0])]
        parent_group = str(value[1])
        params = convert_joint_params_yup_to_zup(list(value[2]))
        
        joint_type = str(value[3])
        movable_groups.append(
            GroupRecord(
                group_id=str(key),
                child_labels=[int(x) for x in child_labels],
                parent_group=parent_group,
                params=params,
                joint_type=joint_type,
            )
        )
    parsed["movable_groups"] = movable_groups
    return parsed


# -----------------------------------------------------------------------------
# Voxel fill for soft parts
# -----------------------------------------------------------------------------


def voxel_fill_mesh(mesh: trimesh.Trimesh, pitch: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    meta: Dict[str, Any] = {"pitch": float(pitch), "fill_mode": "unknown"}
    try:
        vox = mesh.voxelized(pitch)
        try:
            vox = vox.fill()
            meta["fill_mode"] = "interior_fill"
        except Exception:
            meta["fill_mode"] = "surface_only"
        pts = np.asarray(vox.points, dtype=np.float32)
        if pts.size == 0:
            raise ValueError("empty voxel points")
        meta["num_points"] = int(len(pts))
        return pts, meta
    except Exception as e:
        raise RuntimeError(f"voxel_fill_mesh failed: {e}")


def voxel_fill_mesh_collision(mesh: trimesh.Trimesh, pitch: float) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    meta: Dict[str, Any] = {"pitch": float(pitch), "fill_mode": "unknown"}
    try:
        vox = mesh.voxelized(pitch)
        try:
            vox = vox.fill()
            meta["fill_mode"] = "interior_fill"
        except Exception:
            meta["fill_mode"] = "surface_only"
        solid = vox.as_boxes()
        solid = sanitize_mesh(solid)
        meta["num_vertices"] = int(len(solid.vertices))
        meta["num_faces"] = int(len(solid.faces))
        return solid, meta
    except Exception as e:
        raise RuntimeError(f"voxel_fill_mesh_collision failed: {e}")


# -----------------------------------------------------------------------------
# URDF export
# -----------------------------------------------------------------------------


def add_inertial(link: ET.Element, mass: float, inertia_diag: Tuple[float, float, float], xyz: str = "0 0 0") -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=xyz, rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.10f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia_diag[0]:.10f}",
        ixy="0.0",
        ixz="0.0",
        iyy=f"{inertia_diag[1]:.10f}",
        iyz="0.0",
        izz=f"{inertia_diag[2]:.10f}",
    )


def inertia_from_bbox(extents: np.ndarray, mass: float) -> Tuple[float, float, float]:
    ex, ey, ez = [float(max(v, 1e-5)) for v in extents]
    ixx = mass * (ey * ey + ez * ez) / 12.0
    iyy = mass * (ex * ex + ez * ez) / 12.0
    izz = mass * (ex * ex + ey * ey) / 12.0
    return ixx, iyy, izz


def add_mesh_visual_collision(
    link: ET.Element,
    visual_mesh_relpath: str,
    color_rgba: Tuple[float, float, float, float],
    collision_mesh_relpath: Optional[str] = None,
    collision_friction: Optional[float] = None,
    collision_restitution: Optional[float] = None,
) -> None:
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geom = ET.SubElement(visual, "geometry")
    ET.SubElement(geom, "mesh", filename=visual_mesh_relpath, scale="1 1 1")
    mat = ET.SubElement(visual, "material", name=f"mat_{link.attrib['name']}")
    ET.SubElement(mat, "color", rgba=" ".join(f"{v:.5f}" for v in color_rgba))

    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    cgeom = ET.SubElement(collision, "geometry")
    ET.SubElement(cgeom, "mesh", filename=collision_mesh_relpath or visual_mesh_relpath, scale="1 1 1")

    if collision_friction is not None:
        contact = ET.SubElement(collision, "contact")
        ET.SubElement(contact, "lateral_friction", value=str(float(collision_friction)))
        if collision_restitution is not None:
            ET.SubElement(contact, "restitution", value=str(float(collision_restitution)))


def build_joint(
    robot: ET.Element,
    parent: str,
    child: str,
    name: str,
    joint_type: str,
    params: List[float],
    origin_xyz: List[float],
    dynamics_damping: Optional[float] = None,
    dynamics_friction: Optional[float] = None,
) -> None:
    """按照 urdf_gen.py 参考代码严格解析各关节类型参数，生成 URDF 关节节点。

    params 各类型含义（Y-up->Z-up 转换后）：
      A: 无参数，生成 floating 关节
      B: [ax,ay,az, lo, hi]              — 平移轴 + 范围（5维）
      C: [ax,ay,az, px,py,pz, lo, hi]    — 旋转轴 + 锚点 + 角度范围×π（8维）
      D: [ax,ay,az, px,py,pz, ...]       — 球关节，分解为3个revolute（Z/X/Y）
      CB:[ax,ay,az, px,py,pz, lo,hi,
          ax1,ay1,az1, px1,py1,pz1, lo1,hi1] — revolute + prismatic 复合（16维）
    origin_xyz: 在父坐标系中 carrier link 相对父 carrier 的偏移（已由调用方算好）
    """
    jt = str(joint_type)
    origin_str = " ".join(str(float(x)) for x in origin_xyz)

    def _fstr(v: float) -> str:
        return str(float(v))

    def _maybe_add_dynamics(joint_elem: ET.Element) -> None:
        dyn_kwargs: Dict[str, str] = {}
        if dynamics_damping is not None:
            dyn_kwargs["damping"] = _fstr(max(dynamics_damping, 0.0))
        if dynamics_friction is not None:
            dyn_kwargs["friction"] = _fstr(max(dynamics_friction, 0.0))
        if dyn_kwargs:
            ET.SubElement(joint_elem, "dynamics", **dyn_kwargs)

    def _xyz_str(v: List[float]) -> str:
        return " ".join(_fstr(x) for x in v)

    # ── 类型 A：自由运动（floating） ──
    if jt == "A":
        # abstract link 已由调用方创建，这里只加 fixed + floating
        j = ET.SubElement(robot, "joint", name=name, type="floating")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        ET.SubElement(j, "origin", xyz=origin_str, rpy="0 0 0")
        _maybe_add_dynamics(j)
        return

    # ── 类型 E / 静止：固定关节 ──
    if jt == "E":
        j = ET.SubElement(robot, "joint", name=name, type="fixed")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child", link=child)
        ET.SubElement(j, "origin", xyz=origin_str, rpy="0 0 0")
        _maybe_add_dynamics(j)
        return

    # ── 类型 B：平移（prismatic） ──
    # params: [ax,ay,az, lo, hi]  (5维)
    # urdf_gen.py: axis=params[0:3], limit=params[-2], params[-1]
    # origin 放在 0 0 0（urdf_gen 里 B 不偏移锚点）
    if jt == "B":
        axis = params[:3] if len(params) >= 3 else [1.0, 0.0, 0.0]
        lo   = params[-2] if len(params) >= 2 else -0.1
        hi   = params[-1] if len(params) >= 1 else  0.1
        j = ET.SubElement(robot, "joint", name=name, type="prismatic")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child",  link=child)
        ET.SubElement(j, "origin", xyz=origin_str, rpy="0 0 0")
        ET.SubElement(j, "axis",   xyz=_xyz_str(axis))
        ET.SubElement(j, "limit",  lower=_fstr(lo), upper=_fstr(hi),
                      effort="2000.0", velocity="2.0")
        _maybe_add_dynamics(j)
        return

    # ── 类型 C：旋转（revolute） ──
    # params: [ax,ay,az, px,py,pz, lo, hi]  (8维)
    # urdf_gen.py:
    #   point    = params[3:6]  （锚点，作为 revolute joint 的 origin）
    #   pointrev = -params[3:6] （子 link 固定关节的偏移，把子 mesh 移回原点）
    #   axis     = params[0:3]
    #   limit    = params[6]*π .. params[7]*π
    # 这里 origin_xyz 已经是 child_anchor - parent_anchor，
    # 还需在 child 侧加 fixed joint 偏移 -anchor（由调用方处理，见 prepare_physxnet_object）
    if jt == "C":
        axis = params[:3]  if len(params) >= 3 else [0.0, 0.0, 1.0]
        lo   = params[6]   if len(params) >= 8 else (params[-2] if len(params) >= 2 else -1.0)
        hi   = params[7]   if len(params) >= 8 else (params[-1] if len(params) >= 1 else  1.0)
        j = ET.SubElement(robot, "joint", name=name, type="revolute")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child",  link=child)
        ET.SubElement(j, "origin", xyz=origin_str, rpy="0 0 0")
        ET.SubElement(j, "axis",   xyz=_xyz_str(axis))
        ET.SubElement(j, "limit",  lower=_fstr(lo * math.pi), upper=_fstr(hi * math.pi),
                      effort="2000.0", velocity="2.0")
        _maybe_add_dynamics(j)
        return

    # ── 类型 D：球形关节（3轴旋转，分解为 Z→X→Y 三个 revolute） ──
    # params: [ax,ay,az, px,py,pz, ...]  (锚点在 params[3:6])
    # urdf_gen.py 用三个串联 revolute 模拟球关节，范围均为 [-π, π]
    # 链路：parent → abstract_z → abstract_x → child
    if jt == "D":
        abs_z = f"{child}_absz"
        abs_x = f"{child}_absx"
        # abstract_z link
        lz = ET.SubElement(robot, "link", name=abs_z)
        add_inertial(lz, 0.01, inertia_from_bbox(np.asarray([0.01, 0.01, 0.01]), 0.01))
        # abstract_x link
        lx = ET.SubElement(robot, "link", name=abs_x)
        add_inertial(lx, 0.01, inertia_from_bbox(np.asarray([0.01, 0.01, 0.01]), 0.01))

        # joint 1: parent → abs_z，绕 Z 轴旋转，origin 在锚点
        j1 = ET.SubElement(robot, "joint", name=f"{name}_hz", type="revolute")
        ET.SubElement(j1, "parent", link=parent)
        ET.SubElement(j1, "child",  link=abs_z)
        ET.SubElement(j1, "origin", xyz=origin_str, rpy="0 0 0")
        ET.SubElement(j1, "axis",   xyz="0 0 1")
        ET.SubElement(j1, "limit",  lower=_fstr(-math.pi), upper=_fstr(math.pi),
                      effort="2000.0", velocity="2.0")
        _maybe_add_dynamics(j1)

        # joint 2: abs_z → abs_x，绕 X 轴旋转
        j2 = ET.SubElement(robot, "joint", name=f"{name}_hx", type="revolute")
        ET.SubElement(j2, "parent", link=abs_z)
        ET.SubElement(j2, "child",  link=abs_x)
        ET.SubElement(j2, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(j2, "axis",   xyz="1 0 0")
        ET.SubElement(j2, "limit",  lower=_fstr(-math.pi), upper=_fstr(math.pi),
                      effort="2000.0", velocity="2.0")
        _maybe_add_dynamics(j2)

        # joint 3: abs_x → child，绕 Y 轴旋转
        j3 = ET.SubElement(robot, "joint", name=f"{name}_hy", type="revolute")
        ET.SubElement(j3, "parent", link=abs_x)
        ET.SubElement(j3, "child",  link=child)
        ET.SubElement(j3, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(j3, "axis",   xyz="0 1 0")
        ET.SubElement(j3, "limit",  lower=_fstr(-math.pi), upper=_fstr(math.pi),
                      effort="2000.0", velocity="2.0")
        _maybe_add_dynamics(j3)
        return

    # ── 类型 CB：平移 + 旋转复合关节 ──
    # params (16维):
    #   [0:3]  = revolute 轴方向
    #   [3:6]  = 锚点
    #   [6]    = revolute 下限（×π）
    #   [7]    = revolute 上限（×π）
    #   [8:11] = prismatic 轴方向
    #   [11:14]= prismatic 锚点（urdf_gen 未使用）
    #   [14]   = prismatic 下限
    #   [15]   = prismatic 上限
    # urdf_gen.py 顺序：parent → abstract_x(prismatic) → child(revolute)
    if jt == "CB":
        abs_x = f"{child}_absx"
        lx = ET.SubElement(robot, "link", name=abs_x)
        add_inertial(lx, 0.01, inertia_from_bbox(np.asarray([0.01, 0.01, 0.01]), 0.01))

        # prismatic: parent → abstract_x
        axis_pris = params[8:11]  if len(params) >= 11 else [0.0, 1.0, 0.0]
        lo_pris   = params[14]    if len(params) >= 15 else (params[-2] if len(params) >= 2 else 0.0)
        hi_pris   = params[15]    if len(params) >= 16 else (params[-1] if len(params) >= 1 else 0.1)
        j1 = ET.SubElement(robot, "joint", name=f"{name}_pris", type="prismatic")
        ET.SubElement(j1, "parent", link=parent)
        ET.SubElement(j1, "child",  link=abs_x)
        ET.SubElement(j1, "origin", xyz=origin_str, rpy="0 0 0")
        ET.SubElement(j1, "axis",   xyz=_xyz_str(axis_pris))
        ET.SubElement(j1, "limit",  lower=_fstr(lo_pris), upper=_fstr(hi_pris),
                      effort="2000.0", velocity="2.0")
        _maybe_add_dynamics(j1)

        # revolute: abstract_x → child
        axis_rev = params[0:3]  if len(params) >= 3 else [0.0, 0.0, 1.0]
        lo_rev   = params[6]    if len(params) >= 8 else -1.0
        hi_rev   = params[7]    if len(params) >= 8 else  1.0
        j2 = ET.SubElement(robot, "joint", name=f"{name}_revo", type="revolute")
        ET.SubElement(j2, "parent", link=abs_x)
        ET.SubElement(j2, "child",  link=child)
        ET.SubElement(j2, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(j2, "axis",   xyz=_xyz_str(axis_rev))
        ET.SubElement(j2, "limit",  lower=_fstr(lo_rev * math.pi), upper=_fstr(hi_rev * math.pi),
                      effort="2000.0", velocity="2.0")
        


# -----------------------------------------------------------------------------
# Main conversion pipeline
# -----------------------------------------------------------------------------


def color_from_part_id(part_id: int) -> Tuple[float, float, float, float]:
    rng = np.random.RandomState(2026 + int(part_id) * 17)
    return (
        float(rng.uniform(0.15, 0.95)),
        float(rng.uniform(0.15, 0.95)),
        float(rng.uniform(0.15, 0.95)),
        1.0,
    )


def _choose_anchor_for_group(group: GroupRecord, group_mesh: trimesh.Trimesh) -> np.ndarray:
    if len(group.params) >= 6:
        return np.asarray(group.params[3:6], dtype=np.float64)
    return np.asarray(group_mesh.bounding_box.centroid, dtype=np.float64)


def _estimate_part_mass(mesh: trimesh.Trimesh, p: PartPhysical, fallback_density_kgm3: float) -> float:
    density = p.density_kgm3 if p.density_kgm3 is not None else fallback_density_kgm3
    return max(mesh_volume_fallback(mesh) * float(density), 1e-4)


def _mesh_inertial_origin_xyz(mesh: trimesh.Trimesh) -> str:
    try:
        center = np.asarray(mesh.center_mass, dtype=np.float64)
        if not np.all(np.isfinite(center)):
            raise ValueError("non-finite center_mass")
    except Exception:
        center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    return " ".join(f"{float(v):.10f}" for v in center.tolist())


def _load_json(json_path: Path) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _median_optional(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _group_runtime_joint_params(child_part_ids: List[int], part_phys: Dict[int, PartPhysical]) -> Dict[str, Optional[float]]:
    damping = _median_optional([part_phys[pid].damping for pid in child_part_ids if pid in part_phys])
    friction = _median_optional([part_phys[pid].friction for pid in child_part_ids if pid in part_phys])
    return {
        "joint_damping": damping,
        "joint_frictionloss": friction,
    }


def _default_entity_rigid_material(metadata: Dict[str, Any], default_friction: float) -> Dict[str, float]:
    rigid_parts = list(metadata.get("rigid_part_links", []))
    densities = [float(x["density_kgm3"]) for x in rigid_parts if x.get("density_kgm3") is not None]
    frictions = [float(x["friction"]) for x in rigid_parts if x.get("friction") is not None]
    restitutions = [float(x["restitution"]) for x in rigid_parts if x.get("restitution") is not None]
    return {
        "rho": float(np.median(densities)) if densities else 1000.0,
        "friction": float(np.median(frictions)) if frictions else float(default_friction),
        "restitution": float(np.median(restitutions)) if restitutions else 0.10,
    }


def _configure_genesis_rigid_entity_from_metadata(ent: Any, metadata: Dict[str, Any], default_friction: float) -> Dict[str, Any]:
    applied = {
        "link_mass_updates": 0,
        "link_friction_updates": 0,
        "joint_damping_updates": 0,
        "joint_frictionloss_updates": 0,
        "missing_links": [],
        "missing_joints": [],
    }

    for rec in metadata.get("rigid_part_links", []):
        link_name = rec.get("link_name")
        if not link_name:
            continue
        try:
            link = ent.get_link(link_name)
        except Exception:
            applied["missing_links"].append(str(link_name))
            continue

        if rec.get("mass_kg") is not None:
            try:
                link.set_mass(float(rec["mass_kg"]))
                applied["link_mass_updates"] += 1
            except Exception:
                pass

        friction = rec.get("friction", None)
        friction = float(friction) if friction is not None else float(default_friction)
        try:
            link.set_friction(float(np.clip(friction, 1e-2, 5.0)))
            applied["link_friction_updates"] += 1
        except Exception:
            pass

    for rec in metadata.get("rigid_group_carriers", []):
        joint_name = rec.get("joint_name")
        if not joint_name:
            continue
        try:
            joint = ent.get_joint(joint_name)
        except Exception:
            applied["missing_joints"].append(str(joint_name))
            continue

        dofs_idx = list(getattr(joint, "dofs_idx_local", []) or [])
        if not dofs_idx:
            continue

        damping = rec.get("joint_damping", None)
        if damping is not None:
            try:
                ent.set_dofs_damping(np.full((len(dofs_idx),), float(max(damping, 0.0)), dtype=np.float32), dofs_idx)
                applied["joint_damping_updates"] += len(dofs_idx)
            except Exception:
                pass

        frictionloss = rec.get("joint_frictionloss", None)
        if frictionloss is not None:
            try:
                ent.set_dofs_frictionloss(
                    np.full((len(dofs_idx),), float(max(frictionloss, 0.0)), dtype=np.float32),
                    dofs_idx,
                )
                applied["joint_frictionloss_updates"] += len(dofs_idx)
            except Exception:
                pass

    return applied


def _make_genesis_rigid_material(gs: Any, rho: float, friction: float, restitution: Optional[float] = None):
    kwargs: Dict[str, Any] = {
        "rho": float(rho),
        "friction": float(np.clip(friction, 1e-2, 5.0)),
    }
    if restitution is not None:
        kwargs["restitution"] = float(np.clip(restitution, 0.0, 1.2))
    try:
        return gs.materials.Rigid(**kwargs)
    except TypeError:
        kwargs.pop("restitution", None)
        return gs.materials.Rigid(**kwargs)


def _make_pbd_cloth_material_from_part(gs: Any, density: float, friction: Optional[float], youngs: Optional[float], damping: Optional[float]):
    friction_val = float(np.clip(friction if friction is not None else 0.15, 1e-3, 5.0))
    youngs_val = float(max(youngs if youngs is not None else 1e5, 1e3))
    air_resistance = float(max(damping if damping is not None else 1e-3, 1e-6))
    stretch_compliance = float(np.clip(1.0 / youngs_val, 1e-9, 1e-3))
    bending_compliance = float(np.clip(10.0 / youngs_val, 1e-8, 5e-2))
    rho_2d = float(max(density, 1e-3))
    return gs.materials.PBD.Cloth(
        rho=rho_2d,
        static_friction=friction_val,
        kinetic_friction=friction_val,
        stretch_compliance=stretch_compliance,
        bending_compliance=bending_compliance,
        air_resistance=air_resistance,
    )

def prepare_physxnet_object(
    physx_root: Path,
    version: str,
    object_id: str,
    output_root: Path,
    voxel_pitch: float,
    json_override: Optional[Path] = None,
    object_scale_mult: float = 1.0,
    fallback_density_kgm3: float = 800.0,
    solver_family_override: Optional[str] = None,
) -> PreparedObject:
    """
    从 PhysXNet 数据集准备铰接对象，生成 URDF 和元数据
    
    这是一个复杂的资产准备函数，用于将 PhysXNet 数据集中的铰接对象转换为可用于 Genesis 仿真的格式。
    函数执行以下主要步骤：
    
    1. **加载和解析**: 从 JSON 元数据和 OBJ 网格文件加载对象信息
    2. **坐标转换**: 将网格从 Y-up 坐标系转换为 Z-up 坐标系
    3. **缩放和规范化**: 根据对象尺寸和缩放倍数调整网格大小
    4. **部件分类**: 将部件分为刚体、软体、浮动部件等类别
    5. **URDF 生成**: 为刚体部分生成 URDF 文件，包含关节和链接
    6. **碰撞网格**: 使用体素填充生成碰撞网格
    7. **元数据导出**: 保存完整的元数据和物理参数
    
    Args:
        physx_root (Path): PhysXNet 数据集根目录路径
        version (str): 资产版本标识符（如 "version_1"）
        object_id (str): 对象的唯一标识符（如 "obj_001"）
        output_root (Path): 输出目录的根路径，准备好的资产将保存在 output_root/object_id/
        voxel_pitch (float): 体素网格的间距（单位：米），用于碰撞网格和软体粒子生成
                            较小的值会产生更精细的网格但计算成本更高
        json_override (Optional[Path]): 可选的自定义 JSON 元数据文件路径，
                                       如果不提供则使用默认路径 physx_root/version/finaljson/object_id.json
        object_scale_mult (float): 对象缩放倍数，默认为 1.0。用于调整对象的整体大小
        fallback_density_kgm3 (float): 当部件密度信息缺失时的默认密度值（单位：kg/m³），默认为 800.0
    
    Returns:
        PreparedObject: 包含以下属性的命名元组：
            - object_id (str): 对象 ID
            - object_name (str): 对象的人类可读名称
            - category (str): 对象类别
            - dimension_m (list): 对象尺寸 [x, y, z]（单位：米）
            - object_scale (float): 应用的缩放因子
            - base_part_labels (list): 基础（固定）部件的 ID 列表
            - rigid_group_carriers (list): 刚体组的载体链接信息列表
            - rigid_part_links (list): 刚体部件的链接信息列表
            - floating_parts (list): 浮动（未分配）部件的信息列表
            - output_dir (str): 输出目录路径
            - urdf_path (str): 生成的 URDF 文件路径
            - preview_video (Optional[str]): 预览视频路径（如果生成）
            - grounding_offset_z (float): Z 轴接地偏移量
    
    Raises:
        FileNotFoundError: 如果 JSON 元数据文件或部件网格目录不存在
        ValueError: 如果没有找到有效的部件网格
        RuntimeError: 如果无法解析父组的顺序（循环依赖）
    
    Output Directory Structure:
        output_root/object_id/
        ├── parts/                    # 原始部件网格（Z-up，缩放后）
        │   └── part_*.obj
        ├── rigid/                    # 刚体相关文件
        │   ├── object_id.urdf        # 生成的 URDF 文件
        │   ├── part_*_collision.obj  # 碰撞网格
        │   └── group_*_part_*_*.obj  # 组内部件的局部坐标网格
        ├── soft/                     # 软体部件文件
        │   ├── soft_part_*.obj       # 软体网格
        │   └── soft_part_*_particles.npy  # 软体粒子位置
        ├── meta/                     # 元数据
        │   └── metadata.json         # 完整的元数据 JSON
        └── scene_preview/            # 场景预览（如果生成）
    
    Notes:
        - 坐标系转换: 源网格采用 Y-up 约定，转换为 Z-up 约定
        - 部件导出: 刚体部件按原始部件导出，不进行平均或合并
        - 物理参数精确性:
          * 密度: 用于计算每个链接的质量/惯性，运行时再次应用于 Genesis 链接
          * 杨氏模量/泊松比: 直接用于软体部件的 Genesis 材料
          * 摩擦/恢复系数/阻尼: 写入每个刚体碰撞，运行时应用于 Genesis 链接/关节
        - 关节解析: 使用拓扑排序解析嵌套的父组关系
        - 体素填充: 用于生成碰撞网格和软体粒子，精度由 voxel_pitch 控制
    
    Example:
        >>> from pathlib import Path
        >>> result = prepare_physxnet_object(
        ...     physx_root=Path("/data/PhysXNet"),
        ...     version="version_1",
        ...     object_id="obj_001",
        ...     output_root=Path("/output/prepared_assets"),
        ...     voxel_pitch=0.025,
        ...     object_scale_mult=1.0
        ... )
        >>> print(f"Object: {result.object_name}, Category: {result.category}")
        >>> print(f"URDF saved to: {result.urdf_path}")
    """
    # ── 路径解析：确定 JSON 元数据文件和零件 OBJ 目录的路径 ──
    # version_root: PhysXNet 某版本的根目录，如 physx_root/version_1
    # json_path: 优先使用用户指定的 json_override，否则用默认路径
    # objs_dir: 每个零件的 OBJ 分割文件所在目录
    version_root = physx_root / version
    json_path = json_override if json_override is not None else version_root / "finaljson" / f"{object_id}.json"
    objs_dir = version_root / "partseg" / object_id / "objs"

    # 检查必要文件是否存在，不存在则提前报错
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not objs_dir.exists():
        raise FileNotFoundError(objs_dir)

    # ── 缓存检测：若 URDF 和 metadata 均已存在，直接读取返回，跳过重新生成 ──
    out_dir = output_root / object_id
    _cached_urdf = out_dir / "rigid" / f"{object_id}.urdf"
    _cached_meta = out_dir / f"meta" / "metadata.json"
    # 缓存检测需要感知 solver_family_override：
    # 不同 override / 分类策略导出的资产内容不同，不能混用缓存
    # 在 metadata 中记录生成时使用的 override，若与当前不一致则重新生成
    _cache_valid = False
    if _cached_urdf.exists() and _cached_meta.exists():
        with open(_cached_meta, "r", encoding="utf-8") as _f:
            _m = json.load(_f)
        _cached_override = _m.get("solver_family_override", None)
        _cached_policy = _m.get("classification_policy_version", None)
        if _cached_override == solver_family_override and _cached_policy == CLASSIFICATION_POLICY_VERSION:
            _cache_valid = True
    if _cache_valid:
        print(f"[Cache] {object_id}: URDF already exists, loading from cache.")
        return PreparedObject(
            object_id=_m["object_id"],
            object_name=_m["object_name"],
            category=_m["category"],
            dimension_m=_m.get("dimension_m"),
            object_scale=_m["object_scale"],
            base_part_labels=_m["base_part_labels"],
            rigid_group_carriers=_m["rigid_group_carriers"],
            rigid_part_links=_m["rigid_part_links"],
            floating_parts=_m["floating_parts"],
            output_dir=_m.get("output_dir", str(out_dir)),
            urdf_path=str(_cached_urdf),
            preview_video=None,
            grounding_offset_z=_m["grounding_offset_z"],
            object_bbox_min=_m.get("object_bbox_min", [0.0, 0.0, 0.0]),
            object_bbox_max=_m.get("object_bbox_max", [1.0, 1.0, 1.0]),
        )

    # ── 若简化版 URDF（urdf2/）不存在，用 build_urdf_from_json_file 生成 ──
    # 该版本供 urdf_browser 等工具预览使用，路径与 batch_generate_urdfs 一致
    _simple_urdf_dir = version_root / "urdf"
    _simple_urdf_path = _simple_urdf_dir / f"{object_id}.urdf"
    if not _simple_urdf_path.exists():
        _simple_urdf_dir.mkdir(parents=True, exist_ok=True)
        _geopath = str(version_root / "partseg")
        build_urdf_from_json_file(
            jsonfile=str(json_path),
            index=object_id,
            geopath=_geopath,
            output_path=str(_simple_urdf_path),
            robot_name=object_id,
            mesh_rel_root="./../partseg",
        )
        print(f"[URDF] Simple URDF generated: {_simple_urdf_path}")

    # 加载 JSON 元数据（含部件物理属性、运动组信息等）
    # parse_group_info 将 group_info 解析为结构化的 base_group 和 movable_groups
    meta = _load_json(json_path)
    parsed_groups = parse_group_info(meta.get("group_info", {}))

    # ── 输出目录初始化：清空旧结果，创建各子目录 ──
    # parts/       存放原始零件 OBJ（Z-up，缩放后）
    # rigid/       存放 URDF、碰撞网格、局部坐标系网格
    # soft/        存放软体网格和粒子 npy
    # meta/        存放 metadata.json
    # scene_preview/ 存放场景预览（可选）
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "parts")
    ensure_dir(out_dir / "rigid")
    ensure_dir(out_dir / "soft")
    ensure_dir(out_dir / "meta")
    ensure_dir(out_dir / "scene_preview")

    # ── 逐零件加载网格 + 物理参数 ──
    # part_meshes: 零件ID -> trimesh 网格对象（已转换为 Z-up 坐标系）
    # part_phys:   零件ID -> PartPhysical（密度、杨氏模量、泊松比、摩擦等）
    part_meshes: Dict[int, trimesh.Trimesh] = {}
    part_phys: Dict[int, PartPhysical] = {}

    for part in meta["parts"]:
        pid = int(part["label"])
        mesh_path = objs_dir / f"{pid}.obj"
        if not mesh_path.exists():   # 跳过缺失的零件文件
            print(f"Warning: Part {pid} mesh file not found: {mesh_path}")
            continue
        mesh = load_mesh(mesh_path)
        mesh = yup_to_zup_mesh(mesh)   # Y-up → Z-up 坐标系转换（绕X轴旋转-90°）
        part_meshes[pid] = mesh
        print(f"Part {pid} mesh loaded: {mesh_path}")

        part_phys[pid] = build_part_physical(part, mesh_path, solver_family_override=solver_family_override)  # 解析 JSON 中的物理参数

    if not part_meshes:
        raise ValueError(f"No valid part meshes found for object {object_id}")

    # ── 计算缩放因子，将网格归一化到物理真实尺寸 ──
    # 1. 合并所有零件网格，获取原始包围盒尺寸和中心点
    merged_raw = merge_meshes(list(part_meshes.values()))
    raw_extents = np.asarray(merged_raw.extents, dtype=np.float64)   # 原始网格三轴尺寸（无单位）
    raw_center = np.asarray(merged_raw.bounding_box.centroid, dtype=np.float64)  # 原始包围盒中心

    # 2. 从 JSON 的 dimension 字段解析真实物理尺寸（单位：米）
    #    dimension 字段格式如 "0.3*0.2*0.4 m"，对应 X/Y/Z 三轴
    dim_m = parse_dimension_to_meters(str(meta.get("dimension", "")))
    if dim_m is not None:
        # 将维度顺序从源 Y-up 约定转换为目标 Z-up 约定：[X, Y, Z] -> [X, Z, Y]
        dim_m = dim_m[[0, 2, 1]]
        # 缩放因子 = 真实最大尺寸 / 网格最大尺寸，使网格匹配真实物理尺度
        object_scale = float(np.max(dim_m) / max(np.max(raw_extents), 1e-8))
    else:
        # 没有 dimension 信息时，将最大轴归一化到 1 米
        object_scale = float(1.0 / max(np.max(raw_extents), 1e-8))

    # 应用用户传入的额外缩放倍数
    object_scale *= float(object_scale_mult)

    # 3. 对每个零件：平移到原点 → 缩放到物理尺度 → 清理网格 → 导出到 parts/ 目录
    for pid, mesh in list(part_meshes.items()):
        mesh = mesh.copy()
        mesh.apply_translation(-raw_center)   # 以整体包围盒中心为原点
        mesh.apply_scale(object_scale)        # 缩放到真实物理尺寸（米）
        mesh = sanitize_mesh(mesh)            # 修复退化面、重复顶点等问题
        part_meshes[pid] = mesh
        mesh.export(out_dir / "parts" / f"part_{pid:03d}.obj")

    # 4. 计算缩放后整体包围盒，用于接地偏移（让物体底部贴地）
    merged_scaled = merge_meshes(list(part_meshes.values()))
    bbox_min = np.asarray(merged_scaled.bounds[0], dtype=np.float64)  # 包围盒最小角（Z轴最低点）
    bbox_max = np.asarray(merged_scaled.bounds[1], dtype=np.float64)  # 包围盒最大角
    grounding_offset_z = float(-bbox_min[2])  # 将底部抬升到 Z=0 平面所需的偏移量

    # ── 零件分类：确定哪些是底座零件，哪些是可动零件 ──
    # base_group: JSON 中明确标注为固定底座的零件 ID 列表
    # movable_groups: 所有可动运动组（含子零件、关节类型、运动参数）
    # movable_child_labels: 所有被运动组管辖的零件 ID 集合
    # 最终 base_labels：原始底座零件 ∪ 未被任何运动组管辖的零件（孤立零件归底座）
    base_labels = sorted(int(x) for x in parsed_groups.get("base_group", []))
    movable_groups: List[GroupRecord] = parsed_groups.get("movable_groups", [])
    movable_child_labels = sorted({int(lbl) for g in movable_groups for lbl in g.child_labels})
    all_labels = sorted(part_meshes.keys())
    base_labels = sorted(set(base_labels).union(set(all_labels) - set(movable_child_labels)))
    # mov_raw: 原始 group_info 字典，用于按 urdf_gen.py 方式查找父组第一个子零件
    mov_raw: Dict[str, Any] = meta.get("group_info", {})

    # ── 辅助函数：添加固定关节（对应 urdf_gen.py 的 add_fixed_joint） ──
    def _add_fixed_joint(robot_elem: ET.Element, name: str, parent: str, child: str, xyz: str = "0 0 0") -> None:
        j = ET.SubElement(robot_elem, "joint", name=name, type="fixed")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child",  link=child)
        ET.SubElement(j, "origin", xyz=xyz, rpy="0 0 0")

    # ── 创建 URDF XML 根节点和世界基础链接 ──
    # robot 是整个 URDF 的根元素
    # world_base 是一个质量极小的虚拟根链接，作为所有底座零件和运动组的挂载点
    robot = ET.Element("robot", attrib={"name": f"physxnet_{object_id}_strict"})
    # 与 urdf_gen.py 保持一致：根节点命名为 l_world
    world_base = ET.SubElement(robot, "link", name="l_world")
    add_inertial(world_base, 0.01, inertia_from_bbox(np.asarray([0.01, 0.01, 0.01]), 0.01))

    rigid_part_links: List[Dict[str, Any]] = []
    rigid_group_carriers: List[Dict[str, Any]] = []
    floating_parts: List[Dict[str, Any]] = []

    group_anchor_world: Dict[str, np.ndarray] = {"0": np.zeros(3, dtype=np.float64)}
    # 组0的 carrier = l_world（与 urdf_gen.py 一致）
    group_carrier_link: Dict[str, str] = {"0": "l_world"}

    # ── 为底座零件生成 URDF link + 固定关节 ──
    # 与 urdf_gen.py 一致：link 命名为 l_{pid}，用 fixed joint 连接到上一个底座零件（链式）
    # 第一个底座零件通过 fixed joint 连到 l_world
    base_part_ids = [pid for pid in base_labels if pid in part_meshes]
    for pid in base_part_ids:
        p = part_phys[pid]
        mesh = part_meshes[pid]
        link_name = f"l_{pid}"   # 与 urdf_gen.py 保持一致
        visual_mesh_path = out_dir / "parts" / f"part_{pid:03d}.obj"

        # collision_mesh, collision_fill_meta = voxel_fill_mesh_collision(mesh, voxel_pitch)
        
        
        ### dbug
        collision_mesh = mesh
        collision_fill_meta={}

        
        collision_mesh_path = out_dir / "rigid" / f"part_{pid:03d}_non_collision.obj"
        collision_mesh.export(collision_mesh_path)

        # URDF 中使用相对路径（相对于 rigid/ 目录）
        mesh_rel = os.path.relpath(visual_mesh_path, out_dir / "rigid")
        collision_mesh_rel = os.path.relpath(collision_mesh_path, out_dir / "rigid")

        # 创建 URDF link 节点，添加惯性参数和视觉/碰撞几何
        link = ET.SubElement(robot, "link", name=link_name)
        mass = _estimate_part_mass(mesh, p, fallback_density_kgm3)  # 根据体积×密度估算质量
        add_inertial(link, mass, inertia_from_bbox(np.asarray(mesh.extents), mass), xyz=_mesh_inertial_origin_xyz(mesh))
        add_mesh_visual_collision(
            link, mesh_rel, color_from_part_id(pid),
            collision_mesh_relpath=collision_mesh_rel,
            collision_friction=p.friction,
            collision_restitution=p.restitution,
        )

        # 创建固定关节：与 urdf_gen.py 一致，底座零件链式连接
        # 第一个零件连到 l_world，后续零件连到前一个零件
        if pid == base_part_ids[0]:
            parent_for_joint = "l_world"
        else:
            prev_pid = base_part_ids[base_part_ids.index(pid) - 1]
            parent_for_joint = f"l_{prev_pid}"
        joint = ET.SubElement(robot, "joint",
                              name=f"joint_fixed_{parent_for_joint}_{link_name}",
                              type="fixed")
        ET.SubElement(joint, "parent", link=parent_for_joint)
        ET.SubElement(joint, "child",  link=link_name)
        ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")

        rigid_part_links.append(
            {
                "part_id": pid,
                "link_name": link_name,
                "parent_link": parent_for_joint,
                "group_id": "0",
                "mesh_path": str(visual_mesh_path),
                "collision_mesh_path": str(collision_mesh_path),
                "collision_voxel_fill": collision_fill_meta,
                "mesh_frame": "object_frame",
                "color_rgba": list(color_from_part_id(pid)),
                "mass_kg": float(mass),
                "density_kgm3": p.density_kgm3,
                "youngs_pa": p.youngs_pa,
                "poisson": p.poisson,
                "friction": p.friction,
                "restitution": p.restitution,
                "damping": p.damping,
                "solver_family": p.solver_family,
                "simulator_material": p.simulator_material,
                "material_ctor": p.material_ctor,
            }
        )

    # ── 按照 urdf_gen.py 的方式构建可动运动组的 URDF ──
    # 核心逻辑：
    #   1. 父组的代表 link = 父组第一个子零件的 link（l_<pid>）
    #   2. 每组先创建 abstract carrier link，再按关节类型连接
    #   3. C/D/CB 类型：joint origin 放在锚点（params[3:6]），
    #      child 侧用 fixed joint 做 -anchor 偏移
    #   4. 使用拓扑排序处理嵌套父组（父组未处理时跳过，等下一轮）

    # group_first_child_link: 每组第一个子零件对应的 link 名（用于父链接查找）
    group_first_child_link: Dict[str, str] = {}
    # 组0的代表 link = 第一个底座零件
    if base_part_ids:
        group_first_child_link["0"] = f"l_{base_part_ids[0]}"
        group_carrier_link["0"] = f"l_{base_part_ids[0]}"

    # 先把 group_first_child_link 全部算出（不依赖顺序）
    for g in movable_groups:
        if g.child_labels:
            first_pid = g.child_labels[0]
            group_first_child_link[g.group_id] = f"l_{first_pid}"
            group_carrier_link[g.group_id] = f"abstract_{g.group_id}"

    _prebuilt_links: set = set()  # 在 while unresolved 中提前创建的 l_{pid}，后面跳过重复创建
    unresolved = {g.group_id: g for g in movable_groups}
    progress = True
    while unresolved and progress:
        progress = False
        for gid in list(unresolved.keys()):
            g = unresolved[gid]
            # 父组必须已处理（其 carrier link 已知）
            if g.parent_group not in group_carrier_link:
                continue

            # 按 urdf_gen.py：父 link = 父组第一个子零件的 link
            parent_grp_val = mov_raw.get(g.parent_group)
            if g.parent_group == "0" or parent_grp_val is None:
                parent_link_name = group_carrier_link["0"]
            else:
                # 父组的第一个子零件 ID
                pchildren = parent_grp_val[0] if isinstance(parent_grp_val[0], list) else [parent_grp_val[0]]
                parent_link_name = f"l_{pchildren[0]}"

            carrier_name = f"abstract_{g.group_id}"
            child_first_pid = g.child_labels[0]
            child_link_name = f"l_{child_first_pid}"

            # abstract carrier link（虚拟轻质节点）
            abs_link = ET.SubElement(robot, "link", name=carrier_name)
            add_inertial(abs_link, 0.01, inertia_from_bbox(np.asarray([0.01, 0.01, 0.01]), 0.01))

            # ── 提前创建第一个刚体子零件的 l_{pid} link ──
            # urdfpy 要求 joint 引用的 child link 必须在 joint 之前已定义
            # fix_{abstract}_{l_pid} joint 会在下面写入，所以先把 link 创建好
            _p0 = part_phys.get(child_first_pid)
            _mesh0 = part_meshes.get(child_first_pid)
            # 被 joint 引用的第一个子零件必须在 URDF 中有对应 link，否则 urdfpy 会报
            # "invalid child link name" 错误。当 solver_family_override="mpm" 时，
            # 有 Young's Modulus 的 part 会被标记为软体，但作为关节 child 的 part
            # 必须保留为 rigid link（软体语义只影响独立实体，不影响关节骨架）。
            _force_rigid_for_joint = _p0 is not None and _p0.simulator_material != "rigid"
            if _force_rigid_for_joint and _p0 is not None:
                _p0 = dc_replace(_p0, simulator_material="rigid")
                part_phys[child_first_pid] = _p0
            if _p0 is not None and _mesh0 is not None:
                _visual_mesh_path0 = out_dir / "parts" / f"part_{child_first_pid:03d}.obj"
                _coll_mesh0, _coll_fill0 = voxel_fill_mesh_collision(_mesh0, voxel_pitch)
                _coll_path0 = out_dir / "rigid" / f"group_{g.group_id}_part_{child_first_pid:03d}_collision.obj"
                _coll_mesh0.export(_coll_path0)
                _mesh_rel0 = os.path.relpath(_visual_mesh_path0, out_dir / "rigid")
                _coll_rel0 = os.path.relpath(_coll_path0, out_dir / "rigid")
                _link0 = ET.SubElement(robot, "link", name=child_link_name)
                _mass0 = _estimate_part_mass(_mesh0, _p0, fallback_density_kgm3)
                add_inertial(_link0, _mass0,
                             inertia_from_bbox(np.asarray(_mesh0.extents), _mass0),
                             xyz=_mesh_inertial_origin_xyz(_mesh0))
                add_mesh_visual_collision(
                    _link0, _mesh_rel0, color_from_part_id(child_first_pid),
                    collision_mesh_relpath=_coll_rel0,
                    collision_friction=_p0.friction,
                    collision_restitution=_p0.restitution,
                )
                # 记录到 rigid_part_links（parent_link 在下面 jt 分支处理后补全）
                rigid_part_links.append({
                    "part_id": child_first_pid,
                    "link_name": child_link_name,
                    "parent_link": carrier_name,  # abstract carrier
                    "group_id": g.group_id,
                    "mesh_path": str(_visual_mesh_path0),
                    "collision_mesh_path": str(_coll_path0),
                    "collision_voxel_fill": _coll_fill0,
                    "mesh_frame": "object_frame",
                    "color_rgba": list(color_from_part_id(child_first_pid)),
                    "mass_kg": float(_mass0),
                    "density_kgm3": _p0.density_kgm3,
                    "youngs_pa": _p0.youngs_pa,
                    "poisson": _p0.poisson,
                    "friction": _p0.friction,
                    "restitution": _p0.restitution,
                    "damping": _p0.damping,
                    "solver_family": _p0.solver_family,
                    "simulator_material": _p0.simulator_material,
                    "material_ctor": _p0.material_ctor,
                })
                # 记录已提前创建的 link，后面的 for g in movable_groups 跳过它
                _prebuilt_links.add(child_first_pid)

            jt = g.joint_type
            p_raw = g.params  # 已做 Y-up→Z-up 转换
            joint_name = f"joint_group_{g.group_id}"
            group_runtime = _group_runtime_joint_params(g.child_labels, part_phys)

            if jt == "A":
                # fixed: abstract → child_first
                _add_fixed_joint(robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name, child_link_name, xyz="0 0 0")
                # floating: parent → abstract
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "A", p_raw, [0.0, 0.0, 0.0],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])

            elif jt == "B":
                # fixed: abstract → child_first（无锚点偏移）
                _add_fixed_joint(robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name, child_link_name, xyz="0 0 0")
                # prismatic: parent → abstract，axis=params[0:3]
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "B", p_raw, [0.0, 0.0, 0.0],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])

            elif jt == "C":
                # 锚点 = params[3:6]（已转 Z-up）
                anchor = p_raw[3:6] if len(p_raw) >= 6 else [0.0, 0.0, 0.0]
                pointrev = " ".join(str(-float(x)) for x in anchor)
                point    = " ".join(str( float(x)) for x in anchor)
                # fixed: abstract → child，偏移 -anchor（把子网格平移回原点）
                _add_fixed_joint(robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name, child_link_name, xyz=pointrev)
                # revolute: parent → abstract，origin=anchor
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "C", p_raw, [float(x) for x in anchor],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])

            elif jt == "D":
                anchor = p_raw[3:6] if len(p_raw) >= 6 else [0.0, 0.0, 0.0]
                pointrev = " ".join(str(-float(x)) for x in anchor)
                point    = " ".join(str( float(x)) for x in anchor)
                _add_fixed_joint(robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name, child_link_name, xyz=pointrev)
                # D 类型在 build_joint 内部自动创建 abs_z / abs_x 中间 link
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "D", p_raw, [float(x) for x in anchor],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])

            elif jt == "CB":
                anchor = p_raw[3:6] if len(p_raw) >= 6 else [0.0, 0.0, 0.0]
                pointrev = " ".join(str(-float(x)) for x in anchor)
                _add_fixed_joint(robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name, child_link_name, xyz=pointrev)
                # CB 类型在 build_joint 内部创建 abstract_x 中间 link
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "CB", p_raw, [float(x) for x in anchor],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])

            else:
                # 未知类型，降级 fixed
                _add_fixed_joint(robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name, child_link_name, xyz="0 0 0")
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "E", p_raw, [0.0, 0.0, 0.0])

            group_anchor_world[g.group_id] = np.asarray(
                p_raw[3:6] if len(p_raw) >= 6 else [0.0, 0.0, 0.0], dtype=np.float64)
            rigid_group_carriers.append(
                {
                    "group_id": g.group_id,
                    "joint_name": joint_name,
                    "carrier_link": carrier_name,
                    "parent_group": g.parent_group,
                    "parent_link": parent_link_name,
                    "joint_type": jt,
                    "joint_params": [float(x) for x in p_raw],
                    "anchor_world": group_anchor_world[g.group_id].tolist(),
                    "joint_damping": group_runtime["joint_damping"],
                    "joint_frictionloss": group_runtime["joint_frictionloss"],
                }
            )
            progress = True
            unresolved.pop(gid)




    if unresolved:
        raise RuntimeError(f"Could not resolve parent order for groups: {sorted(unresolved.keys())}")

    # ── 为各运动组的子零件生成 l_{pid} link，按 urdf_gen.py 方式链式 fixed joint 连接 ──
    # 组内多个子零件：第一个连到 abstract carrier，后续链式连接
    # 软体零件单独导出粒子，不加入 URDF
    covered_labels = set(base_part_ids)
    for g in movable_groups:
        all_children = [pid for pid in g.child_labels if pid in part_meshes]
        rigid_children = [pid for pid in all_children if part_phys[pid].simulator_material == "rigid"]
        soft_children  = [pid for pid in all_children if part_phys[pid].simulator_material != "rigid"]

        carrier_name = f"abstract_{g.group_id}"

        for i, pid in enumerate(rigid_children):
            p = part_phys[pid]
            covered_labels.add(pid)
            link_name = f"l_{pid}"   # 与 urdf_gen.py 一致

            # 若该 link 已在 while unresolved 中提前创建，跳过 link/metadata 创建
            # 但仍需为 i>0 的后续子零件补充链式 fixed joint
            if pid in _prebuilt_links:
                if i > 0:
                    prev_link = f"l_{rigid_children[i - 1]}"
                    j = ET.SubElement(robot, "joint",
                                      name=f"joint_fixed_{prev_link}_{link_name}",
                                      type="fixed")
                    ET.SubElement(j, "parent", link=prev_link)
                    ET.SubElement(j, "child",  link=link_name)
                    ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
                continue

            visual_mesh_path = out_dir / "parts" / f"part_{pid:03d}.obj"
            collision_mesh, collision_fill_meta = voxel_fill_mesh_collision(part_meshes[pid], voxel_pitch)
            collision_mesh_path = out_dir / "rigid" / f"group_{g.group_id}_part_{pid:03d}_collision.obj"
            collision_mesh.export(collision_mesh_path)
            mesh_rel = os.path.relpath(visual_mesh_path, out_dir / "rigid")
            collision_mesh_rel = os.path.relpath(collision_mesh_path, out_dir / "rigid")

            link = ET.SubElement(robot, "link", name=link_name)
            mass = _estimate_part_mass(part_meshes[pid], p, fallback_density_kgm3)
            add_inertial(link, mass,
                         inertia_from_bbox(np.asarray(part_meshes[pid].extents), mass),
                         xyz=_mesh_inertial_origin_xyz(part_meshes[pid]))
            add_mesh_visual_collision(
                link, mesh_rel, color_from_part_id(pid),
                collision_mesh_relpath=collision_mesh_rel,
                collision_friction=p.friction,
                collision_restitution=p.restitution,
            )

            # 链式 fixed joint：第一个子零件已被 abstract carrier 通过 fix_{carrier}_{l_pid} 连接
            # 后续子零件链式连接到前一个子零件（与 urdf_gen.py 一致）
            if i > 0:
                prev_link = f"l_{rigid_children[i - 1]}"
                j = ET.SubElement(robot, "joint",
                                  name=f"joint_fixed_{prev_link}_{link_name}",
                                  type="fixed")
                ET.SubElement(j, "parent", link=prev_link)
                ET.SubElement(j, "child",  link=link_name)
                ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")

            rigid_part_links.append(
                {
                    "part_id": pid,
                    "link_name": link_name,
                    "parent_link": carrier_name if i == 0 else f"l_{rigid_children[i-1]}",
                    "group_id": g.group_id,
                    "mesh_path": str(visual_mesh_path),
                    "collision_mesh_path": str(collision_mesh_path),
                    "collision_voxel_fill": collision_fill_meta,
                    "mesh_frame": "object_frame",
                    "color_rgba": list(color_from_part_id(pid)),
                    "mass_kg": float(mass),
                    "density_kgm3": p.density_kgm3,
                    "youngs_pa": p.youngs_pa,
                    "poisson": p.poisson,
                    "friction": p.friction,
                    "restitution": p.restitution,
                    "damping": p.damping,
                    "solver_family": p.solver_family,
                    "simulator_material": p.simulator_material,
                    "material_ctor": p.material_ctor,
                }
            )

        for pid in soft_children:
            covered_labels.add(pid)

    # ── 未被任何组覆盖的孤立零件：固定到 l_world ──
    for pid in sorted(all_labels):
        if pid in covered_labels:
            continue
        p = part_phys[pid]
        mesh = part_meshes[pid]
        link_name = f"l_{pid}"
        if p.simulator_material == "rigid":
            visual_mesh_path = out_dir / "parts" / f"part_{pid:03d}.obj"
            collision_mesh, collision_fill_meta = voxel_fill_mesh_collision(mesh, voxel_pitch)
            collision_mesh_path = out_dir / "rigid" / f"standalone_part_{pid:03d}_collision.obj"
            collision_mesh.export(collision_mesh_path)
            mesh_rel = os.path.relpath(visual_mesh_path, out_dir / "rigid")
            collision_mesh_rel = os.path.relpath(collision_mesh_path, out_dir / "rigid")
            link = ET.SubElement(robot, "link", name=link_name)
            mass = _estimate_part_mass(mesh, p, fallback_density_kgm3)
            add_inertial(link, mass, inertia_from_bbox(np.asarray(mesh.extents), mass),
                         xyz=_mesh_inertial_origin_xyz(mesh))
            add_mesh_visual_collision(
                link, mesh_rel, color_from_part_id(pid),
                collision_mesh_relpath=collision_mesh_rel,
                collision_friction=p.friction,
                collision_restitution=p.restitution,
            )
            j = ET.SubElement(robot, "joint",
                              name=f"joint_fixed_l_world_{link_name}", type="fixed")
            ET.SubElement(j, "parent", link="l_world")
            ET.SubElement(j, "child",  link=link_name)
            ET.SubElement(j, "origin", xyz="0 0 0", rpy="0 0 0")
            floating_parts.append(
                {
                    "part_id": pid,
                    "link_name": link_name,
                    "mesh_path": str(visual_mesh_path),
                    "collision_mesh_path": str(collision_mesh_path),
                    "collision_voxel_fill": collision_fill_meta,
                    "material_model": "rigid",
                    "reason": "unassigned_by_group_info",
                    "density_kgm3": p.density_kgm3,
                    "youngs_pa": p.youngs_pa,
                    "poisson": p.poisson,
                    "friction": p.friction,
                    "json_exact_parameters": p.json_exact_parameters,
                }
            )
        else:
            pass

    urdf_path = out_dir / "rigid" / f"{object_id}.urdf"
    indent_xml(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)

    metadata = {
        "object_id": object_id,
        "object_name": meta.get("object_name", object_id),
        "category": meta.get("category", "Unknown"),
        "output_dir": str(out_dir),
        "solver_family_override": solver_family_override,
        "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
        "json_path_used": str(json_path),
        "dimension_raw": meta.get("dimension", None),
        "dimension_m": dim_m.tolist() if dim_m is not None else None,
        "object_scale": object_scale,
        "raw_mesh_extents": raw_extents.tolist(),
        "object_bbox_min": bbox_min.tolist(),
        "object_bbox_max": bbox_max.tolist(),
        "grounding_offset_z": grounding_offset_z,
        "base_part_labels": base_labels,
        "group_info": meta.get("group_info", {}),
        "rigid_group_carriers": rigid_group_carriers,
        "rigid_part_links": rigid_part_links,
        "floating_parts": floating_parts,
        "parts_physical": {str(pid): asdict(p) for pid, p in part_phys.items()},
        "notes": {
            "coordinate_conversion": "source meshes and joint params are converted from Y-up to Z-up",
            "rigid_export": "rigid parts are exported per original part, never averaged or merged into a single colored group",
            "runtime_exactness": {
                "density": "used per rigid part to compute per-link mass/inertia and applied again to Genesis links after URDF load",
                "youngs_poisson": "used directly for soft-part Genesis materials; rigid bodies preserve them in metadata only because rigid contact models do not consume them directly",
                "friction_restitution_damping": "friction and restitution are written per rigid collision; friction and damping are also applied at runtime to rigid links / joints when Genesis exposes the corresponding setters",
            },
            "runtime_soft_reconstruction": "soft entities are reconstructed from parts_physical[*].material_ctor at runtime; soft_parts is no longer emitted",
        },
    }
    with open(out_dir / "meta" / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return PreparedObject(
        object_id=object_id,
        object_name=str(meta.get("object_name", object_id)),
        category=str(meta.get("category", "Unknown")),
        dimension_m=dim_m.tolist() if dim_m is not None else None,
        object_scale=object_scale,
        base_part_labels=base_labels,
        rigid_group_carriers=rigid_group_carriers,
        rigid_part_links=rigid_part_links,
        floating_parts=floating_parts,
        output_dir=str(out_dir),
        urdf_path=str(urdf_path),
        preview_video=None,
        grounding_offset_z=grounding_offset_z,
        object_bbox_min=bbox_min.tolist(),
        object_bbox_max=bbox_max.tolist(),
    )


# -----------------------------------------------------------------------------
# Optional Genesis demo simulation
# -----------------------------------------------------------------------------

def _mesh_bounds_info(mesh_path: Path, scale: float = 1.0) -> Optional[Dict[str, Any]]:
    try:
        mesh = trimesh.load(mesh_path, process=False)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        bmin = bounds[0] * float(scale)
        bmax = bounds[1] * float(scale)
        center = 0.5 * (bmin + bmax)
        size = np.maximum(bmax - bmin, 1e-6)
        return {
            "bounds_min": bmin.tolist(),
            "bounds_max": bmax.tolist(),
            "bounds_center": center.tolist(),
            "bounds_size": size.tolist(),
        }
    except Exception:
        return None


def _resolve_runtime_part_mesh_path(
    obj_dir: Path,
    pid: int,
    part_meta: Optional[dict] = None,
    legacy_soft: Optional[dict] = None,
) -> Optional[Path]:
    candidates: List[Path] = []

    if legacy_soft is not None:
        legacy_mesh = legacy_soft.get("mesh_path")
        if legacy_mesh:
            candidates.append(Path(str(legacy_mesh)))
        group_id = legacy_soft.get("group_id")
        if group_id is not None:
            candidates.append(obj_dir / "soft" / f"soft_group_{group_id}_part_{pid:03d}.obj")
        candidates.append(obj_dir / "soft" / f"soft_part_{pid:03d}.obj")

    candidates.append(obj_dir / "parts" / f"part_{pid:03d}.obj")

    if part_meta is not None and part_meta.get("mesh_path"):
        candidates.append(Path(str(part_meta["mesh_path"])))

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand
    return None


def _legacy_material_ctor_from_model(raw_model: str) -> str:
    raw_model = str(raw_model or "elastic").lower()
    if raw_model == "cloth":
        return "gs.materials.PBD.Cloth"
    if raw_model == "liquid":
        return "gs.materials.SPH.Liquid"
    if raw_model == "sand":
        return "gs.materials.MPM.Sand"
    if raw_model == "snow":
        return "gs.materials.MPM.Snow"
    if raw_model == "elastoplastic":
        return "gs.materials.MPM.ElastoPlastic"
    return "gs.materials.MPM.Elastic"


def _build_soft_spec_from_sources(
    obj_dir: Path,
    pid: int,
    part_meta: dict,
    legacy_soft: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    material_ctor = str(part_meta.get("material_ctor") or "")
    print(f"material_ctor={material_ctor}")
    if not material_ctor and legacy_soft is not None:
        material_ctor = str(legacy_soft.get("material_ctor") or "")
    if not material_ctor:
        material_ctor = _legacy_material_ctor_from_model(
            (legacy_soft or {}).get("material_model", part_meta.get("simulator_material", "elastic"))
        )

    if material_ctor == "gs.materials.Rigid":
        return None

    mesh_path = _resolve_runtime_part_mesh_path(obj_dir=obj_dir, pid=pid, part_meta=part_meta, legacy_soft=legacy_soft)
    if mesh_path is None:
        return None

    density = part_meta.get("density_kgm3", None)
    if density is None and legacy_soft is not None:
        density = legacy_soft.get("density_kgm3", None)

    youngs = part_meta.get("youngs_pa", None)
    if youngs is None and legacy_soft is not None:
        youngs = legacy_soft.get("youngs_pa", None)

    poisson = part_meta.get("poisson", None)
    if poisson is None and legacy_soft is not None:
        poisson = legacy_soft.get("poisson", None)

    friction = part_meta.get("friction", None)
    if friction is None and legacy_soft is not None:
        friction = legacy_soft.get("friction", None)

    restitution = part_meta.get("restitution", None)
    if restitution is None and legacy_soft is not None:
        restitution = legacy_soft.get("restitution", None)

    damping = part_meta.get("damping", None)
    if damping is None and legacy_soft is not None:
        damping = legacy_soft.get("damping", None)

    solver_family = str(part_meta.get("solver_family") or ((legacy_soft or {}).get("solver_family") or ""))
    material_model = str(
        part_meta.get("simulator_material")
        or ((legacy_soft or {}).get("material_model") or "elastic")
    ).lower()
    # print(f"🩶 {solver_family},{material_model}")

    spec = {
        "pid": int(pid),
        "group_id": None if legacy_soft is None else legacy_soft.get("group_id"),
        "part_name": str(part_meta.get("name", f"part_{pid}")),
        "material_name": str(part_meta.get("material_name", "Unknown")),
        "mesh_path": str(mesh_path),
        "scale": 1.0,
        "euler": (0.0, 0.0, 0.0),
        "file_meshes_are_zup": True,
        "density": float(density if density is not None else 800.0),
        "youngs": float(youngs if youngs is not None else 1e7),
        "poisson": float(poisson if poisson is not None else 0.3),
        "friction": friction,
        "restitution": restitution,
        "damping": damping,
        "solver_family": solver_family,
        "material_model": material_model,
        "material_ctor": material_ctor,
        "color": list(color_from_part_id(pid)),
    }
    bounds_info = _mesh_bounds_info(mesh_path, scale=1.0)
    if bounds_info is not None:
        spec.update(bounds_info)
    return spec


def _collect_soft_specs(obj_dir: Path, metadata: dict):
    specs: List[Dict[str, Any]] = []
    parts_physical = metadata.get("parts_physical", {}) if isinstance(metadata, dict) else {}
    # print("parts_physical",parts_physical)
    legacy_soft_index = {}


    for pid_str, part_meta in sorted(parts_physical.items(), key=lambda kv: int(kv[0])):

        pid = int(pid_str)
        print(f"pid={pid}")
        print(isinstance(part_meta, dict))

        if not isinstance(part_meta, dict):
            continue
        legacy_soft = legacy_soft_index.get(pid)
        spec = _build_soft_spec_from_sources(
            obj_dir=obj_dir,
            pid=pid,
            part_meta=part_meta,
            legacy_soft=legacy_soft,
        )
        # print(spec)
        if spec is not None:
            specs.append(spec)
    
    return specs




def _spec_uses_mpm(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor", "")).startswith("gs.materials.MPM.")


def _spec_uses_sph(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor", "")) == "gs.materials.SPH.Liquid"


def _spec_uses_pbd(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor", "")) == "gs.materials.PBD.Cloth"


def _xy_overlap_ratio(a_min_xy, a_max_xy, b_min_xy, b_max_xy) -> float:
    ax0, ay0 = float(a_min_xy[0]), float(a_min_xy[1])
    ax1, ay1 = float(a_max_xy[0]), float(a_max_xy[1])
    bx0, by0 = float(b_min_xy[0]), float(b_min_xy[1])
    bx1, by1 = float(b_max_xy[0]), float(b_max_xy[1])
    inter_x = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    inter_y = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = inter_x * inter_y
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = max(min(area_a, area_b), 1e-12)
    return float(inter / denom)


def _iter_rigid_mesh_bounds(metadata: dict, placed_pos: np.ndarray):
    parts_physical = metadata.get("parts_physical", {}) if isinstance(metadata, dict) else {}
    placed_pos = np.asarray(placed_pos, dtype=np.float64)
    recs = []
    for rec in metadata.get("rigid_part_links", []):
        mesh_path = Path(rec.get("mesh_path", ""))
        if not mesh_path.exists():
            continue
        info = _mesh_bounds_info(mesh_path, scale=1.0)
        if info is None:
            continue
        pid = int(rec.get("part_id", -1))
        pmeta = parts_physical.get(str(pid), {}) if isinstance(parts_physical, dict) else {}
        bmin = np.asarray(info["bounds_min"], dtype=np.float64) + placed_pos
        bmax = np.asarray(info["bounds_max"], dtype=np.float64) + placed_pos
        recs.append(
            {
                "part_id": pid,
                "part_name": str(pmeta.get("name", f"part_{pid}")),
                "material_name": str(pmeta.get("material_name", "Unknown")),
                "bounds_min": bmin,
                "bounds_max": bmax,
            }
        )
    return recs


def _infer_liquid_proxy(spec: Dict[str, Any], metadata: dict, placed_pos, particle_size: float = 0.01) -> Dict[str, Any]:
    placed_pos = np.asarray(placed_pos, dtype=np.float64)
    if "bounds_min" not in spec or "bounds_max" not in spec:
        center = placed_pos.copy()
        return {
            "kind": "box",
            "pos": tuple(center.tolist()),
            "size": (0.08, 0.08, 0.06),
            "support_z": float(center[2] - 0.03),
            "surface_z": float(center[2] + 0.03),
            "source": "fallback_no_bounds",
        }

    lmin = np.asarray(spec["bounds_min"], dtype=np.float64) + placed_pos
    lmax = np.asarray(spec["bounds_max"], dtype=np.float64) + placed_pos
    lcenter = 0.5 * (lmin + lmax)
    lsize = np.maximum(lmax - lmin, 1e-6)
    surface_z = float(lmax[2])

    candidates = []
    for rec in _iter_rigid_mesh_bounds(metadata, placed_pos):
        rmin = rec["bounds_min"]
        rmax = rec["bounds_max"]
        overlap = _xy_overlap_ratio(lmin[:2], lmax[:2], rmin[:2], rmax[:2])
        if overlap < 0.25:
            continue
        top_z = float(rmax[2])
        if top_z >= surface_z - max(0.01, 1.5 * particle_size):
            continue
        name_text = norm_text(rec.get("part_name", ""))
        score = top_z + 0.05 * overlap
        if has_any(name_text, ["bottom", "base", "floor"]):
            score += 5.0
        if has_any(name_text, ["body", "wall", "side", "rim"]):
            score -= 0.25
        candidates.append((score, top_z, rec))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        support_z = float(candidates[0][1])
        support_source = f"rigid_part:{candidates[0][2].get('part_name', 'unknown')}"
    else:
        obj_bbox_min = np.asarray(metadata.get("object_bbox_min", [0.0, 0.0, 0.0]), dtype=np.float64) + placed_pos
        fallback_depth = max(0.04, 0.35 * float(max(lsize[0], lsize[1])))
        support_z = float(max(obj_bbox_min[2], surface_z - fallback_depth))
        support_source = "object_bbox_fallback"

    fill_height = max(surface_z - support_z, max(3.0 * particle_size, 0.03))
    size_x = max(float(lsize[0]) * 0.88, 3.0 * particle_size)
    size_y = max(float(lsize[1]) * 0.88, 3.0 * particle_size)
    center_z = support_z + 0.5 * fill_height
    circular = abs(size_x - size_y) / max(size_x, size_y, 1e-6) < 0.18

    proxy = {
        "support_z": support_z,
        "surface_z": surface_z,
        "source": support_source,
    }
    if circular:
        proxy.update(
            {
                "kind": "cylinder",
                "pos": (float(lcenter[0]), float(lcenter[1]), float(center_z)),
                "radius": 0.44 * min(size_x, size_y),
                "height": float(fill_height),
            }
        )
    else:
        proxy.update(
            {
                "kind": "box",
                "pos": (float(lcenter[0]), float(lcenter[1]), float(center_z)),
                "size": (float(size_x), float(size_y), float(fill_height)),
            }
        )
    return proxy


def _make_soft_morph(gs, spec, placed_pos, metadata=None, particle_size: float = 0.01):
    ctor = str(spec.get("material_ctor", ""))
    if ctor in ["gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid"]:
        proxy = _infer_liquid_proxy(spec, metadata or {}, placed_pos, particle_size=float(particle_size))
        spec["liquid_proxy"] = proxy
        if proxy.get("kind") == "cylinder" and hasattr(gs.morphs, "Cylinder"):
            return gs.morphs.Cylinder(
                pos=tuple(proxy["pos"]),
                euler=(0.0, 0.0, 0.0),
                radius=float(proxy["radius"]),
                height=float(proxy["height"]),
            )
        return gs.morphs.Box(
            pos=tuple(proxy["pos"]),
            euler=(0.0, 0.0, 0.0),
            size=tuple(proxy["size"]),
        )

    return gs.morphs.Mesh(
        file=spec["mesh_path"],
        scale=float(spec.get("scale", 1.0)),
        pos=tuple(np.asarray(placed_pos, dtype=float).tolist()),
        euler=tuple(spec.get("euler", (0.0, 0.0, 0.0))),
        file_meshes_are_zup=bool(spec.get("file_meshes_are_zup", True)),
    )


def _make_soft_material(gs, spec):
    ctor = str(spec.get("material_ctor", "gs.materials.MPM.Elastic"))
    density = float(spec.get("density", 800.0))
    youngs = float(spec.get("youngs", 1e7))
    poisson = float(spec.get("poisson", 0.3))
    friction = spec.get("friction", None)
    damping = spec.get("damping", None)

    if ctor == "gs.materials.SPH.Liquid":
        kwargs = {"rho": density, "sampler": "pbs"}
        # return gs.materials.SPH.Liquid(**kwargs)
        try:
            return gs.materials.SPH.Liquid(**kwargs)
        except TypeError:
            return gs.materials.SPH.Liquid()

    if ctor == "gs.materials.PBD.Cloth":
        kwargs = {}
        if density is not None:
            kwargs["rho"] = density
        if friction is not None:
            kwargs["static_friction"] = float(friction)
            kwargs["kinetic_friction"] = float(friction)
        if damping is not None:
            kwargs["air_resistance"] = float(damping)
        try:
            return gs.materials.PBD.Cloth(**kwargs)
        except TypeError:
            return gs.materials.PBD.Cloth()

    common_kwargs = {"E": youngs, "nu": poisson, "rho": density, "sampler": "pbs"}
    if ctor == "gs.materials.MPM.ElastoPlastic":
        try:
            return gs.materials.MPM.ElastoPlastic(**common_kwargs)
        except TypeError:
            common_kwargs.pop("sampler", None)
            return gs.materials.MPM.ElastoPlastic(**common_kwargs)
    if ctor == "gs.materials.MPM.Sand":
        try:
            return gs.materials.MPM.Sand(**common_kwargs)
        except TypeError:
            common_kwargs.pop("sampler", None)
            return gs.materials.MPM.Sand(**common_kwargs)
    if ctor == "gs.materials.MPM.Snow":
        try:
            return gs.materials.MPM.Snow(**common_kwargs)
        except TypeError:
            common_kwargs.pop("sampler", None)
            return gs.materials.MPM.Snow(**common_kwargs)
    if ctor == "gs.materials.MPM.Liquid":
        kwargs = {"rho": density, "sampler": "pbs"}
        try:
            return gs.materials.MPM.Liquid(**kwargs)
        except TypeError:
            kwargs.pop("sampler", None)
            try:
                return gs.materials.MPM.Liquid(**kwargs)
            except TypeError:
                return gs.materials.MPM.Liquid()

    try:
        return gs.materials.MPM.Elastic(**common_kwargs)
    except TypeError:
        common_kwargs.pop("sampler", None)
        return gs.materials.MPM.Elastic(**common_kwargs)


def _probe_one_mpm_bound(gs, spec, placed_pos, dt, substeps, probe_grid_density=24):
    import trimesh
    import numpy as np

    # 先用 mesh 粗略 bounds 给 probe scene 一个尽量小的 domain
    mesh = trimesh.load(spec["mesh_path"], process=False)
    scale = float(spec.get("scale", 1.0))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)  # [2,3]
    placed = np.asarray(placed_pos, dtype=np.float64)

    # Use AABB center (not mesh origin, not centroid) as domain center.
    # mesh.centroid is the volume/surface centroid which can differ from the AABB center,
    # causing the domain to be off-center and particles to spill out on one side.
    # Using (bounds[0]+bounds[1])/2 gives a symmetric AABB center guaranteed to be
    # equidistant from all particle extremes in the unrotated case.
    bounds_center_local = (bounds[0] + bounds[1]) / 2.0  # AABB center in local mesh space
    domain_center = placed + bounds_center_local * scale  # world-space domain center
    extent_half = (bounds[1] - bounds[0]) * scale / 2.0  # half-extent per axis

    # For rotated meshes the AABB can expand up to the full diagonal; use diagonal as half-extent.
    diag = float(np.linalg.norm(2.0 * extent_half))  # diagonal of the unrotated bbox
    euler_deg = spec.get("euler", (0.0, 0.0, 0.0))
    has_rotation = any(abs(float(a)) > 1e-6 for a in euler_deg)
    if has_rotation:
        extent_half = np.array([diag / 2.0, diag / 2.0, diag / 2.0], dtype=np.float64)

    tmp_margin = max(0.6, diag * 0.25)  # at least 0.6 m, or 25% of diagonal
    lower = tuple((domain_center - extent_half - tmp_margin).tolist())
    upper = tuple((domain_center + extent_half + tmp_margin).tolist())

    probe_scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=float(dt),
            substeps=int(substeps),
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=lower,
            upper_bound=upper,
            grid_density=int(probe_grid_density),
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False,
        ),
    )

    ent = probe_scene.add_entity(
        material=_make_soft_material(gs, spec),
        morph=gs.morphs.Mesh(
            file=spec["mesh_path"],
            scale=float(spec.get("scale", 1.0)),
            pos=tuple(np.asarray(placed_pos, dtype=float).tolist()),
            euler=tuple(spec.get("euler", (0.0, 0.0, 0.0))),
            file_meshes_are_zup=bool(spec.get("file_meshes_are_zup", True)),
        ),
        surface=gs.surfaces.Default(
            color=spec.get("color", (0.8, 0.3, 0.3, 1.0)),
            vis_mode="particle",
        ),
    )

    probe_scene.build()

    pos = ent.get_state().pos
    if hasattr(pos, "detach"):
        pos = pos.detach().cpu().numpy()
    else:
        pos = np.asarray(pos)

    pos = np.asarray(pos).reshape(-1, 3)
    gmin = pos.min(axis=0)
    gmax = pos.max(axis=0)

    try:
        probe_scene.destroy()
    except Exception:
        pass

    return gmin, gmax

def _auto_place_and_domain_by_probe(
    gs,
    mpm_specs,
    placed_pos,
    dt,
    substeps,
    final_grid_density=64,
    probe_grid_density=24,
    floor_clearance=0.01,
    extra_margin=0.1,
):
    placed_pos = np.asarray(placed_pos, dtype=np.float64).copy()

    mins = []
    maxs = []

    for i, spec in enumerate(mpm_specs):
        pmin, pmax = _probe_one_mpm_bound(
            gs=gs,
            spec=spec,
            placed_pos=placed_pos,
            dt=dt,
            substeps=substeps,
            probe_grid_density=probe_grid_density,
        )
        mins.append(pmin)
        maxs.append(pmax)
        print(f"💛 probe part {i}: min={pmin}, max={pmax}")

    mins = np.asarray(mins, dtype=np.float64).reshape(-1, 3)
    maxs = np.asarray(maxs, dtype=np.float64).reshape(-1, 3)
    gmin = mins.min(axis=0)
    gmax = maxs.max(axis=0)

    # 防止穿地
    if gmin[2] < floor_clearance:
        dz = float(floor_clearance - gmin[2])
        placed_pos[2] += dz

        mins = []
        maxs = []
        for spec in mpm_specs:
            pmin, pmax = _probe_one_mpm_bound(
                gs=gs,
                spec=spec,
                placed_pos=placed_pos,
                dt=dt,
                substeps=substeps,
                probe_grid_density=probe_grid_density,
            )
            mins.append(pmin)
            maxs.append(pmax)

        mins = np.asarray(mins, dtype=np.float64).reshape(-1, 3)
        maxs = np.asarray(maxs, dtype=np.float64).reshape(-1, 3)
        gmin = mins.min(axis=0)
        gmax = maxs.max(axis=0)

    safety_pad = 10.0 / float(final_grid_density)
    margin = safety_pad + float(extra_margin)

    lower = gmin - margin
    upper = gmax + margin
    lower[2] = min(lower[2], -margin)

    return placed_pos, tuple(lower.tolist()), tuple(upper.tolist()), gmin-0.2, gmax+0.2
def _pick_runtime_friction(value: Optional[float], default_friction: float) -> float:
    return float(value if value is not None else default_friction)

def simulate_in_genesis(
    prepared: PreparedObject,
    output_root: Path,
    steps: int,
    dt: float,
    substeps: int,
    fps: int,
    default_friction: float,
    object_fixed: bool,
    striker_radius: float,
    striker_speed: float,
    args: argparse.Namespace,
) -> str:
    import genesis as gs
    gs.init()

    obj_dir = Path(prepared.output_dir)
    metadata = json.loads((obj_dir / "meta" / "metadata.json").read_text(encoding="utf-8"))

    rigid_material_cfg = _default_entity_rigid_material(metadata, default_friction=default_friction)

    bbox_min = np.asarray(metadata["object_bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(metadata["object_bbox_max"], dtype=np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
    placed_pos = np.array([0.0, 0.0, float(metadata["grounding_offset_z"]) + 0.002], dtype=np.float64)

    soft_specs = _collect_soft_specs(obj_dir=obj_dir, metadata=metadata)
    mpm_specs = [spec for spec in soft_specs if _spec_uses_mpm(spec)]
    needs_mpm = len(mpm_specs) > 0
    needs_sph = any(_spec_uses_sph(spec) for spec in soft_specs)
    needs_pbd = any(_spec_uses_pbd(spec) for spec in soft_specs)
    has_soft = len(soft_specs) > 0

    mpm_lower = None
    mpm_upper = None
    if needs_mpm:
        mpm_lower = (-2.0, -1.5, -1.0)
        mpm_upper = (2.0, 1.5, 2.0)
        print("💛 fixed MPM lower =", mpm_lower)
        print("💛 fixed MPM upper =", mpm_upper)

    sph_lower = (-2.0, -1.5, -1.0) if needs_sph else None
    sph_upper = (2.0, 1.5, 2.0) if needs_sph else None

    vis_kwargs = {
        "visualize_mpm_boundary": bool(needs_mpm),
        "visualize_sph_boundary": bool(needs_sph),
    }

    scene_kwargs = dict(
        viewer_options=gs.options.ViewerOptions(
            camera_fov=30 if has_soft else 40,
            **({} if has_soft else {
                "camera_pos": (3.5, -1.0, 2.5),
                "camera_lookat": (0.0, 0.0, 0.5),
            })
        ),
        vis_options=gs.options.VisOptions(**vis_kwargs),
    )

    if has_soft:
        scene_kwargs["sim_options"] = gs.options.SimOptions(
            dt=float(dt),
            substeps=int(substeps),
        )
    else:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(
            dt=float(dt),
        )

    if needs_mpm:
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(
            lower_bound=mpm_lower,
            upper_bound=mpm_upper,
            grid_density=48,
        )
    if needs_sph:
        scene_kwargs["sph_options"] = gs.options.SPHOptions(
            lower_bound=sph_lower,
            upper_bound=sph_upper,
            particle_size=0.01,
        )

    scene = gs.Scene(**scene_kwargs)

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(rho=1200.0, friction=0.95),
    )

    articulated_ent = None
    rigid_urdf_path = obj_dir / "rigid" / f"{prepared.object_id}.urdf"
    has_rigid_skeleton = rigid_urdf_path.exists() and (len(metadata.get("rigid_part_links", [])) > 0 or len(metadata.get("rigid_group_carriers", [])) > 0)

    if has_rigid_skeleton:
        urdf_kwargs = dict(
            file=str(rigid_urdf_path),
            scale=1.0,
            pos=tuple(placed_pos.tolist()),
            euler=(0.0, 0.0, 0.0),
            visualization=True,
            collision=True,
            fixed=bool(object_fixed),
            merge_fixed_links=False,
            prioritize_urdf_material=True,
            file_meshes_are_zup=True,
        )

        articulated_ent = scene.add_entity(
            morph=gs.morphs.URDF(**urdf_kwargs),
            material=_make_genesis_rigid_material(
                gs,
                rho=float(rigid_material_cfg["rho"]),
                friction=float(rigid_material_cfg["friction"]),
                restitution=float(rigid_material_cfg["restitution"]),
            ),
        )

    # 参考 HybridEntity 的“刚体骨架 + 软体部件”思路，但为了保留每个 part 自己的材料类型，
    # 这里仍然按 part 单独 add_entity，而不是把整件物体压成单一 material_soft。
    sph_particle_size = 0.01
    for spec in soft_specs:
        material = _make_soft_material(gs, spec)
        ctor = str(spec.get("material_ctor", ""))
        if ctor in ["gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid", "gs.materials.MPM.Sand", "gs.materials.MPM.Snow"]:
            vis_mode = "particle"
        elif ctor == "gs.materials.PBD.Cloth":
            vis_mode = "visual"
        else:
            vis_mode = "particle"

        morph = _make_soft_morph(
            gs,
            spec,
            placed_pos=placed_pos,
            metadata=metadata,
            particle_size=sph_particle_size,
        )
        scene.add_entity(
            material=material,
            morph=morph,
            surface=gs.surfaces.Default(
                color=spec["color"],
                vis_mode=vis_mode,
            ),
        )

    # striker_start = np.array(
    #     [
    #         bbox_min[0] + args.ball_posx+1.8,
    #         # bbox_min[0]+0.05,
    #         float(bbox_center[1]),
    #         bbox_min[2] + 0.65 * bbox_size[2] + placed_pos[2],
    #     ],
    #     dtype=np.float64,
    # )
    striker_start = np.array(
        [
            bbox_min[0] + args.ball_posx,
            # bbox_min[0]+0.05,
            float(bbox_center[1]),
            bbox_min[2] + 0.65 * bbox_size[2] + placed_pos[2]+0.3,
        ],
        dtype=np.float64,
    )
    striker = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=float(striker_radius),
            pos=tuple(striker_start.tolist()),
            euler=(0.0, 0.0, 0.0),
        ),
        material=gs.materials.Rigid(rho=1800.0, friction=0.35),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15, 1.0), vis_mode="visual"),
    )

    cam_distance = max(2.2, 2.0 * float(np.max(bbox_size)) + 1.0)
    cam_height = max(1.1, float(placed_pos[2] + bbox_min[2] + 0.85 * bbox_size[2] + 0.4))
    cam = scene.add_camera(
        res=(960, 720),
        pos=(cam_distance, -cam_distance, cam_height),
        lookat=(0.0, 0.0, float(placed_pos[2] + bbox_min[2] + 0.55 * bbox_size[2])),
        fov=35,
        GUI=False,
    )

    scene.build()

    corrected_pos = placed_pos.copy()
    if articulated_ent is not None:
        aabb = articulated_ent.get_AABB()
        if hasattr(aabb, "detach"):
            aabb = aabb.detach().cpu().numpy()
        else:
            aabb = np.asarray(aabb)

        z_min = float(aabb[0, 2])
        clearance = 0.002
        if abs(z_min - clearance) > 1e-6:
            corrected_pos[2] += (clearance - z_min)
            articulated_ent.set_pos(corrected_pos)

        print("placed_pos before correction:", placed_pos.tolist())
        print("AABB z_min:", z_min)
        print("placed_pos after correction:", corrected_pos.tolist())

        runtime_apply = _configure_genesis_rigid_entity_from_metadata(
            articulated_ent,
            metadata,
            default_friction=default_friction,
        )
        metadata.setdefault("runtime_application", {})
        metadata["runtime_application"]["rigid_entity"] = runtime_apply
        with open(obj_dir / "meta" / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    if needs_sph:
        warmup_steps = max(45, int(round(0.4 / max(float(dt), 1e-6))))
        settle_steps = max(30, int(round(0.25 / max(float(dt), 1e-6))))
        print(f"[Liquid warmup] pre-roll {warmup_steps} steps + settle {settle_steps} steps")
        for _ in range(warmup_steps + settle_steps):
            scene.step()


    striker.set_dofs_velocity([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])





    frames: List[np.ndarray] = []
    preview_dir = output_root / prepared.object_id / "scene_preview"
    ensure_dir(preview_dir)

    save_every = max(1, int(round((1.0 / dt) / fps)))
    for t in range(steps):
        scene.step()
        rgb = cam.render(rgb=True, depth=False, segmentation=False, normal=False)
        if isinstance(rgb, tuple):
            rgb = rgb[0]
        if t % save_every == 0:
            frames.append(np.asarray(rgb))

    video_path = preview_dir / f"preview_{args.ball_posx}.mp4"
    if frames:
        imageio.mimwrite(video_path, frames, fps=fps, quality=8)

    try:
        scene.destroy()
    except Exception:
        pass

    return str(video_path)
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one PhysXNet object into a strict-per-part Genesis-ready asset folder and optional preview simulation.")
    parser.add_argument("--physx_root", type=str, required=True, help="PhysXNet root, e.g. /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
    parser.add_argument("--version", type=str, default="version_1", help="Dataset version folder name")
    parser.add_argument("--object_id", type=str, required=True, help="Object ID, e.g. 48610")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory")
    parser.add_argument("--json_override", type=str, default=None, help="Optional explicit JSON path. If set, this JSON is used instead of finaljson/<object_id>.json")
    parser.add_argument("--voxel_pitch", type=float, default=0.025, help="Voxel size in meters for soft-part fill")
    parser.add_argument("--fallback_density_kgm3", type=float, default=800.0, help="Fallback density only for parts whose JSON lacks density")
    parser.add_argument("--default_friction", type=float, default=0.55, help="Runtime fallback friction only when JSON friction is absent")
    parser.add_argument("--run_genesis", action="store_true", help="Also run a small Genesis demo and render preview.mp4")
    parser.add_argument("--steps", type=int, default=240, help="Simulation steps for preview")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation dt")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps")
    parser.add_argument("--fps", type=int, default=24, help="Preview video fps")
    parser.add_argument("--striker_radius", type=float, default=0.08, help="Radius of the striker sphere")
    parser.add_argument("--striker_speed", type=float, default=2.8, help="+X initial speed of the striker sphere")
    parser.set_defaults(object_fixed=True)
    parser.add_argument("--dynamic_object", dest="object_fixed", action="store_false", help="Let the imported object fall instead of staying fixed on the ground")
    
    parser.add_argument(
    "--object_scale_mult",
    type=float,
    default=1.0,
    help="Extra global scale multiplier applied only to the imported object, not to the striker sphere",
)
    parser.add_argument("--ball_posx", type=float, default=0.0, help="X position of the ball relative to the striker")
    parser.add_argument("--solver_family_override", type=str, default=None, help="Override solver family for all objects")
    
    
    
    
    
    
    return parser















def main() -> None:
    args = build_argparser().parse_args()
    prepared = prepare_physxnet_object(
        physx_root=Path(args.physx_root),
        version=args.version,
        object_id=str(args.object_id),
        output_root=Path(args.output_root),
        voxel_pitch=float(args.voxel_pitch),
        json_override=Path(args.json_override) if args.json_override else None,
        object_scale_mult=float(args.object_scale_mult),
        solver_family_override=args.solver_family_override,
    )

    preview_video = None
    if args.run_genesis:
        preview_video = simulate_in_genesis(
            prepared=prepared,
            output_root=Path(args.output_root),
            steps=int(args.steps),
            dt=float(args.dt),
            substeps=int(args.substeps),
            fps=int(args.fps),
            default_friction=float(args.default_friction),
            object_fixed=bool(args.object_fixed),
            striker_radius=float(args.striker_radius),
            striker_speed=float(args.striker_speed),
            args=args,
        )

    summary = asdict(prepared)
    summary["preview_video"] = preview_video
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
'''
桌布 19925
bowl with Liquid 12093
沙发 39264


rm -r /data/gaoya/AAA_test_video/Dataset_test/physxnet_genesis_tst/19925
python /home/gaoya/Code_Video/Code_data/physxnet_articulation_demo2.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --object_id 19925 \
  --output_root /data/gaoya/AAA_test_video/Dataset_test/physxnet_genesis_tst \
  --run_genesis \
  --object_scale_mult 1.8 \
  --striker_radius 0.05 \
  
'''