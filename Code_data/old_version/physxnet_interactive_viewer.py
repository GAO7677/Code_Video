#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhysXNet articulated viewer + corrected Genesis/URDF exporter.

What this script fixes compared with the previous quick converter:
1) Joint pivots are transformed by the same global translation/scale as the meshes.
2) Each movable group's mesh is re-centered into its own link-local frame at the joint pivot.
3) A local Gradio+Plotly viewer is provided for interactive 3D inspection.
4) A corrected articulated URDF and per-group meshes are exported for Genesis.

Example:
python physxnet_interactive_viewer.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --object_id 48610 \
  --output_root /data/gaoya/AAA_test_video/Dataset_test/physxnet_view_fix \
  --host 0.0.0.0 \
  --port 8011
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

try:
    import gradio as gr
except Exception as e:  # pragma: no cover
    raise RuntimeError("This viewer requires gradio. Please `pip install gradio`.") from e

try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    raise RuntimeError("This viewer requires plotly. Please `pip install plotly`.") from e


# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


MATERIAL_PRESETS = {
    "wood": {"density_kgm3": 650.0, "friction": 0.60, "solver": "rigid"},
    "metal": {"density_kgm3": 7800.0, "friction": 0.45, "solver": "rigid"},
    "glass": {"density_kgm3": 2500.0, "friction": 0.40, "solver": "rigid"},
    "plastic": {"density_kgm3": 1100.0, "friction": 0.35, "solver": "rigid"},
    "rubber": {"density_kgm3": 1100.0, "friction": 0.90, "solver": "mpm_elastic"},
    "foam": {"density_kgm3": 80.0, "friction": 0.80, "solver": "mpm_elastic"},
    "fabric": {"density_kgm3": 300.0, "friction": 0.65, "solver": "pbd_cloth"},
    "cloth": {"density_kgm3": 250.0, "friction": 0.65, "solver": "pbd_cloth"},
    "leather": {"density_kgm3": 860.0, "friction": 0.70, "solver": "pbd_cloth"},
    "gel": {"density_kgm3": 1000.0, "friction": 0.80, "solver": "mpm_elastic"},
}

PALETTE = [
    (0.62, 0.73, 0.89),
    (0.91, 0.63, 0.42),
    (0.52, 0.80, 0.58),
    (0.91, 0.54, 0.76),
    (0.98, 0.85, 0.46),
    (0.64, 0.58, 0.90),
    (0.55, 0.83, 0.86),
    (0.88, 0.88, 0.88),
]


def color_from_index(i: int) -> Tuple[float, float, float]:
    return PALETTE[i % len(PALETTE)]


def rgb_to_plotly(c: Tuple[float, float, float], alpha: float = 1.0) -> str:
    r, g, b = [int(max(0, min(255, round(x * 255)))) for x in c]
    return f"rgba({r},{g},{b},{alpha})"


def sanitize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    mesh = mesh.copy()

    # 兼容不同 trimesh 版本
    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()

    if hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, "unique_faces") and hasattr(mesh, "update_faces"):
        try:
            mesh.update_faces(mesh.unique_faces())
        except Exception:
            pass

    if hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, "nondegenerate_faces") and hasattr(mesh, "update_faces"):
        try:
            mesh.update_faces(mesh.nondegenerate_faces())
        except Exception:
            pass

    if hasattr(mesh, "remove_unreferenced_vertices"):
        mesh.remove_unreferenced_vertices()
    if hasattr(mesh, "fix_normals"):
        mesh.fix_normals()
    return mesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path), process=False, maintain_order=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    return sanitize_mesh(mesh)


def merge_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    meshes = [sanitize_mesh(m) for m in meshes if m is not None and len(m.vertices) > 0]
    if not meshes:
        raise ValueError("No valid meshes to merge")
    return sanitize_mesh(trimesh.util.concatenate(meshes))


def parse_dimension_to_meters(dim_str: str) -> Optional[np.ndarray]:
    if not dim_str:
        return None
    cleaned = dim_str.lower().replace("cm", "").replace("mm", "").replace("m", "")
    sep = "*" if "*" in cleaned else "x" if "x" in cleaned else None
    if sep is None:
        return None
    try:
        vals = [float(x.strip()) for x in cleaned.split(sep)]
        if len(vals) != 3:
            return None
        # PhysXNet dimensions are typically in centimeters.
        return np.asarray(vals, dtype=np.float64) / 100.0
    except Exception:
        return None


def parse_density_to_kgm3(raw: Any, default: float = 1000.0) -> float:
    if raw is None:
        return default
    s = str(raw).strip().lower()
    try:
        if "g/cm^3" in s or "g/cm3" in s:
            v = float(s.split()[0])
            return v * 1000.0
        if "kg/m^3" in s or "kg/m3" in s:
            v = float(s.split()[0])
            return v
        return float(s)
    except Exception:
        return default


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_youngs_pa(x: Any, default_gpa: float = 2.0) -> float:
    if x is None:
        return default_gpa * 1e9
    try:
        return float(x) * 1e9
    except Exception:
        return default_gpa * 1e9


def infer_material_params(material_name: str) -> Dict[str, Any]:
    s = (material_name or "").lower()
    for k, v in MATERIAL_PRESETS.items():
        if k in s:
            return v.copy()
    return {"density_kgm3": 1000.0, "friction": 0.5, "solver": "rigid"}


def axis_angle_transform(axis: np.ndarray, angle_rad: float, pivot: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(axis)
    if n < 1e-8:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        axis = axis / n
    M = trimesh.transformations.rotation_matrix(angle_rad, axis, pivot)
    return M


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class PartPhysical:
    part_id: int
    name: str
    material_name: str
    density_kgm3: float
    youngs_pa: float
    poisson: float
    friction: float
    solver_family: str
    movement_desc: str
    mesh_path: str


@dataclass
class GroupRuntime:
    group_id: str
    child_labels: List[int]
    parent_group: str
    joint_type: str
    axis: List[float]
    pivot_object: List[float]
    limit: List[float]
    mesh_object_path: str
    mesh_local_path: str
    default_angle: float
    slider_min_deg: float
    slider_max_deg: float


# -----------------------------
# Parse PhysXNet object
# -----------------------------


def build_part_physical(part: Dict[str, Any], mesh_path: Path) -> PartPhysical:
    material_name = str(part.get("material", "Unknown"))
    preset = infer_material_params(material_name)
    return PartPhysical(
        part_id=int(part["label"]),
        name=str(part.get("name", f"part_{part['label']}")),
        material_name=material_name,
        density_kgm3=parse_density_to_kgm3(part.get("density"), preset["density_kgm3"]),
        youngs_pa=parse_youngs_pa(part.get("Young's Modulus (GPa)"), 2.0),
        poisson=safe_float(part.get("Poisson's Ratio"), 0.30),
        friction=float(preset["friction"]),
        solver_family=str(preset["solver"]),
        movement_desc=str(part.get("Movement_description", "")),
        mesh_path=str(mesh_path),
    )


def parse_group_info(group_info: Dict[str, Any]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {"base_group": [int(x) for x in group_info.get("0", [])], "movable_groups": []}
    for key, value in group_info.items():
        if str(key) == "0":
            continue
        if not isinstance(value, list) or len(value) < 4:
            continue
        child_labels = [int(x) for x in value[0]]
        parent_group = str(value[1])
        params = value[2] if isinstance(value[2], list) else []
        motion_code = str(value[3])
        if motion_code == "B":
            joint_type = "prismatic"
        elif motion_code == "C":
            joint_type = "revolute"
        else:
            joint_type = "fixed"
        parsed["movable_groups"].append(
            {
                "group_id": str(key),
                "child_labels": child_labels,
                "parent_group": parent_group,
                "joint_type": joint_type,
                "params": params,
                "motion_code": motion_code,
            }
        )
    return parsed


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    indent = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


# -----------------------------
# Export corrected URDF
# -----------------------------


def add_inertial(link: ET.Element, mass: float, inertia_diag: np.ndarray) -> None:
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{inertia_diag[0]:.6e}",
        ixy="0",
        ixz="0",
        iyy=f"{inertia_diag[1]:.6e}",
        iyz="0",
        izz=f"{inertia_diag[2]:.6e}",
    )


def add_mesh_visual_collision(link: ET.Element, mesh_rel_path: str, color: Tuple[float, float, float]) -> None:
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_rel_path, scale="1 1 1")
    material = ET.SubElement(visual, "material", name="mat")
    ET.SubElement(material, "color", rgba=f"{color[0]} {color[1]} {color[2]} 1")

    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    cgeom = ET.SubElement(collision, "geometry")
    ET.SubElement(cgeom, "mesh", filename=mesh_rel_path, scale="1 1 1")


def inertia_from_bbox(extents: np.ndarray, mass: float) -> np.ndarray:
    x, y, z = [max(float(v), 1e-4) for v in extents]
    ixx = mass * (y * y + z * z) / 12.0
    iyy = mass * (x * x + z * z) / 12.0
    izz = mass * (x * x + y * y) / 12.0
    return np.asarray([ixx, iyy, izz], dtype=np.float64)


def mesh_volume_fallback(mesh: trimesh.Trimesh) -> float:
    try:
        vol = float(abs(mesh.volume))
        if vol > 1e-10:
            return vol
    except Exception:
        pass
    ext = np.maximum(np.asarray(mesh.extents, dtype=np.float64), 1e-4)
    return float(np.prod(ext) * 0.15)


def add_joint(robot: ET.Element, parent: str, child: str, name: str, joint_type: str,
              pivot_obj: np.ndarray, axis: np.ndarray, lower: float, upper: float) -> None:
    joint = ET.SubElement(robot, "joint", name=name, type=joint_type)
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=child)
    ET.SubElement(joint, "origin", xyz=" ".join(f"{float(x):.8f}" for x in pivot_obj), rpy="0 0 0")
    if joint_type in {"revolute", "prismatic", "continuous"}:
        axis = np.asarray(axis, dtype=np.float64)
        n = np.linalg.norm(axis)
        if n < 1e-8:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            axis = axis / n
        ET.SubElement(joint, "axis", xyz=" ".join(f"{float(x):.8f}" for x in axis))
    if joint_type in {"revolute", "prismatic"}:
        ET.SubElement(joint, "limit", lower=f"{lower:.8f}", upper=f"{upper:.8f}", effort="30", velocity="3")


# -----------------------------
# Main preparation
# -----------------------------


def prepare_object(physx_root: Path, version: str, object_id: str, output_root: Path) -> Dict[str, Any]:
    version_root = physx_root / version
    json_path = version_root / "finaljson" / f"{object_id}.json"
    objs_dir = version_root / "partseg" / object_id / "objs"
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not objs_dir.exists():
        raise FileNotFoundError(objs_dir)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    out_dir = output_root / object_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "parts_raw")
    ensure_dir(out_dir / "parts_object")
    ensure_dir(out_dir / "rigid")
    ensure_dir(out_dir / "viewer")
    ensure_dir(out_dir / "meta")

    parsed_groups = parse_group_info(meta.get("group_info", {}))

    part_meshes_raw: Dict[int, trimesh.Trimesh] = {}
    part_meshes_obj: Dict[int, trimesh.Trimesh] = {}
    part_phys: Dict[int, PartPhysical] = {}

    for part in meta["parts"]:
        pid = int(part["label"])
        mesh_path = objs_dir / f"{pid}.obj"
        if not mesh_path.exists():
            continue
        raw = load_mesh(mesh_path)
        part_meshes_raw[pid] = raw
        part_phys[pid] = build_part_physical(part, mesh_path)
        raw.export(out_dir / "parts_raw" / f"part_{pid:03d}.obj")

    if not part_meshes_raw:
        raise ValueError(f"No valid part meshes for object {object_id}")

    merged_raw = merge_meshes(list(part_meshes_raw.values()))
    raw_center = np.asarray(merged_raw.bounding_box.centroid, dtype=np.float64)
    raw_extents = np.asarray(merged_raw.extents, dtype=np.float64)

    dim_m = parse_dimension_to_meters(str(meta.get("dimension", "")))
    if dim_m is not None:
        object_scale = float(np.max(dim_m) / max(np.max(raw_extents), 1e-8))
    else:
        object_scale = float(1.0 / max(np.max(raw_extents), 1e-8))

    def to_object_frame(xyz: np.ndarray) -> np.ndarray:
        return (np.asarray(xyz, dtype=np.float64) - raw_center) * object_scale

    # Shared object frame: translate by global center, then uniform scale.
    for pid, raw in part_meshes_raw.items():
        mesh = raw.copy()
        mesh.apply_translation(-raw_center)
        mesh.apply_scale(object_scale)
        mesh = sanitize_mesh(mesh)
        part_meshes_obj[pid] = mesh
        mesh.export(out_dir / "parts_object" / f"part_{pid:03d}.obj")

    base_labels = sorted(int(x) for x in parsed_groups.get("base_group", []))
    movable_groups = parsed_groups.get("movable_groups", [])
    movable_child_labels = sorted({int(lbl) for g in movable_groups for lbl in g["child_labels"]})
    all_labels = sorted(part_meshes_obj.keys())
    base_labels = sorted(set(base_labels).union(set(all_labels) - set(movable_child_labels)))

    robot = ET.Element("robot", attrib={"name": f"physxnet_{object_id}"})

    # base link stays in object frame
    base_mesh = merge_meshes([part_meshes_obj[pid] for pid in base_labels])
    base_mesh_path = out_dir / "rigid" / "base.obj"
    base_mesh.export(base_mesh_path)
    base_link = ET.SubElement(robot, "link", name="base_link")
    base_mass = 0.0
    for pid in base_labels:
        base_mass += mesh_volume_fallback(part_meshes_obj[pid]) * part_phys[pid].density_kgm3
    base_mass = max(base_mass, 0.5)
    add_inertial(base_link, base_mass, inertia_from_bbox(np.asarray(base_mesh.extents), base_mass))
    add_mesh_visual_collision(base_link, os.path.relpath(base_mesh_path, out_dir / "rigid"), color_from_index(0))

    runtime_groups: List[GroupRuntime] = []

    for gi, g in enumerate(movable_groups, start=1):
        child_labels = [int(x) for x in g["child_labels"] if int(x) in part_meshes_obj]
        if not child_labels:
            continue
        params = list(g.get("params", []))
        axis_raw = np.asarray(params[:3] if len(params) >= 3 else [0.0, 1.0, 0.0], dtype=np.float64)
        pivot_raw = np.asarray(params[3:6] if len(params) >= 6 else [0.0, 0.0, 0.0], dtype=np.float64)
        pivot_obj = to_object_frame(pivot_raw)
        lower = float(params[6]) if len(params) >= 7 else 0.0
        upper = float(params[7]) if len(params) >= 8 else 0.0

        # Important fix: the child link mesh must be stored in link-local coordinates centered at the joint pivot.
        group_mesh_obj = merge_meshes([part_meshes_obj[pid] for pid in child_labels])
        group_mesh_obj_path = out_dir / "rigid" / f"group_{g['group_id']}_object.obj"
        group_mesh_obj.export(group_mesh_obj_path)

        group_mesh_local = group_mesh_obj.copy()
        group_mesh_local.apply_translation(-pivot_obj)
        group_mesh_local = sanitize_mesh(group_mesh_local)
        group_mesh_local_path = out_dir / "rigid" / f"group_{g['group_id']}_local.obj"
        group_mesh_local.export(group_mesh_local_path)

        link_name = f"link_group_{g['group_id']}"
        link = ET.SubElement(robot, "link", name=link_name)
        mass = 0.0
        for pid in child_labels:
            mass += mesh_volume_fallback(part_meshes_obj[pid]) * part_phys[pid].density_kgm3
        mass = max(mass, 0.1)
        add_inertial(link, mass, inertia_from_bbox(np.asarray(group_mesh_local.extents), mass))
        add_mesh_visual_collision(link, os.path.relpath(group_mesh_local_path, out_dir / "rigid"), color_from_index(gi))
        add_joint(
            robot=robot,
            parent="base_link" if g["parent_group"] == "0" else "base_link",
            child=link_name,
            name=f"joint_{g['group_id']}",
            joint_type=g["joint_type"],
            pivot_obj=pivot_obj,
            axis=axis_raw,
            lower=lower,
            upper=upper,
        )

        default_angle = 0.0
        if g["joint_type"] == "revolute":
            default_angle = 0.0
            slider_min_deg = math.degrees(min(lower, upper))
            slider_max_deg = math.degrees(max(lower, upper))
            if abs(slider_min_deg - slider_max_deg) < 1e-6:
                slider_min_deg, slider_max_deg = -90.0, 90.0
        elif g["joint_type"] == "prismatic":
            slider_min_deg = min(lower, upper)
            slider_max_deg = max(lower, upper)
        else:
            slider_min_deg, slider_max_deg = 0.0, 0.0

        runtime_groups.append(
            GroupRuntime(
                group_id=str(g["group_id"]),
                child_labels=child_labels,
                parent_group=str(g["parent_group"]),
                joint_type=str(g["joint_type"]),
                axis=axis_raw.tolist(),
                pivot_object=pivot_obj.tolist(),
                limit=[float(lower), float(upper)],
                mesh_object_path=str(group_mesh_obj_path),
                mesh_local_path=str(group_mesh_local_path),
                default_angle=float(default_angle),
                slider_min_deg=float(slider_min_deg),
                slider_max_deg=float(slider_max_deg),
            )
        )

    urdf_path = out_dir / "rigid" / f"{object_id}.urdf"
    indent_xml(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)

    metadata = {
        "object_id": object_id,
        "object_name": meta.get("object_name", object_id),
        "category": meta.get("category", "Unknown"),
        "dimension_raw": meta.get("dimension", None),
        "dimension_m": dim_m.tolist() if dim_m is not None else None,
        "global_transform": {
            "raw_center": raw_center.tolist(),
            "uniform_scale": object_scale,
        },
        "base_labels": base_labels,
        "runtime_groups": [asdict(g) for g in runtime_groups],
        "parts_physical": {str(pid): asdict(pp) for pid, pp in part_phys.items()},
        "notes": {
            "assembly_fix": "child group mesh exported in joint-local frame; joint pivot transformed to shared object frame",
            "urdf_path": str(urdf_path),
        },
    }
    with open(out_dir / "meta" / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return {
        "meta": meta,
        "out_dir": out_dir,
        "urdf_path": urdf_path,
        "base_mesh": base_mesh,
        "base_labels": base_labels,
        "runtime_groups": runtime_groups,
        "part_meshes_obj": part_meshes_obj,
        "part_phys": part_phys,
        "object_scale": object_scale,
        "raw_center": raw_center,
    }


# -----------------------------
# Interactive viewer
# -----------------------------


def make_mesh_trace(mesh: trimesh.Trimesh, color: Tuple[float, float, float], name: str, opacity: float = 1.0) -> go.Mesh3d:
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=rgb_to_plotly(color, opacity),
        name=name,
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.7, fresnel=0.08, specular=0.15, roughness=0.8),
        hovertext=name,
        hoverinfo="text",
    )


def posed_group_mesh(group: GroupRuntime, angle_value: float) -> trimesh.Trimesh:
    mesh = load_mesh(Path(group.mesh_object_path))
    axis = np.asarray(group.axis, dtype=np.float64)
    pivot = np.asarray(group.pivot_object, dtype=np.float64)
    if group.joint_type == "revolute":
        M = axis_angle_transform(axis=axis, angle_rad=math.radians(float(angle_value)), pivot=pivot)
        mesh.apply_transform(M)
    elif group.joint_type == "prismatic":
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        mesh.apply_translation(axis * float(angle_value))
    return mesh


def build_figure(prepared: Dict[str, Any], angles_deg: List[float], show_base: bool = True, opacity: float = 1.0) -> go.Figure:
    fig = go.Figure()
    if show_base:
        fig.add_trace(make_mesh_trace(prepared["base_mesh"], color_from_index(0), "base", opacity=opacity))

    for idx, group in enumerate(prepared["runtime_groups"], start=1):
        angle = angles_deg[idx - 1] if idx - 1 < len(angles_deg) else group.default_angle
        mesh = posed_group_mesh(group, angle)
        label = f"group {group.group_id} | {group.joint_type} | {angle:.1f}"
        fig.add_trace(make_mesh_trace(mesh, color_from_index(idx), label, opacity=opacity))

        pivot = np.asarray(group.pivot_object, dtype=np.float64)
        fig.add_trace(
            go.Scatter3d(
                x=[pivot[0]], y=[pivot[1]], z=[pivot[2]],
                mode="markers",
                marker=dict(size=4, color="red"),
                name=f"pivot {group.group_id}",
                hovertext=f"pivot {group.group_id}",
                hoverinfo="text",
            )
        )

    all_meshes = [prepared["base_mesh"]] + [posed_group_mesh(g, angles_deg[i] if i < len(angles_deg) else g.default_angle) for i, g in enumerate(prepared["runtime_groups"])]
    merged = merge_meshes(all_meshes)
    bounds = merged.bounds
    center = bounds.mean(axis=0)
    size = max(float((bounds[1] - bounds[0]).max()), 1e-3)

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True, range=[center[0] - size * 0.65, center[0] + size * 0.65]),
            yaxis=dict(visible=True, range=[center[1] - size * 0.65, center[1] + size * 0.65]),
            zaxis=dict(visible=True, range=[center[2] - size * 0.65, center[2] + size * 0.65]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.8, z=1.1)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"PhysXNet articulated viewer | {prepared['meta'].get('object_name', prepared['meta'].get('category', 'object'))}",
        showlegend=True,
    )
    return fig


def launch_app(prepared: Dict[str, Any], host: str, port: int) -> None:
    groups: List[GroupRuntime] = prepared["runtime_groups"]
    max_groups = len(groups)

    with gr.Blocks(title="PhysXNet articulated viewer") as demo:
        gr.Markdown(
            f"## PhysXNet 交互式 3D 检查\n"
            f"- object_id: `{prepared['meta'].get('object_name', '')}` / `{prepared['meta'].get('category', '')}`\n"
            f"- base parts: `{prepared['base_labels']}`\n"
            f"- movable groups: `{[g.group_id for g in groups]}`\n"
            f"- corrected URDF: `{prepared['urdf_path']}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                show_base = gr.Checkbox(value=True, label="显示 base")
                opacity = gr.Slider(0.2, 1.0, value=1.0, step=0.05, label="整体透明度")
                sliders = []
                for g in groups:
                    label = f"group {g.group_id} | {g.joint_type} | labels={g.child_labels}"
                    sliders.append(
                        gr.Slider(
                            minimum=g.slider_min_deg,
                            maximum=g.slider_max_deg,
                            value=g.default_angle,
                            step=1.0,
                            label=label,
                        )
                    )
                reset_btn = gr.Button("重置姿态")
                export_btn = gr.Button("导出当前姿态 OBJ")
                export_info = gr.Textbox(label="导出信息", interactive=False)
            with gr.Column(scale=3):
                plot = gr.Plot(label="3D 模型")

        def _render(*vals):
            vals = list(vals)
            sb = bool(vals[0])
            op = float(vals[1])
            angles = [float(x) for x in vals[2:]]
            return build_figure(prepared, angles_deg=angles, show_base=sb, opacity=op)

        inputs = [show_base, opacity] + sliders
        for comp in inputs:
            comp.change(_render, inputs=inputs, outputs=plot)

        def _reset():
            out = [True, 1.0]
            out.extend([g.default_angle for g in groups])
            return out

        reset_btn.click(_reset, inputs=None, outputs=inputs)

        def _export_pose(show_base_val: bool, opacity_val: float, *angles: float):
            angles = [float(x) for x in angles]
            export_dir = Path(prepared["out_dir"]) / "viewer" / "posed_exports"
            ensure_dir(export_dir)
            meshes = []
            if bool(show_base_val):
                meshes.append(prepared["base_mesh"].copy())
            for idx, g in enumerate(groups):
                ang = angles[idx] if idx < len(angles) else g.default_angle
                meshes.append(posed_group_mesh(g, ang))
            merged = merge_meshes(meshes)
            out_path = export_dir / ("posed_" + "_".join(f"g{g.group_id}_{angles[i]:.0f}" for i, g in enumerate(groups)) + ".obj")
            merged.export(out_path)
            return f"已导出: {out_path}"

        export_btn.click(_export_pose, inputs=inputs, outputs=export_info)
        demo.load(lambda: build_figure(prepared, [g.default_angle for g in groups], True, 1.0), inputs=None, outputs=plot)

    demo.launch(server_name=host, server_port=port, share=False, inbrowser=False)


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhysXNet corrected assembly viewer for Genesis export")
    parser.add_argument("--physx_root", type=str, required=True)
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--object_id", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8011)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_object(
        physx_root=Path(args.physx_root),
        version=args.version,
        object_id=str(args.object_id),
        output_root=Path(args.output_root),
    )
    print(f"[OK] corrected asset exported to: {prepared['out_dir']}")
    print(f"[OK] URDF: {prepared['urdf_path']}")
    print(f"[OK] launching interactive viewer on {args.host}:{args.port}")
    launch_app(prepared=prepared, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


'''


CUDA_VISIBLE_DEVICES=0 python /home/gaoya/Code_Video/Code_data/physxnet_interactive_viewer.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --object_id 39264 \
  --output_root /data/gaoya/AAA_test_video/Dataset_test/physxnet_view_fix \
  --host 0.0.0.0 \
  --port 8011



  
'''