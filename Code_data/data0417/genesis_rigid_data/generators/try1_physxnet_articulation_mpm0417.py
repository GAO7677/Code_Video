#!/usr/bin/env python3
# 用途：主 rigid/mpm 数据生成与导出后端。
# -*- coding: utf-8 -*-
"""
该脚本用于把 PhysXNet 对象转换并生成 Genesis 可用的刚体/MPM 数据样本；输入为 /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet 的对象 JSON/网格与生成参数，输出为 output_root 下的样本目录、缓存资产、视频和物理标注。
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
from collections import deque
import copy
import contextlib
import io
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass, asdict, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__)))
_sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
_sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..")))
from dataset_3_utils_dataset import build_urdf_from_json_file

import imageio.v2 as imageio
import numpy as np
import trimesh

from genesis_energy_utils import particle_entity_kinematic_snapshot, rigid_entity_kinematic_snapshot
from core.utils_io import depth_to_vis, save_video as save_vis_video


ASSET_CACHE_DIRNAME = "_asset_cache"
PHYSXNET_OBJECT_CACHE_DIRNAME = "physxnet_objects"
CUSTOM_OBJECT_CACHE_DIRNAME = "custom_object_asset_cache"


def asset_cache_root(output_root: Path) -> Path:
    return Path(output_root) / ASSET_CACHE_DIRNAME


def physxnet_object_cache_root(output_root: Path) -> Path:
    return asset_cache_root(output_root) / PHYSXNET_OBJECT_CACHE_DIRNAME


def custom_object_cache_root(output_root: Path) -> Path:
    return asset_cache_root(output_root) / CUSTOM_OBJECT_CACHE_DIRNAME


def physxnet_object_cache_dir(output_root: Path, object_id: str) -> Path:
    return physxnet_object_cache_root(output_root) / Path(str(object_id)).stem

class _LineFilterStream(io.TextIOBase):
    def __init__(self, target: io.TextIOBase, *, allow_prefixes: Optional[Sequence[str]] = None, drop_substrings: Optional[Sequence[str]] = None, passthrough_all: bool = False):
        self._target = target
        self._allow_prefixes = tuple(allow_prefixes or ())
        self._drop_substrings = tuple(drop_substrings or ())
        self._passthrough_all = bool(passthrough_all)
        self._buffer = ""

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line + "\n")
        return len(s)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
            self._buffer = ""
        self._target.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._target, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self._target, "encoding", "utf-8")

    def _emit_line(self, line: str) -> None:
        if any(token in line for token in self._drop_substrings):
            return
        if self._passthrough_all:
            self._target.write(line)
            return
        if self._allow_prefixes and any(line.startswith(prefix) for prefix in self._allow_prefixes):
            self._target.write(line)


@contextlib.contextmanager
def _quiet_terminal_output(enabled: bool = True):
    if not enabled:
        yield
        return
    stdout_filter = _LineFilterStream(
        _sys.stdout,
        allow_prefixes=("GENERATED ", "SKIP "),
        drop_substrings=("[INFO]", "[CoACD][info]"),
        passthrough_all=False,
    )
    stderr_filter = _LineFilterStream(
        _sys.stderr,
        allow_prefixes=(),
        drop_substrings=("[INFO]", "[CoACD][info]"),
        passthrough_all=True,
    )
    with contextlib.redirect_stdout(stdout_filter), contextlib.redirect_stderr(stderr_filter):
        yield
        stdout_filter.flush()
        stderr_filter.flush()

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
    material = norm_text(part.get("material"))
    text = material
    E_gpa = parse_float(part.get("Young's Modulus (GPa)"))

    if has_any(text, HARD_KWS):
        return {
            "solver_family": "rigid",
            "material_ctor": "gs.materials.Rigid",
            "reason": "rigid material keyword matched from material field"
        }

    if has_any(text, LIQUID_KWS):
        return {
            "solver_family": "sph",
            "material_ctor": "gs.materials.SPH.Liquid",
            "reason": "liquid keyword matched from material field"
        }

    if has_any(text, GRANULAR_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.Sand",
            "reason": "granular keyword matched from material field"
        }

    if has_any(text, SNOW_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.Snow",
            "reason": "snow keyword matched from material field"
        }

    # 关键：先判软体，再判 cloth
    if has_any(text, SOFT_ELASTIC_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.Elastic",
            "reason": "soft elastic keyword matched from material field"
        }

    if has_any(text, CLOTH_BASE_KWS):
        return {
            "solver_family": "pbd",
            "material_ctor": "gs.materials.PBD.Cloth",
            "reason": "cloth-like material keyword matched from material field"
        }

    if has_any(text, SOFT_PLASTIC_KWS):
        return {
            "solver_family": "mpm",
            "material_ctor": "gs.materials.MPM.ElastoPlastic",
            "reason": "soft plastic keyword matched from material field"
        }

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

    return {
        "solver_family": "rigid",
        "material_ctor": "gs.materials.Rigid",
        "reason": "default fallback"
    }

def movement_is_support_fixed(desc: Any) -> bool:
    s = norm_text(desc)
    if not s:
        return False
    prefixes = [
        "fixed to ",
        "fixed onto ",
        "rigidly fixed to ",
        "rigidly connected to ",
        "attached to ",
        "secured to ",
        "mounted to ",
        "integrated into ",
        "integrated with ",
    ]
    return any(p in s for p in prefixes)


def classify_assembly_role(material_ctor: str, movement_desc: Any) -> str:
    if str(material_ctor) == "gs.materials.Rigid":
        return "rigid_skeleton"
    if movement_is_support_fixed(movement_desc):
        return "anchored_soft"
    return "free_soft"





# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_numpy(x: Any) -> np.ndarray:
    return _to_numpy(x)


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


def stable_int_from_text(text: Any) -> int:
    s = str(text)
    acc = 0
    for ch in s:
        acc = ((acc * 131) + ord(ch)) % 2147483647
    return int(acc)


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


def rgb_to_uint8(rgb: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(rgb))
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size > 0 and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0.0, 255.0).astype(np.uint8)


def normalize_depth_map(depth: Any, near: float, far: float) -> np.ndarray:
    arr = np.asarray(to_numpy(depth), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
    near_val = float(near)
    far_val = max(float(far), near_val + 1e-6)
    norm = np.zeros(arr.shape + (1,), dtype=np.float32)
    valid = np.isfinite(arr) & (arr > 0)
    norm[..., 0][valid] = np.clip((arr[valid] - near_val) / (far_val - near_val), 0.0, 1.0)
    return norm


def metric_depth_map(depth: Any) -> np.ndarray:
    arr = np.asarray(to_numpy(depth), dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def depth_to_uint8(depth_norm: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth_norm, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


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


def _rotate_vec_by_euler_deg(vec: Any, euler_deg: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(3)
    euler = np.asarray(euler_deg, dtype=np.float64).reshape(3)
    if np.all(np.abs(euler) <= 1e-9):
        return arr.copy()
    rot = trimesh.transformations.euler_matrix(
        math.radians(float(euler[0])),
        math.radians(float(euler[1])),
        math.radians(float(euler[2])),
        axes="sxyz",
    )[:3, :3]
    return rot @ arr


def _compose_euler_deg_xyz(base_euler_deg: Any, delta_euler_deg: Any) -> np.ndarray:
    base = np.asarray(base_euler_deg, dtype=np.float64).reshape(3)
    delta = np.asarray(delta_euler_deg, dtype=np.float64).reshape(3)
    return base + delta


CUSTOM_OBJECT_SOURCE_ALIASES = {
    "physx": "physxnet",
    "physxnet": "physxnet",
    "physx3d": "physxnet",
    "sophy": "sophy",
    "primitive": "primitive",
}

DEFAULT_SOPHY_SOURCE_ROOTS = [
    Path("/data/gaoya/dataset/SOPHY_data/bag"),
    Path("/data/gaoya/dataset/SOPHY_data/teddy_bear"),
]

CUSTOM_PRIMITIVE_MATERIAL_PRESETS = {
    "foam": {"rho": 120.0, "friction": 0.75, "youngs": 8.0e4, "poisson": 0.18, "ctor": "gs.materials.MPM.Elastic"},
    "rubber": {"rho": 980.0, "friction": 0.90, "youngs": 1.6e5, "poisson": 0.22, "ctor": "gs.materials.MPM.Elastic"},
    "plastic": {"rho": 1150.0, "friction": 0.40, "youngs": 3.5e5, "poisson": 0.28, "ctor": "gs.materials.MPM.ElastoPlastic"},
    "sand": {"rho": 1500.0, "friction": 0.55, "youngs": 1.0e5, "poisson": 0.20, "ctor": "gs.materials.MPM.Sand"},
}


def _find_candidate_mesh(sample_dir: Path) -> Optional[Path]:
    material_obj = sample_dir / "material.obj"
    if material_obj.exists():
        return material_obj
    obj_files = sorted(sample_dir.glob("*.obj"))
    if obj_files:
        return obj_files[0]
    return None


def _try_find_material_json(sample_dir: Path) -> Optional[Path]:
    for name in [
        "mat_params_new_v3.4.json",
        "mat_params_new.json",
        "mat_params.json",
        "material_params.json",
        "material.json",
    ]:
        path = sample_dir / name
        if path.exists():
            return path
    return None


def _custom_asset_safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))[:120]


def _normalize_custom_mesh_to_cache(mesh: trimesh.Trimesh, out_path: Path) -> Tuple[Path, np.ndarray]:
    ensure_dir(out_path.parent)
    mesh = sanitize_mesh(mesh)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(~np.isfinite(extents)) or float(np.max(extents)) < 1e-8:
        raise ValueError(f"invalid custom mesh extents: {extents}")
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / max(float(np.max(extents)), 1e-8))
    mesh = sanitize_mesh(mesh)
    mesh.export(out_path)
    return out_path, np.asarray(mesh.extents, dtype=np.float64)


def _coerce_custom_object_material_ctor(ctor: str) -> str:
    ctor = str(ctor or "gs.materials.MPM.Elastic")
    if ctor in ("gs.materials.MPM.Sand", "gs.materials.MPM.Snow", "gs.materials.MPM.ElastoPlastic", "gs.materials.MPM.Liquid"):
        return ctor
    return "gs.materials.MPM.Elastic"


def _choose_custom_runtime_solver(
    *,
    material_ctor: str,
    youngs: Optional[float],
    rigidify_youngs_threshold_pa: Optional[float],
) -> Tuple[str, str]:
    source_ctor = str(material_ctor or "gs.materials.MPM.Elastic")
    if rigidify_youngs_threshold_pa is not None and youngs is not None and float(youngs) >= float(rigidify_youngs_threshold_pa):
        return "rigid_approx", "gs.materials.Rigid"
    return "mpm", source_ctor


def _build_custom_primitive_asset_bank(cache_root: Path) -> List[Dict[str, Any]]:
    ensure_dir(cache_root)
    rng = np.random.RandomState(20260415)
    shape_builders = {
        "box": lambda: trimesh.creation.box(extents=rng.uniform([0.12, 0.10, 0.08], [0.26, 0.22, 0.18])),
        "sphere": lambda: trimesh.creation.icosphere(subdivisions=2, radius=float(rng.uniform(0.06, 0.11))),
        "cylinder": lambda: trimesh.creation.cylinder(radius=float(rng.uniform(0.05, 0.09)), height=float(rng.uniform(0.12, 0.24)), sections=32),
        "capsule": lambda: trimesh.creation.capsule(radius=float(rng.uniform(0.04, 0.08)), height=float(rng.uniform(0.10, 0.18)), count=[16, 16]),
    }
    bank: List[Dict[str, Any]] = []
    for shape_name, builder in shape_builders.items():
        for mat_name, mat_cfg in CUSTOM_PRIMITIVE_MATERIAL_PRESETS.items():
            asset_id = f"primitive__{shape_name}__{mat_name}"
            cache_path = cache_root / f"{asset_id}.obj"
            if not cache_path.exists():
                mesh = builder()
                _normalize_custom_mesh_to_cache(mesh, cache_path)
            color = rng.uniform(0.20, 0.95, size=3).tolist() + [1.0]
            bank.append(
                {
                    "asset_id": asset_id,
                    "source_kind": "primitive",
                    "source_label": "primitive",
                    "display_id": asset_id,
                    "mesh_path": str(cache_path),
                    "scale_range": [0.14, 0.28],
                    "material_ctor": str(mat_cfg["ctor"]),
                    "density": float(mat_cfg["rho"]),
                    "youngs": float(mat_cfg["youngs"]),
                    "poisson": float(mat_cfg["poisson"]),
                    "friction": float(mat_cfg["friction"]),
                    "color_rgba": color,
                    "shape_name": shape_name,
                    "material_name": mat_name,
                }
            )
    return bank


def _build_custom_sophy_asset_bank(cache_root: Path) -> List[Dict[str, Any]]:
    ensure_dir(cache_root)
    bank: List[Dict[str, Any]] = []
    for dataset_root in DEFAULT_SOPHY_SOURCE_ROOTS:
        if not dataset_root.exists():
            continue
        sample_dirs = [p for p in sorted(dataset_root.iterdir()) if p.is_dir()]
        for sample_dir in sample_dirs[:32]:
            mesh_path = _find_candidate_mesh(sample_dir)
            mat_json = _try_find_material_json(sample_dir)
            if mesh_path is None or mat_json is None:
                continue
            try:
                mat_data = json.loads(mat_json.read_text(encoding="utf-8"))
                mat_items = [dict(v) for v in mat_data.values() if isinstance(v, dict)]
                if not mat_items:
                    continue
                merged_mesh = load_mesh(mesh_path)
                merged_mesh = yup_to_zup_mesh(merged_mesh)
                asset_id = f"sophy__{dataset_root.name}__{sample_dir.name}"
                cache_path = cache_root / f"{_custom_asset_safe_name(asset_id)}.obj"
                if not cache_path.exists():
                    _normalize_custom_mesh_to_cache(merged_mesh, cache_path)
                density_vals = [parse_density_to_kgm3(item.get("rho"), None) for item in mat_items]
                density_vals = [float(v) for v in density_vals if v is not None and v > 0]
                youngs_vals = [safe_optional_float(item.get("E")) for item in mat_items]
                youngs_vals = [float(v) for v in youngs_vals if v is not None and v > 0]
                poisson_vals = [safe_optional_float(item.get("nu")) for item in mat_items]
                poisson_vals = [float(v) for v in poisson_vals if v is not None and v > 0]
                if not density_vals or not youngs_vals or not poisson_vals:
                    continue
                sample_part = {"material": " ".join(str(item.get("mat_name", "")) for item in mat_items), "Young's Modulus (GPa)": None}
                if youngs_vals:
                    sample_part["Young's Modulus (GPa)"] = max(np.mean(youngs_vals) / 1e9, 1e-6)
                choice = choose_genesis_material_type(sample_part)
                bank.append(
                    {
                        "asset_id": asset_id,
                        "source_kind": "sophy",
                        "source_label": dataset_root.name,
                        "display_id": sample_dir.name,
                        "mesh_path": str(cache_path),
                        "scale_range": [0.15, 0.32],
                        "material_ctor": _coerce_custom_object_material_ctor(choice.get("material_ctor", "gs.materials.MPM.Elastic")),
                        "density": float(np.mean(density_vals)),
                        "youngs": float(np.mean(youngs_vals)),
                        "poisson": float(np.clip(np.mean(poisson_vals), 0.05, 0.35)),
                        "friction": 0.55,
                        "color_rgba": [0.72, 0.68, 0.95, 1.0],
                        "material_name": str(mat_items[0].get("mat_name", "sophy")),
                        "strict_dataset_params": True,
                    }
                )
            except Exception:
                continue
    return bank


def _build_custom_physxnet_asset_bank(args: argparse.Namespace, prepared: "PreparedObject", cache_root: Path) -> List[Dict[str, Any]]:
    ensure_dir(cache_root)
    physx_root = Path(str(getattr(args, "physx_root", "")))
    version = str(getattr(args, "version", "version_1"))
    finaljson_dir = physx_root / version / "finaljson"
    partseg_dir = physx_root / version / "partseg"
    if not finaljson_dir.exists() or not partseg_dir.exists():
        return []
    object_ids = [p.stem for p in sorted(finaljson_dir.glob("*.json")) if p.stem != str(prepared.object_id)]
    if not object_ids:
        return []
    rng = np.random.RandomState(int(getattr(args, "case_seed", 20260414)) + stable_int_from_text(prepared.object_id) + 909)
    if len(object_ids) > 24:
        pick = rng.choice(len(object_ids), size=24, replace=False)
        object_ids = [object_ids[int(i)] for i in sorted(pick.tolist())]
    bank: List[Dict[str, Any]] = []
    for obj_id in object_ids:
        meta_path = finaljson_dir / f"{obj_id}.json"
        objs_dir = partseg_dir / obj_id / "objs"
        if not meta_path.exists() or not objs_dir.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            part_meshes: List[trimesh.Trimesh] = []
            parts = meta.get("parts", [])
            for part in parts:
                pid = int(part.get("label", -1))
                mesh_path = objs_dir / f"{pid}.obj"
                if not mesh_path.exists():
                    continue
                part_meshes.append(yup_to_zup_mesh(load_mesh(mesh_path)))
            if not part_meshes:
                continue
            merged_mesh = merge_meshes(part_meshes)
            asset_id = f"physxnet__{obj_id}"
            cache_path = cache_root / f"{asset_id}.obj"
            if not cache_path.exists():
                _normalize_custom_mesh_to_cache(merged_mesh, cache_path)
            densities = []
            youngs = []
            poissons = []
            material_text = []
            for part in parts:
                rho = parse_density_to_kgm3(part.get("density"), None)
                if rho is not None and rho > 0:
                    densities.append(float(rho))
                y = parse_modulus_to_pa(part.get("Young's Modulus (GPa)"), "Young's Modulus (GPa)", None)
                if y is not None and y > 0:
                    youngs.append(float(y))
                nu = safe_optional_float(part.get("Poisson's Ratio"))
                if nu is not None and nu > 0:
                    poissons.append(float(nu))
                material_text.append(str(part.get("material", "")))
            if not densities or not youngs or not poissons:
                continue
            choice = choose_genesis_material_type({"material": " ".join(material_text), "Young's Modulus (GPa)": None})
            bank.append(
                {
                    "asset_id": asset_id,
                    "source_kind": "physxnet",
                    "source_label": "physxnet",
                    "display_id": obj_id,
                    "mesh_path": str(cache_path),
                    "scale_range": [0.15, 0.30],
                    "material_ctor": _coerce_custom_object_material_ctor(choice.get("material_ctor", "gs.materials.MPM.Elastic")),
                    "density": float(np.mean(densities)),
                    "youngs": float(np.mean(youngs)),
                    "poisson": float(np.clip(np.mean(poissons), 0.05, 0.35)),
                    "friction": 0.55,
                    "color_rgba": [0.85, 0.62, 0.28, 1.0],
                    "material_name": str(meta.get("category", "physxnet")),
                    "strict_dataset_params": True,
                }
            )
        except Exception:
            continue
    return bank


def _build_custom_object_asset_bank(
    args: argparse.Namespace,
    prepared: "PreparedObject",
    output_root: Path,
) -> List[Dict[str, Any]]:
    cache_root = custom_object_cache_root(output_root)
    mix_raw = str(getattr(args, "custom_object_source_mix", "primitive,sophy,physxnet") or "").split(",")
    source_mix = [CUSTOM_OBJECT_SOURCE_ALIASES.get(x.strip().lower()) for x in mix_raw if x.strip()]
    source_mix = [x for x in source_mix if x]
    if not source_mix:
        source_mix = ["primitive"]
    bank: List[Dict[str, Any]] = []
    if "primitive" in source_mix:
        bank.extend(_build_custom_primitive_asset_bank(cache_root / "primitive"))
    if "sophy" in source_mix:
        bank.extend(_build_custom_sophy_asset_bank(cache_root / "sophy"))
    if "physxnet" in source_mix:
        bank.extend(_build_custom_physxnet_asset_bank(args, prepared, cache_root / "physxnet"))
    return bank


def metadata_is_physxnet(metadata: Dict[str, Any]) -> bool:
    json_path = str(metadata.get("json_path_used", "") or "")
    if "physxnet" in json_path.lower():
        return True
    return "group_info" in metadata and "rigid_part_links" in metadata


def build_preview_case_configs(
    prepared: "PreparedObject",
    output_root: Path,
    object_fixed: bool,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    obj_dir = Path(prepared.output_dir)
    metadata_path = obj_dir / "meta" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    bbox_min = np.asarray(metadata.get("object_bbox_min", prepared.object_bbox_min), dtype=np.float64)
    bbox_max = np.asarray(metadata.get("object_bbox_max", prepared.object_bbox_max), dtype=np.float64)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
    bbox_volume_est_m3 = float(np.prod(bbox_size))
    is_physxnet_object = metadata_is_physxnet(metadata)
    rigid_material_defaults = _default_entity_rigid_material(
        metadata,
        default_friction=float(getattr(args, "default_friction", 0.55) or 0.55),
    )

    num_cases = max(1, int(getattr(args, "num_random_cases", 1) or 1))
    base_seed = int(getattr(args, "case_seed", 20260414))
    case_scene_mode = str(getattr(args, "case_scene_mode", "auto") or "auto").strip().lower()
    simple_case_resample_index = int(getattr(args, "simple_case_resample_index", 0) or 0)
    numeric_object_id = "".join(ch for ch in str(prepared.object_id) if ch.isdigit())
    object_seed = int(numeric_object_id) if numeric_object_id else stable_int_from_text(prepared.object_id)
    seed_anchor = base_seed + object_seed * 17

    def _case_seed(case_idx: int) -> int:
        return seed_anchor + case_idx * 9973 + 23 + simple_case_resample_index * 104729

    volume_threshold_m3 = float(getattr(args, "physxnet_volume_threshold_m3", 0.20) or 0.20)
    entry_prob = float(np.clip(getattr(args, "physxnet_entry_velocity_prob", 0.35), 0.0, 1.0))
    speed_min = max(0.0, float(getattr(args, "physxnet_entry_speed_min", 0.60) or 0.60))
    speed_max = max(speed_min, float(getattr(args, "physxnet_entry_speed_max", 0.90) or 0.90))

    moving_allowed = bool(
        is_physxnet_object
        and volume_threshold_m3 > 0.0
        and bbox_volume_est_m3 < volume_threshold_m3
    )
    force_static_physxnet = bool(
        is_physxnet_object
        and volume_threshold_m3 > 0.0
        and bbox_volume_est_m3 >= volume_threshold_m3
    )

    move_flags = [False] * num_cases
    if moving_allowed and entry_prob > 0.0:
        rng_flags = np.random.RandomState(seed_anchor + 97)
        move_flags = [bool(rng_flags.rand() < entry_prob) for _ in range(num_cases)]
        if not any(move_flags):
            move_flags[int(rng_flags.randint(0, num_cases))] = True

    case_configs: List[Dict[str, Any]] = []
    base_y_offset = float(getattr(args, "debug_soft_spread_y_offset", 0.85) or 0.85)
    custom_asset_bank = _build_custom_object_asset_bank(args=args, prepared=prepared, output_root=output_root)

    def _fmt_vec3(vec: Any) -> str:
        arr = np.asarray(vec, dtype=np.float64).reshape(3)
        return "[" + ", ".join(f"{v:.3f}" for v in arr.tolist()) + "]"

    def _fmt_rgba(vec: Any) -> str:
        arr = np.asarray(vec, dtype=np.float64).reshape(4)
        return "[" + ", ".join(f"{v:.2f}" for v in arr.tolist()) + "]"

    def _fmt_scalar(value: Any, digits: int = 3) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except Exception:
            return "nan"

    counterfactual_case_id_map = {
        "same_scene_negative": 11001,
        "no_collision_negative": 11002,
    }

    def _rotate_xy(vec: Any, yaw_deg: float) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64).reshape(3)
        theta = math.radians(float(yaw_deg))
        rot = np.array(
            [
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta),  math.cos(theta)],
            ],
            dtype=np.float64,
        )
        out = arr.copy()
        out[:2] = rot @ out[:2]
        return out

    def _to_velocity6(vec: Any) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float64).reshape(-1)
        if arr.size == 3:
            arr = np.concatenate([arr, np.zeros(3, dtype=np.float64)], axis=0)
        if arr.size != 6:
            raise ValueError(f"Expected velocity with 3 or 6 values, got shape={arr.shape}")
        return arr.astype(np.float64)

    def _sample_object_yaw_deg(case_seed: int) -> float:
        yaw_min_raw = getattr(args, "physxnet_object_yaw_deg_min", -180.0)
        yaw_max_raw = getattr(args, "physxnet_object_yaw_deg_max", 180.0)
        yaw_min = -180.0 if yaw_min_raw is None else float(yaw_min_raw)
        yaw_max = 180.0 if yaw_max_raw is None else float(yaw_max_raw)
        if yaw_max < yaw_min:
            yaw_min, yaw_max = yaw_max, yaw_min
        if abs(yaw_max - yaw_min) <= 1e-8:
            return float(yaw_min)
        rng = np.random.RandomState(case_seed + 1701)
        return float(rng.uniform(yaw_min, yaw_max))

    def _sample_entry_velocity(case_seed: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
        # Reproducible per-case random speeds. The mode fixes only direction;
        # magnitude, lateral drift, and yaw rate vary within configured ranges.
        rng = np.random.RandomState(case_seed + 2609)
        speed = float(rng.uniform(speed_min, speed_max))
        lateral = float(rng.uniform(-0.24, 0.24))
        yaw_speed = float(rng.uniform(-1.45, 1.45))
        mode = str(mode)
        if mode == "entry_left":
            linear = np.array([-speed, lateral, 0.0], dtype=np.float64)
        elif mode == "entry_right":
            linear = np.array([-speed, -lateral, 0.0], dtype=np.float64)
        elif mode == "entry_fast_center":
            fast_scale = float(rng.uniform(1.08, 1.30))
            linear = np.array([-speed * fast_scale, float(rng.uniform(-0.06, 0.06)), 0.0], dtype=np.float64)
            yaw_speed *= float(rng.uniform(0.8, 1.25))
        else:
            linear = np.array([-speed, lateral, 0.0], dtype=np.float64)
        angular = np.array([0.0, 0.0, yaw_speed], dtype=np.float64)
        return linear, angular

    def _random_speed_multiplier(case_seed: int, low: float = 1.35, high: float = 2.20) -> float:
        # Keep v2 visibly different from v1; avoid multipliers too close to 1.
        rng = np.random.RandomState(case_seed + 9103)
        return float(rng.uniform(low, high))

    def _make_extra_physxnet_objects_for_case(case_idx: int, case_seed: int) -> List[Dict[str, Any]]:
        if str(getattr(args, "simulator_mode", "rigid")).strip().lower() != "rigid":
            return []
        desired_total_count_raw = getattr(args, "rigid_target_object_count", None)
        if desired_total_count_raw is None:
            desired_total_count = max(1, int(getattr(args, "custom_object_count_max", 1) or 1))
        else:
            desired_total_count = max(1, int(desired_total_count_raw))
        has_striker = not bool(getattr(args, "disable_striker", False))
        base_count = 1 + (1 if has_striker else 0)
        extra_count = max(0, desired_total_count - base_count)
        if extra_count <= 0:
            return []
        if not custom_asset_bank:
            return []

        physxnet_bank = [asset for asset in custom_asset_bank if str(asset.get("source_kind", "")) == "physxnet"]
        if not physxnet_bank:
            return []

        rng = np.random.RandomState(case_seed + 8081)
        chosen_indices = rng.choice(len(physxnet_bank), size=min(extra_count, len(physxnet_bank)), replace=False)
        x_positions = np.linspace(-0.42, 0.42, max(len(chosen_indices), 2))
        y_positions = np.linspace(-0.30, 0.30, max(len(chosen_indices), 2))
        extras: List[Dict[str, Any]] = []
        for local_idx, bank_idx in enumerate(np.asarray(chosen_indices).tolist()):
            asset = physxnet_bank[int(bank_idx)]
            scale_min, scale_max = asset.get("scale_range", [0.15, 0.30])
            obj_scale = float(rng.uniform(float(scale_min), float(scale_max)))
            spawn_offset = [
                float(x_positions[local_idx % len(x_positions)]),
                float(y_positions[::-1][local_idx % len(y_positions)]),
                0.0,
            ]
            extras.append(
                {
                    "custom_object_id": f"aux_physxnet_{case_idx:03d}_{local_idx:02d}",
                    "source_dataset": "physxnet_aux",
                    "source_label": str(asset.get("source_label", "physxnet")),
                    "source_asset_id": str(asset.get("asset_id", f"physxnet_aux_{local_idx:02d}")),
                    "source_display_id": str(asset.get("display_id", asset.get("asset_id", f"physxnet_aux_{local_idx:02d}"))),
                    "mesh_path": str(asset.get("mesh_path", "")),
                    "shape": str(asset.get("shape_name", "mesh")),
                    "material_name": str(asset.get("material_name", "physxnet")),
                    "spawn_direction": "static_in_scene",
                    "spawn_offset": spawn_offset,
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "color_rgba": list(asset.get("color_rgba", [0.85, 0.62, 0.28, 1.0])),
                    "scale": obj_scale,
                    "euler_deg": [0.0, 0.0, float(rng.uniform(-180.0, 180.0))],
                    "density": float(asset.get("density", 900.0)),
                    "youngs": float(asset.get("youngs", 1.0e9)),
                    "poisson": float(asset.get("poisson", 0.25)),
                    "friction": float(asset.get("friction", 0.55)),
                    "material_ctor": "gs.materials.Rigid",
                    "runtime_solver": "rigid_approx",
                    "runtime_material_ctor": "gs.materials.Rigid",
                    "strict_dataset_params": bool(asset.get("strict_dataset_params", True)),
                    "role_hint": "bystander",
                    "motion_type_hint": "static_rest",
                    "motion_group_hint": "auxiliary_static",
                }
            )
        return extras

    def _make_custom_objects_for_case(case_idx: int, case_seed: int) -> List[Dict[str, Any]]:
        if bool(getattr(args, "disable_striker", False)):
            return []
        if str(getattr(args, "simulator_mode", "rigid")).strip().lower() == "rigid":
            # Keep the default yellow striker. When count > 2, mix in extra
            # static PhysXNet rigid objects as scene context.
            return _make_extra_physxnet_objects_for_case(case_idx, case_seed)

        count_min = max(0, int(getattr(args, "custom_object_count_min", 1) or 0))
        count_max = max(count_min, int(getattr(args, "custom_object_count_max", 3) or count_min))
        prefix = str(getattr(args, "custom_object_prefix", "custom_ball") or "custom_ball").strip() or "custom_ball"
        base_speed = max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8))
        rng = np.random.RandomState(case_seed + 4049)
        count = int(rng.randint(count_min, count_max + 1)) if count_max > 0 else 0
        if count <= 0:
            return []

        palette = [
            [0.95, 0.75, 0.15, 1.0],
            [0.20, 0.65, 0.95, 1.0],
            [0.94, 0.36, 0.36, 1.0],
            [0.43, 0.81, 0.42, 1.0],
            [0.88, 0.48, 0.89, 1.0],
        ]
        direction_modes = [
            {"name": "right_to_left", "dir": np.array([-1.0, 0.0, 0.0], dtype=np.float64), "ang": 0.0},
            {"name": "left_to_right", "dir": np.array([1.0, 0.0, 0.0], dtype=np.float64), "ang": math.pi},
            {"name": "front_to_back", "dir": np.array([0.0, 1.0, 0.0], dtype=np.float64), "ang": -0.5 * math.pi},
            {"name": "back_to_front", "dir": np.array([0.0, -1.0, 0.0], dtype=np.float64), "ang": 0.5 * math.pi},
            {"name": "diag_front_right", "dir": np.array([-1.0, 1.0, 0.0], dtype=np.float64), "ang": -0.25 * math.pi},
            {"name": "diag_back_right", "dir": np.array([-1.0, -1.0, 0.0], dtype=np.float64), "ang": 0.25 * math.pi},
            {"name": "top_to_bottom", "dir": np.array([0.0, 0.0, -1.0], dtype=np.float64), "ang": 0.0},
        ]
        forced_spawn_direction = str(getattr(args, "custom_object_spawn_direction", "") or "").strip().lower()
        if forced_spawn_direction:
            filtered_direction_modes = [mode for mode in direction_modes if str(mode["name"]).lower() == forced_spawn_direction]
            if filtered_direction_modes:
                direction_modes = filtered_direction_modes
        rigidify_youngs_threshold_pa_raw = getattr(args, "custom_object_rigidify_youngs_threshold_pa", 1.0e8)
        rigidify_youngs_threshold_pa = None if rigidify_youngs_threshold_pa_raw is None or float(rigidify_youngs_threshold_pa_raw) <= 0 else float(rigidify_youngs_threshold_pa_raw)
        custom_objects: List[Dict[str, Any]] = []
        for obj_idx in range(count):
            if custom_asset_bank:
                asset = custom_asset_bank[int(rng.randint(0, len(custom_asset_bank)))]
            else:
                asset = {
                    "asset_id": "primitive__sphere__foam",
                    "source_kind": "primitive",
                    "source_label": "primitive",
                    "display_id": "primitive__sphere__foam",
                    "mesh_path": "",
                    "scale_range": [0.12, 0.20],
                    "material_ctor": "gs.materials.MPM.Elastic",
                    "density": 120.0,
                    "youngs": 8.0e4,
                    "poisson": 0.18,
                    "friction": 0.75,
                    "color_rgba": palette[(case_idx + obj_idx) % len(palette)],
                    "shape_name": "sphere",
                    "material_name": "foam",
                }
            direction = direction_modes[(case_idx + obj_idx) % len(direction_modes)]
            dir_xy = np.asarray(direction["dir"], dtype=np.float64)
            if str(direction["name"]) == "top_to_bottom":
                spawn_offset_vec = np.array(
                    [
                        float(rng.uniform(-0.18, 0.18) * max(0.22, bbox_size[0])),
                        float(rng.uniform(-0.18, 0.18) * max(0.22, bbox_size[1])),
                        float(max(0.24, 0.65 * bbox_size[2] + rng.uniform(0.10, 0.26))),
                    ],
                    dtype=np.float64,
                )
                spawn_offset_vec[0] += float(args.ball_posx)
                speed = float(base_speed * rng.uniform(0.45, 0.75))
                linear_vec = np.array(
                    [
                        float(rng.uniform(-0.08, 0.08)),
                        float(rng.uniform(-0.08, 0.08)),
                        -speed,
                    ],
                    dtype=np.float64,
                )
            else:
                dir_xy[:2] /= max(np.linalg.norm(dir_xy[:2]), 1e-8)
                tangent_xy = np.array([-dir_xy[1], dir_xy[0], 0.0], dtype=np.float64)
                spawn_distance = float(max(0.45, 1.15 * max(bbox_size[0], bbox_size[1])) + rng.uniform(0.05, 0.38))
                lateral_offset = float(rng.uniform(-0.42, 0.42) * max(0.28, bbox_size[1]))
                z_offset = float(max(0.06, 0.08 + 0.32 * bbox_size[2] + rng.uniform(-0.06, 0.26) * max(0.18, bbox_size[2])))
                spawn_offset_vec = -dir_xy * spawn_distance + tangent_xy * lateral_offset
                spawn_offset_vec[0] += float(args.ball_posx)
                spawn_offset_vec[2] = z_offset
                speed = float(base_speed * rng.uniform(0.60, 1.15))
                lateral_speed = float(rng.uniform(-0.35, 0.35))
                vz = float(rng.uniform(-0.12, 0.18))
                linear_vec = dir_xy * speed + tangent_xy * lateral_speed
                linear_vec[2] = vz
            wx = float(rng.uniform(-2.0, 2.0))
            wy = float(rng.uniform(-2.0, 2.0))
            wz = float(rng.uniform(-1.8, 1.8))
            scale_min, scale_max = asset.get("scale_range", [0.12, 0.24])
            obj_scale = float(rng.uniform(float(scale_min), float(scale_max)))
            yaw_deg = float(np.rad2deg(direction["ang"] + rng.uniform(-0.55, 0.55)))
            pitch_deg = 0.0
            roll_deg = 0.0
            runtime_solver, runtime_material_ctor = _choose_custom_runtime_solver(
                material_ctor=str(asset.get("material_ctor", "gs.materials.MPM.Elastic")),
                youngs=float(asset.get("youngs")) if asset.get("youngs", None) is not None else None,
                rigidify_youngs_threshold_pa=rigidify_youngs_threshold_pa,
            )
            if runtime_solver == "mpm" and str(runtime_material_ctor) == "gs.materials.MPM.Elastic":
                density_val = float(asset.get("density", 0.0) or 0.0)
                if density_val > 0.0 and density_val <= 200.0:
                    linear_vec *= 0.48
                    wz *= 0.35
                    wx *= 0.45
                    wy *= 0.45
            custom_objects.append(
                {
                    "custom_object_id": f"{prefix}_{case_idx:03d}_{obj_idx:02d}",
                    "source_dataset": str(asset.get("source_kind", "primitive")),
                    "source_label": str(asset.get("source_label", asset.get("source_kind", "primitive"))),
                    "source_asset_id": str(asset.get("asset_id", f"asset_{obj_idx:02d}")),
                    "source_display_id": str(asset.get("display_id", asset.get("asset_id", f"asset_{obj_idx:02d}"))),
                    "mesh_path": str(asset.get("mesh_path", "")),
                    "shape": str(asset.get("shape_name", "mesh")),
                    "material_name": str(asset.get("material_name", "unknown")),
                    "spawn_direction": str(direction["name"]),
                    "spawn_offset": spawn_offset_vec.tolist(),
                    "linear_velocity": linear_vec.tolist(),
                    "angular_velocity": [wx, wy, wz],
                    "color_rgba": list(asset.get("color_rgba", palette[(case_idx + obj_idx) % len(palette)])),
                    "scale": obj_scale,
                    "euler_deg": [roll_deg, pitch_deg, yaw_deg],
                    "density": float(asset.get("density", 900.0)),
                    "youngs": float(asset.get("youngs", 1.0e5)),
                    "poisson": float(asset.get("poisson", 0.22)),
                    "friction": float(asset.get("friction", 0.55)),
                    "material_ctor": str(asset.get("material_ctor", "gs.materials.MPM.Elastic")),
                    "runtime_solver": runtime_solver,
                    "runtime_material_ctor": runtime_material_ctor,
                    "strict_dataset_params": bool(asset.get("strict_dataset_params", False)),
                }
            )
        return custom_objects

    def _case_description_from_cfg(cfg: Dict[str, Any]) -> str:
        """Return a short human friendly label describing only position and velocity changes."""
        scene = str(cfg.get("scene_label", cfg.get("case_name", "case")))
        object_id = str(cfg.get("object_id", prepared.object_id))
        object_euler_deg = _fmt_vec3(cfg.get("object_euler_deg", [0.0, 0.0, 0.0]))
        offset = _fmt_vec3(cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]))
        linvel = _fmt_vec3(cfg.get("entry_linear_velocity", [0.0, 0.0, 0.0]))
        angvel = _fmt_vec3(cfg.get("entry_angular_velocity", [0.0, 0.0, 0.0]))
        obj_fixed = bool(cfg.get("object_fixed"))
        moving = bool(cfg.get("use_entry_motion"))
        state = "动态入场" if moving else "初始静止可受力运动"
        if (not moving) and (not obj_fixed) and abs(float(np.asarray(cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64)[2])) > 1e-8:
            state = "高处静止释放"
        lines = []
        counterfactual_meta = dict(cfg.get("counterfactual", {}) or {})
        if counterfactual_meta:
            lines.append(
                "反事实: "
                f"kind={counterfactual_meta.get('kind', 'unknown')} | "
                f"parent={counterfactual_meta.get('parent_case_name', 'unknown')} | "
                f"mode={counterfactual_meta.get('mode', 'unknown')}"
            )
            summary = str(counterfactual_meta.get("summary", "") or "").strip()
            if summary:
                lines.append(f"反事实说明: {summary}")
        lines.extend([
            f"主物体: id={object_id}",
            f"Case: {scene} | 状态={state}",
            f"主物体参数: offset={offset} m | euler_deg={object_euler_deg} | linvel={linvel} m/s | angvel={angvel} rad/s",
        ])
        for custom_obj in cfg.get("custom_objects", []) or []:
            lines.append(
                "自定义物体: "
                f"id={custom_obj['custom_object_id']} | source={custom_obj.get('source_dataset', 'primitive')} | "
                f"asset={custom_obj.get('source_display_id', custom_obj.get('source_asset_id', 'unknown'))} | "
                f"方向={custom_obj.get('spawn_direction', 'unknown')}"
            )
            lines.append(
                f"  参数: offset={_fmt_vec3(custom_obj.get('spawn_offset', [0.0, 0.0, 0.0]))} m | "
                f"euler_deg={_fmt_vec3(custom_obj.get('euler_deg', [0.0, 0.0, 0.0]))} | "
                f"scale={_fmt_scalar(custom_obj.get('scale', 1.0))} m"
            )
            lines.append(
                f"  速度: linvel={_fmt_vec3(custom_obj.get('linear_velocity', [0.0, 0.0, 0.0]))} m/s | "
                f"angvel={_fmt_vec3(custom_obj.get('angular_velocity', [0.0, 0.0, 0.0]))} rad/s"
            )
            lines.append(
                f"  材料: ctor={custom_obj.get('material_ctor', 'gs.materials.MPM.Elastic')} | "
                f"runtime_solver={custom_obj.get('runtime_solver', 'mpm')} | "
                f"runtime_ctor={custom_obj.get('runtime_material_ctor', custom_obj.get('material_ctor', 'gs.materials.MPM.Elastic'))} | "
                f"rho={_fmt_scalar(custom_obj.get('density', 0.0), 1)} kg/m^3 | "
                f"E={_fmt_scalar(custom_obj.get('youngs', 0.0), 1)} Pa | "
                f"nu={_fmt_scalar(custom_obj.get('poisson', 0.0), 3)} | "
                f"strict={'yes' if custom_obj.get('strict_dataset_params', False) else 'no'} | "
                f"color={_fmt_rgba(custom_obj.get('color_rgba', [1.0, 1.0, 1.0, 1.0]))}"
            )
        if not (cfg.get("custom_objects", []) or []):
            lines.append("自定义物体: none")
        return "\n".join(lines)

    def _make_case_cfg(
        *,
        case_idx: int,
        case_name: str,
        seed: int,
        placed_pos_offset: np.ndarray,
        use_entry_motion: bool,
        object_fixed_override: bool,
        entry_linear_velocity: Optional[np.ndarray] = None,
        entry_angular_velocity: Optional[np.ndarray] = None,
        gravity_z_override: Optional[float] = None,
        warmup_steps_override: Optional[int] = None,
        pre_record_delay_steps_override: Optional[int] = None,
        initial_still_frames_override: Optional[int] = None,
        liquid_settle_steps_override: Optional[int] = None,
        liquid_auto_settle_max_steps_override: Optional[int] = None,
        scene_label: Optional[str] = None,
        object_euler_deg: Optional[np.ndarray] = None,
        custom_objects: Optional[List[Dict[str, Any]]] = None,
        striker_speed_override: Optional[float] = None,
        disable_default_striker: bool = False,
    ) -> Dict[str, Any]:
        entry_linear_velocity = np.asarray(
            [0.0, 0.0, 0.0] if entry_linear_velocity is None else entry_linear_velocity,
            dtype=np.float64,
        )
        entry_angular_velocity = np.asarray(
            [0.0, 0.0, 0.0] if entry_angular_velocity is None else entry_angular_velocity,
            dtype=np.float64,
        )
        object_euler_deg = np.asarray(
            [0.0, 0.0, 0.0] if object_euler_deg is None else object_euler_deg,
            dtype=np.float64,
        )
        case_cfg = {
            "case_index": int(case_idx),
            "case_id": int(case_idx),
            "case_variant_index": int(case_idx),
            "case_kind": "factual",
            "case_name": str(case_name),
            "scene_label": str(scene_label or case_name),
            "seed": int(seed),
            "object_id": str(prepared.object_id),
            "is_physxnet_object": bool(is_physxnet_object),
            "object_bbox_volume_est_m3": float(bbox_volume_est_m3),
            "physxnet_volume_threshold_m3": float(volume_threshold_m3),
            "moving_allowed": bool(moving_allowed),
            "use_entry_motion": bool(use_entry_motion),
            "object_fixed": bool(object_fixed_override),
            "placed_pos_offset": np.asarray(placed_pos_offset, dtype=np.float64).tolist(),
            "object_euler_deg": object_euler_deg.tolist(),
            "entry_linear_velocity": entry_linear_velocity.tolist(),
            "entry_angular_velocity": entry_angular_velocity.tolist(),
            "custom_objects": list(custom_objects or []),
            "gravity_z_override": gravity_z_override,
            "warmup_steps_override": warmup_steps_override,
            "pre_record_delay_steps_override": pre_record_delay_steps_override,
            "initial_still_frames_override": initial_still_frames_override,
            "liquid_settle_steps_override": liquid_settle_steps_override,
            "liquid_auto_settle_max_steps_override": liquid_auto_settle_max_steps_override,
            "striker_speed_override": None if striker_speed_override is None else float(striker_speed_override),
            "disable_default_striker": bool(disable_default_striker),
        }
        case_cfg["case_notes"] = _case_description_from_cfg(case_cfg)
        return case_cfg

    def _make_counterfactual_case_cfg(
        *,
        base_cfg: Dict[str, Any],
        variant_slot: int,
        suffix: str,
        kind: str,
        mode: str,
        summary: str,
    ) -> Dict[str, Any]:
        cfg = copy.deepcopy(base_cfg)
        base_case_index = int(base_cfg.get("case_index", -1))
        cfg["case_index"] = int(20000 + max(base_case_index, 0) * 10 + int(variant_slot))
        cfg["case_id"] = int(counterfactual_case_id_map.get(str(kind), 11999))
        cfg["case_variant_index"] = int(cfg["case_index"])
        cfg["case_kind"] = str(kind)
        cfg["case_name"] = f"{str(base_cfg.get('case_name', 'case'))}__{suffix}"
        cfg["seed"] = int(base_cfg.get("seed", seed_anchor + 23)) + 1103 * int(variant_slot)
        cfg["counterfactual"] = {
            "kind": str(kind),
            "case_id": int(cfg["case_id"]),
            "mode": str(mode),
            "parent_case_index": base_case_index,
            "parent_case_id": int(base_cfg.get("case_id", base_case_index)),
            "parent_case_name": str(base_cfg.get("case_name", "case")),
            "scene_label": str(base_cfg.get("scene_label", base_cfg.get("case_name", "case"))),
            "summary": str(summary),
        }
        cfg["case_notes"] = _case_description_from_cfg(cfg)
        return cfg

    def _build_same_scene_negative_case(base_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        has_default_striker = (not bool(getattr(args, "disable_striker", False))) and (not bool(base_cfg.get("disable_default_striker", False)))
        has_custom_objects = bool(base_cfg.get("custom_objects", []) or [])
        uses_entry_motion = bool(base_cfg.get("use_entry_motion", False))
        if not (has_default_striker or has_custom_objects or uses_entry_motion):
            return None

        rng = np.random.RandomState(int(base_cfg.get("seed", seed_anchor + 23)) + 61001)
        speed_scale = float(rng.uniform(0.72, 1.32))
        yaw_delta = float(rng.uniform(12.0, 24.0) * (-1.0 if rng.rand() < 0.5 else 1.0))
        ang_scale = float(rng.uniform(0.55, 1.45))
        base_restitution = float(np.clip(base_cfg.get("rigid_restitution_override", rigid_material_defaults["restitution"]), 0.0, 1.2))
        if base_restitution < 0.35:
            restitution_new = float(np.clip(base_restitution + rng.uniform(0.32, 0.58), 0.02, 1.05))
        else:
            restitution_new = float(np.clip(base_restitution * rng.uniform(0.18, 0.48), 0.02, 1.05))

        cfg = _make_counterfactual_case_cfg(
            base_cfg=base_cfg,
            variant_slot=0,
            suffix="cf_same_scene_neg",
            kind="same_scene_negative",
            mode="perturb_initial_velocity_and_restitution",
            summary=(
                "保持同一物体/背景/相机，只改变初速度、撞击方向和恢复系数，"
                "让 future clip 成为同场景反事实负样本。"
            ),
        )
        cfg["rigid_restitution_override"] = restitution_new

        if uses_entry_motion:
            base_linear = np.asarray(base_cfg.get("entry_linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            base_angular = np.asarray(base_cfg.get("entry_angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
            cfg["entry_linear_velocity"] = _rotate_xy(base_linear * speed_scale, yaw_delta).tolist()
            cfg["entry_angular_velocity"] = (base_angular * ang_scale).tolist()

        if has_default_striker:
            base_speed = float(base_cfg.get("striker_speed_override") or float(getattr(args, "striker_speed", 2.8) or 2.8))
            if bool(getattr(args, "striker_drop_top", False)):
                striker_linear = np.array(
                    [
                        float(rng.uniform(-0.28, 0.28) * base_speed),
                        float(rng.uniform(-0.28, 0.28) * base_speed),
                        -base_speed * speed_scale,
                    ],
                    dtype=np.float64,
                )
            else:
                striker_linear = _rotate_xy(np.array([-base_speed * speed_scale, 0.0, 0.0], dtype=np.float64), yaw_delta)
            cfg["default_striker_velocity_override"] = np.concatenate(
                [striker_linear, np.zeros(3, dtype=np.float64)],
                axis=0,
            ).tolist()

        if has_custom_objects:
            updated_custom_objects: List[Dict[str, Any]] = []
            for custom_obj in cfg.get("custom_objects", []) or []:
                custom_copy = copy.deepcopy(custom_obj)
                linear = np.asarray(custom_copy.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
                angular = np.asarray(custom_copy.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
                if np.linalg.norm(linear) > 1e-8:
                    custom_copy["linear_velocity"] = _rotate_xy(linear * speed_scale, yaw_delta).tolist()
                    custom_copy["angular_velocity"] = (angular * ang_scale).tolist()
                updated_custom_objects.append(custom_copy)
            cfg["custom_objects"] = updated_custom_objects

        cfg["case_notes"] = _case_description_from_cfg(cfg)
        return cfg

    def _build_no_collision_negative_case(base_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        has_default_striker = (not bool(getattr(args, "disable_striker", False))) and (not bool(base_cfg.get("disable_default_striker", False)))
        has_custom_objects = bool(base_cfg.get("custom_objects", []) or [])
        uses_entry_motion = bool(base_cfg.get("use_entry_motion", False))
        if not (has_default_striker or has_custom_objects or uses_entry_motion):
            return None

        miss_y = float(max(0.30, 0.90 * max(float(bbox_size[0]), float(bbox_size[1]))))
        miss_z = float(max(0.06, 0.24 * float(bbox_size[2])))
        cfg = _make_counterfactual_case_cfg(
            base_cfg=base_cfg,
            variant_slot=1,
            suffix="cf_no_collision_neg",
            kind="no_collision_negative",
            mode="miss_contact_keep_preimpact_trend",
            summary=(
                "保持同一视角和背景，让物体沿碰撞前趋势继续运动，"
                "但通过错开接触路径避免发生接触或反弹。"
            ),
        )

        if has_default_striker:
            if bool(getattr(args, "striker_drop_top", False)):
                cfg["default_striker_start_offset"] = [miss_y, miss_y, 0.0]
            else:
                cfg["default_striker_start_offset"] = [0.0, miss_y, miss_z]

        if has_custom_objects:
            updated_custom_objects = []
            for custom_obj in cfg.get("custom_objects", []) or []:
                custom_copy = copy.deepcopy(custom_obj)
                linear = np.asarray(custom_copy.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
                if np.linalg.norm(linear[:2]) > 1e-8:
                    perp = np.array([-linear[1], linear[0], 0.0], dtype=np.float64)
                    perp /= max(np.linalg.norm(perp[:2]), 1e-8)
                    spawn_offset = np.asarray(custom_copy.get("spawn_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
                    spawn_offset = spawn_offset + perp * miss_y
                    spawn_offset[2] += 0.35 * miss_z
                    custom_copy["spawn_offset"] = spawn_offset.tolist()
                updated_custom_objects.append(custom_copy)
            cfg["custom_objects"] = updated_custom_objects

        if uses_entry_motion:
            lifted_offset = np.asarray(cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
            lifted_offset[2] += float(max(0.02, miss_z))
            cfg["placed_pos_offset"] = lifted_offset.tolist()
            cfg["gravity_z_override"] = float(
                getattr(args, "counterfactual_no_collision_gravity_z", -9.81) or -9.81
            )
            cfg["warmup_steps_override"] = 0
            cfg["pre_record_delay_steps_override"] = 0
            cfg["initial_still_frames_override"] = 0
            if np.linalg.norm(
                _to_velocity6(cfg.get("default_striker_velocity_override", [0.0, 0.0, 0.0]))[:3]
            ) <= 1e-8:
                cfg["entry_linear_velocity"] = np.asarray(base_cfg.get("entry_linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64).tolist()

        cfg["case_notes"] = _case_description_from_cfg(cfg)
        return cfg

    def _legacy_random_case(case_idx: int) -> Dict[str, Any]:
        rng = np.random.RandomState(seed_anchor + case_idx * 9973 + 23)
        use_entry_motion = bool(move_flags[case_idx]) if case_idx < len(move_flags) else bool(rng.rand() < entry_prob)

        placed_pos_offset = np.zeros(3, dtype=np.float64)
        entry_linear_velocity = np.zeros(3, dtype=np.float64)
        entry_angular_velocity = np.zeros(3, dtype=np.float64)
        runtime_object_fixed = bool(object_fixed)
        warmup_steps_override = None
        pre_record_delay_steps_override = None
        initial_still_frames_override = None
        liquid_settle_steps_override = None
        liquid_auto_settle_max_steps_override = None

        if use_entry_motion:
            runtime_object_fixed = False
            speed = float(rng.uniform(speed_min, speed_max))
            lateral_speed = float(rng.uniform(-0.18, 0.18))
            yaw_speed = float(rng.uniform(-1.2, 1.2))
            entry_margin_x = 0.30 + float(rng.uniform(0.75, 1.35)) * float(max(0.18, bbox_size[0]))
            entry_margin_y = float(rng.uniform(-0.28, 0.28)) * float(max(0.20, bbox_size[1]))
            placed_pos_offset = np.array(
                [
                    entry_margin_x,
                    -0.15 * base_y_offset + entry_margin_y,
                    0.0,
                ],
                dtype=np.float64,
            )
            entry_linear_velocity = np.array([-speed, lateral_speed, 0.0], dtype=np.float64)
            entry_angular_velocity = np.array([0.0, 0.0, yaw_speed], dtype=np.float64)
            warmup_steps_override = 0
            pre_record_delay_steps_override = 0
            initial_still_frames_override = 0
            liquid_settle_steps_override = 0
            liquid_auto_settle_max_steps_override = 0
        elif moving_allowed or force_static_physxnet:
            runtime_object_fixed = False

        return _make_case_cfg(
            case_idx=case_idx,
            case_name=f"case{case_idx:03d}",
            scene_label="legacy_random",
            seed=seed_anchor + case_idx * 9973 + 23,
            placed_pos_offset=placed_pos_offset,
            object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(seed_anchor + case_idx * 9973 + 23)], dtype=np.float64),
            use_entry_motion=use_entry_motion,
            object_fixed_override=runtime_object_fixed,
            entry_linear_velocity=entry_linear_velocity,
            entry_angular_velocity=entry_angular_velocity,
            warmup_steps_override=warmup_steps_override,
            pre_record_delay_steps_override=pre_record_delay_steps_override,
            initial_still_frames_override=initial_still_frames_override,
            liquid_settle_steps_override=liquid_settle_steps_override,
            liquid_auto_settle_max_steps_override=liquid_auto_settle_max_steps_override,
            custom_objects=_make_custom_objects_for_case(case_idx, seed_anchor + case_idx * 9973 + 23),
        )

    def _estimate_custom_asset_volume_m3(asset: Dict[str, Any], scale: float) -> float:
        mesh_path = str(asset.get("mesh_path", "") or "").strip()
        if mesh_path:
            bounds_info = _mesh_bounds_info(Path(mesh_path), scale=float(scale))
            if bounds_info is not None:
                size = np.asarray(bounds_info.get("bounds_size", [scale, scale, scale]), dtype=np.float64)
                return float(np.prod(np.maximum(size, 1e-6)))
        scale = float(max(1e-3, scale))
        return float(scale ** 3)

    def _sample_free_motion_pose(case_seed: int, slot_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(case_seed + 20011 + 193 * slot_idx)
        euler_deg = np.array(
            [
                float(rng.uniform(-18.0, 18.0)),
                float(rng.uniform(-14.0, 14.0)),
                float(rng.uniform(-180.0, 180.0)),
            ],
            dtype=np.float64,
        )
        angular = np.array(
            [
                float(rng.uniform(-1.15, 1.15)),
                float(rng.uniform(-1.15, 1.15)),
                float(rng.uniform(-1.65, 1.65)),
            ],
            dtype=np.float64,
        )
        return euler_deg, angular

    def _make_multi_object_free_motion_case(
        *,
        case_idx: int,
        total_count: int,
        motion_kind: str,
    ) -> Optional[Dict[str, Any]]:
        if total_count < 2:
            return None
        simulator_mode = str(getattr(args, "simulator_mode", "rigid")).strip().lower()
        if simulator_mode != "rigid":
            return None

        case_seed = _case_seed(case_idx)
        rng = np.random.RandomState(case_seed + 14021)
        label_motion = "projectile" if str(motion_kind).strip().lower() == "projectile" else "drop"
        scene_label = f"multi{int(total_count)}_{label_motion}_nocollision"
        slot_indices = np.arange(int(total_count), dtype=np.float64)
        slot_offsets = 2.0 * slot_indices - (float(total_count) - 1.0)
        lane_gap = float(max(0.95, 2.25 * max(float(bbox_size[0]), float(bbox_size[1]), 0.28)))
        x_gap = float(max(0.30, 0.70 * max(float(bbox_size[0]), 0.24)))
        y_slots = slot_offsets * lane_gap
        x_slots = slot_offsets * x_gap
        main_slot = int(total_count // 2)
        main_euler_deg, main_angular = _sample_free_motion_pose(case_seed, main_slot)
        main_should_move = bool(moving_allowed)
        main_offset_z = 0.0
        main_linear = np.zeros(3, dtype=np.float64)
        main_angular_final = np.zeros(3, dtype=np.float64)
        if main_should_move:
            if label_motion == "projectile":
                main_offset_z = float(max(0.92, 0.85 + 0.75 * float(bbox_size[2])))
                main_linear = np.array(
                    [
                        float(rng.uniform(0.20, 0.38)),
                        0.0,
                        float(rng.uniform(1.32, 1.92)),
                    ],
                    dtype=np.float64,
                )
            else:
                main_offset_z = float(max(1.08, 1.05 + 1.05 * float(bbox_size[2])))
                main_linear = np.array(
                    [
                        float(rng.uniform(-0.04, 0.04)),
                        0.0,
                        0.0,
                    ],
                    dtype=np.float64,
                )
            main_angular_final = main_angular
        else:
            main_euler_deg = np.array([0.0, 0.0, _sample_object_yaw_deg(case_seed)], dtype=np.float64)

        moving_asset_bank = [asset for asset in custom_asset_bank if str(asset.get("source_kind", "")) == "physxnet"]
        if not moving_asset_bank:
            moving_asset_bank = list(custom_asset_bank)
        if not moving_asset_bank:
            moving_asset_bank = [
                {
                    "asset_id": "primitive__sphere__fallback",
                    "source_kind": "primitive",
                    "source_label": "primitive",
                    "display_id": "primitive_fallback",
                    "mesh_path": "",
                    "scale_range": [0.12, 0.18],
                    "material_ctor": "gs.materials.Rigid",
                    "density": 950.0,
                    "youngs": 1.0e9,
                    "poisson": 0.24,
                    "friction": 0.55,
                    "color_rgba": [0.25, 0.72, 0.92, 1.0],
                    "shape_name": "sphere",
                    "material_name": "rigid_fallback",
                }
            ]

        used_asset_ids: set[str] = set()
        custom_objects: List[Dict[str, Any]] = []
        custom_slot_ids = [idx for idx in range(total_count) if idx != main_slot]
        for obj_order, slot_id in enumerate(custom_slot_ids):
            asset_candidates = [asset for asset in moving_asset_bank if str(asset.get("asset_id", "")) not in used_asset_ids]
            if not asset_candidates:
                asset_candidates = moving_asset_bank
            asset = copy.deepcopy(asset_candidates[int(rng.randint(0, len(asset_candidates)))])
            used_asset_ids.add(str(asset.get("asset_id", f"asset_{obj_order:02d}")))
            scale_min, scale_max = asset.get("scale_range", [0.14, 0.24])
            scale_max = min(float(scale_max), 0.20 if total_count >= 3 else 0.22)
            scale_min = min(float(scale_min), scale_max)
            obj_scale = float(rng.uniform(scale_min, scale_max))
            obj_volume_est = _estimate_custom_asset_volume_m3(asset, obj_scale)
            asset_should_move = bool(volume_threshold_m3 <= 0.0 or obj_volume_est < volume_threshold_m3)
            obj_euler_deg, obj_angular = _sample_free_motion_pose(case_seed, int(slot_id) + 7)
            spawn_x = float(x_slots[int(slot_id)] + rng.uniform(-0.025, 0.025))
            spawn_y = float(y_slots[int(slot_id)])
            spawn_z = 0.0
            linear = np.zeros(3, dtype=np.float64)
            angular = np.zeros(3, dtype=np.float64)
            role_hint = "bystander"
            motion_type_hint = "static_rest"
            motion_group_hint = "auxiliary_static"
            if asset_should_move:
                if label_motion == "projectile":
                    spawn_z = float(rng.uniform(0.88, 1.28))
                    linear = np.array(
                        [
                            float(rng.uniform(0.14, 0.26)),
                            0.0,
                            float(rng.uniform(0.95, 1.42)),
                        ],
                        dtype=np.float64,
                    )
                    motion_type_hint = "independent_projectile_motion"
                    motion_group_hint = "projectile_motion"
                else:
                    spawn_z = float(rng.uniform(1.10, 1.72))
                    linear = np.array(
                        [
                            float(rng.uniform(-0.03, 0.03)),
                            0.0,
                            0.0,
                        ],
                        dtype=np.float64,
                    )
                    motion_type_hint = "independent_gravity_drop"
                    motion_group_hint = "gravity_drop"
                angular = obj_angular
                role_hint = "co_actor"

            source_kind = str(asset.get("source_kind", "primitive"))
            source_tag = "physxnet_aux" if source_kind == "physxnet" else "custom_object"
            custom_objects.append(
                {
                    "custom_object_id": f"free_motion_{total_count:02d}_{label_motion}_{obj_order:02d}",
                    "source_dataset": source_tag,
                    "source_label": str(asset.get("source_label", source_kind)),
                    "source_asset_id": str(asset.get("asset_id", f"asset_{obj_order:02d}")),
                    "source_display_id": str(asset.get("display_id", asset.get("asset_id", f"asset_{obj_order:02d}"))),
                    "mesh_path": str(asset.get("mesh_path", "")),
                    "shape": str(asset.get("shape_name", "mesh")),
                    "material_name": str(asset.get("material_name", "physxnet")),
                    "spawn_direction": f"{label_motion}_lane_release",
                    "spawn_offset": [spawn_x, spawn_y, spawn_z],
                    "linear_velocity": linear.tolist(),
                    "angular_velocity": angular.tolist(),
                    "color_rgba": list(asset.get("color_rgba", [0.82, 0.58, 0.22, 1.0])),
                    "scale": obj_scale,
                    "euler_deg": obj_euler_deg.tolist(),
                    "density": float(asset.get("density", 950.0)),
                    "youngs": float(asset.get("youngs", 1.0e9)),
                    "poisson": float(asset.get("poisson", 0.25)),
                    "friction": float(asset.get("friction", 0.55)),
                    "material_ctor": "gs.materials.Rigid",
                    "runtime_solver": "rigid_approx",
                    "runtime_material_ctor": "gs.materials.Rigid",
                    "strict_dataset_params": bool(asset.get("strict_dataset_params", True)),
                    "role_hint": role_hint,
                    "motion_type_hint": motion_type_hint,
                    "motion_group_hint": motion_group_hint,
                    "expected_volume_est_m3": float(obj_volume_est),
                    "no_pairwise_collision_expected": True,
                }
            )

        case_cfg = _make_case_cfg(
            case_idx=case_idx,
            case_name=f"case{case_idx:03d}_{scene_label}",
            scene_label=scene_label,
            seed=case_seed,
            placed_pos_offset=np.array(
                [
                    float(x_slots[main_slot]),
                    float(y_slots[main_slot]),
                    float(main_offset_z),
                ],
                dtype=np.float64,
            ),
            object_euler_deg=main_euler_deg,
            use_entry_motion=bool(main_should_move),
            object_fixed_override=False,
            entry_linear_velocity=main_linear,
            entry_angular_velocity=main_angular_final,
            gravity_z_override=-9.81,
            warmup_steps_override=0,
            pre_record_delay_steps_override=0,
            initial_still_frames_override=0,
            liquid_settle_steps_override=0,
            liquid_auto_settle_max_steps_override=0,
            custom_objects=custom_objects,
            disable_default_striker=True,
        )
        case_cfg["main_object_role_hint"] = "co_actor" if main_should_move else "bystander"
        case_cfg["main_object_motion_type_hint"] = (
            "independent_projectile_motion"
            if main_should_move and label_motion == "projectile"
            else ("independent_gravity_drop" if main_should_move else "static_rest")
        )
        case_cfg["main_object_motion_group_hint"] = (
            "projectile_motion"
            if main_should_move and label_motion == "projectile"
            else ("gravity_drop" if main_should_move else "static_placement")
        )
        case_cfg["no_pairwise_collision_expected"] = True
        case_cfg["gravity_required"] = True
        case_cfg["case_notes"] = _case_description_from_cfg(case_cfg)
        return case_cfg

    x_shift = float(max(0.18, 0.65 * bbox_size[0]))
    y_shift = float(max(0.18, 0.55 * bbox_size[1], 0.30 * base_y_offset))
    allow_highdrop = bool(is_physxnet_object and bbox_volume_est_m3 < volume_threshold_m3)
    high_drop_z = float(max(0.9, 0.85 * bbox_size[2])) if allow_highdrop else 0.0 ### debug
    wide_x_shift = float(max(0.10, 0.30 * bbox_size[0]))
    highdrop_x_shift = float(max(0.06, 0.18 * bbox_size[0]))
    entry_spawn_x = float(max(0.55, 1.05 * bbox_size[0]))
    entry_spawn_y = float(max(0.20, 0.62 * bbox_size[1], 0.34 * base_y_offset))

    def _sample_random_orientation_velocity_case(case_idx: int, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Deterministic per-object random pose/velocity for single-object motion previews.
        rng = np.random.RandomState(seed_anchor + case_idx * 9973 + int(getattr(args, "motion_resample_index", 0)) * 104729 + 4242)
        euler_deg = np.array(
            [
                float(rng.uniform(-28.0, 28.0)),
                float(rng.uniform(-22.0, 22.0)),
                float(rng.uniform(-180.0, 180.0)),
            ],
            dtype=np.float64,
        )
        if mode == "parabola":
            theta = float(rng.uniform(-math.pi, math.pi))
            horizontal_speed = float(rng.uniform(0.35, 0.78))
            linear = np.array(
                [
                    horizontal_speed * math.cos(theta),
                    horizontal_speed * math.sin(theta),
                    float(rng.uniform(0.95, 1.55)),
                ],
                dtype=np.float64,
            )
            angular = np.array(
                [
                    float(rng.uniform(-1.8, 1.8)),
                    float(rng.uniform(-1.8, 1.8)),
                    float(rng.uniform(-2.6, 2.6)),
                ],
                dtype=np.float64,
            )
        else:
            linear = np.array(
                [
                    float(rng.uniform(-0.18, 0.18)),
                    float(rng.uniform(-0.04, 0.04)),
                    0.0,
                ],
                dtype=np.float64,
            )
            angular = np.array(
                [
                    float(rng.uniform(-1.25, 1.25)),
                    float(rng.uniform(-0.45, 0.45)),
                    float(rng.uniform(-1.80, 1.80)),
                ],
                dtype=np.float64,
            )
        return euler_deg, linear, angular

    static_striker_speed_override = float(
        max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(_case_seed(0)))
    )
    diverse_templates: List[Dict[str, Any]] = [
        _make_case_cfg(
            case_idx=0,
            case_name="case000_static_center",
            scene_label="static_center",
            seed=_case_seed(0),
            placed_pos_offset=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(_case_seed(0))], dtype=np.float64),
            use_entry_motion=False,
            object_fixed_override=False,
            custom_objects=_make_custom_objects_for_case(0, _case_seed(0)),
            striker_speed_override=static_striker_speed_override,
        ),
        _make_case_cfg(
            case_idx=1,
            case_name="case001_static_left",
            scene_label="static_left",
            seed=_case_seed(1),
            placed_pos_offset=np.array([-0.55 * wide_x_shift, 1.15 * y_shift, 0.0], dtype=np.float64),
            object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(_case_seed(1))], dtype=np.float64),
            use_entry_motion=False,
            object_fixed_override=False,
            custom_objects=_make_custom_objects_for_case(1, _case_seed(1)),
            striker_speed_override=float(
                max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(_case_seed(1)))
            ),
        ),
        _make_case_cfg(
            case_idx=2,
            case_name="case002_static_right",
            scene_label="static_right",
            seed=_case_seed(2),
            placed_pos_offset=np.array([0.45 * wide_x_shift, -1.10 * y_shift, 0.0], dtype=np.float64),
            object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(_case_seed(2))], dtype=np.float64),
            use_entry_motion=False,
            object_fixed_override=False,
            custom_objects=_make_custom_objects_for_case(2, _case_seed(2)),
            striker_speed_override=float(
                max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(_case_seed(2)))
            ),
        ),
        _make_case_cfg(
            case_idx=3,
            case_name="case003_static_highdrop",
            scene_label="static_highdrop",
            seed=_case_seed(3),
            placed_pos_offset=np.array([highdrop_x_shift, 0.0, high_drop_z], dtype=np.float64),
            object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(_case_seed(3))], dtype=np.float64),
            use_entry_motion=False,
            object_fixed_override=False,
            warmup_steps_override=0 if allow_highdrop else None,
            pre_record_delay_steps_override=0 if allow_highdrop else None,
            initial_still_frames_override=0 if allow_highdrop else None,
            liquid_settle_steps_override=0 if allow_highdrop else None,
            liquid_auto_settle_max_steps_override=0 if allow_highdrop else None,
            custom_objects=_make_custom_objects_for_case(3, _case_seed(3)),
            striker_speed_override=float(
                max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(_case_seed(3)))
            ),
        ),
    ]

    static_v2_templates: List[Dict[str, Any]] = []
    if not bool(getattr(args, "disable_striker", False)):
        static_seed_offset = 500000
        for v2_offset, base_template in enumerate(diverse_templates[:3]):
            template_v2 = copy.deepcopy(base_template)
            template_v2["case_index"] = int(base_template["case_index"]) + 100
            template_v2["case_name"] = str(base_template["case_name"]) + "_v2"
            template_v2["scene_label"] = str(base_template["scene_label"]) + "_v2"
            template_v2["seed"] = int(base_template["seed"]) + static_seed_offset + 97 * v2_offset
            base_speed = float(base_template.get("striker_speed_override") or float(getattr(args, "striker_speed", 2.8) or 2.8))
            template_v2["striker_speed_override"] = float(base_speed * _random_speed_multiplier(int(template_v2["seed"]), 1.45, 2.35))
            template_v2["case_notes"] = _case_description_from_cfg(template_v2)
            static_v2_templates.append(template_v2)

    if moving_allowed:
        entry_left_seed = _case_seed(5)
        entry_left_linear, entry_left_angular = _sample_entry_velocity(entry_left_seed, "entry_left")
        entry_right_seed = _case_seed(6)
        entry_right_linear, entry_right_angular = _sample_entry_velocity(entry_right_seed, "entry_right")
        entry_fast_center_seed = _case_seed(7)
        entry_fast_center_linear, entry_fast_center_angular = _sample_entry_velocity(entry_fast_center_seed, "entry_fast_center")
        dynamic_templates = [
            _make_case_cfg(
                case_idx=5,
                case_name="case005_entry_left",
                scene_label="entry_left",
                seed=entry_left_seed,
                placed_pos_offset=np.array([entry_spawn_x, entry_spawn_y, 0.0], dtype=np.float64),
                object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(entry_left_seed)], dtype=np.float64),
                use_entry_motion=True,
                object_fixed_override=False,
                entry_linear_velocity=entry_left_linear,
                entry_angular_velocity=entry_left_angular,
                striker_speed_override=float(
                    max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(entry_left_seed))
                ),
                warmup_steps_override=0,
                pre_record_delay_steps_override=0,
                initial_still_frames_override=0,
                liquid_settle_steps_override=0,
                liquid_auto_settle_max_steps_override=0,
                custom_objects=_make_custom_objects_for_case(5, entry_left_seed),
            ),
            _make_case_cfg(
                case_idx=6,
                case_name="case006_entry_right",
                scene_label="entry_right",
                seed=entry_right_seed,
                placed_pos_offset=np.array([entry_spawn_x, -entry_spawn_y, 0.0], dtype=np.float64),
                object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(entry_right_seed)], dtype=np.float64),
                use_entry_motion=True,
                object_fixed_override=False,
                entry_linear_velocity=entry_right_linear,
                entry_angular_velocity=entry_right_angular,
                striker_speed_override=float(
                    max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(entry_right_seed))
                ),
                warmup_steps_override=0,
                pre_record_delay_steps_override=0,
                initial_still_frames_override=0,
                liquid_settle_steps_override=0,
                liquid_auto_settle_max_steps_override=0,
                custom_objects=_make_custom_objects_for_case(6, entry_right_seed),
            ),
            _make_case_cfg(
                case_idx=7,
                case_name="case007_entry_fast_center",
                scene_label="entry_fast_center",
                seed=entry_fast_center_seed,
                placed_pos_offset=np.array([entry_spawn_x + 0.40 * x_shift, 0.0, 0.0], dtype=np.float64),
                object_euler_deg=np.array([0.0, 0.0, _sample_object_yaw_deg(entry_fast_center_seed)], dtype=np.float64),
                use_entry_motion=True,
                object_fixed_override=False,
                entry_linear_velocity=entry_fast_center_linear,
                entry_angular_velocity=entry_fast_center_angular,
                striker_speed_override=float(
                    max(0.1, float(getattr(args, "striker_speed", 2.8) or 2.8) * _random_speed_multiplier(entry_fast_center_seed))
                ),
                warmup_steps_override=0,
                pre_record_delay_steps_override=0,
                initial_still_frames_override=0,
                liquid_settle_steps_override=0,
                liquid_auto_settle_max_steps_override=0,
                custom_objects=_make_custom_objects_for_case(7, entry_fast_center_seed),
            ),
        ]
        diverse_templates = [
            diverse_templates[0],
            diverse_templates[1],
            diverse_templates[3],
            dynamic_templates[0],
            diverse_templates[0],
            dynamic_templates[1],
            diverse_templates[2],
            dynamic_templates[2],
        ]

    if static_v2_templates:
        diverse_templates.extend(static_v2_templates)

    random_parabola_euler, random_parabola_linear, random_parabola_angular = _sample_random_orientation_velocity_case(900, "parabola")
    high_drop_euler, high_drop_linear, high_drop_angular = _sample_random_orientation_velocity_case(901, "high_drop")
    high_drop_height_rng = np.random.RandomState(seed_anchor + 901 * 9973 + int(getattr(args, "motion_resample_index", 0)) * 104729 + 9090)
    high_drop_height_abs = float(high_drop_height_rng.uniform(1.65, 2.45))
    random_motion_templates = [
        _make_case_cfg(
            case_idx=900,
            case_name="case900_random_parabola",
            scene_label="random_parabola",
            seed=_case_seed(900),
            placed_pos_offset=np.array([0.0, 0.0, float(np.random.RandomState(seed_anchor + 900 * 9973 + int(getattr(args, "motion_resample_index", 0)) * 104729 + 9090).uniform(0.85, 1.45))], dtype=np.float64),
            object_euler_deg=random_parabola_euler,
            use_entry_motion=True,
            object_fixed_override=False,
            entry_linear_velocity=random_parabola_linear,
            entry_angular_velocity=random_parabola_angular,
            gravity_z_override=-9.81,
            warmup_steps_override=0,
            pre_record_delay_steps_override=0,
            initial_still_frames_override=0,
            liquid_settle_steps_override=0,
            liquid_auto_settle_max_steps_override=0,
            custom_objects=_make_custom_objects_for_case(900, _case_seed(900)),
        ),
        _make_case_cfg(
            case_idx=901,
            case_name="case901_high_drop",
            scene_label="high_drop",
            seed=_case_seed(901),
            placed_pos_offset=np.array([0.0, 0.0, high_drop_height_abs], dtype=np.float64),
            object_euler_deg=high_drop_euler,
            use_entry_motion=True,
            object_fixed_override=False,
            entry_linear_velocity=high_drop_linear,
            entry_angular_velocity=high_drop_angular,
            gravity_z_override=-9.81,
            warmup_steps_override=0,
            pre_record_delay_steps_override=0,
            initial_still_frames_override=0,
            liquid_settle_steps_override=0,
            liquid_auto_settle_max_steps_override=0,
            custom_objects=_make_custom_objects_for_case(901, _case_seed(901)),
        ),
    ]
    diverse_templates.extend(random_motion_templates)

    multi_object_free_motion_templates = [
        _make_multi_object_free_motion_case(case_idx=210, total_count=2, motion_kind="projectile"),
        _make_multi_object_free_motion_case(case_idx=211, total_count=2, motion_kind="drop"),
        _make_multi_object_free_motion_case(case_idx=220, total_count=3, motion_kind="projectile"),
        _make_multi_object_free_motion_case(case_idx=221, total_count=3, motion_kind="drop"),
        _make_multi_object_free_motion_case(case_idx=230, total_count=4, motion_kind="projectile"),
        _make_multi_object_free_motion_case(case_idx=231, total_count=4, motion_kind="drop"),
    ]
    diverse_templates.extend([cfg for cfg in multi_object_free_motion_templates if cfg is not None])

    if case_scene_mode == "diverse":
        case_configs.extend(diverse_templates)
    elif case_scene_mode == "auto":
        explicit_count = min(len(diverse_templates), max(num_cases, len(diverse_templates)))
        for template in diverse_templates[:explicit_count]:
            case_configs.append(template)
        for case_idx in range(len(case_configs), num_cases):
            case_configs.append(_legacy_random_case(case_idx))
    elif case_scene_mode == "legacy_random":
        for case_idx in range(num_cases):
            case_configs.append(_legacy_random_case(case_idx))
    else:
        raise ValueError(f"Unsupported case_scene_mode: {case_scene_mode}")

    if bool(getattr(args, "enable_counterfactual_cases", False)):
        counterfactual_cases: List[Dict[str, Any]] = []
        for base_cfg in list(case_configs):
            same_scene_cfg = _build_same_scene_negative_case(base_cfg)
            if same_scene_cfg is not None:
                counterfactual_cases.append(same_scene_cfg)
            no_collision_cfg = _build_no_collision_negative_case(base_cfg)
            if no_collision_cfg is not None:
                counterfactual_cases.append(no_collision_cfg)
        case_configs.extend(counterfactual_cases)
    if bool(getattr(args, "counterfactual_only", False)):
        case_configs = [cfg for cfg in case_configs if dict(cfg.get("counterfactual", {}) or {})]

    return case_configs


def _case_cfg_or_default(case_cfg: Dict[str, Any], key: str, default: Any) -> Any:
    value = case_cfg.get(key, default)
    if value is None:
        return default
    return value


def _apply_rigid_entry_velocity(entity: Any, linear: np.ndarray, angular: np.ndarray) -> bool:
    linear = np.asarray(linear, dtype=np.float64).reshape(3)
    angular = np.asarray(angular, dtype=np.float64).reshape(3)
    velocity6 = np.concatenate([linear, angular], axis=0).astype(np.float64)

    attempts = [
        {"velocity": velocity6.tolist(), "kwargs": {}},
        {"velocity": linear.tolist(), "kwargs": {"dofs_idx_local": [0, 1, 2]}},
        {"velocity": angular.tolist(), "kwargs": {"dofs_idx_local": [3, 4, 5]}},
    ]

    any_applied = False
    last_exc: Optional[Exception] = None
    for attempt in attempts:
        if np.linalg.norm(np.asarray(attempt["velocity"], dtype=np.float64)) <= 1e-8:
            continue
        try:
            entity.set_dofs_velocity(attempt["velocity"], **attempt["kwargs"])
            any_applied = True
        except Exception as exc:
            last_exc = exc
    if any_applied:
        return True
    if last_exc is not None:
        raise last_exc
    return False


def _apply_custom_runtime_velocity(entity: Any, velocity6: np.ndarray) -> bool:
    velocity6 = np.asarray(velocity6, dtype=np.float64).reshape(6)
    linear = velocity6[:3]
    angular = velocity6[3:]
    if hasattr(entity, "set_dofs_velocity"):
        entity.set_dofs_velocity(velocity6.tolist())
        return True
    if hasattr(entity, "set_velocity"):
        entity.set_velocity(linear.tolist())
        return True
    if hasattr(entity, "set_particles_vel"):
        pts = _entity_particles_numpy(entity, kind="pos")
        if pts is None:
            return False
        vel = np.broadcast_to(linear.reshape(1, 3), pts.shape).copy()
        entity.set_particles_vel(vel)
        return True
    return False


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
    assembly_role: str
    priority_rank: int
    basic_description: str
    functional_description: str
    movement_description: str
    mesh_path: str
    json_exact_parameters: Dict[str, Any]


CLASSIFICATION_POLICY_VERSION = "v6_runtime_ctor_anchored_cloth_to_mpm"
INERTIAL_ORIGIN_POLICY_VERSION = "v2_bbox_fallback_for_nonvolume_meshes"


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


def _apply_object_level_solver_policy(
    part_phys: Dict[int, "PartPhysical"],
    youngs_threshold_gpa: Optional[float],
) -> Tuple[Dict[int, "PartPhysical"], Dict[str, Any]]:
    """
    现在只做对象级统计与记录，不再改写 solver/material_ctor。
    严格保持 solver_from_json() 的输出。
    """
    if youngs_threshold_gpa is None:
        return part_phys, {
            "mode": "disabled",
            "reason": "all_parts_youngs_threshold_gpa is None",
            "threshold_gpa": None,
            "missing_part_ids": [],
            "max_youngs_gpa": None,
            "all_below_threshold": None,
        }

    threshold_pa = float(youngs_threshold_gpa) * 1e9
    vals = []
    missing = []
    for pid, p in sorted(part_phys.items()):
        if p.youngs_pa is None:
            missing.append(pid)
        else:
            vals.append(float(p.youngs_pa))

    all_below = (
        len(missing) == 0
        and len(vals) == len(part_phys)
        and all(v < threshold_pa for v in vals)
    )

    if all_below:
        print("💚 所有 part 的杨氏模量均低于阈值，但仍严格保持 solver_from_json 的结果")
    else:
        print("💚 存在 part 超过阈值或缺失杨氏模量，仍严格保持 solver_from_json 的结果")

    new_part_phys: Dict[int, PartPhysical] = {}
    for pid, p in part_phys.items():
        p2 = dc_replace(p)
        p2.json_exact_parameters = dict(p.json_exact_parameters)
        p2.json_exact_parameters["object_level_solver_policy"] = "record_only"
        p2.json_exact_parameters["object_level_threshold_gpa"] = float(youngs_threshold_gpa)
        p2.json_exact_parameters["object_level_all_below_threshold"] = bool(all_below)
        new_part_phys[pid] = p2

    return new_part_phys, {
        "mode": "record_only",
        "reason": "keep per-part solver/material strictly from solver_from_json",
        "threshold_gpa": float(youngs_threshold_gpa),
        "missing_part_ids": missing,
        "max_youngs_gpa": max(vals) / 1e9 if vals else None,
        "all_below_threshold": bool(all_below),
    }


def _force_part_phys_to_rigid(part_phys: Dict[int, "PartPhysical"]) -> Tuple[Dict[int, "PartPhysical"], Dict[str, Any]]:
    rigid_part_phys: Dict[int, PartPhysical] = {}
    for pid, p in part_phys.items():
        p2 = dc_replace(p)
        p2.solver_family = "rigid"
        p2.simulator_material = "rigid"
        p2.material_ctor = "gs.materials.Rigid"
        p2.assembly_role = "rigid_skeleton"
        p2.json_exact_parameters = dict(p.json_exact_parameters)
        p2.json_exact_parameters["simulator_mode_override"] = "rigid"
        p2.json_exact_parameters["original_solver_family"] = p.solver_family
        p2.json_exact_parameters["original_material_ctor"] = p.material_ctor
        rigid_part_phys[pid] = p2
    return rigid_part_phys, {
        "mode": "force_rigid",
        "reason": "all parts converted to rigid for rigid-only export path",
    }

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

def _runtime_material_ctor_from_spec(part_meta: dict, material_ctor: str, assembly_role: str) -> str:
    """
    原始 material_ctor 只由 material 决定。
    运行时允许做一层最小改写：
      anchored_soft + PBD.Cloth -> MPM.Elastic
    其他情况保持不变。
    """
    ctor = str(material_ctor or "gs.materials.Rigid")
    role = str(assembly_role or "free_soft")

    if role == "anchored_soft" and ctor == "gs.materials.PBD.Cloth":
        return "gs.materials.MPM.Elastic"

    return ctor


def _runtime_solver_family_from_ctor(material_ctor_runtime: str) -> str:
    ctor = str(material_ctor_runtime)
    if ctor == "gs.materials.Rigid":
        return "rigid"
    if ctor == "gs.materials.SPH.Liquid":
        return "sph_liquid"
    if ctor == "gs.materials.PBD.Cloth":
        return "pbd_cloth"
    if ctor == "gs.materials.MPM.Elastic":
        return "mpm_elastic"
    if ctor == "gs.materials.MPM.ElastoPlastic":
        return "mpm_elastoplastic"
    if ctor == "gs.materials.MPM.Sand":
        return "mpm_sand"
    if ctor == "gs.materials.MPM.Snow":
        return "mpm_snow"
    if ctor == "gs.materials.MPM.Liquid":
        return "mpm_liquid"
    return "rigid"


def build_part_physical(
    part: Dict[str, Any],
    mesh_path: Path,
    # solver_family_override: Optional[str] = None,
) -> PartPhysical:
    part_id = int(part.get("label", -1))
    density_kgm3 = parse_density_to_kgm3(part.get("density"), default=None)
    youngs_pa = parse_modulus_to_pa(part.get("Young's Modulus (GPa)"),name = "Young's Modulus (GPa)", default=None)
    poisson = safe_optional_float(part.get("Poisson's Ratio"))
    friction = safe_optional_float(_first_present_key(part, ["Friction Coefficient", "friction", "coefficient_of_friction"]))
    restitution = safe_optional_float(_first_present_key(part, ["Restitution", "restitution"]))
    damping = safe_optional_float(_first_present_key(part, ["Damping", "damping"]))
    solver_family, simulator_material, material_ctor = solver_from_json(part)
    movement_desc = str(part.get("Movement_description", ""))
    assembly_role = classify_assembly_role(material_ctor=material_ctor, movement_desc=movement_desc)
    print(f"💚 {part['material']} solver_family={solver_family},simulator_material={simulator_material},material_ctor={material_ctor},assembly_role={assembly_role}")

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
        assembly_role=assembly_role,
        priority_rank=int(part.get("priority_rank", 0)),
        basic_description=str(part.get("Basic_description", "")),
        functional_description=str(part.get("Functional_description", "")),
        movement_description=movement_desc,
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
            "assembly_role": assembly_role,
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
    soft_parts: List[Dict[str, Any]]
    floating_parts: List[Dict[str, Any]]
    output_dir: str
    urdf_path: Optional[str]
    preview_video: Optional[str]
    grounding_offset_z: float
    object_bbox_min: List[float]
    object_bbox_max: List[float]


EXPORT_CAMERA_RESOLUTION = (960, 720)
ENVIRONMENT_SPECIAL_IDS = {
    "ground": -1,
}


def prepare_case_output_dirs(case_dir: Path) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    ensure_dir(case_dir / "videos")
    ensure_dir(case_dir / "physics")
    ensure_dir(case_dir / "rgb")
    ensure_dir(case_dir / "depth")
    ensure_dir(case_dir / "visualizations")


def _case_dir_from_metadata_path(metadata_path: Optional[str]) -> Optional[Path]:
    if not metadata_path:
        return None
    metadata_file = Path(str(metadata_path))
    if metadata_file.name in {"meta.json", "metadata.json"}:
        return metadata_file.parent
    return metadata_file


def _qa_invalid_root_for_case(case_dir: Path) -> Path:
    bucket_dir = case_dir.parent
    scene_dir = bucket_dir.parent
    if scene_dir == bucket_dir:
        return bucket_dir / "invalid_by_qa"
    return scene_dir / "invalid_by_qa" / bucket_dir.name


def _load_single_object_motion_qa(case_dir: Path, margin_px: float = 24.0) -> Dict[str, Any]:
    try:
        meta_path = case_dir / "meta.json"
        if not meta_path.exists():
            meta_path = case_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        kin = np.load(case_dir / "physics" / "rigid_kinematics.npz")
        anchor = np.load(case_dir / "physics" / "anchor_targets.npz")
        width, height = [int(v) for v in metadata.get("resolution", [EXPORT_CAMERA_RESOLUTION[0], EXPORT_CAMERA_RESOLUTION[1]])]
        pos = np.asarray(kin["com_pos"], dtype=np.float64)[:, 0, :]
        vel = np.asarray(kin["linear_vel"], dtype=np.float64)[:, 0, :]
        uv = np.asarray(anchor["com_uv"], dtype=np.float64)[:, 0, :]
        vis = np.asarray(anchor["visibility_mask"])[:, 0].astype(bool)
        bbox = np.asarray(anchor["bbox_xyxy"], dtype=np.float64)[:, 0, :]
    except Exception as exc:
        return {"valid": False, "reasons": ["qa_load_failed"], "error": f"{type(exc).__name__}: {exc}"}

    finite_uv = np.isfinite(uv).all(axis=1)
    safe = (
        finite_uv
        & vis
        & (uv[:, 0] >= margin_px)
        & (uv[:, 0] < float(width) - margin_px)
        & (uv[:, 1] >= margin_px)
        & (uv[:, 1] < float(height) - margin_px)
    )
    bbox_w = np.maximum(0.0, bbox[:, 2] - bbox[:, 0])
    bbox_h = np.maximum(0.0, bbox[:, 3] - bbox[:, 1])
    bbox_area = bbox_w * bbox_h
    speed = np.linalg.norm(vel, axis=1) if vel.size else np.zeros(0, dtype=np.float64)
    xy_radius = np.linalg.norm(pos[:, :2], axis=1) if pos.size else np.zeros(0, dtype=np.float64)
    metrics: Dict[str, Any] = {
        "visible_ratio": float(np.mean(vis)) if vis.size else 0.0,
        "safe_uv_ratio": float(np.mean(safe)) if safe.size else 0.0,
        "first_visible_frames": int(np.sum(vis[: min(8, len(vis))])) if vis.size else 0,
        "last_visible_frames": int(np.sum(vis[max(0, len(vis) - 8) :])) if vis.size else 0,
        "median_bbox_area_visible": float(np.median(bbox_area[vis])) if np.any(vis) else 0.0,
        "max_speed_mps": float(np.nanmax(speed)) if speed.size else 0.0,
        "max_xy_radius_m": float(np.nanmax(xy_radius)) if xy_radius.size else 0.0,
        "max_abs_z_m": float(np.nanmax(np.abs(pos[:, 2]))) if pos.size else 0.0,
        "initial_com_pos": pos[0].tolist() if pos.shape[0] else [],
        "final_com_pos": pos[-1].tolist() if pos.shape[0] else [],
        "margin_px": float(margin_px),
    }
    reasons: List[str] = []
    if metrics["visible_ratio"] < 0.70:
        reasons.append("low_visible_ratio")
    if metrics["safe_uv_ratio"] < 0.55:
        reasons.append("low_safe_uv_ratio")
    if metrics["first_visible_frames"] < 6:
        reasons.append("not_visible_in_context")
    if metrics["last_visible_frames"] < 4:
        reasons.append("not_visible_near_end")
    if metrics["median_bbox_area_visible"] < 64.0:
        reasons.append("tiny_or_missing_bbox")
    if metrics["max_speed_mps"] > 20.0:
        reasons.append("speed_explosion")
    if metrics["max_xy_radius_m"] > 6.0:
        reasons.append("xy_position_explosion")
    if metrics["max_abs_z_m"] > 8.0:
        reasons.append("z_position_explosion")
    metrics["valid"] = not reasons
    metrics["reasons"] = reasons
    return metrics


def _write_motion_qa(case_dir: Path, metrics: Dict[str, Any]) -> None:
    try:
        (case_dir / "qa_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def camera_intrinsics_dict(camera: Any, fallback_res: Optional[Tuple[int, int]] = None, fallback_fov_deg: Optional[float] = None) -> Dict[str, float]:
    width = None
    height = None
    if fallback_res is not None:
        width = int(fallback_res[0])
        height = int(fallback_res[1])
    if width is None or height is None:
        cam_res = getattr(camera, "res", None)
        if cam_res is not None and len(cam_res) >= 2:
            width = int(cam_res[0])
            height = int(cam_res[1])
    width = width if width is not None else int(EXPORT_CAMERA_RESOLUTION[0])
    height = height if height is not None else int(EXPORT_CAMERA_RESOLUTION[1])

    fx = getattr(camera, "f", None)
    fy = getattr(camera, "f", None)
    if fx is None or fy is None:
        fov_deg = float(fallback_fov_deg if fallback_fov_deg is not None else 35.0)
        focal = 0.5 * float(width) / math.tan(math.radians(fov_deg) / 2.0)
        fx = focal
        fy = focal

    cx = getattr(camera, "cx", None)
    cy = getattr(camera, "cy", None)
    if cx is None:
        cx = 0.5 * float(width)
    if cy is None:
        cy = 0.5 * float(height)

    near = getattr(camera, "near", None)
    far = getattr(camera, "far", None)
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "near": float(near) if near is not None else 0.05,
        "far": float(far) if far is not None else 50.0,
    }


def _normalize_vec(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return np.zeros_like(vec, dtype=np.float64)
    return vec / norm


def camera_axes_from_cfg(camera_cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cam_pos = np.asarray(camera_cfg["pos"], dtype=np.float64).reshape(3)
    cam_lookat = np.asarray(camera_cfg["lookat"], dtype=np.float64).reshape(3)
    forward = _normalize_vec(cam_lookat - cam_pos)
    up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(forward, up_guess))) > 0.98:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize_vec(np.cross(forward, up_guess))
    up = _normalize_vec(np.cross(right, forward))
    return cam_pos, right, up, forward


def project_points_to_image(
    points_world: np.ndarray,
    camera_cfg: Dict[str, Any],
    cam_intrinsics: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    cam_pos, cam_right, cam_up, cam_forward = camera_axes_from_cfg(camera_cfg)
    rel = points_world - cam_pos[None, :]
    x_cam = np.sum(rel * cam_right[None, :], axis=1)
    y_cam = np.sum(rel * cam_up[None, :], axis=1)
    z_cam = np.sum(rel * cam_forward[None, :], axis=1)
    safe_z = np.where(z_cam > 1e-8, z_cam, np.nan)
    u = float(cam_intrinsics["fx"]) * (x_cam / safe_z) + float(cam_intrinsics["cx"])
    v = float(cam_intrinsics["cy"]) - float(cam_intrinsics["fy"]) * (y_cam / safe_z)
    uv = np.stack([u, v], axis=-1).astype(np.float32)
    return uv, z_cam.astype(np.float32)


def _bbox_corners_from_bounds(bounds_min: Any, bounds_max: Any) -> np.ndarray:
    bmin = np.asarray(bounds_min, dtype=np.float64).reshape(3)
    bmax = np.asarray(bounds_max, dtype=np.float64).reshape(3)
    return np.asarray(
        [
            [bmin[0], bmin[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmax[2]],
        ],
        dtype=np.float64,
    )


def _projected_bbox_area_from_local_bounds(
    bounds_min: Any,
    bounds_max: Any,
    *,
    pos_world: Any,
    euler_deg: Any,
    camera_cfg: Dict[str, Any],
    cam_intrinsics: Dict[str, float],
    image_res: Tuple[int, int],
) -> float:
    local_corners = _bbox_corners_from_bounds(bounds_min, bounds_max)
    world_corners = np.stack(
        [
            _rotate_vec_by_euler_deg(local_corner, euler_deg) + np.asarray(pos_world, dtype=np.float64).reshape(3)
            for local_corner in local_corners
        ],
        axis=0,
    )
    uv, z_cam = project_points_to_image(
        world_corners,
        camera_cfg=camera_cfg,
        cam_intrinsics=cam_intrinsics,
    )
    valid = np.isfinite(uv).all(axis=1) & np.isfinite(z_cam) & (z_cam > 1e-5)
    if int(np.sum(valid)) < 2:
        return 0.0
    width, height = int(image_res[0]), int(image_res[1])
    u = np.clip(np.asarray(uv[valid, 0], dtype=np.float64), 0.0, float(width - 1))
    v = np.clip(np.asarray(uv[valid, 1], dtype=np.float64), 0.0, float(height - 1))
    bbox_w = max(0.0, float(np.max(u) - np.min(u)))
    bbox_h = max(0.0, float(np.max(v) - np.min(v)))
    return float(bbox_w * bbox_h)


def _scale_bounds_record_inplace(record: Dict[str, Any], scale: float) -> None:
    scale = float(scale)
    if abs(scale - 1.0) <= 1e-8:
        return
    vec3_keys = [
        "bounds_min",
        "bounds_max",
        "bounds_center",
        "bounds_size",
        "runtime_bounds_min",
        "runtime_bounds_max",
        "runtime_bounds_center",
        "runtime_bounds_size",
        "runtime_alignment_offset",
    ]
    for key in vec3_keys:
        if key not in record:
            continue
        try:
            record[key] = (np.asarray(record[key], dtype=np.float64).reshape(3) * scale).tolist()
        except Exception:
            continue


def _estimate_preview_camera_cfg_for_visibility(
    *,
    scene_label: str,
    placed_pos: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    camera_distance_mult: float,
    custom_start_positions: Optional[Sequence[np.ndarray]] = None,
) -> Dict[str, Any]:
    bbox_min = np.asarray(bbox_min, dtype=np.float64).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=np.float64).reshape(3)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
    placed_pos = np.asarray(placed_pos, dtype=np.float64).reshape(3)
    cam_distance = camera_distance_mult * max(1.78, 1.66 * float(np.max(bbox_size)) + 0.78)
    cam_height = camera_distance_mult * max(0.94, float(placed_pos[2] + bbox_min[2] + 0.65 * bbox_size[2] + 0.24))
    lookat = np.array([0.0, 0.0, float(placed_pos[2] + bbox_min[2] + 0.55 * bbox_size[2])], dtype=np.float64)
    label_l = str(scene_label).strip().lower()
    if label_l in {"random_parabola", "high_drop"}:
        if label_l == "high_drop":
            camera_distance_mult = max(camera_distance_mult, 1.36)
            cam_distance = camera_distance_mult * max(1.92, 1.76 * float(np.max(bbox_size)) + 0.80)
            cam_height = camera_distance_mult * max(0.84, float(0.44 * placed_pos[2] + bbox_min[2] + 0.15 * bbox_size[2] + 0.20))
            lookat = np.array([0.0, 0.0, float(max(0.42, 0.42 * placed_pos[2]))], dtype=np.float64)
            cam_fov = 40
        else:
            camera_distance_mult = max(camera_distance_mult, 1.28)
            cam_distance = camera_distance_mult * max(1.86, 1.72 * float(np.max(bbox_size)) + 0.78)
            cam_height = camera_distance_mult * max(0.86, float(placed_pos[2] + bbox_min[2] + 0.22 * bbox_size[2] + 0.01))
            lookat = np.array([0.0, 0.0, float(max(0.18, placed_pos[2] - 0.55))], dtype=np.float64)
            cam_fov = 36
    elif label_l.startswith("multi") and ("projectile" in label_l or "drop" in label_l):
        start_points = [placed_pos.copy()]
        if custom_start_positions:
            start_points.extend(np.asarray(pos, dtype=np.float64).reshape(3) for pos in custom_start_positions)
        start_arr = np.asarray(start_points, dtype=np.float64).reshape(-1, 3)
        num_multi_objs = int(start_arr.shape[0])
        xy_center = np.mean(start_arr[:, :2], axis=0)
        xy_span = np.ptp(start_arr[:, :2], axis=0)
        z_top = float(np.max(start_arr[:, 2]))
        lookat = np.array(
            [
                float(xy_center[0]),
                float(xy_center[1]),
                float(max(0.28, 0.26 * z_top if "drop" in label_l else 0.34 * z_top)),
            ],
            dtype=np.float64,
        )
        camera_distance_mult = max(
            camera_distance_mult,
            1.02 if num_multi_objs >= 4 else (0.96 if num_multi_objs >= 3 else 1.10),
        )
        span_ref = float(max(np.max(xy_span), np.max(bbox_size), 0.60))
        cam_distance = camera_distance_mult * max(
            1.46 if num_multi_objs >= 4 else (1.32 if num_multi_objs >= 3 else 1.58),
            (1.14 if num_multi_objs >= 4 else (1.04 if num_multi_objs >= 3 else 1.24))
            + (0.84 if num_multi_objs >= 4 else (0.78 if num_multi_objs >= 3 else 0.96)) * span_ref,
        )
        cam_height = camera_distance_mult * max(
            0.82 if num_multi_objs >= 4 else (0.76 if num_multi_objs >= 3 else 0.88),
            0.38 * z_top + 0.44,
        )
        cam_fov = 48 if num_multi_objs >= 4 else (46 if num_multi_objs >= 3 else 48)
        cam_pos = np.array(
            [
                float(lookat[0] + (0.08 if num_multi_objs >= 3 else 0.62) * cam_distance),
                float(lookat[1] - (0.88 if num_multi_objs >= 3 else 0.48) * cam_distance),
                float(1.08 * cam_height),
            ],
            dtype=np.float64,
        )
    else:
        cam_fov = 32

    if not (label_l.startswith("multi") and ("projectile" in label_l or "drop" in label_l)):
        cam_pos = np.array([0.66 * cam_distance, -0.82 * cam_distance, 1.08 * cam_height], dtype=np.float64)
    cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return {
        "pos": cam_pos.astype(np.float64).tolist(),
        "lookat": lookat.astype(np.float64).tolist(),
        "up": cam_up.astype(np.float64).tolist(),
        "fov": float(cam_fov),
        "res": [int(EXPORT_CAMERA_RESOLUTION[0]), int(EXPORT_CAMERA_RESOLUTION[1])],
        "model": "pinhole",
    }


def bbox_xyxy_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((4,), dtype=np.float32)
    return np.asarray([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def compute_anchor_targets(
    seg_frames: np.ndarray,
    depth_metric_frames: np.ndarray,
    com_pos_frames: np.ndarray,
    object_ids: np.ndarray,
    seg_ids: np.ndarray,
    camera_cfg: Dict[str, Any],
    cam_intrinsics: Dict[str, float],
) -> Dict[str, np.ndarray]:
    seg_frames = np.asarray(seg_frames, dtype=np.int32)
    depth_metric_frames = np.asarray(depth_metric_frames, dtype=np.float32)
    com_pos_frames = np.asarray(com_pos_frames, dtype=np.float32)
    object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
    seg_ids = np.asarray(seg_ids, dtype=np.int32).reshape(-1)
    num_frames, num_objects = com_pos_frames.shape[:2]

    com_uv, _ = project_points_to_image(
        com_pos_frames.reshape(-1, 3),
        camera_cfg=camera_cfg,
        cam_intrinsics=cam_intrinsics,
    )
    com_uv = com_uv.reshape(num_frames, num_objects, 2).astype(np.float32)
    bbox_xyxy = np.zeros((num_frames, num_objects, 4), dtype=np.float32)
    visibility_mask = np.zeros((num_frames, num_objects), dtype=np.uint8)
    center_depth = np.zeros((num_frames, num_objects), dtype=np.float32)

    for frame_idx in range(num_frames):
        frame_seg = seg_frames[frame_idx]
        frame_depth = depth_metric_frames[frame_idx]
        for obj_idx, seg_id in enumerate(seg_ids):
            mask = frame_seg == int(seg_id)
            if not np.any(mask):
                continue
            visibility_mask[frame_idx, obj_idx] = 1
            bbox_xyxy[frame_idx, obj_idx] = bbox_xyxy_from_mask(mask)
            center_depth[frame_idx, obj_idx] = float(np.median(frame_depth[mask]))

    return {
        "object_ids": object_ids.astype(np.int32),
        "seg_ids": seg_ids.astype(np.int32),
        "com_uv": com_uv.astype(np.float32),
        "bbox_xyxy": bbox_xyxy.astype(np.float32),
        "visibility_mask": visibility_mask.astype(np.uint8),
        "center_depth": center_depth.astype(np.float32),
    }


def build_segmentation_mapping(scene: Any, entities: Sequence[Any], object_ids: Sequence[int]) -> Dict[int, int]:
    context = getattr(getattr(scene, "visualizer", None), "_context", None)
    seg_idxc_map = getattr(context, "seg_idxc_map", None)
    if seg_idxc_map is None:
        return {}
    entity_idx_to_object_id: Dict[int, int] = {}
    for ent, object_id in zip(entities, object_ids):
        ent_idx = getattr(ent, "idx", None)
        if ent_idx is not None:
            entity_idx_to_object_id[int(ent_idx)] = int(object_id)
    mapping: Dict[int, int] = {}
    for seg_idx, seg_key in seg_idxc_map.items():
        if int(seg_idx) == 0:
            continue
        entity_idx = int(seg_key[0]) if isinstance(seg_key, tuple) and len(seg_key) > 0 else int(seg_key)
        if entity_idx in entity_idx_to_object_id:
            mapping[int(seg_idx)] = int(entity_idx_to_object_id[entity_idx]) + 1
    return mapping


def remap_segmentation(seg_raw: Any, seg_mapping: Dict[int, int]) -> np.ndarray:
    seg_array = np.asarray(to_numpy(seg_raw), dtype=np.int64)
    if seg_array.ndim == 3 and seg_array.shape[-1] == 1:
        seg_array = seg_array[..., 0]
    if seg_array.ndim == 3 and seg_array.shape[0] == 1:
        seg_array = seg_array[0]
        if seg_array.ndim == 3 and seg_array.shape[-1] == 1:
            seg_array = seg_array[..., 0]
    remapped = np.zeros(seg_array.shape, dtype=np.int32)
    for raw_idx, mapped_value in seg_mapping.items():
        remapped[seg_array == int(raw_idx)] = int(mapped_value)
    return remapped


def _pairwise_contact_from_aabbs(aabbs: Sequence[Optional[np.ndarray]], clearance: float = 0.003) -> np.ndarray:
    num_objects = len(aabbs)
    graph = np.zeros((num_objects, num_objects), dtype=np.uint8)
    for idx_a in range(num_objects):
        if aabbs[idx_a] is None:
            continue
        for idx_b in range(idx_a + 1, num_objects):
            if aabbs[idx_b] is None:
                continue
            if _aabb_overlaps(aabbs[idx_a], aabbs[idx_b], clearance=clearance):
                graph[idx_a, idx_b] = 1
                graph[idx_b, idx_a] = 1
    return graph


def _count_bucket_from_num_objects(num_objects: int) -> str:
    num_objects = int(num_objects)
    if num_objects <= 1:
        return "count_01"
    if num_objects == 2:
        return "count_02"
    if num_objects in (3, 4):
        return "count_03_04"
    if num_objects in (5, 6):
        return "count_05_06"
    return f"count_{num_objects:02d}"


def _scene_layout_from_sources(scene_label: str, object_sources: Sequence[str]) -> Tuple[str, str]:
    sources = [str(src) for src in object_sources]
    num_objects = len(sources)
    count_bucket = _count_bucket_from_num_objects(num_objects)
    label = str(scene_label or "").strip().lower()
    if label.startswith("multi") and ("projectile" in label or "drop" in label):
        return "multi_object_free_motion", count_bucket
    has_aux_object = any(src in {"custom_object", "physxnet_aux"} for src in sources)
    if has_aux_object and num_objects >= 2:
        return "interaction_pair_plus_dynamic", count_bucket
    return "single_object_preview", count_bucket


def _motion_group_from_scene_label(scene_label: str) -> str:
    label = str(scene_label or "").strip().lower()
    if not label:
        return "unknown"
    if label.startswith("multi") and "projectile" in label:
        return "projectile_motion"
    if label.startswith("multi") and "drop" in label:
        return "gravity_drop"
    if label.startswith("static_"):
        if label == "static_highdrop":
            return "gravity_drop"
        return "static_placement"
    if label.startswith("entry_"):
        return "entry_motion"
    if label == "random_parabola":
        return "projectile_motion"
    if label == "high_drop":
        return "gravity_drop"
    if label == "legacy_random":
        return "legacy_random"
    return "other"


def _interaction_pattern_from_case(
    scene_label: str,
    scene_composition: str,
    object_sources: Sequence[str],
    apply_object_entry_velocity: bool,
) -> str:
    label = str(scene_label or "").strip().lower()
    sources = [str(src) for src in object_sources]
    if label.startswith("multi") and "projectile" in label:
        return "multi_object_independent_projectile_motion"
    if label.startswith("multi") and "drop" in label:
        return "multi_object_independent_gravity_drop"
    has_interaction_object = any(src in {"custom_object", "physxnet_aux"} for src in sources)
    if has_interaction_object:
        if label in {"static_center", "static_left", "static_right"}:
            return "striker_hits_static_target"
        if label == "static_highdrop":
            return "striker_hits_falling_target"
        if label.startswith("entry_"):
            return "co_moving_collision"
        return "striker_target_interaction"
    if label == "static_highdrop" or label == "high_drop":
        return "gravity_drop"
    if label == "random_parabola":
        return "single_object_projectile_motion"
    if bool(apply_object_entry_velocity):
        return "single_object_entry_motion"
    if scene_composition == "single_object_preview":
        return "single_object_static_preview"
    return "generic_interaction"


def _object_motion_fields(
    *,
    object_index: int,
    source_tag: str,
    scene_label: str,
    scene_composition: str,
    has_custom_object: bool,
    apply_object_entry_velocity: bool,
    role_hint: Optional[str] = None,
    motion_type_hint: Optional[str] = None,
    motion_group_hint: Optional[str] = None,
) -> Dict[str, str]:
    source = str(source_tag)
    label = str(scene_label)
    if role_hint or motion_type_hint or motion_group_hint:
        return {
            "role": str(role_hint or ("initiator" if bool(apply_object_entry_velocity) else "bystander")),
            "object_motion_type": str(motion_type_hint or label or "other"),
            "object_motion_group": str(motion_group_hint or _motion_group_from_scene_label(label)),
        }
    if source == "custom_object":
        return {
            "role": "initiator",
            "object_motion_type": "striker_hit",
            "object_motion_group": "striker",
        }
    if source == "physxnet_aux":
        return {
            "role": "bystander",
            "object_motion_type": "static_rest",
            "object_motion_group": "auxiliary_static",
        }
    if source == "physxnet_main":
        role = "target" if has_custom_object else ("initiator" if bool(apply_object_entry_velocity) else "target")
        return {
            "role": role,
            "object_motion_type": label,
            "object_motion_group": _motion_group_from_scene_label(label),
        }
    role = "bystander"
    if not has_custom_object and object_index == 0 and bool(apply_object_entry_velocity):
        role = "initiator"
    return {
        "role": role,
        "object_motion_type": "passive_dynamic" if bool(apply_object_entry_velocity) else "passive_static",
        "object_motion_group": "auxiliary",
    }


def _contact_graph_with_environment(
    object_aabbs: Sequence[Optional[np.ndarray]],
    object_ids: Sequence[int],
    ground_height: float = 0.0,
    clearance: float = 0.01,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    graph = _pairwise_contact_from_aabbs(object_aabbs, clearance=clearance)
    env_contacts: List[Dict[str, Any]] = []
    for obj_idx, (object_id, aabb) in enumerate(zip(object_ids, object_aabbs)):
        if aabb is None:
            continue
        aabb_arr = np.asarray(aabb, dtype=np.float64).reshape(2, 3)
        if float(aabb_arr[0, 2]) <= float(ground_height) + float(clearance):
            env_contacts.append(
                {
                    "object_idx": int(obj_idx),
                    "object_id": int(object_id),
                    "environment_name": "ground",
                    "environment_id": int(ENVIRONMENT_SPECIAL_IDS["ground"]),
                    "impulse_peak": 0.0,
                }
            )
    return graph.astype(np.uint8), env_contacts


def _summarize_contact_windows(contact_graph_frames: np.ndarray, object_ids: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    contact_graph_frames = np.asarray(contact_graph_frames, dtype=np.uint8)
    object_ids = np.asarray(object_ids, dtype=np.int32).reshape(-1)
    num_frames, num_objects, _ = contact_graph_frames.shape
    frame_phase = np.zeros((num_frames,), dtype=np.int8)
    event_windows: List[Dict[str, Any]] = []
    collision_events: List[Dict[str, Any]] = []
    window_id = 0

    for idx_a in range(num_objects):
        for idx_b in range(idx_a + 1, num_objects):
            active = contact_graph_frames[:, idx_a, idx_b] > 0
            if not np.any(active):
                continue
            start = None
            for frame_idx in range(num_frames + 1):
                is_active = bool(active[frame_idx]) if frame_idx < num_frames else False
                if is_active and start is None:
                    start = frame_idx
                elif (not is_active) and start is not None:
                    end = frame_idx - 1
                    peak = int(start)
                    event_type = "contact_onset" if end == start else "sustained_contact"
                    record = {
                        "window_id": int(window_id),
                        "window_type": event_type,
                        "participants": [int(object_ids[idx_a]), int(object_ids[idx_b])],
                        "participant_indices": [int(idx_a), int(idx_b)],
                        "start_frame": int(start),
                        "peak_frame": int(peak),
                        "end_frame": int(end),
                    }
                    event_windows.append(record)
                    collision_events.append(
                        {
                            "event_id": int(len(collision_events)),
                            "participants": [int(object_ids[idx_a]), int(object_ids[idx_b])],
                            "object_indices": [int(idx_a), int(idx_b)],
                            "start_frame": int(start),
                            "peak_frame": int(peak),
                            "end_frame": int(end),
                            "impulse_peak": 0.0,
                            "contact_duration": int(end - start + 1),
                        }
                    )
                    window_id += 1
                    start = None

    active_pairs_per_frame = contact_graph_frames.sum(axis=(1, 2)) // 2 if num_frames > 0 else np.zeros((0,), dtype=np.int32)
    for frame_idx in range(num_frames):
        if int(active_pairs_per_frame[frame_idx]) > 0:
            onset = False
            sustained = False
            for idx_a in range(num_objects):
                for idx_b in range(idx_a + 1, num_objects):
                    if contact_graph_frames[frame_idx, idx_a, idx_b] == 0:
                        continue
                    was_active = frame_idx > 0 and bool(contact_graph_frames[frame_idx - 1, idx_a, idx_b])
                    onset = onset or (not was_active)
                    sustained = sustained or was_active
            frame_phase[frame_idx] = 1 if onset else 2
        else:
            recent_contact = frame_idx > 0 and int(active_pairs_per_frame[frame_idx - 1]) > 0
            frame_phase[frame_idx] = 3 if recent_contact else 0
    if num_frames > 0:
        tail = min(3, num_frames)
        for frame_idx in range(num_frames - tail, num_frames):
            if frame_phase[frame_idx] == 0:
                frame_phase[frame_idx] = 5
    return frame_phase.astype(np.int8), event_windows, collision_events


def summarize_environment_contact_windows(
    environment_contact_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[int, int, str], List[Dict[str, Any]]] = {}
    for event in environment_contact_events:
        participants = event.get("participants", [])
        if len(participants) != 2:
            continue
        if int(participants[1]) >= 0:
            continue
        key = (
            int(participants[0]),
            int(participants[1]),
            str(event.get("environment_name", "environment")),
        )
        grouped.setdefault(key, []).append(dict(event))

    windows: List[Dict[str, Any]] = []
    window_id = 0
    for (_, _, env_name), records in grouped.items():
        records = sorted(records, key=lambda item: int(item.get("frame_idx", item.get("start_frame", -1))))
        if not records:
            continue
        start_idx = 0
        while start_idx < len(records):
            end_idx = start_idx
            while (
                end_idx + 1 < len(records)
                and int(records[end_idx + 1].get("frame_idx", records[end_idx + 1].get("start_frame", -1)))
                <= int(records[end_idx].get("frame_idx", records[end_idx].get("start_frame", -1))) + 1
            ):
                end_idx += 1
            chunk = records[start_idx : end_idx + 1]
            impulses = [float(item.get("impulse_peak", 0.0)) for item in chunk]
            peak_rel = int(np.argmax(np.asarray(impulses, dtype=np.float32))) if impulses else 0
            peak_record = chunk[peak_rel]
            duration = int(
                int(chunk[-1].get("frame_idx", chunk[-1].get("end_frame", -1)))
                - int(chunk[0].get("frame_idx", chunk[0].get("start_frame", -1)))
                + 1
            )
            windows.append(
                {
                    "window_id": int(window_id),
                    "window_type": "contact_onset" if duration <= 1 else "sustained_contact",
                    "participants": list(chunk[0]["participants"]),
                    "participant_indices": list(chunk[0].get("object_indices", [])),
                    "environment_name": env_name,
                    "start_frame": int(chunk[0].get("frame_idx", chunk[0].get("start_frame", -1))),
                    "peak_frame": int(peak_record.get("frame_idx", peak_record.get("peak_frame", -1))),
                    "end_frame": int(chunk[-1].get("frame_idx", chunk[-1].get("end_frame", -1))),
                    "impulse_peak": float(max(impulses) if impulses else 0.0),
                    "contact_duration": int(duration),
                }
            )
            window_id += 1
            start_idx = end_idx + 1
    return windows


def _build_flow_fallback(
    com_uv: np.ndarray,
    visibility_mask: np.ndarray,
    seg_frames: np.ndarray,
) -> np.ndarray:
    com_uv = np.asarray(com_uv, dtype=np.float32)
    visibility_mask = np.asarray(visibility_mask, dtype=np.uint8)
    seg_frames = np.asarray(seg_frames, dtype=np.int32)
    num_frames = seg_frames.shape[0]
    if num_frames <= 1:
        return np.zeros((0,) + seg_frames.shape[1:] + (2,), dtype=np.float32)
    flow = np.zeros((num_frames - 1, seg_frames.shape[1], seg_frames.shape[2], 2), dtype=np.float32)
    num_objects = com_uv.shape[1] if com_uv.ndim >= 3 else 0
    for frame_idx in range(num_frames - 1):
        for obj_idx in range(num_objects):
            seg_id = obj_idx + 1
            if visibility_mask[frame_idx, obj_idx] == 0 or visibility_mask[frame_idx + 1, obj_idx] == 0:
                continue
            delta = com_uv[frame_idx + 1, obj_idx] - com_uv[frame_idx, obj_idx]
            mask = seg_frames[frame_idx] == int(seg_id)
            if np.any(mask):
                flow[frame_idx][mask] = delta.astype(np.float32)
    return flow


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_rotmat_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rigid_entity_energy_components(entity: Any, gravity: Iterable[float] = (0.0, 0.0, -9.81)) -> Tuple[float, float, float]:
    gravity = np.asarray(tuple(gravity), dtype=np.float64)
    total_linear = 0.0
    total_rot = 0.0
    total_potential = 0.0
    links = list(getattr(entity, "links", []))

    if not links:
        mass = float(entity.get_mass()) if hasattr(entity, "get_mass") else 0.0
        pos = np.asarray(to_numpy(entity.get_pos()), dtype=np.float64).reshape(3)
        vel = np.asarray(to_numpy(entity.get_vel()), dtype=np.float64).reshape(3)
        ang = np.asarray(to_numpy(entity.get_ang()), dtype=np.float64).reshape(3)
        total_linear = 0.5 * mass * float(np.dot(vel, vel))
        inertia_local = getattr(entity, "inertial_i", None)
        if inertia_local is not None:
            link_quat = np.asarray(to_numpy(entity.get_quat()), dtype=np.float64).reshape(4)
            inertial_quat = getattr(entity, "inertial_quat", None)
            if inertial_quat is None:
                inertial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                inertial_quat = np.asarray(inertial_quat, dtype=np.float64).reshape(4)
            world_inertial_quat = _quat_mul_wxyz(link_quat, inertial_quat)
            rot = _quat_to_rotmat_wxyz(world_inertial_quat)
            inertia_world = rot @ np.asarray(inertia_local, dtype=np.float64) @ rot.T
            total_rot = 0.5 * float(ang @ inertia_world @ ang)
        total_potential = -mass * float(np.dot(gravity, pos))
        return total_linear, total_rot, total_potential

    try:
        links_vel = to_numpy(entity.get_links_vel(ref="link_com"))
        if np.asarray(links_vel).ndim == 3:
            links_vel = np.asarray(links_vel)[0]
        links_vel = np.asarray(links_vel, dtype=np.float64)
    except Exception:
        links_vel = None
    try:
        links_ang = to_numpy(entity.get_links_ang())
        if np.asarray(links_ang).ndim == 3:
            links_ang = np.asarray(links_ang)[0]
        links_ang = np.asarray(links_ang, dtype=np.float64)
    except Exception:
        links_ang = None

    for link_idx, link in enumerate(links):
        mass = float(link.get_mass())
        link_pos = np.asarray(to_numpy(link.get_pos()), dtype=np.float64).reshape(3)
        link_quat = np.asarray(to_numpy(link.get_quat()), dtype=np.float64).reshape(4)
        inertial_pos = getattr(link, "inertial_pos", None)
        if inertial_pos is None:
            com_pos = link_pos
        else:
            com_pos = link_pos + _quat_to_rotmat_wxyz(link_quat) @ np.asarray(inertial_pos, dtype=np.float64).reshape(3)
        if links_vel is None:
            link_vel = np.asarray(to_numpy(link.get_vel()), dtype=np.float64).reshape(3)
        else:
            link_vel = np.asarray(links_vel[link_idx], dtype=np.float64).reshape(3)
        if links_ang is None:
            link_ang = np.asarray(to_numpy(link.get_ang()), dtype=np.float64).reshape(3)
        else:
            link_ang = np.asarray(links_ang[link_idx], dtype=np.float64).reshape(3)

        total_linear += 0.5 * mass * float(np.dot(link_vel, link_vel))
        inertia_local = getattr(link, "inertial_i", None)
        if inertia_local is not None:
            inertial_quat = getattr(link, "inertial_quat", None)
            if inertial_quat is None:
                inertial_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                inertial_quat = np.asarray(inertial_quat, dtype=np.float64).reshape(4)
            world_inertial_quat = _quat_mul_wxyz(link_quat, inertial_quat)
            rot = _quat_to_rotmat_wxyz(world_inertial_quat)
            inertia_world = rot @ np.asarray(inertia_local, dtype=np.float64) @ rot.T
            total_rot += 0.5 * float(link_ang @ inertia_world @ link_ang)
        total_potential += -mass * float(np.dot(gravity, com_pos))

    return total_linear, total_rot, total_potential


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


def voxel_fill_mesh(
    mesh: trimesh.Trimesh,
    pitch: float,
    *,
    prefer_closed_proxy: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mesh = sanitize_mesh(mesh)
    meta: Dict[str, Any] = {
        "pitch": float(pitch),
        "fill_mode": "unknown",
        "mesh_is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "mesh_is_volume": bool(getattr(mesh, "is_volume", False)),
        "prefer_closed_proxy": bool(prefer_closed_proxy),
    }
    try:
        if bool(prefer_closed_proxy) and not (meta["mesh_is_watertight"] and meta["mesh_is_volume"]):
            closed_mesh, closed_meta = _build_closed_voxel_fill_proxy(mesh=mesh, pitch=float(pitch))
            meta["closed_proxy_meta"] = closed_meta
            if closed_mesh is not None and bool(getattr(closed_mesh, "is_volume", False)):
                mesh = closed_mesh
                meta["fill_mode"] = "closed_proxy_fill"
                meta["mesh_is_watertight_after_proxy"] = bool(getattr(mesh, "is_watertight", False))
                meta["mesh_is_volume_after_proxy"] = bool(getattr(mesh, "is_volume", False))

        vox = mesh.voxelized(pitch)
        try:
            vox = vox.fill()
            if meta["fill_mode"] == "unknown":
                meta["fill_mode"] = "interior_fill"
        except Exception:
            if meta["fill_mode"] == "unknown":
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
        pts = np.asarray(vox.points, dtype=np.float64)
        if pts.size == 0:
            raise ValueError("empty voxel points")
        solid = trimesh.voxel.ops.multibox(
            centers=pts,
            pitch=float(pitch),
            remove_internal_faces=True,
        )
        solid = sanitize_mesh(solid)
        meta["num_voxels"] = int(len(pts))
        meta["mesh_builder"] = "multibox"
        meta["num_vertices"] = int(len(solid.vertices))
        meta["num_faces"] = int(len(solid.faces))
        return solid, meta
    except Exception as e:
        raise RuntimeError(f"voxel_fill_mesh_collision failed: {e}")


def build_rigid_collision_proxy(mesh: trimesh.Trimesh, base_pitch: float) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    mesh = sanitize_mesh(mesh)
    extents = np.maximum(np.asarray(mesh.extents, dtype=np.float64), 1e-6)
    min_extent = float(np.min(extents))
    max_extent = float(np.max(extents))
    adaptive_pitch = float(
        np.clip(
            min(float(base_pitch), 0.10 * min_extent, 0.04 * max_extent),
            0.004,
            float(base_pitch),
        )
    )

    mesh_is_closed = bool(getattr(mesh, "is_watertight", False) and getattr(mesh, "is_volume", False))
    meta: Dict[str, Any] = {
        "requested_pitch": float(base_pitch),
        "pitch": float(adaptive_pitch),
        "mesh_is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "mesh_is_volume": bool(getattr(mesh, "is_volume", False)),
        "proxy_kind": "original_mesh" if mesh_is_closed else "voxel_filled_proxy",
    }

    if mesh_is_closed:
        meta["fill_mode"] = "original_mesh"
        meta["num_vertices"] = int(len(mesh.vertices))
        meta["num_faces"] = int(len(mesh.faces))
        return mesh.copy(), meta

    solid, fill_meta = voxel_fill_mesh_collision(mesh, adaptive_pitch)
    meta.update(fill_meta)
    return solid, meta


def build_rigid_visual_proxy(
    mesh: trimesh.Trimesh,
    enable_double_sided_shell: bool = True,
    prefer_solid_visual: bool = False,
    solid_proxy_pitch: Optional[float] = None,
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    mesh = sanitize_mesh(mesh)
    meta: Dict[str, Any] = {
        "visual_proxy_kind": "original_mesh",
        "mesh_is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "mesh_is_volume": bool(getattr(mesh, "is_volume", False)),
    }

    if prefer_solid_visual and not meta["mesh_is_volume"]:
        if solid_proxy_pitch is None:
            raise ValueError("solid_proxy_pitch is required when prefer_solid_visual=True")
        solid_visual, solid_meta = build_rigid_collision_proxy(mesh, float(solid_proxy_pitch))
        meta.update(solid_meta)
        meta["visual_proxy_kind"] = "collision_proxy_visual"
        meta["num_vertices"] = int(len(solid_visual.vertices))
        meta["num_faces"] = int(len(solid_visual.faces))
        return solid_visual, meta

    # Many scanned rigid parts are effectively single-sided shells.
    # Duplicate faces with reversed winding so the visual stays visible
    # when the camera looks at the back side (for example, bowl bottoms
    # viewed from inside the container).
    if enable_double_sided_shell and not meta["mesh_is_volume"]:
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(verts) > 0 and len(faces) > 0:
            rev_faces = faces[:, ::-1]
            double_sided = trimesh.Trimesh(
                vertices=verts.copy(),
                faces=np.concatenate([faces, rev_faces], axis=0),
                process=False,
            )
            double_sided = sanitize_mesh(double_sided)
            meta["visual_proxy_kind"] = "double_sided_shell"
            meta["num_vertices"] = int(len(double_sided.vertices))
            meta["num_faces"] = int(len(double_sided.faces))
            return double_sided, meta

    meta["num_vertices"] = int(len(mesh.vertices))
    meta["num_faces"] = int(len(mesh.faces))
    return mesh.copy(), meta


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
    use_collision_for_visual: bool = False,
) -> None:
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geom = ET.SubElement(visual, "geometry")
    visual_path = collision_mesh_relpath if (use_collision_for_visual and collision_mesh_relpath is not None) else visual_mesh_relpath
    ET.SubElement(geom, "mesh", filename=visual_path, scale="1 1 1")
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
        _maybe_add_dynamics(j2)
        return


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


def _object_is_container_like(object_name: str, category: str) -> bool:
    category_norm = str(category or "").strip().lower()
    object_name_norm = str(object_name or "").strip().lower()
    return (
        "container" in category_norm
        or any(tok in object_name_norm for tok in ("bowl", "cup", "mug", "glass", "pot", "bucket"))
    )


def _choose_anchor_for_group(group: GroupRecord, group_mesh: trimesh.Trimesh) -> np.ndarray:
    if len(group.params) >= 6:
        return np.asarray(group.params[3:6], dtype=np.float64)
    return np.asarray(group_mesh.bounding_box.centroid, dtype=np.float64)


def _estimate_part_mass(mesh: trimesh.Trimesh, p: PartPhysical, fallback_density_kgm3: float) -> float:
    density = p.density_kgm3 if p.density_kgm3 is not None else fallback_density_kgm3
    return max(mesh_volume_fallback(mesh) * float(density), 1e-4)


def _mesh_inertial_origin_xyz(mesh: trimesh.Trimesh) -> str:
    center = None
    try:
        if bool(getattr(mesh, "is_watertight", False)) and bool(getattr(mesh, "is_volume", False)):
            candidate = np.asarray(mesh.center_mass, dtype=np.float64)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            if np.all(np.isfinite(candidate)) and bounds.shape == (2, 3):
                # Thin shell scans can produce finite but nonsensical center_mass values.
                # Only trust trimesh's volume COM when the mesh is a proper closed volume
                # and the reported center remains inside the mesh AABB.
                eps = np.maximum(1e-6, 1e-3 * np.maximum(bounds[1] - bounds[0], 1e-6))
                if np.all(candidate >= (bounds[0] - eps)) and np.all(candidate <= (bounds[1] + eps)):
                    center = candidate
    except Exception:
        center = None

    if center is None:
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


def _make_pbd_cloth_material_from_part(
    gs: Any,
    density: float,
    friction: Optional[float],
    youngs: Optional[float],
    damping: Optional[float],
    *,
    free_cloth: bool = False,
):
    friction_val = float(np.clip(friction if friction is not None else (0.42 if free_cloth else 0.15), 1e-3, 5.0))
    youngs_raw = float(max(youngs if youngs is not None else 1e5, 1e3))
    if free_cloth:
        youngs_val = float(np.clip(youngs_raw, 8.0e3, 3.5e4))
        air_resistance = float(np.clip(damping if damping is not None else 0.045, 0.012, 0.18))
        stretch_compliance = float(np.clip(1.0 / youngs_val, 2.5e-5, 3.5e-4))
        bending_compliance = float(np.clip(6.0 / youngs_val, 2.0e-4, 1.8e-2))
    else:
        youngs_val = youngs_raw
        air_resistance = float(max(damping if damping is not None else 1e-3, 1e-6))
        stretch_compliance = float(np.clip(1.0 / youngs_val, 1e-9, 1e-3))
        bending_compliance = float(np.clip(10.0 / youngs_val, 1e-8, 5e-2))
    # Cloth rho is area density (kg/m^2), not volumetric density.
    rho_2d = float(max(density * (0.0012 if free_cloth else 0.002), 0.12 if free_cloth else 0.2))
    return gs.materials.PBD.Cloth(
        rho=rho_2d,
        static_friction=friction_val,
        kinetic_friction=friction_val,
        stretch_compliance=stretch_compliance,
        bending_compliance=bending_compliance,
        air_resistance=air_resistance,
    )


def _part_goes_to_rigid_skeleton(p: PartPhysical) -> bool:
    return str(p.assembly_role) == "rigid_skeleton"


def _object_is_small_liquid_container(
    object_name: Any,
    category: Any,
    part_phys: Dict[int, PartPhysical],
    dim_m: Optional[np.ndarray],
) -> bool:
    has_liquid = any(
        str(p.material_ctor) in ("gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid")
        for p in part_phys.values()
    )
    if not has_liquid:
        return False

    text = norm_text(f"{object_name} {category}")
    container_hints = [
        "bowl",
        "cup",
        "mug",
        "glass",
        "container",
        "jar",
        "bottle",
        "can",
        "pot",
        "kettle",
    ]
    if not has_any(text, container_hints):
        return False

    if dim_m is None:
        return True

    return float(np.max(np.asarray(dim_m, dtype=np.float64))) <= 0.35


def _liquid_container_object_scale_boost(
    object_name: Any,
    category: Any,
    part_phys: Dict[int, PartPhysical],
    dim_m: Optional[np.ndarray],
) -> float:
    if _object_is_small_liquid_container(object_name, category, part_phys, dim_m):
        return 1.20
    return 1.0

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
    all_parts_youngs_threshold_gpa: Optional[float] = None,
    rigid_visual_double_sided_shell: bool = True,
    simulator_mode: str = "rigid",
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
            - soft_parts (list): 软体部件的信息列表
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
    object_id = Path(str(object_id)).stem
    version_root = physx_root / version
    json_path = json_override if json_override is not None else version_root / "finaljson" / f"{object_id}.json"
    objs_dir = version_root / "partseg" / object_id / "objs"

    # 检查必要文件是否存在，不存在则提前报错
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not objs_dir.exists():
        raise FileNotFoundError(objs_dir)

    # ── 缓存检测：若 URDF 和 metadata 均已存在，直接读取返回，跳过重新生成 ──
    out_dir = physxnet_object_cache_dir(output_root, object_id)
    _cached_urdf = out_dir / "rigid" / f"{object_id}.urdf"
    _cached_meta = out_dir / f"meta" / "metadata.json"
    # 缓存检测需要感知 solver_family_override：
    # 不同 override 导出的资产内容不同（rigid 无 soft_parts，mpm 有），不能混用缓存
    # 在 metadata 中记录生成时使用的 override，若与当前不一致则重新生成
    _cache_valid = False
    if _cached_urdf.exists() and _cached_meta.exists():
        try:
            with open(_cached_meta, "r", encoding="utf-8") as _f:
                _m = json.load(_f)
        except Exception as exc:
            print(f"[Cache] {object_id}: invalid cached metadata, rebuild asset cache ({exc})")
            _m = None
        if isinstance(_m, dict):
            _cached_override = _m.get("solver_family_override", None)
            _cached_policy = _m.get("classification_policy_version", None)
            _cached_inertial_origin_policy = _m.get("inertial_origin_policy_version", None)
            _cached_threshold = _m.get("all_parts_youngs_threshold_gpa", None)
            _cached_double_sided = bool(_m.get("rigid_visual_double_sided_shell", True))
            _cached_rigid_group_mode = _m.get("object_solver_policy", {}).get("rigid_only_group_info_mode", None)
            _current_sim_mode = str(simulator_mode).strip().lower()
            if (
                _cached_override == solver_family_override
                and _cached_policy == CLASSIFICATION_POLICY_VERSION
                and _cached_inertial_origin_policy == INERTIAL_ORIGIN_POLICY_VERSION
                and _cached_threshold == all_parts_youngs_threshold_gpa
                and _cached_double_sided == bool(rigid_visual_double_sided_shell)
                and (_current_sim_mode != "rigid" or _cached_rigid_group_mode == "collapsed_to_fixed_assembly")
            ):
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
            soft_parts=_m["soft_parts"],
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
    ensure_dir(out_dir / "rigid_visuals")
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
            continue
        mesh = load_mesh(mesh_path)
        mesh = yup_to_zup_mesh(mesh)   # Y-up → Z-up 坐标系转换（绕X轴旋转-90°）
        part_meshes[pid] = mesh
        part_phys[pid] = build_part_physical(part, mesh_path)  # 解析 JSON 中的物理参数
        # print(f"🤍 Part {part_phys[pid].name}: {part_phys[pid].material_ctor}")

    if not part_meshes:
        raise ValueError(f"No valid part meshes found for object {object_id}")

    part_phys, object_solver_policy = _apply_object_level_solver_policy(
        part_phys,
        youngs_threshold_gpa=all_parts_youngs_threshold_gpa,
    )
    if str(simulator_mode).strip().lower() == "rigid":
        part_phys, object_solver_policy = _force_part_phys_to_rigid(part_phys)
        # Rigid-only dataset export treats each PhysXNet asset as one rigid
        # assembly.  The original PhysXNet group_info may contain revolute /
        # prismatic articulation hints (e.g. headphone ear cups); leaving those
        # active makes parts drift after the first frames.  Collapse all groups
        # into the base group so every part is connected by fixed joints.
        meta = dict(meta)
        meta["group_info_original"] = meta.get("group_info", {})
        meta["group_info"] = {
            "0": sorted(int(pid) for pid in part_meshes.keys()),
        }
        object_solver_policy["rigid_only_group_info_mode"] = "collapsed_to_fixed_assembly"
    # print(f"Object solver policy: {object_solver_policy}")
    # print(f"Part physical properties: {part_phys}")
    # parse_group_info must happen after the rigid-only group collapse above.
    parsed_groups = parse_group_info(meta.get("group_info", {}))

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
    liquid_container_scale_boost = _liquid_container_object_scale_boost(
        object_name=meta.get("object_name", object_id),
        category=meta.get("category", "Unknown"),
        part_phys=part_phys,
        dim_m=dim_m,
    )
    if liquid_container_scale_boost > 1.0:
        object_scale *= float(liquid_container_scale_boost)
        print(
            f"🫙 liquid_container_scale_boost applied x{liquid_container_scale_boost:.2f} "
            f"for object={meta.get('object_name', object_id)}"
        )

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
    soft_parts: List[Dict[str, Any]] = []
    floating_parts: List[Dict[str, Any]] = []

    object_has_liquid = any(
        str(p.material_ctor) in ("gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid")
        for p in part_phys.values()
    )
    rigid_visual_alpha = 1.0
    if object_has_liquid and _object_is_container_like(meta.get("object_name", object_id), meta.get("category", "")):
        rigid_visual_alpha = 0.35

    def _rigid_visual_rgba(pid: int) -> Tuple[float, float, float, float]:
        rgba = list(color_from_part_id(pid))
        rgba[3] = float(rigid_visual_alpha)
        return tuple(rgba)

    group_anchor_world: Dict[str, np.ndarray] = {"0": np.zeros(3, dtype=np.float64)}
    # 组0的 carrier = l_world（与 urdf_gen.py 一致）
    group_carrier_link: Dict[str, str] = {"0": "l_world"}

    # ── 为底座零件生成 URDF link + 固定关节 ──
    # 与 urdf_gen.py 一致：link 命名为 l_{pid}，用 fixed joint 连接到上一个底座零件（链式）
    # 第一个底座零件通过 fixed joint 连到 l_world
    base_all_ids = [pid for pid in base_labels if pid in part_meshes]
    base_rigid_ids = [pid for pid in base_all_ids if _part_goes_to_rigid_skeleton(part_phys[pid])]
    for i, pid in enumerate(base_rigid_ids):
        p = part_phys[pid]
        mesh = part_meshes[pid]
        link_name = f"l_{pid}"
        visual_mesh_src_path = out_dir / "parts" / f"part_{pid:03d}.obj"
        visual_mesh, visual_mesh_meta = build_rigid_visual_proxy(
            mesh,
            enable_double_sided_shell=bool(rigid_visual_double_sided_shell),
            prefer_solid_visual=bool(str(simulator_mode).strip().lower() == "rigid"),
            solid_proxy_pitch=float(voxel_pitch),
        )
        visual_mesh_path = out_dir / "rigid_visuals" / f"part_{pid:03d}_visual.obj"
        visual_mesh.export(visual_mesh_path)

        collision_mesh, collision_fill_meta = build_rigid_collision_proxy(mesh, voxel_pitch)

        collision_mesh_path = out_dir / "rigid" / f"part_{pid:03d}_collision.obj"
        collision_mesh.export(collision_mesh_path)

        mesh_rel = os.path.relpath(visual_mesh_path, out_dir / "rigid")
        collision_mesh_rel = os.path.relpath(collision_mesh_path, out_dir / "rigid")

        link = ET.SubElement(robot, "link", name=link_name)
        mass = _estimate_part_mass(mesh, p, fallback_density_kgm3)
        add_inertial(link, mass, inertia_from_bbox(np.asarray(mesh.extents), mass), xyz=_mesh_inertial_origin_xyz(mesh))
        add_mesh_visual_collision(
            link, mesh_rel, _rigid_visual_rgba(pid),
            collision_mesh_relpath=collision_mesh_rel,
            collision_friction=p.friction,
            collision_restitution=p.restitution,
            use_collision_for_visual=False,
        )

        if i == 0:
            parent_for_joint = "l_world"
        else:
            parent_for_joint = f"l_{base_rigid_ids[i - 1]}"
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
                "visual_mesh_source_path": str(visual_mesh_src_path),
                "visual_mesh_meta": visual_mesh_meta,
                "collision_mesh_path": str(collision_mesh_path),
                "collision_voxel_fill": collision_fill_meta,
                "mesh_frame": "object_frame",
                "color_rgba": list(_rigid_visual_rgba(pid)),
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

    # base rigid part 的第一个 link 作为 skeleton 根；若没有 rigid base，则退回 l_world
    if base_rigid_ids:
        group_carrier_link["0"] = f"l_{base_rigid_ids[0]}"
    _prebuilt_links: set = set()
    unresolved = {g.group_id: g for g in movable_groups}
    progress = True
    while unresolved and progress:
        progress = False
        for gid in list(unresolved.keys()):
            g = unresolved[gid]
            if g.parent_group not in group_carrier_link:
                continue

            parent_link_name = group_carrier_link[g.parent_group]
            carrier_name = f"abstract_{g.group_id}"

            abs_link = ET.SubElement(robot, "link", name=carrier_name)
            add_inertial(abs_link, 0.01, inertia_from_bbox(np.asarray([0.01, 0.01, 0.01]), 0.01))

            p_raw = g.params
            jt = g.joint_type
            joint_name = f"joint_group_{g.group_id}"
            group_runtime = _group_runtime_joint_params(g.child_labels, part_phys)

            rigid_children = [
                pid for pid in g.child_labels
                if pid in part_meshes and _part_goes_to_rigid_skeleton(part_phys[pid])
            ]
            first_rigid_pid = rigid_children[0] if rigid_children else None

            anchor = p_raw[3:6] if len(p_raw) >= 6 else [0.0, 0.0, 0.0]

            if jt == "A":
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "A", p_raw, [0.0, 0.0, 0.0],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])
            elif jt == "B":
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "B", p_raw, [0.0, 0.0, 0.0],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])
            elif jt in ["C", "D", "CB"]:
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            jt, p_raw, [float(x) for x in anchor],
                            dynamics_damping=group_runtime["joint_damping"],
                            dynamics_friction=group_runtime["joint_frictionloss"])
            else:
                build_joint(robot, parent_link_name, carrier_name, joint_name,
                            "E", p_raw, [0.0, 0.0, 0.0])

            if first_rigid_pid is not None:
                child_link_name = f"l_{first_rigid_pid}"
                p0 = part_phys[first_rigid_pid]
                mesh0 = part_meshes[first_rigid_pid]
                visual_mesh_src_path0 = out_dir / "parts" / f"part_{first_rigid_pid:03d}.obj"
                visual_mesh0, visual_mesh_meta0 = build_rigid_visual_proxy(
                    mesh0,
                    enable_double_sided_shell=bool(rigid_visual_double_sided_shell),
                    prefer_solid_visual=bool(str(simulator_mode).strip().lower() == "rigid"),
                    solid_proxy_pitch=float(voxel_pitch),
                )
                visual_mesh_path0 = out_dir / "rigid_visuals" / f"group_{g.group_id}_part_{first_rigid_pid:03d}_visual.obj"
                visual_mesh0.export(visual_mesh_path0)
                coll_mesh0, coll_fill0 = build_rigid_collision_proxy(mesh0, voxel_pitch)
                coll_path0 = out_dir / "rigid" / f"group_{g.group_id}_part_{first_rigid_pid:03d}_collision.obj"
                coll_mesh0.export(coll_path0)

                mesh_rel0 = os.path.relpath(visual_mesh_path0, out_dir / "rigid")
                coll_rel0 = os.path.relpath(coll_path0, out_dir / "rigid")

                link0 = ET.SubElement(robot, "link", name=child_link_name)
                mass0 = _estimate_part_mass(mesh0, p0, fallback_density_kgm3)
                add_inertial(link0, mass0,
                             inertia_from_bbox(np.asarray(mesh0.extents), mass0),
                             xyz=_mesh_inertial_origin_xyz(mesh0))
                add_mesh_visual_collision(
                    link0, mesh_rel0, _rigid_visual_rgba(first_rigid_pid),
                    collision_mesh_relpath=coll_rel0,
                    collision_friction=p0.friction,
                    collision_restitution=p0.restitution,
                    use_collision_for_visual=False,
                )

                if jt in ["C", "D", "CB"]:
                    fix_xyz = " ".join(str(-float(x)) for x in anchor)
                else:
                    fix_xyz = "0 0 0"

                _add_fixed_joint(
                    robot,
                    f"fix_{carrier_name}_{child_link_name}",
                    carrier_name,
                    child_link_name,
                    xyz=fix_xyz,
                )

                rigid_part_links.append(
                    {
                        "part_id": first_rigid_pid,
                        "link_name": child_link_name,
                        "parent_link": carrier_name,
                        "group_id": g.group_id,
                        "mesh_path": str(visual_mesh_path0),
                        "visual_mesh_source_path": str(visual_mesh_src_path0),
                        "visual_mesh_meta": visual_mesh_meta0,
                        "collision_mesh_path": str(coll_path0),
                        "collision_voxel_fill": coll_fill0,
                        "mesh_frame": "object_frame",
                        "color_rgba": list(_rigid_visual_rgba(first_rigid_pid)),
                        "mass_kg": float(mass0),
                        "density_kgm3": p0.density_kgm3,
                        "youngs_pa": p0.youngs_pa,
                        "poisson": p0.poisson,
                        "friction": p0.friction,
                        "restitution": p0.restitution,
                        "damping": p0.damping,
                        "solver_family": p0.solver_family,
                        "simulator_material": p0.simulator_material,
                        "material_ctor": p0.material_ctor,
                    }
                )
                _prebuilt_links.add(first_rigid_pid)

            group_anchor_world[g.group_id] = np.asarray(anchor, dtype=np.float64)
            group_carrier_link[g.group_id] = carrier_name

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
    covered_labels = set(base_rigid_ids)
    for g in movable_groups:
        all_children = [pid for pid in g.child_labels if pid in part_meshes]
        rigid_children = [pid for pid in all_children if _part_goes_to_rigid_skeleton(part_phys[pid])]
        soft_children  = [pid for pid in all_children if not _part_goes_to_rigid_skeleton(part_phys[pid])]

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

            visual_mesh_src_path = out_dir / "parts" / f"part_{pid:03d}.obj"
            visual_mesh, visual_mesh_meta = build_rigid_visual_proxy(
                part_meshes[pid],
                enable_double_sided_shell=bool(rigid_visual_double_sided_shell),
                prefer_solid_visual=bool(str(simulator_mode).strip().lower() == "rigid"),
                solid_proxy_pitch=float(voxel_pitch),
            )
            visual_mesh_path = out_dir / "rigid_visuals" / f"group_{g.group_id}_part_{pid:03d}_visual.obj"
            visual_mesh.export(visual_mesh_path)
            collision_mesh, collision_fill_meta = build_rigid_collision_proxy(part_meshes[pid], voxel_pitch)
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
                link, mesh_rel, _rigid_visual_rgba(pid),
                collision_mesh_relpath=collision_mesh_rel,
                collision_friction=p.friction,
                collision_restitution=p.restitution,
                use_collision_for_visual=False,
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
                    "visual_mesh_source_path": str(visual_mesh_src_path),
                    "visual_mesh_meta": visual_mesh_meta,
                    "collision_mesh_path": str(collision_mesh_path),
                    "collision_voxel_fill": collision_fill_meta,
                    "mesh_frame": "object_frame",
                    "color_rgba": list(_rigid_visual_rgba(pid)),
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
            p = part_phys[pid]
            covered_labels.add(pid)
            mesh = part_meshes[pid]
            pts, fill_meta = voxel_fill_mesh(
                mesh,
                voxel_pitch,
                prefer_closed_proxy=bool(str(p.material_ctor) != "gs.materials.PBD.Cloth"),
            )
            soft_mesh_path = out_dir / "soft" / f"soft_group_{g.group_id}_part_{pid:03d}.obj"
            pts_path = out_dir / "soft" / f"soft_group_{g.group_id}_part_{pid:03d}_particles.npy"
            mesh.export(soft_mesh_path)
            np.save(pts_path, pts)
            soft_parts.append(
                {
                    "group_id": g.group_id,
                    "part_id": pid,
                    "child_labels": [pid],
                    "mesh_path": str(soft_mesh_path),
                    "particles_path": str(pts_path),
                    "voxel_fill": fill_meta,
                    "solver_family": p.solver_family,
                    "material_model": p.simulator_material,
                    "material_ctor": p.material_ctor,
                    "density_kgm3": p.density_kgm3,
                    "youngs_pa": p.youngs_pa,
                    "poisson": p.poisson,
                    "friction": p.friction,
                    "restitution": p.restitution,
                    "damping": p.damping,
                    "joint_type": g.joint_type,
                    "joint_params": [float(x) for x in g.params],
                    "json_exact_parameters": p.json_exact_parameters,
                }
            )

    # ── 未被任何组覆盖的孤立零件：固定到 l_world ──
    for pid in sorted(all_labels):
        if pid in covered_labels:
            continue
        p = part_phys[pid]
        mesh = part_meshes[pid]
        link_name = f"l_{pid}"
        if _part_goes_to_rigid_skeleton(p):
            visual_mesh_src_path = out_dir / "parts" / f"part_{pid:03d}.obj"
            visual_mesh, visual_mesh_meta = build_rigid_visual_proxy(
                mesh,
                enable_double_sided_shell=bool(rigid_visual_double_sided_shell),
                prefer_solid_visual=bool(str(simulator_mode).strip().lower() == "rigid"),
                solid_proxy_pitch=float(voxel_pitch),
            )
            visual_mesh_path = out_dir / "rigid_visuals" / f"standalone_part_{pid:03d}_visual.obj"
            visual_mesh.export(visual_mesh_path)
            collision_mesh, collision_fill_meta = build_rigid_collision_proxy(mesh, voxel_pitch)
            collision_mesh_path = out_dir / "rigid" / f"standalone_part_{pid:03d}_collision.obj"
            collision_mesh.export(collision_mesh_path)
            mesh_rel = os.path.relpath(visual_mesh_path, out_dir / "rigid")
            collision_mesh_rel = os.path.relpath(collision_mesh_path, out_dir / "rigid")
            link = ET.SubElement(robot, "link", name=link_name)
            mass = _estimate_part_mass(mesh, p, fallback_density_kgm3)
            add_inertial(link, mass, inertia_from_bbox(np.asarray(mesh.extents), mass),
                         xyz=_mesh_inertial_origin_xyz(mesh))
            add_mesh_visual_collision(
                link, mesh_rel, _rigid_visual_rgba(pid),
                collision_mesh_relpath=collision_mesh_rel,
                collision_friction=p.friction,
                collision_restitution=p.restitution,
                use_collision_for_visual=False,
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
                    "visual_mesh_source_path": str(visual_mesh_src_path),
                    "visual_mesh_meta": visual_mesh_meta,
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
            pts, fill_meta = voxel_fill_mesh(
                mesh,
                voxel_pitch,
                prefer_closed_proxy=bool(str(p.material_ctor) != "gs.materials.PBD.Cloth"),
            )
            soft_mesh_path = out_dir / "soft" / f"soft_part_{pid:03d}.obj"
            pts_path = out_dir / "soft" / f"soft_part_{pid:03d}_particles.npy"
            mesh.export(soft_mesh_path)
            np.save(pts_path, pts)
            soft_parts.append(
                {
                    "group_id": None,
                    "part_id": pid,
                    "child_labels": [pid],
                    "mesh_path": str(soft_mesh_path),
                    "particles_path": str(pts_path),
                    "voxel_fill": fill_meta,
                    "solver_family": p.solver_family,
                    "material_model": p.simulator_material,
                    "material_ctor": p.material_ctor,
                    "density_kgm3": p.density_kgm3,
                    "youngs_pa": p.youngs_pa,
                    "poisson": p.poisson,
                    "friction": p.friction,
                    "restitution": p.restitution,
                    "damping": p.damping,
                    "joint_type": None,
                    "joint_params": None,
                    "json_exact_parameters": p.json_exact_parameters,
                }
            )

    urdf_path = out_dir / "rigid" / f"{object_id}.urdf"
    indent_xml(robot)
    ET.ElementTree(robot).write(urdf_path, encoding="utf-8", xml_declaration=True)

    metadata = {
        "object_id": object_id,
        "object_name": meta.get("object_name", object_id),
        "category": meta.get("category", "Unknown"),
        "output_dir": str(out_dir),
        "solver_family_override": solver_family_override,
        "simulator_mode": str(simulator_mode),
        "all_parts_youngs_threshold_gpa": all_parts_youngs_threshold_gpa,
        "classification_policy_version": CLASSIFICATION_POLICY_VERSION,
        "inertial_origin_policy_version": INERTIAL_ORIGIN_POLICY_VERSION,
        "object_solver_policy": object_solver_policy,
        "json_path_used": str(json_path),
        "dimension_raw": meta.get("dimension", None),
        "dimension_m": dim_m.tolist() if dim_m is not None else None,
        "object_scale": object_scale,
        "liquid_container_scale_boost": liquid_container_scale_boost,
        "rigid_visual_double_sided_shell": bool(rigid_visual_double_sided_shell),
        "raw_mesh_extents": raw_extents.tolist(),
        "object_bbox_min": bbox_min.tolist(),
        "object_bbox_max": bbox_max.tolist(),
        "grounding_offset_z": grounding_offset_z,
        "base_part_labels": base_labels,
        "group_info": meta.get("group_info", {}),
        "rigid_group_carriers": rigid_group_carriers,
        "rigid_part_links": rigid_part_links,
        "soft_parts": soft_parts,
        "floating_parts": floating_parts,
        "parts_physical": {str(pid): asdict(p) for pid, p in part_phys.items()},
        "notes": {
            "coordinate_conversion": "source meshes and joint params are converted from Y-up to Z-up",
            "rigid_export": "rigid parts are exported per original part, never averaged or merged into a single colored group",
            "inertial_origin": "non-watertight / non-volume rigid meshes use the part AABB centroid as URDF inertial origin instead of trimesh center_mass",
            "runtime_exactness": {
                "density": "used per rigid part to compute per-link mass/inertia and applied again to Genesis links after URDF load",
                "youngs_poisson": "used directly for soft-part Genesis materials; rigid bodies preserve them in metadata only because rigid contact models do not consume them directly",
                "friction_restitution_damping": "friction and restitution are written per rigid collision; friction and damping are also applied at runtime to rigid links / joints when Genesis exposes the corresponding setters",
            },
        },
    }
    with open(out_dir / "meta" / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
# 选rigid还是mpm，其他的都一样，只是选rigid的时候soft_parts为[],
    return PreparedObject(
        object_id=object_id,
        object_name=str(meta.get("object_name", object_id)),
        category=str(meta.get("category", "Unknown")),
        dimension_m=dim_m.tolist() if dim_m is not None else None,
        object_scale=object_scale,
        base_part_labels=base_labels,
        rigid_group_carriers=rigid_group_carriers,
        rigid_part_links=rigid_part_links,
        soft_parts=soft_parts,
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
# Runtime part-spec helpers ----------------------------------------------------

def _mesh_bounds_info(mesh_path: Path, scale: float = 1.0) -> Optional[Dict[str, Any]]:
    try:
        obj = trimesh.load(mesh_path, process=False)
        if isinstance(obj, trimesh.Scene):
            meshes = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
            if not meshes:
                return None
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(obj, trimesh.Trimesh):
            mesh = obj
        else:
            return None

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
    material_ctor = str((part_meta or {}).get("material_ctor") or "")
    is_soft_part = material_ctor != "gs.materials.Rigid"

    # Soft parts are safer when we reuse the dedicated soft-export mesh first.
    # Those meshes are the ones already prepared for voxel filling / particle sampling.
    if is_soft_part:
        if legacy_soft is not None:
            legacy_mesh = legacy_soft.get("mesh_path")
            if legacy_mesh:
                candidates.append(Path(str(legacy_mesh)))
            group_id = legacy_soft.get("group_id")
            if group_id is not None:
                candidates.append(obj_dir / "soft" / f"soft_group_{group_id}_part_{pid:03d}.obj")
        candidates.append(obj_dir / "soft" / f"soft_part_{pid:03d}.obj")

    # 优先使用当前导出目录下的标准 part mesh，确保所有 part 共用同一 object frame
    candidates.append(obj_dir / "parts" / f"part_{pid:03d}.obj")

    # 其次再看 metadata 里显式记录的 mesh_path
    if part_meta is not None and part_meta.get("mesh_path"):
        candidates.append(Path(str(part_meta["mesh_path"])))

    # 旧版 soft 路径作为 rigid/unknown 情况下的保底 fallback
    if legacy_soft is not None and not is_soft_part:
        legacy_mesh = legacy_soft.get("mesh_path")
        if legacy_mesh:
            candidates.append(Path(str(legacy_mesh)))
        group_id = legacy_soft.get("group_id")
        if group_id is not None:
            candidates.append(obj_dir / "soft" / f"soft_group_{group_id}_part_{pid:03d}.obj")
        candidates.append(obj_dir / "soft" / f"soft_part_{pid:03d}.obj")

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
    raw_model = str(raw_model or "").strip().lower()
    raw_model = raw_model.replace("gs.materials.", "")
    raw_model = raw_model.replace("mpm.", "mpm_")
    raw_model = raw_model.replace("sph.", "sph_")
    raw_model = raw_model.replace("pbd.", "pbd_")

    alias_map = {
        "rigid": "gs.materials.Rigid",
        "hard": "gs.materials.Rigid",
        "rigid_body": "gs.materials.Rigid",
        "cloth": "gs.materials.PBD.Cloth",
        "pbd_cloth": "gs.materials.PBD.Cloth",
        "liquid": "gs.materials.SPH.Liquid",
        "sph_liquid": "gs.materials.SPH.Liquid",
        "elastic": "gs.materials.MPM.Elastic",
        "mpm_elastic": "gs.materials.MPM.Elastic",
        "elastoplastic": "gs.materials.MPM.ElastoPlastic",
        "elasto_plastic": "gs.materials.MPM.ElastoPlastic",
        "mpm_elastoplastic": "gs.materials.MPM.ElastoPlastic",
        "sand": "gs.materials.MPM.Sand",
        "mpm_sand": "gs.materials.MPM.Sand",
        "snow": "gs.materials.MPM.Snow",
        "mpm_snow": "gs.materials.MPM.Snow",
        "mpm_liquid": "gs.materials.MPM.Liquid",
    }
    if raw_model in alias_map:
        return alias_map[raw_model]

    if "rigid" in raw_model or "hard" in raw_model:
        return "gs.materials.Rigid"
    if "cloth" in raw_model:
        return "gs.materials.PBD.Cloth"
    if "liquid" in raw_model and "sph" in raw_model:
        return "gs.materials.SPH.Liquid"
    if "liquid" in raw_model and "mpm" in raw_model:
        return "gs.materials.MPM.Liquid"
    if "sand" in raw_model:
        return "gs.materials.MPM.Sand"
    if "snow" in raw_model:
        return "gs.materials.MPM.Snow"
    if "plastic" in raw_model:
        return "gs.materials.MPM.ElastoPlastic"
    if "elastic" in raw_model:
        return "gs.materials.MPM.Elastic"
    return "gs.materials.MPM.Elastic"


def _build_part_spec_from_sources(
    obj_dir: Path,
    pid: int,
    part_meta: dict,
    legacy_soft: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    material_ctor = str(part_meta.get("material_ctor") or "")
    if not material_ctor and legacy_soft is not None:
        material_ctor = str(legacy_soft.get("material_ctor") or "")
    if not material_ctor:
        material_ctor = _legacy_material_ctor_from_model(
            (legacy_soft or {}).get("material_model", part_meta.get("simulator_material", "elastic"))
        )

    mesh_path = _resolve_runtime_part_mesh_path(
        obj_dir=obj_dir,
        pid=pid,
        part_meta=part_meta,
        legacy_soft=legacy_soft,
    )
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
    material_model = str(part_meta.get("simulator_material") or ((legacy_soft or {}).get("material_model") or "elastic")).lower()

    # movement_description = str(part_meta.get("movement_description", ""))
    # assembly_role = str(part_meta.get("assembly_role") or classify_assembly_role(material_ctor, movement_description))
    movement_description = str(part_meta.get("movement_description", ""))
    assembly_role = str(part_meta.get("assembly_role") or classify_assembly_role(material_ctor, movement_description))

    material_ctor_runtime = _runtime_material_ctor_from_spec(
        part_meta=part_meta,
        material_ctor=material_ctor,
        assembly_role=assembly_role,
    )
    solver_family_runtime = _runtime_solver_family_from_ctor(material_ctor_runtime)
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
        "youngs": None if youngs is None else float(youngs),
        "poisson": None if poisson is None else float(poisson),
        "friction": friction,
        "restitution": restitution,
        "damping": damping,

        # 原始分类（来自 material）
        "solver_family": solver_family,
        "material_model": material_model,
        "material_ctor": material_ctor,

        # 运行时分类（允许 anchored_soft + cloth -> mpm_elastic）
        "solver_family_runtime": solver_family_runtime,
        "material_ctor_runtime": material_ctor_runtime,

        "assembly_role": assembly_role,
        "movement_description": movement_description,
        "is_rigid": (material_ctor == "gs.materials.Rigid"),
        "color": list(color_from_part_id(pid)),
    }
    
    bounds_info = _mesh_bounds_info(mesh_path, scale=1.0)
    if bounds_info is not None:
        spec.update(bounds_info)
    return spec


def _collect_part_specs(obj_dir: Path, metadata: dict):
    specs: List[Dict[str, Any]] = []
    parts_physical = metadata.get("parts_physical", {}) if isinstance(metadata, dict) else {}
    legacy_soft_index = {int(x.get("part_id")): x for x in metadata.get("soft_parts", []) if isinstance(x, dict) and x.get("part_id") is not None}

    for pid_str, part_meta in sorted(parts_physical.items(), key=lambda kv: int(kv[0])):
        pid = int(pid_str)
        if not isinstance(part_meta, dict):
            continue
        legacy_soft = legacy_soft_index.get(pid)
        spec = _build_part_spec_from_sources(
            obj_dir=obj_dir,
            pid=pid,
            part_meta=part_meta,
            legacy_soft=legacy_soft,
        )
        if spec is not None:
            specs.append(spec)

    return specs

def _spec_uses_mpm(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor_runtime", spec.get("material_ctor", ""))).startswith("gs.materials.MPM.")

def _spec_uses_sph(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor_runtime", spec.get("material_ctor", ""))) == "gs.materials.SPH.Liquid"

def _is_liquid_spec(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor_runtime", spec.get("material_ctor", ""))) in (
        "gs.materials.SPH.Liquid",
        "gs.materials.MPM.Liquid",
    )


def _prepared_object_has_liquid(prepared: PreparedObject) -> bool:
    candidates: List[Dict[str, Any]] = []
    candidates.extend(list(prepared.soft_parts or []))
    candidates.extend(list(prepared.floating_parts or []))
    for spec in candidates:
        if not isinstance(spec, dict):
            continue
        material_ctor = str(spec.get("material_ctor", "") or "")
        material_model = str(spec.get("material_model", spec.get("simulator_material", "")) or "")
        solver_family = str(spec.get("solver_family", "") or "")
        if material_ctor in ("gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid"):
            return True
        if material_model == "liquid":
            return True
        if solver_family in ("sph_liquid", "mpm_liquid"):
            return True
    return False


def _liquid_prefers_free_surface(spec: Dict[str, Any]) -> bool:
    return bool(spec.get("prefer_free_surface", False))

def _spec_uses_pbd(spec: Dict[str, Any]) -> bool:
    return str(spec.get("material_ctor_runtime", spec.get("material_ctor", ""))) == "gs.materials.PBD.Cloth"


def _suggest_sph_particle_size(part_specs: List[Dict[str, Any]]) -> float:
    liquid_xy_extents: List[float] = []
    free_surface_liquid = False
    for spec in part_specs:
        if not _spec_uses_sph(spec):
            continue
        if _liquid_prefers_free_surface(spec):
            free_surface_liquid = True
        size = spec.get("runtime_bounds_size", spec.get("bounds_size", None))
        if size is None:
            continue
        try:
            size_arr = np.maximum(np.asarray(size, dtype=np.float64), 1e-6)
        except Exception:
            continue
        liquid_xy_extents.append(float(np.min(size_arr[:2])))

    if not liquid_xy_extents:
        return 0.01

    min_xy_extent = float(min(liquid_xy_extents))
    # Small container liquids need denser sampling; otherwise the preview can end up with
    # only a few hundred particles, which visually "vanish" after the first splash.
    if free_surface_liquid:
        if min_xy_extent < 0.18:
            return float(np.clip(0.028 * min_xy_extent, 0.0038, 0.0065))
        return float(np.clip(0.04 * min_xy_extent, 0.005, 0.008))
    if min_xy_extent < 0.18:
        return float(np.clip(0.035 * min_xy_extent, 0.0045, 0.008))
    return float(np.clip(0.05 * min_xy_extent, 0.006, 0.01))


def _clip_runtime_mpm_E(youngs: float) -> float:
    # Genesis MPM becomes unstable when feeding raw GPa-scale stiffness directly.
    return float(np.clip(float(youngs), 3e4, 6e5))


def _stabilize_runtime_mpm_params(ctor: str, youngs: float, poisson: float) -> Tuple[float, float]:
    youngs_val = float(youngs)
    poisson_val = float(poisson)

    if ctor == "gs.materials.MPM.Elastic":
        # Soft foams / cushions from PhysXNet often carry Young's modulus values that are
        # numerically too stiff for the preview MPM setup, which makes particles explode
        # or leave the visible domain in the first few frames.
        youngs_val = float(np.clip(youngs_val, 4e4, 4e5))
        poisson_val = float(np.clip(poisson_val, 0.05, 0.24))
    elif ctor == "gs.materials.MPM.ElastoPlastic":
        youngs_val = float(np.clip(youngs_val, 4e4, 3e5))
        poisson_val = float(np.clip(poisson_val, 0.05, 0.26))
    else:
        youngs_val = _clip_runtime_mpm_E(youngs_val)
        poisson_val = float(np.clip(poisson_val, 0.05, 0.35))

    return youngs_val, poisson_val


def _is_pillow_spec(spec: Dict[str, Any]) -> bool:
    role = str(spec.get("assembly_role", "free_soft"))
    part_name = str(spec.get("part_name", "")).lower()
    return role == "free_soft" and "pillow" in part_name


def _is_seat_surface_spec(spec: Dict[str, Any]) -> bool:
    role = str(spec.get("assembly_role", "free_soft"))
    part_name = str(spec.get("part_name", "")).lower()
    return role == "anchored_soft" and "seat surface" in part_name


def _is_free_cloth_like_spec(spec: Dict[str, Any]) -> bool:
    role = str(spec.get("assembly_role", "free_soft"))
    ctor = str(spec.get("material_ctor_runtime", spec.get("material_ctor", "")))
    return role == "free_soft" and ctor == "gs.materials.PBD.Cloth"


def _make_part_material(gs, spec, default_friction: float = 0.55):
    # ctor = str(spec.get("material_ctor", "gs.materials.MPM.Elastic"))
    ctor = str(spec.get("material_ctor_runtime", spec.get("material_ctor", "gs.materials.MPM.Elastic")))
    density_raw = spec.get("density", None)
    youngs_raw = spec.get("youngs", None)
    poisson_raw = spec.get("poisson", None)
    strict_dataset_params = bool(spec.get("strict_dataset_params", False))
    if strict_dataset_params and (density_raw is None or youngs_raw is None or poisson_raw is None):
        raise ValueError(
            f"strict_dataset_params requires dataset-provided rho/E/nu, got "
            f"rho={density_raw} E={youngs_raw} nu={poisson_raw} part={spec.get('part_name', 'unknown')}"
        )
    density = float(density_raw if density_raw is not None else 800.0)
    youngs = float(youngs_raw if youngs_raw is not None else 1e7)
    poisson = float(poisson_raw if poisson_raw is not None else 0.3)
    friction = spec.get("friction", None)
    restitution = spec.get("restitution", None)
    damping = spec.get("damping", None)
    role = str(spec.get("assembly_role", "free_soft"))
    pid = int(spec.get("pid", -1))
    is_pillow = _is_pillow_spec(spec)

    runtime_bounds_min = spec.get("runtime_bounds_min", spec.get("bounds_min"))
    runtime_bounds_max = spec.get("runtime_bounds_max", spec.get("bounds_max"))
    runtime_extent = None
    if runtime_bounds_min is not None and runtime_bounds_max is not None:
        try:
            runtime_extent = np.maximum(
                np.asarray(runtime_bounds_max, dtype=np.float64) - np.asarray(runtime_bounds_min, dtype=np.float64),
                1e-6,
            )
        except Exception:
            runtime_extent = None

    anchored_sampler = "pbs-8"
    if runtime_extent is not None:
        min_extent = float(np.min(runtime_extent))
        max_extent = float(np.max(runtime_extent))
        volume_est = float(np.prod(runtime_extent))
        # Very thin/small anchored pieces collapse into filament-like renders with pbs-8.
        if min_extent < 0.06 or max_extent < 0.18 or volume_est < 8e-4:
            anchored_sampler = "pbs"

    if ctor == "gs.materials.Rigid":
        return _make_genesis_rigid_material(
            gs,
            rho=density,
            friction=float(friction if friction is not None else default_friction),
            restitution=restitution,
        )

    if ctor == "gs.materials.SPH.Liquid":
        # Respect per-part liquid properties from the dataset JSON.
        # The PhysXNet annotations do not provide SPH-only parameters such as stiffness/mu/gamma,
        # so the runtime preview injects a small set of water-like defaults. For free-surface
        # liquids we prefer the PBS sampler because it initializes a real volume more reliably
        # than a surface-like regular sampling pattern on scanned meshes.
        liquid_sampler = spec.get("liquid_sampler", None)
        if liquid_sampler is None:
            liquid_sampler = "pbs" if _liquid_prefers_free_surface(spec) else "regular"
        kwargs = {"rho": density, "sampler": str(liquid_sampler)}
        liquid_stiffness = spec.get("liquid_stiffness", None)
        if liquid_stiffness is not None:
            kwargs["stiffness"] = float(liquid_stiffness)
        liquid_exponent = spec.get("liquid_exponent", None)
        if liquid_exponent is not None:
            kwargs["exponent"] = float(liquid_exponent)
        liquid_viscosity = spec.get("liquid_viscosity", None)
        if liquid_viscosity is not None:
            kwargs["mu"] = float(liquid_viscosity)
        liquid_surface_tension = spec.get("liquid_surface_tension", None)
        if liquid_surface_tension is not None:
            kwargs["gamma"] = float(liquid_surface_tension)
        print(f"{spec['part_name']} SPH.Liquid kwargs:", kwargs)
        return gs.materials.SPH.Liquid(**kwargs)

    if ctor == "gs.materials.PBD.Cloth":
        mat = _make_pbd_cloth_material_from_part(
            gs,
            density=density,
            friction=friction,
            youngs=youngs,
            damping=damping,
            free_cloth=bool(_is_free_cloth_like_spec(spec)),
        )
        # print(f"{spec['part_name']} PBD.Cloth kwargs: {mat.__dict__}")
        # exit()
        return mat

    youngs_runtime, poisson_runtime = _stabilize_runtime_mpm_params(ctor, youngs, poisson)
    common_kwargs = {"E": youngs_runtime, "nu": poisson_runtime, "rho": density, "sampler": "pbs"}
    if is_pillow and ctor in ("gs.materials.MPM.Elastic", "gs.materials.MPM.ElastoPlastic"):
        # Free pillows are the first parts to numerically explode in the sofa scenes.
        # Clamp them into a much softer preview regime by default.
        common_kwargs["E"] = min(common_kwargs["E"], 1.0e5)
        common_kwargs["nu"] = min(common_kwargs["nu"], 0.05)
    if role == "anchored_soft" and ctor in ("gs.materials.MPM.Elastic", "gs.materials.MPM.ElastoPlastic"):
        # Anchored cushions behave better in preview if they are softer and less volume-preserving.
        common_kwargs["E"] = min(common_kwargs["E"], 2.0e5)
        common_kwargs["nu"] = min(common_kwargs["nu"], 0.18)
        common_kwargs["sampler"] = anchored_sampler
    if ctor == "gs.materials.MPM.ElastoPlastic":
        common_kwargs["E"] = max(common_kwargs["E"], 5e4)
        if role == "anchored_soft":
            common_kwargs["E"] = min(common_kwargs["E"], 2.0e5)
            common_kwargs["nu"] = min(common_kwargs["nu"], 0.18)
            common_kwargs["sampler"] = anchored_sampler
        print(f"{spec['part_name']} MPM.ElastoPlastic kwargs:", common_kwargs)
        return gs.materials.MPM.ElastoPlastic(**common_kwargs)
    if ctor == "gs.materials.MPM.Sand":
        print(f"{spec['part_name']} MPM.Sand kwargs:", common_kwargs)
        return gs.materials.MPM.Sand(**common_kwargs)
    if ctor == "gs.materials.MPM.Snow":
        print(f"{spec['part_name']} MPM.Snow kwargs:", common_kwargs)
        return gs.materials.MPM.Snow(**common_kwargs)
    if ctor == "gs.materials.MPM.Liquid":
        kwargs = {"rho": density, "sampler": "pbs"}
        print(f"{spec['part_name']} MPM.Liquid kwargs:", kwargs)
        return gs.materials.MPM.Liquid(**kwargs)
    common_kwargs["E"] = max(common_kwargs["E"], 5e4)
    if role == "anchored_soft":
        common_kwargs["E"] = min(common_kwargs["E"], 2.0e5)
        common_kwargs["nu"] = min(common_kwargs["nu"], 0.18)
        common_kwargs["sampler"] = anchored_sampler

    debug_E_scale = spec.get("debug_E_scale", None)
    if debug_E_scale is not None:
        common_kwargs["E"] = float(max(1e3, common_kwargs["E"] * float(debug_E_scale)))
    debug_nu_override = spec.get("debug_nu_override", None)
    if debug_nu_override is not None:
        common_kwargs["nu"] = float(np.clip(float(debug_nu_override), 0.01, 0.49))
    debug_sampler_override = spec.get("debug_sampler_override", None)
    if debug_sampler_override:
        common_kwargs["sampler"] = str(debug_sampler_override)

    if debug_E_scale is not None or debug_nu_override is not None or debug_sampler_override:
        print(
            f"🧪 pid={pid} material override "
            f"E_scale={debug_E_scale} nu_override={debug_nu_override} sampler_override={debug_sampler_override}"
        )
    print(f"{spec['part_name']} MPM.Elastic kwargs:", common_kwargs)

    return gs.materials.MPM.Elastic(**common_kwargs)


# backward compatibility for existing helper call-sites
def _make_soft_material(gs, spec):
    return _make_part_material(gs, spec)


def _load_trimesh_single(obj: Any) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj.copy()

    loaded = trimesh.load(obj, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [g.copy() for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
        if not meshes:
            raise ValueError("empty_scene")
        return trimesh.util.concatenate(meshes)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded.copy()
    raise ValueError("unsupported_mesh")


def _voxel_index_set_from_mesh(
    mesh: trimesh.Trimesh,
    pitch: float,
    domain_min: np.ndarray,
    domain_shape: np.ndarray,
    fill_interior: bool = True,
) -> set[Tuple[int, int, int]]:
    vox = mesh.voxelized(pitch)
    if fill_interior:
        try:
            vox = vox.fill()
        except Exception:
            pass
    pts = np.asarray(vox.points, dtype=np.float64)
    if pts.size == 0:
        return set()
    idx = np.floor((pts - domain_min[None, :]) / float(pitch) + 1e-9).astype(np.int32)
    mask = np.all((idx >= 0) & (idx < domain_shape[None, :]), axis=1)
    return {tuple(map(int, row)) for row in idx[mask]}


def _fill_voxel_columns_to_height(
    voxels: set[Tuple[int, int, int]],
    domain_min: np.ndarray,
    pitch: float,
    target_top_z: float,
) -> Tuple[set[Tuple[int, int, int]], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "column_count": 0,
        "filled_voxel_count": 0,
        "mode": "column_solid_fill",
    }
    if not voxels:
        return set(), info

    columns: Dict[Tuple[int, int], List[int]] = {}
    for x, y, z in voxels:
        columns.setdefault((int(x), int(y)), []).append(int(z))

    filled: set[Tuple[int, int, int]] = set()
    target_top_z = float(target_top_z)
    base_z = float(domain_min[2])
    voxel_pitch = float(pitch)

    for (x, y), zs in columns.items():
        unique_zs = sorted(set(int(v) for v in zs))
        if not unique_zs:
            continue

        capped_top: Optional[int] = None
        for z in unique_zs:
            cell_top_z = base_z + voxel_pitch * (float(z) + 1.0)
            if cell_top_z <= target_top_z + 1e-9:
                capped_top = int(z)
            else:
                break
        if capped_top is None:
            continue

        column_bottom = int(min(unique_zs))
        for z in range(column_bottom, capped_top + 1):
            filled.add((int(x), int(y), int(z)))

    info["column_count"] = int(len(columns))
    info["filled_voxel_count"] = int(len(filled))
    return filled, info


def _build_closed_voxel_fill_proxy(
    mesh: trimesh.Trimesh,
    pitch: float,
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    mesh = sanitize_mesh(mesh)
    info: Dict[str, Any] = {
        "requested_pitch": float(pitch),
        "mesh_is_watertight": bool(getattr(mesh, "is_watertight", False)),
        "mesh_is_volume": bool(getattr(mesh, "is_volume", False)),
        "proxy_kind": "none",
    }
    if info["mesh_is_watertight"] and info["mesh_is_volume"]:
        info["proxy_kind"] = "original_mesh"
        return mesh.copy(), info

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    domain_min = bounds[0] - 2.5 * float(pitch)
    domain_max = bounds[1] + 2.5 * float(pitch)
    domain_shape = np.ceil((domain_max - domain_min) / float(pitch)).astype(np.int32) + 1
    if np.any(domain_shape <= 0):
        info["proxy_kind"] = "invalid_domain"
        return None, info

    voxels = _voxel_index_set_from_mesh(
        mesh=mesh,
        pitch=float(pitch),
        domain_min=domain_min,
        domain_shape=domain_shape,
        fill_interior=True,
    )
    info["surface_or_fill_voxel_count"] = int(len(voxels))
    if not voxels:
        info["proxy_kind"] = "empty_voxels"
        return None, info

    closed_mesh, closed_info = _voxel_set_to_closed_mesh(
        voxels=voxels,
        domain_min=domain_min,
        domain_shape=domain_shape,
        pitch=float(pitch),
    )
    info["proxy_kind"] = "voxel_closed_proxy"
    info["closed_mesh_info"] = closed_info
    if closed_mesh is None:
        return None, info
    return closed_mesh, info


def _voxel_set_to_closed_mesh(
    voxels: set[Tuple[int, int, int]],
    domain_min: np.ndarray,
    domain_shape: np.ndarray,
    pitch: float,
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "builder": "multibox",
        "watertight": False,
        "is_volume": False,
        "voxel_count": int(len(voxels)),
    }
    if not voxels:
        return None, info

    voxel_pitch = float(pitch)
    ordered = np.asarray(sorted(voxels), dtype=np.float64)
    centers = np.asarray(domain_min, dtype=np.float64)[None, :] + voxel_pitch * (ordered + 0.5)
    mesh = trimesh.voxel.ops.multibox(
        centers=centers,
        pitch=voxel_pitch,
        remove_internal_faces=True,
    )
    mesh = sanitize_mesh(mesh)

    info["watertight"] = bool(getattr(mesh, "is_watertight", False))
    info["is_volume"] = bool(getattr(mesh, "is_volume", False))
    if info["watertight"] and info["is_volume"]:
        return mesh, info

    dense = np.zeros(tuple(int(v) for v in np.asarray(domain_shape, dtype=np.int32).tolist()), dtype=bool)
    for x, y, z in voxels:
        if 0 <= int(x) < dense.shape[0] and 0 <= int(y) < dense.shape[1] and 0 <= int(z) < dense.shape[2]:
            dense[int(x), int(y), int(z)] = True
    if not np.any(dense):
        info["builder"] = "empty_dense_grid"
        return None, info

    try:
        repaired = trimesh.voxel.ops.matrix_to_marching_cubes(dense, pitch=voxel_pitch)
        repaired.apply_translation(np.asarray(domain_min, dtype=np.float64) + 0.5 * voxel_pitch)
        repaired = sanitize_mesh(repaired)
        info["builder"] = "marching_cubes"
        info["watertight"] = bool(getattr(repaired, "is_watertight", False))
        info["is_volume"] = bool(getattr(repaired, "is_volume", False))
        if len(repaired.vertices) == 0 or len(repaired.faces) == 0:
            return None, info
        return repaired, info
    except Exception as exc:
        info["builder"] = f"marching_cubes_failed:{type(exc).__name__}"
        return mesh, info


def _build_container_filled_liquid_mesh(
    liquid_mesh: trimesh.Trimesh,
    spec: Dict[str, Any],
    other_specs: List[Dict[str, Any]],
    pitch: float,
    prefer_free_surface: bool = False,
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "liquid_volume_fill_attempted": False,
        "liquid_volume_fill_applied": False,
    }

    liquid_bounds = np.asarray(liquid_mesh.bounds, dtype=np.float64)
    liquid_size = np.maximum(liquid_bounds[1] - liquid_bounds[0], 1e-6)
    min_xy = float(np.min(liquid_size[:2]))
    if min_xy <= 1e-6:
        info["liquid_volume_fill_reason"] = "liquid_xy_extent_too_small"
        return None, info

    rigid_candidates: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray]] = []
    liquid_center_xy = 0.5 * (liquid_bounds[0, :2] + liquid_bounds[1, :2])
    xy_pad = min(0.015, max(0.002, 0.08 * min_xy))
    vertical_pad = max(0.02, 3.0 * pitch)

    for other in other_specs:
        if int(other.get("pid", -1)) == int(spec.get("pid", -2)):
            continue
        if str(other.get("assembly_role", "free_soft")) != "rigid_skeleton":
            continue

        bounds_info = None
        collision_mesh_path = other.get("collision_mesh_path")
        if collision_mesh_path:
            bounds_info = _mesh_bounds_info(Path(str(collision_mesh_path)), scale=float(other.get("scale", 1.0)))
        if bounds_info is None:
            bounds_info = {
                "bounds_min": other.get("bounds_min", [0.0, 0.0, 0.0]),
                "bounds_max": other.get("bounds_max", [0.0, 0.0, 0.0]),
            }

        other_min = np.asarray(bounds_info["bounds_min"], dtype=np.float64)
        other_max = np.asarray(bounds_info["bounds_max"], dtype=np.float64)
        xy_overlap_min = np.maximum(liquid_bounds[0, :2] - xy_pad, other_min[:2])
        xy_overlap_max = np.minimum(liquid_bounds[1, :2] + xy_pad, other_max[:2])
        center_inside_xy = bool(np.all(liquid_center_xy >= other_min[:2] - xy_pad) and np.all(liquid_center_xy <= other_max[:2] + xy_pad))
        z_near = bool(other_min[2] <= liquid_bounds[1, 2] + vertical_pad and other_max[2] >= liquid_bounds[0, 2] - 0.12)
        if (np.all(xy_overlap_max > xy_overlap_min) or center_inside_xy) and z_near:
            rigid_candidates.append((other, other_min, other_max))

    info["liquid_volume_fill_attempted"] = True
    info["liquid_volume_fill_candidate_count"] = int(len(rigid_candidates))
    if not rigid_candidates:
        info["liquid_volume_fill_reason"] = "no_nearby_rigid_container"
        return None, info

    liquid_bottom_z = float(liquid_bounds[0, 2])
    support_tops = [
        float(other_max[2])
        for _, other_min, other_max in rigid_candidates
        if float(other_max[2]) <= liquid_bottom_z - max(0.5 * pitch, 0.003)
        and float(other_min[0]) - xy_pad <= float(liquid_center_xy[0]) <= float(other_max[0]) + xy_pad
        and float(other_min[1]) - xy_pad <= float(liquid_center_xy[1]) <= float(other_max[1]) + xy_pad
    ]
    if not support_tops:
        info["liquid_volume_fill_reason"] = "no_support_floor_below_liquid"
        return None, info

    floor_top_z = max(support_tops)
    liquid_top_z = float(liquid_bounds[1, 2])
    fill_height = liquid_top_z - floor_top_z
    if fill_height <= max(2.5 * pitch, 0.01):
        info["liquid_volume_fill_reason"] = "container_fill_height_too_small"
        return None, info
    surface_headroom = 0.0
    if prefer_free_surface:
        surface_headroom = max(1.5 * pitch, 0.08 * fill_height)
    target_top_z = max(floor_top_z + max(2.5 * pitch, 0.01), liquid_top_z - surface_headroom)

    domain_min = np.array(
        [
            float(liquid_bounds[0, 0]),
            float(liquid_bounds[0, 1]),
            float(floor_top_z + 0.5 * pitch),
        ],
        dtype=np.float64,
    )
    domain_max = np.array(
        [
            float(liquid_bounds[1, 0]),
            float(liquid_bounds[1, 1]),
            float(target_top_z),
        ],
        dtype=np.float64,
    )
    domain_shape = np.ceil((domain_max - domain_min) / float(pitch)).astype(np.int32) + 1
    if np.any(domain_shape <= 1):
        info["liquid_volume_fill_reason"] = "invalid_fill_domain"
        return None, info

    occupied: set[Tuple[int, int, int]] = set()
    for other, _, _ in rigid_candidates:
        collision_mesh_path = other.get("collision_mesh_path", None)
        if not collision_mesh_path:
            continue
        try:
            collision_mesh = _load_trimesh_single(Path(str(collision_mesh_path)))
        except Exception:
            continue
        occupied.update(
            _voxel_index_set_from_mesh(
                mesh=collision_mesh,
                pitch=float(pitch),
                domain_min=domain_min,
                domain_shape=domain_shape,
                fill_interior=True,
            )
        )

    seed_voxels = _voxel_index_set_from_mesh(
        mesh=liquid_mesh,
        pitch=float(pitch),
        domain_min=domain_min,
        domain_shape=domain_shape,
        fill_interior=True,
    )
    seed_voxels = {idx for idx in seed_voxels if idx not in occupied}
    if not seed_voxels:
        info["liquid_volume_fill_reason"] = "no_liquid_seed_voxels"
        return None, info

    filled: set[Tuple[int, int, int]] = set(seed_voxels)
    queue: deque[Tuple[int, int, int]] = deque(seed_voxels)
    while queue:
        x, y, z = queue.popleft()
        for nb in (
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z - 1),
        ):
            if nb in filled or nb in occupied:
                continue
            if not (0 <= nb[0] < int(domain_shape[0]) and 0 <= nb[1] < int(domain_shape[1]) and 0 <= nb[2] < int(domain_shape[2])):
                continue
            filled.add(nb)
            queue.append(nb)

    if len(filled) <= len(seed_voxels):
        info["liquid_volume_fill_reason"] = "no_downward_fill_growth"
        return None, info

    column_filled, column_info = _fill_voxel_columns_to_height(
        voxels=filled,
        domain_min=domain_min,
        pitch=float(pitch),
        target_top_z=float(target_top_z),
    )
    if column_filled:
        filled = column_filled

    filled_mesh, mesh_info = _voxel_set_to_closed_mesh(
        voxels=filled,
        domain_min=domain_min,
        domain_shape=domain_shape,
        pitch=float(pitch),
    )
    if filled_mesh is None:
        info["liquid_volume_fill_reason"] = "empty_filled_mesh"
        info["liquid_volume_fill_column_info"] = column_info
        info["liquid_volume_fill_mesh_info"] = mesh_info
        return None, info
    if len(filled_mesh.vertices) == 0 or len(filled_mesh.faces) == 0:
        info["liquid_volume_fill_reason"] = "empty_filled_mesh"
        return None, info

    info.update(
        {
            "liquid_volume_fill_applied": True,
            "liquid_volume_fill_reason": "filled_from_surface_to_container_floor",
            "liquid_volume_fill_pitch": float(pitch),
            "liquid_volume_fill_floor_top_z": float(floor_top_z),
            "liquid_volume_fill_target_top_z": float(target_top_z),
            "liquid_volume_fill_free_surface_headroom": float(surface_headroom),
            "liquid_volume_fill_seed_voxels": int(len(seed_voxels)),
            "liquid_volume_fill_total_voxels": int(len(filled)),
            "liquid_volume_fill_column_info": column_info,
            "liquid_volume_fill_mesh_info": mesh_info,
        }
    )
    return filled_mesh, info


def _compute_liquid_container_bottom_seals(part_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seals: List[Dict[str, Any]] = []
    rigid_specs = [spec for spec in part_specs if str(spec.get("assembly_role", "free_soft")) == "rigid_skeleton"]
    liquid_specs = [
        spec for spec in part_specs
        if _is_liquid_spec(spec) and not _liquid_prefers_free_surface(spec)
    ]
    if not rigid_specs or not liquid_specs:
        return seals

    for liquid_spec in liquid_specs:
        liquid_min = np.asarray(
            liquid_spec.get("runtime_bounds_min", liquid_spec.get("bounds_min", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        liquid_max = np.asarray(
            liquid_spec.get("runtime_bounds_max", liquid_spec.get("bounds_max", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        liquid_size = np.maximum(liquid_max - liquid_min, 1e-6)
        liquid_center_xy = 0.5 * (liquid_min[:2] + liquid_max[:2])
        xy_pad = min(0.012, max(0.003, 0.06 * float(np.min(liquid_size[:2]))))

        support_tops: List[float] = []
        for rigid in rigid_specs:
            rigid_min = np.asarray(rigid.get("bounds_min", [0.0, 0.0, 0.0]), dtype=np.float64)
            rigid_max = np.asarray(rigid.get("bounds_max", [0.0, 0.0, 0.0]), dtype=np.float64)
            support_gap = float(liquid_min[2] - rigid_max[2])
            # Accept a nearby support floor even if the liquid mesh sits just a few millimeters
            # above it. This catches bowl bottoms that are modeled as a separate rigid part.
            if support_gap < -0.004 or support_gap > 0.035:
                continue
            if not (
                float(rigid_min[0]) - xy_pad <= float(liquid_center_xy[0]) <= float(rigid_max[0]) + xy_pad
                and float(rigid_min[1]) - xy_pad <= float(liquid_center_xy[1]) <= float(rigid_max[1]) + xy_pad
            ):
                continue
            support_tops.append(float(rigid_max[2]))

        if not support_tops:
            continue

        floor_top_z = max(support_tops)
        seal_thickness = min(0.008, max(0.004, 0.06 * float(np.min(liquid_size[:2]))))
        raise_above_floor = min(0.0025, 0.35 * seal_thickness)
        seal_min = np.array(
            [
                float(liquid_min[0] - xy_pad),
                float(liquid_min[1] - xy_pad),
                float(floor_top_z - seal_thickness),
            ],
            dtype=np.float64,
        )
        seal_max = np.array(
            [
                float(liquid_max[0] + xy_pad),
                float(liquid_max[1] + xy_pad),
                float(floor_top_z + raise_above_floor),
            ],
            dtype=np.float64,
        )
        if np.any(seal_max - seal_min <= 1e-6):
            continue
        seals.append(
            {
                "pid": int(liquid_spec.get("pid", -1)),
                "part_name": str(liquid_spec.get("part_name", "liquid")),
                "bounds_min": seal_min.tolist(),
                "bounds_max": seal_max.tolist(),
                "floor_top_z": float(floor_top_z),
            }
        )

    return seals


def _build_liquid_container_guard_mesh(
    liquid_mesh_path: Path,
    runtime_mesh_dir: Path,
    pitch: float,
) -> Tuple[Optional[str], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "applied": False,
        "reason": "guard_not_built",
        "mesh_path": "",
    }
    try:
        mesh = _load_trimesh_single(liquid_mesh_path)
    except Exception as exc:
        info["reason"] = f"load_failed:{type(exc).__name__}"
        return None, info

    mesh = sanitize_mesh(mesh)
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        info["reason"] = "empty_liquid_mesh"
        return None, info

    guard_pitch = float(np.clip(pitch, 0.0035, 0.008))
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    domain_min = bounds[0] - 2.5 * guard_pitch
    domain_max = bounds[1] + 2.5 * guard_pitch
    domain_shape = np.ceil((domain_max - domain_min) / guard_pitch).astype(np.int32) + 1
    if np.any(domain_shape <= 2):
        info["reason"] = "invalid_guard_domain"
        return None, info

    liquid_occ = _voxel_index_set_from_mesh(
        mesh=mesh,
        pitch=guard_pitch,
        domain_min=domain_min,
        domain_shape=domain_shape,
        fill_interior=True,
    )
    if not liquid_occ:
        info["reason"] = "empty_liquid_voxels"
        return None, info

    expanded: set[Tuple[int, int, int]] = set()
    for x, y, z in liquid_occ:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (x + dx, y + dy, z + dz)
                    if not (0 <= nb[0] < int(domain_shape[0]) and 0 <= nb[1] < int(domain_shape[1]) and 0 <= nb[2] < int(domain_shape[2])):
                        continue
                    expanded.add(nb)

    shell_occ = expanded.difference(liquid_occ)
    if not shell_occ:
        info["reason"] = "empty_guard_shell"
        return None, info

    liquid_top_z = float(bounds[1, 2])
    top_open_cut = float(liquid_top_z - 1.5 * guard_pitch)
    filtered_shell: set[Tuple[int, int, int]] = set()
    for idx in shell_occ:
        center = domain_min + guard_pitch * (np.asarray(idx, dtype=np.float64) + 0.5)
        if float(center[2]) >= top_open_cut:
            continue
        filtered_shell.add(idx)

    if not filtered_shell:
        info["reason"] = "guard_shell_removed_by_top_open_cut"
        return None, info

    centers = domain_min[None, :] + guard_pitch * (np.asarray(sorted(filtered_shell), dtype=np.float64) + 0.5)
    guard_mesh = trimesh.voxel.ops.multibox(
        centers=centers,
        pitch=guard_pitch,
        remove_internal_faces=True,
    )
    guard_mesh = sanitize_mesh(guard_mesh)
    if len(guard_mesh.vertices) == 0 or len(guard_mesh.faces) == 0:
        info["reason"] = "empty_guard_mesh"
        return None, info

    runtime_mesh_dir.mkdir(parents=True, exist_ok=True)
    guard_path = runtime_mesh_dir / f"{liquid_mesh_path.stem}_container_guard.obj"
    guard_mesh.export(guard_path)
    info.update(
        {
            "applied": True,
            "reason": "liquid_outer_guard_shell",
            "mesh_path": str(guard_path),
            "guard_pitch": float(guard_pitch),
            "liquid_voxel_count": int(len(liquid_occ)),
            "guard_voxel_count": int(len(filtered_shell)),
        }
    )
    return str(guard_path), info


def _compute_rigid_container_cavity_from_paths(
    collision_mesh_paths: List[Path],
    pitch: float,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "applied": False,
        "reason": "rigid_cavity_not_computed",
    }

    if not collision_mesh_paths:
        info["reason"] = "no_rigid_collision_meshes"
        return None, info

    rigid_meshes: List[trimesh.Trimesh] = []
    valid_paths: List[str] = []
    for path in collision_mesh_paths:
        try:
            rigid_meshes.append(_load_trimesh_single(path))
            valid_paths.append(str(path))
        except Exception:
            continue
    if not rigid_meshes:
        info["reason"] = "failed_to_load_rigid_collision_meshes"
        return None, info

    guard_pitch = float(np.clip(pitch, 0.0035, 0.008))
    merged_bounds = np.asarray(
        [
            np.min(np.vstack([m.bounds[0] for m in rigid_meshes]), axis=0),
            np.max(np.vstack([m.bounds[1] for m in rigid_meshes]), axis=0),
        ],
        dtype=np.float64,
    )
    domain_min = merged_bounds[0] - 3.0 * guard_pitch
    domain_max = merged_bounds[1] + 3.0 * guard_pitch
    domain_shape = np.ceil((domain_max - domain_min) / guard_pitch).astype(np.int32) + 1
    if np.any(domain_shape <= 3):
        info["reason"] = "invalid_rigid_guard_domain"
        return None, info

    occupied: set[Tuple[int, int, int]] = set()
    for mesh in rigid_meshes:
        occupied.update(
            _voxel_index_set_from_mesh(
                mesh=mesh,
                pitch=guard_pitch,
                domain_min=domain_min,
                domain_shape=domain_shape,
                fill_interior=True,
            )
        )
    if not occupied:
        info["reason"] = "empty_rigid_occupied_voxels"
        return None, info

    # Close small scan cracks before recovering the interior cavity.
    closed_occupied: set[Tuple[int, int, int]] = set()
    for x, y, z in occupied:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (x + dx, y + dy, z + dz)
                    if not (0 <= nb[0] < int(domain_shape[0]) and 0 <= nb[1] < int(domain_shape[1]) and 0 <= nb[2] < int(domain_shape[2])):
                        continue
                    closed_occupied.add(nb)

    rim_top_z = float(merged_bounds[1, 2])
    lid_z = float(rim_top_z - 0.5 * guard_pitch)
    lid_idx = int(np.clip(np.floor((lid_z - domain_min[2]) / guard_pitch + 1e-9), 0, int(domain_shape[2]) - 1))
    lid_xy_min = merged_bounds[0, :2] - 0.5 * guard_pitch
    lid_xy_max = merged_bounds[1, :2] + 0.5 * guard_pitch

    occupied_with_lid: set[Tuple[int, int, int]] = set(closed_occupied)
    for ix in range(int(domain_shape[0])):
        cx = float(domain_min[0] + guard_pitch * (ix + 0.5))
        if cx < float(lid_xy_min[0]) or cx > float(lid_xy_max[0]):
            continue
        for iy in range(int(domain_shape[1])):
            cy = float(domain_min[1] + guard_pitch * (iy + 0.5))
            if cy < float(lid_xy_min[1]) or cy > float(lid_xy_max[1]):
                continue
            occupied_with_lid.add((ix, iy, lid_idx))

    exterior: set[Tuple[int, int, int]] = set()
    queue: deque[Tuple[int, int, int]] = deque()
    for ix in range(int(domain_shape[0])):
        for iy in range(int(domain_shape[1])):
            for iz in (0, int(domain_shape[2]) - 1):
                idx = (ix, iy, iz)
                if idx in occupied_with_lid or idx in exterior:
                    continue
                exterior.add(idx)
                queue.append(idx)
    for ix in range(int(domain_shape[0])):
        for iz in range(int(domain_shape[2])):
            for iy in (0, int(domain_shape[1]) - 1):
                idx = (ix, iy, iz)
                if idx in occupied_with_lid or idx in exterior:
                    continue
                exterior.add(idx)
                queue.append(idx)
    for iy in range(int(domain_shape[1])):
        for iz in range(int(domain_shape[2])):
            for ix in (0, int(domain_shape[0]) - 1):
                idx = (ix, iy, iz)
                if idx in occupied_with_lid or idx in exterior:
                    continue
                exterior.add(idx)
                queue.append(idx)

    while queue:
        x, y, z = queue.popleft()
        for nb in (
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        ):
            if nb in occupied_with_lid or nb in exterior:
                continue
            if not (0 <= nb[0] < int(domain_shape[0]) and 0 <= nb[1] < int(domain_shape[1]) and 0 <= nb[2] < int(domain_shape[2])):
                continue
            exterior.add(nb)
            queue.append(nb)

    cavity_candidates: set[Tuple[int, int, int]] = set()
    for ix in range(int(domain_shape[0])):
        for iy in range(int(domain_shape[1])):
            for iz in range(lid_idx):
                idx = (ix, iy, iz)
                if idx in occupied_with_lid or idx in exterior:
                    continue
                cavity_candidates.add(idx)
    if not cavity_candidates:
        info["reason"] = "no_enclosed_container_cavity"
        return None, info

    components: List[set[Tuple[int, int, int]]] = []
    remaining = set(cavity_candidates)
    while remaining:
        seed = remaining.pop()
        comp = {seed}
        comp_queue: deque[Tuple[int, int, int]] = deque([seed])
        while comp_queue:
            x, y, z = comp_queue.popleft()
            for nb in (
                (x + 1, y, z),
                (x - 1, y, z),
                (x, y + 1, z),
                (x, y - 1, z),
                (x, y, z + 1),
                (x, y, z - 1),
            ):
                if nb not in remaining:
                    continue
                remaining.remove(nb)
                comp.add(nb)
                comp_queue.append(nb)
        components.append(comp)

    cavity = max(components, key=len)
    cavity_centers = domain_min[None, :] + guard_pitch * (np.asarray(sorted(cavity), dtype=np.float64) + 0.5)
    cavity_bounds = np.asarray(
        [
            np.min(cavity_centers - 0.5 * guard_pitch, axis=0),
            np.max(cavity_centers + 0.5 * guard_pitch, axis=0),
        ],
        dtype=np.float64,
    )

    cavity_data = {
        "collision_mesh_paths": valid_paths,
        "guard_pitch": float(guard_pitch),
        "merged_bounds": merged_bounds,
        "domain_min": domain_min,
        "domain_shape": domain_shape,
        "occupied": occupied,
        "closed_occupied": closed_occupied,
        "lid_idx": int(lid_idx),
        "cavity": cavity,
        "cavity_bounds": cavity_bounds,
        "components": len(components),
    }
    info.update(
        {
            "applied": True,
            "reason": "rigid_container_cavity_recovered",
            "guard_pitch": float(guard_pitch),
            "occupied_voxel_count": int(len(occupied)),
            "cavity_voxel_count": int(len(cavity)),
            "cavity_component_count": int(len(components)),
        }
    )
    return cavity_data, info


def _select_nearby_rigid_container_collision_meshes(
    liquid_bounds: np.ndarray,
    spec: Dict[str, Any],
    other_specs: List[Dict[str, Any]],
    pitch: float,
    fallback_collision_mesh_paths: Optional[List[Path]] = None,
) -> Tuple[List[Path], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "liquid_rigid_container_candidate_count": 0,
        "liquid_rigid_container_mesh_count": 0,
    }

    liquid_size = np.maximum(liquid_bounds[1] - liquid_bounds[0], 1e-6)
    min_xy = float(np.min(liquid_size[:2]))
    if min_xy <= 1e-6:
        info["liquid_rigid_container_reason"] = "liquid_xy_extent_too_small"
        return [], info

    liquid_center_xy = 0.5 * (liquid_bounds[0, :2] + liquid_bounds[1, :2])
    xy_pad = min(0.015, max(0.002, 0.08 * min_xy))
    vertical_pad = max(0.02, 3.0 * pitch)

    candidates: List[Path] = []
    all_rigid_paths: List[Path] = list(fallback_collision_mesh_paths or [])
    for other in other_specs:
        if int(other.get("pid", -1)) == int(spec.get("pid", -2)):
            continue
        if str(other.get("assembly_role", "free_soft")) != "rigid_skeleton":
            continue

        collision_mesh_path = other.get("collision_mesh_path")
        if not collision_mesh_path:
            continue
        path = Path(str(collision_mesh_path))
        if path.exists():
            all_rigid_paths.append(path)

        bounds_info = _mesh_bounds_info(Path(str(collision_mesh_path)), scale=float(other.get("scale", 1.0)))
        if bounds_info is None:
            bounds_info = {
                "bounds_min": other.get("bounds_min", [0.0, 0.0, 0.0]),
                "bounds_max": other.get("bounds_max", [0.0, 0.0, 0.0]),
            }

        other_min = np.asarray(bounds_info["bounds_min"], dtype=np.float64)
        other_max = np.asarray(bounds_info["bounds_max"], dtype=np.float64)
        xy_overlap_min = np.maximum(liquid_bounds[0, :2] - xy_pad, other_min[:2])
        xy_overlap_max = np.minimum(liquid_bounds[1, :2] + xy_pad, other_max[:2])
        center_inside_xy = bool(
            np.all(liquid_center_xy >= other_min[:2] - xy_pad)
            and np.all(liquid_center_xy <= other_max[:2] + xy_pad)
        )
        z_near = bool(other_min[2] <= liquid_bounds[1, 2] + vertical_pad and other_max[2] >= liquid_bounds[0, 2] - 0.12)
        if (np.all(xy_overlap_max > xy_overlap_min) or center_inside_xy) and z_near:
            if path.exists():
                candidates.append(path)

    if not candidates and all_rigid_paths:
        candidates = list(all_rigid_paths)
        info["liquid_rigid_container_reason"] = "fallback_all_rigid_collision_meshes"

    info["liquid_rigid_container_candidate_count"] = int(len(candidates))
    deduped: List[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    info["liquid_rigid_container_mesh_count"] = int(len(deduped))
    return deduped, info


def _build_liquid_mesh_from_rigid_cavity(
    liquid_mesh: trimesh.Trimesh,
    spec: Dict[str, Any],
    other_specs: List[Dict[str, Any]],
    pitch: float,
    fallback_collision_mesh_paths: Optional[List[Path]] = None,
    prefer_free_surface: bool = False,
) -> Tuple[Optional[trimesh.Trimesh], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "liquid_volume_fill_attempted": False,
        "liquid_volume_fill_applied": False,
    }

    liquid_bounds = np.asarray(liquid_mesh.bounds, dtype=np.float64)
    collision_mesh_paths, select_info = _select_nearby_rigid_container_collision_meshes(
        liquid_bounds=liquid_bounds,
        spec=spec,
        other_specs=other_specs,
        pitch=float(pitch),
        fallback_collision_mesh_paths=fallback_collision_mesh_paths,
    )
    info.update(select_info)
    info["liquid_volume_fill_attempted"] = True
    if not collision_mesh_paths:
        info["liquid_volume_fill_reason"] = str(select_info.get("liquid_rigid_container_reason", "no_nearby_rigid_container"))
        return None, info

    cavity_data, cavity_info = _compute_rigid_container_cavity_from_paths(
        collision_mesh_paths=collision_mesh_paths,
        pitch=float(pitch),
    )
    info.update({f"rigid_cavity_{k}": v for k, v in cavity_info.items()})
    if cavity_data is None:
        info["liquid_volume_fill_reason"] = str(cavity_info.get("reason", "rigid_cavity_recovery_failed"))
        return None, info

    guard_pitch = float(cavity_data["guard_pitch"])
    domain_min = np.asarray(cavity_data["domain_min"], dtype=np.float64)
    domain_shape = np.asarray(cavity_data["domain_shape"], dtype=np.int32)
    cavity = set(cavity_data["cavity"])
    cavity_bounds = np.asarray(cavity_data["cavity_bounds"], dtype=np.float64)
    cavity_top_z = float(cavity_bounds[1, 2])
    cavity_bottom_z = float(cavity_bounds[0, 2])
    cavity_height = max(cavity_top_z - cavity_bottom_z, guard_pitch)
    liquid_top_z = float(liquid_bounds[1, 2])
    # A completely full cavity reads like a soft plug rather than a bowl of liquid.
    # Keep some freeboard below the recovered rim so the fluid has room to move.
    freeboard = max(3.0 * guard_pitch, 0.10 * cavity_height)
    liquid_top_cap = liquid_top_z
    fill_height_cap = cavity_bottom_z + 0.84 * cavity_height
    bbox_fill_ratio = float(np.clip((liquid_top_z - cavity_bottom_z) / cavity_height, 0.18, 0.92))
    if prefer_free_surface:
        freeboard = max(4.5 * guard_pitch, 0.18 * cavity_height)
        liquid_top_cap = cavity_top_z - freeboard
        fill_height_cap = cavity_bottom_z + 0.72 * cavity_height
        target_fill_height = float(np.clip(bbox_fill_ratio * cavity_height - 0.75 * guard_pitch, 0.0, cavity_height - freeboard))
        target_top_z = min(
            cavity_bottom_z + target_fill_height,
            liquid_top_cap,
            fill_height_cap,
        )
    else:
        target_top_z = min(
            liquid_top_cap,
            cavity_top_z - freeboard,
            fill_height_cap,
        )
    min_fill_height = max(2.5 * guard_pitch, 0.01)
    if target_top_z - cavity_bottom_z <= min_fill_height:
        info["liquid_volume_fill_reason"] = "rigid_cavity_fill_height_too_small"
        return None, info

    filled_voxels, column_info = _fill_voxel_columns_to_height(
        voxels=cavity,
        domain_min=domain_min,
        pitch=guard_pitch,
        target_top_z=float(target_top_z),
    )
    if not filled_voxels:
        info["liquid_volume_fill_reason"] = "no_voxels_below_target_fill_height"
        return None, info

    filled_mesh, mesh_info = _voxel_set_to_closed_mesh(
        voxels=filled_voxels,
        domain_min=domain_min,
        domain_shape=domain_shape,
        pitch=guard_pitch,
    )
    if filled_mesh is None:
        info["liquid_volume_fill_reason"] = "empty_rigid_cavity_fill_mesh"
        info["liquid_volume_fill_column_info"] = column_info
        info["liquid_volume_fill_mesh_info"] = mesh_info
        return None, info
    if len(filled_mesh.vertices) == 0 or len(filled_mesh.faces) == 0:
        info["liquid_volume_fill_reason"] = "empty_rigid_cavity_fill_mesh"
        return None, info

    info.update(
        {
            "liquid_volume_fill_applied": True,
            "liquid_volume_fill_reason": "filled_from_rigid_container_cavity",
            "liquid_volume_fill_pitch": float(guard_pitch),
            "liquid_volume_fill_target_top_z": float(target_top_z),
            "liquid_volume_fill_cavity_bottom_z": float(cavity_bottom_z),
            "liquid_volume_fill_freeboard": float(freeboard),
            "liquid_volume_fill_liquid_top_cap": float(liquid_top_cap),
            "liquid_volume_fill_bbox_fill_ratio": float(bbox_fill_ratio),
            "liquid_volume_fill_total_voxels": int(len(filled_voxels)),
            "liquid_volume_fill_column_info": column_info,
            "liquid_volume_fill_mesh_info": mesh_info,
        }
    )
    return filled_mesh, info


def _build_rigid_container_guard_mesh(
    metadata: Dict[str, Any],
    runtime_mesh_dir: Path,
    pitch: float,
) -> Tuple[Optional[str], Dict[str, Any]]:
    info: Dict[str, Any] = {
        "applied": False,
        "reason": "rigid_guard_not_built",
        "mesh_path": "",
    }

    category = str(metadata.get("category", "")).strip().lower()
    object_name = str(metadata.get("object_name", "")).strip().lower()
    container_like = (
        "container" in category
        or any(tok in object_name for tok in ("bowl", "cup", "mug", "glass", "pot", "bucket"))
    )
    if not container_like:
        info["reason"] = "object_not_container_like"
        return None, info

    rigid_links = [rec for rec in metadata.get("rigid_part_links", []) if isinstance(rec, dict)]
    collision_mesh_paths: List[Path] = []
    for rec in rigid_links:
        collision_path = rec.get("collision_mesh_path")
        if collision_path:
            path = Path(str(collision_path))
            if path.exists():
                collision_mesh_paths.append(path)

    cavity_data, cavity_info = _compute_rigid_container_cavity_from_paths(
        collision_mesh_paths=collision_mesh_paths,
        pitch=float(pitch),
    )
    if cavity_data is None:
        info["reason"] = str(cavity_info.get("reason", "rigid_cavity_recovery_failed"))
        return None, info

    guard_pitch = float(cavity_data["guard_pitch"])
    domain_shape = np.asarray(cavity_data["domain_shape"], dtype=np.int32)
    cavity = set(cavity_data["cavity"])
    lid_idx = int(cavity_data["lid_idx"])
    domain_min = np.asarray(cavity_data["domain_min"], dtype=np.float64)
    guard_shell: set[Tuple[int, int, int]] = set()
    for x, y, z in cavity:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nb = (x + dx, y + dy, z + dz)
                    if not (0 <= nb[0] < int(domain_shape[0]) and 0 <= nb[1] < int(domain_shape[1]) and 0 <= nb[2] < int(domain_shape[2])):
                        continue
                    if nb in cavity:
                        continue
                    guard_shell.add(nb)

    top_open_cut = lid_idx - 1
    guard_shell = {idx for idx in guard_shell if idx[2] < top_open_cut}
    if not guard_shell:
        info["reason"] = "empty_rigid_guard_shell"
        return None, info

    centers = domain_min[None, :] + guard_pitch * (np.asarray(sorted(guard_shell), dtype=np.float64) + 0.5)
    guard_mesh = trimesh.voxel.ops.multibox(
        centers=centers,
        pitch=guard_pitch,
        remove_internal_faces=True,
    )
    guard_mesh = sanitize_mesh(guard_mesh)
    if len(guard_mesh.vertices) == 0 or len(guard_mesh.faces) == 0:
        info["reason"] = "empty_rigid_guard_mesh"
        return None, info

    runtime_mesh_dir.mkdir(parents=True, exist_ok=True)
    guard_path = runtime_mesh_dir / "rigid_container_guard.obj"
    guard_mesh.export(guard_path)
    info.update(
        {
            "applied": True,
            "reason": "rigid_container_cavity_guard",
            "mesh_path": str(guard_path),
            "guard_pitch": float(guard_pitch),
            "occupied_voxel_count": int(cavity_info.get("occupied_voxel_count", 0)),
            "cavity_voxel_count": int(len(cavity)),
            "guard_voxel_count": int(len(guard_shell)),
        }
    )
    return str(guard_path), info


def _prepare_eroded_soft_mesh(
    spec: Dict[str, Any],
    other_specs: List[Dict[str, Any]],
    runtime_mesh_dir: Path,
    object_bbox_size: np.ndarray,
    anchored_overlap_scale_boost: float = 1.0,
    rigid_collision_mesh_paths: Optional[List[Path]] = None,
) -> Tuple[str, Dict[str, Any]]:
    mesh_path = Path(str(spec["mesh_path"]))
    info = {
        "applied": False,
        "reason": "not_soft",
        "mesh_path": str(mesh_path),
    }

    role = str(spec.get("assembly_role", "free_soft"))
    if role == "rigid_skeleton":
        return str(mesh_path), info

    try:
        obj = trimesh.load(mesh_path, process=False)
        if isinstance(obj, trimesh.Scene):
            meshes = [g.copy() for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
            if not meshes:
                info["reason"] = "empty_scene"
                return str(mesh_path), info
            mesh = trimesh.util.concatenate(meshes)
        elif isinstance(obj, trimesh.Trimesh):
            mesh = obj.copy()
        else:
            info["reason"] = "unsupported_mesh"
            return str(mesh_path), info
    except Exception as exc:
        info["reason"] = f"load_failed:{type(exc).__name__}"
        return str(mesh_path), info

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    size = np.maximum(bounds[1] - bounds[0], 1e-6)
    min_obj_extent = float(np.min(np.maximum(object_bbox_size, 1e-6)))
    is_seat_surface = _is_seat_surface_spec(spec)
    ctor = str(spec.get("material_ctor_runtime", spec.get("material_ctor", "")))

    is_liquid = _is_liquid_spec(spec)
    prefer_free_surface = _liquid_prefers_free_surface(spec)

    # Particle fluids are especially sensitive to paper-thin, non-watertight meshes.
    # Convert them to a voxel-solid mesh first so Genesis can sample interior particles.
    if is_liquid:
        liquid_pitch = float(
            np.clip(
                max(0.6 * float(np.min(size)), min(0.08 * float(np.max(size)), 0.01)),
                0.002,
                0.012,
            )
        )
        if prefer_free_surface:
            info.update(
                {
                    "liquid_solidified": False,
                    "liquid_solidify_pitch": float(liquid_pitch),
                    "liquid_solidify_skipped": "prefer_free_surface",
                }
            )
        else:
            try:
                solid_mesh, solid_meta = voxel_fill_mesh_collision(mesh, liquid_pitch)
                if len(solid_mesh.vertices) > 0 and len(solid_mesh.faces) > 0:
                    mesh = solid_mesh
                    bounds = np.asarray(mesh.bounds, dtype=np.float64)
                    size = np.maximum(bounds[1] - bounds[0], 1e-6)
                    info.update(
                        {
                            "liquid_solidified": True,
                            "liquid_solidify_pitch": float(liquid_pitch),
                            "liquid_solidify_meta": solid_meta,
                        }
                    )
            except Exception as exc:
                info.update(
                    {
                        "liquid_solidified": False,
                        "liquid_solidify_pitch": float(liquid_pitch),
                        "liquid_solidify_error": type(exc).__name__,
                    }
                )
        try:
            filled_liquid_mesh, fill_info = _build_liquid_mesh_from_rigid_cavity(
                liquid_mesh=mesh,
                spec=spec,
                other_specs=other_specs,
                pitch=min(float(liquid_pitch), 0.005),
                fallback_collision_mesh_paths=rigid_collision_mesh_paths,
                prefer_free_surface=prefer_free_surface,
            )
            info.update(fill_info)
            if filled_liquid_mesh is not None and len(filled_liquid_mesh.vertices) > 0 and len(filled_liquid_mesh.faces) > 0:
                mesh = filled_liquid_mesh
                bounds = np.asarray(mesh.bounds, dtype=np.float64)
                size = np.maximum(bounds[1] - bounds[0], 1e-6)
                ensure_dir(runtime_mesh_dir)
                out_path = runtime_mesh_dir / f"{mesh_path.stem}_filled.obj"
                mesh.export(out_path)
                info.update(
                    {
                        "applied": True,
                        "reason": "liquid_container_volume_fill",
                        "mesh_path": str(out_path),
                        "alignment_mode": "keep_top_xy_center",
                    }
                )
                return str(out_path), info
            print(
                f"🫧 rigid_cavity_fill skipped pid={spec.get('pid', -1)} "
                f"reason={fill_info.get('liquid_volume_fill_reason', 'unknown')} "
                f"candidates={fill_info.get('liquid_rigid_container_candidate_count', 0)} "
                f"meshes={fill_info.get('liquid_rigid_container_mesh_count', 0)}"
            )
        except Exception as exc:
            info["liquid_rigid_cavity_fill_error"] = type(exc).__name__
            print(
                f"🫧 rigid_cavity_fill error pid={spec.get('pid', -1)} "
                f"error={type(exc).__name__}"
            )
        try:
            filled_liquid_mesh, fill_info = _build_container_filled_liquid_mesh(
                liquid_mesh=mesh,
                spec=spec,
                other_specs=other_specs,
                pitch=min(float(liquid_pitch), 0.005),
                prefer_free_surface=prefer_free_surface,
            )
            info.update({f"fallback_{k}": v for k, v in fill_info.items()})
            if filled_liquid_mesh is not None and len(filled_liquid_mesh.vertices) > 0 and len(filled_liquid_mesh.faces) > 0:
                mesh = filled_liquid_mesh
                bounds = np.asarray(mesh.bounds, dtype=np.float64)
                size = np.maximum(bounds[1] - bounds[0], 1e-6)
                ensure_dir(runtime_mesh_dir)
                out_path = runtime_mesh_dir / f"{mesh_path.stem}_filled.obj"
                mesh.export(out_path)
                info.update(
                    {
                        "applied": True,
                        "reason": "liquid_container_volume_fill_fallback",
                        "mesh_path": str(out_path),
                        "alignment_mode": "keep_top_xy_center",
                    }
                )
                return str(out_path), info
        except Exception as exc:
            info["liquid_volume_fill_error"] = type(exc).__name__
        ensure_dir(runtime_mesh_dir)
        out_path = runtime_mesh_dir / f"{mesh_path.stem}_solidified.obj"
        mesh.export(out_path)
        info.update(
            {
                "applied": True,
                "reason": "liquid_solidified_preserve_container_overlap",
                "mesh_path": str(out_path),
            }
        )
        return str(out_path), info

    if role == "free_soft":
        is_free_cloth = bool(_is_free_cloth_like_spec(spec))
        clearance = min(0.016 if is_free_cloth else 0.010, max(0.004 if is_free_cloth else 0.003, (0.014 if is_free_cloth else 0.008) * min_obj_extent))
        cloth_follow_gap = min(0.032 if is_free_cloth else 0.020, max(0.012 if is_free_cloth else 0.006, (0.030 if is_free_cloth else 0.015) * min_obj_extent))

        def _collect_rigid_boxes_for_free_soft() -> List[Tuple[np.ndarray, np.ndarray]]:
            boxes: List[Tuple[np.ndarray, np.ndarray]] = []
            for other in other_specs:
                if int(other.get("pid", -1)) == int(spec.get("pid", -2)):
                    continue
                if str(other.get("assembly_role", "free_soft")) != "rigid_skeleton":
                    continue
                collision_mesh_path = other.get("collision_mesh_path")
                box_min = None
                box_max = None
                if collision_mesh_path:
                    bounds_info = _mesh_bounds_info(Path(str(collision_mesh_path)), scale=float(other.get("scale", 1.0)))
                    if bounds_info is not None:
                        box_min = np.asarray(bounds_info["bounds_min"], dtype=np.float64)
                        box_max = np.asarray(bounds_info["bounds_max"], dtype=np.float64)
                if box_min is None or box_max is None:
                    box_min = np.asarray(other.get("bounds_min", [0.0, 0.0, 0.0]), dtype=np.float64)
                    box_max = np.asarray(other.get("bounds_max", [0.0, 0.0, 0.0]), dtype=np.float64)
                if _is_free_cloth_like_spec(spec):
                    box_max = box_max.copy()
                    box_max[2] -= cloth_follow_gap
                boxes.append((box_min - clearance, box_max + clearance))
            return boxes

        def _has_box_overlap(mesh_bounds: np.ndarray, boxes: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
            for box_min, box_max in boxes:
                overlap_min = np.maximum(mesh_bounds[0], box_min)
                overlap_max = np.minimum(mesh_bounds[1], box_max)
                if np.all(overlap_max > overlap_min):
                    return True
            return False

        def _compute_free_soft_exit_shift(
            mesh_bounds: np.ndarray,
            boxes: List[Tuple[np.ndarray, np.ndarray]],
            max_shift: float,
        ) -> Optional[np.ndarray]:
            best_shift = None
            best_mag = None
            center = 0.5 * (mesh_bounds[0] + mesh_bounds[1])
            for box_min, box_max in boxes:
                overlap_min = np.maximum(mesh_bounds[0], box_min)
                overlap_max = np.minimum(mesh_bounds[1], box_max)
                if np.any(overlap_max <= overlap_min):
                    continue
                overlap_size = overlap_max - overlap_min
                axis = int(np.argmin(overlap_size))
                box_center = 0.5 * (box_min + box_max)
                direction = 1.0 if center[axis] >= box_center[axis] else -1.0
                if direction > 0:
                    shift_mag = float(box_max[axis] - mesh_bounds[0, axis] + clearance)
                else:
                    shift_mag = float(mesh_bounds[1, axis] - box_min[axis] + clearance)
                shift_mag = min(max_shift, max(0.0, shift_mag))
                if shift_mag <= 0.0:
                    continue
                shift = np.zeros(3, dtype=np.float64)
                shift[axis] = direction * shift_mag
                if best_mag is None or shift_mag < best_mag:
                    best_mag = shift_mag
                    best_shift = shift
            return best_shift

        center = 0.5 * (bounds[0] + bounds[1])
        base_margin = min(0.018 if is_free_cloth else 0.010, (0.075 if is_free_cloth else 0.04) * min_obj_extent)
        axis_margin = np.minimum(np.full(3, base_margin, dtype=np.float64), 0.12 * size)
        scales = np.clip((size - 2.0 * axis_margin) / size, 0.68 if is_free_cloth else 0.75, 1.0)
        if np.all(scales > 0.999):
            info["reason"] = "erosion_too_small"
            return str(mesh_path), info
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        mesh.vertices = (verts - center) * scales + center
        rigid_boxes = _collect_rigid_boxes_for_free_soft()
        current_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        total_shift = np.zeros(3, dtype=np.float64)
        for _ in range(8):
            if not rigid_boxes or not _has_box_overlap(current_bounds, rigid_boxes):
                break
            shift = _compute_free_soft_exit_shift(
                mesh_bounds=current_bounds,
                boxes=rigid_boxes,
                max_shift=min(0.040 if is_free_cloth else 0.025, (0.22 if is_free_cloth else 0.12) * min_obj_extent),
            )
            if shift is None or float(np.linalg.norm(shift)) <= 1e-8:
                break
            mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) + shift[None, :]
            total_shift += shift
            current_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        ensure_dir(runtime_mesh_dir)
        out_path = runtime_mesh_dir / f"{mesh_path.stem}_eroded.obj"
        mesh.export(out_path)
        info.update(
            {
                "applied": True,
                "reason": "soft_global_shrink",
                "mesh_path": str(out_path),
                "axis_margin": axis_margin.tolist(),
                "scales": scales.tolist(),
                "shift": total_shift.tolist(),
            }
        )
        return str(out_path), info

    clearance = min(0.014, max(0.005, 0.012 * min_obj_extent))
    pitch = min(0.012, max(0.006, 0.02 * min_obj_extent))
    overlap_bounds_min = bounds[0] - clearance
    overlap_bounds_max = bounds[1] + clearance
    is_small_anchored_piece = bool(float(np.max(size)) < 0.35 and float(np.prod(size)) < 0.01)
    is_large_anchored_piece = not is_small_anchored_piece
    anchored_overlap_scale_boost = float(max(1.0, anchored_overlap_scale_boost))
    rigid_boxes: List[Tuple[np.ndarray, np.ndarray]] = []
    rigid_boxes_actual: List[Tuple[np.ndarray, np.ndarray]] = []

    def _effective_overlap_bounds(other: Dict[str, Any], relax: bool) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        collision_mesh_path = other.get("collision_mesh_path")
        box_min = None
        box_max = None
        if collision_mesh_path:
            bounds_info = _mesh_bounds_info(Path(str(collision_mesh_path)), scale=float(other.get("scale", 1.0)))
            if bounds_info is not None:
                box_min = np.asarray(bounds_info["bounds_min"], dtype=np.float64)
                box_max = np.asarray(bounds_info["bounds_max"], dtype=np.float64)
        if box_min is None or box_max is None:
            box_min = np.asarray(other.get("bounds_min", [0.0, 0.0, 0.0]), dtype=np.float64)
            box_max = np.asarray(other.get("bounds_max", [0.0, 0.0, 0.0]), dtype=np.float64)

        box_size = np.maximum(box_max - box_min, 1e-6)
        if not relax:
            return box_min - clearance, box_max + clearance

        # When the soft part is already close to its safe minimum scale, let the rigid overlap proxy
        # retreat slightly instead of forcing the soft shell to collapse into an unnaturally thin slab.
        rigid_relax = np.minimum(
            np.full(3, 0.4 * clearance, dtype=np.float64),
            0.04 * box_size,
        )
        if np.any(box_size - 2.0 * rigid_relax <= 1e-6):
            rigid_relax = np.minimum(rigid_relax, 0.2 * box_size)
        return box_min + rigid_relax - clearance, box_max - rigid_relax + clearance

    for other in other_specs:
        if int(other.get("pid", -1)) == int(spec.get("pid", -2)):
            continue
        if str(other.get("assembly_role", "free_soft")) != "rigid_skeleton":
            continue
        effective_bounds = _effective_overlap_bounds(other, relax=True)
        actual_bounds = _effective_overlap_bounds(other, relax=False)
        if effective_bounds is None or actual_bounds is None:
            continue
        other_bounds_min, other_bounds_max = effective_bounds
        actual_bounds_min, actual_bounds_max = actual_bounds
        if np.any(overlap_bounds_max < other_bounds_min) or np.any(other_bounds_max < overlap_bounds_min):
            continue
        rigid_boxes.append((other_bounds_min, other_bounds_max))
        rigid_boxes_actual.append((actual_bounds_min, actual_bounds_max))

    if not rigid_boxes:
        info["reason"] = "no_rigid_overlap_detected"
        return str(mesh_path), info

    def _compute_box_exit_shift(
        mesh_bounds: np.ndarray,
        boxes: List[Tuple[np.ndarray, np.ndarray]],
        max_shift: float,
    ) -> Optional[np.ndarray]:
        best_shift = None
        best_mag = None
        for box_min, box_max in boxes:
            overlap_min = np.maximum(mesh_bounds[0], box_min)
            overlap_max = np.minimum(mesh_bounds[1], box_max)
            if np.any(overlap_max <= overlap_min):
                continue
            overlap_size = overlap_max - overlap_min
            axis = int(np.argmin(overlap_size))
            center = 0.5 * (mesh_bounds[0] + mesh_bounds[1])
            box_center = 0.5 * (box_min + box_max)
            direction = 1.0 if center[axis] >= box_center[axis] else -1.0
            if direction > 0:
                shift_mag = float(box_max[axis] - mesh_bounds[0, axis] + clearance)
            else:
                shift_mag = float(mesh_bounds[1, axis] - box_min[axis] + clearance)
            shift_mag = min(max_shift, max(0.0, shift_mag))
            if shift_mag <= 0.0:
                continue
            shift = np.zeros(3, dtype=np.float64)
            shift[axis] = direction * shift_mag
            if best_mag is None or shift_mag < best_mag:
                best_mag = shift_mag
                best_shift = shift
        return best_shift

    def _compute_overlap_shifts(
        mesh_bounds: np.ndarray,
        boxes: List[Tuple[np.ndarray, np.ndarray]],
        max_shift: float,
    ) -> List[np.ndarray]:
        shifts: List[np.ndarray] = []
        center = 0.5 * (mesh_bounds[0] + mesh_bounds[1])
        for box_min, box_max in boxes:
            overlap_min = np.maximum(mesh_bounds[0], box_min)
            overlap_max = np.minimum(mesh_bounds[1], box_max)
            if np.any(overlap_max <= overlap_min):
                continue
            overlap_size = overlap_max - overlap_min
            axis = int(np.argmin(overlap_size))
            box_center = 0.5 * (box_min + box_max)
            direction = 1.0 if center[axis] >= box_center[axis] else -1.0
            if direction > 0:
                shift_mag = float(box_max[axis] - mesh_bounds[0, axis] + clearance)
            else:
                shift_mag = float(mesh_bounds[1, axis] - box_min[axis] + clearance)
            shift_mag = min(max_shift, max(0.0, shift_mag))
            if shift_mag <= 0.0:
                continue
            shift = np.zeros(3, dtype=np.float64)
            shift[axis] = direction * shift_mag
            shifts.append(shift)
        return shifts
    def _max_overlap_margin(mesh_bounds: np.ndarray, boxes: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        overlap_margin = np.zeros(3, dtype=np.float64)
        for box_min, box_max in boxes:
            overlap_min = np.maximum(mesh_bounds[0], box_min)
            overlap_max = np.minimum(mesh_bounds[1], box_max)
            if np.any(overlap_max <= overlap_min):
                continue
            overlap_margin = np.maximum(overlap_margin, overlap_max - overlap_min)
        return overlap_margin

    current_bounds = np.asarray(bounds, dtype=np.float64)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    center = 0.5 * (current_bounds[0] + current_bounds[1])
    applied_scales = []
    final_shift = None
    restored_axis_scales: List[float] = [1.0, 1.0, 1.0]

    def _has_overlap(mesh_bounds: np.ndarray, boxes: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        for box_min, box_max in boxes:
            overlap_min = np.maximum(mesh_bounds[0], box_min)
            overlap_max = np.minimum(mesh_bounds[1], box_max)
            if np.all(overlap_max > overlap_min):
                return True
        return False

    def _try_restore_volume_along_safe_axes(
        current_vertices: np.ndarray,
        current_mesh_bounds: np.ndarray,
        original_bounds: np.ndarray,
        boxes: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        if not boxes:
            return current_vertices, current_mesh_bounds, [1.0, 1.0, 1.0]

        working_vertices = np.asarray(current_vertices, dtype=np.float64).copy()
        working_bounds = np.asarray(current_mesh_bounds, dtype=np.float64).copy()
        original_size = np.maximum(original_bounds[1] - original_bounds[0], 1e-6)
        current_size = np.maximum(working_bounds[1] - working_bounds[0], 1e-6)
        target_scale_cap = 6.5 if is_seat_surface else 1.35
        target_scale = np.clip(original_size / current_size, 1.0, target_scale_cap)
        axis_restore = [1.0, 1.0, 1.0]

        # Prefer restoring thickness first, then the middle axis, and preserve the longest axis.
        axis_order = list(np.argsort(current_size))
        restore_center = 0.5 * (working_bounds[0] + working_bounds[1])

        for axis in axis_order:
            goal = float(target_scale[axis])
            if goal <= 1.01:
                continue

            lo = 1.0
            hi = goal
            best = 1.0
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                trial = working_vertices.copy()
                trial[:, axis] = restore_center[axis] + (trial[:, axis] - restore_center[axis]) * mid
                trial_min = trial.min(axis=0)
                trial_max = trial.max(axis=0)
                if _has_overlap(np.stack([trial_min, trial_max], axis=0), boxes):
                    hi = mid
                else:
                    best = mid
                    lo = mid

            if best > 1.001:
                working_vertices[:, axis] = restore_center[axis] + (working_vertices[:, axis] - restore_center[axis]) * best
                working_bounds = np.stack([working_vertices.min(axis=0), working_vertices.max(axis=0)], axis=0)
                restore_center = 0.5 * (working_bounds[0] + working_bounds[1])
                axis_restore[axis] = best

        return working_vertices, working_bounds, axis_restore

    def _try_restore_seat_surface_support(
        current_vertices: np.ndarray,
        current_mesh_bounds: np.ndarray,
        original_bounds: np.ndarray,
        boxes: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        if not is_seat_surface or not boxes:
            return current_vertices, current_mesh_bounds, {}

        working_vertices = np.asarray(current_vertices, dtype=np.float64).copy()
        working_bounds = np.asarray(current_mesh_bounds, dtype=np.float64).copy()
        restored: Dict[str, float] = {}

        def _restore_positive_side(axis: int, target_max: float, tag: str):
            nonlocal working_vertices, working_bounds, restored
            current_min = float(working_bounds[0, axis])
            current_max = float(working_bounds[1, axis])
            if target_max <= current_max + 1e-4:
                return
            span = max(current_max - current_min, 1e-6)
            target_scale = min(6.5, max(1.0, (target_max - current_min) / span))
            lo = 1.0
            hi = target_scale
            best = 1.0
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                trial = working_vertices.copy()
                trial[:, axis] = current_min + (trial[:, axis] - current_min) * mid
                trial_bounds = np.stack([trial.min(axis=0), trial.max(axis=0)], axis=0)
                if _has_overlap(trial_bounds, boxes):
                    hi = mid
                else:
                    best = mid
                    lo = mid
            if best > 1.001:
                working_vertices[:, axis] = current_min + (working_vertices[:, axis] - current_min) * best
                working_bounds = np.stack([working_vertices.min(axis=0), working_vertices.max(axis=0)], axis=0)
                restored[tag] = float(best)

        _restore_positive_side(axis=1, target_max=float(original_bounds[1, 1]), tag="front_y")
        _restore_positive_side(axis=2, target_max=float(original_bounds[1, 2]), tag="top_z")
        return working_vertices, working_bounds, restored

    for _ in range(6):
        overlap_margin = _max_overlap_margin(current_bounds, rigid_boxes)
        if not np.any(overlap_margin > 1e-6):
            break
        if is_large_anchored_piece:
            axis_margin = np.zeros(3, dtype=np.float64)
            preserve_axis = int(np.argmax(size))
            thickness_axis = int(np.argmin(size))
            axis_margin[thickness_axis] = min(0.32 * size[thickness_axis], anchored_overlap_scale_boost * (1.35 * overlap_margin[thickness_axis] + clearance))
            secondary_axis = int(np.argsort(size)[1])
            if secondary_axis != preserve_axis:
                axis_margin[secondary_axis] = min(0.22 * size[secondary_axis], anchored_overlap_scale_boost * (0.9 * overlap_margin[secondary_axis] + 0.7 * clearance))
            min_scale = 0.74
        else:
            margin_multiplier = 1.3 * anchored_overlap_scale_boost
            margin_cap = min(0.28, 0.16 * anchored_overlap_scale_boost)
            min_scale = 0.74
            axis_margin = np.minimum(
                margin_multiplier * overlap_margin + clearance,
                margin_cap * size,
            )
        scales = np.clip((size - 2.0 * axis_margin) / size, min_scale, 1.0)
        if np.all(scales > 0.995):
            break
        verts = (verts - center) * scales + center
        mesh.vertices = verts
        current_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        applied_scales.append(scales.tolist())
        final_shift = _compute_box_exit_shift(
            mesh_bounds=current_bounds,
            boxes=rigid_boxes,
            max_shift=min(0.05 if is_small_anchored_piece else 0.035, 0.16 * min_obj_extent if is_small_anchored_piece else 0.12 * min_obj_extent),
        )
        if final_shift is not None:
            verts = verts + final_shift[None, :]
            mesh.vertices = verts
            current_bounds = np.asarray(mesh.bounds, dtype=np.float64)
        if not _has_overlap(current_bounds, rigid_boxes):
            break

    if applied_scales:
        residual_overlap = _has_overlap(current_bounds, rigid_boxes)
        total_extra_shift = np.zeros(3, dtype=np.float64)
        for _ in range(4):
            if not residual_overlap:
                break
            extra_shift = _compute_box_exit_shift(
                mesh_bounds=current_bounds,
                boxes=rigid_boxes,
                max_shift=min(0.06 if is_small_anchored_piece else 0.045, 0.18 * min_obj_extent if is_small_anchored_piece else 0.14 * min_obj_extent),
            )
            if extra_shift is None or float(np.linalg.norm(extra_shift)) <= 1e-8:
                break
            verts = np.asarray(mesh.vertices, dtype=np.float64) + extra_shift[None, :]
            mesh.vertices = verts
            current_bounds = np.asarray(mesh.bounds, dtype=np.float64)
            total_extra_shift += extra_shift
            residual_overlap = _has_overlap(current_bounds, rigid_boxes)
        final_validation_overlap = _has_overlap(current_bounds, rigid_boxes_actual)
        validation_shift = np.zeros(3, dtype=np.float64)
        for _ in range(6):
            if not final_validation_overlap:
                break
            candidate_shifts = _compute_overlap_shifts(
                mesh_bounds=current_bounds,
                boxes=rigid_boxes_actual,
                max_shift=min(0.02 if is_small_anchored_piece else 0.012, 0.06 * min_obj_extent if is_small_anchored_piece else 0.04 * min_obj_extent),
            )
            if not candidate_shifts:
                break
            extra_shift = candidate_shifts[0]
            if len(candidate_shifts) > 1:
                extra_shift = np.sum(np.asarray(candidate_shifts, dtype=np.float64), axis=0)
            if float(np.linalg.norm(extra_shift)) <= 1e-8:
                break
            verts = np.asarray(mesh.vertices, dtype=np.float64) + extra_shift[None, :]
            mesh.vertices = verts
            current_bounds = np.asarray(mesh.bounds, dtype=np.float64)
            validation_shift += extra_shift
            final_validation_overlap = _has_overlap(current_bounds, rigid_boxes_actual)
        if float(np.linalg.norm(total_extra_shift)) > 0.0:
            final_shift = total_extra_shift if final_shift is None else (np.asarray(final_shift, dtype=np.float64) + total_extra_shift)
        if float(np.linalg.norm(validation_shift)) > 0.0:
            final_shift = validation_shift if final_shift is None else (np.asarray(final_shift, dtype=np.float64) + validation_shift)

        # Recover some of the lost seat/cushion thickness using the original mesh as the target,
        # but only along axes that still remain overlap-free after restoration.
        verts, current_bounds, restored_axis_scales = _try_restore_volume_along_safe_axes(
            current_vertices=np.asarray(mesh.vertices, dtype=np.float64),
            current_mesh_bounds=current_bounds,
            original_bounds=bounds,
            boxes=rigid_boxes_actual,
        )
        seat_support_restore = {}
        verts, current_bounds, seat_support_restore = _try_restore_seat_surface_support(
            current_vertices=verts,
            current_mesh_bounds=current_bounds,
            original_bounds=bounds,
            boxes=rigid_boxes_actual,
        )
        mesh.vertices = verts

        ensure_dir(runtime_mesh_dir)
        out_path = runtime_mesh_dir / f"{mesh_path.stem}_scaled.obj"
        mesh.export(out_path)
        info.update(
            {
                "applied": True,
                "reason": "anchored_soft_overlap_scale",
                "mesh_path": str(out_path),
                "candidate_count": int(len(rigid_boxes)),
                "scales_per_iter": applied_scales,
                "clearance": float(clearance),
                "shift": None if final_shift is None else np.asarray(final_shift, dtype=np.float64).tolist(),
                "residual_overlap": bool(_has_overlap(current_bounds, rigid_boxes_actual)),
                "restored_axis_scales": [float(v) for v in restored_axis_scales],
                "seat_support_restore": seat_support_restore,
            }
        )
        return str(out_path), info

    shift = _compute_box_exit_shift(
        mesh_bounds=current_bounds,
        boxes=rigid_boxes,
        max_shift=min(0.03, 0.12 * min_obj_extent),
    )
    if shift is None:
        info["reason"] = "anchored_overlap_too_small"
        return str(mesh_path), info
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    mesh.vertices = verts + shift[None, :]

    ensure_dir(runtime_mesh_dir)
    out_path = runtime_mesh_dir / f"{mesh_path.stem}_shifted.obj"
    mesh.export(out_path)
    info.update(
        {
            "applied": True,
            "reason": "anchored_soft_exit_shift",
            "mesh_path": str(out_path),
            "candidate_count": int(len(rigid_boxes)),
            "clearance": float(clearance),
            "shift": shift.tolist(),
        }
    )
    return str(out_path), info


def _find_existing_runtime_mesh(
    spec: Dict[str, Any],
    runtime_mesh_dir: Path,
) -> Optional[Path]:
    mesh_stem = Path(str(spec["mesh_path"])).stem
    candidates = [
        runtime_mesh_dir / f"{mesh_stem}_filled.obj",
        runtime_mesh_dir / f"{mesh_stem}_solidified.obj",
        runtime_mesh_dir / f"{mesh_stem}_scaled.obj",
        runtime_mesh_dir / f"{mesh_stem}_shifted.obj",
        runtime_mesh_dir / f"{mesh_stem}_eroded.obj",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _match_anchored_soft_to_rigid_links(
    anchored_specs: List[Dict[str, Any]],
    rigid_specs: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    part_id_to_link = {}
    for rec in metadata.get("rigid_part_links", []):
        pid = rec.get("part_id")
        link_name = rec.get("link_name")
        if pid is not None and link_name:
            part_id_to_link[int(pid)] = str(link_name)

    bindings: List[Dict[str, Any]] = []
    for spec in anchored_specs:
        center = np.asarray(spec.get("bounds_center", [0.0, 0.0, 0.0]), dtype=np.float64)
        nearest = None
        for rigid in rigid_specs:
            rigid_center = np.asarray(rigid.get("bounds_center", [0.0, 0.0, 0.0]), dtype=np.float64)
            dist = float(np.linalg.norm(center - rigid_center))
            rec = (dist, int(rigid["pid"]), str(rigid.get("part_name", "")))
            if nearest is None or rec < nearest:
                nearest = rec
        if nearest is None:
            continue

        _, rigid_pid, rigid_name = nearest
        link_name = part_id_to_link.get(rigid_pid)
        if link_name is None:
            continue

        bindings.append(
            {
                "anchored_pid": int(spec["pid"]),
                "anchored_name": str(spec.get("part_name", f"part_{spec['pid']}")),
                "mesh_path": str(spec.get("runtime_mesh_path", spec["mesh_path"])),
                "euler": tuple(spec.get("euler", (0.0, 0.0, 0.0))),
                "link_name": link_name,
                "rigid_pid": rigid_pid,
                "rigid_name": rigid_name,
            }
        )

    return bindings


def _make_anchored_soft_hybrid_callbacks(gs: Any, bindings: List[Dict[str, Any]], placed_pos: np.ndarray):
    import genesis.utils.geom as gu
    import trimesh

    soft_meshes = []
    for b in bindings:
        obj = trimesh.load(b["mesh_path"], process=False)
        if isinstance(obj, trimesh.Scene):
            meshes = [g.copy() for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
            mesh = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
        elif isinstance(obj, trimesh.Trimesh):
            mesh = obj.copy()
        else:
            mesh = trimesh.Trimesh()
        soft_meshes.append(mesh)
    soft_poss = [tuple(np.asarray(placed_pos, dtype=float).tolist()) for _ in bindings]
    soft_eulers = [tuple(b.get("euler", (0.0, 0.0, 0.0))) for b in bindings]

    def func_instantiate_soft_from_rigid(scene, part_rigid, material_soft, material_hybrid, surface):
        return scene.add_entity(
            material=material_soft,
            morph=gs.morphs.MeshSet(
                files=soft_meshes,
                poss=soft_poss,
                eulers=soft_eulers,
                scale=1.0,
                file_meshes_are_zup=True,
            ),
            surface=surface,
        )

    def func_instantiate_rigid_soft_association(part_rigid, part_soft):
        muscle_group = None  # MeshSet already assigns one group per mesh.
        link_idcs = []
        geom_idcs = []
        trans_local_to_global = []
        quat_local_to_global = []

        for bind in bindings:
            link = part_rigid.get_link(bind["link_name"])
            if len(link.geoms) < 1:
                continue
            geom = link.geoms[0]
            trans, quat = gu.transform_pos_quat_by_trans_quat(
                geom.init_pos,
                geom.init_quat,
                link.init_x_pos,
                link.init_x_quat,
            )
            link_idcs.append(link.idx)
            geom_idcs.append(geom.idx)
            trans_local_to_global.append(trans)
            quat_local_to_global.append(quat)

        return muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global

    return func_instantiate_soft_from_rigid, func_instantiate_rigid_soft_association


def _apply_light_anchored_constraints(
    articulated_ent: Any,
    anchored_runtime_entities: List[Dict[str, Any]],
    anchored_bindings: List[Dict[str, Any]],
    rigid_specs: List[Dict[str, Any]],
    rigid_pos: np.ndarray,
    stiffness: float,
):
    if articulated_ent is None or not anchored_runtime_entities or not anchored_bindings:
        return []

    import torch

    pid_to_binding = {int(b["anchored_pid"]): b for b in anchored_bindings}
    rigid_pid_to_spec = {int(spec["pid"]): spec for spec in rigid_specs}
    applied = []

    for rec in anchored_runtime_entities:
        spec = rec["spec"]
        ent = rec["entity"]
        pid = int(spec["pid"])
        binding = pid_to_binding.get(pid)
        rigid_spec = rigid_pid_to_spec.get(int(binding["rigid_pid"])) if binding is not None else None
        if binding is None or rigid_spec is None:
            continue

        try:
            link = articulated_ent.get_link(binding["link_name"])
        except Exception:
            continue

        bbox_min = np.asarray(rigid_spec.get("bounds_min", [0.0, 0.0, 0.0]), dtype=np.float64) + rigid_pos
        bbox_max = np.asarray(rigid_spec.get("bounds_max", [0.0, 0.0, 0.0]), dtype=np.float64) + rigid_pos
        bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
        proximity = float(min(0.06, max(0.018, 0.18 * float(np.min(bbox_size)))))
        expanded_min = bbox_min - proximity
        expanded_max = bbox_max + proximity

        poss = ent.get_particles_pos()
        if poss.ndim == 3:
            poss = poss[0]
        if poss.shape[0] == 0:
            continue

        bbox_min_t = torch.as_tensor(expanded_min, dtype=poss.dtype, device=poss.device)
        bbox_max_t = torch.as_tensor(expanded_max, dtype=poss.dtype, device=poss.device)
        near_box_mask = ((poss >= bbox_min_t) & (poss <= bbox_max_t)).all(dim=-1)

        if int(near_box_mask.sum().item()) == 0:
            clamped = torch.minimum(torch.maximum(poss, bbox_min_t), bbox_max_t)
            dist = torch.linalg.norm(poss - clamped, dim=-1)
            target_count = max(32, min(int(ent.n_particles * 0.35), 1024))
            target_count = min(target_count, int(dist.numel()))
            nearest_idx = torch.topk(dist, k=target_count, largest=False).indices
            near_box_mask = torch.zeros_like(dist, dtype=torch.bool)
            near_box_mask[nearest_idx] = True

        particles_mask = near_box_mask.unsqueeze(0)
        ent.set_particle_constraints(particles_mask=particles_mask, link_idx=int(link.idx), stiffness=float(stiffness))
        applied.append(
            {
                "pid": pid,
                "part_name": str(spec.get("part_name", f"part_{pid}")),
                "link_name": str(binding["link_name"]),
                "link_idx": int(link.idx),
                "n_constrained": int(near_box_mask.sum().item()),
                "stiffness": float(stiffness),
            }
        )

    return applied


def _apply_layered_anchored_constraints(
    articulated_ent: Any,
    anchored_runtime_entities: List[Dict[str, Any]],
    anchored_bindings: List[Dict[str, Any]],
    rigid_specs: List[Dict[str, Any]],
    rigid_pos: np.ndarray,
    stiffness: float,
    rigid_fixed: bool = True,
):
    if articulated_ent is None or not anchored_runtime_entities or not anchored_bindings:
        return []

    import torch

    pid_to_binding = {int(b["anchored_pid"]): b for b in anchored_bindings}
    rigid_pid_to_spec = {int(spec["pid"]): spec for spec in rigid_specs}
    applied = []

    for rec in anchored_runtime_entities:
        spec = rec["spec"]
        ent = rec["entity"]
        pid = int(spec["pid"])
        binding = pid_to_binding.get(pid)
        rigid_spec = rigid_pid_to_spec.get(int(binding["rigid_pid"])) if binding is not None else None
        if binding is None or rigid_spec is None:
            continue

        try:
            link = articulated_ent.get_link(binding["link_name"])
        except Exception:
            continue

        poss = ent.get_particles_pos()
        if poss.ndim == 3:
            poss = poss[0]
        if poss.shape[0] == 0:
            continue

        bbox_min = np.asarray(rigid_spec.get("bounds_min", [0.0, 0.0, 0.0]), dtype=np.float64) + rigid_pos
        bbox_max = np.asarray(rigid_spec.get("bounds_max", [0.0, 0.0, 0.0]), dtype=np.float64) + rigid_pos
        bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)

        runtime_bounds_min = np.asarray(spec.get("runtime_bounds_min", spec.get("bounds_min", bbox_min)), dtype=np.float64) + rigid_pos
        runtime_bounds_max = np.asarray(spec.get("runtime_bounds_max", spec.get("bounds_max", bbox_max)), dtype=np.float64) + rigid_pos
        soft_size = np.maximum(runtime_bounds_max - runtime_bounds_min, 1e-6)

        # Keep a thin deformable shell while locking the bulk of the cushion to the frame.
        shell_thickness = float(
            np.clip(
                min(np.min(soft_size) * 0.35, np.max(soft_size) * 0.12),
                0.018,
                0.035,
            )
        )
        anchor_margin = np.minimum(
            np.full(3, shell_thickness, dtype=np.float64),
            0.45 * bbox_size,
        )

        bbox_min_t = torch.as_tensor(bbox_min - anchor_margin, dtype=poss.dtype, device=poss.device)
        bbox_max_t = torch.as_tensor(bbox_max + anchor_margin, dtype=poss.dtype, device=poss.device)
        clamped = torch.minimum(torch.maximum(poss, bbox_min_t), bbox_max_t)
        dist_to_box = torch.linalg.norm(poss - clamped, dim=-1)

        near_box_mask = dist_to_box <= shell_thickness

        target_anchor_frac = 0.72 if str(spec.get("assembly_role", "")) == "anchored_soft" else 0.5
        min_free_frac = 0.22
        target_anchor_count = int(np.clip(round(float(poss.shape[0]) * target_anchor_frac), 1, poss.shape[0]))
        max_anchor_count = int(np.clip(round(float(poss.shape[0]) * (1.0 - min_free_frac)), 1, poss.shape[0]))
        target_anchor_count = min(target_anchor_count, max_anchor_count)
        min_anchor_count = min(max(96, int(0.35 * poss.shape[0])), poss.shape[0])

        current_count = int(near_box_mask.sum().item())
        if current_count < min_anchor_count:
            k = min(max(min_anchor_count, target_anchor_count), poss.shape[0])
            nearest_idx = torch.topk(dist_to_box, k=k, largest=False).indices
            near_box_mask = torch.zeros_like(dist_to_box, dtype=torch.bool)
            near_box_mask[nearest_idx] = True
        elif current_count > target_anchor_count:
            nearest_idx = torch.topk(dist_to_box, k=target_anchor_count, largest=False).indices
            near_box_mask = torch.zeros_like(dist_to_box, dtype=torch.bool)
            near_box_mask[nearest_idx] = True

        if rigid_fixed:
            particles_mask = near_box_mask.unsqueeze(0)
            free_mask = (~near_box_mask)
            ent.set_free(free_mask)
            constraint_mode = "fixed_core"
        else:
            # Dynamic articulated objects should preserve their intra-object layout.
            # Bind the whole anchored_soft body to the matched rigid link so sofa
            # cushions and similar attached parts do not drift away from the frame.
            near_box_mask = torch.ones_like(near_box_mask, dtype=torch.bool)
            particles_mask = near_box_mask.unsqueeze(0)
            ent.set_particle_constraints(
                particles_mask=particles_mask,
                link_idx=int(link.idx),
                stiffness=float(max(stiffness, 5.0e4)),
            )
            constraint_mode = "rigid_follow"

        applied.append(
            {
                "pid": pid,
                "part_name": str(spec.get("part_name", f"part_{pid}")),
                "link_name": str(binding["link_name"]),
                "link_idx": int(link.idx),
                "n_constrained": int(near_box_mask.sum().item()),
                "n_total": int(poss.shape[0]),
                "shell_thickness": float(shell_thickness),
                "stiffness": float(stiffness),
                "mode": constraint_mode,
            }
        )

    return applied


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


def _entity_aabb_numpy(entity: Any) -> Optional[np.ndarray]:
    if entity is None or not hasattr(entity, "get_AABB"):
        return None
    try:
        aabb = entity.get_AABB()
    except Exception:
        return None
    try:
        if hasattr(aabb, "detach"):
            aabb = aabb.detach().cpu().numpy()
        else:
            aabb = np.asarray(aabb)
        return np.asarray(aabb, dtype=np.float64).reshape(2, 3)
    except Exception:
        return None


def _merge_aabbs(aabbs: Sequence[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    valid = []
    for aabb in aabbs:
        if aabb is None:
            continue
        arr = np.asarray(aabb, dtype=np.float64).reshape(2, 3)
        valid.append(arr)
    if not valid:
        return None
    mins = np.min(np.stack([arr[0] for arr in valid], axis=0), axis=0)
    maxs = np.max(np.stack([arr[1] for arr in valid], axis=0), axis=0)
    return np.stack([mins, maxs], axis=0)


def _aabb_overlaps(aabb_a: Optional[np.ndarray], aabb_b: Optional[np.ndarray], clearance: float = 0.0) -> bool:
    if aabb_a is None or aabb_b is None:
        return False
    a = np.asarray(aabb_a, dtype=np.float64).reshape(2, 3)
    b = np.asarray(aabb_b, dtype=np.float64).reshape(2, 3)
    overlap_min = np.maximum(a[0] - float(clearance), b[0] - float(clearance))
    overlap_max = np.minimum(a[1] + float(clearance), b[1] + float(clearance))
    return bool(np.all(overlap_max > overlap_min))


def _compute_aabb_separation_delta(
    moving_aabb: Optional[np.ndarray],
    fixed_aabb: Optional[np.ndarray],
    *,
    clearance: float = 0.0,
    prefer_planar: bool = True,
    prefer_upward: bool = False,
) -> np.ndarray:
    if moving_aabb is None or fixed_aabb is None:
        return np.zeros(3, dtype=np.float64)
    moving = np.asarray(moving_aabb, dtype=np.float64).reshape(2, 3)
    fixed = np.asarray(fixed_aabb, dtype=np.float64).reshape(2, 3)
    overlap_min = np.maximum(moving[0] - float(clearance), fixed[0] - float(clearance))
    overlap_max = np.minimum(moving[1] + float(clearance), fixed[1] + float(clearance))
    if np.any(overlap_max <= overlap_min):
        return np.zeros(3, dtype=np.float64)

    overlap = overlap_max - overlap_min
    moving_center = 0.5 * (moving[0] + moving[1])
    fixed_center = 0.5 * (fixed[0] + fixed[1])

    axes = list(range(3))
    if prefer_upward:
        axes.sort(key=lambda axis: (0 if axis == 2 else 1, float(overlap[axis])))
    elif prefer_planar:
        axes.sort(key=lambda axis: (1 if axis == 2 else 0, float(overlap[axis])))
    else:
        axes.sort(key=lambda axis: float(overlap[axis]))

    for axis in axes:
        shift = float(overlap[axis] + clearance)
        if shift <= 0.0:
            continue
        if axis == 2 and prefer_upward:
            sign = 1.0
        else:
            sign = 1.0 if moving_center[axis] >= fixed_center[axis] else -1.0
        if abs(moving_center[axis] - fixed_center[axis]) <= 1e-8:
            sign = 1.0
        if axis == 2 and prefer_planar and moving[0, 2] <= fixed[1, 2] + float(clearance):
            sign = 1.0
        delta = np.zeros(3, dtype=np.float64)
        delta[axis] = sign * shift
        return delta
    return np.zeros(3, dtype=np.float64)


def _resolve_initial_custom_object_overlaps(
    *,
    custom_runtime_objects: Sequence[Dict[str, Any]],
    anchored_aabbs: Sequence[Optional[np.ndarray]],
    clearance: float = 0.01,
    max_passes: int = 24,
) -> List[Dict[str, Any]]:
    if not custom_runtime_objects:
        return []

    adjustments: List[Dict[str, Any]] = []
    placed_aabbs: List[np.ndarray] = []
    anchor_aabbs = [
        np.asarray(aabb, dtype=np.float64).reshape(2, 3)
        for aabb in anchored_aabbs
        if aabb is not None
    ]

    for custom_rec in custom_runtime_objects:
        ent = custom_rec.get("entity")
        if ent is None or not hasattr(ent, "set_pos"):
            continue
        total_delta = np.zeros(3, dtype=np.float64)
        moved = False
        for _ in range(max(1, int(max_passes))):
            current_aabb = _entity_aabb_numpy(ent)
            if current_aabb is None:
                break

            overlap_targets: List[np.ndarray] = []
            overlap_targets.extend(anchor_aabbs)
            overlap_targets.extend(placed_aabbs)
            candidate_delta = np.zeros(3, dtype=np.float64)
            for other_aabb in overlap_targets:
                if not _aabb_overlaps(current_aabb, other_aabb, clearance=clearance):
                    continue
                candidate_delta += _compute_aabb_separation_delta(
                    current_aabb,
                    other_aabb,
                    clearance=clearance,
                    prefer_planar=True,
                )

            if not np.any(np.abs(candidate_delta) > 1e-9):
                break

            start_pos = np.asarray(custom_rec.get("start_pos", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
            new_pos = start_pos + candidate_delta
            try:
                ent.set_pos(tuple(new_pos.tolist()))
            except Exception:
                break
            custom_rec["start_pos"] = new_pos.copy()
            total_delta += candidate_delta
            moved = True

        final_aabb = _entity_aabb_numpy(ent)
        if final_aabb is not None:
            placed_aabbs.append(final_aabb)
        if moved:
            adjustments.append(
                {
                    "custom_object_id": str(custom_rec.get("custom_object_id", "custom_object")),
                    "delta": total_delta.astype(np.float64).tolist(),
                    "final_pos": np.asarray(custom_rec.get("start_pos", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3).tolist(),
                }
            )

    return adjustments


def _resolve_initial_nonrigid_component_overlaps(
    *,
    nonrigid_runtime_records: Sequence[Dict[str, Any]],
    anchored_aabbs: Sequence[Optional[np.ndarray]],
    clearance: float = 0.006,
    max_passes: int = 24,
) -> List[Dict[str, Any]]:
    if not nonrigid_runtime_records:
        return []

    adjustments: List[Dict[str, Any]] = []
    placed_aabbs: List[np.ndarray] = []
    anchor_aabbs = [
        np.asarray(aabb, dtype=np.float64).reshape(2, 3)
        for aabb in anchored_aabbs
        if aabb is not None
    ]

    ordered_records = sorted(
        list(nonrigid_runtime_records),
        key=lambda rec: (
            0 if bool(rec.get("prefer_upward_clearance", False)) else 1,
            int(rec.get("pid", -1)),
        ),
    )

    for rec in ordered_records:
        ent = rec.get("entity")
        if ent is None or not hasattr(ent, "set_pos"):
            continue

        prefer_upward = bool(rec.get("prefer_upward_clearance", False))
        total_delta = np.zeros(3, dtype=np.float64)
        moved = False
        for _ in range(max(1, int(max_passes))):
            current_aabb = _entity_aabb_numpy(ent)
            if current_aabb is None:
                break

            overlap_targets: List[np.ndarray] = []
            overlap_targets.extend(anchor_aabbs)
            overlap_targets.extend(placed_aabbs)
            candidate_delta = np.zeros(3, dtype=np.float64)
            for other_aabb in overlap_targets:
                if not _aabb_overlaps(current_aabb, other_aabb, clearance=clearance):
                    continue
                candidate_delta += _compute_aabb_separation_delta(
                    current_aabb,
                    other_aabb,
                    clearance=clearance,
                    prefer_planar=not prefer_upward,
                    prefer_upward=prefer_upward,
                )

            if not np.any(np.abs(candidate_delta) > 1e-9):
                break

            base_pos = np.asarray(rec.get("base_pos", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3)
            new_pos = base_pos + candidate_delta
            try:
                ent.set_pos(tuple(new_pos.tolist()))
            except Exception:
                break
            rec["base_pos"] = new_pos.copy()
            total_delta += candidate_delta
            moved = True

        final_aabb = _entity_aabb_numpy(ent)
        if final_aabb is not None:
            placed_aabbs.append(final_aabb)
        if moved:
            adjustments.append(
                {
                    "pid": int(rec.get("pid", -1)),
                    "part_name": str(rec.get("part_name", "nonrigid_part")),
                    "delta": total_delta.astype(np.float64).tolist(),
                    "final_pos": np.asarray(rec.get("base_pos", np.zeros(3, dtype=np.float64)), dtype=np.float64).reshape(3).tolist(),
                    "prefer_upward_clearance": prefer_upward,
                }
            )

    return adjustments


def _entity_particle_settle_summary(entity: Any) -> Optional[Dict[str, Any]]:
    if entity is None or not hasattr(entity, "get_particles_pos"):
        return None
    try:
        poss = entity.get_particles_pos()
    except Exception:
        return None


def _entity_particles_numpy(entity: Any, kind: str = "pos") -> Optional[np.ndarray]:
    getter_name = "get_particles_pos" if kind == "pos" else "get_particles_vel"
    if entity is None or not hasattr(entity, getter_name):
        return None
    try:
        vals = getattr(entity, getter_name)()
    except Exception:
        return None
    try:
        if hasattr(vals, "detach"):
            vals = vals.detach().cpu().numpy()
        else:
            vals = np.asarray(vals)
        arr = np.asarray(vals, dtype=np.float64)
    except Exception:
        return None
    if arr.size == 0:
        return arr.reshape(0, 3)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            return arr
        return arr.reshape(-1, 3)
    if arr.ndim >= 2 and arr.shape[-1] == 3:
        return arr.reshape(-1, 3)
    if arr.size % 3 == 0:
        return arr.reshape(-1, 3)
    return arr


def _collect_case_physics_state(
    *,
    prepared: PreparedObject,
    articulated_ent: Any,
    nonrigid_runtime_entities: Sequence[Any],
    part_specs: Sequence[Dict[str, Any]],
    custom_runtime_objects: Sequence[Dict[str, Any]],
    gravity_vec: Sequence[float],
    runtime_case_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    object_ids: List[str] = []
    object_names: List[str] = []
    track_object_ids: List[int] = []
    seg_ids: List[int] = []
    object_types: List[str] = []
    object_sources: List[str] = []
    object_role_hints: List[Optional[str]] = []
    object_motion_type_hints: List[Optional[str]] = []
    object_motion_group_hints: List[Optional[str]] = []
    com_pos: List[np.ndarray] = []
    orientation_quat: List[np.ndarray] = []
    linear_vel: List[np.ndarray] = []
    angular_vel: List[np.ndarray] = []
    kinetic_total = 0.0
    potential_total = 0.0
    kinetic_trans_total = 0.0
    kinetic_rot_total = 0.0
    gravity_vec_np = np.asarray(gravity_vec, dtype=np.float64).reshape(3)
    segmentation_entities: List[Any] = []
    object_aabbs: List[Optional[np.ndarray]] = []
    environment_contacts: List[Dict[str, Any]] = []

    if articulated_ent is not None:
        rigid_runtime_ent = articulated_ent.part_rigid if hasattr(articulated_ent, "part_rigid") else articulated_ent
        rigid_snap = rigid_entity_kinematic_snapshot(rigid_runtime_ent, gravity=gravity_vec_np)
        rigid_mass = float(rigid_runtime_ent.get_mass())
        rigid_potential = -rigid_mass * float(np.dot(gravity_vec_np, rigid_snap.com_pos))
        object_ids.append(f"main::{prepared.object_id}")
        object_names.append(str(prepared.object_id))
        track_object_ids.append(0)
        seg_ids.append(1)
        object_types.append("rigid_assembly")
        object_sources.append("physxnet_main")
        case_cfg_local = dict(runtime_case_cfg or {})
        object_role_hints.append(case_cfg_local.get("main_object_role_hint", None))
        object_motion_type_hints.append(case_cfg_local.get("main_object_motion_type_hint", None))
        object_motion_group_hints.append(case_cfg_local.get("main_object_motion_group_hint", None))
        com_pos.append(rigid_snap.com_pos)
        try:
            quat = np.asarray(_to_numpy(rigid_runtime_ent.get_quat()), dtype=np.float64).reshape(4)
        except Exception:
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        orientation_quat.append(quat.astype(np.float32))
        linear_vel.append(rigid_snap.linear_vel)
        angular_vel.append(rigid_snap.angular_vel)
        kinetic_linear, kinetic_rot, _ = rigid_entity_energy_components(rigid_runtime_ent, gravity=gravity_vec_np)
        kinetic_total += float(rigid_snap.kinetic)
        kinetic_trans_total += float(kinetic_linear)
        kinetic_rot_total += float(kinetic_rot)
        potential_total += rigid_potential
        segmentation_entities.append(rigid_runtime_ent)
        object_aabbs.append(_entity_aabb_numpy(rigid_runtime_ent))

    spec_by_pid = {int(spec.get("pid", -1)): spec for spec in part_specs}
    next_object_id = len(track_object_ids)
    for ent in nonrigid_runtime_entities:
        pid = None
        try:
            pid = int(getattr(ent, "_physics_pid", getattr(ent, "physics_pid", -1)))
        except Exception:
            pid = -1
        spec = spec_by_pid.get(int(pid), {})
        snap = particle_entity_kinematic_snapshot(ent, gravity=gravity_vec_np)
        particle_mass = 0.0
        if hasattr(ent, "solver") and hasattr(ent.solver, "particles_info") and hasattr(ent.solver.particles_info, "mass"):
            try:
                particle_start = int(getattr(ent, "_particle_start", getattr(ent, "particle_start", 0)))
                n_particles = int(getattr(ent, "n_particles", 0))
                mass_field = ent.solver.particles_info.mass
                mass_arr = _to_numpy(mass_field.to_numpy() if hasattr(mass_field, "to_numpy") else mass_field)
                particle_mass = float(np.asarray(mass_arr[particle_start : particle_start + n_particles], dtype=np.float64).sum())
            except Exception:
                particle_mass = 0.0
        potential = -particle_mass * float(np.dot(gravity_vec_np, snap.com_pos))
        object_ids.append(f"soft::{prepared.object_id}::pid_{int(pid):03d}")
        object_names.append(f"{prepared.object_id}::pid_{int(pid):03d}")
        track_object_ids.append(int(next_object_id))
        seg_ids.append(int(next_object_id + 1))
        object_types.append(str(spec.get("assembly_role", "soft_entity")))
        object_sources.append("physxnet_soft")
        object_role_hints.append(None)
        object_motion_type_hints.append(None)
        object_motion_group_hints.append(None)
        com_pos.append(snap.com_pos)
        orientation_quat.append(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        linear_vel.append(snap.linear_vel)
        angular_vel.append(snap.angular_vel)
        kinetic_total += float(snap.kinetic)
        kinetic_trans_total += float(snap.kinetic)
        potential_total += potential
        segmentation_entities.append(ent)
        object_aabbs.append(_entity_aabb_numpy(ent))
        next_object_id += 1

    for custom_rec in custom_runtime_objects:
        ent = custom_rec.get("entity")
        if ent is None:
            continue
        custom_id = str(custom_rec.get("custom_object_id", "custom_object"))
        if _entity_particles_numpy(ent, kind="pos") is not None:
            snap = particle_entity_kinematic_snapshot(ent, gravity=gravity_vec_np)
            particle_mass = 0.0
            if hasattr(ent, "solver") and hasattr(ent.solver, "particles_info") and hasattr(ent.solver.particles_info, "mass"):
                try:
                    particle_start = int(getattr(ent, "_particle_start", getattr(ent, "particle_start", 0)))
                    n_particles = int(getattr(ent, "n_particles", 0))
                    mass_field = ent.solver.particles_info.mass
                    mass_arr = _to_numpy(mass_field.to_numpy() if hasattr(mass_field, "to_numpy") else mass_field)
                    particle_mass = float(np.asarray(mass_arr[particle_start : particle_start + n_particles], dtype=np.float64).sum())
                except Exception:
                    particle_mass = 0.0
            potential = -particle_mass * float(np.dot(gravity_vec_np, snap.com_pos))
            object_type = "custom_particle"
        else:
            snap = rigid_entity_kinematic_snapshot(ent, gravity=gravity_vec_np)
            try:
                particle_mass = float(ent.get_mass())
            except Exception:
                particle_mass = 0.0
            potential = -particle_mass * float(np.dot(gravity_vec_np, snap.com_pos))
            object_type = "custom_rigid"
        object_ids.append(custom_id)
        object_names.append(str(custom_rec.get("source_display_id", custom_id)))
        track_object_ids.append(int(next_object_id))
        seg_ids.append(int(next_object_id + 1))
        object_types.append(object_type)
        object_sources.append(str(custom_rec.get("source_tag", "custom_object")))
        object_role_hints.append(None if custom_rec.get("role_hint", None) is None else str(custom_rec.get("role_hint")))
        object_motion_type_hints.append(None if custom_rec.get("motion_type_hint", None) is None else str(custom_rec.get("motion_type_hint")))
        object_motion_group_hints.append(None if custom_rec.get("motion_group_hint", None) is None else str(custom_rec.get("motion_group_hint")))
        com_pos.append(snap.com_pos)
        if hasattr(ent, "get_quat"):
            try:
                quat = np.asarray(_to_numpy(ent.get_quat()), dtype=np.float64).reshape(4)
            except Exception:
                quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        else:
            quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        orientation_quat.append(quat.astype(np.float32))
        linear_vel.append(snap.linear_vel)
        angular_vel.append(snap.angular_vel)
        kinetic_total += float(snap.kinetic)
        kinetic_trans_total += float(snap.kinetic)
        potential_total += potential
        segmentation_entities.append(ent)
        object_aabbs.append(_entity_aabb_numpy(ent))
        next_object_id += 1

    if com_pos:
        com_pos_arr = np.stack(com_pos, axis=0).astype(np.float32)
        orientation_arr = np.stack(orientation_quat, axis=0).astype(np.float32)
        linear_vel_arr = np.stack(linear_vel, axis=0).astype(np.float32)
        angular_vel_arr = np.stack(angular_vel, axis=0).astype(np.float32)
    else:
        com_pos_arr = np.zeros((0, 3), dtype=np.float32)
        orientation_arr = np.zeros((0, 4), dtype=np.float32)
        linear_vel_arr = np.zeros((0, 3), dtype=np.float32)
        angular_vel_arr = np.zeros((0, 3), dtype=np.float32)

    for obj_idx, aabb in enumerate(object_aabbs):
        if aabb is None:
            continue
        aabb_arr = np.asarray(aabb, dtype=np.float64).reshape(2, 3)
        if float(aabb_arr[0, 2]) <= 0.01:
            environment_contacts.append(
                {
                    "object_idx": int(obj_idx),
                    "object_id": int(track_object_ids[obj_idx]),
                    "environment_name": "ground",
                    "environment_id": int(ENVIRONMENT_SPECIAL_IDS["ground"]),
                    "impulse_peak": 0.0,
                }
            )

    return {
        "object_ids": object_ids,
        "track_object_ids": track_object_ids,
        "object_names": object_names,
        "seg_ids": seg_ids,
        "object_types": object_types,
        "object_sources": object_sources,
        "object_role_hints": object_role_hints,
        "object_motion_type_hints": object_motion_type_hints,
        "object_motion_group_hints": object_motion_group_hints,
        "com_pos": com_pos_arr,
        "orientation_quat": orientation_arr,
        "linear_vel": linear_vel_arr,
        "angular_vel": angular_vel_arr,
        "kinetic_energy": np.float32(kinetic_total),
        "potential_energy": np.float32(potential_total),
        "kinetic_trans": np.float32(kinetic_trans_total),
        "kinetic_rot": np.float32(kinetic_rot_total),
        "potential_gravity": np.float32(potential_total),
        "total_energy": np.float32(kinetic_total + potential_total),
        "segmentation_entities": segmentation_entities,
        "object_aabbs": object_aabbs,
        "environment_contacts": environment_contacts,
    }


def _detect_collision_flag_from_states(
    prev_state: Optional[Dict[str, Any]],
    curr_state: Dict[str, Any],
    *,
    linear_delta_threshold: float = 1.0,
    angular_delta_threshold: float = 4.0,
    ground_height_threshold: float = 0.03,
    pair_distance_threshold: float = 0.10,
) -> bool:
    if prev_state is None:
        return False
    prev_lin = np.asarray(prev_state.get("linear_vel", []), dtype=np.float64)
    curr_lin = np.asarray(curr_state.get("linear_vel", []), dtype=np.float64)
    prev_ang = np.asarray(prev_state.get("angular_vel", []), dtype=np.float64)
    curr_ang = np.asarray(curr_state.get("angular_vel", []), dtype=np.float64)
    curr_pos = np.asarray(curr_state.get("com_pos", []), dtype=np.float64)
    if prev_lin.shape != curr_lin.shape or prev_ang.shape != curr_ang.shape or curr_pos.ndim != 2:
        return False
    if curr_lin.size == 0:
        return False
    lin_jump = np.linalg.norm(curr_lin - prev_lin, axis=1)
    ang_jump = np.linalg.norm(curr_ang - prev_ang, axis=1)
    jump_mask = (lin_jump > float(linear_delta_threshold)) | (ang_jump > float(angular_delta_threshold))
    if not np.any(jump_mask):
        return False
    if np.any(curr_pos[:, 2] < float(ground_height_threshold)):
        return True
    if curr_pos.shape[0] >= 2:
        diff = curr_pos[:, None, :] - curr_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        dist = dist + np.eye(dist.shape[0], dtype=np.float64) * 1e6
        if float(np.min(dist)) < float(pair_distance_threshold):
            return True
    return False
    try:
        if hasattr(vals, "detach"):
            vals = vals.detach().cpu().numpy()
        else:
            vals = np.asarray(vals)
        vals = np.asarray(vals, dtype=np.float64)
        if vals.ndim == 3:
            vals = vals[0]
        return vals.reshape(-1, 3)
    except Exception:
        return None
    try:
        if hasattr(poss, "detach"):
            poss = poss.detach().cpu().numpy()
        else:
            poss = np.asarray(poss)
        poss = np.asarray(poss, dtype=np.float64)
        if poss.ndim == 3:
            poss = poss[0]
        poss = poss.reshape(-1, 3)
        if poss.shape[0] == 0:
            return None

        return {
            "center": np.mean(poss, axis=0).astype(np.float64),
            "z_quantiles": np.quantile(poss[:, 2], [0.05, 0.5, 0.95]).astype(np.float64),
            "aabb": np.asarray([np.min(poss, axis=0), np.max(poss, axis=0)], dtype=np.float64),
        }
    except Exception:
        return None


def path_map(spec_mesh_path):
    objj = int(spec_mesh_path.split("_")[-1].split(".")[0])

    objid = spec_mesh_path.split("/")[-3]
    path = f"/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/partseg/{objid}/objs/{str(objj)}.obj"
    print(path)
    return path
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
    case_cfg: Optional[Dict[str, Any]] = None,
) -> str:
    import genesis as gs

    def _coerce_velocity6(value: Any, default: Optional[Sequence[float]] = None) -> np.ndarray:
        base = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] if default is None else default
        arr = np.asarray(base if value is None else value, dtype=np.float64).reshape(-1)
        if arr.size == 3:
            arr = np.concatenate([arr, np.zeros(3, dtype=np.float64)], axis=0)
        if arr.size != 6:
            raise ValueError(f"Expected velocity with 3 or 6 values, got shape={arr.shape}")
        return arr.astype(np.float64)

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

    obj_dir = Path(prepared.output_dir)
    metadata = json.loads((obj_dir / "meta" / "metadata.json").read_text(encoding="utf-8"))

    rigid_material_cfg = _default_entity_rigid_material(metadata, default_friction=default_friction)

    bbox_min = np.asarray(metadata["object_bbox_min"], dtype=np.float64)
    bbox_max = np.asarray(metadata["object_bbox_max"], dtype=np.float64)
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
    placed_pos = np.array([0.0, 0.0, float(metadata["grounding_offset_z"]) + 0.002], dtype=np.float64)
    runtime_case_cfg = dict(case_cfg or {})
    case_name = str(runtime_case_cfg.get("case_name", "case000"))
    case_id = int(runtime_case_cfg.get("case_id", runtime_case_cfg.get("case_index", 0)))
    case_variant_index = int(runtime_case_cfg.get("case_variant_index", runtime_case_cfg.get("case_index", case_id)))
    scene_label = str(runtime_case_cfg.get("scene_label", case_name))
    case_seed_for_runtime = int(runtime_case_cfg.get("seed", 20260414))
    counterfactual_meta = dict(runtime_case_cfg.get("counterfactual", {}) or {})
    simple_case_resample_index = int(getattr(args, "simple_case_resample_index", 0) or 0)
    placed_pos = placed_pos + np.asarray(runtime_case_cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    object_euler_deg = np.asarray(runtime_case_cfg.get("object_euler_deg", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
    runtime_object_fixed = bool(runtime_case_cfg.get("object_fixed", object_fixed))
    object_entry_linear_velocity = np.asarray(
        runtime_case_cfg.get("entry_linear_velocity", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    object_entry_angular_velocity = np.asarray(
        runtime_case_cfg.get("entry_angular_velocity", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    )
    apply_object_entry_velocity = bool(runtime_case_cfg.get("use_entry_motion", False))
    rigid_restitution_override = runtime_case_cfg.get("rigid_restitution_override", None)
    if rigid_restitution_override is not None:
        rigid_material_cfg["restitution"] = float(np.clip(float(rigid_restitution_override), 0.0, 1.2))
    preview_output_root = getattr(args, "preview_output_root", None)
    if preview_output_root:
        preview_dir = Path(str(preview_output_root))
    else:
        preview_dir = Path(prepared.output_dir) / "scene_preview"
    ensure_dir(preview_dir)
    runtime_mesh_dir = preview_dir / "runtime_soft_meshes"
    anchored_overlap_scale_boost = float(getattr(args, "anchored_overlap_scale_boost", 1.0) or 1.0)
    gravity_z = float(_case_cfg_or_default(runtime_case_cfg, "gravity_z_override", getattr(args, "gravity_z", -9.81)))
    runtime_striker_speed = float(_case_cfg_or_default(runtime_case_cfg, "striker_speed_override", striker_speed))
    runtime_main_object_scale = 1.0
    runtime_striker_radius_mult = 1.0
    auto_visibility_scale_info: Dict[str, Any] = {
        "enabled": False,
        "min_projected_bbox_area_px": float(getattr(args, "min_projected_bbox_area_px", 0.0) or 0.0),
        "max_auto_scale_up_mult": float(getattr(args, "max_auto_scale_up_mult", 1.0) or 1.0),
        "main_object_scale": 1.0,
        "default_striker_radius_mult": 1.0,
        "custom_objects": [],
    }
    custom_object_cfgs = copy.deepcopy(list(runtime_case_cfg.get("custom_objects", []) or []))
    runtime_case_cfg["custom_objects"] = custom_object_cfgs
    custom_object_base_sizes: List[float] = []
    for custom_cfg in custom_object_cfgs:
        mesh_path = str(custom_cfg.get("mesh_path", "") or "")
        if mesh_path:
            custom_object_base_sizes.append(float(max(1e-6, custom_cfg.get("scale", 1.0))))
        else:
            custom_object_base_sizes.append(float(max(0.01, custom_cfg.get("radius", striker_radius))))
    prepared_has_liquid = any(
        str(rec.get("material_ctor", "")) in {"gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid"}
        for rec in (metadata.get("parts_physical", {}) or {}).values()
        if isinstance(rec, dict)
    )
    min_projected_bbox_area_px = float(getattr(args, "min_projected_bbox_area_px", 2500.0) or 0.0)
    max_auto_scale_up_mult = max(1.0, float(getattr(args, "max_auto_scale_up_mult", 2.5) or 2.5))
    if (
        min_projected_bbox_area_px > 1.0
        and not prepared_has_liquid
        and str(getattr(args, "simulator_mode", "rigid")).strip().lower() == "rigid"
    ):
        mesh_bounds_cache: Dict[str, Optional[Dict[str, Any]]] = {}
        base_grounding_offset_z = float(metadata["grounding_offset_z"])
        placed_pos_offset = np.asarray(runtime_case_cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
        disable_default_striker_for_scale = bool(getattr(args, "disable_striker", False) or getattr(args, "debug_spread_soft_parts", False))
        disable_default_striker_for_scale = disable_default_striker_for_scale or bool(runtime_case_cfg.get("disable_default_striker", False))
        for _iter_idx in range(3):
            scaled_bbox_min = bbox_min * runtime_main_object_scale
            scaled_bbox_max = bbox_max * runtime_main_object_scale
            scaled_bbox_center = 0.5 * (scaled_bbox_min + scaled_bbox_max)
            placed_pos_est = np.array(
                [0.0, 0.0, base_grounding_offset_z * runtime_main_object_scale + 0.002],
                dtype=np.float64,
            ) + placed_pos_offset
            custom_start_positions_est = [
                placed_pos_est + np.asarray(cfg.get("spawn_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
                for cfg in custom_object_cfgs
            ]
            camera_cfg_est = _estimate_preview_camera_cfg_for_visibility(
                scene_label=scene_label,
                placed_pos=placed_pos_est,
                bbox_min=scaled_bbox_min,
                bbox_max=scaled_bbox_max,
                camera_distance_mult=float(getattr(args, "camera_distance_mult", 1.0) or 1.0),
                custom_start_positions=custom_start_positions_est,
            )
            cam_intrinsics_est = camera_intrinsics_dict(
                camera=None,
                fallback_res=tuple(EXPORT_CAMERA_RESOLUTION),
                fallback_fov_deg=float(camera_cfg_est["fov"]),
            )
            updates_applied = False

            main_area_px = _projected_bbox_area_from_local_bounds(
                scaled_bbox_min,
                scaled_bbox_max,
                pos_world=placed_pos_est,
                euler_deg=object_euler_deg,
                camera_cfg=camera_cfg_est,
                cam_intrinsics=cam_intrinsics_est,
                image_res=tuple(EXPORT_CAMERA_RESOLUTION),
            )
            main_required_mult = 1.0 if main_area_px >= min_projected_bbox_area_px else math.sqrt(
                min_projected_bbox_area_px / max(main_area_px, 1.0)
            )
            main_allowed_mult = max(1.0, max_auto_scale_up_mult / max(runtime_main_object_scale, 1e-8))
            main_applied_mult = min(main_required_mult, main_allowed_mult)
            if main_applied_mult > 1.03:
                runtime_main_object_scale *= main_applied_mult
                updates_applied = True
            auto_visibility_scale_info["main_object_projected_area_px"] = float(main_area_px)

            if not disable_default_striker_for_scale:
                striker_radius_est = float(striker_radius) * runtime_striker_radius_mult
                striker_clearance_est = max(0.01, 0.20 * striker_radius_est)
                striker_start_est = np.array(
                    [
                        float(placed_pos_est[0] + scaled_bbox_max[0] + striker_radius_est + striker_clearance_est + args.ball_posx),
                        float(placed_pos_est[1] + scaled_bbox_center[1]),
                        float(placed_pos_est[2] + scaled_bbox_min[2] + 0.4 * float(np.max(scaled_bbox_max - scaled_bbox_min)) + striker_radius_est + striker_clearance_est),
                    ],
                    dtype=np.float64,
                )
                striker_area_px = _projected_bbox_area_from_local_bounds(
                    [-striker_radius_est, -striker_radius_est, -striker_radius_est],
                    [striker_radius_est, striker_radius_est, striker_radius_est],
                    pos_world=striker_start_est,
                    euler_deg=[0.0, 0.0, 0.0],
                    camera_cfg=camera_cfg_est,
                    cam_intrinsics=cam_intrinsics_est,
                    image_res=tuple(EXPORT_CAMERA_RESOLUTION),
                )
                striker_required_mult = 1.0 if striker_area_px >= min_projected_bbox_area_px else math.sqrt(
                    min_projected_bbox_area_px / max(striker_area_px, 1.0)
                )
                striker_allowed_mult = max(1.0, max_auto_scale_up_mult / max(runtime_striker_radius_mult, 1e-8))
                striker_applied_mult = min(striker_required_mult, striker_allowed_mult)
                if striker_applied_mult > 1.03:
                    runtime_striker_radius_mult *= striker_applied_mult
                    updates_applied = True
                auto_visibility_scale_info["default_striker_projected_area_px"] = float(striker_area_px)

            custom_scale_records: List[Dict[str, Any]] = []
            for custom_idx, custom_cfg in enumerate(custom_object_cfgs):
                custom_start_pos = placed_pos_est + np.asarray(custom_cfg.get("spawn_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
                custom_euler = np.asarray(custom_cfg.get("euler_deg", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
                mesh_path = str(custom_cfg.get("mesh_path", "") or "")
                base_size = float(custom_object_base_sizes[custom_idx]) if custom_idx < len(custom_object_base_sizes) else 1.0
                if mesh_path and Path(mesh_path).exists():
                    if mesh_path not in mesh_bounds_cache:
                        mesh_bounds_cache[mesh_path] = _mesh_bounds_info(Path(mesh_path), scale=1.0)
                    bounds_info = mesh_bounds_cache.get(mesh_path)
                    if bounds_info is None:
                        custom_scale_records.append({"custom_object_id": str(custom_cfg.get("custom_object_id", custom_idx)), "projected_area_px": 0.0})
                        continue
                    local_bounds_min = np.asarray(bounds_info["bounds_min"], dtype=np.float64) * float(custom_cfg.get("scale", 1.0))
                    local_bounds_max = np.asarray(bounds_info["bounds_max"], dtype=np.float64) * float(custom_cfg.get("scale", 1.0))
                    area_px = _projected_bbox_area_from_local_bounds(
                        local_bounds_min,
                        local_bounds_max,
                        pos_world=custom_start_pos,
                        euler_deg=custom_euler,
                        camera_cfg=camera_cfg_est,
                        cam_intrinsics=cam_intrinsics_est,
                        image_res=tuple(EXPORT_CAMERA_RESOLUTION),
                    )
                    required_mult = 1.0 if area_px >= min_projected_bbox_area_px else math.sqrt(
                        min_projected_bbox_area_px / max(area_px, 1.0)
                    )
                    current_scale = float(max(1e-6, custom_cfg.get("scale", 1.0)))
                    allowed_mult = max(1.0, (max_auto_scale_up_mult * max(base_size, 1e-8)) / current_scale)
                    applied_mult = min(required_mult, allowed_mult)
                    if applied_mult > 1.03:
                        custom_cfg["scale"] = float(current_scale * applied_mult)
                        updates_applied = True
                    custom_scale_records.append(
                        {
                            "custom_object_id": str(custom_cfg.get("custom_object_id", custom_idx)),
                            "projected_area_px": float(area_px),
                            "scale": float(custom_cfg.get("scale", current_scale)),
                        }
                    )
                else:
                    radius = float(max(0.01, custom_cfg.get("radius", striker_radius)))
                    area_px = _projected_bbox_area_from_local_bounds(
                        [-radius, -radius, -radius],
                        [radius, radius, radius],
                        pos_world=custom_start_pos,
                        euler_deg=[0.0, 0.0, 0.0],
                        camera_cfg=camera_cfg_est,
                        cam_intrinsics=cam_intrinsics_est,
                        image_res=tuple(EXPORT_CAMERA_RESOLUTION),
                    )
                    required_mult = 1.0 if area_px >= min_projected_bbox_area_px else math.sqrt(
                        min_projected_bbox_area_px / max(area_px, 1.0)
                    )
                    allowed_mult = max(1.0, (max_auto_scale_up_mult * max(base_size, 1e-8)) / radius)
                    applied_mult = min(required_mult, allowed_mult)
                    if applied_mult > 1.03:
                        custom_cfg["radius"] = float(radius * applied_mult)
                        updates_applied = True
                    custom_scale_records.append(
                        {
                            "custom_object_id": str(custom_cfg.get("custom_object_id", custom_idx)),
                            "projected_area_px": float(area_px),
                            "radius": float(custom_cfg.get("radius", radius)),
                        }
                    )
            auto_visibility_scale_info["custom_objects"] = custom_scale_records
            if not updates_applied:
                break
        auto_visibility_scale_info["enabled"] = True
        auto_visibility_scale_info["main_object_scale"] = float(runtime_main_object_scale)
        auto_visibility_scale_info["default_striker_radius_mult"] = float(runtime_striker_radius_mult)

    bbox_min = bbox_min * runtime_main_object_scale
    bbox_max = bbox_max * runtime_main_object_scale
    bbox_center = 0.5 * (bbox_min + bbox_max)
    bbox_size = np.maximum(bbox_max - bbox_min, 1e-6)
    placed_pos = np.array([0.0, 0.0, float(metadata["grounding_offset_z"]) * runtime_main_object_scale + 0.002], dtype=np.float64)
    placed_pos = placed_pos + np.asarray(runtime_case_cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    print(
        f"🎬 run_case case={case_name} scene={scene_label} "
        f"fixed={runtime_object_fixed} moving={apply_object_entry_velocity} "
        f"gravity_z={gravity_z:.3f} offset={placed_pos.tolist()} euler_deg={object_euler_deg.tolist()} "
        f"main_scale={runtime_main_object_scale:.3f}"
    )
    if counterfactual_meta:
        print(
            f"🪞 counterfactual kind={counterfactual_meta.get('kind', 'unknown')} "
            f"parent={counterfactual_meta.get('parent_case_name', 'unknown')} "
            f"mode={counterfactual_meta.get('mode', 'unknown')}"
        )

    part_specs = _collect_part_specs(obj_dir=obj_dir, metadata=metadata)
    if abs(runtime_main_object_scale - 1.0) > 1e-8:
        for spec in part_specs:
            spec["scale"] = float(max(1e-6, float(spec.get("scale", 1.0)) * runtime_main_object_scale))
            _scale_bounds_record_inplace(spec, runtime_main_object_scale)
    part_pid_filter_raw = str(getattr(args, "part_pid_filter", "") or "").strip()
    prefer_existing_runtime_meshes = bool(getattr(args, "prefer_existing_runtime_meshes", False))
    if part_pid_filter_raw:
        keep_pids = {
            int(tok.strip())
            for tok in part_pid_filter_raw.split(",")
            if tok.strip()
        }
        part_specs = [spec for spec in part_specs if int(spec.get("pid", -1)) in keep_pids]
        print(f"🧪 part_pid_filter active, keeping pids={sorted(keep_pids)}")

    debug_pid_E_scale_specs = list(getattr(args, "debug_pid_E_scale", []) or [])
    debug_pid_nu_override_specs = list(getattr(args, "debug_pid_nu_override", []) or [])
    debug_pid_sampler_specs = list(getattr(args, "debug_pid_sampler", []) or [])

    debug_pid_E_scale: Dict[int, float] = {}
    for item in debug_pid_E_scale_specs:
        if len(item) != 2:
            continue
        debug_pid_E_scale[int(item[0])] = float(item[1])

    debug_pid_nu_override: Dict[int, float] = {}
    for item in debug_pid_nu_override_specs:
        if len(item) != 2:
            continue
        debug_pid_nu_override[int(item[0])] = float(item[1])

    debug_pid_sampler: Dict[int, str] = {}
    for item in debug_pid_sampler_specs:
        if len(item) != 2:
            continue
        debug_pid_sampler[int(item[0])] = str(item[1])

    for spec in part_specs:
        pid = int(spec.get("pid", -1))
        if pid in debug_pid_E_scale:
            spec["debug_E_scale"] = float(debug_pid_E_scale[pid])
        if pid in debug_pid_nu_override:
            spec["debug_nu_override"] = float(debug_pid_nu_override[pid])
        if pid in debug_pid_sampler:
            spec["debug_sampler_override"] = str(debug_pid_sampler[pid])
        if _spec_uses_sph(spec):
            spec["prefer_free_surface"] = bool(getattr(args, "liquid_free_surface", True))
            runtime_liquid_sampler = getattr(args, "liquid_sampler", None)
            if runtime_liquid_sampler is None:
                runtime_liquid_sampler = "pbs" if spec["prefer_free_surface"] else "regular"
            spec["liquid_sampler"] = str(runtime_liquid_sampler)
            spec["liquid_stiffness"] = float(getattr(args, "liquid_stiffness", 35000.0))
            spec["liquid_exponent"] = float(getattr(args, "liquid_exponent", 7.0))
            spec["liquid_viscosity"] = float(getattr(args, "liquid_viscosity", 0.0015))
            spec["liquid_surface_tension"] = float(getattr(args, "liquid_surface_tension", 0.002))

    rigid_collision_mesh_paths: List[Path] = []
    for rec in metadata.get("rigid_part_links", []) if isinstance(metadata, dict) else []:
        if not isinstance(rec, dict):
            continue
        collision_path = rec.get("collision_mesh_path")
        if not collision_path:
            continue
        path = Path(str(collision_path))
        if path.exists():
            rigid_collision_mesh_paths.append(path)

    for spec in part_specs:
        existing_runtime = _find_existing_runtime_mesh(spec=spec, runtime_mesh_dir=runtime_mesh_dir) if prefer_existing_runtime_meshes else None
        if existing_runtime is not None:
            runtime_mesh_path = str(existing_runtime)
            erosion_info = {
                "applied": True,
                "reason": "reused_existing_runtime_mesh",
                "mesh_path": runtime_mesh_path,
            }
        else:
            runtime_mesh_path, erosion_info = _prepare_eroded_soft_mesh(
                spec=spec,
                other_specs=part_specs,
                runtime_mesh_dir=runtime_mesh_dir,
                object_bbox_size=bbox_size,
                anchored_overlap_scale_boost=anchored_overlap_scale_boost,
                rigid_collision_mesh_paths=rigid_collision_mesh_paths,
            )
        spec["runtime_mesh_path"] = runtime_mesh_path
        spec["erosion_info"] = erosion_info
        runtime_bounds_info = _mesh_bounds_info(Path(runtime_mesh_path), scale=float(spec.get("scale", 1.0)))
        if runtime_bounds_info is not None:
            for key, value in runtime_bounds_info.items():
                spec[f"runtime_{key}"] = value

        if "runtime_alignment_offset" not in spec:
            spec["runtime_alignment_offset"] = [0.0, 0.0, 0.0]

        # If runtime mesh generation changed the geometry (fill / erosion / shift / scale),
        # promote the runtime mesh to be the canonical mesh for downstream placement,
        # binding, and bounds-based support logic. Keep the world-space center unchanged by
        # compensating the morph position with the original-vs-runtime bounds-center delta.
        if bool(erosion_info.get("applied", False)) and str(runtime_mesh_path) != str(spec.get("mesh_path", runtime_mesh_path)):
            original_center = np.asarray(spec.get("bounds_center", [0.0, 0.0, 0.0]), dtype=np.float64)
            runtime_center = np.asarray(
                runtime_bounds_info.get("bounds_center", spec.get("bounds_center", [0.0, 0.0, 0.0])) if runtime_bounds_info is not None else spec.get("bounds_center", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            )
            alignment_mode = str(erosion_info.get("alignment_mode", "keep_center"))
            if alignment_mode == "keep_top_xy_center":
                original_bounds_max = np.asarray(spec.get("bounds_max", [0.0, 0.0, 0.0]), dtype=np.float64)
                runtime_bounds_max = np.asarray(
                    runtime_bounds_info.get("bounds_max", spec.get("bounds_max", [0.0, 0.0, 0.0])) if runtime_bounds_info is not None else spec.get("bounds_max", [0.0, 0.0, 0.0]),
                    dtype=np.float64,
                )
                runtime_alignment_offset = np.zeros(3, dtype=np.float64)
                runtime_alignment_offset[:2] = original_center[:2] - runtime_center[:2]
                runtime_alignment_offset[2] = original_bounds_max[2] - runtime_bounds_max[2]
            else:
                runtime_alignment_offset = original_center - runtime_center
            spec["runtime_alignment_offset"] = runtime_alignment_offset.tolist()
            spec["original_mesh_path"] = str(spec.get("mesh_path", runtime_mesh_path))
            spec["mesh_path"] = runtime_mesh_path
            if runtime_bounds_info is not None:
                spec.update(runtime_bounds_info)

    rigid_specs = [spec for spec in part_specs if str(spec.get("assembly_role", "free_soft")) == "rigid_skeleton"]
    anchored_specs = [spec for spec in part_specs if str(spec.get("assembly_role", "free_soft")) == "anchored_soft"]
    free_soft_specs = [spec for spec in part_specs if str(spec.get("assembly_role", "free_soft")) == "free_soft"]
    anchored_bindings = _match_anchored_soft_to_rigid_links(anchored_specs, rigid_specs, metadata)
    debug_hide_rigid_visuals = bool(getattr(args, "debug_hide_rigid_visuals", False))
    debug_disable_free_soft = bool(getattr(args, "debug_disable_free_soft", False))
    debug_highlight_anchored_soft = bool(getattr(args, "debug_highlight_anchored_soft", False))
    anchored_soft_mesh_source = str(getattr(args, "anchored_soft_mesh_source", "runtime"))
    mpm_vis_mode = str(getattr(args, "mpm_vis_mode", "visual"))
    debug_spread_soft_parts = bool(getattr(args, "debug_spread_soft_parts", False))
    custom_object_cfgs = list(runtime_case_cfg.get("custom_objects", []) or [])
    custom_has_mpm_mesh = any(
        str(cfg.get("mesh_path", "") or "").strip() and str(cfg.get("runtime_solver", "mpm")) != "rigid_approx"
        for cfg in custom_object_cfgs
    )

    mpm_specs = [spec for spec in part_specs if _spec_uses_mpm(spec)]
    needs_mpm = len(mpm_specs) > 0 or custom_has_mpm_mesh
    needs_sph = any(_spec_uses_sph(spec) for spec in part_specs)
    needs_pbd = any(_spec_uses_pbd(spec) for spec in part_specs)
    has_soft = any(str(spec.get("material_ctor", "")) != "gs.materials.Rigid" for spec in part_specs)
    sph_particle_size = _suggest_sph_particle_size(part_specs) if needs_sph else None
    free_surface_liquids = any(_spec_uses_sph(spec) and _liquid_prefers_free_surface(spec) for spec in part_specs)
    runtime_substeps = int(substeps)
    if needs_sph and sph_particle_size is not None:
        if free_surface_liquids:
            target_solver_dt = 1.6e-4 if sph_particle_size <= 0.006 else 2.0e-4
        else:
            target_solver_dt = 2.0e-4 if sph_particle_size <= 0.006 else 2.5e-4
        min_substeps = max(1, int(math.ceil(float(dt) / target_solver_dt)))
        if min_substeps > runtime_substeps:
            print(
                f"🫧 auto increase substeps for SPH {runtime_substeps} -> {min_substeps} "
                f"(dt={float(dt):.4f}, particle_size={sph_particle_size:.4f})"
            )
            runtime_substeps = min_substeps

    mpm_lower = None
    mpm_upper = None
    if needs_mpm:
        mpm_lower = (-2.0, -1.5, -1.0)
        mpm_upper = (2.0, 1.5, 2.0)
        if debug_spread_soft_parts:
            mpm_lower = (-2.5, -2.4, -1.0)
            mpm_upper = (2.5, 2.0, 2.2)
        custom_mpm_objects = [
            cfg for cfg in custom_object_cfgs
            if str(cfg.get("mesh_path", "") or "").strip() and str(cfg.get("runtime_solver", "mpm")) != "rigid_approx"
        ]
        if custom_mpm_objects:
            lower = np.asarray(mpm_lower, dtype=np.float64)
            upper = np.asarray(mpm_upper, dtype=np.float64)
            for custom_cfg in custom_mpm_objects:
                spawn_offset = np.asarray(custom_cfg.get("spawn_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
                custom_scale = float(max(1e-3, custom_cfg.get("scale", 0.15)))
                start_pos = placed_pos + spawn_offset
                # Reserve a conservative envelope so spawned MPM meshes always stay inside the solver domain.
                pad = np.array([0.45, 0.45, 0.45], dtype=np.float64) * max(0.12, custom_scale)
                lower = np.minimum(lower, start_pos - pad)
                upper = np.maximum(upper, start_pos + pad)
            mpm_lower = tuple((lower - np.array([0.12, 0.12, 0.12], dtype=np.float64)).tolist())
            mpm_upper = tuple((upper + np.array([0.12, 0.12, 0.12], dtype=np.float64)).tolist())
        print("💛 fixed MPM lower =", mpm_lower)
        print("💛 fixed MPM upper =", mpm_upper)

    sph_lower = (-2.0, -1.5, -1.0) if needs_sph else None
    sph_upper = (2.0, 1.5, 2.0) if needs_sph else None

    vis_kwargs = {
        "visualize_mpm_boundary": False,
        "visualize_sph_boundary": False,
    }

    scene_kwargs = dict(
        sim_options=gs.options.SimOptions(
            dt=float(dt),
            substeps=int(runtime_substeps),
            gravity=(0.0, 0.0, gravity_z),
            floor_height=0.0,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=float(dt),
            gravity=(0.0, 0.0, gravity_z),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_fov=35,
            camera_pos=(3.5, -1.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        vis_options=gs.options.VisOptions(**vis_kwargs),
    )

    if needs_mpm:
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(
            dt=float(dt),
            gravity=(0.0, 0.0, gravity_z),
            lower_bound=tuple(mpm_lower),
            upper_bound=tuple(mpm_upper),
            grid_density=48,
        )

    if needs_sph:
        print(f"🫧 runtime SPH particle_size={sph_particle_size:.4f}")
        scene_kwargs["sph_options"] = gs.options.SPHOptions(
            dt=float(dt),
            gravity=(0.0, 0.0, gravity_z),
            lower_bound=(-2.0, -1.5, -1.0),
            upper_bound=( 2.0,  1.5,  2.0),
            particle_size=float(sph_particle_size),
        )

    scene = gs.Scene(**scene_kwargs)

    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(rho=1200.0, friction=0.95),
    )

    articulated_ent = None
    anchored_runtime_entities: List[Dict[str, Any]] = []
    nonrigid_runtime_entities: List[Any] = []
    nonrigid_runtime_records: List[Dict[str, Any]] = []
    aux_runtime_entities: List[Dict[str, Any]] = []
    rigid_urdf_path = obj_dir / "rigid" / f"{prepared.object_id}.urdf"
    skip_rigid_skeleton = bool(getattr(args, "skip_rigid_skeleton", False))
    has_rigid_skeleton = rigid_urdf_path.exists() and (len(metadata.get("rigid_part_links", [])) > 0 or len(metadata.get("rigid_group_carriers", [])) > 0)
    if skip_rigid_skeleton:
        has_rigid_skeleton = False
    use_anchored_hybrid = bool(
        getattr(args, "use_anchored_hybrid", False) and has_rigid_skeleton and anchored_bindings
    )

    if has_rigid_skeleton:
        urdf_kwargs = dict(
            file=str(rigid_urdf_path),
            scale=float(runtime_main_object_scale),
            pos=tuple(placed_pos.tolist()),
            euler=tuple(object_euler_deg.tolist()),
            visualization=not debug_hide_rigid_visuals,
            collision=True,
            fixed=bool(runtime_object_fixed),
            merge_fixed_links=False,
            prioritize_urdf_material=True,
            file_meshes_are_zup=True,
        )

        if use_anchored_hybrid:
            func_soft_from_rigid, func_assoc = _make_anchored_soft_hybrid_callbacks(
                gs=gs,
                bindings=anchored_bindings,
                placed_pos=placed_pos,
            )
            anchored_density = float(np.median([float(spec.get("density", 800.0)) for spec in anchored_specs])) if anchored_specs else 800.0
            anchored_youngs = float(np.median([float(spec.get("youngs", 1e6) or 1e6) for spec in anchored_specs])) if anchored_specs else 1e6
            anchored_poisson = float(np.median([float(spec.get("poisson", 0.25) or 0.25) for spec in anchored_specs])) if anchored_specs else 0.25
            anchored_E, anchored_nu = _stabilize_runtime_mpm_params(
                "gs.materials.MPM.Elastic",
                anchored_youngs,
                anchored_poisson,
            )

            articulated_ent = scene.add_entity(
                morph=gs.morphs.URDF(**urdf_kwargs),
                material=gs.materials.Hybrid(
                    material_rigid=_make_genesis_rigid_material(
                        gs,
                        rho=float(rigid_material_cfg["rho"]),
                        friction=float(rigid_material_cfg["friction"]),
                        restitution=float(rigid_material_cfg["restitution"]),
                    ),
                    material_soft=gs.materials.MPM.Muscle(
                        E=float(anchored_E),
                        nu=float(anchored_nu),
                        rho=float(anchored_density),
                        sampler="pbs-8",
                        model="corotation",
                        n_groups=max(1, len(anchored_bindings)),
                    ),
                    damping=0.0,
                    soft_dv_coef=0.02,
                    func_instantiate_soft_from_rigid=func_soft_from_rigid,
                    func_instantiate_rigid_soft_association=func_assoc,
                ),
                surface=gs.surfaces.Default(
                    color=(1.0, 0.15, 0.15, 1.0) if debug_highlight_anchored_soft else (0.35, 0.65, 0.35, 1.0),
                    vis_mode="visual",
                ),
            )
        else:
            articulated_ent = scene.add_entity(
                morph=gs.morphs.URDF(**urdf_kwargs),
                material=_make_genesis_rigid_material(
                    gs,
                    rho=float(rigid_material_cfg["rho"]),
                    friction=float(rigid_material_cfg["friction"]),
                    restitution=float(rigid_material_cfg["restitution"]),
                ),
            )

    debug_detach_anchored_offset = np.asarray(
        getattr(args, "debug_detach_anchored_offset", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)
    debug_pid_offset_specs = list(getattr(args, "debug_pid_offset", []) or [])
    debug_pid_offsets: Dict[int, np.ndarray] = {}
    for item in debug_pid_offset_specs:
        if len(item) != 4:
            continue
        pid = int(item[0])
        debug_pid_offsets[pid] = np.asarray(
            [float(item[1]), float(item[2]), float(item[3])],
            dtype=np.float64,
        )
    debug_soft_spread_gap = float(getattr(args, "debug_soft_spread_gap", 0.45) or 0.45)
    debug_soft_spread_y_offset = float(getattr(args, "debug_soft_spread_y_offset", 0.85) or 0.85)
    soft_specs_for_spread = [spec for spec in part_specs if str(spec.get("assembly_role", "free_soft")) != "rigid_skeleton"]
    spread_offsets_by_pid: Dict[int, np.ndarray] = {}
    if debug_spread_soft_parts and soft_specs_for_spread:
        ordered_softs = sorted(soft_specs_for_spread, key=lambda spec: int(spec.get("pid", -1)))
        center_idx = 0.5 * (len(ordered_softs) - 1)
        base_y_offset = -(0.9 * float(bbox_size[1]) + debug_soft_spread_y_offset)
        for idx, soft_spec in enumerate(ordered_softs):
            x_offset = (float(idx) - center_idx) * debug_soft_spread_gap
            spread_offsets_by_pid[int(soft_spec["pid"])] = np.asarray([x_offset, base_y_offset, 0.0], dtype=np.float64)

    seat_surface_spec = next((spec for spec in anchored_specs if _is_seat_surface_spec(spec)), None)
    primary_liquid_target: Optional[Dict[str, Any]] = None
    primary_liquid_entity: Optional[Any] = None
    liquid_bottom_seals = _compute_liquid_container_bottom_seals(part_specs)
    liquid_guard_meshes: Dict[int, Dict[str, Any]] = {}
    rigid_container_guard_mesh_path = None
    rigid_container_guard_info: Dict[str, Any] = {}
    if has_rigid_skeleton:
        rigid_container_guard_mesh_path, rigid_container_guard_info = _build_rigid_container_guard_mesh(
            metadata=metadata,
            runtime_mesh_dir=runtime_mesh_dir,
            pitch=max(0.9 * float(sph_particle_size if sph_particle_size is not None else 0.005), 0.004),
        )
    for spec in part_specs:
        if not _is_liquid_spec(spec):
            continue
        if _liquid_prefers_free_surface(spec):
            continue
        runtime_mesh_path = Path(str(spec.get("runtime_mesh_path", spec.get("mesh_path", ""))))
        if not runtime_mesh_path.exists():
            continue
        liquid_guard_pitch = float(sph_particle_size) if sph_particle_size is not None else 0.005
        guard_mesh_path, guard_info = _build_liquid_container_guard_mesh(
            liquid_mesh_path=runtime_mesh_path,
            runtime_mesh_dir=runtime_mesh_dir,
            pitch=max(0.9 * liquid_guard_pitch, 0.004),
        )
        if guard_mesh_path is not None:
            liquid_guard_meshes[int(spec.get("pid", -1))] = {
                "mesh_path": guard_mesh_path,
                "info": guard_info,
            }

    def _compute_pillow_on_seat_offset(spec: Dict[str, Any]) -> np.ndarray:
        if seat_surface_spec is None or not _is_pillow_spec(spec):
            return np.zeros(3, dtype=np.float64)

        seat_min = np.asarray(
            seat_surface_spec.get("runtime_bounds_min", seat_surface_spec.get("bounds_min", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        seat_max = np.asarray(
            seat_surface_spec.get("runtime_bounds_max", seat_surface_spec.get("bounds_max", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        pillow_min = np.asarray(spec.get("runtime_bounds_min", spec.get("bounds_min", [0.0, 0.0, 0.0])), dtype=np.float64)
        pillow_max = np.asarray(spec.get("runtime_bounds_max", spec.get("bounds_max", [0.0, 0.0, 0.0])), dtype=np.float64)
        pillow_size = np.maximum(pillow_max - pillow_min, 1e-6)
        pillow_center = 0.5 * (pillow_min + pillow_max)

        seat_top = float(seat_max[2])
        seat_front_center = float(0.5 * (seat_min[1] + seat_max[1]))
        seat_front_limit = float(seat_max[1] - 0.5 * pillow_size[1] - 0.005)
        target_y = min(float(pillow_center[1]), seat_front_limit)
        target_y = max(target_y, seat_front_center - 0.08)
        target_z = float(seat_top + 0.5 * pillow_size[2] + 0.01)

        offset = np.zeros(3, dtype=np.float64)
        offset[1] = target_y - float(pillow_center[1])
        offset[2] = target_z - float(pillow_center[2])
        return offset

    for seal in liquid_bottom_seals:
        seal_min_local = np.asarray(seal["bounds_min"], dtype=np.float64)
        seal_max_local = np.asarray(seal["bounds_max"], dtype=np.float64)
        seal_min_world = seal_min_local + placed_pos
        seal_max_world = seal_max_local + placed_pos
        seal_center_world = 0.5 * (seal_min_world + seal_max_world)
        print(
            f"🫙 liquid_bottom_seal pid={seal['pid']} part={seal['part_name']} "
            f"local_min={seal_min_local.tolist()} local_max={seal_max_local.tolist()} "
            f"world_min={seal_min_world.tolist()} world_max={seal_max_world.tolist()}"
        )
        seal_ent = scene.add_entity(
            morph=gs.morphs.Box(
                lower=tuple(seal_min_world.tolist()),
                upper=tuple(seal_max_world.tolist()),
                visualization=False,
                collision=True,
                fixed=True,
            ),
            material=gs.materials.Rigid(rho=1200.0, friction=0.98),
        )
        aux_runtime_entities.append(
            {
                "entity": seal_ent,
                "base_pos": seal_center_world,
                "kind": "liquid_bottom_seal",
                "pid": int(seal.get("pid", -1)),
            }
        )

    if rigid_container_guard_mesh_path is not None:
        print(
            f"🫙 rigid_container_guard reason={rigid_container_guard_info.get('reason', 'unknown')} "
            f"mesh={rigid_container_guard_mesh_path}"
        )
        guard_ent = scene.add_entity(
            material=gs.materials.Rigid(rho=1200.0, friction=0.98),
            morph=gs.morphs.Mesh(
                file=str(rigid_container_guard_mesh_path),
                scale=float(runtime_main_object_scale),
                pos=tuple(placed_pos.tolist()),
                euler=tuple(object_euler_deg.tolist()),
                file_meshes_are_zup=True,
                visualization=False,
                collision=True,
                fixed=True,
                decimate=False,
            ),
        )
        aux_runtime_entities.append(
            {
                "entity": guard_ent,
                "base_pos": placed_pos.copy(),
                "kind": "rigid_container_guard",
                "pid": -1,
            }
        )

    # 参考 HybridEntity 的“刚体骨架 + 软体部件”思路，但为了保留每个 part 自己的材料类型，
    # 这里仍然按 part 单独 add_entity，而不是把整件物体压成单一 material_soft。
    for spec in part_specs:
        role = str(spec.get("assembly_role", "free_soft"))
        if has_rigid_skeleton and role == "rigid_skeleton":
            continue
        if use_anchored_hybrid and role == "anchored_soft":
            continue
        if debug_disable_free_soft and role == "free_soft":
            print(f"🧪 skip free_soft in debug render: pid={spec['pid']} part={spec['part_name']}")
            continue

        # ctor = str(spec.get("material_ctor", ""))
        ctor = str(spec.get("material_ctor_runtime", spec.get("material_ctor", "")))
        if ctor in ["gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid"]:
            vis_mode = str(getattr(args, "liquid_vis_mode", "recon"))
        elif ctor in ["gs.materials.MPM.Sand", "gs.materials.MPM.Snow"]:
            vis_mode = "particle"
        elif ctor in {"gs.materials.PBD.Cloth", "gs.materials.FEM.Cloth"}:
            vis_mode = "visual"
        elif ctor == "gs.materials.MPM.Elastic" or ctor == "gs.materials.MPM.ElastoPlastic":
            vis_mode = mpm_vis_mode
        else:
            vis_mode = "particle"

        material = _make_part_material(gs, spec, default_friction=default_friction)
        runtime_mesh_path = str(spec.get("runtime_mesh_path", spec["mesh_path"]))
        if role == "anchored_soft" and anchored_soft_mesh_source == "original":
            runtime_mesh_path = str(spec["mesh_path"])
        erosion_info = dict(spec.get("erosion_info", {}))

        print(
            f"🩷 pid={spec['pid']} part={spec['part_name']} "
            f"role={role} "
            f"orig_ctor={spec.get('material_ctor')} "
            f"runtime_ctor={ctor} "
            f"vis_mode={vis_mode} "
            f"material={material.__dict__}"
        )
        if role != "rigid_skeleton":
            print(
                f"    soft_mesh_erosion applied={erosion_info['applied']} "
                f"reason={erosion_info['reason']} mesh={erosion_info['mesh_path']}"
            )
        entity_pos = placed_pos.copy()
        runtime_alignment_offset = np.asarray(spec.get("runtime_alignment_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
        if role != "rigid_skeleton":
            # All non-rigid parts belong to the same object frame, so they must inherit
            # the object's initial yaw; otherwise cloth/soft meshes intersect the rotated rigid body.
            runtime_alignment_offset = _rotate_vec_by_euler_deg(runtime_alignment_offset, object_euler_deg)
        entity_pos = entity_pos + runtime_alignment_offset
        if debug_spread_soft_parts and role != "rigid_skeleton":
            entity_pos = entity_pos + spread_offsets_by_pid.get(int(spec["pid"]), np.zeros(3, dtype=np.float64))
        elif role == "anchored_soft" and np.linalg.norm(debug_detach_anchored_offset) > 0.0:
            entity_pos = entity_pos + debug_detach_anchored_offset
        if _is_pillow_spec(spec):
            pillow_on_seat_offset = _compute_pillow_on_seat_offset(spec)
            entity_pos = entity_pos + pillow_on_seat_offset
            if float(np.linalg.norm(pillow_on_seat_offset)) > 1e-8:
                print(
                    f"🛋️ pillow_reseat pid={spec['pid']} part={spec['part_name']} "
                    f"offset={pillow_on_seat_offset.tolist()}"
                )
        if int(spec["pid"]) in debug_pid_offsets:
            entity_pos = entity_pos + debug_pid_offsets[int(spec["pid"])]
        if _is_free_cloth_like_spec(spec):
            cloth_follow_gap = min(
                0.040,
                max(
                    0.016,
                    0.040 * float(np.min(np.maximum(np.asarray(spec.get("bounds_size", [0.05, 0.05, 0.05]), dtype=np.float64), 1e-6))),
                ),
            )
            entity_pos[2] += cloth_follow_gap

        part_euler_deg = _compose_euler_deg_xyz(
            object_euler_deg if role != "rigid_skeleton" else [0.0, 0.0, 0.0],
            spec.get("euler", (0.0, 0.0, 0.0)),
        )

        liquid_guard_cfg = liquid_guard_meshes.get(int(spec["pid"]))
        if liquid_guard_cfg is not None:
            guard_info = dict(liquid_guard_cfg.get("info", {}))
            guard_path = str(liquid_guard_cfg["mesh_path"])
            print(
                f"🫙 liquid_container_guard pid={spec['pid']} part={spec['part_name']} "
                f"reason={guard_info.get('reason', 'unknown')} mesh={guard_path}"
            )
            guard_ent = scene.add_entity(
                material=gs.materials.Rigid(rho=1200.0, friction=0.98),
                morph=gs.morphs.Mesh(
                    file=guard_path,
                    scale=float(spec.get("scale", 1.0)),
                    pos=tuple(entity_pos.tolist()),
                    euler=tuple(part_euler_deg.tolist()),
                    file_meshes_are_zup=bool(spec.get("file_meshes_are_zup", True)),
                    visualization=False,
                    collision=True,
                    fixed=True,
                    decimate=False,
                ),
            )
            aux_runtime_entities.append(
                {
                    "entity": guard_ent,
                    "base_pos": entity_pos.copy(),
                    "kind": "liquid_container_guard",
                    "pid": int(spec.get("pid", -1)),
                }
            )

        liquid_bounds_center = np.asarray(
            spec.get("runtime_bounds_center", spec.get("bounds_center", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        liquid_bounds_max = np.asarray(
            spec.get("runtime_bounds_max", spec.get("bounds_max", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        liquid_bounds_min = np.asarray(
            spec.get("runtime_bounds_min", spec.get("bounds_min", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        liquid_bounds_size = np.asarray(
            spec.get("runtime_bounds_size", spec.get("bounds_size", [0.0, 0.0, 0.0])),
            dtype=np.float64,
        )
        if ctor in ("gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid"):
            candidate = {
                "pid": int(spec["pid"]),
                "part_name": str(spec.get("part_name", f"part_{spec['pid']}")),
                "center_world": entity_pos + liquid_bounds_center,
                "top_world_z": float(entity_pos[2] + liquid_bounds_max[2]),
                "bottom_world_z": float(entity_pos[2] + liquid_bounds_min[2]),
                "xy_extent": float(np.min(np.maximum(liquid_bounds_size[:2], 1e-6))),
                "xy_area": float(np.prod(np.maximum(liquid_bounds_size[:2], 1e-6))),
            }
            if primary_liquid_target is None or candidate["xy_area"] > primary_liquid_target["xy_area"]:
                primary_liquid_target = candidate

        surface_kwargs: Dict[str, Any] = {
            "color": spec["color"],
            "vis_mode": vis_mode,
        }
        if ctor in {"gs.materials.PBD.Cloth", "gs.materials.FEM.Cloth"}:
            surface_kwargs["double_sided"] = True
        if ctor in ("gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid") and vis_mode == "recon":
            surface_kwargs["recon_backend"] = str(getattr(args, "liquid_recon_backend", "splashsurf"))

        part_ent = scene.add_entity(
            material=material,
            morph=gs.morphs.Mesh(
                file=runtime_mesh_path,
                # file = path_map(spec["mesh_path"]),
                scale=float(spec.get("scale", 1.0)),
                pos=tuple(entity_pos.tolist()),
                euler=tuple(part_euler_deg.tolist()),
                file_meshes_are_zup=bool(spec.get("file_meshes_are_zup", True)),
            ),
            surface=gs.surfaces.Default(**surface_kwargs),
        )
        try:
            setattr(part_ent, "_physics_pid", int(spec["pid"]))
        except Exception:
            pass
        if role == "anchored_soft":
            anchored_runtime_entities.append({"spec": spec, "entity": part_ent})
        nonrigid_runtime_entities.append(part_ent)
        nonrigid_runtime_records.append(
            {
                "pid": int(spec.get("pid", -1)),
                "part_name": str(spec.get("part_name", f"part_{spec.get('pid', -1)}")),
                "spec": spec,
                "entity": part_ent,
                "base_pos": np.asarray(entity_pos, dtype=np.float64).copy(),
                "prefer_upward_clearance": bool(_is_free_cloth_like_spec(spec) or role == "free_soft"),
            }
        )
        if (
            primary_liquid_target is not None
            and ctor in ("gs.materials.SPH.Liquid", "gs.materials.MPM.Liquid")
            and int(spec["pid"]) == int(primary_liquid_target["pid"])
        ):
            primary_liquid_entity = part_ent

    liquid_scene = primary_liquid_target is not None
    disable_striker = bool(debug_spread_soft_parts)
    if liquid_scene:
        disable_striker = disable_striker or (not bool(getattr(args, "enable_striker", False)))
    else:
        disable_striker = disable_striker or bool(getattr(args, "disable_striker", False))
    disable_striker = disable_striker or bool(runtime_case_cfg.get("disable_default_striker", False))
    striker = None
    custom_runtime_objects: List[Dict[str, Any]] = []
    striker_radius_runtime = float(striker_radius) * float(runtime_striker_radius_mult)
    striker_clearance = max(0.01, 0.20 * striker_radius_runtime)
    striker_start = np.array(
        [
            float(placed_pos[0] + bbox_max[0] + striker_radius_runtime + striker_clearance + args.ball_posx),
            float(placed_pos[1] + bbox_center[1]),
            float(placed_pos[2] + bbox_min[2] + 0.4 * bbox_size[2] + striker_radius_runtime + striker_clearance),
        ],
        dtype=np.float64,
    )
    striker_velocity = np.array([-runtime_striker_speed, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if bool(getattr(args, "striker_drop_top", False)):
        drop_xy_jitter = float(getattr(args, "striker_drop_xy_jitter", 0.0) or 0.0)
        drop_height = float(max(0.18, float(getattr(args, "striker_drop_height", 0.32) or 0.32)))
        striker_start = np.array(
            [
                float(placed_pos[0] + bbox_center[0] + rng.uniform(-drop_xy_jitter, drop_xy_jitter)),
                float(placed_pos[1] + bbox_center[1] + rng.uniform(-drop_xy_jitter, drop_xy_jitter)),
                float(placed_pos[2] + bbox_max[2] + striker_radius_runtime + drop_height),
            ],
            dtype=np.float64,
        )
        striker_velocity = np.array([0.0, 0.0, -runtime_striker_speed, 0.0, 0.0, 0.0], dtype=np.float64)
    if primary_liquid_target is not None:
        liquid_xy_extent = float(primary_liquid_target["xy_extent"])
        fit_radius = float(np.clip(0.24 * liquid_xy_extent, 0.008, float(striker_radius)))
        if fit_radius < striker_radius_runtime - 1e-6:
            print(
                f"🫧 liquid_drop adjust striker radius {striker_radius_runtime:.4f} -> {fit_radius:.4f} "
                f"for pid={primary_liquid_target['pid']} part={primary_liquid_target['part_name']}"
            )
            striker_radius_runtime = fit_radius

        liquid_center_world = np.asarray(primary_liquid_target["center_world"], dtype=np.float64)
        liquid_top_world_z = float(primary_liquid_target["top_world_z"])
        drop_clearance = max(0.05, 0.75 * striker_radius_runtime, 0.55 * float(bbox_size[2]))
        striker_start = np.array(
            [
                float(liquid_center_world[0]),
                float(liquid_center_world[1]),
                float(liquid_top_world_z + striker_radius_runtime + drop_clearance),
            ],
            dtype=np.float64,
        )
        striker_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        print(
            f"🫧 liquid_drop spawn above liquid center pid={primary_liquid_target['pid']} "
            f"start={striker_start.tolist()} liquid_top_z={liquid_top_world_z:.4f}"
        )
    striker_start = striker_start + np.asarray(
        runtime_case_cfg.get("default_striker_start_offset", [0.0, 0.0, 0.0]),
        dtype=np.float64,
    ).reshape(3)
    striker_velocity = _coerce_velocity6(
        runtime_case_cfg.get("default_striker_velocity_override", None),
        default=striker_velocity.tolist(),
    )
    if not disable_striker:
        striker = scene.add_entity(
            morph=gs.morphs.Sphere(
                radius=float(striker_radius_runtime),
                pos=tuple(striker_start.tolist()),
                euler=(0.0, 0.0, 0.0),
            ),
            material=_make_genesis_rigid_material(
                gs,
                rho=1800.0,
                friction=0.35,
                restitution=float(rigid_material_cfg["restitution"]),
            ),
            surface=gs.surfaces.Default(color=(0.95, 0.75, 0.15, 1.0), vis_mode="visual"),
        )
        custom_runtime_objects.append(
            {
                "custom_object_id": "custom_ball_default",
                "entity": striker,
                "start_pos": striker_start.copy(),
                "velocity6": striker_velocity.copy(),
                "radius": float(striker_radius_runtime),
                "clearance": float(striker_clearance),
                "source_tag": "custom_object",
                "role_hint": "initiator",
                "motion_type_hint": "striker_hit",
                "motion_group_hint": "striker",
                "source_display_id": "yellow_striker_ball",
                "speed_mps": float(runtime_striker_speed),
            }
        )

    if not liquid_scene:
        for custom_cfg in runtime_case_cfg.get("custom_objects", []) or []:
            spawn_offset = np.asarray(custom_cfg.get("spawn_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            start_pos = placed_pos + spawn_offset
            linear = np.asarray(custom_cfg.get("linear_velocity", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            angular = np.asarray(custom_cfg.get("angular_velocity", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)
            velocity6 = np.concatenate([linear, angular], axis=0).astype(np.float64)
            color = tuple(np.asarray(custom_cfg.get("color_rgba", [0.95, 0.75, 0.15, 1.0]), dtype=np.float64).reshape(4).tolist())
            custom_id = str(custom_cfg.get("custom_object_id", f"custom_obj_{len(custom_runtime_objects):02d}"))
            mesh_path = str(custom_cfg.get("mesh_path", "") or "")
            custom_scale = float(max(1e-3, custom_cfg.get("scale", 0.15)))
            custom_euler = tuple(np.asarray(custom_cfg.get("euler_deg", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3).tolist())
            runtime_solver = str(custom_cfg.get("runtime_solver", "mpm"))
            runtime_material_ctor = str(custom_cfg.get("runtime_material_ctor", custom_cfg.get("material_ctor", "gs.materials.MPM.Elastic")))

            if mesh_path and Path(mesh_path).exists() and runtime_solver != "rigid_approx":
                custom_spec = {
                    "pid": -1000 - len(custom_runtime_objects),
                    "part_name": custom_id,
                    "material_ctor_runtime": runtime_material_ctor,
                    "material_ctor": str(custom_cfg.get("material_ctor", "gs.materials.MPM.Elastic")),
                    "density": custom_cfg.get("density", None),
                    "youngs": custom_cfg.get("youngs", None),
                    "poisson": custom_cfg.get("poisson", None),
                    "friction": float(custom_cfg.get("friction", 0.55)),
                    "restitution": None,
                    "damping": None,
                    "assembly_role": "free_soft",
                    "color": color,
                    "strict_dataset_params": bool(custom_cfg.get("strict_dataset_params", False)),
                }
                custom_material = _make_part_material(gs, custom_spec, default_friction=float(custom_cfg.get("friction", 0.55)))
                custom_vis_mode = "visual"
                custom_ctor = str(custom_spec.get("material_ctor_runtime", custom_spec.get("material_ctor", "")))
                if custom_ctor in ("gs.materials.MPM.Sand", "gs.materials.MPM.Snow", "gs.materials.MPM.Liquid"):
                    custom_vis_mode = str(getattr(args, "mpm_vis_mode", "particle"))
                    if custom_vis_mode == "visual":
                        custom_vis_mode = "particle"
                    if custom_ctor == "gs.materials.MPM.Liquid":
                        custom_vis_mode = str(getattr(args, "liquid_vis_mode", "particle"))
                custom_ent = scene.add_entity(
                    material=custom_material,
                    morph=gs.morphs.Mesh(
                        file=mesh_path,
                        scale=custom_scale,
                        pos=tuple(start_pos.tolist()),
                        euler=custom_euler,
                        file_meshes_are_zup=True,
                    ),
                    surface=gs.surfaces.Default(color=color, vis_mode=custom_vis_mode),
                )
                clear_extent = custom_scale
            elif mesh_path and Path(mesh_path).exists() and runtime_solver == "rigid_approx":
                custom_ent = scene.add_entity(
                    material=_make_genesis_rigid_material(
                        gs,
                        rho=float(custom_cfg.get("density", 1000.0)),
                        friction=float(custom_cfg.get("friction", 0.55)),
                        restitution=float(rigid_material_cfg["restitution"]),
                    ),
                    morph=gs.morphs.Mesh(
                        file=mesh_path,
                        scale=custom_scale,
                        pos=tuple(start_pos.tolist()),
                        euler=custom_euler,
                        file_meshes_are_zup=True,
                    ),
                    surface=gs.surfaces.Default(color=color, vis_mode="visual"),
                )
                clear_extent = custom_scale
            else:
                radius = float(max(0.01, custom_cfg.get("radius", striker_radius)))
                custom_ent = scene.add_entity(
                    morph=gs.morphs.Sphere(
                        radius=radius,
                        pos=tuple(start_pos.tolist()),
                        euler=(0.0, 0.0, 0.0),
                    ),
                    material=_make_genesis_rigid_material(
                        gs,
                        rho=1800.0,
                        friction=0.35,
                        restitution=float(rigid_material_cfg["restitution"]),
                    ),
                    surface=gs.surfaces.Default(color=color, vis_mode="visual"),
                )
                clear_extent = radius
            custom_runtime_objects.append(
                {
                    "custom_object_id": custom_id,
                    "entity": custom_ent,
                    "start_pos": start_pos.copy(),
                    "velocity6": velocity6.copy(),
                    "radius": float(max(0.02, clear_extent)),
                    "clearance": max(0.01, 0.20 * float(max(0.02, clear_extent))),
                    "source_tag": str(custom_cfg.get("source_dataset", "custom_object")),
                    "role_hint": str(custom_cfg.get("role_hint", "initiator")),
                    "motion_type_hint": str(custom_cfg.get("motion_type_hint", "striker_hit")),
                    "motion_group_hint": str(custom_cfg.get("motion_group_hint", "striker")),
                    "source_display_id": str(custom_cfg.get("source_display_id", custom_id)),
                }
            )

    camera_distance_mult = float(getattr(args, "camera_distance_mult", 1.0) or 1.0)
    liquid_camera_mode = False
    if primary_liquid_target is not None and camera_distance_mult >= 0.999:
        camera_distance_mult = 0.82
        liquid_camera_mode = True
        print(
            f"🫙 liquid_container_camera_boost applied camera_distance_mult={camera_distance_mult:.2f} "
            f"for pid={primary_liquid_target['pid']} part={primary_liquid_target['part_name']}"
        )

    # Keep rigid previews tighter so medium/small objects occupy more of the frame.
    cam_distance = camera_distance_mult * max(1.78, 1.66 * float(np.max(bbox_size)) + 0.78)
    cam_height = camera_distance_mult * max(0.94, float(placed_pos[2] + bbox_min[2] + 0.65 * bbox_size[2] + 0.24))
    lookat = np.array([0.0, 0.0, float(placed_pos[2] + bbox_min[2] + 0.55 * bbox_size[2])], dtype=np.float64)
    label_l = str(scene_label).strip().lower()
    if label_l in {"random_parabola", "high_drop"}:
        # These single-object gravity cases start above the tray. Keep the camera
        # close but bias the target downward so the landing remains in frame.
        if label_l == "high_drop":
            camera_distance_mult = max(camera_distance_mult, 1.36)
            cam_distance = camera_distance_mult * max(1.92, 1.76 * float(np.max(bbox_size)) + 0.80)
            cam_height = camera_distance_mult * max(0.84, float(0.44 * placed_pos[2] + bbox_min[2] + 0.15 * bbox_size[2] + 0.20))
            lookat = np.array([0.0, 0.0, float(max(0.42, 0.42 * placed_pos[2]))], dtype=np.float64)
            cam_fov = 40
        else:
            camera_distance_mult = max(camera_distance_mult, 1.28)
            cam_distance = camera_distance_mult * max(1.86, 1.72 * float(np.max(bbox_size)) + 0.78)
            cam_height = camera_distance_mult * max(0.86, float(placed_pos[2] + bbox_min[2] + 0.22 * bbox_size[2] + 0.01))
            lookat = np.array([0.0, 0.0, float(max(0.18, placed_pos[2] - 0.55))], dtype=np.float64)
            cam_fov = 36
    elif label_l.startswith("multi") and ("projectile" in label_l or "drop" in label_l):
        start_points = [placed_pos.copy()]
        start_points.extend(np.asarray(rec.get("start_pos", placed_pos), dtype=np.float64).reshape(3) for rec in custom_runtime_objects)
        start_arr = np.asarray(start_points, dtype=np.float64).reshape(-1, 3)
        num_multi_objs = int(start_arr.shape[0])
        xy_center = np.mean(start_arr[:, :2], axis=0)
        xy_span = np.ptp(start_arr[:, :2], axis=0)
        z_top = float(np.max(start_arr[:, 2]))
        lookat = np.array(
            [
                float(xy_center[0]),
                float(xy_center[1]),
                float(max(0.28, 0.26 * z_top if "drop" in label_l else 0.34 * z_top)),
            ],
            dtype=np.float64,
        )
        camera_distance_mult = max(
            camera_distance_mult,
            1.02 if num_multi_objs >= 4 else (0.96 if num_multi_objs >= 3 else 1.10),
        )
        span_ref = float(max(np.max(xy_span), np.max(bbox_size), 0.60))
        cam_distance = camera_distance_mult * max(
            1.46 if num_multi_objs >= 4 else (1.32 if num_multi_objs >= 3 else 1.58),
            (1.14 if num_multi_objs >= 4 else (1.04 if num_multi_objs >= 3 else 1.24))
            + (0.84 if num_multi_objs >= 4 else (0.78 if num_multi_objs >= 3 else 0.96)) * span_ref,
        )
        cam_height = camera_distance_mult * max(
            0.82 if num_multi_objs >= 4 else (0.76 if num_multi_objs >= 3 else 0.88),
            0.38 * z_top + 0.44,
        )
        cam_fov = 48 if num_multi_objs >= 4 else (46 if num_multi_objs >= 3 else 48)
    if debug_spread_soft_parts and spread_offsets_by_pid:
        soft_offsets = np.asarray(list(spread_offsets_by_pid.values()), dtype=np.float64)
        lookat[:2] = np.mean(soft_offsets[:, :2], axis=0) * 0.5
    if label_l.startswith("multi") and ("projectile" in label_l or "drop" in label_l):
        cam_pos = np.array(
            [
                float(lookat[0] + (0.08 if num_multi_objs >= 3 else 0.62) * cam_distance),
                float(lookat[1] - (0.88 if num_multi_objs >= 3 else 0.48) * cam_distance),
                float(1.08 * cam_height),
            ],
            dtype=np.float64,
        )
    else:
        cam_pos = np.array([0.66 * cam_distance, -0.82 * cam_distance, 1.08 * cam_height], dtype=np.float64)
    cam_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if label_l not in {"random_parabola", "high_drop"} and not (label_l.startswith("multi") and ("projectile" in label_l or "drop" in label_l)):
        cam_fov = 32
    if liquid_camera_mode and primary_liquid_target is not None:
        liquid_center_world = np.asarray(primary_liquid_target["center_world"], dtype=np.float64)
        liquid_top_world_z = float(primary_liquid_target["top_world_z"])
        liquid_bottom_world_z = float(primary_liquid_target["bottom_world_z"])
        liquid_xy_extent = float(primary_liquid_target["xy_extent"])
        lookat = liquid_center_world.copy()
        # Bias the target lower and place the camera closer to a side view so the first frame
        # reveals liquid depth inside the bowl instead of reading as a top-only particle sheet.
        lookat[2] = float(0.18 * liquid_top_world_z + 0.82 * liquid_bottom_world_z)
        lateral = max(0.26, 2.10 * liquid_xy_extent)
        depth = max(0.10, 0.72 * liquid_xy_extent)
        height = max(0.09, 0.92 * liquid_xy_extent)
        cam_pos = np.array(
            [
                float(liquid_center_world[0] + lateral),
                float(liquid_center_world[1] - depth),
                float(liquid_top_world_z + height),
            ],
            dtype=np.float64,
        )
        cam_fov = 30
        print(
            f"🫙 liquid_container_camera pose={cam_pos.tolist()} lookat={lookat.tolist()} fov={cam_fov}"
        )
    camera_override_applied = False
    camera_tag_raw = str(getattr(args, "camera_tag", "") or "").strip()
    camera_tag = re.sub(r"[^0-9A-Za-z._-]+", "_", camera_tag_raw).strip("._-")
    camera_pos_override = getattr(args, "camera_pos_override", None)
    camera_lookat_override = getattr(args, "camera_lookat_override", None)
    camera_up_override = getattr(args, "camera_up_override", None)
    camera_fov_override = getattr(args, "camera_fov_override", None)
    if camera_pos_override is not None:
        cam_pos = np.asarray(camera_pos_override, dtype=np.float64).reshape(3)
        camera_override_applied = True
    if camera_lookat_override is not None:
        lookat = np.asarray(camera_lookat_override, dtype=np.float64).reshape(3)
        camera_override_applied = True
    if camera_up_override is not None:
        cam_up = np.asarray(camera_up_override, dtype=np.float64).reshape(3)
        camera_override_applied = True
    if camera_fov_override is not None:
        cam_fov = float(camera_fov_override)
        camera_override_applied = True
    if camera_override_applied:
        print(
            f"📷 camera_override tag={camera_tag or 'untagged'} "
            f"pos={cam_pos.tolist()} lookat={lookat.tolist()} up={cam_up.tolist()} fov={float(cam_fov):.2f}"
        )
    cam = scene.add_camera(
        res=EXPORT_CAMERA_RESOLUTION,
        pos=tuple(cam_pos.tolist()),
        lookat=tuple(lookat.tolist()),
        up=tuple(cam_up.tolist()),
        fov=cam_fov,
        GUI=False,
    )

    scene.build()

    corrected_pos = placed_pos.copy()
    has_nonrigid_parts = any(str(spec.get("assembly_role", "free_soft")) != "rigid_skeleton" for spec in part_specs)

    # if articulated_ent is not None and not has_nonrigid_parts:
    if articulated_ent is not None:
        rigid_runtime_ent = articulated_ent.part_rigid if hasattr(articulated_ent, "part_rigid") else articulated_ent
        aabb = rigid_runtime_ent.get_AABB()
        if hasattr(aabb, "detach"):
            aabb = aabb.detach().cpu().numpy()
        else:
            aabb = np.asarray(aabb)

        z_min = float(aabb[0, 2])
        clearance = 0.002
        placed_offset_z = float(np.asarray(runtime_case_cfg.get("placed_pos_offset", [0.0, 0.0, 0.0]), dtype=np.float64).reshape(3)[2])
        intentional_elevated_spawn = bool(placed_offset_z > 0.10)
        needs_grounding_correction = bool(
            z_min < clearance - 1e-6
            or ((not intentional_elevated_spawn) and abs(z_min - clearance) > 1e-6)
        )
        if needs_grounding_correction:
            corrected_pos = placed_pos.copy()
            corrected_pos[2] += (clearance - z_min)
            rigid_runtime_ent.set_pos(corrected_pos)
            delta_pos = corrected_pos - placed_pos
            for rec in nonrigid_runtime_records:
                try:
                    rec["base_pos"] = np.asarray(rec.get("base_pos", placed_pos), dtype=np.float64) + delta_pos
                    rec["entity"].set_pos(tuple(np.asarray(rec["base_pos"], dtype=np.float64).tolist()))
                except Exception:
                    pass
            for aux_ent in aux_runtime_entities:
                try:
                    base_pos = np.asarray(aux_ent.get("base_pos", placed_pos), dtype=np.float64)
                    aux_ent["entity"].set_pos(tuple((base_pos + delta_pos).tolist()))
                except Exception:
                    pass
            for custom_rec in custom_runtime_objects:
                try:
                    custom_rec["start_pos"] = np.asarray(custom_rec["start_pos"] + delta_pos, dtype=np.float64)
                    custom_rec["entity"].set_pos(tuple(custom_rec["start_pos"].tolist()))
                except Exception:
                    pass

        print("placed_pos before correction:", placed_pos.tolist())
        print("AABB z_min:", z_min)
        print("placed_pos after correction:", corrected_pos.tolist())

        nonrigid_overlap_fixes = _resolve_initial_nonrigid_component_overlaps(
            nonrigid_runtime_records=nonrigid_runtime_records,
            anchored_aabbs=[_entity_aabb_numpy(rigid_runtime_ent)] + [_entity_aabb_numpy(aux_ent.get("entity")) for aux_ent in aux_runtime_entities],
            clearance=max(0.004, 8.0e-3),
            max_passes=24,
        )
        for fix in nonrigid_overlap_fixes:
            print(
                f"🧩 nonrigid_overlap_resolve pid={fix['pid']} "
                f"part={fix['part_name']} delta={fix['delta']} "
                f"prefer_upward={fix['prefer_upward_clearance']}"
            )

        scene_object_aabb = _merge_aabbs(
            [_entity_aabb_numpy(rigid_runtime_ent)] + [_entity_aabb_numpy(ent) for ent in nonrigid_runtime_entities]
        )
        if custom_runtime_objects:
            for custom_rec in custom_runtime_objects:
                custom_aabb = _entity_aabb_numpy(custom_rec["entity"])
                if not _aabb_overlaps(scene_object_aabb, custom_aabb, clearance=1e-4):
                    continue
                custom_pos = np.asarray(custom_rec["start_pos"] + (corrected_pos - placed_pos), dtype=np.float64)
                if scene_object_aabb is not None:
                    custom_pos[0] = float(scene_object_aabb[1, 0] + float(custom_rec["radius"]) + float(custom_rec["clearance"]))
                    custom_pos[1] = float(0.5 * (scene_object_aabb[0, 1] + scene_object_aabb[1, 1]))
                    custom_pos[2] = max(
                        float(custom_pos[2]),
                        float(scene_object_aabb[0, 2] + 0.35 * (scene_object_aabb[1, 2] - scene_object_aabb[0, 2])),
                    )
                custom_rec["entity"].set_pos(tuple(custom_pos.tolist()))
                custom_rec["start_pos"] = custom_pos.copy()
                print(
                    f"🎯 custom_init_clearance applied id={custom_rec['custom_object_id']} "
                    f"pos={custom_pos.tolist()} radius={float(custom_rec['radius']):.4f} "
                    f"clearance={float(custom_rec['clearance']):.4f}"
                )

            anchored_aabbs: List[Optional[np.ndarray]] = []
            if scene_object_aabb is not None:
                anchored_aabbs.append(scene_object_aabb)
            anchored_aabbs.extend(_entity_aabb_numpy(aux_ent.get("entity")) for aux_ent in aux_runtime_entities)
            overlap_fixes = _resolve_initial_custom_object_overlaps(
                custom_runtime_objects=custom_runtime_objects,
                anchored_aabbs=anchored_aabbs,
                clearance=max(0.008, 1.25e-2),
                max_passes=32,
            )
            for fix in overlap_fixes:
                print(
                    f"🎯 custom_overlap_resolve id={fix['custom_object_id']} "
                    f"delta={fix['delta']} final_pos={fix['final_pos']}"
                )

        runtime_apply = _configure_genesis_rigid_entity_from_metadata(
            rigid_runtime_ent,
            metadata,
            default_friction=default_friction,
        )
        metadata.setdefault("runtime_application", {})
        metadata["runtime_application"]["rigid_entity"] = runtime_apply
        with open(obj_dir / "meta" / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    else:
        print("Skip post-build articulated correction because scene contains non-rigid parts.")

    anchored_constraint_stiffness = float(getattr(args, "anchored_constraint_stiffness", 0.0) or 0.0)
    if debug_spread_soft_parts:
        anchored_constraint_stiffness = 0.0
    if (
        anchored_constraint_stiffness > 0.0
        and articulated_ent is not None
        and not use_anchored_hybrid
        and anchored_runtime_entities
        and anchored_bindings
    ):
        applied_constraints = _apply_layered_anchored_constraints(
            articulated_ent=articulated_ent,
            anchored_runtime_entities=anchored_runtime_entities,
            anchored_bindings=anchored_bindings,
            rigid_specs=rigid_specs,
            rigid_pos=corrected_pos,
            stiffness=anchored_constraint_stiffness,
            rigid_fixed=bool(runtime_object_fixed),
        )
        if applied_constraints:
            for rec in applied_constraints:
                print(
                    f"🔗 anchored_constraint pid={rec['pid']} part={rec['part_name']} "
                    f"link={rec['link_name']} link_idx={rec['link_idx']} "
                    f"n_constrained={rec['n_constrained']}/{rec['n_total']} "
                    f"shell={rec['shell_thickness']:.4f} stiffness={rec['stiffness']} "
                    f"mode={rec['mode']}"
                )

    if apply_object_entry_velocity:
        if articulated_ent is not None and not runtime_object_fixed:
            try:
                rigid_entry_target = articulated_ent.part_rigid if hasattr(articulated_ent, "part_rigid") else articulated_ent
                _apply_rigid_entry_velocity(
                    rigid_entry_target,
                    linear=object_entry_linear_velocity,
                    angular=object_entry_angular_velocity,
                )
                print(
                    f"🎬 {case_name} rigid_entry_velocity "
                    f"lin={object_entry_linear_velocity.tolist()} ang={object_entry_angular_velocity.tolist()}"
                )
            except Exception as exc:
                print(
                    f"🎬 {case_name} rigid_entry_velocity skipped "
                    f"error={type(exc).__name__} detail={exc}"
                )
        for ent in nonrigid_runtime_entities:
            try:
                if hasattr(ent, "set_velocity"):
                    ent.set_velocity(object_entry_linear_velocity.tolist())
                else:
                    ent.set_dofs_velocity(np.concatenate([object_entry_linear_velocity, object_entry_angular_velocity], axis=0).astype(np.float64).tolist())
            except Exception:
                pass

    frames: List[np.ndarray] = []
    save_every = max(1, int(round((1.0 / dt) / fps)))
    # `steps` is treated as the number of exported frames; internally we advance
    # the simulator `save_every` substeps between two saved frames.
    total_sim_steps = max(int(steps), 1) * save_every
    warmup_steps = int(_case_cfg_or_default(runtime_case_cfg, "warmup_steps_override", getattr(args, "warmup_steps", 8)))
    liquid_settle_steps = int(
        _case_cfg_or_default(
            runtime_case_cfg,
            "liquid_settle_steps_override",
            getattr(args, "liquid_settle_steps", 96 if primary_liquid_target is not None else 0),
        )
    )
    liquid_auto_settle_max_steps = int(
        _case_cfg_or_default(
            runtime_case_cfg,
            "liquid_auto_settle_max_steps_override",
            getattr(args, "liquid_auto_settle_max_steps", 160 if primary_liquid_target is not None else 0),
        )
    )
    liquid_auto_settle_min_steps = int(getattr(args, "liquid_auto_settle_min_steps", 24 if primary_liquid_target is not None else 0))
    liquid_auto_settle_stable_steps = int(getattr(args, "liquid_auto_settle_stable_steps", 12))
    liquid_auto_settle_aabb_tol = float(getattr(args, "liquid_auto_settle_aabb_tol", 6e-4))
    liquid_auto_settle_center_tol = float(getattr(args, "liquid_auto_settle_center_tol", 3.5e-4))
    liquid_auto_settle_surface_tol = float(getattr(args, "liquid_auto_settle_surface_tol", 4.5e-4))
    pre_record_delay_steps = int(
        _case_cfg_or_default(
            runtime_case_cfg,
            "pre_record_delay_steps_override",
            getattr(args, "pre_record_delay_steps", 24 if primary_liquid_target is not None else 0),
        )
    )
    pre_impact_record_steps = int(getattr(args, "pre_impact_record_steps", 24 if primary_liquid_target is not None else 0))
    initial_still_frames = int(
        _case_cfg_or_default(
            runtime_case_cfg,
            "initial_still_frames_override",
            getattr(args, "initial_still_frames", 3 if primary_liquid_target is not None else 0),
        )
    )
    if custom_runtime_objects and not liquid_scene:
        warmup_steps = 0
        liquid_settle_steps = 0
        liquid_auto_settle_max_steps = 0
        pre_record_delay_steps = 0
        pre_impact_record_steps = 0
    warmup_total = max(0, warmup_steps, liquid_settle_steps)
    for custom_rec in custom_runtime_objects:
        custom_rec["rest_pos"] = np.asarray(custom_rec["start_pos"], dtype=np.float64).copy()
    for _ in range(warmup_total):
        for custom_rec in custom_runtime_objects:
            rest_pos = custom_rec.get("rest_pos")
            if rest_pos is None or not hasattr(custom_rec["entity"], "set_pos"):
                continue
            custom_rec["entity"].set_pos(tuple(np.asarray(rest_pos, dtype=np.float64).tolist()))
            _apply_custom_runtime_velocity(custom_rec["entity"], np.zeros(6, dtype=np.float64))
        scene.step()

    extra_liquid_settle_taken = 0
    if primary_liquid_entity is not None and liquid_auto_settle_max_steps > 0:
        prev_summary = _entity_particle_settle_summary(primary_liquid_entity)
        prev_aabb = prev_summary["aabb"] if prev_summary is not None else _entity_aabb_numpy(primary_liquid_entity)
        stable_count = 0
        last_aabb_delta = float("inf")
        last_center_delta = float("inf")
        last_surface_delta = float("inf")
        for settle_idx in range(liquid_auto_settle_max_steps):
            for custom_rec in custom_runtime_objects:
                rest_pos = custom_rec.get("rest_pos")
                if rest_pos is None or not hasattr(custom_rec["entity"], "set_pos"):
                    continue
                custom_rec["entity"].set_pos(tuple(np.asarray(rest_pos, dtype=np.float64).tolist()))
                _apply_custom_runtime_velocity(custom_rec["entity"], np.zeros(6, dtype=np.float64))
            scene.step()
            extra_liquid_settle_taken += 1
            curr_summary = _entity_particle_settle_summary(primary_liquid_entity)
            curr_aabb = curr_summary["aabb"] if curr_summary is not None else _entity_aabb_numpy(primary_liquid_entity)
            if prev_aabb is None or curr_aabb is None:
                prev_summary = curr_summary
                prev_aabb = curr_aabb
                stable_count = 0
                continue

            aabb_delta = float(np.max(np.abs(curr_aabb - prev_aabb)))
            center_delta = float("inf")
            surface_delta = float("inf")
            if prev_summary is not None and curr_summary is not None:
                center_delta = float(np.max(np.abs(curr_summary["center"] - prev_summary["center"])))
                surface_delta = float(np.max(np.abs(curr_summary["z_quantiles"] - prev_summary["z_quantiles"])))

            last_aabb_delta = aabb_delta
            last_center_delta = center_delta
            last_surface_delta = surface_delta

            stable_now = (
                aabb_delta <= liquid_auto_settle_aabb_tol
                and center_delta <= liquid_auto_settle_center_tol
                and surface_delta <= liquid_auto_settle_surface_tol
            )

            prev_summary = curr_summary
            prev_aabb = curr_aabb
            if stable_now:
                stable_count += 1
            else:
                stable_count = 0
            if settle_idx + 1 >= liquid_auto_settle_min_steps and stable_count >= liquid_auto_settle_stable_steps:
                break
        print(
            f"🫧 liquid_settle warmup={warmup_total} extra={extra_liquid_settle_taken} "
            f"aabb_delta={last_aabb_delta:.6f} center_delta={last_center_delta:.6f} "
            f"surface_delta={last_surface_delta:.6f}"
        )

    if pre_record_delay_steps > 0:
        for _ in range(pre_record_delay_steps):
            for custom_rec in custom_runtime_objects:
                rest_pos = custom_rec.get("rest_pos")
                if rest_pos is None or not hasattr(custom_rec["entity"], "set_pos"):
                    continue
                custom_rec["entity"].set_pos(tuple(np.asarray(rest_pos, dtype=np.float64).tolist()))
                _apply_custom_runtime_velocity(custom_rec["entity"], np.zeros(6, dtype=np.float64))
            scene.step()
        print(f"🫧 pre_record_delay_steps={pre_record_delay_steps}")

    freeze_liquid_without_striker = bool(
        getattr(args, "freeze_liquid_without_striker", True) and primary_liquid_entity is not None and disable_striker
    )
    frozen_liquid_pos = _entity_particles_numpy(primary_liquid_entity, kind="pos") if freeze_liquid_without_striker else None
    if freeze_liquid_without_striker and frozen_liquid_pos is not None:
        try:
            primary_liquid_entity.set_particles_pos(frozen_liquid_pos)
            primary_liquid_entity.set_particles_vel(np.zeros_like(frozen_liquid_pos, dtype=np.float64))
            print(f"🫧 freeze_liquid_without_striker n_particles={int(frozen_liquid_pos.shape[0])}")
        except Exception as exc:
            print(f"🫧 freeze_liquid_without_striker skipped error={type(exc).__name__}")
            freeze_liquid_without_striker = False

    camera_cfg = {
        "pos": cam_pos.astype(np.float64).tolist(),
        "lookat": lookat.astype(np.float64).tolist(),
        "up": cam_up.astype(np.float64).tolist(),
        "fov": float(cam_fov),
        "res": [int(EXPORT_CAMERA_RESOLUTION[0]), int(EXPORT_CAMERA_RESOLUTION[1])],
        "tag": camera_tag or None,
        "override_applied": bool(camera_override_applied),
        "model": "pinhole",
    }
    cam_intrinsics = camera_intrinsics_dict(
        cam,
        fallback_res=tuple(EXPORT_CAMERA_RESOLUTION),
        fallback_fov_deg=float(cam_fov),
    )
    initial_render = cam.render(rgb=True, depth=True, segmentation=True, normal=False)
    if not isinstance(initial_render, tuple) or len(initial_render) < 3:
        raise RuntimeError("Unexpected camera render output.")
    gravity_vec = np.array([0.0, 0.0, float(gravity_z)], dtype=np.float64)
    physics_track_object_ids: Optional[List[str]] = None
    physics_track_object_types: Optional[List[str]] = None
    physics_track_object_sources: Optional[List[str]] = None
    physics_com_frames: List[np.ndarray] = []
    physics_orientation_frames: List[np.ndarray] = []
    physics_linear_vel_frames: List[np.ndarray] = []
    physics_angular_vel_frames: List[np.ndarray] = []
    physics_kinetic_frames: List[np.float32] = []
    physics_potential_frames: List[np.float32] = []
    physics_total_frames: List[np.float32] = []
    physics_kinetic_trans_frames: List[np.float32] = []
    physics_kinetic_rot_frames: List[np.float32] = []
    physics_potential_gravity_frames: List[np.float32] = []
    prev_physics_state: Optional[Dict[str, Any]] = None
    rgb_frames: List[np.ndarray] = []
    depth_metric_frames: List[np.ndarray] = []
    depth_frames: List[np.ndarray] = []
    seg_frames: List[np.ndarray] = []
    object_aabb_frames: List[List[Optional[np.ndarray]]] = []
    environment_contact_frames: List[List[Dict[str, Any]]] = []
    sample_object_ids: Optional[np.ndarray] = None
    sample_seg_ids: Optional[np.ndarray] = None
    sample_object_types: Optional[List[str]] = None
    sample_object_sources: Optional[List[str]] = None
    sample_object_names: Optional[List[str]] = None
    sample_object_role_hints: Optional[List[Optional[str]]] = None
    sample_object_motion_type_hints: Optional[List[Optional[str]]] = None
    sample_object_motion_group_hints: Optional[List[Optional[str]]] = None

    def _record_physics_frame(rendered: Any) -> None:
        nonlocal physics_track_object_ids, physics_track_object_types, physics_track_object_sources, prev_physics_state
        nonlocal sample_object_ids, sample_seg_ids, sample_object_types, sample_object_sources, sample_object_names
        nonlocal sample_object_role_hints, sample_object_motion_type_hints, sample_object_motion_group_hints
        if not isinstance(rendered, tuple) or len(rendered) < 3:
            raise RuntimeError("Unexpected camera render output.")
        record_idx = len(rgb_frames)
        rgb_raw, depth_raw, seg_raw = rendered[0], rendered[1], rendered[2]
        rgb_frame = rgb_to_uint8(rgb_raw)
        depth_metric_frame = metric_depth_map(depth_raw)
        depth_frame = normalize_depth_map(
            depth_metric_frame,
            near=float(cam_intrinsics["near"]),
            far=float(cam_intrinsics["far"]),
        )
        state = _collect_case_physics_state(
            prepared=prepared,
            articulated_ent=articulated_ent,
            nonrigid_runtime_entities=nonrigid_runtime_entities,
            part_specs=part_specs,
            custom_runtime_objects=custom_runtime_objects,
            gravity_vec=gravity_vec,
            runtime_case_cfg=runtime_case_cfg,
        )
        if physics_track_object_ids is None:
            physics_track_object_ids = list(state["object_ids"])
            physics_track_object_types = list(state["object_types"])
            physics_track_object_sources = list(state["object_sources"])
            sample_object_ids = np.asarray(state["track_object_ids"], dtype=np.int32)
            sample_seg_ids = np.asarray(state["seg_ids"], dtype=np.int32)
            sample_object_types = list(state["object_types"])
            sample_object_sources = list(state["object_sources"])
            sample_object_names = list(state["object_names"])
            sample_object_role_hints = list(state.get("object_role_hints", []))
            sample_object_motion_type_hints = list(state.get("object_motion_type_hints", []))
            sample_object_motion_group_hints = list(state.get("object_motion_group_hints", []))
        seg_mapping = build_segmentation_mapping(scene, state["segmentation_entities"], state["track_object_ids"])
        seg_frame = remap_segmentation(seg_raw, seg_mapping)
        physics_com_frames.append(np.asarray(state["com_pos"], dtype=np.float32))
        physics_orientation_frames.append(np.asarray(state["orientation_quat"], dtype=np.float32))
        physics_linear_vel_frames.append(np.asarray(state["linear_vel"], dtype=np.float32))
        physics_angular_vel_frames.append(np.asarray(state["angular_vel"], dtype=np.float32))
        physics_kinetic_frames.append(np.float32(state["kinetic_energy"]))
        physics_potential_frames.append(np.float32(state["potential_energy"]))
        physics_total_frames.append(np.float32(state["total_energy"]))
        physics_kinetic_trans_frames.append(np.float32(state["kinetic_trans"]))
        physics_kinetic_rot_frames.append(np.float32(state["kinetic_rot"]))
        physics_potential_gravity_frames.append(np.float32(state["potential_gravity"]))
        rgb_frames.append(rgb_frame)
        depth_metric_frames.append(depth_metric_frame.astype(np.float32))
        depth_frames.append(depth_frame.astype(np.float32))
        seg_frames.append(seg_frame.astype(np.int32))
        object_aabb_frames.append(state["object_aabbs"])
        environment_contact_frames.append(state["environment_contacts"])
        prev_physics_state = state

    _record_physics_frame(initial_render)
    frames.append(np.asarray(rgb_frames[-1]))
    for _ in range(max(0, initial_still_frames)):
        frames.append(np.asarray(rgb_frames[-1]))
        physics_com_frames.append(np.asarray(physics_com_frames[-1], dtype=np.float32))
        physics_orientation_frames.append(np.asarray(physics_orientation_frames[-1], dtype=np.float32))
        physics_linear_vel_frames.append(np.asarray(physics_linear_vel_frames[-1], dtype=np.float32))
        physics_angular_vel_frames.append(np.asarray(physics_angular_vel_frames[-1], dtype=np.float32))
        physics_kinetic_frames.append(np.float32(physics_kinetic_frames[-1]))
        physics_potential_frames.append(np.float32(physics_potential_frames[-1]))
        physics_total_frames.append(np.float32(physics_total_frames[-1]))
        physics_kinetic_trans_frames.append(np.float32(physics_kinetic_trans_frames[-1]))
        physics_kinetic_rot_frames.append(np.float32(physics_kinetic_rot_frames[-1]))
        physics_potential_gravity_frames.append(np.float32(physics_potential_gravity_frames[-1]))
        rgb_frames.append(np.asarray(rgb_frames[-1]))
        depth_metric_frames.append(np.asarray(depth_metric_frames[-1], dtype=np.float32))
        depth_frames.append(np.asarray(depth_frames[-1], dtype=np.float32))
        seg_frames.append(np.asarray(seg_frames[-1], dtype=np.int32))
        object_aabb_frames.append(list(object_aabb_frames[-1]))
        environment_contact_frames.append(list(environment_contact_frames[-1]))

    custom_objects_released = False
    if custom_runtime_objects and pre_impact_record_steps <= 0:
        for custom_rec in custom_runtime_objects:
            rest_pos = custom_rec.get("rest_pos")
            if rest_pos is not None and hasattr(custom_rec["entity"], "set_pos"):
                custom_rec["entity"].set_pos(tuple(np.asarray(rest_pos, dtype=np.float64).tolist()))
            _apply_custom_runtime_velocity(custom_rec["entity"], np.asarray(custom_rec["velocity6"], dtype=np.float64))
        custom_objects_released = True

    # # 2. 斜上侧抛
    # striker.set_dofs_velocity([2.5, 0.0, 1.2, 0.0, 0.0, 0.0])

    # # 3. 朝 y 方向侧抛
    # striker.set_dofs_velocity([0.0, -2.0, 0.0, 0.0, 0.0, 0.0])

    # # 4. 带自旋的侧抛
    # striker.set_dofs_velocity([2.5, 0.0, 0.3, 0.0, 20.0, 0.0])

    for t in range(total_sim_steps):
        if freeze_liquid_without_striker and frozen_liquid_pos is not None:
            try:
                primary_liquid_entity.set_particles_pos(frozen_liquid_pos)
                primary_liquid_entity.set_particles_vel(np.zeros_like(frozen_liquid_pos, dtype=np.float64))
            except Exception:
                pass
            rendered = cam.render(rgb=True, depth=True, segmentation=True, normal=False)
            # Frame 0 is already recorded above. Export subsequent frames only
            # after a full `save_every` physics-step interval has elapsed so the
            # video frame timing matches the physical simulation time.
            if (t + 1) % save_every == 0:
                _record_physics_frame(rendered)
                frames.append(np.asarray(rgb_frames[-1]))
            continue
        if custom_runtime_objects and not custom_objects_released:
            for custom_rec in custom_runtime_objects:
                rest_pos = custom_rec.get("rest_pos")
                if rest_pos is not None and hasattr(custom_rec["entity"], "set_pos"):
                    custom_rec["entity"].set_pos(tuple(np.asarray(rest_pos, dtype=np.float64).tolist()))
                _apply_custom_runtime_velocity(custom_rec["entity"], np.zeros(6, dtype=np.float64))
            if t >= pre_impact_record_steps:
                for custom_rec in custom_runtime_objects:
                    rest_pos = custom_rec.get("rest_pos")
                    if rest_pos is not None and hasattr(custom_rec["entity"], "set_pos"):
                        custom_rec["entity"].set_pos(tuple(np.asarray(rest_pos, dtype=np.float64).tolist()))
                    _apply_custom_runtime_velocity(custom_rec["entity"], np.asarray(custom_rec["velocity6"], dtype=np.float64))
                custom_objects_released = True
        scene.step()
        rendered = cam.render(rgb=True, depth=True, segmentation=True, normal=False)
        if (t + 1) % save_every == 0:
            _record_physics_frame(rendered)
            frames.append(np.asarray(rgb_frames[-1]))

    video_tag = f"preview_{case_name}_{args.ball_posx}"
    debug_suffix = []
    if debug_hide_rigid_visuals:
        debug_suffix.append("norigidvis")
    if debug_disable_free_soft:
        debug_suffix.append("nofreesoft")
    if debug_highlight_anchored_soft:
        debug_suffix.append("highlightanchored")
    if np.linalg.norm(debug_detach_anchored_offset) > 0.0:
        debug_suffix.append(
            "detachanchored"
            + "-".join(f"{v:.2f}".replace(".", "p").replace("-", "m") for v in debug_detach_anchored_offset.tolist())
        )
    if debug_spread_soft_parts:
        debug_suffix.append("spreadsoft")
    if debug_pid_offsets:
        for pid in sorted(debug_pid_offsets):
            vec = debug_pid_offsets[pid]
            debug_suffix.append(
                f"pidoffset{pid}-" + "-".join(f"{v:.2f}".replace(".", "p").replace("-", "m") for v in vec.tolist())
            )
    for pid in sorted(debug_pid_E_scale):
        debug_suffix.append(f"pidEscale{pid}-{debug_pid_E_scale[pid]:.2f}".replace(".", "p"))
    for pid in sorted(debug_pid_nu_override):
        debug_suffix.append(f"pidnu{pid}-{debug_pid_nu_override[pid]:.2f}".replace(".", "p"))
    for pid in sorted(debug_pid_sampler):
        debug_suffix.append(f"pidsampler{pid}-{debug_pid_sampler[pid]}")
    if abs(camera_distance_mult - 1.0) > 1e-6:
        debug_suffix.append(f"camdist{camera_distance_mult:.2f}".replace(".", "p"))
    if anchored_soft_mesh_source != "runtime":
        debug_suffix.append(f"anchormesh{anchored_soft_mesh_source}")
    if mpm_vis_mode != "visual":
        debug_suffix.append(f"mpmvis{mpm_vis_mode}")
    if abs(gravity_z + 9.81) > 1e-6:
        debug_suffix.append(f"gravz{gravity_z:.2f}".replace(".", "p").replace("-", "m"))
    if camera_tag:
        debug_suffix.append(f"cam{camera_tag}")
    if debug_suffix:
        video_tag = f"{video_tag}_{'_'.join(debug_suffix)}"
    if part_pid_filter_raw:
        video_tag = f"{video_tag}_pids{part_pid_filter_raw.replace(',', '-')}"
    if skip_rigid_skeleton:
        video_tag = f"{video_tag}_norigid"
    video_path = preview_dir / f"{video_tag}.mp4"
    if frames:
        imageio.mimwrite(video_path, frames, fps=fps, quality=8)
    if sample_object_ids is None or sample_seg_ids is None or sample_object_types is None or sample_object_sources is None:
        raise RuntimeError("No tracked objects were recorded for export.")

    com_pos_arr = np.stack(physics_com_frames, axis=0).astype(np.float32)
    orientation_quat_arr = np.stack(physics_orientation_frames, axis=0).astype(np.float32)
    linear_vel_arr = np.stack(physics_linear_vel_frames, axis=0).astype(np.float32)
    angular_vel_arr = np.stack(physics_angular_vel_frames, axis=0).astype(np.float32)
    depth_metric_arr = np.stack(depth_metric_frames, axis=0).astype(np.float32)
    depth_norm_arr = np.stack(depth_frames, axis=0).astype(np.float32)
    seg_arr = np.stack(seg_frames, axis=0).astype(np.int32)
    object_aabb_arr = object_aabb_frames
    contact_graph_frames = []
    environment_contact_events: List[Dict[str, Any]] = []
    environment_names = ["ground"]
    environment_name_to_index = {name: idx for idx, name in enumerate(environment_names)}
    previous_env_contact = np.zeros((sample_object_ids.shape[0], len(environment_names)), dtype=np.uint8)
    for frame_idx, frame_aabbs in enumerate(object_aabb_arr):
        frame_graph, frame_env_contacts = _contact_graph_with_environment(
            frame_aabbs,
            object_ids=sample_object_ids.tolist(),
            ground_height=0.0,
        )
        contact_graph_frames.append(frame_graph)
        frame_env_contact = np.zeros((sample_object_ids.shape[0], len(environment_names)), dtype=np.uint8)
        for env_contact in frame_env_contacts:
            obj_idx = int(env_contact["object_idx"])
            env_name = str(env_contact.get("environment_name", "ground"))
            env_idx = environment_name_to_index[env_name]
            frame_env_contact[obj_idx, env_idx] = 1
            # Treat frame-0 ground contact as support contact, not a collision event.
            if frame_idx <= 0 or previous_env_contact[obj_idx, env_idx] != 0:
                continue
            environment_contact_events.append(
                {
                    "event_id": len(environment_contact_events),
                    "participants": [int(env_contact["object_id"]), int(env_contact["environment_id"])],
                    "object_indices": [obj_idx, -1],
                    "frame_idx": int(frame_idx),
                    "start_frame": int(frame_idx),
                    "peak_frame": int(frame_idx),
                    "end_frame": int(frame_idx),
                    "impulse_peak": float(env_contact.get("impulse_peak", 0.0)),
                    "contact_duration": 1,
                    "environment_name": env_name,
                }
            )
        previous_env_contact = frame_env_contact
    contact_graph_arr = np.stack(contact_graph_frames, axis=0).astype(np.uint8)
    contact_impulse_arr = np.zeros_like(contact_graph_arr, dtype=np.float32)
    frame_phase_arr, event_windows, collision_events = _summarize_contact_windows(contact_graph_arr, sample_object_ids)
    env_event_windows = summarize_environment_contact_windows(environment_contact_events)
    collision_events.extend(environment_contact_events)
    anchor_targets = compute_anchor_targets(
        seg_frames=seg_arr,
        depth_metric_frames=depth_metric_arr,
        com_pos_frames=com_pos_arr,
        object_ids=sample_object_ids,
        seg_ids=sample_seg_ids,
        camera_cfg=camera_cfg,
        cam_intrinsics=cam_intrinsics,
    )
    # flow_arr = _build_flow_fallback(
    #     com_uv=anchor_targets["com_uv"],
    #     visibility_mask=anchor_targets["visibility_mask"],
    #     seg_frames=seg_arr,
    # )
    scene_composition, object_count_bucket = _scene_layout_from_sources(scene_label, sample_object_sources)
    has_custom_object = any(str(src) in {"custom_object", "physxnet_aux"} for src in sample_object_sources)
    interaction_pattern = _interaction_pattern_from_case(
        scene_label=scene_label,
        scene_composition=scene_composition,
        object_sources=sample_object_sources,
        apply_object_entry_velocity=bool(apply_object_entry_velocity),
    )
    sample_name = f"{prepared.object_id}__{case_name}"
    if simple_case_resample_index > 0:
        sample_name = f"{sample_name}__rs{simple_case_resample_index:02d}"
    if camera_tag:
        sample_name = f"{sample_name}__cam_{camera_tag}"
    case_dir = output_root / "train" / "rigid" / scene_composition / object_count_bucket / sample_name
    existing_meta_path = case_dir / "meta.json"
    if not existing_meta_path.exists():
        existing_meta_path = case_dir / "metadata.json"
    if case_dir.exists() and existing_meta_path.exists():
        print(f"SKIP {case_dir}")
        try:
            scene.destroy()
        except Exception:
            pass
        return str(existing_meta_path)
    prepare_case_output_dirs(case_dir)
    scene_input = {
        "object_id": str(prepared.object_id),
        "sample_name": sample_name,
        "case_name": case_name,
        "case_id": int(case_id),
        "case_variant_index": int(case_variant_index),
        "scene_label": scene_label,
        "simulator_mode": str(getattr(args, "simulator_mode", "rigid")),
        "simulator_type": "rigid",
        "scene_composition": scene_composition,
        "interaction_pattern": interaction_pattern,
        "object_count_bucket": object_count_bucket,
        "camera": camera_cfg,
        "camera_tag": camera_tag or None,
        "runtime_main_object_scale": float(runtime_main_object_scale),
        "runtime_default_striker_radius_mult": float(runtime_striker_radius_mult),
        "auto_visibility_scale": auto_visibility_scale_info,
        "entry_linear_velocity": object_entry_linear_velocity.tolist(),
        "entry_angular_velocity": object_entry_angular_velocity.tolist(),
        "use_entry_motion": bool(apply_object_entry_velocity),
        "object_fixed": bool(runtime_object_fixed),
        "gravity": [0.0, 0.0, float(gravity_z)],
        "striker_speed_mps": float(runtime_striker_speed),
        "counterfactual": counterfactual_meta if counterfactual_meta else None,
        "rigid_restitution_override": None if rigid_restitution_override is None else float(rigid_material_cfg["restitution"]),
    }
    (case_dir / "scene_input.json").write_text(json.dumps(scene_input, ensure_ascii=False, indent=2), encoding="utf-8")
    for frame_idx, rgb_frame in enumerate(rgb_frames):
        imageio.imwrite(case_dir / "rgb" / f"frame_{frame_idx:03d}.png", rgb_frame)
    for frame_idx, depth_frame in enumerate(depth_norm_arr):
        imageio.imwrite(case_dir / "depth" / f"frame_{frame_idx:03d}.png", depth_to_uint8(depth_frame))
    np.save(case_dir / "physics" / "depth_metric.npy", depth_metric_arr)
    # np.save(case_dir / "physics" / "depth_normalized.npy", depth_norm_arr)
    np.save(case_dir / "physics" / "seg.npy", seg_arr)
    np.save(case_dir / "physics" / "contact_graph.npy", contact_graph_arr)
    np.save(case_dir / "physics" / "contact_impulse.npy", contact_impulse_arr)
    np.save(case_dir / "physics" / "frame_phase.npy", frame_phase_arr.astype(np.int8))
    # np.save(case_dir / "physics" / "flow.npy", flow_arr.astype(np.float32))
    np.savez_compressed(case_dir / "physics" / "anchor_targets.npz", **anchor_targets)
    np.savez_compressed(
        case_dir / "physics" / "rigid_kinematics.npz",
        object_ids=sample_object_ids.astype(np.int32),
        seg_ids=sample_seg_ids.astype(np.int32),
        com_pos=com_pos_arr,
        orientation_quat=orientation_quat_arr,
        linear_vel=linear_vel_arr,
        angular_vel=angular_vel_arr,
        com_uv=anchor_targets["com_uv"],
        bbox_xyxy=anchor_targets["bbox_xyxy"],
        visibility_mask=anchor_targets["visibility_mask"],
        kinetic_energy=np.asarray(physics_kinetic_frames, dtype=np.float32),
        potential_energy=np.asarray(physics_potential_frames, dtype=np.float32),
        total_energy=np.asarray(physics_total_frames, dtype=np.float32),
    )
    np.savez_compressed(
        case_dir / "physics" / "energy.npz",
        kinetic_trans=np.asarray(physics_kinetic_trans_frames, dtype=np.float32),
        kinetic_rot=np.asarray(physics_kinetic_rot_frames, dtype=np.float32),
        potential_gravity=np.asarray(physics_potential_gravity_frames, dtype=np.float32),
        mechanical_total=np.asarray(physics_total_frames, dtype=np.float32),
    )
    save_vis_video(
        case_dir / "visualizations" / "depth_vis.mp4",
        [depth_to_vis(frame, near=float(cam_intrinsics["near"]), far=float(cam_intrinsics["far"])) for frame in depth_metric_arr],
        fps=fps,
    )
    rgb_video_frames = [np.asarray(frame) for frame in rgb_frames]
    imageio.mimwrite(case_dir / "videos" / "rgb.mp4", rgb_video_frames, fps=fps, quality=8)
    imageio.mimwrite(case_dir / "videos" / "depth.mp4", [depth_to_uint8(frame) for frame in depth_norm_arr], fps=fps, quality=8)
    with open(case_dir / "physics" / "collision_events.json", "w", encoding="utf-8") as f:
        json.dump(collision_events, f, ensure_ascii=False, indent=2)
    with open(case_dir / "physics" / "event_windows.json", "w", encoding="utf-8") as f:
        json.dump(event_windows + env_event_windows, f, ensure_ascii=False, indent=2)
    properties_payload = {
        "object_ids": sample_object_ids.astype(np.int32).tolist(),
        "sampled_restitution": [None if rigid_restitution_override is None else float(rigid_material_cfg["restitution"]) for _ in sample_object_ids],
        "effective_restitution_used": [float(rigid_material_cfg["restitution"]) for _ in sample_object_ids],
        "counterfactual": counterfactual_meta if counterfactual_meta else None,
    }
    with open(case_dir / "physics" / "properties.json", "w", encoding="utf-8") as f:
        json.dump(properties_payload, f, ensure_ascii=False, indent=2)
    metadata_payload = {
        "scene_id": sample_name,
        "object_id": str(prepared.object_id),
        "case_id": int(case_id),
        "case_variant_index": int(case_variant_index),
        "case_name": case_name,
        "seed": int(case_seed_for_runtime),
        "split": "train",
        "family": "physxnet_single_object",
        "simulator_type": "rigid",
        "scene_composition": scene_composition,
        "interaction_pattern": interaction_pattern,
        "object_count_bucket": object_count_bucket,
        "num_objects": int(sample_object_ids.shape[0]),
        "frames": int(com_pos_arr.shape[0]),
        "resolution": [int(EXPORT_CAMERA_RESOLUTION[0]), int(EXPORT_CAMERA_RESOLUTION[1])],
        "motion_category": str(scene_label),
        "sample_role": "counterfactual_negative" if counterfactual_meta else "factual",
        "counterfactual": counterfactual_meta if counterfactual_meta else None,
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
            "dt": float(dt),
            "substeps": int(runtime_substeps),
            "steps_per_frame": int(save_every),
            "frame_dt": float(dt) * float(save_every),
            "video_fps": float(fps),
            "gravity": [0.0, 0.0, float(gravity_z)],
        },
        "camera": camera_cfg,
        "camera_tag": camera_tag or None,
        "camera_intrinsics": cam_intrinsics,
        "runtime_main_object_scale": float(runtime_main_object_scale),
        "runtime_default_striker_radius_mult": float(runtime_striker_radius_mult),
        "auto_visibility_scale": auto_visibility_scale_info,
        "objects": [
            (
                lambda motion_fields, idx=idx: {
                    "object_id": int(sample_object_ids[idx]),
                    "seg_id": int(sample_seg_ids[idx]),
                    "entity_type": "rigid_assembly" if idx == 0 else str(sample_object_types[idx]),
                    "role": motion_fields["role"],
                    "object_motion_type": motion_fields["object_motion_type"],
                    "object_motion_group": motion_fields["object_motion_group"],
                    "motion_type": motion_fields["object_motion_type"],
                    "motion_group": motion_fields["object_motion_group"],
                    "source_tag": str(sample_object_sources[idx]),
                }
            )(
                _object_motion_fields(
                    object_index=idx,
                    source_tag=str(sample_object_sources[idx]),
                    scene_label=scene_label,
                    scene_composition=scene_composition,
                    has_custom_object=has_custom_object,
                    apply_object_entry_velocity=bool(apply_object_entry_velocity),
                    role_hint=None if sample_object_role_hints is None or idx >= len(sample_object_role_hints) else sample_object_role_hints[idx],
                    motion_type_hint=None if sample_object_motion_type_hints is None or idx >= len(sample_object_motion_type_hints) else sample_object_motion_type_hints[idx],
                    motion_group_hint=None if sample_object_motion_group_hints is None or idx >= len(sample_object_motion_group_hints) else sample_object_motion_group_hints[idx],
                )
            )
            for idx in range(sample_object_ids.shape[0])
        ],
        "environment_entities": [
            {
                "name": "ground",
                "special_id": int(ENVIRONMENT_SPECIAL_IDS["ground"]),
                "entity_type": "container",
            }
        ],
        "outputs": {
            "metadata": "meta.json",
            "scene_input": "scene_input.json",
            "rgb_video": "videos/rgb.mp4",
            "depth_video": "videos/depth.mp4",
            "depth_metric": "physics/depth_metric.npy",
            # "depth_normalized": "physics/depth_normalized.npy",
            "segmentation": "physics/seg.npy",
            # "flow": "physics/flow.npy",
            "anchor_targets": "physics/anchor_targets.npz",
            "rigid_kinematics": "physics/rigid_kinematics.npz",
            "energy": "physics/energy.npz",
            "properties": "physics/properties.json",
            "contact_graph": "physics/contact_graph.npy",
            "contact_impulse": "physics/contact_impulse.npy",
            "frame_phase": "physics/frame_phase.npy",
            "event_windows": "physics/event_windows.json",
            "depth_visualization_video": "visualizations/depth_vis.mp4",
        },
        "has_depth_metric": True,
        "has_seg": True,
        "has_contact_graph": True,
        "status": "ok",
    }
    for idx, obj_meta in enumerate(metadata_payload["objects"]):
        if str(sample_object_sources[idx]) == "physxnet_aux":
            obj_meta["dataset_source"] = "PhysXNet"
            obj_meta["source_object_id"] = str(sample_object_names[idx]) if idx < len(sample_object_names) else str(sample_object_ids[idx])
        elif str(sample_object_sources[idx]) == "custom_object":
            obj_meta["dataset_source"] = "Custom"
            obj_meta["source_object_id"] = str(sample_object_names[idx]) if idx < len(sample_object_names) else str(sample_object_ids[idx])
        else:
            obj_meta["dataset_source"] = "PhysXNet"
            obj_meta["source_object_id"] = str(prepared.object_id)
    (case_dir / "meta.json").write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        scene.destroy()
    except Exception:
        pass

    print(f"GENERATED {case_dir / 'videos' / 'rgb.mp4'}")
    return str(case_dir / "meta.json")
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one PhysXNet object into a strict-per-part Genesis-ready asset folder and optional preview simulation.")
    parser.add_argument(
        "--physx_root",
        type=str,
        default="/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet",
        help="PhysXNet root. Defaults to /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet",
    )
    parser.add_argument("--version", type=str, default="version_1", help="Dataset version folder name")
    parser.add_argument("--object_id", type=str, default=None, help="Object ID, e.g. 48610. If omitted, the script randomly samples objects from the dataset.")
    parser.add_argument("--num_random_objects", type=int, default=1, help="How many objects to randomly sample when --object_id is not provided")
    parser.add_argument("--random_object_seed", type=int, default=20260414, help="Seed used when randomly sampling objects if --object_id is not provided")
    parser.add_argument("--output_root", type=str, required=True, help="Output directory")
    parser.add_argument("--json_override", type=str, default=None, help="Optional explicit JSON path. If set, this JSON is used instead of finaljson/<object_id>.json")
    parser.add_argument(
        "--simulator_mode",
        type=str,
        default="rigid",
        choices=["rigid", "mpm"],
        help="Top-level export mode. `rigid` forces all parts into rigid-only simulation/export; `mpm` is reserved for the next step.",
    )
    parser.add_argument("--voxel_pitch", type=float, default=0.025, help="Voxel size in meters for soft-part fill")
    parser.add_argument("--fallback_density_kgm3", type=float, default=800.0, help="Fallback density only for parts whose JSON lacks density")
    parser.add_argument("--default_friction", type=float, default=0.55, help="Runtime fallback friction only when JSON friction is absent")
    parser.add_argument("--run_genesis", action="store_true", help="Also run a small Genesis demo and render preview.mp4")
    parser.add_argument("--steps", type=int, default=240, help="Simulation steps for preview")
    parser.add_argument("--dt", type=float, default=0.005, help="Simulation dt")
    parser.add_argument("--substeps", type=int, default=10, help="Simulation substeps")
    parser.add_argument("--fps", type=int, default=24, help="Preview video fps")
    parser.set_defaults(liquid_free_surface=True)
    parser.add_argument("--disable_liquid_free_surface", dest="liquid_free_surface", action="store_false", help="Use the older more constrained liquid setup with extra seals/guards and denser cavity filling")
    parser.add_argument("--liquid_sampler", type=str, default=None, help="Override Genesis SPH liquid sampler, e.g. pbs or regular")
    parser.add_argument("--liquid_stiffness", type=float, default=35000.0, help="Runtime SPH stiffness used for liquid parts")
    parser.add_argument("--liquid_exponent", type=float, default=7.0, help="Runtime SPH equation-of-state exponent used for liquid parts")
    parser.add_argument("--liquid_viscosity", type=float, default=0.0015, help="Runtime SPH viscosity used for liquid parts")
    parser.add_argument("--liquid_surface_tension", type=float, default=0.002, help="Runtime SPH surface tension used for liquid parts")
    parser.add_argument("--striker_radius", type=float, default=0.08, help="Radius of the striker sphere")
    parser.add_argument("--striker_speed", type=float, default=2.8, help="+X initial speed of the striker sphere")
    parser.add_argument("--static_striker_speed_v2_scale", type=float, default=1.6, help="Multiplier applied to the default striker speed when generating static-target *_v2 cases")
    parser.add_argument("--striker_drop_top", action="store_true", help="Spawn the default striker above the object and let it fall along -Z")
    parser.add_argument("--striker_drop_height", type=float, default=0.32, help="Extra height above the object top when --striker_drop_top is enabled")
    parser.add_argument("--striker_drop_xy_jitter", type=float, default=0.0, help="Optional XY jitter radius for the top-drop striker start position")
    parser.add_argument("--custom_object_prefix", type=str, default="custom_ball", help="Prefix used when naming custom spawned objects in case metadata")
    parser.add_argument("--custom_object_count_min", type=int, default=1, help="Minimum number of custom spawned objects per case")
    parser.add_argument("--custom_object_count_max", type=int, default=3, help="Maximum number of custom spawned objects per case")
    parser.add_argument("--custom_object_spawn_direction", type=str, default="", help="Optional forced spawn direction for custom objects, e.g. top_to_bottom")
    parser.add_argument(
        "--custom_object_source_mix",
        type=str,
        default="primitive,sophy,physxnet",
        help="Comma-separated source mix for inserted MPM objects: primitive,sophy,physxnet",
    )
    parser.add_argument(
        "--custom_object_rigidify_youngs_threshold_pa",
        type=float,
        default=1.0e8,
        help="If a custom object's dataset Young's modulus exceeds this threshold, approximate it as a rigid body while keeping interaction with MPM objects.",
    )
    parser.set_defaults(enable_striker=False)
    parser.add_argument("--enable_striker", action="store_true", help="Explicitly add and release the striker sphere in liquid scenes")
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
    parser.add_argument("--all_parts_youngs_threshold_gpa", type=float, default=None, 
                        help="Record-only threshold for analysis/debug; it no longer rewrites per-part solver/material.")
    parser.add_argument("--anchored_overlap_scale_boost", type=float, default=1.0, help="Extra aggressiveness multiplier for anchored_soft overlap-driven shrinking")
    parser.add_argument("--debug_detach_anchored_offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Debug-only world offset applied to all anchored_soft parts")
    parser.add_argument("--debug_pid_offset", type=float, nargs=4, action="append", default=[], metavar=("PID", "DX", "DY", "DZ"), help="Debug-only world offset for a specific part id; can be passed multiple times")
    parser.add_argument("--debug_pid_E_scale", type=float, nargs=2, action="append", default=[], metavar=("PID", "SCALE"), help="Debug-only multiply runtime MPM Young's modulus E for a specific part id")
    parser.add_argument("--debug_pid_nu_override", type=float, nargs=2, action="append", default=[], metavar=("PID", "NU"), help="Debug-only override runtime MPM Poisson ratio for a specific part id")
    parser.add_argument("--debug_pid_sampler", type=str, nargs=2, action="append", default=[], metavar=("PID", "SAMPLER"), help="Debug-only override runtime MPM sampler for a specific part id")
    parser.add_argument("--debug_spread_soft_parts", action="store_true", help="Debug-only: spread all soft parts away from the rigid skeleton for inspection")
    parser.add_argument("--debug_soft_spread_gap", type=float, default=0.45, help="Spacing in meters between spread soft parts in debug mode")
    parser.add_argument("--debug_soft_spread_y_offset", type=float, default=0.85, help="Extra negative-y offset applied when spreading soft parts for debug inspection")
    parser.add_argument("--camera_distance_mult", type=float, default=1.0, help="Multiplier for preview camera distance and height")
    parser.add_argument("--camera_pos_override", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Optional explicit simulation camera position override")
    parser.add_argument("--camera_lookat_override", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Optional explicit simulation camera look-at override")
    parser.add_argument("--camera_up_override", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"), help="Optional explicit simulation camera up-vector override")
    parser.add_argument("--camera_fov_override", type=float, default=None, help="Optional explicit simulation camera vertical FOV override in degrees")
    parser.add_argument("--camera_tag", type=str, default="", help="Optional camera tag appended to preview/case outputs when using camera overrides")
    parser.add_argument("--min_projected_bbox_area_px", type=float, default=2500.0, help="If an object's estimated first-frame projected bbox area is below this threshold, auto-scale it up before simulation.")
    parser.add_argument("--max_auto_scale_up_mult", type=float, default=2.5, help="Maximum automatic per-object scale-up multiplier used to satisfy --min_projected_bbox_area_px.")
    parser.add_argument("--debug_hide_rigid_visuals", action="store_true", help="Debug render: hide rigid skeleton visuals but keep collisions")
    parser.add_argument("--disable_rigid_visual_double_sided_shell", action="store_true", help="Disable reversed-face duplication for non-watertight rigid visual meshes")
    parser.add_argument("--debug_disable_free_soft", action="store_true", help="Debug render: skip free_soft parts such as pillows")
    parser.add_argument("--debug_highlight_anchored_soft", action="store_true", help="Debug render: color anchored soft bright red for inspection")
    parser.add_argument("--use_anchored_hybrid", action="store_true", help="Use HybridEntity for anchored_soft. Default keeps anchored_soft as independent soft bodies.")
    parser.add_argument("--anchored_soft_mesh_source", choices=["runtime", "original"], default="runtime", help="Which mesh anchored_soft uses for both simulation and rendering")
    parser.add_argument("--mpm_vis_mode", choices=["visual", "particle"], default="visual", help="How MPM soft bodies are rendered")
    parser.add_argument("--liquid_vis_mode", choices=["particle", "recon"], default="particle", help="How liquid parts are rendered in Genesis previews")
    parser.add_argument("--liquid_recon_backend", choices=["splashsurf", "openvdb"], default="splashsurf", help="Backend used when liquid_vis_mode=recon")
    parser.add_argument("--anchored_constraint_stiffness", type=float, default=8000.0, help="Soft spring constraint stiffness used to keep anchored_soft near its matched rigid link while leaving a thin deformable outer shell; set 0 to disable")
    parser.add_argument("--warmup_steps", type=int, default=8, help="Simulation steps to run before recording frames")
    parser.add_argument("--liquid_settle_steps", type=int, default=96, help="Base pre-roll steps for liquid scenes before checking whether the fluid has settled")
    parser.add_argument("--liquid_auto_settle_max_steps", type=int, default=160, help="Maximum extra settle steps while monitoring liquid motion after the base warmup")
    parser.add_argument("--liquid_auto_settle_min_steps", type=int, default=24, help="Minimum monitored settle steps before allowing the liquid to be considered stable")
    parser.add_argument("--liquid_auto_settle_stable_steps", type=int, default=12, help="Consecutive stable monitored steps required before the liquid is considered settled")
    parser.add_argument("--liquid_auto_settle_aabb_tol", type=float, default=6e-4, help="AABB max-delta threshold for liquid settle detection")
    parser.add_argument("--liquid_auto_settle_center_tol", type=float, default=3.5e-4, help="Center-of-mass max-delta threshold for liquid settle detection")
    parser.add_argument("--liquid_auto_settle_surface_tol", type=float, default=4.5e-4, help="Liquid vertical quantile max-delta threshold for settle detection")
    parser.add_argument("--pre_record_delay_steps", type=int, default=24, help="Extra hidden settle steps after liquid stabilization and before the first recorded frame")
    parser.add_argument("--pre_impact_record_steps", type=int, default=24, help="Simulation steps to record with the striker frozen after settling and before releasing it")
    parser.add_argument("--initial_still_frames", type=int, default=3, help="Number of identical opening frames to duplicate so the video starts from a static settled liquid state")
    parser.add_argument("--freeze_liquid_without_striker", type=int, choices=[0, 1], default=1, help="When no striker is used, keep the settled liquid particles fixed in the bowl during rendering")
    parser.add_argument("--skip_rigid_skeleton", action="store_true", help="Do not add the rigid URDF skeleton to the scene")
    parser.add_argument("--part_pid_filter", type=str, default="", help="Comma-separated part ids to keep in the simulation")
    parser.add_argument("--disable_striker", action="store_true", help="Do not add the striker sphere to the scene")
    parser.add_argument("--prefer_existing_runtime_meshes", action="store_true", help="Reuse precomputed runtime soft meshes from scene_preview/runtime_soft_meshes when available")
    parser.add_argument("--gravity_z", type=float, default=-9.81, help="Scene gravity z value. Use 0.0 for zero-gravity diagnostics")
    parser.set_defaults(skip_liquid_preview=True)
    parser.add_argument("--allow_liquid_preview", dest="skip_liquid_preview", action="store_false", help="Do not skip Genesis preview generation for objects that contain liquid parts")
    parser.add_argument("--num_random_cases", type=int, default=4, help="Number of preview simulation cases to generate per object")
    parser.add_argument(
        "--case_scene_mode",
        choices=["auto", "diverse", "legacy_random"],
        default="auto",
        help="How preview cases are generated. 'diverse' expands into explicit scene templates before falling back to legacy random sampling.",
    )
    parser.add_argument(
        "--generate_six_combo_cases",
        action="store_true",
        help="Generate the six canonical combinations together: 3 striker interaction cases plus 3 matching single-object cases.",
    )
    parser.add_argument(
        "--generate_all_count_motion_cases",
        action="store_true",
        help="For one main object, generate all explicit motion cases across count=1,2,3,4. count=1 has only the main object; count>=2 keeps the yellow striker and adds extra static PhysXNet objects.",
    )
    parser.add_argument(
        "--rigid_target_object_count",
        type=int,
        default=None,
        help="Rigid-only target scene object count. If set, scene count is main object + optional striker + extra static PhysXNet objects up to this total.",
    )
    parser.add_argument(
        "--rigid_count_filter",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of count bundles to run with --generate_all_count_motion_cases, e.g. --rigid_count_filter 1 2.",
    )
    parser.add_argument("--case_index_filter", type=int, nargs="*", default=None, help="Optional subset of generated case indices to execute, e.g. --case_index_filter 0 2")
    parser.add_argument("--case_seed", type=int, default=20260414, help="Base seed used to deterministically randomize preview cases")
    parser.add_argument("--enable_counterfactual_cases", action="store_true", help="Append two counterfactual negative cases per compatible base case: same-scene perturbed impact and no-collision continuation.")
    parser.add_argument("--counterfactual_only", action="store_true", help="Execute only counterfactual cases after they are derived from the requested parent cases; skip running base factual cases.")
    parser.add_argument(
        "--counterfactual_no_collision_gravity_z",
        type=float,
        default=-9.81,
        help="Gravity used by the no-collision counterfactual. Defaults to standard gravity so all cases share a unified gravity field.",
    )
    parser.add_argument("--motion_resample_index", type=int, default=0, help="Internal retry index for case900/case901 randomized pose and velocity")
    parser.add_argument("--simple_case_resample_index", type=int, default=0, help="Extra resample index for simple template cases. When >0, seeds and output sample names are varied so repeated generation of case000/001/002/003/901 produces new variants instead of skipping.")
    parser.add_argument(
        "--motion_case_max_retries",
        type=int,
        default=5,
        help="Retry invalid case900/case901 samples by resampling pose/velocity before keeping output; all other cases still run one QA pass.",
    )
    parser.add_argument("--physxnet_volume_threshold_m3", type=float, default=0.20, help="PhysXNet assembled-object AABB volume threshold in m^3. Objects above this stay static on the ground across all cases")
    parser.add_argument("--physxnet_entry_velocity_prob", type=float, default=0.35, help="For PhysXNet objects smaller than the volume threshold, probability that a case enters with initial velocity")
    parser.add_argument("--physxnet_entry_speed_min", type=float, default=0.60, help="Minimum initial entry speed in m/s for moving PhysXNet cases")
    parser.add_argument("--physxnet_entry_speed_max", type=float, default=0.90, help="Maximum initial entry speed in m/s for moving PhysXNet cases")
    parser.add_argument("--physxnet_object_yaw_deg_min", type=float, default=-180.0, help="Minimum initial yaw rotation in degrees for PhysXNet preview cases; roll/pitch remain zero to keep z-up")
    parser.add_argument("--physxnet_object_yaw_deg_max", type=float, default=180.0, help="Maximum initial yaw rotation in degrees for PhysXNet preview cases; roll/pitch remain zero to keep z-up")
    
    
    
    
    
    
    return parser














def _sample_random_object_ids(physx_root: Path, version: str, num_objects: int, seed: int) -> List[str]:
    finaljson_dir = physx_root / version / "finaljson"
    if not finaljson_dir.exists():
        raise FileNotFoundError(finaljson_dir)

    object_ids = sorted(path.stem for path in finaljson_dir.glob("*.json"))
    if not object_ids:
        raise RuntimeError(f"No object metadata found under {finaljson_dir}")

    count = max(1, min(int(num_objects), len(object_ids)))
    rng = random.Random(int(seed))
    if count >= len(object_ids):
        rng.shuffle(object_ids)
        return object_ids
    return rng.sample(object_ids, count)


def _preview_case_bundles(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if bool(getattr(args, "generate_all_count_motion_cases", False)):
        requested_case_filter = getattr(args, "case_index_filter", None)
        if requested_case_filter is None:
            all_case_indices = [0, 1, 2, 3, 5, 6, 7, 100, 101, 102, 210, 211, 220, 221, 230, 231, 900, 901]
        else:
            all_case_indices = [int(idx) for idx in requested_case_filter]
        requested_count_filter = getattr(args, "rigid_count_filter", None)
        if requested_count_filter is None:
            target_counts = [1, 2, 3, 4]
        else:
            target_counts = [int(count) for count in requested_count_filter]
        bundles: List[Dict[str, Any]] = []
        for target_count in target_counts:
            bundles.append(
                {
                    "bundle_name": f"count_{target_count:02d}",
                    "disable_striker": bool(target_count == 1),
                    "case_scene_mode": "diverse",
                    "num_random_cases": max(8, int(getattr(args, "num_random_cases", 8))),
                    "case_index_filter": all_case_indices,
                    "rigid_target_object_count": int(target_count),
                }
            )
        return bundles

    if not bool(getattr(args, "generate_six_combo_cases", False)):
        return [
            {
                "bundle_name": "default",
                "disable_striker": bool(getattr(args, "disable_striker", False)),
                "case_scene_mode": str(getattr(args, "case_scene_mode", "auto")),
                "num_random_cases": int(getattr(args, "num_random_cases", 4)),
                "case_index_filter": None if getattr(args, "case_index_filter", None) is None else [int(idx) for idx in getattr(args, "case_index_filter", None)],
                "rigid_target_object_count": getattr(args, "rigid_target_object_count", None),
            }
        ]

    combo_case_indices = [1, 3, 7]
    return [
        {
            "bundle_name": "with_striker",
            "disable_striker": False,
            "case_scene_mode": "diverse",
            "num_random_cases": max(8, int(getattr(args, "num_random_cases", 8))),
            "case_index_filter": combo_case_indices,
            "rigid_target_object_count": getattr(args, "rigid_target_object_count", None),
        },
        {
            "bundle_name": "single_object",
            "disable_striker": True,
            "case_scene_mode": "diverse",
            "num_random_cases": max(8, int(getattr(args, "num_random_cases", 8))),
            "case_index_filter": combo_case_indices,
            "rigid_target_object_count": 1,
        },
    ]


def _run_single_object(args: argparse.Namespace, object_id: str) -> Dict[str, Any]:
    simulator_mode = str(getattr(args, "simulator_mode", "rigid")).strip().lower()
    rigid_visual_double_sided_shell = not bool(getattr(args, "disable_rigid_visual_double_sided_shell", False))
    if simulator_mode == "rigid":
        # In rigid-only export mode, always keep the double-sided visual proxy
        # for thin scanned meshes to avoid dirty back-face artifacts.
        rigid_visual_double_sided_shell = True

    prepared = prepare_physxnet_object(
        physx_root=Path(args.physx_root),
        version=args.version,
        object_id=str(object_id),
        output_root=Path(args.output_root),
        voxel_pitch=float(args.voxel_pitch),
        json_override=Path(args.json_override) if args.json_override else None,
        object_scale_mult=float(args.object_scale_mult),
        solver_family_override=args.solver_family_override,
        all_parts_youngs_threshold_gpa=args.all_parts_youngs_threshold_gpa,
        rigid_visual_double_sided_shell=rigid_visual_double_sided_shell,
        simulator_mode=simulator_mode,
    )

    preview_video = None
    preview_videos: List[str] = []
    preview_cases: List[Dict[str, Any]] = []
    preview_skipped_reason: Optional[str] = None
    if args.run_genesis:
        if bool(getattr(args, "skip_liquid_preview", True)) and _prepared_object_has_liquid(prepared):
            preview_skipped_reason = "object_contains_liquid"
            print(
                f"⏭️ skip Genesis preview for object {prepared.object_id} "
                f"because liquid parts were detected. Use --allow_liquid_preview to override."
            )
        else:
            preview_dir = Path(prepared.output_dir) / "scene_preview"
            ensure_dir(preview_dir)
            preview_cases_all: List[Dict[str, Any]] = []
            for bundle in _preview_case_bundles(args):
                bundle_args = argparse.Namespace(**vars(args))
                bundle_args.disable_striker = bool(bundle["disable_striker"])
                bundle_args.case_scene_mode = str(bundle["case_scene_mode"])
                bundle_args.num_random_cases = int(bundle["num_random_cases"])
                bundle_args.case_index_filter = list(bundle["case_index_filter"]) if bundle["case_index_filter"] is not None else None
                bundle_args.rigid_target_object_count = bundle.get("rigid_target_object_count", None)

                bundle_cases = build_preview_case_configs(
                    prepared=prepared,
                    output_root=Path(args.output_root),
                    object_fixed=bool(bundle_args.object_fixed),
                    args=bundle_args,
                )
                case_index_filter = getattr(bundle_args, "case_index_filter", None)
                if case_index_filter:
                    requested_case_indices = {int(idx) for idx in case_index_filter}
                    bundle_cases = [
                        case_cfg
                        for case_cfg in bundle_cases
                        if (
                            int(case_cfg.get("case_index", -1)) in requested_case_indices
                            or int(dict(case_cfg.get("counterfactual", {}) or {}).get("parent_case_index", -1)) in requested_case_indices
                        )
                    ]
                    if not bundle_cases:
                        print(
                            f"⏭️ preview bundle={bundle['bundle_name']} has no matched cases for "
                            f"--case_index_filter={sorted(requested_case_indices)}"
                        )
                        continue
                if not bundle_cases:
                    continue

                case_plan_name = "preview_case_plan.json" if bundle["bundle_name"] == "default" else f"preview_case_plan__{bundle['bundle_name']}.json"
                (preview_dir / case_plan_name).write_text(json.dumps(bundle_cases, ensure_ascii=False, indent=2), encoding="utf-8")
                print(
                    f"🧩 preview bundle={bundle['bundle_name']} "
                    f"disable_striker={bundle_args.disable_striker} "
                    f"cases={[int(case_cfg.get('case_index', -1)) for case_cfg in bundle_cases]}"
                )
                for case_cfg in bundle_cases:
                    print(
                        f"🎲 {case_cfg['case_name']} "
                        f"scene={case_cfg.get('scene_label', case_cfg['case_name'])} "
                        f"bbox_vol={case_cfg['object_bbox_volume_est_m3']:.4f} "
                        f"threshold={case_cfg['physxnet_volume_threshold_m3']:.4f} "
                        f"moving={case_cfg['use_entry_motion']} "
                        f"fixed={case_cfg['object_fixed']} "
                        f"gravity={case_cfg.get('gravity_z_override', bundle_args.gravity_z)} "
                        f"offset={case_cfg['placed_pos_offset']} "
                        f"linvel={case_cfg['entry_linear_velocity']}"
                    )
                    case_index = int(case_cfg.get("case_index", -1))
                    enable_motion_retry = case_index in {900, 901}
                    max_motion_attempts = 1
                    if enable_motion_retry:
                        max_motion_attempts = max(1, int(getattr(bundle_args, "motion_case_max_retries", 5)))
                    video_path = None
                    last_motion_metrics: Dict[str, Any] = {}
                    for attempt_idx in range(max_motion_attempts):
                        attempt_args = argparse.Namespace(**vars(bundle_args))
                        attempt_args.motion_resample_index = int(attempt_idx)
                        attempt_cases = build_preview_case_configs(
                            prepared=prepared,
                            output_root=Path(args.output_root),
                            object_fixed=bool(attempt_args.object_fixed),
                            args=attempt_args,
                        )
                        attempt_case_cfg = next(
                            (cfg for cfg in attempt_cases if int(cfg.get("case_index", -1)) == case_index),
                            case_cfg,
                        )
                        video_path = simulate_in_genesis(
                            prepared=prepared,
                            output_root=Path(args.output_root),
                            steps=int(attempt_args.steps),
                            dt=float(attempt_args.dt),
                            substeps=int(attempt_args.substeps),
                            fps=int(attempt_args.fps),
                            default_friction=float(attempt_args.default_friction),
                            object_fixed=bool(attempt_args.object_fixed),
                            striker_radius=float(attempt_args.striker_radius),
                            striker_speed=float(attempt_args.striker_speed),
                            args=attempt_args,
                            case_cfg=attempt_case_cfg,
                        )
                        case_dir = _case_dir_from_metadata_path(video_path)
                        if case_dir is None:
                            print(
                                f"⚠️ motion_case_missing_metadata object={prepared.object_id} "
                                f"case={attempt_case_cfg.get('case_name')} attempt={attempt_idx}"
                            )
                            break
                        last_motion_metrics = _load_single_object_motion_qa(case_dir)
                        last_motion_metrics["attempt_index"] = int(attempt_idx)
                        last_motion_metrics["case_index"] = case_index
                        last_motion_metrics["object_id"] = str(prepared.object_id)
                        _write_motion_qa(case_dir, last_motion_metrics)
                        if bool(last_motion_metrics.get("valid", False)):
                            print(
                                f"✅ motion_case_valid object={prepared.object_id} "
                                f"case={attempt_case_cfg.get('case_name')} attempt={attempt_idx} "
                                f"vis={last_motion_metrics.get('visible_ratio', 0.0):.2f} "
                                f"safe={last_motion_metrics.get('safe_uv_ratio', 0.0):.2f}"
                            )
                            break
                        if attempt_idx < max_motion_attempts - 1:
                            print(
                                f"♻️ motion_case_retry object={prepared.object_id} "
                                f"case={attempt_case_cfg.get('case_name')} attempt={attempt_idx} "
                                f"reasons={last_motion_metrics.get('reasons', [])}"
                            )
                            shutil.rmtree(case_dir, ignore_errors=True)
                        else:
                            bad_root = _qa_invalid_root_for_case(case_dir)
                            ensure_dir(bad_root)
                            bad_dst = bad_root / case_dir.name
                            if bad_dst.exists():
                                shutil.rmtree(bad_dst)
                            if case_dir.exists():
                                shutil.move(str(case_dir), str(bad_dst))
                            video_path = str(bad_dst / "metadata.json")
                            print(
                                f"🚫 motion_case_invalid_quarantined object={prepared.object_id} "
                                f"case={attempt_case_cfg.get('case_name')} "
                                f"reasons={last_motion_metrics.get('reasons', [])} "
                                f"dst={bad_dst}"
                            )
                    if video_path is not None:
                        preview_videos.append(video_path)
                preview_cases_all.extend(bundle_cases)
            preview_cases = preview_cases_all
            if preview_videos:
                preview_video = preview_videos[0]

    summary = asdict(prepared)
    summary["preview_video"] = preview_video
    summary["preview_videos"] = preview_videos
    summary["preview_cases"] = preview_cases
    summary["preview_skipped_reason"] = preview_skipped_reason
    summary["simulator_mode"] = str(getattr(args, "simulator_mode", "rigid"))
    summary_path = Path(prepared.output_dir) / f"{prepared.object_id}_summary.json"
    with open(summary_path, "w") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"⭕️ {preview_video}\n⭕️ {summary_path}")
    return summary


def main() -> None:
    args = build_argparser().parse_args()

    with _quiet_terminal_output(enabled=True):
        if args.json_override and not args.object_id:
            args.object_id = Path(str(args.json_override)).stem
            print(f"INFO inferred object_id={args.object_id} from --json_override")

        if args.object_id:
            object_ids = [str(args.object_id)]
        else:
            object_ids = _sample_random_object_ids(
                physx_root=Path(args.physx_root),
                version=str(args.version),
                num_objects=int(args.num_random_objects),
                seed=int(args.random_object_seed),
            )
            print(f"INFO randomly selected object_ids={object_ids}")

        for idx, object_id in enumerate(object_ids, start=1):
            print(f"INFO object {idx}/{len(object_ids)} object_id={object_id}")
            _run_single_object(args=args, object_id=str(object_id))


if __name__ == "__main__":
    main()
'''
桌布 19925
bowl with Liquid 12093
沙发 39264




    

  900002崩了
12093崩了
19925可以
30264可以



rm -rf /data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_try1_rigid_all_cases
# 指定物体id
python3 /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py \
    --object_id 5050 \
    --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_2_genesis_rigid_data_all_cases \
    --run_genesis \
    --generate_all_count_motion_cases \
    --prefer_existing_runtime_meshes \
    --dt 0.003 \
    --substeps 40 \
    --ball_posx 0.03 \
    --steps 12 \
    --fps 12 
    

# 随机抽取 num_random_objects 个物体作为主物体
python3 /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py \
    --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases \
    --run_genesis \
    --generate_all_count_motion_cases \
    --num_random_objects 1 \
    --prefer_existing_runtime_meshes \
    --simulator_mode rigid

  
# 遍历数据集所有
bash /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/run_try1_physxnet_all_objects_all_cases0419.sh





    # 补900 901
python /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py \
    --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
    --version version_1 \
    --object_id 5050 \
    --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases \
    --run_genesis \
    --num_random_cases 2 \
    --case_scene_mode diverse \
    --case_index_filter 900 901 \
    --prefer_existing_runtime_meshes \
    --dt 0.003 \
    --substeps 40 \
    --steps 49 \
    --fps 12 \
    --simulator_mode rigid \
    --rigid_target_object_count 1 \
    --physxnet_volume_threshold_m3 999999.0 \
    --disable_striker

# 生成caption
python3 /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_video_captions.py \
    --roots /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases


    


python /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/batch_inspect_physics_samples.py \
      --dataset_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_2_genesis_rigid_data_all_cases/train \
      --num_preview_frames 4 \
      --skip_existing \
      --serve \
      --host 0.0.0.0 \
      --port 8095

    '''
