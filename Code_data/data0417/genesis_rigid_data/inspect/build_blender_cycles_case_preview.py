# 用途：把 Genesis 刚体样本导出成 Blender 可渲染 scene spec，并调用 Cycles 生成预览。
"""Build a Blender Cycles preview for one Genesis rigid sample.

This script prepares a compact scene spec from an exported Genesis sample and
then invokes Blender headless to render a short Cycles preview.

It intentionally targets simple rigid cases first:
- a rigid target object reconstructed from cached visual part meshes
- optional striker / bystander spheres reconstructed from recorded kinematics

The preview is designed for quick inspection rather than final-quality render.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import shutil
import subprocess
import sys
import sysconfig
import importlib.util
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if "" in sys.path:
    sys.path.remove("")
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
stdlib_inspect = Path(sysconfig.get_paths()["stdlib"]) / "inspect.py"
spec = importlib.util.spec_from_file_location("inspect", stdlib_inspect)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to resolve stdlib inspect module from {stdlib_inspect}")
py_inspect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(py_inspect)
sys.modules["inspect"] = py_inspect
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import imageio.v2 as imageio
import numpy as np
import trimesh
from PIL import Image

from core.utils_io import load_json

BLENDER_DRIVER = SCRIPT_DIR / "blender_cycles_case_driver.py"
DEFAULT_RENDER_ASSET_ROOT = Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--quality", choices=("preview", "final"), default="preview")
    parser.add_argument("--frame_stride", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--render_asset_root", type=Path, default=DEFAULT_RENDER_ASSET_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_quality_profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.quality == "final":
        profile = {
            "frame_stride": 3,
            "max_frames": 32,
            "width": 1280,
            "height": 720,
            "samples": 384,
            "fps": 15,
            "exposure_bias": 6.0,
            "environment_strength_scale": 3.2,
            "background_strength_scale": 1.0,
            "light_energy_scale": 18.0,
            "sun_energy_scale": 6.0,
            "noise_threshold": 0.012,
            "min_samples": 48,
            "view_transform": "AgX",
            "look": "Medium High Contrast",
            "use_denoising": True,
            "use_motion_blur": True,
            "motion_blur_shutter": 0.24,
            "use_depth_of_field": True,
            "dof_fstop": 5.6,
            "dof_focus_bias": 0.05,
            "device": str(args.device),
        }
    else:
        profile = {
            "frame_stride": 4,
            "max_frames": 24,
            "width": 640,
            "height": 480,
            "samples": 96,
            "fps": 12,
            "exposure_bias": 0.0,
            "environment_strength_scale": 1.0,
            "background_strength_scale": 1.0,
            "light_energy_scale": 1.0,
            "sun_energy_scale": 1.0,
            "noise_threshold": 0.0,
            "min_samples": 0,
            "view_transform": "Standard",
            "look": "None",
            "use_denoising": bool(args.denoise),
            "use_motion_blur": False,
            "motion_blur_shutter": 0.0,
            "use_depth_of_field": False,
            "dof_fstop": 8.0,
            "dof_focus_bias": 0.0,
            "device": str(args.device),
        }

    for key in ("frame_stride", "max_frames", "width", "height", "samples", "fps"):
        value = getattr(args, key)
        if value is not None:
            profile[key] = int(value)
    if args.denoise:
        profile["use_denoising"] = True
    return profile


def estimate_focus_distance(
    *,
    camera_pos: np.ndarray,
    lookat: np.ndarray,
    stacked_positions: np.ndarray,
    focus_bias: float = 0.0,
) -> float:
    forward = np.asarray(lookat, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    norm = float(np.linalg.norm(forward))
    if norm <= 1e-8:
        return 2.0
    forward /= norm
    rel = np.asarray(stacked_positions, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    distances = rel @ forward
    positive = distances[distances > 0.05]
    if positive.size == 0:
        base = norm
    else:
        base = float(np.median(positive))
    return float(max(0.25, base + focus_bias))

def load_meta(sample_dir: Path) -> dict[str, Any]:
    for name in ("meta.json", "metadata.json"):
        path = sample_dir / name
        if path.exists():
            return load_json(path)
    raise FileNotFoundError(f"No meta.json or metadata.json under {sample_dir}")


def find_dataset_root(sample_dir: Path) -> Path:
    for candidate in (sample_dir, *sample_dir.parents):
        if (candidate / "_asset_cache").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate dataset root for {sample_dir}")


def inertial_origin(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        if bool(getattr(mesh, "is_watertight", False)) and bool(getattr(mesh, "is_volume", False)):
            center = np.asarray(mesh.center_mass, dtype=np.float64)
            if np.all(np.isfinite(center)):
                return center
    except Exception:
        pass
    return np.asarray(mesh.bounding_box.centroid, dtype=np.float64)


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g.copy() for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type at {mesh_path}")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def density_material_spec(density_kgm3: float | None, role: str) -> dict[str, Any]:
    density = None if density_kgm3 is None else float(density_kgm3)
    role = str(role)
    if role == "initiator":
        return {
            "base_color": [0.92, 0.72, 0.18, 1.0],
            "roughness": 0.28,
            "specular": 0.52,
            "metallic": 0.0,
            "clearcoat": 0.2,
        }
    if density is None:
        return {
            "base_color": [0.62, 0.64, 0.67, 1.0],
            "roughness": 0.52,
            "specular": 0.38,
            "metallic": 0.0,
            "clearcoat": 0.05,
        }
    if density < 180.0:
        return {
            "base_color": [0.60, 0.66, 0.72, 1.0],
            "roughness": 0.62,
            "specular": 0.26,
            "metallic": 0.0,
            "clearcoat": 0.0,
        }
    if density < 500.0:
        return {
            "base_color": [0.76, 0.77, 0.78, 1.0],
            "roughness": 0.44,
            "specular": 0.42,
            "metallic": 0.0,
            "clearcoat": 0.08,
        }
    if density < 900.0:
        return {
            "base_color": [0.63, 0.53, 0.39, 1.0],
            "roughness": 0.70,
            "specular": 0.22,
            "metallic": 0.0,
            "clearcoat": 0.0,
        }
    return {
        "base_color": [0.58, 0.60, 0.63, 1.0],
        "roughness": 0.27,
        "specular": 0.50,
        "metallic": 0.75,
        "clearcoat": 0.0,
    }


def classify_material_preset(
    *,
    role: str,
    source_object_id: str,
    density_kgm3: float | None,
) -> str:
    role = str(role)
    source_name = str(source_object_id).lower()
    if role == "initiator" or "striker" in source_name or "ball" in source_name:
        return "rubber_plastic"
    if density_kgm3 is None:
        return "painted_metal"
    density = float(density_kgm3)
    if density >= 900.0:
        return "painted_metal"
    if density >= 500.0:
        return "varnished_wood"
    if density >= 220.0:
        return "hard_plastic"
    return "soft_plastic"


def clamp_color01(rgb: np.ndarray) -> list[float]:
    rgb = np.asarray(rgb, dtype=np.float64).reshape(3)
    rgb = np.clip(rgb, 0.0, 1.0)
    return [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]


def blend_rgba(a: list[float] | np.ndarray, b: list[float] | np.ndarray, t: float) -> list[float]:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 3:
        aa = np.concatenate([aa, np.ones(1, dtype=np.float64)], axis=0)
    if bb.size == 3:
        bb = np.concatenate([bb, np.ones(1, dtype=np.float64)], axis=0)
    tt = float(np.clip(t, 0.0, 1.0))
    out = (1.0 - tt) * aa[:4] + tt * bb[:4]
    out = np.clip(out, 0.0, 1.0)
    return [float(out[0]), float(out[1]), float(out[2]), float(out[3])]


def rgb_to_hsv01(rgb: list[float] | np.ndarray) -> tuple[float, float, float]:
    r, g, b = [float(x) for x in np.asarray(rgb, dtype=np.float64).reshape(3)]
    return colorsys.rgb_to_hsv(r, g, b)


def luminance01(rgb: list[float] | np.ndarray) -> float:
    r, g, b = [float(x) for x in np.asarray(rgb, dtype=np.float64).reshape(3)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def stabilize_rgba_for_render(
    rgba: list[float] | np.ndarray,
    *,
    sat_scale: float = 1.0,
    min_value: float = 0.18,
    max_value: float = 0.84,
) -> list[float]:
    rgb = np.asarray(rgba, dtype=np.float64).reshape(-1)
    alpha = 1.0 if rgb.size < 4 else float(rgb[3])
    h, s, v = rgb_to_hsv01(rgb[:3])
    s = float(np.clip(s * sat_scale, 0.04, 0.88))
    v = float(np.clip(v, min_value, max_value))
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
    return [float(rr), float(gg), float(bb), float(np.clip(alpha, 0.0, 1.0))]


def shift_hsv(
    rgba: list[float] | np.ndarray,
    *,
    hue_delta: float = 0.0,
    sat_scale: float = 1.0,
    sat_bias: float = 0.0,
    value_scale: float = 1.0,
    value_bias: float = 0.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> list[float]:
    rgb = np.asarray(rgba, dtype=np.float64).reshape(-1)
    alpha = 1.0 if rgb.size < 4 else float(rgb[3])
    h, s, v = rgb_to_hsv01(rgb[:3])
    h = (h + float(hue_delta)) % 1.0
    s = float(np.clip(s * sat_scale + sat_bias, 0.0, 1.0))
    v = float(np.clip(v * value_scale + value_bias, min_value, max_value))
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, v)
    return [float(rr), float(gg), float(bb), float(np.clip(alpha, 0.0, 1.0))]


def extract_palette_from_pixels(pixels: np.ndarray, max_colors: int = 4) -> list[list[float]]:
    if pixels.size == 0:
        return []
    arr = np.asarray(pixels, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    # Quantize colors to stabilize palette extraction on noisy renders.
    q = np.clip(np.round(arr * 15.0), 0.0, 15.0).astype(np.int16)
    uniq, counts = np.unique(q, axis=0, return_counts=True)
    order = np.argsort(-counts)
    palette: list[list[float]] = []
    for idx in order.tolist():
        rgb = (uniq[idx].astype(np.float32) + 0.5) / 16.0
        if counts[idx] < max(32, int(0.002 * len(arr))):
            continue
        palette.append(clamp_color01(rgb))
        if len(palette) >= max_colors:
            break
    if not palette:
        palette.append(clamp_color01(np.median(arr, axis=0)))
    return palette


def estimate_object_appearance(
    *,
    sample_dir: Path,
    sample_meta: dict[str, Any],
    object_ids: np.ndarray,
    visibility: np.ndarray,
) -> dict[int, dict[str, Any]]:
    seg_path = sample_dir / "physics" / "seg.npy"
    if not seg_path.exists():
        return {}
    try:
        seg = np.load(seg_path, mmap_mode="r")
    except Exception:
        return {}

    meta_objects = {
        int(obj["object_id"]): dict(obj)
        for obj in sample_meta.get("objects", [])
        if isinstance(obj, dict) and obj.get("object_id") is not None and obj.get("seg_id") is not None
    }
    appearance: dict[int, dict[str, Any]] = {}
    max_frames = min(int(seg.shape[0]), int(visibility.shape[0]))
    for local_idx, object_id in enumerate(object_ids.tolist()):
        obj_meta = meta_objects.get(int(object_id))
        if obj_meta is None:
            continue
        seg_id = int(obj_meta["seg_id"])
        chosen = None
        palette: list[list[float]] = []
        for frame_idx in range(max_frames):
            if int(visibility[frame_idx, local_idx]) <= 0:
                continue
            rgb_path = sample_dir / "rgb" / f"frame_{frame_idx:03d}.png"
            if not rgb_path.exists():
                continue
            mask = np.asarray(seg[frame_idx] == seg_id)
            if mask.sum() < 64:
                continue
            with Image.open(rgb_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            pixels = rgb[mask]
            if pixels.size == 0:
                continue
            chosen = np.median(pixels, axis=0)
            palette = extract_palette_from_pixels(pixels, max_colors=4)
            break
        if chosen is not None:
            appearance[int(object_id)] = {
                "base_color": clamp_color01(chosen),
                "palette": palette,
            }
    return appearance


def apply_sampled_color(material_spec: dict[str, Any], sampled_rgba: list[float] | None) -> dict[str, Any]:
    if sampled_rgba is None:
        return material_spec
    out = dict(material_spec)
    out["base_color"] = [float(v) for v in sampled_rgba]
    return out


def part_shape_hint(extents_xyz: np.ndarray) -> str:
    ext = np.maximum(np.asarray(extents_xyz, dtype=np.float64), 1e-6)
    ordered = np.sort(ext)
    if ordered[0] / ordered[2] < 0.16 and ordered[1] / ordered[2] > 0.38:
        return "plate"
    if ordered[2] / ordered[0] > 5.0:
        return "slender"
    return "block"


def choose_palette_color_for_part(
    *,
    palette: list[list[float]],
    part_index: int,
    total_parts: int,
    shape_hint: str,
    fallback_rgba: list[float],
    asset_rgba: list[float] | None = None,
) -> list[float]:
    if not palette and asset_rgba is None:
        return stabilize_rgba_for_render(fallback_rgba)
    if not palette:
        return stabilize_rgba_for_render(asset_rgba or fallback_rgba, sat_scale=0.92)
    cool = [c for c in palette if rgb_to_hsv01(c[:3])[0] >= 0.48 and rgb_to_hsv01(c[:3])[0] <= 0.75]
    warm = [c for c in palette if rgb_to_hsv01(c[:3])[0] <= 0.15 or rgb_to_hsv01(c[:3])[0] >= 0.9]
    greenish = [c for c in palette if 0.22 <= rgb_to_hsv01(c[:3])[0] <= 0.45]
    dark = [c for c in palette if luminance01(c[:3]) < 0.33]
    chosen = None
    if shape_hint == "plate":
        if cool:
            chosen = cool[min(part_index, len(cool) - 1)]
    if shape_hint == "slender":
        pool = dark or warm or greenish
        if pool:
            chosen = pool[part_index % len(pool)]
    if chosen is None and total_parts >= 3 and part_index > 0:
        accent_pool = cool + greenish + warm + dark
        if accent_pool:
            chosen = accent_pool[part_index % len(accent_pool)]
    if chosen is None:
        chosen = palette[min(part_index, len(palette) - 1)]
    if asset_rgba is not None:
        asset_clean = stabilize_rgba_for_render(asset_rgba, sat_scale=0.96, min_value=0.16, max_value=0.82)
        chosen = blend_rgba(asset_clean, chosen, 0.36 if shape_hint == "slender" else 0.48)
    return stabilize_rgba_for_render(chosen, sat_scale=0.96)


def override_preset_from_shape_and_color(
    *,
    base_preset: str,
    shape_hint: str,
    rgba: list[float],
) -> str:
    h, s, v = rgb_to_hsv01(rgba[:3])
    if shape_hint == "plate" and s > 0.18 and 0.48 <= h <= 0.75:
        return "fabric_cloth"
    if shape_hint == "plate" and v > 0.55 and s < 0.18:
        return "painted_plastic"
    if shape_hint == "slender" and s > 0.20 and (h <= 0.14 or h >= 0.9):
        return "varnished_wood"
    if shape_hint == "slender" and 0.22 <= h <= 0.45:
        return "painted_metal"
    if base_preset == "hard_plastic" and s < 0.18:
        return "painted_metal"
    return base_preset


def derive_scene_style(object_appearance: dict[int, dict[str, Any]]) -> dict[str, Any]:
    all_colors: list[list[float]] = []
    for item in object_appearance.values():
        base_color = item.get("base_color")
        if isinstance(base_color, list) and len(base_color) >= 3:
            all_colors.append([float(base_color[0]), float(base_color[1]), float(base_color[2]), 1.0])
        for pal in item.get("palette", []) or []:
            if isinstance(pal, list) and len(pal) >= 3:
                all_colors.append([float(pal[0]), float(pal[1]), float(pal[2]), 1.0])
    if not all_colors:
        all_colors = [[0.44, 0.48, 0.55, 1.0]]

    rgb_stack = np.asarray([c[:3] for c in all_colors], dtype=np.float64)
    dominant = clamp_color01(np.median(rgb_stack, axis=0))
    wall = shift_hsv(
        dominant,
        hue_delta=0.48,
        sat_scale=0.30,
        sat_bias=0.05,
        value_scale=0.44,
        value_bias=0.56,
        min_value=0.78,
        max_value=0.92,
    )
    wall = blend_rgba(wall, [0.92, 0.92, 0.90, 1.0], 0.35)
    floor = shift_hsv(
        dominant,
        hue_delta=0.06,
        sat_scale=0.22,
        sat_bias=0.02,
        value_scale=0.42,
        value_bias=0.10,
        min_value=0.22,
        max_value=0.40,
    )
    floor = blend_rgba(floor, [0.28, 0.29, 0.31, 1.0], 0.55)
    background = blend_rgba(wall, [0.64, 0.66, 0.70, 1.0], 0.62)
    key_light = blend_rgba(
        shift_hsv(dominant, hue_delta=-0.04, sat_scale=0.22, value_scale=0.30, value_bias=0.70, min_value=0.88, max_value=1.0),
        [1.0, 0.95, 0.88, 1.0],
        0.72,
    )
    fill_light = blend_rgba(
        shift_hsv(dominant, hue_delta=0.18, sat_scale=0.18, value_scale=0.22, value_bias=0.76, min_value=0.84, max_value=1.0),
        [0.85, 0.91, 1.0, 1.0],
        0.78,
    )
    rim_light = blend_rgba(
        shift_hsv(dominant, hue_delta=0.52, sat_scale=0.26, value_scale=0.26, value_bias=0.74, min_value=0.86, max_value=1.0),
        [1.0, 0.99, 0.96, 1.0],
        0.58,
    )
    return {
        "dominant": dominant,
        "wall_color": wall,
        "floor_color": floor,
        "background_color": background,
        "key_light_color": key_light,
        "fill_light_color": fill_light,
        "rim_light_color": rim_light,
        "environment_strength": 0.44,
        "background_strength": 0.82,
        "exposure": -0.38,
    }


def load_render_asset_library(asset_root: Path) -> dict[str, Any]:
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        data = load_json(manifest_path)
    except Exception:
        return {}
    data["_asset_root"] = str(asset_root)
    return data


def build_texture_binding(texture_entry: dict[str, Any] | None, *, mapping_scale: list[float]) -> dict[str, Any] | None:
    if not texture_entry:
        return None
    maps = dict(texture_entry.get("maps", {}))
    base_color = maps.get("base_color")
    normal = maps.get("normal")
    roughness = maps.get("roughness")
    if not base_color or not normal or not roughness:
        return None
    binding = {
        "name": str(texture_entry.get("asset_id", texture_entry.get("name", "texture"))),
        "base_color": str(base_color),
        "normal": str(normal),
        "roughness": str(roughness),
        "mapping_scale": [float(v) for v in mapping_scale],
        "normal_strength": float(texture_entry.get("normal_strength", 0.35)),
        "roughness_mix": float(texture_entry.get("roughness_mix", 0.82)),
        "base_mix": float(texture_entry.get("base_mix", 0.92)),
        "projection": str(texture_entry.get("projection", "UV")),
    }
    if maps.get("ao"):
        binding["ao"] = str(maps["ao"])
        binding["ao_mix"] = float(texture_entry.get("ao_mix", 0.22))
    return binding


def choose_render_asset_bindings(
    *,
    object_specs: list[dict[str, Any]],
    scene_style: dict[str, Any],
    asset_library: dict[str, Any],
) -> dict[str, Any]:
    hdris = dict(asset_library.get("hdris", {}))
    textures = dict(asset_library.get("textures", {}))
    if not hdris and not textures:
        return {}

    preset_counts: dict[str, int] = {}
    for obj in object_specs:
        for part in obj.get("parts", []):
            mat = part.get("material", {})
            preset = str(mat.get("material_preset", "unknown"))
            preset_counts[preset] = preset_counts.get(preset, 0) + 1

    wood_like = preset_counts.get("varnished_wood", 0)
    metal_like = preset_counts.get("painted_metal", 0) + preset_counts.get("hard_plastic", 0) + preset_counts.get("painted_plastic", 0)
    cloth_like = preset_counts.get("fabric_cloth", 0)

    if wood_like >= max(2, metal_like):
        hdri_key = "interior_warm"
        floor_key = "wood_floor"
        wall_key = "beige_wall_001"
    elif metal_like >= max(2, wood_like):
        hdri_key = "studio_warm"
        floor_key = "painted_concrete"
        wall_key = "beige_wall_001"
    elif cloth_like >= 2:
        hdri_key = "studio_soft"
        floor_key = "painted_concrete"
        wall_key = "beige_wall_001"
    else:
        hdri_key = "studio_soft"
        floor_key = "painted_concrete"
        wall_key = "beige_wall_001"

    hdri_entry = hdris.get(hdri_key) or next(iter(hdris.values()), None)
    floor_entry = textures.get(floor_key)
    wall_entry = textures.get(wall_key)
    return {
        "hdri": hdri_entry,
        "floor_texture": build_texture_binding(floor_entry, mapping_scale=[1.8, 1.8, 1.8]),
        "wall_texture": build_texture_binding(wall_entry, mapping_scale=[1.1, 1.1, 1.1]),
        "selection": {
            "hdri_key": hdri_key,
            "floor_key": floor_key,
            "wall_key": wall_key,
            "preset_counts": preset_counts,
            "dominant_color": scene_style.get("dominant"),
        },
    }


def estimate_sphere_radius(
    bbox_xyxy: np.ndarray,
    center_depth: np.ndarray,
    visibility: np.ndarray,
    *,
    fx: float,
    fy: float,
) -> float:
    visible = np.where(visibility > 0)[0]
    for frame_idx in visible.tolist():
        bbox = bbox_xyxy[frame_idx]
        depth = float(center_depth[frame_idx])
        if not np.all(np.isfinite(bbox)) or not np.isfinite(depth) or depth <= 1e-6:
            continue
        w_px = float(max(0.0, bbox[2] - bbox[0]))
        h_px = float(max(0.0, bbox[3] - bbox[1]))
        if w_px <= 1.0 or h_px <= 1.0:
            continue
        rx = 0.5 * (w_px * depth / max(fx, 1e-6))
        ry = 0.5 * (h_px * depth / max(fy, 1e-6))
        radius = 0.5 * (rx + ry)
        return float(np.clip(radius, 0.03, 0.25))
    return 0.08


def resolve_asset_meta(dataset_root: Path, source_object_id: str) -> dict[str, Any]:
    asset_meta = dataset_root / "_asset_cache" / "physxnet_objects" / str(source_object_id) / "meta" / "metadata.json"
    if not asset_meta.exists():
        raise FileNotFoundError(f"Missing asset metadata for {source_object_id}: {asset_meta}")
    return load_json(asset_meta)


def build_mesh_object_spec(
    *,
    sample_meta: dict[str, Any],
    dataset_root: Path,
    object_meta: dict[str, Any],
    positions: np.ndarray,
    quats: np.ndarray,
    sampled_rgba: list[float] | None,
    sampled_palette: list[list[float]] | None,
) -> dict[str, Any]:
    source_object_id = str(object_meta["source_object_id"])
    asset_meta = resolve_asset_meta(dataset_root, source_object_id)
    part_links = list(asset_meta.get("rigid_part_links", []) or [])
    if not part_links:
        raise ValueError(f"No rigid_part_links for object {source_object_id}")

    runtime_scale = float(sample_meta.get("runtime_main_object_scale", 1.0))
    weighted_centers = []
    weighted_masses = []
    parts = []

    for part in part_links:
        mesh_path = Path(str(part["mesh_path"]))
        mesh = load_mesh(mesh_path)
        local_center = inertial_origin(mesh) * runtime_scale
        extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64) * runtime_scale
        mass = float(part.get("mass_kg", 1.0) or 1.0)
        weighted_centers.append(local_center * mass)
        weighted_masses.append(mass)
        parts.append(
            {
                "mesh_path": str(mesh_path),
                "density_kgm3": None if part.get("density_kgm3") is None else float(part["density_kgm3"]),
                "extents_xyz": extents.tolist(),
                "asset_rgba": None if part.get("color_rgba") is None else [float(v) for v in part["color_rgba"]],
                "material_preset": classify_material_preset(
                    role=str(object_meta.get("role", "object")),
                    source_object_id=source_object_id,
                    density_kgm3=None if part.get("density_kgm3") is None else float(part["density_kgm3"]),
                ),
            }
        )

    total_mass = float(np.sum(weighted_masses))
    if total_mass <= 1e-8:
        com_local = np.zeros(3, dtype=np.float64)
    else:
        com_local = np.sum(np.stack(weighted_centers, axis=0), axis=0) / total_mass

    palette = list(sampled_palette or [])
    fallback_rgba = list(sampled_rgba or [0.68, 0.68, 0.68, 1.0])
    part_volumes = [float(np.prod(np.maximum(np.asarray(part["extents_xyz"], dtype=np.float64), 1e-6))) for part in parts]
    volume_order = np.argsort(-np.asarray(part_volumes, dtype=np.float64))
    volume_rank = {int(part_idx): int(rank) for rank, part_idx in enumerate(volume_order.tolist())}
    rendered_parts = []
    for part_idx, part in enumerate(parts):
        shape_hint = part_shape_hint(np.asarray(part["extents_xyz"], dtype=np.float64))
        color = choose_palette_color_for_part(
            palette=palette,
            part_index=volume_rank.get(part_idx, part_idx),
            total_parts=len(parts),
            shape_hint=shape_hint,
            fallback_rgba=fallback_rgba,
            asset_rgba=part.get("asset_rgba"),
        )
        rendered_parts.append(
            {
                "mesh_path": part["mesh_path"],
                "local_offset": (-com_local).tolist(),
                "local_scale": [runtime_scale, runtime_scale, runtime_scale],
                "material": {
                    **apply_sampled_color(
                        density_material_spec(part["density_kgm3"], str(object_meta.get("role", "object"))),
                        color,
                    ),
                    "material_preset": override_preset_from_shape_and_color(
                        base_preset=str(part["material_preset"]),
                        shape_hint=shape_hint,
                        rgba=color,
                    ),
                },
            }
        )
    return {
        "kind": "animated_mesh",
        "name": f"{object_meta.get('role','object')}_{source_object_id}",
        "frames": [
            {
                "position": np.asarray(pos, dtype=np.float64).tolist(),
                "quaternion_wxyz": np.asarray(quat, dtype=np.float64).tolist(),
            }
            for pos, quat in zip(positions, quats)
        ],
        "parts": rendered_parts,
    }


def build_sphere_object_spec(
    *,
    object_meta: dict[str, Any],
    positions: np.ndarray,
    quats: np.ndarray,
    bbox_xyxy: np.ndarray,
    center_depth: np.ndarray,
    visibility: np.ndarray,
    camera_intrinsics: dict[str, Any],
    sampled_rgba: list[float] | None,
    sampled_palette: list[list[float]] | None,
) -> dict[str, Any]:
    radius = estimate_sphere_radius(
        bbox_xyxy=bbox_xyxy,
        center_depth=center_depth,
        visibility=visibility,
        fx=float(camera_intrinsics["fx"]),
        fy=float(camera_intrinsics["fy"]),
    )
    sphere_rgba = choose_palette_color_for_part(
        palette=list(sampled_palette or []),
        part_index=0,
        total_parts=1,
        shape_hint="block",
        fallback_rgba=list(sampled_rgba or [0.80, 0.72, 0.22, 1.0]),
    )
    return {
        "kind": "animated_sphere",
        "name": f"{object_meta.get('role','object')}_{object_meta.get('source_object_id','sphere')}",
        "radius": radius,
        "frames": [
            {
                "position": np.asarray(pos, dtype=np.float64).tolist(),
                "quaternion_wxyz": np.asarray(quat, dtype=np.float64).tolist(),
            }
            for pos, quat in zip(positions, quats)
        ],
        "material": apply_sampled_color(
            density_material_spec(None, str(object_meta.get("role", "initiator"))),
            sphere_rgba,
        )
        | {
            "material_preset": override_preset_from_shape_and_color(
                base_preset=classify_material_preset(
                    role=str(object_meta.get("role", "initiator")),
                    source_object_id=str(object_meta.get("source_object_id", "sphere")),
                    density_kgm3=None,
                ),
                shape_hint="block",
                rgba=sphere_rgba,
            )
        },
    }


def sample_frame_indices(total_frames: int, stride: int, max_frames: int) -> list[int]:
    indices = list(range(0, total_frames, max(1, int(stride))))
    if not indices:
        indices = [0]
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    if len(indices) > max_frames and max_frames > 0:
        picked = np.linspace(0, len(indices) - 1, num=max_frames, dtype=np.int32)
        indices = [indices[int(i)] for i in picked.tolist()]
    return sorted(set(indices))


def make_source_gif(sample_dir: Path, frame_indices: list[int], out_path: Path) -> None:
    frames = []
    for frame_idx in frame_indices:
        frame_path = sample_dir / "rgb" / f"frame_{frame_idx:03d}.png"
        if not frame_path.exists():
            continue
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def make_gif_from_video(video_path: Path, gif_path: Path) -> None:
    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        for frame in reader:
            frames.append(Image.fromarray(frame).convert("RGB"))
    finally:
        reader.close()
    if not frames:
        return
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )


def collect_rendered_frames(frame_root: Path) -> list[Path]:
    return sorted(frame_root.glob("frame_*.png"))


def make_gif_from_frames(frame_paths: list[Path], gif_path: Path, *, fps: int) -> None:
    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    if not frames:
        return
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def make_mp4_from_frames(frame_paths: list[Path], video_path: Path, *, fps: int) -> None:
    if not frame_paths:
        return
    try:
        writer = imageio.get_writer(
            str(video_path),
            fps=max(1, int(fps)),
            codec="libx264",
            macro_block_size=None,
        )
    except Exception:
        return
    try:
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                writer.append_data(np.asarray(image.convert("RGB")))
    finally:
        writer.close()


def build_html(output_root: Path, spec: dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Blender Cycles Case Preview</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", sans-serif;
      background: #f7f3ee;
      color: #221b14;
    }}
    .wrap {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 20px 24px 40px;
    }}
    .meta {{
      margin-bottom: 18px;
      line-height: 1.55;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(255,255,255,0.82);
      border: 1px solid #ddd4c8;
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 10px 26px rgba(40, 29, 18, 0.08);
    }}
    h1 {{ margin: 0 0 10px; font-size: 24px; }}
    h2 {{ margin: 0 0 8px; font-size: 16px; }}
    p {{ margin: 4px 0; }}
    img, video {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #e9e4dc;
    }}
    code {{
      background: #f0ebe3;
      padding: 1px 4px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Blender Cycles Case Preview</h1>
    <div class="meta">
      <p><strong>Sample:</strong> <code>{spec['sample_name']}</code></p>
      <p><strong>Source:</strong> <code>{spec['sample_dir']}</code></p>
      <p><strong>Frames:</strong> sampled {len(spec['sampled_frame_indices'])} / {spec['total_frames']} frames, stride={spec['frame_stride']}</p>
      <p><strong>Cycles:</strong> {spec['render']['width']}x{spec['render']['height']}, samples={spec['render']['samples']}, fps={spec['render']['fps']}</p>
      <p><strong>Assets:</strong> <code>{spec.get('render_assets', {})}</code></p>
    </div>
    <div class="grid">
      <div class="card">
        <h2>Source RGB GIF</h2>
        <p>含义：原始 Genesis 采样帧，作为参考真值画面。</p>
        <img src="source_rgb.gif" alt="source rgb">
      </div>
      <div class="card">
        <h2>Cycles Render GIF</h2>
        <p>含义：同一段轨迹在 Blender Cycles 下的离线渲染预览。</p>
        <img src="cycles_preview.gif" alt="cycles preview">
      </div>
      <div class="card">
        <h2>Cycles Render MP4</h2>
        <p>含义：Cycles 预览视频，方便看完整播放。</p>
        <video controls playsinline src="cycles_preview.mp4"></video>
      </div>
      <div class="card">
        <h2>Scene Spec</h2>
        <p>含义：记录导出给 Blender 的相机、物体和时间轴参数。</p>
        <p><a href="scene_spec.json">scene_spec.json</a></p>
        <p><a href="cycles_preview.blend">cycles_preview.blend</a></p>
      </div>
    </div>
  </div>
</body>
</html>
"""
    (output_root / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    quality_profile = build_quality_profile(args)
    sample_dir = args.sample_dir.resolve()
    output_root = args.output_root.resolve()
    render_asset_root = args.render_asset_root.resolve()

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_meta = load_meta(sample_dir)
    dataset_root = find_dataset_root(sample_dir)
    asset_library = load_render_asset_library(render_asset_root)
    kinematics = np.load(sample_dir / "physics" / "rigid_kinematics.npz", allow_pickle=True)
    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz", allow_pickle=True)

    object_ids = np.asarray(kinematics["object_ids"], dtype=np.int32)
    com_pos = np.asarray(kinematics["com_pos"], dtype=np.float64)
    quats = np.asarray(kinematics["orientation_quat"], dtype=np.float64)
    bbox_xyxy = np.asarray(anchor["bbox_xyxy"], dtype=np.float64)
    center_depth = np.asarray(anchor["center_depth"], dtype=np.float64)
    visibility = np.asarray(anchor["visibility_mask"], dtype=np.uint8)
    object_appearance = estimate_object_appearance(
        sample_dir=sample_dir,
        sample_meta=sample_meta,
        object_ids=object_ids,
        visibility=visibility,
    )
    scene_style = derive_scene_style(object_appearance)

    frame_indices = sample_frame_indices(
        com_pos.shape[0],
        quality_profile["frame_stride"],
        quality_profile["max_frames"],
    )
    meta_objects = {
        int(obj["object_id"]): dict(obj)
        for obj in sample_meta.get("objects", [])
        if isinstance(obj, dict) and obj.get("object_id") is not None
    }

    object_specs = []
    all_positions = []
    for local_idx, object_id in enumerate(object_ids.tolist()):
        obj_meta = meta_objects.get(int(object_id))
        if obj_meta is None:
            continue
        sampled_pos = com_pos[frame_indices, local_idx]
        sampled_quat = quats[frame_indices, local_idx]
        all_positions.append(sampled_pos)
        source_id = str(obj_meta.get("source_object_id", ""))
        if source_id == "yellow_striker_ball":
            sampled_appearance = object_appearance.get(int(object_id), {})
            spec = build_sphere_object_spec(
                object_meta=obj_meta,
                positions=sampled_pos,
                quats=sampled_quat,
                bbox_xyxy=bbox_xyxy[:, local_idx],
                center_depth=center_depth[:, local_idx],
                visibility=visibility[:, local_idx],
                camera_intrinsics=sample_meta["camera_intrinsics"],
                sampled_rgba=sampled_appearance.get("base_color"),
                sampled_palette=sampled_appearance.get("palette"),
            )
        else:
            sampled_appearance = object_appearance.get(int(object_id), {})
            spec = build_mesh_object_spec(
                sample_meta=sample_meta,
                dataset_root=dataset_root,
                object_meta=obj_meta,
                positions=sampled_pos,
                quats=sampled_quat,
                sampled_rgba=sampled_appearance.get("base_color"),
                sampled_palette=sampled_appearance.get("palette"),
            )
        for timeline_frame, item in enumerate(spec["frames"], start=1):
            item["timeline_frame"] = timeline_frame
        object_specs.append(spec)

    if not object_specs:
        raise RuntimeError(f"No renderable objects resolved from {sample_dir}")
    render_assets = choose_render_asset_bindings(
        object_specs=object_specs,
        scene_style=scene_style,
        asset_library=asset_library,
    )

    stacked_pos = np.concatenate(all_positions, axis=0)
    xy_min = np.min(stacked_pos[:, :2], axis=0)
    xy_max = np.max(stacked_pos[:, :2], axis=0)
    xy_center = 0.5 * (xy_min + xy_max)
    xy_extent = np.maximum(xy_max - xy_min, 1.6) + 1.2
    camera_pos = np.asarray(sample_meta["camera"]["pos"], dtype=np.float64)
    camera_lookat = np.asarray(sample_meta["camera"]["lookat"], dtype=np.float64)
    focus_distance = estimate_focus_distance(
        camera_pos=camera_pos,
        lookat=camera_lookat,
        stacked_positions=stacked_pos,
        focus_bias=float(quality_profile["dof_focus_bias"]),
    )

    spec = {
        "sample_name": sample_dir.name,
        "sample_dir": str(sample_dir),
        "output_root": str(output_root),
        "frame_stride": int(quality_profile["frame_stride"]),
        "sampled_frame_indices": frame_indices,
        "total_frames": int(com_pos.shape[0]),
        "render": {
            "quality": str(args.quality),
            "width": int(quality_profile["width"]),
            "height": int(quality_profile["height"]),
            "samples": int(quality_profile["samples"]),
            "fps": int(quality_profile["fps"]),
            "use_denoising": bool(quality_profile["use_denoising"]),
            "exposure": float(scene_style["exposure"]) + float(quality_profile["exposure_bias"]),
            "noise_threshold": float(quality_profile["noise_threshold"]),
            "min_samples": int(quality_profile["min_samples"]),
            "view_transform": str(quality_profile["view_transform"]),
            "look": str(quality_profile["look"]),
            "use_motion_blur": bool(quality_profile["use_motion_blur"]),
            "motion_blur_shutter": float(quality_profile["motion_blur_shutter"]),
            "device": str(quality_profile["device"]),
        },
        "camera": {
            "position": [float(v) for v in sample_meta["camera"]["pos"]],
            "lookat": [float(v) for v in sample_meta["camera"]["lookat"]],
            "up": [float(v) for v in sample_meta["camera"].get("up", [0.0, 0.0, 1.0])],
            "fov_deg": float(sample_meta["camera"]["fov"]),
            "depth_of_field": {
                "enabled": bool(quality_profile["use_depth_of_field"]),
                "focus_distance": float(focus_distance),
                "fstop": float(quality_profile["dof_fstop"]),
                "aperture_blades": 7,
                "aperture_rotation_deg": 8.0,
            },
        },
        "timeline": {
            "frame_start": 1,
            "frame_end": len(frame_indices),
        },
        "ground": {
            "center": [float(xy_center[0]), float(xy_center[1]), 0.0],
            "extents_xy": [float(xy_extent[0]), float(xy_extent[1])],
            "material": {
                "base_color": scene_style["floor_color"],
                "roughness": 0.78,
                "specular": 0.18,
                "metallic": 0.0,
                "clearcoat": 0.0,
                "material_preset": "concrete_floor",
                "texture_set": render_assets.get("floor_texture"),
            },
        },
        "room": {
            "enabled": True,
            "center": [float(xy_center[0]), float(xy_center[1]), 0.0],
            "depth": float(max(5.6, xy_extent[1] + 4.6)),
            "width": float(max(6.8, xy_extent[0] + 4.6)),
            "height": 4.4,
            "wall_material": {
                "base_color": scene_style["wall_color"],
                "roughness": 0.88,
                "specular": 0.12,
                "metallic": 0.0,
                "clearcoat": 0.0,
                "material_preset": "painted_wall",
                "texture_set": render_assets.get("wall_texture"),
            },
        },
        "environment": {
            "world_exr": str((render_assets.get("hdri") or {}).get("path", "/usr/share/blender/datafiles/studiolights/world/studio.exr")),
            "strength": float(scene_style["environment_strength"]) * float(quality_profile["environment_strength_scale"]),
            "rotation_deg": 92.0,
            "background_strength": float(scene_style["background_strength"]) * float(quality_profile["background_strength_scale"]),
            "background_color": scene_style["background_color"],
        },
        "lighting": {
            "key_area": {
                "location": [float(xy_center[0] + 1.35), float(xy_center[1] - 2.6), 2.45],
                "rotation_euler_deg": [62.0, 0.0, 18.0],
                "energy": 170.0 * float(quality_profile["light_energy_scale"]),
                "size": 1.7,
                "color": scene_style["key_light_color"],
                "spread_deg": 118.0,
            },
            "fill_area": {
                "location": [float(xy_center[0] - 1.7), float(xy_center[1] + 1.8), 1.55],
                "rotation_euler_deg": [88.0, 0.0, -36.0],
                "energy": 55.0 * float(quality_profile["light_energy_scale"]) * 0.92,
                "size": 2.2,
                "color": scene_style["fill_light_color"],
                "spread_deg": 132.0,
            },
            "rim_area": {
                "location": [float(xy_center[0] + 0.25), float(xy_center[1] + 2.8), 2.3],
                "rotation_euler_deg": [116.0, 0.0, 180.0],
                "energy": 92.0 * float(quality_profile["light_energy_scale"]) * 1.05,
                "size": 2.8,
                "color": scene_style["rim_light_color"],
                "spread_deg": 104.0,
            },
            "sun": {
                "rotation_euler_deg": [34.0, 0.0, 32.0],
                "energy": 0.22 * float(quality_profile["sun_energy_scale"]),
                "color": [1.0, 0.98, 0.95, 1.0],
            },
        },
        "render_assets": render_assets.get("selection", {}),
        "objects": object_specs,
    }
    spec_path = output_root / "scene_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    blender_cmd = [
        "blender",
        "-b",
        "-P",
        str(BLENDER_DRIVER),
        "--",
        "--spec_json",
        str(spec_path),
    ]
    subprocess.run(blender_cmd, check=True)

    source_gif = output_root / "source_rgb.gif"
    make_source_gif(sample_dir, frame_indices, source_gif)
    frame_paths = collect_rendered_frames(output_root / "frames")
    video_path = output_root / "cycles_preview.mp4"
    gif_path = output_root / "cycles_preview.gif"
    if frame_paths:
        make_gif_from_frames(frame_paths, gif_path, fps=int(quality_profile["fps"]))
        make_mp4_from_frames(frame_paths, video_path, fps=int(quality_profile["fps"]))
    build_html(output_root, spec)
    print(f"[DONE] preview page: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
