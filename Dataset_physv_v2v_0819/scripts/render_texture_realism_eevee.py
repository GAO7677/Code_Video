#!/usr/bin/env python3
"""Render a generic PhysV collision demo with varied PBR backgrounds in Eevee.

This file is executed by Blender.  It reuses the material library, room
layout, HDRI, and lighting helpers from ``render_physv_cycles.py`` but keeps
the engine on ``BLENDER_EEVEE``.  Physics is never rerun here: poses are read
from the already-produced PyBullet state file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_physv_cycles as reference  # noqa: E402


BACKGROUND_ASSET_ROOT = Path(
    "/data/gaoya/agent-data/assets/texture_realism_backgrounds_20260825"
)
BACKGROUND_PROFILE_ORDER = (
    "warehouse_cobalt",
    "machine_shop_amber",
    "color_studio",
    "glasshouse_mint",
    "courtyard_terracotta",
    "foundry_safety",
    "garage_teal",
    "neon_studio",
)

# Display colors are deliberately spread over hue/value space.  The renderer
# chooses a subset per video after comparing against all background anchors.
DISPLAY_PALETTE = (
    ("coral", (0.92, 0.12, 0.045)),
    ("azure", (0.035, 0.34, 0.92)),
    ("lime", (0.28, 0.78, 0.055)),
    ("magenta", (0.86, 0.045, 0.46)),
    ("gold", (0.95, 0.58, 0.025)),
    ("violet", (0.34, 0.08, 0.86)),
    ("turquoise", (0.02, 0.72, 0.66)),
    ("crimson", (0.72, 0.035, 0.075)),
    ("pink", (0.96, 0.22, 0.56)),
    ("teal_bright", (0.02, 0.86, 0.78)),
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--exposure", type=float, default=0.35)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument(
        "--background-profile",
        choices=BACKGROUND_PROFILE_ORDER,
        default="warehouse_cobalt",
    )
    return parser.parse_args(argv)


def _pbr_texture_names() -> dict[str, str]:
    return {
        "albedo": "rubber_tiles_diff_2k.jpg",
        "normal": "rubber_tiles_nor_gl_2k.jpg",
        "roughness": "rubber_tiles_rough_2k.jpg",
        "ao": "rubber_tiles_ao_2k.jpg",
    }


def _metal_texture_names() -> dict[str, str]:
    return {
        "albedo": "metal_plate_diff_2k.jpg",
        "normal": "metal_plate_nor_gl_2k.jpg",
        "roughness": "metal_plate_rough_2k.jpg",
        "ao": "metal_plate_ao_2k.jpg",
    }


def _cardboard_texture_names() -> dict[str, str]:
    return {
        "albedo": "beige_wall_001_diff_2k.jpg",
        "normal": "beige_wall_001_nor_gl_2k.jpg",
        "roughness": "beige_wall_001_rough_2k.jpg",
        "ao": "beige_wall_001_ao_2k.jpg",
    }


def build_demo_materials() -> dict[str, bpy.types.Material]:
    """Build semantic aliases with actual color/normal/roughness/AO maps."""

    materials = reference.material_library()
    rubber_dir = reference.EXTRA_TEXTURE_ROOT / "rubber_tiles"
    metal_dir = reference.EXTRA_TEXTURE_ROOT / "metal_plate"
    wall_dir = reference.TEXTURE_ROOT / "beige_wall_001"

    # These aliases preserve the demo's semantic material names while using
    # the same local PBR asset families used by the test70 Cycles previews.
    materials["demo_plastic_orange"] = reference.pbr_material(
        "Demo_PBR_Orange_Plastic",
        texture_dir=rubber_dir,
        texture_names=_pbr_texture_names(),
        tint=(1.45, 0.22, 0.06),
        tint_strength=0.86,
        roughness=0.62,
        uv_scale=2.5,
        normal_strength=0.48,
        detail_bump_strength=0.010,
        detail_bump_scale=22.0,
    )
    materials["demo_plastic_white"] = reference.pbr_material(
        "Demo_PBR_White_Plastic",
        texture_dir=rubber_dir,
        texture_names=_pbr_texture_names(),
        tint=(0.82, 0.88, 0.95),
        tint_strength=0.68,
        roughness=0.48,
        uv_scale=2.5,
        normal_strength=0.40,
        detail_bump_strength=0.008,
        detail_bump_scale=22.0,
    )
    materials["demo_teal_metal"] = reference.pbr_material(
        "Demo_PBR_Teal_Metal",
        texture_dir=metal_dir,
        texture_names=_metal_texture_names(),
        tint=(0.05, 0.70, 0.76),
        tint_strength=0.78,
        roughness=0.48,
        metallic=0.30,
        uv_scale=1.35,
        normal_strength=0.58,
        detail_bump_strength=0.012,
        detail_bump_scale=16.0,
    )
    materials["demo_cardboard"] = reference.pbr_material(
        "Demo_PBR_Cardboard",
        texture_dir=wall_dir,
        texture_names=_cardboard_texture_names(),
        tint=(0.86, 0.61, 0.34),
        tint_strength=0.78,
        roughness=0.88,
        uv_scale=2.0,
        normal_strength=0.52,
        detail_bump_strength=0.012,
        detail_bump_scale=18.0,
    )
    return materials


def material_for_key(materials: dict[str, bpy.types.Material], key: str):
    aliases = {
        "plastic_orange": "demo_plastic_orange",
        "plastic_white": "demo_plastic_white",
        "painted_metal_teal": "demo_teal_metal",
        "painted_metal_blue": "blue_painted",
        "painted_metal_yellow": "yellow_metal",
        "cardboard_kraft": "demo_cardboard",
        "rubber_red": "red_rubber",
        "rubber_blue": "blue_rubber",
        "wood_plywood": "wood",
        "wood_dark": "domino_wood",
        "wood_red": "red_wood",
    }
    resolved = aliases.get(key, key)
    if resolved in materials:
        return materials[resolved]
    return materials["white_painted"]


BACKGROUND_PROFILES = {
    "warehouse_cobalt": {
        "hdri_id": "empty_warehouse_01",
        "floor_asset": "asphalt_floor",
        "wall_asset": "brick_wall_001",
        "floor_color": (0.075, 0.11, 0.16),
        "wall_color": (0.46, 0.105, 0.045),
        "trim_color": (0.035, 0.055, 0.085),
        "accent_color": (0.95, 0.55, 0.02),
        "secondary_color": (0.04, 0.40, 0.82),
        "dominant_rgb": (0.20, 0.14, 0.15),
        "set_kind": "warehouse",
        "hdri_rotation_deg": 28.0,
        "hdri_strength": 0.24,
        "key": ((-4.5, -3.0, 5.4), 980.0, (0.78, 0.90, 1.0)),
        "fill": ((4.8, -0.5, 3.0), 470.0, (0.48, 0.68, 1.0)),
        "top": ((0.0, 5.0, 5.8), 720.0, (0.86, 0.94, 1.0)),
        "rim": ((-5.0, 6.5, 3.8), 420.0, (1.0, 0.55, 0.22)),
    },
    "machine_shop_amber": {
        "hdri_id": "machine_shop_02",
        "floor_asset": "blue_metal_plate",
        "wall_asset": "concrete_block_wall",
        "floor_color": (0.055, 0.12, 0.20),
        "wall_color": (0.30, 0.34, 0.38),
        "trim_color": (0.08, 0.10, 0.12),
        "accent_color": (0.90, 0.44, 0.025),
        "secondary_color": (0.66, 0.70, 0.74),
        "dominant_rgb": (0.20, 0.25, 0.30),
        "set_kind": "machine_shop",
        "hdri_rotation_deg": -42.0,
        "hdri_strength": 0.22,
        "key": ((-3.8, -3.4, 4.7), 910.0, (1.0, 0.78, 0.48)),
        "fill": ((4.6, -0.7, 3.2), 460.0, (0.52, 0.72, 1.0)),
        "top": ((0.0, 4.5, 5.5), 680.0, (1.0, 0.88, 0.64)),
        "rim": ((5.0, 6.0, 4.0), 390.0, (0.38, 0.62, 1.0)),
    },
    "color_studio": {
        "hdri_id": "colorful_studio",
        "floor_asset": "brick_floor",
        "wall_asset": "blue_plaster_wall",
        "floor_color": (0.28, 0.10, 0.055),
        "wall_color": (0.04, 0.24, 0.78),
        "trim_color": (0.08, 0.04, 0.12),
        "accent_color": (0.86, 0.035, 0.40),
        "secondary_color": (0.10, 0.82, 0.42),
        "dominant_rgb": (0.18, 0.16, 0.35),
        "set_kind": "color_studio",
        "hdri_rotation_deg": 74.0,
        "hdri_strength": 0.19,
        "key": ((-4.2, -3.0, 4.8), 850.0, (1.0, 0.35, 0.50)),
        "fill": ((4.2, -1.0, 3.1), 520.0, (0.28, 0.60, 1.0)),
        "top": ((0.0, 4.0, 5.8), 650.0, (0.55, 1.0, 0.70)),
        "rim": ((-5.0, 5.0, 3.6), 440.0, (1.0, 0.72, 0.18)),
    },
    "glasshouse_mint": {
        "hdri_id": "glasshouse_interior",
        "floor_asset": "concrete_block_wall",
        "wall_asset": "blue_plaster_wall",
        "floor_color": (0.38, 0.42, 0.38),
        "wall_color": (0.08, 0.42, 0.58),
        "trim_color": (0.72, 0.78, 0.72),
        "accent_color": (0.08, 0.72, 0.40),
        "secondary_color": (0.88, 0.62, 0.06),
        "dominant_rgb": (0.22, 0.38, 0.32),
        "set_kind": "glasshouse",
        "hdri_rotation_deg": 156.0,
        "hdri_strength": 0.26,
        "key": ((-4.0, -2.8, 5.0), 900.0, (0.74, 1.0, 0.84)),
        "fill": ((4.5, -0.8, 3.4), 470.0, (0.48, 0.78, 1.0)),
        "top": ((0.0, 4.8, 6.0), 760.0, (0.86, 1.0, 0.90)),
        "rim": ((5.0, 6.5, 3.5), 400.0, (1.0, 0.72, 0.25)),
    },
    "courtyard_terracotta": {
        "hdri_id": "courtyard",
        "floor_asset": "brick_floor",
        "wall_asset": "brick_wall_003",
        "floor_color": (0.52, 0.17, 0.045),
        "wall_color": (0.42, 0.16, 0.075),
        "trim_color": (0.68, 0.72, 0.72),
        "accent_color": (0.035, 0.36, 0.82),
        "secondary_color": (0.12, 0.70, 0.22),
        "dominant_rgb": (0.34, 0.17, 0.10),
        "set_kind": "courtyard",
        "hdri_rotation_deg": -112.0,
        "hdri_strength": 0.24,
        "key": ((-4.8, -3.0, 5.2), 1050.0, (1.0, 0.78, 0.50)),
        "fill": ((4.4, -0.6, 3.0), 420.0, (0.48, 0.72, 1.0)),
        "top": ((0.0, 5.0, 6.0), 820.0, (1.0, 0.92, 0.72)),
        "rim": ((-4.0, 6.4, 3.5), 360.0, (0.45, 0.75, 1.0)),
    },
    "foundry_safety": {
        "hdri_id": "industrial_workshop_foundry",
        "floor_asset": "asphalt_floor",
        "wall_asset": "box_profile_metal_sheet",
        "floor_color": (0.10, 0.11, 0.10),
        "wall_color": (0.30, 0.32, 0.28),
        "trim_color": (0.055, 0.06, 0.055),
        "accent_color": (0.92, 0.60, 0.01),
        "secondary_color": (0.84, 0.08, 0.035),
        "dominant_rgb": (0.20, 0.20, 0.16),
        "set_kind": "foundry",
        "hdri_rotation_deg": 18.0,
        "hdri_strength": 0.21,
        "key": ((-4.0, -3.4, 4.8), 1000.0, (1.0, 0.70, 0.34)),
        "fill": ((4.6, -0.8, 3.0), 430.0, (0.42, 0.62, 1.0)),
        "top": ((0.0, 5.0, 5.6), 720.0, (1.0, 0.82, 0.54)),
        "rim": ((5.0, 6.2, 3.7), 430.0, (1.0, 0.18, 0.05)),
    },
    "garage_teal": {
        "hdri_id": "auto_service",
        "floor_asset": "blue_metal_plate",
        "wall_asset": "concrete_block_wall",
        "floor_color": (0.07, 0.20, 0.23),
        "wall_color": (0.32, 0.36, 0.34),
        "trim_color": (0.05, 0.09, 0.10),
        "accent_color": (0.04, 0.72, 0.68),
        "secondary_color": (0.95, 0.23, 0.025),
        "dominant_rgb": (0.16, 0.27, 0.26),
        "set_kind": "garage",
        "hdri_rotation_deg": 92.0,
        "hdri_strength": 0.23,
        "key": ((-4.5, -3.2, 5.0), 980.0, (0.70, 1.0, 0.92)),
        "fill": ((4.2, -0.8, 3.1), 470.0, (0.46, 0.70, 1.0)),
        "top": ((0.0, 5.0, 5.7), 700.0, (0.78, 1.0, 0.94)),
        "rim": ((4.8, 6.0, 3.6), 430.0, (1.0, 0.34, 0.10)),
    },
    "neon_studio": {
        "hdri_id": "ferndale_studio_05",
        "floor_asset": "blue_metal_plate",
        "wall_asset": "blue_plaster_wall",
        "floor_color": (0.045, 0.06, 0.12),
        "wall_color": (0.14, 0.08, 0.34),
        "trim_color": (0.12, 0.04, 0.18),
        "accent_color": (0.92, 0.02, 0.56),
        "secondary_color": (0.10, 0.62, 1.0),
        "dominant_rgb": (0.15, 0.10, 0.32),
        "set_kind": "neon",
        "hdri_rotation_deg": -28.0,
        "hdri_strength": 0.18,
        "key": ((-4.2, -3.0, 4.8), 820.0, (1.0, 0.16, 0.48)),
        "fill": ((4.4, -0.7, 3.0), 610.0, (0.12, 0.42, 1.0)),
        "top": ((0.0, 4.5, 5.8), 600.0, (0.70, 0.18, 1.0)),
        "rim": ((-5.0, 6.0, 3.8), 500.0, (0.12, 1.0, 0.76)),
    },
}


def _asset_hdri_path(asset_id: str) -> Path:
    candidates = sorted((BACKGROUND_ASSET_ROOT / "hdris" / asset_id).glob("*.hdr"))
    if not candidates:
        raise FileNotFoundError(
            f"missing downloaded HDRI {asset_id!r} under {BACKGROUND_ASSET_ROOT}; "
            "run download_texture_realism_backgrounds.py first"
        )
    return candidates[-1]


def _background_texture_names(asset_id: str) -> dict[str, str]:
    root = BACKGROUND_ASSET_ROOT / "textures" / asset_id
    patterns = {
        "albedo": ("*_diff_*.jpg", "*_diffuse_*.jpg", "*_albedo_*.jpg"),
        "normal": ("*_nor_gl_*.jpg", "*_normal_*.jpg"),
        "roughness": ("*_rough_*.jpg", "*_roughness_*.jpg"),
        "ao": ("*_ao_*.jpg", "*_AO_*.jpg"),
    }
    names: dict[str, str] = {}
    for key, candidates in patterns.items():
        for pattern in candidates:
            matches = sorted(root.glob(pattern))
            if matches:
                names[key] = matches[0].name
                break
    return names


def _downloaded_texture_spec(asset_id: str) -> tuple[Path, dict[str, str]] | None:
    """Return a downloaded object/set texture only when its albedo exists."""

    root = BACKGROUND_ASSET_ROOT / "textures" / asset_id
    names = _background_texture_names(asset_id)
    if not root.is_dir() or not names.get("albedo"):
        return None
    return root, names


def _background_material(
    name: str,
    asset_id: str,
    tint: tuple[float, float, float],
    *,
    roughness: float = 0.78,
    metallic: float = 0.0,
    uv_scale: float = 2.0,
) -> bpy.types.Material:
    texture_dir = BACKGROUND_ASSET_ROOT / "textures" / asset_id
    names = _background_texture_names(asset_id)
    material = reference.pbr_material(
        name,
        texture_dir=texture_dir if names else None,
        texture_names=names,
        tint=tuple(0.25 + 1.65 * float(value) for value in tint),
        tint_strength=0.92,
        roughness=roughness,
        metallic=metallic,
        uv_scale=uv_scale,
        normal_strength=0.56,
        detail_bump_strength=0.016,
        detail_bump_scale=18.0,
    )
    _force_material_color(material, tint, factor=0.62)
    return material


def _force_material_color(
    material: bpy.types.Material,
    rgb,
    *,
    factor: float,
    texture_contrast: float = 0.0,
) -> None:
    """Keep PBR detail while making the requested hue visible in RGB output."""

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        return
    base_color = shader.inputs.get("Base Color")
    if base_color is None:
        return
    source = base_color.links[0].from_socket if base_color.links else None
    if base_color.links:
        links.remove(base_color.links[0])
    if source is None:
        base_color.default_value = (*tuple(float(value) for value in rgb), 1.0)
        return
    if texture_contrast > 0.0:
        # PBR albedos can be visually too low-contrast after hue separation at
        # 720p.  Remap their luminance into shades of the requested display
        # hue, preserving the real image texture's weave, grain, scratches,
        # and stains instead of painting a solid-color mask over it.
        luminance = nodes.new("ShaderNodeRGBToBW")
        luminance.name = f"Texture luminance {material.name}"
        ramp = nodes.new("ShaderNodeValToRGB")
        ramp.name = f"Texture contrast {material.name}"
        # Keep the hue but stretch the source texture over a much wider value
        # range.  The previous ramp was deliberately restrained for realism;
        # at 1280x720 that made the downloaded detail disappear after video
        # encoding.  A normalized hue keeps the objects separated while the
        # dark/bright endpoints make weave, grain, ribs, and scratches survive
        # downsampling.
        max_channel = max(float(value) for value in rgb) or 1.0
        hue = [float(value) / max_channel for value in rgb]
        ramp.color_ramp.elements[0].position = 0.24
        ramp.color_ramp.elements[0].color = tuple(
            min(1.0, 0.006 + 0.035 * value) for value in hue
        ) + (1.0,)
        ramp.color_ramp.elements[1].position = 0.66
        ramp.color_ramp.elements[1].color = tuple(
            min(1.0, 0.018 + 0.98 * value) for value in hue
        ) + (1.0,)
        ramp.color_ramp.interpolation = "EASE"
        links.new(source, luminance.inputs[0])
        links.new(luminance.outputs[0], ramp.inputs[0])
        colorize = nodes.new("ShaderNodeMixRGB")
        colorize.name = f"Texture colorize {material.name}"
        colorize.blend_type = "MIX"
        colorize.inputs[0].default_value = min(1.0, float(texture_contrast))
        links.new(source, colorize.inputs[1])
        links.new(ramp.outputs[0], colorize.inputs[2])
        source = colorize.outputs[0]

    mix = nodes.new("ShaderNodeMixRGB")
    mix.name = f"Display hue {material.name}"
    mix.blend_type = "MIX"
    mix.inputs[0].default_value = float(factor)
    mix.inputs[2].default_value = (*tuple(float(value) for value in rgb), 1.0)
    links.new(source, mix.inputs[1])
    links.new(mix.outputs[0], base_color)


def _add_albedo_relief(
    material: bpy.types.Material,
    *,
    strength: float = 0.34,
    distance: float = 0.045,
) -> None:
    """Make real albedo detail catch light in Eevee at final video scale."""

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        return
    albedo = None
    for node in nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        image_path = str(getattr(node.image, "filepath", "")).lower()
        if any(token in image_path for token in ("_diff_", "_diffuse_", "_albedo_")):
            albedo = node
            break
    if albedo is None:
        return
    vector_socket = albedo.inputs.get("Vector")
    if vector_socket is None or not vector_socket.links:
        return

    grayscale = nodes.new("ShaderNodeRGBToBW")
    grayscale.name = f"Albedo relief luminance {material.name}"
    links.new(albedo.outputs["Color"], grayscale.inputs[0])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.name = f"Albedo relief contrast {material.name}"
    ramp.color_ramp.elements[0].position = 0.28
    ramp.color_ramp.elements[1].position = 0.68
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    ramp.color_ramp.interpolation = "EASE"
    links.new(grayscale.outputs[0], ramp.inputs[0])

    # A very small emissive contribution is used only as a visibility aid.
    # It keeps the albedo pattern readable on faces that happen to be turned
    # away from the key/raking lights; it is not intended to make the object
    # look self-luminous.
    emission = shader.inputs.get("Emission") or shader.inputs.get("Emission Color")
    emission_strength = shader.inputs.get("Emission Strength")
    if emission is not None and emission_strength is not None:
        links.new(ramp.outputs["Color"], emission)
        emission_strength.default_value = 0.075

    relief = nodes.new("ShaderNodeBump")
    relief.name = f"Albedo relief bump {material.name}"
    relief.inputs["Strength"].default_value = float(strength)
    relief.inputs["Distance"].default_value = float(distance)
    links.new(ramp.outputs["Color"], relief.inputs["Height"])
    normal_input = shader.inputs.get("Normal")
    if normal_input is not None and normal_input.links:
        links.new(normal_input.links[0].from_socket, relief.inputs["Normal"])
    if normal_input is not None:
        while normal_input.links:
            links.remove(normal_input.links[0])
        links.new(relief.outputs["Normal"], normal_input)


def _add_macro_texture_emphasis(
    material: bpy.types.Material,
    texture_asset_id: str,
    *,
    object_index: int = 0,
) -> None:
    """Add a restrained, large-scale texture cue that survives 720p video.

    The downloaded albedo/normal/roughness maps remain the primary material.
    This secondary cue only prevents fine photographic detail from vanishing
    when a small actor is downsampled: fabric uses a coarse weave, wood uses
    irregular grain bands, and metal/rubber uses broad wear variation.
    """

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if shader is None:
        return
    base_input = shader.inputs.get("Base Color")
    if base_input is None or not base_input.links:
        return
    base_socket = base_input.links[0].from_socket

    albedo = None
    for node in nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        image_path = str(getattr(node.image, "filepath", "")).lower()
        if any(token in image_path for token in ("_diff_", "_diffuse_", "_albedo_")):
            albedo = node
            break
    if albedo is None:
        return
    vector_input = albedo.inputs.get("Vector")
    if vector_input is None or not vector_input.links:
        return
    vector_socket = vector_input.links[0].from_socket

    asset = texture_asset_id.lower()
    if any(token in asset for token in ("hessian", "fabric", "denim", "leather")):
        kind = "weave"
        scale_x, scale_y = 18.0, 24.0
        contrast = 0.38
        bump_strength = 0.26
    elif any(token in asset for token in ("wood", "oak")):
        kind = "grain"
        scale_x, scale_y = 4.5, 8.0
        contrast = 0.34
        bump_strength = 0.22
    elif any(token in asset for token in ("metal", "rust")):
        kind = "wear"
        scale_x, scale_y = 5.0, 3.0
        contrast = 0.42
        bump_strength = 0.20
    else:
        kind = "surface"
        scale_x, scale_y = 7.0, 7.0
        contrast = 0.20
        bump_strength = 0.18

    if kind == "wear":
        pattern_a = nodes.new("ShaderNodeTexNoise")
        pattern_a.name = f"Macro texture wear noise {material.name}"
        pattern_a.inputs["Scale"].default_value = 4.2
        pattern_a.inputs["Detail"].default_value = 6.0
        pattern_a.inputs["Roughness"].default_value = 0.78
        pattern_a.inputs["Distortion"].default_value = 1.4
    else:
        pattern_a = nodes.new("ShaderNodeTexWave")
        pattern_a.name = f"Macro texture {kind} A {material.name}"
        pattern_a.wave_type = "BANDS"
        pattern_a.bands_direction = "X"
        pattern_a.inputs["Scale"].default_value = scale_x
        pattern_a.inputs["Distortion"].default_value = 3.2
        pattern_a.inputs["Detail"].default_value = 5.0
        pattern_a.inputs["Detail Scale"].default_value = 2.0
    links.new(vector_socket, pattern_a.inputs["Vector"])

    pattern_b = nodes.new("ShaderNodeTexWave")
    pattern_b.name = f"Macro texture {kind} B {material.name}"
    pattern_b.wave_type = "BANDS"
    pattern_b.bands_direction = "Y"
    pattern_b.inputs["Scale"].default_value = scale_y
    pattern_b.inputs["Distortion"].default_value = 2.4 if kind != "wear" else 4.5
    pattern_b.inputs["Detail"].default_value = 4.0
    pattern_b.inputs["Detail Scale"].default_value = 2.0
    links.new(vector_socket, pattern_b.inputs["Vector"])

    combine = nodes.new("ShaderNodeMixRGB")
    combine.name = f"Macro texture combine {material.name}"
    combine.blend_type = "MULTIPLY"
    combine.inputs[0].default_value = 1.0
    links.new(pattern_a.outputs["Color"], combine.inputs[1])
    links.new(pattern_b.outputs["Color"], combine.inputs[2])
    pattern_ramp = nodes.new("ShaderNodeValToRGB")
    pattern_ramp.name = f"Macro texture contrast {material.name}"
    pattern_ramp.color_ramp.elements[0].position = 0.28 if kind == "wear" else 0.34
    pattern_ramp.color_ramp.elements[0].color = (0.08, 0.08, 0.08, 1.0)
    pattern_ramp.color_ramp.elements[1].position = 0.56 if kind == "wear" else 0.62
    pattern_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
    links.new(combine.outputs["Color"], pattern_ramp.inputs[0])

    mix = nodes.new("ShaderNodeMixRGB")
    mix.name = f"Macro texture overlay {material.name}"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = contrast
    links.new(base_socket, mix.inputs[1])
    links.new(pattern_ramp.outputs["Color"], mix.inputs[2])
    links.remove(base_input.links[0])
    links.new(mix.outputs["Color"], base_input)

    macro_bump = nodes.new("ShaderNodeBump")
    macro_bump.name = f"Macro texture bump {material.name}"
    macro_bump.inputs["Strength"].default_value = bump_strength
    macro_bump.inputs["Distance"].default_value = 0.035
    links.new(pattern_ramp.outputs["Color"], macro_bump.inputs["Height"])
    normal_input = shader.inputs.get("Normal")
    if normal_input is not None and normal_input.links:
        links.new(normal_input.links[0].from_socket, macro_bump.inputs["Normal"])
    if normal_input is not None:
        while normal_input.links:
            links.remove(normal_input.links[0])
        links.new(macro_bump.outputs["Normal"], normal_input)


def set_profile_hdri(scene: bpy.types.Scene, profile: dict) -> Path:
    """Use a newly downloaded HDRI, never one of the test70 three."""

    path = _asset_hdri_path(str(profile["hdri_id"]))
    world = bpy.data.worlds.new("Fast realism studio world") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = float(profile["hdri_strength"])
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(path), check_existing=True)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(float(profile["hdri_rotation_deg"]))
    texcoord = nodes.new("ShaderNodeTexCoord")
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return path


def _add_background_cylinder(name: str, location, radius: float, depth: float, material, *, horizontal: bool = False):
    bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    if horizontal:
        obj.rotation_euler[1] = math.radians(90.0)
    obj.data.materials.append(material)
    return obj


def _add_profile_props(profile: dict, trim, accent, secondary) -> None:
    """Add a sparse, profile-specific set so backgrounds are not flat color cards."""

    kind = profile["set_kind"]
    if kind == "warehouse":
        for index, x in enumerate((-9.0, -6.0, 6.0, 9.0)):
            reference.add_cube(f"Warehouse column {index}", (x, 7.72, 2.6), (0.13, 0.15, 2.6), trim, bevel=0.02)
        for index, x in enumerate((-5.0, 0.0, 5.0)):
            reference.add_cube(f"Warehouse loading panel {index}", (x, 7.68, 1.8), (1.8, 0.035, 0.85), secondary if index % 2 else accent, bevel=0.03)
        for index, x in enumerate((-6.5, 6.5)):
            reference.add_cube(f"Warehouse crate stack {index}", (x, 6.9, 0.45), (0.65, 0.45, 0.45), trim, bevel=0.04)
            reference.add_cube(f"Warehouse crate top {index}", (x, 6.9, 1.25), (0.65, 0.45, 0.32), accent, bevel=0.04)
    elif kind == "machine_shop":
        for index, z in enumerate((1.0, 2.0, 3.0)):
            _add_background_cylinder(f"Machine rear pipe {index}", (-5.6, 7.25, z), 0.10, 4.2, accent, horizontal=True)
        for index, x in enumerate((-4.8, 4.8)):
            reference.add_cube(f"Machine rack {index}", (x, 7.3, 1.5), (0.85, 0.18, 1.5), trim, bevel=0.025)
            reference.add_cube(f"Machine rack shelf {index}", (x, 6.9, 2.1), (1.15, 0.45, 0.07), secondary, bevel=0.015)
        _add_background_cylinder("Machine tank", (4.8, 6.8, 0.75), 0.52, 1.5, accent)
    elif kind == "color_studio":
        reference.add_cube("Color studio left sweep", (-5.2, 7.65, 2.2), (2.0, 0.04, 1.5), accent, bevel=0.06)
        reference.add_cube("Color studio right sweep", (5.2, 7.65, 2.2), (2.0, 0.04, 1.5), secondary, bevel=0.06)
        for index, x in enumerate((-3.0, 0.0, 3.0)):
            reference.add_cube(f"Color studio light bar {index}", (x, 7.55, 4.55), (0.08, 0.08, 0.42), accent if index % 2 else secondary, bevel=0.02)
    elif kind == "glasshouse":
        for index, x in enumerate((-9.0, -4.5, 0.0, 4.5, 9.0)):
            reference.add_cube(f"Glasshouse frame {index}", (x, 7.70, 2.8), (0.07, 0.08, 2.8), trim, bevel=0.01)
        for index, z in enumerate((1.1, 2.8, 4.5)):
            reference.add_cube(f"Glasshouse crossbar {index}", (0.0, 7.68, z), (9.0, 0.08, 0.06), trim, bevel=0.01)
        for index, x in enumerate((-6.0, 6.0)):
            _add_background_cylinder(f"Glasshouse planter {index}", (x, 6.7, 0.38), 0.48, 0.75, secondary)
            _add_background_cylinder(f"Glasshouse plant stem {index}", (x, 6.7, 1.25), 0.055, 1.1, accent)
    elif kind == "courtyard":
        for index, x in enumerate((-8.0, -4.0, 4.0, 8.0)):
            reference.add_cube(f"Courtyard pilaster {index}", (x, 7.65, 2.0), (0.38, 0.24, 2.0), trim, bevel=0.04)
        reference.add_cube("Courtyard lintel", (0.0, 7.62, 3.75), (8.4, 0.24, 0.28), trim, bevel=0.05)
        for index, x in enumerate((-5.8, 5.8)):
            _add_background_cylinder(f"Courtyard planter {index}", (x, 6.6, 0.42), 0.52, 0.82, accent)
    elif kind == "foundry":
        for index, x in enumerate((-8.0, -2.7, 2.7, 8.0)):
            reference.add_cube(f"Foundry beam {index}", (x, 7.68, 2.8), (0.16, 0.16, 2.8), trim, bevel=0.02)
        for index, x in enumerate((-5.3, 0.0, 5.3)):
            _add_background_cylinder(f"Foundry vessel {index}", (x, 6.8, 0.8), 0.62, 1.6, accent)
        reference.add_cube("Foundry safety stripe", (0.0, 7.58, 1.2), (8.5, 0.04, 0.12), secondary, bevel=0.01)
    elif kind == "garage":
        for index, x in enumerate((-7.5, -5.0, 5.0, 7.5)):
            reference.add_cube(f"Garage door rib {index}", (x, 7.66, 2.5), (0.08, 0.08, 2.5), trim, bevel=0.015)
        for index, z in enumerate((0.9, 1.8, 2.7, 3.6)):
            reference.add_cube(f"Garage door stripe {index}", (0.0, 7.58, z), (8.0, 0.04, 0.055), accent if index % 2 else secondary, bevel=0.01)
        for index, x in enumerate((-6.0, 6.0)):
            _add_background_cylinder(f"Garage tire stack {index}", (x, 6.8, 0.4), 0.46, 0.28, trim, horizontal=True)
            _add_background_cylinder(f"Garage tire stack upper {index}", (x, 6.8, 0.95), 0.46, 0.28, trim, horizontal=True)
    elif kind == "neon":
        reference.add_cube("Neon center panel", (0.0, 7.62, 2.2), (4.4, 0.04, 1.9), trim, bevel=0.04)
        for index, x in enumerate((-3.2, 0.0, 3.2)):
            reference.add_cube(f"Neon vertical strip {index}", (x, 7.52, 2.2), (0.09, 0.06, 1.65), accent if index != 1 else secondary, bevel=0.025)
        reference.add_cube("Neon upper strip", (0.0, 7.50, 4.2), (4.3, 0.06, 0.09), secondary, bevel=0.025)


def add_fast_studio_background(materials: dict[str, bpy.types.Material], profile_name: str) -> dict[str, object]:
    """Build a distinct textured set for the selected downloaded background."""

    profile = BACKGROUND_PROFILES[profile_name]
    floor = _background_material(f"{profile_name} floor", profile["floor_asset"], profile["floor_color"], roughness=0.82, uv_scale=2.8)
    wall = _background_material(f"{profile_name} wall", profile["wall_asset"], profile["wall_color"], roughness=0.86, uv_scale=2.3)
    trim = _background_material("background trim", "blue_metal_plate", profile["trim_color"], roughness=0.43, metallic=0.60, uv_scale=1.2)
    accent = _background_material("background accent", "blue_metal_plate", profile["accent_color"], roughness=0.48, metallic=0.25, uv_scale=1.5)
    secondary = _background_material("background secondary", "box_profile_metal_sheet", profile["secondary_color"], roughness=0.56, metallic=0.20, uv_scale=1.6)

    reference.add_cube("Fast studio floor", (0.0, 2.0, -0.055), (14.0, 11.0, 0.055), floor, bevel=0.0)
    reference.add_cube("Fast studio back wall", (0.0, 9.0, 3.0), (14.0, 0.06, 3.0), wall, bevel=0.0)
    reference.add_cube("Fast studio left wall", (-13.95, 2.0, 3.0), (0.06, 7.0, 3.0), wall, bevel=0.0)
    reference.add_cube("Fast studio right wall", (13.95, 2.0, 3.0), (0.06, 7.0, 3.0), wall, bevel=0.0)
    reference.add_cube("Fast studio back trim", (0.0, 8.86, 0.12), (13.9, 0.05, 0.12), trim, bevel=0.015)
    reference.add_cube("Fast studio accent panel", (0.0, 8.78, 2.35), (4.6, 0.035, 1.65), accent, bevel=0.035)
    reference.add_cube("Fast studio accent inset", (0.0, 8.73, 2.35), (4.05, 0.025, 1.18), secondary, bevel=0.02)
    _add_profile_props(profile, trim, accent, secondary)

    anchors = [
        list(profile["dominant_rgb"]),
        list(profile["floor_color"]),
        list(profile["wall_color"]),
        list(profile["accent_color"]),
        list(profile["secondary_color"]),
    ]
    return {
        "name": profile_name,
        "set_kind": profile["set_kind"],
        "hdri_id": profile["hdri_id"],
        "hdri": str(_asset_hdri_path(str(profile["hdri_id"]))),
        "floor_asset": profile["floor_asset"],
        "wall_asset": profile["wall_asset"],
        "background_color_anchors": anchors,
    }


def add_profile_lighting(profile_name: str) -> str:
    profile = BACKGROUND_PROFILES[profile_name]
    for label, key in (("Key", "key"), ("Fill", "fill"), ("Top", "top"), ("Rim", "rim")):
        location, energy, color = profile[key]
        reference.add_area_light(
            f"Fast studio {label}",
            location,
            (0.0, 1.0, 0.45),
            float(energy),
            3.5 if label == "Key" else 3.0,
            color,
        )
    # A compact, low-angle source is intentional: it turns the real albedo
    # relief and normal map into readable highlights instead of relying only
    # on broad softboxes, which tend to flatten small objects in video.
    reference.add_area_light(
        "Fast studio Texture rake",
        (-3.8, -1.8, 1.15),
        (0.0, 0.55, 0.35),
        330.0,
        1.15,
        (1.0, 0.93, 0.84),
    )
    reference.add_area_light(
        "Fast studio Texture counter-rake",
        (3.6, 0.25, 1.35),
        (0.0, 0.55, 0.40),
        210.0,
        1.0,
        (0.78, 0.88, 1.0),
    )
    return f"{profile_name}_four_point_softboxes"


def _color_distance(left, right) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def choose_display_colors(sample_id: str, names: list[str], profile_name: str) -> dict[str, dict[str, object]]:
    """Assign pairwise-separated display colors for one video.

    The physical material keys remain in metadata; this palette only controls
    the illustrative RGB appearance.  We compare every candidate against the
    floor, wall, accent, and secondary background colors, then against all
    already selected objects.  This prevents two actors or an actor/background
    pair from collapsing to the same visual color.
    """

    profile = BACKGROUND_PROFILES[profile_name]
    anchors = np.asarray(
        [
            profile["dominant_rgb"],
            profile["floor_color"],
            profile["wall_color"],
            profile["accent_color"],
            profile["secondary_color"],
        ],
        dtype=np.float64,
    )
    seed = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    palette = list(DISPLAY_PALETTE)
    selected: list[tuple[str, tuple[float, float, float]]] = []
    result: dict[str, dict[str, object]] = {}
    for object_index, name in enumerate(names):
        ranked = []
        for palette_index, (label, rgb) in enumerate(palette):
            anchor_distance = min(_color_distance(rgb, anchor) for anchor in anchors)
            object_distance = min(
                (_color_distance(rgb, previous_rgb) for _, previous_rgb in selected),
                default=1.0,
            )
            tie_break = (seed + object_index * 37 + palette_index * 101) % len(palette)
            ranked.append((min(anchor_distance, object_distance), anchor_distance, tie_break, label, rgb))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        chosen = None
        for candidate in ranked:
            _, anchor_distance, _, label, rgb = candidate
            object_distance = min(
                (_color_distance(rgb, previous_rgb) for _, previous_rgb in selected),
                default=1.0,
            )
            if anchor_distance >= 0.28 and object_distance >= 0.34:
                chosen = (label, rgb)
                break
        if chosen is None:
            chosen = (ranked[0][3], ranked[0][4])
        label, rgb = chosen
        selected.append((label, rgb))
        result[name] = {
            "label": label,
            "rgb": [float(value) for value in rgb],
            "min_background_distance": min(
                _color_distance(rgb, anchor) for anchor in anchors
            ),
            "min_other_object_distance": min(
                (_color_distance(rgb, previous_rgb) for _, previous_rgb in selected[:-1]),
                default=1.0,
            ),
        }
    return result


def _object_texture_spec(
    material_key: str,
    variant_index: int = 0,
) -> tuple[str, Path, dict[str, str], float, float, float]:
    """Choose a real downloaded PBR surface for each actor.

    The display hue is applied as a restrained mix later in the graph.  The
    downloaded albedo therefore remains visible instead of turning the actor
    into a flat solid-color primitive.
    """

    key = material_key.lower()
    if "rubber" in key or "wheel" in key or "tire" in key:
        candidates = ("rubberized_track", "rubber_tiles")
        legacy_id = "rubber_tiles"
        legacy_dir = reference.EXTRA_TEXTURE_ROOT / legacy_id
        legacy_names = _pbr_texture_names()
        settings = (0.70, 0.04, 0.72)
    elif "metal" in key:
        candidates = ("rusty_metal_03", "metal_plate_02", "painted_metal_shutter")
        legacy_id = "metal_plate"
        legacy_dir = reference.EXTRA_TEXTURE_ROOT / legacy_id
        legacy_names = _metal_texture_names()
        settings = (0.46, 0.38, 0.48)
    elif "cardboard" in key or "paper" in key:
        candidates = ("hessian_230", "fabric_leather_01")
        legacy_id = "beige_wall_001"
        legacy_dir = reference.TEXTURE_ROOT / legacy_id
        legacy_names = _cardboard_texture_names()
        settings = (0.84, 0.02, 0.62)
    elif "wood" in key or "domino" in key:
        candidates = ("oak_wood_planks", "dark_wood")
        legacy_id = "wood_floor"
        legacy_dir = reference.TEXTURE_ROOT / legacy_id
        legacy_names = {
            "albedo": "wood_floor_diff_2k.jpg",
            "normal": "wood_floor_nor_gl_2k.jpg",
            "roughness": "wood_floor_rough_2k.jpg",
            "ao": "wood_floor_ao_2k.jpg",
        }
        settings = (0.58, 0.04, 0.66)
    elif "fabric" in key or "textile" in key or "rope" in key:
        candidates = ("denim_fabric_03", "fabric_leather_01")
        legacy_id = "rubber_tiles"
        legacy_dir = reference.EXTRA_TEXTURE_ROOT / legacy_id
        legacy_names = _pbr_texture_names()
        settings = (0.92, 0.0, 1.20)
    else:
        # Synthetic plastic labels in the source metadata are rendered as
        # varied coated surfaces: visible hessian weave, painted-metal ribs,
        # and wood grain.  This keeps the shapes recognizable while making
        # the image texture legible at the final 1280x720 display size.
        candidates = ("hessian_230", "rusty_metal_03", "oak_wood_planks")
        legacy_id = "rubber_tiles"
        legacy_dir = reference.EXTRA_TEXTURE_ROOT / legacy_id
        legacy_names = _pbr_texture_names()
        settings = (0.82, 0.02, 0.55)

    for offset in range(len(candidates)):
        asset_id = candidates[(int(variant_index) + offset) % len(candidates)]
        downloaded = _downloaded_texture_spec(asset_id)
        if downloaded is not None:
            texture_dir, texture_names = downloaded
            return (asset_id, texture_dir, texture_names, *settings)
    return (f"legacy:{legacy_id}", legacy_dir, legacy_names, *settings)


def build_actor_materials(
    actors: dict[str, dict],
    names: list[str],
    sample_id: str,
    profile_name: str,
) -> tuple[dict[str, bpy.types.Material], dict[str, dict[str, object]]]:
    display_colors = choose_display_colors(sample_id, names, profile_name)
    materials: dict[str, bpy.types.Material] = {}
    for object_index, name in enumerate(names):
        actor = actors[name]
        material_key = str(actor.get("material_key", ""))
        texture_asset_id, texture_dir, texture_names, roughness, metallic, uv_scale = _object_texture_spec(
            material_key,
            object_index,
        )
        rgb = display_colors[name]["rgb"]
        # Keep the downloaded albedo dominant enough to show weave, scratches,
        # grain, or rubber relief; the separate hue mix still separates actors.
        tint = tuple(0.24 + 1.85 * float(value) for value in rgb)
        materials[name] = reference.pbr_material(
            f"Actor_{name}_{display_colors[name]['label']}",
            texture_dir=texture_dir,
            texture_names=texture_names,
            tint=tint,
            tint_strength=0.04,
            roughness=roughness,
            metallic=metallic,
            uv_scale=uv_scale,
            normal_strength=1.50,
            detail_bump_strength=0.105,
            detail_bump_scale=20.0,
        )
        _force_material_color(materials[name], rgb, factor=0.0, texture_contrast=1.0)
        _add_albedo_relief(materials[name], strength=0.72, distance=0.080)
        _add_macro_texture_emphasis(
            materials[name],
            texture_asset_id,
            object_index=object_index,
        )
        display_colors[name]["source_material_key"] = material_key
        display_colors[name]["texture_asset_id"] = texture_asset_id
        display_colors[name]["texture_maps"] = sorted(texture_names)
    return materials, display_colors


def _set_material(obj: bpy.types.Object, material) -> None:
    if hasattr(obj.data, "materials"):
        obj.data.materials.append(material)


def _bevel(obj: bpy.types.Object, width: float) -> None:
    if width <= 0.0:
        return
    modifier = obj.modifiers.new("Soft manufactured edges", "BEVEL")
    modifier.width = float(width)
    modifier.segments = 3


def add_generic_actor(name: str, actor: dict, material) -> bpy.types.Object:
    shape = str(actor.get("shape", "box"))
    size = actor.get("size", {})
    position = tuple(float(v) for v in actor.get("position", (0.0, 0.0, 0.0)))

    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48,
            ring_count=24,
            radius=float(size["radius"]),
            location=position,
        )
        obj = bpy.context.object
        bpy.ops.object.shade_smooth()
    elif shape in {"cylinder", "puck", "wheel_thick"}:
        radius = float(size.get("radius", size.get("flange_radius", 0.15)))
        depth = float(size.get("height", size.get("width", 0.12)))
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=64,
            radius=radius,
            depth=depth,
            location=position,
        )
        obj = bpy.context.object
        bpy.ops.object.shade_smooth()
        _bevel(obj, min(radius * 0.09, 0.018))
    else:
        # A rounded cube is a close visual analogue for both box and the
        # rounded_box collision primitive while retaining the same dimensions.
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=position)
        obj = bpy.context.object
        obj.scale = (
            float(size.get("hx", 0.2)),
            float(size.get("hy", 0.2)),
            float(size.get("hz", 0.2)),
        )
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        _bevel(obj, float(size.get("corner_radius", 0.018)))

    obj.name = name
    _set_material(obj, material)
    return obj


def add_camera(meta: dict, width: int, height: int) -> bpy.types.Object:
    camera_spec = meta["camera"]
    data = bpy.data.cameras.new("Texture realism camera")
    data.type = "PERSP"
    data.sensor_fit = "VERTICAL"
    data.angle_y = math.radians(float(camera_spec.get("yfov_deg", 50.0)))
    data.lens = 50.0
    data.clip_start = 0.01
    data.clip_end = 100.0
    obj = bpy.data.objects.new("Texture realism camera", data)
    bpy.context.collection.objects.link(obj)
    obj.location = tuple(float(v) for v in camera_spec["eye"])
    target = Vector(tuple(float(v) for v in camera_spec["target"]))
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = obj
    return obj


def add_motion_tracking_camera(
    camera: bpy.types.Object,
    meta: dict,
    positions: np.ndarray,
    frame_count: int,
) -> dict[str, object]:
    """Smoothly follow the dynamic-object centroid so wide variants stay visible."""

    dynamic_indices = [
        index for index, actor in enumerate(meta.get("objects", []))
        if bool(actor.get("dynamic", True))
    ]
    if not dynamic_indices:
        return {"enabled": False, "reason": "no dynamic objects"}

    centers = np.mean(positions[:frame_count, dynamic_indices, :], axis=1)
    smoothed = centers.copy()
    alpha = 0.16
    for index in range(1, len(smoothed)):
        smoothed[index] = alpha * centers[index] + (1.0 - alpha) * smoothed[index - 1]

    original_eye = np.asarray(meta["camera"]["eye"], dtype=np.float64)
    original_target = np.asarray(meta["camera"]["target"], dtype=np.float64)
    offset = original_eye - original_target
    distance = float(np.linalg.norm(offset))
    direction = offset / max(distance, 1e-6)
    # Keep enough room for the widest same-frame object arrangement.  The
    # camera follows the centroid, so use the radius around each *frame's*
    # centroid here; using the trajectory's global spread would make the
    # camera unnecessarily distant whenever the whole interaction translates.
    dynamic_positions = positions[:frame_count, dynamic_indices, :]
    intra_frame_radius = float(
        np.max(np.linalg.norm(dynamic_positions - centers[:, None, :], axis=2))
    )
    distance = max(distance, 1.35 * intra_frame_radius / math.tan(math.radians(float(meta["camera"].get("yfov_deg", 50.0))) * 0.5) + 0.75)
    target_offset_z = float(original_target[2] - np.mean(centers[0, 2]))

    for frame_index in range(frame_count):
        target = smoothed[frame_index].copy()
        target[2] += target_offset_z
        eye = target + direction * distance
        camera.location = tuple(float(value) for value in eye)
        camera.rotation_euler = (Vector(tuple(float(value) for value in target)) - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame_index + 1)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame_index + 1)

    if camera.animation_data and camera.animation_data.action:
        for curve in camera.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"

    first_target = smoothed[0].copy()
    first_target[2] += target_offset_z
    first_eye = first_target + direction * distance
    return {
        "enabled": True,
        "mode": "smoothed_dynamic_centroid_follow",
        "dynamic_object_indices": dynamic_indices,
        "distance_m": distance,
        "smoothing_alpha": alpha,
        "first_eye": [float(value) for value in first_eye],
        "first_target": [float(value) for value in first_target],
        "last_target": [float(value) for value in (smoothed[-1] + np.asarray([0.0, 0.0, target_offset_z]))],
    }


def load_states(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    arrays = np.load(path, allow_pickle=False)
    names = [str(v) for v in arrays["object_names"]]
    positions = np.asarray(arrays["positions"], dtype=np.float64)
    quats_xyzw = np.asarray(arrays["quats"], dtype=np.float64)
    frame_times = np.asarray(arrays["frame_times"], dtype=np.float64)
    if positions.ndim != 3 or quats_xyzw.shape[:2] != positions.shape[:2]:
        raise ValueError(f"unexpected state shapes in {path}: {positions.shape}, {quats_xyzw.shape}")
    return names, positions, quats_xyzw, frame_times


def set_animation(objects, names, positions, quats_xyzw, frame_count: int) -> None:
    for object_index, name in enumerate(names):
        obj = objects[name]
        obj.rotation_mode = "QUATERNION"
        for frame_index in range(frame_count):
            obj.location = tuple(float(v) for v in positions[frame_index, object_index])
            q = quats_xyzw[frame_index, object_index]
            obj.rotation_quaternion = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
            obj.keyframe_insert(data_path="location", frame=frame_index + 1)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_index + 1)
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    names, positions, quats_xyzw, frame_times = load_states(args.states)
    frame_count = min(len(frame_times), args.frame_limit) if args.frame_limit > 0 else len(frame_times)
    if frame_count <= 0:
        raise ValueError("state file contains no frames")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference.clear_scene()
    scene = bpy.context.scene
    config = SimpleNamespace(
        engine="BLENDER_EEVEE",
        width=args.width,
        height=args.height,
        samples=args.samples,
        exposure=args.exposure,
        output_format="PNG",
        device="CPU",
    )
    enabled_devices = reference.configure_cycles(scene, config, fps=int(meta.get("fps", 30)))
    materials = build_demo_materials()

    # Use a furniture-free studio profile rather than the F11/test70 room.
    # The object PBR maps still come from the same local asset library, but
    # the set, palette, HDRI orientation, and lights are independent.
    camera = add_camera(meta, args.width, args.height)
    room_scene = add_fast_studio_background(materials, args.background_profile)
    hdri_path = set_profile_hdri(scene, BACKGROUND_PROFILES[args.background_profile])
    lighting_preset = add_profile_lighting(args.background_profile)

    actors = {str(item["name"]): item for item in meta.get("objects", [])}
    for name in names:
        if name not in actors:
            raise KeyError(f"state object {name!r} missing from metadata {args.meta}")
    sample_id = str(meta.get("key", args.meta.stem))
    actor_materials, display_colors = build_actor_materials(
        actors,
        names,
        sample_id,
        args.background_profile,
    )
    objects = {}
    for name in names:
        objects[name] = add_generic_actor(name, actors[name], actor_materials[name])
    set_animation(objects, names, positions, quats_xyzw, frame_count)
    camera_tracking = add_motion_tracking_camera(camera, meta, positions, frame_count)

    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.filepath = str(args.output_dir / "frame_")
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.30
    bpy.ops.render.render(animation=True)

    material_keys = {str(item["name"]): str(item.get("material_key", "")) for item in meta.get("objects", [])}
    report = {
        "schema_version": "physv_fast_realism_eevee_v1",
        "engine": "BLENDER_EEVEE",
        "cycles_used": False,
        "sample_id": sample_id,
        "frame_count": frame_count,
        "fps": int(meta.get("fps", 30)),
        "resolution": [args.width, args.height],
        "room_scene": room_scene,
        "background_profile": args.background_profile,
        "lighting_preset": lighting_preset,
        "hdri": str(hdri_path),
        "camera_tracking": camera_tracking,
        "effective_camera": {
            "eye": camera_tracking.get("first_eye", list(meta["camera"]["eye"])),
            "target": camera_tracking.get("first_target", list(meta["camera"]["target"])),
            "up": list(meta["camera"].get("up", [0.0, 0.0, 1.0])),
            "yfov_deg": float(meta["camera"].get("yfov_deg", 50.0)),
        },
        "material_assignments": material_keys,
        "display_material_assignments": display_colors,
        "color_separation": {
            "constraint": "per-video object/background RGB distance",
            "background_anchors": room_scene["background_color_anchors"],
            "objects": display_colors,
        },
        "texture_profile": "polyhaven_pbr_fast_eevee_varied_backgrounds_textured_objects",
        "texture_sources": [
            str(reference.TEXTURE_ROOT),
            str(reference.EXTRA_TEXTURE_ROOT),
            str(BACKGROUND_ASSET_ROOT),
        ],
        "state_source": str(args.states),
    }
    (args.output_dir / "render_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
