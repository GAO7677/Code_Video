#!/usr/bin/env python3
"""Render an exported PhysV sample from trajectory truth with Blender Cycles.

This script is executed by Blender. It does not run physics again: object poses
come directly from ``raw/trajectories.npz`` so the new RGB render remains
aligned with the existing supervision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ASSET_ROOT = Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1")
TEXTURE_ROOT = ASSET_ROOT / "textures"
EXTRA_TEXTURE_ROOT = Path("/data/gaoya/agent-data/assets/polyhaven_textures_20260820")
REALISM_TEXTURE_ROOT = Path(
    "/data/gaoya/agent-data/assets/texture_realism_backgrounds_20260825/textures"
)
RAMP_BLOCK_TEXTURE_ROOT = Path(
    "/data/gaoya/agent-data/assets/polyhaven_textures_20260829/wood_peeling_paint_weathered"
)
RAMP_FLOOR_TEXTURE_ROOT = Path(
    "/data/gaoya/agent-data/assets/polyhaven_textures_20260829/pavement_01"
)
HDRI_ROOT = ASSET_ROOT / "hdris"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--trajectory-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--frame-limit", type=int, default=0, help="Render only the first N frames; 0 renders all frames.")
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--device", choices=("CUDA", "CPU"), default="CUDA")
    parser.add_argument("--output-format", choices=("PNG", "OPEN_EXR"), default="PNG")
    parser.add_argument(
        "--material-overrides-json",
        type=Path,
        default=None,
        help="JSON object mapping actor names to material-library keys.",
    )
    parser.add_argument(
        "--basketball-texture",
        type=Path,
        default=None,
        help="UV texture used by the optional basketball material.",
    )
    parser.add_argument(
        "--edge-clarity",
        action="store_true",
        help=(
            "Add a small render-only bevel, weighted normals and restrained "
            "grazing-angle highlight to non-sphere actors."
        ),
    )
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.materials, bpy.data.curves, bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            collection.remove(item)


def configure_cycles(scene: bpy.types.Scene, args: argparse.Namespace, fps: int) -> list[str]:
    scene.render.engine = args.engine
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = args.output_format
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16" if args.output_format == "OPEN_EXR" else "8"
    if args.output_format == "OPEN_EXR":
        scene.render.image_settings.exr_codec = "ZIP"
    scene.render.fps = fps
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 0.30
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = args.exposure
    scene.view_settings.gamma = 1.0

    enabled_devices: list[str] = []
    if args.engine == "CYCLES":
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.render.use_persistent_data = True
        scene.cycles.preview_samples = min(args.samples, 16)
        scene.cycles.max_bounces = 6
        scene.cycles.diffuse_bounces = 3
        scene.cycles.glossy_bounces = 3
        scene.cycles.transparent_max_bounces = 2
        scene.cycles.device = "GPU" if args.device == "CUDA" else "CPU"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.get_devices()
        if args.device == "CUDA":
            preferences.compute_device_type = "CUDA"
        for device in preferences.devices:
            device.use = device.type == args.device
            if device.use:
                enabled_devices.append(f"{device.type}:{device.name}")
    else:
        scene.eevee.taa_render_samples = args.samples
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 3.0
        scene.eevee.gtao_factor = 1.25
        scene.eevee.use_soft_shadows = True
        enabled_devices.append("BLENDER_EEVEE")
    return enabled_devices


def image_node(nodes, path: Path, *, non_color: bool = False):
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(str(path), check_existing=True)
    if non_color:
        try:
            node.image.colorspace_settings.name = "Non-Color"
        except TypeError:
            node.image.colorspace_settings.name = "Linear"
    node.interpolation = "Linear"
    node.extension = "REPEAT"
    return node


def pbr_material(
    name: str,
    *,
    texture_dir: Path | None,
    texture_names: dict[str, str] | None = None,
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tint_strength: float = 0.0,
    roughness: float = 0.55,
    metallic: float = 0.0,
    uv_scale: float = 1.0,
    texture_coordinate: str = "Generated",
    normal_strength: float = 0.42,
    detail_bump_strength: float = 0.0,
    detail_bump_scale: float = 24.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Base Color"].default_value = (*tint, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (uv_scale, uv_scale, uv_scale)
    if texture_coordinate not in {"Generated", "UV"}:
        raise ValueError(f"unsupported texture coordinate mode: {texture_coordinate!r}")
    links.new(texcoord.outputs[texture_coordinate], mapping.inputs["Vector"])

    names = texture_names or {}
    albedo_path = texture_dir / names["albedo"] if texture_dir and names.get("albedo") else None
    ao_path = texture_dir / names["ao"] if texture_dir and names.get("ao") else None
    normal_path = texture_dir / names["normal"] if texture_dir and names.get("normal") else None
    rough_path = texture_dir / names["roughness"] if texture_dir and names.get("roughness") else None

    color_socket = None
    if albedo_path and albedo_path.is_file():
        albedo = image_node(nodes, albedo_path)
        links.new(mapping.outputs["Vector"], albedo.inputs["Vector"])
        color_socket = albedo.outputs["Color"]
    if ao_path and ao_path.is_file() and color_socket is not None:
        ao = image_node(nodes, ao_path, non_color=True)
        multiply_ao = nodes.new("ShaderNodeMixRGB")
        multiply_ao.blend_type = "MULTIPLY"
        multiply_ao.inputs["Fac"].default_value = 0.55
        links.new(mapping.outputs["Vector"], ao.inputs["Vector"])
        links.new(color_socket, multiply_ao.inputs[1])
        links.new(ao.outputs["Color"], multiply_ao.inputs[2])
        color_socket = multiply_ao.outputs["Color"]
    if color_socket is not None and tint_strength > 0.0:
        tint_node = nodes.new("ShaderNodeMixRGB")
        tint_node.blend_type = "MULTIPLY"
        tint_node.inputs["Fac"].default_value = tint_strength
        tint_node.inputs[2].default_value = (*tint, 1.0)
        links.new(color_socket, tint_node.inputs[1])
        color_socket = tint_node.outputs["Color"]
    if color_socket is not None:
        links.new(color_socket, shader.inputs["Base Color"])

    if rough_path and rough_path.is_file():
        rough = image_node(nodes, rough_path, non_color=True)
        links.new(mapping.outputs["Vector"], rough.inputs["Vector"])
        links.new(rough.outputs["Color"], shader.inputs["Roughness"])
    normal_socket = None
    if normal_path and normal_path.is_file():
        normal = image_node(nodes, normal_path, non_color=True)
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = normal_strength
        links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        normal_socket = normal_map.outputs["Normal"]
    if detail_bump_strength > 0.0:
        detail_noise = nodes.new("ShaderNodeTexNoise")
        detail_noise.inputs["Scale"].default_value = detail_bump_scale
        detail_noise.inputs["Detail"].default_value = 8.0
        detail_noise.inputs["Roughness"].default_value = 0.45
        links.new(mapping.outputs["Vector"], detail_noise.inputs["Vector"])
        detail_bump = nodes.new("ShaderNodeBump")
        detail_bump.inputs["Strength"].default_value = detail_bump_strength
        detail_bump.inputs["Distance"].default_value = 0.02
        links.new(detail_noise.outputs["Fac"], detail_bump.inputs["Height"])
        if normal_socket is not None:
            links.new(normal_socket, detail_bump.inputs["Normal"])
        links.new(detail_bump.outputs["Normal"], shader.inputs["Normal"])
    elif normal_socket is not None:
        links.new(normal_socket, shader.inputs["Normal"])
    return material


def procedural_material(
    name: str,
    color: tuple[float, float, float],
    *,
    metallic: float,
    roughness: float,
    noise_scale: float = 7.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.65
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.10
    bump.inputs["Distance"].default_value = 0.025
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def basketball_material(name: str, texture_path: Path) -> bpy.types.Material:
    """Build a UV-mapped basketball material without changing object geometry."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.inputs["Roughness"].default_value = 0.78
    if "Specular" in shader.inputs:
        shader.inputs["Specular"].default_value = 0.28
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    albedo = image_node(nodes, texture_path)
    albedo.extension = "REPEAT"
    links.new(mapping.outputs["Vector"], albedo.inputs["Vector"])
    links.new(albedo.outputs["Color"], shader.inputs["Base Color"])

    # The downloaded map contains the leather dimples and seams. A restrained
    # bump keeps that detail visible while leaving the sphere mesh unchanged.
    to_bw = nodes.new("ShaderNodeRGBToBW")
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.012
    links.new(albedo.outputs["Color"], to_bw.inputs["Color"])
    links.new(to_bw.outputs["Val"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return material


def colorized_texture_material(
    name: str,
    *,
    texture_dir: Path,
    texture_names: dict[str, str],
    color: tuple[float, float, float],
    roughness: float = 0.62,
    uv_scale: float = 2.4,
    normal_strength: float = 0.58,
) -> bpy.types.Material:
    """Keep a high-contrast image texture while changing its hue.

    The regular palette materials intentionally use strong flat tints.  That
    is useful for the base benchmark, but it can hide low-contrast albedo maps
    on small dynamic actors.  Refine variants use the COLOR blend mode: a
    high-contrast wood albedo supplies luminance/detail and ``color`` supplies
    hue and saturation, so the grain remains visible instead of being
    multiplied away.
    """
    material = pbr_material(
        name,
        texture_dir=texture_dir,
        texture_names=texture_names,
        roughness=roughness,
        uv_scale=uv_scale,
        normal_strength=normal_strength,
        detail_bump_strength=0.018,
        detail_bump_scale=18.0,
    )
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    if shader is None:
        raise RuntimeError(f"{name}: Principled BSDF node missing")
    base_link = next(
        (link for link in links if link.to_node == shader and link.to_socket.name == "Base Color"),
        None,
    )
    if base_link is None:
        raise RuntimeError(f"{name}: textured Base Color link missing")
    source_socket = base_link.from_socket
    links.remove(base_link)
    colorize = nodes.new("ShaderNodeMixRGB")
    colorize.name = "Visible texture colorization"
    colorize.label = "Wood albedo + visible hue"
    colorize.blend_type = "COLOR"
    colorize.inputs["Fac"].default_value = 0.72
    colorize.inputs[2].default_value = (*color, 1.0)
    links.new(source_socket, colorize.inputs[1])
    links.new(colorize.outputs["Color"], shader.inputs["Base Color"])
    return material


def material_library(basketball_texture: Path | None = None) -> dict[str, bpy.types.Material]:
    wood_names = {
        "albedo": "wood_floor_diff_2k.jpg",
        "normal": "wood_floor_nor_gl_2k.jpg",
        "roughness": "wood_floor_rough_2k.jpg",
        "ao": "wood_floor_ao_2k.jpg",
    }
    peeling_paint_wood_names = {
        "albedo": "wood_peeling_paint_weathered_diff_2k.jpg",
        "normal": "wood_peeling_paint_weathered_nor_gl_2k.jpg",
        "roughness": "wood_peeling_paint_weathered_rough_2k.jpg",
        "ao": "wood_peeling_paint_weathered_ao_2k.jpg",
    }
    pavement_01_names = {
        "albedo": "pavement_01_diff_2k.jpg",
        "normal": "pavement_01_nor_gl_2k.jpg",
        "roughness": "pavement_01_rough_2k.jpg",
        "ao": "pavement_01_ao_2k.jpg",
    }
    wall_names = {
        "albedo": "beige_wall_001_diff_2k.jpg",
        "normal": "beige_wall_001_nor_gl_2k.jpg",
        "roughness": "beige_wall_001_rough_2k.jpg",
        "ao": "beige_wall_001_ao_2k.jpg",
    }
    leather_names = {
        "albedo": "brown_leather_albedo_2k.jpg",
        "normal": "brown_leather_nor_gl_2k.jpg",
        "roughness": "brown_leather_rough_2k.jpg",
        "ao": "brown_leather_ao_2k.jpg",
    }
    leather_surface_names = {
        "normal": "brown_leather_nor_gl_2k.jpg",
        "roughness": "brown_leather_rough_2k.jpg",
        "ao": "brown_leather_ao_2k.jpg",
    }
    concrete_names = {
        "albedo": "painted_concrete_diff_2k.jpg",
        "normal": "painted_concrete_nor_gl_2k.jpg",
        "roughness": "painted_concrete_rough_2k.jpg",
        "ao": "painted_concrete_ao_2k.jpg",
    }
    concrete_surface_names = {
        "normal": "painted_concrete_nor_gl_2k.jpg",
        "roughness": "painted_concrete_rough_2k.jpg",
        "ao": "painted_concrete_ao_2k.jpg",
    }
    fabric_names = {
        "normal": "fabric_pattern_07_nor_gl_2k.jpg",
        "roughness": "fabric_pattern_07_rough_2k.jpg",
        "ao": "fabric_pattern_07_ao_2k.jpg",
    }
    rubber_tile_names = {
        "albedo": "rubber_tiles_diff_2k.jpg",
        "normal": "rubber_tiles_nor_gl_2k.jpg",
        "roughness": "rubber_tiles_rough_2k.jpg",
        "ao": "rubber_tiles_ao_2k.jpg",
    }
    metal_plate_names = {
        "albedo": "metal_plate_diff_2k.jpg",
        "normal": "metal_plate_nor_gl_2k.jpg",
        "roughness": "metal_plate_rough_2k.jpg",
        "ao": "metal_plate_ao_2k.jpg",
    }
    worn_concrete_names = {
        "albedo": "concrete_floor_worn_001_diff_2k.jpg",
        "normal": "concrete_floor_worn_001_nor_gl_2k.jpg",
        "roughness": "concrete_floor_worn_001_rough_2k.jpg",
        "ao": "concrete_floor_worn_001_ao_2k.jpg",
    }
    denim_names = {
        "albedo": "denim_fabric_04_diff_2k.jpg",
        "normal": "denim_fabric_04_nor_gl_2k.jpg",
        "roughness": "denim_fabric_04_rough_2k.jpg",
        "ao": "denim_fabric_04_ao_2k.jpg",
    }
    natural_oak_names = {
        "albedo": "oak_wood_planks_diff_2k.jpg",
        "normal": "oak_wood_planks_nor_gl_2k.jpg",
        "roughness": "oak_wood_planks_rough_2k.jpg",
        "ao": "oak_wood_planks_ao_2k.jpg",
    }
    natural_dark_wood_names = {
        "albedo": "dark_wood_diff_2k.jpg",
        "normal": "dark_wood_nor_gl_2k.jpg",
        "roughness": "dark_wood_rough_2k.jpg",
        "ao": "dark_wood_ao_2k.jpg",
    }
    natural_rubber_names = {
        "albedo": "rubberized_track_diff_2k.jpg",
        "normal": "rubberized_track_nor_gl_2k.jpg",
        "roughness": "rubberized_track_rough_2k.jpg",
        "ao": "rubberized_track_ao_2k.jpg",
    }
    materials = {
        "floor": pbr_material("PBR_Wood_Floor", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, roughness=0.48, uv_scale=3.0, normal_strength=0.58, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "floor_cool": pbr_material("PBR_Cool_Wood_Floor", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(0.62, 0.78, 0.92), tint_strength=0.76, roughness=0.52, uv_scale=3.0, normal_strength=0.58, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "floor_dark_wood": pbr_material("PBR_Dark_Wood_Floor", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(0.12, 0.16, 0.22), tint_strength=0.78, roughness=0.58, uv_scale=3.0, normal_strength=0.60, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "floor_concrete": pbr_material("PBR_Light_Concrete_Floor", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.16, 0.19, 0.23), roughness=0.86, uv_scale=2.4, normal_strength=0.54, detail_bump_strength=0.018, detail_bump_scale=18.0),
        "floor_terracotta": pbr_material("PBR_Terracotta_Floor", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.24, 0.065, 0.035), roughness=0.84, uv_scale=2.8, normal_strength=0.56, detail_bump_strength=0.020, detail_bump_scale=18.0),
        "floor_slate": pbr_material("PBR_Slate_Floor", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.055, 0.075, 0.11), roughness=0.88, uv_scale=2.2, normal_strength=0.54, detail_bump_strength=0.016, detail_bump_scale=18.0),
        "wood": pbr_material("PBR_Wood", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, roughness=0.50, uv_scale=2.0, normal_strength=0.62, detail_bump_strength=0.015, detail_bump_scale=12.0),
        "red_wood": pbr_material("PBR_Red_Wood", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(1.15, 0.30, 0.23), tint_strength=0.72, roughness=0.52, uv_scale=2.4, normal_strength=0.62, detail_bump_strength=0.016, detail_bump_scale=12.0),
        "wood_peeling_paint": pbr_material("PBR_Wood_Peeling_Paint_Weathered_Warm", texture_dir=RAMP_BLOCK_TEXTURE_ROOT, texture_names=peeling_paint_wood_names, tint=(1.20, 0.38, 0.10), tint_strength=0.78, roughness=0.72, uv_scale=2.0, texture_coordinate="UV", normal_strength=0.55, detail_bump_strength=0.012, detail_bump_scale=18.0),
        "floor_stone_pavement": pbr_material("PBR_Stone_Pavement_01", texture_dir=RAMP_FLOOR_TEXTURE_ROOT, texture_names=pavement_01_names, roughness=0.86, uv_scale=5.0, normal_strength=0.62, detail_bump_strength=0.012, detail_bump_scale=20.0),
        "wall": pbr_material("PBR_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, roughness=0.82, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "wall_cool": pbr_material("PBR_Cool_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, tint=(0.66, 0.82, 1.02), tint_strength=0.68, roughness=0.84, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "wall_green": pbr_material("PBR_Sage_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, tint=(0.42, 0.64, 0.48), tint_strength=0.72, roughness=0.85, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "wall_gray": pbr_material("PBR_Gray_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, tint=(0.58, 0.64, 0.72), tint_strength=0.76, roughness=0.86, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "wall_rose": pbr_material("PBR_Dusty_Rose_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, tint=(0.72, 0.40, 0.34), tint_strength=0.68, roughness=0.86, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "wall_charcoal": pbr_material("PBR_Charcoal_Wall", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.085, 0.105, 0.14), roughness=0.90, uv_scale=2.4, normal_strength=0.56, detail_bump_strength=0.014, detail_bump_scale=20.0),
        "concrete": pbr_material("PBR_Concrete", texture_dir=EXTRA_TEXTURE_ROOT / "concrete_floor_worn_001", texture_names=worn_concrete_names, tint=(0.34, 0.39, 0.46), tint_strength=0.62, roughness=0.80, uv_scale=2.0, normal_strength=0.52, detail_bump_strength=0.020, detail_bump_scale=18.0),
        "picture_surface": pbr_material("PBR_Picture_Surface", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_names, tint=(0.22, 0.42, 0.48), tint_strength=0.72, roughness=0.84, uv_scale=1.8, normal_strength=0.44, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "red_rubber": pbr_material("PBR_Red_Rubber", texture_dir=EXTRA_TEXTURE_ROOT / "rubber_tiles", texture_names=rubber_tile_names, tint=(4.0, 0.08, 0.03), tint_strength=0.92, roughness=0.82, uv_scale=2.5, normal_strength=0.65, detail_bump_strength=0.010, detail_bump_scale=22.0),
        "blue_rubber": pbr_material("PBR_Blue_Rubber", texture_dir=EXTRA_TEXTURE_ROOT / "rubber_tiles", texture_names=rubber_tile_names, tint=(0.08, 0.30, 4.0), tint_strength=0.92, roughness=0.80, uv_scale=2.5, normal_strength=0.65, detail_bump_strength=0.010, detail_bump_scale=22.0),
        "yellow_rubber": pbr_material("PBR_Yellow_Rubber", texture_dir=EXTRA_TEXTURE_ROOT / "rubber_tiles", texture_names=rubber_tile_names, tint=(4.0, 2.4, 0.05), tint_strength=0.92, roughness=0.82, uv_scale=2.5, normal_strength=0.65, detail_bump_strength=0.010, detail_bump_scale=22.0),
        "domino_wood": pbr_material("PBR_Domino_Wood", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(0.52, 0.20, 0.08), tint_strength=0.78, roughness=0.58, uv_scale=3.0, normal_strength=0.66, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "blue_painted": pbr_material("PBR_Blue_Painted", texture_dir=EXTRA_TEXTURE_ROOT / "concrete_floor_worn_001", texture_names=worn_concrete_names, tint=(0.05, 0.28, 1.55), tint_strength=0.86, roughness=0.52, metallic=0.14, uv_scale=2.0, normal_strength=0.54, detail_bump_strength=0.020, detail_bump_scale=16.0),
        "teal_metal": pbr_material("PBR_Teal_Metal", texture_dir=EXTRA_TEXTURE_ROOT / "concrete_floor_worn_001", texture_names=worn_concrete_names, tint=(0.04, 0.72, 0.78), tint_strength=0.78, roughness=0.50, metallic=0.20, uv_scale=2.2, normal_strength=0.50, detail_bump_strength=0.020, detail_bump_scale=16.0),
        "barrier_metal": pbr_material("PBR_Barrier_Metal_Plate", texture_dir=EXTRA_TEXTURE_ROOT / "metal_plate", texture_names=metal_plate_names, tint=(0.08, 0.52, 0.58), tint_strength=0.74, roughness=0.48, metallic=0.34, uv_scale=1.35, normal_strength=0.62, detail_bump_strength=0.012, detail_bump_scale=16.0),
        "yellow_metal": pbr_material("PBR_Yellow_Metal", texture_dir=EXTRA_TEXTURE_ROOT / "metal_plate", texture_names=metal_plate_names, tint=(1.85, 0.72, 0.05), tint_strength=0.84, roughness=0.50, metallic=0.25, uv_scale=1.4, normal_strength=0.58, detail_bump_strength=0.012, detail_bump_scale=16.0),
        "dark_metal": pbr_material("PBR_Dark_Metal", texture_dir=EXTRA_TEXTURE_ROOT / "metal_plate", texture_names=metal_plate_names, tint=(0.08, 0.10, 0.13), tint_strength=0.86, roughness=0.42, metallic=0.64, uv_scale=1.25, normal_strength=0.58, detail_bump_strength=0.014, detail_bump_scale=16.0),
        "white_painted": pbr_material("PBR_White_Painted", texture_dir=EXTRA_TEXTURE_ROOT / "concrete_floor_worn_001", texture_names=worn_concrete_names, tint=(0.78, 0.84, 0.90), tint_strength=0.68, roughness=0.56, metallic=0.04, uv_scale=2.0, normal_strength=0.48, detail_bump_strength=0.012, detail_bump_scale=16.0),
        "green_painted": pbr_material("PBR_Green_Painted", texture_dir=EXTRA_TEXTURE_ROOT / "concrete_floor_worn_001", texture_names=worn_concrete_names, tint=(0.12, 0.78, 0.34), tint_strength=0.80, roughness=0.56, metallic=0.08, uv_scale=2.0, normal_strength=0.50, detail_bump_strength=0.014, detail_bump_scale=16.0),
        "coral_painted": pbr_material("PBR_Coral_Painted", texture_dir=EXTRA_TEXTURE_ROOT / "concrete_floor_worn_001", texture_names=worn_concrete_names, tint=(1.45, 0.30, 0.12), tint_strength=0.80, roughness=0.58, metallic=0.05, uv_scale=2.0, normal_strength=0.50, detail_bump_strength=0.014, detail_bump_scale=16.0),
        "window_glass": procedural_material("Window_Glass", (0.055, 0.13, 0.19), metallic=0.22, roughness=0.18, noise_scale=2.0),
        "fabric": pbr_material("PBR_Fabric", texture_dir=EXTRA_TEXTURE_ROOT / "denim_fabric_04", texture_names=denim_names, tint=(0.52, 0.62, 0.70), tint_strength=0.48, roughness=0.94, metallic=0.0, uv_scale=5.0, normal_strength=0.60, detail_bump_strength=0.012, detail_bump_scale=28.0),
        "fabric_green": pbr_material("PBR_Green_Fabric", texture_dir=EXTRA_TEXTURE_ROOT / "denim_fabric_04", texture_names=denim_names, tint=(0.16, 0.72, 0.30), tint_strength=0.78, roughness=0.94, metallic=0.0, uv_scale=5.0, normal_strength=0.60, detail_bump_strength=0.012, detail_bump_scale=28.0),
        "fabric_coral": pbr_material("PBR_Coral_Fabric", texture_dir=EXTRA_TEXTURE_ROOT / "denim_fabric_04", texture_names=denim_names, tint=(1.38, 0.28, 0.12), tint_strength=0.78, roughness=0.94, metallic=0.0, uv_scale=5.0, normal_strength=0.60, detail_bump_strength=0.012, detail_bump_scale=28.0),
        "rope_fabric": pbr_material("PBR_Rope_Fabric", texture_dir=TEXTURE_ROOT / "fabric_pattern_07", texture_names=fabric_names, tint=(0.52, 0.25, 0.08), roughness=0.92, metallic=0.0, uv_scale=18.0, normal_strength=0.72, detail_bump_strength=0.022, detail_bump_scale=34.0),
    }
    # Natural-material variants for refine experiments. They use the original
    # image-backed maps without artificial high-saturation tinting, so a ball,
    # wood block, or rubber puck keeps a familiar everyday appearance.
    materials["natural_oak_wood"] = pbr_material(
        "PBR_Natural_Oak_Wood",
        texture_dir=REALISM_TEXTURE_ROOT / "oak_wood_planks",
        texture_names=natural_oak_names,
        roughness=0.56,
        uv_scale=1.20,
        texture_coordinate="UV",
        normal_strength=0.42,
        detail_bump_strength=0.008,
        detail_bump_scale=18.0,
    )
    materials["natural_dark_wood"] = pbr_material(
        "PBR_Natural_Dark_Wood",
        texture_dir=REALISM_TEXTURE_ROOT / "dark_wood",
        texture_names=natural_dark_wood_names,
        roughness=0.58,
        uv_scale=1.35,
        texture_coordinate="UV",
        normal_strength=0.44,
        detail_bump_strength=0.008,
        detail_bump_scale=18.0,
    )
    materials["natural_black_rubber"] = pbr_material(
        "PBR_Natural_Black_Rubber",
        texture_dir=REALISM_TEXTURE_ROOT / "rubberized_track",
        # Do not use the reddish albedo of the running-track asset for the
        # puck. Keep only its micro normal/roughness maps over a neutral dark
        # polymer base, which is the usual appearance of a hockey/air-hockey
        # puck and avoids a misleading brick-red disk.
        texture_names={
            "normal": natural_rubber_names["normal"],
            "roughness": natural_rubber_names["roughness"],
        },
        tint=(0.035, 0.045, 0.055),
        tint_strength=0.0,
        roughness=0.88,
        uv_scale=3.5,
        normal_strength=0.24,
        detail_bump_strength=0.006,
        detail_bump_scale=30.0,
    )
    # Refine-only variants use the clearly visible wood-floor albedo and a
    # COLOR blend for hue.  They are separate keys so the original benchmark
    # palette above remains unchanged.
    refine_palette = {
        "refine_blue_texture": (0.10, 0.32, 1.00),
        "refine_green_texture": (0.10, 0.84, 0.28),
        "refine_teal_texture": (0.05, 0.76, 0.76),
        "refine_yellow_texture": (1.00, 0.72, 0.08),
        "refine_coral_texture": (1.00, 0.20, 0.10),
        "refine_red_texture": (0.95, 0.07, 0.04),
        "refine_charcoal_texture": (0.06, 0.09, 0.14),
        "refine_purple_texture": (0.48, 0.12, 0.90),
        "refine_orange_texture": (1.00, 0.34, 0.05),
    }
    for key, color in refine_palette.items():
        materials[key] = colorized_texture_material(
            f"PBR_{key.title().replace('_', '')}",
            texture_dir=RAMP_BLOCK_TEXTURE_ROOT,
            texture_names=peeling_paint_wood_names,
            color=color,
            roughness=0.62,
            uv_scale=1.0,
            normal_strength=0.58,
        )
    if basketball_texture is not None:
        if not basketball_texture.is_file():
            raise FileNotFoundError(f"basketball texture not found: {basketball_texture}")
        materials["basketball"] = basketball_material("PBR_Basketball", basketball_texture)
        materials["natural_basketball"] = basketball_material(
            "PBR_Natural_Basketball", basketball_texture
        )
    return materials


def add_cube(name: str, location, half_extents, material, *, bevel: float = 0.025):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half_extents
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel > 0.0:
        modifier = obj.modifiers.new("Edge bevel", "BEVEL")
        modifier.width = min(bevel, min(half_extents) * 0.35)
        modifier.segments = 3
        modifier.affect = "EDGES"
    return obj


def add_edge_clarity_geometry(obj: bpy.types.Object, shape: str, size: dict) -> None:
    """Sharpen hard-object silhouettes without changing collision geometry."""
    if shape == "box":
        half_extents = [float(size[key]) for key in ("hx", "hy", "hz")]
        width = min(min(half_extents) * 0.22, 0.036)
    elif shape in {"cylinder", "puck"}:
        width = min(float(size["radius"]) * 0.16, 0.030)
    else:
        return
    if width <= 0.0:
        return

    bevel = next((modifier for modifier in obj.modifiers if modifier.type == "BEVEL"), None)
    if bevel is None:
        bevel = obj.modifiers.new("Edge clarity bevel", "BEVEL")
        bevel.limit_method = "ANGLE"
        try:
            bevel.affect = "EDGES"
        except AttributeError:
            pass
    bevel.width = max(float(bevel.width), width)
    bevel.segments = max(int(bevel.segments), 3)
    try:
        bevel.harden_normals = True
    except AttributeError:
        pass

    try:
        obj.data.use_auto_smooth = True
    except AttributeError:
        pass
    weighted = obj.modifiers.new("Edge clarity weighted normals", "WEIGHTED_NORMAL")
    weighted.keep_sharp = True
    weighted.weight = 50


def edge_clarity_material(material: bpy.types.Material, actor_name: str) -> bpy.types.Material:
    """Make a per-actor copy with a restrained grazing-angle edge highlight."""
    enhanced = material.copy()
    enhanced.name = f"{material.name}__edge_clarity__{actor_name}"
    if not enhanced.use_nodes:
        return enhanced
    nodes = enhanced.node_tree.nodes
    links = enhanced.node_tree.links
    shader = nodes.get("Principled BSDF")
    if shader is None:
        return enhanced
    base_input = shader.inputs.get("Base Color")
    if base_input is None:
        return enhanced
    source_socket = base_input.links[0].from_socket if base_input.links else None
    default_color = tuple(base_input.default_value)
    for link in list(base_input.links):
        links.remove(link)

    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.45
    fresnel.label = "Restrained edge highlight"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (0.30, 0.30, 0.30, 1.0)
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "SCREEN"
    mix.inputs[2].default_value = (0.82, 0.82, 0.82, 1.0)
    mix.label = "Soft silhouette highlight"
    if source_socket is not None:
        links.new(source_socket, mix.inputs[1])
    else:
        mix.inputs[1].default_value = default_color
    links.new(fresnel.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], base_input)
    return enhanced


def bowl_curve_geometry(size: dict) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Build the same continuous bowl shell used by the PyBullet renderer."""
    radius = float(size["radius"])
    span = float(size["span"])
    bottom_z = float(size["bottom_z"])
    thickness = float(size["thickness"])
    half_y = float(size["half_y"])
    segments = max(8, int(round(float(size.get("segments", 96.0)))))
    vertices: list[tuple[float, float, float]] = []
    for index in range(segments + 1):
        x = -span + 2.0 * span * index / segments
        root = math.sqrt(max(radius * radius - x * x, 1e-10))
        surface_z = bottom_z + radius - root
        nx, nz = -x / radius, root / radius
        ox, oz = x - nx * thickness, surface_z - nz * thickness
        vertices.extend(
            [
                (x, -half_y, surface_z),
                (x, half_y, surface_z),
                (ox, -half_y, oz),
                (ox, half_y, oz),
            ]
        )
    faces: list[tuple[int, int, int]] = []
    for index in range(segments):
        current = 4 * index
        following = 4 * (index + 1)
        faces.extend(
            [
                (current, following, following + 1),
                (current, following + 1, current + 1),
                (current + 2, following + 3, following + 2),
                (current + 2, current + 3, following + 3),
                (current, current + 2, following + 2),
                (current, following + 2, following),
                (current + 1, following + 1, following + 3),
                (current + 1, following + 3, current + 3),
            ]
        )
    left = 0
    right = 4 * segments
    faces.extend(
        [
            (left, left + 1, left + 3),
            (left, left + 3, left + 2),
            (right, right + 3, right + 1),
            (right, right + 2, right + 3),
        ]
    )
    return vertices, faces


def add_actor(name: str, actor: dict, material, *, edge_clarity: bool = False) -> bpy.types.Object:
    shape = actor["shape"]
    size = actor["size_m"]
    position = actor["initial_position_m"]
    if shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=float(size["radius"]), location=position)
        obj = bpy.context.object
        bpy.ops.object.shade_smooth()
        bevel = obj.modifiers.new("Micro subdivision", "SUBSURF")
        bevel.levels = 1
        bevel.render_levels = 1
    elif shape == "box":
        obj = add_cube(name, position, (float(size["hx"]), float(size["hy"]), float(size["hz"])), material, bevel=0.018)
        if edge_clarity:
            add_edge_clarity_geometry(obj, shape, size)
        return obj
    elif shape in {"cylinder", "puck"}:
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=float(size["radius"]), depth=float(size["height"]), location=position)
        obj = bpy.context.object
        bpy.ops.object.shade_smooth()
        bevel = obj.modifiers.new("Cylinder bevel", "BEVEL")
        bevel.width = min(float(size["radius"]) * 0.08, 0.018)
        bevel.segments = 3
    elif shape == "bowl_curve":
        vertices, faces = bowl_curve_geometry(size)
        mesh = bpy.data.meshes.new(f"{name}_mesh")
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = position
        for polygon in mesh.polygons:
            polygon.use_smooth = True
    else:
        raise ValueError(f"unsupported shape {shape!r} for {name}")
    obj.name = name
    obj.data.materials.append(material)
    if edge_clarity:
        add_edge_clarity_geometry(obj, shape, size)
    return obj


def actor_material_key(name: str, actor: dict, family: str) -> str:
    lower = name.lower()

    # Dynamic actors use a family-level appearance contract. Every control
    # case in one group therefore keeps identical color and texture mapping.
    if family == "F11" and lower == "roller_0":
        return "red_rubber"
    if family == "F11" and lower == "table_top_0":
        return "red_wood"
    if family == "F11" and lower.startswith("table_leg"):
        return "dark_metal"
    if family == "F12" and lower == "block_0":
        return "red_wood"
    if family == "V2V_RAMP_PLATFORM":
        if lower == "block_0":
            return "wood_peeling_paint"
        if lower == "incline_board_0":
            return "wood"
        if lower.startswith("ramp_support"):
            return "concrete"
        if lower in {"table_top_0", "horizontal_platform_0"}:
            return "wood"
        if lower.startswith("table_leg"):
            return "dark_metal"
    if family == "V2V_BOWL" and lower == "bowl_ball":
        return "blue_rubber"
    if family == "V2V_BOWL" and (lower == "bowl_base" or lower == "bowl_surface"):
        return "teal_metal"
    if family == "V2V_DOMINO":
        if lower == "domino_trigger_ball":
            return "yellow_rubber"
        if lower.startswith("domino_"):
            return "domino_wood"
    if family == "V2V_GAP" and lower == "gap_ball":
        return "yellow_rubber"
    if family in {"V2V_OBSTACLE", "V2V_OBSTACLE_SIZE"} and lower == "obstacle_ball":
        return "red_rubber"
    if family == "V2V_OBSTACLE" and lower == "obstacle_barrier":
        return "blue_painted"
    if family == "V2V_OBSTACLE_SIZE" and lower == "obstacle_barrier":
        return "barrier_metal"
    if family == "V2V_PENDULUM":
        if lower == "pendulum_bob":
            return "blue_rubber"
        if lower == "pendulum_rope":
            return "yellow_metal"
        if lower == "pendulum_post":
            return "yellow_metal"
        if lower == "pendulum_base":
            return "concrete"
        if lower == "pendulum_crossbar":
            return "dark_metal"
    if family == "V2V_PENDULUM_CABINET":
        if lower == "pendulum_bob":
            return "red_rubber"
        if lower in {"pendulum_rope", "pendulum_post"}:
            return "yellow_metal"
        if lower == "pendulum_base":
            return "concrete"
        if lower == "pendulum_crossbar":
            return "dark_metal"
        if lower == "pendulum_cabinet_body":
            return "domino_wood"
        if lower == "pendulum_cabinet_door":
            return "wood"
        if lower == "pendulum_cabinet_handle":
            return "dark_metal"
    if family == "V2V_SEESAW" and lower == "seesaw_load":
        return "yellow_rubber"
    if family == "V2V_SEESAW" and lower == "seesaw_board":
        return "blue_painted"
    if family == "V2V_SEESAW" and lower == "seesaw_hinge_anchor":
        return "dark_metal"
    if family == "SCENE_PUCK_BARRIER":
        if lower == "puck":
            return "dark_metal"
        if lower == "puck_barrier":
            return "blue_painted"
    if family in {"SCENE_DOOR_FRAME", "SCENE_DOOR_FRAME_BALL"}:
        if lower == "door_ball":
            return "blue_rubber"
        if lower == "door_crate":
            return "red_wood"
        if lower.startswith("door_wall"):
            return "wall_gray"
        if lower.startswith("door_frame"):
            return "wood"

    if "barrier" in lower or ("post" in lower and "pendulum" in lower):
        return "teal_metal"
    if family == "F12":
        return "red_wood" if lower == "block_0" else "yellow_metal"
    if "bowl_ball" in lower or "seesaw_load" in lower:
        return "blue_rubber"
    if actor["shape"] == "sphere" or "ball" in lower or "bob" in lower:
        return "red_rubber"
    if "rope" in lower:
        return "yellow_metal"
    if "pivot" in lower or "riser" in lower or "support" in lower:
        return "concrete"
    if "domino" in lower and lower != "domino_0":
        return "teal_metal"
    return "wood"


ROOM_SCENES = {
    "F11": {
        "name": "sage_living_room",
        "layout": "living",
        "floor": "floor_concrete",
        "wall": "wall_green",
        "trim": "wood",
    },
    "F12": {
        "name": "cool_workshop",
        "layout": "workshop",
        "floor": "floor_dark_wood",
        "wall": "wall_gray",
        "trim": "dark_metal",
    },
    "V2V_RAMP_PLATFORM": {
        "name": "cool_workshop",
        "layout": "workshop",
        "floor": "floor_stone_pavement",
        "wall": "wall_gray",
        "trim": "dark_metal",
    },
    "V2V_BOWL": {
        "name": "warm_art_gallery",
        "layout": "gallery",
        "floor": "floor",
        "wall": "wall_rose",
        "trim": "white_painted",
    },
    "V2V_DOMINO": {
        "name": "blue_library",
        "layout": "library",
        "floor": "floor_concrete",
        "wall": "wall_cool",
        "trim": "wood",
    },
    "V2V_GAP": {
        "name": "industrial_loft",
        "layout": "loft",
        "floor": "floor_dark_wood",
        "wall": "wall_charcoal",
        "trim": "concrete",
    },
    "V2V_OBSTACLE": {
        "name": "daylight_home_office",
        "layout": "office",
        "floor": "floor_cool",
        "wall": "wall_green",
        "trim": "dark_metal",
    },
    "V2V_OBSTACLE_SIZE": {
        "name": "terracotta_gallery",
        "layout": "modern_gallery",
        "floor": "floor_concrete",
        "wall": "wall_rose",
        "trim": "white_painted",
    },
    "V2V_PENDULUM": {
        "name": "quiet_motion_lab",
        "layout": "lab",
        "floor": "floor_dark_wood",
        "wall": "wall_gray",
        "trim": "dark_metal",
    },
    "V2V_PENDULUM_CABINET": {
        "name": "library_cabinet_room",
        "layout": "library",
        "floor": "floor_concrete",
        "wall": "wall_cool",
        "trim": "wood",
    },
    "V2V_SEESAW": {
        "name": "coral_activity_room",
        "layout": "activity",
        "floor": "floor_terracotta",
        "wall": "wall_cool",
        "trim": "wood",
    },
    "SCENE_PUCK_BARRIER": {
        "name": "puck_workshop",
        "layout": "workshop",
        "floor": "floor_slate",
        "wall": "wall_gray",
        "trim": "white_painted",
    },
    "SCENE_DOOR_FRAME": {
        "name": "doorway_loft",
        "layout": "loft",
        "floor": "floor_concrete",
        "wall": "wall_rose",
        "trim": "dark_metal",
    },
}


def add_prop_cylinder(name: str, location, radius: float, depth: float, material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("Prop bevel", "BEVEL")
    bevel.width = min(radius * 0.10, 0.018)
    bevel.segments = 3
    return obj


def add_prop_sphere(name: str, location, radius: float, material, scale=(1.0, 1.0, 1.0)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return obj


def add_wall_art(materials, name: str, x: float, z: float, width: float, height: float, art_key: str) -> None:
    add_cube(f"{name} frame", (x, 3.445, z), (width * 0.5, 0.030, height * 0.5), materials["dark_metal"], bevel=0.018)
    add_cube(f"{name} inset", (x, 3.405, z), (width * 0.43, 0.012, height * 0.40), materials[art_key], bevel=0.008)


def add_window(materials, name: str, x: float, z: float, width: float, height: float) -> None:
    add_cube(f"{name} glass", (x, 3.455, z), (width * 0.5, 0.018, height * 0.5), materials["window_glass"], bevel=0.012)
    frame = materials["white_painted"]
    border = 0.055
    add_cube(f"{name} top", (x, 3.415, z + height * 0.5), (width * 0.54, 0.028, border), frame, bevel=0.008)
    add_cube(f"{name} bottom", (x, 3.415, z - height * 0.5), (width * 0.54, 0.028, border), frame, bevel=0.008)
    for offset in (-width * 0.5, 0.0, width * 0.5):
        add_cube(f"{name} mullion {offset}", (x + offset, 3.410, z), (border, 0.030, height * 0.54), frame, bevel=0.006)


def add_cabinet(materials, name: str, x: float, width: float, body_key: str, door_key: str) -> None:
    add_cube(name, (x, 3.05, 0.52), (width * 0.5, 0.34, 0.52), materials[body_key], bevel=0.035)
    add_cube(f"{name} top", (x, 3.02, 1.07), (width * 0.54, 0.38, 0.035), materials["dark_metal"], bevel=0.012)
    door_count = max(2, int(round(width / 0.55)))
    spacing = width / door_count
    for index in range(door_count):
        door_x = x - width * 0.5 + spacing * (index + 0.5)
        add_cube(f"{name} door {index}", (door_x, 2.69, 0.52), (spacing * 0.42, 0.018, 0.42), materials[door_key], bevel=0.010)


def add_shelf(materials, name: str, x: float, width: float, height: float, shelf_key: str) -> None:
    y = 3.17
    add_cube(f"{name} left", (x - width * 0.5, y, height * 0.5), (0.045, 0.27, height * 0.5), materials[shelf_key], bevel=0.008)
    add_cube(f"{name} right", (x + width * 0.5, y, height * 0.5), (0.045, 0.27, height * 0.5), materials[shelf_key], bevel=0.008)
    shelf_zs = (0.08, height * 0.34, height * 0.66, height)
    for index, shelf_z in enumerate(shelf_zs):
        add_cube(f"{name} shelf {index}", (x, y, shelf_z), (width * 0.5, 0.29, 0.035), materials[shelf_key], bevel=0.008)
    book_materials = ("coral_painted", "green_painted", "white_painted", "wood")
    for index in range(8):
        shelf_level = 0 if index < 4 else 1
        book_x = x - width * 0.34 + (index % 4) * width * 0.21
        book_z = shelf_zs[shelf_level + 1] + 0.16
        add_cube(f"{name} book {index}", (book_x, y - 0.30, book_z), (0.045, 0.055, 0.13), materials[book_materials[index % len(book_materials)]], bevel=0.004)


def add_bench(materials, name: str, x: float, width: float, fabric_key: str) -> None:
    add_cube(f"{name} seat", (x, 2.92, 0.38), (width * 0.5, 0.38, 0.10), materials[fabric_key], bevel=0.045)
    add_cube(f"{name} back", (x, 3.23, 0.75), (width * 0.5, 0.10, 0.30), materials[fabric_key], bevel=0.045)
    for leg_x in (x - width * 0.38, x + width * 0.38):
        add_cube(f"{name} leg {leg_x}", (leg_x, 2.92, 0.18), (0.045, 0.28, 0.18), materials["dark_metal"], bevel=0.008)


def add_desk(materials, name: str, x: float, width: float) -> None:
    add_cube(f"{name} top", (x, 3.00, 0.78), (width * 0.5, 0.38, 0.055), materials["wood"], bevel=0.018)
    for leg_x in (x - width * 0.42, x + width * 0.42):
        add_cube(f"{name} leg {leg_x}", (leg_x, 3.0, 0.38), (0.045, 0.30, 0.38), materials["dark_metal"], bevel=0.008)
    add_cube(f"{name} monitor", (x, 3.19, 1.20), (0.36, 0.055, 0.24), materials["window_glass"], bevel=0.018)
    add_cube(f"{name} monitor stand", (x, 3.18, 0.94), (0.045, 0.06, 0.14), materials["dark_metal"], bevel=0.006)


def add_plant(materials, name: str, x: float) -> None:
    add_prop_cylinder(f"{name} pot", (x, 3.00, 0.26), 0.22, 0.52, materials["coral_painted"])
    add_prop_cylinder(f"{name} stem", (x, 3.00, 0.75), 0.035, 0.68, materials["green_painted"])
    for index, offset in enumerate(((-0.17, 0.00, 0.91), (0.12, -0.03, 1.03), (0.02, 0.03, 1.22))):
        add_prop_sphere(
            f"{name} leaf {index}",
            (x + offset[0], 3.00 + offset[1], offset[2]),
            0.24,
            materials["green_painted"],
            scale=(1.10, 0.58, 1.35),
        )


def add_rug(materials, name: str, x: float, y: float, hx: float, hy: float, material_key: str) -> None:
    add_cube(name, (x, y, 0.012), (hx, hy, 0.012), materials[material_key], bevel=0.015)


def add_room_layout(materials: dict[str, bpy.types.Material], layout: str) -> None:
    # All props stay behind the simulated motion corridor near y=0.
    if layout == "living":
        add_cabinet(materials, "Living credenza", -2.45, 1.65, "wood", "white_painted")
        add_window(materials, "Living window", 2.30, 1.90, 1.55, 1.35)
        add_plant(materials, "Living plant", 0.95)
        add_rug(materials, "Living rug", 2.45, 1.78, 1.05, 0.72, "fabric_coral")
    elif layout == "workshop":
        add_desk(materials, "Workshop bench", 2.85, 1.70)
        add_wall_art(materials, "Workshop board", 2.85, 1.90, 1.55, 0.82, "green_painted")
        add_shelf(materials, "Workshop rack", -1.55, 1.25, 1.70, "dark_metal")
        add_cube("Workshop wall rail", (0.45, 3.40, 1.75), (0.75, 0.035, 0.055), materials["coral_painted"], bevel=0.010)
    elif layout == "gallery":
        add_wall_art(materials, "Gallery left", -2.10, 1.95, 0.95, 1.18, "green_painted")
        add_wall_art(materials, "Gallery center", 0.0, 2.05, 1.15, 0.82, "picture_surface")
        add_wall_art(materials, "Gallery right", 2.15, 1.88, 0.82, 1.30, "coral_painted")
        add_cube("Gallery left plinth", (-2.85, 3.03, 0.44), (0.33, 0.30, 0.44), materials["white_painted"], bevel=0.018)
        add_cube("Gallery right plinth", (2.85, 3.03, 0.30), (0.42, 0.34, 0.30), materials["concrete"], bevel=0.018)
    elif layout == "library":
        add_shelf(materials, "Library shelves", -2.55, 1.70, 2.05, "wood")
        add_bench(materials, "Reading bench", 2.35, 1.55, "fabric_green")
        add_wall_art(materials, "Library print", 0.60, 2.02, 0.95, 0.72, "coral_painted")
        add_rug(materials, "Library rug", 2.25, 1.78, 1.10, 0.68, "fabric_green")
    elif layout == "loft":
        for x in (-2.75, 2.75):
            add_cube(f"Loft pillar {x}", (x, 3.33, 1.55), (0.22, 0.20, 1.55), materials["concrete"], bevel=0.012)
        for index, z in enumerate((0.72, 1.32, 2.15)):
            add_cube(f"Loft pipe {index}", (1.55, 3.39, z), (1.10, 0.055, 0.055), materials["white_painted"], bevel=0.030)
        add_cube("Loft crate lower", (-2.05, 3.02, 0.34), (0.48, 0.38, 0.34), materials["wood"], bevel=0.025)
        add_cube("Loft crate upper", (-2.18, 3.00, 0.91), (0.34, 0.31, 0.23), materials["wood"], bevel=0.025)
    elif layout == "office":
        add_desk(materials, "Office desk", -2.35, 1.70)
        add_window(materials, "Office window", 2.30, 1.90, 1.60, 1.25)
        add_plant(materials, "Office plant", 0.95)
        add_rug(materials, "Office rug", 2.35, 1.78, 1.05, 0.70, "fabric")
    elif layout == "modern_gallery":
        add_wall_art(materials, "Modern art left", -2.20, 1.95, 0.78, 1.28, "green_painted")
        add_wall_art(materials, "Modern art center", 0.0, 2.02, 1.05, 0.74, "picture_surface")
        add_wall_art(materials, "Modern art right", 2.15, 1.88, 0.92, 1.12, "white_painted")
        add_cube("Modern tall plinth", (-2.75, 3.02, 0.62), (0.28, 0.28, 0.62), materials["dark_metal"], bevel=0.014)
        add_cube("Modern low plinth", (2.75, 3.02, 0.27), (0.48, 0.34, 0.27), materials["green_painted"], bevel=0.014)
    elif layout == "lab":
        add_cabinet(materials, "Lab cabinet", 2.45, 1.65, "white_painted", "dark_metal")
        add_desk(materials, "Lab console", 0.85, 1.20)
        add_wall_art(materials, "Lab status board", 2.45, 1.92, 1.40, 0.78, "green_painted")
        for index, x in enumerate((-2.70, -2.25)):
            add_cube(f"Lab acoustic panel {index}", (x, 3.42, 1.70), (0.16, 0.035, 0.72), materials["fabric_coral"], bevel=0.016)
    elif layout == "activity":
        add_shelf(materials, "Activity cubbies", -2.55, 1.65, 1.55, "wood")
        add_bench(materials, "Activity bench", 2.40, 1.55, "fabric_green")
        add_wall_art(materials, "Activity wall panel left", -0.75, 2.10, 0.82, 0.60, "coral_painted")
        add_wall_art(materials, "Activity wall panel right", 0.75, 2.10, 0.82, 0.60, "green_painted")
        add_rug(materials, "Activity rug", 2.35, 1.70, 1.10, 0.72, "fabric_coral")
    else:
        raise ValueError(f"unknown room layout {layout!r}")


def add_room(materials: dict[str, bpy.types.Material], family: str) -> dict[str, str]:
    config = ROOM_SCENES.get(family, ROOM_SCENES["F11"])
    floor_key = config["floor"]
    wall_key = config["wall"]
    trim_key = config["trim"]
    add_cube("Room floor", (0.0, 1.0, -0.055), (8.0, 7.0, 0.055), materials[floor_key], bevel=0.0)
    add_cube("Back wall", (0.0, 3.55, 3.0), (8.0, 0.06, 3.0), materials[wall_key], bevel=0.0)
    add_cube("Left wall", (-7.95, 0.8, 3.0), (0.06, 2.8, 3.0), materials[wall_key], bevel=0.0)
    add_cube("Right wall", (7.95, 0.8, 3.0), (0.06, 2.8, 3.0), materials[wall_key], bevel=0.0)
    add_cube("Back skirting", (0.0, 3.45, 0.085), (7.95, 0.05, 0.085), materials[trim_key], bevel=0.012)
    add_room_layout(materials, config["layout"])
    return dict(config)


def set_world_hdri(scene: bpy.types.Scene, family: str) -> Path:
    hdri_by_family = {
        "F11": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "F12": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
        "V2V_RAMP_PLATFORM": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
        "V2V_BOWL": HDRI_ROOT / "brown_photostudio_02" / "brown_photostudio_02_4k.hdr",
        "V2V_DOMINO": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "V2V_GAP": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
        "V2V_OBSTACLE": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "V2V_OBSTACLE_SIZE": HDRI_ROOT / "brown_photostudio_02" / "brown_photostudio_02_4k.hdr",
        "V2V_PENDULUM": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
        "V2V_PENDULUM_CABINET": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "V2V_SEESAW": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "SCENE_PUCK_BARRIER": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "SCENE_DOOR_FRAME": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
    }
    rotation_by_family = {
        "F11": 22.0,
        "F12": -18.0,
        "V2V_RAMP_PLATFORM": -18.0,
        "V2V_BOWL": 22.0,
        "V2V_DOMINO": 58.0,
        "V2V_GAP": 35.0,
        "V2V_OBSTACLE": -24.0,
        "V2V_OBSTACLE_SIZE": -24.0,
        "V2V_PENDULUM": 88.0,
        "V2V_PENDULUM_CABINET": 52.0,
        "V2V_SEESAW": -42.0,
        "SCENE_PUCK_BARRIER": -12.0,
        "SCENE_DOOR_FRAME": 18.0,
    }
    path = hdri_by_family.get(family, HDRI_ROOT / "brown_photostudio_02" / "brown_photostudio_02_4k.hdr")
    world = bpy.data.worlds.new("PBR World") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    strength_by_family = {
        "F11": 0.26,
        "F12": 0.24,
        "V2V_RAMP_PLATFORM": 0.24,
        "V2V_BOWL": 0.30,
        "V2V_DOMINO": 0.28,
        "V2V_GAP": 0.21,
        "V2V_OBSTACLE": 0.27,
        "V2V_OBSTACLE_SIZE": 0.31,
        "V2V_PENDULUM": 0.24,
        "V2V_PENDULUM_CABINET": 0.28,
        "V2V_SEESAW": 0.28,
        "SCENE_PUCK_BARRIER": 0.25,
        "SCENE_DOOR_FRAME": 0.27,
    }
    background.inputs["Strength"].default_value = strength_by_family.get(family, 0.28)
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(path), check_existing=True)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_by_family.get(family, 22.0))
    texcoord = nodes.new("ShaderNodeTexCoord")
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])
    return path


def point_at(obj: bpy.types.Object, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def add_area_light(name: str, location, target, energy: float, size: float, color) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    point_at(obj, target)


def add_lighting(family: str) -> str:
    presets = {
        "F11": ("warm_window", (-2.6, -2.2, 3.5), 650.0, (1.0, 0.84, 0.70), (3.0, -1.0, 2.8), 350.0, (0.76, 0.88, 1.0), 250.0),
        "F12": ("cool_workshop", (3.0, -2.4, 3.7), 610.0, (0.86, 0.93, 1.0), (-2.8, -1.2, 2.8), 390.0, (1.0, 0.82, 0.68), 290.0),
        "V2V_RAMP_PLATFORM": ("cool_workshop", (3.0, -2.4, 3.7), 610.0, (0.86, 0.93, 1.0), (-2.8, -1.2, 2.8), 390.0, (1.0, 0.82, 0.68), 290.0),
        "V2V_BOWL": ("gallery_soft", (-1.8, -2.3, 3.8), 700.0, (1.0, 0.91, 0.82), (3.1, -1.0, 2.9), 340.0, (0.80, 0.90, 1.0), 320.0),
        "V2V_DOMINO": ("library_warm", (-3.0, -2.0, 3.4), 620.0, (1.0, 0.80, 0.64), (2.8, -1.2, 2.7), 410.0, (0.76, 0.88, 1.0), 270.0),
        "V2V_GAP": ("loft_directional", (3.2, -2.1, 3.7), 720.0, (0.78, 0.88, 1.0), (-2.6, -0.8, 2.5), 300.0, (1.0, 0.76, 0.60), 210.0),
        "V2V_OBSTACLE": ("office_daylight", (2.9, -2.4, 3.5), 680.0, (0.82, 0.92, 1.0), (-2.8, -1.1, 2.8), 360.0, (1.0, 0.86, 0.72), 260.0),
        "V2V_OBSTACLE_SIZE": ("gallery_warm", (-2.8, -2.2, 3.6), 690.0, (1.0, 0.82, 0.68), (3.0, -1.0, 2.8), 370.0, (0.78, 0.90, 1.0), 300.0),
        "V2V_PENDULUM": ("lab_neutral", (3.0, -2.0, 3.9), 650.0, (0.86, 0.94, 1.0), (-3.0, -1.0, 3.0), 390.0, (1.0, 0.84, 0.70), 310.0),
        "V2V_PENDULUM_CABINET": ("library_cabinet_warm", (-2.8, -2.2, 3.7), 690.0, (1.0, 0.84, 0.70), (2.9, -1.1, 2.9), 390.0, (0.78, 0.90, 1.0), 300.0),
        "V2V_SEESAW": ("activity_warm", (-2.7, -2.3, 3.5), 680.0, (1.0, 0.86, 0.72), (3.0, -1.0, 2.8), 380.0, (0.78, 0.90, 1.0), 280.0),
        "SCENE_PUCK_BARRIER": ("puck_daylight", (-2.8, -2.4, 3.7), 720.0, (0.78, 0.88, 1.0), (3.0, -1.0, 2.9), 360.0, (1.0, 0.82, 0.68), 300.0),
        "SCENE_DOOR_FRAME": ("doorway_soft", (-2.6, -2.5, 3.8), 700.0, (1.0, 0.84, 0.74), (3.1, -1.0, 2.9), 380.0, (0.76, 0.88, 1.0), 300.0),
    }
    preset = presets.get(family, presets["F11"])
    name, key_location, key_energy, key_color, fill_location, fill_energy, fill_color, top_energy = preset
    add_area_light("Key softbox", key_location, (0.0, 0.4, 0.5), key_energy, 3.0, key_color)
    add_area_light("Fill softbox", fill_location, (0.2, 0.5, 0.45), fill_energy, 2.6, fill_color)
    add_area_light("Top bounce", (0.0, 1.8, 4.4), (0.0, 0.6, 0.0), top_energy, 3.4, (1.0, 0.97, 0.90))
    return name


CAMERA_FRAMING_PRESETS = {
    # Each family uses one fixed camera fitted against all five full-length
    # trajectories, so framing cannot leak the controlled variable.
    "F11": {"target": (1.088, 0.0, 0.604), "yfov_deg": 42.0},
    "F12": {"target": (1.930, 0.0, 0.620), "yfov_deg": 38.5},
    "V2V_RAMP_PLATFORM": {"target": (0.45, 0.0, 0.86), "yfov_deg": 45.0},
    "V2V_BOWL": {"target": (0.005, 0.0, 0.600), "yfov_deg": 34.5},
    "V2V_DOMINO": {"target": (-0.147, 0.0, 0.480), "yfov_deg": 25.0},
    "V2V_GAP": {"target": (0.288, 0.0, 0.620), "yfov_deg": 35.0},
    "V2V_OBSTACLE": {"target": (-0.260, 0.0, 0.480), "yfov_deg": 24.5},
    "V2V_OBSTACLE_SIZE": {"target": (-0.635, 0.0, 0.480), "yfov_deg": 31.5},
    "V2V_PENDULUM": {"target": (-0.450, 0.0, 1.128), "yfov_deg": 41.5},
    "V2V_PENDULUM_CABINET": {"target": (-0.28, 0.0, 1.48), "yfov_deg": 50.0},
    "V2V_SEESAW": {"target": (0.002, 0.0, 0.460), "yfov_deg": 21.5},
    "SCENE_PUCK_BARRIER": {"target": (0.20, -0.40, 0.24), "yfov_deg": 45.0},
    "SCENE_DOOR_FRAME": {"target": (0.25, 0.0, 0.82), "yfov_deg": 46.0},
}


def add_camera(metadata: dict) -> bpy.types.Object:
    camera_spec = metadata["camera"]
    intrinsics = camera_spec["intrinsics"]
    extrinsics = camera_spec["extrinsics"]
    framing = CAMERA_FRAMING_PRESETS.get(metadata["family_key"], {})
    source_yfov_deg = float(intrinsics["yfov_deg"])
    effective_yfov_deg = float(framing.get("yfov_deg", source_yfov_deg))
    target = framing.get("target", extrinsics["target"])
    data = bpy.data.cameras.new("PhysV Camera")
    data.type = "PERSP"
    data.sensor_fit = "VERTICAL"
    data.angle_y = math.radians(effective_yfov_deg)
    obj = bpy.data.objects.new("PhysV Camera", data)
    bpy.context.collection.objects.link(obj)
    obj.location = extrinsics["eye"]
    point_at(obj, target)
    obj["framing_profile"] = "family_full_trajectory_fit_v1"
    obj["source_yfov_deg"] = source_yfov_deg
    obj["effective_yfov_deg"] = effective_yfov_deg
    obj["target"] = tuple(float(value) for value in target)
    bpy.context.scene.camera = obj
    return obj


def camera_diagnostics(scene: bpy.types.Scene, camera: bpy.types.Object, object_names: list[str]) -> dict:
    scene.frame_set(1)
    forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    projections = {}
    for name in object_names:
        obj = bpy.data.objects[name]
        projected = world_to_camera_view(scene, camera, obj.matrix_world.translation)
        projections[name] = [float(projected.x), float(projected.y), float(projected.z)]
    result = {
        "location": [float(value) for value in camera.location],
        "forward": [float(value) for value in forward],
        "framing_profile": camera["framing_profile"],
        "source_yfov_deg": float(camera["source_yfov_deg"]),
        "effective_yfov_deg": float(camera["effective_yfov_deg"]),
        "target": [float(value) for value in camera["target"]],
        "object_projections_xy_depth": projections,
    }
    print("CAMERA_DIAGNOSTICS", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def animate_objects(
    metadata: dict,
    trajectories,
    materials,
    frame_limit: int,
    material_overrides: dict[str, str] | None = None,
    edge_clarity: bool = False,
) -> tuple[list[str], int, dict[str, str]]:
    family = metadata["family_key"]
    names = trajectories["object_names"]
    available_frames = len(trajectories["frame_times_s"])
    frame_count = min(frame_limit, available_frames) if frame_limit > 0 else available_frames
    material_assignments = {}
    for name in names:
        actor = metadata["actors"][name]
        material_key = (material_overrides or {}).get(name, actor_material_key(name, actor, family))
        if material_key not in materials:
            raise KeyError(f"material override {material_key!r} for {name!r} is not in material library")
        material_assignments[name] = material_key
        material = materials[material_key]
        if edge_clarity and actor.get("shape") != "sphere":
            material = edge_clarity_material(material, name)
        obj = add_actor(name, actor, material, edge_clarity=edge_clarity)
        obj.rotation_mode = "QUATERNION"
        positions = trajectories[f"{name}_positions"]
        rotations = trajectories[f"{name}_rotations"]
        for frame_index, (position, rotation) in enumerate(zip(positions[:frame_count], rotations[:frame_count]), start=1):
            obj.location = tuple(float(value) for value in position)
            obj.rotation_quaternion = tuple(float(value) for value in rotation)
            obj.keyframe_insert(data_path="location", frame=frame_index)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)
        if obj.animation_data and obj.animation_data.action:
            for curve in obj.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
    return names, frame_count, material_assignments


def main() -> None:
    args = parse_args()
    args.sample_dir = args.sample_dir.resolve()
    args.trajectory_json = args.trajectory_json.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.sample_dir / "metadata.json").read_text(encoding="utf-8"))
    trajectories = json.loads(args.trajectory_json.read_text(encoding="utf-8"))
    material_overrides = {}
    if args.material_overrides_json is not None:
        material_overrides = json.loads(args.material_overrides_json.read_text(encoding="utf-8"))
        if not isinstance(material_overrides, dict):
            raise TypeError("--material-overrides-json must contain a JSON object")
        material_overrides = {str(key): str(value) for key, value in material_overrides.items()}
    fps = int(metadata["simulation"]["fps"])
    render_family = "F12" if metadata["family_key"] == "F12_RAMP_LENGTH" else metadata["family_key"]
    if render_family == "SCENE_DOOR_FRAME_BALL":
        render_family = "SCENE_DOOR_FRAME"
    render_metadata = dict(metadata)
    render_metadata["family_key"] = render_family

    clear_scene()
    scene = bpy.context.scene
    enabled_devices = configure_cycles(scene, args, fps)
    materials = material_library(args.basketball_texture)
    room_scene = add_room(materials, render_family)
    hdri_path = set_world_hdri(scene, render_family)
    lighting_preset = add_lighting(render_family)
    camera = add_camera(render_metadata)
    object_names, frame_count, material_assignments = animate_objects(
        render_metadata,
        trajectories,
        materials,
        args.frame_limit,
        material_overrides,
        edge_clarity=args.edge_clarity,
    )
    camera_report = camera_diagnostics(scene, camera, object_names)

    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.filepath = str(args.output_dir / "frame_")
    start = time.monotonic()
    bpy.ops.render.render(animation=True)
    elapsed = time.monotonic() - start
    report = {
        "schema_version": "physv_cycles_pbr_preview_v3",
        "sample_id": metadata["sample_id"],
        "source_sample_dir": str(args.sample_dir),
        "trajectory_source": "raw/trajectories.npz",
        "frame_count": frame_count,
        "fps": fps,
        "resolution": [args.width, args.height],
        "engine": args.engine,
        "samples": args.samples,
        "exposure": args.exposure,
        "output_format": args.output_format,
        "enabled_devices": enabled_devices,
        "texture_sources": {
            "polyhaven_assets": [
                "rubber_tiles",
                "metal_plate",
                "concrete_floor_worn_001",
                "denim_fabric_04",
                "wood_peeling_paint_weathered",
                "pavement_01",
            ],
            "license": "CC0",
            "local_root": str(EXTRA_TEXTURE_ROOT),
            "custom_material_roots": {
                "wood_peeling_paint": str(RAMP_BLOCK_TEXTURE_ROOT),
                "floor_stone_pavement": str(RAMP_FLOOR_TEXTURE_ROOT),
            },
        },
        "texture_coordinates": {
            "wood_peeling_paint": "UV",
            "floor_stone_pavement": "Generated",
            "refine_*_texture": "Generated",
        },
        "visible_refine_texture": {
            "root": str(RAMP_BLOCK_TEXTURE_ROOT),
            "maps": {
                "albedo": "wood_peeling_paint_weathered_diff_2k.jpg",
                "normal": "wood_peeling_paint_weathered_nor_gl_2k.jpg",
                "roughness": "wood_peeling_paint_weathered_rough_2k.jpg",
                "ao": "wood_peeling_paint_weathered_ao_2k.jpg",
            },
            "blend_mode": "COLOR",
            "blend_factor": 0.72,
            "uv_scale": 1.0,
        },
        "natural_texture_assets": {
            "basketball": str(args.basketball_texture) if args.basketball_texture else None,
            "oak_wood": str(REALISM_TEXTURE_ROOT / "oak_wood_planks"),
            "dark_wood": str(REALISM_TEXTURE_ROOT / "dark_wood"),
            "rubber": str(REALISM_TEXTURE_ROOT / "rubberized_track"),
        },
        "hdri": str(hdri_path),
        "room_scene": room_scene,
        "lighting_preset": lighting_preset,
        "object_names": object_names,
        "material_assignments": material_assignments,
        "material_overrides": material_overrides,
        "basketball_texture": str(args.basketball_texture) if args.basketball_texture else None,
        "edge_clarity": bool(args.edge_clarity),
        "edge_clarity_config": {
            "scope": "non-sphere actors only",
            "bevel": "render-only small bevel, max 0.036 m for boxes and 0.030 m for cylinders/pucks",
            "normals": "weighted normals with keep_sharp",
            "highlight": "restrained Fresnel grazing-angle base-color highlight",
            "collision_and_gt_changed": False,
        },
        "camera": camera_report,
        "render_seconds": elapsed,
        "seconds_per_frame": elapsed / max(frame_count, 1),
    }
    (args.output_dir / "render_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
