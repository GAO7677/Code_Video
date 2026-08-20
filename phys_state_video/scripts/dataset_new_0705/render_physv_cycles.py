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
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

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


def material_library() -> dict[str, bpy.types.Material]:
    wood_names = {
        "albedo": "wood_floor_diff_2k.jpg",
        "normal": "wood_floor_nor_gl_2k.jpg",
        "roughness": "wood_floor_rough_2k.jpg",
        "ao": "wood_floor_ao_2k.jpg",
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
    return {
        "floor": pbr_material("PBR_Wood_Floor", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, roughness=0.48, uv_scale=3.0, normal_strength=0.58, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "floor_cool": pbr_material("PBR_Cool_Wood_Floor", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(0.62, 0.78, 0.92), tint_strength=0.76, roughness=0.52, uv_scale=3.0, normal_strength=0.58, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "floor_slate": pbr_material("PBR_Slate_Floor", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.055, 0.075, 0.11), roughness=0.88, uv_scale=2.2, normal_strength=0.54, detail_bump_strength=0.016, detail_bump_scale=18.0),
        "wood": pbr_material("PBR_Wood", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, roughness=0.50, uv_scale=2.0, normal_strength=0.62, detail_bump_strength=0.015, detail_bump_scale=12.0),
        "red_wood": pbr_material("PBR_Red_Wood", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(1.15, 0.30, 0.23), tint_strength=0.72, roughness=0.52, uv_scale=2.4, normal_strength=0.62, detail_bump_strength=0.016, detail_bump_scale=12.0),
        "wall": pbr_material("PBR_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, roughness=0.82, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "wall_cool": pbr_material("PBR_Cool_Wall", texture_dir=TEXTURE_ROOT / "beige_wall_001", texture_names=wall_names, tint=(0.66, 0.82, 1.02), tint_strength=0.68, roughness=0.84, uv_scale=2.2, normal_strength=0.48, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "concrete": pbr_material("PBR_Concrete", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.12, 0.14, 0.17), roughness=0.78, uv_scale=2.0, normal_strength=0.52, detail_bump_strength=0.020, detail_bump_scale=18.0),
        "picture_surface": pbr_material("PBR_Picture_Surface", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_names, tint=(0.22, 0.42, 0.48), tint_strength=0.72, roughness=0.84, uv_scale=1.8, normal_strength=0.44, detail_bump_strength=0.010, detail_bump_scale=18.0),
        "red_rubber": pbr_material("PBR_Red_Rubber", texture_dir=TEXTURE_ROOT / "brown_leather", texture_names=leather_surface_names, tint=(0.35, 0.008, 0.003), roughness=0.78, uv_scale=2.8, normal_strength=0.65, detail_bump_strength=0.010, detail_bump_scale=22.0),
        "blue_rubber": pbr_material("PBR_Blue_Rubber", texture_dir=TEXTURE_ROOT / "brown_leather", texture_names=leather_surface_names, tint=(0.006, 0.028, 0.30), roughness=0.76, uv_scale=2.8, normal_strength=0.65, detail_bump_strength=0.010, detail_bump_scale=22.0),
        "yellow_rubber": pbr_material("PBR_Yellow_Rubber", texture_dir=TEXTURE_ROOT / "brown_leather", texture_names=leather_surface_names, tint=(0.48, 0.16, 0.004), roughness=0.78, uv_scale=2.8, normal_strength=0.65, detail_bump_strength=0.010, detail_bump_scale=22.0),
        "domino_wood": pbr_material("PBR_Domino_Wood", texture_dir=TEXTURE_ROOT / "wood_floor", texture_names=wood_names, tint=(0.52, 0.20, 0.08), tint_strength=0.78, roughness=0.58, uv_scale=3.0, normal_strength=0.66, detail_bump_strength=0.018, detail_bump_scale=14.0),
        "blue_painted": pbr_material("PBR_Blue_Painted", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.006, 0.045, 0.34), roughness=0.46, metallic=0.14, uv_scale=2.0, normal_strength=0.54, detail_bump_strength=0.020, detail_bump_scale=16.0),
        "teal_metal": pbr_material("PBR_Teal_Metal", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.004, 0.11, 0.13), roughness=0.42, metallic=0.28, uv_scale=2.2, normal_strength=0.50, detail_bump_strength=0.020, detail_bump_scale=16.0),
        "yellow_metal": pbr_material("PBR_Yellow_Metal", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.50, 0.20, 0.004), roughness=0.42, metallic=0.22, uv_scale=2.1, normal_strength=0.50, detail_bump_strength=0.020, detail_bump_scale=16.0),
        "dark_metal": pbr_material("PBR_Dark_Metal", texture_dir=TEXTURE_ROOT / "painted_concrete", texture_names=concrete_surface_names, tint=(0.012, 0.016, 0.025), roughness=0.36, metallic=0.55, uv_scale=1.8, normal_strength=0.48, detail_bump_strength=0.024, detail_bump_scale=16.0),
        "fabric": pbr_material("PBR_Fabric", texture_dir=TEXTURE_ROOT / "fabric_pattern_07", texture_names=fabric_names, tint=(0.17, 0.19, 0.19), roughness=0.96, metallic=0.0, uv_scale=5.0, normal_strength=0.56, detail_bump_strength=0.012, detail_bump_scale=28.0),
        "rope_fabric": pbr_material("PBR_Rope_Fabric", texture_dir=TEXTURE_ROOT / "fabric_pattern_07", texture_names=fabric_names, tint=(0.52, 0.25, 0.08), roughness=0.92, metallic=0.0, uv_scale=18.0, normal_strength=0.72, detail_bump_strength=0.022, detail_bump_scale=34.0),
    }


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


def add_actor(name: str, actor: dict, material) -> bpy.types.Object:
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
        return obj
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=float(size["radius"]), depth=float(size["height"]), location=position)
        obj = bpy.context.object
        bpy.ops.object.shade_smooth()
        bevel = obj.modifiers.new("Cylinder bevel", "BEVEL")
        bevel.width = min(float(size["radius"]) * 0.08, 0.018)
        bevel.segments = 3
    else:
        raise ValueError(f"unsupported shape {shape!r} for {name}")
    obj.name = name
    obj.data.materials.append(material)
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
    if family == "V2V_BOWL" and lower == "bowl_ball":
        return "blue_rubber"
    if family == "V2V_BOWL" and (lower == "bowl_base" or lower.startswith("bowl_segment")):
        return "teal_metal"
    if family == "V2V_DOMINO":
        if lower == "domino_trigger_ball":
            return "yellow_rubber"
        if lower.startswith("domino_"):
            return "domino_wood"
    if family == "V2V_GAP" and lower == "gap_ball":
        return "yellow_rubber"
    if family == "V2V_OBSTACLE" and lower == "obstacle_ball":
        return "red_rubber"
    if family == "V2V_PENDULUM":
        if lower == "pendulum_bob":
            return "blue_rubber"
        if lower == "pendulum_rope":
            return "rope_fabric"
        if lower == "pendulum_post":
            return "yellow_metal"
        if lower == "pendulum_base":
            return "concrete"
        if lower == "pendulum_crossbar":
            return "dark_metal"
    if family == "V2V_SEESAW" and lower == "seesaw_load":
        return "yellow_rubber"
    if family == "V2V_SEESAW" and lower == "seesaw_board":
        return "blue_painted"
    if family == "V2V_SEESAW" and lower == "seesaw_hinge_anchor":
        return "dark_metal"

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


def room_material_keys(family: str) -> tuple[str, str]:
    palettes = {
        "F11": ("floor", "wall"),
        "F12": ("floor_cool", "wall_cool"),
        "V2V_BOWL": ("floor_slate", "wall"),
        "V2V_DOMINO": ("floor_cool", "wall_cool"),
        "V2V_GAP": ("floor_slate", "wall"),
        "V2V_OBSTACLE": ("floor_cool", "wall"),
        "V2V_PENDULUM": ("floor", "wall_cool"),
        "V2V_SEESAW": ("floor", "wall_cool"),
    }
    return palettes.get(family, ("floor", "wall"))


def add_room(materials: dict[str, bpy.types.Material], family: str) -> None:
    floor_key, wall_key = room_material_keys(family)
    add_cube("Room floor", (0.0, 1.0, -0.055), (8.0, 7.0, 0.055), materials[floor_key], bevel=0.0)
    add_cube("Back wall", (0.0, 3.55, 3.0), (8.0, 0.06, 3.0), materials[wall_key], bevel=0.0)
    add_cube("Left wall", (-7.95, 0.8, 3.0), (0.06, 2.8, 3.0), materials[wall_key], bevel=0.0)
    add_cube("Right wall", (7.95, 0.8, 3.0), (0.06, 2.8, 3.0), materials[wall_key], bevel=0.0)
    add_cube("Back skirting", (0.0, 3.45, 0.085), (7.95, 0.05, 0.085), materials["wood"], bevel=0.012)

    # Background furniture is deliberately outside the motion corridor.
    add_cube("Sideboard", (-2.35, 3.12, 0.48), (1.05, 0.32, 0.48), materials["wood"], bevel=0.035)
    add_cube("Sideboard top", (-2.35, 3.10, 0.985), (1.12, 0.37, 0.035), materials["dark_metal"], bevel=0.012)
    for x in (-2.95, -2.35, -1.75):
        add_cube(f"Sideboard panel {x}", (x, 2.77, 0.49), (0.27, 0.018, 0.39), materials["dark_metal"], bevel=0.01)
    add_cube("Rug", (2.40, 1.95, 0.012), (1.15, 0.82, 0.012), materials["fabric"], bevel=0.015)

    if family not in {"F11", "F12"}:
        add_cube("Picture frame", (1.65, 3.47, 1.95), (0.72, 0.025, 0.48), materials["dark_metal"], bevel=0.015)
        add_cube("Picture inset", (1.65, 3.43, 1.95), (0.63, 0.012, 0.39), materials["picture_surface"], bevel=0.006)


def set_world_hdri(scene: bpy.types.Scene, family: str) -> Path:
    hdri_by_family = {
        "F11": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "F12": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
        "V2V_BOWL": HDRI_ROOT / "brown_photostudio_02" / "brown_photostudio_02_4k.hdr",
        "V2V_DOMINO": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "V2V_GAP": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
        "V2V_OBSTACLE": HDRI_ROOT / "brown_photostudio_02" / "brown_photostudio_02_4k.hdr",
        "V2V_PENDULUM": HDRI_ROOT / "old_hall" / "old_hall_4k.hdr",
        "V2V_SEESAW": HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr",
    }
    rotation_by_family = {
        "F11": 22.0,
        "F12": -18.0,
        "V2V_BOWL": 22.0,
        "V2V_DOMINO": 58.0,
        "V2V_GAP": 35.0,
        "V2V_OBSTACLE": -24.0,
        "V2V_PENDULUM": 88.0,
        "V2V_SEESAW": -42.0,
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
    background.inputs["Strength"].default_value = 0.30
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


def add_lighting() -> None:
    add_area_light("Key softbox", (-2.2, -2.4, 3.6), (0.0, 0.4, 0.5), 680.0, 3.0, (1.0, 0.88, 0.76))
    add_area_light("Fill softbox", (3.2, -1.1, 2.7), (0.2, 0.5, 0.45), 400.0, 2.6, (0.78, 0.88, 1.0))
    add_area_light("Top bounce", (0.0, 1.8, 4.4), (0.0, 0.6, 0.0), 300.0, 3.4, (1.0, 0.97, 0.90))


def add_camera(metadata: dict) -> bpy.types.Object:
    camera_spec = metadata["camera"]
    intrinsics = camera_spec["intrinsics"]
    extrinsics = camera_spec["extrinsics"]
    data = bpy.data.cameras.new("PhysV Camera")
    data.type = "PERSP"
    data.sensor_fit = "VERTICAL"
    data.angle_y = math.radians(float(intrinsics["yfov_deg"]))
    obj = bpy.data.objects.new("PhysV Camera", data)
    bpy.context.collection.objects.link(obj)
    obj.location = extrinsics["eye"]
    point_at(obj, extrinsics["target"])
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
        "object_projections_xy_depth": projections,
    }
    print("CAMERA_DIAGNOSTICS", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def animate_objects(metadata: dict, trajectories, materials, frame_limit: int) -> tuple[list[str], int, dict[str, str]]:
    family = metadata["family_key"]
    names = trajectories["object_names"]
    available_frames = len(trajectories["frame_times_s"])
    frame_count = min(frame_limit, available_frames) if frame_limit > 0 else available_frames
    material_assignments = {}
    for name in names:
        actor = metadata["actors"][name]
        material_key = actor_material_key(name, actor, family)
        material_assignments[name] = material_key
        material = materials[material_key]
        obj = add_actor(name, actor, material)
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
    fps = int(metadata["simulation"]["fps"])

    clear_scene()
    scene = bpy.context.scene
    enabled_devices = configure_cycles(scene, args, fps)
    materials = material_library()
    add_room(materials, metadata["family_key"])
    hdri_path = set_world_hdri(scene, metadata["family_key"])
    add_lighting()
    camera = add_camera(metadata)
    object_names, frame_count, material_assignments = animate_objects(metadata, trajectories, materials, args.frame_limit)
    camera_report = camera_diagnostics(scene, camera, object_names)

    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.filepath = str(args.output_dir / "frame_")
    start = time.monotonic()
    bpy.ops.render.render(animation=True)
    elapsed = time.monotonic() - start
    report = {
        "schema_version": "physv_cycles_pbr_preview_v1",
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
        "hdri": str(hdri_path),
        "object_names": object_names,
        "material_assignments": material_assignments,
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
