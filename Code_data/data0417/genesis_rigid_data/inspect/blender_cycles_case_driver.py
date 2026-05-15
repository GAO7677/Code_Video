# 用途：在 Blender 中读取导出的 case scene spec，并用 Cycles 离线渲染预览视频。
"""Blender-side driver for Genesis rigid case Cycles previews.

This script is meant to be executed by Blender:

  blender -b -P blender_cycles_case_driver.py -- --spec_json /path/to/spec.json

The host-side script prepares a compact scene spec JSON with:
- camera parameters
- animated rigid object COM + quaternion tracks
- mesh parts for each object in local object coordinates
- preview render settings

The Blender scene uses Cycles with simple but more realistic Principled BSDF
materials instead of saturated random colors.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import mathutils


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec_json", type=Path, required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    for datablock in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.images,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.curves,
    ):
        for item in list(datablock):
            if item.users == 0:
                datablock.remove(item)


def import_obj(filepath: str) -> list[bpy.types.Object]:
    before = {obj.name for obj in bpy.data.objects}
    try:
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward="-Z", axis_up="Y")
    except Exception:
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis="NEGATIVE_Z", up_axis="Y")
    return [obj for obj in bpy.data.objects if obj.name not in before]


def make_principled_material(name: str, spec: dict) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nt = material.node_tree
    nodes = nt.nodes
    links = nt.links
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)

    output = nodes.get("Material Output")
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")

    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (420.0, 0.0)
    base_color = spec.get("base_color", [0.7, 0.7, 0.7, 1.0])
    principled.inputs["Base Color"].default_value = tuple(base_color)
    principled.inputs["Roughness"].default_value = float(spec.get("roughness", 0.45))
    principled.inputs["Specular"].default_value = float(spec.get("specular", 0.45))
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = float(spec.get("metallic", 0.0))
    if "Clearcoat" in principled.inputs:
        principled.inputs["Clearcoat"].default_value = float(spec.get("clearcoat", 0.0))
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-980.0, -20.0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-780.0, -20.0)
    mapping.inputs["Scale"].default_value = (1.4, 1.4, 1.4)
    links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])

    noise_a = nodes.new("ShaderNodeTexNoise")
    noise_a.location = (-560.0, 120.0)
    noise_a.inputs["Scale"].default_value = 6.0
    noise_a.inputs["Detail"].default_value = 8.0
    noise_a.inputs["Roughness"].default_value = 0.55
    links.new(mapping.outputs["Vector"], noise_a.inputs["Vector"])

    noise_b = nodes.new("ShaderNodeTexNoise")
    noise_b.location = (-560.0, -120.0)
    noise_b.inputs["Scale"].default_value = 38.0
    noise_b.inputs["Detail"].default_value = 3.0
    noise_b.inputs["Roughness"].default_value = 0.45
    links.new(mapping.outputs["Vector"], noise_b.inputs["Vector"])

    mixrgb = nodes.new("ShaderNodeMixRGB")
    mixrgb.location = (-170.0, 120.0)
    mixrgb.blend_type = "MULTIPLY"
    mixrgb.inputs["Fac"].default_value = 0.12
    mixrgb.inputs["Color1"].default_value = tuple(base_color)

    bright = nodes.new("ShaderNodeBrightContrast")
    bright.location = (-370.0, 120.0)
    bright.inputs["Bright"].default_value = 0.02
    bright.inputs["Contrast"].default_value = 0.18
    links.new(noise_a.outputs["Color"], bright.inputs["Color"])
    links.new(bright.outputs["Color"], mixrgb.inputs["Color2"])
    base_mix = mixrgb

    layer_weight = nodes.new("ShaderNodeLayerWeight")
    layer_weight.location = (-360.0, 250.0)
    layer_weight.inputs["Blend"].default_value = 0.22

    edge_mix = nodes.new("ShaderNodeMixRGB")
    edge_mix.location = (110.0, 120.0)
    edge_mix.blend_type = "SCREEN"
    edge_mix.inputs["Fac"].default_value = 0.10
    edge_mix.inputs["Color2"].default_value = tuple(base_color)
    links.new(layer_weight.outputs["Facing"], edge_mix.inputs["Fac"])
    links.new(base_mix.outputs["Color"], edge_mix.inputs["Color1"])
    links.new(edge_mix.outputs["Color"], principled.inputs["Base Color"])

    rough_math = nodes.new("ShaderNodeMath")
    rough_math.location = (-180.0, -60.0)
    rough_math.operation = "MULTIPLY_ADD"
    rough_math.inputs[1].default_value = 0.08
    rough_math.inputs[2].default_value = float(spec.get("roughness", 0.45))
    links.new(noise_b.outputs["Fac"], rough_math.inputs[0])
    links.new(rough_math.outputs["Value"], principled.inputs["Roughness"])

    bevel = nodes.new("ShaderNodeBevel")
    bevel.location = (-180.0, -280.0)
    bevel.samples = 4
    bevel.inputs["Radius"].default_value = 0.0016

    bump = nodes.new("ShaderNodeBump")
    bump.location = (100.0, -220.0)
    bump.inputs["Strength"].default_value = 0.06
    bump.inputs["Distance"].default_value = 0.02
    links.new(noise_b.outputs["Fac"], bump.inputs["Height"])
    links.new(bevel.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    preset = str(spec.get("material_preset", "painted_metal"))
    if preset == "varnished_wood":
        wave = nodes.new("ShaderNodeTexWave")
        wave.location = (-560.0, 320.0)
        wave.wave_type = "BANDS"
        wave.bands_direction = "Y"
        wave.inputs["Scale"].default_value = 12.0
        wave.inputs["Distortion"].default_value = 4.0
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])

        wood_mix = nodes.new("ShaderNodeMixRGB")
        wood_mix.location = (-170.0, 320.0)
        wood_mix.blend_type = "MULTIPLY"
        wood_mix.inputs["Fac"].default_value = 0.22
        wood_mix.inputs["Color1"].default_value = tuple(base_color)
        links.new(wave.outputs["Color"], wood_mix.inputs["Color2"])
        links.new(wood_mix.outputs["Color"], edge_mix.inputs["Color1"])
        principled.inputs["Roughness"].default_value = max(0.42, float(spec.get("roughness", 0.45)))
        if "Clearcoat" in principled.inputs:
            principled.inputs["Clearcoat"].default_value = 0.18
        bump.inputs["Strength"].default_value = 0.04
    elif preset == "fabric_cloth":
        weave_a = nodes.new("ShaderNodeTexWave")
        weave_a.location = (-560.0, 300.0)
        weave_a.wave_type = "BANDS"
        weave_a.bands_direction = "X"
        weave_a.inputs["Scale"].default_value = 95.0
        weave_a.inputs["Distortion"].default_value = 1.0
        links.new(mapping.outputs["Vector"], weave_a.inputs["Vector"])

        weave_b = nodes.new("ShaderNodeTexWave")
        weave_b.location = (-560.0, 430.0)
        weave_b.wave_type = "BANDS"
        weave_b.bands_direction = "Y"
        weave_b.inputs["Scale"].default_value = 78.0
        weave_b.inputs["Distortion"].default_value = 1.2
        links.new(mapping.outputs["Vector"], weave_b.inputs["Vector"])

        weave_mix = nodes.new("ShaderNodeMixRGB")
        weave_mix.location = (-300.0, 360.0)
        weave_mix.blend_type = "MULTIPLY"
        weave_mix.inputs["Fac"].default_value = 0.42
        links.new(weave_a.outputs["Color"], weave_mix.inputs["Color1"])
        links.new(weave_b.outputs["Color"], weave_mix.inputs["Color2"])

        fabric_tint = nodes.new("ShaderNodeMixRGB")
        fabric_tint.location = (-70.0, 260.0)
        fabric_tint.blend_type = "MULTIPLY"
        fabric_tint.inputs["Fac"].default_value = 0.18
        fabric_tint.inputs["Color1"].default_value = tuple(base_color)
        links.new(weave_mix.outputs["Color"], fabric_tint.inputs["Color2"])
        links.new(fabric_tint.outputs["Color"], edge_mix.inputs["Color1"])

        rough_math.inputs[1].default_value = 0.14
        rough_math.inputs[2].default_value = max(0.78, float(spec.get("roughness", 0.45)))
        principled.inputs["Specular"].default_value = 0.20
        if "Sheen" in principled.inputs:
            principled.inputs["Sheen"].default_value = 0.35
            principled.inputs["Sheen Tint"].default_value = 0.25
        bump.inputs["Strength"].default_value = 0.09
        bump.inputs["Distance"].default_value = 0.006
        links.new(weave_mix.outputs["Color"], bump.inputs["Height"])
    elif preset == "rubber_plastic":
        mixrgb.inputs["Fac"].default_value = 0.05
        rough_math.inputs[1].default_value = 0.10
        rough_math.inputs[2].default_value = max(0.48, float(spec.get("roughness", 0.45)))
        principled.inputs["Specular"].default_value = 0.28
        bump.inputs["Strength"].default_value = 0.02
    elif preset == "painted_plastic":
        mixrgb.inputs["Fac"].default_value = 0.07
        rough_math.inputs[1].default_value = 0.05
        rough_math.inputs[2].default_value = max(0.36, float(spec.get("roughness", 0.45)))
        principled.inputs["Specular"].default_value = 0.42
        if "Clearcoat" in principled.inputs:
            principled.inputs["Clearcoat"].default_value = 0.16
        bump.inputs["Strength"].default_value = 0.025
    elif preset == "hard_plastic":
        principled.inputs["Specular"].default_value = 0.46
        rough_math.inputs[1].default_value = 0.06
        bump.inputs["Strength"].default_value = 0.03
    elif preset == "concrete_floor":
        mixrgb.inputs["Fac"].default_value = 0.18
        rough_math.inputs[1].default_value = 0.14
        rough_math.inputs[2].default_value = max(0.72, float(spec.get("roughness", 0.45)))
        principled.inputs["Specular"].default_value = 0.16
        bump.inputs["Strength"].default_value = 0.08
        bevel.inputs["Radius"].default_value = 0.0
    elif preset == "painted_wall":
        mixrgb.inputs["Fac"].default_value = 0.08
        rough_math.inputs[1].default_value = 0.06
        rough_math.inputs[2].default_value = max(0.78, float(spec.get("roughness", 0.45)))
        principled.inputs["Specular"].default_value = 0.10
        bump.inputs["Strength"].default_value = 0.025
        bevel.inputs["Radius"].default_value = 0.0
    else:
        if "Metallic" in principled.inputs:
            principled.inputs["Metallic"].default_value = max(float(spec.get("metallic", 0.0)), 0.18)
        if "Clearcoat" in principled.inputs:
            principled.inputs["Clearcoat"].default_value = max(float(spec.get("clearcoat", 0.0)), 0.10)
    return material


def attach_material(obj: bpy.types.Object, material: bpy.types.Material) -> None:
    if obj.type != "MESH":
        return
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def point_camera_at(camera_obj: bpy.types.Object, position: list[float], lookat: list[float]) -> None:
    pos = mathutils.Vector(position)
    target = mathutils.Vector(lookat)
    direction = target - pos
    if direction.length <= 1e-8:
        direction = mathutils.Vector((0.0, 1.0, -0.2))
    camera_obj.location = pos
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = direction.to_track_quat("-Z", "Y")


def create_floor(extents: list[float], center: list[float], material_spec: dict) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = "Ground"
    obj.scale = (float(extents[0]) * 0.5, float(extents[1]) * 0.5, 1.0)
    obj.location = (float(center[0]), float(center[1]), float(center[2]))
    attach_material(obj, make_principled_material("GroundMat", material_spec))
    return obj


def create_wall(
    name: str,
    *,
    location: tuple[float, float, float],
    rotation_euler_deg: tuple[float, float, float],
    size_xy: tuple[float, float],
    material_spec: dict,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.rotation_euler = tuple(math.radians(v) for v in rotation_euler_deg)
    obj.scale = (float(size_xy[0]) * 0.5, float(size_xy[1]) * 0.5, 1.0)
    attach_material(obj, make_principled_material(f"{name}Mat", material_spec))
    return obj


def create_room_shell(room_spec: dict) -> None:
    if not bool(room_spec.get("enabled", False)):
        return
    center = room_spec["center"]
    cx, cy = float(center[0]), float(center[1])
    width = float(room_spec["width"])
    depth = float(room_spec["depth"])
    height = float(room_spec["height"])
    mat = dict(room_spec["wall_material"])
    create_wall(
        "BackWall",
        location=(cx, cy + depth * 0.5, height * 0.5),
        rotation_euler_deg=(90.0, 0.0, 0.0),
        size_xy=(width, height),
        material_spec=mat,
    )
    create_wall(
        "LeftWall",
        location=(cx - width * 0.5, cy, height * 0.5),
        rotation_euler_deg=(90.0, 0.0, 90.0),
        size_xy=(depth, height),
        material_spec=mat,
    )
    create_wall(
        "RightWall",
        location=(cx + width * 0.5, cy, height * 0.5),
        rotation_euler_deg=(90.0, 0.0, -90.0),
        size_xy=(depth, height),
        material_spec=mat,
    )


def create_area_light(name: str, location: list[float], rotation_euler_deg: list[float], energy: float, size: float) -> bpy.types.Object:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = float(energy)
    data.shape = "RECTANGLE"
    data.size = float(size)
    data.size_y = float(size * 0.75)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = tuple(float(v) for v in location)
    obj.rotation_euler = tuple(math.radians(float(v)) for v in rotation_euler_deg)
    return obj


def create_sun_light(name: str, rotation_euler_deg: list[float], energy: float) -> bpy.types.Object:
    data = bpy.data.lights.new(name=name, type="SUN")
    data.energy = float(energy)
    data.angle = math.radians(9.0)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.rotation_euler = tuple(math.radians(float(v)) for v in rotation_euler_deg)
    return obj


def set_light_color(obj: bpy.types.Object, color: list[float] | None) -> None:
    if obj is None or obj.data is None or color is None:
        return
    rgb = tuple(float(max(0.0, min(1.0, v))) for v in color[:3])
    try:
        obj.data.color = rgb
    except Exception:
        pass


def configure_render(scene: bpy.types.Scene, render_spec: dict) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(render_spec.get("samples", 32))
    scene.cycles.preview_samples = max(8, int(render_spec.get("samples", 32) // 2))
    scene.cycles.use_denoising = bool(render_spec.get("use_denoising", False))
    if hasattr(scene.cycles, "use_adaptive_sampling"):
        scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.cycles.transmission_bounces = 6
    scene.cycles.transparent_max_bounces = 4
    scene.cycles.filter_glossy = 0.4
    if hasattr(scene.cycles, "sample_clamp_indirect"):
        scene.cycles.sample_clamp_indirect = 2.0
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.render.resolution_x = int(render_spec.get("width", 640))
    scene.render.resolution_y = int(render_spec.get("height", 480))
    scene.render.resolution_percentage = 100
    scene.render.fps = int(render_spec.get("fps", 12))
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.film_transparent = False
    if hasattr(scene.view_settings, "view_transform"):
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = float(render_spec.get("exposure", -0.55))
    scene.view_settings.gamma = 1.0
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True


def configure_world(scene: bpy.types.Scene, environment_spec: dict) -> None:
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    nodes = nt.nodes
    links = nt.links

    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (420.0, 0.0)

    bg_env = nodes.new("ShaderNodeBackground")
    bg_env.location = (180.0, 120.0)
    bg_env.inputs["Strength"].default_value = float(environment_spec.get("strength", 0.25))

    bg_plain = nodes.new("ShaderNodeBackground")
    bg_plain.location = (180.0, -80.0)
    bg_plain.inputs["Strength"].default_value = float(environment_spec.get("background_strength", 0.02))
    bg_plain.inputs["Color"].default_value = tuple(environment_spec.get("background_color", [0.01, 0.01, 0.02, 1.0]))

    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (360.0, 0.0)

    light_path = nodes.new("ShaderNodeLightPath")
    light_path.location = (-60.0, -120.0)

    texcoord = nodes.new("ShaderNodeTexCoord")
    texcoord.location = (-760.0, 120.0)

    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-560.0, 120.0)
    mapping.inputs["Rotation"].default_value[2] = math.radians(float(environment_spec.get("rotation_deg", 0.0)))

    env_tex = nodes.new("ShaderNodeTexEnvironment")
    env_tex.location = (-320.0, 120.0)
    world_exr = str(environment_spec.get("world_exr", "") or "")
    if world_exr and Path(world_exr).exists():
        env_tex.image = bpy.data.images.load(world_exr, check_existing=True)

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"], bg_env.inputs["Color"])
    links.new(bg_env.outputs["Background"], mix.inputs[1])
    links.new(bg_plain.outputs["Background"], mix.inputs[2])
    links.new(light_path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


def build_animated_mesh_object(
    scene: bpy.types.Scene,
    collection: bpy.types.Collection,
    object_spec: dict,
) -> bpy.types.Object:
    root = bpy.data.objects.new(str(object_spec["name"]), None)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 0.12
    root.rotation_mode = "QUATERNION"
    collection.objects.link(root)

    for part_idx, part in enumerate(object_spec.get("parts", [])):
        imported = import_obj(str(part["mesh_path"]))
        mat = make_principled_material(
            name=f"{root.name}_mat_{part_idx:02d}",
            spec=part["material"],
        )
        for obj in imported:
            if obj.type != "MESH":
                continue
            obj.parent = root
            obj.location = tuple(float(v) for v in part.get("local_offset", [0.0, 0.0, 0.0]))
            obj.scale = tuple(float(v) for v in part.get("local_scale", [1.0, 1.0, 1.0]))
            obj.rotation_euler = (0.0, 0.0, 0.0)
            if getattr(obj.data, "polygons", None) is not None:
                for poly in obj.data.polygons:
                    poly.use_smooth = True
            if hasattr(obj.data, "use_auto_smooth"):
                obj.data.use_auto_smooth = True
            if hasattr(obj, "cycles"):
                obj.cycles.is_shadow_catcher = False
            attach_material(obj, mat)
    return root


def build_animated_sphere(
    scene: bpy.types.Scene,
    collection: bpy.types.Collection,
    object_spec: dict,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=48,
        ring_count=24,
        radius=float(object_spec["radius"]),
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.active_object
    obj.name = str(object_spec["name"])
    obj.rotation_mode = "QUATERNION"
    collection.objects.link(obj)
    if obj.name not in collection.objects:
        try:
            scene.collection.objects.unlink(obj)
        except Exception:
            pass
    attach_material(obj, make_principled_material(f"{obj.name}_mat", object_spec["material"]))
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def keyframe_track(obj: bpy.types.Object, frames: list[dict]) -> None:
    obj.rotation_mode = "QUATERNION"
    for item in frames:
        frame_no = int(item["timeline_frame"])
        obj.location = tuple(float(v) for v in item["position"])
        quat = item.get("quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
        obj.rotation_quaternion = mathutils.Quaternion(tuple(float(v) for v in quat))
        obj.keyframe_insert(data_path="location", frame=frame_no)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_no)


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec_json.read_text(encoding="utf-8"))

    clear_scene()
    scene = bpy.context.scene
    configure_render(scene, spec["render"])
    configure_world(scene, spec["environment"])

    render_collection = bpy.data.collections.new("RenderCase")
    scene.collection.children.link(render_collection)

    create_floor(
        extents=spec["ground"]["extents_xy"],
        center=spec["ground"]["center"],
        material_spec=spec["ground"]["material"],
    )
    create_room_shell(spec["room"])
    key_light = create_area_light(
        name="KeyArea",
        location=spec["lighting"]["key_area"]["location"],
        rotation_euler_deg=spec["lighting"]["key_area"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["key_area"]["energy"]),
        size=float(spec["lighting"]["key_area"]["size"]),
    )
    set_light_color(key_light, spec["lighting"]["key_area"].get("color"))
    fill_light = create_area_light(
        name="FillArea",
        location=spec["lighting"]["fill_area"]["location"],
        rotation_euler_deg=spec["lighting"]["fill_area"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["fill_area"]["energy"]),
        size=float(spec["lighting"]["fill_area"]["size"]),
    )
    set_light_color(fill_light, spec["lighting"]["fill_area"].get("color"))
    rim_light = create_area_light(
        name="RimArea",
        location=spec["lighting"]["rim_area"]["location"],
        rotation_euler_deg=spec["lighting"]["rim_area"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["rim_area"]["energy"]),
        size=float(spec["lighting"]["rim_area"]["size"]),
    )
    set_light_color(rim_light, spec["lighting"]["rim_area"].get("color"))
    sun_light = create_sun_light(
        name="SunKey",
        rotation_euler_deg=spec["lighting"]["sun"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["sun"]["energy"]),
    )
    set_light_color(sun_light, spec["lighting"]["sun"].get("color"))

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(float(spec["camera"]["fov_deg"]))
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    render_collection.objects.link(cam_obj)
    point_camera_at(cam_obj, spec["camera"]["position"], spec["camera"]["lookat"])
    scene.camera = cam_obj

    built_objects = []
    for object_spec in spec.get("objects", []):
        if object_spec["kind"] == "animated_mesh":
            obj = build_animated_mesh_object(scene, render_collection, object_spec)
        elif object_spec["kind"] == "animated_sphere":
            obj = build_animated_sphere(scene, render_collection, object_spec)
        else:
            continue
        keyframe_track(obj, object_spec["frames"])
        built_objects.append(obj)

    timeline = spec["timeline"]
    scene.frame_start = int(timeline["frame_start"])
    scene.frame_end = int(timeline["frame_end"])
    scene.frame_set(scene.frame_start)

    output_root = Path(spec["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_root / "cycles_preview.mp4")
    bpy.ops.wm.save_mainfile(filepath=str(output_root / "cycles_preview.blend"))
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
