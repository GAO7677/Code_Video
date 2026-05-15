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
    principled = nt.nodes.get("Principled BSDF")
    if principled is None:
        principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    base_color = spec.get("base_color", [0.7, 0.7, 0.7, 1.0])
    principled.inputs["Base Color"].default_value = tuple(base_color)
    principled.inputs["Roughness"].default_value = float(spec.get("roughness", 0.45))
    principled.inputs["Specular"].default_value = float(spec.get("specular", 0.45))
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = float(spec.get("metallic", 0.0))
    if "Clearcoat" in principled.inputs:
        principled.inputs["Clearcoat"].default_value = float(spec.get("clearcoat", 0.0))
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


def configure_render(scene: bpy.types.Scene, render_spec: dict) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(render_spec.get("samples", 32))
    scene.cycles.preview_samples = max(8, int(render_spec.get("samples", 32) // 2))
    scene.cycles.use_denoising = bool(render_spec.get("use_denoising", False))
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
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -1.2
    scene.view_settings.gamma = 1.0
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.055, 0.068, 0.092, 1.0)
        bg.inputs[1].default_value = 0.09


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

    render_collection = bpy.data.collections.new("RenderCase")
    scene.collection.children.link(render_collection)

    create_floor(
        extents=spec["ground"]["extents_xy"],
        center=spec["ground"]["center"],
        material_spec=spec["ground"]["material"],
    )
    create_area_light(
        name="KeyArea",
        location=spec["lighting"]["key_area"]["location"],
        rotation_euler_deg=spec["lighting"]["key_area"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["key_area"]["energy"]),
        size=float(spec["lighting"]["key_area"]["size"]),
    )
    create_area_light(
        name="FillArea",
        location=spec["lighting"]["fill_area"]["location"],
        rotation_euler_deg=spec["lighting"]["fill_area"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["fill_area"]["energy"]),
        size=float(spec["lighting"]["fill_area"]["size"]),
    )
    create_area_light(
        name="RimArea",
        location=spec["lighting"]["rim_area"]["location"],
        rotation_euler_deg=spec["lighting"]["rim_area"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["rim_area"]["energy"]),
        size=float(spec["lighting"]["rim_area"]["size"]),
    )
    create_sun_light(
        name="SunKey",
        rotation_euler_deg=spec["lighting"]["sun"]["rotation_euler_deg"],
        energy=float(spec["lighting"]["sun"]["energy"]),
    )

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
