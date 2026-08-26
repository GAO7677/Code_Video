#!/usr/bin/env python3
"""Export CYCLES-coordinate object truth for an existing PhysV sample.

The RGB preview is built by ``render_physv_cycles.py`` from the same sample
metadata, camera, materials, and ``raw/trajectories.npz``.  This companion
script rebuilds that exact Blender scene and renders the Object Index pass at
the preview's native resolution.  It does not rerun PyBullet and it does not
modify the existing RGB preview or the original truth files.

The exported ``dynamic_masks.npz`` and ``cycles_depth.npz`` are therefore
aligned to ``videos/rgb_cycles.mp4`` in frame order and pixel coordinates.
Collision and state truth remain simulator-time labels and are copied by the
batch wrapper; they do not need a new pixel-space render.

In addition to the project-specific aligned-truth files, ``rigidbench/`` is a
small per-case adapter with the file names and array conventions expected by
RigidBench (``masks.npz``, ``depth.npz``, ``trajectories.npz``,
``metadata.json``, and a link to ``video.mp4``).  It is deliberately nested
under the new aligned-truth output so existing raw truth and RGB previews are
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import render_physv_cycles as rgb_render  # noqa: E402


def _argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--device", choices=("CUDA", "CPU"), default="CUDA")
    parser.add_argument("--frame-limit", type=int, default=0)
    return parser.parse_args(_argv())


def _trajectory_payload(sample_dir: Path) -> dict:
    arrays = np.load(sample_dir / "raw" / "trajectories.npz", allow_pickle=False)
    names = [str(value) for value in arrays["object_names"]]
    payload: dict[str, object] = {
        "object_names": names,
        "frame_times_s": arrays["frame_times_s"].tolist(),
    }
    for name in names:
        payload[f"{name}_positions"] = arrays[f"{name}_positions"].tolist()
        payload[f"{name}_rotations"] = arrays[f"{name}_rotations"].tolist()
    return payload


def _dynamic_names(sample_dir: Path) -> list[str]:
    with np.load(sample_dir / "raw" / "masks.npz", allow_pickle=False) as arrays:
        names = [str(value) for value in arrays["object_names"]]
        roles = [str(value) for value in arrays["object_roles"]]
    selected = [name for name, role in zip(names, roles) if role == "dynamic"]
    if not selected:
        selected = names
    return selected


def _configure_index_compositor(scene, output_dir: Path):
    """Write IndexOB and Depth passes to temporary EXRs for each frame.

    Blender 3.6 keeps ``Render Result`` at 0x0 in background mode for this
    pass, while a File Output compositor node receives the full pass.  Reading
    the temporary EXR also avoids relying on a version-specific Render Result
    API.
    """

    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    layers = tree.nodes.new("CompositorNodeRLayers")
    outputs = []
    for pass_name, directory in (("IndexOB", "_index_pass"), ("Depth", "_depth_pass")):
        output = tree.nodes.new("CompositorNodeOutputFile")
        output.base_path = str(output_dir / directory)
        output.file_slots[0].path = "frame_"
        output.format.file_format = "OPEN_EXR"
        output.format.color_depth = "32"
        output.format.color_mode = "RGBA"
        tree.links.new(layers.outputs[pass_name], output.inputs[0])
        outputs.append(output)
    return tuple(outputs)


def _index_pass_path(output_node, output_dir: Path, frame_index: int) -> Path:
    base = Path(output_node.base_path)
    stem = str(output_node.file_slots[0].path)
    return base / f"{stem}{frame_index:04d}.exr"


def _extract_scalar_file(path: Path, width: int, height: int) -> np.ndarray:
    """Read the first channel of a scalar compositor EXR as (H, W) float32."""
    if not path.is_file():
        raise RuntimeError(f"compositor output is missing: {path}")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        if tuple(image.size) != (width, height):
            raise RuntimeError(f"EXR size {tuple(image.size)} != {(width, height)}: {path}")
        values = np.asarray(image.pixels[:], dtype=np.float32)
    finally:
        bpy.data.images.remove(image)
    pixels = width * height
    if values.size == pixels:
        scalar = values.reshape(height, width)
    elif values.size % pixels != 0:
        raise RuntimeError(f"unexpected EXR buffer size {values.size}: {path}")
    else:
        channels = values.size // pixels
        scalar = values.reshape(height, width, channels)[..., 0]
    # Blender image buffers are bottom-up; saved videos and projected tracks
    # use a top-left origin, so both IndexOB and Depth need the same flip.
    return np.flipud(scalar)


def _extract_index_file(path: Path, width: int, height: int) -> np.ndarray:
    return np.rint(_extract_scalar_file(path, width, height)).astype(np.int32)


def _extract_depth_file(path: Path, width: int, height: int) -> np.ndarray:
    depth = _extract_scalar_file(path, width, height)
    # Blender's Z pass can encode non-finite values for rays without a hit.
    # RigidBench uses zero as the invalid/background depth convention.
    return np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0).astype(np.float32)


def _camera_calibration(camera, width: int, height: int) -> dict:
    """Export the CYCLES camera as explicit K and Blender camera pose."""
    angle_y = float(camera.data.angle_y)
    fy = 0.5 * float(height) / np.tan(0.5 * angle_y)
    # The renderer uses sensor_fit=VERTICAL and square pixels.
    fx = fy
    matrix_world = [[float(value) for value in row] for row in camera.matrix_world]
    world_to_camera = camera.matrix_world.inverted()
    world_to_camera = [[float(value) for value in row] for row in world_to_camera]
    quaternion = [float(value) for value in camera.matrix_world.to_quaternion()]
    location = [float(value) for value in camera.matrix_world.translation]
    rotation = camera.matrix_world.to_quaternion()
    forward = rotation @ Vector((0.0, 0.0, -1.0))
    up = rotation @ Vector((0.0, 1.0, 0.0))
    target = [float(value) for value in camera["target"]]
    return {
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(width) / 2.0,
            "cy": float(height) / 2.0,
            "width": int(width),
            "height": int(height),
            "yfov_deg": float(np.degrees(angle_y)),
        },
        "extrinsics": {
            "location": location,
            "rotation": quaternion,
            "eye": location,
            "target": target,
            "forward": [float(value) for value in forward],
            "up": [float(value) for value in up],
            "camera_to_world_blender": matrix_world,
            "world_to_camera_blender": world_to_camera,
            "coordinate_convention": (
                "Blender camera pose; local -Z is forward, local +Y is up; "
                "rotation is quaternion wxyz"
            ),
        },
    }


def _write_rigidbench_adapter(
    output_dir: Path,
    sample_dir: Path,
    metadata: dict,
    dynamic_names: list[str],
    dynamic_masks: np.ndarray,
    depth_frames: np.ndarray,
    camera_calibration: dict,
) -> None:
    """Write one RigidBench-compatible sample beneath the aligned-truth case."""
    adapter_dir = output_dir / "rigidbench"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        adapter_dir / "masks.npz",
        masks=np.transpose(dynamic_masks, (1, 0, 2, 3)),
        object_names=np.asarray(dynamic_names),
        object_roles=np.asarray(["active"] * len(dynamic_names)),
    )
    np.savez_compressed(adapter_dir / "depth.npz", depth=depth_frames)

    with np.load(sample_dir / "raw" / "trajectories.npz", allow_pickle=False) as arrays:
        trajectory_payload = {
            "object_names": arrays["object_names"],
            "frame_times_s": arrays["frame_times_s"],
        }
        for name in arrays["object_names"]:
            name = str(name)
            for suffix in ("positions", "rotations"):
                trajectory_payload[f"{name}_{suffix}"] = arrays[f"{name}_{suffix}"]
    np.savez_compressed(adapter_dir / "trajectories.npz", **trajectory_payload)

    adapter_metadata = dict(metadata)
    adapter_metadata["camera"] = camera_calibration
    adapter_metadata["rigidbench_source"] = {
        "rgb_cycles": str(sample_dir / "videos" / "rgb_cycles.mp4"),
        "aligned_truth_case": str(output_dir),
        "depth_source": "CYCLES compositor Depth/Z pass",
        "mask_source": "CYCLES compositor IndexOB pass",
    }
    adapter_metadata["rigidbench_role_mapping"] = {
        "dynamic": "active",
        "anchored_fixture": "static",
        "anchored_static": "static",
    }
    adapter_metadata["actors"] = {
        name: {**info, "role": "active" if info.get("role") == "dynamic" else "static"}
        for name, info in metadata.get("actors", {}).items()
    }
    adapter_metadata["task_type"] = str(metadata.get("task_type") or metadata.get("family_key", "physv"))
    captions = metadata.get("captions") or {}
    prompt = (
        metadata.get("prompt")
        or metadata.get("caption")
        or (captions.get("abstract") or {}).get("text")
        or (captions.get("specific") or {}).get("text")
        or ""
    )
    adapter_metadata["prompt"] = str(prompt)
    adapter_metadata["rigidbench_prompt_source"] = (
        "metadata.prompt"
        if metadata.get("prompt")
        else "metadata.captions.abstract.text"
        if (captions.get("abstract") or {}).get("text")
        else "metadata.captions.specific.text"
        if (captions.get("specific") or {}).get("text")
        else "missing"
    )
    (adapter_dir / "metadata.json").write_text(
        json.dumps(adapter_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    video_link = adapter_dir / "video.mp4"
    video_link.unlink(missing_ok=True)
    video_link.symlink_to(sample_dir / "videos" / "rgb_cycles.mp4")


def _pixel_centers(scene, camera, names: list[str], width: int, height: int) -> np.ndarray:
    rows = []
    for name in names:
        projected = world_to_camera_view(scene, camera, bpy.data.objects[name].matrix_world.translation)
        rows.append((float(projected.x * width), float((1.0 - projected.y) * height), float(projected.z)))
    return np.asarray(rows, dtype=np.float32)


def main() -> None:
    args = parse_args()
    sample_dir = args.sample_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    trajectory = _trajectory_payload(sample_dir)
    dynamic_names = _dynamic_names(sample_dir)
    render_family = "F12" if metadata["family_key"] == "F12_RAMP_LENGTH" else metadata["family_key"]
    if render_family == "SCENE_DOOR_FRAME_BALL":
        render_family = "SCENE_DOOR_FRAME"
    render_metadata = dict(metadata)
    render_metadata["family_key"] = render_family

    rgb_render.clear_scene()
    scene = bpy.context.scene
    enabled_devices = rgb_render.configure_cycles(
        scene,
        argparse.Namespace(
            engine="CYCLES",
            width=args.width,
            height=args.height,
            samples=args.samples,
            exposure=args.exposure,
            device=args.device,
            output_format="PNG",
        ),
        int(metadata["simulation"]["fps"]),
    )
    materials = rgb_render.material_library()
    room_scene = rgb_render.add_room(materials, render_family)
    hdri_path = rgb_render.set_world_hdri(scene, render_family)
    lighting_preset = rgb_render.add_lighting(render_family)
    camera = rgb_render.add_camera(render_metadata)
    actor_names, frame_count, material_assignments = rgb_render.animate_objects(
        render_metadata, trajectory, materials, args.frame_limit
    )
    missing = [name for name in dynamic_names if name not in actor_names]
    if missing:
        raise RuntimeError(f"dynamic mask objects are absent from Blender scene: {missing}")

    dynamic_indices = {name: index + 1 for index, name in enumerate(dynamic_names)}
    for obj in scene.objects:
        obj.pass_index = dynamic_indices.get(obj.name, 0)
    for view_layer in scene.view_layers:
        view_layer.use_pass_object_index = True
        view_layer.use_pass_z = True
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.filepath = str(output_dir / "index_")
    index_output, depth_output = _configure_index_compositor(scene, output_dir)
    (output_dir / "_index_pass").mkdir(parents=True, exist_ok=True)
    (output_dir / "_depth_pass").mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    dynamic_masks = np.zeros((len(dynamic_names), frame_count, args.height, args.width), dtype=np.bool_)
    dynamic_centers = np.zeros((frame_count, len(dynamic_names), 3), dtype=np.float32)
    depth_frames = np.zeros((frame_count, args.height, args.width), dtype=np.float32)
    for frame_index in range(frame_count):
        scene.frame_set(frame_index + 1)
        bpy.ops.render.render(write_still=False)
        index_path = _index_pass_path(index_output, output_dir, frame_index + 1)
        depth_path = _index_pass_path(depth_output, output_dir, frame_index + 1)
        object_ids = _extract_index_file(index_path, args.width, args.height)
        depth_frames[frame_index] = _extract_depth_file(depth_path, args.width, args.height)
        index_path.unlink(missing_ok=True)
        depth_path.unlink(missing_ok=True)
        for object_index, name in enumerate(dynamic_names):
            dynamic_masks[object_index, frame_index] = object_ids == (object_index + 1)
        dynamic_centers[frame_index] = _pixel_centers(
            scene, camera, dynamic_names, args.width, args.height
        )
        if frame_index == 0 or (frame_index + 1) % 10 == 0 or frame_index + 1 == frame_count:
            print(f"[cycles-truth] {sample_dir.name} {frame_index + 1}/{frame_count}", flush=True)

    dynamic_union = np.any(dynamic_masks, axis=0)
    np.savez_compressed(
        output_dir / "dynamic_masks.npz",
        masks_thw=dynamic_masks,
        union_thw=dynamic_union,
        object_names=np.asarray(dynamic_names),
        object_indices=np.arange(1, len(dynamic_names) + 1, dtype=np.int32),
    )
    np.savez_compressed(
        output_dir / "trajectory_pixels.npz",
        centers_tnc=dynamic_centers,
        object_names=np.asarray(dynamic_names),
        convention=np.asarray("x_pixels_y_pixels_depth"),
    )
    camera_calibration = _camera_calibration(camera, args.width, args.height)
    np.savez_compressed(output_dir / "cycles_depth.npz", depth=depth_frames)
    _write_rigidbench_adapter(
        output_dir,
        sample_dir,
        metadata,
        dynamic_names,
        dynamic_masks,
        depth_frames,
        camera_calibration,
    )
    report = {
        "schema_version": "physv_cycles_aligned_truth_v1",
        "sample_id": metadata["sample_id"],
        "source_sample_dir": str(sample_dir),
        "source_rgb_cycles": str(sample_dir / "videos" / "rgb_cycles.mp4"),
        "source_trajectory": str(sample_dir / "raw" / "trajectories.npz"),
        "frame_count": int(frame_count),
        "fps": int(metadata["simulation"]["fps"]),
        "resolution": [int(args.width), int(args.height)],
        "engine": "CYCLES",
        "samples": int(args.samples),
        "exposure": float(args.exposure),
        "enabled_devices": enabled_devices,
        "camera": {
            "location": [float(value) for value in camera.location],
            "target": [float(value) for value in camera["target"]],
            "source_yfov_deg": float(camera["source_yfov_deg"]),
            "effective_yfov_deg": float(camera["effective_yfov_deg"]),
            "framing_profile": str(camera["framing_profile"]),
        },
        "dynamic_objects": dynamic_names,
        "object_index_mapping": dynamic_indices,
        "mask_file": "dynamic_masks.npz",
        "mask_shape": list(dynamic_masks.shape),
        "union_shape": list(dynamic_union.shape),
        "trajectory_file": "trajectory_pixels.npz",
        "depth_file": "cycles_depth.npz",
        "depth_shape": list(depth_frames.shape),
        "depth_source": "CYCLES compositor Depth/Z pass",
        "camera_calibration": camera_calibration,
        "rigidbench_adapter_dir": "rigidbench",
        "render_seconds": time.monotonic() - started,
        "alignment": {
            "rgb_renderer": "render_physv_cycles.py",
            "scene_builder": "same metadata/camera/material/trajectory path as rgb_cycles.mp4",
            "pixel_origin": "top-left for saved masks and trajectory_pixels",
            "frame_index": "zero-based; frame 0 corresponds to rgb_cycles.mp4 frame 0",
            "object_index_zero": "background and non-dynamic objects are 0; dynamic objects start at 1",
        },
        "room_scene": room_scene,
        "hdri": str(hdri_path),
        "lighting_preset": lighting_preset,
        "material_assignments": material_assignments,
    }
    (output_dir / "truth_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
