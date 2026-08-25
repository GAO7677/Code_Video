#!/usr/bin/env python3
"""Export CYCLES-coordinate object truth for an existing PhysV sample.

The RGB preview is built by ``render_physv_cycles.py`` from the same sample
metadata, camera, materials, and ``raw/trajectories.npz``.  This companion
script rebuilds that exact Blender scene and renders the Object Index pass at
the preview's native resolution.  It does not rerun PyBullet and it does not
modify the existing RGB preview or the original truth files.

The exported ``dynamic_masks.npz`` is therefore aligned to
``videos/rgb_cycles.mp4`` in frame order and pixel coordinates.  Collision and
state truth remain simulator-time labels and are copied by the batch wrapper;
they do not need a new pixel-space render.
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


def _extract_index_pass(width: int, height: int) -> np.ndarray:
    result = bpy.data.images.get("Render Result")
    if result is None:
        raise RuntimeError("Blender did not expose a Render Result image")
    if tuple(result.size) != (width, height):
        raise RuntimeError(f"Render Result size {tuple(result.size)} != {(width, height)}")
    if not result.layers:
        raise RuntimeError("Render Result contains no view layer")
    layer = result.layers[0]
    index_pass = next((item for item in layer.passes if item.name == "IndexOB"), None)
    if index_pass is None:
        available = [item.name for item in layer.passes]
        raise RuntimeError(f"Object Index pass is missing; available passes={available}")
    values = np.asarray(index_pass.rect, dtype=np.float32)
    if values.size != width * height * 4:
        raise RuntimeError(f"unexpected IndexOB buffer size {values.size}")
    return np.rint(values.reshape(height, width, 4)[..., 0]).astype(np.int32)


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
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.frame_start = 1
    scene.frame_end = frame_count
    scene.render.filepath = str(output_dir / "index_")
    started = time.monotonic()
    dynamic_masks = np.zeros((len(dynamic_names), frame_count, args.height, args.width), dtype=np.bool_)
    dynamic_centers = np.zeros((frame_count, len(dynamic_names), 3), dtype=np.float32)
    for frame_index in range(frame_count):
        scene.frame_set(frame_index + 1)
        bpy.ops.render.render(write_still=False)
        object_ids = _extract_index_pass(args.width, args.height)
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
