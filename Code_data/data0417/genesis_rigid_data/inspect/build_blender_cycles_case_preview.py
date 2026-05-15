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
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import trimesh
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
BLENDER_DRIVER = SCRIPT_DIR / "blender_cycles_case_driver.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=24)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
        mass = float(part.get("mass_kg", 1.0) or 1.0)
        weighted_centers.append(local_center * mass)
        weighted_masses.append(mass)
        parts.append(
            {
                "mesh_path": str(mesh_path),
                "density_kgm3": None if part.get("density_kgm3") is None else float(part["density_kgm3"]),
            }
        )

    total_mass = float(np.sum(weighted_masses))
    if total_mass <= 1e-8:
        com_local = np.zeros(3, dtype=np.float64)
    else:
        com_local = np.sum(np.stack(weighted_centers, axis=0), axis=0) / total_mass

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
        "parts": [
            {
                "mesh_path": part["mesh_path"],
                "local_offset": (-com_local).tolist(),
                "material": density_material_spec(part["density_kgm3"], str(object_meta.get("role", "object"))),
            }
            for part in parts
        ],
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
) -> dict[str, Any]:
    radius = estimate_sphere_radius(
        bbox_xyxy=bbox_xyxy,
        center_depth=center_depth,
        visibility=visibility,
        fx=float(camera_intrinsics["fx"]),
        fy=float(camera_intrinsics["fy"]),
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
        "material": density_material_spec(None, str(object_meta.get("role", "initiator"))),
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
    sample_dir = args.sample_dir.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sample_meta = load_meta(sample_dir)
    dataset_root = find_dataset_root(sample_dir)
    kinematics = np.load(sample_dir / "physics" / "rigid_kinematics.npz", allow_pickle=True)
    anchor = np.load(sample_dir / "physics" / "anchor_targets.npz", allow_pickle=True)

    object_ids = np.asarray(kinematics["object_ids"], dtype=np.int32)
    com_pos = np.asarray(kinematics["com_pos"], dtype=np.float64)
    quats = np.asarray(kinematics["orientation_quat"], dtype=np.float64)
    bbox_xyxy = np.asarray(anchor["bbox_xyxy"], dtype=np.float64)
    center_depth = np.asarray(anchor["center_depth"], dtype=np.float64)
    visibility = np.asarray(anchor["visibility_mask"], dtype=np.uint8)

    frame_indices = sample_frame_indices(com_pos.shape[0], args.frame_stride, args.max_frames)
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
            spec = build_sphere_object_spec(
                object_meta=obj_meta,
                positions=sampled_pos,
                quats=sampled_quat,
                bbox_xyxy=bbox_xyxy[:, local_idx],
                center_depth=center_depth[:, local_idx],
                visibility=visibility[:, local_idx],
                camera_intrinsics=sample_meta["camera_intrinsics"],
            )
        else:
            spec = build_mesh_object_spec(
                sample_meta=sample_meta,
                dataset_root=dataset_root,
                object_meta=obj_meta,
                positions=sampled_pos,
                quats=sampled_quat,
            )
        for timeline_frame, item in enumerate(spec["frames"], start=1):
            item["timeline_frame"] = timeline_frame
        object_specs.append(spec)

    if not object_specs:
        raise RuntimeError(f"No renderable objects resolved from {sample_dir}")

    stacked_pos = np.concatenate(all_positions, axis=0)
    xy_min = np.min(stacked_pos[:, :2], axis=0)
    xy_max = np.max(stacked_pos[:, :2], axis=0)
    xy_center = 0.5 * (xy_min + xy_max)
    xy_extent = np.maximum(xy_max - xy_min, 1.6) + 1.2

    spec = {
        "sample_name": sample_dir.name,
        "sample_dir": str(sample_dir),
        "output_root": str(output_root),
        "frame_stride": int(args.frame_stride),
        "sampled_frame_indices": frame_indices,
        "total_frames": int(com_pos.shape[0]),
        "render": {
            "width": int(args.width),
            "height": int(args.height),
            "samples": int(args.samples),
            "fps": int(args.fps),
        },
        "camera": {
            "position": [float(v) for v in sample_meta["camera"]["pos"]],
            "lookat": [float(v) for v in sample_meta["camera"]["lookat"]],
            "up": [float(v) for v in sample_meta["camera"].get("up", [0.0, 0.0, 1.0])],
            "fov_deg": float(sample_meta["camera"]["fov"]),
        },
        "timeline": {
            "frame_start": 1,
            "frame_end": len(frame_indices),
        },
        "ground": {
            "center": [float(xy_center[0]), float(xy_center[1]), 0.0],
            "extents_xy": [float(xy_extent[0]), float(xy_extent[1])],
            "material": {
                "base_color": [0.90, 0.88, 0.84, 1.0],
                "roughness": 0.82,
                "specular": 0.12,
                "metallic": 0.0,
                "clearcoat": 0.0,
            },
        },
        "lighting": {
            "key_area": {
                "location": [float(xy_center[0] + 0.9), float(xy_center[1] - 2.2), 2.6],
                "rotation_euler_deg": [58.0, 0.0, 20.0],
                "energy": 2400.0,
                "size": 2.0,
            },
            "fill_area": {
                "location": [float(xy_center[0] - 1.2), float(xy_center[1] + 1.6), 1.8],
                "rotation_euler_deg": [78.0, 0.0, -32.0],
                "energy": 900.0,
                "size": 1.6,
            },
        },
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
    video_path = output_root / "cycles_preview.mp4"
    gif_path = output_root / "cycles_preview.gif"
    if video_path.exists():
        make_gif_from_video(video_path, gif_path)
    build_html(output_root, spec)
    print(f"[DONE] preview page: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
