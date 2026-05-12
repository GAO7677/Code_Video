#!/usr/bin/env python3
# 用途：导出刚体初始场景为 HTML 3D 可视化。
"""Export an initialized rigid scene as GLB plus a local interactive HTML viewer.

The output focuses on the first frame of one rigid sample:
- every object's mesh is placed into the world at its initial pose
- a ground plane is added for spatial context
- the scene is saved as both ``scene_init.glb`` and a self-contained-ish HTML
  viewer that embeds the GLB as a data URI
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import trimesh


MODEL_VIEWER_URL = "https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--max_faces", type=int, default=30000)
    return parser.parse_args()


def quaternion_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def lookat_world_to_camera(camera_pos: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = lookat - camera_pos
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm <= 1e-12:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm
    up = np.cross(right, forward)
    up = up / max(np.linalg.norm(up), 1e-12)
    return np.stack([right, -up, forward], axis=0)


def project_points(
    world_points: np.ndarray,
    *,
    world_to_camera: np.ndarray,
    camera_pos: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    cam_points = (world_to_camera @ (world_points - camera_pos).T).T
    valid = cam_points[:, 2] > 1e-6
    uv = np.full((world_points.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        uv_valid = np.empty((int(valid.sum()), 2), dtype=np.float64)
        uv_valid[:, 0] = fx * (cam_points[valid, 0] / cam_points[valid, 2]) + cx
        uv_valid[:, 1] = fy * (cam_points[valid, 1] / cam_points[valid, 2]) + cy
        uv[valid] = uv_valid
    return uv, valid


def load_mesh(mesh_path: Path, *, max_faces: int) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g.copy() for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type at {mesh_path}")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > max_faces:
        step = int(math.ceil(len(mesh.faces) / max_faces))
        mesh.update_faces(np.arange(len(mesh.faces))[::step])
        mesh.remove_unreferenced_vertices()
    return mesh


def inertial_origin(mesh: trimesh.Trimesh) -> np.ndarray:
    center = None
    try:
        if bool(getattr(mesh, "is_watertight", False)) and bool(getattr(mesh, "is_volume", False)):
            candidate = np.asarray(mesh.center_mass, dtype=np.float64)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            if np.all(np.isfinite(candidate)) and bounds.shape == (2, 3):
                eps = np.maximum(1e-6, 1e-3 * np.maximum(bounds[1] - bounds[0], 1e-6))
                if np.all(candidate >= (bounds[0] - eps)) and np.all(candidate <= (bounds[1] + eps)):
                    center = candidate
    except Exception:
        center = None
    if center is None:
        center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    return center


def estimate_visual_scale(
    vertices_com_aligned: np.ndarray,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    target_bbox_xyxy: np.ndarray,
    world_to_camera: np.ndarray,
    camera_pos: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    fallback: float,
) -> float:
    bbox = np.asarray(target_bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        return fallback
    obs_wh = bbox[2:4] - bbox[0:2]
    if np.any(obs_wh <= 1.0):
        return fallback

    world_points = vertices_com_aligned @ rotation.T + translation.reshape(1, 3)
    uv, valid = project_points(
        world_points,
        world_to_camera=world_to_camera,
        camera_pos=camera_pos,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    if not np.any(valid):
        return fallback

    proj = uv[valid]
    proj_wh = proj.max(axis=0) - proj.min(axis=0)
    proj_norm = float(np.linalg.norm(proj_wh))
    obs_norm = float(np.linalg.norm(obs_wh))
    if proj_norm <= 1e-6 or not np.isfinite(proj_norm):
        return fallback
    return float(np.clip(fallback * (obs_norm / proj_norm), 0.03, 3.0))


def mesh_color_for_role(role: str) -> np.ndarray:
    role = str(role)
    if role == "target":
        return np.array([76, 110, 245, 255], dtype=np.uint8)
    if role == "initiator":
        return np.array([240, 180, 41, 210], dtype=np.uint8)
    if role == "bystander":
        return np.array([47, 133, 90, 255], dtype=np.uint8)
    return np.array([156, 102, 68, 255], dtype=np.uint8)


def resolve_mesh_path(sample_root: Path, obj_meta: dict) -> tuple[Path, float]:
    source_id = str(obj_meta.get("source_object_id") or "")
    cache_root = sample_root / "_asset_cache" / "custom_object_asset_cache"
    physx_cache = cache_root / "physxnet"
    primitive_cache = cache_root / "primitive"
    physx_root = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/_merged_for_genesis")

    if source_id == "yellow_striker_ball":
        return primitive_cache / "primitive__sphere__rubber.obj", 0.16

    cached = physx_cache / f"physxnet__{source_id}.obj"
    if cached.exists():
        return cached, 1.0

    merged = physx_root / source_id / "merged.obj"
    if merged.exists():
        return merged, 1.0

    fallback = primitive_cache / "primitive__sphere__foam.obj"
    return fallback, 0.16


def make_ground(bounds: np.ndarray) -> trimesh.Trimesh:
    xmin, ymin, zmin = bounds[0]
    xmax, ymax, _ = bounds[1]
    center = np.array([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5, max(-0.015, zmin - 0.015)], dtype=np.float64)
    extents = np.array([max(1.8, xmax - xmin + 0.8), max(1.8, ymax - ymin + 0.8), 0.03], dtype=np.float64)
    ground = trimesh.creation.box(extents=extents)
    ground.apply_translation(center)
    ground.visual.face_colors = np.tile(np.array([235, 228, 214, 255], dtype=np.uint8), (len(ground.faces), 1))
    return ground


def write_model_viewer_bundle(target_dir: Path) -> str:
    script_path = target_dir / "model-viewer.min.js"
    if not script_path.exists():
        try:
            with urllib.request.urlopen(MODEL_VIEWER_URL, timeout=20) as resp:
                script_path.write_bytes(resp.read())
        except Exception:
            return MODEL_VIEWER_URL
    return script_path.name


def build_html(
    *,
    title: str,
    subtitle: str,
    model_viewer_src: str,
    glb_data_uri: str,
    scene_info: list[dict],
) -> str:
    rows = "\n".join(
        f"<tr><td>{item['index']}</td><td>{item['role']}</td><td>{item['source_object_id']}</td><td>{item['mesh_name']}</td></tr>"
        for item in scene_info
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script type="module" src="{model_viewer_src}"></script>
  <style>
    :root {{
      --bg: #f6f1ea;
      --panel: rgba(255, 250, 243, 0.94);
      --ink: #201911;
      --muted: #6f5f4f;
      --line: #dccdbd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(76, 110, 245, 0.10), transparent 26%),
        radial-gradient(circle at top right, rgba(240, 180, 41, 0.12), transparent 24%),
        linear-gradient(180deg, #fbf8f3 0%, var(--bg) 100%);
    }}
    .page {{
      width: min(1400px, calc(100vw - 24px));
      margin: 0 auto;
      padding: 18px 0 26px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(61, 44, 23, 0.10);
      overflow: hidden;
    }}
    .head {{
      padding: 18px 20px 10px;
      border-bottom: 1px solid rgba(220, 205, 189, 0.85);
    }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .viewer {{
      height: min(78vh, 920px);
      width: 100%;
      background: linear-gradient(180deg, #faf6ef 0%, #f1e7d7 100%);
    }}
    .meta {{
      padding: 14px 20px 18px;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr);
      gap: 18px;
    }}
    .meta h2 {{
      margin: 0 0 8px;
      font-size: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid rgba(220, 205, 189, 0.75);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      background: rgba(32, 25, 17, 0.05);
      padding: 2px 6px;
      border-radius: 6px;
    }}
    @media (max-width: 900px) {{
      .meta {{ grid-template-columns: 1fr; }}
      .viewer {{ height: 64vh; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="card">
      <div class="head">
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      <model-viewer class="viewer" src="{glb_data_uri}" camera-controls touch-action="pan-y" shadow-intensity="1" exposure="1.0" auto-rotate ar="false"></model-viewer>
      <div class="meta">
        <div>
          <h2>交互说明</h2>
          <p>鼠标左键旋转，滚轮缩放，右键或双指平移。场景展示的是第 0 帧初始化状态，所有 mesh 都已经按初始位姿放到场景里。</p>
          <p style="margin-top:10px;">如果你的浏览器限制本地模块脚本，可以把这个 HTML 和同目录的 <code>model-viewer.min.js</code> 一起拖进浏览器，或者在 IDE 里用内置预览打开。</p>
        </div>
        <div>
          <h2>对象列表</h2>
          <table>
            <thead><tr><th>#</th><th>role</th><th>source id</th><th>mesh</th></tr></thead>
            <tbody>
              {rows}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    rigid_npz = np.load(sample_dir / "physics" / "rigid_kinematics.npz")

    vis_dir = sample_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    glb_path = vis_dir / "scene_init.glb"
    html_path = vis_dir / "scene_init_interactive.html"
    info_path = vis_dir / "scene_init_info.json"

    sample_root = sample_dir.parents[4]
    camera = metadata["camera"]
    intr = metadata["camera_intrinsics"]
    camera_pos = np.asarray(camera["pos"], dtype=np.float64)
    lookat = np.asarray(camera["lookat"], dtype=np.float64)
    world_to_camera = lookat_world_to_camera(camera_pos, lookat)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    com_pos = np.asarray(rigid_npz["com_pos"], dtype=np.float64)
    orientation_quat = np.asarray(rigid_npz["orientation_quat"], dtype=np.float64)
    bbox_xyxy = np.asarray(rigid_npz["bbox_xyxy"], dtype=np.float64)
    object_meta = list(metadata["objects"])

    scene = trimesh.Scene()
    scene_bounds_points: list[np.ndarray] = []
    info_records: list[dict] = []

    for obj_idx, obj in enumerate(object_meta):
        mesh_path, fallback_scale = resolve_mesh_path(sample_root, obj)
        mesh = load_mesh(mesh_path, max_faces=args.max_faces)
        offset = inertial_origin(mesh)
        verts_com_aligned = np.asarray(mesh.vertices, dtype=np.float64) - offset.reshape(1, 3)

        rotation = quaternion_wxyz_to_matrix(orientation_quat[0, obj_idx])
        scale = estimate_visual_scale(
            verts_com_aligned,
            rotation=rotation,
            translation=com_pos[0, obj_idx],
            target_bbox_xyxy=bbox_xyxy[0, obj_idx],
            world_to_camera=world_to_camera,
            camera_pos=camera_pos,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            fallback=fallback_scale,
        )

        transformed = verts_com_aligned * scale
        transformed = transformed @ rotation.T + com_pos[0, obj_idx].reshape(1, 3)

        placed = trimesh.Trimesh(vertices=transformed, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
        color = mesh_color_for_role(str(obj.get("role") or "object"))
        placed.visual.face_colors = np.tile(color, (len(placed.faces), 1))
        geom_name = f"obj{obj_idx}_{obj.get('role', 'object')}_{obj.get('source_object_id', '')}"
        scene.add_geometry(placed, geom_name=geom_name)
        scene_bounds_points.append(placed.vertices)

        info_records.append(
            {
                "index": int(obj_idx),
                "role": str(obj.get("role") or ""),
                "source_object_id": str(obj.get("source_object_id") or ""),
                "mesh_name": str(mesh_path.name),
                "mesh_path": str(mesh_path),
                "fallback_scale": float(fallback_scale),
                "estimated_scale": float(scale),
                "inertial_origin_xyz": offset.tolist(),
                "initial_com_pos_xyz": com_pos[0, obj_idx].tolist(),
                "initial_quaternion_wxyz": orientation_quat[0, obj_idx].tolist(),
            }
        )

    if scene_bounds_points:
        all_points = np.vstack(scene_bounds_points)
        ground = make_ground(np.stack([all_points.min(axis=0), all_points.max(axis=0)], axis=0))
        scene.add_geometry(ground, geom_name="ground")

    scene.export(glb_path)
    glb_b64 = base64.b64encode(glb_path.read_bytes()).decode("ascii")
    glb_data_uri = f"data:model/gltf-binary;base64,{glb_b64}"

    model_viewer_src = write_model_viewer_bundle(vis_dir)
    subtitle = (
        f"{metadata['scene_id']} | initial frame mesh scene | "
        f"target={metadata['object_id']} | objects={len(object_meta)}"
    )
    html = build_html(
        title=f"{metadata['scene_id']} Init 3D Scene",
        subtitle=subtitle,
        model_viewer_src=model_viewer_src,
        glb_data_uri=glb_data_uri,
        scene_info=info_records,
    )
    html_path.write_text(html, encoding="utf-8")
    info_path.write_text(
        json.dumps(
            {
                "scene_id": metadata["scene_id"],
                "sample_dir": str(sample_dir),
                "camera": metadata["camera"],
                "camera_intrinsics": metadata["camera_intrinsics"],
                "objects": info_records,
                "outputs": {
                    "scene_init_glb": str(glb_path),
                    "scene_init_html": str(html_path),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] wrote {glb_path}")
    print(f"[DONE] wrote {html_path}")
    print(f"[DONE] wrote {info_path}")


if __name__ == "__main__":
    main()
