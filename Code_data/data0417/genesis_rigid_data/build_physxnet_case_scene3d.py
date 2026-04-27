#!/usr/bin/env python3
"""Build an interactive 3D mesh scene and a trajectory overview for one PhysXNet MPM sample."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import trimesh


WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class MeshBundle:
    label: str
    color: str
    opacity: float
    vertices: np.ndarray
    faces: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--max_mesh_faces", type=int, default=18000)
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
    forward = lookat - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm <= 1e-12:
        raise ValueError("Camera lookat equals camera position.")
    forward = forward / forward_norm

    right = np.cross(forward, WORLD_UP)
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


def load_mesh_from_path(mesh_path: Path, *, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {mesh_path}")
    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if faces.shape[0] > max_faces:
        step = int(math.ceil(faces.shape[0] / max_faces))
        faces = faces[::step]
    return vertices, faces


def concatenate_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices_list: list[np.ndarray] = []
    faces_list: list[np.ndarray] = []
    vertex_offset = 0
    for vertices, faces in meshes:
        vertices_list.append(vertices)
        faces_list.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]
    return np.vstack(vertices_list), np.vstack(faces_list)


def center_vertices(vertices: np.ndarray) -> np.ndarray:
    centroid = vertices.mean(axis=0, keepdims=True)
    return vertices - centroid


def resolve_object_mesh(
    obj_meta: dict,
    *,
    object_root: Path,
    asset_cache_root: Path,
    max_faces: int,
) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    source_id = str(obj_meta.get("source_object_id") or "")
    role = str(obj_meta.get("role") or "object")
    source_tag = str(obj_meta.get("source_tag") or "")

    if source_tag == "physxnet_main":
        mesh_paths = sorted((object_root / "rigid_visuals").glob("*.obj"))
        soft_paths = sorted((object_root / "soft").glob("*.obj"))
        mesh_pairs = [load_mesh_from_path(path, max_faces=max_faces) for path in (mesh_paths + soft_paths)]
        if not mesh_pairs:
            raise FileNotFoundError(f"No visual meshes found under {object_root}")
        vertices, faces = concatenate_meshes(mesh_pairs)
        return center_vertices(vertices), faces, "target", "#4c6ef5", 1.0

    if source_id == "yellow_striker_ball":
        mesh_path = asset_cache_root / "primitive" / "primitive__sphere__rubber.obj"
        vertices, faces = load_mesh_from_path(mesh_path, max_faces=max_faces)
        return center_vertices(vertices), faces, "yellow striker ball", "#f0b429", 0.16

    mesh_path = asset_cache_root / "physxnet" / f"physxnet__{source_id}.obj"
    if mesh_path.exists():
        vertices, faces = load_mesh_from_path(mesh_path, max_faces=max_faces)
        color = "#2f855a" if role == "bystander" else "#d97706"
        return center_vertices(vertices), faces, source_id, color, 1.0

    mesh_path = asset_cache_root / "primitive" / "primitive__sphere__foam.obj"
    vertices, faces = load_mesh_from_path(mesh_path, max_faces=max_faces)
    return center_vertices(vertices), faces, source_id or role, "#9c6644", 0.16


def estimate_visual_scale(
    base_vertices: np.ndarray,
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

    world_points = (base_vertices @ rotation.T) + translation.reshape(1, 3)
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
    scale = fallback * (obs_norm / proj_norm)
    return float(np.clip(scale, 0.03, 3.0))


def transform_vertices(vertices: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    return (vertices * scale) @ rotation.T + translation.reshape(1, 3)


def build_axes(axis_len: float) -> list[go.Scatter3d]:
    axis_specs = [
        ("X", "#d94841", np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float64)),
        ("Y", "#2b8a3e", np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float64)),
        ("Z", "#1c7ed6", np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float64)),
    ]
    traces: list[go.Scatter3d] = []
    for label, color, pts in axis_specs:
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines+text",
                text=["", label],
                textposition="top center",
                line={"color": color, "width": 7},
                name=f"axis {label}",
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def make_ground_plane(xmin: float, xmax: float, ymin: float, ymax: float) -> go.Surface:
    xs = np.array([[xmin, xmax], [xmin, xmax]], dtype=np.float64)
    ys = np.array([[ymin, ymin], [ymax, ymax]], dtype=np.float64)
    zs = np.zeros((2, 2), dtype=np.float64)
    return go.Surface(
        x=xs,
        y=ys,
        z=zs,
        colorscale=[[0.0, "#f4f1ea"], [1.0, "#efe6d7"]],
        showscale=False,
        opacity=0.65,
        hoverinfo="skip",
        name="ground",
    )


def make_mesh_trace(bundle: MeshBundle) -> go.Mesh3d:
    return go.Mesh3d(
        x=bundle.vertices[:, 0],
        y=bundle.vertices[:, 1],
        z=bundle.vertices[:, 2],
        i=bundle.faces[:, 0],
        j=bundle.faces[:, 1],
        k=bundle.faces[:, 2],
        color=bundle.color,
        opacity=bundle.opacity,
        name=bundle.label,
        flatshading=False,
        lighting={"ambient": 0.6, "diffuse": 0.9, "specular": 0.15, "roughness": 0.7},
        lightposition={"x": 3, "y": -2, "z": 6},
    )


def save_trajectory_overview(
    output_path: Path,
    *,
    scene_id: str,
    roles: list[str],
    labels: list[str],
    colors: list[str],
    trajectories: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), dpi=180)
    fig.patch.set_facecolor("#fffdf8")
    views = [
        (axes[0], 0, 1, "Top View (X-Y)", "X", "Y"),
        (axes[1], 0, 2, "Side View (X-Z)", "X", "Z"),
    ]
    for ax, dim_a, dim_b, title, label_a, label_b in views:
        ax.set_facecolor("#fffaf2")
        for idx, traj in enumerate(trajectories):
            ax.plot(traj[:, dim_a], traj[:, dim_b], color=colors[idx], linewidth=2.2, label=f"{roles[idx]} / {labels[idx]}")
            ax.scatter(traj[0, dim_a], traj[0, dim_b], color=colors[idx], s=42, marker="o")
            ax.scatter(traj[-1, dim_a], traj[-1, dim_b], color=colors[idx], s=54, marker="X")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(label_a)
        ax.set_ylabel(label_b)
        ax.grid(True, color="#ddd4c6", linewidth=0.8, alpha=0.9)
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_color("#d4c7b2")
    axes[1].legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", edgecolor="#d8ccb8")
    fig.suptitle(f"{scene_id} trajectory overview", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    rigid_npz = np.load(sample_dir / "physics" / "rigid_kinematics.npz")

    vis_dir = sample_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    scene_html = vis_dir / "scene_3d.html"
    trajectory_png = vis_dir / "trajectory_overview.png"

    object_id = str(metadata["object_id"])
    dataset_root = sample_dir.parents[4]
    object_root_candidates = [
        dataset_root / object_id,
        dataset_root / "_asset_cache" / "physxnet_objects" / object_id,
    ]
    asset_cache_candidates = [
        dataset_root / "_custom_object_asset_cache",
        dataset_root / "_asset_cache" / "custom_object_asset_cache",
    ]
    object_root = next((path for path in object_root_candidates if path.exists()), object_root_candidates[-1])
    asset_cache_root = next((path for path in asset_cache_candidates if path.exists()), asset_cache_candidates[-1])

    camera = metadata["camera"]
    intr = metadata["camera_intrinsics"]
    camera_pos = np.asarray(camera["pos"], dtype=np.float64)
    lookat = np.asarray(camera["lookat"], dtype=np.float64)
    world_to_camera = lookat_world_to_camera(camera_pos, lookat)
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])

    object_meta = list(metadata["objects"])
    com_pos = np.asarray(rigid_npz["com_pos"], dtype=np.float64)
    orientation_quat = np.asarray(rigid_npz["orientation_quat"], dtype=np.float64)
    bbox_xyxy = np.asarray(rigid_npz["bbox_xyxy"], dtype=np.float64)
    if com_pos.shape[1] != len(object_meta):
        raise ValueError(f"Object count mismatch: kinematics={com_pos.shape[1]} metadata={len(object_meta)}")

    figure = go.Figure()
    all_points = com_pos.reshape(-1, 3)
    axis_extent = float(np.max(np.ptp(all_points, axis=0)))
    axis_extent = max(axis_extent, 1.2)
    xy_margin = max(0.6, 0.25 * axis_extent)
    xmin = float(np.min(all_points[:, 0]) - xy_margin)
    xmax = float(np.max(all_points[:, 0]) + xy_margin)
    ymin = float(np.min(all_points[:, 1]) - xy_margin)
    ymax = float(np.max(all_points[:, 1]) + xy_margin)
    zmax = float(np.max(all_points[:, 2]) + 0.6)

    figure.add_trace(make_ground_plane(xmin, xmax, ymin, ymax))
    for axis_trace in build_axes(max(1.0, 0.9 * axis_extent)):
        figure.add_trace(axis_trace)

    role_labels: list[str] = []
    source_labels: list[str] = []
    palette: list[str] = []
    traj_stack: list[np.ndarray] = []

    for obj_idx, obj in enumerate(object_meta):
        base_vertices, faces, source_label, color, fallback_scale = resolve_object_mesh(
            obj,
            object_root=object_root,
            asset_cache_root=asset_cache_root,
            max_faces=args.max_mesh_faces,
        )
        rotation0 = quaternion_wxyz_to_matrix(orientation_quat[0, obj_idx])
        scale = estimate_visual_scale(
            base_vertices,
            rotation=rotation0,
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

        start_rot = quaternion_wxyz_to_matrix(orientation_quat[0, obj_idx])
        end_rot = quaternion_wxyz_to_matrix(orientation_quat[-1, obj_idx])
        mid_rot = quaternion_wxyz_to_matrix(orientation_quat[len(orientation_quat) // 2, obj_idx])
        start_vertices = transform_vertices(base_vertices, start_rot, com_pos[0, obj_idx], scale)
        end_vertices = transform_vertices(base_vertices, end_rot, com_pos[-1, obj_idx], scale)
        mid_vertices = transform_vertices(base_vertices, mid_rot, com_pos[len(com_pos) // 2, obj_idx], scale)

        role = str(obj.get("role") or f"object_{obj_idx}")
        source_id = str(obj.get("source_object_id") or source_label)
        label = f"{obj_idx}: {role} / {source_id}"
        figure.add_trace(make_mesh_trace(MeshBundle(label=f"{label} start", color=color, opacity=0.88, vertices=start_vertices, faces=faces)))
        if np.linalg.norm(com_pos[-1, obj_idx] - com_pos[0, obj_idx]) > 0.02:
            figure.add_trace(make_mesh_trace(MeshBundle(label=f"{label} mid", color=color, opacity=0.18, vertices=mid_vertices, faces=faces)))
        figure.add_trace(make_mesh_trace(MeshBundle(label=f"{label} end", color=color, opacity=0.10, vertices=end_vertices, faces=faces)))

        traj = com_pos[:, obj_idx]
        figure.add_trace(
            go.Scatter3d(
                x=traj[:, 0],
                y=traj[:, 1],
                z=traj[:, 2],
                mode="lines+markers",
                line={"color": color, "width": 5},
                marker={"color": color, "size": 2.2, "opacity": 0.88},
                name=label,
            )
        )
        figure.add_trace(
            go.Scatter3d(
                x=[traj[0, 0], traj[-1, 0]],
                y=[traj[0, 1], traj[-1, 1]],
                z=[traj[0, 2], traj[-1, 2]],
                mode="markers+text",
                marker={"color": color, "size": 7, "symbol": ["circle", "diamond"]},
                text=[f"{role} start", f"{role} end"],
                textposition="top center",
                name=f"{label} endpoints",
                showlegend=False,
            )
        )

        role_labels.append(role)
        source_labels.append(source_id)
        palette.append(color)
        traj_stack.append(traj)

    figure.add_trace(
        go.Scatter3d(
            x=[camera_pos[0]],
            y=[camera_pos[1]],
            z=[camera_pos[2]],
            mode="markers+text",
            marker={"color": "#111111", "size": 6},
            text=["camera"],
            textposition="top center",
            name="camera",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[camera_pos[0], lookat[0]],
            y=[camera_pos[1], lookat[1]],
            z=[camera_pos[2], lookat[2]],
            mode="lines",
            line={"color": "#111111", "width": 4, "dash": "dash"},
            name="camera ray",
        )
    )

    figure.update_layout(
        title={
            "text": f"{metadata['scene_id']} 3D mesh scene",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22},
        },
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#fbfaf7",
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        legend={
            "bgcolor": "rgba(255,255,255,0.92)",
            "bordercolor": "rgba(34,34,34,0.08)",
            "borderwidth": 1,
        },
        scene={
            "xaxis": {
                "title": "X",
                "range": [xmin, xmax],
                "backgroundcolor": "#f7f4ee",
                "gridcolor": "#ddd4c6",
                "showbackground": True,
            },
            "yaxis": {
                "title": "Y",
                "range": [ymin, ymax],
                "backgroundcolor": "#f7f4ee",
                "gridcolor": "#ddd4c6",
                "showbackground": True,
            },
            "zaxis": {
                "title": "Z",
                "range": [0.0, zmax],
                "backgroundcolor": "#fbfaf7",
                "gridcolor": "#e8e1d4",
                "showbackground": True,
            },
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.55, "y": -1.45, "z": 0.9},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
            "annotations": [
                {
                    "x": lookat[0],
                    "y": lookat[1],
                    "z": max(0.08, lookat[2]),
                    "text": metadata.get("interaction_pattern", ""),
                    "showarrow": False,
                    "font": {"size": 12, "color": "#5c4a38"},
                    "bgcolor": "rgba(255,255,255,0.75)",
                }
            ],
        },
    )

    figure.write_html(scene_html, include_plotlyjs=True, full_html=True)
    save_trajectory_overview(
        trajectory_png,
        scene_id=str(metadata["scene_id"]),
        roles=role_labels,
        labels=source_labels,
        colors=palette,
        trajectories=np.stack(traj_stack, axis=0),
    )
    print(f"[DONE] wrote {scene_html}")
    print(f"[DONE] wrote {trajectory_png}")


if __name__ == "__main__":
    main()
