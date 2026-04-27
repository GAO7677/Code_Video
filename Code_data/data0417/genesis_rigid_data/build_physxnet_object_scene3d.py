#!/usr/bin/env python3
"""Build an interactive 3D scene directly from one re-exported PhysXNet object folder."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object_dir", type=str, required=True, help="Re-exported object directory, e.g. .../_asset_cache/physxnet_objects/19925")
    parser.add_argument("--max_mesh_faces", type=int, default=24000)
    return parser.parse_args()


def load_mesh(mesh_path: Path, *, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)))
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


def concat_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    offset = 0
    for vertices, faces in meshes:
        vertices_all.append(vertices)
        faces_all.append(faces + offset)
        offset += vertices.shape[0]
    return np.vstack(vertices_all), np.vstack(faces_all)


def mesh_trace(
    *,
    name: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: str,
    opacity: float,
) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=False,
        lighting={"ambient": 0.6, "diffuse": 0.9, "specular": 0.15, "roughness": 0.7},
        lightposition={"x": 2.5, "y": -2.0, "z": 5.5},
    )


def line_trace(points: np.ndarray, *, name: str, color: str, width: int = 7) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line={"color": color, "width": width},
        name=name,
        hoverinfo="skip",
        showlegend=False,
    )


def ground_plane(xmin: float, xmax: float, ymin: float, ymax: float, z: float) -> go.Surface:
    xs = np.array([[xmin, xmax], [xmin, xmax]], dtype=np.float64)
    ys = np.array([[ymin, ymin], [ymax, ymax]], dtype=np.float64)
    zs = np.full((2, 2), z, dtype=np.float64)
    return go.Surface(
        x=xs,
        y=ys,
        z=zs,
        colorscale=[[0.0, "#f4f1ea"], [1.0, "#efe6d7"]],
        showscale=False,
        opacity=0.72,
        hoverinfo="skip",
        name="ground",
    )


def main() -> None:
    args = parse_args()
    object_dir = Path(args.object_dir).resolve()
    metadata_path = object_dir / "meta" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scene_preview_dir = object_dir / "scene_preview"
    scene_preview_dir.mkdir(parents=True, exist_ok=True)
    output_html = scene_preview_dir / "object_scene_3d.html"

    rigid_mesh_paths = sorted((object_dir / "rigid_visuals").glob("*.obj"))
    soft_mesh_paths = sorted((object_dir / "soft").glob("*.obj"))
    runtime_soft_mesh_paths = sorted((scene_preview_dir / "runtime_soft_meshes").glob("*.obj"))

    if not rigid_mesh_paths and not soft_mesh_paths:
        raise FileNotFoundError(f"No meshes found under {object_dir}")

    figure = go.Figure()
    bounds_min = []
    bounds_max = []

    if rigid_mesh_paths:
        rigid_meshes = [load_mesh(path, max_faces=args.max_mesh_faces) for path in rigid_mesh_paths]
        rigid_vertices, rigid_faces = concat_meshes(rigid_meshes)
        bounds_min.append(rigid_vertices.min(axis=0))
        bounds_max.append(rigid_vertices.max(axis=0))
        figure.add_trace(
            mesh_trace(
                name="rigid visuals",
                vertices=rigid_vertices,
                faces=rigid_faces,
                color="#5465ff",
                opacity=0.90,
            )
        )

    if soft_mesh_paths:
        soft_meshes = [load_mesh(path, max_faces=args.max_mesh_faces) for path in soft_mesh_paths]
        soft_vertices, soft_faces = concat_meshes(soft_meshes)
        bounds_min.append(soft_vertices.min(axis=0))
        bounds_max.append(soft_vertices.max(axis=0))
        figure.add_trace(
            mesh_trace(
                name="original soft / cloth",
                vertices=soft_vertices,
                faces=soft_faces,
                color="#f76707",
                opacity=0.42,
            )
        )

    if runtime_soft_mesh_paths:
        runtime_meshes = [load_mesh(path, max_faces=args.max_mesh_faces) for path in runtime_soft_mesh_paths]
        runtime_vertices, runtime_faces = concat_meshes(runtime_meshes)
        bounds_min.append(runtime_vertices.min(axis=0))
        bounds_max.append(runtime_vertices.max(axis=0))
        figure.add_trace(
            mesh_trace(
                name="runtime cloth mesh",
                vertices=runtime_vertices,
                faces=runtime_faces,
                color="#2b8a3e",
                opacity=0.55,
            )
        )

    all_min = np.min(np.stack(bounds_min, axis=0), axis=0)
    all_max = np.max(np.stack(bounds_max, axis=0), axis=0)
    center = 0.5 * (all_min + all_max)
    size = np.maximum(all_max - all_min, 1e-6)
    radius = float(max(size.max(), 0.6))
    margin = 0.24 * radius

    xmin = float(all_min[0] - margin)
    xmax = float(all_max[0] + margin)
    ymin = float(all_min[1] - margin)
    ymax = float(all_max[1] + margin)
    zmin = float(min(0.0, all_min[2] - 0.10 * radius))
    zmax = float(all_max[2] + margin)

    figure.add_trace(ground_plane(xmin, xmax, ymin, ymax, z=0.0))
    axis_len = max(0.6, 0.75 * radius)
    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    figure.add_trace(line_trace(np.vstack([origin, [[axis_len, 0.0, 0.0]]]), name="x", color="#d94841"))
    figure.add_trace(line_trace(np.vstack([origin, [[0.0, axis_len, 0.0]]]), name="y", color="#2b8a3e"))
    figure.add_trace(line_trace(np.vstack([origin, [[0.0, 0.0, axis_len]]]), name="z", color="#1c7ed6"))

    soft_parts = list(metadata.get("soft_parts", []) or [])
    part_labels = []
    for part in soft_parts:
        part_labels.append(
            f"pid={part.get('part_id')} {part.get('part_name')} / {part.get('material_ctor')}"
        )

    annotation_lines = [
        f"object_id={metadata.get('object_id', object_dir.name)}",
        f"simulator_mode={metadata.get('simulator_mode', 'unknown')}",
        f"soft_parts={len(soft_parts)}",
    ]
    if part_labels:
        annotation_lines.extend(part_labels[:4])

    figure.update_layout(
        title={
            "text": f"PhysXNet object {metadata.get('object_id', object_dir.name)} 3D scene",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22},
        },
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#fbfaf7",
        margin={"l": 0, "r": 0, "t": 54, "b": 0},
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
                "range": [zmin, zmax],
                "backgroundcolor": "#fbfaf7",
                "gridcolor": "#e8e1d4",
                "showbackground": True,
            },
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.45, "y": -1.35, "z": 0.95},
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            },
            "annotations": [
                {
                    "x": center[0],
                    "y": center[1],
                    "z": zmax,
                    "text": "<br>".join(annotation_lines),
                    "showarrow": False,
                    "font": {"size": 12, "color": "#5c4a38"},
                    "bgcolor": "rgba(255,255,255,0.78)",
                }
            ],
        },
    )

    figure.write_html(output_html, include_plotlyjs=True, full_html=True)
    print(f"[DONE] wrote {output_html}")


if __name__ == "__main__":
    main()
