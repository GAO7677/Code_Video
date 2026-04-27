#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def build_raw_bowl_mesh() -> trimesh.Trimesh:
    profile = np.asarray(
        [
            [0.082, 0.078],
            [0.062, 0.054],
            [0.040, 0.024],
            [0.018, 0.017],
            [0.000, 0.000],
            [0.056, 0.000],
            [0.074, 0.020],
            [0.094, 0.047],
            [0.108, 0.078],
        ],
        dtype=np.float64,
    )
    mesh = trimesh.creation.revolve(profile, sections=72)
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    return mesh


def _set_equal_axes(ax, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def render_mesh_preview(mesh: trimesh.Trimesh, output_png: Path) -> None:
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure(figsize=(12, 6), dpi=180)
    views = [(24, 35), (18, 125)]
    titles = ["View A", "View B"]

    for idx, ((elev, azim), title) in enumerate(zip(views, titles), start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            color="#c7a47d",
            edgecolor="#444444",
            linewidth=0.18,
            alpha=0.95,
            shade=True,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        _set_equal_axes(ax, vertices)

    fig.suptitle("Raw Bowl Mesh Preview (No Voxel Fill)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path("/home/gaoya/Code_Video/Code_data/demo_outputs/raw_bowl_mesh")
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = build_raw_bowl_mesh()
    obj_path = out_dir / "raw_bowl_mesh.obj"
    png_path = out_dir / "raw_bowl_mesh_preview.png"
    mesh.export(obj_path)
    render_mesh_preview(mesh, png_path)

    print(obj_path)
    print(png_path)
    print(f"faces={len(mesh.faces)} verts={len(mesh.vertices)} watertight={mesh.is_watertight}")


if __name__ == "__main__":
    main()
