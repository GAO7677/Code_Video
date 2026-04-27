#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize which regions of a PhysXNet object are filled by voxel-based rigid collision.

This demo is intended for inspecting articulated objects such as cabinet 48610.
It renders:
1) original rigid-part surface meshes
2) voxel-filled collision meshes produced by the same logic used in
   physxnet_articulation_demo.py

Example:
CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/physxnet_voxel_fill_demo.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --object_id 48610 \
  --output_root /data/gaoya/AAA_test_video/Dataset_test/physxnet_voxel_fill_demo \
  --voxel_pitch 0.025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import trimesh

try:
    import gradio as gr
except Exception as e:  # pragma: no cover
    raise RuntimeError("This demo requires gradio. Please `pip install gradio`.") from e

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as e:  # pragma: no cover
    raise RuntimeError("This demo requires plotly. Please `pip install plotly`.") from e

from physxnet_articulation_demo import (
    color_from_part_id,
    parse_group_info,
    parse_dimension_to_meters,
    ensure_dir,
    load_mesh,
    merge_meshes,
    sanitize_mesh,
    voxel_fill_mesh_collision,
    yup_to_zup_mesh,
)

MAX_PC_POINTS_PER_PART = 12000
MAX_SURFACE_SAMPLE_POINTS_PER_PART = 4000


def rgb_to_plotly(c: Tuple[float, float, float, float], alpha: float) -> str:
    r = int(np.clip(round(c[0] * 255), 0, 255))
    g = int(np.clip(round(c[1] * 255), 0, 255))
    b = int(np.clip(round(c[2] * 255), 0, 255))
    return f"rgba({r},{g},{b},{alpha})"


def mesh3d_from_trimesh(mesh: trimesh.Trimesh, color: str, name: str, opacity: float) -> go.Mesh3d:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int32)
    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=True,
        hovertext=name,
        hoverinfo="text",
        showscale=False,
    )


def scatter3d_from_points(points: np.ndarray, color: str, name: str, size: float = 2.0) -> go.Scatter3d:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) > MAX_PC_POINTS_PER_PART:
        idx = np.linspace(0, len(pts) - 1, MAX_PC_POINTS_PER_PART).astype(np.int64)
        pts = pts[idx]
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.9),
        name=name,
        hovertext=name,
        hoverinfo="text",
        showlegend=False,
    )


def voxel_center_points(mesh: trimesh.Trimesh, pitch: float) -> np.ndarray:
    vox = mesh.voxelized(pitch)
    try:
        vox = vox.fill()
    except Exception:
        pass
    pts = np.asarray(vox.points, dtype=np.float64).reshape(-1, 3)
    return pts


def surface_sample_points(mesh: trimesh.Trimesh, num_points: int = MAX_SURFACE_SAMPLE_POINTS_PER_PART) -> np.ndarray:
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
        return np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    except Exception:
        pts = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        if len(pts) > num_points:
            idx = np.linspace(0, len(pts) - 1, num_points).astype(np.int64)
            pts = pts[idx]
        return pts


def load_and_scale_parts(physx_root: Path, version: str, object_id: str) -> Tuple[Dict[int, trimesh.Trimesh], Dict[str, Any]]:
    version_root = physx_root / version
    json_path = version_root / "finaljson" / f"{object_id}.json"
    objs_dir = version_root / "partseg" / object_id / "objs"

    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not objs_dir.exists():
        raise FileNotFoundError(objs_dir)

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    part_meshes: Dict[int, trimesh.Trimesh] = {}

    for part in meta["parts"]:
        pid = int(part["label"])
        mesh_path = objs_dir / f"{pid}.obj"
        if not mesh_path.exists():
            continue
        mesh = load_mesh(mesh_path)
        mesh = yup_to_zup_mesh(mesh)
        part_meshes[pid] = mesh

    if not part_meshes:
        raise ValueError(f"No valid part meshes found for object {object_id}")

    merged_raw = merge_meshes(list(part_meshes.values()))
    raw_extents = np.asarray(merged_raw.extents, dtype=np.float64)
    raw_center = np.asarray(merged_raw.bounding_box.centroid, dtype=np.float64)

    dim_m = parse_dimension_to_meters(str(meta.get("dimension", "")))
    if dim_m is not None:
        dim_m = dim_m[[0, 2, 1]]
        object_scale = float(np.max(dim_m) / max(np.max(raw_extents), 1e-8))
    else:
        object_scale = float(1.0 / max(np.max(raw_extents), 1e-8))

    for pid, mesh in list(part_meshes.items()):
        m = mesh.copy()
        m.apply_translation(-raw_center)
        m.apply_scale(object_scale)
        part_meshes[pid] = sanitize_mesh(m)

    return part_meshes, meta


def rigid_part_ids(meta: Dict[str, Any]) -> List[int]:
    parsed_groups = parse_group_info(meta.get("group_info", {}))
    movable_groups = parsed_groups.get("movable_groups", [])
    movable_child_labels = sorted({int(lbl) for g in movable_groups for lbl in g.child_labels})
    base_labels = sorted(int(x) for x in parsed_groups.get("base_group", []))
    all_labels = sorted(int(part["label"]) for part in meta.get("parts", []))
    base_labels = sorted(set(base_labels).union(set(all_labels) - set(movable_child_labels)))
    covered = set(base_labels)
    for g in movable_groups:
        for pid in g.child_labels:
            covered.add(int(pid))
    return sorted(covered)


def build_figure(
    part_meshes: Dict[int, trimesh.Trimesh],
    meta: Dict[str, Any],
    voxel_pitch: float,
    selected_part_id: Optional[int] = None,
) -> Tuple[go.Figure, List[Dict[str, Any]]]:
    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        subplot_titles=(
            "Before Fill: Surface Mesh",
            "After Fill: Collision Mesh",
            "Before Fill: Surface Point Cloud",
            "After Fill: Voxel Center Point Cloud",
        ),
    )

    stats: List[Dict[str, Any]] = []
    name_by_id = {int(part["label"]): str(part.get("name", f"part_{part['label']}")) for part in meta.get("parts", [])}

    for pid in rigid_part_ids(meta):
        if pid not in part_meshes:
            continue
        if selected_part_id is not None and pid != int(selected_part_id):
            continue
        mesh = part_meshes[pid]
        rgba = color_from_part_id(pid)
        color_surface = rgb_to_plotly(rgba, 0.35)
        color_voxel = rgb_to_plotly(rgba, 0.88)
        label = f"part {pid} | {name_by_id.get(pid, str(pid))}"

        fig.add_trace(mesh3d_from_trimesh(mesh, color_surface, label, opacity=0.35), row=1, col=1)

        collision_mesh, fill_meta = voxel_fill_mesh_collision(mesh, voxel_pitch)
        fig.add_trace(mesh3d_from_trimesh(collision_mesh, color_voxel, label, opacity=0.88), row=1, col=2)

        surface_points = surface_sample_points(mesh)
        fig.add_trace(scatter3d_from_points(surface_points, color_surface, label, size=1.8), row=2, col=1)

        points = voxel_center_points(mesh, voxel_pitch)
        fig.add_trace(scatter3d_from_points(points, color_voxel, label, size=2.2), row=2, col=2)

        stats.append(
            {
                "part_id": pid,
                "part_name": name_by_id.get(pid, str(pid)),
                "surface_num_points": int(len(surface_points)),
                "voxel_num_points": int(len(points)),
                "surface_extents": np.asarray(mesh.extents, dtype=np.float64).tolist(),
                "collision_extents": np.asarray(collision_mesh.extents, dtype=np.float64).tolist(),
                "collision_fill": fill_meta,
            }
        )

    for scene_name in ("scene", "scene2", "scene3", "scene4"):
        fig.update_layout(
            **{
                scene_name: dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                )
            }
        )

    fig.update_layout(
        title=f"PhysXNet Object {meta.get('object_name', '')} ({meta.get('category', 'Unknown')}) voxel fill demo",
        showlegend=False,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig, stats


def build_stats_preview(stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "part_id": rec["part_id"],
            "part_name": rec["part_name"],
            "surface_num_points": rec["surface_num_points"],
            "voxel_num_points": rec["voxel_num_points"],
            "fill_mode": rec["collision_fill"].get("fill_mode"),
            "num_vertices": rec["collision_fill"].get("num_vertices"),
            "num_faces": rec["collision_fill"].get("num_faces"),
        }
        for rec in stats
    ]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize voxel-filled collision regions for one PhysXNet articulated object.")
    parser.add_argument("--physx_root", type=str, required=True)
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--object_id", type=str, default="48610")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--voxel_pitch", type=float, default=0.025)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8012)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    part_meshes, meta = load_and_scale_parts(
        physx_root=Path(args.physx_root),
        version=args.version,
        object_id=str(args.object_id),
    )
    fig, stats = build_figure(part_meshes, meta, voxel_pitch=float(args.voxel_pitch))

    html_path = output_root / f"voxel_fill_{args.object_id}.html"
    json_path = output_root / f"voxel_fill_{args.object_id}.json"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    json_path.write_text(
        json.dumps(
            {
                "object_id": str(args.object_id),
                "object_name": meta.get("object_name", str(args.object_id)),
                "category": meta.get("category", "Unknown"),
                "voxel_pitch": float(args.voxel_pitch),
                "parts": stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] wrote {html_path}")
    print(f"[OK] wrote {json_path}")

    stats_preview = build_stats_preview(stats)
    part_choices = ["all"] + [str(rec["part_id"]) for rec in stats]

    def _update_view(part_choice: str):
        selected_part_id = None if part_choice == "all" else int(part_choice)
        filtered_fig, filtered_stats = build_figure(
            part_meshes,
            meta,
            voxel_pitch=float(args.voxel_pitch),
            selected_part_id=selected_part_id,
        )
        return filtered_fig, build_stats_preview(filtered_stats)

    with gr.Blocks(title=f"Voxel Fill Demo {args.object_id}") as demo:
        gr.Markdown(
            f"# PhysXNet Voxel Fill Demo\n\n"
            f"- object_id: `{args.object_id}`\n"
            f"- object_name: `{meta.get('object_name', str(args.object_id))}`\n"
            f"- category: `{meta.get('category', 'Unknown')}`\n"
            f"- voxel_pitch: `{float(args.voxel_pitch):.4f}` m\n"
            f"- html: `{html_path}`\n"
            f"- json: `{json_path}`"
        )
        part_selector = gr.Dropdown(
            choices=part_choices,
            value="all",
            label="Only show one part_id",
            info="Choose 'all' or a specific part_id such as cabinet frame or door.",
        )
        plot = gr.Plot(value=fig, label="Before/after fill comparison")
        stats_json = gr.JSON(value=stats_preview, label="Part fill summary")
        part_selector.change(_update_view, inputs=part_selector, outputs=[plot, stats_json])

    print(f"[INFO] launching viewer at http://{args.host}:{args.port}")
    demo.launch(server_name=args.host, server_port=int(args.port), share=False, inbrowser=False)


if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/physxnet_voxel_fill_demo.py \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --object_id 39264 \
  --output_root /data/gaoya/AAA_test_video/Dataset_test/physxnet_voxel_fill_demo \
  --voxel_pitch 0.025 \
  --host 0.0.0.0 \
  --port 8012
'''
