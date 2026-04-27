#!/usr/bin/env python3
"""
data/文件夹存储渲染图像和相机参数，
simulation_data/文件夹存储网格和材料参数，
generated_cache/文件夹则用于存储生成的结果。
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import json
import base64
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output


def load_mesh(mesh_path: Path):
    obj = trimesh.load(mesh_path, process=False)

    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise ValueError(f"Unsupported mesh type: {type(obj)}")

    return mesh


def load_point_cloud(ply_path: Path):
    obj = trimesh.load(ply_path, process=False)

    if hasattr(obj, "vertices"):
        pts = np.asarray(obj.vertices)
    elif hasattr(obj, "points"):
        pts = np.asarray(obj.points)
    else:
        raise ValueError(f"Unsupported point cloud type: {type(obj)}")

    return pts


def read_image_size(img_path: Path, K: np.ndarray):
    if img_path.exists():
        with Image.open(img_path) as im:
            return im.size  # (W, H)

    # 回退：由 principal point 猜测图像尺寸
    W = int(round(K[0, 2] * 2))
    H = int(round(K[1, 2] * 2))
    return W, H


def image_to_data_url(img_path: Path, max_hw=900):
    if not img_path.exists():
        return None

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_hw, max_hw))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


def make_frustum_lines(c2w: np.ndarray, K: np.ndarray, W: int, H: int, depth: float):
    """
    使用 pinhole 模型在相机坐标系中构造 frustum，再通过 c2w 变换到世界坐标。
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    z = depth
    corners_cam = np.array([
        [(0 - cx) / fx * z, (0 - cy) / fy * z, z],
        [(W - cx) / fx * z, (0 - cy) / fy * z, z],
        [(W - cx) / fx * z, (H - cy) / fy * z, z],
        [(0 - cx) / fx * z, (H - cy) / fy * z, z],
    ], dtype=np.float64)

    origin_cam = np.zeros((1, 3), dtype=np.float64)
    frustum_cam = np.vstack([origin_cam, corners_cam])  # [5, 3]

    R = c2w[:3, :3]
    t = c2w[:3, 3]
    frustum_world = (R @ frustum_cam.T).T + t[None, :]

    # 连线：光心到四角 + 四角框
    segs = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1),
    ]

    xs, ys, zs = [], [], []
    for a, b in segs:
        xs += [frustum_world[a, 0], frustum_world[b, 0], None]
        ys += [frustum_world[a, 1], frustum_world[b, 1], None]
        zs += [frustum_world[a, 2], frustum_world[b, 2], None]

    center = frustum_world[0]
    return xs, ys, zs, center


def collect_cameras(static_json_path: Path):
    root = static_json_path.parent
    with open(static_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    cams = []
    for idx, rec in enumerate(records):
        c2w = np.array(rec["c2w"], dtype=np.float64)
        K = np.array(rec["intrinsic"], dtype=np.float64)

        img_rel = rec["file_path"]
        img_path = (root / img_rel).resolve()
        W, H = read_image_size(img_path, K)

        cams.append({
            "idx": idx,
            "raw": rec,
            "c2w": c2w,
            "K": K,
            "W": W,
            "H": H,
            "img_path": img_path,
            "name": f"cam_{idx}",
            "center": c2w[:3, 3].copy(),
        })

    return cams


def compute_scene_stats(mesh, pts, cams):
    arrs = []

    if mesh is not None:
        arrs.append(np.asarray(mesh.vertices))
    if pts is not None and len(pts) > 0:
        arrs.append(np.asarray(pts))
    if len(cams) > 0:
        cam_centers = np.stack([c["center"] for c in cams], axis=0)
        arrs.append(cam_centers)

    if len(arrs) == 0:
        center = np.zeros(3)
        span = 1.0
        return center, span

    allp = np.concatenate(arrs, axis=0)
    mn = allp.min(axis=0)
    mx = allp.max(axis=0)
    center = 0.5 * (mn + mx)
    span = float(np.max(mx - mn))
    span = max(span, 1e-3)
    return center, span


def build_figure(mesh, pts, cams, selected_cam_idx, show_mesh, show_points, show_cams,
                 pc_stride, point_size):
    fig = go.Figure()

    center, span = compute_scene_stats(mesh, pts, cams)
    frustum_depth = span * 0.08

    # mesh
    if show_mesh and mesh is not None:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces)

        fig.add_trace(go.Mesh3d(
            x=v[:, 0], y=v[:, 1], z=v[:, 2],
            i=f[:, 0], j=f[:, 1], k=f[:, 2],
            opacity=0.35,
            name="mesh",
        ))

    # point cloud
    if show_points and pts is not None and len(pts) > 0:
        pts_show = pts[::max(1, pc_stride)]

        fig.add_trace(go.Scatter3d(
            x=pts_show[:, 0],
            y=pts_show[:, 1],
            z=pts_show[:, 2],
            mode="markers",
            marker=dict(size=point_size),
            name=f"points (/{max(1, pc_stride)})",
        ))

    # cameras
    if show_cams and len(cams) > 0:
        for cam in cams:
            xs, ys, zs, c = make_frustum_lines(
                cam["c2w"], cam["K"], cam["W"], cam["H"], depth=frustum_depth
            )

            is_selected = (cam["idx"] == selected_cam_idx)

            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=6 if is_selected else 3),
                name=f"cam_{cam['idx']}",
            ))

            fig.add_trace(go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers+text" if is_selected else "markers",
                text=[f"cam_{cam['idx']}"] if is_selected else None,
                textposition="top center",
                marker=dict(size=6 if is_selected else 4),
                name=f"center_{cam['idx']}",
                showlegend=False,
            ))

    half = span * 0.6
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis=dict(title="x", range=[center[0] - half, center[0] + half]),
            yaxis=dict(title="y", range=[center[1] - half, center[1] + half]),
            zaxis=dict(title="z", range=[center[2] - half, center[2] + half]),
            aspectmode="data",
        ),
        uirevision="keep",
        title="SOPHY sample viewer",
        legend=dict(itemsizing="constant"),
    )

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="例如 /data/gaoya/dataset/SOPHY_data/bag/simulation_data/train/bag/01_000__0",
    )
    parser.add_argument(
        "--static_json",
        type=str,
        required=True,
        help="例如 /data/gaoya/dataset/SOPHY_data/bag/data/train/bag/01_000__0/data_static.json",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--default_pc_stride", type=int, default=1)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    static_json = Path(args.static_json)

    mesh_path = sample_dir / "material.obj"
    ply_path = sample_dir / "sampled_points.ply"

    mesh = load_mesh(mesh_path) if mesh_path.exists() else None
    pts = load_point_cloud(ply_path) if ply_path.exists() else None
    cams = collect_cameras(static_json)

    cam_options = [
        {"label": f"{c['idx']}: {Path(c['raw']['file_path']).name}", "value": c["idx"]}
        for c in cams
    ]

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.H3("SOPHY 可视化"),
            html.Div("选择相机"),
            dcc.Dropdown(
                id="cam-dropdown",
                options=cam_options,
                value=0 if len(cam_options) > 0 else None,
                clearable=False,
            ),
            html.Br(),

            html.Div("显示内容"),
            dcc.Checklist(
                id="layer-checklist",
                options=[
                    {"label": "mesh", "value": "mesh"},
                    {"label": "points", "value": "points"},
                    {"label": "cameras", "value": "cameras"},
                ],
                value=["mesh", "points", "cameras"],
                inline=False,
            ),
            html.Br(),

            html.Div("点云下采样 stride"),
            dcc.Slider(
                id="pc-stride-slider",
                min=1,
                max=50,
                step=1,
                value=max(1, args.default_pc_stride),
                marks={1: "1", 5: "5", 10: "10", 20: "20", 50: "50"},
            ),
            html.Br(),

            html.Div("点大小"),
            dcc.Slider(
                id="point-size-slider",
                min=1,
                max=8,
                step=1,
                value=2,
                marks={1: "1", 2: "2", 4: "4", 8: "8"},
            ),
            html.Br(),

            html.Hr(),
            html.Div(id="cam-meta", style={"whiteSpace": "pre-wrap", "fontSize": "13px"}),
        ], style={
            "width": "20%",
            "padding": "12px",
            "boxSizing": "border-box",
            "borderRight": "1px solid #ddd",
            "height": "100vh",
            "overflowY": "auto",
        }),

        html.Div([
            dcc.Graph(
                id="scene-graph",
                style={"height": "100vh"},
                config={"scrollZoom": True},
            )
        ], style={
            "width": "55%",
            "height": "100vh",
        }),

        html.Div([
            html.H4("当前相机图像"),
            html.Img(
                id="cam-image",
                style={
                    "width": "100%",
                    "height": "auto",
                    "border": "1px solid #ccc",
                }
            )
        ], style={
            "width": "25%",
            "padding": "12px",
            "boxSizing": "border-box",
            "borderLeft": "1px solid #ddd",
            "height": "100vh",
            "overflowY": "auto",
        }),
    ], style={"display": "flex", "fontFamily": "Arial, sans-serif"})

    @app.callback(
        Output("scene-graph", "figure"),
        Output("cam-image", "src"),
        Output("cam-meta", "children"),
        Input("cam-dropdown", "value"),
        Input("layer-checklist", "value"),
        Input("pc-stride-slider", "value"),
        Input("point-size-slider", "value"),
    )
    def update_view(selected_cam_idx, layers, pc_stride, point_size):
        selected_cam_idx = 0 if selected_cam_idx is None else int(selected_cam_idx)
        show_mesh = "mesh" in layers
        show_points = "points" in layers
        show_cams = "cameras" in layers

        fig = build_figure(
            mesh=mesh,
            pts=pts,
            cams=cams,
            selected_cam_idx=selected_cam_idx,
            show_mesh=show_mesh,
            show_points=show_points,
            show_cams=show_cams,
            pc_stride=max(1, int(pc_stride)),
            point_size=int(point_size),
        )

        cam = cams[selected_cam_idx]
        img_src = image_to_data_url(cam["img_path"])

        meta = {
            "camera_idx": cam["idx"],
            "image_file": str(cam["img_path"]),
            "shape_id": cam["raw"].get("shape_id"),
            "style_id": cam["raw"].get("style_id"),
            "K": np.array(cam["K"]).round(4).tolist(),
            "c2w": np.array(cam["c2w"]).round(4).tolist(),
        }

        meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
        return fig, img_src, meta_text

    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()


'''

python demo_my.py \
  --sample_dir /data/gaoya/dataset/SOPHY_data/bag/simulation_data/train/bag/01_000__0 \
  --static_json /data/gaoya/dataset/SOPHY_data/bag/data/train/bag/01_000__0/data_static.json \
  --host 127.0.0.1 \
  --port 8050

'''