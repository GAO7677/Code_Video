import argparse
import json
import random
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from flask import Flask, jsonify
import trimesh

from Code_Video.Code_data.old_version.physxnet_genesis_loader import PhysXNetGenesisLoader


def mesh_to_plotly_figure(mesh_path: str, title: str):
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh))
        )

    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                opacity=1.0,
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def pick_random_object(loader, max_read=None, seed=0, obj_id=None):
    if obj_id is not None:
        return loader.get_object(obj_id, export_merged=True)
    ids = loader.list_object_ids()
    if max_read is not None:
        ids = ids[:max_read]
    rng = random.Random(seed)
    obj_id = rng.choice(ids)
    obj = loader.get_object(obj_id, export_merged=True)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max_read", type=int, default=1000,
                        help="只在前多少个物体里随机抽，避免全量扫描太慢")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    merged_cache_dir = Path(args.root) / args.version / "_merged_for_genesis"

    loader = PhysXNetGenesisLoader(
        root=args.root,
        version=args.version,
        merged_cache_dir=str(merged_cache_dir),
    )

    # obj = pick_random_object(loader, max_read=args.max_read, seed=args.seed, 
    obj = pick_random_object(loader, obj_id="712")  # 712 是一个好看的椅子，先固定展示它)

    print("=== Random Object ===")
    print("object_id:", obj.object_id)
    print("object_name:", obj.object_name)
    print("merged_mesh_path:", obj.merged_mesh_path)
    print("num_parts:", len(obj.parts))
    print("genesis_rigid:", json.dumps(obj.genesis_rigid, ensure_ascii=False, indent=2))

    fig = mesh_to_plotly_figure(
        obj.merged_mesh_path,
        title=f"{obj.object_name} ({obj.object_id})"
    )
    fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

    summary = {
        "object_id": obj.object_id,
        "object_name": obj.object_name,
        "category": obj.category,
        "dimension_cm": obj.dimension_cm,
        "dimension_m": obj.dimension_m,
        "merged_mesh_path": obj.merged_mesh_path,
        "num_parts": len(obj.parts),
        "genesis_rigid": obj.genesis_rigid,
        "parts": [
            {
                "part_id": p.part_id,
                "name": p.name,
                "mesh_path": p.mesh_path,
                "material_name": p.material_name,
                "density_kgm3": p.density_kgm3,
                "youngs_modulus_pa": p.youngs_modulus_pa,
                "poisson_ratio": p.poisson_ratio,
                "priority_rank": p.priority_rank,
                "joint_type": p.joint_type,
            }
            for p in obj.parts
        ],
    }

    app = Flask(__name__)

    @app.route("/")
    def index():
        items = "".join(
            f"<li><b>{p['part_id']}</b> | {p['name']} | material={p['material_name']} | "
            f"density={p['density_kgm3']} | joint={p['joint_type']}</li>"
            for p in summary["parts"]
        )

        return f"""
        <html>
        <head>
          <meta charset="utf-8"/>
          <title>PhysXNet Random Viewer</title>
          <style>
            body {{ font-family: Arial, sans-serif; margin: 24px; }}
            .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 16px; }}
            pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; overflow-x: auto; }}
          </style>
        </head>
        <body>
          <div class="card">
            <h2>{summary["object_name"]} ({summary["object_id"]})</h2>
            <p><b>category:</b> {summary["category"]}</p>
            <p><b>merged_mesh_path:</b> {summary["merged_mesh_path"]}</p>
            <p><b>num_parts:</b> {summary["num_parts"]}</p>
            <p><b>dimension_cm:</b> {summary["dimension_cm"]}</p>
            <p><b>dimension_m:</b> {summary["dimension_m"]}</p>
          </div>

          <div class="card">
            <h3>Parts</h3>
            <ul>{items}</ul>
          </div>

          <div class="card">
            <h3>genesis_rigid</h3>
            <pre>{json.dumps(summary["genesis_rigid"], ensure_ascii=False, indent=2)}</pre>
          </div>

          <div class="card">
            <h3>Merged Mesh Viewer</h3>
            {fig_html}
          </div>
        </body>
        </html>
        """

    @app.route("/api/object")
    def api_object():
        return jsonify(summary)

    url = f"http://127.0.0.1:{args.port}"
    print(f"Viewer running at: {url}")
    print(f"如果在远程机器上运行，请端口转发：ssh -L {args.port}:127.0.0.1:{args.port} user@server")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()