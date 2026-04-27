
import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from flask import Flask, jsonify

# 这里假设你的 loader 文件名就是 physxnet_genesis_loader.py
from demo_my import PhysXNetGenesisLoader


def mesh_to_plotly_figure(mesh_path: str, title: str):
    import trimesh

    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ))

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
                flatshading=False,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="例如 /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
    parser.add_argument("--version", type=str, default="version_1")
    parser.add_argument("--object_id", type=str, default="712")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--open_browser", action="store_true")
    args = parser.parse_args()

    merged_cache_dir = Path(args.root) / args.version / "_merged_for_genesis"

    loader = PhysXNetGenesisLoader(
        root=args.root,
        version=args.version,
        merged_cache_dir=str(merged_cache_dir),
    )

    obj = loader.get_object(args.object_id, export_merged=True)

    print("=== Single Object ===")
    print("object_id:", obj.object_id)
    print("object_name:", obj.object_name)
    print("merged_mesh_path:", obj.merged_mesh_path)
    print("num_parts:", len(obj.parts))
    print("genesis_rigid:", json.dumps(obj.genesis_rigid, ensure_ascii=False, indent=2))

    index_path = Path(args.root) / args.version / "physxnet_genesis_index.json"
    loader.save_index_json(
        save_path=str(index_path),
        export_merged=True,
    )
    print(f"Saved dataset index json: {index_path}")

    fig = mesh_to_plotly_figure(
        obj.merged_mesh_path,
        title=f"{obj.object_name} ({obj.object_id})"
    )
    fig_html = fig.to_html(full_html=True, include_plotlyjs="cdn")

    summary = {
        "object_id": obj.object_id,
        "object_name": obj.object_name,
        "merged_mesh_path": obj.merged_mesh_path,
        "num_parts": len(obj.parts),
        "dimension_cm": obj.dimension_cm,
        "dimension_m": obj.dimension_m,
        "category": obj.category,
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
        ]
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
          <title>PhysXNet Viewer</title>
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
    print("如果你是本机运行，浏览器打开上面这个地址即可。")
    print("如果你是远程服务器运行，请做端口转发，例如：")
    print(f"ssh -L {args.port}:127.0.0.1:{args.port} your_user@your_server")
    print(f"然后在本地浏览器打开 {url}")

    if args.open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
