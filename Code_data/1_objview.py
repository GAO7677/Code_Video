#!/usr/bin/env python3
"""Serve an interactive browser viewer for an OBJ mesh.

Example:
    python obj_viewer_server.py /path/to/model.obj --port 8021

Then on your local machine:
    ssh -L 8021:127.0.0.1:8021 user@remote_server

Open in browser:
    http://127.0.0.1:8021
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import trimesh
from trimesh.viewer import scene_to_html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve an interactive browser-based viewer for an OBJ mesh."
    )
    parser.add_argument(
        "obj_path",
        type=str,
        help="Path to the .obj file to visualize.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Bind host. Use 127.0.0.1 for SSH port forwarding, or 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8021,
        help="Port to serve on.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to open a browser automatically.",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Let trimesh preprocess the mesh (merge/fix a bit). Default is off to preserve raw geometry.",
    )
    return parser.parse_args()


def load_scene(mesh_path: Path, process: bool = False) -> trimesh.Scene:
    """Load the mesh file as a trimesh.Scene."""
    scene_or_mesh = trimesh.load(
        mesh_path,
        force="scene",
        process=process,
        maintain_order=True,
    )

    if isinstance(scene_or_mesh, trimesh.Scene):
        scene = scene_or_mesh
    else:
        scene = trimesh.Scene(scene_or_mesh)

    if len(scene.geometry) == 0:
        raise ValueError(f"No geometry found in: {mesh_path}")

    # Set a reasonable default camera.
    bounds = scene.bounds
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    scale = float(np.max(extents))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    scene.set_camera(
        angles=np.radians([55.0, 0.0, 35.0]),
        distance=2.5 * scale,
        center=center,
    )
    return scene


def write_html(scene: trimesh.Scene, out_dir: Path, title: str) -> Path:
    """Write the interactive HTML viewer."""
    html = scene_to_html(scene)
    html = html.replace("<title>trimesh: threejs viewer</title>", f"<title>{title}</title>")
    index_path = out_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def main() -> int:
    args = parse_args()
    mesh_path = Path(args.obj_path).expanduser().resolve()

    if not mesh_path.exists():
        print(f"[ERROR] File not found: {mesh_path}", file=sys.stderr)
        return 1
    if mesh_path.suffix.lower() != ".obj":
        print(
            f"[WARN] Input is not .obj: {mesh_path.name}. trimesh may still load it if supported.",
            file=sys.stderr,
        )

    try:
        scene = load_scene(mesh_path, process=args.process)
    except Exception as exc:
        print(f"[ERROR] Failed to load mesh: {exc}", file=sys.stderr)
        return 2

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="obj_viewer_")
    temp_dir = Path(temp_dir_obj.name)
    write_html(scene, temp_dir, title=f"OBJ Viewer - {mesh_path.name}")

    handler = partial(SimpleHTTPRequestHandler, directory=str(temp_dir))
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    local_url = f"http://127.0.0.1:{args.port}"
    bind_url = f"http://{args.host}:{args.port}"

    print(f"[INFO] OBJ     : {mesh_path}")
    print(f"[INFO] Serving : {bind_url}")
    print(f"[INFO] Open    : {local_url}")
    if args.host == "127.0.0.1":
        print(
            "[INFO] For remote server usage, create a local SSH tunnel like:\n"
            f"       ssh -L {args.port}:127.0.0.1:{args.port} <user>@<server>"
        )

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(local_url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        httpd.server_close()
        temp_dir_obj.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

python /home/gaoya/Code_Video/Code_data/1_objview.py \
/home/gaoya/Code_Video/Code_data/demo_outputs/raw_bowl_mesh/raw_bowl_mesh.obj \
    --port 8024 --no-browser

    
'''