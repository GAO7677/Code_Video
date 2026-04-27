#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# -----------------------------
# basic math utils (no numpy)
# -----------------------------

def parse_vec3(s, default=(0.0, 0.0, 0.0)):
    if not s:
        return list(default)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"expected 3 numbers, got: {s}")
    return vals

def eye4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

def mat4_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = 0.0
            for k in range(4):
                s += a[i][k] * b[k][j]
            out[i][j] = s
    return out

def scale4(sx, sy, sz):
    return [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

def transform_from_xyz_rpy(xyz, rpy):
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    rx = [
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ]
    ry = [
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ]
    rz = [
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ]

    def mat3_mul(a, b):
        out = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                out[i][j] = sum(a[i][k] * b[k][j] for k in range(3))
        return out

    rot = mat3_mul(mat3_mul(rz, ry), rx)

    T = eye4()
    for i in range(3):
        for j in range(3):
            T[i][j] = rot[i][j]
    T[0][3], T[1][3], T[2][3] = xyz
    return T

def flatten_col_major(m):
    # three.js Matrix4.fromArray expects column-major
    return [
        m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3],
    ]

# -----------------------------
# urdf parsing
# -----------------------------

class Visual:
    def __init__(self, mesh_file, scale_xyz, origin_tf):
        self.mesh_file = mesh_file
        self.scale_xyz = scale_xyz
        self.origin_tf = origin_tf

class Joint:
    def __init__(self, name, jtype, parent, child, origin_tf):
        self.name = name
        self.jtype = jtype
        self.parent = parent
        self.child = child
        self.origin_tf = origin_tf

def parse_urdf(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = []
    link_visuals = {}
    joints = []
    child_to_joint = {}

    for link in root.findall("link"):
        lname = link.attrib["name"]
        links.append(lname)
        link_visuals[lname] = []

        for visual in link.findall("visual"):
            origin = visual.find("origin")
            xyz = parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
            rpy = parse_vec3(origin.attrib.get("rpy") if origin is not None else None)
            origin_tf = transform_from_xyz_rpy(xyz, rpy)

            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is None:
                continue

            mesh_file = mesh.attrib.get("filename")
            if not mesh_file:
                continue

            scale_xyz = parse_vec3(mesh.attrib.get("scale", "1 1 1"), default=(1.0, 1.0, 1.0))
            link_visuals[lname].append(Visual(mesh_file, scale_xyz, origin_tf))

    for joint in root.findall("joint"):
        jname = joint.attrib.get("name", "unnamed_joint")
        jtype = joint.attrib.get("type", "fixed")

        pe = joint.find("parent")
        ce = joint.find("child")
        if pe is None or ce is None:
            continue

        origin = joint.find("origin")
        xyz = parse_vec3(origin.attrib.get("xyz") if origin is not None else None)
        rpy = parse_vec3(origin.attrib.get("rpy") if origin is not None else None)
        origin_tf = transform_from_xyz_rpy(xyz, rpy)

        entry = Joint(
            name=jname,
            jtype=jtype,
            parent=pe.attrib["link"],
            child=ce.attrib["link"],
            origin_tf=origin_tf,
        )
        joints.append(entry)
        child_to_joint[entry.child] = entry

    return links, link_visuals, joints, child_to_joint

def compute_world_tf(links, child_to_joint):
    world = {}

    def solve(link_name):
        if link_name in world:
            return world[link_name]
        if link_name not in child_to_joint:
            world[link_name] = eye4()
            return world[link_name]
        joint = child_to_joint[link_name]
        parent_tf = solve(joint.parent)
        world[link_name] = mat4_mul(parent_tf, joint.origin_tf)
        return world[link_name]

    for ln in links:
        solve(ln)
    return world

# -----------------------------
# colors
# -----------------------------

PALETTE = [
    "#e74c3c", "#3498db", "#2ecc71", "#f1c40f",
    "#9b59b6", "#e67e22", "#1abc9c", "#95a5a6",
    "#7f8c8d", "#c0392b", "#2980b9", "#27ae60",
    "#f39c12", "#8e44ad", "#d35400", "#16a085",
]

# -----------------------------
# html template
# -----------------------------

HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>URDF Web Viewer</title>
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; background: #111; }
    #info {
      position: absolute; left: 12px; top: 12px; z-index: 10;
      color: #eee; font: 14px/1.4 sans-serif;
      background: rgba(0,0,0,0.45); padding: 10px 12px; border-radius: 8px;
      max-width: 420px;
    }
    #legend {
      margin-top: 8px;
      max-height: 35vh;
      overflow: auto;
      font-size: 12px;
    }
    .item { display: flex; align-items: center; margin: 4px 0; }
    .swatch { width: 12px; height: 12px; margin-right: 8px; border-radius: 3px; flex: 0 0 12px; }
    #loading { color: #ffd166; }
  </style>
</head>
<body>
  <div id="info">
    <div><b>URDF Web Viewer</b></div>
    <div id="loading">loading...</div>
    <div>左键旋转，滚轮缩放，右键平移</div>
    <div id="meta"></div>
    <div id="legend"></div>
  </div>

  <script type="module">
    import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { OBJLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/OBJLoader.js';

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000);
    camera.position.set(2.0, 1.5, 2.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.8));

    const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dir1.position.set(3, 4, 5);
    scene.add(dir1);

    const dir2 = new THREE.DirectionalLight(0xffffff, 0.4);
    dir2.position.set(-3, 2, -4);
    scene.add(dir2);

    scene.add(new THREE.AxesHelper(0.5));
    scene.add(new THREE.GridHelper(4.0, 20, 0x666666, 0x333333));

    const loader = new OBJLoader();
    const group = new THREE.Group();
    scene.add(group);

    function fitCameraToObject(object, controls, camera) {
      const box = new THREE.Box3().setFromObject(object);
      if (!isFinite(box.min.x)) return;
      const size = new THREE.Vector3();
      const center = new THREE.Vector3();
      box.getSize(size);
      box.getCenter(center);

      const maxDim = Math.max(size.x, size.y, size.z);
      const fov = camera.fov * Math.PI / 180.0;
      let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
      cameraZ *= 1.8;

      camera.position.set(center.x + cameraZ * 0.7, center.y + cameraZ * 0.4, center.z + cameraZ);
      camera.near = Math.max(0.001, maxDim / 1000);
      camera.far = Math.max(1000, maxDim * 1000);
      camera.updateProjectionMatrix();

      controls.target.copy(center);
      controls.update();
    }

    function addLegend(items) {
      const root = document.getElementById('legend');
      root.innerHTML = "";
      for (const it of items) {
        const row = document.createElement('div');
        row.className = 'item';
        row.innerHTML = `<div class="swatch" style="background:${it.color}"></div><div>${it.name}</div>`;
        root.appendChild(row);
      }
    }

    async function main() {
      const manifest = await fetch('/manifest.json').then(r => r.json());
      document.getElementById('meta').textContent =
        `robot=${manifest.robot_name}, links=${manifest.num_links}, joints=${manifest.num_joints}, visuals=${manifest.items.length}`;
      addLegend(manifest.items.map(x => ({ name: x.name, color: x.color })));

      let loaded = 0;
      const total = manifest.items.length;

      for (const item of manifest.items) {
        await new Promise((resolve, reject) => {
          loader.load(item.url, (obj) => {
            const m = new THREE.Matrix4();
            m.fromArray(item.matrix);

            obj.traverse((child) => {
              if (child.isMesh) {
                child.material = new THREE.MeshStandardMaterial({
                  color: item.color,
                  metalness: 0.05,
                  roughness: 0.8,
                });
                child.castShadow = false;
                child.receiveShadow = false;
              }
            });

            obj.matrixAutoUpdate = false;
            obj.applyMatrix4(m);
            group.add(obj);

            loaded += 1;
            document.getElementById('loading').textContent = `loading ${loaded}/${total}`;
            resolve();
          }, undefined, (err) => {
            console.error('load failed:', item.url, err);
            loaded += 1;
            document.getElementById('loading').textContent = `loading ${loaded}/${total} (some failed)`;
            resolve();
          });
        });
      }

      fitCameraToObject(group, controls, camera);
      document.getElementById('loading').textContent = 'done';
    }

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    main();
    animate();
  </script>
</body>
</html>
"""

# -----------------------------
# app state
# -----------------------------

class AppState:
    def __init__(self, urdf_path: Path):
        self.urdf_path = urdf_path.resolve()
        self.links, self.link_visuals, self.joints, self.child_to_joint = parse_urdf(self.urdf_path)
        self.link_world = compute_world_tf(self.links, self.child_to_joint)
        self.items = []
        self.mesh_paths = []

        robot_name = "scene"
        try:
            root = ET.parse(self.urdf_path).getroot()
            robot_name = root.attrib.get("name", "scene")
        except Exception:
            pass
        self.robot_name = robot_name

        idx = 0
        for li, link_name in enumerate(self.links):
            visuals = self.link_visuals.get(link_name, [])
            for vi, vis in enumerate(visuals):
                mesh_path = (self.urdf_path.parent / vis.mesh_file).resolve()
                if not mesh_path.exists():
                    print(f"[WARN] mesh not found: {mesh_path}")
                    continue

                T = mat4_mul(self.link_world[link_name], mat4_mul(vis.origin_tf, scale4(*vis.scale_xyz)))
                self.mesh_paths.append(mesh_path)
                self.items.append({
                    "name": f"{link_name}_vis{vi}",
                    "url": f"/mesh/{idx}.obj",
                    "matrix": flatten_col_major(T),
                    "color": PALETTE[li % len(PALETTE)],
                    "abs_path": str(mesh_path),
                })
                idx += 1

        print(f"[INFO] robot={self.robot_name}")
        print(f"[INFO] num_links={len(self.links)}")
        print(f"[INFO] num_joints={len(self.joints)}")
        print(f"[INFO] num_visual_items={len(self.items)}")

# -----------------------------
# http handler
# -----------------------------

def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, data: bytes, content_type: str):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path

            if path == "/" or path == "/index.html":
                self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return

            if path == "/manifest.json":
                payload = {
                    "robot_name": state.robot_name,
                    "num_links": len(state.links),
                    "num_joints": len(state.joints),
                    "items": [
                        {
                            "name": it["name"],
                            "url": it["url"],
                            "matrix": it["matrix"],
                            "color": it["color"],
                        }
                        for it in state.items
                    ],
                }
                self._send_bytes(
                    json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
                return

            if path.startswith("/mesh/") and path.endswith(".obj"):
                name = posixpath.basename(path)
                stem = name[:-4]
                try:
                    idx = int(stem)
                except ValueError:
                    self.send_error(404, "bad mesh id")
                    return

                if idx < 0 or idx >= len(state.mesh_paths):
                    self.send_error(404, "mesh id out of range")
                    return

                mesh_path = state.mesh_paths[idx]
                try:
                    data = mesh_path.read_bytes()
                except Exception as e:
                    self.send_error(500, f"failed reading mesh: {e}")
                    return

                self._send_bytes(data, "text/plain; charset=utf-8")
                return

            self.send_error(404, "not found")

        def log_message(self, fmt, *args):
            print("[HTTP]", fmt % args)

    return Handler

# -----------------------------
# main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    state = AppState(urdf_path)
    if not state.items:
        raise RuntimeError("No valid visual mesh found from URDF.")

    Handler = make_handler(state)
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"[INFO] serving on http://{args.host}:{args.port}")
    print("[INFO] open this in your browser.")
    print("[INFO] if using ssh tunnel locally:")
    print(f"       ssh -L {args.port}:127.0.0.1:{args.port} <user>@<server>")
    print(f"       then open http://127.0.0.1:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopping server...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()

'''

python /home/gaoya/Code_Video/PhysX-3D-main/demo_my.py \
  --urdf /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/urdf/39264.urdf \
  --host 0.0.0.0 \
  --port 8021
'''