#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
urdf_browser.py  -  PhysXNet URDF Folder Browser + 3-D Viewer

Usage:
    python /home/gaoya/Code_Video/PhysX-3D-main/urdf_browser.py \
        --urdf_dir /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/urdf \
        --port 8021

Open http://127.0.0.1:8021 in your browser.
Remote: ssh -L 8021:127.0.0.1:8021 user@server
"""

import argparse
import json
import math
import posixpath
import xml.etree.ElementTree as ET
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


def parse_vec3(s, default=(0.0, 0.0, 0.0)):
    if not s:
        return list(default)
    v = [float(x) for x in s.strip().split()]
    return v if len(v) == 3 else list(default)


def eye4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a, b):
    o = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            o[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return o


def scale4(sx, sy, sz):
    return [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def xform(xyz, rpy):
    roll, pitch, yaw = rpy
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy_ = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)

    def m3(a, b):
        return [
            [sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)
        ]

    # Rz * Ry * Rx
    rot = m3(
        m3(
            [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]],
            [[cy, 0, sy_], [0, 1, 0], [-sy_, 0, cy]],
        ),
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]],
    )

    T = eye4()
    for i in range(3):
        for j in range(3):
            T[i][j] = rot[i][j]
    T[0][3], T[1][3], T[2][3] = xyz
    return T


def col_major(m):
    return [m[r][c] for c in range(4) for r in range(4)]


def parse_urdf(p):
    root = ET.parse(p).getroot()
    links, lvis, joints, c2j = [], {}, {}, {}

    for lk in root.findall("link"):
        n = lk.attrib["name"]
        links.append(n)
        lvis[n] = []

        for vis in lk.findall("visual"):
            orig = vis.find("origin")
            xyz = parse_vec3(orig.attrib.get("xyz") if orig is not None else None)
            rpy = parse_vec3(orig.attrib.get("rpy") if orig is not None else None)

            geo = vis.find("geometry")
            if geo is None:
                continue

            mesh = geo.find("mesh")
            if mesh is None:
                continue

            mf = mesh.attrib.get("filename")
            if not mf:
                continue

            sc = parse_vec3(mesh.attrib.get("scale", "1 1 1"), (1.0, 1.0, 1.0))
            lvis[n].append((mf, sc, xform(xyz, rpy)))

    for jt in root.findall("joint"):
        pe = jt.find("parent")
        ce = jt.find("child")
        if pe is None or ce is None:
            continue

        orig = jt.find("origin")
        xyz = parse_vec3(orig.attrib.get("xyz") if orig is not None else None)
        rpy = parse_vec3(orig.attrib.get("rpy") if orig is not None else None)

        e = {
            "name": jt.attrib.get("name", ""),
            "type": jt.attrib.get("type", "fixed"),
            "parent": pe.attrib["link"],
            "child": ce.attrib["link"],
            "tf": xform(xyz, rpy),
        }
        joints[e["name"]] = e
        c2j[e["child"]] = e

    return links, lvis, list(joints.values()), c2j


def world_tfs(links, c2j):
    w = {}

    def solve(n):
        if n in w:
            return w[n]
        if n not in c2j:
            w[n] = eye4()
            return w[n]
        j = c2j[n]
        w[n] = mat4_mul(solve(j["parent"]), j["tf"])
        return w[n]

    for l in links:
        solve(l)
    return w


PALETTE = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f1c40f",
    "#9b59b6",
    "#e67e22",
    "#1abc9c",
    "#95a5a6",
    "#c0392b",
    "#2980b9",
    "#27ae60",
    "#f39c12",
    "#8e44ad",
    "#d35400",
    "#16a085",
    "#7f8c8d",
]


def build_manifest(urdf):
    links, lvis, joints, c2j = parse_urdf(urdf)
    w = world_tfs(links, c2j)

    try:
        rname = ET.parse(urdf).getroot().attrib.get("name", "scene")
    except Exception:
        rname = "scene"

    items, paths = [], []
    idx = 0

    for li, ln in enumerate(links):
        for vi, (mf, sc, otf) in enumerate(lvis.get(ln, [])):
            mp = (urdf.parent / mf).resolve()
            if not mp.exists():
                print(f"[WARN] missing: {mp}")
                continue

            T = mat4_mul(w[ln], mat4_mul(otf, scale4(*sc)))
            paths.append(mp)
            items.append(
                {
                    "name": f"{ln}_v{vi}",
                    "url": f"/mesh/{idx}.obj",
                    "matrix": col_major(T),
                    "color": PALETTE[li % len(PALETTE)],
                }
            )
            idx += 1

    return {
        "robot_name": rname,
        "num_links": len(links),
        "num_joints": len(joints),
        "joints": [{"name": j["name"], "type": j["type"]} for j in joints],
        "items": items,
    }, paths


class State:
    def __init__(self, urdf_dir):
        self.urdf_dir = Path(urdf_dir)
        self.all_urdfs = sorted(self.urdf_dir.glob("*.urdf"))
        self.names = [u.stem for u in self.all_urdfs]
        self._cache = {}

    def get(self, stem):
        if stem in self._cache:
            return self._cache[stem]

        urdf = self.urdf_dir / (stem + ".urdf")
        if not urdf.exists():
            return None, None

        try:
            m, p = build_manifest(urdf)
            self._cache[stem] = (m, p)
            return m, p
        except Exception as e:
            print(f"[ERR] {stem}: {e}")
            return None, None


CSS = r"""
:root {
  --bg: #0b1020;
  --panel: #121933;
  --panel2: #0f152b;
  --text: #e8ecf3;
  --muted: #9aa4b2;
  --line: #24304f;
  --accent: #67b7ff;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: Inter, system-ui, sans-serif;
}

#app {
  display: grid;
  grid-template-columns: 320px 1fr;
  width: 100vw;
  height: 100vh;
}

#sidebar {
  border-right: 1px solid var(--line);
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  display: flex;
  flex-direction: column;
  min-height: 0;
}

#title {
  padding: 18px 18px 10px 18px;
  font-size: 20px;
  font-weight: 700;
}

#subtitle {
  padding: 0 18px 14px 18px;
  color: var(--muted);
  font-size: 13px;
}

#searchWrap {
  padding: 0 18px 12px 18px;
}

#search {
  width: 100%;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--line);
  background: #0b1124;
  color: var(--text);
  outline: none;
}

#fileList {
  flex: 1;
  min-height: 0;
  overflow: auto;
  padding: 0 10px 10px 10px;
}

.fileItem {
  width: 100%;
  text-align: left;
  padding: 10px 12px;
  margin: 6px 0;
  border-radius: 10px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--text);
  cursor: pointer;
  font-size: 13px;
}

.fileItem:hover {
  background: rgba(255,255,255,0.05);
  border-color: var(--line);
}

.fileItem.active {
  background: rgba(103,183,255,0.14);
  border-color: var(--accent);
}

#main {
  display: grid;
  grid-template-rows: 96px 1fr;
  min-width: 0;
  min-height: 0;
}

#topbar {
  border-bottom: 1px solid var(--line);
  display: grid;
  grid-template-columns: 1fr auto;
  align-items: center;
  gap: 12px;
  padding: 14px 18px;
  background: rgba(255,255,255,0.02);
}

#meta {
  min-width: 0;
}

#modelName {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 6px;
}

#modelInfo {
  font-size: 13px;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

#controlsHint {
  font-size: 12px;
  color: var(--muted);
  text-align: right;
}

#viewerWrap {
  position: relative;
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-columns: 1fr 320px;
}

#viewer {
  position: relative;
  min-width: 0;
  min-height: 0;
  background: #0a0f1d;
}

#rightPanel {
  border-left: 1px solid var(--line);
  background: linear-gradient(180deg, var(--panel), var(--panel2));
  overflow: auto;
  padding: 14px;
}

.panelTitle {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 10px;
}

#jointList {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.jointItem {
  padding: 10px 12px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: rgba(255,255,255,0.03);
}

.jointName {
  font-family: "JetBrains Mono", monospace;
  font-size: 12px;
  color: var(--text);
  margin-bottom: 4px;
  word-break: break-all;
}

.jointType {
  font-size: 12px;
  color: var(--muted);
}

#status {
  position: absolute;
  left: 16px;
  bottom: 16px;
  padding: 8px 10px;
  border-radius: 10px;
  background: rgba(0,0,0,0.4);
  color: #dbe7ff;
  font-size: 12px;
  pointer-events: none;
}

#empty {
  color: var(--muted);
  font-size: 13px;
}

a {
  color: var(--accent);
}
"""

HTML_BODY = r"""
<div id="app">
  <aside id="sidebar">
    <div id="title">PhysXNet URDF Browser</div>
    <div id="subtitle">Browse URDF files and inspect meshes in 3D</div>
    <div id="searchWrap">
      <input id="search" placeholder="Search URDF name..." />
    </div>
    <div id="fileList"></div>
  </aside>

  <section id="main">
    <div id="topbar">
      <div id="meta">
        <div id="modelName">No model selected</div>
        <div id="modelInfo">Choose one URDF from the left list.</div>
      </div>
      <div id="controlsHint">
        Left drag: rotate<br />
        Right drag: pan<br />
        Wheel: zoom
      </div>
    </div>

    <div id="viewerWrap">
      <div id="viewer">
        <div id="status">Ready</div>
      </div>

      <div id="rightPanel">
        <div class="panelTitle">Joints</div>
        <div id="jointList">
          <div id="empty">No URDF loaded.</div>
        </div>
      </div>
    </div>
  </section>
</div>
"""

JS = r"""
import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { OBJLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/OBJLoader.js';

const fileListEl = document.getElementById('fileList');
const searchEl = document.getElementById('search');
const modelNameEl = document.getElementById('modelName');
const modelInfoEl = document.getElementById('modelInfo');
const jointListEl = document.getElementById('jointList');
const statusEl = document.getElementById('status');
const viewerEl = document.getElementById('viewer');

let allNames = [];
let activeName = null;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0f1d);

const camera = new THREE.PerspectiveCamera(
  55,
  Math.max(1, viewerEl.clientWidth) / Math.max(1, viewerEl.clientHeight),
  0.01,
  10000
);
camera.position.set(1.8, 1.2, 1.8);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(Math.max(1, viewerEl.clientWidth), Math.max(1, viewerEl.clientHeight));
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0.2, 0);

scene.add(new THREE.AmbientLight(0xffffff, 1.15));

const dir1 = new THREE.DirectionalLight(0xffffff, 1.4);
dir1.position.set(3, 4, 2);
scene.add(dir1);

const dir2 = new THREE.DirectionalLight(0xffffff, 0.8);
dir2.position.set(-2, 3, -4);
scene.add(dir2);

const grid = new THREE.GridHelper(2, 20, 0x3c5a99, 0x24304f);
grid.position.set(0, -0.0001, 0);
scene.add(grid);

const axes = new THREE.AxesHelper(0.3);
scene.add(axes);

const worldGroup = new THREE.Group();
scene.add(worldGroup);

function setStatus(msg) {
  statusEl.textContent = msg;
}

function clearWorld() {
  while (worldGroup.children.length > 0) {
    const obj = worldGroup.children.pop();
    disposeRecursive(obj);
  }
}

function disposeRecursive(obj) {
  obj.traverse((node) => {
    if (node.geometry) node.geometry.dispose?.();
    if (node.material) {
      if (Array.isArray(node.material)) {
        node.material.forEach((m) => m.dispose?.());
      } else {
        node.material.dispose?.();
      }
    }
  });
  if (obj.parent) obj.parent.remove(obj);
}

function applyColMajorMatrix(obj, arr16) {
  const m = new THREE.Matrix4();
  m.fromArray(arr16);
  obj.matrixAutoUpdate = false;
  obj.matrix.copy(m);
}

function fitCameraToObject(root) {
  const box = new THREE.Box3().setFromObject(root);
  if (box.isEmpty()) {
    controls.target.set(0, 0, 0);
    camera.position.set(1.8, 1.2, 1.8);
    controls.update();
    return;
  }

  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const maxDim = Math.max(size.x, size.y, size.z, 1e-6);
  const fov = camera.fov * Math.PI / 180.0;
  let dist = maxDim / (2 * Math.tan(fov / 2));
  dist *= 1.6;

  controls.target.copy(center);
  camera.position.copy(center.clone().add(new THREE.Vector3(dist, dist * 0.7, dist)));
  camera.near = Math.max(maxDim / 1000, 0.001);
  camera.far = Math.max(maxDim * 50, 100);
  camera.updateProjectionMatrix();
  controls.update();

  grid.scale.setScalar(Math.max(1, maxDim));
}

function renderFileList(names) {
  fileListEl.innerHTML = '';
  for (const name of names) {
    const btn = document.createElement('button');
    btn.className = 'fileItem' + (name === activeName ? ' active' : '');
    btn.textContent = name;
    btn.onclick = () => loadModel(name);
    fileListEl.appendChild(btn);
  }
}

function renderJointList(joints) {
  if (!joints || joints.length === 0) {
    jointListEl.innerHTML = '<div id="empty">No joints found.</div>';
    return;
  }
  jointListEl.innerHTML = '';
  for (const j of joints) {
    const item = document.createElement('div');
    item.className = 'jointItem';
    item.innerHTML = `
      <div class="jointName">${escapeHtml(j.name || '')}</div>
      <div class="jointType">type: ${escapeHtml(j.type || '')}</div>
    `;
    jointListEl.appendChild(item);
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

async function fetchJSON(url) {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`HTTP ${resp.status}: ${url}`);
  }
  return await resp.json();
}

async function loadFileList() {
  setStatus('Loading URDF list...');
  const data = await fetchJSON('/filelist.json');
  allNames = data.names || [];
  renderFileList(allNames);
  setStatus(`Found ${allNames.length} URDF files`);
}

searchEl.addEventListener('input', async () => {
  const q = searchEl.value.trim().toLowerCase();
  const names = !q ? allNames : allNames.filter(x => x.toLowerCase().includes(q));
  renderFileList(names);

  if (names.length === 1) {
    try {
      await loadModel(names[0]);
    } catch (err) {
      console.error(err);
      setStatus(`Auto load failed: ${err}`);
    }
  }
});

async function loadModel(name) {
  try {
    activeName = name;
    renderFileList(allNames);
    setStatus(`Loading ${name} ...`);

    const manifest = await fetchJSON(`/manifest.json?id=${encodeURIComponent(name)}`);
    if (manifest.error) {
      throw new Error(manifest.error);
    }

    modelNameEl.textContent = manifest.robot_name || name;
    modelInfoEl.textContent =
      `file: ${name}.urdf   |   links: ${manifest.num_links ?? 0}   |   joints: ${manifest.num_joints ?? 0}   |   visuals: ${(manifest.items || []).length}`;
    renderJointList(manifest.joints || []);

    clearWorld();

    const objLoader = new OBJLoader();
    const items = manifest.items || [];
    const loadedRoots = [];

    for (let i = 0; i < items.length; ++i) {
      const item = items[i];
      const url = `${item.url}?id=${encodeURIComponent(name)}`;
      setStatus(`Loading mesh ${i + 1}/${items.length}: ${item.name}`);

      const obj = await objLoader.loadAsync(url);

      obj.traverse((node) => {
        if (node.isMesh) {
          node.material = new THREE.MeshStandardMaterial({
            color: new THREE.Color(item.color || '#cccccc'),
            metalness: 0.0,
            roughness: 0.85,
          });
          node.castShadow = false;
          node.receiveShadow = false;
        }
      });

      applyColMajorMatrix(obj, item.matrix);
      worldGroup.add(obj);
      loadedRoots.push(obj);
    }

    fitCameraToObject(worldGroup);
    setStatus(`Loaded ${name} (${items.length} visuals)`);
  } catch (err) {
    console.error(err);
    modelNameEl.textContent = 'Load failed';
    modelInfoEl.textContent = String(err);
    jointListEl.innerHTML = '<div id="empty">Failed to load model.</div>';
    setStatus('Error');
  }
}

function onResize() {
  const w = Math.max(1, viewerEl.clientWidth);
  const h = Math.max(1, viewerEl.clientHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}
window.addEventListener('resize', onResize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

await loadFileList();

const params = new URLSearchParams(window.location.search);
const initial = params.get('id');

try {
  if (initial && allNames.includes(initial)) {
    await loadModel(initial);
  } else if (allNames.length > 0) {
    await loadModel(allNames[0]);
  } else {
    setStatus('No URDF files found');
  }
} catch (err) {
  console.error(err);
  setStatus(`Initial load failed: ${err}`);
}
"""


def make_page():
    return (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        "<title>PhysXNet URDF Browser</title>\n"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap' rel='stylesheet'>\n"
        "<style>" + CSS + "</style>\n"
        "</head>\n"
        "<body>\n"
        + HTML_BODY +
        "\n<script type='module'>"
        + JS +
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


PAGE = make_page()


class Handler(BaseHTTPRequestHandler):
    state = None

    def _send(self, data, ct):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            self._send(PAGE, "text/html; charset=utf-8")

        elif path == "/filelist.json":
            self._send(json.dumps({"names": self.state.names}), "application/json")

        elif path == "/manifest.json":
            stem = (qs.get("id") or [""])[0]
            if not stem:
                self._send(json.dumps({"error": "no id"}), "application/json")
                return

            manifest, _ = self.state.get(stem)
            if manifest is None:
                self._send(json.dumps({"error": f"not found: {stem}"}), "application/json")
                return

            self._send(json.dumps(manifest), "application/json")

        elif path.startswith("/mesh/") and path.endswith(".obj"):
            stem = (qs.get("id") or [""])[0]
            if not stem:
                self.send_error(400, "missing id")
                return

            _, paths = self.state.get(stem)
            if paths is None:
                self.send_error(404, "model not found")
                return

            try:
                idx = int(posixpath.basename(path)[:-4])
            except ValueError:
                self.send_error(400, "bad index")
                return

            if idx < 0 or idx >= len(paths):
                self.send_error(404, "mesh index out of range")
                return

            self._send(paths[idx].read_bytes(), "text/plain")

        else:
            self.send_error(404)

    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    args = parser.parse_args()

    state = State(args.urdf_dir)
    print(f"[INFO] Found {len(state.names)} URDF files in {args.urdf_dir}")

    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"[INFO] Serving at {url}")
    print(f"[INFO] Remote tunnel: ssh -L {args.port}:127.0.0.1:{args.port} user@server")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()