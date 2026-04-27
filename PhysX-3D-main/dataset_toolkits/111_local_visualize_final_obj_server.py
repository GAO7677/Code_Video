#!/usr/bin/env python3
import argparse
import mimetypes
import os
from pathlib import Path
from flask import Flask, abort, send_from_directory, render_template_string, url_for

HTML = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>OBJ Viewer</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; font-family: Arial, sans-serif; }
    #app { width: 100%; height: 100%; position: relative; background: #202124; }
    #info {
      position: absolute; top: 12px; left: 12px; z-index: 10;
      background: rgba(0,0,0,0.55); color: #fff; padding: 10px 12px;
      border-radius: 8px; line-height: 1.5; font-size: 14px;
      max-width: min(720px, calc(100vw - 24px));
      white-space: pre-wrap;
    }
    #loading {
      position: absolute; top: 12px; right: 12px; z-index: 10;
      background: rgba(0,0,0,0.55); color: #fff; padding: 8px 10px;
      border-radius: 8px; font-size: 13px;
    }
  </style>
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
    }
  }
  </script>
</head>
<body>
  <div id="app">
    <div id="info">{{ info_text }}</div>
    <div id="loading">加载中...</div>
  </div>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
    import { MTLLoader } from 'three/addons/loaders/MTLLoader.js';

    const app = document.getElementById('app');
    const loading = document.getElementById('loading');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x202124);

    const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000);
    camera.position.set(1.8, 1.2, 1.8);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    app.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(3, 4, 5);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.8);
    fill.position.set(-4, 2, -3);
    scene.add(fill);

    const grid = new THREE.GridHelper(2.0, 20, 0x666666, 0x444444);
    grid.position.y = -0.5;
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.5);
    scene.add(axes);

    const objUrl = {{ obj_url | tojson }};
    const mtlUrl = {{ mtl_url | tojson }};

    function frameObject(object3d) {
      const box = new THREE.Box3().setFromObject(object3d);
      const size = box.getSize(new THREE.Vector3());
      const center = box.getCenter(new THREE.Vector3());
      object3d.position.sub(center);

      const maxDim = Math.max(size.x, size.y, size.z);
      if (maxDim > 0) {
        const fit = 1.2 / maxDim;
        object3d.scale.setScalar(fit);
      }

      const box2 = new THREE.Box3().setFromObject(object3d);
      const size2 = box2.getSize(new THREE.Vector3());
      const maxDim2 = Math.max(size2.x, size2.y, size2.z);
      camera.position.set(maxDim2 * 1.8, maxDim2 * 1.3, maxDim2 * 1.8);
      camera.near = Math.max(maxDim2 / 100, 0.001);
      camera.far = Math.max(maxDim2 * 100, 100);
      camera.updateProjectionMatrix();
      controls.target.set(0, 0, 0);
      controls.update();
    }

    function finish() {
      loading.textContent = '加载完成';
      setTimeout(() => loading.remove(), 1000);
    }

    function fail(err) {
      console.error(err);
      loading.textContent = '加载失败，请看浏览器控制台';
    }

    if (mtlUrl) {
      const mtlLoader = new MTLLoader();
      mtlLoader.load(mtlUrl, (materials) => {
        materials.preload();
        const loader = new OBJLoader();
        loader.setMaterials(materials);
        loader.load(objUrl, (obj) => {
          scene.add(obj);
          frameObject(obj);
          finish();
        }, undefined, fail);
      }, undefined, fail);
    } else {
      const loader = new OBJLoader();
      loader.load(objUrl, (obj) => {
        scene.add(obj);
        frameObject(obj);
        finish();
      }, undefined, fail);
    }

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
'''


def find_view_files(mesh_dir: Path):
    obj_path = mesh_dir / 'model_tex.obj'
    if not obj_path.exists():
        obj_path = mesh_dir / 'model.obj'
    if not obj_path.exists():
        raise FileNotFoundError(f'在 {mesh_dir} 下没有找到 model_tex.obj 或 model.obj')

    mtl_path = None
    candidate_mtl = obj_path.with_suffix('.mtl')
    if candidate_mtl.exists():
        mtl_path = candidate_mtl
    return obj_path, mtl_path


def create_app(mesh_dir: Path, sample_name: str):
    app = Flask(__name__)
    obj_path, mtl_path = find_view_files(mesh_dir)

    @app.route('/')
    def index():
        info_text = (
            f'sample: {sample_name}\n'
            f'dir: {mesh_dir}\n'
            f'obj: {obj_path.name}\n'
            f'mtl: {mtl_path.name if mtl_path else "未找到，按纯 OBJ 加载"}\n'
            '鼠标左键旋转，滚轮缩放，右键平移。'
        )
        return render_template_string(
            HTML,
            info_text=info_text,
            obj_url=url_for('serve_file', filename=obj_path.name),
            mtl_url=url_for('serve_file', filename=mtl_path.name) if mtl_path else None,
        )

    @app.route('/files/<path:filename>')
    def serve_file(filename):
        file_path = mesh_dir / filename
        if not file_path.exists() or not file_path.is_file():
            abort(404)
        mime = mimetypes.guess_type(str(file_path))[0]
        return send_from_directory(mesh_dir, filename, mimetype=mime)

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='通过本地端口可视化最终生成的 OBJ')
    parser.add_argument('--root', type=str, default='/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/phy_dataset/output', help='包含各个样本子目录的根目录')
    parser.add_argument('--name', type=str, required=True, help='样本目录名，例如 39264')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8023)
    args = parser.parse_args()

    mesh_dir = Path(args.root) / args.name
    if not mesh_dir.exists():
        raise FileNotFoundError(f'样本目录不存在: {mesh_dir}')

    app = create_app(mesh_dir, args.name)
    print(f'Viewer URL: http://127.0.0.1:{args.port}')
    print(f'也可从远程机访问: http://{args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=False)
