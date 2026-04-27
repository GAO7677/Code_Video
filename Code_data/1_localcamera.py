from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import gradio as gr
import numpy as np


DEFAULT_MAIN_SCRIPT = Path(__file__).resolve().parent / "1_genesis_demo_physxnet_urdf_loader_camera_tunable.py"
DEFAULT_SCENE_JSON = "/data/gaoya/AAA_test_video/Dataset_test/physxnet_proxy_dataset_v233337/train/train_scene_000001/ann/scene_input.json"


_MODULE_CACHE: Dict[str, Any] = {}


def load_module(script_path: str):
    path = Path(script_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"main script not found: {path}")

    cache_key = f"{path}::{path.stat().st_mtime_ns}"
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    module_name = f"physx_camera_main_{hashlib.md5(str(path).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import script: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    _MODULE_CACHE.clear()
    _MODULE_CACHE[cache_key] = module
    return module


def find_scene_json(scene_json_path: str) -> Path:
    p = Path(scene_json_path).expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_dir():
        candidates = sorted(p.rglob("scene_input.json"))
        if candidates:
            return candidates[0].resolve()
    raise FileNotFoundError(f"scene json not found: {p}")


def load_scene_cfg(scene_json_path: str) -> Dict[str, Any]:
    path = find_scene_json(scene_json_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_slider_ranges(scene_cfg: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    corner = scene_cfg.get("corner", {})
    x0 = float(corner.get("x0", -2.0))
    x1 = float(corner.get("x1", 2.0))
    y0 = float(corner.get("y0", 0.0))
    y1 = float(corner.get("y1", 4.0))
    z1 = float(corner.get("z1", 2.5))
    width = max(x1 - x0, 1.0)
    depth = max(y1 - y0, 1.0)
    return {
        "cam_x": (x0 - 0.8 * width, x1 + 0.8 * width),
        "cam_y": (y0 - 2.5 * depth, y1 + 0.5 * depth),
        "cam_z": (0.05, max(3.5, z1 + 1.5)),
        "look_x": (x0 - 0.5 * width, x1 + 0.5 * width),
        "look_y": (y0 - 0.5 * depth, y1 + 0.5 * depth),
        "look_z": (0.0, max(2.8, z1 + 1.0)),
        "fov": (20.0, 70.0),
    }


def build_camera_dict(cam_x, cam_y, cam_z, look_x, look_y, look_z, fov, scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    current_cam = scene_cfg.get("camera", {})
    return {
        "pos": [float(cam_x), float(cam_y), float(cam_z)],
        "lookat": [float(look_x), float(look_y), float(look_z)],
        "fov": float(fov),
        "res": current_cam.get("res", [960, 720]),
        "GUI": False,
        "camera_style": "manual_override",
    }


def render_once(script_path, scene_json_path, cam_x, cam_y, cam_z, look_x, look_y, look_z, fov):
    module = load_module(script_path)
    scene_cfg = load_scene_cfg(scene_json_path)
    cam = build_camera_dict(cam_x, cam_y, cam_z, look_x, look_y, look_z, fov, scene_cfg)
    rgb = module.render_preview_image_with_camera(scene_cfg, cam)
    cam = module.normalize_camera_cfg(cam)
    cam_json = json.dumps(cam, ensure_ascii=False, indent=2)
    status = (
        f"scene={find_scene_json(scene_json_path)}\n"
        f"distance={cam['distance']:.4f}\n"
        f"camera_style={cam.get('camera_style', 'manual_override')}"
    )
    return rgb, cam_json, status


def load_current_camera(script_path, scene_json_path):
    module = load_module(script_path)
    scene_cfg = load_scene_cfg(scene_json_path)
    cam = module.normalize_camera_cfg(scene_cfg["camera"])
    ranges = compute_slider_ranges(scene_cfg)
    preview = module.render_preview_image_with_camera(scene_cfg, cam)
    cam_json = json.dumps(cam, ensure_ascii=False, indent=2)
    status = f"loaded current camera from {find_scene_json(scene_json_path)}"
    return (
        gr.update(minimum=ranges["cam_x"][0], maximum=ranges["cam_x"][1], value=cam["pos"][0]),
        gr.update(minimum=ranges["cam_y"][0], maximum=ranges["cam_y"][1], value=cam["pos"][1]),
        gr.update(minimum=ranges["cam_z"][0], maximum=ranges["cam_z"][1], value=cam["pos"][2]),
        gr.update(minimum=ranges["look_x"][0], maximum=ranges["look_x"][1], value=cam["lookat"][0]),
        gr.update(minimum=ranges["look_y"][0], maximum=ranges["look_y"][1], value=cam["lookat"][1]),
        gr.update(minimum=ranges["look_z"][0], maximum=ranges["look_z"][1], value=cam["lookat"][2]),
        gr.update(minimum=ranges["fov"][0], maximum=ranges["fov"][1], value=cam["fov"]),
        preview,
        cam_json,
        status,
    )


def save_camera_json(script_path, scene_json_path, camera_json: str, save_path: str):
    module = load_module(script_path)
    scene_path = find_scene_json(scene_json_path)
    if save_path.strip():
        out_path = Path(save_path).expanduser()
    else:
        out_path = scene_path.parent / "camera_manual.json"
    cam = json.loads(camera_json)
    module.dump_camera_cfg(cam, out_path)
    return f"saved: {out_path.resolve()}"


def build_demo(default_script_path: str, default_scene_json: str) -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Genesis 相机交互式调参")
        gr.Markdown("先加载当前场景相机，再拖动滑条。调到满意后可以直接保存成 JSON，主脚本用 `--manual_camera_json` 读取即可。")

        with gr.Row():
            script_path = gr.Textbox(label="主脚本路径", value=default_script_path)
            scene_json_path = gr.Textbox(label="scene_input.json 路径（或场景目录）", value=default_scene_json)

        with gr.Row():
            load_btn = gr.Button("读取当前相机")
            render_btn = gr.Button("刷新预览")

        with gr.Row():
            cam_x = gr.Slider(-5.0, 5.0, value=0.0, step=0.01, label="cam_x")
            cam_y = gr.Slider(-8.0, 4.0, value=-3.2, step=0.01, label="cam_y")
            cam_z = gr.Slider(0.05, 4.0, value=0.8, step=0.01, label="cam_z")

        with gr.Row():
            look_x = gr.Slider(-5.0, 5.0, value=0.0, step=0.01, label="look_x")
            look_y = gr.Slider(-2.0, 5.0, value=0.2, step=0.01, label="look_y")
            look_z = gr.Slider(0.0, 3.0, value=0.3, step=0.01, label="look_z")
            fov = gr.Slider(20.0, 70.0, value=35.0, step=0.1, label="fov")

        with gr.Row():
            out_img = gr.Image(label="预览", type="numpy")
            with gr.Column():
                out_json = gr.Code(label="当前相机 JSON", language="json")
                status = gr.Textbox(label="状态", lines=4)
                save_path = gr.Textbox(label="保存路径（留空则保存到 scene_input.json 同目录 camera_manual.json）", value="")
                save_btn = gr.Button("保存 JSON")
                save_status = gr.Textbox(label="保存结果", lines=2)

        load_btn.click(
            fn=load_current_camera,
            inputs=[script_path, scene_json_path],
            outputs=[cam_x, cam_y, cam_z, look_x, look_y, look_z, fov, out_img, out_json, status],
        )

        render_btn.click(
            fn=render_once,
            inputs=[script_path, scene_json_path, cam_x, cam_y, cam_z, look_x, look_y, look_z, fov],
            outputs=[out_img, out_json, status],
        )

        save_btn.click(
            fn=save_camera_json,
            inputs=[script_path, scene_json_path, out_json, save_path],
            outputs=[save_status],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", type=str, default=str(DEFAULT_MAIN_SCRIPT))
    parser.add_argument("--scene_json", type=str, default=DEFAULT_SCENE_JSON)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    demo = build_demo(default_script_path=args.script_path, default_scene_json=args.scene_json)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=6 python /home/gaoya/Code_Video/Code_data/1_localcamera.py \
  --script_path /home/gaoya/Code_Video/Code_data/1_genesis_demo_physxnet_urdf_loader_camera_tunable.py \
  --scene_json /data/gaoya/AAA_test_video/Dataset_test/physxnet_proxy_dataset_v233338/train/train_scene_000001/ann/scene_input.json \
  --host 0.0.0.0 \
  --port 7861

'''