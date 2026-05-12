#!/usr/bin/env python3
"""Render a curated TDW image preview portal for scenes and realistic models."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tdw.controller import Controller
from tdw.librarian import ModelLibrarian
from tdw.output_data import Images
from tdw.tdw_utils import TDWUtils


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_asset_visual_portal")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = int(os.environ.get("TDW_PORT", "1183"))
BUILD_ADDRESS = "127.0.0.1"
BUILD_BOOT_WAIT = int(os.environ.get("TDW_BUILD_BOOT_WAIT", "18"))
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]
RESOLUTION = {"width": 1280, "height": 720}
PORTAL_TITLE = "TDW Asset Visual Portal"

MODEL_NAMES = [
    "hiker_backpack",
    "backpack",
    "small_purse",
    "coffeemug",
    "coffee_cup",
    "mug",
    "b04_ramlosa_bottle_2015_vray",
    "moet_chandon_bottle_vray",
    "b04_wineglass",
    "serving_bowl",
    "b04_bowl_smooth",
    "kettle",
    "teakettle_01",
    "toaster_002",
    "appliance-ge-profile-microwave3",
    "fridge_large",
    "dining_room_table",
    "brown_leather_dining_chair",
    "chair_eames_plastic_armchair",
    "minotti_helion_3_seater_sofa",
    "pillow01",
    "basket_18inx18inx12iin_wicker",
    "box_18inx18inx12in_cardboard",
    "b03_cooking_pot_01",
    "int_kitchen_accessories_le_creuset_frying_pan_28cm",
    "vase_01",
]

SCENES: list[dict[str, Any]] = [
    {
        "name": "mm_craftroom_1a",
        "camera_position": {"x": -0.6156362579862034, "y": 1.85, "z": -1.6914467174146353},
        "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
    },
    {
        "name": "mm_kitchen_2b",
        "camera_position": {"x": -1.4, "y": 1.78, "z": -2.0},
        "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
    },
    {
        "name": "tdw_room",
        "camera_position": {"x": -2.8, "y": 1.7, "z": -1.7},
        "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
    },
    {
        "name": "building_site",
        "camera_position": {"x": -4.8, "y": 2.4, "z": 4.8},
        "look_at": {"x": 0.0, "y": 0.6, "z": 0.0},
    },
    {
        "name": "suburb_scene_2023",
        "camera_position": {"x": -3.4, "y": 1.7, "z": -0.3},
        "look_at": {"x": 0.0, "y": 1.0, "z": 0.0},
    },
]


def sanitize_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def launch_build(log_path: Path) -> subprocess.Popen:
    ensure_dir(log_path.parent)
    log_fp = open(log_path, "ab")
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return subprocess.Popen(
        [str(BUILD_PATH), f"-port={PORT}", f"-address={BUILD_ADDRESS}", "-force-glcore42"],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        env=env,
    )


def init_controller() -> tuple[Controller, subprocess.Popen]:
    sanitize_proxy_env()
    build_log = OUTPUT_ROOT / "logs" / "build.log"
    print(f"launch build display={DISPLAY} port={PORT} address={BUILD_ADDRESS}", flush=True)
    proc = launch_build(build_log)
    print(f"sleep boot wait {BUILD_BOOT_WAIT}s", flush=True)
    time.sleep(BUILD_BOOT_WAIT)
    print("connect controller", flush=True)
    c = Controller(launch_build=False, check_version=False, port=PORT)
    print("controller connected", flush=True)
    return c, proc


def base_setup(c: Controller) -> None:
    print("base setup communicate", flush=True)
    commands = [
        Controller.get_add_scene(scene_name="tdw_room"),
        {"$type": "set_screen_size", "width": RESOLUTION["width"], "height": RESOLUTION["height"]},
        {"$type": "set_render_quality", "render_quality": 5},
        {"$type": "set_shadow_strength", "strength": 1.0},
        {"$type": "set_img_pass_encoding", "value": False},
        {"$type": "simulate_physics", "value": False},
        {"$type": "set_pass_masks", "pass_masks": ["_img"]},
        {"$type": "send_images", "frequency": "always"},
    ]
    commands.extend(TDWUtils.create_avatar(position={"x": 1.57, "y": 3.0, "z": 3.56}, avatar_id="a"))
    c.communicate(commands)
    print("base setup done", flush=True)


def save_rgb_from_response(resp: list[bytes], output_path: Path) -> None:
    images = Images(resp[0])
    ensure_dir(output_path.parent)
    TDWUtils.save_images(images=images, filename=output_path.stem, output_directory=str(output_path.parent), append_pass=False)
    generated = output_path.parent / f"{output_path.stem}.jpg"
    if generated != output_path and generated.exists():
        generated.rename(output_path)


def render_model_previews(c: Controller) -> list[dict[str, Any]]:
    out_dir = OUTPUT_ROOT / "previews" / "models"
    ensure_dir(out_dir)
    lib = ModelLibrarian("models_core.json")
    records = []
    c.communicate([
        Controller.get_add_scene(scene_name="tdw_room"),
        {"$type": "teleport_avatar_to", "avatar_id": "a", "position": {"x": 1.57, "y": 3.0, "z": 3.56}},
    ])
    for model_name in MODEL_NAMES:
        print(f"render model {model_name}", flush=True)
        record = lib.get_record(model_name)
        object_id = Controller.get_unique_id()
        unit_scale = TDWUtils.get_unit_scale(record) * 2
        resp = c.communicate([
            {"$type": "add_object",
             "name": record.name,
             "url": record.get_url(),
             "scale_factor": record.scale_factor,
             "rotation": record.canonical_rotation,
             "id": object_id},
            {"$type": "scale_object",
             "id": object_id,
             "scale_factor": {"x": unit_scale, "y": unit_scale, "z": unit_scale}},
            {"$type": "look_at",
             "avatar_id": "a",
             "object_id": object_id,
             "use_centroid": True},
        ])
        output_path = out_dir / f"{model_name}.jpg"
        save_rgb_from_response(resp, output_path)
        records.append({
            "name": record.name,
            "image": str(output_path.relative_to(OUTPUT_ROOT)),
            "category": str(record.wcategory),
            "library": "models_core.json",
            "do_not_use": bool(record.do_not_use),
            "composite_object": bool(record.composite_object),
        })
        c.communicate([
            {"$type": "destroy_object", "id": object_id},
            {"$type": "unload_asset_bundles"},
        ])
    return records


def render_scene_previews(c: Controller) -> list[dict[str, Any]]:
    out_dir = OUTPUT_ROOT / "previews" / "scenes"
    ensure_dir(out_dir)
    records: list[dict[str, Any]] = []
    for scene in SCENES:
        print(f"render scene {scene['name']}", flush=True)
        resp = c.communicate([
            Controller.get_add_scene(scene_name=scene["name"]),
            {"$type": "teleport_avatar_to", "avatar_id": "a", "position": scene["camera_position"]},
            {"$type": "look_at_position", "avatar_id": "a", "position": scene["look_at"]},
        ])
        output_path = out_dir / f"{scene['name']}.jpg"
        save_rgb_from_response(resp, output_path)
        records.append({
            "name": scene["name"],
            "image": str(output_path.relative_to(OUTPUT_ROOT)),
            "camera_position": scene["camera_position"],
            "look_at": scene["look_at"],
        })
    return records


def build_html(scene_records: list[dict[str, Any]], model_records: list[dict[str, Any]]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PORTAL_TITLE}</title>
  <style>
    :root {{
      --bg: #efe6d6;
      --panel: rgba(255, 252, 246, 0.96);
      --ink: #171410;
      --muted: #69635b;
      --accent: #3f6a56;
      --border: rgba(52, 42, 29, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(196, 162, 112, 0.26), transparent 28%),
        radial-gradient(circle at right 12%, rgba(126, 153, 141, 0.22), transparent 22%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1700px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .panel, .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 40px rgba(45, 35, 22, 0.10);
    }}
    .hero {{ padding: 28px; margin-bottom: 18px; }}
    .panel {{ padding: 22px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 10px; font-size: clamp(30px, 5vw, 54px); line-height: 0.95; }}
    h2 {{ margin: 0 0 14px; font-size: 26px; }}
    h3 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.65; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}
    .card {{
      overflow: hidden;
    }}
    .card img {{
      width: 100%;
      display: block;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #ddd;
    }}
    .meta {{
      padding: 16px 18px 18px;
    }}
    .pill {{
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(63, 106, 86, 0.12);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    code {{
      display: block;
      margin-top: 10px;
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      word-break: break-all;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="pill">Image Preview</div>
      <div class="pill">TDW</div>
      <h1>{PORTAL_TITLE}</h1>
      <p>这是一版图像可视化页面。TDW 现成资源主要是 Unity asset bundle，不是浏览器原生的 glTF/OBJ，所以这里先给你能直接打开的真实截图页。场景页展示一批代表性背景，模型页展示一批更贴近真实生活物体的公开模型。</p>
    </section>

    <section class="panel">
      <h2>Scene Previews</h2>
      <div class="grid">
        {''.join(
            f'''<article class="card">
  <img src="{item["image"]}" alt="{item["name"]}">
  <div class="meta">
    <div class="pill">scene</div>
    <h3>{item["name"]}</h3>
    <p>代表性 TDW 背景场景截图。</p>
    <code>camera={json.dumps(item["camera_position"], ensure_ascii=False)}</code>
  </div>
</article>'''
            for item in scene_records
        )}
      </div>
    </section>

    <section class="panel">
      <h2>Model Previews</h2>
      <div class="grid">
        {''.join(
            f'''<article class="card">
  <img src="{item["image"]}" alt="{item["name"]}">
  <div class="meta">
    <div class="pill">model</div>
    <div class="pill">{item["category"]}</div>
    <h3>{item["name"]}</h3>
    <p>library={item["library"]} | composite={str(item["composite_object"]).lower()} | do_not_use={str(item["do_not_use"]).lower()}</p>
  </div>
</article>'''
            for item in model_records
        )}
      </div>
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    c, proc = init_controller()
    try:
        base_setup(c)
        scene_records = render_scene_previews(c)
        model_records = render_model_previews(c)
        manifest = {
            "scenes": scene_records,
            "models": model_records,
        }
        (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        (OUTPUT_ROOT / "index.html").write_text(build_html(scene_records, model_records), encoding="utf-8")
        print(OUTPUT_ROOT / "index.html")
    finally:
        try:
            c.communicate({"$type": "terminate"})
        except Exception:
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
