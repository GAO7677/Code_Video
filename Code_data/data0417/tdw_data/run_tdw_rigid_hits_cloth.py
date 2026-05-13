from pathlib import Path
from typing import Dict, List
import os
import shutil
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.obi_data.cloth.sheet_type import SheetType
from tdw.obi_data.collision_materials.collision_material import CollisionMaterial


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_rigid_hits_cloth")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = int(os.environ.get("TDW_PORT", "1093"))
BUILD_BOOT_WAIT = int(os.environ.get("TDW_BUILD_BOOT_WAIT", "12"))
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "mm_craftroom_1a",
    "skybox": "kiara_1_dawn_4k",
    "camera_position": {"x": -1.34, "y": 2.28, "z": -2.92},
    "look_at": {"x": 0.02, "y": 1.02, "z": 0.0},
    "field_of_view": 68,
}

CASES: List[Dict[str, object]] = [
    {
        "name": "rigid_can_hits_draped_cloth",
        "supports": [
            {
                "model_name": "camera_box",
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 12.0, "z": 0.0},
                "mass": 24.0,
            },
        ],
        "cloth": {
            "cloth_material": "cotton",
            "sheet_type": SheetType.cloth_hd,
            "position": {"x": 0.0, "y": 1.94, "z": 0.0},
            "rotation": {"x": 10.0, "y": 0.0, "z": 0.0},
        },
        "striker": {
            "model_name": "102_pepsi_can_12_fl_oz_vray",
            "position": {"x": -1.2, "y": 1.0, "z": -0.14},
            "rotation": {"x": 0.0, "y": 0.0, "z": 84.0},
            "velocity": {"x": 1.35, "y": 0.03, "z": 0.08},
            "angular_velocity": {"x": 0.8, "y": 0.05, "z": 0.34},
            "mass": 0.45,
            "dynamic_friction": 0.4,
            "static_friction": 0.45,
            "bounciness": 0.08,
        },
        "settle_frames": 48,
        "capture_frames": 220,
    },
    {
        "name": "rigid_box_hits_draped_cloth",
        "supports": [
            {
                "model_name": "camera_box",
                "position": {"x": 0.04, "y": 0.0, "z": 0.02},
                "rotation": {"x": 0.0, "y": -10.0, "z": 0.0},
                "mass": 24.0,
            },
        ],
        "cloth": {
            "cloth_material": "canvas",
            "sheet_type": SheetType.cloth_hd,
            "position": {"x": 0.02, "y": 1.92, "z": 0.02},
            "rotation": {"x": 12.0, "y": 0.0, "z": 4.0},
        },
        "striker": {
            "model_name": "camera_box",
            "position": {"x": 1.12, "y": 0.84, "z": 0.14},
            "rotation": {"x": 0.0, "y": -18.0, "z": 0.0},
            "velocity": {"x": -1.0, "y": 0.03, "z": -0.1},
            "angular_velocity": {"x": 0.08, "y": 0.12, "z": 0.1},
            "mass": 1.2,
            "dynamic_friction": 0.6,
            "static_friction": 0.68,
            "bounciness": 0.04,
        },
        "settle_frames": 50,
        "capture_frames": 230,
    },
]


def sanitize_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def launch_build(log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "ab")
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return subprocess.Popen([str(BUILD_PATH), f"-port={PORT}", "-address=127.0.0.1", "-force-glcore42"],
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            env=env)


def convert_video(frames_dir: Path, output_video: Path) -> None:
    subprocess.run(["/home/gaoya/.venvs/tdw/bin/python",
                    "/home/gaoya/Code_Video/tdw-master/Python/example_controllers/frames_to_video.py",
                    str(frames_dir),
                    str(output_video),
                    "24"],
                   check=True)


def add_rigid(c: Controller, spec: Dict[str, object], object_id: int) -> List[dict]:
    return c.get_add_physics_object(model_name=str(spec["model_name"]),
                                    object_id=object_id,
                                    library="models_core.json",
                                    position=spec["position"],
                                    rotation=spec["rotation"],
                                    default_physics_values=False,
                                    mass=float(spec.get("mass", 1.0)),
                                    dynamic_friction=float(spec.get("dynamic_friction", 0.7)),
                                    static_friction=float(spec.get("static_friction", 0.8)),
                                    bounciness=float(spec.get("bounciness", 0.02)))


def add_static_support(c: Controller, spec: Dict[str, object], object_id: int) -> List[dict]:
    return c.get_add_physics_object(model_name=str(spec["model_name"]),
                                    object_id=object_id,
                                    library="models_core.json",
                                    position=spec["position"],
                                    rotation=spec["rotation"],
                                    default_physics_values=False,
                                    mass=float(spec.get("mass", 24.0)),
                                    dynamic_friction=float(spec.get("dynamic_friction", 0.88)),
                                    static_friction=float(spec.get("static_friction", 0.92)),
                                    bounciness=float(spec.get("bounciness", 0.01)),
                                    kinematic=False,
                                    gravity=True)


def build_html() -> None:
    cards = []
    for case in CASES:
        case_root = OUTPUT_ROOT / str(case["name"])
        video_path = case_root / f"{case['name']}.mp4"
        if not video_path.exists():
            continue
        rel = video_path.relative_to(OUTPUT_ROOT.parent)
        cards.append(
            f"""<article class="card">
  <video controls preload="metadata" src="../{rel}"></video>
  <div class="meta">
    <h3>{case['name']}</h3>
    <code>{video_path}</code>
  </div>
</article>"""
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TDW Rigid Hits Cloth</title>
  <style>
    body {{ margin: 0; font-family: Georgia, "Times New Roman", serif; background: linear-gradient(180deg, #f8f3ea 0%, #eee3d2 100%); color: #171410; }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .card {{ background: rgba(255,255,255,0.94); border: 1px solid rgba(52, 42, 29, 0.14); border-radius: 24px; box-shadow: 0 18px 40px rgba(45, 35, 22, 0.10); }}
    .hero {{ padding: 24px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; }}
    video {{ width: 100%; display: block; aspect-ratio: 16/9; background: #000; border-radius: 24px 24px 0 0; }}
    .meta {{ padding: 16px 18px 20px; }}
    code {{ display: block; margin-top: 10px; color: #6a665e; word-break: break-all; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>TDW Rigid Hits Cloth</h1>
      <p>先让布在支撑物上稳定，再把刚体从画外打入，观察软体受撞击后的形变。</p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>
"""
    (OUTPUT_ROOT / "index.html").write_text(html, encoding="utf-8")


def run_case(case: Dict[str, object]) -> None:
    sanitize_proxy_env()
    case_root = OUTPUT_ROOT / str(case["name"])
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)
    build_proc = launch_build(case_root / "build.log")
    time.sleep(BUILD_BOOT_WAIT)
    c = Controller(launch_build=False, check_version=False, port=PORT)
    try:
        obi = Obi(output_data=True,
                  floor_material=CollisionMaterial(dynamic_friction=0.55,
                                                   static_friction=0.6,
                                                   stickiness=0.0,
                                                   stick_distance=0.0))
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=SCENE["camera_position"],
                                   look_at=SCENE["look_at"],
                                   field_of_view=int(SCENE["field_of_view"]))
        capture = ImageCapture(path=case_root / "frames", avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        lighting = InteriorSceneLighting(hdri_skybox=str(SCENE["skybox"]),
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        c.add_ons.extend([lighting, camera, capture, obi])
        commands = [{"$type": "set_screen_size", "width": 1280, "height": 720},
                    {"$type": "set_physics_solver_iterations", "iterations": 16},
                    Controller.get_add_scene(scene_name=str(SCENE["name"])),
                    Controller.get_add_hdri_skybox(skybox_name=str(SCENE["skybox"]))]

        obi.set_solver(substeps=8)
        for support in case["supports"]:
            support_id = c.get_unique_id()
            commands.extend(add_static_support(c, support, support_id))

        print(f"[{case['name']}] initial communicate", flush=True)
        c.communicate(commands)
        # Let Obi create colliders for the floor/supports before adding cloth.
        c.communicate([])

        cloth_id = c.get_unique_id()
        cloth = case["cloth"]
        obi.create_cloth_sheet(cloth_material=str(cloth["cloth_material"]),
                               object_id=cloth_id,
                               sheet_type=cloth["sheet_type"],
                               position=cloth["position"],
                               rotation=cloth["rotation"])
        print(f"[{case['name']}] create cloth", flush=True)
        c.communicate([])

        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        total_capture_frames = int(case["capture_frames"])
        print(f"[{case['name']}] capture frames={total_capture_frames}", flush=True)
        settle_frames = int(case["settle_frames"])
        for frame_idx in range(total_capture_frames):
            if frame_idx == settle_frames:
                striker_id = c.get_unique_id()
                striker = case["striker"]
                striker_cmds = add_rigid(c, striker, striker_id)
                striker_cmds.extend([
                    {"$type": "set_velocity", "id": striker_id, "velocity": striker["velocity"]},
                    {"$type": "set_angular_velocity", "id": striker_id, "angular_velocity": striker["angular_velocity"]},
                ])
                print(f"[{case['name']}] inject striker at frame={frame_idx}", flush=True)
                c.communicate(striker_cmds)
                continue
            c.communicate([])
            if (frame_idx + 1) % 30 == 0 or frame_idx + 1 == total_capture_frames:
                print(f"[{case['name']}] captured {frame_idx + 1}/{total_capture_frames}", flush=True)
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()

    frames_dir = case_root / "frames" / "a"
    output_video = case_root / f"{case['name']}.mp4"
    convert_video(frames_dir, output_video)


def main() -> None:
    sanitize_proxy_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        print(f"Running {case['name']}", flush=True)
        run_case(case)
        print(f"Completed {case['name']}", flush=True)
    build_html()
    print(OUTPUT_ROOT / "index.html", flush=True)


if __name__ == "__main__":
    main()
