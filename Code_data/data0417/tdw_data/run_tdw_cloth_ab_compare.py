from pathlib import Path
from typing import Dict, List
import os
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.obi_data.cloth.sheet_type import SheetType
from tdw.obi_data.collision_materials.collision_material import CollisionMaterial


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_cloth_ab_compare")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1076
BUILD_BOOT_WAIT = 12
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "mm_craftroom_1a",
    "skybox": "kiara_1_dawn_4k",
    "camera_position": {"x": -0.6156362579862034, "y": 1.85, "z": -1.6914467174146353},
    "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
    "field_of_view": 78,
}

COMPARE_CASES: List[Dict[str, object]] = [
    {
        "case_name": "compare_center_bridge",
        "cloth_material": "cotton",
        "sheet_type": SheetType.cloth_hd,
        "position": {"x": -0.12, "y": 1.38, "z": -0.04},
        "rotation": {"x": 24.0, "y": 26.0, "z": -18.0},
        "frames": 180,
        "impulse_force": {"x": 0.22, "y": -0.08, "z": 0.18},
        "impulse_torque": {"x": 0.0, "y": 0.42, "z": 0.28},
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.1, "y": 0.16, "z": -0.02}, "rotation": {"x": 0.0, "y": 12.0, "z": 0.0}},
            {"model_name": "camera_box", "position": {"x": 0.34, "y": 0.16, "z": 0.18}, "rotation": {"x": 0.0, "y": -16.0, "z": 0.0}},
        ],
    },
    {
        "case_name": "compare_front_slide",
        "cloth_material": "cotton",
        "sheet_type": SheetType.cloth_hd,
        "position": {"x": 0.2, "y": 1.42, "z": -0.24},
        "rotation": {"x": 28.0, "y": -10.0, "z": 26.0},
        "frames": 180,
        "impulse_force": {"x": -0.16, "y": -0.06, "z": 0.24},
        "impulse_torque": {"x": 0.12, "y": -0.36, "z": 0.32},
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.14, "y": 0.16, "z": 0.06}, "rotation": {"x": 0.0, "y": 8.0, "z": 0.0}},
            {"model_name": "camera_box", "position": {"x": 0.3, "y": 0.16, "z": -0.08}, "rotation": {"x": 0.0, "y": -14.0, "z": 0.0}},
        ],
    },
]

STABLE_MODE: Dict[str, object] = {
    "name": "stable",
    "substeps": 8,
    "physics_iterations": 16,
    "capture_warmup": 0,
    "floor_material": CollisionMaterial(dynamic_friction=0.55, static_friction=0.6, stickiness=0.0, stick_distance=0.0),
}


def sanitize_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def launch_build() -> subprocess.Popen:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT.joinpath("build.log")
    log_fp = open(log_path, "ab")
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    return subprocess.Popen([str(BUILD_PATH), f"-port={PORT}", "-address=localhost", "-force-glcore42"],
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


def run_stable_variant(case: Dict[str, object]) -> None:
    sanitize_proxy_env()
    build_proc = launch_build()
    time.sleep(BUILD_BOOT_WAIT)
    c = Controller(launch_build=False, port=PORT)
    try:
        case_root = OUTPUT_ROOT.joinpath(str(case["case_name"]), str(STABLE_MODE["name"]))
        obi = Obi(output_data=True, floor_material=STABLE_MODE["floor_material"])
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=SCENE["camera_position"],
                                   look_at=SCENE["look_at"],
                                   field_of_view=int(SCENE["field_of_view"]))
        capture = ImageCapture(path=case_root, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        lighting = InteriorSceneLighting(hdri_skybox=str(SCENE["skybox"]),
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        c.add_ons.extend([lighting, camera, capture, obi])
        commands = [{"$type": "set_screen_size",
                     "width": 1280,
                     "height": 720},
                    {"$type": "set_physics_solver_iterations",
                     "iterations": int(STABLE_MODE["physics_iterations"])},
                    Controller.get_add_scene(scene_name=str(SCENE["name"])),
                    Controller.get_add_hdri_skybox(skybox_name=str(SCENE["skybox"]))]
        for static_object in case["static_objects"]:
            static_id = c.get_unique_id()
            commands.extend(c.get_add_physics_object(model_name=str(static_object["model_name"]),
                                                     object_id=static_id,
                                                     library="models_core.json",
                                                     position=static_object["position"],
                                                     rotation=static_object["rotation"],
                                                     kinematic=True,
                                                     gravity=False))
        cloth_id = c.get_unique_id()
        obi.set_solver(solver_id=0, substeps=int(STABLE_MODE["substeps"]), scale_factor=1)
        obi.create_cloth_sheet(cloth_material=str(case["cloth_material"]),
                               object_id=cloth_id,
                               sheet_type=case["sheet_type"],
                               position=case["position"],
                               rotation=case["rotation"])
        c.communicate(commands)
        for _ in range(int(STABLE_MODE["capture_warmup"])):
            c.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        obi.apply_force_to_cloth(object_id=cloth_id,
                                 force=case["impulse_force"],
                                 torque=case["impulse_torque"])
        c.communicate([])
        for _ in range(int(case["frames"])):
            c.communicate([])
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()

    convert_video(OUTPUT_ROOT.joinpath(str(case["case_name"]), str(STABLE_MODE["name"]), "a"),
                  OUTPUT_ROOT.joinpath(str(case["case_name"]), f"{STABLE_MODE['name']}.mp4"))


def main() -> None:
    sanitize_proxy_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case in COMPARE_CASES:
        case_dir = OUTPUT_ROOT.joinpath(str(case["case_name"]))
        case_dir.mkdir(parents=True, exist_ok=True)
        output_video = case_dir.joinpath(f"{STABLE_MODE['name']}.mp4")
        if output_video.exists():
            print(f"Skipping completed {case['case_name']} / {STABLE_MODE['name']}", flush=True)
            continue
        print(f"Running {case['case_name']} / {STABLE_MODE['name']}", flush=True)
        run_stable_variant(case=case)
        print(f"Completed {case['case_name']} / {STABLE_MODE['name']}", flush=True)


if __name__ == "__main__":
    main()
