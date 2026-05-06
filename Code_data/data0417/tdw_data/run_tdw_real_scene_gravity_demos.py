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


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_real_scene_gravity_demos")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1074
BUILD_BOOT_WAIT = 12
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "mm_craftroom_1a",
    "skybox": "kiara_1_dawn_4k",
    "camera_position": {"x": -0.6156362579862034, "y": 1.85, "z": -1.6914467174146353},
    "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
    "field_of_view": 78,
}

DEMOS: List[Dict[str, object]] = [
    {
        "name": "real_scene_camera_box_drop",
        "type": "rigid",
        "model_name": "camera_box",
        "position": {"x": 0.0, "y": 1.3, "z": 0.0},
        "velocity": {"x": 0.12, "y": 0.0, "z": 0.05},
        "angular_velocity": {"x": 0.2, "y": 0.55, "z": 0.1},
        "frames": 140,
    },
    {
        "name": "real_scene_obi_cloth_drop",
        "type": "obi_cloth",
        "cloth_material": "cotton",
        "position": {"x": 0.0, "y": 2.15, "z": 0.0},
        "rotation": {"x": 10.0, "y": 0.0, "z": 0.0},
        "frames": 180,
    },
    {
        "name": "real_scene_obi_cloth_prop_cluster",
        "type": "obi_cloth",
        "cloth_material": "cotton",
        "position": {"x": 0.02, "y": 2.28, "z": 0.02},
        "rotation": {"x": 18.0, "y": 28.0, "z": -12.0},
        "frames": 210,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.18, "y": 0.16, "z": -0.05}, "rotation": {"x": 0.0, "y": 18.0, "z": 0.0}},
            {"model_name": "basket_18inx18inx12iin_wicker", "position": {"x": 0.38, "y": 0.22, "z": 0.1}, "rotation": {"x": 0.0, "y": -24.0, "z": 0.0}},
            {"model_name": "hiker_backpack", "position": {"x": 0.1, "y": 0.24, "z": -0.34}, "rotation": {"x": 0.0, "y": 35.0, "z": 0.0}},
        ],
    },
    {
        "name": "real_scene_obi_cloth_left_glide",
        "type": "obi_cloth",
        "cloth_material": "cotton",
        "position": {"x": -0.62, "y": 2.34, "z": -0.18},
        "rotation": {"x": 24.0, "y": 54.0, "z": 20.0},
        "frames": 210,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.08, "y": 0.16, "z": 0.0}, "rotation": {"x": 0.0, "y": 12.0, "z": 0.0}},
            {"model_name": "basket_18inx18inx12iin_wicker", "position": {"x": 0.36, "y": 0.22, "z": 0.22}, "rotation": {"x": 0.0, "y": -32.0, "z": 0.0}},
        ],
    },
    {
        "name": "real_scene_obi_cloth_right_fold",
        "type": "obi_cloth",
        "cloth_material": "cotton",
        "position": {"x": 0.68, "y": 2.32, "z": 0.22},
        "rotation": {"x": -18.0, "y": -42.0, "z": -28.0},
        "frames": 210,
        "static_objects": [
            {"model_name": "hiker_backpack", "position": {"x": -0.22, "y": 0.24, "z": -0.08}, "rotation": {"x": 0.0, "y": 18.0, "z": 0.0}},
            {"model_name": "camera_box", "position": {"x": 0.28, "y": 0.16, "z": 0.22}, "rotation": {"x": 0.0, "y": -20.0, "z": 0.0}},
        ],
    },
    {
        "name": "real_scene_obi_cloth_front_corner",
        "type": "obi_cloth",
        "cloth_material": "cotton",
        "position": {"x": 0.14, "y": 2.42, "z": -0.72},
        "rotation": {"x": 32.0, "y": 0.0, "z": 34.0},
        "frames": 220,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.25, "y": 0.16, "z": -0.02}, "rotation": {"x": 0.0, "y": 14.0, "z": 0.0}},
            {"model_name": "basket_18inx18inx12iin_wicker", "position": {"x": 0.26, "y": 0.22, "z": 0.14}, "rotation": {"x": 0.0, "y": -18.0, "z": 0.0}},
            {"model_name": "hiker_backpack", "position": {"x": 0.05, "y": 0.24, "z": -0.36}, "rotation": {"x": 0.0, "y": 52.0, "z": 0.0}},
        ],
    },
]


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


def run_demo(demo: Dict[str, object]) -> None:
    sanitize_proxy_env()
    build_proc = launch_build()
    time.sleep(BUILD_BOOT_WAIT)
    c = Controller(launch_build=False, port=PORT)
    try:
        demo_root = OUTPUT_ROOT.joinpath(str(demo["name"]))
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=SCENE["camera_position"],
                                   look_at=SCENE["look_at"],
                                   field_of_view=int(SCENE["field_of_view"]))
        capture = ImageCapture(path=demo_root, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        lighting = InteriorSceneLighting(hdri_skybox=str(SCENE["skybox"]),
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        add_ons = [lighting, camera, capture]
        if demo["type"] == "obi_cloth":
            add_ons.append(Obi())
        c.add_ons.extend(add_ons)
        commands = [{"$type": "set_screen_size",
                     "width": 1280,
                     "height": 720},
                    Controller.get_add_scene(scene_name=str(SCENE["name"])),
                    Controller.get_add_hdri_skybox(skybox_name=str(SCENE["skybox"]))]
        for static_object in demo.get("static_objects", []):
            static_id = c.get_unique_id()
            commands.extend(c.get_add_physics_object(model_name=str(static_object["model_name"]),
                                                     object_id=static_id,
                                                     library="models_core.json",
                                                     position=static_object["position"],
                                                     rotation=static_object["rotation"],
                                                     kinematic=True,
                                                     gravity=False))
        if demo["type"] == "rigid":
            object_id = c.get_unique_id()
            commands.extend(c.get_add_physics_object(model_name=str(demo["model_name"]),
                                                     object_id=object_id,
                                                     library="models_core.json",
                                                     position=demo["position"]))
        else:
            cloth_id = c.get_unique_id()
            for add_on in c.add_ons:
                if isinstance(add_on, Obi):
                    add_on.create_cloth_sheet(cloth_material=str(demo["cloth_material"]),
                                              object_id=cloth_id,
                                              position=demo["position"],
                                              rotation=demo["rotation"])
                    break
        c.communicate(commands)
        for _ in range(10):
            c.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        if demo["type"] == "rigid":
            c.communicate([{"$type": "set_velocity",
                            "id": object_id,
                            "velocity": demo["velocity"]},
                           {"$type": "set_angular_velocity",
                            "id": object_id,
                            "angular_velocity": demo["angular_velocity"]}])
        for _ in range(int(demo["frames"])):
            c.communicate([])
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()

    convert_video(OUTPUT_ROOT.joinpath(str(demo["name"]), "a"),
                  OUTPUT_ROOT.joinpath(f"{demo['name']}.mp4"))


def main() -> None:
    sanitize_proxy_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for demo in DEMOS:
        output_video = OUTPUT_ROOT.joinpath(f"{demo['name']}.mp4")
        if output_video.exists():
            print(f"Skipping completed demo {demo['name']}", flush=True)
            continue
        print(f"Running demo {demo['name']}", flush=True)
        run_demo(demo)
        print(f"Completed demo {demo['name']}", flush=True)


if __name__ == "__main__":
    main()
