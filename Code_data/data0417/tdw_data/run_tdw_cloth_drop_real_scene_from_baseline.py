from pathlib import Path
import os
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller

from manual_tdw_controller import ManualBuildController


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_cloth_drop_real_scene_from_baseline")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = int(os.environ.get("TDW_PORT", "1141"))
BUILD_BOOT_WAIT = int(os.environ.get("TDW_BUILD_BOOT_WAIT", "18"))
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "suburb_scene_2023",
    "skybox": "sunset_fairway_4k",
    "camera_position": {"x": -3.4, "y": 1.7, "z": -0.3},
    "look_at": {"x": 0.0, "y": 1.0, "z": 0.0},
    "field_of_view": 72,
}

CASES = [
    {
        "name": "sphere_on_ground_real_scene",
        "sphere_position": {"x": 0.0, "y": 0.35, "z": 0.0},
        "pedestal": None,
    },
    {
        "name": "sphere_on_pedestal_real_scene",
        "sphere_position": {"x": 0.0, "y": 0.9, "z": 0.0},
        "pedestal": {
            "position": {"x": 0.0, "y": 0.18, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "scale_factor": {"x": 1.9, "y": 0.7, "z": 1.9},
        },
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


def run_case(case: dict) -> None:
    sanitize_proxy_env()
    build_holder = {}
    c = ManualBuildController(port=PORT, build_callback=lambda: build_holder.setdefault("proc", launch_build()))
    time.sleep(BUILD_BOOT_WAIT)
    try:
        case_root = OUTPUT_ROOT.joinpath(case["name"])
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=SCENE["camera_position"],
                                   look_at=SCENE["look_at"],
                                   field_of_view=int(SCENE["field_of_view"]))
        capture = ImageCapture(path=case_root, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        lighting = InteriorSceneLighting(hdri_skybox=SCENE["skybox"],
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        obi = Obi()
        c.add_ons.extend([lighting, camera, capture, obi])

        cloth_id = c.get_unique_id()
        sphere_id = c.get_unique_id()
        obi.create_cloth_sheet(cloth_material="cotton",
                               object_id=cloth_id,
                               position={"x": 0.0, "y": 2.2, "z": 0.0},
                               rotation={"x": 10.0, "y": 0.0, "z": 0.0})
        commands = [{"$type": "set_screen_size", "width": 1280, "height": 720},
                    Controller.get_add_scene(scene_name=SCENE["name"]),
                    Controller.get_add_hdri_skybox(skybox_name=SCENE["skybox"])]

        if case["pedestal"] is not None:
            pedestal_id = c.get_unique_id()
            commands.extend(Controller.get_add_physics_object(model_name="camera_box",
                                                              object_id=pedestal_id,
                                                              library="models_core.json",
                                                              position=case["pedestal"]["position"],
                                                              rotation=case["pedestal"]["rotation"],
                                                              scale_factor=case["pedestal"]["scale_factor"],
                                                              kinematic=True,
                                                              gravity=False))

        commands.extend(Controller.get_add_physics_object(model_name="sphere",
                                                          object_id=sphere_id,
                                                          library="models_flex.json",
                                                          position=case["sphere_position"],
                                                          kinematic=True,
                                                          gravity=False,
                                                          scale_factor={"x": 0.7, "y": 0.7, "z": 0.7}))
        c.communicate(commands)
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        for _ in range(180):
            c.communicate([])
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_holder["proc"].wait(timeout=10)
        except Exception:
            build_holder["proc"].kill()

    convert_video(OUTPUT_ROOT.joinpath(case["name"], "a"),
                  OUTPUT_ROOT.joinpath(f"{case['name']}.mp4"))


def main() -> None:
    sanitize_proxy_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        print(f"Running {case['name']}", flush=True)
        run_case(case)
        print(f"Completed {case['name']}", flush=True)


if __name__ == "__main__":
    main()
