from pathlib import Path
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


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_real_scene_cloth_drop_stable")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = int(os.environ.get("TDW_PORT", "1121"))
BUILD_BOOT_WAIT = int(os.environ.get("TDW_BUILD_BOOT_WAIT", "12"))
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "suburb_scene_2023",
    "skybox": "sunset_fairway_4k",
    "camera_position": {"x": -2.15, "y": 1.55, "z": 1.05},
    "look_at": {"x": 0.0, "y": 0.95, "z": 0.0},
    "field_of_view": 68,
}

CASE = {
    "name": "soft_cloth_drop_stable",
    "cloth_material": "cotton",
    "sheet_type": SheetType.cloth_hd,
    "substeps": 20,
    "position": {"x": 0.0, "y": 2.1, "z": 0.0},
    "rotation": {"x": 18.0, "y": 8.0, "z": -10.0},
    "frames": 240,
    "static_objects": [
        {"model_name": "camera_box",
         "position": {"x": 0.0, "y": 0.16, "z": 0.0},
         "rotation": {"x": 0.0, "y": 6.0, "z": 0.0},
         "scale_factor": {"x": 2.0, "y": 0.85, "z": 2.0}},
    ],
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


def main() -> None:
    sanitize_proxy_env()
    build_proc = launch_build()
    time.sleep(BUILD_BOOT_WAIT)
    c = Controller(launch_build=False, check_version=False, port=PORT)
    try:
        case_root = OUTPUT_ROOT.joinpath(CASE["name"])
        floor_material = CollisionMaterial(dynamic_friction=0.55,
                                           static_friction=0.6,
                                           stickiness=0.0,
                                           stick_distance=0.0)
        obi = Obi(output_data=True, floor_material=floor_material)
        lighting = InteriorSceneLighting(hdri_skybox=str(SCENE["skybox"]),
                                         aperture=8.0,
                                         focus_distance=4.5,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        c.add_ons.extend([lighting, obi])
        init_commands = [{"$type": "set_screen_size",
                          "width": 1280,
                          "height": 720},
                         {"$type": "set_physics_solver_iterations",
                          "iterations": 20},
                         Controller.get_add_scene(scene_name=str(SCENE["name"])),
                         Controller.get_add_hdri_skybox(skybox_name=str(SCENE["skybox"]))]
        c.communicate(init_commands)
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=SCENE["camera_position"],
                                   look_at=SCENE["look_at"],
                                   field_of_view=int(SCENE["field_of_view"]))
        capture = ImageCapture(path=case_root, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        c.add_ons.extend([camera, capture])
        static_commands = []
        for static_object in CASE["static_objects"]:
            static_id = c.get_unique_id()
            static_commands.extend(c.get_add_physics_object(model_name=str(static_object["model_name"]),
                                                            object_id=static_id,
                                                            library="models_core.json",
                                                            position=static_object["position"],
                                                            rotation=static_object["rotation"],
                                                            scale_factor=static_object["scale_factor"],
                                                            kinematic=True,
                                                            gravity=False))
        if static_commands:
            c.communicate(static_commands)

        cloth_id = c.get_unique_id()
        obi.set_solver(solver_id=0, substeps=int(CASE["substeps"]), scale_factor=1)
        obi.create_cloth_sheet(cloth_material=str(CASE["cloth_material"]),
                               object_id=cloth_id,
                               sheet_type=CASE["sheet_type"],
                               position=CASE["position"],
                               rotation=CASE["rotation"])
        c.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        obi.apply_force_to_cloth(object_id=cloth_id,
                                 force={"x": 0.08, "y": -0.02, "z": 0.05},
                                 torque={"x": 0.0, "y": 0.1, "z": 0.04})
        c.communicate([])
        for _ in range(4):
            c.communicate([])
        for _ in range(int(CASE["frames"])):
            c.communicate([])
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()

    convert_video(OUTPUT_ROOT.joinpath(CASE["name"], "a"),
                  OUTPUT_ROOT.joinpath(f"{CASE['name']}.mp4"))
    print(OUTPUT_ROOT.joinpath(f"{CASE['name']}.mp4"))


if __name__ == "__main__":
    main()
