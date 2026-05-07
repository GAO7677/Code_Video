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
from manual_tdw_controller import ManualBuildController


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_nonfluid_soft_bodies_multi_scene/mm_craftroom_1a")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1101
BUILD_BOOT_WAIT = 12
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "mm_craftroom_1a",
    "skybox": "kiara_1_dawn_4k",
    "camera_position": {"x": -3.4, "y": 2.15, "z": -0.35},
    "look_at": {"x": 0.0, "y": 0.82, "z": 0.02},
    "field_of_view": 72,
}

CASE = {
    "name": "soft_cloth_canvas_drape",
    "cloth_material": "canvas",
    "sheet_type": SheetType.cloth_hd,
    "substeps": 8,
    "position": {"x": 0.0, "y": 2.45, "z": 0.02},
    "rotation": {"x": 18.0, "y": 8.0, "z": -12.0},
    "frames": 240,
    "static_objects": [
        {"model_name": "camera_box", "position": {"x": -0.34, "y": 0.16, "z": 0.06}, "rotation": {"x": 0.0, "y": 12.0, "z": 0.0}},
        {"model_name": "camera_box", "position": {"x": 0.34, "y": 0.16, "z": -0.06}, "rotation": {"x": 0.0, "y": -10.0, "z": 0.0}},
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
    build_holder = {}
    print(f"[phase] bind_and_launch port={PORT}", flush=True)
    c = ManualBuildController(port=PORT, build_callback=lambda: build_holder.setdefault("proc", launch_build()))
    time.sleep(BUILD_BOOT_WAIT)
    print(f"[phase] connect_controller port={PORT}", flush=True)
    try:
        case_root = OUTPUT_ROOT.joinpath(CASE["name"])
        floor_material = CollisionMaterial(dynamic_friction=0.55,
                                           static_friction=0.6,
                                           stickiness=0.0,
                                           stick_distance=0.0)
        obi = Obi(output_data=True, floor_material=floor_material)
        lighting = InteriorSceneLighting(hdri_skybox=str(SCENE["skybox"]),
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        c.add_ons.extend([lighting, obi])
        init_commands = [{"$type": "set_screen_size",
                          "width": 1280,
                          "height": 720},
                         {"$type": "set_physics_solver_iterations",
                          "iterations": 16},
                         Controller.get_add_scene(scene_name=str(SCENE["name"])),
                         Controller.get_add_hdri_skybox(skybox_name=str(SCENE["skybox"]))]
        print("[phase] init_scene", flush=True)
        c.communicate(init_commands)
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=SCENE["camera_position"],
                                   look_at=SCENE["look_at"],
                                   field_of_view=int(SCENE["field_of_view"]))
        capture = ImageCapture(path=case_root, avatar_ids=["a"], pass_masks=["_img"])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        c.add_ons.extend([camera, capture])

        static_commands = []
        for static_object in CASE["static_objects"]:
            static_id = c.get_unique_id()
            static_commands.extend(c.get_add_physics_object(model_name=str(static_object["model_name"]),
                                                            object_id=static_id,
                                                            library="models_core.json",
                                                            position=static_object["position"],
                                                            rotation=static_object["rotation"],
                                                            kinematic=True,
                                                            gravity=False))
        if static_commands:
            print("[phase] add_static_objects", flush=True)
            c.communicate(static_commands)

        print("[phase] create_cloth", flush=True)
        cloth_id = c.get_unique_id()
        obi.set_solver(solver_id=0, substeps=int(CASE["substeps"]), scale_factor=1)
        obi.create_cloth_sheet(cloth_material=str(CASE["cloth_material"]),
                               object_id=cloth_id,
                               sheet_type=CASE["sheet_type"],
                               position=CASE["position"],
                               rotation=CASE["rotation"])

        print("[phase] init_and_simulate", flush=True)
        c.communicate([])
        for _ in range(int(CASE["frames"])):
            c.communicate([])
        print("[phase] terminate", flush=True)
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_holder["proc"].wait(timeout=10)
        except Exception:
            build_holder["proc"].kill()

    print("[phase] convert", flush=True)
    convert_video(OUTPUT_ROOT.joinpath(CASE["name"], "a"),
                  OUTPUT_ROOT.joinpath(f"{CASE['name']}.mp4"))
    print("[phase] done", flush=True)


if __name__ == "__main__":
    main()
