from pathlib import Path
from typing import Dict, List
import os
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_multi_scene_cases")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1071

SCENES: List[Dict[str, object]] = [
    {
        "name": "building_site",
        "skybox": "bergen_4k",
        "camera_position": {"x": -4.8, "y": 2.4, "z": 4.8},
        "look_at": {"x": 0.0, "y": 0.6, "z": 0.0},
        "field_of_view": 72,
    },
    {
        "name": "box_room_2018",
        "skybox": "lookout_4k",
        "camera_position": {"x": -4.4, "y": 2.2, "z": 4.4},
        "look_at": {"x": 0.0, "y": 0.6, "z": 0.0},
        "field_of_view": 70,
    },
    {
        "name": "tdw_room",
        "skybox": "industrial_sunset_4k",
        "camera_position": {"x": -4.2, "y": 2.0, "z": 4.2},
        "look_at": {"x": 0.0, "y": 0.55, "z": 0.0},
        "field_of_view": 68,
    },
    {
        "name": "suburb_scene_2023",
        "skybox": "sunset_fairway_4k",
        "camera_position": {"x": -5.6, "y": 2.6, "z": 5.6},
        "look_at": {"x": 0.0, "y": 0.7, "z": 0.0},
        "field_of_view": 72,
    },
    {
        "name": "suburb_scene_2018",
        "skybox": "noon_grass_4k",
        "camera_position": {"x": -5.2, "y": 2.5, "z": 5.2},
        "look_at": {"x": 0.0, "y": 0.7, "z": 0.0},
        "field_of_view": 72,
    },
    {
        "name": "mm_craftroom_1a",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -4.0, "y": 2.1, "z": 4.0},
        "look_at": {"x": 0.0, "y": 0.6, "z": 0.0},
        "field_of_view": 70,
    },
    {
        "name": "mm_kitchen_2b",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -4.0, "y": 2.1, "z": 4.0},
        "look_at": {"x": 0.0, "y": 0.6, "z": 0.0},
        "field_of_view": 70,
    },
]

CASES: List[Dict[str, object]] = [
    {
        "case_name": "case000_static_center",
        "position": {"x": 0.0, "y": 0.02, "z": 0.0},
        "velocity": {"x": 0.45, "y": 0.0, "z": 0.18},
        "angular_velocity": {"x": 0.0, "y": 0.55, "z": 0.15},
        "frames": 90,
    },
    {
        "case_name": "case003_high_drop",
        "position": {"x": 0.0, "y": 1.7, "z": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular_velocity": {"x": 0.3, "y": 0.4, "z": 0.2},
        "frames": 120,
    },
    {
        "case_name": "case900_random_parabola",
        "position": {"x": -0.15, "y": 0.95, "z": -0.1},
        "velocity": {"x": 0.8, "y": 0.9, "z": 0.55},
        "angular_velocity": {"x": 0.8, "y": 0.5, "z": 1.2},
        "frames": 120,
    },
    {
        "case_name": "case005_entry_left",
        "position": {"x": 1.25, "y": 0.45, "z": 0.0},
        "velocity": {"x": -1.35, "y": 0.0, "z": 0.0},
        "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.6},
        "frames": 120,
    },
]


def launch_build(scene_output_root: Path) -> subprocess.Popen:
    log_path = scene_output_root.joinpath("build.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "ab")
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    return subprocess.Popen([str(BUILD_PATH), f"-port={PORT}", "-address=localhost", "-force-glcore42"],
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            env=env)


def run_case(scene: Dict[str, object], case: Dict[str, object]) -> None:
    scene_name = str(scene["name"])
    skybox_name = str(scene["skybox"])
    scene_output_root = OUTPUT_ROOT.joinpath(scene_name)
    build_proc = launch_build(scene_output_root=scene_output_root)
    time.sleep(6)
    controller = Controller(launch_build=False, port=PORT)
    try:
        case_dir = scene_output_root.joinpath(str(case["case_name"]))
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=scene["camera_position"],
                                   look_at=scene["look_at"],
                                   field_of_view=int(scene["field_of_view"]))
        capture = ImageCapture(path=case_dir, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        lighting = InteriorSceneLighting(hdri_skybox=skybox_name,
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        controller.add_ons.extend([lighting, camera, capture])
        object_id = controller.get_unique_id()
        controller.communicate([{"$type": "set_screen_size",
                                 "width": 1280,
                                 "height": 720},
                                Controller.get_add_scene(scene_name=scene_name),
                                Controller.get_add_hdri_skybox(skybox_name=skybox_name)])
        for _ in range(10):
            controller.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        commands = controller.get_add_physics_object(model_name="iron_box",
                                                     object_id=object_id,
                                                     position=case["position"])
        commands.extend([{"$type": "set_velocity",
                          "id": object_id,
                          "velocity": case["velocity"]},
                         {"$type": "set_angular_velocity",
                          "id": object_id,
                          "angular_velocity": case["angular_velocity"]}])
        controller.communicate(commands)
        for _ in range(int(case["frames"])):
            controller.communicate([])
        controller.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()


def convert_scene_videos(scene_name: str) -> None:
    scene_root = OUTPUT_ROOT.joinpath(scene_name)
    for case in CASES:
        case_name = str(case["case_name"])
        frames_dir = scene_root.joinpath(case_name, "a")
        output_video = scene_root.joinpath(f"{case_name}.mp4")
        if not frames_dir.exists():
            continue
        subprocess.run(["/home/gaoya/.venvs/tdw/bin/python",
                        "/home/gaoya/Code_Video/tdw-master/Python/example_controllers/frames_to_video.py",
                        str(frames_dir),
                        str(output_video),
                        "24"],
                       check=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        scene_name = str(scene["name"])
        scene_root = OUTPUT_ROOT.joinpath(scene_name)
        completed = all(scene_root.joinpath(f"{case['case_name']}.mp4").exists() for case in CASES)
        if completed:
            print(f"Skipping completed scene {scene_name}")
            continue
        for case in CASES:
            print(f"Running {scene['name']} / {case['case_name']}")
            run_case(scene=scene, case=case)
            print(f"Completed {scene['name']} / {case['case_name']}")
        convert_scene_videos(scene_name=scene_name)
        print(f"Converted videos for {scene_name}")


if __name__ == "__main__":
    main()
