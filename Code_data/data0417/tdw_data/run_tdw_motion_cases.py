from pathlib import Path
from typing import Dict, List
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_motion_cases")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1071


def launch_build() -> subprocess.Popen:
    log_path = OUTPUT_ROOT.joinpath("build.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = open(log_path, "ab")
    env = dict(**__import__("os").environ)
    env["DISPLAY"] = DISPLAY
    return subprocess.Popen([str(BUILD_PATH), f"-port={PORT}", "-address=localhost", "-force-glcore42"],
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            env=env)


def run_case(case_name: str,
             position: Dict[str, float],
             velocity: Dict[str, float],
             angular_velocity: Dict[str, float],
             frames: int = 120,
             model_name: str = "iron_box") -> None:
    build_proc = launch_build()
    time.sleep(6)
    controller = Controller(launch_build=False, port=PORT)
    try:
        case_dir = OUTPUT_ROOT.joinpath(case_name)
        cam = ThirdPersonCamera(avatar_id="a",
                                position={"x": 2.8, "y": 1.8, "z": -3.2},
                                look_at={"x": 0, "y": 0.7, "z": 0})
        capture = ImageCapture(path=case_dir, avatar_ids=["a"], pass_masks=["_img"])
        controller.add_ons.extend([cam, capture])
        object_id = controller.get_unique_id()
        commands = [TDWUtils.create_empty_room(12, 12)]
        commands.extend(controller.get_add_physics_object(model_name=model_name,
                                                          object_id=object_id,
                                                          position=position))
        commands.append({"$type": "set_velocity",
                         "id": object_id,
                         "velocity": velocity})
        commands.append({"$type": "set_angular_velocity",
                         "id": object_id,
                         "angular_velocity": angular_velocity})
        controller.communicate(commands)
        for _ in range(frames):
            controller.communicate([])
        controller.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases: List[Dict] = [
        {
            "case_name": "case000_static_center",
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "frames": 90,
        },
        {
            "case_name": "case003_high_drop",
            "position": {"x": 0.1, "y": 2.1, "z": 0.0},
            "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.3, "y": 0.4, "z": 0.2},
            "frames": 120,
        },
        {
            "case_name": "case900_random_parabola",
            "position": {"x": 0.0, "y": 1.1, "z": 0.0},
            "velocity": {"x": 1.6, "y": 0.4, "z": 0.9},
            "angular_velocity": {"x": 0.8, "y": 0.5, "z": 1.2},
            "frames": 120,
        },
        {
            "case_name": "case005_entry_left",
            "position": {"x": 1.8, "y": 0.6, "z": 0.0},
            "velocity": {"x": -2.2, "y": 0.0, "z": 0.0},
            "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.6},
            "frames": 120,
        },
    ]
    for case in cases:
        print(f"Running {case['case_name']}")
        run_case(**case)
        print(f"Completed {case['case_name']}")


if __name__ == "__main__":
    main()
