from pathlib import Path
import os
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_obi_cloth_demo")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1073
FRAMES = 180


def launch_build() -> subprocess.Popen:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_ROOT.joinpath("build.log")
    log_fp = open(log_path, "ab")
    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    return subprocess.Popen([str(BUILD_PATH), f"-port={PORT}", "-address=localhost", "-force-glcore42"],
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            env=env)


def main() -> None:
    build_proc = launch_build()
    time.sleep(12)
    c = Controller(launch_build=False, port=PORT)
    try:
        camera = ThirdPersonCamera(avatar_id="a",
                                   position={"x": -3.4, "y": 1.7, "z": -0.3},
                                   look_at={"x": 0.0, "y": 1.0, "z": 0.0},
                                   field_of_view=72)
        capture = ImageCapture(path=OUTPUT_ROOT, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        obi = Obi()
        c.add_ons.extend([camera, capture, obi])
        cloth_id = c.get_unique_id()
        sphere_id = c.get_unique_id()
        obi.create_cloth_sheet(cloth_material="cotton",
                               object_id=cloth_id,
                               position={"x": 0.0, "y": 2.2, "z": 0.0},
                               rotation={"x": 10.0, "y": 0.0, "z": 0.0})
        commands = [TDWUtils.create_empty_room(12, 12),
                    {"$type": "set_screen_size",
                     "width": 1280,
                     "height": 720}]
        commands.extend(Controller.get_add_physics_object(model_name="sphere",
                                                          object_id=sphere_id,
                                                          library="models_flex.json",
                                                          position={"x": 0.0, "y": 0.55, "z": 0.0},
                                                          kinematic=True,
                                                          gravity=False,
                                                          scale_factor={"x": 0.7, "y": 0.7, "z": 0.7}))
        c.communicate(commands)
        for _ in range(10):
            c.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        for _ in range(FRAMES):
            c.communicate([])
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()

    frames_dir = OUTPUT_ROOT.joinpath("a")
    output_video = OUTPUT_ROOT.joinpath("obi_cloth_sheet.mp4")
    subprocess.run(["/home/gaoya/.venvs/tdw/bin/python",
                    "/home/gaoya/Code_Video/tdw-master/Python/example_controllers/frames_to_video.py",
                    str(frames_dir),
                    str(output_video),
                    "24"],
                   check=True)
    print(output_video)


if __name__ == "__main__":
    main()
