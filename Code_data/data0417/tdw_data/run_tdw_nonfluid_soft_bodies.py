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
from tdw.obi_data.cloth.volume_type import ClothVolumeType
from tdw.obi_data.collision_materials.collision_material import CollisionMaterial
from tdw.obi_data.wind_source import WindSource


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_nonfluid_soft_bodies")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1085
BUILD_BOOT_WAIT = 12
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENE = {
    "name": "mm_craftroom_1a",
    "skybox": "kiara_1_dawn_4k",
    "camera_position": {"x": -0.78, "y": 1.92, "z": -2.08},
    "look_at": {"x": 0.02, "y": 0.95, "z": 0.0},
    "field_of_view": 74,
}

DEMOS: List[Dict[str, object]] = [
    {
        "name": "soft_cloth_canvas_drape",
        "type": "cloth_sheet",
        "cloth_material": "canvas",
        "sheet_type": SheetType.cloth_hd,
        "substeps": 6,
        "position": {"x": 0.02, "y": 2.08, "z": -0.02},
        "rotation": {"x": 10.0, "y": 6.0, "z": -8.0},
        "frames": 220,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.22, "y": 0.16, "z": 0.0}, "rotation": {"x": 0.0, "y": 10.0, "z": 0.0}},
            {"model_name": "camera_box", "position": {"x": 0.2, "y": 0.16, "z": -0.02}, "rotation": {"x": 0.0, "y": -12.0, "z": 0.0}},
        ],
    },
    {
        "name": "soft_cloth_silk_wind",
        "type": "cloth_sheet_wind",
        "cloth_material": "silk",
        "sheet_type": SheetType.cloth,
        "substeps": 6,
        "position": {"x": -0.48, "y": 1.72, "z": 0.06},
        "rotation": {"x": -90.0, "y": 90.0, "z": 0.0},
        "frames": 220,
        "wind": {
            "position": {"x": 0.72, "y": 0.65, "z": 0.04},
            "rotation": {"x": 0.0, "y": -90.0, "z": 0.0},
            "speed": 12.0,
            "capacity": 2400,
            "lifespan": 2.0,
            "emitter_radius": 0.65,
            "smoothing": 0.6,
        },
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.08, "y": 0.16, "z": -0.16}, "rotation": {"x": 0.0, "y": 4.0, "z": 0.0}},
        ],
    },
    {
        "name": "soft_volume_rubber_sphere_drop",
        "type": "cloth_volume",
        "cloth_material": "rubber",
        "volume_type": ClothVolumeType.sphere,
        "substeps": 6,
        "position": {"x": -0.04, "y": 1.42, "z": 0.0},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "scale_factor": {"x": 0.5, "y": 0.5, "z": 0.5},
        "pressure": 3.2,
        "frames": 200,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": 0.22, "y": 0.16, "z": 0.08}, "rotation": {"x": 0.0, "y": -8.0, "z": 0.0}},
        ],
    },
    {
        "name": "soft_volume_wool_cube_bounce",
        "type": "cloth_volume",
        "cloth_material": "wool",
        "volume_type": ClothVolumeType.cube,
        "substeps": 8,
        "position": {"x": 0.18, "y": 1.58, "z": -0.12},
        "rotation": {"x": 12.0, "y": 16.0, "z": 4.0},
        "scale_factor": {"x": 0.46, "y": 0.46, "z": 0.46},
        "pressure": 2.4,
        "frames": 220,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.16, "y": 0.16, "z": 0.1}, "rotation": {"x": 0.0, "y": 10.0, "z": 0.0}},
            {"model_name": "camera_box", "position": {"x": 0.26, "y": 0.16, "z": -0.05}, "rotation": {"x": 0.0, "y": -18.0, "z": 0.0}},
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
        floor_material = CollisionMaterial(dynamic_friction=0.55,
                                           static_friction=0.6,
                                           stickiness=0.0,
                                           stick_distance=0.0)
        obi = Obi(output_data=True, floor_material=floor_material)
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
        c.add_ons.extend([lighting, camera, capture, obi])
        commands = [{"$type": "set_screen_size",
                     "width": 1280,
                     "height": 720},
                    {"$type": "set_physics_solver_iterations",
                     "iterations": 16},
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

        obi.set_solver(solver_id=0, substeps=int(demo["substeps"]), scale_factor=1)

        if demo["type"] == "cloth_sheet":
            cloth_id = c.get_unique_id()
            obi.create_cloth_sheet(cloth_material=str(demo["cloth_material"]),
                                   object_id=cloth_id,
                                   sheet_type=demo["sheet_type"],
                                   position=demo["position"],
                                   rotation=demo["rotation"])
        elif demo["type"] == "cloth_sheet_wind":
            cloth_id = c.get_unique_id()
            wind = demo["wind"]
            wind_id = c.get_unique_id()
            obi.wind_sources[wind_id] = WindSource(wind_id=wind_id,
                                                   position=wind["position"],
                                                   rotation=wind["rotation"],
                                                   emitter_radius=float(wind["emitter_radius"]),
                                                   capacity=int(wind["capacity"]),
                                                   speed=float(wind["speed"]),
                                                   lifespan=float(wind["lifespan"]),
                                                   smoothing=float(wind["smoothing"]))
            obi.create_cloth_sheet(cloth_material=str(demo["cloth_material"]),
                                   object_id=cloth_id,
                                   sheet_type=demo["sheet_type"],
                                   position=demo["position"],
                                   rotation=demo["rotation"])
        elif demo["type"] == "cloth_volume":
            volume_id = c.get_unique_id()
            obi.create_cloth_volume(cloth_material=str(demo["cloth_material"]),
                                    object_id=volume_id,
                                    volume_type=demo["volume_type"],
                                    position=demo["position"],
                                    rotation=demo["rotation"],
                                    scale_factor=demo["scale_factor"],
                                    pressure=float(demo["pressure"]))
        else:
            raise ValueError(f"Unsupported demo type: {demo['type']}")

        c.communicate(commands)
        for _ in range(12):
            c.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
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
