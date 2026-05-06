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
from tdw.obi_data.cloth.volume_type import ClothVolumeType
from tdw.obi_data.collision_materials.collision_material import CollisionMaterial
from tdw.obi_data.fluids.disk_emitter import DiskEmitter
from tdw.obi_data.fluids.fluid import Fluid


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_real_scene_soft_materials")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1077
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
        "name": "soft_material_water_pour",
        "type": "fluid",
        "fluid": Fluid(capacity=1800,
                        resolution=0.9,
                        color={"r": 0.96, "g": 0.98, "b": 1.0, "a": 0.9},
                        rest_density=1000,
                        reflection=0.35,
                        transparency=0.08,
                        refraction=0.02,
                        smoothing=2.8,
                        render_smoothness=1.0,
                        metalness=0.0,
                        viscosity=0.015,
                        vorticity=0.25,
                        surface_tension=1.6,
                        absorption=1.5,
                        radius_scale=1.35,
                        random_velocity=0.035),
        "shape": DiskEmitter(radius=0.11),
        "position": {"x": -0.22, "y": 2.08, "z": -0.62},
        "rotation": {"x": 78.0, "y": 18.0, "z": 0.0},
        "speed": 4.6,
        "lifespan": 8,
        "substeps": 6,
        "frames": 260,
        "receptacle": {
            "model_name": "fluid_receptacle1x1",
            "library": "models_special.json",
            "position": {"x": 0.12, "y": 0.0, "z": -0.06},
            "scale_factor": {"x": 1.55, "y": 1.55, "z": 1.55},
        },
    },
    {
        "name": "soft_material_gravel_pour",
        "type": "granular",
        "fluid": "gravel",
        "shape": DiskEmitter(radius=0.12),
        "position": {"x": 0.18, "y": 1.85, "z": -0.48},
        "rotation": {"x": 78.0, "y": -20.0, "z": 0.0},
        "speed": 2.3,
        "lifespan": 10,
        "substeps": 3,
        "frames": 220,
        "receptacle": {
            "model_name": "fluid_receptacle1x1",
            "library": "models_special.json",
            "position": {"x": -0.05, "y": 0.0, "z": 0.08},
            "scale_factor": {"x": 1.2, "y": 1.2, "z": 1.2},
        },
    },
    {
        "name": "soft_volume_canvas_sphere_drop",
        "type": "cloth_volume",
        "cloth_material": "canvas",
        "volume_type": ClothVolumeType.sphere,
        "position": {"x": -0.05, "y": 1.28, "z": -0.05},
        "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        "scale_factor": {"x": 0.42, "y": 0.42, "z": 0.42},
        "pressure": 2.6,
        "frames": 180,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": 0.22, "y": 0.16, "z": 0.12}, "rotation": {"x": 0.0, "y": -10.0, "z": 0.0}},
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
        obi = Obi(output_data=True,
                  floor_material=CollisionMaterial(dynamic_friction=0.55,
                                                   static_friction=0.6,
                                                   stickiness=0.0,
                                                   stick_distance=0.0))
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

        if demo["type"] in {"fluid", "granular"}:
            receptacle = demo["receptacle"]
            receptacle_id = c.get_unique_id()
            commands.extend(c.get_add_physics_object(model_name=str(receptacle["model_name"]),
                                                     object_id=receptacle_id,
                                                     library=str(receptacle["library"]),
                                                     position=receptacle["position"],
                                                     kinematic=True,
                                                     gravity=False,
                                                     scale_factor=receptacle["scale_factor"]))
            obi.set_solver(substeps=int(demo["substeps"]))
            fluid_id = c.get_unique_id()
            obi.create_fluid(fluid=demo["fluid"],
                             shape=demo["shape"],
                             object_id=fluid_id,
                             position=demo["position"],
                             rotation=demo["rotation"],
                             speed=float(demo["speed"]),
                             lifespan=float(demo["lifespan"]))
        elif demo["type"] == "cloth_volume":
            cloth_id = c.get_unique_id()
            obi.set_solver(substeps=4)
            obi.create_cloth_volume(cloth_material=str(demo["cloth_material"]),
                                    object_id=cloth_id,
                                    volume_type=demo["volume_type"],
                                    position=demo["position"],
                                    rotation=demo["rotation"],
                                    scale_factor=demo["scale_factor"],
                                    pressure=float(demo["pressure"]))
        else:
            raise ValueError(f"Unsupported demo type: {demo['type']}")

        c.communicate(commands)
        for _ in range(8):
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
