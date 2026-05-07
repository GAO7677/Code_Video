from pathlib import Path
from typing import Dict, List
import math
import os
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.obi import Obi
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.output_data import OutputData, Raycast
from tdw.obi_data.cloth.sheet_type import SheetType
from tdw.obi_data.cloth.tether_particle_group import TetherParticleGroup
from tdw.obi_data.cloth.tether_type import TetherType
from tdw.obi_data.cloth.volume_type import ClothVolumeType
from tdw.obi_data.collision_materials.collision_material import CollisionMaterial
from tdw.obi_data.wind_source import WindSource
from tdw.scene_data.scene_bounds import SceneBounds


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_nonfluid_soft_bodies_multi_scene")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = int(os.environ.get("TDW_PORT", "1087"))
BUILD_BOOT_WAIT = int(os.environ.get("TDW_BUILD_BOOT_WAIT", "12"))
SCENE_FILTER = {s.strip() for s in os.environ.get("TDW_SCENE_FILTER", "").split(",") if s.strip()}
CASE_FILTER = {s.strip() for s in os.environ.get("TDW_CASE_FILTER", "").split(",") if s.strip()}
PROXY_ENV_KEYS = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]

SCENES: List[Dict[str, object]] = [
    {
        "name": "mm_craftroom_1a",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -4.0, "y": 2.1, "z": 4.0},
        "look_at": {"x": 0.0, "y": 0.85, "z": 0.0},
        "field_of_view": 72,
    },
    {
        "name": "mm_kitchen_2b",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -4.0, "y": 2.1, "z": 4.0},
        "look_at": {"x": 0.0, "y": 0.85, "z": 0.0},
        "field_of_view": 72,
    },
    {
        "name": "suburb_scene_2023",
        "skybox": "sunset_fairway_4k",
        "camera_position": {"x": -5.6, "y": 2.7, "z": 5.6},
        "look_at": {"x": 0.0, "y": 0.85, "z": 0.0},
        "field_of_view": 74,
    },
    {
        "name": "building_site",
        "skybox": "bergen_4k",
        "camera_position": {"x": -4.8, "y": 2.5, "z": 4.8},
        "look_at": {"x": 0.0, "y": 0.85, "z": 0.0},
        "field_of_view": 72,
    },
]

CASES: List[Dict[str, object]] = [
    {
        "name": "soft_cloth_canvas_drape",
        "type": "cloth_sheet",
        "cloth_material": "silk",
        "sheet_type": SheetType.cloth,
        "substeps": 8,
        "position": {"x": 0.0, "y": 2.2, "z": 0.0},
        "rotation": {"x": 10.0, "y": 0.0, "z": 8.0},
        "frames": 180,
        "pre_capture_frames": 0,
        "camera_position_overrides": {
            "mm_craftroom_1a": {"x": -2.8, "y": 1.7, "z": -0.3}
        },
        "look_at_overrides": {
            "mm_craftroom_1a": {"x": 0.0, "y": 0.95, "z": 0.0}
        },
        "field_of_view_overrides": {
            "mm_craftroom_1a": 72
        },
        "initial_force": {"x": 0.42, "y": -0.1, "z": 0.18},
        "initial_torque": {"x": 0.16, "y": 0.62, "z": 0.28},
        "static_objects": [
            {"model_name": "sphere",
             "library": "models_flex.json",
             "position": {"x": 0.0, "y": 0.55, "z": 0.0},
             "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
             "scale_factor": {"x": 0.7, "y": 0.7, "z": 0.7}},
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
        "frames": 210,
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
        "frames": 180,
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
        "frames": 200,
        "static_objects": [
            {"model_name": "camera_box", "position": {"x": -0.16, "y": 0.16, "z": 0.1}, "rotation": {"x": 0.0, "y": 10.0, "z": 0.0}},
            {"model_name": "camera_box", "position": {"x": 0.26, "y": 0.16, "z": -0.05}, "rotation": {"x": 0.0, "y": -18.0, "z": 0.0}},
        ],
    },
]

AUTO_CAMERA_SCENES = {"mm_craftroom_1a", "mm_kitchen_2b"}


def sanitize_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def launch_build(scene_output_root: Path) -> subprocess.Popen:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = scene_output_root.joinpath("build.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
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


def _distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2 + (a["z"] - b["z"]) ** 2)


def _parse_raycast(resp: List[bytes], raycast_id: int) -> Raycast:
    for i in range(len(resp) - 1):
        if OutputData.get_data_type_id(resp[i]) == "rayc":
            raycast = Raycast(resp[i])
            if raycast.get_raycast_id() == raycast_id:
                return raycast
    raise RuntimeError(f"Missing raycast response for id={raycast_id}")


def _get_target_region(scene_bounds: SceneBounds, look_at: Dict[str, float]):
    target_x = float(look_at["x"])
    target_z = float(look_at["z"])
    containing_regions = [region for region in scene_bounds.regions if region.is_inside(target_x, target_z)]
    if containing_regions:
        return min(containing_regions,
                   key=lambda region: (region.x_max - region.x_min) * (region.z_max - region.z_min))
    return min(scene_bounds.regions,
               key=lambda region: (region.center[0] - target_x) ** 2 + (region.center[2] - target_z) ** 2)


def resolve_camera(scene: Dict[str, object], controller: Controller, scene_resp: List[bytes]) -> None:
    if "_resolved_camera_position" in scene:
        return
    scene_name = str(scene["name"])
    default_position = dict(scene["camera_position"])
    default_fov = int(scene["field_of_view"])
    if scene_name not in AUTO_CAMERA_SCENES:
        scene["_resolved_camera_position"] = default_position
        scene["_resolved_field_of_view"] = default_fov
        return
    try:
        scene_bounds = SceneBounds(resp=scene_resp)
        region = _get_target_region(scene_bounds=scene_bounds, look_at=scene["look_at"])
        margin = 0.75
        room_width = max(0.5, region.x_max - region.x_min - 2 * margin)
        room_depth = max(0.5, region.z_max - region.z_min - 2 * margin)
        max_radius = max(1.8, min(room_width, room_depth) * 0.48)
        base_radius = min(max_radius, max(2.0, min(room_width, room_depth) * 0.32))
        radii = []
        for radius in [base_radius + 0.45, base_radius, base_radius - 0.35, max_radius]:
            radius = round(max(1.6, min(radius, max_radius)), 2)
            if radius not in radii:
                radii.append(radius)
        heights = [2.05, 1.85, 2.25]
        angles = [225, 315, 135, 45, 250, 290, 110, 70, 180, 0, 270, 90]
        target = {"x": float(scene["look_at"]["x"]),
                  "y": float(scene["look_at"]["y"]) + 0.25,
                  "z": float(scene["look_at"]["z"])}
        best_position = None
        best_score = -1e9
        for radius in radii:
            for height in heights:
                for angle in angles:
                    radians = math.radians(angle)
                    x = target["x"] + radius * math.cos(radians)
                    z = target["z"] + radius * math.sin(radians)
                    x = min(max(x, region.x_min + margin), region.x_max - margin)
                    z = min(max(z, region.z_min + margin), region.z_max - margin)
                    wall_clearance = min(x - region.x_min, region.x_max - x, z - region.z_min, region.z_max - z)
                    if wall_clearance < margin:
                        continue
                    origin = {"x": x, "y": height, "z": z}
                    raycast_id = controller.get_unique_id()
                    resp = controller.communicate({"$type": "send_raycast",
                                                  "origin": origin,
                                                  "destination": target,
                                                  "id": raycast_id})
                    raycast = _parse_raycast(resp=resp, raycast_id=raycast_id)
                    target_distance = _distance(origin, target)
                    hit_distance = target_distance
                    if raycast.get_hit():
                        hit_point = raycast.get_point()
                        hit_distance = _distance(origin,
                                                 {"x": hit_point[0], "y": hit_point[1], "z": hit_point[2]})
                        if hit_distance < target_distance - 0.2:
                            continue
                    diagonal_bonus = 0.35 if angle % 90 != 0 else 0.0
                    score = (wall_clearance * 3.5
                             + hit_distance * 1.4
                             + diagonal_bonus
                             - abs(radius - base_radius) * 0.8
                             - abs(height - 1.95) * 0.6)
                    if score > best_score:
                        best_score = score
                        best_position = origin
        if best_position is None:
            scene["_resolved_camera_position"] = default_position
            scene["_resolved_field_of_view"] = max(default_fov, 78)
            return
        scene["_resolved_camera_position"] = best_position
        scene["_resolved_field_of_view"] = max(default_fov, 78)
        print(f"[camera] {scene_name} -> {best_position} fov={scene['_resolved_field_of_view']} score={best_score:.2f}",
              flush=True)
    except Exception as e:
        scene["_resolved_camera_position"] = default_position
        scene["_resolved_field_of_view"] = max(default_fov, 78)
        print(f"[camera] failed to auto-place for {scene_name}: {e}", flush=True)


def run_case(scene: Dict[str, object], case: Dict[str, object]) -> None:
    scene_name = str(scene["name"])
    skybox_name = str(scene["skybox"])
    scene_output_root = OUTPUT_ROOT.joinpath(scene_name)
    print(f"[phase] launch_build scene={scene_name} case={case['name']} port={PORT}", flush=True)
    build_proc = launch_build(scene_output_root=scene_output_root)
    time.sleep(BUILD_BOOT_WAIT)
    print(f"[phase] connect_controller scene={scene_name} case={case['name']}", flush=True)
    c = Controller(launch_build=False, check_version=False, port=PORT)
    try:
        case_root = scene_output_root.joinpath(str(case["name"]))
        floor_material = CollisionMaterial(dynamic_friction=0.55,
                                           static_friction=0.6,
                                           stickiness=0.0,
                                           stick_distance=0.0)
        obi = Obi(output_data=True, floor_material=floor_material)
        lighting = InteriorSceneLighting(hdri_skybox=skybox_name,
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
                         Controller.get_add_scene(scene_name=scene_name),
                         Controller.get_add_hdri_skybox(skybox_name=skybox_name),
                         {"$type": "send_scene_regions"}]
        print(f"[phase] init_scene scene={scene_name} case={case['name']}", flush=True)
        scene_resp = c.communicate(init_commands)
        print(f"[phase] init_scene_done scene={scene_name} case={case['name']}", flush=True)
        resolve_camera(scene=scene, controller=c, scene_resp=scene_resp)
        camera_position = dict(scene["_resolved_camera_position"])
        look_at = dict(scene["look_at"])
        field_of_view = int(scene["_resolved_field_of_view"])
        if scene_name in case.get("camera_position_overrides", {}):
            camera_position = dict(case["camera_position_overrides"][scene_name])
        if scene_name in case.get("look_at_overrides", {}):
            look_at = dict(case["look_at_overrides"][scene_name])
        if scene_name in case.get("field_of_view_overrides", {}):
            field_of_view = int(case["field_of_view_overrides"][scene_name])
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=camera_position,
                                   look_at=look_at,
                                   field_of_view=field_of_view)
        capture = ImageCapture(path=case_root, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        c.add_ons.extend([camera, capture])

        static_commands = []
        for static_object in case.get("static_objects", []):
            static_id = c.get_unique_id()
            static_commands.extend(c.get_add_physics_object(model_name=str(static_object["model_name"]),
                                                            object_id=static_id,
                                                            library=str(static_object.get("library", "models_core.json")),
                                                            position=static_object["position"],
                                                            rotation=static_object["rotation"],
                                                            scale_factor=static_object.get("scale_factor"),
                                                            kinematic=True,
                                                            gravity=False))
        if static_commands:
            print(f"[phase] add_static_objects scene={scene_name} case={case['name']} count={len(case.get('static_objects', []))}",
                  flush=True)
            c.communicate(static_commands)

        obi.set_solver(solver_id=0, substeps=int(case["substeps"]), scale_factor=1)
        if case["type"] == "cloth_sheet":
            cloth_id = c.get_unique_id()
            obi.create_cloth_sheet(cloth_material=str(case["cloth_material"]),
                                   object_id=cloth_id,
                                   sheet_type=case["sheet_type"],
                                   position=case["position"],
                                   rotation=case["rotation"])
        elif case["type"] == "cloth_sheet_wind":
            cloth_id = c.get_unique_id()
            wind = case["wind"]
            wind_id = c.get_unique_id()
            obi.wind_sources[wind_id] = WindSource(wind_id=wind_id,
                                                   position=wind["position"],
                                                   rotation=wind["rotation"],
                                                   emitter_radius=float(wind["emitter_radius"]),
                                                   capacity=int(wind["capacity"]),
                                                   speed=float(wind["speed"]),
                                                   lifespan=float(wind["lifespan"]),
                                                   smoothing=float(wind["smoothing"]))
            obi.create_cloth_sheet(cloth_material=str(case["cloth_material"]),
                                   object_id=cloth_id,
                                   sheet_type=case["sheet_type"],
                                   position=case["position"],
                                   rotation=case["rotation"],
                                   tether_positions={TetherParticleGroup.north_edge:
                                                     TetherType(object_id=cloth_id, is_static=True)})
        elif case["type"] == "cloth_volume":
            volume_id = c.get_unique_id()
            obi.create_cloth_volume(cloth_material=str(case["cloth_material"]),
                                    object_id=volume_id,
                                    volume_type=case["volume_type"],
                                    position=case["position"],
                                    rotation=case["rotation"],
                                    scale_factor=case["scale_factor"],
                                    pressure=float(case["pressure"]))
        else:
            raise ValueError(f"Unsupported case type: {case['type']}")

        pre_capture_frames = int(case.get("pre_capture_frames", 12))
        if pre_capture_frames > 0:
            print(f"[phase] warmup scene={scene_name} case={case['name']} frames={pre_capture_frames}", flush=True)
            for _ in range(pre_capture_frames):
                c.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        if case["type"] == "cloth_sheet" and ("initial_force" in case or "initial_torque" in case):
            obi.apply_force_to_cloth(object_id=cloth_id,
                                     force=case.get("initial_force"),
                                     torque=case.get("initial_torque"))
        print(f"[phase] simulate scene={scene_name} case={case['name']} frames={case['frames']}", flush=True)
        for _ in range(int(case["frames"])):
            c.communicate([])
        print(f"[phase] terminate scene={scene_name} case={case['name']}", flush=True)
        c.communicate({"$type": "terminate"})
    finally:
        try:
            build_proc.wait(timeout=10)
        except Exception:
            build_proc.kill()

    print(f"[phase] convert scene={scene_name} case={case['name']}", flush=True)
    convert_video(scene_output_root.joinpath(str(case["name"]), "a"),
                  scene_output_root.joinpath(f"{case['name']}.mp4"))
    print(f"[phase] convert_done scene={scene_name} case={case['name']}", flush=True)


def main() -> None:
    sanitize_proxy_env()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for scene in SCENES:
        scene_name = str(scene["name"])
        if SCENE_FILTER and scene_name not in SCENE_FILTER:
            print(f"Skipping unselected scene {scene_name}", flush=True)
            continue
        scene_root = OUTPUT_ROOT.joinpath(scene_name)
        selected_cases = [case for case in CASES if not CASE_FILTER or str(case["name"]) in CASE_FILTER]
        completed = all(scene_root.joinpath(f"{case['name']}.mp4").exists() for case in selected_cases)
        if completed:
            print(f"Skipping completed scene {scene_name}", flush=True)
            continue
        for case in CASES:
            if CASE_FILTER and str(case["name"]) not in CASE_FILTER:
                print(f"Skipping unselected case {scene_name} / {case['name']}", flush=True)
                continue
            output_video = scene_root.joinpath(f"{case['name']}.mp4")
            if output_video.exists():
                print(f"Skipping completed case {scene_name} / {case['name']}", flush=True)
                continue
            print(f"Running {scene_name} / {case['name']}", flush=True)
            run_case(scene=scene, case=case)
            print(f"Completed {scene_name} / {case['name']}", flush=True)


if __name__ == "__main__":
    main()
