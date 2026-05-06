from pathlib import Path
from typing import Dict, List
import math
import os
import subprocess
import time

from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.output_data import OutputData, Raycast
from tdw.scene_data.scene_bounds import SceneBounds


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_multi_scene_real_objects")
BUILD_PATH = Path("/data/gaoya/ckpt/TDW_v1.13.0/TDW/TDW.x86_64")
DISPLAY = ":1"
PORT = 1072
BUILD_BOOT_WAIT = 12
SCENE_FILTER = {s.strip() for s in os.environ.get("TDW_SCENE_FILTER", "").split(",") if s.strip()}

SCENES: List[Dict[str, object]] = [
    {
        "name": "mm_craftroom_1a",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -4.0, "y": 2.1, "z": 4.0},
        "look_at": {"x": 0.0, "y": 0.7, "z": 0.0},
        "field_of_view": 70,
        "object_cases": [
            {
                "case_name": "realobj_camera_box_center_tumble",
                "model_name": "camera_box",
                "position": {"x": 0.0, "y": 0.95, "z": 0.0},
                "velocity": {"x": 0.38, "y": 0.15, "z": 0.12},
                "angular_velocity": {"x": 0.3, "y": 0.75, "z": 0.2},
                "frames": 96,
            },
            {
                "case_name": "realobj_wicker_basket_high_drop",
                "model_name": "basket_18inx18inx12iin_wicker",
                "position": {"x": 0.0, "y": 1.85, "z": 0.0},
                "velocity": {"x": 0.05, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.2, "y": 0.35, "z": 0.15},
                "frames": 124,
            },
            {
                "case_name": "realobj_backpack_arc_left",
                "model_name": "hiker_backpack",
                "position": {"x": -0.55, "y": 1.15, "z": -0.2},
                "velocity": {"x": 0.72, "y": 0.72, "z": 0.44},
                "angular_velocity": {"x": 0.45, "y": 0.3, "z": 0.65},
                "frames": 124,
            },
            {
                "case_name": "realobj_duffle_entry_right",
                "model_name": "duffle_bag_sm",
                "position": {"x": 1.15, "y": 0.9, "z": 0.15},
                "velocity": {"x": -1.05, "y": 0.08, "z": -0.08},
                "angular_velocity": {"x": 0.12, "y": 0.2, "z": 0.45},
                "frames": 124,
            },
        ],
    },
    {
        "name": "mm_kitchen_2b",
        "skybox": "kiara_1_dawn_4k",
        "camera_position": {"x": -4.0, "y": 2.1, "z": 4.0},
        "look_at": {"x": 0.0, "y": 0.72, "z": 0.0},
        "field_of_view": 70,
        "object_cases": [
            {
                "case_name": "realobj_jug_center_tumble",
                "model_name": "jug04",
                "position": {"x": 0.0, "y": 0.95, "z": 0.0},
                "velocity": {"x": 0.3, "y": 0.12, "z": 0.14},
                "angular_velocity": {"x": 0.22, "y": 0.72, "z": 0.18},
                "frames": 96,
            },
            {
                "case_name": "realobj_serving_bowl_high_drop",
                "model_name": "serving_bowl",
                "position": {"x": 0.0, "y": 1.65, "z": 0.0},
                "velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.25, "y": 0.2, "z": 0.15},
                "frames": 124,
            },
            {
                "case_name": "realobj_pepsi_can_arc_left",
                "model_name": "102_pepsi_can_12_fl_oz_vray",
                "position": {"x": -0.45, "y": 1.05, "z": -0.15},
                "velocity": {"x": 0.78, "y": 0.85, "z": 0.45},
                "angular_velocity": {"x": 1.0, "y": 0.45, "z": 1.1},
                "frames": 124,
            },
            {
                "case_name": "realobj_toaster_entry_right",
                "model_name": "toaster_002",
                "position": {"x": 1.05, "y": 0.78, "z": 0.0},
                "velocity": {"x": -1.08, "y": 0.0, "z": 0.0},
                "angular_velocity": {"x": 0.05, "y": 0.18, "z": 0.42},
                "frames": 124,
            },
        ],
    },
]

AUTO_CAMERA_SCENES = {"mm_craftroom_1a", "mm_kitchen_2b"}


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
            print(f"[camera] fallback to default for {scene_name}")
            return
        scene["_resolved_camera_position"] = best_position
        scene["_resolved_field_of_view"] = max(default_fov, 78)
        print(f"[camera] {scene_name} -> {best_position} fov={scene['_resolved_field_of_view']} score={best_score:.2f}")
    except Exception as e:
        scene["_resolved_camera_position"] = default_position
        scene["_resolved_field_of_view"] = max(default_fov, 78)
        print(f"[camera] failed to auto-place for {scene_name}: {e}")


def run_case(scene: Dict[str, object], case: Dict[str, object]) -> None:
    scene_name = str(scene["name"])
    skybox_name = str(scene["skybox"])
    scene_output_root = OUTPUT_ROOT.joinpath(scene_name)
    build_proc = launch_build(scene_output_root=scene_output_root)
    time.sleep(BUILD_BOOT_WAIT)
    controller = Controller(launch_build=False, port=PORT)
    try:
        case_dir = scene_output_root.joinpath(str(case["case_name"]))
        lighting = InteriorSceneLighting(hdri_skybox=skybox_name,
                                         aperture=8.0,
                                         focus_distance=4.0,
                                         ambient_occlusion_intensity=0.175,
                                         ambient_occlusion_thickness_modifier=3.5,
                                         shadow_strength=1.0)
        controller.add_ons.append(lighting)
        object_id = controller.get_unique_id()
        scene_resp = controller.communicate([{"$type": "set_screen_size",
                                              "width": 1280,
                                              "height": 720},
                                             Controller.get_add_scene(scene_name=scene_name),
                                             Controller.get_add_hdri_skybox(skybox_name=skybox_name),
                                             {"$type": "send_scene_regions"}])
        resolve_camera(scene=scene, controller=controller, scene_resp=scene_resp)
        camera = ThirdPersonCamera(avatar_id="a",
                                   position=scene["_resolved_camera_position"],
                                   look_at=scene["look_at"],
                                   field_of_view=int(scene["_resolved_field_of_view"]))
        capture = ImageCapture(path=case_dir, avatar_ids=["a"], pass_masks=["_img"])
        capture.set(frequency="never", avatar_ids=["a"], pass_masks=["_img"], save=False)
        controller.add_ons.extend([camera, capture])
        controller.communicate([])
        for _ in range(10):
            controller.communicate([])
        capture.frame = 0
        capture.set(frequency="always", avatar_ids=["a"], pass_masks=["_img"], save=True)
        commands = controller.get_add_physics_object(model_name=str(case["model_name"]),
                                                     library="models_core.json",
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


def convert_scene_videos(scene: Dict[str, object]) -> None:
    scene_root = OUTPUT_ROOT.joinpath(str(scene["name"]))
    for case in scene["object_cases"]:
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
        if SCENE_FILTER and scene_name not in SCENE_FILTER:
            print(f"Skipping unselected scene {scene_name}")
            continue
        scene_root = OUTPUT_ROOT.joinpath(scene_name)
        completed = all(scene_root.joinpath(f"{case['case_name']}.mp4").exists() for case in scene["object_cases"])
        if completed:
            print(f"Skipping completed scene {scene_name}")
            continue
        for case in scene["object_cases"]:
            print(f"Running {scene_name} / {case['case_name']} / {case['model_name']}", flush=True)
            run_case(scene=scene, case=case)
            print(f"Completed {scene_name} / {case['case_name']}", flush=True)
        convert_scene_videos(scene=scene)
        print(f"Converted videos for {scene_name}", flush=True)


if __name__ == "__main__":
    main()
