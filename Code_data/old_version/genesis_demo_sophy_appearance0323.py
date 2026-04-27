import argparse
import colorsys
import json
import csv
import math
import random
import shutil
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
import genesis as gs
import trimesh

# =========================
# 基本配置
# =========================
DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/genesis_sim_sophy11")
IMG_W, IMG_H = 640, 480
N_SCENES = 5

MAX_OBJECT_PC = 2048
OBJECT_PC_STRIDE = 5
CAMERA_PC_STRIDE = 2

ENABLE_CLOTH = False
CLOTH_MESH_PATH = None
STOP_ON_ERROR = False

SCENE_FAMILY_WEIGHTS = {
    "rigid_mix": 0.4,
    "mpm_mix": 0.3,
    "sph_liquid": 0.3,
}

# =========================
# 数据集 asset 配置
# =========================
SOURCE_DATASET_ROOTS = [
    Path("/data/gaoya/dataset/SOPHY_data/bag"),
    Path("/data/gaoya/dataset/SOPHY_data/teddy_bear"),
]

MAX_ASSETS_PER_ROOT = None
ASSET_CACHE_DIR = DATASET_ROOT / "_asset_cache"
ASSET_MANIFEST_PATH = DATASET_ROOT / "asset_manifest.json"

TARGET_MESH_SIZE_RANGE = (0.08, 0.20)
SIMPLIFY_MESH_FACE_COUNT = 3000
MIN_VALID_MESH_EXTENT = 1e-5

# 容器：固定三面体（floor + left + back），不再额外加 guard wall
CONTAINER = {
    "half_x": 0.64,
    "half_y": 0.76,
    "wall_thickness": 0.035,
    "wall_height": 0.64,
    "floor_thickness": 0.04,
    "center": [0.0, 0.0, 0.0],
    "layout": "corner_three_faces",
}

# 数据集 mesh 朝向修正
DATASET_BASE_EULER_BY_DATASET = {
    "teddy_bear": [-math.pi / 2.0, 0.0, 0.0],
    "bag": [0.0, 0.0, 0.0],
}

# rigid 运动模式参数
TOP_DROP_Z_RANGE = (0.95, 1.35)
TOP_TOSS_Z_RANGE = (0.95, 1.30)
SIDE_THROW_Z_RANGE = (0.55, 0.90)
LOW_ENTRY_Z_RANGE = (0.10, 0.24)
FRONT_BACK_THROW_Z_RANGE = (0.40, 0.88)
STATIC_REST_Z_EPS = 0.01

TOP_DROP_VXY = 0.10
TOP_TOSS_VX = 0.75
TOP_TOSS_VY = 0.45
TOP_TOSS_VZ_RANGE = (-1.20, -0.25)

SIDE_THROW_VX_RANGE = (1.20, 2.35)
SIDE_THROW_VY_RANGE = (-0.22, 0.22)
SIDE_THROW_VZ_RANGE = (0.65, 1.55)

FRONT_THROW_VY_RANGE = (1.10, 2.05)
REAR_THROW_VY_RANGE = (-2.05, -1.10)
FRONT_BACK_THROW_VX_RANGE = (-0.35, 0.35)
FRONT_BACK_THROW_VZ_RANGE = (0.45, 1.30)

LOW_ROLL_SPEED_RANGE = (0.80, 1.70)
LOW_SLIDE_SPEED_RANGE = (0.60, 1.30)
LOW_ENTRY_VY_RANGE = (-0.10, 0.10)

DELAYED_ENTRY_STEP_RANGE = (12, 48)
DELAYED_LAUNCH_STEP_RANGE = (18, 64)
DELAYED_LAUNCH_VX_RANGE = (1.30, 2.30)
DELAYED_LAUNCH_VY_RANGE = (-0.18, 0.18)
DELAYED_LAUNCH_VZ_RANGE = (0.90, 1.75)

TOP_DROP_ANGVEL = 2.0
TOP_TOSS_ANGVEL = 4.0
SIDE_THROW_ANGVEL = 6.0
FRONT_BACK_THROW_ANGVEL = 5.5
LOW_ROLL_ANGVEL = 10.0
LOW_SLIDE_ANGVEL = 2.0

USE_DATASET_MESH_OBJECTS = True
DATASET_OBJECT_PROB = 1.0
USE_TEXTURED_DATASET_MESH = False
USE_DISTINCT_SCENE_COLORS = True
SCENE_COLOR_ALPHA = 1.0

RIGID_OBJECT_COUNT_RANGE = (4, 8)
MPM_OBJECT_COUNT_RANGE = (3, 6)
SPH_OBJECT_COUNT_RANGE = (2, 4)

# 固定相机参数：不再每个场景随机采样
FIXED_CAMERA_FOV = 40.0
FIXED_CAMERA_POS_OFFSET = (0.50, -0.42, 0.34)
FIXED_CAMERA_LOOKAT_OFFSET = (-0.02, 0.26, 0.18)

VISIBLE_TARGET_X_FRAC = (0.28, 0.72)
VISIBLE_TARGET_Y_FRAC = (0.28, 0.78)
VISIBLE_TARGET_Y_BACK_BIAS = (0.42, 0.78)
ENTRY_EXTRA_RANGE_NEAR = (0.04, 0.14)
ENTRY_EXTRA_RANGE_FAR = (0.12, 0.24)
GLOBAL_MOTION_SPEED_SCALE = 0.84

RIGID_SCENE_PATTERN_WEIGHTS = {
    "drop_cluster": 0.15,
    "opposed_lanes": 0.12,
    "static_targets_then_enter": 0.16,
    "rolling_impacts": 0.12,
    "mixed_tableau": 0.12,
    "sequential_crossfire": 0.11,
    "centerpiece_bombardment": 0.11,
    "delayed_entry_strike": 0.11,
}

RIGID_MOTION_WEIGHTS = {
    "top_drop": 0.15,
    "top_toss": 0.12,
    "side_throw_left": 0.07,
    "side_throw_right": 0.07,
    "front_throw": 0.06,
    "rear_throw": 0.06,
    "low_roll_left": 0.08,
    "low_roll_right": 0.08,
    "low_slide_left": 0.06,
    "low_slide_right": 0.06,
    "delayed_roll_left": 0.05,
    "delayed_roll_right": 0.05,
    "delayed_slide_left": 0.04,
    "delayed_slide_right": 0.04,
    "delayed_launch_left": 0.03,
    "delayed_launch_right": 0.03,
    "static_rest": 0.03,
}

N_BACKGROUND_PROPS_RANGE = (2, 5)
BACKGROUND_PANEL_Y = 1.10
BACKGROUND_SIDE_X = 1.05
BACKGROUND_Z_RANGE = (0.05, 0.55)

# “无明显可见运动”过滤阈值
MIN_VISIBLE_MOTION_PIXELS = 12.0
MIN_VISIBLE_MOTION_FRAMES = 4
MIN_RGB_FRAME_DIFF = 1.2
MIN_RGB_DIFF_HITS = 3
PROJECTION_MARGIN_PX = 6.0


# =========================
# 通用工具
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def weighted_choice(d: dict):
    keys = list(d.keys())
    probs = np.array(list(d.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def safe_subsample_points(xyz: np.ndarray, max_points=2048):
    if xyz is None:
        return None
    xyz = np.asarray(xyz)
    if xyz.ndim == 3:
        xyz = xyz.reshape(-1, 3)
    if len(xyz) <= max_points:
        return xyz
    idx = np.random.choice(len(xyz), size=max_points, replace=False)
    return xyz[idx]


def save_depth_vis(depth: np.ndarray, path: Path):
    d = depth.copy()
    valid = np.isfinite(d)
    vis = np.zeros_like(d, dtype=np.uint8)
    if valid.any():
        dmin, dmax = d[valid].min(), d[valid].max()
        denom = max(dmax - dmin, 1e-8)
        vis[valid] = ((d[valid] - dmin) / denom * 255).astype(np.uint8)
    imageio.imwrite(path, vis)


def safe_scene_destroy(scene):
    if scene is not None:
        try:
            scene.destroy()
        except Exception:
            pass


def sample_background():
    presets = [
        {"name": "white_studio", "background_color": [1.0, 1.0, 1.0], "ambient_light": [0.36, 0.36, 0.36]},
        {"name": "light_gray_studio", "background_color": [0.92, 0.92, 0.92], "ambient_light": [0.32, 0.32, 0.32]},
        {"name": "sky_soft", "background_color": [0.80, 0.88, 1.00], "ambient_light": [0.30, 0.31, 0.35]},
        {"name": "warm_paper", "background_color": [0.98, 0.95, 0.90], "ambient_light": [0.34, 0.32, 0.30]},
        {"name": "dark_studio", "background_color": [0.10, 0.10, 0.12], "ambient_light": [0.20, 0.20, 0.20]},
        {"name": "mint", "background_color": [0.87, 0.95, 0.92], "ambient_light": [0.30, 0.32, 0.31]},
        {"name": "lavender", "background_color": [0.90, 0.88, 0.97], "ambient_light": [0.31, 0.30, 0.34]},
        {"name": "peach", "background_color": [0.99, 0.90, 0.84], "ambient_light": [0.34, 0.31, 0.29]},
    ]
    bg = random.choice(presets)
    bg["n_props"] = random.randint(*N_BACKGROUND_PROPS_RANGE)
    return bg


def sample_color(alpha=1.0):
    return [
        float(np.random.uniform(0.08, 0.95)),
        float(np.random.uniform(0.08, 0.95)),
        float(np.random.uniform(0.08, 0.95)),
        float(alpha),
    ]


def generate_distinct_scene_colors(n: int, alpha: float = 1.0):
    if n <= 0:
        return []
    hue_offset = float(np.random.uniform(0.0, 1.0))
    colors = []
    for i in range(n):
        h = (hue_offset + i / max(n, 1)) % 1.0
        s = float(np.random.uniform(0.58, 0.88))
        v = float(np.random.uniform(0.72, 0.96))
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append([float(r), float(g), float(b), float(alpha)])
    random.shuffle(colors)
    return colors


def assign_scene_colors(objects):
    if not USE_DISTINCT_SCENE_COLORS:
        for obj in objects:
            if obj.get('color') is None and not obj.get('geom', {}).get('use_texture', False):
                obj['color'] = sample_color(alpha=SCENE_COLOR_ALPHA)
        return objects

    non_textured_indices = [
        i for i, obj in enumerate(objects)
        if not (obj.get('solver') == 'Rigid' and obj.get('geom', {}).get('use_texture', False))
    ]
    palette = generate_distinct_scene_colors(len(non_textured_indices), alpha=SCENE_COLOR_ALPHA)
    for idx, color in zip(non_textured_indices, palette):
        objects[idx]['color'] = color
    return objects


def sample_rigid_scene_pattern():
    return weighted_choice(RIGID_SCENE_PATTERN_WEIGHTS)


def get_dataset_base_euler(dataset_name: str):
    arr = DATASET_BASE_EULER_BY_DATASET.get(dataset_name, [0.0, 0.0, 0.0])
    return np.asarray(arr, dtype=np.float32)


def rgb_to_small_gray(rgb: np.ndarray):
    small = rgb[::8, ::8, :3].astype(np.float32)
    return small.mean(axis=2)


def normalize(v: np.ndarray):
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v.copy()
    return v / n


def camera_basis(camera_cfg: dict):
    pos = np.asarray(camera_cfg['pos'], dtype=np.float32)
    lookat = np.asarray(camera_cfg['lookat'], dtype=np.float32)
    forward = normalize(lookat - pos)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if float(np.linalg.norm(right)) < 1e-6:
        world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = normalize(right)
    up = normalize(np.cross(right, forward))
    return pos, right, up, forward


def project_world_to_image(point_xyz, camera_cfg: dict, img_w: int, img_h: int):
    point = np.asarray(point_xyz, dtype=np.float32)
    pos, right, up, forward = camera_basis(camera_cfg)
    rel = point - pos
    z = float(np.dot(rel, forward))
    if z <= 1e-5:
        return None

    x = float(np.dot(rel, right))
    y = float(np.dot(rel, up))

    vfov = math.radians(float(camera_cfg['fov']))
    tan_half_y = math.tan(vfov * 0.5)
    aspect = float(img_w) / float(max(img_h, 1))
    tan_half_x = tan_half_y * aspect
    if tan_half_x <= 1e-8 or tan_half_y <= 1e-8:
        return None

    ndc_x = x / (z * tan_half_x)
    ndc_y = y / (z * tan_half_y)
    if abs(ndc_x) > 1.0 or abs(ndc_y) > 1.0:
        return None

    u = (ndc_x * 0.5 + 0.5) * img_w
    v = (0.5 - ndc_y * 0.5) * img_h
    return float(u), float(v), z


def has_visible_motion(object_tracks: dict, frame_diff_hits: int, max_frame_diff: float):
    visible_motion = False
    max_pixel_disp = 0.0
    moving_object_id = None

    for obj_id, track in object_tracks.items():
        if len(track) < MIN_VISIBLE_MOTION_FRAMES:
            continue
        pts = np.asarray([[u, v] for _, u, v in track], dtype=np.float32)
        dists = np.linalg.norm(pts - pts[0:1], axis=1)
        disp = float(np.max(dists)) if len(dists) > 0 else 0.0
        if disp > max_pixel_disp:
            max_pixel_disp = disp
            moving_object_id = obj_id
        if disp >= MIN_VISIBLE_MOTION_PIXELS:
            visible_motion = True

    if frame_diff_hits >= MIN_RGB_DIFF_HITS and max_frame_diff >= MIN_RGB_FRAME_DIFF:
        visible_motion = True

    return {
        'visible_motion': bool(visible_motion),
        'max_pixel_displacement': float(max_pixel_disp),
        'moving_object_id': moving_object_id,
        'frame_diff_hits': int(frame_diff_hits),
        'max_frame_diff': float(max_frame_diff),
    }


def sample_rigid_motion(half_x: float, half_y: float, half_z: float, mode: str | None = None):
    mode = mode or weighted_choice(RIGID_MOTION_WEIGHTS)

    hx = CONTAINER["half_x"]
    hy = CONTAINER["half_y"]
    wt = CONTAINER["wall_thickness"]
    floor_z = CONTAINER["floor_thickness"] + half_z + STATIC_REST_Z_EPS

    init_pos = [0.0, 0.0, 1.0]
    linvel = [0.0, 0.0, 0.0]
    angvel = [0.0, 0.0, 0.0]
    scheduled_linvel = [0.0, 0.0, 0.0]
    scheduled_angvel = [0.0, 0.0, 0.0]
    pose_delta = [0.0, 0.0, 0.0]
    role = "dynamic"
    activation_step = 0
    entry_timing = "immediate"

    def _interior_xy(margin_x: float, margin_y: float, bias_to_back=False):
        x_min = -hx + wt + margin_x
        x_max = +hx - wt - margin_x
        if bias_to_back:
            y_min = -0.02
            y_max = +hy - wt - margin_y
        else:
            y_min = -hy + wt + 0.14 + margin_y
            y_max = +hy - wt - margin_y
        x = float(np.random.uniform(x_min, max(x_min + 1e-4, x_max)))
        y = float(np.random.uniform(y_min, max(y_min + 1e-4, y_max)))
        return x, y

    def _clamp_visible_xy(x: float, y: float):
        x_min = -hx * 0.34
        x_max = hx * 0.34
        y_min = hy * 0.12
        y_max = hy * 0.72
        return [float(np.clip(x, x_min, x_max)), float(np.clip(y, y_min, y_max))]

    def _visible_target(back_bias=False, left_bias=None, right_bias=None):
        if left_bias is True:
            x = float(np.random.uniform(-hx * VISIBLE_TARGET_X_FRAC[1], -hx * 0.08))
        elif right_bias is True:
            x = float(np.random.uniform(hx * 0.04, hx * VISIBLE_TARGET_X_FRAC[1] * 0.70))
        else:
            x = float(np.random.uniform(-hx * 0.22, hx * 0.22))
        if back_bias:
            y = float(np.random.uniform(hy * VISIBLE_TARGET_Y_BACK_BIAS[0], hy * VISIBLE_TARGET_Y_BACK_BIAS[1]))
        else:
            y = float(np.random.uniform(hy * VISIBLE_TARGET_Y_FRAC[0], hy * VISIBLE_TARGET_Y_FRAC[1]))
        return _clamp_visible_xy(x, y)

    def _right_spawn(extra_min=ENTRY_EXTRA_RANGE_NEAR[0], extra_max=ENTRY_EXTRA_RANGE_NEAR[1]):
        x = float(hx + half_x + wt + np.random.uniform(extra_min, extra_max))
        y = float(np.random.uniform(hy * 0.04, hy * 0.52))
        return x, y

    def _front_spawn(extra_min=ENTRY_EXTRA_RANGE_NEAR[0], extra_max=ENTRY_EXTRA_RANGE_NEAR[1]):
        x = float(np.random.uniform(-hx * 0.28, hx * 0.28))
        y = float(-hy - half_y - wt - np.random.uniform(extra_min, extra_max))
        return x, y

    def _diag_spawn(extra_min=ENTRY_EXTRA_RANGE_FAR[0], extra_max=ENTRY_EXTRA_RANGE_FAR[1]):
        x = float(hx + half_x + wt + np.random.uniform(extra_min, extra_max))
        y = float(-hy - half_y - wt - np.random.uniform(extra_min, extra_max))
        return x, y

    def _toward(start_xy, target_xy, speed_xy, vz=0.0, tangent_jitter=0.06):
        start_xy = np.asarray(start_xy, dtype=np.float32)
        target_xy = np.asarray(target_xy, dtype=np.float32)
        delta = target_xy - start_xy
        norm = float(np.linalg.norm(delta))
        if norm < 1e-6:
            delta = np.asarray([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction = delta / norm
        tangent = np.asarray([-direction[1], direction[0]], dtype=np.float32)
        mixed = direction + tangent * float(np.random.uniform(-tangent_jitter, tangent_jitter))
        mixed_norm = float(np.linalg.norm(mixed))
        if mixed_norm > 1e-6:
            mixed = mixed / mixed_norm
        vel_xy = mixed * float(speed_xy)
        return [float(vel_xy[0]), float(vel_xy[1]), float(vz)]

    def _spin(scale_xy=1.0, scale_z=1.0):
        return [
            float(np.random.uniform(-scale_xy, scale_xy)),
            float(np.random.uniform(-scale_xy, scale_xy)),
            float(np.random.uniform(-scale_z, scale_z)),
        ]

    def _right_target(left_bias=False):
        return _visible_target(back_bias=True, left_bias=left_bias, right_bias=not left_bias)

    def _front_target(right_bias=False):
        return _visible_target(back_bias=False, right_bias=right_bias, left_bias=not right_bias)

    if mode == "top_drop":
        x, y = _interior_xy(half_x + 0.03, half_y + 0.03, bias_to_back=False)
        z = float(max(np.random.uniform(*TOP_DROP_Z_RANGE), CONTAINER["floor_thickness"] + half_z + 0.30))
        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(-TOP_DROP_VXY, TOP_DROP_VXY)),
            float(np.random.uniform(-TOP_DROP_VXY, TOP_DROP_VXY)),
            float(np.random.uniform(-0.25, -0.05)),
        ]
        angvel = _spin(TOP_DROP_ANGVEL, 1.2)
        pose_delta = [
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
    elif mode == "top_toss":
        x, y = _interior_xy(half_x + 0.03, half_y + 0.03, bias_to_back=False)
        z = float(max(np.random.uniform(*TOP_TOSS_Z_RANGE), CONTAINER["floor_thickness"] + half_z + 0.30))
        init_pos = [x, y, z]
        linvel = [
            float(np.random.uniform(-TOP_TOSS_VX, TOP_TOSS_VX)),
            float(np.random.uniform(-TOP_TOSS_VY, TOP_TOSS_VY)),
            float(np.random.uniform(*TOP_TOSS_VZ_RANGE)),
        ]
        angvel = _spin(TOP_TOSS_ANGVEL, TOP_TOSS_ANGVEL)
        pose_delta = [
            float(np.random.uniform(-0.18, 0.18)),
            float(np.random.uniform(-0.18, 0.18)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
    elif mode in {"side_throw_left", "side_throw_right"}:
        x, y = _right_spawn(0.10, 0.22)
        z = float(np.random.uniform(*SIDE_THROW_Z_RANGE))
        init_pos = [x, y, z]
        target_xy = _right_target(left_bias=(mode == "side_throw_left"))
        linvel = _toward([x, y], target_xy, speed_xy=float(np.random.uniform(*SIDE_THROW_VX_RANGE)), vz=float(np.random.uniform(*SIDE_THROW_VZ_RANGE)), tangent_jitter=0.05)
        linvel[1] += float(np.random.uniform(*SIDE_THROW_VY_RANGE))
        angvel = _spin(SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)
        pose_delta = [
            float(np.random.uniform(-0.30, 0.30)),
            float(np.random.uniform(-0.30, 0.30)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
    elif mode == "front_throw":
        x, y = _front_spawn(0.12, 0.26)
        z = float(np.random.uniform(*FRONT_BACK_THROW_Z_RANGE))
        init_pos = [x, y, z]
        target_xy = _front_target(right_bias=False)
        linvel = _toward([x, y], target_xy, speed_xy=float(np.random.uniform(*FRONT_THROW_VY_RANGE)), vz=float(np.random.uniform(*FRONT_BACK_THROW_VZ_RANGE)), tangent_jitter=0.04)
        linvel[0] += float(np.random.uniform(*FRONT_BACK_THROW_VX_RANGE))
        angvel = _spin(FRONT_BACK_THROW_ANGVEL, FRONT_BACK_THROW_ANGVEL)
        pose_delta = [
            float(np.random.uniform(-0.24, 0.24)),
            float(np.random.uniform(-0.24, 0.24)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
    elif mode == "rear_throw":
        x, y = _diag_spawn(0.20, 0.36)
        z = float(np.random.uniform(*FRONT_BACK_THROW_Z_RANGE))
        init_pos = [x, y, z]
        target_xy = _visible_target(back_bias=True, left_bias=True)
        speed_xy = float(np.random.uniform(abs(REAR_THROW_VY_RANGE[1]), abs(REAR_THROW_VY_RANGE[0])))
        linvel = _toward([x, y], target_xy, speed_xy=speed_xy, vz=float(np.random.uniform(*FRONT_BACK_THROW_VZ_RANGE)), tangent_jitter=0.03)
        linvel[0] += float(np.random.uniform(*FRONT_BACK_THROW_VX_RANGE))
        angvel = _spin(FRONT_BACK_THROW_ANGVEL, FRONT_BACK_THROW_ANGVEL)
        pose_delta = [
            float(np.random.uniform(-0.24, 0.24)),
            float(np.random.uniform(-0.24, 0.24)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
    elif mode in {"low_roll_left", "low_roll_right", "low_slide_left", "low_slide_right"}:
        is_roll = "roll" in mode
        if mode.endswith("left"):
            x, y = _right_spawn(0.08, 0.18)
            target_xy = _visible_target(back_bias=False, left_bias=True)
        else:
            x, y = _front_spawn(0.08, 0.18)
            target_xy = _visible_target(back_bias=True, right_bias=True)
        z = float(max(floor_z, np.random.uniform(*LOW_ENTRY_Z_RANGE)))
        init_pos = [x, y, z]
        speed = float(np.random.uniform(*(LOW_ROLL_SPEED_RANGE if is_roll else LOW_SLIDE_SPEED_RANGE)))
        linvel = _toward([x, y], target_xy, speed_xy=speed, vz=float(np.random.uniform(-0.04, 0.04)), tangent_jitter=0.02)
        if is_roll:
            ang_mag = float(np.random.uniform(0.45 * LOW_ROLL_ANGVEL, LOW_ROLL_ANGVEL))
            angvel = [0.0, -ang_mag if mode.endswith("left") else ang_mag, float(np.random.uniform(-2.0, 2.0))]
        else:
            angvel = _spin(LOW_SLIDE_ANGVEL, LOW_SLIDE_ANGVEL)
        pose_delta = [
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
    elif mode in {"delayed_roll_left", "delayed_roll_right", "delayed_slide_left", "delayed_slide_right", "delayed_launch_left", "delayed_launch_right"}:
        entry_timing = "immediate_far"
        is_roll = "roll" in mode
        is_slide = "slide" in mode
        if mode.endswith("left"):
            if "launch" in mode:
                x, y = _diag_spawn(0.26, 0.44)
            else:
                x, y = _right_spawn(0.22, 0.42)
            target_xy = _visible_target(back_bias=True, left_bias=True)
        else:
            if "launch" in mode:
                x, y = _front_spawn(0.24, 0.42)
            else:
                x, y = _front_spawn(0.22, 0.42)
            target_xy = _visible_target(back_bias=True, right_bias=True)
        z = float(max(floor_z, np.random.uniform(*LOW_ENTRY_Z_RANGE)))
        init_pos = [x, y, z]
        if is_roll:
            speed = float(np.random.uniform(*LOW_ROLL_SPEED_RANGE))
            linvel = _toward([x, y], target_xy, speed_xy=speed, vz=0.0, tangent_jitter=0.02)
            ang_mag = float(np.random.uniform(0.45 * LOW_ROLL_ANGVEL, LOW_ROLL_ANGVEL))
            angvel = [0.0, -ang_mag if mode.endswith("left") else ang_mag, float(np.random.uniform(-1.8, 1.8))]
        elif is_slide:
            speed = float(np.random.uniform(*LOW_SLIDE_SPEED_RANGE))
            linvel = _toward([x, y], target_xy, speed_xy=speed, vz=0.0, tangent_jitter=0.02)
            angvel = _spin(LOW_SLIDE_ANGVEL, LOW_SLIDE_ANGVEL)
        else:
            speed_xy = float(np.random.uniform(*DELAYED_LAUNCH_VX_RANGE))
            linvel = _toward([x, y], target_xy, speed_xy=speed_xy, vz=float(np.random.uniform(*DELAYED_LAUNCH_VZ_RANGE)), tangent_jitter=0.03)
            linvel[1] += float(np.random.uniform(*DELAYED_LAUNCH_VY_RANGE))
            angvel = _spin(SIDE_THROW_ANGVEL, SIDE_THROW_ANGVEL)
        pose_delta = [
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-0.08, 0.08)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
        activation_step = 0
    elif mode == "static_rest":
        x, y = _interior_xy(half_x + 0.03, half_y + 0.03, bias_to_back=True)
        z = float(floor_z)
        init_pos = [x, y, z]
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        pose_delta = [
            float(np.random.uniform(-0.04, 0.04)),
            float(np.random.uniform(-0.04, 0.04)),
            float(np.random.uniform(-math.pi, math.pi)),
        ]
        role = "static"
    else:
        raise ValueError(mode)

    linvel = [float(v * GLOBAL_MOTION_SPEED_SCALE) for v in linvel]
    angvel = [float(v) for v in angvel]
    scheduled_linvel = list(linvel)
    scheduled_angvel = list(angvel)

    return {
        "motion_type": mode,
        "motion_group": role,
        "entry_timing": entry_timing,
        "activation_step": int(activation_step),
        "init_pos": init_pos,
        "pose_delta": pose_delta,
        "init_linvel": linvel,
        "init_angvel": angvel,
        "scheduled_linvel": scheduled_linvel,
        "scheduled_angvel": scheduled_angvel,
    }


def _try_call_methods(obj, method_names, value):
    for name in method_names:
        if hasattr(obj, name):
            fn = getattr(obj, name)
            try:
                fn(value)
                return True
            except Exception:
                try:
                    fn(tuple(np.asarray(value).tolist()))
                    return True
                except Exception:
                    pass
    return False


def _apply_rigid_velocity_fields(ent, linvel=None, angvel=None):
    if linvel is not None:
        v = np.asarray(linvel, dtype=np.float32)
        if np.linalg.norm(v) > 0:
            _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], v)
    if angvel is not None:
        w = np.asarray(angvel, dtype=np.float32)
        if np.linalg.norm(w) > 0:
            _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], w)


def apply_initial_motion_to_rigid_entity(ent, obj_meta):
    if obj_meta.get("solver") != "Rigid":
        return
    _apply_rigid_velocity_fields(
        ent,
        linvel=obj_meta.get("init_linvel", [0.0, 0.0, 0.0]),
        angvel=obj_meta.get("init_angvel", [0.0, 0.0, 0.0]),
    )


def activate_scheduled_rigid_motion(ent, obj_meta):
    if obj_meta.get("solver") != "Rigid":
        return
    _apply_rigid_velocity_fields(
        ent,
        linvel=obj_meta.get("scheduled_linvel", obj_meta.get("init_linvel", [0.0, 0.0, 0.0])),
        angvel=obj_meta.get("scheduled_angvel", obj_meta.get("init_angvel", [0.0, 0.0, 0.0])),
    )


# =========================
# asset 扫描与预处理
# =========================
def find_candidate_mesh(sample_dir: Path):
    p = sample_dir / "material.obj"
    if p.exists():
        return p
    obj_files = sorted(sample_dir.glob("*.obj"))
    if len(obj_files) == 1:
        return obj_files[0]
    if len(obj_files) > 1:
        for x in obj_files:
            if x.name.lower() == "material.obj":
                return x
        return obj_files[0]
    return None


def try_find_material_json(sample_dir: Path):
    candidates = [
        "mat_params_new_v3.4.json",
        "mat_params_new.json",
        "mat_params.json",
        "material_params.json",
        "material.json",
    ]
    for name in candidates:
        p = sample_dir / name
        if p.exists():
            return p
    return None


def load_trimesh_any(mesh_path: Path):
    obj = trimesh.load(mesh_path, process=False)
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if len(geoms) == 0:
            raise ValueError(f"No mesh geometry found in scene: {mesh_path}")
        mesh = trimesh.util.concatenate(geoms)
    elif isinstance(obj, trimesh.Trimesh):
        mesh = obj
    else:
        raise ValueError(f"Unsupported mesh type: {type(obj)}")
    return mesh


def maybe_simplify_mesh(mesh: trimesh.Trimesh):
    if SIMPLIFY_MESH_FACE_COUNT is None:
        return mesh
    try:
        if len(mesh.faces) > SIMPLIFY_MESH_FACE_COUNT:
            mesh = mesh.simplify_quadric_decimation(SIMPLIFY_MESH_FACE_COUNT)
    except Exception:
        pass
    return mesh


def sanitize_mesh(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    for fn_name in [
        'remove_unreferenced_vertices',
        'remove_duplicate_faces',
        'remove_degenerate_faces',
        'merge_vertices',
    ]:
        try:
            getattr(mesh, fn_name)()
        except Exception:
            pass
    mesh = maybe_simplify_mesh(mesh)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(~np.isfinite(extents)) or float(np.max(extents)) < MIN_VALID_MESH_EXTENT:
        raise ValueError(f"Invalid mesh extents: {extents}")
    center = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    mesh.apply_translation(-center)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    scale = 1.0 / max(float(np.max(extents)), 1e-8)
    mesh.apply_scale(scale)
    unit_extents = np.asarray(mesh.extents, dtype=np.float64)
    if np.any(~np.isfinite(unit_extents)) or float(np.max(unit_extents)) < MIN_VALID_MESH_EXTENT:
        raise ValueError(f"Invalid unit extents after normalize: {unit_extents}")
    return mesh, extents, unit_extents, scale


def build_asset_id(root: Path, sample_dir: Path):
    rel = sample_dir.relative_to(root)
    rel_str = "__".join(rel.parts)
    return f"{root.name}__{rel_str}"


def find_asset_dirs_from_roots(asset_roots):
    assets = []
    for root in asset_roots:
        if not root.exists():
            print(f"[WARN] asset root not found: {root}")
            continue
        sample_dirs = sorted({p.parent for p in root.rglob("material.obj")})
        if len(sample_dirs) == 0:
            sample_dirs = sorted({p.parent for p in root.rglob("*.obj")})
        if MAX_ASSETS_PER_ROOT is not None:
            sample_dirs = sample_dirs[:MAX_ASSETS_PER_ROOT]
        print(f"[INFO] scanning root={root} | found candidate dirs={len(sample_dirs)}")
        for d in sample_dirs:
            mesh_path = find_candidate_mesh(d)
            if mesh_path is None:
                continue
            mat_json = try_find_material_json(d)
            ply_path = d / "sampled_points.ply"
            if not ply_path.exists():
                ply_path = None
            asset_id = build_asset_id(root, d)
            assets.append({
                "asset_id": asset_id,
                "dataset_name": root.name,
                "dataset_root": str(root),
                "sample_dir": str(d),
                "mesh_path": str(mesh_path),
                "mat_json": str(mat_json) if mat_json is not None else None,
                "ply_path": str(ply_path) if ply_path is not None else None,
            })
    return assets


def prepare_asset_cache(asset):
    ensure_dir(ASSET_CACHE_DIR)
    asset_id = asset["asset_id"]
    cache_obj = ASSET_CACHE_DIR / f"{asset_id}_unit.obj"
    cache_json = ASSET_CACHE_DIR / f"{asset_id}_unit_meta.json"
    if cache_obj.exists() and cache_json.exists():
        with open(cache_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        asset["unit_mesh_path"] = str(cache_obj)
        asset["render_mesh_path"] = meta.get("render_mesh_path", asset["mesh_path"])
        asset["raw_bbox_extents"] = meta["raw_bbox_extents"]
        asset["unit_bbox_extents"] = meta["unit_bbox_extents"]
        asset["n_vertices"] = meta.get("n_vertices", None)
        asset["n_faces"] = meta.get("n_faces", None)
        asset["has_texture"] = bool(meta.get("has_texture", False))
        return asset

    mesh = load_trimesh_any(Path(asset["mesh_path"]))
    mesh, raw_extents, unit_extents, unit_scale = sanitize_mesh(mesh)
    mesh.export(cache_obj)
    sample_dir = Path(asset["sample_dir"])
    has_texture = (sample_dir / "material.mtl").exists() and len(list(sample_dir.glob("material_*.png"))) > 0
    meta = {
        "asset_id": asset_id,
        "mesh_path": asset["mesh_path"],
        "render_mesh_path": asset["mesh_path"],
        "raw_bbox_extents": raw_extents.tolist(),
        "unit_bbox_extents": unit_extents.tolist(),
        "unit_scale_from_raw": float(unit_scale),
        "n_vertices": int(len(mesh.vertices)),
        "n_faces": int(len(mesh.faces)),
        "has_texture": bool(has_texture),
    }
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    asset["unit_mesh_path"] = str(cache_obj)
    asset["render_mesh_path"] = asset["mesh_path"]
    asset["raw_bbox_extents"] = raw_extents.tolist()
    asset["unit_bbox_extents"] = unit_extents.tolist()
    asset["n_vertices"] = meta["n_vertices"]
    asset["n_faces"] = meta["n_faces"]
    asset["has_texture"] = bool(has_texture)
    return asset


def infer_friction_from_part_info(part_info):
    mat_name = str(part_info.get("mat_name", "")).lower()
    mat_sub = str(part_info.get("mat_sub_type", "")).lower()
    if "metal" in mat_name:
        return 0.25
    if "fabric" in mat_name or "polyester" in mat_sub:
        return 0.75
    if "plastic" in mat_name or "polyamide" in mat_sub or "polypropylene" in mat_sub:
        return 0.45
    return None


def sample_rigid_material():
    libs = {
        "light_plastic": {"rho": (300, 800), "friction": (0.20, 0.80)},
        "wood_like": {"rho": (500, 1000), "friction": (0.30, 0.90)},
        "metal_like": {"rho": (1500, 3000), "friction": (0.18, 0.55)},
        "rubber_like": {"rho": (900, 1300), "friction": (0.85, 1.40)},
        "ceramic_like": {"rho": (1800, 2600), "friction": (0.25, 0.70)},
        "foam_like": {"rho": (120, 320), "friction": (0.35, 0.95)},
    }
    name = random.choice(list(libs.keys()))
    conf = libs[name]
    rho = float(np.random.uniform(*conf["rho"]))
    friction = float(np.clip(np.random.uniform(*conf["friction"]), 1e-2, 5.0))
    return {"family": "Rigid", "name": name, "rho": rho, "friction": friction}


def load_asset_material_or_default(asset):
    default_mat = sample_rigid_material()
    mat_json = asset.get("mat_json", None)
    if mat_json is None:
        return default_mat
    try:
        with open(mat_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "rho" in data:
            return {
                "family": "Rigid",
                "name": data.get("name", f"{asset['dataset_name']}_mesh"),
                "rho": float(data.get("rho", default_mat["rho"])),
                "friction": float(np.clip(data.get("friction", default_mat["friction"]), 1e-2, 5.0)),
            }
        rho_list, fric_list = [], []
        for _, v in data.items():
            if not isinstance(v, dict):
                continue
            if "rho" in v:
                rho_list.append(float(v["rho"]))
            fr = infer_friction_from_part_info(v)
            if fr is not None:
                fric_list.append(fr)
        rho = float(np.mean(rho_list)) if len(rho_list) > 0 else default_mat["rho"]
        friction = float(np.mean(fric_list)) if len(fric_list) > 0 else default_mat["friction"]
        return {"family": "Rigid", "name": f"{asset['dataset_name']}_mesh", "rho": rho, "friction": float(np.clip(friction, 1e-2, 5.0))}
    except Exception:
        return default_mat


def build_asset_bank():
    raw_assets = find_asset_dirs_from_roots(SOURCE_DATASET_ROOTS)
    print(f"[INFO] total raw assets found: {len(raw_assets)}")
    bank, failed = [], []
    for a in raw_assets:
        try:
            bank.append(prepare_asset_cache(a))
        except Exception as e:
            failed.append({"sample_dir": a.get("sample_dir"), "mesh_path": a.get("mesh_path"), "error": str(e)})
            print(f"[WARN] skip asset: {a.get('sample_dir')} | err={e}")
    manifest = {
        "n_raw_assets": len(raw_assets),
        "n_usable_assets": len(bank),
        "n_failed_assets": len(failed),
        "usable_assets": bank,
        "failed_assets": failed,
    }
    with open(ASSET_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INFO] usable assets: {len(bank)}")
    if len(bank) == 0 and USE_DATASET_MESH_OBJECTS:
        print("[WARN] no usable dataset mesh assets found; will fallback to procedural primitives only.")
    return bank


# =========================
# 容器与相机
# =========================
def sample_camera(container_cfg: dict):
    hx = float(container_cfg["half_x"])
    hy = float(container_cfg["half_y"])
    wh = float(container_cfg["wall_height"])
    cx, cy, cz = [float(v) for v in container_cfg["center"]]
    cam_x = cx + hx + FIXED_CAMERA_POS_OFFSET[0]
    cam_y = cy - hy + FIXED_CAMERA_POS_OFFSET[1]
    cam_z = cz + wh + FIXED_CAMERA_POS_OFFSET[2]
    look_x = cx + FIXED_CAMERA_LOOKAT_OFFSET[0]
    look_y = cy + FIXED_CAMERA_LOOKAT_OFFSET[1]
    look_z = cz + FIXED_CAMERA_LOOKAT_OFFSET[2]
    return {
        "res": [IMG_W, IMG_H],
        "distance_scale": 0.0,
        "pos": [float(cam_x), float(cam_y), float(cam_z)],
        "lookat": [float(look_x), float(look_y), float(look_z)],
        "fov": float(FIXED_CAMERA_FOV),
        "GUI": False,
        "camera_mode": "fixed",
    }


def add_container(scene, container_cfg: dict):
    hx = container_cfg["half_x"]
    hy = container_cfg["half_y"]
    wt = container_cfg["wall_thickness"]
    wh = container_cfg["wall_height"]
    ft = container_cfg["floor_thickness"]
    cx, cy, cz = container_cfg["center"]

    wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.95)
    floor_surface = gs.surfaces.Default(color=(0.80, 0.80, 0.82, 1.0))
    left_surface = gs.surfaces.Default(color=(0.72, 0.80, 0.92, 1.0))
    back_surface = gs.surfaces.Default(color=(0.88, 0.82, 0.90, 1.0))

    container_entities = {}
    container_entities["floor"] = scene.add_entity(
        morph=gs.morphs.Box(size=(2 * hx, 2 * hy, ft), pos=(cx, cy, cz + ft / 2.0), fixed=True),
        material=wall_mat,
        surface=floor_surface,
    )
    container_entities["left_wall"] = scene.add_entity(
        morph=gs.morphs.Box(size=(wt, 2 * hy, wh), pos=(cx - hx + wt / 2.0, cy, cz + ft + wh / 2.0), fixed=True),
        material=wall_mat,
        surface=left_surface,
    )
    container_entities["back_wall"] = scene.add_entity(
        morph=gs.morphs.Box(size=(2 * hx, wt, wh), pos=(cx, cy + hy - wt / 2.0, cz + ft + wh / 2.0), fixed=True),
        material=wall_mat,
        surface=back_surface,
    )
    return container_entities


def sample_spawn_xy(margin_x: float, margin_y: float, bias_to_back=False):
    hx = CONTAINER["half_x"]
    hy = CONTAINER["half_y"]
    wt = CONTAINER["wall_thickness"]
    x_min = -hx + wt + margin_x
    x_max = +hx - wt - margin_x
    if bias_to_back:
        y_min = -0.05
        y_max = +hy - wt - margin_y
    else:
        y_min = -hy + wt + 0.14 + margin_y
        y_max = +hy - wt - margin_y
    if x_min >= x_max:
        x_min, x_max = -0.02, 0.02
    if y_min >= y_max:
        y_min, y_max = 0.00, 0.05
    x = float(np.random.uniform(x_min, x_max))
    y = float(np.random.uniform(y_min, y_max))
    return x, y


def sample_mpm_material():
    material_name = random.choice(["soft_foam", "gel", "rubbery"])
    if material_name == "soft_foam":
        E = float(np.random.uniform(2e4, 8e4))
        nu = float(np.random.uniform(0.15, 0.30))
        rho = float(np.random.uniform(500, 900))
    elif material_name == "gel":
        E = float(np.random.uniform(8e4, 2.5e5))
        nu = float(np.random.uniform(0.20, 0.35))
        rho = float(np.random.uniform(900, 1200))
    else:
        E = float(np.random.uniform(2.5e5, 8e5))
        nu = float(np.random.uniform(0.25, 0.40))
        rho = float(np.random.uniform(950, 1400))
    return {
        "family": "MPM",
        "name": material_name,
        "type": "Elastic",
        "E": E,
        "nu": nu,
        "rho": rho,
        "sampler": random.choice(["pbs", "regular", "random"]),
        "model": random.choice(["corotation", "neohooken"]),
    }


def sample_sph_material():
    return {
        "family": "SPH",
        "name": random.choice(["water_like", "viscous_liquid", "high_tension_liquid"]),
        "rho": float(np.random.uniform(900, 1200)),
        "stiffness": float(np.random.uniform(3e4, 8e4)),
        "exponent": float(np.random.uniform(5.0, 8.0)),
        "mu": float(np.random.uniform(0.002, 0.03)),
        "gamma": float(np.random.uniform(0.005, 0.03)),
        "sampler": random.choice(["regular", "pbs"]),
    }


def rigid_geom_and_margins():
    shape = random.choice(["box", "sphere", "cylinder", "box", "cylinder"])
    if shape == "box":
        long_axis = random.choice([0, 1, 2])
        dims = [float(np.random.uniform(0.05, 0.14)), float(np.random.uniform(0.05, 0.14)), float(np.random.uniform(0.05, 0.14))]
        if np.random.rand() < 0.45:
            dims[long_axis] = float(np.random.uniform(0.14, 0.24))
        geom = {"shape": "box", "size": dims}
        half_x, half_y, half_z = dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0
    elif shape == "sphere":
        radius = float(np.random.uniform(0.04, 0.10))
        geom = {"shape": "sphere", "radius": radius}
        half_x = half_y = half_z = radius
    else:
        radius = float(np.random.uniform(0.03, 0.08))
        height = float(np.random.uniform(0.08, 0.24))
        geom = {"shape": "cylinder", "radius": radius, "height": height}
        half_x = half_y = radius
        half_z = height / 2.0
    return geom, half_x, half_y, half_z


def sample_dataset_rigid_object(obj_id: int, asset_bank, pattern="drop_cluster", motion_mode=None, scene_role="dynamic"):
    asset = random.choice(asset_bank)
    unit_ext = np.array(asset["unit_bbox_extents"], dtype=np.float32)
    target_size = float(np.random.uniform(*TARGET_MESH_SIZE_RANGE))
    mesh_scale = target_size / max(float(np.max(unit_ext)), 1e-8)
    scaled_ext = unit_ext * mesh_scale
    half_x = float(scaled_ext[0] / 2.0)
    half_y = float(scaled_ext[1] / 2.0)
    half_z = float(scaled_ext[2] / 2.0)
    motion = sample_rigid_motion(half_x, half_y, half_z, mode=motion_mode)
    base_euler = get_dataset_base_euler(asset["dataset_name"])
    final_euler = (base_euler + np.asarray(motion["pose_delta"], dtype=np.float32)).tolist()
    render_mesh_file = asset["render_mesh_path"] if USE_TEXTURED_DATASET_MESH else asset["unit_mesh_path"]
    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "source_type": "dataset_mesh",
        "scene_role": scene_role,
        "motion_type": motion["motion_type"],
        "motion_group": motion.get("motion_group", "dynamic"),
        "entry_timing": motion.get("entry_timing", "immediate"),
        "activation_step": int(motion.get("activation_step", 0)),
        "geom": {
            "shape": "mesh",
            "mesh_file": render_mesh_file,
            "scale": float(mesh_scale),
            "bbox_extents": scaled_ext.tolist(),
            "asset_id": asset["asset_id"],
            "dataset_name": asset["dataset_name"],
            "sample_dir": asset["sample_dir"],
            "n_vertices": asset.get("n_vertices", None),
            "n_faces": asset.get("n_faces", None),
            "use_texture": bool(asset.get("has_texture", False) and USE_TEXTURED_DATASET_MESH),
        },
        "material": load_asset_material_or_default(asset),
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": [float(x) for x in final_euler],
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
        "scheduled_linvel": [float(x) for x in motion.get("scheduled_linvel", motion["init_linvel"])],
        "scheduled_angvel": [float(x) for x in motion.get("scheduled_angvel", motion["init_angvel"])],
        "color": None,
        "surface_vis_mode": "visual",
    }


def sample_procedural_rigid_object(obj_id: int, pattern="drop_cluster", motion_mode=None, scene_role="dynamic"):
    geom, half_x, half_y, half_z = rigid_geom_and_margins()
    material = sample_rigid_material()
    motion = sample_rigid_motion(half_x, half_y, half_z, mode=motion_mode)
    final_euler = [float(motion["pose_delta"][0]), float(motion["pose_delta"][1]), float(motion["pose_delta"][2])]
    return {
        "object_id": obj_id,
        "solver": "Rigid",
        "source_type": "procedural",
        "scene_role": scene_role,
        "motion_type": motion["motion_type"],
        "motion_group": motion.get("motion_group", "dynamic"),
        "entry_timing": motion.get("entry_timing", "immediate"),
        "activation_step": int(motion.get("activation_step", 0)),
        "geom": geom,
        "material": material,
        "init_pos": [float(x) for x in motion["init_pos"]],
        "init_euler": final_euler,
        "init_linvel": [float(x) for x in motion["init_linvel"]],
        "init_angvel": [float(x) for x in motion["init_angvel"]],
        "scheduled_linvel": [float(x) for x in motion.get("scheduled_linvel", motion["init_linvel"])],
        "scheduled_angvel": [float(x) for x in motion.get("scheduled_angvel", motion["init_angvel"])],
        "color": None,
        "surface_vis_mode": "visual",
    }


def sample_rigid_object(obj_id: int, pattern="drop_cluster", asset_bank=None, motion_mode=None, scene_role="dynamic"):
    if USE_DATASET_MESH_OBJECTS and asset_bank is not None and len(asset_bank) > 0 and (np.random.rand() < DATASET_OBJECT_PROB):
        return sample_dataset_rigid_object(obj_id, asset_bank, pattern=pattern, motion_mode=motion_mode, scene_role=scene_role)
    return sample_procedural_rigid_object(obj_id, pattern=pattern, motion_mode=motion_mode, scene_role=scene_role)


def sample_mpm_object(obj_id: int):
    shape = random.choice(["box", "sphere"])
    material = sample_mpm_material()
    if shape == "box":
        geom = {"shape": "box", "size": [float(np.random.uniform(0.10, 0.20)), float(np.random.uniform(0.10, 0.20)), float(np.random.uniform(0.10, 0.20))]}
        half_x, half_y, half_z = geom["size"][0] / 2.0, geom["size"][1] / 2.0, geom["size"][2] / 2.0
    else:
        geom = {"shape": "sphere", "radius": float(np.random.uniform(0.06, 0.10))}
        half_x = half_y = half_z = geom["radius"]
    x, y = sample_spawn_xy(half_x + 0.03, half_y + 0.03, bias_to_back=True)
    z = float(np.random.uniform(0.30, 0.62))
    z = max(z, CONTAINER["floor_thickness"] + half_z + 0.05)
    return {
        "object_id": obj_id,
        "solver": "MPM",
        "source_type": "procedural",
        "scene_role": "deformable",
        "motion_type": "deformable_drop",
        "motion_group": "dynamic",
        "geom": geom,
        "material": material,
        "init_pos": [float(x), float(y), float(z)],
        "color": None,
        "surface_vis_mode": random.choice(["visual", "particle"]),
    }


def sample_sph_object(obj_id: int):
    material = sample_sph_material()
    size = [float(np.random.uniform(0.18, 0.28)), float(np.random.uniform(0.18, 0.28)), float(np.random.uniform(0.14, 0.22))]
    half_x, half_y, half_z = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    x, y = sample_spawn_xy(half_x + 0.03, half_y + 0.03, bias_to_back=True)
    z = float(np.random.uniform(0.34, 0.58))
    z = max(z, CONTAINER["floor_thickness"] + half_z + 0.03)
    return {
        "object_id": obj_id,
        "solver": "SPH",
        "source_type": "procedural",
        "scene_role": "fluid",
        "motion_type": "fluid_release",
        "motion_group": "dynamic",
        "geom": {"shape": "box", "size": size},
        "material": material,
        "init_pos": [float(x), float(y), float(z)],
        "color": None,
        "surface_vis_mode": "particle",
    }


def sample_rigid_scene_objects(pattern: str, asset_bank=None):
    n_obj = random.randint(*RIGID_OBJECT_COUNT_RANGE)
    objects = []
    if pattern == "drop_cluster":
        candidate_modes = ["top_drop", "top_drop", "top_toss", "top_toss", "side_throw_left", "side_throw_right", "front_throw", "rear_throw"]
        for i in range(n_obj):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=random.choice(candidate_modes), scene_role="dynamic"))
    elif pattern == "opposed_lanes":
        left_modes = ["side_throw_left", "low_roll_left", "low_slide_left", "delayed_roll_left"]
        right_modes = ["side_throw_right", "low_roll_right", "low_slide_right", "delayed_roll_right"]
        modes = (left_modes * ((n_obj + 1) // 2))[: (n_obj + 1) // 2]
        modes += (right_modes * ((n_obj + 1) // 2))[: n_obj - len(modes)]
        random.shuffle(modes)
        for i, mode in enumerate(modes):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=mode, scene_role="dynamic"))
    elif pattern == "static_targets_then_enter":
        n_static = min(max(1, n_obj // 2), 3)
        for i in range(n_static):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode="static_rest", scene_role="target"))
        enter_modes = ["side_throw_left", "side_throw_right", "low_roll_left", "low_roll_right", "delayed_roll_left", "delayed_roll_right", "delayed_launch_left", "delayed_launch_right", "top_toss"]
        for i in range(n_static, n_obj):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=random.choice(enter_modes), scene_role="projectile"))
    elif pattern == "rolling_impacts":
        n_static = 1 if n_obj < 5 else 2
        for i in range(n_static):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode="static_rest", scene_role="target"))
        dynamic_modes = ["low_roll_left", "low_roll_right", "low_slide_left", "low_slide_right", "delayed_roll_left", "delayed_roll_right", "delayed_slide_left", "delayed_slide_right", "side_throw_left", "side_throw_right"]
        for i in range(n_static, n_obj):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=random.choice(dynamic_modes), scene_role="projectile"))
    elif pattern == "mixed_tableau":
        preset = ["static_rest", "top_drop", "top_toss", "side_throw_left", "side_throw_right", "front_throw", "rear_throw", "low_roll_left", "low_slide_right", "delayed_roll_left", "delayed_launch_right"]
        modes = [random.choice(preset) for _ in range(n_obj)]
        if "static_rest" not in modes:
            modes[0] = "static_rest"
        for i, mode in enumerate(modes):
            role = "target" if mode == "static_rest" else "dynamic"
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=mode, scene_role=role))
    elif pattern == "sequential_crossfire":
        n_static = 1 if n_obj < 4 else 2
        for i in range(n_static):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode="static_rest", scene_role="target"))
        seq_modes = ["delayed_roll_left", "delayed_roll_right", "delayed_slide_left", "delayed_slide_right", "delayed_launch_left", "delayed_launch_right", "front_throw", "rear_throw"]
        for i in range(n_static, n_obj):
            mode = seq_modes[(i - n_static) % len(seq_modes)]
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=mode, scene_role="projectile"))
    elif pattern == "centerpiece_bombardment":
        n_static = min(max(1, n_obj // 3), 2)
        for i in range(n_static):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode="static_rest", scene_role="centerpiece"))
        strike_modes = ["top_drop", "top_toss", "side_throw_left", "side_throw_right", "front_throw", "rear_throw", "delayed_launch_left", "delayed_launch_right"]
        for i in range(n_static, n_obj):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=random.choice(strike_modes), scene_role="striker"))
    elif pattern == "delayed_entry_strike":
        n_static = min(max(2, n_obj // 2), 4)
        for i in range(n_static):
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode="static_rest", scene_role="target"))
        delayed_modes = ["delayed_roll_left", "delayed_roll_right", "delayed_slide_left", "delayed_slide_right", "delayed_launch_left", "delayed_launch_right"]
        immediate_modes = ["top_toss", "front_throw", "rear_throw"]
        for i in range(n_static, n_obj):
            mode_pool = delayed_modes if (i - n_static) % 2 == 0 else immediate_modes
            objects.append(sample_rigid_object(i, pattern=pattern, asset_bank=asset_bank, motion_mode=random.choice(mode_pool), scene_role="projectile"))
    else:
        raise ValueError(f"Unknown rigid scene pattern: {pattern}")
    return assign_scene_colors(objects)


def sample_scene_cfg(scene_id: int, asset_bank=None):
    seed = 100000 + scene_id
    set_seed(seed)
    family = weighted_choice(SCENE_FAMILY_WEIGHTS)
    if family == "cloth_drop" and not (ENABLE_CLOTH and CLOTH_MESH_PATH and Path(CLOTH_MESH_PATH).exists()):
        family = "rigid_mix"
    bg = sample_background()
    cam = sample_camera(CONTAINER)
    rigid_pattern = None
    if family == "rigid_mix":
        rigid_pattern = sample_rigid_scene_pattern()
        objects = sample_rigid_scene_objects(rigid_pattern, asset_bank=asset_bank)
        sim_options = {
            "gravity": [0.0, 0.0, -9.81],
            "dt": 4e-3,
            "substeps": 8,
            "num_steps": 200 if rigid_pattern in {"static_targets_then_enter", "rolling_impacts"} else 180,
        }
    elif family == "mpm_mix":
        n_obj = random.randint(*MPM_OBJECT_COUNT_RANGE)
        objects = [sample_mpm_object(i) for i in range(n_obj)]
        assign_scene_colors(objects)
        sim_options = {"gravity": [0.0, 0.0, -9.81], "dt": 4e-3, "substeps": 10, "num_steps": 220}
    elif family == "sph_liquid":
        n_obj = random.randint(*SPH_OBJECT_COUNT_RANGE)
        objects = [sample_sph_object(i) for i in range(n_obj)]
        assign_scene_colors(objects)
        sim_options = {"gravity": [0.0, 0.0, -9.81], "dt": 4e-3, "substeps": 10, "num_steps": 220}
    else:
        raise ValueError(f"Unknown family: {family}")
    return {
        "scene_id": f"train_scene_{scene_id:06d}",
        "seed": seed,
        "family": family,
        "scene_pattern": rigid_pattern if rigid_pattern is not None else family,
        "background": bg,
        "container": CONTAINER,
        "camera": cam,
        "sim_options": sim_options,
        "objects": objects,
    }


def add_background_set(scene, bg_cfg):
    scene.add_entity(
        morph=gs.morphs.Box(size=(3.8, 3.8, 0.03), pos=(0.0, 0.25, -0.015), fixed=True),
        material=gs.materials.Rigid(rho=1200.0, friction=0.9),
        surface=gs.surfaces.Default(color=(0.82, 0.82, 0.84, 1.0)),
    )
    panel_color_bank = [
        (0.86, 0.88, 0.92, 1.0),
        (0.92, 0.90, 0.86, 1.0),
        (0.80, 0.86, 0.90, 1.0),
        (0.25, 0.25, 0.28, 1.0),
    ]
    scene.add_entity(
        morph=gs.morphs.Box(size=(3.2, 0.05, 1.8), pos=(0.0, BACKGROUND_PANEL_Y, 0.9), fixed=True),
        material=gs.materials.Rigid(rho=1200.0, friction=0.95),
        surface=gs.surfaces.Default(color=random.choice(panel_color_bank)),
    )
    for sign in (-1.0, 1.0):
        scene.add_entity(
            morph=gs.morphs.Box(size=(0.05, 2.5, 1.4), pos=(sign * BACKGROUND_SIDE_X, 0.25, 0.7), fixed=True),
            material=gs.materials.Rigid(rho=1200.0, friction=0.95),
            surface=gs.surfaces.Default(color=(0.78, 0.80, 0.84, 1.0)),
        )
    for _ in range(bg_cfg.get("n_props", 0)):
        shape = random.choice(["box", "sphere", "cylinder"])
        x = float(np.random.uniform(-0.95, 0.95))
        y = float(np.random.uniform(0.78, 1.35))
        z = float(np.random.uniform(*BACKGROUND_Z_RANGE))
        color = tuple(sample_color())
        if shape == "box":
            size = (float(np.random.uniform(0.10, 0.30)), float(np.random.uniform(0.08, 0.18)), float(np.random.uniform(0.12, 0.35)))
            scene.add_entity(
                morph=gs.morphs.Box(size=size, pos=(x, y, z + size[2] / 2.0), fixed=True),
                material=gs.materials.Rigid(rho=900.0, friction=0.95),
                surface=gs.surfaces.Default(color=color),
            )
        elif shape == "sphere":
            r = float(np.random.uniform(0.06, 0.14))
            scene.add_entity(
                morph=gs.morphs.Sphere(radius=r, pos=(x, y, z + r), fixed=True),
                material=gs.materials.Rigid(rho=900.0, friction=0.95),
                surface=gs.surfaces.Default(color=color),
            )
        else:
            r = float(np.random.uniform(0.05, 0.11))
            h = float(np.random.uniform(0.12, 0.30))
            scene.add_entity(
                morph=gs.morphs.Cylinder(radius=r, height=h, pos=(x, y, z + h / 2.0), fixed=True),
                material=gs.materials.Rigid(rho=900.0, friction=0.95),
                surface=gs.surfaces.Default(color=color),
            )


def build_scene(scene_cfg: dict):
    vis_options = gs.options.VisOptions(
        show_world_frame=False,
        show_link_frame=False,
        background_color=tuple(scene_cfg["background"]["background_color"]),
        ambient_light=tuple(scene_cfg["background"]["ambient_light"]),
        segmentation_level="entity",
        render_particle_as="sphere",
        particle_size_scale=1.0,
    )
    sim_options = gs.options.SimOptions(
        gravity=tuple(scene_cfg["sim_options"]["gravity"]),
        dt=scene_cfg["sim_options"]["dt"],
        substeps=scene_cfg["sim_options"]["substeps"],
    )
    family = scene_cfg["family"]
    scene_kwargs = dict(sim_options=sim_options, vis_options=vis_options, show_viewer=False)
    if family == "mpm_mix":
        scene_kwargs["mpm_options"] = gs.options.MPMOptions(lower_bound=(-0.9, -0.9, -0.1), upper_bound=(0.9, 0.9, 1.8))
    elif family == "sph_liquid":
        scene_kwargs["sph_options"] = gs.options.SPHOptions(lower_bound=(-0.9, -0.9, -0.1), upper_bound=(0.9, 0.9, 1.8), particle_size=0.01)
    elif family == "cloth_drop":
        scene_kwargs["pbd_options"] = gs.options.PBDOptions()
    try:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(dt=scene_cfg["sim_options"]["dt"], enable_collision=True, use_gjk_collision=True)
    except Exception:
        try:
            scene_kwargs["rigid_options"] = gs.options.RigidOptions(dt=scene_cfg["sim_options"]["dt"])
        except Exception:
            pass
    scene = gs.Scene(**scene_kwargs)
    add_background_set(scene, scene_cfg["background"])
    container_entities = add_container(scene, scene_cfg["container"])

    entities = []
    for obj in scene_cfg["objects"]:
        ent = None
        is_textured_dataset_mesh = (
            obj["solver"] == "Rigid"
            and obj.get("source_type") == "dataset_mesh"
            and obj["geom"]["shape"] == "mesh"
            and obj["geom"].get("use_texture", False)
        )
        surface = None
        if not is_textured_dataset_mesh:
            surface = gs.surfaces.Default(color=tuple(obj["color"]), vis_mode=obj.get("surface_vis_mode", "visual"))

        if obj["solver"] == "Rigid":
            mat = gs.materials.Rigid(rho=obj["material"]["rho"], friction=obj["material"]["friction"])
            euler = tuple(obj["init_euler"])
            pos = tuple(obj["init_pos"])
            if obj["geom"]["shape"] == "mesh":
                kwargs = dict(
                    morph=gs.morphs.Mesh(file=obj["geom"]["mesh_file"], scale=obj["geom"].get("scale", 1.0), pos=pos, euler=euler),
                    material=mat,
                )
                if surface is not None:
                    kwargs["surface"] = surface
                ent = scene.add_entity(**kwargs)
            elif obj["geom"]["shape"] == "box":
                ent = scene.add_entity(morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos, euler=euler), material=mat, surface=surface)
            elif obj["geom"]["shape"] == "sphere":
                ent = scene.add_entity(morph=gs.morphs.Sphere(radius=obj["geom"]["radius"], pos=pos, euler=euler), material=mat, surface=surface)
            elif obj["geom"]["shape"] == "cylinder":
                ent = scene.add_entity(morph=gs.morphs.Cylinder(radius=obj["geom"]["radius"], height=obj["geom"]["height"], pos=pos, euler=euler), material=mat, surface=surface)
            else:
                raise ValueError(obj["geom"]["shape"])
        elif obj["solver"] == "MPM":
            mat = gs.materials.MPM.Elastic(E=obj["material"]["E"], nu=obj["material"]["nu"], rho=obj["material"]["rho"], sampler=obj["material"]["sampler"], model=obj["material"]["model"])
            pos = tuple(obj["init_pos"])
            if obj["geom"]["shape"] == "box":
                ent = scene.add_entity(morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos), material=mat, surface=surface)
            elif obj["geom"]["shape"] == "sphere":
                ent = scene.add_entity(morph=gs.morphs.Sphere(radius=obj["geom"]["radius"], pos=pos), material=mat, surface=surface)
            else:
                raise ValueError(obj["geom"]["shape"])
        elif obj["solver"] == "SPH":
            mat = gs.materials.SPH.Liquid(
                rho=obj["material"]["rho"],
                stiffness=obj["material"]["stiffness"],
                exponent=obj["material"]["exponent"],
                mu=obj["material"]["mu"],
                gamma=obj["material"]["gamma"],
                sampler=obj["material"]["sampler"],
            )
            pos = tuple(obj["init_pos"])
            if obj["geom"]["shape"] == "box":
                ent = scene.add_entity(morph=gs.morphs.Box(size=tuple(obj["geom"]["size"]), pos=pos), material=mat, surface=surface)
            else:
                raise ValueError(obj["geom"]["shape"])
        elif obj["solver"] == "PBD":
            ent = scene.add_entity(
                material=gs.materials.PBD.Cloth(),
                morph=gs.morphs.Mesh(file=CLOTH_MESH_PATH, scale=obj.get("scale", 1.0), pos=tuple(obj["init_pos"]), euler=tuple(obj.get("init_euler", [0.0, 0.0, 0.0]))),
                surface=surface,
            )
        else:
            raise ValueError(obj["solver"])
        entities.append(ent)

    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )
    scene.build()

    for obj, ent in zip(scene_cfg["objects"], entities):
        if obj["solver"] == "Rigid":
            apply_initial_motion_to_rigid_entity(ent, obj)

    for obj, ent in zip(scene_cfg["objects"], entities):
        if obj["solver"] == "MPM" and hasattr(ent, "set_velocity"):
            try:
                pts = to_numpy(ent.get_particles_pos()).reshape(-1, 3)
                if len(pts) > 0:
                    vel = np.zeros((len(pts), 3), dtype=np.float32)
                    vel[:, 0] = np.random.uniform(-0.25, 0.25)
                    vel[:, 1] = np.random.uniform(-0.20, 0.20)
                    vel[:, 2] = np.random.uniform(-0.08, 0.08)
                    ent.set_velocity(vel)
            except Exception:
                pass

    return scene, cam, entities, container_entities


def prepare_output_dirs(out_dir: Path):
    for s in ["rgb", "depth", "depth_vis", "segmentation", "normal", "pointcloud", "object_pointcloud", "trajectories", "camera", "video"]:
        ensure_dir(out_dir / s)


def export_entity_state(ent, obj_meta):
    state = {
        "object_id": obj_meta["object_id"],
        "solver": obj_meta["solver"],
        "centroid": None,
        "quat": None,
        "vel": None,
        "ang": None,
        "pointcloud": None,
        "n_points": 0,
    }
    if hasattr(ent, "get_particles_pos"):
        try:
            pts = to_numpy(ent.get_particles_pos())
            if pts is not None and pts.size > 0:
                pts = pts.reshape(-1, 3)
                state["pointcloud"] = pts
                state["centroid"] = pts.mean(axis=0)
                state["n_points"] = int(len(pts))
                return state
        except Exception:
            pass
    if hasattr(ent, "get_verts"):
        try:
            verts = to_numpy(ent.get_verts())
            if verts is not None and verts.size > 0:
                verts = verts.reshape(-1, 3)
                state["pointcloud"] = verts
                state["centroid"] = verts.mean(axis=0)
                state["n_points"] = int(len(verts))
        except Exception:
            pass
    if hasattr(ent, "get_pos"):
        try:
            pos = to_numpy(ent.get_pos()).reshape(-1)
            state["centroid"] = pos[:3]
        except Exception:
            pass
    if hasattr(ent, "get_quat"):
        try:
            quat = to_numpy(ent.get_quat()).reshape(-1)
            state["quat"] = quat[:4]
        except Exception:
            pass
    if hasattr(ent, "get_vel"):
        try:
            vel = to_numpy(ent.get_vel()).reshape(-1)
            state["vel"] = vel[:3]
        except Exception:
            pass
    if hasattr(ent, "get_ang"):
        try:
            ang = to_numpy(ent.get_ang()).reshape(-1)
            state["ang"] = ang[:3]
        except Exception:
            pass
    return state


def export_scene(scene_cfg: dict):
    out_dir = DATASET_ROOT / "train" / scene_cfg["scene_id"]
    prepare_output_dirs(out_dir)
    with open(out_dir / "scene_input.json", "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, ensure_ascii=False, indent=2)

    scene = cam = entities = container_entities = None
    traj_csv = frame_csv = None
    result_meta = None
    delete_scene_dir = False

    try:
        scene, cam, entities, container_entities = build_scene(scene_cfg)
        try:
            np.save(out_dir / "camera" / "intrinsics.npy", to_numpy(cam.intrinsics))
        except Exception:
            pass
        try:
            np.save(out_dir / "camera" / "extrinsics.npy", to_numpy(cam.extrinsics))
        except Exception:
            pass

        traj_csv = open(out_dir / "trajectories" / "objects_world.csv", "w", newline="", encoding="utf-8")
        traj_writer = csv.writer(traj_csv)
        traj_writer.writerow(["frame", "object_id", "solver", "cx", "cy", "cz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz", "n_points"])

        frame_csv = open(out_dir / "trajectories" / "frame_index.csv", "w", newline="", encoding="utf-8")
        frame_writer = csv.writer(frame_csv)
        frame_writer.writerow(["frame", "timestamp_s", "rgb_path", "depth_path", "depth_vis_path", "seg_path", "normal_path", "pointcloud_path"])

        preview_frames = []
        collision_detected = False
        num_steps = scene_cfg["sim_options"]["num_steps"]
        activated_object_ids = set()
        object_tracks = {obj['object_id']: [] for obj in scene_cfg['objects']}
        prev_small_gray = None
        frame_diff_hits = 0
        max_frame_diff = 0.0

        for t in range(num_steps):
            for obj_meta, ent in zip(scene_cfg["objects"], entities):
                activation_step = int(obj_meta.get("activation_step", 0) or 0)
                if obj_meta.get("solver") == "Rigid" and activation_step > 0 and activation_step == t and obj_meta["object_id"] not in activated_object_ids:
                    activate_scheduled_rigid_motion(ent, obj_meta)
                    activated_object_ids.add(obj_meta["object_id"])

            scene.step()
            rgb, depth, seg, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)

            small_gray = rgb_to_small_gray(rgb)
            if prev_small_gray is not None:
                frame_diff = float(np.mean(np.abs(small_gray - prev_small_gray)))
                max_frame_diff = max(max_frame_diff, frame_diff)
                if frame_diff >= MIN_RGB_FRAME_DIFF:
                    frame_diff_hits += 1
            prev_small_gray = small_gray

            rgb_path = out_dir / "rgb" / f"{t:06d}.png"
            depth_path = out_dir / "depth" / f"{t:06d}.npy"
            depth_vis_path = out_dir / "depth_vis" / f"{t:06d}.png"
            seg_path = out_dir / "segmentation" / f"{t:06d}.npy"
            normal_path = out_dir / "normal" / f"{t:06d}.npy"

            imageio.imwrite(rgb_path, rgb)
            np.save(depth_path, depth)
            save_depth_vis(depth, depth_vis_path)
            np.save(seg_path, seg)
            np.save(normal_path, normal)

            pc_name = ""
            if (t % CAMERA_PC_STRIDE) == 0:
                try:
                    pc, mask = cam.render_pointcloud(world_frame=True)
                    pc_path = out_dir / "pointcloud" / f"{t:06d}.npz"
                    np.savez_compressed(pc_path, xyz=pc, mask=mask)
                    pc_name = pc_path.name
                except Exception:
                    pc_name = ""

            frame_writer.writerow([
                t,
                float(t * scene_cfg["sim_options"]["dt"] * scene_cfg["sim_options"]["substeps"]),
                rgb_path.name,
                depth_path.name,
                depth_vis_path.name,
                seg_path.name,
                normal_path.name,
                pc_name,
            ])

            if t % 3 == 0:
                preview_frames.append(rgb)

            for obj_meta, ent in zip(scene_cfg["objects"], entities):
                if obj_meta["solver"] == "Rigid" and hasattr(ent, "detect_collision"):
                    try:
                        if bool(ent.detect_collision()):
                            collision_detected = True
                    except Exception:
                        pass

                state = export_entity_state(ent, obj_meta)
                c = state["centroid"] if state["centroid"] is not None else [np.nan, np.nan, np.nan]
                q = state["quat"] if state["quat"] is not None else [np.nan] * 4
                v = state["vel"] if state["vel"] is not None else [np.nan] * 3
                w = state["ang"] if state["ang"] is not None else [np.nan] * 3
                traj_writer.writerow([
                    t, obj_meta["object_id"], obj_meta["solver"],
                    float(c[0]), float(c[1]), float(c[2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                    float(v[0]), float(v[1]), float(v[2]),
                    float(w[0]), float(w[1]), float(w[2]),
                    int(state["n_points"]),
                ])

                if state["centroid"] is not None:
                    proj = project_world_to_image(state["centroid"], scene_cfg['camera'], IMG_W, IMG_H)
                    if proj is not None:
                        u, vv, _ = proj
                        if (
                            -PROJECTION_MARGIN_PX <= u <= IMG_W + PROJECTION_MARGIN_PX
                            and -PROJECTION_MARGIN_PX <= vv <= IMG_H + PROJECTION_MARGIN_PX
                        ):
                            object_tracks[obj_meta['object_id']].append((t, float(u), float(vv)))

                if (t % OBJECT_PC_STRIDE) == 0 and state["pointcloud"] is not None:
                    xyz = safe_subsample_points(state["pointcloud"], max_points=MAX_OBJECT_PC)
                    np.savez_compressed(
                        out_dir / "object_pointcloud" / f"{t:06d}_obj{obj_meta['object_id']:02d}.npz",
                        xyz=xyz,
                        solver=obj_meta["solver"],
                        object_id=obj_meta["object_id"],
                    )

        motion_summary = has_visible_motion(object_tracks, frame_diff_hits, max_frame_diff)
        if not motion_summary['visible_motion']:
            result_meta = {
                "scene_id": scene_cfg["scene_id"],
                "seed": scene_cfg["seed"],
                "family": scene_cfg["family"],
                "scene_pattern": scene_cfg.get("scene_pattern", scene_cfg["family"]),
                "status": "skipped_no_visible_motion",
                "motion_filter": motion_summary,
            }
            delete_scene_dir = True
        else:
            if len(preview_frames) > 0:
                imageio.mimsave(out_dir / "video" / "preview.mp4", preview_frames, fps=20)

            material_summary = []
            for obj in scene_cfg["objects"]:
                record = {
                    "object_id": obj["object_id"],
                    "solver": obj["solver"],
                    "source_type": obj.get("source_type", "unknown"),
                    "motion_type": obj.get("motion_type", "unknown"),
                    "entry_timing": obj.get("entry_timing", "immediate"),
                    "activation_step": int(obj.get("activation_step", 0) or 0),
                    "material": obj["material"],
                }
                if obj["solver"] == "Rigid" and obj["geom"]["shape"] == "mesh":
                    record["asset"] = {
                        "asset_id": obj["geom"].get("asset_id"),
                        "dataset_name": obj["geom"].get("dataset_name"),
                        "sample_dir": obj["geom"].get("sample_dir"),
                        "mesh_file": obj["geom"].get("mesh_file"),
                        "scale": obj["geom"].get("scale"),
                        "bbox_extents": obj["geom"].get("bbox_extents"),
                        "n_vertices": obj["geom"].get("n_vertices"),
                        "n_faces": obj["geom"].get("n_faces"),
                    }
                else:
                    record["geom"] = obj["geom"]
                material_summary.append(record)

            scene_metadata = {
                "scene_id": scene_cfg["scene_id"],
                "seed": scene_cfg["seed"],
                "family": scene_cfg["family"],
                "scene_pattern": scene_cfg.get("scene_pattern", scene_cfg["family"]),
                "num_objects": len(scene_cfg["objects"]),
                "num_dataset_mesh_objects": int(sum(1 for x in scene_cfg["objects"] if x.get("source_type") == "dataset_mesh")),
                "num_static_objects": int(sum(1 for x in scene_cfg["objects"] if x.get("motion_group") == "static")),
                "num_dynamic_objects": int(sum(1 for x in scene_cfg["objects"] if x.get("motion_group") != "static")),
                "num_delayed_entry_objects": int(sum(1 for x in scene_cfg["objects"] if int(x.get("activation_step", 0) or 0) > 0)),
                "motion_types": sorted(list({x.get("motion_type", "unknown") for x in scene_cfg["objects"]})),
                "entry_timings": sorted(list({x.get("entry_timing", "immediate") for x in scene_cfg["objects"]})),
                "scene_roles": sorted(list({x.get("scene_role", "unknown") for x in scene_cfg["objects"]})),
                "sim_steps": num_steps,
                "dt": scene_cfg["sim_options"]["dt"],
                "substeps": scene_cfg["sim_options"]["substeps"],
                "duration_seconds": float(num_steps * scene_cfg["sim_options"]["dt"] * scene_cfg["sim_options"]["substeps"]),
                "collision_detected": collision_detected,
                "background_name": scene_cfg["background"]["name"],
                "camera": scene_cfg["camera"],
                "container": scene_cfg["container"],
                "material_summary": material_summary,
                "motion_filter": motion_summary,
                "status": "ok",
            }
            with open(out_dir / "scene_metadata.json", "w", encoding="utf-8") as f:
                json.dump(scene_metadata, f, ensure_ascii=False, indent=2)
            result_meta = scene_metadata
    finally:
        if traj_csv is not None:
            traj_csv.close()
        if frame_csv is not None:
            frame_csv.close()
        safe_scene_destroy(scene)

    if delete_scene_dir and out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    return result_meta


# =========================
# CLI 覆盖
# =========================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate Genesis SOPHY dataset with fixed camera and motion filtering.")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--n-scenes", type=int, default=N_SCENES)
    parser.add_argument("--img-width", type=int, default=IMG_W)
    parser.add_argument("--img-height", type=int, default=IMG_H)
    parser.add_argument("--rigid-min", type=int, default=RIGID_OBJECT_COUNT_RANGE[0])
    parser.add_argument("--rigid-max", type=int, default=RIGID_OBJECT_COUNT_RANGE[1])
    parser.add_argument("--mpm-min", type=int, default=MPM_OBJECT_COUNT_RANGE[0])
    parser.add_argument("--mpm-max", type=int, default=MPM_OBJECT_COUNT_RANGE[1])
    parser.add_argument("--sph-min", type=int, default=SPH_OBJECT_COUNT_RANGE[0])
    parser.add_argument("--sph-max", type=int, default=SPH_OBJECT_COUNT_RANGE[1])
    parser.add_argument("--use-textured-dataset-mesh", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def apply_cli_overrides(args):
    global DATASET_ROOT, N_SCENES, IMG_W, IMG_H, RIGID_OBJECT_COUNT_RANGE
    global MPM_OBJECT_COUNT_RANGE, SPH_OBJECT_COUNT_RANGE, USE_TEXTURED_DATASET_MESH, STOP_ON_ERROR
    global ASSET_CACHE_DIR, ASSET_MANIFEST_PATH
    DATASET_ROOT = Path(args.dataset_root)
    ASSET_CACHE_DIR = DATASET_ROOT / "_asset_cache"
    ASSET_MANIFEST_PATH = DATASET_ROOT / "asset_manifest.json"
    N_SCENES = int(args.n_scenes)
    IMG_W = int(args.img_width)
    IMG_H = int(args.img_height)
    RIGID_OBJECT_COUNT_RANGE = (int(args.rigid_min), int(args.rigid_max))
    MPM_OBJECT_COUNT_RANGE = (int(args.mpm_min), int(args.mpm_max))
    SPH_OBJECT_COUNT_RANGE = (int(args.sph_min), int(args.sph_max))
    USE_TEXTURED_DATASET_MESH = bool(args.use_textured_dataset_mesh)
    STOP_ON_ERROR = bool(args.stop_on_error)


def main():
    args = parse_args()
    apply_cli_overrides(args)
    ensure_dir(DATASET_ROOT)
    ensure_dir(DATASET_ROOT / "train")
    ensure_dir(DATASET_ROOT / "failed_configs")
    ensure_dir(ASSET_CACHE_DIR)

    asset_bank = build_asset_bank()
    backend_used = "cpu"
    try:
        gs.init(backend=gs.gpu)
        backend_used = "gpu"
    except Exception:
        gs.init(backend=gs.cpu)
        backend_used = "cpu"

    manifest = {
        "dataset_name": "genesis_sim_diverse_v1_fixedcam_filter",
        "split": "train",
        "n_scenes_requested": N_SCENES,
        "image_size": [IMG_W, IMG_H],
        "backend_used": backend_used,
        "scene_families": SCENE_FAMILY_WEIGHTS,
        "source_dataset_roots": [str(x) for x in SOURCE_DATASET_ROOTS],
        "use_dataset_mesh_objects": USE_DATASET_MESH_OBJECTS,
        "dataset_object_prob": DATASET_OBJECT_PROB,
        "n_usable_dataset_assets": len(asset_bank),
        "controls": {
            "rigid_object_count_range": list(RIGID_OBJECT_COUNT_RANGE),
            "mpm_object_count_range": list(MPM_OBJECT_COUNT_RANGE),
            "sph_object_count_range": list(SPH_OBJECT_COUNT_RANGE),
            "use_textured_dataset_mesh": USE_TEXTURED_DATASET_MESH,
            "use_distinct_scene_colors": USE_DISTINCT_SCENE_COLORS,
            "fixed_camera": True,
            "three_face_container": True,
        },
        "notes": [
            "Uses z-up convention.",
            "Camera is fixed across scenes.",
            "Container keeps only three faces: floor, left wall, back wall.",
            "Scenes with no visible motion in rendered frames are deleted and skipped.",
            "Rigid scenes can mix dataset mesh objects and procedural primitives.",
            "MPM/SPH scenes remain procedural for stability.",
        ],
        "scenes": [],
        "skipped_scenes": [],
        "failed_scenes": [],
    }

    try:
        for sid in range(N_SCENES):
            scene_cfg = sample_scene_cfg(sid, asset_bank=asset_bank)
            try:
                print(f"[RUN ] {scene_cfg['scene_id']} | family={scene_cfg['family']}")
                meta = export_scene(scene_cfg)
                if meta is not None and meta.get('status') == 'ok':
                    manifest['scenes'].append(meta)
                    print(f"[ OK ] {scene_cfg['scene_id']} | family={scene_cfg['family']} | dataset_mesh={meta['num_dataset_mesh_objects']}/{meta['num_objects']}")
                elif meta is not None and meta.get('status') == 'skipped_no_visible_motion':
                    manifest['skipped_scenes'].append(meta)
                    print(f"[SKIP] {scene_cfg['scene_id']} | no visible motion in rendered frames")
                else:
                    manifest['skipped_scenes'].append({
                        'scene_id': scene_cfg['scene_id'],
                        'family': scene_cfg['family'],
                        'seed': scene_cfg['seed'],
                        'status': 'skipped_unknown',
                    })
                    print(f"[SKIP] {scene_cfg['scene_id']} | skipped_unknown")
            except Exception as e:
                err_info = {"scene_id": scene_cfg["scene_id"], "family": scene_cfg["family"], "seed": scene_cfg["seed"], "error": str(e)}
                manifest["failed_scenes"].append(err_info)
                with open(DATASET_ROOT / "failed_configs" / f"{scene_cfg['scene_id']}.json", "w", encoding="utf-8") as f:
                    json.dump({"scene_cfg": scene_cfg, "error": str(e)}, f, ensure_ascii=False, indent=2)
                print(f"[FAIL] {scene_cfg['scene_id']} | family={scene_cfg['family']} | err={e}")
                if STOP_ON_ERROR:
                    raise
    finally:
        with open(DATASET_ROOT / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        try:
            gs.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    main()



'''


CUDA_VISIBLE_DEVICES=7 python /home/gaoya/Code_Video/Code_data/genesis_demo_sophy_appearance0323.py \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_test/genesis_sim_sophy_fixed \
  --n-scenes 10
'''


'''

python /home/gaoya/Code_Video/Code_data/1_localshow.py \
  --root /data/gaoya/AAA_test_video/Dataset_test/genesis_sim_sophy_fixed/train\
  --host 0.0.0.0 \
  --port 8001



'''