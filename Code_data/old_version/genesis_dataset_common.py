"""
Genesis Dataset Common Utilities

为 MPM 和 Rigid 求解器提供统一的共享函数接口。
"""

import json, math, random, colorsys, csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import imageio.v2 as imageio
import trimesh

# ============ 基础工具 ============

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

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

def weighted_choice(d: dict):
    keys = list(d.keys())
    probs = np.array(list(d.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return np.random.choice(keys, p=probs)

def sample_color(alpha=1.0):
    return [float(np.random.uniform(0.08, 0.95)) for _ in range(3)] + [float(alpha)]

# ============ 几何计算 ============

def euler_xyz_to_matrix(euler_xyz):
    rx, ry, rz = [float(v) for v in euler_xyz]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)

def compute_bound_radius(half_x: float, half_y: float, half_z: float):
    return float(math.sqrt(half_x ** 2 + half_y ** 2 + half_z ** 2))

def compute_vertical_half_extent(half_x: float, half_y: float, half_z: float, euler_xyz):
    R = np.abs(euler_xyz_to_matrix(euler_xyz))
    half_sizes = np.array([half_x, half_y, half_z], dtype=np.float32)
    return float(R[2].dot(half_sizes))

def clamp_float(x, lo, hi):
    return float(max(lo, min(hi, x)))

# ============ Mesh 处理 ============

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

def sanitize_trimesh_preserve_scale(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    for method in ["remove_unreferenced_vertices", "remove_duplicate_faces", "remove_degenerate_faces", "merge_vertices"]:
        try:
            getattr(mesh, method)()
        except Exception:
            pass
    return mesh

def _mesh_volume_safe(m: trimesh.Trimesh) -> float:
    try:
        if m.is_watertight:
            v = float(m.volume)
            if v > 1e-12:
                return v
    except Exception:
        pass
    try:
        return float(m.convex_hull.volume)
    except Exception:
        ex = np.maximum(np.asarray(m.extents, dtype=np.float64), 1e-6)
        return float(ex[0] * ex[1] * ex[2])

def _box_inertia_diag(mass: float, extents: np.ndarray) -> Tuple[float, float, float]:
    ex, ey, ez = [max(float(x), 1e-4) for x in np.asarray(extents).reshape(3).tolist()]
    ixx = max(mass * (ey ** 2 + ez ** 2) / 12.0, 1e-8)
    iyy = max(mass * (ex ** 2 + ez ** 2) / 12.0, 1e-8)
    izz = max(mass * (ex ** 2 + ey ** 2) / 12.0, 1e-8)
    return ixx, iyy, izz

def pick_distinct_part_colors(n: int, key: str = ""):
    def _deterministic_rng_from_key(key: str):
        seed = sum((seed * 131 + ord(ch)) % (2 ** 32 - 1) for ch in str(key))
        return np.random.RandomState(seed)
    rng = _deterministic_rng_from_key(key)
    colors = []
    if n <= 0:
        return colors
    base_h = float(rng.uniform(0.0, 1.0))
    for i in range(n):
        h = (base_h + i / max(n, 1) + rng.uniform(-0.03, 0.03)) % 1.0
        s = float(rng.uniform(0.55, 0.88))
        v = float(rng.uniform(0.68, 0.96))
        rgb = colorsys.hsv_to_rgb(h, s, v)
        colors.append([float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0])
    return colors

# ============ 材料参数 ============

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def parse_density_to_kgm3(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", "")
    try:
        return float(s)
    except Exception:
        pass
    if "g/cm" in s:
        num = safe_float(s.split()[0], None)
        return None if num is None else float(num) * 1000.0
    if "kg/m" in s:
        num = safe_float(s.split()[0], None)
        return None if num is None else float(num)
    return None

def infer_friction_from_part_info(part_info):
    mat_name = str(part_info.get("mat_name", "")).lower()
    mat_sub = str(part_info.get("mat_sub_type", "")).lower()
    if "metal" in mat_name:
        return 0.25
    if "fabric" in mat_name or "polyester" in mat_sub:
        return 0.75
    if "plastic" in mat_name or "polyamide" in mat_sub or "polypropylene" in mat_sub:
        return 0.45
    if "leather" in mat_name or "leather" in mat_sub:
        return 0.50
    return None

def infer_physx3d_friction(material_names):
    vals = []
    for name in material_names:
        s = str(name).lower()
        if "metal" in s:
            vals.append(0.25)
        elif "glass" in s:
            vals.append(0.18)
        elif "plastic" in s:
            vals.append(0.45)
        elif "wood" in s:
            vals.append(0.62)
        elif "rubber" in s:
            vals.append(1.00)
        elif "foam" in s:
            vals.append(0.78)
        elif "fabric" in s or "cloth" in s or "leather" in s:
            vals.append(0.72)
        elif "ceramic" in s or "stone" in s:
            vals.append(0.38)
    return float(np.mean(vals)) if len(vals) > 0 else None

# ============ 坐标系转换 ============

YUP_TO_ZUP_EULER_XYZ = [math.pi / 2.0, 0.0, 0.0]
DATASET_UP_AXIS_BY_DATASET = {"primitive": "z_up", "bag": "y_up", "teddy_bear": "y_up", "physx3d": "y_up"}
DATASET_EXTRA_BASE_EULER_BY_DATASET = {}

def get_dataset_base_euler(dataset_name: str):
    base = np.zeros(3, dtype=np.float32)
    up_axis = str(DATASET_UP_AXIS_BY_DATASET.get(dataset_name, "y_up")).lower()
    if up_axis == "y_up":
        base += np.asarray(YUP_TO_ZUP_EULER_XYZ, dtype=np.float32)
    extra = DATASET_EXTRA_BASE_EULER_BY_DATASET.get(dataset_name, [0.0, 0.0, 0.0])
    base += np.asarray(extra, dtype=np.float32)
    return base

def get_dataset_up_axis(dataset_name: str):
    return str(DATASET_UP_AXIS_BY_DATASET.get(dataset_name, "y_up")).lower()

def convert_bbox_extents_to_scene_frame(extents, dataset_name: str):
    ext = np.asarray(extents, dtype=np.float64).reshape(3)
    if get_dataset_up_axis(dataset_name) == "y_up":
        return ext[[0, 2, 1]]
    return ext

# ============ 导出工具 ============

def prepare_output_dirs(out_dir: Path):
    for s in ["rgb", "depth", "depth_vis", "segmentation", "normal", "pointcloud", "object_pointcloud", "trajectories", "camera", "video"]:
        ensure_dir(out_dir / s)

class SolverStateExporter:
    """统一的实体状态导出器，支持多种求解器"""
    @staticmethod
    def export(ent, obj_meta) -> Dict[str, Any]:
        state = {"object_id": obj_meta["object_id"], "solver": obj_meta["solver"], "centroid": None, "quat": None, "vel": None, "ang": None, "pointcloud": None, "n_points": 0}
        
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
        
        for attr, key in [("get_pos", "centroid"), ("get_quat", "quat"), ("get_vel", "vel"), ("get_ang", "ang")]:
            if hasattr(ent, attr):
                try:
                    val = to_numpy(getattr(ent, attr)()).reshape(-1)
                    if key == "centroid":
                        state[key] = val[:3]
                    else:
                        state[key] = val[:4] if key == "quat" else val[:3]
                except Exception:
                    pass
        
        return state

class TrajectoryWriter:
    """统一的轨迹 CSV 写入器"""
    def __init__(self, csv_path: Path):
        self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["frame", "object_id", "solver", "cx", "cy", "cz", "qx", "qy", "qz", "qw", "vx", "vy", "vz", "wx", "wy", "wz", "n_points"])
    
    def write_frame(self, frame: int, state: Dict[str, Any]):
        c = state["centroid"] if state["centroid"] is not None else [np.nan, np.nan, np.nan]
        q = state["quat"] if state["quat"] is not None else [np.nan] * 4
        v = state["vel"] if state["vel"] is not None else [np.nan] * 3
        w = state["ang"] if state["ang"] is not None else [np.nan] * 3
        self.writer.writerow([frame, state["object_id"], state["solver"], float(c[0]), float(c[1]), float(c[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(v[0]), float(v[1]), float(v[2]), float(w[0]), float(w[1]), float(w[2]), int(state["n_points"])])
    
    def close(self):
        self.csv_file.close()

class ObjectPointcloudWriter:
    """统一的物体点云 NPZ 写入器"""
    def __init__(self, out_dir: Path, max_points: int = 2048):
        self.out_dir = out_dir
        self.max_points = max_points
    
    def write(self, frame: int, state: Dict[str, Any], xyz: np.ndarray):
        xyz = safe_subsample_points(xyz, self.max_points)
        c = state["centroid"] if state["centroid"] is not None else [np.nan, np.nan, np.nan]
        q = state["quat"] if state["quat"] is not None else [np.nan] * 4
        v = state["vel"] if state["vel"] is not None else [np.nan] * 3
        w = state["ang"] if state["ang"] is not None else [np.nan] * 3
        obj_id = state["object_id"]
        out_path = self.out_dir / f"{frame:06d}_obj{obj_id:02d}.npz"
        np.savez_compressed(out_path, xyz=xyz, solver=state["solver"], object_id=obj_id, frame=frame, centroid=np.asarray(c, dtype=np.float32), quat=np.asarray(q, dtype=np.float32), vel=np.asarray(v, dtype=np.float32), ang=np.asarray(w, dtype=np.float32), n_points_raw=int(state["n_points"]), n_points_saved=int(len(xyz)), coordinate_frame="world")
        return out_path.name

# ============ 仿真参数 ============

def resolve_sim_num_steps(dt: float, target_seconds: Optional[float], target_numsteps: Optional[int]) -> int:
    if target_numsteps is not None:
        return max(1, int(target_numsteps))
    if target_seconds is not None:
        return max(1, int(round(float(target_seconds) / max(float(dt), 1e-6))))
    raise ValueError("Specify at least one of target_seconds or target_numsteps.")

def compute_preview_stride(dt: float, num_steps: int, phys_duration_s: float, preview_target_fps: int = 30) -> int:
    dt = float(max(dt, 1e-6))
    n = max(1, int(num_steps))
    phys_duration_s = float(max(phys_duration_s, n * dt))
    target_frames = max(1.0, phys_duration_s * float(preview_target_fps))
    return max(1, int(round(float(n) / target_frames)))
