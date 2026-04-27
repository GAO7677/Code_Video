from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np
import trimesh

try:
    import genesis as gs
except Exception:
    gs = None

# =========================================================
# 配置区
# =========================================================
PHYSXNET_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet")
PHYSXNET_VERSION = "version_1"
OUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_test/physxnet_proxy_dataset")

IMG_W, IMG_H = 960, 720
FPS = 64
N_SCENES = 100

MAX_DATASET_OBJECTS_TO_READ = 500
MIN_OBJECTS_PER_SCENE = 2
MAX_OBJECTS_PER_SCENE = 5
MAX_VOLUME_RATIO_IN_SCENE = 3.0
MAX_SCENE_SAMPLING_RETRIES = 100

SIM_DT = 5e-4
SIM_SUBSTEPS = 24
SIM_NUM_STEPS = 180
WARMUP_STEPS = 8
PREVIEW_FRAME_STRIDE = 3

CAMERA_DISTANCE_MULT_MIN = 5.5
CAMERA_DISTANCE_MULT_MAX = 9.0
CAMERA_FOV_MIN = 35.0
CAMERA_FOV_MAX = 46.0
CAMERA_ELEVATION_MIN = 42.0
CAMERA_ELEVATION_MAX = 62.0
CAMERA_AZIMUTH_MIN = 12.0
CAMERA_AZIMUTH_MAX = 35.0

MERGED_CACHE_DIR = PHYSXNET_ROOT / PHYSXNET_VERSION / "_merged_for_genesis"
PROXY_CACHE_DIR = PHYSXNET_ROOT / PHYSXNET_VERSION / "_collision_proxy_for_genesis"
EXPORT_MERGED_WHEN_LOADING = True
EXPORT_PROXY_WHEN_LOADING = True
PROXY_MODE = "convex_hull"  # choices: merged, convex_hull, bbox_mesh
STOP_ON_ERROR = False

SCENE_FAMILY_WEIGHTS = {
    "free_drop": 0.18,
    "vertical_bounce": 0.10,
    "oblique_throw": 0.16,
    "side_throw": 0.12,
    "rest_then_hit": 0.10,
    "line_chain_collision": 0.10,
    "cross_fire": 0.08,
    "stack_drop": 0.06,
    "late_entry": 0.05,
    "mixed_multi": 0.05,
}

MAX_LINVEL_NORM = 2.5
MAX_ANGVEL_NORM = 8.0

CORNER_BASE = {"center": [0.0, 0.0, 0.0]}

OBJECT_COLOR_PALETTE = [
    (0.90, 0.35, 0.35, 1.0),
    (0.35, 0.75, 0.40, 1.0),
    (0.30, 0.50, 0.90, 1.0),
    (0.90, 0.75, 0.30, 1.0),
    (0.75, 0.35, 0.85, 1.0),
    (0.25, 0.78, 0.82, 1.0),
    (0.95, 0.55, 0.25, 1.0),
    (0.55, 0.55, 0.60, 1.0),
]

CONTAINER_FACE_COLORS = {
    "floor": (0.82, 0.82, 0.84, 1.0),
    "wall_x": (0.72, 0.80, 0.92, 1.0),
    "wall_y": (0.88, 0.82, 0.90, 1.0),
}


# =========================================================
# 数据结构
# =========================================================
@dataclass
class PartSpec:
    part_id: int
    name: str
    mesh_path: str
    image_path: Optional[str]
    material_name: str
    density_kgm3: Optional[float]
    youngs_modulus_pa: Optional[float]
    poisson_ratio: Optional[float]
    priority_rank: Optional[int]
    basic_description: str
    functional_description: str
    movement_description: str
    joint_type: str


@dataclass
class GenesisObjectSpec:
    object_id: str
    object_name: str
    category: str
    dimension_cm: List[float]
    dimension_m: List[float]
    render_mesh_path: str
    collision_mesh_path: str
    proxy_mode: str
    bbox_extents_m: List[float]
    part_mesh_paths: List[str]
    genesis_rigid: Dict[str, Any]
    genesis_parts: List[Dict[str, Any]]
    parts: List[PartSpec]


# =========================================================
# 通用工具
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def weighted_choice(weight_dict: Dict[str, float]) -> str:
    keys = list(weight_dict.keys())
    probs = np.asarray(list(weight_dict.values()), dtype=np.float64)
    probs = probs / probs.sum()
    return str(np.random.choice(keys, p=probs))


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_density_to_kgm3(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", "")
    try:
        return float(s)
    except Exception:
        pass
    if "g/cm" in s:
        num = safe_float(s.split()[0])
        return None if num is None else float(num) * 1000.0
    return None


def parse_youngs_to_pa(v: Optional[float]) -> Optional[float]:
    return None if v is None else float(v) * 1e9


def parse_dimension_to_cm(raw: str) -> List[float]:
    raw = (raw or "").strip().lower().replace("×", "x")
    if not raw:
        return [0.0, 0.0, 0.0]
    nums = []
    cur = ""
    for ch in raw:
        if ch.isdigit() or ch in ".-":
            cur += ch
        else:
            if cur:
                nums.append(cur)
                cur = ""
    if cur:
        nums.append(cur)
    vals = [float(x) for x in nums[:3]] if nums else [0.0, 0.0, 0.0]
    while len(vals) < 3:
        vals.append(0.0)
    if "mm" in raw:
        vals = [v / 10.0 for v in vals]
    elif "m" in raw and "cm" not in raw:
        vals = [v * 100.0 for v in vals]
    return vals


def cm_to_m(vals_cm: Sequence[float]) -> List[float]:
    return [float(x) / 100.0 for x in vals_cm]


def infer_joint_type(movement_description: str) -> str:
    if not movement_description:
        return "unknown"
    s = movement_description.lower()
    if any(k in s for k in ["rigidly fixed", "fixed to", "no movement", "rigidly attached"]):
        return "fixed"
    if any(k in s for k in ["rotate", "rotates", "hinge", "revolve", "swivel"]):
        return "revolute"
    if any(k in s for k in ["slide", "slides", "sliding", "pull out", "push in", "translate"]):
        return "prismatic"
    return "unknown"


def clamp_vec_norm(v: Sequence[float], max_norm: float) -> List[float]:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n > max_norm and n > 1e-8:
        arr = arr / n * max_norm
    return arr.astype(float).tolist()


def to_numpy_host(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.cpu().numpy()
    return np.asarray(x)


def to_uint8_image(img: Any) -> np.ndarray:
    arr = to_numpy_host(img)
    if arr.dtype == np.uint8:
        return arr
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_preview_video(preview_frames: List[np.ndarray], out_video_path: Path, fps: int = FPS) -> Optional[Path]:
    if len(preview_frames) == 0:
        return None
    ensure_dir(out_video_path.parent)
    preview_frames = [to_uint8_image(fr) for fr in preview_frames]
    try:
        writer = imageio.get_writer(
            str(out_video_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,
        )
        for fr in preview_frames:
            writer.append_data(fr)
        writer.close()
        return out_video_path
    except Exception:
        pass
    gif_path = out_video_path.with_suffix(".gif")
    imageio.mimsave(str(gif_path), preview_frames, fps=fps)
    return gif_path


def safe_scene_destroy(scene: Any) -> None:
    if scene is None:
        return
    try:
        scene.destroy()
    except Exception:
        pass


def pick_distinct_colors(n: int) -> List[Tuple[float, float, float, float]]:
    palette = OBJECT_COLOR_PALETTE.copy()
    random.shuffle(palette)
    if n <= len(palette):
        return palette[:n]
    extra = []
    for i in range(n - len(palette)):
        extra.append(OBJECT_COLOR_PALETTE[i % len(OBJECT_COLOR_PALETTE)])
    return palette + extra


def object_bbox_volume(obj: Dict[str, Any]) -> float:
    ex = obj["geom"]["bbox_extents"]
    return float(ex[0] * ex[1] * ex[2])


def compute_scene_volume_ratio(objects: List[Dict[str, Any]]) -> float:
    vols = [object_bbox_volume(o) for o in objects if object_bbox_volume(o) > 1e-8]
    if len(vols) <= 1:
        return 1.0
    return float(max(vols) / min(vols))


def scene_volume_ratio_ok(objects: List[Dict[str, Any]], max_ratio: float = MAX_VOLUME_RATIO_IN_SCENE) -> bool:
    return compute_scene_volume_ratio(objects) <= max_ratio


def aabb_overlap_3d(c1: Sequence[float], e1: Sequence[float], c2: Sequence[float], e2: Sequence[float], margin: float = 0.03) -> bool:
    return (
        abs(c1[0] - c2[0]) < (e1[0] + e2[0]) / 2.0 + margin
        and abs(c1[1] - c2[1]) < (e1[1] + e2[1]) / 2.0 + margin
        and abs(c1[2] - c2[2]) < (e1[2] + e2[2]) / 2.0 + margin
    )


def sample_non_overlapping_position(existing_objects: List[Dict[str, Any]], bbox_extents: Sequence[float], mode: str, max_trials: int = 300) -> List[float]:
    ex = list(map(float, bbox_extents))
    half_x, half_y, half_z = ex[0] / 2.0, ex[1] / 2.0, ex[2] / 2.0
    safe_x_min = 0.45 + half_x
    safe_y_min = 0.45 + half_y

    for _ in range(max_trials):
        if mode in ["free_drop", "vertical_bounce", "stack_drop"]:
            pos = [
                float(np.random.uniform(safe_x_min, 1.35)),
                float(np.random.uniform(safe_y_min, 1.35)),
                float(np.random.uniform(1.00, 1.90)),
            ]
        elif mode in ["oblique_throw", "side_throw", "late_entry", "cross_fire"]:
            pos = [
                float(np.random.uniform(1.50, 2.25)),
                float(np.random.uniform(safe_y_min, 1.35)),
                float(np.random.uniform(0.55, 1.35)),
            ]
        elif mode in ["rest_then_hit", "line_chain_collision", "mixed_multi"]:
            pos = [
                float(np.random.uniform(0.65 + half_x, 1.50)),
                float(np.random.uniform(0.60 + half_y, 1.30)),
                float(np.random.uniform(max(half_z + 0.02, 0.10), 0.65)),
            ]
        else:
            pos = [
                float(np.random.uniform(safe_x_min, 1.35)),
                float(np.random.uniform(safe_y_min, 1.35)),
                float(np.random.uniform(0.95, 1.70)),
            ]

        collide = False
        for obj in existing_objects:
            margin = float(max(0.10, obj.get("placement_margin", 0.10)))
            if aabb_overlap_3d(pos, ex, obj["init_pos"], obj["geom"]["bbox_extents"], margin=margin):
                collide = True
                break
        if not collide:
            return pos

    layer = len(existing_objects)
    return [float(safe_x_min + 0.2), float(safe_y_min + 0.2), float(1.2 + 0.35 * layer)]


# =========================================================
# mesh / proxy 处理
# =========================================================
def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as trimesh.Trimesh: {mesh_path}")
    return mesh


def sanitize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    try:
        mesh.remove_duplicate_faces()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.fix_normals()
    except Exception:
        pass
    return mesh


def merge_meshes(mesh_paths: List[Path], export_path: Path) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    meshes = []
    for p in mesh_paths:
        if not p.exists():
            continue
        meshes.append(sanitize_mesh(load_mesh(p)))
    if not meshes:
        raise FileNotFoundError(f"No valid meshes found for merge: {mesh_paths}")
    merged = trimesh.util.concatenate(meshes)
    merged = sanitize_mesh(merged)
    merged.export(export_path)
    return export_path


def mesh_bbox_extents(mesh: trimesh.Trimesh) -> List[float]:
    ext = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=np.float32)
    return ext.astype(float).tolist()


def export_bbox_proxy(mesh: trimesh.Trimesh, export_path: Path) -> Tuple[Path, List[float]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    ext = mesh_bbox_extents(mesh)
    box = trimesh.creation.box(extents=ext)
    box.export(export_path)
    return export_path, ext


def export_convex_hull_proxy(mesh: trimesh.Trimesh, export_path: Path) -> Tuple[Path, List[float]]:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    hull = mesh.convex_hull
    hull = sanitize_mesh(hull)
    ext = mesh_bbox_extents(hull)
    hull.export(export_path)
    return export_path, ext


def build_collision_proxy(merged_mesh_path: Path, proxy_dir: Path, proxy_mode: str) -> Tuple[Path, List[float]]:
    mesh = sanitize_mesh(load_mesh(merged_mesh_path))
    if proxy_mode == "merged":
        return merged_mesh_path, mesh_bbox_extents(mesh)
    if proxy_mode == "convex_hull":
        return export_convex_hull_proxy(mesh, proxy_dir / "collision_convex_hull.obj")
    if proxy_mode == "bbox_mesh":
        return export_bbox_proxy(mesh, proxy_dir / "collision_bbox.obj")
    raise ValueError(f"Unknown proxy_mode: {proxy_mode}")


# =========================================================
# Loader
# =========================================================
class PhysXNetGenesisLoader:
    def __init__(
        self,
        root: str,
        version: str = "version_1",
        merged_cache_dir: Optional[str] = None,
        proxy_cache_dir: Optional[str] = None,
        merge_ext: str = ".obj",
        proxy_mode: str = PROXY_MODE,
    ):
        self.root = Path(root)
        self.version = version
        self.base_dir = self.root / version
        self.finaljson_dir = self.base_dir / "finaljson"
        self.partseg_dir = self.base_dir / "partseg"
        self.merged_cache_dir = Path(merged_cache_dir) if merged_cache_dir else self.base_dir / "_merged_for_genesis"
        self.proxy_cache_dir = Path(proxy_cache_dir) if proxy_cache_dir else self.base_dir / "_collision_proxy_for_genesis"
        self.merge_ext = merge_ext
        self.proxy_mode = proxy_mode
        if not self.finaljson_dir.exists():
            raise FileNotFoundError(f"finaljson dir not found: {self.finaljson_dir}")
        if not self.partseg_dir.exists():
            raise FileNotFoundError(f"partseg dir not found: {self.partseg_dir}")

    def __len__(self) -> int:
        return len(list(self.finaljson_dir.glob("*.json")))

    def list_object_ids(self) -> List[str]:
        return sorted([p.stem for p in self.finaljson_dir.glob("*.json")])

    def _find_img_for_part(self, imgs_dir: Path, part_id: int) -> Optional[Path]:
        if not imgs_dir.exists():
            return None
        cands = sorted(imgs_dir.glob(f"{part_id}_*.png"))
        if len(cands) == 0:
            cands = sorted(imgs_dir.glob(f"{part_id}_*"))
        return cands[0] if cands else None

    def _build_part_spec(self, obj_id: str, part_info: Dict[str, Any], objs_dir: Path, imgs_dir: Path) -> PartSpec:
        part_id = int(part_info["label"])
        mesh_path = objs_dir / f"{part_id}.obj"
        image_path = self._find_img_for_part(imgs_dir, part_id)
        material_name = str(part_info.get("material", "Unknown"))
        density_kgm3 = parse_density_to_kgm3(part_info.get("density"))
        young_pa = parse_youngs_to_pa(safe_float(part_info.get("Young's Modulus (GPa)")))
        poisson = safe_float(part_info.get("Poisson's Ratio"))
        basic_desc = str(part_info.get("Basic_description", ""))
        func_desc = str(part_info.get("Functional_description", ""))
        move_desc = str(part_info.get("Movement_description", ""))
        return PartSpec(
            part_id=part_id,
            name=str(part_info.get("name", f"part_{part_id}")),
            mesh_path=str(mesh_path),
            image_path=str(image_path) if image_path else None,
            material_name=material_name,
            density_kgm3=density_kgm3,
            youngs_modulus_pa=young_pa,
            poisson_ratio=poisson,
            priority_rank=safe_int(part_info.get("priority_rank")),
            basic_description=basic_desc,
            functional_description=func_desc,
            movement_description=move_desc,
            joint_type=infer_joint_type(move_desc),
        )

    def _build_genesis_part_dict(self, part: PartSpec) -> Dict[str, Any]:
        return {
            "name": part.name,
            "part_id": part.part_id,
            "entity_type": "rigid",
            "morph": {"type": "mesh", "file": part.mesh_path},
            "material": {
                "type": "rigid",
                "density": part.density_kgm3,
                "youngs_modulus": part.youngs_modulus_pa,
                "poisson_ratio": part.poisson_ratio,
                "material_name": part.material_name,
            },
            "semantic": {
                "priority_rank": part.priority_rank,
                "joint_type": part.joint_type,
                "basic_description": part.basic_description,
                "functional_description": part.functional_description,
                "movement_description": part.movement_description,
            },
        }

    def _build_genesis_rigid_dict(
        self,
        obj_id: str,
        object_name: str,
        render_mesh_path: Path,
        collision_mesh_path: Path,
        proxy_mode: str,
        bbox_extents_m: List[float],
        parts: List[PartSpec],
    ) -> Dict[str, Any]:
        densities = [p.density_kgm3 for p in parts if p.density_kgm3 is not None]
        youngs = [p.youngs_modulus_pa for p in parts if p.youngs_modulus_pa is not None]
        poissons = [p.poisson_ratio for p in parts if p.poisson_ratio is not None]
        avg_density = float(np.mean(densities)) if densities else None
        avg_young = float(np.mean(youngs)) if youngs else None
        avg_poisson = float(np.mean(poissons)) if poissons else None
        return {
            "name": f"{object_name}_{obj_id}",
            "entity_type": "rigid",
            "morph": {"type": "mesh", "file": str(collision_mesh_path)},
            "render": {"type": "mesh", "file": str(render_mesh_path)},
            "collision": {
                "proxy_mode": proxy_mode,
                "file": str(collision_mesh_path),
                "bbox_extents_m": bbox_extents_m,
            },
            "material": {
                "type": "rigid",
                "density": avg_density,
                "youngs_modulus": avg_young,
                "poisson_ratio": avg_poisson,
            },
            "source": {
                "object_id": obj_id,
                "num_parts": len(parts),
                "note": "Render mesh and collision proxy are separated. Collision should use proxy rather than merged render mesh.",
            },
        }

    def get_object(self, obj_id: str, export_merged: bool = True, export_proxy: bool = True) -> GenesisObjectSpec:
        json_path = self.finaljson_dir / f"{obj_id}.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        part_dir = self.partseg_dir / obj_id
        objs_dir = part_dir / "objs"
        imgs_dir = part_dir / "imgs"
        if not objs_dir.exists():
            raise FileNotFoundError(f"objs dir not found: {objs_dir}")

        object_name = str(data.get("object_name", obj_id))
        category = str(data.get("category", "Unknown"))
        dim_cm = parse_dimension_to_cm(str(data.get("dimension", "")))
        dim_m = cm_to_m(dim_cm)

        parts_info = data.get("parts", [])
        parts = [self._build_part_spec(obj_id, pinfo, objs_dir, imgs_dir) for pinfo in sorted(parts_info, key=lambda x: int(x["label"]))]
        part_mesh_paths = [Path(p.mesh_path) for p in parts if Path(p.mesh_path).exists()]
        if len(part_mesh_paths) == 0:
            raise FileNotFoundError(f"No part meshes found for object {obj_id}")

        merged_mesh_path = self.merged_cache_dir / obj_id / f"merged{self.merge_ext}"
        if export_merged and not merged_mesh_path.exists():
            merge_meshes(part_mesh_paths, merged_mesh_path)
        ensure_dir(merged_mesh_path.parent)

        proxy_dir = self.proxy_cache_dir / self.proxy_mode / obj_id
        collision_mesh_path, bbox_extents_m = build_collision_proxy(merged_mesh_path, proxy_dir, self.proxy_mode)
        if not export_proxy:
            collision_mesh_path = merged_mesh_path

        genesis_parts = [self._build_genesis_part_dict(p) for p in parts]
        genesis_rigid = self._build_genesis_rigid_dict(
            obj_id=obj_id,
            object_name=object_name,
            render_mesh_path=merged_mesh_path,
            collision_mesh_path=collision_mesh_path,
            proxy_mode=self.proxy_mode,
            bbox_extents_m=bbox_extents_m,
            parts=parts,
        )

        return GenesisObjectSpec(
            object_id=obj_id,
            object_name=object_name,
            category=category,
            dimension_cm=dim_cm,
            dimension_m=dim_m,
            render_mesh_path=str(merged_mesh_path),
            collision_mesh_path=str(collision_mesh_path),
            proxy_mode=self.proxy_mode,
            bbox_extents_m=bbox_extents_m,
            part_mesh_paths=[str(p) for p in part_mesh_paths],
            genesis_rigid=genesis_rigid,
            genesis_parts=genesis_parts,
            parts=parts,
        )

    def iter_objects(self, export_merged: bool = True, export_proxy: bool = True) -> Iterator[GenesisObjectSpec]:
        for obj_id in self.list_object_ids():
            try:
                yield self.get_object(obj_id=obj_id, export_merged=export_merged, export_proxy=export_proxy)
            except Exception as e:
                print(f"[WARN] skip object {obj_id}: {e}")


# =========================================================
# 候选池与场景采样
# =========================================================
def build_physxnet_object_bank(loader: PhysXNetGenesisLoader, max_objects_to_read: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_ids = loader.list_object_ids()
    if max_objects_to_read is not None:
        all_ids = all_ids[:max_objects_to_read]

    bank, failed = [], []
    for i, obj_id in enumerate(all_ids, 1):
        try:
            obj = loader.get_object(obj_id, export_merged=EXPORT_MERGED_WHEN_LOADING, export_proxy=EXPORT_PROXY_WHEN_LOADING)
            dim_m = np.asarray(obj.dimension_m, dtype=np.float32)
            if np.max(dim_m) > 5.0:
                failed.append({"object_id": obj_id, "error": "too_large_raw_dimension"})
                continue
            ratio = float(np.max(dim_m) / max(np.min(dim_m), 1e-6))
            if ratio > 8.0:
                failed.append({"object_id": obj_id, "error": "extreme_aspect_ratio"})
                continue
            bank.append({
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "category": obj.category,
                "dimension_m": obj.dimension_m,
                "bbox_extents_m": obj.bbox_extents_m,
                "render_mesh_path": obj.render_mesh_path,
                "collision_mesh_path": obj.collision_mesh_path,
                "proxy_mode": obj.proxy_mode,
                "genesis_rigid": obj.genesis_rigid,
                "parts": obj.parts,
            })
            if i % 50 == 0 or i == len(all_ids):
                print(f"[INFO] loaded {i}/{len(all_ids)}")
        except Exception as e:
            failed.append({"object_id": obj_id, "error": str(e)})
            print(f"[WARN] skip {obj_id}: {e}")
    return bank, failed


def split_object_bank_by_id(object_bank: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    ids = sorted([x["object_id"] for x in object_bank])
    rnd = random.Random(12345)
    rnd.shuffle(ids)
    n = len(ids)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    out = {"train": [], "val": [], "test": []}
    for x in object_bank:
        oid = x["object_id"]
        if oid in train_ids:
            out["train"].append(x)
        elif oid in val_ids:
            out["val"].append(x)
        else:
            out["test"].append(x)
    return out


def sample_background() -> Dict[str, Any]:
    presets = [
        {"name": "soft_studio", "background_color": [0.98, 0.98, 0.99], "ambient_light": [0.52, 0.52, 0.52]},
        {"name": "light_gray_studio", "background_color": [0.94, 0.94, 0.95], "ambient_light": [0.50, 0.50, 0.50]},
        {"name": "warm_soft", "background_color": [0.98, 0.96, 0.93], "ambient_light": [0.50, 0.48, 0.46]},
        {"name": "cold_clean", "background_color": [0.93, 0.96, 0.99], "ambient_light": [0.46, 0.48, 0.52]},
    ]
    return random.choice(presets)


def build_corner_cfg_from_objects(objects: List[Dict[str, Any]], corner_base: Dict[str, Any]) -> Dict[str, Any]:
    max_x = max(float(obj["geom"]["bbox_extents"][0]) for obj in objects)
    max_y = max(float(obj["geom"]["bbox_extents"][1]) for obj in objects)
    max_z = max(float(obj["geom"]["bbox_extents"][2]) for obj in objects)
    ref = max(max_x, max_y, max_z, 0.25)
    panel_size = float(np.random.uniform(16.5 * ref, 20.0 * ref))
    thickness = max(0.05, 0.07 * ref)
    return {"center": list(corner_base["center"]), "panel_size": panel_size, "thickness": thickness}


def sample_camera_from_objects(objects: List[Dict[str, Any]], corner_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cx, cy, cz = corner_cfg["center"]
    max_x = max(float(obj["geom"]["bbox_extents"][0]) for obj in objects)
    max_y = max(float(obj["geom"]["bbox_extents"][1]) for obj in objects)
    max_z = max(float(obj["geom"]["bbox_extents"][2]) for obj in objects)
    obj_ref = max(max_x, max_y, max_z, 0.25)
    focus_x = cx + max(0.8, 2.0 * obj_ref)
    focus_y = cy + max(0.8, 2.0 * obj_ref)
    focus_z = cz + obj_ref * 0.2
    dist = max(np.random.uniform(CAMERA_DISTANCE_MULT_MIN, CAMERA_DISTANCE_MULT_MAX) * obj_ref, 4.5)
    theta = np.deg2rad(np.random.uniform(CAMERA_ELEVATION_MIN, CAMERA_ELEVATION_MAX))
    phi = np.deg2rad(np.random.uniform(CAMERA_AZIMUTH_MIN, CAMERA_AZIMUTH_MAX))
    cam_x = focus_x + dist * np.sin(theta) * np.cos(phi)
    cam_y = focus_y + dist * np.sin(theta) * np.sin(phi)
    cam_z = focus_z + dist * np.cos(theta)
    return {
        "res": [IMG_W, IMG_H],
        "pos": [float(cam_x), float(cam_y), float(cam_z)],
        "lookat": [float(focus_x), float(focus_y), float(focus_z)],
        "fov": float(np.random.uniform(CAMERA_FOV_MIN, CAMERA_FOV_MAX)),
        "GUI": False,
    }


def sample_motion_for_family(family: str, obj_rank: int, bbox_extents: Sequence[float], existing_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    ex = list(map(float, bbox_extents))
    hx, hy, hz = ex[0] / 2.0, ex[1] / 2.0, ex[2] / 2.0
    init_pos = sample_non_overlapping_position(existing_objects, ex, family)
    init_euler = [float(np.random.uniform(-0.25, 0.25)), float(np.random.uniform(-0.25, 0.25)), float(np.random.uniform(-math.pi, math.pi))]
    linvel = [0.0, 0.0, 0.0]
    angvel = [float(np.random.uniform(-2.0, 2.0)) for _ in range(3)]
    events = []
    mode_name = family

    if family == "free_drop":
        init_pos[2] = float(np.random.uniform(1.0, 1.9))
    elif family == "vertical_bounce":
        init_pos[2] = float(np.random.uniform(1.35, 2.1))
        linvel = [0.0, 0.0, float(np.random.uniform(-0.3, -0.05))]
    elif family == "oblique_throw":
        theta = math.radians(float(np.random.uniform(15.0, 70.0)))
        speed = float(np.random.uniform(1.4, 2.4))
        phi = math.radians(float(np.random.uniform(180.0, 260.0)))
        linvel = [speed * math.cos(theta) * math.cos(phi), speed * math.cos(theta) * math.sin(phi), speed * math.sin(theta)]
        angvel = [float(np.random.uniform(-5.0, 5.0)) for _ in range(3)]
    elif family == "side_throw":
        init_pos = [float(np.random.uniform(1.55, 2.3)), float(np.random.uniform(0.55 + hy, 1.35)), float(np.random.uniform(0.55, 1.25))]
        linvel = [float(np.random.uniform(-2.1, -1.2)), float(np.random.uniform(-0.2, 0.2)), float(np.random.uniform(0.7, 1.4))]
        angvel = [float(np.random.uniform(-6.0, 6.0)) for _ in range(3)]
    elif family == "rest_then_hit":
        if obj_rank == 0:
            init_pos = [1.15, 0.85, max(hz + 0.02, 0.12)]
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            mode_name = "rest_target"
        else:
            init_pos = [2.2, 0.85, max(hz + 0.06, 0.18)]
            linvel = [0.0, 0.0, 0.0]
            angvel = [0.0, 0.0, 0.0]
            events.append({
                "type": "set_motion",
                "frame_idx": int(np.random.randint(18, 36)),
                "linvel": [float(np.random.uniform(-2.0, -1.2)), 0.0, float(np.random.uniform(0.15, 0.55))],
                "angvel": [float(np.random.uniform(-4.0, 4.0)) for _ in range(3)],
            })
            mode_name = "late_hit_actor"
    elif family == "line_chain_collision":
        init_pos = [1.85 - 0.33 * obj_rank, 0.85, max(hz + 0.02, 0.12)]
        linvel = [0.0, 0.0, 0.0]
        if obj_rank == 0:
            linvel = [-1.8, 0.0, 0.0]
            mode_name = "chain_striker"
        else:
            mode_name = "chain_target"
    elif family == "cross_fire":
        if obj_rank % 2 == 0:
            init_pos = [2.1, 0.75 + 0.15 * obj_rank, 0.45]
            linvel = [-1.7, 0.0, 0.5]
        else:
            init_pos = [0.75, 2.0, 0.45]
            linvel = [0.0, -1.6, 0.4]
        mode_name = "cross_fire"
    elif family == "stack_drop":
        init_pos = [0.95, 0.85, 0.45 + 0.35 * obj_rank]
        linvel = [0.0, 0.0, 0.0]
        mode_name = "stack_drop"
    elif family == "late_entry":
        init_pos = [2.6 + 0.4 * obj_rank, 1.0, 0.4]
        linvel = [0.0, 0.0, 0.0]
        angvel = [0.0, 0.0, 0.0]
        events.append({
            "type": "teleport_and_set_motion",
            "frame_idx": int(np.random.randint(15, 45)),
            "request_pos": [2.1, 0.9 + 0.12 * obj_rank, 0.55],
            "linvel": [float(np.random.uniform(-1.8, -1.0)), 0.0, float(np.random.uniform(0.2, 0.8))],
            "angvel": [float(np.random.uniform(-3.0, 3.0)) for _ in range(3)],
        })
        mode_name = "late_entry"
    elif family == "mixed_multi":
        if obj_rank == 0:
            init_pos[2] = float(np.random.uniform(1.2, 1.8))
            mode_name = "mixed_drop"
        elif obj_rank == 1:
            init_pos = [2.0, 0.85, 0.55]
            linvel = [-1.4, 0.0, 0.5]
            mode_name = "mixed_side"
        else:
            init_pos = [1.1, 0.95, max(hz + 0.02, 0.12)]
            linvel = [0.0, 0.0, 0.0]
            mode_name = "mixed_rest"
    linvel = clamp_vec_norm(linvel, MAX_LINVEL_NORM)
    angvel = clamp_vec_norm(angvel, MAX_ANGVEL_NORM)
    return {
        "motion_type": mode_name,
        "init_pos": [float(x) for x in init_pos],
        "init_euler": init_euler,
        "init_linvel": linvel,
        "init_angvel": angvel,
        "script_events": events,
    }


def sample_physxnet_rigid_object(obj_idx: int, bank_item: Dict[str, Any], family: str, existing_objects: Optional[List[Dict[str, Any]]] = None, scene_colors: Optional[List[Tuple[float, float, float, float]]] = None) -> Dict[str, Any]:
    existing_objects = existing_objects or []
    dim_m = np.asarray(bank_item["dimension_m"], dtype=np.float32)
    if np.max(dim_m) <= 1e-6:
        dim_m = np.asarray([0.25, 0.25, 0.25], dtype=np.float32)
    max_dim = float(np.max(dim_m))
    target_max_dim = float(np.random.uniform(0.18, 0.42))
    mesh_scale = target_max_dim / max(max_dim, 1e-8)

    # 用 proxy 的 bbox 控制碰撞体尺寸，而不是 render mesh 原始 dimension_m
    proxy_bbox = np.asarray(bank_item.get("bbox_extents_m", bank_item["dimension_m"]), dtype=np.float32)
    scaled_dim = np.maximum(proxy_bbox * mesh_scale, 0.03)
    motion = sample_motion_for_family(family, obj_idx, scaled_dim.tolist(), existing_objects)
    rigid = bank_item["genesis_rigid"]
    material = rigid["material"]
    rho = material.get("density", 1200.0)
    friction = float(np.random.uniform(0.35, 0.95))
    young = material.get("youngs_modulus", None)
    poisson = material.get("poisson_ratio", None)
    render_color = scene_colors[obj_idx] if scene_colors is not None else pick_distinct_colors(obj_idx + 1)[-1]
    return {
        "scene_object_id": obj_idx,
        "solver": "Rigid",
        "source_type": "physxnet_proxy_mesh",
        "motion_type": motion["motion_type"],
        "object_id": bank_item["object_id"],
        "object_name": bank_item["object_name"],
        "category": bank_item["category"],
        "proxy_mode": bank_item["proxy_mode"],
        "geom": {
            "shape": "mesh",
            "render_mesh_file": bank_item["render_mesh_path"],
            "collision_mesh_file": bank_item["collision_mesh_path"],
            "scale": float(mesh_scale),
            "bbox_extents": scaled_dim.astype(float).tolist(),
        },
        "material": {
            "family": "Rigid",
            "rho": float(rho) if rho is not None else 1200.0,
            "friction": friction,
            "young": young,
            "poisson": poisson,
        },
        "render_color": render_color,
        "placement_margin": float(max(0.10, 0.30 * float(np.max(scaled_dim)))),
        "init_pos": motion["init_pos"],
        "init_euler": motion["init_euler"],
        "init_linvel": motion["init_linvel"],
        "init_angvel": motion["init_angvel"],
        "script_events": motion["script_events"],
    }


def sample_scene_cfg(scene_id: int, split: str, object_bank: List[Dict[str, Any]]) -> Dict[str, Any]:
    seed = 100000 + scene_id
    set_seed(seed)
    family = weighted_choice(SCENE_FAMILY_WEIGHTS)
    bg = sample_background()
    if family in ["line_chain_collision", "cross_fire", "stack_drop", "mixed_multi"]:
        n_obj = random.randint(max(3, MIN_OBJECTS_PER_SCENE), min(MAX_OBJECTS_PER_SCENE, 5))
    elif family in ["rest_then_hit", "late_entry"]:
        n_obj = random.randint(2, min(MAX_OBJECTS_PER_SCENE, 3))
    else:
        n_obj = random.randint(MIN_OBJECTS_PER_SCENE, min(MAX_OBJECTS_PER_SCENE, 4))
    n_obj = min(n_obj, len(object_bank))
    best_objects = None
    for _ in range(MAX_SCENE_SAMPLING_RETRIES):
        chosen = random.sample(object_bank, k=n_obj)
        scene_colors = pick_distinct_colors(n_obj)
        objects = []
        for i, bank_item in enumerate(chosen):
            obj = sample_physxnet_rigid_object(i, bank_item, family=family, existing_objects=objects, scene_colors=scene_colors)
            objects.append(obj)
        if scene_volume_ratio_ok(objects, MAX_VOLUME_RATIO_IN_SCENE):
            best_objects = objects
            break
        if best_objects is None:
            best_objects = objects
    objects = best_objects
    corner_cfg = build_corner_cfg_from_objects(objects, CORNER_BASE)
    camera = sample_camera_from_objects(objects, corner_cfg)
    return {
        "scene_id": f"{split}_scene_{scene_id:06d}",
        "split": split,
        "seed": seed,
        "family": family,
        "background": bg,
        "corner": corner_cfg,
        "camera": camera,
        "sim_options": {"gravity": [0.0, 0.0, -9.81], "dt": SIM_DT, "substeps": SIM_SUBSTEPS, "num_steps": SIM_NUM_STEPS, "fps": FPS},
        "objects": objects,
        "volume_ratio_limit": MAX_VOLUME_RATIO_IN_SCENE,
        "actual_volume_ratio": compute_scene_volume_ratio(objects),
        "proxy_mode": PROXY_MODE,
    }


# =========================================================
# Genesis 兼容辅助
# =========================================================
def _try_call_methods(obj: Any, method_names: Sequence[str], value: Any) -> bool:
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


def _try_get_methods(obj: Any, method_names: Sequence[str]) -> Any:
    for name in method_names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)()
            except Exception:
                pass
    return None


def query_entity_state(ent: Any) -> Dict[str, Optional[List[float]]]:
    def _to_list(x: Any) -> Optional[List[float]]:
        if x is None:
            return None
        return to_numpy_host(x).astype(float).reshape(-1).tolist()
    return {
        "position": _to_list(_try_get_methods(ent, ["get_pos", "get_position"])),
        "euler": _to_list(_try_get_methods(ent, ["get_euler", "get_rotation_euler"])),
        "linear_velocity": _to_list(_try_get_methods(ent, ["get_vel", "get_velocity", "get_linear_velocity"])),
        "angular_velocity": _to_list(_try_get_methods(ent, ["get_angvel", "get_angular_velocity"])),
    }


def apply_initial_motion_to_rigid_entity(ent: Any, obj_meta: Dict[str, Any]) -> None:
    v = np.asarray(obj_meta.get("init_linvel", [0.0, 0.0, 0.0]), dtype=np.float32)
    w = np.asarray(obj_meta.get("init_angvel", [0.0, 0.0, 0.0]), dtype=np.float32)
    if np.linalg.norm(v) > 0:
        _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], v)
    if np.linalg.norm(w) > 0:
        _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], w)


def find_safe_runtime_spawn_position(request_pos: Sequence[float], bbox_extents: Sequence[float], live_states: List[Dict[str, Any]], margin: float = 0.12) -> List[float]:
    req = np.asarray(request_pos, dtype=np.float32)
    ex = np.asarray(bbox_extents, dtype=np.float32)
    offsets = [
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.20], dtype=np.float32),
        np.asarray([0.0, 0.15, 0.0], dtype=np.float32),
        np.asarray([0.0, -0.15, 0.0], dtype=np.float32),
        np.asarray([0.15, 0.0, 0.0], dtype=np.float32),
        np.asarray([-0.15, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.35], dtype=np.float32),
    ]
    for off in offsets:
        pos = req + off
        ok = True
        for st in live_states:
            p = st.get("position")
            e = st.get("bbox_extents")
            if p is None or e is None:
                continue
            if aabb_overlap_3d(pos.tolist(), ex.tolist(), p, e, margin=margin):
                ok = False
                break
        if ok:
            return pos.astype(float).tolist()
    return (req + np.asarray([0.0, 0.0, 0.45], dtype=np.float32)).astype(float).tolist()


def apply_script_events(frame_idx: int, scene_cfg: Dict[str, Any], entities: List[Any]) -> List[Dict[str, Any]]:
    live_states = []
    for obj, ent in zip(scene_cfg["objects"], entities):
        st = query_entity_state(ent)
        live_states.append({"position": st["position"], "bbox_extents": obj["geom"]["bbox_extents"]})
    applied = []
    for obj, ent in zip(scene_cfg["objects"], entities):
        for ev in obj.get("script_events", []):
            if int(ev.get("frame_idx", -1)) != frame_idx:
                continue
            if ev["type"] == "set_motion":
                _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], np.asarray(ev["linvel"], dtype=np.float32))
                _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], np.asarray(ev["angvel"], dtype=np.float32))
                applied.append({"type": "set_motion", "frame_idx": frame_idx, "scene_object_id": obj["scene_object_id"]})
            elif ev["type"] == "teleport_and_set_motion":
                safe_pos = find_safe_runtime_spawn_position(ev["request_pos"], obj["geom"]["bbox_extents"], live_states)
                _try_call_methods(ent, ["set_pos", "set_position"], np.asarray(safe_pos, dtype=np.float32))
                _try_call_methods(ent, ["set_vel", "set_velocity", "set_linear_velocity"], np.asarray(ev["linvel"], dtype=np.float32))
                _try_call_methods(ent, ["set_ang", "set_angvel", "set_angular_velocity"], np.asarray(ev["angvel"], dtype=np.float32))
                obj.setdefault("runtime_spawn_log", []).append({"frame_idx": frame_idx, "requested": ev["request_pos"], "applied": safe_pos})
                applied.append({"type": "teleport_and_set_motion", "frame_idx": frame_idx, "scene_object_id": obj["scene_object_id"], "applied_pos": safe_pos})
    return applied


def add_large_corner(scene: Any, corner_cfg: Dict[str, Any]) -> None:
    cx, cy, cz = corner_cfg["center"]
    big = corner_cfg["panel_size"]
    thick = corner_cfg["thickness"]
    wall_mat = gs.materials.Rigid(rho=1200.0, friction=0.95)
    scene.add_entity(morph=gs.morphs.Box(size=(big, big, thick), pos=(cx + big * 0.5, cy + big * 0.5, cz - thick * 0.5), fixed=True), material=wall_mat, surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["floor"]))
    scene.add_entity(morph=gs.morphs.Box(size=(thick, big, big), pos=(cx - thick * 0.5, cy + big * 0.5, cz + big * 0.5), fixed=True), material=wall_mat, surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_x"]))
    scene.add_entity(morph=gs.morphs.Box(size=(big, thick, big), pos=(cx + big * 0.5, cy - thick * 0.5, cz + big * 0.5), fixed=True), material=wall_mat, surface=gs.surfaces.Default(color=CONTAINER_FACE_COLORS["wall_y"]))


def build_scene(scene_cfg: Dict[str, Any]) -> Tuple[Any, Any, List[Any]]:
    if gs is None:
        raise RuntimeError("genesis is not importable in this environment")
    vis_options = gs.options.VisOptions(
        show_world_frame=False,
        show_link_frame=False,
        background_color=tuple(scene_cfg["background"]["background_color"]),
        ambient_light=tuple(scene_cfg["background"]["ambient_light"]),
        segmentation_level="entity",
    )
    sim_options = gs.options.SimOptions(
        gravity=tuple(scene_cfg["sim_options"]["gravity"]),
        dt=scene_cfg["sim_options"]["dt"],
        substeps=scene_cfg["sim_options"]["substeps"],
    )
    scene_kwargs = dict(sim_options=sim_options, vis_options=vis_options, show_viewer=False)
    try:
        scene_kwargs["rigid_options"] = gs.options.RigidOptions(dt=scene_cfg["sim_options"]["dt"], enable_collision=True, use_gjk_collision=True)
    except Exception:
        pass
    scene = gs.Scene(**scene_kwargs)
    add_large_corner(scene, scene_cfg["corner"])
    entities = []
    for obj in scene_cfg["objects"]:
        mat = gs.materials.Rigid(rho=obj["material"]["rho"], friction=obj["material"]["friction"])
        surface = gs.surfaces.Default(color=obj["render_color"])
        # 关键改动：默认使用 collision proxy mesh 作为 morph，避免直接把 merged render mesh 用于碰撞
        ent = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=obj["geom"]["collision_mesh_file"],
                scale=obj["geom"]["scale"],
                pos=tuple(obj["init_pos"]),
                euler=tuple(obj["init_euler"]),
            ),
            material=mat,
            surface=surface,
        )
        entities.append(ent)
    cam = scene.add_camera(
        res=tuple(scene_cfg["camera"]["res"]),
        pos=tuple(scene_cfg["camera"]["pos"]),
        lookat=tuple(scene_cfg["camera"]["lookat"]),
        fov=scene_cfg["camera"]["fov"],
        GUI=False,
    )
    scene.build()
    for _ in range(WARMUP_STEPS):
        scene.step()
    for obj, ent in zip(scene_cfg["objects"], entities):
        apply_initial_motion_to_rigid_entity(ent, obj)
    return scene, cam, entities


def parse_render_output(render_out: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    rgb, depth, seg = None, None, None
    if isinstance(render_out, tuple):
        if len(render_out) >= 1:
            rgb = render_out[0]
        if len(render_out) >= 2:
            depth = render_out[1]
        if len(render_out) >= 3:
            seg = render_out[2]
    else:
        rgb = render_out
    rgb = None if rgb is None else to_uint8_image(rgb)
    depth = None if depth is None else to_numpy_host(depth)
    seg = None if seg is None else to_numpy_host(seg)
    return rgb, depth, seg


def extract_events(frame_states: Dict[int, List[Dict[str, Any]]], scene_cfg: Dict[str, Any], runtime_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = []
    floor_z = 0.0
    for obj_idx, obj in enumerate(scene_cfg["objects"]):
        hz = obj["geom"]["bbox_extents"][2] / 2.0
        grounded = False
        staticed = False
        for t in sorted(frame_states.keys()):
            st = frame_states[t][obj_idx]
            pos = st["position"]
            vel = st["linear_velocity"]
            if pos is not None and not grounded and pos[2] - hz <= floor_z + 0.02:
                events.append({"event_type": "first_ground_contact", "frame_idx": t, "scene_object_id": obj["scene_object_id"]})
                grounded = True
            if vel is not None and not staticed and np.linalg.norm(np.asarray(vel, dtype=np.float32)) < 0.05 and t > 10:
                events.append({"event_type": "near_static", "frame_idx": t, "scene_object_id": obj["scene_object_id"]})
                staticed = True
        if obj.get("runtime_spawn_log"):
            for r in obj["runtime_spawn_log"]:
                events.append({"event_type": "runtime_spawn_adjusted", "frame_idx": r["frame_idx"], "scene_object_id": obj["scene_object_id"], "requested": r["requested"], "applied": r["applied"]})
    for t in sorted(frame_states.keys()):
        poses = [x["position"] for x in frame_states[t]]
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                if poses[i] is None or poses[j] is None:
                    continue
                pi = np.asarray(poses[i], dtype=np.float32)
                pj = np.asarray(poses[j], dtype=np.float32)
                di = np.asarray(scene_cfg["objects"][i]["geom"]["bbox_extents"], dtype=np.float32)
                dj = np.asarray(scene_cfg["objects"][j]["geom"]["bbox_extents"], dtype=np.float32)
                thresh = 0.42 * float(np.linalg.norm(di) + np.linalg.norm(dj))
                if np.linalg.norm(pi - pj) < thresh:
                    events.append({"event_type": "approx_object_collision", "frame_idx": t, "obj_a": scene_cfg["objects"][i]["scene_object_id"], "obj_b": scene_cfg["objects"][j]["scene_object_id"]})
                    break
    events.extend(runtime_events)
    return events


def export_scene(scene_cfg: Dict[str, Any]) -> Dict[str, Any]:
    out_dir = OUT_ROOT / scene_cfg["split"] / scene_cfg["scene_id"]
    ensure_dir(out_dir / "rgb")
    ensure_dir(out_dir / "depth")
    ensure_dir(out_dir / "seg")
    ensure_dir(out_dir / "video")
    ensure_dir(out_dir / "ann")
    with open(out_dir / "ann" / "scene_input.json", "w", encoding="utf-8") as f:
        json.dump(scene_cfg, f, ensure_ascii=False, indent=2)
    prompt = (
        f"A physics simulation scene with {len(scene_cfg['objects'])} rigid objects. "
        f"Scene family: {scene_cfg['family']}. Collision uses proxy mode {scene_cfg['proxy_mode']}. "
        f"Same-scene volume ratio is limited to <= {scene_cfg['volume_ratio_limit']}, actual ratio {scene_cfg['actual_volume_ratio']:.3f}."
    )
    (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")

    scene = cam = entities = None
    preview_frames = []
    frame_states = {}
    runtime_events = []
    try:
        scene, cam, entities = build_scene(scene_cfg)
        num_steps = scene_cfg["sim_options"]["num_steps"]
        for t in range(num_steps):
            runtime_events.extend(apply_script_events(t, scene_cfg, entities))
            scene.step()
            try:
                render_out = cam.render(rgb=True, depth=True, segmentation=True)
            except Exception:
                render_out = cam.render(rgb=True)
            rgb, depth, seg = parse_render_output(render_out)
            if rgb is None:
                raise RuntimeError(f"render returned rgb=None at step {t}")
            imageio.imwrite(out_dir / "rgb" / f"{t:06d}.png", rgb)
            if depth is not None:
                np.save(out_dir / "depth" / f"{t:06d}.npy", depth)
            if seg is not None:
                np.save(out_dir / "seg" / f"{t:06d}.npy", seg)
            if t % PREVIEW_FRAME_STRIDE == 0:
                preview_frames.append(rgb)
            per_obj_states = []
            for obj, ent in zip(scene_cfg["objects"], entities):
                st = query_entity_state(ent)
                per_obj_states.append({
                    "scene_object_id": obj["scene_object_id"],
                    "position": st["position"],
                    "euler": st["euler"],
                    "linear_velocity": st["linear_velocity"],
                    "angular_velocity": st["angular_velocity"],
                })
            frame_states[t] = per_obj_states
        preview_path = save_preview_video(preview_frames, out_dir / "video" / "preview.mp4", fps=FPS)
        objects_json = []
        for obj in scene_cfg["objects"]:
            objects_json.append({
                "scene_object_id": obj["scene_object_id"],
                "object_id": obj["object_id"],
                "object_name": obj["object_name"],
                "category": obj["category"],
                "render_mesh_file": obj["geom"]["render_mesh_file"],
                "collision_mesh_file": obj["geom"]["collision_mesh_file"],
                "proxy_mode": obj["proxy_mode"],
                "scale": obj["geom"]["scale"],
                "bbox_extents": obj["geom"]["bbox_extents"],
                "render_color": obj["render_color"],
                "motion_type": obj["motion_type"],
                "rho": obj["material"]["rho"],
                "friction": obj["material"]["friction"],
                "init_pos": obj["init_pos"],
                "init_euler": obj["init_euler"],
                "init_linvel": obj["init_linvel"],
                "init_angvel": obj["init_angvel"],
                "script_events": obj.get("script_events", []),
            })
        with open(out_dir / "ann" / "objects.json", "w", encoding="utf-8") as f:
            json.dump(objects_json, f, ensure_ascii=False, indent=2)
        with open(out_dir / "ann" / "frames.jsonl", "w", encoding="utf-8") as f:
            for t in range(num_steps):
                row = {"frame_idx": t, "timestamp_sec": t * scene_cfg["sim_options"]["dt"], "objects": frame_states[t]}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        events = extract_events(frame_states, scene_cfg, runtime_events)
        with open(out_dir / "ann" / "events.json", "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        scene_metadata = {
            "scene_id": scene_cfg["scene_id"],
            "split": scene_cfg["split"],
            "seed": scene_cfg["seed"],
            "family": scene_cfg["family"],
            "proxy_mode": scene_cfg["proxy_mode"],
            "volume_ratio_limit": scene_cfg["volume_ratio_limit"],
            "actual_volume_ratio": scene_cfg["actual_volume_ratio"],
            "camera": scene_cfg["camera"],
            "corner": scene_cfg["corner"],
            "background": scene_cfg["background"],
            "sim_options": scene_cfg["sim_options"],
            "num_objects": len(scene_cfg["objects"]),
            "num_frames": num_steps,
            "preview_file": preview_path.name if preview_path is not None else None,
            "container_face_colors": CONTAINER_FACE_COLORS,
            "status": "ok",
        }
        with open(out_dir / "ann" / "scene_metadata.json", "w", encoding="utf-8") as f:
            json.dump(scene_metadata, f, ensure_ascii=False, indent=2)
        return scene_metadata
    finally:
        safe_scene_destroy(scene)


# =========================================================
# 主程序
# =========================================================
def main() -> None:
    ensure_dir(OUT_ROOT)
    ensure_dir(MERGED_CACHE_DIR)
    ensure_dir(PROXY_CACHE_DIR)
    for sp in ["train", "val", "test"]:
        ensure_dir(OUT_ROOT / sp)
    loader = PhysXNetGenesisLoader(
        root=str(PHYSXNET_ROOT),
        version=PHYSXNET_VERSION,
        merged_cache_dir=str(MERGED_CACHE_DIR),
        proxy_cache_dir=str(PROXY_CACHE_DIR),
        proxy_mode=PROXY_MODE,
    )
    object_bank, failed = build_physxnet_object_bank(loader=loader, max_objects_to_read=MAX_DATASET_OBJECTS_TO_READ)
    if len(object_bank) == 0:
        raise RuntimeError("No usable PhysXNet objects loaded.")
    split_bank = split_object_bank_by_id(object_bank)
    with open(OUT_ROOT / "object_bank_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "physxnet_root": str(PHYSXNET_ROOT),
            "version": PHYSXNET_VERSION,
            "proxy_mode": PROXY_MODE,
            "max_dataset_objects_to_read": MAX_DATASET_OBJECTS_TO_READ,
            "n_usable_objects": len(object_bank),
            "n_failed_objects": len(failed),
            "split_stats": {k: len(v) for k, v in split_bank.items()},
            "failed": failed,
        }, f, ensure_ascii=False, indent=2)
    backend_used = "none"
    if gs is not None:
        try:
            gs.init(backend=gs.gpu)
            backend_used = "gpu"
        except Exception:
            gs.init(backend=gs.cpu)
            backend_used = "cpu"
    manifest = {
        "dataset_name": "physxnet_proxy_dataset",
        "proxy_mode": PROXY_MODE,
        "n_scenes_requested": N_SCENES,
        "image_size": [IMG_W, IMG_H],
        "backend_used": backend_used,
        "sim_dt": SIM_DT,
        "sim_substeps": SIM_SUBSTEPS,
        "max_volume_ratio_in_scene": MAX_VOLUME_RATIO_IN_SCENE,
        "scenes": {"train": [], "val": [], "test": []},
        "failed_scenes": [],
    }
    try:
        split_order = ["train", "val", "test"]
        split_weights = np.asarray([0.8, 0.1, 0.1], dtype=np.float64)
        split_weights = split_weights / split_weights.sum()
        for sid in range(N_SCENES):
            split = str(np.random.choice(split_order, p=split_weights))
            bank = split_bank[split] if len(split_bank[split]) >= MIN_OBJECTS_PER_SCENE else split_bank["train"]
            scene_cfg = sample_scene_cfg(sid, split=split, object_bank=bank)
            try:
                print(f"[RUN ] {scene_cfg['scene_id']} | family={scene_cfg['family']} | n_obj={len(scene_cfg['objects'])} | vol_ratio={scene_cfg['actual_volume_ratio']:.3f} | proxy={scene_cfg['proxy_mode']}")
                meta = export_scene(scene_cfg)
                manifest["scenes"][split].append(meta)
                print(f"[ OK ] {scene_cfg['scene_id']} | preview={meta['preview_file']}")
            except Exception as e:
                err = {"scene_id": scene_cfg["scene_id"], "split": split, "family": scene_cfg["family"], "error": str(e)}
                manifest["failed_scenes"].append(err)
                print(f"[FAIL] {scene_cfg['scene_id']} | err={e}")
                if STOP_ON_ERROR:
                    raise
    finally:
        with open(OUT_ROOT / "dataset_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        if gs is not None:
            try:
                gs.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    main()
