import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Any

import numpy as np
import trimesh


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

    # 针对 Genesis / 仿真最常用的简化字段
    joint_type: str  # fixed / revolute / prismatic / unknown


@dataclass
class GenesisObjectSpec:
    object_id: str
    object_name: str
    category: str

    dimension_cm: List[float]
    dimension_m: List[float]

    merged_mesh_path: str
    part_mesh_paths: List[str]

    # 适合 Genesis 构建整物体 rigid entity
    genesis_rigid: Dict[str, Any]

    # 若你想把每个 part 当单独 rigid entity 加进场景
    genesis_parts: List[Dict[str, Any]]

    parts: List[PartSpec]


# =========================================================
# 工具函数
# =========================================================

def parse_dimension_to_cm(dim_str: str) -> List[float]:
    """
    例如:
        "45*45*80" -> [45.0, 45.0, 80.0]
        "45 x 45 x 80" -> [45.0, 45.0, 80.0]
    """
    if not dim_str:
        return [0.0, 0.0, 0.0]

    s = dim_str.lower().replace("×", "*").replace("x", "*")
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    vals = [float(x) for x in nums[:3]]
    if len(vals) < 3:
        vals = vals + [0.0] * (3 - len(vals))
    return vals


def cm_to_m(vals_cm: List[float]) -> List[float]:
    return [v / 100.0 for v in vals_cm]


def parse_density_to_kgm3(density_str: Optional[str]) -> Optional[float]:
    """
    支持:
        "1.2 g/cm^3" -> 1200
        "7.8 g/cm^3" -> 7800
        "1200 kg/m^3" -> 1200
    """
    if not density_str:
        return None

    s = density_str.strip().lower()
    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if not nums:
        return None

    value = float(nums[0])

    if "g/cm" in s:
        return value * 1000.0
    if "kg/m" in s:
        return value

    # 单位不明时，保守假设是 g/cm^3（该数据集常见格式）
    return value * 1000.0


def parse_youngs_to_pa(young_gpa: Optional[float]) -> Optional[float]:
    if young_gpa is None:
        return None
    return float(young_gpa) * 1e9


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def infer_joint_type(movement_description: str) -> str:
    """
    这是一个简化规则。
    你的 version_1 JSON 里大多是自然语言描述，没有完整 joint 参数。
    """
    if not movement_description:
        return "unknown"

    s = movement_description.lower()

    fixed_keywords = [
        "rigidly fixed",
        "rigidly connected",
        "no relative movement",
        "fixed to",
        "fixed with",
        "rigidly attached",
        "no movement",
    ]
    revolute_keywords = [
        "rotate",
        "rotates",
        "rotating",
        "revolve",
        "revolves",
        "hinge",
        "swivel",
    ]
    prismatic_keywords = [
        "slide",
        "slides",
        "sliding",
        "pull out",
        "push in",
        "translate",
        "translates",
        "telescopic",
    ]

    for kw in fixed_keywords:
        if kw in s:
            return "fixed"
    for kw in revolute_keywords:
        if kw in s:
            return "revolute"
    for kw in prismatic_keywords:
        if kw in s:
            return "prismatic"

    return "unknown"


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load mesh as trimesh.Trimesh: {mesh_path}")
    return mesh


def merge_meshes(mesh_paths: List[Path], export_path: Path) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)

    meshes = []
    for p in mesh_paths:
        if not p.exists():
            continue
        m = load_mesh(p)
        meshes.append(m)

    if not meshes:
        raise FileNotFoundError(f"No valid meshes found for merge: {mesh_paths}")

    merged = trimesh.util.concatenate(meshes)
    merged.merge_vertices()
    merged.export(export_path)
    return export_path


# =========================================================
# 主 Loader
# =========================================================

class PhysXNetGenesisLoader:
    """
    目录假设：
        root/
          version_1/
            finaljson/
              712.json
            partseg/
              712/
                objs/
                  0.obj
                  1.obj
                imgs/
                  0_xxx.png
    """

    def __init__(
        self,
        root: str,
        version: str = "version_1",
        merged_cache_dir: Optional[str] = None,
        merge_ext: str = ".obj",
    ):
        self.root = Path(root)
        self.version = version
        self.base_dir = self.root / version
        self.finaljson_dir = self.base_dir / "finaljson"
        self.partseg_dir = self.base_dir / "partseg"

        if merged_cache_dir is None:
            merged_cache_dir = str(self.base_dir / "_merged_for_genesis")
        self.merged_cache_dir = Path(merged_cache_dir)
        self.merge_ext = merge_ext

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

    def _build_part_spec(
        self,
        obj_id: str,
        part_info: Dict[str, Any],
        objs_dir: Path,
        imgs_dir: Path,
    ) -> PartSpec:
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
        """
        这是一个“适合 Genesis 使用”的中间格式，不直接依赖 Genesis import。
        你后面可以很容易映射到：
            gs.morphs.Mesh(file=...)
            gs.materials.Rigid(...)
        """
        return {
            "name": part.name,
            "part_id": part.part_id,
            "entity_type": "rigid",
            "morph": {
                "type": "mesh",
                "file": part.mesh_path,
            },
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
        merged_mesh_path: Path,
        parts: List[PartSpec],
    ) -> Dict[str, Any]:
        """
        用 merged mesh 构整物体 rigid entity。
        材料上这里给出一个默认策略：
        - 如果所有 part 密度一致，直接用该值
        - 否则取平均值
        """
        densities = [p.density_kgm3 for p in parts if p.density_kgm3 is not None]
        youngs = [p.youngs_modulus_pa for p in parts if p.youngs_modulus_pa is not None]
        poissons = [p.poisson_ratio for p in parts if p.poisson_ratio is not None]

        avg_density = float(np.mean(densities)) if densities else None
        avg_young = float(np.mean(youngs)) if youngs else None
        avg_poisson = float(np.mean(poissons)) if poissons else None

        return {
            "name": f"{object_name}_{obj_id}",
            "entity_type": "rigid",
            "morph": {
                "type": "mesh",
                "file": str(merged_mesh_path),
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
                "note": "Merged mesh from PhysXNet parts; articulation not reconstructed.",
            },
        }

    def get_object(self, obj_id: str, export_merged: bool = True) -> GenesisObjectSpec:
        json_path = self.finaljson_dir / f"{obj_id}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"JSON not found: {json_path}")

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
        parts: List[PartSpec] = [
            self._build_part_spec(obj_id, pinfo, objs_dir, imgs_dir)
            for pinfo in sorted(parts_info, key=lambda x: int(x["label"]))
        ]

        part_mesh_paths = [Path(p.mesh_path) for p in parts if Path(p.mesh_path).exists()]
        if len(part_mesh_paths) == 0:
            raise FileNotFoundError(f"No part meshes found for object {obj_id}")

        merged_mesh_path = self.merged_cache_dir / obj_id / f"merged{self.merge_ext}"
        if export_merged:
            if not merged_mesh_path.exists():
                merge_meshes(part_mesh_paths, merged_mesh_path)
        else:
            merged_mesh_path.parent.mkdir(parents=True, exist_ok=True)

        genesis_parts = [self._build_genesis_part_dict(p) for p in parts]
        genesis_rigid = self._build_genesis_rigid_dict(
            obj_id=obj_id,
            object_name=object_name,
            merged_mesh_path=merged_mesh_path,
            parts=parts,
        )

        return GenesisObjectSpec(
            object_id=obj_id,
            object_name=object_name,
            category=category,
            dimension_cm=dim_cm,
            dimension_m=dim_m,
            merged_mesh_path=str(merged_mesh_path),
            part_mesh_paths=[str(p) for p in part_mesh_paths],
            genesis_rigid=genesis_rigid,
            genesis_parts=genesis_parts,
            parts=parts,
        )

    def iter_objects(self, export_merged: bool = True) -> Iterator[GenesisObjectSpec]:
        for obj_id in self.list_object_ids():
            try:
                yield self.get_object(obj_id=obj_id, export_merged=export_merged)
            except Exception as e:
                print(f"[WARN] skip object {obj_id}: {e}")

    def save_index_json(self, save_path: str, export_merged: bool = True) -> None:
        save_path = str(save_path)
        records = []
        for obj in self.iter_objects(export_merged=export_merged):
            records.append({
                "object_id": obj.object_id,
                "object_name": obj.object_name,
                "category": obj.category,
                "dimension_cm": obj.dimension_cm,
                "dimension_m": obj.dimension_m,
                "merged_mesh_path": obj.merged_mesh_path,
                "part_mesh_paths": obj.part_mesh_paths,
                "genesis_rigid": obj.genesis_rigid,
                "genesis_parts": obj.genesis_parts,
                "parts": [asdict(p) for p in obj.parts],
            })

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


# =========================================================
# 可选：把 loader 输出转成 Genesis 真正可调用的数据
# =========================================================

def build_genesis_entity_kwargs_from_rigid_spec(rigid_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    返回可直接映射到你自己的 Genesis scene.add(...) 的参数字典。
    不直接 import genesis，避免环境里没装时脚本无法运行。
    """
    return {
        "name": rigid_spec["name"],
        "mesh_file": rigid_spec["morph"]["file"],
        "density": rigid_spec["material"]["density"],
        "youngs_modulus": rigid_spec["material"]["youngs_modulus"],
        "poisson_ratio": rigid_spec["material"]["poisson_ratio"],
    }


def build_genesis_entity_kwargs_from_part_spec(part_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": part_spec["name"],
        "mesh_file": part_spec["morph"]["file"],
        "density": part_spec["material"]["density"],
        "youngs_modulus": part_spec["material"]["youngs_modulus"],
        "poisson_ratio": part_spec["material"]["poisson_ratio"],
        "joint_type": part_spec["semantic"]["joint_type"],
    }


# =========================================================
# 示例
# =========================================================

# if __name__ == "__main__":
#     root = "/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet"

#     loader = PhysXNetGenesisLoader(
#         root=root,
#         version="version_1",
#         merged_cache_dir="/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/_merged_for_genesis",
#     )

#     print(f"Total objects: {len(loader)}")

#     # 读取单个物体
#     obj = loader.get_object("712", export_merged=True)
#     print("=== Single Object ===")
#     print("object_id:", obj.object_id)
#     print("object_name:", obj.object_name)
#     print("merged_mesh_path:", obj.merged_mesh_path)
#     print("num_parts:", len(obj.parts))
#     print("genesis_rigid:", obj.genesis_rigid)

#     # 遍历整个数据集并导出索引
#     loader.save_index_json(
#         save_path="/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/physxnet_genesis_index.json",
#         export_merged=True,
#     )
#     print("Saved dataset index json.")


'''




'''