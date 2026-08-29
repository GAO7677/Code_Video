#!/usr/bin/env python3
"""Prepare the natural-material R003 CYCLES refine experiment.

R003 keeps strict CYCLES samples and ground truth immutable. It only changes
the RGB material of role=dynamic actors and uses one semantic natural material
per control-variable family. Large parent data and texture assets are not
copied into the experiment directory.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819")
STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
EXPERIMENT_NAME = "R003_natural_common_textures_20260829"
EXPERIMENT_ROOT = STRICT_ROOT / "refine" / EXPERIMENT_NAME
GPU = "5"
SAMPLES = 32
BASKETBALL_TEXTURE = STRICT_ROOT / "refine/R001_v2v_obstacle_v140_basketball_texture/assets/balldimpled.png"
REALISM_TEXTURE_ROOT = Path(
    "/data/gaoya/agent-data/assets/texture_realism_backgrounds_20260825/textures"
)

FAMILY_MATERIALS = {
    "F11": "natural_basketball",
    "F12": "natural_oak_wood",
    "F12_RAMP_LENGTH": "natural_oak_wood",
    "SCENE_DOOR_FRAME": "natural_oak_wood",
    "SCENE_DOOR_FRAME_BALL": "natural_basketball",
    "SCENE_PUCK_BARRIER": "natural_black_rubber",
    "V2V_BOWL": "natural_basketball",
    "V2V_DOMINO": "natural_dark_wood",
    "V2V_GAP": "natural_basketball",
    "V2V_OBSTACLE": "natural_basketball",
    "V2V_OBSTACLE_SIZE": "natural_basketball",
    "V2V_PENDULUM": "natural_basketball",
    "V2V_PENDULUM_CABINET": "natural_basketball",
    "V2V_SEESAW": "natural_oak_wood",
}

MATERIAL_SOURCES = {
    "natural_basketball": {
        "kind": "UV image texture",
        "asset": "balldimpled.png",
        "root": str(BASKETBALL_TEXTURE.parent),
        "maps": {"albedo": BASKETBALL_TEXTURE.name},
        "page": "https://lpc.opengameart.org/content/basket-ball-texture",
        "file_url": "https://lpc.opengameart.org/sites/default/files/balldimpled.png",
        "license": "CC-BY 3.0",
        "attribution": "Downdate; collaborator Charlie",
    },
    "natural_oak_wood": {
        "kind": "image-backed PBR",
        "asset": "oak_wood_planks",
        "root": str(REALISM_TEXTURE_ROOT / "oak_wood_planks"),
        "maps": {
            "albedo": "oak_wood_planks_diff_2k.jpg",
            "normal": "oak_wood_planks_nor_gl_2k.jpg",
            "roughness": "oak_wood_planks_rough_2k.jpg",
            "ao": "oak_wood_planks_ao_2k.jpg",
        },
        "page": "https://polyhaven.com/a/oak_wood_planks",
        "license": "CC0 (Poly Haven asset package)",
    },
    "natural_dark_wood": {
        "kind": "image-backed PBR",
        "asset": "dark_wood",
        "root": str(REALISM_TEXTURE_ROOT / "dark_wood"),
        "maps": {
            "albedo": "dark_wood_diff_2k.jpg",
            "normal": "dark_wood_nor_gl_2k.jpg",
            "roughness": "dark_wood_rough_2k.jpg",
            "ao": "dark_wood_ao_2k.jpg",
        },
        "page": "https://polyhaven.com/a/dark_wood",
        "license": "CC0 (Poly Haven asset package)",
    },
    "natural_black_rubber": {
        "kind": "image-backed PBR with restrained dark rubber tint",
        "asset": "rubberized_track",
        "root": str(REALISM_TEXTURE_ROOT / "rubberized_track"),
        "maps": {
            "albedo": "rubberized_track_diff_2k.jpg",
            "normal": "rubberized_track_nor_gl_2k.jpg",
            "roughness": "rubberized_track_rough_2k.jpg",
            "ao": "rubberized_track_ao_2k.jpg",
        },
        "page": "https://polyhaven.com/a/rubberized_track",
        "license": "CC0 (Poly Haven asset package)",
    },
}


def load_batch_module():
    path = PROJECT_ROOT / "scripts/run_refine_distinct_texture_batch.py"
    spec = importlib.util.spec_from_file_location("r002_batch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_assets() -> None:
    for material, source in MATERIAL_SOURCES.items():
        root = Path(source["root"])
        for map_name in source["maps"].values():
            path = BASKETBALL_TEXTURE if material == "natural_basketball" else root / map_name
            if not path.is_file():
                raise FileNotFoundError(f"{material}: missing {path}")


def prepare() -> None:
    if (EXPERIMENT_ROOT / "case_selection.jsonl").exists():
        raise RuntimeError(f"experiment already exists: {EXPERIMENT_ROOT}")
    verify_assets()
    batch = load_batch_module()
    rows = batch.discover_target_cases(STRICT_ROOT)
    for row in rows:
        family = row["metadata"]["family_key"]
        if family not in FAMILY_MATERIALS:
            raise RuntimeError(f"missing family mapping: {family}")

    for subdir in ("cases", "shards", "logs", "evaluation", "truth_inheritance"):
        (EXPERIMENT_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    selections = []
    for row in rows:
        case_id = row["case_id"]
        family = row["metadata"]["family_key"]
        material = FAMILY_MATERIALS[family]
        actor_names = list(row["dynamic_actors"])
        shape = str(next(iter(row["dynamic_actors"].values())).get("shape", "unknown"))
        trajectory = __import__("numpy").load(row["trajectory_path"], allow_pickle=False)
        frame_count = int(trajectory["frame_times_s"].shape[0])
        fps = int(row["metadata"]["simulation"]["fps"])
        if frame_count != 90 or fps != 30:
            raise RuntimeError(f"{case_id}: expected 90 frames at 30 FPS, got {frame_count}/{fps}")
        case_root = EXPERIMENT_ROOT / "cases" / case_id
        selected = {
            "selected_material": material,
            "selected_material_source": MATERIAL_SOURCES[material],
            "override": {name: material for name in actor_names},
            "selection_method": "semantic_natural_material_by_object_shape",
            "family_key": family,
            "family_material": material,
            "family_appearance_invariant": True,
            "shape": shape,
            "dynamic_actor_names": actor_names,
        }
        inheritance = batch.truth_inheritance(STRICT_ROOT, case_id, row["sample_dir"])
        inheritance["experiment_id"] = EXPERIMENT_NAME
        case_selection = {
            "schema_version": "physv_cycles_refine_case_selection_v2",
            "experiment_id": EXPERIMENT_NAME,
            "case_id": case_id,
            "parent_sample_dir": str(row["sample_dir"]),
            "parent_render_report": str(row["report_path"]),
            "parent_custom_material_roots": row["report"].get("texture_sources", {}).get("custom_material_roots") or {},
            "family_key": family,
            "dynamic_actors": {
                name: {"role": actor.get("role"), "shape": actor.get("shape")}
                for name, actor in row["dynamic_actors"].items()
            },
            "selected": selected,
            "protocol": {
                "width": 896, "height": 512, "fps": fps,
                "frame_count": frame_count, "samples": SAMPLES,
            },
            "truth_inheritance": str(case_root / "truth_inheritance.json"),
            "status": "planned",
        }
        dump(case_root / "selection.json", case_selection)
        dump(case_root / "material_overrides.json", selected["override"])
        dump(case_root / "truth_inheritance.json", inheritance)
        selections.append(case_selection)

    manifest_path = EXPERIMENT_ROOT / "case_selection.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in selections),
        encoding="utf-8",
    )
    (EXPERIMENT_ROOT / "shards" / f"gpu{GPU}.txt").write_text(
        "\n".join(item["case_id"] for item in selections) + "\n", encoding="utf-8"
    )
    inheritance_manifest = {
        "schema_version": "physv_cycles_refine_inheritance_manifest_v1",
        "experiment_id": EXPERIMENT_NAME,
        "parent_root": str(STRICT_ROOT),
        "case_count": len(selections),
        "cases": {item["case_id"]: item["truth_inheritance"] for item in selections},
    }
    dump(EXPERIMENT_ROOT / "truth_inheritance" / "inheritance.json", inheritance_manifest)
    experiment = {
        "schema_version": "physv_cycles_refine_experiment_v3",
        "experiment_id": EXPERIMENT_NAME,
        "display_title": "R003 · strict CYCLES 自然常见材质变体",
        "variant_label": "R003 refine · natural semantic texture",
        "display_description": "同一 family_key 控制变量组内的 case 使用相同的自然材质：球体为篮球皮革，木制物体为自然木纹，Puck 为深色哑光橡胶。仅改变 role=dynamic 的 RGB 材质。",
        "display_protocol": "896×512 / 30 FPS / 90 frames / CYCLES / 32 samples / CUDA。",
        "status": "planned",
        "created_at": "2026-08-29",
        "parent_dataset": str(STRICT_ROOT),
        "target_selection": {
            "criterion": "videos/rgb_cycles.json texture_sources.custom_material_roots is empty",
            "case_count": len(selections),
            "excluded_custom_texture_cases": [
                "v2v_ramp_platform_l040", "v2v_ramp_platform_l080",
                "v2v_ramp_platform_l120", "v2v_ramp_platform_l160",
            ],
        },
        "change_scope": "RGB material only for metadata actors with role=dynamic",
        "appearance_invariant": "one semantic material per family_key; all dynamic actors within a case share that material",
        "family_materials": FAMILY_MATERIALS,
        "material_sources": MATERIAL_SOURCES,
        "basketball_texture": str(BASKETBALL_TEXTURE),
        "protocol": {
            "width": 896, "height": 512, "fps": 30, "frame_count": 90,
            "engine": "CYCLES", "samples": SAMPLES, "device": "CUDA",
        },
        "gpus_prepared": [GPU],
        "truth_inheritance_manifest": str(EXPERIMENT_ROOT / "truth_inheritance/inheritance.json"),
        "large_assets_copied": False,
        "runs": {
            "smoke": {"status": "not_started"},
            "full": {"status": "not_started"},
        },
    }
    dump(EXPERIMENT_ROOT / "experiment.json", experiment)
    (EXPERIMENT_ROOT / "README.md").write_text(
        f"""# R003：自然常见材质变体\n\n"
        f"本实验基于 strict CYCLES 的 70 个无 custom material root case，仅替换动态物体 RGB 材质；不改变几何、物理、相机、轨迹、分辨率、FPS、帧数或 strict GT。\n\n"
        f"## 外观规则\n\n"
        f"同一 `family_key` 控制变量组内所有 case 使用相同材质：\n\n"
        f"- 球体：`natural_basketball`，使用真实篮球皮革 UV 纹理；\n"
        f"- 木块、木箱、跷跷板载荷：`natural_oak_wood`；\n"
        f"- 多米诺：`natural_dark_wood`；\n"
        f"- Puck：`natural_black_rubber`，深色低对比橡胶微纹理。\n\n"
        f"映射完整记录在 `experiment.json` 和 `case_selection.jsonl`。父数据通过 `truth_inheritance.json` 引用，不复制大文件。\n\n"
        f"## 资产\n\n"
        f"篮球：`{BASKETBALL_TEXTURE}`；Poly Haven 材质：`{REALISM_TEXTURE_ROOT}`。\n\n"
        f"## 运行\n\n"
        f"```bash\n"
        f"cd {PROJECT_ROOT}\n"
        f"python3 scripts/run_refine_distinct_texture_batch.py render --experiment-root {EXPERIMENT_ROOT} --gpu {GPU} --case-list {EXPERIMENT_ROOT}/shards/gpu{GPU}.txt --mode smoke --samples {SAMPLES} --basketball-texture {BASKETBALL_TEXTURE}\n"
        f"python3 scripts/run_refine_distinct_texture_batch.py render --experiment-root {EXPERIMENT_ROOT} --gpu {GPU} --case-list {EXPERIMENT_ROOT}/shards/gpu{GPU}.txt --mode full --samples {SAMPLES} --basketball-texture {BASKETBALL_TEXTURE}\n"
        f"```\n\nGPU4 禁止使用。\n""",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment_root": str(EXPERIMENT_ROOT),
        "case_count": len(selections),
        "gpu": GPU,
        "family_materials": FAMILY_MATERIALS,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
