#!/usr/bin/env python3
"""Archive flat-color R002 outputs and switch the experiment to visible textures."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict/refine/R002_all_cases_distinct_texture_20260829")
ARCHIVE_MANIFEST = ROOT / "case_selection_flat_color_v1.jsonl"
VISIBLE_TEXTURE_ROOT = Path("/data/gaoya/agent-data/assets/polyhaven_textures_20260829/wood_peeling_paint_weathered")
WOOD_MAPS = {
    "albedo": "wood_peeling_paint_weathered_diff_2k.jpg",
    "normal": "wood_peeling_paint_weathered_nor_gl_2k.jpg",
    "roughness": "wood_peeling_paint_weathered_rough_2k.jpg",
    "ao": "wood_peeling_paint_weathered_ao_2k.jpg",
}
MATERIAL_MAP = {
    "blue_rubber": "refine_blue_texture",
    "blue_painted": "refine_blue_texture",
    "green_painted": "refine_green_texture",
    "fabric_green": "refine_green_texture",
    "teal_metal": "refine_teal_texture",
    "yellow_rubber": "refine_yellow_texture",
    "yellow_metal": "refine_yellow_texture",
    "coral_painted": "refine_coral_texture",
    "red_rubber": "refine_red_texture",
    "dark_metal": "refine_charcoal_texture",
}
PALETTE = {
    "refine_blue_texture": (0.10, 0.32, 1.00),
    "refine_green_texture": (0.10, 0.84, 0.28),
    "refine_teal_texture": (0.05, 0.76, 0.76),
    "refine_yellow_texture": (1.00, 0.72, 0.08),
    "refine_coral_texture": (1.00, 0.20, 0.10),
    "refine_red_texture": (0.95, 0.07, 0.04),
    "refine_charcoal_texture": (0.06, 0.09, 0.14),
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    manifest_path = ROOT / "case_selection.jsonl"
    original_manifest = manifest_path.read_text(encoding="utf-8")
    if not ARCHIVE_MANIFEST.exists():
        ARCHIVE_MANIFEST.write_text(original_manifest, encoding="utf-8")

    selections = [json.loads(line) for line in original_manifest.splitlines() if line.strip()]
    if len(selections) != 70:
        raise RuntimeError(f"expected 70 R002 selections, found {len(selections)}")

    source_experiment = load(ROOT / "experiment.json")
    source_experiment.setdefault("rerender_history", []).append({
        "version": "flat_color_v1",
        "status": source_experiment.get("status", "partial"),
        "archived_manifest": str(ARCHIVE_MANIFEST),
        "archived_render_subdir": "render/full_flat_color_v1",
        "reason": "selected image-backed PBR maps were visually flattened by strong tint on small actors",
    })
    source_experiment["visible_texture_repair"] = {
        "version": "visible_texture_v2",
        "status": "planned",
        "texture_package": "wood_peeling_paint_weathered",
        "texture_root": str(VISIBLE_TEXTURE_ROOT),
        "maps": WOOD_MAPS,
        "blend_mode": "COLOR",
        "blend_factor": 0.72,
        "uv_scale": 1.0,
        "normal_strength": 0.58,
        "palette": PALETTE,
        "old_outputs_preserved_under": "render/full_flat_color_v1",
    }
    source_experiment["status"] = "planned"
    source_experiment["progress"] = {"completed": 0, "failed": 0, "pending": 70, "total": 70}
    source_experiment.setdefault("runs", {})["full"] = {
        "status": "planned", "completed": 0, "failed": 0, "total": 70,
    }
    source_experiment["runs"]["flat_color_v1"] = {
        "status": "archived", "completed": 23, "failed": 0, "total": 70,
    }

    new_lines = []
    for selection in selections:
        case_id = selection["case_id"]
        case_root = ROOT / "cases" / case_id
        selection_path = case_root / "selection.json"
        original_selection_path = case_root / "selection_flat_color_v1.json"
        if not original_selection_path.exists():
            shutil.copy2(selection_path, original_selection_path)
        old_selection = load(selection_path)
        selected = selection["selected"]
        old_material = selected.get("original_selected_material", selected["selected_material"])
        if old_material not in MATERIAL_MAP:
            raise RuntimeError(f"{case_id}: no visible material mapping for {old_material}")
        new_material = MATERIAL_MAP[old_material]

        for old_subdir, archive_subdir in (("full", "full_flat_color_v1"), ("smoke", "smoke_flat_color_v1")):
            old_dir = case_root / "render" / old_subdir
            archive_dir = case_root / "render" / archive_subdir
            if old_dir.exists() and not archive_dir.exists():
                old_dir.rename(archive_dir)
        old_overrides = case_root / "material_overrides.json"
        archived_overrides = case_root / "material_overrides_flat_color_v1.json"
        if old_overrides.exists() and not archived_overrides.exists():
            shutil.copy2(old_overrides, archived_overrides)

        updated = copy.deepcopy(selection)
        updated["status"] = "planned"
        updated["selected"]["original_selected_material"] = old_material
        updated["selected"]["selected_material"] = new_material
        updated["selected"]["selected_material_source"] = {
            "package": "wood_peeling_paint_weathered",
            "root": str(VISIBLE_TEXTURE_ROOT),
            "page": "https://polyhaven.com/a/wood_peeling_paint_weathered",
            "maps": WOOD_MAPS,
            "license": "CC0 (Poly Haven asset package)",
        }
        updated["selected"]["override"] = {
            name: new_material for name in selected["override"]
        }
        updated["selected"]["visible_texture_repair"] = {
            "version": "visible_texture_v2",
            "replaced_flat_color_material": old_material,
            "palette_color": PALETTE[new_material],
            "blend_mode": "COLOR",
            "blend_factor": 0.72,
        }
        updated["flat_color_v1_selection"] = str(original_selection_path)
        dump(selection_path, updated)
        dump(old_overrides, updated["selected"]["override"])
        new_lines.append(json.dumps(updated, ensure_ascii=False, separators=(",", ":")))

    manifest_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    dump(ROOT / "experiment.json", source_experiment)

    for log_name in ("gpu5_full.results.jsonl", "gpu5_smoke.results.jsonl"):
        log_path = ROOT / "logs" / log_name
        if log_path.exists():
            archived_log = log_path.with_name(log_path.name.replace(".results.jsonl", "_flat_color_v1.results.jsonl"))
            if not archived_log.exists():
                log_path.rename(archived_log)

    readme = f"""# R002：全部无 custom material root Case 的可见纹理变体

## 当前状态

修正版 `visible_texture_v2` 已完成元数据切换，正式渲染尚未完成：70/70 pending。当前页面只读取 `render/full/`；浏览器刷新即可看到新结果。

旧版 `flat_color_v1` 的已生成结果没有删除，已归档到每个 case 的 `render/full_flat_color_v1/` 和 `render/smoke_flat_color_v1/`，原选择清单保存在 `case_selection_flat_color_v1.jsonl`，用于追溯为什么之前看起来接近纯色。

## 修正版材质

旧版虽然连接了 image-backed PBR 节点，但 `rubber_tiles`、`concrete_floor_worn_001` 等贴图在小球/小方块上又被 `tint_strength=0.80–0.92` 压低了纹理对比度。修正版使用已有 Poly Haven `wood_peeling_paint_weathered` 的 albedo、normal、roughness 和 AO，并通过 Blender `COLOR` blend mode 将高对比度木纹亮度/细节保留、只改变色相和饱和度。

- albedo root: `{VISIBLE_TEXTURE_ROOT}`
- maps: `{", ".join(WOOD_MAPS.values())}`
- COLOR blend factor: `0.72`
- Generated coordinate scale: `1.0`
- normal strength: `0.58`
- 只改变 `metadata.actors[*].role=dynamic` 的 RGB 材质；几何、物理、相机、轨迹、分辨率、FPS、帧数和 strict GT 均不变。

## 目录与协议

实验目录包含 `README.md`、`experiment.json`、`case_selection.jsonl`、`shards/`、`cases/<case_id>/selection.json`、`material_overrides.json`、`truth_inheritance.json`、`render/full/`、`logs/`、`evaluation/` 和 `index.html`。

正式协议：`896×512 / 30 FPS / 90 frames / CYCLES / 32 samples / CUDA`；GPU4 禁止使用。

## 运行命令

```bash
cd /home/gaoya/Code_Video/Dataset_physv_v2v_0819
python3 scripts/run_refine_distinct_texture_batch.py render --gpu 5 --case-list {ROOT}/shards/gpu5.txt --mode full --samples 32
```
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"experiment_root": str(ROOT), "cases": len(selections), "status": "planned"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
