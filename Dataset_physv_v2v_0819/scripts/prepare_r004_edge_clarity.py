#!/usr/bin/env python3
"""Prepare the R004 render-only edge-clarity refine experiment.

R004 reuses R003's 70 strict cases and semantic material assignments.  It
adds only render-mesh edge treatment for hard non-sphere actors; the parent
sample, collision proxy, physical parameters, camera, trajectory and all GT
remain inherited and immutable.
"""
from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819")
STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
R003_ROOT = STRICT_ROOT / "refine/R003_natural_common_textures_20260829"
EXPERIMENT_NAME = "R004_natural_texture_edge_clarity_20260829"
EXPERIMENT_ROOT = STRICT_ROOT / "refine" / EXPERIMENT_NAME
GPU = "5"
SAMPLES = 32
BASKETBALL_TEXTURE = STRICT_ROOT / "refine/R001_v2v_obstacle_v140_basketball_texture/assets/balldimpled.png"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare() -> None:
    manifest_path = R003_ROOT / "case_selection.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if (EXPERIMENT_ROOT / "case_selection.jsonl").exists():
        raise RuntimeError(f"experiment already exists: {EXPERIMENT_ROOT}")

    source_rows = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        source_rows.append(load(R003_ROOT / "cases" / line["case_id"] / "selection.json"))
    if len(source_rows) != 70:
        raise RuntimeError(f"expected 70 R003 selections, found {len(source_rows)}")

    for subdir in ("cases", "shards", "logs", "evaluation", "truth_inheritance", "assets"):
        (EXPERIMENT_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    selections = []
    for source in source_rows:
        case_id = source["case_id"]
        case_root = EXPERIMENT_ROOT / "cases" / case_id
        selected = dict(source["selected"])
        selected["edge_clarity"] = {
            "enabled": True,
            "scope": "non-sphere actors only",
            "render_mesh_only": True,
        }

        inheritance_source = load(Path(source["truth_inheritance"]))
        inheritance_source["experiment_id"] = EXPERIMENT_NAME
        inheritance_source["change_scope"] = (
            "R003 semantic RGB material plus render-only edge clarity on non-sphere actors"
        )
        inheritance_source["unchanged"] = list(dict.fromkeys(
            inheritance_source.get("unchanged", [])
            + ["collision geometry", "physical parameters", "render camera", "strict GT"]
        ))

        row = dict(source)
        row["experiment_id"] = EXPERIMENT_NAME
        row["selected"] = selected
        row["protocol"] = dict(source["protocol"])
        row["protocol"]["edge_clarity"] = True
        row["truth_inheritance"] = str(case_root / "truth_inheritance.json")
        row["status"] = "planned"
        dump(case_root / "selection.json", row)
        dump(case_root / "material_overrides.json", selected["override"])
        dump(case_root / "truth_inheritance.json", inheritance_source)
        selections.append(row)

    (EXPERIMENT_ROOT / "case_selection.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selections),
        encoding="utf-8",
    )
    (EXPERIMENT_ROOT / f"shards/gpu{GPU}.txt").write_text(
        "".join(row["case_id"] + "\n" for row in selections), encoding="utf-8"
    )
    dump(EXPERIMENT_ROOT / "truth_inheritance/inheritance.json", {
        "schema_version": "physv_cycles_refine_inheritance_manifest_v1",
        "experiment_id": EXPERIMENT_NAME,
        "parent_root": str(STRICT_ROOT),
        "appearance_parent_experiment": str(R003_ROOT),
        "case_count": len(selections),
        "cases": {row["case_id"]: row["truth_inheritance"] for row in selections},
    })
    (EXPERIMENT_ROOT / "assets/README.md").write_text(
        "R004 不复制大纹理资产；材质资产沿用 R003 的本地引用，见 experiment.json。\n",
        encoding="utf-8",
    )
    dump(EXPERIMENT_ROOT / "experiment.json", {
        "schema_version": "physv_cycles_refine_experiment_v4",
        "experiment_id": EXPERIMENT_NAME,
        "display_title": "R004 · strict CYCLES 自然材质清晰边缘变体",
        "variant_label": "R004 refine · natural texture + edge clarity",
        "display_description": (
            "沿用 R003 的篮球、自然木材和深灰橡胶材质；仅对非球体动态物体增加轻微渲染倒角、"
            "加权法线和克制的边缘高光，使木块、骨牌和 Puck 的轮廓更清晰。"
        ),
        "display_protocol": "896×512 / 30 FPS / 90 frames / CYCLES / 32 samples / CUDA。",
        "status": "planned",
        "created_at": "2026-08-29",
        "parent_dataset": str(STRICT_ROOT),
        "appearance_parent_experiment": str(R003_ROOT),
        "target_selection": {
            "criterion": "R003's 70 strict cases with semantic natural materials",
            "case_count": len(selections),
        },
        "change_scope": (
            "R003 material assignment plus render-only edge treatment on dynamic actors with shape box/cylinder/puck"
        ),
        "unchanged": [
            "collision proxy", "physics", "camera", "trajectory", "strict GT",
            "resolution", "FPS", "frame count", "basketball geometry",
        ],
        "edge_clarity": {
            "scope": ["box", "cylinder", "puck"],
            "excluded": ["sphere / basketball"],
            "bevel": "small render-only bevel, max 0.036 m for boxes and 0.030 m for cylinders/pucks",
            "normals": "Weighted Normal modifier with keep_sharp",
            "highlight": "restrained Fresnel grazing-angle base-color highlight",
            "collision_and_gt_changed": False,
        },
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
    })
    (EXPERIMENT_ROOT / "README.md").write_text(
        f"""# R004：自然材质 + 非球体清晰边缘

R004 基于 R003 的 70 个 strict CYCLES case。它沿用 R003 的语义材质和同 family 外观一致性，仅对非球体动态物体的渲染网格增加小倒角、加权法线和克制的 Fresnel 边缘高光。

## 严格不变项

碰撞代理、物理参数、相机、逐帧轨迹、分辨率、FPS、帧数和全部 strict GT 均从父数据继承；倒角只存在于渲染网格，不参与物理或 GT。篮球保持 R003 的球体和纹理。

## 可复现运行

```bash
cd {PROJECT_ROOT}
python3 scripts/run_refine_distinct_texture_batch.py render --experiment-root {EXPERIMENT_ROOT} --gpu {GPU} --case-list {EXPERIMENT_ROOT}/shards/gpu{GPU}.txt --mode smoke --samples {SAMPLES} --basketball-texture {BASKETBALL_TEXTURE} --edge-clarity
python3 scripts/run_refine_distinct_texture_batch.py render --experiment-root {EXPERIMENT_ROOT} --gpu {GPU} --case-list {EXPERIMENT_ROOT}/shards/gpu{GPU}.txt --mode full --samples {SAMPLES} --basketball-texture {BASKETBALL_TEXTURE} --edge-clarity
```

GPU4 禁止使用。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment_root": str(EXPERIMENT_ROOT),
        "case_count": len(selections),
        "gpu": GPU,
        "parent_experiment": str(R003_ROOT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare()
