#!/usr/bin/env python3
"""Prepare and render deterministic texture variants for strict CYCLES cases.

The parent strict samples are never modified. This experiment only changes the
RGB material assigned to actors whose metadata role is exactly dynamic; poses,
physics, camera metadata and all strict ground truth remain inherited from the
parent sample.

Subcommands:
  prepare  audit samples and write per-case material choices and shards
  render   render one shard sequentially with Blender Cycles
  index    build a static parent-vs-variant browser index
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
EXPERIMENT_NAME = "R002_all_cases_distinct_texture_20260829"
EXPERIMENT_ROOT = STRICT_ROOT / "refine" / EXPERIMENT_NAME
BLENDER_SCRIPT = PROJECT_ROOT / "scripts" / "render_physv_cycles.py"
DEFAULT_BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
DEFAULT_FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
EXPERIMENT_SEED = "20260829-R002"
EXPECTED_TARGET_COUNT = 70
MIN_COLOR_DISTANCE = 0.20
FORBIDDEN_GPUS = {"4"}

# These are only conservative display-space approximations for a pre-render
# contrast audit. Actual materials are image-backed PBR materials defined by
# render_physv_cycles.py.
MATERIAL_RGB: dict[str, tuple[float, float, float]] = {
    "floor": (0.46, 0.32, 0.18),
    "floor_cool": (0.44, 0.56, 0.70),
    "floor_dark_wood": (0.12, 0.15, 0.20),
    "floor_concrete": (0.28, 0.33, 0.40),
    "floor_terracotta": (0.62, 0.24, 0.12),
    "floor_slate": (0.10, 0.14, 0.21),
    "wood": (0.48, 0.23, 0.09),
    "red_wood": (0.78, 0.19, 0.09),
    "wood_peeling_paint": (0.84, 0.27, 0.08),
    "floor_stone_pavement": (0.38, 0.40, 0.42),
    "wall": (0.63, 0.58, 0.49),
    "wall_cool": (0.57, 0.66, 0.78),
    "wall_green": (0.39, 0.58, 0.43),
    "wall_gray": (0.54, 0.59, 0.68),
    "wall_rose": (0.72, 0.40, 0.35),
    "wall_charcoal": (0.10, 0.12, 0.16),
    "concrete": (0.43, 0.49, 0.58),
    "picture_surface": (0.18, 0.42, 0.47),
    "red_rubber": (0.88, 0.05, 0.025),
    "blue_rubber": (0.025, 0.18, 0.90),
    "yellow_rubber": (0.93, 0.68, 0.02),
    "domino_wood": (0.43, 0.15, 0.045),
    "blue_painted": (0.04, 0.25, 0.88),
    "teal_metal": (0.025, 0.66, 0.70),
    "barrier_metal": (0.035, 0.47, 0.53),
    "yellow_metal": (0.92, 0.52, 0.025),
    "dark_metal": (0.07, 0.09, 0.13),
    "white_painted": (0.76, 0.82, 0.88),
    "green_painted": (0.05, 0.70, 0.20),
    "coral_painted": (0.96, 0.19, 0.06),
    "window_glass": (0.055, 0.13, 0.19),
    "fabric": (0.38, 0.50, 0.64),
    "fabric_green": (0.08, 0.63, 0.22),
    "fabric_coral": (0.96, 0.20, 0.08),
    "rope_fabric": (0.42, 0.18, 0.04),
}

TEXTURE_SOURCES: dict[str, dict[str, str]] = {
    "rubber_tiles": {
        "root": "/data/gaoya/agent-data/assets/polyhaven_textures_20260820/rubber_tiles",
        "page": "https://polyhaven.com/a/rubber_tiles",
        "albedo": "rubber_tiles_diff_2k.jpg",
        "normal": "rubber_tiles_nor_gl_2k.jpg",
        "roughness": "rubber_tiles_rough_2k.jpg",
        "ao": "rubber_tiles_ao_2k.jpg",
    },
    "metal_plate": {
        "root": "/data/gaoya/agent-data/assets/polyhaven_textures_20260820/metal_plate",
        "page": "https://polyhaven.com/a/metal_plate",
        "albedo": "metal_plate_diff_2k.jpg",
        "normal": "metal_plate_nor_gl_2k.jpg",
        "roughness": "metal_plate_rough_2k.jpg",
        "ao": "metal_plate_ao_2k.jpg",
    },
    "concrete_floor_worn_001": {
        "root": "/data/gaoya/agent-data/assets/polyhaven_textures_20260820/concrete_floor_worn_001",
        "page": "https://polyhaven.com/a/concrete_floor_worn_001",
        "albedo": "concrete_floor_worn_001_diff_2k.jpg",
        "normal": "concrete_floor_worn_001_nor_gl_2k.jpg",
        "roughness": "concrete_floor_worn_001_rough_2k.jpg",
        "ao": "concrete_floor_worn_001_ao_2k.jpg",
    },
    "denim_fabric_04": {
        "root": "/data/gaoya/agent-data/assets/polyhaven_textures_20260820/denim_fabric_04",
        "page": "https://polyhaven.com/a/denim_fabric_04",
        "albedo": "denim_fabric_04_diff_2k.jpg",
        "normal": "denim_fabric_04_nor_gl_2k.jpg",
        "roughness": "denim_fabric_04_rough_2k.jpg",
        "ao": "denim_fabric_04_ao_2k.jpg",
    },
    "wood_floor": {
        "root": "/data/gaoya/dataset/blender_render_assets/polyhaven_v1/textures/wood_floor",
        "page": "https://polyhaven.com/a/wood_floor",
        "albedo": "wood_floor_diff_2k.jpg",
        "normal": "wood_floor_nor_gl_2k.jpg",
        "roughness": "wood_floor_rough_2k.jpg",
        "ao": "wood_floor_ao_2k.jpg",
    },
}

MATERIAL_SOURCE: dict[str, str] = {
    "red_rubber": "rubber_tiles",
    "blue_rubber": "rubber_tiles",
    "yellow_rubber": "rubber_tiles",
    "blue_painted": "concrete_floor_worn_001",
    "teal_metal": "concrete_floor_worn_001",
    "white_painted": "concrete_floor_worn_001",
    "green_painted": "concrete_floor_worn_001",
    "coral_painted": "concrete_floor_worn_001",
    "concrete": "concrete_floor_worn_001",
    "yellow_metal": "metal_plate",
    "dark_metal": "metal_plate",
    "barrier_metal": "metal_plate",
    "wood": "wood_floor",
    "red_wood": "wood_floor",
    "domino_wood": "wood_floor",
    "fabric": "denim_fabric_04",
    "fabric_green": "denim_fabric_04",
    "fabric_coral": "denim_fabric_04",
}

CANDIDATES_BY_SHAPE: dict[str, list[str]] = {
    "sphere": [
        "red_rubber", "blue_rubber", "yellow_rubber", "green_painted",
        "coral_painted", "teal_metal", "yellow_metal",
    ],
    "puck": [
        "red_rubber", "blue_rubber", "yellow_rubber", "green_painted",
        "coral_painted", "teal_metal", "yellow_metal",
    ],
    "cylinder": [
        "red_rubber", "blue_rubber", "yellow_rubber", "green_painted",
        "coral_painted", "teal_metal", "yellow_metal",
    ],
    "box": [
        "red_rubber", "blue_rubber", "yellow_rubber", "blue_painted",
        "teal_metal", "yellow_metal", "white_painted", "green_painted",
        "coral_painted", "fabric_green", "fabric_coral", "red_wood",
        "domino_wood", "wood", "concrete", "dark_metal",
    ],
}


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def material_distance(left: str, right: str) -> float:
    a = np.asarray(MATERIAL_RGB[left], dtype=np.float64)
    b = np.asarray(MATERIAL_RGB[right], dtype=np.float64)
    return float(np.linalg.norm(a - b))


def verify_texture_sources() -> None:
    for package, spec in TEXTURE_SOURCES.items():
        root = Path(spec["root"])
        for field in ("albedo", "normal", "roughness", "ao"):
            path = root / spec[field]
            if not path.is_file():
                raise FileNotFoundError(f"missing {package} {field}: {path}")


def discover_target_cases(strict_root: Path) -> list[dict[str, Any]]:
    samples_root = strict_root / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(samples_root)
    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(samples_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        metadata_path = sample_dir / "metadata.json"
        report_path = sample_dir / "videos" / "rgb_cycles.json"
        trajectory_path = sample_dir / "raw" / "trajectories.npz"
        if not (metadata_path.is_file() and report_path.is_file() and trajectory_path.is_file()):
            raise RuntimeError(f"incomplete strict sample: {sample_dir}")
        metadata = load_json(metadata_path)
        report = load_json(report_path)
        custom_roots = report.get("texture_sources", {}).get("custom_material_roots") or {}
        if custom_roots:
            continue
        dynamic = {
            name: actor
            for name, actor in metadata.get("actors", {}).items()
            if actor.get("role") == "dynamic"
        }
        if not dynamic:
            raise RuntimeError(f"target sample has no role=dynamic actor: {sample_dir.name}")
        rows.append({
            "case_id": sample_dir.name,
            "sample_dir": sample_dir,
            "metadata": metadata,
            "report": report,
            "report_path": report_path,
            "trajectory_path": trajectory_path,
            "dynamic_actors": dynamic,
        })
    if len(rows) != EXPECTED_TARGET_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_TARGET_COUNT} no-custom-texture cases, found {len(rows)}; "
            "refusing to silently change the confirmed target set"
        )
    return rows


def choose_material(case_id: str, report: dict[str, Any], dynamic: dict[str, Any]) -> dict[str, Any]:
    first_actor = next(iter(dynamic.values()))
    shape = str(first_actor.get("shape", "box"))
    candidates = CANDIDATES_BY_SHAPE.get(shape, CANDIDATES_BY_SHAPE["box"])
    current_assignments = report.get("material_assignments", {})
    room = report.get("room_scene", {})
    compare_keys = {
        str(value)
        for value in list(current_assignments.values()) + [
            room.get("floor"), room.get("wall"), room.get("trim")
        ]
        if value in MATERIAL_RGB
    }
    eligible = [candidate for candidate in candidates if candidate not in compare_keys]
    if not eligible:
        raise RuntimeError(f"no unused candidate material for {case_id}: {sorted(compare_keys)}")
    scored = sorted(
        (
            min(material_distance(candidate, other) for other in compare_keys),
            candidate,
        )
        for candidate in eligible
    )
    safe = [item for item in scored if item[0] >= MIN_COLOR_DISTANCE]
    if not safe:
        raise RuntimeError(f"no candidate passes color threshold for {case_id}: {scored}")
    top = safe[-min(4, len(safe)):]
    seed_bytes = hashlib.sha256(f"{EXPERIMENT_SEED}|{case_id}".encode()).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    selected_distance, selected = random.Random(seed).choice(top)
    package = MATERIAL_SOURCE[selected]
    source = TEXTURE_SOURCES[package]
    return {
        "selected_material": selected,
        "selected_material_source": {
            "package": package,
            "root": source["root"],
            "page": source["page"],
            "maps": {key: source[key] for key in ("albedo", "normal", "roughness", "ao")},
            "license": "CC0 (Poly Haven asset package)",
        },
        "override": {name: selected for name in dynamic},
        "selection_seed": seed,
        "selection_pool": [candidate for _, candidate in top],
        "selection_scores": {candidate: score for score, candidate in scored},
        "min_base_color_distance": float(selected_distance),
        "color_distance_threshold": MIN_COLOR_DISTANCE,
        "comparison_materials": sorted(compare_keys),
        "shape": shape,
        "dynamic_actor_names": list(dynamic),
    }


def truth_inheritance(strict_root: Path, case_id: str, sample_dir: Path) -> dict[str, Any]:
    truth_dir = strict_root / "truth" / "cases" / case_id
    files: list[dict[str, Any]] = []
    if truth_dir.is_dir():
        for path in sorted(truth_dir.rglob("*")):
            if path.is_file():
                files.append({
                    "relative_path": str(path.relative_to(truth_dir)),
                    "absolute_path": str(path),
                    "size_bytes": path.stat().st_size,
                })
    inherited_sample_files = []
    for relative in (
        "contacts.json", "physics_supervision.json", "physics_supervision.npz",
        "raw/masks.npz", "raw/physics_supervision.npz", "raw/trajectories.npz",
        "videos/rgb_cycles.json",
    ):
        path = sample_dir / relative
        if path.is_file():
            inherited_sample_files.append({
                "relative_path": relative,
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
            })
    return {
        "schema_version": "physv_cycles_refine_truth_inheritance_v1",
        "experiment_id": EXPERIMENT_NAME,
        "case_id": case_id,
        "parent_sample_dir": str(sample_dir),
        "parent_truth_dir": str(truth_dir),
        "inherited_truth_files": files,
        "inherited_sample_supervision_files": inherited_sample_files,
        "generated_variant_uses_future_truth": False,
        "change_scope": "RGB dynamic-object material only",
        "unchanged": [
            "geometry", "physics", "camera", "trajectory", "strict GT",
            "resolution", "FPS", "frame count",
        ],
    }


def write_experiment_readme() -> None:
    readme = f"""# R002：全部无自定义纹理 Case 的动态物体材质变体

## 目的

本实验针对 strict CYCLES 数据集中审计得到的 70 个 texture_sources.custom_material_roots 为空的 case，给 metadata.actors 中 role=dynamic 的主运动物体替换一个确定性随机选择的、本地已有的 image-backed PBR 材质，并重新渲染 RGB 视频。

这里的“无纹理”特指没有 custom material root；父渲染器本身已经为许多物体使用 Poly Haven PBR 贴图。因此 R002 是可追溯的材质/纹理变体，不声称父视频完全没有任何贴图。

## 严格变更边界

- 只改变主动态物体的 RGB 材质；同一 case 中多个 role=dynamic 物体使用同一选中的材质，保持族内一致。
- 不改变 geometry、PyBullet 物理参数、相机、逐帧 trajectory、灯光/房间布局、分辨率、FPS、帧数或任何 strict GT。
- 变体渲染直接读取父 sample 的 raw/trajectories.npz，不会重新运行物理模拟。
- GT 不复制到实验目录；每个 case 的 truth_inheritance.json 指向 strict 父数据。

## 选择与颜色 QA

- 选择种子：{EXPERIMENT_SEED} + case id 的 SHA256 派生整数；重复执行会得到相同材质。
- 在该 case 的现有 actor 材质以及房间 floor/wall/trim 的近似颜色上做保守距离检查。
- 只从分数最高的最多 4 个候选中随机选择，要求最小近似颜色距离 >= {MIN_COLOR_DISTANCE}；选择分数和比较对象写入 case_selection.jsonl 与每个 case 的 selection.json。
- 材质来自已有本地 Poly Haven image-backed PBR 包，具体 map 文件和来源页记录在清单中；不复制大纹理资产。

## 目录

实验目录包含 README.md、experiment.json、case_selection.jsonl、shards/gpu*.txt、cases/<case_id>/selection.json、material_overrides.json、truth_inheritance.json、render/full/、logs/、evaluation/ 和 index.html。

## 当前渲染协议

896×512 / 30 FPS / 90 frames / CYCLES / 32 samples / CUDA。GPU4 禁止使用。smoke 可用 --mode smoke 只渲染 3 帧；正式输出使用 --mode full。

## 可复现命令

python3 {Path(__file__).resolve()} prepare --gpus 5
python3 {Path(__file__).resolve()} render --gpu 5 --case-list {EXPERIMENT_ROOT}/shards/gpu5.txt --mode full
python3 {Path(__file__).resolve()} index

GPU5 只是本次启动时选择的低占用非禁止卡；若调整分片，必须在 experiment.json 和日志中记录。
"""
    (EXPERIMENT_ROOT / "README.md").write_text(readme, encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    strict_root = args.strict_root.resolve()
    experiment_root = args.experiment_root.resolve()
    verify_texture_sources()
    rows = discover_target_cases(strict_root)
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU")
    if set(gpus) & FORBIDDEN_GPUS:
        raise ValueError("GPU4 is forbidden; choose a different --gpus value")
    if any(not gpu.isdigit() for gpu in gpus):
        raise ValueError(f"GPU values must be integer indices: {gpus}")
    for subdir in ("cases", "shards", "logs", "evaluation", "truth_inheritance", "assets"):
        (experiment_root / subdir).mkdir(parents=True, exist_ok=True)

    selections: list[dict[str, Any]] = []
    for row in rows:
        case_id = row["case_id"]
        selected = choose_material(case_id, row["report"], row["dynamic_actors"])
        trajectory = np.load(row["trajectory_path"], allow_pickle=False)
        frame_count = int(trajectory["frame_times_s"].shape[0])
        fps = int(row["metadata"]["simulation"]["fps"])
        case_root = experiment_root / "cases" / case_id
        case_root.mkdir(parents=True, exist_ok=True)
        inheritance = truth_inheritance(strict_root, case_id, row["sample_dir"])
        case_selection = {
            "schema_version": "physv_cycles_refine_case_selection_v1",
            "experiment_id": EXPERIMENT_NAME,
            "case_id": case_id,
            "parent_sample_dir": str(row["sample_dir"]),
            "parent_render_report": str(row["report_path"]),
            "parent_custom_material_roots": row["report"].get("texture_sources", {}).get("custom_material_roots") or {},
            "family_key": row["metadata"]["family_key"],
            "dynamic_actors": {
                name: {"role": actor.get("role"), "shape": actor.get("shape")}
                for name, actor in row["dynamic_actors"].items()
            },
            "selected": selected,
            "protocol": {
                "width": 896, "height": 512, "fps": fps,
                "frame_count": frame_count, "samples": args.samples,
            },
            "truth_inheritance": str(case_root / "truth_inheritance.json"),
            "status": "planned",
        }
        json_dump(case_root / "selection.json", case_selection)
        json_dump(case_root / "material_overrides.json", selected["override"])
        json_dump(case_root / "truth_inheritance.json", inheritance)
        selections.append(case_selection)

    manifest_path = experiment_root / "case_selection.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in selections),
        encoding="utf-8",
    )
    shard_map: dict[str, list[str]] = {gpu: [] for gpu in gpus}
    for index, item in enumerate(selections):
        shard_map[gpus[index % len(gpus)]].append(item["case_id"])
    for gpu, case_ids in shard_map.items():
        (experiment_root / "shards" / f"gpu{gpu}.txt").write_text(
            "\n".join(case_ids) + "\n", encoding="utf-8"
        )
    inheritance_manifest = {
        "schema_version": "physv_cycles_refine_inheritance_manifest_v1",
        "experiment_id": EXPERIMENT_NAME,
        "parent_root": str(strict_root),
        "case_count": len(selections),
        "cases": {item["case_id"]: item["truth_inheritance"] for item in selections},
    }
    json_dump(experiment_root / "truth_inheritance" / "inheritance.json", inheritance_manifest)
    json_dump(
        experiment_root / "experiment.json",
        {
            "schema_version": "physv_cycles_refine_experiment_v2",
            "experiment_id": EXPERIMENT_NAME,
            "status": "planned",
            "created_at": "2026-08-29",
            "parent_dataset": str(strict_root),
            "target_selection": {
                "criterion": "videos/rgb_cycles.json texture_sources.custom_material_roots is empty",
                "case_count": len(selections),
                "excluded_custom_texture_cases": [
                    "v2v_ramp_platform_l040", "v2v_ramp_platform_l080",
                    "v2v_ramp_platform_l120", "v2v_ramp_platform_l160",
                ],
            },
            "change_scope": "RGB material only for metadata actors with role=dynamic",
            "selection": {
                "seed": EXPERIMENT_SEED,
                "min_color_distance": MIN_COLOR_DISTANCE,
                "random_top_k": 4,
                "manifest": str(manifest_path),
            },
            "protocol": {
                "width": 896, "height": 512, "fps": 30, "frame_count": 90,
                "engine": "CYCLES", "samples": args.samples, "device": "CUDA",
            },
            "gpus_prepared": gpus,
            "truth_inheritance_manifest": str(experiment_root / "truth_inheritance" / "inheritance.json"),
            "large_assets_copied": False,
            "runs": {
                "smoke": {"status": "not_started"},
                "full": {"status": "not_started"},
            },
        },
    )
    write_experiment_readme()
    print(json.dumps({
        "experiment_root": str(experiment_root),
        "case_count": len(selections),
        "gpus": gpus,
    }, ensure_ascii=False))


def video_info(path: Path, ffmpeg: Path) -> dict[str, Any]:
    ffprobe = ffmpeg.with_name("ffprobe")
    result = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)["streams"][0]


def export_trajectory_json(source: Path, target: Path) -> int:
    arrays = np.load(source, allow_pickle=False)
    object_names = [str(value) for value in arrays["object_names"]]
    payload: dict[str, Any] = {
        "object_names": object_names,
        "frame_times_s": arrays["frame_times_s"].tolist(),
    }
    for name in object_names:
        payload[f"{name}_positions"] = arrays[f"{name}_positions"].tolist()
        payload[f"{name}_rotations"] = arrays[f"{name}_rotations"].tolist()
    json_dump(target, payload)
    return len(payload["frame_times_s"])


def run_checked(command: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def encode_video(frame_dir: Path, output: Path, fps: int, ffmpeg: Path) -> None:
    if not sorted(frame_dir.glob("frame_*.png")):
        raise RuntimeError(f"no rendered frames in {frame_dir}")
    temporary = output.with_suffix(".tmp.mp4")
    run_checked([
        str(ffmpeg), "-y", "-loglevel", "warning", "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"), "-c:v", "libx264",
        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ])
    temporary.replace(output)


def encode_prefix(input_video: Path, output: Path, frame_count: int, ffmpeg: Path) -> None:
    temporary = output.with_suffix(".tmp.mp4")
    run_checked([
        str(ffmpeg), "-y", "-loglevel", "warning", "-i", str(input_video),
        "-frames:v", str(frame_count), "-c:v", "libx264", "-preset", "slow",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(temporary),
    ])
    temporary.replace(output)


def render_one(case_id: str, args: argparse.Namespace) -> dict[str, Any]:
    if str(args.gpu) in FORBIDDEN_GPUS:
        raise ValueError("GPU4 is forbidden by workspace policy")
    case_root = args.experiment_root / "cases" / case_id
    selection = load_json(case_root / "selection.json")
    sample_dir = Path(selection["parent_sample_dir"])
    metadata = load_json(sample_dir / "metadata.json")
    fps = int(metadata["simulation"]["fps"])
    expected_frames = int(selection["protocol"]["frame_count"])
    if args.mode == "smoke":
        expected_frames = min(3, expected_frames)

    render_root = case_root / "render" / args.mode
    work_root = case_root / "work" / args.mode
    frames_dir = work_root / "frames"
    output_video = render_root / "rgb_cycles.mp4"
    report_path = render_root / "render_metadata.json"
    if output_video.is_file() and report_path.is_file() and not args.force:
        try:
            probe = video_info(output_video, args.ffmpeg)
            if int(probe.get("nb_frames", "0")) == expected_frames:
                return {
                    "case_id": case_id, "status": "skipped",
                    "video": str(output_video), "frames": expected_frames,
                }
        except Exception:
            pass

    render_root.mkdir(parents=True, exist_ok=True)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    trajectory_json = work_root / "trajectories.json"
    source_frames = export_trajectory_json(sample_dir / "raw" / "trajectories.npz", trajectory_json)
    if source_frames < expected_frames:
        raise RuntimeError(f"{case_id}: source trajectory has {source_frames} frames, expected {expected_frames}")
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    if not args.ffmpeg.is_file() or not args.ffmpeg.with_name("ffprobe").is_file():
        raise FileNotFoundError(f"ffmpeg/ffprobe pair not found beside {args.ffmpeg}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run_checked([
        str(args.blender), "-b", "--python", str(BLENDER_SCRIPT), "--",
        "--sample-dir", str(sample_dir), "--trajectory-json", str(trajectory_json),
        "--output-dir", str(frames_dir), "--width", "896", "--height", "512",
        "--samples", str(args.samples), "--exposure", "0", "--frame-limit",
        "3" if args.mode == "smoke" else "0", "--engine", "CYCLES",
        "--device", "CUDA", "--output-format", "PNG", "--material-overrides-json",
        str(case_root / "material_overrides.json"),
    ], env=env)

    rendered_frames = sorted(frames_dir.glob("frame_*.png"))
    blender_report = frames_dir / "render_metadata.json"
    if not blender_report.is_file() or len(rendered_frames) != expected_frames:
        raise RuntimeError(
            f"{case_id}: incomplete render metadata={blender_report.is_file()} "
            f"frames={len(rendered_frames)}/{expected_frames}"
        )
    encode_video(frames_dir, output_video, fps, args.ffmpeg)
    context_videos: dict[str, Any] = {}
    if args.mode == "full":
        for context_frames in (8, 16):
            context_output = render_root / f"context{context_frames}_cycles.mp4"
            encode_prefix(output_video, context_output, context_frames, args.ffmpeg)
            context_videos[f"context{context_frames}"] = {
                "path": str(context_output), "frames": context_frames,
                "video": video_info(context_output, args.ffmpeg),
            }

    blender_metadata = load_json(blender_report)
    output_metadata = dict(blender_metadata)
    output_metadata.update({
        "refine_schema_version": "physv_cycles_refine_texture_batch_v1",
        "experiment_id": EXPERIMENT_NAME,
        "parent_case": case_id,
        "change_scope": "RGB dynamic-object material only; geometry, physics, camera, trajectory and strict GT unchanged",
        "selection": selection["selected"],
        "strict_protocol": {
            "width": 896, "height": 512, "fps": fps, "frame_count": expected_frames,
        },
        "parent_render_report": selection["parent_render_report"],
        "truth_inheritance": selection["truth_inheritance"],
        "output_video": str(output_video),
        "video": video_info(output_video, args.ffmpeg),
        "context_videos": context_videos,
    })
    json_dump(report_path, output_metadata)
    if not args.keep_frames:
        shutil.rmtree(work_root)
    return {
        "case_id": case_id, "status": "completed", "mode": args.mode,
        "video": str(output_video), "frames": expected_frames,
        "material": selection["selected"]["selected_material"],
        "probe": output_metadata["video"],
    }


def read_case_list(path: Path) -> list[str]:
    case_ids = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not case_ids:
        raise ValueError(f"empty case list: {path}")
    return case_ids


def render(args: argparse.Namespace) -> None:
    if not args.case_list.is_file():
        raise FileNotFoundError(args.case_list)
    case_ids = read_case_list(args.case_list)
    results_path = args.experiment_root / "logs" / f"gpu{args.gpu}_{args.mode}.results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with results_path.open("a", encoding="utf-8") as results_file:
        for index, case_id in enumerate(case_ids, start=1):
            print(f"[{index}/{len(case_ids)}] {case_id}", flush=True)
            try:
                result = render_one(case_id, args)
            except Exception as exc:
                failures += 1
                result = {
                    "case_id": case_id, "status": "failed", "mode": args.mode,
                    "error": repr(exc), "traceback": traceback.format_exc(),
                }
                error_path = args.experiment_root / "cases" / case_id / "render" / args.mode / "error.json"
                json_dump(error_path, result)
                print(result["traceback"], file=sys.stderr, flush=True)
            results_file.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            results_file.flush()
            # Keep the static browser page useful while a long shard is still
            # running: a normal browser refresh should expose newly finished
            # cases without requiring a second manual command.
            try:
                build_index(argparse.Namespace(experiment_root=args.experiment_root))
            except Exception as index_error:
                print(f"index refresh failed after {case_id}: {index_error!r}", file=sys.stderr, flush=True)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    print(json.dumps({
        "mode": args.mode, "case_count": len(case_ids), "failures": failures,
    }, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(1)


def build_index(args: argparse.Namespace) -> None:
    manifest_path = args.experiment_root / "case_selection.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    selections = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[str] = []
    completed = 0
    failed = 0
    for selection in selections:
        case_id = selection["case_id"]
        case_root = args.experiment_root / "cases" / case_id
        report = case_root / "render" / "full" / "render_metadata.json"
        video = case_root / "render" / "full" / "rgb_cycles.mp4"
        status = "completed" if report.is_file() and video.is_file() else "pending"
        error_path = case_root / "render" / "full" / "error.json"
        if error_path.is_file():
            failed += 1
            status = "failed"
        if status == "completed":
            completed += 1
        material = selection["selected"]["selected_material"]
        parent_report = load_json(Path(selection["parent_render_report"]))
        parent_assignments = parent_report.get("material_assignments", {})
        dynamic_names = selection["selected"]["dynamic_actor_names"]
        parent_material = "<br>".join(
            f"{html.escape(name)}: {html.escape(str(parent_assignments.get(name, '—')))}"
            for name in dynamic_names
        )
        parent_video = Path(selection["parent_sample_dir"]) / "videos" / "rgb_cycles.mp4"
        variant_rel = Path("cases") / case_id / "render" / "full" / "rgb_cycles.mp4"
        parent_rel = os.path.relpath(parent_video, args.experiment_root)
        variant_cell = (
            f'<video controls preload="metadata" src="{html.escape(str(variant_rel))}"></video>'
            if status == "completed" else '<span class="pending">pending</span>'
        )
        rows.append(
            '<tbody class="case-group">'
            '<tr class="group-start parent-row">'
            f'<td rowspan="2" class="case-group-cell"><code>{html.escape(case_id)}</code>'
            f'<small>{html.escape(selection["family_key"])}</small></td>'
            '<td><span class="variant-label parent-label">原 strict</span></td>'
            f'<td><b>{html.escape(parent_material)}</b><br><small>{html.escape(", ".join(dynamic_names))}</small></td>'
            '<td><span class="state parent-state">unchanged</span></td>'
            f'<td><video controls preload="metadata" src="{html.escape(parent_rel)}"></video></td>'
            '</tr>'
            '<tr class="refine-row">'
            '<td><span class="variant-label refine-label">R002 refine</span></td>'
            f'<td><b>{html.escape(material)}</b><br><small>{html.escape(", ".join(dynamic_names))}</small></td>'
            f'<td><span class="state {"done-state" if status == "completed" else "pending-state"}">{html.escape(status)}</span></td>'
            f'<td>{variant_cell}</td>'
            '</tr></tbody>'
        )
    pending = len(selections) - completed - failed
    state = "completed" if completed == len(selections) else ("partial" if completed else "planned")
    experiment_path = args.experiment_root / "experiment.json"
    experiment = load_json(experiment_path) if experiment_path.is_file() else {}
    experiment["status"] = state
    experiment["progress"] = {
        "completed": completed, "failed": failed, "pending": pending, "total": len(selections),
    }
    experiment.setdefault("runs", {})["full"] = {
        "status": state, "completed": completed, "failed": failed, "total": len(selections),
    }
    json_dump(experiment_path, experiment)

    body = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>R002 CYCLES distinct texture variants</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:24px;background:#f5f7f8;color:#172027}
h1{margin-bottom:6px}.meta{color:#52616b;margin-bottom:18px}.summary{padding:12px 16px;background:#e7f1f4;border-left:4px solid #197278;margin-bottom:18px}
table{border-collapse:separate;border-spacing:0;width:100%;background:white;box-shadow:0 1px 6px #0001}th,td{border-bottom:1px solid #dce3e6;padding:9px;vertical-align:top;text-align:left}th{position:sticky;top:0;background:#234e52;color:white}.case-group tr.group-start td{border-top:3px solid #c5d5d8}.case-group-cell{width:220px;background:#edf5f6}.case-group-cell code{display:block}.case-group-cell small{display:block;margin-top:6px;color:#52616b}.refine-row{background:#fffaf0}.variant-label{display:inline-block;padding:4px 7px;border-radius:3px;font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.parent-label{background:#e5eef0;color:#28545b}.refine-label{background:#f7e4b6;color:#76501a}.state{font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace}.parent-state{color:#42656a}.done-state{color:#267044}.pending-state{color:#9b5b00}.pending{color:#9b5b00;font-weight:600}code{font-size:12px}
</style></head><body>
<h1>R002 · strict CYCLES 动态物体纹理变体</h1>
<div class="meta">每个 case 固定显示两行：同组的原 strict case 与 R002 refine case。仅替换 role=dynamic 的 RGB 材质；几何、轨迹、相机、物理和 strict GT 不变。</div>
<div class="summary">状态：<b>__STATE__</b>　已完成 <b>__COMPLETED__ / __TOTAL__</b>　失败 <b>__FAILED__</b>　待处理 <b>__PENDING__</b><br>协议：896×512 / 30 FPS / 90 frames / CYCLES / 32 samples。刷新本页即可查看已完成 case。</div>
<table><thead><tr><th>Case 分组</th><th>版本</th><th>动态物体材质</th><th>状态</th><th>视频</th></tr></thead>__ROWS__</table>
</body></html>
"""
    body = (
        body.replace("__STATE__", html.escape(state))
        .replace("__COMPLETED__", str(completed))
        .replace("__TOTAL__", str(len(selections)))
        .replace("__FAILED__", str(failed))
        .replace("__PENDING__", str(pending))
        .replace("__ROWS__", "\n".join(rows))
    )
    (args.experiment_root / "index.html").write_text(body, encoding="utf-8")
    print(json.dumps({
        "index": str(args.experiment_root / "index.html"), "status": state,
        "completed": completed, "failed": failed, "pending": pending,
    }, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--strict-root", type=Path, default=STRICT_ROOT)
    prepare_parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    prepare_parser.add_argument("--gpus", default="5")
    prepare_parser.add_argument("--samples", type=int, default=32)
    prepare_parser.set_defaults(func=prepare)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    render_parser.add_argument("--case-list", type=Path, required=True)
    render_parser.add_argument("--gpu", required=True)
    render_parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    render_parser.add_argument("--samples", type=int, default=32)
    render_parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    render_parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    render_parser.add_argument("--force", action="store_true")
    render_parser.add_argument("--keep-frames", action="store_true")
    render_parser.set_defaults(func=render)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    index_parser.set_defaults(func=build_index)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    if hasattr(arguments, "strict_root"):
        arguments.strict_root = arguments.strict_root.resolve()
    arguments.experiment_root = arguments.experiment_root.resolve()
    if hasattr(arguments, "case_list"):
        arguments.case_list = arguments.case_list.resolve()
    if hasattr(arguments, "blender"):
        arguments.blender = arguments.blender.resolve()
    if hasattr(arguments, "ffmpeg"):
        arguments.ffmpeg = arguments.ffmpeg.resolve()
    arguments.func(arguments)
