#!/usr/bin/env python3
"""Generate text captions for generated Genesis/PhysXNet videos.

该脚本用于为 Genesis/PhysXNet 视频样本生成文本描述；输入为一个或多个样本根目录、metadata.json 和 PhysXNet JSON 资产信息，输出为各 sample_dir 下的 caption.txt、caption_simple.txt、caption.json 及可选 manifest。

For every sample directory containing metadata.json and an RGB video, this script
writes a caption txt file into the sample directory by default:

    sample_x/caption.txt

It resolves PhysXNet object names/categories from finaljson/<source_object_id>.json
and parses custom objects such as yellow_striker_ball from metadata.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_PHYSX_ROOT = Path("/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet/version_1/finaljson")
META_FILENAMES = ("meta.json", "metadata.json")

MOTION_DESCRIPTIONS = {
    "random_parabola": "moves along a randomized ballistic arc with initial linear and angular velocity",
    "high_drop": "starts from an elevated position and falls under gravity until it contacts the ground",
    "static_highdrop": "is released from a higher initial position and falls under gravity",
    "static_center": "starts near the center with no prescribed object velocity",
    "static_left": "starts from a left-offset static placement",
    "static_right": "starts from a right-offset static placement",
    "entry_left": "enters the scene from the left side with initial velocity",
    "entry_right": "enters the scene from the right side with initial velocity",
    "entry_fast_center": "enters near the center with a faster initial velocity",
    "legacy_random": "uses a legacy randomized initial placement and motion setup",
    "striker_hit": "acts as a moving striker object that impacts or interacts with the target",
    "static_rest": "remains initially at rest",
}

GROUP_DESCRIPTIONS = {
    "projectile_motion": "projectile motion",
    "gravity_drop": "gravity-driven falling motion",
    "entry_motion": "entry motion",
    "static_placement": "static placement",
    "striker": "striker-driven interaction",
    "auxiliary_static": "auxiliary static context",
}

SCENE_DESCRIPTIONS = {
    "single_object_preview": "a single rigid object scene",
    "interaction_pair_plus_dynamic": "a two-object rigid interaction scene",
    "dual_interaction_groups": "a multi-object scene with two interaction groups",
    "omni_showcase": "a mixed multi-object showcase scene",
}
INVALID_SAMPLE_MARKERS = {"invalid_case900_901", "invalid_by_qa", "_qa_invalid"}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_visual_sample_dir(sample_dir: Path) -> bool:
    return (
        (sample_dir / "videos" / "rgb.mp4").exists()
        or (sample_dir / "rgb.mp4").exists()
        or any((sample_dir / "rgb").glob("frame_*.png"))
    )


def clean_name(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_choice(options: List[str], key: str) -> str:
    if not options:
        return ""
    # Deterministic pseudo-random choice so captions are reproducible.
    idx = sum((i + 1) * ord(ch) for i, ch in enumerate(str(key))) % len(options)
    return options[idx]


def object_label(info: Dict[str, Any]) -> str:
    return str(info.get("name") or "object").strip() or "object"


def simple_object_label(info: Dict[str, Any]) -> str:
    name = object_label(info)
    # Remove inline category hints like "object name (category)" from simple captions.
    name = re.sub(r"\s*\([^)]*\)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "object"


def normalize_simple_caption(text: str) -> str:
    text = re.sub(r"\s*\([^)]*\)", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def physx_info(source_id: str, physx_json_dir: Path, cache: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    sid = str(source_id).strip()
    if not sid:
        return {"name": "unknown PhysXNet object", "category": "unknown"}
    if sid in cache:
        return cache[sid]
    path = physx_json_dir / f"{sid}.json"
    if not path.exists():
        out = {"name": f"PhysXNet object {sid}", "category": "unknown"}
    else:
        try:
            data = load_json(path)
            out = {
                "name": str(data.get("object_name") or f"PhysXNet object {sid}"),
                "category": str(data.get("category") or "unknown"),
            }
        except Exception:
            out = {"name": f"PhysXNet object {sid}", "category": "unknown"}
    cache[sid] = out
    return out


def custom_info(source_id: str) -> Dict[str, str]:
    sid = str(source_id or "custom_object").strip()
    name = clean_name(sid) or "custom object"
    category = "custom object"
    if sid == "yellow_striker_ball":
        name = "yellow striker ball"
        category = "striker sphere"
    elif "ball" in sid.lower() or "sphere" in sid.lower():
        category = "sphere"
    return {"name": name, "category": category}


def object_info(obj: Dict[str, Any], physx_json_dir: Path, cache: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    dataset_source = str(obj.get("dataset_source") or "").strip()
    source_id = str(obj.get("source_object_id") or "").strip()
    if dataset_source == "PhysXNet":
        info = physx_info(source_id, physx_json_dir, cache)
    elif dataset_source == "Custom":
        info = custom_info(source_id)
    else:
        info = {"name": clean_name(source_id or obj.get("source_tag", "object")), "category": dataset_source or "unknown"}
    return {
        "object_id": obj.get("object_id"),
        "seg_id": obj.get("seg_id"),
        "dataset_source": dataset_source or "unknown",
        "source_object_id": source_id,
        "name": info["name"],
        "category": info["category"],
        "role": str(obj.get("role") or "unknown"),
        "motion_type": str(obj.get("motion_type") or obj.get("object_motion_type") or "unknown"),
        "motion_group": str(obj.get("motion_group") or obj.get("object_motion_group") or "unknown"),
        "entity_type": str(obj.get("entity_type") or "unknown"),
    }


def motion_sentence(motion_type: str, motion_group: str) -> str:
    mt = str(motion_type or "unknown")
    mg = str(motion_group or "unknown")
    desc = MOTION_DESCRIPTIONS.get(mt)
    if desc:
        return desc
    if mt.endswith("_v2") and mt[:-3] in MOTION_DESCRIPTIONS:
        return MOTION_DESCRIPTIONS[mt[:-3]] + " with a second randomized variant"
    group_desc = GROUP_DESCRIPTIONS.get(mg)
    if group_desc:
        return f"undergoes {group_desc}"
    return f"has motion type {mt}"


def format_object_phrase(info: Dict[str, Any]) -> str:
    name = info["name"]
    category = info["category"]
    role = info["role"]
    motion = motion_sentence(info["motion_type"], info["motion_group"])
    return (
        f"object_id={info['object_id']}, seg_id={info['seg_id']}: "
        f"a {name} ({category}, source={info['dataset_source']}, source_id={info['source_object_id']}) "
        f"with role={role}; it {motion}"
    )


def format_simple_object_phrase(info: Dict[str, Any]) -> str:
    name = simple_object_label(info)
    motion = motion_sentence(info["motion_type"], info["motion_group"])
    return f"the {name} {motion}"


def format_name_list(names: List[str]) -> str:
    cleaned = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned:
        return "the objects"
    with_articles = [f"the {name}" for name in cleaned]
    if len(with_articles) == 1:
        return with_articles[0]
    if len(with_articles) == 2:
        return f"{with_articles[0]} and {with_articles[1]}"
    return ", ".join(with_articles[:-1]) + f", and {with_articles[-1]}"


def placement_phrase(motion_category: str) -> str:
    label = str(motion_category or "").strip().lower()
    if label.startswith("static_center"):
        return "near the center"
    if label.startswith("static_left"):
        return "on the left side"
    if label.startswith("static_right"):
        return "on the right side"
    if label in {"static_highdrop", "high_drop"}:
        return "from a high position"
    return "in the scene"


def placement_clause(target_name: str, motion_category: str, key: str) -> str:
    label = str(motion_category or "").strip().lower()
    place = placement_phrase(label)
    if label.startswith("static_center") or label.startswith("static_left") or label.startswith("static_right"):
        templates = [
            f" with the {target_name} set {place}",
            f"; the {target_name} starts {place}",
            f" after it is positioned {place}",
            "",
        ]
        return stable_choice(templates, key)
    if label in {"static_highdrop", "high_drop"}:
        templates = [
            f" after it is released {place}",
            f" as it drops {place}",
            "",
        ]
        return stable_choice(templates, key)
    return ""


def single_object_simple_caption(obj: Dict[str, Any], motion_category: str) -> str:
    name = simple_object_label(obj)
    mt = str(obj.get("motion_type") or motion_category or "")
    key = f"single::{name}::{mt}"
    if mt in {"random_parabola"}:
        return stable_choice([
            f"the {name} is launched with a random velocity and follows a ballistic arc.",
            f"the {name} follows a randomized parabolic trajectory.",
            f"the {name} moves through the air in a ballistic arc.",
        ], key)
    if mt in {"high_drop", "static_highdrop"}:
        return stable_choice([
            f"the {name} is released from a high position and falls under gravity.",
            f"the {name} drops from above under gravity.",
            f"the {name} falls from an elevated starting point.",
        ], key)
    if mt.startswith("entry_"):
        return stable_choice([
            f"the {name} enters the scene with an initial velocity.",
            f"the {name} moves into the scene from the side.",
            f"the {name} slides into view with an initial velocity.",
        ], key)
    if mt.startswith("static_"):
        place = placement_phrase(mt)
        return stable_choice([
            f"the {name} is placed {place}.",
            f"the {name} starts {place}.",
            f"the {name} rests {place}.",
        ], key)
    return format_simple_object_phrase(obj) + "."


def build_simple_caption(objects: List[Dict[str, Any]], motion_category: str, interaction_pattern: str) -> str:
    if not objects:
        return normalize_simple_caption(f"An object is shown in a {clean_name(motion_category)} scene.")
    if len(objects) == 1:
        return normalize_simple_caption(single_object_simple_caption(objects[0], motion_category))
    names = [simple_object_label(obj) for obj in objects]
    joined_names = format_name_list(names)
    motion_groups = {str(obj.get("motion_group") or "") for obj in objects}
    motion_types = {str(obj.get("motion_type") or "") for obj in objects}
    if interaction_pattern == "multi_object_independent_projectile_motion" or motion_types == {"independent_projectile_motion"}:
        return normalize_simple_caption(f"{joined_names} move along independent projectile trajectories.")
    if interaction_pattern == "multi_object_independent_gravity_drop" or motion_types == {"independent_gravity_drop"}:
        return normalize_simple_caption(f"{joined_names} fall independently under gravity.")
    if motion_groups == {"static_placement"} or motion_groups == {"auxiliary_static"} or motion_groups == {"static_placement", "auxiliary_static"}:
        return normalize_simple_caption(f"{joined_names} remain in the scene without interacting.")
    initiators = [obj for obj in objects if obj.get("role") == "initiator"]
    targets = [obj for obj in objects if obj.get("role") == "target"]
    others = [obj for obj in objects if obj not in initiators and obj not in targets]
    if initiators and targets:
        init = initiators[0]
        target = targets[0]
        init_name = simple_object_label(init)
        target_name = simple_object_label(target)
        key = f"multi::{init_name}::{target_name}::{motion_category}::{interaction_pattern}"
        extra = placement_clause(target_name, motion_category, key)
        if str(motion_category).startswith("entry_"):
            templates = [
                f"the {target_name} enters the scene while the {init_name} interacts with it",
                f"the {target_name} moves into the scene and the {init_name} interacts with it",
                f"the {init_name} interacts with the incoming {target_name}",
            ]
            core = stable_choice(templates, key)
        elif str(init.get("motion_type")) == "striker_hit" or "striker" in str(init.get("motion_group")):
            templates = [
                f"the {init_name} strikes the {target_name}{extra}",
                f"the {init_name} hits the {target_name}{extra}",
                f"the {target_name} is struck by the {init_name}{extra}",
            ]
            core = stable_choice(templates, key + "::strike")
        else:
            templates = [
                f"the {init_name} interacts with the {target_name}{extra}",
                f"the {target_name} interacts with the {init_name}{extra}",
            ]
            core = stable_choice(templates, key + "::interact")
        if others:
            other_names = format_name_list([simple_object_label(obj) for obj in others])
            core += f", with {other_names} also present"
        return normalize_simple_caption(core + ".")
    phrases = [single_object_simple_caption(obj, str(obj.get("motion_type") or motion_category)).rstrip(".") for obj in objects]
    return normalize_simple_caption("; ".join(phrases) + ".")


def extract_video_relpath(sample_dir: Path, metadata: Dict[str, Any]) -> str:
    outputs = metadata.get("outputs", {}) if isinstance(metadata.get("outputs", {}), dict) else {}
    rel = str(outputs.get("rgb_video") or "videos/rgb.mp4")
    return rel


def build_caption(metadata: Dict[str, Any], sample_dir: Path, physx_json_dir: Path, cache: Dict[str, Dict[str, str]]) -> Tuple[str, Dict[str, Any]]:
    objects = [object_info(obj, physx_json_dir, cache) for obj in metadata.get("objects", []) if isinstance(obj, dict)]
    scene_id = str(metadata.get("scene_id") or sample_dir.name)
    scene_composition = str(metadata.get("scene_composition") or "unknown_scene")
    interaction_pattern = str(metadata.get("interaction_pattern") or "unknown_interaction")
    motion_category = str(metadata.get("motion_category") or "unknown_motion")
    num_objects = int(metadata.get("num_objects") or len(objects))
    frames = int(metadata.get("frames") or 0)
    resolution = metadata.get("resolution", [])
    sim = metadata.get("simulation", {}) if isinstance(metadata.get("simulation", {}), dict) else {}
    gravity = sim.get("gravity", metadata.get("gravity", [0.0, 0.0, -9.81]))
    scene_desc = SCENE_DESCRIPTIONS.get(scene_composition, clean_name(scene_composition))

    object_lines = [format_object_phrase(info) for info in objects]
    object_summary = "; ".join(object_lines) if object_lines else "no object metadata is available"
    role_summary = ", ".join(f"{info['role']}={info['name']}" for info in objects) if objects else "unknown roles"
    simple_caption = build_simple_caption(objects, motion_category, interaction_pattern)

    caption = (
        f"Scene {scene_id}: {scene_desc} with {num_objects} rigid object(s). "
        f"The scene composition is {scene_composition}, the interaction pattern is {interaction_pattern}, "
        f"and the global motion category is {motion_category}. "
        f"Objects: {object_summary}. "
        f"Role summary: {role_summary}. "
        f"The RGB video has {frames} frames at resolution {resolution}; gravity is {gravity}."
    )
    structured = {
        "scene_id": scene_id,
        "sample_dir": str(sample_dir),
        "rgb_video": extract_video_relpath(sample_dir, metadata),
        "scene_composition": scene_composition,
        "interaction_pattern": interaction_pattern,
        "motion_category": motion_category,
        "num_objects": num_objects,
        "frames": frames,
        "resolution": resolution,
        "gravity": gravity,
        "objects": objects,
        "caption": caption,
        "simple_caption": simple_caption,
    }
    return caption, structured


def find_metadata_files(roots: Iterable[Path], include_invalid: bool) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if root.is_file() and root.name in META_FILENAMES:
            files.append(root)
            continue
        for meta_name in META_FILENAMES:
            for p in root.rglob(meta_name):
                if not include_invalid and any(part in INVALID_SAMPLE_MARKERS for part in p.parts):
                    continue
                sample_dir = p.parent
                if is_visual_sample_dir(sample_dir):
                    files.append(p)
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate structured captions for Genesis rigid video samples")
    parser.add_argument("--roots", type=Path, nargs="+", required=True, help="Dataset roots or sample metadata paths to scan")
    parser.add_argument("--physx_json_dir", type=Path, default=DEFAULT_PHYSX_ROOT)
    parser.add_argument("--caption_name", type=str, default="caption.txt", help="Caption txt filename written in each sample directory")
    parser.add_argument("--simple_caption_name", type=str, default="caption_simple.txt", help="Short caption txt filename written in each sample directory")
    parser.add_argument("--json_name", type=str, default="caption.json", help="Structured caption json filename written in each sample directory")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional jsonl manifest with one record per caption")
    parser.add_argument("--include_invalid", action="store_true", help="Also caption samples under invalid_case900_901, invalid_by_qa, or _qa_invalid")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing caption files")
    parser.add_argument(
        "--write_meta_fields",
        action="store_true",
        help="Also write caption/detail_caption fields back into meta.json or metadata.json.",
    )
    args = parser.parse_args()

    metadata_files = find_metadata_files(args.roots, include_invalid=bool(args.include_invalid))
    cache: Dict[str, Dict[str, str]] = {}
    records: List[Dict[str, Any]] = []
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_f = args.manifest.open("w", encoding="utf-8")
    else:
        manifest_f = None

    try:
        for metadata_path in metadata_files:
            sample_dir = metadata_path.parent
            caption_path = sample_dir / args.caption_name
            simple_caption_path = sample_dir / args.simple_caption_name
            json_path = sample_dir / args.json_name
            if caption_path.exists() and simple_caption_path.exists() and json_path.exists() and not args.overwrite:
                continue
            try:
                metadata = load_json(metadata_path)
                caption, structured = build_caption(metadata, sample_dir, args.physx_json_dir, cache)
            except Exception as exc:
                print(f"SKIP {metadata_path}: {type(exc).__name__}: {exc}")
                continue
            caption_path.write_text(caption + "\n", encoding="utf-8")
            simple_caption_path.write_text(str(structured.get("simple_caption", caption)) + "\n", encoding="utf-8")
            json_path.write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.write_meta_fields:
                metadata["caption"] = str(structured.get("simple_caption", caption))
                metadata["detail_caption"] = caption
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            records.append(structured)
            if manifest_f is not None:
                manifest_f.write(json.dumps(structured, ensure_ascii=False) + "\n")
        print(f"captioned={len(records)} scanned={len(metadata_files)}")
    finally:
        if manifest_f is not None:
            manifest_f.close()


if __name__ == "__main__":
    main()
'''
python3 /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_video_captions.py \
    --roots /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases \
    --include_invalid 
    # --overwrite


python3 /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_video_captions.py \
    --roots /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark \
    --include_invalid \
    --overwrite


'''
