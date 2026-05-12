#!/usr/bin/env python3
# 用途：回填 train/rigid 数据的 meta.json 与 caption 字段。
"""Backfill Genesis train/rigid meta.json and captions.

该脚本用于修复 Genesis 自建数据 train/rigid 下缺失的 meta.json，并为所有可识别样本补全
caption.txt、caption_simple.txt、caption.json 以及 meta.json 里的 caption/detail_caption。
"""
from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from generators.generate_video_captions import DEFAULT_PHYSX_ROOT, build_caption, load_json


META_FILENAMES = ("meta.json", "metadata.json")
DEFAULT_TRAIN_RIGID_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)


def find_meta_path(sample_dir: Path) -> Optional[Path]:
    for filename in META_FILENAMES:
        candidate = sample_dir / filename
        if candidate.exists():
            return candidate
    return None


def iter_sample_dirs(root: Path) -> List[Path]:
    sample_dirs = set()
    for meta_name in META_FILENAMES:
        for path in root.rglob(meta_name):
            sample_dirs.add(path.parent)
    for path in root.rglob("scene_input.json"):
        sample_dirs.add(path.parent)
    return sorted(sample_dirs)


def read_png_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Invalid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def infer_frames_and_resolution(sample_dir: Path) -> Tuple[int, List[int]]:
    frames = sorted((sample_dir / "rgb").glob("frame_*.png"))
    if not frames:
        return 0, []
    for frame_path in frames:
        try:
            width, height = read_png_size(frame_path)
            return len(frames), [width, height]
        except Exception:
            continue
    return len(frames), []


def load_scene_input(sample_dir: Path) -> Dict[str, Any]:
    scene_input_path = sample_dir / "scene_input.json"
    if not scene_input_path.exists():
        return {}
    return load_json(scene_input_path)


def meta_candidates_for_pattern(count_dir: Path, pattern: str) -> Iterable[Path]:
    for meta_name in META_FILENAMES:
        yield from sorted(count_dir.glob(f"{pattern}/{meta_name}"))


def find_same_object_donor(sample_dir: Path, object_id: str) -> Optional[Path]:
    count_dir = sample_dir.parent
    for candidate in meta_candidates_for_pattern(count_dir, f"{object_id}__*"):
        if candidate.parent != sample_dir:
            return candidate
    return None


def find_same_case_template(sample_dir: Path, case_name: str) -> Optional[Path]:
    count_dir = sample_dir.parent
    for candidate in meta_candidates_for_pattern(count_dir, f"*__{case_name}"):
        if candidate.parent != sample_dir:
            return candidate
    return None


def combine_objects(template_meta: Dict[str, Any], donor_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    template_objects = list(template_meta.get("objects", []) or [])
    donor_objects = list(donor_meta.get("objects", []) or [])
    combined: List[Dict[str, Any]] = []
    total = max(len(template_objects), len(donor_objects))
    for idx in range(total):
        if idx < len(template_objects):
            item = copy.deepcopy(template_objects[idx])
        elif idx < len(donor_objects):
            item = copy.deepcopy(donor_objects[idx])
        else:
            item = {}
        if idx < len(donor_objects):
            donor = donor_objects[idx]
            for key in (
                "object_id",
                "seg_id",
                "entity_type",
                "source_tag",
                "dataset_source",
                "source_object_id",
            ):
                if key in donor:
                    item[key] = donor[key]
        combined.append(item)
    return combined


def build_missing_meta(sample_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    scene_input = load_scene_input(sample_dir)
    if not scene_input:
        return None, "missing_scene_input"
    sample_name = str(scene_input.get("sample_name") or sample_dir.name)
    object_id = str(scene_input.get("object_id") or sample_dir.name.split("__", 1)[0])
    case_name = str(scene_input.get("case_name") or sample_name.split("__", 1)[1])

    donor_path = find_same_object_donor(sample_dir, object_id)
    if donor_path is None:
        return None, "missing_same_object_donor"
    template_path = find_same_case_template(sample_dir, case_name)
    if template_path is None:
        return None, "missing_same_case_template"

    donor_meta = load_json(donor_path)
    template_meta = load_json(template_path)
    meta = copy.deepcopy(template_meta)

    meta["scene_id"] = sample_name
    meta["object_id"] = object_id
    meta["case_id"] = scene_input.get("case_id")
    meta["case_variant_index"] = scene_input.get("case_variant_index")
    meta["case_name"] = scene_input.get("case_name")
    meta["split"] = "train"
    meta["scene_composition"] = str(scene_input.get("scene_composition") or sample_dir.parent.parent.name)
    meta["interaction_pattern"] = str(scene_input.get("interaction_pattern") or meta.get("interaction_pattern") or "unknown")
    meta["object_count_bucket"] = str(scene_input.get("object_count_bucket") or sample_dir.parent.name)
    meta["motion_category"] = str(scene_input.get("scene_label") or meta.get("motion_category") or "unknown")
    meta["simulator_type"] = str(scene_input.get("simulator_type") or meta.get("simulator_type") or "rigid")
    meta["camera"] = copy.deepcopy(scene_input.get("camera") or meta.get("camera"))
    meta["camera_tag"] = scene_input.get("camera_tag")
    meta["counterfactual"] = copy.deepcopy(scene_input.get("counterfactual"))
    meta["sample_role"] = "counterfactual_negative" if scene_input.get("counterfactual") else "factual"

    frames, resolution = infer_frames_and_resolution(sample_dir)
    if frames > 0:
        meta["frames"] = int(frames)
    if resolution:
        meta["resolution"] = resolution

    simulation = copy.deepcopy(meta.get("simulation", {})) if isinstance(meta.get("simulation"), dict) else {}
    if scene_input.get("gravity") is not None:
        simulation["gravity"] = scene_input.get("gravity")
    meta["simulation"] = simulation

    meta["objects"] = combine_objects(template_meta, donor_meta)
    meta["num_objects"] = len(meta["objects"])
    meta["environment_entities"] = copy.deepcopy(
        donor_meta.get("environment_entities", template_meta.get("environment_entities", []))
    )

    outputs = copy.deepcopy(meta.get("outputs", {})) if isinstance(meta.get("outputs"), dict) else {}
    outputs["metadata"] = "meta.json"
    meta["outputs"] = outputs
    meta["has_depth_metric"] = (sample_dir / "physics" / "depth_metric.npy").exists()
    meta["has_seg"] = (sample_dir / "physics" / "seg.npy").exists()
    meta["has_contact_graph"] = (sample_dir / "physics" / "contact_graph.npy").exists()
    meta["status"] = str(meta.get("status") or "ok")
    return meta, "ok"


def write_meta_and_captions(
    sample_dir: Path,
    metadata: Dict[str, Any],
    *,
    physx_json_dir: Path,
    cache: Dict[str, Dict[str, str]],
) -> None:
    caption, structured = build_caption(metadata, sample_dir, physx_json_dir, cache)
    simple_caption = str(structured.get("simple_caption", caption))
    metadata["caption"] = simple_caption
    metadata["detail_caption"] = caption
    metadata.setdefault("outputs", {})
    metadata["outputs"]["metadata"] = "meta.json"

    (sample_dir / "meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (sample_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
    (sample_dir / "caption_simple.txt").write_text(simple_caption + "\n", encoding="utf-8")
    (sample_dir / "caption.json").write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Genesis train/rigid meta.json and captions")
    parser.add_argument("--root", type=Path, default=DEFAULT_TRAIN_RIGID_ROOT)
    parser.add_argument("--physx_json_dir", type=Path, default=DEFAULT_PHYSX_ROOT)
    args = parser.parse_args()

    root = args.root.resolve()
    cache: Dict[str, Dict[str, str]] = {}
    sample_dirs = iter_sample_dirs(root)
    created_meta = 0
    updated_captions = 0
    skipped_missing_meta: List[Tuple[str, str]] = []

    for sample_dir in sample_dirs:
        meta_path = find_meta_path(sample_dir)
        metadata: Optional[Dict[str, Any]] = None
        if meta_path is None:
            metadata, reason = build_missing_meta(sample_dir)
            if metadata is None:
                skipped_missing_meta.append((str(sample_dir), reason))
                continue
            created_meta += 1
        else:
            metadata = load_json(meta_path)
        write_meta_and_captions(
            sample_dir,
            metadata,
            physx_json_dir=args.physx_json_dir,
            cache=cache,
        )
        updated_captions += 1

    print(f"sample_dirs={len(sample_dirs)}")
    print(f"created_meta={created_meta}")
    print(f"updated_captions={updated_captions}")
    print(f"skipped_missing_meta={len(skipped_missing_meta)}")
    for sample_dir, reason in skipped_missing_meta[:20]:
        print(f"SKIP {reason} {sample_dir}")


if __name__ == "__main__":
    main()
