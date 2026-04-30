#!/usr/bin/env python3
"""Normalize metadata fields for samples referenced by data_summary/by_raw_window."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/by_raw_window")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def find_meta_path(sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    return None


def iter_summary_lists(summary_root: Path) -> Iterable[Path]:
    for path in sorted(summary_root.rglob("*.json")):
        if path.name in {"summary.json", "all_sample_dirs.json"}:
            continue
        yield path
    all_list = summary_root / "all_sample_dirs.json"
    if all_list.exists():
        yield all_list


def collect_sample_dirs(summary_root: Path) -> list[Path]:
    sample_dirs: set[Path] = set()
    for list_path in iter_summary_lists(summary_root):
        data = json.loads(list_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for item in data:
            sample_dirs.add(Path(str(item)))
    return sorted(sample_dirs)


def infer_dataset(sample_dir: Path, meta: dict[str, Any]) -> str:
    for key in ("dataset", "dataset_name", "dataset_source"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    text = str(sample_dir).lower()
    if "movi" in text:
        return "MOVI-D"
    if "genesis" in text or "0417data" in text:
        return "GenesisRigid"
    return "Unknown"


def infer_view_type(sample_dir: Path, meta: dict[str, Any]) -> str:
    paths = meta.get("paths", {}) if isinstance(meta.get("paths"), dict) else {}
    if any(key in paths for key in ("context_video_path", "future_gt_video_path", "full_video_path")):
        return "window"
    if (sample_dir / "context_video.mp4").exists() or (sample_dir / "future_gt_video.mp4").exists():
        return "window"
    return "raw"


def infer_source_sample_dir(sample_dir: Path, meta: dict[str, Any]) -> str:
    source_paths = meta.get("source_paths", {}) if isinstance(meta.get("source_paths"), dict) else {}
    source_sample_dir = str(source_paths.get("source_sample_dir") or "").strip()
    if source_sample_dir:
        return source_sample_dir
    return str(sample_dir)


def infer_scene_id(sample_dir: Path, meta: dict[str, Any]) -> str:
    for key in ("scene_id", "sample_id"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    return sample_dir.name


def infer_scene_composition(sample_dir: Path, meta: dict[str, Any], source_meta: dict[str, Any] | None) -> str:
    for payload in (meta, source_meta or {}):
        value = str(payload.get("scene_composition") or "").strip()
        if value and value.lower() != "unknown":
            return value
    dataset_name = infer_dataset(sample_dir, meta).upper()
    if dataset_name.startswith("MOVI"):
        return "movi_d"
    parts = sample_dir.parts
    if "single_object_preview" in parts:
        return "single_object_preview"
    if "interaction_pair_plus_dynamic" in parts:
        return "interaction_pair_plus_dynamic"
    if "multi_object_free_motion" in parts:
        return "multi_object_free_motion"
    if "movi_d" in parts or "movi-d" in parts:
        return "movi_d"
    return "unknown"


def infer_interaction_pattern(meta: dict[str, Any], source_meta: dict[str, Any] | None) -> str:
    for payload in (meta, source_meta or {}):
        value = str(payload.get("interaction_pattern") or "").strip()
        if value:
            return value
    return "unknown"


def infer_object_count_bucket(sample_dir: Path, meta: dict[str, Any], source_meta: dict[str, Any] | None) -> str:
    for payload in (meta, source_meta or {}):
        value = str(payload.get("object_count_bucket") or "").strip()
        if value and value.lower() != "unknown":
            return value
    num_objects = infer_num_objects(meta, source_meta)
    if num_objects is not None and num_objects > 0:
        return f"count_{num_objects:02d}"
    for part in sample_dir.parts:
        if part.startswith("count_"):
            return part
    return "unknown"


def infer_num_objects(meta: dict[str, Any], source_meta: dict[str, Any] | None) -> int | None:
    for payload in (meta, source_meta or {}):
        value = payload.get("num_objects")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
        objects = payload.get("objects")
        if isinstance(objects, list) and objects:
            return int(len(objects))
    text = " ".join(
        str(meta.get(key) or "")
        for key in ("caption", "description", "prompt")
    )
    match = re.search(r"with\s+(\d+)\s+object\(s\)", text)
    if match:
        return int(match.group(1))
    return None


def infer_fps(meta: dict[str, Any], source_meta: dict[str, Any] | None) -> float | None:
    for payload in (meta, source_meta or {}):
        value = payload.get("fps")
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass
    return None


def movi_detail_caption(meta: dict[str, Any]) -> str:
    scene_id = str(meta.get("scene_id") or meta.get("sample_id") or "unknown_scene")
    prompt = str(meta.get("prompt") or meta.get("caption") or "A MOVI-D scene.").strip()
    background = str(meta.get("background") or "unknown_background")
    num_objects = int(meta.get("num_objects") or len(meta.get("objects", [])) or 0)
    collision_bucket = str(meta.get("collision_type_bucket") or "unknown")
    obj_obj = int(meta.get("obj_obj_event_count") or 0)
    obj_env = int(meta.get("obj_env_event_count") or 0)
    fps = meta.get("fps")
    frames = meta.get("frames")
    object_lines = []
    for obj in meta.get("objects", []):
        if not isinstance(obj, dict):
            continue
        object_lines.append(
            f"object_id={obj.get('object_id')}: "
            f"{obj.get('name') or obj.get('source_object_id') or 'object'} "
            f"(category={obj.get('category')}, role={obj.get('role')}, "
            f"motion_type={obj.get('motion_type')}, dynamic={obj.get('is_dynamic')})"
        )
    object_text = "; ".join(object_lines) if object_lines else "object metadata unavailable"
    return (
        f"Scene {scene_id}: {prompt} "
        f"Background={background}; num_objects={num_objects}; "
        f"collision_type_bucket={collision_bucket}; "
        f"object-object events={obj_obj}; object-environment events={obj_env}. "
        f"Objects: {object_text}. Video frames={frames}, fps={fps}."
    )


def infer_detail_caption(
    sample_dir: Path,
    meta: dict[str, Any],
    source_meta: dict[str, Any] | None,
    dataset_name: str,
    view_type: str,
) -> str:
    for payload in (meta, source_meta or {}):
        value = str(payload.get("detail_caption") or "").strip()
        if value:
            return value
    if dataset_name.upper().startswith("MOVI") and view_type == "raw":
        return movi_detail_caption(meta)
    description = str(meta.get("description") or "").strip()
    if description:
        return description
    caption = str(meta.get("caption") or "").strip()
    if caption:
        return caption
    if source_meta is not None:
        for key in ("description", "caption", "prompt"):
            value = str(source_meta.get(key) or "").strip()
            if value:
                return value
    prompt = str(meta.get("prompt") or "").strip()
    if prompt:
        return prompt
    return sample_dir.name


def infer_caption(
    sample_dir: Path,
    meta: dict[str, Any],
    source_meta: dict[str, Any] | None,
    dataset_name: str,
    view_type: str,
) -> str:
    value = str(meta.get("caption") or "").strip()
    if value:
        return value
    if dataset_name.upper().startswith("MOVI") and view_type == "raw":
        prompt = str(meta.get("prompt") or "").strip()
        if prompt:
            return prompt
    if source_meta is not None:
        source_caption = str(source_meta.get("caption") or "").strip()
        if source_caption:
            return source_caption
    description = str(meta.get("description") or "").strip()
    if description:
        return description
    prompt = str(meta.get("prompt") or "").strip()
    if prompt:
        return prompt
    return sample_dir.name


def normalize_one(sample_dir: Path) -> tuple[bool, str]:
    meta_path = find_meta_path(sample_dir)
    if meta_path is None:
        return False, "missing_meta"
    meta = load_json(meta_path)
    source_meta = None
    source_dir = infer_source_sample_dir(sample_dir, meta)
    source_path = Path(source_dir)
    if source_path.exists() and source_path != sample_dir:
        source_meta_path = find_meta_path(source_path)
        if source_meta_path is not None:
            try:
                source_meta = load_json(source_meta_path)
            except Exception:
                source_meta = None

    dataset_name = infer_dataset(sample_dir, meta)
    view_type = infer_view_type(sample_dir, meta)
    changed = False

    updates: dict[str, Any] = {
        "dataset": dataset_name,
        "view_type": view_type,
        "sample_dir": str(sample_dir),
        "source_sample_dir": source_dir,
        "scene_id": infer_scene_id(sample_dir, meta),
        "scene_composition": infer_scene_composition(sample_dir, meta, source_meta),
        "interaction_pattern": infer_interaction_pattern(meta, source_meta),
        "object_count_bucket": infer_object_count_bucket(sample_dir, meta, source_meta),
        "caption": infer_caption(sample_dir, meta, source_meta, dataset_name, view_type),
        "detail_caption": infer_detail_caption(sample_dir, meta, source_meta, dataset_name, view_type),
    }
    num_objects = infer_num_objects(meta, source_meta)
    if num_objects is not None:
        updates["num_objects"] = num_objects
    fps = infer_fps(meta, source_meta)
    if fps is not None:
        updates["fps"] = fps

    for key, value in updates.items():
        if meta.get(key) != value:
            meta[key] = value
            changed = True

    if changed:
        write_json(meta_path, meta)
    return changed, "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize metadata for samples referenced by by_raw_window")
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    args = parser.parse_args()

    sample_dirs = collect_sample_dirs(args.summary_root.resolve())
    changed = 0
    skipped: list[tuple[str, str]] = []
    for sample_dir in sample_dirs:
        try:
            did_change, status = normalize_one(sample_dir)
        except Exception as exc:
            skipped.append((str(sample_dir), f"{type(exc).__name__}: {exc}"))
            continue
        if status != "ok":
            skipped.append((str(sample_dir), status))
            continue
        if did_change:
            changed += 1
    print(f"samples={len(sample_dirs)} changed={changed} skipped={len(skipped)}")
    for sample_dir, reason in skipped[:20]:
        print(f"SKIP {reason} {sample_dir}")


if __name__ == "__main__":
    main()
