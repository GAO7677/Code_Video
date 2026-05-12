#!/usr/bin/env python3
# 用途：重建 Genesis stage1adapter 的 test/val split。
from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
TRAIN_WINDOW_ROOT = DATASET_ROOT / "stage1adapter" / "train" / "genesis" / "rigid" / "single_object_preview" / "count_01"
TRAIN_WINDOW_COUNT02_ROOT = DATASET_ROOT / "stage1adapter" / "train" / "genesis" / "rigid" / "multi_object_free_motion" / "count_02"
TEST_ROOT = DATASET_ROOT / "stage1adapter" / "test" / "genesis"
FIXED24_ROOT = DATASET_ROOT / "stage1adapter" / "benchmark" / "fixed24" / "genesis"
VALIDATION100_ROOT = DATASET_ROOT / "stage1adapter" / "benchmark" / "validation100" / "genesis"
MANIFEST_ROOT = DATASET_ROOT / "stage1adapter" / "manifests"

CASE_ORDER = [
    "case003_static_highdrop",
    "case900_random_parabola",
    "case901_high_drop",
    "case005_entry_left",
    "case006_entry_right",
    "case007_entry_fast_center",
]
COUNT02_CASE = "case210_multi2_projectile_nocollision"
SPLIT_PLAN: dict[str, dict[str, Any]] = {
    "test": {
        "root": TEST_ROOT,
        "split_value": "test",
        "prefix_start": 3000,
        "case_quota": {case_name: 1 for case_name in CASE_ORDER},
        "count02_quota": 2,
    },
    "fixed24": {
        "root": FIXED24_ROOT,
        "split_value": "val",
        "prefix_start": 4000,
        "case_quota": {case_name: 1 for case_name in CASE_ORDER},
        "count02_quota": 2,
    },
    "validation100": {
        "root": VALIDATION100_ROOT,
        "split_value": "val",
        "prefix_start": 5000,
        "case_quota": {case_name: 4 for case_name in CASE_ORDER},
        "count02_quota": 6,
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_sort_key(sample_dir: Path) -> tuple[int, int, str]:
    name = sample_dir.name
    object_id = name.split("__", 1)[0]
    try:
        object_rank = int(object_id)
    except Exception:
        object_rank = 10**9
    ratio_rank = 99
    if "__ratio11" in name:
        ratio_rank = 11
    elif "__ratio12" in name:
        ratio_rank = 12
    return (object_rank, ratio_rank, name)


def case_name_for_dir(sample_dir: Path) -> str:
    name = sample_dir.name
    for case_name in CASE_ORDER:
        if case_name in name:
            return case_name
    raise ValueError(f"unsupported case family: {sample_dir}")


def collect_candidates() -> dict[str, list[Path]]:
    candidates: dict[str, list[Path]] = {case_name: [] for case_name in CASE_ORDER + [COUNT02_CASE]}
    for meta_path in TRAIN_WINDOW_ROOT.rglob("meta.json"):
        sample_dir = meta_path.parent
        name = sample_dir.name
        if "__cf_" in name:
            continue
        matched_case = None
        for case_name in CASE_ORDER:
            if case_name in name:
                matched_case = case_name
                break
        if matched_case is None:
            continue
        candidates[matched_case].append(sample_dir)
    for case_name in CASE_ORDER:
        candidates[case_name] = sorted(candidates[case_name], key=sample_sort_key)
    for meta_path in TRAIN_WINDOW_COUNT02_ROOT.rglob("meta.json"):
        sample_dir = meta_path.parent
        if COUNT02_CASE not in sample_dir.name:
            continue
        candidates[COUNT02_CASE].append(sample_dir)
    candidates[COUNT02_CASE] = sorted(candidates[COUNT02_CASE], key=sample_sort_key)
    return candidates


def choose_samples(candidates: dict[str, list[Path]]) -> dict[str, list[Path]]:
    chosen: dict[str, list[Path]] = {}
    globally_used: set[Path] = set()
    for split_name, split_cfg in SPLIT_PLAN.items():
        split_used_object_ids: set[str] = set()
        split_selected: list[Path] = []
        for case_name in CASE_ORDER:
            quota = int(split_cfg["case_quota"].get(case_name, 0))
            pool = candidates[case_name]
            picked = 0

            # First pass: avoid reusing the same object id within one split.
            for sample_dir in pool:
                object_id = sample_dir.name.split("__", 1)[0]
                if sample_dir in globally_used or object_id in split_used_object_ids:
                    continue
                split_selected.append(sample_dir)
                globally_used.add(sample_dir)
                split_used_object_ids.add(object_id)
                picked += 1
                if picked >= quota:
                    break

            # Second pass: if needed, allow repeated object ids but keep sample dirs unique.
            if picked < quota:
                for sample_dir in pool:
                    if sample_dir in globally_used:
                        continue
                    split_selected.append(sample_dir)
                    globally_used.add(sample_dir)
                    picked += 1
                    if picked >= quota:
                        break

            if picked < quota:
                raise RuntimeError(f"not enough candidates for {split_name}/{case_name}: need {quota}, got {picked}")

        count02_quota = int(split_cfg.get("count02_quota", 0) or 0)
        if count02_quota > 0:
            picked = 0
            for sample_dir in candidates[COUNT02_CASE]:
                object_id = sample_dir.name.split("__", 1)[0]
                if sample_dir in globally_used or object_id in split_used_object_ids:
                    continue
                split_selected.append(sample_dir)
                globally_used.add(sample_dir)
                split_used_object_ids.add(object_id)
                picked += 1
                if picked >= count02_quota:
                    break
            if picked < count02_quota:
                for sample_dir in candidates[COUNT02_CASE]:
                    if sample_dir in globally_used:
                        continue
                    split_selected.append(sample_dir)
                    globally_used.add(sample_dir)
                    picked += 1
                    if picked >= count02_quota:
                        break
            if picked < count02_quota:
                raise RuntimeError(f"not enough candidates for {split_name}/{COUNT02_CASE}: need {count02_quota}, got {picked}")
        chosen[split_name] = split_selected
    return chosen


def clean_target_root(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def wrapper_name(prefix_id: int, source_dir: Path) -> str:
    return f"genesis_heldout_{prefix_id:04d}__{source_dir.name}"


def update_meta(source_meta: dict[str, Any], target_dir: Path, split_value: str, source_dir: Path) -> dict[str, Any]:
    meta = deepcopy(source_meta)
    sample_id = target_dir.name
    meta["sample_id"] = sample_id
    meta["sample_label"] = sample_id
    meta["scene_id"] = sample_id
    meta["split"] = split_value
    meta["sample_dir"] = str(target_dir)
    meta["view_type"] = "window"
    meta["source_window_dir"] = str(source_dir)

    paths = dict(meta.get("paths") or {})
    paths["sample_dir"] = str(target_dir)
    for key, rel in (
        ("rgb_video", "videos/rgb.mp4"),
        ("depth_video", "videos/depth.mp4"),
        ("future_gt_video_path", "future_gt_video.mp4"),
        ("full_video_path", "full_video.mp4"),
        ("context_video_path", "context_video.mp4"),
        ("first_frame_path", "first_frame.png"),
        ("meta_json_path", "meta.json"),
    ):
        candidate = target_dir / rel
        if candidate.exists() or key == "meta_json_path":
            paths[key] = str(candidate)
    meta["paths"] = paths

    source_paths = dict(meta.get("source_paths") or {})
    source_paths["meta_json_path"] = str(target_dir / "meta.json")
    source_paths["pair_meta_json_path"] = str(target_dir / "pair_meta.json")
    source_paths["segment_state_npz_path"] = str(target_dir / "segment_state.npz")
    source_paths["state_pair_npz_path"] = str(target_dir / "state_pair.npz")
    source_paths["source_window_dir"] = str(source_dir)
    source_paths.setdefault("source_sample_dir", str(source_meta.get("source_sample_dir") or source_dir))
    source_paths.setdefault("source_meta_json_path", str(source_dir / "meta.json"))
    meta["source_paths"] = source_paths

    return meta


def build_manifest_record(target_dir: Path, meta: dict[str, Any], rel_dir: str) -> dict[str, Any]:
    return {
        "sample_id": str(meta["sample_id"]),
        "dataset": "genesis",
        "split": str(meta["split"]),
        "sample_dir": str(target_dir),
        "rel_dir": rel_dir,
        "caption": str(meta.get("caption") or ""),
        "context_frames": int(meta.get("context_frames") or 0),
        "future_frames": int(meta.get("future_frames") or 0),
        "full_frames": int(meta.get("frames") or meta.get("raw_frames") or 0),
        "collision_bucket": "no_collision",
        "motion_complexity": "simple",
        "segment_kind": str((meta.get("window_range") or {}).get("segment_kind") or "full_no_collision_window"),
        "context_video_path": str(target_dir / "context_video.mp4"),
        "future_gt_video_path": str(target_dir / "future_gt_video.mp4"),
        "full_video_path": str(target_dir / "full_video.mp4"),
        "meta_json_path": str(target_dir / "meta.json"),
    }


def create_wrapper(target_dir: Path, source_dir: Path, split_value: str) -> dict[str, Any]:
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.is_symlink() or target_dir.is_file():
            target_dir.unlink()
        else:
            shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for child in source_dir.iterdir():
        if child.name == "meta.json":
            continue
        (target_dir / child.name).symlink_to(child)

    source_meta = load_json(source_dir / "meta.json")
    target_meta = update_meta(source_meta=source_meta, target_dir=target_dir, split_value=split_value, source_dir=source_dir)
    write_json(target_dir / "meta.json", target_meta)
    return target_meta


def rebuild_split(split_name: str, selected: list[Path]) -> list[dict[str, Any]]:
    split_cfg = SPLIT_PLAN[split_name]
    root = Path(split_cfg["root"])
    clean_target_root(root)

    prefix_id = int(split_cfg["prefix_start"])
    manifest_records: list[dict[str, Any]] = []
    for idx, source_dir in enumerate(selected):
        sample_name = wrapper_name(prefix_id + idx, source_dir)
        target_dir = root / sample_name
        meta = create_wrapper(target_dir=target_dir, source_dir=source_dir, split_value=str(split_cfg["split_value"]))
        rel_dir = str(target_dir.relative_to(DATASET_ROOT))
        manifest_records.append(build_manifest_record(target_dir, meta, rel_dir))
    return manifest_records


def main() -> None:
    candidates = collect_candidates()
    selected = choose_samples(candidates)

    test_manifest = rebuild_split("test", selected["test"])
    rebuild_split("fixed24", selected["fixed24"])
    rebuild_split("validation100", selected["validation100"])

    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(MANIFEST_ROOT / "test_genesis_items.json", test_manifest)

    summary = {
        split_name: [sample_dir.name for sample_dir in sample_dirs]
        for split_name, sample_dirs in selected.items()
    }
    write_json(MANIFEST_ROOT / "genesis_eval_rebuild_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
