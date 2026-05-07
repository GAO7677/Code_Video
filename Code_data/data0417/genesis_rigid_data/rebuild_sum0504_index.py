#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sample_bucket_labels import (
    COLLISION_PROFILE_ORDER,
    COUNT_BUCKET_ORDER,
    compute_derived_tags,
    find_sample_meta_path,
    load_sample_arrays,
)


DEFAULT_OUTPUT_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504")
RAW_TRAIN_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
STAGE1_TRAIN_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/train/genesis/rigid"
)
STAGE1_TEST_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/test/genesis"
)
STAGE1_BENCHMARK_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/stage1adapter/benchmark"
)
VAL_SUBSETS = ("fixed24", "validation100")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild sum0504 path-only indices from Genesis raw/window samples.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def json_dump(path: Path, payload: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lines(path: Path, lines: Iterable[str], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_leaf_sample_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    sample_dirs: set[Path] = set()
    for meta_name in ("meta.json", "metadata.json"):
        for meta_path in root.glob("*/*/*/" + meta_name):
            sample_dirs.add(meta_path.parent)
    return sorted(sample_dirs)


def scan_test_or_val_sample_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    sample_dirs: set[Path] = set()
    for meta_name in ("meta.json", "metadata.json"):
        for meta_path in root.rglob(meta_name):
            sample_dirs.add(meta_path.parent)
    if not sample_dirs:
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            try:
                sample_dirs.add(find_sample_meta_path(child).parent)
            except Exception:
                continue
    return sorted(sample_dirs)


def choose_split_roots() -> Dict[str, List[Path]]:
    return {
        "train": [RAW_TRAIN_ROOT, STAGE1_TRAIN_ROOT],
        "test": [STAGE1_TEST_ROOT],
        "val": [STAGE1_BENCHMARK_ROOT / name / "genesis" for name in VAL_SUBSETS],
    }


def resolve_source_sample_dir(metadata: Dict[str, Any]) -> Path | None:
    candidates = [
        metadata.get("source_sample_dir"),
        metadata.get("source_window_dir"),
        (metadata.get("source_paths") or {}).get("source_sample_dir"),
        (metadata.get("source_paths") or {}).get("source_window_dir"),
        (metadata.get("source_paths") or {}).get("heldout_sample_dir"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path
    source_meta_candidates = [
        (metadata.get("source_paths") or {}).get("source_metadata_json_path"),
        (metadata.get("source_paths") or {}).get("source_meta_json_path"),
        (metadata.get("source_paths") or {}).get("meta_json_path"),
    ]
    for candidate in source_meta_candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            return path.parent
    return None


def infer_count_bucket_from_num_objects(num_objects: Any) -> str:
    try:
        value = int(num_objects)
    except Exception:
        return ""
    if value <= 1:
        return "count_01"
    if value == 2:
        return "count_02"
    if value <= 4:
        return "count_03_04"
    return "count_03_04"


def infer_count_bucket_from_path(path: Path | None) -> str:
    if path is None:
        return ""
    for part in path.parts:
        if part in COUNT_BUCKET_ORDER:
            return part
    return ""


def infer_case_name_from_sample_name(sample_name: str) -> str:
    parts = sample_name.split("__")
    if len(parts) >= 3:
        return parts[-1]
    if len(parts) >= 2:
        return parts[1]
    return ""


def enrich_metadata(sample_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(metadata)
    source_sample_dir = resolve_source_sample_dir(enriched)

    if not enriched.get("object_count_bucket"):
        source_bucket = infer_count_bucket_from_path(source_sample_dir)
        if source_bucket:
            enriched["object_count_bucket"] = source_bucket

    if not enriched.get("object_count_bucket"):
        local_bucket = infer_count_bucket_from_path(sample_dir)
        if local_bucket:
            enriched["object_count_bucket"] = local_bucket

    if not enriched.get("object_count_bucket"):
        inferred = infer_count_bucket_from_num_objects(enriched.get("num_objects"))
        if inferred:
            enriched["object_count_bucket"] = inferred

    if source_sample_dir is not None:
        source_meta_path: Path | None = None
        for name in ("meta.json", "metadata.json"):
            candidate = source_sample_dir / name
            if candidate.exists():
                source_meta_path = candidate
                break
        if source_meta_path is not None:
            try:
                source_meta = load_json(source_meta_path)
            except Exception:
                source_meta = {}
            for key in (
                "object_count_bucket",
                "case_name",
                "scene_composition",
                "interaction_pattern",
                "num_objects",
                "detail_caption",
                "caption",
            ):
                if not enriched.get(key) and source_meta.get(key):
                    enriched[key] = source_meta.get(key)

    if not enriched.get("case_name"):
        inferred_case_name = infer_case_name_from_sample_name(sample_dir.name)
        if inferred_case_name:
            enriched["case_name"] = inferred_case_name

    return enriched


def extract_window_frame_indices(metadata: Dict[str, Any]) -> List[int]:
    window_range = metadata.get("window_range")
    if not isinstance(window_range, dict):
        return []
    frame_indices = window_range.get("orig_full_frame_indices")
    if isinstance(frame_indices, list) and frame_indices:
        try:
            return [int(value) for value in frame_indices]
        except Exception:
            return []
    start_index = window_range.get("start_index")
    end_exclusive = window_range.get("end_exclusive")
    if start_index is None or end_exclusive is None:
        return []
    try:
        start = int(start_index)
        end = int(end_exclusive)
    except Exception:
        return []
    if end <= start:
        return []
    return list(range(start, end))


def normalize_payload_frames(payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    target = metadata.get("frames")
    try:
        target_frames = int(target) if target is not None else 0
    except Exception:
        target_frames = 0

    if target_frames <= 0:
        target_frames = min(
            int(payload["linear_vel"].shape[0]),
            int(payload["com_pos"].shape[0]),
            int(payload["bbox_xyxy"].shape[0]),
            int(payload["visibility_mask"].shape[0]),
        )

    frame_indices = extract_window_frame_indices(metadata)

    def _align(array: Any) -> Any:
        if int(array.shape[0]) == target_frames:
            return array
        if (
            int(array.shape[0]) > target_frames
            and frame_indices
            and len(frame_indices) == target_frames
            and max(frame_indices) < int(array.shape[0])
        ):
            return array[frame_indices]
        return array[:target_frames]

    payload["linear_vel"] = _align(payload["linear_vel"])
    payload["com_pos"] = _align(payload["com_pos"])
    payload["bbox_xyxy"] = _align(payload["bbox_xyxy"])
    payload["visibility_mask"] = _align(payload["visibility_mask"])
    payload["metadata"] = metadata
    return payload


def slice_payload_to_window(payload: Dict[str, Any], frame_indices: List[int], metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not frame_indices:
        return normalize_payload_frames(payload, metadata)

    linear_vel = payload["linear_vel"]
    num_frames = int(linear_vel.shape[0])
    clamped = [idx for idx in frame_indices if 0 <= idx < num_frames]
    if not clamped:
        return normalize_payload_frames(payload, metadata)

    payload["metadata"] = metadata
    payload["events"] = []
    payload["linear_vel"] = payload["linear_vel"][clamped]
    payload["com_pos"] = payload["com_pos"][clamped]
    payload["bbox_xyxy"] = payload["bbox_xyxy"][clamped]
    payload["visibility_mask"] = payload["visibility_mask"][clamped]
    return normalize_payload_frames(payload, metadata)


def load_payload_for_classification(sample_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = load_sample_arrays(sample_dir)
        return normalize_payload_frames(payload, metadata)
    except Exception:
        source_sample_dir = resolve_source_sample_dir(metadata)
        if source_sample_dir is None:
            raise
        payload = load_sample_arrays(source_sample_dir)
        return slice_payload_to_window(payload, extract_window_frame_indices(metadata), metadata)


def classify_sample(sample_dir: Path) -> Tuple[str | None, Dict[str, Any] | None, str | None]:
    try:
        meta_path = find_sample_meta_path(sample_dir)
        metadata = enrich_metadata(sample_dir, load_json(meta_path))
    except Exception as exc:
        return None, None, f"meta_error:{type(exc).__name__}"

    count_bucket = str(metadata.get("object_count_bucket", "") or "")
    if count_bucket not in COUNT_BUCKET_ORDER:
        return None, metadata, f"bad_count_bucket:{count_bucket or 'missing'}"

    try:
        payload = load_payload_for_classification(sample_dir, metadata)
        derived_tags = compute_derived_tags(
            metadata=payload["metadata"],
            events=payload["events"],
            linear_vel=payload["linear_vel"],
            visibility_mask=payload["visibility_mask"],
            com_pos=payload["com_pos"],
            bbox_xyxy=payload["bbox_xyxy"],
        )
    except Exception as exc:
        return None, metadata, f"derived_error:{type(exc).__name__}"

    collision_bucket = str(derived_tags.get("collision_profile_bucket", "") or "")
    if collision_bucket not in COLLISION_PROFILE_ORDER:
        return None, metadata, f"bad_collision_bucket:{collision_bucket or 'missing'}"

    return collision_bucket, metadata, None


def iter_split_samples(split: str, roots: List[Path]) -> List[Path]:
    sample_dirs: List[Path] = []
    for root in roots:
        if split == "train":
            if root == RAW_TRAIN_ROOT or root == STAGE1_TRAIN_ROOT:
                sample_dirs.extend(scan_leaf_sample_dirs(root))
            else:
                sample_dirs.extend(scan_test_or_val_sample_dirs(root))
        else:
            sample_dirs.extend(scan_test_or_val_sample_dirs(root))
    return sorted(set(sample_dirs))


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    roots_by_split = choose_split_roots()

    grouped: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    included_by_leaf: Dict[str, int] = {}
    excluded_breakdown: Counter[str] = Counter()
    included_samples = 0
    excluded_samples = 0

    for split, roots in roots_by_split.items():
        for sample_dir in iter_split_samples(split, roots):
            collision_bucket, metadata, error_key = classify_sample(sample_dir)
            if collision_bucket is None:
                dataset_name = "GenesisRigid"
                if metadata is not None:
                    dataset_name = str(metadata.get("dataset") or metadata.get("dataset_name") or "GenesisRigid")
                excluded_breakdown[f"{split}/unmapped/{dataset_name}:{error_key}"] += 1
                excluded_samples += 1
                continue

            key = (
                split,
                "rigid",
                str(metadata.get("object_count_bucket")),
                str(collision_bucket),
            )
            grouped[key].append(str(sample_dir))
            included_samples += 1

    if not args.dry_run and output_root.exists():
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        output_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        leaf_counts: Dict[str, int] = {}
        for count_bucket in COUNT_BUCKET_ORDER:
            for collision_bucket in COLLISION_PROFILE_ORDER:
                key = (split, "rigid", count_bucket, collision_bucket)
                lines = sorted(grouped.get(key, []))
                leaf_rel = Path(split) / "rigid" / count_bucket / collision_bucket
                write_lines(output_root / leaf_rel / "samples.txt", lines, args.dry_run)
                summary = {
                    "split": split,
                    "simulator_type": "rigid",
                    "object_count_bucket": count_bucket,
                    "collision_bucket": collision_bucket,
                    "num_samples": len(lines),
                }
                json_dump(output_root / leaf_rel / "summary.json", summary, args.dry_run)
                included_by_leaf[str(leaf_rel)] = len(lines)
                leaf_counts[f"rigid/{count_bucket}/{collision_bucket}"] = len(lines)
                split_counts[split][f"rigid/{count_bucket}/{collision_bucket}"] = len(lines)

        split_summary = {
            "split": split,
            "num_samples": int(sum(leaf_counts.values())),
            "leaf_counts": leaf_counts,
        }
        json_dump(output_root / split / "summary.json", split_summary, args.dry_run)

    root_summary = {
        "schema": "<split>/<simulator_type>/<object_count_bucket>/<collision_bucket>/samples.txt",
        "notes": [
            "Only path index files are created; raw data folders are not moved.",
            "train includes Genesis raw train/rigid plus stage1adapter/train genesis windows.",
            "test includes stage1adapter/test genesis windows; val includes benchmark/fixed24 and benchmark/validation100 genesis windows.",
            "Samples are included only if they can be stably mapped to count_01/count_02/count_03_04 and to one of the six collision buckets.",
        ],
        "included_samples": int(included_samples),
        "excluded_samples": int(excluded_samples),
        "included_by_leaf": included_by_leaf,
        "excluded_breakdown": dict(sorted(excluded_breakdown.items())),
    }
    json_dump(output_root / "summary.json", root_summary, args.dry_run)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "included_samples": included_samples,
                "excluded_samples": excluded_samples,
                "excluded_breakdown": dict(sorted(excluded_breakdown.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
