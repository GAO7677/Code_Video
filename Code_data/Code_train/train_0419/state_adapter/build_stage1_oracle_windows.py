#!/usr/bin/env python3
"""Build Wan-friendly oracle-state windows from rigid synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from build_stage1_subsets import (
    WINDOW_STRIDE,
    find_samples,
    first_contact_frame_stage1a,
    future_main_object_visibility_ok,
    load_contact_annotations,
    load_raw_state,
    normalize_state,
    resolve_main_object_index,
    rgb_frame_paths,
    window_has_visible_object_every_frame,
)
from motion_complexity import infer_motion_complexity
from window_interactions import infer_window_interactions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Wan-friendly oracle-state windows from Genesis rigid data.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
        ),
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
        ),
    )
    parser.add_argument("--sample_filter", type=str, default="")
    parser.add_argument(
        "--count_buckets",
        type=str,
        default="count_01,count_02,count_03_04",
        help="Comma-separated count buckets to include.",
    )
    parser.add_argument(
        "--context_len",
        type=int,
        default=8,
        help="Raw RGB context frame count.",
    )
    parser.add_argument(
        "--future_lengths",
        type=str,
        default="5,9,13",
        help="Comma-separated raw future lengths. Defaults keep 4n+1 totals with context_len=8.",
    )
    parser.add_argument(
        "--contact_mode",
        type=str,
        default="none",
        choices=["none", "stage1a"],
        help="Use stage1a to keep future strictly before first contact; none uses the full clip.",
    )
    parser.add_argument(
        "--safety_margin",
        type=int,
        default=2,
        help="Keep future windows this many frames away from first contact when contact_mode=stage1a.",
    )
    parser.add_argument(
        "--future_main_visibility_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--max_source_samples",
        type=int,
        default=0,
        help="0 means scan all source samples.",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=0,
        help="0 means no limit.",
    )
    parser.add_argument(
        "--include_invalid_by_qa",
        action="store_true",
        help="Include samples under invalid_by_qa directories.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [int(item) for item in items]


def infer_prompt(sample_dir: Path, metadata: Dict[str, object]) -> str:
    caption_json_path = sample_dir / "caption.json"
    if caption_json_path.exists():
        try:
            payload = json.loads(caption_json_path.read_text(encoding="utf-8"))
            for key in ("simple_caption", "caption"):
                text = str(payload.get(key, "")).strip()
                if text:
                    return text
        except Exception:
            pass

    for name in ("caption_simple.txt", "caption.txt"):
        path = sample_dir / name
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text

    motion = str(metadata.get("motion_category", "")).strip() or "unknown_motion"
    scene = str(metadata.get("scene_composition", "")).strip() or "rigid_scene"
    num_objects = int(metadata.get("num_objects", 0) or 0)
    return f"{scene} with {num_objects} object(s), motion={motion}."


def extract_object_text_meta(sample_dir: Path, metadata: Dict[str, object]) -> List[Dict[str, object]]:
    objects = metadata.get("objects", [])
    merged: List[Dict[str, object]] = []
    if isinstance(objects, list):
        merged = [dict(obj) for obj in objects if isinstance(obj, dict)]

    caption_json_path = sample_dir / "caption.json"
    if not caption_json_path.exists():
        return merged

    try:
        payload = json.loads(caption_json_path.read_text(encoding="utf-8"))
    except Exception:
        return merged

    caption_objects = payload.get("objects", [])
    if not isinstance(caption_objects, list):
        return merged

    by_object_id = {
        int(obj["object_id"]): obj
        for obj in merged
        if isinstance(obj, dict) and obj.get("object_id") is not None
    }
    for obj in caption_objects:
        if not isinstance(obj, dict) or obj.get("object_id") is None:
            continue
        object_id = int(obj["object_id"])
        base = by_object_id.get(object_id, {"object_id": object_id})
        base.update(
            {
                "name": obj.get("name"),
                "category": obj.get("category"),
            }
        )
        by_object_id[object_id] = base
    return [by_object_id[key] for key in sorted(by_object_id)]


def export_window(
    out_root: Path,
    rel_sample: Path,
    window_name: str,
    payload: Dict[str, np.ndarray],
    meta_payload: Dict[str, object],
) -> Path:
    out_dir = out_root / rel_sample / window_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "state_pair.npz", **payload)
    (out_dir / "pair_meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    args = parse_args()
    dataset_train_root = args.dataset_root / "train"
    future_lengths = parse_int_list(args.future_lengths)
    count_buckets = [item.strip() for item in args.count_buckets.split(",") if item.strip()]

    samples = find_samples(dataset_train_root, args.sample_filter, count_buckets)
    if not args.include_invalid_by_qa:
        samples = [sample for sample in samples if "invalid_by_qa" not in sample.parts]
    if int(args.max_source_samples) > 0:
        samples = samples[: int(args.max_source_samples)]
    if not samples:
        raise RuntimeError(f"No source samples found under {dataset_train_root}")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    subset_name = "stage1a_oracle_wan" if args.contact_mode == "stage1a" else "oracle_wan_alltrain"
    manifest = {
        "subset": subset_name,
        "context_len": int(args.context_len),
        "future_lengths": future_lengths,
        "contact_mode": str(args.contact_mode),
        "accepted": [],
        "skipped": [],
    }

    accepted = 0
    for sample_dir in samples:
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        fps = float(metadata.get("fps", metadata.get("video_fps", 12.0)) or 12.0)
        if fps <= 0:
            fps = 12.0

        raw = load_raw_state(sample_dir, fps)
        state_raw = raw["state_raw"]
        visibility_mask = raw["visibility_mask"]
        object_ids = raw["object_ids"]
        seg_ids = raw["seg_ids"]
        T = int(state_raw.shape[0])

        cam = metadata["camera_intrinsics"]
        width, height = map(float, metadata["resolution"])
        state_norm = normalize_state(
            state_raw=state_raw,
            width=width,
            height=height,
            depth_near=float(cam["near"]),
            depth_far=float(cam["far"]),
        )
        if args.contact_mode == "stage1a":
            ann = load_contact_annotations(sample_dir=sample_dir, num_objects=int(object_ids.shape[0]), T=T)
            first_contact = first_contact_frame_stage1a(ann, T)
            valid_end = T if first_contact is None else max(0, int(first_contact) - int(args.safety_margin))
        else:
            first_contact = None
            valid_end = T

        rel_sample = sample_dir.relative_to(dataset_train_root)
        main_object_index = resolve_main_object_index(metadata, object_ids)
        object_text_meta = extract_object_text_meta(sample_dir, metadata)
        prompt = infer_prompt(sample_dir, metadata)
        sample_window_count = 0

        for future_len in future_lengths:
            min_total = int(args.context_len) + int(future_len)
            if valid_end < min_total:
                continue
            max_start = valid_end - min_total
            for start in range(0, max_start + 1, WINDOW_STRIDE):
                c0 = int(start)
                c1 = c0 + int(args.context_len)
                f0 = c1
                f1 = f0 + int(future_len)
                if not window_has_visible_object_every_frame(visibility_mask, c0, c1):
                    continue
                future_visible_ok, future_vis_ratio = future_main_object_visibility_ok(
                    visibility_mask=visibility_mask,
                    start=f0,
                    end=f1,
                    main_object_index=main_object_index,
                    threshold=float(args.future_main_visibility_threshold),
                )
                if not future_visible_ok:
                    continue

                x_idx = np.arange(c0, c1, dtype=np.int32)
                y_idx = np.arange(f0, f1, dtype=np.int32)
                payload = {
                    "object_ids": object_ids.astype(np.int32),
                    "seg_ids": seg_ids.astype(np.int32),
                    "visibility_mask": visibility_mask.astype(np.uint8),
                    "state_raw": state_raw.astype(np.float32),
                    "state_norm": state_norm.astype(np.float32),
                    "x_state_raw": state_raw[c0:c1].astype(np.float32),
                    "y_state_raw": state_raw[f0:f1].astype(np.float32),
                    "x_state_norm": state_norm[c0:c1].astype(np.float32),
                    "y_state_norm": state_norm[f0:f1].astype(np.float32),
                    "x_visibility": visibility_mask[c0:c1].astype(np.uint8),
                    "y_visibility": visibility_mask[f0:f1].astype(np.uint8),
                    "x_frame_indices": x_idx,
                    "y_frame_indices": y_idx,
                    "dt": np.asarray(raw["dt"], dtype=np.float32),
                }
                meta_payload = {
                    "prompt": prompt,
                    "source_scene_id": metadata["scene_id"],
                    "source_sample_dir": str(sample_dir),
                    "context_len": int(args.context_len),
                    "future_len": int(future_len),
                    "start_index": int(start),
                    "first_contact_frame": None if first_contact is None else int(first_contact),
                    "valid_end": int(valid_end),
                    "main_object_index": int(main_object_index),
                    "future_main_visibility_ratio": float(future_vis_ratio),
                    "resolution": metadata.get("resolution"),
                    "camera_intrinsics": metadata.get("camera_intrinsics"),
                    "objects": object_text_meta,
                    "x_frame_paths": rgb_frame_paths(sample_dir, x_idx),
                    "y_frame_paths": rgb_frame_paths(sample_dir, y_idx),
                    "motion_complexity": infer_motion_complexity(
                        state_norm=state_norm[f0:f1].astype(np.float32),
                        visibility_mask=visibility_mask[f0:f1].astype(np.uint8),
                    ),
                }
                meta_payload["window_interactions"] = infer_window_interactions(meta_payload)
                window_name = (
                    f"window_s{start:04d}_ctx{int(args.context_len):02d}_fut{int(future_len):02d}"
                )
                out_dir = export_window(out_root, rel_sample, window_name, payload, meta_payload)
                manifest["accepted"].append(
                    {
                        "sample_dir": str(sample_dir),
                        "future_len": int(future_len),
                        "start_index": int(start),
                        "out_dir": str(out_dir),
                    }
                )
                accepted += 1
                sample_window_count += 1
                if int(args.max_windows) > 0 and accepted >= int(args.max_windows):
                    break
            if int(args.max_windows) > 0 and accepted >= int(args.max_windows):
                break

        if sample_window_count == 0:
            manifest["skipped"].append(
                {
                    "sample_dir": str(sample_dir),
                    "reason": "no_valid_wan_window",
                    "contact_mode": str(args.contact_mode),
                }
            )

        if int(args.max_windows) > 0 and accepted >= int(args.max_windows):
            break

    (out_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "dataset_root": str(args.dataset_root),
        "out_root": str(out_root),
        "accepted_windows": len(manifest["accepted"]),
        "context_len": int(args.context_len),
        "future_lengths": future_lengths,
        "contact_mode": str(args.contact_mode),
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"DONE accepted_windows={len(manifest['accepted'])}")


if __name__ == "__main__":
    main()
