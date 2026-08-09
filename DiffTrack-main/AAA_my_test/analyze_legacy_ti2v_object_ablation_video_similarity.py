#!/usr/bin/env python3
"""Measure decoded-frame similarity for fixed-query and temporal-tube ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


DEFAULT_CASE = "0613pybullet_sample_001460_w002"
DEFAULT_SEED = 47326
DEFAULT_BASELINE = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/runs"
)
DEFAULT_EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326"
)
DEFAULT_FIXED_ROOT = DEFAULT_EXPERIMENT_ROOT / "attention_matrix_ablations_v2"
DEFAULT_TUBE_ROOT = (
    DEFAULT_EXPERIMENT_ROOT / "attention_matrix_ablations_temporal_tube_v1"
)

OBJECT_MASK_MODES = (
    "self_only",
    "incoming_only",
    "outgoing_only",
    "query_row",
    "key_value_column",
    "cross_boundary",
    "row_and_column",
    "literal_kv_zero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--tube-root", type=Path, default=DEFAULT_TUBE_ROOT)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=176)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_video(path: Path, size: tuple[int, int]) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")

    decoded_digest = hashlib.sha256()
    frames = []
    original_shape = None
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if original_shape is None:
            original_shape = tuple(int(value) for value in frame.shape)
            decoded_digest.update(np.asarray(original_shape, dtype=np.int64).tobytes())
        elif tuple(frame.shape) != original_shape:
            raise RuntimeError(f"variable frame shape in {path}")
        decoded_digest.update(frame.tobytes(order="C"))
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    capture.release()
    if not frames:
        raise RuntimeError(f"no decoded frames: {path}")
    return {
        "frames": np.stack(frames),
        "frame_count": len(frames),
        "original_shape": list(original_shape or ()),
        "fps": fps,
        "file_size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "decoded_sha256": decoded_digest.hexdigest(),
    }


def metric_summary(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a, b = left["frames"], right["frames"]
    if a.shape != b.shape:
        raise ValueError(f"decoded comparison shapes differ: {a.shape} vs {b.shape}")

    diff = a.astype(np.float32) - b.astype(np.float32)
    abs_diff = np.abs(diff)
    mse = float(np.mean(np.square(diff), dtype=np.float64))
    mae = float(np.mean(abs_diff, dtype=np.float64) / 255.0)
    changed = float(np.mean(np.any(a != b, axis=-1), dtype=np.float64))
    max_abs = int(np.max(abs_diff))
    psnr = None if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse))

    frame_ssim = np.asarray(
        [
            structural_similarity(x, y, channel_axis=-1, data_range=255)
            for x, y in zip(a, b, strict=True)
        ],
        dtype=np.float64,
    )
    if len(a) > 1:
        delta_a = np.diff(a.astype(np.int16), axis=0)
        delta_b = np.diff(b.astype(np.int16), axis=0)
        temporal_delta_mae = float(
            np.mean(np.abs(delta_a - delta_b), dtype=np.float64) / 255.0
        )
    else:
        temporal_delta_mae = 0.0

    decoded_equal = (
        left["decoded_sha256"] == right["decoded_sha256"]
        and left["original_shape"] == right["original_shape"]
        and left["frame_count"] == right["frame_count"]
    )
    return {
        "decoded_equal": decoded_equal,
        "comparison_frame_count": int(a.shape[0]),
        "mae_0_1": round(mae, 9),
        "mse_0_255": round(mse, 6),
        "psnr_db": None if psnr is None else round(psnr, 6),
        "ssim_mean": round(float(frame_ssim.mean()), 9),
        "ssim_min": round(float(frame_ssim.min()), 9),
        "ssim_max": round(float(frame_ssim.max()), 9),
        "changed_pixel_fraction": round(changed, 9),
        "max_abs_channel_diff": max_abs,
        "temporal_delta_mae_0_1": round(temporal_delta_mae, 9),
    }


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest is not an object: {path}")
    return payload


def collect_protocol_videos(
    protocol: str, root: Path, case: str, seed: int
) -> list[dict[str, Any]]:
    case_root = root / case / f"seed_{seed:05d}"
    records = []
    for manifest_path in sorted(case_root.glob("*__top100/manifest.json")):
        manifest = read_manifest(manifest_path)
        mask_mode = str(manifest.get("mask_mode") or "")
        target_scope = str(manifest.get("target_scope") or "")
        region = str(manifest.get("region") or "")
        if mask_mode not in OBJECT_MASK_MODES:
            continue
        if target_scope not in {"single_object", "all_objects"}:
            continue
        if int(manifest.get("top_n", -1)) != 100:
            continue
        video_path = manifest_path.parent / "generated.mp4"
        if not video_path.is_file():
            raise RuntimeError(f"missing generated video: {video_path}")
        record_id = f"{protocol}:{target_scope}:{region or 'all_objects'}:{mask_mode}"
        audit = manifest.get("audit") if isinstance(manifest.get("audit"), dict) else {}
        records.append(
            {
                "id": record_id,
                "protocol": protocol,
                "target_scope": target_scope,
                "region": region or None,
                "mask_mode": mask_mode,
                "top_n": 100,
                "path": str(video_path),
                "selected_token_count": len(audit.get("query_token_indices") or []),
            }
        )
    expected = 3 * len(OBJECT_MASK_MODES)
    if len(records) != expected:
        raise RuntimeError(
            f"expected {expected} {protocol} videos, found {len(records)} under {case_root}"
        )
    return records


def public_video_record(record: dict[str, Any], loaded: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "frame_count": loaded["frame_count"],
        "original_shape": loaded["original_shape"],
        "fps": round(loaded["fps"], 6),
        "file_size_bytes": loaded["file_size_bytes"],
        "file_sha256": loaded["file_sha256"],
        "decoded_sha256": loaded["decoded_sha256"],
    }


def pair_record(
    left_id: str,
    right_id: str,
    relation: str,
    loaded: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "left_id": left_id,
        "right_id": right_id,
        "relation": relation,
        **metric_summary(loaded[left_id], loaded[right_id]),
    }


def write_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    if not comparisons:
        return
    fields = list(comparisons[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("comparison width and height must be positive")

    output = args.output or (
        args.tube_root
        / args.case
        / f"seed_{args.seed:05d}"
        / "video_similarity_top100.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    baseline_path = (
        args.baseline_root / args.case / f"seed_{args.seed:05d}" / "generated.mp4"
    )
    if not baseline_path.is_file():
        raise RuntimeError(f"missing baseline video: {baseline_path}")

    videos = [
        {
            "id": "baseline",
            "protocol": "baseline",
            "target_scope": None,
            "region": None,
            "mask_mode": None,
            "top_n": None,
            "path": str(baseline_path),
            "selected_token_count": 0,
        },
        *collect_protocol_videos("fixed", args.fixed_root, args.case, args.seed),
        *collect_protocol_videos("tube", args.tube_root, args.case, args.seed),
    ]

    loaded: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(videos, start=1):
        print(f"[{index:02d}/{len(videos):02d}] decode {record['id']}", flush=True)
        loaded[record["id"]] = load_video(
            Path(record["path"]), (args.width, args.height)
        )

    shape_signatures = {
        (item["frame_count"], tuple(item["original_shape"])) for item in loaded.values()
    }
    if len(shape_signatures) != 1:
        raise RuntimeError(f"video frame signatures differ: {sorted(shape_signatures)}")

    comparison_specs: list[tuple[str, str, str]] = []
    non_baseline = [row for row in videos if row["id"] != "baseline"]
    for record in non_baseline:
        comparison_specs.append(("baseline", record["id"], "vs_baseline"))

    by_protocol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in non_baseline:
        by_protocol[record["protocol"]].append(record)
    for protocol in ("fixed", "tube"):
        rows = sorted(by_protocol[protocol], key=lambda row: row["id"])
        protocol_pairs = list(combinations(rows, 2))
        for left, right in protocol_pairs:
            same_target = (
                left["target_scope"], left["region"]
            ) == (right["target_scope"], right["region"])
            relation = (
                "same_protocol_same_target"
                if same_target
                else "same_protocol_cross_target"
            )
            comparison_specs.append((left["id"], right["id"], relation))

    fixed_by_key = {
        (row["target_scope"], row["region"], row["mask_mode"]): row
        for row in by_protocol["fixed"]
    }
    for tube in by_protocol["tube"]:
        key = (tube["target_scope"], tube["region"], tube["mask_mode"])
        fixed = fixed_by_key[key]
        comparison_specs.append(
            (fixed["id"], tube["id"], "fixed_vs_tube_same_operator")
        )

    print(
        f"compute {len(comparison_specs)} comparisons with {args.workers} workers",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        comparisons = list(
            executor.map(
                lambda spec: pair_record(*spec, loaded),
                comparison_specs,
            )
        )

    decoded_groups: dict[str, list[str]] = defaultdict(list)
    for record in videos:
        decoded_groups[loaded[record["id"]]["decoded_sha256"]].append(record["id"])
    exact_groups = [
        ids for ids in decoded_groups.values() if len(ids) > 1
    ]

    same_target_pairs = [
        row for row in comparisons if row["relation"] == "same_protocol_same_target"
    ]
    closest_same_target = sorted(
        same_target_pairs,
        key=lambda row: (-float(row["ssim_mean"]), float(row["mae_0_1"])),
    )[:30]
    most_changed_vs_baseline = sorted(
        (row for row in comparisons if row["relation"] == "vs_baseline"),
        key=lambda row: (float(row["ssim_mean"]), -float(row["mae_0_1"])),
    )[:20]

    comparison_resolution = [args.height, args.width]
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "seed": args.seed,
        "top_n": 100,
        "video_count": len(videos),
        "ablation_video_count": len(non_baseline),
        "comparison_count": len(comparisons),
        "comparison_resolution_hwc": [*comparison_resolution, 3],
        "metric_definition": {
            "decoded_equal": "SHA-256 equality over all native decoded BGR frames and shape",
            "mae_0_1": "mean absolute RGB-channel error at comparison resolution, divided by 255",
            "psnr_db": "PSNR at comparison resolution; null means exact equality/infinite PSNR",
            "ssim_mean": (
                "mean framewise three-channel SSIM on OpenCV-decoded BGR frames; "
                "equivalent to RGB when both videos use the same channel permutation"
            ),
            "changed_pixel_fraction": "fraction of resized pixels where any channel differs",
            "temporal_delta_mae_0_1": "MAE between consecutive-frame differences, divided by 255",
        },
        "videos": [public_video_record(row, loaded[row["id"]]) for row in videos],
        "comparisons": comparisons,
        "exact_decoded_groups": exact_groups,
        "closest_same_target_pairs": closest_same_target,
        "most_changed_vs_baseline": most_changed_vs_baseline,
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output.with_suffix(".csv"), comparisons)
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.csv')}")
    print(f"exact decoded groups: {len(exact_groups)}")


if __name__ == "__main__":
    main()
