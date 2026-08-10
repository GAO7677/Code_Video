#!/usr/bin/env python3
"""Extract RAFT flow and compare motion in fixed-query and tube ablation videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


DEFAULT_CASE = "0613pybullet_sample_001460_w002"
DEFAULT_SEED = 47326
DEFAULT_EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
)
DEFAULT_WEIGHT = Path(
    "/data/gaoya/agent-data/weights/torch/hub/checkpoints/"
    "raft_large_C_T_SKHT_V2-ff5fadd5.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--flow-updates", type=int, default=12)
    parser.add_argument("--active-motion-threshold", type=float, default=0.25)
    parser.add_argument("--roi-dilate-px", type=int, default=6)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-flow-videos", action="store_true")
    parser.add_argument(
        "--flows-only",
        action="store_true",
        help="extract/cache every inventory RAFT field, then stop before standalone pair analysis",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_id(record_id: str) -> str:
    return record_id.replace(":", "__")


def load_inventory(path: Path, case: str, seed: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case") != case or int(payload.get("seed", -1)) != seed:
        raise RuntimeError("pixel-similarity inventory does not match case/seed")
    videos = payload.get("videos")
    if not isinstance(videos, list) or len(videos) < 2:
        raise RuntimeError(
            f"expected baseline plus at least one inventory video, got {len(videos or [])}"
        )
    ids = [str(row.get("id") or "") for row in videos]
    if ids[0] != "baseline" or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("inventory must start with baseline and contain unique IDs")
    for row in videos:
        video_path = Path(str(row.get("path") or ""))
        if not video_path.is_file():
            raise RuntimeError(f"missing inventory video: {video_path}")
        if sha256_file(video_path) != row.get("file_sha256"):
            raise RuntimeError(f"video changed after pixel metrics: {video_path}")
    return videos


def decode_video(path: Path, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != 49:
        raise RuntimeError(f"expected 49 frames in {path}, got {len(frames)}")
    return np.stack(frames)


def load_model(weights_path: Path, device: torch.device):
    if not weights_path.is_file():
        raise RuntimeError(f"missing RAFT weights: {weights_path}")
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = raft_large(weights=None)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model, Raft_Large_Weights.C_T_SKHT_V2.transforms()


def extract_flow(
    frames: np.ndarray,
    model: torch.nn.Module,
    transforms,
    device: torch.device,
    batch_size: int,
    flow_updates: int,
) -> np.ndarray:
    first = torch.from_numpy(frames[:-1]).permute(0, 3, 1, 2)
    second = torch.from_numpy(frames[1:]).permute(0, 3, 1, 2)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(first), batch_size):
            image1, image2 = transforms(
                first[start : start + batch_size], second[start : start + batch_size]
            )
            prediction = model(
                image1.to(device, non_blocking=True),
                image2.to(device, non_blocking=True),
                num_flow_updates=flow_updates,
            )[-1]
            outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def flow_cache_valid(
    flow_path: Path,
    metadata_path: Path,
    record: dict[str, Any],
    settings: dict[str, Any],
) -> bool:
    if not flow_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        flow = np.load(flow_path, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        metadata.get("video_id") == record["id"]
        and metadata.get("video_file_sha256") == record["file_sha256"]
        and metadata.get("settings") == settings
        and tuple(flow.shape)
        == (48, 2, int(settings["height"]), int(settings["width"]))
        and flow.dtype == np.float16
    )


def save_flow(
    path: Path, metadata_path: Path, flow: np.ndarray, metadata: dict[str, Any]
) -> None:
    temporary = path.with_name(path.stem + ".tmp.npy")
    np.save(temporary, flow.astype(np.float16))
    temporary.replace(path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def flow_to_bgr(flow: np.ndarray, max_magnitude: float) -> np.ndarray:
    u, v = flow[0].astype(np.float32), flow[1].astype(np.float32)
    magnitude, angle = cv2.cartToPolar(u, v, angleInDegrees=False)
    hsv = np.empty((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.mod(angle * (90.0 / math.pi), 180.0).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / max(max_magnitude, 1e-6), 0, 1) * 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def write_flow_video(path: Path, flow: np.ndarray, max_magnitude: float) -> None:
    frames = [
        cv2.cvtColor(flow_to_bgr(frame, max_magnitude), cv2.COLOR_BGR2RGB)
        for frame in flow
    ]
    temporary = path.with_name(path.stem + ".tmp.mp4")
    imageio.mimwrite(
        temporary,
        frames,
        fps=30,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=None,
    )
    temporary.replace(path)


def load_dynamic_rois(
    track_path: Path, width: int, height: int, dilate_px: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(track_path, allow_pickle=False) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        region_names = arrays["region_names"].tolist()
        starts = arrays["point_starts"].astype(int).tolist()
        ends = arrays["point_ends"].astype(int).tolist()
        source_height = int(arrays["pixel_height"])
        source_width = int(arrays["pixel_width"])
    if tracks.shape[0] != 49:
        raise RuntimeError(f"expected 49 tracked frames, got {tracks.shape}")

    kernel_size = 2 * dilate_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    rois = {}
    for name, start, end in zip(region_names, starts, ends, strict=True):
        masks = np.zeros((48, height, width), dtype=bool)
        for frame_index in range(48):
            points = tracks[frame_index, start:end].copy()
            points[:, 0] *= width / source_width
            points[:, 1] *= height / source_height
            points = np.rint(points).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            hull = cv2.convexHull(points.reshape(-1, 1, 2))
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull, 1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            masks[frame_index] = mask.astype(bool)
        rois[str(name)] = masks
    rois["all_objects"] = np.logical_or.reduce(list(rois.values()))
    audit = {
        "definition": (
            "baseline-frozen CoTracker point convex hull at each source frame, "
            f"dilated by {dilate_px}px at {width}x{height}"
        ),
        "track_path": str(track_path),
        "source_resolution": [source_height, source_width],
        "flow_resolution": [height, width],
        "mean_area_pixels": {
            name: round(float(mask.sum(axis=(1, 2)).mean()), 3)
            for name, mask in rois.items()
        },
    }
    return rois, audit


def selected_vectors(flow: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    vectors = np.moveaxis(flow, 0, -1)
    return vectors.reshape(-1, 2) if mask is None else vectors[mask]


def flow_statistics(flow: np.ndarray, mask: np.ndarray | None) -> dict[str, Any]:
    profile = []
    all_magnitude = []
    for frame_index in range(flow.shape[0]):
        vectors = selected_vectors(
            flow[frame_index].astype(np.float32),
            None if mask is None else mask[frame_index],
        )
        magnitude = np.linalg.norm(vectors, axis=1)
        profile.append(float(magnitude.mean()))
        all_magnitude.append(magnitude)
    magnitude = np.concatenate(all_magnitude)
    return {
        "mean_magnitude_px": round(float(magnitude.mean()), 6),
        "median_magnitude_px": round(float(np.median(magnitude)), 6),
        "p95_magnitude_px": round(float(np.percentile(magnitude, 95)), 6),
        "max_frame_mean_magnitude_px": round(float(max(profile)), 6),
        "motion_profile_mean_magnitude_px": [round(value, 6) for value in profile],
    }


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 2 or left.std() < 1e-9 or right.std() < 1e-9:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def compare_flow(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray | None,
    active_threshold: float,
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise RuntimeError(f"flow shapes differ: {reference.shape} vs {candidate.shape}")
    epe_sum = magnitude_error_sum = reference_magnitude_sum = candidate_magnitude_sum = 0.0
    dot_sum = reference_energy = candidate_energy = 0.0
    direction_cosine_sum = 0.0
    count = active_count = 0
    frame_epe = []
    reference_profile = []
    candidate_profile = []

    for frame_index in range(reference.shape[0]):
        ref = selected_vectors(
            reference[frame_index].astype(np.float32),
            None if mask is None else mask[frame_index],
        )
        cand = selected_vectors(
            candidate[frame_index].astype(np.float32),
            None if mask is None else mask[frame_index],
        )
        ref_mag = np.linalg.norm(ref, axis=1)
        cand_mag = np.linalg.norm(cand, axis=1)
        epe = np.linalg.norm(ref - cand, axis=1)
        frame_epe.append(float(epe.mean()))
        reference_profile.append(float(ref_mag.mean()))
        candidate_profile.append(float(cand_mag.mean()))
        epe_sum += float(epe.sum(dtype=np.float64))
        magnitude_error_sum += float(np.abs(ref_mag - cand_mag).sum(dtype=np.float64))
        reference_magnitude_sum += float(ref_mag.sum(dtype=np.float64))
        candidate_magnitude_sum += float(cand_mag.sum(dtype=np.float64))
        dot_sum += float(np.sum(ref * cand, dtype=np.float64))
        reference_energy += float(np.sum(ref * ref, dtype=np.float64))
        candidate_energy += float(np.sum(cand * cand, dtype=np.float64))
        active = (ref_mag >= active_threshold) & (cand_mag >= active_threshold)
        if np.any(active):
            cosine = np.sum(ref[active] * cand[active], axis=1) / (
                ref_mag[active] * cand_mag[active] + 1e-12
            )
            direction_cosine_sum += float(cosine.sum(dtype=np.float64))
            active_count += int(active.sum())
        count += len(epe)

    reference_profile_array = np.asarray(reference_profile, dtype=np.float64)
    candidate_profile_array = np.asarray(candidate_profile, dtype=np.float64)
    vector_cosine = dot_sum / math.sqrt(
        max(reference_energy * candidate_energy, 1e-24)
    )
    mean_reference_magnitude = reference_magnitude_sum / count
    mean_candidate_magnitude = candidate_magnitude_sum / count
    return {
        "flow_epe_mean_px": round(epe_sum / count, 6),
        "flow_epe_frame_p95_px": round(float(np.percentile(frame_epe, 95)), 6),
        "flow_epe_over_reference_magnitude": (
            None
            if mean_reference_magnitude < 1e-12
            else round((epe_sum / count) / mean_reference_magnitude, 9)
        ),
        "flow_vector_cosine": round(float(vector_cosine), 9),
        "active_direction_cosine": (
            None
            if active_count == 0
            else round(direction_cosine_sum / active_count, 9)
        ),
        "active_pixel_fraction": round(active_count / count, 9),
        "magnitude_mae_px": round(magnitude_error_sum / count, 6),
        "reference_mean_magnitude_px": round(mean_reference_magnitude, 6),
        "candidate_mean_magnitude_px": round(mean_candidate_magnitude, 6),
        "motion_magnitude_ratio": (
            None
            if mean_reference_magnitude < 1e-12
            else round(mean_candidate_magnitude / mean_reference_magnitude, 9)
        ),
        "motion_profile_correlation": (
            None
            if (correlation := safe_corr(reference_profile_array, candidate_profile_array))
            is None
            else round(correlation, 9)
        ),
        "per_frame_flow_epe_mean_px": [round(value, 6) for value in frame_epe],
    }


def scope_masks(rois: dict[str, np.ndarray]) -> dict[str, np.ndarray | None]:
    return {
        "global": None,
        "object_A_roi": rois["object_A"],
        "object_B_roi": rois["object_B"],
        "all_objects_roi": rois["all_objects"],
    }


def compare_all_scopes(
    left: np.ndarray,
    right: np.ndarray,
    masks: dict[str, np.ndarray | None],
    threshold: float,
) -> dict[str, Any]:
    return {
        name: compare_flow(left, right, mask, threshold)
        for name, mask in masks.items()
    }


def write_comparison_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    rows = []
    for comparison in comparisons:
        for scope, metrics in comparison["scopes"].items():
            rows.append(
                {
                    "left_id": comparison["left_id"],
                    "right_id": comparison["right_id"],
                    "relation": comparison["relation"],
                    "scope": scope,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, list)
                    },
                }
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.width % 8 or args.height % 8:
        raise ValueError("RAFT width and height must be divisible by 8")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device.startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {
        ":4096:8",
        ":16:8",
    }:
        raise RuntimeError(
            "deterministic CUDA RAFT requires CUBLAS_WORKSPACE_CONFIG=:4096:8"
        )

    case_root = (
        args.inventory.expanduser().resolve().parent
        if args.inventory is not None
        else DEFAULT_EXPERIMENT_ROOT / args.case / f"seed_{args.seed:05d}"
    )
    inventory_path = args.inventory or case_root / "video_similarity_top100.json"
    output_root = args.output_root or case_root / "raft_motion_top100_v1"
    flow_root = output_root / "flows"
    flow_video_root = output_root / "flow_videos"
    flow_root.mkdir(parents=True, exist_ok=True)
    flow_video_root.mkdir(parents=True, exist_ok=True)
    output_json = output_root / "raft_motion_similarity_top100.json"

    videos = load_inventory(inventory_path, args.case, args.seed)
    track_path = case_root / "frozen_baseline_tracks" / "tracks.npz"
    if not track_path.is_file():
        raise RuntimeError(f"missing frozen baseline tracks: {track_path}")

    device = torch.device(args.device)
    checkpoint_sha256 = sha256_file(args.weights)
    settings = {
        "model": "torchvision.models.optical_flow.raft_large",
        "weights": "Raft_Large_Weights.C_T_SKHT_V2",
        "weights_path": str(args.weights),
        "weights_sha256": checkpoint_sha256,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "width": args.width,
        "height": args.height,
        "flow_updates": args.flow_updates,
        "inference_dtype": "float32",
        "cache_dtype": "float16",
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "frame_pairs": "F00->F01, ..., F47->F48",
    }
    rois, roi_audit = load_dynamic_rois(
        track_path, args.width, args.height, args.roi_dilate_px
    )
    masks = scope_masks(rois)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = None
    transforms = None

    flow_paths: dict[str, Path] = {}
    extraction_records = []
    for index, record in enumerate(videos, start=1):
        stem = safe_id(str(record["id"]))
        flow_path = flow_root / f"{stem}.npy"
        metadata_path = flow_root / f"{stem}.json"
        flow_paths[str(record["id"])] = flow_path
        cached = not args.overwrite and flow_cache_valid(
            flow_path, metadata_path, record, settings
        )
        started = time.monotonic()
        if cached:
            print(f"[{index:02d}/{len(videos):02d}] cached {record['id']}", flush=True)
        else:
            print(f"[{index:02d}/{len(videos):02d}] RAFT {record['id']}", flush=True)
            if model is None or transforms is None:
                model, transforms = load_model(args.weights, device)
            frames = decode_video(Path(record["path"]), args.width, args.height)
            flow = extract_flow(
                frames,
                model,
                transforms,
                device,
                args.batch_size,
                args.flow_updates,
            )
            metadata = {
                "video_id": record["id"],
                "video_path": record["path"],
                "video_file_sha256": record["file_sha256"],
                "settings": settings,
                "flow_shape": list(flow.shape),
                "flow_definition": "forward RAFT optical flow from frame t to t+1",
            }
            save_flow(flow_path, metadata_path, flow, metadata)
        extraction_records.append(
            {
                "id": record["id"],
                "flow_path": str(flow_path),
                "cache_hit": cached,
                "seconds": round(time.monotonic() - started, 3),
            }
        )

    if model is not None:
        del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.flows_only:
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "case": args.case,
            "seed": args.seed,
            "video_count": len(videos),
            "settings": settings,
            "roi_audit": roi_audit,
            "extraction": extraction_records,
            "note": "flows-only cache; final report computes required comparisons directly",
        }
        (output_root / "flows_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {output_root / 'flows_manifest.json'}")
        return

    baseline_flow = np.load(flow_paths["baseline"], mmap_mode="r")
    baseline_magnitude = np.sqrt(
        np.square(baseline_flow[:, 0].astype(np.float32))
        + np.square(baseline_flow[:, 1].astype(np.float32))
    )
    visual_max_magnitude = float(np.percentile(baseline_magnitude, 99.5))
    del baseline_magnitude

    if not args.skip_flow_videos:
        for index, record in enumerate(videos, start=1):
            video_path = flow_video_root / f"{safe_id(str(record['id']))}.mp4"
            if args.overwrite or not video_path.is_file():
                print(f"[{index:02d}/{len(videos):02d}] encode flow {record['id']}", flush=True)
                write_flow_video(
                    video_path,
                    np.load(flow_paths[str(record["id"])], mmap_mode="r"),
                    visual_max_magnitude,
                )
            record["raft_flow_video"] = str(video_path)

    standalone = {}
    for index, record in enumerate(videos, start=1):
        print(f"[{index:02d}/{len(videos):02d}] summarize {record['id']}", flush=True)
        flow = np.load(flow_paths[str(record["id"])], mmap_mode="r")
        standalone[str(record["id"])] = {
            scope: flow_statistics(flow, mask) for scope, mask in masks.items()
        }

    non_baseline = [row for row in videos if row["id"] != "baseline"]
    by_protocol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in non_baseline:
        by_protocol[str(record["protocol"])].append(record)

    comparison_specs: list[tuple[str, str, str]] = [
        ("baseline", str(record["id"]), "vs_baseline") for record in non_baseline
    ]
    fixed_by_key = {
        (row["target_scope"], row["region"], row["mask_mode"]): row
        for row in by_protocol["fixed"]
    }
    for tube in by_protocol["tube"]:
        key = (tube["target_scope"], tube["region"], tube["mask_mode"])
        comparison_specs.append(
            (
                str(fixed_by_key[key]["id"]),
                str(tube["id"]),
                "fixed_vs_tube_same_operator",
            )
        )
    for protocol in ("fixed", "tube"):
        grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
        for row in by_protocol[protocol]:
            grouped[(row["target_scope"], row["region"])].append(row)
        for rows in grouped.values():
            for left, right in combinations(sorted(rows, key=lambda row: row["id"]), 2):
                comparison_specs.append(
                    (str(left["id"]), str(right["id"]), "same_protocol_same_target")
                )

    comparisons = []
    for index, (left_id, right_id, relation) in enumerate(comparison_specs, start=1):
        print(
            f"[{index:03d}/{len(comparison_specs):03d}] compare {left_id} <> {right_id}",
            flush=True,
        )
        left = np.load(flow_paths[left_id], mmap_mode="r")
        right = np.load(flow_paths[right_id], mmap_mode="r")
        comparisons.append(
            {
                "left_id": left_id,
                "right_id": right_id,
                "relation": relation,
                "scopes": compare_all_scopes(
                    left, right, masks, args.active_motion_threshold
                ),
            }
        )

    pixel_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    pixel_vs_baseline = {
        str(row["right_id"]): row
        for row in pixel_payload.get("comparisons", [])
        if row.get("relation") == "vs_baseline"
    }
    video_by_id = {str(row["id"]): row for row in videos}
    pixel_ssim, flow_epe, flow_cosine, motion_profile = [], [], [], []
    for comparison in comparisons:
        if comparison["relation"] != "vs_baseline":
            continue
        right_id = str(comparison["right_id"])
        video = video_by_id[right_id]
        scope = (
            "all_objects_roi"
            if video.get("target_scope") == "all_objects"
            else f"{video.get('region')}_roi"
        )
        motion = comparison["scopes"][scope]
        pixel = pixel_vs_baseline[right_id]
        pixel_ssim.append(float(pixel["ssim_mean"]))
        flow_epe.append(float(motion["flow_epe_mean_px"]))
        flow_cosine.append(float(motion["flow_vector_cosine"]))
        motion_profile.append(float(motion["motion_profile_correlation"]))
    pixel_ssim_array = np.asarray(pixel_ssim, dtype=np.float64)

    def correlation_with_pixel(values: list[float]) -> dict[str, float | None]:
        value_array = np.asarray(values, dtype=np.float64)
        pearson = safe_corr(pixel_ssim_array, value_array)
        spearman = safe_corr(
            average_ranks(pixel_ssim_array), average_ranks(value_array)
        )
        return {
            "pearson": None if pearson is None else round(pearson, 9),
            "spearman": None if spearman is None else round(spearman, 9),
        }

    pixel_motion_correlation = {
        "sample_count": len(pixel_ssim),
        "scope": "each experiment's own baseline-frozen target ROI",
        "pixel_ssim_vs_flow_epe": correlation_with_pixel(flow_epe),
        "pixel_ssim_vs_flow_vector_cosine": correlation_with_pixel(flow_cosine),
        "pixel_ssim_vs_motion_profile_correlation": correlation_with_pixel(
            motion_profile
        ),
    }

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": args.case,
        "seed": args.seed,
        "top_n": 100,
        "video_count": len(videos),
        "comparison_count": len(comparisons),
        "settings": settings,
        "active_motion_threshold_px": args.active_motion_threshold,
        "roi_audit": roi_audit,
        "flow_visualization": {
            "encoding": "HSV color wheel; hue=direction, value=magnitude",
            "shared_max_magnitude_px": round(visual_max_magnitude, 6),
            "scale_source": "baseline global 99.5th percentile",
        },
        "metric_definition": {
            "flow_epe_mean_px": (
                "mean endpoint distance between two estimated RAFT fields; "
                "cross-video disagreement, not ground-truth optical-flow EPE"
            ),
            "flow_epe_over_reference_magnitude": (
                "mean cross-video flow EPE divided by reference mean flow magnitude; "
                "useful across ROIs, but unstable when reference motion is near zero"
            ),
            "flow_vector_cosine": "magnitude-weighted cosine of the complete flow fields",
            "active_direction_cosine": (
                "mean direction cosine where both flow magnitudes exceed the active threshold"
            ),
            "magnitude_mae_px": "mean absolute difference between flow magnitudes",
            "motion_magnitude_ratio": "candidate mean flow magnitude / reference mean flow magnitude",
            "motion_profile_correlation": (
                "Pearson correlation between 48-frame mean-magnitude profiles"
            ),
        },
        "videos": [
            {
                **row,
                "flow_path": str(flow_paths[str(row["id"])]),
                "motion": standalone[str(row["id"])],
            }
            for row in videos
        ],
        "comparisons": comparisons,
        "analysis": {"pixel_motion_correlation": pixel_motion_correlation},
        "extraction": extraction_records,
    }
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_comparison_csv(output_root / "raft_motion_similarity_top100.csv", comparisons)
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
