#!/usr/bin/env python3
"""Measure target-object retention for Head-Scope ablation videos.

The metric uses SAM2 mask propagation plus DINOv2 identity and mask-area
checks.  It is deliberately separate from CoTracker Track Loss: a tracker can
lose an object that remains visible, while SAM2 can hallucinate a mask after an
object disappears.  Requiring both identity and plausible area makes the
failure interpretation auditable.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_baseline_metrics import (  # noqa: E402
    HEAD_SCOPES,
    atomic_json,
    collect_candidates,
    file_signature,
)
from AAA_my_test.object_query_ablation_metrics.compute_head_scope_trajectory_metrics import (  # noqa: E402
    DEFAULT_OUTPUT_BASE,
    FRAME_COUNT,
    atomic_npz,
    load_video_frames,
    locate_baseline,
    resolve_frozen_baseline_inputs,
    rounded,
    signature_text,
)
from AAA_my_test.object_query_ablation_metrics.compute_perceptual import (  # noqa: E402
    centered_crop,
    load_dino,
)
from AAA_my_test.object_query_ablation_metrics.extract_masks import (  # noqa: E402
    SAM2_CHECKPOINT,
    SAM2_CONFIG,
    track_masks,
)


OBJECT_SURVIVAL_DEFINITION = {
    "name": "SAM2+DINOv2 target-object retention",
    "reference": "same-seed no-intervention Baseline",
    "frame_alive": (
        "SAM2 mask is nonempty AND DINOv2 mask-pooled cosine to the same-frame "
        "Baseline object is at least the calibrated identity threshold AND "
        "candidate/Baseline mask-area ratio is in [0.25,4.0]"
    ),
    "object_survival_rate": "mean_t 1[frame_alive(t)]",
    "object_disappearance_score": "100 * (1 - object_survival_rate)",
    "object_mask_absence_score": "100 * mean_t 1[SAM2 mask is empty]",
    "target_mean_score": "mean_selected_objects(object_disappearance_score)",
    "target_worst_score": "max_selected_objects(object_disappearance_score)",
    "ranking_score": "target_worst_disappearance_score_0_100",
    "mask_absence_ranking_score": "target_worst_mask_absence_score_0_100",
    "identity_threshold": (
        "per-object midpoint between Baseline same-object temporal cosine Q05 "
        "and Baseline cross-object cosine Q95 when separated; otherwise the "
        "same-object temporal Q05"
    ),
    "initialization_quality_gate": "SAM2 F00 prompt IoU >= 0.50",
    "sustained_loss": "first run of at least three consecutive not-alive frames",
    "terminal_window": "last eight frames",
    "interpretation": (
        "larger means weaker target-object retention versus Baseline; the score "
        "includes disappearance, identity corruption, and extreme mask-area change"
    ),
}

AREA_RATIO_MIN = 0.25
AREA_RATIO_MAX = 4.0
MIN_F00_IOU = 0.50
SUSTAINED_LOSS_FRAMES = 3
TERMINAL_FRAMES = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--head-scopes", nargs="+", choices=HEAD_SCOPES, default=list(HEAD_SCOPES)
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-overlays", action="store_true")
    return parser.parse_args()


def pack_masks(masks: np.ndarray) -> np.ndarray:
    return np.packbits(masks.astype(bool).reshape(-1))


def unpack_masks(packed: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape))
    return np.unpackbits(packed)[:count].reshape(shape).astype(bool)


def load_mask_cache(
    path: Path, signature: dict[str, int], object_count: int
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            shape = tuple(int(value) for value in arrays["mask_shape"].tolist())
            cached_signature = str(arrays["video_signature"].item())
            packed = arrays["masks_packed"]
        if (
            shape != (FRAME_COUNT, object_count, 704, 1280)
            or cached_signature != signature_text(signature)
        ):
            return None
        return unpack_masks(packed, shape)
    except (OSError, KeyError, ValueError):
        return None


def save_mask_cache(
    path: Path,
    masks: np.ndarray,
    signature: dict[str, int],
    video_path: Path,
) -> None:
    atomic_npz(
        path,
        masks_packed=pack_masks(masks),
        mask_shape=np.asarray(masks.shape, dtype=np.int32),
        video_signature=np.asarray(signature_text(signature)),
        video_path=np.asarray(str(video_path)),
        segmenter=np.asarray("SAM2.1 Hiera Large video predictor"),
    )


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def centers_from_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    object_slices: dict[str, slice],
) -> dict[str, np.ndarray]:
    centers = {}
    for name, part in object_slices.items():
        values = np.full((FRAME_COUNT, 2), np.nan, dtype=np.float32)
        for frame_index in range(FRAME_COUNT):
            use = visibility[frame_index, part] & np.isfinite(
                tracks[frame_index, part]
            ).all(axis=1)
            if use.any():
                values[frame_index] = np.median(
                    tracks[frame_index, part][use], axis=0
                )
        valid = np.where(np.isfinite(values).all(axis=1))[0]
        if not len(valid):
            raise RuntimeError(f"no finite track centers for {name}")
        first = int(valid[0])
        values[:first] = values[first]
        for frame_index in range(first + 1, FRAME_COUNT):
            if not np.isfinite(values[frame_index]).all():
                values[frame_index] = values[frame_index - 1]
        centers[name] = values
    return centers


def crop_side_from_masks(masks: np.ndarray) -> int:
    sides = []
    for mask in masks:
        y, x = np.where(mask)
        if len(x):
            sides.append(max(int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)))
    if not sides:
        raise RuntimeError("Baseline SAM2 masks are empty")
    return int(np.clip(math.ceil(np.percentile(sides, 99) * 1.5 / 14) * 14, 98, 448))


def object_crops(
    frames: np.ndarray,
    masks: np.ndarray,
    centers: np.ndarray,
    side: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        centered_crop(frame, mask, side, center)
        for frame, mask, center in zip(frames, masks, centers, strict=True)
    ]
    return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])


def dino_pooled_features(
    model,
    crops: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            images = torch.from_numpy(crops[start : start + batch_size]).to(device)
            images = images.float().permute(0, 3, 1, 2).div(255.0)
            features = model.forward_features((images - mean) / std)[
                "x_norm_patchtokens"
            ]
            patch_mask = torch.from_numpy(masks[start : start + batch_size]).to(
                device
            ).float()[:, None]
            patch_mask = F.interpolate(
                patch_mask, size=(16, 16), mode="area"
            ).flatten(1)
            pooled = (features * patch_mask[..., None]).sum(1)
            pooled = pooled / patch_mask.sum(1, keepdim=True).clamp_min(1e-6)
            outputs.append(F.normalize(pooled, dim=-1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


def load_feature_cache(
    path: Path, signature: dict[str, int], object_count: int
) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as arrays:
            pooled = arrays["pooled_features"].astype(np.float32)
            cached_signature = str(arrays["video_signature"].item())
        if (
            pooled.shape[:2] != (FRAME_COUNT, object_count)
            or cached_signature != signature_text(signature)
            or not np.isfinite(pooled).all()
        ):
            return None
        return pooled
    except (OSError, KeyError, ValueError):
        return None


def calibrate_identity_thresholds(
    baseline_features: np.ndarray, object_names: list[str]
) -> dict[str, dict[str, Any]]:
    if not object_names:
        raise RuntimeError("identity calibration requires at least one object")
    result = {}
    for object_index, name in enumerate(object_names):
        values = []
        features = baseline_features[:, object_index]
        for lag in range(1, 5):
            values.extend(np.sum(features[:-lag] * features[lag:], axis=-1).tolist())
        positive_q05 = float(np.quantile(values, 0.05))
        negative_values = []
        for other_index in range(len(object_names)):
            if other_index == object_index:
                continue
            negative_values.extend(
                np.sum(
                    features * baseline_features[:, other_index], axis=-1
                ).tolist()
            )
        negative_q95 = (
            float(np.quantile(negative_values, 0.95))
            if negative_values
            else None
        )
        separated = negative_q95 is not None and positive_q05 > negative_q95
        threshold = (
            0.5 * (positive_q05 + negative_q95) if separated else positive_q05
        )
        result[name] = {
            "threshold": rounded(threshold),
            "baseline_temporal_positive_q05": rounded(positive_q05),
            "baseline_cross_object_negative_q95": rounded(negative_q95),
            "positive_negative_separated": bool(separated),
            "calibration_mode": (
                "temporal_positive_and_cross_object_negative"
                if negative_q95 is not None
                else "temporal_positive_only_single_object"
            ),
        }
    return result


def first_sustained_false(values: np.ndarray, run_length: int) -> int | None:
    for start in range(0, len(values) - run_length + 1):
        if not values[start : start + run_length].any():
            return start
    return None


def object_survival_metrics(
    candidate_masks: np.ndarray,
    baseline_masks: np.ndarray,
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    prompt_mask: np.ndarray,
    identity_threshold: float,
) -> dict[str, Any]:
    candidate_area = candidate_masks.sum(axis=(1, 2)).astype(np.float64)
    baseline_area = baseline_masks.sum(axis=(1, 2)).astype(np.float64)
    area_ratio = candidate_area / np.maximum(baseline_area, 1.0)
    identity = np.sum(candidate_features * baseline_features, axis=-1)
    mask_nonempty = candidate_area > 0
    identity_pass = identity >= identity_threshold
    area_pass = (area_ratio >= AREA_RATIO_MIN) & (area_ratio <= AREA_RATIO_MAX)
    alive = mask_nonempty & identity_pass & area_pass
    f00_iou = mask_iou(candidate_masks[0], prompt_mask)
    quality_pass = f00_iou >= MIN_F00_IOU
    survival_rate = float(alive.mean())
    disappearance = 100.0 * (1.0 - survival_rate)
    first_loss = first_sustained_false(alive, SUSTAINED_LOSS_FRAMES)
    return {
        "quality_pass": bool(quality_pass),
        "f00_prompt_iou": rounded(f00_iou),
        "survival_rate": rounded(survival_rate),
        "retention_score_0_100": rounded(100.0 * survival_rate),
        "disappearance_score_0_100": rounded(disappearance),
        "identity_similarity_mean": rounded(float(identity.mean())),
        "identity_failure_rate": rounded(float((~identity_pass).mean())),
        "area_failure_rate": rounded(float((~area_pass).mean())),
        "empty_mask_rate": rounded(float((~mask_nonempty).mean())),
        "first_sustained_loss_frame": first_loss,
        "terminal_missing_rate": rounded(float((~alive[-TERMINAL_FRAMES:]).mean())),
        "alive_frame_count": int(alive.sum()),
        "frame_count": int(len(alive)),
        "series": {
            "alive": alive.astype(bool).tolist(),
            "identity_similarity": [rounded(float(value), 6) for value in identity],
            "area_ratio_vs_baseline": [rounded(float(value), 6) for value in area_ratio],
        },
    }


def draw_mask_panel(
    frames: np.ndarray,
    masks: np.ndarray,
    object_names: list[str],
    metrics: dict[str, dict[str, Any]] | None,
    frame_index: int,
    title: str,
) -> np.ndarray:
    canvas = cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR)
    base_colors = (
        (36, 170, 245),
        (232, 188, 42),
        (88, 205, 102),
        (205, 95, 210),
        (72, 205, 225),
        (225, 120, 72),
    )
    for object_index, name in enumerate(object_names):
        alive = True if metrics is None else bool(metrics[name]["series"]["alive"][frame_index])
        color = (
            base_colors[object_index % len(base_colors)]
            if metrics is None
            else ((55, 190, 85) if alive else (45, 45, 230))
        )
        contours, _ = cv2.findContours(
            masks[frame_index, object_index].astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, color, 3, cv2.LINE_AA)
        if metrics is not None:
            identity = metrics[name]["series"]["identity_similarity"][frame_index]
            ratio = metrics[name]["series"]["area_ratio_vs_baseline"][frame_index]
            cv2.putText(
                canvas,
                f"{name}: {'ALIVE' if alive else 'LOSS'} id={identity:.3f} area={ratio:.2f}",
                (12, 56 + object_index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        f"{title} | F{frame_index:02d}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cv2.resize(canvas, (640, 352), interpolation=cv2.INTER_AREA)


def render_overlay(
    path: Path,
    variant_id: str,
    baseline_frames: np.ndarray,
    baseline_masks: np.ndarray,
    candidate_frames: np.ndarray,
    candidate_masks: np.ndarray,
    object_names: list[str],
    object_metrics: dict[str, dict[str, Any]],
    target_score: float | None,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    with imageio.get_writer(
        temporary, fps=fps, codec="libx264", quality=7, macro_block_size=None
    ) as writer:
        for frame_index in range(FRAME_COUNT):
            left = draw_mask_panel(
                baseline_frames,
                baseline_masks,
                object_names,
                None,
                frame_index,
                "Baseline",
            )
            right = draw_mask_panel(
                candidate_frames,
                candidate_masks,
                object_names,
                object_metrics,
                frame_index,
                "Ablation",
            )
            body = np.concatenate((left, right), axis=1)
            header = np.full((48, body.shape[1], 3), (41, 49, 75), dtype=np.uint8)
            score = "N/A" if target_score is None else f"{target_score:.2f}"
            cv2.putText(
                header,
                f"{variant_id} | worst target disappearance={score} | green=alive red=loss",
                (12, 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.54,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
            writer.append_data(
                cv2.cvtColor(np.concatenate((header, body), axis=0), cv2.COLOR_BGR2RGB)
            )
    temporary.replace(path)


def rank_records(records: list[dict[str, Any]]) -> None:
    for score_field, rank_field in (
        (
            "target_worst_disappearance_score_0_100",
            "disappearance_rank_within_case_seed",
        ),
        (
            "target_worst_mask_absence_score_0_100",
            "mask_absence_rank_within_case_seed",
        ),
    ):
        valid = [
            row for row in records if row["metrics"].get(score_field) is not None
        ]
        valid.sort(
            key=lambda row: (
                -float(row["metrics"][score_field]),
                row["variant_id"],
            )
        )
        for row in records:
            row[rank_field] = None
        for rank, row in enumerate(valid, start=1):
            row[rank_field] = rank


def report_payload(
    case: str,
    seed: int,
    baseline_path: Path,
    candidates: list[dict[str, Any]],
    records: list[dict[str, Any]],
    calibration: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rank_records(records)
    records.sort(
        key=lambda row: (
            row["disappearance_rank_within_case_seed"] is None,
            row["disappearance_rank_within_case_seed"] or 10**9,
            row["variant_id"],
        )
    )
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "seed": seed,
        "reference": "same-seed no-intervention Baseline",
        "object_survival_definition": OBJECT_SURVIVAL_DEFINITION,
        "identity_calibration": calibration,
        "baseline_path": str(baseline_path),
        "expected_ablation_count": len(candidates),
        "measured_ablation_count": len(records),
        "ranked_ablation_count": sum(
            row["disappearance_rank_within_case_seed"] is not None for row in records
        ),
        "mask_absence_ranked_ablation_count": sum(
            row["mask_absence_rank_within_case_seed"] is not None for row in records
        ),
        "records": records,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    seed_dir = args.result_dir.expanduser().resolve()
    all_candidates = collect_candidates(seed_dir, set(args.head_scopes))
    if not all_candidates:
        raise RuntimeError(f"no completed Head-Scope candidates: {seed_dir}")
    case = str(all_candidates[0]["case"])
    seed = int(all_candidates[0]["seed"])
    if any(row["case"] != case or int(row["seed"]) != seed for row in all_candidates):
        raise RuntimeError("object-survival mode accepts exactly one case/seed")
    baseline_path = locate_baseline(case, seed)
    baseline_signature = file_signature(baseline_path)

    frozen_path, frozen_manifest_path = resolve_frozen_baseline_inputs(seed_dir)
    with np.load(frozen_path, allow_pickle=False) as arrays:
        baseline_tracks = arrays["tracks"].astype(np.float32)
        baseline_visibility = arrays["visibility"].astype(bool)
        query_points = arrays["query_points"].astype(np.float32)
        object_names = [str(value) for value in arrays["region_names"].tolist()]
        starts = arrays["point_starts"].astype(int).tolist()
        ends = arrays["point_ends"].astype(int).tolist()
    object_slices = {
        name: slice(start, end)
        for name, start, end in zip(object_names, starts, ends, strict=True)
    }
    frozen_manifest = json.loads(
        frozen_manifest_path.read_text(encoding="utf-8")
    )
    region_cache = Path(str(frozen_manifest["query_cache_dir"])) / "regions.npz"
    with np.load(region_cache, allow_pickle=False) as arrays:
        prompt_masks = arrays["masks_rhw"].astype(bool)[: len(object_names)]

    output_root = args.output_base.expanduser().resolve() / case / f"seed_{seed:05d}"
    if str(output_root).startswith("/home/gaoya/"):
        raise RuntimeError("large survival artifacts may not be stored under /home/gaoya")
    requested_candidates = (
        all_candidates[: args.limit] if args.limit > 0 else all_candidates
    )
    candidates = [
        candidate
        for candidate in requested_candidates
        if (output_root / "tracks" / f"{candidate['variant_id']}.npz").is_file()
    ]
    deferred = len(requested_candidates) - len(candidates)
    if deferred:
        print(
            f"[defer] {deferred} candidates have no trajectory cache yet; "
            "a later incremental run will measure them",
            flush=True,
        )
    if not candidates:
        raise RuntimeError(
            f"no candidates have trajectory caches yet: {output_root / 'tracks'}"
        )
    survival_root = output_root / "object_survival"
    mask_root = survival_root / "masks"
    feature_root = survival_root / "features"
    overlay_root = survival_root / "overlays"
    temporary_root = survival_root / "tmp"
    for directory in (mask_root, feature_root, overlay_root, temporary_root):
        directory.mkdir(parents=True, exist_ok=True)

    videos = [
        {
            "variant_id": "baseline",
            "path": str(baseline_path),
            "video_signature": baseline_signature,
        },
        *candidates,
    ]
    predictor = None
    try:
        for index, video in enumerate(videos, start=1):
            variant = str(video["variant_id"])
            path = Path(str(video["path"]))
            signature = video["video_signature"]
            mask_path = mask_root / f"{variant}.npz"
            masks = None if args.overwrite else load_mask_cache(
                mask_path, signature, len(object_names)
            )
            if masks is None:
                if predictor is None:
                    from sam2.build_sam import build_sam2_video_predictor

                    predictor = build_sam2_video_predictor(
                        SAM2_CONFIG, str(SAM2_CHECKPOINT), device=args.device
                    )
                    predictor.fill_hole_area = 0
                frames, _fps = load_video_frames(path)
                masks = track_masks(
                    predictor,
                    frames,
                    query_points,
                    prompt_masks,
                    temporary_root,
                    list(object_slices.values()),
                ).astype(bool)
                save_mask_cache(mask_path, masks, signature, path)
                state = "segmented"
                del frames
            else:
                state = "reused-mask"
            print(f"[mask {index:03d}/{len(videos):03d}] {state} {variant}", flush=True)
            del masks
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if predictor is not None:
            del predictor
            gc.collect()
            torch.cuda.empty_cache()

    baseline_frames, baseline_fps = load_video_frames(baseline_path)
    baseline_masks = load_mask_cache(
        mask_root / "baseline.npz", baseline_signature, len(object_names)
    )
    if baseline_masks is None:
        raise RuntimeError("Baseline mask cache disappeared")
    baseline_centers = centers_from_tracks(
        baseline_tracks, baseline_visibility, object_slices
    )
    crop_sides = {
        name: crop_side_from_masks(baseline_masks[:, index])
        for index, name in enumerate(object_names)
    }

    device = torch.device(args.device)
    dino = None

    def pooled_features(
        variant: str,
        video_path: Path,
        signature: dict[str, int],
        frames: np.ndarray,
        masks: np.ndarray,
        centers: dict[str, np.ndarray],
    ) -> np.ndarray:
        nonlocal dino
        feature_path = feature_root / f"{variant}.npz"
        cached = None if args.overwrite else load_feature_cache(
            feature_path, signature, len(object_names)
        )
        if cached is not None:
            return cached
        if dino is None:
            dino = load_dino(device)
        features = []
        for object_index, name in enumerate(object_names):
            crops, local_masks = object_crops(
                frames,
                masks[:, object_index],
                centers[name],
                crop_sides[name],
            )
            features.append(
                dino_pooled_features(
                    dino, crops, local_masks, device, args.batch_size
                )
            )
        pooled = np.stack(features, axis=1)
        atomic_npz(
            feature_path,
            pooled_features=pooled.astype(np.float32),
            video_signature=np.asarray(signature_text(signature)),
            video_path=np.asarray(str(video_path)),
            extractor=np.asarray("official DINOv2 ViT-L/14 mask-pooled patch tokens"),
        )
        return pooled

    baseline_features = pooled_features(
        "baseline",
        baseline_path,
        baseline_signature,
        baseline_frames,
        baseline_masks,
        baseline_centers,
    )
    calibration = calibrate_identity_thresholds(baseline_features, object_names)
    report_path = output_root / "object_survival_report.json"
    existing_records = {}
    if report_path.is_file() and not args.overwrite:
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            existing_records = {
                str(row["variant_id"]): row
                for row in payload.get("records", [])
                if row.get("video_signature")
                == next(
                    (
                        candidate["video_signature"]
                        for candidate in all_candidates
                        if str(candidate["variant_id"]) == str(row["variant_id"])
                    ),
                    None,
                )
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_records = {}
    records: list[dict[str, Any]] = []
    allowed_variants = {str(row["variant_id"]) for row in all_candidates}
    try:
        for index, candidate in enumerate(candidates, start=1):
            variant = str(candidate["variant_id"])
            video_path = Path(str(candidate["path"]))
            signature = candidate["video_signature"]
            previous = existing_records.get(variant)
            previous_metrics = previous.get("metrics", {}) if previous else {}
            previous_complete = bool(
                previous
                and previous.get("video_signature") == signature
                and isinstance(previous_metrics.get("objects"), dict)
                and "quality_pass" in previous_metrics
                and Path(str(previous.get("mask_path") or "")).is_file()
                and Path(str(previous.get("feature_path") or "")).is_file()
                and (
                    args.skip_overlays
                    or Path(str(previous.get("overlay_path") or "")).is_file()
                )
                and load_mask_cache(
                    Path(str(previous.get("mask_path"))), signature, len(object_names)
                )
                is not None
                and load_feature_cache(
                    Path(str(previous.get("feature_path"))), signature, len(object_names)
                )
                is not None
            )
            if previous_complete and not args.overwrite:
                print(
                    f"[metric {index:03d}/{len(candidates):03d}] reuse {variant}",
                    flush=True,
                )
                continue
            masks = load_mask_cache(
                mask_root / f"{variant}.npz", signature, len(object_names)
            )
            if masks is None:
                raise RuntimeError(f"missing candidate masks: {variant}")
            frames, fps = load_video_frames(video_path)
            track_path = output_root / "tracks" / f"{variant}.npz"
            with np.load(track_path, allow_pickle=False) as arrays:
                tracks = arrays["tracks"].astype(np.float32)
                visibility = arrays["visibility"].astype(bool)
            centers = centers_from_tracks(tracks, visibility, object_slices)
            features = pooled_features(
                variant, video_path, signature, frames, masks, centers
            )
            per_object = {
                name: object_survival_metrics(
                    masks[:, object_index],
                    baseline_masks[:, object_index],
                    features[:, object_index],
                    baseline_features[:, object_index],
                    prompt_masks[object_index],
                    float(calibration[name]["threshold"]),
                )
                for object_index, name in enumerate(object_names)
            }
            selected_objects = (
                [str(candidate["region"])]
                if candidate["target_scope"] == "single_object"
                else object_names
            )
            quality_pass = all(per_object[name]["quality_pass"] for name in selected_objects)
            selected_scores = [
                float(per_object[name]["disappearance_score_0_100"])
                for name in selected_objects
            ]
            selected_mask_absence = [
                100.0 * float(per_object[name]["empty_mask_rate"])
                for name in selected_objects
            ]
            metrics = {
                "selected_objects": selected_objects,
                "quality_pass": quality_pass,
                "target_mean_disappearance_score_0_100": rounded(
                    fmean(selected_scores)
                )
                if quality_pass
                else None,
                "target_worst_disappearance_score_0_100": rounded(
                    max(selected_scores)
                )
                if quality_pass
                else None,
                "target_mean_mask_absence_score_0_100": rounded(
                    fmean(selected_mask_absence)
                )
                if quality_pass
                else None,
                "target_worst_mask_absence_score_0_100": rounded(
                    max(selected_mask_absence)
                )
                if quality_pass
                else None,
                "objects": per_object,
            }
            overlay_path = overlay_root / f"{variant}.mp4"
            if not args.skip_overlays and (args.overwrite or not overlay_path.is_file()):
                render_overlay(
                    overlay_path,
                    variant,
                    baseline_frames,
                    baseline_masks,
                    frames,
                    masks,
                    object_names,
                    per_object,
                    metrics["target_worst_disappearance_score_0_100"],
                    fps or baseline_fps,
                )
            record = {
                "variant_id": variant,
                "target_scope": candidate["target_scope"],
                "region": candidate.get("region"),
                "mask_mode": candidate["mask_mode"],
                "head_scope": candidate["head_scope"],
                "video_path": str(video_path),
                "video_signature": signature,
                "mask_path": str(mask_root / f"{variant}.npz"),
                "feature_path": str(feature_root / f"{variant}.npz"),
                "overlay_path": str(overlay_path),
                "metrics": metrics,
            }
            existing_records[variant] = record
            records = [
                existing_records[key]
                for key in sorted(existing_records)
                if key in allowed_variants
            ]
            atomic_json(
                report_path,
                report_payload(
                    case, seed, baseline_path, all_candidates, records, calibration
                ),
            )
            print(
                f"[metric {index:03d}/{len(candidates):03d}] {variant} "
                f"disappearance={metrics['target_worst_disappearance_score_0_100']}",
                flush=True,
            )
            del frames, masks, tracks, visibility, features
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        if dino is not None:
            del dino
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
