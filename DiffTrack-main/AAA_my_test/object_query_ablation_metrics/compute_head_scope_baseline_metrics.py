#!/usr/bin/env python3
"""Incrementally measure M1/M2/M3 head-scope videos against their baseline.

This CPU stage is intentionally small and auditable.  It supplies immediate,
object-local effect measurements while the model-backed CoTracker/SAM2/RAFT/
DINO/VBench pipeline remains a separate (and much more expensive) benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


DEFAULT_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50/runs"
)
PHYSICIQ67_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/runs"
)
MULTICASE_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/multicase_multiseed_baselines"
)
DEFAULT_OUTPUT_BASE = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics/"
    "head_scope_baseline_fast"
)
M123_MODES = (
    "self_only",
    "self_same",
    "self_future",
    "self_past",
    "incoming_only",
    "incoming_same",
    "incoming_future",
    "incoming_past",
    "outgoing_only",
    "outgoing_same",
    "outgoing_future",
    "outgoing_past",
)
HEAD_SCOPES = (
    "top100",
    "bottom100",
    "random100_layer_matched_draw0",
    "all720",
)
CATEGORY_DEFINITIONS = {
    "global_appearance": {
        "name": "全局外观影响",
        "metrics": ["1 - Global SSIM", "Global MAE"],
        "formula": "100 * [0.50*(1-global_SSIM) + 0.50*global_MAE]",
        "direction": "越大表示全画面的结构/像素外观变化越强",
    },
    "target_local": {
        "name": "目标对象局部影响",
        "metrics": ["Target ROI MAE"],
        "formula": "100 * target_ROI_MAE",
        "direction": "越大表示目标对象所在冻结 ROI 的位置/外观变化越强",
    },
    "temporal_appearance": {
        "name": "时序外观变化",
        "metrics": ["Global Delta-MAE", "Target ROI Delta-MAE"],
        "formula": "100 * [0.40*global_delta_MAE + 0.60*target_ROI_delta_MAE]",
        "direction": "越大表示逐帧像素变化模式差异越强；混合运动、外观、形变与闪烁，不用于轨迹排名",
    },
    "outside_spillover": {
        "name": "对象外传播影响",
        "metrics": ["Outside-object MAE", "Outside-object Delta-MAE"],
        "formula": "100 * mean(outside_object_MAE, outside_object_delta_MAE)",
        "direction": "越大表示背景/其他区域的静态及动态 spillover 越强",
    },
}
METRIC_DEFINITIONS = {
    "impact_score_0_100": {
        "definition": "相对 Baseline 的绝对视觉干预强度；不是质量或物理正确性分数。",
        "formula": (
            "100 * [0.20*(1-global_SSIM) + 0.15*global_MAE + "
            "0.15*global_delta_MAE + 0.30*target_ROI_MAE + "
            "0.20*target_ROI_delta_MAE]"
        ),
        "direction": "越大表示相对同 seed Baseline 的可见影响越强",
    },
    "global_ssim": {
        "definition": "全部 49 帧、全画面的逐帧 SSIM 均值。",
        "formula": "mean_t SSIM(I_abl(t), I_base(t))",
        "direction": "越小表示整体画面改变越大",
    },
    "global_mae_0_1": {
        "definition": "全画面 RGB 通道绝对误差，除以 255。",
        "formula": "mean |I_abl-I_base| / 255",
        "direction": "越大表示整体像素影响越强",
    },
    "global_temporal_delta_mae_0_1": {
        "definition": "消融与 Baseline 的相邻帧差分之差。",
        "formula": "mean |Delta I_abl-Delta I_base| / 255",
        "direction": "越大表示逐帧变化/运动模式差异越强",
    },
    "target_roi_mae_0_1": {
        "definition": "Baseline 冻结 CoTracker 对象 tube 凸包 ROI 内的 RGB 绝对误差。",
        "formula": "mean_(t,x in ROI_base(t)) |I_abl-I_base| / 255",
        "direction": "越大表示目标对象所在区域改变越强；包含位置与外观",
    },
    "target_roi_temporal_delta_mae_0_1": {
        "definition": "同一冻结对象 ROI 内的相邻帧差分误差。",
        "formula": "mean_(t,x in ROI_base(t) union ROI_base(t+1)) |Delta I_abl-Delta I_base| / 255",
        "direction": "越大表示目标区域的动态变化越强",
    },
    "outside_objects_mae_0_1": {
        "definition": "排除全部 Baseline 冻结对象 ROI 后的 RGB 绝对误差。",
        "formula": "mean_(t,x outside all object ROIs) |I_abl-I_base| / 255",
        "direction": "越大表示背景/其他区域 spillover 越强",
    },
    "outside_objects_temporal_delta_mae_0_1": {
        "definition": "全部对象 ROI 外的相邻帧差分误差。",
        "formula": "mean_(t,x outside object ROI union) |Delta I_abl-Delta I_base| / 255",
        "direction": "越大表示对象外动态 spillover 越强",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=176)
    parser.add_argument("--roi-dilate-px", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--head-scopes", nargs="+", choices=HEAD_SCOPES, default=list(HEAD_SCOPES)
    )
    parser.add_argument(
        "--watch-seconds",
        type=int,
        default=0,
        help="poll for newly completed videos; zero performs one pass",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size_bytes": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def discover_seed_dirs(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if root.name.startswith("seed_"):
        return [root]
    direct = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if direct:
        return direct
    return sorted(path for path in root.glob("*/seed_*") if path.is_dir())


def collect_candidates(seed_dir: Path, scopes: set[str]) -> list[dict[str, Any]]:
    records = []
    for manifest_path in sorted(seed_dir.glob("*/manifest.json")):
        video_path = manifest_path.parent / "generated.mp4"
        complete_path = manifest_path.parent / "complete.json"
        if not video_path.is_file() or not complete_path.is_file():
            continue
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        mode = str(manifest.get("mask_mode") or "")
        scope = str(manifest.get("head_scope") or "top100")
        target_scope = str(manifest.get("target_scope") or "")
        region = str(manifest.get("region") or "")
        if (
            mode not in M123_MODES
            or scope not in scopes
            or target_scope not in {"single_object", "all_objects"}
            or (target_scope == "single_object" and not region)
        ):
            continue
        records.append(
            {
                "id": str(manifest.get("variant_id") or manifest_path.parent.name),
                "variant_id": manifest_path.parent.name,
                "case": str(manifest.get("case") or seed_dir.parent.name),
                "seed": int(manifest.get("seed", int(seed_dir.name.removeprefix("seed_")))),
                "target_scope": target_scope,
                "region": region or None,
                "mask_mode": mode,
                "head_scope": scope,
                "head_count": int(
                    manifest.get("selected_head_count")
                    or len(manifest.get("selected_entries") or [])
                    or {
                        "top100": 100,
                        "bottom100": 100,
                        "random100_layer_matched_draw0": 100,
                        "all720": 720,
                    }[scope]
                ),
                "ranking_tag": str(manifest.get("ranking_tag") or ""),
                "tracks_npz": str(manifest.get("tracks_npz") or ""),
                "path": str(video_path.resolve()),
                "manifest_path": str(manifest_path.resolve()),
                "video_signature": file_signature(video_path),
            }
        )
    ids = [row["variant_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate variant IDs under {seed_dir}")
    return records


def load_frames(path: Path, width: int, height: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    capture.release()
    if len(frames) != 49:
        raise RuntimeError(f"expected 49 frames, got {len(frames)}: {path}")
    return np.stack(frames)


def load_rois(
    track_path: Path, width: int, height: int, dilate_px: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(track_path, allow_pickle=False) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        names = [str(value) for value in arrays["region_names"].tolist()]
        starts = arrays["point_starts"].astype(int).tolist()
        ends = arrays["point_ends"].astype(int).tolist()
        source_height = int(arrays["pixel_height"])
        source_width = int(arrays["pixel_width"])
    if tracks.shape[0] != 49 or not names:
        raise RuntimeError(f"invalid frozen baseline tracks: {track_path}")
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    rois: dict[str, np.ndarray] = {}
    for name, start, end in zip(names, starts, ends, strict=True):
        masks = np.zeros((49, height, width), dtype=bool)
        for frame_index in range(49):
            points = tracks[frame_index, start:end].copy()
            points[:, 0] *= width / source_width
            points[:, 1] *= height / source_height
            finite = np.isfinite(points).all(axis=1)
            points = np.rint(points[finite]).astype(np.int32)
            if not len(points):
                continue
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            mask = np.zeros((height, width), dtype=np.uint8)
            if len(points) >= 3:
                cv2.fillConvexPoly(mask, cv2.convexHull(points.reshape(-1, 1, 2)), 1)
            else:
                for x, y in points:
                    cv2.circle(mask, (int(x), int(y)), 2, 1, -1)
            masks[frame_index] = cv2.dilate(mask, kernel, iterations=1) > 0
        rois[name] = masks
    rois["all_objects"] = np.logical_or.reduce(list(rois.values()))
    return rois, {
        "definition": (
            "seed-matched Baseline frozen CoTracker point convex hull per frame, "
            f"dilated {dilate_px}px at {width}x{height}"
        ),
        "track_path": str(track_path),
        "region_names": names,
        "mean_area_pixels": {
            name: round(float(mask.sum(axis=(1, 2)).mean()), 3)
            for name, mask in rois.items()
        },
    }


def masked_channel_mae(diff: np.ndarray, mask: np.ndarray) -> float | None:
    count = int(mask.sum())
    if count == 0:
        return None
    return float(diff[mask].sum(dtype=np.float64) / (count * 3 * 255.0))


def compare_frames(
    baseline: np.ndarray,
    candidate: np.ndarray,
    target_roi: np.ndarray,
    all_objects_roi: np.ndarray,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.shape[0] != 49:
        raise RuntimeError("baseline/candidate frame signatures differ")
    absolute = np.abs(candidate.astype(np.int16) - baseline.astype(np.int16)).astype(
        np.float32
    )
    global_mae = float(absolute.mean(dtype=np.float64) / 255.0)
    ssim = float(
        np.mean(
            [
                structural_similarity(left, right, channel_axis=-1, data_range=255)
                for left, right in zip(baseline, candidate, strict=True)
            ],
            dtype=np.float64,
        )
    )
    mse = float(
        np.mean(
            np.square(candidate.astype(np.float32) - baseline.astype(np.float32)),
            dtype=np.float64,
        )
    )
    psnr = None if mse == 0 else float(10.0 * math.log10(255.0**2 / mse))

    baseline_delta = np.diff(baseline.astype(np.int16), axis=0)
    candidate_delta = np.diff(candidate.astype(np.int16), axis=0)
    delta_absolute = np.abs(candidate_delta - baseline_delta).astype(np.float32)
    global_delta = float(delta_absolute.mean(dtype=np.float64) / 255.0)
    target_delta_roi = np.logical_or(target_roi[:-1], target_roi[1:])
    all_delta_roi = np.logical_or(all_objects_roi[:-1], all_objects_roi[1:])
    outside = np.logical_not(all_objects_roi)
    outside_delta = np.logical_not(all_delta_roi)

    target_mae = masked_channel_mae(absolute, target_roi)
    target_delta_mae = masked_channel_mae(delta_absolute, target_delta_roi)
    outside_mae = masked_channel_mae(absolute, outside)
    outside_delta_mae = masked_channel_mae(delta_absolute, outside_delta)
    if target_mae is None or target_delta_mae is None:
        raise RuntimeError("target ROI has no valid pixels")
    impact = 100.0 * (
        0.20 * (1.0 - ssim)
        + 0.15 * global_mae
        + 0.15 * global_delta
        + 0.30 * target_mae
        + 0.20 * target_delta_mae
    )
    spillover = 100.0 * fmean(
        value for value in (outside_mae, outside_delta_mae) if value is not None
    )
    return {
        "impact_score_0_100": round(impact, 8),
        "spillover_score_0_100": round(spillover, 8),
        "global": {
            "ssim_mean": round(ssim, 9),
            "psnr_db": None if psnr is None else round(psnr, 6),
            "mae_0_1": round(global_mae, 9),
            "temporal_delta_mae_0_1": round(global_delta, 9),
        },
        "target_roi": {
            "mae_0_1": round(target_mae, 9),
            "temporal_delta_mae_0_1": round(target_delta_mae, 9),
            "mean_area_fraction": round(float(target_roi.mean()), 9),
        },
        "outside_objects": {
            "mae_0_1": None if outside_mae is None else round(outside_mae, 9),
            "temporal_delta_mae_0_1": (
                None if outside_delta_mae is None else round(outside_delta_mae, 9)
            ),
            "mean_area_fraction": round(float(outside.mean()), 9),
        },
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def assign_ranks(records: list[dict[str, Any]]) -> None:
    if not records:
        return
    for row in records:
        metrics = row["metrics"]
        global_metrics = metrics["global"]
        target_metrics = metrics["target_roi"]
        outside_metrics = metrics["outside_objects"]
        outside_values = [
            float(value)
            for value in (
                outside_metrics.get("mae_0_1"),
                outside_metrics.get("temporal_delta_mae_0_1"),
            )
            if value is not None
        ]
        metrics["category_scores_0_100"] = {
            "global_appearance": round(
                100.0
                * (
                    0.50 * (1.0 - float(global_metrics["ssim_mean"]))
                    + 0.50 * float(global_metrics["mae_0_1"])
                ),
                8,
            ),
            "target_local": round(100.0 * float(target_metrics["mae_0_1"]), 8),
            "temporal_appearance": round(
                100.0
                * (
                    0.40 * float(global_metrics["temporal_delta_mae_0_1"])
                    + 0.60
                    * float(target_metrics["temporal_delta_mae_0_1"])
                ),
                8,
            ),
            "outside_spillover": round(
                100.0 * (fmean(outside_values) if outside_values else 0.0), 8
            ),
        }
    scores = np.asarray([row["metrics"]["impact_score_0_100"] for row in records])
    descending = len(records) + 1.0 - average_ranks(scores)
    for row, rank in zip(records, descending, strict=True):
        row["impact_rank_within_case_seed"] = round(float(rank), 3)
        row["impact_percentile_within_case_seed"] = round(
            float(100.0 * (len(records) - rank) / max(len(records) - 1, 1)), 3
        )
        row["category_ranks_within_case_seed"] = {}
        row["category_percentiles_within_case_seed"] = {}
    for category_id in CATEGORY_DEFINITIONS:
        category_scores = np.asarray(
            [
                row["metrics"]["category_scores_0_100"][category_id]
                for row in records
            ]
        )
        category_ranks = len(records) + 1.0 - average_ranks(category_scores)
        for row, rank in zip(records, category_ranks, strict=True):
            row["category_ranks_within_case_seed"][category_id] = round(
                float(rank), 3
            )
            row["category_percentiles_within_case_seed"][category_id] = round(
                float(100.0 * (len(records) - rank) / max(len(records) - 1, 1)),
                3,
            )


def compute_seed(
    seed_dir: Path,
    args: argparse.Namespace,
    scopes: set[str],
) -> tuple[Path | None, int]:
    candidates = collect_candidates(seed_dir, scopes)
    if not candidates:
        return None, 0
    case = candidates[0]["case"]
    seed = int(candidates[0]["seed"])
    if any(row["case"] != case or int(row["seed"]) != seed for row in candidates):
        raise RuntimeError(f"mixed case/seed manifests under {seed_dir}")
    output = args.output_base / case / f"seed_{seed:05d}" / "report.json"
    try:
        previous = read_json(output)
    except (OSError, ValueError, json.JSONDecodeError):
        previous = {}
    previous_map = {
        str(row.get("variant_id")): row
        for row in previous.get("records", [])
        if isinstance(row, dict)
    }
    baseline_candidates = [
        args.baseline_root / case / f"seed_{seed:05d}" / "generated.mp4",
        PHYSICIQ67_BASELINE_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
        MULTICASE_BASELINE_ROOT / case / f"seed_{seed:05d}" / "generated.mp4",
    ]
    baseline_path = next((path for path in baseline_candidates if path.is_file()), None)
    if baseline_path is None:
        raise FileNotFoundError(
            "no seed-matched Baseline; checked "
            + ", ".join(str(path) for path in baseline_candidates)
        )
    baseline_signature = file_signature(baseline_path)
    baseline_changed = previous.get("baseline_signature") != baseline_signature
    pending = [
        row
        for row in candidates
        if baseline_changed
        or row["variant_id"] not in previous_map
        or previous_map[row["variant_id"]].get("video_signature")
        != row["video_signature"]
    ]
    if (
        not pending
        and len(previous_map) == len(candidates)
        and int(previous.get("schema_version", 0)) >= 3
    ):
        return output, 0

    track_path = seed_dir / "frozen_baseline_tracks" / "tracks.npz"
    if not track_path.is_file():
        external_tracks = {
            Path(str(row["tracks_npz"])).expanduser().resolve()
            for row in candidates
            if str(row.get("tracks_npz") or "")
        }
        if len(external_tracks) == 1:
            track_path = next(iter(external_tracks))
        elif len(external_tracks) > 1:
            raise RuntimeError(
                f"mixed external frozen tracks under {seed_dir}: {sorted(map(str, external_tracks))}"
            )
    if not track_path.is_file():
        raise FileNotFoundError(track_path)
    rois, roi_audit = load_rois(
        track_path, args.width, args.height, args.roi_dilate_px
    )
    baseline = load_frames(baseline_path, args.width, args.height)

    def work(row: dict[str, Any]) -> dict[str, Any]:
        target_key = (
            str(row["region"])
            if row["target_scope"] == "single_object"
            else "all_objects"
        )
        if target_key not in rois:
            raise RuntimeError(f"{row['variant_id']}: missing target ROI {target_key}")
        candidate = load_frames(Path(row["path"]), args.width, args.height)
        metrics = compare_frames(
            baseline, candidate, rois[target_key], rois["all_objects"]
        )
        return {**row, "target_roi_key": target_key, "metrics": metrics}

    updates: dict[str, dict[str, Any]] = {}
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            for index, result in enumerate(executor.map(work, pending), start=1):
                updates[result["variant_id"]] = result
                print(
                    f"[{case}/seed_{seed:05d} {index:03d}/{len(pending):03d}] "
                    f"{result['variant_id']} impact={result['metrics']['impact_score_0_100']:.4f}",
                    flush=True,
                )
    candidate_ids = {row["variant_id"] for row in candidates}
    records = []
    for candidate in candidates:
        variant = candidate["variant_id"]
        if variant in updates:
            records.append(updates[variant])
        elif variant in previous_map:
            records.append(previous_map[variant])
    records = [row for row in records if row["variant_id"] in candidate_ids]
    assign_ranks(records)
    records.sort(
        key=lambda row: (
            -float(row["metrics"]["impact_score_0_100"]), row["variant_id"]
        )
    )
    report = {
        "schema_version": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "seed": seed,
        "reference": "same-seed no-intervention Baseline",
        "metric_family": "fast baseline-relative image/temporal effect",
        "not_included": (
            "This report does not claim CoTracker trajectory, SAM2 shape, RAFT, "
            "DINOv2, LPIPS, VBench, simulator-GT, or physical correctness metrics."
        ),
        "comparison_resolution_hwc": [args.height, args.width, 3],
        "video_count": len(records) + 1,
        "ablation_count": len(records),
        "baseline_path": str(baseline_path),
        "baseline_signature": baseline_signature,
        "roi_audit": roi_audit,
        "metric_definitions": METRIC_DEFINITIONS,
        "category_definitions": CATEGORY_DEFINITIONS,
        "records": records,
    }
    atomic_json(output, report)
    return output, len(pending)


def rebuild_global_ranking(output_base: Path) -> Path:
    source_reports = sorted(output_base.glob("*/seed_*/report.json"))
    rows = []
    for report_path in source_reports:
        try:
            report = read_json(report_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        for record in report.get("records", []):
            rows.append(
                {
                    "case": report.get("case"),
                    "seed": report.get("seed"),
                    "variant_id": record.get("variant_id"),
                    "target_scope": record.get("target_scope"),
                    "region": record.get("region"),
                    "mask_mode": record.get("mask_mode"),
                    "head_scope": record.get("head_scope"),
                    "ranking_tag": record.get("ranking_tag"),
                    "impact_score_0_100": record["metrics"]["impact_score_0_100"],
                    "spillover_score_0_100": record["metrics"]["spillover_score_0_100"],
                    "category_scores_0_100": record["metrics"][
                        "category_scores_0_100"
                    ],
                    "report_path": str(report_path),
                }
            )
    rows.sort(key=lambda row: (-float(row["impact_score_0_100"]), row["variant_id"]))
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(
            row[field]
            for field in (
                "case",
                "target_scope",
                "region",
                "mask_mode",
                "head_scope",
                "ranking_tag",
            )
        )
        grouped[key].append(row)
    aggregates = []
    for key, values in grouped.items():
        case, target_scope, region, mode, scope, ranking_tag = key
        aggregates.append(
            {
                "case": case,
                "target_scope": target_scope,
                "region": region,
                "mask_mode": mode,
                "head_scope": scope,
                "ranking_tag": ranking_tag,
                "sample_count": len(values),
                "seeds": sorted({int(row["seed"]) for row in values}),
                "mean_impact_score_0_100": round(
                    fmean(float(row["impact_score_0_100"]) for row in values), 8
                ),
                "mean_spillover_score_0_100": round(
                    fmean(float(row["spillover_score_0_100"]) for row in values), 8
                ),
                "mean_category_scores_0_100": {
                    category_id: round(
                        fmean(
                            float(row["category_scores_0_100"][category_id])
                            for row in values
                        ),
                        8,
                    )
                    for category_id in CATEGORY_DEFINITIONS
                },
            }
        )
    aggregates.sort(
        key=lambda row: (-float(row["mean_impact_score_0_100"]), str(row))
    )
    output = output_base / "ranking.json"
    atomic_json(
        output,
        {
            "schema_version": 3,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference": "same-seed no-intervention Baseline",
            "ranking_definition": METRIC_DEFINITIONS["impact_score_0_100"],
            "category_definitions": CATEGORY_DEFINITIONS,
            "sample_record_count": len(rows),
            "experiment_group_count": len(aggregates),
            "records": rows,
            "experiment_aggregates": aggregates,
        },
    )
    return output


def run_pass(args: argparse.Namespace) -> tuple[int, int]:
    scopes = set(args.head_scopes)
    reports = updates = 0
    for seed_dir in discover_seed_dirs(args.result_dir):
        try:
            output, changed = compute_seed(seed_dir, args, scopes)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"[skip] {seed_dir}: {exc}", flush=True)
            continue
        if output is not None:
            reports += 1
            updates += changed
    ranking = rebuild_global_ranking(args.output_base)
    print(
        f"[pass] reports={reports} newly_measured={updates} ranking={ranking}",
        flush=True,
    )
    return reports, updates


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.workers <= 0:
        raise ValueError("width, height, and workers must be positive")
    if args.watch_seconds < 0:
        raise ValueError("watch-seconds must be non-negative")
    args.result_dir = args.result_dir.expanduser().resolve()
    args.baseline_root = args.baseline_root.expanduser().resolve()
    args.output_base = args.output_base.expanduser().resolve()
    if str(args.output_base).startswith("/home/gaoya/"):
        raise ValueError("large metric artifacts must be written under /data/gaoya")
    while True:
        run_pass(args)
        if args.watch_seconds == 0:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
