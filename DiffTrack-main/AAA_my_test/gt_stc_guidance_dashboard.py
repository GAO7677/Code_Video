#!/usr/bin/env python3
"""Read-only preflight and runtime dashboard for GT-STC guidance."""

from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_v1"
)
HYBRID_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
SEED = 47326
TOKEN_HEIGHT = 44
TOKEN_WIDTH = 80
LOSS_MODES = ("region", "point")
LOG_FILES = {
    "gpu2_generate": ROOT / "logs/gpu2_full_generate.log",
    "gpu2_retry": ROOT / "logs/gpu2_postcheck_generate.log",
    "gpu7_generate": ROOT / "logs/gpu7_full_generate.log",
}
FILMSTRIP_ROOT = ROOT / "visualizations" / "source_region_filmstrips"
COTRACKER_COMPARISON_ROOT = (
    ROOT / "visualizations" / "sam2_cotracker_anchor_comparisons"
)
HYBRID_COMPARISON_ROOT = (
    HYBRID_ROOT / "visualizations" / "direct_neighbor_final_comparisons"
)
REGION_COLORS = (
    (22, 224, 182),
    (255, 101, 112),
    (255, 202, 66),
    (109, 161, 255),
    (210, 126, 255),
    (255, 143, 58),
)
_filmstrip_lock = threading.Lock()
_cotracker_comparison_lock = threading.Lock()
_hybrid_comparison_lock = threading.Lock()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ordered_cases() -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        case = Path(raw).stem
        if case not in seen:
            seen.add(case)
            result.append(case)
    return result


def _target_specs(
    names: list[str], moving: np.ndarray
) -> list[tuple[str, list[int]]]:
    targets = [(name, [index]) for index, name in enumerate(names)]
    moving_indices = np.flatnonzero(moving).astype(int).tolist()
    if len(moving_indices) >= 2:
        targets.append(("moving_union", moving_indices))
    return targets


def _token_counts(masks_thw: np.ndarray) -> list[int]:
    masks = np.asarray(masks_thw, dtype=bool)
    if masks.shape[1:] != (TOKEN_HEIGHT * 16, TOKEN_WIDTH * 16):
        # Presence is the actual failure condition. Keep a conservative scalar
        # fallback if a later run changes spatial resolution.
        return [int(frame.any()) for frame in masks]
    down = masks.reshape(
        masks.shape[0], TOKEN_HEIGHT, 16, TOKEN_WIDTH, 16
    ).any(axis=(2, 4))
    return down.reshape(masks.shape[0], -1).sum(axis=1).astype(int).tolist()


def _valid_point_pairs(visibility_tn: np.ndarray) -> int:
    visibility = np.asarray(visibility_tn, dtype=bool)
    return sum(
        int(np.any(visibility[query_time] & visibility[key_time]))
        for query_time in range(visibility.shape[0])
        for key_time in range(visibility.shape[0])
        if query_time != key_time
    )


def _variant_name(mode: str, target: str) -> str:
    return f"{mode}__{target}__lambda0p1"


def _variant_status(case: str, variant: str, preflight_fails: bool) -> dict[str, Any]:
    directory = ROOT / "generations" / case / f"seed_{SEED:05d}" / variant
    video = directory / "generated.mp4"
    complete = directory / "complete.json"
    metric = directory / "trajectory_metrics.json"
    if complete.is_file() and video.is_file() and video.stat().st_size > 0:
        state = "complete"
    elif directory.is_dir() and preflight_fails:
        state = "failed_confirmed"
    elif preflight_fails:
        state = "will_fail"
    elif directory.is_dir():
        state = "active_or_interrupted"
    else:
        state = "queued"
    return {
        "variant": variant,
        "state": state,
        "directory": str(directory),
        "video_ready": video.is_file() and video.stat().st_size > 0,
        "metric_ready": metric.is_file(),
    }


@lru_cache(maxsize=1)
def _preflight_cases() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for order, case in enumerate(_ordered_cases()):
        tube_path = ROOT / "gt_tubes" / case / "tube.npz"
        manifest_path = tube_path.with_name("manifest.json")
        manifest = _json(manifest_path)
        if not tube_path.is_file():
            result.append(
                {
                    "case": case,
                    "order": order,
                    "worker": order % 2,
                    "tube_ready": False,
                    "targets": [],
                    "source_video": None,
                }
            )
            continue
        with np.load(tube_path, allow_pickle=False) as tube:
            names = [str(value) for value in tube["region_names"].tolist()]
            moving = np.asarray(tube["moving"], dtype=bool)
            masks = np.asarray(tube["masks_othw"], dtype=bool)
            visibility = np.asarray(tube["visibility_tn"], dtype=bool)
            starts = np.asarray(tube["point_starts"], dtype=int)
            ends = np.asarray(tube["point_ends"], dtype=int)
            targets = []
            for target_name, object_indices in _target_specs(names, moving):
                target_masks = np.logical_or.reduce(masks[object_indices], axis=0)
                token_counts = _token_counts(target_masks)
                empty_frames = [
                    index for index, count in enumerate(token_counts) if count == 0
                ]
                point_indices = np.concatenate(
                    [
                        np.arange(starts[index], ends[index], dtype=int)
                        for index in object_indices
                    ]
                )
                valid_pairs = _valid_point_pairs(visibility[:, point_indices])
                region_fails = bool(empty_frames)
                point_fails = valid_pairs == 0
                targets.append(
                    {
                        "name": target_name,
                        "object_indices": object_indices,
                        "moving": all(bool(moving[index]) for index in object_indices),
                        "token_counts": token_counts,
                        "empty_frames": empty_frames,
                        "valid_point_pairs": valid_pairs,
                        "region_preflight": "fail" if region_fails else "pass",
                        "point_preflight": "fail" if point_fails else "pass",
                    }
                )
        result.append(
            {
                "case": case,
                "order": order,
                "worker": order % 2,
                "tube_ready": True,
                "targets": targets,
                "source_video": manifest.get("source_video"),
                "source_json": manifest.get("source_json"),
                "source_frame_count": manifest.get("source_frame_count"),
                "source_frame_policy": manifest.get("source_frame_policy"),
                "objects": manifest.get("objects") or [],
            }
        )
    return result


def _with_runtime(preflight: dict[str, Any]) -> dict[str, Any]:
    row = dict(preflight)
    case = str(row["case"])
    row["hybrid_tube_ready"] = (
        HYBRID_ROOT / "gt_tubes" / case / "tube.npz"
    ).is_file()
    generation_root = ROOT / "generations" / case / f"seed_{SEED:05d}"
    baseline = generation_root / "baseline"
    baseline_complete = (baseline / "complete.json").is_file() and (
        baseline / "generated.mp4"
    ).is_file()
    row["baseline"] = {
        "state": (
            "complete"
            if baseline_complete
            else "active_or_interrupted"
            if baseline.is_dir()
            else "queued"
        ),
        "metric_ready": (baseline / "trajectory_metrics.json").is_file(),
        "directory": str(baseline),
    }
    targets = []
    for target in row["targets"]:
        target = dict(target)
        region_variant = _variant_name("region", str(target["name"]))
        point_variant = _variant_name("point", str(target["name"]))
        target["region"] = _variant_status(
            case, region_variant, target["region_preflight"] == "fail"
        )
        target["point"] = _variant_status(
            case, point_variant, target["point_preflight"] == "fail"
        )
        targets.append(target)
    row["targets"] = targets
    states = [row["baseline"]["state"]] + [
        target[mode]["state"] for target in targets for mode in LOSS_MODES
    ]
    if "failed_confirmed" in states:
        row["state"] = "failed"
    elif "will_fail" in states:
        row["state"] = "at_risk"
    elif states and all(state == "complete" for state in states):
        row["state"] = "complete"
    elif "active_or_interrupted" in states:
        row["state"] = "active"
    else:
        row["state"] = "queued"
    row["completed_variants"] = states.count("complete")
    row["expected_variants"] = len(states)
    return row


def catalog() -> dict[str, Any]:
    cases = [_with_runtime(row) for row in _preflight_cases()]
    variants = [case["baseline"] for case in cases]
    variants.extend(
        target[mode]
        for case in cases
        for target in case["targets"]
        for mode in LOSS_MODES
    )
    states = [str(variant["state"]) for variant in variants]
    risk_targets = [
        {
            "case": case["case"],
            "target": target["name"],
            "empty_frames": target["empty_frames"],
            "region_state": target["region"]["state"],
            "point_pairs": target["valid_point_pairs"],
        }
        for case in cases
        for target in case["targets"]
        if target["region_preflight"] == "fail" or target["point_preflight"] == "fail"
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "seed": SEED,
        "case_count": len(cases),
        "tube_ready": sum(int(case["tube_ready"]) for case in cases),
        "expected_variants": len(variants),
        "completed_variants": states.count("complete"),
        "metric_ready": sum(int(variant["metric_ready"]) for variant in variants),
        "confirmed_failures": states.count("failed_confirmed"),
        "predicted_failures": states.count("will_fail"),
        "active_or_interrupted": states.count("active_or_interrupted"),
        "risk_targets": risk_targets,
        "cases": cases,
        "log_files": {
            key: {"path": str(path), "ready": path.is_file()}
            for key, path in LOG_FILES.items()
        },
        "definitions": {
            "region_failure": (
                "Region loss requires a non-empty SAM2 token set at every one of the "
                "13 latent times. Any empty R_t raises before denoising step 0."
            ),
            "point_failure": (
                "Point loss requires at least one identity-preserving CoTracker point "
                "visible at both ends of a cross-time pair."
            ),
        },
    }


def log_file(name: str) -> Path | None:
    path = LOG_FILES.get(name)
    if path is None or not path.is_file():
        return None
    try:
        path.resolve().relative_to((ROOT / "logs").resolve())
    except ValueError:
        return None
    return path


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask, dtype=bool)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~interior


def _overlay_regions(
    frame: np.ndarray,
    masks_ohw: np.ndarray,
    names: list[str],
    frame_index: int,
    latent_index: int,
    is_anchor: bool,
    tile_size: tuple[int, int],
) -> Image.Image:
    width, height = tile_size
    base = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3]).resize(
        tile_size, Image.Resampling.BILINEAR
    )
    pixels = np.asarray(base).copy().astype(np.float32)
    small_masks: list[np.ndarray] = []
    empty_names: list[str] = []
    for object_index, mask in enumerate(masks_ohw):
        small = np.asarray(
            Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).resize(
                tile_size, Image.Resampling.NEAREST
            )
        ) > 0
        small_masks.append(small)
        if not small.any():
            empty_names.append(names[object_index])
            continue
        color = np.asarray(REGION_COLORS[object_index % len(REGION_COLORS)])
        pixels[small] = pixels[small] * 0.62 + color * 0.38
        pixels[_mask_edge(small)] = color
    image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image, "RGBA")
    for object_index, small in enumerate(small_masks):
        if not small.any():
            continue
        yy, xx = np.nonzero(small)
        x0, y0, x1, y1 = int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())
        color = REGION_COLORS[object_index % len(REGION_COLORS)]
        draw.rectangle((x0, y0, x1, y1), outline=(*color, 255), width=2)
        label = names[object_index]
        label_y = max(16, y0)
        draw.rectangle(
            (x0, label_y - 13, min(width - 1, x0 + 8 + 7 * len(label)), label_y),
            fill=(7, 15, 19, 215),
        )
        draw.text((x0 + 3, label_y - 12), label, fill=(*color, 255))
    frame_label = f"F{frame_index:02d} -> R{latent_index:02d}"
    if is_anchor:
        frame_label += "  ANCHOR"
    draw.rectangle((0, 0, min(width - 1, 124), 15), fill=(5, 13, 17, 220))
    draw.text((4, 2), frame_label, fill=(255, 255, 255, 255))
    if empty_names:
        message = "EMPTY: " + ",".join(empty_names)
        draw.rectangle((0, height - 16, width - 1, height - 1), fill=(140, 20, 32, 225))
        draw.text((4, height - 14), message, fill=(255, 235, 236, 255))
    draw.rectangle(
        (0, 0, width - 1, height - 1),
        outline=(255, 255, 255, 255) if is_anchor else (72, 90, 100, 255),
        width=3 if is_anchor else 1,
    )
    return image


def _build_region_filmstrip(case: str, destination: Path) -> None:
    tube_path = ROOT / "gt_tubes" / case / "tube.npz"
    manifest = _json(tube_path.with_name("manifest.json"))
    source_video = Path(str(manifest.get("source_video") or ""))
    if not tube_path.is_file() or not source_video.is_file():
        raise FileNotFoundError(f"missing GT tube or source video for {case}")
    frames = np.asarray(iio.imread(source_video, index=None))
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise RuntimeError(f"unexpected source video shape: {frames.shape}")
    frames = frames[:49, ..., :3]
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    with np.load(tube_path, allow_pickle=False) as tube:
        masks = np.asarray(tube["masks_othw"], dtype=bool)
        anchors = np.asarray(tube["anchor_source_frames"], dtype=int)
        names = [str(value) for value in tube["region_names"].tolist()]
    tile_width, tile_height = 256, 141
    label_height = 22
    columns = 7
    rows = int(math.ceil(len(frames) / columns))
    header_height = 58
    sheet = Image.new(
        "RGB",
        (columns * tile_width, header_height + rows * (tile_height + label_height)),
        (18, 34, 43),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{case}  |  source frames={len(frames)}", fill=(245, 249, 250))
    legend_x = 12
    for object_index, name in enumerate(names):
        color = REGION_COLORS[object_index % len(REGION_COLORS)]
        draw.rectangle((legend_x, 31, legend_x + 13, 44), fill=color)
        draw.text((legend_x + 18, 31), name, fill=(220, 230, 234))
        legend_x += 28 + 7 * len(name)
    draw.text(
        (max(legend_x + 12, 650), 31),
        "Every F uses nearest saved latent-anchor SAM2 region; white border = exact anchor",
        fill=(147, 170, 181),
    )
    anchor_set = set(int(value) for value in anchors.tolist())
    for frame_index, frame in enumerate(frames):
        latent_index = int(np.argmin(np.abs(anchors - frame_index)))
        tile = _overlay_regions(
            frame,
            masks[:, latent_index],
            names,
            frame_index,
            latent_index,
            frame_index in anchor_set,
            (tile_width, tile_height),
        )
        x = (frame_index % columns) * tile_width
        y = header_height + (frame_index // columns) * (tile_height + label_height)
        sheet.paste(tile, (x, y))
        coverage = " · ".join(
            f"{name}:{'region' if mask.any() else 'EMPTY'}"
            for name, mask in zip(names, masks[:, latent_index])
        )
        ImageDraw.Draw(sheet).text(
            (x + 4, y + tile_height + 4), coverage, fill=(190, 205, 212)
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    sheet.save(temporary, format="JPEG", quality=89, optimize=True, progressive=True)
    temporary.replace(destination)


def region_filmstrip(case: str) -> Path | None:
    if case not in set(_ordered_cases()):
        return None
    tube_path = ROOT / "gt_tubes" / case / "tube.npz"
    manifest = _json(tube_path.with_name("manifest.json"))
    source_video = Path(str(manifest.get("source_video") or ""))
    if not tube_path.is_file() or not source_video.is_file():
        return None
    destination = FILMSTRIP_ROOT / f"{case}.jpg"
    newest_input = max(tube_path.stat().st_mtime_ns, source_video.stat().st_mtime_ns)
    if destination.is_file() and destination.stat().st_mtime_ns >= newest_input:
        return destination
    with _filmstrip_lock:
        if destination.is_file() and destination.stat().st_mtime_ns >= newest_input:
            return destination
        _build_region_filmstrip(case, destination)
    return destination


def _draw_cotracker_panel(
    frame: np.ndarray,
    tracks_tn2: np.ndarray,
    visibility_tn: np.ndarray,
    names: list[str],
    starts: np.ndarray,
    ends: np.ndarray,
    latent_index: int,
    pixel_hw: tuple[int, int],
    tile_size: tuple[int, int],
) -> Image.Image:
    width, height = tile_size
    pixel_height, pixel_width = pixel_hw
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3]).resize(
        tile_size, Image.Resampling.BILINEAR
    )
    draw = ImageDraw.Draw(image, "RGBA")
    scale_x = (width - 1) / max(pixel_width - 1, 1)
    scale_y = (height - 1) / max(pixel_height - 1, 1)
    history_start = max(0, latent_index - 5)
    total_visible = 0
    total_points = 0
    for object_index, (name, start, end) in enumerate(zip(names, starts, ends)):
        color = REGION_COLORS[object_index % len(REGION_COLORS)]
        start, end = int(start), int(end)
        total_points += end - start
        total_visible += int(visibility_tn[latent_index, start:end].sum())
        for point_index in range(start, end):
            for time_index in range(history_start + 1, latent_index + 1):
                if not (
                    visibility_tn[time_index - 1, point_index]
                    and visibility_tn[time_index, point_index]
                ):
                    continue
                x0, y0 = tracks_tn2[time_index - 1, point_index]
                x1, y1 = tracks_tn2[time_index, point_index]
                alpha = 70 + int(150 * (time_index - history_start) / max(latent_index - history_start, 1))
                draw.line(
                    (x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y),
                    fill=(*color, alpha),
                    width=2,
                )
            if not visibility_tn[latent_index, point_index]:
                continue
            x, y = tracks_tn2[latent_index, point_index]
            x, y = float(x * scale_x), float(y * scale_y)
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(5, 13, 17, 220))
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(*color, 255))
        visible = int(visibility_tn[latent_index, start:end].sum())
        current = np.flatnonzero(visibility_tn[latent_index, start:end])
        if len(current):
            point_index = start + int(current[0])
            x, y = tracks_tn2[latent_index, point_index]
            x, y = float(x * scale_x), float(y * scale_y)
            label = f"{name} {visible}/{end - start}"
            label_width = min(width - 1, 8 + 7 * len(label))
            label_x = max(0, min(width - label_width, int(x + 6)))
            label_y = max(17, min(height - 2, int(y)))
            draw.rectangle(
                (label_x, label_y - 14, label_x + label_width, label_y),
                fill=(5, 13, 17, 210),
            )
            draw.text((label_x + 3, label_y - 13), label, fill=(*color, 255))
    draw.rectangle((0, 0, width - 1, 16), fill=(5, 13, 17, 220))
    draw.text(
        (4, 2),
        f"CoTracker R{latent_index:02d}  visible {total_visible}/{total_points}",
        fill=(255, 255, 255, 255),
    )
    if total_visible == 0:
        draw.rectangle((0, height - 17, width - 1, height - 1), fill=(140, 20, 32, 230))
        draw.text((4, height - 15), "NO VISIBLE POINT", fill=(255, 235, 236, 255))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(255, 255, 255, 255), width=2)
    return image


def _build_cotracker_comparison(case: str, destination: Path) -> None:
    tube_path = ROOT / "gt_tubes" / case / "tube.npz"
    manifest = _json(tube_path.with_name("manifest.json"))
    source_video = Path(str(manifest.get("source_video") or ""))
    if not tube_path.is_file() or not source_video.is_file():
        raise FileNotFoundError(f"missing GT tube or source video for {case}")
    frames = np.asarray(iio.imread(source_video, index=None))[..., :3]
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    with np.load(tube_path, allow_pickle=False) as tube:
        masks = np.asarray(tube["masks_othw"], dtype=bool)
        tracks = np.asarray(tube["tracks_tn2"], dtype=np.float32)
        visibility = np.asarray(tube["visibility_tn"], dtype=bool)
        anchors = np.asarray(tube["anchor_source_frames"], dtype=int)
        names = [str(value) for value in tube["region_names"].tolist()]
        starts = np.asarray(tube["point_starts"], dtype=int)
        ends = np.asarray(tube["point_ends"], dtype=int)
        pixel_hw = (int(tube["pixel_height"]), int(tube["pixel_width"]))
    tile_width, tile_height = 240, 132
    pair_width = tile_width * 2
    block_height = tile_height + 40
    columns = 2
    rows = int(math.ceil(len(anchors) / columns))
    header_height = 66
    sheet = Image.new(
        "RGB",
        (columns * pair_width, header_height + rows * block_height),
        (18, 34, 43),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{case} | exact latent-anchor cross-examination", fill=(245, 249, 250))
    legend_x = 12
    for object_index, name in enumerate(names):
        color = REGION_COLORS[object_index % len(REGION_COLORS)]
        draw.rectangle((legend_x, 31, legend_x + 13, 44), fill=color)
        draw.text((legend_x + 18, 31), name, fill=(220, 230, 234))
        legend_x += 28 + 7 * len(name)
    draw.text(
        (12, 50),
        "left: SAM2 mask | right: visible CoTracker points + 5-anchor identity trails",
        fill=(147, 170, 181),
    )
    for latent_index, source_index in enumerate(anchors):
        frame = frames[int(source_index)]
        sam2 = _overlay_regions(
            frame,
            masks[:, latent_index],
            names,
            int(source_index),
            latent_index,
            True,
            (tile_width, tile_height),
        )
        cotracker = _draw_cotracker_panel(
            frame,
            tracks,
            visibility,
            names,
            starts,
            ends,
            latent_index,
            pixel_hw,
            (tile_width, tile_height),
        )
        x = (latent_index % columns) * pair_width
        y = header_height + (latent_index // columns) * block_height
        sheet.paste(sam2, (x, y))
        sheet.paste(cotracker, (x + tile_width, y))
        statuses = []
        for object_index, (name, start, end) in enumerate(zip(names, starts, ends)):
            mask_state = "M+" if masks[object_index, latent_index].any() else "M-"
            visible = int(visibility[latent_index, int(start) : int(end)].sum())
            statuses.append(f"{name}: {mask_state} P{visible}/{int(end)-int(start)}")
        footer = " | ".join(statuses)
        footer_color = (
            (255, 156, 163)
            if any(not mask.any() for mask in masks[:, latent_index])
            else (190, 205, 212)
        )
        ImageDraw.Draw(sheet).text((x + 4, y + tile_height + 5), footer, fill=footer_color)
        ImageDraw.Draw(sheet).text(
            (x + 4, y + tile_height + 20),
            f"F{int(source_index):02d} / R{latent_index:02d} · M=mask, P=visible points",
            fill=(120, 149, 162),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    sheet.save(temporary, format="JPEG", quality=90, optimize=True, progressive=True)
    temporary.replace(destination)


def cotracker_comparison(case: str) -> Path | None:
    if case not in set(_ordered_cases()):
        return None
    tube_path = ROOT / "gt_tubes" / case / "tube.npz"
    manifest = _json(tube_path.with_name("manifest.json"))
    source_video = Path(str(manifest.get("source_video") or ""))
    if not tube_path.is_file() or not source_video.is_file():
        return None
    destination = COTRACKER_COMPARISON_ROOT / f"{case}.jpg"
    newest_input = max(tube_path.stat().st_mtime_ns, source_video.stat().st_mtime_ns)
    if destination.is_file() and destination.stat().st_mtime_ns >= newest_input:
        return destination
    with _cotracker_comparison_lock:
        if destination.is_file() and destination.stat().st_mtime_ns >= newest_input:
            return destination
        _build_cotracker_comparison(case, destination)
    return destination


def _hybrid_panel(
    frame: np.ndarray,
    masks_ohw: np.ndarray,
    names: list[str],
    title: str,
    border: tuple[int, int, int],
    tile_size: tuple[int, int],
    *,
    tracks_tn2: np.ndarray | None = None,
    visibility_tn: np.ndarray | None = None,
    starts: np.ndarray | None = None,
    ends: np.ndarray | None = None,
    latent_index: int | None = None,
    pixel_hw: tuple[int, int] | None = None,
) -> Image.Image:
    width, height = tile_size
    image = _overlay_regions(
        frame,
        masks_ohw,
        names,
        frame_index=-1,
        latent_index=int(latent_index or 0),
        is_anchor=True,
        tile_size=tile_size,
    ).convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle((0, 0, width - 1, 18), fill=(5, 13, 17, 235))
    draw.text((5, 3), title, fill=(255, 255, 255, 255))
    if (
        tracks_tn2 is not None
        and visibility_tn is not None
        and starts is not None
        and ends is not None
        and latent_index is not None
        and pixel_hw is not None
    ):
        pixel_height, pixel_width = pixel_hw
        scale_x = (width - 1) / max(pixel_width - 1, 1)
        scale_y = (height - 1) / max(pixel_height - 1, 1)
        for object_index, (start, end) in enumerate(zip(starts, ends)):
            color = REGION_COLORS[object_index % len(REGION_COLORS)]
            for point_index in range(int(start), int(end)):
                if not visibility_tn[int(latent_index), point_index]:
                    continue
                x, y = tracks_tn2[int(latent_index), point_index]
                x = float(x) * scale_x
                y = float(y) * scale_y
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(3, 9, 13, 230))
                draw.ellipse((x - 2.8, y - 2.8, x + 2.8, y + 2.8), fill=(*color, 255))
    draw.rectangle((0, 0, width - 1, height - 1), outline=(*border, 255), width=4)
    return image.convert("RGB")


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.logical_or(left, right).sum()
    if union <= 0:
        return None
    return float(np.logical_and(left, right).sum() / union)


def _build_hybrid_comparison(case: str, destination: Path) -> None:
    tube_path = HYBRID_ROOT / "gt_tubes" / case / "tube.npz"
    manifest = _json(tube_path.with_name("manifest.json"))
    source_video = Path(str(manifest.get("source_video") or ""))
    if not tube_path.is_file() or not source_video.is_file():
        raise FileNotFoundError(f"missing hybrid GT tube or source video for {case}")
    frames = np.asarray(iio.imread(source_video, index=None))[..., :3]
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    with np.load(tube_path, allow_pickle=False) as tube:
        required = {
            "direct_masks_othw",
            "neighbor_masks_othw",
            "masks_othw",
            "direct_prompt_counts_ot",
            "neighbor_source_anchor_ot",
            "final_mask_source_ot",
        }
        missing = required - set(tube.files)
        if missing:
            raise RuntimeError(f"hybrid tube misses arrays: {sorted(missing)}")
        direct = np.asarray(tube["direct_masks_othw"], dtype=bool)
        neighbor = np.asarray(tube["neighbor_masks_othw"], dtype=bool)
        final = np.asarray(tube["masks_othw"], dtype=bool)
        prompt_counts = np.asarray(tube["direct_prompt_counts_ot"], dtype=int)
        neighbor_sources = np.asarray(tube["neighbor_source_anchor_ot"], dtype=int)
        final_sources = np.asarray(tube["final_mask_source_ot"], dtype=int)
        tracks = np.asarray(tube["tracks_tn2"], dtype=np.float32)
        visibility = np.asarray(tube["visibility_tn"], dtype=bool)
        anchors = np.asarray(tube["anchor_source_frames"], dtype=int)
        names = [str(value) for value in tube["region_names"].tolist()]
        starts = np.asarray(tube["point_starts"], dtype=int)
        ends = np.asarray(tube["point_ends"], dtype=int)
        pixel_hw = (int(tube["pixel_height"]), int(tube["pixel_width"]))

    tile_width, tile_height = 320, 176
    footer_height = 48
    header_height = 78
    sheet = Image.new(
        "RGB",
        (tile_width * 3, header_height + len(anchors) * (tile_height + footer_height)),
        (18, 34, 43),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{case} | CoTracker -> SAM2 prompt/fallback audit", fill=(245, 249, 250))
    draw.text(
        (12, 29),
        "DIRECT (actual tracked-point prompt) | NEIGHBOR (test candidate) | FINAL (actual applied tube)",
        fill=(197, 214, 221),
    )
    draw.text(
        (12, 50),
        "Rule: direct always wins; neighbor propagation is applied only when direct is unavailable or empty.",
        fill=(255, 196, 102),
    )
    source_labels = {0: "DIRECT", 1: "FALLBACK", 2: "MISSING"}
    source_colors = {0: (35, 177, 137), 1: (232, 151, 43), 2: (188, 54, 65)}
    for latent_index, source_frame in enumerate(anchors):
        frame = frames[int(source_frame)]
        row_y = header_height + latent_index * (tile_height + footer_height)
        direct_title = "DIRECT " + " ".join(
            f"{name}:P{prompt_counts[obj, latent_index]}"
            for obj, name in enumerate(names)
        )
        neighbor_title = "NEIGHBOR " + " ".join(
            f"{name}:R{neighbor_sources[obj, latent_index]:02d}"
            if neighbor_sources[obj, latent_index] >= 0
            else f"{name}:none"
            for obj, name in enumerate(names)
        )
        final_title = "FINAL " + " ".join(
            f"{name}:{source_labels.get(final_sources[obj, latent_index], '?')}"
            for obj, name in enumerate(names)
        )
        final_border = source_colors[
            int(max(final_sources[:, latent_index], default=2))
        ]
        panels = (
            _hybrid_panel(
                frame,
                direct[:, latent_index],
                names,
                direct_title,
                (52, 152, 219),
                (tile_width, tile_height),
                tracks_tn2=tracks,
                visibility_tn=visibility,
                starts=starts,
                ends=ends,
                latent_index=latent_index,
                pixel_hw=pixel_hw,
            ),
            _hybrid_panel(
                frame,
                neighbor[:, latent_index],
                names,
                neighbor_title,
                (232, 151, 43),
                (tile_width, tile_height),
                latent_index=latent_index,
            ),
            _hybrid_panel(
                frame,
                final[:, latent_index],
                names,
                final_title,
                final_border,
                (tile_width, tile_height),
                latent_index=latent_index,
            ),
        )
        for column, panel in enumerate(panels):
            sheet.paste(panel, (column * tile_width, row_y))
        ious = []
        for object_index, name in enumerate(names):
            iou = _mask_iou(
                direct[object_index, latent_index],
                neighbor[object_index, latent_index],
            )
            ious.append(f"{name} IoU={'N/A' if iou is None else f'{iou:.3f}'}")
        draw.text(
            (6, row_y + tile_height + 5),
            f"F{int(source_frame):02d} / R{latent_index:02d} · " + " | ".join(ious),
            fill=(191, 207, 214),
        )
        applied = " | ".join(
            f"{name}: {source_labels.get(final_sources[obj, latent_index], '?')}"
            + (
                f" from R{neighbor_sources[obj, latent_index]:02d}"
                if final_sources[obj, latent_index] == 1
                else ""
            )
            for obj, name in enumerate(names)
        )
        draw.text((6, row_y + tile_height + 24), applied, fill=final_border)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.jpg")
    sheet.save(temporary, format="JPEG", quality=91, optimize=True, progressive=True)
    temporary.replace(destination)


def hybrid_comparison(case: str) -> Path | None:
    if case not in set(_ordered_cases()):
        return None
    tube_path = HYBRID_ROOT / "gt_tubes" / case / "tube.npz"
    manifest = _json(tube_path.with_name("manifest.json"))
    source_video = Path(str(manifest.get("source_video") or ""))
    if not tube_path.is_file() or not source_video.is_file():
        return None
    destination = HYBRID_COMPARISON_ROOT / f"{case}.jpg"
    newest_input = max(tube_path.stat().st_mtime_ns, source_video.stat().st_mtime_ns)
    if destination.is_file() and destination.stat().st_mtime_ns >= newest_input:
        return destination
    with _hybrid_comparison_lock:
        if destination.is_file() and destination.stat().st_mtime_ns >= newest_input:
            return destination
        _build_hybrid_comparison(case, destination)
    return destination


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GT-STC Guidance · 错误预检</title><style>
:root{--lab:#e9edf0;--paper:#f8fafb;--ink:#14222b;--muted:#61717b;--line:#b9c4ca;--pass:#217a66;--pass-bg:#dff1eb;--fail:#a92f37;--fail-bg:#f7dde0;--risk:#a65b0b;--risk-bg:#f8ead1;--active:#2169a3;--active-bg:#dcecf8;--dark:#152d39}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:linear-gradient(90deg,#dfe5e8 1px,transparent 1px),linear-gradient(#dfe5e8 1px,transparent 1px),var(--lab);background-size:24px 24px;font-family:"Avenir Next","Noto Sans CJK SC","Microsoft YaHei",sans-serif}a{color:#175f75}header,main{width:min(1500px,calc(100% - 26px));margin:auto}header{padding:28px 0 18px}.crumb{font-size:13px}.eyebrow{margin-top:30px;font-family:ui-monospace,"SFMono-Regular",monospace;font-size:11px;font-weight:900;letter-spacing:.18em;color:var(--fail)}h1{font-family:"Arial Narrow","Noto Sans CJK SC",sans-serif;font-size:clamp(44px,8vw,104px);line-height:.82;letter-spacing:-.065em;margin:11px 0 22px;max-width:980px}.lead{max-width:970px;font-size:16px;line-height:1.7;color:#334750}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:20px}.toolbar button,.toolbar select{border:1px solid var(--line);background:var(--paper);padding:10px 13px;color:var(--ink);font-weight:800}.toolbar button{cursor:pointer;background:var(--dark);color:white}.updated{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted)}main{padding-bottom:80px}.summary{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:14px 0 24px}.metric{min-height:112px;background:var(--paper);border:1px solid var(--line);padding:14px}.metric b{display:block;font-family:"Arial Narrow",sans-serif;font-size:36px;line-height:1}.metric span{display:block;margin-top:12px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.metric.danger{border-top:5px solid var(--fail)}.metric.risk{border-top:5px solid var(--risk)}section{margin:28px 0}.section-head{display:flex;align-items:end;justify-content:space-between;gap:15px;border-bottom:2px solid var(--dark);padding-bottom:8px}h2{font-size:24px;margin:0}.note{font-size:12px;color:var(--muted)}.risk-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:12px}.risk-card{background:var(--fail-bg);border:1px solid #cb8b90;border-left:8px solid var(--fail);padding:16px}.risk-card.predicted{background:var(--risk-bg);border-color:#d8ae70;border-left-color:var(--risk)}.risk-card h3{font-size:15px;margin:0 0 8px;overflow-wrap:anywhere}.risk-card code{font-size:12px}.risk-card p{font-size:13px;line-height:1.55;margin:8px 0}.case-list{display:grid;gap:9px;margin-top:12px}.case{background:var(--paper);border:1px solid var(--line)}.case>summary{list-style:none;display:grid;grid-template-columns:88px minmax(240px,1fr) 140px 130px;gap:10px;align-items:center;padding:13px;cursor:pointer}.case>summary::-webkit-details-marker{display:none}.case[open]>summary{border-bottom:1px solid var(--line);background:#eef3f5}.case-name{font-family:ui-monospace,"SFMono-Regular",monospace;font-size:12px;overflow-wrap:anywhere}.worker{font-family:ui-monospace,monospace;font-size:11px;color:var(--muted)}.badge{display:inline-flex;align-items:center;justify-content:center;border-radius:99px;padding:5px 9px;font-size:10px;font-weight:900;letter-spacing:.05em;text-transform:uppercase}.complete{background:var(--pass-bg);color:var(--pass)}.failed,.failed_confirmed{background:var(--fail-bg);color:var(--fail)}.at_risk,.will_fail{background:var(--risk-bg);color:var(--risk)}.active,.active_or_interrupted{background:var(--active-bg);color:var(--active)}.queued{background:#e8ecee;color:#586871}.case-body{padding:14px;overflow:auto}.target-table{min-width:980px}.target-row{display:grid;grid-template-columns:155px 1fr 145px 145px;gap:10px;align-items:center;border-bottom:1px solid #dbe2e5;padding:10px 0}.target-row.head{font-size:10px;font-weight:900;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}.target-name{font-family:ui-monospace,monospace;font-size:12px;font-weight:900}.tape{display:grid;grid-template-columns:repeat(13,minmax(22px,1fr));gap:3px}.tick{height:34px;display:grid;place-items:center;border:1px solid #9eb6ad;background:var(--pass-bg);color:var(--pass);font:10px ui-monospace,monospace}.tick.empty{background:var(--fail-bg);border-color:#ca777e;color:var(--fail);font-weight:900}.mode{display:flex;gap:6px;align-items:center;font-size:11px}.mode small{color:var(--muted)}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11px;margin:10px 0;color:var(--muted)}.dot{display:inline-block;width:10px;height:10px;margin-right:5px}.dot.ok{background:var(--pass-bg);border:1px solid #75a997}.dot.bad{background:var(--fail-bg);border:1px solid #ca777e}.filmstrip-wrap{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}.filmstrip-wrap.cotracker{background:#e7eef1;padding:12px;border:1px solid #aebfc7}.filmstrip-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px}.filmstrip-head b{font-size:13px}.filmstrip-head span{font-size:11px;color:var(--muted)}.filmstrip-link{display:block;min-width:980px;background:#12252f}.filmstrip{display:block;width:100%;height:auto}.logs{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}.logs a{background:var(--paper);border:1px solid var(--line);padding:9px 12px;font:11px ui-monospace,monospace;text-decoration:none}.empty-state{padding:30px;border:1px solid var(--line);background:var(--paper)}@media(max-width:1000px){.summary{grid-template-columns:repeat(3,1fr)}.case>summary{grid-template-columns:70px 1fr 105px}.case>summary .worker{display:none}}@media(max-width:650px){header,main{width:calc(100% - 12px)}.summary{grid-template-columns:repeat(2,1fr)}.risk-grid{grid-template-columns:1fr}.case>summary{grid-template-columns:1fr auto}.case>summary .progress{display:none}h1{font-size:54px}}
</style></head><body><header><a class="crumb" href="/">← 返回 8092 总入口</a><div class="eyebrow">ORACLE TUBE / PREFLIGHT BOARD</div><h1>哪一条 Tube<br>会先断？</h1><p class="lead">这页在启动 Wan 前直接审计 13 个 latent 时刻的 SAM2 tube 和 CoTracker 可见性。红色 latent 格表示该时刻没有任何 region token；当前 Region loss 会立即抛错。未运行不等于报错，Point 与 Region 分开判定。下方逐帧胶片覆盖 guidance 实际使用的全部输入帧（长视频取前 49 帧，短视频使用全部帧）；每个 F 帧叠加最近的已保存 latent-anchor Region，白框标记精确 anchor，并显示 Fxx → Rxx 映射。</p><div class="toolbar"><button id="refresh">刷新现场</button><select id="filter"><option value="all">全部 case</option><option value="risk">只看报错/风险</option><option value="active">只看运行中</option><option value="complete">只看已完成</option></select><span id="updated" class="updated">读取中…</span></div></header><main><div id="summary" class="summary"></div><section><div class="section-head"><h2>明确错误与未来阻塞点</h2><span class="note">静态预检 + 实际运行状态</span></div><div id="risks" class="risk-grid"></div></section><section><div class="section-head"><h2>20 Case · 13 Latent Tube 审计</h2><span class="note">R00–R12 · token grid 44×80</span></div><div class="legend"><span><i class="dot ok"></i>该 latent 帧存在 Region token</span><span><i class="dot bad"></i>空 Region，Region/Combined 必报错</span></div><div id="cases" class="case-list"></div></section><section><div class="section-head"><h2>日志</h2><span class="note">白名单只读文本</span></div><div id="logs" class="logs"></div></section></main><script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null;
const labels={complete:'完成',failed:'已报错',at_risk:'将报错',active:'运行中/残留',queued:'待运行',failed_confirmed:'已复现报错',will_fail:'预检必报错',active_or_interrupted:'运行中/中断'};
function badge(state){return `<span class="badge ${state}">${labels[state]||state}</span>`}
function modeCell(mode){return `<div class="mode">${badge(mode.state)}<small>${mode.metric_ready?'指标完成':'指标未完成'}</small></div>`}
function renderSummary(d){const rows=[['Case',d.case_count,''],['GT tubes',`${d.tube_ready}/${d.case_count}`,''],['生成',`${d.completed_variants}/${d.expected_variants}`,''],['指标',d.metric_ready,''],['已复现错误',d.confirmed_failures,'danger'],['未来必报错',d.predicted_failures,'risk']];$('summary').innerHTML=rows.map(([k,v,c])=>`<div class="metric ${c}"><b>${v}</b><span>${k}</span></div>`).join('')}
function renderRisks(d){$('risks').innerHTML=d.risk_targets.length?d.risk_targets.map(r=>{const confirmed=r.region_state==='failed_confirmed';return `<article class="risk-card ${confirmed?'':'predicted'}"><h3>${esc(r.case)} / ${esc(r.target)} / Region</h3>${badge(r.region_state)}<p>空 latent：<code>${r.empty_frames.map(x=>'R'+String(x).padStart(2,'0')).join(', ')}</code></p><p>Region 在构造 <code>rows_by_time</code> 时会抛出 empty token-region。Point 仍有 <b>${r.point_pairs}</b> 个有效有序跨时刻对，不受这条错误影响。</p></article>`}).join(''):'<div class="empty-state">当前没有预检风险。</div>'}
function targetRow(t){const max=Math.max(...t.token_counts,1);const tape=t.token_counts.map((n,i)=>`<span class="tick ${n===0?'empty':''}" title="R${i}: ${n} tokens">${n===0?'×':Math.round(9*n/max)+1}</span>`).join('');return `<div class="target-row"><div class="target-name">${esc(t.name)}</div><div class="tape">${tape}</div>${modeCell(t.region)}${modeCell(t.point)}</div>`}
function filmstrip(c){const u=`/api/gt-stc-guidance-preflight/filmstrip?case=${encodeURIComponent(c.case)}`;const count=c.source_frame_count??'?';return `<div class="filmstrip-wrap"><div class="filmstrip-head"><b>Guidance 输入的全部 ${count} 帧 · Region overlay</b><span>${esc(c.source_frame_policy||'')} · 最近 latent anchor 投影；点击打开原尺寸</span></div><a class="filmstrip-link" href="${u}" target="_blank"><img class="filmstrip" loading="lazy" src="${u}" alt="${esc(c.case)} all guidance input frames with regions"></a></div>`}
function cotrackerComparison(c){const u=`/api/gt-stc-guidance-preflight/cotracker-comparison?case=${encodeURIComponent(c.case)}`;return `<div class="filmstrip-wrap cotracker"><div class="filmstrip-head"><b>SAM2 Region ↔ CoTracker point tube · 精确 R00–R12</b><span>左：mask；右：可见点与同点历史轨迹；M−/P&gt;0 表示 mask 消失但点仍可追踪</span></div><a class="filmstrip-link" href="${u}" target="_blank"><img class="filmstrip" loading="lazy" src="${u}" alt="${esc(c.case)} exact SAM2 and CoTracker anchor comparison"></a></div>`}
function hybridComparison(c){if(!c.hybrid_tube_ready)return '';const u=`/api/gt-stc-guidance-preflight/hybrid-comparison?case=${encodeURIComponent(c.case)}`;return `<div class="filmstrip-wrap cotracker"><div class="filmstrip-head"><b>CoTracker → SAM2 · Direct / Neighbor candidate / Final</b><span>蓝：当前帧真实点提示；橙：相邻帧传播测试；Final 仅在 Direct 不可用时采用橙色 fallback</span></div><a class="filmstrip-link" href="${u}" target="_blank"><img class="filmstrip" loading="lazy" src="${u}" alt="${esc(c.case)} direct neighbor fallback and final mask comparison"></a></div>`}
function renderCases(){const filter=$('filter').value;const rows=data.cases.filter(c=>filter==='all'||filter==='risk'&&['failed','at_risk'].includes(c.state)||filter==='active'&&c.state==='active'||filter==='complete'&&c.state==='complete');$('cases').innerHTML=rows.map(c=>`<details class="case" ${['failed','at_risk','active'].includes(c.state)?'open':''}><summary><span class="worker">W${c.worker} · #${String(c.order).padStart(2,'0')}</span><span class="case-name">${esc(c.case)}</span><span class="progress">${c.completed_variants}/${c.expected_variants} variants</span>${badge(c.state)}</summary><div class="case-body"><div class="target-table"><div class="target-row head"><div>Target</div><div>R00 → R12 token coverage</div><div>Region</div><div>Point</div></div>${c.targets.map(targetRow).join('')}</div><p class="note">Baseline：${badge(c.baseline.state)} · GT tube：${c.tube_ready?'ready':'missing'} · Hybrid v2：${c.hybrid_tube_ready?'ready':'等待测试'} · <code>${esc(c.source_video||'')}</code></p>${hybridComparison(c)}${cotrackerComparison(c)}${filmstrip(c)}</div></details>`).join('')||'<div class="empty-state">该筛选条件下没有 case。</div>'}
function renderLogs(d){$('logs').innerHTML=Object.entries(d.log_files).filter(([,v])=>v.ready).map(([k,v])=>`<a target="_blank" href="/api/gt-stc-guidance-preflight/log?name=${encodeURIComponent(k)}">${esc(k)} · ${esc(v.path)}</a>`).join('')}
async function refresh(){try{const r=await fetch('/api/gt-stc-guidance-preflight/catalog',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);data=await r.json();renderSummary(data);renderRisks(data);renderCases();renderLogs(data);$('updated').textContent=`UTC ${new Date(data.generated_at).toISOString().replace('T',' ').slice(0,19)} · 30 秒自动刷新`}catch(e){$('updated').textContent=`读取失败：${e}`}}
$('refresh').onclick=refresh;$('filter').onchange=renderCases;refresh();setInterval(refresh,30000);
</script></body></html>'''


def page() -> str:
    return PAGE
