#!/usr/bin/env python3
"""Build per-case SAM2 AMG object/background query caches for S-head motion analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_XSSC_DIR = SCRIPT_DIR.parent / "code_vjepa_vggt" / "train_xSSC"
DIFFTRACK_UTILS = Path("/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test")
SAM2_ROOT = Path("/home/gaoya/Grounded-SAM-2-main")
sys.path.insert(0, str(TRAIN_XSSC_DIR))
sys.path.insert(0, str(DIFFTRACK_UTILS))
sys.path.insert(0, str(SAM2_ROOT))

from sam2_region_query_utils import (  # noqa: E402
    QueryRegion,
    RegionQueryCache,
    erode_mask,
    farthest_point_sample,
    save_region_cache,
)
from visualize_movi_c_sam2_amg import (  # noqa: E402
    resolve_sam2_config_name,
    select_xssc_candidates,
)


DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/head-role-dose-control-pilot/manifest.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/cache/wan_dit_s_motion_sam2_regions"
)
DEFAULT_CONFIG = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt"
)
PHYCO_STATIC_TYPES = {
    "dome",
    "ground",
    "cube_platform",
    "wall",
    "pool_table",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-frame", type=int, default=7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--points-per-region", type=int, default=8)
    parser.add_argument("--max-selected", type=int, default=11)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def center_crop_to_aspect(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    target_aspect = target_width / target_height
    source_aspect = width / height
    if source_aspect > target_aspect:
        crop_width = max(1, round(height * target_aspect))
        left = (width - crop_width) // 2
        frame = frame[:, left : left + crop_width]
    else:
        crop_height = max(1, round(width / target_aspect))
        top = (height - crop_height) // 2
        frame = frame[top : top + crop_height]
    interpolation = (
        cv2.INTER_AREA
        if frame.shape[0] >= target_height and frame.shape[1] >= target_width
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)


def read_frame(path: Path, frame_index: int, height: int, width: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frame = None
    for _ in range(frame_index + 1):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{path} has fewer than {frame_index + 1} readable frames")
    capture.release()
    frame = center_crop_to_aspect(frame, width, height)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def phyco_instance_annotations(
    source_video: Path,
    frame_index: int,
    height: int,
    width: int,
) -> tuple[list[dict], Path, Path] | None:
    case_dir = source_video.resolve().parent
    segmentation_video = case_dir / "segmentation.mp4"
    metadata_path = case_dir / "metadata.json"
    if not segmentation_video.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    object_data = metadata.get("object_data", {})
    object_types = list(object_data.get("type", []))
    segmentation_ids = list(object_data.get("segmentation_id", []))
    color_map = metadata.get("segmentation_color_map", {})
    if len(object_types) != len(segmentation_ids) or not color_map:
        raise ValueError(f"Incomplete PhyCo segmentation metadata: {metadata_path}")

    segmentation_rgb = read_frame(
        segmentation_video,
        frame_index,
        height,
        width,
    )
    color_ids = sorted(int(value) for value in color_map)
    colors = np.asarray(
        [color_map[str(value)] for value in color_ids],
        dtype=np.int32,
    )
    pixels = segmentation_rgb.astype(np.int32)
    distances = np.square(
        pixels[:, :, None, :] - colors[None, None, :, :]
    ).sum(axis=3)
    labels = np.asarray(color_ids, dtype=np.int16)[distances.argmin(axis=2)]

    annotations = []
    for object_type, segmentation_id in zip(object_types, segmentation_ids):
        if str(object_type).lower() in PHYCO_STATIC_TYPES:
            continue
        mask = labels == int(segmentation_id)
        if int(mask.sum()) < 64:
            continue
        ys, xs = np.where(mask)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        annotations.append(
            {
                "segmentation": mask,
                "area": int(mask.sum()),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "predicted_iou": 1.0,
                "stability_score": 1.0,
                "region_name": str(object_type),
                "segmentation_id": int(segmentation_id),
            }
        )
    if not annotations:
        raise RuntimeError(
            f"{source_video}: no dynamic PhyCo instance masks at frame {frame_index}"
        )
    return annotations, segmentation_video, metadata_path


def selection_args(max_selected: int) -> Namespace:
    return Namespace(
        max_selected=max_selected,
        min_area_ratio=0.002,
        max_area_ratio=0.35,
        min_bbox_side=7.0,
        background_area_ratio=0.06,
        background_span_ratio=0.75,
        border_area_ratio=0.025,
        border_occupancy_ratio=0.18,
        opposite_edge_area_ratio=0.04,
        shadow_min_area_ratio=0.005,
        shadow_max_luminance_ratio=0.55,
        shadow_max_chromaticity_distance=0.10,
        shadow_max_gradient_mean=20.0,
        duplicate_iou=0.70,
        duplicate_containment=0.85,
    )


def build_cache(
    case_id: str,
    frame: np.ndarray,
    selected: list[dict],
    raw_count: int,
    query_frame: int,
    points_per_region: int,
    context_video: Path,
    region_method: str,
) -> RegionQueryCache:
    height, width = frame.shape[:2]
    assigned = np.zeros((height, width), dtype=bool)
    object_masks: list[np.ndarray] = []
    object_metadata: list[dict] = []
    for annotation in selected:
        raw_mask = np.asarray(annotation["segmentation"], dtype=bool)
        exclusive = raw_mask & ~assigned
        assigned |= raw_mask
        mask = erode_mask(exclusive, 7)
        if mask.sum() < points_per_region:
            mask = exclusive
        if mask.sum() < points_per_region:
            continue
        object_masks.append(mask)
        object_metadata.append(
            {
                "area": int(annotation["area"]),
                "bbox_xywh": [float(value) for value in annotation["bbox"]],
                "predicted_iou": float(annotation["predicted_iou"]),
                "stability_score": float(annotation["stability_score"]),
                "region_name": annotation.get("region_name"),
                "segmentation_id": annotation.get("segmentation_id"),
            }
        )
    if not object_masks:
        raise RuntimeError(f"{case_id}: SAM2 AMG produced no usable object masks")

    object_union = np.logical_or.reduce(object_masks)
    background = erode_mask(~object_union, 31)
    border = 32
    background[:border] = False
    background[-border:] = False
    background[:, :border] = False
    background[:, -border:] = False
    if background.sum() < points_per_region:
        raise RuntimeError(f"{case_id}: background has too few valid pixels")

    query_parts: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    regions: list[QueryRegion] = []
    offset = 0
    for index, mask in enumerate(object_masks):
        points = farthest_point_sample(mask, points_per_region)
        query_parts.append(points)
        masks.append(mask)
        regions.append(
            QueryRegion(
                region_name=f"object_{index:02d}",
                region_type="object",
                region_phrase=object_metadata[index].get("region_name"),
                region_slot=index,
                point_start=offset,
                point_end=offset + len(points),
                mask_area=int(mask.sum()),
                source_mask_frame=query_frame,
                used_frame_fallback=False,
            )
        )
        offset += len(points)

    background_points = farthest_point_sample(background, points_per_region)
    query_parts.append(background_points)
    masks.append(background)
    regions.append(
        QueryRegion(
            region_name="background",
            region_type="background",
            region_phrase=None,
            region_slot=None,
            point_start=offset,
            point_end=offset + len(background_points),
            mask_area=int(background.sum()),
            source_mask_frame=query_frame,
            used_frame_fallback=False,
        )
    )
    return RegionQueryCache(
        case_key=case_id,
        query_points=np.concatenate(query_parts).astype(np.float32),
        masks_rhw=np.stack(masks).astype(np.uint8),
        regions=regions,
        context_frame_rgb=frame,
        metadata={
            "schema_version": 2,
            "case_key": case_id,
            "height": height,
            "width": width,
            "query_context_frame": query_frame,
            "points_per_region": points_per_region,
            "object_erode_px": 7,
            "background_erode_px": 31,
            "object_count": len(object_masks),
            "region_count": len(regions),
            "source_context_video": str(context_video.resolve()),
            "spatial_preprocess": "center_crop_to_7:4_then_resize",
            "region_method": region_method,
            "selection_profile": (
                "phyco_metadata_dynamic_instances_v2"
                if region_method == "phyco_gt_instance_segmentation"
                else "xssc_amg_shadow005_v1"
            ),
            "raw_mask_count": raw_count,
            "selected_mask_count": len(object_masks),
            "selected_annotations": object_metadata,
        },
    )


def cache_is_complete(
    cache_dir: Path,
    query_frame: int,
    expected_method: str,
) -> bool:
    metadata_path = cache_dir / "regions.json"
    arrays_path = cache_dir / "regions.npz"
    if not metadata_path.is_file() or not arrays_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("schema_version") == 2
        and metadata.get("query_context_frame") == query_frame
        and metadata.get("region_method") == expected_method
        and metadata.get("selection_profile")
        == (
            "phyco_metadata_dynamic_instances_v2"
            if expected_method == "phyco_gt_instance_segmentation"
            else "xssc_amg_shadow005_v1"
        )
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = payload["cases"]
    if args.case_id:
        selected_ids = set(args.case_id)
        cases = [case for case in cases if case["id"] in selected_ids]
        missing = selected_ids - {case["id"] for case in cases}
        if missing:
            raise KeyError(f"Unknown case IDs: {sorted(missing)}")
    if not cases:
        raise RuntimeError("No cases selected")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(
        resolve_sam2_config_name(args.sam2_config),
        str(args.sam2_checkpoint.resolve()),
        device=str(device),
        mode="eval",
    )
    generator = SAM2AutomaticMaskGenerator(model)
    select_args = selection_args(args.max_selected)

    completed = 0
    reused = 0
    for index, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        output_dir = args.output_root / case_id
        source_video = Path(case["source_video"])
        phyco_annotations = phyco_instance_annotations(
            source_video,
            args.query_frame,
            args.height,
            args.width,
        )
        expected_method = (
            "phyco_gt_instance_segmentation"
            if phyco_annotations is not None
            else "sam2_amg_filtered"
        )
        if not args.overwrite and cache_is_complete(
            output_dir,
            args.query_frame,
            expected_method,
        ):
            reused += 1
            print(f"[{index}/{len(cases)}] reuse {case_id}", flush=True)
            continue
        context_video = Path(case["context_video"])
        frame = read_frame(
            context_video,
            args.query_frame,
            args.height,
            args.width,
        )
        started = time.time()
        if phyco_annotations is None:
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                annotations = generator.generate(frame)
            selected = select_xssc_candidates(
                annotations,
                frame.shape[0] * frame.shape[1],
                select_args,
                image=frame,
            )
        else:
            selected, segmentation_video, metadata_path = phyco_annotations
            annotations = selected
        cache = build_cache(
            case_id,
            frame,
            selected,
            len(annotations),
            args.query_frame,
            args.points_per_region,
            context_video,
            expected_method,
        )
        cache.metadata["elapsed_seconds"] = time.time() - started
        cache.metadata["sam2_config"] = str(args.sam2_config.resolve())
        cache.metadata["sam2_checkpoint"] = str(args.sam2_checkpoint.resolve())
        if phyco_annotations is not None:
            cache.metadata["segmentation_video"] = str(segmentation_video.resolve())
            cache.metadata["phyco_metadata"] = str(metadata_path.resolve())
        save_region_cache(output_dir, cache)
        completed += 1
        print(
            f"[{index}/{len(cases)}] complete {case_id} "
            f"raw={len(annotations)} objects={len(cache.regions) - 1} "
            f"seconds={cache.metadata['elapsed_seconds']:.1f}",
            flush=True,
        )
    print(f"[regions] complete={completed} reused={reused} root={args.output_root}")


if __name__ == "__main__":
    main()
