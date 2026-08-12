#!/usr/bin/env python3
"""Oracle GT-tube correspondence guidance for legacy Wan2.2 TI2V.

This is an inference-time diagnostic, not training.  GroundingDINO + SAM2
initialize object points, CoTracker transports those points through the source
video, and per-anchor SAM2 point prompts recover a 13-latent object mask tube.
A neighboring-frame SAM2 propagation is always saved for audit, but is applied
only when the same-frame tracked-point prompt is unavailable or empty.  During
each conditional Wan forward, latest3350 Top100 self-attention heads contribute
a cross-time correspondence loss.  Only the current noisy latent receives
gradients; every model parameter remains frozen.

Stages are intentionally separable because SAM2/CoTracker and gradient-enabled
Wan inference have different memory profiles:

  prepare   source video -> CoTracker points -> direct/fallback/final SAM2 tubes
  generate  baseline/region/point guided videos
  evaluate  generated videos -> GT-relative trajectory metrics
  all       prepare, generate, then evaluate (models are released in between)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for import_root in (ROOT, CODE_ROOT, COTRACKER_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (  # noqa: E402
    build_wan_ti2v_pipeline,
    save_video_np,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8  # noqa: E402
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (  # noqa: E402
    DetectedObjectTrack,
)
from AAA_my_test.precompute_legacy_ti2v_firstlatent_physiciq67_regions import (  # noqa: E402
    filter_overlapping_query_tracks,
)
from AAA_my_test.precompute_toydataset_sam2_regions import (  # noqa: E402
    build_provider,
    detect_and_track_objects,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (  # noqa: E402
    build_args,
    load_cotracker,
    run_cotracker,
)
from AAA_my_test.sam2_region_query_utils import (  # noqa: E402
    build_regions_from_grounding,
)


DEFAULT_INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_RANKING = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/"
    "latest3350_top100_cotracker_sam2_v2"
)
OBJECT_PHRASE_CACHE_ROOTS = (
    Path("/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_regions_704x1280"),
    Path(
        "/data/gaoya/agent-data/cache/"
        "wan22_ti2v_legacy_firstlatent_physiciq67_regions_704x1280"
    ),
    Path(
        "/data/gaoya/agent-data/cache/"
        "wan22_ti2v_legacy_attention_zero_seed47326_regions_704x1280"
    ),
)
SEGMENTATION_PROMPT_CACHE_ROOTS = (
    Path("/data/gaoya/agent-data/cache/wan_dit_s_motion_sam2_regions"),
)
PROTOCOL = "wan_gt_spatiotemporal_correspondence_guidance_v2"
HEIGHT = 704
WIDTH = 1280
PIXEL_FRAMES = 49
LATENT_FRAMES = 13
LATENT_PIXEL_ANCHORS = np.arange(LATENT_FRAMES, dtype=np.int64) * 4
HEAD_DIM = 128


@dataclass(frozen=True)
class FrozenTube:
    case: str
    source_json: Path
    source_video: Path
    source_frame_count: int
    anchor_source_frames: np.ndarray
    masks_othw: np.ndarray
    tracks_tn2: np.ndarray
    visibility_tn: np.ndarray
    query_points_n2: np.ndarray
    region_names: tuple[str, ...]
    point_starts: np.ndarray
    point_ends: np.ndarray
    moving: np.ndarray
    pixel_height: int
    pixel_width: int


@dataclass(frozen=True)
class GuidanceTarget:
    name: str
    object_indices: tuple[int, ...]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def cached_object_prompt_spec(
    case: str, cache_roots: Iterable[Path] = OBJECT_PHRASE_CACHE_ROOTS
) -> tuple[list[str], np.ndarray | None, np.ndarray | None, Path | None]:
    """Load validated phrases and optional first-frame prompt boxes, not tracks."""
    for cache_root in cache_roots:
        metadata_path = Path(cache_root) / case / "regions.json"
        if not metadata_path.is_file():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        phrases = [
            str(value).strip()
            for value in payload.get("object_phrases", [])
            if str(value).strip()
        ]
        if not phrases:
            phrases = [
                str(region.get("region_phrase") or "").strip()
                for region in payload.get("regions", [])
                if region.get("region_type") == "object"
                and str(region.get("region_phrase") or "").strip()
            ]
        if not phrases:
            continue
        debug = dict(payload.get("grounding_debug") or {})
        boxes = np.asarray(debug.get("object_prompt_boxes_xyxy") or [], dtype=np.float32)
        scores = np.asarray(debug.get("object_scores") or [], dtype=np.float32)
        if boxes.shape != (len(phrases), 4):
            boxes = None
            scores = None
        else:
            if scores.shape != (len(phrases),):
                scores = np.ones((len(phrases),), dtype=np.float32)
        return phrases, boxes, scores, metadata_path
    return [], None, None, None


def cached_object_phrases(
    case: str, cache_roots: Iterable[Path] = OBJECT_PHRASE_CACHE_ROOTS
) -> tuple[list[str], Path | None]:
    """Compatibility wrapper returning only semantic prompts and provenance."""
    phrases, _, _, metadata_path = cached_object_prompt_spec(case, cache_roots)
    return phrases, metadata_path


def cached_segmentation_prompt_spec(
    case: str,
    cache_roots: Iterable[Path] = SEGMENTATION_PROMPT_CACHE_ROOTS,
    target_hw: tuple[int, int] = (HEIGHT, WIDTH),
) -> tuple[list[str], np.ndarray | None, np.ndarray | None, int, Path | None]:
    """Load validated segmentation-derived boxes for a fresh full-video SAM2 run."""
    target_height, target_width = (int(value) for value in target_hw)
    for cache_root in cache_roots:
        metadata_path = Path(cache_root) / case / "regions.json"
        if not metadata_path.is_file():
            continue
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_height = int(payload.get("height") or 0)
        source_width = int(payload.get("width") or 0)
        annotations = list(payload.get("selected_annotations") or [])
        if source_height <= 0 or source_width <= 0 or not annotations:
            continue
        phrases: list[str] = []
        boxes: list[list[float]] = []
        scores: list[float] = []
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        for index, annotation in enumerate(annotations):
            bbox = annotation.get("bbox_xywh")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = (float(value) for value in bbox)
            if width <= 0 or height <= 0:
                continue
            phrases.append(str(annotation.get("region_name") or f"object_{index:02d}"))
            boxes.append(
                [
                    x * scale_x,
                    y * scale_y,
                    (x + width) * scale_x,
                    (y + height) * scale_y,
                ]
            )
            scores.append(float(annotation.get("predicted_iou") or 1.0))
        if phrases:
            return (
                phrases,
                np.asarray(boxes, dtype=np.float32),
                np.asarray(scores, dtype=np.float32),
                int(payload.get("query_context_frame") or 0),
                metadata_path,
            )
    return [], None, None, 0, None


def track_objects_from_prompt_boxes(
    provider: Any,
    frames_tchw_01: np.ndarray,
    phrases: list[str],
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    prompt_frame_idx: int = 0,
) -> SimpleNamespace:
    """Re-run SAM2 over the current full source video from validated frame-0 boxes."""
    tracks = []
    for phrase, box, score in zip(phrases, boxes_xyxy, scores):
        sam_output = provider.tracker.track(
            frames_tchw_01,
            prompt_frame_idx=int(prompt_frame_idx),
            prompt_box_xyxy=np.asarray(box, dtype=np.float32),
            caption="",
        )
        masks = np.asarray(sam_output.masks_thw, dtype=np.uint8)
        if not masks.any():
            raise RuntimeError(f"SAM2 produced an empty cached-box track for {phrase!r}")
        tracks.append(
            DetectedObjectTrack(
                box_prompt_xyxy=np.asarray(box, dtype=np.float32),
                masks_thw=masks,
                boxes_t4=np.asarray(sam_output.boxes_t4, dtype=np.float32),
                score=float(score),
                phrase=str(phrase),
                source_phrase=str(phrase),
            )
        )
    return SimpleNamespace(
        object_tracks=tracks,
        debug={
            "mode": "validated_frame0_boxes_then_full_source_sam2",
            "object_phrases": list(phrases),
            "object_prompt_boxes_xyxy": np.asarray(boxes_xyxy).tolist(),
            "object_scores": np.asarray(scores).tolist(),
            "prompt_frame_idx": int(prompt_frame_idx),
        },
    )


def deduplicated_json_paths(input_list: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for line in input_list.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(path)
        paths.append(path)
    if not paths:
        raise RuntimeError(f"no JSON cases found in {input_list}")
    return paths


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    for key in ("source_video", "input_caption"):
        if not str(payload.get(key) or "").strip():
            raise ValueError(f"{path}: missing {key}")
    return payload


def read_source_prefix(source_video: Path, max_frames: int = PIXEL_FRAMES) -> np.ndarray:
    frames = iio.imread(source_video, index=None)
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise RuntimeError(f"unexpected source video shape {frames.shape}: {source_video}")
    frames = frames[..., :3]
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    return frames[: int(max_frames)]


def resize_frames(frames: np.ndarray, height: int = HEIGHT, width: int = WIDTH) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return tensor.round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()


def source_anchors(frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("source video has no frames")
    if frame_count >= PIXEL_FRAMES:
        return LATENT_PIXEL_ANCHORS.copy()
    return np.rint(np.linspace(0, frame_count - 1, LATENT_FRAMES)).astype(np.int64)


def tube_dir(output_root: Path, case: str) -> Path:
    return output_root / "gt_tubes" / case


def load_frozen_tube(output_root: Path, case: str) -> FrozenTube:
    root = tube_dir(output_root, case)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    with np.load(root / "tube.npz", allow_pickle=False) as data:
        return FrozenTube(
            case=case,
            source_json=Path(manifest["source_json"]),
            source_video=Path(manifest["source_video"]),
            source_frame_count=int(manifest["source_frame_count"]),
            anchor_source_frames=data["anchor_source_frames"].astype(np.int64),
            masks_othw=data["masks_othw"].astype(bool),
            tracks_tn2=data["tracks_tn2"].astype(np.float32),
            visibility_tn=data["visibility_tn"].astype(bool),
            query_points_n2=data["query_points_n2"].astype(np.float32),
            region_names=tuple(str(value) for value in data["region_names"].tolist()),
            point_starts=data["point_starts"].astype(np.int64),
            point_ends=data["point_ends"].astype(np.int64),
            moving=data["moving"].astype(bool),
            pixel_height=int(data["pixel_height"]),
            pixel_width=int(data["pixel_width"]),
        )


def motion_scores_d0(
    tracks_tn2: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    masks_othw: np.ndarray,
) -> np.ndarray:
    """Robust maximum median displacement normalized by first-mask bbox diagonal."""
    scores = []
    for object_index, (start, end) in enumerate(zip(starts, ends)):
        points = tracks_tn2[:, int(start) : int(end)]
        displacement = np.linalg.norm(points - points[0:1], axis=-1)
        robust_displacement = float(np.max(np.median(displacement, axis=1)))
        yx = np.argwhere(masks_othw[object_index, 0])
        if len(yx):
            height = float(yx[:, 0].max() - yx[:, 0].min() + 1)
            width = float(yx[:, 1].max() - yx[:, 1].min() + 1)
            diagonal = math.hypot(width, height)
        else:
            diagonal = 1.0
        scores.append(robust_displacement / max(diagonal, 1.0))
    return np.asarray(scores, dtype=np.float32)


def prepare_case(
    json_path: Path,
    output_root: Path,
    device: str,
    points_per_object: int,
    moving_threshold_d0: float,
    tube_mask_strategy: str,
    overwrite: bool,
) -> None:
    case = json_path.stem
    output = tube_dir(output_root, case)
    required = (output / "tube.npz", output / "manifest.json", output / "complete.json")
    if all(path.is_file() for path in required) and not overwrite:
        print(f"[prepare] skip {case}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)
    payload = load_payload(json_path)
    source_video = Path(str(payload["source_video"])).expanduser().resolve()
    native_frames = read_source_prefix(source_video)
    frame_count = int(len(native_frames))
    frames = resize_frames(native_frames)
    frames_tchw_01 = frames.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

    provider = build_provider(device, points_per_object)
    prompt_phrases, prompt_boxes, prompt_scores, phrase_cache = cached_object_prompt_spec(case)
    if prompt_phrases:
        print(
            f"[prepare] {case}: explicit object prompts={prompt_phrases} "
            f"from {phrase_cache}",
            flush=True,
        )
        try:
            sample = detect_and_track_objects(provider, frames_tchw_01, prompt_phrases)
            prompt_source = f"validated phrases, fresh GroundingDINO boxes: {phrase_cache}"
        except RuntimeError as error:
            if (
                "could not assign distinct GroundingDINO boxes" not in str(error)
                or prompt_boxes is None
                or prompt_scores is None
            ):
                raise
            print(
                f"[prepare] {case}: strict fresh box assignment failed; "
                f"re-running full-source SAM2 from validated frame-0 boxes",
                flush=True,
            )
            sample = track_objects_from_prompt_boxes(
                provider,
                frames_tchw_01,
                prompt_phrases,
                prompt_boxes,
                prompt_scores,
            )
            prompt_source = f"validated phrases and frame-0 boxes: {phrase_cache}"
    else:
        sample = provider.build_sample(
            frames_tchw_01=frames_tchw_01,
            caption=str(payload["input_caption"]),
            image_hw=(HEIGHT, WIDTH),
        )
        prompt_source = "automatic physical noun phrases from input_caption"
    try:
        sample = filter_overlapping_query_tracks(
            sample, query_frame=0, min_pixels=points_per_object
        )
    except RuntimeError as error:
        if "automatic grounding produced no distinct object tracks" not in str(error):
            raise
        (
            segmentation_phrases,
            segmentation_boxes,
            segmentation_scores,
            segmentation_frame,
            segmentation_cache,
        ) = cached_segmentation_prompt_spec(case)
        if segmentation_boxes is None or segmentation_scores is None:
            raise
        print(
            f"[prepare] {case}: automatic tracks were not distinct; "
            f"re-running full-source SAM2 from validated segmentation boxes "
            f"at source frame {segmentation_frame}",
            flush=True,
        )
        sample = track_objects_from_prompt_boxes(
            provider,
            frames_tchw_01,
            segmentation_phrases,
            segmentation_boxes,
            segmentation_scores,
            prompt_frame_idx=segmentation_frame,
        )
        sample = filter_overlapping_query_tracks(
            sample, query_frame=0, min_pixels=points_per_object
        )
        prompt_source = (
            "validated segmentation boxes followed by fresh full-source SAM2: "
            f"{segmentation_cache}"
        )
    phrases = [
        str(track.source_phrase or track.phrase or f"object_{index:02d}")
        for index, track in enumerate(sample.object_tracks)
    ]
    if not phrases:
        raise RuntimeError(f"{case}: no source objects found")
    cache = build_regions_from_grounding(
        case_key=case,
        grounding_sample=sample,
        object_phrases=phrases,
        context_frame_rgb=frames[0],
        query_frame_index=0,
        points_per_region=points_per_object,
        object_erode_px=11,
        background_erode_px=31,
    )
    object_regions = [region for region in cache.regions if region.region_type == "object"]
    query_parts = [
        cache.query_points[region.point_start : region.point_end]
        for region in object_regions
    ]
    query_points = np.concatenate(query_parts).astype(np.float32)
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for points in query_parts:
        starts.append(cursor)
        cursor += len(points)
        ends.append(cursor)

    # Do not co-reside GroundingDINO/SAM2 and CoTracker on the GPU.  The
    # propagated masks and sampled points above are plain CPU arrays.
    del provider
    gc.collect()
    torch.cuda.empty_cache()
    cotracker = load_cotracker(device)
    try:
        tracks, visibility = run_cotracker(cotracker, frames, query_points, device)
    finally:
        del cotracker
        gc.collect()
        torch.cuda.empty_cache()
    tracks = np.asarray(tracks, dtype=np.float32)
    visibility = np.asarray(visibility, dtype=bool)
    in_bounds = (
        np.isfinite(tracks).all(axis=-1)
        & (tracks[..., 0] >= 0.0)
        & (tracks[..., 0] < float(WIDTH))
        & (tracks[..., 1] >= 0.0)
        & (tracks[..., 1] < float(HEIGHT))
    )
    visibility &= in_bounds
    anchors = source_anchors(frame_count)
    tracks_anchor = tracks[anchors]
    visibility_anchor = visibility[anchors]
    raw_masks = np.stack(
        [np.asarray(track.masks_thw, dtype=np.uint8) for track in sample.object_tracks]
    )
    legacy_masks_anchor = raw_masks[:, anchors]
    object_count = len(phrases)
    if tube_mask_strategy == "cotracker_prompted_sam2":
        print(
            f"[prepare] {case}: CoTracker points -> per-anchor SAM2 direct masks; "
            "neighbor propagation is audit-only unless direct fails",
            flush=True,
        )
        hybrid_provider = build_provider(device, points_per_object)
        try:
            point_prompted = [
                hybrid_provider.tracker.segment_tracked_point_tube(
                    frames_tchw_01,
                    anchors,
                    tracks[:, int(start) : int(end)],
                    visibility[:, int(start) : int(end)],
                )
                for start, end in zip(starts, ends)
            ]
        finally:
            del hybrid_provider
            gc.collect()
            torch.cuda.empty_cache()
        direct_masks_anchor = np.stack(
            [result.direct_masks_ahw for result in point_prompted]
        ).astype(np.uint8)
        neighbor_masks_anchor = np.stack(
            [result.neighbor_masks_ahw for result in point_prompted]
        ).astype(np.uint8)
        masks_anchor = np.stack(
            [result.final_masks_ahw for result in point_prompted]
        ).astype(np.uint8)
        direct_prompt_counts = np.stack(
            [result.direct_prompt_counts_a for result in point_prompted]
        ).astype(np.int16)
        neighbor_source_anchor = np.stack(
            [result.neighbor_source_anchor_a for result in point_prompted]
        ).astype(np.int16)
        final_mask_source = np.stack(
            [result.final_source_a for result in point_prompted]
        ).astype(np.uint8)
    elif tube_mask_strategy == "legacy_sam2_propagation":
        masks_anchor = legacy_masks_anchor.astype(np.uint8)
        direct_masks_anchor = legacy_masks_anchor.astype(np.uint8)
        neighbor_masks_anchor = np.zeros_like(direct_masks_anchor)
        direct_prompt_counts = np.zeros(
            (object_count, len(anchors)), dtype=np.int16
        )
        neighbor_source_anchor = np.full(
            (object_count, len(anchors)), -1, dtype=np.int16
        )
        final_mask_source = np.where(
            masks_anchor.reshape(object_count, len(anchors), -1).any(axis=2),
            0,
            2,
        ).astype(np.uint8)
    else:
        raise ValueError(f"unknown tube mask strategy: {tube_mask_strategy}")
    scores = motion_scores_d0(
        tracks_anchor,
        np.asarray(starts),
        np.asarray(ends),
        masks_anchor,
    )
    moving = scores >= float(moving_threshold_d0)
    # A detector can occasionally return only one valid moving object with a
    # score just below threshold. Keep per-object targets regardless; the union
    # is omitted when fewer than two objects pass this explicit audit rule.
    names = tuple(f"object_{chr(ord('A') + index)}" for index in range(len(phrases)))
    atomic_npz(
        output / "tube.npz",
        anchor_source_frames=anchors,
        masks_othw=masks_anchor.astype(np.uint8),
        direct_masks_othw=direct_masks_anchor.astype(np.uint8),
        neighbor_masks_othw=neighbor_masks_anchor.astype(np.uint8),
        legacy_masks_othw=legacy_masks_anchor.astype(np.uint8),
        direct_prompt_counts_ot=direct_prompt_counts,
        neighbor_source_anchor_ot=neighbor_source_anchor,
        final_mask_source_ot=final_mask_source,
        tracks_tn2=tracks_anchor.astype(np.float32),
        visibility_tn=visibility_anchor.astype(np.uint8),
        query_points_n2=query_points.astype(np.float32),
        region_names=np.asarray(names),
        point_starts=np.asarray(starts, dtype=np.int32),
        point_ends=np.asarray(ends, dtype=np.int32),
        moving=moving.astype(np.uint8),
        motion_score_d0=scores,
        pixel_height=np.int32(HEIGHT),
        pixel_width=np.int32(WIDTH),
    )
    manifest = {
        "protocol": PROTOCOL,
        "case": case,
        "source_json": str(json_path),
        "source_video": str(source_video),
        "source_frame_count": frame_count,
        "source_frame_policy": (
            "first 49 source frames"
            if frame_count >= PIXEL_FRAMES
            else "all source frames; linearly mapped to 13 latent anchors"
        ),
        "anchor_source_frames": anchors.tolist(),
        "latent_anchor_indices": list(range(LATENT_FRAMES)),
        "source_processing": (
            "GroundingDINO/SAM2 initialization -> CoTracker points -> independent "
            "per-anchor SAM2 point prompts; nearest neighboring direct mask is "
            "propagated for audit and used only when direct is unavailable/empty"
            if tube_mask_strategy == "cotracker_prompted_sam2"
            else "GroundingDINO -> full-video SAM2 propagation; first-mask points -> CoTracker"
        ),
        "tube_mask_strategy": tube_mask_strategy,
        "final_mask_source_codes": {
            "0": "direct CoTracker points prompted SAM2 at the same anchor",
            "1": "neighboring direct mask propagated by SAM2 fallback",
            "2": "missing direct and fallback mask",
        },
        "object_prompt_source": prompt_source,
        "objects": [
            {
                "name": name,
                "phrase": phrase,
                "point_start": int(start),
                "point_end": int(end),
                "motion_score_d0": float(score),
                "moving": bool(is_moving),
                "anchor_visibility_rate": float(
                    visibility_anchor[:, int(start) : int(end)].mean()
                ),
                "direct_prompt_counts": direct_prompt_counts[object_index].astype(int).tolist(),
                "direct_mask_frames": np.flatnonzero(
                    direct_masks_anchor[object_index].reshape(len(anchors), -1).any(axis=1)
                ).astype(int).tolist(),
                "fallback_candidate_frames": np.flatnonzero(
                    neighbor_masks_anchor[object_index].reshape(len(anchors), -1).any(axis=1)
                ).astype(int).tolist(),
                "fallback_applied_frames": np.flatnonzero(
                    final_mask_source[object_index] == 1
                ).astype(int).tolist(),
                "missing_final_frames": np.flatnonzero(
                    final_mask_source[object_index] == 2
                ).astype(int).tolist(),
                "neighbor_source_anchor": neighbor_source_anchor[object_index].astype(int).tolist(),
            }
            for object_index, (name, phrase, start, end, score, is_moving) in enumerate(zip(
                names, phrases, starts, ends, scores, moving
            ))
        ],
        "moving_threshold_d0": float(moving_threshold_d0),
        "uses_gt_instance_masks": False,
        "oracle_information": "future source-video CoTracker trajectories and point-prompted SAM2 masks",
    }
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(
        output / "complete.json",
        {"case": case, "object_count": len(names), "moving_count": int(moving.sum())},
    )
    (output / "error.txt").unlink(missing_ok=True)
    print(
        f"[prepare] complete {case}: objects={len(names)} moving={int(moving.sum())}",
        flush=True,
    )


def target_specs(tube: FrozenTube) -> list[GuidanceTarget]:
    targets = [GuidanceTarget(name, (index,)) for index, name in enumerate(tube.region_names)]
    moving_indices = tuple(int(index) for index in np.flatnonzero(tube.moving))
    if len(moving_indices) >= 2:
        targets.append(GuidanceTarget("moving_union", moving_indices))
    return targets


def selected_target_specs(
    tube: FrozenTube, target_names: tuple[str, ...] | None
) -> list[GuidanceTarget]:
    targets = target_specs(tube)
    if not target_names:
        return targets
    requested = set(target_names)
    available = {target.name for target in targets}
    unknown = requested - available
    if unknown:
        raise ValueError(
            f"unknown target names for {tube.case}: {sorted(unknown)}; "
            f"available={sorted(available)}"
        )
    return [target for target in targets if target.name in requested]


def load_target_map(path: Path) -> dict[str, tuple[str, ...]]:
    """Load a frozen case -> target-name map from screening JSON."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "eligible_jobs" in payload:
        rows = payload["eligible_jobs"]
        mapping = {
            str(row["case"]): tuple(str(value) for value in row["targets"])
            for row in rows
        }
    else:
        mapping = {
            str(case): tuple(str(value) for value in targets)
            for case, targets in payload.items()
        }
    if any(not targets for targets in mapping.values()):
        raise ValueError(f"target map contains an empty target list: {path}")
    return mapping


def target_names_for_case(
    case: str,
    global_target_names: tuple[str, ...] | None,
    target_map: dict[str, tuple[str, ...]] | None,
) -> tuple[str, ...] | None:
    if target_map is not None:
        if case not in target_map:
            raise ValueError(f"target map has no registered targets for case: {case}")
        return target_map[case]
    return global_target_names


def validate_latest3350_top100(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = list(payload.get("entries") or [])
    pairs = [(int(row["block"]), int(row["head"])) for row in entries]
    expected = {(block, head) for block in range(30) for head in range(24)}
    if len(entries) != 720 or len(set(pairs)) != 720 or set(pairs) != expected:
        raise RuntimeError(f"invalid 30x24 head ranking: {path}")
    top100 = entries[:100]
    if any(int(row.get("step", 39)) != 39 for row in top100):
        raise RuntimeError("latest3350 Top100 is expected to be the S039 ranking")
    return top100


def selected_object_arrays(
    tube: FrozenTube, target: GuidanceTarget
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.logical_or.reduce(tube.masks_othw[list(target.object_indices)], axis=0)
    point_indices = np.concatenate(
        [
            np.arange(tube.point_starts[index], tube.point_ends[index], dtype=np.int64)
            for index in target.object_indices
        ]
    )
    return mask, tube.tracks_tn2[:, point_indices], tube.visibility_tn[:, point_indices]


def masks_to_token_rows(masks_thw: np.ndarray, token_hw: tuple[int, int]) -> list[torch.Tensor]:
    masks = torch.from_numpy(np.asarray(masks_thw, dtype=np.float32)).unsqueeze(1)
    down = F.adaptive_max_pool2d(masks, token_hw).squeeze(1) > 0
    return [torch.nonzero(frame.flatten(), as_tuple=False).flatten() for frame in down]


def points_to_token_rows(
    tracks_tn2: np.ndarray,
    pixel_hw: tuple[int, int],
    token_hw: tuple[int, int],
) -> torch.Tensor:
    pixel_height, pixel_width = pixel_hw
    token_height, token_width = token_hw
    tracks = torch.from_numpy(np.asarray(tracks_tn2, dtype=np.float32))
    x = torch.floor(tracks[..., 0] * token_width / pixel_width).long().clamp(0, token_width - 1)
    y = torch.floor(tracks[..., 1] * token_height / pixel_height).long().clamp(0, token_height - 1)
    return y * token_width + x


def region_correspondence_loss(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    rows_by_time: list[torch.Tensor],
    token_hw: tuple[int, int],
) -> tuple[torch.Tensor, int]:
    """Cross-time negative log probability mass assigned to the target GT mask."""
    token_height, token_width = token_hw
    frame_tokens = token_height * token_width
    time_count = len(rows_by_time)
    q = q_bshd.view(q_bshd.shape[0], time_count, frame_tokens, q_bshd.shape[2], q_bshd.shape[3])
    k = k_bshd.view(k_bshd.shape[0], time_count, frame_tokens, k_bshd.shape[2], k_bshd.shape[3])
    scale = math.sqrt(q_bshd.shape[-1])
    terms: list[torch.Tensor] = []
    for query_time in range(time_count):
        query_rows = rows_by_time[query_time].to(q.device)
        if not query_rows.numel():
            continue
        query_vectors = q[:, query_time, query_rows].float()
        for key_time in range(time_count):
            if key_time == query_time:
                continue
            target_rows = rows_by_time[key_time].to(q.device)
            if not target_rows.numel():
                continue
            logits = torch.einsum(
                "bqhd,bkhd->bhqk", query_vectors, k[:, key_time].float()
            ) / scale
            log_mass = torch.logsumexp(logits[..., target_rows], dim=-1) - torch.logsumexp(
                logits, dim=-1
            )
            terms.append(-log_mass.mean())
    if not terms:
        raise RuntimeError("region correspondence loss produced no cross-time terms")
    return torch.stack(terms).mean(), len(terms)


def gaussian_targets(
    centers_tn: torch.Tensor,
    token_hw: tuple[int, int],
    sigma_tokens: float,
    device: torch.device,
) -> torch.Tensor:
    height, width = token_hw
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    center_x = (centers_tn % width).float()
    center_y = torch.div(centers_tn, width, rounding_mode="floor").float()
    centers = torch.stack((center_x, center_y), dim=-1).to(device)
    squared = (centers[..., None, :] - xy).square().sum(dim=-1)
    target = torch.exp(-0.5 * squared / max(float(sigma_tokens) ** 2, 1.0e-6))
    return target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def point_correspondence_loss(
    q_bshd: torch.Tensor,
    k_bshd: torch.Tensor,
    point_rows_tn: torch.Tensor,
    point_visibility_tn: torch.Tensor,
    token_hw: tuple[int, int],
    sigma_tokens: float,
) -> tuple[torch.Tensor, int]:
    """Cross-time Gaussian cross entropy for identity-preserving tracked points."""
    token_height, token_width = token_hw
    frame_tokens = token_height * token_width
    time_count, point_count = point_rows_tn.shape
    q = q_bshd.view(q_bshd.shape[0], time_count, frame_tokens, q_bshd.shape[2], q_bshd.shape[3])
    k = k_bshd.view(k_bshd.shape[0], time_count, frame_tokens, k_bshd.shape[2], k_bshd.shape[3])
    scale = math.sqrt(q_bshd.shape[-1])
    targets = gaussian_targets(point_rows_tn, token_hw, sigma_tokens, q.device)
    terms: list[torch.Tensor] = []
    for query_time in range(time_count):
        q_rows = point_rows_tn[query_time].to(q.device)
        point_index = torch.arange(point_count, device=q.device)
        query_vectors = q[:, query_time, q_rows, :, :].float()
        for key_time in range(time_count):
            if key_time == query_time:
                continue
            valid = (
                point_visibility_tn[query_time] & point_visibility_tn[key_time]
            ).to(q.device)
            if not valid.any():
                continue
            logits = torch.einsum(
                "bnhd,bkhd->bhnk", query_vectors[:, point_index], k[:, key_time].float()
            ) / scale
            log_prob = logits.log_softmax(dim=-1)
            target = targets[key_time].to(q.device)
            ce = -(target[None, None] * log_prob).sum(dim=-1)
            terms.append(ce[..., valid].mean())
    if not terms:
        raise RuntimeError("point correspondence loss produced no visible cross-time terms")
    return torch.stack(terms).mean(), len(terms)


class CorrespondenceLossCollector:
    """Patch selected attention modules and collect differentiable scalar losses."""

    def __init__(
        self,
        pipe: Any,
        entries: list[dict[str, Any]],
        loss_mode: str,
        target_masks_thw: np.ndarray,
        target_tracks_tn2: np.ndarray,
        target_visibility_tn: np.ndarray,
        pixel_hw: tuple[int, int],
        gaussian_sigma_tokens: float,
    ) -> None:
        if loss_mode not in {"region", "point", "combined"}:
            raise ValueError(loss_mode)
        self.pipe = pipe
        self.loss_mode = loss_mode
        self.target_masks_thw = target_masks_thw
        self.target_tracks_tn2 = target_tracks_tn2
        self.target_visibility_tn = torch.from_numpy(target_visibility_tn.astype(bool))
        self.pixel_hw = pixel_hw
        self.gaussian_sigma_tokens = float(gaussian_sigma_tokens)
        self.by_block: dict[int, list[int]] = {}
        for entry in entries:
            self.by_block.setdefault(int(entry["block"]), []).append(int(entry["head"]))
        self.active = False
        self.current_grid: tuple[int, int, int] | None = None
        self.losses: list[tuple[torch.Tensor, int]] = []
        self.term_count = 0
        self.head_events = 0
        self._original_forwards: list[tuple[Any, Any]] = []
        self._cached_geometry: dict[tuple[int, int], tuple[list[torch.Tensor], torch.Tensor]] = {}

    def _geometry(self, token_hw: tuple[int, int]) -> tuple[list[torch.Tensor], torch.Tensor]:
        cached = self._cached_geometry.get(token_hw)
        if cached is None:
            rows = masks_to_token_rows(self.target_masks_thw, token_hw)
            points = points_to_token_rows(self.target_tracks_tn2, self.pixel_hw, token_hw)
            cached = (rows, points)
            self._cached_geometry[token_hw] = cached
        return cached

    def reset_step(self, grid: tuple[int, int, int]) -> None:
        if int(grid[0]) != LATENT_FRAMES:
            raise RuntimeError(f"expected 13 latent frames, got {grid}")
        self.current_grid = grid
        self.losses.clear()
        self.term_count = 0
        self.head_events = 0

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, original: Any, block: int):
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        if self.current_grid is None:
            raise RuntimeError("collector grid is unset")
        time_count, token_height, token_width = self.current_grid
        if q.shape[1] != time_count * token_height * token_width:
            raise RuntimeError(f"token geometry mismatch: q={q.shape}, grid={self.current_grid}")
        num_heads = int(q.shape[-1] // HEAD_DIM)
        if num_heads != 24:
            raise RuntimeError(f"expected 24 heads, got {num_heads}")
        selected = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        q_heads = q.view(q.shape[0], q.shape[1], num_heads, HEAD_DIM)[:, :, selected]
        k_heads = k.view(k.shape[0], k.shape[1], num_heads, HEAD_DIM)[:, :, selected]
        region_rows, point_rows = self._geometry((token_height, token_width))
        mode_losses: list[torch.Tensor] = []
        if self.loss_mode in {"region", "combined"}:
            loss, terms = region_correspondence_loss(
                q_heads, k_heads, region_rows, (token_height, token_width)
            )
            mode_losses.append(loss)
            self.term_count += terms * len(heads)
        if self.loss_mode in {"point", "combined"}:
            loss, terms = point_correspondence_loss(
                q_heads,
                k_heads,
                point_rows,
                self.target_visibility_tn,
                (token_height, token_width),
                self.gaussian_sigma_tokens,
            )
            mode_losses.append(loss)
            self.term_count += terms * len(heads)
        self.losses.append((torch.stack(mode_losses).mean(), len(heads)))
        self.head_events += len(heads)
        return original(q, k, v)

    def install(self) -> None:
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for block, heads in self.by_block.items():
                module = model.blocks[block].self_attn.attn
                if any(not 0 <= head < int(module.num_heads) for head in heads):
                    raise ValueError(f"invalid head in block {block}: {heads}")
                original = module.forward
                self._original_forwards.append((module, original))

                def wrapped(q, k, v, *, _original=original, _block=block):
                    return self._attention(q, k, v, _original, _block)

                module.forward = wrapped

    def remove(self) -> None:
        for module, original in self._original_forwards:
            module.forward = original
        self._original_forwards.clear()

    def total_loss(self) -> torch.Tensor:
        if self.head_events != 100:
            raise RuntimeError(f"expected 100 selected-head events, got {self.head_events}")
        if not self.losses:
            raise RuntimeError("no correspondence loss was collected")
        # Every per-module value is already averaged over heads in that block.
        # Weight by physical head count so the final scalar is a true Top100 mean.
        weighted = []
        weights = []
        for loss, head_count in self.losses:
            weighted.append(loss * head_count)
            weights.append(head_count)
        return torch.stack(weighted).sum() / float(sum(weights))


def freeze_model_parameters(pipe: Any) -> None:
    pipe.eval()
    pipe.requires_grad_(False)
    for parameter in pipe.parameters():
        if parameter.requires_grad:
            raise RuntimeError("failed to freeze a Wan parameter")


def prepare_wan_inputs(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    seed: int,
    cfg_scale: float,
    num_inference_steps: int,
    sigma_shift: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Mirror the relevant official DiffSynth WanVideoPipeline input units."""
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)
    inputs_posi = {
        "prompt": prompt,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": negative_prompt,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_shared = {
        "input_image": input_image,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": int(seed),
        "rand_device": "cpu",
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": PIXEL_FRAMES,
        "cfg_scale": float(cfg_scale),
        "cfg_merge": False,
        "sigma_shift": float(sigma_shift),
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": True,
        "tile_size": (30, 52),
        "tile_stride": (15, 26),
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
        "wantodance_music_path": None,
        "wantodance_reference_image": None,
        "wantodance_fps": 30,
        "wantodance_keyframes": None,
        "wantodance_keyframes_mask": None,
        "framewise_decoding": False,
    }
    with torch.no_grad():
        for unit in pipe.units:
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
    return inputs_shared, inputs_posi, inputs_nega


def normalize_guidance_gradient(
    gradient: torch.Tensor,
    reference: torch.Tensor,
    mode: str,
    max_rms_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    gradient_float = gradient.float()
    reference_float = reference.float()
    grad_rms = gradient_float.square().mean().sqrt()
    reference_rms = reference_float.square().mean().sqrt()
    if mode == "rms":
        normalized = gradient_float / grad_rms.clamp_min(1.0e-12) * reference_rms
    elif mode == "none":
        normalized = gradient_float
    else:
        raise ValueError(mode)
    normalized_rms = normalized.square().mean().sqrt()
    maximum = reference_rms * float(max_rms_ratio)
    clip_scale = torch.minimum(
        torch.ones_like(maximum), maximum / normalized_rms.clamp_min(1.0e-12)
    )
    normalized = normalized * clip_scale
    return normalized.to(reference.dtype), {
        "raw_gradient_rms": float(grad_rms.detach().cpu()),
        "noise_prediction_rms": float(reference_rms.detach().cpu()),
        "normalized_gradient_rms": float(normalized.float().square().mean().sqrt().detach().cpu()),
        "gradient_clip_scale": float(clip_scale.detach().cpu()),
    }


def guided_generate(
    pipe: Any,
    collector: CorrespondenceLossCollector | None,
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    seed: int,
    cfg_scale: float,
    guidance_scale: float,
    guidance_start: int,
    guidance_end: int,
    gradient_normalization: str,
    max_gradient_rms_ratio: float,
    use_gradient_checkpointing: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    freeze_model_parameters(pipe)
    inputs_shared, inputs_posi, inputs_nega = prepare_wan_inputs(
        pipe, prompt, negative_prompt, input_image, seed, cfg_scale, 40, 5.0
    )
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    audit: list[dict[str, Any]] = []
    for step, scheduler_timestep in enumerate(pipe.scheduler.timesteps):
        if (
            scheduler_timestep.item() < 0.875 * 1000
            and pipe.dit2 is not None
            and models["dit"] is not pipe.dit2
        ):
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2
        timestep = scheduler_timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        latents = inputs_shared["latents"].detach()
        guided_step = collector is not None and guidance_start <= step <= guidance_end
        if guided_step:
            latent_leaf = latents.requires_grad_(True)
            inputs_shared["latents"] = latent_leaf
            patch = tuple(int(value) for value in models["dit"].patch_size)
            grid = (
                int(latent_leaf.shape[2] // patch[0]),
                int(latent_leaf.shape[3] // patch[1]),
                int(latent_leaf.shape[4] // patch[2]),
            )
            collector.reset_step(grid)
            collector.active = True
            try:
                with torch.enable_grad():
                    noise_pos = pipe.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_posi,
                        timestep=timestep,
                        use_gradient_checkpointing=bool(use_gradient_checkpointing),
                    )
                    loss = collector.total_loss()
                    forward_head_events = collector.head_events
                    forward_term_count = collector.term_count
                    # Non-reentrant checkpointing must execute the identical hook
                    # operations during recomputation. It will append disposable
                    # duplicate losses, but the gradient is taken only from `loss`.
                    gradient = torch.autograd.grad(loss, latent_leaf, only_inputs=True)[0]
            finally:
                collector.active = False
            gradient = gradient.detach()
            gradient[:, :, 0:1] = 0  # the conditioned first latent is immutable
            noise_pos = noise_pos.detach()
            normalized_gradient, gradient_audit = normalize_guidance_gradient(
                gradient, noise_pos, gradient_normalization, max_gradient_rms_ratio
            )
            sigma = float(pipe.scheduler.sigmas[step])
            loss_value = float(loss.detach().cpu())
            # Checkpoint recomputation appends duplicate side-loss tensors. They
            # are not part of the selected scalar and must not retain a graph
            # during the unconditional forward.
            collector.losses.clear()
            del loss
        else:
            inputs_shared["latents"] = latents
            with torch.no_grad():
                noise_pos = pipe.model_fn(
                    **models, **inputs_shared, **inputs_posi, timestep=timestep
                )
            normalized_gradient = torch.zeros_like(noise_pos)
            sigma = float(pipe.scheduler.sigmas[step])
            loss_value = math.nan
            gradient_audit = {
                "raw_gradient_rms": 0.0,
                "noise_prediction_rms": float(noise_pos.float().square().mean().sqrt().cpu()),
                "normalized_gradient_rms": 0.0,
                "gradient_clip_scale": 1.0,
            }
        inputs_shared["latents"] = latents
        with torch.no_grad():
            noise_neg = pipe.model_fn(
                **models, **inputs_shared, **inputs_nega, timestep=timestep
            )
            noise_cfg = noise_neg + cfg_scale * (noise_pos - noise_neg)
            # FlowMatch uses x_{s-1}=x_s+(sigma_next-sigma_s)*velocity.
            # Adding +grad(L) to velocity therefore performs a descent step on L
            # because sigma_next-sigma_s is negative.
            guided_velocity = noise_cfg + float(guidance_scale) * sigma * normalized_gradient
            inputs_shared["latents"] = pipe.scheduler.step(
                guided_velocity, scheduler_timestep, latents
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        audit.append(
            {
                "step": step,
                "sigma": sigma,
                "guided": guided_step,
                "loss": loss_value,
                "selected_head_events": forward_head_events if guided_step else 0,
                "correspondence_terms": forward_term_count if guided_step else 0,
                **gradient_audit,
            }
        )
        print(
            f"[denoise] step={step:02d} sigma={sigma:.5f} guided={guided_step} "
            f"loss={loss_value:.6f}",
            flush=True,
        )
    with torch.no_grad():
        for unit in pipe.post_units:
            inputs_shared, _, _ = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )
        pipe.load_models_to_device(["vae"])
        decoded = pipe.vae.decode(
            inputs_shared["latents"],
            device=pipe.device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        pil_video = pipe.vae_output_to_video(decoded)
        pipe.load_models_to_device([])
    frames = np.stack([np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in pil_video])
    return frames, audit


def guidance_sanity_check(
    pipe: Any,
    collector: CorrespondenceLossCollector,
    prompt: str,
    negative_prompt: str,
    input_image: Image.Image,
    seed: int,
    guidance_scale: float,
    gradient_normalization: str,
    max_gradient_rms_ratio: float,
    use_gradient_checkpointing: bool,
) -> dict[str, Any]:
    """Audit one real Wan step before committing to full guided generation."""
    freeze_model_parameters(pipe)
    inputs_shared, inputs_posi, _ = prepare_wan_inputs(
        pipe, prompt, negative_prompt, input_image, seed, 5.0, 40, 5.0
    )
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    scheduler_timestep = pipe.scheduler.timesteps[0]
    timestep = scheduler_timestep.unsqueeze(0).to(
        dtype=pipe.torch_dtype, device=pipe.device
    )
    latents = inputs_shared["latents"].detach()
    latent_leaf = latents.requires_grad_(True)
    inputs_shared["latents"] = latent_leaf
    patch = tuple(int(value) for value in models["dit"].patch_size)
    grid = (
        int(latent_leaf.shape[2] // patch[0]),
        int(latent_leaf.shape[3] // patch[1]),
        int(latent_leaf.shape[4] // patch[2]),
    )
    collector.reset_step(grid)
    collector.active = True
    try:
        with torch.enable_grad():
            noise_pos = pipe.model_fn(
                **models,
                **inputs_shared,
                **inputs_posi,
                timestep=timestep,
                use_gradient_checkpointing=bool(use_gradient_checkpointing),
            )
            loss = collector.total_loss()
            forward_head_events = collector.head_events
            forward_term_count = collector.term_count
            gradient = torch.autograd.grad(loss, latent_leaf, only_inputs=True)[0]
    finally:
        collector.active = False

    gradient = gradient.detach()
    first_latent_raw_gradient_rms = float(
        gradient[:, :, 0:1].float().square().mean().sqrt().cpu()
    )
    gradient[:, :, 0:1] = 0
    noise_pos = noise_pos.detach()
    normalized_gradient, gradient_audit = normalize_guidance_gradient(
        gradient, noise_pos, gradient_normalization, max_gradient_rms_ratio
    )
    sigma = float(pipe.scheduler.sigmas[0])
    next_sigma = float(pipe.scheduler.sigmas[1])
    delta_sigma = next_sigma - sigma
    guidance_delta = (
        delta_sigma * float(guidance_scale) * sigma * normalized_gradient.float()
    )
    directional_derivative = float(
        (gradient.float() * guidance_delta).sum().detach().cpu()
    )
    model_gradient_tensors = sum(
        int(parameter.grad is not None) for parameter in pipe.parameters()
    )
    finite = all(
        math.isfinite(value)
        for value in (
            float(loss.detach().cpu()),
            directional_derivative,
            *gradient_audit.values(),
        )
    ) and bool(torch.isfinite(guidance_delta).all())
    passed = (
        finite
        and forward_head_events == 100
        and forward_term_count > 0
        and delta_sigma < 0
        and directional_derivative < 0
        and model_gradient_tensors == 0
        and not bool(gradient[:, :, 0:1].any())
    )
    report = {
        "passed": bool(passed),
        "loss": float(loss.detach().cpu()),
        "selected_head_events": int(forward_head_events),
        "correspondence_terms": int(forward_term_count),
        "sigma": sigma,
        "next_sigma": next_sigma,
        "delta_sigma": delta_sigma,
        "guidance_scale": float(guidance_scale),
        "guidance_update_rms": float(
            guidance_delta.square().mean().sqrt().detach().cpu()
        ),
        "gradient_dot_actual_guidance_delta": directional_derivative,
        "first_latent_raw_gradient_rms": first_latent_raw_gradient_rms,
        "first_latent_guidance_forced_zero": not bool(
            gradient[:, :, 0:1].any()
        ),
        "model_gradient_tensors": int(model_gradient_tensors),
        **gradient_audit,
    }
    collector.losses.clear()
    del loss, gradient, normalized_gradient, guidance_delta, noise_pos
    del latent_leaf, latents, inputs_shared, inputs_posi
    gc.collect()
    torch.cuda.empty_cache()
    if not passed:
        raise RuntimeError(f"guidance sanity check failed: {report}")
    return report


def resolve_condition_image(payload: dict[str, Any], source_frames: np.ndarray) -> Image.Image:
    input_image = payload.get("input_image")
    if input_image and Path(str(input_image)).expanduser().is_file():
        image = Image.open(Path(str(input_image)).expanduser()).convert("RGB")
    else:
        image = Image.fromarray(source_frames[0])
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def variant_name(
    loss_mode: str, target: GuidanceTarget | None, guidance_scale: float
) -> str:
    return (
        "baseline"
        if target is None
        else f"{loss_mode}__{target.name}__lambda{float_tag(guidance_scale)}"
    )


def variant_dir(output_root: Path, case: str, seed: int, variant: str) -> Path:
    return output_root / "generations" / case / f"seed_{seed:05d}" / variant


def sanity_check_case(
    pipe_wrapper: Any,
    json_path: Path,
    tube: FrozenTube,
    entries: list[dict[str, Any]],
    output_root: Path,
    seed: int,
    loss_modes: tuple[str, ...],
    target_names: tuple[str, ...] | None,
    guidance_scale: float,
    gaussian_sigma_tokens: float,
    gradient_normalization: str,
    max_gradient_rms_ratio: float,
    use_gradient_checkpointing: bool,
) -> None:
    payload = load_payload(json_path)
    source_frames = read_source_prefix(tube.source_video)
    input_image = resolve_condition_image(payload, source_frames)
    report_root = output_root / "sanity" / tube.case / f"seed_{seed:05d}"
    for target in selected_target_specs(tube, target_names):
        target_masks, target_tracks, target_visibility = selected_object_arrays(
            tube, target
        )
        for loss_mode in loss_modes:
            collector = CorrespondenceLossCollector(
                pipe_wrapper.pipe,
                entries,
                loss_mode,
                target_masks,
                target_tracks,
                target_visibility,
                (tube.pixel_height, tube.pixel_width),
                gaussian_sigma_tokens,
            )
            collector.install()
            try:
                report = guidance_sanity_check(
                    pipe_wrapper.pipe,
                    collector,
                    str(payload["input_caption"]),
                    str(build_args(seed).negative_prompt),
                    input_image,
                    seed,
                    guidance_scale,
                    gradient_normalization,
                    max_gradient_rms_ratio,
                    use_gradient_checkpointing,
                )
            finally:
                collector.remove()
            report.update(
                {
                    "protocol": PROTOCOL,
                    "case": tube.case,
                    "seed": int(seed),
                    "target": target.name,
                    "target_object_indices": list(target.object_indices),
                    "loss_mode": loss_mode,
                    "mask_valid_anchor_count": int(
                        target_masks.reshape(LATENT_FRAMES, -1).any(axis=1).sum()
                    ),
                    "point_visible_anchor_count": int(
                        target_visibility.any(axis=1).sum()
                    ),
                }
            )
            report_path = report_root / f"{loss_mode}__{target.name}.json"
            atomic_write_json(report_path, report)
            print(
                f"[sanity] pass {tube.case}/{loss_mode}/{target.name}: "
                f"loss={report['loss']:.6f} "
                f"grad_dot_delta={report['gradient_dot_actual_guidance_delta']:.6e}",
                flush=True,
            )


def generate_case(
    pipe_wrapper: Any,
    json_path: Path,
    tube: FrozenTube,
    entries: list[dict[str, Any]],
    head_ranking_path: Path,
    output_root: Path,
    seed: int,
    loss_modes: tuple[str, ...],
    target_names: tuple[str, ...] | None,
    guidance_scale: float,
    guidance_start: int,
    guidance_end: int,
    gaussian_sigma_tokens: float,
    gradient_normalization: str,
    max_gradient_rms_ratio: float,
    use_gradient_checkpointing: bool,
    include_baseline: bool,
    overwrite: bool,
) -> None:
    payload = load_payload(json_path)
    source_frames = read_source_prefix(tube.source_video)
    input_image = resolve_condition_image(payload, source_frames)
    tasks: list[tuple[str, GuidanceTarget | None]] = []
    if include_baseline:
        tasks.append(("baseline", None))
    for target in selected_target_specs(tube, target_names):
        tasks.extend((loss_mode, target) for loss_mode in loss_modes)
    for loss_mode, target in tasks:
        variant = variant_name(loss_mode, target, guidance_scale)
        output = variant_dir(output_root, tube.case, seed, variant)
        required = (output / "generated.mp4", output / "manifest.json", output / "complete.json")
        if all(path.is_file() for path in required) and not overwrite:
            print(f"[generate] skip {tube.case}/{variant}", flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        (output / "complete.json").unlink(missing_ok=True)
        collector = None
        if target is not None:
            target_masks, target_tracks, target_visibility = selected_object_arrays(tube, target)
            collector = CorrespondenceLossCollector(
                pipe_wrapper.pipe,
                entries,
                loss_mode,
                target_masks,
                target_tracks,
                target_visibility,
                (tube.pixel_height, tube.pixel_width),
                gaussian_sigma_tokens,
            )
            collector.install()
        try:
            frames, audit = guided_generate(
                pipe_wrapper.pipe,
                collector,
                str(payload["input_caption"]),
                str(build_args(seed).negative_prompt),
                input_image,
                seed,
                5.0,
                guidance_scale if target is not None else 0.0,
                guidance_start,
                guidance_end,
                gradient_normalization,
                max_gradient_rms_ratio,
                use_gradient_checkpointing,
            )
        finally:
            if collector is not None:
                collector.remove()
        temporary = output / "generated.tmp.mp4"
        save_video_np(frames, temporary, fps=30)
        temporary.replace(output / "generated.mp4")
        manifest = {
            "protocol": PROTOCOL,
            "case": tube.case,
            "seed": seed,
            "variant": variant,
            "loss_mode": loss_mode,
            "target": target.name if target else None,
            "target_object_indices": list(target.object_indices) if target else [],
            "source_json": str(json_path),
            "source_gt_video": str(tube.source_video),
            "conditioning": (
                "legacy Wan2.2 TI2V input_image; input_video/context-video JSON field is "
                "not consumed by this baseline pipeline"
            ),
            "head_ranking": str(head_ranking_path),
            "selected_heads": entries,
            "selected_head_count": len(entries),
            "cfg_branch_for_loss": "positive conditional only",
            "denoising_steps": 40,
            "guidance_step_range_inclusive": [guidance_start, guidance_end],
            "guidance_scale": guidance_scale if target is not None else 0.0,
            "gradient_normalization": gradient_normalization,
            "max_gradient_rms_ratio": max_gradient_rms_ratio,
            "gaussian_sigma_tokens": gaussian_sigma_tokens if loss_mode != "baseline" else None,
            "model_parameters_updated": False,
            "latent_update": (
                "velocity_guided = velocity_CFG + lambda*sigma*normalized_grad_x(L_STC); "
                "then the ordinary FlowMatch scheduler step"
            ),
            "first_condition_latent_gradient_forced_zero": True,
            "cross_time_pairs": "all ordered (t_query,t_key) with t_query != t_key",
            "audit": audit,
        }
        atomic_write_json(output / "manifest.json", manifest)
        atomic_write_json(output / "complete.json", {"case": tube.case, "variant": variant})
        print(f"[generate] complete {tube.case}/{variant}", flush=True)
        del frames
        gc.collect()
        torch.cuda.empty_cache()


def trajectory_metrics(
    candidate_tracks: np.ndarray,
    candidate_visibility: np.ndarray,
    tube: FrozenTube,
    target: GuidanceTarget,
) -> dict[str, Any]:
    point_indices = np.concatenate(
        [
            np.arange(tube.point_starts[index], tube.point_ends[index], dtype=np.int64)
            for index in target.object_indices
        ]
    )
    generated_anchors = LATENT_PIXEL_ANCHORS
    candidate = candidate_tracks[generated_anchors][:, point_indices]
    reference = tube.tracks_tn2[:, point_indices]
    candidate_visible = candidate_visibility[generated_anchors][:, point_indices]
    reference_visible = tube.visibility_tn[:, point_indices].astype(bool)
    error = np.linalg.norm(candidate - reference, axis=-1)
    finite = candidate_visible & reference_visible & np.isfinite(error)

    # F00 is the fixed TI2V condition image and its latent gradient is forced to
    # zero.  It is therefore not a predicted trajectory sample and must never
    # make a failed future track look perfect merely because F00 matches GT.
    future_finite = finite[1:]
    future_error = error[1:]
    future_reference = reference_visible[1:] & np.isfinite(reference[1:]).all(axis=-1)
    min_points_per_anchor = min(4, len(point_indices))
    reference_anchor_valid = future_reference.sum(axis=1) >= min_points_per_anchor
    common_anchor_valid = future_finite.sum(axis=1) >= min_points_per_anchor
    reference_anchor_count = int(reference_anchor_valid.sum())
    common_anchor_count = int((common_anchor_valid & reference_anchor_valid).sum())
    anchor_coverage = common_anchor_count / max(reference_anchor_count, 1)
    quality_pass = common_anchor_count >= 4 and anchor_coverage >= 0.8
    first_mask = np.logical_or.reduce(
        tube.masks_othw[list(target.object_indices), 0], axis=0
    )
    yx = np.argwhere(first_mask)
    diagonal = (
        math.hypot(
            float(yx[:, 1].max() - yx[:, 1].min() + 1),
            float(yx[:, 0].max() - yx[:, 0].min() + 1),
        )
        if len(yx)
        else 1.0
    )
    visible_error = future_error[future_finite]
    final_valid = future_finite[-1]
    final_usable = int(final_valid.sum()) >= min_points_per_anchor
    raw_ade_px = float(visible_error.mean()) if len(visible_error) else None
    raw_fde_px = (
        float(future_error[-1, final_valid].mean()) if final_usable else None
    )
    raw_pck_10 = (
        float((visible_error <= 0.10 * diagonal).mean())
        if len(visible_error)
        else None
    )
    raw_pck_20 = (
        float((visible_error <= 0.20 * diagonal).mean())
        if len(visible_error)
        else None
    )
    return {
        "target": target.name,
        "point_count": int(len(point_indices)),
        "condition_frame_valid_comparisons": int(finite[0].sum()),
        "valid_comparisons": int(future_finite.sum()),
        "visibility_rate": float(future_finite.mean()),
        "future_reference_visible_comparisons": int(future_reference.sum()),
        "future_common_point_coverage": float(
            future_finite.sum() / max(int(future_reference.sum()), 1)
        ),
        "future_reference_anchor_count": reference_anchor_count,
        "future_common_anchor_count": common_anchor_count,
        "future_common_anchor_coverage": float(anchor_coverage),
        "future_track_loss_score_0_100": float(100.0 * (1.0 - anchor_coverage)),
        "quality_pass": bool(quality_pass),
        "quality_gate": "future common anchors >= 4 and coverage >= 0.8; F00 excluded",
        "ade_px": raw_ade_px if quality_pass else None,
        "ade_d0": (
            float(raw_ade_px / diagonal)
            if quality_pass and raw_ade_px is not None
            else None
        ),
        "fde_px": raw_fde_px if quality_pass else None,
        "fde_d0": (
            float(raw_fde_px / diagonal)
            if quality_pass and raw_fde_px is not None
            else None
        ),
        "pck_10pct_d0": raw_pck_10 if quality_pass else None,
        "pck_20pct_d0": raw_pck_20 if quality_pass else None,
        "raw_ade_px": raw_ade_px,
        "raw_ade_d0": float(raw_ade_px / diagonal) if raw_ade_px is not None else None,
        "raw_fde_px": raw_fde_px,
        "raw_fde_d0": float(raw_fde_px / diagonal) if raw_fde_px is not None else None,
        "raw_pck_10pct_d0": raw_pck_10,
        "raw_pck_20pct_d0": raw_pck_20,
        "d0_px": float(diagonal),
    }


def evaluate_case(
    json_path: Path,
    tube: FrozenTube,
    output_root: Path,
    seed: int,
    device: str,
    overwrite: bool,
) -> None:
    root = output_root / "generations" / tube.case / f"seed_{seed:05d}"
    videos = sorted(root.glob("*/generated.mp4"))
    if not videos:
        print(f"[evaluate] no generated videos for {tube.case}", flush=True)
        return
    model = load_cotracker(device)
    try:
        for video_path in videos:
            output = video_path.parent
            metrics_path = output / "trajectory_metrics.json"
            if metrics_path.is_file() and not overwrite:
                print(f"[evaluate] skip {tube.case}/{output.name}", flush=True)
                continue
            frames = np.asarray(iio.imread(video_path))[:PIXEL_FRAMES, ..., :3]
            if len(frames) != PIXEL_FRAMES:
                raise RuntimeError(f"expected 49 generated frames: {video_path}")
            tracks, visibility = run_cotracker(
                model, frames, tube.query_points_n2, device
            )
            target_rows = [
                trajectory_metrics(tracks, visibility, tube, target)
                for target in target_specs(tube)
            ]
            atomic_write_json(
                metrics_path,
                {
                    "protocol": PROTOCOL,
                    "case": tube.case,
                    "variant": output.name,
                    "reference": "source-video SAM2/CoTracker GT tube",
                    "generated_anchor_pixel_frames": LATENT_PIXEL_ANCHORS.tolist(),
                    "source_anchor_pixel_frames": tube.anchor_source_frames.tolist(),
                    "metrics": target_rows,
                },
            )
            print(f"[evaluate] complete {tube.case}/{output.name}", flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    summarize_case_metrics(root, tube.case)


def summarize_case_metrics(generation_root: Path, case: str) -> None:
    """Write explicit guided-minus-baseline deltas after all available evaluations."""
    reports: dict[str, dict[str, Any]] = {}
    for path in sorted(generation_root.glob("*/trajectory_metrics.json")):
        reports[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    baseline = reports.get("baseline")
    if baseline is None:
        return
    baseline_by_target = {
        str(row["target"]): row for row in baseline.get("metrics", [])
    }
    comparisons = []
    for variant, report in reports.items():
        if variant == "baseline":
            continue
        for row in report.get("metrics", []):
            target = str(row["target"])
            reference = baseline_by_target.get(target)
            if reference is None:
                continue
            delta: dict[str, float | None] = {}
            for metric in ("ade_px", "ade_d0", "fde_px", "fde_d0"):
                left, right = row.get(metric), reference.get(metric)
                delta[f"delta_{metric}"] = (
                    float(left) - float(right)
                    if left is not None and right is not None
                    else None
                )
            for metric in ("pck_10pct_d0", "pck_20pct_d0"):
                left, right = row.get(metric), reference.get(metric)
                delta[f"delta_{metric}"] = (
                    float(left) - float(right)
                    if left is not None and right is not None
                    else None
                )
            comparisons.append(
                {
                    "variant": variant,
                    "metric_target": target,
                    "baseline_quality_pass": bool(reference.get("quality_pass", False)),
                    "variant_quality_pass": bool(row.get("quality_pass", False)),
                    "baseline_future_track_loss_score_0_100": reference.get(
                        "future_track_loss_score_0_100"
                    ),
                    "variant_future_track_loss_score_0_100": row.get(
                        "future_track_loss_score_0_100"
                    ),
                    "delta_future_track_loss_score_0_100": (
                        float(row["future_track_loss_score_0_100"])
                        - float(reference["future_track_loss_score_0_100"])
                        if row.get("future_track_loss_score_0_100") is not None
                        and reference.get("future_track_loss_score_0_100") is not None
                        else None
                    ),
                    **delta,
                    "interpretation": (
                        "ADE/FDE/PCK deltas are emitted only after both future-track "
                        "quality gates pass; positive Track Loss delta means worse observability"
                    ),
                }
            )
    atomic_write_json(
        generation_root / "comparison_to_baseline.json",
        {
            "protocol": PROTOCOL,
            "case": case,
            "baseline": "baseline",
            "comparisons": comparisons,
        },
    )


def build_pipeline(seed: int) -> Any:
    args = build_args(seed)
    args.backend = "legacy"
    args.sampling_steps = 40
    args.frame_num = PIXEL_FRAMES
    return build_wan_ti2v_pipeline(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--stage",
        choices=("prepare", "sanity", "generate", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--head-ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument(
        "--target-names",
        nargs="*",
        default=None,
        help="Optional exact guidance targets such as object_A or moving_union",
    )
    parser.add_argument(
        "--target-map",
        type=Path,
        default=None,
        help=(
            "Frozen JSON case-to-target mapping; accepts the screening report's "
            "eligible_jobs schema"
        ),
    )
    parser.add_argument("--points-per-object", type=int, default=8)
    parser.add_argument("--moving-threshold-d0", type=float, default=0.05)
    parser.add_argument(
        "--tube-mask-strategy",
        choices=("cotracker_prompted_sam2", "legacy_sam2_propagation"),
        default="cotracker_prompted_sam2",
        help=(
            "Build each latent mask directly from same-frame CoTracker point prompts; "
            "neighbor propagation remains an audited fallback"
        ),
    )
    parser.add_argument("--loss-modes", nargs="+", choices=("region", "point", "combined"), default=("region", "point"))
    parser.add_argument("--guidance-scale", type=float, default=0.10)
    parser.add_argument("--guidance-start", type=int, default=0)
    parser.add_argument("--guidance-end", type=int, default=39)
    parser.add_argument("--gaussian-sigma-tokens", type=float, default=1.5)
    parser.add_argument("--gradient-normalization", choices=("rms", "none"), default="rms")
    parser.add_argument("--max-gradient-rms-ratio", type=float, default=1.0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Generate only the unguided Baseline; ignore loss modes and targets",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0,num-workers)")
    if not 0 <= args.guidance_start <= args.guidance_end < 40:
        raise ValueError("guidance steps must satisfy 0 <= start <= end < 40")
    if args.guidance_scale < 0:
        raise ValueError("guidance-scale must be non-negative")
    if args.gaussian_sigma_tokens <= 0:
        raise ValueError("gaussian-sigma-tokens must be positive")
    if args.baseline_only and args.no_baseline:
        raise ValueError("--baseline-only and --no-baseline are mutually exclusive")
    if args.target_map is not None and args.target_names:
        raise ValueError("--target-map and --target-names are mutually exclusive")
    if args.device in {"cuda:4", "4"}:
        raise ValueError("workspace policy forbids GPU 4")


def selected_cases(args: argparse.Namespace) -> list[Path]:
    paths = deduplicated_json_paths(args.input_list.expanduser().resolve())
    selected = set(args.case_keys or [])
    if args.target_map is not None:
        target_map = load_target_map(args.target_map.expanduser().resolve())
        selected_from_map = set(target_map)
        selected = selected & selected_from_map if selected else selected_from_map
    unknown = selected - {path.stem for path in paths}
    if unknown:
        raise ValueError(f"unknown case keys: {sorted(unknown)}")
    paths = [path for path in paths if not selected or path.stem in selected]
    return paths[args.worker_id :: args.num_workers]


def dry_run_report(args: argparse.Namespace, paths: list[Path]) -> None:
    report = {
        "protocol": PROTOCOL,
        "stage": args.stage,
        "input_list_total_unique": len(deduplicated_json_paths(args.input_list)),
        "worker_case_count": len(paths),
        "worker_cases": [path.stem for path in paths],
        "tube_mask_strategy": args.tube_mask_strategy,
        "loss_modes": list(args.loss_modes),
        "baseline_only": bool(args.baseline_only),
        "target_names": list(args.target_names or []),
        "head_ranking": str(args.head_ranking),
        "selected_head_count": 100,
        "denoising_steps": 40,
        "cfg_loss_branch": "conditional",
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "output_root": str(args.output_root),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


def serializable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """Return CLI arguments with filesystem paths normalized for run_config JSON."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def release_pipeline(wrapper: Any) -> None:
    if wrapper is None:
        return
    if hasattr(wrapper, "pipe"):
        del wrapper.pipe
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    validate_args(args)
    paths = selected_cases(args)
    if not paths:
        raise RuntimeError("this worker has no selected cases")
    entries = validate_latest3350_top100(args.head_ranking.expanduser().resolve())
    target_map = (
        load_target_map(args.target_map.expanduser().resolve())
        if args.target_map is not None
        else None
    )
    if args.dry_run:
        dry_run_report(args, paths)
        return
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        output_root / "run_config.json",
        {
            "protocol": PROTOCOL,
            "input_list": str(args.input_list),
            "deduplicated_case_count": len(deduplicated_json_paths(args.input_list)),
            "head_ranking": str(args.head_ranking),
            "selected_heads": entries,
            "arguments": serializable_arguments(args),
        },
    )
    if args.stage in {"prepare", "all"}:
        for json_path in paths:
            try:
                prepare_case(
                    json_path,
                    output_root,
                    args.device,
                    args.points_per_object,
                    args.moving_threshold_d0,
                    args.tube_mask_strategy,
                    args.overwrite,
                )
            except Exception:
                error = traceback.format_exc()
                error_dir = tube_dir(output_root, json_path.stem)
                error_dir.mkdir(parents=True, exist_ok=True)
                (error_dir / "error.txt").write_text(error, encoding="utf-8")
                print(error, flush=True)
                raise
    if args.stage in {"sanity", "all"}:
        missing = [path.stem for path in paths if not (tube_dir(output_root, path.stem) / "complete.json").is_file()]
        if missing:
            raise RuntimeError(f"prepare stage is incomplete for: {missing}")
        pipeline = build_pipeline(args.seed)
        try:
            for json_path in paths:
                tube = load_frozen_tube(output_root, json_path.stem)
                sanity_check_case(
                    pipeline,
                    json_path,
                    tube,
                    entries,
                    output_root,
                    args.seed,
                    () if args.baseline_only else tuple(args.loss_modes),
                    target_names_for_case(
                        tube.case,
                        tuple(args.target_names) if args.target_names else None,
                        target_map,
                    ),
                    args.guidance_scale,
                    args.gaussian_sigma_tokens,
                    args.gradient_normalization,
                    args.max_gradient_rms_ratio,
                    not args.no_gradient_checkpointing,
                )
        finally:
            release_pipeline(pipeline)
    if args.stage in {"generate", "all"}:
        missing = [path.stem for path in paths if not (tube_dir(output_root, path.stem) / "complete.json").is_file()]
        if missing:
            raise RuntimeError(f"prepare stage is incomplete for: {missing}")
        pipeline = build_pipeline(args.seed)
        try:
            for json_path in paths:
                tube = load_frozen_tube(output_root, json_path.stem)
                generate_case(
                    pipeline,
                    json_path,
                    tube,
                    entries,
                    args.head_ranking.expanduser().resolve(),
                    output_root,
                    args.seed,
                    () if args.baseline_only else tuple(args.loss_modes),
                    target_names_for_case(
                        tube.case,
                        tuple(args.target_names) if args.target_names else None,
                        target_map,
                    ),
                    args.guidance_scale,
                    args.guidance_start,
                    args.guidance_end,
                    args.gaussian_sigma_tokens,
                    args.gradient_normalization,
                    args.max_gradient_rms_ratio,
                    not args.no_gradient_checkpointing,
                    not args.no_baseline,
                    args.overwrite,
                )
        finally:
            release_pipeline(pipeline)
    if args.stage in {"evaluate", "all"}:
        for json_path in paths:
            tube = load_frozen_tube(output_root, json_path.stem)
            evaluate_case(
                json_path,
                tube,
                output_root,
                args.seed,
                args.device,
                args.overwrite,
            )


if __name__ == "__main__":
    main()
