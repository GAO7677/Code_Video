#!/usr/bin/env python3
"""Precompute GroundingDINO + SAM2 object/background query regions."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np


CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    DetectedObjectTrack,
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
)
from AAA_my_test.sam2_region_query_utils import (
    DEFAULT_CACHE_ROOT,
    QUERY_CONTEXT_FRAME,
    build_regions_from_grounding,
    save_region_cache,
)


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--query-context-frame", type=int, default=QUERY_CONTEXT_FRAME)
    parser.add_argument("--points-per-region", type=int, default=8)
    parser.add_argument("--object-erode-px", type=int, default=11)
    parser.add_argument("--background-erode-px", type=int, default=31)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_cases(dataset_root: Path, selected: list[str] | None) -> list[dict]:
    selected_set = set(selected or [])
    cases = []
    for path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        case_key = str(payload["case_key"])
        if selected_set and case_key not in selected_set:
            continue
        base = payload["base"]
        phrases = [str(value).strip() for value in base.get("object_phrases", []) if str(value).strip()]
        if len(phrases) != int(payload["object_count"]):
            raise ValueError(f"{case_key}: object phrase/count mismatch")
        cases.append(
            {
                "case_key": case_key,
                "video": Path(base["video"]),
                "caption": str(base["caption"]),
                "object_phrases": phrases,
                "object_count": int(payload["object_count"]),
            }
        )
    if not cases:
        raise RuntimeError(f"no matching cases under {dataset_root}")
    return cases


def build_provider(device: str, points_per_region: int) -> ViewerGroundingBoxProvider:
    return ViewerGroundingBoxProvider(
        device=device,
        segment_len=8,
        max_objects=8,
        points_per_object=points_per_region,
        proposal_source="gdino_only",
        motion_score_ratio=0.15,
        text_prompt="",
        extra_prompt_terms="",
        include_caption_terms=True,
        gdino_box_threshold=0.20,
        gdino_text_threshold=0.15,
        prompt_frame_mode="first",
        track_dedupe_iou_threshold=0.85,
        container_suppress_ratio_threshold=0.95,
        container_suppress_min_contained=2,
        container_suppress_min_area_ratio=1.5,
        container_suppress_small_iou_threshold=0.7,
        caption_prompt_mode="physical_noun_phrases",
        caption_max_phrases=4,
        caption_min_score=1.0,
    )


def _box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return intersection / max(area_a + area_b - intersection, 1.0e-6)


def _containment(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Return the larger fraction of either box contained by the other."""
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return max(intersection / max(area_a, 1.0e-6), intersection / max(area_b, 1.0e-6))


def detect_and_track_objects(
    provider: ViewerGroundingBoxProvider,
    frames_tchw_01: np.ndarray,
    object_phrases: list[str],
) -> SimpleNamespace:
    """Globally assign distinct GDINO boxes, then propagate each with SAM2."""
    detections = []
    for phrase in object_phrases:
        output = provider.detector.detect(
            frames_tchw_01[0], phrase, guidance_box_xyxy=None
        )
        candidates = [
            {
                "box": np.asarray(box, dtype=np.float32),
                "score": float(score),
                "detected_phrase": str(detected_phrase),
            }
            for box, score, detected_phrase in zip(
                output.boxes_xyxy, output.scores, output.phrases
            )
        ]
        if not candidates:
            raise RuntimeError(f"GroundingDINO found no candidate for {phrase!r}")
        detections.append(candidates)

    best = None
    for indices in itertools.product(*(range(len(items)) for items in detections)):
        selected = [detections[index][candidate] for index, candidate in enumerate(indices)]
        conflict = any(
            _box_iou(selected[i]["box"], selected[j]["box"]) >= 0.50
            or _containment(selected[i]["box"], selected[j]["box"]) >= 0.85
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        )
        if conflict:
            continue
        score = sum(math.log(max(item["score"], 1.0e-8)) for item in selected)
        if best is None or score > best[0]:
            best = (score, indices, selected)
    if best is None:
        candidate_counts = [len(items) for items in detections]
        raise RuntimeError(
            f"could not assign distinct GroundingDINO boxes; candidates={candidate_counts}"
        )

    _, selected_indices, selected = best
    tracks = []
    for phrase, candidate in zip(object_phrases, selected):
        sam_output = provider.tracker.track(
            frames_tchw_01,
            prompt_frame_idx=0,
            prompt_box_xyxy=candidate["box"],
            caption="",
        )
        if not np.asarray(sam_output.masks_thw).any():
            raise RuntimeError(f"SAM2 produced an empty track for {phrase!r}")
        tracks.append(
            DetectedObjectTrack(
                box_prompt_xyxy=candidate["box"],
                masks_thw=np.asarray(sam_output.masks_thw, dtype=np.uint8),
                boxes_t4=np.asarray(sam_output.boxes_t4, dtype=np.float32),
                score=float(candidate["score"]),
                phrase=phrase,
                source_phrase=phrase,
            )
        )
    return SimpleNamespace(
        object_tracks=tracks,
        debug={
            "mode": "per_phrase_global_distinct_box_assignment",
            "object_phrases": object_phrases,
            "selected_candidate_indices": list(selected_indices),
            "selected": [
                {
                    "phrase": phrase,
                    "box_xyxy": [float(value) for value in candidate["box"].tolist()],
                    "score": float(candidate["score"]),
                    "detected_phrase": candidate["detected_phrase"],
                }
                for phrase, candidate in zip(object_phrases, selected)
            ],
            "candidate_counts": [len(items) for items in detections],
        },
    )


def main() -> None:
    args = parse_args()
    cache_root = args.cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.dataset_root.expanduser().resolve(), args.case_keys)
    provider = build_provider(str(args.device), int(args.points_per_region))
    failures = []
    for index, case in enumerate(cases, start=1):
        case_dir = cache_root / case["case_key"]
        if (case_dir / "complete.json").is_file() and not args.overwrite:
            print(f"[{index}/{len(cases)}] skip {case['case_key']}", flush=True)
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(cases)}] start {case['case_key']}", flush=True)
        try:
            frames, frame_indices = read_video_prefix(case["video"], int(args.context_frames))
            context = preprocess_video_rgb_uint8(
                frames,
                (int(args.height), int(args.width)),
                value_range="minus_one_to_one",
                resize_mode="stretch",
            )
            frames_tchw_01 = (
                ((context.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
            )
            sample = detect_and_track_objects(
                provider,
                frames_tchw_01,
                case["object_phrases"],
            )
            query_frame = int(args.query_context_frame)
            frame_rgb = np.transpose(
                (frames_tchw_01[query_frame] * 255.0).round().astype(np.uint8),
                (1, 2, 0),
            )
            cache = build_regions_from_grounding(
                case_key=case["case_key"],
                grounding_sample=sample,
                object_phrases=case["object_phrases"],
                context_frame_rgb=frame_rgb,
                query_frame_index=query_frame,
                points_per_region=int(args.points_per_region),
                object_erode_px=int(args.object_erode_px),
                background_erode_px=int(args.background_erode_px),
            )
            cache.metadata.update(
                {
                    "context_video": str(case["video"]),
                    "context_source_frame_indices": frame_indices.tolist(),
                    "grounding_prompts": case["object_phrases"],
                    "object_phrases": case["object_phrases"],
                    "height": int(args.height),
                    "width": int(args.width),
                    "resize_mode": "stretch",
                    "mask_source": "GroundingDINO text boxes -> SAM2 video propagation",
                    "uses_gt_instance_masks": False,
                }
            )
            save_region_cache(case_dir, cache)
        except Exception:
            error = traceback.format_exc()
            (case_dir / "error.txt").write_text(error, encoding="utf-8")
            failures.append(case["case_key"])
            print(error, flush=True)
            continue
        print(
            f"[{index}/{len(cases)}] complete {case['case_key']}: "
            f"{len(cache.regions) - 1} objects + background",
            flush=True,
        )
    if failures:
        raise SystemExit(f"SAM2 precompute failed for {len(failures)} cases: {failures}")


if __name__ == "__main__":
    main()
