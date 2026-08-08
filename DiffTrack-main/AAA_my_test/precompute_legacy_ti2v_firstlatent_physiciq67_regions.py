#!/usr/bin/env python3
"""Build automatic first-frame object query caches for PhysicIQ67."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import (
    CASES,
    REGION_CACHE_ROOT,
    read_payload,
)
from AAA_my_test.precompute_toydataset_sam2_regions import build_provider
from AAA_my_test.sam2_region_query_utils import build_regions_from_grounding, save_region_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--query-context-frame", type=int, default=0)
    parser.add_argument("--points-per-region", type=int, default=8)
    parser.add_argument("--object-erode-px", type=int, default=11)
    parser.add_argument("--background-erode-px", type=int, default=31)
    parser.add_argument("--case-keys", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-visualizations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    selected_keys = set(args.case_keys or [])
    unknown = selected_keys - {case.key for case in CASES}
    if unknown:
        raise ValueError(f"unknown case keys: {sorted(unknown)}")
    selected = [case for case in CASES if not selected_keys or case.key in selected_keys]
    cases = selected[args.worker_id :: args.num_workers]
    if not cases:
        raise RuntimeError("worker has no matching cases")

    REGION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    provider = build_provider(str(args.device), int(args.points_per_region))
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        output_dir = REGION_CACHE_ROOT / case.key
        if (output_dir / "complete.json").is_file() and not args.overwrite:
            print(f"[{index}/{len(cases)}] skip {case.key}", flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(cases)}] start {case.key}", flush=True)
        try:
            payload = read_payload(case)
            caption = str(payload["input_caption"])
            frames, frame_indices = read_video_prefix(case.source_video, int(args.context_frames))
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
            sample = provider.build_sample(
                frames_tchw_01=frames_tchw_01,
                caption=caption,
                image_hw=(int(args.height), int(args.width)),
            )
            phrases = [
                str(track.source_phrase or track.phrase or f"object_{track_index:02d}")
                for track_index, track in enumerate(sample.object_tracks)
            ]
            if not phrases:
                raise RuntimeError(f"{case.key}: automatic grounding found no object tracks")

            query_frame = int(args.query_context_frame)
            if not 0 <= query_frame < len(frames_tchw_01):
                raise ValueError(
                    f"query-context-frame {query_frame} is outside {len(frames_tchw_01)} frames"
                )
            frame_rgb = np.transpose(
                (frames_tchw_01[query_frame] * 255.0).round().astype(np.uint8),
                (1, 2, 0),
            )
            cache = build_regions_from_grounding(
                case_key=case.key,
                grounding_sample=sample,
                object_phrases=phrases,
                context_frame_rgb=frame_rgb,
                query_frame_index=query_frame,
                points_per_region=int(args.points_per_region),
                object_erode_px=int(args.object_erode_px),
                background_erode_px=int(args.background_erode_px),
            )
            cache.metadata.update(
                {
                    "source_json": str(case.json_path),
                    "source_video": str(case.source_video),
                    "formal_compare_video": str(case.formal_video_path),
                    "formal_compare_json": str(case.formal_json_path),
                    "input_caption": caption,
                    "context_source_frame_indices": frame_indices.tolist(),
                    "grounding_prompts": phrases,
                    "object_phrases": phrases,
                    "object_phrase_source": (
                        "ViewerGroundingBoxProvider caption physical noun phrase extraction"
                    ),
                    "height": int(args.height),
                    "width": int(args.width),
                    "resize_mode": "stretch",
                    "analysis_protocol": "first_pixel_frame_to_first_latent_query",
                    "mask_source": (
                        "caption physical noun phrases -> GroundingDINO first-frame boxes "
                        "-> SAM2 video propagation"
                    ),
                    "uses_gt_instance_masks": False,
                }
            )
            save_region_cache(
                output_dir,
                cache,
                save_visualizations=bool(args.save_visualizations),
            )
            print(
                f"[{index}/{len(cases)}] complete {case.key}: "
                f"{len(phrases)} objects, query_frame={query_frame}",
                flush=True,
            )
        except Exception:
            error = traceback.format_exc()
            (output_dir / "error.txt").write_text(error, encoding="utf-8")
            failures.append(case.key)
            print(error, flush=True)

    if failures:
        raise SystemExit(f"region precompute failed for {len(failures)} cases: {failures}")
    marker = REGION_CACHE_ROOT / f"worker_{args.worker_id}_of_{args.num_workers}_complete.json"
    marker.write_text(
        json.dumps(
            {
                "worker_id": int(args.worker_id),
                "num_workers": int(args.num_workers),
                "case_count": len(cases),
                "cases": [case.key for case in cases],
                "query_frame": int(args.query_context_frame),
                "save_visualizations": bool(args.save_visualizations),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
