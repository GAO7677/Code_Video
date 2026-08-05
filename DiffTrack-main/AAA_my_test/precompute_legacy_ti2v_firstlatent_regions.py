#!/usr/bin/env python3
"""Build first-frame GroundingDINO + SAM2 query caches for six legacy TI2V cases."""

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
from AAA_my_test.legacy_ti2v_firstlatent_common import CASES, REGION_CACHE_ROOT, read_payload
from AAA_my_test.precompute_toydataset_sam2_regions import build_provider, detect_and_track_objects
from AAA_my_test.sam2_region_query_utils import (
    build_regions_from_grounding,
    save_region_cache,
    save_region_query_visualizations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def object_phrases(case) -> list[str]:
    payload = json.loads((case.old_region_cache / "regions.json").read_text(encoding="utf-8"))
    phrases = [
        str(region["region_phrase"])
        for region in payload["regions"]
        if region.get("region_type") == "object" and region.get("region_phrase")
    ]
    if not phrases:
        raise RuntimeError(f"no object phrases in {case.old_region_cache}")
    return phrases


def main() -> None:
    args = parse_args()
    REGION_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    provider = build_provider(str(args.device), points_per_region=8)
    failures = []
    for index, case in enumerate(CASES, start=1):
        output_dir = REGION_CACHE_ROOT / case.key
        if (output_dir / "complete.json").is_file() and not args.overwrite:
            print(f"[{index}/{len(CASES)}] skip {case.key}", flush=True)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = read_payload(case)
            source_video = Path(payload["source_video"])
            phrases = object_phrases(case)
            frames, frame_indices = read_video_prefix(source_video, 8)
            context = preprocess_video_rgb_uint8(
                frames, (704, 1280), value_range="minus_one_to_one", resize_mode="stretch"
            )
            frames_tchw_01 = (
                ((context.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
                .clamp(0.0, 1.0)
                .cpu()
                .numpy()
            )
            sample = detect_and_track_objects(provider, frames_tchw_01, phrases)
            frame_rgb = np.transpose(
                (frames_tchw_01[0] * 255.0).round().astype(np.uint8), (1, 2, 0)
            )
            cache = build_regions_from_grounding(
                case_key=case.key,
                grounding_sample=sample,
                object_phrases=phrases,
                context_frame_rgb=frame_rgb,
                query_frame_index=0,
                points_per_region=8,
                object_erode_px=11,
                background_erode_px=31,
            )
            cache.metadata.update(
                {
                    "source_json": str(case.json_path),
                    "context_video": str(source_video),
                    "context_source_frame_indices": frame_indices.tolist(),
                    "grounding_prompts": phrases,
                    "object_phrases": phrases,
                    "height": 704,
                    "width": 1280,
                    "resize_mode": "stretch",
                    "analysis_protocol": "first_pixel_frame_to_first_latent_query",
                    "mask_source": "GroundingDINO first-frame boxes -> SAM2 video propagation",
                    "uses_gt_instance_masks": False,
                }
            )
            save_region_cache(output_dir, cache)
            save_region_query_visualizations(output_dir, cache)
            print(
                f"[{index}/{len(CASES)}] complete {case.key}: "
                f"{len(phrases)} objects, query_frame=0",
                flush=True,
            )
        except Exception:
            error = traceback.format_exc()
            (output_dir / "error.txt").write_text(error, encoding="utf-8")
            failures.append(case.key)
            print(error, flush=True)
    if failures:
        raise SystemExit(f"first-frame region precompute failed: {failures}")
    (REGION_CACHE_ROOT / "all_complete.json").write_text(
        json.dumps({"cases": [case.key for case in CASES], "query_frame": 0}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
