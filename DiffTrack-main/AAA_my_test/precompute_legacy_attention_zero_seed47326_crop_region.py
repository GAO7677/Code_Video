#!/usr/bin/env python3
"""Build the F00 704x1280 object-query cache for the cropped PhysicIQ case."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.utils.video_io import (  # noqa: E402
    preprocess_video_rgb_uint8,
    read_video_prefix,
)
from AAA_my_test.precompute_toydataset_sam2_regions import (  # noqa: E402
    build_provider,
    detect_and_track_objects,
)
from AAA_my_test.sam2_region_query_utils import (  # noqa: E402
    build_regions_from_grounding,
    save_region_cache,
)


CASE_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json"
)
CACHE_ROOT = Path(
    "/data/gaoya/agent-data/cache/"
    "wan22_ti2v_legacy_attention_zero_seed47326_regions_704x1280"
)
OBJECT_PHRASES = ("brown tennis ball", "orange block")


def main() -> None:
    payload = json.loads(CASE_JSON.read_text(encoding="utf-8"))
    case_key = CASE_JSON.stem
    output = CACHE_ROOT / case_key
    if (output / "complete.json").is_file():
        print(f"skip {case_key}: cache already complete")
        return
    output.mkdir(parents=True, exist_ok=True)
    frames, frame_indices = read_video_prefix(Path(payload["source_video"]), 8)
    context = preprocess_video_rgb_uint8(
        frames, (704, 1280), value_range="minus_one_to_one", resize_mode="stretch"
    )
    frames_tchw_01 = (
        ((context.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )
    provider = build_provider("cuda:0", points_per_region=8)
    sample = detect_and_track_objects(provider, frames_tchw_01, list(OBJECT_PHRASES))
    frame_rgb = np.transpose(
        (frames_tchw_01[0] * 255.0).round().astype(np.uint8), (1, 2, 0)
    )
    cache = build_regions_from_grounding(
        case_key=case_key,
        grounding_sample=sample,
        object_phrases=list(OBJECT_PHRASES),
        context_frame_rgb=frame_rgb,
        query_frame_index=0,
        points_per_region=8,
        object_erode_px=11,
        background_erode_px=31,
    )
    cache.metadata.update(
        {
            "source_json": str(CASE_JSON),
            "source_video": str(payload["source_video"]),
            "context_source_frame_indices": frame_indices.tolist(),
            "grounding_prompts": list(OBJECT_PHRASES),
            "object_phrases": list(OBJECT_PHRASES),
            "height": 704,
            "width": 1280,
            "resize_mode": "stretch",
            "analysis_protocol": "first_pixel_frame_to_first_latent_query",
            "mask_source": "GroundingDINO first-frame boxes -> SAM2 video propagation",
            "uses_gt_instance_masks": False,
        }
    )
    save_region_cache(output, cache)
    print(f"complete {case_key}: {len(OBJECT_PHRASES)} objects, query_frame=0")


if __name__ == "__main__":
    main()
