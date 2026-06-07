#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.mask_tracking import (
    GroundingDINOTextDetector,
    SAM2VideoMaskTracker,
    build_caption_prompt_boxes,
    build_mask_track_outputs,
    build_proxy_prompt_box,
)
from phys_state_video.proxy_state import read_video_frames
from phys_state_video.schemas import StateIndex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export caption_gdino + SAM2 cache for the tailquery benchmark manifest."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-mode", choices=["proxy_box", "caption_gdino"], default="caption_gdino")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sam2-config",
        default="/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--sam2-ckpt",
        default="/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt",
    )
    parser.add_argument("--gdino-repo-root", default="/home/gaoya/Grounded-SAM-2-main")
    parser.add_argument(
        "--gdino-config",
        default="/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/GroundingDINO_SwinT_OGC.cfg.py",
    )
    parser.add_argument(
        "--gdino-ckpt",
        default="/data/gaoya/ckpt/GroundingDINO_SwinT_OGC/groundingdino_swint_ogc.pth",
    )
    parser.add_argument("--gdino-box-threshold", type=float, default=0.25)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-max-boxes", type=int, default=4)
    return parser.parse_args()


def choose_primary_object_index(states: np.ndarray, boxes: np.ndarray) -> int:
    areas = np.maximum((boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1]), 0.0)
    visibility = np.clip(states[..., StateIndex.VISIBILITY], 0.0, 1.0)
    existence = np.clip(states[..., StateIndex.EXISTENCE], 0.0, 1.0)
    scores = (areas * visibility * existence).mean(axis=0)
    return int(np.argmax(scores))


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cases"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or "cuda"
    tracker = SAM2VideoMaskTracker(
        device=device,
        model_cfg=args.sam2_config,
        checkpoint_path=args.sam2_ckpt,
    )
    text_detector = None
    if args.prompt_mode == "caption_gdino":
        text_detector = GroundingDINOTextDetector(
            repo_root=args.gdino_repo_root,
            config_path=args.gdino_config,
            checkpoint_path=args.gdino_ckpt,
            device=device,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
            max_boxes=args.gdino_max_boxes,
        )

    case_records = []
    for spec in manifest["cases"]:
        case_id = str(spec["case_id"])
        source_path = Path(spec["source_video"])
        frames = read_video_frames(
            source_path,
            resize_height=int(spec["height"]),
            resize_width=int(spec["width"]),
        )
        start = int(spec["clip_start"])
        context_steps = int(spec["context_steps"])
        future_steps = int(spec["future_steps"])
        clip = frames[start : start + context_steps + future_steps]
        prompt_frame_idx = context_steps - 1
        proxy_guidance_box = build_proxy_prompt_box(clip, prompt_frame_idx=prompt_frame_idx)
        prompt_boxes_xyxy = proxy_guidance_box[None]
        prompt_phrases: list[str] = []
        resolved_prompt_mode = "proxy_box"
        if args.prompt_mode == "caption_gdino" and text_detector is not None and str(spec["caption"]).strip():
            detection = build_caption_prompt_boxes(
                clip,
                prompt_frame_idx=prompt_frame_idx,
                caption=str(spec["caption"]),
                detector=text_detector,
                guidance_box_xyxy=proxy_guidance_box,
            )
            if detection.boxes_xyxy.shape[0] > 0:
                prompt_boxes_xyxy = detection.boxes_xyxy.astype(np.float32)
                prompt_phrases = [str(x) for x in detection.phrases]
                resolved_prompt_mode = detection.prompt_mode
            else:
                resolved_prompt_mode = "proxy_box_fallback"
        outputs = build_mask_track_outputs(
            clip,
            prompt_frame_idx=prompt_frame_idx,
            prompt_boxes_xyxy=prompt_boxes_xyxy,
            prompt_mode=resolved_prompt_mode,
            tracker=tracker,
        )
        primary_idx = choose_primary_object_index(outputs.states, outputs.boxes)
        npz_path = cache_dir / f"{case_id}.npz"
        np.savez_compressed(
            npz_path,
            states=outputs.states.astype(np.float32),
            boxes=outputs.boxes.astype(np.float32),
            masks=outputs.masks.astype(np.uint8),
            prompt_boxes_xyxy=prompt_boxes_xyxy.astype(np.float32),
            primary_idx=np.asarray([primary_idx], dtype=np.int64),
        )
        case_records.append(
            {
                "case_id": case_id,
                "npz": str(npz_path),
                "prompt_mode": resolved_prompt_mode,
                "prompt_box_count": int(prompt_boxes_xyxy.shape[0]),
                "prompt_phrases": prompt_phrases,
                "primary_object_idx": int(primary_idx),
            }
        )

    report = {
        "manifest_json": str(Path(args.manifest_json)),
        "prompt_mode": args.prompt_mode,
        "case_count": len(case_records),
        "cases": case_records,
    }
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sam2 cache: {output_dir}")
    print(f"cases: {len(case_records)}")


if __name__ == "__main__":
    main()
