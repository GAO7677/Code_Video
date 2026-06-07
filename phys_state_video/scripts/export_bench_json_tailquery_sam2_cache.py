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
    build_mask_track_outputs,
    choose_primary_object_index,
    resolve_prompt_boxes,
)
from phys_state_video.proxy_state import read_video_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export caption_gdino + SAM2 cache for the tailquery benchmark manifest."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-mode", choices=["proxy_box", "caption_gdino"], default="caption_gdino")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sam2-device", default=None)
    parser.add_argument("--gdino-device", default=None)
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
    parser.add_argument("--caption-use-proxy-guidance", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cases"
    cache_dir.mkdir(parents=True, exist_ok=True)

    shared_device = args.device or "cuda"
    sam2_device = args.sam2_device or shared_device
    gdino_device = args.gdino_device or shared_device
    tracker = SAM2VideoMaskTracker(
        device=sam2_device,
        model_cfg=args.sam2_config,
        checkpoint_path=args.sam2_ckpt,
    )
    text_detector = None
    if args.prompt_mode == "caption_gdino":
        text_detector = GroundingDINOTextDetector(
            repo_root=args.gdino_repo_root,
            config_path=args.gdino_config,
            checkpoint_path=args.gdino_ckpt,
            device=gdino_device,
            box_threshold=args.gdino_box_threshold,
            text_threshold=args.gdino_text_threshold,
            max_boxes=args.gdino_max_boxes,
        )

    case_records = []
    total_cases = len(manifest["cases"])

    def write_report() -> None:
        report = {
            "manifest_json": str(Path(args.manifest_json)),
            "prompt_mode": args.prompt_mode,
            "sam2_device": sam2_device,
            "gdino_device": gdino_device,
            "case_count": len(case_records),
            "cases": case_records,
        }
        (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for spec in manifest["cases"]:
        case_id = str(spec["case_id"])
        case_idx = len(case_records) + 1
        print(f"[{case_idx}/{total_cases}] start case={case_id}", flush=True)
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
        prompt_boxes_xyxy: np.ndarray
        prompt_phrases: list[str] = []
        prompt_scores = np.zeros((0,), dtype=np.float32)
        resolved_prompt_mode = "proxy_box"
        if args.prompt_mode == "caption_gdino" and text_detector is not None and str(spec["caption"]).strip():
            print(f"[{case_idx}/{total_cases}] gdino detect case={case_id}", flush=True)
            detection = resolve_prompt_boxes(
                clip,
                prompt_frame_idx=prompt_frame_idx,
                prompt_mode="caption_gdino",
                caption=str(spec["caption"]),
                detector=text_detector,
                use_proxy_guidance_for_caption=bool(args.caption_use_proxy_guidance),
            )
            prompt_boxes_xyxy = detection.boxes_xyxy.astype(np.float32)
            prompt_phrases = [str(x) for x in detection.phrases]
            prompt_scores = detection.scores.astype(np.float32)
            resolved_prompt_mode = detection.prompt_mode
            print(
                f"[{case_idx}/{total_cases}] gdino done case={case_id} mode={resolved_prompt_mode} "
                f"boxes={int(prompt_boxes_xyxy.shape[0])}",
                flush=True,
            )
        else:
            detection = resolve_prompt_boxes(
                clip,
                prompt_frame_idx=prompt_frame_idx,
                prompt_mode="proxy_box",
            )
            prompt_boxes_xyxy = detection.boxes_xyxy.astype(np.float32)
            resolved_prompt_mode = detection.prompt_mode
        print(f"[{case_idx}/{total_cases}] sam2 track case={case_id}", flush=True)
        outputs = build_mask_track_outputs(
            clip,
            prompt_frame_idx=prompt_frame_idx,
            prompt_boxes_xyxy=prompt_boxes_xyxy,
            prompt_mode=resolved_prompt_mode,
            tracker=tracker,
        )
        primary_idx = choose_primary_object_index(
            outputs.states,
            outputs.boxes,
            prompt_scores=prompt_scores,
        )
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
        write_report()
        print(
            f"[{case_idx}/{total_cases}] done case={case_id} primary={int(primary_idx)} "
            f"npz={npz_path.name}",
            flush=True,
        )

    write_report()
    print(f"sam2 cache: {output_dir}")
    print(f"cases: {len(case_records)}")


if __name__ == "__main__":
    main()
