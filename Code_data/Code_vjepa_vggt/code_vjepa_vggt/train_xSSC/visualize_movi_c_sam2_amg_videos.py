#!/usr/bin/env python3
"""Render frame-wise prompt-free SAM2 AMG overlays for fixed MOVi-C videos."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
import time

import imageio_ffmpeg
import numpy as np
import torch


TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
EXPERIMENT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(TRAIN_XSSC_ROOT))
sys.path.insert(0, str(EXPERIMENT / "upstream"))
sys.path.insert(0, "/home/gaoya/Grounded-SAM-2-main")

from visualize_movi_c_sam2_amg import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_REPORT,
    add_title,
    draw_selected_boxes,
    overlay_masks,
    resolve_sam2_config_name,
    select_xssc_candidates,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--max-selected", type=int, default=11)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=0.70)
    parser.add_argument("--background-area-ratio", type=float, default=0.15)
    parser.add_argument("--background-span-ratio", type=float, default=0.85)
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument("--duplicate-containment", type=float, default=0.92)
    return parser.parse_args()


def write_video(path, frames, fps):
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def build_html_section(payload):
    cards = []
    for case in payload["cases"]:
        cards.append(
            f"<article><h2>test index {case['dataset_index']:03d}</h2>"
            f"<p>24 frames; raw AMG masks/frame "
            f"{case['raw_mask_count_min']} to {case['raw_mask_count_max']} "
            f"(mean {case['raw_mask_count_mean']:.1f}); filtered masks/frame "
            f"mean {case['selected_mask_count_mean']:.1f}.</p>"
            f"<video controls muted loop playsinline preload='metadata' "
            f"src='{html.escape(case['video'])}'></video></article>"
        )
    return (
        "<!-- SAM2_AMG_VIDEO_START -->"
        "<section id='sam2-amg-videos'><h1>SAM2 AMG frame-wise video overlays</h1>"
        "<p class='note'>Every frame is segmented independently without caption, "
        "detector, GT, or user prompts. Colors and A-numbers are per-frame masks, "
        "not persistent tracking IDs.</p>"
        f"{''.join(cards)}</section>"
        "<!-- SAM2_AMG_VIDEO_END -->"
    )


def update_report_html(report_dir, payload):
    index_path = report_dir / "index.html"
    page = index_path.read_text()
    page = re.sub(
        r"<!-- SAM2_AMG_VIDEO_START -->.*?<!-- SAM2_AMG_VIDEO_END -->",
        "",
        page,
        flags=re.DOTALL,
    )
    section = build_html_section(payload)
    if "</main>" not in page:
        raise RuntimeError(f"Cannot find </main> in {index_path}")
    index_path.write_text(page.replace("</main>", section + "</main>"))


def main():
    args = parse_args()
    report_dir = args.report_dir.resolve()
    amg_metrics_file = report_dir / "sam2_amg/metrics.json"
    if not amg_metrics_file.is_file():
        raise FileNotFoundError(amg_metrics_file)

    from object_centric_bench.datum import MOViTFRecord
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    amg_metrics = json.loads(amg_metrics_file.read_text())
    indices = [int(index) for index in amg_metrics["indices"]]
    dataset = MOViTFRecord(
        data_file="kubric-movi/movi-c",
        split="test",
        extra_keys=["segment", "bbox"],
        base_dir=args.data_dir.resolve(),
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    resolved_config_name = resolve_sam2_config_name(args.sam2_config)
    sam2 = build_sam2(
        resolved_config_name,
        str(args.sam2_checkpoint.resolve()),
        device=str(device),
        mode="eval",
    )
    generator = SAM2AutomaticMaskGenerator(sam2)

    output_dir = report_dir / "sam2_amg/videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for position, dataset_index in enumerate(indices, start=1):
        sample = dataset[dataset_index]
        video = sample["video"].permute(0, 2, 3, 1).contiguous().numpy()
        rendered = []
        frame_records = []
        case_started = time.time()
        for frame_index, image in enumerate(video):
            started = time.time()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                annotations = generator.generate(image)
            selected = select_xssc_candidates(
                annotations, image.shape[0] * image.shape[1], args
            )
            raw_overlay = overlay_masks(image, annotations)
            selected_overlay = draw_selected_boxes(
                overlay_masks(image, selected), selected
            )
            rendered.append(
                np.concatenate(
                    [
                        add_title(image, f"raw | frame {frame_index:02d}"),
                        add_title(
                            raw_overlay,
                            f"AMG raw | n={len(annotations)}",
                        ),
                        add_title(
                            selected_overlay,
                            f"AMG filtered top-11 | n={len(selected)}",
                        ),
                    ],
                    axis=1,
                )
            )
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "raw_mask_count": len(annotations),
                    "selected_mask_count": len(selected),
                    "seconds": time.time() - started,
                }
            )
            print(
                f"[video {position}/{len(indices)} frame "
                f"{frame_index + 1}/{len(video)}] index={dataset_index} "
                f"raw={len(annotations)} selected={len(selected)}",
                flush=True,
            )

        video_path = output_dir / f"case_{position:02d}_test_index_{dataset_index:03d}.mp4"
        write_video(video_path, rendered, args.fps)
        raw_counts = [record["raw_mask_count"] for record in frame_records]
        selected_counts = [record["selected_mask_count"] for record in frame_records]
        cases.append(
            {
                "dataset_index": dataset_index,
                "frames": len(video),
                "video": str(video_path.relative_to(report_dir)),
                "raw_mask_count_min": min(raw_counts),
                "raw_mask_count_max": max(raw_counts),
                "raw_mask_count_mean": sum(raw_counts) / len(raw_counts),
                "selected_mask_count_min": min(selected_counts),
                "selected_mask_count_max": max(selected_counts),
                "selected_mask_count_mean": sum(selected_counts) / len(selected_counts),
                "seconds": time.time() - case_started,
                "frame_records": frame_records,
            }
        )

    payload = {
        "method": "independent SAM2 AMG on every MOVi-C frame",
        "uses_caption": False,
        "uses_external_detector": False,
        "uses_gt_prompt": False,
        "persistent_instance_ids": False,
        "indices": indices,
        "fps": args.fps,
        "cases": cases,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    update_report_html(report_dir, payload)
    print(f"report={report_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
