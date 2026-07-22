#!/usr/bin/env python3
"""Render MOVi-C GT boxes exactly as exposed to the xSSC data pipeline."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import imageio_ffmpeg
import numpy as np


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(EXPERIMENT / "upstream"))

DEFAULT_SUBSET = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/"
    "val_subset.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/movi_c_gt_box_overlay_test_monitor_5_20260722"
)
PALETTE_RGB = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
        [236, 72, 153],
        [132, 204, 22],
        [20, 184, 166],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--subset-file", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def normalized_box_to_pixels(box, height, width):
    x1, y1, x2, y2 = [float(value) for value in box]
    if x2 - x1 <= 1.0e-6 or y2 - y1 <= 1.0e-6:
        return None
    return (
        int(np.clip(round(x1 * width), 0, width - 1)),
        int(np.clip(round(y1 * height), 0, height - 1)),
        int(np.clip(round(x2 * width), 0, width - 1)),
        int(np.clip(round(y2 * height), 0, height - 1)),
    )


def draw_boxes(frame_rgb, boxes):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = frame_bgr.shape[:2]
    for object_index, box in enumerate(boxes):
        pixels = normalized_box_to_pixels(box, height, width)
        if pixels is None:
            continue
        x1, y1, x2, y2 = pixels
        color_rgb = PALETTE_RGB[object_index % len(PALETTE_RGB)]
        color_bgr = tuple(int(value) for value in color_rgb[::-1])
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)
        label = f"obj {object_index + 1:02d}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )
        label_y1 = max(0, y1 - text_height - 5)
        label_y2 = min(height - 1, label_y1 + text_height + 5)
        label_x2 = min(width - 1, x1 + text_width + 5)
        cv2.rectangle(
            frame_bgr,
            (x1, label_y1),
            (label_x2, label_y2),
            color_bgr,
            thickness=-1,
        )
        text_color = (10, 10, 10) if int(color_rgb.mean()) > 140 else (255, 255, 255)
        cv2.putText(
            frame_bgr,
            label,
            (x1 + 2, label_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            text_color,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def add_panel_title(frame_rgb, title):
    output = frame_rgb.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 20), (18, 18, 18), -1)
    cv2.putText(
        output,
        title,
        (6, 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


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


def build_html(cases):
    cards = []
    for case in cases:
        cards.append(
            f"""
            <article>
              <h2>test index {case['dataset_index']:03d}</h2>
              <p>{case['num_foreground_objects']} foreground objects, 24 frames</p>
              <video controls muted loop playsinline preload="metadata" src="{html.escape(case['video'])}"></video>
              <img loading="lazy" src="{html.escape(case['contact_sheet'])}" alt="First, middle, and last frame box overlays">
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-C GT box overlays</title><style>
body{{margin:0;background:#101214;color:#f3f4f6;font:14px Arial,sans-serif}}main{{max-width:1160px;margin:auto;padding:24px}}
h1{{font-size:24px}}.note{{color:#aeb5bf;margin-bottom:20px}}article{{border-top:1px solid #343a40;padding:18px 0}}
h2{{font-size:17px;margin:0 0 4px}}p{{color:#aeb5bf}}video,img{{display:block;width:100%;height:auto;margin-top:10px;background:#000}}
</style></head><body><main><h1>MOVi-C GT box overlays</h1>
<p class="note">Left: raw 256x256 frame. Right: normalized xyxy boxes recomputed from instance masks by the xSSC dataset loader.</p>
{''.join(cards)}</main></body></html>"""


def main():
    args = parse_args()
    from object_centric_bench.datum import MOViTFRecord

    subset = json.loads(args.subset_file.read_text())
    indices = [int(index) for index in subset["indices"][: args.num_cases]]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = MOViTFRecord(
        data_file="kubric-movi/movi-c",
        split="test",
        extra_keys=["segment", "bbox"],
        base_dir=args.data_dir.resolve(),
    )

    cases = []
    all_contact_rows = []
    for position, dataset_index in enumerate(indices, start=1):
        sample = dataset[dataset_index]
        video = sample["video"].permute(0, 2, 3, 1).cpu().numpy()
        boxes = sample["bbox"].cpu().numpy()
        rendered = []
        overlay_frames = []
        for frame_index, (frame, frame_boxes) in enumerate(zip(video, boxes)):
            overlay = draw_boxes(frame, frame_boxes)
            overlay_frames.append(overlay)
            left = add_panel_title(frame, f"raw | frame {frame_index:02d}")
            right = add_panel_title(overlay, f"GT boxes | frame {frame_index:02d}")
            rendered.append(np.concatenate([left, right], axis=1))

        stem = f"case_{position:02d}_test_index_{dataset_index:03d}"
        video_path = output_dir / f"{stem}.mp4"
        write_video(video_path, rendered, args.fps)
        selected = [0, len(overlay_frames) // 2, len(overlay_frames) - 1]
        contact = np.concatenate(
            [
                add_panel_title(
                    overlay_frames[frame_index], f"frame {frame_index:02d}"
                )
                for frame_index in selected
            ],
            axis=1,
        )
        contact_path = output_dir / f"{stem}_contact.png"
        cv2.imwrite(
            str(contact_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR)
        )
        cases.append(
            {
                "position": position,
                "dataset_index": dataset_index,
                "num_foreground_objects": int(boxes.shape[1]),
                "video_shape": list(video.shape),
                "bbox_shape": list(boxes.shape),
                "bbox_format": "normalized xyxy",
                "video": video_path.name,
                "contact_sheet": contact_path.name,
            }
        )
        all_contact_rows.append(contact)
        print(
            f"[{position}/{len(indices)}] index={dataset_index} objects={boxes.shape[1]} {video_path}",
            flush=True,
        )

    overview = np.concatenate(all_contact_rows, axis=0)
    overview_path = output_dir / "overview.png"
    cv2.imwrite(str(overview_path), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    metadata = {
        "dataset": str(args.data_dir.resolve() / "kubric-movi/movi-c"),
        "split": "test",
        "selection": "first five indices from the fixed test_monitor_300 subset",
        "subset_file": str(args.subset_file.resolve()),
        "indices": indices,
        "fps": args.fps,
        "cases": cases,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(build_html(cases))
    print(json.dumps({"output_dir": str(output_dir), "overview": str(overview_path)}, indent=2))


if __name__ == "__main__":
    main()
