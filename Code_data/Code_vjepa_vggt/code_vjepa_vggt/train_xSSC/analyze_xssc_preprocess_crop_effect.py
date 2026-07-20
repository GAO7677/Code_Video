#!/usr/bin/env python3
"""Analyze how external video cropping interacts with xSSC center-crop preprocessing."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train
from code_vjepa_vggt.train_xSSC.visualize_xssc_slot_attention import (
    _cover_crop_to_tensor,
    _preprocess_xssc,
    _resolve_video_path,
)
from code_vjepa_vggt.utils.video_io import read_video_prefix


def _to_xssc_uint8(context_video_single: torch.Tensor) -> np.ndarray:
    xssc = _preprocess_xssc(context_video_single.unsqueeze(0), 256)[0]
    mean = xssc.new_tensor(train.XSSC_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = xssc.new_tensor(train.XSSC_IMAGENET_STD).view(1, 3, 1, 1)
    pixels = (xssc * std + mean).clamp(0.0, 255.0).byte()
    return pixels.permute(0, 2, 3, 1).cpu().numpy()


def _effective_xssc_source_box(preprocess: dict[str, object]) -> dict[str, float | list[float]]:
    source_h, source_w = [float(v) for v in preprocess["source_hw"]]  # type: ignore[index]
    crop_y, crop_x, _, _ = [float(v) for v in preprocess["cover_crop_yxhw_in_resized"]]  # type: ignore[index]
    target_h, target_w = [float(v) for v in preprocess["target_hw"]]  # type: ignore[index]
    scale = float(preprocess["cover_scale"])
    square = min(target_h, target_w)
    xssc_top = (target_h - square) / 2.0
    xssc_left = (target_w - square) / 2.0
    y0 = (crop_y + xssc_top) / scale
    y1 = (crop_y + xssc_top + square) / scale
    x0 = (crop_x + xssc_left) / scale
    x1 = (crop_x + xssc_left + square) / scale
    return {
        "source_hw": [source_h, source_w],
        "xssc_square_in_512x896": [xssc_top, xssc_left, square, square],
        "effective_source_xyxy": [x0, y0, x1, y1],
        "effective_source_yxhw": [y0, x0, y1 - y0, x1 - x0],
        "effective_source_fraction_xyxy": [x0 / source_w, y0 / source_h, x1 / source_w, y1 / source_h],
    }


def _save_contact(frames: list[np.ndarray], path: Path, *, label: str) -> None:
    images = [Image.fromarray(frame.astype(np.uint8)).resize((180, 180)) for frame in frames]
    canvas = Image.new("RGB", (180 * len(images), 210), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        canvas.paste(image, (i * 180, 0))
        draw.text((i * 180 + 6, 184), f"{label} f{i}", fill=(0, 0, 0))
    canvas.save(path, quality=95)


def _plot_diff_bars(rows: list[dict[str, float | int]], path: Path) -> None:
    xs = [int(row["frame_id"]) for row in rows]
    mae = [float(row["xssc_input_mae"]) for row in rows]
    corr = [float(row["xssc_input_corr"]) for row in rows]
    fig, ax1 = plt.subplots(figsize=(7.8, 4.4), dpi=170)
    ax1.bar(xs, mae, color="#4a82b8", label="MAE")
    ax1.set_xlabel("ctx frame")
    ax1.set_ylabel("xSSC 256 input MAE (0-255)")
    ax2 = ax1.twinx()
    ax2.plot(xs, corr, "-o", color="#a65f24", label="pixel corr")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("pixel correlation")
    ax1.set_title("Original vs crop_top60px after full xSSC preprocessing")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-json", type=Path, required=True)
    parser.add_argument("--cropped-json", type=Path, required=True)
    parser.add_argument("--slots-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=8)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_payloads = []
    for path in (args.original_json.expanduser().resolve(), args.cropped_json.expanduser().resolve()):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_video = _resolve_video_path(payload, path)
        frames, frame_indices = read_video_prefix(source_video, int(args.context_frames))
        context, preprocess = _cover_crop_to_tensor(
            frames,
            target_hw=(512, 896),
            cover_crop_hw=(512, 896),
        )
        xssc_uint8 = _to_xssc_uint8(context)
        case_payloads.append(
            {
                "json": str(path),
                "source_video": str(source_video),
                "frames": frames,
                "frame_indices": frame_indices.tolist(),
                "context": context,
                "preprocess": preprocess,
                "effective_box": _effective_xssc_source_box(preprocess),
                "xssc_uint8": xssc_uint8,
            }
        )

    original, cropped = case_payloads
    rows = []
    diff_frames = []
    for frame_id in range(int(args.context_frames)):
        a = original["xssc_uint8"][frame_id].astype(np.float32)  # type: ignore[index]
        b = cropped["xssc_uint8"][frame_id].astype(np.float32)  # type: ignore[index]
        diff = np.abs(a - b)
        corr = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
        rows.append(
            {
                "frame_id": int(frame_id),
                "xssc_input_mae": float(diff.mean()),
                "xssc_input_p95": float(np.percentile(diff, 95)),
                "xssc_input_max": float(diff.max()),
                "xssc_input_corr": corr,
            }
        )
        vis = np.clip(diff.mean(axis=-1) * 4.0, 0, 255).astype(np.uint8)
        diff_frames.append(np.stack([vis, vis, vis], axis=-1))

    csv_path = output_dir / "xssc_preprocess_crop_effect_per_frame.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _save_contact(list(original["xssc_uint8"]), output_dir / "original_xssc_256_inputs.jpg", label="orig")  # type: ignore[arg-type]
    _save_contact(list(cropped["xssc_uint8"]), output_dir / "crop_top60_xssc_256_inputs.jpg", label="crop")  # type: ignore[arg-type]
    _save_contact(diff_frames, output_dir / "xssc_256_absdiff_x4.jpg", label="diffx4")
    _plot_diff_bars(rows, output_dir / "xssc_preprocess_crop_effect_bars.png")

    slot_summary = {}
    try:
        original_slots = np.load(args.slots_root / "case1_025_Solid_Mechanics_0002_perspective-center_trimmed_frozen_xssc_slots.npz")["slots_tsd"].astype(np.float32)
        cropped_slots = np.load(args.slots_root / "case3_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px_frozen_xssc_slots.npz")["slots_tsd"].astype(np.float32)
        dots = []
        for t in range(original_slots.shape[0]):
            for s in range(original_slots.shape[1]):
                a = original_slots[t, s]
                b = cropped_slots[t, s]
                dots.append(float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1.0e-12)))
        slot_summary = {
            "same_frame_slot_cosine_mean": float(np.mean(dots)),
            "same_frame_slot_cosine_min": float(np.min(dots)),
            "same_frame_slot_cosine_max": float(np.max(dots)),
        }
    except FileNotFoundError:
        slot_summary = {"error": "slot npz files not found"}

    summary = {
        "method": "Compare original vs crop_top60px after full pipeline: cover-crop to 512x896, xSSC center-square crop, resize to 256.",
        "original": {k: original[k] for k in ("json", "source_video", "frame_indices", "preprocess", "effective_box")},
        "cropped": {k: cropped[k] for k in ("json", "source_video", "frame_indices", "preprocess", "effective_box")},
        "per_frame_mean_mae": float(np.mean([row["xssc_input_mae"] for row in rows])),
        "per_frame_mean_corr": float(np.mean([row["xssc_input_corr"] for row in rows])),
        "slot_summary": slot_summary,
        "outputs": {
            "csv": csv_path.name,
            "bars": "xssc_preprocess_crop_effect_bars.png",
            "original_inputs": "original_xssc_256_inputs.jpg",
            "cropped_inputs": "crop_top60_xssc_256_inputs.jpg",
            "absdiff": "xssc_256_absdiff_x4.jpg",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>xSSC preprocess crop effect</title>
<style>body{{margin:0;background:#101114;color:#eceff4;font:14px Arial,sans-serif}}main{{max-width:1500px;margin:auto;padding:22px}}img{{width:100%;height:auto;background:#050608}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}}article{{background:#171a20;border:1px solid #2b2f36;border-radius:8px;padding:12px;overflow:hidden}}code{{color:#b8d7ff}}</style></head>
<body><main><h1>xSSC preprocessing crop effect</h1>
<p>After full preprocessing, mean pixel MAE is <b>{summary['per_frame_mean_mae']:.2f}/255</b>, mean pixel corr is <b>{summary['per_frame_mean_corr']:.4f}</b>. Same frame/slot frozen xSSC cosine mean is <b>{slot_summary.get('same_frame_slot_cosine_mean', 0):.4f}</b>.</p>
<p><a href='summary.json'>summary JSON</a> | <a href='{csv_path.name}'>per-frame CSV</a></p>
<div class='grid'>
<article><h2>original xSSC 256 inputs</h2><img src='original_xssc_256_inputs.jpg'></article>
<article><h2>crop_top60 xSSC 256 inputs</h2><img src='crop_top60_xssc_256_inputs.jpg'></article>
<article><h2>abs diff x4</h2><img src='xssc_256_absdiff_x4.jpg'></article>
<article><h2>metrics</h2><img src='xssc_preprocess_crop_effect_bars.png'></article>
</div></main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"summary": str(output_dir / "summary.json"), "index": str(output_dir / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
