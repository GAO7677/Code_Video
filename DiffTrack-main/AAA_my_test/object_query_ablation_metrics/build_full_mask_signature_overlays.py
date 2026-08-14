#!/usr/bin/env python3
"""Render full-mask latent-token signatures over the frozen Baseline videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from AAA_my_test.object_query_ablation_metrics.full_mask_signature_regions import (
    LATENT_GRID,
    build_signature_partition,
    unpack_mask_cache,
)
from AAA_my_test.object_query_ablation_metrics.run_full_mask_signature_ablations import (
    DEFAULT_CASES,
    DEFAULT_OUTPUT_ROOT,
    mask_cache_for,
    samples_by_key,
)


COLORS = (
    (255, 190, 30), (38, 205, 255), (240, 105, 210), (100, 220, 105),
    (210, 150, 75), (95, 115, 255), (215, 215, 70),
)
SHARED = (60, 55, 255)


def draw_cell(frame, spatial: int, color, alpha: float, thickness: int) -> None:
    gh, gw = LATENT_GRID[1:]
    h, w = frame.shape[:2]
    row, col = divmod(spatial, gw)
    x0, x1 = round(col * w / gw), round((col + 1) * w / gw)
    y0, y1 = round(row * h / gh), round((row + 1) * h / gh)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), color, thickness, cv2.LINE_AA)


def label(frame, text: str, x: int, y: int, scale: float = 0.52) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, 1)
    cv2.rectangle(frame, (x - 4, y - th - 5), (x + tw + 4, y + base + 4), (7, 11, 18), -1)
    cv2.putText(frame, text, (x, y), font, scale, (250, 252, 255), 1, cv2.LINE_AA)


def render(baseline: Path, output: Path, partition) -> dict:
    cap = cv2.VideoCapture(str(baseline))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {baseline}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output.parent.mkdir(parents=True, exist_ok=True)
    avi = output.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {avi}")
    signature_colors = {
        signature: (SHARED if signature.bit_count() > 1 else COLORS[(signature.bit_length() - 1) % len(COLORS)])
        for signature in partition.signature_rows
    }
    written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            latent_t = min(LATENT_GRID[0] - 1, (written + 2) // 4)
            for signature, rows_by_time in partition.signature_rows_by_time.items():
                color = signature_colors[signature]
                for global_row in rows_by_time[latent_t]:
                    spatial = global_row - latent_t * LATENT_GRID[1] * LATENT_GRID[2]
                    draw_cell(frame, spatial, color, 0.42 if signature.bit_count() > 1 else 0.16, 3 if signature.bit_count() > 1 else 1)
            anchor = partition.anchor_frames[latent_t]
            labels = []
            for signature, rows_by_time in partition.signature_rows_by_time.items():
                count = len(rows_by_time[latent_t])
                if count:
                    labels.append(f"{partition.signature_label(signature)}:{count}")
            label(frame, f"F{written:02d} -> latent t{latent_t:02d} (anchor F{anchor:02d})", 16, 29, 0.62)
            label(frame, " | ".join(labels), 16, 55, 0.44)
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    if written != expected:
        raise RuntimeError(f"decoded {written}/{expected} frames")
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(avi),
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], check=True)
    avi.unlink(missing_ok=True)
    return {"fps": fps, "frames": written, "width": w, "height": h}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    samples = samples_by_key()
    units = []
    for case in args.cases:
        sample = samples[(case, args.seed)]
        baseline = Path(str(sample["baseline_video"]))
        names = tuple(row["region_name"] for row in sample["regions"] if row.get("region_type") == "object")
        mask_path = mask_cache_for(case, args.seed)
        partition = build_signature_partition(unpack_mask_cache(mask_path, baseline), names)
        output = args.output_root / "overlays" / case / f"seed_{args.seed:05d}" / "full_mask_signatures.mp4"
        meta = render(baseline, output, partition) if args.overwrite or not output.is_file() else {}
        units.append({
            "case": case,
            "seed": args.seed,
            "baseline_video": str(baseline),
            "mask_cache": str(mask_path),
            "overlay_video": str(output),
            "partition": partition.audit(),
            "video": meta,
        })
        print(f"complete {case}: {output}", flush=True)
    catalog = args.output_root / "overlay_catalog.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(json.dumps({"units": units}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(catalog)


if __name__ == "__main__":
    main()
