#!/usr/bin/env python3
"""Render Top/Bottom PCK@32 Q@K trajectories on original-video anchors."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case"
)
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_overlay_case001_s039"
)
CASE = "case_001_ball_roll"
STEP = 39
MODELS = (
    ("gt", "GT Teacher-Forced", "#55d6be"),
    ("lora", "LoRA", "#f4a261"),
    ("baseline", "Wan2.2 Baseline", "#e76f51"),
)
SOURCE_VIDEO = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset/cases/"
    "case_001_ball_roll/base/videos/case_001_ball_roll_base.mp4"
)
FFMPEG = "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ANCHORS = (0, 4, 8, 12, 16, 20, 24)
PANEL_WIDTH = 640
PANEL_HEIGHT = 366
HEADER = 78
FOOTER = 38
FPS = 12
REPEAT = 6


def load_video_frames() -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(SOURCE_VIDEO))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < 25:
        raise RuntimeError(f"expected at least 25 source frames, got {len(frames)}")
    return frames


def selected_combinations() -> list[dict]:
    frame = pd.read_csv(SOURCE_ROOT / "three_model_combined_summary.csv")
    frame = frame[(frame.step == STEP) & (frame.scope == "objects")].sort_values(
        ["macro_pck32", "macro_mean_error_px"], ascending=[False, True]
    )
    selected = []
    for group, rows in (("top", frame.head(6)), ("bottom", frame.tail(6).sort_values("macro_pck32"))):
        for rank, (_, row) in enumerate(rows.iterrows(), 1):
            selected.append(
                {
                    "group": group,
                    "rank": rank,
                    "block": int(row.block),
                    "head": int(row["head"]),
                    "macro_pck32": float(row.macro_pck32),
                    "worst_model_macro_pck32": float(row.worst_model_macro_pck32),
                    "macro_mean_error_px": float(row.macro_mean_error_px),
                    "gt_macro_pck32": float(row.gt_macro_pck32),
                    "lora_macro_pck32": float(row.lora_macro_pck32),
                    "baseline_macro_pck32": float(row.baseline_macro_pck32),
                }
            )
    return selected


def load_case_metrics(model: str) -> dict[tuple[int, int], float]:
    path = SOURCE_ROOT / model / "cases" / CASE / "metrics.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in rows:
        if (
            int(row.get("step_index", -1)) == STEP
            and row.get("region_type") == "object"
            and str(row.get("method", "")).startswith("qk_head")
        ):
            head = int(str(row["method"]).replace("qk_head", ""))
            result[(int(row["layer"]), head)] = float(row["pck32"])
    return result


def color(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def valid(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def draw_tracks(
    image: Image.Image,
    gt: np.ndarray,
    visibility: np.ndarray,
    prediction: np.ndarray,
    latent: int,
    prediction_color: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(image)
    sx, sy = PANEL_WIDTH / 896, PANEL_HEIGHT / 512
    for point in range(8):
        gt_path = []
        pred_path = []
        for index in range(latent + 1):
            if visibility[index, point] and valid(gt[index, point]):
                gt_path.append((gt[index, point, 0] * sx, gt[index, point, 1] * sy))
            if valid(prediction[index, point]):
                pred_path.append(
                    (prediction[index, point, 0] * sx, prediction[index, point, 1] * sy)
                )
        if len(gt_path) > 1:
            draw.line(gt_path, fill=(255, 255, 255), width=2)
        if len(pred_path) > 1:
            draw.line(pred_path, fill=prediction_color, width=4)
        gt_point = gt[latent, point]
        pred_point = prediction[latent, point]
        if visibility[latent, point] and valid(gt_point):
            gx, gy = gt_point[0] * sx, gt_point[1] * sy
            draw.ellipse((gx - 7, gy - 7, gx + 7, gy + 7), fill=(15, 20, 18), outline="white", width=3)
        if valid(pred_point):
            px, py = pred_point[0] * sx, pred_point[1] * sy
            if visibility[latent, point] and valid(gt_point):
                draw.line((gx, gy, px, py), fill=(190, 190, 190), width=1)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=prediction_color, outline=(15, 20, 18), width=2)


def render_video(
    item: dict,
    source_frames: list[np.ndarray],
    gt_tracks: np.ndarray,
    gt_visibility: np.ndarray,
    predictions: dict[str, np.ndarray],
    case_metrics: dict[str, dict[tuple[int, int], float]],
) -> tuple[str, str]:
    stem = f"{item['group']}_rank{item['rank']:02d}_L{item['block']:02d}_H{item['head']:02d}"
    video_name = stem + ".mp4"
    poster_name = stem + ".jpg"
    video_path = OUTPUT_ROOT / "videos" / video_name
    poster_path = OUTPUT_ROOT / "posters" / poster_name
    width = PANEL_WIDTH * 3
    height = HEADER + PANEL_HEIGHT + FOOTER
    title_font = ImageFont.truetype(FONT, 25)
    label_font = ImageFont.truetype(FONT, 18)
    small_font = ImageFont.truetype(FONT, 14)
    command = [
        FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(video_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    first_frame = None
    for latent, source_index in enumerate(ANCHORS):
        panel_source = Image.fromarray(source_frames[source_index]).resize(
            (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS
        )
        canvas = Image.new("RGB", (width, height), "#101714")
        draw = ImageDraw.Draw(canvas)
        label = "HIGH PCK" if item["group"] == "top" else "LOW PCK"
        draw.text(
            (18, 10),
            f"{label} #{item['rank']}  |  L{item['block']:02d}/H{item['head']:02d}  |  S039",
            font=title_font, fill="#fffdf7",
        )
        draw.text(
            (18, 43),
            f"3-model 50-case Macro PCK@32 {item['macro_pck32']:.2f}%  |  mean error {item['macro_mean_error_px']:.2f}px",
            font=small_font, fill="#b9c5bf",
        )
        for model_index, (model, model_label, model_color) in enumerate(MODELS):
            panel = panel_source.copy()
            draw_tracks(
                panel, gt_tracks, gt_visibility, predictions[model], latent, color(model_color)
            )
            x = model_index * PANEL_WIDTH
            canvas.paste(panel, (x, HEADER))
            model_macro = item[f"{model}_macro_pck32"]
            case_pck = case_metrics[model][(item["block"], item["head"])]
            draw.rectangle((x, HEADER, x + PANEL_WIDTH, HEADER + 32), fill="#08100dcc")
            draw.text(
                (x + 10, HEADER + 5),
                f"{model_label} | Macro {model_macro:.2f}% | selected case {case_pck:.2f}%",
                font=label_font, fill=model_color,
            )
        draw.text(
            (18, HEADER + PANEL_HEIGHT + 9),
            f"Original video frame {source_index} | latent {latent} | white ring: pseudo-GT | colored point: Q@K prediction | exact anchors, no interpolation",
            font=small_font, fill="#d6ddd9",
        )
        frame = np.asarray(canvas, dtype=np.uint8)
        if first_frame is None:
            first_frame = canvas.copy()
        for _ in range(REPEAT):
            process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}")
    first_frame.save(poster_path, quality=91)
    return video_name, poster_name


def main() -> None:
    (OUTPUT_ROOT / "videos").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "posters").mkdir(parents=True, exist_ok=True)
    items = selected_combinations()
    source_frames = load_video_frames()
    gt_path = SOURCE_ROOT / "gt" / "cases" / CASE / "cotracker_pseudo_gt.npz"
    with np.load(gt_path) as payload:
        tracks = payload["tracks"][np.asarray(ANCHORS)].astype(np.float32)
        visibility = payload["visibility"][np.asarray(ANCHORS)].astype(bool)
    case_metrics = {model: load_case_metrics(model) for model, _, _ in MODELS}
    prediction_archives = {
        model: np.load(SOURCE_ROOT / model / "cases" / CASE / "predicted_tracks.npz")
        for model, _, _ in MODELS
    }
    try:
        for index, item in enumerate(items, 1):
            key = (
                f"qk_head{item['head']:02d}_layer{item['block']:02d}_"
                f"step{STEP:03d}_predictions"
            )
            predictions = {
                model: prediction_archives[model][key].astype(np.float32)
                for model, _, _ in MODELS
            }
            video, poster = render_video(
                item, source_frames, tracks, visibility, predictions, case_metrics
            )
            item["video"] = video
            item["poster"] = poster
            item["case_pck32"] = {
                model: case_metrics[model][(item["block"], item["head"])]
                for model, _, _ in MODELS
            }
            print(f"[{index:02d}/{len(items)}] {video}", flush=True)
    finally:
        for archive in prediction_archives.values():
            archive.close()
    manifest = {
        "case": CASE,
        "step": STEP,
        "scope": "objects",
        "anchors": list(ANCHORS),
        "pck_definition": "100 * mean(euclidean_pixel_error <= 32) over valid future-anchor comparisons",
        "items": items,
    }
    (OUTPUT_ROOT / "catalog.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    flat = []
    for item in items:
        row = {key: value for key, value in item.items() if key != "case_pck32"}
        for model, value in item["case_pck32"].items():
            row[f"{model}_case_pck32"] = value
        flat.append(row)
    with (OUTPUT_ROOT / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


if __name__ == "__main__":
    main()
