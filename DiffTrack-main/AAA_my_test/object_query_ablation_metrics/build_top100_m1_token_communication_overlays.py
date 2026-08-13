#!/usr/bin/env python3
"""Render audited Top100-M1 object-token locations on Baseline video frames."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np


ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
GUIDANCE_ROOT = ROOT / "training_free_top100_m1_guidance_v1"
RUNTIME_MANIFEST = ROOT / "stage4_runtime/stage4_manifest.json"
OUTPUT_ROOT = GUIDANCE_ROOT / "token_communication_overlays"
VARIANT = "single_object__object_A__m1_all_time__top100__pag0p5"
CASES = (
    "0613pybullet_sample_001460_w002",
    "0613pybullet_sample_000331_w001",
    "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end",
)
SEED = 47326
GRID_TIME, GRID_HEIGHT, GRID_WIDTH = 13, 22, 40
VIDEO_HEIGHT, VIDEO_WIDTH = 704, 1280
QUERY_BGR = (45, 108, 255)
KEY_BGR = (181, 168, 0)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open Baseline video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) != 49:
        raise RuntimeError(f"expected 49 Baseline frames, got {len(frames)}: {path}")
    if frames[0].shape[:2] != (VIDEO_HEIGHT, VIDEO_WIDTH):
        raise RuntimeError(f"unexpected Baseline frame shape {frames[0].shape}: {path}")
    return frames


def token_records(token_ids: list[int], latent_index: int) -> list[dict[str, Any]]:
    spatial_count = GRID_HEIGHT * GRID_WIDTH
    lower, upper = latent_index * spatial_count, (latent_index + 1) * spatial_count
    records = []
    for token_id in token_ids:
        token_id = int(token_id)
        if not lower <= token_id < upper:
            raise RuntimeError(
                f"token {token_id} is outside latent slab {latent_index}: [{lower}, {upper})"
            )
        spatial = token_id - lower
        grid_y, grid_x = divmod(spatial, GRID_WIDTH)
        x0 = round(grid_x * VIDEO_WIDTH / GRID_WIDTH)
        x1 = round((grid_x + 1) * VIDEO_WIDTH / GRID_WIDTH)
        y0 = round(grid_y * VIDEO_HEIGHT / GRID_HEIGHT)
        y1 = round((grid_y + 1) * VIDEO_HEIGHT / GRID_HEIGHT)
        records.append(
            {
                "token_index": token_id,
                "spatial_index": spatial,
                "grid_y": grid_y,
                "grid_x": grid_x,
                "pixel_bbox_xyxy": [x0, y0, x1, y1],
            }
        )
    return records


def translucent_panel(frame: np.ndarray, y1: int = 88) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], y1), (5, 18, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)


def draw_tokens(
    source: np.ndarray,
    tokens: list[dict[str, Any]],
    *,
    role: str,
    latent_frame: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    frame = source.copy()
    fill = frame.copy()
    for token in tokens:
        x0, y0, x1, y1 = token["pixel_bbox_xyxy"]
        cv2.rectangle(fill, (x0, y0), (x1 - 1, y1 - 1), color, -1)
    cv2.addWeighted(fill, 0.20, frame, 0.80, 0, frame)
    for token in tokens:
        x0, y0, x1, y1 = token["pixel_bbox_xyxy"]
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), color, 3)
        tag = f"{role[0]}{token['token_index']}"
        cv2.putText(frame, tag, (x0 + 2, min(y1 - 5, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, tag, (x0 + 2, min(y1 - 5, y0 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)
    translucent_panel(frame)
    title = f"{role} R_F{latent_frame:02d}  |  {len(tokens)} unique tokens  |  grid 22 x 40"
    detail = (
        "Query receiver cells; also same-frame K/V under M1-all-time"
        if role == "QUERY"
        else "K/V source cells removed from every R_tq Query under M1-all-time"
    )
    cv2.putText(frame, title, (18, 32), cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (226, 237, 241), 1, cv2.LINE_AA)
    return frame


def draw_dual(
    source: np.ndarray, tokens: list[dict[str, Any]], latent_frame: int
) -> np.ndarray:
    frame = source.copy()
    fill = frame.copy()
    for token in tokens:
        x0, y0, x1, y1 = token["pixel_bbox_xyxy"]
        cv2.rectangle(fill, (x0, y0), (x1 - 1, y1 - 1), KEY_BGR, -1)
    cv2.addWeighted(fill, 0.18, frame, 0.82, 0, frame)
    for token in tokens:
        x0, y0, x1, y1 = token["pixel_bbox_xyxy"]
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), QUERY_BGR, 4)
        cv2.rectangle(frame, (x0 + 4, y0 + 4), (x1 - 5, y1 - 5), KEY_BGR, 2)
    translucent_panel(frame)
    cv2.putText(
        frame,
        f"R_F{latent_frame:02d}: ORANGE Query + CYAN same-frame K/V",
        (18, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (240, 244, 246),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "All 13 R_tk frames are deleted sources for this Query; see communication matrix",
        (18, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (226, 237, 241),
        1,
        cv2.LINE_AA,
    )
    return frame


def build_case(case: str, runtime_sample: dict[str, Any], overwrite: bool) -> None:
    guidance_dir = GUIDANCE_ROOT / case / f"seed_{SEED:05d}" / VARIANT
    guidance_manifest_path = guidance_dir / "manifest.json"
    guidance = read_json(guidance_manifest_path)
    if guidance.get("m1_time_scope") != "all_time" or int(
        guidance.get("selected_head_count", -1)
    ) != 100:
        raise RuntimeError(f"not an audited Top100 M1-all-time run: {guidance_manifest_path}")
    token_ids_by_time = guidance["audit"]["query_token_indices_by_latent_frame"]
    if len(token_ids_by_time) != GRID_TIME:
        raise RuntimeError(f"expected {GRID_TIME} latent token groups, got {len(token_ids_by_time)}")
    track_path = Path(str(guidance["tracks_npz"]))
    with np.load(track_path) as arrays:
        anchor_frames = [int(value) for value in arrays["anchor_pixel_frames"].tolist()]
    expected_anchors = list(range(0, 49, 4))
    if anchor_frames != expected_anchors:
        raise RuntimeError(f"unexpected latent anchors: {anchor_frames}")

    baseline_path = Path(str(runtime_sample["baseline_video"]))
    generated_path = Path(str(guidance["output_video"]))
    if not baseline_path.is_file() or not generated_path.is_file():
        raise FileNotFoundError(f"comparison video missing for {case}")
    output = OUTPUT_ROOT / case / f"seed_{SEED:05d}"
    complete_path = output / "complete.json"
    if complete_path.is_file() and not overwrite:
        print(f"skip {case}")
        return
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    frames = read_video(baseline_path)
    anchor_rows = []
    dual_frames = []
    for latent_index, (pixel_frame, token_ids) in enumerate(
        zip(anchor_frames, token_ids_by_time, strict=True)
    ):
        tokens = token_records(token_ids, latent_index)
        query_name = f"query_F{pixel_frame:02d}.jpg"
        key_name = f"key_F{pixel_frame:02d}.jpg"
        query_frame = draw_tokens(
            frames[pixel_frame],
            tokens,
            role="QUERY",
            latent_frame=pixel_frame,
            color=QUERY_BGR,
        )
        key_frame = draw_tokens(
            frames[pixel_frame],
            tokens,
            role="KEY/VALUE",
            latent_frame=pixel_frame,
            color=KEY_BGR,
        )
        cv2.imwrite(str(output / query_name), query_frame, [cv2.IMWRITE_JPEG_QUALITY, 91])
        cv2.imwrite(str(output / key_name), key_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        dual_frames.append(
            cv2.cvtColor(
                draw_dual(frames[pixel_frame], tokens, pixel_frame),
                cv2.COLOR_BGR2RGB,
            )
        )
        anchor_rows.append(
            {
                "latent_index": latent_index,
                "pixel_frame": pixel_frame,
                "token_count": len(tokens),
                "tokens": tokens,
                "query_image": query_name,
                "key_image": key_name,
            }
        )
    overlay_video = output / "anchor_token_overlay.mp4"
    iio.imwrite(overlay_video, np.stack(dual_frames), fps=2, codec="libx264")
    metadata = {
        "protocol": "top100_m1_token_communication_overlay_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case": case,
        "seed": SEED,
        "region": "object_A",
        "head_scope": "latest3350_top100",
        "selected_head_count": 100,
        "m1_time_scope": "all_time",
        "denoising_steps": list(range(40)),
        "baseline_video": str(baseline_path),
        "guidance_video": str(generated_path),
        "guidance_manifest": str(guidance_manifest_path),
        "tracks_npz": str(track_path),
        "overlay_video": str(overlay_video),
        "grid": {
            "time": GRID_TIME,
            "height": GRID_HEIGHT,
            "width": GRID_WIDTH,
            "video_height": VIDEO_HEIGHT,
            "video_width": VIDEO_WIDTH,
            "nominal_cell_height_px": VIDEO_HEIGHT / GRID_HEIGHT,
            "nominal_cell_width_px": VIDEO_WIDTH / GRID_WIDTH,
        },
        "communication": {
            "query_frames": anchor_frames,
            "key_value_frames": anchor_frames,
            "active_pairs": GRID_TIME * GRID_TIME,
            "definition": "For every tq and selected head: Y[R_tq] -= sum_{tk=0..12} A[R_tq,R_tk]V[R_tk]",
            "post_softmax_renormalization": False,
        },
        "anchors": anchor_rows,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_path.write_text(
        json.dumps(
            {
                "protocol": metadata["protocol"],
                "case": case,
                "anchor_count": len(anchor_rows),
                "active_pairs": GRID_TIME * GRID_TIME,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    runtime = read_json(RUNTIME_MANIFEST)
    samples = {
        (str(row["case"]), int(row["seed"])): row for row in runtime["samples"]
    }
    for case in args.case or CASES:
        build_case(case, samples[(case, SEED)], args.overwrite)


if __name__ == "__main__":
    main()
