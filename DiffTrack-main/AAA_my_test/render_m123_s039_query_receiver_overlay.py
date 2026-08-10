#!/usr/bin/env python3
"""Render S039 query-side receiver maps for one M1/M2/M3 replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from AAA_my_test.render_object_query_top100_mean_overlay import (
    FRAMES,
    LABEL_W,
    TILE_H,
    TILE_W,
    video_frames,
)


HEADER_H = 68
KINDS = (
    ("coefficient_mass", "S(q) = sum selected-source attention mass"),
    ("value_contribution_norm", "E(q) = per-head ||sum A(q,k)V(k)||2"),
)


def _scalar(payload: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(payload[key])
    return value.item() if value.ndim == 0 else value


def _global_scale(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float32)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    return max(float(np.percentile(positive, 99.5)), 1e-12) if positive.size else 1.0


def _label_tile(lines: list[str]) -> np.ndarray:
    tile = np.full((TILE_H, LABEL_W, 3), (31, 38, 35), dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(
            tile,
            line,
            (7, 18 + index * 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (245, 248, 246),
            1,
        )
    return tile


def _render_strip(
    frames: list[np.ndarray],
    values: np.ndarray,
    r_tube_mask: np.ndarray,
    vmax: float,
    title: str,
    label_lines: list[str],
) -> np.ndarray:
    canvas = np.full(
        (HEADER_H + TILE_H, LABEL_W + FRAMES * TILE_W, 3), 244, np.uint8
    )
    cv2.putText(
        canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (27, 38, 33), 1
    )
    cv2.line(canvas, (0, 35), (canvas.shape[1] - 1, 35), (218, 222, 219), 1)
    canvas[HEADER_H:, :LABEL_W] = _label_tile(label_lines)
    for frame_index in range(FRAMES):
        x = LABEL_W + frame_index * TILE_W
        cv2.putText(
            canvas,
            f"Q{frame_index:02d}/F{frame_index * 4:02d}",
            (x + 43, 57),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (27, 38, 33),
            1,
        )
        base = cv2.resize(
            frames[frame_index], (TILE_W, TILE_H), interpolation=cv2.INTER_AREA
        )
        heat = cv2.resize(
            values[frame_index].astype(np.float32),
            (TILE_W, TILE_H),
            interpolation=cv2.INTER_LINEAR,
        )
        norm = np.clip(heat / max(vmax, 1e-12), 0, 1)
        color = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO)
        alpha = (0.08 + 0.78 * norm)[..., None]
        rendered = np.uint8(np.clip(base * (1 - alpha) + color * alpha, 0, 255))
        sender = cv2.resize(
            r_tube_mask[frame_index].astype(np.uint8),
            (TILE_W, TILE_H),
            interpolation=cv2.INTER_NEAREST,
        )
        contours, _ = cv2.findContours(sender, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rendered, contours, -1, (40, 235, 245), 1)
        canvas[HEADER_H:, x : x + TILE_W] = rendered
    return canvas


def render_receiver(capture_dir: Path, video_path: Path) -> dict:
    capture_dir = Path(capture_dir)
    with np.load(capture_dir / "receiver.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name], dtype=np.float32) for name, _ in KINDS}
        r_tube_mask = np.asarray(payload["r_tube_mask"], dtype=bool)
        metadata = {
            key: _scalar(payload, key)
            for key in (
                "experiment_id",
                "operator_id",
                "temporal_scope",
                "head_scope",
                "intervention_head_count",
                "target_partition",
                "source_partition",
                "time_predicate",
                "step",
                "seed",
            )
        }
    expected_shape = (FRAMES, 22, 40)
    if any(value.shape != expected_shape for value in arrays.values()):
        raise RuntimeError(f"receiver maps must have shape {expected_shape}")
    if r_tube_mask.shape != expected_shape:
        raise RuntimeError(f"R tube mask must have shape {expected_shape}")

    frames = video_frames(Path(video_path))
    rows = []
    images = {}
    scales = {}
    for name, definition in KINDS:
        values = arrays[name]
        vmax = _global_scale(values)
        scales[name] = vmax
        filename = f"receiver__{name}__s039.jpg"
        title = (
            f"{metadata['experiment_id']} | seed {metadata['seed']} | S{int(metadata['step']):03d} | "
            f"Query-side {definition} | global P99.5={vmax:.6g}"
        )
        image = _render_strip(
            frames,
            values,
            r_tube_mask,
            vmax,
            title,
            [
                f"{metadata['operator_id']} {metadata['temporal_scope']}",
                f"Q={metadata['target_partition']}",
                f"K/V={metadata['source_partition']}",
                f"{metadata['head_scope']}",
            ],
        )
        if not cv2.imwrite(
            str(capture_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 94]
        ):
            raise RuntimeError(f"failed to write {capture_dir / filename}")
        images[name] = filename
        rows.append(image)

    comparison_name = "receiver__s039_query_side_comparison.jpg"
    if not cv2.imwrite(
        str(capture_dir / comparison_name),
        cv2.vconcat(rows),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    ):
        raise RuntimeError(f"failed to write {capture_dir / comparison_name}")
    images["comparison"] = comparison_name
    manifest = {
        **metadata,
        "query_frames": list(range(0, 49, 4)),
        "coefficient_definition": "mean_heads_cfg sum_selected_source_keys A[q,k]",
        "value_definition": "mean_heads_cfg norm2(sum_selected_source_keys A[q,k]V[k])",
        "scale_mode": "one global P99.5 per row and experiment",
        "r_tube_outline": "cyan",
        "global_vmax": scales,
        "images": images,
    }
    (capture_dir / "overlay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    cfg = parse_args()
    render_receiver(cfg.capture_dir, cfg.video)


if __name__ == "__main__":
    main()
