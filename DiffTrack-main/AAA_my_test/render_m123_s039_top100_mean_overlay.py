#!/usr/bin/env python3
"""Render fixed-F04 Object Query Top100 mean maps for one M1/M2/M3 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from AAA_my_test.render_object_query_top100_mean_overlay import (
    FRAMES,
    render,
    video_frames,
)


KINDS = (
    ("before", "Pre-mask Top100 mean"),
    ("effective_after", "Effective post-mask Top100 mean (no renorm)"),
    ("removed", "Removed Top100 coefficient mass"),
)


def _scalar(payload: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(payload[key])
    return value.item() if value.ndim == 0 else value


def _frame_scale(*values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            max(
                float(
                    np.percentile(
                        np.concatenate([value[index].ravel() for value in values]),
                        99.5,
                    )
                ),
                1e-12,
            )
            for index in range(FRAMES)
        ],
        dtype=np.float32,
    )


def render_variant(capture_dir: Path, video_path: Path) -> dict:
    """Render all object records and return the lightweight page manifest."""
    capture_dir = Path(capture_dir)
    frames = video_frames(Path(video_path))
    records = []
    for npz_path in sorted(capture_dir.glob("object_*.npz")):
        with np.load(npz_path, allow_pickle=False) as payload:
            region = str(_scalar(payload, "region_name"))
            before = np.asarray(payload["before"], dtype=np.float32)
            after = np.asarray(payload["effective_after"], dtype=np.float32)
            removed = np.asarray(payload["removed"], dtype=np.float32)
            if before.shape != (FRAMES, 16, 28) or after.shape != before.shape:
                raise RuntimeError(
                    f"{npz_path}: expected {(FRAMES, 16, 28)}, got {before.shape}/{after.shape}"
                )
            query_payload = {
                key: np.asarray(payload[key]).copy()
                for key in ("query_context_frame", "query_mask", "query_points")
            }
            experiment_id = str(_scalar(payload, "experiment_id"))
            operator_id = str(_scalar(payload, "operator_id"))
            temporal_scope = str(_scalar(payload, "temporal_scope"))
            head_scope = str(_scalar(payload, "head_scope"))
            seed = int(_scalar(payload, "seed"))
            step = int(_scalar(payload, "step"))
            phrase = str(_scalar(payload, "region_phrase"))
            local_heads = int(_scalar(payload, "locally_ablated_top100_heads"))
            query_rows = int(_scalar(payload, "locally_ablated_query_rows"))

        shared_scale = _frame_scale(before, after)
        removed_scale = _frame_scale(removed)
        images = {}
        rows = []
        for kind, title_kind in KINDS:
            values = {"before": before, "effective_after": after, "removed": removed}[kind]
            vmax = removed_scale if kind == "removed" else shared_scale
            filename = f"{region}__{kind}__s039_top100_mean.jpg"
            title = (
                f"{experiment_id} | seed {seed} | S{step:03d} | {region} | "
                f"{title_kind} | Fixed F04 Q | {head_scope} intervention"
            )
            image = render(
                query_payload,
                frames,
                values,
                vmax,
                title,
                "FIXED F04 · SUM 8Q / MEAN 100H",
            )
            if not cv2.imwrite(
                str(capture_dir / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 94]
            ):
                raise RuntimeError(f"failed to write {capture_dir / filename}")
            images[kind] = filename
            rows.append(image)
        comparison_name = f"{region}__s039_top100_mean_comparison.jpg"
        comparison = cv2.vconcat(rows)
        if not cv2.imwrite(
            str(capture_dir / comparison_name),
            comparison,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        ):
            raise RuntimeError(f"failed to write {capture_dir / comparison_name}")
        images["comparison"] = comparison_name
        records.append(
            {
                "region_name": region,
                "region_phrase": phrase,
                "experiment_id": experiment_id,
                "operator_id": operator_id,
                "temporal_scope": temporal_scope,
                "head_scope": head_scope,
                "step": step,
                "query_pixel_frame": 4,
                "query_latent_frame": 1,
                "query_aggregation": "sum_8_fixed_f04_queries_then_mean_100_heads_and_2_cfg_calls",
                "effective_after_definition": (
                    "the captured pre-mask softmax coefficients with the exact M1/M2/M3 "
                    "entries set to zero on intervention-selected Top100 heads; no renormalization"
                ),
                "locally_ablated_top100_heads": local_heads,
                "locally_ablated_query_rows": query_rows,
                "before_after_frame_vmax": shared_scale.tolist(),
                "removed_frame_vmax": removed_scale.tolist(),
                "images": images,
            }
        )
    if len(records) != 2:
        raise RuntimeError(f"expected Object A/B captures in {capture_dir}, got {len(records)}")
    manifest_path = capture_dir / "overlay_manifest.json"
    manifest = {
        "video": str(video_path),
        "records": records,
    }
    manifest_path.write_text(
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
    render_variant(cfg.capture_dir, cfg.video)


if __name__ == "__main__":
    main()
