#!/usr/bin/env python3
"""Render dynamic Q_t -> K_t Top100 mean attention on latent anchor frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from AAA_my_test.build_object_query_frozen_trajectory_masks import FRAMES, read_frames, scalar, strip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    args.render_root.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.video)
    records = []
    for path in sorted(args.capture_root.glob("step_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            means = data["mean"].astype(np.float32)
            counts = data["head_counts"].astype(np.int32)
            token_counts = data["query_token_counts"].astype(np.int32)
            names = data["region_names"].astype(str)
            phrases = data["region_phrases"].astype(str)
            seed = int(scalar(data, "seed"))
            step = int(scalar(data, "step"))
            branch = str(scalar(data, "cfg_branch"))
            inference_steps = int(scalar(data, "inference_steps"))
        for index, (name, phrase) in enumerate(zip(names, phrases)):
            mean = means[index]
            frame_max = mean.reshape(FRAMES, -1).max(axis=1)
            normalized = np.divide(
                mean,
                np.maximum(frame_max, 1e-12)[:, None, None],
                out=np.zeros_like(mean),
                where=frame_max[:, None, None] > 0,
            )
            image = strip(
                frames,
                normalized,
                f"{inference_steps}-Step S{step:03d} {branch} | Dynamic Object Q_t -> same-frame K_t | Top100 Mean | per-frame scale",
            )
            image_name = (
                f"seed{seed:06d}__step{step:02d}__{branch}__{name}"
                "__dynamic_same_frame_mean.jpg"
            )
            if not cv2.imwrite(
                str(args.render_root / image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 92]
            ):
                raise RuntimeError(f"Failed to write {args.render_root / image_name}")
            records.append(
                {
                    "seed": seed,
                    "step": step,
                    "cfg_branch": branch,
                    "inference_steps": inference_steps,
                    "region_name": str(name),
                    "region_phrase": str(phrase),
                    "num_heads": int(counts[index].max(initial=0)),
                    "query_token_counts": token_counts[index].tolist(),
                    "image": image_name,
                    "scale_mode": "per_latent_frame",
                }
            )
    (args.render_root / "manifest.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
