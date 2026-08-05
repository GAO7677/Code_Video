#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from AAA_my_test.build_object_query_frozen_trajectory_masks import (
    FRAMES,
    read_frames,
    scalar,
    strip,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()

    args.render_root.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.video)
    records = []
    for path in sorted(args.probe_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            seed = int(scalar(data, "seed"))
            step = int(scalar(data, "step"))
            branch = str(scalar(data, "cfg_branch"))
            region = str(scalar(data, "region_name"))
            phrase = str(scalar(data, "region_phrase"))
            num_heads = int(scalar(data, "num_heads"))
        frame_max = mean.reshape(FRAMES, -1).max(axis=1).clip(min=1e-12)
        normalized = mean / frame_max[:, None, None]
        stem = f"seed{seed:06d}__step{step:02d}__{branch}__{region}"
        image_name = f"{stem}__mean.jpg"
        image = strip(
            frames,
            normalized,
            f"S{step:03d} {branch} · Common No Intervention Top{num_heads} Mean · per-frame scale",
        )
        cv2.imwrite(
            str(args.render_root / image_name),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
        records.append(
            {
                "seed": seed,
                "step": step,
                "cfg_branch": branch,
                "region_name": region,
                "region_phrase": phrase,
                "num_heads": num_heads,
                "images": {"mean": image_name},
            }
        )
    (args.render_root / "manifest.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
