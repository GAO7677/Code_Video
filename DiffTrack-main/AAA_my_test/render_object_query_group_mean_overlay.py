#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def scalar(value):
    item = np.asarray(value)
    return item.item() if item.ndim == 0 else item.tolist()


def read_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def heat_overlay(frame, values, vmax, mask=False, color=(35, 55, 230)):
    height, width = frame.shape[:2]
    resized = cv2.resize(values.astype(np.float32), (width, height), cv2.INTER_NEAREST)
    if mask:
        active = resized > 0.5
        layer = np.empty_like(frame)
        layer[:] = color
        output = frame.copy()
        output[active] = cv2.addWeighted(frame, 0.30, layer, 0.70, 0)[active]
        return output
    normalized = np.clip(resized / max(float(vmax), 1e-12), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.48, colored, 0.52, 0)


def strip(frames, maps, title, vmax, mask=False, color=(35, 55, 230)):
    tiles = []
    for latent_index, values in enumerate(maps):
        pixel_index = min(latent_index * 4, len(frames) - 1)
        frame = frames[pixel_index]
        frame_vmax = float(vmax[latent_index]) if np.ndim(vmax) else float(vmax)
        tile = heat_overlay(frame, values, frame_vmax, mask=mask, color=color)
        tile = cv2.resize(tile, (320, 183), cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (320, 28), (244, 240, 230), -1)
        cv2.putText(
            tile,
            f"K{latent_index:02d}/F{pixel_index:02d}",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (25, 31, 29),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    canvas = np.concatenate(tiles, axis=1)
    header = np.full((42, canvas.shape[1], 3), (237, 232, 219), np.uint8)
    cv2.putText(
        header,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (22, 38, 31),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, canvas], axis=0)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.video)
    records = []
    for path in sorted(args.capture_root.glob("*__group_mean.npz")):
        with np.load(path, allow_pickle=False) as data:
            before = data["before"].astype(np.float32)
            after = data["after"].astype(np.float32)
            removed = data["removed"].astype(np.float32)
            p90 = data["p90"].astype(np.float32)
            main_component = data["main_component"].astype(np.float32)
            forbidden = data["forbidden"].astype(np.float32)
            region_name = str(scalar(data["region_name"]))
            region_phrase = str(scalar(data["region_phrase"]))
            num_heads = int(scalar(data["num_heads"]))
            step = int(scalar(data["step"]))
            seed = int(scalar(data["seed"]))
        shared_frame_vmax = np.maximum(
            before.reshape(before.shape[0], -1).max(axis=1),
            after.reshape(after.shape[0], -1).max(axis=1),
        ).clip(min=1e-12)
        removed_frame_vmax = removed.reshape(removed.shape[0], -1).max(axis=1).clip(min=1e-12)
        stem = f"seed{seed:06d}__{region_name}__step{step:02d}"
        images = {
            "before": f"{stem}__before.jpg",
            "p90": f"{stem}__p90.jpg",
            "main_component": f"{stem}__main_component.jpg",
            "forbidden": f"{stem}__forbidden.jpg",
            "after": f"{stem}__after.jpg",
            "removed": f"{stem}__removed.jpg",
        }
        payloads = {
            "before": strip(frames, before, "Group Before · per-frame scale · SUM 8 queries / MEAN Top100 heads", shared_frame_vmax),
            "p90": strip(frames, p90, "Current candidate · Group Mean P90", 1.0, True, (40, 155, 225)),
            "main_component": strip(frames, main_component, "Previous anchor source · Top-5 Main Connected Component", 1.0, True, (65, 185, 90)),
            "forbidden": strip(frames, forbidden, "Forbidden · P90 outside previous component neighborhood", 1.0, True, (45, 40, 235)),
            "after": strip(frames, after, "Group After · per-frame shared Before/After scale", shared_frame_vmax),
            "removed": strip(frames, removed, "Removed Attention Mass · per-frame scale", removed_frame_vmax),
        }
        for key, image in payloads.items():
            if not cv2.imwrite(str(args.output_root / images[key]), image, [cv2.IMWRITE_JPEG_QUALITY, 91]):
                raise RuntimeError(f"Failed to write {images[key]}")
        records.append(
            {
                "region_name": region_name,
                "region_phrase": region_phrase,
                "num_heads": num_heads,
                "step": step,
                "seed": seed,
                "scale_mode": "per_frame_before_after_shared",
                "frame_vmax": shared_frame_vmax.tolist(),
                "removed_frame_vmax": removed_frame_vmax.tolist(),
                "images": images,
            }
        )
    (args.output_root / "manifest.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
