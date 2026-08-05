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


def scalar(data, key):
    return np.asarray(data[key]).item()


def frames(path):
    cap = cv2.VideoCapture(str(path)); result = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        result.append(frame)
    cap.release()
    if len(result) < 49: raise RuntimeError(f"Expected 49 frames: {path}")
    return [result[index] for index in range(0, 49, 4)]


def strip(video_frames, maps, title, vmax):
    tiles = []
    for index, values in enumerate(maps):
        base = cv2.resize(video_frames[index], (320, 183), interpolation=cv2.INTER_AREA)
        heat = cv2.resize(values.astype(np.float32), (320, 183))
        norm = np.clip(heat / max(float(vmax[index]), 1e-12), 0, 1)
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        tile = cv2.addWeighted(base, .46, color, .54, 0)
        cv2.rectangle(tile, (0, 0), (320, 28), (244, 240, 230), -1)
        cv2.putText(tile, f"K{index:02d}/F{index*4:02d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .54, (25, 31, 29), 1, cv2.LINE_AA)
        tiles.append(tile)
    body = np.concatenate(tiles, axis=1)
    header = np.full((42, body.shape[1], 3), (237, 232, 219), np.uint8)
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .70, (22, 38, 31), 2, cv2.LINE_AA)
    return np.concatenate([header, body], axis=0)


def main():
    args = parse_args(); args.output_root.mkdir(parents=True, exist_ok=True)
    video_frames = frames(args.video); records = []
    for path in sorted(args.capture_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            before = np.asarray(data["before"], np.float32)
            after = np.asarray(data["after"], np.float32)
            removed = np.asarray(data["removed"], np.float32)
            region = str(scalar(data, "region_name")); phrase = str(scalar(data, "region_phrase"))
            branch = str(scalar(data, "cfg_branch")); step = int(scalar(data, "step")); seed = int(scalar(data, "seed"))
            num_heads = int(scalar(data, "num_heads"))
        shared = np.maximum(before.reshape(13, -1).max(1), after.reshape(13, -1).max(1)).clip(min=1e-12)
        removed_max = removed.reshape(13, -1).max(1).clip(min=1e-12)
        stem = f"seed{seed:06d}__step{step:02d}__{branch}__{region}"
        images = {key: f"{stem}__{key}.jpg" for key in ("before", "after", "removed")}
        payloads = {
            "before": strip(video_frames, before, "Frozen-mask Apply Before · per-frame shared scale", shared),
            "after": strip(video_frames, after, "Frozen-mask Apply After · per-frame shared scale", shared),
            "removed": strip(video_frames, removed, "Actual Removed Attention Mass · per-frame scale", removed_max),
        }
        for key, image in payloads.items(): cv2.imwrite(str(args.output_root / images[key]), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({"seed": seed, "step": step, "cfg_branch": branch, "region_name": region, "region_phrase": phrase, "num_heads": num_heads, "images": images})
    (args.output_root / "manifest.json").write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
