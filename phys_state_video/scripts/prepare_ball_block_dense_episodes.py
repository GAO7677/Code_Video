#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.proxy_state import read_video_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dense-prefix ball-block training episodes.")
    parser.add_argument(
        "--input-root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block",
        help="Root directory containing ball_block scenario json/mp4 pairs.",
    )
    parser.add_argument(
        "--output-root",
        default="/data/gaoya/AAA_test_video/Dataset_physV/0526dp/ball_block_dense_episodes",
        help="Output root for episode npz/json files.",
    )
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--future-steps", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(input_root.glob("*.json"))
    if args.limit is not None:
        meta_files = meta_files[: int(args.limit)]

    records: list[dict[str, object]] = []
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        video_path = Path(meta["video"])
        states_path = Path(meta.get("states", meta_path.with_name(meta_path.stem + "_states.npz")))
        if not video_path.exists() or not states_path.exists():
            print(f"[skip] missing video or states for {meta_path.name}")
            continue

        frames = read_video_frames(video_path, resize_height=args.height, resize_width=args.width)
        frames = frames[:: args.frame_stride]
        if frames.shape[0] < 2:
            print(f"[skip] too short: {video_path}")
            continue
        total_frames = min(int(args.num_frames), int(frames.shape[0]))
        if args.future_steps is not None:
            total_frames = min(total_frames, int(args.num_context_frames) + int(args.future_steps))
        if total_frames <= args.num_context_frames:
            print(f"[skip] not enough frames: {video_path}")
            continue

        frames = frames[:total_frames]
        states = np.load(states_path, allow_pickle=False)
        positions = states["positions"][:total_frames]
        quats = states["quats"][:total_frames]
        linvels = states["linear_velocities"][:total_frames]
        angvels = states["angular_velocities"][:total_frames]
        frame_times = states["frame_times"][:total_frames]

        context_frames = frames[: args.num_context_frames].astype(np.float32)
        future_frames = frames[args.num_context_frames :].astype(np.float32)
        num_objects = int(positions.shape[1])
        context_states = np.zeros((args.num_context_frames, num_objects, 10), dtype=np.float32)
        future_states = np.zeros((future_frames.shape[0], num_objects, 10), dtype=np.float32)
        context_boxes = np.zeros((args.num_context_frames, num_objects, 4), dtype=np.float32)
        future_boxes = np.zeros((future_frames.shape[0], num_objects, 4), dtype=np.float32)

        episode_name = meta_path.stem
        episode_dir = output_root
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_npz = episode_dir / f"{episode_name}.npz"
        episode_json = episode_dir / f"{episode_name}.json"
        if episode_npz.exists() and not args.overwrite:
            print(f"[skip] exists: {episode_npz.name}")
            continue

        np.savez_compressed(
            episode_npz,
            context_frames=context_frames,
            future_frames=future_frames,
            context_states=context_states,
            future_states=future_states,
            context_boxes=context_boxes,
            future_boxes=future_boxes,
            appearance=np.zeros((2, 16), dtype=np.float32),
            camera=np.repeat(np.zeros((1, 8), dtype=np.float32), args.num_context_frames, axis=0),
        )
        episode_json.write_text(
            json.dumps(
                {
                    "prompt": meta.get("caption", "ball block dense episode"),
                    "source_video": str(video_path),
                    "source_states": str(states_path),
                    "num_frames": int(total_frames),
                    "num_context_frames": int(args.num_context_frames),
                    "frame_stride": int(args.frame_stride),
                    "frame_times": frame_times.tolist(),
                    "sampling_mode": "prefix",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        records.append(
            {
                "episode": str(episode_npz),
                "source_video": str(video_path),
                "num_frames": int(total_frames),
                "num_context_frames": int(args.num_context_frames),
            }
        )
        print(f"[ok] {episode_npz.name} <- {video_path.name}")

    (output_root / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved manifest to {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
