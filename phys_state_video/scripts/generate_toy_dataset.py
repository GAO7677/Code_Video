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

from phys_state_video.schemas import StateIndex


def _draw_box(frame: np.ndarray, box: np.ndarray, color: np.ndarray) -> None:
    height, width = frame.shape[-2:]
    x0 = max(int(box[0] * width), 0)
    y0 = max(int(box[1] * height), 0)
    x1 = min(int(box[2] * width), width)
    y1 = min(int(box[3] * height), height)
    frame[:, y0:y1, x0:x1] = color[:, None, None]


def _make_episode(seed: int, context_steps: int, future_steps: int, num_objects: int, height: int, width: int):
    rng = np.random.default_rng(seed)
    total_steps = context_steps + future_steps
    frames = np.zeros((total_steps, 3, height, width), dtype=np.float32)
    states = np.zeros((total_steps, num_objects, 10), dtype=np.float32)
    boxes = np.zeros((total_steps, num_objects, 4), dtype=np.float32)
    appearance = np.zeros((num_objects, 64), dtype=np.float32)
    camera = np.zeros((context_steps, 8), dtype=np.float32)

    colors = np.asarray(
        [
            [1.0, 0.2, 0.2],
            [0.2, 1.0, 0.2],
            [0.2, 0.4, 1.0],
        ],
        dtype=np.float32,
    )
    colors = colors[:num_objects]

    centers = rng.uniform(0.2, 0.8, size=(num_objects, 2)).astype(np.float32)
    velocities = rng.uniform(-0.03, 0.03, size=(num_objects, 2)).astype(np.float32)
    sizes = rng.uniform(0.08, 0.16, size=(num_objects, 2)).astype(np.float32)
    depths = rng.uniform(0.8, 1.2, size=(num_objects,)).astype(np.float32)

    for obj_idx in range(num_objects):
        appearance[obj_idx, obj_idx] = 1.0
        appearance[obj_idx, 8:11] = colors[obj_idx]

    for step in range(total_steps):
        frame = np.zeros((3, height, width), dtype=np.float32)
        frame += 0.05
        for obj_idx in range(num_objects):
            if step > 0:
                centers[obj_idx] += velocities[obj_idx]
                for axis in range(2):
                    if centers[obj_idx, axis] < 0.1 or centers[obj_idx, axis] > 0.9:
                        velocities[obj_idx, axis] *= -1.0
                        centers[obj_idx, axis] = np.clip(centers[obj_idx, axis], 0.1, 0.9)

            width_box, height_box = sizes[obj_idx]
            x0 = np.clip(centers[obj_idx, 0] - width_box * 0.5, 0.0, 1.0)
            y0 = np.clip(centers[obj_idx, 1] - height_box * 0.5, 0.0, 1.0)
            x1 = np.clip(centers[obj_idx, 0] + width_box * 0.5, 0.0, 1.0)
            y1 = np.clip(centers[obj_idx, 1] + height_box * 0.5, 0.0, 1.0)
            box = np.asarray([x0, y0, x1, y1], dtype=np.float32)
            boxes[step, obj_idx] = box
            states[step, obj_idx, StateIndex.CENTER_X:StateIndex.CENTER_Y + 1] = centers[obj_idx]
            states[step, obj_idx, StateIndex.DEPTH] = depths[obj_idx]
            states[step, obj_idx, StateIndex.LOG_SCALE] = np.log(max((x1 - x0) * (y1 - y0), 1e-6))
            states[step, obj_idx, StateIndex.VEL_X:StateIndex.VEL_Y + 1] = velocities[obj_idx]
            states[step, obj_idx, StateIndex.DEPTH_VEL] = 0.0
            states[step, obj_idx, StateIndex.VISIBILITY] = 1.0
            states[step, obj_idx, StateIndex.EXISTENCE] = 1.0
            states[step, obj_idx, StateIndex.CONFIDENCE] = 1.0
            _draw_box(frame, box, colors[obj_idx])
        frames[step] = frame

    prompt = "colored squares move smoothly in a bounded scene"
    return {
        "context_frames": frames[:context_steps],
        "future_frames": frames[context_steps:],
        "context_states": states[:context_steps],
        "future_states": states[context_steps:],
        "context_boxes": boxes[:context_steps],
        "future_boxes": boxes[context_steps:],
        "appearance": appearance,
        "camera": camera,
        "prompt": prompt,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a toy dataset for phys-state-video.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--context-steps", type=int, default=4)
    parser.add_argument("--future-steps", type=int, default=6)
    parser.add_argument("--objects", type=int, default=2)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.episodes):
        episode = _make_episode(index, args.context_steps, args.future_steps, args.objects, args.height, args.width)
        np.savez_compressed(
            output_dir / f"episode_{index:04d}.npz",
            context_frames=episode["context_frames"],
            future_frames=episode["future_frames"],
            context_states=episode["context_states"],
            future_states=episode["future_states"],
            context_boxes=episode["context_boxes"],
            future_boxes=episode["future_boxes"],
            appearance=episode["appearance"],
            camera=episode["camera"],
        )
        (output_dir / f"episode_{index:04d}.json").write_text(json.dumps({"prompt": episode["prompt"]}))
    print(f"generated {args.episodes} toy episodes under {output_dir}")


if __name__ == "__main__":
    main()
