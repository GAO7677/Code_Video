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

from phys_state_video.proxy_state import extract_primary_track, read_video_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert benchmark samples into phys-state-video episode files."
    )
    parser.add_argument("--input-root",
                        required=True,
                        help="Directory containing per-sample folders.")
    parser.add_argument("--output-root",
                        required=True,
                        help="Output directory for train/val episode files.")
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--context-steps",
                        type=int,
                        default=None,
                        help="Optional cap on context frames after sampling.")
    parser.add_argument("--future-steps",
                        type=int,
                        default=None,
                        help="Optional cap on future frames after sampling.")
    parser.add_argument("--frame-stride",
                        type=int,
                        default=1,
                        help="Temporal stride applied before truncation.")
    parser.add_argument("--limit",
                        type=int,
                        default=None,
                        help="Optional cap on number of samples.")
    return parser.parse_args()


def load_prompt(meta_path: Path) -> str:
    if not meta_path.exists():
        return ""
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload.get("caption", "")


def choose_split(index: int, total: int, val_ratio: float) -> str:
    val_count = int(round(total * val_ratio))
    if val_count <= 0:
        return "train"
    return "val" if index >= total - val_count else "train"


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    sample_dirs = sorted([path for path in input_root.iterdir() if path.is_dir()])
    if args.limit is not None:
        sample_dirs = sample_dirs[:args.limit]
    if not sample_dirs:
        raise FileNotFoundError(f"no sample directories found under {input_root}")

    manifest: dict[str, object] = {
        "input_root": str(input_root),
        "height": args.height,
        "width": args.width,
        "samples": [],
    }

    for index, sample_dir in enumerate(sample_dirs):
        context_path = sample_dir / "context_video.mp4"
        future_path = sample_dir / "future_gt_video.mp4"
        meta_path = sample_dir / "meta.json"
        if not context_path.exists() or not future_path.exists():
            continue

        context_frames = read_video_frames(context_path,
                                           resize_height=args.height,
                                           resize_width=args.width)
        future_frames = read_video_frames(future_path,
                                          resize_height=args.height,
                                          resize_width=args.width)
        context_frames = context_frames[::args.frame_stride]
        future_frames = future_frames[::args.frame_stride]
        if args.context_steps is not None:
            context_frames = context_frames[-args.context_steps:]
        if args.future_steps is not None:
            future_frames = future_frames[:args.future_steps]
        all_frames = np.concatenate([context_frames, future_frames], axis=0)
        track = extract_primary_track(all_frames)
        context_steps = context_frames.shape[0]

        split = choose_split(index, len(sample_dirs), args.val_ratio)
        sample_name = sample_dir.name
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        episode_path = split_dir / f"{sample_name}.npz"

        np.savez_compressed(
            episode_path,
            context_frames=context_frames.astype(np.float32),
            future_frames=future_frames.astype(np.float32),
            context_states=track.states[:context_steps].astype(np.float32),
            future_states=track.states[context_steps:].astype(np.float32),
            context_boxes=track.boxes[:context_steps].astype(np.float32),
            future_boxes=track.boxes[context_steps:].astype(np.float32),
            full_frames=all_frames.astype(np.float32),
            full_states=track.states.astype(np.float32),
            full_boxes=track.boxes.astype(np.float32),
            appearance=track.appearance.astype(np.float32),
            camera=np.zeros((context_steps, 8), dtype=np.float32),
            camera_full=np.zeros((all_frames.shape[0], 8), dtype=np.float32),
        )
        prompt = load_prompt(meta_path)
        episode_path.with_suffix(".json").write_text(
            json.dumps({"prompt": prompt}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["samples"].append({
            "sample": sample_name,
            "split": split,
            "context_frames": int(context_steps),
            "future_frames": int(future_frames.shape[0]),
            "visible_fraction": track.visible_fraction,
        })
        print(
            f"prepared {sample_name} -> {split} "
            f"(context={context_steps}, future={future_frames.shape[0]}, visible={track.visible_fraction:.3f})"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved manifest to {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
