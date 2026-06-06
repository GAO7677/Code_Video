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

from phys_state_video.mask_tracking import (
    SAM2VideoMaskTracker,
    build_mask_track_outputs,
    build_proxy_prompt_box,
    load_prompt_boxes_from_json,
)
from phys_state_video.proxy_state import read_video_frames


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert benchmark json samples into SAM2-tracked phys-state-video episodes."
    )
    parser.add_argument("--bench-json-root", required=True, help="Directory containing A/B/D style benchmark json files.")
    parser.add_argument("--json-names", nargs="+", default=["A.json", "D.json"])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--height", type=int, default=144)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--limit-per-json", type=int, default=None)
    parser.add_argument("--context-steps", type=int, default=None)
    parser.add_argument("--context-ratio", type=float, default=0.25)
    parser.add_argument("--future-steps", type=int, default=None)
    parser.add_argument("--prompt-mode", choices=["proxy_box", "manual_box_json"], default="proxy_box")
    parser.add_argument("--prompt-box-json", default=None)
    parser.add_argument("--prompt-frame", choices=["last_context", "first"], default="last_context")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sam2-model-id", default="facebook/sam2.1-hiera-small")
    parser.add_argument("--sam2-config", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def clean_output_dir(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            for sub in sorted(child.rglob("*"), reverse=True):
                if sub.is_file() or sub.is_symlink():
                    sub.unlink()
                elif sub.is_dir():
                    sub.rmdir()
            child.rmdir()


def resolve_context_steps(item: dict, total_steps: int, args) -> int:
    if args.context_steps is not None:
        return min(max(int(args.context_steps), 1), total_steps - 1)
    context_video = item.get("context_video")
    if context_video:
        path = Path(str(context_video))
        if path.exists():
            context_frames = read_video_frames(
                path,
                resize_height=args.height,
                resize_width=args.width,
            )[:: args.frame_stride]
            return min(max(int(context_frames.shape[0]), 1), total_steps - 1)
    ratio_steps = int(round(total_steps * float(args.context_ratio)))
    return min(max(ratio_steps, 1), total_steps - 1)


def resolve_prompt_frame_idx(context_steps: int, mode: str) -> int:
    if mode == "first":
        return 0
    return max(int(context_steps) - 1, 0)


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    if args.clean:
        clean_output_dir(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    prompt_boxes = {}
    if args.prompt_mode == "manual_box_json":
        if not args.prompt_box_json:
            raise ValueError("--prompt-box-json is required when --prompt-mode=manual_box_json")
        prompt_boxes = load_prompt_boxes_from_json(args.prompt_box_json)

    tracker = SAM2VideoMaskTracker(
        device=device,
        model_id=args.sam2_model_id,
        model_cfg=args.sam2_config,
        checkpoint_path=args.sam2_ckpt,
    )

    manifest: dict[str, object] = {
        "bench_json_root": args.bench_json_root,
        "json_names": args.json_names,
        "height": args.height,
        "width": args.width,
        "frame_stride": args.frame_stride,
        "prompt_mode": args.prompt_mode,
        "samples": [],
    }

    for json_name in args.json_names:
        json_path = Path(args.bench_json_root) / json_name
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        limit = len(payload) if args.limit_per_json is None else min(len(payload), int(args.limit_per_json))
        for sample_index, item in enumerate(payload[:limit]):
            source_video = Path(str(item["source_video"]))
            if not source_video.exists():
                print(f"skip missing source_video: {source_video}")
                continue
            prompt = str(item.get("caption") or "")
            category = str(item.get("category") or "unknown")
            sample_key = f"{json_path.stem.lower()}_{sample_index:03d}"
            frames = read_video_frames(
                source_video,
                resize_height=args.height,
                resize_width=args.width,
            )[:: args.frame_stride]
            if args.future_steps is not None:
                clip_steps = min(int(frames.shape[0]), resolve_context_steps(item, int(frames.shape[0]), args) + int(args.future_steps))
                frames = frames[:clip_steps]
            if frames.shape[0] < 2:
                print(f"skip too-short video: {source_video}")
                continue
            context_steps = resolve_context_steps(item, int(frames.shape[0]), args)
            if context_steps >= int(frames.shape[0]):
                context_steps = int(frames.shape[0]) - 1
            prompt_frame_idx = resolve_prompt_frame_idx(context_steps, args.prompt_frame)
            if args.prompt_mode == "manual_box_json":
                if sample_key not in prompt_boxes:
                    raise KeyError(f"missing prompt box for {sample_key} in {args.prompt_box_json}")
                boxes_xyxy = prompt_boxes[sample_key]
            else:
                boxes_xyxy = build_proxy_prompt_box(
                    frames,
                    prompt_frame_idx=prompt_frame_idx,
                )[None]
            outputs = build_mask_track_outputs(
                frames,
                prompt_frame_idx=prompt_frame_idx,
                prompt_boxes_xyxy=boxes_xyxy,
                prompt_mode=args.prompt_mode,
                tracker=tracker,
            )

            context_frames = frames[:context_steps]
            future_frames = frames[context_steps:]
            episode_path = output_root / f"{sample_key}_{source_video.stem}.npz"
            np.savez_compressed(
                episode_path,
                context_frames=context_frames.astype(np.float32),
                future_frames=future_frames.astype(np.float32),
                context_states=outputs.states[:context_steps].astype(np.float32),
                future_states=outputs.states[context_steps:].astype(np.float32),
                context_boxes=outputs.boxes[:context_steps].astype(np.float32),
                future_boxes=outputs.boxes[context_steps:].astype(np.float32),
                full_frames=frames.astype(np.float32),
                full_states=outputs.states.astype(np.float32),
                full_boxes=outputs.boxes.astype(np.float32),
                full_masks=outputs.masks.astype(np.uint8),
                appearance=outputs.appearance.astype(np.float32),
                camera=outputs.camera_full[:context_steps].astype(np.float32),
                camera_full=outputs.camera_full.astype(np.float32),
                prompt_boxes_xyxy=outputs.prompt_boxes_xyxy.astype(np.float32),
                prompt_frame_idx=np.asarray([outputs.prompt_frame_idx], dtype=np.int64),
            )
            episode_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "category": category,
                        "source_video": str(source_video),
                        "source_index": int(sample_index),
                        "json_name": json_name,
                        "prompt_mode": outputs.prompt_mode,
                        "prompt_frame_idx": int(outputs.prompt_frame_idx),
                        "context_steps": int(context_steps),
                        "future_steps": int(frames.shape[0] - context_steps),
                        "prompt_boxes_xyxy": outputs.prompt_boxes_xyxy.tolist(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            manifest["samples"].append(
                {
                    "sample_key": sample_key,
                    "episode": str(episode_path),
                    "json_name": json_name,
                    "source_video": str(source_video),
                    "source_index": int(sample_index),
                    "category": category,
                    "context_steps": int(context_steps),
                    "future_steps": int(frames.shape[0] - context_steps),
                    "prompt_mode": outputs.prompt_mode,
                    "prompt_frame_idx": int(outputs.prompt_frame_idx),
                }
            )
            print(
                f"prepared {sample_key} from {json_name} "
                f"(context={context_steps}, future={frames.shape[0] - context_steps}, prompt={outputs.prompt_mode})"
            )

    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved manifest to {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
