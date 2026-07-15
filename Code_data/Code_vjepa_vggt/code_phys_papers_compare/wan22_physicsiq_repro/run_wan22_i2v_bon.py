from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate resumable Wan2.2 BoN candidates.")
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=42000000)
    parser.add_argument("--frame-num", type=int, default=121)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    return parser.parse_args()


def valid_video(path: Path, frames: int) -> bool:
    if not path.is_file():
        return False
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count == frames


def main() -> None:
    args = parse_args()
    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    sys.path.insert(0, str(args.wan_repo.resolve()))
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    config = WAN_CONFIGS["ti2v-5B"]
    pipeline = wan.WanTI2V(
        config=config, checkpoint_dir=str(args.checkpoint_dir.resolve()),
        device_id=args.device, rank=0, t5_fsdp=False, dit_fsdp=False,
        use_sp=False, t5_cpu=True, convert_model_dtype=True,
    )
    for item_index, entry in enumerate(entries):
        case_root = args.output_root / entry["benchmark_id"]
        case_root.mkdir(parents=True, exist_ok=True)
        image = Image.open(entry["conditioning_image"]).convert("RGB")
        for candidate_index in range(args.candidates):
            output = case_root / f"candidate_{candidate_index:02d}.mp4"
            record = case_root / f"candidate_{candidate_index:02d}.json"
            if valid_video(output, args.frame_num) and record.is_file():
                print(f"[skip] case={entry['benchmark_id']} candidate={candidate_index:02d}")
                continue
            seed = args.base_seed + int(entry["benchmark_id"]) * args.candidates + candidate_index
            started = time.monotonic()
            video = pipeline.generate(
                input_prompt=entry["prompt"], img=image,
                max_area=MAX_AREA_CONFIGS["1280*704"], frame_num=args.frame_num,
                shift=args.sample_shift, sample_solver="unipc",
                sampling_steps=args.sample_steps, guide_scale=args.guide_scale,
                seed=seed, offload_model=False,
            )
            save_video(video[None], str(output), fps=config.sample_fps, nrow=1,
                       normalize=True, value_range=(-1, 1))
            record.write_text(json.dumps({
                **entry, "candidate_index": candidate_index, "seed": seed,
                "output_video": str(output), "generation_seconds": time.monotonic() - started,
                "model": "Wan2.2-TI2V-5B", "frame_num": args.frame_num,
                "sample_fps": config.sample_fps, "sample_steps": args.sample_steps,
                "sample_shift": args.sample_shift, "guide_scale": args.guide_scale,
            }, indent=2, ensure_ascii=False) + "\n")
            del video
            torch.cuda.empty_cache()
            print(f"[done] case={entry['benchmark_id']} candidate={candidate_index:02d}")


if __name__ == "__main__":
    main()
