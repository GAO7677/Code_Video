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
    parser = argparse.ArgumentParser(
        description="Run a manifest-driven Wan2.2 TI2V-5B Physics-IQ smoke test."
    )
    parser.add_argument("--wan-repo", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42000)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--frame-num", type=int, default=121)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def probe_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open generated video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"invalid generated video metadata: {path}")
    return {
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps,
        "width": width,
        "height": height,
    }


def main() -> None:
    args = parse_args()
    wan_repo = args.wan_repo.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    raw_root = output_root / "raw"
    records_root = output_root / "records"
    raw_root.mkdir(parents=True, exist_ok=True)
    records_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(wan_repo))
    import wan  # pylint: disable=import-error,import-outside-toplevel
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS  # pylint: disable=import-error,import-outside-toplevel
    from wan.utils.utils import save_video  # pylint: disable=import-error,import-outside-toplevel

    config = WAN_CONFIGS["ti2v-5B"]
    entries = read_manifest(manifest_path)
    if args.max_items is not None:
        entries = entries[: max(0, args.max_items)]
    if not entries:
        raise RuntimeError(f"manifest has no selected entries: {manifest_path}")

    print(f"[load] checkpoint={checkpoint_dir} device=cuda:{args.device}")
    load_started = time.monotonic()
    pipeline = wan.WanTI2V(
        config=config,
        checkpoint_dir=str(checkpoint_dir),
        device_id=args.device,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=True,
        convert_model_dtype=True,
    )
    print(f"[load] completed seconds={time.monotonic() - load_started:.3f}")

    for item_index, entry in enumerate(entries):
        benchmark_id = str(entry["benchmark_id"])
        output_path = raw_root / str(entry["generated_video_name"])
        record_path = records_root / f"{output_path.stem}.json"
        seed = args.base_seed + int(benchmark_id)

        if output_path.exists() and not args.force:
            metadata = probe_video(output_path)
            print(f"[skip] {benchmark_id} {metadata}")
            continue

        image = Image.open(str(entry["conditioning_image"])).convert("RGB")
        started = time.monotonic()
        video = pipeline.generate(
            input_prompt=str(entry["prompt"]),
            img=image,
            max_area=MAX_AREA_CONFIGS["1280*704"],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            seed=seed,
            offload_model=args.offload_model,
        )
        generation_seconds = time.monotonic() - started

        save_video(
            tensor=video[None],
            save_file=str(output_path),
            fps=config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        metadata = probe_video(output_path)
        record = {
            **entry,
            "model": "Wan2.2-TI2V-5B",
            "checkpoint_dir": str(checkpoint_dir),
            "wan_repo": str(wan_repo),
            "device": f"cuda:{args.device}",
            "seed": seed,
            "frame_num": args.frame_num,
            "sample_fps": config.sample_fps,
            "sample_steps": args.sample_steps,
            "sample_shift": args.sample_shift,
            "guide_scale": args.guide_scale,
            "sample_solver": "unipc",
            "offload_model": args.offload_model,
            "generation_seconds": generation_seconds,
            "output_video": str(output_path),
            "actual_video": metadata,
            "item_index": item_index,
        }
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        del video
        torch.cuda.empty_cache()
        print(
            f"[done] {benchmark_id} seconds={generation_seconds:.3f} "
            f"frames={metadata['frames']} fps={metadata['fps']} "
            f"size={metadata['width']}x{metadata['height']}"
        )


if __name__ == "__main__":
    main()
