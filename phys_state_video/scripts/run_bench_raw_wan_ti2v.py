from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.utils import require_torch
from phys_state_video.wan_runtime import load_wan_modules

torch = require_torch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw Wan TI2V base model on benchmark json entries."
    )
    parser.add_argument("--bench-json", required=True, help="Path to A/B/D bench json.")
    parser.add_argument("--output-dir", required=True, help="Flat output directory for mp4/json files.")
    parser.add_argument("--wan-ckpt-dir", required=True, help="Wan TI2V checkpoint directory.")
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--task", default="ti2v-5B")
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guide-scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offload-model", action="store_true", default=False)
    parser.add_argument("--convert-model-dtype", action="store_true", default=False)
    parser.add_argument("--t5-cpu", action="store_true", default=False)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _slugify(text: str) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    normalized = re.sub(r"[^0-9A-Za-z_\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "case"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected list json at {path}, got {type(payload).__name__}")
    cases: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"expected dict case entry, got {type(item).__name__}")
        cases.append(item)
    return cases


def _write_mp4(path: Path, frames_cthw: torch.Tensor, fps: int) -> None:
    import cv2

    frames = frames_cthw.detach().float().cpu()
    if frames.ndim != 4:
        raise ValueError(f"expected generated video with shape [C, T, H, W], got {tuple(frames.shape)}")
    if float(frames.min()) >= -1.0 and float(frames.max()) <= 1.0:
        frames = (frames + 1.0) * 0.5
    frames = frames.clamp(0.0, 1.0).permute(1, 2, 3, 0).numpy()
    frames = (frames * 255.0).round().astype(np.uint8)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _case_stem(index: int, category: str, caption: str) -> str:
    return f"{index:03d}_{_slugify(category)}_{_slugify(caption)}_gen"


def main() -> None:
    args = parse_args()
    if not str(args.device).startswith("cuda"):
        raise ValueError("raw Wan TI2V benchmark script currently requires a CUDA device")

    bench_json = Path(args.bench_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modules = load_wan_modules(args.wan_repo_root)
    if args.task not in modules["WAN_CONFIGS"]:
        raise ValueError(f"unsupported Wan task: {args.task}")
    if args.size not in modules["SUPPORTED_SIZES"][args.task]:
        raise ValueError(
            f"unsupported size '{args.size}' for task '{args.task}', "
            f"expected one of {modules['SUPPORTED_SIZES'][args.task]}"
        )

    config = modules["WAN_CONFIGS"][args.task]
    frame_num = int(args.frame_num or config.frame_num)
    sampling_steps = int(args.sampling_steps or config.sample_steps)
    guide_scale = float(args.guide_scale or config.sample_guide_scale)
    shift = float(args.shift or config.sample_shift)
    fps = int(config.sample_fps)
    size_hw = modules["SIZE_CONFIGS"][args.size]
    max_area = modules["MAX_AREA_CONFIGS"][args.size]

    device_id = int(str(args.device).split(":")[1]) if ":" in str(args.device) else 0
    pipeline = modules["WanTI2V"](
        config=config,
        checkpoint_dir=str(args.wan_ckpt_dir),
        device_id=device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    cases = _load_cases(bench_json)
    end_index = len(cases) if args.end_index is None else min(int(args.end_index), len(cases))

    for case_index in range(int(args.start_index), end_index):
        case = cases[case_index]
        caption = str(case["caption"])
        first_frame_path = Path(case["first_frame"])
        source_video_path = str(case["source_video"])
        category = str(case["category"])
        stem = _case_stem(case_index, category, caption)
        mp4_path = output_dir / f"{stem}.mp4"
        json_path = output_dir / f"{stem}.json"
        if mp4_path.exists() and json_path.exists() and not args.overwrite:
            print(f"[skip] {stem}")
            continue

        image = Image.open(first_frame_path).convert("RGB")
        print(f"[run] {stem}")
        with torch.no_grad():
            video = pipeline.generate(
                input_prompt=caption,
                img=image,
                size=size_hw,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=args.sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                seed=args.seed,
                offload_model=args.offload_model,
            )
        if video is None:
            raise RuntimeError(f"WanTI2V returned None for {stem}")

        _write_mp4(mp4_path, video, fps=fps)
        meta = {
            "input_prompt": caption,
            "input_image": str(first_frame_path),
            "source_video": source_video_path,
            "model_name": "Wan-AI-Wan2.2-TI2V-5B_base",
            "model_ckpt_dir": str(args.wan_ckpt_dir),
            "size": args.size,
            "frame_num": frame_num,
            "sample_solver": args.sample_solver,
            "sampling_steps": sampling_steps,
            "guide_scale": guide_scale,
            "shift": shift,
            "seed": args.seed,
            "output_video": str(mp4_path),
        }
        json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(f"[done] {stem}")


if __name__ == "__main__":
    main()
