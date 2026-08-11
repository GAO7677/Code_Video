#!/usr/bin/env python3
"""Run PhysRVG with the same Verified temporal protocol as the xSSC run.

The untouched official pipeline only fixes two temporal condition latents. This
adapter imports a local pipeline copy whose mask covers every latent encoded from
the full 72-frame input. Wan's temporal VAE maps that input to a 69-frame clean
prefix in a 189-frame sample. The prefix is removed and only the following 120
predicted frames are submitted at 24 FPS.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physrvg-root", type=Path, required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--dit-checkpoint", type=Path, required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--condition-fps", type=int, default=24)
    parser.add_argument("--condition-frames", type=int, default=72)
    parser.add_argument("--model-context-frames", type=int, default=72)
    parser.add_argument("--model-chunk-frames", type=int, default=189)
    parser.add_argument("--clean-prefix-frames", type=int, default=69)
    parser.add_argument("--model-fps", type=int, default=24)
    parser.add_argument("--target-fps", type=int, default=24)
    parser.add_argument("--target-frames", type=int, default=120)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def require(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(source_height * scale), round(source_width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    return torchvision.transforms.functional.center_crop(image, (height, width))


def load_condition(path: Path, fps: int, frame_count: int, height: int, width: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        actual_fps = float(reader.get_meta_data()["fps"])
        for rgb in reader:
            frames.append(crop_and_resize(Image.fromarray(rgb).convert("RGB"), height, width))
    if not math.isclose(actual_fps, fps, abs_tol=0.01):
        raise ValueError(f"condition FPS mismatch for {path}: expected {fps}, got {actual_fps}")
    if len(frames) != frame_count:
        raise ValueError(f"condition frame mismatch for {path}: expected {frame_count}, got {len(frames)}")
    return frames


def resolve_source(case_json: Path, payload: dict) -> Path:
    declared = Path(payload["source_video"])
    if declared.exists():
        return declared.resolve()
    colocated = case_json.parent.parent / "conditioning" / "24FPS" / declared.name
    return require(colocated, "remapped condition video")


def load_cases(input_list: Path, max_items: int | None) -> list[tuple[Path, dict]]:
    paths = [Path(line.strip()) for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != 198:
        raise ValueError(f"strict Verified BPP list must contain 198 cases, found {len(paths)}")
    cases = []
    for declared in paths:
        case_path = declared if declared.exists() else input_list.parent / "jsons" / declared.name
        case_path = require(case_path, "case JSON")
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        if payload.get("prompt_setting") != "bpp" or payload.get("input_mode") != "v2v":
            raise ValueError(f"case is not BPP V2V: {case_path}")
        cases.append((case_path, payload))
    return cases if max_items is None else cases[:max_items]


def load_pipeline(args: argparse.Namespace):
    sys.path.insert(0, str(args.physrvg_root))
    from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel
    from pipeline_wan_v2v_72f import WanImageToVideoPipeline

    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_id), subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model_id), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(args.model_id), transformer=transformer, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.transformer.load_state_dict(load_file(str(args.dit_checkpoint)))
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(args.lora_checkpoint))
    pipe.transformer.set_adapter("default")
    pipe.to(torch.device(args.device))
    return pipe


def generate_future(pipe, prompt: str, condition: list[Image.Image], args: argparse.Namespace) -> list[np.ndarray]:
    set_seed(args.seed)
    sample = pipe(
        video=condition,
        device=torch.device(args.device),
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        height=args.height,
        width=args.width,
        num_frames=args.model_chunk_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        do_cfg=False,
    )[0][0]
    if len(sample) != args.model_chunk_frames:
        raise RuntimeError(f"PhysRVG returned {len(sample)} frames, expected {args.model_chunk_frames}")
    future = list(sample[args.clean_prefix_frames :])
    if len(future) != args.target_frames:
        raise RuntimeError(
            f"prediction segment has {len(future)} frames, expected {args.target_frames}"
        )
    return future


def probe_submission(path: Path, fps: int, frames: int) -> None:
    with imageio.get_reader(str(path), format="FFMPEG") as reader:
        actual_fps = float(reader.get_meta_data()["fps"])
        actual_frames = int(reader.count_frames())
    if not math.isclose(actual_fps, fps, abs_tol=0.01) or actual_frames != frames:
        raise RuntimeError(
            f"invalid submission {path}: expected {frames}@{fps}, got {actual_frames}@{actual_fps}"
        )


def main() -> None:
    args = parse_args()
    args.physrvg_root = require(args.physrvg_root, "PhysRVG root")
    args.model_id = require(args.model_id, "base model")
    args.dit_checkpoint = require(args.dit_checkpoint, "DiT checkpoint")
    args.lora_checkpoint = require(args.lora_checkpoint, "LoRA checkpoint")
    args.input_list = require(args.input_list, "shared input list")
    if args.model_context_frames != 72 or args.model_chunk_frames != 189:
        raise ValueError("xSSC-aligned mode requires 72-frame context and 189 total frames")
    if args.clean_prefix_frames != 69:
        raise ValueError("xSSC-aligned Wan temporal prefix must be 69 frames")
    if args.condition_frames != 72 or args.condition_fps != 24:
        raise ValueError("strict Verified condition requires exactly 72 frames at 24 FPS")
    if args.target_frames != 120 or args.target_fps != 24 or args.model_fps != 24:
        raise ValueError("shared Verified submission requires exactly 120 frames at 24 FPS")

    all_cases = load_cases(args.input_list, args.max_items)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid shard {args.shard_index}/{args.num_shards}; expected 0 <= shard-index < num-shards"
        )
    cases = all_cases[args.shard_index :: args.num_shards]
    run_dir = (args.output_root / args.run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(args)
    manifest = {
        "protocol": "physics-iq-verified-bpp-v2v-strict",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_total_cases": len(all_cases),
        "num_shard_cases": len(cases),
        "input_list": str(args.input_list),
        "condition": {"fps": 24, "frames": 72, "seconds": 3.0},
        "physrvg_xssc_aligned": {
            "input_context_frames": 72,
            "total_frames": 189,
            "clean_prefix_frames": 69,
            "fps": 24,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "do_cfg": False,
            "height": args.height,
            "width": args.width,
            "seed": args.seed,
        },
        "submission": {"prediction_only": True, "fps": 24, "frames": 120, "seconds": 5.0},
        "cases": [],
    }

    for index, (case_json, payload) in enumerate(cases, start=1):
        output_path = run_dir / payload["generated_video_name"]
        sidecar_path = output_path.with_suffix(".json")
        source_path = resolve_source(case_json, payload)
        if output_path.exists() and not args.force:
            probe_submission(output_path, args.target_fps, args.target_frames)
            status = "skipped_valid_existing"
        else:
            condition = load_condition(
                source_path, args.condition_fps, args.condition_frames, args.height, args.width
            )
            submission = generate_future(pipe, payload["input_caption"], condition, args)
            export_to_video(submission, str(output_path), fps=args.target_fps, macro_block_size=1)
            probe_submission(output_path, args.target_fps, args.target_frames)
            status = "generated"
        case_record = {
            "index": index,
            "case_json": str(case_json),
            "source_video": str(source_path),
            "generated_video_name": payload["generated_video_name"],
            "input_caption": payload["input_caption"],
            "condition_frames_validated": 72,
            "model_context_policy": "all 72 official condition frames encoded as condition latents",
            "output_video": str(output_path),
            "status": status,
        }
        sidecar_path.write_text(json.dumps(case_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["cases"].append(case_record)
        print(f"[{index:03d}/{len(cases):03d}] {status}: {output_path.name}", flush=True)

    shard_suffix = "" if args.num_shards == 1 else f"_shard{args.shard_index:02d}-of-{args.num_shards:02d}"
    manifest_path = args.output_root / f"{args.run_name}{shard_suffix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
