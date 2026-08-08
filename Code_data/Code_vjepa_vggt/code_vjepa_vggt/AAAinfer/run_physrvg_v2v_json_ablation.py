#!/usr/bin/env python3
"""Run a flat PhysRVG DiT/LoRA ablation on staged V2V JSON cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import torch
import torchvision.transforms.functional as TF
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode

from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel
from fastvideo.models.wan_v2v.pipeline_wan_v2v import WanImageToVideoPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-output-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dit-checkpoint", required=True)
    parser.add_argument("--lora-checkpoint", required=True)
    parser.add_argument("--model-variant", choices=("dit", "full"), required=True)
    parser.add_argument("--negative-prompt", required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-cfg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    source_width, source_height = image.size
    scale = max(width / source_width, height / source_height)
    resized = TF.resize(
        image,
        (round(source_height * scale), round(source_width * scale)),
        interpolation=InterpolationMode.BILINEAR,
    )
    return TF.center_crop(resized, (height, width))


def read_context(path: Path, count: int, height: int, width: int) -> list[Image.Image]:
    with imageio.get_reader(str(path)) as reader:
        available = int(reader.count_frames())
        if available < count:
            raise ValueError(f"{path} has {available} frames; {count} required")
        frames = [Image.fromarray(reader.get_data(index)) for index in range(count)]
    return [crop_and_resize(frame, height, width) for frame in frames]


def load_pipeline(args: argparse.Namespace) -> WanImageToVideoPipeline:
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        args.model_id,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )
    pipe.transformer.load_state_dict(load_file(args.dit_checkpoint))
    if args.model_variant == "full":
        pipe.transformer = PeftModel.from_pretrained(
            pipe.transformer,
            args.lora_checkpoint,
        )
        pipe.transformer.set_adapter("default")
    pipe.to(torch.device(f"cuda:{args.device}"))
    return pipe


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@torch.no_grad()
def run_case(
    args: argparse.Namespace,
    pipe: WanImageToVideoPipeline,
    video_path: Path,
    origins: dict[str, dict[str, str]],
) -> None:
    stem = video_path.stem
    output_video = args.output_dir / f"{stem}.mp4"
    output_json = args.output_dir / f"{stem}.json"
    if output_video.is_file() and output_json.is_file():
        print(f"[{stem}] complete; skipping", flush=True)
        return

    payload = load_json(video_path.with_suffix(".json"))
    prompt = str(payload["input_caption"])
    context = read_context(
        video_path,
        args.context_frames,
        args.height,
        args.width,
    )
    set_seed(args.seed)
    print(
        f"[{stem}] variant={args.model_variant} seed={args.seed} "
        f"negative_prompt={args.negative_prompt!r} do_cfg={args.do_cfg}",
        flush=True,
    )
    generated = pipe(
        video=context,
        device=torch.device(f"cuda:{args.device}"),
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        do_cfg=args.do_cfg,
    )[0]
    export_to_video(generated[0], str(output_video), fps=args.fps)

    origin = origins[stem]
    method = (
        "physRVG_test5_LoRA_ON_steps40_512x896_08_49f"
        if args.model_variant == "full"
        else "physRVG_test5_LoRA_OFF_steps40_512x896_08_49f"
    )
    metadata_root = args.metadata_output_root
    metadata = {
        "dataset": "test_5",
        "input_json": origin["input_json"],
        "input_video": origin["input_video"],
        "input_video_original": origin["input_video"],
        "input_caption": prompt,
        "source_video": payload.get("source_video"),
        "output_video": str(metadata_root / f"{stem}.mp4"),
        "method": method,
        "seed": args.seed,
        "frame_indices": list(range(args.context_frames)),
        "effective_context_frames": args.context_frames,
        "step": args.num_inference_steps,
        "guidance": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "do_cfg": args.do_cfg,
        "model_args": {
            "model_id": args.model_id,
            "dit_checkpoint": args.dit_checkpoint,
            "lora_checkpoint": (
                args.lora_checkpoint if args.model_variant == "full" else None
            ),
            "model_variant": (
                "finetuned_dit_plus_lora"
                if args.model_variant == "full"
                else "finetuned_dit"
            ),
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "negative_prompt": args.negative_prompt,
            "do_cfg": args.do_cfg,
        },
    }
    atomic_json(output_json, metadata)
    print(f"[{stem}] wrote {output_video}", flush=True)


def main() -> None:
    args = parse_args()
    if args.device == 4:
        raise ValueError("GPU 4 is prohibited")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.input_dir / "staging_manifest.json")
    origins = {row["case"]: row for row in manifest["prepared"]}
    videos = sorted(args.input_dir.glob("*.mp4"))
    if set(path.stem for path in videos) != set(origins):
        raise RuntimeError("Staged MP4 set does not match staging_manifest.json")
    if args.max_cases is not None:
        videos = videos[: args.max_cases]
    pipe = load_pipeline(args)
    for video_path in videos:
        run_case(args, pipe, video_path, origins)
    print(f"complete variant={args.model_variant} cases={len(videos)}", flush=True)


if __name__ == "__main__":
    main()
