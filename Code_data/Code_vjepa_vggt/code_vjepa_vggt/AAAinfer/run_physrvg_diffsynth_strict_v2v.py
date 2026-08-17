#!/usr/bin/env python3
"""Run bare PhysRVG through DiffSynth with a verified strict DiT conversion.

This intentionally loads neither the PhysRVG LoRA nor any xSSC/Object/Full-SA
adapter.  It is for comparing the DiffSynth conversion path against the
existing FastVideo/Diffusers LoRA-OFF reference on the same staged V2V cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# The available xformers build rejects the installed FlashAttention version at
# import time.  DiffSynth can use FlashAttention-2 directly, so disable only
# xformers before importing DiffSynth rather than changing the shared env.
os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "flash_attention_2")
sys.modules.setdefault("xformers", None)

DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for root in (DIFFSYNTH_ROOT, PROJECT_ROOT):
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

import imageio.v2 as imageio
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode

from diffsynth.core import ModelConfig
from diffsynth.utils.data import save_video
from diffsynth.utils.state_dict_converters.wan_video_dit import (
    WanVideoDiTFromDiffusers,
)
from code_vjepa_vggt.train0419_reference.context_wan import (
    ContextAwareWanVideoPipeline,
)


DEFAULT_NEGATIVE_PROMPT = (
    "模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--wan-root",
        type=Path,
        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"),
    )
    parser.add_argument("--physrvg-dit-checkpoint", type=Path, required=True)
    parser.add_argument("--physrvg-revision", default="d8caf2dd7db30d7470d474875c84e6a4eb21b9c6")
    parser.add_argument(
        "--physrvg-sha256",
        default="70c14c374fc9f33a29ed713f68cf7e5db4952ea62ecd1787e63a390ef94918d3",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def resolve_dit_paths(wan_root: Path) -> list[Path]:
    sharded = sorted(wan_root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if sharded:
        return sharded
    single = wan_root / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"No Wan DiT shards found under {wan_root}")


def build_model_configs(wan_root: Path) -> tuple[list[ModelConfig], ModelConfig]:
    dit_paths = resolve_dit_paths(wan_root)
    text_encoder = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    vae = wan_root / "Wan2.2_VAE.pth"
    tokenizer = wan_root / "google" / "umt5-xxl"
    for path in [*dit_paths, text_encoder, vae, tokenizer / "tokenizer.json"]:
        if not path.is_file():
            raise FileNotFoundError(f"Required Wan asset missing: {path}")
    return (
        [
            ModelConfig(path=[str(path) for path in dit_paths]),
            ModelConfig(path=str(text_encoder)),
            ModelConfig(path=str(vae)),
        ],
        ModelConfig(path=str(tokenizer)),
    )


def strict_load_physrvg_dit(
    dit: torch.nn.Module,
    checkpoint: Path,
) -> dict[str, Any]:
    source_state = load_file(str(checkpoint), device="cpu")
    converted_state = WanVideoDiTFromDiffusers(source_state)
    target_state = dit.state_dict()

    missing = sorted(set(target_state) - set(converted_state))
    unexpected = sorted(set(converted_state) - set(target_state))
    shape_mismatch = sorted(
        key
        for key in set(target_state) & set(converted_state)
        if tuple(target_state[key].shape) != tuple(converted_state[key].shape)
    )
    dropped_source_count = len(source_state) - len(converted_state)
    if missing or unexpected or shape_mismatch or dropped_source_count:
        raise RuntimeError(
            "PhysRVG Diffusers-to-DiffSynth DiT conversion is incomplete: "
            f"source={len(source_state)}, converted={len(converted_state)}, "
            f"target={len(target_state)}, dropped_source={dropped_source_count}, "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatch={shape_mismatch[:8]}"
        )

    dit.load_state_dict(converted_state, strict=True)
    return {
        "converter": "WanVideoDiTFromDiffusers",
        "strict": True,
        "source_tensors": len(source_state),
        "converted_tensors": len(converted_state),
        "target_tensors": len(target_state),
        "dropped_source_tensors": dropped_source_count,
        "missing_tensors": 0,
        "unexpected_tensors": 0,
        "shape_mismatches": 0,
    }


def load_pipeline(args: argparse.Namespace) -> tuple[ContextAwareWanVideoPipeline, dict[str, Any]]:
    if args.gpu == 4:
        raise ValueError("GPU 4 is prohibited")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this inference run")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"Invalid GPU index {args.gpu}; found {torch.cuda.device_count()} devices")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    model_configs, tokenizer_config = build_model_configs(args.wan_root)
    pipe = ContextAwareWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    load_info = strict_load_physrvg_dit(pipe.dit, args.physrvg_dit_checkpoint)
    pipe.dit.eval()
    return pipe, load_info


def select_videos(args: argparse.Namespace, origins: dict[str, dict[str, Any]]) -> list[Path]:
    videos = sorted(args.input_dir.glob("*.mp4"))
    stems = {video.stem for video in videos}
    if stems != set(origins):
        missing = sorted(set(origins) - stems)
        unexpected = sorted(stems - set(origins))
        raise RuntimeError(f"Staged video set mismatch: missing={missing}, unexpected={unexpected}")
    if args.case:
        requested = set(args.case)
        absent = sorted(requested - stems)
        if absent:
            raise ValueError(f"Requested case is not staged: {absent}")
        videos = [video for video in videos if video.stem in requested]
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        videos = videos[: args.max_cases]
    return videos


@torch.no_grad()
def run_case(
    args: argparse.Namespace,
    pipe: ContextAwareWanVideoPipeline,
    strict_load: dict[str, Any],
    video_path: Path,
    origin: dict[str, Any],
) -> None:
    stem = video_path.stem
    output_video = args.output_dir / f"{stem}.mp4"
    output_json = args.output_dir / f"{stem}.json"
    if output_video.is_file() and output_json.is_file() and not args.overwrite:
        print(f"[{stem}] complete; skipping", flush=True)
        return

    payload = load_json(video_path.with_suffix(".json"))
    prompt = str(payload["input_caption"])
    context = read_context(video_path, args.context_frames, args.height, args.width)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()
    print(
        f"[{stem}] seed={args.seed} steps={args.num_inference_steps} cfg={args.cfg_scale} "
        f"context={args.context_frames}",
        flush=True,
    )
    video = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        context_video=context,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        tiled=True,
    )
    if len(video) < args.num_frames:
        raise RuntimeError(
            f"[{stem}] DiffSynth returned {len(video)} frames; expected {args.num_frames}"
        )
    video = video[: args.num_frames]

    temporary_video = output_video.with_name(output_video.stem + ".tmp.mp4")
    save_video(video, str(temporary_video), fps=args.fps, quality=args.quality)
    temporary_video.replace(output_video)

    metadata = {
        "dataset": "test_5",
        "input_json": origin["input_json"],
        "input_video": origin["input_video"],
        "input_video_original": origin["input_video"],
        "input_caption": prompt,
        "source_video": payload.get("source_video"),
        "output_video": str(output_video),
        "method": "physRVG_test5_DiffSynth_strict_steps40_512x896_08_49f",
        "seed": args.seed,
        "frame_indices": list(range(args.context_frames)),
        "effective_context_frames": args.context_frames,
        "step": args.num_inference_steps,
        "guidance": args.cfg_scale,
        "negative_prompt": args.negative_prompt,
        "do_cfg": True,
        "model_args": {
            "wan_root": str(args.wan_root),
            "physrvg_dit_checkpoint": str(args.physrvg_dit_checkpoint),
            "physrvg_revision": args.physrvg_revision,
            "physrvg_sha256_expected": args.physrvg_sha256,
            "physrvg_sha256_verified": args.physrvg_sha256_verified,
            "lora_checkpoint": None,
            "model_variant": "finetuned_dit",
            "object_branch": False,
            "full_sa": False,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "negative_prompt": args.negative_prompt,
            "attention_implementation": os.environ.get(
                "DIFFSYNTH_ATTENTION_IMPLEMENTATION"
            ),
        },
        "strict_diffsynth_load": strict_load,
        "runtime_seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(output_json, metadata)
    print(f"[{stem}] wrote {output_video}", flush=True)


def main() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.wan_root = args.wan_root.expanduser().resolve()
    args.physrvg_dit_checkpoint = args.physrvg_dit_checkpoint.expanduser().resolve()
    if not args.physrvg_dit_checkpoint.is_file():
        raise FileNotFoundError(f"PhysRVG DiT checkpoint not found: {args.physrvg_dit_checkpoint}")
    args.physrvg_sha256_verified = sha256_file(args.physrvg_dit_checkpoint)
    if args.physrvg_sha256 and args.physrvg_sha256_verified != args.physrvg_sha256:
        raise RuntimeError(
            "PhysRVG DiT SHA-256 mismatch: "
            f"expected={args.physrvg_sha256}, actual={args.physrvg_sha256_verified}"
        )

    manifest = load_json(args.input_dir / "staging_manifest.json")
    prepared = manifest.get("prepared")
    if not isinstance(prepared, list):
        raise TypeError("staging_manifest.json lacks a prepared case list")
    origins = {str(row["case"]): row for row in prepared}
    videos = select_videos(args, origins)
    if not videos:
        raise RuntimeError("No cases selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipe, strict_load = load_pipeline(args)
    atomic_json(
        args.output_dir / "run_manifest.json",
        {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "selected_cases": [video.stem for video in videos],
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "context_frames": args.context_frames,
            "lora_loaded": False,
            "object_branch_loaded": False,
            "full_sa_loaded": False,
            "strict_diffsynth_load": strict_load,
            "physrvg_revision": args.physrvg_revision,
            "physrvg_sha256_expected": args.physrvg_sha256,
            "physrvg_sha256_verified": args.physrvg_sha256_verified,
        },
    )
    for video_path in videos:
        run_case(args, pipe, strict_load, video_path, origins[video_path.stem])
    print(f"complete cases={len(videos)}", flush=True)


if __name__ == "__main__":
    main()
