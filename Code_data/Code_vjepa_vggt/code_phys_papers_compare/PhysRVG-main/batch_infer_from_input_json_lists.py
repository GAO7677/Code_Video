from __future__ import annotations

"""
Batch PhysRVG inference driven by txt files that list input-json paths.

The input json is expected to contain at least:
  - input_video: context video path
  - input_caption: caption/prompt

Outputs are organized as:
  <output_root>/<dataset_name>/<method_name>/

where:
  dataset_name := derived from the txt filename, e.g.
      v2v_jsons_physicIQ.txt           -> physicIQ
      v2v_jsons_morpheus_real_world.txt -> morpheus_real_world

  method_name := physRVG_ctx{xx}_{yy}f
      xx = effective context frame count read from input_video
      yy = output num_frames

Example:
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main
source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate vjepa2
PYTHONNOUSERSITE=0 python batch_infer_from_input_json_lists.py \
    --input-json-list-paths \
        /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt \
        /data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt \
    --height 512 \
    --width 896 \
    --num-inference-steps 40 
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import imageio
import torch
import torchvision
from PIL import Image
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import PeftModel
from safetensors.torch import load_file

from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel
from fastvideo.models.wan_v2v.pipeline_wan_v2v import WanImageToVideoPipeline


DEFAULT_MODEL_ID = Path("/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers")
DEFAULT_DIT = Path("/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors")
DEFAULT_LORA = Path("/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare")
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run PhysRVG inference over txt files containing one json path per line."
    )
    parser.add_argument(
        "--input-json-list-paths",
        type=Path,
        nargs="+",
        required=True,
        help="One or more txt files containing input-json paths.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dit-checkpoint", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_LORA)
    parser.add_argument(
        "--disable-lora",
        action="store_true",
        help="Load the PhysRVG finetuned DiT without the PhysRVG LoRA adapter.",
    )
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help="Rewrite absolute paths from input lists/JSONs, for example /data/gaoya=/home/gaoya/data.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _ensure_exists(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def _read_list_file(path: Path) -> list[Path]:
    entries: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(Path(line).expanduser().resolve())
    return entries


def _parse_path_prefix_maps(values: list[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        source, separator, destination = value.partition("=")
        if not separator or not source or not destination:
            raise ValueError(f"invalid --path-prefix-map value: {value!r}")
        mappings.append((source.rstrip("/"), destination.rstrip("/")))
    return mappings


def _map_path(path: Path, mappings: list[tuple[str, str]]) -> Path:
    value = str(path.expanduser())
    for source, destination in mappings:
        if value == source or value.startswith(source + "/"):
            value = destination + value[len(source) :]
            break
    return Path(value).resolve()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ensure_str_field(payload: dict, key: str, json_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing non-empty string field '{key}' in {json_path}")
    return value.strip()


def _dataset_name_from_list_path(list_path: Path) -> str:
    stem = list_path.stem
    prefix = "v2v_jsons_"
    if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    return stem


def _method_name(
    num_inference_steps: int,
    height: int,
    width: int,
    context_frames: int,
    output_frames: int,
    disable_lora: bool = False,
) -> str:
    prefix = "physRVG_finetunedDiT_noLoRA" if disable_lora else "physRVG"
    return f"{prefix}_steps{num_inference_steps}_{height}x{width}_{int(context_frames):02d}_{int(output_frames):02d}f"


def _crop_and_resize(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height * scale), round(width * scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image


def _open_video_reader(video_path: Path):
    try:
        return imageio.get_reader(str(video_path), format="FFMPEG")
    except Exception:
        return imageio.get_reader(str(video_path))


def _safe_video_length(reader) -> int:
    try:
        frame_count = int(reader.count_frames())
        if frame_count > 0:
            return frame_count
    except Exception:
        pass

    try:
        frame_count = int(reader.get_length())
        if frame_count > 0:
            return frame_count
    except Exception:
        pass

    try:
        meta = reader.get_meta_data()
    except Exception:
        meta = {}

    fps = meta.get("fps")
    duration = meta.get("duration")
    if fps and duration:
        estimated = int(round(float(fps) * float(duration)))
        if estimated > 0:
            return estimated

    raise RuntimeError("unable to determine video frame count")


def _load_context_video(video_path: Path, target_height: int, target_width: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    with _open_video_reader(video_path) as reader:
        frame_count = _safe_video_length(reader)
        for frame_id in range(frame_count):
            frame = reader.get_data(frame_id)
            pil_image = Image.fromarray(frame).convert("RGB")
            pil_image = _crop_and_resize(pil_image, target_height, target_width)
            frames.append(pil_image)
    if not frames:
        raise RuntimeError(f"no readable frames found in {video_path}")
    return frames


def _save_context_contact_sheet(context_frames: list[Image.Image], output_path: Path) -> None:
    widths = [int(image.width) for image in context_frames]
    heights = [int(image.height) for image in context_frames]
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    cursor_x = 0
    for image in context_frames:
        canvas.paste(image, (cursor_x, 0))
        cursor_x += int(image.width)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def _load_pipe(args: argparse.Namespace) -> WanImageToVideoPipeline:
    vae = AutoencoderKLWan.from_pretrained(
        str(args.model_id), subfolder="vae", torch_dtype=torch.float32
    )
    transformer = WanTransformer3DModel.from_pretrained(
        str(args.model_id), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        str(args.model_id), transformer=transformer, vae=vae, torch_dtype=torch.bfloat16
    )

    state_dict = load_file(str(args.dit_checkpoint))
    pipe.transformer.load_state_dict(state_dict)
    if not args.disable_lora:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(args.lora_checkpoint))
        pipe.transformer.set_adapter("default")
    pipe.to(torch.device(args.device))
    return pipe


def _run_single_case(
    *,
    pipe: WanImageToVideoPipeline,
    args: argparse.Namespace,
    input_json_path: Path,
    dataset_name: str,
    payload: dict,
    summary_entries: dict[str, list[dict]],
) -> tuple[bool, str]:
    input_video = _map_path(
        Path(_ensure_str_field(payload, "input_video", input_json_path)), args.path_prefix_maps
    )
    input_caption = _ensure_str_field(payload, "input_caption", input_json_path)
    source_video = payload.get("source_video")
    source_video = (
        str(_map_path(Path(source_video.strip()), args.path_prefix_maps))
        if isinstance(source_video, str) and source_video.strip()
        else None
    )

    context_frames = _load_context_video(
        input_video, target_height=int(args.height), target_width=int(args.width)
    )
    effective_context_frames = len(context_frames)
    method_name = _method_name(
        args.num_inference_steps,
        args.height,
        args.width,
        effective_context_frames,
        int(args.num_frames),
        disable_lora=bool(args.disable_lora),
    )
    output_dir = args.output_root / dataset_name / method_name
    sample_stem = input_json_path.stem
    output_video = output_dir / f"{sample_stem}.mp4"
    output_json = output_dir / f"{sample_stem}.json"
    output_log = output_dir / f"{sample_stem}.log"
    ctx_sheet = output_dir / f"{sample_stem}_input_ctx{effective_context_frames:02d}.jpg"

    if output_video.exists() and output_json.exists() and not args.force:
        return False, f"[skip] {dataset_name} {method_name} {sample_stem}"

    generator = torch.Generator(device=str(args.device)).manual_seed(int(args.seed))
    sample = pipe(
        video=context_frames,
        device=torch.device(args.device),
        prompt=input_caption,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        height=int(args.height),
        width=int(args.width),
        num_frames=int(args.num_frames),
        num_inference_steps=int(args.num_inference_steps),
        guidance_scale=float(args.guidance_scale),
        do_cfg=False,
        generator=generator,
    )[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_context_contact_sheet(context_frames, ctx_sheet)
    export_to_video(sample[0], str(output_video), fps=int(args.fps))

    result = {
        "input_json": str(input_json_path),
        "input_video": str(ctx_sheet),
        "input_video_original": str(input_video),
        "input_ctx_contact_sheet": str(ctx_sheet),
        "source_video": source_video,
        "input_caption": input_caption,
        "output_video": str(output_video),
        "dataset": dataset_name,
        "method": method_name,
        "seed": int(args.seed),
        "step": int(args.num_inference_steps),
        "guidance": float(args.guidance_scale),
        "effective_context_frames": int(effective_context_frames),
        "frame_indices": list(range(int(effective_context_frames))),
        "model_args": {
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "fps": int(args.fps),
            "model_id": str(args.model_id),
            "dit_checkpoint": str(args.dit_checkpoint),
            "lora_checkpoint": None if args.disable_lora else str(args.lora_checkpoint),
            "model_variant": "finetuned_dit" if args.disable_lora else "finetuned_dit_plus_lora",
        },
    }
    _write_json(output_json, result)
    _write_lines(
        output_log,
        [
            f"[dataset] {dataset_name}",
            f"[method] {method_name}",
            f"[input_json] {input_json_path}",
            f"[input_video] {input_video}",
            f"[input_caption] {input_caption}",
            f"[output_video] {output_video}",
            f"[done] success",
        ],
    )
    summary_entries[method_name].append(result)
    return True, f"[done] {dataset_name} {method_name} {sample_stem} -> {output_video}"


def main() -> None:
    args = parse_args()
    if args.shard_count <= 0:
        raise ValueError("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < shard-count")
    args.path_prefix_maps = _parse_path_prefix_maps(args.path_prefix_map)
    args.model_id = _ensure_exists(args.model_id, "model-id")
    args.dit_checkpoint = _ensure_exists(args.dit_checkpoint, "dit-checkpoint")
    if not args.disable_lora:
        args.lora_checkpoint = _ensure_exists(args.lora_checkpoint, "lora-checkpoint")
    else:
        args.lora_checkpoint = args.lora_checkpoint.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()

    list_paths = [_ensure_exists(path, "input-json-list-path") for path in args.input_json_list_paths]
    pipe = _load_pipe(args)

    for list_path in list_paths:
        dataset_name = _dataset_name_from_list_path(list_path)
        dataset_output_root = args.output_root / dataset_name
        dataset_output_root.mkdir(parents=True, exist_ok=True)

        all_input_json_paths = [
            _map_path(path, args.path_prefix_maps) for path in _read_list_file(list_path)
        ]
        input_json_paths = [
            path
            for index, path in enumerate(all_input_json_paths)
            if index % args.shard_count == args.shard_index
        ]
        manifest = {
            "input_json_list_path": str(list_path),
            "dataset": dataset_name,
            "num_items": len(input_json_paths),
            "num_items_total": len(all_input_json_paths),
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "output_root": str(dataset_output_root),
            "model_id": str(args.model_id),
            "dit_checkpoint": str(args.dit_checkpoint),
            "lora_checkpoint": None if args.disable_lora else str(args.lora_checkpoint),
            "model_variant": "finetuned_dit" if args.disable_lora else "finetuned_dit_plus_lora",
            "device": str(args.device),
            "height": int(args.height),
            "width": int(args.width),
            "num_frames": int(args.num_frames),
            "fps": int(args.fps),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "seed": int(args.seed),
        }
        shard_suffix = (
            "" if args.shard_count == 1 else f".shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
        )
        run_tag = "finetuned-dit-no-lora" if args.disable_lora else "finetuned-dit-plus-lora"
        manifest_name = (
            "batch_manifest.json"
            if not args.disable_lora and args.shard_count == 1
            else f"batch_manifest.{run_tag}{shard_suffix}.json"
        )
        _write_json(
            dataset_output_root / manifest_name,
            manifest,
        )

        method_entries: dict[str, list[dict]] = defaultdict(list)
        num_success = 0
        num_failed = 0
        num_skipped = 0

        for input_json_path in input_json_paths:
            try:
                payload = _load_json(input_json_path)
                did_run, message = _run_single_case(
                    pipe=pipe,
                    args=args,
                    input_json_path=input_json_path,
                    dataset_name=dataset_name,
                    payload=payload,
                    summary_entries=method_entries,
                )
                print(message)
                if did_run:
                    num_success += 1
                else:
                    num_skipped += 1
            except Exception as exc:
                print(f"[error] {dataset_name} {input_json_path}: {exc}")
                num_failed += 1

        for method_name, entries in method_entries.items():
            _write_json(
                dataset_output_root / method_name / f"result{shard_suffix}.json",
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "num_total": len(entries),
                    "num_success": len(entries),
                    "entries": entries,
                },
            )

        summary_name = (
            "summary.json"
            if not args.disable_lora and args.shard_count == 1
            else f"summary.{run_tag}{shard_suffix}.json"
        )
        _write_json(
            dataset_output_root / summary_name,
            {
                "dataset": dataset_name,
                "input_json_list_path": str(list_path),
                "num_total": len(input_json_paths),
                "num_success": num_success,
                "num_failed": num_failed,
                "num_skipped": num_skipped,
                "methods": sorted(method_entries.keys()),
            },
        )


if __name__ == "__main__":
    main()
