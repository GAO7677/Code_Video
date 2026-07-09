from __future__ import annotations

"""
Generate PhysRVG outputs in the official Physics-IQ Verified format.

This script:
1. Reads the official descriptions CSV.
2. Uses the 198 take-1 conditioning videos from the verified dataset.
3. Runs PhysRVG video-to-video generation.
4. Writes one official-evaluator-compatible run folder containing exactly
   198 generated mp4 files named by `generated_video_name`.

Example:
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main
/home/gaoya/miniconda3/envs/wan-cu128/bin/python generate_physics_iq_verified_official.py \
    --device cuda:0 \
    --output-root /data/gaoya/AAA_test_video/0623/test/physicsiq
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
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
DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified")
DEFAULT_DESCRIPTIONS = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/"
    "physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv"
)
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/physicsiq")
DEFAULT_RUN_NAME = "physRVG-verified-bpp-run_01"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate official-format Physics-IQ Verified outputs with PhysRVG."
    )
    parser.add_argument("--model-id", type=Path, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dit-checkpoint", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--descriptions-file", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=149,
        help="Generation frame count. Must satisfy num_frames %% 4 == 1 for Wan V2V. 149 enables exact 5s export.",
    )
    parser.add_argument(
        "--target-duration-seconds",
        type=float,
        default=5.0,
        help="Official verified evaluator expects exactly 5 seconds.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _ensure_exists(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


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


def _read_verified_rows(descriptions_file: Path, max_items: int | None) -> list[dict[str, str]]:
    with descriptions_file.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if "take-1" in row["scenario"]]

    if len(rows) < 198:
        raise ValueError(f"expected at least 198 take-1 rows in {descriptions_file}, found {len(rows)}")

    rows = rows[:198]
    if max_items is not None:
        rows = rows[: max(0, int(max_items))]
    return rows


def _conditioning_video_name(scenario: str) -> str:
    prefix, remainder = scenario.split("_", 1)
    return f"{prefix}_conditioning-videos_30FPS_{remainder}"


def _validate_num_frames(num_frames: int, target_duration_seconds: float) -> float:
    if num_frames % 4 != 1:
        raise ValueError(
            f"num_frames={num_frames} is incompatible with Wan V2V; expected num_frames % 4 == 1"
        )
    if target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be positive")
    return float(num_frames) / float(target_duration_seconds)


def _probe_video(video_path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open generated video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid generated video metadata: fps={fps}, frame_count={frame_count}, path={video_path}")

    return fps, frame_count / fps


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


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
    pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(args.lora_checkpoint))
    pipe.transformer.set_adapter("default")
    pipe.to(torch.device(args.device))
    return pipe


def main() -> None:
    args = parse_args()
    args.model_id = _ensure_exists(args.model_id, "model-id")
    args.dit_checkpoint = _ensure_exists(args.dit_checkpoint, "dit-checkpoint")
    args.lora_checkpoint = _ensure_exists(args.lora_checkpoint, "lora-checkpoint")
    args.dataset_root = _ensure_exists(args.dataset_root, "dataset-root")
    args.descriptions_file = _ensure_exists(args.descriptions_file, "descriptions-file")
    args.output_root = args.output_root.expanduser().resolve()

    conditioning_root = _ensure_exists(
        args.dataset_root / "split-videos" / "conditioning" / "30FPS",
        "conditioning-video-root",
    )
    run_dir = args.output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    export_fps = _validate_num_frames(int(args.num_frames), float(args.target_duration_seconds))
    rows = _read_verified_rows(args.descriptions_file, args.max_items)
    pipe = _load_pipe(args)

    manifest_entries: list[dict] = []
    for index, row in enumerate(rows):
        scenario = row["scenario"]
        prompt = row["description"]
        output_name = row["generated_video_name"]
        conditioning_name = _conditioning_video_name(scenario)
        conditioning_path = _ensure_exists(conditioning_root / conditioning_name, "conditioning-video")
        output_path = run_dir / output_name
        output_json_path = output_path.with_suffix(".json")

        case_payload_base = {
            "index": index,
            "scenario": scenario,
            "generated_video_name": output_name,
            "category": row.get("category"),
            "input_prompt": prompt,
            "conditioning_video": str(conditioning_path),
            "output_video": str(output_path),
            "output_json": str(output_json_path),
            "model": {
                "model_id": str(args.model_id),
                "dit_checkpoint": str(args.dit_checkpoint),
                "lora_checkpoint": str(args.lora_checkpoint),
            },
            "inference": {
                "device": str(args.device),
                "height": int(args.height),
                "width": int(args.width),
                "num_frames": int(args.num_frames),
                "target_duration_seconds": float(args.target_duration_seconds),
                "export_fps": float(export_fps),
                "num_inference_steps": int(args.num_inference_steps),
                "guidance_scale": float(args.guidance_scale),
                "seed": int(args.seed) + index,
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
            },
        }

        if output_path.exists() and not args.force:
            fps, duration = _probe_video(output_path)
            if not math.isclose(duration, float(args.target_duration_seconds), abs_tol=1e-6):
                raise RuntimeError(
                    f"existing output has wrong duration for official verified format: "
                    f"{output_path} duration={duration}. Re-run with --force to overwrite it."
                )
            case_payload = {
                **case_payload_base,
                "actual_video": {
                    "export_fps": float(fps),
                    "duration_seconds": float(duration),
                },
                "status": "skipped_existing",
            }
            _write_json(output_json_path, case_payload)
            manifest_entries.append(case_payload)
            print(f"[skip] {output_name}")
            continue

        context_frames = _load_context_video(
            conditioning_path, target_height=int(args.height), target_width=int(args.width)
        )
        generator = torch.Generator(device=str(args.device)).manual_seed(int(args.seed) + index)
        sample = pipe(
            video=context_frames,
            device=torch.device(args.device),
            prompt=prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            height=int(args.height),
            width=int(args.width),
            num_frames=int(args.num_frames),
            num_inference_steps=int(args.num_inference_steps),
            guidance_scale=float(args.guidance_scale),
            do_cfg=False,
            generator=generator,
        )[0]

        export_to_video(sample[0], str(output_path), fps=export_fps, macro_block_size=1)
        actual_fps, actual_duration = _probe_video(output_path)

        if not math.isclose(actual_duration, float(args.target_duration_seconds), abs_tol=1e-6):
            raise RuntimeError(
                f"generated duration mismatch for {output_path}: "
                f"expected {args.target_duration_seconds}, got {actual_duration}"
            )

        case_payload = {
            **case_payload_base,
            "context_frames": len(context_frames),
            "actual_video": {
                "export_fps": float(actual_fps),
                "duration_seconds": float(actual_duration),
            },
            "status": "generated",
        }
        _write_json(output_json_path, case_payload)
        manifest_entries.append(case_payload)
        print(
            f"[done] {index + 1:03d}/{len(rows):03d} {output_name} "
            f"(ctx={len(context_frames)} frames, fps={actual_fps:.6f}, duration={actual_duration:.6f}s)"
        )

    summary = {
        "run_dir": str(run_dir),
        "dataset_root": str(args.dataset_root),
        "descriptions_file": str(args.descriptions_file),
        "num_items": len(rows),
        "height": int(args.height),
        "width": int(args.width),
        "num_frames": int(args.num_frames),
        "target_duration_seconds": float(args.target_duration_seconds),
        "export_fps": export_fps,
        "num_inference_steps": int(args.num_inference_steps),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
        "entries": manifest_entries,
    }
    _write_json(args.output_root / f"{args.run_name}_manifest.json", summary)

    mp4_count = len(list(run_dir.glob("*.mp4")))
    expected_count = len(rows)
    if mp4_count != expected_count:
        raise RuntimeError(f"run folder {run_dir} has {mp4_count} mp4 files, expected {expected_count}")

    print(f"[summary] wrote {mp4_count} videos to {run_dir}")
    print(f"[summary] manifest: {args.output_root / f'{args.run_name}_manifest.json'}")


if __name__ == "__main__":
    main()
