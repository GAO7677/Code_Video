#!/usr/bin/env python3
"""Run Wan2.2 TI2V on Physics-IQ with stronger reproducibility controls."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import sys
sys.path.append('/home/gaoya/code_my_utils')
from tools.pil import grid_pil,text_pil
from tools.seed import seed_everything

DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
BENCHMARK_ROOT = Path("/home/gaoya/Code_Video/physics-IQ-benchmark-main")
DATASET_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark")
MODEL_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physics_IQ")
DESCRIPTIONS_CSV = BENCHMARK_ROOT / "descriptions" / "descriptions.csv"

DEFAULT_NEGATIVE_PROMPT = ""

sys.path.append(str(DIFFSYNTH_ROOT))

from diffsynth import ModelConfig  # noqa: E402
from diffsynth.pipelines.wan_video import WanVideoPipeline  # noqa: E402
from diffsynth.utils.data import save_video  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wan2.2-TI2V-5B on Physics-IQ and evaluate the outputs reproducibly."
    )
    parser.add_argument("--diffsynth_root", type=Path, default=DIFFSYNTH_ROOT)
    parser.add_argument("--benchmark_root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--model_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--descriptions_csv", type=Path, default=DESCRIPTIONS_CSV)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--model_name", default="wan22_ti2v_5b")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_cases(descriptions_csv: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with descriptions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "take-1" not in row["scenario"]:
                continue
            rows.append(row)
    rows.sort(key=lambda row: row["generated_video_name"])
    if limit is not None:
        rows = rows[:limit]
    return rows


def switch_frame_path(dataset_root: Path, generated_video_name: str) -> Path:
    stem = Path(generated_video_name).stem
    file_id, perspective, scenario = stem.split("_", 2)
    frame_name = f"{file_id}_switch-frames_anyFPS_{perspective}_{scenario}.jpg"
    return dataset_root / "switch-frames" / frame_name


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def maybe_git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": args.device,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "DIFFSYNTH_SKIP_DOWNLOAD": os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD"),
        },
        "repos": {
            "diffsynth_root": str(args.diffsynth_root),
            "benchmark_root": str(args.benchmark_root),
            "diffsynth_commit": maybe_git_commit(args.diffsynth_root),
            "benchmark_commit": maybe_git_commit(args.benchmark_root),
        },
        "model_files": {},
    }
    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
    for rel in [
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
    ]:
        path = args.model_root / rel
        info["model_files"][rel] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return info




def build_pipeline(model_root: Path, device: str) -> WanVideoPipeline:
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                path=[
                    str(model_root / "diffusion_pytorch_model-00001-of-00003.safetensors"),
                    str(model_root / "diffusion_pytorch_model-00002-of-00003.safetensors"),
                    str(model_root / "diffusion_pytorch_model-00003-of-00003.safetensors"),
                ]
            ),
            ModelConfig(path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=str(model_root / "Wan2.2_VAE.pth")),
        ],
        tokenizer_config=ModelConfig(path=str(model_root / "google" / "umt5-xxl")),
    )


def required_output_frames(fps: int) -> int:
    return fps * 5




def generate_one_video(
    pipe: WanVideoPipeline,
    image_path: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    cfg_scale: float,
    num_inference_steps: int,
):
    seed_everything(seed)
    with Image.open(image_path) as img:
        input_image = img.convert("RGB").resize((width, height), Image.LANCZOS)
    with torch.no_grad():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            input_image=input_image,
            height=height,
            width=width,
            num_frames=num_frames,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
        )
    keep = required_output_frames(fps)
    if len(video) < keep:
        raise ValueError(
            f"Generated only {len(video)} frames for {image_path.name}, need at least {keep}."
        )
    return video[:keep]




def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_case_metadata(
    *,
    args: argparse.Namespace,
    row: dict[str, str],
    index: int,
    seed: int,
    frame_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    sidecar_path = output_path.with_suffix(".json")
    return {
        "model_name": args.model_name,
        "generated_video_name": row["generated_video_name"],
        "sample_json_name": sidecar_path.name,
        "index_in_sorted_list": index,
        "scenario": row.get("scenario"),
        "perspective": row.get("perspective"),
        "seed": seed,
        "prompt": row["description"],
        "negative_prompt": args.negative_prompt,
        "input_path": str(frame_path),
        "output_path": str(output_path),
        "generation_params": {
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "num_frames": args.num_frames,
        }
    }


def run_generation(args: argparse.Namespace, generated_dir: Path, metadata_dir: Path) -> None:
    cases = load_cases(args.descriptions_csv, args.limit)
    expected_count = 198 if args.limit is None else args.limit
    if len(cases) != expected_count:
        raise ValueError(
            f"Expected {expected_count} take-1 cases, but found {len(cases)} in {args.descriptions_csv}."
        )

    generated_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    per_case_jsonl = metadata_dir / f"{args.model_name}_per_case.jsonl"
    if args.overwrite and per_case_jsonl.exists():
        per_case_jsonl.unlink()


    pipe = build_pipeline(args.model_root, args.device)

    for index, row in enumerate(cases):
        output_name = row["generated_video_name"]
        output_path = generated_dir / output_name
        sidecar_path = output_path.with_suffix(".json")

        frame_path = switch_frame_path(args.dataset_root, output_name)
        assert_exists(frame_path, "Switch frame")


        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_name} | seed={args.seed}")
            case_payload = build_case_metadata(
                args=args,
                row=row,
                index=index,
                seed=args.seed,
                frame_path=frame_path,
                output_path=output_path,
            )
            if not sidecar_path.exists():
                write_json(sidecar_path, case_payload)
            append_jsonl(per_case_jsonl, case_payload)
            print(f"[skip] {output_name} | seed={args.seed}")
            continue

        print(f"[generate] {output_name} | seed={args.seed}")
        video = generate_one_video(
            pipe=pipe,
            image_path=frame_path,
            prompt=row["description"],
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
        )
        save_video(video, str(output_path), fps=args.fps, quality=args.quality)

        case_payload = build_case_metadata(
            args=args,
            row=row,
            index=index,
            seed=args.seed,
            frame_path=frame_path,
            output_path=output_path,
        )
        write_json(sidecar_path, case_payload)
        append_jsonl(per_case_jsonl, case_payload)


def run_evaluation(
    benchmark_root: Path,
    dataset_root: Path,
    descriptions_csv: Path,
    generated_dir: Path,
    eval_output_dir: Path,
) -> None:
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "code/run_physics_iq.py",
        "--input_folders",
        str(generated_dir),
        "--output_folder",
        str(eval_output_dir),
        "--descriptions_file",
        str(descriptions_csv),
    ]
    print("[eval] " + " ".join(cmd))
    subprocess.run(cmd, cwd=benchmark_root, check=True)

def main() -> None:
    args = parse_args()


    assert_exists(args.diffsynth_root, "DiffSynth root")
    assert_exists(args.benchmark_root, "Benchmark root")
    assert_exists(args.dataset_root, "Dataset root")
    assert_exists(args.model_root, "Model root")
    assert_exists(args.descriptions_csv, "Descriptions CSV")
    if args.limit is not None and not args.skip_evaluation:
        raise ValueError("--limit is only for generation smoke tests. Use it with --skip_evaluation.")

    generated_dir = args.output_root / "generated_videos" / args.model_name
    eval_output_dir = args.output_root / "eval_outputs"
    metadata_dir = args.output_root / "metadata" / args.model_name
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_generation:
        run_generation(args, generated_dir, metadata_dir)

    if not args.skip_evaluation:
        run_evaluation(
            benchmark_root=args.benchmark_root,
            dataset_root=args.dataset_root,
            descriptions_csv=args.descriptions_csv,
            generated_dir=generated_dir,
            eval_output_dir=eval_output_dir,
        )

    print(f"Generated videos: {generated_dir}")
    print(f"Evaluation outputs: {eval_output_dir}")
    print(f"Metadata outputs: {metadata_dir}")


if __name__ == "__main__":
    main()






'''


CUDA_VISIBLE_DEVICES=5 python /home/gaoya/Code_Video/Code_data/Code_benchmark/physicalIQ/wan22_ti2v_physics_iq_eval.py \



 

'''