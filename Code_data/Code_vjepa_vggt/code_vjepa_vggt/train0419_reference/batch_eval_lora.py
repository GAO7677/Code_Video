#!/usr/bin/env python3
"""
/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_full.txt
该脚本用于基于多帧上下文做 Wan TI2V 推理；输入为 /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B、
可选的 LoRA 权重、context_path 和文本提示词，输出为 output_root 下的生成视频与同名 json。
支持单 GPU 和多 GPU（通过 CUDA_VISIBLE_DEVICES 控制可见 GPU）两种模式；在多 GPU 模式下会自
动将输入样本分 shard 处理，并在 runtime_root 中生成 per-case JSONL、run manifest 和
summary.json 以供后续分析。
"""
import argparse
import ast
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image


DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
MODEL_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_META_LIST_PATH = TRAIN0419_ROOT / "benchmark_meta_json_paths_full.txt"
DEFAULT_INPUT_JSON_LIST_PATH = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_100.txt")
DATASET_MARKERS = {
    "vLAR-PhysInOne": "vLAR-PhysInOne",
    "physics-iq-benchmark": "physics-iq-benchmark",
    "mvp-lab-OpenVidHD-0.4M-720p-48fps": "mvp-lab-OpenVidHD-0.4M-720p-48fps",
}

if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

from diffsynth import ModelConfig  # noqa: E402
from diffsynth.utils.data import VideoData, save_video  # noqa: E402
from context_wan import ContextAwareWanVideoPipeline  # noqa: E402

sys.path.append("/home/gaoya/code_my_utils")
from tools.seed import seed_everything  # noqa: E402


DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_SINGLE_CASE_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp")
DEFAULT_SINGLE_CASE_LORA_PATH = Path(
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
)
DEFAULT_SINGLE_CASE_MODEL_NAME = "step-010000"
DEFAULT_SINGLE_CASE_DATASET_NAME = "single_case"
DEFAULT_SINGLE_CASE_SAMPLE_ID = "single_case"
DEFAULT_SINGLE_CASE_HEIGHT = 512
DEFAULT_SINGLE_CASE_WIDTH = 896
DEFAULT_SINGLE_CASE_NUM_FRAMES = 24
DEFAULT_SINGLE_CASE_CONTEXT_FRAMES = 8
DEFAULT_SINGLE_CASE_FPS = 30
DEFAULT_SINGLE_CASE_NUM_INFERENCE_STEPS = 40
DEFAULT_SINGLE_CASE_CFG_SCALE = 5.0
DEFAULT_SINGLE_CASE_SEED = 42
DEFAULT_SINGLE_CASE_CONDITIONING_MODE = "context_aware"
WAN_SPATIAL_DIVISIBILITY = 32
STEP_TAG_PATTERN = re.compile(r"step-(\d+)")
OPENVID_LORA_MARKER = "openvid"
LORA_0613_MARKER = "0613lora"
PATH_FIELD_ORDER = [
    "sample_dir",
    "context_video_path",
    "future_gt_video_path",
    "full_video_path",
    "first_frame_path",
    "meta_json_path",
]
INPUT_PATH_LIST_KEYS = [
    "input_frame_paths",
    "context_frame_paths",
    "frame_paths",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small meta.json-driven benchmark for the context-aware Wan LoRA checkpoint."
    )
    parser.add_argument("--wan_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_SINGLE_CASE_OUTPUT_ROOT)
    parser.add_argument("--runtime_root", type=Path, default=None)
    parser.add_argument("--lora_path", type=Path, default=DEFAULT_SINGLE_CASE_LORA_PATH)
    parser.add_argument("--meta_list_path", type=Path, default=None)
    parser.add_argument("--input_json_list_path", type=Path, default=None)
    parser.add_argument("--meta_json_path", type=Path, default=None)
    parser.add_argument("--context_path", type=Path, default=None)
    parser.add_argument("--output_video_path", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--sample_id", type=str, default=DEFAULT_SINGLE_CASE_SAMPLE_ID)
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_SINGLE_CASE_DATASET_NAME)
    parser.add_argument("--future_gt_path", type=Path, default=None)
    parser.add_argument("--full_video_path", type=Path, default=None)
    parser.add_argument("--first_frame_path", type=Path, default=None)
    parser.add_argument("--context_resize_mode", choices=["auto", "crop", "pad"], default="auto")
    parser.add_argument(
        "--conditioning_mode",
        choices=["context_aware", "input_image_only"],
        default=DEFAULT_SINGLE_CASE_CONDITIONING_MODE,
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=DEFAULT_SINGLE_CASE_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_SINGLE_CASE_WIDTH)
    parser.add_argument("--fps", type=int, default=DEFAULT_SINGLE_CASE_FPS)
    parser.add_argument("--num_frames", type=int, default=DEFAULT_SINGLE_CASE_NUM_FRAMES)
    parser.add_argument("--context_frames", type=int, default=DEFAULT_SINGLE_CASE_CONTEXT_FRAMES)

    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_SINGLE_CASE_NUM_INFERENCE_STEPS)
    parser.add_argument("--cfg_scale", type=float, default=DEFAULT_SINGLE_CASE_CFG_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SINGLE_CASE_SEED)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--model_name", default=DEFAULT_SINGLE_CASE_MODEL_NAME)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_metadata", action="store_true")

    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def round_up_to_multiple(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def align_generation_size(height: int, width: int) -> tuple[int, int]:
    aligned_height = round_up_to_multiple(height, WAN_SPATIAL_DIVISIBILITY)
    aligned_width = round_up_to_multiple(width, WAN_SPATIAL_DIVISIBILITY)
    return aligned_height, aligned_width


def align_generation_num_frames(num_frames: int) -> int:
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    remainder = (num_frames - 1) % 4
    if remainder == 0:
        return num_frames
    return num_frames + (4 - remainder)


def validate_args(args: argparse.Namespace) -> None:
    if args.context_frames < 0:
        raise ValueError(f"context_frames must be >= 0, got {args.context_frames}")
    if args.context_frames >= args.num_frames:
        raise ValueError(
            f"context_frames must be smaller than num_frames, got {args.context_frames} >= {args.num_frames}."
        )
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height and width must be divisible by 16, got {(args.height, args.width)}.")
    if args.num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {args.num_frames}.")
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"Invalid shard setting: shard_id={args.shard_id}, num_shards={args.num_shards}")
    has_list_mode = args.meta_list_path is not None
    has_input_json_list_mode = args.input_json_list_path is not None
    has_meta_json_mode = args.meta_json_path is not None
    has_single_mode = args.context_path is not None or args.output_video_path is not None or args.prompt is not None
    mode_count = sum(bool(flag) for flag in (has_list_mode, has_input_json_list_mode, has_meta_json_mode, has_single_mode))
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one input mode: "
            "--meta_list_path, --input_json_list_path, --meta_json_path, or single-case "
            "(--context_path/--output_video_path/--prompt)."
        )
    if (has_list_mode or has_input_json_list_mode) and args.output_root is None:
        raise ValueError("--output_root is required when using --meta_list_path or --input_json_list_path.")
    if has_meta_json_mode and args.output_root is None and args.output_video_path is None:
        raise ValueError("Provide --output_root or --output_video_path when using --meta_json_path.")
    if has_single_mode:
        if args.context_path is None or not str(args.prompt or "").strip():
            raise ValueError("Single-case mode requires --context_path and non-empty --prompt.")
    if args.no_metadata and not has_single_mode:
        raise ValueError("--no_metadata is only supported in single-case mode.")
    if args.no_metadata and args.output_video_path is None:
        raise ValueError("--no_metadata requires --output_video_path.")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
    lora_exists = bool(args.lora_path and args.lora_path.exists())
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
        "multi_gpu": args.multi_gpu,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "output_root": str(args.output_root),
        "runtime_root": str(args.runtime_root),
        "context_frames": args.context_frames,
        "meta_list_path": str(args.meta_list_path),
        "repos": {
            "train0419_root": str(TRAIN0419_ROOT),
            "train0419_commit": maybe_git_commit(TRAIN0419_ROOT),
        },
        "lora": {
            "path": str(args.lora_path) if args.lora_path is not None else None,
            "exists": lora_exists,
            "sha256": sha256_file(args.lora_path) if lora_exists else None,
        },
    }
    for rel in [
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
    ]:
        path = args.wan_root / rel
        info.setdefault("model_files", {})[rel] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return info


def find_tokenizer_path(wan_root: Path) -> Path:
    candidates = [
        wan_root / "google" / "umt5-xxl",
        wan_root / "google",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Tokenizer directory not found. Checked: {', '.join(str(path) for path in candidates)}"
    )


def build_model_configs(wan_root: Path) -> list[ModelConfig]:
    dit_shards = [
        wan_root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00003-of-00003.safetensors",
    ]
    t5_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = wan_root / "Wan2.2_VAE.pth"
    for path in dit_shards + [t5_path, vae_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")
    return [
        ModelConfig(path=[str(path) for path in dit_shards]),
        ModelConfig(path=str(t5_path)),
        ModelConfig(path=str(vae_path)),
    ]


def build_pipeline(wan_root: Path, device: str, lora_path: Path | None) -> ContextAwareWanVideoPipeline:
    pipe = ContextAwareWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=build_model_configs(wan_root),
        tokenizer_config=ModelConfig(path=str(find_tokenizer_path(wan_root))),
    )
    if lora_path is not None:
        pipe.load_lora(pipe.dit, str(lora_path), alpha=1.0)
    return pipe


def pad_and_resize_frame(frame: Image.Image, height: int, width: int) -> Image.Image:
    src_width, src_height = frame.size
    scale = min(width / src_width, height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))
    resized = frame.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def resolve_context_resize_mode(dataset_name: str) -> str:
    if str(dataset_name).strip().lower() == "movi-d":
        return "pad"
    return "crop"


def load_context_frames(
    context_path: Path,
    context_frames: int,
    height: int,
    width: int,
    resize_mode: str = "crop",
):
    if context_path.is_file():
        data = VideoData(
            video_file=str(context_path),
            height=height if resize_mode == "crop" else None,
            width=width if resize_mode == "crop" else None,
        )
    elif context_path.is_dir():
        data = VideoData(
            image_folder=str(context_path),
            height=height if resize_mode == "crop" else None,
            width=width if resize_mode == "crop" else None,
        )
    else:
        raise FileNotFoundError(f"context_path not found: {context_path}")

    available = len(data)
    if available < 1:
        raise ValueError(f"context source is empty: {context_path}")
    keep = min(context_frames, available)
    frames = [data[index] for index in range(keep)]
    if resize_mode == "pad":
        return [pad_and_resize_frame(frame, height=height, width=width) for frame in frames]
    return frames


def load_input_image(
    *,
    first_frame_path: Path | None,
    context_path: Path,
    height: int,
    width: int,
    resize_mode: str,
) -> Image.Image:
    frames = load_context_frames(
        context_path=context_path,
        context_frames=1,
        height=height,
        width=width,
        resize_mode=resize_mode,
    )
    if not frames:
        raise ValueError(f"failed to load first frame from context source: {context_path}")
    return frames[0]


def sanitize_filename(text: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("._")
    return safe or "sample"


def normalize_ckpt_method_name(name: str) -> str:
    normalized = re.sub(r"^[A-Za-z]+\d+_", "", name, count=1)
    return normalized or name


def build_method_name_from_checkpoint_path(checkpoint_path: Path | None) -> str | None:
    if checkpoint_path is None:
        return None
    candidate_path = checkpoint_path.expanduser()
    if candidate_path.is_file() or candidate_path.suffix:
        step_dir = candidate_path.parent
        if not step_dir.name.startswith("step-"):
            return None
        checkpoint_parent = step_dir.parent
        step_name = step_dir.name
    else:
        step_name = candidate_path.name
        checkpoint_parent = candidate_path.parent
    if not step_name:
        return None
    if checkpoint_parent.name == "checkpoints" and checkpoint_parent.parent.name:
        method_root = normalize_ckpt_method_name(checkpoint_parent.parent.name)
        return f"{method_root}_{step_name}"
    if checkpoint_parent.name:
        method_root = normalize_ckpt_method_name(checkpoint_parent.name)
        return f"{method_root}_{step_name}"
    return None


def build_method_name(lora_path: Path | None) -> str:
    return build_method_name_from_checkpoint_path(lora_path) or "unknown_method"


def build_default_output_name(context_path: Path, prompt: str) -> str:
    context_stem = sanitize_filename(context_path.stem)
    prompt_slug = sanitize_filename(prompt.lower())
    if len(prompt_slug) > 48:
        prompt_slug = prompt_slug[:48].rstrip("._-")
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{context_stem}__{prompt_slug}__{prompt_hash}.mp4"


def build_default_sample_id(context_path: Path, prompt: str) -> str:
    context_stem = sanitize_filename(context_path.stem)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{context_stem}__{prompt_hash}"


def parse_step_tag(model_name: str) -> int | None:
    match = STEP_TAG_PATTERN.fullmatch(model_name)
    if match is None:
        return None
    return int(match.group(1))


def resolve_input_path(
    *,
    source_paths: dict[str, Any],
    context_path: str | None,
    conditioning_mode: str | None,
) -> str | list[str] | None:
    roles = build_input_roles(
        source_paths=source_paths,
        context_path=context_path,
        conditioning_mode=conditioning_mode,
    )
    if roles:
        values = [item["path"] for item in roles if isinstance(item.get("path"), str) and item["path"]]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return values

    explicit_input = source_paths.get("input_path")
    if isinstance(explicit_input, str) and explicit_input:
        return explicit_input
    if isinstance(explicit_input, list):
        values = [item for item in explicit_input if isinstance(item, str) and item]
        if values:
            return values

    for key in INPUT_PATH_LIST_KEYS:
        value = source_paths.get(key)
        if not isinstance(value, list):
            continue
        values = [item for item in value if isinstance(item, str) and item]
        if values:
            return values

    if conditioning_mode in {"input_image_only", "ti2v_firstframe"}:
        first_frame_path = source_paths.get("first_frame_path")
        if isinstance(first_frame_path, str) and first_frame_path:
            return first_frame_path

    if context_path:
        return context_path

    context_video_path = source_paths.get("context_video_path")
    if isinstance(context_video_path, str) and context_video_path:
        return context_video_path
    return None


def build_input_roles(
    *,
    source_paths: dict[str, Any],
    context_path: str | None,
    conditioning_mode: str | None,
) -> list[dict[str, Any]]:
    roles: list[dict[str, Any]] = []
    first_frame_path = source_paths.get("first_frame_path")
    context_video_path = context_path or source_paths.get("context_video_path")

    if conditioning_mode in {"input_image_only", "ti2v_firstframe"}:
        if isinstance(first_frame_path, str) and first_frame_path:
            roles.append({"role": "input_image", "path": first_frame_path})
        elif isinstance(context_video_path, str) and context_video_path:
            roles.append(
                {
                    "role": "input_image",
                    "path": context_video_path,
                    "note": "first frame extracted from context video at runtime",
                }
            )
        return roles

    if conditioning_mode == "context_aware":
        if isinstance(first_frame_path, str) and first_frame_path:
            roles.append({"role": "input_image", "path": first_frame_path})
        elif isinstance(context_video_path, str) and context_video_path:
            roles.append(
                {
                    "role": "input_image",
                    "path": context_video_path,
                    "note": "first frame extracted from context video at runtime",
                }
            )
        if isinstance(context_video_path, str) and context_video_path:
            roles.append({"role": "context_video", "path": context_video_path})
        return roles

    explicit_input = source_paths.get("input_path")
    if isinstance(explicit_input, str) and explicit_input:
        roles.append({"role": "input", "path": explicit_input})
    elif isinstance(explicit_input, list):
        for idx, item in enumerate(explicit_input, start=1):
            if isinstance(item, str) and item:
                roles.append({"role": f"input_{idx}", "path": item})
    return roles


def build_model_input_summary(
    *,
    source_paths: dict[str, Any],
    context_path: str | None,
    conditioning_mode: str | None,
    used_context_frames: int,
) -> dict[str, Any]:
    source_conditions = build_input_roles(
        source_paths=source_paths,
        context_path=context_path,
        conditioning_mode=conditioning_mode,
    )
    pipeline_kwargs: list[str] = []
    if conditioning_mode in {"input_image_only", "ti2v_firstframe"}:
        pipeline_kwargs.append("input_image")
    elif conditioning_mode == "context_aware":
        pipeline_kwargs.extend(["input_image", "context_video"])

    payload: dict[str, Any] = {
        "conditioning_mode": conditioning_mode,
        "pipeline_kwargs": pipeline_kwargs,
        "source_conditions": source_conditions,
    }
    if conditioning_mode == "context_aware":
        payload["used_context_frames"] = used_context_frames
        payload["notes"] = [
            "input_image is the first context frame passed separately",
            "context_video is the resized context frame sequence",
        ]
    elif conditioning_mode in {"input_image_only", "ti2v_firstframe"}:
        payload["notes"] = ["input_image is the first frame condition passed to the model"]
    return payload


def build_paths_payload(
    *,
    source_paths: dict[str, Any],
    context_path: str | None,
    output_path: Path,
    sidecar_path: Path,
    conditioning_mode: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in PATH_FIELD_ORDER:
        value = source_paths.get(key)
        if isinstance(value, str) and value:
            payload[key] = value
    if "context_video_path" not in payload and context_path:
        payload["context_video_path"] = context_path
    input_roles = build_input_roles(
        source_paths=source_paths,
        context_path=context_path,
        conditioning_mode=conditioning_mode,
    )
    input_path = resolve_input_path(
        source_paths=source_paths,
        context_path=context_path,
        conditioning_mode=conditioning_mode,
    )
    if input_path is not None:
        payload["input_path"] = input_path
    if input_roles:
        payload["input_roles"] = input_roles
    payload["output_video_path"] = str(output_path)
    payload["output_json_path"] = str(sidecar_path)
    return payload


def get_output_video_path(entry: dict[str, Any]) -> str | None:
    paths = entry.get("paths")
    if isinstance(paths, dict):
        output_path = paths.get("output_video_path")
        if isinstance(output_path, str) and output_path:
            return output_path
        legacy_output_path = paths.get("output_path")
        if isinstance(legacy_output_path, str) and legacy_output_path:
            return legacy_output_path
    output_path = entry.get("output_path")
    if isinstance(output_path, str) and output_path:
        return output_path
    return None


def infer_dataset_name(meta_path: Path) -> str:
    meta_path_str = str(meta_path)
    for marker, dataset_name in DATASET_MARKERS.items():
        if f"/{marker}/" in meta_path_str:
            return dataset_name
    if "mytest" in meta_path.parts:
        mytest_index = meta_path.parts.index("mytest")
        if mytest_index >= 1:
            return meta_path.parts[mytest_index - 1]
    return meta_path.parent.parent.name if meta_path.parent.name == "mytest" else meta_path.parent.name


def normalize_resize_mode(dataset_name: str, requested_mode: str) -> str:
    if requested_mode != "auto":
        return requested_mode
    return resolve_context_resize_mode(dataset_name)


def load_meta_paths(meta_list_path: Path) -> list[Path]:
    meta_paths: list[Path] = []
    seen: set[str] = set()
    missing_paths: list[str] = []
    for raw_line in meta_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = (meta_list_path.parent / candidate).resolve()
        normalized = str(candidate)
        if normalized in seen:
            continue
        if not candidate.is_file():
            missing_paths.append(str(candidate))
            continue
        seen.add(normalized)
        meta_paths.append(candidate)
    if not meta_paths:
        raise ValueError(f"No meta.json paths found in meta_list_path: {meta_list_path}")
    if missing_paths:
        preview = ", ".join(missing_paths[:3])
        suffix = "" if len(missing_paths) <= 3 else f" ... (+{len(missing_paths) - 3} more)"
        warnings.warn(
            f"Skipped {len(missing_paths)} missing meta.json paths from {meta_list_path}: {preview}{suffix}",
            stacklevel=2,
        )
    return meta_paths


def normalize_meta_paths(paths: dict[str, Any], meta_path: Path) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in paths.items():
        if isinstance(value, str):
            if not value.strip():
                continue
            path = Path(value)
            if not path.is_absolute():
                path = (meta_path.parent / path).resolve()
            normalized[key] = str(path)
            continue
        if isinstance(value, list):
            resolved_values: list[str] = []
            for item in value:
                if not isinstance(item, str) or not item.strip():
                    continue
                path = Path(item)
                if not path.is_absolute():
                    path = (meta_path.parent / path).resolve()
                resolved_values.append(str(path))
            if resolved_values:
                normalized[key] = resolved_values
    normalized["meta_json_path"] = str(meta_path)
    return normalized


def collect_cases(meta_paths: list[Path], limit: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for meta_path in meta_paths:
        dataset_name = infer_dataset_name(meta_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        paths = meta.get("paths", {})
        normalized_paths = normalize_meta_paths(paths, meta_path) if isinstance(paths, dict) else {
            "meta_json_path": str(meta_path)
        }
        context_path = normalized_paths.get("context_video_path")
        future_gt_path = normalized_paths.get("future_gt_video_path")
        full_video_path = normalized_paths.get("full_video_path")
        if not context_path:
            continue
        sample_id = str(meta.get("sample_id") or meta_path.parent.name)
        caption = str(meta.get("caption") or meta.get("description") or "")
        output_name = f"{sanitize_filename(sample_id)}.mp4"
        cases.append(
            {
                "dataset": dataset_name,
                "sample_id": sample_id,
                "caption": caption,
                "context_path": str(Path(context_path)),
                "future_gt_path": str(Path(future_gt_path)) if future_gt_path else None,
                "full_video_path": str(Path(full_video_path)) if full_video_path else None,
                "meta_path": str(meta_path),
                "source_paths": normalized_paths,
                "output_name": output_name,
                "context_resize_mode": resolve_context_resize_mode(dataset_name),
                "scenario": meta.get("scenario"),
                "scenario_slug": meta.get("scenario_slug"),
                "raw_meta": meta,
            }
        )
    if limit is not None:
        cases = cases[:limit]
    return cases


def load_input_json_paths(input_json_list_path: Path) -> list[Path]:
    json_paths: list[Path] = []
    seen: set[str] = set()
    for raw_line in input_json_list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = (input_json_list_path.parent / candidate).resolve()
        normalized = str(candidate)
        if normalized in seen:
            continue
        if not candidate.is_file():
            raise FileNotFoundError(f"input json not found from input_json_list_path: {candidate}")
        seen.add(normalized)
        json_paths.append(candidate)
    if not json_paths:
        raise ValueError(f"No input json paths found in input_json_list_path: {input_json_list_path}")
    return json_paths


def collect_cases_from_input_json_list(input_json_paths: list[Path], limit: int | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for input_json_path in input_json_paths:
        payload = json.loads(input_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"input json must be an object: {input_json_path}")
        input_video = payload.get("input_video")
        input_caption = payload.get("input_caption")
        source_video = payload.get("source_video")
        if not isinstance(input_video, str) or not input_video.strip():
            continue
        if not isinstance(input_caption, str) or not input_caption.strip():
            continue

        normalized_paths: dict[str, Any] = {
            "meta_json_path": str(input_json_path),
            "context_video_path": str(Path(input_video).expanduser().resolve()),
        }
        if isinstance(source_video, str) and source_video.strip():
            normalized_paths["full_video_path"] = str(Path(source_video).expanduser().resolve())

        sample_id = input_json_path.stem
        dataset_name = "input_json_list"
        cases.append(
            {
                "dataset": dataset_name,
                "sample_id": sample_id,
                "caption": input_caption.strip(),
                "context_path": normalized_paths["context_video_path"],
                "future_gt_path": None,
                "full_video_path": normalized_paths.get("full_video_path"),
                "meta_path": str(input_json_path),
                "source_paths": normalized_paths,
                "output_name": f"{sample_id}.mp4",
                "context_resize_mode": "crop",
                "scenario": None,
                "scenario_slug": None,
                "raw_meta": payload,
                "simple_input_json_mode": True,
            }
        )
    if limit is not None:
        cases = cases[:limit]
    return cases


def build_single_case(args: argparse.Namespace) -> dict[str, Any]:
    dataset_name = str(args.dataset_name).strip() or "single_case"
    context_path = Path(args.context_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_video_path = (
        Path(args.output_video_path).expanduser().resolve()
        if args.output_video_path is not None
        else output_root / build_default_output_name(context_path, str(args.prompt).strip())
    )
    source_paths: dict[str, str] = {
        "context_video_path": str(context_path),
    }
    if args.meta_json_path is not None:
        source_paths["meta_json_path"] = str(Path(args.meta_json_path).expanduser().resolve())
    if args.future_gt_path is not None:
        source_paths["future_gt_video_path"] = str(Path(args.future_gt_path).expanduser().resolve())
    if args.full_video_path is not None:
        source_paths["full_video_path"] = str(Path(args.full_video_path).expanduser().resolve())
    if args.first_frame_path is not None:
        source_paths["first_frame_path"] = str(Path(args.first_frame_path).expanduser().resolve())
    return {
        "dataset": dataset_name,
        "sample_id": str(args.sample_id).strip()
        or build_default_sample_id(context_path, str(args.prompt).strip()),
        "caption": str(args.prompt).strip(),
        "context_path": source_paths["context_video_path"],
        "future_gt_path": source_paths.get("future_gt_video_path"),
        "full_video_path": source_paths.get("full_video_path"),
        "meta_path": source_paths.get("meta_json_path"),
        "source_paths": source_paths,
        "output_name": output_video_path.name,
        "output_path_override": str(output_video_path),
        "context_resize_mode": normalize_resize_mode(dataset_name, args.context_resize_mode),
        "scenario": None,
        "scenario_slug": None,
        "raw_meta": {},
    }


def collect_cases_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.input_json_list_path is not None:
        return collect_cases_from_input_json_list(load_input_json_paths(args.input_json_list_path), args.limit)
    if args.meta_json_path is not None:
        return collect_cases([args.meta_json_path], limit=args.limit)
    if args.context_path is not None or args.output_video_path is not None or args.prompt is not None:
        return [build_single_case(args)]
    return collect_cases(load_meta_paths(args.meta_list_path), args.limit)


def build_simple_result_payload(
    *,
    input_json_path: Path,
    input_video: str,
    input_caption: str,
    output_video: Path,
    method: str,
    seed: int,
    step: int,
    guidance: float,
    ckpt: Path | None,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_json": str(input_json_path),
        "input_video": str(input_video),
        "input_caption": str(input_caption),
        "output_video": str(output_video),
        "method": str(method),
        "seed": int(seed),
        "step": int(step),
        "guidance": float(guidance),
        "ckpt": str(ckpt) if ckpt is not None else None,
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    return payload


def generate_one_video(
    pipe: ContextAwareWanVideoPipeline,
    context_path: Path,
    first_frame_path: Path | None,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    cfg_scale: float,
    num_inference_steps: int,
    context_frames: int,
    output_num_frames: int,
    context_resize_mode: str = "crop",
    conditioning_mode: str = "context_aware",
):
    seed_everything(seed)
    context = None
    input_image = None
    if conditioning_mode == "input_image_only":
        input_image = load_input_image(
            first_frame_path=first_frame_path,
            context_path=context_path,
            height=height,
            width=width,
            resize_mode=context_resize_mode,
        )
    elif context_frames > 0:
        context = load_context_frames(
            context_path=context_path,
            context_frames=context_frames,
            height=height,
            width=width,
            resize_mode=context_resize_mode,
        )
    with torch.no_grad():
        generation_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "seed": seed,
            "cfg_scale": cfg_scale,
            "num_inference_steps": num_inference_steps,
            "tiled": True,
        }
        if input_image is not None:
            generation_kwargs["input_image"] = input_image
        if context is not None:
            generation_kwargs["input_image"] = context[0]
            generation_kwargs["context_video"] = context
        video = pipe(**generation_kwargs)
    keep = min(int(output_num_frames), len(video))
    if len(video) < keep:
        raise ValueError(
            f"Generated only {len(video)} frames for {context_path.name}, need at least {keep}."
        )
    if conditioning_mode == "input_image_only":
        return video[:keep], 1
    return video[:keep], 0 if context is None else len(context)


def build_case_metadata(
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    index: int,
    seed: int,
    output_path: Path,
    used_context_frames: int,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    benchmark_step = parse_step_tag(args.model_name)
    sidecar_path = output_path.with_suffix(".json")
    source_paths = row.get("source_paths", {})
    method_name = build_method_name(args.lora_path)
    sample_id = str(row["sample_id"])
    case_key = sanitize_filename(sample_id)
    clip_name = case_key.split("trimmed-")[-1] if "trimmed-" in case_key else case_key
    input_first_frame = source_paths.get("first_frame_path")
    input_context_video = source_paths.get("context_video_path")
    payload = {
        "group": "D_clean",
        "benchmark": "physics-iq-benchmark",
        "method_name": method_name,
        "case_key": case_key,
        "category": str(row["dataset"]).replace(" ", "_"),
        "clip_name": clip_name,
        "input_prompt": str(row["caption"]),
        "input_image": str(input_first_frame) if isinstance(input_first_frame, str) and input_first_frame else None,
        "input_context_video": str(input_context_video) if isinstance(input_context_video, str) and input_context_video else None,
        "generation_params": {
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "requested_output_frames": args.requested_output_frames,
            "aligned_generation_num_frames": args.num_frames,
            "context_frames": args.context_frames,
            "negative_prompt": args.negative_prompt,
            "conditioning_mode": args.conditioning_mode,
            "used_context_frames": used_context_frames,
        },
        "source_video": str(source_paths.get("full_video_path")) if isinstance(source_paths.get("full_video_path"), str) and source_paths.get("full_video_path") else None,
        "source_meta_json": str(source_paths.get("meta_json_path")) if isinstance(source_paths.get("meta_json_path"), str) and source_paths.get("meta_json_path") else None,
        "output_video": str(output_path),
        "output_json": str(sidecar_path),
        "status": status,
        "seed": seed,
        "runtime": {
            "model_name": args.model_name,
            "benchmark_step": benchmark_step,
            "dataset": row["dataset"],
            "sample_id": sample_id,
            "scenario": row.get("scenario"),
            "weights_path": str(args.lora_path) if args.lora_path is not None else None,
            "index_in_sorted_list": index,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        },
        "paths": build_paths_payload(
            source_paths=source_paths,
            context_path=row["context_path"],
            output_path=output_path,
            sidecar_path=sidecar_path,
            conditioning_mode=args.conditioning_mode,
        ),
        "model_inputs": build_model_input_summary(
            source_paths=source_paths,
            context_path=row["context_path"],
            conditioning_mode=args.conditioning_mode,
            used_context_frames=used_context_frames,
        ),
    }
    if error is not None:
        payload["error"] = error
    return payload


def per_case_jsonl_path(metadata_dir: Path, model_name: str, num_shards: int, shard_id: int) -> Path:
    if num_shards <= 1:
        return metadata_dir / f"{model_name}_per_case.jsonl"
    return metadata_dir / f"{model_name}_per_case_shard{shard_id:02d}of{num_shards:02d}.jsonl"


def write_run_manifest(args: argparse.Namespace, metadata_dir: Path) -> None:
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": collect_environment_snapshot(args),
    }
    suffix = "" if args.num_shards <= 1 else f"_shard{args.shard_id:02d}of{args.num_shards:02d}"
    manifest_path = metadata_dir / f"{args.model_name}_run_manifest{suffix}.json"
    write_json(manifest_path, manifest)


def run_generation(args: argparse.Namespace, generated_dir: Path, metadata_dir: Path) -> None:
    cases = collect_cases_from_args(args)
    generated_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args, metadata_dir)
    method_name = build_method_name(args.lora_path)

    per_case_jsonl = per_case_jsonl_path(metadata_dir, args.model_name, args.num_shards, args.shard_id)
    if args.overwrite and per_case_jsonl.exists():
        per_case_jsonl.unlink()
    existing_entries = load_jsonl(per_case_jsonl) if per_case_jsonl.exists() else []
    entries_by_index: dict[int, dict[str, Any]] = {}
    for entry in existing_entries:
        entries_by_index[entry_sort_index(entry)] = entry

    indexed_cases = list(enumerate(cases))
    shard_cases = [(idx, row) for idx, row in indexed_cases if idx % args.num_shards == args.shard_id]
    print(
        f"[worker] shard_id={args.shard_id}/{args.num_shards}, "
        f"num_cases={len(shard_cases)}, device={args.device}, seed={args.seed}, context_frames={args.context_frames}"
    )

    pipe = build_pipeline(args.wan_root, args.device, args.lora_path)

    for index, row in shard_cases:
        output_override = row.get("output_path_override")
        output_path = (
            Path(output_override)
            if isinstance(output_override, str) and output_override
            else generated_dir / row["output_name"]
        )
        sidecar_path = output_path.with_suffix(".json")
        context_path = Path(row["context_path"])
        assert_exists(context_path, "Context video")

        if output_path.exists() and not args.overwrite:
            print(f"[skip][shard {args.shard_id}] {row['output_name']} | seed={args.seed}")
            if row.get("simple_input_json_mode"):
                case_payload = build_simple_result_payload(
                    input_json_path=Path(str(row["meta_path"])).expanduser().resolve(),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=output_path,
                    method=method_name,
                    seed=args.seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=args.lora_path,
                    status="skipped_existing",
                )
            else:
                case_payload = build_case_metadata(
                    args=args,
                    row=row,
                    index=index,
                    seed=args.seed,
                    output_path=output_path,
                    used_context_frames=1 if args.conditioning_mode == "input_image_only" else args.context_frames,
                    status="skipped_existing",
                )
            write_json(sidecar_path, case_payload)
            entries_by_index[index] = case_payload
            continue

        print(
            f"[generate][shard {args.shard_id}] {row['dataset']}::{row['sample_id']} "
            f"-> {row['output_name']} | seed={args.seed}"
        )
        try:
            first_frame_path = None
            raw_first_frame_path = row.get("source_paths", {}).get("first_frame_path")
            if isinstance(raw_first_frame_path, str) and raw_first_frame_path:
                first_frame_path = Path(raw_first_frame_path)
            video, used_context_frames = generate_one_video(
                pipe=pipe,
                context_path=context_path,
                first_frame_path=first_frame_path,
                prompt=row["caption"],
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                fps=args.fps,
                cfg_scale=args.cfg_scale,
                num_inference_steps=args.num_inference_steps,
                context_frames=args.context_frames,
                output_num_frames=args.requested_output_frames,
                context_resize_mode=row.get("context_resize_mode", "crop"),
                conditioning_mode=args.conditioning_mode,
            )
            save_video(video, str(output_path), fps=args.fps, quality=args.quality)
            if row.get("simple_input_json_mode"):
                case_payload = build_simple_result_payload(
                    input_json_path=Path(str(row["meta_path"])).expanduser().resolve(),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=output_path,
                    method=method_name,
                    seed=args.seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=args.lora_path,
                    status="generated",
                )
            else:
                case_payload = build_case_metadata(
                    args=args,
                    row=row,
                    index=index,
                    seed=args.seed,
                    output_path=output_path,
                    used_context_frames=used_context_frames,
                    status="generated",
                )
        except Exception as exc:
            print(f"[error][shard {args.shard_id}] {row['output_name']} | {exc}")
            if row.get("simple_input_json_mode"):
                case_payload = build_simple_result_payload(
                    input_json_path=Path(str(row["meta_path"])).expanduser().resolve(),
                    input_video=str(row["context_path"]),
                    input_caption=str(row["caption"]),
                    output_video=output_path,
                    method=method_name,
                    seed=args.seed,
                    step=args.num_inference_steps,
                    guidance=args.cfg_scale,
                    ckpt=args.lora_path,
                    status="failed",
                    error=repr(exc),
                )
            else:
                case_payload = build_case_metadata(
                    args=args,
                    row=row,
                    index=index,
                    seed=args.seed,
                    output_path=output_path,
                    used_context_frames=0,
                    status="failed",
                    error=repr(exc),
                )
        write_json(sidecar_path, case_payload)
        entries_by_index[index] = case_payload
    write_jsonl(
        per_case_jsonl,
        [entries_by_index[idx] for idx in sorted(entries_by_index)],
    )


def merge_shard_jsonl_files(metadata_dir: Path, model_name: str, num_shards: int) -> Path | None:
    if num_shards <= 1:
        path = metadata_dir / f"{model_name}_per_case.jsonl"
        return path if path.exists() else None

    merged_path = metadata_dir / f"{model_name}_per_case.jsonl"
    shard_paths = [
        metadata_dir / f"{model_name}_per_case_shard{shard_id:02d}of{num_shards:02d}.jsonl"
        for shard_id in range(num_shards)
    ]
    merged_entries: list[dict[str, Any]] = []
    for path in shard_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    merged_entries.append(json.loads(line))
    if not merged_entries:
        return None
    merged_entries.sort(key=lambda item: item.get("index_in_sorted_list", 10**18))
    with merged_path.open("w", encoding="utf-8") as handle:
        for entry in merged_entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return merged_path


def detect_visible_gpu_tokens() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        tokens = [token.strip() for token in env.split(",") if token.strip()]
        if tokens:
            return tokens
    if not torch.cuda.is_available():
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def build_worker_command(script_path: Path, base_argv: list[str], num_shards: int, shard_id: int) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        *base_argv,
        "--worker",
        "--num_shards",
        str(num_shards),
        "--shard_id",
        str(shard_id),
        "--device",
        "cuda:0",
    ]


def launch_multi_gpu_workers(args: argparse.Namespace, generated_dir: Path, metadata_dir: Path) -> int:
    visible_gpu_tokens = detect_visible_gpu_tokens()
    if len(visible_gpu_tokens) <= 1:
        run_generation(args, generated_dir, metadata_dir)
        return 1

    num_shards = len(visible_gpu_tokens)
    script_path = Path(__file__).resolve()
    base_argv = sys.argv[1:]
    print(f"[multi_gpu] Launching {num_shards} workers on visible GPUs: {visible_gpu_tokens}")
    procs: list[tuple[int, str, subprocess.Popen[str]]] = []
    for shard_id, gpu_token in enumerate(visible_gpu_tokens):
        cmd = build_worker_command(script_path, base_argv, num_shards, shard_id)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_token
        print(f"[multi_gpu] shard {shard_id}: CUDA_VISIBLE_DEVICES={gpu_token}")
        proc = subprocess.Popen(cmd, env=env, text=True)
        procs.append((shard_id, gpu_token, proc))

    failures = []
    for shard_id, gpu_token, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failures.append((shard_id, gpu_token, ret))
    if failures:
        raise RuntimeError(f"One or more worker processes failed: {failures}")

    merged_jsonl = merge_shard_jsonl_files(metadata_dir, args.model_name, num_shards)
    if merged_jsonl is not None:
        print(f"[multi_gpu] Merged shard metadata into: {merged_jsonl}")
    return num_shards


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def entry_sort_index(entry: dict[str, Any]) -> int:
    runtime = entry.get("runtime")
    if isinstance(runtime, dict):
        index = runtime.get("index_in_sorted_list")
        if isinstance(index, int):
            return index
    index = entry.get("index_in_sorted_list")
    if isinstance(index, int):
        return index
    return 10**18


def infer_eval_csv_path(output_root: Path, model_name: str) -> Path | None:
    candidates = [
        output_root / "eval_outputs" / model_name / "results" / f"{model_name}.csv",
        output_root / "eval_outputs" / "results" / f"{model_name}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    eval_root = output_root / "eval_outputs"
    if not eval_root.is_dir():
        return None
    for path in sorted(eval_root.rglob(f"{model_name}.csv")):
        if path.is_file() and path.parent.name == "results":
            return path
    return None


def maybe_literal_eval(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    try:
        return ast.literal_eval(text)
    except Exception:
        try:
            return float(text)
        except Exception:
            return text


def is_numeric_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(item, (int, float)) for item in value
    )


def round_metric(value: float) -> float:
    return round(float(value), 4)


def summarize_metric_row(row: dict[str, str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key, raw_value in row.items():
        if key == "scenario" or raw_value is None:
            continue
        parsed = maybe_literal_eval(raw_value)
        if isinstance(parsed, (int, float)):
            metrics[key] = round_metric(parsed)
            continue
        if is_numeric_list(parsed):
            metrics[f"{key}_mean"] = round_metric(sum(parsed) / len(parsed))
    return metrics


def scenario_key_from_entry(entry: dict[str, Any]) -> str | None:
    scenario_slug = entry.get("scenario_slug")
    if isinstance(scenario_slug, str) and scenario_slug.strip():
        slug = scenario_slug.strip()
        return slug if slug.endswith(".mp4") else f"{slug}.mp4"

    scenario = entry.get("scenario")
    if isinstance(scenario, str) and scenario.strip():
        name = Path(scenario).name
        match = re.search(r"(trimmed-.+\.mp4)$", name)
        return match.group(1) if match else name

    sample_id = entry.get("sample_id")
    if isinstance(sample_id, str) and "trimmed-" in sample_id:
        suffix = sample_id[sample_id.index("trimmed-") :]
        return suffix if suffix.endswith(".mp4") else f"{suffix}.mp4"
    return None


def load_eval_metrics(eval_csv_path: Path) -> dict[str, dict[str, float]]:
    metrics_by_scenario: dict[str, dict[str, float]] = {}
    with eval_csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scenario = row.get("scenario")
            if not scenario:
                continue
            metrics_by_scenario[str(scenario)] = summarize_metric_row(row)
    return metrics_by_scenario


def augment_entries_with_eval_metrics(
    entries: list[dict[str, Any]],
    eval_csv_path: Path | None,
) -> tuple[list[dict[str, Any]], int]:
    if eval_csv_path is None or not eval_csv_path.is_file():
        return entries, 0

    metrics_by_scenario = load_eval_metrics(eval_csv_path)
    updated_count = 0
    for entry in entries:
        scenario_key = scenario_key_from_entry(entry)
        if scenario_key is None:
            continue
        metrics = metrics_by_scenario.get(scenario_key)
        if not metrics:
            continue
        evaluation = {
            "eval_csv_path": str(eval_csv_path),
            "scenario_key": scenario_key,
            "metrics": metrics,
        }
        entry["evaluation"] = evaluation
        entry["metrics"] = metrics

        output_path = entry.get("output_path")
        if not isinstance(output_path, str) or not output_path:
            output_path = get_output_video_path(entry)
        if isinstance(output_path, str) and output_path:
            sidecar_path = Path(output_path).with_suffix(".json")
            sidecar_payload = dict(entry)
            if sidecar_path.is_file():
                try:
                    existing_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if isinstance(existing_payload, dict):
                        existing_payload.update(entry)
                        sidecar_payload = existing_payload
                except Exception:
                    sidecar_payload = dict(entry)
            sidecar_payload["evaluation"] = evaluation
            sidecar_payload["metrics"] = metrics
            write_json(sidecar_path, sidecar_payload)
        updated_count += 1
    return entries, updated_count


def build_summary(entries: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(entries)
    generated = sum(1 for item in entries if item.get("status") == "generated")
    failed = sum(1 for item in entries if item.get("status") == "failed")
    skipped = sum(1 for item in entries if item.get("status") == "skipped_existing")
    dataset_counts: dict[str, int] = {}
    dataset_generated: dict[str, int] = {}
    for item in entries:
        dataset = str(item.get("dataset", "unknown"))
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        if item.get("status") in {"generated", "skipped_existing"}:
            dataset_generated[dataset] = dataset_generated.get(dataset, 0) + 1
    summary: dict[str, float | int] = {
        "num_cases": total,
        "num_generated": generated,
        "num_failed": failed,
        "num_skipped_existing": skipped,
        "success_rate": round((generated + skipped) / total, 6) if total else 0.0,
    }
    for dataset, count in sorted(dataset_counts.items()):
        dataset_key = sanitize_filename(dataset).lower()
        summary[f"dataset_cases_{dataset_key}"] = count
        summary[f"dataset_success_rate_{dataset_key}"] = round(
            dataset_generated.get(dataset, 0) / count,
            6,
        )
    return summary


def find_selected_video_paths(generated_dir: Path, entries: list[dict[str, Any]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    seen_datasets: set[str] = set()
    for entry in entries:
        if entry.get("status") not in {"generated", "skipped_existing"}:
            continue
        dataset = str(entry.get("dataset", "unknown"))
        if dataset in seen_datasets:
            continue
        output_path = get_output_video_path(entry)
        if isinstance(output_path, str) and os.path.isfile(output_path):
            selected[sanitize_filename(dataset).lower()] = output_path
            seen_datasets.add(dataset)
    if selected:
        return selected
    for video_path in sorted(generated_dir.rglob("*.mp4"))[:2]:
        selected[video_path.stem] = str(video_path)
    return selected


def main() -> None:
    args = parse_args()
    if args.output_root is None:
        if args.output_video_path is not None:
            args.output_root = Path(args.output_video_path).expanduser().resolve().parent
        elif args.meta_json_path is not None:
            args.output_root = Path.cwd() / "generated_videos"
        else:
            args.output_root = DEFAULT_SINGLE_CASE_OUTPUT_ROOT
    if args.runtime_root is None:
        args.runtime_root = args.output_root
    assert_exists(args.wan_root, "Wan root")
    if args.lora_path is not None:
        assert_exists(args.lora_path, "LoRA checkpoint")
    if args.meta_json_path is not None:
        assert_exists(args.meta_json_path, "Meta json path")
    elif args.input_json_list_path is not None:
        assert_exists(args.input_json_list_path, "Input json list path")
    elif args.meta_list_path is not None:
        assert_exists(args.meta_list_path, "Meta list path")
    if args.context_path is not None:
        assert_exists(args.context_path, "Context path")
    validate_args(args)

    aligned_height, aligned_width = align_generation_size(args.height, args.width)
    if (aligned_height, aligned_width) != (args.height, args.width):
        print(
            "[size_align] Adjusting benchmark generation size from "
            f"{args.height}x{args.width} to {aligned_height}x{aligned_width} "
            f"to satisfy Wan's /{WAN_SPATIAL_DIVISIBILITY} spatial divisibility."
        )
        args.height = aligned_height
        args.width = aligned_width
    args.requested_output_frames = int(args.num_frames)
    aligned_num_frames = align_generation_num_frames(args.num_frames)
    if aligned_num_frames != args.num_frames:
        print(
            "[time_align] Adjusting Wan generation length from "
            f"{args.num_frames} to {aligned_num_frames} to satisfy 4n+1, "
            f"while saving only the first {args.requested_output_frames} frames."
        )
        args.num_frames = aligned_num_frames

    if args.no_metadata:
        pipe = build_pipeline(args.wan_root, args.device, args.lora_path)
        row = build_single_case(args)
        output_path = Path(row["output_path_override"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        context_path = Path(row["context_path"])
        assert_exists(context_path, "Context video")
        first_frame_path = None
        raw_first_frame_path = row.get("source_paths", {}).get("first_frame_path")
        if isinstance(raw_first_frame_path, str) and raw_first_frame_path:
            first_frame_path = Path(raw_first_frame_path)
        video, _ = generate_one_video(
            pipe=pipe,
            context_path=context_path,
            first_frame_path=first_frame_path,
            prompt=row["caption"],
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            context_frames=args.context_frames,
            output_num_frames=args.requested_output_frames,
            context_resize_mode=row.get("context_resize_mode", "crop"),
            conditioning_mode=args.conditioning_mode,
        )
        save_video(video, str(output_path), fps=args.fps, quality=args.quality)
        print(output_path)
        return

    generated_dir = args.output_root
    metadata_dir = args.runtime_root / "metadata" / args.model_name
    summary_json_path = args.runtime_root / "summary.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    effective_num_shards = args.num_shards
    if args.multi_gpu and not args.worker:
        effective_num_shards = launch_multi_gpu_workers(args, generated_dir, metadata_dir)
    else:
        run_generation(args, generated_dir, metadata_dir)

    merged_jsonl = merge_shard_jsonl_files(metadata_dir, args.model_name, effective_num_shards)
    summary_entries_path = merged_jsonl
    if summary_entries_path is None:
        summary_entries_path = per_case_jsonl_path(
            metadata_dir,
            args.model_name,
            effective_num_shards,
            args.shard_id,
        )
    summary_entries = load_jsonl(summary_entries_path)
    eval_csv_path = infer_eval_csv_path(args.output_root, args.model_name)
    summary_entries, num_entries_with_metrics = augment_entries_with_eval_metrics(
        summary_entries,
        eval_csv_path,
    )
    if summary_entries_path is not None and summary_entries:
        write_jsonl(summary_entries_path, summary_entries)
    payload = {
        "model_name": args.model_name,
        "lora_path": str(args.lora_path) if args.lora_path is not None else None,
        "generated_dir": str(generated_dir),
        "metadata_dir": str(metadata_dir),
        "runtime_root": str(args.runtime_root),
        "meta_list_path": str(args.meta_list_path),
        "input_json_list_path": str(args.input_json_list_path),
        "eval_csv": str(eval_csv_path) if eval_csv_path is not None else None,
        "num_entries_with_metrics": num_entries_with_metrics,
        "summary": build_summary(summary_entries),
        "selected_videos": find_selected_video_paths(generated_dir, summary_entries),
    }
    write_json(summary_json_path, payload)
    print(summary_json_path)


if __name__ == "__main__":
    main()
